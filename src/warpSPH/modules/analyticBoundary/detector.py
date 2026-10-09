"""Free-surface detection with analytic walls: the Barecasco detector of `modules/surfaceDetection` with the wall
as a continuum of wall particles (density mu / dx^2) added to its partial sums BEFORE the decisions are taken.

warpSPH's `detectFreeSurfaceBarecasco` decides inside one kernel (cover vector, then the wedge count). With walls the
decision needs sums the fluid kernel does not have: this module runs the two passes as separate launches
(`computeBarecascoCoverWarp`, `computeBarecascoConeCountWarp`, partial sums over the fluid pairs) and adds the wall's
share between them:

    C        = C_fluid - nw (pi H^3 / 3) sum_b cover_b            cover = g0 of the degree-1 cone kernel over the wall (exact edge reduction)
    count    = count_fluid + nw cone_area(c, half angle)          closed-form area of the wall inside the wedge around c = C / |C|
    allcount = n_fluid     + nw cone_area(c, pi)                  (the whole support; used where C vanishes)
    surface  = count < 1/2

The normals are warpSPH's own `computeNormalsLambdaGrad` with the renormalisation matrix of the fluid PLUS the wall
(Mt = Mf + mu sum_b Cov_b): the minimum eigenvalue lMin and the normal -grad(lMin)/|grad(lMin)|, as the boundary-
particle flavour gets from its AllToAll pass (ghosts included). The mask is then dilated as in `detectFreeSurface`.
"""
import math
from dataclasses import replace

import torch

from warpSPHCore import (GradientScheme, OperationDirection, OperationProperties, RenormalizationState, SupportScheme, WarpOperation, scalar_t, warpOperation)
from warpSPHCore.type_config import get_torch_precision

from ..gravity import computeGravity
from ..surfaceDetection.dilation import dilateSurface
from ..surfaceDetection.lambdaGrad import computeNormalsLambdaGrad
from ..surfaceDetection.wp_barecascoCone import computeBarecascoConeCountWarp
from ..surfaceDetection.wp_barecascoCover import computeBarecascoCoverWarp
from ..util import countNeighbors
from .wallTerms import evaluateWall

__all__ = ['detectFreeSurfaceAnalytic', 'sym2LamPinv']

F64 = torch.float64


def sym2LamPinv(M, pinv=True):
    """(min |eigenvalue| [N], pseudo-inverse [N,2,2]) of 2 x 2 matrices that are symmetric to round-off, in closed form (the lower
    triangle defines the matrix; the pseudo-inverse drops eigenvalues below 2 eps max|lambda|, torch's default rtol)."""
    a, b, d = M[:, 0, 0], M[:, 1, 0], M[:, 1, 1]
    mid, disc = 0.5 * (a + d), torch.sqrt((0.5 * (a - d)) ** 2 + b * b)
    l1, l2 = mid + disc, mid - disc
    lam = torch.minimum(l1.abs(), l2.abs())
    if not pinv:
        return lam, None
    ua, ub = torch.stack([b, l1 - a], 1), torch.stack([l1 - d, b], 1)
    u = torch.where((ua * ua).sum(1, keepdim=True) >= (ub * ub).sum(1, keepdim=True), ua, ub)
    nrm = u.norm(dim=1, keepdim=True)
    ex = torch.zeros_like(u)
    ex[:, 0] = 1.0
    u = torch.where(nrm > 1e-300, u / nrm.clamp(min=1e-300), ex)
    w = torch.stack([-u[:, 1], u[:, 0]], 1)
    cut = 2.0 * torch.finfo(M.dtype).eps * torch.maximum(l1.abs(), l2.abs())
    c1 = torch.where(l1.abs() > cut, 1.0 / torch.where(l1 == 0, torch.ones_like(l1), l1), torch.zeros_like(l1))
    c2 = torch.where(l2.abs() > cut, 1.0 / torch.where(l2 == 0, torch.ones_like(l2), l2), torch.zeros_like(l2))
    L = c1[:, None, None] * u[:, :, None] * u[:, None, :] + c2[:, None, None] * w[:, :, None] * w[:, None, :]
    return lam, L


def _spacing(config, state, schemeConfig):
    dx = getattr(config, 'dx', None)
    if dx is None:
        dx = (state.masses / schemeConfig.fluid.restDensity).pow(0.5).mean()
    return float(dx)


def detectFreeSurfaceAnalytic(currentState, config, schemeConfig, surfaceConfig, adjacency, returnNormals=True, wall=None):
    """`detectFreeSurface` for a scene with a boundary provider: (raw mask, dilated mask, normals, renormalisation state, lMin),
    or without the normals. `wall`: the aggregates at these positions when the caller has them."""
    provider = schemeConfig.boundaryProvider
    if wall is None:
        wall = evaluateWall(provider, currentState, config, schemeConfig, computeGravity(currentState, config, schemeConfig, adjacency))
    tp = get_torch_precision()
    domain = config.domain
    op = OperationProperties(kernel=config.kernel, supportMode=SupportScheme.SuperSymmetric, operationMode=OperationDirection.AllToAll, gradientMode=GradientScheme.Naive)
    H, wm = wall.support, wall.wm
    nw = wm / _spacing(config, currentState, schemeConfig) ** 2
    near = wall.near

    C = computeBarecascoCoverWarp(currentState, op, domain, adjacency=adjacency).to(F64)
    nAll = countNeighbors(currentState, config, schemeConfig, adjacency).to(F64) - 1.0                      # the list's neighbours within the support, the particle itself excluded
    C = C - nw * (math.pi * H ** 3 / 3) * wall.agg.out['cover'].sum(0) * near[:, None]
    norm = C.norm(dim=1)
    c = C / norm.clamp(min=1e-300)[:, None]
    half = surfaceConfig.barecascoThreshold / 2
    count = computeBarecascoConeCountWarp(currentState, op, domain, coverAxes=c.to(tp).contiguous(), halfAngle=scalar_t(half), adjacency=adjacency).to(F64)
    areas = wall.agg.cone_area(c, half) * near                                                              # [2, N]: the wedge, the whole disk (the wall continuum)
    count, allcount = count + nw * areas[0], nAll + nw * areas[1]
    raw = torch.where(norm > 1e-12, count, allcount) < 0.5

    fs = raw.clone().to(dtype=currentState.positions.dtype)
    for _ in range(surfaceConfig.expansionIterations):
        fs = dilateSurface(currentState, fs, config, schemeConfig, surfaceConfig, adjacency, overrideIterations=1)

    Mf = warpOperation(currentState, replace(op, operation=WarpOperation.Covariance), domain, adjacency=adjacency, covarianceReturnNumNeighbors=True)[0].to(F64)
    Mt = Mf + wm * wall.agg.out['Cov'].sum(0)
    lam, L = sym2LamPinv(Mt)
    renorm = RenormalizationState(renormalizationMatrices=L.to(tp))
    lMin = lam.to(currentState.positions.dtype)
    if not returnNormals:
        return raw, fs, renorm, lMin
    normals = computeNormalsLambdaGrad(currentState, config, schemeConfig, surfaceConfig, adjacency, lambdas=lam.to(tp), renormalizationState=renorm)
    return raw, fs, normals, renorm, lMin
