"""XSPH velocity smoothing and the boundary friction: the two velocity-filter dissipations of the omniSPH / `DFSPH2D` incompressible loop (post-solve, applied to the velocity, not forces).

    XSPH      v_i += sum_j c_j V_j W_ij (v_j - v_i)                     isotropic smoothing toward the neighbourhood mean; pairwise antisymmetric for a uniform c, so it conserves momentum
    friction  v_i -= min(c_w lambda_b, 1) (v_i - v_b)_tangential        a drag on the tangential velocity relative to the wall, `lambda_b = wm int W dA` the wall's completeness (analytic walls)

`computeXSPH` takes the fluid coefficient and, for boundary-particle scenes, a wall coefficient (the neighbouring wall particles enter with their own `v_j`, 0 for static ones, so their term is a drag
-- omniSPH's `BXSPH` without the wall-normal projection). For analytic walls the equivalent is `computeBoundaryFriction` (`DFSPH2D`'s `boundaryFriction`, which projects out the wall-normal part).
The weights are the apparent volumes `m_j / rho_j` of `WarpOperation.Interpolate` (`DFSPH2D` writes `2 V_j / (rho_i + rho_j)`; the two agree to the density variation, ~1 %).
"""
from typing import Any, Optional

import torch

from warpSPHCore import *

from ...enumTypes import *
from ..analyticBoundary import resolveWall
from ..analyticBoundary.wallTerms import _wallVelocity

__all__ = ['computeXSPH', 'computeBoundaryFriction']

F64 = torch.float64


def computeXSPH(state: Any, config: Any, schemeConfig: Any, adjacency: Any,
                fluidCoefficient: Optional[float] = None, boundaryCoefficient: Optional[float] = None) -> torch.Tensor:
    """The XSPH velocity increment `sum_j c_j V_j W_ij (v_j - v_i)` of the fluid rows (zero on every other row); `WarpOperation.Interpolate` computes `sum_j V_j g_j W_ij`, so folding the per-kind
    coefficient into the reference values collapses the fluid smoothing and the wall-particle drag into `interp(c v) - v_i interp(c)`.

    `fluidCoefficient` / `boundaryCoefficient` default to `schemeConfig.xsphCoefficient` / `schemeConfig.xsphBoundaryCoefficient` (0: off). Both zero returns zeros without a launch."""
    cf = float(getattr(schemeConfig, 'xsphCoefficient', 0.0) if fluidCoefficient is None else fluidCoefficient)
    cb = float(getattr(schemeConfig, 'xsphBoundaryCoefficient', 0.0) if boundaryCoefficient is None else boundaryCoefficient)
    if cf == 0.0 and cb == 0.0:
        return torch.zeros_like(state.velocities)
    fluid = state.kinds == 0
    c = torch.where(fluid, torch.full_like(state.densities, cf), torch.full_like(state.densities, cb))
    props = OperationProperties(kernel=config.kernel, operation=WarpOperation.Interpolate, supportMode=SupportScheme.SuperSymmetric)
    cv = warpOperation(state, props, domain=config.domain, referenceValues=state.velocities * c.unsqueeze(-1), adjacency=adjacency)
    cc = warpOperation(state, props, domain=config.domain, referenceValues=c, adjacency=adjacency)
    dv = cv - state.velocities * cc.unsqueeze(-1)
    return torch.where(fluid.unsqueeze(-1), dv, torch.zeros_like(dv))


def computeBoundaryFriction(state: Any, config: Any, schemeConfig: Any, adjacency: Any,
                            coefficient: Optional[float] = None, wall: Optional[Any] = None, perBody: bool = False) -> torch.Tensor:
    """The boundary-friction velocity increment of the analytic walls (`DFSPH2D`'s `boundaryFriction`), per body `b`:

        dv = - min(c lambda_b, 1) * T_b (v - v_wall,b),      T_b = I - n_b (x) n_b,   n_b = -G_b / |G_b|

    the tangential velocity relative to the wall continuum drag; `lambda_b = wm int W dA` (`WallState.lam`) the completeness of the wall at the particle, so the drag fades as the particle leaves the
    wall's support and saturates at 1 (the whole relative tangential velocity removed) in contact. `v_wall,b = int v_b W dA / int W dA = v_b(x) + omega J m1 / lambda` for a moving body (`m1 = int y W`,
    the provider's first moment). `coefficient` defaults to `schemeConfig.boundaryFriction` (0: off, zeros returned without touching the wall); `perBody` returns the increment of every body, [B, N, 2]. Evaluate at the positions and the body poses the
    friction acts at (the start of a step is the end of the last: `wall=None` resolves the scheme's provider there, shared with the rest of the step).

    No wall (boundary particles, or `wall=False`) returns zeros: their drag is `computeXSPH`'s `boundaryCoefficient`."""
    c = float(getattr(schemeConfig, 'boundaryFriction', 0.0) if coefficient is None else coefficient)
    zeros = torch.zeros_like(state.velocities)
    if c == 0.0:
        return zeros
    wall = resolveWall(state, config, schemeConfig, adjacency, wall)
    if wall is None:
        return zeros
    parts = []
    from warpSPHBoundaries.scene import WallOutput
    v = state.velocities.to(F64)
    out = torch.zeros_like(v)
    m1 = None
    for bi in range(wall.G.shape[0]):
        lam, G = wall.lam[bi], wall.G[bi]
        vw = _wallVelocity(wall, bi)
        omega = wall.omega[bi] if wall.omega is not None else None
        if omega is not None and not wall.pinned[bi] and bool(omega != 0):
            if m1 is None:
                m1 = wall.agg.evaluate((WallOutput('m1', 0, 'm1'),))['m1']
            lb = lam / wall.wm
            ok = lb > 1e-8
            num = vw * lb[:, None] + omega * torch.stack([-m1[bi][:, 1], m1[bi][:, 0]], dim=1)
            vw = torch.where(ok[:, None], num / lb.clamp(min=1e-300)[:, None], torch.zeros_like(num))
        nrm = G.norm(dim=1, keepdim=True)
        n = -G / nrm.clamp(min=1e-300)
        vr = v - vw
        tang = vr - (vr * n).sum(1, keepdim=True) * n
        fac = (c * lam).clamp(max=1.0)
        part = torch.where((nrm[:, 0] > 0)[:, None], -fac[:, None] * tang, torch.zeros_like(tang))
        parts.append(part)
        out = out + part
    fluid = (state.kinds == 0).unsqueeze(-1)
    if perBody:                                                       # [B, N, 2], for the load of every body
        return torch.where(fluid.unsqueeze(0), torch.stack(parts).to(state.velocities.dtype), torch.zeros_like(torch.stack(parts)).to(state.velocities.dtype))
    return torch.where(fluid, out.to(state.velocities.dtype), zeros)
