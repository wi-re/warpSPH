"""Multi-iteration shifting driver: rebuilds the neighborhood, optionally
recomputes density and free-surface state, applies one `computeDeltaShift`
step per iteration, and (if `schemeConfig.surfaceDetectionConfig.active`)
projects the shift near the free surface before adding it to
`systemState.positions`.

Free-surface projection (`schemeConfig.shiftProperties.projectionScheme`,
`ShiftingProjectionScheme`) has four modes: `dot` removes the shift's normal
component and scales the tangential remainder by `surfaceScaling` for
surface particles; `mat` instead projects through a `(I - n n^T)` matrix and
scales by `lMin**2` (then zeroes the surface set anyway); the `zero` fallback
zeroes the shift outright for surface/near-surface particles; `surfaceNormal`
is the actual Sun et al. 2019 (`literature/sun2019`) Eq. (20)-(21) treatment
-- a surface particle whose shift points *into* the surface is cut to
tangential and curvature-gated (`surfaceCurvatureAngle`), one whose shift
points *away* keeps the full unconstrained shift, and `lMin` below
`surfaceLambdaThreshold` in the surface set is zeroed. `dot`/`mat`/`zero`
additionally zero the shift wherever `lMin < 0.4` (a fixed threshold);
`surfaceNormal` uses the configurable `surfaceLambdaThreshold` (default 0.4,
matching the old constant). All modes zero the shift for non-fluid particles
(`kinds != 0`). Normals/`lMin` are recomputed from `detectFreeSurface` each
iteration unless `shiftProperties.reuseNormals` and a prior surface state is
already cached on `systemState`.
"""

import math

import warp as wp
from warp.types import vector, matrix
from typing import Any
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from typing import Optional, Union, Tuple
from warpSPHCore import *




from warpSPH.configurations.simulationConfig import SimulationConfig
from ...enumTypes import *
from ...configurations.moduleConfigurations.surfaceDetection import SurfaceDetectionConfig

from ..util.wp_sum import warpSum
from ..util.wp_numNeighbors import countNeighborsWarp

from ..surfaceDetection import *
from ..density import *
from .delta import computeDeltaShift
from .implicitShifting import computeImplicitShift, computeDynamicImplicitShift
from ..util import *

from ...configurations.moduleConfigurations.shifting import ShiftProperties, ShiftingProjectionScheme, ShiftingScheme

__all__ = ['solveShifting']


def _curvatureGate(normals: torch.Tensor, surfaceMask: torch.Tensor,
                   adjacency: Any, cosThreshold: float) -> torch.Tensor:
    """`kappa` from Sun et al. 2019 Eq. (21): 0 for a particle any of whose
    *surface-set* neighbours' normals deviate from its own by more than the
    curvature angle (`arccos(n_i . n_j) >= angle`), 1 otherwise. Returned as a
    float tensor of shape `(N,)`. Particles with no surface neighbour keep
    `kappa = 1`.

    Only edges where both ends are in `surfaceMask` (the set `F`) are
    considered -- the normal field is only meaningful there, and interior
    normals are ~0 from `LambdaGrad`, which would otherwise gate every
    surface particle adjacent to the bulk.

    `cosThreshold = cos(angle)`; a *larger* min dot product means a flatter
    neighbourhood, so the gate is `min_j (n_i . n_j) >= cosThreshold`.
    """
    i, j = adjacency.i, adjacency.j
    keep = surfaceMask[i] & surfaceMask[j]
    dots = (normals[i] * normals[j]).sum(dim=-1)
    dots = torch.where(keep, dots, dots.new_tensor(float('inf')))
    minDot = normals.new_full((normals.shape[0],), float('inf'))
    minDot.scatter_reduce_(0, i.to(torch.int64), dots, reduce='amin', include_self=False)
    return (minDot >= cosThreshold).to(normals.dtype)


def solveShifting(
    systemState: Any,
    config: SimulationConfig, schemeConfig: Any,
    adjacency: Optional[Union[AdjacencyList, CompactHashMap]],
    dt: float,
    verbose: bool = False):
    with record_function("[warpSPH] - shift"):
        domain = config.domain
        kernel = config.kernel

        shiftIters = schemeConfig.shiftProperties.iterations
        summationDensity = schemeConfig.shiftProperties.summationDensity
        freeSurface = schemeConfig.surfaceDetectionConfig.active
        freeSurfaceScheme = schemeConfig.surfaceDetectionConfig.scheme
        normalScheme = schemeConfig.surfaceDetectionConfig.normalSource
        projectionScheme = schemeConfig.shiftProperties.projectionScheme
        surfaceScaling = schemeConfig.shiftProperties.surfaceScaling
        shiftingThreshold = schemeConfig.shiftProperties.threshold
        surfaceLambdaThreshold = getattr(schemeConfig.shiftProperties, 'surfaceLambdaThreshold', 0.4)
        surfaceLambdaTaper = getattr(schemeConfig.shiftProperties, 'surfaceLambdaTaper', 0.0)
        surfaceCurvatureAngle = getattr(schemeConfig.shiftProperties, 'surfaceCurvatureAngle', 15.0)
        maxShiftVelocityFraction = getattr(schemeConfig.shiftProperties, 'maxShiftVelocityFraction', 0.5)

        rho0 = schemeConfig.fluid.restDensity
        spacing = torch.pow(systemState.masses / rho0, 1/systemState.positions.shape[1]).mean().cpu().item()
        projectQuantities = schemeConfig.shiftProperties.projectQuantities

        initialPositions = systemState.positions.clone()
        initialDensities = systemState.densities.clone()

        for i in range(shiftIters):
            with record_function(f"[warpSPH] - (shift) - adjacency"):
                adjacency = buildVerletList(
                    systemState, 
                    config.domain, verletScale = config.verletScale, supportMode = SupportScheme.SuperSymmetric,
                    priorNeighborhood = adjacency,
                    verbose = False)

            with record_function(f"[warpSPH] - (shift) - countNeighbors"):
                numNeighbors = countNeighbors(systemState, config, schemeConfig, adjacency)

            if summationDensity:
                with record_function(f"[warpSPH] - (shift) - computeDensities"):
                    systemState.densities = computeDensities(systemState, config, schemeConfig, adjacency)
                # ADD MDBC HERE
                
            if freeSurface:
                with record_function(f"[warpSPH] - (shift) - detectFreeSurface"):
                    if schemeConfig.shiftProperties.reuseNormals and systemState.surfaceNormals is not None and systemState.surfaceLambdas is not None:
                        n = systemState.surfaceNormals
                        lMin = systemState.surfaceLambdas
                        surfaceIndicator = systemState.surfaceIndicators == 1
                    else:
                        fs, fsm, n, renormalizationState_, lMin = detectFreeSurface(systemState, config, schemeConfig, schemeConfig.surfaceDetectionConfig, adjacency, returnNormals = True)

                        surfaceIndicator = fsm > 0.5
                    # C, Evals, renormalizationState_ = computeRenormalizationMatrices(
                    #     queryParticles = systemState,
                    #     operationProperties = OperationProperties(
                    #         kernel = config.kernel,
                    #         operation = WarpOperation.Gradient,
                    #         operationMode = OperationDirection.AllToAll,
                    #         supportMode = SupportScheme.SuperSymmetric
                    #     ),
                    #     domain = config.domain,
                    #     adjacency = adjacency,
                    #     returnEigVals = True
                    # )
                    # lMin = torch.min(torch.abs(Evals), dim = -1).values
            else:
                fs = fsm = n = lMin = None

            with record_function(f"[warpSPH] - (shift) - computeShift"):
                if schemeConfig.shiftProperties.scheme == ShiftingScheme.implicit:
                    update, adjacency = computeImplicitShift(systemState, config, schemeConfig, domain, adjacency, iters = 1)
                elif schemeConfig.shiftProperties.scheme == ShiftingScheme.dynamic:
                    update, adjacency = computeDynamicImplicitShift(systemState, config, schemeConfig, domain, adjacency, iters = 1)
                else:
                    update, adjacency = computeDeltaShift(systemState, config, schemeConfig, domain, adjacency, iters = 1)
            # print(f"Iteration {i} [inside solveShifting], max shift magnitude: {update.norm(dim=1).max().item()}")


            if freeSurface:
                with record_function(f"[warpSPH] - (shift) - projectShift"):
                    # lMin = lMin * float(eval_kernelScale(config.kernel.value, config.dim))
                    if projectionScheme == ShiftingProjectionScheme.dot:
                        result = update - torch.einsum('ij,ij->i', update, n).view(-1,1) * n
                        update[fsm > 0.5] = result[fsm > 0.5] * surfaceScaling
                        update[lMin < 0.4] = 0
                    elif projectionScheme == ShiftingProjectionScheme.mat:
                        nMat = torch.einsum('ij, ik -> ikj', n, n)
                        M = torch.diag_embed(systemState.positions.new_ones(systemState.positions.shape)) - nMat
                        result = torch.bmm(M, update.unsqueeze(-1)).squeeze(-1)
                        
                        # update[surfaceIndicator] = result[surfaceIndicator] * surfaceScaling * 5.0
                        update[surfaceIndicator] = (lMin**2.0).view(-1,1)[surfaceIndicator] * result[surfaceIndicator]
                        # update[fs > 0.5] = result[fs> 0.5] * surfaceScaling
                        # update[surfaceIndicator] = 0.0
                        update[lMin < 0.4] = 0
                        update[surfaceIndicator]=0.0
                    elif projectionScheme == ShiftingProjectionScheme.surfaceNormal:
                        # Sun et al. 2019 (literature/sun2019) Eq. (20)-(21).
                        # `n` is the outward surface normal (LambdaGrad/Maronne),
                        # `surfaceIndicator` the dilated surface set F, `lMin`
                        # the min renormalisation-matrix eigenvalue (paper's
                        # lambda). Unlike dot/mat this reads only fields that are
                        # also populated on the `reuseNormals` fast path.
                        inF = surfaceIndicator
                        outward = torch.einsum('ij,ij->i', update, n)   # n . delta-u*
                        tangential = update - outward.view(-1, 1) * n
                        if surfaceCurvatureAngle > 0.0:
                            cosT = math.cos(math.radians(surfaceCurvatureAngle))
                            kappa = _curvatureGate(n, inF, adjacency, cosT).view(-1, 1)
                        else:
                            kappa = update.new_ones((update.shape[0], 1))
                        # in F, shift points into the surface -> tangential (kappa-gated);
                        # in F, shift points away             -> full shift (anti-clustering);
                        # not in F                             -> full shift.
                        restrict = inF & (outward >= 0)
                        update = torch.where(restrict.view(-1, 1), kappa * tangential, update)
                        # lambda gate (Eq. 20 row 1): a hard zero below
                        # `surfaceLambdaThreshold` (taper == 0), else a smoothstep
                        # ramp over `[threshold, threshold + taper]` -- the hard
                        # step is itself a disorder source one layer into the bulk.
                        if surfaceLambdaTaper > 0.0:
                            x = ((lMin - surfaceLambdaThreshold) / surfaceLambdaTaper).clamp(0.0, 1.0)
                            wLambda = (x * x * (3.0 - 2.0 * x)).view(-1, 1)
                        else:
                            wLambda = (lMin >= surfaceLambdaThreshold).to(update.dtype).view(-1, 1)
                        inFcol = inF.view(-1, 1)
                        update = torch.where(inFcol, update * wLambda, update)
                    else:
                        update[fsm > 0.5] = 0
                        update[lMin < 0.4] = 0
                        update[fs > 0.5] = 0
                
            # Sun et al. 2019 Eq. (14): cap the shift magnitude at a fraction of
            # Umax * dt (Umax = max finite particle speed). Physically tied to
            # the flow, unlike the fixed per-component `threshold` clamp below,
            # and the thing that stops a locally exploding grad(C) from feeding
            # an oversized shift into `correctdrhodt`.
            if maxShiftVelocityFraction > 0.0:
                velMag = torch.linalg.norm(systemState.velocities, dim=-1)
                velMag = velMag[torch.isfinite(velMag)]
                uMax = velMag.max() if velMag.numel() > 0 else update.new_tensor(0.0)
                capLength = maxShiftVelocityFraction * uMax * dt
                if capLength > 0:
                    mag = torch.linalg.norm(update, dim=-1, keepdim=True)
                    update = update * (capLength / mag.clamp_min(1e-30)).clamp_(max=1.0)
            update = torch.clamp(update, -shiftingThreshold * spacing, shiftingThreshold * spacing)
            update[systemState.kinds != 0] = 0

            systemState.positions += update# * dt
                        
        dx = systemState.positions - initialPositions
        systemState.positions = initialPositions
        systemState.densities = initialDensities


        return dx
                