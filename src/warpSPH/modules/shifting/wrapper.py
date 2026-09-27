"""Multi-iteration shifting driver: rebuilds the neighborhood, optionally
recomputes density and free-surface state, applies one `computeDeltaShift`
step per iteration, and (if `schemeConfig.surfaceDetectionConfig.active`)
projects the shift near the free surface before adding it to
`systemState.positions`.

Free-surface projection (`schemeConfig.shiftProperties.projectionScheme`,
`ShiftingProjectionScheme`) has five modes: `dot` removes the shift's normal
component and scales the tangential remainder by `surfaceScaling` for
surface particles; `mat` instead projects through a `(I - n n^T)` matrix and
scales by `lMin**2` (then zeroes the surface set anyway); the `zero` fallback
zeroes the shift outright for surface/near-surface particles; `surfaceNormal`
is the actual Sun et al. 2019 (`literature/sun2019`) Eq. (20)-(21) treatment
-- a surface particle whose shift points *into* the surface is cut to
tangential and curvature-gated (`surfaceCurvatureAngle`), one whose shift
points *away* keeps the full unconstrained shift, and `lMin` below
`surfaceLambdaThreshold` in the surface set is zeroed; `michel2022` is Michel
et al. 2022's (`literature/michel2022`) Eq. (48) treatment -- see
`ShiftingProjectionScheme.michel2022`'s own docstring -- and is the only mode
paired with `ShiftingScheme.michel2022` rather than `deltaSPH`/`implicit`/
`dynamic`. `dot`/`mat`/`zero` additionally zero the shift wherever
`lMin < 0.4` (a fixed threshold); `surfaceNormal`/`michel2022` use the
configurable `surfaceLambdaThreshold` and its own literal `0.4` respectively.
All modes zero the shift for non-fluid particles (`kinds != 0`). Normals/
`lMin` are recomputed from `detectFreeSurface` each iteration unless
`shiftProperties.reuseNormals` and a prior surface state is already cached on
`systemState`.
"""

import math

import warp as wp
from warp.types import vector, matrix
from typing import Any
import torch
from torch.profiler import profile, ProfilerActivity
from warpSPHCore.profiling import record_function
from warpSPHCore import compileGlue, compileGlueEnabled
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
from .michel import computeMichelShift
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
    dots = torch.where(keep, dots, torch.full_like(dots, float('inf')))
    minDot = normals.new_full((normals.shape[0],), float('inf'))
    minDot.scatter_reduce_(0, i.to(torch.int64), dots, reduce='amin', include_self=False)
    return (minDot >= cosThreshold).to(normals.dtype)


@compileGlue
def _restrictSurfaceShift(update, n, inF, kappa):
    """Sun et al. 2019 Eq. (20)-(21), the direction part: in the surface set
    F a shift pointing into the surface keeps only its (kappa-gated)
    tangential part; a shift pointing away from it, and every shift outside
    F, is kept whole (anti-clustering). Pure torch (`compileGlue`)."""
    outward = torch.einsum('ij,ij->i', update, n)   # n . delta-u*
    tangential = update - outward.view(-1, 1) * n
    # in F, shift points into the surface -> tangential (kappa-gated);
    # in F, shift points away             -> full shift (anti-clustering);
    # not in F                             -> full shift.
    restrict = inF & (outward >= 0)
    return torch.where(restrict.view(-1, 1), kappa * tangential, update)


@compileGlue
def _lambdaGateShift(update, gateEvals, inF, threshold: float, taper: float):
    """Eq. (20) row 1: scale the shift in F by the lambda gate -- a hard zero
    below `threshold` (taper == 0), else a smoothstep over
    `[threshold, threshold + taper]`. Pure torch (`compileGlue`)."""
    lMinGate = torch.min(torch.abs(gateEvals), dim=-1).values
    if taper > 0.0:
        x = ((lMinGate - threshold) / taper).clamp(0.0, 1.0)
        wLambda = (x * x * (3.0 - 2.0 * x)).view(-1, 1)
    else:
        wLambda = (lMinGate >= threshold).to(update.dtype).view(-1, 1)
    inFcol = inF.view(-1, 1)
    return torch.where(inFcol, update * wLambda, update)


@compileGlue
def _capAndClampShift(update, velocities, kinds, maxShiftVelocityFraction: float, dt, bound):
    """The shift limits: Sun et al. 2019 Eq. (14)'s magnitude cap at a
    fraction of Umax * dt (Umax = max finite particle speed; skipped at
    fraction 0), the per-component clamp at `bound`, and zero on every
    non-fluid row. Branchless and gather-free -- no host sync; the same
    values as the `velMag[isfinite]` / `if capLength > 0` form it replaced.
    Pure torch (`compileGlue`)."""
    if maxShiftVelocityFraction > 0.0:
        velMag = torch.linalg.norm(velocities, dim=-1)
        finiteV = torch.isfinite(velMag)
        uMax = torch.where(
            finiteV.any(),
            torch.where(finiteV, velMag, torch.full_like(velMag, float('-inf'))).max(),
            torch.zeros_like(velMag[0]))
        capLength = maxShiftVelocityFraction * uMax * dt
        mag = torch.linalg.norm(update, dim=-1, keepdim=True)
        capped = update * (capLength / mag.clamp_min(1e-30)).clamp_(max=1.0)
        update = torch.where(capLength > 0, capped, update)
    update = torch.clamp(update, -bound, bound)
    return torch.where((kinds != 0).unsqueeze(-1), torch.zeros_like(update), update)


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
        # kept on the device (no host read); the clamp bounds below are formed
        # in float64 exactly as the host float arithmetic used to form them
        spacing = torch.pow(systemState.masses / rho0, 1/systemState.positions.shape[1]).mean()
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
                    # Michel et al. 2022's Eq. (47)/(48) need the *raw* free-surface
                    # set (d^FS is the distance to the nearest surface particle),
                    # but every scheme caches the *dilated* set in
                    # `surfaceIndicators` (`fsm > 0.5` in deltaSPH/divergenceFree/
                    # artificialCompressible). Reusing it put every particle within
                    # one support of the surface at d^FS = 0 -- beta = 1 and the
                    # normal fully cancelled across that whole layer -- so
                    # michel2022 always re-detects.
                    reuseCached = (schemeConfig.shiftProperties.reuseNormals
                                   and schemeConfig.shiftProperties.scheme != ShiftingScheme.michel2022
                                   and systemState.surfaceNormals is not None
                                   and systemState.surfaceLambdas is not None)
                    if reuseCached:
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

            michelDFS = michelNTilde = None
            if schemeConfig.shiftProperties.scheme == ShiftingScheme.michel2022:
                with record_function(f"[warpSPH] - (shift) - michel beta"):
                    # Eq. (21)'s coefficient, counterbalancing the lowest-
                    # degree truncation term; Eq. (48) requires its
                    # free-surface decay to happen here, *before* the
                    # interior law (modules/shifting/michel.py) is evaluated,
                    # since both branches of that law's norm clamp depend on
                    # beta -- see PST_ALE_PLAN.md Part 2.3.
                    R_i = systemState.supports
                    achievedDx_i = torch.pow(systemState.masses / rho0, 1.0 / systemState.positions.shape[1])
                    betaInterior = (R_i / achievedDx_i) ** 3.0
                    if freeSurface:
                        # Eq. (47)'s search target is the *raw*, undilated
                        # free-surface mask -- `fs` on the fresh-detect path
                        # (detectFreeSurface returns `(fsm_raw, fs_dilated,
                        # ...)`, and this file's own unpacking above binds
                        # position 1 to the name `fs`) -- always freshly bound
                        # for michel2022, see `reuseCached`.
                        rawSurfaceMask = fs.to(n.dtype)
                        michelDFS, michelNTilde = computeNearestSurfaceNormalWarp(
                            systemState,
                            operationProperties=OperationProperties(
                                operation=WarpOperation.Density,
                                kernel=kernel,
                                supportMode=SupportScheme.Gather,
                            ),
                            domain=domain,
                            adjacency=adjacency,
                            freeSurfaceMask=rawSurfaceMask,
                            normals=n,
                        )
                        # Linear decay from beta=1 at d^FS=0 (on the surface)
                        # to beta=(R/dx)^3 at d^FS=R (the vicinity region's
                        # own extent, matching where Eq. 48's sigma also
                        # bottoms out).
                        decay = torch.clamp(michelDFS / R_i, 0.0, 1.0)
                        michelBeta = 1.0 + (betaInterior - 1.0) * decay
                    else:
                        michelBeta = betaInterior

            with record_function(f"[warpSPH] - (shift) - computeShift"):
                if schemeConfig.shiftProperties.scheme == ShiftingScheme.implicit:
                    update, adjacency = computeImplicitShift(systemState, config, schemeConfig, domain, adjacency, iters = 1)
                elif schemeConfig.shiftProperties.scheme == ShiftingScheme.dynamic:
                    update, adjacency = computeDynamicImplicitShift(systemState, config, schemeConfig, domain, adjacency, iters = 1)
                elif schemeConfig.shiftProperties.scheme == ShiftingScheme.michel2022:
                    update, adjacency = computeMichelShift(systemState, config, schemeConfig, domain, adjacency, beta = michelBeta, dt = dt, iters = 1)
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
                        if surfaceCurvatureAngle > 0.0:
                            cosT = math.cos(math.radians(surfaceCurvatureAngle))
                            kappa = _curvatureGate(n, inF, adjacency, cosT).view(-1, 1)
                        else:
                            kappa = update.new_ones((update.shape[0], 1))
                        update = _restrictSurfaceShift(update, n, inF, kappa)
                        # lambda gate (Eq. 20 row 1): a hard zero below
                        # `surfaceLambdaThreshold` (taper == 0), else a smoothstep
                        # ramp over `[threshold, threshold + taper]` -- the hard
                        # step is itself a disorder source one layer into the bulk.
                        #
                        # Sun 2019 Sec. 2.5 (the paragraph right after Eq. (21)):
                        # "the field lambda evaluated with the ghost particles
                        # cannot be used in (20), and it needs to be re-evaluated
                        # without considering the ghost particles... crucial for
                        # maintaining the simulation stable when thin liquid jets
                        # running on the solid wall occurs." `n`/`lMin` above come
                        # from `detectFreeSurface`'s AllToAll pass (ghosts
                        # included, correct for Eq. (19)'s normal) -- re-evaluate
                        # lambda fluid-only, reusing the same adjacency (a kind
                        # filter on an already-built neighbour list, no new
                        # search), so a thin near-wall fluid layer that only
                        # *looks* well-supported because of ghost padding still
                        # gets gated here.
                        _, gateEvals, _ = computeRenormalizationMatrices(
                            systemState,
                            operationProperties=OperationProperties(
                                kernel=kernel,
                                operation=WarpOperation.Gradient,
                                operationMode=OperationDirection.FluidToFluid,
                                supportMode=SupportScheme.SuperSymmetric,
                            ),
                            domain=domain, adjacency=adjacency, returnEigVals=True,
                        )
                        update = _lambdaGateShift(update, gateEvals, inF,
                                                  float(surfaceLambdaThreshold),
                                                  float(surfaceLambdaTaper))
                    elif projectionScheme == ShiftingProjectionScheme.michel2022:
                        # Michel et al. 2022 (literature/michel2022) Eq. (48).
                        # `n` here plays no role -- the projection uses the
                        # *inherited* normal `michelNTilde` (Eq. 47) instead.
                        R_i = systemState.supports
                        sigma = torch.clamp(
                            (michelDFS - R_i) / (0.5 * R_i - R_i), 0.0, 1.0
                        )
                        outward = torch.einsum('ij,ij->i', update, michelNTilde)
                        projected = update - (sigma * outward).view(-1, 1) * michelNTilde
                        lambdaGate = (lMin >= 0.4).to(update.dtype)
                        result = (lambdaGate * lMin.pow(2.0)).view(-1, 1) * projected
                        update = torch.where(surfaceIndicator.view(-1, 1), result, update)
                    else:
                        update[fsm > 0.5] = 0
                        update[lMin < 0.4] = 0
                        update[fs > 0.5] = 0
                
            # Sun et al. 2019 Eq. (14): cap the shift magnitude at a fraction of
            # Umax * dt (Umax = max finite particle speed). Physically tied to
            # the flow, unlike the fixed per-component `threshold` clamp below,
            # and the thing that stops a locally exploding grad(C) from feeding
            # an oversized shift into `correctdrhodt`.
            bound = (shiftingThreshold * spacing.double()).to(update.dtype)
            if compileGlueEnabled() and not isinstance(dt, torch.Tensor):
                # one compiled version for the host-float dt of an eager step
                # and the device dt of a captured one (a float is specialized
                # on its value); float32 either way, as the scalar product was
                dt = torch.tensor(dt, dtype=update.dtype, device=update.device)
            update = _capAndClampShift(update, systemState.velocities, systemState.kinds,
                                       float(maxShiftVelocityFraction), dt, bound)

            systemState.positions += update# * dt
                        
        dx = systemState.positions - initialPositions
        systemState.positions = initialPositions
        systemState.densities = initialDensities


        return dx
                