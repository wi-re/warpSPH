"""The delta-SPH (weakly compressible) step, and delta+-SPH (`Sun2017DeltaSPHConfig`: the same step with two config defaults).

`deltaSPH_step` builds the adjacency and then evaluates the right-hand side `_deltaSPH_rhs`, eagerly or (with `schemeConfig.cudaGraph`, when
`graphRoute.rhsIsGraphable`) from a CUDA graph. The right-hand side is a list of single-purpose regions (one `record_function` each, numbered in call order,
`[deltaSPH - NN]`):

    density (if unset) -> mDBC density -> boundary velocities -> Dirichlet BCs -> weakly compressible EOS -> analytic wall aggregates -> free-surface
    detection -> gradRho / gradRhoL -> density diffusion -> velocity diffusion -> continuity -> surface-aware pressure force (+ wall terms) -> forcing
    -> gravity -> mDBC no-penetration shift -> update -> BCs on the update

Details worth knowing (each used to be a paragraph of this docstring):

* ``computeMdbcDensity`` / ``computeBoundaryVelocities`` no-op when there are no boundary-kind particles; ``computeBoundaryVelocities`` overwrites only the
  boundary-kind entries of ``velocities`` and the overwrite is not reverted, so those entries carry the BC-enforced velocity through the step and into the
  next (no effect on ``dxdt``: non-fluid ``dxdt`` / ``dvdt`` / ``drhodt`` are zeroed after ``enforceUpdates``, so boundary motion comes from the BC machinery).
* ``gradRho`` / ``gradRhoL`` are computed only when ``schemeConfig.diffusionParams.densityDiffusionTerm`` needs them (``denormalized(Only)`` vs ``deltaSPH`` / ``deltaOnly``).
* ``freezeDiffusionAcrossStages`` (Sun et al. 2017 Sec. 2, the delta+ preset) evaluates the two diffusive terms once per real step and holds them across the RK stages.
* The mDBC no-penetration shift is a velocity correction made an acceleration by ``/ config.dt`` (the real step length, not the stage dt): see region 19.
"""

from ...configurations import SimulationConfig, WeaklyCompressibleSPHConfig
from ...systems import CompSPHSystem, WeaklyCompressibleSystemUpdate
from ...modules.boundaryConditions import computeForcing, enforceDirichlet, enforceUpdates
from ...modules.boundaryConditions.pinned import pinnedKeepWeight
from ...modules.deltaSPH import computeDensityDiffusion, computeVelocityDiffusion
from ...modules.density import computeDensities, computeGradRho, computeGradRhoL
from ...modules.eos import weaklyCompressibleEOS
from ...modules.gravity import computeGravity, computeBodyForce, bodyForceVector
from ...modules.mdbc import computeBoundaryVelocities, computeMdbcDensity, computeMdbcDensityBand, computeMdbcDensityEnglish2025, computeMdbcNoPenShift
from ...modules.momentum import computeMomentum
from ...modules.pressure import computePressureForceSurfaceAware
from ...modules.surfaceDetection import detectFreeSurface
from ...enumTypes import DensityDiffusionScheme
from warpSPHCore import (SupportScheme, buildVerletList, OperationProperties,
                         WarpOperation, warpOperation, RenormalizationState)
from warpSPHCore.profiling import record_function

import torch

from .analyticWall import bookWallLoads, evaluateStageWall, wallPressureViscousAcceleration
from .graphRoute import graphedRHS, liveConfigKey, rhsIsGraphable
from .surfaceMask import surfaceMaskAcrossStages

__all__ = ['deltaSPH_step']

#: (kept for the callers that import the old private names)
_rhsIsGraphable = rhsIsGraphable
_surfaceMaskAcrossStages = surfaceMaskAcrossStages


def deltaSPH_step(
    system: CompSPHSystem,
    dt: float,
    config: SimulationConfig,
    schemeConfig: WeaklyCompressibleSPHConfig,
    verbose = False,
    stageIndex = None,
):
    currentSystem = system
    currentState = currentSystem.state

    with record_function("[warpSPH] - [deltaSPH - 01] adjacency"):
        currentSystem.adjacency = buildVerletList(
            currentState,
            config.domain, verletScale = config.verletScale, supportMode = SupportScheme.SuperSymmetric,
            priorNeighborhood = currentSystem.adjacency,
            verbose = False)

    # Everything after the adjacency update is the right-hand side proper; with `schemeConfig.cudaGraph` it is captured once per
    # Verlet-list generation and replayed (graphRoute.py).
    if getattr(schemeConfig, 'cudaGraph', False) and rhsIsGraphable(schemeConfig, stageIndex):
        return graphedRHS(schemeConfig, _deltaSPH_rhs)(currentSystem, dt, config, schemeConfig, verbose, stageIndex,
                                                       extraKey=(stageIndex, liveConfigKey(schemeConfig)))
    return _deltaSPH_rhs(currentSystem, dt, config, schemeConfig, verbose, stageIndex)


def _gatedPressureRenormalization(currentState, config, adjacency, renormalizationState):
    """The renormalization matrices of the pressure force, gated per particle (DELTASPH_VALIDATION_PLAN sections 5.23 / 5.32).

    Applying ``Li`` unconditionally makes Marrone 3.1's free-surface peak 8.8x WORSE: ``Li`` comes from ``detectFreeSurface``'s covariance fit, which is most
    ill-conditioned exactly in the sparse / disordered neighbourhoods where the pressure-force truncation artifact lives. DualSPHysics' advanced-shifting
    extension instead gates the renormalized form on kernel-sum COMPLETENESS (``poup1 > 0.95``, a plain partition-of-unity sum) AND bulk classification, falling
    back to the raw gradient everywhere else (every free-surface particle included). Reproduced here on the matrix itself (per particle): the identity matrix for
    a gated-off particle makes ``matmul(Li, gradw_ij) == gradw_ij``, with no kernel change. Same support scheme as ``computePressureForceSurfaceAware``,
    ``surfaceIndicators`` as the bulk / free-surface classification."""
    completenessProps = OperationProperties(
        kernel=config.kernel, operation=WarpOperation.Interpolate,
        supportMode=SupportScheme.SuperSymmetric)
    kernelCompleteness = warpOperation(
        currentState, completenessProps, domain=config.domain,
        referenceValues=torch.ones_like(currentState.densities),
        adjacency=adjacency)
    wellConditioned = (kernelCompleteness > 0.95) & (currentState.surfaceIndicators == 0)
    Li = renormalizationState.renormalizationMatrices
    dim = currentState.positions.shape[1]
    identity = torch.eye(dim, dtype=Li.dtype, device=Li.device)
    gatedMatrices = torch.where(wellConditioned.view(-1, 1, 1), Li, identity)
    return RenormalizationState(renormalizationMatrices=gatedMatrices)


def _deltaSPH_rhs(
    currentSystem: CompSPHSystem,
    dt: float,
    config: SimulationConfig,
    schemeConfig: WeaklyCompressibleSPHConfig,
    verbose = False,
    stageIndex = None,
):
    currentState = currentSystem.state
    adjacency = currentSystem.adjacency

    with record_function("[warpSPH] - [deltaSPH - 02] density"):
        if currentState.densities is None:
            print("Computing densities...")
            currentState.densities = computeDensities(currentState, config, schemeConfig, adjacency)

    with record_function("[warpSPH] - [deltaSPH - 03] mDBC density"):
        _mdbcDensityScheme = getattr(schemeConfig, 'mdbcDensityScheme', 'ramped')
        if _mdbcDensityScheme == 'band':
            currentState.densities = computeMdbcDensityBand(currentState, config, schemeConfig, adjacency)
        elif _mdbcDensityScheme == 'english2025':
            currentState.densities = computeMdbcDensityEnglish2025(currentState, config, schemeConfig, adjacency)
        else:
            currentState.densities = computeMdbcDensity(currentState, config, schemeConfig, adjacency)

    with record_function("[warpSPH] - [deltaSPH - 04] boundary velocities"):
        currentState.velocities = computeBoundaryVelocities(currentState, config, schemeConfig, adjacency)

    with record_function("[warpSPH] - [deltaSPH - 05] Dirichlet BCs"):
        enforceDirichlet(currentSystem, currentSystem.t, config.dt, config, schemeConfig)

    with record_function("[warpSPH] - [deltaSPH - 06] EOS"):
        currentState.pressures = weaklyCompressibleEOS(currentState, schemeConfig)

    # Analytic boundaries (a boundary provider, no wall particles): the kernel integrals over the solid at the fluid particles, once per right-hand-side
    # evaluation. None without analytic bodies: the particle path above is then untouched.
    provider = getattr(schemeConfig, 'boundaryProvider', None)
    with record_function("[warpSPH] - [deltaSPH - 07] analytic wall aggregates"):
        wall = evaluateStageWall(currentState, config, schemeConfig, adjacency)

    with record_function("[warpSPH] - [deltaSPH - 08] free-surface detection"):
        fs, fsm, n, renormalizationState_, lMin = detectFreeSurface(currentState, config, schemeConfig, schemeConfig.surfaceDetectionConfig, adjacency, returnNormals = True)
        currentState.surfaceIndicators = (fsm > 0.5).to(torch.int32)
        currentState.surfaceNormals = n
        currentState.surfaceLambdas = lMin
        surfaceMaskAcrossStages(currentState, dt, schemeConfig, stageIndex, currentSystem.t)

    # The two diffusive terms. Sun et al. 2017 Sec. 2 (Antuono / Jameson technique) evaluates them ONCE per real step, at the committed t^n state, and holds
    # them fixed across every RK sub-stage: `schemeConfig.freezeDiffusionAcrossStages` (default False; the delta+ preset sets it) opts in. `stageIndex` is threaded in by
    # `warpSPHIntegrators.RungeKuttaB` (0, or None under a single-evaluation integrator where freezing is a no-op, for the first stage; 1.. for later ones).
    freezeDiffusion = getattr(schemeConfig, 'freezeDiffusionAcrossStages', False)
    isFirstStage = stageIndex is None or stageIndex == 0
    cached = getattr(schemeConfig, '_frozenDiffusionCache', None) if freezeDiffusion else None
    diffusionTerm = schemeConfig.diffusionParams.densityDiffusionTerm

    if freezeDiffusion and not isFirstStage and cached is not None:
        with record_function("[warpSPH] - [deltaSPH - 09-12] reuse frozen diffusion"):
            drhodt_diss, dvdt_diss = cached
    else:
        with record_function("[warpSPH] - [deltaSPH - 09] gradRho"):
            if diffusionTerm == DensityDiffusionScheme.denormalized or diffusionTerm == DensityDiffusionScheme.denormalizedOnly:
                gradRho = computeGradRho(currentState, config, schemeConfig, adjacency)
            else:
                gradRho = None
        with record_function("[warpSPH] - [deltaSPH - 10] gradRhoL"):
            if diffusionTerm == DensityDiffusionScheme.deltaSPH or diffusionTerm == DensityDiffusionScheme.deltaOnly:
                gradRhoL = computeGradRhoL(currentState, config, schemeConfig, adjacency, L = renormalizationState_)
            else:
                gradRhoL = None
        with record_function("[warpSPH] - [deltaSPH - 11] density diffusion"):
            drhodt_diss = computeDensityDiffusion(currentState, config, schemeConfig, adjacency, gradRho, gradRhoL)
        # `approachOnly=False`: Marrone 2011 Eq. (5b) / Sun 2017 Eq. (1) apply the artificial-viscosity term `alpha h c0 pi_ij` to *every* pair, not only
        # approaching ones; the approach-only clamp is Monaghan's shock viscosity, which under-damps the tensile / shear regions a violent free-surface impact
        # grows (DELTASPH_VALIDATION_PLAN Part 1).
        with record_function("[warpSPH] - [deltaSPH - 12] velocity diffusion"):
            dvdt_diss = computeVelocityDiffusion(currentState, config, schemeConfig, adjacency,
                                                 approachOnly=False, wall=wall)      # the wall's share is inside the (possibly frozen) tuple
        if freezeDiffusion:
            schemeConfig._frozenDiffusionCache = (drhodt_diss, dvdt_diss)

    with record_function("[warpSPH] - [deltaSPH - 13] continuity"):
        drhodt = computeMomentum(currentState, config, schemeConfig, adjacency, wall=wall)      # wall: free-slip mirror in the continuity equation

    with record_function("[warpSPH] - [deltaSPH - 14] pressure force"):
        pressureRenormalizationState = None
        if getattr(schemeConfig, 'pressureForceRenormalized', False):
            pressureRenormalizationState = _gatedPressureRenormalization(currentState, config, adjacency, renormalizationState_)
        dvdt_pressure = computePressureForceSurfaceAware(
            currentState, config, schemeConfig, adjacency,
            renormalizationState=pressureRenormalizationState, wall=wall)      # wall: the wall's pressure force, same Antuono switch as the fluid pairs

    with record_function("[warpSPH] - [deltaSPH - 15] wall pressure with viscous term"):
        accVP = wallPressureViscousAcceleration(wall, schemeConfig, dvdt_diss, currentState)
        if accVP is not None:
            dvdt_pressure = dvdt_pressure + accVP.sum(0).to(dvdt_pressure.dtype)

    bookWallLoads(wall, accVP, currentState, config, schemeConfig, adjacency, provider)           # (region 16: only when asked for or a body is dynamic)

    with record_function("[warpSPH] - [deltaSPH - 17] forcing"):
        forcing = computeForcing(currentSystem, config.dt, currentSystem.t, config, schemeConfig)
        dvdt_forcing = forcing / currentState.masses.view(-1,1)

    with record_function("[warpSPH] - [deltaSPH - 18] gravity"):
        gravity = computeGravity(currentState, config, schemeConfig, adjacency)
        dvdt_gravity = gravity
        if bodyForceVector(schemeConfig) is not None:
            dvdt_gravity = gravity + computeBodyForce(currentState, schemeConfig)          # the channel driver: momentum only, not in the density diffusion's hydrostatic term

    with record_function("[warpSPH] - [deltaSPH - 19] mDBC no-penetration shift"):
        # Only the `'derivative'` placement contributes here; `'finalize'` applies the correction once per step in `WeaklyCompressibleSystem.finalize` instead
        # (DualSPHysics' structure), and `'off'` drops it (DELTASPH_VALIDATION_PLAN 5.9).
        if getattr(schemeConfig, 'mdbcNoPenShiftMode', 'derivative') == 'derivative':
            nopenshift = computeMdbcNoPenShift(currentState, config, schemeConfig, adjacency)
            # `config.dt`, the real step length -- NOT the stage `dt` this function was called with. `nopenshift` is a *velocity* correction, so `/dt` makes it an
            # acceleration the integrator multiplies by the step length again; with the full step those cancel and the particle gets exactly the intended correction,
            # with a stage `dt` they do not. `symplecticEuler` calls the derivative with `dt/2` but updates with `dt`, so the correction landed at **2x** in every stage.
            dvdt_nopenshift = nopenshift / config.dt
        else:
            dvdt_nopenshift = torch.zeros_like(currentState.velocities)

    with record_function("[warpSPH] - [deltaSPH - 20] update"):
        update = WeaklyCompressibleSystemUpdate(
            dxdt = currentState.velocities.clone(),
            dvdt = dvdt_pressure + dvdt_forcing + dvdt_gravity + dvdt_diss+ dvdt_nopenshift,
            drhodt = drhodt + drhodt_diss,
            passive = torch.zeros(currentState.densities.shape, device=currentState.densities.device, dtype=torch.bool),
            # the velocity-linear part of drhodt, for the density as a drift field (warpSPHIntegrators/drift.py, CEILING_STICKING_PLAN.md section 6)
            drhodt_kin = drhodt,
        )

    with record_function("[warpSPH] - [deltaSPH - 21] enforce updates"):
        enforceUpdates(update, currentSystem, config.dt, currentSystem.t, config, schemeConfig)
        keepBand = pinnedKeepWeight(currentState, config, schemeConfig)           # a pinned band: no acceleration inside (mask-free, capturable)
        if keepBand is not None:
            update.dvdt = update.dvdt * keepBand
        nonFluidMask = (currentState.kinds != 0).unsqueeze(-1)
        update.dxdt = torch.where(nonFluidMask, torch.zeros_like(update.dxdt), update.dxdt)
        update.dvdt = torch.where(nonFluidMask, torch.zeros_like(update.dvdt), update.dvdt)
        update.drhodt = torch.where(nonFluidMask.squeeze(-1), torch.zeros_like(update.drhodt), update.drhodt)
        update.drhodt_kin = torch.where(nonFluidMask.squeeze(-1), torch.zeros_like(update.drhodt_kin), update.drhodt_kin)
    return update, adjacency, currentState
