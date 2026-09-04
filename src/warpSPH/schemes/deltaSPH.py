"""The delta-SPH (weakly-compressible) step: build adjacency, then density
via `computeDensities` (only if unset) followed by an mDBC density
correction (`computeMdbcDensity`). The call site's inline comment claims this
is "skipped since no boundaries are present", which is stale as a comment --
the call itself is unconditional -- but harmless in practice, since
`computeMdbcDensity`/`computeBoundaryVelocities` both no-op internally
(`if not torch.any(currentState.kinds == 1): return unchanged`) when there
are no boundary-kind particles. `computeBoundaryVelocities` only overwrites
the boundary-kind (`kinds == 1`) entries of `currentState.velocities` (fluid
entries pass through unchanged); the in-place update is not reverted
afterwards (the revert line is commented out), so those entries carry the
BC-enforced velocity into the rest of the step and into the next one -- for
`dxdt` specifically this has no observable effect, since non-fluid particles'
`dxdt`/`dvdt`/`drhodt` are zeroed after `enforceUpdates` regardless (see
below). After the weakly-compressible EOS and free-surface detection,
`gradRho`/`gradRhoL` are computed only when
`schemeConfig.diffusionParams.densityDiffusionTerm` actually needs them
(`denormalized(Only)` vs `deltaSPH`/`deltaOnly`) and fed to
`computeDensityDiffusion`; velocity diffusion, continuity `drhodt`,
surface-aware pressure force, forcing, gravity, and the mDBC no-penetration
shift (converted from a position correction to an acceleration via `/dt`)
are summed into the final update. Non-fluid particles (`kinds != 0`) have
their `dxdt`/`dvdt`/`drhodt` zeroed after `enforceUpdates`, so boundary/ghost
motion comes entirely from the BC machinery, not integration. Large stretches
of commented-out code (an alternate Shepard-corrected density path, per-term
diagnostic prints, a `TimedBlock`/`performanceDict` profiling scaffold
superseded by the active `record_function` markers) are left in place, not
removed (tracked separately in docs/historic_plans/CLEANUP_PLAN.md).
"""

from ..configurations import SimulationConfig, WeaklyCompressibleSPHConfig
from ..systems import CompSPHSystem, WeaklyCompressibleSystemUpdate
from ..modules.boundaryConditions import computeForcing, enforceDirichlet, enforceUpdates
from ..modules.deltaSPH import computeDensityDiffusion, computeVelocityDiffusion
from ..modules.density import computeDensities, computeGradRho, computeGradRhoL
from ..modules.eos import weaklyCompressibleEOS
from ..modules.gravity import computeGravity
from ..modules.mdbc import computeBoundaryVelocities, computeMdbcDensity, computeMdbcNoPenShift
from ..modules.momentum import computeMomentum
from ..modules.pressure import computePressureForceSurfaceAware
from ..modules.surfaceDetection import detectFreeSurface
from ..modules.util import countNeighbors
from ..enumTypes import DensityDiffusionScheme
from warpSPHCore import SupportScheme, buildVerletList

import torch
from ..utils.timer import TimedBlock
from torch.profiler import profile, record_function, ProfilerActivity

__all__ = ['deltaSPH_step']


def deltaSPH_step(
    system: CompSPHSystem,
    dt: float,
    config: SimulationConfig,
    schemeConfig: WeaklyCompressibleSPHConfig,
    verbose = False,        
):
    currentSystem = system#
    currentState = currentSystem.state
    adjacency = currentSystem.adjacency

    # 1. Compute adjacency
    # with TimedBlock('compute adjacency', use_cuda=True, device=config.device) as tb_adjacency:
    with record_function("[warpSPH] - [deltaSPH - 01] - compute adjacency"):
        verletScale = config.verletScale

        adjacency = buildVerletList(
            currentState, 
            config.domain, verletScale = config.verletScale, supportMode = SupportScheme.SuperSymmetric,
            priorNeighborhood = adjacency,
            verbose = False)
        currentSystem.adjacency = adjacency

    # 2. Compute density if density is none
    # with TimedBlock('compute density', use_cuda=True, device=config.device) as tb_density:
    with record_function("[warpSPH] - [deltaSPH - 02] - compute density"):
        if currentState.densities is None:
            print("Computing densities...")
            currentState.densities = computeDensities(currentState, config, schemeConfig, adjacency)
    # 3. Skipped mDBC density computation since no boundaries are present
    # with TimedBlock('compute mDBC density', use_cuda=True, device=config.device) as tb_mdbc:
    with record_function("[warpSPH] - [deltaSPH - 03] - compute mDBC density"):
        numNeighbors = countNeighbors(currentState, config, schemeConfig, adjacency)
        # rho = computeDensities(currentState, config, schemeConfig, adjacency)
        # rhoCurrent = currentState.densities.clone()
        # currentState.densities = rho
        # shepDenom = warpOperation(
        #     currentState,
        #     OperationProperties(
        #         kernel = config.kernel,
        #         operation = WarpOperation.Interpolate,
        #         supportMode = SupportScheme.Gather, # cullen switch E.1 in the CRK paper uses gather for density estimation
        #     ),
        #     queryValues = torch.ones_like(currentState.densities),
        #     domain = config.domain,
        #     adjacency = adjacency,
        # )
        # currentState.densities = rhoCurrent
        # mask = torch.logical_and(numNeighbors < 9, currentState.kinds == 0)
        # currentState.densities[mask] = rho[mask] / shepDenom[mask]
        # currentState.densities[numNeighbors == 1] = schemeConfig.fluid.restDensity

        currentState.densities = computeMdbcDensity(currentState, config, schemeConfig, adjacency)

        # print(f'Fluid density stats: min={currentState.densities[currentState.kinds == 0].min().item()}, max={currentState.densities[currentState.kinds == 0].max().item()}, mean={currentState.densities[currentState.kinds == 0].mean().item()}')
        # print(f'Boundary density stats: min={currentState.densities[currentState.kinds == 1].min().item()}, max={currentState.densities[currentState.kinds == 1].max().item()}, mean={currentState.densities[currentState.kinds == 1].mean().item()}')
    # 4. enforce BCs
    with record_function("[warpSPH] - [deltaSPH - 06] - compute boundary velocities"):
        currentVelocities = currentState.velocities.clone()
        currentState.velocities = computeBoundaryVelocities(currentState, config, schemeConfig, adjacency)
    

    # with TimedBlock('enforce BCs', use_cuda=True, device=config.device) as tb_bcs:
    with record_function("[warpSPH] - [deltaSPH - 04] - enforce BCs"):
        enforceDirichlet(currentSystem, currentSystem.t, config.dt, config, schemeConfig)
    # 5. compute EOS (WC version)
    # with TimedBlock('compute EOS', use_cuda=True, device=config.device) as tb_eos:
    with record_function("[warpSPH] - [deltaSPH - 05] - compute EOS"):
        currentState.pressures = weaklyCompressibleEOS(currentState, schemeConfig)
    # 6. Skipped boundary velocity computation since no boundaries are present
    # with TimedBlock('compute boundary velocities', use_cuda=True, device=config.device) as tb_boundary_velocities:

    # 7. Compute Covariance Matrices for gradRho_l terms
    # Done in gradRhoL for now
    # 8. Run surface detection (only if free surface)
    # with TimedBlock('compute surface detection', use_cuda=True, device=config.device) as tb_surface:
    with record_function("[warpSPH] - [deltaSPH - 08] - compute surface detection"):
        fs, fsm, n, renormalizationState_, lMin = detectFreeSurface(currentState, config, schemeConfig, schemeConfig.surfaceDetectionConfig, adjacency, returnNormals = True) 
        currentState.surfaceIndicators = (fsm > 0.5).to(torch.int32)
        currentState.surfaceNormals = n
        currentState.surfaceLambdas = lMin
        # print(f'Surface particles: {currentState.surfaceIndicators.sum().item()} / {currentState.surfaceIndicators.shape[0]} ({100 * currentState.surfaceIndicators.sum().item() / currentState.surfaceIndicators.shape[0]:.2f}%)')


    #9. Compute gradRho and gradRhoL
    # with TimedBlock('compute gradRho', use_cuda=True, device=config.device) as tb_gradRho:
    with record_function("[warpSPH] - [deltaSPH - 09] - compute gradRho and gradRhoL"):
        if schemeConfig.diffusionParams.densityDiffusionTerm == DensityDiffusionScheme.denormalized or schemeConfig.diffusionParams.densityDiffusionTerm == DensityDiffusionScheme.denormalizedOnly:
            gradRho = computeGradRho(currentState, config, schemeConfig, adjacency)
        else:
            gradRho = None
    # with TimedBlock('compute gradRhoL', use_cuda=True, device=config.device) as tb_gradRhoL:
    with record_function("[warpSPH] - [deltaSPH - 09] - compute gradRhoL"):
        if schemeConfig.diffusionParams.densityDiffusionTerm == DensityDiffusionScheme.deltaSPH or schemeConfig.diffusionParams.densityDiffusionTerm == DensityDiffusionScheme.deltaOnly:
            gradRhoL = computeGradRhoL(currentState, config, schemeConfig, adjacency, L = renormalizationState_)
        else:
            gradRhoL = None


    # 10. Compute drhodt_diss
    # with TimedBlock('compute drhodt_diss', use_cuda=True, device=config.device) as tb_drhodt_diss:
    with record_function("[warpSPH] - [deltaSPH - 10] - compute drhodt_diss"):
        drhodt_diss = computeDensityDiffusion(currentState, config, schemeConfig, adjacency, gradRho, gradRhoL)

    # 11. Compute dvdt_diss
    # with TimedBlock('compute dvdt_diss', use_cuda=True, device=config.device) as tb_dvdt_diss:
    with record_function("[warpSPH] - [deltaSPH - 11] - compute dvdt_diss"):
        dvdt_diss = computeVelocityDiffusion(currentState, config, schemeConfig, adjacency)
    
    # 12. Compute drhodt
    # with TimedBlock('compute drhodt', use_cuda=True, device=config.device) as tb_drhodt:
    with record_function("[warpSPH] - [deltaSPH - 12] - compute drhodt"):
        drhodt = computeMomentum(currentState, config, schemeConfig, adjacency)

    # 13. Compute dvdt from pressure
    # with TimedBlock('compute dvdt', use_cuda=True, device=config.device) as tb_dvdt:
    with record_function("[warpSPH] - [deltaSPH - 13] - compute dvdt from pressure"):
        dvdt_pressure = computePressureForceSurfaceAware(currentState, config, schemeConfig, adjacency)

    # 14. Apply forcing
    # with TimedBlock('compute forcing', use_cuda=True, device=config.device) as tb_forcing:
    with record_function("[warpSPH] - [deltaSPH - 14] - compute forcing"):
        forcing = computeForcing(currentSystem, config.dt, currentSystem.t, config, schemeConfig)
        dvdt_forcing = forcing / currentState.masses.view(-1,1)


    # 15. Compute gravity
    # with TimedBlock('compute gravity', use_cuda=True, device=config.device) as tb_gravity:
    with record_function("[warpSPH] - [deltaSPH - 15] - compute gravity"):
        gravity = computeGravity(currentState, config, schemeConfig, adjacency)
        dvdt_gravity = gravity

    # Revert boundary velocity
    # with TimedBlock('compute mDBC no-pen shift', use_cuda=True, device=config.device) as tb_nopenshift:
    with record_function("[warpSPH] - [deltaSPH - 16] - compute mDBC no-pen shift"):
        nopenshift = computeMdbcNoPenShift(currentState, config, schemeConfig, adjacency)
        dvdt_nopenshift = nopenshift / dt
    # currentState.velocities = currentVelocities

    # 16. build update
    # with TimedBlock('build update', use_cuda=True, device=config.device) as tb_update:
    # print(f'End of step: t={currentSystem.t:.6f}, dt={config.dt:.6f}, min density={currentState.densities.min().item():.6f}, max density={currentState.densities.max().item():.6f}, mean density={currentState.densities.mean().item():.6f}')
    # print(f'\tmin pressure={currentState.pressures.min().item():.6f}, max pressure={currentState.pressures.max().item():.6f}, mean pressure={currentState.pressures.mean().item():.6f}')
    # print(f'\tmin velocity={currentState.velocities.norm(dim=1).min().item():.6f}, max velocity={currentState.velocities.norm(dim=1).max().item():.6f}, mean velocity={currentState.velocities.norm(dim=1).mean().item():.6f}')
    # print(f'\tdvdt:')
    # print(f'\t[pressure]\tmin dvdt={dvdt_pressure.norm(dim=1).min().item():.6f}, max dvdt={dvdt_pressure.norm(dim=1).max().item():.6f}, mean dvdt={dvdt_pressure.norm(dim=1).mean().item():.6f}')
    # print(f'\t[forcing]\tmin dvdt={dvdt_forcing.norm(dim=1).min().item():.6f}, max dvdt={dvdt_forcing.norm(dim=1).max().item():.6f}, mean dvdt={dvdt_forcing.norm(dim=1).mean().item():.6f}')
    # print(f'\t[gravity]\tmin dvdt={dvdt_gravity.norm(dim=1).min().item():.6f}, max dvdt={dvdt_gravity.norm(dim=1).max().item():.6f}, mean dvdt={dvdt_gravity.norm(dim=1).mean().item():.6f}')
    # print(f'\t[nopenshift]\tmin dvdt={dvdt_nopenshift.norm(dim=1).min().item():.6f}, max dvdt={dvdt_nopenshift.norm(dim=1).max().item():.6f}, mean dvdt={dvdt_nopenshift.norm(dim=1).mean().item():.6f}')
    # print(f'\t[dissipation]\tmin dvdt={dvdt_diss.norm(dim=1).min().item():.6f}, max dvdt={dvdt_diss.norm(dim=1).max().item():.6f}, mean dvdt={dvdt_diss.norm(dim=1).mean().item():.6f}')
    # print(f'\tdrhodt:')
    # print(f'\t[drhodt]\tmin drhodt={drhodt.min().item():.6f}, max drhodt={drhodt.max().item():.6f}, mean drhodt={drhodt.mean().item():.6f}')    
    # print(f'\t[drhodt_diss]\tmin drhodt={drhodt_diss.min().item():.6f}, max drhodt={drhodt_diss.max().item():.6f}, mean drhodt={drhodt_diss.mean().item():.6f}')
    # print(f'------------------------------------------------------------')

    with record_function("[warpSPH] - [deltaSPH - 16] - build update"):
        update = WeaklyCompressibleSystemUpdate(
            dxdt = currentState.velocities.clone(),#+ dvdt_nopenshift  * dt,
            dvdt = dvdt_pressure + dvdt_forcing + dvdt_gravity + dvdt_diss+ dvdt_nopenshift,
            drhodt = drhodt + drhodt_diss,
            passive = torch.zeros(currentState.densities.shape, device=currentState.densities.device, dtype=torch.bool)
        )
    # update.drhodt = update.drhodt

    # 17. Enforce BCs on update
    # with TimedBlock('enforce updates', use_cuda=True, device=config.device) as tb_enforce:
    with record_function("[warpSPH] - [deltaSPH - 17] - enforce updates"):
        enforceUpdates(update, currentSystem, config.dt, currentSystem.t, config, schemeConfig)
        nonFluidMask = (currentState.kinds != 0).unsqueeze(-1)
        update.dxdt = torch.where(nonFluidMask, torch.zeros_like(update.dxdt), update.dxdt)
        update.dvdt = torch.where(nonFluidMask, torch.zeros_like(update.dvdt), update.dvdt)
        update.drhodt = torch.where(nonFluidMask.squeeze(-1), torch.zeros_like(update.drhodt), update.drhodt)

    # performanceDict = {
    #     'tb_adjacency': tb_adjacency,
    #     'tb_density': tb_density,
    #     'tb_mdbc': tb_mdbc,
    #     'tb_bcs': tb_bcs,
    #     'tb_eos': tb_eos,
    #     'tb_boundary_velocities': tb_boundary_velocities,
    #     'tb_surface': tb_surface,
    #     'tb_gradRho': tb_gradRho,
    #     'tb_gradRhoL': tb_gradRhoL,
    #     'tb_drhodt_diss': tb_drhodt_diss,
    #     'tb_dvdt_diss': tb_dvdt_diss,
    #     'tb_drhodt': tb_drhodt,
    #     'tb_dvdt': tb_dvdt,
    #     'tb_forcing': tb_forcing,
    #     'tb_gravity': tb_gravity,
    #     'tb_nopenshift': tb_nopenshift,
    #     'tb_update': tb_update,
    #     'tb_enforce': tb_enforce,
    # }

    # for key, tb in performanceDict.items():
    #     tb.cuda_ms = tb._start_event.elapsed_time(tb._end_event) if tb.use_cuda else None
    #     print(f"[{key}] CPU: {tb.cpu_ms:.3f} ms | CUDA: {tb.cuda_ms:.3f} ms, ratio {tb.cuda_ms / tb.cpu_ms if tb.cuda_ms is not None else 'N/A'}")

    return update, adjacency, currentState