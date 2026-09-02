"""The DFSPH (Divergence-Free SPH) step for the incompressible scheme:
adjacency, density (mDBC-corrected), boundary velocities/Dirichlet BCs,
free-surface detection, delta-SPH velocity diffusion, continuity `drhodt`,
forcing/gravity/mDBC no-penetration shift folded into an explicit `dvdt`,
then `solveDivergenceFree` projects that `dvdt` onto a divergence-free
pressure correction (`dvdt_pressure`). Several stretches are commented out
rather than removed (tracked separately in CLEANUP_PLAN.md), including an
alternate implicit-particle-shift path (`solveIncompressible`) and the
delta-SPH density-diffusion term. Whether the returned `drhodt` actually
drives the density update or is overridden depends on
`schemeConfig.solverConfig.integrateRho`: when it is false, density is instead
recomputed from scratch each step via `computeDensities`.
"""

from warpSPH.configurations import SimulationConfig, WeaklyCompressibleSPHConfig
from warpSPH.schemes.omniIncompressible import _solve
from warpSPH.systems import CompSPHSystem, WeaklyCompressibleSystemUpdate
from warpSPH.modules.boundaryConditions import computeForcing, enforceDirichlet, enforceUpdates
from warpSPH.modules.deltaSPH import computeVelocityDiffusion
from warpSPH.modules.density import computeDensities
from warpSPH.modules.gravity import computeGravity
from warpSPH.modules.incompressible import solveDivergenceFree, solveIncompressible
from warpSPH.modules.mdbc import (
    computeBoundaryVelocities, computeMdbcDensity, computeMdbcNoPenShift,
    computeMdbcPressure,
)
from warpSPH.configurations import BoundaryPressureMode, ShiftApplication, DensityEvolution, resolveDensityEvolution
from warpSPH.modules.momentum import computeMomentum
from warpSPH.modules.momentum.incompressible import computeMomentumIncompressible
from warpSPH.modules.surfaceDetection import detectFreeSurface
from warpSPHCore import SupportScheme, buildVerletList

import torch
from warpSPH.utils.timer import TimedBlock
from torch.profiler import profile, record_function, ProfilerActivity

from ..systems.incompressible import IncompressibleSystem, IncompressibleState, IncompressibleSystemUpdate

import numpy as np

__all__ = ['dfsph_step']


def dfsph_step(
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
            config.domain, verletScale = verletScale, supportMode = SupportScheme.SuperSymmetric,
            priorNeighborhood = currentSystem.adjacency,
            verbose = False)
        currentSystem.adjacency = adjacency

    # 2. Compute density if density is none
    # with TimedBlock('compute density', use_cuda=True, device=config.device) as tb_density:
    with record_function("[warpSPH] - [deltaSPH - 02] - compute density"):
        # `DensityEvolution.summation` (the default) re-sums here every step;
        # `continuity`/`hybrid` keep the density the integrator advanced with
        # `drhodt` instead. `finalize`'s own re-sum is controlled separately --
        # see `DensityEvolution`, and note that this branch alone used to be
        # `integrateRho`'s whole effect, which `finalize` then overwrote.
        # densityEvolution = resolveDensityEvolution(schemeConfig.solverConfig)
        # if currentState.densities is None or densityEvolution is DensityEvolution.summation:
            # print("Computing densities...")
        currentState.densities = computeDensities(currentState, config, schemeConfig, adjacency)
        currentState.densities[currentState.kinds==2] = 1.0 # reset boundary densities to 1.0

    print(f'step {currentSystem.t:.6g}: density stats: mean={currentState.densities.mean().item():.6g}, min={currentState.densities.min().item():.6g}, max={currentState.densities.max().item():.6g}')

    # print(f'step {currentSystem.t:.6g}: density stats: mean={currentState.densities.mean().item():.6g}, min={currentState.densities.min().item():.6g}, max={currentState.densities.max().item():.6g}')
    # if currentState.densities.max() > 1.01:
        # print(f"Warning: Density exceeds 1.01, max density: {currentState.densities.max().item():.6g}")
    # 3. mDBC density computation -- `BoundaryPressureMode.plain` skips it
    # (boundary particles get plain SPH summation density, like a fluid
    # particle, matching Part 2 Option a of DFSPH_IMPROVEMENT_PLAN.md);
    # both other modes keep the historical always-on extrapolation.
    boundaryPressureMode = schemeConfig.solverConfig.boundaryPressureMode
    # with TimedBlock('compute mDBC density', use_cuda=True, device=config.device) as tb_mdbc:
    # with record_function("[warpSPH] - [deltaSPH - 03] - compute mDBC density"):
    #     if boundaryPressureMode != BoundaryPressureMode.plain:
    #         currentState.densities = computeMdbcDensity(currentState, config, schemeConfig, adjacency)

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
    # with record_function("[warpSPH] - [deltaSPH - 05] - compute EOS"):
        # currentState.pressures = weaklyCompressibleEOS(currentState, schemeConfig)
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

    # #9. Compute gradRho and gradRhoL
    # # with TimedBlock('compute gradRho', use_cuda=True, device=config.device) as tb_gradRho:
    # with record_function("[warpSPH] - [deltaSPH - 09] - compute gradRho and gradRhoL"):
    #     if schemeConfig.diffusionParams.densityDiffusionTerm == DensityDiffusionScheme.denormalized or schemeConfig.diffusionParams.densityDiffusionTerm == DensityDiffusionScheme.denormalizedOnly:
    #         gradRho = computeGradRho(currentState, config, schemeConfig, adjacency)
    #     else:
    #         gradRho = None
    # # with TimedBlock('compute gradRhoL', use_cuda=True, device=config.device) as tb_gradRhoL:
    # with record_function("[warpSPH] - [deltaSPH - 09] - compute gradRhoL"):
    #     if schemeConfig.diffusionParams.densityDiffusionTerm == DensityDiffusionScheme.deltaSPH or schemeConfig.diffusionParams.densityDiffusionTerm == DensityDiffusionScheme.deltaOnly:
    #         gradRhoL = computeGradRhoL(currentState, config, schemeConfig, adjacency, L = renormalizationState_)
    #     else:
    #         gradRhoL = None


    # # 10. Compute drhodt_diss
    # # with TimedBlock('compute drhodt_diss', use_cuda=True, device=config.device) as tb_drhodt_diss:
    # with record_function("[warpSPH] - [deltaSPH - 10] - compute drhodt_diss"):
    #     drhodt_diss = computeDensityDiffusion(currentState, config, schemeConfig, adjacency, gradRho, gradRhoL)

    # 11. Compute dvdt_diss
    # with TimedBlock('compute dvdt_diss', use_cuda=True, device=config.device) as tb_dvdt_diss:
    with record_function("[warpSPH] - [deltaSPH - 11] - compute dvdt_diss"):
        dvdt_diss = computeVelocityDiffusion(currentState, config, schemeConfig, adjacency)
    
    # 12. Compute drhodt
    # with TimedBlock('compute drhodt', use_cuda=True, device=config.device) as tb_drhodt:
    with record_function("[warpSPH] - [deltaSPH - 12] - compute drhodt"):
        drhodt = computeMomentum(currentState, config, schemeConfig, adjacency)

    # 14. Apply forcing
    # with TimedBlock('compute forcing', use_cuda=True, device=config.device) as tb_forcing:
    with record_function("[warpSPH] - [deltaSPH - 14] - compute forcing"):
        forcing = computeForcing(currentSystem, config.dt, currentSystem.t, config, schemeConfig)*0
        dvdt = forcing / currentState.masses.view(-1,1)


    # 15. Compute gravity
    # with TimedBlock('compute gravity', use_cuda=True, device=config.device) as tb_gravity:
    with record_function("[warpSPH] - [deltaSPH - 15] - compute gravity"):
        gravity = computeGravity(currentState, config, schemeConfig, adjacency)
        gravity[currentState.kinds != 0] = 0.0
        dvdt += gravity

    # Revert boundary velocity
    # with TimedBlock('compute mDBC no-pen shift', use_cuda=True, device=config.device) as tb_nopenshift:
    # with record_function("[warpSPH] - [deltaSPH - 16] - compute mDBC no-pen shift"):
    #     # Experimental A/B toggle (DFSPH_IMPROVEMENT_PLAN.md) -- the original
    #     # DFSPH paper has no such term and relies on the pressure projection
    #     # alone to prevent penetration; default True preserves this scheme's
    #     # historical always-on behavior pending that comparison's outcome.
    #     if schemeConfig.solverConfig.mdbcNoPenetrationShift:
    #         nopenshift = computeMdbcNoPenShift(currentState, config, schemeConfig, adjacency)
    #         dvdt += nopenshift / dt
    # currentState.velocities = currentVelocities

    # First we project the velocity field to be divergence free using the DFSph solver
    # This is the standard incompressible approach
    # However, this can lead to issues with particle disorder and clustering 
    currentState.pressures = torch.zeros_like(currentState.densities) if currentState.pressures is None else currentState.pressures
    # dvdt_pressure, pressure, errors, pressures = solveDivergenceFree(
    #     particles = currentState,
    #     config = config,
    #     schemeConfig = schemeConfig,
    #     adjacency = adjacency,
    #     dvdt = dvdt + dvdt_diss,
    #     dt = dt
    # )
    accel = dvdt + dvdt_diss 
    accel[currentState.kinds != 0] = 0.0
    vEnter = currentState.velocities + dt * accel
    fluid = currentState.kinds == 0
    rho0 = schemeConfig.fluid.restDensity
    a_p_div, _, nDiv, errDiv = _solve(
        currentState, config, schemeConfig, adjacency, fluid=fluid, rho0=rho0,
        vEnter=vEnter, warmStart=torch.zeros_like(currentState.densities), dt=dt,
        mode='divergence', minIters=2,
        maxIters=32, tol=1e-2)
    dvdt_pressure = a_p_div
    # print(f'step {currentSystem.t:.6g}: pressure stats: mean={pressure.mean().item():.6g}, min={pressure.min().item():.6g}, max={pressure.max().item():.6g}')
    # currentState.pressures = pressure
    # if boundaryPressureMode == BoundaryPressureMode.mdbcMlsPressure:
        # Project this step's fluid pressure solution onto boundary
        # particles (Part 2 Option c) -- see `computeMdbcPressure`'s own
        # docstring for why this runs after, not inside, the solve above.
        # with record_function("[warpSPH] - [deltaSPH - 03b] - compute mDBC pressure"):
            # currentState.pressures = computeMdbcPressure(currentState, config, schemeConfig, adjacency)

    # `ShiftApplication.inStepVelocity`: run the constant-density solve here,
    # inside the step, and fold its correction into the same `dvdt` the
    # integrator advects with -- where DFSPH proper puts it. The placement is
    # the point: applied here it is visible to the *next* step's
    # divergence-free projection, which removes whatever part of it was not
    # divergence-free. `IncompressibleSystem.finalize`'s
    # `positionAndVelocity` adds the same correction *after* the integrator,
    # where nothing ever cleans it up -- and that uncorrected remainder is the
    # dissipation it shows on `tgv`. See `ShiftApplication`'s docstring.
    # dvdt_inStep = None
    # if schemeConfig.solverConfig.shiftApplication is ShiftApplication.inStepVelocity:
    #     # `warmStartConstantDensity`: seed this solve from the previous step's
    #     # constant-density pressure (carried on the unused `soundspeeds`) rather
    #     # than the cold zero start. MEASURED NEGATIVE (Part 24): on
    #     # `hydrostaticColumn` a warm start on `solveIncompressible` *as it
    #     # stands* -- two-sided source, 0.9 rhoStar clamp, linear operator,
    # nonNegativeClamp gauge -- NaNs by step 13 (the linear operator's wall
    # truncation inflates kappa, and the warm start feeds it). It only helps
    # paired with the one-sided / per-iteration-resummed inner loop the
    # `dfsphReference` scheme uses. Kept as an off-by-default hook.
    # _cdWarm = None
    # _cdWS = getattr(schemeConfig.solverConfig, 'warmStartConstantDensity', False)
    # if _cdWS:
    #     _s = getattr(currentState, 'soundspeeds', None)
    #     if _s is not None and _s.shape == currentState.densities.shape:
    #         _cdWarm = _s
    # dvdt_inStep, _p_incomp, _e_incomp, _ps_incomp = solveIncompressible(
    #     particles = currentState,
    #     config = config,
    #     schemeConfig = schemeConfig,
    #     adjacency = adjacency,
    #     dvdt = dvdt + dvdt_diss + dvdt_pressure*0.0,
    #     dt = dt,
    #     warmStartPressure = None,
    # )
    # print(f'step {currentSystem.t:.6g}: incompressible pressure stats: mean={_p_incomp.mean().item():.6g}, min={_p_incomp.min().item():.6g}, max={_p_incomp.max().item():.6g}')
    # if _cdWS:
        # currentState.soundspeeds = _p_incomp.detach()


    # --- 4. densitySolve() -- min 3 / max 256, warm start 0.5 * p_prior ---
    accel = dvdt + dvdt_diss + dvdt_pressure
    accel[currentState.kinds != 0] = 0.0
    vEnter = currentState.velocities + dt * accel
    fluid = currentState.kinds == 0
    rho0 = schemeConfig.fluid.restDensity
    a_p_rho, pRho, nRho, errRho = _solve(
        currentState, config, schemeConfig, adjacency, fluid=fluid, rho0=rho0,
        vEnter=vEnter, warmStart=0.5 * torch.zeros_like(currentState.densities), dt=dt, mode='density',
        minIters=3, maxIters=256,
        tol=1e-3)
    dvdt_inStep = a_p_rho

    # To resolve the issues with particle disorder and clustering, we can also solve for the incompressible pressure using the Incompressible SPH solver
    # This effectively acts as a particle shifting term that helps to maintain particle order and prevent clustering
    # Instead of being explicit, e.g., as in delta SPH, this is an implicit particle shift
    # dvdt_incomp, pressure_incomp, errors_incomp, pressures_incomp = solveIncompressible(
    #     particles = currentState,
    #     config = config,
    #     schemeConfig = schemeConfig,
    #     adjacency = adjacency,
    #     dvdt = dvdt + dvdt_diss + dvdt_pressure,
    #     dt = dt
    # )
    # The shift is directly applied to the particle positions
    # As the timestep for our tests is small the shift is small and does not significantly affect the velocity field
    # vPrime = currentState.velocities + dt * (dvdt + dvdt_diss + dvdt_pressure)

    # gradV = -warpOperation(
    #     queryParticles=currentState,
    #     operationProperties=OperationProperties(
    #         kernel = config.kernel,
    #         operation = WarpOperation.Gradient,
    #         gradientMode = GradientScheme.Difference,
    #         supportMode = SupportScheme.Scatter,
    #     ),
    #     queryValues = vPrime,
    #     domain = config.domain,
    #     adjacency=adjacency,
    # )
    # vCorrection = torch.einsum('nij, ni->nj', gradV, dt * dvdt_incomp)
    # dvdt_correction = vCorrection / dt
    # dvdt += dvdt_correction


    # Under `summation` the carried density is discarded and re-summed next
    # step, so it does not matter that `drhodt` above was evaluated on
    # `currentState.velocities` -- the velocity *before* the divergence-free
    # projection. Under `continuity`/`hybrid` it matters completely: the
    # particles are advected with the projected velocity, whose divergence the
    # solve just drove to ~0, so integrating the *unprojected* divergence feeds
    # the carried density exactly the error the solver was there to remove,
    # every step. Re-evaluate it on the velocity the integrator will actually
    # use, with the same operator the solvers form `rhoStar` from.
    # if densityEvolution is not DensityEvolution.summation:
    #     with record_function("[warpSPH] - [deltaSPH - 12b] - recompute drhodt (projected)"):
    #         advection = currentState.velocities + dt * (
    #             dvdt + dvdt_diss + dvdt_pressure
    #             + (dvdt_inStep if dvdt_inStep is not None else 0.0))
    #         drhodt = computeMomentumIncompressible(
    #             currentState = currentState, config = config, schemeConfig = schemeConfig,
    #             adjacency = adjacency, advectionVelocities = advection)

    # 16. build update
    # with TimedBlock('build update', use_cuda=True, device=config.device) as tb_update:
    with record_function("[warpSPH] - [deltaSPH - 16] - build update"):
        update = WeaklyCompressibleSystemUpdate(
            dxdt = currentState.velocities.clone(),# + dt * dvdt_incomp,
            dvdt = (dvdt + dvdt_diss + dvdt_pressure + dvdt_inStep),
                    # + (dvdt_inStep if dvdt_inStep is not None else 0.0)),
            drhodt = torch.zeros_like(currentState.densities),# + drhodt_diss,
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

    return update, adjacency, currentState, ([-1],[0])#, (errors_incomp, pressures_incomp)