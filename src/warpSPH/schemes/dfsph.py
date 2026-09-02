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
from warpSPH.schemes.omniIncompressible import _solve, _xsphFilter
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

#: How the constant-density solve's `(1 - rho/rho0)` source is treated at the
#: kernel-truncated free surface (see `omniIncompressible._solve`). `'full'` is
#: the omniSPH form; `'clamp'` / `'mask'` / `'shepard'` stop the solve chasing
#: an unreachable rho0 in the surface skin. MEASURED (hydrostaticColumn nx=64,
#: 500 steps): all three just trade the surface error between the density axis
#: and the velocity axis -- `shepard` lifts `densityP05` 0.947 -> 0.972 but
#: pushes `|v|` 0.066 -> 0.20 and `embeddedMinDensity` 0.999 -> 0.93; `clamp`
#: is a wash. Kept as an off-by-default hook; `'full'` is the graded default.
SURFACE_SOURCE = 'full'

#: Which routine runs the divergence-free projection. Both use the same
#: operator pair (`_divergence` difference form + `computePressureAccelIISPH`);
#: they differ in *solver quality*, and that is where the `tgv` energy shows up.
#:   'omni' -- `omniIncompressible._solve` (mode='divergence'): under-relaxed
#:            Jacobi, `OMEGA = 0.3`, cold start, hard 2-iteration cap, no gauge
#:            re-centre. Two sweeps from zero do not fully project `vEnter`, and
#:            the semi-implicit integrator then does `+work` with the residual
#:            divergence over the cold-start transient -- `tgv` fluid KE grows
#:            ~6% in the first ~15 steps (the `test_tgvKineticEnergy*` failure).
#:            Iterating it to `tol=1e-5` still leaves ~4.5% (the under-relaxed
#:            Jacobi plateaus on this operator).
#:   'vdps' -- `modules/incompressible.solveDivergenceFree`: optimal-step
#:            (`omega_k = <r,q>/<q,q>`, the exact 1-D minimiser), 0.75x warm
#:            start, per-iterate mean-zero gauge. A genuinely convergent
#:            projection: `tgv` KE is flat to ~0.8% over 20 steps. BUT the
#:            optimal step + mean-centre is the spurious-force move
#:            DFSPH_FINDINGS.md 1.5 / Part 26 forbid at a free surface -- it
#:            cannot sustain a body force and `hydrostaticColumn` blows up
#:            (|v| -> 5, slope ratio 0.24), and with the position-shift path
#:            removed from `IncompressibleSystem.finalize` it also no longer
#:            *decays* `tgv` at the analytic rate. Kept for the A/B only.
#: Default 'omni'. (Note: 'omni' was thought to cost the periodic KE tests,
#: but §1.17 / Part 47 traced that to a *missing particle shift*, not this
#: projection -- restoring the shift, gated by `INSTEP_CD` / `_RESTORE_PS_SHIFT
#: = 'auto'`, makes the suite green with 'omni'.)
DIVERGENCE_SOLVER = 'omni'

#: Whether the in-step constant-density `_solve` result is folded into `dvdt`
#: (`inStepVelocity` semantics). It is the *only* support a body-force column
#: has (fall-and-push-back), but its unprojected impulse is exactly what
#: destabilises `tgv` / the bounded box, and applying it *and* the VD+PS
#: position shift double-counts the density error (tgv KE grows, column NaNs).
#:   `'auto'` (default) -- on when `schemeConfig.gravityConfig.active`, i.e.
#:                         `hydrostaticColumn` keeps it, everything else drops
#:                         it and takes the VD+PS shift instead
#:                         (`incompressible._RESTORE_PS_SHIFT = 'auto'`).
#:   `True` / `False`   -- force it. See DFSPH_FINDINGS.md 1.17.
INSTEP_CD = 'auto'

#: TEMP (ordering experiment): order of the two `_solve` passes when
#: `DIVERGENCE_SOLVER == 'omni'`.
#:   'div_then_cd' -- as shipped: divergence projection first (sees
#:                    `dvdt + dvdt_diss`), then the constant-density solve
#:                    (sees `... + dvdt_pressure`). The CD impulse is only
#:                    projected by *next* step's divergence pass.
#:   'cd_then_div' -- constant-density first (sees `dvdt + dvdt_diss`), then
#:                    divergence (sees `... + dvdt_inStep`), so the divergence
#:                    pass cleans the CD impulse's non-div-free part *this*
#:                    step. omniSPH's own loop is `div` then `density`.
SOLVE_ORDER = 'div_then_cd'

#: Post-solve XSPH velocity filter (omniSPH `SPHSimulation::XSPH`, folded into
#: `dvdt` as `scale * sum_j c_j V_j (v_j - v_i) W_ij / dt`, `c` from
#: `omniIncompressible.XSPH_FLUID`). The `hydrostaticColumn` residual `|v|` is
#: an *undamped inviscid* free-surface limit cycle (it neither grows nor
#: decays); the scheme carries no viscosity, so a light velocity smoother is
#: the only thing that can decay it. `scale = 1.0` == omniSPH's coefficient;
#: `0.0` reproduces the historical no-filter behaviour.
XSPH_SCALE = 0.0

#: TEMP (ablation experiment): `None`, or `(amplitude, period)` -- multiplies
#: the gravity acceleration by `amplitude * sin(2*pi * t / period)` each step,
#: so a periodic no-wall no-free-surface box can be put under an *oscillating*
#: body force. Isolates "does the position shift break under a body force"
#: (Part 23) from "does it break at a free surface". `None` == inert.
GRAVITY_OSC = None


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
        if GRAVITY_OSC is not None:
            _amp, _per = GRAVITY_OSC
            gravity = gravity * (_amp * np.sin(2.0 * np.pi * currentSystem.t / _per))
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
    fluid = currentState.kinds == 0
    rho0 = schemeConfig.fluid.restDensity
    _zeros = torch.zeros_like(currentState.densities)
    # `'auto'`: fold the in-step CD into the velocity only for a body-force
    # case (`hydrostaticColumn`), where the fall-and-push-back cycle is the
    # column's only support -- and leave it off elsewhere, where that
    # unprojected impulse destabilises `tgv` / the bounded box and the VD+PS
    # position shift regularises the distribution instead (FINDINGS 1.17).
    _instepCD = (schemeConfig.gravityConfig.active if INSTEP_CD == 'auto'
                 else INSTEP_CD)

    def _omniPass(mode, priorAccel, warmStart, minIters, maxIters, tol,
                  surfaceSource='full'):
        """One `omniIncompressible._solve` pass: build `vEnter` from
        `dvdt + dvdt_diss + priorAccel`, refresh the Dirichlet wall velocity on
        it, solve, restore the velocity."""
        accel = dvdt + dvdt_diss + priorAccel
        accel[currentState.kinds != 0] = 0.0
        vEnter = currentState.velocities + dt * accel
        vSaved = currentState.velocities.clone()
        currentState.velocities = vEnter
        vEnter = computeBoundaryVelocities(currentState, config, schemeConfig, adjacency)
        currentState.velocities = vSaved
        return _solve(currentState, config, schemeConfig, adjacency, fluid=fluid,
                      rho0=rho0, vEnter=vEnter, warmStart=warmStart, dt=dt,
                      mode=mode, minIters=minIters, maxIters=maxIters, tol=tol,
                      surfaceSource=surfaceSource)

    if DIVERGENCE_SOLVER == 'vdps':
        dvdt_pressure, pressure, errDiv, _ = solveDivergenceFree(
            particles=currentState, config=config, schemeConfig=schemeConfig,
            adjacency=adjacency, dvdt=dvdt + dvdt_diss, dt=dt)
        currentState.pressures = pressure
        a_p_rho, pRho, nRho, errRho = _omniPass(
            'density', dvdt_pressure, 0.5 * _zeros, 3, 256, 1e-3, SURFACE_SOURCE)
        dvdt_inStep = a_p_rho if _instepCD else torch.zeros_like(a_p_rho)
    elif DIVERGENCE_SOLVER == 'omni':
        if SOLVE_ORDER == 'cd_then_div':
            a_p_rho, pRho, nRho, errRho = _omniPass(
                'density', 0.0, 0.5 * _zeros, 3, 256, 1e-3, SURFACE_SOURCE)
            dvdt_inStep = a_p_rho if _instepCD else torch.zeros_like(a_p_rho)
            a_p_div, _, nDiv, errDiv = _omniPass(
                'divergence', dvdt_inStep, _zeros, 2, 32, 1e-2)
            dvdt_pressure = a_p_div
        else:  # 'div_then_cd' (shipped)
            a_p_div, _, nDiv, errDiv = _omniPass(
                'divergence', 0.0, _zeros, 2, 32, 1e-2)
            dvdt_pressure = a_p_div
            a_p_rho, pRho, nRho, errRho = _omniPass(
                'density', dvdt_pressure, 0.5 * _zeros, 3, 256, 1e-3, SURFACE_SOURCE)
            dvdt_inStep = a_p_rho if _instepCD else torch.zeros_like(a_p_rho)
    else:
        raise ValueError(f'Unknown DIVERGENCE_SOLVER: {DIVERGENCE_SOLVER!r}')
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


    # (divergence + constant-density solves run above, ordered by SOLVE_ORDER)

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

    # Post-solve XSPH velocity smoother (see `XSPH_SCALE`). The only sink for
    # the undamped free-surface limit cycle in this otherwise-inviscid scheme.
    # A case may override the module default via `schemeConfig.xsphFilterScale`
    # (e.g. `hydrostaticColumn`'s `xsphScale` param); everything else is off.
    xsphScale = getattr(schemeConfig, 'xsphFilterScale', XSPH_SCALE)
    dvdt_xsph = torch.zeros_like(dvdt)
    if xsphScale:
        dvdt_xsph = xsphScale * _xsphFilter(
            currentState, config, adjacency, currentState.kinds == 0) / dt
        dvdt_xsph[currentState.kinds != 0] = 0.0

    # 16. build update
    # with TimedBlock('build update', use_cuda=True, device=config.device) as tb_update:
    with record_function("[warpSPH] - [deltaSPH - 16] - build update"):
        update = WeaklyCompressibleSystemUpdate(
            dxdt = currentState.velocities.clone(),# + dt * dvdt_incomp,
            dvdt = (dvdt + dvdt_diss + dvdt_pressure + dvdt_inStep + dvdt_xsph),
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