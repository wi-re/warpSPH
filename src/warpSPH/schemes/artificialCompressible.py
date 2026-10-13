"""The artificial-compressibility SPH step (De Courcy et al. 2024; see
`ACSPH_PLAN.md` for the full equation inventory and this file's roadmap).

Each real step runs a **pseudo-time loop to steady state**, and the integrator
outside it has nothing left to do. Rather than teach `warpSPHIntegrators` a
`dualTime` scheme -- coupling a general library to one solver -- the step
returns an **exact delta**:

    dxdt = (x^{n+1} - x^n)/dt,  dvdt = (v^{n+1} - v^n)/dt,  dpdt = (p^{n+1} - p^n)/dt

Forward Euler on an exact delta is the identity, so the runner reproduces the
converged state byte for byte with no framework change. That contract only
holds for a **one-stage, one-evaluation** integrator: under RK2 the solve would
run twice per step and the results would be *blended*, which is wrong and not
visibly wrong. `ArtificialCompressibleSPHConfig.validate` (called once per run by the runner) therefore refuses anything else,
loudly, at step entry -- `cases/freeSurface/dambreak.py` documents the same class of trap
for `divergenceFree`/`semiImplicitEuler` and notes that nothing enforces it;
here it is enforced.

The loop (Eqs. 38-48)
---------------------
    for m in 0..maxPseudoIterations:
        u0 = u                                  # frozen for the BDF source
        D^p = pressureSmoothing(u0)             # frozen across RK stages
        for s in 1..rkStages:
            r  = spatial residual at u^{s-1}    # Eqs. 23, 25, 26
            r* = (r - I_c (alpha_t u0 + beta_t u^n + gamma_t u^{n-1})) / alpha_PI
            u^s = u0 + dtau sum_l a_{s,l} r*_l
        u = u0 + dtau sum_l b_l r*_l
        if eps_v(tilde v) < target: break
    roll the BDF history

`I_c = diag{0, 1, 1}` -- the continuity equation has no real-time derivative,
which is exactly what makes `r* -> 0` enforce `div v = 0` *at* time level n+1.

Three choices worth naming, all from `ACSPH_PLAN.md` Part 5:

- **The RK sweep is the general explicit Butcher form**, taken from
  `warpSPHIntegrators.getButcherTableau` ('midpoint' / 'SSPRK3' / 'RK4' are
  Fig. 1 of the paper verbatim). Eq. (40) as printed is the Jameson low-storage
  form, which can only represent a tableau whose `A` is sub-diagonal and whose
  `b` is its last stage row -- true of the RK2 midpoint tableau, false of both
  SSPRK3 and RK4 as Fig. 1 prints them. The general form reproduces Fig. 1
  exactly and degenerates to Eq. (40) for RK2, which is the recommended
  operating point anyway (Sec. 4.3: higher order buys no accuracy here, the
  BDF2 sets it). Plan Sec. 5.2; **ask the authors which the CUDA code does.**
- **`k2 D^p`, not `k2 h D^p`.** Eq. (30) prints the extra `h`; Eqs. (23), (51)
  and (54) do not, and dimensional analysis says they are right. Plan Sec. 5.3.
- **The diffusion is frozen across RK stages but re-evaluated every dual-time
  iteration**, which is what the paper says ("the diffusive terms are evaluated
  at each dual-time iteration and cannot be fixed without loss of stability").

Still to land (see `ACSPH_PLAN.md` Part 8)
------------------------------------------
step 6 the Eq. (46) timestep and its growth clamp; step 7 the Michel et al.
shifting and Eqs. (58)-(59); step 8 AC-4 and AC-JST; step 10 the `tilde v`
advective corrections (Eqs. 27-31), internal shifting, `k3`. Each is gated by
its config field and raises rather than silently no-opping.
"""

from warpSPHCore import SupportScheme, buildVerletList
from warpSPHCore.profiling import record_function

from ..configurations import ArtificialCompressibleSPHConfig, SimulationConfig
from ..modules.artificialCompressible import (
    acParameters, bdfHistory, bodyForces, contactMasks, detectSurface, exactDeltaUpdate,
    makeDualTimeProblem, makeWallBounce, solveDualTime, zeroNonFluidRows)
from ..modules.boundaryConditions import enforceDirichlet, enforceUpdates
from ..modules.mdbc import computeBoundaryVelocities
from ..systems.artificialCompressible import ArtificialCompressibleSystem

__all__ = ['artificialCompressible_step', 'PHYSICS_IMPLEMENTED']

PHYSICS_IMPLEMENTED = True


def artificialCompressible_step(
    system: ArtificialCompressibleSystem,
    dt: float,
    config: SimulationConfig,
    schemeConfig: ArtificialCompressibleSPHConfig,
    verbose: bool = False,
):
    state = system.state

    with record_function("[warpSPH] - [acsph - 01] - compute adjacency"):
        adjacency = buildVerletList(
            state, config.domain, verletScale=config.verletScale,
            supportMode=SupportScheme.SuperSymmetric,
            priorNeighborhood=system.adjacency, verbose=False)
        system.adjacency = adjacency

    with record_function("[warpSPH] - [acsph - 02] - boundary velocities"):
        state.velocities = computeBoundaryVelocities(state, config, schemeConfig, adjacency)

    with record_function("[warpSPH] - [acsph - 03] - enforce BCs"):
        enforceDirichlet(system, system.t, config.dt, config, schemeConfig)

    with record_function("[warpSPH] - [acsph - 04] - surface detection"):
        surface = detectSurface(state, config, schemeConfig, adjacency)

    with record_function("[warpSPH] - [acsph - 05] - step parameters and BDF coefficients"):
        dtau, beta, k1, k2, nu = acParameters(state, config, schemeConfig, dt)
        alphaT, betaT, gammaT, bdfOrder = system.bdfCoefficients(dt)

    with record_function("[warpSPH] - [acsph - 06] - forcing"):
        bodyForce, wallBodyForce = bodyForces(system, dt, config, schemeConfig, adjacency)

    with record_function("[warpSPH] - [acsph - 07] - contact masks and wall bounce setup"):
        contact = contactMasks(state, config, schemeConfig, adjacency, surface)
        bounce = makeWallBounce(state, adjacency, config, schemeConfig.acParams)

    with record_function("[warpSPH] - [acsph - 08] - dual-time problem setup"):
        problem = makeDualTimeProblem(
            state, config, schemeConfig, adjacency, dtau=dtau, k1=k1, k2=k2, nu=nu,
            bdf=(alphaT, betaT, gammaT), history=bdfHistory(system, state),
            bodyForce=bodyForce, wallBodyForce=wallBodyForce, surface=surface,
            contact=contact, bounce=bounce)

    with record_function("[warpSPH] - [acsph - 09] - dual-time loop"):
        result = solveDualTime(problem, state.positions, state.velocities, state.pressures, verbose)

    if verbose:
        print(f"[acsph] t={system.t:.6g} dt={dt:.4g} BDF{bdfOrder} "
              f"{result.iterations} pseudo-iterations, eps_v={result.epsilonV:.3f} "
              f"(target {schemeConfig.acParams.epsilonV})")

    with record_function("[warpSPH] - [acsph - 10] - build update"):
        update = exactDeltaUpdate(result, state, dt, bdfOrder)

    with record_function("[warpSPH] - [acsph - 11] - enforce updates"):
        enforceUpdates(update, system, config.dt, system.t, config, schemeConfig)
        zeroNonFluidRows(update, state)

    return update, adjacency, state
