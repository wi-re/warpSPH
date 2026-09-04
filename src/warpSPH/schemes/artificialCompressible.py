"""The artificial-compressibility SPH step (De Courcy et al. 2024; see
`ACSPH_PLAN.md` for the full equation inventory and this file's roadmap).

    STATUS -- SCAFFOLD. `artificialCompressible_step` currently returns a
    **zero update**: it builds the neighbourhood, enforces the boundary
    conditions and runs free-surface detection, then advances nothing. It
    exists so the scheme family is wired end to end (enum, state, system,
    config, round-trip, registration, timestep dispatch) and testable before
    the physics lands. `PHYSICS_IMPLEMENTED` is `False` and every entry warns
    once; a case run against it will simply not move. Plan step 5 replaces the
    marked block with the dual-time driver.

Why the step owns the whole real advance
----------------------------------------
ACSPH is not a "compute a derivative, hand it to an integrator" scheme. Each
real step runs a pseudo-time loop to convergence, and the integrator outside it
has nothing left to do. Rather than teach `warpSPHIntegrators` a `dualTime`
scheme -- coupling a general library to one solver -- the step returns an
**exact delta**:

    dxdt = (x^{n+1} - x^n)/dt,  dvdt = (v^{n+1} - v^n)/dt,  dpdt = (p^{n+1} - p^n)/dt

Forward Euler on an exact delta is the identity, so the runner reproduces the
converged state byte for byte with no framework change. That contract only
holds for a **one-stage, one-evaluation** integrator: under RK2 the solve would
run twice per step and the results would be *blended*, which is wrong and not
visibly wrong. `validateIntegrationScheme` therefore refuses anything else,
loudly, at step entry -- `cases/dambreak.py` documents the same class of trap
for `divergenceFree`/`semiImplicitEuler` and notes that nothing enforces it;
here it is enforced.

What the dual-time driver will do (plan Sec. 4.1)
-------------------------------------------------
    for m in 0..maxPseudoIterations:
        u0 = u                                  # frozen for the BDF source
        D^p = pressureSmoothing(u0)             # frozen across RK stages
        for s in 1..rkStages:
            r  = spatial residual at u^{s-1}    # Eqs. 23, 25, 26
            r* = (r - I_c (alpha_t u0 + beta_t u^n + gamma_t u^{n-1})) / alpha_PI
            u^s = u0 + a_s dtau r*
        u = accumulate(b_s)
        if epsilonV(tilde v) < target: break
    apply the shifting displacement (Eq. 58) + its BDF correction (Eq. 59)
    system.rollHistory(dt)

`I_c = diag{0, 1, 1}` -- the continuity equation has no real-time derivative,
which is exactly what makes `r* -> 0` enforce `div v = 0` *at* time level n+1.
The BDF history and `bdfCoefficients` already live on
`systems/artificialCompressible.py`.
"""

from typing import Any

import torch
from torch.profiler import record_function
from warpSPHCore import SupportScheme, buildVerletList
from warpSPHIntegrators.integration import IntegrationSchemeType

from ..configurations import ArtificialCompressibleSPHConfig, SimulationConfig
from ..modules.boundaryConditions import enforceDirichlet, enforceUpdates
from ..modules.mdbc import computeBoundaryVelocities
from ..modules.surfaceDetection import detectFreeSurface
from ..systems.artificialCompressible import (ArtificialCompressibleSystem,
                                              ArtificialCompressibleSystemUpdate)

__all__ = ['artificialCompressible_step', 'validateIntegrationScheme',
           'PHYSICS_IMPLEMENTED']

#: Flipped to True by plan step 5, when the dual-time driver replaces the
#: zero update below. Read it rather than inferring from behaviour: a scheme
#: that advances nothing looks like a converged one.
PHYSICS_IMPLEMENTED = False

#: The only integrator whose action on an exact delta is the identity. Kept as
#: a set so `explicitEuler` (an alias member) is accepted too.
_EXACT_DELTA_INTEGRATORS = {IntegrationSchemeType.forwardEuler,
                            IntegrationSchemeType.explicitEuler}

_warned = False


def validateIntegrationScheme(config: SimulationConfig) -> None:
    """Refuse any integrator that would evaluate the step more than once, or
    scale its result. See this module's docstring for why.

    Raised, not warned: a multi-stage integrator here does not fail visibly --
    it runs the whole dual-time solve twice and blends two converged states,
    producing a plausible-looking but wrong answer.
    """
    scheme = getattr(config, 'integrationScheme', None)
    if scheme in _EXACT_DELTA_INTEGRATORS:
        return
    name = getattr(scheme, 'name', scheme)
    raise ValueError(
        f"artificialCompressible requires integrationScheme=forwardEuler, got "
        f"{name!r}. The step returns an exact per-step delta (dx/dt = "
        f"(x^{{n+1}} - x^n)/dt), which only a single-evaluation integrator "
        f"applies unchanged; a multi-stage one would run the dual-time solve "
        f"once per stage and blend the results. Set "
        f"`config.integrationScheme = IntegrationSchemeType.forwardEuler` "
        f"(CaseSpec: `--integrationScheme forwardEuler`).")


def artificialCompressible_step(
    system: ArtificialCompressibleSystem,
    dt: float,
    config: SimulationConfig,
    schemeConfig: ArtificialCompressibleSPHConfig,
    verbose: bool = False,
):
    global _warned
    validateIntegrationScheme(config)
    if not PHYSICS_IMPLEMENTED and not _warned:
        _warned = True
        print("[warpSPH] artificialCompressible: SCAFFOLD ONLY -- the dual-time "
              "driver is not implemented yet, this step advances nothing. See "
              "schemes/artificialCompressible.py and ACSPH_PLAN.md step 5.")

    currentSystem = system
    currentState = currentSystem.state
    adjacency = currentSystem.adjacency

    with record_function("[warpSPH] - [acsph - 01] - compute adjacency"):
        adjacency = buildVerletList(
            currentState, config.domain, verletScale=config.verletScale,
            supportMode=SupportScheme.SuperSymmetric,
            priorNeighborhood=adjacency, verbose=False)
        currentSystem.adjacency = adjacency

    with record_function("[warpSPH] - [acsph - 02] - boundary velocities"):
        currentState.velocities = computeBoundaryVelocities(
            currentState, config, schemeConfig, adjacency)

    with record_function("[warpSPH] - [acsph - 03] - enforce BCs"):
        enforceDirichlet(currentSystem, currentSystem.t, config.dt, config, schemeConfig)

    with record_function("[warpSPH] - [acsph - 04] - surface detection"):
        fs, fsm, n, renormalizationState, lMin = detectFreeSurface(
            currentState, config, schemeConfig, schemeConfig.surfaceDetectionConfig,
            adjacency, returnNormals=True)
        currentState.surfaceIndicators = (fsm > 0.5).to(torch.int32)
        currentState.surfaceNormals = n
        currentState.surfaceLambdas = lMin

    # ---------------------------------------------------------------------
    # PLAN STEP 5 GOES HERE: the dual-time driver. Everything above is the
    # per-real-step setup it needs (neighbourhood, enforced BCs, the free-
    # surface sets `F`/`V` and the renormalisation matrices `L`); everything
    # below is the exact-delta hand-off, which does not change.
    # ---------------------------------------------------------------------
    with record_function("[warpSPH] - [acsph - 05] - build update"):
        update = ArtificialCompressibleSystemUpdate(
            dxdt=torch.zeros_like(currentState.positions),
            dvdt=torch.zeros_like(currentState.velocities),
            dpdt=torch.zeros_like(currentState.pressures),
            passive=torch.zeros(currentState.pressures.shape,
                                device=currentState.pressures.device, dtype=torch.bool),
        )

    with record_function("[warpSPH] - [acsph - 06] - enforce updates"):
        enforceUpdates(update, currentSystem, config.dt, currentSystem.t, config,
                       schemeConfig)
        nonFluid = (currentState.kinds != 0).unsqueeze(-1)
        update.dxdt = torch.where(nonFluid, torch.zeros_like(update.dxdt), update.dxdt)
        update.dvdt = torch.where(nonFluid, torch.zeros_like(update.dvdt), update.dvdt)
        update.dpdt = torch.where(nonFluid.squeeze(-1), torch.zeros_like(update.dpdt),
                                  update.dpdt)

    return update, adjacency, currentState
