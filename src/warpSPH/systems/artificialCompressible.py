"""State/update/system triad for artificial-compressibility SPH
(`schemes/artificialCompressible.py`, De Courcy et al. 2024; `ACSPH_PLAN.md`).

Two structural differences from `WeaklyCompressibleState`, and they are the
whole point of the scheme:

- **`pressures` is an integrated field** (`integrated('dpdt')`). ACSPH has no
  equation of state; the pressure obeys its own evolution equation
  `Dp/Dtau = -k1 rho div v + k2 D^p` (Eq. 23), driven to steady state in
  pseudo-time inside each real step.
- **`densities` is constant.** The scheme is density-invariant by construction:
  `rho == rho0` for all time, so `V_j = m_j/rho0` and the symmetric
  `(p_i + p_j)` pressure gradient is exactly the `sum m_j (p_i/rho_i^2 +
  p_j/rho_j^2) gradW` form (paper Sec. 3.1.2). Nothing integrates it.

Everything else -- free-surface fields, ghost bookkeeping, rigid bodies --
mirrors `WeaklyCompressibleState` so that `rigidBody.update.
updateBodyParticlesWCSPH` and the mDBC/surface-detection modules work on it
unchanged (they dispatch on `type(particleState)` and on attribute presence,
not on a scheme tag).

**The BDF2 history lives on the system, not the state** (`ACSPH_PLAN.md`
Sec. 4.1). `x^n, x^{n-1}, v^n, v^{n-1}` and the two previous real timesteps are
inputs to the real-time source term, not integrated quantities: putting them on
the state would make `initializeNewState` clone them at every integrator stage,
and they must be *frozen* across the whole step. `WeaklyCompressibleSystem`
already carries non-state data this way (`adjacency`, `t`, `domain`), so this
follows the established precedent. `rollHistory` is the single place they
advance; the first step has no `u^{n-1}` and falls back to BDF1, which
`bdfCoefficients` reports via its `order` return.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
from torch.profiler import record_function
from warpSPHCore import *
from warpSPHIntegrators import *

from ..rigidBody.integrate import integrateRigidBody
from ..rigidBody.update import updateBodyParticlesWCSPH
from ..modules.shifting.wrapper import solveShifting

__all__ = ['ArtificialCompressibleState', 'ArtificialCompressibleSystemUpdate',
           'ArtificialCompressibleSystem', 'bdfCoefficients']


@dataclass
class ArtificialCompressibleState(BaseState):
    positions: torch.Tensor = integrated('dxdt', tags=('position',))
    velocities: torch.Tensor = integrated('dvdt', tags=('velocity',))
    #: The scheme's third integrated field -- see the module docstring.
    pressures: torch.Tensor = integrated('dpdt', tags=('pressure',))

    supports: torch.Tensor = constant(tags=('particle_support',))
    masses: torch.Tensor = constant(tags=('particle_mass',))
    #: Invariant at `rho0`. Kept as a field (every operator reads it for the
    #: volume weight `m_j/rho_j`) but never integrated.
    densities: torch.Tensor = constant(tags=('particle_density',))

    kinds: torch.Tensor = constant(tags=('particle_kind',))
    materials: torch.Tensor = constant(tags=('particle_material',))
    UIDs: torch.Tensor = constant(tags=('particle_UID',))
    UIDcounter: int = constant(tags=('particle_UIDcounter',))

    soundspeeds: torch.Tensor = constant(tags=('soundSpeed',), default=None)
    surfaceIndicators: torch.Tensor = constant(tags=('surfaceIndicator',), default=None)
    surfaceNormals: torch.Tensor = constant(tags=('surfaceNormal',), default=None)
    surfaceLambdas: torch.Tensor = constant(tags=('surfaceLambda',), default=None)

    ghostIndices: torch.Tensor = constant(tags=('ghostIndices',), default=None)
    ghostOffsets: torch.Tensor = constant(tags=('ghostOffsets',), default=None)


@dataclass
class ArtificialCompressibleSystemUpdate:
    dxdt: torch.Tensor = tagged(tags=('position_derivative',))
    dvdt: torch.Tensor = tagged(tags=('velocity_derivative',))
    dpdt: torch.Tensor = tagged(tags=('pressure_derivative',))
    passive: Optional[torch.Tensor] = tagged(tags=('passive_derivative',), default=None)


def bdfCoefficients(dt: float, dtPrev: Optional[float]) -> Tuple[float, float, float, int]:
    """`(alpha_t, beta_t, gamma_t, order)` for the real-time source, De Courcy
    et al. 2024 Eq. (42):

        alpha_t = (2 dt^n + dt^{n-1}) / ((dt^n + dt^{n-1}) dt^n)
        beta_t  = -(dt^n + dt^{n-1}) / (dt^n dt^{n-1})
        gamma_t = dt^n / ((dt^n + dt^{n-1}) dt^{n-1})

    so that `Du/Dt ~ alpha_t u^{n+1} + beta_t u^n + gamma_t u^{n-1}`. At
    `dtPrev == dt` this is the familiar `(1.5, -2, 0.5)/dt`.

    `dtPrev is None` (the first step, which has no `u^{n-1}`) falls back to
    BDF1, `(1, -1, 0)/dt`, and reports `order = 1` so a caller can say so.
    """
    if dtPrev is None or dtPrev <= 0.0:
        return 1.0 / dt, -1.0 / dt, 0.0, 1
    alpha = (2.0 * dt + dtPrev) / ((dt + dtPrev) * dt)
    beta = -(dt + dtPrev) / (dt * dtPrev)
    gamma = dt / ((dt + dtPrev) * dtPrev)
    return alpha, beta, gamma, 2


@dataclass
class ArtificialCompressibleSystem(BaseIntegrationSystem):
    state: ArtificialCompressibleState = reference_state(tags=('physics_state',))
    adjacency: Optional[AdjacencyList] = None
    domain: Optional[DomainDescription] = None
    t: float = 0.0

    #: BDF2 history: `u^n` (the state at the start of the current real step) and
    #: `u^{n-1}` (the step before). `None` until `rollHistory` has run; the
    #: step function treats a missing `u^{n-1}` as "first step, use BDF1".
    positionsPrev: Optional[torch.Tensor] = None
    velocitiesPrev: Optional[torch.Tensor] = None
    positionsPrev2: Optional[torch.Tensor] = None
    velocitiesPrev2: Optional[torch.Tensor] = None
    #: `dt^{n-1}`, needed for Eq. (42)'s variable-step coefficients.
    dtPrev: Optional[float] = None

    def initializeNewState(self, *args, verbose=False, **kwargs):
        state = get_reference_state(self)
        verbosePrint(verbose, f'Initializing new state [t={self.t}]')
        # The history is carried by reference, deliberately: it is frozen input
        # to the step, not per-stage state, and cloning it at every integrator
        # stage would be pure waste. `rollHistory` is the only writer.
        return ArtificialCompressibleSystem(
            state=state.initializeNewState(), adjacency=self.adjacency,
            t=self.t, domain=self.domain,
            positionsPrev=self.positionsPrev, velocitiesPrev=self.velocitiesPrev,
            positionsPrev2=self.positionsPrev2, velocitiesPrev2=self.velocitiesPrev2,
            dtPrev=self.dtPrev)

    def rollHistory(self, dt: float):
        """`(x^{n-1}, v^{n-1}) <- (x^n, v^n) <- (x^{n+1}, v^{n+1})`, and
        `dt^{n-1} <- dt^n`. Call once per *real* step, after the dual-time
        solve has converged -- never inside it, and never per integrator
        stage."""
        self.positionsPrev2 = self.positionsPrev
        self.velocitiesPrev2 = self.velocitiesPrev
        self.positionsPrev = self.state.positions.detach().clone()
        self.velocitiesPrev = self.state.velocities.detach().clone()
        self.dtPrev = dt

    def bdfCoefficients(self, dt: float):
        """This system's `(alpha_t, beta_t, gamma_t, order)`. Degrades to BDF1
        while there is no `u^{n-1}` to reference, whatever `dtPrev` says."""
        if self.positionsPrev2 is None:
            return bdfCoefficients(dt, None)
        return bdfCoefficients(dt, self.dtPrev)

    def apply_position_update(self, update, spec: PositionUpdateSpec, **kwargs):
        return update_position(self, update, spec, 'position', 'position_derivative',
                               'velocity', 'velocity_derivative')

    def apply_velocity_update(self, update, spec: ComponentUpdateSpec, **kwargs):
        return update_component(self, update, spec, 'velocity', 'velocity_derivative')

    def apply_quantity_update(self, update, spec: ComponentUpdateSpec, **kwargs):
        return update_component(self, update, spec, 'pressure', 'pressure_derivative')

    def apply_state_update(self, update, spec: ComponentUpdateSpec, **kwargs):
        # Time is the integrator's to advance, not ours.
        position_spec = PositionUpdateSpec(derivative_dt=spec.derivative_dt,
                                           blend=spec.blend)
        self.apply_position_update(update, position_spec, **kwargs)
        self.apply_velocity_update(update, spec, **kwargs)
        self.apply_quantity_update(update, spec, **kwargs)
        return self

    def finalize(self, initialState, dt, returnValues, updateValues, weights=...,
                 *args, **kwargs):
        """Copy back the derived fields the step recomputed, apply the
        particle shift, roll the BDF history, and advance any rigid bodies.

        Thinner than `WeaklyCompressibleSystem.finalize` because there is no
        density update to reconcile -- density is invariant here.

        The particle shift (`ACSPH_PLAN.md` step 7, `PST_ALE_PLAN.md` Stage A)
        runs here rather than in the step function: Eq. (58) applies it as a
        displacement *outside* the pseudo-time loop, and `finalize` is exactly
        that point -- the step owns the dual-time solve, and the shift by
        construction happens after it has converged. Same place
        `WeaklyCompressibleSystem.finalize` runs `solveShifting`, minus that
        system's `correctdrhodt`/`correctdvdt` (nothing to correct: density is
        invariant here).
        """
        self.adjacency = returnValues[-1][0]
        lastState = returnValues[-1][1]

        self.state.supports.copy_(lastState.supports)
        for name in ('soundspeeds', 'surfaceIndicators', 'surfaceNormals',
                     'surfaceLambdas'):
            incoming = getattr(lastState, name, None)
            current = getattr(self.state, name, None)
            if current is not None and incoming is not None:
                current.copy_(incoming)
            else:
                setattr(self.state, name,
                        incoming.clone() if incoming is not None else None)

        config = kwargs.get('config', None)
        schemeConfig = kwargs.get('schemeConfig', None)

        if config is not None and schemeConfig is not None and schemeConfig.shiftProperties.active:
            # Eq. (58): the particle shift applied as a displacement outside
            # the pseudo-time loop -- see this method's own docstring. Unlike
            # `WeaklyCompressibleSystem.finalize`, no `correctdrhodt`/
            # `correctdvdt` here: those correct the shift's divergence
            # contribution to density/momentum, and density is invariant in
            # this scheme by construction (nothing for them to correct).
            with record_function("[warpSPH] - [ACSPH] - solve shifting"):
                dx = solveShifting(
                    systemState=self.state,
                    config=config,
                    schemeConfig=schemeConfig,
                    adjacency=self.adjacency,
                    dt=dt,
                )
                self.state.positions = self.state.positions + dx

        if schemeConfig is not None:
            for rigidBody in schemeConfig.rigidBodies:
                rigidBody = integrateRigidBody(rigidBody, 0, 0, dt)
                self.state = updateBodyParticlesWCSPH(self.state, rigidBody)

        self.rollHistory(dt)
        return super().finalize(initialState, dt, returnValues, updateValues,
                                weights, *args, **kwargs)
