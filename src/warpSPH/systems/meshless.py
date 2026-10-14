"""State / update / system triad for the meshless finite-mass / finite-volume (MFM / MFV)
scheme (Hopkins 2015; COMPRESSIBLE_SPH_PLAN E1, `schemes/compressible/meshless/` (`mfm.py`, `mfv.py`, `temporal.py`)).

The integrated physics quantity is the *conserved* vector ``Q = (m, m v, m e_tot)`` per
particle (a single ``(N, 2+dim)`` tensor) together with the positions. The primitives
(rho, v, u, P) are derived from ``Q`` and the geometric volume every step. Integrating
``Q`` directly -- not the primitives -- is what keeps the total mass / momentum / energy
conserved to machine precision for both MFM (dm = 0) and MFV (dm != 0), which the G1 gate
(conservation 1e-15 in float64) requires.

``supports`` (h) and ``lam`` (the closure warm start) are non-integrated: the step
re-derives them and `finalize` copies them from the last substep, as for the other
compressible schemes. The trailing derived-primitive fields are diagnostics only (never
integrated); the step refreshes them and `finalize` carries the last substep's values
forward so the runner's plot / diagnostic hooks can read them.
"""

from warpSPHIntegrators import *
from dataclasses import dataclass
import torch
from typing import Optional
from warpSPHCore import *

__all__ = ['MeshlessState', 'MeshlessSystemUpdate', 'MeshlessSystem']


@dataclass
class MeshlessState(BaseState):
    positions: torch.Tensor = integrated('dxdt', tags=('position',))
    # the conserved vector (m, m v, m e_tot); the single integrated physics quantity
    Q: torch.Tensor = integrated('dQdt', tags=('conserved',))
    supports: torch.Tensor = constant(tags=('particle_support',))

    kinds: torch.Tensor = constant(tags=('particle_kind',))
    materials: torch.Tensor = constant(tags=('particle_material',))
    UIDs: torch.Tensor = constant(tags=('particle_UID',))
    UIDcounter: int = constant(tags=('particle_UIDcounter',))

    # the previous step's closure potential (N, dim): warm start for the CG solve
    lam: Optional[torch.Tensor] = constant(tags=('closure_lam',), default=None)
    # derived primitives (diagnostics), refreshed by the step and copied in finalize
    masses: torch.Tensor = constant(tags=('particle_mass',), default=None)
    densities: torch.Tensor = constant(tags=('particle_density',), default=None)
    velocities: torch.Tensor = constant(tags=('velocity',), default=None)
    internalEnergies: torch.Tensor = constant(tags=('internalEnergy',), default=None)
    totalEnergies: torch.Tensor = constant(tags=('energy',), default=None)
    entropies: torch.Tensor = constant(tags=('entropy',), default=None)
    pressures: torch.Tensor = constant(tags=('pressure',), default=None)
    soundspeeds: torch.Tensor = constant(tags=('soundSpeed',), default=None)


@dataclass
class MeshlessSystemUpdate:
    dxdt: torch.Tensor = tagged(tags=('position_derivative',))
    dQdt: torch.Tensor = tagged(tags=('conserved_derivative',))


@dataclass
class MeshlessSystem(BaseIntegrationSystem):
    state: MeshlessState = reference_state(tags=('physics_state',))
    adjacency: Optional[AdjacencyList] = None
    domain: Optional[DomainDescription] = None
    t: float = 0.0

    def initializeNewState(self, *args, verbose=False, **kwargs):
        state = get_reference_state(self)
        if verbose:   # self.t may be a device tensor: format it only when printing
            print(f'Initializing new state [t={self.t}]')
        return MeshlessSystem(state=state.initializeNewState(), adjacency=self.adjacency, t=self.t, domain=self.domain)

    def apply_position_update(self, update, spec: PositionUpdateSpec, **kwargs):
        # applies the step's exact net drift delta (pos += dt * dxdt, wrapping): the step
        # (`schemes/compressible/meshless/` (`mfm.py`, `mfv.py`, `temporal.py`), D4) owns the temporal scheme, so dxdt is its time-centred /
        # drift velocity, not a raw start-of-step velocity. 'velocity' fills the integrator's
        # kick-tag slot only: the kick terms (current_velocity_dt / update_velocity_dt) are
        # None for the single-evaluation integrators the step admits, so it is never read --
        # the conserved momentum lives in Q, not in a separate field
        update_position(self, update, spec, 'position', 'position_derivative', 'velocity')
        # wrap into the domain (periodic axes): the drift can cross a periodic edge, and the
        # core driver wraps at every move
        if self.domain is not None:
            self.state.positions = torch.where(
                self.domain.periodic,
                self.domain.min + torch.remainder(self.state.positions - self.domain.min,
                                                  self.domain.max - self.domain.min),
                self.state.positions)
        return self

    def apply_quantity_update(self, update, spec: ComponentUpdateSpec, **kwargs):
        # advance the conserved vector: Q += dt * dQdt
        return update_component(self, update, spec, 'conserved', 'conserved_derivative')

    def apply_state_update(self, update, spec: ComponentUpdateSpec, **kwargs):
        # Note: DO NOT advance self.t here. Time is managed by the integrator.
        position_spec = PositionUpdateSpec(derivative_dt=spec.derivative_dt, blend=spec.blend)
        self.apply_position_update(update, position_spec, **kwargs)
        self.apply_quantity_update(update, spec, **kwargs)
        return self

    def finalize(self, initialState, dt, returnValues, updateValues, weights=..., *args, **kwargs):
        self.adjacency = returnValues[-1][0]
        # Copy the last substep's non-integrated values into the current state
        lastState = returnValues[-1][1]
        self.state.supports.copy_(lastState.supports)
        if self.state.lam is not None and lastState.lam is not None:
            self.state.lam.copy_(lastState.lam)
        for _f in ('masses', 'densities', 'velocities', 'internalEnergies', 'totalEnergies',
                   'entropies', 'pressures', 'soundspeeds'):
            if getattr(self.state, _f) is not None and getattr(lastState, _f) is not None:
                getattr(self.state, _f).copy_(getattr(lastState, _f))
        return super().finalize(initialState, dt, returnValues, updateValues, weights, *args, **kwargs)
