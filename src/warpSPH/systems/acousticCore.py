"""State and system containers for the rudimentary WCSPH acoustic core
(`JFNK_PLAN.md` Phase B): continuity + EOS + pressure force, nothing else --
no surface treatment, no diffusion, no shifting, no boundaries, no rigid
bodies. A second, real-fluid validation rung for `warpSPHIntegrators`'
`JFNKSolver`, between the wave equation (Phase A, one linear operator) and
the full `deltaSPH_step` acoustic subsystem (Phase C/D).

Field layout matches `WeaklyCompressibleState` (`systems/weaklyCompressible.py`)
for the three physical fields this needs -- `positions`/`velocities`/
`densities` -- but drops every field that scheme carries for surface
detection, mDBC, and shifting, none of which this core has code paths for.

See :mod:`warpSPH.schemes.experimental.acousticCore` for the step function.
"""

import torch
from typing import Optional
from dataclasses import dataclass

from warpSPHIntegrators import *
from warpSPHCore import *

__all__ = ['AcousticCoreState', 'AcousticCoreSystemUpdate', 'AcousticCoreSystem']


@dataclass
class AcousticCoreState(BaseState):
    positions : torch.Tensor = integrated('dxdt', tags=('position',))
    velocities: torch.Tensor = integrated('dvdt', tags=('velocity',))
    densities : torch.Tensor = integrated('drhodt', tags=('density',))
    supports : torch.Tensor = constant(tags=('particle_support',))
    masses : torch.Tensor = constant(tags=('particle_mass',))

    kinds : torch.Tensor = constant(tags=('particle_kind',))
    materials : torch.Tensor = constant(tags=('particle_material',))
    UIDs : torch.Tensor = constant(tags=('particle_UID',))
    UIDcounter : int = constant(tags=('particle_UIDcounter',))


@dataclass
class AcousticCoreSystemUpdate:
    dxdt: torch.Tensor = tagged(tags=('position_derivative',))
    dvdt: torch.Tensor = tagged(tags=('velocity_derivative',))
    drhodt: torch.Tensor = tagged(tags=('density_derivative',))


@dataclass
class AcousticCoreSystem(BaseIntegrationSystem):
    state: AcousticCoreState = reference_state(tags=('physics_state',))
    adjacency: Optional[AdjacencyList] = None
    domain: Optional[DomainDescription] = None
    t: float = 0.0

    def initializeNewState(self, *args, verbose=False, **kwargs):
        state = get_reference_state(self)
        if verbose:   # self.t may be a device tensor: format it only when printing
            print(f'Initializing new state [t={self.t}]')
        return AcousticCoreSystem(state=state.initializeNewState(), adjacency=self.adjacency, t=self.t, domain=self.domain)

    def apply_position_update(self, update, spec: PositionUpdateSpec, **kwargs):
        return update_position(self, update, spec, 'position', 'position_derivative', 'velocity', 'velocity_derivative')
    def apply_velocity_update(self, update, spec: ComponentUpdateSpec, **kwargs):
        return update_component(self, update, spec, 'velocity', 'velocity_derivative')
    def apply_quantity_update(self, update, spec: ComponentUpdateSpec, **kwargs):
        return update_component(self, update, spec, 'density', 'density_derivative')
    def apply_state_update(self, update, spec: ComponentUpdateSpec, **kwargs):
        # Note: DO NOT advance self.t here. Time is managed by the integrator.
        position_spec = PositionUpdateSpec(derivative_dt=spec.derivative_dt, blend=spec.blend)
        self.apply_position_update(update, position_spec, **kwargs)
        self.apply_velocity_update(update, spec, **kwargs)
        self.apply_quantity_update(update, spec, **kwargs)
        return self

    def finalize(self, initialState, dt, returnValues, updateValues, weights = ..., *args, **kwargs):
        self.adjacency = returnValues[-1][0]  # Assuming the adjacency list is the last return value from the derivative function
        return super().finalize(initialState, dt, returnValues, updateValues, weights, *args, **kwargs)
