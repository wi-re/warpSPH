"""State/update/system triad for the artificial-viscosity ('Monaghan')
compressible scheme (`schemes/monaghan.py`), carrying internal energy as the
integrated quantity plus the EOS outputs and CullenDehnen2010 shock-detector
fields every energy-based compressible scheme needs. `CompressibleSystemUpdate`
is reused as the generic update container across schemes beyond this one --
`schemes/compSPH.py` and `schemes/crkSPH.py` build one too, since `compSPH.py`
(this package) defines no update class of its own. Its `passive` field is
populated (as an all-`False` mask) by every scheme that constructs one, but is
never read back anywhere; the one call site that would have used it
(`compSPH.CompSPHSystem.apply_velocity_update`) is unreachable dead code after
an earlier `return`.
"""

from warpSPHIntegrators import *
from dataclasses import dataclass
import torch
from typing import Optional
from warpSPHCore import *

__all__ = ['CompressibleState', 'CompressibleSystemUpdate', 'CompressibleSystem']


@dataclass
class CompressibleState(BaseState):
    positions : torch.Tensor = integrated('dxdt', tags=('position',))
    velocities: torch.Tensor = integrated('dvdt', tags=('velocity',))
    supports : torch.Tensor = constant(tags=('particle_support',))
    masses : torch.Tensor = constant(tags=('particle_mass',))
    densities : torch.Tensor = constant(tags=('particle_density',))

    kinds : torch.Tensor = constant(tags=('particle_kind',))
    materials : torch.Tensor = constant(tags=('particle_material',))
    UIDs : torch.Tensor = constant(tags=('particle_UID',))
    UIDcounter : int = constant(tags=('particle_UIDcounter',))

    internalEnergies : torch.Tensor = integrated('dudt', tags=('internalEnergy',))
    totalEnergies : torch.Tensor = constant(tags=('energy',), default=None)
    entropies : torch.Tensor = constant(tags=('entropy',), default=None)
    pressures : torch.Tensor = constant(tags=('pressure',), default=None)
    soundspeeds : torch.Tensor = constant(tags=('soundSpeed',), default=None)

    divergence : torch.Tensor = constant(tags=('velocity_divergence',), default=None)
    alpha0s: torch.Tensor = constant(tags=('alpha0',), default=None)
    alphas: torch.Tensor = constant(tags=('alpha',), default=None)
    # step-boundary switches (Rosswog2020): entropy s^{n-1} and its time, and the last
    # step's entropy rate (Eq. 16); set by the system's finalize, None for the others
    entropiesPrev: torch.Tensor = constant(tags=('entropy_previous',), default=None)
    entropiesPrevTime: float = constant(tags=('entropy_previous_time',), default=None)
    entropyRates: torch.Tensor = constant(tags=('entropy_rate',), default=None)

    # unit wall normals (fluid -> wall) of the solid rows, zero on fluid rows; None
    # when the case gives no geometry (modules/compressibleWall then estimates them)
    wallNormals: torch.Tensor = constant(tags=('wall_normal',), default=None)

@dataclass
class CompressibleSystemUpdate:
    dxdt: torch.Tensor = tagged(tags=('position_derivative',))
    dvdt: torch.Tensor = tagged(tags=('velocity_derivative',))
    dudt: torch.Tensor = tagged(tags=('internalEnergy_derivative',))
    dEdt: torch.Tensor = tagged(tags=('totalEnergy_derivative',))
    drhodt: torch.Tensor = tagged(tags=('density_derivative',))
    passive: Optional[torch.Tensor] = tagged(tags=('passive_derivative',), default=None)

@dataclass
class CompressibleSystem(BaseIntegrationSystem):
    state: CompressibleState = reference_state(tags=('physics_state',))
    adjacency: Optional[AdjacencyList] = None
    domain: Optional[DomainDescription] = None
    t: float = 0.0
    def initializeNewState(self, *args, verbose=False, **kwargs):
        state = get_reference_state(self)
        verbosePrint(verbose, f'Initializing new state [t={self.t}]')
        return CompressibleSystem(state=state.initializeNewState(), adjacency=self.adjacency, t=self.t, domain=self.domain)
    
    def apply_position_update(self, update, spec: PositionUpdateSpec, **kwargs):
        return update_position(self, update, spec, 'position', 'position_derivative', 'velocity', 'velocity_derivative')
    def apply_velocity_update(self, update, spec: ComponentUpdateSpec, **kwargs):
        return update_component(self, update, spec, 'velocity', 'velocity_derivative')
    def apply_quantity_update(self, update, spec: ComponentUpdateSpec, **kwargs):
        updatedsystem = update_component(self, update, spec, 'internalEnergy', 'internalEnergy_derivative')
        # updatedsystem = update_component(updatedsystem, update, spec, 'totalEnergy', 'totalEnergy_derivative')
        # updatedsystem = update_component(updatedsystem, update, spec, 'density', 'density_derivative')
        return updatedsystem

    def apply_state_update(self, update, spec: ComponentUpdateSpec, **kwargs):
        # Note: DO NOT advance self.t here. Time is managed by the integrator.
        position_spec = PositionUpdateSpec(derivative_dt=spec.derivative_dt, blend=spec.blend)
        self.apply_position_update(update, position_spec, **kwargs)
        self.apply_velocity_update(update, spec, **kwargs)
        self.apply_quantity_update(update, spec, **kwargs)
        return self
    
    def finalize(self, initialState, dt, returnValues, updateValues, weights = ..., *args, **kwargs):
        self.adjacency = returnValues[-1][0]  # Assuming the adjacency list is the last return value from the derivative function
        # Copy the last substeps values into the current state to ensure the final state is correct
        lastState = returnValues[-1][1]  # Assuming the state is the second return value from the derivative function
        # General attributes
        self.state.supports.copy_(lastState.supports)
        self.state.densities.copy_(lastState.densities)
        # wall rows carry a density-dependent mass set by modules/compressibleWall each
        # substep; masses is tagged constant so the integrator would otherwise drop
        # it back to the initial value and the wall would read as rest density
        self.state.masses.copy_(lastState.masses)

        # Compressible system specific attributes (internal eneryg is integrated)
        self.state.totalEnergies.copy_(lastState.totalEnergies)
        self.state.entropies.copy_(lastState.entropies)
        self.state.pressures.copy_(lastState.pressures)
        self.state.soundspeeds.copy_(lastState.soundspeeds)

        # Information for artificial viscosity switches
        self.state.divergence.copy_(lastState.divergence)
        self.state.alpha0s.copy_(lastState.alpha0s)
        self.state.alphas.copy_(lastState.alphas)
        from ..modules.shockCapturing.wrapper import advanceViscositySwitchStep  # (circular at import time)
        advanceViscositySwitchStep(self, initialState, returnValues,
                                   kwargs.get('schemeConfig', args[1] if len(args) > 1 else None))

        return super().finalize(initialState, dt, returnValues, updateValues, weights, *args, **kwargs)
    