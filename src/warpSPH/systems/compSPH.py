"""State/system pair for the CompSPH scheme (Owen's compatible-energy
compressible formulation, `schemes/compSPH.py` and, sharing this state,
`schemes/crkSPH.py`/`schemes/deltaSPH.py`). Extends the base particle fields
with internal energy as the integrated quantity plus EOS outputs
(pressure/soundspeed/entropy/total energy), the CullenDehnen2010
shock-detector state (`divergence`, `alpha0s`/`alphas`), and the pairwise
accel/energy-partition arrays (`ap_ij`/`av_ij`/`f_ij`) `compSPH_deltaU_multistep`
consumes to keep energy exchange compatible between neighbors. There is no
`CompSPHSystemUpdate` here -- callers build a `compressibleMonaghan.
CompressibleSystemUpdate` instead (its commented-out draft below is dead).
`CompSPHSystem.finalize` only runs the multistep energy correction when
`schemeConfig.compatibleEnergy` is set.
"""

from warpSPHIntegrators import *
from dataclasses import dataclass
import torch
from typing import Optional
from warpSPHCore import *

__all__ = ['CompSPHState', 'CompSPHSystem']


@dataclass
class CompSPHState(BaseState):
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
    # Sphenix2022 (step-boundary): div v at the previous step boundary and its time (Eq. 22)
    divergencePrevStep: torch.Tensor = constant(tags=('velocity_divergence_previous',), default=None)
    divergencePrevStepTime: float = constant(tags=('velocity_divergence_previous_time',), default=None)

    ap_ij: torch.Tensor = constant(tags=('pairwise_acceleration',), default=None)
    av_ij: torch.Tensor = constant(tags=('pairwise_acceleration',), default=None)
    f_ij: torch.Tensor = constant(tags=('pairwise_energy_partition',), default=None)

    # unit wall normals (fluid -> wall) of the solid rows, zero on fluid rows; None
    # when the case gives no geometry (modules/compressibleWall then estimates them)
    wallNormals: torch.Tensor = constant(tags=('wall_normal',), default=None)


# from .compressibleMonaghan import CompressibleSystemUpdate
# from ..configurations.compSPHConfig import CompSPHConfig
# from ..configurations.simulationConfig import SimulationConfig
# from ..enumTypes import EnergyScheme
# @dataclass
# class CompressibleSystemUpdate:
#     dxdt: torch.Tensor = tagged(tags=('position_derivative',))
#     dvdt: torch.Tensor = tagged(tags=('velocity_derivative',))
#     dudt: torch.Tensor = tagged(tags=('internalEnergy_derivative',))
#     dEdt: torch.Tensor = tagged(tags=('totalEnergy_derivative',))
#     drhodt: torch.Tensor = tagged(tags=('density_derivative',))

from ..modules.compSPH.multistep import compSPH_deltaU_multistep


@dataclass
class CompSPHSystem(BaseIntegrationSystem):
    state: CompSPHState = reference_state(tags=('physics_state',))
    adjacency: Optional[AdjacencyList] = None
    domain: Optional[DomainDescription] = None
    t: float = 0.0
    def initializeNewState(self, *args, verbose=False, **kwargs):
        state = get_reference_state(self)
        verbosePrint(verbose, f'Initializing new state [t={self.t}]')
        return CompSPHSystem(state=state.initializeNewState(), adjacency=self.adjacency, t=self.t, domain=self.domain)
    
    def apply_position_update(self, update, spec: PositionUpdateSpec, **kwargs):
        return update_position(self, update, spec, 'position', 'position_derivative', 'velocity', 'velocity_derivative')
    def apply_velocity_update(self, update, spec: ComponentUpdateSpec, **kwargs):
        return update_component(self, update, spec, 'velocity', 'velocity_derivative')
        # Passive particles only drift with the velocity update, they do not get accelerated
        # if hasattr(update, 'passive') and update.passive is not None:
            # if hasattr(update, 'dxdt') and update.dxdt is not None:
            # self.state.velocities[update.passive] = update.dxdt[update.passive]
            # else:
                # updated.state.velocities[update.passive] = self.state.velocities[update.passive]
        return updated

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
    
    def finalize(self, initialState, dt, returnValues, updateValues, weights, config, schemeConfig, *args, **kwargs):
        self.adjacency = returnValues[-1][0]  # Assuming the adjacency list is the last return value from the derivative function
        # Copy the last substeps values into the current state to ensure the final state is correct
        lastState = returnValues[-1][1]  # Assuming the state is the second return value from the derivative function
        # General attributes

        # print(returnValues)
        # for r, rv in enumerate(returnValues):
        #     print(f'Index: {r}: type(rv): {type(rv)}')
        #     for item in rv:
        #         print(f'\ttype(item): {type(item)}')

        
        if schemeConfig.compatibleEnergy:
            delta_u = compSPH_deltaU_multistep(
                dt,
                initialState.state,
                returnValues,
                updateValues,
                weights,
                config,
                schemeConfig,
                # verbose = verbose
            )
            self.state.internalEnergies = initialState.state.internalEnergies + delta_u * dt

        # solid wall rows (modules/compressibleWall): the state of the last stage, not an
        # integrated one -- their pair work went to the fluid (f_ij = 1), and their mass
        # is density-dependent while masses is tagged constant
        solid = self.state.kinds != 0
        self.state.internalEnergies = torch.where(solid, lastState.internalEnergies, self.state.internalEnergies)
        self.state.masses.copy_(lastState.masses)

        self.state.supports.copy_(lastState.supports)
        self.state.densities.copy_(lastState.densities)

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
        advanceViscositySwitchStep(self, initialState, returnValues, schemeConfig, config)

        updateValues[-1].dudt = (self.state.internalEnergies - initialState.state.internalEnergies) /dt

        return super().finalize(initialState, dt, returnValues, updateValues, weights, config, schemeConfig, *args, **kwargs)
    