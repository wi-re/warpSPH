"""State and system containers for the wave-equation demo.

``WaveSystemStatev3`` carries the usual particle constants (positions,
supports, masses, ...) so the same samplers and neighbour search work on it,
plus the four fields the wave equation actually needs: ``u`` and ``v`` (the
integrated pair), ``c`` (per-particle wave speed, which is what makes
obstacles and boundaries "hard") and ``damping`` (the absorbing profile).

``u`` is tagged ``'dxdt'``/``'position'`` and ``v`` ``'dvdt'``/``'velocity'``
purely so ``warpSPHIntegrators`` treats the pair with its position/velocity
integrators; neither is a position or a velocity. That reuse is the point --
the wave system rides the same integrator machinery as the fluid schemes.

Not a fluid scheme; see :mod:`warpSPH.schemes.pde.waveEquation` for the step
function and :mod:`warpSPH.cases.demos.waveEquation` for the registered `Case`.
"""

import torch
from typing import Optional
from dataclasses import dataclass

from ..configurations import SimulationConfig
from ..sample.bySamplingScheme import sampleParticles

from ..utils import *
from warpSPHIntegrators.state import BaseIntegrationSystem, BaseState, ComponentUpdateSpec, PositionUpdateSpec, constant, get_reference_state, integrated, reference_state, tagged
from warpSPHIntegrators.state.fields import update_component, update_position
from warpSPHCore import *


@dataclass
class WaveSystemStatev3(BaseState):
    positions : torch.Tensor = constant(tags=('particle_position',))
    supports : torch.Tensor = constant(tags=('particle_support',))
    masses : torch.Tensor = constant(tags=('particle_mass',))
    densities : torch.Tensor = constant(tags=('particle_density',))

    kinds : torch.Tensor = constant(tags=('particle_kind',))
    materials : torch.Tensor = constant(tags=('particle_material',))
    UIDs : torch.Tensor = constant(tags=('particle_UID',))
    UIDcounter : int = constant(tags=('particle_UIDcounter',))

    u : torch.Tensor = integrated('dxdt', tags=('position',))
    v : torch.Tensor = integrated('dvdt', tags=('velocity',))
    c : torch.Tensor = constant(tags=('soundSpeed',))
    damping : torch.Tensor = constant(tags=('damping',))

@dataclass
class WaveSystemUpdatev3:
    dudt : torch.Tensor = tagged(tags=('position_derivative',))
    dvdt : torch.Tensor = tagged(tags=('velocity_derivative',))

@dataclass
class WaveSystemv3(BaseIntegrationSystem):
    state: WaveSystemStatev3 = reference_state(tags=('oscillator_state',))
    adjacency: Optional[AdjacencyList] = None
    domain: Optional[DomainDescription] = None
    t: float = 0.0
    def initializeNewState(self, *args, verbose=False, **kwargs):
        state = get_reference_state(self)
        if verbose:   # self.t may be a device tensor: format it only when printing
            print(f'Initializing new state [t={self.t}]')
        return WaveSystemv3(state=state.initializeNewState(), adjacency=self.adjacency, t=self.t, domain=self.domain)
    
    def apply_position_update(self, update, spec: PositionUpdateSpec, **kwargs):
        return update_position(self, update, spec, 'position', 'position_derivative', 'velocity', 'velocity_derivative')
    def apply_velocity_update(self, update, spec: ComponentUpdateSpec, **kwargs):
        return update_component(self, update, spec, 'velocity', 'velocity_derivative')
    def apply_quantity_update(self, update, spec: ComponentUpdateSpec, **kwargs):
        return self
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
    

from ..configurations import WaveCaseConfig

def sampleInitialWaveState(nx, config: SimulationConfig, caseConfig: WaveCaseConfig):
    particles = sampleParticles(nx, config)
    particleState = WaveSystemStatev3(
        positions=particles.positions,
        supports=particles.supports,
        masses=particles.masses,
        densities=particles.densities,
        
        kinds=torch.zeros(particles.positions.shape[0], device=config.device, dtype=torch.int32),
        materials=torch.zeros(particles.positions.shape[0], device=config.device, dtype=torch.int32),
        UIDs=torch.arange(particles.positions.shape[0], device=config.device, dtype=torch.int32),
        UIDcounter=particles.positions.shape[0],
        
        u=torch.zeros(particles.positions.shape[0], device=config.device),
        v=torch.zeros(particles.positions.shape[0], device=config.device),
        c=torch.ones(particles.positions.shape[0], device=config.device) * caseConfig.defaultSpeed,
        damping=torch.zeros(particles.positions.shape[0], device=config.device)
    )
    return particleState


from typing import List
def computeDt(waveSystem: WaveSystemv3, config: SimulationConfig, caseConfig: WaveCaseConfig, args, obstacleSpeeds: List[float], verbose=False):
    # Compute CFL number based on initial conditions
    n = waveSystem.state.u.shape[0]
    domainVolume = 1.0
    for d in range(config.dim):
        domainVolume *= (config.domain.max[d] - config.domain.min[d])
    dx = (domainVolume / n) ** (1 / config.dim)

    dt = 0.02
    cflNumber = max(obstacleSpeeds + [caseConfig.defaultSpeed]) * dt / dx

    # runningState = waveSystem.initializeNewState(verbose=verbose)
    hMax = waveSystem.state.supports.max().item()
    if verbose:
        print(f'Max wave speed: {waveSystem.state.c.max().item():.4f}')
        print(f'Max support radius: {hMax:.4f}')
        print(f'CFL number (based on dx): {cflNumber:.4f} [dx={dx:.4f}, dt={dt:.4f}]')
        print(f'CFL number (based on hMax): {max(obstacleSpeeds + [caseConfig.defaultSpeed]) * dt / hMax:.4f} [hMax={hMax:.4f}, dt={dt:.4f}]')

    targetCFL = config.cflFactor
    # Adjust dt such that the CFL number based on hMax is equal to targetCFL
    dt = targetCFL * hMax / max(obstacleSpeeds + [caseConfig.defaultSpeed])
    if verbose:
        print(f'Adjusted dt for target CFL of {targetCFL}: {dt:.4f}')
        print(f'New CFL number (based on dx): {max(obstacleSpeeds + [caseConfig.defaultSpeed]) * dt / dx:.4f}')
        print(f'New CFL number (based on hMax): {max(obstacleSpeeds + [caseConfig.defaultSpeed]) * dt / hMax:.4f}')
    return dt