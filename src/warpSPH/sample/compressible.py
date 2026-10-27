"""Shared compressible-scheme initial state: a uniform, at-rest regular
lattice (zero pressure/velocity, unit density) relaxed to optimal
supports/densities via `evaluateOptimalSupport`, with pressure/energy/entropy
/soundspeed derived through `idealGasEOS`. Re-exported at the top-level
`warpSPH` package; used directly or as the common starting point by most
compressible cases (Sod-like/noh/hydrostatic/Kelvin-Helmholtz/Rayleigh-Taylor
/Gresho/triple-point/Yee-vortex -- see `caseUtils/compressible/*`).
`sampleShockRegions1D` builds on it for 1D multi-region shock tubes (e.g.
Woodward-Colella): it overwrites pressure/density on `abs(x)`-band masks and
re-derives the internal energy/timestep from the patched state.
"""

import torch
from ..modules.timestep.compressible import computeTimestep
from .regular import sampleRegularParticles
from ..modules import *
from ..configurations import CompressibleSPHConfig
from ..enumTypes import *
from warpSPHCore import *

__all__ = ['setupBasicCompressibleInitialState', 'sampleShockRegions1D']

def setupBasicCompressibleInitialState(
        nx,
        config, schemeConfig,
        SimulationState, SimulationSystem,
        jitter=0.0,
):
    particles_l = sampleRegularParticles(nx, config.domain, config.targetNeighbors, jitter = jitter)
    Pinitial = torch.zeros_like(particles_l.densities)
    rhoInitial = torch.ones_like(particles_l.densities)
    v_initial = torch.zeros_like(particles_l.positions)

        
    A_, u_, P_, c_s = idealGasEOS(A = None, u = None, P = Pinitial, rho = particles_l.densities, gamma = schemeConfig.gamma)

    internalEnergy = u_ 
    kineticEnergy = torch.linalg.norm(v_initial, dim = -1) **2/ 2
    totalEnergy = (internalEnergy + kineticEnergy) * particles_l.masses


    simulationState = SimulationState(
        positions = particles_l.positions,
        supports = particles_l.supports,
        masses = particles_l.masses,
        densities = particles_l.densities,        
        velocities = v_initial,

        kinds = torch.zeros_like(particles_l.positions[:,0], dtype = torch.int32),
        materials = torch.zeros_like(particles_l.positions[:,0], dtype = torch.int32),
        UIDs = torch.arange(particles_l.positions.shape[0], device = config.device, dtype = torch.int32),
        UIDcounter= particles_l.positions.shape[0],
        
        internalEnergies = u_,
        totalEnergies = totalEnergy,
        entropies = A_,
        pressures = P_,
        soundspeeds = c_s,

        alphas = torch.ones_like(particles_l.densities),
        alpha0s = torch.ones_like(particles_l.densities),
        divergence = torch.zeros_like(particles_l.densities)
    )
        
    compressibleSPHConfigAdapt = CompressibleSPHConfig(
        adaptiveSupportIterations=16,
        adaptiveSupportThreshold=1e-3,
        adaptiveSupportScheme=AdaptiveSupportScheme.NoScheme,
    )
    rho_optimal, h_optimal, adjacency, rhos_iter, supports_iter = evaluateOptimalSupport(simulationState, config, supportScheme = SupportScheme.Gather, compParams = compressibleSPHConfigAdapt)

    simulationState.supports = h_optimal
    simulationState.densities = rho_optimal


    compressibleSystem = SimulationSystem(
        state=simulationState, 
        adjacency = None, 
        domain = config.domain
    )
        
    dx = simulationState.masses.min() ** (1.0 / config.dim)
    config.dx = dx
    config.dt = computeTimestep(compressibleSystem, config, schemeConfig, dt = config.dt)

    return compressibleSystem



def sampleShockRegions1D(nx, config, schemeConfig, SimulationState, SimulationSystem, initialRegions):
    particleSystem = setupBasicCompressibleInitialState(nx, config, schemeConfig, SimulationState, SimulationSystem)

    for region in initialRegions:
        begin = region['begin']
        end = region['end']
        pressure = region['pressure']
        density = region['density']
        
        mask = (particleSystem.state.positions[:,0].abs() >= begin) & (particleSystem.state.positions[:,0].abs() < end)
        particleSystem.state.pressures[mask] = pressure
        particleSystem.state.densities[mask] = density

    A_, u_, P_, c_s = idealGasEOS(A = None, u = None, P = particleSystem.state.pressures, rho = particleSystem.state.densities, gamma = schemeConfig.gamma)
    particleSystem.state.internalEnergies = u_

    config.dt = computeTimestep(particleSystem, config, schemeConfig, dt = None)

    return particleSystem
