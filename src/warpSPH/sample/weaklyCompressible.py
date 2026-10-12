"""Shared weakly-compressible initial state: a uniform, at-rest regular
lattice (zero pressure/velocity, unit density/soundspeed), wrapped straight
into `SimulationState`/`SimulationSystem` with no support/density relaxation
step (unlike `sample.compressible`'s counterpart). Re-exported at the
top-level `warpSPH` package; used by the TGV cases (`cases/periodic/tgv.py`,
`cases/periodic/tgvWeaklyCompressible.py`).
"""

import torch
from ..modules.timestep.compressible import computeTimestep
from .regular import sampleRegularParticles
from ..modules import *
from ..configurations import CompressibleSPHConfig
from ..enumTypes import *
from warpSPHCore import *

__all__ = ['setupBasicWeaklyCompressibleInitialState']

def setupBasicWeaklyCompressibleInitialState(
        nx,
        config, schemeConfig,
        SimulationState, SimulationSystem,
):
    particles_l = sampleRegularParticles(nx, config.domain, config.targetNeighbors, jitter = 0.0)
    Pinitial = torch.zeros_like(particles_l.densities)
    rhoInitial = torch.ones_like(particles_l.densities)
    v_initial = torch.zeros_like(particles_l.positions)



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
        
        pressures = Pinitial,
        soundspeeds = torch.ones_like(particles_l.densities),
    )
        
    compressibleSystem = SimulationSystem(
        state=simulationState, 
        adjacency = None, 
        domain = config.domain
    )
        
    dx = simulationState.masses.min() ** (1.0 / config.dim)
    config.dx = dx
    # config.dt = computeTimestep(compressibleSystem, config, schemeConfig, dt = config.dt)

    return compressibleSystem
