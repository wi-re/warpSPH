"""Equal-resolution sampler for `cases/compressible/triplePoint.py`'s three-region setup:
one uniform lattice from `setupBasicCompressibleInitialState`, with each
particle's density/pressure set from `splitX`/`splitY`-derived region masks
(unlike `equalMass.py`, these masks do use the passed-in `splitX`/`splitY`)
and masses set to ``cell area * region density`` -- so, unlike the equal-mass
sampler, particle mass varies by up to the region density ratio. Densities
and supports are then relaxed with `evaluateOptimalSupport` before the EOS
state is derived via `idealGasEOS`.
"""

from ....sample import *
import torch
from ....sample.compressible import setupBasicCompressibleInitialState
from ....modules import *
from warpSPHCore import *
from ....modules.timestep.compressible import computeTimestep
import math
import numpy as np
from warpSPH import *

__all__ = ['sampleTriplePointEqualResolution']


def sampleTriplePointEqualResolution(
        splitX, splitY,
        rho_I, p_I,
        rho_II, p_II,
        rho_III, p_III,
        nx, config, schemeConfig, extraData, SimulationState, SimulationSystem
):
    compressibleSystem = setupBasicCompressibleInitialState(nx, config, schemeConfig, SimulationState, SimulationSystem)
    # print(f"Number of particles: {compressibleSystem.state.positions.shape[0]}")
    particles = compressibleSystem.state

    rhoInitial = particles.densities.clone()
    Pi = particles.pressures.clone()

    regionI = torch.logical_or(particles.positions[:,0] <= splitX, particles.positions[:,0] >= config.domain.max[0] - splitX)
    regionII = torch.logical_and(~regionI, particles.positions[:,1].abs() >= splitY)
    regionIII = torch.logical_and(~regionI, particles.positions[:,1].abs() < splitY)

    rhoInitial[regionI] = rho_I
    Pi[regionI] = p_I
    rhoInitial[regionII] = rho_II
    Pi[regionII] = p_II
    rhoInitial[regionIII] = rho_III
    Pi[regionIII] = p_III

    densityRatio = rhoInitial / particles.densities
    # particles.masses = particles.masses * densityRatio

    Lx = config.domain.max[0] - config.domain.min[0]
    Ly = config.domain.max[1] - config.domain.min[1]
    aspect = Ly / Lx

    dx = Lx / (nx / aspect)
    dy = Ly / nx

    area = Lx * Ly / nx**2 * aspect

    compressibleSystem.state.masses = area * rhoInitial
    compressibleSPHConfigAdaptiveH = CompressibleSPHConfig(
        adaptiveSupportIterations=16,
        adaptiveSupportThreshold=1e-3,
        adaptiveSupportScheme=AdaptiveSupportScheme.Owen,
    )

    rho_optimal, h_optimal, adjacency, rhos_iter, supports_iter = evaluateOptimalSupport(compressibleSystem.state, config, supportScheme = SupportScheme.Gather, compParams = compressibleSPHConfigAdaptiveH)

    compressibleSystem.state.supports = h_optimal
    compressibleSystem.state.densities = rho_optimal


    A_, u_, P_, c_s = idealGasEOS(A = None, u = None, P = Pi, rho = compressibleSystem.state.densities, gamma = schemeConfig.gamma)
    # v_initial = torch.zeros_like(particles_l.positions)

    vInitial = torch.zeros_like(particles.positions)
    internalEnergy = u_ 
    kineticEnergy = torch.linalg.norm(vInitial, dim = -1) **2/ 2
    totalEnergy = (internalEnergy + kineticEnergy) * compressibleSystem.state.masses

    compressibleSystem.state.internalEnergies = u_
    compressibleSystem.state.totalEnergies = totalEnergy
    compressibleSystem.state.pressures = P_
    compressibleSystem.state.soundspeeds = c_s
    compressibleSystem.state.velocities = vInitial
    compressibleSystem.state.densities = rhoInitial

    config.dt = computeTimestep(compressibleSystem, config, schemeConfig, dt = None) #* 2/3

    return compressibleSystem