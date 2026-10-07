"""Kelvin-Helmholtz instability initial state, used by `warpSPH.cases.kelvinHelmholtz`.

Builds the classic four-layer shear-layer setup on the unit square: density and
streamwise velocity switch between `rho1`/`v1` (outer bands, `y<1/4` or
`y>3/4`) and `rho2`/`v2` (inner band), smoothed across each interface with an
`exp(.../delta)` ramp, plus a symmetric single-mode transverse velocity
perturbation (`vy`) confined near `y=1/4` and `y=3/4` by Gaussians of width
`sigma`. Density is a sharp step by default; `smoothDensity=True` applies the
same ramp (Frontiere et al. 2017, Eq. 100). Pressure is uniform
at 2.5.
"""

from ....sample import *
import torch
from ....sample.compressible import setupBasicCompressibleInitialState
from ....modules import *
from warpSPHCore import *
from ....modules.timestep.compressible import computeTimestep
import math
import numpy as np

__all__ = ['sampleKHH']


def vy(ri, sigma, freq, w0  ):
        thpt = 1.0/(2.0*sigma*sigma)
        return (w0*torch.sin(freq*np.pi*ri[:, 0]) *
                (torch.exp(-((ri[:, 1] - 0.25)**2 * thpt)) +
                 torch.exp(-((ri[:, 1] - 0.75)**2 * thpt))))*torch.abs(0.5 - ri[:, 1])


def sampleKHH(rho1, rho2, v1, v2, delta, sigma, freq, w0, nx, config, schemeConfig, SimulationState, SimulationSystem,
              smoothDensity=False):
    compressibleSystem = setupBasicCompressibleInitialState(nx, config, schemeConfig, SimulationState, SimulationSystem)

    v_y = vy(compressibleSystem.state.positions, sigma, freq, w0)
    Pinitial = torch.ones_like(compressibleSystem.state.densities) * 2.5

    rhom = (rho1 - rho2) / 2
    vm = (v1 - v2) / 2
    positions = compressibleSystem.state.positions

    region1 = torch.logical_and(positions[:, 1] >= 0, positions[:, 1] < 1/4)
    region2 = torch.logical_and(positions[:, 1] >= 1/4, positions[:, 1] < 1/2)
    region3 = torch.logical_and(positions[:, 1] >= 1/2, positions[:, 1] < 3/4)
    region4 = torch.logical_and(positions[:, 1] >= 3/4, positions[:, 1] <= 1)

    rho = torch.zeros_like(compressibleSystem.state.densities)

    rho[region1] = rho1
    rho[region2] = rho2
    rho[region3] = rho2
    rho[region4] = rho1
    if smoothDensity:
        # Robertson et al. (2010) / Frontiere et al. (2017) Eq. (100): the same exp ramp as v_x, so the
        # contact is smooth over `delta` instead of a sharp step (the default stays sharp: M0 baseline)
        rho[region1] = rho1 - rhom * torch.exp((positions[region1, 1] - 1/4) / delta)
        rho[region2] = rho2 + rhom * torch.exp((1/4 - positions[region2, 1]) / delta)
        rho[region3] = rho2 + rhom * torch.exp((positions[region3, 1] - 3/4) / delta)
        rho[region4] = rho1 - rhom * torch.exp((3/4 - positions[region4, 1]) / delta)

    # print(f'rho1 = {rho1}, rho2 = {rho2}, rhom = {rhom}')
    # print(f'rho min = {rho.min()}, rho max = {rho.max()}, mean = {rho.mean()}')


    v_x = torch.zeros_like(positions[:, 0])
    v_x[region1] = v1 - vm * torch.exp((positions[region1, 1] - 1/4) / delta)
    v_x[region2] = v2 + vm * torch.exp((1/4 - positions[region2, 1]) / delta)
    v_x[region3] = v2 + vm * torch.exp((positions[region3, 1] - 3/4) / delta)
    v_x[region4] = v1 - vm * torch.exp((3/4 - positions[region4, 1]) / delta)

    # print(f'v1 = {v1}, v2 = {v2}, vm = {vm}')
    # print(f'v min = {v_x.min()}, v max = {v_x.max()}, mean = {v_x.mean()}')

    v_initial = torch.stack([v_x, v_y], dim = 1)


    A_, u_, P_, c_s = idealGasEOS(A = None, u = None, P = Pinitial, rho = rho, gamma = schemeConfig.gamma)
    # v_initial = torch.zeros_like(particles_l.positions)

    internalEnergy = u_ 
    kineticEnergy = torch.linalg.norm(v_initial, dim = -1) **2/ 2
    totalEnergy = (internalEnergy + kineticEnergy) * compressibleSystem.state.masses

    compressibleSystem.state.internalEnergies = u_
    compressibleSystem.state.totalEnergies = totalEnergy
    compressibleSystem.state.pressures = P_
    compressibleSystem.state.soundspeeds = c_s
    compressibleSystem.state.velocities = v_initial
    compressibleSystem.state.densities = rho
    compressibleSystem.state.masses = config.dx**config.dim * rho

    # print(f'initial state: rho min = {compressibleSystem.state.densities.min()}, rho max = {compressibleSystem.state.densities.max()}, mean = {compressibleSystem.state.densities.mean()}')


    config.dt = computeTimestep(compressibleSystem, config, schemeConfig, dt = config.dt) * 2/3

    return compressibleSystem
