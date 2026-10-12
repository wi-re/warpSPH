"""Hydrostatic-equilibrium initial state, used by `warpSPH.cases.compressible.hydrostatic`.

Samples a regular lattice, assigns `rho_high` to a dense square in the domain
center (`|x|, |y| < L/4`) and `rho_low` elsewhere, then re-derives an optimal
support/density via `evaluateOptimalSupport` and CRK renormalization before
setting a uniform initial pressure of 1 through `idealGasEOS`. Velocities are
left at whatever `setupBasicCompressibleInitialState` set (zero), which is the
point of the test: the exact solution never moves.
"""

from ....sample import *
import torch
from ....sample.compressible import setupBasicCompressibleInitialState
from ....modules import *
from warpSPHCore import *

__all__ = ['buildHydrostaticInitialState']


def buildHydrostaticInitialState(rho_low, rho_high, nx, config, schemeConfig, SimulationState, SimulationSystem):
    compressibleSystem = setupBasicCompressibleInitialState(nx, config, schemeConfig, SimulationState, SimulationSystem)
    L = config.domain.max[0] - config.domain.min[0]
    dim = config.dim

    area = L**dim / (nx**dim)

    mask = torch.logical_and(compressibleSystem.state.positions[:,0].abs() < L/4, compressibleSystem.state.positions[:,1].abs() < L/4)

    compressibleSystem.state.masses[mask] = area * rho_high
    compressibleSystem.state.densities[mask] = rho_high
    compressibleSystem.state.masses[~mask] = area * rho_low
    compressibleSystem.state.densities[~mask] = rho_low

    schemeConfig.adaptiveSupportIterations = 16

    rho_optimal, h_optimal, adjacency, rhos_iter, supports_iter = evaluateOptimalSupport(compressibleSystem.state, config, supportScheme = SupportScheme.Gather, compParams = schemeConfig)
    compressibleSystem.state.supports = h_optimal
    schemeConfig.adaptiveSupportIterations = 1

    adjacency = buildVerletList(
        compressibleSystem.state, 
        config.domain, verletScale = 1.4, supportMode = SupportScheme.SuperSymmetric,
        priorNeighborhood = None,
        verbose = False)

    apparentVolume, compressibleSystem.state.densities, crkState = computeCRKFactors(compressibleSystem.state, config.domain, config.kernel, adjacency = adjacency)


    P_initial = torch.ones_like(compressibleSystem.state.densities)
    # u = 1 / (gamma - 1) * (P_initial / simulationState_.densities)
    # A_, u_, P_, c_s = idealGasEOS(A = None, u = None, P = P_initial, rho = rho_optimal, gamma = gamma)
    A_, u_, P_, c_s = idealGasEOS(A = None, u = None, P = P_initial, rho = compressibleSystem.state.densities, gamma = schemeConfig.gamma)
    compressibleSystem.state.internalEnergies = u_
    compressibleSystem.state.presses = P_
    compressibleSystem.state.soundspeeds = c_s

    return compressibleSystem