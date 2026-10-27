"""Hydrostatic-equilibrium initial state, used by `warpSPH.cases.compressible.hydrostatic`.

Samples a regular lattice, assigns `rho_high` to a dense square in the domain
center (`|x|, |y| < L/4`) and `rho_low` elsewhere, then re-derives an optimal
support/density via `evaluateOptimalSupport` and CRK renormalization before
setting a uniform initial pressure of 1 through the scalar or material EOS.
Optional elliptical geometry and common translation extend the same contact
control; the legacy square and zero velocity remain the defaults.
"""

from warpSPH.configurations.compute import neighborSearchOptions

from ....sample import *
import torch
from ....sample.compressible import setupBasicCompressibleInitialState
from ....modules import *
from warpSPHCore import *

__all__ = ['buildHydrostaticInitialState']


def buildHydrostaticInitialState(rho_low, rho_high, nx, config, schemeConfig, SimulationState, SimulationSystem,
                               *, shape='box', boost=0.0):
    compressibleSystem = setupBasicCompressibleInitialState(nx, config, schemeConfig, SimulationState, SimulationSystem)
    L = config.domain.max[0] - config.domain.min[0]
    dim = config.dim

    area = L**dim / (nx**dim)

    mask = torch.logical_and(compressibleSystem.state.positions[:,0].abs() < L/4, compressibleSystem.state.positions[:,1].abs() < L/4)
    if shape == 'ellipse':
        mask = (compressibleSystem.state.positions[:, 0] / (L/4)).square() + (compressibleSystem.state.positions[:, 1] / (L/8)).square() < 1
    elif shape != 'box':
        raise ValueError('shape must be box or ellipse')
    if schemeConfig.materials is not None:
        if len(schemeConfig.materials) != 2:
            raise ValueError('hydrostatic requires exactly two EOS entries when materials is configured')
        compressibleSystem.state.materials = mask.to(torch.int32)

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
        config.domain,
        verletScale=1.4,
        supportMode=SupportScheme.SuperSymmetric,
        priorNeighborhood=None,
        verbose=False,
        **neighborSearchOptions(config))

    if getattr(schemeConfig, 'crkDensityPolicy', 'legacy') == 'legacy':
        apparentVolume, compressibleSystem.state.densities, crkState = computeCRKFactors(compressibleSystem.state, config.domain, config.kernel, adjacency=adjacency)
    else:
        from ....modules.crk import crkFactors
        apparentVolume, compressibleSystem.state.densities, crkState = crkFactors(
            compressibleSystem.state, config.domain, config.kernel, adjacency,
            densityPolicy=schemeConfig.crkDensityPolicy,
            volumes=torch.full_like(compressibleSystem.state.masses, area) if schemeConfig.crkEvolution == 'spheral' else None)
        if schemeConfig.crkEvolution == 'spheral':
            compressibleSystem.state.crkVolumes = apparentVolume
            compressibleSystem.state.crkMassDensities = compressibleSystem.state.densities.clone()


    P_initial = torch.ones_like(compressibleSystem.state.densities)
    # u = 1 / (gamma - 1) * (P_initial / simulationState_.densities)
    # A_, u_, P_, c_s = idealGasEOS(A = None, u = None, P = P_initial, rho = rho_optimal, gamma = gamma)
    A_, u_, P_, c_s = evaluateEOS(compressibleSystem.state.densities, compressibleSystem.state.materials,
                                 schemeConfig, P=P_initial)
    compressibleSystem.state.internalEnergies = u_
    compressibleSystem.state.pressures = P_
    compressibleSystem.state.entropies = A_
    compressibleSystem.state.soundspeeds = c_s
    compressibleSystem.state.velocities[:, 0] = boost
    compressibleSystem.state.totalEnergies = compressibleSystem.state.masses * (u_ + .5 * boost**2)

    return compressibleSystem
