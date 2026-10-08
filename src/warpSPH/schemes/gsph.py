"""Godunov SPH step (GODUNOV_SPH_PLAN layer 2): the Monaghan step's adaptive support solve, density, EOS and continuity, with the pressure force,
the viscosity and the energy equation replaced by the Riemann pair rates of `modules/godunov` (no artificial viscosity, no viscosity switch,
no conductivity). Same system / state / update classes as the Monaghan scheme.
"""

import torch

from ..configurations import SimulationConfig
from ..configurations.gsph import GSPHConfig
from ..configurations.moduleConfigurations.diffusionParameters import VelocityPairPolicy, resolveReconstructionLimiter
from ..systems import CompressibleSystem, CompressibleSystemUpdate
from ..modules.adaptiveSupport import computeOmega, evaluateOptimalSupport
from ..modules.boundaryConditions import computeForcing, enforceDirichlet, enforceUpdates
from ..modules.compressibleWall import beginCompressibleWall
from ..modules.density import computeDensities
from ..modules.eos import idealGasEOS
from ..modules.godunov import computeGodunovWarp
from ..modules.momentum import computeMomentumConsistent
from ..modules.reconstruction import computeStateGradients, computeVelocityJacobian
from warpSPHCore import GradHState, OperationProperties, SupportScheme, buildVerletList

__all__ = ['compressibleSPH_GSPH']


def compressibleSPH_GSPH(
    system: CompressibleSystem,
    dt: float,
    config: SimulationConfig,
    schemeConfig: GSPHConfig,
    verbose = False,
):
    currentSystem = system
    currentState = currentSystem.state
    t = currentSystem.t
    wall = beginCompressibleWall(currentState, config)

    rho_optimal, h_optimal, currentSystem.adjacency, *_ = evaluateOptimalSupport(
        currentState, config, schemeConfig, SupportScheme.Gather, currentSystem.adjacency)
    currentState.supports = h_optimal
    currentState.densities = rho_optimal

    adjacency = buildVerletList(
        currentState, config.domain, verletScale = config.verletScale, supportMode = SupportScheme.SuperSymmetric,
        priorNeighborhood = currentSystem.adjacency, verbose = False)

    currentState.densities = computeDensities(currentState, config, schemeConfig, adjacency, supportMode = config.supportMode)
    if wall is not None:
        wall.apply(currentState, config, schemeConfig, adjacency)
        currentState.densities = computeDensities(currentState, config, schemeConfig, adjacency, supportMode = config.supportMode)
        wall.apply(currentState, config, schemeConfig, adjacency)

    enforceDirichlet(currentSystem, t, dt, config, schemeConfig)
    currentState.entropies, _, currentState.pressures, currentState.soundspeeds = idealGasEOS(
        A = None, u = currentState.internalEnergies, P = None, rho = currentState.densities, gamma = schemeConfig.gamma)

    if schemeConfig.adaptiveSupportCorrections:
        omega = computeOmega(
            currentState,
            OperationProperties(kernel = config.kernel, supportMode = SupportScheme.Gather),
            domain = config.domain, adjacency = adjacency)
        gradHState = GradHState(queryOmegas = omega)
    else:
        gradHState = None

    # the Riemann states: first order uses the particles' own values; second order needs the velocity Jacobian and, for rho and P, their gradients
    params = resolveReconstructionLimiter(schemeConfig.diffusionParams, config.n_h)
    order = 1 if params.velocityPairPolicy == VelocityPairPolicy.Raw.value else 2
    J = G = None
    if order == 2:
        J = computeVelocityJacobian(currentState, config, SupportScheme.SuperSymmetric, adjacency,
                                    corrected = params.correctReconstructionGradient)
        if params.riemannReconstruction:
            G = computeStateGradients(currentState, config, SupportScheme.SuperSymmetric, adjacency,
                                      corrected = params.correctReconstructionGradient)

    dvdt, dudt = computeGodunovWarp(
        currentState,
        OperationProperties(kernel = config.kernel, supportMode = SupportScheme.KernelMeanSymmetric),
        domain = config.domain,
        params = params, gamma = float(schemeConfig.gamma), order = order,
        adjacency = adjacency,
        queryVelocityTensor = J, queryStateGradients = G)

    drhodt = computeMomentumConsistent(
        currentState, config, schemeConfig = schemeConfig, adjacency = adjacency, gradH = gradHState)
    currentState.divergence = -drhodt / currentState.densities

    dEdt = currentState.masses * torch.einsum('ij,ij->i', currentState.velocities, dvdt) + currentState.masses * dudt

    forcing = computeForcing(currentSystem, dt, t, config, schemeConfig)
    dvdt += forcing / currentState.masses.view(-1, 1)

    update = CompressibleSystemUpdate(
        dxdt = currentState.velocities,
        dvdt = dvdt,
        dudt = dudt,
        drhodt = drhodt,
        dEdt = dEdt,
    )
    enforceUpdates(update, currentSystem, dt, t, config, schemeConfig)
    if wall is not None:
        wall.finishUpdate(update)
    return update, adjacency, currentState
