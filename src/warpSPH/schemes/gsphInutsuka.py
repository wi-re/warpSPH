"""Inutsuka (2002) Godunov SPH step (GODUNOV_SPH_PLAN layer 3): the Monaghan step's adaptive support solve, with the density replaced by Murante's symmetrised
Gaussian sum (and its gradient) and the force / energy rates from the kernel-convolution form of `modules/godunov/wp_inutsuka.py`. No artificial viscosity,
no switch, no conductivity; same system / state / update classes as the Monaghan scheme. Solid walls are not supported.
"""

import torch

from ..configurations import SimulationConfig
from ..configurations.gsph import InutsukaGSPHConfig
from ..configurations.moduleConfigurations.diffusionParameters import VelocityPairPolicy, resolveReconstructionLimiter
from ..systems import CompressibleSystem, CompressibleSystemUpdate
from ..modules.adaptiveSupport import evaluateOptimalSupport
from ..modules.boundaryConditions import computeForcing, enforceDirichlet, enforceUpdates
from ..modules.eos import idealGasEOS
from ..modules.godunov import computeGaussianDensityWarp, computeInutsukaWarp
from ..modules.reconstruction import computeStateGradients, computeVelocityJacobian
from warpSPHCore import OperationProperties, SupportScheme, buildVerletList

__all__ = ['compressibleSPH_InutsukaGSPH']



def compressibleSPH_InutsukaGSPH(
    system: CompressibleSystem,
    dt: float,
    config: SimulationConfig,
    schemeConfig: InutsukaGSPHConfig,
    verbose = False,
):
    currentSystem = system
    currentState = currentSystem.state
    t = currentSystem.t

    rho_optimal, h_optimal, currentSystem.adjacency, *_ = evaluateOptimalSupport(
        currentState, config, schemeConfig, SupportScheme.Gather, currentSystem.adjacency)
    currentState.supports = h_optimal
    currentState.densities = rho_optimal

    # the Gaussian's h: eta times the local particle spacing (Inutsuka 2002 Eq. 81); every Gaussian is cut at 3 sqrt2 h, so the list must reach it
    gaussianH = float(schemeConfig.diffusionParams.gaussianEta) * (currentState.masses / currentState.densities) ** (1.0 / config.dim)
    reach = float((3.0 * 2.0 ** 0.5 * gaussianH / currentState.supports).max())
    adjacency = buildVerletList(
        currentState, config.domain, verletScale = max(config.verletScale, 1.05 * reach),
        supportMode = SupportScheme.SuperSymmetric, priorNeighborhood = currentSystem.adjacency, verbose = False)

    props = OperationProperties(kernel = config.kernel, supportMode = SupportScheme.KernelMeanSymmetric)
    gradRho, rho = computeGaussianDensityWarp(currentState, props, domain = config.domain, adjacency = adjacency, gaussianH = gaussianH)
    currentState.densities = rho

    enforceDirichlet(currentSystem, t, dt, config, schemeConfig)
    currentState.entropies, _, currentState.pressures, currentState.soundspeeds = idealGasEOS(
        A = None, u = currentState.internalEnergies, P = None, rho = currentState.densities, gamma = schemeConfig.gamma)

    params = resolveReconstructionLimiter(schemeConfig.diffusionParams, config.n_h)
    order = 1 if params.velocityPairPolicy == VelocityPairPolicy.Raw.value else 2
    J = G = None
    if order == 2:
        J = computeVelocityJacobian(currentState, config, SupportScheme.SuperSymmetric, adjacency,
                                    corrected = params.correctReconstructionGradient)
        G = computeStateGradients(currentState, config, SupportScheme.SuperSymmetric, adjacency,
                                  corrected = params.correctReconstructionGradient)

    dvdt, dudt = computeInutsukaWarp(
        currentState, props, domain = config.domain, params = params, gamma = float(schemeConfig.gamma), gradRho = gradRho,
        order = order, cubic = True, adjacency = adjacency, queryVelocityTensor = J, queryStateGradients = G, gaussianH = gaussianH)
    if params.timeCentredEnergy:
        # Inutsuka (2002) Eq. 67: the energy rate at the time-centred velocity v_i + a_i dt/2 (a second pass: it needs the accelerations)
        _, dudt = computeInutsukaWarp(
            currentState, props, domain = config.domain, params = params, gamma = float(schemeConfig.gamma), gradRho = gradRho,
            order = order, cubic = True, adjacency = adjacency, queryVelocityTensor = J, queryStateGradients = G,
            queryAccelerations = dvdt, halfDt = 0.5 * dt, gaussianH = gaussianH)

    # the density is re-summed (Gaussian) at the start of every step, so the integrators' estimate of d rho / dt is not used: zero
    drhodt = torch.zeros_like(currentState.densities)
    currentState.divergence = torch.zeros_like(currentState.densities)

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
    return update, adjacency, currentState
