"""Inutsuka (2002) Godunov SPH step (GODUNOV_SPH_PLAN layer 3): the Monaghan step's adaptive support solve, with the density replaced by Murante's symmetrised
Gaussian sum (and its gradient) and the force / energy rates from the kernel-convolution form of `modules/godunov/wp_inutsuka.py`. No artificial viscosity,
no switch, no conductivity; same system / state / update classes as the Monaghan scheme. Solid walls are not supported.

`compressibleSPH_InutsukaGSPH` is a list of single-purpose regions (one `record_function` each, numbered in call order).
"""

import torch

from ...configurations import SimulationConfig
from ...configurations.gsph import InutsukaGSPHConfig
from ...systems import CompressibleSystem, CompressibleSystemUpdate
from ...modules.adaptiveSupport import solveStageSupport
from ...modules.boundaryConditions import computeForcing, enforceDirichlet, enforceUpdates
from ...modules.eos import idealGasEOS
from ...modules.godunov import computeGaussianDensityWarp, computeInutsukaWarp, godunovReconstruction
from warpSPHCore import OperationProperties, SupportScheme, buildVerletList
from warpSPHCore.profiling import record_function

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

    with record_function("[warpSPH] - [inutsuka - 01] optimal support"):
        solveStageSupport(currentSystem, currentState, config, schemeConfig, computeDensity = True)     # (the Gaussian h below is built from this density)

    with record_function("[warpSPH] - [inutsuka - 02] pair list"):
        # the Gaussian's h: eta times the local particle spacing (Inutsuka 2002 Eq. 81); every Gaussian is cut at 3 sqrt2 h, so the list must reach it
        gaussianH = float(schemeConfig.diffusionParams.gaussianEta) * (currentState.masses / currentState.densities) ** (1.0 / config.dim)
        reach = float((3.0 * 2.0 ** 0.5 * gaussianH / currentState.supports).max())          # (a host read: the list scale is a launch argument)
        adjacency = buildVerletList(
            currentState, config.domain, verletScale = max(config.verletScale, 1.05 * reach),
            supportMode = SupportScheme.SuperSymmetric, priorNeighborhood = currentSystem.adjacency, verbose = False)

    props = OperationProperties(kernel = config.kernel, supportMode = SupportScheme.KernelMeanSymmetric)

    with record_function("[warpSPH] - [inutsuka - 03] Gaussian density"):
        gradRho, rho = computeGaussianDensityWarp(currentState, props, domain = config.domain, adjacency = adjacency, gaussianH = gaussianH)
        currentState.densities = rho

    with record_function("[warpSPH] - [inutsuka - 04] Dirichlet BC"):
        enforceDirichlet(currentSystem, t, dt, config, schemeConfig)

    with record_function("[warpSPH] - [inutsuka - 05] EOS"):
        currentState.entropies, _, currentState.pressures, currentState.soundspeeds = idealGasEOS(
            A = None, u = currentState.internalEnergies, P = None, rho = currentState.densities, gamma = schemeConfig.gamma)

    with record_function("[warpSPH] - [inutsuka - 06] Riemann-state reconstruction inputs"):
        params, order, J, G = godunovReconstruction(currentState, config, schemeConfig, adjacency, alwaysStateGradients = True)

    pairKwargs = dict(domain = config.domain, params = params, gamma = float(schemeConfig.gamma), gradRho = gradRho, order = order,
                      cubic = True, adjacency = adjacency, queryVelocityTensor = J, queryStateGradients = G, gaussianH = gaussianH)

    with record_function("[warpSPH] - [inutsuka - 07] Riemann pair rates"):
        dvdt, dudt = computeInutsukaWarp(currentState, props, **pairKwargs)

    if params.timeCentredEnergy:
        with record_function("[warpSPH] - [inutsuka - 08] time-centred energy rate"):
            # Inutsuka (2002) Eq. 67: the energy rate at the time-centred velocity v_i + a_i dt/2 (a second pass: it needs the accelerations)
            _, dudt = computeInutsukaWarp(currentState, props, **pairKwargs, queryAccelerations = dvdt, halfDt = 0.5 * dt)

    with record_function("[warpSPH] - [inutsuka - 09] drho/dt and divergence"):
        # the density is re-summed (Gaussian) at the start of every step, so the integrators' estimate of d rho / dt is not used: zero
        drhodt = torch.zeros_like(currentState.densities)
        currentState.divergence = torch.zeros_like(currentState.densities)

    with record_function("[warpSPH] - [inutsuka - 10] dE/dt"):
        dEdt = currentState.masses * torch.einsum('ij,ij->i', currentState.velocities, dvdt) + currentState.masses * dudt

    with record_function("[warpSPH] - [inutsuka - 11] forcing"):
        forcing = computeForcing(currentSystem, dt, t, config, schemeConfig)
        dvdt += forcing / currentState.masses.view(-1, 1)

    with record_function("[warpSPH] - [inutsuka - 12] update"):
        update = CompressibleSystemUpdate(
            dxdt = currentState.velocities,
            dvdt = dvdt,
            dudt = dudt,
            drhodt = drhodt,
            dEdt = dEdt,
        )
        enforceUpdates(update, currentSystem, dt, t, config, schemeConfig)
    return update, adjacency, currentState
