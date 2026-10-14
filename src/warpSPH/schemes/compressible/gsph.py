"""Godunov SPH step (GODUNOV_SPH_PLAN layer 2): the Monaghan step's adaptive support solve, density, EOS and continuity, with the pressure force,
the viscosity and the energy equation replaced by the Riemann pair rates of `modules/godunov` (no artificial viscosity, no viscosity switch,
no conductivity). Same system / state / update classes as the Monaghan scheme.

`compressibleSPH_GSPH` is a list of single-purpose regions (one `record_function` each, numbered in call order), in the same shape as the other
compressible steps; the support solve validates the pair list once (`adaptiveSupport.solveStageSupport`).
"""

import torch

from ...configurations import SimulationConfig
from ...configurations.gsph import GSPHConfig
from ...systems import CompressibleSystem, CompressibleSystemUpdate
from ...modules.adaptiveSupport import computeOmega, solveStageSupport
from ...modules.boundaryConditions import computeForcing, enforceDirichlet, enforceUpdates
from ...modules.compressibleWall import beginCompressibleWall, clampSupportsToVolume
from ...modules.density import computeDensities
from ...modules.eos import idealGasEOS
from ...modules.godunov import computeGodunovWarp, godunovReconstruction
from ...modules.momentum import computeMomentumConsistent
from warpSPHCore import GradHState, OperationProperties, SupportScheme
from warpSPHCore.profiling import record_function

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

    with record_function("[warpSPH] - [gsph - 01] wall"):
        wall = beginCompressibleWall(currentState, config)

    with record_function("[warpSPH] - [gsph - 02] optimal support"):
        solveStageSupport(currentSystem, currentState, config, schemeConfig)
        adjacency = currentSystem.adjacency

    with record_function("[warpSPH] - [gsph - 03] support clamp"):
        # only reduces h, so the list stays valid; wall cases only (modules/compressibleWall/supportClamp.py)
        currentState.supports = clampSupportsToVolume(currentState, config, adjacency, wall)

    with record_function("[warpSPH] - [gsph - 04] density"):
        currentState.densities = computeDensities(currentState, config, schemeConfig, adjacency, supportMode = config.supportMode)

    if wall is not None:
        with record_function("[warpSPH] - [gsph - 05] wall closure"):
            wall.apply(currentState, config, schemeConfig, adjacency)
            currentState.densities = computeDensities(currentState, config, schemeConfig, adjacency, supportMode = config.supportMode)
            wall.apply(currentState, config, schemeConfig, adjacency)

    with record_function("[warpSPH] - [gsph - 06] Dirichlet BC"):
        enforceDirichlet(currentSystem, t, dt, config, schemeConfig)

    with record_function("[warpSPH] - [gsph - 07] EOS"):
        currentState.entropies, _, currentState.pressures, currentState.soundspeeds = idealGasEOS(
            A = None, u = currentState.internalEnergies, P = None, rho = currentState.densities, gamma = schemeConfig.gamma)

    with record_function("[warpSPH] - [gsph - 08] grad-h terms"):
        if schemeConfig.adaptiveSupportCorrections:
            omega = computeOmega(
                currentState,
                OperationProperties(kernel = config.kernel, supportMode = SupportScheme.Gather),
                domain = config.domain, adjacency = adjacency)
            gradHState = GradHState(queryOmegas = omega)
        else:
            gradHState = None

    with record_function("[warpSPH] - [gsph - 09] Riemann-state reconstruction inputs"):
        # first order uses the particles' own values; second order needs the velocity Jacobian and, for rho and P, their gradients
        params, order, J, G = godunovReconstruction(currentState, config, schemeConfig, adjacency)

    pairKwargs = dict(domain = config.domain, params = params, gamma = float(schemeConfig.gamma), order = order, adjacency = adjacency,
                      queryVelocityTensor = J, queryStateGradients = G)
    pairProperties = OperationProperties(kernel = config.kernel, supportMode = SupportScheme.KernelMeanSymmetric)

    with record_function("[warpSPH] - [gsph - 10] Riemann pair rates"):
        dvdt, dudt = computeGodunovWarp(currentState, pairProperties, **pairKwargs)

    if params.timeCentredEnergy:
        with record_function("[warpSPH] - [gsph - 11] time-centred energy rate"):
            # Inutsuka (2002) Eq. 67: the energy rate at the time-centred velocity v_i + a_i dt/2, which needs the accelerations of a first pass
            _, dudt = computeGodunovWarp(currentState, pairProperties, **pairKwargs, queryAccelerations = dvdt, halfDt = 0.5 * dt)

    with record_function("[warpSPH] - [gsph - 12] drho/dt"):
        drhodt = computeMomentumConsistent(
            currentState, config, schemeConfig = schemeConfig, adjacency = adjacency, gradH = gradHState)

    with record_function("[warpSPH] - [gsph - 13] divergence"):
        currentState.divergence = -drhodt / currentState.densities

    with record_function("[warpSPH] - [gsph - 14] dE/dt"):
        dEdt = currentState.masses * torch.einsum('ij,ij->i', currentState.velocities, dvdt) + currentState.masses * dudt

    with record_function("[warpSPH] - [gsph - 15] forcing"):
        forcing = computeForcing(currentSystem, dt, t, config, schemeConfig)
        dvdt += forcing / currentState.masses.view(-1, 1)

    with record_function("[warpSPH] - [gsph - 16] update"):
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
