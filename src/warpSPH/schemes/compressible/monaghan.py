"""The classic Monaghan (1992-style) compressible SPH step: adaptive support
solve, super-symmetric adjacency, Gather-mode density, ideal-gas EOS, optional
grad-h corrections (`GradHState`), symmetric pressure force, Monaghan dudt,
and separate artificial-viscosity/conductivity/thermal-dissipation terms
added on top.

`compressibleSPH_Monaghan` is a list of single-purpose regions (one `record_function` each, numbered in call order),
in the same shape as `compSPH_step`: every region does one thing, so a profile or a call audit names it. The support
solve validates the pair list once (`adaptiveSupport.solveStageSupport`), so the step needs no second check of its own.
"""

from ...configurations import SimulationConfig, CompressibleSPHConfig
from ...modules.adaptiveSupport import computeOmega, solveStageSupport
from ...modules.boundaryConditions import computeForcing, enforceDirichlet, enforceUpdates
from ...modules.compressibleWall import beginCompressibleWall
from ...modules.density import computeDensities
from ...modules.dissipation import computeConductivity, computeThermalDissipation, computeViscosity
from ...modules.eos import idealGasEOS
from ...modules.internalEnergy import computeDudtMonaghan
from ...modules.momentum import computeMomentumConsistent
from ...modules.pressure import computePressureForceSymmetric, computePerSidePressureWarp
from ...modules.reconstruction import reconstructionInputs, stateGradientInputs
from ...modules.shockCapturing import computeViscositySwitchTerms, updateViscositySwitch
from ...systems import CompressibleSystem, CompressibleSystemUpdate
from warpSPHCore import GradHState, OperationProperties, SupportScheme
from warpSPHCore.profiling import record_function
import torch

__all__ = ['compressibleSPH_Monaghan']


def _dissipationOperator(config):
    """The properties every dissipation term is evaluated with."""
    return OperationProperties(kernel = config.kernel, supportMode = SupportScheme.KernelMeanSymmetric)


def compressibleSPH_Monaghan(
    system: CompressibleSystem,
    dt: float,
    config: SimulationConfig,
    schemeConfig: CompressibleSPHConfig,
    verbose = False,
):
    currentSystem = system
    currentState = currentSystem.state
    t = currentSystem.t

    with record_function("[warpSPH] - [monaghan - 01] wall"):
        wall = beginCompressibleWall(currentState, config)

    with record_function("[warpSPH] - [monaghan - 02] optimal support"):
        solveStageSupport(currentSystem, currentState, config, schemeConfig)
        adjacency = currentSystem.adjacency

    with record_function("[warpSPH] - [monaghan - 03] density"):
        currentState.densities = computeDensities(
            currentState, config, schemeConfig, adjacency, supportMode = config.supportMode)

    if wall is not None:
        with record_function("[warpSPH] - [monaghan - 04] wall closure"):
            wall.apply(currentState, config, schemeConfig, adjacency)
            # the wall masses just changed, so the fluid density sum has to be redone
            currentState.densities = computeDensities(
                currentState, config, schemeConfig, adjacency, supportMode = config.supportMode)
            wall.apply(currentState, config, schemeConfig, adjacency)

    with record_function("[warpSPH] - [monaghan - 05] Dirichlet BC"):
        enforceDirichlet(currentSystem, t, dt, config, schemeConfig)

    with record_function("[warpSPH] - [monaghan - 06] EOS"):
        currentState.entropies, _, currentState.pressures, currentState.soundspeeds = idealGasEOS(
            A = None,
            u = currentState.internalEnergies,
            P = None,
            rho = currentState.densities,
            gamma = schemeConfig.gamma,
        )

    with record_function("[warpSPH] - [monaghan - 07] grad-h terms"):
        if schemeConfig.adaptiveSupportCorrections:
            omega = computeOmega(currentState,
                    OperationProperties(kernel = config.kernel, supportMode = SupportScheme.Gather),
                    domain = config.domain,
                    adjacency = adjacency)
            gradHState = GradHState(queryOmegas = omega)
        else:
            gradHState = None

    with record_function("[warpSPH] - [monaghan - 08] viscosity-switch terms"):
        # Per-particle alphas from the current state; the wrapper passes the stored alphas through for the NoneSwitch baseline.
        currentState.alphas, switchState = computeViscositySwitchTerms(
            dt,
            currentState,
            config, schemeConfig,
            SupportScheme.SuperSymmetric,
            adjacency)

    if getattr(schemeConfig, 'pressureFormulation', 'meanKernel') == 'perSide':
        with record_function("[warpSPH] - [monaghan - 09] pressure force + dudt (per side)"):
            # Price 2012 Eqs. 43-45: each side's own h, the conjugate pdV work; Omega only with grad-h on
            dvdt, dudt = computePerSidePressureWarp(
                currentState,
                OperationProperties(kernel = config.kernel, supportMode = SupportScheme.KernelMeanSymmetric),
                domain = config.domain,
                adjacency = adjacency,
                queryOmegas = gradHState.queryOmegas if gradHState is not None else None,
            )
    else:
        with record_function("[warpSPH] - [monaghan - 09] pressure force"):
            dvdt = computePressureForceSymmetric(
                currentState,
                config,
                supportScheme = SupportScheme.KernelMeanSymmetric,
                adjacency = adjacency,
                gradH = gradHState
            )
        with record_function("[warpSPH] - [monaghan - 10] dudt"):
            dudt = computeDudtMonaghan(
                currentState,
                config,
                supportScheme = SupportScheme.KernelMeanSymmetric,
                adjacency = adjacency,
                gradH = gradHState
            )

    with record_function("[warpSPH] - [monaghan - 11] drho/dt"):
        drhodt = computeMomentumConsistent(
            currentState,
            config,
            schemeConfig = schemeConfig,
            adjacency = adjacency,
            gradH = gradHState
        )

    with record_function("[warpSPH] - [monaghan - 12] pair-velocity reconstruction inputs"):
        # The pair velocity the viscosity sees (AV_PLAN Phases 3-4): the raw path needs nothing more; a reconstructing
        # policy needs the velocity Jacobian and the n_h-derived limiter constants, the Balsara variants the factor B
        diffusionParams = schemeConfig.diffusionParams
        diffusionParams, velocityTensor, balsara = reconstructionInputs(currentState, config, diffusionParams, adjacency)
        # the Riemann dissipation's reconstructed (rho, P) states (GODUNOV_SPH_PLAN layer 1), when asked for
        diffusionParams, stateGradients = stateGradientInputs(currentState, config, diffusionParams, adjacency)

    with record_function("[warpSPH] - [monaghan - 13] artificial viscosity"):
        dvdt_diss = computeViscosity(
            currentState,
            operationProperties = _dissipationOperator(config),
            domain = config.domain,
            adjacency = adjacency,
            viscosityParams = diffusionParams,
            queryAlphas = currentState.alphas,
            queryVelocityTensor = velocityTensor,
            queryBalsara = balsara,
            queryStateGradients = stateGradients,
        )

    with record_function("[warpSPH] - [monaghan - 14] artificial conductivity"):
        dudt_diss = computeConductivity(
            currentState,
            operationProperties = _dissipationOperator(config),
            domain = config.domain,
            adjacency = adjacency,
            conductivityParams = diffusionParams,
            queryAlphas = currentState.alphas,
        )

    with record_function("[warpSPH] - [monaghan - 15] thermal dissipation"):
        dudt_thermal = computeThermalDissipation(
            currentState,
            operationProperties = _dissipationOperator(config),
            domain = config.domain,
            adjacency = adjacency,
            conductivityParams = diffusionParams,
            queryAlphas = currentState.alphas,
            queryVelocityTensor = velocityTensor,
            queryBalsara = balsara,
            queryStateGradients = stateGradients,
        )

    with record_function("[warpSPH] - [monaghan - 16] viscosity-switch update"):
        # Advance the switch's stored alpha0; the hydrodynamic acceleration (pressure + viscosity, pre-forcing) is what the
        # switch's second-order-divergence estimate needs. A no-op for the NoneSwitch baseline.
        currentState.alpha0s, switchState = updateViscositySwitch(
            switchState,
            dt, dvdt + dvdt_diss,
            currentState,
            config, schemeConfig,
            SupportScheme.SuperSymmetric,
            adjacency)

    with record_function("[warpSPH] - [monaghan - 17] divergence"):
        # stored for the next step's second-order-divergence finite difference
        currentState.divergence = -drhodt / currentState.densities

    with record_function("[warpSPH] - [monaghan - 18] entropy dissipation"):
        # ReadHayfield2012 internal-energy rate (eqs. 33-35): only that switch populates `switchState.dudt_diss`; the
        # others leave it None (or the switch state is itself None), so this collapses to zero for them.
        if switchState is not None and getattr(switchState, 'dudt_diss', None) is not None:
            dudt_entropy = switchState.dudt_diss
        else:
            dudt_entropy = torch.zeros_like(dudt)

    with record_function("[warpSPH] - [monaghan - 19] dE/dt"):
        dEdt = currentState.masses * torch.einsum('ij,ij->i', currentState.velocities, (dvdt + dvdt_diss)) + currentState.masses * (dudt + dudt_diss + dudt_entropy)

    with record_function("[warpSPH] - [monaghan - 20] forcing"):
        forcing = computeForcing(currentSystem, dt, t, config, schemeConfig)
        dvdt += forcing / currentState.masses.view(-1,1)

    with record_function("[warpSPH] - [monaghan - 21] update"):
        update = CompressibleSystemUpdate(
            dxdt = currentState.velocities,
            dvdt = dvdt + dvdt_diss,
            dudt = dudt + dudt_diss + dudt_thermal + dudt_entropy,
            drhodt = drhodt,
            dEdt = dEdt,
        )
        enforceUpdates(update, currentSystem, dt, t, config, schemeConfig)
        if wall is not None:
            wall.finishUpdate(update)

    return update, adjacency, currentState
