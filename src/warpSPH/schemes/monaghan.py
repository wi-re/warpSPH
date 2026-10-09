"""The classic Monaghan (1992-style) compressible SPH step: adaptive support
solve, super-symmetric adjacency, Gather-mode density, ideal-gas EOS, optional
grad-h corrections (`GradHState`), symmetric pressure force, Monaghan dudt,
and separate artificial-viscosity/conductivity/thermal-dissipation terms
added on top.
"""

# from warpSPH.modules import evaluateOptimalSupport, idealGasEOS, computeOmega
# from warpSPHCore import SupportScheme
# from warpSPH.modules import computePressureForceSymmetric, computeDudtMonaghan, computeMomentumConsistent
# from warpSPH.modules import computeViscosity, computeConductivity, computeThermalDissipation
from ..configurations.moduleConfigurations.diffusionParameters import DiffusionParameters, ViscosityTerms
from ..systems import CompressibleSystem, CompressibleSystemUpdate
from ..configurations import SimulationConfig, CompressibleSPHConfig
import torch

from ..modules.adaptiveSupport import computeOmega, evaluateOptimalSupport
from ..modules.boundaryConditions import computeForcing, enforceDirichlet, enforceUpdates
from ..modules.compressibleWall import beginCompressibleWall
from ..modules.density import computeDensities
from ..modules.dissipation import computeConductivity, computeThermalDissipation, computeViscosity
from ..modules.eos import idealGasEOS
from ..modules.internalEnergy import computeDudtMonaghan
from ..modules.momentum import computeMomentumConsistent
from ..modules.pressure import computePressureForceSymmetric, computePerSidePressureWarp
from ..modules.reconstruction import reconstructionInputs, stateGradientInputs
from ..modules.shockCapturing import computeViscositySwitchTerms, updateViscositySwitch
from warpSPHCore import (
    GradHState, OperationProperties, SupportScheme, buildVerletList,
)

__all__ = ['compressibleSPH_Monaghan']


def compressibleSPH_Monaghan(
    system: CompressibleSystem,
    dt: float,
    config: SimulationConfig,
    schemeConfig: CompressibleSPHConfig,
    verbose = False,
):
    currentSystem = system#.initializeNewState()
    currentState = currentSystem.state
    t = currentSystem.t
    wall = beginCompressibleWall(currentState, config)

    rho_optimal, h_optimal, currentSystem.adjacency, *_ = evaluateOptimalSupport(currentState, config, schemeConfig, SupportScheme.Gather, currentSystem.adjacency)
    currentState.supports = h_optimal
    currentState.densities = rho_optimal

    # verletScale = 2 ** (1/config.dim)
    # verletScale = 1
    verletScale = config.verletScale

    adjacency = buildVerletList(
        currentState, 
        config.domain, verletScale = verletScale, supportMode = SupportScheme.SuperSymmetric,
        priorNeighborhood = currentSystem.adjacency,
        verbose = False)

    numNeighbors = adjacency.numNeighbors

    currentState.densities = computeDensities(
        currentState, config, schemeConfig, adjacency,
        supportMode = config.supportMode)
    if wall is not None:
        wall.apply(currentState, config, schemeConfig, adjacency)
        # the wall masses just changed, so the fluid density sum has to be redone
        currentState.densities = computeDensities(
            currentState, config, schemeConfig, adjacency,
            supportMode = config.supportMode)
        wall.apply(currentState, config, schemeConfig, adjacency)

    enforceDirichlet(currentSystem, t, dt, config, schemeConfig)
    currentState.entropies, _, currentState.pressures, currentState.soundspeeds = idealGasEOS(
        A = None,
        u = currentState.internalEnergies,
        P = None,
        rho = currentState.densities,
        gamma = schemeConfig.gamma,
    )

    if schemeConfig.adaptiveSupportCorrections:
        omega = computeOmega(currentState, 
                OperationProperties(
                    kernel = config.kernel,
                    supportMode = SupportScheme.Gather,
                ),
                domain = config.domain,
                adjacency = adjacency
        )

        gradHState = GradHState(
            queryOmegas = omega
        )
    else:
        gradHState = None

    # Viscosity switch (Cullen-Dehnen / Hopkins / none). Computes the
    # per-particle alphas from the current state; the wrapper is a no-op that
    # passes the stored alphas through for the NoneSwitch baseline.
    currentState.alphas, switchState = computeViscositySwitchTerms(
        dt,
        currentState,
        config, schemeConfig,
        SupportScheme.SuperSymmetric,
        adjacency)

    if getattr(schemeConfig, 'pressureFormulation', 'meanKernel') == 'perSide':
        # Price 2012 Eqs. 43-45: each side's own h, the conjugate pdV work; Omega only with grad-h on
        dvdt, dudt = computePerSidePressureWarp(
            currentState,
            OperationProperties(kernel = config.kernel, supportMode = SupportScheme.KernelMeanSymmetric),
            domain = config.domain,
            adjacency = adjacency,
            queryOmegas = gradHState.queryOmegas if gradHState is not None else None,
        )
    else:
        dvdt = computePressureForceSymmetric(
            currentState,
            config,
            supportScheme = SupportScheme.KernelMeanSymmetric,
            adjacency = adjacency,
            gradH = gradHState
        )

        dudt = computeDudtMonaghan(
            currentState,
            config,
            supportScheme = SupportScheme.KernelMeanSymmetric,
            adjacency = adjacency,
            gradH = gradHState
        )

    drhodt = computeMomentumConsistent(
        currentState,
        config,
        schemeConfig = schemeConfig,
        adjacency = adjacency,
        gradH = gradHState
    )


    diffusionParams = schemeConfig.diffusionParams
    # The pair velocity the viscosity sees (AV_PLAN Phases 3-4): the raw path needs nothing more; a reconstructing
    # policy needs the velocity Jacobian and the n_h-derived limiter constants, the Balsara variants the factor B
    diffusionParams, velocityTensor, balsara = reconstructionInputs(currentState, config, diffusionParams, adjacency)
    # the Riemann dissipation's reconstructed (rho, P) states (GODUNOV_SPH_PLAN layer 1), when asked for
    diffusionParams, stateGradients = stateGradientInputs(currentState, config, diffusionParams, adjacency)
    dvdt_diss = computeViscosity(
        currentState,
        # queryVelocities=currentState.velocities,
        operationProperties = OperationProperties(
            kernel = config.kernel,
            supportMode = SupportScheme.KernelMeanSymmetric,
        ),
        domain = config.domain,
        adjacency = adjacency,
        viscosityParams = diffusionParams,
        queryAlphas = currentState.alphas,
        queryVelocityTensor = velocityTensor,
        queryBalsara = balsara,
        queryStateGradients = stateGradients,
    )


    dudt_diss = computeConductivity(
        currentState,
        # queryVelocities=currentState.velocities,
        operationProperties = OperationProperties(
            kernel = config.kernel,
            supportMode = SupportScheme.KernelMeanSymmetric,
        ),
        domain = config.domain,
        adjacency = adjacency,
        conductivityParams = diffusionParams,
        queryAlphas = currentState.alphas,
    )


    dudt_thermal = computeThermalDissipation(
        currentState,
        # queryVelocities=currentState.velocities,
        operationProperties = OperationProperties(
            kernel = config.kernel,
            supportMode = SupportScheme.KernelMeanSymmetric,
        ),
        domain = config.domain,
        adjacency = adjacency,
        conductivityParams = diffusionParams,
        queryAlphas = currentState.alphas,
        queryVelocityTensor = velocityTensor,
        queryBalsara = balsara,
        queryStateGradients = stateGradients,
    )

    # Advance the viscosity switch's stored alpha0 and store the velocity
    # divergence for the next step's second-order-divergence finite
    # difference. Mirrors the compSPH wiring; for the NoneSwitch baseline the
    # wrapper is a no-op (alpha0 passes through unchanged). The hydrodynamic
    # acceleration (pressure + viscosity, pre-forcing) is what the switch's
    # second-order-divergence estimate needs.
    currentState.alpha0s, switchState = updateViscositySwitch(
        switchState,
        dt, dvdt + dvdt_diss,
        currentState,
        config, schemeConfig,
        SupportScheme.SuperSymmetric,
        adjacency)
    currentState.divergence = -drhodt / currentState.densities

    # ReadHayfield2012 entropy-dissipation internal-energy rate (eqs. 33-35).
    # Only the R&H switch populates `switchState.dudt_diss`; the Cullen-Dehnen,
    # Hopkins and NoneSwitch baselines leave it `None` (or the switch state is
    # itself `None`), so this collapses to zero for them.
    if switchState is not None and getattr(switchState, 'dudt_diss', None) is not None:
        dudt_entropy = switchState.dudt_diss
    else:
        dudt_entropy = torch.zeros_like(dudt)

    dEdt = currentState.masses * torch.einsum('ij,ij->i', currentState.velocities, (dvdt + dvdt_diss)) + currentState.masses * (dudt + dudt_diss + dudt_entropy)

    forcing = computeForcing(currentSystem, dt, t, config, schemeConfig)
    dvdt += forcing / currentState.masses.view(-1,1)


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