"""The CRKSPH (Conservative Reproducing Kernel) step: adaptive support solve,
`computeCRKFactors` for the CRK-corrected apparent volume/density/`crkState`,
ideal-gas EOS, the Cullen-Hopkins viscosity switch, a CRK velocity gradient
used both for `drhodt` (via its trace) and as an input to the CRK
pressure/viscosity acceleration and dudt kernels, and the Monaghan-Price
energy-balance term `f_ij`. Large stretches of an earlier
`computeCompSPHAccelWarp`/grad-h formulation are commented out rather than
removed (tracked separately in docs/historic_plans/CLEANUP_PLAN.md); `gradHState` is
unconditionally `None` (its adaptive-support branch is also commented out).
It used to only be assigned *after* the `currentState.divergence is None`
fallback branch that reads it -- an `UnboundLocalError` waiting to happen on
any first call with unset `divergence` -- fixed 2026-08-15 by assigning it
before that branch instead.
"""

from ..modules.adaptiveSupport import computeOmega, evaluateOptimalSupport
from ..modules.boundaryConditions import computeForcing, enforceDirichlet, enforceUpdates
from ..modules.compSPH.accel import computeCompSPHAccelWarp
from ..modules.compSPH.dudt import computeCompSPHdudtWarp
from ..modules.compSPH.balance import computeCompSPHBalanceTermWarp
from ..modules.crk import computeCrkSPHdudtWarp
from ..modules.eos import idealGasEOS
from ..modules.momentum import computeMomentumConsistent
from ..modules.shockCapturing import computeViscositySwitchTerms
from ..enumTypes import EnergyScheme

from warpSPHCore import (
    GradientScheme, OperationProperties, SupportScheme,
    WarpOperation, buildVerletList, computeCRKFactors,
    warpOperation,
)
from ..systems.compSPH import CompSPHSystem, CompSPHState
from ..configurations.compSPHConfig import CompSPHConfig
from ..configurations.simulationConfig import SimulationConfig
import torch
from ..systems.compressibleMonaghan import CompressibleSystemUpdate

from ..modules.shockCapturing.CullenHopkins import computeHopkinsTerms, computeHopkinsUpdate

from ..modules.crk.accel import computeCrkSPHAccelWarp

__all__ = ['crkSPH_step']


def crkSPH_step(
    system: CompSPHSystem,
    dt: "float | torch.Tensor",
    config: SimulationConfig,
    schemeConfig: CompSPHConfig,
    verbose = False,
):

    currentSystem = system#
    currentState = currentSystem.state
    t = currentSystem.t

    IE = currentState.internalEnergies * currentState.masses
    KE = 0.5 * currentState.masses * torch.einsum('ij,ij->i', currentState.velocities, currentState.velocities)
    TE = IE + KE

    # print(f"TE: {TE.sum().item()}, IE: {IE.sum().item()}, KE: {KE.sum().item()}")
    # print(f'\tmin/max/mean TE: {TE.min().item()}/{TE.max().item()}/{TE.mean().item()}')
    # print(f'\tmin/max/mean IE: {IE.min().item()}/{IE.max().item()}/{IE.mean().item()}')
    # print(f'\tmin/max/mean KE: {KE.min().item()}/{KE.max().item()}/{KE.mean().item()}')

    rho_optimal, h_optimal, currentSystem.adjacency, *_ = evaluateOptimalSupport(currentState, config, schemeConfig, SupportScheme.Gather, currentSystem.adjacency)
    currentState.supports = h_optimal
    currentState.densities = rho_optimal

    # print(f"\tOptimal support: min/max/mean h: {h_optimal.min().item()}/{h_optimal.max().item()}/{h_optimal.mean().item()}")
    # print(f'\tDensity: min/max/mean rho: {rho_optimal.min().item()}/{rho_optimal.max().item()}/{rho_optimal.mean().item()}')

    # meanSupport = currentState.supports[10:-10].mean().item()
    # currentState.supports[:10] = meanSupport
    # currentState.supports[-10:] = meanSupport
    # currentState.supports[:] = meanSupport

    verletScale = config.verletScale

    adjacency = buildVerletList(
        currentState, 
        config.domain, verletScale = verletScale, supportMode = SupportScheme.SuperSymmetric,
        priorNeighborhood = currentSystem.adjacency,
        verbose = False)
    currentSystem.adjacency = adjacency

    apparentVolume, currentState.densities, crkState = computeCRKFactors(currentState, config.domain, config.kernel, adjacency = adjacency)

    # currentState.densities = warpOperation(
    #     currentState,
    #     OperationProperties(
    #         kernel = config.kernel,
    #         operation = WarpOperation.Density,
    #         supportMode = SupportScheme.Gather, # cullen switch E.1 in the CRK paper uses gather for density estimation
    #     ),
    #     domain = config.domain,
    #     adjacency = adjacency,
    # )
    # gradHState is only ever actually set below (currently unconditionally None --
    # see the commented-out adaptiveSupportCorrections branch further down); assigned
    # here too so the first-call fallback below doesn't read it before it exists.
    gradHState = None
    if currentState.divergence is None:
        print('Warning: divergence is None, computing for the first time')
        drhodt = computeMomentumConsistent(
            currentState,
            config,
            schemeConfig = schemeConfig,
            adjacency = adjacency,
            gradH = gradHState
        )
        currentState.divergence = -drhodt/currentState.densities

    # currentState.densities = warpOperation(
    #     currentState,
    #     OperationProperties(
    #         kernel = config.kernel,
    #         operation = WarpOperation.Density,
    #         supportMode = SupportScheme.Gather, # cullen switch E.1 in the CRK paper uses gather for density estimation
    #     ),
    #     domain = config.domain,
    #     adjacency = adjacency,
    # )
    enforceDirichlet(currentSystem, t, dt, config, schemeConfig)
    currentState.entropies, _, currentState.pressures, currentState.soundspeeds = idealGasEOS(
        A = None,
        u = currentState.internalEnergies,
        P = None,
        rho = currentState.densities,
        gamma = schemeConfig.gamma,
    )

    # nabla_dot_v = warpOperation(
    #     currentState,
    #     OperationProperties(
    #         kernel = config.kernel,
    #         operation = WarpOperation.Divergence,
    #         supportMode = SupportScheme.Scatter, # E.3
    #         gradientMode = GradientScheme.Difference, # E.3
    #     ),
    #     queryValues = currentState.velocities,
    #     domain = config.domain,
    #     adjacency = adjacency,
    #     queryVolumes = apparentVolume,
    #     crkState= crkState,
    # )
    # nabla_times_v = warpOperation(
    #     currentState,
    #     OperationProperties(
    #         kernel = config.kernel,
    #         operation = WarpOperation.Curl,
    #         supportMode = SupportScheme.Scatter, # E.3
    #         gradientMode = GradientScheme.Difference, # E.3
    #     ),
    #     queryValues = currentState.velocities,
    #     domain = config.domain,
    #     adjacency = adjacency,
    #     queryVolumes = apparentVolume,
    #     crkState= crkState,
    # )

    # balsara = torch.abs(nabla_dot_v) / (torch.abs(nabla_dot_v) + torch.norm(nabla_times_v, dim=-1) + 1e-4 * currentState.soundspeeds )
    # currentState.alphas = torch.clamp(balsara, 0.0, 1.0) 

    # if schemeConfig.adaptiveSupportCorrections:
    #     omega = computeOmega(currentState, 
    #             OperationProperties(
    #                 kernel = config.kernel,
    #                 supportMode = SupportScheme.Gather, # E.5
    #             ),
    #             domain = config.domain,
    #             adjacency = adjacency
    #     )

    #     gradHState = GradHState(
    #         queryOmegas = omega
    #     )
    # else:
    #     gradHState = None
    # (gradHState is set to None near the top of this function instead, see there)

    # currentState.alphas, switchState = computeViscositySwitchTerms(
    #     dt,
    #     currentState, 
    #     config, schemeConfig, 
    #     SupportScheme.SuperSymmetric, 
    #     adjacency)   

    velocityGradient = warpOperation(
        currentState,
        OperationProperties(
            kernel = config.kernel,
            operation = WarpOperation.Gradient,
            supportMode = SupportScheme.Scatter, # E.3
            gradientMode = GradientScheme.Difference, # E.3
        ),
        queryValues = currentState.velocities,
        domain = config.domain,
        adjacency = adjacency,
        queryVolumes = apparentVolume,
        crkState= crkState,
    ).mT
    drhodt = -torch.einsum('...ii->...', velocityGradient) * currentState.densities


    # dvdt, currentState.ap_ij, currentState.av_ij = computeCompSPHAccelWarp(
    #     queryParticles = currentState,
    #     operationProperties = OperationProperties(
    #         kernel = config.kernel,
    #         supportMode =  SupportScheme.KernelMeanSymmetric
    #     ),
    #     domain = config.domain,
    #     conductivityParams= schemeConfig.diffusionParams,

    #     queryEnergies = currentState.internalEnergies,
    #     queryVelocities= currentState.velocities,
    #     queryCs = currentState.soundspeeds,
    #     queryAlphas = currentState.alphas,
    #     queryPressures = currentState.pressures,

    #     adjacency = adjacency,
    #     gradHState = gradHState
    # )


    currentState.alphas, switchState = computeViscositySwitchTerms(
        dt,
        currentState, 
        config, schemeConfig, 
        SupportScheme.SuperSymmetric, 
        adjacency)   


    # dvdt, currentState.ap_ij, currentState.av_ij = computeCompSPHAccelWarp(
    #     queryParticles = currentState,
    #     operationProperties = OperationProperties(
    #         kernel = config.kernel,
    #         supportMode =  SupportScheme.KernelMeanSymmetric
    #     ),
    #     domain = config.domain,
    #     conductivityParams= schemeConfig.diffusionParams,

    #     queryEnergies = currentState.internalEnergies,
    #     queryVelocities= currentState.velocities,
    #     queryCs = currentState.soundspeeds,
    #     queryAlphas = currentState.alphas,
    #     queryPressures = currentState.pressures,

    #     adjacency = adjacency,
    #     gradHState = gradHState
    # )

    # dudt = computeCompSPHdudtWarp(
    #     queryParticles = currentState,
    #     operationProperties = OperationProperties(
    #         kernel = config.kernel,
    #         supportMode = SupportScheme.Gather #E.3
    #      ),
    #     domain = config.domain,
    #     conductivityParams= schemeConfig.diffusionParams,

    #     queryEnergies = currentState.internalEnergies,
    #     queryVelocities= currentState.velocities,
    #     queryCs = currentState.soundspeeds,
    #     queryAlphas = currentState.alphas,
    #     queryPressures = currentState.pressures,

    #     adjacency = adjacency,
    #     gradHState = gradHState
    # )

    dvdt, currentState.ap_ij, currentState.av_ij = computeCrkSPHAccelWarp(
        queryParticles = currentState,
        operationProperties = OperationProperties(
            kernel = config.kernel,
            supportMode =  SupportScheme.KernelMeanSymmetric
        ),
        domain = config.domain,
        conductivityParams= schemeConfig.diffusionParams,
        crkViscosityParams = schemeConfig.crkViscosityParams,
        queryVelocityTensor= velocityGradient,
        queryEnergies = currentState.internalEnergies,
        queryVelocities= currentState.velocities,
        queryCs = currentState.soundspeeds,
        queryAlphas = currentState.alphas,
        queryPressures = currentState.pressures,
        queryVolumes = apparentVolume,
        crkState = crkState,

        adjacency = adjacency,
        gradHState = gradHState,
    )

    dudt = computeCrkSPHdudtWarp(
        queryParticles = currentState,
        operationProperties = OperationProperties(
            kernel = config.kernel,
            supportMode = SupportScheme.KernelMeanSymmetric #E.3
         ),
        domain = config.domain,
        conductivityParams= schemeConfig.diffusionParams,
        crkViscosityParams = schemeConfig.crkViscosityParams,
        queryVelocityTensor= velocityGradient,

        queryEnergies = currentState.internalEnergies,
        queryVelocities= currentState.velocities,
        queryCs = currentState.soundspeeds,
        queryAlphas = currentState.alphas,
        queryPressures = currentState.pressures,
        queryVolumes = apparentVolume,
        crkState = crkState,

        adjacency = adjacency,
        gradHState = gradHState
    )
    # dudt = computeCompSPHdudtWarp(
    #     queryParticles = currentState,
    #     operationProperties = OperationProperties(
    #         kernel = config.kernel,
    #         supportMode = SupportScheme.Gather #E.3
    #      ),
    #     domain = config.domain,
    #     conductivityParams= schemeConfig.diffusionParams,

    #     queryEnergies = currentState.internalEnergies,
    #     queryVelocities= currentState.velocities,
    #     queryCs = currentState.soundspeeds,
    #     queryAlphas = currentState.alphas,
    #     queryPressures = currentState.pressures,
    #     # queryVolumes = apparentVolume,
    #     # crkState = crkState,

    #     adjacency = adjacency,
    #     gradHState = gradHState
    # )

    # particles.alpha0s, switchState = updateViscositySwitch(particles, wrappedKernel, neighbors.get('noghost'), SupportScheme.Gather, config, dt, dvdt, switchState)

    # currentState.alpha0s, switchState = updateViscositySwitch(
    #     switchState,
    #     dt, dvdt,
    #     currentState, 
    #     config, schemeConfig, 
    #     SupportScheme.SuperSymmetric, 
    #     adjacency)   


    # drhodt = computeMomentumConsistent(
    #     currentState,
    #     config,
    #     supportScheme = SupportScheme.Gather,
    #     adjacency = adjacency,
    #     gradH = gradHState
    # )
    currentState.divergence = -drhodt / currentState.densities
    dEdt = currentState.masses * torch.einsum('ij,ij->i', currentState.velocities, (dvdt)) + currentState.masses * (dudt)

    forcing = computeForcing(currentSystem, dt, t, config, schemeConfig)
    dvdt += forcing / currentState.masses.view(-1,1)

    update = CompressibleSystemUpdate(
        dxdt = currentState.velocities.clone(),
        dvdt = dvdt,
        dudt = dudt,
        drhodt = drhodt,
        dEdt = dEdt,
        passive = torch.zeros(currentState.densities.shape, device=currentState.densities.device, dtype=torch.bool)
    )

    enforceUpdates(update, currentSystem, dt, t, config, schemeConfig)
    
    v_halfstep = currentState.velocities + 0.5 * dt * update.dvdt

    currentState.f_ij = computeCompSPHBalanceTermWarp(
        queryParticles = currentState,
        operationProperties = OperationProperties(
            kernel = config.kernel,
            supportMode = config.supportMode
        ),
        domain = config.domain,

        queryEnergies = currentState.internalEnergies,
        queryVelocities= v_halfstep,
        queryPressures = currentState.pressures,

        pairWise_pressureAccel= currentState.ap_ij,
        pairWise_viscosityAccel = currentState.av_ij,
        energyScheme = schemeConfig.energyScheme,
        dt= dt,
        gamma = schemeConfig.gamma,

        adjacency = adjacency,
        gradHState = gradHState
    )


    return update, adjacency, currentState