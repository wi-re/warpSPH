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

import warnings

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
from ..configurations.crkSPH import resolveCRKLimiter

__all__ = ['crkSPH_step']

# CRKSPH is formulated with the kernel-mean pair kernel
# W_ij = [W_i(x_ij, h_i) + W_j(x_ij, h_j)] / 2 (Frontiere et al. 2017, Eq. 8);
# the acceleration and dudt below hard-code SupportScheme.KernelMeanSymmetric
# for that reason, but other parts of the step read `config.supportMode`.
# With a non-symmetric mode the run silently stops conserving total energy
# to round-off: Sod (1D, nx=200, RK2, Owen supports) drifts -7.3e-6 under
# SupportScheme.Gather vs +2.5e-16 under KernelMeanSymmetric (CompSPH is
# exact under both). Pinning the f_ij energy-balance term to
# KernelMeanSymmetric does NOT remove the drift -- the other consumer on this
# path is the Owen adaptive-support solve (optimalSupportOwen.py) -- so no
# Gather-conserving variant exists here; making one is derivation work (the
# paper's Eq. 60 keeps both a_ij and a_ji in general). Hence a warning, once
# per process and support mode.
_CRK_CONSERVATIVE_SUPPORT = SupportScheme.KernelMeanSymmetric
_warnedSupportModes: set = set()


class CRKSupportWarning(RuntimeWarning):
    """CRKSPH run with a support mode it does not conserve energy under."""


def _warnNonConservativeSupport(config: SimulationConfig) -> None:
    mode = getattr(config, 'supportMode', None)
    if mode is None or mode == _CRK_CONSERVATIVE_SUPPORT or mode in _warnedSupportModes:
        return
    _warnedSupportModes.add(mode)
    # Force-show this category for this one call, so a blanket
    # warnings.filterwarnings("ignore") anywhere in the process (one used to
    # sit in geometry/sdfFunctionality/implicitFunctions.py) cannot hide it.
    # The once-per-mode dedupe above keeps it quiet.
    with warnings.catch_warnings():
        warnings.simplefilter('always', CRKSupportWarning)
        _emitSupportWarning(mode)


def _emitSupportWarning(mode) -> None:
    warnings.warn(
        f'CRKSPH with supportMode={getattr(mode, "name", mode)}: CRKSPH is '
        f'formulated with the kernel-mean pair kernel (Frontiere et al. 2017, '
        f'Eq. 8, SupportScheme.KernelMeanSymmetric); with any other support '
        f'mode total energy is not conserved to round-off (Sod nx=200: '
        f'-7.3e-6 under Gather vs 2.5e-16 under KernelMeanSymmetric). Set '
        f'supportMode=KernelMeanSymmetric unless the drift is intended.',
        CRKSupportWarning, stacklevel=4)


def crkSPH_step(
    system: CompSPHSystem,
    dt: "float | torch.Tensor",
    config: SimulationConfig,
    schemeConfig: CompSPHConfig,
    verbose = False,
):

    _warnNonConservativeSupport(config)
    crkViscosityParams = resolveCRKLimiter(schemeConfig.crkViscosityParams, config.n_h)
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
        crkViscosityParams = crkViscosityParams,
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
        crkViscosityParams = crkViscosityParams,
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

    # NOTE (AV_PLAN S2): CRKSPH does NOT advance the viscosity switch's `alpha0s` -- the
    # `updateViscositySwitch` call that used to sit here was commented out. `alphas` are
    # recomputed above from the initial `alpha0s` every step, so Cullen-Dehnen / Read-Hayfield
    # under CRKSPH have no decay memory (instant response, no relaxation). Restoring the
    # call would change CRKSPH results; it was deleted rather than restored (user decision).


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