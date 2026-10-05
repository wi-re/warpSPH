"""The CompSPH (Owen 2010-style compatible-hydro) step: adaptive support
solve, super-symmetric adjacency, Gather-mode density, optional grad-h
(`GradHState`), ideal-gas EOS, the Cullen-Hopkins viscosity switch, the
CompSPH pressure/viscosity acceleration and dudt kernels, and the
Monaghan-Price energy-balance term `f_ij` (evaluated at the half-step
velocity). Falls back to computing `divergence` from `computeMomentumConsistent`
the first time a state has none (e.g. right after a resume that doesn't
persist it).
"""

from ..modules.adaptiveSupport import computeOmega, evaluateOptimalSupport
from ..modules.boundaryConditions import computeForcing, enforceDirichlet, enforceUpdates
from ..modules.compressibleWall import beginCompressibleWall
from ..modules.compSPH.accel import computeCompSPHAccelWarp
from ..modules.compSPH.dudt import computeCompSPHdudtWarp
from ..modules.compSPH.balance import computeCompSPHBalanceTermWarp
from ..modules.density import computeDensities
from ..modules.eos import idealGasEOS
from ..modules.momentum import computeMomentumConsistent
from ..modules.shockCapturing import computeViscositySwitchTerms, updateViscositySwitch
from ..enumTypes import EnergyScheme, ViscositySwitch

from warpSPHCore import (
    GradHState, OperationProperties, SupportScheme, buildVerletList,
)
from ..systems.compSPH import CompSPHSystem, CompSPHState
from ..configurations.compSPHConfig import CompSPHConfig
from ..configurations.simulationConfig import SimulationConfig
import torch
from ..systems.compressibleMonaghan import CompressibleSystemUpdate

from ..modules.shockCapturing.CullenHopkins import computeHopkinsTerms, computeHopkinsUpdate

lut = None

__all__ = ['compSPH_step']


def compSPH_step(
    system: CompSPHSystem,
    dt: "float | torch.Tensor",
    config: SimulationConfig,
    schemeConfig: CompSPHConfig,
    verbose = False,
    # dsphConfig = None,
):        
    global lut
    currentSystem = system#
    currentState = currentSystem.state
    # currentSystem.adjacency = None

    t = currentSystem.t
    wall = beginCompressibleWall(currentState, config)
    rho_optimal, h_optimal, currentSystem.adjacency, *_ = evaluateOptimalSupport(currentState, config, schemeConfig, SupportScheme.Gather, currentSystem.adjacency)
    currentState.supports = h_optimal
    currentState.densities = rho_optimal

    verletScale = config.verletScale

    adjacency = buildVerletList(
        currentState, 
        config.domain, verletScale = verletScale, supportMode = SupportScheme.SuperSymmetric,
        priorNeighborhood = None,
        verbose = False)
    currentSystem.adjacency = adjacency

    # with TimedBlock('compute csr', use_cuda=True, device=device):
        # csr_neighrs = coo_to_csr(neighbors.get('noghost')[0])
        # adjacency.i = neighbors.neighbors.row
        # adjacency.j = neighbors.neighbors.col
        # adjacency.numNeighbors = csr_neighrs.rowEntries.to(torch.int32)
        # adjacency.edgeOffsets = csr_neighrs.indptr.to(torch.int32)
        # # currentSystem.adjacency = adjacency

    # gather (the default) -- cullen switch E.1 in the CRK paper uses gather
    # for density estimation
    currentState.densities = computeDensities(
        currentState, config, schemeConfig, adjacency)
    if wall is not None:
        wall.apply(currentState, config, schemeConfig, adjacency)
        # the wall masses just changed, so the fluid density sum has to be redone
        currentState.densities = computeDensities(
            currentState, config, schemeConfig, adjacency)
        wall.apply(currentState, config, schemeConfig, adjacency)
    # Computed here (rather than where it is otherwise used, further down)
    # because a state reconstructed with `divergence=None` (e.g. resumed from
    # a trajectory export, which doesn't persist divergence) needs it in the
    # first-time branch immediately below; `computeOmega` only reads
    # positions/supports/masses/densities, all already finalised above, so
    # moving this earlier changes nothing about what it computes.
    if schemeConfig.adaptiveSupportCorrections:
        omega = computeOmega(currentState,
                OperationProperties(
                    kernel = config.kernel,
                    supportMode = SupportScheme.Gather, # E.5
                ),
                domain = config.domain,
                adjacency = adjacency
        )

        gradHState = GradHState(
            queryOmegas = omega
        )
    else:
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
        currentState.divergence = drhodt

    # enforceDirichlet(currentSystem.state, dsphConfig, t, dt)

    enforceDirichlet(currentSystem, t, dt, config, schemeConfig)
    currentState.entropies, _, currentState.pressures, currentState.soundspeeds = idealGasEOS(
        A = None,
        u = currentState.internalEnergies,
        P = None,
        rho = currentState.densities,
        gamma = schemeConfig.gamma,
    )

    currentState.alphas, switchState = computeViscositySwitchTerms(
        dt,
        currentState, 
        config, schemeConfig, 
        SupportScheme.SuperSymmetric, 
        adjacency)   


    dvdt, currentState.ap_ij, currentState.av_ij = computeCompSPHAccelWarp(
        queryParticles = currentState,
        operationProperties = OperationProperties(
            kernel = config.kernel,
            supportMode =  SupportScheme.KernelMeanSymmetric
        ),
        domain = config.domain,
        conductivityParams= schemeConfig.diffusionParams,

        queryEnergies = currentState.internalEnergies,
        queryVelocities= currentState.velocities,
        queryCs = currentState.soundspeeds,
        queryAlphas = currentState.alphas,
        queryPressures = currentState.pressures,

        adjacency = adjacency,
        gradHState = gradHState
    )

    dudt = computeCompSPHdudtWarp(
        queryParticles = currentState,
        operationProperties = OperationProperties(
            kernel = config.kernel,
            supportMode = SupportScheme.Gather #E.3
         ),
        domain = config.domain,
        conductivityParams= schemeConfig.diffusionParams,

        queryEnergies = currentState.internalEnergies,
        queryVelocities= currentState.velocities,
        queryCs = currentState.soundspeeds,
        queryAlphas = currentState.alphas,
        queryPressures = currentState.pressures,

        adjacency = adjacency,
        gradHState = gradHState
    )

    # particles.alpha0s, switchState = updateViscositySwitch(particles, wrappedKernel, neighbors.get('noghost'), SupportScheme.Gather, config, dt, dvdt, switchState)

    currentState.alpha0s, switchState = updateViscositySwitch(
        switchState,
        dt, dvdt,
        currentState, 
        config, schemeConfig, 
        SupportScheme.SuperSymmetric, 
        adjacency)   


    drhodt = computeMomentumConsistent(
        currentState,
        config,
        schemeConfig = None,
        adjacency = adjacency,
        gradH = gradHState
    )
    currentState.divergence = -drhodt / currentState.densities
    dEdt = currentState.masses * torch.einsum('ij,ij->i', currentState.velocities, (dvdt)) + currentState.masses * (dudt)

    # drhodt = torch.zeros_like(currentState.densities)
    # dEdt = torch.zeros_like(currentState.densities)

    # with TimedBlock('compute forcing', use_cuda=True, device=device):
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

    # with TimedBlock('enforce updates', use_cuda=True, device=device):
    enforceUpdates(update, currentSystem, dt, t, config, schemeConfig)
    if wall is not None:
        wall.finishUpdate(update)

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
    if wall is not None:
        currentState.f_ij = wall.balanceFractions(currentState.f_ij, adjacency, currentState)

    return update, adjacency, currentState