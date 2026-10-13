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

from warpSPHCore.profiling import record_function
from warpSPHCore import (
    GradHState, OperationProperties, SupportScheme, buildVerletList,
)
from ..modules.compSPH.pairWork import accumulateStage, startStage, stageRebuildSlack
from ..systems.compSPH import CompSPHSystem, CompSPHState
from ..configurations.compSPHConfig import CompSPHConfig
from ..configurations.simulationConfig import SimulationConfig
import torch
from ..systems.compressibleMonaghan import CompressibleSystemUpdate

from ..modules.shockCapturing.CullenHopkins import computeHopkinsTerms, computeHopkinsUpdate
from ..modules.reconstruction import requireRawPairVelocity

lut = None

__all__ = ['compSPH_step']


def compSPH_step(
    system: CompSPHSystem,
    dt: "float | torch.Tensor",
    config: SimulationConfig,
    schemeConfig: CompSPHConfig,
    verbose = False,
    stageIndex = None,     # injected by the Runge-Kutta integrator (it inspects this signature)
    # dsphConfig = None,
):        
    global lut
    requireRawPairVelocity(schemeConfig.diffusionParams, 'CompSPH')
    currentSystem = system#
    currentState = currentSystem.state
    # currentSystem.adjacency = None

    t = currentSystem.t
    with record_function("[warpSPH] - [compSPH - 01] wall + optimal support"):
        wall = beginCompressibleWall(currentState, config)
        startStage(currentSystem, currentState, config, stageIndex)
        rho_optimal, h_optimal, currentSystem.adjacency, *_ = evaluateOptimalSupport(
            currentState, config, schemeConfig, SupportScheme.Gather, currentSystem.adjacency,
            validateOnce=True, rebuildSlack=stageRebuildSlack(currentSystem, stageIndex), computeDensity=False)
        currentState.supports = h_optimal
        currentState.densities = rho_optimal

    with record_function("[warpSPH] - [compSPH - 02] adjacency"):
        verletScale = config.verletScale

        # the solver validated this list against the particles and every support it used, and rebuilt it if needed
        adjacency = currentSystem.adjacency

    # with TimedBlock('compute csr', use_cuda=True, device=device):
        # csr_neighrs = coo_to_csr(neighbors.get('noghost')[0])
        # adjacency.i = neighbors.neighbors.row
        # adjacency.j = neighbors.neighbors.col
        # adjacency.numNeighbors = csr_neighrs.rowEntries.to(torch.int32)
        # adjacency.edgeOffsets = csr_neighrs.indptr.to(torch.int32)
        # # currentSystem.adjacency = adjacency

    with record_function("[warpSPH] - [compSPH - 03] density (+ wall closure)"):
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
    with record_function("[warpSPH] - [compSPH - 04] grad-h terms"):
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

    with record_function("[warpSPH] - [compSPH - 05] EOS + viscosity-switch terms"):
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


    with record_function("[warpSPH] - [compSPH - 06] pressure / viscosity acceleration"):
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

    with record_function("[warpSPH] - [compSPH - 07] dudt"):
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

    with record_function("[warpSPH] - [compSPH - 08] switch update, divergence, dE/dt"):
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
    with record_function("[warpSPH] - [compSPH - 09] forcing + update"):
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

    with record_function("[warpSPH] - [compSPH - 10] energy-partition f_ij"):
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

    if currentSystem.pairWork is not None:
        currentSystem.pairWork.latest = adjacency
    if schemeConfig.compatibleEnergy and currentSystem.pairWork is not None:
        with record_function("[warpSPH] - [compSPH - 11] pair work -> running sum"):
            accumulateStage(currentSystem.pairWork, stageIndex, adjacency, currentState, schemeConfig.energyScheme)
            # the pair arrays now live in the running sum: do not keep one set per stage on the returned state
            currentState.ap_ij = currentState.av_ij = currentState.f_ij = None

    return update, adjacency, currentState