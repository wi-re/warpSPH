"""The CompSPH (Owen 2010-style compatible-hydro) step: adaptive support
solve, super-symmetric adjacency, Gather-mode density, optional grad-h
(`GradHState`), ideal-gas EOS, the Cullen-Hopkins viscosity switch, the
CompSPH pressure/viscosity acceleration and dudt operators, and the
Monaghan-Price energy-balance term `f_ij` (evaluated at the half-step
velocity). Falls back to computing `divergence` from `computeMomentumConsistent`
the first time a state has none (e.g. right after a resume that doesn't
persist it).

`compSPH_step` is a list of single-purpose regions (one `record_function` each, numbered
in call order): every region does one thing, so a profile or a call audit names it.
"""

from ...modules.adaptiveSupport import computeOmega, solveStageSupport
from ...modules.boundaryConditions import computeForcing, enforceDirichlet, enforceUpdates
from ...modules.compressibleWall import beginCompressibleWall, clampSupportsToVolume
from ...modules.compSPH.operators import computeCompSPHAccel, computeCompSPHdudt, computeCompSPHBalanceTerm
from ...modules.compSPH.pairWork import accumulateStage, startStage, stageRebuildSlack
from ...modules.compSPH.supportGrid import solvesOnGrid
from ...modules.density import computeDensities
from ...modules.eos import idealGasEOS
from ...modules.momentum import computeMomentumConsistent
from ...modules.reconstruction import requireRawPairVelocity
from ...modules.shockCapturing import computeViscositySwitchTerms, updateViscositySwitch
from ...systems.compSPH import CompSPHSystem
from ...systems.compressibleMonaghan import CompressibleSystemUpdate
from ...configurations.compSPHConfig import CompSPHConfig
from ...configurations.simulationConfig import SimulationConfig
from ...utils.pairList import stagePairList
from warpSPHCore import GradHState, OperationProperties, SupportScheme
from warpSPHCore.profiling import record_function
import torch

__all__ = ['compSPH_step']


def compSPH_step(
    system: CompSPHSystem,
    dt: "float | torch.Tensor",
    config: SimulationConfig,
    schemeConfig: CompSPHConfig,
    verbose = False,
    stageIndex = None,     # injected by the Runge-Kutta integrator (it inspects this signature)
):
    requireRawPairVelocity(schemeConfig.diffusionParams, 'CompSPH')
    currentSystem = system
    currentState = currentSystem.state

    t = currentSystem.t
    onGrid = solvesOnGrid(schemeConfig)
    rebuildSlack = stageRebuildSlack(currentSystem, stageIndex)

    with record_function("[warpSPH] - [compSPH - 01] wall"):
        wall = beginCompressibleWall(currentState, config)
        startStage(currentSystem, currentState, config, stageIndex)

    with record_function("[warpSPH] - [compSPH - 02] optimal support"):
        solveStageSupport(currentSystem, currentState, config, schemeConfig, onGrid=onGrid, rebuildSlack=rebuildSlack)

    with record_function("[warpSPH] - [compSPH - 03] adjacency"):
        adjacency = stagePairList(currentSystem, currentState, config, schemeConfig, onGrid=onGrid, rebuildSlack=rebuildSlack)

    with record_function("[warpSPH] - [compSPH - 04] support clamp"):
        # only reduces h, so the list stays valid; wall cases only (modules/compressibleWall/supportClamp.py)
        currentState.supports = clampSupportsToVolume(currentState, config, adjacency, wall)

    with record_function("[warpSPH] - [compSPH - 05] density"):
        # gather (the default) -- cullen switch E.1 in the CRK paper uses gather for density estimation
        currentState.densities = computeDensities(currentState, config, schemeConfig, adjacency)

    if wall is not None:
        with record_function("[warpSPH] - [compSPH - 06] wall closure"):
            wall.apply(currentState, config, schemeConfig, adjacency)
            # the wall masses just changed, so the fluid density sum has to be redone
            currentState.densities = computeDensities(currentState, config, schemeConfig, adjacency)
            wall.apply(currentState, config, schemeConfig, adjacency)

    with record_function("[warpSPH] - [compSPH - 07] grad-h terms"):
        # Computed before the divergence fallback below; `computeOmega` only reads
        # positions/supports/masses/densities, all finalised above.
        if schemeConfig.adaptiveSupportCorrections:
            omega = computeOmega(currentState,
                    OperationProperties(
                        kernel = config.kernel,
                        supportMode = SupportScheme.Gather, # E.5
                    ),
                    domain = config.domain,
                    adjacency = adjacency
            )
            gradHState = GradHState(queryOmegas = omega)
        else:
            gradHState = None

    if currentState.divergence is None:
        # a state reconstructed without `divergence` (e.g. resumed from a trajectory export, which doesn't persist it)
        with record_function("[warpSPH] - [compSPH - 08] divergence (first time)"):
            print('Warning: divergence is None, computing for the first time')
            currentState.divergence = computeMomentumConsistent(
                currentState,
                config,
                schemeConfig = schemeConfig,
                adjacency = adjacency,
                gradH = gradHState
            )

    with record_function("[warpSPH] - [compSPH - 09] Dirichlet BC"):
        enforceDirichlet(currentSystem, t, dt, config, schemeConfig)

    with record_function("[warpSPH] - [compSPH - 10] EOS"):
        currentState.entropies, _, currentState.pressures, currentState.soundspeeds = idealGasEOS(
            A = None,
            u = currentState.internalEnergies,
            P = None,
            rho = currentState.densities,
            gamma = schemeConfig.gamma,
        )

    with record_function("[warpSPH] - [compSPH - 11] viscosity-switch terms"):
        currentState.alphas, switchState = computeViscositySwitchTerms(
            dt,
            currentState,
            config, schemeConfig,
            SupportScheme.SuperSymmetric,
            adjacency)

    with record_function("[warpSPH] - [compSPH - 12] pressure / viscosity acceleration"):
        dvdt, currentState.ap_ij, currentState.av_ij = computeCompSPHAccel(
            currentState, config, schemeConfig, adjacency, gradHState)

    with record_function("[warpSPH] - [compSPH - 13] dudt"):
        dudt = computeCompSPHdudt(currentState, config, schemeConfig, adjacency, gradHState)

    with record_function("[warpSPH] - [compSPH - 14] viscosity-switch update"):
        currentState.alpha0s, switchState = updateViscositySwitch(
            switchState,
            dt, dvdt,
            currentState,
            config, schemeConfig,
            SupportScheme.SuperSymmetric,
            adjacency)

    with record_function("[warpSPH] - [compSPH - 15] divergence / drho/dt"):
        drhodt = computeMomentumConsistent(
            currentState,
            config,
            schemeConfig = None,
            adjacency = adjacency,
            gradH = gradHState
        )
        currentState.divergence = -drhodt / currentState.densities

    with record_function("[warpSPH] - [compSPH - 16] dE/dt"):
        dEdt = currentState.masses * torch.einsum('ij,ij->i', currentState.velocities, dvdt) + currentState.masses * dudt

    with record_function("[warpSPH] - [compSPH - 17] forcing"):
        forcing = computeForcing(currentSystem, dt, t, config, schemeConfig)
        dvdt += forcing / currentState.masses.view(-1,1)

    with record_function("[warpSPH] - [compSPH - 18] update"):
        update = CompressibleSystemUpdate(
            dxdt = currentState.velocities.clone(),
            dvdt = dvdt,
            dudt = dudt,
            drhodt = drhodt,
            dEdt = dEdt,
            passive = torch.zeros(currentState.densities.shape, device=currentState.densities.device, dtype=torch.bool)
        )
        enforceUpdates(update, currentSystem, dt, t, config, schemeConfig)
        if wall is not None:
            wall.finishUpdate(update)

    with record_function("[warpSPH] - [compSPH - 19] energy-partition f_ij"):
        v_halfstep = currentState.velocities + 0.5 * dt * update.dvdt
        currentState.f_ij = computeCompSPHBalanceTerm(
            currentState, config, schemeConfig, adjacency, v_halfstep, dt, gradHState)
        if wall is not None:
            currentState.f_ij = wall.balanceFractions(currentState.f_ij, adjacency, currentState)

    if currentSystem.pairWork is not None:
        currentSystem.pairWork.latest = adjacency
    if schemeConfig.compatibleEnergy and currentSystem.pairWork is not None:
        with record_function("[warpSPH] - [compSPH - 20] pair work -> running sum"):
            accumulateStage(currentSystem.pairWork, stageIndex, adjacency, currentState, schemeConfig.energyScheme)
            # the pair arrays now live in the running sum: do not keep one set per stage on the returned state
            currentState.ap_ij = currentState.av_ij = currentState.f_ij = None

    return update, adjacency, currentState
