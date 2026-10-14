"""The CRKSPH (Conservative Reproducing Kernel) step: adaptive support solve,
`computeCRKFactors` for the CRK-corrected apparent volume/density/`crkState`,
ideal-gas EOS, the Cullen-Hopkins viscosity switch, a CRK velocity gradient
used both for `drhodt` (via its trace) and as an input to the CRK
pressure/viscosity acceleration and dudt kernels, and the Monaghan-Price
energy-balance term `f_ij`. `gradHState` is unconditionally `None` (the grad-h formulation is not used with CRK).

`crkSPH_step` is a list of single-purpose regions (one `record_function` each, numbered in call order), in the same shape as
`compSPH_step` and `compressibleSPH_Monaghan`. The CRK correction terms come from `modules/crk/terms.py` (one warp kernel, no host
read; the torch version stays in the core for the forward-mode tangent path).
"""

from ...modules.adaptiveSupport import solveStageSupport
from ...modules.boundaryConditions import computeForcing, enforceDirichlet, enforceUpdates
from ...modules.compressibleWall import beginCompressibleWall, clampSupportsToVolume
from ...modules.compSPH.operators import computeCompSPHBalanceTerm
from ...modules.compSPH.pairWork import accumulateStage, startStage, stageRebuildSlack
from ...modules.compSPH.supportGrid import solvesOnGrid
from ...utils.pairList import stagePairList
from ...modules.crk import CRKStage, computeCRKAccel, computeCRKdudt, crkFactors
from ...modules.eos import idealGasEOS
from ...modules.momentum import computeMomentumConsistent
from ...modules.reconstruction import requireRawPairVelocity
from ...modules.shockCapturing import computeViscositySwitchTerms, updateViscositySwitch
from ...configurations.compSPHConfig import CompSPHConfig
from ...configurations.crkSPH import resolveCRKLimiter
from ...configurations.simulationConfig import SimulationConfig
from ...systems.compSPH import CompSPHSystem
from ...systems.compressibleMonaghan import CompressibleSystemUpdate

import warnings

from warpSPHCore import GradientScheme, OperationProperties, SupportScheme, WarpOperation, warpOperation
from warpSPHCore.profiling import record_function
import torch

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
    stageIndex = None,     # injected by the Runge-Kutta integrator (it inspects this signature)
):
    _warnNonConservativeSupport(config)
    requireRawPairVelocity(schemeConfig.diffusionParams, 'CRKSPH')
    crkViscosityParams = resolveCRKLimiter(schemeConfig.crkViscosityParams, config.n_h)
    currentSystem = system
    currentState = currentSystem.state
    t = currentSystem.t
    gradHState = None

    onGrid = solvesOnGrid(schemeConfig)             # stepMode 'graph' (or supportSolveOnGrid): the support solve on the fixed-shape grid
    rebuildSlack = stageRebuildSlack(currentSystem, stageIndex)

    with record_function("[warpSPH] - [crkSPH - 01] wall"):
        wall = beginCompressibleWall(currentState, config)
        startStage(currentSystem, currentState, config, stageIndex)

    with record_function("[warpSPH] - [crkSPH - 02] optimal support"):
        solveStageSupport(currentSystem, currentState, config, schemeConfig, onGrid=onGrid, rebuildSlack=rebuildSlack)

    with record_function("[warpSPH] - [crkSPH - 02b] adjacency"):
        # list mode: the solver validated the list against the particles and every support it used (and rebuilt it if needed);
        # grid mode: an optimistic stage with a deferred device flag, or the explicit check / rebuild (utils/pairList.py)
        adjacency = stagePairList(currentSystem, currentState, config, schemeConfig, onGrid=onGrid, rebuildSlack=rebuildSlack)

    with record_function("[warpSPH] - [crkSPH - 03] support clamp"):
        # only reduces h, so the list stays valid; wall cases only (modules/compressibleWall/supportClamp.py)
        currentState.supports = clampSupportsToVolume(currentState, config, adjacency, wall)

    with record_function("[warpSPH] - [crkSPH - 04] CRK factors"):
        apparentVolume, currentState.densities, crkState = crkFactors(currentState, config.domain, config.kernel, adjacency = adjacency)

    if wall is not None:
        with record_function("[warpSPH] - [crkSPH - 05] wall closure"):
            # wall rows enter the CRK moment sums like fluid rows; their support and mass change here, so the factors (and
            # the fluid density) are redone
            wall.apply(currentState, config, schemeConfig, adjacency, latticeSupport=True)
            apparentVolume, currentState.densities, crkState = crkFactors(currentState, config.domain, config.kernel, adjacency = adjacency)
            wall.apply(currentState, config, schemeConfig, adjacency, latticeSupport=True)

    if currentState.divergence is None:
        with record_function("[warpSPH] - [crkSPH - 06] divergence (first time)"):
            print('Warning: divergence is None, computing for the first time')
            drhodt = computeMomentumConsistent(
                currentState,
                config,
                schemeConfig = schemeConfig,
                adjacency = adjacency,
                gradH = gradHState
            )
            currentState.divergence = -drhodt/currentState.densities

    with record_function("[warpSPH] - [crkSPH - 07] Dirichlet BC"):
        enforceDirichlet(currentSystem, t, dt, config, schemeConfig)

    with record_function("[warpSPH] - [crkSPH - 08] EOS"):
        currentState.entropies, _, currentState.pressures, currentState.soundspeeds = idealGasEOS(
            A = None,
            u = currentState.internalEnergies,
            P = None,
            rho = currentState.densities,
            gamma = schemeConfig.gamma,
        )

    with record_function("[warpSPH] - [crkSPH - 09] velocity gradient"):
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

    with record_function("[warpSPH] - [crkSPH - 10] drho/dt"):
        drhodt = -torch.einsum('...ii->...', velocityGradient) * currentState.densities

    with record_function("[warpSPH] - [crkSPH - 11] viscosity-switch terms"):
        currentState.alphas, switchState = computeViscositySwitchTerms(
            dt,
            currentState,
            config, schemeConfig,
            SupportScheme.SuperSymmetric,
            adjacency)

    stage = CRKStage(volumes = apparentVolume, crkState = crkState, velocityGradient = velocityGradient,
                     viscosityParams = crkViscosityParams)

    with record_function("[warpSPH] - [crkSPH - 12] pressure / viscosity acceleration"):
        dvdt, currentState.ap_ij, currentState.av_ij = computeCRKAccel(
            currentState, config, schemeConfig, adjacency, stage, gradHState)

    with record_function("[warpSPH] - [crkSPH - 13] dudt"):
        dudt = computeCRKdudt(currentState, config, schemeConfig, adjacency, stage, gradHState)

    with record_function("[warpSPH] - [crkSPH - 14] viscosity-switch update"):
        # Advance the switch's stored alpha0 (mirrors the Monaghan / compSPH wiring). AV_PLAN S2 had this call deleted; but
        # `computeCullenTerms` only decays the stored alpha0 by ONE step, so without writing it back it stays at its initial 1 and
        # alpha sticks at ~0.97-0.98 -- C&D / R&H under CRKSPH were fixed-alpha runs (AV_PLAN Phase 1 clean-up, item 3).
        currentState.alpha0s, switchState = updateViscositySwitch(
            switchState,
            dt, dvdt,
            currentState,
            config, schemeConfig,
            SupportScheme.SuperSymmetric,
            adjacency)

    with record_function("[warpSPH] - [crkSPH - 15] divergence"):
        currentState.divergence = -drhodt / currentState.densities

    with record_function("[warpSPH] - [crkSPH - 16] dE/dt"):
        dEdt = currentState.masses * torch.einsum('ij,ij->i', currentState.velocities, (dvdt)) + currentState.masses * (dudt)

    with record_function("[warpSPH] - [crkSPH - 17] forcing"):
        forcing = computeForcing(currentSystem, dt, t, config, schemeConfig)
        dvdt += forcing / currentState.masses.view(-1,1)

    with record_function("[warpSPH] - [crkSPH - 18] update"):
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

    with record_function("[warpSPH] - [crkSPH - 19] energy-partition f_ij"):
        v_halfstep = currentState.velocities + 0.5 * dt * update.dvdt
        currentState.f_ij = computeCompSPHBalanceTerm(
            currentState, config, schemeConfig, adjacency, v_halfstep, dt, gradHState)
        if wall is not None:
            currentState.f_ij = wall.balanceFractions(currentState.f_ij, adjacency, currentState)

    if currentSystem.pairWork is not None:
        currentSystem.pairWork.latest = adjacency
    if schemeConfig.compatibleEnergy and currentSystem.pairWork is not None:
        with record_function("[warpSPH] - [crkSPH - 20] pair work -> running sum"):
            accumulateStage(currentSystem.pairWork, stageIndex, adjacency, currentState, schemeConfig.energyScheme)
            # the pair arrays now live in the running sum: do not keep one set per stage on the returned state
            currentState.ap_ij = currentState.av_ij = currentState.f_ij = None

    return update, adjacency, currentState
