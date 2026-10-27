"""The ACSPH dual-time solve (De Courcy et al. 2024, Eqs. 38-48), as plain data and functions.

`DualTimeProblem` is the frozen description of one real step (everything that is fixed while the
pseudo-time loop runs: `dtau`, the BDF coefficients and history, `k1`, `k2`, `nu`, the adjacency,
the masks, the surface flags); `iterate(problem, x, v, p)` is one pseudo-iteration and
`solveDualTime(problem, x0, v0, p0)` the loop. Nothing here holds mutable state.

    u0 = u                                   # frozen for the BDF source
    D^p = pressureSmoothing(u0)              # frozen across RK stages
    for s in 1..rkStages:
        r  = spatial residual at u^{s-1}     # Eqs. 23, 25, 26
        r* = (r - I_c (alpha_t u0 + beta_t u^n + gamma_t u^{n-1})) / alpha_PI
        u^s = u0 + dtau sum_l a_{s,l} r*_l
    u = u0 + dtau sum_l b_l r*_l

`I_c = diag{0, 1, 1}`: the continuity equation has no real-time derivative, which is what makes
`r* -> 0` enforce `div v = 0` *at* time level n+1. The RK sweep is the general explicit Butcher
form (`warpSPHIntegrators.getButcherTableau`; ACSPH_PLAN.md Sec. 5.2), `k2 D^p` carries no extra
`h` (Sec. 5.3), and the diffusion is frozen across the stages but re-evaluated every iteration.
`tilde v = v - Dx/Dt` is simultaneously the position-row residual and the convergence metric
(Eq. 26 / Sec. 1.6).
"""

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
from warpSPHCore.profiling import record_function
from warpSPHIntegrators.schemes.coefficients.explicit import getButcherTableau

from ...systems.artificialCompressible import ArtificialCompressibleSystemUpdate
from ...configurations.compute import resolveCheckSchedule
from ..incompressible.convergence import ConvergenceCheckSchedule
from ..mdbc import computeBoundaryVelocities
from .parameters import convergenceMetricDevice
from .pressureSmoothing import computePressureSmoothing
from .residual import spatialResidual
from .surfaceFlags import SurfaceFlags, detectSurface
from .wallClosure import ContactMasks, WallBounce, bounceVelocities, projectPressure, wallPressures
from .workingState import workingState


@dataclass(frozen=True)
class DualTimeProblem:
    """One real step's dual-time problem; fixed for the whole pseudo-time loop."""
    state: Any
    config: Any
    schemeConfig: Any
    adjacency: Any
    dtau: float
    k1: torch.Tensor
    k2: torch.Tensor
    nu: torch.Tensor
    bdf: Tuple[float, float, float]                 # alpha_t, beta_t, gamma_t
    history: Tuple[torch.Tensor, ...]               # xPrev, xPrev2, vPrev, vPrev2
    bodyForce: torch.Tensor
    wallBodyForce: Optional[torch.Tensor]           # `(g - a_wall)` for Eq. (61), None without gravity
    surface: SurfaceFlags
    contact: ContactMasks
    bounce: Optional[WallBounce]
    # derived once per step
    tableau: Any
    fluid: torch.Tensor
    fluidRows: torch.Tensor
    fluidIndex: torch.Tensor
    nFluid: int


@dataclass
class DualTimeResult:
    positions: torch.Tensor
    velocities: torch.Tensor
    pressures: torch.Tensor
    iterations: int
    epsilonV: float


def makeDualTimeProblem(state, config, schemeConfig, adjacency, *, dtau, k1, k2, nu, bdf, history,
                        bodyForce, wallBodyForce, surface, contact, bounce) -> DualTimeProblem:
    fluid = state.kinds == 0
    fluidIndex = fluid.nonzero().squeeze(1)
    return DualTimeProblem(
        state, config, schemeConfig, adjacency, dtau, k1, k2, nu, tuple(bdf), tuple(history),
        bodyForce, wallBodyForce, surface, contact, bounce,
        getButcherTableau(schemeConfig.acParams.butcherTableauName),
        fluid, fluid.unsqueeze(-1), fluidIndex, int(fluidIndex.numel()))


def bdfHistory(system, state):
    """`(xPrev, xPrev2, vPrev, vPrev2)`: the BDF history, falling back to the current
    state on the first steps."""
    xPrev = system.positionsPrev if system.positionsPrev is not None else state.positions
    vPrev = system.velocitiesPrev if system.velocitiesPrev is not None else state.velocities
    xPrev2 = system.positionsPrev2 if system.positionsPrev2 is not None else xPrev
    vPrev2 = system.velocitiesPrev2 if system.velocitiesPrev2 is not None else vPrev
    return xPrev, xPrev2, vPrev, vPrev2


# -- pieces of one iteration ----------------------------------------------------------------

def surfaceAt(problem: DualTimeProblem, stage) -> SurfaceFlags:
    """The frozen per-step flags, or a fresh detection at `stage` (D2)."""
    if not problem.schemeConfig.acParams.redetectSurfacePerIteration:
        return problem.surface
    return detectSurface(stage, problem.config, problem.schemeConfig, problem.adjacency, store=False)


def closedWallPressure(problem: DualTimeProblem, view, flags: SurfaceFlags):
    """Eq. (61) wall pressure of `view`, then the contact projection."""
    acParams = problem.schemeConfig.acParams
    p = wallPressures(view, problem.config, problem.adjacency, problem.wallBodyForce,
                      edgeRelax=acParams.wallPressureEdgeRelax,
                      edgeFlags=flags.rawEdge, lambdas=flags.lambdas)
    return projectPressure(problem.contact, p)


def bdfSource(problem: DualTimeProblem, x, v):
    """The BDF source `alpha_t u + beta_t u^n + gamma_t u^{n-1}` for positions and velocities."""
    alphaT, betaT, gammaT = problem.bdf
    xPrev, xPrev2, vPrev, vPrev2 = problem.history
    return alphaT * x + betaT * xPrev + gammaT * xPrev2, alphaT * v + betaT * vPrev + gammaT * vPrev2


def pointImplicit(problem: DualTimeProblem, stageIndex: int) -> float:
    """`alpha_PI = 1 + alpha_s dtau alpha_t` (Eqs. 43-45); `alpha_s` is the fraction of `dtau` at
    which stage `s`'s residual is applied, i.e. the node `c[s+1]` of the stage it produces and 1
    for the last. Exact for the RK2 midpoint tableau; ambiguous for RK3/RK4 where Eq. (40) and
    Fig. 1 disagree (ACSPH_PLAN.md Sec. 5.2), and the paper reports `alpha_PI = 1` works."""
    acParams = problem.schemeConfig.acParams
    if not acParams.usePointImplicit:
        return 1.0
    last = stageIndex + 1 >= acParams.rkStages
    alphaS = 1.0 if last else float(problem.tableau.c[stageIndex + 1])
    return 1.0 + alphaS * problem.dtau * problem.bdf[0]


def stageSlopes(problem: DualTimeProblem, stageIndex, stage0, slopes, diffusion, flags, bdfRates):
    """`(kx, kv, kp)` of RK stage `stageIndex`, fluid rows only: wall and ghost rows are geometry
    plus closures (wall pressure, velocity mirror), so their increments are zero. `stage0` is
    `(x, v, p)` at the start of the iteration, `slopes` the `(kx, kv, kp)` lists of earlier
    stages, `bdfRates` the `(dxdtBdf, dvdtBdf)` source."""
    acParams, tableau, dtau = problem.schemeConfig.acParams, problem.tableau, problem.dtau
    xs, vs, ps = stage0
    for l in range(stageIndex):
        a = float(tableau.a[stageIndex, l])
        if a == 0.0:
            continue
        xs = xs + dtau * a * slopes[0][l]
        vs = vs + dtau * a * slopes[1][l]
        ps = ps + dtau * a * slopes[2][l]
    ps = projectPressure(problem.contact, ps)

    # D5: the velocity the position row sees; for target 'stage'/'both' it is also the velocity
    # this stage's residual reads
    vsBounced = bounceVelocities(problem.bounce, xs, vs)
    vsResidual = vs if acParams.wallBounceTarget == 'position' else vsBounced
    view = workingState(problem.state, xs, vsResidual, ps)
    view.pressures = closedWallPressure(problem, view, flags)
    # the wall velocity is a closure of the *current* fluid velocity, re-applied per stage
    view.velocities = computeBoundaryVelocities(view, problem.config, problem.schemeConfig, problem.adjacency)
    rP, rV = spatialResidual(view, problem.config, problem.schemeConfig, problem.adjacency,
                             problem.k1, problem.k2, diffusion, problem.bodyForce, problem.nu,
                             unilateralWall=acParams.cavitationProjection == 'wall',
                             taper=flags.taper)
    alphaPI = pointImplicit(problem, stageIndex)
    zero = torch.zeros_like
    dxdtBdf, dvdtBdf = bdfRates
    return (torch.where(problem.fluidRows, (vsBounced - dxdtBdf) / alphaPI, zero(vsBounced)),
            torch.where(problem.fluidRows, (rV - dvdtBdf) / alphaPI, zero(rV)),
            torch.where(problem.fluid, rP / alphaPI, zero(rP)))


def iterate(problem: DualTimeProblem, x, v, p):
    """One dual-time pseudo-iteration: `(x, v, p) -> (x, v, p, eps_v)`."""
    acParams, dtau = problem.schemeConfig.acParams, problem.dtau
    alphaT, betaT, gammaT = problem.bdf
    xPrev, xPrev2, _, _ = problem.history
    with record_function("[dualTime] stage-0 closures (BDF source, surface, wall pressure, diffusion)"):
        # the BDF source is evaluated at the FROZEN stage-0 value (Eq. 41's `u^{n+1,m+1,0}`)
        bdfRates = bdfSource(problem, x, v)
        stage0 = workingState(problem.state, x, v, p)
        flags = surfaceAt(problem, stage0)
        stage0.pressures = closedWallPressure(problem, stage0, flags)
        diffusion = computePressureSmoothing(stage0, problem.config, problem.schemeConfig, problem.adjacency,
                                             flags.renormalization, pressures=stage0.pressures)

    kx, kv, kp = [], [], []
    for s in range(acParams.rkStages):
        with record_function("[dualTime] RK stage slopes (residual)"):
            dx_, dv_, dp_ = stageSlopes(problem, s, (x, v, p), (kx, kv, kp), diffusion, flags, bdfRates)
        kx.append(dx_), kv.append(dv_), kp.append(dp_)

    with record_function("[dualTime] RK accumulation and projection"):
        xN, vN, pN = x, v, p
        for l in range(acParams.rkStages):
            b = float(problem.tableau.b[l])
            if b == 0.0:
                continue
            xN = xN + dtau * b * kx[l]
            vN = vN + dtau * b * kv[l]
            pN = pN + dtau * b * kp[l]
        pN = projectPressure(problem.contact, pN)
        if problem.bounce is not None and acParams.wallBounceTarget == 'both':
            vN = bounceVelocities(problem.bounce, xN, vN)

    with record_function("[dualTime] convergence metric (device)"):
        # `tilde v = v - Dx/Dt`: zero exactly when the position row holds at level n+1
        vPos = bounceVelocities(problem.bounce, xN, vN)
        tildeV = vPos - (alphaT * xN + betaT * xPrev + gammaT * xPrev2)
        epsT = convergenceMetricDevice(tildeV, vN, problem.fluidIndex, problem.nFluid, problem.schemeConfig)
    return xN, vN, pN, epsT


# -- the loop -----------------------------------------------------------------------------

def graphedIteration(problem: DualTimeProblem):
    """`iterate` captured into a CUDA graph (once per real step; the first capture of a run is
    checked bitwise against eager, utils/cudaGraph.py), or None when graphs are off."""
    schemeConfig = problem.schemeConfig
    if not (getattr(schemeConfig, 'cudaGraph', False) and problem.state.positions.is_cuda
            and getattr(schemeConfig, '_acsphGraphDisabled', None) is None):
        return None
    from ...utils.cudaGraph import GraphedTensorFunction
    return GraphedTensorFunction(
        lambda x, v, p: iterate(problem, x, v, p), 'ACSPH dual-time pseudo-iteration',
        validate=not getattr(schemeConfig, '_acsphGraphValidated', False))


def solveDualTime(problem: DualTimeProblem, x0, v0, p0, verbose=False) -> DualTimeResult:
    """Iterate to `eps_v < acParams.epsilonV` or `maxPseudoIterations`. `eps_v` stays on the device
    and is read back only at the check schedule's checkpoints (each read is a sync)."""
    schemeConfig = problem.schemeConfig
    acParams = schemeConfig.acParams
    x, v, p = x0, v0, projectPressure(problem.contact, p0)
    schedule = ConvergenceCheckSchedule(
        schemeConfig, 'acsphDualTime', resolveCheckSchedule(acParams, problem.config),
        max(0, acParams.minPseudoIterations - 1), acParams.maxPseudoIterations, verbose)
    graphed = graphedIteration(problem)
    epsHistory = []
    iterations = 0
    for m in range(acParams.maxPseudoIterations):
        iterations = m + 1
        if graphed is not None and m >= 1:
            with record_function("[dualTime] graphed iteration (capture on the first, then replay)"):
                x, v, p, epsT = graphed(x, v, p)
            if graphed.disabled is not None:
                schemeConfig._acsphGraphDisabled = graphed.disabled
                graphed = None
            else:
                schemeConfig._acsphGraphValidated = True
        else:
            with record_function("[dualTime] eager iteration"):
                x, v, p, epsT = iterate(problem, x, v, p)
        epsHistory.append(epsT)
        with record_function("[dualTime] stop test (host read at schedule checkpoints)"):
            if schedule.check(m) and bool(epsT < acParams.epsilonV):
                break
    with record_function("[dualTime] eps history read (host)"):
        epsHost = torch.stack(epsHistory).cpu().tolist()
    schedule.record([e < acParams.epsilonV for e in epsHost])
    return DualTimeResult(x, v, p, iterations, epsHost[-1])


def exactDeltaUpdate(result: DualTimeResult, state, dt, bdfOrder) -> ArtificialCompressibleSystemUpdate:
    """The converged step as an exact delta, `(u^{n+1} - u^n)/dt`: forward Euler on it
    is the identity, so the runner reproduces the converged state byte for byte."""
    update = ArtificialCompressibleSystemUpdate(
        dxdt=(result.positions - state.positions) / dt,
        dvdt=(result.velocities - state.velocities) / dt,
        dpdt=(result.pressures - state.pressures) / dt,
        passive=torch.zeros(state.pressures.shape, device=state.pressures.device, dtype=torch.bool),
    )
    update.pseudoIterations = result.iterations
    update.epsilonV = result.epsilonV
    update.bdfOrder = bdfOrder
    return update


def zeroNonFluidRows(update, state) -> None:
    """Wall and ghost rows are geometry plus closures, not unknowns: their delta is zero
    (applied after the boundary-condition updates, which may touch them)."""
    nonFluid = (state.kinds != 0).unsqueeze(-1)
    update.dxdt = torch.where(nonFluid, torch.zeros_like(update.dxdt), update.dxdt)
    update.dvdt = torch.where(nonFluid, torch.zeros_like(update.dvdt), update.dvdt)
    update.dpdt = torch.where(nonFluid.squeeze(-1), torch.zeros_like(update.dpdt), update.dpdt)
