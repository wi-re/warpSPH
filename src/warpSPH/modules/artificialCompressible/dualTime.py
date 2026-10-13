"""The ACSPH dual-time solver (De Courcy et al. 2024, Eqs. 38-48).

Each real step runs a pseudo-time loop to steady state; `DualTimeSolver.solve`
is that loop, `DualTimeSolver.iterate` one pseudo-iteration

    u0 = u                                   # frozen for the BDF source
    D^p = pressureSmoothing(u0)              # frozen across RK stages
    for s in 1..rkStages:
        r  = spatial residual at u^{s-1}     # Eqs. 23, 25, 26
        r* = (r - I_c (alpha_t u0 + beta_t u^n + gamma_t u^{n-1})) / alpha_PI
        u^s = u0 + dtau sum_l a_{s,l} r*_l
    u = u0 + dtau sum_l b_l r*_l

`I_c = diag{0, 1, 1}`: the continuity equation has no real-time derivative, which
is what makes `r* -> 0` enforce `div v = 0` *at* time level n+1. The RK sweep is the
general explicit Butcher form (`warpSPHIntegrators.getButcherTableau`; ACSPH_PLAN.md
Sec. 5.2), `k2 D^p` carries no extra `h` (Sec. 5.3), and the diffusion is frozen across
the stages but re-evaluated every iteration.

`tilde v = v - Dx/Dt` is simultaneously the position-row residual and the convergence
metric (Eq. 26 / Sec. 1.6).
"""

from dataclasses import dataclass
from typing import Any, Optional

import torch
from warpSPHIntegrators.butcher import getButcherTableau

from ...systems.artificialCompressible import ArtificialCompressibleSystemUpdate
from ..incompressible.convergence import ConvergenceCheckSchedule
from ..mdbc import computeBoundaryVelocities
from .parameters import TABLEAUS, convergenceMetricDevice
from .pressureSmoothing import computePressureSmoothing
from .residual import spatialResidual
from .surfaceFlags import SurfaceFlags, detectSurface
from .wallClosure import ContactMasks, WallBounce, wallPressures
from .workingState import workingState


@dataclass
class DualTimeResult:
    positions: torch.Tensor
    velocities: torch.Tensor
    pressures: torch.Tensor
    iterations: int
    epsilonV: float


class DualTimeSolver:
    """The dual-time loop of one real step. Everything it holds (`dtau`, the BDF
    coefficients and history, `k1`, `k2`, `nu`, the adjacency, the masks, the surface
    flags) is fixed for the step."""

    def __init__(self, state, config, schemeConfig, adjacency, *, dtau, k1, k2, nu,
                 bdf, history, bodyForce, wallBodyForce, surface: SurfaceFlags,
                 contact: ContactMasks, bounce: Optional[WallBounce]):
        self.state, self.config, self.schemeConfig, self.adjacency = state, config, schemeConfig, adjacency
        self.acParams = schemeConfig.acParams
        self.dtau, self.k1, self.k2, self.nu = dtau, k1, k2, nu
        self.alphaT, self.betaT, self.gammaT = bdf
        self.xPrev, self.xPrev2, self.vPrev, self.vPrev2 = history
        self.bodyForce, self.wallBodyForce = bodyForce, wallBodyForce
        self.surface, self.contact, self.bounce = surface, contact, bounce
        self.tableau = getButcherTableau(TABLEAUS[self.acParams.rkStages])
        self.fluid = state.kinds == 0
        self.fluidRows = self.fluid.unsqueeze(-1)
        self.fluidIndex = self.fluid.nonzero().squeeze(1)
        self.nFluid = int(self.fluidIndex.numel())

    # -- pieces of one iteration ------------------------------------------------

    def _surfaceAt(self, stage) -> SurfaceFlags:
        """The frozen per-step flags, or a fresh detection at `stage` (D2)."""
        if not self.acParams.redetectSurfacePerIteration:
            return self.surface
        return detectSurface(stage, self.config, self.schemeConfig, self.adjacency, store=False)

    def _wallPressure(self, view, flags: SurfaceFlags):
        return wallPressures(view, self.config, self.adjacency, self.wallBodyForce,
                             edgeRelax=self.acParams.wallPressureEdgeRelax,
                             edgeFlags=flags.rawEdge, lambdas=flags.lambdas)

    def _bounced(self, positions, velocities):
        return velocities if self.bounce is None else self.bounce(positions, velocities)

    def _bdfSource(self, x, v):
        return (self.alphaT * x + self.betaT * self.xPrev + self.gammaT * self.xPrev2,
                self.alphaT * v + self.betaT * self.vPrev + self.gammaT * self.vPrev2)

    def _pointImplicit(self, stageIndex):
        """`alpha_PI = 1 + alpha_s dtau alpha_t` (Eqs. 43-45); `alpha_s` is the
        fraction of `dtau` at which stage `s`'s residual is applied, i.e. the node
        `c[s+1]` of the stage it produces and 1 for the last. Exact for the RK2
        midpoint tableau; ambiguous for RK3/RK4 where Eq. (40) and Fig. 1 disagree
        (ACSPH_PLAN.md Sec. 5.2), and the paper reports `alpha_PI = 1` works."""
        if not self.acParams.usePointImplicit:
            return 1.0
        last = stageIndex + 1 >= self.acParams.rkStages
        alphaS = 1.0 if last else float(self.tableau.c[stageIndex + 1])
        return 1.0 + alphaS * self.dtau * self.alphaT

    def _stageSlopes(self, stageIndex, xStage0, vStage0, pStage0, slopes, diffusion, flags, dxdtBdf, dvdtBdf):
        """`(kx, kv, kp)` of RK stage `stageIndex`, fluid rows only: wall and ghost
        rows are geometry plus closures (wall pressure, velocity mirror), so their
        increments are zero."""
        acParams, tableau, dtau = self.acParams, self.tableau, self.dtau
        xs, vs, ps = xStage0, vStage0, pStage0
        for l in range(stageIndex):
            a = float(tableau.a[stageIndex, l])
            if a == 0.0:
                continue
            xs = xs + dtau * a * slopes[0][l]
            vs = vs + dtau * a * slopes[1][l]
            ps = ps + dtau * a * slopes[2][l]
        ps = self.contact.project(ps)

        # D5: the velocity the position row sees; for target 'stage'/'both' it is
        # also the velocity this stage's residual reads
        vsBounced = self._bounced(xs, vs)
        vsResidual = vs if acParams.wallBounceTarget == 'position' else vsBounced
        view = workingState(self.state, xs, vsResidual, ps)
        view.pressures = self.contact.project(self._wallPressure(view, flags))
        # the wall velocity is a closure of the *current* fluid velocity, re-applied per stage
        view.velocities = computeBoundaryVelocities(view, self.config, self.schemeConfig, self.adjacency)
        rP, rV = spatialResidual(view, self.config, self.schemeConfig, self.adjacency,
                                 self.k1, self.k2, diffusion, self.bodyForce, self.nu,
                                 unilateralWall=acParams.cavitationProjection == 'wall',
                                 taper=flags.taper)
        alphaPI = self._pointImplicit(stageIndex)
        zero = torch.zeros_like
        return (torch.where(self.fluidRows, (vsBounced - dxdtBdf) / alphaPI, zero(vsBounced)),
                torch.where(self.fluidRows, (rV - dvdtBdf) / alphaPI, zero(rV)),
                torch.where(self.fluid, rP / alphaPI, zero(rP)))

    def iterate(self, x, v, p):
        """One dual-time pseudo-iteration: `(x, v, p) -> (x, v, p, eps_v)`."""
        acParams, dtau = self.acParams, self.dtau
        # the BDF source is evaluated at the FROZEN stage-0 value (Eq. 41's `u^{n+1,m+1,0}`)
        dxdtBdf, dvdtBdf = self._bdfSource(x, v)
        stage0 = workingState(self.state, x, v, p)
        flags = self._surfaceAt(stage0)
        stage0.pressures = self.contact.project(self._wallPressure(stage0, flags))
        diffusion = computePressureSmoothing(stage0, self.config, self.schemeConfig, self.adjacency,
                                             flags.renormalization, pressures=stage0.pressures)

        kx, kv, kp = [], [], []
        for s in range(acParams.rkStages):
            dx_, dv_, dp_ = self._stageSlopes(s, x, v, p, (kx, kv, kp), diffusion, flags, dxdtBdf, dvdtBdf)
            kx.append(dx_), kv.append(dv_), kp.append(dp_)

        xN, vN, pN = x, v, p
        for l in range(acParams.rkStages):
            b = float(self.tableau.b[l])
            if b == 0.0:
                continue
            xN = xN + dtau * b * kx[l]
            vN = vN + dtau * b * kv[l]
            pN = pN + dtau * b * kp[l]
        pN = self.contact.project(pN)
        if self.bounce is not None and acParams.wallBounceTarget == 'both':
            vN = self.bounce(xN, vN)

        # `tilde v = v - Dx/Dt`: zero exactly when the position row holds at level n+1
        vPos = self._bounced(xN, vN)
        tildeV = vPos - (self.alphaT * xN + self.betaT * self.xPrev + self.gammaT * self.xPrev2)
        return xN, vN, pN, convergenceMetricDevice(tildeV, vN, self.fluidIndex, self.nFluid, self.schemeConfig)

    # -- the loop ---------------------------------------------------------------

    def solve(self, x0, v0, p0, verbose=False) -> DualTimeResult:
        """Iterate to `eps_v < acParams.epsilonV` or `maxPseudoIterations`. `eps_v`
        stays on the device and is read back only at the check schedule's checkpoints
        (each read is a sync). With `schemeConfig.cudaGraph`, iterations 2.. replay
        from a graph captured once per real step; the first capture of a run is
        checked bitwise against eager (utils/cudaGraph.py)."""
        acParams, schemeConfig = self.acParams, self.schemeConfig
        x, v, p = x0, v0, self.contact.project(p0)
        schedule = ConvergenceCheckSchedule(
            schemeConfig, 'acsphDualTime', getattr(acParams, 'convergenceCheckSchedule', 'adaptive'),
            max(0, acParams.minPseudoIterations - 1), acParams.maxPseudoIterations, verbose)
        graphed = self._graphedIteration()
        epsHistory = []
        iterations = 0
        for m in range(acParams.maxPseudoIterations):
            iterations = m + 1
            if graphed is not None and m >= 1:
                x, v, p, epsT = graphed(x, v, p)
                if graphed.disabled is not None:
                    schemeConfig._acsphGraphDisabled = graphed.disabled
                    graphed = None
                else:
                    schemeConfig._acsphGraphValidated = True
            else:
                x, v, p, epsT = self.iterate(x, v, p)
            epsHistory.append(epsT)
            if schedule.check(m) and bool(epsT < acParams.epsilonV):
                break
        epsHost = torch.stack(epsHistory).cpu().tolist()
        schedule.record([e < acParams.epsilonV for e in epsHost])
        return DualTimeResult(x, v, p, iterations, epsHost[-1])

    def _graphedIteration(self):
        schemeConfig = self.schemeConfig
        if not (getattr(schemeConfig, 'cudaGraph', False) and self.state.positions.is_cuda
                and getattr(schemeConfig, '_acsphGraphDisabled', None) is None):
            return None
        from ...utils.cudaGraph import GraphedTensorFunction
        return GraphedTensorFunction(
            self.iterate, 'ACSPH dual-time pseudo-iteration',
            validate=not getattr(schemeConfig, '_acsphGraphValidated', False))


def bdfHistory(system, state):
    """`(xPrev, xPrev2, vPrev, vPrev2)`: the BDF history, falling back to the current
    state on the first steps."""
    xPrev = system.positionsPrev if system.positionsPrev is not None else state.positions
    vPrev = system.velocitiesPrev if system.velocitiesPrev is not None else state.velocities
    xPrev2 = system.positionsPrev2 if system.positionsPrev2 is not None else xPrev
    vPrev2 = system.velocitiesPrev2 if system.velocitiesPrev2 is not None else vPrev
    return xPrev, xPrev2, vPrev, vPrev2


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
