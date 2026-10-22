"""Local omni-style pressure iteration, with explicit policy and source units.

Scheduling is runStationary; geometry/forces are operators; source assembly
is sources. See docs/incompressible/INVENTORY_2026-10-21.md for differences
from the pinned C++ oracle and the published projection methods.
"""

from typing import Any
import torch
from warpSPHCore.linalg import runStationary
from ...configurations import BoundaryPressureMode
from .convergence import ConvergenceCheckSchedule
from .consistent import applyConsistentCoupling
from .wallPressure import wallPressureExtrapolation
from .wp_alpha import computeAlpha
from .operators import scatterDivergence as _divergence, pressureAcceleration as _pressureAccel
from .policies import OmniPressurePolicy
from .sources import omniSource
from ..analyticBoundary import resolveWall
from ..shifting.bicgstab import bicgstabSolve
from ..shifting.gmres import gmresSolve


def solveOmni(
    state: Any,
    config: Any,
    schemeConfig: Any,
    adjacency: Any,
    *,
    checkSchedule: str = "every",
    fluid: torch.Tensor,
    rho0: float,
    vEnter: torch.Tensor,
    warmStart: torch.Tensor,
    dt: float,
    mode: str,
    minIters: int,
    maxIters: int,
    tol: float,
    surfaceSource: str = "full",
    policy: OmniPressurePolicy = None,
):
    """One omniSPH pressure solve (`divergenceSolve` / `densitySolve` inner
    loop, minus the boundary-triangle passes).

    `A p = s` by relaxed Jacobi, `s` fixed from `vEnter` (the velocity after
    the accelerations accumulated so far):

      mode == 'divergence':  s = dt * _divergence(vEnter)
      mode == 'density':     s = (1 - rho/rho0) + dt * _divergence(vEnter)

    `surfaceSource` reshapes the density-error part `(1 - rho/rho0)` of the
    constant-density source (`mode == 'density'` only); `'full'` is omniSPH as
    written. At a free surface plain SPH summation density is truncated
    (`rho ~ 0.69 rho0` in the top row, `~0.94` in the second), so `'full'`
    reads a large positive source there and the solve perpetually accelerates
    the skin outward trying to "fill in" mass that kernel deficiency put out
    of reach -- that push is the residual `|v|` on `hydrostaticColumn`. The
    other modes stop demanding rho0 where it is unreachable:

      'full'   -- (1 - rho/rho0), unmodified.
      'clamp'  -- min(1 - rho/rho0, 0): one-sided, resist compression only
                  (DFSPH-proper's non-negative pressure/kappa has the same
                  effect). The bulk `dt*div` term is untouched.
      'mask'   -- drop the density-error term on `surfaceIndicators == 1`
                  rows outright; keep it in the bulk.

    Each iteration (omniSPH `computeAcceleration` + `updatePressure`):
      a_p   = _pressureAccel(p)
      Ap    = -dt**2 * _divergence(a_p)                      # the Laplacian probe
      p    <- p + (omega / alpha) * (s - Ap)                 # alpha <= 0
      p    <- max(p, 0)   only for mode == 'density'
    `alpha` is omniSPH's `fluidAlpha` = `dt*dt * computeAlpha(...)`. Rows with
    `|alpha| < 1e-25`, non-finite `p`, or `|p| > 1e25` are zeroed
    (omniSPH: `pressure = residual = 0`). Convergence:
    `mean_fluid(max(Ap - s, -1e-3)) <= tol` after at least `minIters`.
    """
    policy = OmniPressurePolicy() if policy is None else policy
    if mode == "divergence" and getattr(schemeConfig, "projection", "jacobi") == "compact":
        # the divergence stage as the compact approximate projection (`modules/incompressible/compactProjection.py`, DFSPH2D `projection='compact'`): converged CG instead of 3 relaxed Jacobi sweeps
        from ...modules.incompressible.compactProjection import CompactCG, solveCompactProjection

        wall = resolveWall(state, config, schemeConfig, adjacency)
        if wall is None and bool((state.kinds != 0).any()):
            raise NotImplementedError(
                "projection='compact' is written for analytic walls (the compact Laplacian has no boundary-particle terms)"
            )
        solver = getattr(schemeConfig, "_compactCG", None)
        if solver is None:
            solver = schemeConfig._compactCG = CompactCG(
                warmStart=bool(getattr(schemeConfig, "projectionWarmStart", True))
            )
        return solveCompactProjection(
            state,
            config,
            schemeConfig,
            adjacency,
            fluid=fluid,
            rho0=rho0,
            vEnter=vEnter,
            dt=dt,
            wall=wall,
            solver=solver,
        )
    with applyConsistentCoupling(
        state, config, schemeConfig, adjacency, BoundaryPressureMode.consistent
    ):
        apparent = state.masses / state.densities
        # analytic walls (`modules/analyticBoundary`): the wall's integrals at the fluid particles, one evaluation per position set (the cache is shared with the
        # density, the alpha and the pressure acceleration below); None with boundary particles
        wall = resolveWall(state, config, schemeConfig, adjacency)
        if mode == "divergence" and not getattr(schemeConfig, "analyticWallInDivergence", False):
            # omniSPH's divergence solve ignores the wall (and the boundaries repo's DFSPH2D follows it for static walls, `boundaryInDivergence`); the density solve carries it.
            # `False`, not `None`: the modules that resolve the wall themselves (`computeAlpha`) read `None` as "take the scheme's wall", and the diagonal must not carry
            # a wall the operator does not (a first-row alpha 3x too small: Jacobi unstable at n_h = 4, OPEN_PROBLEMS 31)
            wall = False
        # omniSPH folds -dt**2 into fluidAlpha; computeAlpha returns the
        # negated IISPH a_ii bracket, so `alpha <= 0` as omniSPH's is.
        alpha = (
            dt
            * dt
            * computeAlpha(
                state,
                config,
                schemeConfig,
                adjacency,
                apparentVolumes=apparent,
                includeBoundaryReaction=False,
                wall=wall,
            )
        )
        alphaBad = alpha.abs() < policy.ALPHA_FLOOR
        if wall is False:
            wall = None  # the operators below take `None` as "no wall" (they never resolve one themselves)

        divEnter = _divergence(state, config, adjacency, vEnter, wall, bodyVelocity=True)
        source = omniSource(
            state,
            config,
            adjacency,
            rho0=rho0,
            dt=dt,
            divergence=divEnter,
            mode=mode,
            surfaceSource=surfaceSource,
            wall=wall,
        )
        source = torch.where(fluid, source, torch.zeros_like(source))

        # Closed-domain compatibility projection (see `policy.CD_SOURCE_PROJECT`).
        project = False
        if mode == "density" and policy.CD_SOURCE_PROJECT != "off" and bool(fluid.any()):
            bm = source[fluid].mean()
            sn = source[fluid].norm().clamp_min(1e-30)
            fracUniform = 1.0 - float((source[fluid] - bm).norm() / sn)
            project = (
                policy.CD_SOURCE_PROJECT == "always"
                or bool(
                    getattr(schemeConfig, "closedDomain", False)
                )  # `closedDomain`: no free surface (DFSPH2D closedDomain): the mean of the source is in the null space of the operator
                or fracUniform > policy.CD_PROJECT_THRESHOLD
            )
            if project:
                source = source - torch.where(fluid, bm.expand_as(source), torch.zeros_like(source))

        # Tikhonov diagonal shift for the near-singular free-surface operator
        # (see `policy.CD_TIKHONOV`). Applied only for the density solve when the
        # compatibility projection did NOT fire (closed box / periodic need no
        # shift). Uniform *absolute* shift `tik * med|alpha_fluid|`: a shift
        # relative to each row's own `alpha` leaves the kernel-deficient
        # near-surface rows (tiny `|alpha|`) unregularised, which is exactly
        # where the near-null space lives.
        tik = policy.CD_TIKHONOV if (mode == "density" and not project) else 0.0
        if tik and bool(fluid.any()):
            shift = tik * float(alpha[fluid].abs().median())
        else:
            shift = 0.0
        alphaEff = alpha - shift
        invAlpha = policy.OMEGA / alphaEff

        # the boundary particles' per-iterate wall pressure extrapolation has no analytic counterpart: the wall's pressure is the mirrored `p_i^+` plus the hydrostatic term
        wallP = policy.WALL_PRESSURE_MODE if (mode == "density" and wall is None) else None

        # analyticWallPressure = 'mls' (DFSPH2D wallPressure='linear'): the wall's pressure gradient fitted to the fluid's current pressure every iterate (the fit's geometry once per solve)
        mlsFit = None
        if (
            wall is not None
            and getattr(schemeConfig, "analyticWallPressure", "hydrostatic") == "mls"
        ):
            from ...modules.analyticBoundary import buildMLSPressureFit

            mlsFit = buildMLSPressureFit(state, config, adjacency, fluid, rho0)

        def accel(pt, clampWall=True):
            pin = (
                wallPressureExtrapolation(
                    state, config, adjacency, pt, fluid, mode=wallP, clampNonNeg=clampWall
                )
                if wallP
                else pt
            )
            return _pressureAccel(state, config, adjacency, pin, fluid, wall, rho0, mlsFit)

        def applyA(pt, a_p):
            """`A_shift p = -dt**2 div(a_p(p)) - shift*p`, masked to fluid
            (`shift >= 0`, so `- shift*p` deepens the negative diagonal)."""
            aP = -dt * dt * _divergence(state, config, adjacency, a_p, wall)
            if shift:
                aP = aP - shift * pt
            return torch.where(fluid, aP, torch.zeros_like(aP))

        # --- non-symmetric Krylov density solve (Part 43; policy.CD_SOLVER) ---------
        # The relaxed Jacobi below stalls on a free surface (Part 42). Drive
        # the same linear system `A p = s` with BiCGStab / GMRES instead. The
        # operator must be linear, so the wall-pressure closure runs
        # `clampNonNeg=False` and the `p >= 0` free-surface guard is applied
        # once, after the solve. The near-singular free-surface operator needs
        # `policy.CD_TIKHONOV` (a diagonal shift) to keep the solution bounded; even
        # then a Krylov breakdown can hand back a wild iterate, so the result
        # is only accepted if its true residual actually beats `|source|`
        # (otherwise the step takes no density correction -- like a skipped
        # solve -- rather than detonating).
        if mode == "density" and policy.CD_SOLVER != "jacobi":
            alphaSafe = torch.clamp(alphaEff, max=-1e-6)
            precond = torch.where(fluid, 1.0 / alphaSafe, torch.zeros_like(alpha))

            def matvec(pt):
                pt = torch.where(fluid, pt, torch.zeros_like(pt))
                return applyA(pt, accel(pt, clampWall=False))

            # No warm start: the CD field "integrates" (ranked-queue item 3),
            # so a carried iterate poisons the Krylov recurrence -- start clean.
            x0 = torch.zeros_like(source)
            if policy.CD_SOLVER == "bicgstab":
                x, _status, conv = bicgstabSolve(
                    matvec,
                    source,
                    x0,
                    rtol=policy.CD_KRYLOV_RTOL,
                    maxiter=policy.CD_KRYLOV_MAXITER,
                    precond=precond,
                    dim=1,
                )
            elif policy.CD_SOLVER == "gmres":
                x, _status, conv = gmresSolve(
                    matvec,
                    source,
                    x0,
                    rtol=policy.CD_KRYLOV_RTOL,
                    maxiter=policy.CD_KRYLOV_MAXITER,
                    precond=precond,
                    restart=policy.CD_KRYLOV_RESTART,
                    dim=1,
                )
            else:
                raise ValueError(f"Unknown policy.CD_SOLVER: {policy.CD_SOLVER!r}")

            # `conv[-1]` is the returned iterate's verified true residual norm
            # `|A x - s|`. Reject the solve unless it (a) reduced the residual
            # below `|s|` AND (b) returned a bounded solution -- a near-singular
            # operator hands back a tiny-residual, huge-norm iterate along its
            # near-null space (`|p| ~ 1e9` on `hydrostaticColumn` step 1), which
            # detonates the run. `|x|_2 <= 1e3 * |s|` is the physical bound
            # (pressure should not be orders of magnitude above the
            # density-error source). On rejection the step takes no density
            # correction, like a skipped solve.
            srcNorm = float(source.norm())
            resNorm = float(conv[-1]) if len(conv) else srcNorm
            reject = (
                not (resNorm < srcNorm)
                or not bool(torch.isfinite(x).all())
                or float(x.norm()) > 1e3 * max(srcNorm, 1e-12)
                or float(x.abs().max()) > 1e5
            )
            if reject:
                return (torch.zeros_like(vEnter), torch.zeros_like(source), len(conv), resNorm)

            if project:
                x = x - torch.where(fluid, x[fluid].mean().expand_as(x), torch.zeros_like(x))
            else:
                x = x.clamp(min=0.0)
            bad = alphaBad | (~torch.isfinite(x)) | (x.abs() > 1e25) | ~fluid
            x = torch.where(bad, torch.zeros_like(x), x)
            return accel(x), x, len(conv), resNorm

        p = torch.where(fluid, warmStart, torch.zeros_like(warmStart))
        # The loop, its device-side stopping test and the one history read are `warpSPHCore.linalg.runStationary` (host syncs only at the
        # check schedule's checkpoints, `convergence.py` / `convergenceCheckSchedule`): the error stays on the device, and the fluid rows are
        # gathered through an index computed once per solve, which selects exactly what `x[fluid]` did (bitwise-equal statistic).
        hasFluid = bool(fluid.any())
        fluidIndex = fluid.nonzero().squeeze(1) if hasFluid else None
        # `it + 1 >= minIters` here is `it >= minIters - 1` in the schedule's terms
        schedule = ConvergenceCheckSchedule(
            schemeConfig,
            f"omniIncompressible.{mode}",
            checkSchedule,
            max(0, minIters - 1),
            maxIters,
            policy.VERBOSE_SOLVE,
        )

        def step(p, it):
            a_p = accel(p)
            aP = applyA(p, a_p)
            p = p + invAlpha * (source - aP)
            if mode == "density":
                if project:
                    p = p - torch.where(
                        fluid,
                        p.index_select(0, fluidIndex).mean().expand_as(p),
                        torch.zeros_like(p),
                    )
                else:
                    p = p.clamp(min=0.0)
            bad = alphaBad | (~torch.isfinite(p)) | (p.abs() > 1e25) | ~fluid
            p = torch.where(bad, torch.zeros_like(p), p)
            if hasFluid:
                residual = aP - source
                errT = (
                    torch.clamp(residual, min=policy.RESIDUAL_FLOOR)
                    .index_select(0, fluidIndex)
                    .mean()
                )
            else:
                errT = torch.zeros((), dtype=p.dtype, device=p.device)
            return p, errT, None

        out = runStationary(
            step, p, maxIters=maxIters, tolerance=tol, inclusive=True, schedule=schedule
        )
        p = out.x
        it = out.iterations - 1
        err = out.stats[-1] if out.stats else 0.0
        if policy.VERBOSE_SOLVE:
            print(
                f"[{mode}] {it + 1} iters, err {err:.3g}, p[{float(p.min()):+.3g}, {float(p.max()):+.3g}]"
            )
        a_p = accel(p)
    return a_p, p, it + 1, err
