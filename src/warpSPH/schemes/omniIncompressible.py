"""Direct port of the omniSPH incompressible solver loop
(`~/dev/omniSPH/simulation/{SPH,fluidMechanics}.cpp`).

This is a clean transcription of the DFSPH-style two-solve step omniSPH runs
in `SPHSimulation::timestep`, kept deliberately literal: no free-surface
pressure gauge, no particle-deficiency guard, no damped warm start, no
surface/deficiency masking -- only the loop as omniSPH runs it.

omniSPH per-step (`timestep`):

  1. neighbourhood + summation density (boundary particles included)
  2. external forces -> `a`  (gravity only)
  3. `divergenceSolve()`  -- exactly 3 relaxed-Jacobi iterations of `A p = s_div`
     with `s_div` the predicted-velocity divergence; `a += a_p^div`
  4. `densitySolve()`     -- relaxed-Jacobi of `A p = s_rho`, min 3 / max 256
     iterations, warm-started from `0.5 * p_prior`; `a += a_p^rho`
  5. XSPH velocity filter (`XSPH` + `BXSPH`): isotropic smoothing of `v`
     toward the neighbourhood mean + a static-wall drag (`XSPH_FLUID` /
     `XSPH_BOUNDARY`; 0/0 = the faithful no-dissipation loop)
  6. integrate: `v += dt a`,  `v *= (1 - damping)`,  `x += dt v`

Both solves run on the SAME positions and the SAME neighbourhood (omniSPH does
not move `x` between them -- the only `x` update is the final symplectic step),
and every pressure acceleration is accumulated into one `a` that a single
semi-implicit Euler step consumes. That accumulate-then-integrate structure,
and the two-solve order with omniSPH's own relaxation (`omega = 0.5`) and
iteration budgets, is what this module reproduces. Contrast `dfsphReference`
(SPlisHSPlasH order: an `x` advance and a neighbourhood rebuild between the
two solves, `v += dt a_p` applied per solve) and `iisph` (constant-density
solve only).

Operator mapping (omniSPH C++  ->  warpSPH composed operator):

  * `computeAlpha`      -> `computeAlpha(...)` with the IISPH `a_ii` bracket,
    `includeBoundaryReaction=False` (a static particle takes no reaction, so
    it enters only the vector-sum term) and `apparentVolumes = m/rho`. omniSPH
    folds `-dt**2` into `fluidAlpha`; here `alpha = dt*dt * computeAlpha(...)`
    (`computeAlpha` already returns the negated bracket, so `alpha <= 0`).
  * `sum_j V_j (X_i - X_j) . gradW`  -> `-_divergence(X)`, where `_divergence`
    is the scatter / difference-form `WarpOperation.Divergence` (the same
    operator `computeMomentumIncompressible` wraps: for a `div = +1` field it
    returns `+1` in the bulk). So omniSPH's source term
    `s = (rho_target - rho) - dt sum_j V_j (v_i^* - v_j^*) . gradW`
    becomes `s = (rho_target - rho) + dt * _divergence(v^*)`, and the
    Laplacian probe `dt**2 sum_j V_j (a_i - a_j) . gradW` becomes
    `A p = -dt**2 * _divergence(a_p(p))`.
  * `computeAcceleration` (symmetric `-sum_j m_j (p_i/rho_i^2 + p_j/rho_j^2)
    gradW`)  -> `computePressureAccelIISPH`, masked to the fluid rows (a
    static particle takes no reaction).

Boundary: `kind == 1` particles enter both solves as "static fluid at rho0"
with Akinci rest-density volumes via `applyConsistentCoupling`
(`BoundaryPressureMode.consistent`) -- the particle-boundary analogue of
omniSPH's triangle `boundaryFunc`, and the same coupling `iisph` /
`dfsphReference` use. On top of that, `WALL_PRESSURE_MODE` (default `'shepard'`,
Part 42) ports omniSPH's per-iterate boundary-pressure extrapolation into the
*density* solve: every density-mode Jacobi iterate recomputes `p_b` on the
`kind == 1` rows from the current fluid pressure and feeds its gradient into
`a_p`, the Robin closure that makes the near-wall iteration contract (without
it `hydrostaticColumn` nx=128 diverges at the bottom corners by step ~10 --
Part 35/41). `'shepard'` is the zero-order mirror; `'mls'` (Part 41's first
default) adds omniSPH's linear `beta*x + gamma*y` term but assumes a
locally-linear near-wall pressure and diverges on the sheared
`randomFlowIncompressible --bounded` (Part 42), so `'shepard'` -- which holds
both -- is the default. `None` falls back to Bender-Westhofen-Jeske 2023
Eq. 33 at zero boundary pressure. The divergence solve always runs at zero
boundary pressure (omniSPH's `divergenceSolve` has no wall-pressure term).
See `WALL_PRESSURE_MODE` below.

The density solve is omniSPH's relaxed Jacobi by default; `CD_SOLVER` can
swap in a non-symmetric Krylov method (BiCGStab / GMRES) on the same linear
system `A p = s`, for the free-surface cases where the Jacobi stalls
(Part 42/43). See `CD_SOLVER` below.

The step does its own integration and hands the integrator a no-op update;
`DFSPHReferenceSystem` (systems/incompressible.py) copies the advanced fields
across in `finalize`.
"""

from __future__ import annotations

from typing import Any

import torch

from warpSPHCore import (GradientScheme, OperationDirection, OperationProperties,
                         SupportScheme, WarpOperation, buildVerletList,
                         warpOperation)

from ..configurations import BoundaryPressureMode
from ..modules.boundaryConditions import (computeForcing, enforceDirichlet,
                                          enforceUpdates)
from ..modules.density import computeDensities
from ..modules.gravity import computeGravity
from ..modules.incompressible.consistent import applyConsistentCoupling
from ..modules.incompressible.wallPressure import wallPressureExtrapolation
from ..modules.incompressible.wp_alpha import computeAlpha
from ..modules.pressure.iisph import computePressureAccelIISPH
from ..modules.shifting.bicgstab import bicgstabSolve
from ..modules.shifting.gmres import gmresSolve
from ..systems.incompressible import IncompressibleSystemUpdate

__all__ = ['omniIncompressible_step']

#: Relaxed-Jacobi relaxation. omniSPH's `updatePressure` hardcodes
#: `scalar omega = 0.5`, but on this codebase's *composed* pressure operator
#: the fixed-point Jacobi stability window is ~[0.2, 0.35] (measured -- see
#: `dfsphReference` Part 29 / `probe_dfsphReferenceContraction.py`), so the
#: faithful 0.5 detonates `hydrostaticColumn`'s density solve by t ~ 0.06.
#: 0.3 is the nearest stable value and holds the column bounded for 2000
#: steps. Set to 0.5 for the byte-faithful omniSPH iteration.
OMEGA = 0.3
#: omniSPH `divergenceSolve`: `while (counter++ < 3 || (error > limit && counter < 3))`
#: -- i.e. always exactly 3 iterations.
DIVERGENCE_ITERATIONS = 3
#: omniSPH `densitySolve`: `while (counter++ < 3 || (error > limit && counter < 256))`.
DENSITY_MIN_ITERATIONS = 3
DENSITY_MAX_ITERATIONS = 256
#: omniSPH `updatePressure`: `fluidDpDt[i] = max(residual, -0.001) * fluidArea[i]`
#: and the convergence sum is `mean_i max(residual_i, -0.001)`.
RESIDUAL_FLOOR = -1e-3
#: omniSPH `updatePressure`: `abs(fluidAlpha[i]) < 1e-25` -> pressure zeroed.
ALPHA_FLOOR = 1e-25
#: omniSPH `Integrate`: `vi *= (1.0 - damping)` with `props.damping` (0 here).
DAMPING = 0.0
#: Per-solve one-line iteration/residual/pressure-range trace on stdout. A
#: `c637785`-era debug leftover; `False` silences it (the physics is unchanged).
VERBOSE_SOLVE = False

#: omniSPH's XSPH velocity filter (`SPHSimulation::XSPH` / `BXSPH`), the
#: scheme's only dissipation -- a post-solve, isotropic smoothing of `v`
#: toward the neighbourhood mean:
#:     v_i <- v_i + sum_j c_j V_j W_ij (v_j - v_i)
#: with `c_j = XSPH_FLUID` for fluid neighbours and `XSPH_BOUNDARY` for the
#: static wall (v_j = 0 there, so that term is a wall drag -- omniSPH's
#: `BXSPH`, minus the wall-normal projection). omniSPH runs this *after* the
#: pressure solves and *before* the symplectic step, on the start-of-step
#: velocity (the solves only touch `accel`), so the filtered `v` and the
#: pressure impulse `dt*accel` are independent additions -- matched here.
#: This is what damps the free-slip bulk slosh the pressure solve leaves
#: behind (`dfsphReference` Part 32); at 0.0/0.0 the step is the faithful
#: no-dissipation omniSPH loop. `XSPH_FLUID ~ 0.05` is a light smoothing;
#: omniSPH's own `2*W` weighting makes its coefficient ~2x smaller for the
#: same effect.
XSPH_FLUID = 0.05
XSPH_BOUNDARY = 0.0

#: Per-iterate wall-pressure closure (Part 41). omniSPH's `densitySolve`
#: recomputes an MLS wall pressure `p_b` from the CURRENT fluid pressure every
#: Jacobi iterate and feeds its gradient into `a_p` / the Laplacian probe /
#: `alpha` -- a Robin closure that makes the near-wall iteration contract.
#: This port had the wall in `alpha` only (the Akinci band) with boundary
#: `p == 0` (BWJ23 Eq. 33), so the near-wall row is inconsistent (`D` carries
#: a wall term the operator does not) and the Jacobi does not contract at the
#: floor band -- `hydrostaticColumn` nx=128 hits the 256-iter cap with
#: `errRho` rising and blows up at the bottom corners. When set, every
#: density-mode `_solve` iterate recomputes a boundary pressure on the
#: `kind == 1` rows from the current fluid `p` and passes `p_all` into
#: `_pressureAccel`, so the symmetric gradient picks up the wall term
#: `-sum_k m_k (p_i/rho_i^2 + p_b/rho0^2) gradW_ik` (the divergence still sees
#: `a_p == 0` on the masked boundary rows, which reproduces omniSPH's
#: `a_i . gk` self-term).
#:
#:   None       -- off (faithful BWJ23 Eq. 33, boundary p == 0)
#:   'shepard'  -- zero-order mirror `p_b[k] = sum_f V_f p_f W_kf / sum_f V_f
#:                 W_kf` (omniSPH's MLS `alpha` term, no linear correction)
#:   'mls'      -- the full first-order Liu-Liu MLS extrapolation
#:                 (`modules/liu`, the same fit `computeMdbcPressure` uses):
#:                 evaluate the value+gradient fit at each ghost point and
#:                 Taylor-correct to the owning boundary particle. This is
#:                 omniSPH's `alpha + beta*x_b + gamma*y_b` in full.
#:
#: Default `'shepard'` (Part 42). Some wall-pressure closure is required:
#: with `None` the composed density Jacobi does not contract at the wall
#: band and `hydrostaticColumn` nx=128 diverges at the bottom corners by
#: step ~10 (Part 41). Part 41 first shipped `'mls'`, but its linear term
#: `beta*x + gamma*y` assumes a locally-linear near-wall pressure -- exact
#: for the hydrostatic column, wrong for a sheared flow, where it amplifies
#: the real near-wall pressure structure and pumps energy into the Jacobi:
#: `'mls'` **diverges** `randomFlowIncompressible --bounded` on step 1
#: (`errRho` 3.4e-2, `|v|max` ~84) where `None` and `'shepard'` hold it
#: (Part 42). `'shepard'` (0th order, no linear term) threads both:
#: `hydrostaticColumn` nx=128 holds (`|v|max` ~0.5, the exact hydrostatic
#: gradient) and `randomFlowIncompressible --bounded` holds (`|v|max` decays
#: to ~0.4). No relaxation / cross-step lag (unlike `computeMdbcPressure`),
#: so not the `mdbcMlsPressure` feedback instability. `'mls'` is kept as an
#: option for quiescent free-surface cases where its first-order accuracy
#: recovers a slightly better near-wall density (Part 41).
WALL_PRESSURE_MODE = 'shepard'

#: Compatibility projection of the constant-density source for a closed
#: (pure-Neumann) domain (Part 42). On a fully-walled box with no free surface
#: the pressure operator `A` is singular (`A·1 ≈ 0`), so the constant part of
#: the source `s = (1 - ρ/ρ0) + dt·div(v*)` -- which the `n_h = 4` lattice
#: density bias (§1.1) makes non-zero -- lies in `null(A)`: the relaxed Jacobi
#: cannot reduce the residual below it and instead ramps `p` linearly to the
#: iteration cap (§1.7; measured on `randomFlowIncompressible --bounded`,
#: where it diverged). The fix is the textbook pure-Neumann compatibility
#: projection: subtract `mean(s)` over the fluid and solve for the resolvable
#: spatial part with `p` kept mean-zero (a closed box has no free surface, so
#: no `p ≥ 0` tensile guard is needed). The residual `mean(ρ) ≠ ρ0` is a
#: rest-density calibration offset the solve legitimately ignores.
#:
#:   'auto'   -- project only when the source is mean-dominated, i.e.
#:               `1 - |s - mean(s)| / |s| > CD_PROJECT_THRESHOLD` (a free
#:               surface makes the spatial part large, so this is a no-op there)
#:   'always' -- project every density solve
#:   'off'    -- never (the pre-Part-42 behaviour)
CD_SOURCE_PROJECT = 'auto'
CD_PROJECT_THRESHOLD = 0.7

#: Constant-density (density-mode) linear solver (Part 43). omniSPH's own
#: iteration is the relaxed Jacobi `p <- p + (omega/alpha)(s - A p)`
#: (`'jacobi'` -- the default, and the ONLY usable setting today).
#:
#:   'jacobi'   -- omniSPH's relaxed Jacobi. The `p >= 0` free-surface guard
#:                 and (closed box) the mean-zero projection are applied every
#:                 iterate. Slow on the near-singular free-surface operator
#:                 (`CD_TIKHONOV` restores its iteration budget), but stable.
#:   'bicgstab' -- BiCGStab (`modules/shifting/bicgstab.py`) on `A p = s`,
#:                 preconditioned by the Jacobi diagonal `1/alpha`, zero warm
#:                 start, no per-iterate clamp (the operator must stay linear,
#:                 so the wall-pressure closure runs `clampNonNeg=False`).
#:   'gmres'    -- restarted GMRES(`CD_KRYLOV_RESTART`), same system / precond.
#:
#: `A p := -dt**2 * _divergence(accel(p))` with `accel` the same closure the
#: Jacobi iterate uses (wall-pressure extrapolation + symmetric IISPH
#: gradient), restricted to the fluid rows; the divergence solve is unaffected
#: (always omniSPH's exact 3 Jacobi iterations).
#:
#: **`'bicgstab'` / `'gmres'` do NOT work yet** (Part 43): the composed wall
#: operator is near-singular *and* strongly non-symmetric near the wall
#: (MINRES: `status -13`; BiCGStab: breaks down at iteration ~1-3), so a
#: matrix-free Krylov solve returns a tiny-residual / huge-norm (`|p| ~ 1e9`)
#: iterate along the near-null space -- on every wall-bounded case, free
#: surface AND compatibility-projected closed box, with or without
#: `CD_TIKHONOV`. A reject guard (see `_solve`) stops the detonation but the
#: run then loses density control. These are kept as scaffolding for the
#: principled fix -- `band2018pb` (boundary samples as solve unknowns; makes
#: the near-wall block consistent + symmetric) or an explicit symmetrisation
#: of `A` via `computePressureShiftIISPH` -- see DFSPH_IMPROVEMENT_PLAN.md
#: ranked-queue item 0.
CD_SOLVER = 'jacobi'
#: Krylov iteration cap / relative-residual tolerance / GMRES restart length,
#: used only when `CD_SOLVER != 'jacobi'`.
CD_KRYLOV_MAXITER = 256
CD_KRYLOV_RTOL = 1e-3
CD_KRYLOV_RESTART = 50

#: Tikhonov diagonal shift on the density-mode operator, as a fraction of the
#: IISPH diagonal `alpha` (Part 43). The *free-surface* constant-density
#: operator `A p = -dt**2 div(a_p(p))` is **near-singular** -- the free
#: surface pins the pressure constant only weakly (kernel deficiency), and in
#: the relaxed-Jacobi path the per-iterate `p >= 0` clamp is what actually
#: regularises it. Without the clamp (the linear system a Krylov method
#: needs, or an unclamped Jacobi), the minimum-residual solution has a huge
#: norm along the near-null space: measured on `hydrostaticColumn` nx=64
#: step 30, raw BiCGStab / GMRES drive `|r|/|b|` to 1e-2 but with
#: `|p|max ~ 450` (vs the clamped Jacobi's physical `~23`), which detonates
#: the run. Deepening the (negative) diagonal by a uniform absolute shift
#: `CD_TIKHONOV * median(|alpha_fluid|)` restores diagonal dominance and
#: solves a nearby (slightly compressible) problem -- `(A - eps|D|) p = s` --
#: the standard regularisation for a near-singular PPE. The shift is uniform,
#: not per-row proportional to `alpha`: the kernel-deficient near-surface
#: rows have tiny `|alpha|`, and that is exactly where the near-null space
#: lives, so a proportional shift would leave it unregularised. Applied to
#: both the Jacobi and the Krylov density path, ONLY when the compatibility
#: projection did NOT fire (a closed box is handled by `CD_SOURCE_PROJECT`
#: instead; a periodic case has no wall/surface singularity). Measured
#: (`hydrostaticColumn` nx=128, `CD_SOLVER='jacobi'`, 400 steps): `0.1` takes
#: the density solve off its 256-iteration cap (mean ~210 -> ~75 iters) with
#: the run quality neutral-to-better (embeddedMinDensity 0.984 -> 0.99, slope
#: 0.999 -> 1.005, maxRho 1.012 -> 1.027); a strict wash on `dambreak` (its
#: CD solve already converges in the 3-iteration minimum). `0.0` = off (the
#: pre-Part-43 behaviour, exact on the periodic / closed cases; a Krylov
#: `CD_SOLVER` needs it non-zero but does not work regardless -- see there).
CD_TIKHONOV = 0.0


def _rebuildAdjacency(state: Any, system: Any, config: Any):
    adjacency = buildVerletList(
        state, config.domain, verletScale=config.verletScale,
        supportMode=SupportScheme.SuperSymmetric,
        priorNeighborhood=system.adjacency, verbose=False)
    system.adjacency = adjacency
    return adjacency


def _divergence(state: Any, config: Any, adjacency: Any,
                field: torch.Tensor) -> torch.Tensor:
    """Scatter / difference-form SPH divergence -- the operator
    `computeMomentumIncompressible` wraps (`WarpOperation.Divergence`,
    `GradientScheme.Difference`, `SupportScheme.Scatter`), without its
    `-rho0` factor. `sum_j (m_j/rho_j) (field_j - field_i) . gradW_ij`; for a
    `div = +1` field it returns `+1` in the bulk. omniSPH's per-particle sum
    `sum_j V_j (X_i - X_j) . gradW` is the negative of this."""
    return warpOperation(
        state,
        OperationProperties(
            kernel=config.kernel,
            operation=WarpOperation.Divergence,
            gradientMode=GradientScheme.Difference,
            supportMode=SupportScheme.Scatter,
        ),
        queryValues=field,
        domain=config.domain,
        adjacency=adjacency,
        consistentDivergence=False,
    )


def _xsphFilter(state: Any, config: Any, adjacency: Any,
                fluidMask: torch.Tensor) -> torch.Tensor:
    """omniSPH's XSPH velocity filter (`SPHSimulation::XSPH` + `BXSPH`), as the
    per-fluid-row velocity increment `sum_j c_j V_j W_ij (v_j - v_i)`.

    `WarpOperation.Interpolate` computes `sum_j (m_j/rho_j) g_j W_ij` over all
    non-ghost neighbours, so folding the per-kind coefficient `c` into the
    reference values collapses the two omniSPH sums (fluid XSPH + wall drag)
    into `interp(c v) - v_i interp(c)`. The static wall enters with `v_j = 0`
    and `c_j = XSPH_BOUNDARY`, so its contribution `-c_k V_k W_ik v_i` is a
    drag; the output is masked back to the fluid rows (the wall takes no
    reaction, like the rest of this scheme)."""
    c = torch.where(fluidMask,
                    torch.full_like(state.densities, XSPH_FLUID),
                    torch.full_like(state.densities, XSPH_BOUNDARY))
    props = OperationProperties(
        kernel=config.kernel, operation=WarpOperation.Interpolate,
        supportMode=SupportScheme.SuperSymmetric)
    cv = warpOperation(state, props, domain=config.domain,
                       referenceValues=state.velocities * c.unsqueeze(-1),
                       adjacency=adjacency)
    cc = warpOperation(state, props, domain=config.domain,
                       referenceValues=c, adjacency=adjacency)
    dv = cv - state.velocities * cc.unsqueeze(-1)
    return torch.where(fluidMask.unsqueeze(-1), dv, torch.zeros_like(dv))


def _pressureAccel(state: Any, config: Any, adjacency: Any,
                   p: torch.Tensor, fluidMask: torch.Tensor) -> torch.Tensor:
    """omniSPH `computeAcceleration`: the symmetric SPH pressure gradient
    `-sum_j m_j (p_i/rho_i^2 + p_j/rho_j^2) gradW`, as an acceleration. Zeroed
    on non-fluid rows -- a static particle takes no reaction
    (`operatorMovesBoundary=False`), and ghosts are not real particles."""
    a_p = computePressureAccelIISPH(
        state=state, pressureValues=p, config=config,
        supportScheme=SupportScheme.Scatter, adjacency=adjacency)
    return torch.where(fluidMask.unsqueeze(-1), a_p, torch.zeros_like(a_p))


def _solve(state: Any, config: Any, schemeConfig: Any, adjacency: Any, *,
           fluid: torch.Tensor, rho0: float, vEnter: torch.Tensor,
           warmStart: torch.Tensor, dt: float, mode: str,
           minIters: int, maxIters: int, tol: float,
           surfaceSource: str = 'full'):
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
    with applyConsistentCoupling(state, config, schemeConfig, adjacency,
                                 BoundaryPressureMode.consistent):
        apparent = state.masses / state.densities
        # omniSPH folds -dt**2 into fluidAlpha; computeAlpha returns the
        # negated IISPH a_ii bracket, so `alpha <= 0` as omniSPH's is.
        alpha = dt * dt * computeAlpha(
            state, config, schemeConfig, adjacency,
            apparentVolumes=apparent, includeBoundaryReaction=False)
        alphaBad = alpha.abs() < ALPHA_FLOOR

        divEnter = _divergence(state, config, adjacency, vEnter)
        if mode == 'density':
            densityError = 1.0 - state.densities / rho0
            if surfaceSource == 'clamp':
                densityError = densityError.clamp(max=0.0)
            elif surfaceSource == 'mask':
                surf = getattr(state, 'surfaceIndicators', None)
                if surf is not None:
                    densityError = torch.where(
                        surf.to(torch.bool), torch.zeros_like(densityError),
                        densityError)
            elif surfaceSource == 'shepard':
                # 0th-order (Shepard) density: rho_sum / sum_j (m_j/rho_j) W_ij.
                # At a flat free surface both sums truncate the same way, so
                # this reads ~rho0 in the skin (source ~0) while still seeing
                # genuine bulk compression. No free-surface flag needed.
                props = OperationProperties(
                    kernel=config.kernel, operation=WarpOperation.Interpolate,
                    supportMode=SupportScheme.SuperSymmetric)
                weight = warpOperation(
                    state, props, domain=config.domain,
                    referenceValues=torch.ones_like(densityError),
                    adjacency=adjacency)
                rhoShep = state.densities / weight.clamp_min(1e-6)
                densityError = 1.0 - rhoShep / rho0
            elif surfaceSource != 'full':
                raise ValueError(f'Unknown surfaceSource: {surfaceSource!r}')
            source = densityError + dt * divEnter
        else:
            source = dt * divEnter
        source = torch.where(fluid, source, torch.zeros_like(source))

        # Closed-domain compatibility projection (see `CD_SOURCE_PROJECT`).
        project = False
        if mode == 'density' and CD_SOURCE_PROJECT != 'off' and bool(fluid.any()):
            bm = source[fluid].mean()
            sn = source[fluid].norm().clamp_min(1e-30)
            fracUniform = 1.0 - float((source[fluid] - bm).norm() / sn)
            project = (CD_SOURCE_PROJECT == 'always'
                       or fracUniform > CD_PROJECT_THRESHOLD)
            if project:
                source = source - torch.where(fluid, bm.expand_as(source),
                                              torch.zeros_like(source))

        # Tikhonov diagonal shift for the near-singular free-surface operator
        # (see `CD_TIKHONOV`). Applied only for the density solve when the
        # compatibility projection did NOT fire (closed box / periodic need no
        # shift). Uniform *absolute* shift `tik * med|alpha_fluid|`: a shift
        # relative to each row's own `alpha` leaves the kernel-deficient
        # near-surface rows (tiny `|alpha|`) unregularised, which is exactly
        # where the near-null space lives.
        tik = CD_TIKHONOV if (mode == 'density' and not project) else 0.0
        if tik and bool(fluid.any()):
            shift = tik * float(alpha[fluid].abs().median())
        else:
            shift = 0.0
        alphaEff = alpha - shift
        invAlpha = OMEGA / alphaEff

        wallP = WALL_PRESSURE_MODE if mode == 'density' else None

        def accel(pt, clampWall=True):
            pin = wallPressureExtrapolation(
                state, config, adjacency, pt, fluid, mode=wallP,
                clampNonNeg=clampWall) if wallP else pt
            return _pressureAccel(state, config, adjacency, pin, fluid)

        def applyA(pt, a_p):
            """`A_shift p = -dt**2 div(a_p(p)) - shift*p`, masked to fluid
            (`shift >= 0`, so `- shift*p` deepens the negative diagonal)."""
            aP = -dt * dt * _divergence(state, config, adjacency, a_p)
            if shift:
                aP = aP - shift * pt
            return torch.where(fluid, aP, torch.zeros_like(aP))

        # --- non-symmetric Krylov density solve (Part 43; CD_SOLVER) ---------
        # The relaxed Jacobi below stalls on a free surface (Part 42). Drive
        # the same linear system `A p = s` with BiCGStab / GMRES instead. The
        # operator must be linear, so the wall-pressure closure runs
        # `clampNonNeg=False` and the `p >= 0` free-surface guard is applied
        # once, after the solve. The near-singular free-surface operator needs
        # `CD_TIKHONOV` (a diagonal shift) to keep the solution bounded; even
        # then a Krylov breakdown can hand back a wild iterate, so the result
        # is only accepted if its true residual actually beats `|source|`
        # (otherwise the step takes no density correction -- like a skipped
        # solve -- rather than detonating).
        if mode == 'density' and CD_SOLVER != 'jacobi':
            alphaSafe = torch.clamp(alphaEff, max=-1e-6)
            precond = torch.where(fluid, 1.0 / alphaSafe,
                                  torch.zeros_like(alpha))

            def matvec(pt):
                pt = torch.where(fluid, pt, torch.zeros_like(pt))
                return applyA(pt, accel(pt, clampWall=False))

            # No warm start: the CD field "integrates" (ranked-queue item 3),
            # so a carried iterate poisons the Krylov recurrence -- start clean.
            x0 = torch.zeros_like(source)
            if CD_SOLVER == 'bicgstab':
                x, _status, conv = bicgstabSolve(
                    matvec, source, x0, rtol=CD_KRYLOV_RTOL,
                    maxiter=CD_KRYLOV_MAXITER, precond=precond, dim=1)
            elif CD_SOLVER == 'gmres':
                x, _status, conv = gmresSolve(
                    matvec, source, x0, rtol=CD_KRYLOV_RTOL,
                    maxiter=CD_KRYLOV_MAXITER, precond=precond,
                    restart=CD_KRYLOV_RESTART, dim=1)
            else:
                raise ValueError(f'Unknown CD_SOLVER: {CD_SOLVER!r}')

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
            reject = (not (resNorm < srcNorm)
                      or not bool(torch.isfinite(x).all())
                      or float(x.norm()) > 1e3 * max(srcNorm, 1e-12)
                      or float(x.abs().max()) > 1e5)
            if reject:
                return (torch.zeros_like(vEnter), torch.zeros_like(source),
                        len(conv), resNorm)

            if project:
                x = x - torch.where(fluid, x[fluid].mean().expand_as(x),
                                    torch.zeros_like(x))
            else:
                x = x.clamp(min=0.0)
            bad = alphaBad | (~torch.isfinite(x)) | (x.abs() > 1e25) | ~fluid
            x = torch.where(bad, torch.zeros_like(x), x)
            return accel(x), x, len(conv), resNorm

        p = torch.where(fluid, warmStart, torch.zeros_like(warmStart))
        err = 0.0
        it = 0
        for it in range(maxIters):
            a_p = accel(p)
            aP = applyA(p, a_p)
            p = p + invAlpha * (source - aP)
            if mode == 'density':
                if project:
                    p = p - torch.where(fluid, p[fluid].mean().expand_as(p),
                                        torch.zeros_like(p))
                else:
                    p = p.clamp(min=0.0)
            bad = alphaBad | (~torch.isfinite(p)) | (p.abs() > 1e25) | ~fluid
            p = torch.where(bad, torch.zeros_like(p), p)
            if fluid.any():
                residual = aP - source
                err = float(torch.clamp(residual, min=RESIDUAL_FLOOR)[fluid].mean())
            if it + 1 >= minIters and err <= tol:
                break
        if VERBOSE_SOLVE:
            print(f'[{mode}] {it + 1} iters, err {err:.3g}, p[{float(p.min()):+.3g}, {float(p.max()):+.3g}]')
        a_p = accel(p)
    return a_p, p, it + 1, err


def omniIncompressible_step(system: Any, dt: float, config: Any,
                            schemeConfig: Any, verbose: bool = False):
    st = system.state
    fluid = st.kinds == 0
    fcol = fluid.unsqueeze(-1)
    rho0 = schemeConfig.fluid.restDensity

    solver = schemeConfig.solverConfig
    divCfg = solver.divergenceFreeSolver   # omniSPH dfsph.divergenceEta
    denCfg = solver.pressureSolver          # omniSPH dfsph.densityEta
    # Boundary particles enter both solves as static fluid at rho0 with Akinci
    # rest-density volumes -- the particle-boundary analogue of omniSPH's
    # triangle boundaryFunc (and what iisph / dfsphReference use).
    solver.akinciBoundaryVolume = True

    # --- 1. neighbourhood + summation density (boundary included) ----------
    adjacency = _rebuildAdjacency(st, system, config)
    st.densities = computeDensities(st, config, schemeConfig, adjacency)

    if st.pressures is None:
        st.pressures = torch.zeros_like(st.densities)
    pPrior = st.pressures.clone()          # omniSPH fluidPriorPressure

    # --- 2. external forces -> a  (omniSPH externalForces: gravity only) ---
    enforceDirichlet(system, system.t, dt, config, schemeConfig)
    accel = computeGravity(st, config, schemeConfig, adjacency)
    forcing = computeForcing(system, dt, system.t, config, schemeConfig)
    accel = accel + forcing / st.masses.view(-1, 1)
    accel = torch.where(fcol, accel, torch.zeros_like(accel))

    # --- 3. divergenceSolve() -- exactly 3 iterations, zero warm start ----
    vEnter = st.velocities + dt * accel
    a_p_div, _, nDiv, errDiv = _solve(
        st, config, schemeConfig, adjacency, fluid=fluid, rho0=rho0,
        vEnter=vEnter, warmStart=torch.zeros_like(st.densities), dt=dt,
        mode='divergence', minIters=DIVERGENCE_ITERATIONS,
        maxIters=DIVERGENCE_ITERATIONS, tol=divCfg.tolerance)
    accel = accel + a_p_div

    # --- 4. densitySolve() -- min 3 / max 256, warm start 0.5 * p_prior ---
    vEnter = st.velocities + dt * accel
    a_p_rho, pRho, nRho, errRho = _solve(
        st, config, schemeConfig, adjacency, fluid=fluid, rho0=rho0,
        vEnter=vEnter, warmStart=0.5 * pPrior, dt=dt, mode='density',
        minIters=DENSITY_MIN_ITERATIONS, maxIters=DENSITY_MAX_ITERATIONS,
        tol=denCfg.tolerance)
    accel = accel + a_p_rho

    # --- 5. XSPH velocity filter (omniSPH XSPH + BXSPH, post-solve) --------
    # omniSPH filters the start-of-step velocity (the solves only touched
    # `accel`), so this and the pressure impulse `dt*accel` add independently.
    # if XSPH_FLUID != 0.0 or XSPH_BOUNDARY != 0.0:
        # st.velocities = st.velocities + _xsphFilter(st, config, adjacency, fluid)

    # --- 6. integrate (omniSPH Integrate: single semi-implicit Euler) -----
    st.velocities = st.velocities + dt * torch.where(
        fcol, accel, torch.zeros_like(accel))
    # if DAMPING != 0.0:
        # st.velocities = st.velocities * (1.0 - DAMPING)
    st.positions = st.positions + dt * torch.where(
        fcol, st.velocities, torch.zeros_like(st.velocities))

    # omniSPH: fluidPriorPressure = fluidPressure1 (the density solve's field).
    st.pressures = pRho

    if verbose:
        vmax = float(st.velocities[fluid].norm(dim=-1).max()) if fluid.any() else 0.0
        pf = pRho[fluid]
        print(f'[omniIncompressible] t={system.t + dt:.4g}  '
              f'DIV {nDiv:2d} it err {errDiv:.3g}   '
              f'RHO {nRho:3d} it err {errRho:.3g}   |v|max {vmax:.4g}   '
              f'p[{float(pf.min()):+.3g}, {float(pf.max()):+.3g}]')

    zerosV = torch.zeros_like(st.velocities)
    update = IncompressibleSystemUpdate(
        dxdt=zerosV.clone(), dvdt=zerosV.clone(),
        drhodt=torch.zeros_like(st.densities),
        passive=torch.zeros_like(st.densities, dtype=torch.bool))
    enforceUpdates(update, system, dt, system.t, config, schemeConfig)
    return update, adjacency, st, ([], [errDiv, errRho])
