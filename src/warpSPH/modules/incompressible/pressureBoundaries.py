"""Band et al. 2018, *Pressure Boundaries for Implicit Incompressible SPH*
(ACM TOG 37(2):14) -- the extended PPE in which boundary samples enter the
pressure solve as their own unknowns.

Why (DFSPH_IMPROVEMENT_PLAN.md active track / Parts 41-44). The DFSPH/IISPH
family in this codebase (`omniIncompressible`, `iisph`, `dfsphReference`) puts
the wall into the diagonal `alpha` only and runs the pressure gradient at
boundary `p == 0` (Bender-Westhofen-Jeske 2023 Eq. 33). Part 44 measured the
resulting constant-density operator: it is symmetric to fp32, but
**rank-deficient at the wall corners** -- `median|alpha_fluid|` falls to 1e-5,
the relaxed Jacobi stalls at `|r|/|s| ~ 0.9`, and MINRES / CG / BiCGStab all
diverge along the near-null space. A per-iterate wall-pressure extrapolation
(`wallPressureExtrapolation`, Part 41) closes the *quiescent* column but not a
sheared wall-bounded flow, and it is an *approximation* of a boundary pressure,
not a solved one.

`band2018pb` removes the rank deficiency at its source: the near-wall rows get
their own non-trivial equation (Eq. 10) and diagonal (Eq. 20), so the system
`A p = s` over the concatenated unknown `p = [p_f ; p_b]` is no longer singular
there.

Discretization (paper Section 2.3, volume-centric, one-way *static* walls --
the case this codebase's `applyConsistentCoupling` schemes handle):

  * Rest volume        V0_i   -- fluid: the nominal particle volume m_f/rho0
                                 (the paper's h^d); boundary: gamma / sum_bb W_bbb
                                 (Eq. 12, gamma = GAMMA).
  * Actual volume       V_i   -- fluid: V0_f / sum_{all j} V0_j W_fj  (Eq. 13);
                                 boundary: V0_b / (sum_{f} V0_f W_bf + gamma + beta)
                                 (Eq. 15, beta = BETA * V0_f).
  * Pressure accel      a^p_f = -(V_f/m_f) sum_{all j} V_j (p_f + p_j) grad W_fj
                                 (Eq. 8) -- unified over fluid AND boundary
                                 neighbours, no mirroring; a^p_b := 0 (static).
  * Operator            (A p)_i = dt^2 sum_j V_j (a^p_i - a^p_j) . grad W_ij
                                 -- fluid Eq. 9 and boundary Eq. 10 at once: with
                                 a^p zero on boundary rows, boundary-boundary
                                 pairs vanish and a fluid neighbour of a boundary
                                 row contributes -dt^2 V_f a^p_f . grad W_bf,
                                 exactly Eq. 10.
  * Velocity divergence div v*_i = sum_j V_j (v*_j - v*_i) . grad W_ij, with
                                 v*_b := 0 -- Eq. 16 for fluid rows; for a static
                                 wall the boundary-boundary term of Eq. 16
                                 vanishes so this also gives Eq. 17 on boundary
                                 rows.
  * Source             s_i    = 1 - V0_i/V_i + dt div v*_i   (RHS of Eqs. 9/10).
  * Diagonal           a_ii   -- fluid Eq. 19:
        a_ff = -dt^2 [ (V_f/m_f) |sum_j V_j grad W_fj|^2
                       + V_f sum_{f nbr} (V_j^2/m_j) |grad W_fj|^2 ]
                     boundary Eq. 20 (no first term, since a^p_b = 0):
        a_bb = -dt^2   V_b sum_{f nbr} (V_j^2/m_j) |grad W_bj|^2 .
    `computeAlpha(apparentVolumes=V, includeBoundaryReaction=False)` returns
    exactly `-(first + second)` with the second sum over fluid neighbours only;
    on boundary rows the first term is added back to cancel it.
  * Relaxation         omega_i -- omega_f = OMEGA_FLUID, omega_b = 0.5 V0_b/V0_f
                                 (paper's per-sample factor; small boundary
                                 samples get a smaller step).
  * Update (Eq. 18)    p_i <- max(p_i + (omega_i / a_ii) (s_i - (A p)_i), 0)
                                 on fluid AND boundary rows. In a fully
                                 enclosed domain the `max(., 0)` is replaced by
                                 a zero-mean projection -- see
                                 `bandConstantModeRatio` and
                                 `band2018pb.CLOSED_DOMAIN_GAUGE`.
  * Convergence        mean over fluid+boundary rows of `(A p)_i - s_i`
                                 (a relative volume error -- Eq. 21).

Implementation: every operator is composed from the existing `warpOperation`
primitives and `computeAlpha`, using a band-volume *injection* -- a context
manager that sets `state.densities := state.masses / V` for the solve so that
`warpOperation`'s apparent-volume weight `m_j/rho_j` becomes `V_j`, restoring
it on the way out (the same trick `applyConsistentCoupling` uses). No new
`@wp.kernel`.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import torch

from warpSPHCore import (GradientScheme, OperationDirection, OperationProperties,
                         SupportScheme, WarpOperation, warpOperation)

from .wp_alpha import computeAlpha

__all__ = [
    'MIN_FLUID_CONTACT', 'MIN_DIAG_FRAC',
    'bandRestVolumes', 'bandActualVolumes', 'bandBoundaryUnknownMask',
    'bandWellPosedMask', 'bandVelocityDivergence', 'bandInjectVolumes',
    'bandPressureAccel', 'bandApplyOperator', 'bandDiagonal', 'bandRelaxation',
    'bandConstantModeRatio',
]

#: A row (fluid or boundary) is a pressure unknown only if its diagonal
#: `|dt^2 a_ii|` is at least this fraction of the fluid-row median. Rows below
#: it are kernel-deficient (a free surface, a thin gap) -- there is no
#: meaningful pressure equation there and `omega / a_ii` detonates -- and are
#: held at `p = 0` instead of regularised with a large Tikhonov shift (which
#: solves a more-compressible problem and leaves the column over-pressurised).
MIN_DIAG_FRAC = 0.05

#: A `kind == 1` sample is a pressure unknown only if the kernel-weighted rest
#: volume of its *fluid* neighbours `sum_bf V0_f W_bf` exceeds this. The paper
#: assumes a single boundary layer; this codebase samples a five-layer Akinci
#: band, and the ~3-4 layers with no fluid contact must NOT enter the PPE --
#: they have a zero Eq. 20 diagonal (no fluid neighbours) and would be
#: decoupled unknowns. `sum_bf V0_f W_bf` is dimensionless and ~O(0.3-1) for a
#: fully fluid-facing interface sample.
MIN_FLUID_CONTACT = 0.02

#: Paper constants `gamma` (Eq. 12) / `beta` (Eq. 14) are NOT used: they
#: calibrate a one-layer cubic-spline `2h` wall so that `V0_b ~ h^d`, and do
#: not transfer to this codebase's Wendland2 / `n_h = 4` five-layer band
#: (measured: they inflate `V_b` to ~1.4 x `V0_b`, injecting a spurious +0.3
#: boundary source). Instead the boundary uses the same nominal rest volume
#: and the same full-neighbour actual-volume formula as the fluid -- an
#: interface sample with a full band behind it and fluid in front then has
#: near-complete support, so `V_b ~ V0_b` and its source ~ 0 at rest.


def _bareKernelSum(state: Any, config: Any, adjacency: Any,
                   direction: OperationDirection,
                   weights: torch.Tensor | None = None) -> torch.Tensor:
    """`sum_{j in `direction`} w_j W_ij` via an `Interpolate` whose `m_j/rho_j`
    weight is divided back out. `weights=None` -> `w_j = 1` (a plain kernel
    sum); otherwise `w_j = weights[j]`. Independent of the actual density value
    (the `rho_j` cancels), so it is valid before the volume injection."""
    inv = state.densities / state.masses
    ref = inv if weights is None else weights * inv
    return warpOperation(
        state,
        OperationProperties(
            kernel=config.kernel, operation=WarpOperation.Interpolate,
            supportMode=SupportScheme.SuperSymmetric,
            operationMode=direction),
        referenceValues=ref, domain=config.domain, adjacency=adjacency)


def bandRestVolumes(state: Any, config: Any, adjacency: Any,
                    rho0: float) -> torch.Tensor:
    """`V0_i = m_i / rho0` for every row -- the nominal particle volume (the
    paper's Eq. 11 `h^d`), used for fluid and boundary alike (see the note on
    `gamma`/`beta` above)."""
    return state.masses / rho0


def bandActualVolumes(state: Any, config: Any, adjacency: Any,
                      V0: torch.Tensor, rho0: float) -> torch.Tensor:
    """`V_i = V0_i / sum_{all j} V0_j W_ij` -- Eq. 13, applied to fluid and
    boundary rows alike. Near-complete support (a fluid row in the bulk, or a
    boundary interface row with a full band behind it) gives `V_i ~ V0_i`."""
    denom = _bareKernelSum(state, config, adjacency,
                           OperationDirection.AllToAll, weights=V0)
    return V0 / denom.clamp_min(1e-12)


def bandBoundaryUnknownMask(state: Any, config: Any, adjacency: Any,
                            rho0: float) -> torch.Tensor:
    """`kind == 1` rows with enough fluid contact to be pressure unknowns --
    `sum_bf V0_f W_bf > MIN_FLUID_CONTACT`. The 3-4 band layers behind the
    interface get no fluid neighbours (zero Eq. 20 diagonal) and are excluded;
    their pressure stays 0 and is never read by a fluid row anyway."""
    boundary = state.kinds == 1
    V0f = state.masses / rho0
    kf = _bareKernelSum(state, config, adjacency,
                        OperationDirection.FluidToBoundary, weights=V0f)
    return boundary & (kf > MIN_FLUID_CONTACT)


@contextmanager
def bandInjectVolumes(state: Any, V: torch.Tensor):
    """Set `state.densities := state.masses / V` for the duration of the block
    (so every `warpOperation` apparent-volume weight `m_j/rho_j` becomes the
    band actual volume `V_j`) and restore it afterwards. Masses are untouched."""
    saved = state.densities
    try:
        state.densities = state.masses / V.clamp_min(1e-12)
        yield
    finally:
        state.densities = saved


def bandWellPosedMask(diag: torch.Tensor, fluid: torch.Tensor,
                      solveRows: torch.Tensor) -> torch.Tensor:
    """Restrict `solveRows` to rows whose diagonal is not near-null --
    `|diag_i| >= MIN_DIAG_FRAC * median(|diag_fluid|)`. `diag` is
    `bandDiagonal`'s output (`dt^2 a_ii <= 0`), evaluated inside
    `bandInjectVolumes`. Kernel-deficient free-surface / thin-gap rows fall
    below the cut and are held at `p = 0`."""
    med = float(diag[fluid].abs().median()) if bool(fluid.any()) else 0.0
    return solveRows & (diag.abs() >= MIN_DIAG_FRAC * med)


def bandVelocityDivergence(state: Any, config: Any, adjacency: Any,
                           vStar: torch.Tensor,
                           solveRows: torch.Tensor) -> torch.Tensor:
    """`div v*_i = sum_j V_j (v*_j - v*_i) . grad W_ij` with `v*` zeroed on
    non-solve rows (static walls / ghosts). Call inside `bandInjectVolumes`.
    Eq. 16 on fluid rows; = Eq. 17 on boundary rows for a static wall."""
    vs = torch.where(solveRows.unsqueeze(-1), vStar, torch.zeros_like(vStar))
    return warpOperation(
        state,
        OperationProperties(
            kernel=config.kernel, operation=WarpOperation.Divergence,
            gradientMode=GradientScheme.Difference,
            supportMode=SupportScheme.Scatter),
        queryValues=vs, domain=config.domain, adjacency=adjacency,
        consistentDivergence=False)


def bandPressureAccel(state: Any, config: Any, adjacency: Any,
                      p: torch.Tensor, V: torch.Tensor,
                      fluid: torch.Tensor) -> torch.Tensor:
    """Eq. 8: `a^p_f = -(V_f/m_f) sum_j V_j (p_f + p_j) grad W_fj`, unified over
    fluid + boundary neighbours (`p` carries the boundary unknowns). Zero on
    non-fluid rows (`a^p_b := 0`). Call inside `bandInjectVolumes`."""
    summ = warpOperation(
        state,
        OperationProperties(
            kernel=config.kernel, operation=WarpOperation.Gradient,
            gradientMode=GradientScheme.Summation,
            supportMode=SupportScheme.Scatter),
        queryValues=p, domain=config.domain, adjacency=adjacency)
    pref = -(V / state.masses).unsqueeze(-1)
    a_p = pref * summ
    return torch.where(fluid.unsqueeze(-1), a_p, torch.zeros_like(a_p))


def bandApplyOperator(state: Any, config: Any, adjacency: Any,
                      a_p: torch.Tensor, dt: float,
                      solveRows: torch.Tensor) -> torch.Tensor:
    """`(A p)_i = dt^2 sum_j V_j (a^p_i - a^p_j) . grad W_ij` -- Eqs. 9 and 10
    at once (`a^p` zero on boundary rows). `warpOperation(Divergence,
    Difference)` returns `sum_j V_j (a^p_j - a^p_i) . grad W`, hence the sign.
    Call inside `bandInjectVolumes`."""
    div = warpOperation(
        state,
        OperationProperties(
            kernel=config.kernel, operation=WarpOperation.Divergence,
            gradientMode=GradientScheme.Difference,
            supportMode=SupportScheme.Scatter),
        queryValues=a_p, domain=config.domain, adjacency=adjacency,
        consistentDivergence=False)
    Ap = -dt * dt * div
    return torch.where(solveRows, Ap, torch.zeros_like(Ap))


def bandDiagonal(state: Any, config: Any, schemeConfig: Any, adjacency: Any,
                 V: torch.Tensor, dt: float) -> torch.Tensor:
    """`dt^2 * a_ii` -- Eq. 19 (fluid) and Eq. 20 (boundary). Returns a value
    `<= 0` (the Jacobi divides the residual by it). Call inside
    `bandInjectVolumes`.

    `computeAlpha(apparentVolumes=V, includeBoundaryReaction=False)` returns
    `-( (V_i/m_i)|sum_j V_j gradW|^2 + V_i sum_{f nbr}(V_j^2/m_j)|gradW|^2 )`
    -- exactly Eq. 19. Eq. 20 is the same without the first term, so on
    boundary rows that term (`(V_b/m_b)|g_b|^2`, recomputed here from a
    `Naive` gradient of a ones field = `sum_j V_j gradW_bj`) is added back."""
    boundary = state.kinds == 1
    alpha = computeAlpha(state, config, schemeConfig, adjacency,
                         apparentVolumes=V, includeBoundaryReaction=False)
    g = warpOperation(
        state,
        OperationProperties(
            kernel=config.kernel, operation=WarpOperation.Gradient,
            gradientMode=GradientScheme.Naive,
            supportMode=SupportScheme.Scatter),
        queryValues=torch.ones_like(V), domain=config.domain,
        adjacency=adjacency)
    firstTerm = (V / state.masses) * (g * g).sum(-1)
    alpha = alpha + torch.where(boundary, firstTerm, torch.zeros_like(firstTerm))
    return dt * dt * alpha


def bandConstantModeRatio(state: Any, config: Any, adjacency: Any,
                          diag: torch.Tensor, V: torch.Tensor,
                          fluid: torch.Tensor, solveRows: torch.Tensor,
                          dt: float) -> float:
    """`rms|A.1| / rms|a_ii|` over the solve rows -- how far the *constant*
    pressure field is from the operator's null space.

    Eq. 8 is a summation gradient, so a uniform `p = c` gives
    `a^p_i = -(V_i/m_i) 2c sum_j V_j grad W_ij`, which vanishes wherever the
    kernel support is complete. In a fully enclosed domain that holds
    everywhere, so `A.1 ~ 0`: the pressure is determined only up to an additive
    constant, and since the Eq. 18 `max(., 0)` clamp can only ever push a row
    *up*, the unpinned constant ratchets away (measured on
    `randomFlowIncompressible --bounded`: `p in [1.6e3, 2.6e3]` on step 1 --
    the whole field offset, not a gradient). A free surface breaks the mode,
    because `sum_j V_j grad W_ij != 0` there.

    Measured at nx=64 step 1: `randomFlowIncompressible --bounded` (closed box)
    0.032, `hydrostaticColumn` (free surface) 1.29 -- a ~40x separation, which
    is what `band2018pb.CLOSED_DOMAIN_GAUGE = 'auto'` keys on. Costs one extra
    operator application per step (not per iteration). Call inside
    `bandInjectVolumes`.
    """
    if not bool(solveRows.any()):
        return float('inf')
    ones = torch.where(solveRows, torch.ones_like(diag), torch.zeros_like(diag))
    a1 = bandPressureAccel(state, config, adjacency, ones, V, fluid)
    A1 = bandApplyOperator(state, config, adjacency, a1, dt, solveRows)
    rmsA1 = float(A1[solveRows].pow(2).mean().sqrt())
    rmsD = float(diag[solveRows].pow(2).mean().sqrt())
    return rmsA1 / rmsD if rmsD > 0.0 else float('inf')


def bandRelaxation(state: Any, V0: torch.Tensor, rho0: float,
                   omegaFluid: float) -> torch.Tensor:
    """Per-sample relaxation `omega_i = omegaFluid * V0_i / V0_f` -- the paper's
    `omega_i = 0.5 V0_i/h^d` with the base constant taken from `omegaFluid`
    rather than pinned at the paper's 0.5.

    The scaling is *relative*: the paper's single constant multiplies fluid and
    boundary rows alike, and a boundary sample's smaller rest volume is what
    gives it the smaller step. Detuning only the fluid (this codebase runs
    `omegaFluid = 0.05`, not 0.5, at `n_h = 4`) while leaving the boundary at a
    hardcoded 0.5 breaks that: with `V0_b ~ V0_f` here the boundary rows kept
    `omega_b = 0.5` -- a 10x larger relaxation on rows whose Eq. 20 diagonal is
    itself ~10x *smaller* than the fluid's Eq. 19 one (it lacks the first term),
    i.e. a Jacobi step `omega/a_ii` about 100x the fluid's. Measured on
    `hydrostaticColumn` nx=128: the boundary pressure ran away to 8x the peak
    fluid pressure, Eq. 8 turned that into `|a_p| ~ 1e3` on the near-wall fluid,
    and the column was kicked apart (`pressureSlopeRatio` 0.02,
    `embeddedMinDensity` 0.14, 1695 spray particles). Scaling both rows by
    `omegaFluid` gives `slope` 0.996 / `embMin` 0.96 / 37 spray at nx=128 and is
    a wash at nx=32/64.
    """
    V0f = state.masses / rho0
    return omegaFluid * V0 / V0f.clamp_min(1e-12)
