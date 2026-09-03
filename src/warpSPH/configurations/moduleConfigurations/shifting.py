"""`ShiftProperties`, `ShiftingScheme`, `ShiftingProjectionScheme`: delta-SPH
particle-shifting settings, embedded as `.shiftProperties` on
`WeaklyCompressibleSPHConfig`/`IncompressibleSPHConfig` and read by
`modules/shifting/wrapper.py`. Note `buildDefaultShiftProperties()` overrides
several of the dataclass's own field defaults (`computeMach` False->True,
`projectionScheme` `dot`->`mat`) -- both are live values used depending on
whether a caller constructs `ShiftProperties()` directly or via the builder.

The `implicit*` fields configure `modules/shifting/implicitShifting.py`'s
matrix-free Krylov (BiCGStab/GMRES) solve, used when
`scheme == ShiftingScheme.implicit` (or the improved `ShiftingScheme.dynamic`,
which is the same path with the `implicitFallback` chain enabled by default);
they are ignored for `ShiftingScheme.deltaSPH`. The `implicitFallback` field
is the opt-in switch for the inner-solve fallback chain (default `none` = the
historical single-solver behavior): set it to `krylov` or
`krylov_richardson` on `ShiftingScheme.implicit`, or simply use
`ShiftingScheme.dynamic`, to make a bailed-out primary solver retry the other
Krylov solver (and optionally a bounded Richardson polish) instead of being
used as-is. `implicitPreconditioner` selects the inner-solve preconditioner
(default `scalar` = the historical Jacobi diagonal; `block` inverts the full
`dim x dim` diagonal blocks and is the general form though a wash for the
current operators; `off` disables it), and `implicitNullSpaceLift` is an
opt-in Tikhonov lift of the operator's near-null-space eigenvalues (default
`0.0` = off; aimed at the indefinite `exactHessian` operator).
"""

__all__ = ['ShiftingScheme', 'ShiftingProjectionScheme', 'ShiftingImplicitInitializer', 'ShiftingImplicitOperator', 'ShiftingImplicitSolver', 'ShiftingImplicitFallback', 'ShiftingImplicitPreconditioner', 'ShiftProperties', 'buildDefaultShiftProperties']

from ...enumTypes import *
from typing import Optional, Union, List
from dataclasses import dataclass, field
import torch
from enum import Enum

class ShiftingScheme(Enum):
    none = 0
    deltaSPH = 1
    implicit = 2
    dynamic = 3


class ShiftingProjectionScheme(Enum):
    """How `modules/shifting/wrapper.solveShifting` treats the raw shift near
    the free surface.

    - `zero`: hard-zero the shift for surface / near-surface / low-`lMin`
      particles ("don't shift the free surface").
    - `dot`: remove the shift's normal component and scale the tangential
      remainder by `surfaceScaling` for the dilated surface set; then
      hard-zero `lMin < 0.4`.
    - `mat`: `(I - n n^T)` projection scaled by `lMin**2` for the surface
      set, then hard-zero it anyway (the projection line is currently dead).
    - `surfaceNormal`: the actual Sun et al. 2019 (`literature/sun2019`)
      Eq. (20)-(21) algorithm -- a surface particle whose shift points *into*
      the surface (`n . dx >= 0`, `n` outward) is cut to tangential and
      gated by the `kappa` curvature test (`surfaceCurvatureAngle`); one
      whose shift points *away* from the surface keeps the full, unconstrained
      shift; `lMin < surfaceLambdaThreshold` in the surface set is zeroed.
      The away-branch is the anti-clustering mechanism the other three lack.
    """
    zero = 0
    dot = 1
    mat = 2
    surfaceNormal = 3


class ShiftingImplicitInitializer(Enum):
    zero = 0
    deltaPlus = 1
    deltaMinus = 2


class ShiftingImplicitOperator(Enum):
    """Which matrix `implicitShifting.computeImplicitShift` assembles and
    solves `A @ dx = -grad(C)` against, `C_i = sum_j omega_j W_ij`. See
    `modules/shifting/implicitShifting.py`'s module docstring for the full
    derivation and the empirical evidence behind the default choice.

    - `legacyPairwise` (default): ported byte-for-byte from diffSPH's
      original `getShiftingMatrices`/`bicgstab_shifting` (self-pair included
      with its raw value, off-diagonal block *not* negated). Provably *not*
      the true Hessian of `C` (222% relative Frobenius error against a
      finite-difference `Hess(C)` on a 36-particle test case, vs. 1.7e-4 for
      `exactHessian`), but empirically far more globally convergent from
      random/far-from-equilibrium starts -- confirmed by A/B testing both
      operators, sign-matched, through this codebase's own solver/clamp/
      relaxation pipeline: `legacyPairwise` converges cleanly and
      monotonically from fully-random initial positions where `exactHessian`
      stalls or oscillates on the same seeds.
    - `exactHessian`: the exact Newton Hessian of `C` (self-pair dropped --
      an exact translation-invariance identity, see the module docstring --
      diagonal block `sum_{j!=i} H_ij`, off-diagonal `-H_ij`). Mathematically
      correct and locally quadratically convergent once near the solution,
      but (per Newton's method's well-known limits on non-convex objectives)
      not reliably stable far from it even when damped
      (`implicitRelaxation`) and step-clamped (`ShiftProperties.threshold`).
      Kept as an explicit opt-in for comparison against the default, and
      because `implicitShiftingAutomatic.computeImplicitShiftAutomatic`'s
      autodiff-sourced Hessian only ever represents this formulation (there
      is no automatic-differentiation counterpart to `legacyPairwise`, since
      it is not an actual Hessian of anything).
    """
    legacyPairwise = 0
    exactHessian = 1


class ShiftingImplicitSolver(Enum):
    """Which matrix-free Krylov solver the implicit shifting Newton step
    (`A @ dx = grad(C)`, `modules/shifting/implicitShifting.py` and its
    `implicitShiftingAutomatic.py` twin) is driven with. Both share the same
    `matvec`-closure + diagonal-`precond` interface and status-code
    convention (`bicgstab.bicgstabSolve` / `gmres.gmresSolve`), so they are
    drop-in for each other.

    - `bicgstab` (default): the diffSPH-ported BiCGStab -- 2 matvecs/iterate,
      with rho/rv/omega breakdown bailouts. Kept as the default for
      continuity with diffSPH's own solve (the port is checked bit-identical
      to it on the same linear system, see
      `docs/regression/implicit_shifting_operator_choice.md`).
    - `gmres`: restarted GMRES (length `implicitRestart`) -- 1 matvec/
      iterate, no breakdown modes, monotone residual estimate within a cycle;
      the robust choice for the `exactHessian` operator (symmetric,
      indefinite, exact translation null space) and for nonsymmetric
      `omega_j`-weighted systems, at the cost of an `m x n` Krylov basis.
    """
    bicgstab = 0
    gmres = 1


class ShiftingImplicitFallback(Enum):
    """Fallback chain run when the primary Krylov solver (`implicitSolver`)
    bails out (status `< 0`) in the implicit shifting Newton solve. Opt-in:
    the default `none` runs exactly one solver and uses its result
    unconditionally -- the historical behavior, which a bailed-out `xk` was
    used exactly as if it had converged. See
    `modules/shifting/solverDriver.py` for the chain semantics and
    `modules/shifting/richardson.py` for why Richardson is a last resort.

    - `none` (default): no fallback; a bailed-out iterate is used as-is.
    - `krylov`: retry with the other Krylov solver (BiCGStab<->GMRES) from a
      clean start and keep the better iterate by stamped residual. The
      high-value fallback -- the two solvers fail on different regimes (e.g.
      GMRES converges the near-breakdown cases BiCGStab bails on).
    - `krylov_richardson`: `krylov`, plus a bounded Richardson polish from
      the best iterate. Richardson is deliberately last (see richardson.py).

    `ShiftingScheme.dynamic` presets this to at least `krylov` (an explicit
    `krylov_richardson` is respected); `ShiftingScheme.implicit` leaves it at
    whatever the config says, defaulting to `none`.
    """
    none = 0
    krylov = 1
    krylov_richardson = 2


class ShiftingImplicitPreconditioner(Enum):
    """Preconditioner applied inside the implicit-shifting Krylov solve
    (`modules/shifting/bicgstab.py`/`gmres.py`, which each accept either a
    flat diagonal vector -- applied elementwise -- or a callable `M^-1`),
    built from the assembled operator's diagonal blocks in
    `modules/shifting/preconditioner.py`.

    - `off`: no preconditioner (`precond=None`).
    - `scalar` (default, historical behavior): scalar Jacobi, `M = diag(A)` --
      the flat `[n*dim]` vector of `1/diag`. Byte-identical to the pre-enum
      production path (the old `implicitUsePreconditioner=True`).
    - `block`: block Jacobi, `M = block-diag(diagBlock)`, each block the full
      `dim x dim` diagonal block, applied via a batched inverse (a callable
      `M^-1`). The general form for a block-structured operator. On the
      *current* implicit-shifting operators it is a wash, not a win (see
      `docs/regression/implicit_shifting_operator_choice.md`): for
      `legacyPairwise` the diagonal blocks are isotropic (`c*I`, the kernel
      Hessian at the self-point of a radial kernel), so it is bit-identical to
      `scalar`; for `exactHessian` it is slightly worse. For `dim == 1` it
      reduces to `scalar`.
    """
    off = 0
    scalar = 1
    block = 2


@dataclass
class ShiftProperties:
    iterations: int = field(default=1, metadata={"description": "Number of iterations for shifting"})
    CFL: float = field(default=0.3, metadata={"description": "CFL number for the delta-SPH shift"})
    computeMach: bool = field(default=False, metadata={"description": "Whether to compute Mach number for the delta-SPH shift"})
    maxC: float = field(default=0.3, metadata={"description": "Maximum sound speed for the delta-SPH shift"})
    active: bool = field(default=True, metadata={"description": "Whether to apply the shifting"})

    scheme: ShiftingScheme = field(default=ShiftingScheme.deltaSPH, metadata={"description": "Shifting scheme to use"})
    projectionScheme: ShiftingProjectionScheme = field(default=ShiftingProjectionScheme.dot, metadata={"description": "Projection scheme to use for shifting"})

    summationDensity: bool = field(default=False, metadata={"description": "Whether to use summation density"})
    surfaceScaling: float = field(default=0.1, metadata={"description": "Scaling factor for the surface detection"})
    threshold: float = field(default=0.5, metadata={"description": "Threshold for shifting magnitude"})

    surfaceLambdaThreshold: float = field(default=0.4, metadata={"description": "lMin (min renormalisation-matrix eigenvalue) below which a surface-set particle's shift is zeroed. Was a hardcoded 0.4 in wrapper.py's dot/mat/zero paths; exposed here and used by the surfaceNormal projection scheme. Sun et al. 2019 Eq. (20) uses 0.55 for their lambda normalisation and C2 Wendland h=2dx -- calibrate per kernel."})
    surfaceCurvatureAngle: float = field(default=15.0, metadata={"description": "Curvature gate for ShiftingProjectionScheme.surfaceNormal (Sun et al. 2019 Eq. 21): a surface particle is zeroed when any neighbour's surface normal deviates from its own by more than this angle (degrees), i.e. the local radius of curvature is below the kernel radius. 0.0 disables the gate. 15 deg is the paper's value for C2 Wendland with h=2dx."})
    maxShiftVelocityFraction: float = field(default=0.5, metadata={"description": "Sun et al. 2019 Eq. (14) robustness limiter: cap the per-step shift magnitude at this fraction of Umax*dt (Umax = the max finite particle speed, the paper's 'maximum expected velocity'). The paper uses 1/2. This is a magnitude (L2) cap and the physically-scaled counterpart of the per-component `threshold` clamp (0.5*dx), which stays as a coarse backstop. 0.0 disables it. Without it the delta+ shift has no bound tied to the flow, and a locally exploding grad(C) -- e.g. an arm beading under tensile instability -- feeds an oversized shift straight into correctdrhodt."})

    projectQuantities: bool = field(default=False, metadata={"description": "Whether to project quantities after shifting"})

    correctdrhodt: bool = field(default=False, metadata={"description": "Whether to correct drhodt after shifting"})
    correctdvdt: bool = field(default=False, metadata={"description": "Whether to correct dvdt after shifting"})

    reuseNormals: bool = field(default=True, metadata={"description": "Whether to reuse normals from previous iteration for surface detection"})

    implicitTolerance: float = field(default=0.0, metadata={"description": "Absolute residual floor for the implicit shifting Krylov solve; 0.0 = relative tolerance only (this field was previously accepted by bicgstabSolve but never enforced, so 0.0 is the historical effective behavior)"})
    implicitRelativeTolerance: float = field(default=1e-4, metadata={"description": "Relative residual tolerance for the implicit shifting Krylov (BiCGStab/GMRES) solve"})
    implicitMaxSolverIter: int = field(default=64, metadata={"description": "Maximum Krylov (BiCGStab/GMRES) iterations for the implicit shifting solve"})
    implicitInitializer: ShiftingImplicitInitializer = field(default=ShiftingImplicitInitializer.zero, metadata={"description": "Initial guess for the implicit shifting solve"})
    implicitOperator: ShiftingImplicitOperator = field(default=ShiftingImplicitOperator.legacyPairwise, metadata={"description": "Which matrix the implicit shifting Newton solve assembles: legacyPairwise (default, empirically more globally convergent) or exactHessian (mathematically exact, comparison/opt-in) -- see ShiftingImplicitOperator's docstring"})
    implicitSolver: ShiftingImplicitSolver = field(default=ShiftingImplicitSolver.bicgstab, metadata={"description": "Matrix-free Krylov solver for the implicit shifting solve: bicgstab (default, diffSPH port) or gmres (restarted, no breakdown modes) -- see ShiftingImplicitSolver's docstring"})
    implicitFallback: ShiftingImplicitFallback = field(default=ShiftingImplicitFallback.none, metadata={"description": "Fallback chain for the implicit shifting solve when the primary Krylov solver bails out (status < 0): none (default, historical behavior) / krylov (retry the other Krylov solver) / krylov_richardson (krylov + bounded Richardson polish) -- see ShiftingImplicitFallback's docstring. ShiftingScheme.dynamic presets this to at least krylov"})
    implicitRestart: int = field(default=30, metadata={"description": "Krylov restart length (GMRES(m)) for the implicit shifting solve; used only by ShiftingImplicitSolver.gmres"})
    implicitPreconditioner: ShiftingImplicitPreconditioner = field(default=ShiftingImplicitPreconditioner.scalar, metadata={"description": "Preconditioner for the implicit shifting Krylov solve: scalar (default, historical scalar-Jacobi diagonal) / block (invert the full dim x dim diagonal blocks; the general block-structured form, a wash for the current operators -- see ShiftingImplicitPreconditioner's docstring) / off (none). Replaces the old boolean implicitUsePreconditioner (True==scalar, False==off)"})
    implicitNullSpaceLift: float = field(default=0.0, metadata={"description": "Tikhonov lift added to the implicit shifting operator's diagonal blocks (A -> A + lift*I); 0.0 (default) = off. Lifting the exactHessian operator's near-zero (translation-null-space) eigenvalues improves its conditioning for the Krylov solve at an O(lift) solution bias, so it is an opt-in aimed at that operator only -- the default legacyPairwise operator is well-conditioned and needs it not"})
    implicitSolverThreshold: Optional[float] = field(default=None, metadata={"description": "Per-particle shift-magnitude divergence threshold for the implicit shifting solve; defaults to dx/2 when None"})
    implicitRelaxation: float = field(default=0.1, metadata={"description": "Damping factor applied to each implicit-shifting Newton step (1.0 = full step); the assembled system is a graph-Laplacian-style operator with an exact translation null space, so a full undamped step is only reliably stable extremely close to the solution. 0.1 was swept against a jittered-lattice convergence test (repeatable across runs; 0.15 was already occasionally unstable), matching this codebase's own IISPH Jacobi relaxation precedent"})

def buildDefaultShiftProperties() -> ShiftProperties:
    return ShiftProperties(
        iterations=1,
        CFL=0.3,
        computeMach=True,
        maxC=0.3,
        active=True,
        scheme=ShiftingScheme.deltaSPH,
        # The real Sun et al. 2019 Eq. (20)-(21) free-surface treatment. Was
        # `mat` (which hard-zeroes the surface set); `surfaceNormal` keeps the
        # surface regularised without the volume blow-up. See
        # WCSPH_SHIFTING_PLAN.md: strictly better on the rotating square patch
        # (nx 64/96, t up to 1), on par on `sloshingTank` (clears the t~2.6 s
        # NaN), no `test_physics.py` regression.
        projectionScheme=ShiftingProjectionScheme.surfaceNormal,
        summationDensity=False,
        surfaceScaling=0.1,
        threshold=0.5,
        projectQuantities=False
    )
