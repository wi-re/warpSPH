"""Implicit particle shifting (IPS): instead of `delta.computeDeltaShift`'s
explicit, per-step anti-clustering nudge, this solves directly for the
position shift that (locally) equalizes the SPH concentration field
`C_i = sum_j omega_j * W_ij`, `omega_j = m_j/rho_j` (or `m_j/rho0`), via one
Newton-style step on `grad C = 0`: `A @ dx = -grad(C)`, solved with a
matrix-free, Jacobi-preconditioned Krylov solver over the neighbor graph --
BiCGStab (`bicgstab.bicgstabSolve`) by default, restartable GMRES
(`gmres.gmresSolve`) via `ShiftingImplicitSolver.gmres`. Both calls run
through `solverDriver.solveImplicitSystem`, which applies the *opt-in*
fallback chain (`ShiftProperties.implicitFallback`, default `none` = the
historical single-solver behavior) when the primary solver bails out
(status `< 0`). `computeDynamicImplicitShift` (dispatched by
`ShiftingScheme.dynamic`) is this same path with the `krylov` fallback
enabled by default -- the improved inner-solve policy, distinct from the
byte-identical legacy `computeImplicitShift`/`ShiftingScheme.implicit`.
Ported from `diffSPH.modules.shifting.implicitShifting`; the per-pair kernel
terms come from `wp_implicitShifting.computeShiftingPairTerms`.

`A` is one of two matrices, selected by `ShiftingImplicitOperator` (see that
enum's own docstring for the field-level summary; this is the full
derivation and evidence behind it):

- `exactHessian`: the true Newton Hessian of `C`. Assembling it from the
  pairwise kernel Hessian `H_ij` needs care, and differs from a literal
  transcription of diffSPH's block layout in two ways verified against a
  finite-difference Hessian of `C` and a dense exact solve:

  - Self-pairs (`i == j`, zero separation) are dropped before assembly -- **and
    both bullets below are the same fact, not two unrelated pitfalls, per a
    correction made during `warpSPHCore/warpier_forward_mode_plan.md` Phase 4
    step 3.** `C_i = sum_j omega_j W(x_i - x_j, h_ij)` depends on `x_i` through
    every term, but the `j = i` term's *own* `x_j` argument is also `x_i` --
    writing that term as a function of the one shared variable,
    `f(x) = W(x - x, h) = W(0, h)`, makes it visibly constant (a particle's
    distance to itself is identically zero everywhere in configuration space,
    not just at the current position), so `f'(x) = f''(x) = 0` for *any*
    kernel. Expanding via the chain rule on `g(a, b) = W(a - b, h)` (the two
    occurrences of `x_i` held temporarily independent as `a`/`b`, then set
    equal) shows *why*: translation invariance forces `dg/db = -dg/da` and
    `d^2g/(da db) = -d^2g/da^2 = -d^2g/db^2`, so `f''(x) = H(0,h) - 2H(0,h) +
    H(0,h) = 0` identically, regardless of what `H(0,h)` numerically equals.
    **An earlier version of this docstring claimed instead that this was a
    numerical-safety drop -- that `sphKernelHessian`'s near-origin
    regularization branch "produced a large, effectively arbitrary value
    (floating-point noise divided by a near-zero epsilon)" at `r=0`. That
    claim was wrong**: `sphKernelHessian` returns a well-defined, finite,
    physically meaningful value there (the kernel's own curvature at its peak
    -- `warpSPHCore/wp_kernels.ipynb` checks this directly for `Wendland2` and
    finds a smooth `-15.0`, continuous with its `-14.88` neighbors either
    side). The value was never the problem; the identity above is why it still
    has to be excluded from `Hess(C)`'s diagonal regardless of how well-behaved
    it is -- it was never really part of `d(grad_i C)/dx_i` to begin with.
  - The *same* identity is why `Hess(C)`'s row-`i` block is *not* `sum_j
    omega_j H_ij` placed at column `j` the way `grad C`'s row-`i` term is (that
    would make `Hess(C)` nonsymmetric): the chain-rule expansion above gives
    `dg/da = -dg/db` and `d^2g/(da db) = -d^2g/da^2`, i.e. the *off-diagonal*
    mixed partial always carries a sign flip relative to the diagonal one.
    Differentiating `grad_i C = sum_j omega_j gradW(x_i - x_j)` w.r.t. `x_i`
    and w.r.t. a specific neighbor `x_k` separately gives `d(grad_i C)/dx_i =
    sum_{j != i} omega_j H_ij` (the diagonal block, self excluded per the
    identity above) and `d(grad_i C)/dx_k = -omega_k H_ik` (off-diagonal,
    negated).

  This assembly was verified to match a finite-difference `Hess(C)` to
  1.7e-4 relative Frobenius error on a 36-particle test case (see
  `project_implicit_shifter_instability_root_cause` memory / the commit that
  introduced this docstring section). It is mathematically exact -- but, per
  Newton's method's well-known limits on non-convex objectives, *only
  locally* convergent: **an A/B test driving both operators through this
  module's own solver/clamp/relaxation pipeline (`wrapper.solveShifting`,
  identical seeds, fully-random initial positions) showed `exactHessian`
  stalling/oscillating over 40+ outer iterations where `legacyPairwise`
  converges cleanly and monotonically** -- confirmed not to be a clamp,
  relaxation-magnitude, RHS, or solver-implementation issue (all of those
  were independently ruled out first: `bicgstabSolve` was checked
  bit-identical to diffSPH's own `bicgstab_shifting` on the same system, and
  `grad(C)` matches diffSPH's real pipeline to 5+ significant figures on the
  same random positions). The assembled operator is a graph-Laplacian-style
  matrix -- symmetric, with an exact null space along uniform translation
  (`diag_i = sum_{j!=i} H_ij` cancels the off-diagonal row exactly for a
  constant shift) -- which is why it's solved with BiCGStab rather than a
  direct solve, and why `implicitRelaxation` damps each step (matching this
  codebase's own IISPH Jacobi relaxation precedent,
  `modules/incompressible/incompressible.py`) even though that damping alone
  does not make it globally convergent.

- `legacyPairwise` (default): ported byte-for-byte from diffSPH's original
  `getShiftingMatrices`/`bicgstab_shifting` block layout -- the one the
  `exactHessian` bullets above explain is analytically *not* `Hess(C)`:
  self-pairs are kept (with their raw, uncancelled `H_ii` value) and the
  off-diagonal block is *not* sign-flipped. Confirmed by the same
  finite-difference check to be a poor approximation of `Hess(C)` (222%
  relative Frobenius error), i.e. this is not "the exact Hessian written a
  different way" -- it is a different, more diagonally-dominant operator,
  closer in spirit to a graph-Laplacian smoother than a Newton step. That is
  almost certainly *why* it is the empirically robust one: unlike a Newton
  step, a diagonally-dominant relaxation operator degrades gracefully far
  from the fixed point instead of extrapolating a locally-valid quadratic
  model into a regime where it no longer applies.

  **This matches the source paper, not a drift away from it.** Rastelli,
  Vacondio, Marongiu, Fourtakas & Rogers, "Implicit iterative particle
  shifting for meshless numerical schemes using kernel basis functions",
  CMAME 393 (2022) 114716 ("IIPS") derives *this* operator, not
  `exactHessian`. Its Eq. (17)-(18) (1D) and Eq. (30)-(37) (2D) resolve
  `d(grad C)_i/dx_j` for every `j` with one rule -- "the only term ...
  non-null is the one in which j = k" -- applied uniformly including at
  `j = i`. That's the `j != i` case's reasoning (only the explicit
  neighbor-slot term survives) used where the `j = i` case actually needs
  the *other* slot: `x_i` also appears, implicitly, in every other term of
  the sum through the shared `x_i - x_k` argument, and differentiating that
  shared slot instead is what produces the sign-flipped, self-excluded,
  neighbor-summed diagonal this module's `exactHessian` bullet builds. The
  paper's own assembled system, Eq. (21)/(42), therefore has a self-included,
  uniform-sign diagonal (`H(0,h)`, literally `d^2W_ii/dx_i^2 * omega_i` in
  their notation) -- `legacyPairwise`'s layout, entry for entry, not
  `exactHessian`'s -- despite the surrounding text calling the procedure
  "a Newton-Raphson algorithm". See
  `docs/regression/implicit_shifting_operator_choice.md`'s "Comparison
  against the source paper" section for the full derivation and why the
  paper's own reported robustness (Figs. 5-9: convergence in 3-5
  Newton-Raphson iterations even from `sigma/Delta = 0.25` initial disorder)
  corroborates `legacyPairwise`'s bounded-diagonal conditioning specifically,
  not `exactHessian`'s -- the paper's equations never assemble the latter.

  Sign note: diffSPH's own `computeShifting` also computes `update = -xk`,
  but its outer `solveShifting` applies it as `positions -= update`
  (`diffSPH/v2/modules/shifting.py:999`) -- the *opposite* of this
  codebase's `wrapper.solveShifting`, which does `positions += update`. To
  reproduce diffSPH's true end-to-end behavior under this codebase's `+=`
  convention, `legacyPairwise`'s matvec is built already negated relative to
  a literal transcription of diffSPH's `multiplySparseShifting` (see
  `_buildDiagBlock` below) -- verified empirically: a first attempt to A/B
  the two operators that skipped this and kept diffSPH's raw sign produced
  the *opposite* conclusion (`exactHessian` looking better), which is what
  exposed the sign mismatch in the first place.

Free-surface/boundary handling: rows for `currentState.kinds != 0` particles
get a zero RHS/initial-guess (so the solver doesn't chase a target for
particles that get clamped to zero shift anyway by
`wrapper.solveShifting`'s post-hoc `update[kinds != 0] = 0`). diffSPH also
zeroed rows of the per-pair Hessian by `kinds`, but indexed it with a
particle-shaped mask against a pair-shaped tensor -- inconsistent shapes
that would raise unless `numPairs == numParticles`; that step is dropped
here rather than ported as-is.
"""

from typing import Any, Optional, Tuple
import torch
from warpSPHCore.profiling import record_function
from warpSPHCore import *

from warpSPH.math import scatter_sum
from warpSPH.configurations.simulationConfig import SimulationConfig
from ...configurations.moduleConfigurations.shifting import ShiftingImplicitInitializer, ShiftingImplicitOperator, ShiftingImplicitSolver, ShiftingImplicitFallback, ShiftingImplicitPreconditioner

from .wp_implicitShifting import computeShiftingPairTerms
from .preconditioner import buildScalarJacobiPrecond, buildBlockJacobiPrecond
from .solverDriver import solveImplicitSystem
from .delta import computeDeltaShift

__all__ = ['computeImplicitShift', 'computeDynamicImplicitShift']


def _multiplyLaplacianBlock(
    diagBlock: torch.Tensor, H: torch.Tensor, x: torch.Tensor,
    i: torch.Tensor, j: torch.Tensor, numParticles: int, dim: int,
) -> torch.Tensor:
    """`out_i = diagBlock_i @ x_i - sum_{j: neighbor of i, j != i} H_ij @ x_j`.
    Shared by both `ShiftingImplicitOperator` modes -- only how `diagBlock`
    and `H`/`i`/`j` (always self-pair-excluded here) are built upstream
    differs; see `_buildDiagBlock`."""
    xr = x.view(numParticles, dim)
    out = torch.einsum('nab,nb->na', diagBlock, xr)
    for a in range(dim):
        for b in range(dim):
            out[:, a] -= scatter_sum(H[:, a, b] * xr[j, b], i, dim=0, dim_size=numParticles)
    return out.flatten()


def _buildDiagBlock(
    operator: ShiftingImplicitOperator,
    i_all: torch.Tensor, j_all: torch.Tensor, Hw_all: torch.Tensor,
    numParticles: int, dim: int, device: torch.device, dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Builds `(diagBlock, i, j, H)` for `_multiplyLaplacianBlock`, `i`/`j`/`H`
    always with self-pairs excluded (self-pairs contribute zero to `grad(C)`
    for any operator -- see module docstring -- so they only ever matter
    through `diagBlock`). `i_all`/`j_all`/`Hw_all` are the *unmasked* pairs
    (self included), `Hw_all = H * omega[j_all]`.

    - `exactHessian`: `diagBlock_i = sum_{j != i} Hw_ij` (the true Hessian
      diagonal).
    - `legacyPairwise`: `diagBlock_i = -Hw_ii` (the raw self-pair value,
      negated -- see module docstring's sign note; this negation is what
      makes `_multiplyLaplacianBlock`'s `diagBlock @ x - sum H_ij @ x_j`
      compute `-(Hw_ii @ x_i + sum_{j!=i} Hw_ij @ x_j)`, diffSPH's own
      `multiplySparseShifting` operator with the sign this codebase's
      `positions += update` convention needs).
    """
    selfMask = i_all == j_all
    i, j, Hw = i_all[~selfMask], j_all[~selfMask], Hw_all[~selfMask]

    if operator == ShiftingImplicitOperator.exactHessian:
        diagBlock = scatter_sum(Hw, i, dim=0, dim_size=numParticles)
    else:
        diagBlock = torch.zeros(numParticles, dim, dim, device=device, dtype=dtype)
        diagBlock[i_all[selfMask]] = -Hw_all[selfMask]

    return diagBlock, i, j, Hw


def _buildSystem(
    currentState: Any,
    config: SimulationConfig,
    schemeConfig: Any,
    domain: Any,
    adjacency: AdjacencyList,
    i_all: torch.Tensor,
    j_all: torch.Tensor,
    J: torch.Tensor,
    H: torch.Tensor,
    rho0: float,
    dim: int,
    numParticles: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Builds the RHS `B = grad(C)` and initial guess `x0` (mode-independent:
    self-pairs contribute zero to `grad(C)` regardless of `i_all`/`j_all`
    including them, see module docstring, so `B` needs no masking) and the
    per-pair `Hw = H * omega[j]` `_buildDiagBlock` turns into the operator."""
    device, dtype = currentState.positions.device, currentState.positions.dtype

    if schemeConfig.shiftProperties.summationDensity:
        omega = currentState.masses / currentState.densities
    else:
        omega = currentState.masses / rho0

    Jw = scatter_sum(J * omega[j_all, None], i_all, dim=0, dim_size=numParticles)
    Hw_all = H * omega[j_all, None, None]
    B = Jw.flatten().clone()

    initializer = schemeConfig.shiftProperties.implicitInitializer
    if initializer in (ShiftingImplicitInitializer.deltaPlus, ShiftingImplicitInitializer.deltaMinus):
        delta, _ = computeDeltaShift(currentState, config, schemeConfig, domain, adjacency, iters=1)
        sign = -0.5 if initializer == ShiftingImplicitInitializer.deltaPlus else 0.5
        x0 = (delta.flatten() * sign).clone()
    else:
        x0 = torch.zeros(numParticles * dim, device=device, dtype=dtype)

    if torch.any(currentState.kinds != 0):
        boundary = currentState.kinds != 0
        B.view(numParticles, dim)[boundary] = 0
        x0.view(numParticles, dim)[boundary] = 0

    return Hw_all, B, x0


def computeImplicitShift(
    currentState: Any,
    config: SimulationConfig,
    schemeConfig: Any,
    domain: Any,
    adjacency: AdjacencyList,
    iters: int = -1,
    fallback_override: Optional[ShiftingImplicitFallback] = None,
    wall: Optional[Any] = None,
):
    """Solves for the equilibrium implicit-shifting position delta in a
    single BiCGStab solve (`iters` is accepted, matching
    `delta.computeDeltaShift`'s call signature, but is not otherwise used --
    the outer per-iteration adjacency rebuild already happens in
    `wrapper.solveShifting`'s loop). Returns `(delta, adjacency)`.

    Which matrix is solved is controlled by
    `schemeConfig.shiftProperties.implicitOperator` (`ShiftingImplicitOperator`
    -- see its docstring and this module's docstring for the full tradeoff).
    """
    with record_function("[warpSPH] - (shift) - implicit"):
        numParticles = currentState.positions.shape[0]
        dim = currentState.positions.shape[1]
        device, dtype = currentState.positions.device, currentState.positions.dtype

        rho0 = schemeConfig.fluid.restDensity
        operator = schemeConfig.shiftProperties.implicitOperator

        _K, J, H = computeShiftingPairTerms(currentState, domain, config.kernel, adjacency)
        i_all, j_all = adjacency.i, adjacency.j

        Hw_all, B, x0 = _buildSystem(
            currentState, config, schemeConfig, domain, adjacency, i_all, j_all, J, H, rho0, dim, numParticles,
        )
        from ..analyticBoundary import resolveWall
        wall = resolveWall(currentState, config, schemeConfig, adjacency, wall)
        if wall is not None:
            # analytic walls: the wall continuum's share of grad C (modules/analyticBoundary/shifting.py). The wall particles are fixed, so the default legacyPairwise matrix
            # (self-pair diagonal, off-diagonal blocks of the fluid pairs) is unchanged; exactHessian's diagonal would need the wall's Hessian integral
            if operator == ShiftingImplicitOperator.exactHessian:
                raise NotImplementedError('implicit shifting with analytic walls: the exactHessian operator needs the wall Hessian integral (use legacyPairwise)')
            from ..analyticBoundary import wallConcentrationGradient
            B = B + wallConcentrationGradient(wall, currentState, schemeConfig).to(dtype).flatten()
        diagBlock, i, j, Hw = _buildDiagBlock(operator, i_all, j_all, Hw_all, numParticles, dim, device, dtype)

        # opt-in Tikhonov lift of the operator's near-null-space eigenvalues
        # (A -> A + lift*I): lifts the exactHessian operator's near-zero
        # translation-mode eigenvalues, improving the Krylov solve's
        # conditioning at an O(lift) solution bias. 0.0 (default) = off; the
        # default legacyPairwise operator is well-conditioned and needs it not.
        nullLift = schemeConfig.shiftProperties.implicitNullSpaceLift
        if nullLift > 0:
            diagBlock = diagBlock + nullLift * torch.eye(dim, device=device, dtype=dtype).unsqueeze(0)

        if torch.any(currentState.kinds != 0):
            activeMask = currentState.kinds[i] == 0
        else:
            activeMask = torch.ones_like(i, dtype=torch.bool)
        iA, jA, HwA = i[activeMask], j[activeMask], Hw[activeMask]

        precondMode = schemeConfig.shiftProperties.implicitPreconditioner
        if precondMode == ShiftingImplicitPreconditioner.off:
            precond = None
        elif precondMode == ShiftingImplicitPreconditioner.block:
            precond = buildBlockJacobiPrecond(diagBlock)
        else:
            precond = buildScalarJacobiPrecond(diagBlock)

        def matvec(x: torch.Tensor) -> torch.Tensor:
            return _multiplyLaplacianBlock(diagBlock, HwA, x, iA, jA, numParticles, dim)

        dx = config.dx.cpu().item() if isinstance(config.dx, torch.Tensor) else config.dx
        threshold = schemeConfig.shiftProperties.implicitSolverThreshold
        if threshold is None:
            threshold = dx / 2

        solverArgs = dict(
            tol=schemeConfig.shiftProperties.implicitTolerance,
            rtol=schemeConfig.shiftProperties.implicitRelativeTolerance,
            maxiter=schemeConfig.shiftProperties.implicitMaxSolverIter,
            precond=precond,
            threshold=threshold,
            dim=dim,
        )
        fallback = (fallback_override if fallback_override is not None
                    else schemeConfig.shiftProperties.implicitFallback)
        xk, solverIters, convergence = solveImplicitSystem(
            matvec, B, x0, solverArgs,
            primary_solver=schemeConfig.shiftProperties.implicitSolver,
            restart=schemeConfig.shiftProperties.implicitRestart,
            fallback=fallback)

        update = -xk.view(numParticles, dim) * schemeConfig.shiftProperties.implicitRelaxation
        return update, adjacency


def computeDynamicImplicitShift(
    currentState: Any,
    config: SimulationConfig,
    schemeConfig: Any,
    domain: Any,
    adjacency: AdjacencyList,
    iters: int = -1,
):
    """`ShiftingScheme.dynamic` entry point: `computeImplicitShift` with the
    robust fallback chain enabled by default. If the config's
    `implicitFallback` is at its default `none`, it is upgraded to `krylov`
    for this call (an explicit `krylov_richardson` is respected, and an
    explicit choice is never downgraded). This is the improved inner-solve
    policy; the legacy `ShiftingScheme.implicit` path is left byte-identical
    (its `implicitFallback` defaults to `none`). See
    `configurations.moduleConfigurations.shifting.ShiftingImplicitFallback`
    and `modules/shifting/solverDriver.py` for the chain semantics."""
    fallback = schemeConfig.shiftProperties.implicitFallback
    if fallback == ShiftingImplicitFallback.none:
        fallback = ShiftingImplicitFallback.krylov
    return computeImplicitShift(currentState, config, schemeConfig, domain, adjacency,
                                iters=iters, fallback_override=fallback)

