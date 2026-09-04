"""Matrix-free MINRES (minimum residual method) for the incompressible
pressure solve. Same ``matvec``-closure + flat-diagonal-``precond`` interface
and same ``(x, status, convergence)`` return convention as
``bicgstab.bicgstabSolve`` / ``gmres.gmresSolve`` / ``cg.cgSolve``; the
``atol = max(atol, tol, rtol*||b||)`` floor and the status-code family match
that family.

MINRES is the Krylov method designed for a **symmetric but not necessarily
definite** operator -- exactly the IISPH pressure operator (symmetric to fp32
precision, negative-semi-definite with a constant gauge null space; see the
operator probe and the "BiCGStab deep-dive" section of
``docs/historic_plans/INCOMPRESSIBLE_SOLVER_PLAN.md``). Unlike CG it needs no positive
definiteness (no sign flip in the dispatch); unlike BiCGStab it has no shadow
system, so there is no shadow orthogonality for fp32 round-off to destroy. It
minimizes the true residual at every step and its residual estimate is
monotone non-increasing -- measured on the TGV probe state it was the best
and cleanest of all the methods (no breakdown at 1200 iters, fp32/fp64 within
~3x).

Method: Lanczos on the operator, then the normal equation on the tridiagonal
``M_k = [T_k; beta_k e_k^T]`` ((k+1) x k), whose minimizer ``y_k`` gives the
FULL iterate ``x_k = x0 + V_k y_k`` (``y_k[:k-1] != y_{k-1}``, so there is no
incremental update). The per-step normal equation is solved with the standard
MINRES Givens-LQ recursion (one 2x2 rotation per step, O(1)/step) rather than
a dense least squares: ``M_k = Q_k L_k`` with ``L_k = [L~_k; 0]`` and ``L~_k``
upper banded of band 2 -- diag ``r_j = hypot(u_j, beta_j)``, superdiag
``m_j = beta_j c_{j-1} c_j - s_j alpha_{j+1}``, second superdiag
``-beta_{j+1} s_j`` -- where ``u_k = s_{k-1} c_{k-2} beta_{k-1} + c_{k-1}
alpha_k`` (``c_0 := 1``; the new Lanczos column is pre-rotated by ALL earlier
Givens, which is where the ``c_{k-2}`` factor comes from) and ``c_k = u_k /
r_k``, ``s_k = -beta_k / r_k``. The residual estimate ``rho_k = |z_k(k+1)|``
is the exact MINRES residual in exact arithmetic. The per-step solve is a
triangular back-substitution on the growing ``L~_k`` (O(k^2)/step; an
O(k)/step incremental update of the solution vector is the planned
production optimization), verified per iterate against the dense-``lstsq``
reference in
``tests/test_incompressibleKrylov.py::test_minresGivensMatchesDenseLstsq`` on
random SPD and NSD+gauge systems (the throwaway prototype's dense-``lstsq``
per-step solve is what this replaces; its bugs are catalogued in the plan's
"BiCGStab deep-dive" section and covered by that test).

Preconditioning -- a **symmetrizing congruence**, not left preconditioning
(which would break the symmetry MINRES needs): with the flat diagonal
``precond = 1/D`` (``D = diag A``), set ``d = sqrt(|D|) = 1/sqrt(|precond|)``
elementwise (``precond=None`` => ``d = 1``) and solve
``A~ u~ = c`` with ``A~ v := d * (A (d * v))`` (symmetric when ``A`` is;
negative-semi-definite is fine for MINRES), ``c = d * b``; the original-space
solution is ``x = d * u~`` (``A~ u~ = c  =>  A (d * u~) = b``), and ``x0``
enters as ``u~0 = x0 / d``. All Lanczos/LQ bookkeeping lives in the
transformed space; the threshold check and the returned ``x`` are in the
original space. On a uniform lattice ``D`` is a scalar, so the congruence
degenerates to a scaling (harmless); on deformed states it does real work.
The per-step residual/breakdown tests compare the d-weighted (transformed)
residual against ``dmin * atol`` (``||r|| <= ||d * r|| / dmin``), while the
final convergence verification checks the original-space ``||b - A x||``
against ``atol`` itself -- comparing the d-scaled residual to an
original-space floor would read as an instant false "breakdown" under a
strong preconditioner (the whole transformed operator sits below that floor).

Memory: the Lanczos basis ``V`` (``n x min(maxiter, n)`` vectors) is kept for
the full recombination -- intrinsic to MINRES (restarting would lose the
residual-minimizing property; the process terminates by ``n`` steps in exact
arithmetic, which caps the useful basis). The per-step bookkeeping itself is
O(k) scalars (no dense matrix of the operator is ever built).

Status codes: ``>= 0`` converged at that (0-based) iterate; ``-12``
per-particle ``|x|`` threshold bailout; ``-13`` Lanczos breakdown /
stagnation (``beta_k -> 0`` with the residual still above tolerance -- the
Krylov subspace is A-invariant -- or a non-finite scalar in the recurrence);
``-14`` max-iteration budget exhausted. ``convergence`` holds the per-step
MINRES residual estimate (the exact residual in exact arithmetic, monotone
non-increasing) in the congruence-transformed space, with the final entry
always the verified true residual ``||b - A x||`` of the returned iterate in
the original space (the ``finish()`` pattern from ``bicgstabSolve``).
"""

from typing import Callable, List, Optional, Tuple, Union
import torch


def minresSolve(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    x0: Optional[torch.Tensor] = None,
    tol: float = 0.0,
    rtol: float = 1e-5,
    atol: float = 0.0,
    maxiter: Optional[int] = None,
    precond: Union[torch.Tensor, Callable[[torch.Tensor], torch.Tensor], None] = None,
    verbose: bool = False,
    threshold: Optional[float] = None,
    dim: int = 1,
) -> Tuple[torch.Tensor, int, List[torch.Tensor]]:
    """Solve ``matvec(x) == b`` matrix-free with MINRES. The operator must be
    symmetric (not necessarily definite -- see the module docstring). The flat
    ``precond`` vector ``1/D`` is used as a symmetrizing congruence; a
    callable preconditioner is rejected (it cannot be symmetrized)."""
    xk0 = x0.clone() if x0 is not None else torch.zeros_like(b)

    bnrm2 = torch.linalg.norm(b)
    atol = max(float(atol), float(tol), float(rtol) * float(bnrm2))
    convergence: List[torch.Tensor] = []
    if verbose:
        print(f'MINRES: initial |b| = {bnrm2}, atol = {atol}')

    if bnrm2 == 0:
        return xk0, 0, convergence

    n = b.shape[0]
    if maxiter is None:
        maxiter = n * 10
    # The Lanczos process terminates (in exact arithmetic) by n steps, so the
    # basis it can ever need is at most n columns.
    kmax = max(0, min(int(maxiter), n))

    if precond is not None and callable(precond):
        raise ValueError(
            'minresSolve needs the flat diagonal preconditioner (it applies a '
            'symmetrizing congruence, which a callable preconditioner cannot '
            'provide)')
    if precond is None:
        d = torch.ones_like(b)
    else:
        eps = torch.finfo(b.dtype).eps
        # d = sqrt(|D|) with D = 1/precond; the clamp only guards an exactly
        # zero diagonal entry (the IISPH builder already clamps away from 0)
        d = 1.0 / torch.sqrt(precond.abs().clamp(min=eps))

    # All the per-step tests below (initial residual, the beta breakdown
    # check, the rho estimate) run in the congruence-transformed space, where
    # the residual is d-weighted: ||r_orig|| <= ||d * r_orig|| / dmin, so the
    # transformed-space tolerance is dmin * atol (dmin == 1 when precond is
    # None). This keeps the family contract -- converge when ||b - A x|| <
    # atol in the ORIGINAL space (the final verification below uses atol
    # itself) -- instead of comparing the d-scaled residual against an
    # original-space floor, which a strong preconditioner (d << 1) would trip
    # instantly as a false "breakdown".
    dmin = float(d.min()) if d.numel() > 0 else 1.0
    atol_t = dmin * atol

    # --- symmetrizing congruence (see module docstring) ----------------------
    c = d * b
    u0 = xk0 / d

    def amat(v: torch.Tensor) -> torch.Tensor:
        return d * matvec(d * v)

    def thresholdExceeded(x: torch.Tensor) -> bool:
        if threshold is None:
            return False
        return bool(torch.any(torch.linalg.norm(x.view(-1, dim), dim=-1) > threshold))

    def finish(x_orig: torch.Tensor, status: int) -> Tuple[torch.Tensor, int, List[torch.Tensor]]:
        # stamp every returned iterate with its verified true residual (in the
        # original space), so callers always know the quality of what they got
        convergence.append(torch.linalg.norm(matvec(x_orig) - b))
        return x_orig, status, convergence

    # --- initial residual ----------------------------------------------------
    # zero u0 is the common production case: A~ @ 0 == 0, skip the matvec
    if x0 is None or not x0.any():
        r0 = c
    else:
        r0 = c - amat(u0)
    beta1 = torch.linalg.norm(r0)
    convergence.append(beta1)
    if beta1 < atol_t:
        if verbose:
            print(f'\t[  0] already converged, |r| = {beta1}')
        return xk0, 0, convergence
    if kmax == 0:
        return finish(xk0, -14)
    if thresholdExceeded(xk0):
        return finish(xk0, -12)

    # --- Lanczos + Givens-LQ -------------------------------------------------
    # math indexing (1-based, matching the plan's handoff spec):
    #   w = A~ v_j - beta_{j-1} v_{j-1},  alpha_j = v_j . w,
    #   w -= alpha_j v_j,  beta_j = ||w||,  v_{j+1} = w / beta_j
    # LQ bookkeeping (0-based arrays; index j0 = j - 1 holds the quantities of
    # math step j):
    #   r[j0] = r_j = hypot(u_j, beta_j),  c[j0] = u_j / r_j,  s[j0] = -beta_j/r_j
    #   u_j = alpha_j (j == 1) or s_{j-1} c_{j-2} beta_{j-1} + c_{j-1} alpha_j
    #         (c_0 := 1; the new Lanczos column is pre-rotated by ALL earlier
    #         Givens, so the c_{j-2} factor is essential -- without it the
    #         estimate diverges from the dense-lstsq reference from j = 3 on)
    #   m[j0-1] = c_{j-2} c_{j-1} beta_{j-1} - s_{j-1} alpha_j
    #   (second superdiag) L[j0-2, j0] = -beta_{j-1} s_{j-2}  (j >= 3)
    # z tracks Q_j^T (beta_1 e_1): after math step j the MINRES residual
    # estimate is rho_j = |z[j0 + 1]| (0-based).
    v = r0 / beta1
    V = torch.empty(n, kmax, device=b.device, dtype=b.dtype)
    V[:, 0] = v

    beta_prev = torch.zeros((), device=b.device, dtype=b.dtype)  # beta_0 = 0
    v_prev = None
    # Givens history, shifted one slot back at the end of each step: at the
    # start of math step j, (c_prev2, c_prev, s_prev) = (c_{j-2}, c_{j-1},
    # s_{j-1}) under the convention c_0 := 1 (the j == 1 branch never reads
    # c_prev2 / s_prev, so the seed below is safe)
    c_prev2 = None
    c_prev = torch.ones((), device=b.device, dtype=b.dtype)  # c_0 := 1
    s_prev = torch.zeros((), device=b.device, dtype=b.dtype)

    r = torch.empty(kmax, device=b.device, dtype=b.dtype)
    cArr = torch.empty(kmax, device=b.device, dtype=b.dtype)
    sArr = torch.empty(kmax, device=b.device, dtype=b.dtype)
    m = torch.empty(max(0, kmax - 1), device=b.device, dtype=b.dtype)
    z = torch.zeros(kmax + 1, device=b.device, dtype=b.dtype)
    z[0] = beta1
    # L~ is upper banded (band 2): diag r_i, superdiag m_i, second superdiag
    # -beta_{i+1} s_i; each new row / column gets exactly its three entries
    # written when it enters (the array is zero-initialized and entries never
    # change once written), so no per-step re-fill is needed
    L = torch.zeros(kmax, kmax, device=b.device, dtype=b.dtype)

    x_prev = xk0  # last fully recombined iterate (original space)
    for j0 in range(kmax):
        j = j0 + 1  # math step index (1-based)
        beta_jm1 = beta_prev  # beta_{j-1} (beta_0 = 0); captured before the
        # Lanczos stage below overwrites beta_prev
        w = amat(v)
        if v_prev is not None:
            w = w - beta_jm1 * v_prev
        alpha_j = torch.dot(v, w)
        w = w - alpha_j * v
        beta_j = torch.linalg.norm(w)
        if (not bool(torch.isfinite(beta_j))) or beta_j < atol_t:
            # breakdown: the Krylov subspace is (approximately) A-invariant
            # with the residual still above tolerance -- the current iterate
            # is the best MINRES can do here
            if verbose:
                print(f'\t[{j:3d}] Lanczos breakdown: beta = {beta_j} < {atol_t}')
            return finish(x_prev, -13)
        if j < kmax:
            v_prev = v
            beta_prev = beta_j
            v = w / beta_j
            V[:, j] = v  # column i holds v_{i+1}: v_{j+1} goes in column j

        # Givens step j: annihilate (u_j, beta_j) -> (r_j, 0). u_j is the
        # (j, j) entry of Q_{j-1}^T M_j -- the new Lanczos column after ALL
        # earlier rotations -- which is where the c_{j-2} factor enters
        if j == 1:
            u = alpha_j
        else:
            u = s_prev * c_prev2 * beta_jm1 + c_prev * alpha_j
            m[j0 - 1] = c_prev2 * c_prev * beta_jm1 - s_prev * alpha_j
        r_j = torch.sqrt(u * u + beta_j * beta_j)
        c_j = u / r_j
        s_j = -beta_j / r_j
        if not (bool(torch.isfinite(r_j)) and bool(torch.isfinite(c_j))
                and bool(torch.isfinite(s_j))):
            if verbose:
                print(f'\t[{j:3d}] non-finite LQ scalar: r={r_j} c={c_j} s={s_j}')
            return finish(x_prev, -13)
        r[j0] = r_j
        cArr[j0] = c_j
        sArr[j0] = s_j
        c_prev2, c_prev, s_prev = c_prev, c_j, s_j
        # NOTE: z[j0] / z[j0+1] are 0-dim *views* sharing storage with z --
        # clone the scalars before the indexed writes, or the second write
        # would read the just-updated z[j0] through the aliased view
        zj = z[j0].clone()
        zj1 = z[j0 + 1].clone()
        z[j0] = c_j * zj - s_j * zj1
        z[j0 + 1] = s_j * zj + c_j * zj1
        rho = z[j0 + 1].abs()
        convergence.append(rho)
        if verbose:
            print(f'\t[{j:3d}] residual estimate: {rho}')

        # normal equation: y = argmin ||M_j y - beta_1 e_1||, M_j = [T_j;
        # beta_j e_j^T]; with the LQ factorization M_j = Q_j L_j = Q_j
        # [L~_j; 0] (last row of L_j is zero, so the minimizer satisfies
        # L~_j y = z[:j] and the residual is rho_j) this is the
        # back-substitution on the upper banded L~_j against z[:j]
        L[j0, j0] = r_j
        if j >= 2:
            L[j0 - 1, j0] = m[j0 - 1]
        if j >= 3:
            L[j0 - 2, j0] = -beta_jm1 * sArr[j0 - 2]
        y = torch.linalg.solve_triangular(L[:j, :j], z[:j].unsqueeze(-1),
                                          upper=True).squeeze(-1)
        if not bool(torch.isfinite(y).all()):
            if verbose:
                print(f'\t[{j:3d}] non-finite normal-equation solution')
            return finish(x_prev, -13)

        # FULL recombination (not an incremental update): x_j = x0 + V_j y_j
        x_orig = d * (u0 + V[:, :j] @ y)
        if rho < atol_t:
            # the estimate is exact in exact arithmetic; in fp32 it can
            # under-report once orthogonality is lost, so verify with the true
            # residual before declaring convergence (one extra matvec)
            trueResid = torch.linalg.norm(matvec(x_orig) - b)
            if trueResid < atol:
                if verbose:
                    print(f'\t[{j:3d}] converged, |r| = {trueResid}')
                convergence.append(trueResid)
                return x_orig, j0, convergence
            # otherwise fall through and keep iterating
        if thresholdExceeded(x_orig):
            if verbose:
                print(f'\t[{j:3d}] x threshold exceeded')
            return finish(x_orig, -12)
        x_prev = x_orig

    return finish(x_prev, -14)
