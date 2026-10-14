"""Matrix-free (left-)preconditioned conjugate gradient (PCG) for the
incompressible pressure solve. Same ``matvec``-closure + flat-diagonal-``precond``
interface and same ``(x, status, convergence)`` return convention as
``bicgstab.bicgstabSolve`` / ``gmres.gmresSolve``; the
``atol = max(atol, tol, rtol*||b||)`` floor and the status-code family match that
family (PCG additionally uses ``-16`` for an indefinite / zero-curvature
breakdown, i.e. ``p^T A p <= 0`` or a non-finite curvature).

CG is only well-posed for a symmetric operator and converges in finitely many
steps only when it is symmetric *positive definite*. The discrete SPH pressure
operator is *expected* to be nonsymmetric and -- by its negative IISPH diagonal --
negative-definite, so the dispatch in ``incompressible.krylov`` flips the sign of
the operator/RHS/preconditioner before calling this when the diagonal is
negative, giving PCG a positive-definite system. If the operator is genuinely
indefinite (not merely negative-definite) -- the question the Phase-0 operator
probe settles -- PCG bails with ``-16``. See ``docs/historic_plans/INCOMPRESSIBLE_SOLVER_PLAN.md``.

Status codes: ``>= 0`` converged at that iterate; ``-12`` per-particle ``|x|``
threshold bailout; ``-14`` max-iteration budget exhausted; ``-16`` indefinite /
zero-curvature breakdown. ``convergence`` holds the per-iterate residual norm,
with the final entry always the verified true residual ``||b - A x||`` of the
returned iterate.

The solver itself is `warpSPHCore.linalg.krylov.cg` (device-predicated: the stopping and breakdown tests run on the device, the iterate is frozen from
the iteration a test fires, and the host reads one flag per `checkEvery` iterations -- bitwise the iterate, status and history of the host-branching
loop that lived here, `tests/linalg/_legacyKrylov.py` in the core). This module keeps the legacy `(x, status, convergence)` call.
"""

from typing import Callable, List, Optional, Tuple, Union

import torch

from warpSPHCore.linalg.krylov import cg as _cg

__all__ = ['cgSolve']


def cgSolve(
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
    checkEvery: int = 8,
) -> Tuple[torch.Tensor, int, List[torch.Tensor]]:
    """Solve ``matvec(x) == b`` matrix-free with (left-)preconditioned conjugate gradient. The operator must be symmetric positive definite (see
    the module docstring for how the IISPH operator's negative diagonal is sign-flipped by the dispatch before calling here)."""
    result = _cg(matvec, b, x0, tol=tol, rtol=rtol, atol=atol, maxiter=maxiter, precond=precond, threshold=threshold, dim=dim,
                 checkEvery=checkEvery)
    out = result.legacy()
    if verbose:
        print(f'PCG: status {out[1]}, {len(out[2])} residual entries, final |r| = {float(out[2][-1]) if out[2] else float("nan")}')
    return out
