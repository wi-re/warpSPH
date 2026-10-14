"""Matrix-free (left-)preconditioned bi-conjugate gradient (BiCG) for the
incompressible pressure solve. Same ``matvec``-closure + flat-diagonal-``precond``
interface and same ``(x, status, convergence)`` return convention as
``bicgstab.bicgstabSolve`` / ``gmres.gmresSolve``; the
``atol = max(atol, tol, rtol*||b||)`` floor and the status-code family match that
family.

BiCG is the only one of the four Krylov methods that needs the **adjoint**
operator ``A^T`` (the dual residual lives in the ``A^T``-Krylov space), so it is
the most expensive to wire in: ``warpSPHCore`` exposes no ready adjoint of the
pressure ``accel``/``shift`` pair, so the dispatch in ``incompressible.krylov``
supplies ``A^T`` via ``buildIISPHMatvecT`` -- currently a *self-adjoint
placeholder* (``A^T ~= A``). That placeholder is only correct if the operator is
symmetric, which the Phase-0 operator probe quantifies; a rigorously derived
transpose is the Phase-4 work item. Until then BiCG results are provisional.

Method: left-preconditioned BiCG on ``M^{-1} A x = M^{-1} b``. The adjoint of
``M^{-1} A`` is ``A^T M^{-T}``; for the diagonal IISPH preconditioner
``M^{-T} = M^{-1}``, so the adjoint matvec is ``psolve(matvecT(q))``. One forward
(``A p``) and one adjoint (``A^T p``) matvec per iterate.

Status codes: ``>= 0`` converged at that iterate; ``-10`` rho breakdown;
``-11`` alpha/denominator/beta breakdown; ``-12`` per-particle ``|x|`` threshold
bailout; ``-14`` max-iteration budget exhausted. ``convergence`` holds the
per-iterate (preconditioned) residual norm, with the final entry always the
verified true residual ``||b - A x||`` of the returned iterate.

The solver itself is `warpSPHCore.linalg.krylov.bicg` (device-predicated, see `cg.py`). 2026-10-14: the loop that lived here was not BiCG -- its `beta`
carried the BiCGStab factor `alpha / alpha_prev` and was lagged by one iteration, and its adjoint applied the preconditioner on the wrong side
(`M^{-1} A^T q` instead of `A^T M^{-1} q`); on a symmetric system it needed 14 iterations where CG needs 9. The core version is textbook BiCG and
coincides with CG for a symmetric operator (`tests/linalg/test_krylovBiCG.py`). This module keeps the legacy `(x, status, convergence)` call.
"""

from typing import Callable, List, Optional, Tuple, Union

import torch

from warpSPHCore.linalg.krylov import bicg as _bicg

__all__ = ['bicgSolve']


def bicgSolve(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    matvecT: Callable[[torch.Tensor], torch.Tensor],
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
    """Solve ``matvec(x) == b`` matrix-free with (left-)preconditioned BiCG. ``matvecT`` is the adjoint operator (required by the dual-residual
    update). ``convergence`` holds the per-iterate (preconditioned) residual norm, with the final entry of a failing solve the verified true
    residual ``||b - A x||`` (matching ``bicgstabSolve``)."""
    out = _bicg(matvec, matvecT, b, x0, tol=tol, rtol=rtol, atol=atol, maxiter=maxiter, precond=precond, threshold=threshold, dim=dim,
                checkEvery=checkEvery).legacy()
    if verbose:
        print(f'BiCG: status {out[1]}, {len(out[2])} residual entries')
    return out
