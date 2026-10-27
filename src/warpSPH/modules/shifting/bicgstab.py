"""Matrix-free (preconditioned) BiCGStab, ported from diffSPH's
`sparseSolver.bicg.bicgstab_shifting`, generalized to take a caller-supplied
`matvec` closure and a flat diagonal `precond` vector instead of diffSPH's
hardcoded 2-component scatter-sum and `(values, (row, col), n)` sparse-COO
preconditioner -- both of which were always diagonal in practice here, so
`psolve` is just an elementwise multiply.

The per-iterate recurrence (rho/alpha/omega updates) is unchanged from the
port, so on non-pathological inputs the iterate sequence is bit-identical to
diffSPH's `bicgstab_shifting` (the check the regression note in
`docs/regression/implicit_shifting_operator_choice.md` relies on). The
changes below only alter what happens near breakdown and what is reported:

- `tol` is now enforced: `atol = max(atol, tol, rtol*||b||)`. The port
  accepted `tol` but never read it, so callers' absolute tolerance (e.g.
  `ShiftProperties.implicitTolerance`) had no effect. Both this function's
  and the production config's default are `0.0` (= "relative tolerance
  only"), which is the behavior the port actually had.
- Breakdown guards are relative as well as absolute, in addition to the
  port's absolute `eps**2` floor:
  - `|rho| < 1e-6*||r0||*||rk||` -- `|rho|/(||r0||*||rk||)` is
    `|cos(angle(rk, r0))|`, a scale-free indicator of the `alpha = rho/rv`
    blow-up; the port's bare `eps**2` (1.4e-14 in fp32) only caught rho
    values at the edge of underflow, so near-breakdown produced a
    ~1e6-scaled update instead of a clean bailout.
  - `|rv| < 1e-6*|rho|` replaces the port's `rv == 0` exact-zero test
    (same rationale: it caught nothing short of exact underflow).
- `omega` is now validated **before** the update that uses it (the port
  validated after, so a broken `omega` had already corrupted `xk`), with a
  `torch.isfinite` guard (the port's `abs(omega) < eps**2` is `False` for
  NaN/Inf, so those passed the check), and the port's `iteration > 0` skip
  is dropped.
- 2 matvecs/iterate instead of the port's 3: the per-iterate `convergence`
  entry is the recurrence residual norm (the port's 3rd matvec recomputed
  the true residual every iterate purely to record it). The true residual is
  now computed once, on every return: it verifies `||b - A xk|| < atol`
  before a recurrence-based convergence is declared (and the iterate is
  stamped with the verified value), and it is appended on breakdown/
  threshold/budget bailouts too, so `convergence[-1]` always reports the
  actual quality of the returned iterate.
- Status codes: `>= 0` converged at that iterate; `-10` rho breakdown;
  `-11` rv/omega breakdown; `-12` per-particle `|x|` threshold bailout;
  `-14` max-iteration budget exhausted (the port returned `maxiter` here,
  indistinguishable from a converged status). The port's s-convergence
  branch returned a literal `0` at every iterate; it now returns the actual
  `iteration`.

The solver itself is `warpSPHCore.linalg.krylov.bicgstab` (device-predicated, see `cg.py`; the module docstring there lists the two intended
differences from the host-branching loop that lived here -- the verification of a would-be converged residual moved behind the loop with a restart
if the recurrence lied, and the half-step candidate uses the preconditioned direction). This module keeps the legacy `(x, status, convergence)`
call; `convergence[-1]` is the verified true residual of the returned iterate on every return.
"""

from typing import Callable, List, Optional, Tuple, Union

import torch

from warpSPHCore.linalg.krylov import bicgstab as _bicgstab

__all__ = ['bicgstabSolve']


def bicgstabSolve(
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
    """Solves `matvec(x) == b` matrix-free. `tol` is an absolute residual floor (0.0 = relative tolerance only), `rtol*||b||` the relative one,
    `atol` an additional explicit floor. See the module docstring for the status-code convention and the `convergence` list semantics."""
    out = _bicgstab(matvec, b, x0, tol=tol, rtol=rtol, atol=atol, maxiter=maxiter, precond=precond, threshold=threshold, dim=dim,
                    checkEvery=checkEvery).legacy()
    if verbose:
        print(f'BiCGStab: status {out[1]}, {len(out[2])} residual entries')
    return out
