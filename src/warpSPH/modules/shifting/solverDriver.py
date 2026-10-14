"""Shared solve driver for the implicit shifting Newton step: picks the
primary Krylov solver (`ShiftingImplicitSolver`), runs it, and -- when
`ShiftProperties.implicitFallback` is set and the primary bails out (status
`< 0`) -- runs the fallback chain and returns the best iterate by stamped
true residual. Used by both `implicitShifting.computeImplicitShift` and
`implicitShiftingAutomatic.computeImplicitShiftAutomatic`, so the two stay in
lockstep.

The default (`implicitFallback == ShiftingImplicitFallback.none`) runs exactly
one solver and returns its result unconditionally -- byte-identical to the
pre-fallback behavior, so `ShiftingScheme.implicit` users see no change. The
fallback chain is opt-in:

  - `krylov`: on a primary bailout, retry with the *other* Krylov solver
    (BiCGStab<->GMRES) from a clean start `x0` and keep the better iterate
    (by stamped residual). This is the high-value fallback: the two solvers
    fail on different regimes -- e.g. the troubleshooting harness shows
    GMRES converging the near-breakdown cases where BiCGStab hits its rho
    guard.
  - `krylov_richardson`: `krylov`, plus a bounded Richardson polish
    (`richardson.richardsonSolve`) warm-started from the best Krylov iterate,
    as a last resort. Richardson is deliberately last: an eigenvalue probe
    shows it converges but is much slower than GMRES on the SPD
    `legacyPairwise` operator, and does not converge at all on the indefinite
    `exactHessian` operator (see `richardson.py`'s module docstring).

On any fallback activation a `warnings.warn` is emitted with the primary's
status and stamped residual, so a silently-bailed inner solve is no longer
invisible (the pre-fallback behavior used a bailed-out `xk` exactly as if it
had converged).
"""

import warnings
from typing import Callable, List, Tuple

import torch

from .bicgstab import bicgstabSolve
from .gmres import gmresSolve
from .richardson import richardsonSolve
from ...configurations.moduleConfigurations.shifting import ShiftingImplicitFallback, ShiftingImplicitSolver

__all__ = ['solveImplicitSystem', 'runKrylov']

# the Richardson polish is a bounded last resort, not a primary solver: cap it
# well below the Krylov budget so a full fallback chain can't blow the step's
# matvec cost (see richardson.py for its convergence characteristics)
_RICHARDSON_POLISH_MAXITER = 32


def runKrylov(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    x0: torch.Tensor,
    solver: ShiftingImplicitSolver,
    solverArgs: dict,
    restart: int,
) -> Tuple[torch.Tensor, int, List[torch.Tensor]]:
    """Runs one Krylov solver with `solverArgs` (plus GMRES's `restart`);
    returns `(x, iters, convergence)`. `solverArgs` must carry `tol`, `rtol`,
    `maxiter`, `precond`, `threshold`, `dim` -- the dict both production
    callers already build."""
    if solver == ShiftingImplicitSolver.gmres:
        # (GMRES reads the host once per cycle; the still-running-flag interval is for the device-predicated solvers)
        return gmresSolve(matvec, b, x0, restart=restart, **{k: v for k, v in solverArgs.items() if k != 'checkEvery'})
    return bicgstabSolve(matvec, b, x0, **solverArgs)


def _stampedResidual(convergence: List[torch.Tensor]) -> float:
    # both Krylov solvers stamp the verified true residual as their final
    # history entry on every return, so this is the quality of the iterate
    return float(convergence[-1]) if convergence else float('inf')


def solveImplicitSystem(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    B: torch.Tensor,
    x0: torch.Tensor,
    solverArgs: dict,
    primary_solver: ShiftingImplicitSolver,
    restart: int,
    fallback: ShiftingImplicitFallback,
) -> Tuple[torch.Tensor, int, List[torch.Tensor]]:
    """Solves `matvec(x) == B` with the primary Krylov solver, applying the
    opt-in fallback chain when (and only when) the primary bails out
    (status `< 0`) and `fallback != none`. Returns the best iterate by
    stamped true residual, with its status and history."""
    xk, iters, conv = runKrylov(matvec, B, x0, primary_solver, solverArgs, restart)
    if fallback == ShiftingImplicitFallback.none or iters >= 0:
        # no fallback requested, or the primary converged: the result is the
        # primary's, exactly as the pre-fallback code produced it
        return xk, iters, conv

    warnings.warn(
        f'implicit shifting: primary {primary_solver.name} solve bailed out '
        f'(status {iters}, stamped residual {_stampedResidual(conv):.3e}); '
        f'running implicitFallback={fallback.name}')

    best = (xk, iters, conv)
    if fallback in (ShiftingImplicitFallback.krylov, ShiftingImplicitFallback.krylov_richardson):
        other = (ShiftingImplicitSolver.gmres if primary_solver == ShiftingImplicitSolver.bicgstab
                 else ShiftingImplicitSolver.bicgstab)
        x2, i2, c2 = runKrylov(matvec, B, x0, other, solverArgs, restart)
        if _stampedResidual(c2) <= _stampedResidual(best[2]):
            best = (x2, i2, c2)
    if fallback == ShiftingImplicitFallback.krylov_richardson:
        # bounded last resort, warm-started from the best Krylov iterate: if
        # that iterate already meets the tolerance this returns immediately
        # (its initial-residual check passes)
        xr, ir, cr = richardsonSolve(
            matvec, B, best[0],
            tol=solverArgs['tol'], rtol=solverArgs['rtol'],
            maxiter=min(int(solverArgs['maxiter']), _RICHARDSON_POLISH_MAXITER),
            threshold=solverArgs['threshold'], dim=solverArgs['dim'])
        if _stampedResidual(cr) < _stampedResidual(best[2]):
            best = (xr, ir, cr)
    return best