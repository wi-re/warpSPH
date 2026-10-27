"""The relaxed-Jacobi solvers' stopping test, in one place.

Three loops iterate the same `A p = b` and each one used to spell its own
convergence test inline: `solveIncompressible`, `_solveDivergenceFreeImpl` and
`_solveDivergenceFreeOptimal`. They did not spell the same one --
`solveIncompressible` compared a *floored one-sided average* of the residual
against `tolerance` while both divergence-free loops compared a *mean absolute*
one -- and neither form was reachable from the config, so nothing could be
measured against anything. `JacobiConvergenceCriterion` names the three forms
and `evaluateResidual` computes them, so the two historical behaviours are
now two values of one setting rather than two pieces of code.

The forms, with `r = b - A p` restricted to fluid rows:

- ``flooredOneSided``: ``mean(clamp(-r, min=-tolerance))``.
  `solveIncompressible`'s historical test. One-sided (only over-compression
  counts) *and* floored: an under-dense particle contributes at most
  `-tolerance` instead of its full negative value, so it cannot cancel an
  over-dense one. Both published criteria are one-sided on the plain average
  ([BK] Alg. 3 `rho_avg - rho0 > eta`, [I] §5.1) and neither floors.
- ``oneSided``: ``mean(-r)``. The published form.
- ``meanAbsolute``: ``mean(|r|)``. Both divergence-free loops' historical test,
  and the only one of the three that is a norm.

`rtol`/`atol` add a *relative* disjunct on top, the same contract the Krylov
path already states: stop when ``mean|r| <= atol + rtol * mean|b|``. It is a
disjunction, not a replacement -- either test satisfied ends the solve --
because the absolute test is the one the papers state and the relative one is
the one that stays meaningful when the source carries a component the operator
cannot remove (`DFSPH_IMPROVEMENT_PLAN.md` §1.1, §1.7). `rtol = 0` disables it.

See `DFSPH_IMPROVEMENT_PLAN.md` §1.7 and Part 15.
"""

__all__ = ['evaluateResidual', 'sourceNorm', 'residualStats', 'ConvergenceCheckSchedule']

from typing import Optional, Tuple

import torch

from warpSPHCore.linalg import CheckSchedule, residualStatistic

from ...configurations import JacobiConvergenceCriterion


def sourceNorm(sourceTerm: torch.Tensor, fluidMask: torch.Tensor,
               rtol: float) -> Optional[float]:
    """``mean|b|`` over fluid rows, for the relative disjunct -- or `None` when
    `rtol` is 0 and the relative test is off.

    The source does not change during a solve, so this is hoisted out of the
    iteration: the loop then costs one device->host transfer per iteration,
    exactly as it did before the relative test existed.
    """
    if rtol <= 0.0:
        return None
    return float(torch.mean(torch.abs(sourceTerm[fluidMask])).cpu())


def evaluateResidual(residual: torch.Tensor, fluidMask: torch.Tensor,
                     criterion: JacobiConvergenceCriterion, threshold: float,
                     bNorm: Optional[float]) -> Tuple[float, Optional[float]]:
    """Return ``(error, rNorm)`` for one iteration's residual.

    `error` is the configured statistic, compared against `tolerance` by the
    caller. `rNorm` is ``mean|r|`` for the relative disjunct, or `None` when
    that test is off. Both are read back in a single transfer, so turning the
    relative test on does not add a second synchronisation per iteration.
    """
    r = residual[fluidMask]
    if criterion is JacobiConvergenceCriterion.flooredOneSided:
        stat = torch.mean(torch.clamp(-r, min=-threshold))
    elif criterion is JacobiConvergenceCriterion.oneSided:
        stat = torch.mean(-r)
    else:
        stat = torch.mean(torch.abs(r))

    if bNorm is None:
        return float(stat.cpu()), None
    if criterion is JacobiConvergenceCriterion.meanAbsolute:
        # Already the norm the relative test wants -- do not compute it twice.
        both = stat.cpu()
        return float(both), float(both)
    both = torch.stack([stat, torch.mean(torch.abs(r))]).cpu()
    return float(both[0]), float(both[1])


def residualStats(residual: torch.Tensor, fluidIndex: torch.Tensor,
                  criterion: JacobiConvergenceCriterion, threshold: float,
                  relative: bool) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """`evaluateResidual` without the device->host read: ``(stat, rNorm)`` as
    0-d device tensors (``rNorm`` is ``None`` when the relative test is off).
    `fluidIndex` is ``fluidMask.nonzero()`` computed once per solve, so the
    gather selects the same rows in the same order as ``residual[fluidMask]``
    and the statistics are bitwise the ones `evaluateResidual` returns.
    (`warpSPHCore.linalg.residualStatistic`, with the criterion by name.)"""
    return residualStatistic(residual, fluidIndex, criterion.name, threshold, relative)


#: the check schedule lives in the core now (`warpSPHCore.linalg.CheckSchedule`); the old name stays for the existing imports
ConvergenceCheckSchedule = CheckSchedule
