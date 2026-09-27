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
    and the statistics are bitwise the ones `evaluateResidual` returns."""
    r = residual.index_select(0, fluidIndex)
    if criterion is JacobiConvergenceCriterion.flooredOneSided:
        stat = torch.mean(torch.clamp(-r, min=-threshold))
    elif criterion is JacobiConvergenceCriterion.oneSided:
        stat = torch.mean(-r)
    else:
        stat = torch.mean(torch.abs(r))
    if not relative:
        return stat, None
    if criterion is JacobiConvergenceCriterion.meanAbsolute:
        return stat, stat
    return stat, torch.mean(torch.abs(r))


class ConvergenceCheckSchedule:
    """At which iterations a relaxed-Jacobi loop reads its convergence test
    back to the host (see `RelaxedJacobiSolverConfig.convergenceCheckSchedule`).

    Each read is a device sync, and the loops used to read after every
    iteration -- on `tgv` (divergenceFree, 16k particles) ~310 syncs per step.
    'adaptive' assumes this solve needs about as many iterations `P` as the
    previous one did to *first* meet the test, and checks at ceil(0.50 P),
    ceil(0.75 P), ceil(0.85 P), then every other iteration. The previous count
    is the first-converged iteration recovered from the device-side history,
    not the (possibly overshot) iteration the loop stopped at, so overshoot
    does not feed back into the next solve's schedule.

    `key` names the solve (a step may run several differently-behaved solves);
    the history lives on `store` (the scheme config) under
    ``_jacobiIterationHistory``.
    """

    def __init__(self, store, key: str, mode: str, minIters: int, maxIters: int, verbose: bool = False):
        self.store, self.key = store, key
        self.minIters, self.maxIters = minIters, maxIters
        history = getattr(store, '_jacobiIterationHistory', None)
        if history is None:
            history = {}
            try:
                setattr(store, '_jacobiIterationHistory', history)
            except Exception:  # noqa: BLE001 -- a frozen/slotted config: no memory, check every iteration
                pass
        self.history = history
        previous = history.get(key)
        self.every = mode != 'adaptive' or verbose or previous is None
        if not self.every:
            import math
            self.marks = sorted({max(1, math.ceil(f * previous)) for f in (0.50, 0.75, 0.85)})
            self.dense = self.marks[-1]

    def check(self, i: int) -> bool:
        """Read the test back after iteration index ``i`` (``i + 1`` done)?"""
        if i < self.minIters:
            return False            # the loop may not stop here anyway
        if self.every:
            return True
        n = i + 1
        return n in self.marks or (n > self.dense and (n - self.dense) % 2 == 0)

    def record(self, convergedFlags) -> None:
        """Remember the first iteration (count) that met the test, from the
        host copy of the per-iteration flags; `maxIters` if none did."""
        first = next((k + 1 for k, c in enumerate(convergedFlags) if c and k >= self.minIters),
                     self.maxIters)
        self.history[self.key] = first
