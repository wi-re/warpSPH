"""Residual statistics shared with ACSPH; native projection policies live in iisph/dfsph."""
from enum import Enum

__all__ = ['JacobiConvergenceCriterion']

class JacobiConvergenceCriterion(Enum):
    """Which statistic of the residual the relaxed-Jacobi loops compare against
    ``tolerance``.

    The two solvers shipped two different tests and neither was reachable from
    the config, so `DFSPH_IMPROVEMENT_PLAN.md` §1.7 -- "the stopping criterion
    is broken" -- could be argued but not measured. These are the same two
    tests plus the published one, as one setting. With ``r = b - A p`` over
    fluid rows:

    - ``flooredOneSided``: ``mean(clamp(-r, min=-tolerance))``.
      `solveIncompressible`'s historical test, and its default. One-sided (only
      over-compression counts towards the error) *and* floored, so an
      under-dense particle contributes at most ``-tolerance`` rather than its
      full negative value and cannot cancel an over-dense one. Neither
      published criterion floors: [BK] Alg. 3 tests ``rho_avg - rho0 > eta``
      and [I] §5.1 the same shape, both on the plain average.
    - ``oneSided``: ``mean(-r)``, i.e. the published form with the floor
      removed. This is the one-line difference §1.7 identifies.
    - ``meanAbsolute``: ``mean(|r|)``. Both divergence-free loops' historical
      test, and their default. The only one of the three that is a norm, so the
      only one that cannot be satisfied by cancellation.

    The three are *not* interchangeable at a fixed `tolerance` -- they have
    different scales, and a value tuned for one is meaningless for another.
    Changing the criterion means re-tuning `tolerance` with it.

    Orthogonal to this, `rtol`/`atol` add a relative disjunct
    (``mean|r| <= atol + rtol * mean|b|``) to whichever statistic is chosen.

    See `modules/incompressible/convergence.py` and Part 15.
    """
    flooredOneSided = 0
    oneSided = 1
    meanAbsolute = 2
