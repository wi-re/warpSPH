"""The scheme + internal-solver configurations the benchmark suites measure.

Each entry pairs one `warpSPHIntegrators` time-integration scheme (resolved
through the exact name path `warpSPH.runner.buildContext` uses, i.e. an
`IntegrationSchemeType` member name) with, for the implicit entries, the
`NonlinearSolver` configuration that drives its stage equations:

* `FixedPointSolver` -- the registry-default Picard iteration, with its fixed
  iteration count varied (the "internal solver loop limit" axis); and
* `JFNKSolver` -- the Newton-Krylov rung, with the matvec choice (`'fd'`
  finite difference vs. `'jvp'` exact forward mode), the Newton tolerance,
  and the outer-iteration budget varied.

The registry is data, not code: adding a configuration is one line. Orders
are read from the integrator registry itself rather than duplicated here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

from warpSPHIntegrators.solvers import FixedPointSolver, JFNKSolver
from warpSPHIntegrators import get_scheme

#: `IntegrationSchemeType` member names of the explicit linear multistep
#: schemes -- the only ones whose nominal one-evaluation-per-step cost
#: requires threading `IntegrationResult.history` back into the next call
#: (see `warpSPHIntegrators.drivers.history.adams`'s docstring: called without it they
#: transparently re-run their high-order starter every step). The suite's
#: step loop threads `history=` for exactly these names.
MULTISTEP_SCHEMES = frozenset({'ab2', 'ab3', 'ab4', 'ab5', 'abm2', 'abm3', 'abm4'})


@dataclass(frozen=True)
class SchemeSpec:
    """One benchmarkable configuration: a time-integration scheme and, for
    implicit ones, the internal stage-equation solver it runs with."""

    key: str                          # unique registry key, e.g. 'sdirk2_jfnk_jvp_1e-6'
    label: str                        # display name, e.g. 'SDIRK2 + JFNK(jvp, tol=1e-6, maxit=15)'
    kind: str                         # 'explicit' | 'implicit'
    integrationScheme: str            # IntegrationSchemeType member name
    order: int                        # nominal temporal order (from the integrator registry)
    makeSolver: Optional[Callable]    # () -> NonlinearSolver; None for explicit schemes
    solverDesc: str                   # display string for the internal solver configuration


def _explicit(key: str, label: str, integrationScheme: str) -> SchemeSpec:
    return SchemeSpec(
        key=key, label=label, kind='explicit', integrationScheme=integrationScheme,
        order=get_scheme(integrationScheme).order, makeSolver=None, solverDesc='-')


def _implicit(tag: str, label: str, integrationScheme: str,
              stag: str, slabel: str, factory: Callable, sdesc: str) -> SchemeSpec:
    return SchemeSpec(
        key=f'{tag}_{stag}', label=f'{label} + {slabel}', kind='implicit',
        integrationScheme=integrationScheme,
        order=get_scheme(integrationScheme).order, makeSolver=factory, solverDesc=sdesc)


def _picard(iterations: int) -> Callable:
    return lambda: FixedPointSolver(iterations=iterations)


def _jfnk(matvec: str, tol: float, max_iterations: int) -> Callable:
    return lambda: JFNKSolver(matvec=matvec, tol=tol, max_iterations=max_iterations)

# --- Explicit schemes (one-step RK, symplectic, TVD, embedded, multistep) ----
# (key, label, IntegrationSchemeType member name). Order is looked up, not
# stated, so it cannot drift from the integrator registry.
_EXPLICIT = [
    ('euler',           'Euler (forward)',           'forwardEuler'),
    ('eulerSemi',       'Semi-implicit Euler',       'semiImplicitEuler'),
    ('rk2mid',          'RK2 (midpoint)',            'rungeKutta2'),
    ('rk2heun',         "RK2 (Heun's)",              'heunsMethod'),
    ('rk3',             'RK3',                       'rungeKutta3'),
    ('ssprk3',          'SSP RK3',                   'sspRK3'),
    ('tvrk3',           'TVD RK3',                   'tvdRK3'),
    ('rk4',             'RK4',                       'rungeKutta4'),
    ('rk4alt',          'RK4 (alternative)',         'rungeKutta4alt'),
    ('nystrom5',        'Nystrom 5th',               'nystrom5th'),
    ('bs32',            'Bogacki-Shampine 3(2)',     'bogackiShampine'),
    ('dp54',            'Dormand-Prince 5(4)',       'dormandPrince'),
    ('ck54',            'Cash-Karp 5(4)',            'cashKarp'),
    ('leapfrog',        'Leap Frog',                 'leapFrog'),
    ('symplecticEuler', 'Symplectic Euler',          'symplecticEuler'),
    ('velocityVerlet',  'Velocity Verlet',           'velocityVerlet'),
    ('pefrl',           'PEFRL',                     'pefrl'),
    ('ab2',             'Adams-Bashforth 2',         'ab2'),
    ('ab3',             'Adams-Bashforth 3',         'ab3'),
    ('ab4',             'Adams-Bashforth 4',         'ab4'),
    ('ab5',             'Adams-Bashforth 5',         'ab5'),
    ('abm2',            'Adams-Bashforth-Moulton 2', 'abm2'),
    ('abm3',            'Adams-Bashforth-Moulton 3', 'abm3'),
    ('abm4',            'Adams-Bashforth-Moulton 4', 'abm4'),
]

# --- DIRK tableaus (the four implicit schemes warpSPHIntegrators ships) ------
_DIRK = [
    ('be',     'Backward Euler',    'backwardEuler'),
    ('im',     'Implicit Midpoint', 'implicitMidpoint'),
    ('trap',   'Trapezoidal',       'trapezoidal'),
    ('sdirk2', 'SDIRK2',            'sdirk2'),
]

# --- Internal stage-equation solver configurations ----------------------------
# Picard at its fixed-count axis (2 is the registry default), then JFNK with
# the matvec choice and the (tol, max_iterations) loop-limit axis.
# `jfnk_*_1e-6` with maxit=15 is exactly the notebook's configuration.
_SOLVERS = [
    ('picard2',   'Picard(2)',   _picard(2),   'FixedPointSolver(iterations=2)'),
    ('picard4',   'Picard(4)',   _picard(4),   'FixedPointSolver(iterations=4)'),
    ('picard8',   'Picard(8)',   _picard(8),   'FixedPointSolver(iterations=8)'),
    ('picard16',  'Picard(16)',  _picard(16),  'FixedPointSolver(iterations=16)'),
    ('jfnk_fd_1e-4',   'JFNK(fd, tol=1e-4, maxit=5)',   _jfnk('fd', 1e-4, 5),
     "JFNKSolver(matvec='fd', tol=1e-4, max_iterations=5)"),
    ('jfnk_fd_1e-6',   'JFNK(fd, tol=1e-6, maxit=15)',  _jfnk('fd', 1e-6, 15),
     "JFNKSolver(matvec='fd', tol=1e-6, max_iterations=15)"),
    ('jfnk_fd_1e-8',   'JFNK(fd, tol=1e-8, maxit=20)',  _jfnk('fd', 1e-8, 20),
     "JFNKSolver(matvec='fd', tol=1e-8, max_iterations=20)"),
    ('jfnk_jvp_1e-4',  'JFNK(jvp, tol=1e-4, maxit=5)',  _jfnk('jvp', 1e-4, 5),
     "JFNKSolver(matvec='jvp', tol=1e-4, max_iterations=5)"),
    ('jfnk_jvp_1e-6',  'JFNK(jvp, tol=1e-6, maxit=15)', _jfnk('jvp', 1e-6, 15),
     "JFNKSolver(matvec='jvp', tol=1e-6, max_iterations=15)"),
    ('jfnk_jvp_1e-8',  'JFNK(jvp, tol=1e-8, maxit=20)', _jfnk('jvp', 1e-8, 20),
     "JFNKSolver(matvec='jvp', tol=1e-8, max_iterations=20)"),
]

SCHEMES = {}
for _key, _label, _enum in _EXPLICIT:
    SCHEMES[_key] = _explicit(_key, _label, _enum)
for _tag, _label, _enum in _DIRK:
    for _stag, _slabel, _factory, _sdesc in _SOLVERS:
        SCHEMES[f'{_tag}_{_stag}'] = _implicit(_tag, _label, _enum, _stag, _slabel, _factory, _sdesc)




# --- Default scheme sets per suite --------------------------------------------
# Deliberately not the full registry: the full 60+ entries is a many-hour run.
# `--schemes all` (or any explicit list) still reaches every entry.

#: Accuracy suite: one representative per explicit family (order 1-5,
#: symplectic, multistep) plus the implicit set the notebook motivates --
#: each DIRK tableau at its shipped Picard default, and the JFNK matvec axis
#: (fd vs. jvp) at the notebook's (tol=1e-6, maxit=15) configuration.
ACCURACY_DEFAULT = [
    'euler', 'rk2mid', 'rk3', 'ssprk3', 'rk4', 'dp54', 'leapfrog', 'ab4',
    'be_picard2', 'im_picard2', 'trap_picard2', 'sdirk2_picard2',
    'sdirk2_jfnk_fd_1e-6', 'sdirk2_jfnk_jvp_1e-6', 'be_jfnk_jvp_1e-6',
]

#: Performance suite: the same explicit spread, and for the implicit side the
#: cost-relevant trio -- shipped-default Picard, and JFNK's two matvecs.
PERFORMANCE_DEFAULT = [
    'euler', 'rk2mid', 'ssprk3', 'rk4', 'dp54', 'leapfrog', 'ab4',
    'sdirk2_picard2', 'sdirk2_jfnk_fd_1e-6', 'sdirk2_jfnk_jvp_1e-6', 'be_jfnk_jvp_1e-6',
]

#: Stability suite: every cheap explicit scheme (they are the ones that
#: actually have a stability limit to find).
STABILITY_EXPLICIT_DEFAULT = [
    'euler', 'rk2mid', 'rk3', 'ssprk3', 'tvrk3', 'rk4', 'dp54', 'leapfrog', 'ab4',
]

#: Stability suite: the two L-stable tableaus across the full internal-solver
#: loop-limit matrix (Picard count axis, JFNK matvec x tol x maxit axis) --
#: the question answered there is "which solver settings survive past the
#: explicit limit, and at what iteration cost".
STABILITY_IMPLICIT_DEFAULT = [
    f'{tag}_{stag}'
    for tag in ('be', 'sdirk2')
    for stag in ('picard2', 'picard4', 'picard8', 'picard16',
                 'jfnk_jvp_1e-4', 'jfnk_jvp_1e-6', 'jfnk_jvp_1e-8', 'jfnk_fd_1e-6')
]


def getScheme(key: str) -> SchemeSpec:
    """Resolve one registry key, case-insensitively, with a useful error."""
    try:
        return SCHEMES[key.lower()]
    except KeyError:
        known = ', '.join(sorted(SCHEMES))
        raise KeyError(f"Unknown scheme key {key!r}. Known keys: {known}") from None


def getSchemes(keys: Optional[List[str]], default: List[str]) -> List[SchemeSpec]:
    """Resolve a `--schemes` argument: None/empty -> the suite's default set,
    `['all']` -> the whole registry, otherwise the named keys in the given order."""
    if not keys:
        return [getScheme(k) for k in default]
    if len(keys) == 1 and keys[0].lower() == 'all':
        return list(SCHEMES.values())
    return [getScheme(k) for k in keys]
