"""Riemann solvers for the ideal-gas Euler equations on a pair normal (AV_PLAN Phase 7b).

The second half of a Godunov pair flux: `modules/reconstruction` produces the left / right states, this module turns
them into the interface state. Written as branch-free `wp.func`s with no iteration, so they can be called from any pair
loop (the Riemann dissipation term, a future MFM / MFV flux):

* `riemannStarState(solver, ...)` -> `(p*, u*)`, for every `RiemannSolver`;
* `hllcFlux(...)` -> the face flux `(F_rho, F_momentum, F_energy, S*)` in the face frame.

The exact (iterative) solution is the test reference only (`tests/test_riemann.py`).
"""

from .solvers import riemannStarState, hllcFlux, RiemannSolver

__all__ = ['riemannStarState', 'hllcFlux', 'RiemannSolver']
