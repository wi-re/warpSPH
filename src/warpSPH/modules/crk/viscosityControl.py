"""Opt-in previous-RHS viscosity-gradient timing for midpoint comparisons.

Spheral's non-rigorous Q gradient is copied from hydro derivatives after
each state update. The shared step holder carries stage zero's gradient to
the midpoint; the final state's cache carries the midpoint gradient to the
next step. Replays must seed a new holder from the initial state's cache.
The first RHS seeds from its current gradient (exact for the zero-velocity
initial conditions used by the comparison; not a general Spheral startup
gradient reconstruction). Pressure, continuity and support use current data.
"""


def viscosityGradient(system, current, mode):
    if mode == 'current':
        return current
    if mode != 'cached':
        raise ValueError('Unknown CRK viscosity gradient timing')
    if system.pairWork is None:
        raise RuntimeError('Cached CRK Q gradient requires a midpoint step holder')
    previous = system.pairWork.crkQGradient
    system.pairWork.crkQGradient = current
    system.state.crkQGradient = current
    return current if previous is None else previous
