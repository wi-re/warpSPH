"""A shallow view of a state with the three evolving fields replaced."""

import copy


def workingState(state, positions, velocities, pressures):
    """A shallow view of `state` with the three evolving fields replaced. The
    SPH operators read attributes off whatever they are handed, so this is all
    an intermediate RK stage needs -- and it keeps the real state untouched
    until the pseudo-time loop has converged."""
    view = copy.copy(state)
    view.positions = positions
    view.velocities = velocities
    view.pressures = pressures
    return view
