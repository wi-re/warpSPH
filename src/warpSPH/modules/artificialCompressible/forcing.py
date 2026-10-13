"""Body forces of one ACSPH real step: gravity, the optional mDBC no-penetration
acceleration and the case forcing."""

import torch

from ..boundaryConditions import computeForcing
from ..gravity import computeGravity
from ..mdbc import computeMdbcNoPenShift


def bodyForces(system, dt, config, schemeConfig, adjacency):
    """`(bodyForce, wallBodyForce)`: the acceleration added to the momentum residual,
    and `(g - a_wall)` for the Eq. (61) wall pressure (None without gravity: the
    correction is then identically zero).

    The mDBC no-penetration shift (`schemeConfig.noPenetrationShift`, off by
    default, not in the paper) enters as an acceleration `shift / dt`. Static walls
    have `a_wall = 0`; a moving-wall ACSPH case would need a per-particle wall
    acceleration (`modules/mdbc/velocity.py` documents the same gap).
    `computeForcing` returns a *force*, divided by mass here as `deltaSPH_step` does."""
    state = system.state
    noPenShift = torch.zeros_like(state.velocities)
    if schemeConfig.noPenetrationShift:
        noPenShift = computeMdbcNoPenShift(state, config, schemeConfig, adjacency) / dt
    gravity = computeGravity(state, config, schemeConfig, adjacency)
    forcing = computeForcing(system, config.dt, system.t, config,
                             schemeConfig) / state.masses.view(-1, 1)
    wallBodyForce = gravity if bool(gravity.abs().any()) else None
    return gravity + noPenShift + forcing, wallBodyForce
