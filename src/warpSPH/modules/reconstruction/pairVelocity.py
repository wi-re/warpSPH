"""`VelocityPairPolicy` implementations. A policy turns the two particle velocities into the pair
velocity difference `u_ij` that `dissipation.pi.computePi_pair` consumes.

`RawVelocity` is the identity: `u_ij = v_i - v_j`, bit for bit what `computePi_actual` computed before
the pair velocity became an argument (the Phase 1 gate is bit-identity against the M0 baseline)."""

from warpSPHCore import *
import warp as wp
from warp.types import vector
from typing import Any

__all__ = ['rawPairVelocity']


@wp.func
def rawPairVelocity(
    v_i: vector(dtype = scalar_t, length=Any), v_j: vector(dtype = scalar_t, length=Any), # type: ignore
):
    return v_i - v_j
