"""`VelocityPairPolicy` implementations. A policy turns the two particle velocities into the pair
velocity difference `u_ij` that `dissipation.pi.computePi_pair` consumes.

`RawVelocity` is the identity: `u_ij = v_i - v_j`, bit for bit what `computePi_actual` computed before
the pair velocity became an argument (the Phase 1 gate is bit-identity against the M0 baseline).

`linearPairVelocity` is the midpoint reconstruction of Garcia-Senz & Cabezon (2026) Eqs. (11)-(12) /
Frontiere et al. (2017): with `x_ij = x_i - x_j`,

    v'_i = v_i - phi/2 J_i x_ij,    v'_j = v_j + phi/2 J_j x_ij,    u_ij = v'_i - v'_j,

`J` the velocity Jacobian (`J x` is the velocity change along `x`, see `gradient.py`). For `phi = 1` a linear
field gives `u_ij = 0`; `phi_ij = phi_ji` (the limiter is symmetric) keeps `u_ji = -u_ij`, which is what
keeps the pairwise viscosity antisymmetric. `limitedPairPhi` is CRKSPH's limiter (`limiters.py`), and
`reconstructPairVelocity` dispatches on the policy; `BalsaraLimited` scales phi by `1 - Bbar^p` (Eqs. 18-19).
"""

from warpSPHCore import *
import warp as wp
from warp.types import vector, matrix
from typing import Any
from ...configurations.moduleConfigurations.diffusionParameters import VelocityPairPolicy
from .limiters import computeVanLeer, crkLimiter

__all__ = ['rawPairVelocity', 'linearPairVelocity', 'limitedPairPhi', 'reconstructPairVelocity']


@wp.func
def rawPairVelocity(
    v_i: vector(dtype = scalar_t, length=Any), v_j: vector(dtype = scalar_t, length=Any), # type: ignore
):
    return v_i - v_j


@wp.func
def linearPairVelocity(
    v_i: vector(dtype = scalar_t, length=Any), v_j: vector(dtype = scalar_t, length=Any), # type: ignore
    J_i: matrix(shape=(Any, Any), dtype=scalar_t), J_j: matrix(shape=(Any, Any), dtype=scalar_t), # type: ignore
    x_ij: vector(dtype = scalar_t, length=Any), # type: ignore
    phi: scalar_t,
):
    """v'_i - v'_j with both velocities extrapolated phi/2 of the way to the pair midpoint (Garcia-Senz Eqs. 11-12).
    The operation order is CRKSPH's (`modules/crk/accel.py` before the extraction)."""
    v_corr_i = phi / scalar_t(2.0) * matmul(J_i, x_ij)
    v_corr_j = phi / scalar_t(2.0) * matmul(J_j, x_ij)
    v_dot_i = v_i - v_corr_i
    v_dot_j = v_j + v_corr_j
    return v_dot_i - v_dot_j


@wp.func
def limitedPairPhi(
    x_ij: vector(dtype = scalar_t, length=Any), # type: ignore
    h_i: scalar_t, h_j: scalar_t,
    v_i: vector(dtype = scalar_t, length=Any), v_j: vector(dtype = scalar_t, length=Any), # type: ignore
    J_i: matrix(shape=(Any, Any), dtype=scalar_t), J_j: matrix(shape=(Any, Any), dtype=scalar_t), # type: ignore
    kernel_int: wp.int32, dim: wp.int32,
    vanLeer: wp.bool, closePairTaper: wp.bool,
    eta_crit: scalar_t, eta_fold: scalar_t,
    limiterType: wp.int32 = 0,                  # a `LimiterType` value; 0 = VanLeerFrontiere
):
    """phi_ij in [0, 1]: the van Leer-like limiter of the two Jacobians' quadratic forms along x_ij (Eq. 13/17),
    times the close-pair Gaussian taper (Eq. 14). Without `vanLeer` phi is 0 (no reconstruction): that is
    CRKSPH's `enableVanLeerLimiter = False`, and the taper only ever multiplies the van Leer value."""
    phi = scalar_t(0.0)
    factor = scalar_t(1.0)
    if closePairTaper:
        factor = crkLimiter(x_ij, h_i, h_j, kernel_int, dim, eta_crit, eta_fold)
    if vanLeer:
        phi = computeVanLeer(x_ij, v_i, v_j, J_i, J_j, limiterType) * factor
    return phi


@wp.func
def reconstructPairVelocity(
    policy: wp.int32,
    v_i: vector(dtype = scalar_t, length=Any), v_j: vector(dtype = scalar_t, length=Any), # type: ignore
    J_i: matrix(shape=(Any, Any), dtype=scalar_t), J_j: matrix(shape=(Any, Any), dtype=scalar_t), # type: ignore
    x_ij: vector(dtype = scalar_t, length=Any), # type: ignore
    h_i: scalar_t, h_j: scalar_t,
    kernel_int: wp.int32, dim: wp.int32,
    eta_crit: scalar_t, eta_fold: scalar_t,
    B_i: scalar_t, B_j: scalar_t, balsaraPower: scalar_t,
    limiterType: wp.int32 = 0,                  # a `LimiterType` value; 0 = VanLeerFrontiere
):
    """u_ij for a `VelocityPairPolicy`. Callers on the `Raw` path should not even read the Jacobians (they may be a
    one-element placeholder) -- branch on the policy and call `rawPairVelocity` instead. `B_i`, `B_j` (the Balsara
    factors) are only read by `BalsaraLimited`."""
    if policy == wp.static(VelocityPairPolicy.Raw.value):
        return rawPairVelocity(v_i, v_j)
    phi = scalar_t(1.0)
    if policy != wp.static(VelocityPairPolicy.Linear.value):
        phi = limitedPairPhi(x_ij, h_i, h_j, v_i, v_j, J_i, J_j, kernel_int, dim, True, True, eta_crit, eta_fold, limiterType)
        phi = wp.max(wp.min(phi, scalar_t(1.0)), scalar_t(0.0))
    if policy == wp.static(VelocityPairPolicy.BalsaraLimited.value):
        # Garcia-Senz & Cabezon (2026) Eqs. (18)-(19): reconstruct less where the flow is compressive
        B_bar = scalar_t(0.5) * (B_i + B_j)
        # guarded before the pow, not after: B = 0 exactly (a fluid at rest) is a legal state, and pow's adjoint at
        # a zero base is not safe for every p (the AD rule of limiters.py's van Leer division, AV_PLAN §2.6)
        if B_bar > scalar_t(0.0):
            phi = phi * (scalar_t(1.0) - wp.pow(B_bar, balsaraPower))
    return linearPairVelocity(v_i, v_j, J_i, J_j, x_ij, phi)
