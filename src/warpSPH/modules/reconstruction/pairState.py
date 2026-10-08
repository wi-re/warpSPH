"""Left / right states of a scalar field at the pair midpoint (GODUNOV_SPH_PLAN layer 1): the scalar twin of
`pairVelocity.linearPairVelocity`.

With `x_ij = x_i - x_j` and the per-particle gradients `g_i`, `g_j` of a scalar `s` (`g . dx` = the change of `s` along `dx`),

    s'_i = s_i - phi/2 (g_i . x_ij),    s'_j = s_j + phi/2 (g_j . x_ij),

the two particles' linear profiles evaluated at the midpoint. `phi_ij` is the same limiter as the velocity's: the `LimiterType`
function of `r = min(a/b, b/a)`, `a = g_i . x_ij`, `b = g_j . x_ij`, times the close-pair taper (`limiters.crkLimiter`), symmetric in
`i <-> j`. A reconstructed state is additionally kept between the two particle values, so a positive field stays positive and
no new extremum is created (the monotonicity MUSCL asks of an interface state; Inutsuka 2002 Eq. 74 imposes the same on the
velocity gradient).
"""

from warpSPHCore import *
import warp as wp
from warp.types import vector, matrix
from typing import Any
from .limiters import limiterPsi, crkLimiter

__all__ = ['reconstructPairScalar', 'reconstructPairRiemannStates', 'pairRiemannStates']


@wp.func
def reconstructPairScalar(
    s_i: scalar_t, s_j: scalar_t,
    g_i: vector(dtype = scalar_t, length=Any), g_j: vector(dtype = scalar_t, length=Any), # type: ignore
    x_ij: vector(dtype = scalar_t, length=Any), # type: ignore
    taper: scalar_t, limiterType: wp.int32,
):
    """`(s'_i, s'_j)`, the midpoint states of `s`. `taper` multiplies the limiter (the close-pair taper, 1 for none)."""
    a = wp.dot(g_i, x_ij)
    b = wp.dot(g_j, x_ij)
    # the two extrapolations' ratio, guarded before the division (the AD rule of limiters.py's van Leer ratio)
    ri = scalar_t(1.0)
    if wp.abs(b) > scalar_t(1.0e-30):
        ri = a / b
    rj = scalar_t(1.0)
    if wp.abs(a) > scalar_t(1.0e-30):
        rj = b / a
    phi = limiterPsi(limiterType, wp.min(ri, rj)) * taper
    phi = wp.max(wp.min(phi, scalar_t(1.0)), scalar_t(0.0))
    sl = s_i - scalar_t(0.5) * phi * a
    sr = s_j + scalar_t(0.5) * phi * b
    lo = wp.min(s_i, s_j)
    hi = wp.max(s_i, s_j)
    return wp.min(wp.max(sl, lo), hi), wp.min(wp.max(sr, lo), hi)


@wp.func
def reconstructPairRiemannStates(
    rho_i: scalar_t, rho_j: scalar_t, P_i: scalar_t, P_j: scalar_t,
    G_i: matrix(shape=(Any, Any), dtype=scalar_t), G_j: matrix(shape=(Any, Any), dtype=scalar_t), # type: ignore
    x_ij: vector(dtype = scalar_t, length=Any), # type: ignore
    h_i: scalar_t, h_j: scalar_t, kernel_int: wp.int32, dim: wp.int32,
    eta_crit: scalar_t, eta_fold: scalar_t, limiterType: wp.int32,
):
    """`(rho'_i, rho'_j, P'_i, P'_j)`: the density and pressure of the pair at the midpoint. `G` is the per-particle
    `(2, dim)` matrix of the density gradient (row 0) and the pressure gradient (row 1)."""
    taper = crkLimiter(x_ij, h_i, h_j, kernel_int, dim, eta_crit, eta_fold)
    rl, rr = reconstructPairScalar(rho_i, rho_j, G_i[0], G_j[0], x_ij, taper, limiterType)
    pl, pr = reconstructPairScalar(P_i, P_j, G_i[1], G_j[1], x_ij, taper, limiterType)
    return rl, rr, pl, pr


@wp.func
def pairRiemannStates(
    enabled: wp.bool,
    rho_i: scalar_t, rho_j: scalar_t, P_i: scalar_t, P_j: scalar_t,
    G_i: matrix(shape=(Any, Any), dtype=scalar_t), referenceStateGradients: wp.array(dtype = matrix(shape=(Any, Any), dtype=scalar_t)), # type: ignore
    j: wp.int32,
    x_ij: vector(dtype = scalar_t, length=Any), # type: ignore
    h_i: scalar_t, h_j: scalar_t, kernel_int: wp.int32, dim: wp.int32,
    eta_crit: scalar_t, eta_fold: scalar_t, limiterType: wp.int32,
):
    """`reconstructPairRiemannStates` for pair `(i, j)` when `enabled`, else `(-1, -1, -1, -1)` (not reconstructed: the Pi term
    then uses the particle values). The early return keeps the unconditional single assignment at the call site, which the
    reverse pass handles; the placeholder gradient array is never indexed when not enabled."""
    if not enabled:
        return scalar_t(-1.0), scalar_t(-1.0), scalar_t(-1.0), scalar_t(-1.0)
    return reconstructPairRiemannStates(rho_i, rho_j, P_i, P_j, G_i, referenceStateGradients[j], x_ij, h_i, h_j, kernel_int, dim,
                                        eta_crit, eta_fold, limiterType)
