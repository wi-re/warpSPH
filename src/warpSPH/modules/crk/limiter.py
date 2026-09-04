"""CRKSPH slope limiters for the pseudo-viscosity term.

`computeVanLeer` estimates a monotonicity ratio from each particle's
reconstructed velocity gradient and applies a van Leer flux limiter; the
zero-denominator branch must return early (before dividing) rather than
patching a NaN afterward, since reverse-mode AD differentiates the
un-guarded division itself (see the inline note referencing
`scripts/gradcheck_crk.py`, docs/historic_plans/CLEANUP_PLAN.md Phase 4.1 Tier 1). `crkLimiter`
tapers the viscosity to zero for particle pairs closer than `eta_crit`
smoothing lengths apart, via a Gaussian fall-off scaled by `eta_fold`.
"""

import warp as wp
from warp.types import vector, matrix
from typing import Any
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from typing import Optional, Union, Tuple
from warpSPHCore import *

__all__ = ['computeVanLeer', 'crkLimiter']

@wp.func
def limiterVL(x: scalar_t):
    if x <= scalar_t(0.0):
        return scalar_t(0.0)
    # x = wp.min(x, scalar_t(1.0e6))
    vL = scalar_t(2.0) / (scalar_t(1.0) + x)
    return x * vL*vL

    # return torch.where(x > scalar_t(0.0), x * vL**scalar_t(2.0), scalar_t(0.0))

@wp.func
def sgn(x: scalar_t):
    if x >= scalar_t(0.0):
        return scalar_t(1.0)
    else:
        return scalar_t(-1.0)

@wp.func
def computeVanLeer(
    xij_  : vector(length=Any, dtype=scalar_t),  # type: ignore
    vel_i : vector(length=Any, dtype=scalar_t),  # type: ignore
    vel_j : vector(length=Any, dtype=scalar_t),  # type: ignore
    DvDxi : matrix(shape=(Any, Any), dtype=scalar_t),  # type: ignore
    DvDxj : matrix(shape=(Any, Any), dtype=scalar_t)   # type: ignore
):
    xij = scalar_t(0.5) * (xij_)
    # velocity difference variant
    # # Gradient correction vectors: linear extrapolation from each particle to the midpoint.
    # # corr_i = DvDxi * 0.5*(xi-xj)  =  velocity change predicted by i's gradient toward midpoint
    # # corr_j = DvDxj * 0.5*(xi-xj)  (same offset, applied from j side)
    # corr_i = matmul(DvDxi, xij)
    # corr_j = matmul(DvDxj, xij)

    # # Actual velocity difference as the reference signal (Spheral GSPH scalar approach,
    # # generalised to vectors by projecting onto the velocity-difference direction).
    # # The quadratic form (DvDx*x).x is blind to shear because it measures only the
    # # normal-strain component along x_ij; projecting onto vdiff_hat captures shear too.
    # vdiff     = vel_i - vel_j
    # vdiff_mag = safe_sqrt(wp.dot(vdiff, vdiff))

    # # If there is no velocity difference the flow is trivially smooth: allow full correction.
    # if vdiff_mag < scalar_t(1.0e-30):
    #     return scalar_t(1.0)

    # vdiff_hat = vdiff / vdiff_mag

    # # Scalar projections along the velocity-difference direction (equiv. to Spheral's Dy0s / Dyis)
    # Dyis = wp.dot(corr_i, vdiff_hat)
    # Dyjs = wp.dot(corr_j, vdiff_hat)

    # # Denominator: half the actual velocity-difference magnitude, same as Spheral's
    # # denom = 2/(sgn(Dy0)*|Dy0|) where Dy0s > 0 by construction here.
    # denom = scalar_t(2.0) / vdiff_mag
    # ri = Dyis * denom
    # rj = Dyjs * denom

    # standard quadratic form approach, blind to shear
    corr_i = matmul(DvDxi, xij)
    corr_j = matmul(DvDxj, xij)

    grad_i = wp.dot(corr_i, xij)
    grad_j = wp.dot(corr_j, xij)

    # A zero denominator here (most commonly the self-interaction pair,
    # xij == 0, so grad_i == grad_j == 0) means there is no flow or the flow
    # is perfectly linear, so the limiter should just be 1 in these cases --
    # same intent as the original NaN-check-after-the-fact version, but the
    # guard now has to happen *before* the division, not after: computing
    # the division unconditionally and only patching the resulting NaN value
    # is forward-safe but not backward-safe. The un-guarded division's local
    # derivative is itself inf/nan at the singular point, and that poisons
    # the adjoint even though the forward value gets overwritten -- reverse
    # AD differentiates the expression that was evaluated, not the value it
    # was replaced with afterward. Confirmed via
    # scripts/gradcheck_crk.py (docs/historic_plans/CLEANUP_PLAN.md Phase 4.1 Tier 1): forward
    # values were finite, but every gradient came back NaN.
    denom_j = sgn(grad_j) * abs(grad_j)
    if wp.abs(denom_j) > scalar_t(1.0e-30):
        ri = grad_i / denom_j
    else:
        ri = scalar_t(1.0)

    denom_i = sgn(grad_i) * abs(grad_i)
    if wp.abs(denom_i) > scalar_t(1.0e-30):
        rj = grad_j / denom_i
    else:
        rj = scalar_t(1.0)


    rij = wp.min(ri, rj)

    phi = limiterVL(rij)
    return phi


@wp.func
def crkLimiter(
    x_ij: vector(length=Any, dtype=scalar_t), # type: ignore
    hi: scalar_t,
    hj: scalar_t,
    kernel_int: wp.int32,
    dim: wp.int32,
    eta_crit: scalar_t,
    eta_fold: scalar_t
):
    w_xi = sphKernel_xi(kernel_int, dim)
    # xi = Kernel_xi(config['kernel'], particles.positions.shape[0])
    # eta_max = getSetConfig(config, 'CRKSPH', 'eta_max', 4.0)
    ks = sphKernelScale(kernel_int, dim)
    eta_max = w_xi
    # eta_max = 1.0


    # The logic here gets a bit messy because of h/H shenanigans in SPH.
    # In our code we define the support of a particle as the cut-off distance, i.e., the distance at which the kernel value goes to zero. 
    # In other codes, such as spheral, they define the cut-off radius as a mulitple of the smoothing scale, i.e., H = ks * h, where ks is the cut-off in terms of smoothing lengths.
    # That means that we are effectively storing H as a property of a particle instead of h.
    # Consequently when the CRK paper refers to eta_ij = r_ij / h, we need to convert this to our definition of support which gives us eta_ij = r_ij / H * ks
    # By scaling eta_crit instead 

    # eta_crit = scalar_t(1.0)/scalar_t(4.0)# * ks #  * eta_max
    # eta_fold = scalar_t(0.2) #/ ks
    eta_i = x_ij/hi#*ks #* eta_max
    eta_j = x_ij/hj#*ks #* eta_max
        
    eta_i_norm = safe_sqrt(wp.dot(eta_i, eta_i))
    eta_j_norm = safe_sqrt(wp.dot(eta_j, eta_j))
    eta_ij = wp.min(eta_i_norm, eta_j_norm)

    factor = scalar_t(1.0)
    if eta_ij < eta_crit:
        factor = wp.exp(- ((eta_ij - eta_crit)/eta_fold)**scalar_t(2.0))
    
    # return scalar_t(1.0)
    return factor