"""`PairData`: everything a pairwise dissipation formulation needs about one (i, j) pair, computed once.

Each formulation in `terms.py` is a small function of a `PairData`; the means ("bars") are formed here, and
`pick` selects the bar or the one-sided value a formulation asks for -- there is no second place that has to
be told which bars were already computed."""

from warpSPHCore import *
import warp as wp
from warp.types import vector
from typing import Any
from ....configurations.moduleConfigurations.diffusionParameters import DiffusionParameters
from .coefficients import switchedCoefficients

__all__ = ['PairData', 'buildPair', 'pick']


@wp.struct
class PairData:
    # geometry and velocity: ux = u_ij . x_ij (negative for an approaching pair), r = |x_ij|
    r: scalar_t
    ux: scalar_t
    # the two particles, their pair means, and the dimension
    h_i: scalar_t
    h_j: scalar_t
    h_bar: scalar_t
    rho_i: scalar_t
    rho_j: scalar_t
    rho_bar: scalar_t
    c_i: scalar_t
    c_j: scalar_t
    c_bar: scalar_t
    alpha_i: scalar_t
    alpha_j: scalar_t
    P_i: scalar_t
    P_j: scalar_t
    explicitPressure: wp.bool
    dim: wp.int32
    # coefficients: pair-mean switched (alpha, beta) and the base linear coefficient
    C_l: scalar_t
    C_q: scalar_t
    C_l_: scalar_t
    C_q_: scalar_t
    # the kernel-dependent length factor `sphKernel_xi` (packing ratio x kernel scale) -- NOT Cullen & Dehnen's
    # limiter Xi nor Wadsley's xi (AV_PLAN Eq. 28); 1 unless `correctXi`
    kernelXi: scalar_t
    # support radius / smoothing length (`sphKernelScale`): the smoothing length of a particle is `h / kernelScale`
    kernelScale: scalar_t
    # density and pressure of the two particles reconstructed to the pair midpoint (`Riemann` term, GODUNOV_SPH_PLAN); negative = not
    # reconstructed (the particle's own value)
    rhoRec_i: scalar_t
    rhoRec_j: scalar_t
    PRec_i: scalar_t
    PRec_j: scalar_t
    # which side of the pair a one-sided formulation evaluates
    useJ: wp.bool


@wp.func
def pick(q_i: scalar_t, q_j: scalar_t, q_bar: scalar_t, useBar: wp.bool, useJ: wp.bool):
    """The pair mean, or the value of particle i (or j when `useJ`)."""
    out = q_i
    if useBar:
        out = q_bar
    else:
        if useJ:
            out = q_j
    return out


@wp.func
def buildPair(
    x_i: vector(dtype = scalar_t, length=Any), x_j:  vector(dtype = scalar_t, length=Any), # type: ignore
    h_i: scalar_t, h_j: scalar_t, # type: ignore
    rho_i: scalar_t, rho_j: scalar_t, # type: ignore
    explicitPressure: wp.bool, P_i: scalar_t, P_j: scalar_t, # type: ignore
    u_ij: vector(dtype = scalar_t, length=Any), # the pair velocity difference v_i - v_j # type: ignore
    domainState: domainData,
    kernel_int: wp.int32,
    c_i: scalar_t, c_j: scalar_t,
    alpha_i: scalar_t, alpha_j: scalar_t,
    viscosityParams: DiffusionParameters,
    useJ: wp.bool,
    thermalConductivity: wp.bool,
    rhoRec_i: scalar_t = scalar_t(-1.0), rhoRec_j: scalar_t = scalar_t(-1.0),
    PRec_i: scalar_t = scalar_t(-1.0), PRec_j: scalar_t = scalar_t(-1.0),
):
    pair = PairData()
    pair.rhoRec_i = rhoRec_i
    pair.rhoRec_j = rhoRec_j
    pair.PRec_i = PRec_i
    pair.PRec_j = PRec_j
    pair.rho_i = rho_i
    pair.rho_j = rho_j
    pair.rho_bar = scalar_t(1.0)/scalar_t(2.0) * (rho_i + rho_j)
    pair.c_i = c_i
    pair.c_j = c_j
    pair.c_bar = scalar_t(1.0)/scalar_t(2.0) * (c_i + c_j)
    pair.h_i = h_i
    pair.h_j = h_j
    pair.h_bar = scalar_t(1.0)/scalar_t(2.0) * (h_i + h_j)
    pair.alpha_i = alpha_i
    pair.alpha_j = alpha_j
    pair.P_i = P_i
    pair.P_j = P_j
    pair.explicitPressure = explicitPressure
    pair.dim = domainState.dim
    pair.useJ = useJ

    pair.kernelScale = sphKernelScale(kernel_int, domainState.dim)
    pair.kernelXi = sphKernel_xi(kernel_int, domainState.dim)
    if not viscosityParams.correctXi:
        pair.kernelXi = scalar_t(1.0)

    pair.C_l_ = viscosityParams.C_l
    pair.C_q_ = viscosityParams.C_q
    if thermalConductivity:
        pair.C_l_ = viscosityParams.Cu_l
        pair.C_q_ = viscosityParams.Cu_q
    C_l, C_q = switchedCoefficients(scalar_t(1.0)/scalar_t(2.0) * (alpha_i + alpha_j), pair.C_l_, pair.C_q_, viscosityParams)
    pair.C_l = C_l
    pair.C_q = C_q

    x_ij = computeDistanceVec(x_i, x_j, domainState)
    pair.r = safe_sqrt(wp.dot(x_ij, x_ij))
    pair.ux = wp.dot(u_ij, x_ij)
    return pair
