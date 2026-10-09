"""One-sided viscous pressures built on the pairwise term: CRKSPH's `Q_i` / `Q_j`."""

from warpSPHCore import *
import warp as wp
from warp.types import vector
from typing import Any
from ....configurations.moduleConfigurations.diffusionParameters import DiffusionParameters, ViscosityTerms
from .dispatch import computePi_term

__all__ = ['computeFrontiereQ']


@wp.func
def computeFrontiereQ(
    x_i: vector(dtype = scalar_t, length=Any), x_j:  vector(dtype = scalar_t, length=Any), # type: ignore
    h_i: scalar_t, h_j: scalar_t, # type: ignore
    rho_i: scalar_t, rho_j: scalar_t, # type: ignore
    u_ij: vector(dtype = scalar_t, length=Any), # the pair velocity difference v_i - v_j, reconstructed for CRKSPH # type: ignore
    domainState: domainData,
    kernel_int : wp.int32,
    c_i: scalar_t, c_j: scalar_t,
    alpha_i: scalar_t, alpha_j: scalar_t,
    viscosityParams: DiffusionParameters,
    useJ : wp.bool,
):
    """Frontiere et al. (2017) viscous pressure `Q` of particle i (or j with `useJ`):
    `Q_i = rho_i (-C_l c_i mu_i + C_q mu_i^2)`, from the `Frontiere2017` term of `computePi_term` (the formula lives there once).

    Pi's convention is `a_i += (m_j / rho_j) val w grad W` for a pair term `-m_j Pi_ab grad W`, with `w = (u_ij . x_ij) / r`;
    CRKSPH's force is `-(Q_i + Q_j) V_i V_j / m_i grad W`, so `Q_i = -rho_i^2 val w / rho_j`, and `Q_j = -rho_j val w` when
    `useJ` (where `val` carries rho_j / rho_j)."""
    val = computePi_term(
        wp.static(ViscosityTerms.Frontiere2017.value), x_i, x_j, h_i, h_j, scalar_t(1.0), scalar_t(1.0), rho_i, rho_j,
        False, scalar_t(0.0), scalar_t(0.0), u_ij, domainState, kernel_int, c_i, c_j, alpha_i, alpha_j,
        viscosityParams, useJ, False)
    x_ij = computeDistanceVec(x_i, x_j, domainState)
    r_ij = safe_sqrt(wp.dot(x_ij, x_ij))
    w = wp.dot(u_ij, x_ij) / (r_ij + scalar_t(1.0e-14) * h_i)
    rho_side = rho_i
    if useJ:
        rho_side = rho_j
    return -rho_side * rho_side / rho_j * val * w
