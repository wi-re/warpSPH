"""`computePi_pair` / `computePi_actual`: the entry points of the pairwise dissipation term.

Note these return the term multiplied by rho_j (see `terms.py`) -- different from diffSPH, which does not."""

from warpSPHCore import *
import warp as wp
from warp.types import vector
from typing import Any
from ....configurations.moduleConfigurations.diffusionParameters import DiffusionParameters
from ...reconstruction import rawPairVelocity
from .pair import buildPair
from .terms import evaluateTerm

__all__ = ['computePi_term', 'computePi_pair', 'computePi_actual']


@wp.func
def computePi_term(
    viscosityTerm: wp.int32,                # which formulation, as a `ViscosityTerms` value (callers may fix it statically)
    x_i: vector(dtype = scalar_t, length=Any), x_j:  vector(dtype = scalar_t, length=Any), # type: ignore
    h_i: scalar_t, h_j: scalar_t, # type: ignore
    m_i: scalar_t, m_j: scalar_t, # type: ignore
    rho_i: scalar_t, rho_j: scalar_t, # type: ignore
    explicitPressure: wp.bool, P_i: scalar_t, P_j: scalar_t, # type: ignore
    u_ij: vector(dtype = scalar_t, length=Any), # the pair velocity difference v_i - v_j (raw, or reconstructed to the pair midpoint) # type: ignore

    domainState: domainData,
    kernel_int : wp.int32,
    c_i: scalar_t, c_j: scalar_t,
    alpha_i: scalar_t, alpha_j: scalar_t,

    viscosityParams: DiffusionParameters,
    useJ : wp.bool = False,                 # evaluate a one-sided formulation on particle j's values instead of i's
    thermalConductivity : wp.bool = False,  # the conductivity coefficients (`Cu_l`, `Cu_q`); no Monaghan switch
    rhoRec_i: scalar_t = scalar_t(-1.0), rhoRec_j: scalar_t = scalar_t(-1.0), # density / pressure at the pair midpoint (`Riemann` term); negative: not reconstructed
    PRec_i: scalar_t = scalar_t(-1.0), PRec_j: scalar_t = scalar_t(-1.0),
):
    pair = buildPair(
        x_i, x_j, h_i, h_j, rho_i, rho_j, explicitPressure, P_i, P_j, u_ij,
        domainState, kernel_int, c_i, c_j, alpha_i, alpha_j, viscosityParams, useJ, thermalConductivity,
        rhoRec_i, rhoRec_j, PRec_i, PRec_j)

    val = evaluateTerm(viscosityTerm, pair, viscosityParams)

    # Monaghan's switch: no viscosity for receding pairs (not applied to conduction)
    if viscosityParams.monaghanSwitch and not thermalConductivity:
        if pair.ux > 0:
            val = scalar_t(0.0)
    return val


@wp.func
def computePi_pair(
    x_i: vector(dtype = scalar_t, length=Any), x_j:  vector(dtype = scalar_t, length=Any), # type: ignore
    h_i: scalar_t, h_j: scalar_t, # type: ignore
    m_i: scalar_t, m_j: scalar_t, # type: ignore
    rho_i: scalar_t, rho_j: scalar_t, # type: ignore
    explicitPressure: wp.bool, P_i: scalar_t, P_j: scalar_t, # type: ignore
    u_ij: vector(dtype = scalar_t, length=Any), # the pair velocity difference v_i - v_j (raw, or reconstructed to the pair midpoint) # type: ignore

    domainState: domainData,
    kernel_int : wp.int32,
    c_i: scalar_t, c_j: scalar_t,
    alpha_i: scalar_t, alpha_j: scalar_t,

    viscosityParams: DiffusionParameters,
    useJ : wp.bool = False,
    thermalConductivity : wp.bool = False,
    rhoRec_i: scalar_t = scalar_t(-1.0), rhoRec_j: scalar_t = scalar_t(-1.0),
    PRec_i: scalar_t = scalar_t(-1.0), PRec_j: scalar_t = scalar_t(-1.0),
):
    """`computePi_term` for the formulation selected in `viscosityParams` (`viscosityTerm`, or `thermalConductivityTerm`)."""
    viscosityTerm = viscosityParams.viscosityTerm
    if thermalConductivity:
        viscosityTerm = viscosityParams.thermalConductivityTerm
    return computePi_term(
        viscosityTerm, x_i, x_j, h_i, h_j, m_i, m_j, rho_i, rho_j,
        explicitPressure, P_i, P_j, u_ij, domainState, kernel_int, c_i, c_j, alpha_i, alpha_j,
        viscosityParams, useJ, thermalConductivity, rhoRec_i, rhoRec_j, PRec_i, PRec_j)


@wp.func
def computePi_actual(
    x_i: vector(dtype = scalar_t, length=Any), x_j:  vector(dtype = scalar_t, length=Any), # type: ignore
    h_i: scalar_t, h_j: scalar_t, # type: ignore
    m_i: scalar_t, m_j: scalar_t, # type: ignore
    rho_i: scalar_t, rho_j: scalar_t, # type: ignore
    explicitPressure: wp.bool, P_i: scalar_t, P_j: scalar_t, # type: ignore
    v_i: vector(dtype = scalar_t, length=Any), v_j: vector(dtype = scalar_t, length=Any), # type: ignore
    domainState: domainData,
    kernel_int : wp.int32,
    c_i: scalar_t, c_j: scalar_t,
    alpha_i: scalar_t, alpha_j: scalar_t,
    viscosityParams: DiffusionParameters,
    useJ : wp.bool = False,
    thermalConductivity : wp.bool = False,
):
    """`computePi_pair` on the raw velocity difference `v_i - v_j` (the `RawVelocity` pair policy, AV_PLAN Phase 1)."""
    return computePi_pair(
        x_i, x_j, h_i, h_j, m_i, m_j, rho_i, rho_j,
        explicitPressure, P_i, P_j,
        rawPairVelocity(v_i, v_j),
        domainState, kernel_int, c_i, c_j, alpha_i, alpha_j,
        viscosityParams, useJ, thermalConductivity)
