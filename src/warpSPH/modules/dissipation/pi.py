"""Artificial-viscosity "Pi" term shared by the momentum-viscosity,
thermal-conductivity and thermal-dissipation modules in this package.
`computePi_actual` dispatches on `viscosityParams.viscosityTerm` (or
`.thermalConductivityTerm` when `thermalConductivity=True`) to select one of
several published formulations (Monaghan & Gingold 1983, Cleary 1998,
Monaghan 1992/1997a/1997b, Dukowicz, Price 2008/2012, Wadsley 2008,
DeltaSPH/Default). The returned value is multiplied by `rho_j` (see the
inline note below), and a Monaghan switch zeroes the term for diverging
particle pairs (`ux_ij > 0`) unless `thermalConductivity` is set.
"""

from warpSPHCore import *
import warp as wp
from ...configurations.moduleConfigurations.diffusionParameters import ViscosityTerms, BetaMode, DiffusionParameters
from .util import compute_mu_ij, compute_bars
from warp.types import vector, matrix
from typing import Any

__all__ = ['computePi_actual']

# Note this function returns the term multiplied by rhoj!!
# This is to enable the computation with mj/rhoj as the apparrent volume so the rhoj cancels out for those formulations. This is different from diffSPH which does not multiply by rhoj.
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
    # c_s: scalar_t, # Speed of sound, used in some formulations to compute the signal velocity
    # C_l: scalar_t, C_q: scalar_t, # Viscosity coefficients also referred to as alpha and beta in some formulations
    # K_: scalar_t, # Overall viscosity scaling factor
    # viscosityTerm: wp.int32, # Viscosity formulation to use, e.g. Monaghan1992, Monaghan1997, Cleary1998 etc.
    # scaleBeta : wp.bool = False, # If true then the quadratic viscosity term is scaled by the linear viscosity term, as suggested in some papers to reduce excessive viscosity in certain scenarios. This is only relevant for formulations that use a quadratic term, such as Monaghan1992 and Monaghan1997.
    # switch : wp.bool = True, # Whether to apply the Monaghan switch that turns off viscosity for diverging particles, i.e. particles that are moving away from each other. This is a common technique to reduce excessive viscosity in expanding flows and is used in many formulations such as Monaghan1992 and Monaghan1997.
    # correctXi : wp.bool = False, # Whether to apply the xi correction factor to the viscosity term. This is a correction factor that can be applied to account for errors in the estimation of the velocity divergence and is discussed in some papers such as "Correcting SPH for accurate viscous forces" by Adami et al. 2013.
    useJ : wp.bool = False, # Whether to use the properties of the j particle instead of the i particle in the viscosity computation. This can be relevant for certain formulations and scenarios, such as when computing the viscosity force on particle i due to particle j, it might make sense to use the properties of particle j in the computation. This is also related to the use of rho_bar, c_bar and h_bar which are typically computed as averages of the i and j particle properties.
    thermalConductivity : wp.bool = False, # Whether this viscosity computation is being used for thermal conductivity. This can be relevant for certain formulations that use different coefficients or terms for thermal conductivity compared to momentum viscosity, such as in the case of the Monaghan1997 formulation where the thermal conductivity term has a different form and coefficients compared to the momentum viscosity term.
):
    rho_bar = scalar_t(1.0)/scalar_t(2.0) * (rho_i + rho_j)

    # c_i = viscosityParams.c_s
    # c_j = viscosityParams.c_s
    # c_bar = viscosityParams.c_s
    c_bar = scalar_t(1.0)/scalar_t(2.0) * (c_i + c_j)
    # alpha_i = scalar_t(1.0)
    # alpha_j = scalar_t(1.0) # No viscosity switch here

    h_bar = scalar_t(1.0)/scalar_t(2.0) * (h_i + h_j)


    # The kernel-dependent length factor `sphKernel_xi` (packing ratio x kernel scale) -- NOT
    # Cullen & Dehnen's limiter Xi (`shockCapturing/CullenDehnen2010.py`) nor Wadsley's xi
    # (AV_PLAN Eq. 28).
    kernelXi = sphKernel_xi(kernel_int, domainState.dim)
    if not viscosityParams.correctXi:
        kernelXi = scalar_t(1.0)

    C_l_ = viscosityParams.C_l
    C_q_ = viscosityParams.C_q
    if thermalConductivity:
        C_l_ = viscosityParams.Cu_l
        C_q_ = viscosityParams.Cu_q

    C_l = scalar_t(1.0)/scalar_t(2.0) * (alpha_i + alpha_j) * C_l_
    if viscosityParams.betaMode == wp.static(BetaMode.Fixed.value):
        C_q = C_q_              # beta independent of the switched alpha (PHANTOM / Garcia-Senz 2026 / Sphenix)
    else:
        C_q = scalar_t(1.0)/scalar_t(2.0) * (alpha_i + alpha_j) * C_q_   # Coupled: beta follows alpha
        if viscosityParams.scaleBeta:
            C_q = C_q * C_l     # legacy: beta = alpha_bar^2 C_l C_q

    x_ij = computeDistanceVec(x_i, x_j, domainState)
    r_ij = safe_sqrt(wp.dot(x_ij, x_ij))

    u_ij = v_i - v_j
    ux_ij = wp.dot(u_ij, x_ij)

    viscosityTerm = viscosityParams.viscosityTerm 
    if thermalConductivity:
        viscosityTerm = viscosityParams.thermalConductivityTerm

    mu_ij, scalingFactor = compute_mu_ij(ux_ij, r_ij, h_bar, viscosityTerm, kernelXi)

    # if viscosityParams.monaghanSwitch and ux_ij > 0:
    #     mu_ij = scalar_t(0.0)

    v_sig = scalar_t(scalar_t(0.0))
    K = scalar_t(viscosityParams.K)


    rho, c, h = compute_bars(
        rho_i, rho_j, rho_bar, 
        c_i, c_j, c_bar, 
        h_i, h_j, h_bar, 
        viscosityTerm, useJ)

    if viscosityTerm == wp.static(ViscosityTerms.MonaghanGingold1983.value): # MonaghanGingold1983
        # Monaghan and Gingold 1983: The terms are given in (scalar_t(8.3)) and (scalar_t(8.4))  of Monaghan 2005 and are
        # Pi_ab = -nu ( v_ab \cdot r_ab) / (r_ab^2 + epsilon^2 h_ab^2)
        # nu = alpha h_bar c_bar / rho_bar
        # Rewording this slightly we get the 'Monaghan1992' viscosity Term (with xi correction)
        # combined with using c_bar, rho_bar and h_bar. 
        # Consequently this uses 
        # v_sig = c_bar
        # K = 1
        v_sig = c
        K = scalar_t(1.0)
    elif viscosityTerm == wp.static(ViscosityTerms.Cleary1998.value): # Cleary1998
        # Cleary 1998: The terms are given in (scalar_t(8.8)) and (scalar_t(8.9)) of Monaghan 2005 and are
        # mu_a = 1/8 alpha_a h_a c_a rho_a
        # Pi_ab = - 16 mu_a mu_b / (rho_a rho_b (mu_a + mu_b)) mu_ij
        f = scalar_t(1.0)/(scalar_t(2.0)*(scalar_t(domainState.dim)+scalar_t(2.0))) # Based on estimations based on Monaghan 2005, not given for 1D
        mu_i = f * alpha_i * C_l * h_i * c_i * rho_i / kernelXi
        mu_j = f * alpha_j * C_l * h_j * c_j * rho_j / kernelXi
        # scalar_t(19.8) based on Cleary and Ha 2002
        v_sig = scalar_t(19.8) * mu_i * mu_j / (rho_i * rho_j * (mu_i + mu_j)) / (r_ij + scalar_t(1e-14) * h)
        K = scalar_t(1.0)
    elif viscosityTerm == wp.static(ViscosityTerms.Monaghan1992.value): # Monaghan1992         
        # Monaghan 1992: The term is given in (scalar_t(8.10)) of Monaghan 2005 and is
        # mu = h / rho ( alpha c - beta mu_ij)
        # This uses the Monaghan 1992 viscosity term with alpha = 1 and beta = 2
        v_sig = C_l * c  - C_q * mu_ij
        K = scalar_t(1.0)
    elif viscosityTerm == wp.static(ViscosityTerms.Monaghan1997a.value): # Monaghan1997a
        # Monaghan 1997: The term is given in (scalar_t(8.11)) of Monaghan 2005 and is very similar
        # to the Monaghan1992 term but uses the Monaghan1997 viscosity term. denoted as j
        # in the 1997 paper and has a strange wording in 2005 of using scalar_t(1.0)/scalar_t(2.0) instead of 1 for K
        # c_i + c_j instead of c_bar and beta = 4. Cancelling these terms out gives the normal
        # c_bar term with alpha = 1 and beta = 4! This is also eq scalar_t(3.7) in Monaghan1997
        v_sig = C_l * c - C_q * mu_ij
        K = scalar_t(1.0)
    elif viscosityTerm == wp.static(ViscosityTerms.Monaghan1997b.value): # Monaghan1997b
        # Note that the C_q here is not the usual quadratic coefficient. For lim C_q -> 0 the term collapses to c_i + c_j, i.e., 2 c_bar. This is equivalent to C_l = 2 and C_q = 1 in the 1997a formulation. 
        # For this formulation as C_q increases the viscosity increases, however, for large C_q this term becomes quickly unstable. In the paper the statement is 'where beta [this is our C_q] is a parameter that could be determined by numerical experiments' with no specific suggestion of value.
        # Based on Monaghan 1997 eq scalar_t(4.7):
        v_sig = safe_sqrt(c_i*c_i + C_q * mu_ij*mu_ij) + safe_sqrt(c_j*c_j + C_q * mu_ij*mu_ij) - mu_ij
        K = scalar_t(1.0)
    elif viscosityTerm == wp.static(ViscosityTerms.Dukowicz.value): # Dukowicz
        # The term is given in (scalar_t(4.8)) of Monaghan 1997 and is simply the 1997a term with a 3/4 factor
        v_sig = C_l * c - scalar_t(3.0)/scalar_t(4.0) * C_q * mu_ij
        K = scalar_t(1.0)
    elif viscosityTerm == wp.static(ViscosityTerms.Price2012_98.value): # Price2012_98
        # This term is identical to Monaghan 1992, equation 98 in Price 2012
        v_sig = C_l * c - C_q * mu_ij
        K = scalar_t(1.0)
    elif viscosityTerm == wp.static(ViscosityTerms.Price2012.value): # Price2012
        # Based on equation 103
        v_sig = C_l * c - C_q / scalar_t(2.0) * mu_ij
        K = scalar_t(1.0)
    elif viscosityTerm == wp.static(ViscosityTerms.Price2008.value): # Price2008
        # This formulation and the next are only mentioned in the Price 2012 after equation 103, no explicit equation numbers
        # P_i = queryPressures[i]
        # P_j = referencePressures[j]
        # rho_bar = (rho_i + rho_j) / 2
        # v_sig = C_l * safe_sqrt(wp.abs(P_i - P_j) / (rho_bar + scalar_t(1e-14) * h))
        # K = scalar_t(1.0)
        # Since we don't have access to the pressures here, we can use an approximation based on the ideal gas law, P = rho * c^2
        P_i_ = rho_i * c_i * c_i
        P_j_ = rho_j * c_j * c_j
        if explicitPressure:
            P_i_ = P_i
            P_j_ = P_j
        rho_bar = (rho_i + rho_j) / scalar_t(2.0)
        v_sig = C_l * safe_sqrt(wp.abs(P_i_ - P_j_) / (rho_bar + scalar_t(1e-14) * h))
        K = scalar_t(1.0)
    elif viscosityTerm == wp.static(ViscosityTerms.Wadsley2008.value): # Wadsley2008
        v_sig = C_l * wp.abs(mu_ij)
        K = scalar_t(1.0)
    elif viscosityTerm == wp.static(ViscosityTerms.DeltaSPH.value) or viscosityTerm == wp.static(ViscosityTerms.Default.value): # DeltaSPH / Default
        v_sig = C_l * c - C_q * mu_ij
        K = scalar_t(1.0)


    val = rho_j * K / rho * v_sig * scalingFactor #* mu_ij

    if viscosityParams.monaghanSwitch and not thermalConductivity:
        if ux_ij > 0:
            val = scalar_t(0.0)

    return val
