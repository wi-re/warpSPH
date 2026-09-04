"""Weakly-compressible pressure closures (Tait-family and related) and the
`weaklyCompressibleEOS` dispatcher that reads `schemeConfig.fluid.eosType`
to pick one.

All closures return zero pressure at `rho == rho0`. `weaklyCompressibleEOS`
reads density straight off `particleState.densities`, unclamped, on purpose:
a commented-out `torch.clamp(..., min=0.8)` above it is a deliberately
disabled guard, per the user (2026-08-15) — non-physical densities are meant
to blow up rather than be silently masked, so they get caught instead of
hidden.
"""

from ...enumTypes import *
from torch.profiler import profile, record_function, ProfilerActivity
import torch

__all__ = ['weaklyCompressibleEOS']

def stiffTaitEOS(rho, rho0: float, c_s: float, polytropicExponent: float):

    return rho0 * c_s**2 / polytropicExponent * ((rho / rho0)**polytropicExponent - 1)

def TaitEOS(rho, rho0: float, kappa: float):
    return kappa * (rho - rho0)

def isoThermalEOS(rho, rho0: float, c_s: float):
    return c_s**2 * (rho - rho0)

def polytropicEOS(rho, polytropicExponent : float, kappa : float):
    return kappa * (rho)**polytropicExponent

def murnaghanEOS(rho, rho0: float, kappa: float, exponent: float):
    return kappa / exponent * ((rho / rho0)**exponent - 1)


def weaklyCompressibleEOS(particleState, schemeConfig):
    with record_function("[warpSPH] - weaklyCompressibleEOS"):
        rho0 = schemeConfig.fluid.restDensity
        c_s = schemeConfig.fluid.fixedSoundSpeed
        
        eosType = schemeConfig.fluid.eosType
        kappa = schemeConfig.fluid.kappa
        polytropicExponent = schemeConfig.fluid.polytropicExponent

        # rho = torch.clamp(particleState.densities, min=0.8)  # rough attempt at clamping against instabilities; left disabled on purpose -- non-physical densities should blow up rather than be silently masked, so they get caught instead. This is where you'd clamp it if that's ever actually wanted.
        rho = particleState.densities

        # Uniform background pressure (docs/historic_plans/WCSPH_SHIFTING_PLAN.md 2a): p <- p_EOS + p_b.
        # Default 0.0, so this is inert unless a case sets it.
        p_b = getattr(schemeConfig.fluid, 'backgroundPressure', 0.0)

        if eosType == EquationOfState.stiffTait:
            return stiffTaitEOS(rho, rho0, c_s, polytropicExponent) + p_b
        elif eosType == EquationOfState.Tait:
            return TaitEOS(rho, rho0, kappa) + p_b
        elif eosType == EquationOfState.isoThermal:
            return isoThermalEOS(rho, rho0, c_s) + p_b
        elif eosType == EquationOfState.Polytropic:
            return polytropicEOS(rho, polytropicExponent, kappa) + p_b
        elif eosType == EquationOfState.Murnaghan:
            return murnaghanEOS(rho, rho0, kappa, polytropicExponent) + p_b
        else:
            raise ValueError('EOS type not recognized')
