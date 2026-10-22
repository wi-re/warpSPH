"""Meshless HLLC flux and its original robustness retry chain (Hopkins 2015).

Ported verbatim from the core reference; canonical pair orientation is set by flux.py.
"""
import warp as wp
from typing import Any
from warpSPHCore.type_config import scalar_t
from .solvers import hllcContact as _starFromSpeeds

_TINY = 1.e-30

@wp.func
def _speedsGaburov(uL: scalar_t, uR: scalar_t, aL: scalar_t, aR: scalar_t):
    # GIZMO's primary estimator (reimann.h): extremal wave speeds
    return wp.min(uL, uR) - wp.max(aL, aR), wp.max(uL, uR) + wp.max(aL, aR)


@wp.func
def _speedsRoe(rhoL: scalar_t, uL: scalar_t, vtL: Any, PL: scalar_t, aL: scalar_t,
               rhoR: scalar_t, uR: scalar_t, vtR: Any, PR: scalar_t, aR: scalar_t, gamma: scalar_t):
    R = wp.sqrt(rhoR / rhoL)
    EL = PL / (gamma - scalar_t(1.0)) + scalar_t(0.5) * rhoL * (uL * uL + wp.dot(vtL, vtL))
    ER = PR / (gamma - scalar_t(1.0)) + scalar_t(0.5) * rhoR * (uR * uR + wp.dot(vtR, vtR))
    HL = (EL + PL) / rhoL
    HR = (ER + PR) / rhoR
    ut = (uL + R * uR) / (scalar_t(1.0) + R)
    vtt = (vtL + R * vtR) / (scalar_t(1.0) + R)
    H = (HL + R * HR) / (scalar_t(1.0) + R)
    a2 = (gamma - scalar_t(1.0)) * (H - scalar_t(0.5) * (ut * ut + wp.dot(vtt, vtt)))
    at = wp.sqrt(wp.max(a2, scalar_t(0.0)))
    return wp.min(uL - aL, ut - at), wp.max(uR + aR, ut + at)


@wp.func
def _speedsPvrs(rhoL: scalar_t, uL: scalar_t, PL: scalar_t, aL: scalar_t,
                rhoR: scalar_t, uR: scalar_t, PR: scalar_t, aR: scalar_t, gamma: scalar_t):
    rbar = scalar_t(0.5) * (rhoL + rhoR)
    abar = scalar_t(0.5) * (aL + aR)
    pstar = wp.max(scalar_t(0.5) * (PL + PR) - scalar_t(0.5) * (uR - uL) * rbar * abar, scalar_t(0.0))
    qL = scalar_t(1.0)
    qR = scalar_t(1.0)
    if pstar > PL:
        qL = wp.sqrt(scalar_t(1.0) + (gamma + scalar_t(1.0)) / (scalar_t(2.0) * gamma) * (pstar / PL - scalar_t(1.0)))
    if pstar > PR:
        qR = wp.sqrt(scalar_t(1.0) + (gamma + scalar_t(1.0)) / (scalar_t(2.0) * gamma) * (pstar / PR - scalar_t(1.0)))
    return uL - aL * qL, uR + aR * qR


@wp.func
def _okStar(Ss: scalar_t, Ps: scalar_t):
    return Ps > scalar_t(0.0) and wp.isfinite(Ps) and wp.isfinite(Ss)


@wp.func
def hllcStar(rhoL: scalar_t, uL: scalar_t, vtL: Any, PL: scalar_t,
             rhoR: scalar_t, uR: scalar_t, vtR: Any, PR: scalar_t, gamma: scalar_t):
    """``(S_L, S_R, S*, P*)`` with the Gaburov -> Roe -> PVRS -> Rusanov fallback chain (GIZMO)."""
    aL = wp.sqrt(gamma * PL / rhoL)
    aR = wp.sqrt(gamma * PR / rhoR)
    SL, SR = _speedsGaburov(uL, uR, aL, aR)
    Ss, Ps = _starFromSpeeds(SL, SR, rhoL, uL, PL, rhoR, uR, PR)
    if not _okStar(Ss, Ps):
        SL, SR = _speedsRoe(rhoL, uL, vtL, PL, aL, rhoR, uR, vtR, PR, aR, gamma)
        Ss, Ps = _starFromSpeeds(SL, SR, rhoL, uL, PL, rhoR, uR, PR)
        if not _okStar(Ss, Ps):
            SL, SR = _speedsPvrs(rhoL, uL, PL, aL, rhoR, uR, PR, aR, gamma)
            Ss, Ps = _starFromSpeeds(SL, SR, rhoL, uL, PL, rhoR, uR, PR)
            if not _okStar(Ss, Ps):
                s = wp.max(wp.abs(uL) + aL, wp.abs(uR) + aR)
                SL = -s
                SR = s
                Ss, Ps = _starFromSpeeds(SL, SR, rhoL, uL, PL, rhoR, uR, PR)
    return SL, SR, Ss, Ps


@wp.func
def _hllcSide(rho: scalar_t, u: scalar_t, vt: Any, P: scalar_t, S: scalar_t, Ss: scalar_t, gamma: scalar_t, star: wp.bool):
    """Mass / normal-momentum / tangential-momentum / energy flux of one side (``star``: its star region)."""
    E = P / (gamma - scalar_t(1.0)) + scalar_t(0.5) * rho * (u * u + wp.dot(vt, vt))
    Fm = rho * u
    Fn = rho * u * u + P
    Ft = Fm * vt
    Fe = (E + P) * u
    if star:
        coef = rho * (S - u) / (S - Ss)
        Us_m = coef
        Us_n = coef * Ss
        Us_t = coef * vt
        Us_e = coef * (E / rho + (Ss - u) * (Ss + P / (rho * (S - u))))
        Fm = Fm + S * (Us_m - rho)
        Fn = Fn + S * (Us_n - rho * u)
        Ft = Ft + S * (Us_t - rho * vt)
        Fe = Fe + S * (Us_e - E)
    return Fm, Fn, Ft, Fe


@wp.func
def faceFluxPair(rhoL: scalar_t, uL: scalar_t, vtL: Any, PL: scalar_t,
                 rhoR: scalar_t, uR: scalar_t, vtR: Any, PR: scalar_t, gamma: scalar_t,
                 mode: wp.int32, normal: Any, vframe: Any):
    """Lab-frame flux per unit face area: ``(mass, momentum vector, energy)``."""
    SL, SR, Ss, Ps = hllcStar(rhoL, uL, vtL, PL, rhoR, uR, vtR, PR, gamma)
    vn = wp.dot(vframe, normal)
    if mode == 0:  # MFM: the face moves with the contact -> no mass flux
        return scalar_t(0.0), Ps * normal, Ps * (Ss + vn), Ss, Ps
    # MFV: full HLLC in the frame of the quadrature point, then de-boost (Eq. A8)
    Fm = scalar_t(0.0)
    Fn = scalar_t(0.0)
    Ft = vtL * scalar_t(0.0)
    Fe = scalar_t(0.0)
    if SL >= scalar_t(0.0):
        Fm, Fn, Ft, Fe = _hllcSide(rhoL, uL, vtL, PL, SL, Ss, gamma, False)
    elif Ss >= scalar_t(0.0):
        Fm, Fn, Ft, Fe = _hllcSide(rhoL, uL, vtL, PL, SL, Ss, gamma, True)
    elif SR > scalar_t(0.0):
        Fm, Fn, Ft, Fe = _hllcSide(rhoR, uR, vtR, PR, SR, Ss, gamma, True)
    else:
        Fm, Fn, Ft, Fe = _hllcSide(rhoR, uR, vtR, PR, SR, Ss, gamma, False)
    mom = Fn * normal + Ft
    momLab = mom + Fm * vframe
    eLab = Fe + wp.dot(vframe, mom) + scalar_t(0.5) * wp.dot(vframe, vframe) * Fm
    return Fm, momLab, eLab, Ss, Ps

