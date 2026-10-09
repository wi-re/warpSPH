"""Approximate Riemann solvers for the 1D Euler equations of an ideal gas (Toro, *Riemann Solvers and Numerical
Methods for Fluid Dynamics*, 3rd ed.: ch. 9 for the star-pressure estimates, ch. 10 for HLLC).

A state is `(rho, u, p)` with `u` the velocity along the face normal pointing from the left to the right state;
`gamma` is the ideal-gas index. Every function floors `rho` and `p` first (vacuum is not solved: the star state of a
near-vacuum pair degrades to the floors, never to a NaN, which is also what keeps reverse-mode AD finite), and divides
only after the guard (AV_PLAN §2.6).

`RiemannSolver` (defined beside the other dissipation enums in `configurations/moduleConfigurations/diffusionParameters.py`)
selects the estimate of the star state `(p*, u*)`:

* `Acoustic`  -- the linearised problem with the impedances `Z_K = rho_K a_K` (Godunov-SPH, Inutsuka 2002; Monaghan 1997
                 is this with a pair-mean impedance). Exact for weak waves, the Monaghan-1997a linear viscosity.
* `PVRS`      -- Toro Eq. (9.20): the acoustic solution with the mean impedance `rho_bar a_bar`.
* `TRRS`      -- Toro Eqs. (9.31)-(9.32), two rarefactions: exact for two rarefaction waves, good for expansions.
* `TSRS`      -- Toro Eqs. (9.41)-(9.43), two shocks, with the PVRS pressure as the shock-speed estimate: good for
                 compressions (the shock viscosity quadratic term `beta` comes from it).
* `Adaptive`  -- Toro §9.5.2: PVRS where the pressure ratio is < 2 and PVRS is in range, otherwise TRRS (expansion)
                 or TSRS (compression).
* `HLLC`      -- the star pressure / speed of the HLLC wave-speed model (Toro Eqs. 10.37, 10.59-10.61).
"""

import warp as wp
from warpSPHCore import *
from ...configurations.moduleConfigurations.diffusionParameters import RiemannSolver

__all__ = ['RiemannSolver', 'riemannStarState', 'hllcFlux']


# smallest density / pressure a state may have in the solver (relative scale is the caller's: these only stop a
# division by, or the square root of, zero)
_FLOOR = 1.0e-30


@wp.func
def _soundSpeed(rho: scalar_t, p: scalar_t, gamma: scalar_t):
    return wp.sqrt(gamma * p / rho)


@wp.func
def _acousticStar(rhoL: scalar_t, uL: scalar_t, pL: scalar_t, aL: scalar_t,
                  rhoR: scalar_t, uR: scalar_t, pR: scalar_t, aR: scalar_t, ZL: scalar_t, ZR: scalar_t):
    """The linearised star state for the impedances ZL, ZR."""
    Zs = ZL + ZR
    pStar = (ZR * pL + ZL * pR + ZL * ZR * (uL - uR)) / Zs
    uStar = (ZL * uL + ZR * uR + pL - pR) / Zs
    return pStar, uStar


@wp.func
def _pvrsPressure(rhoL: scalar_t, uL: scalar_t, pL: scalar_t, aL: scalar_t,
                  rhoR: scalar_t, uR: scalar_t, pR: scalar_t, aR: scalar_t):
    """Toro Eq. (9.20), the primitive-variable estimate of the star pressure (unfloored)."""
    return scalar_t(0.5) * (pL + pR) - scalar_t(0.125) * (uR - uL) * (rhoL + rhoR) * (aL + aR)


@wp.func
def _trrsStar(rhoL: scalar_t, uL: scalar_t, pL: scalar_t, aL: scalar_t,
              rhoR: scalar_t, uR: scalar_t, pR: scalar_t, aR: scalar_t, gamma: scalar_t):
    """Two-rarefaction star state, Toro Eqs. (9.31)-(9.32)."""
    z = (gamma - scalar_t(1.0)) / (scalar_t(2.0) * gamma)
    pLR = wp.pow(pL / pR, z)
    num = aL + aR - scalar_t(0.5) * (gamma - scalar_t(1.0)) * (uR - uL)
    den = aL / wp.pow(pL, z) + aR / wp.pow(pR, z)
    pStar = wp.pow(wp.max(num, scalar_t(_FLOOR)) / den, scalar_t(1.0) / z)
    uStar = (pLR * uL / aL + uR / aR + scalar_t(2.0) * (pLR - scalar_t(1.0)) / (gamma - scalar_t(1.0))) / (pLR / aL + scalar_t(1.0) / aR)
    return pStar, uStar


@wp.func
def _tsrsStar(rhoL: scalar_t, uL: scalar_t, pL: scalar_t, aL: scalar_t,
              rhoR: scalar_t, uR: scalar_t, pR: scalar_t, aR: scalar_t, gamma: scalar_t, p0: scalar_t):
    """Two-shock star state, Toro Eqs. (9.41)-(9.43), `p0` the shock-speed pressure estimate (>= 0)."""
    gp1 = gamma + scalar_t(1.0)
    gm1 = gamma - scalar_t(1.0)
    # the ratios overflow float32 for a (floored) vacuum state; cap them, the star pressure then tends to the other side's
    gL = wp.sqrt(wp.min(scalar_t(2.0) / (gp1 * rhoL) / (p0 + gm1 / gp1 * pL), scalar_t(1.0e30)))
    gR = wp.sqrt(wp.min(scalar_t(2.0) / (gp1 * rhoR) / (p0 + gm1 / gp1 * pR), scalar_t(1.0e30)))
    pStar = (gL * pL + gR * pR - (uR - uL)) / (gL + gR)
    uStar = scalar_t(0.5) * (uL + uR) + scalar_t(0.5) * ((pStar - pR) * gR - (pStar - pL) * gL)
    return pStar, uStar


@wp.func
def _hllcWaveSpeeds(rhoL: scalar_t, uL: scalar_t, pL: scalar_t, aL: scalar_t,
                    rhoR: scalar_t, uR: scalar_t, pR: scalar_t, aR: scalar_t, gamma: scalar_t):
    """Toro Eqs. (10.59)-(10.61) with the PVRS pressure: (S_L, S_R, S*), S* by Eq. (10.37)."""
    pStarEst = wp.max(_pvrsPressure(rhoL, uL, pL, aL, rhoR, uR, pR, aR), scalar_t(0.0))
    k = (gamma + scalar_t(1.0)) / (scalar_t(2.0) * gamma)
    qL = scalar_t(1.0)
    if pStarEst > pL:
        qL = wp.sqrt(scalar_t(1.0) + k * (pStarEst / pL - scalar_t(1.0)))
    qR = scalar_t(1.0)
    if pStarEst > pR:
        qR = wp.sqrt(scalar_t(1.0) + k * (pStarEst / pR - scalar_t(1.0)))
    SL = uL - aL * qL
    SR = uR + aR * qR
    num = pR - pL + rhoL * uL * (SL - uL) - rhoR * uR * (SR - uR)
    den = rhoL * (SL - uL) - rhoR * (SR - uR)      # < 0: SL < uL and SR > uR
    Sstar = num / den
    return SL, SR, Sstar


@wp.func
def riemannStarState(
    solver: wp.int32,
    rhoL: scalar_t, uL: scalar_t, pL: scalar_t,
    rhoR: scalar_t, uR: scalar_t, pR: scalar_t,
    gamma: scalar_t,
):
    """`(p*, u*)`: the pressure and normal velocity of the contact for the left / right state, `solver` a `RiemannSolver`
    value. `u` is along the left -> right normal. For `L = R` every solver returns `(p, u)`; swapping the states and
    negating the velocities swaps the star state's `u*` sign and leaves `p*` (the solvers are mirror-symmetric)."""
    rhoL = wp.max(rhoL, scalar_t(_FLOOR))
    rhoR = wp.max(rhoR, scalar_t(_FLOOR))
    pL = wp.max(pL, scalar_t(_FLOOR))
    pR = wp.max(pR, scalar_t(_FLOOR))
    aL = _soundSpeed(rhoL, pL, gamma)
    aR = _soundSpeed(rhoR, pR, gamma)

    pStar = scalar_t(0.0)
    uStar = scalar_t(0.0)
    if solver == wp.static(RiemannSolver.Acoustic.value):
        pStar, uStar = _acousticStar(rhoL, uL, pL, aL, rhoR, uR, pR, aR, rhoL * aL, rhoR * aR)
    elif solver == wp.static(RiemannSolver.PVRS.value):
        Zbar = scalar_t(0.25) * (rhoL + rhoR) * (aL + aR)
        pStar, uStar = _acousticStar(rhoL, uL, pL, aL, rhoR, uR, pR, aR, Zbar, Zbar)
    elif solver == wp.static(RiemannSolver.TRRS.value):
        pStar, uStar = _trrsStar(rhoL, uL, pL, aL, rhoR, uR, pR, aR, gamma)
    elif solver == wp.static(RiemannSolver.TSRS.value):
        p0 = wp.max(_pvrsPressure(rhoL, uL, pL, aL, rhoR, uR, pR, aR), scalar_t(0.0))
        pStar, uStar = _tsrsStar(rhoL, uL, pL, aL, rhoR, uR, pR, aR, gamma, p0)
    elif solver == wp.static(RiemannSolver.Adaptive.value):
        pPV = _pvrsPressure(rhoL, uL, pL, aL, rhoR, uR, pR, aR)
        pMin = wp.min(pL, pR)
        pMax = wp.max(pL, pR)
        pPVf = wp.max(pPV, scalar_t(0.0))
        if pMax / pMin <= scalar_t(2.0) and pPV >= pMin and pPV <= pMax:
            Zbar = scalar_t(0.25) * (rhoL + rhoR) * (aL + aR)
            pStar, uStar = _acousticStar(rhoL, uL, pL, aL, rhoR, uR, pR, aR, Zbar, Zbar)
        elif pPV < pMin:
            pStar, uStar = _trrsStar(rhoL, uL, pL, aL, rhoR, uR, pR, aR, gamma)
        else:
            pStar, uStar = _tsrsStar(rhoL, uL, pL, aL, rhoR, uR, pR, aR, gamma, pPVf)
    else:
        SL, SR, Sstar = _hllcWaveSpeeds(rhoL, uL, pL, aL, rhoR, uR, pR, aR, gamma)
        pStar = pL + rhoL * (SL - uL) * (Sstar - uL)
        uStar = Sstar
    # a rarefaction strong enough to open a vacuum has p* = 0, not the (negative) value the linearised / shock formulas give
    return wp.max(pStar, scalar_t(0.0)), uStar


@wp.func
def hllcFlux(
    rhoL: scalar_t, uL: scalar_t, pL: scalar_t,
    rhoR: scalar_t, uR: scalar_t, pR: scalar_t,
    gamma: scalar_t,
):
    """The HLLC flux through a face with the left -> right normal (Toro §10.4): `(F_rho, F_momentum, F_energy, S*)`,
    the fluxes of mass, normal momentum and total energy `E = p / (gamma - 1) + rho u^2 / 2` per unit area. Tangential
    momentum is passive: its flux is `F_rho` times the tangential velocity of the upwind side (`F_rho >= 0` takes the
    left state), which the caller applies. `S*` is the contact speed."""
    rhoL = wp.max(rhoL, scalar_t(_FLOOR))
    rhoR = wp.max(rhoR, scalar_t(_FLOOR))
    pL = wp.max(pL, scalar_t(_FLOOR))
    pR = wp.max(pR, scalar_t(_FLOOR))
    aL = _soundSpeed(rhoL, pL, gamma)
    aR = _soundSpeed(rhoR, pR, gamma)
    SL, SR, Sstar = _hllcWaveSpeeds(rhoL, uL, pL, aL, rhoR, uR, pR, aR, gamma)

    gm1 = gamma - scalar_t(1.0)
    EL = pL / gm1 + scalar_t(0.5) * rhoL * uL * uL
    ER = pR / gm1 + scalar_t(0.5) * rhoR * uR * uR

    F_rho = scalar_t(0.0)
    F_mom = scalar_t(0.0)
    F_E = scalar_t(0.0)
    if SL >= scalar_t(0.0):
        F_rho = rhoL * uL
        F_mom = rhoL * uL * uL + pL
        F_E = (EL + pL) * uL
    elif SR <= scalar_t(0.0):
        F_rho = rhoR * uR
        F_mom = rhoR * uR * uR + pR
        F_E = (ER + pR) * uR
    elif Sstar >= scalar_t(0.0):
        # F*_L = F_L + S_L (U*_L - U_L), U*_K = rho_K (S_K - u_K) / (S_K - S*) (1, S*, E_K/rho_K + (S* - u_K)(S* + p_K/(rho_K (S_K - u_K))))
        c = rhoL * (SL - uL) / (SL - Sstar)
        UsE = c * (EL / rhoL + (Sstar - uL) * (Sstar + pL / (rhoL * (SL - uL))))
        F_rho = rhoL * uL + SL * (c - rhoL)
        F_mom = rhoL * uL * uL + pL + SL * (c * Sstar - rhoL * uL)
        F_E = (EL + pL) * uL + SL * (UsE - EL)
    else:
        c = rhoR * (SR - uR) / (SR - Sstar)
        UsE = c * (ER / rhoR + (Sstar - uR) * (Sstar + pR / (rhoR * (SR - uR))))
        F_rho = rhoR * uR + SR * (c - rhoR)
        F_mom = rhoR * uR * uR + pR + SR * (c * Sstar - rhoR * uR)
        F_E = (ER + pR) * uR + SR * (UsE - ER)
    return F_rho, F_mom, F_E, Sstar
