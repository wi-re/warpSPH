"""Meshless primitive slope and pair limiters, and Hancock face prediction.

Formulas retain the core reference operand order and positivity fallback.
"""
import warp as wp
from typing import Any
from warpSPHCore.type_config import *
from warpSPHCore.autograd import *
from warpSPHCore.autograd.operator_spec import RowRole
from warpSPHCore.dataTypes import *
from warpSPHCore.radiusSearch.grid_util import getIndexRangeLane
from warpSPHCore.math import *
from warpSPHCore.kernels import *
from warpSPHCore.util import *
from warpSPHCore.enumTypes import *
from warpSPHCore.mfm.wp_mfm import _vecDtype

_TINY = 1.0e-30

@wp.func
def alphaScalar(phi: scalar_t, ngbHi: scalar_t, ngbLo: scalar_t, midHi: scalar_t, midLo: scalar_t, beta: scalar_t):
    """Eq. B2 for one component: ``min(1, beta min(up_ratio, dn_ratio))``."""
    tiny = scalar_t(_TINY)
    big = scalar_t(1.0e30)
    up = midHi - phi
    dn = phi - midLo
    rUp = big
    rDn = big
    if up > tiny:
        rUp = (ngbHi - phi) / up
    if dn > tiny:
        rDn = (phi - ngbLo) / dn
    return wp.min(scalar_t(1.0), beta * wp.min(rUp, rDn))


@wp.func
def _sgn(x: scalar_t):
    if x > scalar_t(0.0):
        return scalar_t(1.0)
    if x < scalar_t(0.0):
        return scalar_t(-1.0)
    return scalar_t(0.0)


@wp.func
def pairLimit(phiI: scalar_t, phiJ: scalar_t, phi0: scalar_t, frac: scalar_t, psi1: scalar_t, psi2: scalar_t):
    """Eq. B4: the limited face value seen from ``i``."""
    if phiI == phiJ:
        return phiI
    tiny = scalar_t(_TINY)
    dphi = wp.abs(phiI - phiJ)
    d1 = psi1 * dphi
    d2 = psi2 * dphi
    bar = phiI + frac * (phiJ - phiI)
    pmin = wp.min(phiI, phiJ)
    pmax = wp.max(phiI, phiJ)
    lo = pmin - d1
    if _sgn(lo) != _sgn(pmin):
        lo = pmin / (scalar_t(1.0) + d1 / wp.max(wp.abs(pmin), tiny))
    hi = pmax + d1
    if _sgn(hi) != _sgn(pmax):
        hi = pmax / (scalar_t(1.0) + d1 / wp.max(wp.abs(pmax), tiny))
    if phiI < phiJ:
        return wp.max(lo, wp.min(bar + d2, phi0))
    return wp.min(hi, wp.max(bar - d2, phi0))
@wp.func
def mfmLimiter_Func_i(
    i: wp.int32, iPtcl: Any, referenceState: Any, domainState: domainData, kernelProperties: kernelState,
    beginIndex: offset_t, numIndices: wp.int32, offsetArray: wp.array(dtype=index_t),  # type: ignore
    rho: wp.array(dtype=scalar_t), vel: wp.array(dtype=Any), pres: wp.array(dtype=scalar_t),  # type: ignore
    gRho: wp.array(dtype=Any), gVel: wp.array(dtype=Any), gP: wp.array(dtype=Any),  # type: ignore
    nbHiRho: scalar_t, nbLoRho: scalar_t, mdHiRho: scalar_t, mdLoRho: scalar_t,
    nbHiP: scalar_t, nbLoP: scalar_t, mdHiP: scalar_t, mdLoP: scalar_t,
    nbHiV: Any, nbLoV: Any, mdHiV: Any, mdLoV: Any,
):
    for n in range(numIndices):
        j = wp.int32(offsetArray[beginIndex + offset_t(n)])
        if j == i:
            continue
        jPtcl = getParticleData(referenceState, j)
        d = -computeDistanceVec(iPtcl.position, jPtcl.position, domainState)
        r = wp.length(d)
        if r > wp.max(iPtcl.support, jPtcl.support):
            continue
        frac = iPtcl.support / (iPtcl.support + jPtcl.support)
        disp = frac * d
        nbHiRho = wp.max(nbHiRho, rho[j])
        nbLoRho = wp.min(nbLoRho, rho[j])
        nbHiP = wp.max(nbHiP, pres[j])
        nbLoP = wp.min(nbLoP, pres[j])
        mR = rho[i] + wp.dot(gRho[i], disp)
        mP = pres[i] + wp.dot(gP[i], disp)
        mdHiRho = wp.max(mdHiRho, mR)
        mdLoRho = wp.min(mdLoRho, mR)
        mdHiP = wp.max(mdHiP, mP)
        mdLoP = wp.min(mdLoP, mP)
        vj = vel[j]
        mV = vel[i] + matmul(gVel[i], disp)
        for c in range(nbHiV.length):
            nbHiV[c] = wp.max(nbHiV[c], vj[c])
            nbLoV[c] = wp.min(nbLoV[c], vj[c])
            mdHiV[c] = wp.max(mdHiV[c], mV[c])
            mdLoV[c] = wp.min(mdLoV[c], mV[c])
    return nbHiRho, nbLoRho, mdHiRho, mdLoRho, nbHiP, nbLoP, mdHiP, mdLoP, nbHiV, nbLoV, mdHiV, mdLoV


@wp.kernel(enable_backward=False)
def mfmLimiter_Kernel(
    queryState: Any, referenceState: Any, domainState: domainData,
    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any, kernelProperties: kernelState,
    # canonical ABI prefix -- do not change

    rho: wp.array(dtype=scalar_t), vel: wp.array(dtype=Any), pres: wp.array(dtype=scalar_t),  # type: ignore
    gRho: wp.array(dtype=Any), gVel: wp.array(dtype=Any), gP: wp.array(dtype=Any),  # type: ignore
    cond: wp.array(dtype=scalar_t), betaMin: scalar_t, betaMax: scalar_t, condCrit: scalar_t,

    rowStart: wp.int32,
    output_aRho: wp.array(dtype=scalar_t), output_aVel: wp.array(dtype=Any), output_aP: wp.array(dtype=scalar_t),  # type: ignore
):
    i = wp.tid()
    if i >= queryState.positions.shape[0]:
        return
    gi = i + rowStart
    iPtcl = getParticleData(queryState, i)
    vi = vel[gi]
    nbHiRho = rho[gi]
    nbLoRho = rho[gi]
    mdHiRho = rho[gi]
    mdLoRho = rho[gi]
    nbHiP = pres[gi]
    nbLoP = pres[gi]
    mdHiP = pres[gi]
    mdLoP = pres[gi]
    nbHiV = vi
    nbLoV = vi
    mdHiV = vi
    mdLoV = vi
    numOffsets = gridState.numOffsets if not useAdjacency else 1
    for o in range(numOffsets):
        beginIndex, numIndices = getIndexRangeLane(i, o, 0, 1, useAdjacency, adjacencyState, gridState, queryState, domainState)
        if beginIndex < 0:
            continue
        nbHiRho, nbLoRho, mdHiRho, mdLoRho, nbHiP, nbLoP, mdHiP, mdLoP, nbHiV, nbLoV, mdHiV, mdLoV = mfmLimiter_Func_i(
            gi, iPtcl, referenceState, domainState, kernelProperties, beginIndex, numIndices,
            adjacencyState.neighborList if useAdjacency else gridState.sortIndex,
            rho, vel, pres, gRho, gVel, gP,
            nbHiRho, nbLoRho, mdHiRho, mdLoRho, nbHiP, nbLoP, mdHiP, mdLoP, nbHiV, nbLoV, mdHiV, mdLoV)
    b = wp.max(betaMin, betaMax * wp.min(scalar_t(1.), condCrit / cond[gi]))
    output_aRho[i] = alphaScalar(rho[gi], nbHiRho, nbLoRho, mdHiRho, mdLoRho, b)
    output_aP[i] = alphaScalar(pres[gi], nbHiP, nbLoP, mdHiP, mdLoP, b)
    aV = zero_like_warp(vi)
    for c in range(vi.length):
        aV[c] = alphaScalar(vi[c], nbHiV[c], nbLoV[c], mdHiV[c], mdLoV[c], b)
    output_aVel[i] = aV


_LIMITER_SPEC = OperatorSpec(
    rowChunked=True,
    kernel=mfmLimiter_Kernel,
    outputs=(OutputSpec(dtype=scalar_t), OutputSpec(dtype=_vecDtype), OutputSpec(dtype=scalar_t)),
    extras=(ExtraSpec("rho", ExtraKind.TENSOR, RowRole.REFERENCE), ExtraSpec("vel", ExtraKind.TENSOR, RowRole.REFERENCE), ExtraSpec("pres", ExtraKind.TENSOR, RowRole.REFERENCE),
            ExtraSpec("gRho", ExtraKind.TENSOR, RowRole.REFERENCE), ExtraSpec("gVel", ExtraKind.TENSOR, RowRole.REFERENCE), ExtraSpec("gP", ExtraKind.TENSOR, RowRole.REFERENCE),
            ExtraSpec("cond", ExtraKind.TENSOR, RowRole.REFERENCE), ExtraSpec("betaMin", ExtraKind.SCALAR),
            ExtraSpec("betaMax", ExtraKind.SCALAR), ExtraSpec("condCrit", ExtraKind.SCALAR), ExtraSpec("rowStart", ExtraKind.SCALAR, RowRole.ROW_START)),
)
@wp.func
def sideTimeDerivative(rho: scalar_t, vBoost: Any, pres: scalar_t, gRho: Any, gVel: Any, gP: Any, gamma: scalar_t):
    """Eq. A4 for one particle's (limited) gradients; ``vBoost`` the lab-frame velocity (the
    half-step prediction is lab-frame, as in GIZMO)."""
    divV = scalar_t(0.0)
    for c in range(vBoost.length):
        divV += gVel[c, c]
    dRho = -(wp.dot(vBoost, gRho) + rho * divV)
    dV = -(matmul(gVel, vBoost) + gP / rho)
    dP = -(wp.dot(vBoost, gP) + gamma * pres * divV)
    return dRho, dV, dP


@wp.func
def reconstructSide(
    rhoS: scalar_t, vS: Any, pS: scalar_t, rhoO: scalar_t, vO: Any, pO: scalar_t,
    gRho: Any, gVel: Any, gP: Any, aRho: scalar_t, aVel: Any, aP: scalar_t,
    disp: Any, frac: scalar_t, vframe: Any, dt: scalar_t, gamma: scalar_t, order: wp.int32,
    psi1: scalar_t, psi2: scalar_t, vdt: wp.int32,
):
    """Face state of one side: limited linear reconstruction (lab frame), pairwise limiter,
    half-step prediction (lab frame, as in GIZMO), then the boost. Returns the boosted
    ``(rho, v, P)``."""
    rhoF = rhoS
    vF = vS
    pF = pS
    if order >= 2:
        rhoF = pairLimit(rhoS, rhoO, rhoS + aRho * wp.dot(gRho, disp), frac, psi1, psi2)
        pF = pairLimit(pS, pO, pS + aP * wp.dot(gP, disp), frac, psi1, psi2)
        lin = matmul(gVel, disp)
        for c in range(vS.length):
            vF[c] = pairLimit(vS[c], vO[c], vS[c] + aVel[c] * lin[c], frac, psi1, psi2)
    vF = vF - vframe
    if order >= 2 and dt > scalar_t(0.0):
        # GIZMO predicts the lab-frame particle state with the fixed-point time derivative and
        # boosts the predicted state to the face frame (the face-frame PDE would add the
        # face-advection term vframe . grad W, an O(h) difference in the face state)
        gVelL = gVel * scalar_t(0.0)
        for c in range(vS.length):
            for k in range(vS.length):
                gVelL[c, k] = aVel[c] * gVel[c, k]
        dR, dV, dPp = sideTimeDerivative(rhoS, vS, pS, aRho * gRho, gVelL, aP * gP, gamma)
        half = scalar_t(0.5) * dt
        rP = rhoF + half * dR
        pP = pF + half * dPp
        if rP > scalar_t(0.0) and pP > scalar_t(0.0):
            rhoF = rP
            pF = pP
            vF = vF + half * dV
    return rhoF, vF, pF
