"""Meshless conserved flux accumulation, one CSR row per particle.

Both sides evaluate the same canonical pair; this preserves antisymmetry.
The timestep stays on device, including Hancock prediction and the MFV mass cap.
"""
import warp as wp
from typing import Any
from warpSPHCore.type_config import *
from warpSPHCore.autograd import *
from warpSPHCore.dataTypes import *
from warpSPHCore.radiusSearch.grid_util import getIndexRangeLane
from warpSPHCore.math import *
from warpSPHCore.kernels import *
from warpSPHCore.util import *
from warpSPHCore.enumTypes import *
from warpSPHCore.mfm.wp_mfm import _vecDtype

_TINY = 1.0e-30

from warpSPHCore.mfm.wp_mfm import guardedFace
from ..reconstruction.meshless import reconstructSide, sideTimeDerivative
from ..riemann.meshless import faceFluxPair, hllcStar

@wp.func
def mfmFlux_Func_i(
    i: wp.int32, iPtcl: Any, referenceState: Any, domainState: domainData, kernelProperties: kernelState,
    beginIndex: offset_t, numIndices: wp.int32, offsetArray: wp.array(dtype=index_t),  # type: ignore
    rho: wp.array(dtype=scalar_t), vel: wp.array(dtype=Any), pres: wp.array(dtype=scalar_t),  # type: ignore
    gRho: wp.array(dtype=Any), gVel: wp.array(dtype=Any), gP: wp.array(dtype=Any),  # type: ignore
    aRho: wp.array(dtype=scalar_t), aVel: wp.array(dtype=Any), aP: wp.array(dtype=scalar_t),  # type: ignore
    vol: wp.array(dtype=scalar_t), Einv: wp.array(dtype=Any), lam: wp.array(dtype=Any), condBad: wp.array(dtype=wp.int32),  # type: ignore
    gamma: scalar_t, dt: scalar_t, mode: wp.int32, order: wp.int32, psi1: scalar_t, psi2: scalar_t,
    power: scalar_t, timeCentred: wp.int32, centred: wp.int32, areaCap: wp.int32, retry: wp.int32, massLimit: scalar_t,
    faceGuard: wp.int32, closureCap: scalar_t,
    dmIn: scalar_t, dpIn: Any, deIn: scalar_t,
):
    dm = dmIn
    dp = dpIn
    de = deIn
    for n in range(numIndices):
        j = wp.int32(offsetArray[beginIndex + offset_t(n)])
        if j == i:
            continue
        # canonical pair (a, b), a < b: both threads of the pair compute the identical flux
        a = wp.min(i, j)
        b = wp.max(i, j)
        aPtcl = getParticleData(referenceState, a)
        bPtcl = getParticleData(referenceState, b)
        d = -computeDistanceVec(aPtcl.position, bPtcl.position, domainState)        # x_b - x_a
        r = wp.length(d)
        if r > wp.max(aPtcl.support, bPtcl.support):
            continue
        A0 = guardedFace(d, vol[a], vol[b], Einv[a], Einv[b], aPtcl.support, bPtcl.support, condBad[a], condBad[b],
                         kernelProperties.kernelFunction, domainState.dim, centred, areaCap)
        kappa = wp.pow(wp.length(A0), power)
        dlam = lam[b] - lam[a]
        if closureCap > scalar_t(0.0):
            # bounded closure (E2 wave 2 D): the potential difference of a face enters through the vector clip
            # to magnitude `closureCap` -- odd and symmetric (A_ab = -A_ba), so the correction can never reverse
            # or out-weigh the face (|A' - A_g| <= cap |A_g| for power 1); the solve uses the same clip
            dmag = wp.length(dlam)
            if dmag > closureCap:
                dlam = dlam * (closureCap / dmag)
        A = A0 + kappa * dlam
        if faceGuard != 0:
            # EXPERIMENTAL (E2 wave 2 D, default off): the corrected face may point backwards
            # (A . d < 0), which turns the pair pressure into an attraction. 1: drop the
            # correction on such a pair; 2: scale it to the largest s in [0, 1] that keeps
            # A . d >= 0.25 A0 . d.
            ad0 = wp.dot(A0, d)
            adc = wp.dot(A, d)
            if faceGuard == 1:
                if adc <= scalar_t(0.0):
                    A = A0
            elif faceGuard >= 3:
                # 3 / 4: cap |A' - A_g| at 0.5 / 0.25 of |A_g|: the correction can neither reverse
                # nor strongly amplify the face (A'.A_g >= (1 - c) |A_g|^2 > 0)
                cap = scalar_t(0.5)
                if faceGuard == 4:
                    cap = scalar_t(0.25)
                corrv = A - A0
                cm = wp.length(corrv)
                lim = cap * wp.length(A0)
                if cm > lim:
                    A = A0 + corrv * (lim / cm)
            else:
                floor = scalar_t(0.25) * ad0
                if adc < floor:
                    corr = adc - ad0
                    sc = scalar_t(0.0)
                    if corr < scalar_t(0.0):
                        sc = wp.max(scalar_t(0.0), (floor - ad0) / corr)
                    A = A0 + sc * (A - A0)
        Amag = wp.length(A)
        if Amag <= scalar_t(0.0):
            continue
        nrm = A / Amag

        frac = aPtcl.support / (aPtcl.support + bPtcl.support)
        va = vel[a]
        vb = vel[b]
        # face (quadrature point) velocity from the half-step particle velocities (Eq. 21)
        vha = va
        vhb = vb
        if order >= 2 and dt > scalar_t(0.0) and timeCentred != 0:
            gvl_a = gVel[a] * scalar_t(0.0)
            gvl_b = gVel[b] * scalar_t(0.0)
            for c in range(va.length):
                for k in range(va.length):
                    gvl_a[c, k] = aVel[a][c] * gVel[a][c, k]
                    gvl_b[c, k] = aVel[b][c] * gVel[b][c, k]
            _r1, dva, _p1 = sideTimeDerivative(rho[a], va, pres[a], aRho[a] * gRho[a], gvl_a, aP[a] * gP[a], gamma)
            _r2, dvb, _p2 = sideTimeDerivative(rho[b], vb, pres[b], aRho[b] * gRho[b], gvl_b, aP[b] * gP[b], gamma)
            vha = va + scalar_t(0.5) * dt * dva
            vhb = vb + scalar_t(0.5) * dt * dvb
        # the face frame is time-centred in both modes (GIZMO's per-direction MFV bias averages
        # over the two directions to this frame in the equal-timestep limit; a shared biased
        # frame is only first order)
        vframe = vha + frac * (vhb - vha)

        rL, vL, pL = reconstructSide(
            rho[a], va, pres[a], rho[b], vb, pres[b], gRho[a], gVel[a], gP[a], aRho[a], aVel[a], aP[a],
            frac * d, frac, vframe, dt, gamma, order, psi1, psi2, timeCentred)
        rR, vR, pR = reconstructSide(
            rho[b], vb, pres[b], rho[a], va, pres[a], gRho[b], gVel[b], gP[b], aRho[b], aVel[b], aP[b],
            -(scalar_t(1.0) - frac) * d, scalar_t(1.0) - frac, vframe, dt, gamma, order, psi1, psi2, timeCentred)
        rL = wp.max(rL, scalar_t(_TINY))
        rR = wp.max(rR, scalar_t(_TINY))
        pL = wp.max(pL, scalar_t(_TINY))
        pR = wp.max(pR, scalar_t(_TINY))
        uL = wp.dot(vL, nrm)
        uR = wp.dot(vR, nrm)
        vtL = vL - uL * nrm
        vtR = vR - uR * nrm
        Fm, Fp, Fe, Ss, Ps = faceFluxPair(rL, uL, vtL, pL, rR, uR, vtR, pR, gamma, mode, nrm, vframe)
        if retry != 0:
            # GIZMO's failure handling: invalid star pressure -> first-order states -> zero velocity jump
            s1 = wp.max(scalar_t(0.0), -wp.dot(vb - va, d) / wp.max(r, scalar_t(_TINY)))
            s2 = wp.max(scalar_t(0.0), wp.dot(va - vb, nrm))
            v2 = wp.max(s1, s2)
            v2 = v2 * v2
            limit = scalar_t(1.1) * wp.max(pres[a] + rho[a] * v2, pres[b] + rho[b] * v2)
            if mode == 1:
                limit = limit * scalar_t(2.0)
            ok = Ps > scalar_t(0.0) and wp.isfinite(Ps) and wp.isfinite(Ss) and Ps <= scalar_t(1.4) * limit and wp.isfinite(Fm) and wp.isfinite(Fe)
            if not ok:
                vL1 = va - vframe
                vR1 = vb - vframe
                uL1 = wp.dot(vL1, nrm)
                uR1 = wp.dot(vR1, nrm)
                r1L = wp.max(rho[a], scalar_t(_TINY))
                r1R = wp.max(rho[b], scalar_t(_TINY))
                p1L = wp.max(pres[a], scalar_t(_TINY))
                p1R = wp.max(pres[b], scalar_t(_TINY))
                Fm, Fp, Fe, Ss, Ps = faceFluxPair(r1L, uL1, vL1 - uL1 * nrm, p1L, r1R, uR1, vR1 - uR1 * nrm, p1R,
                                                  gamma, mode, nrm, vframe)
                ok2 = Ps > scalar_t(0.0) and wp.isfinite(Ps) and wp.isfinite(Ss) and wp.isfinite(Fm) and wp.isfinite(Fe)
                if not ok2:
                    zero = scalar_t(0.0)
                    Fm, Fp, Fe, Ss, Ps = faceFluxPair(r1L, zero, vL1 * zero, p1L, r1R, zero, vR1 * zero, p1R,
                                                      gamma, mode, nrm, vframe)
        if mode == 1 and massLimit > scalar_t(0.0) and dt > scalar_t(0.0):
            # GIZMO: a pair moves at most massLimit of the donor's mass per step (mass update only)
            dmass = Amag * Fm * dt
            cap = massLimit * rho[a] * vol[a]
            if dmass < scalar_t(0.0):
                cap = massLimit * rho[b] * vol[b]
            Fm = wp.max(wp.min(dmass, cap), -cap) / (Amag * dt)
        # dQ_a -= |A| F, dQ_b += |A| F
        sgn = scalar_t(1.0)
        if i == a:
            sgn = scalar_t(-1.0)
        dm += sgn * Amag * Fm
        dp += sgn * Amag * Fp
        de += sgn * Amag * Fe
    return dm, dp, de


def _makeFluxKernel(listMode: bool):
    """One pair implementation, separate grid and neighbour traversal kernels."""
    @wp.kernel(name="mfmFlux_neighbors" if listMode else "mfmFlux_grid", enable_backward=False)
    def flux(
        queryState: Any, referenceState: Any, domainState: domainData,
        useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
        correctionData: Any, kernelProperties: kernelState,
        # canonical ABI prefix -- do not change
    
        rho: wp.array(dtype=scalar_t), vel: wp.array(dtype=Any), pres: wp.array(dtype=scalar_t),  # type: ignore
        gRho: wp.array(dtype=Any), gVel: wp.array(dtype=Any), gP: wp.array(dtype=Any),  # type: ignore
        aRho: wp.array(dtype=scalar_t), aVel: wp.array(dtype=Any), aP: wp.array(dtype=scalar_t),  # type: ignore
        vol: wp.array(dtype=scalar_t), Einv: wp.array(dtype=Any), lam: wp.array(dtype=Any), condBad: wp.array(dtype=wp.int32),  # type: ignore
        gamma: scalar_t, dt: wp.array(dtype=scalar_t), mode: wp.int32, order: wp.int32, psi1: scalar_t, psi2: scalar_t,
        power: scalar_t, timeCentred: wp.int32, centred: wp.int32, areaCap: wp.int32, retry: wp.int32, massLimit: scalar_t,
        faceGuard: wp.int32, closureCap: scalar_t,
    
        output_dm: wp.array(dtype=scalar_t), output_dp: wp.array(dtype=Any), output_de: wp.array(dtype=scalar_t),  # type: ignore
    ):
        i = wp.tid()
        if i >= queryState.positions.shape[0]:
            return
        iPtcl = getParticleData(queryState, i)
        dm = scalar_t(0.0)
        dp = zero_like_warp(output_dp[i])
        de = scalar_t(0.0)
        traversal = wp.static(listMode)
        stepDt = dt[0]
        numOffsets = gridState.numOffsets if not traversal else 1
        for o in range(numOffsets):
            beginIndex, numIndices = getIndexRangeLane(i, o, 0, 1, traversal, adjacencyState, gridState, queryState, domainState)
            if beginIndex < 0:
                continue
            dm, dp, de = mfmFlux_Func_i(
                i, iPtcl, referenceState, domainState, kernelProperties, beginIndex, numIndices,
                adjacencyState.neighborList if traversal else gridState.sortIndex,
                rho, vel, pres, gRho, gVel, gP, aRho, aVel, aP, vol, Einv, lam, condBad,
                gamma, stepDt, mode, order, psi1, psi2, power, timeCentred, centred, areaCap, retry, massLimit, faceGuard, closureCap, dm, dp, de)
        # rates dQ/dt = -sum_j |A| F with the sign convention folded into the accumulation above
        output_dm[i] = dm
        output_dp[i] = dp
        output_de[i] = de
    return flux


_FLUX_SPEC = OperatorSpec(
    kernel=_makeFluxKernel(True),
    outputs=(OutputSpec(dtype=scalar_t), OutputSpec(dtype=_vecDtype), OutputSpec(dtype=scalar_t)),
    extras=(ExtraSpec("rho", ExtraKind.TENSOR), ExtraSpec("vel", ExtraKind.TENSOR), ExtraSpec("pres", ExtraKind.TENSOR),
            ExtraSpec("gRho", ExtraKind.TENSOR), ExtraSpec("gVel", ExtraKind.TENSOR), ExtraSpec("gP", ExtraKind.TENSOR),
            ExtraSpec("aRho", ExtraKind.TENSOR), ExtraSpec("aVel", ExtraKind.TENSOR), ExtraSpec("aP", ExtraKind.TENSOR),
            ExtraSpec("vol", ExtraKind.TENSOR), ExtraSpec("Einv", ExtraKind.TENSOR), ExtraSpec("lam", ExtraKind.TENSOR),
            ExtraSpec("condBad", ExtraKind.TENSOR),
            ExtraSpec("gamma", ExtraKind.SCALAR), ExtraSpec("dt", ExtraKind.TENSOR), ExtraSpec("mode", ExtraKind.SCALAR),
            ExtraSpec("order", ExtraKind.SCALAR), ExtraSpec("psi1", ExtraKind.SCALAR), ExtraSpec("psi2", ExtraKind.SCALAR),
            ExtraSpec("power", ExtraKind.SCALAR), ExtraSpec("timeCentred", ExtraKind.SCALAR),
            ExtraSpec("centred", ExtraKind.SCALAR), ExtraSpec("areaCap", ExtraKind.SCALAR), ExtraSpec("retry", ExtraKind.SCALAR),
            ExtraSpec("massLimit", ExtraKind.SCALAR), ExtraSpec("faceGuard", ExtraKind.SCALAR),
            ExtraSpec("closureCap", ExtraKind.SCALAR)),
)


_TRAVERSAL_SPECS = {True: _FLUX_SPEC}


def fluxSpec(useAdjacency):
    """Select using prepared host metadata; never infer traversal from a device flag."""
    from dataclasses import replace
    useAdjacency = bool(useAdjacency)
    if useAdjacency not in _TRAVERSAL_SPECS:
        _TRAVERSAL_SPECS[useAdjacency] = replace(_FLUX_SPEC, kernel=_makeFluxKernel(useAdjacency))
    return _TRAVERSAL_SPECS[useAdjacency]
