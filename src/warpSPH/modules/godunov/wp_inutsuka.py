"""Godunov SPH of Inutsuka (2002), the kernel-convolution form with a Gaussian kernel (GODUNOV_SPH_PLAN layer 3).

Derivation (Inutsuka 2002 §2-3, Murante et al. 2011 §2-3 and App. A): convolve the momentum and energy equations with the kernel, integrate by parts and
use `1 = sum_j m_j W(x - x_j)/rho(x)`:

    a_i      = - sum_j m_j p*_ij  sum_{h in {h_i, h_j}} V_ij^2(h) grad_i W(x_ij, sqrt2 h)
    du_i/dt  = - sum_j m_j p*_ij (u*_ij - u_i)  n_ij . sum_h V_ij^2(h) grad_i W(x_ij, sqrt2 h)

with the Gaussian `W(x, H) = (pi H^2)^(-d/2) exp(-|x|^2 / H^2)`, so that `int W(x - x_i, h) W(x - x_j, h) dx = W(x_ij, sqrt2 h)`, `V_ij^2(h)` the factor
that `1/rho^2(x)`, interpolated along the pair axis, contributes:

    V(t) = A t^3 + B t^2 + C t + D        (cubic Hermite through V_i, V_j and the axis derivatives V'_i, V'_j at t = +-s_ij/2; Eq. 60-61;
                                           linear, A = B = 0, when `V'_i V'_j < 0` or if asked for)
    V_ij^2(h) = E[V(t)^2],  t ~ N(0, h^2/4)  = 15/64 h^6 A^2 + 3/16 h^4 (2AC + B^2) + 1/4 h^2 (2BD + C^2) + D^2          (Eq. 52 / 64)
    s*_ij(h)  = E[t V(t)^2] / E[V(t)^2]      = (15/32 h^6 AB + 3/8 h^4 (AD + BC) + 1/2 h^2 CD) / V_ij^2(h)                  (Eq. 57 / 65)

(`s*` is where the linearly interpolated field takes the `V^2 W W`-weighted mean value: the interface at which the Riemann problem is solved; the
two smoothing lengths' values are averaged, antisymmetric under `i <-> j`.) The momentum is conserved because `p*_ij` is symmetric and the
bracket is antisymmetric (`K_ij` below is symmetric, `x_ij` is not); the pair's total energy is conserved exactly as in `wp_gsph.py`.

`h` is the Gaussian's smoothing length: `support / 3` (the Gaussian truncated at 3h holds as many neighbours as the compact kernel's support, Murante
et al. 2011 §3.1); every Gaussian is cut at `3 sqrt2 h`, so the neighbour list has to reach `sqrt2 * max(support)`. The density is Murante's symmetrised
sum (Eq. 15) `rho_i = sum_j m_j [W(x_ij, sqrt2 h_i) + W(x_ij, sqrt2 h_j)] / 2` (an even function of the pair, removing the SPH density asymmetry that
Cha et al. 2010 blame for the spurious force), and `grad rho_i` the same sum of gradients: V and V' come from these.

No time-centring (`xdot*_i = v_i + a_i dt/2`) and no `C dt/2` evolution of the states: the right-hand side is an ODE system for the integrator.
"""

import warp as wp
from warp.types import vector, matrix
from typing import Any, Optional, Union
import torch
from warpSPHCore.profiling import record_function
from warpSPHCore import *
from ...configurations.moduleConfigurations.diffusionParameters import DiffusionParameters, StateLimiter
from ..reconstruction import reconstructPairIncrements, reconstructPair1D
from ..riemann import riemannStarState

__all__ = ['computeGaussianDensityWarp', 'computeInutsukaWarp', 'inutsukaVolumes']

_SQRT2 = 1.4142135623730951
#: the Gaussian is truncated at CUT * (its own sqrt2 h), CUT = 3 (Murante et al. 2011)
_CUT = 3.0


@wp.func
def gaussian(r2: scalar_t, H: scalar_t, dim: wp.int32):
    """`W(x, H) = (pi H^2)^(-d/2) exp(-|x|^2 / H^2)` for `|x|^2 = r2`."""
    return wp.pow(scalar_t(3.141592653589793) * H * H, -scalar_t(0.5) * scalar_t(dim)) * wp.exp(-r2 / (H * H))


# ----------------------------------------------------------------------------------------------- pass 1: density and its gradient

@wp.func
def gaussianDensity_Func_Adjacency(
    i: wp.int32, dim: wp.int32,
    queryState: Any, referenceState: Any, domainState: domainData,
    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData, numOffsets: wp.int32,
    kernelProperties: kernelState, widthFactor: scalar_t,
    queryH: wp.array(dtype=scalar_t), referenceH: wp.array(dtype=scalar_t),  # type: ignore
    gradValue: Any,
):
    xi, hi, mi, rhoi, ki = getParticle(queryState, i)
    grad = zero_like_warp(gradValue)
    rho = scalar_t(0.0)
    if kernelProperties.operationMode != wp.static(OperationDirection.TrueAllToToAll.value):
        if not checkDirectionality_i(ki, kernelProperties.operationMode):
            return grad, rho
    hGi = queryH[i]
    for o in range(numOffsets):
        beginIndex = wp.int32(0)
        numIndices = wp.int32(0)
        if useAdjacency:
            beginIndex = adjacencyState.neighborOffsets[i]
            numIndices = adjacencyState.numNeighbors[i]
        else:
            beginIndex, numIndices = checkOffset(
                i, queryState.positions, gridState.numCells, gridState.D,
                o, gridState.cellOffsets, gridState.hashTable, gridState.cellTable,
                domainState.periodicity, gridState.qMin, gridState.qMax, gridState.hCell
            )
            if beginIndex < 0:
                continue
        offsetArray = gridState.sortIndex
        if useAdjacency:
            offsetArray = adjacencyState.neighborList
        for neighborIndex in range(numIndices):
            jj = beginIndex + neighborIndex
            j = wp.int32(offsetArray[jj])
            if kernelProperties.operationMode != wp.static(OperationDirection.TrueAllToToAll.value):
                if not checkDirectionality_j(referenceState.kinds[j], kernelProperties.operationMode):
                    continue
            xj, hj, mj, rhoj, kj = getParticle(referenceState, j)
            hGj = referenceH[j]
            x_ij = computeDistanceVec(xi, xj, domainState)
            r2 = wp.dot(x_ij, x_ij)
            # each Gaussian (sqrt2 h wide) is cut at 3 sqrt2 h
            wi = scalar_t(0.0)
            wj = scalar_t(0.0)
            Hi = widthFactor * hGi
            Hj = widthFactor * hGj
            if r2 < scalar_t(9.0) * Hi * Hi:
                wi = gaussian(r2, Hi, dim)
            if r2 < scalar_t(9.0) * Hj * Hj:
                wj = gaussian(r2, Hj, dim)
            rho += scalar_t(0.5) * mj * (wi + wj)
            # grad_x W(x, H) = -2 x / H^2 W
            grad -= scalar_t(0.5) * mj * (scalar_t(2.0) * wi / (Hi * Hi) + scalar_t(2.0) * wj / (Hj * Hj)) * x_ij
    return grad, rho


@wp.kernel
def gaussianDensity_Kernel(
    queryState: Any, referenceState: Any, domainState: domainData,
    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any,
    kernelProperties: kernelState,
    # Do not change the parameters above
    widthFactor: scalar_t,
    queryH: wp.array(dtype=scalar_t), referenceH: wp.array(dtype=scalar_t),  # type: ignore
    gradRho: wp.array(dtype=vector(length=Any, dtype=scalar_t)),  # type: ignore
    rho: wp.array(dtype=scalar_t),  # type: ignore
):
    i = wp.tid()
    if i >= queryState.positions.shape[0]:
        return
    g, r = gaussianDensity_Func_Adjacency(
        i, domainState.dim, queryState, referenceState, domainState,
        useAdjacency, adjacencyState, gridState, gridState.numOffsets if not useAdjacency else 1,
        kernelProperties, widthFactor, queryH, referenceH, gradRho[i])
    gradRho[i] = g
    rho[i] = r


def _vecDtype(ctx, extras):
    return castTorchToWarpAsBuiltins(ctx.query.positions).dtype


def _scalarDtype(ctx, extras):
    return castTorchToWarpAsBuiltins(ctx.query.densities).dtype


_GAUSS_DENSITY = OperatorSpec(
    kernel=gaussianDensity_Kernel,
    outputs=(OutputSpec(dtype=_vecDtype, shape=ShapeOf.QUERY), OutputSpec(dtype=_scalarDtype, shape=ShapeOf.QUERY)),
    extras=(ExtraSpec("widthFactor", ExtraKind.SCALAR), ExtraSpec("queryH", ExtraKind.TENSOR), ExtraSpec("referenceH", ExtraKind.TENSOR)),
)


def computeGaussianDensityWarp(
    queryParticles: ParticleState,
    operationProperties: OperationProperties,
    domain: DomainDescription,
    adjacency: Optional[Union[AdjacencyList, CompactHashMap]] = None,
    widthFactor: float = _SQRT2,
    gaussianH: Optional[torch.Tensor] = None,
):
    """`(grad rho_i, rho_i)`: Murante's symmetrised Gaussian density and its gradient (see the module docstring). `gaussianH` (default `supports / 3`) is the Gaussian's `h`; the
    density's Gaussians are `widthFactor * h` wide (`sqrt2`: Murante Eq. 15; 1: the plain sum `sum m W(x_ij, h)` under which Inutsuka's identity is exact)."""
    with record_function("warpSPH[computeGaussianDensity]"):
        ctx = SPHContext(
            query=queryParticles, properties=operationProperties, domain=domain,
            adjacency=adjacency, reference=queryParticles,
            corrections=Corrections(volumes=(None, None), crk=None, gradH=None, renorm=None),
        )
        hG = gaussianH if gaussianH is not None else queryParticles.supports / 3.0
        return launchOperator(_GAUSS_DENSITY, ctx, widthFactor=scalar_t(widthFactor), queryH=hG, referenceH=hG)


# ------------------------------------------------------------------------------------------------------- the pair volume integral

@wp.func
def inutsukaVolumes(
    Vi: scalar_t, Vj: scalar_t, dVi: scalar_t, dVj: scalar_t, s: scalar_t, h: scalar_t, cubic: wp.bool,
):
    """`(V_ij^2(h), s*(h))` of the specific-volume interpolant along the pair axis (`s = |x_ij|`, `dVi = V'_i`, `dVj = V'_j` the axis derivatives at the two
    ends). Cubic Hermite unless `cubic` is off or the two end derivatives disagree in sign (Inutsuka 2002 §3.1.2, which falls back to the linear
    interpolant there to avoid over- and undershoot)."""
    A = scalar_t(0.0)
    B = scalar_t(0.0)
    dV = Vi - Vj
    C = dV / s
    D = scalar_t(0.5) * (Vi + Vj)
    if cubic and dVi * dVj >= scalar_t(0.0):
        A = -scalar_t(2.0) * dV / (s * s * s) + (dVi + dVj) / (s * s)
        B = scalar_t(0.5) * (dVi - dVj) / s
        C = scalar_t(1.5) * dV / s - scalar_t(0.25) * (dVi + dVj)
        D = scalar_t(0.5) * (Vi + Vj) - scalar_t(0.125) * (dVi - dVj) * s
    h2 = h * h
    h4 = h2 * h2
    h6 = h4 * h2
    V2 = (scalar_t(15.0 / 64.0) * h6 * A * A + scalar_t(3.0 / 16.0) * h4 * (scalar_t(2.0) * A * C + B * B)
          + scalar_t(0.25) * h2 * (scalar_t(2.0) * B * D + C * C) + D * D)
    sNum = scalar_t(15.0 / 32.0) * h6 * A * B + scalar_t(3.0 / 8.0) * h4 * (A * D + B * C) + scalar_t(0.5) * h2 * C * D
    return V2, sNum / V2


@wp.func
def inutsuka_Func_Adjacency(
    i: wp.int32, dim: wp.int32,
    queryState: Any, referenceState: Any, domainState: domainData,
    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData, numOffsets: wp.int32,
    kernelProperties: kernelState,
    queryVelocities: wp.array(dtype=vector(length=Any, dtype=scalar_t)), referenceVelocities: wp.array(dtype=vector(length=Any, dtype=scalar_t)),  # type: ignore
    queryPressures: wp.array(dtype=scalar_t), referencePressures: wp.array(dtype=scalar_t),  # type: ignore
    queryGradRho: wp.array(dtype=vector(length=Any, dtype=scalar_t)), referenceGradRho: wp.array(dtype=vector(length=Any, dtype=scalar_t)),  # type: ignore
    queryGaussianH: wp.array(dtype=scalar_t), referenceGaussianH: wp.array(dtype=scalar_t),  # type: ignore
    queryVelocityTensor: wp.array(dtype=matrix(shape=(Any, Any), dtype=scalar_t)), referenceVelocityTensor: wp.array(dtype=matrix(shape=(Any, Any), dtype=scalar_t)),  # type: ignore
    queryStateGradients: wp.array(dtype=matrix(shape=(Any, Any), dtype=scalar_t)), referenceStateGradients: wp.array(dtype=matrix(shape=(Any, Any), dtype=scalar_t)),  # type: ignore
    queryAccelerations: wp.array(dtype=vector(length=Any, dtype=scalar_t)), halfDt: scalar_t,  # type: ignore
    params: DiffusionParameters, gamma: scalar_t, order: wp.int32, cubic: wp.bool,
    accelValue: Any,
):
    xi, hi, mi, rhoi, ki = getParticle(queryState, i)
    acc = zero_like_warp(accelValue)
    work = scalar_t(0.0)
    if kernelProperties.operationMode != wp.static(OperationDirection.TrueAllToToAll.value):
        if not checkDirectionality_i(ki, kernelProperties.operationMode):
            return acc, work
    hGi = queryGaussianH[i]
    vel_i = queryVelocities[i]
    vEnergy_i = queryVelocities[i]
    if params.timeCentredEnergy and halfDt > scalar_t(0.0):
        vEnergy_i = queryVelocities[i] + halfDt * queryAccelerations[i]
    P_i = queryPressures[i]
    gradRho_i = queryGradRho[i]
    J_i = queryVelocityTensor[0]
    G_i = queryStateGradients[0]
    if order >= 2:
        J_i = queryVelocityTensor[i]
        G_i = queryStateGradients[i]
    for o in range(numOffsets):
        beginIndex = wp.int32(0)
        numIndices = wp.int32(0)
        if useAdjacency:
            beginIndex = adjacencyState.neighborOffsets[i]
            numIndices = adjacencyState.numNeighbors[i]
        else:
            beginIndex, numIndices = checkOffset(
                i, queryState.positions, gridState.numCells, gridState.D,
                o, gridState.cellOffsets, gridState.hashTable, gridState.cellTable,
                domainState.periodicity, gridState.qMin, gridState.qMax, gridState.hCell
            )
            if beginIndex < 0:
                continue
        offsetArray = gridState.sortIndex
        if useAdjacency:
            offsetArray = adjacencyState.neighborList
        for neighborIndex in range(numIndices):
            jj = beginIndex + neighborIndex
            j = wp.int32(offsetArray[jj])
            if kernelProperties.operationMode != wp.static(OperationDirection.TrueAllToToAll.value):
                if not checkDirectionality_j(referenceState.kinds[j], kernelProperties.operationMode):
                    continue
            if i == j:
                continue
            xj, hj, mj, rhoj, kj = getParticle(referenceState, j)
            hGj = referenceGaussianH[j]
            x_ij = computeDistanceVec(xi, xj, domainState)
            r2 = wp.dot(x_ij, x_ij)
            r = safe_sqrt(r2)
            wi = scalar_t(0.0)
            wj = scalar_t(0.0)
            if r2 < scalar_t(18.0) * hGi * hGi:
                wi = gaussian(r2, scalar_t(1.4142135623730951) * hGi, dim)
            if r2 < scalar_t(18.0) * hGj * hGj:
                wj = gaussian(r2, scalar_t(1.4142135623730951) * hGj, dim)
            if wi + wj <= scalar_t(0.0):
                continue
            n = x_ij / (r + scalar_t(1.0e-14) * hi)
            vel_j = referenceVelocities[j]
            P_j = referencePressures[j]

            # specific volumes and their axis derivatives (grad V = -grad rho / rho^2), the interpolant moments and the interface position
            Vi = scalar_t(1.0) / rhoi
            Vj = scalar_t(1.0) / rhoj
            dVi = -wp.dot(n, gradRho_i) * Vi * Vi
            dVj = -wp.dot(n, referenceGradRho[j]) * Vj * Vj
            V2i, sStarI = inutsukaVolumes(Vi, Vj, dVi, dVj, r, hGi, cubic and params.cubicVolume)
            V2j, sStarJ = inutsukaVolumes(Vi, Vj, dVi, dVj, r, hGj, cubic and params.cubicVolume)
            sStar = scalar_t(0.5) * (sStarI + sStarJ)
            if params.interfaceClamp:
                sStar = wp.max(wp.min(sStar, scalar_t(0.5) * r), -scalar_t(0.5) * r)

            # states of the Riemann problem at the interface s*: i at +s/2, j at -s/2 on the axis n (j to i)
            uI = wp.dot(vel_i, n)
            uJ = wp.dot(vel_j, n)
            uR = uI
            uL = uJ
            rhoR = rhoi
            rhoL = rhoj
            PR = P_i
            PL = P_j
            if order >= 2:
                J_j = referenceVelocityTensor[j]
                G_j = referenceStateGradients[j]
                uR, uL = reconstructPairIncrements(params.stateLimiter, uI, uJ, wp.dot(n, matmul(J_i, x_ij)), wp.dot(n, matmul(J_j, x_ij)), sStar / r)
                rhoR, rhoL = reconstructPairIncrements(params.stateLimiter, rhoi, rhoj, wp.dot(G_i[0], x_ij), wp.dot(G_j[0], x_ij), sStar / r)
                PR, PL = reconstructPairIncrements(params.stateLimiter, P_i, P_j, wp.dot(G_i[1], x_ij), wp.dot(G_j[1], x_ij), sStar / r)
                # first-order states in compressive pairs: C (v_j - v_i) . n > min(c_i, c_j)  (Inutsuka 2002 Eq. 75)
                if params.shockSwitchC > scalar_t(0.0):
                    ci = wp.sqrt(gamma * P_i / rhoi)
                    cj = wp.sqrt(gamma * P_j / rhoj)
                    if params.shockSwitchC * (uJ - uI) > wp.min(ci, cj):
                        uR = uI
                        uL = uJ
                        rhoR = rhoi
                        rhoL = rhoj
                        PR = P_i
                        PL = P_j
            pStar, uStar = riemannStarState(params.riemannSolver, rhoL, uL, PL, rhoR, uR, PR, gamma)

            # sum over the two halves of the integration volume: grad_x W(x, sqrt2 h) = -x/h^2 W
            K = V2i * wi / (hGi * hGi) + V2j * wj / (hGj * hGj)
            acc += mj * pStar * K * x_ij
            work += mj * pStar * (uStar - wp.dot(vEnergy_i, n)) * K * r
    return acc, work


@wp.kernel
def computeInutsuka_Kernel(
    queryState: Any, referenceState: Any, domainState: domainData,
    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any,
    kernelProperties: kernelState,
    # Do not change the parameters above
    queryVelocities: wp.array(dtype=vector(length=Any, dtype=scalar_t)), referenceVelocities: wp.array(dtype=vector(length=Any, dtype=scalar_t)),  # type: ignore
    queryPressures: wp.array(dtype=scalar_t), referencePressures: wp.array(dtype=scalar_t),  # type: ignore
    queryGradRho: wp.array(dtype=vector(length=Any, dtype=scalar_t)), referenceGradRho: wp.array(dtype=vector(length=Any, dtype=scalar_t)),  # type: ignore
    queryGaussianH: wp.array(dtype=scalar_t), referenceGaussianH: wp.array(dtype=scalar_t),  # type: ignore
    queryVelocityTensor: wp.array(dtype=matrix(shape=(Any, Any), dtype=scalar_t)), referenceVelocityTensor: wp.array(dtype=matrix(shape=(Any, Any), dtype=scalar_t)),  # type: ignore
    queryStateGradients: wp.array(dtype=matrix(shape=(Any, Any), dtype=scalar_t)), referenceStateGradients: wp.array(dtype=matrix(shape=(Any, Any), dtype=scalar_t)),  # type: ignore
    queryAccelerations: wp.array(dtype=vector(length=Any, dtype=scalar_t)), halfDt: scalar_t,  # type: ignore
    params: DiffusionParameters, gamma: scalar_t, order: wp.int32, cubic: wp.bool,
    # outputs
    accel: wp.array(dtype=vector(length=Any, dtype=scalar_t)),  # type: ignore
    dudt: wp.array(dtype=scalar_t),  # type: ignore
):
    i = wp.tid()
    if i >= queryState.positions.shape[0]:
        return
    a, w = inutsuka_Func_Adjacency(
        i, domainState.dim, queryState, referenceState, domainState,
        useAdjacency, adjacencyState, gridState, gridState.numOffsets if not useAdjacency else 1,
        kernelProperties,
        queryVelocities, referenceVelocities, queryPressures, referencePressures, queryGradRho, referenceGradRho,
        queryGaussianH, referenceGaussianH,
        queryVelocityTensor, referenceVelocityTensor, queryStateGradients, referenceStateGradients,
        queryAccelerations, halfDt,
        params, gamma, order, cubic,
        accel[i],
    )
    accel[i] = a
    dudt[i] = w


_INUTSUKA = OperatorSpec(
    kernel=computeInutsuka_Kernel,
    outputs=(OutputSpec(dtype=_vecDtype, shape=ShapeOf.QUERY), OutputSpec(dtype=_scalarDtype, shape=ShapeOf.QUERY)),
    extras=(
        ExtraSpec("queryVelocities", ExtraKind.TENSOR),
        ExtraSpec("referenceVelocities", ExtraKind.TENSOR),
        ExtraSpec("queryPressures", ExtraKind.TENSOR),
        ExtraSpec("referencePressures", ExtraKind.TENSOR),
        ExtraSpec("queryGradRho", ExtraKind.TENSOR),
        ExtraSpec("referenceGradRho", ExtraKind.TENSOR),
        ExtraSpec("queryGaussianH", ExtraKind.TENSOR),
        ExtraSpec("referenceGaussianH", ExtraKind.TENSOR),
        ExtraSpec("queryVelocityTensor", ExtraKind.TENSOR),
        ExtraSpec("referenceVelocityTensor", ExtraKind.TENSOR),
        ExtraSpec("queryStateGradients", ExtraKind.TENSOR),
        ExtraSpec("referenceStateGradients", ExtraKind.TENSOR),
        ExtraSpec("queryAccelerations", ExtraKind.TENSOR),
        ExtraSpec("halfDt", ExtraKind.SCALAR),
        ExtraSpec("params", ExtraKind.SCALAR),
        ExtraSpec("gamma", ExtraKind.SCALAR),
        ExtraSpec("order", ExtraKind.SCALAR),
        ExtraSpec("cubic", ExtraKind.SCALAR),
    ),
)


def computeInutsukaWarp(
    queryParticles: ParticleState,
    operationProperties: OperationProperties,
    domain: DomainDescription,
    params: DiffusionParameters,
    gamma: float,
    gradRho: torch.Tensor,
    order: int = 2,
    cubic: bool = True,
    adjacency: Optional[Union[AdjacencyList, CompactHashMap]] = None,
    queryVelocityTensor: Optional[torch.Tensor] = None,
    queryStateGradients: Optional[torch.Tensor] = None,
    queryVelocities: Optional[torch.Tensor] = None,
    queryPressures: Optional[torch.Tensor] = None,
    queryAccelerations: Optional[torch.Tensor] = None,
    halfDt: float = 0.0,
    gaussianH: Optional[torch.Tensor] = None,
):
    """`(a_i, du_i/dt)` of Inutsuka's GSPH (with `params.timeCentredEnergy`: the energy rate at `v_i + halfDt * queryAccelerations`). `queryParticles.densities` must be the Gaussian density and `gradRho` its gradient
    (`computeGaussianDensityWarp`); the particles' `supports` are the compact kernel's, the Gaussian's `h` is `supports / 3`. Order 2 needs the velocity
    Jacobian and the `(rho, P)` gradients (`G[:, 0]` is replaced by `gradRho`: the density gradient of the same Gaussian sum) and uses `params.stateLimiter`
    (not `PairRatio`) with the states evaluated at the interface `s*`."""
    vel = queryVelocities if queryVelocities is not None else queryParticles.velocities
    P = queryPressures if queryPressures is not None else queryParticles.pressures
    dim = queryParticles.positions.shape[1]
    dummyJ = getCachedDummyTensor((1, dim, dim), dtype=queryParticles.positions.dtype, device=queryParticles.positions.device)
    dummyG = getCachedDummyTensor((1, 2, dim), dtype=queryParticles.positions.dtype, device=queryParticles.positions.device)
    if order >= 2:
        if queryVelocityTensor is None or queryStateGradients is None:
            raise ValueError('order 2 needs the velocity Jacobian and the (rho, P) gradients')
        if params.stateLimiter == StateLimiter.PairRatio.value:
            raise ValueError('the Inutsuka scheme limits its states with a 1D limiter (StateLimiter other than PairRatio)')
        G = queryStateGradients.clone()
        G[:, 0] = gradRho
    else:
        G = dummyG
    # the time-centred energy rate is evaluated in a second pass, with the accelerations of the first (halfDt = 0: the particle's own velocity)
    centred = bool(params.timeCentredEnergy) and halfDt > 0.0
    if centred and queryAccelerations is None:
        raise ValueError('a time-centred energy rate (halfDt > 0) needs the accelerations of a first pass (queryAccelerations)')
    dummyA = getCachedDummyTensor((1, dim), dtype=queryParticles.positions.dtype, device=queryParticles.positions.device)
    accs = queryAccelerations if centred else dummyA
    J = queryVelocityTensor if order >= 2 else dummyJ
    hG = gaussianH if gaussianH is not None else queryParticles.supports / 3.0
    with record_function("warpSPH[computeInutsuka]"):
        ctx = SPHContext(
            query=queryParticles, properties=operationProperties, domain=domain,
            adjacency=adjacency, reference=queryParticles,
            corrections=Corrections(volumes=(None, None), crk=None, gradH=None, renorm=None),
        )
        return launchOperator(
            _INUTSUKA, ctx,
            queryVelocities=vel, referenceVelocities=vel,
            queryPressures=P, referencePressures=P,
            queryGradRho=gradRho, referenceGradRho=gradRho,
            queryGaussianH=hG, referenceGaussianH=hG,
            queryVelocityTensor=J, referenceVelocityTensor=J,
            queryStateGradients=G, referenceStateGradients=G,
            queryAccelerations=accs, halfDt=scalar_t(halfDt),
            params=params, gamma=scalar_t(gamma), order=order, cubic=cubic,
        )
