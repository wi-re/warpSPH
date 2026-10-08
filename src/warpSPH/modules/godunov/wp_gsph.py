"""Simplified Godunov SPH pair rates, one neighbour pass (GODUNOV_SPH_PLAN layer 2).

    F_ij     = grad_i W_ij(h_i) / rho_i^2 + grad_i W_ij(h_j) / rho_j^2
    a_i      = - sum_j m_j p*_ij F_ij
    du_i/dt  = - sum_j m_j p*_ij (u*_ij - u_i) (n_ij . F_ij)

with `n_ij = (x_i - x_j) / |x_i - x_j|` (pointing from j to i), `(p*, u*)` the star state of the one-dimensional Riemann problem along
`n_ij` with the left state j and the right state i (`modules/riemann`), and `u_i = v_i . n_ij`. This is Cha & Whitworth (2003) Eqs. (13)-(14)
(Case 3), Iwasaki & Inutsuka (2011) Eq. (24), Puri & Ramachandran (2014) Eq. (15): the standard SPH `(P_i/rho_i^2 + P_j/rho_j^2)` of the
momentum equation with the pressure replaced by the Riemann star pressure, and no artificial viscosity. The tangential part of the
star velocity drops out of `(v* - v_i) . F` because `F` is parallel to `n`.

Conservation, pair by pair: `p*_ij` and `u*_ij n_ij` are invariant under swapping `i <-> j` (the solvers are mirror-symmetric) and
`F_ji = -F_ij`, so momentum is conserved and the pair's total-energy change `-m_i m_j p* (u*n.F_ij + u*n.F_ji) = 0` vanishes exactly. The
papers' time-centred `v_i + a_i dt/2` (Inutsuka Eq. 67, Puri Eq. 36) is not used: this is an ODE right-hand side, the temporal order is the
integrator's. The piecewise-constant states of the first-order scheme are the particles' own; with `order = 2` they are extrapolated to the
pair midpoint: the velocity with the limited Jacobian reconstruction of AV_PLAN Phase 3, `rho` and `P` with the limited gradients of
GODUNOV_SPH_PLAN layer 1.

This form keeps standard SPH's zeroth-order inconsistency at a density gradient (Cha et al. 2010): for constant `P` it IS the standard SPH
force. It is the cheap baseline; the convolution form of Inutsuka (2002) is the scheme (layer 3).

The neighbour list must cover `max(h_i, h_j)` (the SuperSymmetric Verlet list the compressible schemes build does).
"""

import warp as wp
from warp.types import vector, matrix
from typing import Any, Optional, Union
import torch
from warpSPHCore.profiling import record_function
from warpSPHCore import *
from ...configurations.moduleConfigurations.diffusionParameters import DiffusionParameters, StateLimiter, VelocityPairPolicy
from ..reconstruction import limitedPairPhi, pairRiemannStates, reconstructPairIncrements, reconstructPair1D
from ..riemann import riemannStarState

__all__ = ['computeGodunovWarp']


@wp.func
def godunovStates(
    params: DiffusionParameters, order: wp.int32,
    x_ij: vector(dtype=scalar_t, length=Any), n: vector(dtype=scalar_t, length=Any),  # type: ignore
    hi: scalar_t, hj: scalar_t,
    vel_i: vector(length=Any, dtype=scalar_t), vel_j: vector(length=Any, dtype=scalar_t),  # type: ignore
    rhoi: scalar_t, rhoj: scalar_t, P_i: scalar_t, P_j: scalar_t,
    J_i: matrix(shape=(Any, Any), dtype=scalar_t), referenceVelocityTensor: wp.array(dtype=matrix(shape=(Any, Any), dtype=scalar_t)),  # type: ignore
    G_i: matrix(shape=(Any, Any), dtype=scalar_t), referenceStateGradients: wp.array(dtype=matrix(shape=(Any, Any), dtype=scalar_t)),  # type: ignore
    j: wp.int32, kernel_int: wp.int32, dim: wp.int32, gamma: scalar_t,
):
    """`(u_R, u_L, rho_R, rho_L, P_R, P_L)`: the states of the pair's Riemann problem along `n` (right = i, left = j). First order: the particles' own
    values. Second order: extrapolated to the pair midpoint -- `StateLimiter.PairRatio`: the AV_PLAN Phase 3 limiter of the velocity Jacobian ratio and
    the layer-1 `(rho, P)` reconstruction; the others: the 1D limiters of Murante / Iwasaki / Inutsuka on the normal velocity (projected Jacobian)
    and, with `riemannReconstruction`, `rho` and `P` (projected gradients), then the optional first-order shock switch (Inutsuka Eq. 75)."""
    uI = wp.dot(vel_i, n)
    uJ = wp.dot(vel_j, n)
    if order < 2:
        return uI, uJ, rhoi, rhoj, P_i, P_j
    J_j = referenceVelocityTensor[j]
    if params.stateLimiter == wp.static(StateLimiter.PairRatio.value):
        phi = limitedPairPhi(
            x_ij, hi, hj, vel_i, vel_j, J_i, J_j, kernel_int, dim,
            True, True, params.reconstructionEtaCrit, params.reconstructionEtaFold, params.limiterType)
        phi = wp.max(wp.min(phi, scalar_t(1.0)), scalar_t(0.0))
        vR = vel_i - scalar_t(0.5) * phi * matmul(J_i, x_ij)
        vL = vel_j + scalar_t(0.5) * phi * matmul(J_j, x_ij)
        rR, rL, pR_, pL_ = pairRiemannStates(
            params.riemannReconstruction, rhoi, rhoj, P_i, P_j, G_i, referenceStateGradients, j, x_ij, hi, hj,
            kernel_int, dim, params.reconstructionEtaCrit, params.reconstructionEtaFold, params.limiterType)
        use = rR >= scalar_t(0.0)
        return wp.dot(vR, n), wp.dot(vL, n), wp.where(use, rR, rhoi), wp.where(use, rL, rhoj), wp.where(use, pR_, P_i), wp.where(use, pL_, P_j)
    uR, uL = reconstructPairIncrements(params.stateLimiter, uI, uJ, wp.dot(n, matmul(J_i, x_ij)), wp.dot(n, matmul(J_j, x_ij)))
    rhoR = rhoi
    rhoL = rhoj
    PR = P_i
    PL = P_j
    if params.riemannReconstruction:
        G_j = referenceStateGradients[j]
        rhoR, rhoL = reconstructPair1D(params.stateLimiter, rhoi, rhoj, G_i[0], G_j[0], x_ij)
        PR, PL = reconstructPair1D(params.stateLimiter, P_i, P_j, G_i[1], G_j[1], x_ij)
    # first-order states in compressive pairs (a shock surface): C (v_j - v_i) . n > min(c_i, c_j)
    if params.shockSwitchC > scalar_t(0.0):
        ci = wp.sqrt(gamma * P_i / rhoi)
        cj = wp.sqrt(gamma * P_j / rhoj)
        if params.shockSwitchC * (uJ - uI) > wp.min(ci, cj):
            return uI, uJ, rhoi, rhoj, P_i, P_j
    return uR, uL, rhoR, rhoL, PR, PL


@wp.func
def godunov_Func_i(
    i: wp.int32,
    xi: vector(dtype=scalar_t, length=Any), hi: scalar_t, mi: scalar_t, rhoi: scalar_t,  # type: ignore
    referenceState: Any,
    domainState: domainData,
    kernelProperties: kernelState,
    dim: wp.int32,
    beginIndex: wp.int32, numIndices: wp.int32, offsetArray: wp.array(dtype=wp.int64),  # type: ignore
    ki: wp.int32, referenceKinds: wp.array(dtype=wp.int32),  # type: ignore
    vel_i: vector(length=Any, dtype=scalar_t), referenceVelocities: wp.array(dtype=vector(length=Any, dtype=scalar_t)),  # type: ignore
    P_i: scalar_t, referencePressures: wp.array(dtype=scalar_t),  # type: ignore
    J_i: matrix(shape=(Any, Any), dtype=scalar_t), referenceVelocityTensor: wp.array(dtype=matrix(shape=(Any, Any), dtype=scalar_t)),  # type: ignore
    G_i: matrix(shape=(Any, Any), dtype=scalar_t), referenceStateGradients: wp.array(dtype=matrix(shape=(Any, Any), dtype=scalar_t)),  # type: ignore
    params: DiffusionParameters, gamma: scalar_t, order: wp.int32,
    vEnergy_i: vector(length=Any, dtype=scalar_t),  # type: ignore
    accelValue: Any, workValue: scalar_t,
):
    acc = zero_like_warp(accelValue)
    work = workValue
    for neighborIndex in range(numIndices):
        jj = beginIndex + neighborIndex
        j = wp.int32(offsetArray[jj])
        if kernelProperties.operationMode != wp.static(OperationDirection.TrueAllToToAll.value):
            if not checkDirectionality_j(referenceKinds[j], kernelProperties.operationMode):
                continue
        if i == j:
            continue
        xj, hj, mj, rhoj, kj = getParticle(referenceState, j)
        vel_j = referenceVelocities[j]
        P_j = referencePressures[j]
        x_ij = computeDistanceVec(xi, xj, domainState)
        r = safe_sqrt(wp.dot(x_ij, x_ij))
        n = x_ij / (r + scalar_t(1.0e-14) * hi)

        # left state j, right state i along n: particle values (first order) or midpoint extrapolations (second order)
        uR, uL, rhoR, rhoL, PR, PL = godunovStates(
            params, order, x_ij, n, hi, hj, vel_i, vel_j, rhoi, rhoj, P_i, P_j, J_i, referenceVelocityTensor,
            G_i, referenceStateGradients, j, kernelProperties.kernelFunction, dim, gamma)
        pStar, uStar = riemannStarState(params.riemannSolver, rhoL, uL, PL, rhoR, uR, PR, gamma)

        gradWi = sphKernelGradient(xi, xj, hi, hi, kernelProperties, domainState)
        gradWj = sphKernelGradient(xi, xj, hj, hj, kernelProperties, domainState)
        F = gradWi / (rhoi * rhoi) + gradWj / (rhoj * rhoj)
        acc -= mj * pStar * F
        work -= mj * pStar * (uStar - wp.dot(vEnergy_i, n)) * wp.dot(n, F)
    return acc, work


@wp.func
def godunov_Func_Adjacency(
    i: wp.int32, dim: wp.int32,
    queryState: Any, referenceState: Any,
    domainState: domainData,
    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData, numOffsets: wp.int32,
    kernelProperties: kernelState,
    queryVelocities: wp.array(dtype=vector(length=Any, dtype=scalar_t)), referenceVelocities: wp.array(dtype=vector(length=Any, dtype=scalar_t)),  # type: ignore
    queryPressures: wp.array(dtype=scalar_t), referencePressures: wp.array(dtype=scalar_t),  # type: ignore
    queryVelocityTensor: wp.array(dtype=matrix(shape=(Any, Any), dtype=scalar_t)), referenceVelocityTensor: wp.array(dtype=matrix(shape=(Any, Any), dtype=scalar_t)),  # type: ignore
    queryStateGradients: wp.array(dtype=matrix(shape=(Any, Any), dtype=scalar_t)), referenceStateGradients: wp.array(dtype=matrix(shape=(Any, Any), dtype=scalar_t)),  # type: ignore
    queryAccelerations: wp.array(dtype=vector(length=Any, dtype=scalar_t)), halfDt: scalar_t,  # type: ignore
    params: DiffusionParameters, gamma: scalar_t, order: wp.int32,
    accelValue: Any,
):
    xi, hi, mi, rhoi, ki = getParticle(queryState, i)
    acc = zero_like_warp(accelValue)
    work = scalar_t(0.0)
    if kernelProperties.operationMode != wp.static(OperationDirection.TrueAllToToAll.value):
        if not checkDirectionality_i(ki, kernelProperties.operationMode):
            return acc, work
    # the velocity of the energy equation: the particle's own, or the time-centred one (Inutsuka 2002 Eq. 67)
    vEnergy_i = queryVelocities[i]
    if params.timeCentredEnergy and halfDt > scalar_t(0.0):
        vEnergy_i = queryVelocities[i] + halfDt * queryAccelerations[i]
    J_i = queryVelocityTensor[0]
    G_i = queryStateGradients[0]
    if order >= 2:
        J_i = queryVelocityTensor[i]
        if params.riemannReconstruction:
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
        a_o, w_o = godunov_Func_i(
            i, xi, hi, mi, rhoi, referenceState, domainState, kernelProperties, dim,
            beginIndex, numIndices, adjacencyState.neighborList if useAdjacency else gridState.sortIndex,
            ki, referenceState.kinds,
            queryVelocities[i], referenceVelocities,
            queryPressures[i], referencePressures,
            J_i, referenceVelocityTensor, G_i, referenceStateGradients,
            params, gamma, order, vEnergy_i,
            accelValue, scalar_t(0.0),
        )
        acc += a_o
        work += w_o
    return acc, work


@wp.kernel
def computeGodunov_Kernel(
    queryState: Any, referenceState: Any, domainState: domainData,
    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any,
    kernelProperties: kernelState,
    # Do not change the parameters above
    queryVelocities: wp.array(dtype=vector(length=Any, dtype=scalar_t)), referenceVelocities: wp.array(dtype=vector(length=Any, dtype=scalar_t)),  # type: ignore
    queryPressures: wp.array(dtype=scalar_t), referencePressures: wp.array(dtype=scalar_t),  # type: ignore
    queryVelocityTensor: wp.array(dtype=matrix(shape=(Any, Any), dtype=scalar_t)), referenceVelocityTensor: wp.array(dtype=matrix(shape=(Any, Any), dtype=scalar_t)),  # type: ignore
    queryStateGradients: wp.array(dtype=matrix(shape=(Any, Any), dtype=scalar_t)), referenceStateGradients: wp.array(dtype=matrix(shape=(Any, Any), dtype=scalar_t)),  # type: ignore
    queryAccelerations: wp.array(dtype=vector(length=Any, dtype=scalar_t)), halfDt: scalar_t,  # type: ignore
    params: DiffusionParameters, gamma: scalar_t, order: wp.int32,
    # outputs
    accel: wp.array(dtype=vector(length=Any, dtype=scalar_t)),  # type: ignore
    dudt: wp.array(dtype=scalar_t),  # type: ignore
):
    i = wp.tid()
    if i >= queryState.positions.shape[0]:
        return
    a, w = godunov_Func_Adjacency(
        i, domainState.dim, queryState, referenceState, domainState,
        useAdjacency, adjacencyState, gridState, gridState.numOffsets if not useAdjacency else 1,
        kernelProperties,
        queryVelocities, referenceVelocities, queryPressures, referencePressures,
        queryVelocityTensor, referenceVelocityTensor, queryStateGradients, referenceStateGradients,
        queryAccelerations, halfDt,
        params, gamma, order,
        accel[i],
    )
    accel[i] = a
    dudt[i] = w


def _accelDtype(ctx, extras):
    return castTorchToWarpAsBuiltins(ctx.query.positions).dtype


def _scalarDtype(ctx, extras):
    return castTorchToWarpAsBuiltins(ctx.query.densities).dtype


_GODUNOV = OperatorSpec(
    kernel=computeGodunov_Kernel,
    outputs=(OutputSpec(dtype=_accelDtype, shape=ShapeOf.QUERY),
             OutputSpec(dtype=_scalarDtype, shape=ShapeOf.QUERY)),
    extras=(
        ExtraSpec("queryVelocities", ExtraKind.TENSOR),
        ExtraSpec("referenceVelocities", ExtraKind.TENSOR),
        ExtraSpec("queryPressures", ExtraKind.TENSOR),
        ExtraSpec("referencePressures", ExtraKind.TENSOR),
        ExtraSpec("queryVelocityTensor", ExtraKind.TENSOR),
        ExtraSpec("referenceVelocityTensor", ExtraKind.TENSOR),
        ExtraSpec("queryStateGradients", ExtraKind.TENSOR),
        ExtraSpec("referenceStateGradients", ExtraKind.TENSOR),
        ExtraSpec("queryAccelerations", ExtraKind.TENSOR),
        ExtraSpec("halfDt", ExtraKind.SCALAR),
        ExtraSpec("params", ExtraKind.SCALAR),
        ExtraSpec("gamma", ExtraKind.SCALAR),
        ExtraSpec("order", ExtraKind.SCALAR),
    ),
)


def computeGodunovWarp(
    queryParticles: ParticleState,
    operationProperties: OperationProperties,
    domain: DomainDescription,
    params: DiffusionParameters,
    gamma: float,
    order: int = 1,
    adjacency: Optional[Union[AdjacencyList, CompactHashMap]] = None,
    queryVelocityTensor: Optional[torch.Tensor] = None,
    queryStateGradients: Optional[torch.Tensor] = None,
    queryVelocities: Optional[torch.Tensor] = None,
    queryPressures: Optional[torch.Tensor] = None,
    referenceParticles: Optional[ParticleState] = None,
    queryAccelerations: Optional[torch.Tensor] = None,
    halfDt: float = 0.0,
):
    """`(a_i, du_i/dt)` of the simplified Godunov SPH pair. With `params.timeCentredEnergy` the energy rate uses `v_i + halfDt * queryAccelerations`
    (the accelerations of a first call; Inutsuka 2002 Eq. 67). `order` 1: the particles' own states; 2: extrapolated to the pair midpoint with
    the velocity Jacobian `queryVelocityTensor` (required) and, if `params.riemannReconstruction`, the `(rho, P)` gradients
    `queryStateGradients` (required then). `params` supplies `riemannSolver`, `limiterType` and the taper constants (resolved, see
    `resolveReconstructionLimiter`)."""
    referenceParticles = referenceParticles if referenceParticles is not None else queryParticles
    vel = queryVelocities if queryVelocities is not None else queryParticles.velocities
    P = queryPressures if queryPressures is not None else queryParticles.pressures
    dim = queryParticles.positions.shape[1]
    dummyJ = getCachedDummyTensor((1, dim, dim), dtype=queryParticles.positions.dtype, device=queryParticles.positions.device)
    dummyG = getCachedDummyTensor((1, 2, dim), dtype=queryParticles.positions.dtype, device=queryParticles.positions.device)
    if order >= 2:
        if queryVelocityTensor is None:
            raise ValueError('order 2 needs the velocity Jacobian (queryVelocityTensor, see reconstruction.computeVelocityJacobian)')
        if params.riemannReconstruction and queryStateGradients is None:
            raise ValueError('riemannReconstruction needs the (rho, P) gradients (queryStateGradients, see reconstruction.computeStateGradients)')
    # the time-centred energy rate is evaluated in a second pass, with the accelerations of the first (halfDt = 0: the particle's own velocity)
    centred = bool(params.timeCentredEnergy) and halfDt > 0.0
    if centred and queryAccelerations is None:
        raise ValueError('a time-centred energy rate (halfDt > 0) needs the accelerations of a first pass (queryAccelerations)')
    dummyA = getCachedDummyTensor((1, dim), dtype=queryParticles.positions.dtype, device=queryParticles.positions.device)
    accs = queryAccelerations if centred else dummyA
    J = queryVelocityTensor if order >= 2 else dummyJ
    G = queryStateGradients if (order >= 2 and params.riemannReconstruction) else dummyG
    with record_function("warpSPH[computeGodunov]"):
        ctx = SPHContext(
            query=queryParticles, properties=operationProperties, domain=domain,
            adjacency=adjacency, reference=referenceParticles,
            corrections=Corrections(volumes=(None, None), crk=None, gradH=None, renorm=None),
        )
        return launchOperator(
            _GODUNOV, ctx,
            queryVelocities=vel, referenceVelocities=vel,
            queryPressures=P, referencePressures=P,
            queryVelocityTensor=J, referenceVelocityTensor=J,
            queryStateGradients=G, referenceStateGradients=G,
            queryAccelerations=accs, halfDt=scalar_t(halfDt),
            params=params, gamma=scalar_t(gamma), order=order,
        )
