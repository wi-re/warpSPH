"""Warp kernels for the pairwise parts of Read & Hayfield (2012), SPHS: the relaxation signal
velocity and the entropy dissipation.

Both loops were once written in torch on the host from `adjacency.i/j` with raw position
differences `x_i - x_j` (no minimum image on a periodic domain, so a pair across the boundary had
|x_ij| ~ the domain size: wrong direction, wrong `K_ij` cut-off) and a hand-copied B7 / Wendland2
`dW/dr` that silently fell back to Wendland2 for any other kernel. Here they run on the device and
use the same core pieces as every other operator: `computeDistanceVec` (minimum image),
`sphKernelGradient` (the kernel, support mode and normalisation the rest of the scheme uses) and
`computePairwiseSupport`.

* `computeReadHayfieldVmaxWarp`: `v_max,i = max_j (c_i + c_j - 3 w_ij)`, `w_ij = v_ij . x_ij / r_ij`
  (eqs. 24-25; the relaxation time is `tau = h / v_max`). The self pair is a neighbour (`w = 0`), so
  `v_max >= 2 c_i`. A loop-carried `wp.max` silently zeroes its adjoint under warp-lang 1.15, so this
  uses the argmax-then-re-evaluate split of `wp_vsig.py` (read its docstring).
* `computeReadHayfieldEntropyDissipationWarp`: eq. (33),

      dA_i/dt = sum_j (m_j / rho_ij) alpha_ij v^p_sig,ij L_ij [A_i - A_j (rho_j/rho_i)^(gamma-1)] K_ij,

  `K_ij = r_hat_ij . grad_i W_ij`, `v^p_sig = max(c_i + c_j - 3 w_ij, 0)` (eq. 34),
  `L_ij = |P_i - P_j| / (P_i + P_j)` (eq. 35). A row sum per particle in a fixed neighbour order, so
  it is deterministic (no atomics).
"""

import warp as wp
from warp.types import vector, matrix
from typing import Any
import torch
from warpSPHCore.profiling import record_function
from typing import Optional, Union
from warpSPHCore import *

__all__ = ['computeReadHayfieldVmaxWarp', 'computeReadHayfieldEntropyDissipationWarp']


# ------------------------------------------------------------------------------------------------
# v_max: argmax pass + single differentiable re-evaluation
# ------------------------------------------------------------------------------------------------

@wp.func
def rhVsig_valueAt(
    xi: vector(dtype=scalar_t, length=Any), hi: scalar_t,  # type: ignore
    referenceState: Any, j: wp.int32,
    domainState: domainData,
    vel_i: vector(length=Any, dtype=scalar_t), referenceVelocities: wp.array(dtype=vector(length=Any, dtype=scalar_t)),  # type: ignore
    cs_i: scalar_t, referenceCs: wp.array(dtype=scalar_t),  # type: ignore
) -> scalar_t:
    xj, hj, mj, rhoj, kj = getParticle(referenceState, j)
    x_ij = computeDistanceVec(xi, xj, domainState)
    r_ij = safe_sqrt(wp.dot(x_ij, x_ij))
    w_ij = wp.dot(vel_i - referenceVelocities[j], x_ij) / (r_ij + scalar_t(1.0e-14) * hi)
    return cs_i + referenceCs[j] - scalar_t(3.0) * w_ij


@wp.func
def rhVmax_Func_i_argmax(
    i: wp.int32,
    xi: vector(dtype=scalar_t, length=Any), hi: scalar_t,  # type: ignore
    referenceState: Any,
    domainState: domainData,
    kernelProperties: kernelState,
    beginIndex: wp.int32, numIndices: wp.int32, offsetArray: wp.array(dtype=wp.int64),  # type: ignore
    ki: wp.int32, referenceKinds: wp.array(dtype=wp.int32),  # type: ignore
    vel_i: vector(length=Any, dtype=scalar_t), referenceVelocities: wp.array(dtype=vector(length=Any, dtype=scalar_t)),  # type: ignore
    cs_i: scalar_t, referenceCs: wp.array(dtype=scalar_t),  # type: ignore
):
    found = wp.bool(False)
    bestVal = scalar_t(0.0)
    bestJ = wp.int32(0)
    for neighborIndex in range(numIndices):
        jj = beginIndex + neighborIndex
        j = wp.int32(offsetArray[jj])
        if kernelProperties.operationMode != wp.static(OperationDirection.TrueAllToToAll.value):
            if not checkDirectionality_j(referenceKinds[j], kernelProperties.operationMode):
                continue
        xj, hj, mj, rhoj, kj = getParticle(referenceState, j)
        x_ij = computeDistanceVec(xi, xj, domainState)
        r_ij = safe_sqrt(wp.dot(x_ij, x_ij))
        # exact compact-support filter (the grid path returns whole cells)
        hij = computePairwiseSupport(hi, hj, kernelProperties.supportMode)
        if r_ij >= hij and i != j:
            continue
        w_ij = wp.dot(vel_i - referenceVelocities[j], x_ij) / (r_ij + scalar_t(1.0e-14) * hi)
        vsig = cs_i + referenceCs[j] - scalar_t(3.0) * w_ij
        if (not found) or vsig > bestVal:
            bestVal = vsig
            bestJ = j
            found = wp.bool(True)
    return found, bestVal, bestJ


@wp.func
def rhVmax_Func_Adjacency_argmax(
    i: wp.int32,
    queryState: Any, referenceState: Any,
    domainState: domainData,
    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData, numOffsets: wp.int32,
    kernelProperties: kernelState,
    queryVelocities: wp.array(dtype=vector(length=Any, dtype=scalar_t)), referenceVelocities: wp.array(dtype=vector(length=Any, dtype=scalar_t)),  # type: ignore
    queryCs: wp.array(dtype=scalar_t), referenceCs: wp.array(dtype=scalar_t),  # type: ignore
):
    xi, hi, mi, rhoi, ki = getParticle(queryState, i)
    if kernelProperties.operationMode != wp.static(OperationDirection.TrueAllToToAll.value):
        if not checkDirectionality_i(ki, kernelProperties.operationMode):
            return wp.bool(False), wp.int32(0)
    vel_i = queryVelocities[i]
    cs_i = queryCs[i]

    globalFound = wp.bool(False)
    globalBestVal = scalar_t(0.0)
    globalBestJ = wp.int32(0)
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
        found, bestVal, bestJ = rhVmax_Func_i_argmax(
            i, xi, hi, referenceState, domainState, kernelProperties,
            beginIndex, numIndices, adjacencyState.neighborList if useAdjacency else gridState.sortIndex,
            ki, referenceState.kinds,
            vel_i, referenceVelocities, cs_i, referenceCs,
        )
        if found and ((not globalFound) or bestVal > globalBestVal):
            globalBestVal = bestVal
            globalBestJ = bestJ
            globalFound = wp.bool(True)
    return globalFound, globalBestJ


@wp.kernel
def computeReadHayfieldVmax_Kernel(
    queryState: Any, referenceState: Any, domainState: domainData,
    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any,
    kernelProperties: kernelState,
    # Do not change the parameters above
    queryVelocities: wp.array(dtype=vector(length=Any, dtype=scalar_t)), referenceVelocities: wp.array(dtype=vector(length=Any, dtype=scalar_t)),  # type: ignore
    queryCs: wp.array(dtype=scalar_t), referenceCs: wp.array(dtype=scalar_t),  # type: ignore
    # The last parameter is always the output array and should not be changed
    outputValues: wp.array(dtype=scalar_t)  # type: ignore
):
    i = wp.tid()
    if i >= queryState.positions.shape[0]:
        return
    found, bestJ = rhVmax_Func_Adjacency_argmax(
        i, queryState, referenceState, domainState,
        useAdjacency, adjacencyState, gridState, gridState.numOffsets if not useAdjacency else 1,
        kernelProperties,
        queryVelocities, referenceVelocities, queryCs, referenceCs,
    )
    xi, hi, mi, rhoi, ki = getParticle(queryState, i)
    if found:
        # the one differentiable evaluation, outside any loop (see wp_vsig.py)
        outputValues[i] = rhVsig_valueAt(
            xi, hi, referenceState, bestJ, domainState,
            queryVelocities[i], referenceVelocities, queryCs[i], referenceCs)
    else:
        outputValues[i] = queryCs[i]


def _rhDtype(ctx, extras):
    return castTorchToWarpAsBuiltins(ctx.query.densities).dtype


_RH_VMAX = OperatorSpec(
    kernel=computeReadHayfieldVmax_Kernel,
    outputs=(OutputSpec(dtype=_rhDtype, shape=ShapeOf.QUERY),),
    extras=(
        ExtraSpec("queryVelocities", ExtraKind.TENSOR),
        ExtraSpec("referenceVelocities", ExtraKind.TENSOR),
        ExtraSpec("queryCs", ExtraKind.TENSOR),
        ExtraSpec("referenceCs", ExtraKind.TENSOR),
    ),
)


def computeReadHayfieldVmaxWarp(
    queryParticles: ParticleState,
    operationProperties: OperationProperties,
    domain: DomainDescription,
    queryVelocities: Optional[torch.Tensor] = None, referenceVelocities: Optional[torch.Tensor] = None,
    queryCs: Optional[torch.Tensor] = None, referenceCs: Optional[torch.Tensor] = None,
    adjacency: Optional[Union[AdjacencyList, CompactHashMap]] = None,
    referenceParticles: Optional[ParticleState] = None,
):
    """`v_max,i = max_j (c_i + c_j - 3 w_ij)` over the neighbours (self included), minimum image."""
    referenceParticles = referenceParticles if referenceParticles is not None else queryParticles
    queryVelocities = queryVelocities if queryVelocities is not None else queryParticles.velocities
    queryCs = queryCs if queryCs is not None else queryParticles.soundspeeds
    referenceVelocities = referenceVelocities if referenceVelocities is not None else queryVelocities
    referenceCs = referenceCs if referenceCs is not None else queryCs
    with record_function("warpSPH[computeReadHayfieldVmax]"):
        ctx = SPHContext(
            query=queryParticles, properties=operationProperties, domain=domain,
            adjacency=adjacency, reference=referenceParticles,
            corrections=Corrections(volumes=(None, None), crk=None, gradH=None, renorm=None),
        )
        return launchOperator(
            _RH_VMAX, ctx,
            queryVelocities=queryVelocities, referenceVelocities=referenceVelocities,
            queryCs=queryCs, referenceCs=referenceCs,
        )


# ------------------------------------------------------------------------------------------------
# entropy dissipation, eq. (33)
# ------------------------------------------------------------------------------------------------

@wp.func
def rhEntropy_Func_i(
    i: wp.int32,
    xi: vector(dtype=scalar_t, length=Any), hi: scalar_t, mi: scalar_t, rhoi: scalar_t,  # type: ignore
    referenceState: Any,
    domainState: domainData,
    kernelProperties: kernelState,
    beginIndex: wp.int32, numIndices: wp.int32, offsetArray: wp.array(dtype=wp.int64),  # type: ignore
    ki: wp.int32, referenceKinds: wp.array(dtype=wp.int32),  # type: ignore
    vel_i: vector(length=Any, dtype=scalar_t), referenceVelocities: wp.array(dtype=vector(length=Any, dtype=scalar_t)),  # type: ignore
    cs_i: scalar_t, referenceCs: wp.array(dtype=scalar_t),  # type: ignore
    alpha_i: scalar_t, referenceAlphas: wp.array(dtype=scalar_t),  # type: ignore
    P_i: scalar_t, referencePressures: wp.array(dtype=scalar_t),  # type: ignore
    A_i: scalar_t, referenceEntropies: wp.array(dtype=scalar_t),  # type: ignore
    gamma: scalar_t,
    outputValue: Any,  # type: ignore
):
    out = zero_like_warp(outputValue)
    for neighborIndex in range(numIndices):
        jj = beginIndex + neighborIndex
        j = wp.int32(offsetArray[jj])
        if kernelProperties.operationMode != wp.static(OperationDirection.TrueAllToToAll.value):
            if not checkDirectionality_j(referenceKinds[j], kernelProperties.operationMode):
                continue
        if i == j:
            continue
        xj, hj, mj, rhoj, kj = getParticle(referenceState, j)
        x_ij = computeDistanceVec(xi, xj, domainState)
        r_ij = safe_sqrt(wp.dot(x_ij, x_ij))
        hij = computePairwiseSupport(hi, hj, kernelProperties.supportMode)
        if r_ij >= hij or r_ij <= scalar_t(0.0):
            continue

        cs_j = referenceCs[j]
        w_ij = wp.dot(vel_i - referenceVelocities[j], x_ij) / r_ij
        # eq. (34): positive-definite signal velocity, larger for approaching pairs
        vsigp = cs_i + cs_j - scalar_t(3.0) * w_ij
        if vsigp <= scalar_t(0.0):
            continue
        P_j = referencePressures[j]
        L_ij = wp.abs(P_i - P_j) / (P_i + P_j + scalar_t(1.0e-14))                    # eq. (35)
        gradw = sphKernelGradient(xi, xj, hi, hj, kernelProperties, domainState)
        K_ij = wp.dot(gradw, x_ij) / r_ij                                             # r_hat . grad_i W_ij
        rho_ij = scalar_t(0.5) * (rhoi + rhoj)
        alpha_ij = scalar_t(0.5) * (alpha_i + referenceAlphas[j])
        ratio = wp.pow(rhoj / wp.max(rhoi, scalar_t(1.0e-14)), gamma - scalar_t(1.0))
        # eq. (33): the density ratio multiplies A_j INSIDE the bracket (energy-conserving pair form)
        out += (mj / rho_ij) * alpha_ij * vsigp * L_ij * (A_i - referenceEntropies[j] * ratio) * K_ij
    return out


@wp.func
def rhEntropy_Func_Adjacency(
    i: wp.int32,
    queryState: Any, referenceState: Any,
    domainState: domainData,
    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData, numOffsets: wp.int32,
    kernelProperties: kernelState,
    queryVelocities: wp.array(dtype=vector(length=Any, dtype=scalar_t)), referenceVelocities: wp.array(dtype=vector(length=Any, dtype=scalar_t)),  # type: ignore
    queryCs: wp.array(dtype=scalar_t), referenceCs: wp.array(dtype=scalar_t),  # type: ignore
    queryAlphas: wp.array(dtype=scalar_t), referenceAlphas: wp.array(dtype=scalar_t),  # type: ignore
    queryPressures: wp.array(dtype=scalar_t), referencePressures: wp.array(dtype=scalar_t),  # type: ignore
    queryEntropies: wp.array(dtype=scalar_t), referenceEntropies: wp.array(dtype=scalar_t),  # type: ignore
    gamma: scalar_t,
    outputValue: Any,  # type: ignore
):
    xi, hi, mi, rhoi, ki = getParticle(queryState, i)
    if kernelProperties.operationMode != wp.static(OperationDirection.TrueAllToToAll.value):
        if not checkDirectionality_i(ki, kernelProperties.operationMode):
            return zero_like_warp(outputValue)
    out = zero_like_warp(outputValue)
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
        out += rhEntropy_Func_i(
            i, xi, hi, mi, rhoi, referenceState, domainState, kernelProperties,
            beginIndex, numIndices, adjacencyState.neighborList if useAdjacency else gridState.sortIndex,
            ki, referenceState.kinds,
            queryVelocities[i], referenceVelocities,
            queryCs[i], referenceCs,
            queryAlphas[i], referenceAlphas,
            queryPressures[i], referencePressures,
            queryEntropies[i], referenceEntropies,
            gamma, outputValue,
        )
    return out


@wp.kernel
def computeReadHayfieldEntropy_Kernel(
    queryState: Any, referenceState: Any, domainState: domainData,
    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any,
    kernelProperties: kernelState,
    # Do not change the parameters above
    queryVelocities: wp.array(dtype=vector(length=Any, dtype=scalar_t)), referenceVelocities: wp.array(dtype=vector(length=Any, dtype=scalar_t)),  # type: ignore
    queryCs: wp.array(dtype=scalar_t), referenceCs: wp.array(dtype=scalar_t),  # type: ignore
    queryAlphas: wp.array(dtype=scalar_t), referenceAlphas: wp.array(dtype=scalar_t),  # type: ignore
    queryPressures: wp.array(dtype=scalar_t), referencePressures: wp.array(dtype=scalar_t),  # type: ignore
    queryEntropies: wp.array(dtype=scalar_t), referenceEntropies: wp.array(dtype=scalar_t),  # type: ignore
    gamma: scalar_t,
    # The last parameter is always the output array and should not be changed
    outputValues: wp.array(dtype=scalar_t)  # type: ignore
):
    i = wp.tid()
    if i >= queryState.positions.shape[0]:
        return
    outputValues[i] = rhEntropy_Func_Adjacency(
        i, queryState, referenceState, domainState,
        useAdjacency, adjacencyState, gridState, gridState.numOffsets if not useAdjacency else 1,
        kernelProperties,
        queryVelocities, referenceVelocities, queryCs, referenceCs,
        queryAlphas, referenceAlphas, queryPressures, referencePressures,
        queryEntropies, referenceEntropies,
        gamma, zero_like_warp(outputValues),
    )


_RH_ENTROPY = OperatorSpec(
    kernel=computeReadHayfieldEntropy_Kernel,
    outputs=(OutputSpec(dtype=_rhDtype, shape=ShapeOf.QUERY),),
    extras=(
        ExtraSpec("queryVelocities", ExtraKind.TENSOR),
        ExtraSpec("referenceVelocities", ExtraKind.TENSOR),
        ExtraSpec("queryCs", ExtraKind.TENSOR),
        ExtraSpec("referenceCs", ExtraKind.TENSOR),
        ExtraSpec("queryAlphas", ExtraKind.TENSOR),
        ExtraSpec("referenceAlphas", ExtraKind.TENSOR),
        ExtraSpec("queryPressures", ExtraKind.TENSOR),
        ExtraSpec("referencePressures", ExtraKind.TENSOR),
        ExtraSpec("queryEntropies", ExtraKind.TENSOR),
        ExtraSpec("referenceEntropies", ExtraKind.TENSOR),
        ExtraSpec("gamma", ExtraKind.SCALAR),
    ),
)


def computeReadHayfieldEntropyDissipationWarp(
    queryParticles: ParticleState,
    operationProperties: OperationProperties,
    domain: DomainDescription,
    gamma: float,
    queryAlphas: torch.Tensor,
    queryVelocities: Optional[torch.Tensor] = None, referenceVelocities: Optional[torch.Tensor] = None,
    queryCs: Optional[torch.Tensor] = None, referenceCs: Optional[torch.Tensor] = None,
    queryPressures: Optional[torch.Tensor] = None, referencePressures: Optional[torch.Tensor] = None,
    queryEntropies: Optional[torch.Tensor] = None, referenceEntropies: Optional[torch.Tensor] = None,
    referenceAlphas: Optional[torch.Tensor] = None,
    adjacency: Optional[Union[AdjacencyList, CompactHashMap]] = None,
    referenceParticles: Optional[ParticleState] = None,
):
    """Eq. (33) entropy rate `dA_i/dt` per particle (not yet converted to an energy rate). Every
    `reference*` tensor defaults to its `query*` counterpart (the same particle set)."""
    referenceParticles = referenceParticles if referenceParticles is not None else queryParticles
    queryVelocities = queryVelocities if queryVelocities is not None else queryParticles.velocities
    queryCs = queryCs if queryCs is not None else queryParticles.soundspeeds
    queryPressures = queryPressures if queryPressures is not None else queryParticles.pressures
    queryEntropies = queryEntropies if queryEntropies is not None else queryParticles.entropies
    referenceVelocities = referenceVelocities if referenceVelocities is not None else queryVelocities
    referenceCs = referenceCs if referenceCs is not None else queryCs
    referencePressures = referencePressures if referencePressures is not None else queryPressures
    referenceEntropies = referenceEntropies if referenceEntropies is not None else queryEntropies
    referenceAlphas = referenceAlphas if referenceAlphas is not None else queryAlphas
    with record_function("warpSPH[computeReadHayfieldEntropy]"):
        ctx = SPHContext(
            query=queryParticles, properties=operationProperties, domain=domain,
            adjacency=adjacency, reference=referenceParticles,
            corrections=Corrections(volumes=(None, None), crk=None, gradH=None, renorm=None),
        )
        return launchOperator(
            _RH_ENTROPY, ctx,
            queryVelocities=queryVelocities, referenceVelocities=referenceVelocities,
            queryCs=queryCs, referenceCs=referenceCs,
            queryAlphas=queryAlphas, referenceAlphas=referenceAlphas,
            queryPressures=queryPressures, referencePressures=referencePressures,
            queryEntropies=queryEntropies, referenceEntropies=referenceEntropies,
            gamma=scalar_t(gamma),
        )
