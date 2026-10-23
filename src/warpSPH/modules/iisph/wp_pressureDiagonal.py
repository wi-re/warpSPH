"""Base diagonal of continuity composed with pressure acceleration, without dt squared."""

from typing import Any
import torch
import warp as wp
from warp.types import vector
from warpSPHCore import (
    ParticleState, OperationProperties, DomainDescription, SPHContext,
    OperatorSpec, OutputSpec, ExtraSpec, ExtraKind, ShapeOf, launchOperator,
    domainData, adjacencyData, gridData, kernelState, getParticle, checkOffset,
    zero_like_warp, computeDistanceVec,
    sphKernelGradient_ij,
)
from warpSPHCore.type_config import scalar_t, index_t, offset_t


@wp.func
def computeIISPHPressureDiagonal_Func_i(
    i: wp.int32,
    queryState: Any, referenceState: Any,
    domainState: domainData, kernelProperties: kernelState,
    beginIndex: offset_t, numIndices: wp.int32,
    offsetArray: wp.array(dtype=index_t),
    restVolumes: wp.array(dtype=scalar_t),
    outputValue: Any,
):
    xi, hi, mi, rhoi, ki = getParticle(queryState, i)
    out = zero_like_warp(outputValue)
    for neighborIndex in range(numIndices):
        neighborOffset = beginIndex + offset_t(neighborIndex)
        j = wp.int32(offsetArray[neighborOffset])
        xj, hj, mj, rhoj, kj = getParticle(referenceState, j)
        if kj == 0:
            gradient = sphKernelGradient_ij(computeDistanceVec(xi, xj, domainState),
                                            hi, hj, kernelProperties, domainState)
            out += restVolumes[j] * wp.dot(gradient, gradient)
    return out


@wp.func
def computeIISPHPressureDiagonal_Func_Adjacency(
    i: wp.int32,
    queryState: Any, referenceState: Any,
    domainState: domainData, useAdjacency: wp.bool,
    adjacencyState: adjacencyData, gridState: gridData,
    kernelProperties: kernelState,
    restVolumes: wp.array(dtype=scalar_t),
    densityRatio: wp.array(dtype=scalar_t),
    gradientSum: wp.array(dtype=vector(length=Any, dtype=scalar_t)),
    outputValue: Any,
):
    if queryState.kinds[i] != 0:
        return zero_like_warp(outputValue)
    out = zero_like_warp(outputValue)
    numOffsets = 1 if useAdjacency else gridState.numOffsets
    for offset in range(numOffsets):
        beginIndex = offset_t(0)
        numIndices = wp.int32(0)
        if useAdjacency:
            beginIndex = adjacencyState.neighborOffsets[i]
            numIndices = adjacencyState.numNeighbors[i]
        else:
            gridBegin, numIndices = checkOffset(
                i, queryState.positions, gridState.numCells, gridState.D,
                offset, gridState.cellOffsets, gridState.hashTable, gridState.cellTable,
                domainState.periodicity, gridState.qMin, gridState.qMax, gridState.hCell,
            )
            beginIndex = offset_t(gridBegin)
            if beginIndex < 0:
                continue
        out += computeIISPHPressureDiagonal_Func_i(
            i, queryState, referenceState, domainState, kernelProperties,
            beginIndex, numIndices,
            adjacencyState.neighborList if useAdjacency else gridState.sortIndex,
            restVolumes, outputValue,
        )
    return out


@wp.kernel
def computeIISPHPressureDiagonal_Kernel(
    queryState: Any, referenceState: Any, domainState: domainData,
    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any, kernelProperties: kernelState,
    restVolumes: wp.array(dtype=scalar_t),
    densityRatio: wp.array(dtype=scalar_t),
    gradientSum: wp.array(dtype=vector(length=Any, dtype=scalar_t)),
    outputValues: wp.array(dtype=scalar_t),
):
    i = wp.tid()
    if i >= queryState.positions.shape[0]:
        return
    gradientSquares = computeIISPHPressureDiagonal_Func_Adjacency(
        i, queryState, referenceState, domainState,
        useAdjacency, adjacencyState, gridState, kernelProperties,
        restVolumes, densityRatio, gradientSum, zero_like_warp(outputValues),
    )
    diagonal = scalar_t(0.0)
    if queryState.kinds[i] == 0:
        # Apply the self-volume factor outside the reduction to preserve its adjoint.
        diagonal = -(wp.dot(gradientSum[i], gradientSum[i]) + restVolumes[i] * gradientSquares) / (
            densityRatio[i] * densityRatio[i])
    outputValues[i] = diagonal


_PRESSURE_DIAGONAL = OperatorSpec(
    kernel=computeIISPHPressureDiagonal_Kernel,
    outputs=(OutputSpec(dtype=scalar_t, shape=ShapeOf.QUERY),),
    extras=(
        ExtraSpec("restVolumes", ExtraKind.TENSOR),
        ExtraSpec("densityRatio", ExtraKind.TENSOR),
        ExtraSpec("gradientSum", ExtraKind.TENSOR),
    ),
)


def computeIISPHPressureDiagonalWarp(
    queryParticles: ParticleState,
    operationProperties: OperationProperties,
    domain: DomainDescription,
    *,
    restVolumes: torch.Tensor,
    densityRatio: torch.Tensor,
    gradientSum: torch.Tensor,
    adjacency=None,
    referenceParticles=None,
) -> torch.Tensor:
    """Base diagonal of continuity composed with pressure acceleration, without dt squared."""
    ctx = SPHContext(
        query=queryParticles, properties=operationProperties, domain=domain,
        adjacency=adjacency, reference=referenceParticles,
    )
    return launchOperator(
        _PRESSURE_DIAGONAL, ctx,
        restVolumes=restVolumes, densityRatio=densityRatio, gradientSum=gradientSum,
    )


def computeIISPHPressureDiagonal(state, config, adjacency, restVolumes, densityRatio, gradientSum):
    """Return the base pressure diagonal, without contact pressure or dt squared."""
    return computeIISPHPressureDiagonalWarp(state, OperationProperties(kernel=config.kernel),
        config.domain, adjacency=adjacency, restVolumes=restVolumes,
        densityRatio=densityRatio, gradientSum=gradientSum)

