"""Independent diagonal contribution diag(C F_contact), without dt squared."""

from typing import Any
import torch
import warp as wp
from warp.types import vector
from warpSPHCore import (
    ParticleState, OperationProperties, DomainDescription, SPHContext,
    OperatorSpec, OutputSpec, ExtraSpec, ExtraKind, ShapeOf, launchOperator,
    domainData, adjacencyData, gridData, kernelState, getParticle, checkOffset,
    zero_like_warp, computeDistanceVec,
    sphKernelGradient_ij, sphKernel_ij,
)
from warpSPHCore.type_config import scalar_t, index_t, offset_t


@wp.func
def computeIISPHContactPressureDiagonal_Func_i(
    i: wp.int32,
    queryState: Any, referenceState: Any,
    domainState: domainData, kernelProperties: kernelState,
    beginIndex: offset_t, numIndices: wp.int32,
    offsetArray: wp.array(dtype=index_t),
    restVolumes: wp.array(dtype=scalar_t),
    wallGradient: wp.array(dtype=vector(length=Any, dtype=scalar_t)),
    contactPoints: wp.array(dtype=vector(length=Any, dtype=scalar_t)),
    denominator: wp.array(dtype=scalar_t),
    outputValue: Any,
):
    xi, hi, mi, rhoi, ki = getParticle(queryState, i)
    out = zero_like_warp(outputValue)
    for neighborIndex in range(numIndices):
        neighborOffset = beginIndex + offset_t(neighborIndex)
        j = wp.int32(offsetArray[neighborOffset])
        xj, hj, mj, rhoj, kj = getParticle(referenceState, j)
        if kj == 0 and denominator[j] > scalar_t(0.0):
            gradient = sphKernelGradient_ij(computeDistanceVec(xi, xj, domainState),
                                            hi, hj, kernelProperties, domainState)
            contactWeight = sphKernel_ij(computeDistanceVec(xi, contactPoints[j], domainState),
                                        hj, hj, kernelProperties, domainState)
            interpolationWeight = restVolumes[i] * contactWeight / denominator[j]
            out += restVolumes[j] * interpolationWeight * wp.dot(gradient, wallGradient[j])
    return out


@wp.func
def computeIISPHContactPressureDiagonal_Func_Adjacency(
    i: wp.int32,
    queryState: Any, referenceState: Any,
    domainState: domainData, useAdjacency: wp.bool,
    adjacencyState: adjacencyData, gridState: gridData,
    kernelProperties: kernelState,
    restVolumes: wp.array(dtype=scalar_t),
    wallGradient: wp.array(dtype=vector(length=Any, dtype=scalar_t)),
    gradientSum: wp.array(dtype=vector(length=Any, dtype=scalar_t)),
    contactPoints: wp.array(dtype=vector(length=Any, dtype=scalar_t)),
    denominator: wp.array(dtype=scalar_t),
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
        out += computeIISPHContactPressureDiagonal_Func_i(
            i, queryState, referenceState, domainState, kernelProperties,
            beginIndex, numIndices,
            adjacencyState.neighborList if useAdjacency else gridState.sortIndex,
            restVolumes, wallGradient, contactPoints, denominator, outputValue,
        )
    return out


@wp.kernel
def computeIISPHContactPressureDiagonal_Kernel(
    queryState: Any, referenceState: Any, domainState: domainData,
    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any, kernelProperties: kernelState,
    restVolumes: wp.array(dtype=scalar_t),
    wallGradient: wp.array(dtype=vector(length=Any, dtype=scalar_t)),
    gradientSum: wp.array(dtype=vector(length=Any, dtype=scalar_t)),
    contactPoints: wp.array(dtype=vector(length=Any, dtype=scalar_t)),
    denominator: wp.array(dtype=scalar_t),
    outputValues: wp.array(dtype=scalar_t),
):
    i = wp.tid()
    if i >= queryState.positions.shape[0]:
        return
    diagonal = computeIISPHContactPressureDiagonal_Func_Adjacency(
        i, queryState, referenceState, domainState,
        useAdjacency, adjacencyState, gridState, kernelProperties,
        restVolumes, wallGradient, gradientSum, contactPoints, denominator, zero_like_warp(outputValues),
    )
    if queryState.kinds[i] == 0 and denominator[i] > scalar_t(0.0):
        xi, hi, mi, rhoi, ki = getParticle(queryState, i)
        contactWeight = sphKernel_ij(computeDistanceVec(xi, contactPoints[i], domainState),
                                    hi, hi, kernelProperties, domainState)
        interpolationWeight = restVolumes[i] * contactWeight / denominator[i]
        diagonal -= interpolationWeight * wp.dot(gradientSum[i], wallGradient[i])
    outputValues[i] = diagonal


_CONTACT_PRESSURE_DIAGONAL = OperatorSpec(
    kernel=computeIISPHContactPressureDiagonal_Kernel,
    outputs=(OutputSpec(dtype=scalar_t, shape=ShapeOf.QUERY),),
    extras=(
        ExtraSpec("restVolumes", ExtraKind.TENSOR),
        ExtraSpec("wallGradient", ExtraKind.TENSOR),
        ExtraSpec("gradientSum", ExtraKind.TENSOR),
        ExtraSpec("contactPoints", ExtraKind.TENSOR),
        ExtraSpec("denominator", ExtraKind.TENSOR),
    ),
)


def computeIISPHContactPressureDiagonalWarp(
    queryParticles: ParticleState,
    operationProperties: OperationProperties,
    domain: DomainDescription,
    *,
    restVolumes: torch.Tensor,
    wallGradient: torch.Tensor,
    gradientSum: torch.Tensor,
    contactPoints: torch.Tensor,
    denominator: torch.Tensor,
    adjacency=None,
    referenceParticles=None,
) -> torch.Tensor:
    """Independent diagonal contribution diag(C F_contact), without dt squared."""
    ctx = SPHContext(
        query=queryParticles, properties=operationProperties, domain=domain,
        adjacency=adjacency, reference=referenceParticles,
    )
    return launchOperator(
        _CONTACT_PRESSURE_DIAGONAL, ctx,
        restVolumes=restVolumes, wallGradient=wallGradient, gradientSum=gradientSum, contactPoints=contactPoints, denominator=denominator,
    )


def computeIISPHContactPressureDiagonal(state, config, adjacency, restVolumes,
                                      wallGradient, gradientSum, contactPoints, contactDenominator):
    """Return diag(C F_contact) independently of the base diagonal."""
    return computeIISPHContactPressureDiagonalWarp(state, OperationProperties(kernel=config.kernel),
        config.domain, adjacency=adjacency, restVolumes=restVolumes,
        wallGradient=wallGradient, gradientSum=gradientSum,
        contactPoints=contactPoints, denominator=contactDenominator)

