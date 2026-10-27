"""Known pressure acceleration from gravity extended into sampled solids."""

from typing import Any
import torch
import warp as wp
from warp.types import vector
from warpSPHCore import (
    ParticleState, OperationProperties, DomainDescription, SPHContext,
    OperatorSpec, OutputSpec, ExtraSpec, ExtraKind, ShapeOf, launchOperator,
    domainData, adjacencyData, gridData, kernelState, getParticle, checkOffset,
    zero_like_warp, castTorchToWarpAsBuiltins, computeDistanceVec,
    sphKernelGradient_ij,
)
from warpSPHCore.type_config import scalar_t, index_t, offset_t


@wp.func
def computeIISPHSampledGravity_Func_i(
    i: wp.int32,
    queryState: Any, referenceState: Any,
    domainState: domainData, kernelProperties: kernelState,
    beginIndex: offset_t, numIndices: wp.int32,
    offsetArray: wp.array(dtype=index_t),
    restVolumes: wp.array(dtype=scalar_t),
    contactPoints: wp.array(dtype=vector(length=Any, dtype=scalar_t)),
    gravity: wp.array(dtype=vector(length=Any, dtype=scalar_t)),
    outputValue: Any,
):
    xi, hi, mi, rhoi, ki = getParticle(queryState, i)
    out = zero_like_warp(outputValue)
    for neighborIndex in range(numIndices):
        neighborOffset = beginIndex + offset_t(neighborIndex)
        j = wp.int32(offsetArray[neighborOffset])
        xj, hj, mj, rhoj, kj = getParticle(referenceState, j)
        if kj == 1:
            gradient = sphKernelGradient_ij(computeDistanceVec(xi, xj, domainState),
                                            hi, hj, kernelProperties, domainState)
            offset = computeDistanceVec(xj, contactPoints[i], domainState)
            out -= restVolumes[j] * wp.dot(gravity[i], offset) * gradient
    return out


@wp.func
def computeIISPHSampledGravity_Func_Adjacency(
    i: wp.int32,
    queryState: Any, referenceState: Any,
    domainState: domainData, useAdjacency: wp.bool,
    adjacencyState: adjacencyData, gridState: gridData,
    kernelProperties: kernelState,
    restVolumes: wp.array(dtype=scalar_t),
    contactPoints: wp.array(dtype=vector(length=Any, dtype=scalar_t)),
    gravity: wp.array(dtype=vector(length=Any, dtype=scalar_t)),
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
        out += computeIISPHSampledGravity_Func_i(
            i, queryState, referenceState, domainState, kernelProperties,
            beginIndex, numIndices,
            adjacencyState.neighborList if useAdjacency else gridState.sortIndex,
            restVolumes, contactPoints, gravity, outputValue,
        )
    return out


@wp.kernel
def computeIISPHSampledGravity_Kernel(
    queryState: Any, referenceState: Any, domainState: domainData,
    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any, kernelProperties: kernelState,
    restVolumes: wp.array(dtype=scalar_t),
    contactPoints: wp.array(dtype=vector(length=Any, dtype=scalar_t)),
    gravity: wp.array(dtype=vector(length=Any, dtype=scalar_t)),
    outputValues: wp.array(dtype=vector(length=Any, dtype=scalar_t)),
):
    i = wp.tid()
    if i >= queryState.positions.shape[0]:
        return
    outputValues[i] = computeIISPHSampledGravity_Func_Adjacency(
        i, queryState, referenceState, domainState,
        useAdjacency, adjacencyState, gridState, kernelProperties,
        restVolumes, contactPoints, gravity, zero_like_warp(outputValues),
    )


def _outputDtype(ctx, extras):
    return castTorchToWarpAsBuiltins(ctx.query.positions).dtype


_SAMPLED_GRAVITY = OperatorSpec(
    kernel=computeIISPHSampledGravity_Kernel,
    outputs=(OutputSpec(dtype=_outputDtype, shape=ShapeOf.QUERY),),
    extras=(
        ExtraSpec("restVolumes", ExtraKind.TENSOR),
        ExtraSpec("contactPoints", ExtraKind.TENSOR),
        ExtraSpec("gravity", ExtraKind.TENSOR),
    ),
)


def computeIISPHSampledGravityWarp(
    queryParticles: ParticleState,
    operationProperties: OperationProperties,
    domain: DomainDescription,
    *,
    restVolumes: torch.Tensor,
    contactPoints: torch.Tensor,
    gravity: torch.Tensor,
    adjacency=None,
    referenceParticles=None,
) -> torch.Tensor:
    """Known pressure acceleration from gravity extended into sampled solids."""
    ctx = SPHContext(
        query=queryParticles, properties=operationProperties, domain=domain,
        adjacency=adjacency, reference=referenceParticles,
    )
    return launchOperator(
        _SAMPLED_GRAVITY, ctx,
        restVolumes=restVolumes, contactPoints=contactPoints, gravity=gravity,
    )

