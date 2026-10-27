"""Total self-pressure gradient c_i = sum_fluid V_j grad W_ij + G_i."""

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
def computeIISPHGradientSum_Func_i(
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
            out += restVolumes[j] * gradient
    return out


@wp.func
def computeIISPHGradientSum_Func_Adjacency(
    i: wp.int32,
    queryState: Any, referenceState: Any,
    domainState: domainData, useAdjacency: wp.bool,
    adjacencyState: adjacencyData, gridState: gridData,
    kernelProperties: kernelState,
    restVolumes: wp.array(dtype=scalar_t),
    wallGradient: wp.array(dtype=vector(length=Any, dtype=scalar_t)),
    outputValue: Any,
):
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
        out += computeIISPHGradientSum_Func_i(
            i, queryState, referenceState, domainState, kernelProperties,
            beginIndex, numIndices,
            adjacencyState.neighborList if useAdjacency else gridState.sortIndex,
            restVolumes, outputValue,
        )
    return out + wallGradient[i]


@wp.kernel
def computeIISPHGradientSum_Kernel(
    queryState: Any, referenceState: Any, domainState: domainData,
    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any, kernelProperties: kernelState,
    restVolumes: wp.array(dtype=scalar_t),
    wallGradient: wp.array(dtype=vector(length=Any, dtype=scalar_t)),
    outputValues: wp.array(dtype=vector(length=Any, dtype=scalar_t)),
):
    i = wp.tid()
    if i >= queryState.positions.shape[0]:
        return
    outputValues[i] = computeIISPHGradientSum_Func_Adjacency(
        i, queryState, referenceState, domainState,
        useAdjacency, adjacencyState, gridState, kernelProperties,
        restVolumes, wallGradient, zero_like_warp(outputValues),
    )


def _outputDtype(ctx, extras):
    return castTorchToWarpAsBuiltins(ctx.query.positions).dtype


_GRADIENT_SUM = OperatorSpec(
    kernel=computeIISPHGradientSum_Kernel,
    outputs=(OutputSpec(dtype=_outputDtype, shape=ShapeOf.QUERY),),
    extras=(
        ExtraSpec("restVolumes", ExtraKind.TENSOR),
        ExtraSpec("wallGradient", ExtraKind.TENSOR),
    ),
)


def computeIISPHGradientSumWarp(
    queryParticles: ParticleState,
    operationProperties: OperationProperties,
    domain: DomainDescription,
    *,
    restVolumes: torch.Tensor,
    wallGradient: torch.Tensor,
    adjacency=None,
    referenceParticles=None,
) -> torch.Tensor:
    """Total self-pressure gradient c_i = sum_fluid V_j grad W_ij + G_i."""
    ctx = SPHContext(
        query=queryParticles, properties=operationProperties, domain=domain,
        adjacency=adjacency, reference=referenceParticles,
    )
    return launchOperator(
        _GRADIENT_SUM, ctx,
        restVolumes=restVolumes, wallGradient=wallGradient,
    )


def computeIISPHGradientSum(state, config, adjacency, restVolumes, wallGradient):
    """Return c_i = sum_fluid V_j grad W_ij + G_i without modifying G."""
    return computeIISPHGradientSumWarp(state, OperationProperties(kernel=config.kernel),
        config.domain, adjacency=adjacency, restVolumes=restVolumes, wallGradient=wallGradient)

