"""Rest-volume continuity C(v); positive values indicate compression."""

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
def computeIISPHDensityRate_Func_i(
    i: wp.int32,
    queryState: Any, referenceState: Any,
    domainState: domainData, kernelProperties: kernelState,
    beginIndex: offset_t, numIndices: wp.int32,
    offsetArray: wp.array(dtype=index_t),
    restVolumes: wp.array(dtype=scalar_t),
    field: wp.array(dtype=vector(length=Any, dtype=scalar_t)),
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
            out += restVolumes[j] * wp.dot(field[i] - field[j], gradient)
    return out


@wp.func
def computeIISPHDensityRate_Func_Adjacency(
    i: wp.int32,
    queryState: Any, referenceState: Any,
    domainState: domainData, useAdjacency: wp.bool,
    adjacencyState: adjacencyData, gridState: gridData,
    kernelProperties: kernelState,
    restVolumes: wp.array(dtype=scalar_t),
    wallGradient: wp.array(dtype=vector(length=Any, dtype=scalar_t)),
    field: wp.array(dtype=vector(length=Any, dtype=scalar_t)),
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
        out += computeIISPHDensityRate_Func_i(
            i, queryState, referenceState, domainState, kernelProperties,
            beginIndex, numIndices,
            adjacencyState.neighborList if useAdjacency else gridState.sortIndex,
            restVolumes, field, outputValue,
        )
    return out + wp.dot(field[i], wallGradient[i])


@wp.kernel
def computeIISPHDensityRate_Kernel(
    queryState: Any, referenceState: Any, domainState: domainData,
    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any, kernelProperties: kernelState,
    restVolumes: wp.array(dtype=scalar_t),
    wallGradient: wp.array(dtype=vector(length=Any, dtype=scalar_t)),
    field: wp.array(dtype=vector(length=Any, dtype=scalar_t)),
    outputValues: wp.array(dtype=scalar_t),
):
    i = wp.tid()
    if i >= queryState.positions.shape[0]:
        return
    outputValues[i] = computeIISPHDensityRate_Func_Adjacency(
        i, queryState, referenceState, domainState,
        useAdjacency, adjacencyState, gridState, kernelProperties,
        restVolumes, wallGradient, field, zero_like_warp(outputValues),
    )


_DENSITY_RATE = OperatorSpec(
    kernel=computeIISPHDensityRate_Kernel,
    outputs=(OutputSpec(dtype=scalar_t, shape=ShapeOf.QUERY),),
    extras=(
        ExtraSpec("restVolumes", ExtraKind.TENSOR),
        ExtraSpec("wallGradient", ExtraKind.TENSOR),
        ExtraSpec("field", ExtraKind.TENSOR),
    ),
)


def computeIISPHDensityRateWarp(
    queryParticles: ParticleState,
    operationProperties: OperationProperties,
    domain: DomainDescription,
    *,
    restVolumes: torch.Tensor,
    wallGradient: torch.Tensor,
    field: torch.Tensor,
    adjacency=None,
    referenceParticles=None,
) -> torch.Tensor:
    """Rest-volume continuity C(v); positive values indicate compression."""
    ctx = SPHContext(
        query=queryParticles, properties=operationProperties, domain=domain,
        adjacency=adjacency, reference=referenceParticles,
    )
    return launchOperator(
        _DENSITY_RATE, ctx,
        restVolumes=restVolumes, wallGradient=wallGradient, field=field,
    )


def computeIISPHDensityRate(state, config, adjacency, field, restVolumes, wallGradient):
    """Return C(field), including the static wall gradient; positive is compression."""
    return computeIISPHDensityRateWarp(state, OperationProperties(kernel=config.kernel),
        config.domain, adjacency=adjacency,
        restVolumes=restVolumes, wallGradient=wallGradient, field=field)

