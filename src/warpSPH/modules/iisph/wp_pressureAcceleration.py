"""Linear IISPH pressure acceleration F(q), with q = p / rho0."""

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
from .contactPressure import computeIISPHContactPressure
from .wp_densityRate import computeIISPHDensityRate


@wp.func
def computeIISPHPressureAcceleration_Func_i(
    i: wp.int32,
    queryState: Any, referenceState: Any,
    domainState: domainData, kernelProperties: kernelState,
    beginIndex: offset_t, numIndices: wp.int32,
    offsetArray: wp.array(dtype=index_t),
    restVolumes: wp.array(dtype=scalar_t),
    densityRatio: wp.array(dtype=scalar_t),
    normalizedPressure: wp.array(dtype=scalar_t),
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
            out -= restVolumes[j] * normalizedPressure[j] / (
                densityRatio[j] * densityRatio[j]) * gradient
    return out


@wp.func
def computeIISPHPressureAcceleration_Func_Adjacency(
    i: wp.int32,
    queryState: Any, referenceState: Any,
    domainState: domainData, useAdjacency: wp.bool,
    adjacencyState: adjacencyData, gridState: gridData,
    kernelProperties: kernelState,
    restVolumes: wp.array(dtype=scalar_t),
    densityRatio: wp.array(dtype=scalar_t),
    gradientSum: wp.array(dtype=vector(length=Any, dtype=scalar_t)),
    wallGradient: wp.array(dtype=vector(length=Any, dtype=scalar_t)),
    contactPressure: wp.array(dtype=scalar_t),
    normalizedPressure: wp.array(dtype=scalar_t),
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
        out += computeIISPHPressureAcceleration_Func_i(
            i, queryState, referenceState, domainState, kernelProperties,
            beginIndex, numIndices,
            adjacencyState.neighborList if useAdjacency else gridState.sortIndex,
            restVolumes, densityRatio, normalizedPressure, outputValue,
        )
    return (-gradientSum[i] * normalizedPressure[i] / (densityRatio[i] * densityRatio[i])
            - wallGradient[i] * contactPressure[i]) + out


@wp.kernel
def computeIISPHPressureAcceleration_Kernel(
    queryState: Any, referenceState: Any, domainState: domainData,
    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any, kernelProperties: kernelState,
    restVolumes: wp.array(dtype=scalar_t),
    densityRatio: wp.array(dtype=scalar_t),
    gradientSum: wp.array(dtype=vector(length=Any, dtype=scalar_t)),
    wallGradient: wp.array(dtype=vector(length=Any, dtype=scalar_t)),
    contactPressure: wp.array(dtype=scalar_t),
    normalizedPressure: wp.array(dtype=scalar_t),
    outputValues: wp.array(dtype=vector(length=Any, dtype=scalar_t)),
):
    i = wp.tid()
    if i >= queryState.positions.shape[0]:
        return
    outputValues[i] = computeIISPHPressureAcceleration_Func_Adjacency(
        i, queryState, referenceState, domainState,
        useAdjacency, adjacencyState, gridState, kernelProperties,
        restVolumes, densityRatio, gradientSum, wallGradient, contactPressure, normalizedPressure, zero_like_warp(outputValues),
    )


def _outputDtype(ctx, extras):
    return castTorchToWarpAsBuiltins(ctx.query.positions).dtype


_PRESSURE_ACCELERATION = OperatorSpec(
    kernel=computeIISPHPressureAcceleration_Kernel,
    outputs=(OutputSpec(dtype=_outputDtype, shape=ShapeOf.QUERY),),
    extras=(
        ExtraSpec("restVolumes", ExtraKind.TENSOR),
        ExtraSpec("densityRatio", ExtraKind.TENSOR),
        ExtraSpec("gradientSum", ExtraKind.TENSOR),
        ExtraSpec("wallGradient", ExtraKind.TENSOR),
        ExtraSpec("contactPressure", ExtraKind.TENSOR),
        ExtraSpec("normalizedPressure", ExtraKind.TENSOR),
    ),
)


def computeIISPHPressureAccelerationWarp(
    queryParticles: ParticleState,
    operationProperties: OperationProperties,
    domain: DomainDescription,
    *,
    restVolumes: torch.Tensor,
    densityRatio: torch.Tensor,
    gradientSum: torch.Tensor,
    wallGradient: torch.Tensor,
    contactPressure: torch.Tensor,
    normalizedPressure: torch.Tensor,
    adjacency=None,
    referenceParticles=None,
) -> torch.Tensor:
    """Linear IISPH pressure acceleration F(q), with q = p / rho0."""
    ctx = SPHContext(
        query=queryParticles, properties=operationProperties, domain=domain,
        adjacency=adjacency, reference=referenceParticles,
    )
    return launchOperator(
        _PRESSURE_ACCELERATION, ctx,
        restVolumes=restVolumes, densityRatio=densityRatio, gradientSum=gradientSum, wallGradient=wallGradient, contactPressure=contactPressure, normalizedPressure=normalizedPressure,
    )


def computeIISPHPressureAcceleration(state, config, adjacency, normalizedPressure,
                                     restVolumes, densityRatio, gradientSum, wallGradient, *,
                                     contactPoints=None, contactAdjacency=None, contactDenominator=None):
    """Return F(q), excluding the prescribed gravity extension."""
    contactPressure = computeIISPHContactPressure(state, config, normalizedPressure, restVolumes,
        contactPoints, contactAdjacency, contactDenominator)
    return computeIISPHPressureAccelerationWarp(state, OperationProperties(kernel=config.kernel),
        config.domain, adjacency=adjacency, restVolumes=restVolumes, densityRatio=densityRatio,
        gradientSum=gradientSum, wallGradient=wallGradient,
        contactPressure=contactPressure, normalizedPressure=normalizedPressure)


def computeIISPHPressureOperator(state, config, adjacency, normalizedPressure,
                               restVolumes, densityRatio, gradientSum, wallGradient, *,
                               contactPoints=None, contactAdjacency=None, contactDenominator=None):
    """Return C F(q) without assembling a pressure matrix."""
    acceleration = computeIISPHPressureAcceleration(state, config, adjacency, normalizedPressure,
        restVolumes, densityRatio, gradientSum, wallGradient,
        contactPoints=contactPoints, contactAdjacency=contactAdjacency, contactDenominator=contactDenominator)
    return computeIISPHDensityRate(state, config, adjacency, acceleration, restVolumes, wallGradient)

