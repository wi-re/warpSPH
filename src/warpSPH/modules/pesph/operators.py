"""Two neighbour passes for both PESPH variants, using the core operator API.

The first pass gathers n, y, their support derivatives and mass density at
h_i. The second evaluates the conservative per-side force and conjugate
thermal work. Kernel evaluation, periodicity, adjacency/grid traversal and
autograd launching use warpSPHCore. No pair arrays survive either pass.
"""

from typing import Any

import warp as wp
from warp.types import vector
from warpSPHCore import (
    Corrections, ExtraKind, ExtraSpec, OperationProperties, OperatorSpec,
    OutputSpec, RowRole, ShapeOf, SPHContext, SupportScheme, adjacencyData,
    castTorchToWarpAsBuiltins, checkOffset, domainData, getParticle, gridData,
    kernelState, launchOperator, scalar_t, sphKernel, sphKernelDkDh,
    sphKernelGradient, zero_like_warp,
)
from warpSPHCore.type_config import offset_t
from warpSPHCore.profiling import record_function

from .thermodynamics import PressureSums, closePressure, pressureWeights
from .chunked import pesphInputs


@wp.func
def _neighborRange(i: wp.int32, state: Any, domain: domainData,
                   useAdjacency: wp.bool, adjacency: adjacencyData,
                   grid: gridData, offset: wp.int32):
    begin = offset_t(0)
    count = wp.int32(0)
    if useAdjacency:
        begin = adjacency.neighborOffsets[i]
        count = adjacency.numNeighbors[i]
    else:
        gridBegin, count = checkOffset(
            i, state.positions, grid.numCells, grid.D, offset,
            grid.cellOffsets, grid.hashTable, grid.cellTable,
            domain.periodicity, grid.qMin, grid.qMax, grid.hCell)
        begin = offset_t(gridBegin)
        if gridBegin < 0:
            count = wp.int32(0)
    return begin, count


@wp.kernel
def _pressureSums(
    queryState: Any, referenceState: Any, domainState: domainData,
    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any, kernelProperties: kernelState,
    weights: wp.array(dtype=scalar_t),
    numberDensity: wp.array(dtype=scalar_t), weightedSum: wp.array(dtype=scalar_t),
    numberDerivative: wp.array(dtype=scalar_t), weightedDerivative: wp.array(dtype=scalar_t),
    massDensity: wp.array(dtype=scalar_t),
):
    i = wp.tid()
    if i >= queryState.positions.shape[0]:
        return
    xi, hi, mi, rhoi, ki = getParticle(queryState, i)
    n, y, dn, dy, rho = scalar_t(0.0), scalar_t(0.0), scalar_t(0.0), scalar_t(0.0), scalar_t(0.0)
    numOffsets = wp.int32(1)
    if not useAdjacency:
        numOffsets = gridState.numOffsets
    for o in range(numOffsets):
        begin, count = _neighborRange(i, queryState, domainState, useAdjacency,
                                      adjacencyState, gridState, o)
        for k in range(count):
            j = wp.int32(0)
            if useAdjacency:
                j = wp.int32(adjacencyState.neighborList[begin + offset_t(k)])
            else:
                j = wp.int32(gridState.sortIndex[begin + offset_t(k)])
            xj, hj, mj, rhoj, kj = getParticle(referenceState, j)
            W = sphKernel(xi, xj, hi, hi, kernelProperties, domainState)
            dW = sphKernelDkDh(xi, xj, hi, hi, kernelProperties, domainState)
            n += W
            y += weights[j] * W
            dn += dW
            dy += weights[j] * dW
            rho += mj * W
    numberDensity[i] = n
    weightedSum[i] = y
    numberDerivative[i] = dn
    weightedDerivative[i] = dy
    massDensity[i] = rho


@wp.kernel
def _pressureForce(
    queryState: Any, referenceState: Any, domainState: domainData,
    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any, kernelProperties: kernelState,
    queryVelocities: wp.array(dtype=vector(length=Any, dtype=scalar_t)),
    referenceVelocities: wp.array(dtype=vector(length=Any, dtype=scalar_t)),
    queryWeights: wp.array(dtype=scalar_t), referenceWeights: wp.array(dtype=scalar_t),
    queryFactors: wp.array(dtype=scalar_t), referenceFactors: wp.array(dtype=scalar_t),
    queryGradH: wp.array(dtype=scalar_t), referenceGradH: wp.array(dtype=scalar_t),
    rowStart: wp.int32,
    acceleration: wp.array(dtype=vector(length=Any, dtype=scalar_t)),
    thermalWork: wp.array(dtype=scalar_t),
):
    i = wp.tid()
    if i >= queryState.positions.shape[0]:
        return
    xi, hi, mi, rhoi, ki = getParticle(queryState, i)
    acc = zero_like_warp(acceleration[i])
    work = scalar_t(0.0)
    numOffsets = wp.int32(1)
    if not useAdjacency:
        numOffsets = gridState.numOffsets
    for o in range(numOffsets):
        begin, count = _neighborRange(i, queryState, domainState, useAdjacency,
                                      adjacencyState, gridState, o)
        for k in range(count):
            j = wp.int32(0)
            if useAdjacency:
                j = wp.int32(adjacencyState.neighborList[begin + offset_t(k)])
            else:
                j = wp.int32(gridState.sortIndex[begin + offset_t(k)])
            if i + rowStart == j:
                continue
            xj, hj, mj, rhoj, kj = getParticle(referenceState, j)
            gradWi = sphKernelGradient(xi, xj, hi, hi, kernelProperties, domainState)
            gradWj = sphKernelGradient(xi, xj, hj, hj, kernelProperties, domainState)
            # w_i*w_j*fH_ij = w_i*(w_j-b_i): avoid a division by w_j.
            left = queryWeights[i] * (referenceWeights[j] - queryGradH[i]) * queryFactors[i] / mi
            right = referenceWeights[j] * (queryWeights[i] - referenceGradH[j]) * referenceFactors[j] / mi
            acc -= left * gradWi + right * gradWj
            work += left * wp.dot(queryVelocities[i] - referenceVelocities[j], gradWi)
    acceleration[i] = acc
    thermalWork[i] = work


def _scalarDtype(ctx, extras):
    return castTorchToWarpAsBuiltins(ctx.query.densities).dtype


def _vectorDtype(ctx, extras):
    return castTorchToWarpAsBuiltins(ctx.query.positions).dtype


_SUMS = OperatorSpec(
    rowChunked=True,
    kernel=_pressureSums,
    outputs=tuple(OutputSpec(dtype=_scalarDtype, shape=ShapeOf.QUERY) for _ in range(5)),
    extras=(ExtraSpec('weights', ExtraKind.TENSOR, RowRole.REFERENCE),),
)
_FORCE = OperatorSpec(
    rowChunked=True,
    kernel=_pressureForce,
    outputs=(OutputSpec(dtype=_vectorDtype, shape=ShapeOf.QUERY),
             OutputSpec(dtype=_scalarDtype, shape=ShapeOf.QUERY)),
    extras=tuple(ExtraSpec(prefix + name, ExtraKind.TENSOR, role)
                 for name in ('Velocities', 'Weights', 'Factors', 'GradH')
                 for prefix, role in (('query', RowRole.QUERY), ('reference', RowRole.REFERENCE)))
           + (ExtraSpec('rowStart', ExtraKind.SCALAR, RowRole.ROW_START),),
)


def _context(state, config, adjacency, supportMode=SupportScheme.Gather):
    return SPHContext(
        query=state, reference=state, domain=config.domain, adjacency=adjacency,
        properties=OperationProperties(kernel=config.kernel, supportMode=supportMode,
                                       n_h=getattr(config, 'n_h', None),
                                       calibrateNormalization=getattr(config, 'calibrateNormalization', False)),
        corrections=Corrections(volumes=(None, None), crk=None, gradH=None, renorm=None),
    )


def computePressureSums(state, config, weights, adjacency=None):
    """Gather at each h_i, including self; list must cover max(h_i,h_j)."""
    with record_function('warpSPH[PESPH pressure sums]'):
        return PressureSums(*launchOperator(_SUMS, _context(state, config, adjacency), weights=weights))


def computePressureState(state, config, schemeConfig, adjacency=None):
    """One shared sum pass and variant-specific thermodynamic closure."""
    weights = pesphInputs(adjacency, lambda: pressureWeights(state, schemeConfig.variant, schemeConfig.gamma))
    sums = computePressureSums(state, config, weights, adjacency)
    return pesphInputs(adjacency, lambda: closePressure(
        state, sums, weights, schemeConfig.variant, schemeConfig.gamma,
        adaptive=schemeConfig.adaptiveSupportCorrections))


def computePressureForce(state, pressure, config, adjacency=None):
    """Return conservative acceleration and specific pdV work for either variant."""
    with record_function('warpSPH[PESPH pressure force]'):
        return launchOperator(_FORCE, _context(state, config, adjacency, SupportScheme.SuperSymmetric),
                              queryVelocities=state.velocities, referenceVelocities=state.velocities,
                              queryWeights=pressure.weights, referenceWeights=pressure.weights,
                              queryFactors=pressure.pressureFactors, referenceFactors=pressure.pressureFactors,
                              queryGradH=pressure.gradH, referenceGradH=pressure.gradH, rowStart=0)
