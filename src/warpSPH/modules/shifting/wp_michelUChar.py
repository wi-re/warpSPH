"""Warp kernel for Michel et al. 2022 (`literature/michel2022`) Eq. (20)'s
characteristic velocity `U_char_i = U_lim_i = max_j |(u_j - u_i) . x_hat_ij|`,
the relative (hence Galilean- and local-rotation-invariant, PST_ALE_PLAN.md
Part 2.1) velocity scale that replaces the Mach-number-based `U_char` of
`sample.wp_deltaShift`'s delta-SPH law in `modules.shifting.michel`'s Eq. (22).

Split into a forward-only argmax pass (`computeUChar_Func_i_argmax` /
`computeUChar_Func_Adjacency_argmax`) that finds the winning neighbor index
via a loop-carried `wp.max`-style reassignment, followed by a single
re-evaluation of the actual value for just that index
(`computeUChar_valueAt`), done outside any loop -- the same split
`modules.shockCapturing.wp_vsig` uses and for the same reason: warp-lang
1.15.0's reverse-mode AD silently zeroes the adjoint of a value produced by
loop-carried max reassignment, but is safe once only the winning *index* (an
int, never differentiated by Warp) crosses out of the loop.
`computeUCharWarp` is the torch-facing entry point. A particle with no
in-support neighbor returns `0.0`.
"""

import warp as wp
from warp.types import vector, matrix
from typing import Any
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from typing import Optional, Union, Tuple
from warpSPHCore import *

__all__ = ['computeUCharWarp']


@wp.func
def computeUChar_valueAt(
    dim: wp.int32,

    xi: vector(dtype = scalar_t, length=Any), hi: scalar_t, # type: ignore
    vel_i: vector(length=Any, dtype=scalar_t), # type: ignore

    referenceState: Any, # particleDataSoA with the exact type based on the dimensionality
    referenceVelocities: wp.array(dtype = vector(length=Any, dtype=scalar_t)), # type: ignore
    j: wp.int32,

    domainState: domainData,
) -> scalar_t:
    # The per-neighbor U_char formula, evaluated once for a single,
    # already-known neighbor index -- deliberately outside the loop, see this
    # file's module docstring.
    xj, hj, mj, rhoj, kj = getParticle(referenceState, j)
    vel_j = referenceVelocities[j]

    x_ij = computeDistanceVec(xi, xj, domainState)
    r_ij = safe_sqrt(wp.dot(x_ij, x_ij))

    v_ij = vel_j - vel_i
    return wp.abs(wp.dot(v_ij, x_ij) / (r_ij + scalar_t(1.0e-14) * hi))


@wp.func
def computeUChar_Func_i_argmax(
    i : wp.int32,  dim: wp.int32,

    xi: vector(dtype = scalar_t, length=Any), hi: scalar_t, mi: scalar_t, rhoi: scalar_t, # type: ignore

    referenceState: Any, # particleDataSoA with the exact type based on the dimensionality

    domainState: domainData,
    kernelProperties: kernelState,

    beginIndex: wp.int32, # type: ignore
    numIndices: wp.int32, # type: ignore
    offsetArray: wp.array(dtype = wp.int64), # type: ignore

    ki : wp.int32, referenceKinds : wp.array(dtype = wp.int32), # type: ignore

    vel_i: vector(length=Any, dtype=scalar_t), referenceVelocities: wp.array(dtype = vector(length=Any, dtype=scalar_t)), # type: ignore
):
    # Forward-only: find which neighbor achieves the max |(u_j-u_i).x_hat_ij|,
    # and its value, via the same loop-carried wp.max reassignment as
    # wp_vsig.py -- not used differentiably, only the winning index crosses
    # out (see computeUChar_valueAt).
    found = wp.bool(False)
    bestVal = scalar_t(0.0)
    bestJ = wp.int32(0)

    for neighborIndex in range(numIndices):
        jj = beginIndex + neighborIndex
        j  = wp.int32(offsetArray[jj])
        if kernelProperties.operationMode != wp.static(OperationDirection.TrueAllToToAll.value):
            if not checkDirectionality_j(referenceKinds[j], kernelProperties.operationMode):
                continue
        ##########################################################
        #   The core particle-particle interaction starts here   #
        ##########################################################

        xj, hj, mj, rhoj, kj = getParticle(referenceState, j)

        x_ij = computeDistanceVec(xi, xj, domainState)
        r_ij = safe_sqrt(wp.dot(x_ij, x_ij))

        # Compact-support filter (see wp_vsig.py's own note: required for
        # correctness under grid traversal, which returns every particle in a
        # nearby cell rather than an exact, pre-filtered neighbor list).
        hij = computePairwiseSupport(hi, hj, kernelProperties.supportMode)
        if r_ij >= hij and i != j:
            continue
        if i == j:
            continue

        vel_j = referenceVelocities[j]
        v_ij = vel_j - vel_i
        val = wp.abs(wp.dot(v_ij, x_ij) / (r_ij + scalar_t(1.0e-14) * hi))

        if val > bestVal:
            bestVal = val
            bestJ = j
            found = wp.bool(True)

    return found, bestVal, bestJ


@wp.func
def computeUChar_Func_Adjacency_argmax(
    i : wp.int32, dim: wp.int32,

    queryState: Any, # particleDataSoA with the exact type based on the dimensionality
    referenceState: Any, # particleDataSoA with the exact type based on the dimensionality
    correctionData: Any, # correctionData_1 or correctionData_2 or correctionData_3

    domainState: domainData,
    useAdjacency: wp.bool,
    adjacencyState: adjacencyData,
    gridState: gridData,
    numOffsets: wp.int32,

    kernelProperties: kernelState,

    queryVelocities: wp.array(dtype = vector(length=Any, dtype=scalar_t)), referenceVelocities: wp.array(dtype = vector(length=Any, dtype=scalar_t)), # type: ignore
):
    # Mirrors computeVsig_Func_Adjacency_argmax: finds the single neighbor
    # with the globally largest value across every offset (a max over the
    # union, not a sum of per-offset maxima -- see wp_vsig.py's note on why
    # summing per-offset winners would be wrong for grid traversal).
    xi, hi, mi, rhoi, ki = getParticle(queryState, i)
    if kernelProperties.operationMode != wp.static(OperationDirection.TrueAllToToAll.value):
        if not checkDirectionality_i(ki, kernelProperties.operationMode):
            return wp.bool(False), wp.int32(0)

    vel_i = queryVelocities[i]

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

        found, bestVal, bestJ = computeUChar_Func_i_argmax(
            i, dim,
            xi, hi, mi, rhoi,
            referenceState, domainState,
            kernelProperties,

            beginIndex, numIndices, adjacencyState.neighborList if useAdjacency else gridState.sortIndex,
            ki, referenceState.kinds,

            vel_i, referenceVelocities,
        )
        if found and bestVal > globalBestVal:
            globalBestVal = bestVal
            globalBestJ = bestJ
            globalFound = wp.bool(True)

    return globalFound, globalBestJ


@wp.kernel
def computeUChar_Kernel(
    queryState: Any,
    referenceState: Any,
    domainState: domainData,

    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any,

    kernelProperties: kernelState,
    # Do not change the parameters above
    queryVelocities: wp.array(dtype = vector(length=Any, dtype=scalar_t)), referenceVelocities: wp.array(dtype = vector(length=Any, dtype=scalar_t)), # type: ignore
    # The last parameter is always the output array and should not be changed
    outputValues : wp.array(dtype = scalar_t) # type: ignore
):
    i = wp.tid()
    numParticles = queryState.positions.shape[0]
    if i >= numParticles:
        return

    found, bestJ = computeUChar_Func_Adjacency_argmax(
        i, domainState.dim,
        queryState, referenceState, correctionData, domainState,
        useAdjacency, adjacencyState, gridState, gridState.numOffsets if not useAdjacency else 1,
        kernelProperties,
        queryVelocities, referenceVelocities,
    )

    if found:
        # The one and only differentiable evaluation, done here rather than
        # per-offset in the traversal function -- see this file's module
        # docstring and wp_vsig.py's analogous note.
        xi, hi, mi, rhoi, ki = getParticle(queryState, i)
        vel_i = queryVelocities[i]
        outputValues[i] = computeUChar_valueAt(
            domainState.dim, xi, hi, vel_i,
            referenceState, referenceVelocities, bestJ,
            domainState,
        )
    else:
        outputValues[i] = zero_like_warp(outputValues)


def _uCharDtype(ctx, extras):
    return castTorchToWarpAsBuiltins(ctx.query.densities).dtype


_UCHAR = OperatorSpec(
    kernel=computeUChar_Kernel,
    outputs=(OutputSpec(dtype=_uCharDtype, shape=ShapeOf.QUERY),),
    extras=(
        ExtraSpec("queryVelocities", ExtraKind.TENSOR),
        ExtraSpec("referenceVelocities", ExtraKind.TENSOR),
    ),
)


def computeUCharWarp(
    queryParticles: ParticleState,
    operationProperties: OperationProperties,
    domain: DomainDescription,

    queryVelocities: Optional[torch.Tensor] = None, referenceVelocities: Optional[torch.Tensor] = None,

    queryVolumes: Optional[torch.Tensor] = None, referenceVolumes: Optional[torch.Tensor] = None,
    adjacency: Optional[Union[AdjacencyList, CompactHashMap]] = None,
    referenceParticles: Optional[ParticleState] = None,
    crkState: Optional[CRKState] = None,
    gradHState: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], GradHState]] = None,
    renormalizationState: Optional[Union[torch.Tensor,RenormalizationState]] = None,
):
    # Mirrors wp_vsig.py's computeVsigWarp fallback chain exactly: fall back
    # to queryVelocities first (so a caller that only supplies
    # queryVelocities -- the common case, a bare reference ParticleState with
    # no .velocities attribute -- doesn't hit an AttributeError), then to
    # each particle set's own .velocities attribute if it has one.
    if referenceVelocities is None:
        referenceVelocities = queryVelocities

    referenceParticles = referenceParticles if referenceParticles is not None else queryParticles
    queryVelocities_ = queryVelocities if queryVelocities is not None else (queryParticles.velocities if hasattr(queryParticles, 'velocities') else None)
    referenceVelocities_ = referenceVelocities if referenceVelocities is not None else (referenceParticles.velocities if hasattr(referenceParticles, 'velocities') else None)

    if queryVelocities_ is None:
        raise ValueError("Velocities must be provided either through queryVelocities or as a property of queryParticles.")

    with record_function("warpSPH[computeUChar]"):
        ctx = SPHContext(
            query=queryParticles, properties=operationProperties, domain=domain,
            adjacency=adjacency, reference=referenceParticles,
            corrections=Corrections(
                volumes=(queryVolumes, referenceVolumes),
                crk=crkState, gradH=gradHState, renorm=renormalizationState,
            ),
        )
        return launchOperator(
            _UCHAR, ctx,
            queryVelocities=queryVelocities_, referenceVelocities=referenceVelocities_,
        )
