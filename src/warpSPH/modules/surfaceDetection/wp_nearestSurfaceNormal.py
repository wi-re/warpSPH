"""Warp kernel for Michel et al. 2022 (`literature/michel2022`) Eq. (47): the
"inherited normal" `n_tilde_i` used by the free-surface PST projection
(`modules.shifting.wrapper`'s `ShiftingProjectionScheme.michel2022` branch,
Eq. 48) and by the linear beta decay (`PST_ALE_PLAN.md` Part 2.3/5.1). A
free-surface particle (`freeSurfaceMask[i] > 0.5`, the *raw*, undilated mask --
not the dilated vicinity set) uses its own normal and `d_i^FS = 0`; any other
particle inherits the normal of, and its distance to, its *nearest*
free-surface-particle neighbor (searched within kernel support only -- a
particle with no free-surface neighbor in support returns the `NOT_FOUND`
sentinel distance, `1e30`, which every caller treats as "interior", never
`inf`, to keep every downstream arithmetic op finite).

Same two-pass split as `modules.shockCapturing.wp_vsig` /
`modules.shifting.wp_michelUChar`, with the argmin variant: a forward-only
pass (`computeNearestSurfaceNormal_Func_i_argmin`) finds the nearest
qualifying neighbor's *index* via a loop-carried `wp.min`-style reassignment
(never differentiated -- only the winning int index crosses the loop
boundary), then `computeNearestSurfaceNormal_valueAt` recomputes the actual
distance and gathers the normal for that one index, outside any loop.
`computeNearestSurfaceNormalWarp` is the torch-facing entry point, returning
`(d_FS, n_tilde)`.
"""

import warp as wp
from warp.types import vector, matrix
from typing import Any
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from typing import Optional, Union, Tuple
from warpSPHCore import *

__all__ = ['computeNearestSurfaceNormalWarp']

NOT_FOUND_DISTANCE = 1.0e30


@wp.func
def computeNearestSurfaceNormal_valueAt(
    xi: vector(dtype = scalar_t, length=Any), # type: ignore
    referenceState: Any, # particleDataSoA with the exact type based on the dimensionality
    referenceNormals: wp.array(dtype = vector(length=Any, dtype=scalar_t)), # type: ignore
    j: wp.int32,
    domainState: domainData,
):
    # The actual (differentiable) distance + normal gather for a single,
    # already-known neighbor index -- deliberately outside the loop, see this
    # file's module docstring.
    xj, hj, mj, rhoj, kj = getParticle(referenceState, j)
    x_ij = computeDistanceVec(xi, xj, domainState)
    r_ij = safe_sqrt(wp.dot(x_ij, x_ij))
    return r_ij, referenceNormals[j]


@wp.func
def computeNearestSurfaceNormal_Func_i_argmin(
    i : wp.int32,  dim: wp.int32,

    xi: vector(dtype = scalar_t, length=Any), hi: scalar_t, mi: scalar_t, rhoi: scalar_t, # type: ignore

    referenceState: Any, # particleDataSoA with the exact type based on the dimensionality

    domainState: domainData,
    kernelProperties: kernelState,

    beginIndex: wp.int32, # type: ignore
    numIndices: wp.int32, # type: ignore
    offsetArray: wp.array(dtype = wp.int64), # type: ignore

    ki : wp.int32, referenceKinds : wp.array(dtype = wp.int32), # type: ignore

    referenceFreeSurfaceMask: wp.array(dtype = scalar_t), # type: ignore
):
    # Forward-only: find the nearest neighbor with referenceFreeSurfaceMask
    # > 0.5, via a loop-carried wp.min-style reassignment -- not used
    # differentiably, only the winning index crosses out (see
    # computeNearestSurfaceNormal_valueAt).
    found = wp.bool(False)
    bestVal = scalar_t(NOT_FOUND_DISTANCE)
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
        if referenceFreeSurfaceMask[j] <= scalar_t(0.5):
            continue

        xj, hj, mj, rhoj, kj = getParticle(referenceState, j)
        x_ij = computeDistanceVec(xi, xj, domainState)
        r_ij = safe_sqrt(wp.dot(x_ij, x_ij))

        hij = computePairwiseSupport(hi, hj, kernelProperties.supportMode)
        if r_ij >= hij and i != j:
            continue

        if r_ij < bestVal:
            bestVal = r_ij
            bestJ = j
            found = wp.bool(True)

    return found, bestVal, bestJ


@wp.func
def computeNearestSurfaceNormal_Func_Adjacency_argmin(
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

    referenceFreeSurfaceMask: wp.array(dtype = scalar_t), # type: ignore
):
    xi, hi, mi, rhoi, ki = getParticle(queryState, i)
    if kernelProperties.operationMode != wp.static(OperationDirection.TrueAllToToAll.value):
        if not checkDirectionality_i(ki, kernelProperties.operationMode):
            return wp.bool(False), wp.int32(0)

    globalFound = wp.bool(False)
    globalBestVal = scalar_t(NOT_FOUND_DISTANCE)
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

        found, bestVal, bestJ = computeNearestSurfaceNormal_Func_i_argmin(
            i, dim,
            xi, hi, mi, rhoi,
            referenceState, domainState,
            kernelProperties,

            beginIndex, numIndices, adjacencyState.neighborList if useAdjacency else gridState.sortIndex,
            ki, referenceState.kinds,

            referenceFreeSurfaceMask,
        )
        if found and bestVal < globalBestVal:
            globalBestVal = bestVal
            globalBestJ = bestJ
            globalFound = wp.bool(True)

    return globalFound, globalBestJ


@wp.kernel
def computeNearestSurfaceNormal_Kernel(
    queryState: Any,
    referenceState: Any,
    domainState: domainData,

    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any,

    kernelProperties: kernelState,
    # Do not change the parameters above
    queryFreeSurfaceMask: wp.array(dtype = scalar_t), # type: ignore
    referenceFreeSurfaceMask: wp.array(dtype = scalar_t), # type: ignore
    queryNormals: wp.array(dtype = vector(length=Any, dtype=scalar_t)), # type: ignore
    referenceNormals: wp.array(dtype = vector(length=Any, dtype=scalar_t)), # type: ignore

    # The last parameters are always the output arrays and should not be changed
    outputDistances : wp.array(dtype = scalar_t), # type: ignore
    outputNormals : wp.array(dtype = vector(length=Any, dtype=scalar_t)), # type: ignore
):
    i = wp.tid()
    numParticles = queryState.positions.shape[0]
    if i >= numParticles:
        return

    # Eq. (47) case 1: a free-surface particle uses its own normal and d^FS = 0.
    if queryFreeSurfaceMask[i] > scalar_t(0.5):
        outputDistances[i] = scalar_t(0.0)
        outputNormals[i] = queryNormals[i]
        return

    xi, hi, mi, rhoi, ki = getParticle(queryState, i)
    found, bestJ = computeNearestSurfaceNormal_Func_Adjacency_argmin(
        i, domainState.dim,
        queryState, referenceState, correctionData, domainState,
        useAdjacency, adjacencyState, gridState, gridState.numOffsets if not useAdjacency else 1,
        kernelProperties,
        referenceFreeSurfaceMask,
    )

    if found:
        # The one and only differentiable evaluation, done here rather than
        # per-offset in the traversal function.
        r_ij, nJ = computeNearestSurfaceNormal_valueAt(xi, referenceState, referenceNormals, bestJ, domainState)
        outputDistances[i] = r_ij
        outputNormals[i] = nJ
    else:
        outputDistances[i] = scalar_t(NOT_FOUND_DISTANCE)
        outputNormals[i] = queryNormals[i]


def _nearestSurfaceDistanceDtype(ctx, extras):
    return castTorchToWarpAsBuiltins(ctx.query.densities).dtype


def _nearestSurfaceNormalDtype(ctx, extras):
    return castTorchToWarpAsBuiltins(ctx.query.positions).dtype


_NEAREST_SURFACE_NORMAL = OperatorSpec(
    kernel=computeNearestSurfaceNormal_Kernel,
    outputs=(
        OutputSpec(dtype=_nearestSurfaceDistanceDtype, shape=ShapeOf.QUERY),
        OutputSpec(dtype=_nearestSurfaceNormalDtype, shape=ShapeOf.QUERY),
    ),
    extras=(
        ExtraSpec("queryFreeSurfaceMask", ExtraKind.TENSOR),
        ExtraSpec("referenceFreeSurfaceMask", ExtraKind.TENSOR),
        ExtraSpec("queryNormals", ExtraKind.TENSOR),
        ExtraSpec("referenceNormals", ExtraKind.TENSOR),
    ),
)


def computeNearestSurfaceNormalWarp(
    queryParticles: ParticleState,
    operationProperties: OperationProperties,
    domain: DomainDescription,

    freeSurfaceMask: torch.Tensor,
    normals: torch.Tensor,

    referenceFreeSurfaceMask: Optional[torch.Tensor] = None,
    referenceNormals: Optional[torch.Tensor] = None,

    queryVolumes: Optional[torch.Tensor] = None, referenceVolumes: Optional[torch.Tensor] = None,
    adjacency: Optional[Union[AdjacencyList, CompactHashMap]] = None,
    referenceParticles: Optional[ParticleState] = None,
    crkState: Optional[CRKState] = None,
    gradHState: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], GradHState]] = None,
    renormalizationState: Optional[Union[torch.Tensor,RenormalizationState]] = None,
):
    referenceParticles = referenceParticles if referenceParticles is not None else queryParticles
    referenceFreeSurfaceMask = referenceFreeSurfaceMask if referenceFreeSurfaceMask is not None else freeSurfaceMask
    referenceNormals = referenceNormals if referenceNormals is not None else normals

    with record_function("warpSPH[computeNearestSurfaceNormal]"):
        ctx = SPHContext(
            query=queryParticles, properties=operationProperties, domain=domain,
            adjacency=adjacency, reference=referenceParticles,
            corrections=Corrections(
                volumes=(queryVolumes, referenceVolumes),
                crk=crkState, gradH=gradHState, renorm=renormalizationState,
            ),
        )
        return launchOperator(
            _NEAREST_SURFACE_NORMAL, ctx,
            queryFreeSurfaceMask=freeSurfaceMask, referenceFreeSurfaceMask=referenceFreeSurfaceMask,
            queryNormals=normals, referenceNormals=referenceNormals,
        )
