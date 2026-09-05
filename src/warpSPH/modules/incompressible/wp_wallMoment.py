"""The kernel-weighted position moment Eq. (61)'s wall pressure needs:

    M_i = sum_j V_j rho_j (x_i - x_j) W_ij

per query row, gathered over whichever direction `operationProperties` selects
(`FluidToBoundary` for the wall BC). Dotting it with `(g - a_w)` and dividing by
the Shepard denominator `sum_j V_j W_ij` gives Adami et al. 2012's hydrostatic
correction, and De Courcy et al. 2024 Eq. (61) with it.

**Why this is a kernel and not two `Interpolate` gathers.** It was two gathers:
`x_i sum_j V_j rho_j W_ij - sum_j V_j rho_j x_j W_ij`, valid because `(g - a_w)`
is a per-*query* quantity and factors out of the sum. That algebra is exact --
but only in an unwrapped domain. Splitting `x_i - x_j` across two gathers
discards the minimum-image convention, so a pair that wraps contributes `+-L_d`
of error per periodic direction `d`. Here `x_ij` comes from
`computeDistanceVec`, the same call every other operator in this package uses,
so the moment is minimum-image correct by construction and periodic domains need
no restriction, no guard, and no case-level workaround.

The measured cost of the old restriction: on `hydrostaticColumn` (periodic,
walled, gravity along a periodic axis) the guard tripped once fluid penetrated
the wall band far enough to become a wrapped neighbour of the opposite wall --
step 45 at `band=5`, step 72 at `band=8`. Deeper walls only delayed it, because
the trigger was penetration depth, which grows.

`rho_j` is read from the reference state rather than folded into the volume, so
the operator is still correct when apparent volumes are supplied (where
`V_j != m_j/rho_j` and `V_j rho_j != m_j`).
"""

import warp as wp
from warp.types import vector, matrix
from typing import Any, Optional, Tuple, Union

import torch
from torch.profiler import record_function
from warpSPHCore import *

__all__ = ['computeWallMomentWarp']


@wp.func
def computeWallMoment_Func_i(
    # General Shape Parameters and indices
    i : wp.int32,  dim: wp.int32,

    # SPH properties for the query set (indexed by i)
    xi: vector(dtype = scalar_t, length=Any), hi: scalar_t, mi: scalar_t, rhoi: scalar_t, # type: ignore

    # SPH properties for the reference set (indexed by j in the neighbor loop)
    referenceState: Any,

    # Domain and kernel parameters
    domainState: domainData,
    kernelProperties: kernelState,

    beginIndex: wp.int32, # type: ignore
    numIndices: wp.int32, # type: ignore
    offsetArray: wp.array(dtype = wp.int64), # type: ignore

    # Operation Mode for masking certain kinds of interactions
    ki : wp.int32, referenceKinds : wp.array(dtype = wp.int32), # type: ignore

    # Optional Correction Terms (accepted for signature parity with the other
    # operators; only the volume one is meaningful for a plain moment).
    useGradientRenormalization: wp.bool, Li: matrix(shape=(Any, Any), dtype=scalar_t), # type: ignore
    useGradHTerms: wp.bool, omega_i: scalar_t, referenceOmegas: wp.array(dtype = scalar_t),  # type: ignore
    useVolume: bool, Vi: scalar_t, referenceVolumes: wp.array(dtype = scalar_t), # type: ignore
    useCRK: bool, Ai: scalar_t, Bi: vector(length=Any, dtype=scalar_t), gradAi: vector(length=Any, dtype=scalar_t), gradBi: matrix(shape=(Any, Any), dtype=scalar_t), # type: ignore

    outputValue: Any, # type: ignore
):
    out = zero_like_warp(outputValue)

    for neighborIndex in range(numIndices):
        jj = beginIndex + neighborIndex
        j  = wp.int32(offsetArray[jj])
        if kernelProperties.operationMode != wp.static(OperationDirection.TrueAllToToAll.value):
            if not checkDirectionality_j(referenceKinds[j], kernelProperties.operationMode):
                continue

        xj, hj, mj, rhoj, kj = getParticle(referenceState, j)
        apparentVolume = mj / rhoj if not useVolume else referenceVolumes[j]

        kernel = computeKernelCRK(xi, xj, hi, hj, kernelProperties, domainState,
                                  useCRK, Ai, Bi)
        # The whole point of the kernel: minimum-image, like every other
        # operator here. See the module docstring.
        x_ij = computeDistanceVec(xi, xj, domainState)

        out += apparentVolume * rhoj * kernel * x_ij

    return out


@wp.func
def computeWallMoment_Func_Adjacency(
    i : wp.int32, dim: wp.int32,

    queryState: Any,
    referenceState: Any,
    correctionData: Any,

    domainState: domainData,
    useAdjacency: wp.bool,
    adjacencyState: adjacencyData,
    gridState: gridData,
    numOffsets: wp.int32,

    kernelProperties: kernelState,

    outputValue : Any, # type: ignore
):
    xi, hi, mi, rhoi, ki = getParticle(queryState, i)
    if kernelProperties.operationMode != wp.static(OperationDirection.TrueAllToToAll.value):
        if not checkDirectionality_i(ki, kernelProperties.operationMode):
            return zero_like_warp(outputValue)

    useGradientRenormalization, Li = getL_i(correctionData, i)
    useGradHTerms, omega_i = getGradH_i(correctionData, i)
    useVolume, Vi = getVolume_i(correctionData, i)
    useCRK, Ai, Bi, gradA_i, gradB_i = getCRK_i(correctionData, i)

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

        out += computeWallMoment_Func_i(
            i, dim,
            xi, hi, mi, rhoi,
            referenceState, domainState,
            kernelProperties,

            beginIndex, numIndices, adjacencyState.neighborList if useAdjacency else gridState.sortIndex,
            ki, referenceState.kinds,

            useGradientRenormalization, Li,
            useGradHTerms, omega_i, correctionData.referenceOmegas,
            useVolume, Vi, correctionData.referenceVolumes,
            useCRK, Ai, Bi, gradA_i, gradB_i,

            outputValue,
        )
    return out


@wp.kernel
def computeWallMoment_Kernel(
    queryState: Any,
    referenceState: Any,
    domainState: domainData,

    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any,

    kernelProperties: kernelState,
    # The last parameter is always the output array and should not be changed
    outputValues : wp.array(dtype = vector(length=Any, dtype=scalar_t)) # type: ignore
):
    i = wp.tid()
    numParticles = queryState.positions.shape[0]
    if i >= numParticles:
        return

    outputValues[i] = computeWallMoment_Func_Adjacency(
        i, domainState.dim,
        queryState, referenceState, correctionData, domainState,
        useAdjacency, adjacencyState, gridState, gridState.numOffsets if not useAdjacency else 1,
        kernelProperties,
        zero_like_warp(outputValues)
    )


def _wallMomentDtype(ctx, extras):
    return castTorchToWarpAsBuiltins(ctx.query.positions).dtype


_WALL_MOMENT = OperatorSpec(
    kernel=computeWallMoment_Kernel,
    outputs=(OutputSpec(dtype=_wallMomentDtype, shape=ShapeOf.QUERY),),
    extras=(),
)


def computeWallMomentWarp(
    queryParticles: ParticleState,
    operationProperties: OperationProperties,
    domain: DomainDescription,

    queryVolumes: Optional[torch.Tensor] = None, referenceVolumes: Optional[torch.Tensor] = None,
    adjacency: Optional[Union[AdjacencyList, CompactHashMap]] = None,
    referenceParticles: Optional[ParticleState] = None,
    crkState: Optional[CRKState] = None,
    gradHState: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], GradHState]] = None,
    renormalizationState: Optional[Union[torch.Tensor, RenormalizationState]] = None,
) -> torch.Tensor:
    """`sum_j V_j rho_j (x_i - x_j) W_ij`, one vector per query row."""
    with record_function("warpSPH[computeWallMoment]"):
        referenceParticles = referenceParticles if referenceParticles is not None else queryParticles
        ctx = SPHContext(
            query=queryParticles, properties=operationProperties, domain=domain,
            adjacency=adjacency, reference=referenceParticles,
            corrections=Corrections(
                volumes=(queryVolumes, referenceVolumes),
                crk=crkState, gradH=gradHState, renorm=renormalizationState,
            ),
        )
        return launchOperator(_WALL_MOMENT, ctx)
