"""Warp kernel for De Courcy et al. 2024 Eq. (37)'s `chi_i`, the AC-JST
smoothness switch:

    chi_i = Sum_j |p_i - p_j| / max(|p_i + p_j|, eps) * W_ij * V_j
            -------------------------------------------------------
                        Sum_j W_ij * V_j

No gradient, no CRK/grad-H/renormalisation -- `chi` is a plain kernel-weighted
average of a *pairwise* nonlinear function of both `p_i` and `p_j`, which is
why it needs a dedicated kernel rather than the generic `WarpOperation.Interpolate`
dispatcher (that one only ever interpolates a function of the *reference*
particle's own field). Structurally the simplest custom kernel in this module
family -- no CRK/grad-H boilerplate (`modules/deltaSPH/wp_densityDelta.py`),
no argmin two-pass split (`modules/surfaceDetection/wp_nearestSurfaceNormal.py`,
`modules/shifting/wp_michelUChar.py`) -- just a kernel-value-weighted pairwise
sum, following the same `_Func_i` / `_Func_Adjacency` / `_Kernel` layering as
those.

Returns the raw `(numerator, denominator)` sums rather than dividing inside
the kernel: `computeJstSwitchWarp` does `numerator / clamp(denominator, eps)`
in torch, where autograd handles the division robustly (matching the
`|p_i-p_j|/max(|p_i+p_j|, eps)` guard on the *pairwise* ratio inside the
kernel, which uses `wp.abs` on both numerator and denominator directly --
since only the absolute value of the ratio is ever used, this is exactly
`|(p_i-p_j)/(p_i+p_j)|` wherever `|p_i+p_j| > eps`, and a smooth cap
otherwise, with no sign ambiguity to resolve).
"""

import warp as wp
from warp.types import vector, matrix
from typing import Any, Optional, Tuple, Union
import torch
from torch.profiler import record_function
from warpSPHCore import *

__all__ = ['computeJstSwitchWarp']

#: Floor on both `|p_i+p_j|` (the pairwise ratio's denominator) and the
#: kernel-sum normalisation -- guards division by zero without biasing the
#: ratio anywhere it is not already near-singular.
_EPS = 1.0e-8


@wp.func
def computeJstSwitch_Func_i(
    i: wp.int32, dim: wp.int32,

    xi: vector(dtype=scalar_t, length=Any), hi: scalar_t, mi: scalar_t, rhoi: scalar_t, pi: scalar_t, # type: ignore

    referenceState: Any,
    referencePressures: wp.array(dtype=scalar_t), # type: ignore

    domainState: domainData,
    kernelProperties: kernelState,

    beginIndex: wp.int32, # type: ignore
    numIndices: wp.int32, # type: ignore
    offsetArray: wp.array(dtype=wp.int64), # type: ignore

    ki: wp.int32, referenceKinds: wp.array(dtype=wp.int32), # type: ignore

    useVolume: bool, referenceVolumes: wp.array(dtype=scalar_t), # type: ignore

    numeratorOut: Any, denominatorOut: Any, # type: ignore
):
    numerator = zero_like_warp(numeratorOut)
    denominator = zero_like_warp(denominatorOut)

    for neighborIndex in range(numIndices):
        jj = beginIndex + neighborIndex
        j = wp.int32(offsetArray[jj])
        if kernelProperties.operationMode != wp.static(OperationDirection.TrueAllToToAll.value):
            if not checkDirectionality_j(referenceKinds[j], kernelProperties.operationMode):
                continue
        ##########################################################
        #   The core particle-particle interaction starts here   #
        ##########################################################
        xj, hj, mj, rhoj, kj = getParticle(referenceState, j)
        apparentVolume = mj / rhoj if not useVolume else referenceVolumes[j]

        x_ij = computeDistanceVec(xi, xj, domainState)
        w_ij = sphKernel_ij(x_ij, hi, hj, kernelProperties, domainState)

        pj = referencePressures[j]
        ratio = wp.abs(pi - pj) / wp.max(wp.abs(pi + pj), scalar_t(_EPS))

        numerator += ratio * w_ij * apparentVolume
        denominator += w_ij * apparentVolume

    return numerator, denominator


@wp.func
def computeJstSwitch_Func_Adjacency(
    i: wp.int32, dim: wp.int32,

    queryState: Any,
    referenceState: Any,
    correctionData: Any,

    domainState: domainData,
    useAdjacency: wp.bool,
    adjacencyState: adjacencyData,
    gridState: gridData,
    numOffsets: wp.int32,

    kernelProperties: kernelState,

    queryPressures: wp.array(dtype=scalar_t), referencePressures: wp.array(dtype=scalar_t), # type: ignore

    numeratorOut: Any, denominatorOut: Any, # type: ignore
):
    xi, hi, mi, rhoi, ki = getParticle(queryState, i)
    if kernelProperties.operationMode != wp.static(OperationDirection.TrueAllToToAll.value):
        if not checkDirectionality_i(ki, kernelProperties.operationMode):
            return zero_like_warp(numeratorOut), zero_like_warp(denominatorOut)

    useVolume, Vi = getVolume_i(correctionData, i)
    pi = queryPressures[i]

    numerator = zero_like_warp(numeratorOut)
    denominator = zero_like_warp(denominatorOut)
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

        n, d = computeJstSwitch_Func_i(
            i, dim,
            xi, hi, mi, rhoi, pi,
            referenceState, referencePressures,
            domainState, kernelProperties,
            beginIndex, numIndices, adjacencyState.neighborList if useAdjacency else gridState.sortIndex,
            ki, referenceState.kinds,
            useVolume, correctionData.referenceVolumes,
            numeratorOut, denominatorOut,
        )
        numerator += n
        denominator += d
    return numerator, denominator


@wp.kernel
def computeJstSwitch_Kernel(
    queryState: Any,
    referenceState: Any,
    domainState: domainData,

    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any,

    kernelProperties: kernelState,
    # Do not change the parameters above
    queryPressures: wp.array(dtype=scalar_t), referencePressures: wp.array(dtype=scalar_t), # type: ignore

    # The last parameters are always the output arrays and should not be changed
    outputNumerator: wp.array(dtype=scalar_t), # type: ignore
    outputDenominator: wp.array(dtype=scalar_t), # type: ignore
):
    i = wp.tid()
    numParticles = queryState.positions.shape[0]
    if i >= numParticles:
        return

    numerator, denominator = computeJstSwitch_Func_Adjacency(
        i, domainState.dim,
        queryState, referenceState, correctionData, domainState,
        useAdjacency, adjacencyState, gridState, gridState.numOffsets if not useAdjacency else 1,
        kernelProperties,
        queryPressures, referencePressures,
        zero_like_warp(outputNumerator), zero_like_warp(outputDenominator),
    )
    outputNumerator[i] = numerator
    outputDenominator[i] = denominator


def _jstSwitchDtype(ctx, extras):
    return castTorchToWarpAsBuiltins(ctx.query.densities).dtype


_JST_SWITCH = OperatorSpec(
    kernel=computeJstSwitch_Kernel,
    outputs=(
        OutputSpec(dtype=_jstSwitchDtype, shape=ShapeOf.QUERY),
        OutputSpec(dtype=_jstSwitchDtype, shape=ShapeOf.QUERY),
    ),
    extras=(
        ExtraSpec("queryPressures", ExtraKind.TENSOR),
        ExtraSpec("referencePressures", ExtraKind.TENSOR),
    ),
)


def computeJstSwitchWarp(
    queryParticles: ParticleState,
    operationProperties: OperationProperties,
    domain: DomainDescription,

    pressures: torch.Tensor,
    referencePressures: Optional[torch.Tensor] = None,

    queryVolumes: Optional[torch.Tensor] = None, referenceVolumes: Optional[torch.Tensor] = None,
    adjacency: Optional[Union[AdjacencyList, CompactHashMap]] = None,
    referenceParticles: Optional[ParticleState] = None,
) -> torch.Tensor:
    """Eq. (37)'s `chi_i`, clamped-denominator division of the raw kernel sums."""
    referenceParticles = referenceParticles if referenceParticles is not None else queryParticles
    referencePressures = referencePressures if referencePressures is not None else pressures

    with record_function("warpSPH[computeJstSwitch]"):
        ctx = SPHContext(
            query=queryParticles, properties=operationProperties, domain=domain,
            adjacency=adjacency, reference=referenceParticles,
            corrections=Corrections(volumes=(queryVolumes, referenceVolumes)),
        )
        numerator, denominator = launchOperator(
            _JST_SWITCH, ctx,
            queryPressures=pressures, referencePressures=referencePressures,
        )
    return numerator / torch.clamp(denominator, min=_EPS)
