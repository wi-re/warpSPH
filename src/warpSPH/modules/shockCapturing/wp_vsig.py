"""Warp kernel for the Monaghan-style signal velocity v_sig = max_j(c_bar_ij - mu_ij, mu_ij <= 0) used by the viscosity switches.

Split into a forward-only argmax pass (``computeVsig_Func_i_argmax`` /
``computeVsig_Func_Adjacency_argmax``) that finds the winning neighbor index
via a loop-carried ``wp.max``-style reassignment, followed by a single
re-evaluation of the actual v_sig formula for just that index
(``computeVsig_valueAt``), done outside any loop. This two-pass split exists
because warp-lang 1.15.0's reverse-mode AD silently zeroes the adjoint of a
value produced by loop-carried max reassignment (see git history: "vsig
computation doesnt work due to the loop max function") -- the argmax pass is
safe because only the winning *index* (an int, never differentiated by Warp)
crosses out of it. ``computeVsigWarp`` is the torch-facing entry point.
"""

import warp as wp
from warp.types import vector, matrix
from typing import Any
import torch
from torch.profiler import profile, ProfilerActivity
from warpSPHCore.profiling import record_function
from typing import Optional, Union, Tuple
from warpSPHCore import *

__all__ = ['computeVsigWarp']


@wp.func
def computeVsig_valueAt(
    # General Shape Parameters and indices
    dim: wp.int32,

    # SPH properties for the query set
    xi: vector(dtype = scalar_t, length=Any), hi: scalar_t, # type: ignore

    # SPH properties for the reference set (indexed by j, a single, already-known index)
    referenceState: Any, # particleDataSoA with the exact type based on the dimensionality, e.g., particleDataSoA_2 for 2D, particleDataSoA_3 for 3D, etc.
    j: wp.int32,

    domainState: domainData,

    vel_i: vector(length=Any, dtype=scalar_t), referenceVelocities: wp.array(dtype = vector(length=Any, dtype=scalar_t)), # type: ignore
    individual_cs: wp.bool, cs_i: scalar_t, referenceCs: wp.array(dtype = scalar_t), # type: ignore
) -> scalar_t:
    # The per-neighbor vsig formula, evaluated once for a single, already-known
    # neighbor index -- deliberately *not* inside a loop. See this file's module
    # docstring: this is the piece that must stay outside the dynamic neighbor loop
    # for Warp's reverse-mode AD to differentiate it correctly.
    xj, hj, mj, rhoj, kj = getParticle(referenceState, j)
    vel_j = referenceVelocities[j]
    cs_j = access_optional(referenceCs, j, individual_cs, scalar_t(1.0))

    x_ij = computeDistanceVec(xi, xj, domainState)
    r_ij = safe_sqrt(wp.dot(x_ij, x_ij))

    v_ij = vel_i - vel_j
    mu_ij = wp.dot(v_ij, x_ij) / (r_ij + scalar_t(1.0e-14) * hi)

    c_bar = scalar_t(0.5) * (cs_i + cs_j)
    vsigs = c_bar - mu_ij
    if mu_ij > 0:
        vsigs = scalar_t(0.0)
    return vsigs


@wp.func
def computeVsig_Func_i_argmax(
    # General Shape Parameters and indices
    i : wp.int32,  dim: wp.int32,

    # SPH properties for the query set (indexed by i)
    xi: vector(dtype = scalar_t, length=Any), hi: scalar_t, mi: scalar_t, rhoi: scalar_t, # type: ignore

    # SPH properties for the reference set (indexed by j in the neighbor loop)
    referenceState: Any, # particleDataSoA with the exact type based on the dimensionality, e.g., particleDataSoA_2 for 2D, particleDataSoA_3 for 3D, etc.

    # Domain and kernel parameters
    # periodicity : wp.array(dtype = wp.bool), domainMin : wp.array(dtype = scalar_t), domainMax : wp.array(dtype = scalar_t), # type: ignore
    domainState: domainData,
    kernelProperties: kernelState,

    # Operation specific parameters
     # type: ignore

    beginIndex: wp.int32, # type: ignore
    numIndices: wp.int32, # type: ignore
    offsetArray: wp.array(dtype = wp.int64), # type: ignore

    # Operation Mode for masking certain kinds of interactions, e.g. for directional operations
    ki : wp.int32, referenceKinds : wp.array(dtype = wp.int32), # type: ignore

    vel_i: vector(length=Any, dtype=scalar_t), referenceVelocities: wp.array(dtype = vector(length=Any, dtype=scalar_t)), # type: ignore

    individual_cs: wp.bool, cs_i: scalar_t, referenceCs: wp.array(dtype = scalar_t), # type: ignore
):
    # Forward-only pass: find which neighbor achieves the max vsig, and its value,
    # via the exact same loop-carried wp.max reassignment as before. That pattern is
    # confirmed to silently zero this value's own adjoint under warp-lang 1.15.0 (see
    # this file's module docstring) -- but nothing here is used differentiably: only
    # the winning *index* (an int, which Warp never differentiates) crosses into the
    # caller. computeVsig_valueAt() recomputes the actual differentiable value for
    # that one index afterward, outside any loop.
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
        vel_j = referenceVelocities[j]
        cs_j = access_optional(referenceCs, j, individual_cs, scalar_t(1.0))

        x_ij = computeDistanceVec(xi, xj, domainState)
        r_ij = safe_sqrt(wp.dot(x_ij, x_ij))
        # Unlike every sibling module in this family (wp_dilate.py, wp_sum.py, ...),
        # this loop had no compact-support filter at all -- every entry in the
        # candidate list was treated as a genuine neighbor unconditionally. That was
        # invisible for the AdjacencyList path (radiusSearchCompactHashMap already
        # returns an exact, pre-filtered neighbor list) but wrong for grid traversal
        # (checkOffset returns every particle in a nearby *cell*, which is coarser
        # than the exact kernel support radius) -- confirmed by comparing grid vs.
        # AdjacencyList results on the same particle set before/after this check:
        # they disagreed without it and match exactly with it.
        hij = computePairwiseSupport(hi, hj, kernelProperties.supportMode)
        if r_ij >= hij and i != j:
            continue

        v_ij = vel_i - vel_j
        mu_ij = wp.dot(v_ij, x_ij) / (r_ij + scalar_t(1.0e-14) * hi)

        c_bar = scalar_t(0.5) * (cs_i + cs_j)
        vsigs = c_bar - mu_ij
        if mu_ij > 0:
            vsigs = scalar_t(0.0)

        if vsigs > bestVal:
            bestVal = vsigs
            bestJ = j
            found = wp.bool(True)

    return found, bestVal, bestJ



@wp.func
def computeVsig_Func_Adjacency_argmax(
    i : wp.int32, dim: wp.int32,

    queryState: Any, # particleDataSoA with the exact type based on the dimensionality, e.g., particleDataSoA_2 for 2D, particleDataSoA_3 for 3D, etc.
    referenceState: Any, # particleDataSoA with the exact type based on the dimensionality, e.g., particleDataSoA_2 for 2D, particleDataSoA_3 for 3D, etc.
    correctionData: Any, # correctionData_1 or correctionData_2 or correctionData_3, containing all the optional correction terms and their usage flags

    domainState: domainData,
    useAdjacency: wp.bool,
    adjacencyState: adjacencyData,
    gridState: gridData,
    numOffsets: wp.int32,

    kernelProperties: kernelState,

    queryVelocities: wp.array(dtype = vector(length=Any, dtype=scalar_t)), referenceVelocities: wp.array(dtype = vector(length=Any, dtype=scalar_t)), # type: ignore
    individual_cs: wp.bool, queryCs: wp.array(dtype = scalar_t), referenceCs: wp.array(dtype = scalar_t), # type: ignore
):
    # Forward-only, mirrors computeVsig_Func_i_argmax's own argmax approach, but one
    # level up: finds the single neighbor with the globally largest vsig across
    # *every* offset, not a per-offset winner. This must compare across offsets
    # rather than accumulate (+=) them -- unlike a linear accumulation (safe to sum
    # across a dynamic outer loop, as computeAlphaWarp's fix relies on), each
    # offset's own vsig winner is already a *max*, and summing several offsets'
    # maxes together is a different (and for grid traversal, wrong -- up to 27
    # offsets in 3D) quantity than the max over their union. checkDirectionality_i's
    # early-return case is folded into "not found" here (dim/mi/rhoi/xi/hi are read
    # once by the caller and passed through unused by this function's early-return
    # path in the original, so nothing is lost by returning found=False instead of a
    # zero value).
    xi, hi, mi, rhoi, ki = getParticle(queryState, i)
    if kernelProperties.operationMode != wp.static(OperationDirection.TrueAllToToAll.value):
        if not checkDirectionality_i(ki, kernelProperties.operationMode):
            return wp.bool(False), wp.int32(0)

    vel_i = queryVelocities[i]
    cs_i = access_optional(queryCs, i, individual_cs, scalar_t(1.0))

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

        found, bestVal, bestJ = computeVsig_Func_i_argmax(
            i, dim,
            xi, hi, mi, rhoi,
            referenceState, domainState,
            kernelProperties,

            beginIndex, numIndices, adjacencyState.neighborList if useAdjacency else gridState.sortIndex,
            ki, referenceState.kinds,

            vel_i, referenceVelocities,
            individual_cs, cs_i, referenceCs,
        )
        if found and bestVal > globalBestVal:
            globalBestVal = bestVal
            globalBestJ = bestJ
            globalFound = wp.bool(True)

    return globalFound, globalBestJ



@wp.kernel
def computeVsig_Kernel(
    queryState: Any,
    referenceState: Any,
    domainState: domainData,

    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any,
    
    kernelProperties: kernelState,
    # Do not change the parameters above
    queryVelocities: wp.array(dtype = vector(length=Any, dtype=scalar_t)), referenceVelocities: wp.array(dtype = vector(length=Any, dtype=scalar_t)), # type: ignore
    individual_cs: wp.bool, queryCs: wp.array(dtype = scalar_t), referenceCs: wp.array(dtype = scalar_t), # type: ignore
    # The last parameter is always the output array and should not be changed
    outputValues : wp.array(dtype = scalar_t) # type: ignore
):                                                                                    
    i = wp.tid()
    numParticles = queryState.positions.shape[0]
    if i >= numParticles:
        return

    found, bestJ = computeVsig_Func_Adjacency_argmax(
        i, domainState.dim,
        queryState, referenceState, correctionData, domainState,
        useAdjacency, adjacencyState, gridState, gridState.numOffsets if not useAdjacency else 1,
        kernelProperties,  #queryKinds, referenceKinds,
        # The parameters above are default parameters and shold not be changed
        queryVelocities, referenceVelocities,
        individual_cs, queryCs, referenceCs,
    )

    if found:
        # The one and only differentiable evaluation, done here in the primary
        # caller rather than per-offset in the traversal function -- see this
        # file's module docstring and computeVsig_Func_Adjacency_argmax's own
        # docstring for why summing per-offset winners would be wrong (grid
        # traversal visits up to 27 offsets in 3D) as well as non-differentiable.
        xi, hi, mi, rhoi, ki = getParticle(queryState, i)
        vel_i = queryVelocities[i]
        cs_i = access_optional(queryCs, i, individual_cs, scalar_t(1.0))
        outputValues[i] = computeVsig_valueAt(
            domainState.dim, xi, hi, referenceState, bestJ, domainState,
            vel_i, referenceVelocities,
            individual_cs, cs_i, referenceCs,
        )
    else:
        outputValues[i] = zero_like_warp(outputValues)

def _vsigDtype(ctx, extras):
    return castTorchToWarpAsBuiltins(ctx.query.densities).dtype


_VSIG = OperatorSpec(
    kernel=computeVsig_Kernel,
    outputs=(OutputSpec(dtype=_vsigDtype, shape=ShapeOf.QUERY),),
    extras=(
        ExtraSpec("queryVelocities", ExtraKind.TENSOR),
        ExtraSpec("referenceVelocities", ExtraKind.TENSOR),
        ExtraSpec("individualCs", ExtraKind.SCALAR),
        ExtraSpec("queryCs", ExtraKind.TENSOR),
        ExtraSpec("referenceCs", ExtraKind.TENSOR),
    ),
)


def computeVsigWarp(
    queryParticles: ParticleState,
    operationProperties: OperationProperties,
    domain: DomainDescription,
    
    queryVelocities : Optional[torch.Tensor] = None, referenceVelocities: Optional[torch.Tensor] = None,
    queryCs: Optional[torch.Tensor] = None, referenceCs: Optional[torch.Tensor] = None,
    
    queryVolumes: Optional[torch.Tensor] = None, referenceVolumes: Optional[torch.Tensor] = None,
    adjacency: Optional[Union[AdjacencyList, CompactHashMap]] = None, # if none a datastructure is created for EVERY operation!,
    referenceParticles: Optional[ParticleState] = None,
    crkState: Optional[CRKState] = None,
    gradHState: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], GradHState]] = None,
    renormalizationState: Optional[Union[torch.Tensor,RenormalizationState]] = None,
):
    if referenceVelocities is None:
        referenceVelocities = queryVelocities
    if referenceCs is None:
        referenceCs = queryCs
    with record_function("warpSPH[computeVsig]"):
        with record_function("warpSPH[computeVsig] - Preprocessing"):
            # Preprocessing and input validation
            # args, device, dim = parseArguments(
            #     queryParticles, operationProperties, domain,
            #     queryVolumes, referenceVolumes,
            #     adjacency,
            #     referenceParticles,
            #     crkState,
            #     gradHState,
            #     renormalizationState,
            # )
            device = queryParticles.positions.device

            referenceParticles = referenceParticles if referenceParticles is not None else queryParticles
            queryVelocities_ = queryVelocities if queryVelocities is not None else (queryParticles.velocities if hasattr(queryParticles, 'velocities') else None)
            queryCs_ = queryCs if queryCs is not None else (queryParticles.soundspeeds if hasattr(queryParticles, 'soundspeeds') else getCachedDummyTensor((1,), dtype=get_torch_precision(), device=device))
            
            referenceVelocities_ = referenceVelocities if referenceVelocities is not None else (referenceParticles.velocities if hasattr(referenceParticles, 'velocities') else None)
            referenceCs_ = referenceCs if referenceCs is not None else (referenceParticles.soundspeeds if hasattr(referenceParticles, 'soundspeeds') else getCachedDummyTensor((1,), dtype=get_torch_precision(), device=device))
            
            if queryCs is not None or (hasattr(queryParticles, 'soundspeeds') and queryParticles.soundspeeds is not None):
                individual_cs = True
            else:            
                individual_cs = False
            
            if queryVelocities_ is None:
                raise ValueError("Velocities must be provided either through queryVelocities or as a property of queryParticles.")

        with record_function("warpSPH[computeVsig] - Kernel Execution"):
            ctx = SPHContext(
                query=queryParticles, properties=operationProperties, domain=domain,
                adjacency=adjacency, reference=referenceParticles,
                corrections=Corrections(
                    volumes=(queryVolumes, referenceVolumes),
                    crk=crkState, gradH=gradHState, renorm=renormalizationState,
                ),
            )
            return launchOperator(
                _VSIG, ctx,
                queryVelocities=queryVelocities_, referenceVelocities=referenceVelocities_,
                individualCs=individual_cs, queryCs=queryCs_, referenceCs=referenceCs_,
            )

        # with record_function("warpSPH[CRKVolume] - Kernel Execution"):
        #     warp_result = warpWrapper(
        #         launch_kernel, computeVsig_Kernel, outputSize, outputDtype,
        #         *args,
        #         queryVelocities_, referenceVelocities_,
        #         individual_cs, queryCs_, referenceCs_,
        #         viscositySwitch, queryAlphas_, referenceAlphas_,
        #         explicitPressure, queryPressures_, referencePressures_,
        #         viscosityParams
        #     )

    return warp_result
