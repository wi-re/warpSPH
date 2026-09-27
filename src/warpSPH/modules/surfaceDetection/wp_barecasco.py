"""Warp kernel implementation of Barecasco-style surface detection. A first
neighbor pass accumulates a per-particle "cover vector" (sum of unit
`x_ij / (r_ij + 1e-14 * h_i)` directions over kernel-supported neighbors,
the epsilon avoiding a zero-distance divide); a second pass then counts, per
particle, how many neighbors have their direction within
`barecascoThreshold / 2` of the (normalized) cover vector, or fall back to
counting all neighbors when the cover vector's magnitude is `<= 0.5`. The
caller (`barecascoDetection.py`) thresholds this count (`< 0.5`) to get a
surface flag, and uses the normalized cover vector as the surface normal.
"""

import warp as wp
from warp.types import vector, matrix
from typing import Any
import torch
from torch.profiler import profile, ProfilerActivity
from warpSPHCore.profiling import record_function
from typing import Optional, Union, Tuple
from warpSPHCore import *



from ...enumTypes import *

__all__ = ['computeBarecascoSurfaceDetectionWarp']

# @torch.jit.script
# def detectFreeSurfaceBarecasco(
#                                particles : WeaklyCompressibleState,
#                                barecascoThreshold : float,
#                                neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
#                                supportScheme: SupportScheme = SupportScheme.Scatter,):
#     with record_function("[SPH] - [Surface Detection] - Detect Free Surface (Barecasco)"):
#         xij = neighborhood[1].x_ij
#         i, j = neighborhood[0].row, neighborhood[0].col
#         n_ij = torch.nn.functional.normalize(xij, dim = 1)
        
#         coverVector = scatter_sum(-n_ij, i, dim = 0, dim_size = particles.positions.shape[0])
#         normalized = torch.nn.functional.normalize(coverVector)
#         angle = torch.arccos(torch.einsum('ij,ij->i', n_ij, normalized[i]))
#         threshold = barecascoThreshold
#         condition = ((angle <= threshold / 2) & (i != j)) | (torch.linalg.norm(normalized, dim = -1)[i] <= 0.5)
#         # condition = (torch.linalg.norm(normalized, dim = -1)[i] <= 0.5)
#         fs = ~scatter_sum(condition, i, dim = 0, dim_size = particles.positions.shape[0])
#         return fs , normalized


@wp.func
def computeBarecascoSurfaceDetection_Func_i_first(
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

    # Optional Correction Terms:
    # Gradient renormalization matrices for each query point, used for correcting the kernel gradient based on the local particle distribution.
    useGradientRenormalization: wp.bool, Li: matrix(shape=(Any, Any), dtype=scalar_t), # type: ignore
    # Grad-h correction terms for each query and reference point, used for correcting the kernel gradient based on the local particle distribution and smoothing length variations.
    useGradHTerms: wp.bool, omega_i: scalar_t, referenceOmegas: wp.array(dtype = scalar_t),  # type: ignore
    # Whether to use actual volume (mass/density) or apparent volume for the gradient computation, and the corresponding volumes if needed.
    useVolume: bool, Vi: scalar_t, referenceVolumes: wp.array(dtype = scalar_t), # type: ignore
    # Whether to use CRK kernel correction for the computation, and the corresponding correction terms if needed.
    useCRK: bool, Ai: scalar_t, Bi: vector(length=Any, dtype=scalar_t), gradAi: vector(length=Any, dtype=scalar_t), gradBi: matrix(shape=(Any, Any), dtype=scalar_t), # type: ignore
    
    # Dummy value to allow allocation
):
    # Initialize the output value
    out     = zero_like_warp(xi)
    
    # # Loop over neighbors to compute the gradient contribution from each neighbor    
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

        apparentVolume = mj / rhoj if not useVolume else referenceVolumes[j]

        gradw_ij = computeKernelGradientCRK(
            xi, xj, 
            hi, hj,
            kernelProperties, domainState,
            useCRK, Ai, Bi, gradAi, gradBi
        )
        if useGradientRenormalization:
            gradw_ij = matmul(Li, gradw_ij)

        
        w_ij = computeKernelCRK(
            xi, xj,
            hi, hj,
            kernelProperties, domainState,
            useCRK, Ai, Bi
        )

        
        x_ij = computeDistanceVec(xi, xj, domainState)
        r_ij = safe_sqrt(wp.dot(x_ij, x_ij))
        n_ij = x_ij / (r_ij + scalar_t(1.0e-14) * hi)

        if w_ij > 0.0:
            out += n_ij
            
        
    return out



@wp.func
def computeBarecascoSurfaceDetection_Func_Adjacency_first(
    i : wp.int32, dim: wp.int32, lane: wp.int32, lanes: wp.int32, 

    queryState: Any, # particleDataSoA with the exact type based on the dimensionality, e.g., particleDataSoA_2 for 2D, particleDataSoA_3 for 3D, etc.
    referenceState: Any, # particleDataSoA with the exact type based on the dimensionality, e.g., particleDataSoA_2 for 2D, particleDataSoA_3 for 3D, etc.
    correctionData: Any, # correctionData_1 or correctionData_2 or correctionData_3, containing all the optional correction terms and their usage flags

    domainState: domainData,
    useAdjacency: wp.bool,
    adjacencyState: adjacencyData,
    gridState: gridData,
    numOffsets: wp.int32,

    kernelProperties: kernelState, 
):
    xi, hi, mi, rhoi, ki = getParticle(queryState, i)
    if kernelProperties.operationMode != wp.static(OperationDirection.TrueAllToToAll.value):
        if not checkDirectionality_i(ki, kernelProperties.operationMode):
            return zero_like_warp(queryState.positions[i])
        
    useGradientRenormalization, Li = getL_i(correctionData, i)
    useGradHTerms, omega_i = getGradH_i(correctionData, i)
    useVolume, Vi = getVolume_i(correctionData, i)
    useCRK, Ai, Bi, gradA_i, gradB_i = getCRK_i(correctionData, i)
    
    out = zero_like_warp(queryState.positions[i])
    for o in range(numOffsets):
        # grid traversal: lanes take whole cells round-robin (no-op for lanes == 1)
        if not useAdjacency and (o % lanes) != lane:
            continue
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
        
        beginIndex, numIndices = laneSlice(beginIndex, numIndices, lane, lanes, useAdjacency)
        out += computeBarecascoSurfaceDetection_Func_i_first(
            i, dim, 
            xi, hi, mi, rhoi,
            referenceState, domainState,
            kernelProperties,

            beginIndex, numIndices, adjacencyState.neighborList if useAdjacency else gridState.sortIndex,
            ki, referenceState.kinds,

            useGradientRenormalization, Li,
            useGradHTerms, omega_i, correctionData.referenceOmegas,
            useVolume, Vi , correctionData.referenceVolumes,
            useCRK, Ai, Bi, gradA_i, gradB_i,

            # Viscosity function parameters
        )
    return out


@wp.func
def computeBarecascoSurfaceDetection_Func_i_second(
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

    # Optional Correction Terms:
    # Gradient renormalization matrices for each query point, used for correcting the kernel gradient based on the local particle distribution.
    useGradientRenormalization: wp.bool, Li: matrix(shape=(Any, Any), dtype=scalar_t), # type: ignore
    # Grad-h correction terms for each query and reference point, used for correcting the kernel gradient based on the local particle distribution and smoothing length variations.
    useGradHTerms: wp.bool, omega_i: scalar_t, referenceOmegas: wp.array(dtype = scalar_t),  # type: ignore
    # Whether to use actual volume (mass/density) or apparent volume for the gradient computation, and the corresponding volumes if needed.
    useVolume: bool, Vi: scalar_t, referenceVolumes: wp.array(dtype = scalar_t), # type: ignore
    # Whether to use CRK kernel correction for the computation, and the corresponding correction terms if needed.
    useCRK: bool, Ai: scalar_t, Bi: vector(length=Any, dtype=scalar_t), gradAi: vector(length=Any, dtype=scalar_t), gradBi: matrix(shape=(Any, Any), dtype=scalar_t), # type: ignore
    
    coverVector: vector(dtype = scalar_t, length=Any), # type: ignore
    threshold: scalar_t, # type: ignore
    # Dummy value to allow allocation
):
    # Initialize the output value
    out     = zero_like_warp(hi)
    
    # # Loop over neighbors to compute the gradient contribution from each neighbor    
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

        apparentVolume = mj / rhoj if not useVolume else referenceVolumes[j]

        gradw_ij = computeKernelGradientCRK(
            xi, xj, 
            hi, hj,
            kernelProperties, domainState,
            useCRK, Ai, Bi, gradAi, gradBi
        )
        if useGradientRenormalization:
            gradw_ij = matmul(Li, gradw_ij)

        
        x_ij = computeDistanceVec(xi, xj, domainState)
        r_ij = safe_sqrt(wp.dot(x_ij, x_ij))
        n_ij = x_ij / (r_ij + scalar_t(1.0e-14) * hi)
    
        w_ij = computeKernelCRK(
            xi, xj,
            hi, hj,
            kernelProperties, domainState,
            useCRK, Ai, Bi
        )

        if w_ij > 0.0:
            dot = wp.dot(-n_ij, coverVector)
            angle = wp.acos(dot)
            condition = True
            if (angle <= threshold / scalar_t(2.0)) and (i != j):
                condition = False
            if safe_sqrt(wp.dot(coverVector, coverVector)) <= scalar_t(0.5):
                condition = False
            if not condition:
                out += scalar_t(1.0)

        # out += apparentVolume
        
    return out



@wp.func
def computeBarecascoSurfaceDetection_Func_Adjacency_second(
    i : wp.int32, dim: wp.int32, lane: wp.int32, lanes: wp.int32, 

    queryState: Any, # particleDataSoA with the exact type based on the dimensionality, e.g., particleDataSoA_2 for 2D, particleDataSoA_3 for 3D, etc.
    referenceState: Any, # particleDataSoA with the exact type based on the dimensionality, e.g., particleDataSoA_2 for 2D, particleDataSoA_3 for 3D, etc.
    correctionData: Any, # correctionData_1 or correctionData_2 or correctionData_3, containing all the optional correction terms and their usage flags

    domainState: domainData,
    useAdjacency: wp.bool,
    adjacencyState: adjacencyData,
    gridState: gridData,
    numOffsets: wp.int32,

    kernelProperties: kernelState, 

    coverVector: vector(dtype = scalar_t, length=Any), # type: ignore
    threshold: scalar_t, # type: ignore
):
    xi, hi, mi, rhoi, ki = getParticle(queryState, i)
    if kernelProperties.operationMode != wp.static(OperationDirection.TrueAllToToAll.value):
        if not checkDirectionality_i(ki, kernelProperties.operationMode):
            return zero_like_warp(queryState.supports[i])
        
    useGradientRenormalization, Li = getL_i(correctionData, i)
    useGradHTerms, omega_i = getGradH_i(correctionData, i)
    useVolume, Vi = getVolume_i(correctionData, i)
    useCRK, Ai, Bi, gradA_i, gradB_i = getCRK_i(correctionData, i)
    
    out = zero_like_warp(queryState.supports[i])
    for o in range(numOffsets):
        # grid traversal: lanes take whole cells round-robin (no-op for lanes == 1)
        if not useAdjacency and (o % lanes) != lane:
            continue
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
        
        beginIndex, numIndices = laneSlice(beginIndex, numIndices, lane, lanes, useAdjacency)
        out += computeBarecascoSurfaceDetection_Func_i_second(
            i, dim, 
            xi, hi, mi, rhoi,
            referenceState, domainState,
            kernelProperties,

            beginIndex, numIndices, adjacencyState.neighborList if useAdjacency else gridState.sortIndex,
            ki, referenceState.kinds,

            useGradientRenormalization, Li,
            useGradHTerms, omega_i, correctionData.referenceOmegas,
            useVolume, Vi , correctionData.referenceVolumes,
            useCRK, Ai, Bi, gradA_i, gradB_i,

            coverVector, threshold
        )
    return out



@wp.kernel
def computeBarecascoSurfaceDetection_Kernel(
    queryState: Any,
    referenceState: Any,
    domainState: domainData,

    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any,
    
    kernelProperties: kernelState,
    # Do not change the parameters above
    barecascoThreshold: scalar_t, # type: ignore

    # The last parameter is always the output array and should not be changed
    outputNormals : wp.array(dtype = vector(length=Any, dtype=scalar_t)), # type: ignore
    outputValues: wp.array(dtype = scalar_t) # type: ignore
):                                                                                    
    i = wp.tid()
    numParticles = queryState.positions.shape[0]
    if i >= numParticles:
        return



    coverVector = computeBarecascoSurfaceDetection_Func_Adjacency_first(
        i, domainState.dim, 0, 1, 
        queryState, referenceState, correctionData, domainState,
        useAdjacency, adjacencyState, gridState, gridState.numOffsets if not useAdjacency else 1,
        kernelProperties,  #queryKinds, referenceKinds,
        # The parameters above are default parameters and shold not be changed
    )
    normalizedCoverVector = wp.normalize(coverVector)

    outputNormals[i] = normalizedCoverVector

    outputValues[i] = computeBarecascoSurfaceDetection_Func_Adjacency_second(
        i, domainState.dim, 0, 1,
        queryState, referenceState, correctionData, domainState,
        useAdjacency, adjacencyState, gridState, gridState.numOffsets if not useAdjacency else 1,   
        kernelProperties,  #queryKinds, referenceKinds,
        normalizedCoverVector, barecascoThreshold
    )


@wp.kernel
def computeBarecascoSurfaceDetection_KernelTiled(
    queryState: Any,
    referenceState: Any,
    domainState: domainData,

    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any,
    
    kernelProperties: kernelState,
    # Do not change the parameters above
    barecascoThreshold: scalar_t, # type: ignore

    # The last parameter is always the output array and should not be changed
    outputNormals : wp.array(dtype = vector(length=Any, dtype=scalar_t)), # type: ignore
    outputValues: wp.array(dtype = scalar_t) # type: ignore
):
    # Multi-lane variant of computeBarecascoSurfaceDetection_Kernel (warpSPHCore autograd/lanes.py):
    # launched dim=[N, lanes]; each lane walks a slice of i's neighbours.

    i, lane = wp.tid()
    coverPartial = computeBarecascoSurfaceDetection_Func_Adjacency_first(
        i, domainState.dim, lane, wp.block_dim(),
        queryState, referenceState, correctionData, domainState,
        useAdjacency, adjacencyState, gridState, gridState.numOffsets if not useAdjacency else 1,
        kernelProperties,
    )
    normalizedCoverVector = wp.normalize(laneSum(coverPartial))

    valuePartial = computeBarecascoSurfaceDetection_Func_Adjacency_second(
        i, domainState.dim, lane, wp.block_dim(),
        queryState, referenceState, correctionData, domainState,
        useAdjacency, adjacencyState, gridState, gridState.numOffsets if not useAdjacency else 1,
        kernelProperties,
        normalizedCoverVector, barecascoThreshold
    )
    value = laneSum(valuePartial)
    if lane == 0:
        outputNormals[i] = normalizedCoverVector
        outputValues[i] = value


def _barecascoNormalsDtype(ctx, extras):
    return castTorchToWarpAsBuiltins(ctx.query.positions).dtype


def _barecascoValuesDtype(ctx, extras):
    return castTorchToWarpAsBuiltins(ctx.query.densities).dtype


_BARECASCO_SURFACE_DETECTION = OperatorSpec(
    kernel=computeBarecascoSurfaceDetection_Kernel,
    tiledKernel=computeBarecascoSurfaceDetection_KernelTiled,
    outputs=(
        OutputSpec(dtype=_barecascoNormalsDtype, shape=ShapeOf.QUERY),
        OutputSpec(dtype=_barecascoValuesDtype, shape=ShapeOf.QUERY),
    ),
    extras=(ExtraSpec("barecascoThreshold", ExtraKind.SCALAR),),
)


def computeBarecascoSurfaceDetectionWarp(
    queryParticles: ParticleState,
    operationProperties: OperationProperties,
    domain: DomainDescription,

    barecascoThreshold: float = 1.5, # type: ignore

    queryVolumes: Optional[torch.Tensor] = None, referenceVolumes: Optional[torch.Tensor] = None,
    adjacency: Optional[Union[AdjacencyList, CompactHashMap]] = None, # if none a datastructure is created for EVERY operation!,
    referenceParticles: Optional[ParticleState] = None,
    crkState: Optional[CRKState] = None,
    gradHState: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], GradHState]] = None,
    renormalizationState: Optional[Union[torch.Tensor,RenormalizationState]] = None,
):
    with record_function("warpSPH[computeBarecascoSurfaceDetection]"):
        with record_function("warpSPH[computeBarecascoSurfaceDetection] - Preprocessing"):
            # Preprocessing and input validation
            device = queryParticles.positions.device
            # args, device, dim = parseArguments(
            #     queryParticles, operationProperties, domain,
            #     queryVolumes, referenceVolumes,
            #     adjacency,
            #     referenceParticles,
            #     crkState,
            #     gradHState,
            #     renormalizationState,
            # )

            referenceParticles = referenceParticles if referenceParticles is not None else queryParticles

        with record_function("warpSPH[computeBarecascoSurfaceDetection] - Kernel Execution"):
            ctx = SPHContext(
                query=queryParticles, properties=operationProperties, domain=domain,
                adjacency=adjacency, reference=referenceParticles,
                corrections=Corrections(
                    volumes=(queryVolumes, referenceVolumes),
                    crk=crkState, gradH=gradHState, renorm=renormalizationState,
                ),
            )
            return launchOperator(
                _BARECASCO_SURFACE_DETECTION, ctx,
                barecascoThreshold=scalar_t(barecascoThreshold),
            )


        # with record_function("warpSPH[CRKVolume] - Kernel Execution"):
        #     warp_result = warpWrapper(
        #         launch_kernel, computeBarecascoSurfaceDetection_Kernel, outputSize, outputDtype,
        #         *args,
        #         queryVelocities_, referenceVelocities_,
        #         queryEnergies_, referenceEnergies_,
        #         individual_cs, queryCs_, referenceCs_,
        #         viscositySwitch, queryAlphas_, referenceAlphas_,
        #         explicitPressure, queryPressures_, referencePressures_,
        #         conductivityParams
        #     )

    # return warp_result

