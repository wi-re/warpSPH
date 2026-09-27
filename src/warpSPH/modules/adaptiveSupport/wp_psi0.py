"""Warp kernel computing the per-particle Owen psi_0/psi_0_H statistics (kernel-value and kernel-gradient sums, h-rescaled and dim-th-rooted) used to drive the Owen adaptive-support lookup table.

``computePsi0Warp`` is the torch-facing entry point; ``psi_0`` is the
``1/dim``-th root of the accumulated (h-rescaled) kernel-value sum and
``psi_0_H`` the same for the kernel-gradient-norm sum, matching the
quantities the Owen lookup table (``owenLUT.py``) was built against.
"""

import warp as wp
from warp.types import vector, matrix
from typing import Any
import torch
from torch.profiler import profile, ProfilerActivity
from warpSPHCore.profiling import record_function
from typing import Optional, Union, Tuple
from warpSPHCore import *

__all__ = ['computePsi0Warp']


@wp.func
def computePsi0_Func_i_raw(
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
    useGradientRenormalization: wp.bool, Li: matrix(shape=(dim_t, dim_t), dtype=scalar_t), # type: ignore
    # Grad-h correction terms for each query and reference point, used for correcting the kernel gradient based on the local particle distribution and smoothing length variations.
    useGradHTerms: wp.bool, omega_i: scalar_t, referenceOmegas: wp.array(dtype = scalar_t),  # type: ignore
    # Whether to use actual volume (mass/density) or apparent volume for the gradient computation, and the corresponding volumes if needed.
    useVolume: bool, Vi: scalar_t, referenceVolumes: wp.array(dtype = scalar_t), # type: ignore
    # Whether to use CRK kernel correction for the computation, and the corresponding correction terms if needed.
    useCRK: bool, Ai: scalar_t, Bi: vector(length=Any, dtype=scalar_t), gradAi: vector(length=Any, dtype=scalar_t), gradBi: matrix(shape=(Any, Any), dtype=scalar_t), # type: ignore
    
):
    # Initialize the output value
    psi0  = zero_like_warp(rhoi)
    psi0h = zero_like_warp(rhoi)
    
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
        # Omega Functionality begins here        

        # dim = particles.positions.shape[1]    
        # kernelValues = neighborhood[1]
        
        # h = particles.supports[i]
        h = hi
        hReferenceFactor_W = scalar_t(1.0)**(-scalar_t(dim))
        hActualFactor_W = h**(-scalar_t(dim))
        hScaling_W = hReferenceFactor_W / hActualFactor_W
        
        hReferenceFactor_WH = scalar_t(1.0)**(-scalar_t(dim + 1))
        hActualFactor_WH = h**(-scalar_t(dim + 1))
        hScaling_WH = hReferenceFactor_WH / hActualFactor_WH
        
        # print(particles.positions.shape, i.shape, j.shape)
        # print(get_i(particles.positions, i).shape, get_j(particles.positions, j).shape)
        # print(domain)
        
        # xij = kernelValues.x_ij
        
        kTerm = hScaling_W * sphKernel(
            xi, xj, 
            hi, hj, 
            kernelProperties, domainState,                   
        )
        gradW = sphKernelGradient(
            xi, xj,
            hi, hj,
            kernelProperties, domainState,    
        )
        gradW_norm = safe_sqrt(wp.dot(gradW, gradW))

        kTerm_H = hScaling_WH * gradW_norm

        psi0 += kTerm
        psi0h += kTerm_H
    return psi0, psi0h


@wp.func
def computePsi0_Func_i(
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
    useGradientRenormalization: wp.bool, Li: matrix(shape=(dim_t, dim_t), dtype=scalar_t), # type: ignore
    # Grad-h correction terms for each query and reference point, used for correcting the kernel gradient based on the local particle distribution and smoothing length variations.
    useGradHTerms: wp.bool, omega_i: scalar_t, referenceOmegas: wp.array(dtype = scalar_t),  # type: ignore
    # Whether to use actual volume (mass/density) or apparent volume for the gradient computation, and the corresponding volumes if needed.
    useVolume: bool, Vi: scalar_t, referenceVolumes: wp.array(dtype = scalar_t), # type: ignore
    # Whether to use CRK kernel correction for the computation, and the corresponding correction terms if needed.
    useCRK: bool, Ai: scalar_t, Bi: vector(length=Any, dtype=scalar_t), gradAi: vector(length=Any, dtype=scalar_t), gradBi: matrix(shape=(Any, Any), dtype=scalar_t), # type: ignore
    
):
    # The dim-th roots are applied to the finished neighbour sum, so the
    # multi-lane kernel (which must reduce the raw sums across lanes first)
    # calls `computePsi0_Func_i_raw` instead; this path is unchanged.
    psi0, psi0h = computePsi0_Func_i_raw(
        i, dim, xi, hi, mi, rhoi, referenceState, domainState, kernelProperties,
        beginIndex, numIndices, offsetArray, ki, referenceKinds,
        useGradientRenormalization, Li, useGradHTerms, omega_i, referenceOmegas,
        useVolume, Vi, referenceVolumes, useCRK, Ai, Bi, gradAi, gradBi)
    return psi0**(scalar_t(1.0)/scalar_t(dim)), psi0h**(scalar_t(1.0)/scalar_t(dim))



@wp.func
def computePsi0_Func_Adjacency(
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
            return scalar_t(0.0), scalar_t(0.0)
        
    useGradientRenormalization, Li = getL_i(correctionData, i)
    useGradHTerms, omega_i = getGradH_i(correctionData, i)
    useVolume, Vi = getVolume_i(correctionData, i)
    useCRK, Ai, Bi, gradA_i, gradB_i = getCRK_i(correctionData, i)

    out_psi0 = zero_like_warp(queryState.densities) # type: ignore
    out_psi0h = zero_like_warp(queryState.densities)
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
        psi0_, psi0h_ = computePsi0_Func_i(
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

            # Omega function parameters
        )
        out_psi0 += psi0_
        out_psi0h += psi0h_
    return out_psi0, out_psi0h



@wp.func
def computePsi0_Func_Adjacency_raw(
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
            return scalar_t(0.0), scalar_t(0.0)
        
    useGradientRenormalization, Li = getL_i(correctionData, i)
    useGradHTerms, omega_i = getGradH_i(correctionData, i)
    useVolume, Vi = getVolume_i(correctionData, i)
    useCRK, Ai, Bi, gradA_i, gradB_i = getCRK_i(correctionData, i)

    out_psi0 = zero_like_warp(queryState.densities) # type: ignore
    out_psi0h = zero_like_warp(queryState.densities)
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
        psi0_, psi0h_ = computePsi0_Func_i_raw(
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

            # Omega function parameters
        )
        out_psi0 += psi0_
        out_psi0h += psi0h_
    return out_psi0, out_psi0h



@wp.kernel
def computePsi0_Kernel(
    queryState: Any,
    referenceState: Any,
    domainState: domainData,

    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any,
    
    kernelProperties: kernelState,
    # Do not change the parameters above
    
    # The last parameter is always the output array and should not be changed
    output_psi0 : wp.array(dtype = scalar_t), # type: ignore
    output_psi0_H : wp.array(dtype = scalar_t) # type: ignore
):                                                                                    
    i = wp.tid()
    numParticles = queryState.positions.shape[0]
    if i >= numParticles:
        return

    psi0, psi0h = computePsi0_Func_Adjacency(
        i, domainState.dim, 0, 1, 
        queryState, referenceState, correctionData, domainState,
        useAdjacency, adjacencyState, gridState, gridState.numOffsets if not useAdjacency else 1,
        kernelProperties,  #queryKinds, referenceKinds,
        # The parameters above are default parameters and shold not be changed
    )
    output_psi0[i] = psi0
    output_psi0_H[i] = psi0h

@wp.kernel
def computePsi0_KernelTiled(
    queryState: Any,
    referenceState: Any,
    domainState: domainData,

    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any,
    
    kernelProperties: kernelState,
    # Do not change the parameters above
    
    # The last parameter is always the output array and should not be changed
    output_psi0 : wp.array(dtype = scalar_t), # type: ignore
    output_psi0_H : wp.array(dtype = scalar_t) # type: ignore
):
    # Multi-lane variant of computePsi0_Kernel (warpSPHCore autograd/lanes.py):
    # launched dim=[N, lanes]; each lane walks a slice of i's neighbours.

    i, lane = wp.tid()
    if useAdjacency:
        # raw sums reduced across lanes, THEN the roots (see computePsi0_Func_i)
        psi0, psi0h = computePsi0_Func_Adjacency_raw(
            i, domainState.dim, lane, wp.block_dim(),
            queryState, referenceState, correctionData, domainState,
            useAdjacency, adjacencyState, gridState, 1,
            kernelProperties,
        )
        s_psi0 = laneSum(psi0)
        s_psi0h = laneSum(psi0h)
        if lane == 0:
            output_psi0[i] = s_psi0**(scalar_t(1.0)/scalar_t(domainState.dim))
            output_psi0_H[i] = s_psi0h**(scalar_t(1.0)/scalar_t(domainState.dim))
    else:
        # grid traversal takes the roots per cell (historical semantics), which
        # does not decompose over lanes: lane 0 runs the full sum (the branch is
        # uniform across the block, so no tile op diverges)
        if lane == 0:
            psi0, psi0h = computePsi0_Func_Adjacency(
                i, domainState.dim, 0, 1,
                queryState, referenceState, correctionData, domainState,
                useAdjacency, adjacencyState, gridState, gridState.numOffsets,
                kernelProperties,
            )
            output_psi0[i] = psi0
            output_psi0_H[i] = psi0h


def _psi0Dtype(ctx, extras):
    return castTorchToWarpAsBuiltins(ctx.query.densities).dtype


_PSI0 = OperatorSpec(
    kernel=computePsi0_Kernel,
    tiledKernel=computePsi0_KernelTiled,
    outputs=(
        OutputSpec(dtype=_psi0Dtype, shape=ShapeOf.QUERY),
        OutputSpec(dtype=_psi0Dtype, shape=ShapeOf.QUERY),
    ),
)


def computePsi0Warp(
    queryParticles: ParticleState,
    operationProperties: OperationProperties,
    domain: DomainDescription,
    
    queryVolumes: Optional[torch.Tensor] = None, referenceVolumes: Optional[torch.Tensor] = None,
    adjacency: Optional[Union[AdjacencyList, CompactHashMap]] = None, # if none a datastructure is created for EVERY operation!,
    referenceParticles: Optional[ParticleState] = None,
    crkState: Optional[CRKState] = None,
    gradHState: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], GradHState]] = None,
    renormalizationState: Optional[Union[torch.Tensor,RenormalizationState]] = None,
):
    with record_function("warpSPH[computePsi0]"):
        ctx = SPHContext(
            query=queryParticles, properties=operationProperties, domain=domain,
            adjacency=adjacency, reference=referenceParticles,
            corrections=Corrections(
                volumes=(queryVolumes, referenceVolumes),
                crk=crkState, gradH=gradHState, renorm=renormalizationState,
            ),
        )
        return launchOperator(_PSI0, ctx)


        # with record_function("warpSPH[CRKVolume] - Preprocessing"):
        #     # Preprocessing and input validation
        #     args, device, dim = parseArguments(
        #         queryParticles, operationProperties, domain,
        #         queryVolumes, referenceVolumes,
        #         adjacency,
        #         referenceParticles,
        #         crkState,
        #         gradHState,
        #         renormalizationState,
        #     )

        #     outputSize = queryParticles.positions.shape[0]
        #     outputDtype = castTorchToWarpAsBuiltins(queryParticles.densities).dtype

        # with record_function("warpSPH[CRKVolume] - Kernel Execution"):
        #     warp_result = warpWrapper(
        #         launch_kernel, computePsi0_Kernel, outputSize, outputDtype,
        #         *args,
        #     )

    return warp_result
