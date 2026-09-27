"""Per-particle neighbor count: `countNeighborsWarp` counts, for each query
particle, the number of reference particles with a nonzero kernel value
(i.e. within kernel support) under the given `OperationProperties`/
correction terms; `countNeighbors` is a convenience wrapper that builds
those properties (`SupportScheme.SuperSymmetric`, all-to-all, naive
gradient mode) from a `SimulationConfig`. The neighbor loop reuses the
generic `WarpOperation.Gradient` kernel-launch plumbing even though no
gradient is actually computed.
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

__all__ = ['countNeighborsWarp', 'countNeighbors']

@wp.func
def countNeighbors_Func_i(
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
    outputValue: Any, # type: ignore

    # Viscosity function parameters
):
    # Initialize the output value
    out     = wp.int32(0)
    
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

        # gradw_ij = computeKernelGradientCRK(
        #     xi, xj, 
        #     hi, hj,
        #     kernelProperties, domainState,
        #     useCRK, Ai, Bi, gradAi, gradBi
        # )
        w_ij = computeKernelCRK(
            xi, xj,
            hi, hj,
            kernelProperties, domainState,
            useCRK, Ai, Bi
        )

        if w_ij > 0.0:
            out += 1

    return out



@wp.func
def countNeighbors_Func_Adjacency(
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
    
    outputValue : Any, # type: ignore
):
    xi, hi, mi, rhoi, ki = getParticle(queryState, i)
    if kernelProperties.operationMode != wp.static(OperationDirection.TrueAllToToAll.value):
        if not checkDirectionality_i(ki, kernelProperties.operationMode):
            return 0
        
    useGradientRenormalization, Li = getL_i(correctionData, i)
    useGradHTerms, omega_i = getGradH_i(correctionData, i)
    useVolume, Vi = getVolume_i(correctionData, i)
    useCRK, Ai, Bi, gradA_i, gradB_i = getCRK_i(correctionData, i)
    
    out = wp.int32(0)
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
        out += countNeighbors_Func_i(
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


            outputValue,

            # Viscosity function parameters
        )
    return out



@wp.kernel
def countNeighbors_Kernel(
    queryState: Any,
    referenceState: Any,
    domainState: domainData,

    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any,
    
    kernelProperties: kernelState,
    # Do not change the parameters above
    # The last parameter is always the output array and should not be changed
    outputValues : wp.array(dtype = wp.int32) # type: ignore
):                                                                                    
    i = wp.tid()
    numParticles = queryState.positions.shape[0]
    if i >= numParticles:
        return

    outputValues[i] = countNeighbors_Func_Adjacency(
        i, domainState.dim, 0, 1, 
        queryState, referenceState, correctionData, domainState,
        useAdjacency, adjacencyState, gridState, gridState.numOffsets if not useAdjacency else 1,
        kernelProperties,  #queryKinds, referenceKinds,
        # The parameters above are default parameters and shold not be changed

        0
    )


@wp.kernel
def countNeighbors_KernelTiled(
    queryState: Any,
    referenceState: Any,
    domainState: domainData,

    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any,
    
    kernelProperties: kernelState,
    # Do not change the parameters above
    # The last parameter is always the output array and should not be changed
    outputValues : wp.array(dtype = wp.int32) # type: ignore
):
    # Multi-lane variant of countNeighbors_Kernel (warpSPHCore autograd/lanes.py):
    # launched dim=[N, lanes]; each lane walks a slice of i's neighbours.

    i, lane = wp.tid()
    count = countNeighbors_Func_Adjacency(
        i, domainState.dim, lane, wp.block_dim(),
        queryState, referenceState, correctionData, domainState,
        useAdjacency, adjacencyState, gridState, gridState.numOffsets if not useAdjacency else 1,
        kernelProperties,
        0
    )
    countT = laneSum(count)
    if lane == 0:
        outputValues[i] = countT


_COUNT_NEIGHBORS = OperatorSpec(
    kernel=countNeighbors_Kernel,
    tiledKernel=countNeighbors_KernelTiled,
    outputs=(OutputSpec(dtype=wp.int32, shape=ShapeOf.QUERY),),
)


def countNeighborsWarp(
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
    with record_function("warpSPH[countNeighbors]"):
        with record_function("warpSPH[countNeighbors] - Preprocessing"):
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

        with record_function("warpSPH[countNeighbors] - Kernel Execution"):
            ctx = SPHContext(
                query=queryParticles, properties=operationProperties, domain=domain,
                adjacency=adjacency, reference=referenceParticles,
                corrections=Corrections(
                    volumes=(queryVolumes, referenceVolumes),
                    crk=crkState, gradH=gradHState, renorm=renormalizationState,
                ),
            )
            return launchOperator(_COUNT_NEIGHBORS, ctx)


        # with record_function("warpSPH[CRKVolume] - Kernel Execution"):
        #     warp_result = warpWrapper(
        #         launch_kernel, countNeighbors_Kernel, outputSize, outputDtype,
        #         *args,
        #         queryVelocities_, referenceVelocities_,
        #         queryEnergies_, referenceEnergies_,
        #         individual_cs, queryCs_, referenceCs_,
        #         viscositySwitch, queryAlphas_, referenceAlphas_,
        #         explicitPressure, queryPressures_, referencePressures_,
        #         conductivityParams
        #     )

    return warp_result


from ...configurations.simulationConfig import SimulationConfig
def countNeighbors(
    queryState: Any,
    config : SimulationConfig, schemeConfig: Any,
    adjacency: Optional[Union[AdjacencyList, CompactHashMap]] = None,
):
    return countNeighborsWarp(
        queryParticles = queryState,
        operationProperties = OperationProperties(
            kernel = config.kernel,
            operation = WarpOperation.Gradient,
            supportMode = SupportScheme.SuperSymmetric,
            operationMode = OperationDirection.AllToAll,
            gradientMode = GradientScheme.Naive
        ),
        domain = config.domain,
        adjacency = adjacency
    )