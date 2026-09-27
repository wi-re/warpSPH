"""Warp kernels assembling the moving-least-squares moment system used by the
`liu` interpolation routines.

For each query point, accumulates a `(dim+1)`-vector `[<q>, <grad q>]` and the
corresponding `(dim+1) x (dim+1)` moment matrix (kernel-weighted 0th/1st-order
terms: `[1, x]` interpolation and covariance blocks) over its neighbors, plus
a Shepard normalization sum and a raw neighbor count (particles within one
support radius). `computeLiuMatricesWarp` is the public entry point;
`interpolateLiuLiu` in `interp.py` inverts the resulting matrix to solve for
the local linear fit.
"""

import warp as wp
from warp.types import vector, matrix
from typing import Any
import torch
from torch.profiler import profile, ProfilerActivity
from warpSPHCore.profiling import record_function
from typing import Optional, Union, Tuple
from warpSPHCore import *

__all__ = ['computeLiuMatricesWarp']

@wp.func
def computeLiuMatrices_Func_i(
    # General Shape Parameters and indices
    i : wp.int32,  dim: wp.int32, 

    # SPH properties for the query set (indexed by i)
    xi: vector(dtype = scalar_t, length=Any), # type: ignore

    # SPH properties for the reference set (indexed by j in the neighbor loop)
    referenceState: Any, # particleDataSoA with the exact type based on the dimensionality, e.g., particleDataSoA_2 for 2D, particleDataSoA_3 for 3D, etc.
    correctionData: Any,
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
    referenceKinds : wp.array(dtype = wp.int32), # type: ignore

    # Optional Correction Terms:
    # Gradient renormalization matrices for each query point, used for correcting the kernel gradient based on the local particle distribution.
    useGradientRenormalization: wp.bool, Li: matrix(shape=(Any, Any), dtype=scalar_t), # type: ignore
    # Grad-h correction terms for each query and reference point, used for correcting the kernel gradient based on the local particle distribution and smoothing length variations.
    useGradHTerms: wp.bool, omega_i: scalar_t,   # type: ignore
    # Whether to use actual volume (mass/density) or apparent volume for the gradient computation, and the corresponding volumes if needed.
    useVolume: bool, Vi: scalar_t,  # type: ignore
    # Whether to use CRK kernel correction for the computation, and the corresponding correction terms if needed.
    useCRK: bool, Ai: scalar_t, Bi: vector(length=Any, dtype=scalar_t), gradAi: vector(length=Any, dtype=scalar_t), gradBi: matrix(shape=(Any, Any), dtype=scalar_t), # type: ignore
    
    referenceQuantities: wp.array(dtype = scalar_t), # type: ignore

    vector_out : Any, # type: ignore
    matrix_out: Any, # type: ignore
    shep_out : Any, # type: ignore
):\
    # Initialize the output value
    out_vec = zero_like_warp(vector_out)
    out_mat = zero_like_warp(matrix_out)
    out_nbrs = wp.int32(0)
    out_shep = scalar_t(0.0)

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
        _, Vj = getVolume_j(correctionData, j)

        x_ij = computeDistanceVec(xi, xj, domainState)
        r_ij = safe_sqrt(wp.dot(x_ij, x_ij))

        gradKernel = computeKernelGradientCRK(
            xi, xj, 
            hj, hj,
            kernelProperties, # scatter mode for gradW
            domainState,
            useCRK, Ai, Bi, gradAi, gradBi
        )
        if useGradientRenormalization:
            gradKernel = matmul(Li, gradKernel)

        kernel = computeKernelCRK(
            xi, xj,
            hj, hj,
            kernelProperties, domainState,
            useCRK, Ai, Bi
        )

        V_j = mj / rhoj if not useVolume else Vj

        temp_vec = zero_like_warp(out_vec)
        temp_mat = zero_like_warp(out_mat)

        Qj = referenceQuantities[j]

        scalarTerm = V_j * Qj * kernel
        vectorTerm = V_j * Qj * gradKernel

        temp_vec[0] = scalarTerm #scalar_t(1.0) if r_ij < hj else scalar_t(0.0)
        for d in range(dim):
            temp_vec[d+1] = vectorTerm[d]

        # the output matrix contains several terms
        # [<1>, <x>, <y>, <z>]
        # [\nabla <1>[x], \nabla <x>[x], \nabla <y>[x], \nabla <z>[x]]
        # [\nabla <1>[y], \nabla <x>[y], \nabla <y>[y], \nabla <z>[y]]
        # [\nabla <1>[z], \nabla <x>[z], \nabla <y>[z], \nabla <z>[z]]

        covarianceMat = -V_j * wp.outer(x_ij, gradKernel) # mat(d, d)
        position_interp = -V_j * x_ij * kernel # vec(d)

        one_interp = V_j * kernel # scalar
        one_grad = V_j * gradKernel # vec(d)

        temp_mat[0, 0] = one_interp
        for d in range(dim):
            temp_mat[d+1, 0] = one_grad[d]
            temp_mat[0, d+1] = position_interp[d]
            for dd in range(dim):
                temp_mat[d+1, dd+1] = covarianceMat[d, dd]

        out_vec += temp_vec
        out_mat += temp_mat

        out_shep += Vj * kernel

        if r_ij < hj:
            out_nbrs += 1



        
    return out_shep, out_vec, out_mat, out_nbrs



@wp.func
def computeLiuMatrices_Func_Adjacency(
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
    
    queryPositions: wp.array(dtype = vector(length=Any, dtype=scalar_t)), # type: ignore
    referenceQuantities: wp.array(dtype = scalar_t), # type: ignore

    vector_out : Any, # type: ignore
    matrix_out: Any, # type: ignore
    shep_out : Any, # type: ignore
):
    xi, h_i, m_i, rho_i, k_i = getParticle(queryState, i)
    xi = queryPositions[i]
        
    useGradientRenormalization, Li = getL_i(correctionData, i)
    useGradHTerms, omega_i = getGradH_i(correctionData, i)
    useVolume, Vi = getVolume_i(correctionData, i)
    useCRK, Ai, Bi, gradA_i, gradB_i = getCRK_i(correctionData, i)
    
    sh_out = scalar_t(0.0)
    vec_out = zero_like_warp(vector_out)
    mat_out = zero_like_warp(matrix_out)
    nnbrs = wp.int32(0)

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
                i, queryPositions, gridState.numCells, gridState.D, 
                o, gridState.cellOffsets, gridState.hashTable, gridState.cellTable,
                domainState.periodicity, gridState.qMin, gridState.qMax, gridState.hCell
            )
            if beginIndex < 0:
                continue
        
        beginIndex, numIndices = laneSlice(beginIndex, numIndices, lane, lanes, useAdjacency)
        out_shep, out_vec, out_mat, nbrs = computeLiuMatrices_Func_i(
            i, dim, 
            xi, 
            referenceState,  correctionData, domainState,
            kernelProperties,

            beginIndex, numIndices, adjacencyState.neighborList if useAdjacency else gridState.sortIndex,
            referenceState.kinds,

            useGradientRenormalization, Li,
            useGradHTerms, omega_i,
            useVolume, Vi , 
            useCRK, Ai, Bi, gradA_i, gradB_i,
            
            referenceQuantities,
            vector_out, matrix_out, shep_out
        )
    
        sh_out += out_shep
        vec_out += out_vec
        mat_out += out_mat
        nnbrs += nbrs

    return sh_out, vec_out, mat_out, nnbrs



@wp.kernel
def computeLiuMatrices_Kernel(
    queryState: Any,
    referenceState: Any,
    domainState: domainData,

    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any,
    
    kernelProperties: kernelState,
    # Do not change the parameters above
    queryPositions: wp.array(dtype = vector(length=Any, dtype=scalar_t)), # type: ignore
    referenceQuantities: wp.array(dtype = scalar_t), # type: ignore
    # The last parameter is always the output array and should not be changed

    shep_out : wp.array(dtype = scalar_t), # type: ignore
    vector_out : wp.array(dtype = vector(length=Any, dtype=scalar_t)), # type: ignore
    matrix_out: wp.array(dtype = matrix(shape=(Any, Any), dtype=scalar_t)), # type: ignore
    numNeighbors_out: wp.array(dtype = wp.int32) # type: ignore
):                                                                                    
    i = wp.tid()
    numParticles = queryState.positions.shape[0]
    if i >= numParticles:
        return

    shep, vec, mat, nnbrs = computeLiuMatrices_Func_Adjacency(
        i, domainState.dim, 0, 1, 
        queryState, referenceState, correctionData, domainState,
        useAdjacency, adjacencyState, gridState, gridState.numOffsets if not useAdjacency else 1,
        kernelProperties,  #queryKinds, referenceKinds,
        # The parameters above are default parameters and shold not be changed
        queryPositions, referenceQuantities,
        zero_like_warp(vector_out), zero_like_warp(matrix_out), zero_like_warp(shep_out)
    )

    shep_out[i] = mat[0, 0]
    vector_out[i] = vec
    matrix_out[i] = mat
    numNeighbors_out[i] = nnbrs


from warpSPHCore import *

from copy import deepcopy


@wp.kernel
def computeLiuMatrices_KernelTiled(
    queryState: Any,
    referenceState: Any,
    domainState: domainData,

    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any,
    
    kernelProperties: kernelState,
    # Do not change the parameters above
    queryPositions: wp.array(dtype = vector(length=Any, dtype=scalar_t)), # type: ignore
    referenceQuantities: wp.array(dtype = scalar_t), # type: ignore
    # The last parameter is always the output array and should not be changed

    shep_out : wp.array(dtype = scalar_t), # type: ignore
    vector_out : wp.array(dtype = vector(length=Any, dtype=scalar_t)), # type: ignore
    matrix_out: wp.array(dtype = matrix(shape=(Any, Any), dtype=scalar_t)), # type: ignore
    numNeighbors_out: wp.array(dtype = wp.int32) # type: ignore
):
    # Multi-lane variant of computeLiuMatrices_Kernel (warpSPHCore autograd/lanes.py):
    # launched dim=[N, lanes]; each lane walks a slice of i's neighbours.

    i, lane = wp.tid()
    shep, vec, mat, nnbrs = computeLiuMatrices_Func_Adjacency(
        i, domainState.dim, lane, wp.block_dim(),
        queryState, referenceState, correctionData, domainState,
        useAdjacency, adjacencyState, gridState, gridState.numOffsets if not useAdjacency else 1,
        kernelProperties,
        queryPositions, referenceQuantities,
        zero_like_warp(vector_out), zero_like_warp(matrix_out), zero_like_warp(shep_out)
    )
    vecT = laneSum(vec)
    matT = laneSum(mat)
    nnbrsT = laneSum(nnbrs)
    if lane == 0:
        shep_out[i] = matT[0, 0]
        vector_out[i] = vecT
        matrix_out[i] = matT
        numNeighbors_out[i] = nnbrsT


def _liuOutputShape(ctx, extras):
    return extras["queryPositions"].shape[0]


def _liuScalarDtype(ctx, extras):
    return _torch_scalar_to_warp_dtype(extras["queryPositions"].dtype)


def _liuVecDtype(ctx, extras):
    qp = extras["queryPositions"]
    return _get_warp_vector_dtype(qp.shape[1] + 1, qp.dtype)


def _liuMatDtype(ctx, extras):
    qp = extras["queryPositions"]
    return _get_warp_matrix_dtype(qp.shape[1] + 1, qp.shape[1] + 1, qp.dtype)


_LIU_MATRICES = OperatorSpec(
    kernel=computeLiuMatrices_Kernel,
    tiledKernel=computeLiuMatrices_KernelTiled,
    outputs=(
        OutputSpec(dtype=_liuScalarDtype, shape=_liuOutputShape),
        OutputSpec(dtype=_liuVecDtype, shape=_liuOutputShape),
        OutputSpec(dtype=_liuMatDtype, shape=_liuOutputShape),
        OutputSpec(dtype=wp.int32, shape=_liuOutputShape),
    ),
    extras=(
        ExtraSpec("queryPositions", ExtraKind.TENSOR),
        ExtraSpec("referenceQuantities", ExtraKind.TENSOR),
    ),
)


def computeLiuMatricesWarp(
    queryPositions: torch.Tensor,
    referenceParticles: ParticleState,
    referenceQuantities: torch.Tensor,
    operationProperties: OperationProperties,
    domain: DomainDescription,


    queryVolumes: Optional[torch.Tensor] = None, referenceVolumes: Optional[torch.Tensor] = None,
    adjacency: Optional[Union[AdjacencyList, CompactHashMap]] = None, # if none a datastructure is created for EVERY operation!,
    crkState: Optional[CRKState] = None,
    gradHState: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], GradHState]] = None,
    renormalizationState: Optional[Union[torch.Tensor,RenormalizationState]] = None,
):
    with record_function("warpSPH[computeLiuMatrices]"):
        with record_function("warpSPH[computeLiuMatrices] - Preprocessing"):
            # Preprocessing and input validation
            device = referenceParticles.positions.device
            # args, device, dim = parseArguments(
            #     queryParticles, operationProperties, domain,
            #     queryVolumes, referenceVolumes,
            #     adjacency,
            #     referenceParticles,
            #     crkState,
            #     gradHState,
            #     renormalizationState,
            # )

        with record_function("warpSPH[computeLiuMatrices] - Kernel Execution"):
            ctx = SPHContext(
                query=referenceParticles, properties=operationProperties, domain=domain,
                adjacency=adjacency, reference=referenceParticles,
                corrections=Corrections(
                    volumes=(queryVolumes, referenceVolumes),
                    crk=crkState, gradH=gradHState, renorm=renormalizationState,
                ),
            )
            return launchOperator(
                _LIU_MATRICES, ctx,
                queryPositions=queryPositions,
                referenceQuantities=referenceQuantities,
            )


        # with record_function("warpSPH[CRKVolume] - Kernel Execution"):
        #     warp_result = warpWrapper(
        #         launch_kernel, computeLiuMatrices_Kernel, outputSize, outputDtype,
        #         *args,
        #         queryVelocities_, referenceVelocities_,
        #         queryEnergies_, referenceEnergies_,
        #         individual_cs, queryCs_, referenceCs_,
        #         viscositySwitch, queryAlphas_, referenceAlphas_,
        #         explicitPressure, queryPressures_, referencePressures_,
        #         conductivityParams
        #     )

    return warp_result
