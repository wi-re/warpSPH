"""Warp kernel implementation of the Maronne surface-detection geometry test.
For each particle, projects a trial point `T = x_i + n_i * h_i / xi` along
its normal (`xi` = `sphKernel_xi`, the kernel's characteristic support
ratio) and, per neighbor within `r_ij < h_j`, checks two conditions: (A) the
neighbor is far (`r_ij >= sqrt(2) * h_j / xi`) and inside the tube around
`T`; (B) the neighbor is near and inside a tangential band around `T`
(tangent built by a 2D perpendicular or 3D cross-product-style rotation of
`n_i`); a neighbor only counts toward A/B if the particle's normal is
non-degenerate (`|n_i| > 0.5`). `computeMaronneSurfaceDetection` (the
torch-facing wrapper) sums the per-neighbor A/B hits and thresholds the
total at `< 0.5` to produce the free-surface flag (so particles with no
qualifying neighbor, including ones with a degenerate normal, come out
flagged as surface).
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

__all__ = ['computeMaronneSurfaceDetection']

# @torch.jit.script
# def detectFreeSurfaceMaronne(
#     particles : WeaklyCompressibleState,
#     normals: torch.Tensor,
#     domain: DomainDescription,
#     xi :float, 
#     neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
#     supportScheme: SupportScheme = SupportScheme.Scatter
#     ):
#     with record_function("[SPH] - [Surface Detection] - Detect Free Surface (Maronne)"):
#         positions = particles.positions
#         n = normals
#         numParticles = positions.shape[0]
#         supports = particles.supports
#         periodicity = domain.periodic
#         domainMin = domain.min
#         domainMax = domain.max
        
#         rij = neighborhood[1].r_ij
#         i, j = neighborhood[0].row, neighborhood[0].col
        
#         T = positions + n * supports.view(-1,1) / xi
        
#         hij = supports[j]
        
#         tau = torch.vstack((-n[:,1], n[:,0])).mT
        
#         xjt = positions[j] - T[i]
#         xjt = torch.stack([xjt[:,i_] if not periodicity[i_] else mod(xjt[:,i_], domainMin[i_], domainMax[i_]) for i_ in range(xjt.shape[1])], dim = -1)
        
#         condA1 = rij >= math.sqrt(2) * hij / xi
#         condA2 = torch.linalg.norm(xjt, dim = -1) <= hij / xi
#         condA = (condA1 & condA2) & (i != j)
#         cA = scatter_sum(condA, i, dim = 0, dim_size = numParticles)
        
#         condB1 = rij < math.sqrt(2) * hij / xi
#         condB2 = torch.abs(torch.einsum('ij,ij->i', -n[i], xjt)) + torch.abs(torch.einsum('ij,ij->i', tau[i], xjt)) < hij / xi
#         condB = (condB1 & condB2) & (i != j)
#         cB = scatter_sum(condB, i, dim = 0, dim_size = numParticles)
        
#         fs = torch.where(~cA & ~cB & (torch.linalg.norm(n, dim = -1) > 0.5), 1.,0.)
#         return fs, cA, cB

@wp.func
def computeMaronneSurfaceDetection_Func_i(
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
    
    normals: wp.array(dtype = vector(length=Any, dtype=scalar_t)), # type: ignore

    # Dummy value to allow allocation
):
    # Initialize the output value
    out     = zero_like_warp(hi)
    ni = normals[i]
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


    
        w_ij = computeKernelCRK(
            xi, xj,
            hi, hj,
            kernelProperties, domainState,
            useCRK, Ai, Bi
        )

        kernelXi = sphKernel_xi(kernelProperties.kernelFunction, dim)

        T = xi + ni * hi / kernelXi
        hij = hj

        tau = ni * scalar_t(0.0)
        if dim == 2:
            tau[0] = -ni[1]
            tau[1] = ni[0]
        elif dim == 3:
            tau[0] = ni[1] * ni[2]
            tau[1] = -ni[0] * ni[2]
            tau[2] = ni[0] * ni[1]

        xij = computeDistanceVec(xi, xj, domainState)
        rij = safe_sqrt(wp.dot(xij, xij))

        if rij >= hij:
            continue

        xjt = computeDistanceVec(xj, T, domainState)
        rjt = safe_sqrt(wp.dot(xjt, xjt))

        condA1 = rij >= safe_sqrt(scalar_t(2.0)) * hij / kernelXi
        condA2 = rjt <= hij / kernelXi
        # condA2 = True
        condA = condA1 and condA2 and (i != j)
        # condA = False

        condB1 = rij < safe_sqrt(scalar_t(2.0)) * hij / kernelXi    
        condB2 = wp.abs(wp.dot(ni, xjt)) + wp.abs(wp.dot(tau, xjt)) < hij / kernelXi
        condB = condB1 and condB2 and (i != j)
        # condB = False

        if (condA or condB) and safe_sqrt(wp.dot(ni, ni)) > scalar_t(0.5):
            
            out += scalar_t(1.0)
        
    return out



@wp.func
def computeMaronneSurfaceDetection_Func_Adjacency(
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
    
    normals: wp.array(dtype = vector(length=Any, dtype=scalar_t)), # type: ignore
):
    xi, hi, mi, rhoi, ki = getParticle(queryState, i)
    if kernelProperties.operationMode != wp.static(OperationDirection.TrueAllToToAll.value):
        if not checkDirectionality_i(ki, kernelProperties.operationMode):
            return zero_like_warp(queryState.supports)
        
    useGradientRenormalization, Li = getL_i(correctionData, i)
    useGradHTerms, omega_i = getGradH_i(correctionData, i)
    useVolume, Vi = getVolume_i(correctionData, i)
    useCRK, Ai, Bi, gradA_i, gradB_i = getCRK_i(correctionData, i)
    
    out = zero_like_warp(queryState.supports)
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
        
        out += computeMaronneSurfaceDetection_Func_i(
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


            normals

            # Viscosity function parameters
        )
    return out



@wp.kernel
def computeMaronneSurfaceDetection_Kernel(
    queryState: Any,
    referenceState: Any,
    domainState: domainData,

    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any,
    
    kernelProperties: kernelState,
    # Do not change the parameters above

    normals: wp.array(dtype = vector(length=Any, dtype=scalar_t)), # type: ignore

    # The last parameter is always the output array and should not be changed
    outputValues : wp.array(dtype = scalar_t) # type: ignore
):                                                                                    
    i = wp.tid()
    numParticles = queryState.positions.shape[0]
    if i >= numParticles:
        return

    outputValues[i] = computeMaronneSurfaceDetection_Func_Adjacency(
        i, domainState.dim, 
        queryState, referenceState, correctionData, domainState,
        useAdjacency, adjacencyState, gridState, gridState.numOffsets if not useAdjacency else 1,
        kernelProperties,  #queryKinds, referenceKinds,
        # The parameters above are default parameters and shold not be changed
        normals
    )


def _maronneDtype(ctx, extras):
    return castTorchToWarpAsBuiltins(ctx.query.densities).dtype


_MARONNE_SURFACE_DETECTION = OperatorSpec(
    kernel=computeMaronneSurfaceDetection_Kernel,
    outputs=(OutputSpec(dtype=_maronneDtype, shape=ShapeOf.QUERY),),
    extras=(ExtraSpec("surfaceNormals", ExtraKind.TENSOR),),
)


def computeMaronneSurfaceDetection(
    queryParticles: ParticleState,
    operationProperties: OperationProperties,
    domain: DomainDescription,

    surfaceNormals: torch.Tensor,

    queryVolumes: Optional[torch.Tensor] = None, referenceVolumes: Optional[torch.Tensor] = None,
    adjacency: Optional[Union[AdjacencyList, CompactHashMap]] = None, # if none a datastructure is created for EVERY operation!,
    referenceParticles: Optional[ParticleState] = None,
    crkState: Optional[CRKState] = None,
    gradHState: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], GradHState]] = None,
    renormalizationState: Optional[Union[torch.Tensor,RenormalizationState]] = None,
):
    with record_function("warpSPH[computeDensityDiffusion]"):
        with record_function("warpSPH[computeDensityDiffusion] - Preprocessing"):
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

        with record_function("warpSPH[computeMaronneSurfaceDetection] - Kernel Execution"):
            ctx = SPHContext(
                query=queryParticles, properties=operationProperties, domain=domain,
                adjacency=adjacency, reference=referenceParticles,
                corrections=Corrections(
                    volumes=(queryVolumes, referenceVolumes),
                    crk=crkState, gradH=gradHState, renorm=renormalizationState,
                ),
            )
            result = launchOperator(_MARONNE_SURFACE_DETECTION, ctx, surfaceNormals=surfaceNormals) < 0.5

            return (result).to(dtype = queryParticles.positions.dtype)


        # with record_function("warpSPH[CRKVolume] - Kernel Execution"):
        #     warp_result = warpWrapper(
        #         launch_kernel, computeMaronneSurfaceDetection_Kernel, outputSize, outputDtype,
        #         *args,
        #         queryVelocities_, referenceVelocities_,
        #         queryEnergies_, referenceEnergies_,
        #         individual_cs, queryCs_, referenceCs_,
        #         viscositySwitch, queryAlphas_, referenceAlphas_,
        #         explicitPressure, queryPressures_, referencePressures_,
        #         conductivityParams
        #     )

    return warp_result
