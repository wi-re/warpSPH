"""IISPH diagonal coefficient `a_ii` (per-particle pressure-Laplacian
normalization) used by both incompressible solvers to turn a divergence/
density residual into a pressure update.

`computeAlpha` returns the negated per-particle sum
`areaI/mi * |sum_j V_j gradW_ij|^2 + areaI * sum_j (V_j^2/mj) |gradW_ij|^2`
(apparent-area-weighted, gather support mode), matching the IISPH `a_ii`
term; callers divide the pressure residual by this value each Jacobi
iteration. `includeBoundaryReaction=False` restricts the *second* sum to
`kind == 0` neighbours, which is what the published formulations do (a static
boundary particle takes no reaction force, so `dp_i/dx_j` does not exist for
it) -- see `BoundaryOperatorTerms`; the first sum always runs over everything.
Contains an unused, commented-out Barecasco free-surface detector left over
from an earlier version of this file.
"""

import warp as wp
from warp.types import vector, matrix
from typing import Any
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from typing import Optional, Union, Tuple
from warpSPHCore import *



from warpSPH.enumTypes import *
from warpSPH.configurations.simulationConfig import SimulationConfig

__all__ = ['computeAlpha']

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
def computeAlpha_Func_i_first(
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
    
    # Viscosity function parameters
    areaI: scalar_t, referenceApparentAreas: wp.array(dtype = scalar_t), # type: ignore
    # Whether a static (kind != 0) neighbour contributes to the second sum,
    # i.e. to its own reaction against `i`'s pressure -- see
    # `BoundaryOperatorTerms`. 1 = historical AllToAll, 0 = fluid neighbours only.
    # boundaryReactionTerm: wp.int32, # type: ignore
    # The output value is returned directly from the function, and should

    # Dummy value to allow allocation
):
    # Initialize the output value
    sumA = zero_like_warp(xi)
    sumB = zero_like_warp(rhoi)
    
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

        # apparentVolume = mj / rhoj if not useVolume else referenceVolumes[j]

        gradw_ij = computeKernelGradientCRK(
            xi, xj, 
            hi, hj,
            kernelProperties, domainState,
            useCRK, Ai, Bi, gradAi, gradBi
        )
        if useGradientRenormalization:
            gradw_ij = matmul(Li, gradw_ij)

        volumeJ =  referenceApparentAreas[j]    

        gradw_ij2 = wp.dot(gradw_ij, gradw_ij)

        term1 = volumeJ * gradw_ij
        term2 = volumeJ * volumeJ / mj * gradw_ij2

        sumA += term1
        # `j` static (`kind != 0`) means `F^p_{j<-i} = 0`: it never accelerates
        # under `i`'s pressure, so its `dp_i/dx_j` term does not exist ([BK]
        # 3.2). The first sum (`dp_i/dx_i`) keeps it either way.
        if kj == 0:
            sumB += term2
            
    # alpha = areaI / mi * wp.dot(sumA, sumA) + areaI * sumB
    # alpha = areaI * sumB

    return sumA, sumB



@wp.func
def computeAlpha_Func_Adjacency_first(
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

    queryApparentAreas: wp.array(dtype = scalar_t), referenceApparentAreas: wp.array(dtype = scalar_t), # type: ignore
    # boundaryReactionTerm: wp.int32, # type: ignore
):
    xi, hi, mi, rhoi, ki = getParticle(queryState, i)
    if kernelProperties.operationMode != wp.static(OperationDirection.TrueAllToToAll.value):
        if not checkDirectionality_i(ki, kernelProperties.operationMode):
            return zero_like_warp(queryState.densities)
        
    useGradientRenormalization, Li = getL_i(correctionData, i)
    useGradHTerms, omega_i = getGradH_i(correctionData, i)
    useVolume, Vi = getVolume_i(correctionData, i)
    useCRK, Ai, Bi, gradA_i, gradB_i = getCRK_i(correctionData, i)

    areaI = queryApparentAreas[i]
    
    out = zero_like_warp(queryState.densities)
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
        
        sumA, sumB = computeAlpha_Func_i_first(
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
            areaI, referenceApparentAreas,
            # boundaryReactionTerm
        )
        alpha = areaI / mi * wp.dot(sumA, sumA) + areaI * sumB
        out += alpha
    return out


@wp.kernel
def computeAlpha_Kernel(
    queryState: Any,
    referenceState: Any,
    domainState: domainData,

    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any,
    
    kernelProperties: kernelState,
    # Do not change the parameters above
    queryApparentAreas: wp.array(dtype = scalar_t), referenceApparentAreas: wp.array(dtype = scalar_t), # type: ignore
    # boundaryReactionTerm: wp.int32, # type: ignore

    # The last parameter is always the output array and should not be changed
    outputValues: wp.array(dtype = scalar_t) # type: ignore
):                                                                                    
    i = wp.tid()
    numParticles = queryState.positions.shape[0]
    if i >= numParticles:
        return



    alpha = computeAlpha_Func_Adjacency_first(
        i, domainState.dim, 
        queryState, referenceState, correctionData, domainState,
        useAdjacency, adjacencyState, gridState, gridState.numOffsets if not useAdjacency else 1,
        kernelProperties,  #queryKinds, referenceKinds,
        # The parameters above are default parameters and shold not be changed
        queryApparentAreas, referenceApparentAreas,
        # boundaryReactionTerm,
    )

    outputValues[i] = alpha



def _alphaDtype(ctx, extras):
    return castTorchToWarpAsBuiltins(ctx.query.densities).dtype


_ALPHA = OperatorSpec(
    kernel=computeAlpha_Kernel,
    outputs=(OutputSpec(dtype=_alphaDtype, shape=ShapeOf.QUERY),),
    extras=(
        ExtraSpec("queryApparentAreas", ExtraKind.TENSOR),
        ExtraSpec("referenceApparentAreas", ExtraKind.TENSOR),
        # ExtraSpec("boundaryReactionTerm", ExtraKind.SCALAR),
    ),
)


def computeAlphaWarp(
    queryParticles: ParticleState,
    operationProperties: OperationProperties,
    domain: DomainDescription,

    queryApparentAreas: torch.Tensor, referenceApparentAreas: Optional[torch.Tensor] = None,
    includeBoundaryReaction: bool = True,

    queryVolumes: Optional[torch.Tensor] = None, referenceVolumes: Optional[torch.Tensor] = None,
    adjacency: Optional[Union[AdjacencyList, CompactHashMap]] = None, # if none a datastructure is created for EVERY operation!,
    referenceParticles: Optional[ParticleState] = None,
    crkState: Optional[CRKState] = None,
    gradHState: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], GradHState]] = None,
    renormalizationState: Optional[Union[torch.Tensor,RenormalizationState]] = None,
):
    with record_function("warpSPH[computeAlpha]"):
        with record_function("warpSPH[computeAlpha] - Preprocessing"):
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
            referenceApparentAreas = referenceApparentAreas if referenceApparentAreas is not None else queryApparentAreas

        with record_function("warpSPH[computeAlpha] - Kernel Execution"):
            ctx = SPHContext(
                query=queryParticles, properties=operationProperties, domain=domain,
                adjacency=adjacency, reference=referenceParticles,
                corrections=Corrections(
                    volumes=(queryVolumes, referenceVolumes),
                    crk=crkState, gradH=gradHState, renorm=renormalizationState,
                ),
            )
            return launchOperator(
                _ALPHA, ctx,
                queryApparentAreas=queryApparentAreas, referenceApparentAreas=referenceApparentAreas,
                # boundaryReactionTerm=wp.int32(1 if includeBoundaryReaction else 0),
            )


        # with record_function("warpSPH[CRKVolume] - Kernel Execution"):
        #     warp_result = warpWrapper(
        #         launch_kernel, computeAlpha_Kernel, outputSize, outputDtype,
        #         *args,
        #         queryVelocities_, referenceVelocities_,
        #         queryEnergies_, referenceEnergies_,
        #         individual_cs, queryCs_, referenceCs_,
        #         viscositySwitch, queryAlphas_, referenceAlphas_,
        #         explicitPressure, queryPressures_, referencePressures_,
        #         conductivityParams
        #     )

    # return warp_result


def computeAlpha(currentState: Any, config: SimulationConfig, schemeConfig: Any, adjacency: Optional[Union[AdjacencyList, CompactHashMap]], apparentVolumes: torch.Tensor, includeBoundaryReaction: bool = True) -> torch.Tensor:
    with record_function("[warpSPH] - computeDensities"):
        dt = config.dt
        alpha = computeAlphaWarp(
            queryParticles=currentState,
            operationProperties=OperationProperties(
                kernel = config.kernel,
                operation = WarpOperation.Density,
                supportMode = SupportScheme.Gather, # cullen switch E.1 in the CRK paper uses gather for density estimation
            ),
            domain=config.domain,
            adjacency=adjacency,

            queryApparentAreas=apparentVolumes,
            # includeBoundaryReaction=includeBoundaryReaction,
        )
        # return alpha
        return - alpha