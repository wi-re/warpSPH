"""Pressure-force operator supporting several pressure-symmetrization
formulations selected via `enumTypes.PressureForceScheme`
(`conservative`, `nonConservative`, an Antuono-style surface-aware blend,
one-sided `i`/`j`, and `symmetric`). The Antuono formulation is the Tensile
Instability Control switch of Sun, Colagrossi, Marrone, Antuono & Zhang,
*Multi-resolution Delta-plus-SPH with tensile instability control*, Comput.
Phys. Commun. 224:63-80 (2018) (`sun2018`), Eq. (9): it switches to the
symmetric form when the *query* particle's own pressure is non-negative, or
the query particle is flagged as a free-surface particle (`querySurfaceMask`),
and falls back to the antisymmetric ("conservative") form only when the query
particle is both in tension and away from the free surface. The condition is
on the query particle alone -- the reference particle's pressure does not
participate in the switch, per the equation. Output is the negated
pressure-gradient term divided by density.
"""

import warp as wp
from warp.types import vector, matrix
from typing import Any
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from typing import Optional, Union, Tuple
from warpSPHCore import *


from ...enumTypes import PressureForceScheme

__all__ = ['computePressureSurfaceAwareWarp']

@wp.func
def computePressureSurfaceAware_Func_i(
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
    useGradHTerms: wp.bool, PressureSurfaceAware_i: scalar_t, referencePressureSurfaceAwares: wp.array(dtype = scalar_t),  # type: ignore
    # Whether to use actual volume (mass/density) or apparent volume for the gradient computation, and the corresponding volumes if needed.
    useVolume: bool, Vi: scalar_t, referenceVolumes: wp.array(dtype = scalar_t), # type: ignore
    # Whether to use CRK kernel correction for the computation, and the corresponding correction terms if needed.
    useCRK: bool, Ai: scalar_t, Bi: vector(length=Any, dtype=scalar_t), gradAi: vector(length=Any, dtype=scalar_t), gradBi: matrix(shape=(Any, Any), dtype=scalar_t), # type: ignore
    
    P_i: scalar_t, referencePressures: wp.array(dtype = scalar_t), # type: ignore
    mask_i: wp.int32, referenceSurfaceMask: wp.array(dtype = wp.int32), # type: ignore
    pressureTerm: wp.int32, # type: ignore

    # Dummy value to allow allocation
    outputValue: Any, # type: ignore
):
    # Initialize the output value
    out     = zero_like_warp(outputValue)
    
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
        P_j = access_optional(referencePressures, j, referencePressures.shape[0] > 1, referencePressures[0])
        mask_j = access_optional(referenceSurfaceMask, j, referenceSurfaceMask.shape[0] > 1, 0)

        apparentVolume = mj / rhoj if not useVolume else referenceVolumes[j]
        
        gradw_ij = computeKernelGradientCRK(
            xi, xj, 
            hi, hj,
            kernelProperties, domainState,
            useCRK, Ai, Bi, gradAi, gradBi
        )
        if useGradientRenormalization:
            gradw_ij = matmul(Li, gradw_ij)

        p_ij = scalar_t(0.0)
        if pressureTerm == wp.static(wp.int32(PressureForceScheme.conservative.value)):
            p_ij = P_j - P_i
        elif pressureTerm == wp.static(wp.int32(PressureForceScheme.nonConservative.value)):
            p_ij = P_j + P_i
        elif pressureTerm == wp.static(wp.int32(PressureForceScheme.Antuono.value)):
            sw = P_i >= 0.0 or (mask_i == 1)
            p_ij = (P_j + P_i) if sw else (P_j - P_i)
        elif pressureTerm == wp.static(wp.int32(PressureForceScheme.i.value)):
            p_ij = P_i
        elif pressureTerm == wp.static(wp.int32(PressureForceScheme.j.value)):
            p_ij = P_j
        elif pressureTerm == wp.static(wp.int32(PressureForceScheme.symmetric.value)):
            p_ij = (P_j / (rhoj * rhoj) + P_i / (rhoi * rhoi)) * rhoj
        

        out += apparentVolume * p_ij * gradw_ij
        
    return out



@wp.func
def computePressureSurfaceAware_Func_Adjacency(
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
    
    queryPressures: wp.array(dtype = scalar_t), referencePressures: wp.array(dtype = scalar_t), # type: ignore
    querySurfaceMask: wp.array(dtype = wp.int32), referenceSurfaceMask: wp.array(dtype = wp.int32), # type: ignore
    pressureTerm: wp.int32, # type: ignore

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
    
    P_i = access_optional(queryPressures, i, queryPressures.shape[0] > 1, queryPressures[0])
    mask_i = access_optional(querySurfaceMask, i, querySurfaceMask.shape[0] > 1,0)

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
        
        out += computePressureSurfaceAware_Func_i(
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
            
            P_i, referencePressures,
            mask_i, referenceSurfaceMask,
            pressureTerm,


            outputValue,

            # PressureSurfaceAware function parameters
        )
    return out



@wp.kernel
def computePressureSurfaceAware_Kernel(
    queryState: Any,
    referenceState: Any,
    domainState: domainData,

    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any,
    
    kernelProperties: kernelState,
    # Do not change the parameters above
    queryPressures: wp.array(dtype = scalar_t), referencePressures: wp.array(dtype = scalar_t), # type: ignore
    querySurfaceMask: wp.array(dtype = wp.int32), referenceSurfaceMask: wp.array(dtype = wp.int32), # type: ignore
    pressureTerm: wp.int32, # type: ignore
    # The last parameter is always the output array and should not be changed
    outputValues : wp.array(dtype = vector(length=Any, dtype=scalar_t)) # type: ignore
):                                                                                    
    i = wp.tid()
    numParticles = queryState.positions.shape[0]
    if i >= numParticles:
        return

    outputValues[i] = -computePressureSurfaceAware_Func_Adjacency(
        i, domainState.dim, 
        queryState, referenceState, correctionData, domainState,
        useAdjacency, adjacencyState, gridState, gridState.numOffsets if not useAdjacency else 1,
        kernelProperties,  #queryKinds, referenceKinds,
        # The parameters above are default parameters and shold not be changed
        queryPressures, referencePressures,
        querySurfaceMask, referenceSurfaceMask,
        pressureTerm,

        zero_like_warp(outputValues)
    ) / queryState.densities[i]


def _surfaceAwareDtype(ctx, extras):
    return castTorchToWarpAsBuiltins(ctx.query.positions).dtype


_PRESSURE_SURFACE_AWARE = OperatorSpec(
    kernel=computePressureSurfaceAware_Kernel,
    outputs=(OutputSpec(dtype=_surfaceAwareDtype, shape=ShapeOf.QUERY),),
    extras=(
        ExtraSpec("queryPressures", ExtraKind.TENSOR),
        ExtraSpec("referencePressures", ExtraKind.TENSOR),
        ExtraSpec("querySurfaceMask", ExtraKind.TENSOR),
        ExtraSpec("referenceSurfaceMask", ExtraKind.TENSOR),
        ExtraSpec("pressureTerm", ExtraKind.SCALAR),
    ),
)


def computePressureSurfaceAwareWarp(
    queryParticles: ParticleState,
    operationProperties: OperationProperties,
    domain: DomainDescription,
    
    pressureTerm: PressureForceScheme = PressureForceScheme.conservative,
    queryPressures: Optional[torch.Tensor] = None, referencePressures: Optional[torch.Tensor] = None,
    querySurfaceMask: Optional[torch.Tensor] = None, referenceSurfaceMask: Optional[torch.Tensor] = None,
    
    queryVolumes: Optional[torch.Tensor] = None, referenceVolumes: Optional[torch.Tensor] = None,
    adjacency: Optional[Union[AdjacencyList, CompactHashMap]] = None, # if none a datastructure is created for EVERY operation!,
    referenceParticles: Optional[ParticleState] = None,
    crkState: Optional[CRKState] = None,
    gradHState: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], GradHState]] = None,
    renormalizationState: Optional[Union[torch.Tensor,RenormalizationState]] = None,
):
    if referencePressures is None:
        referencePressures = queryPressures
    if referenceSurfaceMask is None:
        referenceSurfaceMask = querySurfaceMask
    with record_function("warpSPH[computePressureSurfaceAware]"):
        with record_function("warpSPH[computePressureSurfaceAware] - Preprocessing"):
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
            querySurfaceMask_ = querySurfaceMask if querySurfaceMask is not None else getCachedDummyTensor((1,), dtype=torch.int32, device=device)
            referenceSurfaceMask_ = referenceSurfaceMask if referenceSurfaceMask is not None else getCachedDummyTensor((1,), dtype=torch.int32, device=device)

        with record_function("warpSPH[computePressureSurfaceAware] - Kernel Execution"):
            ctx = SPHContext(
                query=queryParticles, properties=operationProperties, domain=domain,
                adjacency=adjacency, reference=referenceParticles,
                corrections=Corrections(
                    volumes=(queryVolumes, referenceVolumes),
                    crk=crkState, gradH=gradHState, renorm=renormalizationState,
                ),
            )
            return launchOperator(
                _PRESSURE_SURFACE_AWARE, ctx,
                queryPressures=queryPressures, referencePressures=referencePressures,
                querySurfaceMask=querySurfaceMask_, referenceSurfaceMask=referenceSurfaceMask_,
                pressureTerm=wp.int32(pressureTerm.value),
            )

        # with record_function("warpSPH[CRKVolume] - Kernel Execution"):
        #     warp_result = warpWrapper(
        #         launch_kernel, computePressureSurfaceAware_Kernel, outputSize, outputDtype,
        #         *args,
        #         queryVelocities_, referenceVelocities_,
        #         individual_cs, queryCs_, referenceCs_,
        #         viscositySwitch, queryAlphas_, referenceAlphas_,
        #         explicitPressure, queryPressures_, referencePressures_,
        #         viscosityParams
        #     )

    return warp_result
