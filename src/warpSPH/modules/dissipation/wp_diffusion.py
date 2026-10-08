"""SPH momentum-viscosity acceleration: builds the artificial-viscosity
Pi_ij term via `dissipation.pi.computePi_pair` (`thermalConductivity=False`) on the pair velocity of
`viscosityParams.velocityPairPolicy` (raw, or reconstructed -- `modules/reconstruction`)
and combines it with the kernel gradient to form the per-particle viscous
acceleration contribution, using the same query/reference neighbor loop and
gradient-renormalization/CRK/grad-h correction pattern as the other
`modules/` operators.
"""

import warp as wp
from warp.types import vector, matrix
from typing import Any
import torch
from torch.profiler import profile, ProfilerActivity
from warpSPHCore.profiling import record_function
from typing import Optional, Union, Tuple
from warpSPHCore import *

from .pi import computePi_pair
from ..reconstruction import rawPairVelocity, reconstructPairVelocity, velocityTensorArguments
from ...configurations.moduleConfigurations.diffusionParameters import DiffusionParameters, VelocityPairPolicy

__all__ = ['computeViscosityWarp']

@wp.func
def computeViscosity_Func_i(
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
    useGradHTerms: wp.bool, Viscosity_i: scalar_t, referenceViscositys: wp.array(dtype = scalar_t),  # type: ignore
    # Whether to use actual volume (mass/density) or apparent volume for the gradient computation, and the corresponding volumes if needed.
    useVolume: bool, Vi: scalar_t, referenceVolumes: wp.array(dtype = scalar_t), # type: ignore
    # Whether to use CRK kernel correction for the computation, and the corresponding correction terms if needed.
    useCRK: bool, Ai: scalar_t, Bi: vector(length=Any, dtype=scalar_t), gradAi: vector(length=Any, dtype=scalar_t), gradBi: matrix(shape=(Any, Any), dtype=scalar_t), # type: ignore
    
    vel_i: vector(length=Any, dtype=scalar_t), referenceVelocities: wp.array(dtype = vector(length=Any, dtype=scalar_t)), # type: ignore

    individual_cs: wp.bool, cs_i: scalar_t, referenceCs: wp.array(dtype = scalar_t), # type: ignore
    viscositySwitch: wp.bool, alpha_i: scalar_t, referenceAlphas: wp.array(dtype = scalar_t), # type: ignore
    explicitPressure: wp.bool, P_i: scalar_t, referencePressures: wp.array(dtype = scalar_t), # type: ignore
    viscosityParams: DiffusionParameters,
    # velocity Jacobians for a reconstructing `velocityPairPolicy` and the Balsara factors for `BalsaraLimited` /
    # `balsaraPairLimiter` (one-element placeholders when not read)
    J_i: matrix(shape=(Any, Any), dtype=scalar_t), referenceVelocityTensor: wp.array(dtype = matrix(shape=(Any, Any), dtype=scalar_t)), # type: ignore
    B_i: scalar_t, referenceBalsara: wp.array(dtype = scalar_t), # type: ignore

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
        vel_j = referenceVelocities[j]

        apparentVolume = access_optional(referenceVolumes,j, useVolume, mj / rhoj)

        x_ij = computeDistanceVec(xi, xj, domainState)
        r_ij = safe_sqrt(wp.dot(x_ij, x_ij))
        # the pair velocity the viscosity sees (AV_PLAN Phase 3): raw, or reconstructed to the pair midpoint; both
        # the Pi term and the mu factor below take it
        B_j = scalar_t(0.0)
        if viscosityParams.velocityPairPolicy == wp.static(VelocityPairPolicy.BalsaraLimited.value) or viscosityParams.balsaraPairLimiter:
            B_j = referenceBalsara[j]
        u_ij = rawPairVelocity(vel_i, vel_j)
        if viscosityParams.velocityPairPolicy != wp.static(VelocityPairPolicy.Raw.value):
            u_ij = reconstructPairVelocity(
                viscosityParams.velocityPairPolicy, vel_i, vel_j, J_i, referenceVelocityTensor[j], x_ij, hi, hj,
                kernelProperties.kernelFunction, dim,
                viscosityParams.reconstructionEtaCrit, viscosityParams.reconstructionEtaFold,
                B_i, B_j, viscosityParams.reconstructionBalsaraPower, viscosityParams.limiterType)

        pi = computePi_pair(
            xi, xj, 
            hi, hj,
            mi, mj,
            rhoi, rhoj,
            explicitPressure, P_i, access_optional(referencePressures, j, explicitPressure, scalar_t(0.0)),
            u_ij,
            domainState,
            kernelProperties.kernelFunction,
            cs_i, access_optional(referenceCs, j, individual_cs, viscosityParams.c_s),
            alpha_i, access_optional(referenceAlphas, j, viscositySwitch, scalar_t(1.0)),
            viscosityParams, 
            False)
        
        gradw_ij = computeKernelGradientCRK(
            xi, xj, 
            hi, hj,
            kernelProperties, domainState,
            useCRK, Ai, Bi, gradAi, gradBi
        )
        if useGradientRenormalization:
            gradw_ij = matmul(Li, gradw_ij)
        # Balsara (1995) as a pair limiter (Garcia-Senz & Cabezon 2026 Eq. 7; Sphenix's alpha_ij = alpha_bar Bbar)
        if viscosityParams.balsaraPairLimiter:
            pi = pi * scalar_t(0.5) * (B_i + B_j)

        
        ux_ij = wp.dot(u_ij, x_ij)
        mu_ij = ux_ij /(r_ij + scalar_t(1.0e-14) * hi)

        out += apparentVolume * pi * gradw_ij * mu_ij
        
    return out



@wp.func
def computeViscosity_Func_Adjacency(
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
    viscositySwitch: wp.bool, queryAlphas: wp.array(dtype = scalar_t), referenceAlphas: wp.array(dtype = scalar_t), # type: ignore
    explicitPressure: wp.bool, queryPressures: wp.array(dtype = scalar_t), referencePressures: wp.array(dtype = scalar_t), # type: ignore
    viscosityParams: DiffusionParameters,
    queryVelocityTensor: wp.array(dtype = matrix(shape=(Any, Any), dtype=scalar_t)), referenceVelocityTensor: wp.array(dtype = matrix(shape=(Any, Any), dtype=scalar_t)), # type: ignore
    queryBalsara: wp.array(dtype = scalar_t), referenceBalsara: wp.array(dtype = scalar_t), # type: ignore
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
    vel_i = queryVelocities[i]

    cs_i = access_optional(queryCs, i, individual_cs, viscosityParams.c_s)
    alpha_i = access_optional(queryAlphas, i, viscositySwitch, scalar_t(1.0))
    P_i = access_optional(queryPressures, i, explicitPressure, scalar_t(0.0))
    J_i = queryVelocityTensor[0]
    if viscosityParams.velocityPairPolicy != wp.static(VelocityPairPolicy.Raw.value):
        J_i = queryVelocityTensor[i]
    B_i = queryBalsara[0]
    if viscosityParams.velocityPairPolicy == wp.static(VelocityPairPolicy.BalsaraLimited.value) or viscosityParams.balsaraPairLimiter:
        B_i = queryBalsara[i]

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
        
        out += computeViscosity_Func_i(
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
            vel_i, referenceVelocities,
            individual_cs, cs_i, referenceCs,
            viscositySwitch, alpha_i, referenceAlphas,
            explicitPressure, P_i, referencePressures,
            viscosityParams,
            J_i, referenceVelocityTensor,
            B_i, referenceBalsara,

            outputValue,

            # Viscosity function parameters
        )
    return out



@wp.kernel
def computeViscosity_Kernel(
    queryState: Any,
    referenceState: Any,
    domainState: domainData,

    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any,
    
    kernelProperties: kernelState,
    # Do not change the parameters above
    queryVelocities: wp.array(dtype = vector(length=Any, dtype=scalar_t)), referenceVelocities: wp.array(dtype = vector(length=Any, dtype=scalar_t)), # type: ignore
    individual_cs: wp.bool, queryCs: wp.array(dtype = scalar_t), referenceCs: wp.array(dtype = scalar_t), # type: ignore
    viscositySwitch: wp.bool, queryAlphas: wp.array(dtype = scalar_t), referenceAlphas: wp.array(dtype = scalar_t), # type: ignore
    explicitPressure: wp.bool, queryPressures: wp.array(dtype = scalar_t), referencePressures: wp.array(dtype = scalar_t), # type: ignore
    viscosityParams: DiffusionParameters,
    queryVelocityTensor: wp.array(dtype = matrix(shape=(Any, Any), dtype=scalar_t)), referenceVelocityTensor: wp.array(dtype = matrix(shape=(Any, Any), dtype=scalar_t)), # type: ignore
    queryBalsara: wp.array(dtype = scalar_t), referenceBalsara: wp.array(dtype = scalar_t), # type: ignore
    # The last parameter is always the output array and should not be changed
    outputValues : wp.array(dtype = vector(length=Any, dtype=scalar_t)) # type: ignore
):                                                                                    
    i = wp.tid()
    numParticles = queryState.positions.shape[0]
    if i >= numParticles:
        return

    outputValues[i] = computeViscosity_Func_Adjacency(
        i, domainState.dim, 
        queryState, referenceState, correctionData, domainState,
        useAdjacency, adjacencyState, gridState, gridState.numOffsets if not useAdjacency else 1,
        kernelProperties,  #queryKinds, referenceKinds,
        # The parameters above are default parameters and shold not be changed
        queryVelocities, referenceVelocities,
        individual_cs, queryCs, referenceCs,
        viscositySwitch, queryAlphas, referenceAlphas,
        explicitPressure, queryPressures, referencePressures,
        viscosityParams,
        queryVelocityTensor, referenceVelocityTensor,
        queryBalsara, referenceBalsara,


        zero_like_warp(outputValues)
    )

def _viscosityDtype(ctx, extras):
    return castTorchToWarpAsBuiltins(ctx.query.positions).dtype


_VISCOSITY = OperatorSpec(
    kernel=computeViscosity_Kernel,
    outputs=(OutputSpec(dtype=_viscosityDtype, shape=ShapeOf.QUERY),),
    extras=(
        ExtraSpec("queryVelocities", ExtraKind.TENSOR),
        ExtraSpec("referenceVelocities", ExtraKind.TENSOR),
        ExtraSpec("individualCs", ExtraKind.SCALAR),
        ExtraSpec("queryCs", ExtraKind.TENSOR),
        ExtraSpec("referenceCs", ExtraKind.TENSOR),
        ExtraSpec("viscositySwitch", ExtraKind.SCALAR),
        ExtraSpec("queryAlphas", ExtraKind.TENSOR),
        ExtraSpec("referenceAlphas", ExtraKind.TENSOR),
        ExtraSpec("explicitPressure", ExtraKind.SCALAR),
        ExtraSpec("queryPressures", ExtraKind.TENSOR),
        ExtraSpec("referencePressures", ExtraKind.TENSOR),
        ExtraSpec("viscosityParams", ExtraKind.SCALAR),
        ExtraSpec("queryVelocityTensor", ExtraKind.TENSOR),
        ExtraSpec("referenceVelocityTensor", ExtraKind.TENSOR),
        ExtraSpec("queryBalsara", ExtraKind.TENSOR),
        ExtraSpec("referenceBalsara", ExtraKind.TENSOR),
    ),
)


def computeViscosityWarp(
    queryParticles: ParticleState,
    operationProperties: OperationProperties,
    domain: DomainDescription,
    
    viscosityParams: DiffusionParameters,
    queryVelocities : Optional[torch.Tensor] = None, referenceVelocities: Optional[torch.Tensor] = None,
    queryPressures: Optional[torch.Tensor] = None, referencePressures: Optional[torch.Tensor] = None,
    queryCs: Optional[torch.Tensor] = None, referenceCs: Optional[torch.Tensor] = None,
    queryAlphas: Optional[torch.Tensor] = None, referenceAlphas: Optional[torch.Tensor] = None,
    
    queryVolumes: Optional[torch.Tensor] = None, referenceVolumes: Optional[torch.Tensor] = None,
    adjacency: Optional[Union[AdjacencyList, CompactHashMap]] = None, # if none a datastructure is created for EVERY operation!,
    referenceParticles: Optional[ParticleState] = None,
    crkState: Optional[CRKState] = None,
    gradHState: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], GradHState]] = None,
    renormalizationState: Optional[Union[torch.Tensor,RenormalizationState]] = None,
    queryVelocityTensor: Optional[torch.Tensor] = None, referenceVelocityTensor: Optional[torch.Tensor] = None,
    queryBalsara: Optional[torch.Tensor] = None, referenceBalsara: Optional[torch.Tensor] = None,
):
    """`queryVelocityTensor` / `referenceVelocityTensor`: the velocity Jacobians (`J @ dx` = velocity change,
    `reconstruction.computeVelocityJacobian`), required when `viscosityParams.velocityPairPolicy` is not Raw."""
    if referenceVelocities is None:
        referenceVelocities = queryVelocities
    if referenceCs is None:
        referenceCs = queryCs
    if referenceAlphas is None:
        referenceAlphas = queryAlphas
    if referencePressures is None:
        referencePressures = queryPressures
    with record_function("warpSPH[computeViscosity]"):
        with record_function("warpSPH[computeViscosity] - Preprocessing"):
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
            queryAlphas_ = queryAlphas if queryAlphas is not None else (queryParticles.alphas if hasattr(queryParticles, 'alphas') else getCachedDummyTensor((1,), dtype=get_torch_precision(), device=device))
            queryPressures_ = queryPressures if queryPressures is not None else (queryParticles.pressures if hasattr(queryParticles, 'pressures') else getCachedDummyTensor((1,), dtype=get_torch_precision(), device=device))

            referenceVelocities_ = referenceVelocities if referenceVelocities is not None else (referenceParticles.velocities if hasattr(referenceParticles, 'velocities') else None)
            referenceCs_ = referenceCs if referenceCs is not None else (referenceParticles.soundspeeds if hasattr(referenceParticles, 'soundspeeds') else getCachedDummyTensor((1,), dtype=get_torch_precision(), device=device))
            referenceAlphas_ = referenceAlphas if referenceAlphas is not None else (referenceParticles.alphas if hasattr(referenceParticles, 'alphas') else getCachedDummyTensor((1,), dtype=get_torch_precision(), device=device))
            referencePressures_ = referencePressures if referencePressures is not None else (referenceParticles.pressures if hasattr(referenceParticles, 'pressures') else getCachedDummyTensor((1,), dtype=get_torch_precision(), device=device))

            if queryAlphas is not None or (hasattr(queryParticles, 'alphas') and queryParticles.alphas is not None):
                viscositySwitch = True
            else:
                viscositySwitch = False
            if queryCs is not None or (hasattr(queryParticles, 'soundspeeds') and queryParticles.soundspeeds is not None):
                individual_cs = True
            else:            
                individual_cs = False
            if queryPressures is not None or (hasattr(queryParticles, 'pressures') and queryParticles.pressures is not None):
                explicitPressure = True
            else:
                explicitPressure = False

            if queryVelocities_ is None:
                raise ValueError("Velocities must be provided either through queryVelocities or as a property of queryParticles.")
            queryVelocityTensor_, referenceVelocityTensor_, queryBalsara_, referenceBalsara_ = velocityTensorArguments(
                viscosityParams, queryVelocityTensor, referenceVelocityTensor, queryParticles.positions,
                queryBalsara, referenceBalsara)

        with record_function("warpSPH[computeViscosity] - Kernel Execution"):
            ctx = SPHContext(
                query=queryParticles, properties=operationProperties, domain=domain,
                adjacency=adjacency, reference=referenceParticles,
                corrections=Corrections(
                    volumes=(queryVolumes, referenceVolumes),
                    crk=crkState, gradH=gradHState, renorm=renormalizationState,
                ),
            )
            return launchOperator(
                _VISCOSITY, ctx,
                queryVelocities=queryVelocities_, referenceVelocities=referenceVelocities_,
                individualCs=individual_cs, queryCs=queryCs_, referenceCs=referenceCs_,
                viscositySwitch=viscositySwitch, queryAlphas=queryAlphas_, referenceAlphas=referenceAlphas_,
                explicitPressure=explicitPressure, queryPressures=queryPressures_, referencePressures=referencePressures_,
                viscosityParams=viscosityParams,
                queryVelocityTensor=queryVelocityTensor_, referenceVelocityTensor=referenceVelocityTensor_,
                queryBalsara=queryBalsara_, referenceBalsara=referenceBalsara_,
            )

        # with record_function("warpSPH[CRKVolume] - Kernel Execution"):
        #     warp_result = warpWrapper(
        #         launch_kernel, computeViscosity_Kernel, outputSize, outputDtype,
        #         *args,
        #         queryVelocities_, referenceVelocities_,
        #         individual_cs, queryCs_, referenceCs_,
        #         viscositySwitch, queryAlphas_, referenceAlphas_,
        #         explicitPressure, queryPressures_, referencePressures_,
        #         viscosityParams
        #     )

    return warp_result
