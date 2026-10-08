"""SPH viscous-heating term (the du/dt production from artificial viscosity
converting kinetic energy into heat): builds the momentum Pi_ij term via
`dissipation.pi.computePi_actual` (`thermalConductivity=False`) and
multiplies it by `ux_ij**2` and the kernel-gradient Laplacian, using the
same query/reference neighbor loop and correction-term pattern as the other
`modules/` operators. This is distinct from `wp_conductivity.py`, which
diffuses internal-energy *differences* rather than producing heat from
velocity divergence.
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

__all__ = ['computeThermalDissipationWarp']

@wp.func
def computeThermalDissipation_Func_i(
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
    
    vel_i: vector(length=Any, dtype=scalar_t), referenceVelocities: wp.array(dtype = vector(length=Any, dtype=scalar_t)), # type: ignore
    u_i: scalar_t, referenceEnergies: wp.array(dtype = scalar_t), # type: ignore

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
        u_j = referenceEnergies[j]

        apparentVolume = access_optional(referenceVolumes, j, useVolume, mj / rhoj)

        x_ij = computeDistanceVec(xi, xj, domainState)
        r_ij = safe_sqrt(wp.dot(x_ij, x_ij))
        # the pair velocity of the viscous force (`wp_diffusion.py`): Pi is evaluated on it
        B_j = scalar_t(0.0)
        if viscosityParams.velocityPairPolicy == wp.static(VelocityPairPolicy.BalsaraLimited.value) or viscosityParams.balsaraPairLimiter:
            B_j = referenceBalsara[j]
        uHat_ij = rawPairVelocity(vel_i, vel_j)
        if viscosityParams.velocityPairPolicy != wp.static(VelocityPairPolicy.Raw.value):
            uHat_ij = reconstructPairVelocity(
                viscosityParams.velocityPairPolicy, vel_i, vel_j, J_i, referenceVelocityTensor[j], x_ij, hi, hj,
                kernelProperties.kernelFunction, dim,
                viscosityParams.reconstructionEtaCrit, viscosityParams.reconstructionEtaFold,
                B_i, B_j, viscosityParams.reconstructionBalsaraPower)

        pi = computePi_pair(
            xi, xj, 
            hi, hj,
            mi, mj,
            rhoi, rhoj,
            explicitPressure, P_i, access_optional(referencePressures, j, explicitPressure, scalar_t(0.0)),
            uHat_ij,
            domainState,
            kernelProperties.kernelFunction,
            cs_i, access_optional(referenceCs, j, individual_cs, viscosityParams.c_s),
            alpha_i, access_optional(referenceAlphas, j, viscositySwitch, scalar_t(1.0)),
            viscosityParams, 
            False, False)
        
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

        
        u_ij = vel_j - vel_i
        ux_ij = wp.dot(u_ij, x_ij) / (r_ij + scalar_t(1.0e-14) * hi)
        mu_ij = ux_ij #/ (r_ij + scalar_t(1.0e-14) * hi)

        laplacian_ij = wp.dot(
            gradw_ij, x_ij / (r_ij + scalar_t(1.0e-14) * hi)
        )
        h_bar = scalar_t(1.0)/scalar_t(2.0) * (hi + hj)
        
        fac = scalar_t(1.0) / (r_ij + scalar_t(1.0e-14) * hi)

        # Viscous heating du_i/dt = 1/2 sum_j m_j Pi_ij v_ij . gradW_ij (Monaghan 1992): the 1/2 is what makes the
        # heat equal the kinetic energy the viscous force removes (each pair's dissipation is split between the
        # two particles). Without it the heating is exactly 2x the loss, so the scheme gains one full dissipation's
        # worth of total energy (OPEN_PROBLEMS section 16; found with scripts/probe_monaghanEnergy.py).
        # With a reconstructed pair velocity the force is Pi(u') (u'.x/r) grad W, so the kinetic energy it removes is
        # 1/2 sum m_i m_j Pi (u'.x/r)(v_ij.x/r) |grad W|: one factor reconstructed, one raw (as CRKSPH's dudt and
        # Garcia-Senz & Cabezon 2026 do). The raw policy keeps ux^2 bit for bit.
        uxProduct = iPow(ux_ij, 2)
        if viscosityParams.velocityPairPolicy != wp.static(VelocityPairPolicy.Raw.value):
            uxProduct = -wp.dot(uHat_ij, x_ij) / (r_ij + scalar_t(1.0e-14) * hi) * ux_ij
        out += - scalar_t(0.5) * apparentVolume * pi * uxProduct * laplacian_ij #* fac
        
    return out



@wp.func
def computeThermalDissipation_Func_Adjacency(
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
    queryEnergies: wp.array(dtype = scalar_t), referenceEnergies: wp.array(dtype = scalar_t), # type: ignore
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
    u_i = access_optional(queryEnergies, i, True, scalar_t(0.0))
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
        
        out += computeThermalDissipation_Func_i(
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
            u_i, referenceEnergies,
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
def computeThermalDissipation_Kernel(
    queryState: Any,
    referenceState: Any,
    domainState: domainData,

    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any,
    
    kernelProperties: kernelState,
    # Do not change the parameters above
    queryVelocities: wp.array(dtype = vector(length=Any, dtype=scalar_t)), referenceVelocities: wp.array(dtype = vector(length=Any, dtype=scalar_t)), # type: ignore
    queryEnergies: wp.array(dtype = scalar_t), referenceEnergies: wp.array(dtype = scalar_t), # type: ignore
    individual_cs: wp.bool, queryCs: wp.array(dtype = scalar_t), referenceCs: wp.array(dtype = scalar_t), # type: ignore
    viscositySwitch: wp.bool, queryAlphas: wp.array(dtype = scalar_t), referenceAlphas: wp.array(dtype = scalar_t), # type: ignore
    explicitPressure: wp.bool, queryPressures: wp.array(dtype = scalar_t), referencePressures: wp.array(dtype = scalar_t), # type: ignore
    viscosityParams: DiffusionParameters,
    queryVelocityTensor: wp.array(dtype = matrix(shape=(Any, Any), dtype=scalar_t)), referenceVelocityTensor: wp.array(dtype = matrix(shape=(Any, Any), dtype=scalar_t)), # type: ignore
    queryBalsara: wp.array(dtype = scalar_t), referenceBalsara: wp.array(dtype = scalar_t), # type: ignore
    # The last parameter is always the output array and should not be changed
    outputValues : wp.array(dtype = scalar_t) # type: ignore
):                                                                                    
    i = wp.tid()
    numParticles = queryState.positions.shape[0]
    if i >= numParticles:
        return

    outputValues[i] = computeThermalDissipation_Func_Adjacency(
        i, domainState.dim, 
        queryState, referenceState, correctionData, domainState,
        useAdjacency, adjacencyState, gridState, gridState.numOffsets if not useAdjacency else 1,
        kernelProperties,  #queryKinds, referenceKinds,
        # The parameters above are default parameters and shold not be changed
        queryVelocities, referenceVelocities,
        queryEnergies, referenceEnergies,
        individual_cs, queryCs, referenceCs,
        viscositySwitch, queryAlphas, referenceAlphas,
        explicitPressure, queryPressures, referencePressures,
        viscosityParams,
        queryVelocityTensor, referenceVelocityTensor,
        queryBalsara, referenceBalsara,

        zero_like_warp(outputValues)
    )

def _thermalDissipationDtype(ctx, extras):
    return castTorchToWarpAsBuiltins(ctx.query.densities).dtype


_THERMAL_DISSIPATION = OperatorSpec(
    kernel=computeThermalDissipation_Kernel,
    outputs=(OutputSpec(dtype=_thermalDissipationDtype, shape=ShapeOf.QUERY),),
    extras=(
        ExtraSpec("queryVelocities", ExtraKind.TENSOR),
        ExtraSpec("referenceVelocities", ExtraKind.TENSOR),
        ExtraSpec("queryEnergies", ExtraKind.TENSOR),
        ExtraSpec("referenceEnergies", ExtraKind.TENSOR),
        ExtraSpec("individualCs", ExtraKind.SCALAR),
        ExtraSpec("queryCs", ExtraKind.TENSOR),
        ExtraSpec("referenceCs", ExtraKind.TENSOR),
        ExtraSpec("viscositySwitch", ExtraKind.SCALAR),
        ExtraSpec("queryAlphas", ExtraKind.TENSOR),
        ExtraSpec("referenceAlphas", ExtraKind.TENSOR),
        ExtraSpec("explicitPressure", ExtraKind.SCALAR),
        ExtraSpec("queryPressures", ExtraKind.TENSOR),
        ExtraSpec("referencePressures", ExtraKind.TENSOR),
        ExtraSpec("conductivityParams", ExtraKind.SCALAR),
        ExtraSpec("queryVelocityTensor", ExtraKind.TENSOR),
        ExtraSpec("referenceVelocityTensor", ExtraKind.TENSOR),
        ExtraSpec("queryBalsara", ExtraKind.TENSOR),
        ExtraSpec("referenceBalsara", ExtraKind.TENSOR),
    ),
)


def computeThermalDissipationWarp(
    queryParticles: ParticleState,
    operationProperties: OperationProperties,
    domain: DomainDescription,
    
    conductivityParams: DiffusionParameters,
    queryEnergies: Optional[torch.Tensor] = None, referenceEnergies: Optional[torch.Tensor] = None,
    queryVelocities : Optional[torch.Tensor] = None, referenceVelocities: Optional[torch.Tensor] = None,
    queryCs: Optional[torch.Tensor] = None, referenceCs: Optional[torch.Tensor] = None,
    queryAlphas: Optional[torch.Tensor] = None, referenceAlphas: Optional[torch.Tensor] = None,
    queryPressures: Optional[torch.Tensor] = None, referencePressures: Optional[torch.Tensor] = None,

    queryVolumes: Optional[torch.Tensor] = None, referenceVolumes: Optional[torch.Tensor] = None,
    adjacency: Optional[Union[AdjacencyList, CompactHashMap]] = None, # if none a datastructure is created for EVERY operation!,
    referenceParticles: Optional[ParticleState] = None,
    crkState: Optional[CRKState] = None,
    gradHState: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], GradHState]] = None,
    renormalizationState: Optional[Union[torch.Tensor,RenormalizationState]] = None,
    queryVelocityTensor: Optional[torch.Tensor] = None, referenceVelocityTensor: Optional[torch.Tensor] = None,
    queryBalsara: Optional[torch.Tensor] = None, referenceBalsara: Optional[torch.Tensor] = None,
):
    if referenceVelocities is None:
        referenceVelocities = queryVelocities
    if referenceCs is None:
        referenceCs = queryCs
    if referenceAlphas is None:
        referenceAlphas = queryAlphas
    if referencePressures is None:
        referencePressures = queryPressures
    with record_function("warpSPH[computeThermalDissipation]"):
        with record_function("warpSPH[computeThermalDissipation] - Preprocessing"):
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
            
            queryEnergies_ = queryEnergies if queryEnergies is not None else (queryParticles.internalEnergies if hasattr(queryParticles, 'internalEnergies') else None)
            queryVelocities_ = queryVelocities if queryVelocities is not None else (queryParticles.velocities if hasattr(queryParticles, 'velocities') else None)
            queryCs_ = queryCs if queryCs is not None else (queryParticles.soundspeeds if hasattr(queryParticles, 'soundspeeds') else getCachedDummyTensor((1,), dtype=get_torch_precision(), device=device))
            queryAlphas_ = queryAlphas if queryAlphas is not None else (queryParticles.alphas if hasattr(queryParticles, 'alphas') else getCachedDummyTensor((1,), dtype=get_torch_precision(), device=device))
            queryPressures_ = queryPressures if queryPressures is not None else (queryParticles.pressures if hasattr(queryParticles, 'pressures') else getCachedDummyTensor((1,), dtype=get_torch_precision(), device=device))

            referenceEnergies_ = referenceEnergies if referenceEnergies is not None else (referenceParticles.internalEnergies if hasattr(referenceParticles, 'internalEnergies') else None)
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
            if queryEnergies_ is None:
                raise ValueError("Energies must be provided either through queryEnergies or as a property of queryParticles.")
            queryVelocityTensor_, referenceVelocityTensor_, queryBalsara_, referenceBalsara_ = velocityTensorArguments(
                conductivityParams, queryVelocityTensor, referenceVelocityTensor, queryParticles.positions,
                queryBalsara, referenceBalsara)

        with record_function("warpSPH[computeThermalDissipation] - Kernel Execution"):
            ctx = SPHContext(
                query=queryParticles, properties=operationProperties, domain=domain,
                adjacency=adjacency, reference=referenceParticles,
                corrections=Corrections(
                    volumes=(queryVolumes, referenceVolumes),
                    crk=crkState, gradH=gradHState, renorm=renormalizationState,
                ),
            )
            return launchOperator(
                _THERMAL_DISSIPATION, ctx,
                queryVelocities=queryVelocities_, referenceVelocities=referenceVelocities_,
                queryEnergies=queryEnergies_, referenceEnergies=referenceEnergies_,
                individualCs=individual_cs, queryCs=queryCs_, referenceCs=referenceCs_,
                viscositySwitch=viscositySwitch, queryAlphas=queryAlphas_, referenceAlphas=referenceAlphas_,
                explicitPressure=explicitPressure, queryPressures=queryPressures_, referencePressures=referencePressures_,
                conductivityParams=conductivityParams,
                queryVelocityTensor=queryVelocityTensor_, referenceVelocityTensor=referenceVelocityTensor_,
                queryBalsara=queryBalsara_, referenceBalsara=referenceBalsara_,
            )


        # with record_function("warpSPH[CRKVolume] - Kernel Execution"):
        #     warp_result = warpWrapper(
        #         launch_kernel, computeThermalDissipation_Kernel, outputSize, outputDtype,
        #         *args,
        #         queryVelocities_, referenceVelocities_,
        #         queryEnergies_, referenceEnergies_,
        #         individual_cs, queryCs_, referenceCs_,
        #         viscositySwitch, queryAlphas_, referenceAlphas_,
        #         explicitPressure, queryPressures_, referencePressures_,
        #         conductivityParams
        #     )

    return warp_result
