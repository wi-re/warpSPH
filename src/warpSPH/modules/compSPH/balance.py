"""compSPH pairwise energy-partition factor.

Computes `f_ij`, the fraction of pair-wise thermal work assigned to particle
`i` (vs. `j`) for each interacting pair, consuming the pressure/viscosity
accelerations produced by `accel.py`. Selectable via `EnergyScheme`:
`equalWork` (fixed 0.5), `PdV` (pressure-work weighted, noted in-code as not
working with the current multistep code), `diminishing`, `monotonic`,
`hybrid` (blend of the two), and `CRK` (entropy-ratio based, per CRKSPH).
Several branches use small epsilons (`1.0e-14`) to guard divisions.
"""

import warp as wp
from warp.types import vector, matrix
from typing import Any
import torch
from torch.profiler import profile, ProfilerActivity
from warpSPHCore.profiling import record_function
from typing import Optional, Union, Tuple
from warpSPHCore import *

__all__ = ['computeCompSPHBalanceTermWarp']


@wp.func 
def sgn(x: scalar_t) -> scalar_t:
    return scalar_t(1.0) if x > 0 else (scalar_t(-1.0) if x < 0 else scalar_t(0.0))

@wp.func
def computeCompSPHBalanceTerm_Func_i(
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

    P_i: scalar_t, referencePressures: wp.array(dtype = scalar_t), # type: ignore
    ap_ij: wp.array(dtype = vector(length=Any, dtype=scalar_t)), # type: ignore
    av_ij: wp.array(dtype = vector(length=Any, dtype=scalar_t)), # type: ignore
    energyScheme_int: wp.int32, dt: scalar_t, gamma: scalar_t,  # type: ignore

    f_ij : wp.array(dtype = Any) # type: ignore
):
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

        apparentVolume = mj / rhoj if not useVolume else referenceVolumes[j]


        v_ji            = vel_j - vel_i    
        u_ji            = u_j - u_i
        a_ij            = ap_ij[jj] + av_ij[jj]
        P_j = referencePressures[j]

        deltaE_thermal  = mi * wp.dot(v_ji, a_ij) * dt

        f = scalar_t(0.5)

        if energyScheme_int == wp.static(EnergyScheme.equalWork.value):
            f = scalar_t(0.5)
        elif energyScheme_int == wp.static(EnergyScheme.PdV.value):
        # raise NotImplementedError # Doesnt work with the current multistep code        
            f = (P_i / (rhoi*rhoi)) / (P_i / (rhoi * rhoi) + P_j / (rhoj * rhoj))
        elif energyScheme_int == wp.static(EnergyScheme.diminishing.value):
            u_ji_norm = wp.abs(u_ji)        
            f = scalar_t(0.5) * (scalar_t(1.0) + (u_ji * sgn(deltaE_thermal)) / (u_ji_norm + scalar_t(1.0) / (scalar_t(1.0) + u_ji_norm)))
        elif energyScheme_int == wp.static(EnergyScheme.monotonic.value):
            A = deltaE_thermal / (u_ji + scalar_t(1.0e-14))
            # B = torch.where(A >= 0, A / m_i, A / m_j)
            if A >= scalar_t(0.0):
                B = A / mi
            else:
                B = A / mj

            # term_1 = torch.maximum(torch.zeros_like(B), sgn(B))
            term_1 = wp.max(scalar_t(0.0), sgn(B))
            term_2 = mi / (deltaE_thermal + scalar_t(1.0e-14)) * ( (deltaE_thermal + mi * u_i + mj * u_j) / (mi + mj) - u_i)

            # f_ij = torch.where(torch.abs(B) <= 1, term_1, term_2)
            if wp.abs(B) <= scalar_t(1.0):
                f = term_1
            else:
                f = term_2
        elif energyScheme_int == wp.static(EnergyScheme.hybrid.value):
            u_ji_norm = wp.abs(u_ji)
            A = deltaE_thermal / (u_ji + scalar_t(1.0e-14))
            # B = torch.where(A >= 0, A / m_i, A / m_j)
            if A >= scalar_t(0.0):
                B = A / mi
            else:
                B = A / mj

            # term_1 = torch.maximum(torch.zeros_like(B), sgn(B))
            term_1 = wp.max(scalar_t(0.0), sgn(B))
            term_2 = mi / (deltaE_thermal + scalar_t(1.0e-14)) * ( (deltaE_thermal + mi * u_i + mj * u_j) / (mi + mj) - u_i)

            # f_ij_mono = torch.where(torch.abs(B) <= 1, term_1, term_2)
            if wp.abs(B) <= scalar_t(1.0):
                f_ij_mono = term_1
            else:
                f_ij_mono = term_2
            f_ij_sm = scalar_t(0.5) * (scalar_t(1.0) + (u_ji * sgn(deltaE_thermal)) / (u_ji_norm + scalar_t(1.0) / (scalar_t(1.0) + u_ji_norm)))
            
            chi = wp.abs(u_ji) / (wp.abs(u_i) + wp.abs(u_j) + scalar_t(1.0e-14))
            # chi = 1-chi
            f = chi * f_ij_mono + (scalar_t(1.0) - chi) * f_ij_sm
        elif energyScheme_int == wp.static(EnergyScheme.CRK.value):
            s_i = P_i / wp.pow(rhoi, gamma)
            s_j = P_j / wp.pow(rhoj, gamma)
            s_min = wp.min(wp.abs(s_i), wp.abs(s_j))
            s_max = wp.max(wp.abs(s_i), wp.abs(s_j))

            f = scalar_t(0.5)
            
            maskA = wp.abs(s_i - s_j) == scalar_t(0.0)

            # maskB = torch.logical_or(torch.logical_and(deltaE_thermal >= 0, s_i >= s_j), torch.logical_and(deltaE_thermal < 0, s_i < s_j))
            maskB = (deltaE_thermal >= scalar_t(0.0) and s_i >= s_j) or (deltaE_thermal < scalar_t(0.0) and s_i < s_j)

            # maskC = torch.logical_or(torch.logical_and(deltaE_thermal >= 0, s_i < s_j), torch.logical_and(deltaE_thermal < 0, s_i >= s_j))
            maskC = (deltaE_thermal >= scalar_t(0.0) and s_i < s_j) or (deltaE_thermal < scalar_t(0.0) and s_i >= s_j)

            f_ijA = scalar_t(0.5)
            f_ijB = s_min / (s_min + s_max + scalar_t(1.0e-14))
            f_ijC = s_max / (s_min + s_max + scalar_t(1.0e-14))
                
            # f_ij = torch.where(maskA, f_ijA, f_ij)
            f = f_ijA if maskA else f
            # f_ij = torch.where(torch.logical_and(torch.logical_not(maskA), maskB), f_ijB, f_ij)
            f = f_ijB if (not maskA and maskB) else f
            # f_ij = torch.where(torch.logical_and(torch.logical_and(torch.logical_not(maskA), torch.logical_not(maskB)), maskC), f_ijC, f_ij)
            f = f_ijC if (not maskA and not maskB and maskC) else f

            f = wp.max(scalar_t(0.0), wp.min(scalar_t(1.0), f))

        f_ij[jj] = f




@wp.func
def computeCompSPHBalanceTerm_Func_Adjacency(
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
    
    queryPressures: wp.array(dtype = scalar_t), referencePressures: wp.array(dtype = scalar_t), # type: ignore
    pressureAccel_ij: wp.array(dtype = vector(length=Any, dtype=scalar_t)), # type: ignore
    viscosityAccel_ij: wp.array(dtype = vector(length=Any, dtype=scalar_t)), # type: ignore
    energyScheme_int: wp.int32, dt: scalar_t, gamma: scalar_t,  # type: ignore

    f_ij : wp.array(dtype = Any), # type: ignore
):
    xi, hi, mi, rhoi, ki = getParticle(queryState, i)
    if kernelProperties.operationMode != wp.static(OperationDirection.TrueAllToToAll.value):
        if not checkDirectionality_i(ki, kernelProperties.operationMode):
            return zero_like_warp(pressureAccel_ij)
        
    useGradientRenormalization, Li = getL_i(correctionData, i)
    useGradHTerms, omega_i = getGradH_i(correctionData, i)
    useVolume, Vi = getVolume_i(correctionData, i)
    useCRK, Ai, Bi, gradA_i, gradB_i = getCRK_i(correctionData, i)
    vel_i = queryVelocities[i]

    u_i = queryEnergies[i]
    P_i = queryPressures[i]

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
        
        computeCompSPHBalanceTerm_Func_i(
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
            
            P_i, referencePressures,
            pressureAccel_ij, viscosityAccel_ij,
            energyScheme_int, dt, gamma,
            f_ij
        )



@wp.kernel
def computeCompSPHBalanceTerm_Kernel(
    queryState: Any,
    referenceState: Any,
    domainState: domainData,

    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any,
    
    kernelProperties: kernelState,
    # Do not change the parameters above
    queryVelocities: wp.array(dtype = vector(length=Any, dtype=scalar_t)), referenceVelocities: wp.array(dtype = vector(length=Any, dtype=scalar_t)), # type: ignore
    queryEnergies: wp.array(dtype = scalar_t), referenceEnergies: wp.array(dtype = scalar_t), # type: ignore
    queryPressures: wp.array(dtype = scalar_t), referencePressures: wp.array(dtype = scalar_t), # type: ignore

    pressureAccel_ij: wp.array(dtype = vector(length=Any, dtype=scalar_t)), # type: ignore
    viscosityAccel_ij: wp.array(dtype = vector(length=Any, dtype=scalar_t)), # type: ignore
    energyScheme_int: wp.int32, dt: wp.array(dtype = scalar_t), gamma: wp.array(dtype = scalar_t), # type: ignore

    # The last parameter is always the output array and should not be changed
    f_ij : wp.array(dtype = Any), # type: ignore
):
    i = wp.tid()
    numParticles = queryState.positions.shape[0]
    if i >= numParticles:
        return

    computeCompSPHBalanceTerm_Func_Adjacency(
        i, domainState.dim,
        queryState, referenceState, correctionData, domainState,
        useAdjacency, adjacencyState, gridState, gridState.numOffsets if not useAdjacency else 1,
        kernelProperties,  #queryKinds, referenceKinds,
        # The parameters above are default parameters and shold not be changed
        queryVelocities, referenceVelocities,
        queryEnergies, referenceEnergies,
        queryPressures, referencePressures,
        pressureAccel_ij, viscosityAccel_ij,
        energyScheme_int, dt[0], gamma[0],
        f_ij
    )

from ...enumTypes import EnergyScheme

def _balanceTermDtype(ctx, extras):
    return castTorchToWarpAsBuiltins(ctx.query.densities).dtype


def _balancePairShape(ctx, extras):
    return ctx.adjacency.i.shape[0]


def _balanceNumThreads(ctx, extras):
    return ctx.query.positions.shape[0]


_COMPSPH_BALANCE_TERM = OperatorSpec(
    kernel=computeCompSPHBalanceTerm_Kernel,
    outputs=(OutputSpec(dtype=_balanceTermDtype, shape=_balancePairShape),),
    numThreads=_balanceNumThreads,
    extras=(
        ExtraSpec("queryVelocities", ExtraKind.TENSOR),
        ExtraSpec("referenceVelocities", ExtraKind.TENSOR),
        ExtraSpec("queryEnergies", ExtraKind.TENSOR),
        ExtraSpec("referenceEnergies", ExtraKind.TENSOR),
        ExtraSpec("queryPressures", ExtraKind.TENSOR),
        ExtraSpec("referencePressures", ExtraKind.TENSOR),
        ExtraSpec("pairWisePressureAccel", ExtraKind.TENSOR),
        ExtraSpec("pairWiseViscosityAccel", ExtraKind.TENSOR),
        ExtraSpec("energyScheme", ExtraKind.SCALAR),
        ExtraSpec("dt", ExtraKind.TENSOR),
        ExtraSpec("gamma", ExtraKind.TENSOR),
    ),
)


def computeCompSPHBalanceTermWarp(
    queryParticles: ParticleState,
    operationProperties: OperationProperties,
    domain: DomainDescription,
    
    energyScheme: EnergyScheme,
    dt: Union[float, torch.Tensor],
    gamma: Union[float, torch.Tensor],

    pairWise_pressureAccel: torch.Tensor, #ap_ij
    pairWise_viscosityAccel: torch.Tensor, #av_ij

    queryEnergies: Optional[torch.Tensor] = None, referenceEnergies: Optional[torch.Tensor] = None,
    queryVelocities : Optional[torch.Tensor] = None, referenceVelocities: Optional[torch.Tensor] = None,
    queryPressures: Optional[torch.Tensor] = None, referencePressures: Optional[torch.Tensor] = None,
    queryVolumes: Optional[torch.Tensor] = None, referenceVolumes: Optional[torch.Tensor] = None,

    adjacency: Optional[Union[AdjacencyList, CompactHashMap]] = None, # if none a datastructure is created for EVERY operation!,
    referenceParticles: Optional[ParticleState] = None,
    crkState: Optional[CRKState] = None,
    gradHState: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], GradHState]] = None,
    renormalizationState: Optional[Union[torch.Tensor,RenormalizationState]] = None,
):
    if adjacency is None or isinstance(adjacency, CompactHashMap):
        raise ValueError("This module requires an adjacency structure.")
    if referenceVelocities is None:
        referenceVelocities = queryVelocities
    if referenceEnergies is None:
        referenceEnergies = queryEnergies
    if referencePressures is None:
        referencePressures = queryPressures
    with record_function("warpSPH[computeCompSPHBalanceTerm]"):
        with record_function("warpSPH[computeCompSPHBalanceTerm] - Preprocessing"):
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
            queryPressures_ = queryPressures if queryPressures is not None else (queryParticles.pressures if hasattr(queryParticles, 'pressures') else None)

            referenceEnergies_ = referenceEnergies if referenceEnergies is not None else (referenceParticles.internalEnergies if hasattr(referenceParticles, 'internalEnergies') else None)
            referenceVelocities_ = referenceVelocities if referenceVelocities is not None else (referenceParticles.velocities if hasattr(referenceParticles, 'velocities') else None)
            referencePressures_ = referencePressures if referencePressures is not None else (referenceParticles.pressures if hasattr(referenceParticles, 'pressures') else None)


            if queryVelocities_ is None:
                raise ValueError("Velocities must be provided either through queryVelocities or as a property of queryParticles.")
            if queryEnergies_ is None:
                raise ValueError("Energies must be provided either through queryEnergies or as a property of queryParticles.")
            if queryPressures_ is None:
                raise ValueError("Pressures must be provided either through queryPressures or as a property of queryParticles.")
            
            # print(f'Energy Scheme: {energyScheme} [value: {energyScheme.value}]')

        with record_function("warpSPH[computeCompSPHBalanceTerm] - Kernel Execution"):
            ctx = SPHContext(
                query=queryParticles, properties=operationProperties, domain=domain,
                adjacency=adjacency, reference=referenceParticles,
                corrections=Corrections(
                    volumes=(queryVolumes, referenceVolumes),
                    crk=crkState, gradH=gradHState, renorm=renormalizationState,
                ),
            )
            return launchOperator(
                _COMPSPH_BALANCE_TERM, ctx,
                queryVelocities=queryVelocities_, referenceVelocities=referenceVelocities_,
                queryEnergies=queryEnergies_, referenceEnergies=referenceEnergies_,
                queryPressures=queryPressures_, referencePressures=referencePressures_,
                pairWisePressureAccel=pairWise_pressureAccel, pairWiseViscosityAccel=pairWise_viscosityAccel,
                energyScheme=wp.int32(energyScheme.value), dt=asScalarArg(dt, device=device),
                gamma=asScalarArg(gamma, device=device),
            )


        # with record_function("warpSPH[CRKVolume] - Kernel Execution"):
        #     warp_result = warpWrapper(
        #         launch_kernel, computeCompSPHBalanceTerm_Kernel, outputSize, outputDtype,
        #         *args,
        #         queryVelocities_, referenceVelocities_,
        #         queryEnergies_, referenceEnergies_,
        #         individual_cs, queryCs_, referenceCs_,
        #         viscositySwitch, queryAlphas_, referenceAlphas_,
        #         explicitPressure, queryPressures_, referencePressures_,
        #         conductivityParams
        #     )

    return warp_result
