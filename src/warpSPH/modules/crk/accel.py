"""CRKSPH pressure and artificial-viscosity acceleration.

Computes the CRK-corrected SPH momentum-equation acceleration: a
volume-weighted pressure force plus a Riemann-like pseudo-viscosity `Q_i,
Q_j` built from velocity-gradient-reconstructed pair velocities
(`v_dot_i`/`v_dot_j`), optionally slope-limited via the van Leer
(`enableVanLeerLimiter`) and eta-based (`enableCRKLimiter`) limiters from
`modules/reconstruction` (each individually forceable on/off via
`crkViscosityParams.forceVanLeerOn/Off`). Also returns the per-pair
pressure/viscosity contributions consumed elsewhere for diagnostics.
"""

import warp as wp
from warp.types import vector, matrix
from typing import Any
import torch
from torch.profiler import profile, ProfilerActivity
from warpSPHCore.profiling import record_function
from typing import Optional, Union, Tuple
from warpSPHCore import *

from ...configurations.crkSPH import CRKViscosity
from ..dissipation import DiffusionParameters, computeFrontiereQ

from ..reconstruction import limitedPairPhi, linearPairVelocity

__all__ = ['computeCrkSPHAccelWarp']

@wp.func
def computeCrkSPHAccel_Func_i(
    # General Shape Parameters and indices
    i : wp.int32,  dim: wp.int32, 

    # SPH properties for the query set (indexed by i)
    xi: vector(dtype = scalar_t, length=Any), hi: scalar_t, mi: scalar_t, rhoi: scalar_t, # type: ignore

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
    ki : wp.int32, referenceKinds : wp.array(dtype = wp.int32), # type: ignore

    # Optional Correction Terms:
    # Gradient renormalization matrices for each query point, used for correcting the kernel gradient based on the local particle distribution.
    useGradientRenormalization: wp.bool, Li: matrix(shape=(Any, Any), dtype=scalar_t), # type: ignore
    # Grad-h correction terms for each query and reference point, used for correcting the kernel gradient based on the local particle distribution and smoothing length variations.
    useGradHTerms: wp.bool, omega_i: scalar_t,   # type: ignore
    # Whether to use actual volume (mass/density) or apparent volume for the gradient computation, and the corresponding volumes if needed.
    useVolume: bool, Vi: scalar_t,  # type: ignore
    # Whether to use CRK kernel correction for the computation, and the corresponding correction terms if needed.
    useCRK: bool, Ai: scalar_t, Bi: vector(length=Any, dtype=scalar_t), gradAi: vector(length=Any, dtype=scalar_t), gradBi: matrix(shape=(Any, Any), dtype=scalar_t), # type: ignore
    
    vel_i: vector(length=Any, dtype=scalar_t), referenceVelocities: wp.array(dtype = vector(length=Any, dtype=scalar_t)), # type: ignore
    u_i: scalar_t, referenceEnergies: wp.array(dtype = scalar_t), # type: ignore

    individual_cs: wp.bool, cs_i: scalar_t, referenceCs: wp.array(dtype = scalar_t), # type: ignore
    viscositySwitch: wp.bool, alpha_i: scalar_t, referenceAlphas: wp.array(dtype = scalar_t), # type: ignore
    explicitPressure: wp.bool, P_i: scalar_t, referencePressures: wp.array(dtype = scalar_t), # type: ignore
    viscosityParams: DiffusionParameters,
    crkViscosityParams: CRKViscosity,

    gradV_i: matrix(shape=(Any, Any), dtype=scalar_t), referenceVelocityTensor: wp.array(dtype = matrix(shape=(Any, Any), dtype=scalar_t)),# type: ignore

    # Dummy value to allow allocation
    pressureAccel_ij: wp.array(dtype = vector(length=Any, dtype=scalar_t)), # type: ignore
    viscosityAccel_ij: wp.array(dtype = vector(length=Any, dtype=scalar_t)), # type: ignore
):
    pressureTerm_i = P_i / (rhoi*rhoi )/ (omega_i if useGradHTerms else scalar_t(1.0))

    # Initialize the output value
    out = zero_like_warp(pressureAccel_ij[i])
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

        _, Vj = getVolume_j(correctionData, j)
        P_j = access_optional(referencePressures, j, explicitPressure, scalar_t(0.0))
        cs_j = access_optional(referenceCs, j, individual_cs, viscosityParams.c_s)


        x_ij = computeDistanceVec(xi, xj, domainState)
        r_ij = safe_sqrt(wp.dot(x_ij, x_ij))

        gradV_j = referenceVelocityTensor[j]

        # CRKSPH's limited midpoint reconstruction (`modules/reconstruction`, AV_PLAN Phase 3): phi_ij from the van Leer
        # limiter times the close-pair taper, unless forced off (raw velocities) or on (unlimited linear). With
        # x_ij = x_i - x_j: v_dot_i = v_i - phi/2 J_i x_ij, v_dot_j = v_j + phi/2 J_j x_ij (garciasenz2026 Eqs. 11-12)
        phi_ij = limitedPairPhi(
            x_ij, hi, hj, vel_i, vel_j, gradV_i, gradV_j,
            kernelProperties.kernelFunction, dim,
            crkViscosityParams.enableVanLeerLimiter, crkViscosityParams.enableCRKLimiter,
            crkViscosityParams.eta_crit, crkViscosityParams.eta_fold)
        if crkViscosityParams.forceVanLeerOff:
            phi_ij = scalar_t(0.0)
        if crkViscosityParams.forceVanLeerOn:
            phi_ij = scalar_t(1.0)
        phi_ij = wp.max(wp.min(phi_ij, scalar_t(1.0)), scalar_t(0.0)) # Ensure phi is between 0 and 1
        vij_dot = linearPairVelocity(vel_i, vel_j, gradV_i, gradV_j, x_ij, phi_ij)

        gradw_i = computeKernelGradientCRK(
            xi, xj, 
            hj, hj, # forces gather
            kernelProperties, domainState,    
            True, Ai, Bi, gradAi, gradBi
        )
        if useGradientRenormalization:
            gradw_i = matmul(Li, gradw_i)

        _, Aj, Bj, gradAj, gradBj = getCRK_j(correctionData, j)
        gradw_j = -computeKernelGradientCRK(
            xj, xi,
            hi, hi, # forces scatter
            kernelProperties, domainState,    
            True, Aj, Bj, gradAj, gradBj
        )
        if useGradientRenormalization:
            gradw_j = matmul(Li, gradw_j)

        # Frontiere et al. (2017) Eq. (69) one-sided viscous pressures Q_i, Q_j of the reconstructed pair velocity;
        # the formula (mu with the particle's own h, eps^2 = 1e-2, min(0, .), switched alpha / BetaMode) lives once in the
        # `Frontiere2017` term of `modules/dissipation/pi/terms.py`
        alpha_j = access_optional(referenceAlphas, j, viscositySwitch, scalar_t(1.0))
        Q_i = computeFrontiereQ(xi, xj, hi, hj, rhoi, rhoj, vij_dot, domainState, kernelProperties.kernelFunction,
                                cs_i, cs_j, alpha_i, alpha_j, viscosityParams, False)
        Q_j = computeFrontiereQ(xi, xj, hi, hj, rhoi, rhoj, vij_dot, domainState, kernelProperties.kernelFunction,
                                cs_i, cs_j, alpha_i, alpha_j, viscosityParams, True)


        gradw_ij = scalar_t(0.5) * (gradw_i + gradw_j)
        # gradw_ij = gradw_i
        # gradw_j = gradw_i # E.2 in crksph suggests using the super symmetric form


        # omegaj = referenceOmegas[j] if useGradHTerms else scalar_t(1.0)
        # pressureTerm_j = Pj / (rhoj*rhoj) / omegaj
        
        mu_ij = scalar_t(1.0)
        # mu_ij = ux_ij #/ (r_ij + scalar_t(1.0e-14) * hi)

        # note that the term here should be multiplied with rho_i if it was a gradient operation
        # however, because we are computing the pressure force this cancel out with the division by rho_i in the pressure term, so we do not include it here
        # we do need to include the minus sign because the pressure force is -P gradW
        # Vj = referenceVolumes[j]
        # Vi = mi/rhoi
        # Vj = mj/rhoj
        pairWeight = Vi * Vj
        if crkViscosityParams.meanVolumeWeights:
            pairWeight = scalar_t(0.25) * (Vi + Vj) * (Vi + Vj)
        pressureTerm_ij = -(P_i + P_j) * gradw_ij * pairWeight / mi
        # pressureTerms_i = -P_i * gradw_i * Vi * Vj / mi
        # pressureTerms_j = -P_j * gradw_j * Vi * Vj / mi


        # pressureTerms_i = - P_i * gradw_i * Vi * Vj / mi
        # pressureTerms_j = - P_j * gradw_j * Vi * Vj / mi
        # pressureTerm_ij = (pressureTerms_i + pressureTerms_j)

        rho_bar = scalar_t(0.5) * (rhoi + rhoj)
        # Q_i = pi_i * rho_bar / rhoj * rhoi
        # Q_j = pi_j * rho_bar

        viscosityTerm_ij = -(Q_i + Q_j) * mu_ij * gradw_ij * pairWeight / mi
        # viscosityTerms_i = - Q_i * Vj * mu_ij * gradw_i * Vi / mi
        # viscosityTerms_j = - Q_j * Vj * mu_ij * gradw_j * Vi / mi
        # viscosityTerm_ij = (viscosityTerms_i + viscosityTerms_j)
        # viscosityTerm_ij = -(pi_i + pi_j) * gradw_ij * Vi * Vj / mi * mu_ij

        viscosityAccel_ij[jj] = viscosityTerm_ij
        pressureAccel_ij[jj] = pressureTerm_ij
        out += pressureTerm_ij + viscosityTerm_ij
    return out



@wp.func
def computeCrkSPHAccel_Func_Adjacency(
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
    
    queryVelocities: wp.array(dtype = vector(length=Any, dtype=scalar_t)), referenceVelocities: wp.array(dtype = vector(length=Any, dtype=scalar_t)), # type: ignore
    queryEnergies: wp.array(dtype = scalar_t), referenceEnergies: wp.array(dtype = scalar_t), # type: ignore
    individual_cs: wp.bool, queryCs: wp.array(dtype = scalar_t), referenceCs: wp.array(dtype = scalar_t), # type: ignore
    viscositySwitch: wp.bool, queryAlphas: wp.array(dtype = scalar_t), referenceAlphas: wp.array(dtype = scalar_t), # type: ignore
    explicitPressure: wp.bool, queryPressures: wp.array(dtype = scalar_t), referencePressures: wp.array(dtype = scalar_t), # type: ignore
    viscosityParams: DiffusionParameters,
    crkViscosityParams: CRKViscosity,

    queryVelocityTensor: wp.array(dtype = matrix(shape=(Any, Any), dtype=scalar_t)), referenceVelocityTensor: wp.array(dtype = matrix(shape=(Any, Any), dtype=scalar_t)),# type: ignore
    accel: wp.array(dtype = vector(length=Any, dtype=scalar_t)), # type: ignore
    pressureAccel_ij: wp.array(dtype = vector(length=Any, dtype=scalar_t)), # type: ignore
    viscosityAccel_ij: wp.array(dtype = vector(length=Any, dtype=scalar_t)) # type: ignore
):
    xi, hi, mi, rhoi, ki = getParticle(queryState, i)
    if kernelProperties.operationMode != wp.static(OperationDirection.TrueAllToToAll.value):
        if not checkDirectionality_i(ki, kernelProperties.operationMode):
            return zero_like_warp(accel[i])

    useGradientRenormalization, Li = getL_i(correctionData, i)
    useGradHTerms, omega_i = getGradH_i(correctionData, i)
    useVolume, Vi = getVolume_i(correctionData, i)
    useCRK, Ai, Bi, gradA_i, gradB_i = getCRK_i(correctionData, i)
    vel_i = queryVelocities[i]

    cs_i = access_optional(queryCs, i, individual_cs, viscosityParams.c_s)
    alpha_i = queryAlphas[i] if viscositySwitch else scalar_t(1.0)
    u_i = queryEnergies[i]
    P_i = access_optional(queryPressures, i, explicitPressure, scalar_t(0.0))
    gradV_i = queryVelocityTensor[i]

    out = zero_like_warp(accel[i])
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
        out += computeCrkSPHAccel_Func_i(
            i, dim, 
            xi, hi, mi, rhoi,
            referenceState,  correctionData, domainState,
            kernelProperties,

            beginIndex, numIndices, adjacencyState.neighborList if useAdjacency else gridState.sortIndex,
            ki, referenceState.kinds,

            useGradientRenormalization, Li,
            useGradHTerms, omega_i,
            useVolume, Vi , 
            useCRK, Ai, Bi, gradA_i, gradB_i,
            vel_i, referenceVelocities,
            u_i, referenceEnergies,
            individual_cs, cs_i, referenceCs,
            viscositySwitch, alpha_i, referenceAlphas,
            explicitPressure, P_i, referencePressures,
            viscosityParams,
            crkViscosityParams,
            gradV_i, referenceVelocityTensor,

            pressureAccel_ij, viscosityAccel_ij
        )
    return out



@wp.kernel
def computeCrkSPHAccel_Kernel(
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
    crkViscosityParams: CRKViscosity,

    queryVelocityTensor: wp.array(dtype = matrix(shape=(Any, Any), dtype=scalar_t)), referenceVelocityTensor: wp.array(dtype = matrix(shape=(Any, Any), dtype=scalar_t)),# type: ignore
    # The last parameter is always the output array and should not be changed
    accel : wp.array(dtype = vector(length=Any, dtype=scalar_t)), # type: ignore
    pressureAccel_ij: wp.array(dtype = vector(length=Any, dtype=scalar_t)), # type: ignore
    viscosityAccel_ij: wp.array(dtype = vector(length=Any, dtype=scalar_t)), # type: ignore
):
    i = wp.tid()
    numParticles = queryState.positions.shape[0]
    if i >= numParticles:
        return

    accel[i] = computeCrkSPHAccel_Func_Adjacency(
        i, domainState.dim, 0, 1,
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
        crkViscosityParams,
        queryVelocityTensor, referenceVelocityTensor,
        accel, pressureAccel_ij, viscosityAccel_ij
    )

@wp.kernel
def computeCrkSPHAccel_KernelTiled(
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
    crkViscosityParams: CRKViscosity,

    queryVelocityTensor: wp.array(dtype = matrix(shape=(Any, Any), dtype=scalar_t)), referenceVelocityTensor: wp.array(dtype = matrix(shape=(Any, Any), dtype=scalar_t)),# type: ignore
    # The last parameter is always the output array and should not be changed
    accel : wp.array(dtype = vector(length=Any, dtype=scalar_t)), # type: ignore
    pressureAccel_ij: wp.array(dtype = vector(length=Any, dtype=scalar_t)), # type: ignore
    viscosityAccel_ij: wp.array(dtype = vector(length=Any, dtype=scalar_t)), # type: ignore
):
    # Multi-lane variant of computeCrkSPHAccel_Kernel (warpSPHCore autograd/lanes.py):
    # launched dim=[N, lanes]; each lane walks a slice of i's neighbours.

    i, lane = wp.tid()
    partial = computeCrkSPHAccel_Func_Adjacency(
        i, domainState.dim, lane, wp.block_dim(),
        queryState, referenceState, correctionData, domainState,
        useAdjacency, adjacencyState, gridState, gridState.numOffsets if not useAdjacency else 1,
        kernelProperties,
        queryVelocities, referenceVelocities,
        queryEnergies, referenceEnergies,
        individual_cs, queryCs, referenceCs,
        viscositySwitch, queryAlphas, referenceAlphas,
        explicitPressure, queryPressures, referencePressures,
        viscosityParams,
        crkViscosityParams,
        queryVelocityTensor, referenceVelocityTensor,
        accel, pressureAccel_ij, viscosityAccel_ij
    )
    total = laneSum(partial)
    if lane == 0:
        accel[i] = total


def _crkAccelPairShape(ctx, extras):
    return ctx.adjacency.i.shape[0]


def _crkAccelDtype(ctx, extras):
    return castTorchToWarpAsBuiltins(ctx.query.positions).dtype


_CRK_ACCEL = OperatorSpec(
    kernel=computeCrkSPHAccel_Kernel,
    tiledKernel=computeCrkSPHAccel_KernelTiled,
    outputs=(
        OutputSpec(dtype=_crkAccelDtype, shape=ShapeOf.QUERY),
        OutputSpec(dtype=_crkAccelDtype, shape=_crkAccelPairShape),
        OutputSpec(dtype=_crkAccelDtype, shape=_crkAccelPairShape),
    ),
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
        ExtraSpec("crkViscosityParams", ExtraKind.SCALAR),
        ExtraSpec("queryVelocityTensor", ExtraKind.TENSOR),
        ExtraSpec("referenceVelocityTensor", ExtraKind.TENSOR),
    ),
)


def computeCrkSPHAccelWarp(
    queryParticles: ParticleState,
    operationProperties: OperationProperties,
    domain: DomainDescription,
    
    conductivityParams: DiffusionParameters,
    crkViscosityParams: CRKViscosity,
    queryVelocityTensor: torch.Tensor, referenceVelocityTensor: Optional[torch.Tensor] = None,
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
):
    if adjacency is None or isinstance(adjacency, CompactHashMap):
        raise ValueError("This module requires an adjacency structure.")
    if referenceVelocities is None:
        referenceVelocities = queryVelocities
    if referenceVelocityTensor is None:
        referenceVelocityTensor = queryVelocityTensor
    if referenceCs is None:
        referenceCs = queryCs
    if referenceAlphas is None:
        referenceAlphas = queryAlphas
    if referencePressures is None:
        referencePressures = queryPressures
    with record_function("warpSPH[computeCrkSPHAccel]"):
        with record_function("warpSPH[computeCrkSPHAccel] - Preprocessing"):
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
            if queryVolumes is None:
                raise ValueError("Volumes must be provided either through queryVolumes or as a property of queryParticles.")
            if crkState is None:
                raise ValueError("CRKState must be provided for CRKSPH computations.")
            if queryVelocities_ is None:
                raise ValueError("Velocities must be provided either through queryVelocities or as a property of queryParticles.")
            if queryEnergies_ is None:
                raise ValueError("Energies must be provided either through queryEnergies or as a property of queryParticles.")

        with record_function("warpSPH[computeCrkSPHAccel] - Kernel Execution"):
            ctx = SPHContext(
                query=queryParticles, properties=operationProperties, domain=domain,
                adjacency=adjacency, reference=referenceParticles,
                corrections=Corrections(
                    volumes=(queryVolumes, referenceVolumes),
                    crk=crkState, gradH=gradHState, renorm=renormalizationState,
                ),
            )
            return launchOperator(
                _CRK_ACCEL, ctx,
                queryVelocities=queryVelocities_, referenceVelocities=referenceVelocities_,
                queryEnergies=queryEnergies_, referenceEnergies=referenceEnergies_,
                individualCs=individual_cs, queryCs=queryCs_, referenceCs=referenceCs_,
                viscositySwitch=viscositySwitch, queryAlphas=queryAlphas_, referenceAlphas=referenceAlphas_,
                explicitPressure=explicitPressure, queryPressures=queryPressures_, referencePressures=referencePressures_,
                conductivityParams=conductivityParams,
                crkViscosityParams=crkViscosityParams,
                queryVelocityTensor=queryVelocityTensor, referenceVelocityTensor=referenceVelocityTensor,
            )


        # with record_function("warpSPH[CRKVolume] - Kernel Execution"):
        #     warp_result = warpWrapper(
        #         launch_kernel, computeCrkSPHAccel_Kernel, outputSize, outputDtype,
        #         *args,
        #         queryVelocities_, referenceVelocities_,
        #         queryEnergies_, referenceEnergies_,
        #         individual_cs, queryCs_, referenceCs_,
        #         viscositySwitch, queryAlphas_, referenceAlphas_,
        #         explicitPressure, queryPressures_, referencePressures_,
        #         conductivityParams
        #     )

    return warp_result
