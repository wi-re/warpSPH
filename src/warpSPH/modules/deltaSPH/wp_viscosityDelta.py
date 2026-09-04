"""Per-particle velocity-dissipation kernel for delta-SPH, plus
`alphaToNu`/`nuToAlpha` conversions between its two dissipation
parameterizations.

`inviscid=True` (the default) gives a Monaghan-style artificial-viscosity
term with coefficient `alpha * c_s * h_i / kernelXi`; `inviscid=False` gives
a physical-viscosity Laplacian term with coefficient `2*(dim+2) * nu`. **Not**
a real Morris et al. 1997 laminar term despite the coefficient's shape: both
branches are gated by the same "approach only" switch
`mu_ij = dot(vel_ij, x_ij) / (|x_ij|^2 + eps*h_i^2)`, clamped to `<= 0` so the
term only acts when neighbors are approaching (`mu_ij > 0` — receding — is
zeroed); the contribution is `apparentVolume * mu_ij * factor * gradW_ij /
mean(rho_i, rho_j)`. That clamp is an artificial-viscosity device (it is what
makes the `inviscid=True` branch stable), and it normal-projects the
`inviscid=False` branch too — a real Morris term needs the full `v_ij` vector
with no such clamp, i.e. real tangential/shear stress, which this branch does
not carry (DFSPH_IMPROVEMENT_PLAN.md "What's realistically open" item 1: the
shear-carrying Morris term is a separate, not-yet-added option). A `1e-14 *
h_i^2` epsilon guards the `mu_ij` denominator at zero separation. Contains a
commented-out alternative symmetrization factor for the physical-viscosity
branch, left disabled.
`alphaToNu`/`nuToAlpha` convert between the two coefficients (`alpha` given
`c_s`, `h`, `n=dim`, and vice versa) so a case can be configured in either
unit system.
"""

import warp as wp
from warp.types import vector, matrix
from typing import Any
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from typing import Optional, Union, Tuple
from warpSPHCore import *

__all__ = ['computeVelocityDiffusionDeltaSPH', 'nuToAlpha', 'alphaToNu']


from ...enumTypes import *

@wp.func
def computeVelocityDiffusionDeltaSPH_Func_i(
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
    
    vel_i: vector(length=Any, dtype=scalar_t), # type: ignore
    referenceVelocities: wp.array(dtype = vector(length=Any, dtype=scalar_t)), # type: ignore

    inviscid: wp.bool, alpha: scalar_t, c_s: scalar_t, nu: scalar_t, n: wp.int32,

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

        apparentVolume = mj / rhoj if not useVolume else referenceVolumes[j]

        gradw_ij = computeKernelGradientCRK(
            xi, xj, 
            hi, hj,
            kernelProperties, domainState,
            useCRK, Ai, Bi, gradAi, gradBi
        )
        if useGradientRenormalization:
            gradw_ij = matmul(Li, gradw_ij)

        
        x_ij = computeDistanceVec(xi, xj, domainState)
        r_ij = safe_sqrt(wp.dot(x_ij, x_ij))
        n_ij = x_ij / (r_ij + scalar_t(1.0e-14) * hi)

        kernelXi = sphKernel_xi(kernelProperties.kernelFunction, n)
        factor = scalar_t(0.0)
        if inviscid:
            factor = alpha * c_s * hi / kernelXi
        else:
            factor = scalar_t(2 * (n + 2)) * nu
            # factor = factor * scalar_t(2.0) # account for rhoi + rhoj in the denominator of the symmetrization term
            # factor = scalar_t(2 * (n + 2)) * factor
            # factor = factor * scalar_t(2.0) * rhoi  * rhoj / (rhoi + rhoj) # This is the standard SPH symmetrization for viscosity to ensure momentum conservation and stability.

        vel_ij = vel_i - referenceVelocities[j]

        mu_ij = wp.dot(vel_ij, x_ij) / ((wp.dot(x_ij, x_ij)) + scalar_t(1.0e-14) * hi*hi)
        if mu_ij > 0:
            mu_ij = scalar_t(0.0)


        out += apparentVolume * mu_ij * factor * gradw_ij / ( (rhoj + rhoi) / scalar_t(2.0)) # * hi
        
    return out

def alphaToNu(
    alpha: float, c_s: float, h: float, n: int
):
    return alpha * c_s * h / (2 * (n + 2))
def nuToAlpha(
    nu: float, c_s: float, h: float, n: int
):
    return nu * (2 * (n + 2)) / (c_s * h)



@wp.func
def computeVelocityDiffusionDeltaSPH_Func_Adjacency(
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
    inviscid: wp.bool, alpha: scalar_t, c_s: scalar_t, nu: scalar_t, n: wp.int32,
    
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
    
    v_i = queryVelocities[i]

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
        
        out += computeVelocityDiffusionDeltaSPH_Func_i(
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
            
            v_i, referenceVelocities,
            inviscid, alpha, c_s, nu, n,


            outputValue,

            # Viscosity function parameters
        )
    return out



@wp.kernel
def computeVelocityDiffusionDeltaSPH_Kernel(
    queryState: Any,
    referenceState: Any,
    domainState: domainData,

    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any,
    
    kernelProperties: kernelState,
    # Do not change the parameters above
    queryVelocities: wp.array(dtype = vector(length=Any, dtype=scalar_t)), referenceVelocities: wp.array(dtype = vector(length=Any, dtype=scalar_t)), # type: ignore
    inviscid: wp.bool, alpha: scalar_t, c_s: scalar_t, nu: scalar_t, dim: wp.int32,

    # The last parameter is always the output array and should not be changed
    outputValues : wp.array(dtype = vector(length=Any, dtype=scalar_t)) # type: ignore
):                                                                                    
    i = wp.tid()
    numParticles = queryState.positions.shape[0]
    if i >= numParticles:
        return

    outputValues[i] = computeVelocityDiffusionDeltaSPH_Func_Adjacency(
        i, domainState.dim, 
        queryState, referenceState, correctionData, domainState,
        useAdjacency, adjacencyState, gridState, gridState.numOffsets if not useAdjacency else 1,
        kernelProperties,  #queryKinds, referenceKinds,
        # The parameters above are default parameters and shold not be changed
        queryVelocities, referenceVelocities,
        inviscid, alpha, c_s, nu, dim,


        zero_like_warp(outputValues)
    )


def _velocityDiffusionDtype(ctx, extras):
    return castTorchToWarpAsBuiltins(ctx.query.velocities).dtype


_VELOCITY_DIFFUSION_DELTA_SPH = OperatorSpec(
    kernel=computeVelocityDiffusionDeltaSPH_Kernel,
    outputs=(OutputSpec(dtype=_velocityDiffusionDtype, shape=ShapeOf.QUERY),),
    extras=(
        ExtraSpec("queryVelocities", ExtraKind.TENSOR),
        ExtraSpec("referenceVelocities", ExtraKind.TENSOR),
        ExtraSpec("inviscid", ExtraKind.SCALAR),
        ExtraSpec("alpha", ExtraKind.SCALAR),
        ExtraSpec("c_s", ExtraKind.SCALAR),
        ExtraSpec("nu", ExtraKind.SCALAR),
        ExtraSpec("dim", ExtraKind.SCALAR),
    ),
)


def computeVelocityDiffusionDeltaSPH(
    queryParticles: ParticleState,
    operationProperties: OperationProperties,
    domain: DomainDescription,
    
    inviscid: bool = True,
    alpha: float = 0.01,
    c_s: float = 1.0,
    nu: float = 1e-3,

    queryVelocities: Optional[torch.Tensor] = None, referenceVelocities: Optional[torch.Tensor] = None,

    queryVolumes: Optional[torch.Tensor] = None, referenceVolumes: Optional[torch.Tensor] = None,
    adjacency: Optional[Union[AdjacencyList, CompactHashMap]] = None, # if none a datastructure is created for EVERY operation!,
    referenceParticles: Optional[ParticleState] = None,
    crkState: Optional[CRKState] = None,
    gradHState: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], GradHState]] = None,
    renormalizationState: Optional[Union[torch.Tensor,RenormalizationState]] = None,
):
    if referenceVelocities is None:
        referenceVelocities = queryVelocities

    with record_function("warpSPH[computeVelocityDiffusion]"):
        with record_function("warpSPH[computeVelocityDiffusion] - Preprocessing"):
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

        with record_function("warpSPH[computeVelocityDiffusionDeltaSPH] - Kernel Execution"):
            ctx = SPHContext(
                query=queryParticles, properties=operationProperties, domain=domain,
                adjacency=adjacency, reference=referenceParticles,
                corrections=Corrections(
                    volumes=(queryVolumes, referenceVolumes),
                    crk=crkState, gradH=gradHState, renorm=renormalizationState,
                ),
            )
            return launchOperator(
                _VELOCITY_DIFFUSION_DELTA_SPH, ctx,
                queryVelocities=queryVelocities, referenceVelocities=referenceVelocities,
                inviscid=wp.bool(inviscid), alpha=scalar_t(alpha), c_s=scalar_t(c_s), nu=scalar_t(nu),
                dim=wp.int32(domain.dim),
            )


        # with record_function("warpSPH[CRKVolume] - Kernel Execution"):
        #     warp_result = warpWrapper(
        #         launch_kernel, computeVelocityDiffusionDeltaSPH_Kernel, outputSize, outputDtype,
        #         *args,
        #         queryVelocities_, referenceVelocities_,
        #         queryEnergies_, referenceEnergies_,
        #         individual_cs, queryCs_, referenceCs_,
        #         viscositySwitch, queryAlphas_, referenceAlphas_,
        #         explicitPressure, queryPressures_, referencePressures_,
        #         conductivityParams
        #     )

    return warp_result
