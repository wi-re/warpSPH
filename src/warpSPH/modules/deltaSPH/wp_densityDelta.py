"""Per-particle delta-SPH density-diffusion kernel: for each neighbor pair,
builds a Fick's-law-style flux `psi_ij` from the inter-particle density
difference and (depending on `densityScheme`) a renormalized or
unrenormalized density gradient, then accumulates `apparentVolume *
dot(psi_ij, gradW_ij)` as an unscaled divergence — the delta/c_s/h prefactor
is applied by the caller (`densityDiffusion.py`).

`densityScheme` (a `DensityDiffusionScheme` int) selects the flux:
- `deltaSPH`: `psi_ij = -(gradRhoL_i + gradRhoL_j) - 2*(rho_j-rho_i)*n_ij/r_ij`
  — renormalized gradients combined with the density-difference term.
- `denormalized`: the same combination but with the unrenormalized `gradRho`.
- `densityOnly`: just `-2*(rho_j-rho_i)*n_ij/r_ij` (no gradient term).
- `deltaOnly` / `denormalizedOnly`: just the (negated) renormalized/
  unrenormalized gradient sum (no density-difference term).

**On the relative sign of the two terms** (fixed 2026-09-05; it was `+grad -
rho` before, i.e. the gradient term entered with the wrong sign). Marrone et
al. 2011 Eq. (6) is `psi_ij = 2(rho_j - rho_i) r_ji/|r_ij|^2 - (<grad rho>^L_i
+ <grad rho>^L_j)` with `r_ji = r_j - r_i = -x_ij`, i.e. exactly `-rho_ij -
grad_ij` in this function's variables. The defining property is that the two
terms **cancel for a field that is linear in space**: with `f = a.x`,
`grad_ij.gradW = 2 a.gradW` and `rho_ij.gradW = -2 a.gradW` identically, so
`psi.gradW == 0` pair by pair. That is the whole point of the Antuono
correction (Antuono et al. 2010/2012) -- it promotes the plain
Molteni-Colagrossi Laplacian to a *bi*-Laplacian, which is what lets the
diffusive term reach a free surface without eating the hydrostatic gradient.
With the terms added instead of cancelled the operator degenerates to twice
the uncorrected Laplacian on any smooth field: still diffusive, so nothing blew
up, but second-order rather than fourth and actively diffusing exactly the
gradients it is supposed to leave alone.
`tests/test_deltaSPHDiffusion.py` pins both the linear (`~1e-13`) and the
quadratic (a bi-Laplacian annihilates those too) cancellation.
All branches guard the `x_ij` normalization and the density-difference
term's `1/r_ij` with a `1e-14 * h_i` epsilon to avoid division by zero at
zero separation. `computeDensityDiffusionDeltaSPH` is the public torch/warp
bridge; `_Func_i`/`_Func_Adjacency`/`_Kernel` are its warp-side
implementation and are not meant to be called directly.

**The diffused field is not necessarily the density.** Pass
`queryField`/`referenceField` (a scalar pair, both or neither) and the
`(f_j - f_i)` difference term reads those instead of `rho_j - rho_i`; the
`gradRho`/`gradRhoL` arguments are then that field's gradients. The volume
weight `m_j/rho_j` is untouched -- it is a quadrature weight, not part of the
diffused quantity. Nothing else in the operator is density-specific, so this
turns the same kernel into a general renormalized scalar Laplacian:

  - `densityOnly`     -> Molteni-Colagrossi, De Courcy et al. 2024 Eq. (32)
                         ("AC-2"), the plain pressure Laplacian.
  - `deltaSPH`        -> Antuono-corrected bi-Laplacian, De Courcy Eq. (33)
                         ("AC-2L"), the paper's working default.

On Eq. (33) specifically (ACSPH_PLAN.md Part 3): the paper writes the
*projected* form, contracting `(grad p_i + grad p_j)` onto `x_ij/|x_ij|`
before dotting with `gradW`, where this kernel uses the *unprojected* Marrone
et al. 2011 form. The two are algebraically identical whenever
`gradW_ij || x_ij` -- true for any isotropic kernel, since
`((g_i+g_j).xhat)(xhat.gradW) = W'(r) (g_i+g_j).x_ij / r = (g_i+g_j).gradW`.
They diverge only if `useGradientRenormalization` is on, which makes
`L gradW` no longer parallel to `x_ij`; `scripts/probe_deltaSPHPsiProjection.py`
measures both claims.
"""

import warp as wp
from warp.types import vector, matrix
from typing import Any
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from typing import Optional, Union, Tuple
from warpSPHCore import *

__all__ = ['computeDensityDiffusionDeltaSPH']


from ...enumTypes import *

@wp.func
def computeDensityDiffusionDeltaSPH_Func_i(
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
    
    gradRho_i: vector(length=Any, dtype=scalar_t), referenceGradRho: wp.array(dtype = vector(length=Any, dtype=scalar_t)), # type: ignore
    gradRhoL_i: vector(length=Any, dtype=scalar_t), referenceGradRhoL: wp.array(dtype = vector(length=Any, dtype=scalar_t)), # type: ignore
    # The diffused scalar field. `useField == 0` reads the state's density (the
    # delta-SPH case); otherwise these replace it -- see the module docstring.
    useField: wp.int32, field_i: scalar_t, referenceField: wp.array(dtype = scalar_t), # type: ignore
    densityScheme: wp.int32,
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

        # The diffused scalar. Density unless a field pair was supplied; the
        # volume weight above stays `m_j/rho_j` either way.
        f_i = rhoi
        f_j = rhoj
        if useField != wp.int32(0):
            f_i = field_i
            f_j = referenceField[j]


        # grad_ij = zero_like_warp(gradw_ij)
        # rho_ij = scalar_t(0.0)
        psi_ij = zero_like_warp(gradw_ij)
        if densityScheme == wp.int32(DensityDiffusionScheme.deltaSPH.value):
            grad_ij = gradRhoL_i + referenceGradRhoL[j]
            rho_ij = scalar_t(2.0) * (f_j - f_i) * n_ij / (r_ij + scalar_t(1.0e-14) * hi)
            psi_ij = - grad_ij - rho_ij
        elif densityScheme == wp.int32(DensityDiffusionScheme.denormalized.value):
            grad_ij = gradRho_i + referenceGradRho[j]
            rho_ij = scalar_t(2.0) * (f_j - f_i) * n_ij / (r_ij + scalar_t(1.0e-14) * hi)
            psi_ij = - grad_ij - rho_ij
        elif densityScheme == wp.int32(DensityDiffusionScheme.densityOnly.value):
            grad_ij = zero_like_warp(gradw_ij)
            rho_ij = scalar_t(2.0) * (f_j - f_i) * n_ij / (r_ij + scalar_t(1.0e-14) * hi)
            psi_ij = - rho_ij
        elif densityScheme == wp.int32(DensityDiffusionScheme.deltaOnly.value):
            grad_ij = gradRhoL_i + referenceGradRhoL[j]
            rho_ij = zero_like_warp(gradw_ij)
            psi_ij = - grad_ij
        elif densityScheme == wp.int32(DensityDiffusionScheme.denormalizedOnly.value):
            grad_ij = gradRho_i + referenceGradRho[j]
            rho_ij = zero_like_warp(gradw_ij)
            psi_ij = - grad_ij
        
        prod = wp.dot(psi_ij, gradw_ij)



        out += apparentVolume * prod
        
    return out



@wp.func
def computeDensityDiffusionDeltaSPH_Func_Adjacency(
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
    
    queryGradRho: wp.array(dtype = vector(length=Any, dtype=scalar_t)), referenceGradRho: wp.array(dtype = vector(length=Any, dtype=scalar_t)), # type: ignore
    queryGradRhoL: wp.array(dtype = vector(length=Any, dtype=scalar_t)), referenceGradRhoL: wp.array(dtype = vector(length=Any, dtype=scalar_t)), # type: ignore
    useField: wp.int32, queryField: wp.array(dtype = scalar_t), referenceField: wp.array(dtype = scalar_t), # type: ignore
    densityScheme: wp.int32,

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
    
    gradRho_i = queryGradRho[i]
    gradRhoL_i = queryGradRhoL[i]
    field_i = scalar_t(0.0)
    if useField != wp.int32(0):
        field_i = queryField[i]

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
        
        out += computeDensityDiffusionDeltaSPH_Func_i(
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
            gradRho_i, referenceGradRho,
            gradRhoL_i, referenceGradRhoL,
            useField, field_i, referenceField,
            densityScheme,


            outputValue,

            # Viscosity function parameters
        )
    return out



@wp.kernel
def computeDensityDiffusionDeltaSPH_Kernel(
    queryState: Any,
    referenceState: Any,
    domainState: domainData,

    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any,

    kernelProperties: kernelState,
    # Do not change the parameters above
    queryGradRho: wp.array(dtype = vector(length=Any, dtype=scalar_t)), referenceGradRho: wp.array(dtype = vector(length=Any, dtype=scalar_t)), # type: ignore
    queryGradRhoL: wp.array(dtype = vector(length=Any, dtype=scalar_t)), referenceGradRhoL: wp.array(dtype = vector(length=Any, dtype=scalar_t)), # type: ignore
    useField: wp.int32, queryField: wp.array(dtype = scalar_t), referenceField: wp.array(dtype = scalar_t), # type: ignore
    densityScheme: wp.int32,

    # The last parameter is always the output array and should not be changed
    outputValues : wp.array(dtype = scalar_t) # type: ignore
):                                                                                    
    i = wp.tid()
    numParticles = queryState.positions.shape[0]
    if i >= numParticles:
        return

    outputValues[i] = computeDensityDiffusionDeltaSPH_Func_Adjacency(
        i, domainState.dim, 
        queryState, referenceState, correctionData, domainState,
        useAdjacency, adjacencyState, gridState, gridState.numOffsets if not useAdjacency else 1,
        kernelProperties,
        # The parameters above are default parameters and shold not be changed
        queryGradRho, referenceGradRho,
        queryGradRhoL, referenceGradRhoL,
        useField, queryField, referenceField,
        densityScheme,


        zero_like_warp(outputValues)
    )


def _densityDiffusionDtype(ctx, extras):
    return castTorchToWarpAsBuiltins(ctx.query.densities).dtype


_DENSITY_DIFFUSION_DELTA_SPH = OperatorSpec(
    kernel=computeDensityDiffusionDeltaSPH_Kernel,
    outputs=(OutputSpec(dtype=_densityDiffusionDtype, shape=ShapeOf.QUERY),),
    extras=(
        ExtraSpec("queryGradRho", ExtraKind.TENSOR),
        ExtraSpec("referenceGradRho", ExtraKind.TENSOR),
        ExtraSpec("queryGradRhoL", ExtraKind.TENSOR),
        ExtraSpec("referenceGradRhoL", ExtraKind.TENSOR),
        ExtraSpec("useField", ExtraKind.SCALAR),
        ExtraSpec("queryField", ExtraKind.TENSOR),
        ExtraSpec("referenceField", ExtraKind.TENSOR),
        ExtraSpec("densityScheme", ExtraKind.SCALAR),
    ),
)


def computeDensityDiffusionDeltaSPH(
    queryParticles: ParticleState,
    operationProperties: OperationProperties,
    domain: DomainDescription,
    
    densityScheme: DensityDiffusionScheme,

    queryGradRho: Optional[torch.Tensor] = None, referenceGradRho: Optional[torch.Tensor] = None,
    queryGradRhoL: Optional[torch.Tensor] = None, referenceGradRhoL: Optional[torch.Tensor] = None,
    queryField: Optional[torch.Tensor] = None, referenceField: Optional[torch.Tensor] = None,

    queryVolumes: Optional[torch.Tensor] = None, referenceVolumes: Optional[torch.Tensor] = None,
    adjacency: Optional[Union[AdjacencyList, CompactHashMap]] = None, # if none a datastructure is created for EVERY operation!,
    referenceParticles: Optional[ParticleState] = None,
    crkState: Optional[CRKState] = None,
    gradHState: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], GradHState]] = None,
    renormalizationState: Optional[Union[torch.Tensor,RenormalizationState]] = None,
):
    if referenceGradRho is None:
        referenceGradRho = queryGradRho
    if referenceGradRhoL is None:
        referenceGradRhoL = queryGradRhoL
    if referenceField is None:
        referenceField = queryField
    if (queryField is None) != (referenceField is None):
        raise ValueError(
            "queryField and referenceField must be supplied together (or "
            "neither, to diffuse the state's density)")
    useField = queryField is not None

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

            outputSize = queryParticles.positions.shape[0]

            referenceParticles = referenceParticles if referenceParticles is not None else queryParticles

            queryGradRho_ = queryGradRho if queryGradRho is not None else getCachedDummyTensor((outputSize, domain.dim), dtype=get_torch_precision(), device=device)
            queryGradRhoL_ = queryGradRhoL if queryGradRhoL is not None else getCachedDummyTensor((outputSize, domain.dim), dtype=get_torch_precision(), device=device)
            referenceGradRho_ = referenceGradRho if referenceGradRho is not None else getCachedDummyTensor((outputSize, domain.dim), dtype=get_torch_precision(), device=device)
            referenceGradRhoL_ = referenceGradRhoL if referenceGradRhoL is not None else getCachedDummyTensor((outputSize, domain.dim), dtype=get_torch_precision(), device=device)
            # `useField == 0` never indexes these, so a length-`outputSize`
            # dummy is enough -- but it must still be a real array, since warp
            # types the kernel argument regardless.
            queryField_ = queryField if queryField is not None else getCachedDummyTensor((outputSize,), dtype=get_torch_precision(), device=device)
            referenceField_ = referenceField if referenceField is not None else getCachedDummyTensor((referenceParticles.positions.shape[0],), dtype=get_torch_precision(), device=device)

        with record_function("warpSPH[computeDensityDiffusionDeltaSPH] - Kernel Execution"):
            ctx = SPHContext(
                query=queryParticles, properties=operationProperties, domain=domain,
                adjacency=adjacency, reference=referenceParticles,
                corrections=Corrections(
                    volumes=(queryVolumes, referenceVolumes),
                    crk=crkState, gradH=gradHState, renorm=renormalizationState,
                ),
            )
            return launchOperator(
                _DENSITY_DIFFUSION_DELTA_SPH, ctx,
                queryGradRho=queryGradRho_, referenceGradRho=referenceGradRho_,
                queryGradRhoL=queryGradRhoL_, referenceGradRhoL=referenceGradRhoL_,
                useField=wp.int32(1 if useField else 0),
                queryField=queryField_, referenceField=referenceField_,
                densityScheme=wp.int32(densityScheme.value),
            )


        # with record_function("warpSPH[CRKVolume] - Kernel Execution"):
        #     warp_result = warpWrapper(
        #         launch_kernel, computeDensityDiffusionDeltaSPH_Kernel, outputSize, outputDtype,
        #         *args,
        #         queryVelocities_, referenceVelocities_,
        #         queryEnergies_, referenceEnergies_,
        #         individual_cs, queryCs_, referenceCs_,
        #         viscositySwitch, queryAlphas_, referenceAlphas_,
        #         explicitPressure, queryPressures_, referencePressures_,
        #         conductivityParams
        #     )

    return warp_result
