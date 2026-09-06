"""Warp kernel computing the raw (unscaled) delta-SPH particle-shifting term:
`sum_j weight_ij * [1 + R*(w_ij/W_0)^n] * gradW_ij` (Lind et al./Sun et
al.-style anti-clustering correction), evaluated per-particle over an
adjacency list or a compact hash grid. `weight_ij` is
`0.5*m_j/(rho_i+rho_j)` (Sun's mean-density weight, the historical default,
`volumeWeighted=False`) or the plain apparent volume `V_j` (Michel 2022 Eq.
2-3's weighting, `volumeWeighted=True`) -- these are different formulas, not
just a naming difference; pass `R=0.2` to match Eq. 3 exactly when using the
latter. `computeDeltaShiftWarp` is the Python entry point;
`modules.shifting.delta.computeDeltaShift` (which re-exports it) applies the
actual `-CFL * Ma * 2 * h^2` position-delta scaling externally -- by design
this kernel does not apply it, so its `CFL`/`computeMach`/`c_max`/`dx`
arguments are accepted but do not affect the returned value (only `rho0`
feeds the output, via the reference spacing `dx_2`). Also used directly by
`sample.optimal.sampleOptimal` to relax a lattice into a glass-like layout,
and by `modules.shifting.michel` for the Michel 2022 PST's `grad-tilde(C)`.
"""

import warp as wp
from warp.types import vector, matrix
from typing import Any
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from typing import Optional, Union, Tuple
from warpSPHCore import *
from warpSPHCore.kernels.eval_kernel import eval_k, eval_dkdq, eval_C_d

__all__ = ['computeDeltaShiftWarp']


@wp.func
def computeDeltaShift_Func_i(
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
    correctionData: Any, # correctionData_1 or correctionData_2 or correctionData_3, containing all the optional correction terms and their usage flags

    # Dummy value to allow allocation
    outputValue: Any, # type: ignore

    # DeltaShift function parameters begin here
    R: float, n: int, CFL: float, computeMach: bool, c_max: float,
    rho0: float, dx: float, volumeWeighted: bool,

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
        # DeltaShift Functionality begins here        
        w_ij = computeKernelCRK(
            xi, xj, 
            hi, hj, 
            kernelProperties, domainState,
            useCRK, Ai, Bi
        )
        gradw_ij = computeKernelGradientCRK(
            xi, xj, 
            hi, hj,
            kernelProperties, domainState,
            useCRK, Ai, Bi, gradAi, gradBi
        )
        # Alternative crk correction scheme for reference
        # useCRK, Aj, Bj, gradAj, gradBj = getCRK_i(correctionData, i)
        # gradw_ji = computeKernelGradientCRK(
        #     xj, xi, 
        #     hj, hi,
        #     kernelProperties, domainState,
        #     useCRK, Aj, Bj, gradAj, gradBj
        # )
        # gradw_ij = (gradw_ij - gradw_ji) * 0.5
        if useGradientRenormalization:
            gradw_ij = matmul(Li, gradw_ij)

        ### GENERIC CODE STOPS HERE ###

        dx_2 = wp.pow(mj / scalar_t(rho0), scalar_t(1.0) /scalar_t(dim))
        
        dx_ = dx_2 / sphKernelScale(kernelProperties.kernelFunction, dim)
        
        x_ij = computeDistanceVec(xi, xj, domainState)
        r_ij = safe_sqrt(wp.dot(x_ij, x_ij))

        hij = computePairwiseSupport(hi, hj, kernelProperties.supportMode)
        q = dx_ / hij
        W_0 = eval_k(q, dim, kernelProperties.kernelFunction) * eval_C_d(dim, kernelProperties.kernelFunction) / iPow(hij, dim)
        k = w_ij / W_0

        term = scalar_t(scalar_t(1.0) + scalar_t(R) * wp.pow(k, scalar_t(n)))
        densityTerm =scalar_t(0.5) * mj / (rhoi + rhoj)
        weight = scalar_t(apparentVolume) if volumeWeighted else densityTerm

        phi_ij = scalar_t(1.0  )
        scalarTerm = term * weight * phi_ij

        shiftAmount = scalarTerm * gradw_ij


        Ma = scalar_t(0.1)
        if computeMach:
            Ma = scalar_t(c_max)
        h2 = (hi / sphKernelScale(kernelProperties.kernelFunction, dim) * scalar_t(2.0))
        shiftScaling = -scalar_t(CFL) * Ma * h2 #* h2
        
        out += shiftAmount
    return out



@wp.func
def computeDeltaShift_Func_Adjacency(
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

    outputValue : Any, # type: ignore

    R: float, n: int, CFL: float, computeMach: bool, c_max: float,
    rho0: float, dx: float, volumeWeighted: bool,
):
    xi, hi, mi, rhoi, ki = getParticle(queryState, i)
    if kernelProperties.operationMode != wp.static(OperationDirection.TrueAllToToAll.value):
        if not checkDirectionality_i(ki, kernelProperties.operationMode):
            return zero_like_warp(outputValue)
        
    useGradientRenormalization, Li = getL_i(correctionData, i)
    useGradHTerms, omega_i = getGradH_i(correctionData, i)
    useVolume, Vi = getVolume_i(correctionData, i)
    useCRK, Ai, Bi, gradA_i, gradB_i = getCRK_i(correctionData, i)

    out = type(outputValue)() * scalar_t(0.0)
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
        
        out += computeDeltaShift_Func_i(
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
            correctionData,
            

            outputValue,

            # DeltaShift function parameters
            R, n, CFL, computeMach, c_max,
            rho0, dx, volumeWeighted,
        )
    return out



@wp.kernel
def computeDeltaShift_Kernel(
    queryState: Any,
    referenceState: Any,
    domainState: domainData,

    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any,
    
    kernelProperties: kernelState,
    # Do not change the parameters above

    R: float, n: wp.int32, CFL: float, computeMach: wp.bool, c_max: float,
    rho0: float, dx: float, volumeWeighted: wp.bool,

    # The last parameter is always the output array and should not be changed
    outputValues : wp.array(dtype = Any) # type: ignore
):
    i = wp.tid()
    numParticles = queryState.positions.shape[0]
    if i >= numParticles:
        return

    outputValues[i] = computeDeltaShift_Func_Adjacency(
        i, domainState.dim,
        queryState, referenceState, correctionData, domainState,
        useAdjacency, adjacencyState, gridState, gridState.numOffsets if not useAdjacency else 1,
        kernelProperties,  #queryKinds, referenceKinds,
        # The parameters above are default parameters and shold not be changed

        zero_like_warp(outputValues),

        R, n, CFL, computeMach, c_max,
        rho0, dx, volumeWeighted,
    )

def _deltaShiftDtype(ctx, extras):
    return castTorchToWarpAsBuiltins(ctx.query.positions).dtype


_DELTA_SHIFT = OperatorSpec(
    kernel=computeDeltaShift_Kernel,
    outputs=(OutputSpec(dtype=_deltaShiftDtype, shape=ShapeOf.QUERY),),
    extras=(
        ExtraSpec("R", ExtraKind.SCALAR),
        ExtraSpec("n", ExtraKind.SCALAR),
        ExtraSpec("CFL", ExtraKind.SCALAR),
        ExtraSpec("computeMach", ExtraKind.SCALAR),
        ExtraSpec("c_max", ExtraKind.SCALAR),
        ExtraSpec("rho0", ExtraKind.SCALAR),
        ExtraSpec("dx", ExtraKind.SCALAR),
        ExtraSpec("volumeWeighted", ExtraKind.SCALAR),
    ),
)


def computeDeltaShiftWarp(
    queryParticles: ParticleState,
    operationProperties: OperationProperties,
    domain: DomainDescription,

    CFL: float, computeMach: bool, c_max: float,
    rho0: float, dx: float,

    R: float = 0.25, n: int = 4, volumeWeighted: bool = False,

    queryVolumes: Optional[torch.Tensor] = None, referenceVolumes: Optional[torch.Tensor] = None,
    adjacency: Optional[Union[AdjacencyList, CompactHashMap]] = None, # if none a datastructure is created for EVERY operation!,
    referenceParticles: Optional[ParticleState] = None,
    crkState: Optional[CRKState] = None,
    gradHState: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], GradHState]] = None,
    renormalizationState: Optional[Union[torch.Tensor,RenormalizationState]] = None,
):
    with record_function("warpSPH[CRKVolume]"):
        ctx = SPHContext(
            query=queryParticles, properties=operationProperties, domain=domain,
            adjacency=adjacency, reference=referenceParticles,
            corrections=Corrections(
                volumes=(queryVolumes, referenceVolumes),
                crk=crkState, gradH=gradHState, renorm=renormalizationState,
            ),
        )
        return launchOperator(
            _DELTA_SHIFT, ctx,
            R=R, n=n, CFL=CFL, computeMach=computeMach, c_max=c_max,
            rho0=rho0, dx=dx, volumeWeighted=volumeWeighted,
        )

