"""mDBC no-penetration velocity shift.

Computes a per-fluid-particle corrective velocity shift that discourages
penetration through nearby mDBC boundary/ghost surfaces: for each fluid-ghost
pair within a distance/normal-offset threshold and closing on the boundary
(three per-axis conditions on distance, normal offset, and sign of the
relative-velocity/normal dot product), it accumulates a `-factor * dv * norm
* norm` correction term along the ghost's outward normal and averages it over
qualifying neighbor directions. `computeMdbcNoPenShift` is the public
entry point (`BoundaryToFluid` direction); it returns zeros when there are no
boundary particles.

The first ~185 lines of this file are dead, commented-out code from an
earlier `mDBCPenetrationCheck` implementation and are not executed; the live
code begins at `computeMdbcNoPenShift_Func_i` below (separate, already-tracked
cleanup item -- see docs/historic_plans/CLEANUP_PLAN.md -- to remove them).
"""

__all__ = ['computeMdbcNoPenShift']

# def mDBCPenetrationCheck(particles: Union[CompressibleState, WeaklyCompressibleState],
#         kernel: SPHKernel,
#         neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
#         supportScheme: SupportScheme = SupportScheme.Scatter,
#         config: Dict = {},
#         clampDensity: bool = True        
# ):
    
#     neighbors, kernelValues = neighborhood
#     fluidIndices = neighbors.row 
#     boundaryIndices = neighbors.col

#     boundaryNormal = particles.ghostOffsets[boundaryIndices]
#     n_b = torch.linalg.norm(boundaryNormal, dim = 1,)

#     # boundaryNormal = boundaryNormal / (n_b[:,None] + 1e-12)

#     kind_i = particles.kinds[fluidIndices]
#     kind_j = particles.kinds[boundaryIndices]

#     if not torch.all(kind_i == 0):
#         print(f'[mDBC] - Warning: fluidIndices contains non-fluid particles: {kind_i}')
#     if not torch.all(kind_j == 1):
#         print(f'[mDBC] - Warning: boundaryIndices contains non-boundary particles: {kind_j}')   

#     # print(f'n_b stats: min {torch.min(n_b).item()}, max {torch.max(n_b).item()}, mean {torch.mean(n_b).item()}')

#     x_ib = particles.positions[fluidIndices] - particles.positions[boundaryIndices]
#     r_ib = torch.linalg.norm(x_ib, dim = 1)
#     # print(f'r_ib stats: min {torch.min(r_ib).item()}, max {torch.max(r_ib).item()}, mean {torch.mean(r_ib).item()}')

#     dp = config['particle']['dx']

#     # r_ib = rrmag

#     check_a = r_ib < 1.25 * dp

#     norm = torch.linalg.norm(boundaryNormal, dim = 1)
#     normalized = boundaryNormal / (norm[:,None] + 1e-12)
#     normdist = torch.einsum('ni, ni -> n', x_ib, normalized).abs()

#     check_b = torch.logical_and(normdist < 0.75 * norm, norm < 1.75 * dp)
#     # check_b = normdist < 0.75 * norm

#     # print(f'Dot: {torch.einsum("ni, ni -> n", x_ib, boundaryNormal)}')

#     check_c = torch.einsum('ni, ni -> n', particles.velocities[fluidIndices] - particles.velocities[boundaryIndices], boundaryNormal) < 0.0

#     # print(f'[mDBC] - Total checks: {check_a.shape[0]}')
#     # print(f'[mDBC] - Check A (r_ib > 1.25 * n_b): {torch.sum(check_a).item()} [check_a shape: {check_a.shape}]')
#     # print(f'[mDBC] - Check B (dot(x_ib, n_b) > 0.75 * n_b): {torch.sum(check_b).item()} [check_b shape: {check_b.shape}]')
#     # print(f'[mDBC] - Check C (dot(v_i - v_b, n_b) > 0): {torch.sum(check_c).item()} [check_c shape: {check_c.shape}]')

#     check_ab = torch.logical_and(check_a, check_b)
#     check_abc = torch.logical_and(check_ab, check_c)

#     # print(f'[mDBC] - Check AB (A and B): {torch.sum(check_ab).item()}')
#     # print(f'[mDBC] - Check ABC (A and B and C): {torch.sum(check_abc).item()}')

#     # penetrationMask = torch.logical_and(
#     #     (check_a),
#     #     torch.logical_and(
#     #     (check_b), 
#     #     (check_c)
#     #     )
#     # )

#     adjustedVelocities = particles.velocities.clone()[fluidIndices]

#     # print(f'[mDBC] - Penetration corrections: {torch.sum(penetrationMask).item()} / {penetrationMask.shape[0]}')

#     # print(f'Mask shape: {penetrationMask.shape}')
#     # print(f'adjusted velocities shape: {adjustedVelocities.shape}')
#     # print(f'boundary normal shape: {boundaryNormal.shape}')
#     # print(f'r_ib shape: {r_ib.shape}')

#     nopenshift = torch.zeros_like(adjustedVelocities)
#     nopencount = torch.zeros(adjustedVelocities.shape, dtype = torch.int32, device = adjustedVelocities.device)

#     nopen_mask_a = torch.zeros(adjustedVelocities.shape, dtype = torch.int32, device = adjustedVelocities.device)
#     nopen_mask_b = torch.zeros(adjustedVelocities.shape, dtype = torch.int32, device = adjustedVelocities.device)
#     nopen_mask_c = torch.zeros(adjustedVelocities.shape, dtype = torch.int32, device = adjustedVelocities.device)

#     for d in range(particles.positions.shape[1]):
#         u_i = particles.velocities[fluidIndices][:,d]
#         u_b = particles.velocities[boundaryIndices][:,d]

#         # print((r_ib / n_b).shape)

#         norm = -boundaryNormal[:,d]
#         dr = x_ib[:,d]
#         absx = normalized[:,d].abs()

#         mask = check_ab

#         mask_a = torch.logical_and(dr * -normalized[:,d] < 0.75, absx > 0.001 * dp)

#         nopen_mask_a[:,d] = mask_a.int()
#         nopen_mask_b[:,d] = mask.int()

#         mask = torch.logical_and(mask, mask_a)
#         nopen_mask_c[:,d] = mask.int()

#         dv = u_i - u_b
#         vfc = dv * norm

#         mask_b = torch.logical_and(mask, vfc < 0)

#         ratio = torch.clamp((dr / norm).abs(), min = 0.25)
#         # ratio = torch.ones_like(ratio) *0.25
#         factor = - 4 * ratio + 3

#         nopenshiftTerm = -factor * dv * norm * norm
#         if torch.sum(mask_b) == 0:
#             # print(f'[mDBC] - Direction {d}: No penetration corrections applied.')
#             continue
#         # print(f'[mDBC] - Direction {d}: Applying {torch.sum(mask_b).item()} penetration corrections.')
#         # print(f'[mDBC] - Direction {d}: Max correction magnitude: {torch.max(nopenshiftTerm[mask_b].abs()).item():.6f}')
#         # print(f'[mDBC] - Direction {d}: Avg correction magnitude: {torch.mean(nopenshiftTerm[mask_b].abs()).item():.6f}')
#         # print(f'[mDBC] - dv: max {torch.max(dv[mask_b]).item():.6f}, min {torch.min(dv[mask_b]).item():.6f}, mean {torch.mean(dv[mask_b]).item():.6f}')
#         # print(f'[mDBC] - norm: max {torch.max(norm[mask_b]).item():.6f}, min {torch.min(norm[mask_b]).item():.6f}, mean {torch.mean(norm[mask_b]).item():.6f}')
#         # print(f'[mDBC] - ratio: max {torch.max(ratio[mask_b]).item():.6f}, min {torch.min(ratio[mask_b]).item():.6f}, mean {torch.mean(ratio[mask_b]).item():.6f}')
#         # print(f'[mDBC] - factor: max {torch.max(factor[mask_b]).item():.6f}, min {torch.min(factor[mask_b]).item():.6f}, mean {torch.mean(factor[mask_b]).item():.6f}')

#         nopenshift[:,d] += torch.where(
#             mask_b,
#             nopenshiftTerm,
#             torch.zeros_like(nopenshiftTerm)
#         )
#         nopencount[:,d] += mask_b.int()

#         # u_adj_k = - (3 - 4 * torch.clamp(r_ib / n_b, min = 0.25)) * (u_i - u_b) * (boundaryNormal[:,d]**2)

#         # print(f'u_adj_k shape: {u_adj_k.shape}')
#         # print(f'u_i shape: {u_i.shape}')
#         # print(f'u_b shape: {u_b.shape}')

#         # adjustedVelocities[:,d] = torch.where(
#         #     penetrationMask.view(-1),
#         #     u_adj_k,
#         #     adjustedVelocities[:,d]
#         # )
    
#     nopenshift = scatter_sum(nopenshift, fluidIndices, dim = 0, dim_size = particles.positions.shape[0])
#     nopencount = scatter_sum(nopencount, fluidIndices, dim = 0, dim_size = particles.positions.shape[0])

#     nopencounta = scatter_sum(nopen_mask_a.int(), fluidIndices, dim = 0, dim_size = particles.positions.shape[0])
#     nopencountb = scatter_sum(nopen_mask_b.int(), fluidIndices, dim = 0, dim_size = particles.positions.shape[0])
#     nopencountc = scatter_sum(nopen_mask_c.int(), fluidIndices, dim = 0, dim_size = particles.positions.shape[0])

#     avgShift = nopenshift / (nopencount.float() + 1e-12) * 2
#     # print('-' * 40)
#     # print(f'[mDBC] - Total penetration corrections applied: {torch.sum(nopencount > 0).item()} [{torch.sum(nopencounta > 0).item()} / {torch.sum(nopencountb > 0).item()} / {torch.sum(nopencountc > 0).item()}] / {particles.positions.shape[0]}')
#     # print(f'[mDBC] - Average penetration correction magnitude: {torch.mean(torch.linalg.norm(avgShift[nopencount > 0], dim = 0)).item():.6f}')
#     # print(f'[mDBC] - Max penetration correction magnitude: {torch.max(torch.linalg.norm(avgShift[nopencount > 0], dim = 0)).item():.6f}')



#     mergedVelocities = particles.velocities.clone()
#     # adjustedParticles = fluidIndices[penetrationMask]

#     # particles.velocities += avgShift

#     checked = scatter_sum(check_ab.int(), fluidIndices, dim = 0, dim_size = particles.positions.shape[0])

#     # particles.velocities[checked > 0,:] = 0

#     # avgShift[checked>0] = -particles.velocities[checked>0]

#     return avgShift



import warp as wp
from warp.types import vector, matrix
from typing import Any
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from typing import Optional, Union, Tuple
from warpSPHCore import *
from ._util import stateHasBoundaryParticles




@wp.func
def computeMdbcNoPenShift_Func_i(
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
    ghostOffset_i: vector(length=Any, dtype=scalar_t), referenceGhostOffsets: wp.array(dtype = vector(length=Any, dtype=scalar_t)), # type: ignore
    outCtr: Any
):
    # Initialize the output value
    out     = zero_like_warp(xi)
    outCounter = zero_like_warp(outCtr)
    
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
        apparentVolume = mj / rhoj if not useVolume else referenceVolumes[j]

        gradw_ij = computeKernelGradientCRK(
            xi, xj, 
            hi, hj,
            kernelProperties, domainState,
            useCRK, Ai, Bi, gradAi, gradBi
        )

        w_ij = computeKernelCRK(
            xi, xj, 
            hi, hj,
            kernelProperties, domainState,
            useCRK, Ai, Bi
        )
        if useGradientRenormalization:
            gradw_ij = matmul(Li, gradw_ij)

        
        x_ij = computeDistanceVec(xi, xj, domainState)
        r_ij = safe_sqrt(wp.dot(x_ij, x_ij))
        
        # The ghost offset points in the opposite direction of the normal, so we negate it to get the outward normal direction
        offset_j = referenceGhostOffsets[j]
        norm_j = (safe_sqrt(wp.dot(offset_j, offset_j)) + scalar_t(1e-12))
        normal_j = -offset_j / norm_j
        normDist = wp.dot(x_ij, normal_j)

        dp_i = mi**(scalar_t(1.0) / scalar_t(dim))

        tempOut = zero_like_warp(xi)
        tempCtr = zero_like_warp(outCtr)
        if w_ij > scalar_t(0.0):
            condition_a = r_ij < scalar_t(1.25) * dp_i
            condition_b = wp.abs(normDist) < scalar_t(0.75) * norm_j and norm_j < scalar_t(1.75) * dp_i
            condition_c = wp.dot(vel_i - vel_j, normal_j) < 0

            condition_ab = condition_a and condition_b
            condition_abc = condition_ab and condition_c 
            
            for d in range(dim):
                u_i = vel_i[d]
                u_j = vel_j[d]
                dv = u_i - u_j
                norm = normal_j[d]
                dr = x_ij[d]
                
                mask = condition_ab and (dr * normal_j[d] < scalar_t(0.75)) and (wp.abs(normal_j[d]) > scalar_t(0.001) * dp_i)

                vfc = dv * norm
                ratio = wp.clamp(wp.abs((dr / norm)), scalar_t(0.25), scalar_t(1.0))
                factor = - scalar_t(4.0) * ratio + scalar_t(3.0)

                nopenshiftTerm = -factor * dv * norm * norm
                if mask and vfc < 0:
                    tempOut[d] = nopenshiftTerm
                    tempCtr[d] = 1

            out += tempOut 
            outCounter += tempCtr
        
    return out, outCounter
from warpSPHCore import *

@wp.func
def computeMdbcNoPenShift_Func_Adjacency(
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
    queryOffsets: wp.array(dtype = vector(length=Any, dtype=scalar_t)), referenceOffsets: wp.array(dtype = vector(length=Any, dtype=scalar_t)), # type: ignore
    outCtr: Any
):
    xi, hi, mi, rhoi, ki = getParticle(queryState, i)
    if kernelProperties.operationMode != wp.static(OperationDirection.TrueAllToToAll.value):
        if not checkDirectionality_i(ki, kernelProperties.operationMode):
            return zero_like_warp(queryState.positions), zero_like_warp(outCtr)
        
    useGradientRenormalization, Li = getL_i(correctionData, i)
    useGradHTerms, omega_i = getGradH_i(correctionData, i)
    useVolume, Vi = getVolume_i(correctionData, i)
    useCRK, Ai, Bi, gradA_i, gradB_i = getCRK_i(correctionData, i)
    vel_i = queryVelocities[i]
    offset_i = queryOffsets[i]

    out = zero_like_warp(queryState.positions)
    outCounter = zero_like_warp(outCtr)

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
        
        ret_out, ret_ctr = computeMdbcNoPenShift_Func_i(
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
            offset_i, referenceOffsets,

            outCtr
        )
        out += ret_out
        outCounter += ret_ctr
    return out, outCounter



@wp.kernel
def computeMdbcNoPenShift_Kernel(
    queryState: Any,
    referenceState: Any,
    domainState: domainData,

    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any,
    
    kernelProperties: kernelState,
    # Do not change the parameters above
    queryVelocities: wp.array(dtype = vector(length=Any, dtype=scalar_t)), referenceVelocities: wp.array(dtype = vector(length=Any, dtype=scalar_t)), # type: ignore
    queryOffsets: wp.array(dtype = vector(length=Any, dtype=scalar_t)), referenceOffsets: wp.array(dtype = vector(length=Any, dtype=scalar_t)), # type: ignore
    # The last parameter is always the output array and should not be changed
    outputValues : wp.array(dtype = Any), # type: ignore
    outputCounters : wp.array(dtype = Any) # type: ignore
):                                                                                    
    i = wp.tid()
    numParticles = queryState.positions.shape[0]
    if i >= numParticles:
        return

    ret_out, ret_ctr = computeMdbcNoPenShift_Func_Adjacency(
        i, domainState.dim, 
        queryState, referenceState, correctionData, domainState,
        useAdjacency, adjacencyState, gridState, gridState.numOffsets if not useAdjacency else 1,
        kernelProperties,  #queryKinds, referenceKinds,
        # The parameters above are default parameters and shold not be changed
        queryVelocities, referenceVelocities,
        queryOffsets, referenceOffsets,

        outputCounters
    )
    avg_out = zero_like_warp(ret_out)
    for d in range(domainState.dim):
        avg_out[d] = ret_out[d] / (scalar_t(ret_ctr[d]) + scalar_t(1e-12))

    outputValues[i] = avg_out
    outputCounters[i] = ret_ctr

def _mdbcShiftDtype(ctx, extras):
    return castTorchToWarpAsBuiltins(ctx.query.positions).dtype


def _mdbcCounterDtype(ctx, extras):
    return vector(length=ctx.query.positions.shape[1], dtype=wp.int32)


_MDBC_NOPEN_SHIFT = OperatorSpec(
    kernel=computeMdbcNoPenShift_Kernel,
    outputs=(
        OutputSpec(dtype=_mdbcShiftDtype, shape=ShapeOf.QUERY),
        OutputSpec(dtype=_mdbcCounterDtype, shape=ShapeOf.QUERY),
    ),
    extras=(
        ExtraSpec("queryVelocities", ExtraKind.TENSOR),
        ExtraSpec("referenceVelocities", ExtraKind.TENSOR),
        ExtraSpec("queryOffsets", ExtraKind.TENSOR),
        ExtraSpec("referenceOffsets", ExtraKind.TENSOR),
    ),
)


def computeMdbcNoPenShiftWarp(
    queryParticles: ParticleState,
    operationProperties: OperationProperties,
    domain: DomainDescription,
    
    queryVelocities : Optional[torch.Tensor] = None, referenceVelocities: Optional[torch.Tensor] = None,
    queryOffsets: Optional[torch.Tensor] = None, referenceOffsets: Optional[torch.Tensor] = None,

    queryVolumes: Optional[torch.Tensor] = None, referenceVolumes: Optional[torch.Tensor] = None,
    adjacency: Optional[Union[AdjacencyList, CompactHashMap]] = None, # if none a datastructure is created for EVERY operation!,
    referenceParticles: Optional[ParticleState] = None,
    crkState: Optional[CRKState] = None,
    gradHState: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], GradHState]] = None,
    renormalizationState: Optional[Union[torch.Tensor,RenormalizationState]] = None,
):
    if referenceVelocities is None:
        referenceVelocities = queryVelocities
    if referenceOffsets is None:
        referenceOffsets = queryOffsets
    with record_function("warpSPH[computeMdbcNoPenShift]"):
        with record_function("warpSPH[computeMdbcNoPenShift] - Preprocessing"):
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

        with record_function("warpSPH[computeMdbcNoPenShift] - Kernel Execution"):
            ctx = SPHContext(
                query=queryParticles, properties=operationProperties, domain=domain,
                adjacency=adjacency, reference=referenceParticles,
                corrections=Corrections(
                    volumes=(queryVolumes, referenceVolumes),
                    crk=crkState, gradH=gradHState, renorm=renormalizationState,
                ),
            )
            return launchOperator(
                _MDBC_NOPEN_SHIFT, ctx,
                queryVelocities=queryVelocities, referenceVelocities=referenceVelocities,
                queryOffsets=queryOffsets, referenceOffsets=referenceOffsets,
            )


        # with record_function("warpSPH[CRKVolume] - Kernel Execution"):
        #     warp_result = warpWrapper(
        #         launch_kernel, computeMdbcNoPenShift_Kernel, outputSize, outputDtype,
        #         *args,
        #         queryVelocities_, referenceVelocities_,
        #         queryEnergies_, referenceEnergies_,
        #         individual_cs, queryCs_, referenceCs_,
        #         viscositySwitch, queryAlphas_, referenceAlphas_,
        #         explicitPressure, queryPressures_, referencePressures_,
        #         conductivityParams
        #     )


def computeMdbcNoPenShift(currentState: Any, config: Any, schemeConfig: Any, adjacency: Optional[Union[AdjacencyList, CompactHashMap]]) -> Tuple[torch.Tensor, torch.Tensor]:
    if not stateHasBoundaryParticles(currentState, config):
        # No boundary particles, return zero shift
        return torch.zeros_like(currentState.velocities)
    with record_function("warpSPH - (mdbc) - computeMdbcNoPenShift"):
        nopenshift = computeMdbcNoPenShiftWarp(
            currentState,
            operationProperties = OperationProperties(
                kernel = config.kernel,
                operation = WarpOperation.Interpolate,
                supportMode = SupportScheme.Gather,
                operationMode = OperationDirection.BoundaryToFluid,
            ),
            domain = config.domain,
            adjacency = adjacency,
            queryOffsets = currentState.ghostOffsets,
            queryVelocities = currentState.velocities
        )
        return nopenshift[0]