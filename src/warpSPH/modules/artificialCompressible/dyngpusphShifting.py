"""Named 2D dynGPUSPH external-shift/transport compatibility operator.

Port of supplied getContribShift3 + shiftandInterpolateD, not a replacement
for the published Michel law. De Courcy Eq58 calls for Taylor remapping;
the compiled reference uses uncorrected pressure gradients and only the
diagonal velocity-gradient entries. Its cohesion/caps/pressure guards also
differ from the paper. Keep these choices explicit in reference comparisons.
"""
import copy
from typing import Any
import torch
import warp as wp
from warp.types import vector, matrix
from warpSPHCore import *
from warpSPHCore.type_config import offset_t
from ...configurations.compute import neighborSearchOptions

vec2 = vector(length=2, dtype=scalar_t)
mat2 = matrix(shape=(2, 2), dtype=scalar_t)


@wp.func
def _unit(v: vec2):
    n = wp.length(v)
    out = vec2(scalar_t(0.0))
    if n > scalar_t(0.0):
        out = v/n
    return out


@wp.func
def source_cell_pair(x: vec2, old: vec2, R: scalar_t, origin: vec2,
                     nx: wp.int32, ny: wp.int32, width: wp.int32):
    ix = wp.int32(wp.floor((x[0]-origin[0])/R))
    iy = wp.int32(wp.floor((x[1]-origin[1])/R))
    jx = wp.int32(wp.floor((old[0]-origin[0])/R))
    jy = wp.int32(wp.floor((old[1]-origin[1])/R))
    dx = (jx-ix) & (nx-1)
    dy = (jy-iy) & (ny-1)
    return (dx <= width or dx >= nx-width) and (dy <= width or dy >= ny-width)


@wp.kernel
def _kernel(query: Any, reference: Any, domain: domainData,
            useAdjacency: wp.bool, adjacency: adjacencyData, grid: gridData,
            correction: Any, properties: kernelState,
            velocities: Any, pressures: Any, lambdas: Any, edges: Any, normals: Any,
            dx: scalar_t, dt: scalar_t, pairMultiplicity: scalar_t,
            cellPositions: Any, useSourceCells: wp.bool,
            outShift: Any, outRecorded: Any, outP: Any, outV: Any,
            outActive: Any, outGradP: Any, outDiagV: Any, outLambda: Any):
    i = wp.tid()
    xi, searchR, mi, rhoi, ki = getParticle(query, i)
    outShift[i] = vec2(scalar_t(0.0))
    outRecorded[i] = vec2(scalar_t(0.0))
    outP[i] = pressures[i]
    outV[i] = velocities[i]
    outActive[i] = scalar_t(0.0)
    outGradP[i] = vec2(scalar_t(0.0))
    outDiagV[i] = vec2(scalar_t(0.0))
    outLambda[i] = scalar_t(1.0)
    if ki != 0:
        return
    R = searchR/scalar_t(2.0)
    sr = R/scalar_t(2.0)
    lam = lambdas[i]
    edge = edges[i] == 1
    vicinity = edge
    normal = normals[i]
    total = wp.int32(0)
    fluid = wp.int32(0)
    surfDistance = R
    umax = scalar_t(0.0)
    angle = scalar_t(0.0)
    concentration = vec2(scalar_t(0.0))
    lambdaGradient = vec2(scalar_t(0.0))
    wideGradient = vec2(scalar_t(0.0))
    pressureGradient = vec2(scalar_t(0.0))
    velocityDiagonal = vec2(scalar_t(0.0))
    moment = mat2(scalar_t(0.0))
    collision = vec2(scalar_t(0.0))
    collisionMass = scalar_t(0.0)
    pressureSum = scalar_t(0.0)
    pressureWeight = scalar_t(0.0)
    wideWeight = scalar_t(0.0)
    wideLambda = scalar_t(0.0)
    if edges[i] == 0:
        wideWeight = sphKernel_ij(vec2(scalar_t(0.0)), searchR, searchR, properties, domain)*mi/rhoi
        wideLambda = lam*wideWeight
    wdx = sphKernel_ij(vec2(dx, scalar_t(0.0)), R, R, properties, domain)
    offsets = wp.int32(1)
    if not useAdjacency:
        offsets = grid.numOffsets
    for o in range(offsets):
        begin = offset_t(0)
        count = wp.int32(0)
        if useAdjacency:
            begin = adjacency.neighborOffsets[i]
            count = adjacency.numNeighbors[i]
        else:
            first, count = checkOffset(i, query.positions, grid.numCells, grid.D,
                o, grid.cellOffsets, grid.hashTable, grid.cellTable,
                domain.periodicity, grid.qMin, grid.qMax, grid.hCell)
            begin = offset_t(first)
            if begin < 0:
                continue
        for n in range(count):
            j = wp.int32(0)
            if useAdjacency:
                j = wp.int32(adjacency.neighborList[begin+offset_t(n)])
            else:
                j = wp.int32(grid.sortIndex[begin+offset_t(n)])
            xj, hj, mj, rhoj, kj = getParticle(reference, j)
            if i == j or kj == 2:
                continue
            if useSourceCells and not source_cell_pair(xi,cellPositions[j],R,vec2(scalar_t(-.05),scalar_t(-.05)),1024,512,2):
                continue
            r = computeDistanceVec(xi, xj, domain)
            distance = wp.length(r)
            volume = mj/rhoj*pairMultiplicity
            if distance < searchR:
                if edges[j] == 0:
                    weight = sphKernel_ij(r, searchR, searchR, properties, domain)*volume
                    wideLambda += lambdas[j]*weight
                    wideWeight += weight
                wideGradient += (lambdas[j]-lam)*sphKernelGradient_ij(r, searchR, searchR, properties, domain)*volume
                if edges[j] == 1:
                    # CUDA fmax ignores the NaN from an out-of-range acos.
                    cosine = wp.dot(normal, normals[j])
                    if cosine >= scalar_t(-1.0) and cosine <= scalar_t(1.0):
                        angle = wp.max(angle, wp.acos(cosine))
            if distance < R:
                total += 1
                weight = sphKernel_ij(r, R, R, properties, domain)*volume
                grad = sphKernelGradient_ij(r, R, R, properties, domain)*volume
                w = sphKernel_ij(r, R, R, properties, domain)
                concentration += (scalar_t(1.0)+scalar_t(.2)*wp.pow(w/wdx,scalar_t(4.0)))*grad
                lambdaGradient += (lambdas[j]-lam)*grad
                pressureGradient += (pressures[j]-pressures[i])*grad
                velocityDiagonal += wp.cw_mul(velocities[j]-velocities[i],grad)
                moment -= wp.outer(r, grad)
                pressureSum += pressures[j]*weight
                pressureWeight += weight
                if kj == 0:
                    fluid += 1
                    if distance > scalar_t(0.0):
                        umax = wp.max(umax, wp.abs(wp.dot(velocities[i]-velocities[j],r/distance)))
                    if edges[j] == 1:
                        vicinity = True
                        surfDistance = wp.min(surfDistance,distance)
                if kj == 1 and distance < scalar_t(.9)*dx and distance > scalar_t(0.0):
                    collision += pairMultiplicity*mj*(distance-scalar_t(.9)*dx)*r/distance
                    collisionMass += pairMultiplicity*mj
    outGradP[i] = pressureGradient
    outDiagV[i] = velocityDiagonal
    if total == 0:
        outP[i] = scalar_t(0.0)
        return
    collisionShift = -collision/(mi+collisionMass)
    if fluid == 0:
        outShift[i] = collisionShift
        return
    outActive[i] = scalar_t(1.0)
    outward = vec2(scalar_t(0.0))
    if vicinity and wp.abs(wp.determinant(moment)) > scalar_t(1e-8):
        outward = -_unit(wp.inverse(moment)*lambdaGradient)
    averageLambda = scalar_t(1.0)
    if wideWeight > scalar_t(1e-6):
        averageLambda = wideLambda/wideWeight
    outLambda[i] = averageLambda
    ratio = R/dx
    beta0 = ratio*ratio*ratio
    beta = beta0
    if surfDistance < R:
        beta = scalar_t(1.0)+surfDistance*(beta0-scalar_t(1.0))/R
    if edge:
        beta = scalar_t(1.0)
        surfDistance = scalar_t(0.0)
    raw = scalar_t(.5)*ratio*_unit(concentration)
    if wp.length(beta0*R*concentration) < scalar_t(.5)*ratio:
        raw = beta*R*concentration
    raw *= -scalar_t(.2)*umax*dt
    if lam < scalar_t(.4):
        raw = vec2(scalar_t(0.0))
    cohesion = scalar_t(5.0)*sr*_unit(wideGradient)*(scalar_t(1.0)-wp.min(averageLambda,scalar_t(1.0)))
    if not vicinity or lam < scalar_t(.3) or angle > scalar_t(.3*3.141592653589793):
        cohesion = vec2(scalar_t(0.0))
    magnitude = wp.length(cohesion)
    if magnitude > scalar_t(1e-10):
        cohesion = wp.min(magnitude,scalar_t(.01)*sr)*_unit(cohesion)
    else:
        cohesion = vec2(scalar_t(0.0))
    if vicinity and wp.dot(_unit(outward),_unit(raw)) >= scalar_t(0.0):
        sigma = scalar_t(0.0)
        if surfDistance < dx:
            # Supplied helper_double::clamp is clamp(value,low,high),
            # but the call is clamp(0,1,1-d/dx): it returns exactly 1.
            sigma = scalar_t(1.0)
        raw -= sigma*wp.dot(raw,outward)*outward
    shift = raw*lam*lam+cohesion
    magnitude = wp.length(shift)
    if magnitude > scalar_t(1e-10):
        shift = wp.min(magnitude,scalar_t(.05)*sr)*_unit(shift)
    else:
        shift = vec2(scalar_t(0.0))
    shift += collisionShift
    outShift[i] = shift
    outRecorded[i] = shift
    p = pressures[i]+wp.dot(shift,pressureGradient)
    outV[i] = velocities[i]+wp.cw_mul(shift,velocityDiagonal)
    if pressureWeight > scalar_t(.1) and not vicinity:
        averageP = pressureSum/pressureWeight
        if wp.abs(pressures[i]) > scalar_t(200.0)*wp.abs(averageP):
            p = averageP
    if edge and lam < scalar_t(.1):
        p = scalar_t(0.0)
    outP[i] = p


def _scalar(ctx, extras):
    return castTorchToWarpAsBuiltins(ctx.query.densities).dtype


def _vector(ctx, extras):
    return castTorchToWarpAsBuiltins(ctx.query.positions).dtype


_SPEC = OperatorSpec(kernel=_kernel,
    outputs=tuple(OutputSpec(dtype=f) for f in
        (_vector,_vector,_scalar,_vector,_scalar,_vector,_vector,_scalar)),
    extras=tuple(ExtraSpec(name,ExtraKind.TENSOR) for name in
        ('velocities','pressures','lambdas','edges','normals'))+
        (ExtraSpec('dx',ExtraKind.SCALAR),ExtraSpec('dt',ExtraKind.SCALAR),
         ExtraSpec('pairMultiplicity',ExtraKind.SCALAR),
         ExtraSpec('cellPositions',ExtraKind.TENSOR),ExtraSpec('useSourceCells',ExtraKind.SCALAR)))


@wp.kernel
def _surface_fields(query: Any, reference: Any, domain: domainData,
            useAdjacency: wp.bool, adjacency: adjacencyData, grid: gridData,
            correction: Any, properties: kernelState, outLambda: Any, outNormal: Any):
    i = wp.tid()
    xi, R, mi, rhoi, ki = getParticle(query, i)
    outLambda[i] = scalar_t(1.0)
    outNormal[i] = vec2(scalar_t(0.0))
    if ki != 0:
        return
    moment = mat2(scalar_t(0.0))
    normal = vec2(scalar_t(0.0))
    offsets = wp.int32(1)
    if not useAdjacency:
        offsets = grid.numOffsets
    for o in range(offsets):
        begin = offset_t(0)
        count = wp.int32(0)
        if useAdjacency:
            begin = adjacency.neighborOffsets[i]
            count = adjacency.numNeighbors[i]
        else:
            first, count = checkOffset(i, query.positions, grid.numCells, grid.D,
                o, grid.cellOffsets, grid.hashTable, grid.cellTable,
                domain.periodicity, grid.qMin, grid.qMax, grid.hCell)
            begin = offset_t(first)
            if begin < 0:
                continue
        for n in range(count):
            j = wp.int32(0)
            if useAdjacency:
                j = wp.int32(adjacency.neighborList[begin+offset_t(n)])
            else:
                j = wp.int32(grid.sortIndex[begin+offset_t(n)])
            xj, hj, mj, rhoj, kj = getParticle(reference, j)
            if i == j or kj == 2:
                continue
            r = computeDistanceVec(xi, xj, domain)
            distance = wp.length(r)
            if distance < R:
                grad = sphKernelGradient_ij(r, R, R, properties, domain)*mj/rhoj
                moment -= wp.outer(r, grad)
                normal -= grad
    lam = scalar_t(0.0)
    if wp.abs(wp.determinant(moment)) > scalar_t(1e-6):
        trace = moment[0,0]+moment[1,1]
        diagonal = moment[0,0]-moment[1,1]
        disc = wp.max(diagonal*diagonal+scalar_t(4.0)*moment[0,1]*moment[1,0],scalar_t(0.0))
        lam = (trace-wp.sqrt(disc))/scalar_t(2.0)
        normal = wp.inverse(moment)*normal
    outLambda[i] = lam
    if wp.length(normal) > scalar_t(1e-6):
        outNormal[i] = _unit(normal)


@wp.kernel
def _surface_cone(query: Any, reference: Any, domain: domainData,
            useAdjacency: wp.bool, adjacency: adjacencyData, grid: gridData,
            correction: Any, properties: kernelState, lambdas: Any, normals: Any, outEdge: Any):
    i = wp.tid()
    xi, R, mi, rhoi, ki = getParticle(query, i)
    outEdge[i] = scalar_t(0.0)
    if ki != 0 or lambdas[i] > scalar_t(.75):
        return
    outEdge[i] = scalar_t(1.0)
    if lambdas[i] < scalar_t(.2):
        return
    normal = _unit(normals[i])
    offsets = wp.int32(1)
    if not useAdjacency:
        offsets = grid.numOffsets
    for o in range(offsets):
        begin = offset_t(0)
        count = wp.int32(0)
        if useAdjacency:
            begin = adjacency.neighborOffsets[i]
            count = adjacency.numNeighbors[i]
        else:
            first, count = checkOffset(i, query.positions, grid.numCells, grid.D,
                o, grid.cellOffsets, grid.hashTable, grid.cellTable,
                domain.periodicity, grid.qMin, grid.qMax, grid.hCell)
            begin = offset_t(first)
            if begin < 0:
                continue
        for n in range(count):
            j = wp.int32(0)
            if useAdjacency:
                j = wp.int32(adjacency.neighborList[begin+offset_t(n)])
            else:
                j = wp.int32(grid.sortIndex[begin+offset_t(n)])
            xj, hj, mj, rhoj, kj = getParticle(reference, j)
            if i == j or kj == 2:
                continue
            r = computeDistanceVec(xi, xj, domain)
            distance = wp.length(r)
            # Supplied detectSurfaceCell's outer test makes its tube branch
            # unreachable. Exact +/-1 cosine is excluded by the CUDA source.
            if distance < R/scalar_t(4.0) and distance > scalar_t(.0005)*R:
                cosine = wp.dot(normal,-r/distance)
                if wp.abs(cosine) < scalar_t(1.0):
                    if wp.acos(cosine) < scalar_t(.7853981633974483):
                        outEdge[i] = scalar_t(0.0)


@wp.kernel
def _surface_vicinity(query: Any, reference: Any, domain: domainData,
            useAdjacency: wp.bool, adjacency: adjacencyData, grid: gridData,
            correction: Any, properties: kernelState, raw: Any, outEdge: Any):
    i = wp.tid()
    xi, R, mi, rhoi, ki = getParticle(query, i)
    outEdge[i] = raw[i]
    if ki != 0 or raw[i] == scalar_t(1.0):
        return
    offsets = wp.int32(1)
    if not useAdjacency:
        offsets = grid.numOffsets
    for o in range(offsets):
        begin = offset_t(0)
        count = wp.int32(0)
        if useAdjacency:
            begin = adjacency.neighborOffsets[i]
            count = adjacency.numNeighbors[i]
        else:
            first, count = checkOffset(i, query.positions, grid.numCells, grid.D,
                o, grid.cellOffsets, grid.hashTable, grid.cellTable,
                domain.periodicity, grid.qMin, grid.qMax, grid.hCell)
            begin = offset_t(first)
            if begin < 0:
                continue
        for n in range(count):
            j = wp.int32(0)
            if useAdjacency:
                j = wp.int32(adjacency.neighborList[begin+offset_t(n)])
            else:
                j = wp.int32(grid.sortIndex[begin+offset_t(n)])
            xj, hj, mj, rhoj, kj = getParticle(reference, j)
            if i == j or kj == 2:
                continue
            r = computeDistanceVec(xi, xj, domain)
            distance = wp.length(r)
            if distance < R and kj == 0 and raw[j] == scalar_t(1.0):
                outEdge[i] = scalar_t(2.0)


_FIELDS_SPEC = OperatorSpec(kernel=_surface_fields,
    outputs=(OutputSpec(dtype=_scalar),OutputSpec(dtype=_vector)))
_CONE_SPEC = OperatorSpec(kernel=_surface_cone,outputs=(OutputSpec(dtype=_scalar),),
    extras=(ExtraSpec('lambdas',ExtraKind.TENSOR),ExtraSpec('normals',ExtraKind.TENSOR)))
_VICINITY_SPEC = OperatorSpec(kernel=_surface_vicinity,outputs=(OutputSpec(dtype=_scalar),),
    extras=(ExtraSpec('raw',ExtraKind.TENSOR),))


def reference_surface(stage0, accepted, config, *, stage_adjacency=None, accepted_adjacency=None):
    """Compiled deltaHash4D/detectSurfaceCell inputs for the external shift.

    Lambda, color-gradient normal and raw edges belong to the final dual-time
    iteration's stage zero. The residual's vicinity tags use accepted geometry.
    This does not change the pressure solver's existing surface/taper model.
    """
    def context(state, adjacency):
        if adjacency is None:
            adjacency = buildVerletList(state,config.domain,verletScale=config.verletScale,
                supportMode=SupportScheme.Gather,verbose=False,**neighborSearchOptions(config))
        return SPHContext(query=state,properties=OperationProperties(kernel=config.kernel,
            supportMode=SupportScheme.Gather,operationMode=OperationDirection.AllToAll),
            domain=config.domain,adjacency=adjacency)
    stage = context(stage0,stage_adjacency)
    lambdas,normals = launchOperator(_FIELDS_SPEC,stage)
    raw = launchOperator(_CONE_SPEC,stage,lambdas=lambdas,normals=normals)
    edges = launchOperator(_VICINITY_SPEC,context(accepted,accepted_adjacency),raw=raw)
    return lambdas,normals,edges


def shift_and_remap(state, config, *, dt, lambdas, edges, normals, pair_multiplicity=1, cell_positions=None):
    """Return dx, recorded-dx, p, v, history-mask and audit intermediates.

    Inputs are the accepted fields and last dual-iteration surface fields,
    matching dynGPUSPH's external ShiftHash call. Ghost rows never contribute.
    This explicit 2D fixed-support reference path does not implement analytic
    walls, periodic domains or moving bodies.

    The supplied Lobovsky hash grid has two z cells. Its five-cell shift
    stencil wraps onto the populated plane three times. Set multiplicity=3
    for that compiled configuration; 1 evaluates each physical pair once.
    """
    if state.positions.shape[1] != 2 or bool(config.domain.periodic.any()):
        raise ValueError('dynGPUSPH shift compatibility is a closed/free-surface 2D reference path')
    search = copy.copy(state)
    search.supports = 2*state.supports
    adjacency = buildVerletList(search,config.domain,verletScale=config.verletScale,
        supportMode=SupportScheme.Gather,verbose=False,**neighborSearchOptions(config))
    wall = state.kinds == 1
    lambdas = torch.where(wall,torch.ones_like(lambdas),lambdas)
    edges = torch.where(state.kinds==0,edges.to(torch.int32),torch.zeros_like(state.kinds))
    context = SPHContext(query=search,properties=OperationProperties(kernel=config.kernel,
        supportMode=SupportScheme.Gather,operationMode=OperationDirection.AllToAll),
        domain=config.domain,adjacency=adjacency)
    return launchOperator(_SPEC,context,velocities=state.velocities,pressures=state.pressures,
        lambdas=lambdas,edges=edges,normals=normals,dx=scalar_t(config.dx),dt=scalar_t(dt),
        pairMultiplicity=scalar_t(pair_multiplicity),
        cellPositions=state.positions if cell_positions is None else cell_positions,
        useSourceCells=cell_positions is not None)
