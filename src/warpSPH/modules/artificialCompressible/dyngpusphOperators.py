"""Static 2D AC-2L operators from the qualified dynGPUSPH Lobovsky build.

These are explicit reference-code choices, not generic ACSPH defaults. Pair
closures keep physical wall velocity separate from its Shepard average. No
shared boundary or inverse policy is changed. See DYNGPUSPH_COMPARISON.md.
"""
from typing import Any
import torch
import warp as wp
from warp.types import vector, matrix
from warpSPHCore import *
from warpSPHCore.type_config import offset_t
from .dyngpusphShifting import _unit, _CONE_SPEC, source_cell_pair

vec2 = vector(length=2, dtype=scalar_t)
mat2 = matrix(shape=(2, 2), dtype=scalar_t)




def _scalar(ctx, extras):
    return castTorchToWarpAsBuiltins(ctx.query.densities).dtype


def _vector(ctx, extras):
    return castTorchToWarpAsBuiltins(ctx.query.positions).dtype


def context(state, config, adjacency):
    return SPHContext(query=state, properties=OperationProperties(kernel=config.kernel,
        supportMode=SupportScheme.Gather, operationMode=OperationDirection.AllToAll),
        domain=config.domain, adjacency=adjacency)

@wp.kernel
def _closures(query: Any, reference: Any, domain: domainData,
        useAdjacency: wp.bool, adjacency: adjacencyData, grid: gridData,
        correction: Any, properties: kernelState,
        cellPositions: Any, cells: wp.bool, cellOrigin: vec2, nx: wp.int32, ny: wp.int32,
        velocities: Any, pressures: Any, priorLambda: Any, priorEdges: Any, bodyForce: Any, outP: Any, outNormal: Any, outAverage: Any, outDiv: Any, outLap: Any):
    i = wp.tid()
    xi, R, mi, rhoi, ki = getParticle(query, i)
    sr = R/scalar_t(2.0)
    pSum = scalar_t(0.0)
    weightSum = scalar_t(0.0)
    lambdaSum = scalar_t(0.0)
    average = vec2(scalar_t(0.0))
    meanPosition = vec2(scalar_t(0.0))
    normal = vec2(scalar_t(0.0))
    lap = vec2(scalar_t(0.0))
    div = scalar_t(0.0)
    surface = wp.bool(False)

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
            if cells and not source_cell_pair(xi, cellPositions[j], R, cellOrigin, nx, ny, 1):
                continue
            r = computeDistanceVec(xi, xj, domain)
            d2 = wp.dot(r,r)
            if d2 >= R*R:
                continue
            distance = wp.sqrt(d2)
            volume = mj/rhoj
            g = sphKernelGradient_ij(r, R, R, properties, domain)*volume
            if kj == 0:
                weight = sphKernel_ij(r,R,R,properties,domain)*volume
                weightSum += weight
                pSum += (pressures[j]+rhoj*wp.dot(bodyForce[i],r))*weight
                average += velocities[j]*weight
                meanPosition += xj*weight
                lambdaSum += wp.max(scalar_t(0.0),scalar_t(2.0)*(priorLambda[j]-scalar_t(.5)))*weight
                surface = surface or priorEdges[j] == 1
                div += wp.dot(velocities[j]-velocities[i],g)
                lap += wp.dot(velocities[i]-velocities[j],r)*g/(d2+scalar_t(.01)*sr*sr)
            if kj == 1:
                normal -= g/volume
    outP[i] = pressures[i]
    outNormal[i] = vec2(scalar_t(0.0))
    outAverage[i] = vec2(scalar_t(0.0))
    outDiv[i] = scalar_t(0.0)
    outLap[i] = lap
    if ki == 0:
        outDiv[i] = div
    if ki == 1:
        if wp.length(normal) > scalar_t(1e-6):
            normal = _unit(normal)
            if weightSum > scalar_t(0.0) and wp.dot(normal,meanPosition/weightSum-xi) < scalar_t(0.0):
                normal = -normal
            outNormal[i] = normal
        p = scalar_t(0.0)
        if weightSum > scalar_t(0.0):
            p = pSum/weightSum
            if surface and p < scalar_t(0.0):
                p *= lambdaSum/weightSum
            outAverage[i] = average/weightSum
        outP[i] = p


@wp.kernel
def _moments(query: Any, reference: Any, domain: domainData,
        useAdjacency: wp.bool, adjacency: adjacencyData, grid: gridData,
        correction: Any, properties: kernelState,
        cellPositions: Any, cells: wp.bool, cellOrigin: vec2, nx: wp.int32, ny: wp.int32,
        pressures: Any, outLambda: Any, outNormal: Any, outGrad: Any, outLap: Any):
    i = wp.tid()
    xi, R, mi, rhoi, ki = getParticle(query, i)
    sr = R/scalar_t(2.0)
    outLambda[i] = scalar_t(1.0)
    outNormal[i] = vec2(scalar_t(0.0))
    outGrad[i] = vec2(scalar_t(0.0))
    outLap[i] = scalar_t(0.0)
    if ki != 0:
        return
    moment = mat2(scalar_t(0.0))
    normal = vec2(scalar_t(0.0))
    grad = vec2(scalar_t(0.0))
    lap = scalar_t(0.0)

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
            if cells and not source_cell_pair(xi, cellPositions[j], R, cellOrigin, nx, ny, 1):
                continue
            r = computeDistanceVec(xi, xj, domain)
            d2 = wp.dot(r,r)
            if d2 >= R*R:
                continue
            distance = wp.sqrt(d2)
            volume = mj/rhoj
            g = sphKernelGradient_ij(r, R, R, properties, domain)*volume
            moment -= wp.outer(r,g)
            normal -= g
            grad += (pressures[j]-pressures[i])*g
            if distance > scalar_t(.01)*sr:
                lap += scalar_t(2.0)*(pressures[i]-pressures[j])*wp.dot(r,g)/d2
    lam = scalar_t(0.0)
    if wp.abs(wp.determinant(moment)) > scalar_t(1e-6):
        diagonal = moment[0,0]-moment[1,1]
        disc = wp.max(diagonal*diagonal+scalar_t(4.0)*moment[0,1]*moment[1,0],scalar_t(0.0))
        lam = (moment[0,0]+moment[1,1]-wp.sqrt(disc))/scalar_t(2.0)
        inverse = wp.inverse(moment)
        normal = inverse*normal
        grad = inverse*grad
    outLambda[i] = lam
    if wp.length(normal) > scalar_t(1e-6):
        outNormal[i] = _unit(normal)
    outGrad[i] = grad
    outLap[i] = lap


@wp.kernel
def _residual(query: Any, reference: Any, domain: domainData,
        useAdjacency: wp.bool, adjacency: adjacencyData, grid: gridData,
        correction: Any, properties: kernelState,
        cellPositions: Any, cells: wp.bool, cellOrigin: vec2, nx: wp.int32, ny: wp.int32,
        velocities: Any, pressures: Any, wallNormal: Any, wallAverage: Any, fluidDiv: Any, velocityLap: Any, pressureGrad: Any, pressureLap: Any, edges: Any, frozenSwitch: Any, freezeSwitch: wp.bool, outDiv: Any, outPressure: Any, outVisc: Any, outAcoustic: Any, outSecond: Any, outFourth: Any, outWallDistance: Any, outDiffusion: Any, outSwitch: Any, outEdges: Any):
    i = wp.tid()
    xi, R, mi, rhoi, ki = getParticle(query, i)
    sr = R/scalar_t(2.0)
    div = scalar_t(0.0)
    pressure = vec2(scalar_t(0.0))
    visc = vec2(scalar_t(0.0))
    acoustic = vec2(scalar_t(0.0))
    second = vec2(scalar_t(0.0))
    fourth = vec2(scalar_t(0.0))
    diffusion = scalar_t(0.0)
    switch = scalar_t(0.0)
    wallDistance = scalar_t(1.05)*R
    edge = edges[i]

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
            if cells and not source_cell_pair(xi, cellPositions[j], R, cellOrigin, nx, ny, 1):
                continue
            r = computeDistanceVec(xi, xj, domain)
            d2 = wp.dot(r,r)
            if d2 >= R*R:
                continue
            distance = wp.sqrt(d2)
            volume = mj/rhoj
            g = sphKernelGradient_ij(r, R, R, properties, domain)*volume
            vj = velocities[j]
            physical = vj
            if kj == 1:
                normal = wallNormal[j]
                tangent = vec2(normal[1],-normal[0])
                physical = normal*wp.dot(vj,normal)+tangent*wp.dot(velocities[i],tangent)
                vj = normal*(scalar_t(2.0)*wp.dot(velocities[j],normal)-wp.dot(wallAverage[j],normal))+tangent*wp.dot(wallAverage[j],tangent)
                wallDistance = wp.min(wallDistance,distance)
            vab = velocities[i]-physical
            div += wp.dot(vj-velocities[i],g)
            pressure -= (pressures[i]+pressures[j])*g/rhoi
            if distance > scalar_t(.01)*sr:
                visc += scalar_t(8.0)*wp.dot(vab,r)*g/d2
                gradP = scalar_t(.5)*(pressureGrad[i]+pressureGrad[j])
                if kj == 1:
                    gradP = pressureGrad[i]
                diffusion += scalar_t(2.0)*((pressures[i]-pressures[j])-wp.dot(gradP,r))*wp.dot(r,g)/d2
            second += wp.dot(vab,r)*g/(d2+scalar_t(1e-6)*sr*sr)
            fourth += wp.dot(velocityLap[i]-velocityLap[j],r)*g/(d2+scalar_t(1e-6)*sr*sr)
            if kj == 0:
                acoustic += (fluidDiv[i]+fluidDiv[j])*g
                if edges[j] == 1 and edge != 1:
                    edge = scalar_t(2.0)
            if wp.abs(pressures[i]+pressures[j]) > scalar_t(1e-5):
                switch += wp.abs(pressureLap[j]-pressureLap[i])/wp.abs(pressures[i]+pressures[j])
    if freezeSwitch:
        switch = frozenSwitch[i]
    else:
        switch *= scalar_t(.1)*sr*sr
    outDiv[i] = div
    outPressure[i] = pressure
    outVisc[i] = visc
    outAcoustic[i] = acoustic
    outSecond[i] = second
    outFourth[i] = fourth
    outWallDistance[i] = wallDistance
    outDiffusion[i] = diffusion
    outSwitch[i] = switch
    outEdges[i] = edge



_COMMON = tuple(ExtraSpec(n, ExtraKind.TENSOR) for n in ('cellPositions',)) + (
    ExtraSpec('cells',ExtraKind.SCALAR),ExtraSpec('cellOrigin',ExtraKind.SCALAR),
    ExtraSpec('nx',ExtraKind.SCALAR),ExtraSpec('ny',ExtraKind.SCALAR))

def _spec(kernel, tensors, outputTypes, scalarExtras=()):
    return OperatorSpec(kernel=kernel,
        outputs=tuple(OutputSpec(dtype=t) for t in outputTypes),
        extras=_COMMON+tuple(ExtraSpec(n,ExtraKind.TENSOR) for n in tensors)+
            tuple(ExtraSpec(n,ExtraKind.SCALAR) for n in scalarExtras))

_CLOSURES = _spec(_closures,('velocities','pressures','priorLambda','priorEdges','bodyForce'),
    (_scalar,_vector,_vector,_scalar,_vector))
_MOMENTS = _spec(_moments,('pressures',),(_scalar,_vector,_vector,_scalar))
_RESIDUAL = _spec(_residual,('velocities','pressures','wallNormal','wallAverage',
    'fluidDiv','velocityLap','pressureGrad','pressureLap','edges','frozenSwitch'),
    (_scalar,_vector,_vector,_vector,_vector,_vector,_scalar,_scalar,_scalar,_scalar),
    ('freezeSwitch',))


def cell_arguments(state, cell_positions=None, origin=(-.05,-.05), grid=(1024,512)):
    return dict(cellPositions=state.positions if cell_positions is None else cell_positions,
        cells=cell_positions is not None,cellOrigin=vec2(*origin),nx=wp.int32(grid[0]),ny=wp.int32(grid[1]))


def closures(state,config,adjacency,prior_lambda,prior_edges,body_force,**cell_args):
    return launchOperator(_CLOSURES,context(state,config,adjacency),
        velocities=state.velocities,pressures=state.pressures,priorLambda=prior_lambda,
        priorEdges=prior_edges,bodyForce=body_force,**cell_arguments(state,**cell_args))


def moments(state,config,adjacency,**cell_args):
    return launchOperator(_MOMENTS,context(state,config,adjacency),pressures=state.pressures,
        **cell_arguments(state,**cell_args))


def raw_surface(state,config,adjacency,lambdas,normals):
    return launchOperator(_CONE_SPEC,context(state,config,adjacency),lambdas=lambdas,normals=normals)


def residual_terms(state,config,adjacency,wall_normal,wall_average,fluid_div,velocity_lap,
                   pressure_grad,pressure_lap,edges,frozen_switch=None,**cell_args):
    return launchOperator(_RESIDUAL,context(state,config,adjacency),velocities=state.velocities,
        pressures=state.pressures,wallNormal=wall_normal,wallAverage=wall_average,
        fluidDiv=fluid_div,velocityLap=velocity_lap,pressureGrad=pressure_grad,
        pressureLap=pressure_lap,edges=edges,
        frozenSwitch=torch.zeros_like(state.pressures) if frozen_switch is None else frozen_switch,
        freezeSwitch=frozen_switch is not None,**cell_arguments(state,**cell_args))


def momentum_jst(state,terms,velocity_lap,beta):
    """Native pressure-sensor velocity JST, independent of pressure smoothing."""
    _,_,_,_,second,fourth,distance,_,switch,_=terms
    h=state.supports/2
    eps2=.01*switch.clamp(max=10)
    eps4=(.005-.1*eps2).clamp(min=0)
    q=((distance-h)/h).clamp(0,1)
    # C2_interp(distance,2h,h): 1 near the wall, 0 at 2h.
    weight=(1-q).pow(4)*(4*q+1)
    near=distance<=2*h
    ep2=weight*torch.maximum(eps2,eps4)+(1-weight)*eps2
    ep4=(1-weight)*eps4
    lap=torch.where(near[:,None],second,velocity_lap)
    a2=torch.where(near,ep2,eps2)*beta*h
    a4=torch.where(near,ep4,eps4)*beta*h.pow(3)
    return a2[:,None]*lap-a4[:,None]*fourth
