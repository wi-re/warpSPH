"""Fused ACSPH pair sums (Eqs. 23/25), using the core neighbour/operator ABI.

The pressure, divergence, MG viscosity and optional gradient of fluid divergence
share pair geometry/gradW. The fluid-divergence field is a prior pass, never an
in-kernel dependency between threads. No analytic/unilateral wall model is
introduced here; callers keep those on their existing operators.
"""
from typing import Any
import torch
import warp as wp
from warp.types import vector
from warpSPHCore import *
from warpSPHCore.type_config import offset_t
from ...enumTypes import PressureForceScheme


@wp.func
def _sums(i: wp.int32, lane: wp.int32, lanes: wp.int32,
          query: Any, reference: Any, domain: domainData,
          useAdjacency: wp.bool, adjacency: adjacencyData, grid: gridData,
          properties: kernelState, velocities: Any, viscosityVelocities: Any,
          pressures: Any, masks: Any, pressureTerm: wp.int32,
          dynamicNu: scalar_t, fluidDiv: Any, acoustic: wp.bool, zeroV: Any):
    xi, hi, mi, rhoi, ki = getParticle(query, i)
    div = scalar_t(0.0)
    gradP = zero_like_warp(zeroV)
    visc = zero_like_warp(zeroV)
    gradDiv = zero_like_warp(zeroV)
    offsets = wp.int32(1)
    if not useAdjacency:
        offsets = grid.numOffsets
    for o in range(offsets):
        if not useAdjacency and (o % lanes) != lane:
            continue
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
        begin, count = laneSlice(begin, count, lane, lanes, useAdjacency)
        for neighbour in range(count):
            jj = begin + offset_t(neighbour)
            j = wp.int32(0)
            if useAdjacency:
                j = wp.int32(adjacency.neighborList[jj])
            else:
                j = wp.int32(grid.sortIndex[jj])
            xj, hj, mj, rhoj, kj = getParticle(reference, j)
            if kj == 2:
                continue
            volume = mj / rhoj
            # Core wrappers receive no CRK/grad-h/volume/gradient corrections
            # in these ACSPH calls. Use the identical uncorrected helper.
            displacement = computeDistanceVec(xi, xj, domain)
            grad = sphKernelGradient_ij(displacement, hi, hj, properties, domain)
            div += wp.dot((velocities[j] - velocities[i]) * volume, grad)
            pi = pressures[i]
            pj = pressures[j]
            pairP = pj + pi
            if pressureTerm == wp.static(PressureForceScheme.conservative.value):
                pairP = pj - pi
            elif pressureTerm == wp.static(PressureForceScheme.Antuono.value):
                mask = wp.int32(0)
                if masks.shape[0] > 1:
                    mask = masks[i]
                if pi < scalar_t(0.0) and mask != 1:
                    pairP = pj - pi
            elif pressureTerm == wp.static(PressureForceScheme.i.value):
                pairP = pi
            elif pressureTerm == wp.static(PressureForceScheme.j.value):
                pairP = pj
            elif pressureTerm == wp.static(PressureForceScheme.symmetric.value):
                pairP = (pj / (rhoj*rhoj) + pi / (rhoi*rhoi)) * rhoj
            gradP += volume * pairP * grad
            mu = wp.dot(velocities[i] - viscosityVelocities[j], displacement) / (
                wp.dot(displacement, displacement) + scalar_t(1e-14)*hi*hi)
            factor = scalar_t(2*(domain.dim+2)) * dynamicNu
            visc += volume * mu * factor * grad / ((rhoj+rhoi)/scalar_t(2.0))
            if acoustic and ki == 0 and kj == 0:
                gradDiv += (fluidDiv[i] + fluidDiv[j]) * volume * grad
    return div, gradP, visc, gradDiv


@wp.kernel
def _kernel(query: Any, reference: Any, domain: domainData,
            useAdjacency: wp.bool, adjacency: adjacencyData, grid: gridData,
            correction: Any, properties: kernelState,
            velocities: Any, viscosityVelocities: Any, pressures: Any, masks: Any,
            pressureTerm: wp.int32, dynamicNu: scalar_t, fluidDiv: Any, acoustic: wp.bool,
            outDiv: wp.array(dtype=scalar_t), outP: Any, outVisc: Any, outGradDiv: Any):
    i = wp.tid()
    if i >= query.positions.shape[0]:
        return
    d, p, v, a = _sums(i,0,1,query,reference,domain,useAdjacency,adjacency,grid,
        properties,velocities,viscosityVelocities,pressures,masks,pressureTerm,
        dynamicNu,fluidDiv,acoustic,zero_like_warp(outP))
    outDiv[i] = d
    outP[i] = -p/query.densities[i]
    outVisc[i] = v
    outGradDiv[i] = a


@wp.kernel
def _tiled(query: Any, reference: Any, domain: domainData,
           useAdjacency: wp.bool, adjacency: adjacencyData, grid: gridData,
           correction: Any, properties: kernelState,
           velocities: Any, viscosityVelocities: Any, pressures: Any, masks: Any,
           pressureTerm: wp.int32, dynamicNu: scalar_t, fluidDiv: Any, acoustic: wp.bool,
           outDiv: wp.array(dtype=scalar_t), outP: Any, outVisc: Any, outGradDiv: Any):
    i, lane = wp.tid()
    d, p, v, a = _sums(i,lane,wp.block_dim(),query,reference,domain,useAdjacency,adjacency,grid,
        properties,velocities,viscosityVelocities,pressures,masks,pressureTerm,
        dynamicNu,fluidDiv,acoustic,zero_like_warp(outP))
    d = laneSum(d);p = laneSum(p);v = laneSum(v);a = laneSum(a)
    if lane == 0:
        outDiv[i] = d
        outP[i] = -p/query.densities[i]
        outVisc[i] = v
        outGradDiv[i] = a


def _scalar(ctx, extras):
    return castTorchToWarpAsBuiltins(ctx.query.densities).dtype


def _vector(ctx, extras):
    return castTorchToWarpAsBuiltins(ctx.query.positions).dtype


_SPEC = OperatorSpec(kernel=_kernel,tiledKernel=_tiled,
    outputs=(OutputSpec(dtype=_scalar),OutputSpec(dtype=_vector),
             OutputSpec(dtype=_vector),OutputSpec(dtype=_vector)),
    extras=tuple(ExtraSpec(name,ExtraKind.TENSOR) for name in
        ('velocities','viscosityVelocities','pressures','masks')) +
        (ExtraSpec('pressureTerm',ExtraKind.SCALAR),ExtraSpec('dynamicNu',ExtraKind.SCALAR),
         ExtraSpec('fluidDiv',ExtraKind.TENSOR),ExtraSpec('acoustic',ExtraKind.SCALAR)))


def fusedPairSums(view, config, scheme, adjacency, dynamicNu, referenceVelocities, fluidDiv=None):
    dummy=getCachedDummyTensor((1,),dtype=view.pressures.dtype,device=view.positions.device)
    masks=view.surfaceIndicators
    if masks is None:
        masks=getCachedDummyTensor((1,),dtype=torch.int32,device=view.positions.device)
    context=SPHContext(query=view,properties=OperationProperties(kernel=config.kernel,
        supportMode=SupportScheme.SuperSymmetric,operationMode=OperationDirection.AllToAll),
        domain=config.domain,adjacency=adjacency)
    return launchOperator(_SPEC,context,velocities=view.velocities,
        viscosityVelocities=referenceVelocities,pressures=view.pressures,masks=masks,
        pressureTerm=wp.int32(scheme.pressureForceTerm.value),dynamicNu=scalar_t(dynamicNu),
        fluidDiv=dummy if fluidDiv is None else fluidDiv,acoustic=fluidDiv is not None)
