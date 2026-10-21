"""Opt-in material density rules; geometric volumes/moments remain all-material.

sameMaterial: Spheral's sum(m_j V_j W_j)/sum(V_j^2 W_j), restricted to
the same material, with the positive uncorrected scatter kernel.
massReplacement: Frontiere et al. section 3.4's corrected-kernel ratio,
with m_ij=m_i across material IDs and m_j within a material.

These are density policies, not a port of Spheral's continuity-volume update.
Self is already included by the core neighbor traversal; do not add it twice.
"""
from typing import Any
import torch
import warp as wp
from warpSPHCore import (scalar_t, index_t, offset_t, domainData, adjacencyData,
    gridData, kernelState, OperationProperties, OperationDirection, SupportScheme)
from warpSPHCore.autograd import SPHContext, Corrections, OperatorSpec, OutputSpec, ExtraSpec, ExtraKind, launchOperator
from warpSPHCore.util import getParticleData, getParticleCorrectionData_i, getVolume_j
from warpSPHCore.math import computeDistanceVec
from warpSPHCore.kernels import sphKernel_ij
from warpSPHCore.crk.kernel import computeKernelCRK
from warpSPHCore.radiusSearch.grid_util import getIndexRangeLane, laneSum


POLICIES = ('legacy', 'sameMaterial', 'massReplacement')


@wp.func
def _materialDensitySums(i: wp.int32, lane: wp.int32, lanes: wp.int32,
        queryState: Any, referenceState: Any, domainState: domainData,
        useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
        correctionData: Any, properties: kernelState,
        ids: wp.array(dtype=wp.int32), policy: wp.int32):
    pi = getParticleData(queryState, i)
    ci = getParticleCorrectionData_i(correctionData, i)
    numerator, denominator = scalar_t(0.), scalar_t(0.)
    for o in range(gridState.numOffsets if not useAdjacency else 1):
        begin, count = getIndexRangeLane(i, o, lane, lanes, useAdjacency, adjacencyState, gridState, queryState, domainState)
        if begin >= 0:
            for k in range(count):
                at = begin + offset_t(k)
                j = wp.int32(adjacencyState.neighborList[at] if useAdjacency else gridState.sortIndex[at])
                if policy != 1 or ids[i] == ids[j]:
                    pj = getParticleData(referenceState, j)
                    _, vj = getVolume_j(correctionData, j)
                    weight = scalar_t(0.)
                    mass = pj.mass
                    if policy == 1:
                        delta = computeDistanceVec(pi.position, pj.position, domainState)
                        weight = sphKernel_ij(delta, pi.support, pj.support, properties, domainState)
                    else:
                        weight = computeKernelCRK(pi.position, pj.position, pi.support, pj.support,
                                                  properties, domainState, True, ci.A, ci.B)
                        if ids[i] != ids[j]:
                            mass = pi.mass
                    numerator += mass * vj * weight
                    denominator += vj * vj * weight
    # Keep division outside the function with dynamic loops (Warp reverse AD).
    return numerator, denominator


@wp.kernel
def _materialDensityKernel(queryState: Any, referenceState: Any, domainState: domainData,
        useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
        correctionData: Any, properties: kernelState,
        ids: wp.array(dtype=wp.int32), policy: wp.int32,
        output: wp.array(dtype=scalar_t)):
    i = wp.tid()
    if i < queryState.positions.shape[0]:
        a, b = _materialDensitySums(i, 0, 1, queryState, referenceState, domainState,
            useAdjacency, adjacencyState, gridState, correctionData, properties, ids, policy)
        output[i] = a / b if b != scalar_t(0.) else scalar_t(0.)


@wp.kernel
def _materialDensityTiled(queryState: Any, referenceState: Any, domainState: domainData,
        useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
        correctionData: Any, properties: kernelState,
        ids: wp.array(dtype=wp.int32), policy: wp.int32,
        output: wp.array(dtype=scalar_t)):
    i, lane = wp.tid()
    a, b = _materialDensitySums(i, lane, wp.block_dim(), queryState, referenceState, domainState,
        useAdjacency, adjacencyState, gridState, correctionData, properties, ids, policy)
    numerator, denominator = laneSum(a), laneSum(b)
    if lane == 0:
        output[i] = numerator / denominator if denominator != scalar_t(0.) else scalar_t(0.)


_SPEC = OperatorSpec(kernel=_materialDensityKernel, tiledKernel=_materialDensityTiled,
    outputs=(OutputSpec(dtype=scalar_t),), extras=(ExtraSpec('ids', ExtraKind.TENSOR), ExtraSpec('policy', ExtraKind.SCALAR)))


def materialDensity(particles, domain, kernel, volumes, crkState, adjacency, policy):
    if policy not in POLICIES[1:]:
        raise ValueError(f'Unknown material CRK density policy {policy!r}')
    ids = particles.materials
    if ids.dtype != torch.int32 or ids.shape != particles.masses.shape or ids.device != particles.masses.device:
        raise ValueError('material density requires int32 particle IDs matching particle masses')
    ctx = SPHContext(query=particles, domain=domain, adjacency=adjacency,
        properties=OperationProperties(kernel=kernel, supportMode=SupportScheme.Scatter,
                                       operationMode=OperationDirection.AllToAll),
        corrections=Corrections(volumes=(volumes, volumes), crk=crkState))
    return launchOperator(_SPEC, ctx, ids=ids, policy=POLICIES.index(policy))
