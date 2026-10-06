"""Per-side (Hamiltonian) SPH pressure force and its conjugate pdV work, one neighbour pass
(SUPPORT_SOLVER_PLAN option C; Price 2012, Eqs. 43-45; Springel & Hernquist 2002):

    f_i      = P_i / (Omega_i rho_i^2)
    a_i      = - sum_j m_j [ f_i grad_i W_ij(h_i) + f_j grad_i W_ij(h_j) ]
    du_i/dt  =   f_i sum_j m_j (v_i - v_j) . grad_i W_ij(h_i)

Each side's term uses its own smoothing length -- not the mean kernel `(grad W(h_i) + grad W(h_j))/2`
the default symmetric gradient uses. Momentum and total energy are conserved pair by pair for *any*
per-particle factor f_i, so the pair is exactly conservative with or without Omega; with
`Omega_i = 1 + h_i / (d rho_i) sum_j m_j dW_ij(h_i)/dh_i` (`computeOmega`, Gather) and h = eta (m/rho)^(1/d)
solved with a gather density (`AdaptiveSupportScheme.Monaghan`) it is the Lagrangian of Price 2012 Eq. 43.
Omega is meaningless for an h that is not h(rho) (Owen); pass `queryOmegas=None` there (Omega = 1).

`grad W(h)` per side is `sphKernelGradient(x_i, x_j, h, h, ...)`: with both supports equal every support
mode reduces to the plain kernel gradient at h, with the configured normalisation. The neighbour list must
cover `max(h_i, h_j)` (the SuperSymmetric Verlet list the compressible schemes build does).
"""

import warp as wp
from warp.types import vector
from typing import Any, Optional, Union
import torch
from warpSPHCore.profiling import record_function
from warpSPHCore import *

__all__ = ['computePerSidePressureWarp']


@wp.func
def perSide_Func_i(
    i: wp.int32,
    xi: vector(dtype=scalar_t, length=Any), hi: scalar_t, mi: scalar_t, rhoi: scalar_t,  # type: ignore
    referenceState: Any,
    domainState: domainData,
    kernelProperties: kernelState,
    beginIndex: wp.int32, numIndices: wp.int32, offsetArray: wp.array(dtype=wp.int64),  # type: ignore
    ki: wp.int32, referenceKinds: wp.array(dtype=wp.int32),  # type: ignore
    vel_i: vector(length=Any, dtype=scalar_t), referenceVelocities: wp.array(dtype=vector(length=Any, dtype=scalar_t)),  # type: ignore
    f_i: scalar_t, referencePressures: wp.array(dtype=scalar_t), referenceOmegas: wp.array(dtype=scalar_t),  # type: ignore
    useOmega: wp.bool,
    accelValue: Any, workValue: scalar_t,
):
    acc = zero_like_warp(accelValue)
    work = workValue
    for neighborIndex in range(numIndices):
        jj = beginIndex + neighborIndex
        j = wp.int32(offsetArray[jj])
        if kernelProperties.operationMode != wp.static(OperationDirection.TrueAllToToAll.value):
            if not checkDirectionality_j(referenceKinds[j], kernelProperties.operationMode):
                continue
        if i == j:
            continue
        xj, hj, mj, rhoj, kj = getParticle(referenceState, j)
        omega_j = scalar_t(1.0)
        if useOmega:
            omega_j = referenceOmegas[j]
        f_j = referencePressures[j] / (omega_j * rhoj * rhoj)
        gradWi = sphKernelGradient(xi, xj, hi, hi, kernelProperties, domainState)
        gradWj = sphKernelGradient(xi, xj, hj, hj, kernelProperties, domainState)
        acc -= mj * (f_i * gradWi + f_j * gradWj)
        work += mj * wp.dot(vel_i - referenceVelocities[j], gradWi)
    return acc, work


@wp.func
def perSide_Func_Adjacency(
    i: wp.int32,
    queryState: Any, referenceState: Any,
    domainState: domainData,
    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData, numOffsets: wp.int32,
    kernelProperties: kernelState,
    queryVelocities: wp.array(dtype=vector(length=Any, dtype=scalar_t)), referenceVelocities: wp.array(dtype=vector(length=Any, dtype=scalar_t)),  # type: ignore
    queryPressures: wp.array(dtype=scalar_t), referencePressures: wp.array(dtype=scalar_t),  # type: ignore
    queryOmegas: wp.array(dtype=scalar_t), referenceOmegas: wp.array(dtype=scalar_t),  # type: ignore
    useOmega: wp.bool,
    accelValue: Any,
):
    xi, hi, mi, rhoi, ki = getParticle(queryState, i)
    acc = zero_like_warp(accelValue)
    work = scalar_t(0.0)
    if kernelProperties.operationMode != wp.static(OperationDirection.TrueAllToToAll.value):
        if not checkDirectionality_i(ki, kernelProperties.operationMode):
            return acc, work
    omega_i = scalar_t(1.0)
    if useOmega:
        omega_i = queryOmegas[i]
    f_i = queryPressures[i] / (omega_i * rhoi * rhoi)
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
        a_o, w_o = perSide_Func_i(
            i, xi, hi, mi, rhoi, referenceState, domainState, kernelProperties,
            beginIndex, numIndices, adjacencyState.neighborList if useAdjacency else gridState.sortIndex,
            ki, referenceState.kinds,
            queryVelocities[i], referenceVelocities,
            f_i, referencePressures, referenceOmegas, useOmega,
            accelValue, scalar_t(0.0),
        )
        acc += a_o
        work += w_o
    return acc, f_i * work


@wp.kernel
def computePerSidePressure_Kernel(
    queryState: Any, referenceState: Any, domainState: domainData,
    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any,
    kernelProperties: kernelState,
    # Do not change the parameters above
    queryVelocities: wp.array(dtype=vector(length=Any, dtype=scalar_t)), referenceVelocities: wp.array(dtype=vector(length=Any, dtype=scalar_t)),  # type: ignore
    queryPressures: wp.array(dtype=scalar_t), referencePressures: wp.array(dtype=scalar_t),  # type: ignore
    queryOmegas: wp.array(dtype=scalar_t), referenceOmegas: wp.array(dtype=scalar_t),  # type: ignore
    useOmega: wp.bool,
    # outputs
    accel: wp.array(dtype=vector(length=Any, dtype=scalar_t)),  # type: ignore
    dudt: wp.array(dtype=scalar_t),  # type: ignore
):
    i = wp.tid()
    if i >= queryState.positions.shape[0]:
        return
    a, w = perSide_Func_Adjacency(
        i, queryState, referenceState, domainState,
        useAdjacency, adjacencyState, gridState, gridState.numOffsets if not useAdjacency else 1,
        kernelProperties,
        queryVelocities, referenceVelocities, queryPressures, referencePressures,
        queryOmegas, referenceOmegas, useOmega,
        accel[i],
    )
    accel[i] = a
    dudt[i] = w


def _accelDtype(ctx, extras):
    return castTorchToWarpAsBuiltins(ctx.query.positions).dtype


def _scalarDtype(ctx, extras):
    return castTorchToWarpAsBuiltins(ctx.query.densities).dtype


_PER_SIDE = OperatorSpec(
    kernel=computePerSidePressure_Kernel,
    outputs=(OutputSpec(dtype=_accelDtype, shape=ShapeOf.QUERY),
             OutputSpec(dtype=_scalarDtype, shape=ShapeOf.QUERY)),
    extras=(
        ExtraSpec("queryVelocities", ExtraKind.TENSOR),
        ExtraSpec("referenceVelocities", ExtraKind.TENSOR),
        ExtraSpec("queryPressures", ExtraKind.TENSOR),
        ExtraSpec("referencePressures", ExtraKind.TENSOR),
        ExtraSpec("queryOmegas", ExtraKind.TENSOR),
        ExtraSpec("referenceOmegas", ExtraKind.TENSOR),
        ExtraSpec("useOmega", ExtraKind.SCALAR),
    ),
)


def computePerSidePressureWarp(
    queryParticles: ParticleState,
    operationProperties: OperationProperties,
    domain: DomainDescription,
    adjacency: Optional[Union[AdjacencyList, CompactHashMap]] = None,
    queryOmegas: Optional[torch.Tensor] = None,
    queryVelocities: Optional[torch.Tensor] = None,
    queryPressures: Optional[torch.Tensor] = None,
    referenceParticles: Optional[ParticleState] = None,
):
    """`(a_i, du_i/dt)` of the per-side pressure pair. `queryOmegas=None` -> Omega = 1 (no grad-h)."""
    referenceParticles = referenceParticles if referenceParticles is not None else queryParticles
    vel = queryVelocities if queryVelocities is not None else queryParticles.velocities
    P = queryPressures if queryPressures is not None else queryParticles.pressures
    useOmega = queryOmegas is not None
    omegas = queryOmegas if useOmega else torch.ones_like(P)
    with record_function("warpSPH[computePerSidePressure]"):
        ctx = SPHContext(
            query=queryParticles, properties=operationProperties, domain=domain,
            adjacency=adjacency, reference=referenceParticles,
            corrections=Corrections(volumes=(None, None), crk=None, gradH=None, renorm=None),
        )
        return launchOperator(
            _PER_SIDE, ctx,
            queryVelocities=vel, referenceVelocities=vel,
            queryPressures=P, referencePressures=P,
            queryOmegas=omegas, referenceOmegas=omegas,
            useOmega=useOmega,
        )
