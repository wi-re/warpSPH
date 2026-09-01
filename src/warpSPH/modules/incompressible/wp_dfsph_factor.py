"""DFSPH pressure-stiffness denominator (Bender & Koschier 2015/2017), as
implemented by SPlisHSPlasH's `TimeStepDFSPH.cpp::computeDFSPHFactor`.

This is the Jacobi step-size denominator for the two DFSPH pressure solves.
It is *not* the IISPH diagonal that `computeAlpha` (`wp_alpha.py`) returns,
even though the two agree in the bulk:

- `computeAlpha` (IISPH `a_ii`) is
  `areaI/mi * |sum_j V_j gradW_ij|^2 + areaI * sum_j (V_j^2/m_j) |gradW_ij|^2`,
  i.e. every term is re-weighted by the query particle's apparent area
  `areaI = V_i` and (in the second sum) by `1/m_j`.
- the DFSPH factor is
  `sum_j |V_j gradW_ij|^2 + |sum_j V_j gradW_ij|^2`.

  The gate on the *first* (sum-of-squares / back-reaction) term is on the
  QUERY kind (`ki == 0`), not the neighbour kind: a fluid particle accumulates
  it over **all** non-ghost neighbours (fluid AND boundary), a boundary/ghost
  query gets 0 (it is not an unknown in the solve, so only the vector term is
  ever read from it). This is a deliberate departure from SPlisHSPlasH's
  `computeDFSPHFactor` and Bender-Westhofen-Jeske 2023 Eq. 32, which restrict
  the sum-of-squares to fluid neighbours (a *static* boundary particle takes
  no reaction). Folding the boundary into it instead treats the wall as the
  "static fluid at rho0" that `applyConsistentCoupling` already models it as,
  and gives near-wall fluid a larger denominator -> smaller `alpha` -> gentler
  pressure updates at the wall (same intent as `akinciBoundaryVolumeScale`,
  DFSPH_IMPROVEMENT_PLAN.md Part 24/25). In the bulk (no boundary neighbours)
  it is identical to the reference form.

For a uniform fluid (`rho_i = rho0 = 1`, equal masses) the two coincide, which
is why `diag(A)/alphas ~ 1.0001` was measured in the bulk
(`DFSPH_IMPROVEMENT_PLAN.md` 2). They diverge at a wall, where the boundary
carries an Akinci apparent volume (`applyConsistentCoupling` substitutes it
into `state.masses`) and the IISPH `areaI/m_i` / `1/m_j` weightings stop being
`1`. This kernel returns the DFSPH form (bulk-identical to SPlisHSPlasH's;
near-wall it folds the boundary into the back-reaction term, see above).

The apparent volumes `V_j = m_j/rho_j` are supplied by the caller as
`referenceApparentAreas`; for boundary rows the caller must run the kernel
inside `applyConsistentCoupling` so `state.masses` already carries the Akinci
volume (this is how `dfsphReference` invokes it). Ghosts (`kind == 2`) are
excluded from both sums by the `OperationDirection.AllToAll` directionality
filter (`checkDirectionality_j(kind, 9) == (kind != 2)`), mirroring
`computeAlpha`.
"""

import warp as wp
from warp.types import vector, matrix
from typing import Any
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from typing import Optional, Union, Tuple
from warpSPHCore import *

from warpSPH.enumTypes import *
from warpSPH.configurations.simulationConfig import SimulationConfig

__all__ = ['computeDFSPHFactor']


@wp.func
def computeDFSPHFactor_Func_i_first(
    # General Shape Parameters and indices
    i: wp.int32, dim: wp.int32,

    # SPH properties for the query set (indexed by i)
    xi: vector(dtype=scalar_t, length=Any), hi: scalar_t, mi: scalar_t, rhoi: scalar_t,  # type: ignore

    # SPH properties for the reference set (indexed by j in the neighbor loop)
    referenceState: Any,  # particleDataSoA, dimension-specific

    # Domain and kernel parameters
    domainState: domainData,
    kernelProperties: kernelState,

    # Operation specific parameters
    beginIndex: wp.int32,  # type: ignore
    numIndices: wp.int32,  # type: ignore
    offsetArray: wp.array(dtype=wp.int64),  # type: ignore

    # Operation Mode for masking certain kinds of interactions
    ki: wp.int32, referenceKinds: wp.array(dtype=wp.int32),  # type: ignore

    # Optional Correction Terms
    useGradientRenormalization: wp.bool, Li: matrix(shape=(Any, Any), dtype=scalar_t),  # type: ignore
    useGradHTerms: wp.bool, omega_i: scalar_t, referenceOmegas: wp.array(dtype=scalar_t),  # type: ignore
    useVolume: bool, Vi: scalar_t, referenceVolumes: wp.array(dtype=scalar_t),  # type: ignore
    useCRK: bool, Ai: scalar_t, Bi: vector(length=Any, dtype=scalar_t), gradAi: vector(length=Any, dtype=scalar_t), gradBi: matrix(shape=(Any, Any), dtype=scalar_t),  # type: ignore

    # Apparent volumes V_j = m_j / rho_j (boundary rows carry the Akinci volume
    # when the caller runs this inside applyConsistentCoupling).
    referenceApparentAreas: wp.array(dtype=scalar_t),  # type: ignore
):
    # Vector sum over ALL non-ghost neighbours (fluid + boundary): the DFSPH
    # second term |sum_j V_j gradW_ij|^2. SPlisHSPlasH's `grad_p_i`.
    sumA = zero_like_warp(xi)
    # Sum of |V_j gradW_ij|^2 -- the back-reaction term. Gated on the QUERY
    # kind below (`ki == 0`): a fluid query accumulates it over every non-ghost
    # neighbour (fluid AND boundary); a boundary/ghost query leaves it at 0
    # (only the vector term is ever read from a non-fluid row). See the module
    # docstring for why this differs from the fluid-neighbours-only reference.
    sumSq = zero_like_warp(rhoi)

    for neighborIndex in range(numIndices):
        jj = beginIndex + neighborIndex
        j = wp.int32(offsetArray[jj])
        if kernelProperties.operationMode != wp.static(OperationDirection.TrueAllToToAll.value):
            if not checkDirectionality_j(referenceKinds[j], kernelProperties.operationMode):
                continue

        xj, hj, mj, rhoj, kj = getParticle(referenceState, j)

        gradw_ij = computeKernelGradientCRK(
            xi, xj,
            hi, hj,
            kernelProperties, domainState,
            useCRK, Ai, Bi, gradAi, gradBi
        )
        if useGradientRenormalization:
            gradw_ij = matmul(Li, gradw_ij)

        volumeJ = referenceApparentAreas[j]
        gradw_ij2 = wp.dot(gradw_ij, gradw_ij)
        term = volumeJ * gradw_ij

        # Vector sum: fluid AND boundary (SPlisHSPlasH `grad_p_i -= grad_p_j`
        # for both, with grad_p_j = -V_j gradW).
        sumA += term
        # Back-reaction sum: query-kind gated. `ki` is the query kind and is
        # constant across the loop, so a fluid query (`ki == 0`) accumulates
        # over every neighbour reached here (fluid + boundary; ghosts already
        # filtered by the directionality check above), and a non-fluid query
        # accumulates nothing. Bare apparent volume, no /mj, no areaI.
        if ki == 0:
            sumSq += volumeJ * volumeJ * gradw_ij2

    return sumSq + wp.dot(sumA, sumA)


@wp.func
def computeDFSPHFactor_Func_Adjacency_first(
    i: wp.int32, dim: wp.int32,

    queryState: Any,
    referenceState: Any,
    correctionData: Any,

    domainState: domainData,
    useAdjacency: wp.bool,
    adjacencyState: adjacencyData,
    gridState: gridData,
    numOffsets: wp.int32,

    kernelProperties: kernelState,

    referenceApparentAreas: wp.array(dtype=scalar_t),  # type: ignore
):
    xi, hi, mi, rhoi, ki = getParticle(queryState, i)
    if kernelProperties.operationMode != wp.static(OperationDirection.TrueAllToToAll.value):
        if not checkDirectionality_i(ki, kernelProperties.operationMode):
            return zero_like_warp(queryState.densities)

    useGradientRenormalization, Li = getL_i(correctionData, i)
    useGradHTerms, omega_i = getGradH_i(correctionData, i)
    useVolume, Vi = getVolume_i(correctionData, i)
    useCRK, Ai, Bi, gradA_i, gradB_i = getCRK_i(correctionData, i)

    out = zero_like_warp(queryState.densities)
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

        out += computeDFSPHFactor_Func_i_first(
            i, dim,
            xi, hi, mi, rhoi,
            referenceState, domainState,
            kernelProperties,

            beginIndex, numIndices, adjacencyState.neighborList if useAdjacency else gridState.sortIndex,
            ki, referenceState.kinds,

            useGradientRenormalization, Li,
            useGradHTerms, omega_i, correctionData.referenceOmegas,
            useVolume, Vi, correctionData.referenceVolumes,
            useCRK, Ai, Bi, gradA_i, gradB_i,

            referenceApparentAreas,
        )
    return out


@wp.kernel
def computeDFSPHFactor_Kernel(
    queryState: Any,
    referenceState: Any,
    domainState: domainData,

    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any,

    kernelProperties: kernelState,
    referenceApparentAreas: wp.array(dtype=scalar_t),  # type: ignore

    # The last parameter is always the output array
    outputValues: wp.array(dtype=scalar_t)  # type: ignore
):
    i = wp.tid()
    numParticles = queryState.positions.shape[0]
    if i >= numParticles:
        return

    outputValues[i] = computeDFSPHFactor_Func_Adjacency_first(
        i, domainState.dim,
        queryState, referenceState, correctionData, domainState,
        useAdjacency, adjacencyState, gridState, gridState.numOffsets if not useAdjacency else 1,
        kernelProperties,
        referenceApparentAreas,
    )


def _dfsphFactorDtype(ctx, extras):
    return castTorchToWarpAsBuiltins(ctx.query.densities).dtype


_DFSphFACTOR = OperatorSpec(
    kernel=computeDFSPHFactor_Kernel,
    outputs=(OutputSpec(dtype=_dfsphFactorDtype, shape=ShapeOf.QUERY),),
    extras=(
        ExtraSpec("referenceApparentAreas", ExtraKind.TENSOR),
    ),
)


def computeDFSPHFactorWarp(
    queryParticles: ParticleState,
    operationProperties: OperationProperties,
    domain: DomainDescription,

    referenceApparentAreas: torch.Tensor,
    adjacency: Optional[Union[AdjacencyList, CompactHashMap]] = None,
    referenceParticles: Optional[ParticleState] = None,
    crkState: Optional[CRKState] = None,
    gradHState: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], GradHState]] = None,
    renormalizationState: Optional[Union[torch.Tensor, RenormalizationState]] = None,
):
    with record_function("warpSPH[computeDFSPHFactor]"):
        referenceParticles = referenceParticles if referenceParticles is not None else queryParticles

        ctx = SPHContext(
            query=queryParticles, properties=operationProperties, domain=domain,
            adjacency=adjacency, reference=referenceParticles,
            corrections=Corrections(
                volumes=(None, None),
                crk=crkState, gradH=gradHState, renorm=renormalizationState,
            ),
        )
        return launchOperator(
            _DFSphFACTOR, ctx,
            referenceApparentAreas=referenceApparentAreas,
        )


def computeDFSPHFactor(currentState: Any, config: SimulationConfig, schemeConfig: Any, adjacency: Optional[Union[AdjacencyList, CompactHashMap]], apparentVolumes: torch.Tensor) -> torch.Tensor:
    """Positive DFSPH stiffness denominator `sum_grad_p_k` (see module
    docstring). Callers take the reciprocal / sign as their Jacobi step size
    needs. Mirrors `computeAlpha`'s launch setup (Density operation, Gather
    support) because it is a gather-style per-particle neighbour reduction."""
    with record_function("[warpSPH] - computeDFSPHFactor"):
        return computeDFSPHFactorWarp(
            queryParticles=currentState,
            operationProperties=OperationProperties(
                kernel=config.kernel,
                operation=WarpOperation.Density,
                supportMode=SupportScheme.Gather,
            ),
            domain=config.domain,
            adjacency=adjacency,
            referenceApparentAreas=apparentVolumes,
        )
