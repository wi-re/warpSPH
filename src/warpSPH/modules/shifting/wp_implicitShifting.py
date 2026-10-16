"""Per-pair (edge-parallel) kernel evaluator for implicit particle shifting:
launches one thread per neighbor pair `(i, j)` -- unlike the rest of warpSPH,
which launches one thread per query particle and loops over neighbors inside
the kernel -- because the implicit shift's matrix-free BiCGStab solve
(`implicitShifting.computeImplicitShift`) needs the raw per-pair kernel
value/gradient/Hessian as flat, edge-indexed torch tensors to drive its own
`scatter_sum`-based sparse matvec, rather than a per-particle accumulated
sum. `OperatorSpec`/`launchOperator` only supports per-query-particle thread
counts (see `warpSPHCore.autograd.operator_spec.ThreadSpec`), so this bypasses
that machinery with a hand-rolled `wp.kernel` + `wp.launch`, following the
calling convention of `warpSPHCore.radiusSearch.small.wp_radius_small`.
"""

from warpSPHCore.type_config import index_t, offset_t

from typing import Any
import warp as wp
from warp.types import vector, matrix
from warpSPHCore import *

__all__ = ['computeShiftingPairTerms']

_SoA_BY_DIM = {1: particleDataSoA_1, 2: particleDataSoA_2, 3: particleDataSoA_3}


@wp.kernel
def computeShiftingPairTerms_Kernel(
    queryState: Any,
    domainState: domainData,
    kernelProperties: kernelState,
    edgeI: wp.array(dtype=index_t),
    edgeJ: wp.array(dtype=index_t),
    outK: wp.array(dtype=scalar_t),
    outJ: wp.array(dtype=Any),
    outH: wp.array(dtype=Any),
):
    e = wp.tid()
    if e >= edgeI.shape[0]:
        return

    i = wp.int32(edgeI[e])
    j = wp.int32(edgeJ[e])

    xi, hi, mi, rhoi, ki = getParticle(queryState, i)
    xj, hj, mj, rhoj, kj = getParticle(queryState, j)

    outK[e] = sphKernel(xi, xj, hi, hj, kernelProperties, domainState)
    outJ[e] = sphKernelGradient(xi, xj, hi, hj, kernelProperties, domainState)
    outH[e] = sphKernelHessian(xi, xj, hi, hj, kernelProperties, domainState)


def _buildParticleSoA(state: ParticleState, dim: int):
    SoA = _SoA_BY_DIM[dim]()
    SoA.positions = castTorchToWarpAsBuiltins(state.positions)
    SoA.supports = castTorchToWarp(state.supports)
    SoA.masses = castTorchToWarp(state.masses)
    SoA.densities = castTorchToWarp(state.densities)
    SoA.kinds = castTorchToWarp(state.kinds)
    return SoA


def _buildDomainState(domain: DomainDescription) -> domainData:
    d = domainData()
    d.domainMin = castTorchToWarp(domain.min)
    d.domainMax = castTorchToWarp(domain.max)
    d.periodicity = castTorchToWarp(domain.periodic)
    d.dim = domain.dim
    return d


def _buildKernelState(kernel: KernelFunctions, supportMode: SupportScheme = SupportScheme.Gather) -> kernelState:
    k = kernelState()
    k.kernelFunction = kernel.value
    k.supportMode = supportSchemeToUint(supportMode)
    return k


def computeShiftingPairTerms(
    state: ParticleState,
    domain: DomainDescription,
    kernel: KernelFunctions,
    adjacency: AdjacencyList,
):
    """Per-pair kernel value `K`, gradient `J` and Hessian `H` (shapes
    `[numPairs]`, `[numPairs, dim]`, `[numPairs, dim, dim]`) for every
    `(i, j) = adjacency.livePairs()` pair, evaluated with gather support
    (`h = h_i`), matching diffSPH's `implicitShifting.evalKernel` convention.
    """
    dim = domain.dim
    liveI, liveJ = adjacency.livePairs()
    numPairs = liveI.shape[0]

    queryState = _buildParticleSoA(state, dim)
    domainState = _buildDomainState(domain)
    kernelProperties = _buildKernelState(kernel)

    edgeI = castTorchToWarp(liveI)
    edgeJ = castTorchToWarp(liveJ)

    K_t, K_w = allocateTorchWarp(numPairs, scalar_t, edgeI.device)
    J_t, J_w = allocateTorchWarp(numPairs, vector(length=dim, dtype=scalar_t), edgeI.device)
    H_t, H_w = allocateTorchWarp(numPairs, matrix(shape=(dim, dim), dtype=scalar_t), edgeI.device)

    wp.launch(
        computeShiftingPairTerms_Kernel,
        dim=numPairs,
        inputs=[queryState, domainState, kernelProperties, edgeI, edgeJ, K_w, J_w, H_w],
        device=edgeI.device,
    )

    return K_t, J_t, H_t
