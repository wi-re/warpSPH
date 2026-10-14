"""The CRK correction terms A, B, grad A, grad B (Frontiere et al. 2017, Eqs. 12-13 and their derivatives) from the kernel moments, as one warp
kernel, and the factors pipeline around it (`crkFactors`).

`warpSPHCore.crk.computeCRKTermsWarp` is the same algebra written in torch (~290 ops per call and two host reads: `torch.linalg.pinv` and an
`if mask.any(): print(...)` warning). That version is kept for what needs a tape -- `computeCRKFactorsJVP` differentiates through it with
`torch.autograd.functional.jvp` -- and `computeCRKTerms` falls back to it whenever an input requires grad. The kernel is one thread per
particle on small dense matrices (d <= 3), the pseudo-inverse of the symmetric moment `m_2` from `warpSPH.linalg` (torch's `d * eps`
cut-off), no host read:

    M = pinv(m_2),  mv = M m_1
    A      = 1 / (m_0 - m_1 . mv)                                          (Eq. 12)
    B      = -mv                                                           (Eq. 13)
    gradA_k = -A^2 (dm_0_k - 2 sum_i dm_1_ki mv_i + sum_lm mv_l dm_2_klm mv_m)
    gradB_ki = -sum_j M_ij dm_1_kj + sum_lm M_il dm_2_klm mv_m

with `k` the differentiation axis (the layout `crk/kernel.py` documents). A particle with fewer than two neighbours or an (absolute)
`det m_2` below 1e-14 gets the identity correction (A = 1, the rest 0), as before. The torch version prints a warning when that happens; the
kernel does not (it would be a host read per call).
"""

import torch
import warp as wp

from warpSPHCore import scalar_t
from warpSPHCore.crk import computeCRKTermsWarp as computeCRKTermsTorch

from ...linalg import loadPadded3x3, symmetricPinv3

__all__ = ['computeCRKTerms', 'crkFactors']


@wp.func
def _det(M: wp.mat33f if scalar_t == wp.float32 else wp.mat33d, dim: int):
    d = M[0, 0]
    if dim == 2:
        d = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    elif dim == 3:
        d = wp.determinant(M)
    return d


@wp.kernel
def _crkTermsKernel(m0: wp.array(dtype=scalar_t), m1: wp.array2d(dtype=scalar_t), m2: wp.array2d(dtype=scalar_t),
                    dm0: wp.array2d(dtype=scalar_t), dm1: wp.array2d(dtype=scalar_t), dm2: wp.array2d(dtype=scalar_t),
                    numNeighbors: wp.array(dtype=wp.int32), dim: int, rtol: scalar_t,
                    A: wp.array(dtype=scalar_t), B: wp.array2d(dtype=scalar_t), gradA: wp.array2d(dtype=scalar_t),
                    gradB: wp.array2d(dtype=scalar_t)):
    i = wp.tid()
    M2 = loadPadded3x3(m2, i, dim)
    Minv = symmetricPinv3(M2, rtol)
    mv = wp.vector(scalar_t(0.0), scalar_t(0.0), scalar_t(0.0))
    for a in range(3):
        if a < dim:
            for b in range(3):
                if b < dim:
                    mv[a] = mv[a] + Minv[a, b] * m1[i, b]
    denom = m0[i]
    for a in range(3):
        if a < dim:
            denom = denom - m1[i, a] * mv[a]
    Ai = scalar_t(1.0) / denom
    singular = wp.abs(_det(M2, dim)) < scalar_t(1.0e-14)
    if singular or numNeighbors[i] < 2:
        A[i] = scalar_t(1.0)
        for k in range(3):
            if k < dim:
                B[i, k] = scalar_t(0.0)
                gradA[i, k] = scalar_t(0.0)
                for a in range(3):
                    if a < dim:
                        gradB[i, k * dim + a] = scalar_t(0.0)
        return
    A[i] = Ai
    for a in range(3):
        if a < dim:
            B[i, a] = -mv[a]
    for k in range(3):
        if k < dim:
            t2 = scalar_t(0.0)
            t4 = scalar_t(0.0)
            for a in range(3):
                if a < dim:
                    t2 = t2 + dm1[i, k * dim + a] * mv[a]
                    for b in range(3):
                        if b < dim:
                            t4 = t4 + mv[a] * dm2[i, (k * dim + a) * dim + b] * mv[b]
            gradA[i, k] = -Ai * Ai * (dm0[i, k] - scalar_t(2.0) * t2 + t4)
            for a in range(3):
                if a < dim:
                    g = scalar_t(0.0)
                    for b in range(3):
                        if b < dim:
                            g = g - Minv[a, b] * dm1[i, k * dim + b]
                            for c in range(3):
                                if c < dim:
                                    g = g + Minv[a, b] * dm2[i, (k * dim + b) * dim + c] * mv[c]
                    gradB[i, k * dim + a] = g


def computeCRKTerms(m_0, m_1, m_2, dm_0dgamma, dm_1dgamma, dm_2dgamma, num_nbrs, supports):
    """`computeCRKTermsWarp`'s `(A, B, gradA, gradB)`; the kernel unless an input needs a tape (then the torch version)."""
    if any(t.requires_grad for t in (m_0, m_1, m_2, dm_0dgamma, dm_1dgamma, dm_2dgamma)) or m_0.shape[0] == 0:
        return computeCRKTermsTorch(m_0, m_1, m_2, dm_0dgamma, dm_1dgamma, dm_2dgamma, num_nbrs, supports)
    n, dim = m_1.shape
    A = torch.empty_like(m_0)
    B, gradA = torch.empty_like(m_1), torch.empty_like(m_1)
    gradB = torch.empty(n, dim, dim, dtype=m_0.dtype, device=m_0.device)
    w = lambda t, shape=None: wp.from_torch((t if shape is None else t.reshape(shape)).contiguous())
    nn = num_nbrs if num_nbrs.dtype == torch.int32 else num_nbrs.to(torch.int32)
    wp.launch(_crkTermsKernel, dim=n, device=wp.device_from_torch(m_0.device),
              inputs=[w(m_0), w(m_1), w(m_2, (n, dim * dim)), w(dm_0dgamma), w(dm_1dgamma, (n, dim * dim)),
                      w(dm_2dgamma, (n, dim ** 3)), w(nn), dim, dim * torch.finfo(m_0.dtype).eps,
                      w(A), w(B), w(gradA), w(gradB.view(n, dim * dim))])
    return A, B, gradA, gradB


def crkFactors(queryParticles, domain, kernel, adjacency=None):
    """`warpSPHCore.crk.computeCRKFactors` with the terms from the kernel: `(apparentVolume, crkDensity, CRKState)`."""
    from warpSPHCore import CRKState, OperationDirection, OperationProperties, SupportScheme
    from warpSPHCore.crk.crk_density import _computeCRKDensity_stateBackend
    from warpSPHCore.crk.crk_moments import _computeCRKMoments_stateBackend
    from warpSPHCore.crk.crk_volume import _computeCRKVolume_stateBackend
    mode = OperationDirection.AllToAll
    apparentArea = _computeCRKVolume_stateBackend(
        queryParticles, OperationProperties(kernel=kernel, supportMode=SupportScheme.Gather, operationMode=mode), domain,
        adjacency=adjacency, referenceParticles=None)
    m_0, m_1, m_2, dm_0, dm_1, dm_2, numNeighbors = _computeCRKMoments_stateBackend(
        queryParticles, OperationProperties(kernel=kernel, supportMode=SupportScheme.Scatter, operationMode=mode), domain,
        queryVolumes=apparentArea, referenceVolumes=apparentArea, adjacency=adjacency, referenceParticles=None)
    A, B, gradA, gradB = computeCRKTerms(m_0, m_1, m_2, dm_0, dm_1, dm_2, numNeighbors, queryParticles.supports)
    state = CRKState(A=A, B=B, gradA=gradA, gradB=gradB)
    crkDensity = _computeCRKDensity_stateBackend(
        queryParticles, OperationProperties(kernel=kernel, supportMode=SupportScheme.Scatter, operationMode=mode), domain,
        crkState=state, queryVolumes=apparentArea, referenceVolumes=apparentArea, adjacency=adjacency, referenceParticles=None)
    return apparentArea, crkDensity, state
