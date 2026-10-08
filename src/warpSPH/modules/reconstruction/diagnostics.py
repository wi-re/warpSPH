"""Diagnostics of the reconstruction: the per-particle mean of the pair factor `phi_ij` that scales the midpoint
extrapolation (AV_PLAN Phase 4 "detector map": the `phi` field next to the alpha field).

`phi_ij` here is the factor the policy applies -- 1 for `Linear`, the limited value for `Limited`, times
`1 - Bbar^p` for `BalsaraLimited`, 0 for `Raw` -- evaluated with the same `wp.func`s the viscosity kernels call, on
the Verlet list of the last step (pairs beyond the support included, as in the kernels; they carry no weight there).
Off the hot path: only costs anything when called.
"""

from __future__ import annotations

from typing import Any

import torch
import warp as wp
from warp.types import matrix, vector

from warpSPHCore import *
from warpSPHCore.coreOperations._jvpCommon import buildDomainState
from warpSPHCore.util import castTorchToWarp, castTorchToWarpAsBuiltins

from ...configurations.moduleConfigurations.diffusionParameters import VelocityPairPolicy
from .gradient import reconstructionInputs
from .pairVelocity import limitedPairPhi

__all__ = ['computePairPhiMean']


@wp.kernel
def _pairPhiKernel(
    I: wp.array(dtype=wp.int64), J: wp.array(dtype=wp.int64),
    x: wp.array(dtype=vector(length=Any, dtype=scalar_t)), v: wp.array(dtype=vector(length=Any, dtype=scalar_t)),  # type: ignore
    h: wp.array(dtype=scalar_t), grad: wp.array(dtype=matrix(shape=(Any, Any), dtype=scalar_t)),  # type: ignore
    B: wp.array(dtype=scalar_t),
    domainState: domainData, policy: wp.int32, kernel_int: wp.int32,
    eta_crit: scalar_t, eta_fold: scalar_t, balsaraPower: scalar_t,
    out: wp.array(dtype=scalar_t),
):
    k = wp.tid()
    i = wp.int32(I[k])
    j = wp.int32(J[k])
    x_ij = computeDistanceVec(x[i], x[j], domainState)
    phi = scalar_t(1.0)
    if policy != wp.static(VelocityPairPolicy.Linear.value):
        phi = limitedPairPhi(x_ij, h[i], h[j], v[i], v[j], grad[i], grad[j], kernel_int, domainState.dim,
                             True, True, eta_crit, eta_fold)
        phi = wp.max(wp.min(phi, scalar_t(1.0)), scalar_t(0.0))
    if policy == wp.static(VelocityPairPolicy.BalsaraLimited.value):
        B_bar = scalar_t(0.5) * (B[i] + B[j])
        if B_bar > scalar_t(0.0):
            phi = phi * (scalar_t(1.0) - wp.pow(B_bar, balsaraPower))
    out[k] = phi


def computePairPhiMean(system, simulationConfig, diffusionParams) -> torch.Tensor:
    """Per-particle mean of `phi_ij` over its Verlet-list neighbours (self pair excluded); zeros for `Raw`."""
    st, adj = system.state, system.adjacency
    n = st.positions.shape[0]
    if diffusionParams.velocityPairPolicy == VelocityPairPolicy.Raw.value:
        return torch.zeros(n, dtype=st.positions.dtype, device=st.positions.device)
    params, grad, B = reconstructionInputs(st, simulationConfig, diffusionParams, adj)
    if B is None:
        B = torch.zeros_like(st.supports)
    dev = str(st.positions.device)
    nPairs = adj.i.shape[0]
    phi = wp.zeros(nPairs, dtype=scalar_t, device=dev)
    wp.launch(_pairPhiKernel, dim=nPairs, device=dev, inputs=[
        castTorchToWarp(adj.i.to(torch.int64)), castTorchToWarp(adj.j.to(torch.int64)),
        castTorchToWarpAsBuiltins(st.positions), castTorchToWarpAsBuiltins(st.velocities),
        castTorchToWarp(st.supports), castTorchToWarpAsBuiltins(grad.contiguous()), castTorchToWarp(B.contiguous()),
        buildDomainState(simulationConfig.domain), int(params.velocityPairPolicy), simulationConfig.kernel.value,
        float(params.reconstructionEtaCrit), float(params.reconstructionEtaFold), float(params.reconstructionBalsaraPower),
        phi])
    phi = wp.to_torch(phi)
    notSelf = (adj.i != adj.j).to(phi.dtype)
    # deterministic per-row sums (no atomics): the Verlet list is row-sorted
    rows = adj.i.to(torch.int64)
    counts = torch.bincount(rows, minlength=n)
    sums = torch.segment_reduce(phi * notSelf, 'sum', lengths=counts)
    nb = torch.segment_reduce(notSelf, 'sum', lengths=counts)
    return sums / nb.clamp_min(1)
