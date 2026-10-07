"""`VelocityPairPolicy` (AV_PLAN Phase 1): the pair velocity `u_ij` is an argument of
`computePi_pair`; `computePi_actual` is the `RawVelocity` wrapper around it, and `rawPairVelocity`
is the identity `v_i - v_j`. Bit-identical, not approximately: the Phase 1 gate against M0."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import warp as wp
from warp.types import vector

from warpSPHCore import DomainDescription, scalar_t
from warpSPHCore.coreOperations._jvpCommon import buildDomainState
from warpSPHCore.dataTypes.domain_t import domainData
from warpSPHCore.util import castTorchToWarp, castTorchToWarpAsBuiltins

from warpSPH.configurations.moduleConfigurations.diffusionParameters import (
    DiffusionParameters, ViscosityTerms, buildDefaultDiffusionParamsCompressibleSPH)
from warpSPH.modules.dissipation.pi import computePi_actual, computePi_pair
from warpSPH.modules.reconstruction import rawPairVelocity

DIM = 2
vec = vector(length=DIM, dtype=scalar_t)


@wp.kernel
def _pairKernel(
    x_i: wp.array(dtype=vec), x_j: wp.array(dtype=vec),
    v_i: wp.array(dtype=vec), v_j: wp.array(dtype=vec),
    h: wp.array(dtype=scalar_t), rho: wp.array(dtype=scalar_t), c: wp.array(dtype=scalar_t),
    alpha: wp.array(dtype=scalar_t),
    domainState: domainData, params: DiffusionParameters, scale: scalar_t,
    outActual: wp.array(dtype=scalar_t), outPair: wp.array(dtype=scalar_t),
    outPairScaled: wp.array(dtype=scalar_t), outRaw: wp.array(dtype=vec),
):
    k = wp.tid()
    one = scalar_t(1.0)
    outActual[k] = computePi_actual(
        x_i[k], x_j[k], h[k], h[k], one, one, rho[k], rho[k] * scalar_t(1.3),
        False, one, one, v_i[k], v_j[k], domainState, 1, c[k], c[k] * scalar_t(0.9),
        alpha[k], alpha[k], params)
    u = v_i[k] - v_j[k]
    outPair[k] = computePi_pair(
        x_i[k], x_j[k], h[k], h[k], one, one, rho[k], rho[k] * scalar_t(1.3),
        False, one, one, u, domainState, 1, c[k], c[k] * scalar_t(0.9),
        alpha[k], alpha[k], params)
    outPairScaled[k] = computePi_pair(
        x_i[k], x_j[k], h[k], h[k], one, one, rho[k], rho[k] * scalar_t(1.3),
        False, one, one, u * scale, domainState, 1, c[k], c[k] * scalar_t(0.9),
        alpha[k], alpha[k], params)
    outRaw[k] = rawPairVelocity(v_i[k], v_j[k])


def _run(term: ViscosityTerms, monaghanSwitch: bool, n: int = 512):
    wp.init()
    dtype = torch.float64 if scalar_t is wp.float64 else torch.float32
    dev = 'cuda' if wp.get_cuda_device_count() else 'cpu'
    g = torch.Generator().manual_seed(0)

    def rnd(*shape, lo=0.0, hi=1.0):
        return (torch.rand(*shape, generator=g, dtype=dtype) * (hi - lo) + lo).to(dev)

    xi = rnd(n, DIM)
    xj = xi + rnd(n, DIM, lo=-0.05, hi=0.05)
    vi, vj = rnd(n, DIM, lo=-1, hi=1), rnd(n, DIM, lo=-1, hi=1)     # converging and diverging pairs
    params = buildDefaultDiffusionParamsCompressibleSPH()
    params.viscosityTerm = term.value
    params.C_q = 2.0
    params.monaghanSwitch = monaghanSwitch
    domain = DomainDescription(min=torch.full((DIM,), -10.0, dtype=dtype, device=dev),
                               max=torch.full((DIM,), 10.0, dtype=dtype, device=dev),
                               periodic=torch.tensor([False] * DIM, device=dev), dim=DIM)
    outs = [wp.zeros(n, dtype=scalar_t, device=dev) for _ in range(3)]
    outRaw = wp.zeros(n, dtype=vec, device=dev)
    wp.launch(_pairKernel, dim=n, device=dev, inputs=[
        castTorchToWarpAsBuiltins(xi), castTorchToWarpAsBuiltins(xj),
        castTorchToWarpAsBuiltins(vi), castTorchToWarpAsBuiltins(vj),
        castTorchToWarp(rnd(n, lo=0.05, hi=0.15)), castTorchToWarp(rnd(n, lo=0.5, hi=2.0)),
        castTorchToWarp(rnd(n, lo=0.5, hi=1.5)), castTorchToWarp(rnd(n, lo=0.1, hi=1.0)),
        buildDomainState(domain), params, scalar_t(-1.0)] + outs + [outRaw])
    wp.synchronize()
    return [wp.to_torch(o).clone() for o in outs], wp.to_torch(outRaw).clone(), (vi, vj)


@pytest.mark.parametrize('term', list(ViscosityTerms), ids=lambda t: t.name)
@pytest.mark.parametrize('monaghanSwitch', [True, False])
def test_actual_is_pair_on_the_raw_difference_bit_for_bit(term, monaghanSwitch):
    (actual, pair, scaled), raw, _ = _run(term, monaghanSwitch)
    assert torch.equal(actual, pair)
    assert torch.isfinite(actual).all()
    if monaghanSwitch:
        # not vacuous: the pair velocity is what the term reads (flipping u_ij swaps converging and
        # diverging pairs, which the Monaghan switch zeroes; some formulations read only its sign)
        assert not torch.equal(pair, scaled)


def test_raw_pair_velocity_is_the_identity_difference():
    _, raw, (vi, vj) = _run(ViscosityTerms.Price2012_98, True)
    assert torch.equal(raw, vi - vj)
