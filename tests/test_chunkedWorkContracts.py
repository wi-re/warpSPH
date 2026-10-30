"""Compatible-work storage, unsupported-context gates and delayed stream producers."""
from dataclasses import replace
import os
from types import SimpleNamespace as NS

import pytest
import torch

from warpSPHCore import DomainDescription, ParticleState, radiusSearchCompactHashMap
from warpSPHCore.radiusSearch import radiusSearchRowChunks, ChunkedPairField, RowChunkPolicy
from warpSPHCore.type_config import get_torch_precision
from warpSPH.modules.compSPH.workReduction import reducePairWork, reduceReferenceEntropyWork
from warpSPH.modules.compSPH.chunkedWork import checkChunkedStage
from warpSPH.modules.compSPH.pairWork import beginPairWork
from warpSPH.configurations.compSPHConfig import CompSPHConfig
from warpSPH.configurations.crkSPH import CRKSPHConfig
from warpSPH.modules.crk.terms import computeCRKTerms


def sample(n=9):
    device = torch.device(os.environ.get('WARPSPH_TEST_DEVICE', 'cpu'))
    dtype = get_torch_precision()
    t = torch.arange(n, device=device, dtype=dtype)
    x = torch.stack((t*.05+.1, t*.03+.15), 1)
    p = ParticleState(x, t.new_full((n,), .8), torch.ones_like(t),
                      torch.zeros(n, device=device, dtype=torch.int32), torch.ones_like(t))
    d = DomainDescription(t.new_zeros(2), t.new_ones(2), torch.zeros(2, device=device, dtype=torch.bool), 2)
    policy = RowChunkPolicy(edgeBudget=40, maxRows=2, bytesPerEdge=4+2*t.element_size(), maxElementsPerEdge=2)
    chunks = radiusSearchRowChunks(p, d, policy=policy)
    flat = radiusSearchCompactHashMap(p, d, storeI=False)
    return p, d, chunks, flat


@pytest.mark.parametrize('n', [0, 9])
def test_reduction_ignores_padding_and_preserves_live_order(n):
    p, d, chunks, flat = sample(n)
    velocity = torch.arange(n*2, device=p.positions.device, dtype=p.positions.dtype).reshape(n, 2)
    work = ChunkedPairField.empty(chunks, trailingShape=(2,), dtype=p.positions.dtype)
    full = p.positions.new_full((len(flat.paddedJ), 2), float('nan'))
    full[:flat.numEdges] = torch.arange(flat.numEdges*2, device=full.device, dtype=full.dtype).reshape(-1, 2) * .03125
    for c, w in zip(chunks, work.buffers):
        w.fill_(float('nan'))
        m = c.metadata
        w[:m.liveEdges] = full[m.edgeStart:m.edgeStart+m.liveEdges]
    torch.testing.assert_close(reducePairWork(chunks, velocity, work), reducePairWork(flat, velocity, full), rtol=0, atol=0)
    with pytest.raises(ValueError, match='generation'):
        reducePairWork(chunks, velocity, replace(work, adjacency=replace(chunks)))
    if n:
        with pytest.raises(NotImplementedError, match='AD'):
            reducePairWork(chunks, velocity.clone().requires_grad_(), work)
    assert work.memoryUsage()['capacityBytes'] == sum(w.untyped_storage().nbytes() for w in work.buffers)


@pytest.mark.parametrize('family', ['CRK', 'Comp'])
@pytest.mark.parametrize('control', ['expanded', 'graph', 'materials', 'walls', 'ad', 'budget'])
def test_unqualified_contexts_fail_before_pair_allocation(family, control):
    p, d, chunks, _ = sample(3)
    sc = CRKSPHConfig() if family == 'CRK' else CompSPHConfig()
    config = NS(domain=d, compute=NS(execution='eager'), _hasBoundaryParticles=False)
    if control == 'expanded':
        setattr(sc, family.lower()+'PairStorage', 'expanded')
    elif control == 'graph':
        config.compute.execution = 'graph'
    elif control == 'materials':
        sc.materials = [object()]
    elif control == 'walls':
        config._hasBoundaryParticles = True
    elif control == 'ad':
        p = replace(p, masses=p.masses.clone().requires_grad_())
    else:
        chunks = replace(chunks, plan=replace(chunks.plan, policy=RowChunkPolicy(edgeBudget=40)))
    system = NS(pairWork=beginPairWork((.5, .5)), adjacency=chunks)
    with pytest.raises((NotImplementedError, ValueError)):
        checkChunkedStage(system, p, config, sc, 0, .001, family)


def test_nondefault_stream_delayed_terms_and_pair_reductions():
    if not torch.cuda.is_available() or os.environ.get('WARPSPH_TEST_DEVICE', 'cpu') != 'cuda':
        pytest.skip('CUDA required')
    p, d, chunks, flat = sample()
    n = len(p.positions)
    velocity = torch.zeros_like(p.positions)
    target = torch.arange(n*2, device=velocity.device, dtype=velocity.dtype).reshape(n, 2)
    work = ChunkedPairField.empty(chunks, trailingShape=(2,), dtype=velocity.dtype)
    full = velocity.new_zeros((len(flat.paddedJ), 2))
    for w in work.buffers:
        w.zero_()
    entropy = velocity.new_ones(n)
    expected = []
    for i in range(n):
        start, count = int(flat.edgeOffsets[i]), int(flat.numNeighbors[i])
        j = flat.paddedJ[start:start+count].long()
        expected.append((target[j]-target[i]).sum())
    expected = torch.stack(expected)
    m0 = velocity.new_full((n,), 2.)
    m1 = torch.zeros_like(velocity)
    m2 = torch.eye(2, dtype=velocity.dtype, device=velocity.device).repeat(n, 1, 1)
    dm0, dm1 = m1.clone(), torch.zeros_like(m2)
    dm2 = velocity.new_zeros(n, 2, 2, 2)
    counts = torch.full((n,), 9, device=velocity.device, dtype=torch.int32)
    # Warm all Warp specializations before deliberately delaying the producer.
    computeCRKTerms(m0,m1,m2,dm0,dm1,dm2,counts,p.supports)
    reducePairWork(flat, velocity, full)
    reducePairWork(chunks, velocity, work)
    reduceReferenceEntropyWork(flat, velocity, full, entropy)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        torch.cuda._sleep(10_000_000)
        m0.fill_(4.)
        velocity.copy_(target)
        full[:flat.numEdges].fill_(1.)
        for c,w in zip(chunks,work.buffers):
            w[:c.metadata.liveEdges].fill_(1.)
            w[c.metadata.liveEdges:].fill_(float('nan'))
        A = computeCRKTerms(m0,m1,m2,dm0,dm1,dm2,counts,p.supports)[0].clone()
        dense = reducePairWork(flat, velocity, full).clone()
        owned = reducePairWork(chunks, velocity, work).clone()
        weighted = reduceReferenceEntropyWork(flat, velocity, full, entropy).clone()
    stream.synchronize()
    torch.testing.assert_close(A, torch.full_like(A,.25), rtol=0, atol=0)
    torch.testing.assert_close(dense, expected, rtol=0, atol=0)
    torch.testing.assert_close(owned, expected, rtol=0, atol=0)
    torch.testing.assert_close(weighted, .5*expected, rtol=0, atol=0)

    # The repaired flat helpers also remain usable inside an existing graph.
    from warpSPHCore.autograd.stream import torchWarpStream, torchWarpCapture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torchWarpStream(velocity.device):
        with torchWarpCapture(graph):
            capturedA = computeCRKTerms(m0,m1,m2,dm0,dm1,dm2,counts,p.supports)[0]
            capturedWork = reducePairWork(flat, velocity, full)
            capturedEntropy = reduceReferenceEntropyWork(flat, velocity, full, entropy)
            with pytest.raises(NotImplementedError, match='capture'):
                reducePairWork(chunks, velocity, work)
        m0.fill_(8.)
        velocity.mul_(2.)
        graph.replay()
    stream.synchronize()
    torch.testing.assert_close(capturedA, torch.full_like(capturedA,.125), rtol=0, atol=0)
    torch.testing.assert_close(capturedWork, 2*expected, rtol=0, atol=0)
    torch.testing.assert_close(capturedEntropy, expected, rtol=0, atol=0)


def test_chunked_checkpoint_export_rejects_before_creating_a_file(tmp_path):
    p, _, chunks, _ = sample(3)
    from warpSPH.io.export import exportSimulationSystem
    from warpSPH.io.hdf5 import dumpAdjacency
    from warpSPH.enumTypes import CompressibleSPHScheme
    target = tmp_path/'uncreated'
    with pytest.raises(NotImplementedError, match='checkpoint export'):
        exportSimulationSystem(target, 'state', CompressibleSPHScheme.CRKSPH,
                               NS(adjacency=chunks, state=p, t=0.))
    assert not target.exists()
    with pytest.raises(NotImplementedError, match='chunk-local'):
        dumpAdjacency(chunks, None)
