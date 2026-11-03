"""Tiny native compatible-work A/Bs; no videos for these bounded two-step fixtures."""
from dataclasses import replace
import importlib
import os
import json

import pytest
import torch

from warpSPH.runner import RunConfiguration
from warpSPH.cases.compressible.drivenTurbulence import drivenTurbulenceCase as case
from warpSPH.enumTypes import AdaptiveSupportScheme
from warpSPHCore import setNeighborLanes, SupportScheme
from warpSPHCore.radiusSearch import buildChunkedVerletList, ChunkedAdjacency, ChunkedPairField, RowChunkPolicy


def _run(monkeypatch, scheme, dim, tableau, support, chunked, change=False, lanes=1):
    workModule = importlib.import_module('warpSPH.modules.' + ('crk' if scheme == 'CRKSPH' else 'compSPH') + '.work')
    fn = 'accumulateCRKWork' if scheme == 'CRKSPH' else 'accumulateCompWork'
    original = getattr(workModule, fn)
    records, launches = [], []
    cursor = 0
    def capture(system, state, update, *a, **kw):
        nonlocal cursor
        original(system, state, update, *a, **kw)
        pw = system.pairWork
        stageLaunches = launches[cursor:]
        cursor = len(launches)
        seenWork = False
        for tag, *_ in stageLaunches:
            if tag == 'W':
                seenWork = True
            else:
                assert not seenWork, 'force kernels must all finish before the stage work sweep'
        expectedForces = len(a[2].chunks) if chunked else 1
        assert sum(tag == 'F' for tag, *_ in stageLaunches) == expectedForces
        if isinstance(pw.work, ChunkedPairField):
            assert state.ap_ij is state.av_ij is state.f_ij is None
            buffers = tuple(t[:c.metadata.liveEdges].clone() for c, t in zip(pw.adjacency, pw.work.buffers))
            assert all(t.shape[0] <= 1024 for t in pw.work.buffers)
        elif pw.work is not None:
            buffers = pw.work[:pw.adjacency.numEdges].clone()
        else:
            buffers = None
        records.append(dict(adjacency=pw.adjacency, work=buffers, reducing=pw.halfVelocity is not None,
                            delta=None if pw.deltaU is None else pw.deltaU.clone(),
                            dvdt=update.dvdt.clone(), dudt=update.dudt.clone()))
    with monkeypatch.context() as patch:
        launcher = importlib.import_module('warpSPHCore.autograd.launcher')
        forceModule = importlib.import_module('warpSPH.modules.' + ('crk' if scheme == 'CRKSPH' else 'compSPH') + '.accel')
        forceSpec = forceModule._CRK_ACCEL if scheme == 'CRKSPH' else forceModule._COMPSPH_ACCEL
        launch = launcher._launchMaybeTiled
        def allocationAudit(kernel, n, inputs, device, lanes, **kwargs):
            if kernel in (forceSpec.kernel, forceSpec.tiledKernel):
                outputs = inputs[-4:]
                assert outputs[1].shape == outputs[2].shape == (1,)
                if chunked:
                    assert n <= 5
                sizes = [a._tensor.untyped_storage().nbytes() for a in outputs]
                launches.append(('F', n, sum(sizes)))
            elif kernel is workModule._SPEC.kernel:
                launches.append(('W', n, inputs[-1]._tensor.untyped_storage().nbytes()))
            return launch(kernel, n, inputs, device, lanes, **kwargs)
        patch.setattr(launcher, '_launchMaybeTiled', allocationAudit)
        patch.setattr(workModule, fn, capture)
        schemeModule = importlib.import_module('warpSPH.modules.compressibleStage')
        pairList = schemeModule.stagePairList
        visits = {}
        rebuilt = []
        def forceChange(system, state, config, sc, **kw):
            adj = pairList(system, state, config, sc, **kw)
            pw = system.pairWork
            # Keep the attempt objects alive to avoid Python id reuse across steps.
            key = id(pw)
            if key not in visits:
                visits[key] = [pw, 0]
            visits[key][1] += 1
            if change and not pw.conservative and visits[key][1] == 2:
                if isinstance(adj, ChunkedAdjacency):
                    adj = buildChunkedVerletList(state, config.domain,
                        policy=replace(adj.plan.policy, maxRows=2), priorNeighborhood=adj,
                        supportMode=SupportScheme.SuperSymmetric, verletScale=config.verletScale, forceRebuild=True)
                else:
                    from warpSPHCore import buildVerletList
                    adj = buildVerletList(state, config.domain, priorNeighborhood=adj,
                        supportMode=SupportScheme.SuperSymmetric, verletScale=config.verletScale,
                        forceRebuild=True, storeI=False, queryOrder='input')
                system.adjacency = adj
                rebuilt.append(adj)
            return adj
        patch.setattr(schemeModule, 'stagePairList', forceChange)
        def build(ctx):
            system = case.buildSystem(ctx)
            sc = ctx.schemeConfig
            sc.adaptiveSupportScheme = getattr(AdaptiveSupportScheme, support)
            sc.adaptiveSupportIterations = 4
            sc.adaptiveSupportCorrections = scheme == 'CompSPH'
            s = system.state
            x = s.positions
            s.velocities[:, 0] = .13 * torch.sin(6.283 * x[:, 1])
            s.velocities[:, 1] = .08 * torch.cos(6.283 * x[:, 0])
            if dim == 3:
                s.velocities[:, 2] = .06 * torch.sin(6.283 * x[:, 2])
            s.internalEnergies *= 1 + .08 * torch.cos(6.283 * x[:, 0])
            from warpSPH.modules.eos import refreshEOS
            refreshEOS(s, sc)
            ctx.scratch['initialEnergy'] = (s.masses.double() *
                (s.internalEnergies.double() + .5 * s.velocities.double().square().sum(-1))).sum()
            if chunked:
                def forbidden(*a, **kw):
                    raise AssertionError('chunked compatible step requested a monolithic neighbor list')
                module = importlib.import_module('warpSPHCore.radiusSearch.verlet.build')
                patch.setattr(module, 'radiusSearchCompactHashMap_', forbidden)
                policy = RowChunkPolicy(edgeBudget=1024, maxRows=5,
                    bytesPerEdge=4 + dim*s.positions.element_size(), maxElementsPerEdge=dim)
                system.adjacency = buildChunkedVerletList(s, ctx.config.domain, policy=policy,
                    verletScale=ctx.config.verletScale, supportMode=SupportScheme.SuperSymmetric)
            return system
        cfg = RunConfiguration('drivenTurbulence', simulation=dict(
            device=os.environ.get('WARPSPH_TEST_DEVICE', 'cpu'), dim=dim, nx=8 if dim == 2 else 6,
            nSteps=2, maxDt=.0002, plot=False, video=False, store=False, storeMode='trajectory',
            quiet=True, show=False, progress=True, stallProgress=1e-3, velocityAlarmPlotInterval=1,
            integrationScheme=tableau), params=dict(accelerationRMS=.1),
            solver=dict(scheme=scheme, **({'crkPairStorage':'recompute'} if scheme == 'CRKSPH' else {'compPairStorage':'recompute'})))
        setNeighborLanes(lanes)
        try:
            result = cfg.run(replace(case, buildSystem=build))
        finally:
            setNeighborLanes(None)
    retained = result.state.pairWork.work
    workBytes = (retained.memoryUsage()['capacityBytes'] if isinstance(retained, ChunkedPairField)
                 else retained.untyped_storage().nbytes() if retained is not None else 0)
    print('CHUNK_WORK_AUDIT ' + json.dumps(dict(scheme=scheme, dim=dim, tableau=tableau,
        support=support, chunked=chunked, change=change, lanes=lanes, device=str(result.state.state.positions.device),
        rhs=len(records), wallSeconds=result.wallTime, msPerStep=1000*result.wallTime/result.nSteps,
        retainedWorkStorageBytes=workBytes, largestForceKernelOutputBytes=max(b for tag,n,b in launches if tag=='F'),
        note='kernel output storage excludes assembled particle outputs, other scratch, JIT and allocator reserves')),
        flush=True)
    assert not result.diverged and result.nSteps == 2
    assert min(result.trajectory[-1][k] for k in ('minDensity', 'minPressure', 'minEnergy')) > 0
    assert abs(result.trajectory[-1]['relativeEnergyBudgetResidual']) < 2e-6
    if change:
        assert rebuilt and any(r['reducing'] for r in records)
    return result, records


def _compare(flat, chunks, history, otherHistory):
    assert len(history) == len(otherHistory) > 0
    assert flat.state.t == chunks.state.t
    for recordIndex, (a, b) in enumerate(zip(history, otherHistory)):
        assert a['reducing'] == b['reducing']
        for name in ('delta', 'dvdt', 'dudt'):
            if a[name] is None:
                assert b[name] is None
            else:
                torch.testing.assert_close(a[name], b[name], rtol=0, atol=0, msg=lambda m: f'stage {recordIndex} {name}: ' + m)
        if a['work'] is None:
            assert b['work'] is None
        else:
            adj = b['adjacency']
            for chunk, w in zip(adj, b['work']):
                m = chunk.metadata
                torch.testing.assert_close(w, a['work'][m.edgeStart:m.edgeStart+m.liveEdges], rtol=0, atol=0)
                torch.testing.assert_close(chunk.adjacency.paddedJ[:m.liveEdges],
                    a['adjacency'].paddedJ[m.edgeStart:m.edgeStart+m.liveEdges], rtol=0, atol=0)

    for name, value in vars(flat.state.state).items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, getattr(chunks.state.state, name), rtol=0, atol=0, msg=lambda m: name + ': ' + m)


@pytest.mark.parametrize('scheme', ['CRKSPH', 'CompSPH'])
@pytest.mark.parametrize('dim,tableau,support', [(2, 'rungeKutta2', 'NoScheme'),
    (2, 'rungeKutta4', 'NoScheme'), (3, 'rungeKutta2', 'Owen')])
def test_native_work_and_full_fields_are_bitwise(monkeypatch, scheme, dim, tableau, support):
    flat, history = _run(monkeypatch, scheme, dim, tableau, support, False)
    chunks, other = _run(monkeypatch, scheme, dim, tableau, support, True)
    _compare(flat, chunks, history, other)
    if support == "NoScheme":
        assert len(history) == (8 if tableau == "rungeKutta4" else 4)
    assert isinstance(chunks.state.pairWork.work, ChunkedPairField)


@pytest.mark.parametrize('scheme', ['CRKSPH', 'CompSPH'])
def test_midstage_generation_and_chunk_layout_changes_replay(monkeypatch, scheme):
    flat, history = _run(monkeypatch, scheme, 2, 'rungeKutta4', 'NoScheme', False, change=True)
    chunks, other = _run(monkeypatch, scheme, 2, 'rungeKutta4', 'NoScheme', True, change=True)
    _compare(flat, chunks, history, other)
    assert len(history) == 16 and sum(r["reducing"] for r in other) == 8
    assert chunks.state.pairWork.halfVelocity is not None and chunks.state.pairWork.work is None


@pytest.mark.parametrize('scheme', ['CRKSPH', 'CompSPH'])
def test_chunk_work_on_nondefault_stream_with_32_neighbor_lanes(monkeypatch, scheme):
    if not torch.cuda.is_available() or os.environ.get('WARPSPH_TEST_DEVICE', 'cpu') != 'cuda':
        pytest.skip('CUDA required')
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        flat, history = _run(monkeypatch, scheme, 2, 'rungeKutta4', 'NoScheme', False, lanes=32)
        chunks, other = _run(monkeypatch, scheme, 2, 'rungeKutta4', 'NoScheme', True, lanes=32)
        _compare(flat, chunks, history, other)
    stream.synchronize()
