"""Real-job restart across policies, world sizes and configured offset widths."""
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import h5py
import pytest
import torch.distributed as dist


def launch(tmp_path,action,scheme,ranks,rows,offset,*,steps=2,checkpoint=None,extra=()):
    path=checkpoint or tmp_path/'checkpoint.h5'
    worker=Path(__file__).with_name('distributed')/'_checkpointWorker.py'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',
             warpSPHCore_OFFSET_DTYPE=offset)
    label=f'{action}-{ranks}-{rows}-{offset}-{path.stem}-'+sha256(repr(extra).encode()).hexdigest()[:8]
    output=tmp_path/label
    command=[sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc-per-node={ranks}',
        str(worker),'--action',action,'--scheme',scheme,'--rows',str(rows),'--steps',str(steps),
        '--checkpoint',str(path),'--oracle',str(tmp_path/'oracle.pt'),'--output',str(output),*extra]
    result=subprocess.run(command,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=240)
    (tmp_path/f'{label}.log').write_text(result.stdout)
    assert result.returncode==0,result.stdout
    assert result.stdout.count(f'{action} {scheme}: passed')==ranks,result.stdout
    reports=[json.loads(p.read_text()) for p in output.glob('*.json')]
    assert len(reports)==ranks and all(d['status']=='passed' for d in reports)


@pytest.mark.skipif(not dist.is_gloo_available(),reason='Gloo required')
@pytest.mark.parametrize('scheme,sourceRanks,targetRanks', [('CRKSPH',2,3),('CompSPH',1,2),('Monaghan',3,1),('GSPH',2,3),('InutsukaGSPH',3,2)])
def test_accepted_restart_changes_budget_rank_count_and_offset(tmp_path,scheme,sourceRanks,targetRanks):
    launch(tmp_path,'reference',scheme,1,4,'int64',steps=4)
    launch(tmp_path,'save',scheme,sourceRanks,3,'int32',extra=('--contracts',) if scheme=='CRKSPH' else ())
    with h5py.File(tmp_path/'checkpoint.h5','r') as file:
        assert file.attrs['complete']==1 and file.attrs['completedSteps']==2
        names=[]
        file.visit(names.append)
        assert not any(name.split('/')[-1] in ('paddedJ','paddedI','edgeOffsets','W','work') for name in names)
    launch(tmp_path,'resume',scheme,targetRanks,1,'int64',extra=('--replica-checks','contract'))
    if scheme=='CRKSPH':
        # One large chunk -> two empty compute owners, including empty W buffers.
        launch(tmp_path,'resume',scheme,3,80,'int64',extra=('--edge-budget','8192'))
        corrupt=tmp_path/'corrupt.h5'
        shutil.copy2(tmp_path/'checkpoint.h5',corrupt)
        with h5py.File(corrupt,'r+') as file:
            file['state/velocities'][0,0]+=.01
        launch(tmp_path,'reject',scheme,2,2,'int64',checkpoint=corrupt,extra=('--expect-error','checksum mismatch'))
        launch(tmp_path,'reject',scheme,2,2,'int64',extra=('--amplitude','.2','--expect-error','configuration is incompatible'))


@pytest.mark.skipif(not dist.is_gloo_available(),reason='Gloo required')
@pytest.mark.parametrize('scheme', ['CRKSPH','CompSPH','Monaghan'])
@pytest.mark.parametrize('layout,sourceOffset', [('csr','int32'),('coo','int64')])
def test_import_legacy_csr_coo_without_global_pair_read(tmp_path,scheme,layout,sourceOffset):
    extra=('--legacy',layout)
    launch(tmp_path,'reference',scheme,1,4,sourceOffset,steps=4,extra=extra)
    launch(tmp_path,'save',scheme,1,3,sourceOffset,extra=extra)
    launch(tmp_path,'resume',scheme,3,1,'int64')
    if scheme=='CRKSPH' and layout=='coo':
        for name,field,text in [('offsets','edgeOffsets','Invalid legacy CSR offsets'),
                                ('pairs','paddedJ','pair order/content'),
                                ('rows','paddedI','COO row IDs')]:
            corrupt=tmp_path/(name+'.h5')
            shutil.copy2(tmp_path/'checkpoint.h5',corrupt)
            with h5py.File(corrupt,'r+') as file:
                dataset=file['dict_sourceAdjacency/'+field]
                dataset[0]=-1
            launch(tmp_path,'reject',scheme,2,2,'int64',checkpoint=corrupt,extra=('--expect-error',text))


@pytest.mark.parametrize('scheme', ['CRKSPH', 'CompSPH'])
@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('site', ['work', 'neighbors'])
def test_allocation_failure_preserves_accepted_checkpoint_and_restart(tmp_path, monkeypatch, scheme, dim, site):
    """A failed trial may advance forcing caches, never the accepted solution.

    Tiny repeated CPU/CUDA controls intentionally omit video. Both native
    allocation exception types and partially allocated work are exercised.
    """
    from dataclasses import replace
    import importlib
    import weakref
    import torch
    from warpSPH.runner import RunConfiguration, RunAllocationError
    from warpSPH.cases.compressible.drivenTurbulence import drivenTurbulenceCase as base, buildSPHSystem
    from warpSPHCore.radiusSearch import RowChunkPolicy, buildChunkedVerletList, ChunkedPairField
    chunk = importlib.import_module('warpSPHCore.radiusSearch.chunked')
    device = os.environ.get('WARPSPH_TEST_DEVICE', 'cpu')
    policy = RowChunkPolicy(edgeBudget=4096, maxRows=4, bytesPerEdge=4+dim*4, maxElementsPerEdge=dim)
    held = {}; partial = []
    def build(ctx):
        held['ctx'] = ctx
        return buildSPHSystem(ctx, neighborhoodFactory=lambda state, cfg: buildChunkedVerletList(
            state, cfg.domain, policy=policy, verletScale=cfg.verletScale, supportMode=cfg.supportMode))
    def snapshot(ctx, system, completed):
        if completed == 2:
            held['accepted'] = {k:v.detach().cpu().clone() for k,v in vars(system.state).items() if isinstance(v, torch.Tensor)}
            held['time'] = float(system.t)
            held['forcing'] = system.sourceTerms.acceleration.state_dict()
    def configuration(steps, resume=None):
        return RunConfiguration('drivenTurbulence', simulation=dict(device=device, dim=dim,
            nx=6, nSteps=steps, maxDt=.0002, integrationScheme='rungeKutta2',
            plot=False, video=False, store=False, storeMode='trajectory', progress=True, quiet=True, show=False,
            exportRoot=str(tmp_path), **(dict(resumeFrom=str(resume)) if resume else {})),
            solver=dict(scheme=scheme))
    case = replace(base, buildSystem=build, acceptedStep=snapshot)
    baseline = configuration(4).run(case)
    with monkeypatch.context() as patch:
        empty = ChunkedPairField.empty
        allocate = chunk._allocateChunks
        def fail():
            # A rejected attempt has requested future OU coefficients. The
            # checkpoint must restore the accepted generator and knot history.
            held['ctx'].scratch['driving'].accelerationBound(.2)
            if site == 'work':
                raise torch.OutOfMemoryError('injected owned work allocation')
            raise MemoryError('injected neighbor allocation')
        @classmethod
        def work(cls, adjacency, **kwargs):
            if site == 'work' and held['ctx'].scratch.get('completedSteps') == 2:
                # Real partial pair storage must be released on failure.
                value = torch.empty((adjacency.chunks[0].metadata.capacity, dim), device=device)
                partial.append(weakref.ref(value))
                fail()
            return empty(adjacency, **kwargs)
        def neighbors(*args, **kwargs):
            if site == 'neighbors' and held['ctx'].scratch.get('completedSteps') == 2:
                fail()
            return allocate(*args, **kwargs)
        patch.setattr(ChunkedPairField, 'empty', work)
        patch.setattr(chunk, '_allocateChunks', neighbors)
        if site == 'neighbors':
            module = importlib.import_module('warpSPH.schemes.compressible.' + ('crkSPH' if scheme=='CRKSPH' else 'compSPH'))
            native = module.stagePairList
            def rebuild(system, state, cfg, sc, **kwargs):
                adjacency = native(system, state, cfg, sc, **kwargs)
                if held['ctx'].scratch.get('completedSteps') == 2:
                    return buildChunkedVerletList(state, cfg.domain, policy=policy,
                        priorNeighborhood=adjacency, forceRebuild=True,
                        verletScale=cfg.verletScale, supportMode=cfg.supportMode)
                return adjacency
            patch.setattr(module, 'stagePairList', rebuild)
        with pytest.raises(RunAllocationError) as caught:
            configuration(4).run(case)
    error = caught.value
    assert error.recoveryError is None, error.recoveryError
    assert error.checkpointPath and Path(error.checkpointPath).is_file()
    assert error.result.nSteps == 2 and float(error.result.state.t) == held['time']
    assert all(ref() is None for ref in partial)
    for name, value in held['accepted'].items():
        torch.testing.assert_close(getattr(error.result.state.state, name).cpu(), value, rtol=0, atol=0)
    for name in ('knots', 'generator'):
        torch.testing.assert_close(error.result.state.sourceTerms.acceleration.state_dict()[name], held['forcing'][name], rtol=0, atol=0)
    resumed = configuration(2, error.checkpointPath).run(case)
    assert resumed.ctx.scratch['completedSteps'] == 4
    for name, value in vars(baseline.state.state).items():
        if isinstance(value, torch.Tensor):
            tol = 2e-12 if name == 'sourceEnergyBudget' else 0
            torch.testing.assert_close(getattr(resumed.state.state, name), value, rtol=tol, atol=tol, msg=name)
    assert float(resumed.state.t) == float(baseline.state.t)


@pytest.mark.skipif(not dist.is_gloo_available(), reason='Gloo required')
@pytest.mark.parametrize('site', ['work', 'neighbors'])
@pytest.mark.parametrize('rank', [0, 1])
def test_coordinated_allocation_failure_checkpoint_resumes_across_world_sizes(tmp_path, site, rank):
    launch(tmp_path, 'reference', 'CRKSPH', 1, 4, 'int64', steps=4)
    launch(tmp_path, 'fail', 'CRKSPH', 2, 3, 'int32', steps=4,
           extra=('--fail-site', site, '--fail-rank', str(rank)))
    launch(tmp_path, 'resume', 'CRKSPH', 3, 1, 'int64', steps=2)


def test_failed_recovery_publication_keeps_original_error_and_accepted_particles(tmp_path, monkeypatch):
    from dataclasses import replace
    import importlib
    import torch
    from warpSPH.runner import RunConfiguration
    from warpSPH.runner.recovery import captureBoundary, recoverAllocation
    from warpSPH.cases.compressible.drivenTurbulence import drivenTurbulenceCase as base, buildSPHSystem
    from warpSPHCore.radiusSearch import RowChunkPolicy, buildChunkedVerletList
    def build(ctx):
        return buildSPHSystem(ctx, neighborhoodFactory=lambda state,cfg:buildChunkedVerletList(
            state,cfg.domain,policy=RowChunkPolicy(edgeBudget=4096,maxRows=4,bytesPerEdge=12,maxElementsPerEdge=2),
            verletScale=cfg.verletScale,supportMode=cfg.supportMode))
    configuration=RunConfiguration('drivenTurbulence',simulation=dict(device='cpu',nx=6,dim=2,nSteps=2,
        maxDt=.0002,plot=False,video=False,store=False,storeMode='trajectory',progress=False,show=False,quiet=True,
        exportRoot=str(tmp_path)),solver=dict(scheme='CRKSPH'))
    result=configuration.run(replace(base,buildSystem=build))
    accepted=result.state.state.positions.clone()
    boundary=captureBoundary(result.ctx,result.state)
    def reject(*a,**kw):raise OSError('disk publication failed')
    monkeypatch.setattr(importlib.import_module('warpSPH.io.chunkCheckpoint'),'writeChunkedCheckpoint',reject)
    original=MemoryError('primary bulk rejection')
    error=recoverAllocation(result.ctx,result,boundary,original)
    assert str(error)=='primary bulk rejection' and error.checkpointPath is None
    assert error.recoveryError=='OSError: disk publication failed'
    assert result.nSteps==2
    torch.testing.assert_close(result.state.state.positions,accepted,rtol=0,atol=0)
    assert any('primary bulk rejection' in note for note in error.__notes__)
    assert not list(tmp_path.rglob('*.h5'))
