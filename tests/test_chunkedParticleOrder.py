"""Combined fixed-count Morton/owned-row controls; tiny fixtures omit video."""
from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from warpSPHCore.radiusSearch import ChunkedAdjacency, RowChunkPolicy, buildChunkedVerletList
from warpSPHCore import SupportScheme
from warpSPH.io.chunkCheckpoint import writeChunkedCheckpoint

SCHEMES=('CompSPH','CRKSPH','deltaSPH','sun2017DeltaSPH','iisph','dfsph','dfsphPredictor','dfsphShift')


def fixture(scheme, *, communication=None, rows=3, steps=4, resume=None, accepted=None, order='morton'):
    if scheme in ('CompSPH','CRKSPH'):
        from warpSPH.runner import RunConfiguration
        from warpSPH.cases.compressible.drivenTurbulence import drivenTurbulenceCase as base
        cfg=RunConfiguration('drivenTurbulence',simulation=dict(nx=8,dim=2,maxDt=.0002,
            integrationScheme='rungeKutta4'),params=dict(accelerationRMS=.1,forcingInterval=.00025,seed=1234),
            solver=dict(scheme=scheme))
    elif scheme in ('deltaSPH','sun2017DeltaSPH'):
        from test_chunkedDelta import configuration
        from warpSPH.cases.periodic.tgvWeaklyCompressible import tgvWeaklyCompressibleCase as base
        cfg=configuration(scheme)
    else:
        from test_chunkedProjection import activeConfiguration
        from warpSPH.cases.periodic.tgv import tgvCase as base
        cfg=activeConfiguration(scheme)
    cfg.simulation.update(device=os.environ.get('WARPSPH_TEST_DEVICE','cpu'),nSteps=steps,
        plot=False,video=False,store=False,show=False,quiet=True,progress=True,
        execution='eager',cudaGraph=False,pipelineOutputs=False,particleOrder=order,
        resumeFrom=None if resume is None else str(resume))
    def build(ctx):
        def factory(state,config):
            dim=state.positions.shape[1]
            return buildChunkedVerletList(state,config.domain,
                policy=RowChunkPolicy(edgeBudget=16384,maxRows=rows,bytesPerEdge=8+dim*state.positions.element_size(),maxElementsPerEdge=dim),
                supportMode=config.supportMode if scheme in ('CompSPH','CRKSPH') else SupportScheme.SuperSymmetric,
                verletScale=config.verletScale,storeI=scheme not in ('CompSPH','CRKSPH'),communication=communication)
        return base.buildSystem(ctx,neighborhoodFactory=factory)
    def end(ctx,system,completed):
        if accepted is not None:accepted(ctx,system,completed)
        # The next attempt must sort/rebuild in a new basis, with its own W.
        if completed in (1,3):
            shift=system.state.positions.new_tensor([.37,.19])*(ctx.config.domain.max-ctx.config.domain.min)
            system.state.positions=system.state.positions+shift
    return cfg,replace(base,buildSystem=build,acceptedStep=end,extraData=None)


def compare(a,b):
    x,y=a.state.state,b.state.state
    pa,pb=torch.argsort(x.UIDs),torch.argsort(y.UIDs)
    assert float(a.state.t)==float(b.state.t)
    for name,value in vars(x).items():
        other=getattr(y,name)
        if not isinstance(value,torch.Tensor):continue
        if value.ndim and value.shape[0]==len(pa):value,other=value[pa],other[pb]
        tol=2e-12 if name=='sourceEnergyBudget' else 0
        torch.testing.assert_close(value,other,rtol=tol,atol=tol,msg=name)
    torch.testing.assert_close(torch.as_tensor(a.ctx.config.dt),torch.as_tensor(b.ctx.config.dt),rtol=0,atol=0)
    for name in ('sourceEnergyBudget',):
        value=getattr(a.state,name,None)
        if value is not None:torch.testing.assert_close(value,getattr(b.state,name),rtol=2e-12,atol=2e-12)
    source=getattr(a.state,'sourceTerms',None)
    if source is not None:
        p,q=source.acceleration.state_dict(),b.state.sourceTerms.acceleration.state_dict()
        assert p['parameters']==q['parameters']
        for field in ('knots','generator'):torch.testing.assert_close(p[field],q[field],rtol=0,atol=0)


@pytest.mark.parametrize('scheme',SCHEMES)
def test_chunked_morton_restarts_with_new_partition_and_retains_numerical_memory(tmp_path,scheme):
    seen=[]
    def record(ctx,system,completed):
        assert isinstance(system.adjacency,ChunkedAdjacency)
        seen.append((system.adjacency,system.state.UIDs.clone()))
    cfg,case=fixture(scheme,accepted=record)
    full=cfg.run(case)
    assert not full.diverged and len(seen)==4
    assert full.ctx.scratch['nativeLayout'].sorts>=2
    assert seen[0][0] is not seen[1][0]
    cfg,case=fixture(scheme,steps=2,rows=1<<20)
    first=cfg.run(case)
    checkpoint=writeChunkedCheckpoint(tmp_path/'state.h5',first.ctx,first.state)
    cfg,case=fixture(scheme,steps=2,rows=1,resume=checkpoint)
    resumed=cfg.run(case)
    assert not resumed.diverged and resumed.ctx.scratch['completedSteps']==4
    compare(full,resumed)


def test_chunked_layout_reuses_valid_list_and_rejects_implicit_before_sort():
    cfg,case=fixture('CompSPH',steps=1)
    result=cfg.run(case)
    adapter=result.ctx.scratch['nativeLayout']
    s,c,sc,_=adapter.prepare(result.state,result.ctx.config,result.ctx.schemeConfig)
    same=adapter.prepare(s,c,sc)
    assert same[0] is s and same[0].adjacency is s.adjacency and not same[3]
    from warpSPH.evaluation.chunkedLayout import ChunkedMortonLayout
    from warpSPHIntegrators import get_scheme
    with pytest.raises(ValueError,match='one-step explicit'):
        ChunkedMortonLayout(s,get_scheme('backwardEuler'),sc)


def test_distributed_morton_empty_owner_rebuild_and_restart(tmp_path):
    worker=Path(__file__).with_name('distributed')/'_particleOrderWorker.py'
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=2',str(worker),'--output',str(tmp_path)]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2')
    completed=subprocess.run(command,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=240)
    (tmp_path/'worker.log').write_text(completed.stdout)
    assert completed.returncode==0,completed.stdout
    assert completed.stdout.count('particle-order integration passed')==2


def test_driven_global_budget_is_not_a_particle_axis_when_n_equals_two():
    from warpSPH.caseUtils.drivenTurbulence import DrivenCompSPHState
    from warpSPH.evaluation.layout import PARTICLE_FIELDS,permute_state
    cfg,case=fixture('CompSPH',steps=1)
    state=cfg.run(case).state.state
    values={k:(v[:2] if k in PARTICLE_FIELDS and isinstance(v,torch.Tensor) else v)
            for k,v in vars(state).items()}
    state=DrivenCompSPHState(**values)
    state.sourceEnergyBudget=state.masses.new_tensor([7.,11.],dtype=torch.float64)
    order=torch.tensor([1,0],device=state.positions.device)
    result=permute_state(state,order)
    assert result.UIDs.tolist()==state.UIDs[order].tolist()
    assert result.sourceEnergyBudget.tolist()==[7.,11.]


@pytest.mark.parametrize('scheme',['CompSPH','deltaSPH','dfsph'])
def test_allocation_recovery_after_resort_keeps_the_accepted_basis(tmp_path,monkeypatch,scheme):
    import importlib
    import h5py
    from warpSPH.runner import RunAllocationError
    launcher=importlib.import_module('warpSPHCore.autograd.launcher')
    original=launcher._allocate_output
    armed=False
    inAttempt=False
    accepted={}
    def allocate(*args,**kwargs):
        nonlocal armed
        if armed and inAttempt:
            armed=False
            raise MemoryError('injected post-sort allocation failure')
        return original(*args,**kwargs)
    from warpSPH.evaluation.chunkedLayout import ChunkedMortonLayout
    prepare=ChunkedMortonLayout.prepare
    def prepared(self,*args,**kwargs):
        nonlocal inAttempt
        result=prepare(self,*args,**kwargs)
        inAttempt=armed
        return result
    def mark(ctx,system,completed):
        nonlocal armed
        if completed==1:
            armed=True
            accepted['system']=system
    monkeypatch.setattr(launcher,'_allocate_output',allocate)
    monkeypatch.setattr(ChunkedMortonLayout,'prepare',prepared)
    cfg,case=fixture(scheme,accepted=mark)
    cfg.simulation['exportRoot']=str(tmp_path)
    with pytest.raises(RunAllocationError) as caught:cfg.run(case)
    error=caught.value
    assert error.checkpointPath is not None and error.recoveryError is None
    before,after=accepted['system'],error.result.state
    p,q=torch.argsort(before.state.UIDs),torch.argsort(after.state.UIDs)
    for name,value in vars(before.state).items():
        if not isinstance(value,torch.Tensor):continue
        other=getattr(after.state,name)
        if value.ndim and value.shape[0]==len(p):value,other=value[p],other[q]
        torch.testing.assert_close(value,other,rtol=0,atol=0,msg=name)
    if getattr(before,'deltaStages',None) is not None:
        for name in ('surfaceStart','previousSurfaceStart'):
            value=getattr(before.deltaStages,name)
            if value is not None:torch.testing.assert_close(value[p],getattr(after.deltaStages,name)[q],rtol=0,atol=0)
    with h5py.File(error.checkpointPath) as file:
        assert file.attrs['completedSteps']==1
        assert file['state/UIDs'][:].tolist()==after.state.UIDs.tolist()
