"""Small restart fixtures: numerical memory is portable, spatial caches are not."""
from dataclasses import replace
from pathlib import Path

import h5py
import pytest
import torch

from warpSPH.evaluation import MortonStepper,adopt_spatial_continuation
from warpSPH.io.export import exportSimulationSystem
from warpSPH.io.continuation import readContinuation
from warpSPH.io.particleLayout import restoreCheckpoint
from warpSPHIntegrators import get_scheme
from warpSPHIntegrators.runtime import Continuation
from warpSPHIntegrators.state.history import StepHistory
from warpSPHIntegrators.solvers import JFNKSolver
from test_spatialContinuation import fixture,assert_physics


def saved(tmp_path,*,name='ab3',fluid='CompSPH'):
    ctx,state=fixture(fluid,fsal=True)
    state.state.UIDcounter=int(state.state.UIDs.max())+1
    initial=state.initializeNewState()
    integrator=get_scheme(name)
    stepper=MortonStepper(integrator)
    memory=Continuation(state,StepHistory(max(1,integrator.metadata.history_length)))
    kwargs=dict(config=ctx.config,schemeConfig=ctx.schemeConfig)
    if integrator.implicit:kwargs['solver']=JFNKSolver(tol=1e-6)
    for _ in range(3):
        result=stepper(memory.state,1e-4,history=memory.history,**kwargs)
        memory=adopt_spatial_continuation(Continuation(result.state,result.history or memory.history,result.stages[-1]))
    exportSimulationSystem(str(tmp_path),'checkpoint',fluid,memory.state,
        continuation=memory,integrator=integrator,config=ctx.config,schemeConfig=ctx.schemeConfig,
        last_dt=1e-4,next_dt=1e-4)
    return ctx,initial,stepper,memory,kwargs,tmp_path/'trajectory/checkpoint.h5'


@pytest.mark.parametrize('fluid',['CompSPH','CRKSPH'])
@pytest.mark.parametrize('name',['ab3','bdf2','dormandPrince'])
def test_direct_checkpoint_continues_history_states_and_raw_fsal_without_search_caches(tmp_path,fluid,name):
    ctx,initial,stepper,memory,kwargs,path=saved(tmp_path,name=name,fluid=fluid)
    with h5py.File(path) as f:
        assert 'adjacency' not in f
        restoreCheckpoint(f,initial,ctx.config,ctx.scheme)
        restored,last_dt,next_dt=readContinuation(f['integration'],initial,
            integrator=stepper.integrator,config=ctx.config,schemeConfig=ctx.schemeConfig)
        assert last_dt==next_dt==1e-4
        assert len(restored.history)==len(memory.history)
        assert not any(key.endswith('adjacency') for key in f['integration'])
    assert restored.state.adjacency is None and restored.state.spatialContext is None
    assert all(e.state is None or e.state.adjacency is None for e in restored.history)
    for _ in range(2):
        a=stepper(memory.state,1e-4,history=memory.history,priorStep=memory.prior_step if name=='dormandPrince' else None,**kwargs)
        b=stepper(restored.state,1e-4,history=restored.history,priorStep=restored.prior_step if name=='dormandPrince' else None,**kwargs)
        assert_physics(a.state,b.state)
        memory=adopt_spatial_continuation(Continuation(a.state,a.history or memory.history,a.stages[-1]))
        restored=adopt_spatial_continuation(Continuation(b.state,b.history or restored.history,b.stages[-1]))


@pytest.mark.parametrize('bad',['version','method','configuration','time','uid','update','snapshot','edge'])
def test_corrupt_numerical_memory_is_rejected_before_it_can_be_consumed(tmp_path,bad):
    ctx,initial,stepper,memory,kwargs,path=saved(tmp_path,name='bdf2')
    with h5py.File(path,'r+') as f:
        group=f['integration']
        if bad=='version':group.attrs['version']=99
        elif bad=='method':group.attrs['method']='ab3'
        elif bad=='configuration':ctx.schemeConfig.gamma+=.01
        elif bad=='time':group.attrs['time']=100.
        elif bad=='uid':group['history/0/state/UIDs'][0]=99
        elif bad=='update':del group['history/0/update/dudt']
        elif bad=='snapshot':group['history/0/snapshot'].attrs['time']=100.
        else:group['history/0/state'].attrs['ap_ij']='bad'
        restoreCheckpoint(f,initial,ctx.config,ctx.scheme)
        with pytest.raises((ValueError,TypeError)):
            readContinuation(group,initial,integrator=stepper.integrator,config=ctx.config,schemeConfig=ctx.schemeConfig)


def test_global_derivative_budget_preserves_double_precision_in_float_particle_checkpoint(tmp_path):
    ctx,initial,stepper,memory,kwargs,path=saved(tmp_path)
    rates=torch.tensor([1.00000000003,-.00000000007],dtype=torch.float64)
    memory.history.latest.update.sourceEnergyRates=rates
    exportSimulationSystem(str(tmp_path),'global-budget','CompSPH',memory.state,
        continuation=memory,integrator=stepper.integrator,config=ctx.config,schemeConfig=ctx.schemeConfig,
        last_dt=1e-4,next_dt=1e-4)
    with h5py.File(tmp_path/'trajectory/global-budget.h5') as f:
        restoreCheckpoint(f,initial,ctx.config,ctx.scheme)
        loaded,_,_=readContinuation(f['integration'],initial,integrator=stepper.integrator,config=ctx.config,schemeConfig=ctx.schemeConfig)
    assert loaded.history.latest.update.sourceEnergyRates.dtype==torch.float64
    torch.testing.assert_close(loaded.history.latest.update.sourceEnergyRates,rates,rtol=0,atol=0)


@pytest.mark.parametrize('field',['positions','update/dudt','update/passive'])
def test_integer_corruption_cannot_bypass_particle_or_derivative_precision_checks(tmp_path,field):
    import numpy as np
    ctx,initial,stepper,memory,kwargs,path=saved(tmp_path)
    with h5py.File(path,'r+') as f:
        root=f['integration/history/0']
        key='state/positions' if field=='positions' else field
        values=np.array(root[key],dtype=np.int64);del root[key];root.create_dataset(key,data=values)
        restoreCheckpoint(f,initial,ctx.config,ctx.scheme)
        with pytest.raises(ValueError,match='precision'):
            readContinuation(f['integration'],initial,integrator=stepper.integrator,config=ctx.config,schemeConfig=ctx.schemeConfig)


@pytest.mark.parametrize('name',['ab3','bdf2'])
def test_runner_resumes_completed_history_without_repeating_the_starter(tmp_path,name,monkeypatch):
    from warpSPH.runner import run
    from warpSPH.cases.compressible.linearWave import linearWaveCase
    from test_particleLayoutIO import uid_case
    from test_runnerParticleOrder import options
    def configure(ctx):
        linearWaveCase.configureScheme(ctx)
        ctx.schemeConfig.compatibleEnergy=False
    case=replace(uid_case(),configureScheme=configure)
    args=options();args.update(integrationScheme=name,particleOrder='morton',nSteps=5)
    complete=run(case,**args)
    args.update(nSteps=3,store=True,exportRoot=str(tmp_path))
    prefix=run(case,**args)
    checkpoint=Path(prefix.exportPath)/'trajectory/finalState.h5'
    counts=[]
    native=MortonStepper.__call__
    def track(self,*a,**kw):
        result=native(self,*a,**kw);counts.append(len(result.stages));return result
    monkeypatch.setattr(MortonStepper,'__call__',track)
    args.update(nSteps=2,resumeFrom=str(checkpoint),resumeStepOffset=3,store=False)
    suffix=run(case,**args)
    assert not suffix.diverged and counts[0]==1
    assert_physics(suffix.state,complete.state)
    assert len(suffix.ctx.scratch['stepHistory'])==len(complete.ctx.scratch['stepHistory'])


def test_failed_memory_write_preserves_the_previous_complete_checkpoint(tmp_path,monkeypatch):
    import warpSPH.io.continuation as persistence
    ctx,initial,stepper,memory,kwargs,path=saved(tmp_path)
    old=path.read_bytes()
    def fail(*a,**kw):raise RuntimeError('injected write failure')
    monkeypatch.setattr(persistence,'writeContinuation',fail)
    with pytest.raises(RuntimeError,match='injected'):
        exportSimulationSystem(str(tmp_path),'checkpoint','CompSPH',memory.state,
            continuation=memory,integrator=stepper.integrator,config=ctx.config,
            schemeConfig=ctx.schemeConfig,last_dt=1e-4,next_dt=1e-4)
    assert path.read_bytes()==old
    assert not list(path.parent.glob('*.tmp-*'))
