"""The run result must not pin an obsolete system's accepted-step scratch."""
from dataclasses import replace
import importlib,weakref
import pytest,torch
from warpSPH.runner import RunConfiguration
from warpSPH.cases.compressible.drivenTurbulence import drivenTurbulenceCase as base


@pytest.mark.parametrize('scheme',['CRKSPH','CompSPH'])
@pytest.mark.parametrize('steps',[0,1,4])
def test_result_follows_accepted_state_and_releases_first_work(monkeypatch,scheme,steps):
    module=importlib.import_module('warpSPH.runner.runner');original=module.RunResult;held={};seen=[]
    def result(*args,**kwargs):
        value=original(*args,**kwargs);held['result']=weakref.ref(value);return value
    monkeypatch.setattr(module,'RunResult',result)
    def diagnostics(ctx,system):
        assert held['result']().state is system
        step=ctx.scratch['completedSteps'];seen.append(step)
        if step==1:held['firstWork']=weakref.ref(system.pairWork.work)
        if step>=2:assert held['firstWork']() is None
        return base.diagnostics(ctx,system)
    c=RunConfiguration('drivenTurbulence',simulation=dict(device='cpu',dim=2,nx=12,nSteps=steps,maxDt=.001,
        plot=False,video=False,store=False,progress=False,show=False,quiet=True),solver=dict(scheme=scheme))
    value=c.run(replace(base,diagnostics=diagnostics))
    assert value.nSteps==steps and len(seen)==steps+1
    assert not value.diverged
    assert torch.isfinite(value.state.state.internalEnergies).all()


def test_early_stop_result_is_latest_accepted_state(monkeypatch):
    module=importlib.import_module('warpSPH.runner.runner');original=module.RunResult;held={}
    def result(*args,**kwargs):
        value=original(*args,**kwargs);held['result']=weakref.ref(value);return value
    monkeypatch.setattr(module,'RunResult',result)
    def stop(ctx,system,step):
        assert held['result']().state is system
        system.state.velocities.fill_(float('nan'))
    c=RunConfiguration('drivenTurbulence',simulation=dict(device='cpu',nx=12,nSteps=4,maxDt=.001,
        plot=False,video=False,store=False,progress=False,show=False,quiet=True),solver=dict(scheme='CRKSPH'))
    r=c.run(replace(base,postStep=stop))
    assert r.diverged and r.stopReason=='non-finite velocities' and r.nSteps==1


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA pipelined graph')
def test_pipelined_result_follows_accepted_state(monkeypatch):
    from test_cudaGraph import _marrone
    module=importlib.import_module('warpSPH.runner.runner');original=module.RunResult;held={};calls=[]
    def result(*args,**kwargs):
        value=original(*args,**kwargs);held['result']=weakref.ref(value);return value
    monkeypatch.setattr(module,'RunResult',result)
    case_module=importlib.import_module('warpSPH.cases.freeSurface.dambreak');case=case_module.dambreakCase
    def diagnostics(ctx,system):
        assert held['result']().state is system
        calls.append(float(system.t));return case.diagnostics(ctx,system)
    monkeypatch.setattr(case_module,'dambreakCase',replace(case,diagnostics=diagnostics))
    value=_marrone(24,None,cudaGraph=True,pipelineOutputs=True,tLimit=.02)
    # This tests the pipelined runner loop even when graph capture falls back
    # to eager; graph replay itself has an independent existing regression.
    assert not value.diverged and len(calls)>2 and value.ctx.config.compute.pipelineOutputs


@pytest.mark.parametrize('scheme',['CompSPH','CRKSPH','Monaghan','GSPH','InutsukaGSPH','MFM','MFV'])
def test_previous_stage_result_is_released_before_next_rhs(scheme):
    import sys
    held={};checks=[]
    def build(ctx):
        system=base.buildSystem(ctx);native=ctx.stepFunction
        def observed(s,dt,stageIndex=None,**kwargs):
            if stageIndex==0 and 'previous' in held:
                assert held['previous']() is None
                checks.append(1)
            return native(s,dt,stageIndex=stageIndex,**kwargs)
        ctx.bundle=replace(ctx.bundle,stepFunction=observed);return system
    def diagnostics(ctx,system):
        if ctx.scratch['completedSteps']:
            held['previous']=weakref.ref(sys._getframe(1).f_locals['stepResult'])
        return base.diagnostics(ctx,system)
    c=RunConfiguration('drivenTurbulence',simulation=dict(device='cpu',nx=12,nSteps=4,maxDt=.001,plot=False,video=False,store=False,progress=False,show=False,quiet=True),solver=dict(scheme=scheme,**(dict(temporalScheme='ssprk') if scheme in ['MFM','MFV'] else {})))
    value=c.run(replace(base,buildSystem=build,diagnostics=diagnostics))
    assert not value.diverged and len(checks)==3


def test_failure_cleanup_preserves_primary_error_and_attempts_every_resource(monkeypatch):
    """Even a broken renderer/trajectory close cannot hide allocation evidence."""
    from types import SimpleNamespace
    module=importlib.import_module('warpSPH.runner.runner')
    events=[]; primary=MemoryError('bulk allocation rejected')
    class File:
        def close(self):
            events.append('file');raise OSError('file close failed')
    ctx=SimpleNamespace(device=torch.device('cpu'),scratch={'trajectoryFile':File()})
    monkeypatch.setattr(module,'prepareContext',lambda *a:ctx)
    def fail(*a):raise primary
    monkeypatch.setattr(module,'_runContext',fail)
    monkeypatch.setattr(module,'_closeProgress',lambda ctx:events.append('progress'))
    def drain():
        events.append('frames');raise OSError('PNG writer failed')
    monkeypatch.setattr(module,'_drainFrameWrites',drain)
    monkeypatch.setattr(module,'_teardownPlot',lambda ctx,**kw:events.append(('plot',kw['failed'])))
    with pytest.raises(MemoryError) as caught:module._run(None,None,0)
    assert caught.value is primary
    assert events==['progress','frames',('plot',True),'file']
    assert any('PNG writer failed' in note for note in primary.__notes__)
    assert any('file close failed' in note for note in primary.__notes__)


def test_timer_cleanup_does_not_replace_an_allocation_failure(monkeypatch):
    module=importlib.import_module('warpSPH.runner.runner')
    primary=MemoryError('bulk allocation rejected')
    timer=module._Timer(torch.device('cpu'));timer.cuda=True
    class Event:
        def record(self):raise RuntimeError('device synchronization failed')
    timer._end=Event()
    assert timer.__exit__(MemoryError,primary,None) is False
    assert 'device synchronization failed' in primary.__notes__[0]
    with pytest.raises(RuntimeError,match='device synchronization failed'):
        timer.__exit__(None,None,None)
