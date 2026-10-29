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
