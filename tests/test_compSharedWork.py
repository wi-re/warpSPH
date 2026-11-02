"""Native CompSPH storage paths must preserve forces, energy and list replay."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
import importlib,os
import pytest,torch
from warpSPH.runner import RunConfiguration
from warpSPH.cases.compressible.drivenTurbulence import drivenTurbulenceCase as case
from warpSPH.configurations.compSPHConfig import CompSPHConfig,compSPHConfigToDict,dictToCompSPHConfig
from test_crkSharedWork import patterned,assert_matching


def configured(mode=None,dim=2,integrator='rungeKutta2'):
    return RunConfiguration('drivenTurbulence',simulation=dict(device=os.environ.get('WARPSPH_TEST_DEVICE','cpu'),dim=dim,nx=12,nSteps=3,maxDt=.001,plot=False,video=False,store=False,progress=False,show=False,quiet=True,integrationScheme=integrator),solver=dict(scheme='CompSPH',**({} if mode is None else dict(compPairStorage=mode))))


@pytest.mark.parametrize('dim,integrator',[(2,'rungeKutta2'),(3,'rungeKutta2'),(2,'rungeKutta4')])
@pytest.mark.parametrize('gradH',[False,True])
def test_work_matches_expanded(dim,integrator,gradH):
    def build(ctx):
        system=patterned(ctx);ctx.schemeConfig.adaptiveSupportCorrections=gradH;return system
    a=configured('expanded',dim,integrator).run(replace(case,buildSystem=build))
    b=configured('recompute',dim,integrator).run(replace(case,buildSystem=build))
    assert_matching(a,b)
    assert b.state.state.ap_ij is None and b.state.state.av_ij is None and b.state.state.f_ij is None


def test_changed_list_replay_matches_expanded(monkeypatch):
    module=importlib.import_module('warpSPH.modules.compressibleStage');original=module.stagePairList
    def changed(system,*args,**kw):
        adjacency=original(system,*args,**kw)
        return adjacency if system.pairWork.conservative else replace(adjacency)
    monkeypatch.setattr(module,'stagePairList',changed)
    a=configured('expanded',2,'rungeKutta4').run(replace(case,buildSystem=patterned))
    b=configured('recompute',2,'rungeKutta4').run(replace(case,buildSystem=patterned))
    assert_matching(a,b);assert b.state.pairWork.halfVelocity is not None


def test_storage_roundtrip_and_validation():
    c=CompSPHConfig();d=compSPHConfigToDict(c)
    assert dictToCompSPHConfig(d).compPairStorage=='recompute'
    del d['compPairStorage'];assert dictToCompSPHConfig(d).compPairStorage=='recompute'
    with pytest.raises(ValueError,match='compPairStorage'):CompSPHConfig(compPairStorage='bad').validate()


def test_restart_can_change_storage_only(tmp_path):
    full=configured('expanded');full.simulation['nSteps']=6;reference=full.run()
    first=configured('expanded');first.simulation.update(store=True,storeMode='states',exportRoot=str(tmp_path/'first'))
    part=first.run();resume=configured('recompute');resume.simulation['resumeFrom']=str(Path(part.exportPath)/'trajectory/finalState.h5')
    assert_matching(reference,resume.run())


def test_default_route_and_fallbacks():
    from warpSPH.modules.compSPH.work import useRecomputedWork
    from warpSPH.enumTypes import EnergyScheme
    system=NS(pairWork=NS(weights=(0.,1.)));state=NS(positions=torch.ones(2,2));scheme=CompSPHConfig()
    assert useRecomputedWork(system,state,scheme,None,1)
    assert not useRecomputedWork(system,state,scheme,object(),1)
    for policy in [dict(compPairStorage='expanded'),dict(compatibleEnergy=False),dict(energyScheme=EnergyScheme.equalWork),dict(materials=[object()]),dict(stepMode='graph'),dict(stepMode='auto')]:
        current=CompSPHConfig()
        for k,v in policy.items():setattr(current,k,v)
        assert not useRecomputedWork(system,state,current,None,1)
    state.positions.requires_grad_();assert not useRecomputedWork(system,state,scheme,None,1)
    with torch.autograd.forward_ad.dual_level():
        state.positions=torch.autograd.forward_ad.make_dual(torch.ones(2,2),torch.ones(2,2))
        assert not useRecomputedWork(system,state,scheme,None,1)


def test_late_differentiable_force_keeps_expanded(monkeypatch):
    module=importlib.import_module('warpSPH.schemes.compressible.compSPH');work=importlib.import_module('warpSPH.modules.compSPH.work');force=torch.tensor(.1,requires_grad=True)
    def external(system,*args):return force*system.state.masses[:,None]*torch.ones_like(system.state.velocities)
    def unexpected(*args,**kw):raise AssertionError('differentiable force used in-place work')
    monkeypatch.setattr(module,'computeForcing',external);monkeypatch.setattr(work,'accumulateCompWork',unexpected)
    c=configured();c.simulation.update(device='cpu',nSteps=1);r=c.run()
    assert not r.diverged and r.state.state.velocities.requires_grad


@pytest.mark.parametrize('gradH',[False,True])
def test_heun_reference_matches_direct_work(gradH):
    from warpSPH.runner.case import getCase
    from warpSPH.cases import importAll
    importAll();base=getCase('sod2d');configure=base.configureScheme
    def scheme(ctx):
        if configure:configure(ctx)
        ctx.schemeConfig.adaptiveSupportCorrections=gradH
    results=[]
    for mode in ['expanded','recompute']:
        c=RunConfiguration('sod2d',simulation=dict(device=os.environ.get('WARPSPH_TEST_DEVICE','cpu'),nx=16,nSteps=3,maxDt=.001,plot=False,video=False,store=False,progress=False,quiet=True,integrationScheme='heunsMethod'),solver=dict(scheme='CompSPH',compPairStorage=mode))
        results.append(c.run(replace(base,configureScheme=scheme)))
    assert not any(r.diverged for r in results)
    for key in ['positions','velocities','internalEnergies','densities','supports']:
        torch.testing.assert_close(getattr(results[0].state.state,key),getattr(results[1].state.state,key),rtol=3e-6,atol=3e-7)
