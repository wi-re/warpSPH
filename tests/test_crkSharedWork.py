"""Native expanded versus shared recomputation: stages, work, and list replay."""
from dataclasses import replace
import importlib,os
import pytest,torch
from warpSPH.runner import RunConfiguration
from warpSPH.cases.compressible.drivenTurbulence import drivenTurbulenceCase as case
from warpSPH.configurations.crkSPH import CRKSPHConfig,crkSPHConfigToDict,dictToCRKSPHConfig


def configured(mode=None,dim=2,integrator='rungeKutta2'):
    return RunConfiguration('drivenTurbulence',simulation=dict(device=os.environ.get('WARPSPH_TEST_DEVICE','cpu'),dim=dim,nx=12,nSteps=2,maxDt=.001,plot=False,video=False,store=False,progress=False,show=False,quiet=True,integrationScheme=integrator),
        params=dict(accelerationRMS=.3),solver=dict(scheme='CRKSPH',**({} if mode is None else dict(crkPairStorage=mode))))


def patterned(ctx):
    s=case.buildSystem(ctx)
    x=s.state.positions
    s.state.velocities[:,0]=.15*torch.sin(6.283185307179586*x[:,1])
    s.state.velocities[:,1]=.1*torch.cos(6.283185307179586*x[:,0])
    if x.shape[1]==3:s.state.velocities[:,2]=.07*torch.sin(6.283185307179586*x[:,2])
    s.state.internalEnergies*=1+.15*torch.cos(6.283185307179586*x[:,0])
    from warpSPH.modules.eos import refreshEOS
    refreshEOS(s.state,ctx.schemeConfig)
    ctx.scratch['initialEnergy']=(s.state.masses.double()*(s.state.internalEnergies.double()+.5*s.state.velocities.double().square().sum(-1))).sum()
    return s


def assert_matching(a,b):
    assert not a.diverged and not b.diverged
    for k in ('positions','velocities','internalEnergies','densities','supports','sourceEnergyBudget'):
        torch.testing.assert_close(getattr(a.state.state,k),getattr(b.state.state,k),rtol=3e-6,atol=3e-7)
    if a.state.pairWork.work is not None and b.state.pairWork.work is not None:
        live=a.state.pairWork.adjacency.numEdges
        torch.testing.assert_close(a.state.pairWork.work[:live],b.state.pairWork.work[:live],rtol=3e-6,atol=3e-7)
    assert a.state.t==b.state.t
    assert min(b.trajectory[-1][k] for k in ('minDensity','minPressure','minEnergy'))>0
    assert abs(b.trajectory[-1]['relativeEnergyBudgetResidual'])<2e-6


@pytest.mark.parametrize('dim,integrator',[(2,'rungeKutta2'),(3,'rungeKutta2'),(2,'rungeKutta4')])
def test_native_work_recomputation_matches_expanded(dim,integrator):
    a=configured('expanded',dim,integrator).run(replace(case,buildSystem=patterned))
    b=configured('recompute',dim,integrator).run(replace(case,buildSystem=patterned))
    assert_matching(a,b)
    assert b.state.state.ap_ij is None and b.state.state.av_ij is None and b.state.state.f_ij is None


def test_rebuilt_list_replay_matches_reference(monkeypatch):
    module=importlib.import_module('warpSPH.modules.compressibleStage');original=module.stagePairList
    def changed(system,*args,**kwargs):
        adjacency=original(system,*args,**kwargs)
        return adjacency if system.pairWork.conservative else replace(adjacency)
    monkeypatch.setattr(module,'stagePairList',changed)
    a=configured('expanded',2,'rungeKutta4').run(replace(case,buildSystem=patterned))
    b=configured('recompute',2,'rungeKutta4').run(replace(case,buildSystem=patterned))
    assert_matching(a,b)
    assert b.state.pairWork.halfVelocity is not None


def test_storage_configuration_roundtrip_and_legacy_default():
    c=CRKSPHConfig(crkPairStorage='recompute');d=crkSPHConfigToDict(c)
    assert dictToCRKSPHConfig(d).crkPairStorage=='recompute'
    del d['crkPairStorage'];assert dictToCRKSPHConfig(d).crkPairStorage=='recompute'
    assert CRKSPHConfig().crkPairStorage=='recompute'
    with pytest.raises(ValueError,match='crkPairStorage'):CRKSPHConfig(crkPairStorage='unknown').validate()


def test_special_energy_control_keeps_expanded_path(monkeypatch):
    work=importlib.import_module('warpSPH.modules.crk.work')
    def unexpected(*args,**kwargs):raise AssertionError('special energy control used recomputation')
    monkeypatch.setattr(work,'accumulateCRKWork',unexpected)
    results=[]
    for mode in ('expanded','recompute'):
        c=RunConfiguration('sod',simulation=dict(device=os.environ.get('WARPSPH_TEST_DEVICE','cpu'),nx=32,nSteps=2,maxDt=.001,
            plot=False,video=False,store=False,progress=False,show=False,quiet=True),
            solver=dict(scheme='CRKSPH',crkPairStorage=mode,crkEnergyPartition='spheral',crkEnergyPredictor='compatible'))
        results.append(c.run())
    assert not any(r.diverged for r in results)
    for key in ('positions','velocities','internalEnergies','densities','supports'):
        torch.testing.assert_close(getattr(results[0].state.state,key),getattr(results[1].state.state,key),rtol=0,atol=0)


def test_checkpoint_can_change_only_pair_storage(tmp_path):
    from pathlib import Path
    full=configured('expanded');full.simulation['nSteps']=4
    reference=full.run()
    first=configured('expanded');first.simulation.update(store=True,storeMode='states',exportRoot=str(tmp_path/'first'))
    initial=first.run()
    resumed=configured('recompute');resumed.simulation['resumeFrom']=str(Path(initial.exportPath)/'trajectory/finalState.h5')
    assert_matching(reference,resumed.run())


def test_default_uses_shared_path(monkeypatch):
    work=importlib.import_module('warpSPH.modules.crk.work');original=work.accumulateCRKWork;called=[]
    def counted(*args,**kwargs):
        called.append(1);return original(*args,**kwargs)
    monkeypatch.setattr(work,'accumulateCRKWork',counted)
    result=configured().run(replace(case,buildSystem=patterned))
    assert not result.diverged and len(called)==4


@pytest.mark.parametrize('policy',[{'crkPairStorage':'expanded'},{'compatibleEnergy':False},
    {'crkEnergyPartition':'entropy'},{'crkEnergyPredictor':'compatible'},
    {'crkEvolution':'paper'},{'crkDensityPolicy':'sameMaterial'},{'crkViscosityGradient':'cached'},
    {'materials':[object()]},{'stepMode':'graph'},{'stepMode':'auto'}])
def test_default_preserves_reference_contexts(policy):
    from types import SimpleNamespace as NS
    from warpSPH.modules.crk.storage import useRecomputedWork
    scheme=CRKSPHConfig()
    for key,value in policy.items():setattr(scheme,key,value)
    assert not useRecomputedWork(NS(pairWork=NS(weights=(0.,1.))),NS(positions=torch.ones(2,2)),scheme,None,NS(volumes=torch.ones(2)),1)


def test_default_preserves_wall_and_differentiation_routes():
    from types import SimpleNamespace as NS
    from warpSPH.modules.crk.storage import useRecomputedWork
    system=NS(pairWork=NS(weights=(0.,1.)));state=NS(positions=torch.ones(2,2));stage=NS(volumes=torch.ones(2));scheme=CRKSPHConfig()
    assert useRecomputedWork(system,state,scheme,None,stage,1)
    assert not useRecomputedWork(system,state,scheme,object(),stage,1)
    state.positions.requires_grad_();assert not useRecomputedWork(system,state,scheme,None,stage,1)
    with torch.autograd.forward_ad.dual_level():
        state.positions=torch.autograd.forward_ad.make_dual(torch.ones(2,2),torch.ones(2,2))
        assert not useRecomputedWork(system,state,scheme,None,stage,1)


def test_differentiable_external_force_retains_reference_work(monkeypatch):
    crk=importlib.import_module('warpSPH.schemes.compressible.crkSPH')
    work=importlib.import_module('warpSPH.modules.crk.work')
    force=torch.tensor(.1,requires_grad=True)
    def external(system,*args):return force*system.state.masses[:,None]*torch.ones_like(system.state.velocities)
    def unexpected(*args,**kwargs):raise AssertionError('differentiable force used in-place work')
    monkeypatch.setattr(crk,'computeForcing',external);monkeypatch.setattr(work,'accumulateCRKWork',unexpected)
    cfg=configured();cfg.simulation.update(device='cpu',nSteps=1)
    r=cfg.run()
    assert not r.diverged and r.state.state.velocities.requires_grad
