"""Prescribed thermal state at startup/acceptance, projection exchange and replay."""
from dataclasses import replace
from pathlib import Path
import sys
import h5py
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from probe_thermalBoundaries import boundary_toy
from test_pesphOperators import setup, DOUBLE, TOL
from warpSPH.configurations import BoundaryCondition,BoundaryConditionType
from warpSPH.enumTypes import PESPHVariant
from warpSPH.modules.boundaryConditions.thermal import finishThermalDirichlet,caloricEnergy
from warpSPH.modules.pesph.stage import refreshPESPHState
from warpSPH.systems.pesph import PESPHSystem


@pytest.mark.parametrize('scheme,variant',[(s,None) for s in ['CRKSPH','CompSPH','Monaghan','GSPH','InutsukaGSPH']]+[('PESPH','PressureEnergy'),('PESPH','PressureEntropy')])
@pytest.mark.parametrize('tableau',['rungeKutta2','rungeKutta4'])
def test_native_accepted_temperature_and_projection_heat_match_analytic_ramp(scheme,variant,tableau,tmp_path):
    row=boundary_toy(scheme,variant,16,.002,.004,tableau,'uniform',tmp_path)
    tolerance=2e-10 if DOUBLE else 3e-6
    assert not row['diverged'] and row['steps']==2
    assert row['initialURelativeError']<tolerance
    assert row['maxStageURelativeError']<tolerance
    assert row['acceptedURelativeError']<tolerance
    assert abs(row['caloricResidualIncludingInitialReset'])<3*tolerance
    thermal,kinetic=row['boundaryProjectionBudget']
    assert thermal==pytest.approx(.008,abs=3*tolerance)
    assert abs(kinetic)<tolerance


@pytest.mark.parametrize('gamma',[1.4,5/3,3.])
@pytest.mark.parametrize('target',[.2,3.])
def test_entropy_boundary_holds_u_without_changing_unconstrained_A_and_counts_nonlocal_energy(gamma,target):
    state,cfg,scheme=setup(variant=PESPHVariant.PressureEntropy)
    scheme.gamma=gamma
    system=PESPHSystem(state=state,domain=cfg.domain)
    refreshPESPHState(system,cfg,scheme)
    originalA=state.entropies.clone(); originalU=state.internalEnergies.clone()
    before=caloricEnergy(state)
    mask=state.positions[:,0]<0
    scheme.boundaryConditions=[BoundaryCondition(BoundaryConditionType.dynamic,
        sdf=lambda x:(x[:,0],torch.zeros_like(x)),
        dirichletFunctions={'internalEnergies':lambda s,*args:torch.full_like(s.masses,target)})]
    assert finishThermalDirichlet(system,0.,cfg,scheme)
    torch.testing.assert_close(state.internalEnergies[mask],torch.full_like(state.masses[mask],target),atol=TOL,rtol=TOL)
    assert torch.equal(state.entropies[~mask],originalA[~mask])
    # The free rows retain A, not u: the pressure sum changes their energy.
    assert (state.internalEnergies[~mask]-originalU[~mask]).abs().max()>1e-4
    torch.testing.assert_close(state.boundaryProjectionBudget,caloricEnergy(state)-before,atol=1e-12,rtol=1e-12)
    localOnly=(state.masses[mask].double()*(state.internalEnergies[mask].double()-originalU[mask].double())).sum()
    assert abs(float(state.boundaryProjectionBudget[0]-localOnly))>1e-4


def test_entropy_boundary_iteration_failure_is_explicit_and_does_not_commit_A():
    from warpSPH.modules.pesph.thermalBoundary import matchBoundaryInternalEnergy
    state,cfg,scheme=setup(variant=PESPHVariant.PressureEntropy)
    system=PESPHSystem(state=state,domain=cfg.domain)
    refreshPESPHState(system,cfg,scheme)
    old=state.entropies.clone()
    with pytest.raises(RuntimeError,match='did not converge'):
        matchBoundaryInternalEnergy(state,cfg,scheme,torch.full_like(state.masses,3.),
            torch.ones_like(state.masses,dtype=torch.bool),system.adjacency,maxIterations=0)
    assert torch.equal(state.entropies,old)


def test_no_thermal_boundary_is_a_bitwise_noop():
    state,cfg,scheme=setup(variant=PESPHVariant.PressureEntropy)
    system=PESPHSystem(state=state,domain=cfg.domain)
    before={k:v.clone() for k,v in vars(state).items() if isinstance(v,torch.Tensor)}
    assert not finishThermalDirichlet(system,.01,cfg,scheme)
    for name,value in before.items():assert torch.equal(getattr(state,name),value)
    assert state.boundaryProjectionBudget is None


@pytest.mark.parametrize('scheme',['CRKSPH','CompSPH'])
def test_compatible_replay_does_not_count_projection_twice(scheme,monkeypatch,tmp_path):
    import warpSPH.modules.compSPH.pairWork as pw
    import warpSPH.schemes.compressible.crkSPH as crk
    import warpSPH.schemes.compressible.compSPH as comp
    baseline=boundary_toy(scheme,None,16,.002,.004,'rungeKutta2','uniform',tmp_path)
    original=pw.accumulateStage;count=[0]
    def spoiled(pair,stageIndex,adjacency,state,energyScheme):
        original(pair,stageIndex,adjacency,state,energyScheme)
        if stageIndex==1 and pair.halfVelocity is None:
            pair.consistent,pair.work=False,None;count[0]+=1
    monkeypatch.setattr(crk,'accumulateStage',spoiled);monkeypatch.setattr(comp,'accumulateStage',spoiled)
    import warpSPH.modules.crk.work as crkWork
    shared=crkWork.accumulateCRKWork
    def spoiledShared(system,state,update,config,schemeConfig,adjacency,stage,index,dt):
        shared(system,state,update,config,schemeConfig,adjacency,stage,index,dt)
        if index==1 and system.pairWork.halfVelocity is None:
            system.pairWork.consistent,system.pairWork.work=False,None;count[0]+=1
    monkeypatch.setattr(crkWork,'accumulateCRKWork',spoiledShared)
    other=tmp_path/'replay';other.mkdir()
    replay=boundary_toy(scheme,None,16,.002,.004,'rungeKutta2','uniform',other)
    assert count[0]==2
    assert replay['boundaryProjectionBudget']==baseline['boundaryProjectionBudget']
    assert replay['acceptedURelativeError']==baseline['acceptedURelativeError']


@pytest.mark.parametrize('variant',[PESPHVariant.PressureEnergy,PESPHVariant.PressureEntropy])
def test_projection_budget_survives_native_state_roundtrip(variant,tmp_path):
    from warpSPH.io.hdf5 import dumpState,loadState
    state,cfg,scheme=setup(variant=variant)
    system=PESPHSystem(state=state,domain=cfg.domain)
    refreshPESPHState(system,cfg,scheme)
    scheme.boundaryConditions=[BoundaryCondition(BoundaryConditionType.dynamic,
        sdf=lambda x:(x[:,0],torch.zeros_like(x)),
        dirichletFunctions={'internalEnergies':lambda s,*args:torch.full_like(s.masses,3.)})]
    finishThermalDirichlet(system,0.,cfg,scheme)
    with h5py.File(tmp_path/'state.h5','w') as f:dumpState(state,f.create_group('state'))
    with h5py.File(tmp_path/'state.h5','r') as f:restored=loadState(f['state'],'cpu',type(state))
    assert torch.equal(restored.boundaryProjectionBudget,state.boundaryProjectionBudget)


@pytest.mark.parametrize('scheme',['MFM','MFV'])
@pytest.mark.parametrize('temporal',['ssprk','hancock','kdk'])
def test_meshless_prescribed_u_uses_fresh_geometry_and_counts_accepted_projection(scheme,temporal):
    from warpSPH.runner import RunConfiguration
    from warpSPH.cases.compressible.sod import sodCase
    from warpSPH.modules.meshless.boundary import DrivenParticles
    from warpSPH.modules.boundaryConditions.thermal import caloricConserved
    saved={}
    def build(ctx):
        ctx.config.adaptiveDt=True
        system=sodCase.buildSystem(ctx)
        ctx.config.adaptiveDt=False;ctx.config.dt=.002
        s=system.state;reference=s.positions.clone()
        velocity=.1*torch.sin(torch.pi*reference)
        system.drivenParticles=DrivenParticles(torch.ones_like(s.masses,dtype=torch.bool),reference,s.masses.clone(),
            lambda t,x:x+t*velocity,lambda t,x:(velocity,None),
            specificInternalEnergy=lambda t,x:torch.full_like(s.masses,1.+float(t)))
        # Define the consistent prescribed initial state; its construction is
        # not a heat input during the measured integration interval.
        s.Q[:,1:-1]=s.masses[:,None]*velocity
        s.Q[:,-1]=s.masses*(1.+.5*velocity.square().sum(-1))
        saved['initialEnergy']=caloricConserved(s.Q).sum().clone()
        return system
    cfg=RunConfiguration('sod',simulation=dict(device='cpu',nx=16,nSteps=2,dt=.002,adaptiveDt=False,
        plot=False,video=False,store=False,show=False,quiet=True,progress=False),
        params=dict(gamma=1.4,samplingRatio=1,left_rho=1.,right_rho=1.,left_pressure=.4,right_pressure=.4),
        solver=dict(scheme=scheme,temporalScheme=temporal))
    result=cfg.run(replace(sodCase,buildSystem=build))
    s=result.state.state;target=1.+float(result.state.t)
    tolerance=2e-10 if DOUBLE else 4e-6
    torch.testing.assert_close(s.internalEnergies,torch.full_like(s.masses,target),rtol=tolerance,atol=tolerance)
    torch.testing.assert_close(s.pressures,.4*s.densities*s.internalEnergies,rtol=tolerance,atol=tolerance)
    delta=caloricConserved(s.Q).sum()-saved['initialEnergy']
    torch.testing.assert_close(s.boundaryProjectionBudget.sum(),delta,rtol=0,atol=3*tolerance)
    assert not result.diverged


def test_meshless_u_constraint_is_independent_of_volume_and_checks_positivity():
    from types import SimpleNamespace
    from warpSPH.modules.meshless.boundary import DrivenParticles
    mass=torch.ones(4)/4;position=torch.zeros(4,1)
    s=SimpleNamespace(Q=torch.ones(4,3),positions=position)
    driver=DrivenParticles(torch.ones(4,dtype=torch.bool),position,mass,lambda t,x:x,
        lambda t,x:(torch.zeros_like(x),None),specificInternalEnergy=lambda t,x:1.)
    for volume in [mass,mass*2,mass*.1]:
        driver.conserved(s,SimpleNamespace(volume=volume),0.,1.4)
        assert torch.equal(s.Q[:,-1],mass)
    driver.specificInternalEnergy=lambda t,x:0.
    with pytest.raises(RuntimeError,match='finite and positive'):
        driver.conserved(s,SimpleNamespace(volume=mass),0.,1.4)


@pytest.mark.parametrize('scheme,variant',[('CRKSPH',None),('Monaghan',None),('PESPH','PressureEntropy')])
def test_native_restart_continues_thermal_constraint_and_exchange_budget(scheme,variant,tmp_path):
    from warpSPH.runner import RunConfiguration
    from warpSPH.cases.compressible.sod import sodCase
    from warpSPH.modules.eos import refreshEOS
    def build(ctx):
        system=sodCase.buildSystem(ctx);s=system.state
        s.internalEnergies.fill_(1.);s.velocities.zero_();refreshEOS(s,ctx.schemeConfig)
        s.totalEnergies=s.masses*s.internalEnergies
        ctx.schemeConfig.boundaryConditions=[BoundaryCondition(BoundaryConditionType.dynamic,
            sdf=lambda x:(-torch.ones_like(x[:,0]),torch.zeros_like(x)),
            dirichletFunctions={'internalEnergies':lambda s,cfg,sc,x,d,n,t,dt:torch.full_like(s.masses,1.+float(t))},
            updateFunctions={'dxdt':lambda s,*args:torch.zeros_like(s.velocities),
                             'dvdt':lambda s,*args:torch.zeros_like(s.velocities)})]
        return system
    case=replace(sodCase,buildSystem=build)
    def config(steps,**kwargs):
        solver=dict(scheme=scheme)
        if variant:solver.update(variant=variant,dissipation=False)
        return RunConfiguration('sod',simulation=dict(device='cpu',nx=16,nSteps=steps,dt=.002,adaptiveDt=False,
            plot=False,video=False,store=False,show=False,quiet=True,progress=False,**kwargs),
            params=dict(gamma=1.4,samplingRatio=1,left_rho=1.,right_rho=1.,left_pressure=.4,right_pressure=.4),solver=solver)
    full=config(10).run(case)
    firstCfg=config(5);firstCfg.simulation.update(store=True,storeMode='states',exportRoot=str(tmp_path/'first'))
    first=firstCfg.run(case)
    path=Path(first.exportPath)/'trajectory/finalState.h5'
    resumed=config(5,resumeFrom=str(path)).run(case)
    tolerance=2e-10 if DOUBLE else 4e-6
    assert float(full.state.t)==float(resumed.state.t)
    for name in ['positions','velocities','internalEnergies','boundaryProjectionBudget']:
        torch.testing.assert_close(getattr(resumed.state.state,name),getattr(full.state.state,name),rtol=tolerance,atol=tolerance)


@pytest.mark.parametrize('scheme',['CRKSPH','Monaghan','PESPH'])
def test_trajectory_frame_restores_projection_budget(scheme,tmp_path):
    from warpSPH.runner import RunConfiguration
    from warpSPH.io import loadTrajectory,loadTrajectoryFrame
    solver=dict(scheme=scheme)
    if scheme=='PESPH':solver.update(variant='PressureEntropy',dissipation=False)
    cfg=RunConfiguration('rayleighTaylor',simulation=dict(device='cpu',nx=16,nSteps=2,maxDt=.001,
        plot=False,video=False,store=True,storeMode='trajectory',storeInterval=1,
        exportRoot=str(tmp_path/'trajectory'),show=False,quiet=True,progress=False),
        params=dict(band=2),solver=solver)
    result=cfg.run()
    f,meta=loadTrajectory(result.exportPath,'cpu',extraFields=('internalEnergies','supports'))
    try:
        assert 'boundaryProjectionBudget' in f
        restored,_=loadTrajectoryFrame(f,meta,-1,schemeConfig=result.ctx.schemeConfig)
    finally:f.close()
    assert torch.equal(restored.state.boundaryProjectionBudget,result.state.state.boundaryProjectionBudget)
