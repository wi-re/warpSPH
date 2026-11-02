"""UID trajectory and physical checkpoint lifetimes on tiny CPU fixtures."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from warpSPH.evaluation import MortonStepper, adopt_spatial_layout, canonical_view
from warpSPH.io.export import exportSimulationSystem, writeFrame, writeInitialData
from warpSPH.runner import run
from test_particleLayout import shuffled, particle_pairs
from test_runnerParticleOrder import fixture_case, options


def uid_case(**changes):
    base=fixture_case(**changes)
    def build(ctx):
        system=base.buildSystem(ctx)
        system.state.UIDs=torch.randperm(len(system.state.UIDs),generator=torch.Generator().manual_seed(91)).to(ctx.device)+2**40
        system.state.UIDcounter=2**40+len(system.state.UIDs)
        return system
    return replace(base,buildSystem=build)


def ordered(system):
    return canonical_view(system,include_adjacency=False)[0].state


@pytest.mark.parametrize('method',['CompSPH','CRKSPH'])
def test_runner_trajectory_joins_every_frame_to_int64_uid_roster_across_resorts(method,tmp_path):
    seen={}
    def move(ctx,state,i):
        state.state.positions.add_(.21)
        seen[i]=ordered(state)
    case=replace(uid_case(postStep=move),extraFields=('internalEnergies','supports'))
    args=options();args.update(scheme=method,particleOrder='morton',store=True,storeMode='trajectory',
                              exportInterval=1e-6,exportRoot=str(tmp_path))
    result=run(case,**args)
    assert not result.diverged and result.nSteps==3
    with h5py.File(Path(result.exportPath)/'trajectory.h5') as f:
        assert f.attrs['particleLayoutVersion']==1
        ids=f['combinedUIDs'][:]
        assert ids.dtype==np.int64 and min(ids)>=2**40
        assert (ids==np.sort(ids)).all()
        assert f['fluidUIDs'].dtype==np.int64
        assert f['combinedPositions'].dtype==seen[0].positions.numpy().dtype
        for i,snapshot in seen.items():
            key=f'frame_{i:05d}'
            assert f['frameUIDs'][key].id==f['combinedUIDs'].id
            for name in ('positions','velocities','densities','internalEnergies','supports'):
                np.testing.assert_array_equal(f[name][key][:],getattr(snapshot,name).numpy())
                assert f[name].attrs['axis']=='particle'
                assert f[name].attrs['uidDataset']=='/combinedUIDs'


def test_direct_trajectory_stage_axes_global_budget_and_pre_post_adoption_are_consistent(tmp_path):
    ctx,initial=shuffled()
    initial.state.UIDs=initial.state.UIDs.long()+2**40
    initial.state.UIDcounter=2**40+1000
    result=MortonStepper(ctx.integrator)(initial,1e-4,config=ctx.config,schemeConfig=ctx.schemeConfig)
    result.state.state.emitters=torch.arange(32,dtype=torch.int64,device=ctx.device)+2**41
    result.state.state.alphas=torch.arange(32,dtype=ctx.dtype,device=ctx.device)/32
    result.state.state.boundaryProjectionBudget=torch.arange(32,device=ctx.device,dtype=ctx.dtype)
    (tmp_path/'config.json').write_text('{}')
    extra=('internalEnergies','supports','boundaryProjectionBudget','alphas')
    with h5py.File(tmp_path/'trajectory.h5','w') as f:
        groups=writeInitialData(str(tmp_path),f,ctx.scheme,ctx.config,ctx.schemeConfig,
            SimpleNamespace(exportInterval=1e-4),result.state,extraFields=extra)
        writeFrame(groups,0,result.state,result.stages,ctx.config,ctx.schemeConfig,
                   writeStages=True,extraFields=extra,extraData={'label':'before'})
        adopted=adopt_spatial_layout(result)
        writeFrame(groups,1,adopted.state,adopted.stages,ctx.config,ctx.schemeConfig,
                   writeStages=True,extraFields=extra,extraData={'label':'after'})
        for name in ('positions','velocities','densities',*extra):
            np.testing.assert_array_equal(f[name]['frame_00000'][:],f[name]['frame_00001'][:])
        assert f['boundaryProjectionBudget'].attrs['axis']=='global'
        np.testing.assert_array_equal(f['boundaryProjectionBudget/frame_00000'][:],np.arange(32))
        for i,stage in enumerate(result.stages):
            group=f[f'stages/frame_00000/stage_{i}']
            order=torch.argsort(stage.aux[1].UIDs)
            np.testing.assert_array_equal(group['dvdt'][:],stage.update.dvdt[order].numpy())
            ids=group['aux/1_state/UIDs'][:]
            np.testing.assert_array_equal(ids,f['combinedUIDs'][:])
            adj=group['aux/0_adjacency']
            actual=sorted(zip(ids[adj['i'][:]],ids[adj['j'][:]]))
            assert actual==particle_pairs(stage.aux[2].adjacency,stage.aux[2].layout.compute_ids)
        # Failure must precede any datasets for that frame.
        state=result.state.initializeNewState();state.state.UIDs[0]+=10000
        with pytest.raises(ValueError,match='membership/order|roster'):
            writeFrame(groups,2,state,result.stages,ctx.config,ctx.schemeConfig,extraFields=extra)
        assert 'frame_00002' not in f['frameUIDs'] and 'frame_00002' not in f['positions']
        with pytest.raises(ValueError,match='header axes'):
            writeFrame(groups,2,result.state,None,ctx.config,ctx.schemeConfig,extraFields=extra[::-1])
        with pytest.raises(ValueError,match='explicit axes'):
            writeFrame(groups,2,result.state,None,ctx.config,ctx.schemeConfig,extraFields=extra,
                       extraData={'unknownRows':torch.arange(32)})
        assert 'frame_00002' not in f['positions']
        changed=result.state.initializeNewState();changed.state.masses*=1.01
        with pytest.raises(ValueError,match='masses changed'):
            writeFrame(groups,2,changed,None,ctx.config,ctx.schemeConfig,extraFields=extra)
        assert 'frame_00002' not in f['positions']


    from warpSPH.io.importIO import loadTrajectory,loadTrajectoryFrame
    handle,meta=loadTrajectory(str(tmp_path),ctx.device,extraFields=extra)
    try:
        loaded,_=loadTrajectoryFrame(handle,meta,0,schemeConfig=ctx.schemeConfig)
        assert loaded.state.UIDs.dtype==torch.int64 and loaded.state.UIDcounter==2**40+1000
        assert loaded.state.emitters.dtype==torch.int64 and min(loaded.state.emitters)>=2**41
        assert loaded.canonicalExport
        for name in extra:
            torch.testing.assert_close(getattr(loaded.state,name),getattr(ordered(result.state),name),rtol=0,atol=0)
    finally:handle.close()


@pytest.mark.parametrize('method',['CompSPH','CRKSPH'])
def test_state_checkpoint_rebuilds_cache_and_continues_physics_and_switch_history(method,tmp_path,monkeypatch):
    case=uid_case()
    args=options();args.update(nSteps=4,scheme=method,particleOrder='morton',params={'viscositySwitch':'Rosswog2020'})
    control=run(case,**args)
    args.update(nSteps=2,store=True,exportRoot=str(tmp_path/'prefix'))
    prefix=run(case,**args)
    checkpoint=Path(prefix.exportPath)/'trajectory/finalState.h5'
    entered=[]
    original=MortonStepper.__call__
    def step(self,state,*a,**kw):
        if not entered:
            assert state.spatialContext is None and state.adjacency is None and state.pairWork is None
            assert state.supportGrid is None and state.listGrid is None
            assert state.state.UIDs.tolist()==sorted(state.state.UIDs.tolist())
        entered.append(state)
        return original(self,state,*a,**kw)
    monkeypatch.setattr(MortonStepper,'__call__',step)
    args.update(resumeFrom=str(checkpoint),resumeStepOffset=2,exportRoot=str(tmp_path/'resume'))
    resumed=run(case,**args)
    assert not resumed.diverged and resumed.nSteps==2 and len(entered)==2
    assert [r['step'] for r in resumed.trajectory[1:]]==[2,3]
    a,b=ordered(control.state),ordered(resumed.state)
    for name in ('UIDs','positions','velocities','internalEnergies','supports','densities','alphas','entropiesPrev','divergencePrevStep'):
        x,y=getattr(a,name),getattr(b,name)
        assert (x is None)==(y is None)
        if x is not None:torch.testing.assert_close(x,y,rtol=3e-6,atol=3e-7)
    assert a.UIDcounter==b.UIDcounter==2**40+32


@pytest.mark.parametrize('corrupt, message',[
    ('domain','checkpoint domain'),('precision','checkpoint precision'),
    ('scheme','checkpoint scheme'),('roster','checkpoint UID roster'),('schema','UID-layout state checkpoint'),
    ('supports','checkpoint supports'),
])
def test_restart_rejects_mismatched_checkpoint_before_output(corrupt,message,tmp_path):
    case=uid_case()
    args=options();args.update(nSteps=1,scheme='CompSPH',particleOrder='morton',store=True,
                              exportRoot=str(tmp_path/'prefix'))
    prefix=run(case,**args)
    checkpoint=Path(prefix.exportPath)/'trajectory/finalState.h5'
    with h5py.File(checkpoint,'r+') as f:
        if corrupt=='domain':f.attrs['particleDomain_min']=np.array([-123.])
        if corrupt=='precision':f.attrs['particlePrecision']='unsupported'
        if corrupt=='scheme':f.attrs['scheme']='other'
        if corrupt=='roster':f['state/UIDs'][0]=2**45
        if corrupt=='schema':del f.attrs['particleLayoutVersion']
        if corrupt=='supports':f['state/supports'][0]=-1
    args.update(resumeFrom=str(checkpoint),exportRoot=str(tmp_path/'rejected'))
    with pytest.raises(ValueError,match=message):run(case,**args)
    assert not (tmp_path/'rejected').exists()



def test_renewed_implicit_checkpoint_restarts_with_fresh_candidates(tmp_path):
    from warpSPH.cases.compressible.linearWave import linearWaveCase
    from warpSPH.enumTypes import AdaptiveSupportScheme
    def configure(ctx):
        linearWaveCase.configureScheme(ctx)
        ctx.schemeConfig.compatibleEnergy=False
        ctx.schemeConfig.adaptiveSupportScheme=AdaptiveSupportScheme.NoScheme
    def initial(ctx,system):
        linearWaveCase.initialConditions(ctx,system)
        s=system.state
        s.velocities+=.05*torch.sin(2*torch.pi*s.positions)
        s.totalEnergies=s.masses*(s.internalEnergies+.5*s.velocities.square().sum(-1))
    case=replace(uid_case(),configureScheme=configure,initialConditions=initial)
    args=options();args.update(nSteps=2,scheme='CompSPH',particleOrder='morton',
                              integrationScheme='backwardEuler',verletScale=1.)
    with torch.no_grad():
        control=run(case,**args)
        args.update(nSteps=1,store=True,exportRoot=str(tmp_path/'prefix'))
        prefix=run(case,**args)
        assert prefix.state.spatialContext.coverage_renewals>=1
        args.update(resumeFrom=str(Path(prefix.exportPath)/'trajectory/finalState.h5'),resumeStepOffset=1,
                    exportRoot=str(tmp_path/'resume'))
        resumed=run(case,**args)
    assert not resumed.diverged and resumed.state.spatialContext.coverage_renewals>=1
    for name in ('positions','velocities','internalEnergies','densities','supports'):
        torch.testing.assert_close(getattr(ordered(control.state),name),getattr(ordered(resumed.state),name),
                                   rtol=3e-5,atol=3e-6)


def test_scalar_extras_cannot_override_layout_associations(tmp_path):
    ctx,initial=shuffled()
    result=MortonStepper(ctx.integrator)(initial,1e-4,config=ctx.config,schemeConfig=ctx.schemeConfig)
    with pytest.raises(ValueError,match='reserved layout metadata'):
        exportSimulationSystem(str(tmp_path),'bad','CompSPH',result.state,extraData={'particleLayoutVersion':99})
    assert not (tmp_path/'trajectory').exists()



@pytest.mark.parametrize('dim',[1,2,3])
def test_full_frame_reader_handles_layout_array_metadata_and_preserves_uid64(dim,tmp_path):
    from warpSPH.io.importIO import importSimulationSystem
    from warpSPH.utils.domain import buildDomainDescription
    ctx,initial=shuffled()
    initial.state.UIDs=initial.state.UIDs.long()+2**40
    initial.state.UIDcounter=2**40+32
    initial.state.positions=initial.state.positions.repeat(1,dim)
    initial.state.velocities=initial.state.velocities.repeat(1,dim)
    initial.domain=buildDomainDescription(l=2,dim=dim,periodic=True,device=ctx.device,dtype=ctx.dtype)
    initial.canonicalExport=True
    exportSimulationSystem(str(tmp_path),'frame',ctx.scheme,initial,extraData={'label':'reader'})
    loaded,stages,scheme,extras=importSimulationSystem(str(tmp_path/'trajectory/frame.h5'),ctx.device,ctx.dtype)
    assert loaded.canonicalExport and loaded.spatialContext is None
    assert loaded.state.UIDs.dtype==torch.int64 and min(loaded.state.UIDs)>=2**40
    assert loaded.domain.dim==dim and extras['label']=='reader'
    torch.testing.assert_close(loaded.domain.min,initial.domain.min,rtol=0,atol=0)
    torch.testing.assert_close(loaded.state.positions,ordered(initial).positions,rtol=0,atol=0)



def test_uid_frame_reader_preserves_numeric_stage_order_beyond_nine(tmp_path):
    from warpSPH.io.importIO import importSimulationSystem
    ctx,initial=shuffled()
    result=MortonStepper(ctx.integrator)(initial,1e-4,config=ctx.config,schemeConfig=ctx.schemeConfig)
    prototype=result.stages[0]
    stages=[prototype._replace(update=replace(prototype.update,dvdt=torch.full_like(prototype.update.dvdt,i))) for i in range(12)]
    exportSimulationSystem(str(tmp_path),'many',ctx.scheme,result.state,stages=stages)
    _,loaded,_,_=importSimulationSystem(str(tmp_path/'trajectory/many.h5'),ctx.device,ctx.dtype)
    assert [int(stage.update.dvdt[0,0]) for stage in loaded]==list(range(12))
