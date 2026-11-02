"""Tiny evolving A/B fixtures; video and viewer intentionally omitted."""
from dataclasses import replace
from pathlib import Path

import h5py
import pytest
import torch

from warpSPH.cases.periodic.tgv import tgvCase
from warpSPH.cases.periodic.tgvWeaklyCompressible import tgvWeaklyCompressibleCase
from warpSPH.evaluation import canonical_view
from warpSPH.evaluation.nativeLayout import HISTORY_ROWS
from warpSPH.runner import run


def fixture(scheme,**overrides):
    projection=scheme in ('iisph','dfsph','dfsphPredictor','dfsphShift')
    case=tgvCase if projection else tgvWeaklyCompressibleCase
    opts=dict(scheme=scheme,nx=8,n_h=2,nSteps=3,dt=1e-2 if projection else 1e-4,adaptiveDt=False,
        device='cpu',quiet=True,progress=True,plot=False,video=False,store=False,
        params=dict(relaxSteps=0,jitter=0,shifting=False) if projection else
                    dict(shuffleIters=0,jitter=0,shifting=False,targetDt=1e-4))
    opts.update(overrides)
    return case,opts


def close(a,b):
    aa=canonical_view(a,include_adjacency=False)[0]
    bb=canonical_view(b,include_adjacency=False)[0]
    for name in ('UIDs','positions','velocities','supports','densities','pressures'):
        x,y=getattr(aa.state,name),getattr(bb.state,name)
        if name=='pressures' and type(a).__name__ in ('IISPHSystem','DFSPHSystem'):
            # Periodic projection pressure has an arbitrary additive gauge.
            x,y=x-x.mean(),y-y.mean()
        # At fp32 the periodic projection's near-null pressure modes amplify
        # row-reduction rounding. FP64 uses the strict threshold for all fields.
        projection32=name=='pressures' and type(a).__name__ in ('IISPHSystem','DFSPHSystem') and x.dtype==torch.float32
        torch.testing.assert_close(x,y,rtol=2e-4,atol=5e-3 if projection32 else 3e-5)
    for name in HISTORY_ROWS:
        if hasattr(aa,name) and getattr(aa,name) is not None:
            torch.testing.assert_close(getattr(aa,name),getattr(bb,name),rtol=2e-4,atol=3e-5)


@pytest.mark.parametrize('scheme',['deltaSPH','sun2017DeltaSPH','artificialCompressible','iisph','dfsph','dfsphPredictor','dfsphShift'])
def test_native_runner_evolving_equivalence_and_rebuild(scheme):
    case,opts=fixture(scheme)
    def translate(ctx,system,i):
        system.state.positions.add_(.71)
    case=replace(case,postStep=translate)
    baseline=run(case,**opts)
    sorted_=run(case,**opts,particleOrder='morton')
    assert not baseline.diverged and not sorted_.diverged
    assert sorted_.ctx.scratch['nativeLayout'].generation==3
    close(sorted_.state,baseline.state)
    print(f'{scheme}: native {baseline.wallTime:.3f}s / Morton {sorted_.wallTime:.3f}s; '
          f'{sorted_.series("stepTime_ms").mean():.3f} ms/step',flush=True)


@pytest.mark.parametrize('scheme',['deltaSPH','artificialCompressible','iisph','dfsph'])
def test_native_checkpoint_preserves_uid_history(scheme,tmp_path):
    case,opts=fixture(scheme,store=True,storeInterval=1,exportRoot=str(tmp_path),particleOrder='morton')
    full=run(case,**opts)
    checkpoint=Path(full.exportPath,'trajectory','state_0001.h5')
    # File tags are selected by the shared runner, not by the layout adapter.
    if not checkpoint.exists():
        checkpoint=next(p for p in Path(full.exportPath,'trajectory').glob('*.h5') if p.stem not in ('initialState','finalState'))
    with h5py.File(checkpoint) as f:
        offset=int(f.attrs['frame_num'])+1
        ids=f['state/UIDs'][:].tolist()
        assert ids==sorted(ids)
        assert 'nativeMemory' in f
    from warpSPH.io.importIO import importSimulationSystem
    imported,stages,_,_=importSimulationSystem(str(checkpoint),device='cpu',dtype=full.state.state.positions.dtype)
    assert imported.state.UIDs.tolist()==ids and stages
    opts.update(nSteps=3-offset,resumeFrom=str(checkpoint),resumeStepOffset=offset,store=False)
    resumed=run(case,**opts)
    close(resumed.state,full.state)


@pytest.mark.parametrize('scheme',['deltaSPH','artificialCompressible','iisph','dfsph'])
def test_native_trajectory_uses_declared_uid_axes(scheme,tmp_path):
    case,opts=fixture(scheme,store=True,storeMode='trajectory',exportInterval=1e-5,
        exportRoot=str(tmp_path),particleOrder='morton')
    result=run(case,**opts)
    path=next(Path(result.exportPath).glob('*.hdf5'),None)
    if path is None:path=next(Path(result.exportPath).glob('*.h5'))
    with h5py.File(path) as f:
        assert f['combinedUIDs'][:].tolist()==sorted(result.state.state.UIDs.tolist())
        assert len(f['frameUIDs'])==result.nSteps
        assert f['positions'].attrs['axis']=='particle'


def test_native_moving_body_keeps_local_geometry_and_global_ghost_scatter():
    from types import SimpleNamespace
    from test_particleBoundaryInputs import ghost_state
    from warpSPH.configurations.region import RegionType,BCType
    from warpSPH.evaluation import AllParticles
    from warpSPH.evaluation.nativeLayout import system_view,config_view
    from warpSPH.rigidBody.build import buildRigidBody
    from warpSPH.rigidBody import integrateRigidBody,updateBodyParticlesWCSPH
    from warpSPH.systems.weaklyCompressible import WeaklyCompressibleSystem
    state=ghost_state()
    body=buildRigidBody(state,[SimpleNamespace(type=RegionType.Boundary,sdf=AllParticles(),kind=BCType.constant)],0)
    body.linearVelocity=state.positions.new_tensor([.1,-.2])
    body.angularVelocity=state.positions.new_tensor(1.3)
    original=WeaklyCompressibleSystem(state)
    cfg=SimpleNamespace();scheme=SimpleNamespace(rigidBodies=[body])
    current=original
    for i in range(4):
        p=torch.tensor([4,2,1,3,0])
        current=system_view(current,p)
        cfg,scheme=config_view(cfg,scheme,current.state)
        original.state=updateBodyParticlesWCSPH(original.state,integrateRigidBody(body,0,0,.01))
        current.state=updateBodyParticlesWCSPH(current.state,integrateRigidBody(scheme.rigidBodies[0],0,0,.01))
        a,b=canonical_view(original,include_adjacency=False)[0].state,canonical_view(current,include_adjacency=False)[0].state
        for name in ('positions','velocities','ghostIndices','ghostOffsets','boundaryAccelerations'):
            torch.testing.assert_close(getattr(a,name),getattr(b,name),rtol=0,atol=0)


def test_native_uid_forcing_and_pinned_geometry_follow_rows():
    from warpSPH.evaluation import UIDValues,AllParticles
    from warpSPH.configurations.moduleConfigurations.boundaryConditions import BoundaryCondition,BoundaryConditionType
    from warpSPH.modules.boundaryConditions.pinned import PinnedBand,pinnedBandBC
    case,opts=fixture('deltaSPH')
    initialize=case.initialConditions
    def initial(ctx,system):
        initialize(ctx,system)
        values=torch.stack((torch.sin(system.state.UIDs.float()),torch.cos(system.state.UIDs.float())),dim=-1).to(system.state.positions)*.001
        ctx.schemeConfig.boundaryConditions += [BoundaryCondition(type=BoundaryConditionType.dynamic,sdf=AllParticles(),forcingFunctions=[UIDValues(system.state.UIDs,values)]),
            pinnedBandBC(PinnedBand(((0,.2,.15),),velocity=(.1,0.),hi=(float(ctx.spec.L),)*2))]
    case=replace(case,initialConditions=initial)
    baseline=run(case,**opts)
    ordered=run(case,**opts,particleOrder='morton')
    close(ordered.state,baseline.state)


@pytest.mark.parametrize('solid',['inside','outside'])
def test_native_analytic_provider_and_body_pose_restart(tmp_path,solid):
    from types import SimpleNamespace
    from warpSPH.runner import prepareContext,CaseSpec
    from warpSPH.evaluation import AllParticles
    from warpSPH.evaluation.nativeLayout import system_view,config_view
    from warpSPH.io.export import exportSimulationSystem
    from warpSPH.io.particleLayout import restoreCheckpoint
    from warpSPH.io.nativeMemory import readNativeMemory
    from warpSPH.rigidBody import buildAnalyticRigidBody
    from warpSPH.configurations.region import BCType
    from warpSPH.boundary.provider import bindBodies
    from warpSPHBoundaries.scene import Body,Scene,AnalyticBoundary
    from warpSPHBoundaries.scene.scene import BoxRep
    ctx=prepareContext(tgvWeaklyCompressibleCase,CaseSpec(scheme='deltaSPH',device='cpu',nx=8,
        params=dict(tgvWeaklyCompressibleCase.params,shuffleIters=0,jitter=0,shifting=False)))
    system=tgvWeaklyCompressibleCase.buildSystem(ctx)
    geometry=Body(center=(.5,.5),reps=[BoxRep((-.1,-.1),(.1,.1),solid=solid)])
    provider=AnalyticBoundary(Scene([geometry],'cpu'))
    body=buildAnalyticRigidBody(SimpleNamespace(sdf=AllParticles(),kind=BCType.constant,representation=geometry),0,system.state)
    ctx.schemeConfig.rigidBodies=[body];ctx.schemeConfig.boundaryProvider=provider
    p=torch.arange(len(system.state.UIDs)-1,-1,-1)
    ordered=system_view(system,p)
    cfg,sc=config_view(ctx.config,ctx.schemeConfig,ordered.state)
    assert sc.boundaryProvider is not provider
    assert sc.boundaryProvider.rigidBodies[0] is sc.rigidBodies[0]
    old=provider.signed_distance(system.state.positions)
    new=sc.boundaryProvider.signed_distance(ordered.state.positions)
    for a,b in zip(old,new):torch.testing.assert_close(b,a[p],rtol=0,atol=0)
    body=sc.rigidBodies[0]
    body.centerOfMass.add_(.03);body.orientation.add_(.2)
    bindBodies(sc.boundaryProvider,sc.rigidBodies)
    ordered.t=.1
    exportSimulationSystem(str(tmp_path),'analytic',ctx.scheme,ordered,config=cfg,
        schemeConfig=sc,integrator=ctx.integrator,next_dt=.001)
    template=system.initializeNewState()
    with h5py.File(tmp_path/'trajectory/analytic.h5') as file:
        restoreCheckpoint(file,template,ctx.config,ctx.scheme)
        readNativeMemory(file,template,config=ctx.config,schemeConfig=ctx.schemeConfig,integrator=ctx.integrator)
    restored=ctx.schemeConfig.rigidBodies[0]
    torch.testing.assert_close(restored.centerOfMass,body.centerOfMass,rtol=0,atol=0)
    torch.testing.assert_close(restored.orientation,body.orientation,rtol=0,atol=0)


def test_canonical_body_header_and_ghost_offsets_keep_declared_axes(tmp_path):
    from types import SimpleNamespace
    from test_particleBoundaryInputs import ghost_state
    from warpSPH.evaluation import AllParticles
    from warpSPH.rigidBody.build import buildRigidBody
    from warpSPH.configurations.region import BCType,RegionType
    from warpSPH.systems.weaklyCompressible import WeaklyCompressibleSystem
    from warpSPH.io.export import writeInitialData
    state=ghost_state()
    body=buildRigidBody(state,[SimpleNamespace(type=RegionType.Boundary,sdf=AllParticles(),kind=BCType.constant)],0)
    (tmp_path/'config.json').write_text('{}')
    with h5py.File(tmp_path/'body.h5','w') as file:
        writeInitialData(str(tmp_path),file,'deltaSPH',SimpleNamespace(dt=.01),
            SimpleNamespace(rigidBodies=[body]),SimpleNamespace(exportInterval=.01),
            WeaklyCompressibleSystem(state,canonicalExport=True))
        ids=file['combinedUIDs'][:]
        rows=file['rigidBodies/rigidBody_00/particleIndices'][:]
        assert ids[rows].tolist()==body.particleUIDs.tolist()
        ghosts=file['rigidBodies/rigidBody_00/ghostParticleIndices'][:]
        assert ids[ghosts].tolist()==body.ghostParticleUIDs.tolist()
        assert file['boundaryOffsets'].dtype.kind=='f'
        assert file['boundaryOffsets'][:].max()>0


def test_native_absolute_position_bdf_restart(tmp_path):
    case,opts=fixture('artificialCompressible',particleOrder='morton',store=True,
        storeInterval=1,exportRoot=str(tmp_path))
    configure=case.configureScheme
    def classic(ctx):
        configure(ctx)
        ctx.schemeConfig.acParams.incrementalPositions=False
    case=replace(case,configureScheme=classic)
    full=run(case,**opts)
    checkpoint=Path(full.exportPath,'trajectory','state_0001.h5')
    resumed=run(case,**dict(opts,store=False,nSteps=1,resumeFrom=str(checkpoint),resumeStepOffset=2))
    assert resumed.state.positionLow is None and resumed.state.positionStep is None
    close(full.state,resumed.state)


def test_batched_layout_flags_reuse_readback_but_detect_later_mutation(monkeypatch):
    case,opts=fixture('deltaSPH',nSteps=1,particleOrder='morton')
    result=run(case,**opts)
    manager=result.ctx.scratch['nativeLayout']
    flags=manager.flags
    def forbidden(*args):raise AssertionError('retained layout performed another device read')
    monkeypatch.setattr(manager,'flags',forbidden)
    _,_,_,changed=manager.prepare(result.state,result.ctx.config,result.ctx.schemeConfig)
    assert not changed
    stale=flags(result.state,result.ctx.config)
    manager.remember(result.state,result.ctx.config,bool(stale))
    result.state.state.positions.add_(.71)
    monkeypatch.setattr(manager,'flags',flags)
    _,_,_,changed=manager.prepare(result.state,result.ctx.config,result.ctx.schemeConfig)
    assert changed and manager.generation==2


@pytest.mark.parametrize('replacement',[False,True])
def test_native_uid_constant_guard_checks_hook_writes(replacement):
    case,opts=fixture('deltaSPH',nSteps=1,particleOrder='morton')
    def corrupt(ctx,system,i):
        if replacement:system.state.UIDs=system.state.UIDs.flip(0)
        else:system.state.UIDs.add_(1)
    with pytest.raises(ValueError,match='particle order changed'):
        run(replace(case,postStep=corrupt),**opts)
