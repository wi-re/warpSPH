"""Small numerical/identity fixtures, also run explicitly on CUDA."""
from copy import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from warpSPH.evaluation import (MortonStepper,UIDValues,AllParticles,PlaneRegion,
    adopt_spatial_layout,permute_boundary_state,reorder_boundary_system)
from warpSPH.evaluation.boundaryLayout import boundary_bodies
from warpSPH.evaluation.layout import morton_order,permute_update
from warpSPH.configurations.moduleConfigurations.boundaryConditions import BoundaryCondition,BoundaryConditionType
from warpSPHIntegrators import get_scheme
from test_spatialContinuation import fixture,assert_physics

DEVICES=['cpu']+(['cuda'] if torch.cuda.is_available() else [])


@pytest.mark.parametrize('fluid',['CompSPH','CRKSPH'])
@pytest.mark.parametrize('geometry',[AllParticles(),PlaneRegion(torch.tensor([1.]),.2)])
@pytest.mark.parametrize('device',DEVICES)
def test_uid_force_and_update_inputs_match_native_after_adoption_and_resort(fluid,geometry,device):
    ctx,state=fixture(fluid,fsal=True,device=device)
    ids=state.state.UIDs.clone()
    force=UIDValues(ids,.2*torch.sin(state.state.positions)*state.state.masses[:,None])
    bc=BoundaryCondition(BoundaryConditionType.forcing,geometry,forcingFunctions=[force],
        updateFunctions={'dxdt':UIDValues(ids,.01*torch.cos(state.state.positions))})
    ctx.schemeConfig.boundaryConditions=[bc]
    control=state.initializeNewState();stepper=MortonStepper(ctx.integrator)
    for _ in range(3):
        native=ctx.integrator(control,1e-4,ctx.stepFunction,config=ctx.config,schemeConfig=ctx.schemeConfig)
        result=stepper(state,1e-4,config=ctx.config,schemeConfig=ctx.schemeConfig)
        assert_physics(result.state,native.state)
        state=adopt_spatial_layout(result).state;control=native.state
        state=copy(state);state.spatialContext=replace(state.spatialContext,adjacency=None)
    assert torch.equal(force.ids,ids)


def test_nonperiodic_morton_clamps_endpoints_instead_of_wrapping():
    from warpSPH.utils.domain import buildDomainDescription
    from warpSPHCore import get_torch_precision
    domain=buildDomainDescription(l=2,dim=1,periodic=False,dtype=get_torch_precision())
    positions=torch.tensor([[1.],[-1.],[3.],[-3.],[0.]],dtype=domain.min.dtype)
    assert morton_order(positions,domain,bits=3).tolist()==[1,3,4,0,2]


@pytest.mark.parametrize('fluid',['CompSPH','CRKSPH'])
@pytest.mark.parametrize('device',DEVICES)
def test_solid_wall_volumes_remain_with_particles_through_rebuild_and_restart(fluid,tmp_path,device):
    from warpSPH.io.export import exportSimulationSystem
    from warpSPH.io.particleLayout import restoreCheckpoint
    from warpSPH.io.continuation import readContinuation
    from warpSPHIntegrators.runtime import Continuation
    import h5py
    ctx,state=fixture(fluid,fsal=True,device=device)
    ctx.config.domain.periodic[:]=False
    x=state.state.positions[:,0]
    wall=(x < -.3)|(x > .3)
    state.state.kinds[wall]=1
    state.state.velocities[wall]=0
    state.state.wallNormals=torch.sign(x)[:,None]
    state.state.UIDcounter=int(state.state.UIDs.max())+1
    # Nonuniform volumes make a stale same-length config cache detectable.
    state.state.masses*=1+.05*torch.sin(3*x)
    template=state.initializeNewState();control=state.initializeNewState()
    stepper=MortonStepper(ctx.integrator)
    for _ in range(3):
        native=ctx.integrator(control,1e-4,ctx.stepFunction,config=ctx.config,schemeConfig=ctx.schemeConfig)
        result=stepper(state,1e-4,config=ctx.config,schemeConfig=ctx.schemeConfig)
        assert_physics(result.state,native.state)
        state=adopt_spatial_layout(result).state;control=native.state
        state=copy(state);state.spatialContext=replace(state.spatialContext,adjacency=None)
    exportSimulationSystem(str(tmp_path),'walls','CompSPH' if fluid=='CompSPH' else 'CRKSPH',state,
        continuation=Continuation(state,None),integrator=ctx.integrator,config=ctx.config,schemeConfig=ctx.schemeConfig,
        next_dt=1e-4,last_dt=1e-4)
    with h5py.File(tmp_path/'trajectory/walls.h5') as f:
        restoreCheckpoint(f,template,ctx.config,ctx.scheme)
        memory,_,_=readContinuation(f['integration'],template,integrator=ctx.integrator,config=ctx.config,schemeConfig=ctx.schemeConfig)
    a=stepper(state,1e-4,config=ctx.config,schemeConfig=ctx.schemeConfig)
    b=stepper(memory.state,1e-4,config=ctx.config,schemeConfig=ctx.schemeConfig)
    assert_physics(a.state,b.state)


@pytest.mark.parametrize('fluid',['CompSPH','CRKSPH'])
@pytest.mark.parametrize('device',DEVICES)
def test_native_wall_cache_has_a_uid_owner_before_entering_a_different_layout(fluid,device):
    from warpSPH.evaluation.layout import permute_state
    ctx,state=fixture(fluid,fsal=True,device=device)
    ctx.config.domain.periodic[:]=False
    x=state.state.positions[:,0]
    state.state.kinds[(x < -.3)|(x > .3)]=1
    state.state.masses*=1+.07*torch.sin(7*x)
    state.state.velocities[state.state.kinds == 1]=0
    first=ctx.integrator(state.initializeNewState(),1e-4,ctx.stepFunction,config=ctx.config,schemeConfig=ctx.schemeConfig)
    owner=ctx.config._wallVolumeUIDs.clone();volumes=ctx.config._wallVolumes.clone()
    p=torch.arange(31,-1,-1,device=device)
    reordered=copy(first.state);reordered.state=permute_state(first.state.state,p);reordered.adjacency=None
    native=ctx.integrator(first.state.initializeNewState(),1e-4,ctx.stepFunction,config=ctx.config,schemeConfig=ctx.schemeConfig)
    result=MortonStepper(ctx.integrator)(reordered,1e-4,config=ctx.config,schemeConfig=ctx.schemeConfig)
    assert_physics(result.state,native.state)
    from warpSPH.evaluation.continuation import uid_order
    torch.testing.assert_close(result.state.state.wallVolumes,volumes[uid_order(owner,result.state.state.UIDs)],rtol=0,atol=0)
    assert torch.equal(ctx.config._wallVolumeUIDs,owner) and torch.equal(ctx.config._wallVolumes,volumes)
    legacy=copy(ctx.config);del legacy._wallVolumeUIDs
    with pytest.raises(ValueError,match='UID owner'):
        MortonStepper(ctx.integrator)(reordered,1e-4,config=legacy,schemeConfig=ctx.schemeConfig)


def ghost_state(device='cpu'):
    from warpSPH.systems.weaklyCompressible import WeaklyCompressibleState
    from warpSPHCore import get_torch_precision
    x=torch.tensor([[.2,.3],[.8,.3],[.3,.8],[.2,.5],[.3,.6]],device=device,dtype=get_torch_precision())
    return WeaklyCompressibleState(positions=x,velocities=x*.1,masses=x.new_ones(5),
        supports=x.new_full((5,),.3),densities=x.new_ones(5),
        kinds=torch.tensor([1,0,1,2,2],device=device,dtype=torch.int32),
        materials=torch.zeros(5,device=device,dtype=torch.int32),bodyIDs=torch.tensor([0,-1,0,0,0],device=device),
        UIDs=torch.tensor([0,2**35,19,23,29],device=device),UIDcounter=2**35+1,
        ghostIndices=torch.tensor([3,-1,4,0,2],device=device),
        ghostOffsets=torch.tensor([[0,-.2],[0,0],[0,.2],[0,.2],[0,-.2]],device=device,dtype=x.dtype))


@pytest.mark.parametrize('device',DEVICES)
def test_ghost_pairing_body_scatter_and_kind_cache_follow_uid_not_row(device):
    from warpSPH.rigidBody.build import buildRigidBody
    from warpSPH.rigidBody.update import updateBodyParticlesWCSPH
    from warpSPH.configurations.region import RegionType,BCType
    from warpSPH.systems.weaklyCompressible import WeaklyCompressibleSystem
    from warpSPH.utils.syncFree import cachedKindIndex
    state=ghost_state(device)
    body=buildRigidBody(state,[SimpleNamespace(type=RegionType.Boundary,sdf=AllParticles(),kind=BCType.constant)],0)
    body.orientation=state.positions.new_tensor(.7)
    body.angularVelocity=state.positions.new_tensor(1.3)
    body.linearVelocity=state.positions.new_tensor([.1,-.2])
    p=torch.tensor([4,2,1,3,0],device=device)
    config=SimpleNamespace();scheme=SimpleNamespace(rigidBodies=[body])
    old_index=cachedKindIndex(state.kinds,1,config).clone()
    new,cfg,sc=reorder_boundary_system(WeaklyCompressibleSystem(state),p,config=config,schemeConfig=scheme)
    assert cachedKindIndex(new.state.kinds,1,cfg).tolist()==[1,4]
    assert torch.equal(cachedKindIndex(state.kinds,1,config),old_index)
    valid=new.state.ghostIndices>=0
    assert torch.equal(new.state.UIDs[new.state.ghostIndices[valid]],state.UIDs[state.ghostIndices[p][valid]])
    a=updateBodyParticlesWCSPH(state,body)
    b=updateBodyParticlesWCSPH(new.state,sc.rigidBodies[0])
    for name in ('positions','velocities','ghostOffsets','boundaryAccelerations'):
        torch.testing.assert_close(getattr(b,name),getattr(a,name)[p],rtol=0,atol=0)
    assert body.particleIndices.dtype==torch.bool and sc.rigidBodies[0].particleIndices.dtype==torch.int64
    assert new.adjacency is None and new.deltaStages is None
    assert torch.equal(new.state.ghostOffsets[new.state.ghostIndices[valid]],-new.state.ghostOffsets[valid])


def test_fresh_ghost_identity_avoids_uid_zero_collision_without_changing_legacy_default(monkeypatch):
    import warpSPH.rigidBody.ghostParticles as module
    from warpSPH.configurations.region import RegionType
    state=ghost_state();state=replace(state,**{name:getattr(state,name)[:3] if isinstance(getattr(state,name),torch.Tensor) else getattr(state,name)
        for name in state.__dataclass_fields__})
    monkeypatch.setattr(module,'_gridSnapGhostOffsets',lambda p,*args:torch.zeros_like(p)+.1)
    region=SimpleNamespace(type=RegionType.Boundary,sdf=AllParticles(),representation=None)
    legacy=module.addBoundaryGhostParticles([region],state)
    result=module.addBoundaryGhostParticles([region],state,persistent_ids=True)
    assert legacy.UIDs.tolist()==[0,2**35,19,0,-19]
    assert len(result.UIDs.unique())==5 and result.UIDcounter==state.UIDcounter+2
    assert result.UIDs[-2:].tolist()==[state.UIDcounter,state.UIDcounter+1]
    torch.testing.assert_close(result.positions,legacy.positions,rtol=0,atol=0)


def test_global_source_work_heat_vector_is_not_reordered_when_n_equals_two():
    from warpSPH.systems.compressibleMonaghan import CompressibleSystemUpdate
    update=CompressibleSystemUpdate(dxdt=torch.tensor([[1.],[2.]]),dvdt=torch.tensor([[3.],[4.]]),dudt=torch.tensor([5.,6.]),dEdt=None,drhodt=None,sourceEnergyRates=torch.tensor([7.,8.]))
    result=permute_update(update,torch.tensor([1,0]))
    assert result.dudt.tolist()==[6.,5.] and result.sourceEnergyRates.tolist()==[7.,8.]


def test_unadapted_closure_is_rejected_before_sorting(monkeypatch):
    import warpSPH.evaluation.sortedStep as module
    ctx,state=fixture(fsal=True)
    ctx.schemeConfig.boundaryConditions=[BoundaryCondition(BoundaryConditionType.forcing,AllParticles(),forcingFunctions=[lambda *a:state.state.positions])]
    monkeypatch.setattr(module,'morton_order',lambda *a,**k:pytest.fail('unadapted input reached sorting'))
    with pytest.raises(ValueError,match='UID input adapter'):
        MortonStepper(ctx.integrator)(state,1e-4,config=ctx.config,schemeConfig=ctx.schemeConfig)


@pytest.mark.parametrize('fluid',['CompSPH','CRKSPH'])
@pytest.mark.parametrize('field',['velocities','internalEnergies'])
def test_uid_dirichlet_projection_matches_native(fluid,field):
    ctx,state=fixture(fluid,fsal=True)
    values=getattr(state.state,field).clone()*.99
    ctx.schemeConfig.boundaryConditions=[BoundaryCondition(BoundaryConditionType.dynamic,
        PlaneRegion(state.state.positions.new_tensor([1.]),.2),dirichletFunctions={field:UIDValues(state.state.UIDs,values)})]
    native=ctx.integrator(state.initializeNewState(),1e-4,ctx.stepFunction,config=ctx.config,schemeConfig=ctx.schemeConfig)
    actual=MortonStepper(ctx.integrator)(state,1e-4,config=ctx.config,schemeConfig=ctx.schemeConfig)
    assert_physics(actual.state,native.state)


def test_boundary_acceptance_preserves_surface_history_and_rejects_opaque_inputs():
    from warpSPH.modules.deltaSPH.stages import DeltaStages
    from warpSPH.systems.weaklyCompressible import WeaklyCompressibleSystem
    state=ghost_state();p=torch.tensor([4,2,1,3,0]);mask=state.kinds==0
    owner=DeltaStages(diffusion=(state.densities,state.velocities),surfaceStart=mask,previousSurfaceStart=~mask)
    system=WeaklyCompressibleSystem(state,deltaStages=owner)
    scheme=SimpleNamespace(rigidBodies=[],boundaryConditions=[])
    result,_,_=reorder_boundary_system(system,p,config=SimpleNamespace(),schemeConfig=scheme)
    assert result.deltaStages.diffusion is None
    assert torch.equal(result.deltaStages.surfaceStart,mask[p])
    assert torch.equal(result.deltaStages.previousSurfaceStart,(~mask)[p])
    assert torch.equal(owner.surfaceStart,mask)
    scheme.boundaryProvider=object()
    with pytest.raises(ValueError,match='analytic'):
        reorder_boundary_system(system,p,config=SimpleNamespace(),schemeConfig=scheme)


@pytest.mark.parametrize('family',['wc','isph','dfsph','ac'])
@pytest.mark.parametrize('device',DEVICES)
def test_every_boundary_state_schema_declares_all_rows(family,device):
    from dataclasses import fields
    from warpSPH.systems.weaklyCompressible import WeaklyCompressibleState
    from warpSPH.systems.incompressible import IncompressibleState,DFSPHState
    from warpSPH.systems.artificialCompressible import ArtificialCompressibleState
    cls={'wc':WeaklyCompressibleState,'isph':IncompressibleState,'dfsph':DFSPHState,'ac':ArtificialCompressibleState}[family]
    source=ghost_state(device)
    values={f.name:getattr(source,f.name,None) for f in fields(cls)}
    for f in fields(cls):
        if values[f.name] is None:
            values[f.name]=torch.arange(5,dtype=source.positions.dtype,device=device)
    state=cls(**values);p=torch.tensor([3,2,4,1,0],device=device)
    result=permute_boundary_state(state,p)
    for f in fields(cls):
        if f.name in ('UIDcounter','ghostIndices'):continue
        torch.testing.assert_close(getattr(result,f.name),getattr(state,f.name)[p],rtol=0,atol=0)
    assert result.UIDcounter==state.UIDcounter
