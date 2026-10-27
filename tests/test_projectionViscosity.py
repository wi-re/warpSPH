"""Case viscosity and free-slip references must survive the projection adapter."""
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_fluidMatrix import make_plan
from prepared_fluidMatrix import make_prepared_plan,prepared_case


def prepared_context(tmp_path,test='marrone31',revision='consistency-v1'):
    from warpSPH.runner import prepareContext,CaseSpec
    from warpSPH.cases.freeSurface.dambreak import dambreakCase
    from warpSPH.cases.freeSurface.sloshingTank import sloshingTankCase
    row=next(r for r in make_prepared_plan(make_plan,methods=('dfsph',),revision=revision)['rows']
             if r['test']==test and r['boundary']=='particles')
    row['spec'].update(nx=24,device='cpu')
    case=prepared_case(sloshingTankCase if test=='sloshing' else dambreakCase,tmp_path,row)
    spec=CaseSpec(scheme='dfsph',params={**case.params,**row['params']}).merged(**case.defaults).merged(**row['spec'])
    ctx=prepareContext(case,spec)
    return row,case,ctx


@pytest.mark.parametrize('test',['marrone31','sloshing'])
def test_prepared_free_surface_keeps_case_viscosity(tmp_path,test):
    row,case,ctx=prepared_context(tmp_path,test)
    dp=ctx.schemeConfig.diffusionParams
    assert row['numerical_preset']['viscosityPolicy']=='case'
    if test=='marrone31':
        assert dp.inviscid and dp.inviscidAlpha==pytest.approx(.01)
        assert ctx.schemeConfig.fluid.fixedSoundSpeed==pytest.approx(
            row['params']['referenceVelocity']/row['params']['machTarget'])
    else:
        assert not dp.inviscid and dp.viscidNu==pytest.approx(1e-6)
    _,_,old=prepared_context(tmp_path/'old',test,revision='stability-v1')
    assert not old.schemeConfig.diffusionParams.inviscid
    assert old.schemeConfig.diffusionParams.viscidNu==pytest.approx(.01)


def test_native_ghost_viscosity_preserves_uniform_tangential_flow(tmp_path):
    from warpSPHCore import buildVerletList
    from warpSPH.modules.deltaSPH import computeVelocityDiffusion
    from warpSPH.modules.mdbc.viscosityVelocity import computeViscosityBoundaryVelocities
    from warpSPH.configurations.region import BCType
    row,case,ctx=prepared_context(tmp_path)
    system=case.buildSystem(ctx);st=system.state;cfg=ctx.config;sc=ctx.schemeConfig
    assert (st.kinds==2).any()  # Regression requires stored mDBC ghosts.
    assert sc.rigidBodies[0].kind==BCType.freeSlip
    fluid=st.kinds==0;st.velocities.zero_();st.velocities[fluid,0]=1.;st.densities.fill_(1.)
    original=st.velocities.clone();adj=buildVerletList(st,cfg.domain)
    reference=computeViscosityBoundaryVelocities(st,cfg,sc,adjacency=adj,boundaryVelocitiesPrepared=False)
    # Flat bed far from vertical walls: reflection must retain tangential speed.
    x=st.positions;support=st.supports.max();left=x[fluid,0].min();right=x[fluid,0].max()
    bed=(st.kinds==1)&(x[:,1]<-.5)&(x[:,1]>-.5-support*.5)&(x[:,0]>left+support)&(x[:,0]<right-support)
    assert bed.any()
    torch.testing.assert_close(reference[bed,0],torch.ones_like(reference[bed,0]),rtol=1e-5,atol=1e-5)
    sc.diffusionParams.inviscid=False;sc.diffusionParams.viscidNu=.01
    old=computeVelocityDiffusion(st,cfg,sc,adj,boundaryRepresentation='particles')
    fixed=computeVelocityDiffusion(st,cfg,sc,adj,boundaryRepresentation='particles',boundaryVelocitiesPrepared=False)
    probe=fluid&(x[:,1]<-.5+support*.5)&(x[:,0]>left+support)&(x[:,0]<right-support)
    assert probe.any() and old[probe,0].mean()<-.001
    assert fixed[probe].abs().max()<1e-4
    assert torch.equal(st.velocities,original)  # Pressure/rigid-body state untouched.
    assert computeViscosityBoundaryVelocities(st,cfg,sc,adjacency=adj,boundaryVelocitiesPrepared=True) is st.velocities
