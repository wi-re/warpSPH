"""The matrix must not evaluate a pressure map at a roundoff residual dt."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


path = Path(__file__).resolve().parents[1] / 'scripts/run_fluidMatrix.py'
spec = importlib.util.spec_from_file_location('fluid_matrix_harness', path)
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)


@pytest.mark.parametrize('time,end', [
    (1.9999999999998905, 2.0),
    (2.999999999999697, 3.0),
    (6.999999999997025, 7.0),
    (24.999999999968413, 25.0),
])
def test_roundoff_endpoint_does_not_request_another_physics_step(time, end):
    state = SimpleNamespace(t=time)
    assert harness.snapEndpoint(state, end)
    assert state.t == end


@pytest.mark.parametrize('time,end', [(1.999, 2.0), (24.9998, 25.0), (5.00001, 5.0)])
def test_ordinary_remaining_time_is_preserved(time, end):
    state = SimpleNamespace(t=time)
    assert not harness.snapEndpoint(state, end)
    assert state.t == time


def test_new_matrix_uses_calibrated_wide_support_and_adaptive_projection_limits():
    plan = harness.make_plan()
    assert harness.DEFAULT_ROOT.parts[-2] == '.tmp'
    for row in plan['rows']:
        assert row['spec']['n_h'] == 4.0
        if row['scheme'] in harness.PROJECTION:
            assert row['spec']['calibrateNormalization']
            assert row['spec'].get('adaptiveDt', True)


def test_projection_rerun_enables_band_viscosity_only_for_periodic_controls():
    plan = harness.make_plan(harness.PROJECTION)
    assert len(plan['rows']) == 75 and plan['version'] == 5
    for row in plan['rows']:
        if row['scheme'] == 'band2018pb':
            assert bool(row['scheme_settings'].get('bandViscosity')) == (row['boundary'] == 'periodic')
        if row['test'] == 'englishWedge':
            assert row['params']['unifyAttachedObstacles']


def test_source_archive_does_not_resolve_user_or_group_names(tmp_path, monkeypatch):
    import io
    import tarfile
    def forbidden(*args):
        raise AssertionError('Source archive attempted a network-backed account lookup')
    monkeypatch.setattr(tarfile.pwd, 'getpwuid', forbidden)
    monkeypatch.setattr(tarfile.grp, 'getgrgid', forbidden)
    source = tmp_path / 'source.py'
    source.write_bytes(b'physical_source = 1\n')
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w') as archive:
        harness.addSourceFile(archive, source, Path('src/source.py'))
    buffer.seek(0)
    with tarfile.open(fileobj=buffer, mode='r') as archive:
        member = archive.getmember('src/source.py')
        assert member.uid == source.stat().st_uid and member.gid == source.stat().st_gid
        assert archive.extractfile(member).read() == source.read_bytes()


def test_prepared_profile_preserves_volume_and_declares_periodic_source_variant():
    import sys
    sys.path.insert(0,str(path.parent))
    from prepared_fluidMatrix import make_prepared_plan, METHODS
    plan=make_prepared_plan(harness.make_plan)
    assert plan['version']==9 and plan['revision']=='sampling-recovery-v1' and plan['scale']=='full'
    assert len(plan['rows'])==15*len(METHODS)
    for row in plan['rows']:
        assert row['spec']['kernel']==('Wendland2' if row['test']=='taylorCouette' else 'Wendland4')
        assert not row['spec']['calibrateNormalization']
        assert row['spec']['device']=='cuda'
        if row['solver']=='dfsph':assert row['df_sampling_gate']['minDivergenceDensityRatio']==.9
        else:assert 'df_sampling_gate' not in row
        if row['test']=='tgv':assert row['spec']['nx']**2>=32768
        if row['boundary']=='periodic':
            assert row['initialization']=='periodic-jitter-damped'
            if row['solver'] in ('dfsph','iisph'):
                assert row['density_source_policy']=='periodicMeanFree'
        if row['test']=='boundedVortices':
            assert row['initialization']=='closed-jitter-damped'
            if row['scheme']!='band2018pb':assert row['skip'] is None
    screen=make_prepared_plan(harness.make_plan,scale='screen')
    assert all(r['spec']['nx']<=64 for r in screen['rows'])
    assert [r['spec']['tLimit'] for r in screen['rows']]==[r['spec']['tLimit'] for r in plan['rows']]


@pytest.mark.parametrize('revision',['original','stability-v1','consistency-v1'])
def test_historical_prepared_revisions_explicitly_preserve_ungated_df(revision):
    import sys
    sys.path.insert(0,str(path.parent))
    from prepared_fluidMatrix import make_prepared_plan
    plan=make_prepared_plan(harness.make_plan,revision=revision)
    assert all(r['df_sampling_gate']['minDivergenceDensityRatio']==0.
               for r in plan['rows'] if r['solver']=='dfsph')


@pytest.mark.parametrize('method',['dfsph','dfsphPredictor','dfsphShift','iisph'])
def test_canonical_preparation_recovery_is_independent_of_evolution_method(tmp_path,monkeypatch,method):
    import sys
    sys.path.insert(0,str(path.parent))
    from prepared_fluidMatrix import make_prepared_plan,prepared_case,relax_fixture
    from warpSPH.runner import prepareContext,CaseSpec
    from warpSPH.cases.periodic.randomFlowIncompressible import randomFlowIncompressibleCase
    from warpSPH.modules.dfsph import composition
    row=next(r for r in make_prepared_plan(harness.make_plan,methods=(method,),scale='screen')['rows']
             if r['test']=='boundedVortices' and r['boundary']=='particles')
    # Even a deliberately ungated evolution control gets the common fixture.
    row['df_sampling_gate']={'minDivergenceDensityRatio':0.}
    row['spec'].update(nx=12,device='cpu')
    case=prepared_case(randomFlowIncompressibleCase,tmp_path,row)
    ctx=prepareContext(case,CaseSpec(scheme=row['solver'],**row['spec'],params=row['params']))
    assert ctx.schemeConfig.dfsph.minDivergenceDensityRatio==0.
    system=case.buildSystem(ctx)
    class ReachedPreparation(Exception):pass
    def inspect(system,dt,cfg,sc,order):
        assert sc.dfsph.minDivergenceDensityRatio==.9 and order=='df_np_cd'
        raise ReachedPreparation
    monkeypatch.setattr(composition,'advanceDFSPH',inspect)
    with pytest.raises(ReachedPreparation):relax_fixture(ctx,system,tmp_path,False)


def test_corrected_presets_preserve_unaffected_rows_and_old_manifest_replay():
    import sys
    sys.path.insert(0,str(path.parent))
    from prepared_fluidMatrix import make_prepared_plan
    original=make_prepared_plan(harness.make_plan,revision='original')
    corrected=make_prepared_plan(harness.make_plan,revision='stability-v1')
    assert original['version']==6 and all('numerical_preset' not in r for r in original['rows'])
    affected=[]
    for old,new in zip(original['rows'],corrected['rows']):
        if 'numerical_preset' not in new:
            assert new==old
            continue
        if not new['skip']:affected.append(new['id'])
        assert new['initialization']==old['initialization'] and new['params']==old['params']
        preset=new['numerical_preset']
        if new['test']=='shearDecay':
            assert preset['viscosityTerm']=='morris1997'
            assert .95<preset['viscosityCalibration']['factor']<1.01
            assert preset['targetKinematicViscosity']==.0185
        elif new['test']=='sloshing':assert new['spec']['dt']==new['spec']['maxDt']==.0002
        else:assert new['id']=='tgv__dfsphShift__periodic' and preset['divergenceTolerancePercent']==.001
    assert len(affected)==24


def test_prepared_jitter_is_bounded_and_velocity_field_is_wall_compatible():
    import sys
    import torch
    sys.path.insert(0,str(path.parent))
    from prepared_fluidMatrix import bounded_jitter,sample_field
    x=torch.rand(128,2)
    j=bounded_jitter(x,.1,1948)
    assert torch.equal(j,bounded_jitter(x,.1,1948))
    assert float((j-x).norm(dim=1).max())<=.0010001
    walls=torch.tensor([[0.,.2],[1.,.6],[.3,0.],[.7,1.]])
    assert sample_field(walls,'boundedVortices',1.).abs().max()<1e-12
    # The analytic streamfunction curl must have zero continuum divergence.
    q=torch.rand(50,2,dtype=torch.float64,requires_grad=True)
    v=sample_field(q,'boundedVortices',1.)
    dx=torch.autograd.grad(v[:,0].sum(),q,create_graph=True)[0][:,0]
    dy=torch.autograd.grad(v[:,1].sum(),q)[0][:,1]
    assert (dx+dy).abs().max()<1e-12


@pytest.mark.parametrize('method',['dfsph','band2018pb'])
def test_prepared_sampled_box_keeps_physical_volume_and_auxiliary_ghosts(tmp_path,method):
    import sys
    import torch
    sys.path.insert(0,str(path.parent))
    from prepared_fluidMatrix import make_prepared_plan,prepared_case
    from warpSPH.runner import prepareContext,CaseSpec
    from warpSPH.cases.periodic.randomFlowIncompressible import randomFlowIncompressibleCase
    plan=make_prepared_plan(harness.make_plan,methods=(method,),scale='screen')
    row=next(r for r in plan['rows'] if r['test']=='boundedVortices' and r['boundary']=='particles')
    row['spec'].update(nx=12,device='cpu')
    case=prepared_case(randomFlowIncompressibleCase,tmp_path,row)
    ctx=prepareContext(case,CaseSpec(scheme=row['solver'],**row['spec'],params=row['params']))
    system=case.buildSystem(ctx)
    st=system.state
    fluid=st.kinds==0
    assert int(fluid.sum())==144
    assert float(st.masses[fluid].double().sum())==pytest.approx(4.,rel=1e-7)
    assert int((st.kinds==2).sum())==int((st.kinds==1).sum())>0
    if method=='band2018pb':assert ctx.schemeConfig.boundaryProvider is None
    else:assert ctx.schemeConfig.iisph.boundaryRepresentation=='particles'


@pytest.mark.parametrize('boundary',['particles','analytic'])
@pytest.mark.parametrize('gate',[None,0.,.9])
def test_prepared_sampling_gate_reaches_native_config_for_both_walls(tmp_path,boundary,gate):
    import sys
    sys.path.insert(0,str(path.parent))
    from prepared_fluidMatrix import make_prepared_plan,prepared_case
    from warpSPH.runner import prepareContext,CaseSpec
    from warpSPH.cases.freeSurface.sloshingTank import sloshingTankCase
    plan=make_prepared_plan(harness.make_plan,methods=('dfsph',),scale='screen')
    row=next(r for r in plan['rows'] if r['test']=='sloshing' and r['boundary']==boundary)
    row['spec'].update(nx=40,device='cpu')
    row.pop('df_sampling_gate',None)  # A historical manifest without the field.
    if gate is not None:row['df_sampling_gate']={'minDivergenceDensityRatio':gate}
    case=prepared_case(sloshingTankCase,tmp_path,row)
    ctx=prepareContext(case,CaseSpec(caseName=case.name,scheme=row['solver'],params={**case.params,**row['params']}).merged(**case.defaults).merged(**row['spec']))
    assert ctx.schemeConfig.dfsph.minDivergenceDensityRatio==(0. if gate is None else gate)
    assert ctx.schemeConfig.iisph.boundaryRepresentation==boundary
    assert ctx.schemeConfig.dfsph.density.maxErrorPercent==.01
    assert ctx.schemeConfig.dfsph.divergence.maxErrorPercent==.1


@pytest.mark.parametrize('method',['divergenceFree','omniIncompressible','dfsphReference'])
@pytest.mark.parametrize('density_applied',[True,False])
def test_legacy_pressure_observer_is_read_only_and_forwards_exact_call(monkeypatch,method,density_applied):
    import sys
    import torch
    sys.path.insert(0,str(path.parent))
    from prepared_fluidMatrix import observe_legacy_pressures
    from warpSPH.modules.incompressible import loads
    rho=torch.tensor([1.,2.]);div=torch.tensor([3.,4.]);original_rho=rho.clone();original_div=div.clone()
    args=(object(),object(),SimpleNamespace(densitySolve=True),object())
    kwargs=dict(pDiv=div,pRho=rho,densityApplied=density_applied)
    token=object();calls=[]
    def original(*a,**k):
        calls.append((a,k));return token
    monkeypatch.setattr(loads,'bookProjectionWallLoads',original)
    observed,restore=observe_legacy_pressures(method)
    try:
        assert loads.bookProjectionWallLoads(*args,**kwargs) is token
        assert all(a is b for a,b in zip(calls[0][0],args))
        assert calls[0][1]['pDiv'] is div and calls[0][1]['pRho'] is rho
        assert observed['divergence'] is div and observed['density'] is rho
        assert observed['densityApplied']==density_applied
        assert torch.equal(rho,original_rho) and torch.equal(div,original_div)
    finally:restore()
    assert loads.bookProjectionWallLoads is original


@pytest.mark.parametrize('mode,key',[('density','density'),('divergence','divergence')])
def test_reference_pressure_observation_without_wall_bookkeeping(monkeypatch,mode,key):
    import sys
    import torch
    import importlib
    sys.path.insert(0,str(path.parent))
    from prepared_fluidMatrix import observe_legacy_pressures
    module=importlib.import_module('warpSPH.schemes.incompressible.dfsphReference')
    pressure=torch.tensor([2.,3.]);before=pressure.clone();calls=[]
    result=(object(),pressure,object(),object());args=(object(),object())
    kwargs=dict(mode=mode,opDt=.000173)
    def solve(*a,**k):
        calls.append((a,k));return result
    monkeypatch.setattr(module,'_jacobiSolve',solve)
    observed,restore=observe_legacy_pressures('dfsphReference')
    try:
        assert module._jacobiSolve(*args,**kwargs) is result
        assert all(a is b for a,b in zip(calls[0][0],args))
        assert calls[0][1]==kwargs
        assert observed[key] is pressure and observed['densityApplied']
        assert torch.equal(pressure,before)
    finally:restore()
    assert module._jacobiSolve is solve


@pytest.mark.parametrize('ordering',['df_np_cd','np_df_cd','np_df_shift'])
@pytest.mark.parametrize('dt',[.001,.000173])
def test_shared_hydrostatic_seed_uses_actual_stage_and_timestep_units(ordering,dt):
    import sys
    import torch
    sys.path.insert(0,str(path.parent))
    from prepared_fluidMatrix import seed_projection_pressure
    pressure=torch.tensor([3.,2.,1.,0.])
    state=SimpleNamespace(pressures=pressure,positions=torch.randn(4,2),velocities=torch.zeros(4,2),masses=torch.ones(4))
    before={k:getattr(state,k).clone() for k in ('pressures','positions','velocities','masses')}
    seed_projection_pressure(state,ordering,dt)
    if ordering=='np_df_shift':
        torch.testing.assert_close(state.divergencePressureImpulse/dt,pressure)
        assert not state.densityPressureImpulse.any()
    else:
        torch.testing.assert_close(state.densityPressureImpulse/dt**2,pressure)
        assert not state.divergencePressureImpulse.any()
    for key,value in before.items():assert torch.equal(getattr(state,key),value)


def test_prepared_overlay_retains_failures_and_actual_sources(tmp_path):
    import sys,json
    sys.path.insert(0,str(path.parent))
    from compare_preparedFluidMatrix import overlay_prepared
    base,rerun,out=(tmp_path/name for name in ('base','rerun','out'))
    for root in (base,rerun):root.mkdir()
    ids=['shearDecay__iisph__periodic','tgv__iisph__periodic']
    rows=[dict(id=i,test=i.split('__')[0],scheme='iisph',boundary='periodic',
               status='completed',target_time=2.,actual_time=2.,data_root=str(tmp_path/'raw-original'),run_date='2026-10-25') for i in ids]
    plan=dict(profile='prepared-incompressible',scale='full',schemes=['iisph'],created='2026-10-25',version=6,
              rows=[dict(id=r['id'],test=r['test'],scheme='iisph',boundary='periodic',spec=dict(tLimit=2.)) for r in rows])
    (base/'manifest.json').write_text(json.dumps(plan));(base/'comparison_metrics.json').write_text(json.dumps(rows))
    changed={**rows[0], 'status':'failed','actual_time':.5,'data_root':str(rerun),'run_date':'2026-10-26'}
    updated={**plan,'rows':plan['rows'][:1],'version':7,'created':'2026-10-26'}
    (rerun/'manifest.json').write_text(json.dumps(updated));(rerun/'comparison_metrics.json').write_text(json.dumps([changed]))
    manifest,metrics=overlay_prepared(base,rerun,out)
    assert metrics[0]['status']=='failed' and metrics[0]['actual_time']==.5
    assert metrics[0]['data_root']==str(rerun)
    assert metrics[1]['data_root']==str(tmp_path/'raw-original') and metrics[1]['run_date']=='2026-10-25'
    assert json.loads((base/'comparison_metrics.json').read_text())==rows
    assert manifest['rows'][1]['data_root']==str(tmp_path/'raw-original')
    # A shorter successful sample cannot silently replace a full-endpoint failure.
    changed['target_time']=.5
    (rerun/'comparison_metrics.json').write_text(json.dumps([changed]))
    with pytest.raises(ValueError,match='endpoint'):
        overlay_prepared(base,rerun,tmp_path/'bad-overlay')
