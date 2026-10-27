"""Sampling recovery gates pressure unknowns, while preserving pair forces."""
from types import SimpleNamespace as NS
import math
import pytest
import torch


def periodic_projection(ratio=1.):
    from warpSPHCore import DomainDescription, KernelFunctions, SupportScheme, buildVerletList
    from warpSPHCore.type_config import get_torch_precision
    from warpSPH.configurations import IISPHConfig, DFSPHConfig
    from warpSPH.modules.iisph import computeIISPHCoefficients
    dtype = get_torch_precision()
    dx = 2*math.pi/16
    line = (torch.arange(16, dtype=dtype)+.5)*dx
    x = torch.stack(torch.meshgrid(line, line, indexing='ij'), -1).reshape(-1,2).contiguous()
    n = len(x)
    state = NS(positions=x, velocities=-.1*torch.sin(x),
        supports=torch.full((n,), 4*dx, dtype=dtype),
        masses=torch.full((n,), dx*dx, dtype=dtype),
        densities=torch.full((n,), ratio, dtype=dtype),
        kinds=torch.zeros(n, dtype=torch.int32))
    config = NS(domain=DomainDescription(torch.zeros(2, dtype=dtype),
        torch.full((2,), 2*math.pi, dtype=dtype), torch.ones(2, dtype=torch.bool), 2),
        kernel=KernelFunctions.Wendland4, n_h=4., calibrateNormalization=False)
    scheme = NS(fluid=NS(restDensity=1.), rigidBodies=[], iisph=IISPHConfig(), dfsph=DFSPHConfig(minDivergenceDensityRatio=0.))
    adj = buildVerletList(state, config.domain, supportMode=SupportScheme.SuperSymmetric)
    return state,config,scheme,adj,computeIISPHCoefficients


def project(f, mode='divergence', signed=False):
    from warpSPH.modules.dfsph import solveDFSPHProjection
    st,cfg,sc,adj,coefficients = f
    policy = getattr(sc.dfsph, mode)
    policy.clampPressure = not signed
    policy.relaxationFactor = .25
    policy.minIterations = policy.maxIterations = 8
    g = coefficients(st,cfg,sc,adj)
    return solveDFSPHProjection(st,cfg,sc,.001,adj,*g[:5],mode=mode,
        velocity=st.velocities,pressureImpulse=torch.ones_like(st.densities))


def test_disabledRowsClearWarmHistoryAndDoNotRegeneratePressure():
    f = periodic_projection(.5)
    ungated = project(f)
    assert ungated.pressure.max() > 0
    f[2].dfsph.minDivergenceDensityRatio = .9
    gated = project(f)
    assert (gated.source < 0).any()  # Actual compression is allowed below threshold.
    assert torch.count_nonzero(gated.pressure) == 0
    assert torch.count_nonzero(gated.acceleration) == 0
    assert gated.error == 0


def test_gateIncludesThresholdAndRetainsSymmetricNeighbourForces():
    f = periodic_projection()
    f[0].densities[0] = .5
    f[0].densities[1] = .9
    f[2].dfsph.minDivergenceDensityRatio = .9
    result = project(f)
    assert result.pressure[0] == 0
    assert result.pressure[1] > 0  # Equality remains active.
    assert result.acceleration[0].norm() > 0  # No asymmetric force mask.
    momentum = (f[0].masses[:,None]*result.acceleration*.001).sum(0)/f[0].masses.sum()
    assert momentum.norm() < 2.e-6


@pytest.mark.parametrize('mode,signed,ratio', [
    ('density',False,.5), ('divergence',True,.5), ('divergence',False,1.)])
def test_densitySignedProjectionAndCompleteBulkAreBitwiseUnchanged(mode,signed,ratio):
    f = periodic_projection(ratio)
    original = project(f,mode,signed)
    f[2].dfsph.minDivergenceDensityRatio = .9
    gated = project(f,mode,signed)
    assert torch.equal(original.pressure,gated.pressure)
    assert torch.equal(original.acceleration,gated.acceleration)
    assert original.error == gated.error
    assert original.iterations == gated.iterations


@pytest.mark.parametrize('value',[-.1,1.01,float('nan'),float('inf')])
def test_densityGateRejectsInvalidConfiguration(value):
    from warpSPH.configurations import DFSPHConfig
    with pytest.raises(ValueError,match='minDivergenceDensityRatio'):
        DFSPHConfig(minDivergenceDensityRatio=value).validate()


def test_densityGateDefaultAndExplicitReferenceRoundTrip():
    from warpSPH.configurations.incompressible import IncompressibleSPHConfig,incompressibleConfigToDict,dictToIncompressibleSPHConfig
    cfg = IncompressibleSPHConfig()
    assert cfg.dfsph.minDivergenceDensityRatio == .9
    cfg.dfsph.minDivergenceDensityRatio = 0.
    restored = dictToIncompressibleSPHConfig(incompressibleConfigToDict(cfg))
    assert restored.dfsph.minDivergenceDensityRatio == 0.


def test_analyticWallCompletionKeepsDenseWallLayerActive(tmp_path):
    """Gate on wall-completed sampling, not the deficient fluid-only sum."""
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from prepared_fluidMatrix import make_prepared_plan,prepared_case
    from run_fluidMatrix import make_plan
    from warpSPH.runner import prepareContext,CaseSpec
    from warpSPH.cases.freeSurface.sloshingTank import sloshingTankCase
    from warpSPH.modules.density import computeDensities
    from warpSPH.modules.iisph import initializeIISPHBoundaries,computeIISPHProjectionGeometry
    from warpSPH.modules.dfsph import solveDFSPHProjection
    from warpSPHCore import buildVerletList,SupportScheme
    plan=make_prepared_plan(make_plan,methods=('dfsph',),scale='screen')
    row=next(r for r in plan['rows'] if r['test']=='sloshing' and r['boundary']=='analytic')
    row['spec'].update(nx=80,device='cpu')
    row['df_sampling_gate']={'minDivergenceDensityRatio':.9}
    case=prepared_case(sloshingTankCase,tmp_path,row)
    ctx=prepareContext(case,CaseSpec(caseName=case.name,scheme='dfsph',params={**case.params,**row['params']}).merged(**case.defaults).merged(**row['spec']))
    system=case.buildSystem(ctx);case.initialConditions(ctx,system)
    state=system.state;cfg=ctx.config;sc=ctx.schemeConfig
    support,provider,_=initializeIISPHBoundaries(state,sc)
    assert provider is not None and (state.kinds==0).all()
    adj=buildVerletList(state,cfg.domain,supportMode=SupportScheme.SuperSymmetric)
    raw=computeDensities(state,cfg,sc,adj,wall=False)/sc.fluid.restDensity
    g=computeIISPHProjectionGeometry(state,cfg,sc,adj,analyticProvider=provider,boundarySupport=support)
    ratio=g[1];floor=(state.positions[:,1]<cfg.dx)&(state.positions[:,0].abs()<.2)
    completed=floor&(raw<.9)&(ratio>=.9)
    assert completed.any() and (ratio[completed]>raw[completed]+.1).all()
    policy=sc.dfsph.divergence;policy.minIterations=policy.maxIterations=8
    velocity=torch.zeros_like(state.velocities);velocity[:,1]=-1.
    result=solveDFSPHProjection(state,cfg,sc,.0002,adj,*g[:5],mode='divergence',
        velocity=velocity,contactPoints=None)
    assert (result.pressure[completed]>0).any()
    assert torch.count_nonzero(result.pressure[ratio<.9])==0
    assert torch.isfinite(result.acceleration).all()


@pytest.mark.parametrize('boundary',['particles','analytic'])
@pytest.mark.parametrize('order',['df_np_cd','np_df_cd','np_df_shift'])
@pytest.mark.parametrize('device',['cpu',pytest.param('cuda',marks=pytest.mark.skipif(
    not torch.cuda.is_available(),reason='CUDA'))])
def test_allNativeCompositionsUsePrescribedDensityAndSharedRecovery(tmp_path,monkeypatch,boundary,order,device):
    """Exercise actual stage routing, wall operators and the post-drift CD solve."""
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from prepared_fluidMatrix import make_prepared_plan,prepared_case
    from run_fluidMatrix import make_plan
    from warpSPH.runner import prepareContext,CaseSpec
    from warpSPH.cases.freeSurface.sloshingTank import sloshingTankCase
    from warpSPH.configurations.moduleConfigurations.boundaryConditions import BoundaryCondition,BoundaryConditionType
    from warpSPH.schemes.incompressible import dfsph
    from warpSPH.modules.dfsph import composition
    from warpSPH.modules.iisph import initializeIISPHBoundaries
    method={'df_np_cd':'dfsph','np_df_cd':'dfsphPredictor','np_df_shift':'dfsphShift'}[order]
    plan=make_prepared_plan(make_plan,methods=(method,),scale='screen')
    row=next(r for r in plan['rows'] if r['test']=='sloshing' and r['boundary']==boundary)
    row['spec'].update(nx=40,device=device)
    case=prepared_case(sloshingTankCase,tmp_path,row)
    ctx=prepareContext(case,CaseSpec(caseName=case.name,scheme='dfsph',params={**case.params,**row['params']}).merged(**case.defaults).merged(**row['spec']))
    system=case.buildSystem(ctx);case.initialConditions(ctx,system)
    st,cfg,sc=system.state,ctx.config,ctx.schemeConfig
    system.boundarySupport,system.analyticBoundaryProvider,system.contactBoundaryProvider=initializeIISPHBoundaries(st,sc)
    system._projectedStepActive=True
    fluid=st.kinds==0
    middle=st.positions.new_tensor([0.,.045])
    indices=(st.positions-middle).square().sum(1).masked_fill(~fluid,float('inf')).argsort()[:2]
    def region(positions):
        distance=torch.ones_like(positions[:,0]);distance[indices]=-1
        return distance,torch.zeros_like(positions)
    def prescribe(state,*args):
        density=state.densities.clone()
        density[indices]=density.new_tensor([.5,.9])*sc.fluid.restDensity
        return density
    sc.boundaryConditions=[BoundaryCondition(BoundaryConditionType.constant,region,
        dirichletFunctions={'densities':prescribe})]
    st.velocities[fluid]=-.01*st.positions[fluid]
    st.divergencePressureImpulse=torch.ones_like(st.masses)*1.e-4
    assert sc.dfsph.minDivergenceDensityRatio==.9
    module=dfsph if order=='df_np_cd' else composition
    original=module.solveDFSPHProjection;calls=[]
    def observe(*args,**kw):
        result=original(*args,**kw);ratio=args[6]
        if kw['mode']=='divergence':
            torch.testing.assert_close(ratio[indices],ratio.new_tensor([.5,.9]),rtol=0,atol=0)
            assert torch.count_nonzero(result.pressure[fluid&(ratio<.9)])==0
        else:
            # Same density input must be independent of the DF gate, including
            # the position-only correction after VD+PS drift.
            sc.dfsph.minDivergenceDensityRatio=0.
            try:ungated=original(*args,**kw)
            finally:sc.dfsph.minDivergenceDensityRatio=.9
            assert torch.equal(result.pressure,ungated.pressure)
            assert torch.equal(result.acceleration,ungated.acceleration)
        calls.append(kw['mode']);return result
    monkeypatch.setattr(module,'solveDFSPHProjection',observe)
    if order=='df_np_cd':dfsph.dfsph_step(system,.0002,cfg,sc)
    else:composition.advanceDFSPH(system,.0002,cfg,sc,order)
    assert calls==['divergence','density']
    assert torch.isfinite(st.positions).all() and torch.isfinite(st.velocities).all()
