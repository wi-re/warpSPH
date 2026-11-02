"""Small CPU search-envelope/whole-attempt fixtures; no simulation campaign."""
from types import SimpleNamespace

import pytest
import torch

from warpSPH.enumTypes import AdaptiveSupportScheme
from warpSPH.evaluation import CompressibleSpatialResources, MortonStepper, adopt_spatial_layout
from warpSPH.evaluation.coverage import (CoverageBuilder, CoverageEnvelope,
    CoverageInvalidated, CoverageRetryExhausted)
from warpSPHIntegrators import get_scheme
from warpSPHIntegrators.solvers import JFNKSolver
from warpSPHIntegrators.state import current_evaluation, resource_scope
from warpSPHCore import SupportScheme, buildVerletList, get_torch_precision
from warpSPH.utils.domain import buildDomainDescription
from test_particleLayout import shuffled
from test_spatialEvaluationResources import setup, evaluate


def request(path=(0,), purpose='fd_probe'):
    return SimpleNamespace(stage_path=path,purpose=purpose)


def pairs(adj):
    i,j=adj.livePairs()
    return set(zip(i.tolist(),j.tolist()))


@pytest.mark.parametrize('dim',[1,2,3])
def test_renewed_candidate_contains_independent_periodic_physical_pairs_and_owns_evidence(dim):
    dtype=get_torch_precision()
    domain=buildDomainDescription(l=2,dim=dim,periodic=True,dtype=dtype)
    anchors=torch.tensor([[-.92,-.1,.1],[.91,.2,-.2],[-.1,.1,-.1],[.3,-.3,.2]],dtype=dtype)[:,:dim].contiguous()
    radii=torch.full((4,),.2,dtype=dtype)
    old=buildVerletList(SimpleNamespace(positions=anchors,supports=radii),domain,
                        verletScale=1.,supportMode=SupportScheme.SuperSymmetric)
    ids=torch.arange(4,dtype=torch.int64)+2**40
    particle=SimpleNamespace(positions=anchors+.13,supports=torch.tensor([.4,.2,.5,.3],dtype=dtype))
    builder=CoverageBuilder(old,request(),ids)
    savedJ,savedA=old.paddedJ.clone(),old.queryPositions.clone()
    with pytest.raises(CoverageInvalidated) as caught:
        builder(particle,domain,verletScale=1.)
    renewal=caught.value.renewal
    assert renewal.purpose=='fd_probe' and renewal.envelope.stage_path==(0,)
    assert torch.all(renewal.envelope.radii>radii)
    config=SimpleNamespace(domain=domain,verletScale=1.,compute=None)
    new=renewal.envelope.build(config)
    # Independent all-pairs oracle in physical geometry, not CSR remapping.
    physical=set()
    for i in range(4):
        for j in range(4):
            dx=particle.positions[i]-particle.positions[j]
            dx=(dx+1).remainder(2)-1
            if dx.norm()<max(particle.supports[i],particle.supports[j]):physical.add((i,j))
    assert physical<=pairs(new)
    assert CoverageBuilder(new,request(),ids)(particle,domain) is new
    torch.testing.assert_close(old.paddedJ,savedJ,rtol=0,atol=0)
    torch.testing.assert_close(old.queryPositions,savedA,rtol=0,atol=0)
    # A producer changing its tensors after unwinding cannot alter the proposal.
    owned=renewal.required_supports.clone()
    particle.supports.zero_();ids.zero_()
    torch.testing.assert_close(renewal.required_supports,owned,rtol=0,atol=0)
    assert renewal.envelope.input_ids.tolist()==[2**40+i for i in range(4)]


def test_support_iteration_peak_is_checked_even_when_final_support_shrinks(monkeypatch):
    import warpSPH.modules.adaptiveSupport.optimalSupportOwen as owen
    ctx,initial=setup('CompSPH',compatible=False,adaptive=True)
    supplied=initial.state.supports.clone()
    ctx.schemeConfig.adaptiveSupportIterations=2
    original=initial.state
    fixed=buildVerletList(initial.state,ctx.config.domain,verletScale=1.1,
                          supportMode=SupportScheme.SuperSymmetric)
    envelope=CoverageEnvelope((0,),initial.state.UIDs.clone(),fixed.queryPositions.clone(),fixed.querySupports.clone())
    visits=[]
    def supports(*args):
        result=supplied*(2 if len(visits)==0 else 1)
        visits.append(result.clone())
        return result
    monkeypatch.setattr(owen,'computeNewSupport',supports)
    resources=CompressibleSpatialResources()
    with resources.session(coverage={(0,):(envelope,fixed)}), resource_scope('stage',key=0), resource_scope('nonlinear_solve'), resource_scope('linearization'):
        with pytest.raises(CoverageInvalidated) as caught, resource_scope('fd_probe'):
            evaluate(ctx,initial,initial)
    assert len(visits)==2
    torch.testing.assert_close(visits[-1],supplied,rtol=0,atol=0)
    torch.testing.assert_close(caught.value.renewal.required_supports,2*supplied,rtol=0,atol=0)
    assert initial.state is original and initial.pairWork is None
    torch.testing.assert_close(initial.state.supports,supplied,rtol=0,atol=0)
    assert current_evaluation() is None


@pytest.mark.parametrize('method',['CompSPH','CRKSPH'])
@pytest.mark.parametrize('name',['Backward Euler (implicit)','Gauss-Legendre 2'])
@pytest.mark.parametrize('adaptive',['NoScheme','Owen','Monaghan','NumberDensity'])
def test_real_fd_coverage_loss_restarts_whole_solve_without_resorting_or_input_publication(method,name,adaptive,monkeypatch):
    import warpSPH.evaluation.sortedStep as boundary
    ctx,initial=shuffled(method,compatible=False,adaptive=False)
    ctx.schemeConfig.adaptiveSupportScheme=getattr(AdaptiveSupportScheme,adaptive)
    ctx.config.verletScale=1.0 # no skin: a genuine FD geometry perturbation needs renewal
    initial.state.velocities+=.05*torch.sin(2*torch.pi*initial.state.positions)
    before={n:v.clone() for n,v in vars(initial.state).items() if isinstance(v,torch.Tensor)}
    order_calls=[]
    original=boundary.morton_order
    def sort(*args,**kwargs):
        order_calls.append(1);return original(*args,**kwargs)
    monkeypatch.setattr(boundary,'morton_order',sort)
    seen=[]
    def observe(req,prepared):
        solve=req.find('nonlinear_solve')
        seen.append((req.purpose,req.stage_path, None if solve is None else solve.resource,
                     req.state.state.UIDs.clone(),prepared.adjacency))
    driver=get_scheme(name)
    with torch.no_grad():
        result=MortonStepper(driver,observe=observe)(initial,1e-4,
            config=ctx.config,schemeConfig=ctx.schemeConfig,solver=JFNKSolver(tol=1e-10))
        control=driver(initial.initializeNewState(),1e-4,ctx.stepFunction,
            config=ctx.config,schemeConfig=ctx.schemeConfig,solver=JFNKSolver(tol=1e-10))
    assert result.step_status.successful and len(order_calls)==1
    assert result.state.spatialContext.coverage_renewals>=1
    tokens=[]
    for purpose,path,token,ids,adj in seen:
        if token is not None and not any(token is t for t in tokens):tokens.append(token)
        assert torch.equal(ids,result.state.spatialContext.layout.compute_ids)
    assert len(tokens)>=2 # failed and fresh nonlinear_solve handles, never one resumed solve
    assert initial.adjacency is None and initial.spatialContext is None and initial.pairWork is None
    for field,value in before.items():torch.testing.assert_close(getattr(initial.state,field),value,rtol=0,atol=0)
    for field in ('positions','velocities','internalEnergies','supports','densities'):
        torch.testing.assert_close(getattr(result.state.state,field),getattr(control.state.state,field),rtol=3e-5,atol=3e-6)
    # An independent Butcher stage equation checks the returned stage association.
    tableau=driver.definition.tableau
    for i,stage in enumerate(result.stages):
        expected=initial.state.positions.clone()
        for j,other in enumerate(result.stages):expected+=1e-4*float(tableau.a[i][j])*other.update.dxdt
        torch.testing.assert_close(stage.aux[1].positions,expected,rtol=1e-6,atol=1e-7)
    adopted=adopt_spatial_layout(result)
    assert adopted.state.spatialContext.coverage_renewals==result.state.spatialContext.coverage_renewals
    assert adopted.state.adjacency is result.state.spatialContext.adjacency
    assert current_evaluation() is None


def test_zero_retry_budget_reports_owned_proposal_and_keeps_input_untouched():
    ctx,initial=shuffled(compatible=False,adaptive=False)
    ctx.config.verletScale=1.
    initial.state.velocities+=.05*torch.sin(2*torch.pi*initial.state.positions)
    positions=initial.state.positions.clone()
    with torch.no_grad(),pytest.raises(CoverageRetryExhausted) as caught:
        MortonStepper(get_scheme('Backward Euler (implicit)'),max_coverage_retries=0)(initial,1e-4,
            config=ctx.config,schemeConfig=ctx.schemeConfig,solver=JFNKSolver(tol=1e-10))
    assert caught.value.attempts==1 and caught.value.renewal.purpose=='fd_probe'
    assert initial.spatialContext is None and initial.adjacency is None
    torch.testing.assert_close(initial.state.positions,positions,rtol=0,atol=0)
    assert current_evaluation() is None


@pytest.mark.parametrize('bad',[-1,True,1.5])
def test_invalid_renewal_budget_is_rejected(bad):
    with pytest.raises(ValueError,match='max_coverage_retries'):
        MortonStepper(get_scheme('rungeKutta4'),max_coverage_retries=bad)



def test_coupled_retry_budget_counts_complete_attempts_and_does_not_leak_to_next_call():
    ctx,initial=shuffled(compatible=False,adaptive=False)
    ctx.config.verletScale=1.
    initial.state.velocities+=.05*torch.sin(2*torch.pi*initial.state.positions)
    stepper=MortonStepper(get_scheme('Gauss-Legendre 2'),max_coverage_retries=1)
    with torch.no_grad(),pytest.raises(CoverageRetryExhausted) as caught:
        stepper(initial,1e-4,config=ctx.config,schemeConfig=ctx.schemeConfig,solver=JFNKSolver(tol=1e-10))
    assert caught.value.attempts==2
    assert caught.value.renewal.envelope.stage_path==(1,)
    assert initial.adjacency is None and initial.spatialContext is None
    stepper.max_coverage_retries=2
    with torch.no_grad():
        result=stepper(initial,1e-4,config=ctx.config,schemeConfig=ctx.schemeConfig,solver=JFNKSolver(tol=1e-10))
    assert result.state.spatialContext.coverage_renewals==2
    assert current_evaluation() is None


@pytest.mark.parametrize('retry_budget',[0,2])
def test_runner_adopts_only_after_successful_renewed_attempts(retry_budget):
    from dataclasses import replace
    from warpSPH.runner import run
    from warpSPH.cases.compressible.linearWave import linearWaveCase
    from test_runnerParticleOrder import fixture_case, options
    called=[]
    def configure(ctx):
        linearWaveCase.configureScheme(ctx)
        ctx.schemeConfig.compatibleEnergy=False
        ctx.schemeConfig.adaptiveSupportScheme=AdaptiveSupportScheme.NoScheme
    def initial(ctx,system):
        linearWaveCase.initialConditions(ctx,system)
        system.state.velocities+=.05*torch.sin(2*torch.pi*system.state.positions)
    case=replace(fixture_case(postStep=lambda ctx,state,i:called.append(i)),
                 configureScheme=configure,initialConditions=initial)
    args=options();args.update(nSteps=2,scheme='CompSPH',integrationScheme='backwardEuler',
                              verletScale=1.,particleOrder='morton',coverageRetries=retry_budget)
    if retry_budget==0:
        with torch.no_grad(),pytest.raises(CoverageRetryExhausted):run(case,**args)
        assert called==[]
        return
    with torch.no_grad():result=run(case,**args)
    assert called==[0,1] and result.nSteps==2 and not result.diverged
    assert sum(row.get('coverageRenewals',0) for row in result.trajectory)>=1
    assert result.state.spatialContext.layout.identity



@pytest.mark.parametrize('bad', ['nan_position','nan_support','negative_support'])
def test_invalid_probe_geometry_cannot_generate_a_search_envelope(bad):
    ctx,initial=setup('CompSPH',adaptive=False)
    fixed=buildVerletList(initial.state,ctx.config.domain,verletScale=ctx.config.verletScale,
                          supportMode=SupportScheme.SuperSymmetric)
    particle=initial.state.initializeNewState()
    if bad=='nan_position':particle.positions[0]=float('nan')
    elif bad=='nan_support':particle.supports[0]=float('nan')
    else:particle.supports[0]=-1
    with pytest.raises(ValueError,match='finite geometry'):
        CoverageBuilder(fixed,request(),particle.UIDs)(particle,ctx.config.domain)


def test_envelope_identity_mismatch_is_refused_before_equations():
    ctx,initial=setup('CompSPH',adaptive=False)
    fixed=buildVerletList(initial.state,ctx.config.domain,verletScale=ctx.config.verletScale,
                          supportMode=SupportScheme.SuperSymmetric)
    envelope=CoverageEnvelope((0,),initial.state.UIDs.roll(1),fixed.queryPositions,fixed.querySupports)
    resources=CompressibleSpatialResources()
    with resources.session(coverage={(0,):(envelope,fixed)}),resource_scope('stage',key=0),resource_scope('nonlinear_solve'):
        with pytest.raises(ValueError,match='identity/order changed'):
            evaluate(ctx,initial,initial)
    assert initial.pairWork is None and current_evaluation() is None
