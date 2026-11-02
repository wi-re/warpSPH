"""Tiny CPU continuation fixtures; no long simulation/video campaign."""
from dataclasses import replace
from copy import copy

import pytest
import torch

from warpSPH.evaluation import (CompressibleSpatialResources, MortonStepper, adopt_spatial_layout,
                                adopt_spatial_continuation, canonical_view)
from warpSPH.evaluation.continuation import history_view, stage_view, uid_order
from warpSPH.enumTypes import ViscositySwitch
from warpSPHIntegrators import get_scheme
from warpSPHIntegrators.runtime import Continuation, solve
from warpSPHIntegrators.solvers import JFNKSolver
from warpSPHIntegrators.state.history import StepHistory
from test_particleLayout import shuffled, particle_pairs


def fixture(method='CompSPH', *, fsal=False):
    ctx, state = shuffled(method, compatible=False, adaptive=not fsal)
    state.state.UIDs = (2**35 + 13 * torch.randperm(32, generator=torch.Generator().manual_seed(8))).long()
    state.state.velocities += .03 * torch.sin(2*torch.pi*state.state.positions)
    if fsal:
        ctx.schemeConfig.viscositySwitchParams.scheme = ViscositySwitch.NoneSwitch
    return ctx, state


def assert_physics(a, b):
    a, b = canonical_view(a, include_adjacency=False)[0].state, canonical_view(b, include_adjacency=False)[0].state
    for name in ('positions', 'velocities', 'internalEnergies', 'supports', 'densities', 'alpha0s'):
        torch.testing.assert_close(getattr(a,name), getattr(b,name), rtol=4e-5, atol=4e-6)


def input_order_step(scheme,state,history,kwargs):
    """Independent unsorted control with the same support/probe/renewal policy."""
    from warpSPH.evaluation.coverage import CoverageInvalidated
    coverage={}
    rhs=kwargs.pop('rhs')
    for _ in range(17):
        try:
            with CompressibleSpatialResources().session(coverage=coverage):
                return scheme(state.initializeNewState(),1e-4,rhs,history=history,**kwargs)
        except CoverageInvalidated as failure:
            envelope=failure.renewal.envelope
            coverage[envelope.stage_path]=(envelope,envelope.build(kwargs['config']))
    pytest.fail('input-order control exhausted 16 complete renewals')


@pytest.mark.parametrize('fluid', ['CompSPH', 'CRKSPH'])
@pytest.mark.parametrize('name', ['ab2','ab3','ab4','ab5','abm2','abm3','abm4','abm5',
                                  'am2','am3','am4','bdf1','bdf2','bdf3','bdf4','bdf5'])
@pytest.mark.parametrize('adopt', [False,True])
def test_multistep_matches_native_with_uid_joined_history_and_boundary_resorts(fluid,name,adopt):
    ctx, initial = fixture(fluid)
    initial.state.velocities += 120.
    ctx.config.verletScale = 1.05
    scheme = get_scheme(name)
    # High-order BDF starters have several implicit stage envelopes; this
    # low-skin translating fixture intentionally needs more than two renewals.
    stepper = MortonStepper(scheme,max_coverage_retries=16)
    state, control = initial, initial.initializeNewState()
    # The native control also preserves raw Adams derivatives. The adapter sets
    # this opt-in flag privately; the default legacy runner remains unchanged.
    control.retainStageDerivatives = True
    history = control_history = StepHistory(maxlen=max(1,scheme.metadata.history_length))
    orders, snapshots, compute_ids = [], [], []
    kwargs = dict(config=ctx.config, schemeConfig=ctx.schemeConfig)
    if scheme.implicit:
        kwargs['solver'] = JFNKSolver(tol=1e-6 if initial.state.positions.dtype == torch.float32 else 1e-10, gmres_maxiter=96)
    for step in range(max(4,scheme.metadata.history_length+1)):
        if state.spatialContext is not None:
            # Evict only the boundary cache. Implicit endpoint materialization
            # can leave a fresh list, so motion alone need not trigger a sort.
            state=copy(state)
            state.spatialContext=replace(state.spatialContext,adjacency=None)
        old_history = [(e.update.dudt.clone(), e.aux[1].UIDs.clone()) for e in history]
        result = stepper(state, 1e-4, history=history, **kwargs)
        # Match preparation/probe isolation independently of coordinate order.
        # Legacy implicit RHS evaluation can mutate support scratch in probes.
        baseline = input_order_step(scheme,control,control_history,dict(kwargs,rhs=ctx.stepFunction))
        assert result.step_status.successful and baseline.step_status.successful
        assert_physics(result.state, baseline.state)
        orders.append(result.state.spatialContext.layout.identity)
        compute_ids.append(result.state.spatialContext.layout.compute_ids.clone())
        for entry, saved in zip(history, old_history):
            torch.testing.assert_close(entry.update.dudt, saved[0], rtol=0,atol=0)
            assert torch.equal(entry.aux[1].UIDs, saved[1])
        assert len(result.history) == min(step+1,history.maxlen)
        for entry, reference in zip(result.history, baseline.history):
            order = uid_order(reference.aux[1].UIDs, entry.aux[1].UIDs)
            for field in ('dxdt','dvdt','dudt'):
                atol=5e-6 if initial.state.positions.dtype == torch.float32 else 1e-8
                torch.testing.assert_close(getattr(entry.update,field),getattr(reference.update,field)[order],rtol=5e-5,atol=atol)
            if entry.state is not None:
                assert_physics(entry.state, reference.state)
                assert torch.equal(entry.state.state.UIDs,result.state.state.UIDs)
        snapshots.append((result,result.state.state.UIDs.clone()))
        accepted = adopt_spatial_layout(result) if adopt else result
        state, history = accepted.state, accepted.history
        control, control_history = baseline.state, baseline.history
        assert all(torch.equal(entry.aux[1].UIDs,state.state.UIDs) for entry in history)
    assert not all(orders)
    assert any(not torch.equal(a,b) for a,b in zip(compute_ids,compute_ids[1:]))
    assert all(torch.equal(result.state.state.UIDs,ids) for result,ids in snapshots)
    # Once bootstrapped, these methods perform their own formula, not DP5 again.
    assert len(result.stages) == 1


def test_history_views_preserve_csr_pairs_and_snapshot_ownership():
    ctx, initial = fixture()
    stepper = MortonStepper(get_scheme('bdf2'))
    result = stepper(initial, 1e-4, config=ctx.config, schemeConfig=ctx.schemeConfig)
    history = result.history
    target = torch.flip(initial.state.UIDs,[0])
    mapped = history_view(history,target)
    native = history_view(mapped,target,native=True)
    original = history[0].aux[2]
    assert particle_pairs(native[0].aux[0],target) == particle_pairs(original.adjacency,original.layout.compute_ids)
    assert torch.equal(native[0].state.state.UIDs,target)
    assert native[0].state.adjacency is None and native[0].state.spatialContext is None
    original_positions=history[0].state.state.positions.clone()
    native[0].state.state.positions.add_(1)
    torch.testing.assert_close(history[0].state.state.positions,original_positions,rtol=0,atol=0)
    assert torch.equal(history[0].state.state.UIDs,initial.state.UIDs)


@pytest.mark.parametrize('bad', ['context','foreign','snapshot','capacity','edge','update'])
def test_invalid_history_fails_before_sort_or_rhs(bad,monkeypatch):
    import warpSPH.evaluation.sortedStep as boundary
    ctx, initial = fixture()
    stepper = MortonStepper(get_scheme('bdf3'))
    first = stepper(initial,1e-4,config=ctx.config,schemeConfig=ctx.schemeConfig)
    history = first.history
    if bad == 'context':
        history = StepHistory(3,history.entries,context={'unknown':torch.ones(32)})
    elif bad == 'foreign':
        history[0].aux[1].UIDs[0] += 1
    elif bad == 'snapshot':
        history[0].state.state.UIDs = history[0].state.state.UIDs.flip(0)
    elif bad == 'edge':
        history[0].aux[2].edge_fields['unknown']=torch.ones(32)
    elif bad == 'update':
        history[0].update.unknown=torch.ones(32)
    else:
        history = StepHistory(1)
    monkeypatch.setattr(boundary,'morton_order',lambda *a,**k:pytest.fail('invalid history reached sort'))
    with pytest.raises((ValueError,TypeError)):
        stepper(initial,1e-4,config=ctx.config,schemeConfig=ctx.schemeConfig,history=history)


@pytest.mark.parametrize('fluid', ['CompSPH','CRKSPH'])
@pytest.mark.parametrize('adopt',[False,True])
def test_fsal_reuses_raw_endpoint_after_adoption_and_preserves_stage_observations(fluid,adopt):
    ctx, initial = fixture(fluid,fsal=True)
    seen = []
    stepper = MortonStepper(get_scheme('dormandPrince'),observe=lambda r,p:seen.append(r.stage_path))
    first = stepper(initial,1e-4,config=ctx.config,schemeConfig=ctx.schemeConfig,history=StepHistory(1))
    prior = first.stages[-1]
    assert prior.aux[2].reuse is not None
    raw = prior.aux[2].reuse.update
    # The diagnostic replacement really differs, so this catches accidentally
    # reusing step-average thermal rates as an endpoint derivative.
    order = uid_order(prior.aux[2].reuse.ids,initial.state.UIDs)
    assert not torch.equal(raw.dudt[order],prior.update.dudt)
    memory = Continuation(first.state,first.history,prior)
    if adopt:
        memory = adopt_spatial_continuation(memory)
    assert torch.equal(prior.aux[1].UIDs,initial.state.UIDs)
    assert all(torch.equal(e.aux[1].UIDs,memory.state.state.UIDs) for e in memory.history)
    count = len(seen)
    second = stepper(memory.state,1e-4,config=ctx.config,schemeConfig=ctx.schemeConfig,
                     history=memory.history,priorStep=memory.prior_step)
    assert len(seen)-count == 6
    fresh = stepper(memory.state,1e-4,config=ctx.config,schemeConfig=ctx.schemeConfig)
    assert_physics(second.state,fresh.state)
    assert [path for path in seen[count:count+6]] == [(i,) for i in range(1,7)]


@pytest.mark.parametrize('change',['dt','state','config','support','switch','compatible'])
def test_fsal_refreshes_when_the_endpoint_contract_changes(change):
    ctx, initial = fixture(fsal=True)
    seen=[]
    stepper = MortonStepper(get_scheme('dormandPrince'),observe=lambda r,p:seen.append(r.stage_path))
    first = stepper(initial,1e-4,config=ctx.config,schemeConfig=ctx.schemeConfig)
    state = first.state
    dt = 1e-4
    if change == 'dt':dt *= 2
    elif change == 'state':state.state.internalEnergies += .01
    elif change == 'config':ctx.config.verletScale *= 1.1; state.spatialContext = None
    elif change == 'support':
        from warpSPH.enumTypes import AdaptiveSupportScheme
        ctx.schemeConfig.adaptiveSupportScheme = AdaptiveSupportScheme.Owen
    elif change == 'switch':ctx.schemeConfig.viscositySwitchParams.scheme = ViscositySwitch.CullenHopkins
    else:ctx.schemeConfig.compatibleEnergy = True
    count=len(seen)
    result = stepper(state,dt,config=ctx.config,schemeConfig=ctx.schemeConfig,priorStep=first.stages[-1])
    assert len(seen)-count >= 7 and seen[count] == (0,)
    fresh = stepper(state,dt,config=ctx.config,schemeConfig=ctx.schemeConfig)
    assert_physics(result.state,fresh.state)


@pytest.mark.parametrize('name',['dormandPrince','ab3','bdf2'])
def test_generic_runtime_uses_accepted_memory_without_rewriting_callback_stages(name):
    ctx, initial = fixture(fsal=True)
    scheme = get_scheme(name)
    seen, records = [], []
    stepper = MortonStepper(scheme,observe=lambda r,p:seen.append(r.stage_path))
    kwargs = dict(config=ctx.config,schemeConfig=ctx.schemeConfig)
    if scheme.implicit:kwargs['solver']=JFNKSolver(tol=1e-6)
    result = solve(stepper.as_scheme(),initial,4e-4,ctx.stepFunction,dt=1e-4,adaptive=False,
                   reuse=True,continuation=adopt_spatial_continuation,
                   allow_stagnation=True,callback=lambda record:records.append(record),**kwargs)
    assert result.successful and result.accepted_steps == 4
    assert result.state.spatialContext.layout.identity
    assert all(torch.equal(e.aux[1].UIDs,result.state.state.UIDs) for e in result.history)
    assert torch.equal(result.prior_step.aux[1].UIDs,result.state.state.UIDs)
    assert torch.equal(records[0].result.state.state.UIDs,initial.state.UIDs)
    assert all(torch.equal(s.aux[1].UIDs,record.result.state.state.UIDs) for record in records for s in record.result.stages)
    if name == 'dormandPrince':
        # A floating-point terminal truncation may change dt by an ulp. The
        # conservative cache requires exactly equal dt and refreshes then.
        assert len(seen) == 7+sum(6 if a.dt == b.dt else 7 for a,b in zip(records,records[1:]))


def test_runner_threads_multistep_history_only_on_opt_in_route():
    from warpSPH.runner import run
    from test_runnerParticleOrder import fixture_case,options
    histories=[]
    def post(ctx,state,i):
        histories.append(ctx.scratch.get('stepHistory'))
    from warpSPH.cases.compressible.linearWave import linearWaveCase
    def configure(ctx):
        linearWaveCase.configureScheme(ctx)
        ctx.schemeConfig.compatibleEnergy=False
    case=replace(fixture_case(postStep=post),configureScheme=configure)
    args=options();args.update(nSteps=4,integrationScheme='ab3',particleOrder='morton')
    result=run(case,**args)
    assert not result.diverged
    history=result.ctx.scratch['stepHistory']
    assert len(history)==2 and [0 if h is None else len(h) for h in histories]==[0,1,2,2]
    assert all(torch.equal(e.aux[1].UIDs,result.state.state.UIDs) for e in history)


def test_runner_physical_hook_restarts_history_and_state_only_restart_is_rejected(tmp_path):
    from warpSPH.runner import run
    from warpSPH.cases.compressible.linearWave import linearWaveCase
    from test_runnerParticleOrder import fixture_case,options
    def configure(ctx):
        linearWaveCase.configureScheme(ctx)
        ctx.schemeConfig.compatibleEnergy=False
    def jump(ctx,state,i):
        state.state.velocities.add_(.001)
    case=replace(fixture_case(postStep=jump),configureScheme=configure)
    args=options();args.update(integrationScheme='ab2',particleOrder='morton',
                               store=True,exportRoot=str(tmp_path))
    result=run(case,**args)
    assert len(result.ctx.scratch['stepHistory'])==0
    args.update(resumeFrom=str(tmp_path/'not-used.h5'))
    # Preflight directly: the existing state checkpoint format has no numerical
    # history. Do not silently turn a resumed multistep run into a cold startup.
    from warpSPH.runner.runner import _spatialStepper
    from warpSPH.runner import CaseSpec
    with pytest.raises(ValueError,match='persisted integrator history'):
        _spatialStepper(result.ctx,CaseSpec(**args),result.state)


@pytest.mark.parametrize('name',['ab2','abm3'])
def test_adams_history_keeps_actual_start_derivative_including_thermal_rate(name):
    from warpSPHIntegrators.state.lifecycle import updateStep
    ctx,state=fixture(fsal=True)
    stepper=MortonStepper(get_scheme(name))
    history=None
    for _ in range(4):
        oracle=state.initializeNewState()
        oracle.initialize(1e-4,config=ctx.config,schemeConfig=ctx.schemeConfig)
        raw,_=updateStep(oracle,oracle.initializeNewState(),1e-4,ctx.stepFunction,
                         config=ctx.config,schemeConfig=ctx.schemeConfig)
        result=stepper(state,1e-4,config=ctx.config,schemeConfig=ctx.schemeConfig,history=history)
        latest=result.history.latest
        torch.testing.assert_close(latest.update.dudt,raw.dudt,rtol=3e-5,atol=3e-6)
        state,history=adopt_spatial_layout(result).state,adopt_spatial_layout(result).history


def test_runtime_dense_samples_use_raw_derivatives_and_adaptive_rejections_do_not_adopt():
    ctx,initial=fixture(fsal=True)
    initial.state.velocities += .3*torch.sin(2*torch.pi*initial.state.positions)
    stepper=MortonStepper(get_scheme('dormandPrince'))
    adopted=[]
    def accept(memory):
        adopted.append(float(memory.state.t))
        return adopt_spatial_continuation(memory)
    kwargs=dict(dt=.1,adaptive=True,rtol=1e-5,atol=1e-7,
                sample_times=(.003,.009,.1),config=ctx.config,schemeConfig=ctx.schemeConfig)
    result=solve(stepper.as_scheme(),initial,.1,ctx.stepFunction,continuation=accept,**kwargs)
    control=initial.initializeNewState();control.retainStageDerivatives=True
    baseline=solve(get_scheme('dormandPrince'),control,.1,ctx.stepFunction,reuse=False,**kwargs)
    assert result.successful and result.rejected_steps>0
    assert len(adopted)==result.accepted_steps
    for sample,reference in zip(result.samples,baseline.samples):
        assert sample.t==reference.t
        assert_physics(sample.state,reference.state)


@pytest.mark.parametrize('name',['dormandPrince','bdf2'])
def test_runtime_in_memory_checkpoint_continues_history_and_reuse(name):
    ctx,initial=fixture(fsal=True)
    scheme=MortonStepper(get_scheme(name)).as_scheme()
    kwargs=dict(dt=1e-4,adaptive=False,allow_stagnation=True,
                continuation=adopt_spatial_continuation,config=ctx.config,schemeConfig=ctx.schemeConfig)
    if name=='bdf2':kwargs['solver']=JFNKSolver(tol=1e-6)
    first=solve(scheme,initial,2e-4,ctx.stepFunction,**kwargs)
    resumed=solve(scheme,None,4e-4,ctx.stepFunction,restart=first.checkpoint,**kwargs)
    complete=solve(scheme,initial,4e-4,ctx.stepFunction,**kwargs)
    assert_physics(resumed.state,complete.state)
    assert len(resumed.history)==len(complete.history)
    for a,b in zip(resumed.history,complete.history):
        torch.testing.assert_close(a.update.dudt,b.update.dudt,rtol=0,atol=0)


def test_runtime_compatible_energy_and_non_fsal_prior_are_explicitly_rejected():
    ctx,initial=shuffled()
    with pytest.raises(ValueError,match='compatibleEnergy=False'):
        solve(MortonStepper(get_scheme('dormandPrince')).as_scheme(),initial,1e-4,
              ctx.stepFunction,dt=1e-4,adaptive=False,config=ctx.config,schemeConfig=ctx.schemeConfig)
    stepper=MortonStepper(get_scheme('rungeKutta4'))
    first=stepper(initial,1e-4,config=ctx.config,schemeConfig=ctx.schemeConfig)
    with pytest.raises(ValueError,match='explicit FSAL'):
        stepper(first.state,1e-4,config=ctx.config,schemeConfig=ctx.schemeConfig,priorStep=first.stages[-1])


def test_legacy_prior_tuple_and_history_bookkeeping_do_not_enable_implicit_reuse():
    ctx,state=fixture(fsal=True)
    seen=[]
    stepper=MortonStepper(get_scheme('dormandPrince'),observe=lambda r,p:seen.append(r.stage_path))
    history=StepHistory(1,_uid_fingerprint=(state.state.UIDs.data_ptr(),tuple(state.state.UIDs.shape)))
    first=stepper(state,1e-4,config=ctx.config,schemeConfig=ctx.schemeConfig,history=history)
    assert first.history._uid_fingerprint is not None
    prior=first.history.as_prior_step()
    count=len(seen)
    result=stepper(first.state,1e-4,config=ctx.config,schemeConfig=ctx.schemeConfig,
                   priorStep=(prior.update,prior.aux))
    assert len(seen)-count==6
    count=len(seen)
    fresh=stepper(first.state,1e-4,config=ctx.config,schemeConfig=ctx.schemeConfig,history=first.history)
    assert len(seen)-count==7
    assert_physics(result.state,fresh.state)
