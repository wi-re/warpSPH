"""Runner lifecycle checks use tiny CPU fixtures; no video or flow campaign."""
from dataclasses import replace
from pathlib import Path

import h5py
import pytest
import torch

from warpSPH.cases.compressible.linearWave import linearWaveCase
from warpSPH.evaluation import MortonStepper, canonical_view
from warpSPH.evaluation.layout import permute_state
from warpSPH.runner import CaseSpec, run


def fixture_case(*, postStep=None, diagnostics=None, extraData=None):
    def build(ctx):
        system = linearWaveCase.buildSystem(ctx)
        order = torch.randperm(len(system.state.UIDs), generator=torch.Generator().manual_seed(17))
        system.state = permute_state(system.state, order)
        system.adjacency = None
        return system
    return replace(linearWaveCase, buildSystem=build, postStep=postStep,
                   diagnostics=diagnostics or linearWaveCase.diagnostics,
                   extraData=extraData or linearWaveCase.extraData)


def options(**kwargs):
    return dict(nx=32, nSteps=3, dt=1e-4, adaptiveDt=False, device='cpu',
                integrationScheme='rungeKutta4', quiet=True, progress=True,
                plot=False, store=False, **kwargs)


def by_uid(system):
    return canonical_view(system, include_adjacency=False)[0].state


@pytest.mark.parametrize('scheme', ['CompSPH', 'CRKSPH'])
def test_runner_adopts_after_hooks_and_outputs_then_reuses_identity_list(scheme, monkeypatch, tmp_path):
    import warpSPH.evaluation as evaluation
    import warpSPH.runner.runner as runner
    events, inputs, outputs, snapshots, adjacency = [], [], [], [], []
    native_call, native_adopt, native_output = MortonStepper.__call__, evaluation.adopt_spatial_layout, runner._plotAndStore
    def step(self, state, *args, **kwargs):
        inputs.append(state.state.UIDs.clone())
        if state.spatialContext is not None:
            assert state.spatialContext.layout.identity
            assert state.adjacency is adjacency[-1]
        result = native_call(self, state, *args, **kwargs)
        assert torch.equal(result.state.state.UIDs, inputs[-1])
        assert all(torch.equal(stage.aux[1].UIDs, inputs[-1]) for stage in result.stages)
        outputs.append(result)
        adjacency.append(result.state.spatialContext.adjacency)
        events.append('step')
        return result
    def adopt(result):
        assert events[-1] == 'output'
        events.append('adopt')
        snapshots.append(result.state.state.UIDs.clone())
        return native_adopt(result)
    def post(ctx, state, i):
        events.append('hook')
        assert torch.equal(state.state.UIDs, inputs[-1])
        torch.testing.assert_close(ctx.scratch['lastStageUpdate'].dvdt, outputs[-1].stages[-1].update.dvdt)
    def output(*args, **kwargs):
        events.append('output')
        return native_output(*args, **kwargs)
    monkeypatch.setattr(MortonStepper, '__call__', step)
    monkeypatch.setattr(evaluation, 'adopt_spatial_layout', adopt)
    monkeypatch.setattr(runner, '_plotAndStore', output)
    args = options()
    args.update(scheme=scheme, particleOrder='morton', store=True, storeInterval=1, exportRoot=str(tmp_path))
    result = run(fixture_case(postStep=post), **args)
    assert not result.diverged and result.nSteps == 3
    assert events == ['step','hook','output','adopt'] * 3
    assert not torch.equal(inputs[0], inputs[1])
    assert torch.equal(inputs[1], inputs[2])
    assert all(torch.equal(old.state.state.UIDs, ids) for old, ids in zip(outputs, snapshots))
    assert all(row['layoutAdoption_ms'] > 0 for row in result.trajectory[1:])
    for path in Path(result.exportPath, 'trajectory').glob('*.h5'):
        with h5py.File(path) as f:
            ids = f['state/UIDs'][:].tolist()
            assert ids == sorted(ids), path
            assert 'frame_num' in f.attrs
            if 'stages' in f:
                for stage in f['stages'].values():
                    assert stage['aux/1_state/UIDs'][:].tolist() == ids
    assert result.state.spatialContext.layout.identity
    # Scratch exposed to callers follows the adopted final state's rows too.
    p = outputs[-1].state.spatialContext.layout.p
    torch.testing.assert_close(result.ctx.scratch['lastStageUpdate'].dvdt,
                               outputs[-1].stages[-1].update.dvdt[p])


@pytest.mark.parametrize('scheme', ['CompSPH', 'CRKSPH'])
def test_runner_sorted_trajectory_matches_input_with_hook_forced_boundary_resorts(scheme, monkeypatch):
    calls, evaluated = [], []
    original = MortonStepper.__call__
    def track(self, state, *args, **kwargs):
        out = original(self, state, *args, **kwargs)
        calls.append(out.state.spatialContext.layout.identity)
        evaluated.append(out)
        return out
    monkeypatch.setattr(MortonStepper, '__call__', track)
    def translate(ctx, state, i):
        # A periodic translation preserves physics but exhausts the skin and
        # rotates Morton rows. The same physical hook runs on both controls.
        state.state.positions.add_(.21)
    case = fixture_case(postStep=translate)
    baseline = run(case, **options(), scheme=scheme)
    sorted_run = run(case, **options(), scheme=scheme, particleOrder='morton')
    assert not sorted_run.diverged
    assert calls == [False, False, False]
    last = evaluated[-1]
    p = last.state.spatialContext.layout.p
    torch.testing.assert_close(sorted_run.ctx.scratch['lastStageUpdate'].dvdt,
                               last.stages[-1].update.dvdt[p], rtol=0, atol=0)
    for name in ('UIDs', 'positions', 'velocities', 'internalEnergies', 'supports', 'densities'):
        torch.testing.assert_close(getattr(by_uid(sorted_run.state),name), getattr(by_uid(baseline.state),name),
                                   rtol=3e-5, atol=3e-6)


def test_input_default_does_not_construct_optional_stepper(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('default path entered optional Morton adapter')
    monkeypatch.setattr(MortonStepper, '__init__', forbidden)
    a = run(fixture_case(), **options(), scheme='CompSPH')
    b = run(fixture_case(), **options(), scheme='CompSPH', particleOrder='input')
    assert a.state.spatialContext is None and not a.state.canonicalExport
    for name in ('positions','velocities','supports','internalEnergies'):
        torch.testing.assert_close(getattr(a.state.state,name),getattr(b.state.state,name),rtol=0,atol=0)


@pytest.mark.parametrize('override, message', [
    ({'execution':'graph'}, 'eager execution'),
    ({'execution':'auto'}, 'eager execution'),
    ({'integrationScheme':'ab2'}, 'compatibleEnergy=False'),
    ({'scheme':'Monaghan'}, 'CompSPH or CRKSPH'),
])
def test_unsupported_runner_modes_fail_before_output_or_sort(override, message, tmp_path, monkeypatch):
    import warpSPH.evaluation.sortedStep as boundary
    def forbidden(*args, **kwargs):
        raise AssertionError('unsupported mode reached sorting')
    monkeypatch.setattr(boundary, 'morton_order', forbidden)
    args = options()
    args.update(scheme='CompSPH', store=True, exportRoot=str(tmp_path), particleOrder='morton')
    args.update(override)
    with pytest.raises(ValueError, match=message):
        run(linearWaveCase if override.get('scheme') == 'Monaghan' else fixture_case(), **args)
    assert not list(tmp_path.iterdir())


def test_runner_rejects_unlabelled_extra_arrays_before_creating_output(tmp_path):
    case = fixture_case(extraData=lambda ctx,state: {'samples':state.state.UIDs.clone()})
    args = options()
    args.update(store=True, exportRoot=str(tmp_path), particleOrder='morton', scheme='CompSPH')
    with pytest.raises(ValueError, match='extraData needs explicit axes'):
        run(case, **args)
    assert not list(tmp_path.iterdir())


def test_failed_step_is_not_published_to_hooks_or_adoption(monkeypatch):
    import warpSPH.evaluation as evaluation
    calls = []
    original = MortonStepper.__call__
    def fail(self, state, *args, **kwargs):
        out = original(self, state, *args, **kwargs)
        out.state.state.velocities.fill_(float('nan'))
        return out
    monkeypatch.setattr(MortonStepper, '__call__', fail)
    monkeypatch.setattr(evaluation, 'adopt_spatial_layout', lambda result:calls.append('adopt'))
    case = fixture_case(postStep=lambda *args:calls.append('hook'))
    result = run(case, **options(), scheme='CompSPH', particleOrder='morton')
    assert result.diverged and result.nSteps == 0 and calls == []
    assert result.stopReason == 'unsuccessful integrator step'
    assert torch.isfinite(result.state.state.velocities).all()
    assert result.state.spatialContext is None


def test_particle_order_configuration_and_cli_round_trip():
    from warpSPH.runner import RunConfiguration
    from warpSPH.runner.caseSpec import buildArgumentParser, specFromArgs
    from warpSPH.runner.configuration import simulationParameters
    parser = buildArgumentParser(caseParams=linearWaveCase.params, defaults=linearWaveCase.defaults)
    spec = specFromArgs(parser.parse_args(['--particleOrder', 'morton','--coverageRetries','3']), linearWaveCase.params, linearWaveCase.defaults)
    assert spec.particleOrder == 'morton' and spec.coverageRetries == 3
    assert CaseSpec.fromDict(spec.toDict()).particleOrder == 'morton'
    assert simulationParameters()['coverageRetries'].minimum == 0
    assert simulationParameters()['particleOrder'].choices == ('input','morton')
    config = RunConfiguration('linearWave', simulation={'particleOrder':'morton'}, solver={'scheme':'CompSPH'})
    assert config.prepare(linearWaveCase).spec.particleOrder == 'morton'
    with pytest.raises(ValueError, match='particleOrder'):
        run(linearWaveCase, particleOrder='oops')


@pytest.mark.parametrize('method', ['backwardEuler', 'gaussLegendre2'])
def test_runner_implicit_one_step_methods_keep_uid_associations(method):
    args = options()
    args.update(integrationScheme=method, scheme='CompSPH')
    def configure(ctx):
        linearWaveCase.configureScheme(ctx)
        ctx.schemeConfig.compatibleEnergy = False
    case = replace(fixture_case(), configureScheme=configure)
    with torch.no_grad():
        control = run(case, **args)
        spatial = run(case, **args, particleOrder='morton')
    assert not spatial.diverged and spatial.nSteps == 3
    for field in ('positions','velocities','internalEnergies','supports'):
        torch.testing.assert_close(getattr(by_uid(spatial.state),field),getattr(by_uid(control.state),field),
                                   rtol=3e-5,atol=3e-6)


def test_coverage_abort_never_adopts_or_runs_post_step(monkeypatch):
    from warpSPHIntegrators.state import ResourceInvalidated
    import warpSPH.evaluation as evaluation
    before, effects = [], []
    def abort(self, state, *args, **kwargs):
        before.append(state)
        raise ResourceInvalidated('renew complete solve')
    monkeypatch.setattr(MortonStepper, '__call__', abort)
    monkeypatch.setattr(evaluation, 'adopt_spatial_layout', lambda value:effects.append('adopt'))
    with pytest.raises(ResourceInvalidated, match='renew complete solve'):
        run(fixture_case(postStep=lambda *args:effects.append('hook')), **options(),
            scheme='CompSPH',particleOrder='morton')
    assert effects == [] and before[0].spatialContext is None
    assert torch.isfinite(before[0].state.positions).all()



def test_time_limit_terminal_step_runs_outputs_before_adopting(monkeypatch):
    import warpSPH.evaluation as evaluation
    import warpSPH.runner.runner as runner
    events = []
    native_output, native_adopt = runner._plotAndStore, evaluation.adopt_spatial_layout
    def output(*args, **kwargs):
        events.append(('output',kwargs['final']))
        return native_output(*args, **kwargs)
    def adopt(value):
        assert events[-1][0] == 'output'
        events.append(('adopt',True))
        return native_adopt(value)
    monkeypatch.setattr(runner,'_plotAndStore',output)
    monkeypatch.setattr(evaluation,'adopt_spatial_layout',adopt)
    args = options()
    args.update(nSteps=None,tLimit=1.5e-4)
    case = replace(fixture_case(),timestep=lambda ctx,state:1e-4)
    result = run(case,**args,scheme='CompSPH',particleOrder='morton')
    assert result.nSteps == 2 and not result.diverged
    assert events == [('output',False),('adopt',True),('output',True),('adopt',True)]
    assert result.state.spatialContext.layout.identity



def test_custom_runner_rhs_is_rejected_before_output(tmp_path):
    base = fixture_case()
    def build(ctx):
        system = base.buildSystem(ctx)
        native = ctx.stepFunction
        ctx.bundle = replace(ctx.bundle, stepFunction=lambda *args,**kwargs:native(*args,**kwargs))
        return system
    args = options()
    args.update(store=True, exportRoot=str(tmp_path), particleOrder='morton', scheme='CompSPH')
    with pytest.raises(ValueError,match='custom RHS'):
        run(replace(base,buildSystem=build),**args)
    assert not list(tmp_path.iterdir())
