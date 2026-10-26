"""Matrix contracts and a short native PESPH endpoint regression."""
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(SCRIPTS))
import compressible_matrix as matrix
import run_compressibleMatrix as runner
import plot_compressibleMatrix as plotting


def planned(root, cases=('sod', 'gresho'), methods=('Monaghan', 'CRKSPH'), **kwargs):
    plan = matrix.make_plan(cases=cases, methods=methods, **kwargs)
    matrix.write_json(root/'manifest.json', plan)
    matrix.write_json(root/'selection.json', {})
    return plan


def result(root, row, run_id, error=.1, seconds=4., status='completed', **extra):
    directory = root/'runs'/run_id
    record = dict(runId=run_id, caseId=row['caseId'], method=row['method'], rowId=row['id'],
        inputSHA256=row['inputSHA256'], status=status, actualTime=.15, targetTime=.15,
        metrics=dict(densityVolumeWeightedL1=error, tangentialVelocityL1=error, loopWallSeconds=seconds), **extra)
    matrix.write_json(directory/'result.json', record)
    selected = matrix.read_json(root/'selection.json')
    selected[row['id']] = run_id
    matrix.write_json(root/'selection.json', selected)
    return directory/'result.json'


def test_filters_are_cartesian_and_unknown_names_fail():
    plan = matrix.make_plan()
    assert len(plan['rows']) == len(matrix.CASES)*len(matrix.METHODS) == 120
    assert len(matrix.select_rows(plan, cases=['noh'])) == 10
    assert len(matrix.select_rows(plan, methods=['MFV'])) == 12
    assert len(matrix.select_rows(plan, ['sod', 'gresho'], ['CRKSPH', 'MFM'])) == 4
    with pytest.raises(ValueError, match='Unknown'):
        matrix.select_rows(plan, ['typo'])


def test_presets_are_explicit_and_base_inputs_not_mutated():
    original = deepcopy(matrix.METHODS)
    plan = matrix.make_plan(['sod'], ['trial'], presets={'trial': {'base': 'CRKSPH',
        'solver': {'diffusionParams': {'C_q': 2.}}, 'simulation': {'cflFactor': .075}}})
    row = plan['rows'][0]
    assert row['configuration']['solver'] == {'scheme': 'CRKSPH', 'diffusionParams': {'C_q': 2.}}
    assert row['configuration']['simulation']['cflFactor'] == .075
    assert matrix.METHODS == original
    assert row['inputSHA256'] == matrix.digest({k: v for k, v in row.items() if k != 'inputSHA256'})
    with pytest.raises(ValueError, match='Unsafe'):
        matrix.method_catalog({'../bad': {'base': 'CRKSPH'}})
    with pytest.raises(ValueError, match='fixed case controls'):
        matrix.make_plan(['sod'], ['trial'], presets={'trial': {'base': 'CRKSPH', 'simulation': {'nx': 8}}})


def test_default_policies_and_smoke_are_distinct():
    full, smoke = matrix.make_plan(), matrix.make_plan(smoke=True)
    for row in full['rows']:
        spec = row['configuration']['simulation']
        assert spec['nx'] == matrix.CASES[row['caseId']]['simulation']['nx']
        assert not spec['plot'] and not spec['store'] and not spec['video']
        assert spec['nSteps'] is None
    entropy = next(r for r in full['rows'] if r['method'] == 'PESPH-A')
    assert entropy['configuration']['solver']['entropyHeating'] == 'SwiftLocal'
    assert all(r['configuration']['simulation']['nSteps'] == 3 for r in smoke['rows'])
    assert not matrix.complete('finished', .15, .15, smoke=True)
    assert not matrix.complete('finished', .1, .15)
    assert not matrix.complete('finished', .3, .15)


@pytest.mark.parametrize('actual,target,precision', [
    (.004999999888241291, .005, 'float32'),
    (.14999999105930328, .15, 'float32'),
    (.14999999999999997, .15, 'float64'),
])
def test_endpoint_uses_time_precision_without_running_zero_steps(actual, target, precision):
    state = SimpleNamespace(t=actual)
    assert matrix.snap_endpoint(state, target, precision)
    assert state.t == target


def test_endpoint_keeps_small_physical_remainders():
    state = SimpleNamespace(t=4.e-10)
    assert not matrix.snap_endpoint(state, 5.e-10, 'float32')
    assert state.t == 4.e-10


def test_independent_quality_cost_ranks_and_failures_never_win(tmp_path):
    plan = planned(tmp_path, cases=['sod'], methods=['Monaghan', 'CRKSPH', 'MFV'])
    result(tmp_path, plan['rows'][0], 'slow-accurate', error=.01, seconds=12.)
    result(tmp_path, plan['rows'][1], 'fast-inaccurate', error=.3, seconds=2.)
    result(tmp_path, plan['rows'][2], 'failed-cheap', error=0., seconds=.01, status='stopped')
    rows, _ = plotting.collect(matrix.comparison(tmp_path))
    by_method = {r['method']: r for r in rows}
    assert by_method['Monaghan']['qualityRank'] == 1
    assert by_method['CRKSPH']['runtimeRank'] == 1
    assert by_method['MFV']['qualityRank'] is None
    assert by_method['MFV']['runtimeRank'] is None


def test_selection_keeps_old_attempt_and_frozen_comparison(tmp_path):
    plan = planned(tmp_path, cases=['sod'], methods=['CRKSPH'])
    first = result(tmp_path, plan['rows'][0], 'first')
    frozen = matrix.comparison(tmp_path)
    result(tmp_path, plan['rows'][0], 'second', error=.02)
    assert matrix.read_json(first)['metrics']['densityVolumeWeightedL1'] == .1
    assert plotting.collect(frozen)[0][0]['runId'] == 'first'
    assert plotting.collect(matrix.comparison(tmp_path))[0][0]['runId'] == 'second'
    assert runner.main(['select', '--root', str(tmp_path), '--cases', 'sod',
                        '--methods', 'CRKSPH', '--run-id', 'first']) == 0
    assert plotting.collect(matrix.comparison(tmp_path))[0][0]['runId'] == 'first'


def test_changed_sources_missing_metrics_and_kh_reference_are_not_ranked(tmp_path):
    plan = planned(tmp_path, cases=['sod', 'khSmooth'], methods=['CRKSPH'])
    result(tmp_path, plan['rows'][0], 'changed', sourcesStable=False)
    path = result(tmp_path, plan['rows'][1], 'kh')
    record = matrix.read_json(path)
    record['metrics']['khAmplitude'] = .2
    matrix.write_json(path, record)
    rows, _ = plotting.collect(matrix.comparison(tmp_path))
    assert rows[0]['status'] == 'source_changed' and rows[0]['runtimeRank'] is None
    assert rows[1]['qualityRank'] is None and rows[1]['diagnosticAmplitude'] == .2
    assert rows[1]['runtimeRank'] == 1


def test_trials_merge_without_overwriting_or_mixing_precision(tmp_path):
    main, trial = tmp_path/'main', tmp_path/'trial'
    plan = planned(main, cases=['sod'], methods=['CRKSPH'])
    other = planned(trial, cases=['sod'], methods=['CRK-trial'],
        presets={'CRK-trial': {'base': 'CRKSPH', 'solver': {'diffusionParams': {'C_q': 2.}}}})
    result(main, plan['rows'][0], 'base')
    result(trial, other['rows'][0], 'candidate')
    view = matrix.comparison(main, includes=[trial])
    assert set(view['methods']) == {'CRKSPH', 'CRK-trial'}
    assert len(plotting.collect(view)[0]) == 2
    with pytest.raises(ValueError, match='Duplicate method'):
        matrix.comparison(main, includes=[main])
    other = matrix.make_plan(['sod'], ['CRK-trial'], precision='float64',
                            presets={'CRK-trial': {'base': 'CRKSPH'}})
    matrix.write_json(trial/'manifest.json', other)
    with pytest.raises(ValueError, match='precisions'):
        matrix.comparison(main, includes=[trial])


def test_partial_trial_root_leaves_unrun_cases_pending(tmp_path):
    main, trial = tmp_path/'main', tmp_path/'trial'
    planned(main, cases=['sod', 'gresho'], methods=['CRKSPH'])
    other = planned(trial, cases=['sod'], methods=['CRK-trial'],
        presets={'CRK-trial': {'base': 'CRKSPH'}})
    result(trial, other['rows'][0], 'candidate')
    view = matrix.comparison(main, cases=['sod', 'gresho'], includes=[trial])
    rows, _ = plotting.collect(view)
    assert len(rows) == 4
    pending = next(r for r in rows if r['caseId'] == 'gresho' and r['method'] == 'CRK-trial')
    assert pending['status'] == 'pending'


def test_report_has_separate_provenance_and_immutable_selection(tmp_path):
    root = tmp_path/'runs'
    plan = planned(root, cases=['sod'], methods=['CRKSPH'])
    path = result(root, plan['rows'][0], 'retained')
    matrix.write_json(path.parent/'provenance.json', dict(created='2001-02-03', deviceName='fixture',
        repositories={'warpSPH': dict(commit='abc123', commitURL='https://github.com/wi-re/warpSPH/commit/abc123', dirty=True)}))
    view = matrix.comparison(root)
    destination = tmp_path/'report'
    plotting.report(view, destination)
    assert (destination/'matrix.png').stat().st_size > 1000
    assert 'abc123' in (destination/'provenance.csv').read_text()
    assert '2001-02-03' not in (destination/'matrix.svg').read_text()
    assert matrix.read_json(destination/'comparison.json') == view
    with pytest.raises(ValueError, match='already exists'):
        plotting.report(view, destination)


def test_plan_cannot_be_overwritten_and_lock_prevents_concurrency(tmp_path):
    plan = planned(tmp_path, cases=['sod'], methods=['CRKSPH'])
    with pytest.raises(SystemExit):
        runner.main(['plan', '--root', str(tmp_path)])
    (tmp_path/'.run.lock').write_text('another controller')
    with pytest.raises(ValueError, match='another controller'):
        runner.execute(tmp_path, plan, plan['rows'])


def test_selected_input_cannot_be_substituted(tmp_path):
    plan = planned(tmp_path, cases=['sod'], methods=['CRKSPH'])
    path = result(tmp_path, plan['rows'][0], 'bad')
    record = matrix.read_json(path)
    record['inputSHA256'] = 'different'
    matrix.write_json(path, record)
    with pytest.raises(ValueError, match='mismatch'):
        plotting.collect(matrix.comparison(tmp_path))


def test_edited_plan_is_rejected():
    plan = matrix.make_plan(['sod'], ['CRKSPH'])
    plan['rows'][0]['configuration']['simulation']['nx'] = 8
    with pytest.raises(ValueError, match='edited'):
        matrix.select_rows(plan)


def test_timeout_retains_progress_and_does_not_remove_previous_attempt(tmp_path, monkeypatch):
    plan = planned(tmp_path, cases=['sod'], methods=['CRKSPH'])
    first = result(tmp_path, plan['rows'][0], 'first')
    stopped = []
    def popen(command, **kwargs):
        run_id = command[-1]
        row = matrix.read_json(tmp_path/'runs'/run_id/'input.json')
        path = result(tmp_path, row, run_id, status='running')
        record = matrix.read_json(path)
        record['actualTime'] = .04
        matrix.write_json(path, record)
        class Process:
            returncode = -15
            def wait(self, timeout):
                raise runner.subprocess.TimeoutExpired(command, timeout)
        return Process()
    monkeypatch.setattr(runner.subprocess, 'Popen', popen)
    monkeypatch.setattr(runner, 'terminate', lambda process: stopped.append(process))
    assert runner.execute(tmp_path, plan, plan['rows'], rerun=True, timeout=1) == 2
    run_id = matrix.read_json(tmp_path/'selection.json')[plan['rows'][0]['id']]
    record = matrix.read_json(tmp_path/'runs'/run_id/'result.json')
    assert record['status'] == 'timeout' and record['actualTime'] == .04
    assert len(stopped) == 1 and first.exists()


def test_resume_skips_attempts_and_rerun_appends_using_fresh_process(tmp_path, monkeypatch):
    plan = planned(tmp_path, cases=['sod'], methods=['CRKSPH'])
    first = result(tmp_path, plan['rows'][0], 'first')
    def popen(command, **kwargs):
        run_id = command[-1]
        assert kwargs['start_new_session']
        row = matrix.read_json(tmp_path/'runs'/run_id/'input.json')
        result(tmp_path, row, run_id)
        class Process:
            returncode = 0
            def wait(self, timeout):
                return 0
        return Process()
    monkeypatch.setattr(runner.subprocess, 'Popen', popen)
    assert runner.execute(tmp_path, plan, plan['rows']) == 0
    assert matrix.read_json(tmp_path/'selection.json')[plan['rows'][0]['id']] == 'first'
    assert runner.execute(tmp_path, plan, plan['rows'], rerun=True) == 0
    assert matrix.read_json(tmp_path/'selection.json')[plan['rows'][0]['id']] != 'first'
    assert first.exists()


def test_json_has_no_nonstandard_nan_or_infinity(tmp_path):
    path = tmp_path/'metrics.json'
    matrix.write_json(path, {'failed': float('nan'), 'overflow': float('inf')})
    assert matrix.read_json(path) == {'failed': None, 'overflow': None}


def test_requested_resolutions_precisions_and_native_support():
    plan = matrix.make_plan()
    expected = dict(linearWave=128, sod=800, sod2d=512, noh=256, kidder=200,
                    sedov1d=800, sedov2d=256, sedov3d=64, gresho=256,
                    hydrostatic=100, kelvinHelmholtz=256, khSmooth=256)
    for row in plan['rows']:
        cid = row['caseId']
        assert row['configuration']['simulation']['nx'] == expected[cid]
        assert row['configuration']['simulation']['precision'] == (
            'float64' if cid in ('noh', 'kidder') else 'float32')
        assert not row.get('unsupportedReason')
    assert matrix.case_precisions(plan) == plan['casePrecisions']


@pytest.mark.parametrize('actual,positive,stop,expected', [
    (.15, True, None, 'completed'),
    (.15, False, None, 'completed_invalid'),
    (.15025, True, None, 'endpoint_mismatch'),
    (.05, True, 'velocity alarm', 'stopped'),
    (.15, True, 'velocity alarm', 'stopped'),
])
def test_legacy_stop_records_distinguish_endpoint_and_admissibility(actual, positive, stop, expected):
    record = dict(status='stopped', actualTime=actual, targetTime=.15,
                  stopReason=stop, admissibility=dict(positive=positive, finite=True))
    assert matrix.classify_outcome(record) == expected


def test_completed_invalid_retains_cost_but_never_accuracy_rank(tmp_path):
    plan = planned(tmp_path, cases=['sod'], methods=['CRKSPH', 'CompSPH'])
    result(tmp_path, plan['rows'][0], 'invalid', status='stopped', seconds=2.,
           admissibility=dict(positive=False, finite=True))
    result(tmp_path, plan['rows'][1], 'overshoot', status='stopped', seconds=1.,
           admissibility=dict(positive=True, finite=True))
    path = tmp_path/'runs/overshoot/result.json'
    record = matrix.read_json(path)
    record['actualTime'] = .151
    matrix.write_json(path, record)
    rows, _ = plotting.collect(matrix.comparison(tmp_path))
    assert rows[0]['status'] == 'completed_invalid'
    assert rows[0]['qualityRank'] is None and rows[0]['runtimeSeconds'] == 2.
    assert rows[1]['status'] == 'endpoint_mismatch'
    assert rows[1]['qualityRank'] is None and rows[1]['runtimeRank'] is None


@pytest.mark.parametrize('method', ['PESPH-E', 'PESPH-A'])
def test_native_pesph_callback_cannot_bypass_endpoint_cap(tmp_path, monkeypatch, method):
    # The native runner replaces Case.timestep for PESPH. A multi-step physical
    # run catches that replacement bypassing a cap, unlike a helper-only test.
    definition = deepcopy(matrix.CASES['linearWave'])
    definition['simulation'].update(nx=24, tLimit=.037)
    monkeypatch.setitem(matrix.CASES, 'endpointRegression', definition)
    plan = planned(tmp_path, cases=['endpointRegression'], methods=[method], device='cpu')
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    for key, folder in (('WARP_CACHE_PATH', 'warp'), ('NUMBA_CACHE_DIR', 'numba'), ('MPLCONFIGDIR', 'matplotlib')):
        env.setdefault(key, str(tmp_path/'cache'/folder))
    run = subprocess.run([sys.executable, str(SCRIPTS/'run_compressibleMatrix.py'), 'run',
                          '--root', str(tmp_path)], env=env, capture_output=True, text=True, timeout=300)
    selection = matrix.read_json(tmp_path/'selection.json')
    record = matrix.read_json(tmp_path/'runs'/selection[plan['rows'][0]['id']]/'result.json')
    assert run.returncode == 0, record.get('error', run.stdout+run.stderr)
    assert record['status'] == 'completed'
    assert record['nSteps'] > 2 and record['actualTime'] == record['targetTime'] == .037
    assert record['admissibility']['positive'] and record['admissibility']['finite']
