"""Case migration coverage, structured fields and sparse legacy-run bridging."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from warpSPH.cases import importAll
from warpSPH.runner import (CaseSpec, Parameter, RunConfiguration, configurationCatalogue,
                            configurationSchema, getCase, listCases, run)
importAll()
COMPRESSIBLE = [getCase(n) for n in listCases()
                if '.cases.compressible.' in getCase(n).buildSystem.__module__]
MIGRATED = COMPRESSIBLE + [getCase('tgv-wc'), getCase('randomFlow')]


@pytest.mark.parametrize('case', MIGRATED, ids=lambda c: c.name)
def test_all_declared_parameters_have_a_complete_and_valid_contract(case):
    assert set(case.parameterDefinitions) == set(case.params)
    assert case.scheme in case.supportedSchemes
    spec = RunConfiguration(case.name).resolve(case)
    ctx = RunConfiguration(case.name).prepare(case)
    assert ctx.scheme.name == case.scheme
    schema = configurationSchema(case)
    json.dumps(schema, allow_nan=False)
    jsonschema = pytest.importorskip('jsonschema')
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(RunConfiguration(case.name).toDict(), schema)
    assert spec.nx == case.defaults['nx']
    for name, value in case.params.items():
        assert case.parameterDefinitions[name].description
        assert name in schema['properties']['params']['properties']


def test_catalogue_reports_missing_or_stale_metadata_without_initializing_a_context():
    catalogue = configurationCatalogue()
    ready = {c['id'] for c in catalogue['cases'] if c['status'] == 'ready'}
    assert ready >= {c.name for c in MIGRATED}
    broken = replace(getCase('gresho'), params={**getCase('gresho').params, 'newParameter': 1})
    status = configurationCatalogue([broken])['cases'][0]
    assert status['status'] == 'unmigrated'
    assert status['missingParameters'] == ['newParameter']
    with pytest.raises(ValueError, match='migration'):
        RunConfiguration('gresho').resolve(broken)
    stale = replace(getCase('gresho'), parameterDefinitions={**getCase('gresho').parameterDefinitions,
                    'obsolete': Parameter('number', 'Stale metadata')})
    assert configurationCatalogue([stale])['cases'][0]['staleParameters'] == ['obsolete']


@pytest.mark.parametrize('bad,path', [
    ([{'begin': 0, 'end': 1, 'pressure': -1, 'density': 1}], r'regions\[0\].pressure'),
    ([{'begin': 0, 'end': 1, 'presure': 1, 'density': 1}], r'regions\[0\]'),
    ([{'begin': 1, 'end': .5, 'pressure': 1, 'density': 1}], r'regions\[0\].end'),
    ([], r'params.regions'),
    ([{'begin': .1, 'end': 1, 'pressure': 1, 'density': 1}], r'regions\[0\].begin'),
])
def test_region_lists_validate_the_actual_nested_parameter_path(bad, path):
    with pytest.raises(ValueError, match=path):
        RunConfiguration('woodwardColella', params={'regions': bad}).resolve()
    schema = configurationSchema(getCase('woodwardColella'))
    assert schema['properties']['params']['properties']['regions']['items']['additionalProperties'] is False


def test_vectors_have_finite_values_and_exact_dimensions():
    for offset in ([0], [0, 0, 0], [0, 'x']):
        with pytest.raises(ValueError, match='obstacleOffset'):
            RunConfiguration('randomFlow', params={'obstacle': True, 'obstacleOffset': offset}).resolve()
    assert RunConfiguration('randomFlow', params={'obstacle': True, 'obstacleOffset': [.1, -.1]}).resolve().params['obstacleOffset'] == [.1, -.1]


@pytest.mark.parametrize('dim', [1, 2, 3])
def test_kidder_exposes_derived_geometry_and_gamma_instead_of_overwriting_user_values(dim):
    c = RunConfiguration('kidder', simulation={'dim': dim})
    spec = c.resolve()
    assert spec.params['nu'] == dim
    assert spec.params['gamma'] == 1 + 2 / dim
    assert c.prepare().schemeConfig.gamma == 1 + 2 / dim
    with pytest.raises(ValueError, match='params.gamma: derived'):
        RunConfiguration('kidder', simulation={'dim': dim}, params={'gamma': 1.1}).resolve()
    with pytest.raises(ValueError, match='params.nu: derived'):
        RunConfiguration('kidder', params={'nu': 2}).resolve()


@pytest.mark.parametrize('case,key', [('hydrostatic', 'E0'), ('linearWave', 'E0'), ('triplePoint', 'aspect'), ('triplePoint', 'band')])
def test_unused_legacy_fields_are_documented_and_cannot_pretend_to_change_physics(case, key):
    c = getCase(case)
    with pytest.raises(ValueError, match=f'params.{key}: read-only'):
        RunConfiguration(case, params={key: c.params[key] + 1}).resolve(c)
    schema = configurationSchema(c)['properties']['params']['properties'][key]
    assert schema['readOnly'] and schema['deprecated']


def test_meshless_constraints_are_explicit():
    with pytest.raises(ValueError, match='periodic'):
        RunConfiguration('gresho', simulation={'periodic': False}, solver={'scheme': 'MFM'}).resolve()
    with pytest.raises(ValueError, match='expected one of'):
        RunConfiguration('rayleighTaylor', solver={'scheme': 'MFM'}).resolve()
    assert RunConfiguration('yee', solver={'scheme': 'MFM'}).resolve().params['bufferRings'] == 0
    with pytest.raises(ValueError, match='bufferRings: derived'):
        RunConfiguration('yee', params={'bufferRings': 2}, solver={'scheme': 'MFM'}).resolve()


def test_derived_values_validate_inputs_before_arithmetic():
    with pytest.raises(ValueError, match='simulation.dim'):
        RunConfiguration('kidder', simulation={'dim': 'two'}).resolve()
    with pytest.raises(ValueError, match='params.gamma'):
        RunConfiguration('noh', params={'gamma': 'five'}).resolve()


def test_sparse_warpsphcases_style_overrides_use_the_same_resolver():
    case = getCase('gresho')
    overrides = {'scheme': 'CRKSPH', 'nx': 64, 'params': {'cuspWidth': .02},
                 'schemeOverrides': {'diffusionParams': {'K': .25}}}
    config = RunConfiguration.fromOverrides(case, overrides)
    assert config.variant == 'default'  # run labels such as "demo" are not physical variants
    ctx = config.prepare(case)
    assert ctx.spec.nx == 64
    assert ctx.param('cuspWidth') == .02
    assert float(ctx.schemeConfig.diffusionParams.K) == .25
    assert overrides['params'] == {'cuspWidth': .02}
    with pytest.raises(ValueError, match='runner overrides'):
        RunConfiguration.fromOverrides(case, {'nxx': 32})


@pytest.mark.parametrize('name,nx,params', [
    ('gresho', 24, {}), ('kelvinHelmholtz', 24, {}), ('rayleighTaylor', 32, {}),
    ('woodwardColella', 80, {}), ('kidder', 64, {}), ('randomFlow', 24, {'seed': 123}),
])
def test_two_step_configured_runs_match_legacy_python(name, nx, params):
    options = {'nx': nx, 'nSteps': 2, 'quiet': True, 'plot': False, 'store': False,
               'progress': False, 'device': 'cuda'}
    config = RunConfiguration(name, simulation=options, params=params)
    configured = config.run()
    legacy = run(getCase(name), **options, params=params)
    assert not configured.diverged and configured.nSteps == 2
    for field in ('positions', 'velocities', 'densities'):
        torch.testing.assert_close(getattr(configured.state.state, field), getattr(legacy.state.state, field))


def test_catalogue_cli_does_not_start_a_simulation(tmp_path):
    target = tmp_path / 'catalogue.json'
    result = subprocess.run([sys.executable, '-m', 'warpSPHRun', '--configurationCatalogue', str(target)],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads(target.read_text())
    assert data['schemaVersion'] == 1
    assert next(c for c in data['cases'] if c['id'] == 'gresho')['status'] == 'ready'


def test_catalogue_detects_invalid_defaults_too():
    broken = replace(getCase('gresho'), params={**getCase('gresho').params, 'cuspWidth': -1})
    entry = configurationCatalogue([broken])['cases'][0]
    assert entry['status'] == 'invalid'
    assert 'cuspWidth' in entry['error']


def test_legacy_override_bridge_does_not_accept_a_hidden_solver_selector():
    with pytest.raises(ValueError, match='put scheme at the top level'):
        RunConfiguration.fromOverrides(getCase('gresho'), {'schemeOverrides': {'scheme': 'MFM'}})


def test_derived_reference_values_allow_roundoff_equivalent_input():
    assert RunConfiguration('noh', params={'v_s': 1 / 3}).resolve().params['v_s'] == pytest.approx(1 / 3)


@pytest.mark.parametrize('case,flags', [('kidder', []), ('yee', ['--scheme', 'MFM'])])
def test_legacy_check_cli_resolves_derived_defaults(case, flags):
    result = subprocess.run([sys.executable, '-m', 'warpSPHRun', case, '--checkConfig', *flags],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'Configuration valid' in result.stdout


def test_artificial_compressibility_has_its_own_controls_and_physical_viscosity():
    config = RunConfiguration('tgv-wc', params={'nu': .02}, solver={
        'scheme': 'artificialCompressible', 'acParams': {'epsilonV': -5, 'maxPseudoIterations': 40}})
    ctx = config.prepare()
    assert ctx.schemeConfig.acParams.nu == .02
    assert ctx.schemeConfig.acParams.epsilonV == -5
    assert ctx.config.integrationScheme.name == 'forwardEuler'
    schema = configurationSchema(getCase('tgv-wc'), 'artificialCompressible')
    assert 'diffusionParams' not in schema['properties']['solver']['properties']
    assert 'acParams' in schema['properties']['solver']['properties']
    with pytest.raises(ValueError, match='not used by artificialCompressible'):
        RunConfiguration('tgv-wc', params={'initialPressure': True}, solver={'scheme': 'artificialCompressible'}).resolve()
    with pytest.raises(ValueError, match='not used by artificialCompressible'):
        RunConfiguration('randomFlow', params={'alpha': .1}, solver={'scheme': 'artificialCompressible'}).resolve()


def test_ac_iteration_limits_are_checked_together():
    with pytest.raises(ValueError, match='minPseudoIterations'):
        RunConfiguration('tgv-wc', solver={'scheme': 'artificialCompressible', 'acParams': {
            'minPseudoIterations': 50, 'maxPseudoIterations': 10}}).prepare()


def test_ac_two_step_configuration_matches_legacy_python():
    options = {'nx': 16, 'nSteps': 2, 'device': 'cuda', 'quiet': True, 'store': False, 'plot': False}
    config = RunConfiguration('tgv-wc', simulation=options, params={'shuffleIters': 0},
                              solver={'scheme': 'artificialCompressible'})
    actual = config.run()
    legacy = run(getCase('tgv-wc'), **options, params={'shuffleIters': 0}, scheme='artificialCompressible')
    assert not actual.diverged and actual.nSteps == 2
    torch.testing.assert_close(actual.state.state.velocities, legacy.state.state.velocities)


@pytest.mark.parametrize('name', ['linearWave', 'sod2d', 'sod3d'])
def test_meshless_initial_sampling_accepts_its_consumed_support_control(name):
    spec = RunConfiguration(name, params={'adaptiveSupportScheme': 'Owen'},
                            solver={'scheme': 'MFM'}).resolve(getCase(name))
    assert spec.params['adaptiveSupportScheme'] == 'Owen'
