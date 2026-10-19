"""The versioned contract: parsing, setup precedence, discovery and numerical parity."""
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from warpSPH.runner import (CaseSpec, RunConfiguration, configurationSchema,
                            parameterReference, resolvedConfiguration, run)
from warpSPH.cases.compressible.sod import sodCase
from warpSPH.cases.periodic.tgvWeaklyCompressible import tgvWeaklyCompressibleCase as tgvCase


@pytest.mark.parametrize('case', [sodCase, tgvCase])
def test_default_resolution_preserves_legacy_case_defaults(case):
    actual = RunConfiguration(case.name).resolve(case)
    expected = CaseSpec(caseName=case.name, scheme=case.scheme, params=dict(case.params)).merged(**case.defaults)
    assert replace(actual, configuration=None) == expected
    assert set(case.params) == set(case.parameterDefinitions)


@pytest.mark.parametrize('suffix', ['.json', '.yaml'])
def test_document_roundtrip_variants_and_user_precedence(tmp_path, suffix):
    config = RunConfiguration('sod', variant='classic', params={'right_pressure': 0.2},
                              simulation={'nx': 80}, solver={'scheme': 'CRKSPH'})
    assert RunConfiguration.load(config.save(tmp_path / ('run' + suffix))) == config
    spec = config.resolve(sodCase)
    assert spec.params['gamma'] == 1.4
    assert spec.params['right_pressure'] == 0.2
    assert spec.params['right_rho'] == 0.125
    assert spec.kernel == 'B7'
    assert sodCase.params['right_pressure'] == 0.1795


@pytest.mark.parametrize('patch,path', [
    ({'params': {'left_ro': 1}}, 'params: unknown'),
    ({'params': {'left_rho': -1}}, 'params.left_rho'),
    ({'params': {'gamma': 1}}, 'params.gamma'),
    ({'simulation': {'nx': True}}, 'simulation.nx'),
    ({'simulation': {'L': 3}}, 'simulation.L'),
    ({'simulation': {'dim': 2}}, 'simulation.dim'),
    ({'simulation': {'kernel': 'no-such-kernel'}}, 'simulation.kernel'),
    ({'params': {'samplingRatio': 1000}}, 'params.samplingRatio'),
    ({'solver': {'diffusionParams': {'K': -0.1}}}, 'solver.diffusionParams.K'),
    ({'solver': {'diffusionParams': {'KK': 1}}}, 'solver: unknown'),
    ({'solver': {'scheme': 'MFM', 'diffusionParams': {'K': 1}}}, 'solver: unknown'),
    ({'solver': {'scheme': 'CRKSPH', 'diffusionParams': {'velocityPairPolicy': 'Limited'}}}, 'solver.diffusionParams.velocityPairPolicy'),
    ({'variant': 'misspelled'}, 'configuration.variant'),
])
def test_bad_values_fail_with_a_parameter_path(patch, path):
    with pytest.raises(ValueError, match=path):
        RunConfiguration.fromDict({'schemaVersion': 1, 'case': 'sod', **patch}).resolve(sodCase)


@pytest.mark.parametrize('data', [
    {'schemaVersion': True, 'case': 'sod'},
    {'schemaVersion': 2, 'case': 'sod'},
    {'schemaVersion': 1, 'case': 'sod', 'simulation': []},
    {'schemaVersion': 1, 'case': 'sod', 'params': {'gamma': float('nan')}},
    {'schemaVersion': 1, 'case': 'sod', 'unexpected': True},
])
def test_structure_is_strict_before_runtime_setup(data):
    with pytest.raises(ValueError):
        RunConfiguration.fromDict(data)


def test_solver_overrides_win_after_case_configuration_without_sampling():
    def forbidden(ctx):
        pytest.fail('checking a configuration must not sample particles')
    case = replace(sodCase, buildSystem=forbidden)
    config = RunConfiguration('sod', params={'alpha_min': 0.2},
        solver={'diffusionParams': {'K': 0.3}, 'viscositySwitchParams': {'alpha_min': 0.1, 'alpha_max': 0.8}})
    ctx = config.prepare(case)
    assert float(ctx.schemeConfig.diffusionParams.K) == pytest.approx(0.3)
    assert ctx.schemeConfig.viscositySwitchParams.alpha_min == 0.1
    snapshot = resolvedConfiguration(ctx)
    assert snapshot['requested'] == config.toDict()
    assert snapshot['solverConfig']['diffusionParams']['K'] == pytest.approx(0.3)
    assert snapshot['stage'] == 'setup (no particle sampling)'
    json.dumps(snapshot, allow_nan=False)
    with pytest.raises(ValueError, match='alpha_min'):
        RunConfiguration('sod', solver={'viscositySwitchParams': {'alpha_min': 3, 'alpha_max': 2}}).prepare(case)


@pytest.mark.parametrize('scheme', sodCase.supportedSchemes)
def test_sod_solver_specific_contracts_can_prepare(scheme):
    ctx = RunConfiguration('sod', solver={'scheme': scheme}).prepare(sodCase)
    assert ctx.scheme.name == scheme
    schema = configurationSchema(sodCase, scheme)
    if scheme in ('MFM', 'MFV'):
        assert 'diffusionParams' not in schema['properties']['solver']['properties']
        assert 'temporalScheme' in schema['properties']['solver']['properties']


def test_wc_uses_its_own_diffusion_and_shared_parameter_definitions():
    config = RunConfiguration('tgv-wc', params={'alpha': 0.02, 'inviscid': True},
                              solver={'diffusionParams': {'densityDelta': 0.05}})
    ctx = config.prepare(tgvCase)
    assert ctx.schemeConfig.diffusionParams.densityDelta == 0.05
    assert ctx.schemeConfig.diffusionParams.inviscidAlpha == 0.02
    with pytest.raises(ValueError, match='params.alpha'):
        RunConfiguration('tgv-wc', params={'alpha': 0.1}).resolve(tgvCase)
    with pytest.raises(ValueError, match='referenceVelocity'):
        RunConfiguration('tgv-wc', params={'machTarget': 0.1}).resolve(tgvCase)


@pytest.mark.parametrize('case', [sodCase, tgvCase])
def test_discovery_and_documentation_use_the_same_runtime_metadata(case):
    schema = configurationSchema(case)
    assert schema['properties']['simulation']['properties']['nx']['default'] == case.defaults['nx']
    assert schema['properties']['params']['properties']['jitter']['description']
    assert 'params.jitter' in parameterReference(case)
    jsonschema = pytest.importorskip('jsonschema')
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(RunConfiguration(case.name).toDict(), schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(RunConfiguration(case.name, params={'jitter': -1}).toDict(), schema)


@pytest.mark.parametrize('case,params,nx', [
    (sodCase, {}, 80),
    (tgvCase, {'shuffleIters': 0}, 16),
])
def test_two_steps_from_file_match_legacy_python(case, params, nx, tmp_path):
    document = RunConfiguration(case.name, simulation={'nx': nx, 'nSteps': 2, 'quiet': True,
        'progress': False, 'store': False, 'plot': False, 'device': 'cuda'}, params=params)
    path = document.save(tmp_path / 'run.yaml')
    configured = RunConfiguration.load(path).run(case)
    legacy = run(case, nx=nx, nSteps=2, quiet=True, progress=False, store=False,
                 plot=False, device='cuda', params=params)
    assert configured.nSteps == legacy.nSteps == 2
    for name in ('positions', 'velocities', 'densities'):
        torch.testing.assert_close(getattr(configured.state.state, name), getattr(legacy.state.state, name))
    assert resolvedConfiguration(configured.ctx)['stage'] == 'initialized'


def test_cli_resolves_file_case_and_precision_before_import(tmp_path):
    document = RunConfiguration('sod', simulation={'precision': 'float64', 'nx': 40, 'quiet': True})
    path = document.save(tmp_path / 'run.yaml')
    output = tmp_path / 'resolved.json'
    result = subprocess.run([sys.executable, '-m', 'warpSPHRun', '--config', str(path),
        '--checkConfig', '--nx', '80', '--saveResolvedConfig', str(output)],
        env=os.environ.copy(), capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    actual = json.loads(output.read_text())
    assert actual['spec']['precision'] == 'float64'
    assert actual['spec']['nx'] == 80
    assert actual['spec']['params']['left_rho'] == 1.0
    assert actual['stage'] == 'setup (no particle sampling)'


def test_nullable_cli_parameter_types_come_from_descriptors():
    from warpSPH.runner.caseSpec import buildArgumentParser
    parser = buildArgumentParser(caseParams=tgvCase.params,
                                 parameterDefinitions=tgvCase.parameterDefinitions)
    args = parser.parse_args(['--no-shifting'])
    assert args.shifting is False
    parser = buildArgumentParser(caseParams=sodCase.params,
                                 parameterDefinitions=sodCase.parameterDefinitions)
    assert parser.parse_args(['--viscositySwitch', 'Rosswog2020']).viscositySwitch == 'Rosswog2020'


@pytest.mark.parametrize('suffix,content', [
    ('.json', '{"schemaVersion":1,"case":"sod","params":{"gamma":1.4,"gamma":1.6}}'),
    ('.yaml', 'schemaVersion: 1\ncase: sod\nparams:\n  gamma: 1.4\n  gamma: 1.6\n'),
])
def test_duplicate_file_keys_are_rejected(tmp_path, suffix, content):
    target = tmp_path / ('duplicate' + suffix)
    target.write_text(content)
    with pytest.raises(ValueError, match='duplicate configuration key'):
        RunConfiguration.load(target)


def test_migrated_case_rejects_overrides_of_unused_meshless_settings():
    with pytest.raises(ValueError, match='not used by MFM'):
        RunConfiguration('sod', params={'smoothIC': True}, solver={'scheme': 'MFM'}).resolve(sodCase)


def test_recorded_configuration_writes_initialized_manifest(tmp_path):
    config = RunConfiguration('sod', simulation={'nx': 80, 'nSteps': 1, 'quiet': True,
        'store': True, 'storeMode': 'trajectory', 'plot': False, 'device': 'cuda',
        'exportRoot': str(tmp_path)})
    result = config.run(sodCase)
    manifest = json.loads((Path(result.ctx.exportPath) / 'resolvedConfiguration.json').read_text())
    assert manifest['requested'] == config.toDict()
    assert manifest['stage'] == 'initialized'
    assert manifest['spec']['nx'] == 80
    assert manifest['versions']['warpSPH']['commit']
    assert (Path(result.ctx.exportPath) / 'trajectory.h5').is_file()


def test_schema_defaults_include_the_configured_solver_values():
    schema = configurationSchema(sodCase, 'CRKSPH')
    solver = schema['properties']['solver']['properties']
    assert solver['diffusionParams']['properties']['C_q']['default'] == 1.0
    assert solver['diffusionParams']['properties']['velocityPairPolicy']['default'] == 'Raw'
    assert solver['viscositySwitchParams']['properties']['scheme']['default'] == 'NoneSwitch'


def test_yaml_anchors_keep_explicit_override_precedence(tmp_path):
    path = tmp_path / 'anchors.yaml'
    path.write_text('schemaVersion: 1\ncase: sod\nparams: &base\n  gamma: 1.4\n'
                    '  left_rho: 1.0\n')
    assert RunConfiguration.load(path).resolve(sodCase).params['gamma'] == 1.4
    # A common legacy file pattern: YAML merging itself has well-defined precedence.
    from warpSPHConfig import readMapping
    path.write_text('defaults: &base {nx: 80, gamma: 1.4}\n'
                    'run: {<<: *base, nx: 100}\n')
    assert readMapping(path)['run'] == {'nx': 100, 'gamma': 1.4}
