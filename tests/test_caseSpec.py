"""CaseSpec serialisation and CLI precedence. No GPU, no simulation."""

import json

import pytest

from warpSPH.runner.caseSpec import CaseSpec, buildArgumentParser, specFromArgs


def test_roundTripsThroughJson(tmp_path):
    spec = CaseSpec(caseName='x', nx=64, dim=1, tLimit=0.25, dt=None,
                    params={'gamma': 1.4, 'smoothIC': True})
    path = spec.save(str(tmp_path / 'spec.json'))
    assert CaseSpec.load(path) == spec


def test_roundTripsThroughYaml(tmp_path):
    pytest.importorskip('yaml')
    spec = CaseSpec(caseName='x', nx=64, params={'gamma': 1.4})
    path = spec.save(str(tmp_path / 'spec.yaml'))
    assert CaseSpec.load(path) == spec


def test_unknownFieldRaisesUnlessRoutedToParams():
    with pytest.raises(ValueError, match='Unknown CaseSpec fields'):
        CaseSpec.fromDict({'nx': 32, 'gamma': 1.4})
    assert CaseSpec.fromDict({'nx': 32, 'gamma': 1.4}, strict=False).param('gamma') == 1.4


def test_cliOverridesConfigFileWhichOverridesCaseDefaults(tmp_path):
    configPath = tmp_path / 'spec.json'
    configPath.write_text(json.dumps({'nx': 64, 'tLimit': 0.5}))

    parser = buildArgumentParser(caseParams={'gamma': 5 / 3})
    args = parser.parse_args(['--config', str(configPath), '--nx', '128', '--gamma', '1.4'])
    spec = specFromArgs(args, caseParams={'gamma': 5 / 3},
                        defaults={'nx': 800, 'tLimit': 0.15, 'kernel': 'B7'})

    assert spec.nx == 128          # CLI beats the config file
    assert spec.tLimit == 0.5      # config file beats the case default
    assert spec.kernel == 'B7'     # case default survives, being unmentioned elsewhere
    assert spec.param('gamma') == 1.4


def test_booleansAreOverridableInBothDirections(tmp_path):
    configPath = tmp_path / 'spec.json'
    configPath.write_text(json.dumps({'plot': True, 'adaptiveDt': True}))

    parser = buildArgumentParser()
    spec = specFromArgs(parser.parse_args(['--config', str(configPath), '--no-plot']))

    assert spec.plot is False       # a config file's `true` stays overridable
    assert spec.adaptiveDt is True  # and an untouched flag is left alone


def test_optionalBooleansAreBooleanFlags():
    """`progress: Optional[bool] = None` gets `--progress/--no-progress`, not a
    float-typed flag (it used to reject `--progress`)."""
    from warpSPH.runner.caseSpec import buildArgumentParser, specFromArgs
    parser = buildArgumentParser()
    assert specFromArgs(parser.parse_args(['--progress'])).progress is True
    assert specFromArgs(parser.parse_args(['--no-progress'])).progress is False
    assert specFromArgs(parser.parse_args([])).progress is None


def test_mergedKeepsExistingParams():
    spec = CaseSpec(params={'a': 1, 'b': 2}).merged(nx=16, params={'b': 3})
    assert spec.nx == 16
    assert spec.params == {'a': 1, 'b': 3}
