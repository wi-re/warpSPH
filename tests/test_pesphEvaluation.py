"""The evaluation matrix must describe runnable configurations, not silent skips."""

import importlib.util
from pathlib import Path

import pytest

from warpSPH.runner import RunConfiguration

path = Path(__file__).resolve().parents[1]/'scripts/evaluate_pesph.py'
spec = importlib.util.spec_from_file_location('pesph_evaluation', path)
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


@pytest.mark.parametrize('case', evaluation.cases(), ids=lambda c: c['name'])
@pytest.mark.parametrize('scheme', evaluation.SCHEMES)
def test_matrix_configuration_is_accepted(case, scheme):
    solver = {'scheme': 'CRKSPH' if scheme == 'CRKSPH' else 'PESPH'}
    if scheme != 'CRKSPH':
        solver.update(variant='PressureEntropy' if scheme == 'PESPH-A' else 'PressureEnergy',
                      dissipation=True, entropyHeating='SwiftLocal')
    RunConfiguration(case['case'], simulation=case['simulation'], params=case['params'], solver=solver).prepare()


@pytest.mark.parametrize('case', evaluation.cases(), ids=lambda c: c['name'])
def test_entropy_case_startup(tmp_path, case):
    # A tight two-step adapter check, not a physical evaluation: no video.
    simulation = dict(case['simulation'], device='cpu', nx=24, nSteps=2,
                      plot=False, video=False, progress=False, quiet=True,
                      exportRoot=str(tmp_path), cflFactor=.1)
    if case['case'] == 'kidder':
        simulation['nx'] = 48
    if case['case'] == 'blob':
        simulation['nx'] = 64
    if simulation.get('dim') == 3:
        simulation['nx'] = 10
    result = RunConfiguration(case['case'], simulation=simulation, params=case['params'],
        solver=dict(scheme='PESPH', variant='PressureEntropy', dissipation=False,
                    entropyHeating='SwiftLocal')).run()
    assert not result.diverged
