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
