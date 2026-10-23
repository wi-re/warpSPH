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


def test_new_matrix_uses_calibrated_wide_support_and_adaptive_projection_limits():
    plan = harness.make_plan()
    assert harness.DEFAULT_ROOT.parts[-2] == '.tmp'
    for row in plan['rows']:
        assert row['spec']['n_h'] == 4.0
        if row['scheme'] in harness.PROJECTION:
            assert row['spec']['calibrateNormalization']
            assert row['spec'].get('adaptiveDt', True)


def test_projection_rerun_enables_band_viscosity_only_for_periodic_controls():
    plan = harness.make_plan(harness.PROJECTION)
    assert len(plan['rows']) == 75 and plan['version'] == 5
    for row in plan['rows']:
        if row['scheme'] == 'band2018pb':
            assert bool(row['scheme_settings'].get('bandViscosity')) == (row['boundary'] == 'periodic')
        if row['test'] == 'englishWedge':
            assert row['params']['unifyAttachedObstacles']


def test_source_archive_does_not_resolve_user_or_group_names(tmp_path, monkeypatch):
    import io
    import tarfile
    def forbidden(*args):
        raise AssertionError('Source archive attempted a network-backed account lookup')
    monkeypatch.setattr(tarfile.pwd, 'getpwuid', forbidden)
    monkeypatch.setattr(tarfile.grp, 'getgrgid', forbidden)
    source = tmp_path / 'source.py'
    source.write_bytes(b'physical_source = 1\n')
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w') as archive:
        harness.addSourceFile(archive, source, Path('src/source.py'))
    buffer.seek(0)
    with tarfile.open(fileobj=buffer, mode='r') as archive:
        member = archive.getmember('src/source.py')
        assert member.uid == source.stat().st_uid and member.gid == source.stat().st_gid
        assert archive.extractfile(member).read() == source.read_bytes()
