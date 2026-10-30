"""The external reference export preserves fields, identity and physical clocks."""
import struct
import h5py
import numpy as np
import pytest

from scripts.acsph_reference.export_viewer import export_run, reference_snapshot_steps


def write_snapshot(path, t, pressure=12.):
    x = np.array([[1.609, .003], [1.609, .015], [1.611, 0.]])
    n = len(x)
    with path.open('wb') as f:
        f.write(struct.pack('<id', n, t))
        for array in (x, np.full(n, pressure), np.full(n, 1000.), np.ones((n, 2))*t):
            f.write(array.astype('<f8').tobytes())
        f.write(np.array([1, 1, 3], dtype='<i4').tobytes())
        for array in (np.zeros(n), np.full(n, .00075**2), np.zeros(n)):
            f.write(array.astype('<f8').tobytes())


def test_lossless_export_and_duplicate_terminal_clock(tmp_path):
    run = tmp_path/'native'; (run/'snapshots').mkdir(parents=True)
    for i, t in enumerate((0., .01, .01)):
        write_snapshot(run/'snapshots'/f'{i:06d}.bin', t)
    output = tmp_path/'trajectory.h5'
    receipt = export_run(run, output, 'test')
    assert receipt['frames'] == 2
    assert receipt['skipped_duplicate_times'] == 1
    with h5py.File(output, 'r') as h:
        np.testing.assert_array_equal(h['time'], [0., .01])
        np.testing.assert_array_equal(h['particles/kinds'], [0, 0, 1])
        np.testing.assert_array_equal(h['particles/UIDs'], [0, 1, 2])
        np.testing.assert_array_equal(h['frames/pressures'][1], [12., 12., 12.])
        np.testing.assert_allclose(h['probes/pressure'][0], [12., 12., 0., 0.])
        assert h['frames/positions'].dtype == np.dtype('float64')
        assert h['frames/positions'].chunks[0] == 1
        assert h.attrs['complete']
    assert reference_snapshot_steps(run, [{'t': .005}, {'t': .01}]) == {0, 2}


def test_distinct_states_at_duplicate_clock_rejected(tmp_path):
    (tmp_path/'snapshots').mkdir()
    write_snapshot(tmp_path/'snapshots/000000.bin', 0.)
    write_snapshot(tmp_path/'snapshots/000001.bin', 0., pressure=13.)
    with pytest.raises(ValueError, match='Distinct physical states'):
        export_run(tmp_path, tmp_path/'bad.h5', 'test')


def test_reference_clock_must_match_a_real_step(tmp_path):
    (tmp_path/'snapshots').mkdir()
    write_snapshot(tmp_path/'snapshots/000000.bin', .012)
    with pytest.raises(ValueError, match='not a recorded real step'):
        reference_snapshot_steps(tmp_path, [{'t': .01}])
