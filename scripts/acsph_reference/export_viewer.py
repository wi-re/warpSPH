#!/usr/bin/env python3
"""Losslessly package fixed-row Lobovsky snapshots for warpSPHRemote.

No simulator imports or coordinate wrapping. Constant density is physical rho,
not a reconstructed sampling-density diagnostic. UIDs are original output-row
indices: both comparison writers emit unsorted, fixed-membership physical rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct

import h5py
import numpy as np

try:
    from .render import read_snapshot, metrics
except ImportError:
    from render import read_snapshot, metrics


PROBES = np.array([[1.61, y] for y in (.003, .015, .030, .080)])


def snapshot_clock(path):
    with Path(path).open('rb') as f:
        header = f.read(12)
    if len(header) != 12:
        raise ValueError(f'Truncated snapshot header: {path}')
    return struct.unpack('<id', header)[1]


def reference_snapshot_steps(run, rows):
    """Find physical-step indices for the native output clocks, including final."""
    times = np.array([0.] + [r['t'] for r in rows])
    indices = set()
    for path in sorted((Path(run) / 'snapshots').glob('*.bin')):
        t = snapshot_clock(path)
        index = int(np.argmin(np.abs(times - t)))
        if abs(times[index] - t) > 1e-11:
            raise ValueError(f'Snapshot clock is not a recorded real step: {path}')
        indices.add(index)
    return indices


def pressure_probes(data, support=.003):
    """Fluid-only normalized Wendland C2, matching the native probe weights.

    Evaluated on stored, post-shift physical coordinates in BOTH recordings;
    native probe.plt uses an earlier RK field and remains separate evidence.
    An unsupported/dry probe is explicitly zero, with a separate wet mask.
    """
    fluid = data['bc'] == 1
    x, p = data['x'][fluid], data['p'][fluid]
    values, wet = [], []
    for point in PROBES:
        q = np.linalg.norm(x - point, axis=1) / support
        w = np.maximum(1 - q, 0)**4 * (1 + 4*q)
        total = w.sum()
        wet.append(bool(total > 0))
        values.append(float(np.dot(w, p) / total) if total > 0 else 0.)
    return np.array(values), np.array(wet, dtype=np.uint8)


def export_run(run, output, label, support=.003):
    run, output = Path(run).resolve(), Path(output).resolve()
    paths = sorted((run / 'snapshots').glob('*.bin'))
    if not paths:
        raise ValueError(f'No snapshots in {run}')
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + '.partial')
    first = read_snapshot(paths[0])
    n = len(first['x'])
    if not np.isin(first['bc'], [1, 3]).all():
        raise ValueError('Exporter is scoped to fixed fluid/static-wall Lobovsky rows')
    mass = first['rho'] * first['volume']
    kept, previous = [], None
    # Native terminal output can duplicate the final scheduled snapshot.
    for path in paths:
        t = snapshot_clock(path)
        if not np.isfinite(t) or (previous is not None and t < previous):
            raise ValueError(f'Invalid snapshot clock: {path}')
        if previous is not None and t == previous:
            a, b = read_snapshot(kept[-1]), read_snapshot(path)
            if not all(np.array_equal(a[k], b[k]) for k in ('x', 'v', 'p', 'rho', 'volume', 'bc')):
                raise ValueError(f'Distinct physical states at the same clock: {path}')
            continue
        kept.append(path)
        previous = t
    count = len(kept)
    receipts, stats, probes, wet = [], [], [], []
    with h5py.File(partial, 'x') as h:
        h.attrs['description'] = label
        h.attrs['complete'] = False
        h.attrs['source'] = str(run)
        h.attrs['identityProvenance'] = 'Original unsorted output row; fixed membership, no compaction'
        h.attrs['densityMeaning'] = 'Physical constant density; not kernel sampling density'
        h.attrs['units'] = 'SI, two-dimensional per unit breadth'
        h['domain/min'] = [0., 0.]
        h['domain/max'] = [1.61, 1.4]
        h['domain/periodic'] = np.zeros(2, dtype=np.uint8)
        h['particles/UIDs'] = np.arange(n, dtype=np.int64)
        h['particles/kinds'] = np.where(first['bc'] == 1, 0, 1).astype(np.int32)
        h['particles/sourceMarkers'] = first['bc']
        h['particles/masses'] = mass
        h['particles/supports'] = np.full(n, support)
        names = {'positions': 'x', 'velocities': 'v', 'pressures': 'p', 'densities': 'rho'}
        datasets = {}
        for name, key in names.items():
            tail = first[key].shape
            datasets[name] = h.create_dataset('frames/' + name, shape=(count,) + tail,
                dtype='<f8', chunks=(1, min(n, 8192)) + tail[1:],
                compression='gzip', compression_opts=1, shuffle=True)
        times = h.create_dataset('time', shape=(count,), dtype='<f8')
        for i, path in enumerate(kept):
            data = read_snapshot(path)
            if not np.array_equal(data['bc'], first['bc']):
                raise ValueError(f'Particle membership changed: {path}')
            np.testing.assert_allclose(data['rho'] * data['volume'], mass, rtol=1e-12, atol=1e-15)
            if not all(np.isfinite(data[k]).all() for k in names.values()):
                raise ValueError(f'Nonfinite field: {path}')
            for name, key in names.items():
                datasets[name][i] = data[key]
            times[i] = data['t']
            stats.append(metrics(data))
            values, mask = pressure_probes(data, support)
            probes.append(values); wet.append(mask)
            with path.open('rb') as f:
                digest = hashlib.file_digest(f, 'sha256').hexdigest()
            receipts.append({'snapshot': path.name, 'time': data['t'], 'sha256': digest})
            if i % 20 == 0 or i == count - 1:
                print(f'VIEWER EXPORT {i+1}/{count} t={data["t"]:.6g} {output.name}', flush=True)
        h['diagnostics/kineticEnergy'] = [s['ke'] for s in stats]
        h['diagnostics/maxSpeed'] = [s['vmax'] for s in stats]
        h['diagnostics/meanDensity'] = np.full(count, first['rho'][first['bc'] == 1].mean())
        h['diagnostics/particles'] = np.array([s['fluid'] for s in stats], dtype=np.int64)
        h['diagnostics/fluidMass'] = [s['mass'] for s in stats]
        h['diagnostics/meanPressure'] = [s['pmean'] for s in stats]
        h['diagnostics/maxPressure'] = [s['pmax'] for s in stats]
        h['probes/positions'] = PROBES
        h['probes/pressure'] = probes
        h['probes/wet'] = wet
        h['probes'].attrs['method'] = 'Fluid-only normalized Wendland C2 on post-shift physical snapshots; dry=0'
        h.attrs['complete'] = True
    partial.rename(output)
    receipt = {'label': label, 'run': str(run), 'recording': str(output), 'frames': count,
        'particles': n, 'dtype': 'float64 (lossless promotion for FP32 sources)',
        'skipped_duplicate_times': len(paths) - count, 'snapshots': receipts,
        'metrics': stats, 'sha256': None}
    with output.open('rb') as f:
        receipt['sha256'] = hashlib.file_digest(f, 'sha256').hexdigest()
    output.with_suffix('.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--label', required=True)
    args = ap.parse_args()
    result = export_run(args.run, args.out, args.label)
    print(json.dumps({k: v for k, v in result.items() if k not in ('snapshots', 'metrics')}), flush=True)


if __name__ == '__main__':
    main()
