#!/usr/bin/env python3
"""Compare matched physical snapshots through a Lobovsky trajectory."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import h5py
import numpy as np


def analyze(reference, local, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    rows = []
    with h5py.File(reference, 'r') as a, h5py.File(local, 'r') as b:
        source_runs = {label: Path(h.attrs['source']) for label, h in (('reference', a), ('local', b))}
        ta, tb = a['time'][()], b['time'][()]
        np.testing.assert_array_equal(a['particles/UIDs'], b['particles/UIDs'])
        np.testing.assert_array_equal(a['particles/kinds'], b['particles/kinds'])
        fluid = a['particles/kinds'][()] == 0
        mass_a, mass_b = a['particles/masses'][()][fluid], b['particles/masses'][()][fluid]
        edges = (np.arange(0, 1.62001, .01), np.arange(0, 1.41001, .01))
        def binned(x, value):
            count = np.histogram2d(x[:, 0], x[:, 1], bins=edges)[0]
            total = np.histogram2d(x[:, 0], x[:, 1], bins=edges, weights=value)[0]
            return total / np.maximum(count, 1), count
        for i, t in enumerate(ta):
            j = int(np.argmin(abs(tb-t)))
            if abs(tb[j]-t) > 1e-10:
                continue
            xa, xb = (h['frames/positions'][k][fluid] for h, k in ((a, i), (b, j)))
            va, vb = (h['frames/velocities'][k][fluid] for h, k in ((a, i), (b, j)))
            pa, pb = (h['frames/pressures'][k][fluid] for h, k in ((a, i), (b, j)))
            common = []
            for component in range(2):
                aa, ca = binned(xa, va[:, component]); bb, cb = binned(xb, vb[:, component])
                mask = (ca >= 4) & (cb >= 4)
                common.append((aa[mask]-bb[mask])**2)
            row = {'t': float(t), 'reference_frame': i, 'local_frame': j,
                'position_rms': float(np.sqrt(np.mean(np.sum((xa-xb)**2, axis=1)))),
                'velocity_rms': float(np.sqrt(np.mean(np.sum((va-vb)**2, axis=1)))),
                'pressure_rms': float(np.sqrt(np.mean((pa-pb)**2))),
                'eulerian_velocity_rms_1cm': float(np.sqrt(np.mean(common[0]+common[1]))) if len(common[0]) else None}
            for label, x, v, p, m, h, k in (('reference', xa, va, pa, mass_a, a, i), ('local', xb, vb, pb, mass_b, b, j)):
                row[label] = {'ke': float(.5*np.sum(m*np.sum(v*v, axis=1))),
                    'potential_energy': float(9.81*np.dot(m, x[:, 1])),
                    'vmax': float(np.linalg.norm(v, axis=1).max()),
                    'front_p99': float(np.quantile(x[:, 0], .99)),
                    'height_p99': float(np.quantile(x[:, 1], .99)),
                    'pmax': float(p.max()), 'pmean': float(np.average(p, weights=m)),
                    'pressure_probes': h['probes/pressure'][k].tolist()}
            rows.append(row)
    if not rows:
        raise ValueError('No common physical snapshots')
    costs = {}
    for label, run in source_runs.items():
        result_path = run/'result.json'
        if not result_path.exists():
            continue
        result = json.loads(result_path.read_text())
        steps = result.get('steps', result.get('trajectory', []))
        steps = [r for r in steps if r.get('step', 0) >= 0]
        iterations = [r.get('iterations', r.get('pseudoIterations', 0)) for r in steps]
        wall = result['wall_seconds']
        costs[label] = {'steps': len(steps), 'wall_seconds': wall,
            'wall_ms_per_step': 1000*wall/len(steps) if steps else None,
            'capped_steps': sum(n >= 200 for n in iterations),
            'median_iterations': float(np.median(iterations)) if iterations else None,
            'stop_reason': result.get('stop_reason'), 'diverged': result.get('diverged', False)}
        timing = run/'solve-timing.jsonl'
        if timing.exists():
            receipts = [json.loads(line) for line in timing.read_text().splitlines()]
            costs[label]['graph_captured_steps'] = sum(r['captured'] for r in receipts)
            costs[label]['graph_disabled_reasons'] = sorted({r['disabled'] for r in receipts if r['disabled']})
    summary = {'reference': str(reference), 'local': str(local), 'common_frames': len(rows), 'costs': costs,
        'common_endpoint': rows[-1], 'rows': rows,
        'caveats': ['FP32 local / FP64 native; exact original mass with explicit FP32 rounding.',
            'Row errors are material-particle errors; chaotic trajectories can separate.',
            'Eulerian velocity uses 1 cm cells containing at least four particles in both runs.',
            'Pressure probes use the same post-shift fluid-only Wendland C2 estimator.',
            'Raw native probe.plt samples an earlier RK field; do not conflate the two.']}
    campaign_status = source_runs['reference'].parent/'status.json'
    if campaign_status.exists():
        campaign = json.loads(campaign_status.read_text())
        if campaign.get('allow_concurrent_gpu'):
            summary['caveats'].append('The native reference ran during authorized GPU sharing; its wall cost is not an isolated throughput baseline. The local replay is a separate run; these costs do not establish a controlled precision or hardware speed comparison.')
        if campaign.get('reference_limit'):
            limit = campaign['reference_limit']
            summary['caveats'].append(f'Native trajectory ended at its {limit["steps"]}-step guard, t={limit["actual_time"]:.9g} s, before the requested {limit["requested_time"]} s. Replay uses that actual endpoint; this is not a full requested-duration result.')
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    endpoint = rows[-1]
    report = ['# Lobovsky trajectory comparison', '',
        f'Common endpoint: {endpoint["t"]:.9g} s; {len(rows)} common physical frames.', '',
        '| Implementation | KE [J/m] | Max speed [m/s] | Maximum pressure [Pa] |',
        '|---|---:|---:|---:|']
    for label in ('reference', 'local'):
        m = endpoint[label]
        report.append(f'| {label} | {m["ke"]:.8g} | {m["vmax"]:.8g} | {m["pmax"]:.8g} |')
    report += ['', '| Implementation | Wall s | Wall ms/step | Steps at cap / total |', '|---|---:|---:|---:|']
    for label, c in costs.items():
        report.append(f'| {label} | {c["wall_seconds"]:.3f} | {c["wall_ms_per_step"]:.3f} | {c["capped_steps"]} / {c["steps"]} |')
    report += ['', f'Endpoint pressure material-row RMS difference: {endpoint["pressure_rms"]:.8g} Pa.',
        f'Endpoint velocity RMS on common 1 cm cells: {endpoint["eulerian_velocity_rms_1cm"]} m/s.', '',
        '![Comparison curves](comparison.png)', '', 'Caveats:', '']
    report += ['- '+s for s in summary['caveats']]
    report += ['', 'The full lossless viewer recordings are one directory above this report.',
        'Inspect campaign status.json and each result.json before treating an early endpoint as a full 2 s run.', '']
    (output/'report.md').write_text('\n'.join(report))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), layout='constrained')
    times = [r['t'] for r in rows]
    for label, title, color in (('reference', 'dynGPUSPH FP64', 'black'), ('local', 'warpSPH FP32 aligned', '#0072b2')):
        for ax, key, ylabel in ((axes[0, 0], 'ke', 'Kinetic energy [J/m]'), (axes[0, 1], 'vmax', 'Maximum speed [m/s]'),
                (axes[1, 0], 'front_p99', 'Fluid x, 99th percentile [m]'), (axes[1, 1], 'height_p99', 'Fluid y, 99th percentile [m]')):
            ax.plot(times, [r[label][key] for r in rows], label=title, color=color)
            ax.set_ylabel(ylabel)
        for sensor, ax in ((0, axes[2, 0]), (3, axes[2, 1])):
            ax.plot(times, [r[label]['pressure_probes'][sensor]/1000 for r in rows], label=title, color=color)
            ax.set_ylabel(f'Wall pressure P{sensor+1} [kPa]')
    axes[0, 0].legend()
    for ax in axes.flat:
        ax.set_xlabel('Time [s]'); ax.grid(alpha=.25)
    fig.suptitle('Lobovsky: supplied particles/mass, common reference timesteps\nAligned reference controls; convergence/cap history reported separately')
    fig.savefig(output/'comparison.png', dpi=160); plt.close(fig)
    print(json.dumps({'common_frames': len(rows), 'endpoint': rows[-1]}, indent=2), flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference', type=Path, required=True)
    ap.add_argument('--local', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    analyze(args.reference, args.local, args.out)


if __name__ == '__main__':
    main()
