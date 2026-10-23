"""Rebuild quantitative fluid-matrix comparisons from saved evidence only."""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from run_fluidMatrix import DEFAULT_ROOT, REPO, write_json
from plot_sloshingOverlay import smooth, WINDOWS, ROLL


def finite(value):
    value = float(value)
    return value if math.isfinite(value) else None


def tail(arr):
    return finite(np.nanmedian(arr[-max(len(arr) // 10, 1):]))


def csv_write(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def profiles(out, row):
    """Raw final profiles stay available independently of scalar verdicts."""
    if row['test'] not in ('hydrostatic', 'englishWedge', 'taylorCouette'):
        return
    final, output = out / 'final_state.pt', out / 'final_profile.csv'
    if not final.exists() or output.exists():
        return
    import torch
    st = torch.load(final, map_location='cpu', weights_only=True)
    f = st['kinds'].numpy() == 0
    x, v = st['positions'].numpy()[f], st['velocities'].numpy()[f]
    if row['test'] == 'taylorCouette':
        r = np.linalg.norm(x, axis=1)
        u = (x[:, 0] * v[:, 1] - x[:, 1] * v[:, 0]) / r
        exact = .2 ** 2 * (.5 ** 2 / r - r) / (.5 ** 2 - .2 ** 2)
        csv_write(output, [dict(radius=float(ri), azimuthal_velocity=float(ui),
                                steady_exact_velocity=float(ei)) for ri, ui, ei in zip(r, u, exact)])
    elif st.get('pressures') is not None:
        p = st['pressures'].numpy()[f]
        csv_write(output, [dict(x=float(xi[0]), y=float(xi[1]), pressure=float(pi),
                                density=float(ri)) for xi, pi, ri in zip(x, p, st['densities'].numpy()[f])])


def analyze(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    rows, curves, peaks, html_rows = [], {}, [], []
    for row in manifest['rows']:
        out = root / row['id']
        file = out / 'result.json'
        data = json.loads(file.read_text()) if file.exists() else {}
        state = 'unsupported' if row['skip'] else data.get('status', 'pending')
        if state == 'pending' and (out / 'progress.json').exists():
            state = 'running'
        m = dict(id=row['id'], test=row['test'], scheme=row['scheme'], boundary=row['boundary'],
                 status=state, target_time=row['spec']['tLimit'], actual_time=data.get('time'),
                 particles=data.get('initial', {}).get('particles'), steps=data.get('steps'),
                 wall_seconds=data.get('wall_seconds'), wall_ms_per_step=data.get('wall_ms_per_step'),
                 integrator_ms=data.get('integrator_ms_mean'),
                 reason=row['skip'] or data.get('stop_reason') or data.get('error'), notes=row['notes'])
        m.update(matrix_version=manifest['version'], run_date=manifest['created'],
                 band_viscosity=bool(row.get('scheme_settings', {}).get('bandViscosity', False)))
        series = out / 'series.npz'
        # Exceptions and wall-budget stops still have every diagnostic row
        # written before the failure. Export those as explicitly partial data.
        journal = out / 'diagnostics.jsonl'
        if not series.exists() and data and journal.exists():
            partial = []
            for line in journal.read_text().splitlines():
                try:
                    partial.append(json.loads(line))
                except json.JSONDecodeError:
                    break
            if partial:
                csv_write(out / 'diagnostics.csv', partial)
                keys = set().union(*(r.keys() for r in partial))
                np.savez_compressed(series, **{k: np.asarray([r.get(k, np.nan) for r in partial]) for k in keys})
                write_json(out / 'partial_data.json', dict(reason=data.get('error'), rows=len(partial), complete=False))
        if series.exists():
            with np.load(series) as z:
                a = {key: z[key] for key in z.files}
            # The original end-time clamp could take one extra ~1e-13 s
            # advance after a fixed-step run had reached its target within
            # floating-point summation error. Preserve raw exports verbatim;
            # exclude that sample from derived metrics, including harmless
            # instances. Some pressure maps divide by dt and corrupt it.
            tt = a['t']
            if len(tt) > 3:
                increments = np.diff(tt)
                typical = float(np.median(increments[-min(len(increments), 100):]))
                remainder = float(increments[-1])
                near_end = abs(float(tt[-2]) - row['spec']['tLimit']) <= 1e-10 * max(1, row['spec']['tLimit'])
                if typical > 0 and 0 <= remainder < 1e-6 * typical and near_end:
                    ee, vv = a['fluidKineticEnergy'], a['maxVelocity']
                    jump = float(ee[-1] / max(abs(ee[-2]), 1e-300))
                    m.update(roundoff_terminal_sample_excluded=True, raw_final_time=float(tt[-1]),
                             raw_last_dt=remainder, ordinary_dt=typical,
                             raw_final_kinetic_energy=finite(ee[-1]),
                             raw_velocity_peak=finite(np.nanmax(vv)),
                             terminal_energy_jump=finite(jump),
                             endpoint_corrupted=bool(jump > 1000))
                    a = {key: value[:-1] for key, value in a.items()}
            curves[row['id']] = a
            t, energy = a['t'], a['fluidKineticEnergy']
            m['actual_time'] = float(t[-1])
            if energy[0] > 0:
                m['energy_final_ratio'] = finite(energy[-1] / energy[0])
                m['energy_max_ratio'] = finite(np.nanmax(energy / energy[0]))
            if 'maxVelocity' in a:
                m['velocity_peak'] = finite(np.nanmax(a['maxVelocity']))
                m['velocity_tail_median'] = tail(a['maxVelocity'])
            m['density_min_all_times'] = finite(np.nanmin(a['fluidDensityMin']))
            m['density_max_all_times'] = finite(np.nanmax(a['fluidDensityMax']))
            m['mass_relative_drift'] = finite(a['fluidMass'][-1] / a['fluidMass'][0] - 1)
            m['reached_target'] = bool(t[-1] >= row['spec']['tLimit'] - 1e-6)
            for key in ('pressureSlopeRatio', 'matrixPressureSlopeRatio', 'pressureResidual',
                        'profileAmplitude', 'embeddedMinDensity', 'densityP05', 'voidFraction'):
                if key in a:
                    m[key + '_tail_median'] = tail(a[key])
            for key in ('nPenetrating', 'maxPenetrationDx', 'nObstaclePen', 'maxObstaclePenDx'):
                if key in a:
                    m[key + '_max'] = finite(np.nanmax(a[key]))
            if row['test'] == 'tgv':
                expected = math.exp(-.04 * t[-1])
                m['analytic_energy_final_ratio'] = expected
                m['energy_relative_error'] = finite(energy[-1] / energy[0] / expected - 1)
                mask = (t > .2) & np.isfinite(energy) & (energy > 0)
                if mask.sum() > 10:
                    m['nu_effective_energy_fit'] = finite(-np.polyfit(t[mask], np.log(energy[mask]), 1)[0] / 4)
            if row['test'] == 'shearDecay' and 'modeAmplitude' in a:
                mask = (t > .1) & np.isfinite(a['modeAmplitude']) & (a['modeAmplitude'] > 1e-6)
                if mask.sum() > 10:
                    m['nu_effective_shear_fit'] = finite(-np.polyfit(t[mask], np.log(a['modeAmplitude'][mask]), 1)[0] / (2 * np.pi) ** 2)
            if row['test'] == 'taylorCouette':
                for key in ('torqueInner', 'torqueOuter'):
                    if key in a and 'torqueExact' in a:
                        m[key + '_ratio_tail_median'] = tail(a[key] / a['torqueExact'] * (-1 if key == 'torqueOuter' else 1))
            if row['test'] == 'sloshing':
                for key in ('sensorPressure', 'sensorPressureProbe', 'sensorPressureWall'):
                    if key not in a or len(t) < 3 or not np.isfinite(a[key]).any():
                        continue
                    grid, sm = smooth(t, a[key], .01)
                    np.savez_compressed(out / (key + '_smoothed_10ms.npz'), t=grid, pressure=sm)
                    for i, (lo, hi) in enumerate(WINDOWS):
                        mask = (grid >= lo) & (grid < hi) & np.isfinite(sm)
                        # Never rank a truncated window against a full record.
                        if mask.any() and t[-1] >= hi:
                            xx, yy = grid[mask], sm[mask]
                            k = int(np.argmax(yy))
                            peaks.append(dict(id=row['id'], estimator=key, impact=i + 1,
                                              peak_pa=float(yy[k]), peak_time=float(xx[k])))
            profiles(out, row)
            write_json(out / 'metrics.json', m)
        rows.append(m)
        def fmt(key):
            val = m.get(key)
            return '' if val is None else f'{val:.4g}' if isinstance(val, float) else str(val)
        links = []
        for path, label in ((out / 'diagnostics.csv', 'CSV'), (out / 'series.npz', 'NPZ'),
                            (out / 'final_profile.csv', 'profile'), (out / 'metrics.json', 'metrics'),
                            (out / 'run.log', 'log')):
            if path.exists():
                links.append(f'<a href="{path.relative_to(root)}">{label}</a>')
        videos = list(out.rglob('*.mp4')) if out.exists() else []
        if videos:
            links.insert(0, f'<a href="{videos[0].relative_to(root)}">video</a>')
        recordings = list(out.rglob('trajectory.h5')) if out.exists() else []
        if recordings:
            links.append(f'<a href="{recordings[0].relative_to(root)}">particle HDF5</a>')
        fields = ('test', 'scheme', 'boundary', 'status', 'actual_time', 'target_time', 'particles',
                  'wall_ms_per_step', 'energy_final_ratio', 'energy_relative_error',
                  'matrixPressureSlopeRatio_tail_median', 'profileAmplitude_tail_median', 'reason')
        html_rows.append('<tr>' + ''.join('<td>' + html.escape(fmt(k)) + '</td>' for k in fields)
                         + '<td>' + ' · '.join(links) + '</td></tr>')
    # Correct like-for-like experimental pressure comparison, same smoothing.
    raw = np.genfromtxt(ROLL, delimiter='\t', skip_header=1)
    gt, gp = smooth(raw[:, 0], raw[:, 1] * 100, .01)
    for i, (lo, hi) in enumerate(WINDOWS):
        mask = (gt >= lo) & (gt < hi) & np.isfinite(gp)
        xx, yy = gt[mask], gp[mask]
        k = int(np.argmax(yy))
        peaks.append(dict(id='SPHERIC_measurement', estimator='sensor1_10ms', impact=i + 1,
                          peak_pa=float(yy[k]), peak_time=float(xx[k])))
    csv_write(root / 'comparison_metrics.csv', rows)
    write_json(root / 'comparison_metrics.json', rows)
    csv_write(root / 'sloshing_impact_peaks_10ms.csv', peaks)
    plots = []
    for test in dict.fromkeys(r['test'] for r in manifest['rows']):
        group = [r for r in manifest['rows'] if r['test'] == test and r['id'] in curves]
        if not group:
            continue
        metrics = [('fluidKineticEnergy', 'fluid kinetic energy')]
        if test in ('tgv', 'decayingVortices', 'boundedVortices'):
            metrics = [('energy_ratio', 'E / E(0)')]
        if test == 'shearDecay':
            metrics = [('modeAmplitude', 'shear amplitude / initial')]
        elif test in ('hydrostatic', 'englishWedge'):
            metrics += [('matrixPressureSlopeRatio', 'pressure gradient / hydrostatic')]
        elif test == 'sloshing':
            metrics += [('sensorPressure', 'Sensor 1 pressure [Pa], 10 ms Gaussian'),
                        ('sensorPressureProbe', 'fluid probe pressure [Pa], 10 ms Gaussian')]
        elif test == 'taylorCouette':
            metrics += [('profileAmplitude', 'velocity profile / exact steady profile')]
        elif test == 'marrone31':
            # Native probe key names are preserved in exported CSV/NPZ.
            metrics += [(k, k) for k in ('pProbe0', 'pProbe1', 'pProbe2')]
        for key, label in metrics:
            fig, ax = plt.subplots(figsize=(11, 6))
            plotted = False
            for row in group:
                a = curves[row['id']]
                t = a['t']
                if key == 'energy_ratio':
                    if a['fluidKineticEnergy'][0] <= 0:
                        continue
                    y = a['fluidKineticEnergy'] / a['fluidKineticEnergy'][0]
                elif key in a:
                    y = a[key]
                else:
                    continue
                if test == 'sloshing' and key.startswith('sensor'):
                    if not np.isfinite(y).any() or len(t) < 3:
                        continue
                    t, y = smooth(t, y, .01)
                ax.plot(t, y, label=row['scheme'] + '/' + row['boundary'],
                        linestyle='--' if row['boundary'] == 'analytic' else '-', linewidth=1)
                plotted = True
            if plotted:
                if test == 'tgv':
                    t = np.linspace(0, 5, 300)
                    ax.plot(t, np.exp(-.04 * t), 'k:', linewidth=2, label='analytic ν=0.01')
                if test in ('hydrostatic', 'englishWedge') and key == 'matrixPressureSlopeRatio':
                    ax.axhline(1, color='black', linestyle=':')
                if test == 'sloshing' and key.startswith('sensor'):
                    ax.plot(gt, gp, 'k:', linewidth=2, label='SPHERIC Sensor 1, same 10 ms smoothing')
                ax.set(xlabel='time [s]', ylabel=label, title=test)
                if key == 'energy_ratio':
                    ax.set_yscale('log')
                ax.grid(alpha=.25)
                ax.legend(fontsize=7, ncol=2)
                fig.tight_layout()
                name = f'comparison_{test}_{key}.png'
                fig.savefig(root / name, dpi=140)
                plots.append(f'<a href="{name}"><img width="600" src="{name}"></a>')
            plt.close(fig)
    counts = {s: sum(r['status'] == s for r in rows) for s in sorted({r['status'] for r in rows})}
    header = '<!doctype html><meta charset="utf-8"><title>Fluid matrix comparison</title><style>body{font:14px system-ui;margin:24px}td,th{padding:6px;border-bottom:1px solid #ddd;text-align:left;max-width:320px}img{max-width:100%}table{border-collapse:collapse}</style>'
    short_count = sum(bool(r.get('roundoff_terminal_sample_excluded')) for r in rows)
    corrupt_count = sum(bool(r.get('endpoint_corrupted')) for r in rows)
    text = (f'<h1>Classic fluid comparison</h1><p>{time.strftime("%Y-%m-%d %H:%M:%S %Z")}: {html.escape(str(counts))}</p>'
            '<p>Eight schemes; FP32; sequential execution; established case-specific presets. Completion is not an accuracy verdict. '
            'Unsupported pairs are explicit. No solver issues are tuned during this pass. '
            'Wall ms/step includes video, diagnostics and particle exports. TGV error is E/E₀ divided by exp(−0.04t), minus 1. '
            f'{corrupt_count} shear endpoints were corrupted by an extra floating-point remainder step in the harness. '
            f'Derived metrics exclude all {short_count} such tiny terminal steps, using the preceding sample already at the target within rounding error; raw data and videos are unchanged. '
            'Pressure slopes and Couette amplitudes should approach 1; raw metrics and profiles are linked. '
            'Band currently omits physical viscosity. Initial lattices, support sizes and some boundary estimators differ by preset.</p>'
            '<p><a href="comparison_metrics.csv">All quantitative metrics CSV</a> · '
            '<a href="sloshing_impact_peaks_10ms.csv">Sloshing peaks with matched smoothing CSV</a> · '
            '<a href="manifest.json">Inputs</a> · <a href="index.html">Queue summary</a></p>')
    cols = ('test', 'scheme', 'wall', 'status', 't', 'target', 'N', 'wall ms/step', 'E/E₀',
            'TGV energy error', 'hydrostatic slope', 'Couette amplitude', 'reason', 'evidence')
    target = root / 'comparison.html'
    temp = target.with_suffix('.tmp')
    temp.write_text(header + text + ''.join(plots) + '<table><tr>'
                    + ''.join('<th>' + k + '</th>' for k in cols) + '</tr>' + ''.join(html_rows) + '</table>')
    temp.replace(target)
    print('MATRIX_ANALYSIS ' + json.dumps(counts), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    analyze(p.parse_args().root)
