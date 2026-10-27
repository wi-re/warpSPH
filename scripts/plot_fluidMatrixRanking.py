"""One figure of case-relative agreement and cost; saved trajectories only.

Vortices use the user's energy-retention criterion. Failure is categorical,
and runtime never substitutes for physical agreement. Export exact losses and
ranking recipes beside the figure for inspection and alternative weighting.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch, Rectangle

from run_fluidMatrix import DEFAULT_ROOT, write_json
from probe_deltaSPHMarrone import BUCHNER, _rollingMedian
from plot_sloshingOverlay import smooth, ROLL


SCHEMES = ('deltaSPH', 'sun2017DeltaSPH', 'divergenceFree', 'omniIncompressible',
           'band2018pb', 'iisph', 'dfsphReference', 'artificialCompressible')
NAMES = ('Delta-SPH', 'Sun delta-plus', 'VD+PS', 'Omni', 'Band pressure BC',
         'IISPH', 'Reference DFSPH', 'ACSPH')
COLUMNS = [
    ('tgv', 'periodic', 'TGV', 'energy'),
    ('decayingVortices', 'periodic', 'Vortices', 'retention'),
    ('shearDecay', 'periodic', 'Shear decay', 'energy'),
    ('boundedVortices', 'particles', 'Box vortices', 'retention'),
    ('marrone31', 'particles', 'Marrone 3.1', 'pressure'),
    ('marrone31', 'analytic', 'Marrone 3.1', 'pressure'),
    ('sloshing', 'particles', 'Sloshing', 'pressure'),
    ('sloshing', 'analytic', 'Sloshing', 'pressure'),
    ('hydrostatic', 'particles', 'Hydrostatic', 'rest'),
    ('hydrostatic', 'analytic', 'Hydrostatic', 'rest'),
    ('englishWedge', 'particles', 'English wedge', 'rest'),
    ('englishWedge', 'analytic', 'English wedge', 'rest'),
    ('taylorCouette', 'analytic', 'Couette', 'profile / torque'),
]
RECIPES = {
    'tgv': '100 sqrt(integral((E/E0-exp(-0.04t))^2) dt / integral(exp(-0.04t)^2) dt).',
    'shearDecay': 'Same energy-curve relative L2 error; expected E/E0=exp(-2*0.0185*(2*pi)^2*t). Includes energy in unwanted modes.',
    'decayingVortices': 'User-requested relative energy retention, not analytic accuracy: completed runs whose maximum E/E0 <= 1.001 rank by 100*(1-Efinal/E0), less loss better. Any >0.1% rise above initial energy puts a run behind all nongrowing runs, ordered by maximum gain. Cells show loss %, or GAIN and maximum gain %. Physical viscosity and sampling differ; Band viscosity is included only when the row settings enable it.',
    'boundedVortices': 'Same energy-retention ranking and growth penalty as periodic vortices.',
    'marrone31': '100 sqrt(mean per-sensor normalized MSE), P1/P2 samples against existing coarse Buchner digitization in probe_deltaSPHMarrone.py. Resample simulated nondimensional pressure to dt*=0.005 and use a 0.1 t* rolling median before sampling at reference times. Dry zero readings are retained. Reference approximate +/-0.05 P*, +/-0.15 t*: close ranks not significant.',
    'sloshing': 'Relative L2 pressure-trace error, 100 sqrt(integral((p_sim-p_exp)^2)/integral(p_exp^2)), over common full record; case-native sensorPressure. Both experiment and simulation use existing gap-aware 10ms Gaussian smoothing on 1ms grid; at least 95% coverage required. Includes pressure magnitude and timing; no time shifting. Native estimators differ between some schemes/wall representations; common fluid probe has dry/unresolved gaps and is not silently filled to create a ranking.',
    'hydrostatic': '100 max(abs(tail median bulk pressure-gradient ratio - 1), tail median max-fluid-speed / sqrt(9.81*0.5)). Heuristic worst normalized static-water diagnostic, not a universal percent accuracy. Tail medians from existing comparison (last 10% of steps).',
    'englishWedge': 'Same static-water diagnostic as hydrostatics. Bulk gradient excludes the wedge neighborhood; does not certify corner pressure accuracy.',
    'taylorCouette': '100 max(abs(profileAmplitude-1), abs(inner torque/exact-1), abs(outer opposing torque/exact-1)); existing tail medians.',
    'runtime': 'Total run wall seconds including initialization, video, diagnostics and HDF5 exports. Only completed runs ranked; presets, dt, initialization and I/O work differ. Not an isolated solver benchmark.',
    'colors': 'Independent log-loss min/max per case/boundary column among eligible completed runs. Numeric ranks are ordinal; close metric values have close colors. Only one candidate gets neutral blue, not a claim of comparative superiority.',
    'endpoint': 'Historical negligible terminal remainder samples are excluded using retained audited flags; four original corrupted shear endpoints used valid preterminal samples. New runs snap rounding-sized endpoint labels before another physics step. Raw source data are unchanged.',
    'coverage': 'Unsupported, stopped and missing-data cases are separate categories; they receive no invented numeric loss. No averaging across cases and no combined quality/speed score.',
}


def series(root, item):
    source = Path(item.get('data_root', root))
    with np.load(source / item['id'] / 'series.npz') as z:
        arrays = {key: z[key] for key in z.files}
    if item.get('roundoff_terminal_sample_excluded'):
        arrays = {key: value[:-1] for key, value in arrays.items()}
    return arrays


def nrmse(t, observed, expected):
    ok = np.isfinite(t) & np.isfinite(observed) & np.isfinite(expected)
    if ok.sum() < 3:
        return None
    denom = np.trapezoid(expected[ok] ** 2, t[ok])
    if denom <= 0:
        return None
    return float(100 * np.sqrt(np.trapezoid((observed[ok] - expected[ok]) ** 2, t[ok]) / denom))


def loss(root, item, experimental):
    test = item['test']
    if test in ('decayingVortices', 'boundedVortices'):
        if not all(math.isfinite(item[key]) for key in ('energy_max_ratio', 'energy_final_ratio')):
            return None, 'missing_data'
        growth = item['energy_max_ratio'] > 1.001
        return (100 * (item['energy_max_ratio'] - 1) if growth
                else 100 * max(0, 1 - item['energy_final_ratio'])), 'energy_growth' if growth else 'ranked'
    a = series(root, item)
    if test in ('tgv', 'shearDecay'):
        rate = .04 if test == 'tgv' else 2 * .0185 * (2 * np.pi) ** 2
        value = nrmse(a['t'], a['fluidKineticEnergy'] / a['fluidKineticEnergy'][0], np.exp(-rate * a['t']))
    elif test in ('hydrostatic', 'englishWedge'):
        value = 100 * max(abs(item['matrixPressureSlopeRatio_tail_median'] - 1),
                          item['velocity_tail_median'] / math.sqrt(9.81 * .5))
    elif test == 'taylorCouette':
        value = 100 * max(abs(item[key] - 1) for key in
                         ('profileAmplitude_tail_median', 'torqueInner_ratio_tail_median',
                          'torqueOuter_ratio_tail_median'))
    elif test == 'marrone31':
        per_sensor = []
        for k, name in enumerate(('P1', 'P2')):
            tt, pp = a['tStar'], a[f'pProbe{k}Star']
            ref = np.asarray(BUCHNER[name])
            valid = np.isfinite(tt) & np.isfinite(pp)
            if valid.sum() < 3 or tt[valid][-1] < ref[:, 0].max():
                return None, 'missing_data'
            grid = np.arange(0.0, tt[valid][-1], .005)
            median = _rollingMedian(np.interp(grid, tt[valid], pp[valid]), 21)
            observed = np.interp(ref[:, 0], grid, median)
            per_sensor.append(np.mean((observed - ref[:, 1]) ** 2) / np.mean(ref[:, 1] ** 2))
        value = float(100 * np.sqrt(np.mean(per_sensor)))
    elif test == 'sloshing':
        if 'sensorPressure' not in a:
            return None, 'missing_data'
        t, p = smooth(a['t'], a['sensorPressure'], .01)
        et, ep = experimental
        mask = (t >= et[0]) & (t <= et[-1])
        t, p = t[mask], p[mask]
        if len(t) < 3 or np.isfinite(p).mean() < .95:
            return None, 'missing_data'
        value = nrmse(t, p, np.interp(t, et, ep))
    else:
        raise ValueError(test)
    return value, 'ranked' if value is not None and math.isfinite(value) else 'missing_data'


def ranking(values):
    order = sorted(values, key=lambda pair: (pair[2] if len(pair) > 2 else 0, pair[1]))
    result = {}
    has_growth = any(len(v) > 2 and v[2] for v in order)
    normal_logs = [math.log10(max(v[1], 1e-12)) for v in order if len(v) < 3 or not v[2]]
    growth_logs = [math.log10(max(v[1], 1e-12)) for v in order if len(v) > 2 and v[2]]
    for i, entry in enumerate(order):
        key, val = entry[:2]
        growth = len(entry) > 2 and entry[2]
        logs = growth_logs if growth else normal_logs
        logarithm = math.log10(max(val, 1e-12))
        normalized = ((logarithm - min(logs)) / (max(logs) - min(logs)) if max(logs) > min(logs) else (1 if growth else .5))
        if has_growth:
            normalized = .9 + .1 * normalized if growth else .8 * normalized
        result[key] = dict(rank=i + 1, value=val, color_fraction=normalized, candidates=len(order))
    return result


def measured_loss(root, item, experimental):
    """Incomplete exports remain missing data, never an invented score."""
    if (item.get('test') in ('marrone31','sloshing','hydrostatic','englishWedge')
            and item.get('pressure_observation_complete') is False):
        return None,'missing_data','Incomplete physical-pressure observation; see per-row qualification.'
    try:
        value, status = loss(root, item, experimental)
        if value is None or not math.isfinite(value):
            return None, 'missing_data', 'No finite case metric'
        return value, status, None
    except (KeyError, TypeError, ValueError, OSError, IndexError) as error:
        return None, 'missing_data', f'{type(error).__name__}: {error}'


def display_layout(manifest):
    """A combined report declares its rows without changing legacy defaults."""
    schemes, names, columns = SCHEMES, NAMES, list(COLUMNS)
    if manifest.get('profile') in ('prepared-incompressible', 'classic-fluid-comparison'):
        schemes = tuple(manifest['schemes'])
        names = tuple(manifest['labels'][k] for k in schemes)
        columns.insert(4, ('boundedVortices', 'analytic', 'Box vortices', 'retention'))
    return schemes, names, columns


def plot(root, destination, metrics=None):
    manifest=json.loads((root/'manifest.json').read_text())
    schemes, names, columns = display_layout(manifest)
    rows = json.loads((metrics or root / 'comparison_metrics.json').read_text())
    lookup = {r['id']: r for r in rows}
    raw = np.genfromtxt(ROLL, delimiter='\t', skip_header=1)
    experimental = smooth(raw[:, 0], raw[:, 1] * 100, .01)
    table = []
    agreement, cost = {}, {}
    for j, (test, boundary, _, _) in enumerate(columns):
        loss_values, time_values = [], []
        for scheme in schemes:
            key = f'{test}__{scheme}__{boundary}'
            item = lookup.get(key,dict(status='pending',target_time=1.))
            record = dict(id=key, test=test, boundary=boundary, scheme=scheme,
                          run_status=item['status'], completion_fraction=min(1, (item.get('actual_time') or 0) / item['target_time']),
                          loss=None, accuracy_status=item['status'], runtime_seconds=None,
                          metric_error=None,
                          run_date=item.get('run_date'), data_root=item.get('data_root', str(root)),
                          band_viscosity=bool(item.get('band_viscosity', False)),
                          cap_count=sum(item.get(p+'_cap_count',0) for p in ('DF','CD','IISPH')))
            if item['status'] == 'completed':
                value, status, error = measured_loss(root, item, experimental)
                record.update(loss=value, accuracy_status=status, metric_error=error,
                              runtime_seconds=item.get('evolution_and_output_seconds',item['wall_seconds']))
                if status in ('ranked', 'energy_growth'):
                    loss_values.append((key, value, status == 'energy_growth'))
                time_values.append((key,item.get('evolution_and_output_seconds',item['wall_seconds'])))
            table.append(record)
        agreement.update(ranking(loss_values))
        cost.update(ranking(time_values))
    for row in table:
        row['accuracy_rank'] = agreement.get(row['id'], {}).get('rank')
        row['runtime_rank'] = cost.get(row['id'], {}).get('rank')
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / 'case_ranking.json', dict(recipes=RECIPES, rows=table))
    with (destination / 'case_ranking.csv').open('w', newline='') as handle:
        w = csv.DictWriter(handle, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    by_id = {r['id']: r for r in table}
    cmap = LinearSegmentedColormap.from_list('case_loss', ['#268781', '#f3edb0', '#cf5b49'])
    fig = plt.figure(figsize=(23 if len(schemes)>8 else 21.5, 13.2 + .65*max(0,len(schemes)-8)), facecolor='white')
    grid = fig.add_gridspec(2, 1, left=.13, right=.975, top=.815, bottom=.215, hspace=.40)
    labels = [name + '\n' + ('periodic' if boundary == 'periodic' else boundary)
              for _, boundary, name, _ in columns]
    for panel, (scores, title) in enumerate(((agreement, 'A  Quality: physical agreement or energy retention'),
                                           (cost, 'B  Wall time to complete the case'))):
        ax = fig.add_subplot(grid[panel])
        for i, scheme in enumerate(schemes):
            for j, (test, boundary, _, metric) in enumerate(columns):
                key = f'{test}__{scheme}__{boundary}'
                item = by_id[key]
                score = scores.get(key)
                hatch = None
                if score:
                    only = score['candidates'] == 1
                    color = '#b9d7e5' if only else cmap(score['color_fraction'])
                    suffix = '%' if panel == 0 else ' s'
                    number = f'{score["value"]:,.0f}' if score['value'] >= 100 else f'{score["value"]:.3g}'
                    detail = number + suffix
                    label = ('only' if only else f'#{score["rank"]}') + '\n' + detail
                    if item.get('cap_count'):label+=' *'
                    if panel == 0 and item['accuracy_status'] == 'energy_growth':
                        label = f'#{score["rank"]}\nGAIN +{score["value"]:.3g}%'
                    if panel == 0 and item['scheme'] == 'band2018pb' and not item['band_viscosity'] and test in ('tgv', 'shearDecay', 'decayingVortices', 'boundedVortices'):
                        label += ' †'
                elif item['run_status'] == 'unsupported':
                    color, label = '#e8eaed', '—'
                elif item['run_status'] in ('pending', 'running'):
                    color, label = '#454d59', item['run_status'].upper()
                elif item['run_status'] != 'completed':
                    progress = 100 * item['completion_fraction']
                    color, label = '#454d59', 'STOP\n' + ('<1% t' if progress < 1 else f'{progress:.0f}% t')
                elif item['accuracy_status'] == 'no_reference':
                    color, label, hatch = '#fafafa', 'no\nreference', '///'
                else:
                    color, label = '#f2e4d9', 'missing\nmetric'
                ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, facecolor=color, edgecolor='white',
                                       linewidth=1.8, hatch=hatch))
                dark = item['run_status'] not in ('completed', 'unsupported')
                ax.text(j, i, label, ha='center', va='center', color='white' if dark else '#14242b',
                        fontsize=8.5, linespacing=1.4)
        ax.set(xlim=(-.5, len(columns) - .5), ylim=(len(schemes) - .5, -.5))
        ax.set_xticks(range(len(columns)), labels, fontsize=8.5)
        row_names = []
        for scheme, name in zip(schemes, names):
            dates = sorted({r['run_date'] for r in table if r['scheme'] == scheme and r['run_date']})
            date_label=(', '.join(dates) if len(dates)<2 or dates[0][:8]!=dates[-1][:8]
                        else dates[0]+'–'+dates[-1][-2:])
            row_names.append(name + ('\n' + date_label if dates else ''))
        ax.set_yticks(range(len(schemes)), row_names, fontsize=10)
        ax.tick_params(length=0, pad=9)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(title + ('\ncell = rank + case-specific loss (%); GAIN is penalized below nongrowing runs' if panel == 0
                             else '\ncell = rank + total seconds; completed runs only'),
                     loc='left', pad=15, fontsize=12, fontweight='bold')
    title = manifest.get('comparison_title') or (('Prepared incompressible comparison — '+manifest['scale']+' scale') if manifest.get('profile')=='prepared-incompressible' else 'Classic-fluid schemes: quantitative ranking by case')
    fig.suptitle(title, x=.13, ha='left', y=.977,
                 fontsize=21, fontweight='bold', color='#14242b')
    fig.text(.13, .94, 'FP32 saved runs  ·  each column has its own best-to-worst color range  ·  quality and cost remain separate',
             fontsize=12, color='#46545b')
    cb = fig.add_axes([.13, .895, .30, .015])
    colorbar = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=cmap), cax=cb, orientation='horizontal')
    colorbar.set_ticks([0, 1], labels=['Best in column', 'Worst completed in column'])
    colorbar.ax.tick_params(length=0, labelsize=9)
    colorbar.outline.set_visible(False)
    fig.legend(handles=[Patch(facecolor='#454d59', label='Stopped: % of target time reached'),
                        Patch(facecolor='#e8eaed', label='Unsupported'),
                        Patch(facecolor='#b9d7e5', label='Only one completed candidate')],
               loc='upper left', bbox_to_anchor=(.46, .925), fontsize=9, frameon=False, ncol=2)
    notes = [
        'Loss definitions (lower is better): TGV / shear = relative L2 error of the energy-decay curve; Marrone = P1/P2 error against coarse digitized measurements.',
        'Sloshing = relative L2 error of case-native Sensor 1 with identical 10 ms smoothing of experiment and simulation; includes phase error. Estimators can differ.',
        'Hydrostatic / wedge = max(|bulk pressure-gradient ratio − 1|, tail max-speed / √gH) × 100; this is a heuristic rest-state loss, not corner-pressure validation.',
        'Vortices = energy loss 1 − Efinal/Einitial (less is better); >0.1% growth above initial energy is ranked below all nongrowing runs. GAIN shows maximum growth.',
        'Couette = worst relative error of tail profile amplitude and the two torques. Energy retention is a relative criterion, not exact viscous-decay accuracy.',
        'Colors use log-loss scaling independently within a column: ranks and raw values are shown. A best color is not a physics pass; close ranks are not significant.',
        '* Native projection iteration caps occurred; legacy cap counters unavailable. A diagnostic rank does not qualify convergence. † This Band cell uses its nonviscous predictor. Inspect per-row viscosity settings; dates distinguish runs from retained controls.',
        'Prepared-profile cost subtracts fixture creation; preparation is reported separately. Times include setup, video, diagnostics and exports. No timing repeats or overall aggregate.',
    ]
    if manifest.get('profile') in ('prepared-incompressible','classic-fluid-comparison'):
        notes.append('Native periodic rows use a mean-free density source as well as prepared ICs; legacy source policies differ. Mean sampled density can drift. Improved ranks do not validate unchanged absolute-density evolution.')
    if manifest.get('profile')=='classic-fluid-comparison':
        notes.append('Dated Delta / Sun / AC controls retain their original ICs, C2 kernels and presets; prepared projection rows use C4 (Couette C2). This is a capability survey, not a solver-only A/B. Separate TGV IC control excluded.')
    notes.extend(manifest.get('comparison_notes',[]))
    for i, note in enumerate(notes):
        spacing = min(.017, .15 / max(1, len(notes)-1))
        fig.text(.13, .17 - spacing * i, note, fontsize=9.1, color='#435058')
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(destination / ('case_ranking.' + ext), dpi=220, facecolor='white')
    plt.close(fig)
    page = root / 'comparison.html'
    if page.exists() and destination.is_relative_to(root):
        document = page.read_text()
        start, end = '<!-- RANKING_FIGURE -->', '<!-- END_RANKING_FIGURE -->'
        if start in document:
            before, remainder = document.split(start, 1)
            document = before + remainder.split(end, 1)[1]
        relative = str((destination / 'case_ranking.png').relative_to(root))
        block = start + f'<p><a href="{relative}">Quantitative rankings by case</a></p><a href="{relative}"><img width="1300" src="{relative}"></a>' + end
        document = document.replace('<h1>Classic fluid comparison</h1>', '<h1>Classic fluid comparison</h1>' + block)
        page.write_text(document)
    print(json.dumps(dict(figure=str(destination / 'case_ranking.png'), entries=len(table),
                          accuracy_ranked=len(agreement), runtime_ranked=len(cost)), indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    p.add_argument('--out', type=Path)
    p.add_argument('--metrics', type=Path, help='Explicit merged metrics with per-row source roots and run dates.')
    args = p.parse_args()
    plot(args.root, args.out or args.root / 'ranking', args.metrics)
