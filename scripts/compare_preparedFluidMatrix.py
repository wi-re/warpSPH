"""Combine saved prepared projections with the normal classic-fluid controls.

This is postprocessing, not an executable simulation plan. Whole scheme rows
are selected by declared provenance, including failures and unsupported slots.
The input datasets are never modified. Alternate IC experiments stay separate.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import date
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from analyze_fluidMatrix import csv_write
from plot_fluidMatrixRanking import SCHEMES, NAMES, plot, series
from run_fluidMatrix import write_json


RETAINED = ('deltaSPH', 'sun2017DeltaSPH', 'artificialCompressible')
QUALIFICATION = (
    'Prepared incompressible rows replace the earlier projection results in the normal '
    'matrix. Delta-SPH, Sun delta-plus and ACSPH retain their original dated runs. '
    'Initial states, kernels, viscosity discretizations, source policies and timestep '
    'presets differ: this is a capability comparison, not a controlled solver-only A/B. '
    'Native periodic projections use periodicMeanFree during evolution; preparation '
    'alone does not account for their results. The separate rearranged TGV control is '
    'excluded from these rankings. Completion and evidence integrity do not certify '
    'physical correctness. No overall score is computed.'
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_dataset(root):
    root = root.resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    rows = json.loads((root / 'comparison_metrics.json').read_text())
    for records, kind in ((manifest['rows'], 'manifest'), (rows, 'metrics')):
        ids = [r['id'] for r in records]
        if len(ids) != len(set(ids)):
            raise ValueError(f'Duplicate {kind} IDs in {root}')
    if {r['id'] for r in manifest['rows']} != {r['id'] for r in rows}:
        raise ValueError(f'Manifest/metrics row mismatch in {root}')
    return manifest, rows


def compose(baseline, prepared):
    baseline, prepared = baseline.resolve(), prepared.resolve()
    bm, br = load_dataset(baseline)
    pm, pr = load_dataset(prepared)
    if pm.get('profile') != 'prepared-incompressible' or pm.get('scale') != 'full':
        raise ValueError('The normal full matrix requires the full prepared profile')
    if set(pm['schemes']) & set(RETAINED):
        raise ValueError('Prepared scheme IDs collide with retained control methods')
    # A subset/control must never silently discard other incompressible methods.
    required = {'dfsph', 'dfsphPredictor', 'dfsphShift', 'iisph',
                'divergenceFree', 'omniIncompressible', 'dfsphReference', 'band2018pb'}
    if set(pm['schemes']) != required:
        raise ValueError('Expected the complete eight-method prepared matrix')
    labels = dict(zip(SCHEMES, NAMES))
    labels.update(pm['labels'])
    schemes = list(RETAINED[:2]) + list(pm['schemes']) + [RETAINED[-1]]
    selected, plans = [], []
    for root, source_manifest, metrics, keep, cohort in (
        (baseline, bm, br, RETAINED, 'retained-classic'),
        (prepared, pm, pr, pm['schemes'], 'prepared-incompressible'),
    ):
        source_plans = {r['id']: r for r in source_manifest['rows']}
        present = {r['scheme'] for r in metrics}
        if not set(keep) <= present:
            raise ValueError(f'Missing requested methods in {root}')
        for original in metrics:
            if original['scheme'] not in keep:
                continue
            item = copy.deepcopy(original)
            source = Path(item.get('data_root', root))
            if not source.is_absolute():
                raise ValueError('Input per-row data_root must be absolute')
            item.update(data_root=str(source), run_date=item.get('run_date', source_manifest['created']),
                        comparison_cohort=cohort)
            expected = f"{item['test']}__{item['scheme']}__{item['boundary']}"
            if item['id'] != expected:
                raise ValueError(f'Noncanonical identity: {item["id"]}')
            if item['status'] in ('pending', 'running'):
                raise ValueError(f'Unfinished source queue: {item["id"]}')
            plan = copy.deepcopy(source_plans[item['id']])
            plan.update(data_root=str(source), run_date=item['run_date'], comparison_cohort=cohort)
            plans.append(plan)
            selected.append(item)
    ids = [r['id'] for r in selected]
    if len(ids) != len(set(ids)):
        raise ValueError('Input selection produced duplicate rows')
    slots = {(r['test'], r['boundary']) for r in plans}
    if any({(r['test'], r['boundary']) for r in plans if r['scheme'] == s} != slots for s in schemes):
        raise ValueError('Methods do not have identical declared case/boundary slots')
    manifest = dict(version=1, created=date.today().isoformat(), profile='classic-fluid-comparison',
                    artifact_type='comparison-only', comparison_title='Classic-fluid matrix — prepared incompressible results',
                    schemes=schemes, labels={s: labels[s] for s in schemes}, rows=plans,
                    qualification=QUALIFICATION,
                    sources=[dict(root=str(root), manifest_sha256=digest(root/'manifest.json'),
                                  metrics_sha256=digest(root/'comparison_metrics.json'))
                             for root in (baseline, prepared)])
    for key in ('comparison_title','comparison_notes','details_root','correction_sources'):
        if key in pm:manifest[key]=pm[key]
    return manifest, selected


def overlay_prepared(baseline, replacements, output):
    """Replace declared complete rows, including failed outcomes, never best-of runs."""
    baseline,replacements,output=baseline.resolve(),replacements.resolve(),output.resolve()
    manifest,old=load_dataset(baseline)
    changes,new=load_dataset(replacements)
    if manifest.get('profile')!='prepared-incompressible' or changes.get('profile')!='prepared-incompressible':
        raise ValueError('Expected prepared input datasets')
    if manifest.get('scale')!=changes.get('scale') or manifest.get('schemes')!=changes.get('schemes'):
        raise ValueError('Replacement scale/method declarations must match the full source')
    existing={r['id']:r for r in old};new_by_id={r['id']:r for r in new}
    if not set(new_by_id)<=set(existing):raise ValueError('Replacement contains unknown row IDs')
    plans={r['id']:r for r in changes['rows']}
    for row in new:
        original=existing[row['id']]
        if row['status'] in ('pending','running','unsupported'):
            raise ValueError('Replacement must be an attempted terminal result')
        if row['target_time']!=original['target_time'] or row['boundary']!=original['boundary']:
            raise ValueError('Replacement must retain endpoint and boundary identity')
    rows=[]
    for index,plan in enumerate(manifest['rows']):
        changed=plan['id'] in new_by_id;root=replacements if changed else baseline
        item=copy.deepcopy(new_by_id[plan['id']] if changed else existing[plan['id']])
        actual_plan=copy.deepcopy(plans[plan['id']] if changed else plan)
        item['data_root']=str(Path(item.get('data_root',root)).resolve())
        item['run_date']=item.get('run_date',changes['created'] if changed else manifest['created'])
        actual_plan.update(data_root=item['data_root'],run_date=item['run_date'])
        manifest['rows'][index]=actual_plan;rows.append(item)
    manifest.update(version=changes['version'],created=date.today().isoformat(),artifact_type='comparison-only',
                    revision='mixed-with-stability-v1',details_root=str(baseline),
                    comparison_title='Classic-fluid matrix — corrected shear, sloshing and VD+PS TGV',
                    comparison_notes=['26 Oct corrections: C4-calibrated Morris shear; sloshing dt ≤ 0.2 ms; native VD+PS TGV DF 0.001%. All replaced rows reuse the exact saved physical ICs.'],
                    correction_sources=[dict(root=str(r),manifest_sha256=digest(r/'manifest.json'),metrics_sha256=digest(r/'comparison_metrics.json')) for r in (baseline,replacements)],
                    replaced_ids=list(new_by_id))
    output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()):raise ValueError('Use an empty overlay destination')
    write_json(output/'manifest.json',manifest);write_json(output/'comparison_metrics.json',rows)
    csv_write(output/'comparison_metrics.csv',rows)
    return manifest,rows


def overlays(root, manifest, rows):
    """Plot actual completed and partial records; preserve genuine terminal spikes."""
    names = manifest['labels']
    colors = dict(zip(manifest['schemes'], plt.get_cmap('tab20').colors))
    panels = {
        'tgv': [('energy_ratio', 'E / E(0)')],
        'decayingVortices': [('energy_ratio', 'E / E(0)')],
        'shearDecay': [('energy_ratio', 'E / E(0)')],
        'boundedVortices': [('energy_ratio', 'E / E(0)')],
        'marrone31': [('pProbe0Star', 'P1 / ρgH'), ('pProbe1Star', 'P2 / ρgH')],
        'sloshing': [('sensorPressure', 'Sensor 1 pressure [Pa], 10 ms smoothing')],
        'hydrostatic': [('fluidKineticEnergy', 'Fluid kinetic energy'),
                        ('matrixPressureSlopeRatio', 'Pressure gradient / hydrostatic')],
        'englishWedge': [('fluidKineticEnergy', 'Fluid kinetic energy'),
                         ('matrixPressureSlopeRatio', 'Pressure gradient / hydrostatic')],
        'taylorCouette': [('profileAmplitude', 'Velocity profile / exact steady profile')],
    }
    images = []
    for test, quantities in panels.items():
        group = [r for r in rows if r['test'] == test and (Path(r['data_root'])/r['id']/'series.npz').exists()]
        boundaries = [b for b in ('periodic', 'particles', 'analytic') if any(r['boundary'] == b for r in group)]
        if not boundaries:
            continue
        fig, axes = plt.subplots(len(quantities), len(boundaries), figsize=(8*len(boundaries), 4*len(quantities)+1.5), squeeze=False)
        for j, boundary in enumerate(boundaries):
            for k, (key, label) in enumerate(quantities):
                ax = axes[k, j]
                for item in group:
                    if item['boundary'] != boundary:
                        continue
                    if key.startswith(('pProbe', 'sensor', 'matrixPressure')) and item.get('pressure_observation_complete') is False:
                        continue
                    arrays = series(root, item)
                    if key == 'energy_ratio':
                        y = arrays['fluidKineticEnergy']/arrays['fluidKineticEnergy'][0]
                    elif key in arrays:
                        y = arrays[key]
                    else:
                        continue
                    t = arrays['t']
                    if test == 'sloshing':
                        from plot_sloshingOverlay import smooth
                        t, y = smooth(t, y, .01)
                    ax.plot(t, y, color=colors[item['scheme']], linewidth=1.1,
                            linestyle='-' if item['status']=='completed' else '--',
                            label=names[item['scheme']])
                if test in ('tgv', 'shearDecay'):
                    t = np.linspace(0, max(r['target_time'] for r in group), 400)
                    rate = .04 if test=='tgv' else 2*.0185*(2*np.pi)**2
                    ax.plot(t, np.exp(-rate*t), 'k:', label='Analytic viscous decay')
                if key == 'matrixPressureSlopeRatio':
                    ax.axhline(1, color='black', linestyle=':')
                if key in ('energy_ratio', 'fluidKineticEnergy'):
                    ax.set_yscale('symlog', linthresh=1e-6)
                ax.set(xlabel='Time [s]', ylabel=label, title=boundary)
                ax.grid(alpha=.2)
        handles, labels = [], []
        for ax in axes.flat:
            hh, ll = ax.get_legend_handles_labels()
            for h, label in zip(hh, ll):
                if label not in labels:
                    handles.append(h); labels.append(label)
        fig.suptitle(f'{test} — dated classic controls and prepared projections', y=.99)
        fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=8)
        fig.text(.5, .14, 'Solid: completed; dashed: partial stop. Different ICs / kernels / presets. Retained controls: 21 Oct; prepared: 25–26 Oct 2026.', ha='center', fontsize=8)
        fig.tight_layout(rect=(0, .17, 1, .96))
        name = f'comparison_{test}.png'
        fig.savefig(root/name, dpi=150)
        plt.close(fig)
        images.append(name)
    return images


def report(root, manifest, rows, images):
    rankings = json.loads((root/'ranking/case_ranking.json').read_text())['rows']
    by_id = {r['id']: r for r in rankings}
    names = manifest['labels']
    lines = ['# Classic-fluid matrix with prepared incompressible results', '', QUALIFICATION, '',
             '![Per-case quality and cost](ranking/case_ranking.png)', '',
             '[Interactive report and curves](comparison.html) · [Exact rankings](ranking/case_ranking.json) · '
             '[All metrics](comparison_metrics.csv) · [Input settings and source roots](manifest.json) · [Provenance](provenance.json)', '',
             '| Method | Endpoints | Stops/errors | Unsupported |', '|---|---:|---:|---:|']
    for scheme in manifest['schemes']:
        counts = Counter(r['status'] for r in rows if r['scheme']==scheme)
        lines.append(f"| {names[scheme]} | {counts['completed']} | {counts['failed']+counts['error']} | {counts['unsupported']} |")
    lines.extend(['', *manifest.get('comparison_notes', []), ''])
    prepared = Path(manifest['sources'][1]['root'])
    details = Path(manifest.get('details_root',prepared))
    incomplete=sum(r.get('pressure_observation_complete') is False for r in rows)
    lines += ['', 'The prepared primary matrix includes unfinished-quietness fixtures, initial closed-box wall crossings, '
              'native iteration caps, periodic density drift and genuine shortened-final-step pressure responses. '
              f'These qualifications remain in the source report. Its {incomplete} incomplete legacy pressure records stay excluded '
              'from pressure metrics.', '',
              f'[Prepared report]({prepared}/README.md) · [Static-water details]({details}/STATIC_WATER.md) · '
              f'[Separate TGV preparation control]({details}/TGV_PREPARATION_CONTROL.md)', '',
              'The alternate TGV control does not replace a single failed primary cell. Historical roundoff exclusions '
              'in retained controls are carried forward exactly; new legitimate terminal samples are kept. '
              'Prepared runtime subtracts shared fixture creation; preparation cost is a separate metrics field. '
              'Runtime includes setup and output and is not an isolated solver benchmark.', '',
              '## Best declared case metrics', '', '| Case / wall | Method | Loss |', '|---|---|---:|']
    for r in rankings:
        if r.get('accuracy_rank') == 1:
            lines.append(f"| {r['test']} / {r['boundary']} | {names[r['scheme']]} | {r['loss']:.5g}% |")
    lines += ['', 'A best relative rank is not a physical pass. Vortex retention and static-water loss use the existing '
              'heuristics; numerical values across different cases are not a common accuracy scale. The ranking JSON '
              'contains the unchanged metric definitions. Rendering this comparison uses saved simulation records.', '']
    (root/'README.md').write_text('\n'.join(lines))
    esc = html.escape
    body = ['<!doctype html><meta charset="utf-8"><title>Classic-fluid matrix</title>',
            '<style>body{font:15px system-ui;margin:24px;color:#14242b}img{max-width:100%}table{border-collapse:collapse}td,th{padding:7px;border-bottom:1px solid #ddd;text-align:left}details{margin:1em 0}</style>',
            '<h1>Classic fluid comparison</h1>', f'<p>{esc(QUALIFICATION)}</p>',
            '<p><a href="README.md">Report</a> · <a href="comparison_metrics.csv">Metrics CSV</a> · '
            '<a href="manifest.json">Per-row input settings</a> · <a href="provenance.json">Source receipts</a></p>',
            '<p>Prepared fixture qualification and separate TGV control: '
            f'<a href="{esc(str(prepared/"README.md"))}">source report</a>. '
            'Iteration caps are marked *. Unsupported rows and stops are unscored. A rank does not establish physical validity.</p>']
    for image in images:
        body.append(f'<details><summary>{esc(image.removeprefix("comparison_").removesuffix(".png"))} curves</summary><a href="{image}"><img src="{image}"></a></details>')
    body.extend(f'<p>{esc(note)}</p>' for note in manifest.get('comparison_notes', []))
    body += ['<h2>All declared slots</h2><table><tr>'+''.join(f'<th>{s}</th>' for s in
             ('Case','Wall','Method','Run date','Cohort','Status','Time / target [s]','Quality loss','Rank','Runtime [s]','Evidence'))+'</tr>']
    for item in sorted(rows, key=lambda r:(r['test'],r['boundary'],manifest['schemes'].index(r['scheme']))):
        rank = by_id.get(item['id'], {})
        directory = Path(item['data_root'])/item['id']
        evidence = ' · '.join(f'<a href="{esc(str(directory/name))}">{label}</a>' for name,label in
                              (('result.json','result'),('series.csv','series'),('effective_spec.json','effective inputs'))
                              if (directory/name).exists())
        quality = f"{rank['loss']:.5g}%" if rank.get('loss') is not None else '—'
        runtime = f"{rank['runtime_seconds']:.5g}" if rank.get('runtime_seconds') is not None else '—'
        cells = [item['test'],item['boundary'],names[item['scheme']],item['run_date'],item['comparison_cohort'],item['status'],
                 f"{item.get('actual_time') or 0:.6g} / {item['target_time']:.6g}",quality,str(rank.get('accuracy_rank') or '—'),runtime]
        body.append('<tr>'+''.join(f'<td>{esc(str(v))}</td>' for v in cells)+f'<td>{evidence}</td></tr>')
    (root/'comparison.html').write_text('\n'.join(body)+ '</table>')


def build(baseline, prepared, output):
    manifest, rows = compose(baseline, prepared)
    output = output.resolve()
    # A separate empty destination prevents any accidental source/report overwrite.
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('Use an empty output directory; historical reports are preserved')
    write_json(output/'manifest.json', manifest)
    write_json(output/'comparison_metrics.json', rows)
    csv_write(output/'comparison_metrics.csv', rows)
    images = overlays(output, manifest, rows)
    # First render computes rank records for the textual report; second adds its link.
    plot(output, output/'ranking')
    report(output, manifest, rows, images)
    plot(output, output/'ranking')
    source_files = {}
    for item in rows:
        folder = Path(item['data_root'])/item['id']
        for name in ('series.npz','result.json'):
            path = folder/name
            if path.exists():
                source_files[str(path)] = digest(path)
    write_json(output/'provenance.json', dict(
        artifact_type='comparison-only', selected_rows=len(rows), status_counts=dict(Counter(r['status'] for r in rows)),
        source_inputs=manifest['sources'], source_files_sha256=source_files,
        generated_sha256={str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file()},
        tool_sha256={str(Path(__file__).resolve()):digest(Path(__file__)),
                     str(Path(__file__).with_name('plot_fluidMatrixRanking.py').resolve()):digest(Path(__file__).with_name('plot_fluidMatrixRanking.py'))},
        qualification=QUALIFICATION))
    (output/'KEEP').write_text('Comparison-only view; source datasets and failed runs remain untouched.\n')
    print(json.dumps(dict(report=str(output/'comparison.html'),rows=len(rows)),indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    build(args.baseline, args.prepared, args.out)
