"""Rebuild two-panel compressible matrices and provenance tables from saved runs.

No Warp/Torch imports or simulations. Called by run_compressibleMatrix.py report.
"""
from __future__ import annotations

import csv
import html
import math
from pathlib import Path

from compressible_matrix import RECIPES, classify_outcome, read_json, rank, write_json


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def case_recipe(view, cid):
    """Case variants retain their source case's metric and explicit protocol note."""
    definition = view['cases'][cid]
    recipe = RECIPES.get(cid, RECIPES.get(definition['case'], 'See the saved case definition.'))
    return recipe + (' ' + definition['metricNote'] if definition.get('metricNote') else '')


def collect(view):
    lookup = {(c['caseId'], c['method']): c for c in view['cells']}
    if len(lookup) != len(view['cells']):
        raise ValueError('Comparison contains duplicate cells')
    rows, provenance = [], []
    for method in view['methods']:
        for cid, definition in view['cases'].items():
            cell = lookup.get((cid, method), {})
            path = Path(cell['result']) if cell.get('result') else None
            result = read_json(path) if path else {}
            source_cid = cell.get('sourceCaseId', cid)
            if result and (result.get('inputSHA256') != cell['inputSHA256'] or
                           result.get('caseId') != source_cid or result.get('method') != method):
                raise ValueError(f'Result/input mismatch: {path}')
            boundary = cell.get('boundaryTreatment', '')
            marker = cell.get('marker', '')
            metrics = result.get('metrics', {})
            status = classify_outcome(result, smoke=view['smoke'])
            if result.get('sourcesStable') is False:
                status = 'source_changed'
            target, actual = result.get('targetTime'), result.get('actualTime')
            fraction = min(1., max(0., actual/target)) if finite(actual) and finite(target) and target > 0 else None
            recipe = definition['quality']
            value = metrics.get(recipe['metric']) if recipe['metric'] else None
            quality = value*recipe['scale'] if finite(value) and value >= 0 else None
            runtime = metrics.get('loopWallSeconds')
            if status != 'completed' or view['smoke']:
                quality = None
            if status not in ('completed', 'completed_invalid') or view['smoke']:
                runtime = None
            if not finite(runtime) or (runtime is not None and runtime <= 0):
                runtime = None
            rows.append(dict(caseId=cid, method=method, status=status, runId=result.get('runId'),
                sourceCaseId=source_cid, boundaryTreatment=boundary, marker=marker,
                completionFraction=fraction, quality=quality, qualityMetric=recipe['metric'],
                diagnosticAmplitude=metrics.get('khAmplitude'), runtimeSeconds=runtime,
                qualityRank=None, runtimeRank=None, result=str(path) if path else None,
                metrics=metrics, error=result.get('error') or result.get('metricsError') or result.get('stopReason')))
            if path:
                source = path.parent/'provenance.json'
                meta = read_json(source) if source.exists() else {}
                repos = meta.get('repositories', {})
                frontend = repos.get('warpSPH', {})
                provenance.append(dict(caseId=cid, method=method, runId=result.get('runId'),
                    sourceCaseId=source_cid, boundaryTreatment=boundary,
                    date=result.get('created'), status=status, githubCommit=frontend.get('commit'),
                    githubURL=frontend.get('commitURL'), dirty=frontend.get('dirty'),
                    sourceSHA256=meta.get('sourceSHA256'), inputSHA256=cell['inputSHA256'],
                    coreCommit=repos.get('warpSPHCore', {}).get('commit'),
                    integratorsCommit=repos.get('warpSPHIntegrators', {}).get('commit'),
                    precision=meta.get('precision'), device=meta.get('deviceName'),
                    torch=meta.get('torch'), warp=meta.get('warp'),
                    configuration=str(path.parent/'input.json'), resolved=str(path.parent/'resolved.json'),
                    sourceArchive=str(path.parent/'sources.tar.gz'), result=str(path),
                    provenance=str(source)))
    for cid in view['cases']:
        selected = [r for r in rows if r['caseId'] == cid]
        for metric, rank_key in (('quality', 'qualityRank'), ('runtimeSeconds', 'runtimeRank')):
            ranks = rank({r['method']: r[metric] for r in selected})
            for row in selected:
                row[rank_key] = ranks.get(row['method'])
    return rows, provenance


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def render(view, rows, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.cm import ScalarMappable
    from matplotlib.patches import Patch, Rectangle
    import numpy as np

    cases, methods = list(view['cases']), list(view['methods'])
    lookup = {(r['caseId'], r['method']): r for r in rows}
    cmap = LinearSegmentedColormap.from_list('comparison', ['#2b8781', '#f1eab1', '#cd5d4d'])
    figure = plt.figure(figsize=(max(11., 2.9+1.55*len(cases)), max(9., 5.3+.57*len(methods))), facecolor='white')
    grids = figure.add_gridspec(2, 1, left=.20, right=.98, top=.80, bottom=.18, hspace=.45)
    for panel, (key, rank_key, title) in enumerate((
            ('quality', 'qualityRank', 'A  Quality: case-specific error · rank and raw value'),
            ('runtimeSeconds', 'runtimeRank', 'B  Loop wall time: completed endpoints · rank and seconds'))):
        ax = figure.add_subplot(grids[panel])
        for j, cid in enumerate(cases):
            candidates = [lookup[cid, mid][key] for mid in methods if lookup[cid, mid][key] is not None]
            logs = np.log10(np.maximum(candidates, 1e-15)) if candidates else []
            low, high = (float(min(logs)), float(max(logs))) if len(logs) else (0., 0.)
            for i, mid in enumerate(methods):
                row = lookup[cid, mid]
                value, ranking = row[key], row[rank_key]
                dark = False
                if ranking is not None:
                    fraction = (math.log10(max(value, 1e-15))-low)/(high-low) if high > low else 0.
                    color = '#bad8e5' if len(candidates) == 1 else cmap(fraction)
                    label = ('only' if len(candidates) == 1 else f'#{ranking}')+f'\n{value:.3g}'+(' s' if panel else '')
                    if row['status'] == 'completed_invalid':
                        label += ' †'
                elif row['status'] == 'completed':
                    color = '#edf1f2'
                    label = (f"diag\nA={row['diagnosticAmplitude']:.3g}" if panel == 0 and finite(row['diagnosticAmplitude'])
                             else 'no reference' if panel == 0 and not row['qualityMetric'] else 'missing metric')
                elif row['status'] == 'pending':
                    color, label = '#edf0f3', 'pending'
                elif row['status'] in ('unsupported', 'rejected'):
                    color, label = '#e6e8ec', 'unsupported' if row['status'] == 'unsupported' else 'rejected'
                elif row['status'] == 'smoke':
                    color, label = '#e6e8ec', 'SMOKE'
                elif row['status'] == 'completed_invalid':
                    color, label = '#e8bb8d', 'INVALID\nendpoint reached'
                elif row['status'] == 'endpoint_mismatch':
                    color, label = '#d6c8e8', 'ENDPOINT\novershoot'
                else:
                    color, dark = '#454d59', True
                    progress = row['completionFraction']
                    tag = ('SOURCE Δ' if row['status'] == 'source_changed' else
                           'ERROR' if row['status'] == 'error' else 'STOP')
                    label = tag + (f'\n{100*progress:.0f}% t' if progress is not None else '\n'+row['status'])
                if row.get('marker'):
                    first, separator, rest = label.partition('\n')
                    label = first + ' ' + row['marker'] + separator + rest
                ax.add_patch(Rectangle((j-.5, i-.5), 1, 1, facecolor=color, edgecolor='white', linewidth=1.5))
                ax.text(j, i, label, ha='center', va='center', fontsize=8,
                        color='white' if dark else '#1b3036', linespacing=1.4)
        labels = [(view['cases'][c]['label']+'\n'+view.get('casePrecisions', {}).get(c, 'float32').replace('float', 'FP'))
                  if not view['smoke'] else c+'\n3-step smoke' for c in cases]
        ax.set(xticks=range(len(cases)), yticks=range(len(methods)), xlim=(-.5, len(cases)-.5),
               ylim=(len(methods)-.5, -.5))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_yticklabels([view['methods'][m]['label'] for m in methods], fontsize=9)
        ax.tick_params(length=0, pad=8)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(title, loc='left', fontsize=11, fontweight='bold', pad=14)
    figure.suptitle('Compressible methods: fixed-resolution comparison', x=.20, y=.98,
                   ha='left', fontsize=19, fontweight='bold', color='#14282e')
    figure.text(.20, .94, 'Saved runs · independent column scales · quality and cost stay separate'
                + (' · SMOKE ONLY' if view['smoke'] else ''), fontsize=10, color='#506068')
    cb = figure.add_axes([.20, .89, .33, .015])
    bar = figure.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=cmap), cax=cb, orientation='horizontal')
    bar.set_ticks([0, 1], labels=['Best in column', 'Worst completed in column'])
    bar.ax.tick_params(length=0, labelsize=8)
    bar.outline.set_visible(False)
    figure.legend(handles=[Patch(facecolor='#454d59', label='Stopped / error: progress retained'),
        Patch(facecolor='#e8bb8d', label='Invalid state at completed endpoint (cost retained)'),
        Patch(facecolor='#d6c8e8', label='Endpoint overshoot (unranked)'),
        Patch(facecolor='#edf1f2', label='KH amplitude: diagnostic, unranked'),
        Patch(facecolor='#bad8e5', label='Only one completed candidate')],
        loc='upper right', bbox_to_anchor=(.98, .925), frameon=False, fontsize=8)
    notes = list(view.get('figureNotes', [])) + [
        '* PESPH-A uses experimental SwiftLocal entropy heating. † Endpoint reached with invalid states: timing retained, accuracy unranked.',
        'Headless loop time includes diagnostics/timestep work; setup/compilation, integrator timing and peak Torch GPU memory are separate metrics.',
        'Log-scaled column colors are relative, not pass/fail. Provenance, dates, Git commits and source snapshots are in separate tables.']
    # Long recipes belong in HTML/JSON; keep the figure readable as a matrix.
    figure.text(.20, .10, '\n'.join(notes), va='top', fontsize=8, color='#506068', linespacing=1.6)
    for suffix in ('png', 'svg', 'pdf'):
        figure.savefig(destination/f'matrix.{suffix}', dpi=160)
    plt.close(figure)


def report(view, destination):
    destination = Path(destination)
    if destination.exists():
        raise ValueError(f'Report directory already exists: {destination}; choose a new --out')
    rows, provenance = collect(view)
    destination.mkdir(parents=True)
    write_json(destination/'comparison.json', view)
    write_json(destination/'metrics.json', rows)
    write_json(destination/'metric_definitions.json', {cid: case_recipe(view, cid) for cid in view['cases']})
    write_csv(destination/'metrics.csv', [dict({k: v for k, v in r.items() if k != 'metrics'},
        **{'metric.'+k: v for k, v in r['metrics'].items()}) for r in rows])
    write_csv(destination/'provenance.csv', provenance)
    render(view, rows, destination)
    esc = lambda value: html.escape('' if value is None else str(value), quote=True)
    columns = ('caseId', 'sourceCaseId', 'method', 'boundaryTreatment', 'runId', 'date', 'githubCommit', 'dirty', 'coreCommit',
               'integratorsCommit', 'device', 'precision', 'status')
    header = ''.join('<th>'+esc(c)+'</th>' for c in columns)
    body = ''
    for item in provenance:
        body += '<tr>'
        for key in columns:
            value = esc(item.get(key))
            if key == 'githubCommit' and item.get('githubURL'):
                value = f'<a href="{esc(item["githubURL"])}">{value}</a>'
            body += '<td>'+value+'</td>'
        body += '</tr>'
    definitions = ''.join(f'<li><b>{esc(cid)}</b>: {esc(case_recipe(view, cid))}</li>' for cid in view['cases'])
    figure_notes = ''.join(f'<p>{esc(note)}</p>' for note in view.get('figureNotes', []))
    devices = {(p['device'], p['torch'], p['warp']) for p in provenance if p.get('device')}
    warning = '<p><b>Mixed hardware/runtime versions: inspect provenance before interpreting cost ranks.</b></p>' if len(devices) > 1 else ''
    page = f'''<!doctype html><meta charset="utf-8"><title>Compressible comparison</title>
<style>body{{font:15px system-ui;margin:30px;color:#183038}}img{{width:100%}}table{{border-collapse:collapse;font-size:12px}}
td,th{{padding:8px;border:1px solid #dce2e5;text-align:left}}th{{background:#edf2f3}}.scroll{{overflow:auto}}</style>
<h1>Compressible methods: fixed-resolution comparison</h1>{warning}
<p>Retained runs selected explicitly; no overall winner or convergence claim. Smoke attempts never receive endpoint ranks.
PESPH-A SwiftLocal is experimental. Missing KH references do not become accuracy scores.</p>
{figure_notes}<img src="matrix.png" alt="Quality and runtime comparison matrix">
<p><a href="matrix.pdf">PDF</a> · <a href="matrix.svg">SVG</a> · <a href="metrics.csv">All metrics</a> · <a href="provenance.csv">Provenance</a> ·
<a href="comparison.json">Exact comparison selection</a></p>
<h2>Metric definitions</h2><ul>{definitions}</ul>
<p>Runtime: measured headless loop wall time, including diagnostics and timestep selection. Initialization and compilation
are included in totalRunWallSeconds; warmup-excluded integrator times are separate. Peak memory covers Torch allocations,
including initialization, and does not claim to measure all Warp/driver memory. See each run's resolved.json and source archive.</p>
<h2>Provenance</h2><div class="scroll"><table><tr>{header}</tr>{body}</table></div>
'''
    (destination/'index.html').write_text(page)
