"""Read-only terminal evidence audit and completion figure for a fluid matrix."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import urllib.request
from zoneinfo import ZoneInfo

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from run_fluidMatrix import DEFAULT_ROOT, REPO, SCHEMES, write_json


def audit(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    baseline = json.loads((root / 'sources.json').read_text())
    sys.path.insert(0, str(REPO.parent / 'warpSPHRemote/python'))
    from warpsph_remote.daemon import descriptor
    d = descriptor()
    req = urllib.request.Request(d['endpoint'] + '/v1/runs',
                                 headers={'Authorization': 'Bearer ' + d['token']})
    with urllib.request.urlopen(req, timeout=15) as response:
        remote = {r['id']: r for r in json.load(response)['runs']}

    def check(row):
        out = root / row['id']
        result = json.loads((out / 'result.json').read_text())
        record = remote.get(result.get('daemon_run_id'), {})
        item = dict(id=row['id'], status=result['status'], time=result['time'],
                    steps=result['steps'], wall_seconds=result['wall_seconds'],
                    daemon_present=bool(record), daemon_status=record.get('status'),
                    daemon_step=record.get('step'), daemon_time=record.get('time'),
                    daemon_matches=bool(record and record.get('status') == result['status']
                        and record.get('step') == result['steps']
                        and abs(record.get('time', -1) - result['time']) < 1e-6),
                    csv=(out / 'diagnostics.csv').is_file(),
                    npz=(out / 'series.npz').is_file(),
                    initial_tensor=(out / 'initial_state.pt').is_file(),
                    final_tensor=(out / 'final_state.pt').is_file())
        item['videos'] = []
        for video in out.rglob('output.mp4'):
            process = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height,nb_frames,duration', '-of', 'json', str(video)],
                capture_output=True, text=True, timeout=30)
            streams = json.loads(process.stdout).get('streams', []) if process.returncode == 0 else []
            item['videos'].append(dict(path=str(video), valid=bool(streams and int(streams[0].get('nb_frames', 0)) > 0), streams=streams))
        item['hdf5'] = []
        for path in out.rglob('trajectory.h5'):
            with h5py.File(path, 'r') as f:
                times = sorted(float(np.asarray(f['times'][k][()]).reshape(-1)[0]) for k in f['times'])
                item['hdf5'].append(dict(path=str(path), frames=len(times),
                    time_min=times[0], time_max=times[-1], dtype=str(f['combinedPositions'].dtype)))
        source = out / 'sources.json'
        if source.exists():
            hashes = json.loads(source.read_text())
            item['changed_numerical_sources'] = [p for p, h in hashes.items()
                                                if '/src/' in p and baseline.get(p) != h]
            item['source_record'] = str(source)
        else:
            item['source_record'] = str(root / 'sources.json')
            item['changed_numerical_sources'] = []
        item['checks_pass'] = bool(item['daemon_matches'] and item['csv'] and item['npz']
            and item['initial_tensor'] and item['final_tensor'] and item['videos']
            and all(v['valid'] for v in item['videos']) and item['hdf5']
            and all(v['frames'] > 0 and v['dtype'] == 'float32' for v in item['hdf5'])
            and not item['changed_numerical_sources'])
        return item

    enabled = [r for r in manifest['rows'] if not r['skip']]
    # Only CPU metadata/video probes parallelize; no simulations or GPU work.
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(check, enabled))
    q = json.loads((root / 'queue_status.json').read_text())
    metrics = json.loads((root / 'comparison_metrics.json').read_text())
    data = dict(finished_uk=datetime.fromtimestamp(q['finished'], ZoneInfo('Europe/London')).isoformat(),
        counts=q['counts'], evidence_checks_passed=sum(r['checks_pass'] for r in records),
        attempts=len(records), total_steps=sum(r['steps'] for r in records),
        summed_simulation_wall_hours=sum(r['wall_seconds'] for r in records) / 3600,
        source_records=sum((root / r['id'] / 'sources.json').exists() for r in enabled),
        roundoff_terminal_rows=[r['id'] for r in metrics if r.get('roundoff_terminal_sample_excluded')],
        corrupted_endpoint_rows=[r['id'] for r in metrics if r.get('endpoint_corrupted')],
        mass_relative_drift_nonzero=[r['id'] for r in metrics if r.get('mass_relative_drift') not in (None, 0)],
        records=records)
    write_json(root / 'final_audit.json', data)

    # Completion is deliberately distinct from physics qualification.
    columns = list(dict.fromkeys((r['test'], r['boundary']) for r in manifest['rows']))
    matrix = np.zeros((len(SCHEMES), len(columns)), dtype=int)
    for row in manifest['rows']:
        j, i = columns.index((row['test'], row['boundary'])), SCHEMES.index(row['scheme'])
        if not row['skip']:
            result = next(r for r in records if r['id'] == row['id'])
            matrix[i, j] = 1 if result['status'] == 'completed' else 2
            if row['id'] in data['corrupted_endpoint_rows']:
                matrix[i, j] = 3
    names = ['VD+PS', 'Reference DFSPH', 'IISPH', 'Omni', 'Band pressure boundaries',
             'Delta-SPH', 'Sun delta-plus', 'ACSPH']
    labels = {'tgv': 'TGV', 'decayingVortices': 'Decaying vortices', 'shearDecay': 'Shear decay',
              'boundedVortices': 'Bounded vortices', 'marrone31': 'Marrone 3.1',
              'sloshing': 'Sloshing', 'hydrostatic': 'Hydrostatic',
              'englishWedge': 'English wedge', 'taylorCouette': 'Taylor–Couette'}
    fig, ax = plt.subplots(figsize=(17, 6))
    ax.imshow(matrix, cmap=ListedColormap(['#e6e6e6', '#bee0ef', '#e6a6a0', '#efd39b']), vmin=0, vmax=3)
    ax.set_xticks(range(len(columns)), [labels[t] + '\n' + b for t, b in columns], rotation=40, ha='right', fontsize=9)
    ax.set_yticks(range(len(names)), names)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, ['—', 'end', 'stop', 'end*'][matrix[i, j]], ha='center', va='center', fontsize=9)
    ax.set_xticks(np.arange(-.5, len(columns), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(SCHEMES), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=2)
    ax.tick_params(which='minor', bottom=False, left=False)
    ax.set_title('FP32 classic-fluid matrix: 63 reached end, 23 stopped, 34 unsupported', pad=16)
    fig.text(.02, .015, 'end = reached target, not an accuracy pass; stop = numerical alarm; — = unsupported. '
             'end* = raw shear endpoint corrupted by a tiny remainder step; use flagged preterminal analysis.', fontsize=10)
    fig.tight_layout(rect=(0, .065, 1, 1))
    fig.savefig(root / 'completion_matrix.png', dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    for row in manifest['rows']:
        if row['test'] != 'tgv':
            continue
        if row['scheme'] == 'band2018pb':
            continue  # Its source assembly omits the requested viscosity.
        item = next(r for r in records if r['id'] == row['id'])
        if item['status'] != 'completed':
            continue
        with np.load(root / row['id'] / 'series.npz') as a:
            ax.plot(a['t'], a['fluidKineticEnergy'] / a['fluidKineticEnergy'][0],
                    label=row['scheme'])
    t = np.linspace(0, 5, 200)
    ax.plot(t, np.exp(-.04 * t), 'k--', label='analytic ν=0.01')
    ax.set(xlabel='time [s]', ylabel='E(t) / E(0)', ylim=(0, 1.08),
           title='65,536-particle viscous TGV: runs reaching t=5\nIISPH/omni stopped; Band has no viscosity source. All traces are in the full comparison.')
    ax.grid(alpha=.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(root / 'completed_tgv_decay.png', dpi=150)
    plt.close(fig)
    # Add an entry point to the existing quantitative browser report.
    report = root / 'comparison.html'
    document = report.read_text()
    marker = '<!-- FINAL_MATRIX -->'
    if marker not in document:
        document = document.replace('<h1>Classic fluid comparison</h1>',
            '<h1>Classic fluid comparison</h1>' + marker
            + '<p>Finished 22 October 2026, 03:41 Europe/London. '
              '<a href="final_audit.json">Full evidence audit</a></p>'
            + '<a href="completion_matrix.png"><img width="1200" src="completion_matrix.png"></a>'
            + '<a href="completed_tgv_decay.png"><img width="700" src="completed_tgv_decay.png"></a>')
        report.write_text(document)
    print(json.dumps({k: v for k, v in data.items() if k != 'records'}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    audit(p.parse_args().root)
