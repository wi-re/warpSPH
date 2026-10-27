"""Read-only terminal evidence audit and completion figure for a fluid matrix."""
import argparse
import hashlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import urllib.request
from zoneinfo import ZoneInfo

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from run_fluidMatrix import DEFAULT_ROOT, REPO, SCHEMES, write_json


def daemonMatches(result, record):
    if not record:
        return False
    if result['status'] in ('error', 'timeout'):
        # Exceptions have partial publications, not a returned terminal state.
        return record.get('status') in ('failed', 'timeout')
    return bool(record.get('status') == result['status']
                and record.get('step') == result.get('steps')
                and abs(record.get('time', -1) - result.get('time', -2)) < 1e-6)


def verifySourceArchive(out, hashes):
    expected = {}
    for path, digest in hashes.items():
        try:
            name = str(Path(path).relative_to(REPO.parent))
        except ValueError:
            name = 'external/' + path.lstrip('/')
        expected[name] = digest
    seen = set()
    with tarfile.open(out / 'sources.tar.gz', 'r:gz') as archive:
        for member in archive:
            if not member.isfile() or member.name not in expected or member.name in seen:
                return False
            digest = hashlib.sha256(archive.extractfile(member).read()).hexdigest()
            if digest != expected[member.name]:
                return False
            seen.add(member.name)
    return seen == set(expected)


def audit(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    baseline = json.loads((root / 'sources.json').read_text())
    archive_fix_file = root / 'verification/source_archive_fix.json'
    archive_fix = json.loads(archive_fix_file.read_text()) if archive_fix_file.exists() else {}
    stop_fix_file = root / 'verification/source_stop_policy_update.json'
    stop_fix = json.loads(stop_fix_file.read_text()) if stop_fix_file.exists() else {}
    inactive_file = root / 'verification/inactive_source_changes.json'
    inactive = json.loads(inactive_file.read_text()) if inactive_file.exists() else {}
    diagnostics_file=root/'verification/diagnostic_source_changes.json'
    diagnostic_changes=json.loads(diagnostics_file.read_text()) if diagnostics_file.exists() else {}
    seed_file=root/'verification/initialization_source_changes.json'
    seed_changes=json.loads(seed_file.read_text()) if seed_file.exists() else {}
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
        partial = result['status'] in ('error', 'timeout')
        initial_file=out/'initial.json'
        initial=result.get('initial') or (json.loads(initial_file.read_text()) if initial_file.exists() else {})
        item = dict(id=row['id'], status=result['status'], time=result.get('time'),
                    steps=result.get('steps'), wall_seconds=result.get('wall_seconds', result.get('elapsed_seconds', 0)),
                    partial_error_evidence=partial, error=result.get('error'),
                    daemon_present=bool(record), daemon_status=record.get('status'),
                    daemon_step=record.get('step'), daemon_time=record.get('time'),
                    daemon_matches=daemonMatches(result, record),
                    csv=(out / 'diagnostics.csv').is_file(),
                    npz=(out / 'series.npz').is_file(),
                    initial_tensor=(out / 'initial_state.pt').is_file(),
                    final_tensor=(out / 'final_state.pt').is_file())
        if not (out/'series.npz').exists():
            item.update(recorded_steps=0,setup_failure=True,videos=[],hdf5=[],
                checks_pass=False,evidence_note='Stopped before any initial physical diagnostic; see retained source archive and error log.')
            return item
        with np.load(out / 'series.npz') as arrays:
            item['last_recorded_time'] = float(arrays['t'][-1])
            item['recorded_steps'] = len(arrays['t'])-1
            item['initial_energy'] = float(arrays['fluidKineticEnergy'][0])
            item['initial_density_range'] = [float(arrays['fluidDensityMin'][0]), float(arrays['fluidDensityMax'][0])]
            item['initial_speed'] = float(arrays['maxVelocity'][0])
            item['first_step_speed'] = float(arrays['maxVelocity'][1]) if len(arrays['t']) > 1 else None
            item['first_step_time'] = float(arrays['t'][1]) if len(arrays['t']) > 1 else None
            item['mass_relative_drift'] = float(arrays['fluidMass'][-1] / arrays['fluidMass'][0] - 1)
            if 'stepLimitAtStart' in arrays:
                dt, limit = arrays['takenDt'], arrays['stepLimitAtStart']
                valid = np.isfinite(limit) & (dt > 0)
                item['timestep_samples'] = int(valid.sum())
                item['timestep_violations'] = int(np.sum(dt[valid] > limit[valid] * (1 + 1e-7) + 1e-10))
                item['maximum_dt_limit_ratio'] = float(np.max(dt[valid] / limit[valid])) if valid.any() else None
        effective = json.loads((out / 'effective_spec.json').read_text())
        item['n_h'] = effective['n_h']
        item['velocity_stop_ratio'] = effective.get('velocityAlarmStopRatio')
        termination = out / 'termination_policy.json'
        item['external_stop_policy'] = json.loads(termination.read_text()) if termination.exists() else None
        item['band_viscosity'] = initial.get('band_viscosity', False)
        calibration = out / 'initial_calibration.json'
        item['initial_calibration'] = json.loads(calibration.read_text()) if calibration.exists() else None
        if manifest.get('profile')=='prepared-incompressible':
            fixture=out/'fixture.json'
            item['shared_fixture']=json.loads(fixture.read_text()) if fixture.exists() else None
            if item['shared_fixture'] and item['initial_tensor']:
                import torch
                from types import SimpleNamespace
                from prepared_fluidMatrix import tensor_hash
                saved=torch.load(out/'initial_state.pt',map_location='cpu',weights_only=True)
                item['fixture_fields_match']=tensor_hash(SimpleNamespace(**saved))==item['shared_fixture']['fingerprint']
                if manifest.get('physical_fixture_source'):
                    original=torch.load(Path(manifest['physical_fixture_source'])/row['id']/'initial_state.pt',map_location='cpu',weights_only=True)
                    item['original_initial_fields_match']={k:torch.equal(saved[k],original[k])
                        for k in ('positions','velocities','masses','supports','kinds','densities','pressures')}
                    item['fixture_fields_match'] &= all(item['original_initial_fields_match'].values())
                if row.get('solver')=='dfsph' and row['params'].get('hydrostaticInit'):
                    seed_path=out/'pressure_seed.json';first_path=out/'first_step_state.pt'
                    item['pressure_seed_matches']=False
                    if seed_path.exists() and first_path.exists():
                        seed=json.loads(seed_path.read_text())
                        first=torch.load(first_path,map_location='cpu',weights_only=True)
                        dt=seed['actualFirstDt'];pressure=saved['pressures']
                        shift=row['ordering']=='np_df_shift';zero=torch.zeros_like(pressure)
                        item['pressure_seed_matches']=bool(item['first_step_time'] is not None
                            and abs(dt-item['first_step_time'])<=1e-12+1e-7*dt
                            and torch.equal(first['pressures'],pressure)
                            and torch.equal(first['densityPressureImpulse'],zero if shift else pressure*(dt*dt))
                            and torch.equal(first['divergencePressureImpulse'],pressure*dt if shift else zero))
                if row['test']=='boundedVortices':
                    def geometry(state):
                        x=state['positions'][state['kinds']==0].numpy()
                        length=row['spec']['L'];dx=length/row['spec']['nx']
                        depth=np.maximum(np.maximum(-x,x-length).max(axis=1),0)
                        return dict(outside_particles=int((depth>1e-6*dx).sum()),
                                    maximum_penetration_dx=float(depth.max()/dx))
                    # Physical qualifications are distinct from evidence
                    # integrity: retain defects in the prepared input too.
                    item['closed_geometry_initial']=geometry(saved)
                    if item['final_tensor']:
                        final=torch.load(out/'final_state.pt',map_location='cpu',weights_only=True)
                        item['closed_geometry_final']=geometry(final)
        if manifest.get('version', 1) >= 3:
            cal = item['initial_calibration'] or {}
            item['setup_matches_manifest'] = bool(item['n_h'] == row['spec']['n_h']
                and cal.get('normalized') == row['spec']['calibrateNormalization']
                and cal.get('n_h') == row['spec']['n_h']
                and item['band_viscosity'] == bool(row.get('scheme_settings', {}).get('bandViscosity', False)))
            if row.get('solver') in ('dfsph','iisph') and row.get('initialization'):
                item['setup_matches_manifest'] &= initial.get('density_source_policy')==row['density_source_policy']
                if row.get('ordering'):item['setup_matches_manifest'] &= initial.get('dfsph_ordering')==row['ordering']
            if row['test'] in ('hydrostatic', 'englishWedge', 'sloshing'):
                item['hydrostatic_seed'] = initial.get('pressure_seed')
                item['setup_matches_manifest'] &= bool(item['hydrostatic_seed'])
            if row['test'] == 'englishWedge' and row['boundary'] == 'analytic':
                item['analytic_bodies'] = initial.get('analytic_bodies')
                item['setup_matches_manifest'] &= item['analytic_bodies'] == 1
        item['videos'] = []
        for video in out.rglob('output.mp4'):
            process = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height,nb_frames,duration', '-of', 'json', str(video)],
                capture_output=True, text=True, timeout=30)
            streams = json.loads(process.stdout).get('streams', []) if process.returncode == 0 else []
            decode = subprocess.run(['ffmpeg', '-v', 'error', '-threads', '1', '-i', str(video), '-f', 'null', '-'],
                                    capture_output=True, text=True, timeout=120)
            item['videos'].append(dict(path=str(video), valid=bool(streams and int(streams[0].get('nb_frames', 0)) > 0),
                                       fully_decoded=decode.returncode == 0, decode_error=decode.stderr or None, streams=streams))
        item['hdf5'] = []
        for path in out.rglob('trajectory.h5'):
            with h5py.File(path, 'r') as f:
                times = sorted(float(np.asarray(f['times'][k][()]).reshape(-1)[0]) for k in f['times'])
                item['hdf5'].append(dict(path=str(path), frames=len(times),
                    time_min=times[0], time_max=times[-1], dtype=str(f['combinedPositions'].dtype)))
                preset=row.get('numerical_preset',{})
                if preset:
                    checks=[]
                    if preset.get('viscosityPolicy')=='case':
                        dp=f['config/schemeConfig/diffusionParams'].attrs
                        params=effective['params']
                        inviscid=(False if row['test']=='sloshing' else params.get('inviscid',True))
                        checks.extend([bool(dp['inviscid'])==inviscid,
                            initial.get('actual_inviscid')==inviscid])
                        if inviscid:
                            alpha=params.get('alpha')
                            if alpha is None:alpha=.01
                            checks.append(abs(float(dp['inviscidAlpha'])-alpha)<1e-12)
                        else:
                            checks.append(abs(float(dp['viscidNu'])-params['nu'])<1e-12)
                        if 'artificialViscositySoundSpeed' in preset:
                            checks.append(float(f['config/schemeConfig'].attrs['fixedSoundSpeed'])==preset['artificialViscositySoundSpeed'])
                    if 'viscosityTerm' in preset:
                        dp=f['config/schemeConfig/diffusionParams'].attrs
                        expected=preset['targetKinematicViscosity']/preset['viscosityCalibration']['factor']
                        checks.extend([dp['viscousTerm']==preset['viscosityTerm'],
                            abs(float(dp['viscidNu'])-expected)<1e-10,
                            initial.get('actual_viscosity_term')==preset['viscosityTerm'],
                            initial.get('viscosity_calibration')==preset['viscosityCalibration']])
                    if 'divergenceTolerancePercent' in preset:
                        checks.append(float(f['config/schemeConfig/dfsph/divergence'].attrs['maxErrorPercent'])==preset['divergenceTolerancePercent'])
                    if 'stepCeilingSeconds' in preset:
                        checks.extend([effective['maxDt']==preset['stepCeilingSeconds'],
                            float(f['config/config'].attrs['maxDt'])==preset['stepCeilingSeconds']])
                    item['runtime_preset_matches']=all(checks)
                    item['setup_matches_manifest'] &= item['runtime_preset_matches']
        source = out / 'sources.json'
        if source.exists():
            hashes = json.loads(source.read_text())
            changed = [p for p, h in hashes.items() if baseline.get(p) != h]
            allowed_archive_update = [p for p in changed if p == archive_fix.get('path')
                and baseline.get(p) == archive_fix.get('old_sha256')
                and hashes[p] == archive_fix.get('new_sha256')
                and archive_fix.get('numerical_contract_unchanged')]
            allowed_stop_update = [p for p in changed if p == stop_fix.get('path')
                and baseline.get(p) == archive_fix.get('old_sha256')
                and stop_fix.get('old_sha256') == archive_fix.get('new_sha256')
                and hashes[p] == stop_fix.get('new_sha256')
                and stop_fix.get('only_stop_rule_changed')]
            item['source_archive_writer_update'] = allowed_archive_update
            item['source_stop_policy_update'] = allowed_stop_update
            allowed_inactive = [p for p in changed if inactive.get('reviewed_inactive')
                and p in inactive.get('files', {})
                and baseline.get(p) == inactive['files'][p]['baseline_sha256']
                and hashes[p] in inactive['files'][p]['observed_sha256']]
            allowed_diagnostics=[p for p in changed if diagnostic_changes.get('reviewed_only_diagnostics')
                and p in diagnostic_changes.get('files',{})
                and baseline.get(p)==diagnostic_changes['files'][p]['baseline_sha256']
                and hashes[p] in diagnostic_changes['files'][p]['observed_sha256']]
            allowed_seed=[p for p in changed if seed_changes.get('reviewed_initialization_bugfix')
                and p in seed_changes.get('files',{})
                and baseline.get(p)==seed_changes['files'][p]['baseline_sha256']
                and hashes[p] in seed_changes['files'][p]['observed_sha256']]
            item['initialization_source_changes']=allowed_seed
            item['diagnostic_source_changes']=allowed_diagnostics
            item['changed_source_files'] = changed
            item['changed_inactive_sources'] = allowed_inactive
            item['changed_numerical_sources'] = [p for p in changed if p not in allowed_archive_update + allowed_stop_update + allowed_inactive + allowed_diagnostics + allowed_seed]
            item['source_record'] = str(source)
            item['source_archive_verified'] = verifySourceArchive(out, hashes)
        else:
            item['source_record'] = str(root / 'sources.json')
            item['changed_numerical_sources'] = []
            item['source_archive_verified'] = verifySourceArchive(root, baseline)
        item['checks_pass'] = bool(item['daemon_matches'] and item['csv'] and item['npz']
            and item['initial_tensor'] and (item['final_tensor'] or (partial and item['error'])) and item['videos']
            and all(v['valid'] and v['fully_decoded'] for v in item['videos']) and item['hdf5']
            and all(v['frames'] > 0 and v['dtype'] == 'float32' for v in item['hdf5'])
            and item['source_archive_verified'] and not item['changed_numerical_sources'] and not item.get('timestep_violations', 0)
            and item.get('fixture_fields_match',True) and item.get('setup_matches_manifest', True) and abs(item['mass_relative_drift']) < 1e-7)
        item['checks_pass'] &= item.get('pressure_seed_matches',True)
        return item

    enabled = [r for r in manifest['rows'] if not r['skip']]
    # Only CPU metadata/video probes parallelize; no simulations or GPU work.
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(check, enabled))
    q = json.loads((root / 'queue_status.json').read_text())
    metrics = json.loads((root / 'comparison_metrics.json').read_text())
    data = dict(finished_uk=datetime.fromtimestamp(q['finished'], ZoneInfo('Europe/London')).isoformat(),
        counts=q['counts'], evidence_checks_passed=sum(r['checks_pass'] for r in records),
        attempts=len(records), total_steps_known=sum(r['steps'] or r['recorded_steps'] for r in records),
        steps_are_lower_bound_for_partial_errors=any(r['partial_error_evidence'] for r in records),
        partial_error_rows=[r['id'] for r in records if r['partial_error_evidence']],
        missing_terminal_tensors=[r['id'] for r in records if not r['final_tensor']],
        summed_simulation_wall_hours=sum(r['wall_seconds'] for r in records) / 3600,
        source_records=sum((root / r['id'] / 'sources.json').exists() for r in enabled),
        roundoff_terminal_rows=[r['id'] for r in metrics if r.get('roundoff_terminal_sample_excluded')],
        corrupted_endpoint_rows=[r['id'] for r in metrics if r.get('endpoint_corrupted')],
        mass_relative_drift_nonzero=[r['id'] for r in metrics if r.get('mass_relative_drift') not in (None, 0)],
        timestep_samples=sum(r.get('timestep_samples', 0) for r in records),
        timestep_violations=sum(r.get('timestep_violations', 0) for r in records),
        setup_mismatches=[r['id'] for r in records if not r.get('setup_matches_manifest', True)],
        records=records)
    write_json(root / 'final_audit.json', data)

    # Completion is deliberately distinct from physics qualification.
    columns = list(dict.fromkeys((r['test'], r['boundary']) for r in manifest['rows']))
    schemes = manifest.get('schemes',[s for s in SCHEMES if any(r['scheme']==s for r in manifest['rows'])])
    matrix = np.zeros((len(schemes), len(columns)), dtype=int)
    for row in manifest['rows']:
        j, i = columns.index((row['test'], row['boundary'])), schemes.index(row['scheme'])
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
    ax.set_yticks(range(len(schemes)), [manifest.get('labels',{}).get(s,dict(zip(SCHEMES,names)).get(s,s)) for s in schemes])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, ['—', 'end', 'stop', 'end*'][matrix[i, j]], ha='center', va='center', fontsize=9)
    ax.set_xticks(np.arange(-.5, len(columns), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(schemes), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=2)
    ax.tick_params(which='minor', bottom=False, left=False)
    counts = q['counts']
    stops = sum(v for k, v in counts.items() if k not in ('completed', 'unsupported'))
    ax.set_title(f"FP32 classic-fluid matrix: {counts.get('completed', 0)} reached end, {stops} stopped, "
                 f"{counts.get('unsupported', 0)} unsupported", pad=16)
    fig.text(.02, .015, 'end = reached target, not an accuracy pass; stop = numerical alarm; — = unsupported. '
             'end* = raw shear endpoint corrupted by a tiny remainder step; use flagged preterminal analysis.', fontsize=10)
    fig.tight_layout(rect=(0, .065, 1, 1))
    fig.savefig(root / 'completion_matrix.png', dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    for row in manifest['rows']:
        if row['test'] != 'tgv':
            continue
        if row['scheme'] == 'band2018pb' and not row.get('scheme_settings', {}).get('bandViscosity', False):
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
           title='65,536-particle viscous TGV: runs reaching t=5\nStopped traces and settings remain in the full comparison.')
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
            + f"<p>Finished {data['finished_uk']}. "
              '<a href="final_audit.json">Full evidence audit</a></p>'
            + '<a href="completion_matrix.png"><img width="1200" src="completion_matrix.png"></a>'
            + '<a href="completed_tgv_decay.png"><img width="700" src="completed_tgv_decay.png"></a>')
        report.write_text(document)
    print(json.dumps({k: v for k, v in data.items() if k != 'records'}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    audit(p.parse_args().root)
