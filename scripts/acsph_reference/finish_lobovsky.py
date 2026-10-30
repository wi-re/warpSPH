#!/usr/bin/env python3
"""Continue a valid native trajectory if the campaign's real-step guard ends it.

Observe the original supervisor until it finishes the native/export phases.
Only its explicit short-endpoint condition is recovered: simulation errors are
left failed. Reuse frozen solver code and all actual recorded timesteps.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    partial = path.with_suffix('.tmp')
    partial.write_text(json.dumps(value, indent=2)+'\n')
    partial.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    args = ap.parse_args(); root = args.root.resolve()
    (root/'continuation.pid').write_text(str(os.getpid())+'\n')
    receipt = {'started': now(), 'phase': 'watching reference endpoint'}
    write(root/'continuation-status.json', receipt)
    while True:
        status = json.loads((root/'status.json').read_text())
        phase = status['phase']
        if phase == 'failed':
            break
        if phase not in ('native', 'export-native'):
            receipt.update(phase='not needed', observed_phase=phase, finished=now())
            write(root/'continuation-status.json', receipt)
            return
        time.sleep(15)
    if status.get('error') != 'Native phase did not reach the requested endpoint':
        receipt.update(phase='no recovery: native/export error', error=status.get('error'), finished=now())
        write(root/'continuation-status.json', receipt)
        return
    native = json.loads((root/'native-f64/result.json').read_text())
    if native['returncode'] or native['stop_reason'] or len(native['steps']) != 10000:
        raise RuntimeError('Unexpected native stop; refusing to treat it as the step guard')
    endpoint = native['steps'][-1]['t']
    if not (root/'dynGPUSPH-lobovsky.h5').exists():
        raise RuntimeError('Native viewer export is missing')
    write(root/'original-short-endpoint-status.json', status)
    status['reference_limit'] = {'reason': 'Native 10000-real-step guard',
        'requested_time': status['target_time'], 'actual_time': endpoint, 'steps': 10000}
    status['prior_error'] = status.pop('error')
    status.pop('finished', None)
    status['continuation_pid'] = os.getpid()
    frozen = root/'source'; scripts = frozen/'scripts/acsph_reference'
    env = os.environ.copy()
    env['PYTHONPATH'] = str(frozen/'src')+os.pathsep+env.get('PYTHONPATH', '')
    env['WARP_CACHE_PATH'] = str(root/'warp-cache')
    env['PYTHONUNBUFFERED'] = '1'
    def save():
        status['updated'] = now()
        write(root/'status.json', status)
    def command(phase, args):
        path = root/(phase+'.log')
        if path.exists():
            raise FileExistsError(path)
        row = {'phase': phase, 'started': now(), 'command': list(map(str, args)), 'log': str(path)}
        status['phases'].append(row)
        status['phase'] = phase
        receipt.update(phase=phase, reference_endpoint=endpoint)
        write(root/'continuation-status.json', receipt)
        start = last = time.monotonic(); size = 0
        with path.open('wb') as log:
            proc = subprocess.Popen(row['command'], cwd=frozen, env=env,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            row['pid'] = proc.pid; status['child_pid'] = proc.pid; save()
            while proc.poll() is None:
                time.sleep(10)
                current = path.stat().st_size
                if current != size:
                    size = current; last = time.monotonic()
                if time.monotonic()-last > 300 or time.monotonic()-start > 43200:
                    row['stop_reason'] = 'continuation progress/wall watchdog'
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                row['wall_seconds'] = time.monotonic()-start; save()
            row.update(returncode=proc.returncode, finished=now(), wall_seconds=time.monotonic()-start)
            save()
        return proc.returncode
    py = sys.executable; local = root/'warpsph-f32-aligned'
    try:
        local_code = command('warpsph', [py, '-u', scripts/'compare_lobovsky.py',
            '--reference', root/'native-f64', '--out', local, '--steps', '0', '--long-run', '--plot-interval', '10',
            '--precision', 'float32', '--position-form', 'increment', '--fused-residual',
            '--coefficients', 'mapped', '--history', 'source', '--wall-diffusion', 'source',
            '--acoustic-damper', 'source', '--rk-treatment', 'source', '--wall-velocity', 'source'])
        export_code = command('export-warpsph', [py, '-u', scripts/'export_viewer.py', local,
            '--out', root/'warpSPH-lobovsky.h5', '--label', 'warpSPH FP32 optimized, aligned reference controls'])
        analysis_code = command('analysis', [py, '-u', scripts/'analyze_trajectory.py',
            '--reference', root/'dynGPUSPH-lobovsky.h5', '--local', root/'warpSPH-lobovsky.h5', '--out', root/'analysis']) if not export_code else 1
        if local_code or export_code or analysis_code:
            raise RuntimeError('Replay/export/analysis failed; retained partial evidence')
        status.update(phase='complete', child_pid=None, finished=now())
        receipt.update(phase='complete with short native endpoint', reference_endpoint=endpoint, finished=now())
        save(); write(root/'continuation-status.json', receipt)
    except Exception as error:
        status.update(phase='failed', error=str(error), child_pid=None, finished=now()); save()
        receipt.update(phase='failed', error=str(error), finished=now())
        write(root/'continuation-status.json', receipt)
        raise


if __name__ == '__main__':
    main()
