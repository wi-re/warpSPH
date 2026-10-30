#!/usr/bin/env python3
"""Serial, restart-independent Lobovsky reference/replay/export campaign.

Wait for existing compute users unless sharing is explicitly authorized;
never stop another task. Freeze Python sources
before waiting so concurrent development cannot change the queued comparison.
Each child has progress/stall/wall limits and a separate retained log.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


def now():
    return datetime.now(timezone.utc).isoformat()


def freeze_runtime(root):
    repo = Path(__file__).resolve().parents[2]
    frozen = root/'source'; src = frozen/'src'
    src.mkdir(parents=True)
    ignore = shutil.ignore_patterns('__pycache__', '*.egg-info', '*.pyc')
    shutil.copytree(repo/'src', src, dirs_exist_ok=True, ignore=ignore)
    for package in ('warpSPHCore', 'warpSPHPlotting', 'warpSPHIntegrators'):
        spec = importlib.util.find_spec(package)
        source = Path(spec.origin).parent.parent
        shutil.copytree(source, src, dirs_exist_ok=True, ignore=ignore)
    shutil.copytree(Path(__file__).parent, frozen/'scripts/acsph_reference', ignore=ignore)
    # The comparison's field provenance resolves paths relative to this root.
    receipts = {}
    for path in sorted(frozen.rglob('*')):
        if path.is_file():
            receipts[str(path.relative_to(frozen))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (root/'source-sha256.json').write_text(json.dumps(receipts, indent=2)+'\n')
    (root/'working-tree.patch').write_bytes(subprocess.check_output(['git', 'diff', 'HEAD'], cwd=repo))
    (root/'working-tree-status.txt').write_bytes(subprocess.check_output(['git', 'status', '--short'], cwd=repo))
    return frozen


def gpu_users():
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
        '--format=csv,noheader'], capture_output=True, text=True, check=True)
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--binary', type=Path, required=True)
    ap.add_argument('--end-time', type=float, default=2.)
    ap.add_argument('--phase-wall-limit', type=float, default=43200.)
    ap.add_argument('--allow-concurrent-gpu', action='store_true', help='Explicitly share the GPU with other tasks; timings are contended')
    ap.add_argument('--resume-waiting', action='store_true', help='Reuse a frozen campaign that has not launched any child phase; stop its old supervisor first')
    args = ap.parse_args()
    root = args.root.resolve()
    if args.resume_waiting:
        status = json.loads((root/'status.json').read_text())
        if status['phases'] or not status['phase'].startswith('waiting for GPU'):
            raise SystemExit('Only an idle, never-launched campaign can be resumed this way')
        old_pid = int((root/'supervisor.pid').read_text())
        try:
            os.kill(old_pid, 0)
        except ProcessLookupError:
            pass
        else:
            raise SystemExit('Old queue supervisor is still alive')
        if status['target_time'] != args.end_time:
            raise SystemExit('Cannot change the frozen campaign endpoint during queue resume')
        frozen = root/'source'
        for name, digest in json.loads((root/'source-sha256.json').read_text()).items():
            if hashlib.sha256((frozen/name).read_bytes()).hexdigest() != digest:
                raise SystemExit('Frozen source hash changed: '+name)
        status['resumed'] = now()
    else:
        root.mkdir(parents=True, exist_ok=False)
        status = {'started': now(), 'target_time': args.end_time, 'phase': 'freezing sources', 'phases': []}
        frozen = freeze_runtime(root)
    status['allow_concurrent_gpu'] = args.allow_concurrent_gpu
    status['timing_context'] = 'shared GPU; not isolated performance measurements' if args.allow_concurrent_gpu else 'GPU idle at each phase launch'
    (root/'supervisor.pid').write_text(str(os.getpid())+'\n')
    (root/'KEEP').write_text('Retain native/replay trajectories, source snapshot, logs, failures and analysis.\n')
    shutil.copy2(__file__, root/'launch-supervisor.py')
    def save(**values):
        status.update(values, updated=now())
        partial = root/'status.json.tmp'
        partial.write_text(json.dumps(status, indent=2)+'\n')
        partial.replace(root/'status.json')
    def wait_gpu(phase):
        if args.allow_concurrent_gpu:
            return
        while True:
            users = gpu_users()
            if not users:
                return
            save(phase='waiting for GPU before '+phase, gpu_users=users)
            print('CAMPAIGN WAIT', phase, json.dumps(users), flush=True)
            time.sleep(30)
    scripts = frozen/'scripts/acsph_reference'
    env = os.environ.copy()
    env['PYTHONPATH'] = str(frozen/'src') + os.pathsep + env.get('PYTHONPATH', '')
    env['PYTHONUNBUFFERED'] = '1'
    env['WARP_CACHE_PATH'] = str(root/'warp-cache')
    def command(phase, arguments, gpu=False):
        if gpu:
            wait_gpu(phase)
        log_path = root/(phase+'.log')
        row = {'phase': phase, 'command': list(map(str, arguments)), 'started': now(), 'log': str(log_path)}
        if gpu:
            row['gpu_users_at_launch'] = gpu_users()
        status['phases'].append(row)
        start = last = time.monotonic(); size = 0
        with log_path.open('wb') as log:
            proc = subprocess.Popen(row['command'], cwd=frozen, env=env, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
            row['pid'] = proc.pid
            save(phase=phase, child_pid=proc.pid, gpu_users=[])
            print('CAMPAIGN START', phase, 'pid', proc.pid, flush=True)
            while proc.poll() is None:
                time.sleep(10)
                current = log_path.stat().st_size
                if current != size:
                    size = current; last = time.monotonic()
                elapsed = time.monotonic()-start
                if elapsed > args.phase_wall_limit or time.monotonic()-last > 300:
                    row['stop_reason'] = 'phase wall limit' if elapsed > args.phase_wall_limit else 'no child progress for 300 s'
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                row['wall_seconds'] = time.monotonic()-start
                save()
            row.update(returncode=proc.returncode, wall_seconds=time.monotonic()-start, finished=now())
            save()
        print('CAMPAIGN FINISH', phase, json.dumps(row), flush=True)
        return proc.returncode
    py = sys.executable
    reference, local = root/'native-f64', root/'warpsph-f32-aligned'
    try:
        code = command('native', [py, '-u', scripts/'run.py', 'dambreak_Lobovsky_400_2L',
            '--binary', args.binary.resolve(), '--out', reference, '--steps', '10000',
            '--t-limit', str(args.end_time), '--wall-limit', str(args.phase_wall_limit-60),
            '--stall-seconds', '300', '--max-speed', '100'], gpu=True)
        # Retain a viewer recording even if the native trajectory ends early.
        export_code = command('export-native', [py, '-u', scripts/'export_viewer.py', reference,
            '--out', root/'dynGPUSPH-lobovsky.h5', '--label', 'dynGPUSPH 2dbc27a FP64 supplied Lobovsky'])
        if code or export_code:
            raise RuntimeError('Native phase failed; retained partial evidence, no full-run comparison claimed')
        native_result = json.loads((reference/'result.json').read_text())
        if not native_result['steps'] or native_result['steps'][-1]['t'] < args.end_time:
            raise RuntimeError('Native phase did not reach the requested endpoint')
        code = command('warpsph', [py, '-u', scripts/'compare_lobovsky.py',
            '--reference', reference, '--out', local, '--steps', '0', '--long-run', '--plot-interval', '10',
            '--precision', 'float32', '--position-form', 'increment', '--fused-residual',
            '--coefficients', 'mapped', '--history', 'source', '--wall-diffusion', 'source',
            '--acoustic-damper', 'source', '--rk-treatment', 'source', '--wall-velocity', 'source'], gpu=True)
        export_code = command('export-warpsph', [py, '-u', scripts/'export_viewer.py', local,
            '--out', root/'warpSPH-lobovsky.h5', '--label', 'warpSPH FP32 optimized, aligned reference controls'])
        analysis_code = command('analysis', [py, '-u', scripts/'analyze_trajectory.py',
            '--reference', root/'dynGPUSPH-lobovsky.h5', '--local', root/'warpSPH-lobovsky.h5', '--out', root/'analysis']) if not export_code else 1
        if code or export_code or analysis_code:
            raise RuntimeError('Local/export/analysis phase failed; inspect retained logs and common endpoint')
        save(phase='complete', child_pid=None, finished=now())
    except Exception as error:
        save(phase='failed', error=str(error), child_pid=None, finished=now())
        raise


if __name__ == '__main__':
    main()
