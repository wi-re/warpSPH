"""Sequential baseline campaign; stop on failure/Remote Stop, preserve evidence.

Can wait for an already running reference process, then resume from completed
stage artifacts. This is a normal foreground study driver, not a scheduler.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--wait-for-pid', type=int)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--cases', nargs='+', default=['box-equal', 'sod-equal', 'sod-unequal', 'box-unequal'])
    a = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    root = a.input.resolve()
    reference = repo/'results/materialEOS/spheral-build/install/bin/spheral'
    script = repo/'scripts/compare_spheralMaterials.py'
    manifest = dict(status='running', pid=os.getpid(), stages=[], threads=a.threads,
        policy='Sequential physical trajectories, stop on first failure/Remote Stop. No production hydro modifications.')
    def save():
        (root/'study.json').write_text(json.dumps(manifest, indent=2)+'\n')
    if a.wait_for_pid:
        manifest['status'] = 'waitingForExistingReference'; manifest['waitingPID'] = a.wait_for_pid; save()
        print('Waiting for existing reference PID', a.wait_for_pid, flush=True)
        while True:
            try: os.kill(a.wait_for_pid, 0)
            except ProcessLookupError: break
            time.sleep(5)
        if not (root/'box-equal/spheral/comparisonRun.json').exists():
            manifest['status'] = 'failed'; manifest['error'] = 'Existing reference stopped without a completed summary'; save(); return 1
    env = dict(os.environ, OMP_NUM_THREADS=str(a.threads), OPENBLAS_NUM_THREADS='1')
    for case in a.cases:
        directory = root/case
        for mode in ('spheral', 'render', 'warp'):
            stage = dict(case=case, mode=mode); manifest['stages'].append(stage)
            result = directory/('spheral' if mode=='render' else mode)/('output.mp4' if mode=='render' else 'comparisonRun.json')
            if result.exists():
                if mode != 'render' and not json.loads(result.read_text())['passed']:
                    stage['status'] = 'failed'; manifest['status'] = 'failed'; save(); return 1
                stage['status'] = 'existing'; save(); continue
            command = ([str(reference), '-u'] if mode=='spheral' else [sys.executable]) + [str(script), mode, '--output', str(directory)]
            log = root/(case+'-'+mode+'.log')
            stage.update(status='running', command=command, log=str(log)); manifest['status'] = 'running'; save()
            print('Starting', case, mode, 'log:', log, flush=True)
            start = time.perf_counter()
            with log.open('w') as stream:
                child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env, cwd=repo)
                stage['pid'] = child.pid; save()
                code = child.wait()
            stage.update(returncode=code, wallSeconds=time.perf_counter()-start, status='completed' if code==0 else 'failed'); save()
            print('Finished', case, mode, 'returncode', code, 'wall', stage['wallSeconds'], flush=True)
            if code:
                # Retain a viewable partial reference when it failed numerically.
                if mode=='spheral' and (directory/'spheral/trajectory.h5').exists():
                    subprocess.run([sys.executable, str(script), 'render', '--output', str(directory)], env=env, cwd=repo)
                manifest['status'] = 'failed'; save(); return code
    manifest['status'] = 'completed'; save(); return 0


if __name__ == '__main__':
    raise SystemExit(main())
