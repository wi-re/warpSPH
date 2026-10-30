#!/usr/bin/env python3
"""Run a prepared, source-pinned serial GPU queue independently of the chat.

The manifest declares commands, environments, prerequisites and endpoints.
Failures block their dependents; independent methods continue. No retries,
numerical retuning or termination of unrelated GPU processes is performed.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, data):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data, indent=2)+'\n')
    temporary.replace(path)


def gpu_users():
    p = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                        '--format=csv,noheader'], capture_output=True, text=True, check=True)
    return p.stdout.strip().splitlines()


def terminate(child):
    if child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()


class AdoptedProcess:
    """Observe an existing detached child after replacing only its supervisor.

    A disappeared process has no retrievable wait status. Its physical result
    must pass the same endpoint checks before the adopted phase can succeed.
    """
    def __init__(self, pid, command):
        self.pid = pid
        self.returncode = None
        self.observed_returncode = None
        self.proc = Path('/proc')/str(pid)
        if self.proc.exists():
            actual = self.proc.joinpath('cmdline').read_bytes().rstrip(b'\0').split(b'\0')
            if actual != [os.fsencode(arg) for arg in command]:
                raise RuntimeError('Active PID does not match the saved command')
            self.start_ticks = self._stat()[19]
        else:
            self.start_ticks = None

    def _stat(self):
        return self.proc.joinpath('stat').read_text().rsplit(')', 1)[1].split()

    def poll(self):
        if self.returncode is not None:
            return self.returncode
        try:
            stat = self._stat()
            if self.start_ticks is not None and stat[19] != self.start_ticks:
                self.returncode = 0
            elif stat[0] == 'Z':
                self.observed_returncode = os.waitstatus_to_exitcode(int(stat[49]))
                self.returncode = self.observed_returncode
        except FileNotFoundError:
            self.returncode = 0
        return self.returncode

    def wait(self, timeout=None):
        start = time.monotonic()
        while self.poll() is None:
            if timeout is not None and time.monotonic()-start > timeout:
                raise subprocess.TimeoutExpired(str(self.pid), timeout)
            time.sleep(.1)
        return self.returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('manifest', type=Path)
    ap.add_argument('--adopt-active', action='store_true',
                    help='Resume saved queue state after replacing its supervisor; preserve its running child')
    args = ap.parse_args()
    manifest = args.manifest.resolve()
    root = manifest.parent
    plan = json.loads(manifest.read_text())
    # All executable source files are sealed before the supervisor starts.
    for name, digest in plan['sourceHashes'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise RuntimeError('Pinned source changed: '+name)
    if args.adopt_active:
        state = json.loads((root/'status.json').read_text())
        previous = state['supervisorPID']
        try:
            previous_stat = Path(f'/proc/{previous}/stat').read_text().rsplit(')', 1)[1].split()
        except FileNotFoundError:
            previous_stat = ['Z']
        if previous_stat[0] != 'Z':
            raise RuntimeError('Previous supervisor must have exited before adoption')
        state.update(previousSupervisorPID=previous, supervisorPID=os.getpid(), adopted=now())
    else:
        state = dict(started=now(), supervisorPID=os.getpid(), phase='starting', jobs=[])
    write(root/'status.json', state)
    (root/'supervisor.pid').write_text(str(os.getpid())+'\n')
    outcomes = {row['name']: row['status'] for row in state['jobs'] if row['status'] != 'running'}
    active = None
    def save(**kw):
        state.update(kw, updated=now())
        write(root/'status.json', state)
    def stopped(signum, frame):
        if active is not None:
            terminate(active)
        save(phase='stopped', finished=now(), childPID=None)
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM, stopped)
    signal.signal(signal.SIGINT, stopped)
    try:
        # Wait for the entire external serial queue, including its pending jobs.
        # GPU idleness alone can otherwise race the gap between two jobs.
        for dependency in plan.get('waitForQueues', []):
            directory = Path(dependency)
            expected = {job['name'] for job in json.loads((directory/'manifest.json').read_text())['jobs']}
            while True:
                upstream = json.loads((directory/'status.json').read_text())
                rows = {row['name']: row['status'] for row in upstream['jobs']}
                if (upstream['phase'] in ('complete', 'complete with failures')
                        and expected <= rows.keys()
                        and all(rows[name] in ('passed', 'failed', 'skipped') for name in expected)):
                    save(upstreamCompletion=dict(queue=str(directory), phase=upstream['phase'],
                                                finished=upstream.get('finished')))
                    break
                if upstream['phase'] in ('stopped', 'failed'):
                    save(phase='blocked by upstream queue', upstreamQueue=str(directory),
                         upstreamPhase=upstream['phase'], finished=now())
                    raise RuntimeError('Upstream queue stopped without finishing: '+str(directory))
                save(phase='waiting for upstream queue', upstreamQueue=str(directory),
                     upstreamPhase=upstream['phase'])
                time.sleep(20)
        for job in plan['jobs']:
            if job['name'] in outcomes:
                continue
            row = next((r for r in state['jobs'] if r['name'] == job['name'] and r['status'] == 'running'), None)
            adopting = row is not None
            if adopting:
                if not job.get('result') or row['command'] != job['command']:
                    raise RuntimeError('Adoption requires an unchanged command and validated physical result')
            else:
                row = dict(name=job['name'], status='pending', command=job['command'])
                state['jobs'].append(row)
            if any(outcomes.get(name) != 'passed' for name in job.get('requires', [])):
                row.update(status='skipped', reason='prerequisite failed or skipped')
                outcomes[job['name']] = row['status']
                save()
                continue
            if job.get('gpu') and not adopting:
                while users := gpu_users():
                    save(phase='waiting for GPU before '+job['name'], gpuUsers=users)
                    time.sleep(20)
                row['gpuUsersAtLaunch'] = []
            logpath = root/'logs'/(job['name']+'.log')
            logpath.parent.mkdir(exist_ok=True)
            env = dict(os.environ, **plan.get('environment', {}), **job.get('environment', {}))
            prior_wall = max(0., (datetime.now(timezone.utc)-datetime.fromisoformat(row['started'])).total_seconds()) if adopting else 0.
            start = time.monotonic()-prior_wall
            last = time.monotonic()
            previous_size = logpath.stat().st_size if adopting else 0
            row.update(status='running', log=str(logpath))
            if not adopting:
                row['started'] = now()
            with logpath.open('ab' if adopting else 'wb') as log:
                if adopting:
                    active = AdoptedProcess(row['pid'], job['command'])
                    row['adoptedBySupervisor'] = os.getpid()
                else:
                    active = subprocess.Popen(job['command'], cwd=job['cwd'], env=env,
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                row['pid'] = active.pid
                save(phase=job['name'], childPID=active.pid, gpuUsers=[])
                print('START', job['name'], active.pid, flush=True)
                while active.poll() is None:
                    time.sleep(5)
                    size = logpath.stat().st_size
                    if size != previous_size:
                        last = time.monotonic()
                        previous_size = size
                    row['wallSeconds'] = time.monotonic()-start
                    if row['wallSeconds'] > job.get('wallLimitSeconds', 28800):
                        row['stopReason'] = 'wall limit'
                        terminate(active)
                    elif time.monotonic()-last > job.get('silenceLimitSeconds', 600):
                        row['stopReason'] = 'no progress output'
                        terminate(active)
                    save()
                row.update(returncode=active.returncode, wallSeconds=time.monotonic()-start,
                           finished=now(), status='passed' if active.returncode == 0 else 'failed')
                if adopting:
                    row.update(returncode=active.observed_returncode,
                               completionBasis='process ended plus physical result/endpoint validation')
                active = None
            # Successful process exit alone does not prove a physical endpoint.
            if row['status'] == 'passed' and job.get('result'):
                try:
                    result = json.loads(Path(job['result']).read_text())
                    if job['kind'] == 'acsph':
                        valid = (result['nSteps'] == job['expectedSteps'] and not result['diverged']
                                 and not result['stop_reason'])
                        row['msPerStep'] = result['wall_seconds']*1000/result['nSteps']
                    else:
                        valid = result['status'] == 'completed' and result['actualTime'] == job['targetTime']
                        row['msPerStep'] = result['loopWallSeconds']*1000/result['nSteps']
                    row['resultSummary'] = {k: result[k] for k in ('nSteps', 'actualTime', 'maxMach',
                        'maxAbsoluteEnergyBudgetResidual', 'stageChecks', 'graph_disabled') if k in result}
                    if not valid:
                        row.update(status='failed', reason='result or endpoint validation failed')
                except Exception as error:
                    row.update(status='failed', reason='cannot validate result: '+str(error))
            outcomes[job['name']] = row['status']
            print('FINISH', job['name'], row['status'], row['wallSeconds'], flush=True)
            save(childPID=None)
        save(phase='complete' if all(v == 'passed' for v in outcomes.values()) else 'complete with failures',
             childPID=None, finished=now())
    finally:
        if active is not None:
            terminate(active)


if __name__ == '__main__':
    main()
