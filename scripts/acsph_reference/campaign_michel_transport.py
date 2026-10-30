#!/usr/bin/env python3
"""Frozen, serial original-IC Michel/remap/BDF comparison after the prior queue."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from campaign_late_ringing import atomic,birth,gpu_users,now,terminate_owned_tree
from michel_trials import rows


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--prepare',action='store_true');args=ap.parse_args();root=args.root.resolve()
    if args.prepare:
        repo=Path(__file__).resolve().parents[2];prior=repo/'.tmp/acsph-ablation-2026-10-29'
        original=json.loads((prior/'manifest.json').read_text());root.mkdir(parents=True,exist_ok=False)
        for name,digest in original['sourceSHA256'].items():assert sha(prior/'source'/name)==digest,name
        # Freeze the integrated checkouts, not the old pre-merge dependencies.
        # Long runs must be insulated from concurrent chunk/multi-lane work.
        from campaign_lobovsky import freeze_runtime
        repositories={}
        for name,path in [('warpSPH',repo)]+[(name,Path(importlib.util.find_spec(name).origin).parent.parent.parent)
                for name in ('warpSPHCore','warpSPHIntegrators','warpSPHPlotting')]:
            repositories[name]=dict(path=str(path),commit=subprocess.check_output(
                ['git','rev-parse','HEAD'],cwd=path,text=True).strip(),
                status=subprocess.check_output(['git','status','--short'],cwd=path,text=True))
        assert Path(repositories['warpSPHCore']['path']).resolve()==repo.parent/'warpSPHCore'
        freeze_runtime(root)
        remote=Path(importlib.util.find_spec('warpsph_remote').origin).parent
        shutil.copytree(remote,root/'source/src/warpsph_remote',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        remote_repo=remote.parent.parent
        repositories['warpsph_remote']=dict(path=str(remote_repo),commit=subprocess.check_output(
            ['git','rev-parse','HEAD'],cwd=remote_repo,text=True).strip(),
            status=subprocess.check_output(['git','status','--short'],cwd=remote_repo,text=True))
        shutil.copy2(__file__,root/'controller.py')
        m=dict(prepared=now(),repo=str(repo),source=str(root/'source'),targetSteps=7641,
            targetTime=original['targetTime'],reference=original['reference'],native=original['native'],
            runtimeRepositories=repositories,runtimeOrigin='post-integration checkouts, copied in full before dispatch',
            baseline=original['baseline'],prior=str(prior),rows=rows(),reuse={},
            screening=original['screening'],scope='Michel baseline, +full Taylor p/v remap, +generic BDF correction',
            sourceSHA256={str(p.relative_to(root/'source')):sha(p) for p in (root/'source').rglob('*') if p.is_file()},
            controllerSHA256=sha(root/'controller.py'),referenceInputSHA256=original['referenceInputSHA256'],
            oracle=str(prior/'qualification/02-accepted-fields'),
            physicalControls='same original IC, FP32, volume ghosts, ramp0.5, nu, exact supplied timesteps, epsilonV=-8, max200; generic spatial map and accepted-state velocity stopping',
            limits=['Warmup and existing fixture source damper/wall diffusion/RK settings retained; not a fully paper-faithful solver.',
                'Taylor transport uses shared unrenormalized difference gradients, not an assumed discrete product rule.',
                'BDF candidate is the existing generic position-history correction, not source-specific duplicated-shift bookkeeping.'])
        atomic(root/'manifest.json',m);shutil.copy2('/tmp/acsph-michel-resume-tests.log',root/'cpu-tests.log')
        (root/'KEEP').write_text('Retain original-IC Michel/remap/BDF controls, failed outputs, sources and qualification.\n')
        atomic(root/'status.json',dict(phase='prepared',complete=False));print('PREPARED',root);return
    m=json.loads((root/'manifest.json').read_text());source=Path(m['source']);py=sys.executable
    assert sha(__file__)==m['controllerSHA256']
    state=dict(started=now(),phase='waiting for prior queue',complete=False,supervisor_pid=os.getpid(),phases=[])
    def save(**kw):state.update(kw,updated=now());atomic(root/'status.json',state)
    cancelled=False
    def cancel(*_):
        nonlocal cancelled
        cancelled=True
    signal.signal(signal.SIGTERM,cancel);signal.signal(signal.SIGINT,cancel)
    def check():
        if cancelled or (root/'STOP').exists():raise InterruptedError('STOP requested')
    def verify():
        for name,digest in m['sourceSHA256'].items():assert sha(source/name)==digest,name
        for name,digest in m['referenceInputSHA256'].items():assert sha(Path(m['reference'])/name)==digest,name
    def progress(path):return [json.loads(s) for s in (path/'progress.jsonl').read_text().splitlines()]
    env=os.environ.copy();env.update(PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
        WARP_CACHE_PATH='/tmp/acsph-parity-repair-cache',MPLCONFIGDIR='/tmp/acsph-parity-mpl',
        PYTHONPATH=str(source/'src')+os.pathsep+str(source/'scripts/acsph_reference'))
    def command(name,path,steps=7641,eager=False,daemon=None):
        cmd=[py,'-u',source/'scripts/acsph_reference/run_lobovsky_ablation.py',
            '--runtime',source,'--reference',m['reference'],'--out',path,'--michel-trial',name,'--steps',steps]
        if eager:cmd.append('--eager')
        if daemon is not None:cmd.append('--daemon' if daemon else '--no-daemon')
        return list(map(str,cmd))
    def run(name,cmd,path=None,physical=False,allow_failure=False):
        verify();check()
        if physical:
            while gpu_users():save(phase='waiting for GPU: '+name);check();time.sleep(15)
        phase=dict(name=name,command=list(map(str,cmd)),started=now());state['phases'].append(phase)
        tick=last=time.monotonic();last_t=-1.;reason=None;paused_seconds=0.;observed=tick
        with (root/(name+'.log')).open('xb') as log:
            child=subprocess.Popen(phase['command'],cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            phase['pid']=child.pid;save(phase=name,child_pid=child.pid)
            try:
                while child.poll() is None:
                    time.sleep(5);check();moment=time.monotonic()
                    remote=json.loads((path/'daemon-status.json').read_text()) if path and (path/'daemon-status.json').exists() else {}
                    paused=remote.get('status')=='paused'
                    if paused:paused_seconds+=moment-observed;last=moment
                    observed=moment;elapsed=moment-tick
                    phase['paused_seconds']=paused_seconds
                    if physical and path and (path/'progress.jsonl').exists():
                        p=path/'progress.jsonl'
                        with p.open('rb') as f:
                            f.seek(max(0,p.stat().st_size-16384));lines=f.read().splitlines()
                        try:r=json.loads(lines[-1])
                        except (ValueError,IndexError):r=None
                        if r:
                            phase['last_progress']=r
                            if r['t']>last_t:last_t=r['t'];last=time.monotonic()
                            if not math.isfinite(r['t']) or not math.isfinite(r['maxVelocity']) or r['maxVelocity']>100:
                                reason='nonfinite/speed>100m/s';break
                    if elapsed-paused_seconds>(14400 if physical else 1800):reason='wall limit';break
                    if physical and not paused and time.monotonic()-last>300:reason='clock stalled300s';break
                    phase['wall_seconds']=elapsed;save()
            finally:
                if child.poll() is None:phase['terminated_owned_pids']=terminate_owned_tree(child.pid)
                child.wait();phase.update(returncode=child.returncode,stop_reason=reason,wall_seconds=time.monotonic()-tick,finished=now());save(child_pid=None)
        if (phase['returncode'] or reason) and not allow_failure:raise RuntimeError('Failed phase '+name)
        return phase
    def qualifies(path):
        r=json.loads((path/'result.json').read_text())
        assert r['nSteps']==3 and not r['diverged'] and r['graph_disabled'] is None
        timing=[json.loads(l) for l in (path/'solve-timing.jsonl').read_text().splitlines()]
        assert len(timing)==3 and all(t['captured'] and t['replays']>0 and t['disabled'] is None for t in timing)
    save()
    try:
        # Honor the existing queue; never terminate or overlap it.
        while True:
            check();prior=json.loads((Path(m['prior'])/'run-status.json').read_text())
            if prior.get('complete') or prior['phase'] in ('failed','cancelled'):break
            if birth(prior['supervisor_pid']) is None:raise RuntimeError('Prior supervisor vanished without a terminal receipt')
            save(priorPhase=prior['phase']);time.sleep(15)
        import numpy as np
        from render import read_snapshot
        expected=read_snapshot(Path(m['baseline'])/'snapshots/000000.bin')
        q=root/'qualification';q.mkdir()
        oracle=q/'oracle';run('qualify-oracle',command('oracle',oracle,3),oracle,True);qualifies(oracle)
        assert (oracle/'snapshots/000003.bin').read_bytes()==(Path(m['oracle'])/'snapshots/000003.bin').read_bytes()
        for row in m['rows']:
            path=q/row['id'];run('qualify-'+row['id'],command(row['id'].split('-')[-1],path,3),path,True);qualifies(path)
            initial=read_snapshot(path/'snapshots/000000.bin')
            assert all(np.array_equal(initial[k],expected[k]) for k in ('x','v','p','rho','volume','bc'))
        # Transport and BDF integration must also survive eager/graph parity.
        for row in m['rows'][1:]:
            path=q/(row['id']+'-eager');run('eager-'+row['id'],command(row['id'].split('-')[-1],path,3,True),path,True)
            assert (path/'snapshots/000003.bin').read_bytes()==(q/row['id']/'snapshots/000003.bin').read_bytes()
        # Publication must preserve the exact accepted trajectory as well.
        path=q/'daemon-baseline'
        run('qualify-daemon-baseline',command('baseline',path,3,daemon=True),path,True);qualifies(path)
        assert (path/'snapshots/000003.bin').read_bytes()==(q/m['rows'][0]['id']/'snapshots/000003.bin').read_bytes()
        assert (path/'daemon-run.json').exists()
        atomic(root/'qualification.json',dict(complete=True,finished=now(),allPhysicalICsBitwise=True,
            baselinePlumbingOracleBitwise=True,allGraphsActive=True,transportBDFGraphEagerBitwise=True,daemonSnapshotBitwise=True))
        for row in m['rows']:
            path=root/row['id'];phase=run('physical-'+row['id'],command(row['id'].split('-')[-1],path),path,True,True)
            atomic(path/'campaign-row-status.json',phase)
            if phase['returncode'] or phase['stop_reason']:
                remote=json.loads((path/'daemon-status.json').read_text()) if (path/'daemon-status.json').exists() else {}
                if remote.get('status')=='stopped':raise InterruptedError('Campaign stopped from daemon')
                result=json.loads((path/'result.json').read_text()) if (path/'result.json').exists() else {}
                receipt=json.loads((path/'failure/receipt.json').read_text()) if (path/'failure/receipt.json').exists() else {}
                known=bool(phase['stop_reason'] or result.get('diverged') or receipt.get('error','').startswith(('Reference solve exceeded certified candidate motion:','Nonfinite dynGPUSPH reference residual')))
                if not known:raise RuntimeError('Unknown implementation/output failure: '+row['id'])
                if any((path/'snapshots').glob('*.bin')):
                    run('render-failed-'+row['id'],[py,source/'scripts/acsph_reference/render.py',path,'--label','warpSPH Michel trial; accepted prefix'])
            run('analysis-'+row['id'],[py,source/'scripts/acsph_reference/analyze_lobovsky_ablation.py','--root',root,'--out',root/('analysis-through-'+row['id'])])
        verify();save(phase='complete',complete=True,finished=now(),failedRows=[p['name'] for p in state['phases'] if p['name'].startswith('physical-') and (p['returncode'] or p['stop_reason'])])
    except BaseException as error:
        save(phase='cancelled' if isinstance(error,InterruptedError) else 'failed',error=str(error),finished=now());raise


if __name__=='__main__':main()
