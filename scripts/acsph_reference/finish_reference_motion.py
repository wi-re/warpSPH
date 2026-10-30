#!/usr/bin/env python3
"""Supervise the qualified ACSPH restart, retain its prefix, finish comparison.

This is a complete-history restart, never a restart from visual snapshots.
Original failures remain untouched. STOP cancels this supervisor's children.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from campaign_late_ringing import atomic,terminate_owned_tree,gpu_users,now


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,required=True)
    args=ap.parse_args();root=args.root.resolve();repo=Path(__file__).resolve().parents[2]
    source=root/'source-repair';capture=root/'reproduction/capture'
    campaign=root.parent/'acsph-late-ringing-2026-10-29'
    prefix=campaign/'warpsph-f32-repaired';native=campaign/'dyngpusph-f64-dense'
    reference=root.parent/'acsph-lobovsky-long-2026-10-27/native-f64'
    suffix=root/'continuation';merged=root/'warpsph-f32-complete';end=8751;start=6488
    expected=json.loads((root/'repair-source-receipt.json').read_text())['sha256']
    qualified=json.loads((root/'restart-repaired-one-step/restart-audit.json').read_text())
    assert all(qualified['originalRejectedTrialBitwise'].values())
    old=json.loads((root/'restart-old-guard/restart-audit.json').read_text())
    assert all(old['originalRejectedTrialBitwise'].values())
    assert not suffix.exists() and not merged.exists()
    state={'phase':'starting','started':now(),'supervisor_pid':os.getpid(),'complete':False,
        'startStep':start,'endStep':end,'source':str(source),'suffix':str(suffix),
        'prefix':str(prefix),'qualification':str(root/'restart-repaired-one-step/restart-audit.json')}
    def save(**kw):state.update(kw,updated=now());atomic(root/'continuation-status.json',state)
    stopped=False
    def cancel(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,cancel);signal.signal(signal.SIGINT,cancel)
    def check():
        if stopped or (root/'STOP').exists():raise InterruptedError('Stop requested')
    def verify():
        for f,digest in expected.items():
            assert hashlib.sha256((source/f).read_bytes()).hexdigest()==digest,f
    env=os.environ.copy();env.update(WARP_CACHE_PATH='/tmp/acsph-parity-repair-cache',
        MPLCONFIGDIR='/tmp/acsph-parity-mpl',PYTHONUNBUFFERED='1')
    py=sys.executable
    def run(label,cmd,physical=False):
        verify();check()
        if physical:
            users=gpu_users();assert not users,users
        tick=last=time.monotonic();last_t=-1.
        with (root/(label+'.log')).open('xb') as log:
            child=subprocess.Popen(list(map(str,cmd)),cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            save(phase=label,child_pid=child.pid,command=list(map(str,cmd)))
            try:
                while child.poll() is None:
                    time.sleep(5);check();elapsed=time.monotonic()-tick
                    progress=suffix/'progress.jsonl'
                    if physical and progress.exists():
                        with progress.open('rb') as f:
                            f.seek(max(0,progress.stat().st_size-16384));lines=f.read().splitlines()
                        row=json.loads(lines[-1]);state['last_progress']=row
                        if row['t']>last_t:last_t=row['t'];last=time.monotonic()
                    if elapsed>(10800 if physical else 1800):raise TimeoutError(label+' wall limit')
                    if physical and time.monotonic()-last>300:raise TimeoutError('Physical clock stalled300s')
                    save(phase_wall_seconds=elapsed)
                if child.returncode:raise RuntimeError(label+' exited '+str(child.returncode))
            finally:
                if child.poll() is None:state['terminated_owned_pids']=terminate_owned_tree(child.pid)
                child.wait();save(child_pid=None)
    save()
    try:
        run('continue-physical',[py,'-u',repo/'scripts/acsph_reference/resume_reference_motion.py',
            '--runtime',source,'--reference',reference,'--capture',capture,'--out',suffix,
            '--end-step',end,'--qualify-trial'],physical=True)
        # Recombine only accepted states. The duplicate checkpoint diagnostic is
        # removed; the old failed trial is never part of this trajectory.
        save(phase='merge-and-validate')
        a=[json.loads(s) for s in (prefix/'progress.jsonl').read_text().splitlines()]
        b=[json.loads(s) for s in (suffix/'progress.jsonl').read_text().splitlines()]
        assert len(a)==start+1 and len(b)==end-start+1,(len(a),len(b))
        assert a[-1]['t']==b[0]['t']
        rows=a+b[1:];assert len(rows)==end+1
        times=[0.]+[s['t'] for s in json.loads((reference/'result.json').read_text())['steps'][:end]]
        assert [s['t'] for s in rows]==times
        assert all(abs(s['maxVelocity'])<100 for s in rows)
        result=json.loads((suffix/'result.json').read_text())
        assert not result['diverged'] and result['graph_disabled'] is None
        assert result['nSteps']==end-start
        merged.mkdir();(merged/'snapshots').mkdir()
        (merged/'progress.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in rows))
        (merged/'progress.json').write_text(json.dumps(rows,indent=2)+'\n')
        for directory in (prefix,suffix):
            for f in (directory/'snapshots').glob('*.bin'):
                dest=merged/'snapshots'/f.name
                if dest.exists():assert dest.read_bytes()==f.read_bytes()
                else:os.link(f,dest)
        timing=[]
        for directory,offset in ((prefix,0),(suffix,start)):
            for line in (directory/'solve-timing.jsonl').read_text().splitlines():
                row=json.loads(line);row['step']+=offset
                if row['step']<=end:timing.append(row)
        assert [s['step'] for s in timing]==list(range(1,end+1))
        (merged/'solve-timing.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in timing))
        receipt={'kind':'Original IC accepted prefix plus complete-history bitwise-qualified restart',
            'originalPrefix':str(prefix),'restartSuffix':str(suffix),'checkpointStep':start,
            'sourceReceipt':str(root/'repair-source-receipt.json'),'physicalSteps':end,
            'clockMaxAbs':0.,'endTime':rows[-1]['t'],
            'initialPrefixNewGuardAudited':False,'failedOriginalPreserved':True,
            'note':'All-stage motion checks apply to the repaired suffix; the accepted prefix is unchanged.'}
        atomic(merged/'restart-provenance.json',receipt)
        run('late-analysis',[py,'-u',source/'scripts/acsph_reference/analyze_late_ringing.py',
            '--reference',native,'--local',merged,'--out',root/'late-analysis'])
        summary=json.loads((root/'late-analysis/summary.json').read_text())
        assert summary['fullLateCoverage']
        run('export-viewer',[py,'-u',source/'scripts/acsph_reference/export_viewer.py',
            merged,'--out',root/'warpSPH-lobovsky-complete.h5','--label',
            'warpSPH FP32; geometric guard repair; qualified restart at step6488'])
        run('render-full',[py,'-u',source/'scripts/acsph_reference/render.py',merged,
            '--label','warpSPH FP32; original prefix + qualified geometric-guard restart'])
        verify();save(phase='complete',complete=True,finished=now(),merged=str(merged),analysis=str(root/'late-analysis'))
    except BaseException as e:
        save(phase='cancelled' if isinstance(e,InterruptedError) else 'failed',error=str(e),finished=now())
        raise


if __name__=='__main__':main()
