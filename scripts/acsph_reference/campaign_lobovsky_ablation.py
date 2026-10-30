#!/usr/bin/env python3
"""Prepare, qualify, then serially run the frozen cumulative Lobovsky ladder.

STOP cancels only owned children. A failed physical row remains a result;
preflight/parity/invariant failures stop the campaign before long replays.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

try:
    from .ablation_matrix import rows
    from .campaign_late_ringing import atomic,birth,gpu_users,now,terminate_owned_tree
except ImportError:
    from ablation_matrix import rows
    from campaign_late_ringing import atomic,birth,gpu_users,now,terminate_owned_tree


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def historical_progress_equal(actual,expected):
    """The historical baseline predates two nullable diagnostic columns.

    Require every recorded value exactly. Only missing epsilonP/exitReason
    with a null value in the newer baseline schema may be normalized.
    """
    if len(actual)!=len(expected):return False
    for a,b in zip(actual,expected):
        if any(k not in a or a[k]!=v for k,v in b.items()):return False
        extra=a.keys()-b.keys()
        if not extra<={'epsilonP','exitReason'} or any(a[k] is not None for k in extra):return False
    return True


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--mode',choices=('prepare','qualify','run'),required=True)
    ap.add_argument('--resume-qualification',action='store_true',help='Reuse successful same-command preflights under the unchanged source manifest')
    args=ap.parse_args();root=args.root.resolve();repo=Path(__file__).resolve().parents[2]
    old=repo/'.tmp/acsph-lobovsky-consistent-closure-2026-10-28'
    prior=repo/'.tmp/acsph-motion-investigation-2026-10-29'
    reference=repo/'.tmp/acsph-lobovsky-long-2026-10-27/native-f64'
    baseline=old/'warpsph-f32-consistent-closure';current=prior/'warpsph-f32-partial-8395'
    native=repo/'.tmp/acsph-late-ringing-2026-10-29/dyngpusph-f64-dense'
    source=root/'source';py=sys.executable
    additions=['ablation_matrix.py','ablation_controls.py','run_lobovsky_ablation.py',
        'analyze_lobovsky_ablation.py']
    if args.mode=='prepare':
        root.mkdir(parents=True,exist_ok=False)
        assert shutil.disk_usage(root).free>100*1024**3
        receipt=json.loads((prior/'repair-source-receipt.json').read_text())
        for name,digest in receipt['sha256'].items():assert sha(Path(receipt['runtime'])/name)==digest,name
        shutil.copytree(receipt['runtime'],source,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        for name in additions:shutil.copy2(Path(__file__).with_name(name),source/'scripts/acsph_reference'/name)
        hashes={str(p.relative_to(source)):sha(p) for p in source.rglob('*') if p.is_file()}
        old_hashes=json.loads((old/'manifest.json').read_text())['sourceHashes']
        for name,digest in old_hashes.items():assert sha(Path(name))==digest,name
        manifest={'prepared':now(),'targetSteps':7641,'targetTime':1.5300141248223804,
            'source':str(source),'sourceSHA256':hashes,'oldSourceSHA256':old_hashes,
            'reference':str(reference),'referenceInputSHA256':{name:sha(reference/name)
                for name in ('result.json','input_values.dat','snapshots/000000.bin')},
            'baseline':str(baseline),'current':str(current),'native':str(native),
            'rows':rows(),'scope':'Original-IC FP32 Lobovsky; cumulative controlled hybrids; source operator block coordinated',
            'reuse':{'00-baseline':str(baseline),'10-plateau-trim':str(current)},
            'screening':'1.44..1.53s ringing RMS >2x and >1kPa extra; caps >2x and >=10 extra. Descriptive, not statistical.',
            'limits':['Source shift geometry/history and source operator block are coupled groups.',
                'Cumulative order can contain interactions; first jump needs a confirming control.',
                'Reference guard remains unchanged; failures remain explicit results.']}
        times=json.loads((reference/'result.json').read_text())['steps']
        manifest['targetTime']=times[7640]['t']
        atomic(root/'manifest.json',manifest)
        shutil.copy2('/tmp/acsph-ablation-tests.log',root/'cpu-tests.log')
        (root/'KEEP').write_text('Retain ablation sources, original-IC trajectories, checkpoints, videos and failures.\n')
        atomic(root/'status.json',dict(phase='prepared',complete=False,prepared=now()))
        print('ABLATION PREPARED',root,flush=True);return
    manifest=json.loads((root/'manifest.json').read_text())
    previous=None
    if args.resume_qualification:
        assert args.mode=='qualify'
        previous=json.loads((root/'qualify-status.json').read_text())
        shutil.copy2(root/'qualify-status.json',root/'qualify-status-before-schema-repair.json')
    state={'phase':args.mode,'complete':False,'started':now(),'supervisor_pid':os.getpid(),
        'supervisorSourceSHA256':sha(Path(__file__)),'phases':[]}
    def save(**kw):
        state.update(kw,updated=now(),mode=args.mode)
        atomic(root/(args.mode+'-status.json'),state);atomic(root/'status.json',state)
    stopped=False
    def cancel(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,cancel);signal.signal(signal.SIGINT,cancel)
    def check():
        if stopped or (root/'STOP').exists():raise InterruptedError('STOP requested')
    def verify():
        for name,digest in manifest['sourceSHA256'].items():assert sha(source/name)==digest,name
        for name,digest in manifest['referenceInputSHA256'].items():assert sha(reference/name)==digest,name
    env=os.environ.copy();env.update(PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
        WARP_CACHE_PATH='/tmp/acsph-parity-repair-cache',MPLCONFIGDIR='/tmp/acsph-parity-mpl',
        PYTHONPATH=str(source/'src')+os.pathsep+env.get('PYTHONPATH',''))
    def command(row,out,steps=7641,eager=False):
        cmd=[py,'-u',source/'scripts/acsph_reference/run_lobovsky_ablation.py',
            '--runtime',source,'--reference',reference,'--out',out,'--row',row['id'],'--steps',steps]
        if eager:cmd+=['--eager']
        return cmd
    def run(name,cmd,directory=None,physical=False,allow_failure=False,old_runtime=False):
        verify();check()
        if previous is not None:
            found=[p for p in previous['phases'] if p['name']==name]
            if found:
                old_phase=found[-1]
                assert old_phase['command']==list(map(str,cmd)) and old_phase['returncode']==0 and old_phase['stop_reason'] is None
                assert directory is not None and (directory/'result.json').exists()
                phase=dict(old_phase,reusedSuccessfulPreflight=True)
                state['phases'].append(phase);save(phase='reused '+name)
                return phase
        if physical:
            while gpu_users():save(phase='waiting for GPU: '+name);check();time.sleep(30)
        phase={'name':name,'started':now(),'command':list(map(str,cmd))};state['phases'].append(phase)
        path=root/(name+'.log');tick=last=time.monotonic();last_t=-1.;stop_reason=None
        with path.open('xb') as log:
            child=subprocess.Popen(phase['command'],cwd=old/'source' if old_runtime else source,
                env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            phase['pid']=child.pid;save(phase=name,child_pid=child.pid)
            try:
                while child.poll() is None:
                    time.sleep(5);check();elapsed=time.monotonic()-tick
                    if physical and directory and (directory/'progress.jsonl').exists():
                        p=directory/'progress.jsonl'
                        with p.open('rb') as f:f.seek(max(0,p.stat().st_size-16384));lines=f.read().splitlines()
                        row=json.loads(lines[-1]);phase['last_progress']=row
                        if row['t']>last_t:last_t=row['t'];last=time.monotonic()
                        if not math.isfinite(row['t']) or not math.isfinite(row['maxVelocity']) or row['maxVelocity']>100:
                            stop_reason='nonfinite state or speed>100m/s';break
                    if elapsed>(14400 if physical else 1800):stop_reason='wall limit';break
                    if physical and time.monotonic()-last>300:stop_reason='clock stalled300s';break
                    phase['wall_seconds']=elapsed;save()
            finally:
                if child.poll() is None:phase['terminated_owned_pids']=terminate_owned_tree(child.pid)
                child.wait();phase.update(returncode=child.returncode,wall_seconds=time.monotonic()-tick,stop_reason=stop_reason,finished=now());save(child_pid=None)
        if (phase['returncode'] or stop_reason) and not allow_failure:raise RuntimeError(name+' failed: '+str(phase))
        return phase
    def progress(path):return [json.loads(s) for s in (path/'progress.jsonl').read_text().splitlines()]
    save()
    try:
        if args.mode=='qualify':
            import numpy as np
            from render import read_snapshot
            q=root/'qualification';q.mkdir(exist_ok=args.resume_qualification)
            # Native paths are deliberately not inherited from the new source.
            # Import each oracle through its own PYTHONPATH in a fresh process.
            old_cmd=[py,'-u',old/'source/scripts/acsph_reference/compare_lobovsky.py','--reference',reference,
                '--out',q/'old-oracle','--steps','3','--precision','float32','--position-form','increment',
                '--fused-residual','--wall-diffusion','source','--acoustic-damper','source','--rk-treatment','source',
                '--wall-velocity','source-consistent','--wall-bounce-ramp','.5','--ghost-spacing','volume',
                '--history','source','--coefficients','mapped','--long-run']
            previous_pythonpath=env['PYTHONPATH'];env['PYTHONPATH']=str(old/'source/src')
            run('qualify-old-oracle',old_cmd,q/'old-oracle',True,old_runtime=True)
            env['PYTHONPATH']=previous_pythonpath
            expected_initial=read_snapshot(baseline/'snapshots/000000.bin')
            for row in rows():
                path=q/row['id'];run('qualify-'+row['id'],command(row,path,3),path,True)
                result=json.loads((path/'result.json').read_text())
                assert result['nSteps']==3 and not result['diverged'] and result['graph_disabled'] is None
                data=read_snapshot(path/'snapshots/000003.bin')
                assert all(np.isfinite(data[k]).all() for k in ('x','v','p'))
                initial=read_snapshot(path/'snapshots/000000.bin')
                assert all(np.array_equal(initial[k],expected_initial[k]) for k in ('x','v','p','rho','volume','bc'))
                timing=[json.loads(s) for s in (path/'solve-timing.jsonl').read_text().splitlines()]
                assert len(timing)==3 and all(r['captured'] and r['disabled'] is None for r in timing)
            baseline_rows=progress(baseline)[:4];actual_rows=progress(q/rows()[0]['id'])
            assert historical_progress_equal(actual_rows,baseline_rows)
            assert (q/'old-oracle/snapshots/000003.bin').read_bytes()==(q/rows()[0]['id']/'snapshots/000003.bin').read_bytes()
            endpoint=q/rows()[-1]['id']
            oracle=repo/'.tmp/acsph-parity-repair-static-2026-10-29/prefix-graph'
            assert (endpoint/'snapshots/000003.bin').read_bytes()==(oracle/'snapshots/000003.bin').read_bytes()
            assert progress(endpoint)==progress(current)[:4]
            for index in (6,7):
                row=rows()[index];path=q/(row['id']+'-eager')
                run('qualify-eager-'+row['id'],command(row,path,3,True),path,True)
                assert (path/'snapshots/000003.bin').read_bytes()==(q/row['id']/'snapshots/000003.bin').read_bytes()
            verify();save(phase='qualified',complete=True,finished=now(),
                gates={'allPhysicalICsBitwise':True,'baselineHistoricalDiagnosticsExact':True,
                    'baselineOldRuntimeSnapshotBitwise':True,'currentEndpointSnapshotAndDiagnosticsBitwise':True,
                    'allRowGraphsActive':True,'warmupAndOperatorGraphEagerBitwise':True})
        else:
            assert json.loads((root/'qualify-status.json').read_text())['complete']
            for row in rows():
                if row['reuse']:continue
                path=root/row['id']
                phase=run('physical-'+row['id'],command(row,path),path,True,True)
                phase['row']=row['id']
                atomic(path/'campaign-row-status.json',phase)
                failed=bool(phase['returncode'] or phase['stop_reason'])
                if failed:
                    result=json.loads((path/'result.json').read_text()) if (path/'result.json').exists() else {}
                    failure=json.loads((path/'failure/receipt.json').read_text()) if (path/'failure/receipt.json').exists() else {}
                    known=bool(phase['stop_reason'] or result.get('diverged') or
                        failure.get('error','').startswith(('Reference solve exceeded certified candidate motion:',
                            'Nonfinite dynGPUSPH reference residual')))
                    phase['failureClass']='numerical guard/budget' if known else 'unknown implementation or output failure'
                    atomic(path/'campaign-row-status.json',phase)
                    if not known:
                        raise RuntimeError('Unknown row failure needs investigation before continuing: '+row['id'])
                if phase['returncode'] and any((path/'snapshots').glob('*.bin')):
                    run('render-failed-'+row['id'],[py,source/'scripts/acsph_reference/render.py',path,
                        '--label','warpSPH ablation '+row['id']+'; accepted prefix of failed row'],allow_failure=True)
                run('analysis-'+row['id'],[py,source/'scripts/acsph_reference/analyze_lobovsky_ablation.py',
                    '--root',root,'--out',root/('analysis-through-'+row['id'])])
            verify();save(phase='complete',complete=True,finished=now(),
                failedRows=[p['name'] for p in state['phases'] if p['name'].startswith('physical-') and (p['returncode'] or p['stop_reason'])])
    except BaseException as error:
        save(phase='cancelled' if isinstance(error,InterruptedError) else 'failed',error=str(error),finished=now())
        raise


if __name__=='__main__':main()
