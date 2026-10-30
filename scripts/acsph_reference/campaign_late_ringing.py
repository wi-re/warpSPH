#!/usr/bin/env python3
"""Run source-pinned, serial late-time Lobovsky replays and automatic analysis.

Reuses the qualified static-wall solver and read-only dynGPUSPH observer.
Starts both physical runs from the original IC, never from a visual snapshot.
STOP in the campaign directory cancels this queue and only its owned children.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time


def now():return datetime.now(timezone.utc).isoformat()


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def atomic(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)


def last_line(path):
    if not path.exists():return ''
    with path.open('rb') as f:
        f.seek(max(0,path.stat().st_size-65536));lines=f.read().replace(b'\r',b'\n').splitlines()
    return lines[-1].decode(errors='replace') if lines else ''


def gpu_users():
    return subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip().splitlines()


def birth(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
    except (FileNotFoundError,ProcessLookupError):return None


def terminate_owned_tree(pid):
    # Wrappers may put a native executable in its own process group. Capture
    # descendants before signalling, and check process birth before each kill.
    rows=subprocess.check_output(['ps','-eo','pid=,ppid='],text=True).splitlines()
    pairs=[tuple(map(int,row.split())) for row in rows]
    owned={pid};changed=True
    while changed:
        old=len(owned);owned.update(child for child,parent in pairs if parent in owned);changed=len(owned)!=old
    identities={p:birth(p) for p in owned}
    for sig in (signal.SIGTERM,signal.SIGKILL):
        for p,start in identities.items():
            if start is not None and birth(p)==start:
                try:os.kill(p,sig)
                except ProcessLookupError:pass
        if sig==signal.SIGTERM:
            for _ in range(30):
                if not any(v is not None and birth(p)==v for p,v in identities.items()):break
                time.sleep(1)
    return sorted(owned)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--end-time',type=float,default=1.70)
    ap.add_argument('--phase-wall-limit',type=float,default=28800)
    args=ap.parse_args();repo=Path(__file__).resolve().parents[2];root=args.root.resolve()
    root.mkdir(parents=True,exist_ok=False)
    old=root.parent/'acsph-parity-repair-static-2026-10-29'
    dense=root.parent/'acsph-dense-reference-2026-10-29'
    reference=root.parent/'acsph-lobovsky-long-2026-10-27/native-f64'
    state={'started':now(),'phase':'freezing','phases':[],'complete':False,'supervisor_pid':os.getpid(),
        'queue':['warpsph-f32-repaired','dyngpusph-f64-dense','validation-export-analysis'],
        'target_time_requested':args.end_time,'solver_source':str(old/'source'),
        'differences':['dynGPUSPH FP64 versus warpSPH FP32','original hard lagged versus smooth fresh-stage ramp0.5 bounce','source versus physical-fluid-ID pressure trim']}
    def save(**values):state.update(values,updated=now());atomic(root/'status.json',state)
    save();(root/'supervisor.pid').write_text(str(os.getpid())+'\n')
    shutil.copy2(__file__,root/'launch-supervisor.py')
    (root/'KEEP').write_text('Retain both trajectories, dense probes, source/binary receipts, logs, failures and comparison.\n')
    if shutil.disk_usage(root).free<20*1024**3:raise RuntimeError('Less than20GiB free')
    for name,digest in json.loads((old/'source-sha256.json').read_text()).items():
        if sha(old/'source'/name)!=digest:raise RuntimeError('Previously qualified source changed: '+name)
    frozen=root/'source'
    shutil.copytree(old/'source',frozen,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    scripts=frozen/'scripts/acsph_reference'
    for name in ('run.py','analyze_late_ringing.py'):
        shutil.copy2(Path(__file__).with_name(name),scripts/name)
    hashes={str(p.relative_to(frozen)):sha(p) for p in sorted(frozen.rglob('*')) if p.is_file()}
    atomic(root/'source-sha256.json',hashes)
    # Copy the already compiled qualified observer, including its build recipe.
    shutil.copytree(dense/'qualified',root/'qualified-dyngpusph',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    binary=root/'qualified-dyngpusph/build/gpuSph2d'
    recipe=json.loads((root/'qualified-dyngpusph/build-manifest.json').read_text())
    assert sha(binary)==recipe['binaries']['gpuSph2d']
    preset=root/'preset-inputs';preset.mkdir()
    expected=json.loads((dense/'dyngpusph-f64-dense/manifest.json').read_text())['source_inputs']
    for name,digest in expected.items():
        original=Path.home()/'dev/dynGPUSPH/tosend/dambreak_Lobovsky_400_2L'/name
        assert sha(original)==digest,'Preset changed since the qualified first-impact reference: '+name
        shutil.copy2(original,preset/name)
    pinned_reference={name:sha(reference/name) for name in ('result.json','input_values.dat','snapshots/000000.bin')}
    (reference/'KEEP-late-comparison').write_text('Pinned by '+str(root)+'; retain through late comparison.\n')
    rows=json.loads((reference/'result.json').read_text())['steps']
    count=next(i for i,row in enumerate(rows,1) if row['t']>=args.end_time)
    target=rows[count-1]['t']
    state.update(target_steps=count,target_time=target,reference_clock=str(reference),
        preset_input_sha256=expected,reference_input_sha256=pinned_reference,
        previous_prefix_qualification=json.loads((old/'prefix-status.json').read_text()),
        observer_qualification=json.loads((dense/'campaign-status.json').read_text())['observationNoninterferenceGate'])
    (root/'working-tree.patch').write_bytes(subprocess.check_output(['git','diff','HEAD'],cwd=repo))
    (root/'working-tree-status.txt').write_bytes(subprocess.check_output(['git','status','--short'],cwd=repo))
    atomic(root/'manifest.json',{k:v for k,v in state.items() if k not in ('phases','phase')})
    env=os.environ.copy();env.update(PYTHONPATH=str(frozen/'src')+os.pathsep+env.get('PYTHONPATH',''),
        PYTHONUNBUFFERED='1',WARP_CACHE_PATH='/tmp/acsph-parity-repair-cache',MPLCONFIGDIR='/tmp/acsph-parity-mpl')
    py=sys.executable;stop={'requested':False}
    def requested(signum,frame):stop['requested']=True
    signal.signal(signal.SIGTERM,requested);signal.signal(signal.SIGINT,requested)
    def check_stop():
        if stop['requested'] or (root/'STOP').exists():raise InterruptedError('Campaign stop requested; queue cancelled')
    def verify():
        for name,digest in hashes.items():
            if sha(frozen/name)!=digest:raise RuntimeError('Frozen source changed: '+name)
        if sha(binary)!=recipe['binaries']['gpuSph2d']:raise RuntimeError('Frozen binary changed')
        for name,digest in recipe['patched_source_sha256'].items():
            if sha(root/'qualified-dyngpusph/build/src'/name)!=digest:raise RuntimeError('Frozen dynGPUSPH source changed: '+name)
        for name,digest in expected.items():
            if sha(preset/name)!=digest:raise RuntimeError('Frozen preset changed: '+name)
        for name,digest in pinned_reference.items():
            if sha(reference/name)!=digest:raise RuntimeError('Pinned reference input changed: '+name)
    def wait_gpu(name):
        while True:
            check_stop();users=gpu_users()
            if not users:return
            save(phase='waiting for GPU before '+name,gpu_users=users)
            print('WAIT',name,json.dumps(users),flush=True);time.sleep(30)
    def run(name,cmd,gpu=False,physical=None):
        verify();check_stop()
        if gpu:wait_gpu(name)
        phase={'name':name,'command':list(map(str,cmd)),'started':now(),'gpu':gpu}
        path=root/(name+'.log');state['phases'].append(phase)
        start=last_progress=time.monotonic();last_t=-1.;proc=None;last_occupancy=-60.
        print('START',name,flush=True)
        try:
            with path.open('xb') as log:
                proc=subprocess.Popen(phase['command'],cwd=frozen,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                phase['pid']=proc.pid;save(phase=name,child_pid=proc.pid,gpu_users=[])
                if name in state['queue']:state['queue'].remove(name);save()
                while proc.poll() is None:
                    time.sleep(5);check_stop();elapsed=time.monotonic()-start
                    if gpu and elapsed-last_occupancy>=60:
                        phase.setdefault('gpu_occupancy_samples',[]).append({'elapsed':elapsed,'users':gpu_users()});last_occupancy=elapsed
                    if physical:
                        data=root/name
                        try:
                            if physical=='warp':
                                line=last_line(data/'progress.jsonl');row=json.loads(line) if line else {}
                                t=float(row.get('t',-1.));v=float(row.get('maxVelocity',0))
                                phase['last_progress']=row
                                graph=last_line(data/'solve-timing.jsonl')
                                if graph:phase['last_solve']=json.loads(graph)
                            else:
                                line=last_line(data/'accepted-probes.csv');parts=line.split(',')
                                t=float(parts[1]) if parts and parts[0].isdigit() else -1.;v=0.
                                phase['last_progress']={'step':int(parts[0]),'t':t} if t>=0 else {}
                            if not math.isfinite(t) or not math.isfinite(v) or v>100:raise RuntimeError('Nonfinite state or velocity exceeds100m/s')
                            if t>last_t:last_progress=time.monotonic();last_t=t
                        except (ValueError,json.JSONDecodeError):pass # a concurrently appended partial line
                    else:
                        # Export/render/analysis are bounded too; their progress
                        # need not follow a simulation clock.
                        if path.exists() and path.stat().st_size!=phase.get('log_size',0):
                            phase['log_size']=path.stat().st_size;last_progress=time.monotonic()
                    if elapsed>args.phase_wall_limit or time.monotonic()-last_progress>300:
                        raise TimeoutError('Phase wall limit' if elapsed>args.phase_wall_limit else 'No advancing physical clock/output for300s')
                    phase['wall_seconds']=elapsed;save()
                phase.update(returncode=proc.returncode,wall_seconds=time.monotonic()-start,finished=now())
        except BaseException as error:
            phase['error']=str(error)
            if proc is not None and proc.poll() is None:phase['terminated_owned_pids']=terminate_owned_tree(proc.pid);proc.wait()
            phase.update(returncode=proc.returncode if proc else None,wall_seconds=time.monotonic()-start,finished=now())
            save(child_pid=None)
            raise
        save(child_pid=None);print('FINISH',name,phase['returncode'],phase['wall_seconds'],flush=True)
        return phase['returncode']
    try:
        warp=root/'warpsph-f32-repaired';native=root/'dyngpusph-f64-dense'
        warp_code=run(warp.name,[py,'-u',scripts/'compare_lobovsky.py','--reference',reference,
            '--out',warp,'--steps',count,'--precision','float32','--position-form','increment',
            '--solver-model','dyngpusph-static-2d','--wall-bounce-ramp','.5','--ghost-spacing','volume',
            '--long-run','--plot-interval','10'],True,'warp')
        # A physical nonzero exit is retained; the independent reference still
        # runs. A watchdog/error or user stop raises and cancels the queue.
        native_code=run(native.name,[py,'-u',scripts/'run.py','dambreak_Lobovsky_400_2L',
            '--binary',binary,'--preset-directory',preset,'--out',native,'--steps',count,'--t-limit',format(target,'.17g'),
            '--wall-limit',args.phase_wall_limit,'--stall-seconds','180','--max-speed','100','--accepted-probes'],True,'native')
        for name,folder in [('warpSPH',warp),('dynGPUSPH',native)]:
            if (folder/'snapshots').exists():
                code=run('export-'+name,[py,'-u',scripts/'export_viewer.py',folder,'--out',root/(name+'-lobovsky-late.h5'),
                    '--label',name+' late ringing; original IC; precision/bounce/trim named in manifest'])
                if code:state.setdefault('postprocess_failures',[]).append('export-'+name)
        # Probe validation and original-clock reproduction precede comparison.
        sys.path.insert(0,str(scripts));from render import read_snapshot
        from export_viewer import pressure_probes
        import numpy as np
        probes=np.genfromtxt(native/'accepted-probes.csv',delimiter=',',names=True)
        new=json.loads((native/'result.json').read_text())['steps']
        clock_error=max(abs(row['t']-rows[i]['t']) for i,row in enumerate(new)) if new else None
        assert clock_error is not None and clock_error<1e-12,clock_error
        errors=[];frames=0
        save(phase='validating accepted probes and physical state')
        for p in sorted((native/'snapshots').glob('*.bin')):
            frame=read_snapshot(p);i=int(np.abs(probes['t']-frame['t']).argmin());assert abs(probes['t'][i]-frame['t'])<1e-12
            expected_probe,wet=pressure_probes(frame);actual=np.array([probes['p'+str(j)][i] for j in range(1,5)])
            errors.append(float(np.abs(expected_probe-actual).max()));frames+=1
        assert max(errors)<1e-7,max(errors)
        state['reference_qualification']={'max_clock_difference':clock_error,'probe_snapshot_max_abs_pa':max(errors),'frames':frames}
        # Include the actual every-step reference measurements in the viewer
        # archive as well as CSV. Refresh its receipt after augmentation.
        archive=root/'dynGPUSPH-lobovsky-late.h5'
        if archive.exists():
            import h5py
            with h5py.File(archive,'r+') as h:
                group=h.create_group('probes/every_step');group['time']=probes['t'];group['step']=probes['step'].astype(np.int64)
                for key,prefix in [('pressure','p'),('weight','w'),('count','n')]:
                    group[key]=np.column_stack([probes[prefix+str(j)] for j in range(1,5)])
                group.attrs['stage']='accepted post-shift, every physical step'
            receipt_path=archive.with_suffix('.json');receipt=json.loads(receipt_path.read_text())
            receipt.update(snapshotOnlySha256=receipt['sha256'],sha256=sha(archive),everyStepProbeSamples=len(probes),everyStepProbeDataset='probes/every_step')
            atomic(receipt_path,receipt)
        # Retain exact-prefix comparison as an observation, independently of
        # later trajectory separation between solvers.
        prefix={}
        for step in (34,2113):
            a=warp/'snapshots'/f'{step:06d}.bin';b=old/'warpsph-f32-reference-solver/snapshots'/a.name
            if a.exists() and b.exists():prefix[str(step)]=a.read_bytes()==b.read_bytes()
        state['warp_qualified_prefix_snapshot_bitwise']=prefix;save()
        code=run('late-analysis',[py,'-u',scripts/'analyze_late_ringing.py','--reference',native,'--local',warp,'--out',root/'analysis'])
        if code:raise RuntimeError('Late analysis failed; retained logs and trajectories')
        summary=json.loads((root/'analysis/summary.json').read_text())
        verify();save(phase='complete' if not warp_code and not native_code and summary['fullLateCoverage'] else 'completed with incomplete/failed trajectory',
            complete=bool(not warp_code and not native_code and summary['fullLateCoverage'] and not state.get('postprocess_failures')),
            finished=now(),queue=[])
        print('CAMPAIGN FINISHED',json.dumps({'complete':state['complete'],'phase':state['phase']}),flush=True)
    except BaseException as error:
        save(phase='cancelled' if isinstance(error,InterruptedError) else 'failed',error=str(error),finished=now(),queue=[])
        raise


if __name__=='__main__':main()
