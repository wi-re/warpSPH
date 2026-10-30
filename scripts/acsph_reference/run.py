#!/usr/bin/env python3
"""Run one supplied dynGPUSPH preset in isolation, with explicit short-run overrides.

Original particle data, numerical parameters and output schedules are retained.
Only input_values.dat's terminal time can be overridden. Snapshots are always
rendered with render.py after completion. Exit/failure status is always retained.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import shutil
import subprocess
import sys
import time

PRESETS = ('Impact200','LDCAVITY_400_2L_RE1000','SquareRotation400_2L','dambreak_Lobovsky_400_2L')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('preset', choices=PRESETS)
    ap.add_argument('--source',type=Path,default=Path.home()/'dev/dynGPUSPH')
    ap.add_argument('--preset-directory',type=Path,
                    help='Pinned copy of the six preset inputs, independent of checkout edits')
    ap.add_argument('--binary',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--steps',type=int,default=20)
    ap.add_argument('--t-limit',type=float)
    ap.add_argument('--wall-limit',type=float,default=600)
    ap.add_argument('--stall-seconds',type=float,default=90)
    ap.add_argument('--max-speed',type=float,default=100)
    ap.add_argument('--accepted-probes', action='store_true',
                    help='Observe accepted post-shift probes every step; requires observer-qualified build')
    args=ap.parse_args()
    out=args.out.resolve(); out.mkdir(parents=True,exist_ok=False)
    preset=args.preset_directory or args.source/'tosend'/args.preset
    files=('input_values.dat','fluid.dat','boundaries.dat','fsi.dat','gas.dat','probe.dat')
    hashes={}
    for name in files:
        shutil.copyfile(preset/name,out/name); hashes[name]=sha(preset/name)
    if args.t_limit is not None:
        p=out/'input_values.dat'; lines=p.read_text().splitlines(); lines[12]=f'{args.t_limit:.17g} # Explicit short reference terminal time'; p.write_text('\n'.join(lines)+'\n')
    binary=args.binary.resolve()
    recipe_path=binary.parent.parent/'build-manifest.json'
    recipe=json.loads(recipe_path.read_text())
    expected_binary='gpuSph2d-cavity' if args.preset.startswith('LDCAVITY') else 'gpuSph2d'
    if binary.name!=expected_binary or recipe.get('binaries',{}).get(binary.name)!=sha(binary):
        raise SystemExit('Binary or case-specific flags do not match the build receipt')
    if args.accepted_probes and not recipe.get('observation_patch_sha256'):
        raise SystemExit('Every-step accepted probes require an observation-qualified build')
    manifest={'preset':args.preset,'preset_directory':str(preset.resolve()),'source_commit':subprocess.check_output(['git','-C',str(args.source),'rev-parse','HEAD'],text=True).strip(),
              'source_inputs':hashes,'run_inputs':{n:sha(out/n) for n in files},'binary':str(binary),'binary_sha256':sha(binary),
              'limits':{'steps':args.steps,'t_limit':args.t_limit,'wall_seconds':args.wall_limit,'stall_seconds':args.stall_seconds,'max_speed':args.max_speed},
              'build_recipe':recipe, 'video':'pending',
              'accepted_probes': {'enabled':args.accepted_probes,
                  'stage':'accepted post-shift', 'support':.003,
                  'positions':[[1.61,y] for y in (.003,.015,.030,.080)],
                  'method':'fluid-only normalized Wendland C2; dry=0, weight/count recorded'}
                  if args.accepted_probes else {'enabled':False}}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    env=os.environ.copy(); env['DYNGPU_MAX_STEPS']=str(args.steps); env['DYNGPU_MAX_SPEED']=str(args.max_speed)
    env.pop('DYNGPU_ACCEPTED_PROBES',None)
    if args.accepted_probes:
        env['DYNGPU_ACCEPTED_PROBES']=str(out/'accepted-probes.csv')
    start=time.monotonic(); last=start; reason=None; progress=[]
    proc=subprocess.Popen([str(binary)],cwd=out,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,bufsize=0,start_new_session=True)
    selector=selectors.DefaultSelector(); selector.register(proc.stdout,selectors.EVENT_READ)
    pending=b''
    with (out/'run.log').open('wb') as log:
        while selector.get_map():
            now=time.monotonic()
            if proc.poll() is None and (now-start>args.wall_limit or now-last>args.stall_seconds):
                reason='wall timeout' if now-start>args.wall_limit else 'no completed real step'; proc.terminate()
            events=selector.select(timeout=1)
            for key,_ in events:
                data=os.read(key.fileobj.fileno(),65536)
                if not data: selector.unregister(key.fileobj); continue
                log.write(data); log.flush(); pending+=data
                while b'\n' in pending:
                    line,pending=pending.split(b'\n',1); line=line.decode(errors='replace')
                    if line.startswith('REFERENCE STEP'):
                        last=time.monotonic(); row={k:float(v) for k,v in re.findall(r'(\w+)=([^ ]+)',line)}
                        row['wall_seconds']=last-start; progress.append(row); print(line,flush=True)
                        if len(progress)>1 and row['t']<=progress[-2]['t']:
                            reason='time did not advance'; proc.terminate()
                    elif line.startswith('REFERENCE SNAPSHOT') or 'failed' in line.lower() or 'error' in line.lower() or 'ALARM' in line:
                        print(line,flush=True)
        code=proc.wait(timeout=10)
    manifest.update(returncode=code,wall_seconds=time.monotonic()-start,stop_reason=reason,steps=progress,
                    ms_per_step_wall=(1000*(time.monotonic()-start)/len(progress) if progress else None))
    (out/'result.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if (out/'snapshots').exists():
        render=subprocess.run([sys.executable,str(Path(__file__).with_name('render.py')),str(out)])
        manifest['video']='output.mp4' if render.returncode==0 else 'render failed'
        manifest['render_returncode']=render.returncode
        (out/'result.json').write_text(json.dumps(manifest,indent=2)+'\n')
    else:
        manifest['render_returncode']=1
    print(f'REFERENCE COMPLETE code={code} steps={len(progress)} wall={manifest["wall_seconds"]:.3f}s ms/step={manifest["ms_per_step_wall"]}',flush=True)
    if code or reason or manifest['render_returncode']: raise SystemExit(1)

if __name__=='__main__': main()
