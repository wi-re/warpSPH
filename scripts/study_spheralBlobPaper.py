"""Sequential, daemon-visible paper-configured Spheral blob qualification.

Use the Warp Python environment. Each native run is followed by vispy video,
independent operator audits and daemon verification, including on failure.
Stop on a failed qualification or Remote Stop; never advance by changing physics.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--threads',type=int,default=4)
    a=p.parse_args();repo=Path(__file__).resolve().parents[1]
    root=a.output.resolve();root.mkdir(parents=True,exist_ok=True)
    report=a.report.resolve();report.mkdir(parents=True,exist_ok=True)
    library=(repo/'results/materialEOS/spheral-paper-policies/libSpheralPaperPolicies.so').resolve(strict=True)
    native=repo/'results/materialEOS/spheral-build/install/.venv/bin/python'
    env=dict(os.environ,OMP_NUM_THREADS=str(a.threads),OPENBLAS_NUM_THREADS='1',MPLCONFIGDIR='/tmp/warpsph-material-mpl')
    env.pop('LD_PRELOAD',None)
    manifest=dict(status='running',pid=os.getpid(),library=str(library),threads=a.threads,stages=[],
        policy='Sequential coarse 128x32/8KH then published 512x128/8KH. Full run only after coarse admissibility, conservation and formula gates pass. Both daemon-visible; no hidden fallback.')
    def save():
        temporary=root/'study.tmp.json';temporary.write_text(json.dumps(manifest,indent=2)+'\n');temporary.replace(root/'study.json')
    def command(name,cmd,extra=None):
        stage=dict(name=name,command=[str(x) for x in cmd],status='running',log=str(root/(name+'.log')))
        manifest['stages'].append(stage);save();print('Starting',name,stage['log'],flush=True)
        start=time.perf_counter()
        with Path(stage['log']).open('w') as stream:
            with subprocess.Popen(stage['command'],stdout=stream,stderr=subprocess.STDOUT,cwd=repo,env=extra or env) as child:
                stage['pid']=child.pid;save()
                watcher=None
                if name.endswith('-run'):
                    watcher=subprocess.Popen([sys.executable,str(repo/'scripts/watch_spheralBlobPaper.py'),'--pid',str(child.pid),
                        '--log',stage['log'],'--output',str(root/(name+'-watchdog.json'))],stdout=stream,stderr=subprocess.STDOUT,env=env,cwd=repo)
                code=child.wait()
                if watcher is not None:watcher.wait(timeout=15)
        stage.update(status='completed' if code==0 else 'failed',returncode=code,wallSeconds=time.perf_counter()-start)
        save();print('Finished',name,'code',code,'wall',stage['wallSeconds'],flush=True);return code
    save()
    try:
        for nx in (128,512):
            name=f'n{nx}-8kh';folder=root/name
            # Refuse accidental re-execution into existing artifacts.
            if folder.exists():raise RuntimeError(f'Output already exists: {folder}')
            runEnv=dict(env,LD_PRELOAD=str(library))
            rc=command(name+'-run',[native,'-u',repo/'scripts/run_spheralBlobExample.py','run','--paper','--nx',nx,'--tkh',8,
                '--strict-positive','--frame-every',10 if nx==128 else 50,'--output',folder],runEnv)
            shutil.copyfile(root/(name+'-run.log'),folder/'runtime.log')
            # Render even a failed trajectory; initialization failure has no fake movie.
            checks=[]
            if (folder/'trajectory.h5').exists():
                checks.append(command(name+'-render',[sys.executable,repo/'scripts/run_spheralBlobExample.py','render','--output',folder]))
            for rhs in (False,True):
                checks.append(command(name+('-rhs-audit' if rhs else '-density-audit'),[sys.executable,repo/'scripts/audit_spheralPaperPolicies.py','--input',folder]+(['--rhs'] if rhs else [])))
            if (folder/'summary.json').exists():
                checks.append(command(name+'-analyze',[sys.executable,repo/'scripts/analyze_spheralBlobExample.py','--input',folder,'--output',report/name,'--verify-daemon']))
            qualification=report/name/'qualification.json'
            passed=qualification.exists() and json.loads(qualification.read_text())['passed']
            if rc or any(checks) or not passed:
                manifest.update(status='failed',failedCase=name,error='Run, formula, video/daemon verification or qualification failed; inspect stage logs. No next trajectory launched.')
                save();return 1
        manifest['status']='completed';save();return 0
    except BaseException as error:
        manifest.update(status='failed',error=repr(error));save();raise


if __name__=='__main__':raise SystemExit(main())
