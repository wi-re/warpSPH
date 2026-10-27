"""Sequential, daemon-visible standalone Eq.78 Sedov controls.

Rows are immutable settings/input folders prepared before execution. Numerical
watchdog stops are retained evidence, not accepted solutions. Unexpected setup
or implementation failures stop the study. All physical runs use the existing
video/progress/watchdog harness; native playback is rendered afterward.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from compare_spheralMaterials import save
from compare_spheralSedov import sample,time_grid,ANALYTIC


def prepare(root):
    """Create new immutable row inputs without depending on earlier runs."""
    import numpy as np
    root.mkdir(parents=True,exist_ok=False)
    for policy,destination,library in (
        ('paper',root,'results/materialEOS/spheral-heating-sign-policies/libSpheralPaperPolicies.so'),
        ('entropy',root/'entropy','results/materialEOS/spheral-thermal-policies/libSpheralPaperPolicies.so')):
        destination.mkdir(exist_ok=True)
        for split in (1,2,4):
            folder=destination/f'n64-dt{split}';folder.mkdir()
            np.savez_compressed(folder/'input.npz',**sample(64))
            grid=np.repeat(time_grid(64)/split,split)
            settings=dict(case='cylindricalSedov',gammaVariant=folder.name,gamma=[5/3],nx=64,dim=2,end=.05,dt=float(grid[0]),timeGrid=grid.tolist(),
                support='adaptive',frameEvery=len(grid)//50,exactSteps=len(grid),domain=[-.5,.5],kernel='B8',spheralKernelOrder=7,extent=4.,nPerh=1.,n_h=4.,volumePolicy='kernelSum',
                inputPath=str((folder/'input.npz').resolve()),compatibleEnergy=False,energyPartition='native',energyPredictor='raw',thermalRate=policy,
                energyTolerance=.01,precision='float64',eosSound='strict',auditEOS=True,auditThermal=True,
                namePrefix='Standalone '+policy+' thermal Sedov',comparisonPolicy='Standalone '+policy+' thermal ODE; no compatible or total-energy update; fresh midpoint RK2',
                referenceViscosity=dict(Cl=2.,Cq=1.,epsilon2=.01,etaCritFrac=1.,etaFoldFrac=.2),
                source='Common normalized W7 spike, rho=1, gamma=5/3, ambient pressure=1e-6. Experimental fresh/paper stage controls; entropy RHS is a local modification in both codes.')
            save(folder/'settings.json',settings)
        files=[Path(name) for name in ('scripts/compare_spheralSedov.py','scripts/compare_spheralMaterials.py','scripts/compare_spheralKH.py',
            'scripts/crkThermalRate.py',__file__,'scripts/analyze_sedovThermal.py','benchmarks/eos/paperSpheral/paperPolicies.cc',
            'benchmarks/eos/paperSpheral/thermalPolicies.cc','src/warpSPH/runner/configuration.py','src/warpSPH/modules/crk/dudt.py',
            'src/warpSPH/systems/compSPH.py','literature/frontiere2017_crksph.pdf')]
        files.append(ANALYTIC);sources={};(destination/'sources').mkdir()
        for source in files:
            key=hashlib.sha256(source.read_bytes()).hexdigest();sources[str(source)]=key
            shutil.copy2(source,destination/'sources'/(source.stem+'_'+key[:12]+source.suffix))
        library=Path(library).resolve(strict=True)
        save(destination/'provenance.json',dict(sources=sources,library=str(library),librarySHA256=hashlib.sha256(library.read_bytes()).hexdigest(),
            intent='Paper Eq78 raw control' if policy=='paper' else 'Experimental instantaneous Eq63 pair-power ODE in both solvers; no final compatible balancing.',
            energyGate='1% exploratory ceiling; actual drift and refinement must be reported; this is not an accuracy qualification.'))


def run(root, rows):
    manifest=dict(status='running',pid=os.getpid(),stages=[])
    target=root/'study.json'
    if target.exists():
        manifest['previous']=json.loads(target.read_text())
    provenance=json.loads((root/'provenance.json').read_text())
    if 'librarySHA256' in provenance:
        assert hashlib.sha256(Path(provenance['library']).read_bytes()).hexdigest()==provenance['librarySHA256']
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',
             MPLCONFIGDIR='/tmp/warpsph-thermal-mpl',PYTHONFAULTHANDLER='1')
    def execute(name,mode,folder):
        native=mode=='spheral'
        python='results/materialEOS/spheral-build/install/.venv/bin/python' if native else sys.executable
        command=[python,'-u','scripts/compare_spheralSedov.py',mode,'--output',str(folder)]
        logpath=root/(name+'.log');stage=dict(name=name,command=command,log=str(logpath),status='running')
        manifest['stages'].append(stage);save(target,manifest)
        start=time.perf_counter();last=None;print('START',name,flush=True)
        with logpath.open('w') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,
                env=dict(env,LD_PRELOAD=provenance['library']) if native else env)
            stage['pid']=child.pid;save(target,manifest)
            try:
                while child.poll() is None:
                    if (root/'STOP').exists():child.send_signal(signal.SIGINT)
                    try:child.wait(timeout=5)
                    except subprocess.TimeoutExpired:pass
                    lines=logpath.read_text(errors='replace').replace('\r','\n').splitlines()
                    latest=next((s for s in reversed(lines) if any(v in s for v in ('| t=','SEDOV RESULT','EOS NEGATIVE','Remote run registered:','Traceback'))),'')
                    if latest and latest!=last:print(name+': '+latest[:400],flush=True);last=latest
                code=child.wait()
            except BaseException:
                child.send_signal(signal.SIGINT);child.wait();raise
        failure=folder/mode/'failure.json'
        numerical=bool(mode!='render' and failure.exists() and 'finite-state/velocity watchdog' in json.loads(failure.read_text())['error'])
        stage.update(returncode=code,status='numerical_failure' if numerical else ('completed' if code==0 else 'failed'),wallSeconds=time.perf_counter()-start)
        save(target,manifest);print('END',name,stage['status'],flush=True)
        if code and not numerical:raise RuntimeError('Unexpected failure: '+name+'; '+str(logpath))
        if numerical and mode=='warp':
            from warpSPH.runner.media import encodeFrames
            for images in (folder/'warp').glob('*/images'):
                if not (images.parent/'output.mp4').exists():encodeFrames(str(images),str(images.parent))
    try:
        for row in rows:
            folder=root/row
            for mode in ('spheral','warp'):
                if not (folder/mode/'comparisonRun.json').exists() and not (folder/mode/'failure.json').exists():execute(row+'-'+mode,mode,folder)
                if mode=='spheral' and not (folder/'spheral/output.mp4').exists():execute(row+'-render','render',folder)
        manifest['status']='completed'
    except BaseException as error:
        manifest.update(status='failed',error=repr(error));raise
    finally:save(target,manifest)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--prepare',action='store_true',help='Create fresh inputs/settings for both policies; no simulation')
    p.add_argument('--rows',nargs='+',default=['n64-dt1','n64-dt2','n64-dt4'])
    args=p.parse_args()
    if args.prepare:prepare(args.input.resolve())
    else:run(args.input.resolve(),args.rows)
