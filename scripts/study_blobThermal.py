"""Short blob comparison: unchanged/entropy thermal RHS x compatible on/off.

Exact qualified n128 inputs and original Warp blob policies are held fixed.
No new Spheral physics, full-resolution run or production model change.
"""
import argparse
import copy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import numpy as np

from compare_spheralMaterials import save,warp_run
from compare_spheralBlobPaper import context as blob_context,initialize as blob_initialize,PaperMeasures
from study_crkEnergySuite import POLICIES,install_audit


def prepare(root):
    base=Path('results/materialEOS/blob_matrix_2026-10-20/matched').resolve()
    settings=json.loads((base/'settings.json').read_text());grid=np.array(settings['timeGrid']);initial=dict(np.load(base/'input.npz'))
    assert len(grid)==525 and np.all(grid>0)
    assert len(initial['masses'])==4380 and len(set(settings['gamma']))==1
    root.mkdir(parents=True,exist_ok=False);rows=[]
    for split in (1,2):
        sequence=np.repeat(grid/split,split)
        for policy,choice in POLICIES.items():
            folder=root/f'dt{split}-{policy}';folder.mkdir();np.savez_compressed(folder/'input.npz',**initial)
            s=copy.deepcopy(settings);s.update(case='blob',policy=policy,choice=choice,gammaVariant=folder.name,kernel='B8',
                timeGrid=sequence.tolist(),dt=float(sequence[0]),steps=len(sequence),end=float(sequence.sum()),frameEvery=10*split,
                inputPath=str((folder/'input.npz').resolve()),thermalGamma=settings['gamma'][0],
                auditBaseline=dict(density='massReplacement',Cl=2.,Cq=1.,viscosityKernelScale=4.),
                comparisonPolicy='Exact short blob startup, original Warp summation/massReplacement/Cl2 Cq1/R4 Q/native fraction; '+policy,
                matrixControls=dict(solver=dict(compatibleEnergy=choice['compatible'],crkEnergyPartition='native',crkEnergyPredictor='raw')))
            save(folder/'settings.json',s);rows.append(folder.name)
    sourcefiles=[Path(p) for p in (__file__,'scripts/study_crkEnergySuite.py','scripts/crkThermalRate.py',
        'scripts/compare_spheralBlobPaper.py','scripts/compare_spheralMaterials.py','scripts/study_blobPartition.py',
        'src/warpSPH/modules/crk/dudt.py','src/warpSPH/schemes/compressible/crkSPH.py','src/warpSPH/systems/compSPH.py')]
    (root/'sources').mkdir();sources={}
    for p in sourcefiles:
        key=hashlib.sha256(p.read_bytes()).hexdigest();shutil.copy2(p,root/'sources'/(p.stem+'_'+key[:12]+p.suffix));sources[str(p)]=key
    references={name:str(Path(path).resolve()) for name,path in dict(
        grid='results/materialEOS/crk_heating_v2_2026-10-20/spheral-grid-averaged',
        halfGrid='results/materialEOS/crk_heating_v2_2026-10-20/spheral-halfGrid-averaged').items()}
    save(root/'provenance.json',dict(rows=rows,sources=sources,baseline=str(base),references=references,
        inputArraysSHA256={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in initial.items()},
        scope='Only thermal RHS and compatible flag vary within Warp. Stored Spheral averaged-Q/fresh-CRK/entropy-final reference is reused; density schedule/support/final-sign differences remain.'))


def context(settings):
    case,_,sim,params,solver=blob_context(settings)
    from warpSPH.runner import RunConfiguration
    base_post=case.postStep
    def post(c,s,step):base_post(c,s,step);c.scratch['matrixNextStep']=step+1
    def timestep(c,s):return settings['timeGrid'][min(c.scratch.get('matrixNextStep',0),len(settings['timeGrid'])-1)]
    case=replace(case,postStep=post,timestep=timestep,extraFields=('internalEnergies','supports','pressures'))
    return case,RunConfiguration('blob',simulation=sim,params=params,solver=solver).prepare(case),sim,params,solver


class Measures(PaperMeasures):
    def __init__(self,settings,a):
        super().__init__(settings,a);self.s0=np.log(a['pressures']/a['densities']**settings['thermalGamma'])
    def __call__(self,a,t):
        row=super().__call__(a,t);cloud=a['materials']==1;m=a['masses'];mass=m[cloud].sum();x=a['positions'][cloud]
        with np.errstate(invalid='ignore',divide='ignore'):ds=np.log(a['pressures']/a['densities']**self.settings['thermalGamma'])-self.s0
        centre=np.sum(m[cloud,None]*x,axis=0)/mass
        row.update(entropyChangePerMass=float(np.dot(m,ds)/m.sum()),entropyL1PerMass=float(np.dot(m,abs(ds))/m.sum()),
            cloudEntropyChange=float(np.dot(m[cloud],ds[cloud])/mass),cloudEntropyL1=float(np.dot(m[cloud],abs(ds[cloud]))/mass),
            cloudCOMx=float(centre[0]),cloudCOMy=float(centre[1]),cloudX05=float(np.quantile(x[:,0],.05)),cloudX95=float(np.quantile(x[:,0],.95)),
            cloudX99=float(np.quantile(x[:,0],.99)),cloudYspan90=float(np.quantile(x[:,1],.95)-np.quantile(x[:,1],.05)),
            cloudRMSRadius=float(np.sqrt(np.dot(m[cloud],np.sum((x-centre)**2,axis=1))/mass)))
        return row


def summarize(root,solver,settings,history,initial,final,wall,steps,remote,extra):
    clean=[{k:v if np.isfinite(v) else None for k,v in row.items()} for row in history]
    gates=dict(completed=abs(history[-1]['time']-settings['end'])<1.e-10,
        positive=all(r['minDensity']>0 and r['minEnergy']>0 and r['minPressure']>0 for r in history),
        finite=all(np.isfinite(v).all() for v in final.values()),identities=all(np.array_equal(initial[k],final[k]) for k in ('UIDs','materials','masses')))
    extra=dict(extra);extra['baseHarnessPolicyLabel']=extra.pop('policy',None)
    row=dict(solver=solver,settings=settings,remoteRunID=remote,steps=int(steps),wallSeconds=wall,meanStepMilliseconds=wall*1000/max(steps,1),
        time=history[-1]['time'],final=clean[-1],gates={k:bool(v) for k,v in gates.items()},passed=bool(all(gates.values())),
        maxAbsoluteEnergyDrift=max(abs(r['relativeEnergyDrift']) for r in history),**extra)
    for name,a in [('initialState',initial),('finalState',final)]:np.savez_compressed(root/(name+'.npz'),**a)
    save(root/'diagnosticHistory.json',clean);save(root/'comparisonRun.json',row)
    print('BLOB THERMAL RESULT',settings['gammaVariant'],'passed',row['passed'],'wall',wall,'ms/step',row['meanStepMilliseconds'],'final',clean[-1],flush=True)
    return row


def run_row(args):
    sys.path.insert(0,str(args.publisher_path));settings=json.loads((args.output/'settings.json').read_text());audit=install_audit(settings)
    def initialize(c,system,a,s):
        blob_initialize(c,system,a,s);c.scratch['_titleProvenance']='Blob thermal | '+s['gammaVariant']+' | original short matched Warp setup'
    try:return warp_run(args,settings,context,Measures,summarize,'Blob thermal energy',initialize_factory=initialize)
    finally:save(args.output/'energyAudit.json',audit)


def render_recording(args):
    sys.path.insert(0,str(args.publisher_path))
    from warpsph_remote.sources import HDF5Source
    from types import SimpleNamespace as NS
    settings=json.loads((args.output/'settings.json').read_text());folder=next(p.parent for p in (args.output/'warp').glob('*/trajectory.h5'))
    source=HDF5Source(folder/'trajectory.h5');case,ctx,*_=context(settings)
    import torch
    from warpSPH.runner.media import encodeFrames
    from warpSPH.runner.display import closeWindow
    ctx.imagePath=str(folder/'recordingImages');Path(ctx.imagePath).mkdir();ctx.exportPath=str(folder)
    ctx.scratch['_titleProvenance']='Saved trajectory | Blob thermal | '+settings['gammaVariant'];plotter=None
    for i,t in enumerate(source.times):
        state=NS(state=NS(**{k:torch.as_tensor(v,device=ctx.device) for k,v in source.read(i).items()}),t=float(t))
        if plotter is None:plotter=case.setupPlot(ctx,state)
        else:case.updatePlot(ctx,state,plotter,i)
    if hasattr(plotter,'drain'):plotter.drain()
    closeWindow(plotter);(folder/'output.mp4').rename(folder/'outputOriginal.mp4');encodeFrames(ctx.imagePath,str(folder))
    save(args.output/'warp/videoRebuild.json',dict(frames=len(source.times),physicsRerun=False))


def study(root):
    provenance=json.loads((root/'provenance.json').read_text());target=root/'study.json'
    manifest=json.loads(target.read_text()) if target.exists() else dict(stages=[])
    manifest.update(status='running',pid=os.getpid());env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',MPLCONFIGDIR='/tmp/warpsph-blob-thermal-mpl',PYTHONFAULTHANDLER='1')
    try:
        for name in provenance['rows']:
            folder=root/name;data=folder/'warp'
            if (data/'comparisonRun.json').exists() or (data/'failure.json').exists():continue
            command=[sys.executable,'-u',__file__,'row','--output',str(folder)];logpath=root/(name+'.log')
            stage=dict(name=name,command=command,log=str(logpath),status='running');manifest['stages'].append(stage);save(target,manifest)
            print('START',name,flush=True);start=time.perf_counter();last=None
            with logpath.open('w')as log:
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env);stage['pid']=child.pid;save(target,manifest)
                try:
                    while child.poll() is None:
                        if (root/'STOP').exists():child.send_signal(signal.SIGINT)
                        try:child.wait(timeout=5)
                        except subprocess.TimeoutExpired:pass
                        lines=logpath.read_text(errors='replace').replace('\r','\n').splitlines()
                        latest=next((s for s in reversed(lines) if any(v in s for v in ('| t=','BLOB THERMAL RESULT','Remote run registered:','Traceback'))),'')
                        if latest and latest!=last:print(name+': '+latest[:300],flush=True);last=latest
                    code=child.wait()
                except BaseException:child.send_signal(signal.SIGINT);child.wait();raise
            failure=data/'failure.json';numerical=failure.exists() and 'finite-state/velocity watchdog' in json.loads(failure.read_text())['error']
            stage.update(returncode=code,status='numerical_failure' if numerical else ('completed' if code==0 else 'failed'),wallSeconds=time.perf_counter()-start);save(target,manifest)
            print('END',name,stage['status'],flush=True)
            if code and not numerical:raise RuntimeError('Unexpected blob thermal failure: '+name)
            if numerical:
                from warpSPH.runner.media import encodeFrames
                for images in data.glob('*/images'):
                    if not (images.parent/'output.mp4').exists():encodeFrames(str(images),str(images.parent))
        manifest['status']='completed'
    except BaseException as error:manifest.update(status='failed',error=repr(error));raise
    finally:save(target,manifest)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('prepare','study','row','render'));p.add_argument('--output',type=Path,required=True)
    p.add_argument('--publisher-path',type=Path,default=Path('/home/lu26029/dev/warpSPHRemote/python'));args=p.parse_args();args.output=args.output.resolve()
    if args.mode=='prepare':prepare(args.output)
    elif args.mode=='study':study(args.output)
    elif args.mode=='render':render_recording(args)
    else:raise SystemExit(run_row(args))
