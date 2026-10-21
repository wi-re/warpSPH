"""Isolate CRK density reset/continuity timing; geometric volumes stay fixed.

Fresh midpoint, original compatible energy, no volume or support transplant.
Every physical row is daemon-visible and recorded through the shared runner.
"""
import argparse
import copy
from dataclasses import replace
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import numpy as np

from compare_spheralMaterials import save, warp_run
from study_blobThermal import context as blob_context, Measures as BlobMeasures
from compare_spheralBlobPaper import initialize as blob_initialize
from study_blobThermal import summarize
from study_crkEnergySuite import context as simple_context, sample, CASES, Measures as SimpleMeasures


def prepare(root):
    base=Path('results/materialEOS/blob_matrix_2026-10-20/matched').resolve()
    blob=json.loads((base/'settings.json').read_text());root.mkdir(parents=True,exist_ok=False);rows=[]
    for case in ('wave','sod','blob'):
        initial=dict(np.load(base/'input.npz')) if case=='blob' else sample(case)
        cfg=blob if case=='blob' else CASES[case]
        grid=np.asarray(blob['timeGrid']) if case=='blob' else np.full(cfg['steps'],cfg['end']/cfg['steps'])
        for split in (1,2):
            sequence=np.repeat(grid/split,split)
            for schedule in ('summation','paper'):
                name=f'{case}-dt{split}-{schedule}';folder=root/name;folder.mkdir();np.savez_compressed(folder/'input.npz',**initial)
                s=copy.deepcopy(cfg);s.update(case=case,gammaVariant=name,kernel='B8',schedule=schedule,
                    inputPath=str((folder/'input.npz').resolve()),timeGrid=sequence.tolist(),dt=float(sequence[0]),
                    steps=len(sequence),exactSteps=len(sequence),end=float(sequence.sum()),frameEvery=max(1,len(sequence)//40),
                    choice=dict(compatible=True,thermal='base'),thermalGamma=5/3 if case=='blob' else cfg['gamma'],
                    comparisonPolicy='Density timing only: '+schedule+'; geometric Eq75 every RHS, fresh midpoint, original compatible energy, Owen support')
                save(folder/'settings.json',s);rows.append(name)
    sources={};(root/'sources').mkdir()
    for name in (__file__,'scripts/study_blobThermal.py','scripts/study_crkEnergySuite.py','scripts/compare_spheralMaterials.py',
        'scripts/compare_spheralBlobPaper.py','src/warpSPH/modules/crk/evolution.py','src/warpSPH/schemes/compressible/crkSPH.py',
        'src/warpSPH/systems/compSPH.py','src/warpSPH/modules/crk/terms.py','src/warpSPH/configurations/crkSPH.py',
        'benchmarks/eos/paperSpheral/paperPolicies.cc'):
        p=Path(name);sha=hashlib.sha256(p.read_bytes()).hexdigest();shutil.copyfile(p,root/'sources'/f'{p.stem}_{sha[:12]}{p.suffix}');sources[str(p)]=sha
    save(root/'provenance.json',dict(rows=rows,sources=sources,baseline=str(base),
        references=dict(blobGrid=str(Path('results/materialEOS/crk_heating_v2_2026-10-20/spheral-grid-averaged').resolve()),
            blobHalf=str(Path('results/materialEOS/crk_heating_v2_2026-10-20/spheral-halfGrid-averaged').resolve())),
        scope='Only density scheduling varies within each case/grid. Eq75 geometric volume, Owen H, fresh midpoint and original compatible energy fixed. No experimental thermal RHS. Stored native blob reference retains native H and final-work entropy differences.'))


def context(s):
    if s['case']=='blob':return blob_context(dict(s,matrixControls=dict(solver=dict(crkEvolution=s['schedule']))))
    from warpSPH.runner import RunConfiguration
    case,_,sim,params,solver=simple_context(s)
    solver.update(crkEvolution=s['schedule'],crkDensityPolicy='massReplacement',
        crkViscosityParams=dict(oneSidedWork=False,viscosityKernelScale=4.),diffusionParams=dict(C_l=2.,C_q=1.))
    return case,RunConfiguration(case.name,simulation=sim,params=params,solver=solver).prepare(case),sim,params,solver


def weighted_change(a,b,m):
    """Relative L1 and maximum pointwise relative discrepancy, b as reference."""
    delta=abs(a-b)
    return dict(relativeL1=float(np.dot(m,delta)/max(np.dot(m,abs(b)),1.e-100)),
        relativeMax=float(np.max(delta/np.maximum(abs(b),1.e-100))))


def install_audit(s):
    from warpSPHBootstrap import bootstrap
    bootstrap(precision='float64')
    import torch
    module=importlib.import_module('warpSPH.schemes.compressible.crkSPH')
    evolution=importlib.import_module('warpSPH.modules.crk.evolution')
    eos=importlib.import_module('warpSPH.modules.eos.materials')
    original_factor,original_prepare,original_update,original_eos=module.crkFactors,evolution.prepareStep,evolution.updateContinuityState,eos.evaluateEOS
    audit=dict(rhs=0,resets=[],stageComparisons=[],continuityUpdates=0,maxContinuityUpdateError=0.,maxDensityAliasError=0.,negativeEOSCalls=0)
    cpu=lambda a:a.detach().cpu().numpy()
    def factors(state,*args,**kwargs):
        assert kwargs.get('volumes') is None,'geometric volume must not become m/rho or evolved volume'
        assert kwargs['computeDensity']==(s['schedule']=='summation') and kwargs['densityPolicy']=='massReplacement'
        result=original_factor(state,*args,**kwargs);audit['rhs']+=1
        # Static shadow evaluation only: never replace the physical stage rho.
        if audit['rhs']%max(2,2*s['steps']//40)==0 or audit['rhs']<=2:
            summed=original_factor(state,*args,**dict(kwargs,computeDensity=True))[1] if s['schedule']=='paper' else result[1]
            row=dict(rhs=audit['rhs'],**weighted_change(cpu(result[1]),cpu(summed),cpu(state.masses)))
            if state.crkMassDensities is not None:
                audit['maxDensityAliasError']=max(audit['maxDensityAliasError'],float((state.densities-state.crkMassDensities).abs().max()))
            audit['stageComparisons'].append(row)
        return result
    def prepare(system,*args,**kwargs):
        state=system.state;before=state.densities.clone();u=state.internalEnergies.clone()
        original_prepare(system,*args,**kwargs)
        after=state.densities;ratio=after/before;m=state.masses;g=s['thermalGamma']
        assert torch.equal(u,state.internalEnergies),'density reset must not adjust thermal energy'
        audit['resets'].append(dict(step=len(audit['resets']),time=float(system.t),
            **weighted_change(cpu(after),cpu(before),cpu(m)),
            signedEntropyJump=float(torch.dot(m,(1-g)*torch.log(ratio))/m.sum()),
            entropyJumpL1=float(torch.dot(m,abs((1-g)*torch.log(ratio)))/m.sum())))
    def update(system,derivative,spec,*args,**kwargs):
        before=system.state.crkMassDensities.clone();expected=before+spec.derivative_dt*derivative.drhodt
        original_update(system,derivative,spec,*args,**kwargs);audit['continuityUpdates']+=1
        audit['maxContinuityUpdateError']=max(audit['maxContinuityUpdateError'],float((expected-system.state.densities).abs().max()))
        assert system.state.crkVolumes is None
    def evaluate(rho,*args,**kwargs):
        u=kwargs.get('u')
        audit['negativeEOSCalls']+=int(bool((rho<=0).any() or (u is not None and (u<0).any())))
        return original_eos(rho,*args,**kwargs)
    module.crkFactors=factors;evolution.prepareStep=prepare;evolution.updateContinuityState=update;eos.evaluateEOS=evaluate
    return audit


def run_row(args):
    sys.path.insert(0,str(args.publisher_path));s=json.loads((args.output/'settings.json').read_text())
    save(args.output/'executionSources.json',{name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
        (__file__,'src/warpSPH/modules/crk/evolution.py','src/warpSPH/schemes/compressible/crkSPH.py','src/warpSPH/systems/compSPH.py')})
    audit=install_audit(s)
    def initialize(c,system,a,s):
        if s['case']=='blob':blob_initialize(c,system,a,s)
        system.state.crkVolumes=None
        c.scratch['_titleProvenance']='Density timing | '+s['gammaVariant']+' | geometric volume / fresh midpoint'
    try:return warp_run(args,s,context,BlobMeasures if s['case']=='blob' else SimpleMeasures,summarize,'Density evolution',initialize_factory=initialize)
    finally:save(args.output/'densityAudit.json',audit)


def study(root):
    rows=json.loads((root/'provenance.json').read_text())['rows'];target=root/'study.json'
    manifest=json.loads(target.read_text()) if target.exists() else dict(stages=[])
    if 'error' in manifest:manifest.setdefault('previousErrors',[]).append(manifest.pop('error'))
    manifest.update(status='running',pid=os.getpid())
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',MPLCONFIGDIR='/tmp/warpsph-density-mpl',PYTHONFAULTHANDLER='1')
    try:
        for name in rows:
            folder=root/name
            if (folder/'warp/comparisonRun.json').exists() or (folder/'warp/failure.json').exists():continue
            logpath=root/(name+'.log');command=[sys.executable,'-u',__file__,'row','--output',str(folder)]
            stage=dict(name=name,status='running',command=command,log=str(logpath));manifest['stages'].append(stage);save(target,manifest)
            print('START',name,flush=True);start=time.perf_counter();last=None
            with logpath.open('w') as log:
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env);stage['pid']=child.pid;save(target,manifest)
                try:
                    while child.poll() is None:
                        if (root/'STOP').exists():child.send_signal(signal.SIGINT)
                        try:child.wait(timeout=5)
                        except subprocess.TimeoutExpired:pass
                        lines=logpath.read_text(errors='replace').replace('\r','\n').splitlines()
                        latest=next((v for v in reversed(lines) if any(k in v for k in ('| t=','BLOB THERMAL RESULT','Remote run registered:','Traceback'))),'')
                        if latest and latest!=last:print(name+': '+latest[:250],flush=True);last=latest
                    code=child.wait()
                except BaseException:child.send_signal(signal.SIGINT);child.wait();raise
            failure=folder/'warp/failure.json';numerical=failure.exists() and 'finite-state/velocity watchdog' in json.loads(failure.read_text())['error']
            stage.update(status='numerical_failure' if numerical else 'completed' if code==0 else 'failed',returncode=code,wallSeconds=time.perf_counter()-start);save(target,manifest)
            if numerical:
                from warpSPH.runner.media import encodeFrames
                for images in (folder/'warp').glob('*/images'):
                    if not (images.parent/'output.mp4').exists():encodeFrames(str(images),str(images.parent))
            elif code:raise RuntimeError('Unexpected density study failure: '+name)
            print('END',name,stage['status'],flush=True)
        manifest['status']='completed'
    except BaseException as error:manifest.update(status='failed',error=repr(error));raise
    finally:save(target,manifest)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('prepare','study','row'));p.add_argument('--output',required=True,type=Path)
    p.add_argument('--publisher-path',type=Path,default=Path('/home/lu26029/dev/warpSPHRemote/python'));args=p.parse_args();args.output=args.output.resolve()
    if args.mode=='prepare':prepare(args.output)
    elif args.mode=='study':study(args.output)
    else:raise SystemExit(run_row(args))
