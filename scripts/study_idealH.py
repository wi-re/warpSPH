"""Short blob support-policy chain, each link isolates one declared choice.

Research overlays only; fresh midpoint, paper density and native energy fixed.
Every physical row is sequential, daemon-visible and recorded.
"""
import argparse
import copy
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

from compare_spheralMaterials import save,warp_run
from study_blobThermal import Measures,summarize
from study_densityEvolution import context,install_audit as density_audit
from compare_spheralBlobPaper import initialize as blob_initialize

POLICIES=('warp','bounded','shell','nativeLookup','weighted','replacement')


def prepare(root):
    parent=Path('results/materialEOS/density_evolution_2026-10-20').resolve();root.mkdir(parents=True,exist_ok=False);rows=[]
    table=Path('reports/materialEOS/idealH/operators/tables.npz').resolve()
    assert json.loads(table.with_name('audit.json').read_text())['cases']['blobNative']['nativeGpuSupportMaxRelative']<1.e-10
    shutil.copyfile(table,root/'tables.npz')
    for split in (1,2):
        source=parent/f'blob-dt{split}-paper';base=json.loads((source/'settings.json').read_text())
        for policy in POLICIES:
            name=f'blob-dt{split}-{policy}';folder=root/name;folder.mkdir();shutil.copyfile(source/'input.npz',folder/'input.npz')
            s=copy.deepcopy(base);s.update(gammaVariant=name,supportPolicy=policy,frameEvery=base['frameEvery'],
                inputPath=str((folder/'input.npz').resolve()),idealTable=str((root/'tables.npz').resolve()),
                comparisonPolicy='Only support policy changes along declared chain; paper density / fresh midpoint / original compatible energy; '+policy)
            save(folder/'settings.json',s);rows.append(name)
    sources={};(root/'sources').mkdir()
    for name in (__file__,'scripts/crkIdealH.py','scripts/study_densityEvolution.py','scripts/study_blobThermal.py',
        'scripts/compare_spheralMaterials.py','src/warpSPH/modules/adaptiveSupport/optimalSupportOwen.py',
        'src/warpSPH/modules/adaptiveSupport/owenLUT.py','src/warpSPH/modules/adaptiveSupport/wp_psi.py',
        'src/warpSPH/modules/compressibleStage.py','src/warpSPH/modules/crk/evolution.py','src/warpSPH/systems/compSPH.py','src/warpSPH/schemes/compressible/crkSPH.py'):
        p=Path(name);h=hashlib.sha256(p.read_bytes()).hexdigest();shutil.copyfile(p,root/'sources'/f'{p.stem}_{h[:12]}{p.suffix}');sources[str(p)]=h
    save(root/'provenance.json',dict(rows=rows,sources=sources,parent=str(parent),tableSHA256=hashlib.sha256((root/'tables.npz').read_bytes()).hexdigest(),
        policies=POLICIES,references=json.loads((parent/'provenance.json').read_text())['references'],
        scope='Conditional chain: unchanged lattice Owen; absolute h bounds; shell lookup; native exact lookup/range; cross-material volume weighting; native replacement timing. All fresh midpoint, original compatible energy, Eq76 reset/continuity density and Eq75 geometric volumes. No production changes.'))


def install_support(s):
    import torch
    from warpSPHCore import buildVerletList,SupportScheme
    from warpSPH.configurations.compute import neighborSearchOptions
    from crkIdealH import ideal_support
    module=importlib.import_module('warpSPH.schemes.compressible.crkSPH')
    preparation=importlib.import_module('warpSPH.modules.compressibleStage');systems=importlib.import_module('warpSPH.systems.compSPH')
    solve,du,apply,finalize=preparation.solveStageSupport,module.computeCRKdudt,systems.CompSPHSystem.apply_state_update,systems.CompSPHSystem.finalize
    policy=s['supportPolicy'];audit=dict(rhs=0,replacements=0,finalReplacements=0,absoluteBoundCount=0,minimumSupport=None,maximumSupport=0.,samples=[]);latest={}
    def observe(state):
        lo=float(state.supports.min());hi=float(state.supports.max())
        audit['minimumSupport']=lo if audit['minimumSupport'] is None else min(audit['minimumSupport'],lo)
        audit['maximumSupport']=max(audit['maximumSupport'],hi)
    def neighborhood(system,state,c):
        system.adjacency=buildVerletList(state,c.domain,verletScale=c.verletScale,supportMode=SupportScheme.SuperSymmetric,
            priorNeighborhood=system.adjacency,**neighborSearchOptions(c))
        return system.adjacency
    def solve_support(system,state,c,sc,**kwargs):
        assert sc.adaptiveSupportIterations==1 and sc.crkEvolution=='paper' and sc.compatibleEnergy
        assert sc.crkEnergyPartition==s.get('matchingEnergyPartition','native') and sc.crkViscosityGradient=='current' and sc.crkEnergyPredictor=='raw'
        if policy in ('warp','bounded','shell','material','generated','generatedNative'):
            if policy=='material':sc.owenMaterialWeights=True
            if policy in ('generated','generatedNative'):
                sc.owenMaterialWeights=True;sc.owenTable='hermiteShell'
                if policy=='generatedNative':
                    from warpSPH.modules.adaptiveSupport.owenHermite import HermiteOwen
                    from warpSPH.modules.adaptiveSupport.optimalSupportOwen import PsiLUTs
                    key=(c.kernel,c.domain.dim,'hermiteShell')
                    if key not in PsiLUTs:PsiLUTs[key]=HermiteOwen(c.kernel,c.domain.dim,points=1000,inverse_space='linear')
            if policy=='shell':sc.owenTable='shell'
            solve(system,state,c,sc,**kwargs)
            if policy in ('bounded','shell','generated','generatedNative'):
                audit['absoluteBoundCount']+=int(((state.supports>2.)|(state.supports<4.e-5)).sum())
                state.supports=state.supports.clamp(4.e-5,2.)
        elif policy=='replacement':neighborhood(system,state,c)
        else:
            adj=neighborhood(system,state,c)
            state.supports=ideal_support(state,c.domain,adj,s['idealTable'],weighted=policy=='weighted')[0]
            neighborhood(system,state,c)
        observe(state)
    def derivative(state,c,sc,adj,stage,*args):
        result=du(state,c,sc,adj,stage,*args);audit['rhs']+=1
        if policy=='replacement':latest['target']=ideal_support(state,c.domain,adj,s['idealTable'],weighted=True)[0]
        if audit['rhs']<=2 or audit['rhs']%max(2,2*s['steps']//40)==0:
            audit['samples'].append(dict(rhs=audit['rhs'],supportMin=float(state.supports.min()),supportMax=float(state.supports.max()),supportMean=float(state.supports.mean())))
        return result
    def state_update(system,update,spec,**kwargs):
        result=apply(system,update,spec,**kwargs)
        if policy=='replacement':
            system.state.supports=latest.pop('target').clone();audit['replacements']+=1
            observe(system.state)
        return result
    def finish(system,*args,**kwargs):
        accepted=system.state.supports.clone() if policy=='replacement' else None
        result=finalize(system,*args,**kwargs)
        if policy=='replacement' and system.rerunRequest is None:
            # Native IdealH replaces H with the midpoint target at final update.
            # The normal Warp finalizer otherwise restores the midpoint RHS H.
            system.state.supports=accepted;audit['finalReplacements']+=1
        return result
    preparation.solveStageSupport=solve_support;module.computeCRKdudt=derivative
    systems.CompSPHSystem.apply_state_update=state_update;systems.CompSPHSystem.finalize=finish
    return audit


def run_row(args):
    sys.path.insert(0,str(args.publisher_path));s=json.loads((args.output/'settings.json').read_text())
    d=density_audit(s);h=install_support(s)
    save(args.output/'executionSources.json',{name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in (__file__,'scripts/crkIdealH.py','scripts/study_densityEvolution.py')})
    def initialize(c,system,a,s):
        blob_initialize(c,system,a,s);system.state.crkVolumes=None
        c.scratch['_titleProvenance']='IdealH support | '+s['gammaVariant']+' | paper density / fresh midpoint'
    try:return warp_run(args,s,context,Measures,summarize,'IdealH support',initialize_factory=initialize)
    finally:save(args.output/'densityAudit.json',d);save(args.output/'supportAudit.json',h)


def study(root, row_script=None):
    rows=json.loads((root/'provenance.json').read_text())['rows'];target=root/'study.json'
    manifest=json.loads(target.read_text()) if target.exists() else dict(stages=[]);manifest.update(status='running',pid=os.getpid())
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',MPLCONFIGDIR='/tmp/warpsph-idealh-mpl',PYTHONFAULTHANDLER='1')
    try:
        for name in rows:
            folder=root/name
            if (folder/'warp/comparisonRun.json').exists() or (folder/'warp/failure.json').exists():continue
            path=root/(name+'.log');command=[sys.executable,'-u',str(row_script or __file__),'row','--output',str(folder)]
            stage=dict(name=name,status='running',command=command,log=str(path));manifest['stages'].append(stage);save(target,manifest)
            print('START',name,flush=True);start=time.perf_counter();last=None
            with path.open('w') as log:
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env);stage['pid']=child.pid;save(target,manifest)
                try:
                    while child.poll() is None:
                        if (root/'STOP').exists():child.send_signal(signal.SIGINT)
                        try:child.wait(timeout=5)
                        except subprocess.TimeoutExpired:pass
                        lines=path.read_text(errors='replace').replace('\r','\n').splitlines();line=next((v for v in reversed(lines) if any(k in v for k in ('| t=','BLOB THERMAL RESULT','Remote run registered:','Traceback'))),'')
                        if line and line!=last:print(name+': '+line[:230],flush=True);last=line
                    code=child.wait()
                except BaseException:child.send_signal(signal.SIGINT);child.wait();raise
            f=folder/'warp/failure.json';numerical=f.exists() and 'finite-state/velocity watchdog' in json.loads(f.read_text())['error']
            stage.update(status='numerical_failure' if numerical else 'completed' if code==0 else 'failed',returncode=code,wallSeconds=time.perf_counter()-start);save(target,manifest)
            if numerical:
                from warpSPH.runner.media import encodeFrames
                for images in (folder/'warp').glob('*/images'):
                    if not (images.parent/'output.mp4').exists():encodeFrames(str(images),str(images.parent))
            elif code:raise RuntimeError('Unexpected support-study error: '+name)
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
