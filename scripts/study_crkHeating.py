"""Isolated CRK interim Q heating, fresh RK2 and final entropy work held fixed.

Sequential daemon/video trajectories, including a short native observer-parity
replay before the full controls. Static pair/stage analyses are separate.
"""
import argparse
import copy
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


def save(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def node_checks(folder):
    a=np.load(folder/'heating_nodeAudit.npz')['rows'];real=a[:,3]>0
    expected=a[:,33] if 'averaged' in folder.name else a[:,32]
    error=float(np.max(np.abs(a[real,31]-expected[real])/(1+np.abs(expected[real]))))
    if error>1.e-10:raise ValueError(('Native heating formula mismatch',folder,error))
    return error


def partition_controls(root):
    """One final-work policy at a time; numerical failures remain observations."""
    output=root/'partitionControls';output.mkdir(exist_ok=True)
    library=Path('results/materialEOS/spheral-heating-sign-policies/libSpheralPaperPolicies.so').resolve()
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',MPLCONFIGDIR='/tmp/warpsph-heating-mpl')
    manifest=json.loads((output/'study.json').read_text()) if (output/'study.json').exists() else dict(stages=[])
    if 'error'in manifest:manifest.setdefault('previousErrors',[]).append(manifest.pop('error'))
    manifest.update(status='running',pid=os.getpid(),library=str(library),librarySHA256=hashlib.sha256(library.read_bytes()).hexdigest())
    (output/'sources').mkdir(exist_ok=True)
    for name in ('benchmarks/eos/paperSpheral/paperPolicies.cc','scripts/run_spheralBlobExample.py',__file__):
        p=Path(name);dest=output/'sources'/p.name
        if dest.exists() and dest.read_bytes()!=p.read_bytes():dest=dest.with_name(p.stem+'_'+hashlib.sha256(p.read_bytes()).hexdigest()[:12]+p.suffix)
        shutil.copy2(p,dest)
    def execute(name,command,native=False):
        record=dict(name=name,command=list(map(str,command)),status='running',log=str(output/(name+'.log')))
        manifest['stages'].append(record);save(output/'study.json',manifest);start=time.perf_counter();print('START',name,flush=True)
        with Path(record['log']).open('w') as log:
            child=subprocess.Popen(record['command'],stdout=log,stderr=subprocess.STDOUT,env=dict(env,LD_PRELOAD=str(library)) if native else env)
            record['pid']=child.pid;save(output/'study.json',manifest)
            try:
                while child.poll() is None:
                    if (root/'STOP').exists():child.send_signal(signal.SIGINT)
                    try:child.wait(timeout=5)
                    except subprocess.TimeoutExpired:pass
                    lines=Path(record['log']).read_text(errors='replace').replace('\r','\n').splitlines()
                    latest=next((s for s in reversed(lines) if any(x in s for x in ('BLOB ','| t=','Remote run registered:','Traceback'))),'')
                    if latest:print(name+': '+latest[:350],flush=True)
                rc=child.wait()
            except BaseException:
                child.send_signal(signal.SIGINT);child.wait();raise
        record.update(status='completed' if rc==0 else 'failed',returncode=rc,wallSeconds=time.perf_counter()-start)
        save(output/'study.json',manifest);print('END',name,rc,round(record['wallSeconds'],2),'s',flush=True)
        if (root/'STOP').exists() or (rc and 'SimulationStopped' in Path(record['log']).read_text(errors='replace')):raise KeyboardInterrupt('Stopped by user')
        return rc
    try:
        for grid,partition in (('grid','entropy'),('grid','dudt'),('halfGrid','dudt')):
            name='spheral-'+grid+'-'+partition;folder=output/name
            if not (folder/'summary.json').exists():
                rc=execute(name,['results/materialEOS/spheral-build/install/.venv/bin/python','-u','scripts/run_spheralBlobExample.py','run',
                    '--paper','--nx','128','--tkh','1','--strict-positive','--integrator','fresh','--partition',partition,
                    '--av-gradient','crk','--heating','one-sided','--time-grid',root/(grid+'.json'),'--output',folder],True)
            else:rc=int(json.loads((folder/'summary.json').read_text())['status']!='completed')
            if not (folder/'summary.json').exists():raise RuntimeError('Native setup failure: '+name)
            row=json.loads((folder/'summary.json').read_text())
            if rc and row['positiveEnergy']:raise RuntimeError('Unexpected failure: '+name)
            if not (folder/'output.mp4').exists():
                if execute(name+'-render',[sys.executable,'scripts/run_spheralBlobExample.py','render','--output',folder]):raise RuntimeError('Render failed')
            if partition=='entropy':
                prior=Path(json.loads((root/'provenance.json').read_text())['reusedNative']['grid'])
                a,b=(dict(np.load(p/'finalState.npz')) for p in (folder,prior))
                parity={k:float(np.abs(a[k]-b[k]).max()) for k in a}
                save(output/'observerParity.json',parity)
                if max(parity.values())>1.e-8:raise ValueError('Stage observer changed full trajectory')
        for grid in ('grid','halfGrid'):
            name='warp-'+grid+'-dudt';folder=output/name;folder.mkdir(exist_ok=True)
            if (folder/'warp/comparisonRun.json').exists() or (folder/'warp/failure.json').exists():continue
            parent=root/('warp-'+grid+'-one-sided')
            shutil.copy2(parent/'input.npz',folder/'input.npz');settings=json.loads((parent/'settings.json').read_text())
            settings['matrixControls']['solver']['crkEnergyPartition']='spheral';settings['gammaVariant']=name
            settings['comparisonPolicy']='Only final-work partition changes from entropy to sign-filtered node specific dudt; one-sided raw predictor and fresh CRK RK2 fixed.'
            save(folder/'settings.json',settings)
            rc=execute(name,[sys.executable,'-u','scripts/study_blobPaperMatrix.py','row','--output',folder])
            if rc:
                failure=folder/'warp/failure.json'
                if not failure.exists():raise RuntimeError('Warp setup failure: '+name)
                h=json.loads((folder/'warp/diagnosticHistory.json').read_text())
                if h[-1]['minEnergy'] is not None and h[-1]['minEnergy']>0:raise RuntimeError('Unexpected Warp failure: '+name)
        manifest['status']='completed'
    except BaseException as error:
        manifest.update(status='stopped' if isinstance(error,KeyboardInterrupt) else 'failed',error=repr(error));raise
    finally:save(output/'study.json',manifest)


def study(root):
    root.mkdir(parents=True,exist_ok=False)
    av=Path('results/materialEOS/spheral_av_2026-10-20').resolve()
    history=json.loads((av/'grid.json').read_text());grid=np.diff([r['time'] for r in history])
    save(root/'grid.json',history);save(root/'halfGrid.json',[dict(time=float(t)) for t in np.r_[0,np.cumsum(np.repeat(grid/2,2))]])
    save(root/'auditGrid.json',history[:9])
    base=json.loads(Path('results/materialEOS/blob_paper_short_2026-10-20/settings.json').read_text())
    library=Path('results/materialEOS/spheral-heating-policies/libSpheralPaperPolicies.so').resolve()
    provenance=dict(library=str(library),librarySHA256=hashlib.sha256(library.read_bytes()).hexdigest(),
        reusedNative={g:str(av/(g+'-crk')) for g in ('grid','halfGrid')},
        scope='Within each solver/grid only raw viscous heating changes: Q_j versus (Q_i+Q_j)/2. Fresh CRK AV gradient, fresh midpoint, raw predictor and final-work entropy partition fixed. Cross-solver density/support differences remain.')
    paths=['benchmarks/eos/paperSpheral/paperPolicies.cc','scripts/run_spheralBlobExample.py',__file__,
        'src/warpSPH/modules/crk/dudt.py','src/warpSPH/modules/crk/accel.py','src/warpSPH/modules/crk/energyControl.py',
        'src/warpSPH/modules/compSPH/dudt.py','src/warpSPH/modules/compSPH/accel.py','src/warpSPH/modules/compSPH/balance.py',
        'src/warpSPH/modules/compSPH/pairWork.py','scripts/compare_spheralBlobPaper.py']
    (root/'sources').mkdir();provenance['sources']={}
    for name in paths:
        p=Path(name);shutil.copy2(p,root/'sources'/(name.replace('/','_')))
        provenance['sources'][name]=hashlib.sha256(p.read_bytes()).hexdigest()
    save(root/'provenance.json',provenance)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',MPLCONFIGDIR='/tmp/warpsph-heating-mpl')
    manifest=dict(status='running',pid=os.getpid(),stages=[])
    def execute(name,command,native=False):
        stage=dict(name=name,command=list(map(str,command)),status='running',log=str(root/(name+'.log')))
        manifest['stages'].append(stage);save(root/'study.json',manifest)
        print('START',name,flush=True);start=time.perf_counter()
        with Path(stage['log']).open('w') as log:
            child=subprocess.Popen(stage['command'],stdout=log,stderr=subprocess.STDOUT,env=dict(env,LD_PRELOAD=str(library)) if native else env)
            stage['pid']=child.pid;save(root/'study.json',manifest)
            try:
                while child.poll() is None:
                    if (root/'STOP').exists():child.send_signal(signal.SIGINT)
                    try:child.wait(timeout=5)
                    except subprocess.TimeoutExpired:pass
                    lines=Path(stage['log']).read_text(errors='replace').replace('\r','\n').splitlines()
                    latest=next((s for s in reversed(lines) if any(x in s for x in ('BLOB ','| t=','Remote run registered:','Traceback','stopping'))),'')
                    if latest:print(name+': '+latest[-220:],flush=True)
                rc=child.wait()
            except BaseException:
                child.send_signal(signal.SIGINT);child.wait();raise
        stage.update(status='completed' if rc==0 else 'failed',returncode=rc,wallSeconds=time.perf_counter()-start)
        save(root/'study.json',manifest);print('END',name,stage['status'],round(stage['wallSeconds'],2),'s',flush=True)
        if (root/'STOP').exists() or (rc and 'SimulationStopped' in Path(stage['log']).read_text(errors='replace')):raise KeyboardInterrupt('Stopped by user')
        if rc:raise RuntimeError('Stage failed: '+name)
    def native(name,heating,gridfile,tkh):
        folder=root/name
        execute(name,['results/materialEOS/spheral-build/install/.venv/bin/python','-u','scripts/run_spheralBlobExample.py',
            'run','--paper','--nx','128','--tkh',str(tkh),'--strict-positive','--integrator','fresh','--partition','entropy',
            '--av-gradient','crk','--heating',heating,'--time-grid',gridfile,'--output',folder],True)
        node_checks(folder)
        execute(name+'-render',[sys.executable,'scripts/run_spheralBlobExample.py','render','--output',folder])
    try:
        native('audit-one-sided','one-sided',root/'auditGrid.json',history[8]['time']/base['tKH'])
        import h5py
        final=dict(np.load(root/'audit-one-sided/finalState.npz'))
        with h5py.File(av/'grid-crk/trajectory.h5') as f:
            old=f['states']['frame_0000008']
            parity={k:float(np.abs(v-old[k][()]).max()) for k,v in final.items()}
        save(root/'observerParity.json',parity)
        if max(parity.values())>1.e-10:raise ValueError('Read-only observer changed native trajectory')
        for g in ('grid','halfGrid'):
            native('spheral-'+g+'-averaged','averaged',root/(g+'.json'),1.)
        for g in ('grid','halfGrid'):
            dt=grid if g=='grid' else np.repeat(grid/2,2)
            for heating in ('averaged','one-sided'):
                folder=root/('warp-'+g+'-'+heating);folder.mkdir()
                shutil.copy2('results/materialEOS/blob_paper_short_2026-10-20/input.npz',folder/'input.npz')
                settings=copy.deepcopy(base);settings.update(gammaVariant=folder.name,precision='float64',
                    matrixControls=dict(solver=dict(crkEnergyPartition='entropy',crkEnergyPredictor='raw',
                        crkViscosityParams=dict(oneSidedWork=heating=='one-sided',viscosityKernelScale=4.))),
                    steps=len(dt),dt=float(dt[0]),end=float(dt.sum()),timeGrid=dt.tolist(),frameEvery=10,
                    comparisonPolicy=provenance['scope'],source='Isolated interim heating control, working-tree Warp.')
                save(folder/'settings.json',settings)
                execute(folder.name,[sys.executable,'-u','scripts/study_blobPaperMatrix.py','row','--output',folder])
        manifest['status']='completed'
    except BaseException as error:
        manifest.update(status='stopped' if isinstance(error,KeyboardInterrupt) else 'failed',error=repr(error));raise
    finally:save(root/'study.json',manifest)


def analyze(root,report):
    import h5py
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report.mkdir(parents=True,exist_ok=True)
    provenance=json.loads((root/'provenance.json').read_text());folders={}
    for g in ('grid','halfGrid'):
        folders['spheral-'+g+'-one-sided']=Path(provenance['reusedNative'][g])
        folders['spheral-'+g+'-averaged']=root/('spheral-'+g+'-averaged')
        for h in ('one-sided','averaged'):folders['warp-'+g+'-'+h]=root/('warp-'+g+'-'+h)/'warp'
    rows={};histories={};finals={};startups={}
    for name,folder in folders.items():
        native=name.startswith('spheral');run=json.loads((folder/('summary.json' if native else 'comparisonRun.json')).read_text())
        h=json.loads((folder/'diagnosticHistory.json').read_text());histories[name]=h
        a=dict(np.load(folder/'finalState.npz'));finals[name]=a
        startup=dict(np.load(folder/'initializedState.npz')) if native else dict(np.load(folder.parent/'input.npz'));startups[name]=startup
        result=run.get('report',{})
        rows[name]=dict(status=run.get('status','completed'),steps=len(h)-1,end=h[-1]['tKH'],coldMass=h[-1]['cloudMassFraction'],
            minEnergy=min(r['minEnergy'] for r in h),minDensity=min(r['minDensity'] for r in h),
            energyDrift=max(abs(r['relativeEnergyDrift']) for r in h),momentumError=max(abs(r['relativeMomentumError']) for r in h),
            materialMassError=max(abs(r['materialMassError']) for r in h),wallSeconds=run.get('wallSeconds',result.get('wallSeconds')))
        if not native:
            rows[name]['timing']=run.get('timing',run.get('summary',{}))
            rows[name]['runnerMeanStepMilliseconds']=run['runnerMeanStepMilliseconds']
        if rows[name]['wallSeconds'] is not None:rows[name]['msPerStep']=1000*rows[name]['wallSeconds']/rows[name]['steps']
    def distance(a,b):
        np.testing.assert_array_equal(a['UIDs'],b['UIDs']);d=a['positions']-b['positions'];L=np.array([40.,10.]);d-=L*np.round(d/L);cloud=a['materials']==1
        return dict(allPositionRMS=float(np.sqrt(np.mean(d**2))),cloudPositionRMS=float(np.sqrt(np.mean(d[cloud]**2))),
            energyRMS=float(np.sqrt(np.mean((a['internalEnergies']-b['internalEnergies'])**2))))
    comparisons={};initial={}
    for solver in ('spheral','warp'):
        for g in ('grid','halfGrid'):
            a,b=(solver+'-'+g+'-'+h for h in ('averaged','one-sided'))
            comparisons[a+' vs '+b]=distance(finals[a],finals[b])
            initial[solver+'-'+g]={k:float(np.abs(startups[a][k]-startups[b][k]).max()) for k in startups[b] if k in startups[a]}
            for key,value in initial[solver+'-'+g].items():
                if value>1.e-12:raise ValueError(('Unequal initial state',solver,g,key,value))
        for heating in ('averaged','one-sided'):
            a,b=(solver+'-'+g+'-'+heating for g in ('grid','halfGrid'))
            comparisons[a+' vs '+b]=distance(finals[a],finals[b])
    for g in ('grid','halfGrid'):
        for heating in ('averaged','one-sided'):
            a,b=('warp-'+g+'-'+heating,'spheral-'+g+'-'+heating);comparisons[a+' vs '+b]=distance(finals[a],finals[b])
    stages={}
    for folder in [root/'audit-one-sided',root/'spheral-grid-averaged',root/'spheral-halfGrid-averaged']:
        n=np.load(folder/'heating_nodeAudit.npz')['rows'];p=np.load(folder/'heating_pairAudit.npz')['rows'];f=np.load(folder/'partitionAudit.npz')['rows']
        data=dict(formulaRelativeMaximum=node_checks(folder),phases={})
        for phase in np.unique(n[:,0]):
            b=n[(n[:,0]==phase)&(n[:,3]>0)];cloud=b[:,1]==1
            data['phases'][str(int(phase))]=dict(nodes=len(b),nativeVsAveragedRateRMS=float(np.sqrt(np.mean((b[:,32]-b[:,33])**2))),
                cloudNativeVsAveragedRateRMS=float(np.sqrt(np.mean((b[cloud,32]-b[cloud,33])**2))),
                rawMinimum=float(b[:,31].min()),rawMaximum=float(b[:,31].max()))
        pairmass={tuple(row[1:3].astype(int)):row[4] for row in n[n[:,0]==2]}
        massi=np.array([pairmass[tuple(row[1:3].astype(int))] for row in p]);massj=np.array([pairmass[tuple(row[3:5].astype(int))] for row in p])
        data['sampledPairs']=len(p);data['viscousPairPowerDifferenceMax']=float(np.max(np.abs(massi*(p[:,10]-p[:,12])+massj*(p[:,11]-p[:,13]))))
        active=np.abs(f[:,4])>1.e-12
        data['finalPartition']=dict(sampledPairs=len(f),activePairs=int(active.sum()),
            entropyVsDudtFractionRMS=float(np.sqrt(np.mean((f[active,10]-f[active,12])**2))),
            entropyVsDudtFractionMaximum=float(np.max(np.abs(f[active,10]-f[active,12]))),
            baseInstantaneousVsFinalWorkSignChanges=int(((f[:,4]*f[:,5]<0)&active).sum()),
            signChoiceFractionMaximum=float(np.max(np.abs(f[active,10]-f[active,11]))))
        stages[folder.name]=data
    partitions={};partitionStages={}
    controls=root/'partitionControls'
    for folder in sorted(controls.glob('*')):
        if not folder.is_dir() or folder.name=='sources':continue
        native=folder.name.startswith('spheral');data=folder if native else folder/'warp'
        if not (data/'diagnosticHistory.json').exists():continue
        h=json.loads((data/'diagnosticHistory.json').read_text());meta=data/('summary.json' if native else 'comparisonRun.json')
        if not meta.exists():meta=data/'failure.json'
        run=json.loads(meta.read_text())
        partitions[folder.name]=dict(status=run.get('status','completed' if run.get('passed',False) else 'failed'),
            steps=len(h)-1,finalTKH=h[-1]['tKH'],minEnergy=min(r['minEnergy'] for r in h),
            energyDrift=max(abs(r['relativeEnergyDrift']) for r in h),momentumError=max(abs(r['relativeMomentumError']) for r in h),
            wallSeconds=run.get('wallSeconds'),history=str(data/'diagnosticHistory.json'))
        if not native and 'runnerMeanStepMilliseconds'in run:partitions[folder.name]['runnerMeanStepMilliseconds']=run['runnerMeanStepMilliseconds']
        fpath=folder/'partitionAudit.npz'
        if native and fpath.exists():
            f=np.load(fpath)['rows']
            if f.shape[1]<17:continue
            summary={}
            for step in np.unique(f[:,16]):
                b=f[f[:,16]==step];active=np.abs(b[:,4])>1.e-12
                si,sj,w=b[:,6],b[:,7],b[:,15];small=np.minimum(abs(si),abs(sj));large=np.maximum(abs(si),abs(sj))
                smallChoice=((w>=0)&(si>=sj))|((w<0)&(si<sj))
                estimated=np.where(smallChoice,small/(small+large),large/(small+large))
                estimated=np.where((abs(w)<=1.e-15)|(abs(si-sj)<=1.e-15),.5,estimated)
                summary[str(int(step))]=dict(sampledPairs=len(b),activePairs=int(active.sum()),
                    rhsEstimatedVsFinalWorkSignChanges=int(((b[:,4]*b[:,15]<0)&active).sum()),
                    rhsEstimatedVsFinalEntropyFractionMaximum=float(np.max(np.abs(estimated[active]-b[active,10]))),
                    entropyVsNodeDudtFractionRMS=float(np.sqrt(np.mean((b[active,10]-b[active,12])**2))),
                    entropyVsNodeDudtFractionMaximum=float(np.max(np.abs(b[active,10]-b[active,12]))))
            partitionStages[folder.name]=summary
    save(report/'comparison.json',dict(provenance=provenance,rows=rows,comparisons=comparisons,initialDifferences=initial,stages=stages,
        partitionControls=partitions,partitionStages=partitionStages))
    fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    for name,h in histories.items():
        for ax,key in zip(axes,('cloudMassFraction','minEnergy','relativeEnergyDrift')):
            ax.plot([r['tKH'] for r in h],[r[key] for r in h],label=name,ls='--' if 'halfGrid' in name else '-')
    for ax,key in zip(axes,('Cold mass fraction','Minimum u','Relative energy drift')):ax.set(xlabel='t / KH',ylabel=key);ax.grid(alpha=.2)
    axes[0].legend(fontsize=6);fig.savefig(report/'diagnostics.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(16,7),constrained_layout=True)
    for ax,(name,a) in zip(axes.flat,finals.items()):
        ax.scatter(*a['positions'].T,c=a['materials'],cmap='coolwarm',s=2,vmin=0,vmax=1)
        ax.set(xlim=(3.,12.),ylim=(1.,9.),aspect='equal',title=name)
    fig.savefig(report/'interfaces.png',dpi=180);plt.close(fig)
    if partitions:
        fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
        for name,row in partitions.items():
            h=json.loads(Path(row['history']).read_text())
            for ax,key in zip(axes,('minEnergy','relativeEnergyDrift')):
                ax.plot([r['tKH'] for r in h],[r[key] for r in h],label=name)
        axes[0].axhline(0,color='black',lw=.7);axes[0].set(ylabel='Minimum u');axes[1].set(ylabel='Relative energy drift')
        for ax in axes:ax.set(xlabel='t / KH');ax.grid(alpha=.2)
        axes[0].legend(fontsize=7);fig.savefig(report/'partitions.png',dpi=180);plt.close(fig)
    print(json.dumps(dict(rows=rows,comparisons=comparisons,stages=stages,partitionControls=partitions,partitionStages=partitionStages),indent=2))


def verify(root,report):
    import urllib.request
    from warpsph_remote.daemon import descriptor
    from warpsph_remote.sources import HDF5Source
    info=descriptor()
    def get(path):
        request=urllib.request.Request(info['endpoint']+path,headers={'Authorization':'Bearer '+info['token']})
        with urllib.request.urlopen(request,timeout=15) as response:return json.load(response)
    runs={r['id']:r for r in get('/v1/runs')['runs']};rows={}
    provenance=json.loads((root/'provenance.json').read_text())
    registrations=list(root.glob('*/registration.json'))+list(root.glob('*/warp/registration.json'))
    registrations+=list((root/'partitionControls').glob('*/registration.json'))+list((root/'partitionControls').glob('*/warp/registration.json'))
    registrations += [Path(p)/'registration.json' for p in provenance['reusedNative'].values()]
    for registration in registrations:
        folder=registration.parent;rid=json.loads(registration.read_text())['remoteRunID'];run=runs[rid]
        path=folder
        if not (path/'trajectory.h5').exists():
            meta=json.loads((folder/'comparisonRun.json').read_text());path=Path(meta['exportPath'])
        source=HDF5Source(path/'trajectory.h5');history=get('/v1/runs/'+rid+'/history')
        np.testing.assert_allclose(history['times'],source.times,rtol=0,atol=1.e-12)
        if (folder/'nativeConfiguration.json').exists() and run['status']=='failed' and not (folder/'videoTitleRevision.json').exists():
            print('Render retained native states with corrected policy label:',folder.name,flush=True)
            with (folder/'videoTitleRevision.log').open('w') as log:
                subprocess.run([sys.executable,'scripts/run_spheralBlobExample.py','render','--output',str(folder)],
                    stdout=log,stderr=subprocess.STDOUT,check=True,env=dict(os.environ,MPLCONFIGDIR='/tmp/warpsph-heating-mpl'))
            save(folder/'videoTitleRevision.json',dict(rendererSHA256=hashlib.sha256(Path('scripts/run_spheralBlobExample.py').read_bytes()).hexdigest(),
                action='Offline vispy rerender of stored states; correct native dudt/Warp spheral label translation and title. No physical integration.'))
        if not (path/'output.mp4').exists():
            from warpSPH.runner.media import encodeFrames
            print('Encode retained vispy frames:',folder.parent.name,flush=True)
            encodeFrames(str(path/'images'),str(path))
        video=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=nb_frames','-of','json',str(path/'output.mp4')]))
        frames=int(video['streams'][0]['nb_frames'])
        if frames!=len(source.times) or run['status'] not in ('completed','failed'):raise ValueError((folder,frames,len(source.times),run['status']))
        rows[str(folder)]=dict(id=rid,name=run['name'],status=run['status'],historyFrames=len(source.times),videoFrames=frames,historyMatches=True)
    expected=14 if (root/'partitionControls').exists() else 9
    if len(rows)!=expected:raise ValueError(('Unexpected recording count',len(rows),expected))
    save(report/'recordingVerification.json',rows);print('Verified',len(rows),'histories/videos')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('run','partitions','analyze','verify'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--report',type=Path,default=Path('reports/materialEOS/spheral/heating'))
    a=p.parse_args()
    if a.mode=='run':study(a.output.resolve())
    elif a.mode=='partitions':partition_controls(a.output.resolve())
    elif a.mode=='analyze':analyze(a.output.resolve(),a.report.resolve())
    else:verify(a.output.resolve(),a.report.resolve())
