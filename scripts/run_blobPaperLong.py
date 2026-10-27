"""Persistent daemon-visible paper-resolution blob using the qualified match.

Fresh midpoint, paper density/geometric volumes, native weighted support target
at RK state updates and native final-work entropy fraction. No defaults change.
"""
import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from compare_spheralMaterials import FIELDS, save, host


def next_dt(candidate,t,end,kh):
    """Land exactly on quarter-KH checkpoints, including paper times 1/4/8."""
    if t>=end-1.e-10:return float(candidate)
    targets=np.arange(1,int(round(end/kh*4))+1)*(.25*kh)
    future=targets[targets>t+1.e-10]
    target=min(end,float(future[0])) if len(future) else end
    return min(float(candidate),target-t)


def prepare(root,tkh=8.,max_dt=.0005):
    if tkh<=0 or abs(tkh*4-round(tkh*4))>1.e-10 or max_dt<=0:
        raise ValueError('Positive quarter-KH duration and positive maxDt required')
    source=Path('results/materialEOS/blob_paper_2026-10-20/n512-8kh').resolve()
    root.mkdir(parents=True,exist_ok=False)
    native=json.loads((source/'settings.json').read_text());a=dict(np.load(source/'initializedState.npz'))
    assert len(a['masses'])==70181 and np.max(abs(a['H'][:,0,1]))<1.e-14
    assert np.max(abs(a['H'][:,0,0]-a['H'][:,1,1]))<1.e-14
    np.savez_compressed(root/'input.npz',**{k:a[k] for k in FIELDS})
    table=Path('results/materialEOS/idealH_2026-10-20/tables.npz');shutil.copyfile(table,root/'tables.npz')
    settings=dict(native,case='blob',gammaVariant='paper-n512-8KH-matched',nx=512,dim=2,kernel='B8',n_h=4.,
        gamma=[5/3,5/3],thermalGamma=5/3,requestedKH=tkh,end=tkh*native['tKH'],dt=.0001,maxDt=max_dt,
        steps=math.ceil(tkh*native['tKH']/max_dt),frameEvery=100,precision='float64',schedule='paper',
        supportPolicy='replacement',matchingEnergyPartition='entropy',matchingPolicy='reference',
        idealTable=str((root/'tables.npz').resolve()),inputPath=str((root/'input.npz').resolve()),
        comparisonPolicy='Qualified short-blob match: paper density/geometric volumes; native weighted target/state replacement; raw midpoint dudt, final-work native entropy fraction; averaged Q/current gradient, Cl2 Cq1 R4; fresh midpoint.',
        timestepPolicy='Warp acoustic CFL .25, maxDt cap, seed1e-4, clip to quarter-KH outputs. Not a replay of a new native adaptive grid.',
        sourceInitialization=str(source),visualization=dict(quantity='densities',normalization='linear',minimum=.75,maximum=10.,
            localPalette='RdBu_r',remotePalette='coolwarm',display='grid',resolution=1024,reference='Frontiere2017 Fig21',
            note='Remote coolwarm is the available blue-white-red approximation; local RdBu_r follows the paper palette.'))
    save(root/'settings.json',settings)
    paths=[Path(__file__),*[Path('scripts')/f for f in ('study_idealH.py','crkIdealH.py','study_spheralMatching.py','study_blobThermal.py','compare_spheralMaterials.py')]]
    paths+=list(Path('src/warpSPH/modules/crk').glob('*.py'))+list(Path('src/warpSPH/modules/compSPH').glob('*.py'))
    paths += [Path(f) for f in ('src/warpSPH/systems/compSPH.py','src/warpSPH/schemes/compressible/crkSPH.py','src/warpSPH/modules/timestep/compressible.py')]
    (root/'sources').mkdir();hashes={}
    for p in paths:
        h=hashlib.sha256(p.read_bytes()).hexdigest();hashes[str(p)]=h;shutil.copyfile(p,root/'sources'/(p.stem+'_'+h[:12]+p.suffix))
    save(root/'provenance.json',dict(sources=hashes,inputArraysSHA256={k:hashlib.sha256(a[k].tobytes()).hexdigest() for k in FIELDS},
        sourceInitializationSHA256=hashlib.sha256((source/'initializedState.npz').read_bytes()).hexdigest(),
        tableSHA256=hashlib.sha256(table.read_bytes()).hexdigest(),shortQualification='reports/materialEOS/spheralMatching/validation.json',
        scope='Paper geometry/resolution/duration with qualified matching policies. New adaptive capped timestep, same common gamma. Not a new high-resolution native parity result.'))
    print('Prepared paper blob:',len(a['masses']),'particles,',tkh,'KH; capped-step estimate',settings['steps'],flush=True)


def run_case(root):
    sys.path.insert(0,'/home/lu26029/dev/warpSPHRemote/python')
    from warpSPHBootstrap import bootstrap
    bootstrap(precision='float64')
    import torch
    from warpSPH.runner import run
    from warpSPH.cases.compressible.blob import blobCase
    from warpSPH.cases.common.plotting import Field,particlePlot
    from warpSPH.modules.eos import refreshEOS
    from warpSPH.modules.timestep.compressible import computeTimestep
    import warpSPH.modules.crk.energyControl as energy
    from warpsph_remote.publisher import SimulationPublisher,SimulationStopped
    from study_spheralMatching import reference_rate
    from study_idealH import install_support
    from study_blobThermal import Measures,summarize
    s=json.loads((root/'settings.json').read_text());a=dict(np.load(root/'input.npz'));start=time.perf_counter();history=[];latest={};last_save=[start]
    audit=install_support(s);energy.entropyRate=reference_rate;measure=Measures(s,a)
    publisher=SimulationPublisher('Paper blob n512 | 8 KH | matched CRKSPH',measurements=True)
    save(root/'registration.json',dict(remoteRunID=publisher.id,pid=os.getpid()))
    print('Remote run registered:',publisher.id,flush=True)
    checkpoints=root/'checkpoints';checkpoints.mkdir()
    base_build,base_post=blobCase.buildSystem,blobCase.postStep
    def build(c):
        system=base_build(c)
        for k,v in a.items():setattr(system.state,k,torch.as_tensor(v,device=c.device).clone())
        state=system.state;state.crkMassDensities=state.densities.clone();state.crkVolumes=None
        state.totalEnergies=state.masses*(state.internalEnergies+.5*state.velocities.square().sum(-1));system.adjacency=None
        refreshEOS(state,c.schemeConfig)
        assert c.schemeConfig.crkEvolution=='paper' and c.schemeConfig.crkEnergyPartition=='entropy'
        c.scratch['_titleProvenance']='Paper blob | 512 x 128 ambient | fresh midpoint | matched final work'
        return system
    saved=set()
    def checkpoint(c,system,step,tag):
        name='state_'+str(step).zfill(7);path=checkpoints/(name+'.npz');temp=checkpoints/(name+'.tmp.npz')
        np.savez_compressed(temp,**host(system.state));temp.replace(path)
        save(checkpoints/'latest.json',dict(path=str(path),step=step,time=float(system.t),dt=float(c.config.dt),tag=tag,
            explicitSupport=True,scope='Accepted physical state plus time/dt; use same script policies for continuation.'))
        if tag.startswith('KH'):shutil.copyfile(path,checkpoints/(tag+'.npz'))
    def post(c,system,step):
        base_post(c,system,step);kh=float(system.t)/s['tKH'];q=round(kh*4)
        milestone=q>0 and abs(kh-q/4)<1.e-8
        if milestone and q not in saved:
            checkpoint(c,system,step+1,'KH'+str(q/4).replace('.','p'));saved.add(q)
        elif (step+1)%500==0:checkpoint(c,system,step+1,'periodic')
    def timestep(c,system):
        candidate=computeTimestep(system,c.config,c.schemeConfig,dt=c.config.dt)
        return next_dt(float(candidate),float(system.t),s['end'],s['tKH'])
    def diagnostics(c,system):
        frame=host(system.state);t=float(system.t);row=measure(frame,t);history.append(row);latest.update(frame=frame,ctx=c,time=t)
        if not all(np.isfinite(v).all() for v in frame.values()) or min(row[k] for k in ('minDensity','minEnergy','minPressure'))<=0 or row['maxVelocity']>100:
            raise RuntimeError('Long blob finite-state/positivity/velocity watchdog')
        now=time.perf_counter()
        if now-last_save[0]>=5 or t==0:
            elapsed=now-start;step=len(history)-1
            save(root/'progress.json',dict(status='running',pid=os.getpid(),remoteRunID=publisher.id,steps=step,wallSeconds=elapsed,
                time=t,timeKH=t/s['tKH'],targetKH=s['requestedKH'],dt=float(c.config.dt),wallMillisecondsPerStep=1000*elapsed/max(step,1),
                etaSeconds=elapsed*(s['end']-t)/t if t>0 else None,diagnostics=row,exportPath=str(c.exportPath)))
            last_save[0]=now
        return {k:v for k,v in row.items() if k!='time'}
    setup,update=particlePlot([Field('densities','Density',colorMap='RdBu',colorMapKind='diverging',flip=True,vMin=.75,vMax=10.,midPoint=5.375)],figsize=(15,4),dpi=120)
    case=replace(blobCase,buildSystem=build,postStep=post,timestep=timestep,diagnostics=diagnostics,setupPlot=setup,updatePlot=update,
        extraFields=('internalEnergies','supports','pressures'))
    sim=dict(nx=512,tLimit=s['end'],nSteps=None,dt=s['dt'],adaptiveDt=True,minDt=1.e-8,maxDt=s['maxDt'],cflFactor=.25,
        precision='float64',execution='eager',kernel='B8',n_h=4.,integrationScheme='rungeKutta2')
    params=dict(viscositySwitch='NoneSwitch',adaptiveSupportScheme='Owen',massMatch=True,initialDensity='prescribed',markerSize=1.)
    solver=dict(crkEvolution='paper',crkDensityPolicy='massReplacement',crkEnergyPartition='entropy',crkEnergyPredictor='raw',
        crkViscosityGradient='current',crkViscosityParams=dict(oneSidedWork=False,viscosityKernelScale=4.),diffusionParams=dict(C_l=2.,C_q=1.))
    save(root/'launchConfiguration.json',dict(simulation=sim,params=params,solver=solver))
    try:
        result=run(publisher.bind_case(case),**sim,params=params,schemeOverrides=solver,plot=True,video=True,plotBackend='vispy',progress=True,
            plotInterval=100,store=True,storeMode='trajectory',exportInterval=.01,storeInterval=100,
            show=False,holdPlot=False,exportRoot=str(root/'warp'),stallProgress=1.e-3,stallDtSteps=100,velocityAlarmPlotInterval=1)
        publisher.recording(result.ctx);publisher.finish(result.ctx,result.state,result.nSteps)
        summary=summarize(root,'warpSPH',s,history,a,host(result.state.state),result.wallTime,result.nSteps,publisher.id,
            dict(exportPath=str(result.ctx.exportPath),runnerMeanStepMilliseconds=float(result.series('stepTime_ms').mean())))
        status='completed' if summary['passed'] else 'failed';publisher.close(status)
        save(root/'progress.json',dict(status=status,remoteRunID=publisher.id,steps=result.nSteps,time=history[-1]['time'],timeKH=history[-1]['tKH'],wallSeconds=result.wallTime))
    except BaseException as error:
        if 'frame' in latest:np.savez_compressed(root/'failedState.npz',**latest['frame'])
        save(root/'failure.json',dict(error=repr(error),time=latest.get('time'),wallSeconds=time.perf_counter()-start))
        if 'ctx' in latest:publisher.recording(latest['ctx'])
        publisher.close('stopped' if isinstance(error,SimulationStopped) else 'failed',str(error));raise
    finally:
        save(root/'diagnosticHistory.json',history);save(root/'supportAudit.json',audit)


def launch(root):
    if (root/'launch.json').exists():raise RuntimeError('Run already launched; inspect launch.json')
    env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MPLCONFIGDIR='/tmp/warpsph-blob-long-mpl',PYTHONUNBUFFERED='1')
    command=[sys.executable,'-u',str(Path(__file__).resolve()),'run','--output',str(root)]
    with (root/'run.log').open('w') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=Path(__file__).resolve().parents[1],start_new_session=True)
    record=dict(pid=child.pid,command=command,log=str(root/'run.log'),persistent=True);save(root/'launch.json',record);print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('prepare','run','launch'));p.add_argument('--output',required=True,type=Path)
    p.add_argument('--tkh',type=float,default=8.);p.add_argument('--max-dt',type=float,default=.0005);args=p.parse_args();root=args.output.resolve()
    if args.mode=='prepare':prepare(root,args.tkh,args.max_dt)
    elif args.mode=='launch':launch(root)
    else:run_case(root)
