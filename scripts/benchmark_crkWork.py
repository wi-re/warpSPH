#!/usr/bin/env python3
"""Short source-paired CRK storage benchmark with native RHS/source guards.

Independent 3-step warmup omits video; measured finite windows record slice
videos. Native runner CUDA-event step timings and accepted-state wall intervals
are retained without allocator/section profiling. Use serial processes.
"""
from pathlib import Path
from dataclasses import replace
import argparse,gc,gzip,hashlib,json,pickle,shutil,sys,time,traceback
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--runtime',type=Path,required=True);p.add_argument('--configuration',type=Path,required=True)
p.add_argument('--mode',choices=('legacy','expanded','recompute'),required=True)
p.add_argument('--out',type=Path,required=True);p.add_argument('--resume',type=Path)
p.add_argument('--steps',type=int,default=36);p.add_argument('--repeats',type=int,default=2);p.add_argument('--discard',type=int,default=6)
p.add_argument('--memory-history',action='store_true',help='Separate allocation-capture run; timings are instrumented, not throughput')
a=p.parse_args();a.runtime=a.runtime.resolve();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=False)
sys.path[:0]=[str(a.runtime/'src'),str(a.runtime/'scripts')]
from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')
import numpy as np,torch
from warpSPH.runner import RunConfiguration,resolvedConfiguration
from warpSPH.cases.compressible.drivenTurbulence import drivenTurbulenceCase as base
from warpSPH.cases.common.plotting import Field,particlePlot
from run_compressibleMatrix import source_inventory,source_paths

def write(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
configuration=json.loads(a.configuration.read_text())
configuration['simulation'].update(nSteps=a.steps,tLimit=4.,store=False,plot=True,video=True,
    plotInterval=12,progress=True,exportRoot=str(a.out/'exports'),velocityAlarmPlotInterval=1)
for key in ('storeInterval','storeMode'):configuration['simulation'].pop(key,None)
if a.mode!='legacy':configuration['solver']['crkPairStorage']=a.mode
if a.resume:configuration['simulation']['resumeFrom']=str(a.resume.resolve())
write(a.out/'input.json',configuration);shutil.copy2(__file__,a.out/'driver.py')
roots=source_paths();before={k:v[1] for k,v in source_inventory(roots).items()};write(a.out/'sourceHashes.json',before)
write(a.out/'provenance.json',dict(runtime=str(a.runtime),driverSHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    resume=str(a.resume) if a.resume else None,resumeSHA256=hashlib.sha256(a.resume.read_bytes()).hexdigest() if a.resume else None,
    mode=a.mode,device=torch.cuda.get_device_name(),torch=torch.__version__,warmupSteps=3,discardSteps=a.discard))
setup,plotUpdate=particlePlot([Field('densities','Density: z=0',gridResolution=96,sliceAxis=2,slicePosition=0.),
    Field('velocities','Speed: z=0',mapping='L2Norm',colorMap='inferno',gridResolution=96,sliceAxis=2,slicePosition=0.)],figsize=(10,5))
case=replace(base,setupPlot=setup,updatePlot=plotUpdate)
warm=json.loads(json.dumps(configuration));warm['simulation'].update(nSteps=3,plot=False,video=False,progress=True)
print('WARMUP',a.mode,flush=True)
w=RunConfiguration.fromDict(warm).run(case);assert not w.diverged and not w.stopReason
del w;gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache()
def allocator_state(snapshot):
    states={};free_segments=0;split_inactive=0;largest_inactive=0
    for segment in snapshot['segments']:
        if segment['device']!=0:continue
        if segment['active_size']==0:free_segments+=segment['total_size']
        for block in segment['blocks']:
            states[block['state']]=states.get(block['state'],0)+block['size']
            if block['state']=='inactive':
                largest_inactive=max(largest_inactive,block['size'])
                if segment['active_size']>0:split_inactive+=block['size']
    return dict(states=states,fullyInactiveSegments=free_segments,inactiveInsideActiveSegments=split_inactive,largestInactiveBlock=largest_inactive,
        allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved())

results=[]
try:
 for rep in range(a.repeats):
    folder=a.out/f'repeat{rep}';folder.mkdir()
    cfg=json.loads(json.dumps(configuration));cfg['simulation']['exportRoot']=str(folder/'exports')
    stats=dict(count=0,bad=0,minEnergy=1e99,minPressure=1e99);history=[];stamps=[];last={};allocationPoints=[]
    if a.memory_history:torch.cuda.memory._record_memory_history(enabled='all',context='all',stacks='python',max_entries=400000,clear_history=True)
    def arrays(system):
        return {k:v.detach().cpu().numpy() for k,v in vars(system.state).items() if isinstance(v,torch.Tensor)}
    def build(ctx):
        system=base.buildSystem(ctx);last['ctx']=ctx
        write(folder/'resolved.json',resolvedConfiguration(ctx))
        native=ctx.stepFunction
        def observed(s,dt,stageIndex=None,**kw):
            out=native(s,dt,stageIndex=stageIndex,**kw)
            st=s.state;u,P,rho=st.internalEnergies,st.pressures,st.densities
            probes=torch.stack([u.min(),P.min(),rho.min()]+[torch.isfinite(x).all().to(u) for x in (st.positions,st.velocities,u,P,rho,out[0].dvdt,out[0].dudt,out[0].dEdt)]).cpu().tolist()
            stats['count']+=1;stats['minEnergy']=min(stats['minEnergy'],probes[0]);stats['minPressure']=min(stats['minPressure'],probes[1])
            stats['bad']+=int(min(probes[:3])<=0 or not all(probes[3:]));return out
        ctx.bundle=replace(ctx.bundle,stepFunction=observed)
        return system
    def diag(ctx,system):
        row=base.diagnostics(ctx,system);row['t']=float(system.t)
        if not history:np.savez_compressed(folder/'initial.npz',t=np.array(float(system.t)),**arrays(system))
        history.append(row);stamps.append(time.perf_counter())
        if a.memory_history and (len(history)==1 or (len(history)-1)%6==0):
            allocationPoints.append(dict(step=len(history)-1,t=row['t'],**allocator_state(torch.cuda.memory._snapshot())))
        print('ADVANCE',rep,len(history)-1,row['t'],'badRHS',stats['bad'],flush=True)
        write(folder/'progress.json',dict(row,steps=len(history)-1,stages=stats))
        return row
    torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
    result=RunConfiguration.fromDict(cfg).run(replace(case,buildSystem=build,diagnostics=diag));torch.cuda.synchronize()
    total=time.perf_counter()-started
    peakAllocated=torch.cuda.max_memory_allocated();peakReserved=torch.cuda.max_memory_reserved()
    if a.memory_history:
        snapshot=torch.cuda.memory._snapshot()
        with gzip.open(folder/'allocator-history.pickle.gz','wb') as f:pickle.dump(snapshot,f)
        with (folder/'allocator.pickle').open('wb') as f:pickle.dump(snapshot,f)
        allocationPoints.append(dict(phase='after-run',**allocator_state(snapshot)))
        torch.cuda.empty_cache()
        allocationPoints.append(dict(phase='after-empty-cache',**allocator_state(torch.cuda.memory._snapshot())))
        write(folder/'allocation-points.json',allocationPoints)
        torch.cuda.memory._record_memory_history(enabled=None)
    loop=stamps[-1]-stamps[0];wall=np.diff(stamps);native=np.array([x['stepTime_ms'] for x in result.trajectory[1:]])
    np.savez_compressed(folder/'final.npz',t=np.array(float(result.state.t)),**arrays(result.state))
    pw=result.state.pairWork
    if pw.work is not None:
        live=int(pw.adjacency.numEdges);ids=torch.linspace(0,live-1,4096,device=pw.work.device).long().unique()
        np.savez_compressed(folder/'work-sample.npz',indices=ids.cpu().numpy(),work=pw.work[ids].cpu().numpy(),live=np.array(live),capacity=np.array(pw.work.shape[0]))
    torch.save(result.state.sourceTerms.acceleration.state_dict(),folder/'forcing.pt')
    positive=all(min(row['minDensity'],row['minPressure'],row['minEnergy'])>0 for row in history)
    after={k:v[1] for k,v in source_inventory(roots).items()}
    good=not result.diverged and not result.stopReason and result.nSteps==a.steps and stats['bad']==0 and positive and before==after and stats['count']==2*a.steps and max(abs(row['relativeEnergyBudgetResidual']) for row in history)<2e-6
    record=dict(status='completed' if good else 'failed',nSteps=result.nSteps,actualTime=float(result.state.t),instrumented=a.memory_history,initialTime=history[0]['t'],mode=a.mode,
        stageChecks=stats,extraRHS=stats['count']-2*result.nSteps,acceptedPositiveFinite=positive,sourcesStable=before==after,
        totalWallSeconds=total,loopWallSeconds=loop,loopMsPerStep=1000*loop/result.nSteps,
        steadyWallMsPerStep=float(1000*wall[a.discard:].mean()),steadyWallMedianMs=float(1000*np.median(wall[a.discard:])),
        nativeMeanMs=float(native[a.discard:].mean()),nativeMedianMs=float(np.median(native[a.discard:])),
        nativeStepMs=native.tolist(),wallIntervals=wall.tolist(),discardSteps=a.discard,
        peakTorchAllocatedBytes=peakAllocated,peakTorchReservedBytes=peakReserved,
        maxAbsoluteEnergyBudgetResidual=max(abs(row['relativeEnergyBudgetResidual']) for row in history),
        exportPath=str(result.exportPath),stopReason=result.stopReason)
    write(folder/'history.json',history);write(folder/'result.json',record);results.append(record)
    print('RESULT',json.dumps({k:v for k,v in record.items() if k not in ('nativeStepMs','wallIntervals')}),flush=True)
    del result,pw;gc.collect();torch.cuda.empty_cache()
    if not good:raise RuntimeError('short native comparison guards failed')
 write(a.out/'result.json',dict(status='completed',runs=results,nSteps=a.steps*a.repeats,sourcesStable=True,mode=a.mode))
except Exception:
 write(a.out/'failure.json',dict(error=traceback.format_exc(),completedRuns=results));raise
