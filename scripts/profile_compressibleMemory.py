#!/usr/bin/env python3
"""Short native 3D compressible memory window; optional separate allocator capture.

No solver changes. Default settings match the driven-flow qualification cohort.
Instrumented windows are not throughput benchmarks. Each invocation is isolated.
"""
from pathlib import Path
from dataclasses import replace
import argparse,contextlib,gc,gzip,hashlib,json,pickle,shutil,sys,time,traceback
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--runtime',type=Path,required=True)
p.add_argument('--configuration',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
p.add_argument('--steps',type=int,default=36)
p.add_argument('--discard',type=int,default=6)
p.add_argument('--capture',action='store_true')
p.add_argument('--device',default='cuda:0')
p.add_argument('--no-video',action='store_true',help='Only for tiny CPU smoke controls')
a=p.parse_args();a.runtime=a.runtime.resolve();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=False)
sys.path[:0]=[str(a.runtime/'src'),str(a.runtime/'scripts')]
from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')
import torch,numpy as np
from warpSPH.runner import RunConfiguration,resolvedConfiguration
from warpSPH.cases.compressible.drivenTurbulence import drivenTurbulenceCase as base
from warpSPH.cases.common.plotting import Field,particlePlot
from warpSPH.utils.memoryAudit import tensorStorageInventory
from run_compressibleMatrix import source_paths,source_inventory
cuda=a.device.startswith('cuda')
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def mem():
    if not cuda:return {}
    return dict(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),peakAllocated=torch.cuda.max_memory_allocated(),peakReserved=torch.cuda.max_memory_reserved())
def allocator(snapshot):
    states={};free=split=0
    for s in snapshot['segments']:
        if s['device']!=0:continue
        if s['active_size']==0:free+=s['total_size']
        for b in s['blocks']:
            states[b['state']]=states.get(b['state'],0)+b['size']
            if b['state']=='inactive' and s['active_size']>0:split+=b['size']
    return dict(states=states,fullyInactiveSegments=free,inactiveInsideActiveSegments=split,**mem())
def groups(system):
    st=system.state
    return dict(particleState={k:v for k,v in vars(st).items() if k not in ('ap_ij','av_ij','f_ij','_meshlessGeometry','_meshlessNeighbors')},
        stagePairArrays={k:getattr(st,k,None) for k in ('ap_ij','av_ij','f_ij')},
        adjacency=getattr(system,'adjacency',None),compatiblePairWork=getattr(system,'pairWork',None),
        meshlessGeometry=getattr(st,'_meshlessGeometry',None),meshlessNeighbors=getattr(st,'_meshlessNeighbors',None),
        systemOther={k:v for k,v in vars(system).items() if k not in ('state','adjacency','pairWork')})
def inventory(roots):return tensorStorageInventory(roots,deviceType='cuda' if cuda else 'cpu')
def hashes(system):
    return {k:hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest() for k,v in vars(system.state).items() if isinstance(v,torch.Tensor) and k not in ('ap_ij','av_ij','f_ij')}
configuration=json.loads(a.configuration.read_text());configuration['simulation'].update(device=a.device,nSteps=a.steps,tLimit=4.,store=False,plot=not a.no_video,video=not a.no_video,plotInterval=12,progress=True,exportRoot=str(a.out/'exports'))
for k in ('storeInterval','storeMode'):configuration['simulation'].pop(k,None)
write(a.out/'input.json',configuration);shutil.copy2(__file__,a.out/'driver.py')
roots=source_paths();before={k:v[1] for k,v in source_inventory(roots).items()};write(a.out/'sourceHashes.json',before)
write(a.out/'provenance.json',dict(runtime=str(a.runtime),driverSHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),device=torch.cuda.get_device_name() if cuda else 'cpu',torch=torch.__version__,capture=a.capture,warmupSteps=3))
setup,update=particlePlot([Field('densities','Density: z=0',gridResolution=96,sliceAxis=2,slicePosition=0.),Field('velocities','Speed: z=0',mapping='L2Norm',colorMap='inferno',gridResolution=96,sliceAxis=2,slicePosition=0.)],figsize=(10,5))
case=replace(base,setupPlot=setup,updatePlot=update)
stats=dict(count=0,bad=0,minEnergy=1e99,minPressure=1e99,minDensity=1e99);history=[];stamps=[];latest={};points=[];regions=[];current=dict(step=0,stage=None);peakOwnership={};observedPeak=[0]
class Section:
    def __init__(self,name):self.name=name
    def __enter__(self):
        self.frame=sys._getframe(1);self.before=mem();return self
    def __exit__(self,*error):
        state=mem();row=dict(name=self.name,**current,before=self.before,after=state)
        regions.append(row)
        if state.get('peakAllocated',0)>observedPeak[0]:
            observedPeak[0]=state['peakAllocated'];roots={};f=self.frame;depth=0
            while f is not None:
                if f.f_code.co_filename.find('/warpSPH')>=0:
                    roots[f'{depth}:{f.f_code.co_name}']={k:v for k,v in f.f_locals.items() if k not in ('ctx','config','schemeConfig','case','spec') and not k.startswith('_')}
                f=f.f_back;depth+=1
            peakOwnership.clear();peakOwnership.update(region=row,inventory=inventory(roots))
        self.frame=None;return False
patches=[]
def install_sections():
    for m in list(sys.modules.values()):
        if m is not None and getattr(m,'__name__','').startswith(('warpSPH','warpSPHCore','warpSPHIntegrators')) and hasattr(m,'record_function'):
            if not any(x[0] is m for x in patches):patches.append((m,m.record_function));m.record_function=Section

def build(ctx):
    system=base.buildSystem(ctx);latest['ctx']=ctx;write(a.out/'resolved.json',resolvedConfiguration(ctx));native=ctx.stepFunction
    def observed(s,dt,stageIndex=None,**kw):
        current.update(step=ctx.scratch['completedSteps'],stage=stageIndex)
        if stageIndex is not None:kw['stageIndex']=stageIndex
        if a.capture:install_sections()
        out=native(s,dt,**kw);state=s.state;stage=getattr(s,'meshlessContext',None)
        u,P,rho=(stage.u,stage.P,stage.rho) if stage is not None else (state.internalEnergies,state.pressures,state.densities)
        rates=[out[0].dQdt] if stage is not None else [out[0].dvdt,out[0].dudt,out[0].dEdt]
        probes=torch.stack([u.min(),P.min(),rho.min()]+[torch.isfinite(v).all().to(u) for v in [state.positions,u,P,rho,*rates,(state.Q if stage is not None else state.velocities)]]).cpu().tolist()
        stats['count']+=1
        for k,x in zip(('minEnergy','minPressure','minDensity'),probes[:3]):stats[k]=min(stats[k],x)
        stats['bad']+=int(min(probes[:3])<=0 or not all(probes[3:]));return out
    ctx.bundle=replace(ctx.bundle,stepFunction=observed);return system

def diagnostics(ctx,system):
    row=base.diagnostics(ctx,system);row['t']=float(system.t);step=ctx.scratch['completedSteps'];history.append(row);stamps.append(time.perf_counter())
    if step==0:
        write(a.out/'initial-ownership.json',inventory(groups(system)));write(a.out/'initial-hashes.json',hashes(system))
    if step==6:write(a.out/'step6-hashes.json',hashes(system))
    if a.capture:
        points.append(dict(step=step,t=row['t'],**(allocator(torch.cuda.memory._snapshot()) if cuda else {})))
        if step==a.steps:
            write(a.out/'accepted-ownership.json',inventory(dict(**groups(system),stepResult=sys._getframe(1).f_locals.get('stepResult'))))
    write(a.out/'progress.json',dict(row,steps=step,stages=stats));print('ADVANCE',step,row['t'],'badRHS',stats['bad'],flush=True)
    return row

try:
    warm=json.loads(json.dumps(configuration));warm['simulation'].update(nSteps=3,plot=False,video=False,progress=True)
    print('WARMUP',configuration['solver']['scheme'],configuration['simulation']['nx'],flush=True)
    w=RunConfiguration.fromDict(warm).run(case);assert not w.diverged and not w.stopReason
    del w;gc.collect()
    if cuda:torch.cuda.synchronize();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    if a.capture and cuda:torch.cuda.memory._record_memory_history(enabled='all',context='all',stacks='python',max_entries=400000,clear_history=True)
    started=time.perf_counter();result=RunConfiguration.fromDict(configuration).run(replace(case,buildSystem=build,diagnostics=diagnostics))
    if cuda:torch.cuda.synchronize()
    elapsed=time.perf_counter()-started;high=mem()
    if a.capture and cuda:
        snapshot=torch.cuda.memory._snapshot()
        with (a.out/'allocator.pickle').open('wb') as f:pickle.dump(snapshot,f)
        with gzip.open(a.out/'allocator.pickle.gz','wb') as f:pickle.dump(snapshot,f)
        points.append(dict(phase='after-run',**allocator(snapshot)))
    adj=result.state.adjacency
    ticket=getattr(result.state.state,'_meshlessGeometry',None)
    geometry=getattr(ticket,'geometry',None)
    adjacencies={'accepted':adj,'geometry':getattr(geometry,'adjacency',None)}
    write(a.out/'neighbor-counts.json',{k:dict(edges=int(v.numEdges),capacity=int(v.paddedJ.numel()),indexDtype=str(v.paddedJ.dtype),ownership=inventory({'adjacency':v})) for k,v in adjacencies.items() if v is not None})
    write(a.out/'final-ownership.json',inventory(groups(result.state)));write(a.out/'final-hashes.json',hashes(result.state));write(a.out/'history.json',history)
    write(a.out/'regions.json',regions);write(a.out/'peak-ownership.json',peakOwnership);write(a.out/'allocation-points.json',points)
    torch.save(result.state.sourceTerms.acceleration.state_dict(),a.out/'forcing.pt')
    after={k:v[1] for k,v in source_inventory(roots).items()};positive=all(min(r['minEnergy'],r['minPressure'],r['minDensity'])>0 for r in history)
    budget=max(abs(r['relativeEnergyBudgetResidual']) for r in history);good=not result.diverged and not result.stopReason and result.nSteps==a.steps and stats['bad']==0 and stats['count']==2*a.steps and before==after and positive and budget<2e-6
    intervals=np.diff(stamps);native=np.array([x['stepTime_ms'] for x in result.trajectory[1:]])
    record=dict(status='completed' if good else 'failed',scheme=configuration['solver']['scheme'],nx=configuration['simulation']['nx'],particles=len(result.state.state.positions),nSteps=result.nSteps,actualTime=float(result.state.t),instrumented=a.capture,stageChecks=stats,extraRHS=stats['count']-2*result.nSteps,sourcesStable=before==after,acceptedPositiveFinite=positive,totalWallSeconds=elapsed,loopWallSeconds=float(stamps[-1]-stamps[0]),loopMsPerStep=float(1000*(stamps[-1]-stamps[0])/result.nSteps),steadyWallMsPerStep=float(1000*intervals[a.discard:].mean()) if len(intervals)>a.discard else None,nativeMeanMs=float(native[a.discard:].mean()) if len(native)>a.discard else None,maxAbsoluteEnergyBudgetResidual=budget,memory=high,exportPath=str(result.exportPath),stopReason=result.stopReason)
    write(a.out/'result.json',record);print('RESULT',json.dumps(record),flush=True)
    if not good:raise RuntimeError('native memory-window guards failed')
except Exception:
    error=traceback.format_exc()
    if cuda:
        try:
            with gzip.open(a.out/'failure-allocator.pickle.gz','wb') as f:pickle.dump(torch.cuda.memory._snapshot(),f)
        except Exception:pass
    write(a.out/'failure.json',dict(error=error,stageChecks=stats,memory=mem()));raise
finally:
    for m,original in patches:m.record_function=original
    if a.capture and cuda:torch.cuda.memory._record_memory_history(enabled=None)
