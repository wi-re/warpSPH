#!/usr/bin/env python3
"""Short CRK stage/storage ownership probe. Instrumented timings are not benchmarks.

Read-only Python hooks and allocator history preserve native fresh midpoint RHS,
X6 accumulation and neighbour policies. Exact physical-support filtering happens
only after the trajectory and never replaces its buffered neighbour list.
"""
from pathlib import Path
from dataclasses import replace
import argparse,contextlib,gzip,hashlib,inspect,json,os,pickle,shutil,sys,time
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--runtime',type=Path,required=True);p.add_argument('--nx',type=int,required=True)
p.add_argument('--out',type=Path,required=True);p.add_argument('--resume',type=Path)
p.add_argument('--configuration',type=Path)
p.add_argument('--accounting-helper',type=Path)
p.add_argument('--crk-pair-storage',choices=('expanded','recompute'))
p.add_argument('--steps',type=int,default=4);a=p.parse_args()
a.runtime=a.runtime.resolve();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=False)
sys.path[:0]=[str(a.runtime/'src'),str(a.runtime/'scripts')]
from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')
import torch,numpy as np
from warpSPH.runner import RunConfiguration,resolvedConfiguration
from warpSPH.cases.compressible.drivenTurbulence import drivenTurbulenceCase as base
from warpSPH.cases.common.plotting import Field,particlePlot
from run_compressibleMatrix import capture_sources,source_inventory,source_paths
# The accounting helper is instrumentation, not part of the frozen physical runtime.
import importlib.util
helper=(a.accounting_helper or Path(__file__).resolve().parents[1]/'src/warpSPH/utils/memoryAudit.py').resolve()
spec=importlib.util.spec_from_file_location('memoryAudit',helper);audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)

def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
def memory():
    torch.cuda.synchronize()
    free,total=torch.cuda.mem_get_info()
    return dict(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),deviceUsed=total-free,deviceTotal=total)

def storage_roots(frame):
    local=frame.f_locals;state=local.get('currentState')
    roots={}
    if state is not None and hasattr(state,'ap_ij'):
        roots['particleState']={k:v for k,v in vars(state).items() if k not in ('ap_ij','av_ij','f_ij')}
        roots['stagePairArrays']={k:getattr(state,k,None) for k in ('ap_ij','av_ij','f_ij')}
    system=local.get('currentSystem')
    if system is not None:
        roots['stageAdjacency']=local.get('adjacency',system.adjacency)
        roots['compatiblePairWork']=system.pairWork
        roots['grids']=[system.supportGrid,system.listGrid]
    roots['stageLocals']={k:v for k,v in local.items() if k not in ('ctx','config','schemeConfig','currentSystem','currentState') and not k.startswith('_')}
    outer=frame.f_back
    while outer is not None:
        name=outer.f_code.co_name
        if name=='_RungeKuttaBOnce':
            roots['integratorInitial']=outer.f_locals.get('initialState')
            roots['integratorRetained']={k:outer.f_locals[k] for k in ('ks','rs','k0','r0','stages','lastStageState') if k in outer.f_locals}
        if outer.f_code.co_filename.endswith('/runner/runner.py') and 'stepResult' in outer.f_locals:
            roots['runnerPreviousResult']=outer.f_locals.get('stepResult')
            roots['runnerAccepted']=outer.f_locals.get('runningState')
            roots['runnerResult']=outer.f_locals.get('result')
        outer=outer.f_back
    return roots

rows=[];current={'stage':None,'step':0};enabled=[False]
class Section:
    def __init__(self,name):self.name=name
    def __enter__(self):
        self.frame=sys._getframe(1);self.before=memory();torch.cuda.reset_peak_memory_stats()
        self.cursor=len(torch.cuda.memory._snapshot()['device_traces'][0])
        self.start=time.perf_counter();return self
    def __exit__(self,*error):
        after=memory();peak=torch.cuda.max_memory_allocated();elapsed=time.perf_counter()-self.start
        row=dict(name=self.name,step=current['step'],stage=current['stage'],before=self.before,after=after,peakAllocated=peak,
                 transientAboveLargerBoundary=max(0,peak-max(self.before['allocated'],after['allocated'])),seconds=elapsed,
                 traceBegin=self.cursor,traceEnd=len(torch.cuda.memory._snapshot()['device_traces'][0]),
                 ownership=audit.tensorStorageInventory(storage_roots(self.frame),deviceType='cuda'))
        rows.append(row);self.frame=None
        print('MEMORY',row['step'],row['stage'],self.name,'allocated',after['allocated'],'peak',peak,flush=True)
        return False

def section(name):
    if enabled[0] and (name.startswith('[warpSPH] - [crkSPH - ') or name=='[warpSPH] - [compSPH finalize - 01] compatible energy'):
        return Section(name)
    return contextlib.nullcontext()

configuration=json.loads((a.configuration or a.runtime.parents[1]/'runs'/f'weak{a.nx}cubed-CRKSPH-s1234/input.json').read_text())
if a.crk_pair_storage:configuration['solver']['crkPairStorage']=a.crk_pair_storage
configuration['simulation'].update(nSteps=a.steps,tLimit=4.,store=False,plot=True,video=True,plotInterval=1,progress=True,exportRoot=str(a.out/'exports'))
for k in ('storeInterval','storeMode'):configuration['simulation'].pop(k,None)
if a.resume:configuration['simulation']['resumeFrom']=str(a.resume.resolve())
setup,update=particlePlot([Field('densities','Density: z=0',gridResolution=96,sliceAxis=2,slicePosition=0.),Field('velocities','Speed: z=0',mapping='L2Norm',colorMap='inferno',gridResolution=96,sliceAxis=2,slicePosition=0.)],figsize=(10,5))
case=replace(base,setupPlot=setup,updatePlot=update)
shutil.copy2(__file__,a.out/'driver.py');shutil.copy2(helper,a.out/'memoryAudit.py');write(a.out/'input.json',configuration)
roots=source_paths();before=capture_sources(a.out,roots);write(a.out/'instrumentation-hashes.json',{str(helper):hashlib.sha256(helper.read_bytes()).hexdigest(),str(Path(__file__)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
# Import all functions before replacing their bound record_function globals.
from warpSPH.modules import crk
from warpSPH.modules.compSPH import pairWork
from warpSPH.systems.compSPH import CompSPHSystem
stats=dict(rhs=0,bad=0,minU=float('inf'),minP=float('inf'));history=[];accepted=[];latest={}
original_finalize=CompSPHSystem.finalize

def observed_finalize(self,initialState,dt,returnValues,updateValues,weights,config,schemeConfig,*args,**kw):
    value=original_finalize(self,initialState,dt,returnValues,updateValues,weights,config,schemeConfig,*args,**kw)
    if enabled[0]:
        accepted.append(dict(phase='finalize',step=current['step'],memory=memory(),ownership=audit.tensorStorageInventory(dict(acceptedState=self.state,acceptedAdjacency=self.adjacency,currentPairWork=self.pairWork,initialSystem=initialState,stageAux=returnValues,stageUpdates=updateValues),deviceType='cuda')))
    return value
CompSPHSystem.finalize=observed_finalize

def build(ctx):
    system=base.buildSystem(ctx)
    latest.update(ctx=ctx,initialStateInventory=audit.tensorStorageInventory(dict(state=system.state),deviceType='cuda'))
    native=ctx.stepFunction
    def observed(s,dt,stageIndex=None,**kw):
        current['stage']=stageIndex
        current['step']=ctx.scratch['completedSteps']
        enabled[0]=True
        for m in list(sys.modules.values()):
            if m is not None and getattr(m,'__name__','').startswith(('warpSPH','warpSPHCore')) and hasattr(m,'record_function'):
                setattr(m,'record_function',section)
        out=native(s,dt,stageIndex=stageIndex,**kw)
        u,P=s.state.internalEnergies,s.state.pressures
        values=torch.stack([u.min(),P.min()]+[torch.isfinite(x).all().to(u) for x in (s.state.positions,u,P,s.state.densities,out[0].dvdt,out[0].dudt)]).cpu().tolist()
        stats['rhs']+=1;stats['minU']=min(stats['minU'],values[0]);stats['minP']=min(stats['minP'],values[1]);stats['bad']+=int(min(values[:2])<=0 or not all(values[2:]))
        return out
    ctx.bundle=replace(ctx.bundle,stepFunction=observed);write(a.out/'resolved.json',resolvedConfiguration(ctx));return system

def diagnostics(ctx,system):
    row=base.diagnostics(ctx,system);row['t']=float(system.t);history.append(row)
    frame=sys._getframe(1)
    stepResult=frame.f_locals.get('stepResult')
    accepted.append(dict(phase='accepted',step=ctx.scratch['completedSteps'],memory=memory(),ownership=audit.tensorStorageInventory(dict(state=system.state,adjacency=system.adjacency,pairWork=system.pairWork,stepResult=stepResult,grids=[system.supportGrid,system.listGrid]),deviceType='cuda')))
    print('ACCEPTED',ctx.scratch['completedSteps'],row['t'],row['relativeEnergyBudgetResidual'],flush=True)
    return row

torch.cuda.memory._record_memory_history(enabled='all',context='all',stacks='python',max_entries=100000,clear_history=True)
latest['historyInitialMemory']=memory();start=time.perf_counter()
try:
    result=RunConfiguration.fromDict(configuration).run(replace(case,buildSystem=build,diagnostics=diagnostics));elapsed=time.perf_counter()-start
    enabled[0]=False
    snapshot=torch.cuda.memory._snapshot()
    with gzip.open(a.out/'allocator-history.pickle.gz','wb') as f:pickle.dump(snapshot,f)
    write(a.out/'regions.json',rows);write(a.out/'accepted.json',accepted);write(a.out/'history.json',history)
    state=result.state.state
    inventory=audit.tensorStorageInventory(dict(state=state,adjacency=result.state.adjacency,pairWork=result.state.pairWork,resultStages=getattr(result,'stages',None)),deviceType='cuda')
    write(a.out/'final-ownership.json',inventory)
    from warpSPHCore import filterVerletList,SupportScheme
    from warpSPHCore.dataTypes.adjacency_t import CapacityPolicy
    torch.cuda.reset_peak_memory_stats();gstart=time.perf_counter()
    physical=filterVerletList(state,result.state.domain,result.state.adjacency,supportMode=SupportScheme.SuperSymmetric,capacity=None)
    torch.cuda.synchronize();geometrySeconds=time.perf_counter()-gstart
    comparison=audit.tensorStorageInventory(dict(state=state,buffered=result.state.adjacency,physical=physical),deviceType='cuda')
    write(a.out/'neighbor-comparison.json',dict(particles=len(state.positions),verletScale=latest['ctx'].config.verletScale,
        bufferedEdges=result.state.adjacency.numEdges,bufferedCapacity=result.state.adjacency.paddedI.numel(),
        physicalEdges=physical.numEdges,physicalCapacity=physical.paddedI.numel(),ownership=comparison,
        geometrySeconds=geometrySeconds,geometryPeakAllocated=torch.cuda.max_memory_allocated(),policy='Current accepted positions/supports, SuperSymmetric physical support; filter only, not used for evolution.'))
    after={k:v[1] for k,v in source_inventory(roots).items()}
    good=not result.diverged and not result.stopReason and stats['bad']==0 and stats['rhs']==2*a.steps and before==after and all(min(r['minEnergy'],r['minPressure'],r['minDensity'])>0 for r in history)
    summary=dict(status='passed' if good else 'failed',nx=a.nx,steps=result.nSteps,actualTime=float(result.state.t),wallSeconds=elapsed,msPerStep=1000*elapsed/result.nSteps,instrumented=True,stats=stats,sourcesStable=before==after,maxBudget=max(abs(r['relativeEnergyBudgetResidual']) for r in history),historyInitialMemory=latest['historyInitialMemory'],initialStateInventory=latest['initialStateInventory'],regions=len(rows),acceptedFrames=len(accepted),stopReason=result.stopReason)
    write(a.out/'result.json',summary);print('RESULT',json.dumps({k:v for k,v in summary.items() if k!='initialStateInventory'}),flush=True)
    if not good:raise RuntimeError('memory probe physical/source guard failed')
except Exception:
    import traceback
    write(a.out/'failure.json',dict(error=traceback.format_exc(),stats=stats));raise
finally:torch.cuda.memory._record_memory_history(enabled=None)
