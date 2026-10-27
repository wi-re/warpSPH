"""Short prepared-DFSPH profiling windows, video omitted for tiny A/B."""
import argparse,json,sys,time,shutil,statistics
from pathlib import Path
from dataclasses import replace
sys.path.insert(0,str(Path.cwd()/'scripts'))
from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')
import torch
assert torch.cuda.is_available()
torch.set_num_threads(4)
from warpSPH.cases import importAll
from warpSPH.runner.case import getCase
from warpSPH.runner import run
from warpSPHCore import callAudit
from prepared_fluidMatrix import prepared_case
importAll()
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--steps',type=int,default=300);p.add_argument('--warmup',type=int,default=30);p.add_argument('--profile',type=int,default=0);p.add_argument('--audit',action='store_true');p.add_argument('--diagnostics',action='store_true');p.add_argument('--state',type=Path);p.add_argument('--start-time',type=float,default=0.);p.add_argument('--reuse-wall-integrals',action='store_true');p.add_argument('--sections',action='store_true');p.add_argument('--tensor-transform',action='store_true');a=p.parse_args()
a.out.mkdir(parents=True,exist_ok=False)
row=json.loads((a.root/'manifest.json').read_text())['rows'][0]
shutil.copytree(a.root/'fixtures',a.out/'fixtures');(a.out/row['id']).mkdir()
case=prepared_case(getCase(row['case']),a.out,row)
conf,init,post,diag=case.configureScheme,case.initialConditions,case.postStep,case.diagnostics
last=[None];gaps=[];counts=[];prof=[None];hooks=[];diags=[];wall_cache={};wall_calls=[0,0];section_times={}
def configure(ctx):
 if conf:conf(ctx)
 for k,v in row['scheme_settings'].items():setattr(ctx.schemeConfig,k,v)
def initialize(ctx,system):
 if init:init(ctx,system)
 if a.state:
  data=torch.load(a.state,map_location=ctx.device,weights_only=True)
  for k,v in data.items():
   if hasattr(system.state,k):setattr(system.state,k,v.clone())
  system.adjacency=None
  system.t=a.start_time
 if a.reuse_wall_integrals:
  provider=ctx.schemeConfig.boundaryProvider
  original_aggregate=provider.aggregate
  def aggregate(ps,support,*args,**kwargs):
   wall_calls[0]+=1;key=(support,args,tuple(sorted(kwargs.items())))
   if key not in wall_cache:wall_cache[key]=original_aggregate(ps,support,*args,**kwargs);wall_calls[1]+=1
   return wall_cache[key]
  provider.aggregate=aggregate
 st=system.state
 torch.save({k:v.detach().cpu() for k,v in vars(st).items() if torch.is_tensor(v)},a.out/'initial.pt')
 print('INITIAL',len(st.positions),int((st.kinds==0).sum()),str(st.positions.dtype),str(st.positions.device),flush=True)
def diagnostic(ctx,system):
 t=time.perf_counter();res=diag(ctx,system) if diag else {};diags.append(time.perf_counter()-t);return res
def marker(ctx,system,step):
 torch.cuda.synchronize();now=time.perf_counter()
 if last[0] is not None and a.warmup<=step<a.warmup+a.steps:gaps.append(1000*(now-last[0]))
 last[0]=now
 tick=time.perf_counter()
 if post:post(ctx,system,step)
 hooks.append(time.perf_counter()-tick)
 wall_cache.clear()
 u=ctx.scratch.get('lastStageUpdate')
 if u is not None:counts.append([u.divergenceResult.iterations,u.densityResult.iterations])
 if step%50==0:print('BENCH',step,float(system.t),flush=True)
 if a.profile and step==a.warmup+a.steps-1:
  prof[0]=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=False,with_stack=False);prof[0].start()
 if a.profile and step==a.warmup+a.steps+a.profile-1:
  prof[0].stop();prof[0].export_chrome_trace(str(a.out/'trace.json'))
if a.sections:
 from contextlib import contextmanager
 from warpSPH.schemes.incompressible import dfsph
 @contextmanager
 def section(name):
  start=time.perf_counter()
  yield
  if a.warmup<=len(counts)<a.warmup+a.steps:section_times.setdefault(name,[]).append(1000*(time.perf_counter()-start))
 dfsph.record_function=section
if a.tensor_transform:
 from warpSPH.rigidBody import update as body_update
 def tensor_transform(body):
  center=body.centerOfMass;angle=torch.as_tensor(body.orientation,device=center.device,dtype=center.dtype).reshape(())
  c,s=torch.cos(angle),torch.sin(angle);z,o=torch.zeros_like(c),torch.ones_like(c)
  rotation=torch.stack([c,-s,z,s,c,z,z,z,o]).view(3,3)
  translation=torch.stack([o,z,center[0],z,o,center[1],z,z,o]).view(3,3)
  return translation@rotation@torch.eye(3,device=center.device,dtype=center.dtype)
 body_update.getTransformationMatrix=tensor_transform
case=replace(case,configureScheme=configure,initialConditions=initialize,postStep=marker,diagnostics=diagnostic if a.diagnostics else None)
if a.audit:callAudit.enable(torchOps=True,warmupSteps=a.warmup)
spec=dict(row['spec']);spec.update(nSteps=a.warmup+a.steps+a.profile,tLimit=None)
r=run(case,scheme='dfsph',params=row['params'],**spec,quiet=True,progress=True,plot=False,video=False,store=False,cudaGraph=False,velocityAlarmPlotInterval=1,stallProgress=.001)
if a.audit:callAudit.writeReport(str(a.out/'audit.txt'))
torch.save({k:v.detach().cpu() for k,v in vars(r.state.state).items() if torch.is_tensor(v)},a.out/'final.pt')
rows=r.trajectory;times=[z['stepTime_ms'] for z in rows if a.warmup<z['step']<=a.warmup+a.steps]
result=dict(row=row,checkpoint='529f431',root=str(a.root),state=str(a.state),gpu=torch.cuda.get_device_name(),wall_ms=statistics.mean(gaps),engine_ms=statistics.mean(times),run_wall_seconds=r.wallTime,steps=r.nSteps,run_wall_ms=1000*r.wallTime/r.nSteps,iterations_mean=[statistics.mean(z[i] for z in counts[a.warmup:a.warmup+a.steps]) for i in [0,1]],hook_ms=1000*statistics.mean(hooks[a.warmup:]),diagnostics_ms=1000*statistics.mean(diags[a.warmup:]) if diags else 0,diverged=r.diverged,stop=r.stopReason,reuse_wall_integrals=a.reuse_wall_integrals,tensor_transform=a.tensor_transform,wall_calls=wall_calls,sections_host_ms={k:statistics.mean(v) for k,v in section_times.items()})
(a.out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print('SUMMARY',json.dumps(result),flush=True)
if prof[0] is not None:
 (a.out/'profiler.txt').write_text(prof[0].key_averages().table(sort_by='self_cpu_time_total',row_limit=45))
