"""Current native local solver on the original shared465-particle reference IC."""
import ast,json,time,statistics,argparse
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')
import torch
assert torch.cuda.is_available();torch.set_num_threads(4)
from warpSPH.runner import Case,run
from warpSPH.configurations import GravityType
from warpSPH.configurations.region import ParticleRegion,RegionType,BCType
from warpSPH.boundary import buildBoundaryProvider
from warpSPH.rigidBody.build import buildAnalyticRigidBody
from warpSPHBoundaries.scene import Body
from warpSPHBoundaries.scene.scene import BoxRep
from warpSPH.caseUtils.projection import initializeProjectionDensity
from warpSPH.schemes.incompressible import dfsph as native
from warpSPH.modules.deltaSPH import computeVelocityDiffusion
from warpSPHCore import OperationDirection
def fluid_viscosity(state,config,sc,adj,**kw):
 return computeVelocityDiffusion(state,config,sc,adj,operationMode=OperationDirection.FluidToFluid,wall=False,**kw)
native.computeVelocityDiffusion=fluid_viscosity
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
root=Path('.tmp/dfsph-native-init-2026-10-25/default-tol');saved=torch.load(root/'native-input.pt',weights_only=False);calibration=json.loads((root/'calibration.json').read_text());LO,HI=json.loads((root/'geometry.json').read_text())['bounds'];TGV=False;a=SimpleNamespace(case='analytic')
# Reuse only the archived config/build definitions, never its experimental step.
tree=ast.parse((root/'run.py').read_text());defs=[z for z in tree.body if isinstance(z,ast.FunctionDef) and z.name in ['configure','build']];exec(compile(ast.Module(body=defs,type_ignores=[]),str(root/'run.py'),'exec'))
conf=configure
counts=[];gaps=[];last=[None]
def configure(ctx):
 conf(ctx);ctx.schemeConfig.dfsph.minDivergenceDensityRatio=.9;ctx.schemeConfig.dfsph.divergence.maxIterations=100
 ctx.schemeConfig.diffusionParams.viscidNu=.01
def initialize(ctx,system):
 initializeProjectionDensity(ctx,system)
 st=system.state;torch.save({k:v.detach().cpu() for k,v in vars(st).items() if torch.is_tensor(v)},args.out/'initial.pt')
def post(ctx,system,step):
 torch.cuda.synchronize();now=time.perf_counter()
 if last[0] is not None and step>=30:gaps.append(1000*(now-last[0]))
 last[0]=now;u=ctx.scratch['lastStageUpdate'];counts.append([u.divergenceResult.iterations,u.densityResult.iterations])
 if step%50==0:print('BENCH',step,float(system.t),flush=True)
case=Case('native_IC_tank_performance','dfsph',build,configureScheme=configure,initialConditions=initialize,postStep=post,timestep=lambda ctx,sys:.001)
r=run(case,scheme='dfsph',device='cuda',nx=32,L=1.,kernel='CubicSpline',n_h=2.,dt=.001,maxDt=.001,nSteps=330,tLimit=None,integrationScheme='semiImplicitEuler',calibrateNormalization=False,adaptiveDt=False,pipelineOutputs=False,quiet=True,progress=True,plot=False,video=False,store=False,cudaGraph=False,velocityAlarmPlotInterval=1,stallProgress=.001)
s=dict(engine_ms=statistics.mean(z['stepTime_ms'] for z in r.trajectory if z['step']>=30),wall_ms=statistics.mean(gaps),run_wall_seconds=r.wallTime,run_wall_ms=1000*r.wallTime/r.nSteps,steps=r.nSteps,particles=len(r.state.state.positions),iterations_mean=[statistics.mean(z[i] for z in counts[30:]) for i in [0,1]],diverged=r.diverged,stop=r.stopReason,gate=.9,relaxation=.25,source=str(root.resolve()))
(args.out/'summary.json').write_text(json.dumps(s,indent=2)+'\n');print('SUMMARY',json.dumps(s),flush=True)
