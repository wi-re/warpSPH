"""Short CPU reference timing windows; no observer/video in timed loop."""
import argparse,json,sys,time,statistics,os,hashlib,subprocess
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--warmup',type=int,default=30);p.add_argument('--steps',type=int,default=300);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
root=Path('/home/lu26029/dev/SPlisHSPlasH');sys.path.insert(0,str(root/'build/lib'));import pysplishsplash as sph
source=Path('.tmp/dfsph-native-init-2026-10-25/default-tol/native/scene.json');scene=json.loads(source.read_text());scene['Configuration']['DFSPH'].update(maxIterationsV=100)
scene['RigidBodies'][0]['geometryFile']=str(root/'data/models/UnitBox.obj')
path=a.out/'scene.json';path.write_text(json.dumps(scene,indent=2)+'\n')
base=sph.Exec.SimulatorBase();base.init(sceneFile=str(path.resolve()),useGui=False,initialPause=False,useCache=False,outputDir=str((a.out/'native_output').resolve()),stopAt=.5);base.initSimulation();sim=sph.Simulation.getCurrent();fluid=sim.getFluidModel(0);ts=sim.getTimeStep();tm=sph.TimeManager.getCurrent()
tm.setTime(1.);base.runSimulation();assert tm.getTime()==1.;tm.setTime(0.);base.setValueFloat(base.STOP_AT,2.)
np.savez_compressed(a.out/'initial.npz',positions=np.asarray(fluid.getFieldBuffer('position')).copy()[:fluid.numActiveParticles()],velocities=np.asarray(fluid.getFieldBuffer('velocity')).copy()[:fluid.numActiveParticles()],masses=np.array([fluid.getMass(i) for i in range(fluid.numActiveParticles())],dtype='f4'),supports=np.full(fluid.numActiveParticles(),sim.getSupportRadius(),dtype='f4'))
times=[];iters=[]
for step in range(a.warmup+a.steps):
 tm.setTime(float(step*.001));tm.setTimeStepSize(.001)
 tick=time.perf_counter();assert base.timeStepNoGUI();elapsed=time.perf_counter()-tick
 if step>=a.warmup:times.append(1000*elapsed);iters.append([int(ts.getValueUInt(ts.SOLVER_ITERATIONS_V)),int(ts.getNumIterations())])
 if step%50==0:
  speed=np.linalg.norm(np.asarray(fluid.getFieldBuffer('velocity')),axis=1).max();print('BENCH',step,step*.001,float(speed),flush=True)
  assert np.isfinite(speed) and speed<20
summary=dict(source=str(source),particles=fluid.numActiveParticles(),precision=str(np.asarray(fluid.getFieldBuffer('position')).dtype),device='CPU',threads=os.environ.get('OMP_NUM_THREADS'),engine_ms=statistics.mean(times),median_ms=statistics.median(times),steps=a.steps,warmup=a.warmup,iterations_mean=[statistics.mean(z[i] for z in iters) for i in [0,1]],support=sim.getSupportRadius(),kernel=sim.getKernel(),wall_method=int(sim.getBoundaryHandlingMethod()),head=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),binding_sha256=hashlib.sha256(Path(sph.__file__).read_bytes()).hexdigest())
(a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print('SUMMARY',json.dumps(summary),flush=True)
