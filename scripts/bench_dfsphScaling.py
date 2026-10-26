"""Frozen-start particle dam-break compute scaling, local GPU versus native CPU.

Run `plan --root PATH`, then `run --root PATH`, then `report --root PATH`.
No trajectory recording: repeated, identical first-step microbenchmarks, with
reset/output excluded from the timed integration. Boundary sources are copied
into the plan and never modified. All choices are written before measurements.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob('*.py'))}


def lattice_factor(kernel, nh=4):
    import numpy as np
    z = np.arange(-nh, nh + 1, dtype=float)
    x, y = np.meshgrid(z, z)
    q = np.minimum(np.hypot(x, y) / nh, 1)
    if kernel == 'CubicSpline':
        w = 40/(7*np.pi*nh**2) * np.where(q < .5, 1-6*q*q+6*q**3, 2*(1-q)**3)
    else:
        w = 9/(np.pi*nh**2) * (1-q)**6 * (1+6*q+35/3*q*q)
    return float(1/w.sum())


def plan(a):
    import numpy as np
    a.root.mkdir(parents=True, exist_ok=False)
    dep = REPO.parent/'curvatureBoundaries/src/warpSPHBoundaries'
    before = hashes(dep)
    shutil.copytree(dep, a.root/'deps/warpSPHBoundaries', ignore=shutil.ignore_patterns('__pycache__'))
    if before != hashes(dep) or before != hashes(a.root/'deps/warpSPHBoundaries'):
        raise RuntimeError('Boundary source changed during snapshot; preserve this attempt and re-plan.')
    manifest = dict(checkpoint=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        resolutions=a.nx, warmup=a.warmup, blocks=a.blocks, samples_per_block=a.samples,
        cpu_threads=a.threads, cpu_affinity='4/8 OpenMP threads bind to P cores0–7; >8 threads use cores0–23; close binding, OMP_DYNAMIC=FALSE', support_over_dx=4., kernels=['CubicSpline','Wendland4'],
        geometry=dict(width=3.2,height=1.,fluid_width=1.2,fluid_height=.6,wall_layers=6),
        dt='min(1e-4*sqrt(dx/0.01), dx^2)', viscosity=.01,
        policy='Repeated first integration at rest; reset x/v/rho/p/history outside timer, retain valid main adjacency; native searches every call. Default DF/CD stop tests, not forced iteration counts.',
        boundary_policy='Local sampled walls/contactShepard plus auxiliary frozen contour; native Akinci2012 sampled walls/self pressure. Common cubic4dx primary; current C4/4dx overlay.',
        boundary_snapshot_sha256=hashlib.sha256(json.dumps(before,sort_keys=True).encode()).hexdigest(),
        boundary_sources=before,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write(a.root/'manifest.json', manifest)
    shutil.copyfile(__file__, a.root/'benchmark.py')
    make_fixtures(a.root,a.nx)
    (a.root/'KEEP').write_text('Preserve particle DFSPH startup scaling inputs, failures and timing receipts.\n')
    print('PLAN',a.root,manifest['resolutions'],flush=True)


def make_fixtures(root,resolutions):
    import numpy as np
    for nx in resolutions:
        dx=1/nx;lo=np.array([-1.6,-.5]);hi=np.array([1.6,.5]);ncol=round(1.2*nx);nrow=round(.6*nx)
        xx,yy=np.meshgrid((np.arange(ncol)+.5)*dx+lo[0],(np.arange(nrow)+.5)*dx+lo[1],indexing='ij')
        x=np.stack((xx,yy),-1).reshape(-1,2).astype('f4')
        xx,yy=np.meshgrid((np.arange(-6,round(3.2*nx)+6)+.5)*dx+lo[0],(np.arange(-6,nx+6)+.5)*dx+lo[1],indexing='ij')
        wall=np.stack((xx,yy),-1).reshape(-1,2)
        wall=wall[~((wall>=lo)&(wall<=hi)).all(1)].astype('f4')
        directory=root/'fixtures'/str(nx);directory.mkdir(parents=True)
        np.savez(directory/'points.npz',fluid=x,wall=wall)
        for name,points in [('fluid',x),('wall',wall)]:
            with (directory/(name+'.geo')).open('w') as f:
                f.write(f'PGEOMETRY V5\nNPoints {len(points)} NPrims 0\nNPointGroups 0 NPrimGroups 0\nNPointAttrib 0 NVertexAttrib 0 NPrimAttrib 0 NAttrib 0\nPointAttrib\n')
                np.savetxt(f,points,fmt='%.9g %.9g 0 1 ()')
                f.write('beginExtra\nendExtra\n')
        write(directory/'metadata.json',dict(nx=nx,dx=dx,support=float(np.float32(4*dx)),dt=min(1e-4*(dx/.01)**.5,dx*dx),fluid=len(x),wall=len(wall),bounds=[lo.tolist(),hi.tolist()],
            masses={k:float(np.float32(dx*dx*lattice_factor(k))) for k in ['CubicSpline','Wendland4']},
            arrays_sha256={k:hashlib.sha256(z.tobytes()).hexdigest() for k,z in [('fluid',x),('wall',wall)]}))

def summarize(times, counts, cfg):
    warm=cfg['warmup'];n=cfg['samples_per_block'];used=times[warm:]
    blocks=[statistics.mean(used[i:i+n]) for i in range(0,len(used),n)]
    return dict(mean_ms=statistics.mean(used),median_ms=statistics.median(used),block_means_ms=blocks,
                block_min_ms=min(blocks),block_max_ms=max(blocks),samples=len(used),warmup=warm,
                DF_iterations=sorted(set(z[0] for z in counts[warm:])),CD_iterations=sorted(set(z[1] for z in counts[warm:])),
                DF_mean=statistics.mean(z[0] for z in counts[warm:]),CD_mean=statistics.mean(z[1] for z in counts[warm:]))


def local(a, cfg, meta, out):
    sys.path.insert(0,str(a.root/'deps'))
    from warpSPHBootstrap import bootstrap
    bootstrap(precision='float32')
    import numpy as np
    import torch
    assert torch.cuda.is_available(), 'GPU FP32 required'
    torch.set_num_threads(4)
    from warpSPH.runner import Case,run
    from warpSPH.geometry.types import ParticleSet
    from warpSPH.configurations.region import ParticleRegion,RegionType,BCType
    from warpSPH.initializers.weaklyCompressible import initializeState
    from warpSPH.rigidBody.ghostParticles import addBoundaryGhostParticles
    from warpSPH.rigidBody.build import buildAnalyticRigidBody
    from warpSPH.boundary import buildBoundaryProvider
    from warpSPH.caseUtils.weaklyCompressible import analyticTankBody
    from warpSPHCore import DomainDescription
    from warpSPH.regions import sampleDomainSDF
    from warpSPH.caseUtils.projection import initializeProjectionDensity
    from warpSPH.configurations import GravityType
    data=np.load(a.root/'fixtures'/str(a.nx)/'points.npz');initial={};counts=[];validations=[];neighbor={}
    def configure(ctx):
        sc,c=ctx.schemeConfig,ctx.config;dx=meta['dx']
        c.domain.min[:]=torch.tensor(np.array(meta['bounds'][0])-7*dx,device=ctx.device,dtype=ctx.dtype)
        c.domain.max[:]=torch.tensor(np.array(meta['bounds'][1])+7*dx,device=ctx.device,dtype=ctx.dtype);c.domain.periodic[:]=False
        c.calibrateNormalization=False;sc.fluid.restDensity=1.;sc.fluid.fixedSoundSpeed=10.
        sc.gravityConfig.active=True;sc.gravityConfig.type=GravityType.Directional;sc.gravityConfig.magnitude=9.81;sc.gravityConfig.direction=[0.,-1.]
        sc.iisph.boundaryRepresentation='particles';sc.iisph.boundaryPressure='contactShepard';sc.iisph.contactGravity=True
        sc.diffusionParams.inviscid=False;sc.diffusionParams.viscidNu=.01;sc.diffusionParams.approachOnly=False
        sc.shiftProperties.active=False;sc.surfaceDetectionConfig.active=False
        for p in [sc.dfsph.density,sc.dfsph.divergence]:p.relaxationFactor=.25;p.maxIterations=100
        ctx.velocityScale=1.;ctx.velocityScaleSource='Frozen startup microbenchmark'
    def build(ctx):
        dx=meta['dx'];h=meta['support'];m=meta['masses'][a.kernel]
        def region(points,kind):
            x=torch.as_tensor(points,device=ctx.device);n=len(x)
            ps=ParticleSet(x,torch.full((n,),h,device=ctx.device),torch.full((n,),m,device=ctx.device),torch.ones(n,device=ctx.device))
            return ParticleRegion(sdf=lambda z:torch.ones(len(z),device=z.device),type=kind,particles=ps,initialConditions={},kind=BCType.freeSlip)
        interior=DomainDescription(torch.tensor(meta['bounds'][0],device=ctx.device),torch.tensor(meta['bounds'][1],device=ctx.device),torch.zeros(2,device=ctx.device,dtype=torch.bool),2)
        regs=[region(data['fluid'],RegionType.Fluid),region(data['wall'],RegionType.Boundary)]
        regs[-1].sdf=lambda z:sampleDomainSDF(z,interior,invert=False)
        st=initializeState(regs,ctx.config,ctx.schemeConfig,ctx.SimulationState,verbose=False)
        st=addBoundaryGhostParticles(regs,st)
        contour=ParticleRegion(sdf=regs[-1].sdf,type=RegionType.Boundary,particles=None,kind=BCType.freeSlip,representation=analyticTankBody(interior))
        rb=buildAnalyticRigidBody(contour,0,st);provider=buildBoundaryProvider([contour],ctx.device);provider.rigidBodies=[rb]
        ctx.schemeConfig.boundaryProvider=provider;ctx.schemeConfig.rigidBodies=[rb]
        ctx.config.regions=ctx.schemeConfig.regions=regs+[contour];ctx.config.rigidBodies=[rb]
        return ctx.SimulationSystem(state=st,domain=ctx.config.domain)
    def initialize(ctx,system):
        initializeProjectionDensity(ctx,system);st=system.state
        for key in ['positions','velocities','densities','pressures','soundspeeds','boundaryAccelerations','divergencePressures','densityPressureImpulse','divergencePressureImpulse']:
            value=getattr(st,key,None)
            initial[key]=None if value is None else value.clone()
        # No hydrostatic pressure pre-seed; the column is initially released.
        for key in ['pressures','divergencePressures','densityPressureImpulse','divergencePressureImpulse']:
            if initial[key] is not None:initial[key].zero_();getattr(st,key).zero_()
        f=st.kinds==0;b=st.kinds==1
        checks={}
        for key,mask,name in [('positions',f,'fluid'),('positions',b,'wall')]:
            raw=getattr(st,key)[mask].cpu().numpy();checks[name]=hashlib.sha256(raw.tobytes()).hexdigest()==meta['arrays_sha256'][name]
        assert all(checks.values()),checks
        write(out/'initial_audit.json',dict(equal_fixture=checks,fluid=int(f.sum()),wall=int(b.sum()),ghost=int((st.kinds==2).sum()),total=len(st.positions),dtype=str(st.positions.dtype),mass=float(st.masses[f][0]),support=float(st.supports[f][0]),density_min=float(st.densities[f].min()),density_max=float(st.densities[f].max()),boundary_dependency=str(sys.modules['warpSPHBoundaries'].__file__),verletScale=float(ctx.config.verletScale)))
    def reset(ctx,system,step):
        st=system.state;u=ctx.scratch['lastStageUpdate'];counts.append([u.divergenceResult.iterations,u.densityResult.iterations])
        assert u.divergenceResult.termination=='converged' and u.densityResult.termination=='converged','Unconverged startup projection'
        if step in (0,cfg['warmup'],cfg['warmup']+cfg['blocks']*cfg['samples_per_block']-1):
            f=st.kinds==0;vmax=float(st.velocities[f].norm(dim=-1).max());motion=float((st.positions[f]-initial['positions'][f]).norm(dim=-1).max()/meta['dx'])
            assert np.isfinite(vmax) and vmax<10 and motion<.1
            validations.append(dict(step=step,vmax=vmax,displacement_dx=motion,DF_error=u.divergenceResult.error,CD_error=u.densityResult.error))
            if step==0:
                adj=system.adjacency
                neighbor.update(rows=len(st.positions),candidate_edges=int(adj.numNeighbors.sum()),fluid_candidate_mean=float(adj.numNeighbors[f].float().mean()))
            print('CHECK',a.nx,a.kernel,validations[-1],flush=True)
        for key,value in initial.items():
            setattr(st,key,None if value is None else value.clone())
        system.t=0.
        # Reset work and its synchronization are outside the integration timer.
        torch.cuda.synchronize()
    case=Case('frozen_particle_dambreak','dfsph',build,configureScheme=configure,initialConditions=initialize,postStep=reset,timestep=lambda ctx,system:meta['dt'])
    n=cfg['warmup']+cfg['blocks']*cfg['samples_per_block'];started=time.perf_counter()
    r=run(case,scheme='dfsph',device='cuda',nx=a.nx,L=1.,kernel=a.kernel,n_h=4.,dt=meta['dt'],maxDt=meta['dt'],adaptiveDt=False,calibrateNormalization=False,integrationScheme='semiImplicitEuler',pipelineOutputs=False,nSteps=n,tLimit=None,plot=False,video=False,store=False,cudaGraph=False,quiet=True,progress=True,velocityAlarmPlotInterval=1,stallProgress=0.)
    times=[z['stepTime_ms'] for z in r.trajectory if z['step']>=0]
    result=summarize(times,counts,cfg);result.update(device=torch.cuda.get_device_name(),total_wall_seconds=time.perf_counter()-started,runner_wall_seconds=r.wallTime,runner_wall_ms=1000*r.wallTime/n,neighbor=neighbor,checks=validations,diverged=r.diverged,stop=r.stopReason,torch_peak_bytes=torch.cuda.max_memory_allocated())
    assert not r.diverged
    return result


def native(a,cfg,meta,out):
    import numpy as np
    native_root=REPO.parent/'SPlisHSPlasH';sys.path.insert(0,str(native_root/'build/lib'));import pysplishsplash as sph
    fixture=a.root/'fixtures'/str(a.nx);data=np.load(fixture/'points.npz');x=data['fluid'];wall=data['wall'];m=meta['masses']['CubicSpline'];H=meta['support']
    scene=json.loads((native_root/'data/Scenes/DamBreakModel_2D.json').read_text());scene.pop('FluidBlocks');scene['FluidModels']=[dict(particleFile=str(fixture/'fluid.geo'),translation=[0,0,0],scale=[1,1,1])]
    scene['Configuration'].update(timeStepSize=meta['dt'],cflMethod=0,particleRadius=H/4,kernel=0,gradKernel=0,gravitation=[0,-9.81,0],boundaryHandlingMethod=0)
    scene['Configuration']['DFSPH'].update(maxError=.01,minIterations=2,maxIterations=100,maxErrorV=.1,maxIterationsV=100,enableDivergenceSolver=True)
    scene['Materials'][0].update(density0=1.,viscosityMethod=1,vorticityMethod=0,surfaceTensionMethod=0)
    scene['Materials'][0]['Standard viscosity'].update(viscosity=.01,viscosityBoundary=0.)
    scene['RigidBodies'][0].update(geometryFile=str(native_root/'data/models/UnitBox.obj'),particleFile=os.path.relpath(fixture/'wall.geo',out),translation=[0,0,0],scale=[1,1,1])
    write(out/'scene.json',scene)
    base=sph.Exec.SimulatorBase();base.init(sceneFile=str(out/'scene.json'),useGui=False,initialPause=False,useCache=False,outputDir=str(out/'native_output'),stopAt=.5);base.initSimulation()
    sim=sph.Simulation.getCurrent();fm=sim.getFluidModel(0);ts=sim.getTimeStep();tm=sph.TimeManager.getCurrent();n=len(x)
    def field(name):return np.asarray(fm.getFieldBuffer(name))[:n]
    assert n==fm.numActiveParticles()
    radius=np.float32(.5*np.sqrt(m/.8));candidates=[radius];lo=hi=radius
    for _ in range(8):lo=np.nextafter(lo,np.float32(0));hi=np.nextafter(hi,np.float32(1));candidates.extend([lo,hi])
    trial=[]
    for rad in candidates:sim.setParticleRadius(float(rad));fm.setDensity0(1.);trial.append((abs(fm.getVolume(0)-m),float(rad)))
    _,rad=min(trial);sim.setParticleRadius(rad);fm.setDensity0(1.);volume=fm.getVolume(0)
    assert abs(volume/m-1)<3e-7
    for i in range(n):fm.setMass(i,m)
    sim.setParticleRadius(H/4);sim.getNeighborhoodSearch().setSearchRadius(H);sim.setKernel(1);sim.setKernel(0);sim.setGradKernel(1);sim.setGradKernel(0)
    tm.setTime(1.);base.runSimulation();assert tm.getTime()==1.;base.setValueFloat(base.STOP_AT,2.)
    bm=sim.getBoundaryModel(0);assert bm.numberOfParticles()==len(wall)
    for i in range(len(wall)):bm.setVolume(i,m);bm.setVelocity(i,np.zeros(3,dtype='f4'))
    # Spatial sorting may reorder buffers: use native IDs on every reset.
    def reset():
        ids=field('id').astype(int);field('position')[:,:2]=x[ids];field('position')[:,2]=0.;field('velocity')[:]=0.;ts.reset();tm.setTime(0.);tm.setTimeStepSize(meta['dt'])
    reset();ids=field('id').astype(int)
    assert np.array_equal(field('position')[:,:2],x[ids])
    actual=np.array([bm.getPosition(i)[:2] for i in range(len(wall))],dtype='f4')
    assert np.array_equal(actual[np.lexsort((actual[:,1],actual[:,0]))],wall[np.lexsort((wall[:,1],wall[:,0]))])
    write(out/'initial_audit.json',dict(fluid=n,wall=len(wall),total=n+len(wall),dtype=str(field('position').dtype),fluid_positions_equal=True,wall_positions_equal=True,zero_velocities=True,mass=m,effective_volume=volume,volume_relative_error=volume/m-1,support=sim.getSupportRadius(),head=subprocess.check_output(['git','-C',str(native_root),'rev-parse','HEAD'],text=True).strip(),binding_sha256=hashlib.sha256(Path(sph.__file__).read_bytes()).hexdigest()))
    times=[];counts=[];checks=[];started=time.perf_counter();steps=cfg['warmup']+cfg['blocks']*cfg['samples_per_block'];neighbor={}
    for i in range(steps):
        reset()
        if i in (0,cfg['warmup'],steps-1):
            ids=field('id').astype(int)
            assert np.array_equal(field('position')[:,:2],x[ids]) and not np.any(field('velocity'))
            assert not np.any(field('p / rho^2')) and not np.any(field('p_v / rho^2'))
        t0=time.perf_counter();assert base.timeStepNoGUI();times.append(1000*(time.perf_counter()-t0));counts.append([int(ts.getValueUInt(ts.SOLVER_ITERATIONS_V)),int(ts.getNumIterations())])
        assert counts[-1][0]<100 and counts[-1][1]<100,'Capped reference startup solve'
        if i in (0,cfg['warmup'],steps-1):
            v=field('velocity');vmax=float(np.linalg.norm(v,axis=1).max());ids=field('id').astype(int);motion=float(np.linalg.norm(field('position')[:,:2]-x[ids],axis=1).max()/meta['dx'])
            assert np.isfinite(vmax) and vmax<10 and motion<.1
            checks.append(dict(step=i,vmax=vmax,displacement_dx=motion));print('CHECK',a.nx,'CPU',checks[-1],flush=True)
            if i==0:
                sample=np.linspace(0,n-1,min(n,2048),dtype=int);neighbor['sampled_queries']=len(sample);neighbor['mean_fluid_neighbors']=float(np.mean([sim.numberOfNeighbors(0,0,int(j)) for j in sample]));neighbor['mean_wall_neighbors']=float(np.mean([sim.numberOfNeighbors(0,1,int(j)) for j in sample]))
        if i%24==0:print('PROGRESS',i,steps,statistics.mean(times[-24:]),flush=True)
    result=summarize(times,counts,cfg);result.update(total_wall_seconds=time.perf_counter()-started,neighbor=neighbor,checks=checks,device='CPU',threads=int(os.environ['OMP_NUM_THREADS']))
    return result


def worker(a):
    if a.backend=='cpu' and a.kernel!='CubicSpline':raise ValueError('Native CPU benchmark uses its cubic kernel')
    a.root=a.root.resolve();cfg=json.loads((a.root/'manifest.json').read_text());meta=json.loads((a.root/'fixtures'/str(a.nx)/'metadata.json').read_text())
    out=a.root/'runs'/f'{a.backend}_{a.kernel}_{a.nx}_t{a.worker_threads}';out.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter()
    result=local(a,cfg,meta,out) if a.backend=='gpu' else native(a,cfg,meta,out)
    result.update(benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),nx=a.nx,kernel=a.kernel,backend=a.backend,fluid=meta['fluid'],wall=meta['wall'],dx=meta['dx'],support=meta['support'],dt=meta['dt'],process_wall_seconds=time.perf_counter()-start)
    write(out/'result.json',result);print('RESULT',json.dumps(result),flush=True)


def run(a):
    cfg=json.loads((a.root/'manifest.json').read_text())
    resolutions=a.nx or cfg['resolutions']
    jobs=[]
    for nx in resolutions:
        jobs.extend([('gpu',k,nx,4) for k in cfg['kernels']])
        jobs.extend([('cpu','CubicSpline',nx,t) for t in (a.threads or cfg['cpu_threads'])])
    for backend,kernel,nx,threads in jobs:
        label=f'{backend}_{kernel}_{nx}_t{threads}';out=a.root/'runs'/label
        if (out/'result.json').exists():print('SKIP completed',label,flush=True);continue
        log=a.root/'logs'/(label+'.log');log.parent.mkdir(exist_ok=True)
        print('START',label,flush=True)
        env=dict(os.environ,OMP_NUM_THREADS=str(threads),PYTHONPATH=str(REPO/'src'),WARP_CACHE_PATH='/tmp/warpsph-warp-cache',MPLCONFIGDIR='/tmp/warpsph-mpl',PYOPENGL_PLATFORM='egl',EGL_PLATFORM='surfaceless',LIBGL_ALWAYS_SOFTWARE='1')
        if backend=='cpu':env.update(OMP_PLACES=','.join('{'+str(i)+'}' for i in range(24 if threads>8 else 8)),OMP_PROC_BIND='close',OMP_DYNAMIC='FALSE')
        with log.open('w') as f:
            process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'worker','--root',str(a.root),'--nx',str(nx),'--backend',backend,'--kernel',kernel,'--worker-threads',str(threads)],env=env,stdout=f,stderr=subprocess.STDOUT,cwd=REPO)
            while process.poll() is None:
                time.sleep(5)
                with log.open('rb') as watch:
                    watch.seek(max(0,log.stat().st_size-1500));tail=watch.read().decode(errors='replace').replace('\r','\n').splitlines()
                line=tail[-1] if tail else 'initializing'
                print(label,line[:220],flush=True)
            if process.returncode:raise RuntimeError(f'{label} failed; preserve {log}')
        result=json.loads((out/'result.json').read_text());print('DONE',label,result['mean_ms'],'ms',result['DF_iterations'],result['CD_iterations'],flush=True)


def report(a):
    import csv
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import EngFormatter
    rows=[json.loads(p.read_text()) for p in (a.root/'runs').glob('*/result.json')]
    rows.sort(key=lambda r:(r['backend'],r['kernel'],r.get('threads',4),r['fluid']))
    series={}
    for row in rows:
        key=(row['backend'],row['kernel'],row.get('threads',4))
        series.setdefault(key,[]).append(row)
    fits={};crossings=[];knees={}
    for key,data in series.items():
        x=np.array([z['fluid'] for z in data],float);y=np.array([z['mean_ms'] for z in data])
        if len(data)>=3:
            if key[0]=='gpu':
                floor=float(np.median(y[x<=7200]));bracket=None
                for lo,hi in zip(data,data[1:]):
                    if lo['mean_ms']<=1.2*floor<hi['mean_ms']:
                        bracket=[lo['fluid'],hi['fluid']];break
                knees[str(key)]=dict(small_N_floor_ms=floor,departure_rule='First observed bracket above 1.2 times small-N floor',bracket=bracket)
            # Descriptive large-N trend, not an extrapolated crossover.
            take=x>=115200
            if take.sum()>=2:
                slope,intercept=np.polyfit(x[take],y[take],1)
                fits[str(key)]=dict(large_N_slope_ms_per_particle=float(slope),intercept_ms=float(intercept),range=[int(x[take].min()),int(x[take].max())])
    for gpu_key in [k for k in series if k[0]=='gpu']:
        gpu={z['fluid']:z for z in series[gpu_key]}
        for cpu_key in [k for k in series if k[0]=='cpu']:
            cpu={z['fluid']:z for z in series[cpu_key]};common=sorted(gpu.keys()&cpu.keys())
            brackets=[]
            for lo,hi in zip(common,common[1:]):
                dl=cpu[lo]['mean_ms']-gpu[lo]['mean_ms'];dh=cpu[hi]['mean_ms']-gpu[hi]['mean_ms']
                if dl*dh<0:
                    estimate=lo+(hi-lo)*(-dl)/(dh-dl)
                    robust=(cpu[lo]['block_max_ms']<gpu[lo]['block_min_ms'] and cpu[hi]['block_min_ms']>gpu[hi]['block_max_ms'])
                    brackets.append(dict(fluid_bracket=[lo,hi],linear_interpolation_particles=estimate,block_ranges_separated=robust))
            crossings.append(dict(gpu=str(gpu_key),cpu=str(cpu_key),observed_crossings=brackets))
    summary=dict(manifest=json.loads((a.root/'manifest.json').read_text()),rows=rows,departure=knees,crossings=crossings,large_N_trends=fits)
    write(a.root/'summary.json',summary)
    keys=['backend','kernel','threads','nx','fluid','wall','mean_ms','median_ms','block_min_ms','block_max_ms','DF_mean','CD_mean','runner_wall_ms','total_wall_seconds']
    with (a.root/'timings.csv').open('w') as f:
        w=csv.DictWriter(f,keys,extrasaction='ignore',lineterminator='\n');w.writeheader();w.writerows(rows)
    fig,axes=plt.subplots(1,2,figsize=(13,5.4),layout='constrained')
    styles={('gpu','CubicSpline',4):('#1f77b4','GPU cubic, 4dx','o','-'),('gpu','Wendland4',4):('#188a64','GPU C4, 4dx','s','--'),('cpu','CubicSpline',4):('#cf6722','CPU cubic, 4 P cores','^','-'),('cpu','CubicSpline',8):('#9c629e','CPU cubic, 8 P cores','v',':'),('cpu','CubicSpline',24):('#666666','CPU cubic, all 24 cores','x','-.')}
    for key,data in series.items():
        color,label,marker,ls=styles[key];x=np.array([z['fluid'] for z in data]);y=np.array([z['mean_ms'] for z in data]);low=np.array([z['block_min_ms'] for z in data]);high=np.array([z['block_max_ms'] for z in data])
        for ax in axes:
            ax.plot(x,y,color=color,label=label,marker=marker,ls=ls,lw=1.7,ms=4)
            ax.fill_between(x,low,high,color=color,alpha=.12)
    axes[0].set(xscale='log',yscale='log',title='Full measured range',xlabel='Fluid particles',ylabel='Warmed integration time [ms / step]')
    axes[1].set(xlim=(0,190000),ylim=(0,85),title='Crossover and departure from the fixed-cost plateau',xlabel='Fluid particles',ylabel='Warmed integration time [ms / step]')
    floor=knees.get(str(('gpu','CubicSpline',4)),{}).get('small_N_floor_ms')
    if floor:
        axes[1].axhline(floor,color='#666666',lw=1,ls='--')
        axes[1].axhline(1.2*floor,color='#888888',lw=1,ls=':')
        axes[1].text(185000,20,'1.2 × small-N floor',ha='right',fontsize=8,color='#555555')
    primary=next((z for z in crossings if z['gpu']==str(('gpu','CubicSpline',4)) and z['cpu']==str(('cpu','CubicSpline',4))),None)
    if primary and primary['observed_crossings']:
        c=primary['observed_crossings'][0];lo,hi=c['fluid_bracket'];estimate=c['linear_interpolation_particles']
        axes[1].axvspan(lo,hi,color='#cf6722',alpha=.08)
        axes[1].annotate(f'4-core crossing ~{estimate/1000:.0f}k\nobserved bracket {lo/1000:.1f}–{hi/1000:.1f}k',xy=(estimate,12.2),xytext=(65000,27),fontsize=8,arrowprops=dict(arrowstyle='->',color='#444444'),bbox=dict(facecolor='white',edgecolor='#dddddd',alpha=.9))
    for ax in axes:
        ax.xaxis.set_major_formatter(EngFormatter());ax.grid(True,which='both',alpha=.18)
    axes[0].legend(loc='upper left',fontsize=8)
    fig.suptitle('DFSPH particle dam-break startup scaling | FP32, fixed geometry, native stop tests',fontsize=12)
    fig.supxlabel('Shading: range of three 24-step block means, not a confidence interval. Geometry reset and output excluded from timing.',fontsize=8)
    fig.savefig(a.root/'scaling.png',dpi=190);plt.close(fig)
    print('REPORT',a.root/'scaling.png',json.dumps(crossings),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['plan','extend','run','worker','report']);p.add_argument('--root',type=Path,required=True);p.add_argument('--nx',nargs='+',type=int);p.add_argument('--warmup',type=int,default=12);p.add_argument('--blocks',type=int,default=3);p.add_argument('--samples',type=int,default=24);p.add_argument('--threads',nargs='+',type=int);p.add_argument('--backend',choices=['gpu','cpu']);p.add_argument('--kernel',default='CubicSpline');p.add_argument('--worker-threads',type=int,default=4)
    a=p.parse_args();a.root=a.root.resolve()
    if a.action=='plan':a.nx=a.nx or [25,50,100,200,400,800,1600];a.threads=a.threads or [4];plan(a)
    elif a.action=='extend':
        if not a.nx:p.error('extend needs --nx')
        cfg=json.loads((a.root/'manifest.json').read_text())
        fresh=sorted(set(a.nx)-set(cfg['resolutions']));make_fixtures(a.root,fresh)
        cfg.setdefault('resolution_refinements',[]).append(dict(resolutions=fresh,reason='Refine observed crossover and departure from launch plateau; numerical policies unchanged.'))
        cfg['resolutions']=sorted(set(cfg['resolutions'])|set(fresh));write(a.root/'manifest.json',cfg)
    elif a.action=='report':report(a)
    elif a.action=='run':run(a)
    else:
        if not a.nx or len(a.nx)!=1:p.error('worker needs one --nx')
        a.nx=a.nx[0];worker(a)


if __name__=='__main__':main()
