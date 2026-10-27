"""Daemon-recorded 2D research contact/acoustic trajectories, not production CRK.

Periodic unit square, compact cubic scatter RK, global edge projection,
coherent pressure/volume work and explicit midpoint. No AV or shock claims.
"""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
from crkClosure2D import sample,geometry,close_edges,pressure_rhs,compact_taper


def kh_initial(n,jitter=0.):
    """Match existing sampleKHH's smooth-density parameters and logical order.

    rho1/2=1/2, vx1/2=.5/-.5, delta=.025, w0=.1, sigma=.05/sqrt(2),
    freq=4, pressure=2.5 and gamma=5/3. This is a smooth-flow research control.
    """
    x,V=sample(n,unequal=False,jitter=jitter);y=x[:,1];delta=.025
    inner=(y>=.25)&(y<.75)
    distance=np.minimum(abs(y-.25),abs(y-.75));ramp=np.exp(-distance/delta)
    rho=np.where(inner,2-.5*ramp,1+.5*ramp)
    vx=np.where(inner,-.5+.5*ramp,.5-.5*ramp)
    sigma=.05/np.sqrt(2)
    vy=.1*np.sin(4*np.pi*x[:,0])*(np.exp(-.5*((y-.25)/sigma)**2)+np.exp(-.5*((y-.75)/sigma)**2))*abs(.5-y)
    return x,V,rho,np.stack((vx,vy),axis=1)


def render(root):
    import h5py
    from dataclasses import replace
    from warpSPHBootstrap import bootstrap
    bootstrap(precision='float32')
    import torch
    from warpSPH.runner import RunConfiguration
    from warpSPH.cases.compressible.hydrostatic import hydrostaticCase
    from warpSPH.cases.common.plotting import Field,particlePlot
    from warpSPH.runner.display import closeWindow
    from warpSPH.runner.media import encodeFrames
    settings=json.loads((root/'researchRun.json').read_text())
    ctx=RunConfiguration('hydrostatic',simulation=dict(nx=24,plot=True,video=True,show=False,plotBackend='vispy')).prepare(hydrostaticCase)
    ctx.case=replace(ctx.case,name='2D closure '+settings['case']+' '+settings['shape'])
    ctx.exportPath=str(root);ctx.imagePath=str(root/'images');Path(ctx.imagePath).mkdir(exist_ok=True)
    ctx.config.dt=settings['dt'];ctx.config.domain.min[:]=0;ctx.config.domain.max[:]=1
    variant='tapered' if settings.get('taper') else 'untapered'
    ctx.scratch['_titleProvenance']='Experimental 2D '+variant+' | '+settings.get('backend','cpu')+' | coupled pressure/work | midpoint | no AV'
    acoustic=settings['case']=='acoustic';A=settings['amplitude']
    fields=[Field('densities','Density',vMin=1-2*A if acoustic else .2,vMax=1+2*A),
            Field('pressures','Pressure',vMin=1-2*A if acoustic else .999999,vMax=1+2*A if acoustic else 1.000001),
            Field('velocities','Velocity',mapping='L2Norm',vMin=0,vMax=2*A if acoustic else .4)]
    if settings['case']=='kh':
        fields=[Field('densities','Density',vMin=.8,vMax=2.2),
                Field('velocities','Velocity',mapping='L2Norm',vMin=0,vMax=.7),
                Field('materials','Initial layer',vMin=0,vMax=1)]
    setup,update=particlePlot(fields,figsize=(12,4));plotter=None
    with h5py.File(root/'trajectory.h5','r') as f:
        for number,key in enumerate(sorted(f['states'])):
            g=f['states'][key];state=NS(**{k:torch.as_tensor(v[()],device=ctx.device) for k,v in g.items()})
            system=NS(state=state,t=float(g.attrs['time']))
            if plotter is None:plotter=setup(ctx,system)
            else:update(ctx,system,plotter,number)
    if plotter:
        if hasattr(plotter,'drain'):plotter.drain()
        closeWindow(plotter)
    return encodeFrames(ctx.imagePath,str(root))


def run(args):
    import h5py
    from warpsph_remote.publisher import SimulationPublisher,SimulationStopped
    root=args.output;root.mkdir(parents=True,exist_ok=False)
    x,V=sample(args.n,unequal=not args.uniform,jitter=args.jitter,shape=args.shape);n=len(V)
    acoustic=args.case=='acoustic';kh=args.case=='kh';fine=V<2/args.n**2
    if args.uniform:
        fine=np.linalg.norm(x-.5,axis=1)<.25 if args.shape=='circle' else x[:,0]<.5
    rho0=np.ones(n) if acoustic else np.where(fine,1.,.25)
    gamma=np.full(n,1.4) if acoustic else np.where(fine,1.4,5/3)
    boost=np.array([.3,.2]) if args.case=='advected' else np.zeros(2)
    P0=1+args.amplitude*np.sin(2*np.pi*x.sum(axis=1)) if acoustic else np.ones(n)
    velocity=np.tile(boost,(n,1))
    if kh:
        x,V,rho0,velocity=kh_initial(args.n,args.jitter);n=len(V)
        gamma=np.full(n,5/3);P0=np.full(n,2.5);fine=(x[:,1]>=.25)&(x[:,1]<.75)
    mass=V*rho0
    # x/y position, x/y velocity, volume, specific internal energy.
    y=np.column_stack((x,velocity,V,P0/((gamma-1)*rho0)))
    x0=x.copy();V0=V.copy();entropy0=P0/rho0**gamma
    energy0=float(np.sum(mass*(y[:,5]+.5*np.sum(y[:,2:4]**2,axis=1))))
    publisher=SimulationPublisher(f'2D closure research {args.case} {args.backend} n{args.n} dt{args.dt:g}'+(' tapered' if args.taper else ''),measurements=True,diagnostic_interval=.001)
    print('Remote run registered:',publisher.id,flush=True)
    file=h5py.File(root/'trajectory.h5','w')
    ctx=NS(spec=NS(store=True,storeMode='trajectory',tLimit=args.time),scratch={'trajectoryFile':file},
        exportPath=str(root.resolve()),scheme='Experimental 2D global edge closure',
        config=NS(dt=args.dt,domain=NS(min=np.zeros(2),max=np.ones(2),periodic=np.ones(2,dtype=bool)),
                  kernel='CubicSpline',integrationScheme='explicitMidpoint'))
    history=[];steps=0;t=0.;last_frame=-1;geometry_seconds=0.;closure_seconds=0.;rhs_calls=0;maxiterations=0
    gpu=None;setup_wall=0.;max_constraint_residual=0.
    def arrays():
        rho=mass/y[:,4];P=(gamma-1)*rho*y[:,5]
        return dict(positions=y[:,:2],velocities=y[:,2:4],densities=rho,pressures=P,
            internalEnergies=y[:,5],supports=4*np.sqrt(y[:,4]),volumes=y[:,4],masses=mass,
            materials=(~fine).astype(np.int32),UIDs=np.arange(n,dtype=np.int64),kinds=np.zeros(n,dtype=np.int32))
    def rhs(state):
        nonlocal geometry_seconds,closure_seconds,rhs_calls,maxiterations,max_constraint_residual
        if not np.isfinite(state).all() or np.any(state[:,4:]<=0):raise RuntimeError('Invalid trial state')
        x=state[:,:2];v=state[:,2:4];V=state[:,4];rho=mass/V;P=(gamma-1)*rho*state[:,5]
        if gpu is not None:
            info=gpu.solve(x,V,return_numpy=False);closure_seconds+=info['wallSeconds'];rhs_calls+=1
            maxiterations=max(maxiterations,info['iterations']);max_constraint_residual=max(max_constraint_residual,info['constraintResidual'])
            a,dV,du=gpu.pressure_rhs(V,mass,P,v)
            return np.column_stack((v,a,dV,du))
        start=time.perf_counter();i,j,r,q0,R,d=geometry(x,V);geometry_seconds+=time.perf_counter()-start
        q,info=close_edges(V,i,j,r,q0,tolerance=1.e-13,
            edge_scale=compact_taper(V,i,j,r) if args.taper else None)
        closure_seconds+=info['wallSeconds'];rhs_calls+=1
        maxiterations=max(maxiterations,*info['iterations'])
        a,dV,du=pressure_rhs(V,mass,P,v,i,j,q)
        return np.column_stack((v,a,dV,du))
    def record(force=False):
        nonlocal last_frame
        a=arrays();P=a['pressures'];rho=a['densities'];v=y[:,2:4]
        energy=float(np.sum(mass*(y[:,5]+.5*np.sum(v*v,axis=1))))
        row=dict(step=steps,time=t,pressureError=float(np.max(abs(P-(2.5 if kh else 1)))),
            maxPeculiarVelocity=float(np.max(np.linalg.norm(v-boost,axis=1))),
            relativeEnergyDrift=energy/energy0-1,volumeError=float(abs(y[:,4].sum()/V0.sum()-1)),
            momentumError=float(np.linalg.norm(np.sum(mass[:,None]*(v-velocity),axis=0))),
            entropyRelativeError=float(np.max(abs(P/rho**gamma/entropy0-1))),
            minVolume=float(y[:,4].min()),minEnergy=float(y[:,5].min()))
        if acoustic:
            c=np.sqrt(1.4);phase=2*np.pi*y[:,:2].sum(axis=1);omega=c*2*np.pi*np.sqrt(2)
            exactP=1+args.amplitude*np.sin(phase)*np.cos(omega*t)
            exactv=(-args.amplitude/(c*np.sqrt(2))*np.cos(phase)*np.sin(omega*t))[:,None]*np.ones((1,2))
            row['pressureWaveL1']=float(np.dot(y[:,4],abs(P-exactP))/y[:,4].sum())
            row['velocityWaveL1']=float(np.dot(y[:,4],np.linalg.norm(v-exactv,axis=1))/y[:,4].sum())
        elif kh:
            row['transverseRMS']=float(np.sqrt(np.dot(mass,v[:,1]**2)/mass.sum()))
            row['seedMode']=float(2*np.dot(y[:,4],v[:,1]*np.sin(4*np.pi*y[:,0]))/y[:,4].sum())
        else:row['densityRelativeError']=float(np.max(abs(rho/rho0-1)))
        history.append(row)
        if steps%args.frame_every==0 or force:
            if last_frame!=steps:
                g=file.require_group('states').create_group(f'frame_{steps:06d}');g.attrs['time']=t
                for k,vv in a.items():g.create_dataset(k,data=vv)
                file.flush();last_frame=steps
        publisher.post_step(ctx,NS(state=NS(**a),t=t),steps)
        if steps%args.frame_every==0 or force:
            print(f'step={steps} | t={t:.7f} | dt={ctx.config.dt:.3g} | Perror={row["pressureError"]:.3g} | speed={row["maxPeculiarVelocity"]:.3g} | energyDrift={row["relativeEnergyDrift"]:.3g} | maxSolveIters={maxiterations}',flush=True)
        return row
    overall_started=time.perf_counter();started=overall_started
    try:
        if args.backend!='cpu':
            from crkClosureGPU2D import GPUClosure
            publisher.update(message='Preparing GPU closure and warming kernels')
            gpu=GPUClosure(n,logical_n=args.n if args.backend=='gpu-fft' else 0,
                coarse_k=args.coarse_k if args.backend=='gpu-coarse' else 0)
            gpu.solve(y[:,:2],y[:,4],return_numpy=False)
            setup_wall=time.perf_counter()-overall_started;started=time.perf_counter()
        record()
        while t<args.time-1.e-14:
            dt=min(args.dt,args.time-t);ctx.config.dt=dt
            if kh:
                rho=mass/y[:,4];P=(gamma-1)*rho*y[:,5]
                signal=np.max(np.linalg.norm(y[:,2:4],axis=1)+np.sqrt(gamma*P/rho))
                dt=min(dt,.2*np.sqrt(y[:,4].min())/signal);ctx.config.dt=dt
                if dt<1.e-7 or signal>100:raise RuntimeError('KH timestep/velocity watchdog')
            mid=y+.5*dt*rhs(y);mid[:,:2]%=1
            y=y+dt*rhs(mid);y[:,:2]%=1;steps+=1;t+=dt
            if not np.isfinite(y).all() or np.any(y[:,4:]<=0):raise RuntimeError('Invalid completed state')
            record()
            if steps>100000:raise RuntimeError('Step watchdog exceeded')
        record(force=True);integration_wall=time.perf_counter()-started;a=arrays()
        publisher.finish(ctx,NS(state=NS(**a),t=t),steps)
        np.savez_compressed(root/'finalState.npz',**a,gamma=gamma,initialPositions=x0,initialVolumes=V0)
        file.close();(root/'diagnosticHistory.json').write_text(json.dumps(history,indent=2)+'\n')
        maximum=lambda name:max(abs(r[name]) for r in history)
        gates=dict(completed=abs(t-args.time)<1.e-12,energy=maximum('relativeEnergyDrift')<1.e-6,
            volume=maximum('volumeError')<1.e-8,momentum=maximum('momentumError')<1.e-10,entropy=maximum('entropyRelativeError')<1.e-5)
        if not acoustic and not kh:gates.update(pressure=maximum('pressureError')<1.e-8,quiescent=maximum('maxPeculiarVelocity')<1.e-8,density=maximum('densityRelativeError')<1.e-8)
        elif acoustic:gates.update(pressureWave=maximum('pressureWaveL1')<.2*args.amplitude,velocityWave=maximum('velocityWaveL1')<.2*args.amplitude)
        else:gates.update(boundedVelocity=maximum('maxPeculiarVelocity')<2.,nontrivialEvolution=abs(history[-1]['transverseRMS']-history[0]['transverseRMS'])>1.e-6)
        summary=dict(case=args.case,shape=args.shape,n=args.n,particles=n,jitter=args.jitter,dt=args.dt,taper=args.taper,uniform=args.uniform,backend=args.backend,coarseK=args.coarse_k if args.backend=='gpu-coarse' else 0,
            amplitude=args.amplitude,time=t,steps=steps,remoteRunID=publisher.id,integrationWallSeconds=integration_wall,
            meanStepMilliseconds=integration_wall/steps*1000,geometrySeconds=geometry_seconds,closureSeconds=closure_seconds,
            closureTimingScope='GPU closureSeconds includes neighbor/skin, RK geometry, projection and verification; CPU geometrySeconds and closureSeconds are separate.',
            rhsCalls=rhs_calls,maxSolveIterations=maxiterations,maxConstraintResidual=max_constraint_residual,setupWallSeconds=setup_wall,
            topologyRebuilds=gpu.rebuilds if gpu is not None else rhs_calls,initialVolumeSum=float(V0.sum()),final=history[-1],
            maxima={k:maximum(k) for k in history[-1] if k not in ('step','time')},
            gates={k:bool(v) for k,v in gates.items()},passed=all(gates.values()),
            qualificationScope='Internal gates only; KH physical accuracy and particle distribution remain unqualified.' if kh else 'Listed contact/wave gates only; no general method qualification.',
            limits='Independent 2D research model, not production CRK. No AV, shocks or exact time-discrete compatible energy. GPU pipeline uses CPU neighbor search/skin and CPU integration/IO. FFT preconditioner requires logical lattice indexing; actual moving geometry constraints are checked. KH gates check stability/conservation, not literature solution accuracy or particle-distribution quality.')
        (root/'researchRun.json').write_text(json.dumps(summary,indent=2)+'\n')
        publisher.update(message='Rendering recorded frames with vispy');video=render(root)
        summary.update(video=video,totalWallSeconds=time.perf_counter()-overall_started)
        (root/'researchRun.json').write_text(json.dumps(summary,indent=2)+'\n')
        publisher.close('completed' if summary['passed'] else 'failed','' if summary['passed'] else 'Research qualification gate failed')
        print(json.dumps(summary,indent=2),flush=True)
        return 0 if summary['passed'] else 1
    except BaseException as error:
        if not isinstance(error,SimulationStopped) and file.id.valid and np.isfinite(y).all() and np.all(y[:,4:]>0):
            try:record(force=True)
            except Exception:pass
        if file.id.valid:file.close()
        (root/'diagnosticHistory.json').write_text(json.dumps(history,indent=2)+'\n')
        publisher.close('stopped' if isinstance(error,SimulationStopped) else 'failed',str(error))
        (root/'failure.json').write_text(json.dumps(dict(step=steps,time=t,error=str(error)),indent=2)+'\n')
        if not (root/'researchRun.json').exists():
            wall=time.perf_counter()-started
            failed=dict(case=args.case,shape=args.shape,n=args.n,particles=n,jitter=args.jitter,dt=args.dt,
                taper=args.taper,backend=args.backend,amplitude=args.amplitude,time=t,steps=steps,
                remoteRunID=publisher.id,passed=False,error=str(error),integrationWallSeconds=wall,
                meanStepMilliseconds=wall/max(steps,1)*1000,final=history[-1] if history else {},
                limits='Failed research run; timing includes the unsuccessful final solve.')
            (root/'researchRun.json').write_text(json.dumps(failed,indent=2)+'\n')
        if history and not isinstance(error,SimulationStopped):
            try:render(root)
            except Exception as render_error:
                (root/'failedRender.json').write_text(json.dumps(dict(error=str(render_error)),indent=2)+'\n')
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',choices=('stationary','advected','acoustic','kh'),required=True)
    p.add_argument('--n',type=int,default=24);p.add_argument('--shape',choices=('planar','circle'),default='planar')
    p.add_argument('--jitter',type=float,default=0.);p.add_argument('--dt',type=float,default=.002)
    p.add_argument('--time',type=float,default=.1);p.add_argument('--amplitude',type=float,default=1.e-4)
    p.add_argument('--taper',action='store_true',help='Smoothly suppress edge corrections at the compact support boundary.')
    p.add_argument('--backend',choices=('cpu','gpu-jacobi','gpu-fft','gpu-coarse'),default='cpu')
    p.add_argument('--coarse-k',type=int,default=4,help='Physical coarse Fourier frequency cutoff.')
    p.add_argument('--uniform',action='store_true',help='Full logical n-by-n initial lattice for FFT controls.')
    p.add_argument('--frame-every',type=int,default=5)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--render-only',action='store_true')
    args=p.parse_args()
    if args.backend!='cpu' and not args.taper:p.error('GPU experiment requires --taper')
    if args.backend=='gpu-fft' and not (args.uniform or args.case=='kh'):p.error('FFT mode requires --uniform or --case kh')
    if args.frame_every<1:p.error('frame-every must be positive')
    if args.coarse_k<1:p.error('coarse-k must be positive')
    if args.n<20 or args.n%4 or args.dt<=0 or args.time<=0 or args.amplitude<=0 or not 0<=args.jitter<=.2:p.error('Use n>=20 divisible by4, positive dt/time/amplitude, jitter in [0,.2]')
    if args.render_only:print(render(args.output),flush=True)
    else:raise SystemExit(run(args))
