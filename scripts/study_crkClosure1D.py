"""Daemon-recorded CPU research trajectories for the periodic 1D closure solve.

This is a separate experimental pressure/volume-work model, not production
warpSPH or Spheral CRK. No AV/shocks; use smooth acoustics or exact contacts.
Video uses vispy after integration. Every physical invocation registers with
the resident daemon, supports pause/stop, and records its trajectory.
"""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np

from crkClosure1D import close_edges,pressure_rhs
from probe_crkSupportClosure import cubic,corrected


def edge_geometry(x,V):
    R=4*V
    delta=x[:,None]-x[None,:];delta-=2*np.round(delta/2)
    W,dW=cubic(delta,R[None,:]);_,G=corrected(delta,W,dW,V)
    i,j=np.where(np.triu((abs(delta)<np.maximum(R[:,None],R[None,:])) & (abs(delta)>0),1))
    q0=.5*V[i]*V[j]*(G[i,j]-G[j,i])
    return i,j,q0,R


def render(root,case):
    from warpSPHBootstrap import bootstrap
    bootstrap(precision='float32')
    import h5py
    import torch
    from warpSPH.runner import RunConfiguration
    from warpSPH.cases.compressible.twoGasSod import twoGasSodCase,plotView
    from warpSPH.cases.common.plotting import Field,particlePlot
    from warpSPH.runner.display import closeWindow
    from warpSPH.runner.media import encodeFrames
    ctx=RunConfiguration('twoGasSod',simulation=dict(plot=True,video=True,show=False,plotBackend='vispy'),
        params=dict(pressureRight=1.)).prepare(twoGasSodCase)
    ctx.exportPath=str(root);ctx.imagePath=str(root/'images');Path(ctx.imagePath).mkdir(exist_ok=True)
    settings=json.loads((root/'researchRun.json').read_text())
    ctx.config.dt=settings['dt']
    ctx.scratch['_titleProvenance']='Experimental 1D cycle closure | coupled pressure/volume work | midpoint | no AV'
    delta=2*settings.get('amplitude',1.e-4) if case=='acoustic' else .01
    setup,update=particlePlot([Field('densities','Density',vMin=1-delta if case=='acoustic' else .2,vMax=1+delta),
        Field('pressures','Pressure',vMin=1-delta,vMax=1+delta),
        Field('velocities','Velocity',mapping='x',vMin=-delta if case=='acoustic' else -.01,
              vMax=delta if case=='acoustic' else .31)],figsize=(12,3))
    plotter=None
    with h5py.File(root/'trajectory.h5','r') as f:
        for number,key in enumerate(sorted(f['states'])):
            g=f['states'][key]
            state=NS(**{k:torch.as_tensor(v[()],device=ctx.device) for k,v in g.items()})
            system=NS(state=state,t=float(g.attrs['time']))
            c,view=plotView(ctx,system)
            if plotter is None:plotter=setup(c,view)
            else:update(c,view,plotter,number)
    if plotter:
        if hasattr(plotter,'drain'):plotter.drain()
        closeWindow(plotter)
    return encodeFrames(ctx.imagePath,str(root))


def run(args):
    import h5py
    from warpsph_remote.publisher import SimulationPublisher,SimulationStopped
    root=args.output;root.mkdir(parents=True,exist_ok=False)
    n=args.n_left;nr=n//4
    x0=np.r_[-1+(np.arange(n)+.5)/n,(np.arange(nr)+.5)/nr]
    V0=np.r_[np.full(n,1/n),np.full(nr,1/nr)]
    rho0=np.ones_like(x0) if args.case=='acoustic' else np.where(x0<0,1.,.25)
    gamma=np.full_like(x0,1.4) if args.case=='acoustic' else np.where(x0<0,1.4,5/3)
    P0=1+args.amplitude*np.sin(np.pi*x0) if args.case=='acoustic' else np.ones_like(x0)
    boost=.3 if args.case=='advected' else 0.
    mass=V0*rho0
    # Rows are x, v, V, u. The EOS is always evaluated from rho=m/V.
    y=np.stack((x0,np.full_like(x0,boost),V0,P0/((gamma-1)*rho0)))
    initial_energy=float(np.sum(mass*(y[3]+.5*y[1]**2)))
    entropy0=P0/rho0**gamma
    UIDs=np.arange(len(x0),dtype=np.int64);ids=(x0>=0).astype(np.int32)
    publisher=SimulationPublisher('Closure research '+args.case+f' n{n} dt{args.dt:g}'+
        (f' A{args.amplitude:g}' if args.case=='acoustic' else ''),
        measurements=True,diagnostic_interval=.001)
    print('Remote run registered:',publisher.id,flush=True)
    file=h5py.File(root/'trajectory.h5','w')
    ctx=NS(spec=NS(store=True,storeMode='trajectory',tLimit=args.time),scratch={'trajectoryFile':file},
        exportPath=str(root.resolve()),scheme='Experimental 1D cycle closure',
        config=NS(dt=args.dt,domain=NS(min=np.array([-1.]),max=np.array([1.]),periodic=np.array([True])),
                  kernel='CubicSpline',integrationScheme='explicitMidpoint'))
    history=[];steps=0;t=0.;last_frame=-1;geometry_seconds=0.;closure_seconds=0.;rhs_calls=0
    def arrays():
        rho=mass/y[2];P=(gamma-1)*rho*y[3]
        return dict(positions=y[0,:,None],velocities=y[1,:,None],densities=rho,pressures=P,
            internalEnergies=y[3],supports=4*y[2],volumes=y[2],masses=mass,materials=ids,
            UIDs=UIDs,kinds=np.zeros_like(ids))
    def rhs(state):
        nonlocal geometry_seconds,closure_seconds,rhs_calls
        if not np.isfinite(state).all() or np.any(state[2:]<=0):
            raise RuntimeError('Nonfinite or nonpositive trial state')
        x,v,V,u=state;rho=mass/V;P=(gamma-1)*rho*u
        start=time.perf_counter();i,j,q0,R=edge_geometry(x,V);geometry_seconds+=time.perf_counter()-start
        start=time.perf_counter();q=close_edges(x,V,i,j,q0,2.);closure_seconds+=time.perf_counter()-start;rhs_calls+=1
        acceleration,dV,du=pressure_rhs(V,mass,P,v,i,j,q)
        return np.stack((v,acceleration,dV,du))
    def record(force=False):
        nonlocal last_frame
        a=arrays();P=a['pressures'];rho=a['densities']
        energy=np.sum(mass*(y[3]+.5*y[1]**2))
        row=dict(step=steps,time=t,pressureError=float(np.max(abs(P-1))),
            maxPeculiarVelocity=float(np.max(abs(y[1]-boost))),
            relativeEnergyDrift=float(energy/initial_energy-1),
            volumeError=float(abs(y[2].sum()/V0.sum()-1)),
            momentumError=float(abs(np.dot(mass,y[1]-boost))),
            entropyRelativeError=float(np.max(abs((P/rho**gamma)/entropy0-1))),
            minVolume=float(y[2].min()),minEnergy=float(y[3].min()))
        if args.case!='acoustic':row['densityRelativeError']=float(np.max(abs(rho/rho0-1)))
        else:
            c=np.sqrt(1.4)
            exactP=1+args.amplitude*np.sin(np.pi*y[0])*np.cos(c*np.pi*t)
            exactV=-args.amplitude/c*np.cos(np.pi*y[0])*np.sin(c*np.pi*t)
            row['pressureWaveL1']=float(np.sum(y[2]*abs(P-exactP))/y[2].sum())
            row['velocityWaveL1']=float(np.sum(y[2]*abs(y[1]-exactV))/y[2].sum())
        history.append(row)
        if steps%25==0 or force:
            if last_frame!=steps:
                g=file.require_group('states').create_group(f'frame_{steps:06d}');g.attrs['time']=t
                for k,v in a.items():g.create_dataset(k,data=v)
                file.flush();last_frame=steps
        publisher.post_step(ctx,NS(state=NS(**a),t=t),steps)
        if steps%25==0 or force:
            print(f'step={steps} | t={t:.7f} | dt={ctx.config.dt:.3g} | Perror={row["pressureError"]:.3g} | speed={row["maxPeculiarVelocity"]:.3g} | energyDrift={row["relativeEnergyDrift"]:.3g}',flush=True)
        return row
    started=time.perf_counter()
    try:
        record()
        while t<args.time-1.e-14:
            dt=min(args.dt,args.time-t)
            ctx.config.dt=dt
            k1=rhs(y);mid=y+.5*dt*k1;mid[0]=(mid[0]+1)%2-1
            new=y+dt*rhs(mid);new[0]=(new[0]+1)%2-1
            y=new;steps+=1;t+=dt
            if not np.isfinite(y).all() or np.any(y[2:]<=0):raise RuntimeError('Nonpositive or nonfinite completed state')
            record()
            if steps>100000:raise RuntimeError('Step watchdog exceeded')
        record(force=True)
        integration_wall=time.perf_counter()-started
        a=arrays();publisher.finish(ctx,NS(state=NS(**a),t=t),steps)
        np.savez_compressed(root/'finalState.npz',**a,gamma=gamma,initialPositions=x0,initialVolumes=V0)
        file.close()
        (root/'diagnosticHistory.json').write_text(json.dumps(history,indent=2)+'\n')
        maximum=lambda name:max(abs(r[name]) for r in history)
        gates=dict(completed=abs(t-args.time)<1.e-12,energy=maximum('relativeEnergyDrift')<1.e-6,
            volume=maximum('volumeError')<1.e-10,momentum=maximum('momentumError')<1.e-10,
            entropy=maximum('entropyRelativeError')<1.e-5)
        if args.case!='acoustic':gates.update(pressure=maximum('pressureError')<1.e-8,
            quiescent=maximum('maxPeculiarVelocity')<1.e-8,density=maximum('densityRelativeError')<1.e-8)
        else:gates.update(pressureWave=maximum('pressureWaveL1')<.1*args.amplitude,
                          velocityWave=maximum('velocityWaveL1')<.1*args.amplitude)
        summary=dict(case=args.case,nLeft=n,dt=args.dt,amplitude=args.amplitude,time=t,steps=steps,remoteRunID=publisher.id,
            integrationWallSeconds=integration_wall,meanStepMilliseconds=integration_wall/steps*1000,
            geometrySeconds=geometry_seconds,closureSeconds=closure_seconds,rhsCalls=rhs_calls,
            final=history[-1],maxima={k:maximum(k) for k in history[-1] if k not in ('step','time')},
            gates={k:bool(v) for k,v in gates.items()},passed=all(gates.values()),
            limits='Experimental NumPy 1D pressure/volume-work model, not production CRKSPH. No viscosity, shock or multi-D qualification. Acoustic target is linearized and has O(amplitude^2) error.')
        (root/'researchRun.json').write_text(json.dumps(summary,indent=2)+'\n')
        publisher.update(message='Rendering recorded frames with vispy')
        video=render(root,args.case)
        summary.update(video=video,totalWallSeconds=time.perf_counter()-started)
        (root/'researchRun.json').write_text(json.dumps(summary,indent=2)+'\n')
        publisher.close('completed' if summary['passed'] else 'failed','' if summary['passed'] else 'Research qualification gate failed')
        print(json.dumps(summary,indent=2),flush=True)
        return 0 if summary['passed'] else 1
    except BaseException as error:
        if file.id.valid:file.close()
        publisher.close('stopped' if isinstance(error,SimulationStopped) else 'failed',str(error))
        (root/'failure.json').write_text(json.dumps(dict(step=steps,time=t,error=str(error)),indent=2)+'\n')
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',choices=('stationary','advected','acoustic'),required=True)
    p.add_argument('--n-left',type=int,default=128)
    p.add_argument('--dt',type=float,default=1.e-4);p.add_argument('--time',type=float,default=.1)
    p.add_argument('--amplitude',type=float,default=1.e-4);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--render-only',action='store_true',help='Repair/render a completed research trajectory without advancing it.')
    a=p.parse_args()
    if a.n_left<32 or a.n_left%4 or a.dt<=0 or a.time<=0:p.error('Use n-left >=32 divisible by4, positive dt/time')
    if a.render_only:
        print(render(a.output,a.case),flush=True)
    else:
        raise SystemExit(run(a))
