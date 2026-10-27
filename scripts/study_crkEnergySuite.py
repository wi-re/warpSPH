"""Warp-only four-way CRK thermal-rate/compatible-correction case matrix.

Production CRK defaults are retained except the declared energy choice, common
inputs, strict one-material EOS, fresh midpoint and shared timestep sequence.
The entropy RHS is a research hook, not a production model default.
"""
import argparse
import copy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import numpy as np

from compare_spheralMaterials import save,warp_run
from compare_spheralSod1d import exact as sod_exact
from compare_spheralKH import sample as kh_sample,mixing_mode
from compare_spheralSedov import sample as sedov_sample,time_grid,exact as sedov_exact,front

POLICIES={
    'compatible':dict(compatible=True,thermal='base'),
    'raw':dict(compatible=False,thermal='base'),
    'entropy':dict(compatible=False,thermal='entropy'),
    'entropyCompatible':dict(compatible=True,thermal='entropy'),
}
CASES=dict(
    sod=dict(dim=1,nx=200,gamma=1.4,domain=[-1.5,1.5],end=.2,steps=400),
    kh=dict(dim=2,nx=64,gamma=5/3,domain=[0.,1.],end=2.,steps=1000),
    wave=dict(dim=1,nx=128,gamma=1.4,domain=[0.,1.],end=1.,steps=400),
    vortex=dict(dim=2,nx=64,gamma=1.4,domain=[-5.,5.],end=2.,steps=400),
    sedov=dict(dim=2,nx=64,gamma=5/3,domain=[-.5,.5],end=.05,steps=600),
)


def vortex_fields(x,t=0.):
    # Yee formula from the repository sampler, on a common Cartesian lattice.
    # Background advection is a Galilean transform of that stationary vortex.
    g=1.4;beta=5.;background=np.array([.5,.25])
    r=(np.asarray(x)-t*background+5)%10-5;r2=np.sum(r*r,axis=1)
    factor=beta/(2*np.pi)*np.exp((1-r2)/2)
    v=background+factor[:,None]*np.column_stack((-r[:,1],r[:,0]))
    T=1-(g-1)*beta**2/(8*g*np.pi**2)*np.exp(1-r2)
    rho=T**(1/(g-1));P=rho**g
    return dict(density=rho,pressure=P,velocity=v,energy=P/((g-1)*rho))


def wave_fields(x,t=0.):
    # Right-going acoustic eigenmode: rho0=c0=1, P0=1/gamma.
    delta=1.e-4*np.sin(2*np.pi*(np.asarray(x)[:,0]-t));rho=1+delta;P=1/1.4+delta
    return dict(density=rho,pressure=P,velocity=delta[:,None],energy=P/(.4*rho))


def sample(case):
    cfg=CASES[case];n=cfg['nx'];g=cfg['gamma'];lo,hi=cfg['domain']
    if case=='kh':return kh_sample(n)
    if case=='sedov':return sedov_sample(n)
    if case=='sod':
        x=(lo+(np.arange(3*n)+.5)/n)[:,None];reference,_=sod_exact(x[:,0],0.);dx=1/n
    elif case=='wave':
        dx=(hi-lo)/n;x=(lo+(np.arange(n)+.5)*dx)[:,None];reference=wave_fields(x)
    else:
        dx=(hi-lo)/n;axis=lo+(np.arange(n)+.5)*dx;a,b=np.meshgrid(axis,axis,indexing='xy');x=np.column_stack((a.ravel(),b.ravel()));reference=vortex_fields(x)
    rho=reference['density'];P=reference['pressure'];v=reference['velocity']
    if v.ndim==1:v=v[:,None]
    return dict(positions=x,velocities=v,densities=rho,pressures=P,internalEnergies=P/((g-1)*rho),
        supports=np.full(len(x),4*dx),masses=rho*dx**cfg['dim'],materials=np.zeros(len(x),dtype=np.int32),
        UIDs=np.arange(len(x),dtype=np.int64),kinds=np.zeros(len(x),dtype=np.int32))


def prepare(root):
    root.mkdir(parents=True,exist_ok=False);rows=[]
    for name,cfg in CASES.items():
        initial=sample(name)
        for split in (1,2):
            grid=np.repeat(time_grid(64,True)/split,split) if name=='sedov' else np.full(cfg['steps']*split,cfg['end']/(cfg['steps']*split))
            for policy,choice in POLICIES.items():
                folder=root/f'{name}-dt{split}-{policy}';folder.mkdir()
                np.savez_compressed(folder/'input.npz',**initial)
                settings=dict(cfg,case=name,policy=policy,choice=choice,gammaVariant=f'dt{split}-{policy}',
                    dt=float(grid[0]),timeGrid=grid.tolist(),exactSteps=len(grid),frameEvery=max(1,len(grid)//40),
                    inputPath=str((folder/'input.npz').resolve()),kernel='B8',precision='float64',
                    comparisonPolicy='Warp base CRK summation/legacy/current gradient, default AV/limiter; '+policy)
                save(folder/'settings.json',settings);rows.append(folder.name)
    sources={};(root/'sources').mkdir()
    for name in (__file__,'scripts/crkThermalRate.py','scripts/compare_spheralMaterials.py','scripts/compare_spheralKH.py',
        'scripts/compare_spheralSod1d.py','scripts/compare_spheralSedov.py','src/warpSPH/modules/crk/dudt.py',
        'src/warpSPH/configurations/crkSPH.py','src/warpSPH/schemes/compressible/crkSPH.py','src/warpSPH/systems/compSPH.py'):
        p=Path(name);key=hashlib.sha256(p.read_bytes()).hexdigest();shutil.copy2(p,root/'sources'/(p.stem+'_'+key[:12]+p.suffix));sources[str(p)]=key
    save(root/'provenance.json',dict(rows=rows,sources=sources,policies=POLICIES,
        scope='Warp only; repo CRK density/volume/AV defaults; B8 n_h4 and fresh midpoint; one strict ideal-gas material. No Spheral stage-policy overlays. Both timestep levels use identical initial samples.'))


def context(settings):
    from warpSPHBootstrap import bootstrap
    bootstrap(precision='float64')
    import torch
    from warpSPH.runner import Case,RunConfiguration
    from warpSPH.cases.compressible.common import COMPRESSIBLE_DEFAULTS,COMPRESSIBLE_PARAMS,configureCompressible
    from warpSPH.cases.common.parameters import COMPRESSIBLE_CASE_PARAMETERS
    from warpSPH.cases.common.plotting import Field,particlePlot
    from warpSPHCore import DomainDescription
    def configure(c):
        configureCompressible(c);c.config.domain.min[:]=settings['domain'][0];c.config.domain.max[:]=settings['domain'][1]
    def build(c):
        values={k:torch.as_tensor(v,device=c.device).clone() for k,v in dict(np.load(settings['inputPath'])).items()};rho=values['densities']
        values.update(UIDcounter=len(rho),alphas=torch.ones_like(rho),alpha0s=torch.ones_like(rho),divergence=torch.zeros_like(rho))
        state=c.SimulationState(**values);state.totalEnergies=state.masses*(state.internalEnergies+.5*state.velocities.square().sum(-1))
        return c.SimulationSystem(state=state,domain=c.config.domain)
    def post(c,s,step):c.scratch['energySuiteNext']=step+1
    def timestep(c,s):return settings['timeGrid'][min(c.scratch.get('energySuiteNext',0),settings['exactSteps']-1)]
    fields=[Field('densities','Density',colorMap='viridis'),Field('internalEnergies','Specific thermal energy',colorMap='inferno')]
    setup,update=particlePlot(fields)
    def view(c,s):
        if settings['dim']==2:return c,s
        out=copy.copy(s);out.state=copy.copy(s.state);out.state.positions=torch.cat((s.state.positions,torch.zeros_like(s.state.positions)),dim=1)
        cfg=copy.copy(c.config);lo,hi=settings['domain'];height=.04*(hi-lo)
        cfg.domain=DomainDescription(torch.tensor([lo,-height],device=c.device,dtype=c.dtype),torch.tensor([hi,height],device=c.device,dtype=c.dtype),torch.zeros(2,device=c.device,dtype=torch.bool),2)
        return replace(c,config=cfg),out
    def draw(c,s):return setup(*view(c,s))
    def refresh(c,s,p,step):return update(*view(c,s),p,step)
    draw.usesWarp=refresh.usesWarp=False
    case=Case(name='crkEnergySuite_'+settings['case'],scheme='CRKSPH',buildSystem=build,configureScheme=configure,
        defaults=dict(COMPRESSIBLE_DEFAULTS),params=dict(COMPRESSIBLE_PARAMS,gamma=settings['gamma']),parameterDefinitions=COMPRESSIBLE_CASE_PARAMETERS,
        supportedSchemes=('CRKSPH',),postStep=post,timestep=timestep,setupPlot=draw,updatePlot=refresh,
        extraFields=('internalEnergies','supports','pressures'))
    sim=dict(nx=settings['nx'],dim=settings['dim'],L=(settings['domain'][1]-settings['domain'][0])/2,
        tLimit=settings['end'],nSteps=settings['exactSteps'],dt=settings['dt'],adaptiveDt=False,periodic=True,
        kernel='B8',n_h=4.,precision='float64',execution='eager',integrationScheme='rungeKutta2',supportMode='KernelMeanSymmetric',cflFactor=.25)
    params=dict(gamma=settings['gamma'],viscositySwitch='NoneSwitch',adaptiveSupportScheme='Owen')
    solver=dict(compatibleEnergy=settings['choice']['compatible'],materials=[dict(name='gas',eos=dict(type='idealGas',gamma=settings['gamma']))])
    return case,RunConfiguration(case.name,simulation=sim,params=params,solver=solver).prepare(case),sim,params,solver


class Measures:
    def __init__(self,settings,initial):
        self.settings=settings;self.initial=initial;self.m=initial['masses'];self.mass=self.m.sum()
        self.E=np.dot(self.m,initial['internalEnergies']+.5*np.sum(initial['velocities']**2,axis=1));self.p=np.sum(self.m[:,None]*initial['velocities'],axis=0)
        self.s0=np.log(initial['pressures']/initial['densities']**settings['gamma'])
    def __call__(self,a,t):
        m,rho,u,v,x=(np.asarray(a[k],dtype=float) for k in ('masses','densities','internalEnergies','velocities','positions'));P=a['pressures'];g=self.settings['gamma'];V=m/rho
        with np.errstate(invalid='ignore',divide='ignore'):ds=np.log(P/rho**g)-self.s0
        kinetic=.5*np.dot(m,np.sum(v*v,axis=1))
        out=dict(time=float(t),relativeEnergyDrift=float((np.dot(m,u)+kinetic)/self.E-1),momentumError=float(np.linalg.norm(np.sum(m[:,None]*v,axis=0)-self.p)/self.mass),
            materialMassError=float(abs(m.sum()/self.mass-1)),minDensity=float(rho.min()),minEnergy=float(u.min()),minPressure=float(P.min()),
            maxVelocity=float(np.linalg.norm(v,axis=1).max()),kineticEnergy=float(kinetic),entropyChangePerMass=float(np.dot(m,ds)/self.mass),
            entropyL1PerMass=float(np.dot(m,abs(ds))/self.mass),negativeEntropyFraction=float(m[ds<-1.e-8].sum()/self.mass))
        case=self.settings['case']
        if case=='kh':
            out.update(modeAmplitude=mixing_mode(a),transverseKineticEnergy=float(.5*np.dot(m,v[:,1]**2)),pressureRMS=float(np.sqrt(np.dot(V,(P-2.5)**2)/V.sum())))
        if case=='vortex':
            reference=vortex_fields(x,t);mask=np.ones(len(m),dtype=bool)
            out['angularMomentumAboutAdvectedCentre']=float(np.dot(m,(x[:,0]-.5*t)*(v[:,1]-.25)-(x[:,1]-.25*t)*(v[:,0]-.5)))
        elif case=='wave':reference=wave_fields(x,t);mask=np.ones(len(m),dtype=bool)
        elif case=='sod':reference,_=sod_exact(x[:,0],t);mask=abs(x[:,0])<.5
        elif case=='sedov':
            r=np.linalg.norm(x,axis=1);reference,R=sedov_exact(r,t);mask=r<.45
            out['density2Front']=front(a,self.settings['nx']);out['exactShockRadius']=R
        if case!='kh':
            velocity=v
            if case=='sedov':velocity=np.sum(x*v,axis=1)/np.maximum(np.linalg.norm(x,axis=1),1.e-30)
            elif case=='sod':velocity=v[:,0]
            for name,field in [('density',rho),('pressure',P),('velocity',velocity),('energy',u)]:
                error=abs(field-reference[name]);error=np.linalg.norm(error,axis=1) if error.ndim>1 else error
                out[name+'L1']=float(np.dot(V[mask],error[mask])/V[mask].sum())
            if case=='wave':
                phase=2*np.pi*(x[:,0]-t);z=2*np.sum(V*v[:,0]*np.exp(-1j*phase))/V.sum()
                out.update(waveVelocityAmplitude=float(abs(z)),waveVelocityPhase=float(np.angle(z)+np.pi/2))
        return out


def summarize(root,solver,settings,history,initial,final,wall,steps,remote,extra):
    finite=all(np.isfinite(v).all() for v in final.values())
    gates=dict(completed=abs(history[-1]['time']-settings['end'])<1.e-10,finite=finite,
        positive=all(r['minEnergy']>0 and r['minDensity']>0 and r['minPressure']>0 for r in history),
        mass=max(abs(r['materialMassError']) for r in history)<1.e-12)
    # Energy/accuracy are measured outcomes, never used to hide a completed run.
    extras=dict(extra);extras['baseHarnessPolicyLabel']=extras.pop('policy',None)
    row=dict(settings=settings,solver=solver,remoteRunID=remote,steps=int(steps),wallSeconds=wall,meanStepMilliseconds=1000*wall/max(steps,1),
        time=history[-1]['time'],final=history[-1],gates={k:bool(v) for k,v in gates.items()},passed=bool(all(gates.values())),
        maxAbsoluteEnergyDrift=max(abs(r['relativeEnergyDrift']) for r in history),**extras)
    for name,a in [('initialState',initial),('finalState',final)]:np.savez_compressed(root/(name+'.npz'),**a)
    # A failed EOS can leave invalid entropy diagnostics. Preserve as null.
    clean=[{k:v if np.isfinite(v) else None for k,v in r.items()} for r in history]
    row['final']=clean[-1]
    save(root/'diagnosticHistory.json',clean);save(root/'comparisonRun.json',row)
    print('ENERGY SUITE RESULT',settings['case'],settings['policy'],'passed',row['passed'],'wall',round(wall,3),'ms/step',round(row['meanStepMilliseconds'],3),'final',clean[-1],flush=True)
    return row


def install_audit(settings):
    from warpSPHBootstrap import bootstrap
    bootstrap(precision='float64')
    import importlib
    import torch
    from crkThermalRate import thermal_entropy_rate
    module=importlib.import_module('warpSPH.schemes.compressible.crkSPH');system=importlib.import_module('warpSPH.systems.compSPH')
    from warpSPH.modules.eos import materials as eos
    audit=dict(rhs=0,compatibleAccumulations=0,compatibleFinalizations=0,eosCalls=0,negativeEOSCalls=0,worstEnergy=0.,stagePower=[])
    original_acc,original_du,original_accumulate,original_finalize,original_eos=module.computeCRKAccel,module.computeCRKdudt,module.accumulateStage,system.energyIncrement,eos.evaluateEOS
    last={};choice=settings['choice']
    expected=settings.get('auditBaseline',dict(density='legacy',Cl=1.,Cq=1.,viscosityKernelScale=0.))
    def acceleration(state,config,policy,*args):
        assert policy.compatibleEnergy==choice['compatible'] and policy.crkEvolution=='summation' and policy.crkDensityPolicy==expected['density']
        assert policy.crkEnergyPartition=='native' and policy.crkEnergyPredictor=='raw' and policy.crkViscosityGradient=='current'
        assert not policy.crkViscosityParams.oneSidedWork and policy.diffusionParams.C_l==expected['Cl'] and policy.diffusionParams.C_q==expected['Cq']
        assert policy.crkViscosityParams.viscosityKernelScale==expected['viscosityKernelScale']
        result=original_acc(state,config,policy,*args);last['acc']=result[0];last['pairs']=result[1]+result[2]
        return result
    def derivative(state,*args):
        rate=original_du(state,*args);audit['rhs']+=1
        if choice['thermal']=='entropy':rate=thermal_entropy_rate(args[2],last['pairs'],state.pressures/state.densities.pow(settings.get('thermalGamma',settings['gamma'])),state.velocities)
        mechanical=state.masses*torch.sum(state.velocities*last['acc'],dim=1);thermal=state.masses*rate
        values=torch.stack(((mechanical+thermal).sum(),(mechanical.abs()+thermal.abs()).sum(),state.internalEnergies.min())).detach().cpu().tolist()
        audit['stagePower'].append([v if np.isfinite(v) else None for v in values])
        return rate
    def accumulate(*args,**kwargs):
        assert choice['compatible'];audit['compatibleAccumulations']+=1;return original_accumulate(*args,**kwargs)
    def finalize(*args,**kwargs):
        assert choice['compatible'];audit['compatibleFinalizations']+=1;return original_finalize(*args,**kwargs)
    def observed(*args,**kwargs):
        result=original_eos(*args,**kwargs);audit['eosCalls']+=1;u=kwargs.get('u');minimum=float(u.min()) if u is not None else 0.
        if minimum<0:audit['negativeEOSCalls']+=1;audit['worstEnergy']=min(audit['worstEnergy'],minimum)
        return result
    module.computeCRKAccel,module.computeCRKdudt,module.accumulateStage,system.energyIncrement,eos.evaluateEOS=acceleration,derivative,accumulate,finalize,observed
    return audit


def run_row(args):
    sys.path.insert(0,str(args.publisher_path));settings=json.loads((args.output/'settings.json').read_text());audit=install_audit(settings)
    def initialize(c,system,a,settings):
        system.state.crkVolumes=system.state.crkMassDensities=None
        c.scratch['_titleProvenance']='Warp base CRK | '+settings['case']+' | '+settings['policy']+' | '+settings['gammaVariant']
        save(args.output/'resolvedCRK.json',dict(compatibleEnergy=c.schemeConfig.compatibleEnergy,crkEvolution=c.schemeConfig.crkEvolution,
            crkDensityPolicy=c.schemeConfig.crkDensityPolicy,crkEnergyPartition=c.schemeConfig.crkEnergyPartition,crkEnergyPredictor=c.schemeConfig.crkEnergyPredictor,
            oneSidedWork=bool(c.schemeConfig.crkViscosityParams.oneSidedWork),Cl=float(c.schemeConfig.diffusionParams.C_l),Cq=float(c.schemeConfig.diffusionParams.C_q)))
    try:return warp_run(args,settings,context,Measures,summarize,'CRK energy suite',initialize_factory=initialize)
    finally:save(args.output/'energyAudit.json',audit)


def render_recording(args):
    """Offline rendering only, matching every stored trajectory frame."""
    sys.path.insert(0,str(args.publisher_path))
    from warpsph_remote.sources import HDF5Source
    from types import SimpleNamespace as NS
    settings=json.loads((args.output/'settings.json').read_text());data=args.output/'warp'
    folder=next(p.parent for p in data.glob('*/trajectory.h5'));source=HDF5Source(folder/'trajectory.h5')
    case,ctx,*_=context(settings)
    import torch
    from warpSPH.runner.media import encodeFrames
    from warpSPH.runner.display import closeWindow
    images=folder/'recordingImages';images.mkdir(exist_ok=False);ctx.imagePath=str(images);ctx.exportPath=str(folder)
    ctx.scratch['_titleProvenance']='Saved trajectory playback | Warp base CRK | '+settings['case']+' | '+settings['policy']
    plotter=None
    for index,t in enumerate(source.times):
        frame={k:torch.as_tensor(v,device=ctx.device) for k,v in source.read(index).items()}
        state=NS(state=NS(**frame),t=float(t))
        if plotter is None:plotter=case.setupPlot(ctx,state)
        else:case.updatePlot(ctx,state,plotter,index)
    if hasattr(plotter,'drain'):plotter.drain()
    closeWindow(plotter)
    original=folder/'output.mp4';backup=folder/'outputOriginal.mp4'
    if backup.exists():raise ValueError('Original video backup already exists')
    original.rename(backup);encodeFrames(str(images),str(folder))
    save(data/'videoRebuild.json',dict(source=str(source.path),frames=len(source.times),originalVideo=str(backup),
        video=str(original),physicsRerun=False,reason='Stored export stride and live plot stride differ; video now follows every saved frame.'))
    print('RENDERED SAVED RECORDING',args.output.name,len(source.times),'frames',flush=True)


def repair_recordings(root,selected=None):
    """Keep videos synchronized with the actual recorder, without new physics."""
    sys.path.insert(0,'/home/lu26029/dev/warpSPHRemote/python')
    from warpsph_remote.sources import HDF5Source
    records=[]
    for result in sorted(root.glob('*/warp/comparisonRun.json')):
        folder=result.parent.parent
        if selected and not any(folder.name.startswith(s) for s in selected):continue
        run=json.loads(result.read_text())
        if not run['passed']:continue
        export=Path(run['exportPath']);source=HDF5Source(export/'trajectory.h5')
        count=int(json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=nb_frames','-of','json',str(export/'output.mp4')]))['streams'][0]['nb_frames'])
        if count==len(source.times):continue
        print('RENDER SAVED FRAMES',folder.name,count,'->',len(source.times),flush=True)
        command=[sys.executable,'-u',__file__,'render','--output',str(folder)];start=time.perf_counter()
        with (root/(folder.name+'-render.log')).open('w')as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,env=dict(os.environ,MPLCONFIGDIR='/tmp/warpsph-energy-suite-mpl'))
        records.append(dict(row=folder.name,oldFrames=count,newFrames=len(source.times),wallSeconds=time.perf_counter()-start,physicsRerun=False))
    save(root/'recordingRepairs.json',records)


def study(root,selected=None):
    provenance=json.loads((root/'provenance.json').read_text());target=root/'study.json'
    manifest=json.loads(target.read_text()) if target.exists() else dict(stages=[])
    manifest.update(status='running',pid=os.getpid());env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',MPLCONFIGDIR='/tmp/warpsph-energy-suite-mpl',PYTHONFAULTHANDLER='1')
    try:
        for name in provenance['rows']:
            if selected and not any(name.startswith(x) for x in selected):continue
            folder=root/name;data=folder/'warp'
            if (data/'comparisonRun.json').exists() or (data/'failure.json').exists():continue
            command=[sys.executable,'-u',__file__,'row','--output',str(folder)]
            logpath=root/(name+'.log');stage=dict(name=name,command=command,status='running',log=str(logpath));manifest['stages'].append(stage);save(target,manifest)
            print('START',name,flush=True);start=time.perf_counter();last=None
            with logpath.open('w') as log:
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env);stage['pid']=child.pid;save(target,manifest)
                try:
                    while child.poll() is None:
                        if (root/'STOP').exists():child.send_signal(signal.SIGINT)
                        try:child.wait(timeout=5)
                        except subprocess.TimeoutExpired:pass
                        lines=logpath.read_text(errors='replace').replace('\r','\n').splitlines()
                        latest=next((s for s in reversed(lines) if any(v in s for v in ('| t=','ENERGY SUITE RESULT','Remote run registered:','Traceback'))),'')
                        if latest and latest!=last:print(name+': '+latest[:280],flush=True);last=latest
                    code=child.wait()
                except BaseException:child.send_signal(signal.SIGINT);child.wait();raise
            failure=data/'failure.json';numerical=failure.exists() and 'finite-state/velocity watchdog' in json.loads(failure.read_text())['error']
            stage.update(returncode=code,status='numerical_failure' if numerical else ('completed' if code==0 else 'failed'),wallSeconds=time.perf_counter()-start);save(target,manifest)
            print('END',name,stage['status'],flush=True)
            if code and not numerical:raise RuntimeError('Unexpected suite failure '+name+'; '+str(logpath))
            if numerical:
                from warpSPH.runner.media import encodeFrames
                for images in data.glob('*/images'):
                    if not (images.parent/'output.mp4').exists():encodeFrames(str(images),str(images.parent))
        repair_recordings(root,selected)
        manifest['status']='completed'
    except BaseException as error:manifest.update(status='failed',error=repr(error));raise
    finally:save(target,manifest)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('prepare','study','row','render','repair'));p.add_argument('--output',type=Path,required=True)
    p.add_argument('--select',nargs='+');p.add_argument('--publisher-path',type=Path,default=Path('/home/lu26029/dev/warpSPHRemote/python'))
    args=p.parse_args();args.output=args.output.resolve()
    if args.mode=='prepare':prepare(args.output)
    elif args.mode=='study':study(args.output,args.select)
    elif args.mode=='render':render_recording(args)
    elif args.mode=='repair':repair_recordings(args.output,args.select)
    else:raise SystemExit(run_row(args))
