"""Single physical material, classical 1D Sod: exact solution and two CRK solvers."""
import argparse
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
from types import SimpleNamespace as NS

import numpy as np
from compare_spheralMaterials import FIELDS,save,warp_run,spheral_run


def exact(x,t):
    g=1.4;cl=np.sqrt(g);cr=np.sqrt(g*.1/.125)
    def f(p,rho,P,c):
        if p>P:return (p-P)*np.sqrt(2/((g+1)*rho)/(p+(g-1)/(g+1)*P))
        return 2*c/(g-1)*((p/P)**((g-1)/(2*g))-1)
    lo,hi=.1,1.
    for _ in range(80):
        p=.5*(lo+hi)
        if f(p,1.,1.,cl)+f(p,.125,.1,cr)>0:hi=p
        else:lo=p
    p=.5*(lo+hi);v=.5*(f(p,.125,.1,cr)-f(p,1.,1.,cl));cstar=cl*p**((g-1)/(2*g))
    mu=(g-1)/(g+1);rhol=p**(1/g);rhor=.125*(p/.1+mu)/(mu*p/.1+1)
    speed=v/(1-.125/rhor)
    x=np.asarray(x);xi=x/max(t,1.e-30);ratio=np.maximum((2*cl-(g-1)*xi)/(g+1)/cl,1.e-15)
    rho=np.where(xi<-cl,1.,np.where(xi<v-cstar,ratio**(2/(g-1)),np.where(xi<v,rhol,np.where(xi<speed,rhor,.125))))
    P=np.where(xi<-cl,1.,np.where(xi<v-cstar,ratio**(2*g/(g-1)),np.where(xi<speed,p,.1)))
    vel=np.where(xi<-cl,0.,np.where(xi<v-cstar,2/(g+1)*(cl+xi),np.where(xi<speed,v,0.)))
    if t<=0:rho=np.where(x<0,1.,.125);P=np.where(x<0,1.,.1);vel=np.zeros_like(x)
    return dict(density=rho,pressure=P,velocity=vel,energy=P/((g-1)*rho)),dict(pressure=p,velocity=v,densityLeft=rhol,densityRight=rhor,shockSpeed=speed,fanHead=-cl,fanTail=v-cstar)


class Measures:
    def __init__(self,settings,initial):
        self.initial=initial;self.mass=initial['masses'].sum();self.E=np.sum(initial['masses']*(initial['internalEnergies']+.5*initial['velocities'][:,0]**2))
    def __call__(self,a,t):
        m,rho,u,v=(np.asarray(a[k],dtype=float) for k in ('masses','densities','internalEnergies','velocities'));v=v[:,0]
        out=dict(time=float(t),relativeEnergyDrift=float(np.sum(m*(u+.5*v*v))/self.E-1),
            momentumError=float(abs(np.dot(m,v))/self.mass),materialMassError=float(abs(m.sum()/self.mass-1)),
            minDensity=float(rho.min()),minEnergy=float(u.min()),minPressure=float(a['pressures'].min()),maxVelocity=float(abs(v).max()))
        x=a['positions'][:,0];mask=abs(x)<.5;reference,_=exact(x[mask],t);weights=m[mask]/rho[mask]
        for name,values in [('density',rho),('pressure',a['pressures']),('velocity',v),('energy',u)]:
            out[name+'L1']=float(np.dot(weights,abs(values[mask]-reference[name]))/weights.sum())
        return out


def summarize(root,solver,settings,history,initial,final,wall,steps,remote,extra):
    maxima={k:max(abs(r[k]) for r in history) for k in history[0] if k!='time'}
    gates=dict(completed=abs(history[-1]['time']-settings['end'])<1.01*settings['dt'],
        positive=all(r['minEnergy']>0 and r['minDensity']>0 for r in history),
        finite=all(np.isfinite(v).all() for v in final.values()),energy=maxima['relativeEnergyDrift']<1.e-10,
        momentum=maxima['momentumError']<1.e-10,mass=maxima['materialMassError']<1.e-12,
        singleMaterial=np.all(final['materials']==0))
    extras=dict(extra);extras['baseHarnessPolicyLabel']=extras.pop('policy',None)
    row=dict(solver=solver,settings=settings,remoteRunID=remote,steps=int(steps),wallSeconds=wall,meanStepMilliseconds=1000*wall/max(steps,1),
        time=history[-1]['time'],final=history[-1],maxima=maxima,gates={k:bool(v) for k,v in gates.items()},passed=bool(all(gates.values())),
        policy='One material, classical Sod, fresh CRK Q/RK2, Eq75 volume per RHS, Eq76 density each step/continuity stages, neighbour-Q raw predictor and entropy final work.',**extras)
    for name,a in [('initialState',initial),('finalState',final)]:np.savez_compressed(root/(name+'.npz'),**a)
    save(root/'diagnosticHistory.json',history);save(root/'comparisonRun.json',row)
    print('SOD RESULT',solver,settings['gammaVariant'],'passed',row['passed'],'wall',round(wall,3),'ms/step',round(row['meanStepMilliseconds'],3),'errors',history[-1],flush=True)
    return row


def context(settings):
    from warpSPHBootstrap import bootstrap
    bootstrap(precision='float64')
    import torch
    from warpSPH.runner import RunConfiguration
    from warpSPH.cases.compressible.twoGasSod import twoGasSodCase
    from warpSPH.cases.compressible.common import configureCompressible
    from warpSPH.configurations import MaterialConfig,EOSConfig
    def configure(c):
        configureCompressible(c);c.schemeConfig.materials=[MaterialConfig('gas',EOSConfig(gamma=1.4))]
        c.config.domain.min[:]=-1.5;c.config.domain.max[:]=1.5
    def build(c):
        a=dict(np.load(Path(settings['inputPath'])))
        values={k:torch.as_tensor(v,device=c.device).clone() for k,v in a.items()}
        rho=values['densities'];values.update(UIDcounter=len(rho),alphas=torch.ones_like(rho),alpha0s=torch.ones_like(rho),divergence=torch.zeros_like(rho))
        state=c.SimulationState(**values);state.totalEnergies=state.masses*state.internalEnergies
        return c.SimulationSystem(state=state,domain=c.config.domain)
    case=replace(twoGasSodCase,name='singleMaterialSodComparison',configureScheme=configure,buildSystem=build)
    sim=dict(nx=settings['nx'],dim=1,L=1.5,tLimit=settings['end'],nSteps=round(settings['end']/settings['dt']),dt=settings['dt'],adaptiveDt=False,
        kernel='B8',n_h=4.,precision='float64',execution='eager',cflFactor=.25)
    params=dict(gamma=1.4,viscositySwitch='NoneSwitch',adaptiveSupportScheme='NoScheme' if settings['support']=='fixed' else 'Owen',rhoRight=.125,pressureRight=.1,equalMass=False)
    solver=dict(materials=[dict(name='gas',eos=dict(type='idealGas',gamma=1.4))],crkEvolution='paper',crkDensityPolicy='massReplacement',crkEnergyPartition='entropy',crkEnergyPredictor='raw',
        crkViscosityParams=dict(oneSidedWork=True,viscosityKernelScale=4.),diffusionParams=dict(C_l=2.,C_q=1.))
    ctx=RunConfiguration(case.name,simulation=sim,params=params,solver=solver).prepare(case)
    return case,ctx,sim,params,solver


def native_factory(s,db,hydro,nodes,settings,root):
    import ctypes
    api=ctypes.CDLL(None);assert api.warpsph_sod_version()==1 and len(nodes)==1
    hydro.Q.rigorousVelocityGradient=False
    save(root/'nativeConfiguration.json',dict(nodeLists=len(nodes),hydro=type(hydro).__name__,kernelExtent=4.,kernelOrder=7,nPerh=1.,
        Q={k:float(getattr(hydro.Q,k)) for k in ('Cl','Cq','epsilon2','etaCritFrac','etaFoldFrac')},freshCRKGradient=True,integrator='SynchronousRK2',
        overlay=str(Path(os.environ['LD_PRELOAD']).resolve()),overlaySHA256=hashlib.sha256(Path(os.environ['LD_PRELOAD']).read_bytes()).hexdigest()))
    return s.SynchronousRK2Integrator(db)


def run_row(args,settings):
    if args.mode=='warp':
        def initialize(c,system,a,settings):
            system.state.crkVolumes=None;system.state.crkMassDensities=system.state.densities.clone()
            c.scratch['_titleProvenance']='Single material Sod | Warp CRKSPH | '+settings['gammaVariant']+' | fresh RK2 / entropy work'
        return warp_run(args,settings,context,Measures,summarize,'Single material Sod',initialize_factory=initialize)
    if args.mode=='spheral':
        rc=spheral_run(args,settings,Measures,summarize,'Single material Sod matched steps',native_factory)
        import ctypes
        api=ctypes.CDLL(None);api.warpsph_sod_rhs_rows.restype=ctypes.c_ulonglong;api.warpsph_sod_rhs_data.restype=ctypes.POINTER(ctypes.c_double)
        n=api.warpsph_sod_rhs_rows();np.savez_compressed(args.output/'spheral/firstRHS.npz',rows=np.ctypeslib.as_array(api.warpsph_sod_rhs_data(),shape=(n,19)).copy())
        save(args.output/'spheral/overlayCalls.json',dict(steps=int(api.warpsph_sod_steps())))
        return rc
    from compare_spheralMaterials import render
    return render(args,settings,context)


def study(root):
    root.mkdir(exist_ok=True,parents=True);library=Path('results/materialEOS/spheral-sod1d-policies/libSpheralPaperPolicies.so').resolve()
    provenance=dict(library=str(library),librarySHA256=hashlib.sha256(library.read_bytes()).hexdigest(),rows=[],sources={})
    (root/'sources').mkdir(exist_ok=True)
    for name in (__file__,'scripts/compare_spheralMaterials.py','benchmarks/eos/paperSpheral/sodPolicies1d.cc'):
        p=Path(name);key=hashlib.sha256(p.read_bytes()).hexdigest();shutil.copy2(p,root/'sources'/(p.stem+'_'+key[:12]+p.suffix));provenance['sources'][str(p)]=key
    for nx in (200,400):
        x=-1.5+(np.arange(3*nx)+.5)/nx;rho=np.where(x<0,1.,.125);P=np.where(x<0,1.,.1)
        a=dict(positions=x[:,None],velocities=np.zeros((len(x),1)),densities=rho,pressures=P,internalEnergies=P/(.4*rho),
            supports=np.full(len(x),4/nx),masses=rho/nx,materials=np.zeros(len(x),dtype=np.int32),UIDs=np.arange(len(x),dtype=np.int64),kinds=np.zeros(len(x),dtype=np.int32))
        for support in ('fixed','adaptive'):
            folder=root/(f'n{nx}-'+support);folder.mkdir(exist_ok=True)
            if not (folder/'input.npz').exists():np.savez_compressed(folder/'input.npz',**a)
            settings=dict(case='sod',gammaVariant=folder.name,gamma=[1.4],nx=nx,dim=1,end=.2,dt=.05/nx,support=support,
                frameEvery=20,exactSteps=round(.2/(.05/nx)),domain=[-1.5,1.5],kernel='B8',spheralKernelOrder=7,extent=4.,nPerh=1.,n_h=4.,volumePolicy='kernelSum',
                inputPath=str((folder/'input.npz').resolve()),energyPartition='entropy',energyPredictor='raw',precision='float64',
                referenceViscosity=dict(Cl=2.,Cq=1.,epsilon2=.01,etaCritFrac=1.,etaFoldFrac=.2),
                source='One physical material and one NodeList; equal-spacing variable-mass classical Sod; no startup H relaxation. Matched paper/fresh stage controls, native support policies declared.')
            save(folder/'settings.json',settings);provenance['rows'].append(str(folder))
    save(root/'provenance.json',provenance)
    manifest=json.loads((root/'study.json').read_text()) if (root/'study.json').exists() else dict(stages=[])
    if 'error'in manifest:manifest.setdefault('previousErrors',[]).append(manifest.pop('error'))
    manifest.update(status='running',pid=os.getpid());env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',MPLCONFIGDIR='/tmp/warpsph-sod1d-mpl')
    def execute(name,command,native=False):
        record=dict(name=name,command=list(map(str,command)),status='running',log=str(root/(name+'.log')));manifest['stages'].append(record);save(root/'study.json',manifest)
        print('START',name,flush=True);start=time.perf_counter()
        with Path(record['log']).open('w') as log:
            child=subprocess.Popen(record['command'],stdout=log,stderr=subprocess.STDOUT,env=dict(env,LD_PRELOAD=str(library)) if native else env)
            record['pid']=child.pid;save(root/'study.json',manifest)
            try:
                while child.poll() is None:
                    if (root/'STOP').exists():child.send_signal(signal.SIGINT)
                    try:child.wait(timeout=5)
                    except subprocess.TimeoutExpired:pass
                    lines=Path(record['log']).read_text(errors='replace').replace('\r','\n').splitlines()
                    latest=next((s for s in reversed(lines) if any(v in s for v in ('| t=','SOD RESULT','Remote run registered:','Traceback'))),'')
                    if latest:print(name+': '+latest[:320],flush=True)
                rc=child.wait()
            except BaseException:child.send_signal(signal.SIGINT);child.wait();raise
        record.update(status='completed' if rc==0 else 'failed',returncode=rc,wallSeconds=time.perf_counter()-start);save(root/'study.json',manifest)
        print('END',name,rc,round(record['wallSeconds'],2),'s',flush=True)
        if rc:raise RuntimeError('Study stopped: '+name)
    try:
        for foldername in provenance['rows']:
            folder=Path(foldername)
            if not (folder/'spheral/comparisonRun.json').exists():execute(folder.name+'-spheral',['results/materialEOS/spheral-build/install/.venv/bin/python','-u',__file__,'spheral','--output',folder],True)
            if not (folder/'spheral/output.mp4').exists():execute(folder.name+'-render',[sys.executable,__file__,'render','--output',folder])
            if not (folder/'warp/comparisonRun.json').exists():execute(folder.name+'-warp',[sys.executable,'-u',__file__,'warp','--output',folder])
        manifest['status']='completed'
    except BaseException as error:manifest.update(status='failed',error=repr(error));raise
    finally:save(root/'study.json',manifest)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('study','spheral','warp','render'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--publisher-path',type=Path,default=Path('/home/lu26029/dev/warpSPHRemote/python'))
    a=p.parse_args()
    if a.mode=='study':study(a.output.resolve())
    else:raise SystemExit(run_row(a,json.loads((a.output/'settings.json').read_text())))
