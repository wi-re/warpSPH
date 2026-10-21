"""Shared-input, single-material smooth McNally KH in two fresh CRK solvers.

The pinned Spheral driver supplies the physical IC and mode diagnostic, not
the stage policies. Those are the declared paper/fresh controls from Sod.
"""
import argparse
import ctypes
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
from compare_spheralMaterials import save, warp_run, spheral_run


def initial_fields(positions):
    x,y=np.asarray(positions).T
    outer=(y<.25)|(y>=.75)
    # Four branches of the bundled McNally exponential transition.
    distance=np.minimum(abs(y-.25),abs(y-.75))
    blend=.5*np.exp(-distance/.025)
    rho=np.where(outer,1.+blend,2.-blend)
    vx=np.where(outer,.5-blend,-.5+blend)
    return rho,np.column_stack((vx,.01*np.sin(4*np.pi*x)))


def sample(nx):
    axis=(np.arange(nx)+.5)/nx;x,y=np.meshgrid(axis,axis,indexing='xy')
    pos=np.column_stack((x.ravel(),y.ravel()));rho,v=initial_fields(pos);P=np.full(nx*nx,2.5)
    return dict(positions=pos,velocities=v,densities=rho,pressures=P,internalEnergies=P/((5/3-1)*rho),
        supports=np.full(len(pos),4/nx),masses=rho/nx**2,materials=np.zeros(len(pos),dtype=np.int32),
        UIDs=np.arange(len(pos),dtype=np.int64),kinds=np.zeros(len(pos),dtype=np.int32))


def mixing_mode(a):
    x,y=(a['positions']%1).T;vy=a['velocities'][:,1]
    # Native McNally mixingScale uses h^2, not m/rho, as quadrature weights.
    weight=(a['supports']/4)**2*np.exp(-4*np.pi*np.minimum(abs(y-.25),abs(y-.75)))
    z=np.sum(weight*vy*np.exp(4j*np.pi*x))/weight.sum()
    return float(2*abs(z))


class Measures:
    def __init__(self,settings,initial):
        self.mass=initial['masses'].sum();self.E=np.sum(initial['masses']*(initial['internalEnergies']+.5*(initial['velocities']**2).sum(1)))
        self.p=np.sum(initial['masses'][:,None]*initial['velocities'],axis=0)
        self.entropy=np.sum(initial['masses']*np.log(initial['pressures']/initial['densities']**(5/3)))
    def __call__(self,a,t):
        m,rho,u,v=(np.asarray(a[k],dtype=float) for k in ('masses','densities','internalEnergies','velocities'));P=a['pressures']
        return dict(time=float(t),relativeEnergyDrift=float(np.sum(m*(u+.5*(v*v).sum(1)))/self.E-1),
            momentumError=float(np.linalg.norm(np.sum(m[:,None]*v,axis=0)-self.p)/self.mass),
            materialMassError=float(abs(m.sum()/self.mass-1)),minDensity=float(rho.min()),minEnergy=float(u.min()),
            minPressure=float(P.min()),maxPressure=float(P.max()),maxVelocity=float(np.linalg.norm(v,axis=1).max()),
            modeAmplitude=mixing_mode(a),transverseKineticEnergy=float(.5*np.dot(m,v[:,1]**2)),
            maxTransverseKineticDensity=float((.5*rho*v[:,1]**2).max()),
            pressureRMS=float(np.sqrt(np.dot(m/rho,(P-2.5)**2)/np.sum(m/rho))),
            entropyChangePerMass=float((np.sum(m*np.log(P/rho**(5/3)))-self.entropy)/self.mass))


def summarize(root,solver,settings,history,initial,final,wall,steps,remote,extra):
    maxima={k:max(abs(r[k]) for r in history) for k in history[0] if k!='time'}
    gates=dict(completed=abs(history[-1]['time']-settings['end'])<settings['dt']*.01,
        positive=all(r['minEnergy']>0 and r['minDensity']>0 and r['minPressure']>0 for r in history),
        finite=all(np.isfinite(v).all() for v in final.values()),energy=maxima['relativeEnergyDrift']<1.e-10,
        momentum=maxima['momentumError']<1.e-10,mass=maxima['materialMassError']<1.e-12,
        singleMaterial=np.all(final['materials']==0))
    extras=dict(extra);extras['baseHarnessPolicyLabel']=extras.pop('policy',None)
    row=dict(solver=solver,settings=settings,remoteRunID=remote,steps=int(steps),wallSeconds=wall,
        meanStepMilliseconds=1000*wall/max(steps,1),time=history[-1]['time'],final=history[-1],maxima=maxima,
        gates={k:bool(v) for k,v in gates.items()},passed=bool(all(gates.values())),
        policy='One EOS/NodeList; smooth McNally KH; fresh CRK/RK2; Eq75 volume per RHS; Eq76 density per step; raw neighbour-Q predictor; entropy final work.',**extras)
    for name,a in [('initialState',initial),('finalState',final)]:np.savez_compressed(root/(name+'.npz'),**a)
    save(root/'diagnosticHistory.json',history);save(root/'comparisonRun.json',row)
    print('KH RESULT',solver,settings['gammaVariant'],'passed',row['passed'],'wall',round(wall,3),
        'ms/step',round(row['meanStepMilliseconds'],3),'final',history[-1],flush=True)
    return row


def context(settings):
    from warpSPHBootstrap import bootstrap
    bootstrap(precision='float64')
    import torch
    from warpSPH.runner import RunConfiguration
    from warpSPH.cases.compressible.kelvinHelmholtz import kelvinHelmholtzCase,configureScheme
    from warpSPH.cases.common.plotting import Field,particlePlot
    from warpSPH.configurations import MaterialConfig,EOSConfig
    def configure(c):
        configureScheme(c);c.schemeConfig.materials=[MaterialConfig('gas',EOSConfig(gamma=5/3))]
    def build(c):
        a=dict(np.load(settings['inputPath']));values={k:torch.as_tensor(v,device=c.device).clone() for k,v in a.items()}
        rho=values['densities'];values.update(UIDcounter=len(rho),alphas=torch.ones_like(rho),alpha0s=torch.ones_like(rho),divergence=torch.zeros_like(rho))
        state=c.SimulationState(**values);state.totalEnergies=state.masses*(state.internalEnergies+.5*(state.velocities**2).sum(-1))
        return c.SimulationSystem(state=state,domain=c.config.domain)
    setup,update=particlePlot([Field('densities','Density',colorMap='viridis',vMin=1.,vMax=2.),
        Field('velocities','Transverse velocity',mapping='y',colorMap='RdBu',colorMapKind='diverging',vMin=-.5,vMax=.5)])
    case=replace(kelvinHelmholtzCase,name='smoothSingleMaterialKHComparison',configureScheme=configure,buildSystem=build,setupPlot=setup,updatePlot=update)
    sim=dict(nx=settings['nx'],dim=2,L=1.,tLimit=settings['end'],nSteps=settings['exactSteps'],dt=settings['dt'],adaptiveDt=False,
        kernel='B8',n_h=4.,precision='float64',execution='eager',cflFactor=.25)
    params=dict(gamma=5/3,viscositySwitch='NoneSwitch',adaptiveSupportScheme='Owen',smoothDensity=1)
    solver=dict(materials=[dict(name='gas',eos=dict(type='idealGas',gamma=5/3))],crkEvolution='paper',crkDensityPolicy='massReplacement',
        crkEnergyPartition='entropy',crkEnergyPredictor='raw',crkViscosityParams=dict(oneSidedWork=True,viscosityKernelScale=4.),diffusionParams=dict(C_l=2.,C_q=1.))
    return case,RunConfiguration(case.name,simulation=sim,params=params,solver=solver).prepare(case),sim,params,solver


def native_factory(s,db,hydro,nodes,settings,root):
    api=ctypes.CDLL(None);assert api.warpsph_paper_policy_version()==1 and len(nodes)==1
    api.warpsph_paper_fresh_rk2(1);api.warpsph_paper_dudt_partition(0);api.warpsph_paper_av_gradient(2);api.warpsph_paper_averaged_heating(0)
    if settings.get('thermalRate')=='entropy':api.warpsph_paper_entropy_thermal(1)
    hydro.Q.rigorousVelocityGradient=False
    library=Path(os.environ['LD_PRELOAD']).resolve()
    save(root/'nativeConfiguration.json',dict(nodeLists=len(nodes),hydro=type(hydro).__name__,kernelExtent=4.,kernelOrder=7,nPerh=1.,
        compatibleEnergy=bool(hydro.compatibleEnergyEvolution),evolveTotalEnergy=bool(hydro.evolveTotalEnergy),
        thermalRate=settings.get('thermalRate','paper'),
        Q={k:float(getattr(hydro.Q,k)) for k in ('Cl','Cq','epsilon2','etaCritFrac','etaFoldFrac')},freshCRKGradient=True,
        integrator='SynchronousRK2',overlay=str(library),overlaySHA256=hashlib.sha256(library.read_bytes()).hexdigest()))
    return s.SynchronousRK2Integrator(db)


def run_row(args,settings):
    if args.mode=='warp':
        def initialize(c,system,a,settings):
            system.state.crkVolumes=None;system.state.crkMassDensities=system.state.densities.clone()
            c.scratch['_titleProvenance']='Smooth KH | Warp CRKSPH | '+settings['gammaVariant']+' | fresh RK2 / entropy work'
        return warp_run(args,settings,context,Measures,summarize,'Smooth single-material KH',initialize_factory=initialize)
    if args.mode=='spheral':
        rc=spheral_run(args,settings,Measures,summarize,'Smooth single-material KH',native_factory)
        api=ctypes.CDLL(None);api.warpsph_paper_policy_calls.restype=ctypes.c_ulonglong
        save(args.output/'spheral/overlayCalls.json',dict(zip(('volume','density','finalEnergy','rawPredictor','RHS'),[api.warpsph_paper_policy_calls(i) for i in range(5)])))
        return rc
    from compare_spheralMaterials import render
    return render(args,settings,context)


def study(root,dt_control=False):
    root.mkdir(exist_ok=True,parents=True);library=Path('results/materialEOS/spheral-heating-sign-policies/libSpheralPaperPolicies.so').resolve()
    provenance=dict(library=str(library),librarySHA256=hashlib.sha256(library.read_bytes()).hexdigest(),rows=[],sources={})
    (root/'sources').mkdir(exist_ok=True)
    for name in (__file__,'scripts/compare_spheralMaterials.py','benchmarks/eos/paperSpheral/paperPolicies.cc',
        'results/materialEOS/spheral-build/source/tests/functional/Hydro/KelvinHelmholtz/KelvinHelmholtz-2d_McNally.py'):
        p=Path(name);key=hashlib.sha256(p.read_bytes()).hexdigest();shutil.copy2(p,root/'sources'/(p.stem+'_'+key[:12]+p.suffix));provenance['sources'][str(p)]=key
    controls=[(64,1),(128,1)]+([(64,2)] if dt_control else [])
    for nx,divider in controls:
        folder=root/(f'n{nx}'+('-halfdt' if divider==2 else ''));folder.mkdir(exist_ok=True)
        if not (folder/'input.npz').exists():np.savez_compressed(folder/'input.npz',**sample(nx))
        dt=.128/nx/divider
        settings=dict(case='smoothKH',gammaVariant=folder.name,gamma=[5/3],nx=nx,dim=2,end=2.,dt=dt,support='adaptive',
            frameEvery=round(.04/dt),exactSteps=round(2/dt),domain=[0.,1.],kernel='B8',spheralKernelOrder=7,extent=4.,nPerh=1.,n_h=4.,volumePolicy='kernelSum',
            inputPath=str((folder/'input.npz').resolve()),energyPartition='entropy',energyPredictor='raw',precision='float64',
            referenceViscosity=dict(Cl=2.,Cq=1.,epsilon2=.01,etaCritFrac=1.,etaFoldFrac=.2),
            source='Pinned Spheral McNally physical IC/diagnostic; shared input, one EOS/NodeList, equal spacing/variable mass. Fresh RK2 and declared paper policies, not unchanged bundled driver.')
        save(folder/'settings.json',settings);provenance['rows'].append(str(folder))
    save(root/'provenance.json',provenance)
    manifest=json.loads((root/'study.json').read_text()) if (root/'study.json').exists() else dict(stages=[])
    if 'error' in manifest:manifest.setdefault('previousErrors',[]).append(manifest.pop('error'))
    manifest.update(status='running',pid=os.getpid());env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',MPLCONFIGDIR='/tmp/warpsph-kh-mpl')
    def execute(name,command,native=False):
        record=dict(name=name,command=list(map(str,command)),status='running',log=str(root/(name+'.log')));manifest['stages'].append(record);save(root/'study.json',manifest)
        print('START',name,flush=True);start=time.perf_counter()
        last_printed=None
        with Path(record['log']).open('w') as log:
            child=subprocess.Popen(record['command'],stdout=log,stderr=subprocess.STDOUT,env=dict(env,LD_PRELOAD=str(library)) if native else env)
            record['pid']=child.pid;save(root/'study.json',manifest)
            try:
                while child.poll() is None:
                    if (root/'STOP').exists():child.send_signal(signal.SIGINT)
                    try:child.wait(timeout=5)
                    except subprocess.TimeoutExpired:pass
                    lines=Path(record['log']).read_text(errors='replace').replace('\r','\n').splitlines()
                    latest=next((s for s in reversed(lines) if any(v in s for v in ('| t=','KH RESULT','Remote run registered:','Traceback'))),'')
                    if latest and latest!=last_printed:
                        print(name+': '+latest[:320],flush=True);last_printed=latest
                rc=child.wait()
            except BaseException:child.send_signal(signal.SIGINT);child.wait();raise
        record.update(status='completed' if rc==0 else 'failed',returncode=rc,wallSeconds=time.perf_counter()-start);save(root/'study.json',manifest)
        print('END',name,rc,round(record['wallSeconds'],2),'s',flush=True)
        if rc:raise RuntimeError('Study stopped: '+name)
    try:
        for foldername in provenance['rows']:
            folder=Path(foldername)
            for solver in ('spheral','warp'):
                result=folder/solver/'comparisonRun.json'
                if result.exists() and not json.loads(result.read_text())['passed']:
                    raise RuntimeError('Retained failed result; use a fresh output directory: '+str(result))
            if not (folder/'spheral/comparisonRun.json').exists():execute(folder.name+'-spheral',['results/materialEOS/spheral-build/install/.venv/bin/python','-u',__file__,'spheral','--output',folder],True)
            if not (folder/'spheral/output.mp4').exists():execute(folder.name+'-render',[sys.executable,__file__,'render','--output',folder])
            if not (folder/'warp/comparisonRun.json').exists():execute(folder.name+'-warp',[sys.executable,'-u',__file__,'warp','--output',folder])
        manifest['status']='completed'
    except BaseException as error:manifest.update(status='failed',error=repr(error));raise
    finally:save(root/'study.json',manifest)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('study','spheral','warp','render'))
    p.add_argument('--dt-control',action='store_true',help='Also repeat the 64² pair with half the timestep.')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--publisher-path',type=Path,default=Path('/home/lu26029/dev/warpSPHRemote/python'))
    a=p.parse_args()
    if a.mode=='study':study(a.output.resolve(),a.dt_control)
    else:raise SystemExit(run_row(a,json.loads((a.output/'settings.json').read_text())))
