"""Common-input cylindrical Sedov, matched fresh/paper CRK solver controls."""
import argparse
import ctypes
from dataclasses import replace
from functools import lru_cache
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import numpy as np
from compare_spheralMaterials import save,warp_run,spheral_run,SOURCE
from compare_spheralKH import native_factory

ANALYTIC=SOURCE/'tests/functional/Hydro/Sedov/SedovAnalyticSolution.py'


def spike_shape(q):
    """W7 radial shape in support-radius units; normalization cancels."""
    return np.maximum(1-q,0)**7-8*np.maximum(.75-q,0)**7+28*np.maximum(.5-q,0)**7-56*np.maximum(.25-q,0)**7


def sample(nx):
    axis=(np.arange(nx)+.5)/nx-.5;x,y=np.meshgrid(axis,axis,indexing='xy');pos=np.column_stack((x.ravel(),y.ravel()))
    rho=np.ones(nx*nx);m=rho/nx**2;weight=spike_shape(np.linalg.norm(pos,axis=1)/(2/nx));u=1.e-6/(5/3-1)+weight/weight.sum()/m
    return dict(positions=pos,velocities=np.zeros_like(pos),densities=rho,pressures=(5/3-1)*rho*u,internalEnergies=u,
        supports=np.full(len(pos),4/nx),masses=m,materials=np.zeros(len(pos),dtype=np.int32),
        UIDs=np.arange(len(pos),dtype=np.int64),kinds=np.zeros(len(pos),dtype=np.int32))


def time_grid(nx,half=False):
    n=300*nx//64;grid=np.diff(.05*(np.arange(n+1)/n)**2)
    return np.repeat(grid/2,2) if half else grid


@lru_cache(None)
def oracle():
    spec=importlib.util.spec_from_file_location('native_sedov_analytic',ANALYTIC);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    answer=module.SedovSolution(2,gamma=5/3,rho0=1.,E0=1.)
    # The pinned constructor passes nbins to a tolerance argument. Preserve its
    # alpha as provenance; normalize the same table by an exact piecewise
    # polynomial energy integral, independently checked against physical energy.
    raw_alpha=answer.alpha;x=np.array(answer._lam);z,w=np.polynomial.legendre.leggauss(3)
    q=x[:-1,None]+.5*np.diff(x)[:,None]*(1+z)
    rho,v,P=[np.interp(q,x,getattr(answer,k)) for k in ('_rholam','_vlam','_Plam')]
    integral=np.sum(.5*np.diff(x)[:,None]*w*(rho*v*v+P)*q)
    answer.alpha=8*2*np.pi/((answer.gamma-1)*(answer.gamma+1)*16)*integral;answer.E=1/answer.alpha
    return answer,raw_alpha


def exact(r,t):
    answer,_=oracle();vs,R,v2,rho2,P2=answer.shockState(max(t,1.e-30));q=np.asarray(r)/R;inside=q<1
    rho=np.where(inside,np.interp(q,answer._lam,answer._rholam)*rho2,1.)
    P=np.where(inside,np.interp(q,answer._lam,answer._Plam)*P2,0.)
    v=np.where(inside,np.interp(q,answer._lam,answer._vlam)*v2,0.)
    return dict(density=rho,pressure=P,velocity=v,energy=P/((5/3-1)*np.maximum(rho,1.e-100))),R


def radial_profile(a,nx):
    r=np.linalg.norm(a['positions'],axis=1);V=a['masses']/a['densities'];edges=np.arange(0,.451,1/nx/2)
    norm=np.histogram(r,edges,weights=V)[0];values={}
    vr=np.sum(a['positions']*a['velocities'],axis=1)/np.maximum(r,1.e-30)
    for name,field in [('density',a['densities']),('pressure',a['pressures']),('velocity',vr)]:
        num=np.histogram(r,edges,weights=V*field)[0];values[name]=np.divide(num,norm,out=np.full_like(norm,np.nan),where=norm>0)
    return .5*(edges[:-1]+edges[1:]),values


def front(a,nx,threshold=2.):
    r,fields=radial_profile(a,nx);valid=np.isfinite(fields['density']);r=r[valid];rho=fields['density'][valid]
    cross=np.flatnonzero((rho[:-1]>=threshold)&(rho[1:]<threshold))
    if not len(cross):return 0.
    j=cross[-1];return float(r[j]+(threshold-rho[j])/(rho[j+1]-rho[j])*(r[j+1]-r[j]))


class Measures:
    def __init__(self,settings,initial):
        self.settings=settings;self.mass=initial['masses'].sum();self.E=np.sum(initial['masses']*initial['internalEnergies'])
    def __call__(self,a,t):
        m,rho,u,v=(np.asarray(a[k],dtype=float) for k in ('masses','densities','internalEnergies','velocities'));x=a['positions'];r=np.linalg.norm(x,axis=1)
        speed2=(v*v).sum(1);vr=(v*x).sum(1)/np.maximum(r,1.e-30);vt=(x[:,0]*v[:,1]-x[:,1]*v[:,0])/np.maximum(r,1.e-30)
        out=dict(time=float(t),relativeEnergyDrift=float(np.dot(m,u+.5*speed2)/self.E-1),
            momentumError=float(np.linalg.norm(np.sum(m[:,None]*v,axis=0))/self.mass),materialMassError=float(abs(m.sum()/self.mass-1)),
            angularMomentum=float(np.dot(m,x[:,0]*v[:,1]-x[:,1]*v[:,0])),
            minDensity=float(rho.min()),minEnergy=float(u.min()),minPressure=float(a['pressures'].min()),
            maxVelocity=float(np.sqrt(speed2.max())),maxDensity=float(rho.max()),maxPressure=float(a['pressures'].max()),
            kineticEnergy=float(.5*np.dot(m,speed2)),tangentialKineticFraction=float(np.dot(m,vt**2)/max(np.dot(m,speed2),1.e-100)))
        # Finite-source startup is not the point-source similarity solution.
        if t>0:
            reference,R=exact(r,t);V=m/rho;mask=r<.45
            for name,field in [('density',rho),('pressure',a['pressures']),('velocity',vr)]:
                out[name+'L1']=float(np.dot(V[mask],abs(field[mask]-reference[name][mask]))/V[mask].sum())
            mask=(r>4/self.settings['nx'])&(r<.45)
            out['outerEnergyL1']=float(np.dot(V[mask],abs(u[mask]-reference['energy'][mask]))/V[mask].sum())
            measured=front(a,self.settings['nx']);out.update(exactShockRadius=float(R),density2Front=measured,
                density125Front=front(a,self.settings['nx'],1.25),density25Front=front(a,self.settings['nx'],2.5),
                relativeFrontError=float(measured/R-1) if measured else 0.)
        else:out.update({k:0. for k in ('densityL1','pressureL1','velocityL1','outerEnergyL1','exactShockRadius','density2Front','density125Front','density25Front','relativeFrontError')})
        return out


def summarize(root,solver,settings,history,initial,final,wall,steps,remote,extra):
    maxima={k:max(abs(r[k]) for r in history) for k in history[0] if k!='time'}
    gates=dict(completed=abs(history[-1]['time']-settings['end'])<1.e-10,
        positive=all(r['minEnergy']>0 and r['minDensity']>0 and r['minPressure']>0 for r in history),
        finite=all(np.isfinite(v).all() for v in final.values()),energy=maxima['relativeEnergyDrift']<settings.get('energyTolerance',1.e-10),
        momentum=maxima['momentumError']<1.e-10,mass=maxima['materialMassError']<1.e-12,singleMaterial=np.all(final['materials']==0))
    extras=dict(extra);extras['baseHarnessPolicyLabel']=extras.pop('policy',None)
    row=dict(solver=solver,settings=settings,remoteRunID=remote,steps=int(steps),wallSeconds=wall,meanStepMilliseconds=1000*wall/max(steps,1),
        time=history[-1]['time'],final=history[-1],maxima=maxima,gates={k:bool(v) for k,v in gates.items()},passed=bool(all(gates.values())),
        policy='Single material cylindrical Sedov; common normalized W7 spike; fresh CRK/RK2; paper volume/density timing; '+('neighbour-Q raw predictor/entropy final work.' if settings.get('compatibleEnergy',True) else 'standalone '+settings.get('thermalRate','paper')+' thermal du/dt; compatible and total-energy updates disabled.'),**extras)
    for name,a in [('initialState',initial),('finalState',final)]:np.savez_compressed(root/(name+'.npz'),**a)
    save(root/'diagnosticHistory.json',history);save(root/'comparisonRun.json',row)
    print('SEDOV RESULT',solver,settings['gammaVariant'],'passed',row['passed'],'wall',round(wall,3),'ms/step',round(row['meanStepMilliseconds'],3),'final',history[-1],flush=True)
    return row


def context(settings):
    from warpSPHBootstrap import bootstrap
    bootstrap(precision='float64')
    import torch
    from warpSPH.runner import RunConfiguration
    from warpSPH.cases.compressible.sedov import sedovCase
    from warpSPH.cases.compressible.common import configureCompressible
    from warpSPH.cases.common.plotting import Field,particlePlot
    from warpSPH.configurations import MaterialConfig,EOSConfig
    def configure(c):
        configureCompressible(c);c.schemeConfig.materials=[MaterialConfig('gas',EOSConfig(gamma=5/3))]
        c.config.domain.min[:]=-.5;c.config.domain.max[:]=.5
    def build(c):
        a=dict(np.load(settings['inputPath']));values={k:torch.as_tensor(v,device=c.device).clone() for k,v in a.items()};rho=values['densities']
        values.update(UIDcounter=len(rho),alphas=torch.ones_like(rho),alpha0s=torch.ones_like(rho),divergence=torch.zeros_like(rho))
        state=c.SimulationState(**values);state.totalEnergies=state.masses*state.internalEnergies
        return c.SimulationSystem(state=state,domain=c.config.domain)
    def post(c,s,step):c.scratch['sedovNextStep']=step+1
    def timestep(c,s):return settings['timeGrid'][min(c.scratch.get('sedovNextStep',0),settings['exactSteps']-1)]
    setup,update=particlePlot([Field('densities','Density',colorMap='viridis',vMin=0.,vMax=4.),
        Field('pressures','Pressure',colorMap='inferno',vMin=0.,vMax=5.)])
    case=replace(sedovCase,name='matchedCylindricalSedovComparison',configureScheme=configure,buildSystem=build,postStep=post,timestep=timestep,setupPlot=setup,updatePlot=update,
        simulationParameters={k:v for k,v in sedovCase.simulationParameters.items() if k!='tLimit'})
    sim=dict(nx=settings['nx'],dim=2,L=.5,tLimit=settings['end'],nSteps=settings['exactSteps'],dt=settings['dt'],adaptiveDt=False,
        kernel='B8',n_h=4.,precision='float64',execution='eager',cflFactor=.25)
    params=dict(gamma=5/3,rho0=1.,E0=1.,goalRadius=.26,viscositySwitch='NoneSwitch',adaptiveSupportScheme='Owen')
    solver=dict(materials=[dict(name='gas',eos=dict(type='idealGas',gamma=5/3))],crkEvolution='paper',crkDensityPolicy='massReplacement',
        crkEnergyPartition='entropy',crkEnergyPredictor='raw',crkViscosityParams=dict(oneSidedWork=True,viscosityKernelScale=4.),diffusionParams=dict(C_l=2.,C_q=1.))
    if not settings.get('compatibleEnergy',True):
        solver.update(compatibleEnergy=False,crkEnergyPartition='native')
    return case,RunConfiguration(case.name,simulation=sim,params=params,solver=solver).prepare(case),sim,params,solver


def run_row(args,settings):
    if args.mode=='warp':
        audit=None
        thermal=None
        if settings.get('auditThermal'):
            thermal=install_thermal_audit(settings,args.output)
        if settings.get('auditEOS'):
            # Research instrumentation only: installed EOS and model defaults
            # remain unchanged. The optional continuation changes c alone at
            # u<0, retaining negative predictor energy/pressure as evidence.
            from warpSPHBootstrap import bootstrap
            bootstrap(precision='float64')
            import torch
            import traceback
            from warpSPH.modules.eos import materials as eos
            original=eos.evaluateEOS;audit=dict(calls=0,negativeCalls=0,worstEnergy=0.,soundPolicy=settings.get('eosSound','strict'))
            def observed(rho,ids,config,**kwargs):
                result=original(rho,ids,config,**kwargs);audit['calls']+=1
                u=kwargs.get('u');minimum=float(u.min()) if u is not None else 0.
                if minimum<0:
                    audit['negativeCalls']+=1;audit['worstEnergy']=min(audit['worstEnergy'],minimum)
                    if 'firstNegative' not in audit:
                        audit['firstNegative']=dict(call=audit['calls'],minEnergy=minimum,count=int((u<0).sum()),
                            nonfiniteSound=int((~torch.isfinite(result[3])).sum()),
                            stack=[dict(file=row.filename,line=row.lineno,function=row.name) for row in traceback.extract_stack()[-9:]])
                        np.savez_compressed(args.output/'firstNegativeEOS.npz',rho=rho.detach().cpu().numpy(),u=u.detach().cpu().numpy(),
                            pressure=result[2].detach().cpu().numpy(),sound=result[3].detach().cpu().numpy(),materials=ids.detach().cpu().numpy())
                        save(args.output/'eosAudit.json',audit);print('EOS NEGATIVE PREDICTOR',audit['firstNegative'],flush=True)
                if settings.get('eosSound')=='spheralZero' and u is not None:
                    result=(*result[:3],torch.where(u<0,torch.zeros_like(result[3]),result[3]))
                return result
            eos.evaluateEOS=observed
        def initialize(c,system,a,settings):
            system.state.crkVolumes=None;system.state.crkMassDensities=system.state.densities.clone()
            c.scratch['_titleProvenance']='Cylindrical Sedov | Warp CRKSPH | '+settings['gammaVariant']+' | '+settings.get('eosSound','strict')+' sound | fresh RK2 / '+('entropy' if settings.get('compatibleEnergy',True) else 'standalone du/dt')
        try:return warp_run(args,settings,context,Measures,summarize,settings.get('namePrefix','Matched cylindrical Sedov'),initialize_factory=initialize)
        finally:
            if audit is not None:save(args.output/'eosAudit.json',audit)
            if thermal is not None:save(args.output/'thermalAudit.json',thermal)
    if args.mode=='spheral':
        try:
            rc=spheral_run(args,settings,Measures,summarize,settings.get('namePrefix','Matched cylindrical Sedov'),native_factory)
        finally:
            api=ctypes.CDLL(None);api.warpsph_paper_policy_calls.restype=ctypes.c_ulonglong
            save(args.output/'spheral/overlayCalls.json',dict(zip(('volume','density','finalEnergy','rawPredictor','RHS'),[api.warpsph_paper_policy_calls(i) for i in range(5)])))
            if settings.get('auditThermal'):
                captured={}
                for name,columns in (('node',40),('pair',16)):
                    rows=getattr(api,'warpsph_paper_heating_'+name+'_rows');rows.restype=ctypes.c_ulonglong
                    data=getattr(api,'warpsph_paper_heating_'+name+'_data');data.restype=ctypes.POINTER(ctypes.c_double)
                    n=rows()
                    captured[name]=np.ctypeslib.as_array(data(),shape=(n*columns,)).reshape(n,columns).copy()
                np.savez_compressed(args.output/'spheral/thermalStageAudit.npz',**captured)
        return rc
    from compare_spheralMaterials import render
    return render(args,settings,context)


def install_thermal_audit(settings,output):
    """Observe actual production RHS power; prohibit compatible correction.

    Child-process research instrumentation, with extra host synchronization.
    No replacement derivative, clamp, or model source modification.
    """
    from warpSPHBootstrap import bootstrap
    bootstrap(precision='float64')
    import importlib
    scheme=importlib.import_module('warpSPH.schemes.compressible.crkSPH')
    system=importlib.import_module('warpSPH.systems.compSPH')
    policy=settings.get('thermalRate','paper')
    if policy not in ('paper','entropy'):raise ValueError('Unknown thermal rate '+policy)
    audit=dict(rows=[],compatibleCalls=0,formula='Eq.78 P_j+Q_j' if policy=='paper' else 'Instantaneous pair power with Eq.63 entropy partition; dt=0 limit',compatibleEnergy=False)
    original_acc,original_du=scheme.computeCRKAccel,scheme.computeCRKdudt
    last={}
    def prohibited(*args,**kwargs):
        audit['compatibleCalls']+=1
        raise AssertionError('Compatible energy entered standalone thermal study')
    scheme.accumulateStage=system.energyIncrement=prohibited
    def acceleration(state,config,policy,*args):
        assert not policy.compatibleEnergy and policy.crkEnergyPartition=='native' and policy.crkEnergyPredictor=='raw'
        result=original_acc(state,config,policy,*args);last['acc']=result[0];last['pairs']=result[1]+result[2]
        return result
    def derivative(state,*args):
        du=original_du(state,*args)
        if policy=='entropy':
            from crkThermalRate import thermal_entropy_rate
            config,scheme_config,adjacency=args[:3]
            gamma=state.densities.new_full(state.densities.shape,5/3)
            du=thermal_entropy_rate(adjacency,last['pairs'],state.pressures/state.densities.pow(gamma),state.velocities)
        m=state.masses;u=state.internalEnergies;v=state.velocities;a=last['acc']
        mechanical=(m[:,None]*v*a).sum();heat=(m*du).sum()
        values=[mechanical,heat,(mechanical+heat),u.min(),du.min(),du.max()]
        import torch
        values=torch.stack(values).detach().cpu().numpy()
        row=dict(zip(('kineticPower','thermalPower','powerResidual','minEnergy','minDudt','maxDudt'),map(float,values)))
        row.update(rhs=len(audit['rows'])+1,normalizedPowerResidual=float(abs(values[2])/max(abs(values[0])+abs(values[1]),1.e-100)))
        audit['rows'].append({k:v if not isinstance(v,float) or np.isfinite(v) else None for k,v in row.items()})
        # Preserve the first negative thermal RHS at initially cold particles;
        # their positive source energy and derivative determine admissible dt.
        cold=(u<1.e-5)&(du<0)
        if 'firstColdCooling' not in audit and bool(cold.any()):
            host=lambda z:z.detach().cpu().numpy()
            np.savez_compressed(output/'coldCoolingRHS.npz',positions=host(state.positions),u=host(u),du=host(du),a=host(a),v=host(v),m=host(m),rho=host(state.densities),cold=host(cold))
            audit['firstColdCooling']=row['rhs']
        return du
    scheme.computeCRKAccel,scheme.computeCRKdudt=acceleration,derivative
    return audit


def study(root,dt_control=False):
    root.mkdir(exist_ok=True,parents=True);library=Path('results/materialEOS/spheral-heating-sign-policies/libSpheralPaperPolicies.so').resolve()
    provenance=dict(library=str(library),librarySHA256=hashlib.sha256(library.read_bytes()).hexdigest(),rows=[],sources={})
    (root/'sources').mkdir(exist_ok=True)
    for name in (__file__,'scripts/compare_spheralMaterials.py','scripts/compare_spheralKH.py','benchmarks/eos/paperSpheral/paperPolicies.cc',
        ANALYTIC,ANALYTIC.with_name('Sedov-cylindrical-2d.py')):
        p=Path(name);key=hashlib.sha256(p.read_bytes()).hexdigest();shutil.copy2(p,root/'sources'/(p.stem+'_'+key[:12]+p.suffix));provenance['sources'][str(p)]=key
    for nx,half in [(64,False),(128,False)]+([(64,True)] if dt_control else []):
        folder=root/(f'n{nx}'+('-halfdt' if half else ''));folder.mkdir(exist_ok=True)
        if not (folder/'input.npz').exists():np.savez_compressed(folder/'input.npz',**sample(nx))
        grid=time_grid(nx,half);steps=len(grid)
        settings=dict(case='cylindricalSedov',gammaVariant=folder.name,gamma=[5/3],nx=nx,dim=2,end=float(grid.sum()),dt=float(grid[0]),timeGrid=grid.tolist(),
            support='adaptive',frameEvery=steps//50,exactSteps=steps,domain=[-.5,.5],kernel='B8',spheralKernelOrder=7,extent=4.,nPerh=1.,n_h=4.,volumePolicy='kernelSum',
            inputPath=str((folder/'input.npz').resolve()),energyPartition='entropy',energyPredictor='raw',precision='float64',
            referenceViscosity=dict(Cl=2.,Cq=1.,epsilon2=.01,etaCritFrac=1.,etaFoldFrac=.2),
            source='Explicit common-input full-square cylindrical Sedov, native normalized W7 smooth spike, tiny ambient pressure; existing matched fresh/paper policies, not unchanged bundled driver.')
        if (folder/'settings.json').exists() and json.loads((folder/'settings.json').read_text())!=settings:raise ValueError('Existing settings differ: '+str(folder))
        save(folder/'settings.json',settings);provenance['rows'].append(str(folder))
    answer,raw=oracle();provenance['oracle']=dict(nativeAlpha=raw,normalizedAlpha=answer.alpha,shockState=answer.shockState(.05),
        note='Same native similarity table; energy normalized by exact three-point Gauss integration per table interval. Native constructor passes nbins into tolerance argument; no native source modified.')
    previous=root/'provenance.json'
    if previous.exists():shutil.copy2(previous,root/'previousProvenance.json')
    save(previous,provenance)
    manifest=json.loads((root/'study.json').read_text()) if (root/'study.json').exists() else dict(stages=[])
    if 'error' in manifest:manifest.setdefault('previousErrors',[]).append(manifest.pop('error'))
    manifest.update(status='running',pid=os.getpid());env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',MPLCONFIGDIR='/tmp/warpsph-sedov-mpl')
    def execute(name,command,native=False):
        record=dict(name=name,command=list(map(str,command)),status='running',log=str(root/(name+'.log')));manifest['stages'].append(record);save(root/'study.json',manifest)
        print('START',name,flush=True);start=time.perf_counter();last_printed=None
        with Path(record['log']).open('w') as log:
            child=subprocess.Popen(record['command'],stdout=log,stderr=subprocess.STDOUT,env=dict(env,LD_PRELOAD=str(library)) if native else env)
            record['pid']=child.pid;save(root/'study.json',manifest)
            try:
                while child.poll() is None:
                    if (root/'STOP').exists():child.send_signal(signal.SIGINT)
                    try:child.wait(timeout=5)
                    except subprocess.TimeoutExpired:pass
                    lines=Path(record['log']).read_text(errors='replace').replace('\r','\n').splitlines()
                    latest=next((s for s in reversed(lines) if any(v in s for v in ('| t=','SEDOV RESULT','Remote run registered:','Traceback'))),'')
                    if latest and latest!=last_printed:print(name+': '+latest[:320],flush=True);last_printed=latest
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
                if result.exists() and not json.loads(result.read_text())['passed']:raise RuntimeError('Retained failure; use fresh output: '+str(result))
            if not (folder/'spheral/comparisonRun.json').exists():execute(folder.name+'-spheral',['results/materialEOS/spheral-build/install/.venv/bin/python','-u',__file__,'spheral','--output',folder],True)
            if not (folder/'spheral/output.mp4').exists():execute(folder.name+'-render',[sys.executable,__file__,'render','--output',folder])
            if not (folder/'warp/comparisonRun.json').exists():execute(folder.name+'-warp',[sys.executable,'-u',__file__,'warp','--output',folder])
        manifest['status']='completed'
    except BaseException as error:manifest.update(status='failed',error=repr(error));raise
    finally:save(root/'study.json',manifest)


def investigate(root):
    """Retain the strict failure, then isolate the native sound continuation."""
    library=Path('results/materialEOS/spheral-heating-sign-policies/libSpheralPaperPolicies.so').resolve()
    manifest=json.loads((root/'diagnosticStudy.json').read_text()) if (root/'diagnosticStudy.json').exists() else dict(stages=[])
    manifest.update(status='running',pid=os.getpid());env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',
        MPLCONFIGDIR='/tmp/warpsph-sedov-mpl',PYTHONFAULTHANDLER='1')
    def execute(name,mode,folder,native=False,expected_failure=False):
        cmd=[('results/materialEOS/spheral-build/install/.venv/bin/python' if native else sys.executable),'-u',__file__,mode,'--output',str(folder)]
        logpath=root/(name+'.log');record=dict(name=name,command=cmd,log=str(logpath),status='running');manifest['stages'].append(record)
        start=time.perf_counter();last=None;print('START',name,flush=True)
        with logpath.open('w') as log:
            child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=dict(env,LD_PRELOAD=str(library)) if native else env)
            record['pid']=child.pid;save(root/'diagnosticStudy.json',manifest)
            try:
                while child.poll() is None:
                    if (root/'STOP').exists():child.send_signal(signal.SIGINT)
                    try:child.wait(timeout=5)
                    except subprocess.TimeoutExpired:pass
                    lines=logpath.read_text(errors='replace').replace('\r','\n').splitlines()
                    latest=next((s for s in reversed(lines) if any(v in s for v in ('| t=','SEDOV RESULT','EOS NEGATIVE','Traceback'))),'')
                    if latest and latest!=last:print(name+': '+latest[:400],flush=True);last=latest
                rc=child.wait()
            except BaseException:child.send_signal(signal.SIGINT);child.wait();raise
        known=bool(rc and expected_failure and (folder/'warp/failure.json').exists())
        record.update(returncode=rc,status='expected_numerical_failure' if known else ('completed' if rc==0 else 'failed'),wallSeconds=time.perf_counter()-start)
        save(root/'diagnosticStudy.json',manifest);print('END',name,record['status'],flush=True)
        if rc and not known:raise RuntimeError('Unexpected diagnostic failure: '+name)
        if known:
            from warpSPH.runner.media import encodeFrames
            exports=list((folder/'warp').glob('*/images'))
            if exports:encodeFrames(str(exports[-1]),str(exports[-1].parent))
    try:
        # Preserve the already completed native64; finish the unchanged native
        # refinement before the separate Warp EOS experiments.
        folder=root/'n128'
        if not (folder/'spheral/comparisonRun.json').exists():execute('n128-spheral','spheral',folder,True)
        if not (folder/'spheral/output.mp4').exists():execute('n128-render','render',folder)
        variants=[]
        for nx,policy,half,label in ((64,'strict',False,'auditStrict'),(64,'strict',True,'halfStrict'),
            (64,'spheralZero',False,'zeroSound'),(128,'spheralZero',False,'zeroSound')):
            folder=root/f'n{nx}-{label}';folder.mkdir(exist_ok=True);base=root/f'n{nx}'
            if not (folder/'input.npz').exists():shutil.copy2(base/'input.npz',folder/'input.npz')
            settings=json.loads((base/'settings.json').read_text());grid=time_grid(nx,half)
            settings.update(inputPath=str((folder/'input.npz').resolve()),gammaVariant=folder.name,eosSound=policy,auditEOS=True,
                timeGrid=grid.tolist(),dt=float(grid[0]),exactSteps=len(grid),frameEvery=len(grid)//50)
            save(folder/'settings.json',settings);variants.append(dict(folder=str(folder),reference=str(base/'spheral'),soundPolicy=policy,expectedFailure=policy=='strict'))
            if not (folder/'warp/comparisonRun.json').exists() and not (folder/'warp/failure.json').exists():
                execute(folder.name,'warp',folder,expected_failure=policy=='strict')
        save(root/'diagnosticVariants.json',variants)
        half=root/'n64-halfStrict'
        if (half/'warp/comparisonRun.json').exists():
            # A successful strict half-grid is a valid time-sensitivity
            # control; compare it to the same native sequence, not the
            # coarser native time grid used for the zero-sound isolation.
            if not (half/'spheral/comparisonRun.json').exists():execute('n64-halfStrict-spheral','spheral',half,True)
            if not (half/'spheral/output.mp4').exists():execute('n64-halfStrict-render','render',half)
        # The original watchdog stop also retains a playable partial video.
        original=root/'n64/warp'
        if (original/'failure.json').exists():
            from warpSPH.runner.media import encodeFrames
            for imagepath in original.glob('*/images'):
                if not (imagepath.parent/'output.mp4').exists():encodeFrames(str(imagepath),str(imagepath.parent))
        manifest['status']='completed'
    except BaseException as error:manifest.update(status='failed',error=repr(error));raise
    finally:save(root/'diagnosticStudy.json',manifest)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('study','investigate','spheral','warp','render'));p.add_argument('--dt-control',action='store_true')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--publisher-path',type=Path,default=Path('/home/lu26029/dev/warpSPHRemote/python'));a=p.parse_args()
    if a.mode=='study':study(a.output.resolve(),a.dt_control)
    elif a.mode=='investigate':investigate(a.output.resolve())
    else:raise SystemExit(run_row(a,json.loads((a.output/'settings.json').read_text())))
