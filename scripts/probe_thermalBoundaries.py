"""Small CPU diagnostics for thermal prescriptions, not convection certification.

Whole-domain thermostats isolate state lifecycle from wall geometry. Fourier
probes freeze hydrodynamics and characterize the existing artificial heat
operator. Reservoir volume probes isolate its prescribed-pressure contract.
These tiny numerical controls intentionally omit video and stream progress.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import inspect
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import shutil
import tarfile
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'src'))


def thermostat_error(mass, velocity, energy, target):
    """Independent caloric u/T and missing reset-heat measurements (c_v=1)."""
    import torch
    m, v, u = mass.double(), velocity.double(), energy.double()
    expected = torch.as_tensor(target, dtype=u.dtype, device=u.device).expand_as(u)
    return dict(maxURelativeError=float(((u-expected).abs()/expected.abs()).max()),
                totalEnergy=float((m*(u+.5*v.square().sum(-1))).sum()),
                missingResetHeat=float((m*(expected-u)).sum()), mass=float(m.sum()))


def mode_response(x, energy, rate, mass, length=2.):
    """Project onto the prescribed sine; no fitted rate is a physical input."""
    import torch
    wave = torch.sin(2*math.pi*x.double()/length)
    mode = energy.double()-1.
    r, m = rate.double(), mass.double()
    lam = -float((m*mode*r).sum()/(m*mode.square()).sum())
    norm = float((m*r.square()).sum().sqrt())
    leak = float((m*(r+lam*mode).square()).sum().sqrt())/max(norm,1e-30)
    power = m*r
    return dict(fittedDecayRate=lam, fittedDiffusivity=lam/(2*math.pi/length)**2,
        nonFourierRateFraction=leak, energyRate=float(power.sum()),
        relativeEnergyRateResidual=abs(float(power.sum()))/max(float(power.abs().sum()),1e-30),
        entropyRate=float((m*r/energy.double()).sum()),
        projectedAmplitude=float((m*mode*wave).sum()/(m*wave.square()).sum()))


def boundary_toy(scheme, variant, nx, dt, end, tableau, pattern, output):
    import numpy as np
    import torch
    from warpSPH.runner import RunConfiguration
    from warpSPH.cases.compressible.sod import sodCase
    from warpSPH.configurations import BoundaryCondition, BoundaryConditionType
    from warpSPH.modules.eos import refreshEOS
    from warpSPH.modules.meshless.boundary import DrivenParticles
    from warpSPH.modules.meshless.geometry import buildGeometry

    history, stages, saved = [], [], {}
    def target(t):
        return 1.+float(t)
    def build(ctx):
        # The meshless Sod factory derives its startup timestep without passing
        # a fixed dt. Bootstrap that factory adaptively, then restore the exact
        # prescribed dt before any measured integration step.
        adaptive=ctx.config.adaptiveDt
        if scheme in ('MFM','MFV'):ctx.config.adaptiveDt=True
        try:
            system = sodCase.buildSystem(ctx)
        finally:
            ctx.config.adaptiveDt=adaptive
            ctx.config.dt=dt
        s = system.state
        saved['ctx'] = ctx
        if scheme in ('MFM','MFV'):
            geometry = buildGeometry(s.positions,s.supports,s.lam,ctx.config,ctx.schemeConfig,system.domain)
            rho = s.masses/geometry.volume
            driven = DrivenParticles(torch.ones_like(s.masses,dtype=torch.bool),s.positions.clone(),s.masses.clone(),
                lambda t,ref: ref, lambda t,pos: (torch.zeros_like(pos),(ctx.schemeConfig.gamma-1)*rho*target(t)))
            system.drivenParticles = driven
        else:
            if pattern == 'density-wave':
                s.masses = s.masses*(1.+.2*torch.cos(math.pi*s.positions[:,0]))
            density = lambda pos: (1.+.2*torch.cos(math.pi*pos[:,0]) if pattern == 'density-wave' else torch.ones_like(pos[:,0]))
            s.velocities.zero_(); s.densities = density(s.positions)
            s.internalEnergies.fill_(target(0.))
            refreshEOS(s,ctx.schemeConfig)
            s.totalEnergies=s.masses*s.internalEnergies
            ctx.schemeConfig.boundaryConditions = [BoundaryCondition(BoundaryConditionType.dynamic,
                sdf=lambda pos: (-torch.ones_like(pos[:,0]),torch.zeros_like(pos)),
                dirichletFunctions={'internalEnergies':lambda state,cfg,sc,pos,d,n,t,dt: torch.full_like(state.masses,target(t)),
                    'densities':lambda state,cfg,sc,pos,*args: density(pos),
                    'velocities':lambda state,*args: torch.zeros_like(state.velocities)},
                updateFunctions={'dxdt':lambda state,*args: torch.zeros_like(state.velocities),
                    'dvdt':lambda state,*args: torch.zeros_like(state.velocities)})]
        native=ctx.stepFunction; wants=scheme in ('MFM','MFV') or 'stageIndex' in inspect.signature(native).parameters
        def checked(system,dt,config,schemeConfig,verbose=False,stageIndex=None,**kwargs):
            extra=dict(kwargs)
            if wants:extra['stageIndex']=stageIndex
            result=native(system,dt,config,schemeConfig,verbose=verbose,**extra)
            state=system.state
            # Meshless public primitive fields retain a start-of-step policy.
            stage=getattr(system,'meshlessContext',None)
            u=stage.u if stage is not None else state.internalEnergies
            v=stage.vel if stage is not None else state.velocities
            stages.append(dict(t=float(system.t),stageIndex=stageIndex,
                **thermostat_error(stage.Q[:,0] if stage is not None else state.masses,v,u,target(system.t))))
            return result
        ctx.bundle=replace(ctx.bundle,stepFunction=checked)
        return system
    def diagnostics(ctx,system):
        s=system.state
        row=dict(t=float(system.t),**thermostat_error(s.masses,s.velocities,s.internalEnergies,target(system.t)))
        history.append(row)
        return row
    solver=dict(scheme=scheme)
    if scheme in ('MFM','MFV'):solver['temporalScheme']='ssprk'
    if variant:solver.update(variant=variant,dissipation=variant=='PressureEnergy')
    cfg=RunConfiguration('sod',simulation=dict(device='cpu',nx=nx,dt=dt,adaptiveDt=False,
        nSteps=round(end/dt),integrationScheme=tableau,plot=False,video=False,store=False,
        show=False,quiet=True,progress=True),solver=solver,
        params=dict(gamma=1.4,samplingRatio=1,left_rho=1.,right_rho=1.,left_pressure=.4,
            right_pressure=.4,left_velocity=0.,right_velocity=0.))
    result=cfg.run(replace(sodCase,buildSystem=build,diagnostics=diagnostics))
    first,last=history[0],history[-1]
    expectedHeat=first['mass']*(target(last['t'])-target(first['t']))
    row=dict(scheme=scheme,variant=variant,nx=nx,dt=dt,tableau=tableau if scheme not in ('MFM','MFV') else 'ssprk',
        pattern=pattern,steps=result.nSteps,t=last['t'],diverged=result.diverged,
        initialURelativeError=first['maxURelativeError'],acceptedURelativeError=last['maxURelativeError'],
        maxStageURelativeError=max(s['maxURelativeError'] for s in stages),
        prescribedHeat=expectedHeat,actualEnergyChange=last['totalEnergy']-first['totalEnergy'],
        caloricEnergyResidual=last['totalEnergy']-first['totalEnergy']-expectedHeat,
        initialResetHeat=first['missingResetHeat'],
        caloricResidualIncludingInitialReset=last['totalEnergy']-first['totalEnergy']-expectedHeat-first['missingResetHeat'],
        missingResetHeat=last['missingResetHeat'])
    s=result.state.state
    np.savez_compressed(output/'fields.npz',positions=s.positions.detach().numpy(),masses=s.masses.detach().numpy(),
        velocities=s.velocities.detach().numpy(),internalEnergies=s.internalEnergies.detach().numpy())
    (output/'history.json').write_text(json.dumps(dict(accepted=history,stages=stages),indent=2)+'\n')
    (output/'input.json').write_text(json.dumps(dict(simulation=cfg.simulation,solver=solver,params=cfg.params),indent=2)+'\n')
    return row


def conduction_toy(nx, amplitude, term, dt, end, output, isobaric=False):
    import numpy as np
    import torch
    from warpSPH.configurations import buildConfig
    from warpSPH.configurations.moduleConfigurations.diffusionParameters import DiffusionParameters,ViscosityTerms
    from warpSPH.modules.dissipation import computeConductivity
    from warpSPH.utils.domain import buildDomainDescription
    from warpSPHCore import KernelFunctions,OperationProperties,SupportScheme,WarpOperation,buildVerletList
    from warpSPHCore.type_config import get_torch_precision
    dtype=get_torch_precision();dx=2./nx
    x=(torch.arange(nx,dtype=dtype)+.5)*dx-1.
    u=1.+amplitude*torch.sin(math.pi*x)
    rho=1./u if isobaric else torch.ones_like(u)
    domain=buildDomainDescription(2.,1,periodic=True,device='cpu',dtype=dtype)
    cfg,_=buildConfig(dim=1,domain=domain,kernel=KernelFunctions.CubicSpline,n_h=4.,device='cpu',dtype=dtype)
    state=SimpleNamespace(positions=x[:,None],supports=torch.full_like(u,4*dx),masses=rho*dx,
        densities=rho,velocities=torch.zeros(nx,1,dtype=dtype),internalEnergies=u.clone(),
        kinds=torch.zeros(nx,dtype=torch.int32),soundspeeds=torch.ones_like(u),alphas=torch.ones_like(u))
    adjacency=buildVerletList(state,domain,verletScale=1.,supportMode=SupportScheme.SuperSymmetric)
    params=DiffusionParameters();params.thermalConductivity=.5;params.thermalConductivityTerm=ViscosityTerms[term].value
    params.Cu_l=1.;params.Cu_q=0.;params.correctXi=False
    props=OperationProperties(kernel=cfg.kernel,supportMode=SupportScheme.KernelMeanSymmetric,operation=WarpOperation.Interpolate)
    def rate(values):
        state.internalEnergies=values
        # Prescribed constant signal speed isolates this operator; hydro is off.
        state.pressures=torch.full_like(values,.4) if isobaric else .4*state.densities*values
        return computeConductivity(state,operationProperties=props,domain=domain,
            conductivityParams=params,adjacency=adjacency)
    initial=u.clone();initialRate=rate(u)
    response=mode_response(x,u,initialRate,state.masses)
    initialEnergy=float((state.masses.double()*u.double()).sum())
    rows=[dict(t=0.,**response)]
    if isobaric and end:
        raise ValueError('isobaric probes measure the initial operator only')
    steps=round(end/dt) if end else 0
    if steps and not math.isclose(steps*dt,end,rel_tol=0.,abs_tol=1e-12):
        raise ValueError('conduction end time must be an integer number of steps')
    for step in range(steps):
        k1=rate(u);k2=rate(u+.5*dt*k1);u=u+dt*k2
        if (step+1)%max(1,steps//4)==0:
            print(f'conduction nx={nx} term={term} step={step+1}/{steps} t={(step+1)*dt:.6g}',flush=True)
            rows.append(dict(t=(step+1)*dt,**mode_response(x,u,rate(u),state.masses)))
    ref=1.+amplitude*math.exp(-response['fittedDecayRate']*end)*torch.sin(math.pi*x)
    change=float((state.masses.double()*(u.double()-initial.double())).sum())
    np.savez_compressed(output/'fields.npz',positions=x.numpy(),masses=state.masses.numpy(),initial=initial.numpy(),
        initialRate=initialRate.numpy(),final=u.numpy(),fittedReference=ref.numpy())
    (output/'history.json').write_text(json.dumps(rows,indent=2)+'\n')
    return dict(nx=nx,amplitude=amplitude,term=term,dt=dt,t=end,steps=steps,isobaric=isobaric,
        **response,energyChange=change,relativeEnergyChange=change/initialEnergy,
        minU=float(u.min()),maxU=float(u.max()),fittedReferenceL2=float((u-ref).double().square().mean().sqrt()),
        frozenSignalSpeed=1.,thermalConductivityMultiplier=.5,
        physicalDiffusionQualified=False)


def reservoir_toy(output):
    import numpy as np
    import torch
    from warpSPH.modules.meshless.boundary import DrivenParticles
    from warpSPHCore.type_config import get_torch_precision
    dtype=get_torch_precision();mass=torch.ones(8,dtype=dtype)/8
    pos=torch.linspace(-.5,.5,8,dtype=dtype)[:,None]
    state=SimpleNamespace(positions=pos.clone(),Q=torch.zeros(8,3,dtype=dtype))
    gamma=1.4;pressure=torch.full_like(mass,gamma-1)
    driven=DrivenParticles(torch.ones(8,dtype=torch.bool),pos.clone(),mass,
        lambda t,x:x*(1+t),lambda t,x:(torch.zeros_like(x),pressure))
    history=[]
    for scale in [1.,1.1,1.25]:
        geometry=SimpleNamespace(volume=mass*scale)
        driven.position(state,scale-1);driven.conserved(state,geometry,scale-1,gamma)
        u=state.Q[:,-1]/mass
        actualP=(gamma-1)*state.Q[:,-1]/geometry.volume
        history.append(dict(volumeRatio=scale,uMean=float(u.mean()),fixedUTargetError=float((u-1).abs().max()),
            pressureRelativeError=float(((actualP-pressure)/pressure).abs().max()),
            totalEnergy=float(state.Q[:,-1].double().sum())))
    np.savez_compressed(output/'fields.npz',Q=state.Q.numpy(),positions=state.positions.numpy())
    (output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    return dict(rows=history,interpretation='Prescribed pressure reservoir; volumes supplied analytically, no hydro solve.',steps=0)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--precision',choices=['float32','float64'],default='float64')
    p.add_argument('--nx',type=int,nargs='+',default=[32,64])
    p.add_argument('--dt',type=float,nargs='+',default=[.002,.001])
    p.add_argument('--time',type=float,default=.02)
    p.add_argument('--sections',nargs='+',choices=['boundary','conduction','reservoir'],default=['boundary','conduction','reservoir'])
    args=p.parse_args(argv)
    if args.time<=0 or any(n<16 for n in args.nx) or any(dt<=0 or abs(args.time/dt-round(args.time/dt))>1e-8 for dt in args.dt):
        p.error('require nx>=16, positive time/dt, and an integral step count')
    root=args.root.resolve();root.mkdir(parents=True,exist_ok=False)
    shutil.copy2(__file__,root/'driver.py')
    from warpSPHBootstrap import bootstrap
    bootstrap(precision=args.precision)
    import torch
    if torch.cuda.is_initialized():raise RuntimeError('this is a CPU-only probe')
    torch.set_num_threads(1)
    import warpSPH,warpSPHCore,warpSPHIntegrators
    packages=[Path(module.__file__).parent for module in (warpSPH,warpSPHCore,warpSPHIntegrators)]
    def inventory():
        return {str(path):hashlib.sha256(path.read_bytes()).hexdigest()
                for package in packages for path in sorted(package.rglob('*.py'))}
    before=inventory()
    for package in packages:
        with tarfile.open(root/f'{package.name}-sources.tar.gz','w:gz') as archive:
            archive.add(package,arcname=package.name,
                filter=lambda item:None if '__pycache__' in item.name or item.name.endswith('.pyc') else item)
    results=[]
    def run(section,label,fn):
        output=root/label;output.mkdir();start=time.perf_counter()
        print(f'START {label}',flush=True)
        try:row=dict(section=section,label=label,**fn(output))
        except Exception as e:
            row=dict(section=section,label=label,error=f'{type(e).__name__}: {e}')
            import traceback
            (output/'error.txt').write_text(traceback.format_exc())
        row['wallSeconds']=time.perf_counter()-start
        if row.get('steps'):row['wallMsPerStepIncludingSetup']=1000*row['wallSeconds']/row['steps']
        results.append(row);(output/'result.json').write_text(json.dumps(row,indent=2)+'\n')
        (root/'results.json').write_text(json.dumps(results,indent=2)+'\n')
        print('RESULT '+json.dumps(row),flush=True)
    if 'boundary' in args.sections:
        schemes=[(s,None) for s in ['CRKSPH','CompSPH','Monaghan','GSPH','InutsukaGSPH','MFM','MFV']]+[('PESPH','PressureEnergy'),('PESPH','PressureEntropy')]
        for nx in args.nx:
            for dt in args.dt:
                for scheme,variant in schemes:
                    tableaus=['rungeKutta2'] if scheme in ('MFM','MFV') else ['rungeKutta2','rungeKutta4']
                    for tableau in tableaus:
                        label=f'boundary-{scheme}-{variant or "native"}-{nx}-{dt}-{tableau}'
                        run('boundary',label,lambda out:boundary_toy(scheme,variant,nx,dt,args.time,tableau,'uniform',out))
                for variant in ['PressureEnergy','PressureEntropy']:
                    label=f'boundary-PESPH-{variant}-{nx}-{dt}-density-wave'
                    run('boundary',label,lambda out:boundary_toy('PESPH',variant,nx,dt,args.time,'rungeKutta2','density-wave',out))
    if 'conduction' in args.sections:
        for nx in args.nx:
            for amplitude in [.01,.1]:
                for term in ['Price2012_98','Price2008','Hopkins2015']:
                    label=f'conduction-rate-{nx}-{amplitude}-{term}'
                    run('conduction',label,lambda out:conduction_toy(nx,amplitude,term,.0005,0.,out))
        for dt in [.0005,.00025]:
            for term in ['Price2012_98','Price2008']:
                label=f'conduction-evolve-64-{dt}-{term}'
                run('conduction',label,lambda out:conduction_toy(64,.1,term,dt,args.time,out))
        for term in ['Price2012_98','Price2008','Hopkins2015']:
            run('conduction',f'conduction-isobaric-64-{term}',lambda out:conduction_toy(64,.1,term,.0005,0.,out,True))
    if 'reservoir' in args.sections:run('reservoir','reservoir-volume',reservoir_toy)
    provenance=dict(precision=args.precision,torch=torch.__version__,device='cpu',threads=torch.get_num_threads(),
        input=vars(args)|{'root':str(root)},scriptSHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        head=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),
        sourceSHA256=before,sourceUnchanged=before==inventory(),
        scope='diagnostic variants, no solver defaults or physical diffusion implemented')
    (root/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(f'COMPLETE {len(results)} controls; {sum("error" in r for r in results)} execution errors',flush=True)
    return int(any('error' in r for r in results))


if __name__=='__main__':raise SystemExit(main())
