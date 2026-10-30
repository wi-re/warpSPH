#!/usr/bin/env python3
"""Run the analytic oscillating-shear body-force control on native compressible hydro.

No viscosity is prescribed. Errors include the native method's numerical
transport; total energy conservation alone is not an accuracy certificate.
"""
from pathlib import Path
from dataclasses import replace, asdict
import argparse, hashlib, importlib.util, json, sys, time, traceback
repo=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--root',type=Path,required=True)
p.add_argument('--runtime-root',type=Path,default=repo)
p.add_argument('--scheme',choices=['CRKSPH','MFM','MFV'],default='CRKSPH')
p.add_argument('--nx',type=int,default=64)
p.add_argument('--amplitude',type=float,default=.2)
p.add_argument('--period',type=float,default=2.)
p.add_argument('--mode',type=int,default=1)
p.add_argument('--device',default='cuda:0')
args=p.parse_args();runtime=args.runtime_root.resolve()
sys.path[:0]=[str(runtime/'src'),str(runtime/'scripts')]
from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')
import numpy as np
import torch
from warpSPH.runner import RunConfiguration,resolvedConfiguration
from warpSPH.cases.compressible.drivenTurbulence import drivenTurbulenceCase as base
from run_compressibleMatrix import capture_sources,source_inventory,source_paths
from compressible_matrix import clean,write_json
source=repo/'src/warpSPH/modules/forcing/analytic.py'
spec=importlib.util.spec_from_file_location('shear_control',source)
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
field=module.OscillatingShear(amplitude=args.amplitude,period=args.period,mode=args.mode)
root=args.root.resolve();root.mkdir(parents=True,exist_ok=False)
(root/'analytic.py').write_bytes(source.read_bytes());(root/'driver.py').write_bytes(Path(__file__).read_bytes())
write_json(root/'control.json',dict(type='oscillatingTransverseShear',parameters=asdict(field),
    formula='v_y=A sin(omega t) sin(kx), a_y=A omega cos(omega t) sin(kx)',
    continuumKineticEnergy='M A^2 sin^2(omega t)/4',
    continuumWork='delta K',thermalModel='adiabatic',physicalViscosity=None,
    analyticSHA256=hashlib.sha256(source.read_bytes()).hexdigest()))
configuration=dict(schemaVersion=1,case='drivenTurbulence',
    simulation=dict(device=args.device,precision='float32',dim=2,L=1.,periodic=True,
        kernel='B7',n_h=4.,nx=args.nx,nSteps=None,tLimit=args.period,
        plot=True,video=True,store=False,show=False,quiet=True,progress=True,
        plotBackend='vispy',plotBackendOptions={'app_backend':'egl','jupyter_backend':'image'},
        plotInterval=100,maxDt=.01,exportRoot=str(root/'exports'),cudaGraph=False,
        execution='eager',velocityAlarmStopRatio=20.,stallProgress=1e-3),
    params=dict(gamma=5/3,rho0=1.,soundSpeed=1.,accelerationRMS=0.,
        thermalModel='adiabatic',removeMeanAcceleration=True),
    solver=dict(scheme=args.scheme,**(dict(temporalScheme='ssprk',cfl=.1) if args.scheme in ('MFM','MFV') else {})))
write_json(root/'input.json',configuration)
roots=source_paths();before=capture_sources(root,roots)
latest={};history=[];stats=dict(count=0,bad=0,minEnergy=float('inf'),minPressure=float('inf'))
hf=(root/'history.jsonl').open('w',buffering=1)

def arrays(system):
    s=system.state
    return {k:getattr(s,k).detach().cpu().numpy() for k in ('positions','velocities','densities','masses','internalEnergies','pressures','Q','supports','sourceEnergyBudget')}

def build(ctx):
    system=base.buildSystem(ctx)
    system.sourceTerms.acceleration=field
    ctx.scratch['driving']=field
    ctx.config.dt=base.timestep(ctx,system)
    s=system.state;latest.update(x0=s.positions.clone(),u0=s.internalEnergies.clone(),rho0=s.densities.clone(),p0=s.pressures.clone(),mass0=s.masses.clone(),start=time.perf_counter())
    np.savez_compressed(root/'initial.npz',**arrays(system))
    write_json(root/'resolved.json',resolvedConfiguration(ctx))
    native=ctx.stepFunction
    def observed(s,dt,stageIndex=None,**kw):
        if stageIndex is not None:kw['stageIndex']=stageIndex
        update,*rest=native(s,dt,**kw)
        stage=getattr(s,'meshlessContext',None);state=s.state
        u,P,rho=(stage.u,stage.P,stage.rho) if stage is not None else (state.internalEnergies,state.pressures,state.densities)
        rates=[update.dQdt] if stage is not None else [update.dvdt,update.dudt,update.dEdt]
        probes=[u.min(),P.min()]+[torch.isfinite(v).all().to(u) for v in (u,P,rho,state.positions,state.Q,*rates)]
        val=torch.stack(probes).detach().cpu().tolist();stats['count']+=1
        stats['minEnergy']=min(stats['minEnergy'],val[0]);stats['minPressure']=min(stats['minPressure'],val[1]);stats['bad']+=int(val[0]<=0 or val[1]<=0 or not all(val[2:]))
        return (update,*rest)
    ctx.bundle=replace(ctx.bundle,stepFunction=observed)
    return system

def diagnostics(ctx,system):
    row=base.diagnostics(ctx,system);s=system.state;t=float(system.t)
    # Eulerian exact velocity and Lagrangian exact displacement are independent checks.
    exact=field.velocity(s.positions,t);displacement=s.positions-latest['x0']-field.displacement(latest['x0'],t)
    displacement=torch.remainder(displacement+.5,1.)-.5
    m=s.masses.double();mass=m.sum();err=(s.velocities-exact).double()
    sine=torch.sin(2*torch.pi*field.mode*s.positions[:,0]).double()
    referenceK=field.kineticEnergy(t,float(mass));_,power,heat=system.sourceTerms.fields(s.masses,s.positions,s.velocities,s.internalEnergies,t)
    vals=torch.stack(((m[:,None]*err.square()).sum().sqrt()/mass.sqrt(),
        (m[:,None]*displacement.double().square()).sum().sqrt()/mass.sqrt(),
        (m*(s.internalEnergies-latest['u0']).double()).sum(),
        (s.densities-latest['rho0']).double().abs().max(),
        (s.pressures-latest['p0']).double().abs().max(),
        (m*s.velocities[:,1].double()*sine).sum()/(m*sine.square()).sum(),
        power.double().sum(),heat.double().sum(),
        (m[:,None]*s.velocities.double()).sum(0).norm(),
        (s.masses-latest['mass0']).double().abs().max())).cpu().tolist()
    row.update(zip(('velocityL2','displacementL2','internalEnergyChange','maxDensityChange','maxPressureChange','fittedAmplitude','instantaneousSourcePower','instantaneousBathPower','momentumNorm','maxParticleMassChange'),vals))
    row.update(t=t,exactKineticEnergy=referenceK,exactAmplitude=field.amplitude*np.sin(field.omega*t),kineticEnergyError=row['kineticEnergy']-referenceK,workError=row['forceWork']-referenceK)
    history.append(row);hf.write(json.dumps(clean(row))+'\n');return row

record=dict(status='running')
try:
    case=replace(base,buildSystem=build,diagnostics=diagnostics,extraData=None,trajectoryData=None)
    r=RunConfiguration.fromDict(configuration).run(case)
    if args.device.startswith('cuda'):torch.cuda.synchronize()
    elapsed=time.perf_counter()-latest['start'];np.savez_compressed(root/'final.npz',**arrays(r.state))
    after={k:v[1] for k,v in source_inventory(roots).items()}
    positive=all(row['minEnergy']>0 and row['minPressure']>0 and row['minDensity']>0 and all(np.isfinite(v) for v in row.values() if isinstance(v,(int,float))) for row in history)
    good=not r.diverged and not r.stopReason and positive and stats['bad']==0 and float(r.state.t)==args.period and before==after
    record=dict(status='completed' if good else 'failed',scheme=args.scheme,nx=args.nx,nSteps=r.nSteps,actualTime=float(r.state.t),targetTime=args.period,loopWallSeconds=elapsed,msPerStep=1000*elapsed/r.nSteps,stageChecks=stats,extraRHS=stats['count']-2*r.nSteps,acceptedPositiveFinite=positive,sourcesStable=before==after,maxAbsoluteEnergyBudgetResidual=max(abs(row['relativeEnergyBudgetResidual']) for row in history),maxVelocityL2=max(row['velocityL2'] for row in history),maxWorkError=max(abs(row['workError']) for row in history),maxKineticEnergyError=max(abs(row['kineticEnergyError']) for row in history),final=history[-1],stopReason=r.stopReason)
except Exception:
    record.update(status='error',error=traceback.format_exc());raise
finally:
    hf.close();write_json(root/'result.json',record);print(json.dumps(clean(record)),flush=True)
if record['status']!='completed':raise SystemExit(2)
