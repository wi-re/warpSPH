#!/usr/bin/env python3
"""Offline 2D/3D Favre budgets from retained qualification samples and forcing.pt.

NumPy analysis loads its utility without importing the hydrodynamic runtime.
Requires input.json, samples/*.npz (t, positions, masses, densities, velocities,
pressures, internalEnergies) and the complete past forcing checkpoint. Existing
accepted-step history supplies an independent native work/bath budget when
available. No simulations or CUDA allocations are performed.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import torch
from audit_drivenForcing import independent, evaluate

UTILITY=Path(__file__).resolve().parents[1]/'src/warpSPH/utils/turbulenceBudget.py'
spec=importlib.util.spec_from_file_location('turbulence_budget_numpy',UTILITY)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
PeriodicBudget,depositConservative=module.PeriodicBudget,module.depositConservative


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def average(t, values, lo, hi):
    t=np.asarray(t);value=np.asarray(values)
    clock=np.r_[lo,t[(t>lo)&(t<hi)],hi]
    return float(np.trapezoid(np.interp(clock,t,value),clock)/(hi-lo))


def summary(series, lo, hi, stride=1):
    indices=list(range(0,len(series['t']),stride))
    if indices[-1]!=len(series['t'])-1:indices.append(len(series['t'])-1)
    t=series['t'][indices]
    means={key:average(t,np.asarray(value)[indices],lo,hi) for key,value in series.items() if key!='t'}
    delta=(np.interp(hi,t,series['kineticEnergyDensity'][indices])-np.interp(lo,t,series['kineticEnergyDensity'][indices]))/(hi-lo)
    means.update(observedKineticRate=float(delta),unclosedResidual=float(delta-means['predictedRate']),sampleStride=stride)
    if 'predictedObservedRate' in means:
        means['observationCorrectedResidual']=float(delta-means['predictedObservedRate'])
    return means


def analyze(run, output, grids, radii, scales, window):
    start=time.perf_counter();cfg=json.loads((run/'input.json').read_text())
    params=cfg['params'];length=cfg['simulation']['L']
    state=torch.load(run/'forcing.pt',map_location='cpu',weights_only=True)
    if state['parameters']['length']!=length:raise ValueError('forcing/domain lengths differ')
    paths=sorted((run/'samples').glob('*.npz'))
    if len(paths)<8:raise ValueError('need at least8 saved states')
    header=dict(path=str(run),inputSHA256=digest(run/'input.json'),forcingSHA256=digest(run/'forcing.pt'))
    dim=state['parameters']['dim'];combos={};ledger=[];evidence={}
    output.mkdir(parents=True,exist_ok=False)
    for n in grids:
        for radius in radii:
            if radius<2*length/n:raise ValueError('observation radius must span >=2 cells on every grid')
            combos[n,radius]=dict(engine=PeriodicBudget((n,)*dim,length),rows=[],metadata=[],kernelSeconds=0.)
    for index,path in enumerate(paths):
        with np.load(path) as item:s=dict(item)
        t=float(s['t']);x=s['positions'].astype(float);m=s['masses'].astype(float);v=s['velocities'].astype(float)
        if x.shape[1]!=dim:raise ValueError('sample/forcing dimensions differ')
        modes,c=independent(state,t);a,_=evaluate(x,modes,c,length)
        if params.get('removeMeanAcceleration',True):a-=np.sum(m[:,None]*a,axis=0)/m.sum()
        u=s['internalEnergies'].astype(float)
        if params.get('thermalModel')=='relaxation':
            target=params['soundSpeed']**2/(params['gamma']*(params['gamma']-1))
            heat=(target-u)/params['coolingTime']
        elif params.get('thermalModel')=='adiabatic':heat=np.zeros_like(u)
        else:raise ValueError('declare relaxation or adiabatic thermal model')
        native=dict(t=t,kineticEnergy=float(.5*np.sum(m[:,None]*v*v)),internalEnergy=float(np.sum(m*u)),
            instantaneousSourcePower=float(np.sum(m[:,None]*v*a)),instantaneousBathPower=float(np.sum(m*heat)),
            maxNativeSupport=float(np.max(s['supports'])) if 'supports' in s else None)
        ledger.append(native);evidence[path.name]=digest(path)
        for (n,radius),entry in combos.items():
            st=time.perf_counter()
            grid=depositConservative(x,m,v,s['densities'],s['pressures'],u,a,gridSize=n,length=length,radius=radius,heatRate=heat)
            entry['kernelSeconds']+=time.perf_counter()-st
            if grid['metadata']['emptyCellFraction']>0:raise ValueError('empty observation cells: no spectral imputation allowed')
            entry['metadata'].append(grid['metadata']);rho,vel,p,f=[grid[k] for k in ('density','velocity','pressure','forceDensity')]
            engine=entry['engine'];drho,dmom=engine.inviscidRHS(rho,vel,p,f)
            rows=[]
            for ell in scales:
                measurement=engine.measure(rho,vel,p,ell=ell,forceDensity=f,observationMomentumFlux=grid['momentumFlux'])
                rhs=engine.kineticRate(rho,vel,drho,dmom,ell)
                measurement.update(collocationRate=rhs,collocationIdentityResidual=rhs-measurement['predictedRate'])
                rows.append(measurement)
            entry['rows'].append(rows)
        if index%20==0 or index==len(paths)-1:
            print(json.dumps(dict(run=run.name,sample=index+1,total=len(paths),t=t,wallSeconds=time.perf_counter()-start)),flush=True)
    times=np.array([row['t'] for row in ledger]);lo,hi=window
    if np.any(np.diff(times)<=0) or not times[0]<=lo<hi<=times[-1]:raise ValueError('strict times and complete assessment window required')
    results=[]
    for (n,radius),entry in combos.items():
        table={key:np.array([[r[key] for r in row] for row in entry['rows']]) for key in entry['rows'][0][0]}
        np.savez_compressed(output/f'grid{n}-radius{radius:g}.npz',t=times,**table)
        records=[]
        for j,ell in enumerate(scales):
            series=dict(t=times,**{key:value[:,j] for key,value in table.items() if key not in ('ell','viscousStressProvided')})
            record=dict(ell=ell,means=summary(series,lo,hi),timeSampling=[summary(series,lo,hi,stride) for stride in (2,4)])
            record['blocks']=[dict(window=[a,b],means=summary(series,a,b)) for a,b in zip(np.linspace(lo,hi,4)[:-1],np.linspace(lo,hi,4)[1:])]
            records.append(record)
        meta=entry['metadata']
        results.append(dict(gridSize=n,observationRadius=radius,scales=records,mappingSeconds=entry['kernelSeconds'],
            maxMassError=max(abs(m['relativeMassError']) for m in meta),maxExtensiveConservationError=max(m['maxExtensiveConservationError'] for m in meta),
            meanUnresolvedKineticEnergy=average(times,[m['unresolvedKineticEnergy'] for m in meta],lo,hi),
            meanUnresolvedSourcePower=average(times,[m['unresolvedSourcePower'] for m in meta],lo,hi),
            maxUnresolvedKineticFraction=max(m['unresolvedKineticEnergy']/max(m['particleKineticEnergy'],1e-30) for m in meta)))
    nativeMeans={k:average(times,[row[k] for row in ledger],lo,hi) for k in ('kineticEnergy','internalEnergy','instantaneousSourcePower','instantaneousBathPower')}
    history=run/'history.jsonl'
    if history.exists():
        h=[json.loads(line) for line in history.open() if line.strip()];ht=np.array([r['t'] for r in h])
        if not ht[0]<=lo<hi<=ht[-1]:raise ValueError('native history does not cover assessment window')
        for key in ('forceWork','thermalExchange','kineticEnergy','totalEnergy'):
            val=np.array([r[key] for r in h]);nativeMeans[key+'Rate']=float((np.interp(hi,ht,val)-np.interp(lo,ht,val))/(hi-lo))
        nativeMeans['maxNativeEnergyBudgetResidual']=max(abs(r['relativeEnergyBudgetResidual']) for r in h)
        nativeMeans['sourceSamplingError']=nativeMeans['instantaneousSourcePower']-nativeMeans['forceWorkRate']
        nativeMeans['bathSamplingError']=nativeMeans['instantaneousBathPower']-nativeMeans['thermalExchangeRate']
    result=dict(**header,dim=dim,length=length,window=window,samples=len(paths),timeRange=[float(times[0]),float(times[-1])],
        medianSampleSpacing=float(np.median(np.diff(times))),native=nativeMeans,observations=results,
        currentDissipationModel='Not inferred. No viscous stress is supplied to this adapter; consult the saved simulation configuration and source. Bath model is reconstructed from declared params.',
        forceReconstruction='Saved coefficients at actual sample t, independent FP64 evaluation; actual mass-mean subtraction. Not the exact per-stage GPU half-kick work.',
        rateUnits='Spatial mean per volume. Positive totalTransfer removes resolved KE; pressureDilatation>0 feeds KE. No supplied viscous stress.',
        residualScope='Base residual excludes kinetic observation stress. observationCorrectedResidual includes its measured signed work; pressure reconstruction, native force/discrete transport, collocation and time sampling remain. Neither residual is a physical viscosity measurement.',
        deAliasing='None; smooth observation/coarse filters plus grid/time/filter sensitivity and Euler collocation identity checks.',
        observationAxes='ij: array axis j is physical coordinate j',sampleEvidence=evidence,
        driverSHA256=digest(Path(__file__)),utilitySHA256=digest(UTILITY),wallSeconds=time.perf_counter()-start)
    (output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (output/'particle-ledger.json').write_text(json.dumps(ledger,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(run=run.name,native=nativeMeans,wallSeconds=result['wallSeconds'])),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs',nargs='+',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--grids',nargs='+',type=int,default=[32,64])
    p.add_argument('--radii',nargs='+',type=float,default=[.0625,.09375])
    p.add_argument('--scales',nargs='+',type=float,default=[.0625,.125,.25])
    p.add_argument('--window',nargs=2,type=float,default=[10.,40.])
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    summaries=[analyze(run.resolve(),args.output/run.name,args.grids,args.radii,args.scales,args.window) for run in args.runs]
    (args.output/'summary.json').write_text(json.dumps(summaries,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
