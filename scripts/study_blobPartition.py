"""Matched raw-predictor entropy/dudt partition controls on the wind-cloud blob.

Both use Warp CRKSPH. This isolates the work partition, not full solver parity
with the paper or current compiled Spheral. Every trajectory uses daemon/video.
"""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from compare_spheralMaterials import save,warp_run


def prepare(args):
    from warpSPH.caseUtils.compressible.blob import sampleBlob,blobScales
    scales=blobScales();dt=args.dt or .64/args.nx;steps=math.ceil(args.tkh*scales['tKH']/dt)
    a=sampleBlob(args.nx);args.output.mkdir(parents=True,exist_ok=False)
    for partition,policy in (('entropy','native'),('dudt','spheral'),('entropy-common-work','entropy')):
        folder=args.output/partition;folder.mkdir();np.savez_compressed(folder/'input.npz',**a)
        save(folder/'settings.json',dict(case='blob',gammaVariant='nx'+str(args.nx)+'-'+partition,
            nx=args.nx,dim=2,gamma=[5/3,5/3],end=steps*dt,dt=dt,steps=steps,frameEvery=args.frame_every,
            n_h=4.05,kernel='QuinticSpline',volumePolicy='massOverDensity',support='adaptive',
            domainAxes=[[0.,40.],[0.,10.]],boundary='periodic',energyPartition=policy,
            energyPredictor='raw',oneSidedWork=False,requestedTKH=args.tkh,**scales,
            source='Frontiere 2017 sec4.5.1 geometry; pinned Spheral BlobTest-2d rounded cloud lattice. Raw predictor fixed, only final pair-work partition differs. Native Warp support/Q, same-material density and experimental mass/rho evolution fixed. No claim of full paper/Spheral operator parity.'))
    print('Prepared',len(a['masses']),'particles;',steps,'steps to',steps*dt,'=',steps*dt/scales['tKH'],'tKH',flush=True)


def context(settings):
    from warpSPHBootstrap import bootstrap
    bootstrap(precision='float32')
    from warpSPH.runner import RunConfiguration
    from warpSPH.cases.compressible.blob import blobCase
    params=dict(viscositySwitch='NoneSwitch',adaptiveSupportScheme='Owen',massMatch=True,initialDensity='prescribed')
    solver=dict(crkDensityPolicy='sameMaterial',crkEvolution='spheral',crkEnergyPartition=settings['energyPartition'],
        crkEnergyPredictor=settings['energyPredictor'],crkViscosityParams={'oneSidedWork':settings['oneSidedWork'],'viscosityKernelScale':0.},diffusionParams={'C_l':1.,'C_q':1.})
    sim=dict(nx=settings['nx'],tLimit=settings['end'],nSteps=settings['steps'],kernel='QuinticSpline',n_h=4.05,
        dt=settings['dt'],adaptiveDt=False,execution='eager',cflFactor=.15)
    ctx=RunConfiguration('blob',simulation=sim,params=params,solver=solver).prepare(blobCase)
    return blobCase,ctx,sim,params,solver


class Measures:
    def __init__(self,settings,a):
        self.settings=settings;m=a['masses'].astype(float);v=a['velocities'].astype(float);u=a['internalEnergies'].astype(float)
        self.energy=float(np.dot(m,u+.5*np.sum(v*v,axis=1)));self.mass=np.array([m[a['materials']==k].sum() for k in (0,1)])
        self.momentum=np.sum(m[:,None]*v,axis=0)
    def __call__(self,a,t):
        m=a['masses'].astype(float);v=a['velocities'].astype(float);u=a['internalEnergies'].astype(float);rho=a['densities'].astype(float)
        cloud=a['materials']==1;cold=(rho>6.4)&(u<1.35);masscloud=m[cloud].sum()
        momentum=np.sum(m[:,None]*v,axis=0)
        return dict(time=float(t),tKH=float(t/self.settings['tKH']),
            relativeEnergyDrift=float(np.dot(m,u+.5*np.sum(v*v,axis=1))/self.energy-1),
            relativeMomentumError=float(np.linalg.norm(momentum-self.momentum)/np.linalg.norm(self.momentum)),
            materialMassError=float(np.max(abs(np.array([m[a['materials']==k].sum() for k in (0,1)])/self.mass-1))),
            minDensity=float(rho.min()),minEnergy=float(u.min()),minPressure=float(a['pressures'].min()),maxPressure=float(a['pressures'].max()),
            maxVelocity=float(np.linalg.norm(v,axis=1).max()),cloudMassFraction=float(m[cloud&cold].sum()/masscloud),
            cloudDenseFraction=float(m[cloud&(rho>6.4)].sum()/masscloud),cloudMeanEnergy=float(np.dot(m[cloud],u[cloud])/masscloud))


def summary(root,solver,settings,history,initial,final,wall,steps,remote,extra):
    maxima={k:float(max(abs(r[k]) for r in history)) for k in history[0] if k!='time'}
    gates=dict(completed=abs(history[-1]['time']-settings['end'])<=settings['dt']*1.01,
        finitePositive=all(np.isfinite(v) for row in history for v in row.values()) and all(min(r[k] for r in history)>0 for k in ('minDensity','minEnergy','minPressure')),
        energy=maxima['relativeEnergyDrift']<1e-4,momentum=maxima['relativeMomentumError']<1e-5,
        materialMass=maxima['materialMassError']<1e-6,identities=all(np.array_equal(initial[k],final[k]) for k in ('UIDs','materials','masses')))
    result=dict(settings=settings,solver=solver,remoteRunID=remote,time=history[-1]['time'],steps=steps,wallSeconds=wall,
        meanStepMilliseconds=1000*wall/max(steps,1),maxima=maxima,final=history[-1],gates={k:bool(v) for k,v in gates.items()},passed=all(gates.values()),
        qualificationScope=('Operational checks only; compiled Spheral with native energy/support/Q policies, same saved initial arrays as Warp; no exact nonlinear blob oracle.' if solver=='Spheral' else
            'Operational checks only; paired final-work policies with raw predictor fixed, no exact nonlinear blob oracle.'),**extra)
    for name,a in [('initialState',initial),('finalState',final)]:np.savez_compressed(root/(name+'.npz'),**a)
    save(root/'diagnosticHistory.json',history);save(root/'comparisonRun.json',result);print(json.dumps(result,indent=2),flush=True);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('prepare','run'));p.add_argument('--output',type=Path,required=True)
    p.add_argument('--nx',type=int,default=256);p.add_argument('--tkh',type=float,default=8.);p.add_argument('--dt',type=float)
    p.add_argument('--frame-every',type=int,default=100);p.add_argument('--partition',choices=('entropy','dudt','entropy-common-work'))
    args=p.parse_args()
    if args.mode=='prepare':
        # Geometry-only preparation imports no Warp runtime-dependent types.
        from warpSPHBootstrap import bootstrap
        bootstrap(precision='float32')
        prepare(args)
    else:
        if args.partition is None:p.error('--partition required for run')
        args.output=args.output/args.partition;settings=json.loads((args.output/'settings.json').read_text())
        raise SystemExit(warp_run(args,settings,context,Measures,summary,'Blob partition'))
