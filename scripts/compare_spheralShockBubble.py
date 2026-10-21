"""Actual compiled Spheral shock-bubble versus reflected-periodic warpSPH.

Uniform lattice source variant, two physical materials, common cubic kernel.
Native reference walls and four Warp reflection tiles; no force modifications.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from compare_spheralMaterials import PIN, SOURCE, FIELDS, save, warp_run, spheral_run, render


def reflect(a):
    blocks = []
    for axis_x, axis_y in ((0, 0), (0, 1), (1, 0), (1, 1)):
        b = {k: v.copy() for k, v in a.items()}
        for axis, flag, extent in ((0, axis_x, 1.1), (1, axis_y, .4)):
            if flag:
                b['positions'][:, axis] = 2*extent-b['positions'][:, axis]
                b['velocities'][:, axis] *= -1
        b['UIDs'] += len(blocks)*len(a['masses']); blocks.append(b)
    return {k: np.concatenate([b[k] for b in blocks]) for k in a}


def primary(a, count):
    mask = a['UIDs'] < count
    b = {k: v[mask].copy() for k, v in a.items()}
    order = np.argsort(b['UIDs'])
    return {k: v[order] for k, v in b.items()}


def prepare(args):
    n = args.n
    if n < 40 or n % 20: raise ValueError('n must be >=40 and divisible by20 to tile both walls exactly')
    nx, ny = 11*n//10, 2*n//5
    xx, yy = np.meshgrid((np.arange(nx)+.5)/n, (np.arange(ny)+.5)/n, indexing='ij')
    x = np.column_stack((xx.ravel(), yy.ravel())).astype(np.float32)
    material = (np.linalg.norm(x-np.array([.8, .2]), axis=1)<.1).astype(np.int32)
    order = np.argsort(material, kind='stable'); x, material = x[order], material[order]
    rho = np.where(material==0, 1., .182); P = np.where(x[:, 0]>1., args.driver, 1.)
    gamma = np.array([1.4, 1.648])[material]
    a = dict(positions=x, velocities=np.zeros_like(x), densities=rho.astype(np.float32),
        pressures=P.astype(np.float32), internalEnergies=(P/((gamma-1)*rho)).astype(np.float32),
        supports=np.full(len(x), 3.02/n, dtype=np.float32), masses=(rho/n**2).astype(np.float32),
        materials=material, UIDs=np.arange(len(x), dtype=np.int64), kinds=np.zeros(len(x), dtype=np.int32))
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output/'input.npz', **a)
    np.savez_compressed(args.output/'warpInput.npz', **reflect(a))
    settings = dict(case='shockBubble', gammaVariant='n'+str(n), gamma=[1.4, 1.648], nx=n,
        dim=2, end=args.time, dt=args.dt or .016/n, support='adaptive', frameEvery=args.frame_every,
        referencePin=PIN, sourceRoot=str(SOURCE), domain=[0., 1.1], domainAxes=[[0., 1.1], [0., .4]],
        boundary='reflecting', kernel='CubicSpline', extent=2., nPerh=1.51, n_h=3.02,
        primaryParticles=len(x), volumePolicy=args.volume, driverPressure=args.driver,
        referenceViscosity=dict(Cl=1., Cq=.75, epsilon2=.01),
        precision=dict(warp='float32', spheral='float64'),
        provenance='Pressure-driven Spheral ShockBubble-Variant-2d, nondimensional rhoAir=1, pAmbient=1, driver=10, rhoHe=.182. Uniform lattice (air2He1=air2He2=1, HeSeed=lattice); two physical materials merge both air regions. Native reflecting Spheral walls versus four Warp reflected-periodic copies. XSPH disabled in both; native support/limited-AV/compatible-work details differ. Controller MassOverDensity default retained unless explicitly selected otherwise.')
    save(args.output/'settings.json', settings)
    print('Prepared primary',len(x),'Warp',4*len(x),'dt',settings['dt'],flush=True)


class Measures:
    def __init__(self, settings, a):
        self.settings=settings; self.count=settings['primaryParticles']
        mass=a['masses'].astype(float); u=a['internalEnergies'].astype(float); v=a['velocities'].astype(float)
        self.energy=float(np.sum(mass*(u+.5*np.sum(v*v,axis=1))))
        self.mass=np.array([mass[a['materials']==k].sum() for k in (0,1)])
    def __call__(self, a, t):
        m=a['masses'].astype(float);v=a['velocities'].astype(float);u=a['internalEnergies'].astype(float)
        b=primary(a,self.count); bubble=b['materials']==1; w=b['masses'][bubble].astype(float);x=b['positions'][bubble].astype(float)
        center=np.sum(w[:,None]*x,axis=0)/w.sum()
        return dict(time=float(t),relativeEnergyDrift=float(np.sum(m*(u+.5*np.sum(v*v,axis=1)))/self.energy-1),
            materialMassError=float(np.max(abs(np.array([m[a['materials']==k].sum() for k in (0,1)])/self.mass-1))),
            minDensity=float(a['densities'].min()),minEnergy=float(u.min()),minPressure=float(a['pressures'].min()),
            maxVelocity=float(np.linalg.norm(v,axis=1).max()), maxPressure=float(a['pressures'].max()),
            bubbleCenterX=float(center[0]),bubbleCenterY=float(center[1]),
            bubbleWidth=float(np.ptp(x[:,0])),bubbleHeight=float(np.ptp(x[:,1])),
            bubbleVolume=float(np.sum(b['masses'][bubble]/b['densities'][bubble])),
            bubbleMeanPressure=float(np.average(b['pressures'][bubble],weights=w)),
            wallPenetration=float(max(0.,-b['positions'].min(),np.max(b['positions'][:,0]-1.1),np.max(b['positions'][:,1]-.4))))


def summary(root,solver,settings,history,initial,final,wall,steps,remote,extra):
    maxima={k:float(max(abs(r[k]) for r in history)) for k in history[0] if k!='time'}
    gates=dict(completed=abs(history[-1]['time']-settings['end'])<=settings['dt']*1.01,
        finitePositive=all(np.isfinite(v) for r in history for v in r.values()) and all(min(r[k] for r in history)>0 for k in ('minDensity','minEnergy','minPressure')),
        energy=maxima['relativeEnergyDrift']<1.e-4, materialMass=maxima['materialMassError']<1.e-6,
        identities=all(np.array_equal(initial[k],final[k]) for k in ('UIDs','materials','masses')),
        boundedVelocity=maxima['maxVelocity']<20.,walls=maxima['wallPenetration']<.25/settings['nx'])
    row=dict(solver=solver,settings=settings,remoteRunID=remote,time=history[-1]['time'],steps=steps,
        wallSeconds=wall,meanStepMilliseconds=1000*wall/max(1,steps),maxima=maxima,final=history[-1],
        gates={k:bool(v) for k,v in gates.items()},passed=all(gates.values()),
        qualificationScope='Operational conservation/positivity/wall checks only. Cross-code deformation/circulation agreement measured separately; no exact nonlinear oracle.',
        timingScope='Reference integration/diagnostics/publishing; excludes startup and video.' if solver=='Spheral' else 'Runner wall includes initialization, rendering and video; step-only series separate.',**extra)
    np.savez_compressed(root/'initialState.npz',**initial);np.savez_compressed(root/'finalState.npz',**final)
    save(root/'diagnosticHistory.json',history);save(root/'comparisonRun.json',row)
    print(json.dumps(row,indent=2),flush=True);return row


def context(settings):
    from warpSPHBootstrap import bootstrap
    bootstrap(precision='float32')
    from warpSPH.runner import RunConfiguration
    from warpSPH.cases.compressible.shockBubble import shockBubbleCase
    from study_materialCases import materials
    params=dict(viscositySwitch='NoneSwitch',adaptiveSupportScheme='Owen',driverPressure=settings['driverPressure'])
    solver=dict(materials=materials(*settings['gamma']),crkDensityPolicy='sameMaterial',
        crkEvolution='summation' if settings['volumePolicy']=='kernelSum' else 'spheral',diffusionParams={'C_l':1.,'C_q':.75})
    sim=dict(nx=settings['nx'],tLimit=settings['end'],nSteps=round(settings['end']/settings['dt']),kernel='CubicSpline',n_h=3.02,dt=settings['dt'],
        adaptiveDt=False,execution='eager',cflFactor=.15)
    ctx=RunConfiguration('shockBubble',simulation=sim,params=params,solver=solver).prepare(shockBubbleCase)
    return shockBubbleCase,ctx,sim,params,solver


def build(ctx,a,settings):
    import torch
    from warpSPH.caseUtils.compressible.materialValidation import buildMaterialSystem
    from warpSPH.enumTypes import AdaptiveSupportScheme
    ctx.schemeConfig.adaptiveSupportScheme=AdaptiveSupportScheme.NoScheme
    to=lambda k:torch.as_tensor(a[k],device=ctx.device)
    system=buildMaterialSystem(ctx,to('positions'),to('masses')/to('densities'),to('densities'),to('pressures'),to('materials'))
    ctx.schemeConfig.adaptiveSupportScheme=AdaptiveSupportScheme.Owen
    return system


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('prepare','spheral','warp','render'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--n',type=int,default=80)
    p.add_argument('--time',type=float,default=.3);p.add_argument('--dt',type=float)
    p.add_argument('--driver',type=float,default=10.);p.add_argument('--frame-every',type=int,default=25)
    p.add_argument('--volume',choices=('massOverDensity','kernelSum'),default='massOverDensity')
    p.add_argument('--publisher-path',type=Path,default=Path('/home/lu26029/dev/warpSPHRemote/python'));a=p.parse_args()
    if a.mode=='prepare':return prepare(a)
    settings=json.loads((a.output/'settings.json').read_text())
    if a.mode=='spheral':return spheral_run(a,settings,Measures,summary,'Shock bubble')
    if a.mode=='warp':return warp_run(a,settings,context,Measures,summary,'Shock bubble',build,'warpInput.npz')
    return render(a,settings,context)


if __name__=='__main__':raise SystemExit(main())
