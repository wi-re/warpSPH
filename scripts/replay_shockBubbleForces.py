"""Actual Warp pressure/AV/dudt replay on fixed native stage fields.

Float64 isolates operator algebra. Imported coefficients versus rebuilding
Warp coefficients is a controlled distinction; supports/volumes/P/v/Q-grad
are held fixed. No support solve, density replacement or time integration.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from compare_spheralMaterials import save
from shockBubbleForceAudit import coefficients,native_split


def replay(path,out,device,one_sided_work=False):
    from warpSPHBootstrap import bootstrap
    bootstrap(precision='float64')
    import torch
    from warpSPHCore import (ParticleState,DomainDescription,KernelFunctions,SupportScheme,
        OperationProperties,CRKState,buildVerletList)
    from warpSPH.configurations import CRKSPHConfig
    from warpSPH.configurations.crkSPH import resolveCRKLimiter
    from warpSPH.modules.crk.accel import computeCrkSPHAccelWarp
    from warpSPH.modules.crk.dudt import computeCrkSPHdudtWarp
    from warpSPH.modules.crk.terms import crkFactors
    out.mkdir(parents=True,exist_ok=False);a=dict(np.load(path));source='Spheral' if 'coefficients' in a else 'warpSPH'
    a={k:v.astype(float) if np.issubdtype(v.dtype,np.floating) else v for k,v in a.items()}
    native=native_split(a);save(out/'nativeChecks.json',native['checks'])
    T=lambda v:torch.as_tensor(v,dtype=torch.float64,device=device).contiguous()
    s=ParticleState(T(a['positions']),T(a['supports']),T(a['masses']),torch.zeros(len(a['masses']),dtype=torch.int32,device=device),T(a['densities']))
    for field in ('velocities','internalEnergies','pressures','soundspeeds'):
        setattr(s,field,T(a[field]))
    s.alphas=torch.ones_like(s.densities)
    if source=='Spheral':
        assert np.max(abs(a['H'][:,0,1]))<1.e-12 and np.max(abs(a['H'][:,0,0]-a['H'][:,1,1]))<1.e-12
        lo=a['positions'].min(0)-2*a['supports'].max();hi=a['positions'].max(0)+2*a['supports'].max();periodic=[False,False]
    else:lo=[0.,0.];hi=[2.2,.8];periodic=[True,True]
    domain=DomainDescription(T(lo),T(hi),torch.tensor(periodic,device=device),2)
    adj=buildVerletList(s,domain,supportMode=SupportScheme.SuperSymmetric,verletScale=1.01)
    c=coefficients(a);imported=CRKState(**{k:T(v) for k,v in c.items()});V=T(a['volumes'])
    _,_,rebuilt=crkFactors(s,domain,KernelFunctions.CubicSpline,adjacency=adj,volumes=V,computeDensity=False)
    config=CRKSPHConfig();config.diffusionParams.C_l=1.;config.diffusionParams.C_q=.75
    params=resolveCRKLimiter(config.crkViscosityParams,3.02);params.viscosityKernelScale=2. if source=='Spheral' else 0.
    params.oneSidedWork=one_sided_work
    args=dict(queryParticles=s,operationProperties=OperationProperties(kernel=KernelFunctions.CubicSpline,supportMode=SupportScheme.KernelMeanSymmetric),
        domain=domain,conductivityParams=config.diffusionParams,crkViscosityParams=params,queryVelocityTensor=T(a['qGradient']),
        queryEnergies=T(a['internalEnergies']),queryVelocities=T(a['velocities']),queryCs=T(a['soundspeeds']),
        queryAlphas=torch.ones_like(s.densities),queryPressures=T(a['pressures']),queryVolumes=V,adjacency=adj)
    I,J=adj.livePairs();i,j=I.detach().cpu().numpy(),J.detach().cpu().numpy()
    for mode,crk in (('imported',imported),('rebuilt',rebuilt)):
        acc,ap,av=computeCrkSPHAccelWarp(**args,crkState=crk)
        du=computeCrkSPHdudtWarp(**args,crkState=crk)
        host=lambda x:x.detach().cpu().numpy().copy()
        ap,av=host(ap)[:len(i)],host(av)[:len(i)]
        p=np.zeros_like(a['positions'],dtype=float);q=np.zeros_like(p);np.add.at(p,i,ap);np.add.at(q,i,av)
        fields=dict(pressure=p,viscous=q,acceleration=host(acc),dudt=host(du),i=i,j=j,pressurePair=ap,viscousPair=av,
            **{k:host(v) for k,v in vars(crk).items()})
        np.savez_compressed(out/(mode+'.npz'),**fields)
        interface=a['real']&(a['positions'][:,0]>.4)&(a['positions'][:,0]<.85)&(a['positions'][:,1]>.04)&(a['positions'][:,1]<.36)
        if source=='warpSPH':interface &= a['UIDs']<2816
        relative=lambda x,y:float(np.linalg.norm((x-y)[interface])/max(np.linalg.norm(y[interface]),1.e-300))
        checks=dict(pressureRelativeL2=relative(p,native['pressure']),viscousRelativeL2=relative(q,native['viscous']),
            accelerationRelativeL2=relative(fields['acceleration'],a['nativeAcceleration']),dudtRelativeL2=relative(fields['dudt'],a['nativeDudt']),
            pairReductionRelativeL2=relative(p+q,fields['acceleration']))
        save(out/(mode+'Checks.json'),checks);print(path.name,source,mode,checks,flush=True)
    save(out/'provenance.json',dict(input=str(path),source=source,device=device,precision='float64',
        oneSidedWork=one_sided_work,limits='Fixed fields, imported or rebuilt RK coefficients; not a trajectory. Spheral ghosts explicit/nonperiodic; Warp uses saved periodic tiles. Common interface window excludes physical walls. Q gradient and length imported per source.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--one-sided-work',action='store_true')
    a=p.parse_args();replay(a.input,a.output,a.device,a.one_sided_work)
