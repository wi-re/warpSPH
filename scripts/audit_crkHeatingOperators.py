"""Fixed 2D pairs: production CRK/Comp heating versus independent work algebra.

No physical trajectory. Pressure-only and viscous pieces are measured separately;
the CompSPH gather predictor is not assumed to match its symmetric force.
"""
import argparse
import json
from pathlib import Path

from warpSPHBootstrap import bootstrap
bootstrap(precision='float64')
import numpy as np
import torch
import warp as wp
wp.config.kernel_cache_dir='/tmp/warpsph-heating-kernels'
from warpSPHCore import (scalar_t,domainData,ParticleState,DomainDescription,KernelFunctions,
    OperationProperties,SupportScheme,CRKState,buildVerletList)
from warpSPH.configurations import CRKSPHConfig,CompSPHConfig
from warpSPH.configurations.crkSPH import resolveCRKLimiter
from warpSPH.modules.crk.accel import computeCrkSPHAccelWarp
from warpSPH.modules.crk.dudt import computeCrkSPHdudtWarp
from warpSPH.modules.compSPH.accel import computeCompSPHAccelWarp
from warpSPH.modules.compSPH.dudt import computeCompSPHdudtWarp
from warpSPH.modules.compSPH.balance import computeCompSPHBalanceTermWarp
from warpSPH.modules.compSPH.pairWork import beginPairWork,accumulateStage,energyIncrement
from warpSPH.modules.crk.energyControl import entropyRate,spheralRate
from warpSPH.modules.dissipation import DiffusionParameters,computePi_actual,computeFrontiereQ
from warpSPH.enumTypes import EnergyScheme
from shockBubbleForceAudit import gradient
from types import SimpleNamespace as NS

Vec=wp.types.vector(length=2,dtype=scalar_t)


@wp.kernel
def viscous(x:wp.array(dtype=Vec),v:wp.array(dtype=Vec),R:wp.array(dtype=scalar_t),rho:wp.array(dtype=scalar_t),
            cs:wp.array(dtype=scalar_t),P:wp.array(dtype=scalar_t),domain:domainData,p:DiffusionParameters,
            out:wp.array2d(dtype=scalar_t)):
    n=wp.tid();i=n;j=i+1
    if i%2==1:j=i-1
    out[n,0]=computeFrontiereQ(x[i],x[j],R[i],R[j],rho[i],rho[j],v[i]-v[j],domain,
        wp.static(KernelFunctions.CubicSpline.value),cs[i],cs[j],scalar_t(1.),scalar_t(1.),p,False,scalar_t(2.))
    out[n,1]=computeFrontiereQ(x[i],x[j],R[i],R[j],rho[i],rho[j],v[i]-v[j],domain,
        wp.static(KernelFunctions.CubicSpline.value),cs[i],cs[j],scalar_t(1.),scalar_t(1.),p,True,scalar_t(2.))
    out[n,2]=computePi_actual(x[i],x[j],R[i],R[j],scalar_t(1.),scalar_t(1.),rho[i],rho[j],True,P[i],P[j],
        v[i],v[j],domain,wp.static(KernelFunctions.CubicSpline.value),cs[i],cs[j],scalar_t(1.),scalar_t(1.),p,False,False)
    out[n,3]=computePi_actual(x[i],x[j],R[i],R[j],scalar_t(1.),scalar_t(1.),rho[i],rho[j],True,P[i],P[j],
        v[i],v[j],domain,wp.static(KernelFunctions.CubicSpline.value),cs[i],cs[j],scalar_t(1.),scalar_t(1.),p,True,False)


def audit(root):
    root.mkdir(parents=True,exist_ok=True)
    rows=[];labels=[]
    for ratio in (1.,.65,1.4):
        for rhoj in (1.,.1,10.):
            for sign in (-1.,1.):
                labels.append(f'Rj/Ri={ratio:g} rhoj={rhoj:g} '+('compress' if sign<0 else 'expand'))
                rows.append((ratio,rhoj,sign))
    n=len(rows);x=np.array([[3*k,.2] for k in range(n) for _ in (0,1)],dtype=float);x[1::2]+=[.3,.11]
    R=np.array([[1.,a] for a,b,c in rows]).ravel();rho=np.array([[1.,b] for a,b,c in rows]).ravel()
    m=np.tile([.8,.2],n);V=np.tile([.7,.03],n);P=np.tile([2.,.3],n);u=P/((5/3-1)*rho)
    v=np.zeros_like(x);v[::2]=-np.array([c for a,b,c in rows])[:,None]*np.array([.4,.12])
    cs=np.sqrt((5/3)*P/rho)
    T=lambda a:torch.as_tensor(a,dtype=torch.float64).contiguous()
    s=ParticleState(T(x),T(R),T(m),torch.zeros(2*n,dtype=torch.int32),T(rho))
    s.velocities=T(v);s.internalEnergies=T(u);s.pressures=T(P);s.soundspeeds=T(cs);s.alphas=torch.ones_like(s.densities)
    s.materials=torch.zeros(2*n,dtype=torch.int32)
    domain=DomainDescription(T([-2.,-2.]),T([3*n+2.,2.]),torch.tensor([False,False]),2)
    adj=buildVerletList(s,domain,supportMode=SupportScheme.SuperSymmetric,verletScale=1.01)
    i,j=adj.livePairs();i=i.numpy();j=j.numpy();live=len(i)
    coeff=CRKState(A=T(np.ones(2*n)),B=T(np.zeros((2*n,2))),gradA=T(np.zeros((2*n,2))),gradB=T(np.zeros((2*n,2,2))))
    common=dict(queryParticles=s,domain=domain,adjacency=adj,queryEnergies=s.internalEnergies,
        queryVelocities=s.velocities,queryCs=s.soundspeeds,queryAlphas=s.alphas,queryPressures=s.pressures)
    crk=CRKSPHConfig();cp=CompSPHConfig();crk.diffusionParams.C_l=cp.diffusionParams.C_l=2.;crk.diffusionParams.C_q=cp.diffusionParams.C_q=1.
    params=resolveCRKLimiter(crk.crkViscosityParams,4.);params.viscosityKernelScale=2.;params.forceVanLeerOff=True
    data={};summaries={}
    r=x[i]-x[j]
    gi=gradient(r,R[i],np.ones(live),np.zeros((live,2)),np.zeros((live,2)),np.zeros((live,2,2)))
    gj=gradient(r,R[j],np.ones(live),np.zeros((live,2)),np.zeros((live,2)),np.zeros((live,2,2)))
    G=.5*(gi+gj);dv=v[i]-v[j];dot=lambda a:np.sum(dv*a,axis=1)
    for name in ('CRK','Comp'):
        cfg=crk if name=='CRK' else cp;p=cfg.diffusionParams
        for visc in (False,True):
            p.C_l=2. if visc else 0.;p.C_q=1. if visc else 0.
            kwargs=dict(common,conductivityParams=p,operationProperties=OperationProperties(kernel=KernelFunctions.CubicSpline,supportMode=SupportScheme.KernelMeanSymmetric))
            d=domainData();d.dim=2;d.domainMin=wp.array([-100.,-100.],dtype=scalar_t,device='cpu');d.domainMax=wp.array([3*n+100.,100.],dtype=scalar_t,device='cpu');d.periodicity=wp.array([False,False],dtype=wp.bool,device='cpu')
            q=wp.empty((2*n,4),dtype=scalar_t,device='cpu')
            wp.launch(viscous,2*n,inputs=[wp.from_torch(s.positions,dtype=Vec),wp.from_torch(s.velocities,dtype=Vec),wp.from_torch(s.supports),
                wp.from_torch(s.densities),wp.from_torch(s.soundspeeds),wp.from_torch(s.pressures),d,p,q],device='cpu');q=q.numpy()
            qi=q[i,0];qj=q[i,1]
            if name=='CRK':
                kwargs.update(crkState=coeff,queryVolumes=T(V),queryVelocityTensor=T(np.zeros((2*n,2,2))),crkViscosityParams=params)
                acc,ap,av=computeCrkSPHAccelWarp(**kwargs);rates={}
                for one in (False,True):
                    params.oneSidedWork=one;du=computeCrkSPHdudtWarp(**kwargs)
                    expected=V[i]*V[j]/m[i]*(P[j]+(qj if one else .5*(qi+qj)))*dot(G)
                    reduced=np.bincount(i,weights=expected,minlength=2*n)
                    np.testing.assert_allclose(du.numpy(),reduced,rtol=2.e-11,atol=2.e-11)
                    rates['one-sided' if one else 'averaged']=du
                raw=rates['averaged'];other=rates['one-sided']
                expectedA=-(V[i]*V[j]/m[i]*(P[i]+P[j]+qi+qj))[:,None]*G
                np.testing.assert_allclose((ap+av).numpy()[:live],expectedA,rtol=2.e-11,atol=2.e-11)
            else:
                acc,ap,av=computeCompSPHAccelWarp(**kwargs)
                kwargs['operationProperties']=OperationProperties(kernel=KernelFunctions.CubicSpline,supportMode=SupportScheme.Gather)
                raw=computeCompSPHdudtWarp(**kwargs);other=raw
                # Pi in Warp multiplies approach speed and apparent volume.
                # Translate it into the coefficient of grad W before comparing
                # the gather and mean-gradient pressure/viscous work.
                pi=q[i,2]*.5*(rho[i]+rho[j])/rho[i]
                pj=q[i,3]*.5*(rho[i]+rho[j])/rho[j]
                mu=-np.sum(dv*r,axis=1)/(np.linalg.norm(r,axis=1)+1.e-14*R[i])
                expected=m[j]*P[i]/rho[i]**2*dot(gi)+.5*m[j]/rho[j]*pi*mu*dot(gi)
                np.testing.assert_allclose(raw.numpy(),np.bincount(i,weights=expected,minlength=2*n),rtol=2.e-11,atol=2.e-11)
                expectedA=-m[j,None]*(P[i]/rho[i]**2+P[j]/rho[j]**2)[:,None]*G-.5*(m[j]/rho[j]*(pi+pj)*mu)[:,None]*G
                np.testing.assert_allclose((ap+av).numpy()[:live],expectedA,rtol=2.e-11,atol=2.e-11)
            kinetic=np.sum(m[:,None]*v*acc.numpy(),axis=1).reshape(n,2).sum(1)
            thermal=(m*raw.numpy()).reshape(n,2).sum(1);thermalOther=(m*other.numpy()).reshape(n,2).sum(1)
            # Directed pair accelerations and actual production final-work path.
            params.oneSidedWork=False
            s.ap_ij=ap;s.av_ij=av
            f=computeCompSPHBalanceTermWarp(queryParticles=s,operationProperties=OperationProperties(kernel=KernelFunctions.CubicSpline,supportMode=SupportScheme.KernelMeanSymmetric),
                domain=domain,energyScheme=EnergyScheme.CRK,dt=.01,gamma=5/3,pairWise_pressureAccel=ap,pairWise_viscosityAccel=av,
                queryEnergies=s.internalEnergies,queryVelocities=s.velocities+.005*acc,queryPressures=s.pressures,adjacency=adj)
            s.f_ij=f;pw=beginPairWork((1.,));accumulateStage(pw,0,adj,s,EnergyScheme.CRK)
            rate,request=energyIncrement(pw,s,[NS(dvdt=acc)],(1.,),.01);assert request is None
            dKE=.5*np.sum(m[:,None]*((v+.01*acc.numpy())**2-v*v))
            closure=float(dKE+.01*np.dot(m,rate.numpy()))
            label=name+('-viscous' if visc else '-pressure');record=[]
            for k,desc in enumerate(labels):
                record.append(dict(case=desc,rawPowerResidual=float(kinetic[k]+thermal[k]),rawPowerScaledResidual=float((kinetic[k]+thermal[k])/(1+abs(kinetic[k])+abs(thermal[k]))),
                    rawRates=raw.numpy()[2*k:2*k+2].tolist(),alternativeRates=other.numpy()[2*k:2*k+2].tolist(),
                    thermalPartitionSumDifference=float(thermalOther[k]-thermal[k])))
            data[label]=dict(cases=record,finiteStepCompatibleEnergyResidual=closure)
            summaries[label]=dict(maximumInstantaneousPowerResidual=max(abs(r['rawPowerResidual']) for r in record),
                maximumScaledPowerResidual=max(abs(r['rawPowerScaledResidual']) for r in record),compatibleEnergyResidual=closure,
                maximumLocalRateDifference=float(np.max(np.abs(raw.numpy()-other.numpy()))),
                maximumPairMomentumResidual=float(np.max(np.abs((m[:,None]*acc.numpy()).reshape(n,2,2).sum(1)))),
                maximumPairTorqueResidual=float(np.max(np.abs((m*(x[:,0]*acc.numpy()[:,1]-x[:,1]*acc.numpy()[:,0])).reshape(n,2).sum(1)))))
            np.savez_compressed(root/(label+'.npz'),positions=x,supports=R,masses=m,densities=rho,volumes=V,pressures=P,
                velocities=v,i=i,j=j,ap=ap.numpy()[:live],av=av.numpy()[:live],acceleration=acc.numpy(),raw=raw.numpy(),alternative=other.numpy(),f=f.numpy()[:live])
    save=dict(summary=summaries,operators=data,scope='Fixed 2D pairs, no time evolution. CRK coefficients A1 B0, unequal independent CRK volumes, actual compiled Warp force/heating/balance and compatible accumulation; Comp uses its production mean-force/gather-heating conventions.')
    (root/'operators.json').write_text(json.dumps(save,indent=2)+'\n');print(json.dumps(summaries,indent=2))
    return save


def replay_native(folder,root):
    """Actual Warp kernels on exported native RHS states, including ghosts.

    The installed Spheral uses a 1000-entry table; Warp evaluates its spline
    directly. Imported native RK polynomials isolate the force/work algebra.
    """
    from shockBubbleForceAudit import coefficients
    root.mkdir(parents=True,exist_ok=True)
    nodes=np.load(folder/'heating_nodeAudit.npz')['rows'];result={}
    for phase in np.unique(nodes[:,0]):
        a=nodes[nodes[:,0]==phase];real=a[:,3]>0
        T=lambda x:torch.tensor(x,dtype=torch.float64).contiguous()
        s=ParticleState(T(a[:,5:7]),T(4/a[:,14]),T(a[:,4]),torch.zeros(len(a),dtype=torch.int32),T(a[:,8]))
        s.velocities=T(a[:,11:13]);s.internalEnergies=T(a[:,10]);s.pressures=T(a[:,9]);s.soundspeeds=T(a[:,13]);s.alphas=torch.ones_like(s.densities)
        assert np.max(np.abs(a[:,15]))<1.e-12 and np.max(np.abs(a[:,14]-a[:,16]))<1.e-12
        domain=DomainDescription(T(a[:,5:7].min(0)-10),T(a[:,5:7].max(0)+10),torch.tensor([False,False]),2)
        adj=buildVerletList(s,domain,supportMode=SupportScheme.SuperSymmetric,verletScale=1.01)
        crk=CRKState(**{k:T(v) for k,v in coefficients(dict(coefficients=a[:,17:26])).items()})
        cfg=CRKSPHConfig();cfg.diffusionParams.C_l=2.;cfg.diffusionParams.C_q=1.
        params=resolveCRKLimiter(cfg.crkViscosityParams,4.);params.viscosityKernelScale=4.
        args=dict(queryParticles=s,operationProperties=OperationProperties(kernel=KernelFunctions.B8,supportMode=SupportScheme.KernelMeanSymmetric),
            domain=domain,conductivityParams=cfg.diffusionParams,crkViscosityParams=params,queryVelocityTensor=T(a[:,26:30].reshape(-1,2,2)),
            queryEnergies=s.internalEnergies,queryVelocities=s.velocities,queryCs=s.soundspeeds,queryAlphas=s.alphas,
            queryPressures=s.pressures,queryVolumes=T(a[:,7]),adjacency=adj,crkState=crk)
        acc,_,_=computeCrkSPHAccelWarp(**args);data={}
        for one in (False,True):
            params.oneSidedWork=one;du=computeCrkSPHdudtWarp(**args).numpy();target=a[:,32 if one else 33]
            delta=du[real]-target[real]
            data['one-sided' if one else 'averaged']=dict(relativeL2=float(np.linalg.norm(delta)/max(np.linalg.norm(target[real]),1.e-30)),
                scaledMaximum=float(np.max(np.abs(delta)/(1+np.abs(target[real])))))
            if data['one-sided' if one else 'averaged']['scaledMaximum']>2.e-4:raise ValueError(('Native/Warp common-state heating mismatch',phase,data))
        d=acc.numpy()[real]-a[real,34:36]
        data['accelerationRelativeL2']=float(np.linalg.norm(d)/np.linalg.norm(a[real,34:36]));result[str(int(phase))]=data
        print('Common native RHS phase',int(phase),data,flush=True)
    (root/'nativeReplay.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=Path('reports/materialEOS/spheral/heating/operators'))
    p.add_argument('--native',type=Path)
    a=p.parse_args()
    if a.native:replay_native(a.native,a.output)
    else:audit(a.output)
