"""Static production-operator audit across 1D resolution jumps, no trajectory.

Run float32/float64 in separate processes (bootstrap fixes Warp scalar types).
No simulation advances, daemon registration or video is needed for this small
operator sweep. Writes interface/bulk errors and particle-level outputs.
"""
import argparse
import json
from pathlib import Path
import time


def audit(args):
    from warpSPHBootstrap import bootstrap
    bootstrap(precision=args.precision)
    import numpy as np
    import torch
    from dataclasses import replace
    from warpSPHCore import (ParticleState, DomainDescription, KernelFunctions, SupportScheme,
        buildVerletList, OperationProperties, OperationDirection, WarpOperation, GradientScheme,
        CRKState, warpOperation)
    from warpSPHCore.crk.crk_moments import _computeCRKMoments_stateBackend
    from warpSPH.modules.crk.terms import computeCRKTerms, crkFactors
    from probe_crkSupportClosure import corrected, cubic
    dtype=torch.float64 if args.precision=='float64' else torch.float32
    device=args.device
    kernel=KernelFunctions.CubicSpline
    args.output.mkdir(parents=True,exist_ok=False)
    records=[]
    def cpu(x): return x.detach().cpu().numpy()
    for ratio in (1,2,4,8):
        n=256; nr=n//ratio
        positions=torch.cat((-1+(torch.arange(n,dtype=dtype,device=device)+.5)/n,
                             (torch.arange(nr,dtype=dtype,device=device)+.5)/nr))[:,None]
        v=torch.cat((torch.full((n,),1/n,dtype=dtype,device=device),
                     torch.full((nr,),1/nr,dtype=dtype,device=device)))
        rho=torch.where(positions[:,0]<0,torch.ones_like(v),torch.full_like(v,.25))
        s=ParticleState(positions=positions,supports=4*v,masses=v*rho,densities=rho,
                        kinds=torch.zeros(len(v),dtype=torch.int32,device=device))
        domain=DomainDescription(positions.new_tensor([-1.]),positions.new_tensor([1.]),
                                torch.ones(1,dtype=torch.bool,device=device),1)
        adj=buildVerletList(s,domain,verletScale=1.4,supportMode=SupportScheme.SuperSymmetric)
        regions=dict(interface=positions[:,0].abs()<2*s.supports.max(),
                     bulk=(positions[:,0].abs()>=2*s.supports.max()) & (positions[:,0].abs()<.5))
        # Nonperiodic affine data are only assessed where no stencil reaches the seam.
        observed=positions[:,0].abs()<.5
        fields=dict(constant=torch.ones_like(v),linear=.7+1.3*positions[:,0])
        for volume in ('physical','legacyGather'):
            vol,_,production=crkFactors(s,domain,kernel,adj,volumes=v if volume=='physical' else None,computeDensity=False)
            for mode in (SupportScheme.Scatter,SupportScheme.Gather,SupportScheme.KernelMeanSymmetric):
                props=OperationProperties(kernel=kernel,supportMode=mode,operationMode=OperationDirection.AllToAll)
                if mode==SupportScheme.Scatter:
                    crk=production
                else:
                    moments=_computeCRKMoments_stateBackend(s,props,domain,queryVolumes=vol,
                        referenceVolumes=vol,adjacency=adj,referenceParticles=None)
                    crk=CRKState(*computeCRKTerms(*moments,s.supports))
                # Reassign masses/densities at fixed geometry and V. The actual
                # production factors must preserve A/B and their derivatives.
                relabeled=replace(s,masses=torch.full_like(v,1/n),densities=(1/n)/vol)
                if mode==SupportScheme.Scatter:
                    relabeled_crk=crkFactors(relabeled,domain,kernel,adj,volumes=vol,computeDensity=False)[2]
                    for key in ('A','B','gradA','gradB'):
                        torch.testing.assert_close(getattr(crk,key),getattr(relabeled_crk,key),rtol=0,atol=0)
                variants=[('crk',crk,mode)]
                if mode==SupportScheme.Scatter:
                    variants += [('uncorrected',None,mode),('mismatchedGather',crk,SupportScheme.Gather)]
                for variant,correction,opmode in variants:
                    values={}
                    for field,f in fields.items():
                        values[field+'Interpolation']=warpOperation(s,replace(props,operation=WarpOperation.Interpolate,
                            supportMode=opmode),domain,queryValues=f,queryVolumes=vol,adjacency=adj,crkState=correction)
                        for gradient in (GradientScheme.Naive,GradientScheme.Difference):
                            values[field+'Gradient'+gradient.name]=warpOperation(s,
                                replace(props,operation=WarpOperation.Gradient,gradientMode=gradient,supportMode=opmode),
                                domain,queryValues=f,queryVolumes=vol,adjacency=adj,crkState=correction).reshape(-1)
                    errors={key:value-(fields['constant'] if key=='constantInterpolation' else
                        fields['linear'] if key=='linearInterpolation' else
                        0. if key.startswith('constant') else 1.3) for key,value in values.items()}
                    row=dict(spacingRatio=ratio,volume=volume,momentSupport=mode.name,operatorSupport=opmode.name,
                             variant=variant,particles=len(v),precision=args.precision,device=device,
                             regions={name:{key:dict(max=float(error[mask].abs().max()),
                                 rms=float(error[mask].square().mean().sqrt())) for key,error in errors.items()}
                                 for name,mask in regions.items()})
                    if variant=='crk':
                        tol_i,tol_g=(5.e-6,1.e-3) if dtype==torch.float32 else (2.e-12,2.e-10)
                        row['passed']=all(measure['max']<(tol_i if key.endswith('Interpolation') else tol_g)
                            for region in row['regions'].values() for key,measure in region.items())
                        # Independent dense double reference on the actual rounded coordinates/volumes.
                        x=cpu(positions[:,0]).astype(float); delta=x[:,None]-x[None,:]
                        delta-=2*np.round(delta/2); radii=cpu(s.supports).astype(float)
                        if mode==SupportScheme.Scatter: W,dW=cubic(delta,radii[None,:])
                        elif mode==SupportScheme.Gather: W,dW=cubic(delta,radii[:,None])
                        else:
                            W0,dW0=cubic(delta,radii[None,:]);W1,dW1=cubic(delta,radii[:,None])
                            W,dW=(W0+W1)*.5,(dW0+dW1)*.5
                        K,G=corrected(delta,W,dW,cpu(vol).astype(float))
                        vf=cpu(vol).astype(float); selected=cpu(observed)
                        dense={}
                        for field,f in fields.items():
                            ff=cpu(f).astype(float)
                            dense[field+'Interpolation']=K@(vf*ff)
                            dense[field+'GradientNaive']=G@(vf*ff)
                            dense[field+'GradientDifference']=(G*vf[None,:]*(ff[None,:]-ff[:,None])).sum(1)
                        row['denseDifferenceMax']={key:float(np.abs(cpu(value)[selected]-dense[key][selected]).max())
                                                  for key,value in values.items()}
                    arrays=dict(x=cpu(positions[:,0]),volumes=cpu(vol),supports=cpu(s.supports),
                                interface=cpu(regions['interface']),observed=cpu(observed))
                    arrays.update({key:cpu(value) for key,value in values.items()})
                    for key in ('A','B','gradA','gradB'): arrays['crk'+key]=cpu(getattr(crk,key))
                    name=f'ratio{ratio}-{volume}-{mode.name}-{variant}'
                    np.savez_compressed(args.output/(name+'.npz'),**arrays)
                    row['data']=str((args.output/(name+'.npz')).resolve())
                    records.append(row)
        print(f'ratio {ratio}: production operators evaluated, {len(records)} records',flush=True)
    if device.startswith('cuda'): torch.cuda.synchronize()
    result=dict(precision=args.precision,device=device,records=records,
                limits='Static 1D scalar interpolation/gradient calls, cubic kernel, no force or dynamics. Periodic seam excluded. Non-scatter coherent modes are operator controls, not alternate CRK force implementations.')
    (args.output/'operators.json').write_text(json.dumps(result,indent=2)+'\n')
    corrected_rows=[r for r in records if r['variant']=='crk']
    mismatch=[r for r in records if r['variant']=='mismatchedGather' and r['spacingRatio']>1]
    assert any(r['regions']['interface']['constantInterpolation']['max']>.01 for r in mismatch)
    failures=[r for r in corrected_rows if not r['passed']]
    print(f'{len(corrected_rows)} coherent corrected configurations; {len(failures)} failed. Deliberately mismatched-support controls detected.',flush=True)
    return 1 if failures else 0


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--precision',choices=('float32','float64'),required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    start=time.perf_counter()
    code=audit(a)
    print(f'Operator audit wall time: {time.perf_counter()-start:.2f}s',flush=True)
    raise SystemExit(code)
