"""Research-only scalar Spheral IdealH operator for the 2D W7 comparison.

Tables are copied from the pinned native TableKernel, including its actual
quadratic gradient and monotone cubic-Hermite nPerh lookup coefficients.
No production support policy or default is changed.
"""
from pathlib import Path
import numpy as np
import torch
import warp as wp
from warpSPHCore import scalar_t,index_t,offset_t


@wp.func
def hermite(x:scalar_t,meta:wp.array(dtype=scalar_t),data:wp.array(dtype=scalar_t)):
    lo=meta[0];hi=meta[1];step=meta[2];n=int(meta[3])
    value=scalar_t(0.)
    if x<lo:
        value=data[0]+data[n]*(x-lo)
    elif x>hi:
        # Native extrapolation uses x-xmin even above xmax; retain it verbatim.
        value=data[n-1]+data[2*n-1]*(x-lo)
    else:
        i=wp.min(n-2,wp.max(0,int((x-lo)/step)))
        t=wp.clamp((x-lo-scalar_t(i)*step)/step,scalar_t(0.),scalar_t(1.))
        t2=t*t;t3=t2*t
        value=(scalar_t(2.)*t3-scalar_t(3.)*t2+scalar_t(1.))*data[i]+(-scalar_t(2.)*t3+scalar_t(3.)*t2)*data[i+1]
        value+=step*((t3-scalar_t(2.)*t2+t)*data[n+i]+(t3-t2)*data[n+i+1])
    return wp.max(scalar_t(0.),value)


@wp.func
def kernel_gradient(eta:scalar_t,meta:wp.array(dtype=scalar_t),data:wp.array(dtype=scalar_t)):
    value=scalar_t(0.)
    if eta<scalar_t(4.):
        n=int(meta[3])//3
        i=3*wp.min(n-1,wp.max(0,int((eta-meta[0])/meta[2])))
        value=wp.abs(data[i]+(data[i+1]+data[i+2]*eta)*eta)
    return value


@wp.kernel
def ideal_kernel(x:wp.array2d(dtype=scalar_t),R:wp.array(dtype=scalar_t),m:wp.array(dtype=scalar_t),
    rho:wp.array(dtype=scalar_t),tag:wp.array(dtype=wp.int32),count:wp.array(dtype=wp.int32),
    offset:wp.array(dtype=offset_t),jlist:wp.array(dtype=index_t),length:wp.array(dtype=scalar_t),periodic:wp.array(dtype=wp.bool),
    gmeta:wp.array(dtype=scalar_t),gdata:wp.array(dtype=scalar_t),nmeta:wp.array(dtype=scalar_t),ndata:wp.array(dtype=scalar_t),
    weighted:bool,target:scalar_t,damping:scalar_t,hmin:scalar_t,hmax:scalar_t,
    result:wp.array(dtype=scalar_t),root:wp.array(dtype=scalar_t),current:wp.array(dtype=scalar_t)):
    i=wp.tid();total=scalar_t(0.)
    for p in range(count[i]):
        j=int(jlist[offset[i]+offset_t(p)])
        dx=x[i,0]-x[j,0];dy=x[i,1]-x[j,1]
        if periodic[0]:dx-=wp.round(dx/length[0])*length[0]
        if periodic[1]:dy-=wp.round(dy/length[1])*length[1]
        eta=scalar_t(4.)*wp.sqrt(dx*dx+dy*dy)/R[i]
        weight=scalar_t(1.)
        if weighted and tag[i]!=tag[j]:weight=m[j]*rho[i]/(m[i]*rho[j])
        total+=weight*kernel_gradient(eta,gmeta,gdata)
    z=wp.sqrt(wp.max(scalar_t(0.),total));n=hermite(z,nmeta,ndata)
    if wp.abs(z)<scalar_t(1.e-15):n=scalar_t(.5)*target
    s=wp.clamp(target/(n+scalar_t(1.e-30)),scalar_t(.25),scalar_t(4.))
    a=scalar_t(.4)*(scalar_t(1.)+s*s)
    if s>=scalar_t(1.):a=scalar_t(.4)*(scalar_t(1.)+scalar_t(1.)/(s*s*s))
    h=R[i]/scalar_t(4.);hnew=wp.clamp(h*wp.pow(scalar_t(1.)-a+a*s,damping),hmin,hmax)
    result[i]=scalar_t(4.)*hnew;root[i]=z;current[i]=n


_tables={}
def tables(path,device,dtype=torch.float64):
    key=(str(Path(path).resolve()),str(device),dtype)
    if key not in _tables:
        a=dict(np.load(path));_tables[key]={k:torch.as_tensor(a[k],device=device,dtype=dtype).contiguous() for k in ('gradMeta','gradData','lookupMeta','lookupData')}
    return _tables[key]


def ideal_support(state,domain,adjacency,table_path,*,weighted=True,target=1.,damping=1.,hmin=1.e-5,hmax=.5):
    if state.positions.shape[1]!=2:raise ValueError('IdealH research control requires 2D')
    t=tables(table_path,state.positions.device,state.positions.dtype);out=torch.empty_like(state.supports);root=torch.empty_like(out);current=torch.empty_like(out)
    w=lambda a:wp.from_torch(a.contiguous())
    tag=state.materials.to(torch.int32);length=domain.max-domain.min
    wp.launch(ideal_kernel,dim=len(out),device=wp.device_from_torch(out.device),inputs=[w(state.positions),w(state.supports),w(state.masses),w(state.densities),w(tag),
        w(adjacency.numNeighbors),w(adjacency.edgeOffsets),w(adjacency.paddedJ),w(length),w(domain.periodic),
        w(t['gradMeta']),w(t['gradData']),w(t['lookupMeta']),w(t['lookupData']),weighted,target,damping,hmin,hmax,w(out),w(root),w(current)])
    return out,root,current
