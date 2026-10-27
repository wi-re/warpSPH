"""Locally generated, wide-range Owen shell calibration in compact-R units.

Algorithm audited against pinned Spheral TableKernel / CubicHermiteInterpolator:
quadratic gradient samples, radial shell sum and Han--Guo derivative knots.
The linear compatibility preset inverts the unlimited forward cubic before
Fritsch--Carlson limiting. The general 4096-point/log-inverse preset limits
the forward cubic first to preserve invertibility for narrow kernels.
No Spheral imports, exported tables or force-kernel substitutions are used.
All host fitting is float64; kernel samples follow the configured core precision.
"""
from functools import lru_cache
import math
import numpy as np
import torch
import warp as wp
from warpSPHCore import scalar_t,KernelFunctions
from warpSPHCore.kernels.eval_kernel import eval_C_d,eval_dkdq


@wp.kernel
def _sample_gradient(kernel:wp.int32,dim:wp.int32,n:wp.int32,out:wp.array(dtype=scalar_t)):
    i=wp.tid();q=scalar_t(i)/scalar_t(n-1)
    out[i]=eval_C_d(dim,kernel)*eval_dkdq(q,dim,kernel)


@wp.func
def quadratic_gradient(q:scalar_t,coeff:wp.array(dtype=scalar_t),step:scalar_t):
    result=scalar_t(0.)
    if q<scalar_t(1.):
        j=3*wp.min(coeff.shape[0]//3-1,wp.max(0,int(q/step)))
        result=wp.abs(coeff[j]+(coeff[j+1]+coeff[j+2]*q)*q)
    return result


def quadratic_coefficients(values):
    values=np.asarray(values,dtype=np.float64);segments=(len(values)-1)//2
    if len(values)<3 or len(values)%2!=1:raise ValueError('Quadratic gradient samples must have odd length >=3')
    step=1./segments;x=np.arange(segments)*step
    y0,y1,y2=values[:-2:2],values[1:-1:2],values[2::2]
    c=2*(y2-2*y1+y0)/step**2;b=(y2-y0)/step-2*c*(x+.5*step);a=y0-b*x-c*x*x
    return np.column_stack((a,b,c)).ravel()


def gradient_at(x,coeff):
    x=np.asarray(x,dtype=np.float64);n=len(coeff)//3;i=np.clip((x*n).astype(int),0,n-1)*3
    return np.where(x<1,abs(coeff[i]+(coeff[i+1]+coeff[i+2]*x)*x),0.)


def derivative_knots(y,step):
    """Solve the audited tridiagonal Han--Guo system, O(N), no scipy dependency."""
    y=np.asarray(y,dtype=np.float64);n=len(y)
    diagonal=np.full(n,4.);upper=np.full(n-1,-.5);lower=np.full(n-1,-.5)
    upper[0]=-1.;lower[-1]=-1.
    rhs=np.empty(n);rhs[0]=3*(y[1]-y[0])/step;rhs[-1]=3*(y[-1]-y[-2])/step
    rhs[1:-1]=1.5*(y[2:]-y[:-2])/step
    for i in range(1,n):
        ratio=lower[i-1]/diagonal[i-1];diagonal[i]-=ratio*upper[i-1];rhs[i]-=ratio*rhs[i-1]
    slopes=rhs/diagonal
    for i in range(n-2,-1,-1):slopes[i]=(rhs[i]-upper[i]*slopes[i+1])/diagonal[i]
    return slopes


def monotonic_slopes(y,slopes,step):
    """Sequential Fritsch--Carlson sweeps, including native .99 safety factor."""
    y=np.asarray(y);slopes=slopes.copy();sec=np.diff(y)/step
    for k in range(1,len(y)-1):
        if sec[k-1]*sec[k]<=0:slopes[k]=0.
        if sec[k]==0:slopes[k:k+2]=0.
    for _ in range(100):
        changed=False
        for k,c in enumerate(sec):
            if c==0:
                changed=changed or bool(np.any(slopes[k:k+2]!=0));slopes[k:k+2]=0.;continue
            a,b=slopes[k:k+2]/c
            if a<0:slopes[k]=0.;a=0.;changed=True
            if b<0:slopes[k+1]=0.;b=0.;changed=True
            norm=math.hypot(a,b)
            if norm>3:
                slopes[k:k+2]=.99*3/norm*np.array([a,b])*c;changed=True
        if not changed:return slopes
    raise ValueError('Hermite monotonic slope limiter did not converge')


def hermite_values(x,lo,hi,y,slopes):
    x=np.asarray(x,dtype=np.float64);step=(hi-lo)/(len(y)-1)
    i=np.clip(((x-lo)/step).astype(int),0,len(y)-2);t=np.clip((x-lo-i*step)/step,0,1);t2=t*t;t3=t2*t
    out=(2*t3-3*t2+1)*y[i]+(-2*t3+3*t2)*y[i+1]+step*((t3-2*t2+t)*slopes[i]+(t3-t2)*slopes[i+1])
    # Endpoint-tangent extrapolation. The pinned native upper extrapolation
    # uses x-xmin instead; do not perpetuate that discontinuity here.
    return np.where(x<lo,y[0]+slopes[0]*(x-lo),np.where(x>hi,y[-1]+slopes[-1]*(x-hi),out))


def shell_values(nh,dim,gradient_coeff):
    nh=np.asarray(nh,dtype=np.float64);dq=1/nh;q=dq.copy();total=np.zeros_like(nh)
    for _ in range(int(np.ceil(nh.max()))+1):
        live=q<1
        if not live.any():break
        factor=2. if dim==1 else (2*np.pi*q[live]/dq[live] if dim==2 else 4*np.pi*(q[live]/dq[live])**2)
        total[live]+=factor*gradient_at(q[live],gradient_coeff);q+=dq
    return total**(1/dim)


def build_coefficients(gradient_samples,dim,*,points=4096,minimum=1.1,maximum=256.,inverse_space='log'):
    if dim not in (1,2,3):raise ValueError('Owen shell dimension must be 1, 2 or 3')
    if points<8 or not 1<minimum<maximum:raise ValueError('Invalid shell lookup range or point count')
    if inverse_space not in ('linear','log'):raise ValueError('inverse_space must be linear or log')
    if not np.isfinite(gradient_samples).all():raise ValueError('Kernel has non-finite radial gradient samples')
    grad=quadratic_coefficients(gradient_samples);nh=np.linspace(minimum,maximum,points)
    z=shell_values(nh,dim,grad)
    if not np.isfinite(z).all() or np.any(np.diff(z)<=0):
        raise ValueError('Kernel shell statistic is not strictly increasing over the requested range')
    dy=derivative_knots(z,(maximum-minimum)/(points-1))
    forward_slopes=monotonic_slopes(z,dy,(maximum-minimum)/(points-1))
    # The linear mode reproduces native construction order. The general log
    # mode limits the forward cubic first so a narrow kernel cannot give a
    # non-monotone spline between its monotone knots during root inversion.
    forward_fit=dy if inverse_space=='linear' else forward_slopes
    # Float32 narrow-kernel tails can have an exactly zero first moment.
    # A shifted log coordinate retains that endpoint without log(0) or
    # changing the chosen nh range. Positive-endpoint tables retain log(z).
    log_offset=float(z[1]) if inverse_space=='log' and z[0]==0 else 0.
    coordinate=np.log(z[[0,-1]]+log_offset) if inverse_space=='log' else z[[0,-1]]
    zi=np.linspace(*coordinate,points);targets=np.maximum(0.,np.exp(zi)-log_offset) if inverse_space=='log' else zi
    left=np.full(points,minimum);right=np.full(points,maximum)
    for _ in range(64):
        mid=(left+right)/2;value=hermite_values(mid,minimum,maximum,z,forward_fit)
        left=np.where(value<targets,mid,left);right=np.where(value>=targets,mid,right)
    inverse=(left+right)/2;inverse[0]=minimum;inverse[-1]=maximum
    interval=(zi[-1]-zi[0])/(points-1)
    slopes=monotonic_slopes(inverse,derivative_knots(inverse,interval),interval)
    return dict(gradData=grad,gradMeta=np.array([0.,1.,2/(len(gradient_samples)-1),len(grad)]),
        lookupMeta=np.array([zi[0],zi[-1],interval,points]),lookupData=np.r_[inverse,slopes],inverseLog=np.array(inverse_space=='log'),inverseLogOffset=np.array(log_offset),
        forwardMeta=np.array([minimum,maximum,(maximum-minimum)/(points-1),points]),forwardData=np.r_[z,forward_slopes])


@lru_cache(maxsize=64)
def coefficients(kernel,dim,points=4096,minimum=1.1,maximum=256.,inverse_space='log'):
    # Surface-force / singular viscosity kernels are not density-support kernels.
    if kernel in (KernelFunctions.ViscosityKernel,KernelFunctions.CohesionKernel,KernelFunctions.AdhesionKernel):
        raise ValueError('This kernel is not supported as a hydrodynamic Owen calibration kernel')
    if dim not in (1,2,3):raise ValueError('Owen shell dimension must be 1, 2 or 3')
    n=points+1 if points%2==0 else points
    # CPU sampling avoids making coefficient generation depend on a CUDA device.
    values=wp.empty(n,dtype=scalar_t,device='cpu')
    wp.launch(_sample_gradient,dim=n,inputs=[kernel.value,dim,n,values],device='cpu')
    return build_coefficients(values.numpy().astype(np.float64),dim,points=points,minimum=minimum,maximum=maximum,inverse_space=inverse_space)


class HermiteOwen:
    """Per-(kernel,dimension) coefficients; lazy device copies, no RHS host reads."""
    def __init__(self,kernel,dim,**kwargs):
        self.arrays=coefficients(kernel,dim,**kwargs);self.devices={}
    def tensors(self,device,dtype=torch.float64):
        key=(str(device),dtype)
        if key not in self.devices:self.devices[key]={k:torch.as_tensor(v.copy(),device=device,dtype=dtype) for k,v in self.arrays.items()}
        return self.devices[key]
    def fromPsiH(self,x):
        data=self.tensors(x.device);meta=self.arrays['lookupMeta'];n=int(meta[3]);y=data['lookupData'];v=x.double()
        if bool(self.arrays['inverseLog']):
            if float(self.arrays['inverseLogOffset'])!=0:v=v.clamp_min(0)+float(self.arrays['inverseLogOffset'])
            v=v.clamp_min(np.finfo(np.float64).tiny).log()
        index=((v-meta[0])/meta[2]).long().clamp(0,n-2);t=((v-meta[0]-index*meta[2])/meta[2]).clamp(0,1);t2=t*t;t3=t2*t
        result=(2*t3-3*t2+1)*y[index]+(-2*t3+3*t2)*y[index+1]+meta[2]*((t3-2*t2+t)*y[n+index]+(t3-t2)*y[n+index+1])
        result=torch.where(v<meta[0],y[0]+y[n]*(v-meta[0]),torch.where(v>meta[1],y[n-1]+y[2*n-1]*(v-meta[1]),result))
        return result.clamp_min(0).to(x.dtype)
