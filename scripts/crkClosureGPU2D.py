"""Experimental double-precision Warp 2D tapered closure.

Matrix-free dual CG with block-Jacobi, FFT or physical coarse preconditioning;
graph-captured iterations.
Independent research pressure/volume-work discretization, not production CRK.
Neighbor topology is built on the CPU with a checked reusable skin. All RK
geometry, constraint assembly, projection and pressure RHS execute on GPU.
"""
import time
import numpy as np
import warp as wp
from warp.optim.linear import LinearOperator,cg
from scipy.spatial import cKDTree

wp.set_module_options({'enable_backward':False})
D=wp.float64
V2=wp.vec2d
V3=wp.vec3d
M2=wp.mat22d
M3=wp.mat33d


@wp.func
def cubic(z:D):
    k=D(0.)
    if z<D(.5):k=D(1.)-D(6.)*z*z+D(6.)*z*z*z
    elif z<D(1.):k=D(2.)*(D(1.)-z)**D(3.)
    return k


@wp.func
def kernel(delta:V2,R:D):
    distance=wp.length(delta);z=distance/R;dk=D(0.)
    if z<D(.5):dk=-D(12.)*z+D(18.)*z*z
    elif z<D(1.):dk=-D(6.)*(D(1.)-z)*(D(1.)-z)
    C=D(40.)/(D(7.)*D(3.141592653589793))
    gradient=V2(D(0.))
    if distance>D(0.):gradient=C*dk/(R*R*R*distance)*delta
    return C*cubic(z)/(R*R),gradient


@wp.func
def displacement(a:V2,b:V2):
    r=b-a
    return r-V2(wp.round(r[0]),wp.round(r[1]))


@wp.kernel
def rk_terms(x:wp.array(dtype=V2),V:wp.array(dtype=D),offset:wp.array(dtype=int),
             neighbor:wp.array(dtype=int),A:wp.array(dtype=D),B:wp.array(dtype=V2),
             dA:wp.array(dtype=V2),dB:wp.array(dtype=M2)):
    i=wp.tid();m0=D(0.);m1=V2(D(0.));m2=M2(D(0.));dm0=V2(D(0.));dm1=M2(D(0.))
    dm2x=M2(D(0.));dm2y=M2(D(0.));identity=wp.identity(n=2,dtype=D)
    # Include the source at i; kernel differentiation keeps source positions fixed.
    Wself,_=kernel(V2(D(0.)),D(4.)*wp.sqrt(V[i]))
    m0=V[i]*Wself;dm1=V[i]*Wself*identity
    for at in range(offset[i],offset[i+1]):
        j=neighbor[at];d=-displacement(x[i],x[j]);R=D(4.)*wp.sqrt(V[j]);W,g=kernel(d,R)
        m0+=V[j]*W;m1+=V[j]*W*d;m2+=V[j]*W*wp.outer(d,d);dm0+=V[j]*g
        dm1+=V[j]*(W*identity+wp.outer(g,d))
        for k in range(2):
            dm=M2(D(0.))
            for a in range(2):
                for b in range(2):
                    value=d[a]*d[b]*g[k]
                    if k==a:value+=d[b]*W
                    if k==b:value+=d[a]*W
                    dm[a,b]=V[j]*value
            if k==0:dm2x+=dm
            else:dm2y+=dm
    inv=wp.inverse(m2);z=inv*m1;aa=D(1.)/(m0-wp.dot(m1,z));bb=-z
    ga=V2(D(0.));gb=M2(D(0.))
    for k in range(2):
        dm=dm2x
        if k==1:dm=dm2y
        row=V2(dm1[k,0],dm1[k,1])
        ga[k]=-aa*aa*(dm0[k]-D(2.)*wp.dot(row,z)+wp.dot(z,dm*z))
        br=-inv*row+inv*(dm*z)
        for a in range(2):gb[k,a]=br[a]
    A[i]=aa;B[i]=bb;dA[i]=ga;dB[i]=gb


@wp.func
def corrected(d:V2,R:D,A:D,B:V2,dA:V2,dB:M2):
    W,g=kernel(d,R);factor=D(1.)+wp.dot(B,d)
    return (dA*factor+A*(dB*d+B))*W+A*factor*g


@wp.kernel
def logical_mapping(x:wp.array(dtype=V2),logical_n:int,inverse:wp.array(dtype=M2),enabled:int):
    i=wp.tid();matrix=wp.identity(n=2,dtype=D)
    if logical_n>0 and enabled>0:
        a=i//logical_n;b=i%logical_n
        for k in range(2):
            minus1=((a-1+logical_n)%logical_n)*logical_n+b
            plus1=((a+1)%logical_n)*logical_n+b
            minus2=((a-2+logical_n)%logical_n)*logical_n+b
            plus2=((a+2)%logical_n)*logical_n+b
            if k==1:
                minus1=a*logical_n+(b-1+logical_n)%logical_n;plus1=a*logical_n+(b+1)%logical_n
                minus2=a*logical_n+(b-2+logical_n)%logical_n;plus2=a*logical_n+(b+2)%logical_n
            d1=displacement(x[minus1],x[i])+displacement(x[i],x[plus1])
            d2=displacement(x[minus2],x[minus1])+d1+displacement(x[plus1],x[plus2])
            column=D(logical_n)*(D(8.)*d1-d2)/D(12.)
            matrix[0,k]=column[0];matrix[1,k]=column[1]
    inverse[i]=wp.inverse(matrix)


@wp.kernel
def transport_guess(n:int,old:wp.array(dtype=M2),new:wp.array(dtype=M2),potential:wp.array(dtype=V3)):
    index=wp.tid();i=index%n;v=potential[index]
    physical=wp.transpose(old[i])*V2(v[1],v[2])
    mapped=wp.transpose(wp.inverse(new[i]))*physical
    potential[index]=V3(v[0],mapped[0],mapped[1])


@wp.kernel
def edge_geometry(x:wp.array(dtype=V2),V:wp.array(dtype=D),ii:wp.array(dtype=int),jj:wp.array(dtype=int),
                  A:wp.array(dtype=D),B:wp.array(dtype=V2),dA:wp.array(dtype=V2),dB:wp.array(dtype=M2),
                  q0:wp.array(dtype=V2),ci:wp.array(dtype=V3),cj:wp.array(dtype=V3),scale2:wp.array(dtype=D),logical_n:int,Jinv:wp.array(dtype=M2)):
    e=wp.tid();i=ii[e];j=jj[e];r=displacement(x[i],x[j]);Ri=D(4.)*wp.sqrt(V[i]);Rj=D(4.)*wp.sqrt(V[j])
    gi=corrected(-r,Rj,A[i],B[i],dA[i],dB[i]);gj=corrected(r,Ri,A[j],B[j],dA[j],dB[j])
    q0[e]=D(.5)*V[i]*V[j]*(gi-gj)
    ci[e]=V3(D(1.)/wp.sqrt(V[i]),r[0]/V[i],r[1]/V[i])
    cj[e]=V3(-D(1.)/wp.sqrt(V[j]),r[0]/V[j],r[1]/V[j])
    if logical_n>0:
        ri=D(logical_n)*(Jinv[i]*r);rj=D(logical_n)*(Jinv[j]*r)
        ci[e]=V3(D(1.),ri[0],ri[1]);cj[e]=V3(-D(1.),rj[0],rj[1])
    s=D(.5)*(cubic(wp.length(r)/Ri)+cubic(wp.length(r)/Rj));scale2[e]=s*s


@wp.kernel
def prepare(n:int,offset:wp.array(dtype=int),edge:wp.array(dtype=int),sign:wp.array(dtype=int),
            ci:wp.array(dtype=V3),cj:wp.array(dtype=V3),s2:wp.array(dtype=D),q0:wp.array(dtype=V2),
            rhs:wp.array(dtype=V3),inverse:wp.array(dtype=M3),V:wp.array(dtype=D),logical_n:int,Jinv:wp.array(dtype=M2)):
    i=wp.tid();block=M3(D(0.));r0=V3(D(0.));r1=V3(D(0.))
    for at in range(offset[i],offset[i+1]):
        e=edge[at];c=ci[e]
        if sign[at]<0:c=cj[e]
        r0+=c*q0[e][0];r1+=c*q0[e][1];block+=s2[e]*wp.outer(c,c)
    target=D(1.)
    if logical_n>0:target=V[i]*D(logical_n)
    rhs[i]=V3(D(0.),target,D(0.))-r0;rhs[n+i]=V3(D(0.),D(0.),target)-r1
    if logical_n>0:
        matrix=Jinv[i]
        rhs[i]=V3(D(0.),target*matrix[0,0],target*matrix[1,0])-r0
        rhs[n+i]=V3(D(0.),target*matrix[0,1],target*matrix[1,1])-r1
    # Pin one redundant incidence row, eliminating the constant dual gauge.
    if i==0 and logical_n==0:
        rhs[i]=V3(D(0.),rhs[i][1],rhs[i][2]);rhs[n+i]=V3(D(0.),rhs[n+i][1],rhs[n+i][2])
        block[0,0]=D(1.);block[0,1]=D(0.);block[0,2]=D(0.);block[1,0]=D(0.);block[2,0]=D(0.)
    inverse[i]=wp.inverse(block)


@wp.kernel
def gram(n:int,offset:wp.array(dtype=int),edge:wp.array(dtype=int),sign:wp.array(dtype=int),
         ii:wp.array(dtype=int),jj:wp.array(dtype=int),ci:wp.array(dtype=V3),cj:wp.array(dtype=V3),s2:wp.array(dtype=D),
         x:wp.array(dtype=V3),y:wp.array(dtype=V3),z:wp.array(dtype=V3),alpha:D,beta:D,logical_n:int):
    index=wp.tid();i=index%n;batch=index//n;value=V3(D(0.))
    for at in range(offset[i],offset[i+1]):
        e=edge[at];a=ii[e];b=jj[e];va=x[batch*n+a];vb=x[batch*n+b]
        if a==0 and logical_n==0:va=V3(D(0.),va[1],va[2])
        if b==0 and logical_n==0:vb=V3(D(0.),vb[1],vb[2])
        c=ci[e]
        if sign[at]<0:c=cj[e]
        value+=c*(s2[e]*(wp.dot(ci[e],va)+wp.dot(cj[e],vb)))
    if i==0 and logical_n==0:value=V3(x[index][0],value[1],value[2])
    z[index]=alpha*value+beta*y[index]


@wp.kernel
def precondition(n:int,inverse:wp.array(dtype=M3),x:wp.array(dtype=V3),y:wp.array(dtype=V3),
                 z:wp.array(dtype=V3),alpha:D,beta:D):
    i=wp.tid();z[i]=alpha*(inverse[i%n]*x[i])+beta*y[i]


@wp.kernel
def reconstruct(n:int,ii:wp.array(dtype=int),jj:wp.array(dtype=int),ci:wp.array(dtype=V3),cj:wp.array(dtype=V3),
                s2:wp.array(dtype=D),q0:wp.array(dtype=V2),potential:wp.array(dtype=V3),q:wp.array(dtype=V2),logical_n:int):
    e=wp.tid();i=ii[e];j=jj[e];value=q0[e]
    for k in range(2):
        a=potential[k*n+i];b=potential[k*n+j]
        if i==0 and logical_n==0:a=V3(D(0.),a[1],a[2])
        if j==0 and logical_n==0:b=V3(D(0.),b[1],b[2])
        value[k]+=s2[e]*(wp.dot(ci[e],a)+wp.dot(cj[e],b))
    q[e]=value


@wp.kernel
def residual(n:int,offset:wp.array(dtype=int),edge:wp.array(dtype=int),sign:wp.array(dtype=int),
             ci:wp.array(dtype=V3),cj:wp.array(dtype=V3),q:wp.array(dtype=V2),error:wp.array(dtype=D),V:wp.array(dtype=D),logical_n:int,Jinv:wp.array(dtype=M2)):
    i=wp.tid();r0=V3(D(0.));r1=V3(D(0.))
    for at in range(offset[i],offset[i+1]):
        e=edge[at];c=ci[e]
        if sign[at]<0:c=cj[e]
        r0+=c*q[e][0];r1+=c*q[e][1]
    if logical_n>0:
        matrix=wp.inverse(Jinv[i]);m0=matrix*V2(r0[1],r0[2])/(V[i]*D(logical_n));m1=matrix*V2(r1[1],r1[2])/(V[i]*D(logical_n))
        r0=V3(r0[0]/wp.sqrt(V[i]),m0[0],m0[1]);r1=V3(r1[0]/wp.sqrt(V[i]),m1[0],m1[1])
    r0-=V3(D(0.),D(1.),D(0.));r1-=V3(D(0.),D(0.),D(1.))
    for k in range(3):
        if not wp.isfinite(r0[k]) or not wp.isfinite(r1[k]):wp.atomic_max(error,0,D(1.e30))
        wp.atomic_max(error,0,wp.abs(r0[k]));wp.atomic_max(error,0,wp.abs(r1[k]))


@wp.kernel
def fft_output(n:int,real:wp.array(dtype=D),y:wp.array(dtype=V3),z:wp.array(dtype=V3),alpha:D,beta:D):
    index=wp.tid();i=index%n;batch=index//n
    value=V3(real[(batch*3)*n+i],real[(batch*3+1)*n+i],real[(batch*3+2)*n+i])
    z[index]=alpha*value+beta*y[index]


def spectral_inverse(n):
    """Exact uniform-lattice tapered Gram symbol, with constant gauge removed."""
    kx=2*np.pi*np.fft.fftfreq(n)[:,None];ky=2*np.pi*np.fft.rfftfreq(n)[None,:]
    symbol=np.zeros((n,n//2+1,3,3),dtype=np.complex128)
    for dx in range(-4,5):
        for dy in range(-4,5):
            radius=np.hypot(dx,dy);z=radius/4
            if not 0<radius<4:continue
            w=(1-6*z*z+6*z**3 if z<.5 else 2*(1-z)**3)**2
            a=np.array([1.,dx,dy]);b=np.array([-1.,dx,dy])
            phase=np.exp(1j*(kx*dx+ky*dy))
            symbol+=w*(np.outer(a,a)+phase[:,:,None,None]*np.outer(a,b))
    symbol=.5*(symbol+np.swapaxes(symbol.conj(),-1,-2))
    symbol[0,0,0,0]=1
    inverse=np.linalg.inv(symbol);inverse[0,0,0,:]=0;inverse[0,0,:,0]=0
    return inverse


class FFTPreconditioner:
    """Reference-grid FFT used only as a preconditioner on actual geometry.

    Requires stable logical n-by-n indexing (index=ix*n+iy). This is not an
    approximation of the converged force; its true constraints are checked.
    Buffers persist across external CUDA graph captures, including FFT output.
    """
    def __init__(self,n,device,stream):
        import torch
        self.torch=torch;self.n=n;self.N=n*n;self.device=device
        self.stream=wp.stream_to_torch(stream)
        self.symbol=torch.as_tensor(spectral_inverse(n),device=str(device))
        self.frequency=torch.empty((2,3,n,n//2+1),dtype=torch.complex128,device=str(device))
        self.product=torch.empty((2,n,n//2+1,3,1),dtype=torch.complex128,device=str(device))
        self.real=torch.empty((2,3,n,n),dtype=torch.float64,device=str(device))
        self.real_wp=wp.from_torch(self.real.reshape(-1))

    def matvec(self,x,y,z,alpha,beta):
        with self.torch.cuda.stream(self.stream):self._matvec(x,y,z,alpha,beta)

    def _matvec(self,x,y,z,alpha,beta):
        n=self.n;torch=self.torch
        values=wp.to_torch(x).view(2,n,n,3).permute(0,3,1,2)
        torch.fft.rfft2(values,out=self.frequency)
        torch.matmul(self.symbol,self.frequency.permute(0,2,3,1).unsqueeze(-1),out=self.product)
        torch.fft.irfft2(self.product[...,0].permute(0,3,1,2),s=(n,n),out=self.real)
        wp.launch(fft_output,2*self.N,inputs=[self.N,self.real_wp,y,z,alpha,beta],device=self.device)


@wp.kernel
def physical_basis(x:wp.array(dtype=V2),V:wp.array(dtype=D),wave:wp.array(dtype=V2),basis:wp.array2d(dtype=D)):
    i,k=wp.tid();w=wave[k//2];angle=wp.dot(w,x[i]);origin=wp.dot(w,x[0])
    value=wp.cos(angle)-wp.cos(origin);gradient=-wp.sin(angle)*w
    if k%2==1:value=wp.sin(angle)-wp.sin(origin);gradient=wp.cos(angle)*w
    basis[3*i,k]=wp.sqrt(V[i])*value
    basis[3*i+1,k]=D(.5)*V[i]*gradient[0];basis[3*i+2,k]=D(.5)*V[i]*gradient[1]


@wp.kernel
def coarse_edges(ii:wp.array(dtype=int),jj:wp.array(dtype=int),ci:wp.array(dtype=V3),cj:wp.array(dtype=V3),
                 s2:wp.array(dtype=D),basis:wp.array2d(dtype=D),Q:wp.array2d(dtype=D)):
    e,k=wp.tid();i=ii[e];j=jj[e]
    a=V3(basis[3*i,k],basis[3*i+1,k],basis[3*i+2,k]);b=V3(basis[3*j,k],basis[3*j+1,k],basis[3*j+2,k])
    Q[e,k]=wp.sqrt(s2[e])*(wp.dot(ci[e],a)+wp.dot(cj[e],b))


@wp.kernel
def add_coarse(n:int,result:wp.array2d(dtype=D),z:wp.array(dtype=V3),alpha:D):
    index=wp.tid();i=index%n;k=index//n
    z[index]+=alpha*V3(result[3*i,k],result[3*i+1,k],result[3*i+2,k])


class PhysicalCoarsePreconditioner:
    """Additive block Jacobi + physical Fourier/Hermite Galerkin correction.

    Coarse columns sample phi(x), grad(phi)/2 in the dual coordinates. They
    capture near-null modes of the edge difference/gradient operator on the
    actual geometry. No logical particle lattice is required. Gram factors
    may be lagged as a preconditioner; the actual operator never is.
    """
    def __init__(self,n,kmax,device,stream):
        import torch
        self.torch=torch;self.n=n;self.device=device;self.stream=wp.stream_to_torch(stream)
        modes=[(a,b) for a in range(kmax+1) for b in range(-kmax,kmax+1) if a>0 or b>0]
        self.K=2*len(modes);K=self.K
        self.wave=wp.array(2*np.pi*np.array(modes,dtype=float),dtype=V2,device=device)
        self.B=torch.empty((3*n,K),dtype=torch.float64,device=str(device));self.Bwp=wp.from_torch(self.B)
        self.W=torch.empty_like(self.B)
        self.H=torch.empty((K,K),dtype=torch.float64,device=str(device));self.L=torch.empty_like(self.H)
        self.info=torch.empty((),dtype=torch.int32,device=str(device))
        self.rhs=torch.empty((K,2),dtype=torch.float64,device=str(device));self.weights=torch.empty_like(self.rhs)
        self.temporary=torch.empty_like(self.rhs)
        self.result=torch.empty((3*n,2),dtype=torch.float64,device=str(device));self.resultwp=wp.from_torch(self.result)
        self.Q=None;self.calls=0;self.factorizations=0

    def update(self,owner,X,V):
        with self.torch.cuda.stream(self.stream):
            wp.launch(physical_basis,(self.n,self.K),inputs=[X,V,self.wave,self.Bwp],device=self.device)
            if self.Q is None or self.Q.shape[0]!=owner.e:
                self.Q=self.torch.empty((owner.e,self.K),dtype=self.torch.float64,device=str(self.device));self.Qwp=wp.from_torch(self.Q)
                self.calls=0
            # Refresh at a bounded interval, or if the preceding solve struggled.
            if self.calls%20==0 or owner.last_info.get('iterations',0)>500:
                wp.launch(coarse_edges,(owner.e,self.K),inputs=[owner.ii,owner.jj,owner.ci,owner.cj,owner.s2,self.Bwp,self.Qwp],device=self.device)
                self.torch.mm(self.Q.T,self.Q,out=self.H)
                self.H.diagonal().add_(1.e-14*self.H.diagonal().max())
                self.torch.linalg.cholesky_ex(self.H,out=(self.L,self.info))
                if int(self.info.item())!=0:raise RuntimeError('Physical coarse Gram factorization failed')
                self.factorizations+=1
            self.calls+=1
            # Whiten once per RHS, avoiding two triangular solves per CG step.
            self.torch.linalg.solve_triangular(self.L,self.B.T,upper=False,out=self.W.T)

    def matvec(self,x,z,alpha):
        with self.torch.cuda.stream(self.stream):
            value=wp.to_torch(x).view(2,3*self.n).T
            self.torch.mm(self.W.T,value,out=self.rhs)
            self.torch.mm(self.W,self.rhs,out=self.result)
            wp.launch(add_coarse,2*self.n,inputs=[self.n,self.resultwp,z,alpha],device=self.device)


@wp.kernel
def hydro_rhs(n:int,offset:wp.array(dtype=int),edge:wp.array(dtype=int),sign:wp.array(dtype=int),neighbor:wp.array(dtype=int),
              q:wp.array(dtype=V2),V:wp.array(dtype=D),mass:wp.array(dtype=D),P:wp.array(dtype=D),v:wp.array(dtype=V2),
              acceleration:wp.array(dtype=V2),dV:wp.array(dtype=D),du:wp.array(dtype=D)):
    i=wp.tid();force=V2(D(0.));volume=D(0.)
    for at in range(offset[i],offset[i+1]):
        j=neighbor[at];qq=D(sign[at])*q[edge[at]]
        force-=(P[i]+P[j])*qq;volume+=wp.dot(qq,v[j]-v[i])
    acceleration[i]=force/mass[i];dV[i]=volume;du[i]=-P[i]*volume/mass[i]


class GPUClosure:
    """Reusable GPU state and checked CPU skin; particle IDs must be stable."""
    def __init__(self,n,device='cuda:0',tolerance=1.e-11,maxiter=4000,check_every=0,logical_n=0,map_jacobian=True,coarse_k=0):
        self.n=n;self.device=wp.get_device(device);self.tolerance=tolerance;self.maxiter=maxiter
        self.check_every=check_every
        self.logical_n=logical_n;self.fft=None;self.coarse=None;self.stream=wp.get_stream(self.device)
        self.map_jacobian=int(map_jacobian);self.Jinv=wp.empty(n,dtype=M2,device=self.device)
        self.previous_Jinv=wp.empty_like(self.Jinv)
        if logical_n or coarse_k:
            if logical_n and coarse_k:raise ValueError('Choose FFT or physical coarse preconditioner')
            if logical_n and logical_n**2!=n:raise ValueError('FFT preconditioner requires a complete logical square lattice')
        if logical_n or coarse_k:
            import torch
            self.stream=wp.Stream(self.device)
            with wp.ScopedStream(self.stream),torch.cuda.stream(wp.stream_to_torch(self.stream)):
                if logical_n:self.fft=FFTPreconditioner(logical_n,self.device,self.stream)
                else:self.coarse=PhysicalCoarsePreconditioner(n,coarse_k,self.device,self.stream)
        self.potential=wp.zeros(2*n,dtype=V3,device=self.device);self.rhs=wp.empty_like(self.potential)
        self.inverse=wp.empty(n,dtype=M3,device=self.device)
        self.batches=wp.array([0,3*n,6*n],dtype=int,device=self.device)
        self.last_x=None;self.solver=None;self.rebuilds=0;self.last_info={}

    def topology(self,x,V):
        R=4*np.sqrt(V)
        if R.max()>=.5:raise ValueError('Support must be below half the unit periodic box')
        if self.last_x is not None:
            dx=x-self.last_x;dx-=np.round(dx)
            if 2*np.linalg.norm(dx,axis=1).max()+np.maximum(R-self.last_R,0).max()<self.skin:return
        self.skin=.25*R.min();pairs=cKDTree(x%1,boxsize=1.).query_pairs(float(R.max()+self.skin),output_type='ndarray')
        i,j=pairs.T;r=x[j]-x[i];r-=np.round(r);keep=np.linalg.norm(r,axis=1)<np.maximum(R[i],R[j])+self.skin
        self.i=i[keep].astype(np.int32);self.j=j[keep].astype(np.int32);self.e=len(self.i)
        owner=np.r_[self.i,self.j];order=np.argsort(owner,kind='stable')
        self.offset=wp.array(np.r_[0,np.cumsum(np.bincount(owner,minlength=self.n))].astype(np.int32),device=self.device)
        self.edge=wp.array(np.tile(np.arange(self.e,dtype=np.int32),2)[order],device=self.device)
        self.sign=wp.array(np.r_[np.ones(self.e,dtype=np.int32),-np.ones(self.e,dtype=np.int32)][order],device=self.device)
        self.neighbor=wp.array(np.r_[self.j,self.i][order],device=self.device)
        self.ii=wp.array(self.i,device=self.device);self.jj=wp.array(self.j,device=self.device)
        self.q0=wp.empty(self.e,dtype=V2,device=self.device);self.q=wp.empty_like(self.q0)
        self.ci=wp.empty(self.e,dtype=V3,device=self.device);self.cj=wp.empty_like(self.ci)
        self.s2=wp.empty(self.e,dtype=D,device=self.device)
        self.last_x=x.copy();self.last_R=R.copy();self.rebuilds+=1

    def solve(self,x,V,return_numpy=True):
        x=np.asarray(x);V=np.asarray(V)
        if x.shape!=(self.n,2) or V.shape!=(self.n,) or not np.isfinite(x).all() or not np.isfinite(V).all() or np.any(V<=0):
            raise ValueError('Need finite 2D positions and positive volumes matching particle IDs')
        with wp.ScopedStream(self.stream):return self._solve(x,V,return_numpy)

    def _solve(self,x,V,return_numpy=True):
        started=time.perf_counter();n=self.n;self.topology(x,V)
        X=wp.array(x,dtype=V2,device=self.device);vol=wp.array(V,dtype=D,device=self.device)
        aa=wp.empty(n,dtype=D,device=self.device);bb=wp.empty(n,dtype=V2,device=self.device)
        da=wp.empty_like(bb);db=wp.empty(n,dtype=M2,device=self.device)
        wp.launch(rk_terms,n,inputs=[X,vol,self.offset,self.neighbor,aa,bb,da,db],device=self.device)
        if self.solver is not None:wp.copy(self.previous_Jinv,self.Jinv)
        wp.launch(logical_mapping,n,inputs=[X,self.logical_n,self.Jinv,self.map_jacobian],device=self.device)
        if self.solver is not None and self.logical_n:
            wp.launch(transport_guess,2*n,inputs=[n,self.previous_Jinv,self.Jinv,self.potential],device=self.device)
        wp.launch(edge_geometry,self.e,inputs=[X,vol,self.ii,self.jj,aa,bb,da,db,self.q0,self.ci,self.cj,self.s2,self.logical_n,self.Jinv],device=self.device)
        wp.launch(prepare,n,inputs=[n,self.offset,self.edge,self.sign,self.ci,self.cj,self.s2,self.q0,self.rhs,self.inverse,vol,self.logical_n,self.Jinv],device=self.device)
        if self.coarse is not None:self.coarse.update(self,X,vol)
        def matvec(x,y,z,alpha,beta):
            wp.launch(gram,2*n,inputs=[n,self.offset,self.edge,self.sign,self.ii,self.jj,self.ci,self.cj,self.s2,x,y,z,alpha,beta,self.logical_n],device=self.device)
        def prec(x,y,z,alpha,beta):
            if self.fft is not None:self.fft.matvec(x,y,z,alpha,beta)
            else:
                wp.launch(precondition,2*n,inputs=[n,self.inverse,x,y,z,alpha,beta],device=self.device)
                if self.coarse is not None:self.coarse.matvec(x,z,alpha)
        operator=LinearOperator((6*n,6*n),D,self.device,matvec,self.batches,3*n)
        pre=LinearOperator((6*n,6*n),D,self.device,prec,self.batches,3*n)
        self.operator=operator;self.preconditioner=pre
        if self.solver is None:
            self.solver=cg(operator,self.rhs,self.potential,M=pre,tol=self.tolerance,atol=self.tolerance,
                maxiter=self.maxiter,check_every=self.check_every,use_cuda_graph=True,run=False)
        iterations,norm,target=self.solver(A=operator,M=pre)
        wp.launch(reconstruct,self.e,inputs=[n,self.ii,self.jj,self.ci,self.cj,self.s2,self.q0,self.potential,self.q,self.logical_n],device=self.device)
        error=wp.zeros(1,dtype=D,device=self.device)
        wp.launch(residual,n,inputs=[n,self.offset,self.edge,self.sign,self.ci,self.cj,self.q,error,vol,self.logical_n,self.Jinv],device=self.device)
        value=float(error.numpy()[0])
        if self.check_every==0:
            iterations=int(iterations.numpy()[0]);norm=float(np.sqrt(np.max(norm.numpy())))
            target=float(np.sqrt(np.min(target.numpy())))
        if not np.isfinite(value) or not np.isfinite(norm) or value>1.e-8:raise RuntimeError(f'GPU closure residual {value}, iterations {iterations}, norm {norm}')
        self.last_info=dict(wallSeconds=time.perf_counter()-started,iterations=int(iterations),constraintResidual=value,
            dualResidual=float(norm),dualTolerance=float(target),topologyRebuilds=self.rebuilds,edges=self.e,mappedPreconditioner=bool(self.logical_n and self.map_jacobian),
            coarseColumns=self.coarse.K if self.coarse is not None else 0,coarseFactorizations=self.coarse.factorizations if self.coarse is not None else 0)
        return (self.i,self.j,self.q.numpy(),self.last_info) if return_numpy else self.last_info

    def pressure_rhs(self,V,mass,P,v):
        with wp.ScopedStream(self.stream):return self._pressure_rhs(V,mass,P,v)

    def _pressure_rhs(self,V,mass,P,v):
        n=self.n;to=lambda x,dtype:wp.array(x,dtype=dtype,device=self.device)
        a=wp.empty(n,dtype=V2,device=self.device);dv=wp.empty(n,dtype=D,device=self.device);du=wp.empty_like(dv)
        wp.launch(hydro_rhs,n,inputs=[n,self.offset,self.edge,self.sign,self.neighbor,self.q,to(V,D),to(mass,D),to(P,D),to(v,V2),a,dv,du],device=self.device)
        return a.numpy(),dv.numpy(),du.numpy()
