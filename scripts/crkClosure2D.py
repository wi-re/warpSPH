"""Experimental periodic 2D vector edge closure; NumPy/SciPy research only.

Unlike the 1D ring construction, this uses a globally coupled least-change
projection. It is not integrated into either production hydrodynamics code.
"""
import time
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix, vstack
from scipy.sparse.linalg import lsmr


def sample(n=24, unequal=True, jitter=0.,shape='planar'):
    """Unit periodic square, 4:1 area/2:1 linear resolution jump at x=0,.5."""
    if n < 16 or n % 4:
        raise ValueError('n must be >=16 and divisible by four')
    def grid(nx, ny, start, width):
        a,b=np.meshgrid(start+(np.arange(nx)+.5)*width/nx,
                        (np.arange(ny)+.5)/ny,indexing='ij')
        return np.stack((a.ravel(),b.ravel()),axis=1)
    if shape=='circle' and unequal:
        fine=grid(n,n,0.,1.);coarse=grid(n//2,n//2,0.,1.)
        fine=fine[np.linalg.norm(fine-.5,axis=1)<.25]
        coarse=coarse[np.linalg.norm(coarse-.5,axis=1)>=.25]
        x=np.concatenate((fine,coarse));V=np.r_[np.full(len(fine),1/n**2),np.full(len(coarse),4/n**2)]
        # Cutting two lattices at a circle does not exactly tile its area.
        # Normalize quadrature volume globally, retaining the 4:1 ratio.
        # These remain approximate cell volumes, not Voronoi areas.
        V/=V.sum()
    elif shape=='planar' or not unequal:
        left=grid(n//2,n,0.,.5)
        right=grid(n//4,n//2,.5,.5) if unequal else grid(n//2,n,.5,.5)
        x=np.concatenate((left,right));V=np.r_[np.full(len(left),.5/len(left)),np.full(len(right),.5/len(right))]
    else:
        raise ValueError('Unknown geometry')
    x=(x+np.random.default_rng(142).uniform(-jitter,jitter,x.shape)*np.sqrt(V[:,None]))%1
    return x,V


def geometry(x,V,length=1.,cutoff=4.):
    """Scatter cubic RK gradients and antisymmetric pressure coefficients.

    Sparse directed sums include self terms. Derivatives include derivatives
    of both RK coefficients, with the derivative index first throughout.
    """
    n=len(V);R=cutoff*np.sqrt(V)
    if np.max(R)>=length/2:
        raise ValueError('Cutoff must be below half the periodic box length')
    pairs=cKDTree(x%length,boxsize=length).query_pairs(float(R.max()),output_type='ndarray')
    i,j=pairs.T
    r=x[j]-x[i];r-=length*np.round(r/length)
    keep=np.linalg.norm(r,axis=1)<np.maximum(R[i],R[j]);i,j,r=i[keep],j[keep],r[keep]
    e=len(i)
    a=np.r_[i,j,np.arange(n)];b=np.r_[j,i,np.arange(n)]
    d=np.concatenate((-r,r,np.zeros((n,2)))) # x_a - x_b
    distance=np.linalg.norm(d,axis=1);z=distance/R[b]
    k=np.where(z<.5,1-6*z*z+6*z**3,np.where(z<1,2*(1-z)**3,0))
    dk=np.where(z<.5,-12*z+18*z*z,np.where(z<1,-6*(1-z)**2,0))
    W=40/(7*np.pi)*k/R[b]**2
    g=40/(7*np.pi)*dk[:,None]/R[b,None]**3*np.divide(d,distance[:,None],out=np.zeros_like(d),where=distance[:,None]>0)
    def sumrow(t):
        result=np.zeros((n,)+t.shape[1:]);np.add.at(result,a,V[b].reshape((-1,)+(1,)*(t.ndim-1))*t)
        return result
    eye=np.eye(2)
    m0=sumrow(W);m1=sumrow(d*W[:,None]);m2=sumrow(d[:,:,None]*d[:,None,:]*W[:,None,None])
    dm0=sumrow(g)
    dm1=sumrow(eye[None]*W[:,None,None]+g[:,:,None]*d[:,None,:])
    dm2=sumrow((eye[None,:,:,None]*d[:,None,None,:]+eye[None,:,None,:]*d[:,None,:,None])*W[:,None,None,None]
               +g[:,:,None,None]*d[:,None,:,None]*d[:,None,None,:])
    inverse=np.linalg.inv(m2);zv=np.einsum('nab,nb->na',inverse,m1)
    A=1/(m0-np.einsum('na,na->n',m1,zv));B=-zv
    dA=-A[:,None]**2*(dm0-2*np.einsum('nka,na->nk',dm1,zv)+np.einsum('na,nkab,nb->nk',zv,dm2,zv))
    dB=-np.einsum('nab,nkb->nka',inverse,dm1)+np.einsum('nab,nkbc,nc->nka',inverse,dm2,zv)
    factor=1+np.einsum('na,na->n',B[a],d)
    K=A[a]*factor*W
    G=(dA[a]*factor[:,None]+A[a,None]*(np.einsum('nka,na->nk',dB[a],d)+B[a]))*W[:,None]+A[a,None]*factor[:,None]*g
    q0=.5*V[i,None]*V[j,None]*(G[:e]-G[e:2*e])
    diagnostics=dict(interpolationConstantError=float(np.max(abs(sumrow(K)-1))),
        interpolationLinearError=float(np.max(abs(sumrow(K[:,None]*d)))),
        gradientConstantError=float(np.max(abs(sumrow(G)))),
        gradientLinearError=float(np.max(abs(sumrow(G[:,:,None]*d[:,None,:])+eye))))
    return i,j,r,q0,R,diagnostics


def compact_taper(V,i,j,r,cutoff=4.):
    """Smooth union-support column scale, zero when both kernels vanish."""
    distance=np.linalg.norm(r,axis=1)
    def cubic(z):
        return np.where(z<.5,1-6*z*z+6*z**3,np.where(z<1,2*(1-z)**3,0.))
    return .5*(cubic(distance/(cutoff*np.sqrt(V[i])))+cubic(distance/(cutoff*np.sqrt(V[j]))))


def close_edges(V,i,j,r,q0,tolerance=1.e-12,edge_scale=None):
    """Enforce sum q=0 and sum q_alpha*r_beta=V*delta_alpha_beta.

    Zero-sum rows scaled by sqrt(V)/V to balance moment rows. The result
    minimizes Euclidean edge change subject to these constraints (iterative),
    or change divided by edge_scale when supplied. A smooth compact scale
    prevents finite corrections from appearing/disappearing at the cutoff.
    No angular-momentum, positivity, shock-stability or locality guarantee.
    """
    started=time.perf_counter();n=len(V);e=len(i);columns=np.r_[np.arange(e),np.arange(e)]
    rows=np.r_[i,j]
    C0=coo_matrix((np.r_[1/np.sqrt(V[i]),-1/np.sqrt(V[j])],(rows,columns)),shape=(n,e)).tocsr()
    moments=[coo_matrix((np.r_[r[:,k]/V[i],r[:,k]/V[j]],(rows,columns)),shape=(n,e)).tocsr() for k in range(2)]
    C=vstack((C0,*moments)).tocsr();q=q0.copy();iterations=[];stops=[]
    scale=np.ones(e) if edge_scale is None else np.asarray(edge_scale)
    if scale.shape!=(e,) or not np.isfinite(scale).all() or np.any(scale<0):
        raise ValueError('Need finite nonnegative scale per edge')
    solve_matrix=C.multiply(scale[None,:]).tocsr() if edge_scale is not None else C
    for k in range(2):
        target=np.zeros((3,n));target[1+k]=1
        solution=lsmr(solve_matrix,target.ravel()-C@q0[:,k],atol=tolerance,btol=tolerance,conlim=1.e12,maxiter=10000)
        q[:,k]+=scale*solution[0];stops.append(int(solution[1]));iterations.append(int(solution[2]))
    error=[]
    for k in range(2):
        target=np.zeros((3,n));target[1+k]=1;error.append(np.max(abs(C@q[:,k]-target.ravel())))
    info=dict(wallSeconds=time.perf_counter()-started,iterations=iterations,stops=stops,
              constraintResidual=float(max(error)),relativeEdgeChangeL2=float(np.linalg.norm(q-q0)/np.linalg.norm(q0)))
    if info['constraintResidual']>1.e-8:
        raise RuntimeError(f'2D closure did not converge: {info}')
    return q,info


def pressure_rhs(V,mass,P,v,i,j,q):
    n=len(V);pair=-(P[i]+P[j])[:,None]*q
    force=np.zeros((n,2));np.add.at(force,i,pair);np.add.at(force,j,-pair)
    work=np.einsum('na,na->n',q,v[j]-v[i])
    dV=np.bincount(i,work,minlength=n)+np.bincount(j,work,minlength=n)
    return force/mass[:,None],dV,-P*dV/mass
