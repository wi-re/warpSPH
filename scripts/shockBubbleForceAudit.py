"""Pure-array bridge checks for native CRK pair forces and RK coefficients.

No time integration. This module does not propose a new force formulation.
"""
import numpy as np


def coefficients(a):
    if 'coefficients' not in a:return {k:a[k] for k in ('A','B','gradA','gradB')}
    c=a['coefficients'];A=c[:,0];B=c[:,1:3]/A[:,None]
    derivative=c[:,3:9].reshape(-1,2,3);gradA=derivative[:,:,0]
    gradB=(derivative[:,:,1:3]-gradA[:,:,None]*B[:,None,:])/A[:,None,None]
    return dict(A=A,B=B,gradA=gradA,gradB=gradB)


def delta(a):
    r=a['positions'][a['i']]-a['positions'][a['j']]
    if 'coefficients' not in a:
        L=np.array([2.2,.8]);r-=np.round(r/L)*L
    return r


def gradient(r,R,A,B,gradA,gradB):
    r,R,A,B,gradA,gradB=(np.asarray(v,dtype=float) for v in (r,R,A,B,gradA,gradB))
    q=np.linalg.norm(r,axis=1)/R
    k=np.where(q<.5,1-6*q*q+6*q**3,np.where(q<1,2*(1-q)**3,0))
    dk=np.where(q<.5,-12*q+18*q*q,np.where(q<1,-6*(1-q)**2,0))
    W=(40/(7*np.pi))*k/R**2
    g=(40/(7*np.pi))*dk[:,None]*r/np.maximum(np.linalg.norm(r,axis=1),1e-300)[:,None]/R[:,None]**3
    dot=np.sum(B*r,axis=1)
    return A[:,None]*(1+dot)[:,None]*g+W[:,None]*(A[:,None]*B+(1+dot)[:,None]*gradA+A[:,None]*np.einsum('nij,nj->ni',gradB,r))


def pair_gradients(a):
    c=coefficients(a);i,j=a['i'],a['j'];r=delta(a)
    return (gradient(r,a['supports'][j],**{k:v[i] for k,v in c.items()}),
            gradient(-r,a['supports'][i],**{k:v[j] for k,v in c.items()}))


def reduce_pairs(a,values):
    out=np.zeros((len(a['masses']),)+values.shape[1:]);np.add.at(out,a['i'],values)
    if 'coefficients' in a:
        ratio=a['masses'][a['i']]/a['masses'][a['j']]
        np.add.at(out,a['j'],-values*ratio.reshape((-1,)+(1,)*(values.ndim-1)))
    return out


def q_pressures(a,scale):
    i,j=a['i'],a['j'];r=delta(a);J=a['qGradient']
    gi=np.einsum('ni,nij,nj->n',r,J[i],r);gj=np.einsum('ni,nij,nj->n',r,J[j],r)
    ri=gi/(np.where(gj<0,-1.,1.)*np.maximum(abs(gj),1e-30))
    rj=gj/(np.where(gi<0,-1.,1.)*np.maximum(abs(gi),1e-30));ratio=np.minimum(ri,rj)
    phi=np.zeros_like(ratio);positive=ratio>0
    phi[positive]=4*ratio[positive]/(1+ratio[positive])**2
    eta=np.minimum(np.linalg.norm(r,axis=1)/a['supports'][i],np.linalg.norm(r,axis=1)/a['supports'][j])
    phi*=np.where(eta<1/3.02,np.exp(-((eta-1/3.02)/(.2/3.02))**2),1.)
    dv=a['velocities'][i]-a['velocities'][j]-.5*phi[:,None]*np.einsum('nij,nj->ni',J[i]+J[j],r)
    ux=np.sum(dv*r,axis=1);rr=np.sum(r*r,axis=1);out=[]
    for side in (i,j):
        h=a['supports'][side]/scale;mu=np.minimum(h*ux/(rr+.01*h*h),0.)
        out.append(a['densities'][side]*(-a['soundspeeds'][side]*mu+.75*mu*mu))
    return out


def native_split(a):
    a={k:v.astype(float) if np.issubdtype(v.dtype,np.floating) else v for k,v in a.items()}
    i,j=a['i'],a['j'];V=a['volumes'];m=a['masses'];P=a['pressures']
    if 'coefficients' in a:
        gi,gj=a['gradientI'],a['gradientJ'];G=.5*(gi-gj)
    else:
        gi,gj=pair_gradients(a);G=.5*(gi-gj)
    ap=-(V[i]*V[j]/m[i])[:,None]*(P[i]+P[j])[:,None]*G
    qi,qj=q_pressures(a,2. if 'coefficients' in a else 1.778002)
    av=-(V[i]*V[j]/m[i])[:,None]*(qi+qj)[:,None]*G
    native=a['nativePairAcceleration'] if 'coefficients' in a else a['pressurePairAcceleration']+a['viscousPairAcceleration']
    real=a['real'];total=reduce_pairs(a,native);computed=reduce_pairs(a,ap+av)
    # Native RHS validates pair ordering, ghost handling and time-level capture.
    error=lambda x,y:float(np.linalg.norm((x-y)[real])/max(np.linalg.norm(y[real]),1e-300))
    dv=a['velocities'][i]-a['velocities'][j];s=np.sum(dv*G,axis=1)
    work_i_q=qj if 'coefficients' in a else .5*(qi+qj)
    dui=V[i]*V[j]/m[i]*(P[j]+work_i_q)*s
    if 'coefficients' in a:
        duj=V[i]*V[j]/m[j]*(P[i]+qi)*s
        u=np.zeros(len(m));np.add.at(u,i,dui);np.add.at(u,j,duj)
    else:
        u=np.zeros(len(m));np.add.at(u,i,dui)
    return dict(pressurePair=ap,viscousPair=av,pressure=reduce_pairs(a,ap),viscous=reduce_pairs(a,av),
        gradientI=gi,gradientJ=gj,dudt=u,checks=dict(nativePairReductionRelativeL2=error(total,a['nativeAcceleration']),
            nativeAccelerationFormulaRelativeL2=error(computed,a['nativeAcceleration']),
            nativeDudtFormulaRelativeL2=error(u,a['nativeDudt']),
            pairFormulaRelativeL2=float(np.linalg.norm(ap+av-native)/max(np.linalg.norm(native),1e-300))))
