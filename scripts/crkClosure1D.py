"""Experimental periodic 1D conservative-force geometry; NumPy only.

Independent of Warp and Spheral so both can exercise the same research solve.
Requires compact edges connecting consecutive particles and their next-nearest
neighbors. Pair coefficients q_ij are oriented from edge_i to edge_j.
"""
import numpy as np


def close_edges(x,volume,edge_i,edge_j,q0,length):
    """Enforce sum q=0 and sum q*r=V without an iterative projection.

    Sort/map cost is O(N log N + E log E); the ring/Poisson solve is O(N+E).
    Existing edges are retained. Only nearest/next-nearest coefficients change.
    This is a 1D periodic research construction, not a production CRK mode.
    """
    x=np.asarray(x,dtype=np.float64).reshape(-1)
    volume=np.asarray(volume,dtype=np.float64).reshape(-1)
    i=np.asarray(edge_i,dtype=np.int64);j=np.asarray(edge_j,dtype=np.int64)
    q0=np.asarray(q0,dtype=np.float64)
    n=len(x)
    if n<5 or volume.shape!=(n,) or np.any(volume<=0) or not np.isfinite(volume).all():
        raise ValueError('Need >=5 particles and finite positive volumes')
    if not np.isfinite(x).all() or not np.isfinite(length) or length<=0 or np.ptp(x)>=length:
        raise ValueError('Positions must span less than one positive periodic length')
    if i.shape!=j.shape or i.shape!=q0.shape or i.ndim!=1 or not np.isfinite(q0).all():
        raise ValueError('Invalid oriented edge arrays')
    if np.any(i<0) or np.any(j<0) or np.any(i>=n) or np.any(j>=n) or np.any(i==j):
        raise ValueError('Edges must connect distinct valid particle indices')
    order=np.argsort(x); rank=np.empty(n,dtype=np.int64);rank[order]=np.arange(n)
    d=np.diff(np.r_[x[order],x[order[0]]+length])
    if np.any(d<=0) or np.any(d+np.roll(d,-1)>=.5*length):
        raise ValueError('Need distinct ordered particles and local next-nearest edges')
    ri,rj=rank[i],rank[j]
    keys=np.minimum(ri,rj)*n+np.maximum(ri,rj)
    key_order=np.argsort(keys); sorted_keys=keys[key_order]
    if np.any(np.diff(sorted_keys)==0): raise ValueError('Duplicate undirected edges')
    def ring_edges(offset):
        a=np.arange(n);b=(a+offset)%n
        target=np.minimum(a,b)*n+np.maximum(a,b)
        at=np.searchsorted(sorted_keys,target)
        if np.any(at>=len(keys)) or np.any(sorted_keys[np.minimum(at,len(keys)-1)]!=target):
            raise ValueError('Compact graph must contain nearest and next-nearest ring edges')
        edge=key_order[at]
        sign=np.where(ri[edge]==a,1.,-1.)
        return edge,sign
    nearest,sign1=ring_edges(1); next_nearest,sign2=ring_edges(2)
    r=x[j]-x[i]; r-=length*np.round(r/length)
    divergence=np.bincount(i,q0,minlength=n)-np.bincount(j,q0,minlength=n)
    moment=np.bincount(i,q0*r,minlength=n)+np.bincount(j,q0*r,minlength=n)
    div=divergence[order]
    # Repair divergence with directed nearest-ring flow. Fix its circulation
    # so total first moment equals sum(V), making the Poisson RHS compatible.
    flux=-np.cumsum(div-div.mean())
    flux+=(volume.sum()-moment.sum()-2*np.dot(flux,d))/(2*length)
    b=volume[order]-moment[order]-flux*d-np.roll(flux*d,1)
    b-=b.mean()  # Remove roundoff in the analytically zero-mean source.
    # Triangle cycles centered at a node have a weighted-Laplacian moment.
    # Solve (u[j+1]-u[j])/d[j] - (u[j]-u[j-1])/d[j-1] = b[j].
    poisson_flux=np.cumsum(b)
    poisson_flux-=np.dot(poisson_flux,d)/length
    u=np.r_[0.,np.cumsum((d*poisson_flux)[:-1])]
    weight=1/(d*np.roll(d,-1))
    a=np.roll(u,-1)*weight
    A=a+np.roll(a,1); B=weight+np.roll(weight,1)
    # The Poisson constant is free. Choose it to minimize the total squared
    # coefficient change within this nearest/next-nearest correction subspace.
    gauge=(np.dot(B,flux-A)-np.dot(weight,a))/(np.dot(weight,weight)+np.dot(B,B))
    triangle=a+gauge*weight
    q=q0.copy()
    q[nearest]+=sign1*(flux-triangle-np.roll(triangle,1))
    q[next_nearest]+=sign2*triangle
    return q


def pressure_rhs(volume,mass,pressure,velocity,i,j,q):
    """Coherent pressure force, volume divergence and adiabatic local work.

    With rho=m/V and P=(gamma-1)m u/V, these share the same corrected edge
    geometry. Pair antisymmetry gives instantaneous energy/momentum balance;
    closure gives total-volume balance and constant-velocity invariance.
    No artificial viscosity or shock treatment is included.
    """
    n=len(volume)
    pair_force=-(pressure[i]+pressure[j])*q
    force=np.bincount(i,pair_force,minlength=n)-np.bincount(j,pair_force,minlength=n)
    pair_volume=q*(velocity[j]-velocity[i])
    dvolume=np.bincount(i,pair_volume,minlength=n)+np.bincount(j,pair_volume,minlength=n)
    du=-pressure*dvolume/mass
    return force/mass,dvolume,du
