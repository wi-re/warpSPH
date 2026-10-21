"""Published 2D wind-cloud geometry, with Spheral's rounded inner lattice."""
import math
import numpy as np


def blobScales(gamma=5/3,mach=2.7,contrast=10.):
    wind=mach*math.sqrt(gamma)
    crush=2*math.sqrt(contrast)/wind
    return dict(wind=wind,tCrush=crush,tKH=1.6*crush)


def sampleBlob(nx=256,massMatch=True,n_h=4.05,gamma=5/3,mach=2.7,contrast=10.,length=1.):
    if nx<64 or nx%4:raise ValueError('blob nx must be >=64 and divisible by four')
    dx=40./nx;ny=nx//4
    xx,yy=np.meshgrid((np.arange(nx)+.5)*dx,(np.arange(ny)+.5)*dx,indexing='ij')
    x=np.column_stack((xx.ravel(),yy.ravel()));inside=np.sum((x-[5.,5.])**2,axis=1)<1
    if massMatch:
        # Same integer-rounded cloud resolution as the pinned 2D driver.
        nc=max(2,int(math.sqrt(4*contrast/dx**2)+.5));dc=2./nc
        cx,cy=np.meshgrid(4+(np.arange(nc)+.5)*dc,4+(np.arange(nc)+.5)*dc,indexing='ij')
        cloud=np.column_stack((cx.ravel(),cy.ravel()));cloud=cloud[np.sum((cloud-[5.,5.])**2,axis=1)<1]
        ambient=x[~inside];x=np.r_[ambient,cloud];ids=np.r_[np.zeros(len(ambient),dtype=np.int32),np.ones(len(cloud),dtype=np.int32)]
        volume=np.r_[np.full(len(ambient),dx**2),np.full(len(cloud),dc**2)]
    else:
        order=np.argsort(inside,kind='stable');x=x[order];ids=inside[order].astype(np.int32);volume=np.full(len(x),dx**2)
    x*=length;volume*=length**2;rho=np.where(ids==1,contrast,1.)
    velocity=np.zeros_like(x);velocity[ids==0,0]=blobScales(gamma,mach,contrast)['wind']
    return dict(positions=x.astype(np.float32),velocities=velocity.astype(np.float32),
        masses=(rho*volume).astype(np.float32),densities=rho.astype(np.float32),pressures=np.ones(len(x),dtype=np.float32),
        internalEnergies=(1/((gamma-1)*rho)).astype(np.float32),supports=(n_h*np.sqrt(volume)).astype(np.float32),
        materials=ids,UIDs=np.arange(len(x),dtype=np.int64),kinds=np.zeros(len(x),dtype=np.int32))
