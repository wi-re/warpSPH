"""Dense float64, initial-state-only CRK consistency audit; no trajectory.

Separate reproducing-gradient closure from conservative force closure at an
unequal-spacing contact. Alternate kernel/volume choices are algebra controls,
not proposed solver fixes. The scatter baseline is checked against the actual
pinned Spheral acceleration saved by compare_spheralContact.py.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def cubic(x, radius):
    q=np.abs(x)/radius
    k=np.where(q<.5, 1-6*q*q+6*q**3, 2*(1-q)**3)
    dk=np.where(q<.5, -12*q+18*q*q, -6*(1-q)**2)
    k=np.where(q<1,k,0.)
    dk=np.where(q<1,dk,0.)
    return (4./3.)*k/radius, (4./3.)*dk*np.sign(x)/radius**2


def corrected(x, W, dW, volume):
    v=volume[None,:]
    m0=(v*W).sum(1); m1=(v*x*W).sum(1); m2=(v*x*x*W).sum(1)
    d0=(v*dW).sum(1); d1=(v*(W+x*dW)).sum(1)
    d2=(v*(2*x*W+x*x*dW)).sum(1)
    A=1/(m0-m1*m1/m2); B=-m1/m2
    dA=-A*A*(d0-2*m1*d1/m2+m1*m1*d2/(m2*m2))
    dB=-d1/m2+m1*d2/(m2*m2)
    A,B,dA,dB=(z[:,None] for z in (A,B,dA,dB))
    K=A*(1+B*x)*W
    G=(dA*(1+B*x)+A*dB*x+A*B)*W+A*(1+B*x)*dW
    return K,G


def isolate_1d():
    """Exact-double sampling controls: support jump versus quadrature jump."""
    rows=[]
    for sampling,nr in (('equalSpacing',400),('equalMass',100)):
        v=np.concatenate((np.full(400,1./400),np.full(nr,1./nr)))
        x=np.concatenate((-1+(np.arange(400)+.5)/400,(np.arange(nr)+.5)/nr))
        delta=x[:,None]-x[None,:]; delta-=2*np.round(delta/2)
        rho=np.where(x<0,1.,.25)
        m=v*rho
        mask=np.abs(x)<.5
        for support in ('spacingScaled','uniformCoarse'):
            R=4*v if support=='spacingScaled' else np.full(len(x),4./100)
            W,dW=cubic(delta,R[None,:])
            K,G=corrected(delta,W,dW,v)
            pair=-(v[:,None]*v[None,:])*(G-G.T)
            force=pair.sum(1)
            # Density/material changes at prescribed P leave geometry and V
            # untouched. Compare force density, independent of inertial mass.
            closure=force/v
            rows.append(dict(sampling=sampling,support=support,particles=len(x),
                constantGradientMax=float(np.abs((G*v[None,:]).sum(1)[mask]).max()),
                linearGradientErrorMax=float(np.abs((G*v[None,:]*(-delta)).sum(1)[mask]-1).max()),
                pressureForceDensityMax=float(np.abs(closure[mask]).max()),
                accelerationAtDensityJumpMax=float(np.abs(force[mask]/m[mask]).max()),
                accelerationAtUniformDensityMax=float(np.abs(closure[mask]).max()),
                totalForce=float(force.sum())))
            assert rows[-1]['constantGradientMax']<1.e-10,rows[-1]
            assert rows[-1]['linearGradientErrorMax']<1.e-10,rows[-1]
            assert abs(rows[-1]['totalForce'])<1.e-10,rows[-1]
            if sampling=='equalSpacing':
                assert rows[-1]['accelerationAtDensityJumpMax']<1.e-8,rows[-1]
    return rows


def audit(path):
    a=dict(np.load(path/'input.npz'))
    x=a['positions'][:,0].astype(float)
    delta=x[:,None]-x[None,:]; delta-=2*np.round(delta/2)
    radius=a['supports'].astype(float)
    mass=a['masses'].astype(float)
    volume=mass/a['densities'].astype(float)
    Wg,dWg=cubic(delta,radius[:,None]); Ws,dWs=cubic(delta,radius[None,:])
    mask=np.abs(x)<.5
    raw=dict(gather=(Wg,dWg),scatter=(Ws,dWs),symmetric=((Wg+Ws)*.5,(dWg+dWs)*.5))
    rows=[]
    for label in ('physical','gather','scatter','symmetric'):
        v=volume if label=='physical' else 1/raw[label][0].sum(1)
        for mode in ('scatter','symmetric'):
            W,dW=raw[mode]
            K,G=corrected(delta,W,dW,v)
            # P=1. Pair force is antisymmetric by construction, as in CRK.
            force=-(v[:,None]*v[None,:])*(G-G.T)
            acceleration=force.sum(1)/mass
            row=dict(volume=label,momentsAndForceKernel=mode,
                volumeRelativeMax=float(np.max(np.abs(v[mask]/volume[mask]-1))),
                constantReproductionMax=float(np.max(np.abs((K*v[None,:]).sum(1)[mask]-1))),
                linearReproductionMax=float(np.max(np.abs((K*v[None,:]*delta).sum(1)[mask]))),
                constantGradientMax=float(np.max(np.abs((G*v[None,:]).sum(1)[mask]))),
                linearGradientErrorMax=float(np.max(np.abs((G*v[None,:]*(-delta)).sum(1)[mask]-1))),
                conservativeAccelerationMax=float(np.max(np.abs(acceleration[mask]))),
                reciprocalGradientSumMax=float(np.max(np.abs((G.T*v[None,:]).sum(1)[mask]))),
                totalForce=float(np.sum(force)),pairReciprocityMax=float(np.max(np.abs(force+force.T))))
            if label=='physical' and mode=='scatter':
                actual=np.load(path/'spheral/initialRHS.npz')['acceleration'][:,0]
                row['actualReferenceRelativeL2']=float(np.linalg.norm(acceleration[mask]-actual[mask])/np.linalg.norm(actual[mask]))
                row['actualReferenceDifferenceMax']=float(np.max(np.abs(acceleration[mask]-actual[mask])))
                # Table interpolation and ~1e-7 EOS rounding explain tiny disagreement.
                assert row['actualReferenceRelativeL2']<1.e-3 or row['actualReferenceDifferenceMax']<1.e-3,row
            assert row['constantReproductionMax']<1.e-10,row
            assert row['linearGradientErrorMax']<1.e-9,row
            assert row['pairReciprocityMax']==0,row
            rows.append(row)
    # Candidate interactions present in only one directed raw support. The
    # conservative pair uses their union, so this is not a missing-pair count.
    directed=(np.abs(delta)<radius[:,None])
    asymmetric=directed!=directed.T
    out=dict(input=str(path.resolve()),dimension=1,
             oneSidedSupportPairsInObservation=int(np.count_nonzero(asymmetric[mask])),
             exactDoubleSamplingControls=isolate_1d(),
             rows=rows,limits='No dynamics advanced. Density/pressure held prescribed. Changing only volume is a consistency diagnostic, not an EOS-consistent replacement density policy. Symmetric rows rebuild kernel moments and gradients together but have no derived or tested volume/density/energy evolution.')
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    result=audit(args.input)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
