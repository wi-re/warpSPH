"""Static research probe: conservative edge closure versus affine force accuracy.

The baseline is published CRK, whose conservation/consistency tradeoff is
explicit in Frontiere et al. 2017 Appendix B. Sparse edge projections below
are algebra experiments, not a production method or a hydrodynamic run.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from scipy.sparse import coo_matrix, vstack
from scipy.sparse.linalg import lsmr

from probe_crkSupportClosure import cubic, corrected


def experiment(n,ratio,volume_kind='physical'):
    nr=n//ratio
    x=np.r_[-1+(np.arange(n)+.5)/n,(np.arange(nr)+.5)/nr]
    cell=np.r_[np.full(n,1./n),np.full(nr,1./nr)]
    right=np.roll(x,-1)-x; right[-1]+=2
    voronoi=.5*(right+np.roll(right,1))
    volume=cell if volume_kind=='physical' else voronoi
    rho=np.where(x<0,1.,.25); mass=volume*rho
    R=4*cell
    delta=x[:,None]-x[None,:];delta-=2*np.round(delta/2)
    W,dW=cubic(delta,R[None,:]);K,G=corrected(delta,W,dW,volume)
    q=.5*volume[:,None]*volume[None,:]*(G-G.T)
    # Exactly the union of compact supports, with each undirected edge once.
    i,j=np.where(np.triu((np.abs(delta)<np.maximum(R[:,None],R[None,:])) & (np.abs(delta)>0),1))
    e=np.arange(len(i)); N=len(x); r=-delta[i,j]
    C0=coo_matrix((np.r_[np.ones(len(e)),-np.ones(len(e))],
        (np.r_[i,j],np.r_[e,e])),shape=(N,len(e))).tocsr()
    C1=coo_matrix((np.r_[r/volume[i],r/volume[j]],
        (np.r_[i,j],np.r_[e,e])),shape=(N,len(e))).tocsr()
    q0=q[i,j]
    mask=np.abs(x)<.5
    rows=[]; arrays=dict(x=x,volume=volume,physicalCellVolume=cell,support=R,observed=mask,i=i,j=j)
    for variant in ('publishedCRK','constantClosure','constantAndAffineClosure'):
        started=time.perf_counter()
        if variant=='publishedCRK':
            edge=q0; iterations=0; stop=0
        else:
            C=C0 if variant=='constantClosure' else vstack((C0,C1)).tocsr()
            b=np.zeros(N) if variant=='constantClosure' else np.r_[np.zeros(N),np.ones(N)]
            solution=lsmr(C,b-C@q0,atol=1.e-14,btol=1.e-14,conlim=1.e13,maxiter=30000)
            edge=q0+solution[0]; stop=int(solution[1]);iterations=int(solution[2])
        wall=time.perf_counter()-started
        Q=np.zeros((N,N));Q[i,j]=edge;Q[j,i]=-edge
        def acceleration(P): return -((P[:,None]+P[None,:])*Q).sum(1)/mass
        ac=acceleration(np.ones(N))
        al=acceleration(1+x); al0=acceleration(x)
        ap=acceleration(1+.1*np.sin(np.pi*x));expected=-.1*np.pi*np.cos(np.pi*x)/rho
        # Test pair work with nontrivial pressure and velocity, without a
        # hydro step: matched pressure work cancels instantaneous kinetic power.
        P=1+.1*np.sin(np.pi*x);velocity=.13*np.sin(2*np.pi*x)+.07*np.cos(np.pi*x)
        force=-(P[:,None]+P[None,:])*Q
        thermal_pair=P[None,:]*Q*(velocity[:,None]-velocity[None,:])
        power=float((velocity*force.sum(1)).sum()+thermal_pair.sum())
        correction=edge-q0
        far=(np.minimum(np.abs(x),1-np.abs(x))>.25)
        far_edges=far[i]&far[j]
        force_density=-(2*Q).sum(1)/volume
        row=dict(nLeft=n,spacingRatio=ratio,volume=volume_kind,variant=variant,particles=N,edges=len(e),
            constantAccelerationMax=float(np.abs(ac[mask]).max()),
            affineAccelerationErrorMax=float(np.abs(al[mask]+1/rho[mask]).max()),
            pressureOffsetSensitivityMax=float(np.abs(al[mask]-al0[mask]).max()),
            closureResidualMax=float(np.abs(C0@edge).max()),
            affineMomentErrorMax=float(np.abs(C1@edge-1).max()),
            sineAccelerationL1=float(np.sum(volume*np.abs(ap-expected))/volume.sum()),
            sineAccelerationMaxError=float(np.max(np.abs(ap-expected))),
            relativeEdgeChangeL2=float(np.linalg.norm(correction)/np.linalg.norm(q0)),
            bulkEdgeChangeMax=float(np.max(np.abs(correction[far_edges]))),
            reversedOrientedEdgeFraction=float(np.mean(edge*r<0)),
            pairForceReciprocityMax=float(np.max(np.abs(force+force.T))),
            totalForce=float(force.sum()),pairPowerResidual=power,iterations=iterations,
            solverStop=stop,projectionWallSeconds=wall,
            volumeRelativeDifferenceMax=float(np.max(np.abs(volume/cell-1))),
            directedConstantGradientMax=float(np.max(np.abs((G*volume[None,:]).sum(1)[mask]))),
            directedAffineGradientErrorMax=float(np.max(np.abs((G*volume[None,:]*(-delta)).sum(1)[mask]-1))))
        row['conservationPassed']=row['pairForceReciprocityMax']==0 and abs(row['totalForce'])<1.e-10 and abs(power)<1.e-10
        assert row['conservationPassed'],row
        if variant!='publishedCRK': assert row['constantAccelerationMax']<1.e-6,row
        if variant=='constantAndAffineClosure': assert row['affineAccelerationErrorMax']<1.e-6,row
        rows.append(row)
        for name,value in dict(edge=edge,constantAcceleration=ac,affineAcceleration=al,
            sineAcceleration=ap,pressureForceDensity=force_density).items(): arrays[variant+'_'+name]=value
    return rows,arrays


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=False)
    rows=[]
    jobs=[(128,r,'physical') for r in (1,2,4,8)]+[(128,4,'voronoi')]+[(n,4,'physical') for n in (64,256)]
    for n,ratio,volume in jobs:
        current,arrays=experiment(n,ratio,volume); rows+=current
        np.savez_compressed(args.output/f'n{n}-ratio{ratio}-{volume}.npz',**arrays)
        print(f'n={n} ratio={ratio} volume={volume}: '+', '.join(
            f'{r["variant"]} a0={r["constantAccelerationMax"]:.3g} affineError={r["affineAccelerationErrorMax"]:.3g} iterations={r["iterations"]}' for r in current),flush=True)
    result=dict(rows=rows,limits='Static double edge-geometry experiments. Projections are globally coupled least-change solves; no GPU implementation, evolving geometry, stability, density/AV integration or time-discrete compatible-energy qualification is claimed.')
    (args.output/'forceClosure.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__': main()
