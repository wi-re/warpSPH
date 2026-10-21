"""Static 2D tensor consistency and cost; no trajectory is advanced."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from crkClosure2D import sample,geometry,close_edges,pressure_rhs


def metrics(x,V,i,j,r,q):
    n=len(V);closure=np.zeros((n,2));moment=np.zeros((n,2,2))
    np.add.at(closure,i,q);np.add.at(closure,j,-q)
    qr=q[:,:,None]*r[:,None,:];np.add.at(moment,i,qr);np.add.at(moment,j,qr)
    P=1+.1*np.sin(2*np.pi*(x[:,0]+x[:,1]));v=np.stack((np.cos(2*np.pi*x[:,1]),.3*np.sin(2*np.pi*x[:,0])),axis=1)
    a,dV,du=pressure_rhs(V,V,P,v,i,j,q)
    constant,_,_=pressure_rhs(V,V,np.ones(n),v,i,j,q)
    # Affine fields on a torus require local unwrapped edge coordinates.
    # The tensor test is valid everywhere, without a discontinuous P=x seam.
    tensor=moment/V[:,None,None]-np.eye(2)
    return dict(constantAccelerationMax=float(np.max(np.linalg.norm(constant,axis=1))),
        affineTensorErrorMax=float(np.max(abs(tensor))),
        offDiagonalMomentMax=float(np.max(abs(tensor[:,(0,1),(1,0)]))),
        momentumRateResidual=float(np.max(abs(np.sum(V[:,None]*a,axis=0)))),
        energyPowerResidual=float(abs(np.sum(V*(np.einsum('na,na->n',v,a)+du)))),
        volumeRateResidual=float(abs(dV.sum())),
        negativeRadialEdgeFraction=float(np.mean(np.einsum('na,na->n',q,r)<0)),
        noncentralEdgeFraction=float(np.mean(abs(q[:,0]*r[:,1]-q[:,1]*r[:,0])>1.e-10)))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False);rows=[]
    jobs=[(24,False,0,'planar'),(24,True,0,'planar'),(24,True,.15,'planar'),
          (32,True,0,'planar'),(48,True,0,'planar'),(48,True,.15,'planar'),(24,True,.1,'circle')]
    for n,unequal,jitter,shape in jobs:
        x,V=sample(n,unequal,jitter,shape);started=time.perf_counter()
        i,j,r,q0,R,directed=geometry(x,V);wall=time.perf_counter()-started
        q,solve=close_edges(V,i,j,r,q0)
        row=dict(n=n,unequal=unequal,jitter=jitter,shape=shape,initialVolumeSum=float(V.sum()),particles=len(V),edges=len(i),geometrySeconds=wall,
                 directed=directed,baseline=metrics(x,V,i,j,r,q0),corrected=metrics(x,V,i,j,r,q),solve=solve)
        assert max(directed.values())<1.e-10,row
        assert row['corrected']['constantAccelerationMax']<1.e-7,row
        assert row['corrected']['affineTensorErrorMax']<1.e-8,row
        assert row['corrected']['energyPowerResidual']<1.e-10,row
        assert row['corrected']['volumeRateResidual']<1.e-8,row
        name=f'n{n}-'+('unequal' if unequal else 'uniform')+f'-jitter{jitter:g}-{shape}'
        np.savez_compressed(args.output/(name+'.npz'),x=x,volume=V,i=i,j=j,r=r,q0=q0,q=q)
        rows.append(row);(args.output/'static.json').write_text(json.dumps(rows,indent=2)+'\n')
        print(json.dumps(row),flush=True)


if __name__=='__main__':main()
