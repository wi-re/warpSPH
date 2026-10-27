"""Static GPU accuracy/cost controls, no hydrodynamic trajectory advances."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from crkClosure2D import sample,geometry,close_edges,compact_taper,pressure_rhs
from crkClosureGPU2D import GPUClosure


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--preconditioner',choices=('jacobi','fft','coarse'),default='coarse')
    p.add_argument('--coarse-k',type=int,default=4)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False);rows=[]
    jobs=[(n,False,.02,'planar') for n in (24,48,64,128)] if args.preconditioner=='fft' else [(24,True,.1,'planar'),(24,True,.1,'circle'),(48,True,.1,'planar')]
    for n,unequal,jitter,shape in jobs:
        x,V=sample(n,unequal,jitter,shape);solver=GPUClosure(len(V),logical_n=n if args.preconditioner=='fft' else 0,
            coarse_k=args.coarse_k if args.preconditioner=='coarse' else 0)
        ii,jj,qq,info=solver.solve(x,V)
        row=dict(n=n,particles=len(V),unequal=unequal,jitter=jitter,shape=shape,preconditioner=args.preconditioner,coldSetup=info)
        if len(V)<=1440:
            i,j,r,q0,R,d=geometry(x,V);q,cpu=close_edges(V,i,j,r,q0,1.e-13,compact_taper(V,i,j,r))
            lookup={(int(a),int(b)):z for a,b,z in zip(ii,jj,qq)}
            actual=np.array([lookup[(int(a),int(b))] for a,b in zip(i,j)])
            raw=solver.q0.numpy();lookup0={(int(a),int(b)):z for a,b,z in zip(ii,jj,raw)}
            raw_actual=np.array([lookup0[(int(a),int(b))] for a,b in zip(i,j)])
            row.update(cpuReference=cpu,coefficientRelativeL2=float(np.linalg.norm(actual-q)/np.linalg.norm(q)),
                rawGeometryRelativeL2=float(np.linalg.norm(raw_actual-q0)/np.linalg.norm(q0)))
            assert row['coefficientRelativeL2']<1.e-7 and row['rawGeometryRelativeL2']<1.e-11,row
        # Cold starts repeatedly require a real solve; unchanged-geometry warm
        # cache hits are explicitly excluded from the reported cold median.
        cold=[];moving=[]
        for k in range(3):
            solver.potential.zero_();_,_,_,s=solver.solve(x,V);cold.append(s['wallSeconds'])
        for k in range(3):
            moved=(x+1.e-4*(k+1)*np.stack((np.sin(2*np.pi*x[:,1]),np.cos(2*np.pi*x[:,0])),axis=1))%1
            _,_,_,s=solver.solve(moved,V);moving.append(s['wallSeconds'])
        row.update(coldMedianSeconds=float(np.median(cold)),movingWarmMedianSeconds=float(np.median(moving)),lastSolve=s)
        P=1+.1*np.sin(2*np.pi*moved.sum(axis=1));velocity=np.stack((np.cos(2*np.pi*moved[:,1]),.3*np.sin(2*np.pi*moved[:,0])),axis=1)
        a,dv,du=solver.pressure_rhs(V,V,P,velocity)
        row['momentumResidual']=float(np.max(abs(np.sum(V[:,None]*a,axis=0))))
        row['powerResidual']=float(abs(np.dot(V,np.einsum('na,na->n',velocity,a)+du)))
        row['volumeRateResidual']=float(abs(dv.sum()))
        assert row['momentumResidual']<1.e-10 and row['powerResidual']<1.e-10 and row['volumeRateResidual']<1.e-8,row
        rows.append(row);(args.output/'static.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(row),flush=True)


if __name__=='__main__':main()
