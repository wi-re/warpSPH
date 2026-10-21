"""Static continuity check at compact-neighbor graph changes, no time advance."""
import argparse
import json
from pathlib import Path
import numpy as np
from crkClosure2D import sample,geometry,close_edges,compact_taper


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();x,V=sample(24);n=len(V)
    perturb=np.stack((np.sin(2*np.pi*x[:,0])*np.cos(2*np.pi*x[:,1]),
                      np.cos(2*np.pi*x[:,0])*np.sin(2*np.pi*x[:,1])),axis=1)
    def evaluate(position):
        i,j,r,q0,R,d=geometry(position,V)
        q,_=close_edges(V,i,j,r,q0,1.e-13)
        qt,_=close_edges(V,i,j,r,q0,1.e-13,compact_taper(V,i,j,r))
        values=[]
        for edge in (q0,q,qt):
            matrix=np.zeros((n,n,2));matrix[i,j]=edge;values.append(matrix)
        return values,set(zip(i,j))
    baseline,initial_edges=evaluate(x);rows=[]
    for epsilon in (1.e-4,1.e-6,1.e-8,1.e-10):
        current,edges=evaluate((x+epsilon*perturb)%1)
        row=dict(displacementScale=epsilon,changedEdges=len(initial_edges.symmetric_difference(edges)),
            relativeCoefficientChanges={name:float(np.linalg.norm(a-b)/np.linalg.norm(b)) for name,a,b in zip(('rawCRK','untapered','tapered'),current,baseline)})
        rows.append(row);print(json.dumps(row),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':main()
