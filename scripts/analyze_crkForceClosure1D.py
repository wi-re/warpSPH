"""Summarize static edge-closure experiments without advancing a solver."""
import argparse
import json
from pathlib import Path
import numpy as np


def analyze(root,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output.mkdir(parents=True,exist_ok=True)
    data=json.loads((root/'forceClosure.json').read_text())
    a=np.load(root/'n128-ratio4-physical.npz')
    fig,axes=plt.subplots(1,3,figsize=(13,3.8),constrained_layout=True)
    labels=dict(publishedCRK='Published CRK',constantClosure='Constant closure',
                constantAndAffineClosure='Constant + affine closure')
    select=a['observed']; x=a['x'][select]
    rho=np.where(x<0,1.,.25)
    refinement={}
    for variant,label in labels.items():
        axes[0].semilogy(x,np.maximum(np.abs(a[variant+'_constantAcceleration'][select]),1.e-14),'.-',ms=2,label=label)
        axes[1].semilogy(x,np.maximum(np.abs(a[variant+'_affineAcceleration'][select]+1/rho),1.e-14),'.-',ms=2,label=label)
        rows=sorted([r for r in data['rows'] if r['variant']==variant and r['spacingRatio']==4 and r['volume']=='physical'],key=lambda r:r['nLeft'])
        n=np.array([r['nLeft'] for r in rows]);e=np.array([r['sineAccelerationL1'] for r in rows])
        axes[2].loglog(n,e,'.-',label=label)
        refinement[variant]=dict(nLeft=n.tolist(),sineL1=e.tolist(),
            observedOrders=(np.log(e[:-1]/e[1:])/np.log(n[1:]/n[:-1])).tolist())
    axes[0].set(title='Constant-pressure acceleration',xlabel='x',ylabel='absolute acceleration')
    axes[1].set(title='Linear-pressure acceleration error',xlabel='x',ylabel='absolute error')
    axes[2].set(title='Smooth periodic pressure: refinement',xlabel='left particle count',ylabel='volume-weighted L1 acceleration error')
    for ax in axes:ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    fig.suptitle('Static 1D force geometry: 4:1 spacing jump; experimental global edge projections')
    fig.savefig(output/'forceClosure.png',dpi=160);plt.close(fig)
    data.update(refinement=refinement,source=str((root/'forceClosure.json').resolve()))
    (output/'summary.json').write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(refinement,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();analyze(a.root,a.output)
