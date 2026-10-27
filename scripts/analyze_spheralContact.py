"""CPU-only comparison of recorded Spheral/warpSPH contact experiments."""
import argparse
import json
from pathlib import Path

import numpy as np


def analyze(root, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output.mkdir(parents=True, exist_ok=True)
    pairs=[]
    for directory in sorted(root.glob('*-*')):
        if not all((directory/s/'comparisonRun.json').exists() for s in ('warp','spheral')):
            continue
        records={s:json.loads((directory/s/'comparisonRun.json').read_text()) for s in ('warp','spheral')}
        rhs={s:dict(np.load(directory/s/'initialRHS.npz')) for s in records}
        # Same particles/UIDs: compare by identity, never nearest-neighbor remap.
        for k in ('UIDs','positions','masses','materials','internalEnergies'):
            np.testing.assert_array_equal(rhs['warp'][k],rhs['spheral'][k],err_msg=k)
        mask=np.abs(rhs['warp']['positions'][:,0])<.5
        difference=rhs['warp']['acceleration'][mask]-rhs['spheral']['acceleration'][mask]
        scale=np.linalg.norm(rhs['spheral']['acceleration'][mask])
        initial=dict(accelerationDifferenceMax=float(np.abs(difference).max()),
            accelerationRelativeL2=float(np.linalg.norm(difference)/scale),
            densityDifferenceMax=float(np.abs(rhs['warp']['densities'][mask]-rhs['spheral']['densities'][mask]).max()),
            supportDifferenceMax=float(np.abs(rhs['warp']['supports'][mask]-rhs['spheral']['supports'][mask]).max()))
        fig,axes=plt.subplots(2,3,figsize=(12,6.5),constrained_layout=True)
        invariant_checks={}
        for solver,label in (('warp','warpSPH (float32)'),('spheral','Spheral (float64)')):
            data=dict(np.load(directory/solver/'finalState.npz'))
            original=dict(np.load(directory/solver/'initialState.npz'))
            for field in ('UIDs','materials','masses'):
                np.testing.assert_array_equal(data[field],original[field],err_msg=solver+' '+field)
            mass_error=max(abs(data['masses'][data['materials']==k].sum()/
                               original['masses'][original['materials']==k].sum()-1) for k in (0,1))
            invariant_checks[solver]=dict(materialMassError=float(mass_error),identitiesUnchanged=True)
            if records[solver]['support']=='fixed':
                np.testing.assert_array_equal(data['supports'],original['supports'])
                invariant_checks[solver]['fixedSupportUnchanged']=True
            x=data['positions'][:,0]; select=np.abs(x)<.1
            for ax,field in zip(axes[0],('densities','pressures','velocities')):
                v=data[field].reshape(-1)
                ax.plot(x[select],v[select],'.-',ms=3,lw=.7,label=label)
                ax.set(xlabel='x',title='Final '+field)
            x0=rhs[solver]['positions'][:,0]; select0=np.abs(x0)<.1
            axes[1,0].plot(x0[select0],rhs[solver]['acceleration'][select0,0],'.-',ms=3,lw=.7,label=label)
            history=json.loads((directory/solver/'diagnosticHistory.json').read_text())
            for ax,field in zip(axes[1,1:],('pressureError','maxPeculiarVelocity')):
                ax.plot([r['time'] for r in history],[r[field] for r in history],label=label)
                ax.set(xlabel='time',title=field+' (|x| < 0.5)')
        axes[1,0].set(xlabel='x',title='Initial acceleration, before evolution')
        for ax in axes.flat: ax.grid(alpha=.2)
        axes[0,0].legend(fontsize=8)
        fig.suptitle(directory.name+'; common cubic kernel, two ideal gases; t = 0.02')
        figure=output/(directory.name+'.png');fig.savefig(figure,dpi=160);plt.close(fig)
        row=dict(case=directory.name,initialComparison=initial,invariants=invariant_checks,runs=records,
                 source=str(directory.resolve()),figure=str(figure.resolve()))
        pairs.append(row)
        print(directory.name,json.dumps(initial),flush=True)
    result=dict(pairs=pairs,limits='Actual pinned Spheral trajectories. Fixed support isolates force response; adaptive controls retain different support updates and AV/compatible-energy details. Equal-mass equilibrium failures are retained.')
    (output/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    analyze(a.root,a.output)
