"""Summarize the static solve and daemon-recorded research trajectories."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np


def analyze(root,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output.mkdir(parents=True,exist_ok=True)
    static=json.loads((root/'static/structuredClosure.json').read_text())
    runs=[]
    for path in sorted(root.glob('*/researchRun.json')):
        d=json.loads(path.read_text());d['source']=str(path.resolve());runs.append(d)
    names=('finite-coarse','finite-medium','finite-fine')
    fields=('positions','velocities','volumes','internalEnergies','pressures')
    states=[dict(np.load(root/name/'finalState.npz')) for name in names]
    for a in states[1:]:np.testing.assert_array_equal(a['UIDs'],states[0]['UIDs'])
    convergence={}
    weights=states[0]['initialVolumes'];weights=weights/weights.sum()
    for field in fields:
        errors=[]
        for a,b in zip(states[:-1],states[1:]):
            delta=(a[field]-b[field]).reshape(-1)
            if field=='positions':delta=(delta+1)%2-1
            errors.append(float(np.dot(weights,abs(delta))))
        convergence[field]=dict(coarseMediumL1=errors[0],mediumFineL1=errors[1],
            differenceRatio=errors[0]/errors[1],observedOrder=float(np.log2(errors[0]/errors[1])))
    fig,axes=plt.subplots(1,3,figsize=(13,3.8),constrained_layout=True)
    wave=dict(np.load(root/'acoustic/finalState.npz'))
    t=json.loads((root/'acoustic/researchRun.json').read_text())['time'];x=wave['positions'][:,0]
    c=np.sqrt(1.4)
    axes[0].plot(x,1.e-4*np.sin(np.pi*x)*np.cos(c*np.pi*t),'k-',label='linear acoustic target')
    axes[0].plot(x,wave['pressures']-1,'.',ms=3,label='experimental model')
    axes[0].set(title='Small-amplitude wave at t=0.1',xlabel='x',ylabel='P - 1');axes[0].legend(fontsize=7)
    for name,label in zip(names,('dt=2e-4','dt=1e-4','dt=5e-5')):
        history=json.loads((root/name/'diagnosticHistory.json').read_text())
        axes[1].semilogy([r['time'] for r in history],np.maximum([abs(r['relativeEnergyDrift']) for r in history],1.e-17),label=label)
    axes[1].set(title='Finite-amplitude compression: energy',xlabel='time',ylabel='absolute relative energy drift');axes[1].legend(fontsize=7)
    costs=static['costOnlyBenchmark']
    axes[2].loglog([r['particles'] for r in costs],[r['medianSolveMilliseconds'] for r in costs],'.-')
    axes[2].set(title='Standalone solve cost (synthetic q0)',xlabel='particle count',ylabel='median milliseconds')
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('Experimental periodic 1D cycle closure; coherent pressure/volume work; no AV')
    fig.savefig(output/'structuredClosure.png',dpi=160);plt.close(fig)
    result=dict(static=static,runs=runs,timeConvergence=convergence)
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    shutil.copy(root/'spheralInterfaceCheck.json',output/'spheralInterfaceCheck.json')
    print(json.dumps(convergence,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();analyze(a.root,a.output)
