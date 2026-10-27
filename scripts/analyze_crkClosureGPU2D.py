"""Summarize GPU research trajectories and matched CPU/GPU evolution."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=a.input;report=a.output;report.mkdir(parents=True,exist_ok=True)
    runs={p.parent.name:json.loads(p.read_text()) for p in root.glob('*/researchRun.json')}
    comparisons={}
    for cpu,gpu in [('acoustic-cpu','acoustic-gpu'),('acoustic-cpu','acoustic-coarse'),('kh-cpu-n24','kh-coarse-n24')]:
        if not all((root/name/'finalState.npz').exists() for name in (cpu,gpu)):continue
        left=np.load(root/cpu/'finalState.npz');right=np.load(root/gpu/'finalState.npz');fields={}
        for key in ('positions','velocities','volumes','internalEnergies','pressures','densities'):
            delta=right[key]-left[key]
            if key=='positions':delta=(delta+.5)%1-.5
            fields[key]=dict(maxAbsolute=float(np.max(abs(delta))),relativeL2=float(np.linalg.norm(delta)/np.linalg.norm(left[key])))
        comparisons[gpu]=dict(fields=fields,speedup=runs[cpu]['meanStepMilliseconds']/runs[gpu]['meanStepMilliseconds'])
    result=dict(runs=runs,comparisons=comparisons)
    (report/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    names=[name for name in ('acoustic-cpu','acoustic-gpu','acoustic-coarse') if name in runs]
    axes[0].bar(['CPU','GPU FFT','GPU coarse'][:len(names)],[runs[name]['meanStepMilliseconds'] for name in names])
    axes[0].set(ylabel='Full integration ms/step',title='Matched 576-particle acoustic run');axes[0].set_yscale('log')
    for name in ('kh-gpu-n24','kh-coarse-n64'):
        path=root/name/'diagnosticHistory.json'
        if not path.exists():continue
        rows=json.loads(path.read_text());axes[1].plot([r['time'] for r in rows],[r['transverseRMS'] for r in rows],label=name)
    axes[1].set(xlabel='Time',ylabel='Transverse velocity RMS',title='KH evolution (not an exact-reference test)');axes[1].legend(fontsize=8)
    if (root/'kh-coarse-n64/finalState.npz').exists():
        data=np.load(root/'kh-coarse-n64/finalState.npz');s=axes[2].scatter(*data['positions'].T,c=data['materials'],s=3,cmap='coolwarm',vmin=0,vmax=1)
        axes[2].set(xlim=(0,1),ylim=(0,1),title='KH material labels at t=1');axes[2].set_aspect('equal')
    fig.savefig(report/'gpuClosure2D.png',dpi=180);plt.close(fig)
    print(json.dumps(dict(comparisons=comparisons,runs={k:dict(passed=v['passed'],time=v['time'],msPerStep=v.get('meanStepMilliseconds')) for k,v in runs.items()}),indent=2))


if __name__=='__main__':main()
