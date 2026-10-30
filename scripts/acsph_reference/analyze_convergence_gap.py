#!/usr/bin/env python3
"""Summarize the stopped campaign and bounded SDF-contract controls."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--campaign',type=Path,required=True)
    ap.add_argument('--investigation',type=Path,required=True)
    args=ap.parse_args();root=args.investigation
    history=json.loads((root/'history-summary.json').read_text())
    result=json.loads((root/'corrected-sdf-prefix236/prefix/result.json').read_text())
    corrected=result['trajectory'][1:]
    old=[json.loads(s) for s in (args.campaign/'warpsph-f32-aligned/progress.jsonl').read_text().splitlines()][1:]
    native=json.loads((args.campaign/'native-f64/result.json').read_text())['steps']
    def controls(rows,key):
        subset=rows[:len(corrected)]; settled=[r for r in subset if r['t']>=.005]
        return {'steps':len(subset),'endpoint':subset[-1]['t'],'caps':sum(r[key]>=200 for r in subset),
            'post_startup_steps':len(settled),'post_startup_caps':sum(r[key]>=200 for r in settled),
            'post_startup_median_iterations':float(np.median([r[key] for r in settled]))}
    stats={'history':history,'native_prefix':controls(native,'iterations'),
        'bad_fixture_prefix':controls(old,'pseudoIterations'),
        'corrected_fixture_prefix':controls(corrected,'pseudoIterations'),
        'corrected_wall_seconds':result['wall_seconds'],
        'corrected_wall_ms_per_step':result['wall_seconds']/len(corrected)*1000,
        'bad_step56':json.loads((root/'step56/target-summary.json').read_text()),
        'corrected_step56':json.loads((root/'corrected-sdf-prefix236/target-summary.json').read_text())}
    (root/'summary.json').write_text(json.dumps(stats,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for name,label,color in [('hard-eager','Bad geometry, hard bounce','#d55e00'),
            ('bounce-off','Same bad geometry, bounce off','#999999'),
            ('smooth-half','Same bad geometry, smooth bounce','#009e73')]:
        data=[json.loads(s) for s in (root/'step56'/(name+'.jsonl')).read_text().splitlines()]
        axes[0].plot([r['iteration']+1 for r in data],[r['epsilonV'] for r in data],label=label,color=color)
    data=[json.loads(s) for s in (root/'corrected-sdf-prefix236/hard-eager.jsonl').read_text().splitlines()]
    axes[0].plot([r['iteration']+1 for r in data],[r['epsilonV'] for r in data],label='Correct SDF, hard bounce',color='#0072b2')
    axes[0].axhline(-8,color='black',ls=':',label='Tolerance')
    axes[0].set(xlabel='Pseudo-iteration',ylabel='log10 position residual metric',title='First failing real step: 56')
    axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
    for rows,key,label,color in [(old,'pseudoIterations','Bad fixture','#d55e00'),
            (native,'iterations','dynGPUSPH','#333333'),(corrected,'pseudoIterations','Corrected fixture','#0072b2')]:
        data=rows[:len(corrected)]
        axes[1].plot([r['t'] for r in data],[r[key] for r in data],'.-',ms=2,lw=.7,label=label,color=color)
    axes[1].set(xlabel='Physical time [s]',ylabel='Iterations per real step',title='Identical physical ICs and reference timesteps')
    axes[1].legend();axes[1].grid(alpha=.2)
    fig.savefig(root/'convergence.png',dpi=170);plt.close(fig)
    print(json.dumps({k:v for k,v in stats.items() if k!='history'},indent=2))


if __name__=='__main__':main()
