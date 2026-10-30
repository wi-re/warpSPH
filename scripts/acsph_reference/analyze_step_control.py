#!/usr/bin/env python3
"""Summarize the retained step-468 same-input bounce/precision experiment."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('root',type=Path)
    args=ap.parse_args();root=args.root
    f32=json.loads((root/'float32-prefix/target-summary.json').read_text())
    f64=json.loads((root/'float64-saved/summary.json').read_text())
    reload32=json.loads((root/'float32-saved/summary.json').read_text())
    hard=read_rows(root/'float32-prefix/hard-eager.jsonl')
    double=read_rows(root/'float64-saved/eager.jsonl')
    tail=hard[-20:];last=hard[-1]
    top2=float(np.square(last['top_residuals'][:2]).sum()/last['residual_l2']**2)
    uid=last['top_uids'][0]
    contact=[]
    for row in tail:
        i=row['top_uids'].index(uid)
        contact.append(dict(iteration=row['iteration']+1,
            distance_dx=row['top_wall_contact'][i]['distance_dx'],
            normal_velocity=row['top_wall_contact'][i]['normal_velocity'],
            positionResidualNormal=row['top_residuals'][i][1]))
    prefix=json.loads((root/'float32-prefix/prefix/result.json').read_text())
    summary=dict(step=468,physicalTime=prefix['trajectory'][-1]['t'],
        float32=f32,float64=f64,fp32Reload=reload32,
        top2SquaredResidualFraction=top2,leadingUID=uid,contactTail=contact,
        neighborhoods=json.loads((root/'neighborhoods.json').read_text()),
        prefixWallSeconds=prefix['wall_seconds'],prefixWallMsPerStep=prefix['wall_seconds']/468*1000,
        prefixParity=json.loads((root/'prefix-parity.json').read_text()),
        verdict='Hard position-bounce switching produces a two-cycle in this exact solve; persists in promoted FP64 and matches graph/eager. Same-input off/smooth controls meet the position criterion in13 iterations. Not a full-trajectory remedy qualification.')
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(16,4.8),layout='constrained')
    ax=axes[0]
    for folder,name,label,color,style in [('float32-prefix','hard-eager','FP32 hard bounce','#d55e00','-'),
            ('float64-saved','eager','FP64 same-input hard','#222222','--'),
            ('float32-prefix','bounce-off','FP32 bounce off','#999999','-'),
            ('float32-prefix','smooth-half','FP32 smooth ramp 0.5','#009e73','-'),
            ('float64-saved','smooth','FP64 smooth ramp 0.5','#0072b2',':')]:
        rows=read_rows(root/folder/(name+'.jsonl'))
        ax.plot([r['iteration']+1 for r in rows],[r['epsilonV'] for r in rows],label=label,color=color,ls=style,lw=1.4)
    ax.axhline(-8,color='black',ls=':',lw=1);ax.set(xlabel='Pseudo-iteration',ylabel='Position residual metric, log10',title='Same solve input; no trajectory change')
    ax.legend(fontsize=8)
    ax=axes[1]
    ax.plot([r['iteration'] for r in contact],[r['distance_dx'] for r in contact],'.-',color='#d55e00')
    ax.axhline(.25,color='black',ls='--',label='Hard threshold = 0.25 dx')
    ax.set(xlabel='Pseudo-iteration',ylabel='Distance to nearest wall row / dx',title=f'UID {uid}: threshold crossing');ax.legend(fontsize=8)
    ax=axes[2]
    ax.plot([r['iteration'] for r in contact],[r['positionResidualNormal'] for r in contact],'.-',color='#0072b2')
    ax.axhline(0,color='black',lw=.7)
    ax.set(xlabel='Pseudo-iteration',ylabel='Normal position-row residual [m/s]',title=f'Two particles carry {100*top2:.3f}% of squared residual')
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('Corrected Lobovsky step 468, t=0.163517 s: hard-bounce limit cycle')
    fig.savefig(root/'mechanism.png',dpi=170);plt.close(fig)
    print(json.dumps(dict(top2SquaredFraction=top2,leadingUID=uid,
        fp32HardIterations=f32['hard-eager']['iterations'],fp64HardIterations=f64['controls']['eager']['iterations'],
        fp32SmoothIterations=f32['smooth-half']['iterations'],fp64SmoothIterations=f64['controls']['smooth']['iterations']),indent=2))


if __name__=='__main__':main()
