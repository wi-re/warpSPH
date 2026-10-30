#!/usr/bin/env python3
"""Compare sparse capped-step controls in a preserved physical prefix."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('root',type=Path)
    ap.add_argument('--overnight',type=Path,required=True)
    ap.add_argument('--native',type=Path,required=True)
    a=ap.parse_args();root=a.root
    prefix=root/'prefix606';diagnostics=rows(prefix/'prefix/progress.jsonl')
    prior=rows(a.overnight/'progress.jsonl')[:len(diagnostics)]
    assert diagnostics==prior, 'Instrumented samples changed the physical prefix'
    native=json.loads((a.native/'result.json').read_text())['steps']
    targets=sorted(prefix.glob('step-*'),key=lambda p:int(p.name.split('-')[-1]))
    fig,axes=plt.subplots(len(targets),2,figsize=(12,3.7*len(targets)),layout='constrained',squeeze=False)
    records=[]
    for k,p in enumerate(targets):
        step=int(p.name.split('-')[-1]);report=json.loads((p/'target-summary.json').read_text())
        hard=rows(p/'hard-eager.jsonl');last=hard[-1];uid=last['top_uids'][0]
        two=float(np.square(last['top_residuals'][:2]).sum()/last['residual_l2']**2)
        trace=[]
        for row in hard[-30:]:
            if uid not in row['top_uids']:continue
            j=row['top_uids'].index(uid);contact=row['top_wall_contact'][j]
            trace.append(dict(iteration=row['iteration']+1,distanceDx=contact['distance_dx'],
                normal=contact['normal'],normalVelocity=contact['normal_velocity'],
                normalPositionResidual=float(np.dot(row['top_residuals'][j],contact['normal']))))
        distances=[r['distanceDx'] for r in trace]
        d=np.asarray(distances)
        period=next((lag for lag in range(1,min(9,len(d)))
                     if float(np.max(np.abs(d[lag:]-d[:-lag])))<1e-6),None)
        equal=all(report['hard-eager'][field]==0 for field in
            ['position_max_difference','velocity_max_difference','pressure_rms_difference'])
        record=dict(step=step,time=native[step-1]['t'],nativeIterations=native[step-1]['iterations'],
            controls=report,topUID=uid,top1SquaredFraction=last['top1_fraction'],
            top2SquaredFraction=two,top10SquaredFraction=last['top10_fraction'],
            minimumTailDistanceDx=min(distances),maximumTailDistanceDx=max(distances),
            tailCrossesHardThreshold=min(distances)<.25<max(distances),
            endpointDistanceApproximatePeriod=period,periodToleranceDx=1e-6,
            graphEagerBitwiseEqual=equal,contactTrace=trace,
            lastTrimmed95Metric=last.get('trimmed95_epsilonV'))
        records.append(record)
        for name,label,color in [('hard-eager','Hard bounce','#d55e00'),
                ('bounce-off','Bounce off','#999999'),('smooth-half','Smooth ramp 0.5','#009e73')]:
            data=rows(p/(name+'.jsonl'))
            axes[k,0].plot([r['iteration']+1 for r in data],[r['epsilonV'] for r in data],label=label,color=color)
        axes[k,0].axhline(-8,color='black',ls=':',lw=1)
        axes[k,0].set(title=f'Step {step}, t={record["time"]:.6f} s',ylabel='Position metric, log10',xlabel='Pseudo-iteration')
        axes[k,0].legend(fontsize=8)
        axes[k,1].plot([r['iteration'] for r in trace],distances,'.-',color='#0072b2')
        axes[k,1].axhline(.25,color='black',ls='--',lw=1)
        axes[k,1].set(title=f'Leading UID {uid}; top two carry {100*two:.3f}%',ylabel='Distance to wall row / dx',xlabel='Pseudo-iteration')
    for ax in axes.flat:ax.grid(alpha=.2)
    fig.suptitle('Additional isolated caps: same physical prefix, same-input bounce controls')
    fig.savefig(root/'sampled-caps.png',dpi=170);plt.close(fig)
    result=json.loads((prefix/'prefix/result.json').read_text())
    summary=dict(samples=records,physicalDiagnosticRowsCompared=len(diagnostics),
        exactOvernightPrefix=True,prefixSteps=result['nSteps'],prefixEndpoint=result['trajectory'][-1]['t'],
        wallSeconds=result['wall_seconds'],wallMsPerPhysicalStep=result['wall_seconds']/result['nSteps']*1000,
        productionDefaultsChanged=False,scope='Three additional early isolated spikes. Sustained post-impact blocks remain unclassified.')
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({**{k:v for k,v in summary.items() if k!='samples'},'samples':[
        {k:v for k,v in r.items() if k not in ('controls','contactTrace')} for r in records]},indent=2))


if __name__=='__main__':main()
