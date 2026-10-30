#!/usr/bin/env python3
"""Validate exact smooth-prefix captures and summarize spacing-normal controls."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(path):return json.loads(path.read_text())
def rows(path):return [json.loads(s) for s in path.read_text().splitlines()]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prefix',type=Path,required=True)
    p.add_argument('--spacing',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    a=p.parse_args();r=a.prefix
    physical=rows(r/'probe/prefix/progress.jsonl');baseline=rows(a.baseline/'progress.jsonl')
    assert physical==baseline[:len(physical)]
    receipts=rows(r/'probe/prefix-step-receipts.jsonl')
    assert len(receipts)==1727
    assert all(x['iterations']==physical[i+1]['pseudoIterations'] and x['epsilonV']==physical[i+1]['epsilonV'] for i,x in enumerate(receipts))
    n=read(r/'probe/prefix/initial-audit.json')['fluid_particles']
    report=dict(prefixSteps=1727,allPhysicalDiagnosticsExact=True,allIterationReceiptsExact=True,targets={})
    fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    for row,step in enumerate((1707,1727)):
        target=r/f'probe/step-{step}';base=read(target/'target-summary.json')
        fresh=read(a.spacing/f'step-{step}/summary.json')
        graph=torch.load(target/'original-graph-result.pt',weights_only=True,map_location='cpu')
        eager=torch.load(target/'base-eager-result.pt',weights_only=True,map_location='cpu')
        assert graph.keys()==eager.keys() and all(torch.equal(graph[k],eager[k]) for k in graph)
        corrected=torch.load(a.spacing/f'step-{step}/spacing-normals-result.pt',weights_only=True,map_location='cpu')
        trace=rows(target/'base-eager.jsonl');tail=trace[-30:]
        normal=np.array([x['top_wall_contact'][0]['normal'] for x in tail])
        period=None
        position=np.array([x['top_positions'][0] for x in tail])
        for lag in range(1,9):
            if np.max(np.abs(position[lag:]-position[:-lag]))<1e-10:period=lag;break
        record=dict(time=physical[step]['t'],controls=base,spacingControl=fresh['controls']['spacing-normals'],
            freshOriginalBitwise=fresh['controls']['graph']['originalResultBitwiseEqual'],graphEagerAllFiveBitwise=True,
            finalTopUID=trace[-1]['top_uids'][0],finalTop1Fraction=trace[-1]['top1_fraction'],
            finalContact=trace[-1]['top_wall_contact'][0],endpointPeriod=period,
            normalXRange=normal[:,0].min().item(),normalXRangeMax=normal[:,0].max().item(),
            correctedFluidPressureRMS=float((corrected['pressures'][:n]-graph['pressures'][:n]).square().mean().sqrt()))
        report['targets'][str(step)]=record
        for label,color in [('base-eager','#0072b2'),('bounce-off','#009e73'),('hard-switch','#d55e00'),('smooth-full','#cc79a7')]:
            values=rows(target/(label+'.jsonl'));axes[row,0].plot([x['iteration'] for x in values],[x['epsilonV'] for x in values],label=label,color=color)
        values=rows(a.spacing/f'step-{step}/spacing-normals.jsonl')
        axes[row,0].plot([x['iteration'] for x in values],[x['epsilonV'] for x in values],label='correct spacing normals',color='black',linestyle='--')
        axes[row,0].axhline(-8,color='gray',linestyle=':');axes[row,0].set_ylabel(f'Step {step}: log position metric');axes[row,0].set_xlabel('Pseudo-iteration')
        contact=trace[-30:];index=[x['iteration'] for x in contact]
        axes[row,1].plot(index,[x['top_wall_contact'][0]['distance_dx'] for x in contact],label='nearest normal distance / dx')
        axes[row,1].plot(index,[x['top_wall_contact'][0]['normal'][0] for x in contact],label='selected normal x component')
        axes[row,1].set_xlabel('Pseudo-iteration');axes[row,1].set_ylabel(f'UID {trace[-1]["top_uids"][0]}')
    axes[0,0].legend(fontsize=8);axes[0,1].legend(fontsize=8)
    for ax in axes.flat:ax.grid(alpha=.25)
    fig.savefig(r/'remaining-cap-mechanism.png',dpi=170);plt.close(fig)
    (r/'analysis-summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({s:{k:v[k] for k in ('finalTopUID','finalTop1Fraction','endpointPeriod','correctedFluidPressureRMS')} for s,v in report['targets'].items()},indent=2))


if __name__=='__main__':main()
