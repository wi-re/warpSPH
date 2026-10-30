#!/usr/bin/env python3
"""Offline pressure/convergence screen for the cumulative original-IC ladder."""
import argparse
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from ablation_matrix import notable
from analyze_late_ringing import ringing_metrics


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();root=args.root.resolve();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    tick=time.monotonic();manifest=json.loads((root/'manifest.json').read_text())
    count=manifest['targetSteps']+1
    native=np.genfromtxt(Path(manifest['native'])/'accepted-probes.csv',delimiter=',',names=True)
    clock=native['t'][:count];pn=np.column_stack([native['p'+str(i)] for i in range(1,5)])[:count]
    report={'scope':'Original-IC cumulative hypotheses; adjacent changes may interact',
        'primaryWindow':[1.44,1.53],'targetTime':float(clock[-1]),'rows':[],
        'thresholds':manifest['screening'],'firstNotableAdjacentRow':None,'physicalStepsRunByAnalysis':0}
    data={}
    for row in manifest['rows']:
        key=row['id'];path=Path(manifest['reuse'][key]) if row['reuse'] else root/key
        record=dict(row=row,path=str(path),status='pending',metrics=None,flags=[])
        report['rows'].append(record)
        p=path/'progress.jsonl'
        if not p.exists():continue
        raw=[json.loads(s) for s in p.read_text().splitlines()][:count];valid=[]
        for r in raw:
            if not np.isfinite([r['t'],r['kineticEnergy'],r['maxVelocity'],*r['pressureProbes']]).all():break
            valid.append(r)
        if not valid:record['status']='failed';record['flags']=['no finite observations'];continue
        t=np.array([r['t'] for r in valid]);p=np.array([r['pressureProbes'] for r in valid])
        assert np.array_equal(t,clock[:len(t)]),(key,'physical clocks differ')
        d=data[key]={'t':t,'p':p,'ke':np.array([r['kineticEnergy'] for r in valid]),
            'it':np.array([r['pseudoIterations'] for r in valid])}
        status_path=path/'campaign-row-status.json'
        status=json.loads(status_path.read_text()) if status_path.exists() else None
        failed=bool(status and (status['returncode'] or status['stop_reason'])) or len(valid)!=len(raw)
        record.update(status='reused' if row['reuse'] else 'failed' if failed else 'completed' if len(t)==count else 'partial',
            observedSteps=len(raw)-1,finitePrefixSteps=len(valid)-1,finitePrefixEnd=float(t[-1]),
            physicalFailure=failed,clockMaxAbs=0.,windows={})
        windows=[(.4,.46),(.65,.95),(.95,1.05),(1.25,1.53),(1.44,1.53)]
        for lo,hi in windows:
            m=(t>=lo)&(t<=hi)
            metrics={'probes':{f'P{j+1}':ringing_metrics(t,p[:,j],lo,hi) for j in range(4)},
                'capsInObservedWindow':int((d['it'][m]>=200).sum()),'windowComplete':bool(t[-1]>=hi)}
            if metrics['windowComplete'] and m.any():
                metrics['meanKE']=float(np.trapezoid(d['ke'][m],t[m])/(t[m][-1]-t[m][0]))
                metrics['meanIterations']=float(d['it'][m].mean())
                metrics['pressureRmsDifferenceNativePa']=np.sqrt(np.mean((p[m]-pn[:len(t)][m])**2,axis=0)).tolist()
            record['windows'][f'{lo}..{hi}']=metrics
        main=record['windows']['1.44..1.53'];available=main['windowComplete']
        record['metrics']={'P1RmsPa':main['probes']['P1']['detrended']['0.001']['rmsPa'] if available else None,
            'P4RmsPa':main['probes']['P4']['detrended']['0.001']['rmsPa'] if available else None,
            'caps':main['capsInObservedWindow'] if available else None,'physicalFailure':failed}
        timing=path/'solve-timing.jsonl'
        if timing.exists():
            s=[json.loads(l) for l in timing.read_text().splitlines()][:count-1]
            record['recordedMeanSolveMs']=float(np.mean([r['solve_wall_ms'] for r in s])) if s else None
            record['commonObservedCaps']=int((d['it'][1:]>=200).sum())
        if row['index']:
            previous=report['rows'][-2]
            if previous['metrics'] is not None:
                record['flags']=notable(previous['metrics'],record['metrics'])
                if record['flags'] and report['firstNotableAdjacentRow'] is None:report['firstNotableAdjacentRow']=key
    # Raw traces for each adjacent available pair, preserving failed windows.
    nrows=(len(report['rows'])+2)//3
    fig,axs=plt.subplots(nrows,3,figsize=(15,3*nrows),layout='constrained',squeeze=False)
    for row,ax in zip(report['rows'],axs.flat):
        key=row['row']['id'];index=row['row']['index']
        m=(clock>=1.44)&(clock<=1.53);ax.plot(clock[m],pn[m,0]/1000,color='.6',lw=.6,label='dynGPUSPH FP64')
        for name,color,label in ((manifest['rows'][max(0,index-1)]['id'],'#009e73','Previous stage'),
            (key,'#d55e00',key)):
            if name in data:
                d=data[name];take=(d['t']>=1.44)&(d['t']<=1.53)
                ax.plot(d['t'][take],d['p'][take,0]/1000,color=color,lw=.85,label=label)
        ax.set(title=key+' ['+row['status']+']',xlim=(1.44,1.53),xlabel='Time [s]',ylabel='P1 [kPa]')
        ax.grid(alpha=.2)
    for ax in list(axs.flat)[len(report['rows']):]:ax.axis('off')
    axs.flat[0].legend(fontsize=7)
    fig.suptitle('Cumulative Lobovsky ladder: raw P1 pressure versus preceding stage\nPending/failed coverage is explicit; every physical clock matches; reference differences remain')
    fig.savefig(out/'adjacent-pressure.png',dpi=150);plt.close(fig)
    labels=[r['row']['id'] for r in report['rows']]
    values=[[r['metrics'].get(key) if r['metrics'] else None for r in report['rows']]
        for key in ('P1RmsPa','P4RmsPa','caps')]
    fig,axs=plt.subplots(1,3,figsize=(14,7),layout='constrained')
    for ax,ys,title in zip(axs,values,('P1 ringing RMS [kPa]','P4 ringing RMS [kPa]','Capped solves')):
        ys=[np.nan if y is None else y/(1000 if 'kPa' in title else 1) for y in ys]
        ax.barh(labels,ys,color=['#d55e00' if r['flags'] else '#0072b2' for r in report['rows']])
        ax.invert_yaxis();ax.set(title=title);ax.grid(axis='x',alpha=.2)
    fig.suptitle('Completed1.44–1.53s window; missing bars mean pending/incomplete\nRMS:1ms Gaussian trend removed; first-jump thresholds are descriptive screening only')
    fig.savefig(out/'screening.png',dpi=160);plt.close(fig)
    report['wallSeconds']=time.monotonic()-tick
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# Cumulative Lobovsky ablation','',
        '| Stage | Status | Finite end [s] | P1 RMS [kPa] | P4 RMS [kPa] | Caps | Screen |',
        '| --- | --- | ---: | ---: | ---: | ---: | --- |']
    def number(value,scale=1):return '—' if value is None else f'{value/scale:.3f}'
    for row in report['rows']:
        m=row['metrics'] or {}
        lines.append('| '+ ' | '.join([row['row']['id'],row['status'],number(row.get('finitePrefixEnd')),
            number(m.get('P1RmsPa'),1000),number(m.get('P4RmsPa'),1000),number(m.get('caps')),
            '; '.join(row['flags'])])+' |')
    lines+=['','First notable adjacent row: '+str(report['firstNotableAdjacentRow']),
        '',report['scope'],manifest['screening'],
        'Coupled operator/shift groups require subdivision before attributing a single term.',
        'Recorded costs are shared-system observations. Failed raw outputs are retained.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print('ABLATION ANALYSIS',report['firstNotableAdjacentRow'],report['wallSeconds'],flush=True)


if __name__=='__main__':main()
