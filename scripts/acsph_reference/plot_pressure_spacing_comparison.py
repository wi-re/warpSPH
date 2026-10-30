#!/usr/bin/env python3
"""Compare before/after ghost-spacing wall probes without cadence mixing.

Shared-clock H5 observations include native. Local every-step traces are
displayed separately; native's earlier-RK raw probe.plt is never overlaid.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS={'before':'Before: mass-derived ghost spacing',
        'after':'After: volume-derived ghost spacing',
        'native':'dynGPUSPH FP64 (shared frames)'}
COLORS={'before':'#d55e00','after':'#0072b2','native':'#333333'}
HEIGHTS=[3,15,30,80]


def window_stats(t,p,lo,hi):
    use=(t>=lo)&(t<=hi);indices=np.flatnonzero(use)
    j=indices[np.argmax(p[use])];k=indices[np.argmin(p[use])]
    interior=(t>lo)&(t<hi);clock=np.r_[lo,t[interior],hi]
    return dict(peakPa=float(p[j]),peakTime=float(t[j]),minimumPa=float(p[k]),
                minimumTime=float(t[k]),impulsePaSeconds=float(np.trapezoid(np.interp(clock,t,p),clock)))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('before','after','native','out'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--before-label', default=LABELS['before'])
    ap.add_argument('--after-label', default=LABELS['after'])
    ap.add_argument('--comparison-title', default='before / after ghost-spacing correction')
    args=ap.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    labels=dict(LABELS,before=args.before_label,after=args.after_label)
    start=time.monotonic();common={};dense={};provenance={}
    for name in ('before','after','native'):
        path=getattr(args,name).resolve()
        with h5py.File(path) as h:
            assert h.attrs['complete']
            t=h['time'][:];p=h['probes/pressure'][:];wet=h['probes/wet'][:]
            assert p.shape==(len(t),4) and np.isfinite(p).all()
            assert np.all(np.diff(t)>0)
            if common:np.testing.assert_allclose(t,common['before']['t'],atol=1e-12,rtol=0)
            common[name]=dict(t=t,p=p,wet=wet)
            source=Path(h.attrs['source'])
            provenance[name]=dict(recording=str(path),source=str(source),sharedFrames=len(t),
                sharedPressureSHA256=hashlib.sha256(p.tobytes()).hexdigest())
        if name!='native':
            rows=[json.loads(line) for line in (source/'progress.jsonl').read_text().splitlines()]
            t=np.array([r['t'] for r in rows]);p=np.array([r['pressureProbes'] for r in rows])
            assert len(t)==10001 and np.all(np.diff(t)>0) and np.isfinite(p).all()
            if dense:np.testing.assert_allclose(t,dense['before']['t'],atol=1e-12,rtol=0)
            dense[name]=dict(t=t,p=p)
            provenance[name]['acceptedStepObservations']=len(t)
    plt.rcParams.update({'font.size':10,'axes.titlesize':11,'axes.labelsize':10,
                         'legend.fontsize':10,'svg.fonttype':'none'})

    def finish(fig,name,title,subtitle,handles):
        fig.suptitle(title+'\n'+subtitle,fontsize=13,y=.995)
        fig.legend(*handles,loc='upper center',bbox_to_anchor=(.5,.945),ncol=2 if len(handles[0])==2 else 3,frameon=False)
        for ext in ('png','svg'):fig.savefig(out/(name+'.'+ext),dpi=180)
        plt.close(fig)

    def decorate(ax,k):
        ax.set_title(f'P{k+1}: y = {HEIGHTS[k]} mm, x = 1.61 m')
        ax.set_ylabel('Pressure [kPa]');ax.set_xlabel('Physical time [s]')
        ax.grid(alpha=.22);ax.axhline(0,color='#888888',linewidth=.5)

    fig,axes=plt.subplots(2,2,figsize=(14,8))
    fig.subplots_adjust(top=.83,bottom=.08,left=.065,right=.98,hspace=.34,wspace=.18)
    for k,ax in enumerate(axes.flat):
        for name in ('native','before','after'):
            series=common[name]
            ax.plot(series['t'],series['p'][:,k]/1000,label=labels[name],color=COLORS[name],
                    linewidth=1.2 if name=='native' else 1.6,alpha=.65 if name=='native' else 1.)
        decorate(ax,k);ax.set_xlim(0,common['before']['t'][-1])
    finish(fig,'pressure-shared-frames','Lobovsky wall pressure: '+args.comparison_title,
           'Same 193 post-shift frames (~10 ms); both local runs use ramp 0.5; no collision shift',axes.flat[0].get_legend_handles_labels())

    fig,axes=plt.subplots(2,2,figsize=(14,8))
    fig.subplots_adjust(top=.83,bottom=.08,left=.065,right=.98,hspace=.34,wspace=.18)
    for k,ax in enumerate(axes.flat):
        for name in ('before','after'):
            series=dense[name];ax.plot(series['t'],series['p'][:,k]/1000,color=COLORS[name],label=labels[name],linewidth=.85)
        decorate(ax,k);ax.set_xlim(0,dense['before']['t'][-1])
    finish(fig,'pressure-every-step','Lobovsky wall pressure: unfiltered accepted-step traces',
           '10,001 observations per local run, including initial state; no smoothing or averaging',axes.flat[0].get_legend_handles_labels())

    windows=[(.4,.7),(1.25,1.65)]
    fig,axes=plt.subplots(4,2,figsize=(14,13))
    fig.subplots_adjust(top=.89,bottom=.05,left=.07,right=.98,hspace=.46,wspace=.18)
    for k in range(4):
        for col,(lo,hi) in enumerate(windows):
            ax=axes[k,col]
            for name in ('before','after'):
                series=dense[name];use=(series['t']>=lo)&(series['t']<=hi)
                ax.plot(series['t'][use],series['p'][use,k]/1000,color=COLORS[name],label=labels[name],linewidth=1.)
            decorate(ax,k);ax.set_xlim(lo,hi)
            ax.set_title(f'P{k+1}, y={HEIGHTS[k]} mm — '+('first impact' if col==0 else 'returning-wave interaction'))
    finish(fig,'pressure-impact-details','Lobovsky impacts: local before / after, every accepted step',
           'Identical probe estimator and timestep clocks; axes retain all positive and negative spikes',axes[0,0].get_legend_handles_labels())

    metrics=[]
    for lo,hi in windows+[(.95,1.05)]:
        for k in range(4):
            row=dict(start=lo,end=hi,probe=k+1,heightMm=HEIGHTS[k],sharedFrames={},everyStep={})
            for name,series in common.items():row['sharedFrames'][name]=window_stats(series['t'],series['p'][:,k],lo,hi)
            for name,series in dense.items():row['everyStep'][name]=window_stats(series['t'],series['p'][:,k],lo,hi)
            metrics.append(row)
    (out/'pressure-metrics.json').write_text(json.dumps(dict(sources=provenance,windows=metrics,
        estimator='Post-shift fluid-only normalized Wendland C2 at x=1.61m, support0.003m',
        notes=['No smoothing, averaging or clipping. Dry probe convention is zero.',
               'Native raw probe.plt is a different RK stage/cadence and is not overlaid.'],
        wallSeconds=time.monotonic()-start),indent=2)+'\n')
    print('PLOTS COMPLETE',out,flush=True)
    for k in (0,7):
        row=metrics[k];print('PEAK',row['probe'],row['start'],{n:round(v['peakPa']/1000,3) for n,v in row['everyStep'].items()},flush=True)


if __name__=='__main__':main()
