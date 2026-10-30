#!/usr/bin/env python3
"""Compare every-step accepted post-shift dynGPUSPH and warpSPH pressure.

Requires matching clocks. This comparison preserves and names each solver's
bounce and precision; it does not attribute a difference to one model choice.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def extrema(t, p, lo, hi):
    keep = (t >= lo) & (t <= hi)
    x, y = t[keep], p[keep]
    return {'maximumPa':float(y.max()),'maximumTime':float(x[y.argmax()]),
            'minimumPa':float(y.min()),'minimumTime':float(x[y.argmin()])}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference', type=Path, required=True)
    ap.add_argument('--local', type=Path, required=True)
    ap.add_argument('--sparse-reference', type=Path, required=True)
    ap.add_argument('--previous-local', type=Path,
                    help='Optional earlier warpSPH control on the same clock prefix')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(); args.out.mkdir(parents=True,exist_ok=False)
    r = np.genfromtxt(args.reference/'accepted-probes.csv',delimiter=',',names=True)
    rows = [json.loads(line) for line in (args.local/'progress.jsonl').read_text().splitlines()]
    t = np.array([row['t'] for row in rows])
    p = np.array([row['pressureProbes'] for row in rows])
    local_wet = np.array([row['pressureProbesWet'] for row in rows],dtype=bool)
    q = np.column_stack([r['p'+str(i)] for i in range(1,5)])
    assert len(r)==len(t) and np.isfinite(q).all() and np.isfinite(p).all()
    clock_error=float(np.abs(t-r['t']).max())
    assert clock_error<1e-12,clock_error
    with h5py.File(args.sparse_reference) as h:
        sparse_t=h['time'][:];sparse_p=h['probes/pressure'][:]
    report={'scope':'Original-IC first-impact window; matched clocks, probe locations/weights and accepted post-shift stage',
        'dynGPUSPH':'FP64; original hard lagged wall bounce; original pressure-trim bookkeeping',
        'warpSPH':'FP32 reference solver; smooth fresh-stage wall bounce ramp0.5; physical-fluid-ID pressure trim',
        'limitations':'Numerical choices listed above remain different; this is actual dense reference data, not a precision-only or bounce-only attribution.',
        'steps':len(t)-1,'endTime':float(t[-1]),'clockMaxAbs':clock_error,'probes':{}}
    for j in range(4):
        windows={}
        for name,lo,hi in [('firstImpact',.38,.46),('postPeak',.418,.46)]:
            keep=(t>=lo)&(t<=hi)
            both_wet=keep & local_wet[:,j] & (r['w'+str(j+1)]>0)
            windows[name]={'dynGPUSPH':extrema(t,q[:,j],lo,hi),
                'warpSPH':extrema(t,p[:,j],lo,hi),
                'differenceRmsPa':float(np.sqrt(np.mean((p[keep,j]-q[keep,j])**2))),
                'differenceMaxAbsPa':float(np.max(np.abs(p[keep,j]-q[keep,j]))),
                'bothWetSamples':int(both_wet.sum()),
                'bothWetDifferenceRmsPa':float(np.sqrt(np.mean((p[both_wet,j]-q[both_wet,j])**2)))
                    if both_wet.any() else None}
        # A single low-weight neighbour can make a normalized point probe
        # highly sensitive. Retain the denominator and count at the maximum.
        keep=np.flatnonzero((t>=.38)&(t<=.46));peak=keep[q[keep,j].argmax()]
        windows['dynGPUSPHPeakSupport']={'weight':float(r['w'+str(j+1)][peak]),
            'fluidCount':int(r['n'+str(j+1)][peak]),'step':int(r['step'][peak])}
        windows['firstWetTime']={
            'dynGPUSPH':float(t[np.flatnonzero(r['w'+str(j+1)]>0)[0]])
                if (r['w'+str(j+1)]>0).any() else None,
            'warpSPH':float(t[np.flatnonzero(local_wet[:,j])[0]])
                if local_wet[:,j].any() else None}
        report['probes']['P'+str(j+1)]=windows
    if args.previous_local:
        old_rows=[json.loads(line) for line in (args.previous_local/'progress.jsonl').read_text().splitlines()][:len(t)]
        old_t=np.array([row['t'] for row in old_rows])
        old_p=np.array([row['pressureProbes'] for row in old_rows])
        assert len(old_t)==len(t) and np.max(np.abs(old_t-t))<1e-12
        keep=(t>=.38)&(t<=.46)
        before=np.sqrt(np.mean((old_p[keep]-q[keep])**2,axis=0))
        after=np.sqrt(np.mean((p[keep]-q[keep])**2,axis=0))
        report['previousSolverControl']={'source':str(args.previous_local.resolve()),
            'window':[.38,.46], 'previousRmsPa':before.tolist(),
            'repairedRmsPa':after.tolist(),'reductionPercent':((1-after/before)*100).tolist()}
    labels=['P1: y=3 mm','P2: y=15 mm','P3: y=30 mm','P4: y=80 mm']
    for filename,limits in [('pressure-dense-first-impact.png',(.38,.46)),
                            ('pressure-dense-first-pulse.png',(.411,.425))]:
        fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
        for j,ax in enumerate(axes.flat):
            ax.plot(t,q[:,j]/1000,color='0.35',label='dynGPUSPH FP64: original hard bounce',lw=1.4)
            ax.plot(t,p[:,j]/1000,color='tab:orange',label='warpSPH FP32: smooth bounce 0.5',lw=1.)
            ax.scatter(sparse_t,sparse_p[:,j]/1000,color='black',s=14,zorder=3,label='dynGPUSPH old 10 ms snapshots')
            ax.set(xlim=limits,xlabel='Time [s]',ylabel='Pressure [kPa]',title=labels[j]);ax.grid(alpha=.2)
            selected=(t>=limits[0])&(t<=limits[1]);both=np.r_[p[selected,j],q[selected,j]]/1000
            span=max(float(np.ptp(both)),.05);ax.set_ylim(float(both.min())-.05*span,float(both.max())+.1*span)
            if j==0:ax.legend(fontsize=8)
        fig.suptitle('Every accepted post-shift step; identical probes and clocks\nBounce, precision and trim differences remain explicit')
        fig.savefig(args.out/filename,dpi=170);plt.close(fig)
    (args.out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
