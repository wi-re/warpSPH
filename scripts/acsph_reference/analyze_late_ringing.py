#!/usr/bin/env python3
"""Dense accepted-state late-ringing comparison, with explicit time support.

Amplitude metrics are separate from pressure difference and impact peaks.
No filtering is applied to the raw plots. Filtered metrics declare their
scale and exclude endpoints; they are descriptive, not physical error norms.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d


def load_local(path):
    rows=[json.loads(line) for line in (path/'progress.jsonl').read_text().splitlines()]
    return {'t':np.array([r['t'] for r in rows]),
        'p':np.array([r['pressureProbes'] for r in rows]),
        'wet':np.array([r['pressureProbesWet'] for r in rows],dtype=bool),
        'ke':np.array([r['kineticEnergy'] for r in rows]),
        'vmax':np.array([r['maxVelocity'] for r in rows]),
        'iterations':np.array([r['pseudoIterations'] for r in rows])}


def ringing_metrics(t,p,lo,hi):
    m=(t>=lo)&(t<=hi)
    if m.sum()<20 or t[-1]+1e-12<hi:
        return {'available':False,'samples':int(m.sum())}
    # Same uniform sampling and filters in both solvers. Remove3sigma at
    # either end. Two scales expose dependence on the trend definition.
    u=np.arange(lo,hi+1e-10,.0001)
    raw=np.interp(u,t,p)
    result={'available':True,'samples':int(m.sum()),'minimumPa':float(p[m].min()),
        'maximumPa':float(p[m].max()),'rawPeakToPeakPa':float(np.ptp(p[m])),
        'maximumTime':float(t[m][p[m].argmax()]),'minimumTime':float(t[m][p[m].argmin()]),
        'detrended':{}}
    for sigma in (.001,.005):
        hp=raw-gaussian_filter1d(raw,sigma/.0001,mode='reflect')
        interior=(u>=lo+3*sigma)&(u<=hi-3*sigma)
        z=hp[interior]
        if not len(z):continue
        result['detrended'][str(sigma)]={'rmsPa':float(np.sqrt(np.mean(z*z))),
            'peakToPeakPa':float(np.ptp(z)),'absoluteP95Pa':float(np.quantile(abs(z),.95)),
            'evaluatedWindow':[float(u[interior][0]),float(u[interior][-1])]}
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference',type=Path,required=True)
    ap.add_argument('--local',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    local=load_local(args.local)
    rows=np.genfromtxt(args.reference/'accepted-probes.csv',delimiter=',',names=True)
    ref={'t':rows['t'],'p':np.column_stack([rows['p'+str(j)] for j in range(1,5)])}
    n=min(len(ref['t']),len(local['t']))
    assert n>1 and np.all(np.diff(local['t'])>0) and np.all(np.diff(ref['t'])>0)
    assert np.isfinite(local['p']).all() and np.isfinite(ref['p']).all()
    clock=float(np.abs(ref['t'][:n]-local['t'][:n]).max())
    # No implicit interpolation of a mismatched physical step sequence.
    assert clock<1e-12,clock
    report={'scope':'Original-IC late comparison; accepted post-shift every-step pressures',
        'dynGPUSPH':'FP64, original hard lagged bounce and source pressure trim',
        'warpSPH':'FP32 repaired reference solver, smooth fresh-stage bounce0.5 and physical-fluid-ID trim',
        'clockMaxAbs':clock,'commonEnd':float(min(ref['t'][-1],local['t'][-1])),
        'fullLateCoverage':bool(min(ref['t'][-1],local['t'][-1])>=1.70),
        'filters':'Uniform0.1ms sampling; Gaussian trend sigma1ms and5ms; exclude3sigma edges. Raw peak-to-peak includes impact/trend.',
        'windows':{}}
    windows=[(.65,.95),(.95,1.05),(1.25,1.65),(1.44,1.53),(1.59,1.68)]
    for lo,hi in windows:
        probes={}
        for j in range(4):
            d={name:ringing_metrics(data['t'],data['p'][:,j],lo,hi)
               for name,data in [('dynGPUSPH',ref),('warpSPH',local)]}
            m=(local['t'][:n]>=lo)&(local['t'][:n]<=hi)
            if m.any():
                delta=local['p'][:n,j][m]-ref['p'][:n,j][m]
                d['differenceRmsPa']=float(np.sqrt(np.mean(delta*delta)))
                both=m & local['wet'][:n,j] & (rows['w'+str(j+1)][:n]>0)
                d['bothWetSamples']=int(both.sum())
                if both.any():d['bothWetDifferenceRmsPa']=float(np.sqrt(np.mean((local['p'][:n,j][both]-ref['p'][:n,j][both])**2)))
                peak=np.flatnonzero(m)[ref['p'][:n,j][m].argmax()]
                d['referencePeakSupport']={'weight':float(rows['w'+str(j+1)][peak]),'count':int(rows['n'+str(j+1)][peak])}
            probes['P'+str(j+1)]=d
        report['windows'][f'{lo}..{hi}']=probes
        fig,axs=plt.subplots(2,2,figsize=(13,8),layout='constrained')
        for j,ax in enumerate(axs.flat):
            for data,color,label in [(ref,'0.35','dynGPUSPH FP64: original hard bounce'),
                (local,'tab:orange','warpSPH FP32: repaired, smooth bounce0.5')]:
                m=(data['t']>=lo)&(data['t']<=hi)
                ax.plot(data['t'][m],data['p'][m,j]/1000,color=color,lw=.8,label=label)
            ax.set(xlim=(lo,hi),xlabel='Time [s]',ylabel='Pressure [kPa]',title='P'+str(j+1));ax.grid(alpha=.2)
            if j==0:ax.legend(fontsize=8)
        fig.suptitle(f'Late pressure {lo}–{hi} s: raw every-step accepted state\nBounce, precision and trim differences remain explicit')
        fig.savefig(args.out/f'pressure-{lo:.2f}-{hi:.2f}.png',dpi=170);plt.close(fig)
    # Descriptive amplitude envelope, without fitting a decay law across
    # repeated impacts. Raw traces remain the primary evidence.
    fig,axs=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    report['envelope']={'trendSigmaSeconds':.005,'rmsSigmaSeconds':.01,'series':{}}
    for name,data,color in [('dynGPUSPH',ref,'0.35'),('warpSPH',local,'tab:orange')]:
        end=min(1.70,float(data['t'][-1]));u=np.arange(.65,end,.0001)
        if len(u)<500:continue
        keep=(u>=1.25)&(u<=end-.03);curves=[]
        for j,ax in enumerate(axs.flat):
            y=np.interp(u,data['t'],data['p'][:,j]);hp=y-gaussian_filter1d(y,50.)
            envelope=np.sqrt(gaussian_filter1d(hp*hp,100.))
            ax.plot(u[keep],envelope[keep]/1000,color=color,label=name)
            ax.set(xlabel='Time [s]',ylabel='Ringing envelope [kPa RMS]',title='P'+str(j+1));ax.grid(alpha=.2)
            # Retain a1ms envelope series for reproducible decay comparisons.
            curves.append(envelope[keep][::10].tolist())
        report['envelope']['series'][name]={'t':u[keep][::10].tolist(),'pressureRmsPa':curves}
    axs[0,0].legend();fig.suptitle('Amplitude envelope:5ms trend removed,10ms RMS smoothing; no decay-law fit')
    fig.savefig(args.out/'ringing-envelope.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(3,1,figsize=(12,9),layout='constrained')
    result=json.loads((args.reference/'result.json').read_text())
    axs[0].plot(local['t'],local['ke'],color='tab:orange',label='warpSPH FP32')
    axs[1].plot(local['t'],local['vmax'],color='tab:orange',label='warpSPH FP32')
    axs[2].plot(local['t'],local['iterations'],color='tab:orange',label='warpSPH FP32')
    rt=np.array([r['t'] for r in result['steps']])
    axs[1].plot(rt,[r['vmax'] for r in result['steps']],color='0.35',lw=.8,label='dynGPUSPH FP64')
    axs[2].plot(rt,[r['iterations'] for r in result['steps']],color='0.35',lw=.8,label='dynGPUSPH FP64')
    # Native accepted KE/mean pressure are independently stored in snapshots.
    from render import read_snapshot,metrics
    samples=[metrics(read_snapshot(p)) for p in sorted((args.reference/'snapshots').glob('*.bin'))]
    axs[0].plot([s['t'] for s in samples],[s['ke'] for s in samples],color='0.35',label='dynGPUSPH FP64 snapshots')
    for ax,label in zip(axs,['Kinetic energy [J/m]','Maximum speed [m/s]','Pseudo-iterations']):
        ax.set(xlabel='Time [s]',ylabel=label,xlim=(0,1.70));ax.grid(alpha=.2);ax.legend()
    fig.savefig(args.out/'dynamics.png',dpi=170);plt.close(fig)
    for name,data in [('warpSPH',local),('dynGPUSPH',{'t':rt,'iterations':np.array([r['iterations'] for r in result['steps']])})]:
        report[name+'Caps']={f'{lo}..{hi}':int(((data['t']>=lo)&(data['t']<=hi)&(data['iterations']>=200)).sum()) for lo,hi in windows}
    solve=args.local/'solve-timing.jsonl'
    if solve.exists():
        s=[json.loads(line) for line in solve.read_text().splitlines()]
        report['warpSPHGraph']={'steps':len(s),'allCaptured':all(r['captured'] and r['disabled'] is None for r in s)}
    (args.out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print('LATE ANALYSIS',json.dumps({'coverage':report['fullLateCoverage'],'commonEnd':report['commonEnd'],'clockMaxAbs':clock}),flush=True)


if __name__=='__main__':main()
