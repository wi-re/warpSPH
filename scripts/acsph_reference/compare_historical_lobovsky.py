#!/usr/bin/env python3
"""Offline comparison of retained Lobovsky runs before/after the reference port.

No simulator imports, GPU work or new physical run. All pressure/convergence
comparisons require identical accepted-step clocks and physical initial arrays.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from analyze_late_ringing import ringing_metrics
from render import read_snapshot,metrics


COHORT={
    'hard':('warpSPH: earlier hard bounce / mass ghosts',
        '.tmp/overnight-acsph-driven-2026-10-27/acsph/warpsph-f32-corrected'),
    'ramp_mass':('warpSPH: ramp0.5 / mass ghosts / mixed closure',
        '.tmp/acsph-lobovsky-ramp05-2026-10-28/warpsph-f32-ramp05'),
    'ramp_volume':('warpSPH: ramp0.5 / volume ghosts / mixed closure',
        '.tmp/acsph-lobovsky-volume-ghost-2026-10-28/warpsph-f32-volume-ghost'),
    'consistent':('warpSPH: previous solver / consistent closure / Michel shift',
        '.tmp/acsph-lobovsky-consistent-closure-2026-10-28/warpsph-f32-consistent-closure'),
    'current':('warpSPH: current reference solver / reference shift-remap',
        '.tmp/acsph-motion-investigation-2026-10-29/warpsph-f32-partial-8395')}
COLORS={'hard':'#9467bd','ramp_mass':'#56b4e9','ramp_volume':'#cc79a7',
    'consistent':'#009e73','current':'#e69f00','dyngpusph':'#444444'}


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def series(rows):
    out={'t':np.array([r['t'] for r in rows]),'p':np.array([r['pressureProbes'] for r in rows]),
        'wet':np.array([r['pressureProbesWet'] for r in rows],dtype=bool),
        'ke':np.array([r['kineticEnergy'] for r in rows]),
        'vmax':np.array([r['maxVelocity'] for r in rows]),
        'iterations':np.array([r['pseudoIterations'] for r in rows])}
    assert np.all(np.diff(out['t'])>0)
    assert all(np.isfinite(out[k]).all() for k in ('p','ke','vmax','iterations'))
    return out


def mean_time(t,y,lo,hi):
    u=np.r_[lo,t[(t>lo)&(t<hi)],hi]
    return float(np.trapezoid(np.interp(u,t,y),u)/(hi-lo))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[2])
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();repo=args.repo.resolve();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    tick=time.monotonic();data={};report={'scope':'Offline retained-trajectory comparison; no new simulation',
        'cohort':{},'windows':{},'physicalICBitwise':{},'clockMaxAbs':{}}
    current_path=repo/COHORT['current'][1]
    current_rows=[json.loads(s) for s in (current_path/'progress.jsonl').read_text().splitlines()]
    current_initial=read_snapshot(current_path/'snapshots/000000.bin')
    n=len(current_rows);end=current_rows[-1]['t'];assert n==8396
    report.update(commonAcceptedSteps=n-1,commonEnd=end)
    for key,(label,relative) in COHORT.items():
        path=repo/relative;rows=[json.loads(s) for s in (path/'progress.jsonl').read_text().splitlines()]
        assert len(rows)>=n
        d=data[key]=series(rows[:n]);clock=float(abs(d['t']-data['hard']['t']).max())
        assert clock==0.;report['clockMaxAbs'][key]=clock
        initial=read_snapshot(path/'snapshots/000000.bin')
        equality={k:np.array_equal(initial[k],current_initial[k]) for k in ('x','v','p','rho','volume','bc')}
        assert all(equality.values()),(key,equality)
        report['physicalICBitwise'][key]=equality
        timing=[json.loads(s) for s in (path/'solve-timing.jsonl').read_text().splitlines()][:n-1]
        assert [r['step'] for r in timing]==list(range(1,n))
        result=json.loads((path/'result.json').read_text()) if (path/'result.json').exists() else None
        audit_path=(repo/'.tmp/acsph-late-ringing-2026-10-29/warpsph-f32-repaired/initial-audit.json'
            if key=='current' else path/'initial-audit.json')
        audit=json.loads(audit_path.read_text())
        original_initial=json.loads((repo/'.tmp/acsph-lobovsky-consistent-closure-2026-10-28/warpsph-f32-consistent-closure/initial-audit.json').read_text())
        for field in ('dx','support','rho0','nu'):
            assert audit['parameters'][field]==original_initial['parameters'][field]
        complete=bool(result and result['nSteps']==10000 and not result['diverged'] and result['stop_reason'] is None)
        if key!='current':assert complete
        report['cohort'][key]={'label':label,'path':str(path),'precision':audit['dtype'],
            'fullAcceptedSteps':len(rows)-1,'fullEnd':rows[-1]['t'],'completed10000Steps':complete,
            'wallSecondsFull':result['wall_seconds'] if result else None,
            'commonSolveMeanMs':float(np.mean([r['solve_wall_ms'] for r in timing])),
            'commonIterationsMean':float(d['iterations'][1:].mean()),
            'commonIterationsMedian':float(np.median(d['iterations'][1:])),
            'commonCaps':int((d['iterations'][1:]>=200).sum()),
            'allGraphsCaptured':all(r['captured'] and r['disabled'] is None for r in timing),
            'endpointKE':float(d['ke'][-1]),'endpointVmax':float(d['vmax'][-1]),
            'maximumSpeedOverCommonRun':float(d['vmax'].max()),
            'progressSHA256':sha(path/'progress.jsonl'),'initialSnapshotSHA256':sha(path/'snapshots/000000.bin'),
            'initialAuditSHA256':sha(audit_path),'initialParameters':audit['parameters'],
            'ghostGeometry':audit.get('ghost_geometry',{'mode':'legacy-mass',
                'note':'Historical fixture default; this older audit has no ghost-geometry receipt'}),
            'startupHistory':audit['startup_history']}
        if key=='current':
            audit_caps={}
            for lo in (0.,1.25,1.44):
                hi=1.53 if lo==1.44 else end
                selected=[r for r in timing if r['iterations']>=200 and lo<=d['t'][r['step']]<=hi]
                audit_caps[f'{lo}..{hi}']={
                    'count':len(selected),
                    'checkedVelocityFails':sum(r['checkedV']>=-8 for r in selected),
                    'checkedPressureFails':sum(r['checkedP']>=-7 for r in selected),
                    'pressureOnlyFailure':sum(r['checkedV']<-8 and r['checkedP']>=-7 for r in selected),
                    'trimmed':sum(r['trimmed'] for r in selected)}
            report['currentCapResiduals']=audit_caps
        print('LOADED',key,len(rows)-1,'accepted steps',flush=True)
    native=repo/'.tmp/acsph-late-ringing-2026-10-29/dyngpusph-f64-dense'
    probes=np.genfromtxt(native/'accepted-probes.csv',delimiter=',',names=True)
    result=json.loads((native/'result.json').read_text());steps=result['steps']
    d=data['dyngpusph']={'t':probes['t'][:n],
        'p':np.column_stack([probes['p'+str(j)] for j in range(1,5)])[:n],
        'wet':np.column_stack([probes['w'+str(j)]>0 for j in range(1,5)])[:n],
        'vmax':np.r_[0.,[s['vmax'] for s in steps]][:n],
        'iterations':np.r_[0.,[s['iterations'] for s in steps]][:n]}
    assert np.array_equal(d['t'],data['current']['t']) and np.isfinite(d['p']).all()
    samples=[]
    for p in sorted((native/'snapshots').glob('*.bin')):
        row=metrics(read_snapshot(p))
        if samples and row['t']==samples[-1]['t']:continue
        samples.append(row)
    nt=np.array([s['t'] for s in samples]);nk=np.array([s['ke'] for s in samples])
    report['cohort']['dyngpusph']={'label':'dynGPUSPH FP64: original hard lagged bounce / source trim',
        'path':str(native),'fullAcceptedSteps':len(steps),'fullEnd':steps[-1]['t'],
        'commonCaps':int((d['iterations'][1:]>=200).sum()),
        'commonIterationsMean':float(d['iterations'][1:].mean()),
        'endpointVmax':float(d['vmax'][-1]),'maximumSpeedOverCommonRun':float(d['vmax'].max()),
        'denseProbeSHA256':sha(native/'accepted-probes.csv'),
        'kineticEnergyCadence':'Native post-shift snapshots, approximately10ms; no every-step KE claim'}
    report['clockMaxAbs']['dyngpusph']=0.
    windows=[(.4,.46),(.65,.95),(.95,1.05),(1.25,end),(1.44,1.53),(1.59,end)]
    for lo,hi in windows:
        w={};report['windows'][f'{lo}..{hi}']=w
        for key,d in data.items():
            m=(d['t']>=lo)&(d['t']<=hi);delta=d['p'][m]-data['dyngpusph']['p'][m]
            w[key]={'probes':{f'P{j+1}':ringing_metrics(d['t'],d['p'][:,j],lo,hi) for j in range(4)},
                'caps':int((d['iterations'][m]>=200).sum()),
                'meanIterations':float(d['iterations'][m].mean()),
                'pressureDifferenceRmsNativePa':np.sqrt(np.mean(delta*delta,axis=0)).tolist(),
                'meanMaxSpeed':mean_time(d['t'],d['vmax'],lo,hi),
                'maxSpeed':float(d['vmax'][m].max())}
            both=d['wet'][m]&data['dyngpusph']['wet'][m]
            w[key]['bothWetDifferenceRmsNativePa']=[float(np.sqrt(np.mean(delta[both[:,j],j]**2)))
                if both[:,j].any() else None for j in range(4)]
        # For bulk KE, sample each local run at the same native snapshot clock.
        u=nt[(nt>=lo)&(nt<=hi)];assert len(u)>1
        native_mean=mean_time(nt,nk,u[0],u[-1]);w['nativeSnapshotKEWindow']=[float(u[0]),float(u[-1])]
        w['nativeSnapshotMeanKE']=native_mean
        for key in COHORT:
            d=data[key];idx=np.searchsorted(d['t'],u);assert np.array_equal(d['t'][idx],u)
            value=mean_time(u,d['ke'][idx],u[0],u[-1])
            w[key]['sharedFrameMeanKE']=value;w[key]['sharedFrameKERatioNative']=value/native_mean
    labels={k:v['label'] for k,v in report['cohort'].items()}
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none'})
    def save(fig,name):
        for ext in ('png','svg'):fig.savefig(out/(name+'.'+ext),dpi=170)
        plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(14,8.5),layout='constrained')
    for row,j in enumerate((0,3)):
        for col,(lo,hi) in enumerate(((1.25,end),(1.44,1.53))):
            ax=axs[row,col]
            for key in ('dyngpusph','consistent','current'):
                d=data[key];m=(d['t']>=lo)&(d['t']<=hi)
                ax.plot(d['t'][m],d['p'][m,j]/1000,color=COLORS[key],lw=1.,label=labels[key])
            ax.set(xlabel='Time [s]',ylabel='Pressure [kPa]',xlim=(lo,hi),title=f'P{j+1}: '+('common late interval' if col==0 else 'completed ringing window'))
            ax.grid(alpha=.2)
    axs[0,0].legend(fontsize=7.5)
    fig.suptitle('Lobovsky: previous versus current warpSPH and dense dynGPUSPH\nRaw accepted post-shift pressures; identical physical IC arrays and timestep clocks')
    save(fig,'previous-current-pressure')
    fig,axs=plt.subplots(2,2,figsize=(14,8.5),layout='constrained')
    for j,ax in enumerate(axs.flat):
        for key,d in data.items():
            m=(d['t']>=1.44)&(d['t']<=1.53)
            ax.plot(d['t'][m],d['p'][m,j]/1000,color=COLORS[key],label=labels[key],
                lw=1.1 if key in ('current','dyngpusph') else .8)
        ax.set(xlabel='Time [s]',ylabel='Pressure [kPa]',xlim=(1.44,1.53),title=f'P{j+1}')
        ax.grid(alpha=.2)
    axs[0,0].legend(fontsize=6.5)
    fig.suptitle('Lobovsky historical cohort: all retained accepted-step pressure traces\nFP32 warpSPH variants; FP64 dynGPUSPH; no filtering or clipping')
    save(fig,'historical-pressure')
    fig,axs=plt.subplots(3,1,figsize=(13,10),layout='constrained')
    for key in ('consistent','current','dyngpusph'):
        d=data[key]
        kt,ke=(nt,nk) if key=='dyngpusph' else (d['t'],d['ke'])
        axs[0].plot(kt,ke,color=COLORS[key],label=labels[key],lw=1.)
        axs[1].plot(d['t'],d['vmax'],color=COLORS[key],lw=.8)
        axs[2].plot(d['t'][1:],d['iterations'][1:],color=COLORS[key],lw=.5,alpha=.6)
    for ax,ylabel in zip(axs,('Kinetic energy [J/m]','Maximum speed [m/s]','Pseudo-iterations')):
        ax.set(xlim=(0,end),xlabel='Time [s]',ylabel=ylabel);ax.grid(alpha=.2)
    axs[0].legend(fontsize=8)
    fig.suptitle('Lobovsky dynamics and convergence on the common accepted interval\nNative KE uses snapshots; local KE, all speeds and iterations use every accepted step')
    save(fig,'previous-current-dynamics')
    report.update(wallSeconds=time.monotonic()-tick,analysisScriptSHA256=sha(Path(__file__)),
        filters='Ringing metrics:0.1ms uniform interpolation; Gaussian trends sigma1/5ms; exclude3sigma edges. Raw extrema unfiltered.',
        limitations=['Comparison changes multiple solver/transport terms; does not isolate a causal operator.',
            'Capped iterations use different stopping/plateau logic across solver implementations.',
            'Solve costs are recorded wall times on a shared machine, not an isolated benchmark.',
            'Earlier implementations did not have the new geometric guard; completion is distinct from physical accuracy.',
            'All amplitudes are pressure at virtual SPH probes, not experimental accuracy norms.'])
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print('HISTORICAL COMPARISON COMPLETE',report['wallSeconds'],'seconds',flush=True)


if __name__=='__main__':main()
