#!/usr/bin/env python3
"""Compare saved CRKSPH 64/128 and 192/256 driven-flow runs; no simulations."""
import argparse
import json
from pathlib import Path

import numpy as np
from summarize_drivenCohort import arrays, digest, difference, mean


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True,help='Previous campaign turbulence directory')
    p.add_argument('--campaign',type=Path,required=True,help='New resolution campaign directory')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    rows=[];reports={};series={};evidence={};states=[]
    for nx in (64,128,192,256):
        root=args.baseline if nx<=128 else args.campaign
        name=f'weak{nx}-CRKSPH-s1234';run=root/'runs'/name
        report=json.loads((root/'analysis'/name/'summary.json').read_text())
        result=json.loads((run/'result.json').read_text());initial=arrays(run/'initial.npz')
        observations=arrays(root/'analysis'/name/'particle-statistics.npz')
        resolved=json.loads((run/'resolved.json').read_text())
        assert result['status']=='completed' and result['actualTime']==40 and result['sourcesStable']
        assert result['stageChecks']['bad']==0 and result['stageChecks']['count']>=2*result['nSteps']
        assert result['acceptedPositiveFinite'] and resolved['spec']['cflFactor']==.15
        assert resolved['integrator']['identifier']=='rungeKutta2'
        assert report['grid']['16']['minFilterRadius']==report['grid']['16']['maxFilterRadius']==.125
        history=[json.loads(line) for line in (run/'history.jsonl').open() if line.strip()]
        t=np.array([h['t'] for h in history]);assert np.all(np.diff(t)>0) and t[0]==0 and t[-1]==40
        assert len(history)==result['nSteps']+1
        # SPH evolves primitive u; its Q is an FP32 diagnostic reconstruction.
        # A non-power-of-two particle mass can expose an initial Q rounding offset.
        primitive_e0=np.sum(initial['masses'].astype(float)*(initial['internalEnergies'].astype(float)+.5*np.sum(initial['velocities'].astype(float)**2,axis=1)))
        derived_q_e0=float(initial['Q'][:,-1].astype(float).sum())
        e0=float(primitive_e0)
        residual=max(abs((h['totalEnergy']-e0-h['forceWork']-h['thermalExchange'])/e0) for h in history)
        assert np.isclose(residual,report['budget']['maxRelativeEnergyResidual'],rtol=1e-5,atol=1e-14)
        row=dict(nx=nx,initialEnergyReference=e0,initialDerivedQEnergy=derived_q_e0,particles=result['particles'],steps=result['nSteps'],extraRHS=result['stageChecks']['count']-2*result['nSteps'],wallSeconds=result['loopWallSeconds'],
            msPerStep=result['loopWallSeconds']*1000/result['nSteps'],wallSecondsPerPhysicalTime=result['loopWallSeconds']/40,
            meanMach=report['means']['rmsMachMass'],meanK=report['means']['kineticEnergy'],
            denseHistoryMeanMach=mean(t,[h['rmsMachMass'] for h in history]),denseHistoryMeanK=mean(t,[h['kineticEnergy'] for h in history]),
            meanLogDensityStd=report['means']['logDensityStdMass'],meanU=report['means']['internalEnergyMeanMass'],
            kineticBlockMeans=report['screens']['kineticEnergy']['blockMeans'],kineticBlockSpread=report['screens']['kineticEnergy']['relativeSpread'],
            machBlockMeans=report['screens']['rmsMachMass']['blockMeans'],stationarity=report['stationaryScalarScreenPassed'],
            effectiveFieldSamples=report['grid']['32']['fieldCorrelation']['effectiveSamples'],
            boxModeFraction=report['grid']['32']['boxModeFractionMean'],energyBudgetResidual=float(residual),
            minRawU=result['stageChecks']['minEnergy'],minRawPressure=result['stageChecks']['minPressure'],
            maxRMSMach=result['maxMach'],peakAllocatedMiB=result['peakTorchAllocatedBytes']/2**20,
            peakReservedMiB=result['peakTorchReservedBytes']/2**20,
            sourceDigest=json.loads((run/'provenance.json').read_text())['sourceSHA256'])
        rows.append(row);reports[str(nx)]=report;series[nx]=observations
        evidence[str(nx)]={n:digest(run/n) for n in ('input.json','result.json','resolved.json','history.jsonl','forcing.pt')}
        import torch
        states.append(torch.load(run/'forcing.pt',map_location='cpu',weights_only=True))
    common=min(len(s['knots']) for s in states)
    assert common>=802 and all(states[0]['parameters']==s['parameters'] and np.array_equal(states[0]['knots'][:common].numpy(),s['knots'][:common].numpy()) for s in states)
    assert len({r['sourceDigest'] for r in rows})==1
    adjacent={}
    for a,b in zip(rows[:-1],rows[1:]):
        ra,rb=reports[str(a['nx'])],reports[str(b['nx'])]
        x,y=[np.array(r['grid']['16']['means']['velocityEnergy'])[1:5] for r in (ra,rb)]
        adjacent[f'{a["nx"]}-{b["nx"]}']=dict(Mach=difference(a['meanMach'],b['meanMach']),
            K=difference(a['meanK'],b['meanK']),logDensityStd=difference(a['meanLogDensityStd'],b['meanLogDensityStd']),
            matchedFilterSpectrumSymmetricL1=float(2*abs(x-y).sum()/(abs(x).sum()+abs(y).sum())),
            densityPDFTotalVariation=float(.5*abs(np.array(ra['meanPDFs']['logDensityMass'])-rb['meanPDFs']['logDensityMass']).sum()))
    gpu_observations=[json.loads(line) for line in (args.campaign/'gpu-progress.jsonl').open() if line.strip()]
    physical_samples=[r for r in gpu_observations if r['phase'].startswith('weak')]
    contention=[r for r in physical_samples if r['queryReturncode']!=0 or any(
        line.split(',')[0].strip()!=str(r['childPID']) for line in r['gpuUsers'])]
    load_context=dict(physicalSamples=len(physical_samples),otherProcessOrQueryErrorSamples=contention,
        intervalSeconds=30,scope='Sampled compute users, not a guarantee against all transient activity')
    data=dict(rows=rows,adjacent=adjacent,commonForcingKnots=common,commonSourceDigest=rows[0]['sourceDigest'],evidence=evidence,loadContext=load_context,
              scope='Matched one-seed finite-window means, t=10..40. Block ranges are not confidence intervals.',driverSHA256=digest(Path(__file__)))
    (out/'comparison.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    (out/'individual-reports.json').write_text(json.dumps(reports,indent=2,allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    n=np.array([r['nx'] for r in rows]);colors=plt.get_cmap('tab10').colors
    for ax,key,blocks,label in [(axes[0],'meanMach','machBlockMeans','Mass RMS Mach'),(axes[1],'meanK','kineticBlockMeans','Particle kinetic energy')]:
        ax.plot(n,[r[key] for r in rows],'o-',label='Mean t=10–40')
        for j,r in enumerate(rows):ax.scatter(r['nx']+np.array([-3,0,3]),r[blocks],s=20,color='.5',marker='x',label='Three block means' if j==0 else None)
        ax.set_ylabel(label);ax.set_xticks(n);ax.set_xlabel('Particles per side');ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    axes[2].plot(n,[r['wallSeconds']/60 for r in rows],'o-',label='Run loop minutes')
    axes[2].set_ylabel('Wall time to t=40 (min)');axes[2].set_xlabel('Particles per side');axes[2].set_xticks(n);axes[2].grid(alpha=.2)
    ax=axes[2].twinx();ax.plot(n,[r['msPerStep'] for r in rows],'s--',color=colors[1]);ax.set_ylabel('ms/step',color=colors[1])
    fig.suptitle('CRKSPH resolution/cost: one seed, native midpoint/CFL, block values are not confidence intervals')
    fig.savefig(out/'resolution.png',dpi=160);fig.savefig(out/'resolution.pdf');plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True,layout='constrained')
    for nx in n:
        s=series[nx];axes[0].plot(s['t'],s['kineticEnergy'],label=f'{nx}²',lw=1)
        axes[1].plot(s['t'],s['logDensityStdMass'],lw=1)
    for ax in axes:
        for t in (10,20,30):ax.axvline(t,color='.7',ls='--',lw=.8)
        ax.grid(alpha=.2)
    axes[0].legend(ncol=4);axes[0].set_ylabel('Particle kinetic energy')
    axes[1].set_ylabel('Mass log-density std');axes[1].set_xlabel('Physical time')
    fig.suptitle('CRKSPH driven flow: resolution and finite-window variability')
    fig.savefig(out/'evolution.png',dpi=160);fig.savefig(out/'evolution.pdf');plt.close(fig)
    for row in rows:print(json.dumps(row),flush=True)


if __name__=='__main__':main()
