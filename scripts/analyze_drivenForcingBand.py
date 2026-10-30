#!/usr/bin/env python3
"""Compare one retained CRKSPH128 forcing-band control; no solver execution."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from audit_drivenForcing import history_blocks, independent
from summarize_drivenCohort import arrays, digest, mean


def force_statistics(state, lo=10., hi=40.):
    interval=state['parameters']['updateInterval']
    knots=np.arange(int(round(lo/interval)), int(round(hi/interval))+1)*interval
    coeff=np.stack([independent(state,float(t))[1] for t in knots])
    modes,_=independent(state,lo)
    # Exact quadrature of squared linearly interpolated coefficients.
    integrated=.5*interval/3*np.sum(coeff[:-1]**2+coeff[:-1]*coeff[1:]+coeff[1:]**2,axis=(0,1,3))
    mode_mean=integrated/(hi-lo)
    n2=np.sum(modes*modes,axis=1).astype(int)
    return dict(rmsSquaredMean=float(mode_mean.sum()), rms=float(np.sqrt(mode_mean.sum())),
        boxModeForceVarianceFraction=float(mode_mean[n2==1].sum()/mode_mean.sum()),
        kSquared=sorted(set(n2.tolist())),
        varianceByKSquared=[float(mode_mean[n2==k].sum()) for k in sorted(set(n2.tolist()))],
        modes=modes.tolist(), modeVariance=mode_mean.tolist(),
        quadrature='Exact integral of squared piecewise-linear coefficients over fixed-clock intervals')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True,help='Original turbulence campaign directory')
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    runs=[args.baseline/'runs/weak128-CRKSPH-s1234',args.campaign/'runs/weak128-CRKSPH-k2to4-s1234']
    inputs=[json.loads((r/'input.json').read_text()) for r in runs]
    normalized=json.loads(json.dumps(inputs[1]));normalized['params'].update(kMin=1,kMax=2)
    normalized['simulation']['exportRoot']=inputs[0]['simulation']['exportRoot']
    assert normalized==inputs[0], 'Unexpected configuration change'
    rows=[];series=[];states=[];reports=[];raw=[];evidence={}
    initial=[arrays(r/'initial.npz') for r in runs]
    assert initial[0].keys()==initial[1].keys()
    assert all(np.array_equal(initial[0][key],initial[1][key]) for key in initial[0]), 'Initial fluid state differs'
    for run,base in zip(runs,[args.baseline,args.campaign]):
        report=json.loads((base/'analysis'/run.name/'summary.json').read_text())
        result=json.loads((run/'result.json').read_text())
        resolved=json.loads((run/'resolved.json').read_text())
        provenance=json.loads((run/'provenance.json').read_text())
        data=arrays(base/'analysis'/run.name/'particle-statistics.npz')
        grid=arrays(base/'analysis'/run.name/'grid-32.npz')
        assert result['status']=='completed' and result['actualTime']==40
        assert result['stageChecks']['bad']==0 and result['stageChecks']['count']==2*result['nSteps']
        assert result['sourcesStable'] and result['acceptedPositiveFinite']
        assert resolved['spec']['cflFactor']==.15 and resolved['integrator']['identifier']=='rungeKutta2'
        assert report['grid']['32']['minFilterRadius']==report['grid']['32']['maxFilterRadius']==.0625
        assert report['grid']['32']['maxEmptyCellFraction']==0
        history=[json.loads(line) for line in (run/'history.jsonl').open() if line.strip()]
        t=np.array([r['t'] for r in history]);assert len(t)==result['nSteps']+1
        assert np.all(np.diff(t)>0) and t[0]==0 and t[-1]==40
        state=torch.load(run/'forcing.pt',map_location='cpu',weights_only=True);states.append(state)
        row=dict(name=run.name,band=[report['params']['kMin'],report['params']['kMax']],
            nx=128,steps=result['nSteps'],rhs=result['stageChecks']['count'],
            wallSeconds=result['loopWallSeconds'],msPerStep=1000*result['loopWallSeconds']/result['nSteps'],
            meanMach=report['means']['rmsMachMass'],meanK=report['means']['kineticEnergy'],
            meanLogDensityStd=report['means']['logDensityStdMass'],meanU=report['means']['internalEnergyMeanMass'],
            denseMeanMach=mean(t,[h['rmsMachMass'] for h in history]),denseMeanK=mean(t,[h['kineticEnergy'] for h in history]),
            boxModeFraction=report['grid']['32']['boxModeFractionMean'],
            effectiveFieldSamples=report['grid']['32']['fieldCorrelation']['effectiveSamples'],
            stationary=report['stationaryScalarScreenPassed'],maxEnergyBudgetResidual=report['budget']['maxRelativeEnergyResidual'],
            minRawU=result['stageChecks']['minEnergy'],minRawP=result['stageChecks']['minPressure'],
            peakAllocatedMiB=result['peakTorchAllocatedBytes']/2**20,sourceDigest=provenance['sourceSHA256'],
            forcing=force_statistics(state),**history_blocks(run))
        for block in row['blocks']:
            a,b=block['start'],block['end']
            block['boxModeFraction']=mean(grid['t'],grid['boxModeFraction'],a,b)
            block['meanMach']=mean(t,[h['rmsMachMass'] for h in history],a,b)
            block['forcing']=force_statistics(state,a,b)
        rows.append(row);series.append((data,grid));reports.append(report);raw.append(history)
        evidence[run.name]={name:digest(run/name) for name in ('input.json','resolved.json','result.json','provenance.json','forcing.pt','history.jsonl')}
    assert rows[0]['sourceDigest']==rows[1]['sourceDigest']
    p0,p1=[dict(s['parameters']) for s in states]
    p1.update(kMin=1,kMax=2);assert p0==p1
    manifest=json.loads((args.campaign/'manifest.json').read_text())
    changed=[name for name,sha in manifest['sourceHashes'].items() if digest(Path(name))!=sha]
    assert not changed, 'Pinned executable sources changed'
    gpu=[json.loads(line) for line in (args.campaign/'gpu-progress.jsonl').open() if line.strip()]
    foreign=[r for r in gpu if r['queryReturncode']!=0 or (r['phase'].startswith('weak') and any(
        line.split(',')[0].strip()!=str(r['childPID']) for line in r['gpuUsers']))]
    data=dict(rows=rows,evidence=evidence,sourceFilesChecked=len(manifest['sourceHashes']),changedPinnedFiles=changed,
        initialStatesBitwiseIdentical=True,onlyPhysicalChanges={'kMin':[1,2],'kMax':[2,4]},
        fieldFilterRadius=.0625,fieldGrid=32,window=[10,40],
        gpuSamples=len(gpu),foreignGPUSamples=foreign,driverSHA256=digest(Path(__file__)),
        limitations='One realization per band; mode count/draw mapping differ. Changing forcing length also changes turnover-time ratios at fixed acceleration and OU clock. No flux measurement or stationarity guarantee. Field spectrum is volume velocity variance, not full mass kinetic energy.')
    (out/'comparison.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    (out/'individual-reports.json').write_text(json.dumps(reports,indent=2,allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for row,(s,g),state in zip(rows,series,states):
        label=f"Modes {row['band'][0]}–{row['band'][1]}"
        axes[0,0].plot(s['t'],s['kineticEnergy'],label=label,lw=1)
        axes[0,1].plot(g['t'],g['boxModeFraction'],label=label,lw=1)
        ft=np.linspace(0,40,1601)
        fr=np.array([np.sqrt(.5*np.sum(independent(state,float(t))[1]**2)) for t in ft])
        axes[1,0].plot(ft,fr,label=label,lw=.8)
        axes[1,1].plot([15,25,35],[b['forceWork'] for b in row['blocks']],'o-',label=label)
    for ax in axes.flat:ax.grid(alpha=.2);ax.set_xlabel('Physical time')
    axes[0,0].set_ylabel('Particle kinetic energy');axes[0,0].legend()
    axes[0,1].set_ylabel('Exact |n|=1 velocity-energy fraction\n32² field, filter radius 0.0625')
    axes[1,0].set_ylabel('Continuum acceleration RMS')
    axes[1,1].set_ylabel('Block mean mechanical input power / mass')
    fig.suptitle('CRKSPH128² forcing-band control: one seed, unchanged remaining physics')
    fig.savefig(out/'evolution.png',dpi=160);fig.savefig(out/'evolution.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for row,report in zip(rows,reports):
        label=f"Modes {row['band'][0]}–{row['band'][1]}"
        axes[0].plot(row['forcing']['kSquared'],row['forcing']['varianceByKSquared'],'o-',label=label)
        e=np.array(report['grid']['32']['means']['velocityEnergy'])
        axes[1].loglog(np.arange(1,len(e)),e[1:],'o-',label=label)
    axes[0].set_xlabel('Exact squared integer wave number |n|²')
    axes[0].set_ylabel('Mean acceleration variance in mode group')
    axes[1].set_xlabel('Rounded radial velocity-spectrum shell')
    axes[1].set_ylabel('Mean volume velocity energy (shell sum)')
    for ax in axes:ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Force variance and filtered velocity spectrum, t=10–40; no flux measurement')
    fig.savefig(out/'spectra.png',dpi=160);fig.savefig(out/'spectra.pdf');plt.close(fig)
    endpoints=[arrays(run/'final.npz') for run in runs]
    density_limits=(min(float(s['densities'].min()) for s in endpoints),max(float(s['densities'].max()) for s in endpoints))
    speeds=[np.linalg.norm(s['velocities'].astype(float),axis=1) for s in endpoints]
    speed_limits=(0.,max(float(v.max()) for v in speeds))
    fig,axes=plt.subplots(2,2,figsize=(10,9),layout='constrained',sharex=True,sharey=True)
    for i,(row,state,speed) in enumerate(zip(rows,endpoints,speeds)):
        x=np.remainder(state['positions'].astype(float)+.5,1.)-.5
        for j,(value,limits,title) in enumerate(((state['densities'],density_limits,'Density'),(speed,speed_limits,'Speed'))):
            artist=axes[i,j].scatter(x[:,0],x[:,1],c=value,s=1.4,cmap='viridis',vmin=limits[0],vmax=limits[1],linewidths=0,rasterized=True)
            axes[i,j].set(xlim=(-.5,.5),ylim=(-.5,.5),aspect='equal',xlabel='x',ylabel='y',title=f"Modes {row['band'][0]}–{row['band'][1]}: {title}")
            fig.colorbar(artist,ax=axes[i,j],shrink=.8)
    fig.suptitle('CRKSPH128² at t=40: matched color scales, wrapped particle positions')
    fig.savefig(out/'endpoints.png',dpi=180);fig.savefig(out/'endpoints.pdf');plt.close(fig)
    for row in rows:print(json.dumps(row),flush=True)


if __name__=='__main__':main()
