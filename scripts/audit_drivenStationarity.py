#!/usr/bin/env python3
"""Postflight audit and fixed-window plots for the retained three-seed campaign.

CPU-only. Does not import a solver or allocate CUDA. Completes provenance,
source/snapshot integrity and force-variance checks independently of the worker.
"""
from pathlib import Path
import argparse,hashlib,json,itertools
import numpy as np
import torch
from analyze_drivenForcingBand import force_statistics
from audit_drivenForcing import independent
from watch_drivenStationarity import verify_run


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args();root=a.root.resolve()
    physical=json.loads((root/'status.json').read_text());companion=json.loads((root/'analysis-status.json').read_text());plan=json.loads((root/'manifest.json').read_text());analysisPlan=json.loads((root/'analysis-manifest.json').read_text())
    assert physical['phase']=='complete' and companion['status']=='completed'
    assert len(physical['jobs'])==len(companion['runs'])==3
    pins={**plan['sourceHashes'],**analysisPlan['sourceHashes']};changed=[k for k,v in pins.items() if digest(Path(k))!=v];assert not changed
    baseline=Path(analysisPlan['baseline']);states=[];rows=[];snapshotsChecked=0;fullSnapshotCount=0
    for seed in (1234,5678,9012):
        run=root/'runs'/f'long128-CRKSPH-s{seed}';directory=root/'analysis'/run.name
        provenance=verify_run(run,baseline,seed);r=json.loads((run/'result.json').read_text());report=json.loads((directory/'summary.json').read_text());resolved=json.loads((run/'resolved.json').read_text());cfg=json.loads((run/'input.json').read_text())
        assert r['stageChecks']['count']==2*r['nSteps'] and r['extraRHS']==0
        assert resolved['spec']['cflFactor']==.15 and resolved['integrator']['identifier']=='rungeKutta2'
        sampling=json.loads((run/'sampling.json').read_text());paths=sorted((run/'samples').glob('*.npz'))
        assert len(paths)==len(sampling['times'])==12801 and sampling['requestedInterval']==.0125 and sampling['missedTargets']==0
        assert np.all(np.diff(sampling['times'])>0) and sampling['times'][0]==0 and sampling['times'][-1]==160
        fullSnapshotCount+=len(paths)
        evidence=report['evidence']
        for filename,key in [('input.json','inputSHA256'),('history.jsonl','historySHA256'),('result.json','resultSHA256')]:assert digest(run/filename)==evidence[key]
        for filename,h in evidence['sampleSHA256'].items():
            assert digest(run/'samples'/filename)==h;snapshotsChecked+=1
        assert len(evidence['sampleSHA256'])==1601
        with np.load(directory/'native-series.npz') as n:
            assert len(n['t'])==r['nSteps']+1 and n['t'][0]==0 and n['t'][-1]==160 and np.all(np.diff(n['t'])>0)
        state=torch.load(run/'forcing.pt',map_location='cpu',weights_only=True);states.append(state)
        assert state['parameters']['seed']==seed and cfg['params']['seed']==seed
        force=[force_statistics(state,lo,hi) for lo,hi in [(40,80),(80,120),(120,160)]]
        for w,f in zip(report['windows'],force):f.update(window=w['window'],nativeMechanicalInput=w['budget']['forceWorkRate'])
        means=np.array([w['native']['kineticEnergy']['mean'] for w in report['windows']])
        ciMissing=sum('unavailable' in w['native']['kineticEnergy']['bootstrap'] for w in report['windows'])
        row=dict(seed=seed,provenance=provenance,steps=r['nSteps'],rawRHS=r['stageChecks']['count'],badRHS=r['stageChecks']['bad'],minRawU=r['stageChecks']['minEnergy'],minRawP=r['stageChecks']['minPressure'],maxMach=r['maxMach'],maxEnergyBudgetResidual=r['maxAbsoluteEnergyBudgetResidual'],maxMassDrift=report['maxNativeMassDrift'],cost=report['cost'],KBlockMeans=means.tolist(),KLateMean=float(means.mean()),KRelativeBlockSpread=report['screens']['kineticEnergy']['relativeSpread'],boxFractions=[w['field']['boxModeFraction']['mean'] for w in report['windows']],fieldCorrelation=report['fieldCorrelation'],kineticWindowConfidenceUnavailable=ciMissing,forcing=force,distributionTV=report['distributionChangeTotalVariation'],analysisSeconds=report['wallSeconds'],sampleManifestSHA256=digest(run/'sampling.json'),forcingSHA256=digest(run/'forcing.pt'))
        rows.append(row)
    old=torch.load(baseline/'forcing.pt',map_location='cpu',weights_only=True)
    assert torch.equal(states[0]['knots'][:len(old['knots'])],old['knots'])
    pairs=[]
    for i,j in itertools.combinations(range(3),2):
        assert not torch.equal(states[i]['knots'],states[j]['knots'])
        times=np.arange(40,160.0001,.05)
        a=np.stack([independent(states[i],float(t))[1] for t in times]);b=np.stack([independent(states[j],float(t))[1] for t in times])
        pairs.append(dict(seeds=[rows[i]['seed'],rows[j]['seed']],coefficientHistoriesDifferent=True,coefficientCosine=float(np.sum(a*b)/np.sqrt(np.sum(a*a)*np.sum(b*b)))))
    totals=dict(steps=sum(r['steps'] for r in rows),rawRHS=sum(r['rawRHS'] for r in rows),invalidRHS=sum(r['badRHS'] for r in rows),loopWallSeconds=sum(r['cost']['loopWallSeconds'] for r in rows),analysisSeconds=sum(r['analysisSeconds'] for r in rows),snapshotsRetained=fullSnapshotCount,analyzedSnapshotsHashed=snapshotsChecked,pinnedFilesChecked=len(pins))
    data=dict(status='passed',rows=rows,totals=totals,sourceChanges=changed,forcingPairs=pairs,forcingPrefixMatchesOldBaseline=True,visualReview=dict(ensemblePlot=True,representativeFramesPerRun=5,allThreeContactSheets=True,observation='Box-scale coherent structures reorient and fluctuate; no blow-up or boundary-ejection pattern in inspected frames. Representative inspection, not every encoded frame.'),limits='Fixed CRK128/one model. Independent coefficient seeds and three finite trajectories do not certify stationarity, ensemble precision or physical transport.',driverSHA256=digest(Path(__file__)))
    (root/'postflight-review.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(11,10),layout='constrained');colors=plt.get_cmap('tab10').colors
    for i,seed in enumerate((1234,5678,9012)):
        report=json.loads((root/'analysis'/f'long128-CRKSPH-s{seed}'/'summary.json').read_text())
        with np.load(root/'analysis'/f'long128-CRKSPH-s{seed}'/'statistics.npz') as z:edges=z['pdfEdges']
        centers=(edges[1:]+edges[:-1])/2
        for j,w in enumerate(report['windows']):
            pdf=np.asarray(w['distributions']['logDensityMass']);spectrum=np.asarray(w['spectra']['velocityEnergy']);label=f"t{w['window'][0]:g}–{w['window'][1]:g}"
            axes[i,0].plot(centers,pdf[1:-1]/np.diff(edges),color=colors[j],label=label)
            k=np.arange(1,9);axes[i,1].semilogy(k,spectrum[k],color=colors[j],marker='.',label=label)
        axes[i,0].set(xlim=(-1,1),xlabel='ln(rho / box mean)',ylabel='Mass-weighted PDF',title=f'Seed{seed}: density distribution')
        axes[i,1].set(xlabel='Rounded radial shell k',ylabel='Mean velocity shell energy',title=f'Seed{seed}: fixed observation radius0.0625')
        for ax in axes[i]:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Predeclared late windows: PDFs and spectra; grid32, comparison k≤8\nSpectra are volume velocity variance, not mass kinetic-energy flux; PDF overflow retained in JSON')
    for suffix in ('png','pdf'):fig.savefig(root/'analysis'/f'distributions-spectra.{suffix}',dpi=160)
    print(json.dumps(dict(totals=totals,forceRMSByWindow={r['seed']:[f['rms'] for f in r['forcing']] for r in rows},pairs=pairs),indent=2))


if __name__=='__main__':main()
