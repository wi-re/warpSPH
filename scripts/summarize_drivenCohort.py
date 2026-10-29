#!/usr/bin/env python3
"""Audit and compare an existing overnight driven cohort; never run a solver.

Uses each run's own report, since repeated single-run analyzer invocations
replace the top-level summary. Original outputs and frozen sources stay intact.
"""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

SCHEMES = ('MFM','MFV','CRKSPH','CompSPH','Monaghan','GSPH','InutsukaGSPH')


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean(t, value, lo=10., hi=40.):
    t, value = np.asarray(t), np.asarray(value)
    clock = np.r_[lo,t[(t>lo)&(t<hi)],hi]
    return float(np.trapezoid(np.interp(clock,t,value),clock)/(hi-lo))


def difference(a,b):
    return float(2*(b-a)/(abs(a)+abs(b))) if abs(a)+abs(b)>0 else 0.


def arrays(path):
    with np.load(path) as data:
        return dict(data)


def forcing_audit(root, names):
    import torch
    states=[torch.load(root/'runs'/name/'forcing.pt',map_location='cpu',weights_only=True) for name in names]
    parameters=states[0]['parameters']
    count=min(len(state['knots']) for state in states)
    prefix_equal=all(torch.equal(states[0]['knots'][:count],state['knots'][:count]) for state in states)
    parameters_equal=all(state['parameters']==parameters for state in states)
    required=int(np.floor(40/parameters['updateInterval']))+2
    finite=all(bool(torch.isfinite(state['knots']).all()) for state in states)
    assert prefix_equal and parameters_equal and finite and count>=required
    return dict(commonKnotPrefixBitwiseEqual=prefix_equal,parametersIdentical=parameters_equal,
                commonPrefixKnots=count,requiredThroughEndpoint=required,allFinite=finite,
                knotsPerRun={name:len(state['knots']) for name,state in zip(names,states)},
                scope='Saved final OU checkpoint contains full past knots; no per-frame native-output reconstruction performed')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--baseline',type=Path)
    args=parser.parse_args()
    campaign=args.campaign.resolve();root=campaign/'turbulence';out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    manifest=read(campaign/'manifest.json')
    changes=[name for name,sha in manifest['sourceHashes'].items()
             if not Path(name).exists() or digest(Path(name))!=sha]
    audit=dict(sourceFilesChecked=len(manifest['sourceHashes']),changedPinnedFiles=changes,
               campaign=str(campaign),manifestSHA256=digest(campaign/'manifest.json'))
    original_tests=[]
    for file in sorted(campaign.glob('gpu-source-*.xml')):
        suites=list(ET.parse(file).getroot().iter('testsuite'))
        original_tests.append(dict(file=str(file),tests=sum(int(s.get('tests',0)) for s in suites),
            failures=sum(int(s.get('failures',0)) for s in suites),errors=sum(int(s.get('errors',0)) for s in suites)))
    receipt=read(campaign/'postflight-export-device/receipt.json')
    old=campaign/'turbulence/source/warpSPH/tests/test_drivenSchemes.py'
    new=campaign/'postflight-export-device/test_drivenSchemes.py'
    expected=old.read_text().replace('assert_close(q, result.state.state.Q, rtol=0, atol=0)',
                                   'assert_close(q, result.state.state.Q.cpu(), rtol=0, atol=0)')
    audit['sourceControls']=dict(original=original_tests,postflight=receipt,
        correctionExactlyDeviceTransfer=expected==new.read_text(),
        originalHashMatches=digest(old)==receipt['originalSHA256'],
        correctedHashMatches=digest(new)==receipt['correctedSHA256'],
        postflightSuites=[s.attrib for s in ET.parse(campaign/'postflight-export-device/results.xml').getroot().iter('testsuite')])
    reports={};rows=[];series={};raw_evidence={}
    for nx in (64,128):
        for scheme in SCHEMES:
            name=f'weak{nx}-{scheme}-s1234';path=root/'runs'/name
            report=read(root/'analysis'/name/'summary.json');result=read(path/'result.json')
            provenance=read(path/'provenance.json');resolved=read(path/'resolved.json')
            initial,final=arrays(path/'initial.npz'),arrays(path/'final.npz')
            data=arrays(root/'analysis'/name/'particle-statistics.npz');series[name]=data
            history=[json.loads(line) for line in (path/'history.jsonl').open() if line.strip()]
            times=np.array([r['t'] for r in history]);e0=float(initial['Q'][:,-1].astype(float).sum())
            residual=np.array([(r['totalEnergy']-e0-r['forceWork']-r['thermalExchange'])/e0 for r in history])
            assert len(history)==result['nSteps']+1 and np.all(np.diff(times)>0)
            assert times[0]==0 and times[-1]==40 and result['actualTime']==40
            assert result['stageChecks']['count']==2*result['nSteps'] and result['stageChecks']['bad']==0
            assert result['sourcesStable'] and result['acceptedPositiveFinite'] and result['status']=='completed'
            assert np.isclose(abs(residual).max(),report['budget']['maxRelativeEnergyResidual'],rtol=1e-5,atol=1e-14)
            raw_mean_k=mean(times,[r['kineticEnergy'] for r in history])
            raw_mean_mach=mean(times,[r['rmsMachMass'] for r in history])
            scale=float(initial['masses'].mean())
            gamma=report['params']['gamma'];m=final['masses'].astype(float);v=final['velocities'].astype(float)
            bulk=np.average(v,weights=m,axis=0)
            local_mach=np.linalg.norm(v-bulk,axis=1)/np.sqrt(gamma*(gamma-1)*final['internalEnergies'])
            g16,g32=report['grid']['16'],report['grid']['32']
            assert g16['minFilterRadius']==g16['maxFilterRadius']==.125
            row=dict(name=name,scheme=scheme,nx=nx,steps=result['nSteps'],wallSeconds=result['loopWallSeconds'],
                msPerStep=1000*result['loopWallSeconds']/result['nSteps'],wallSecondsPerPhysicalTime=result['loopWallSeconds']/40,
                meanMach=report['means']['rmsMachMass'],meanK=report['means']['kineticEnergy'],
                rawHistoryMeanMach=raw_mean_mach,rawHistoryMeanK=raw_mean_k,
                meanLogDensityStd=report['means']['logDensityStdMass'],meanU=report['means']['internalEnergyMeanMass'],
                energyResidual=float(abs(residual).max()),minRawU=result['stageChecks']['minEnergy'],
                minRawPressure=result['stageChecks']['minPressure'],massChange=report['budget']['maxRelativeMassChange'],
                meanSpeed=report['budget']['maxMeanSpeed'],scalarScreen=report['stationaryScalarScreenPassed'],
                kineticBlockMeans=report['screens']['kineticEnergy']['blockMeans'],
                kineticBlockSpread=report['screens']['kineticEnergy']['relativeSpread'],
                effectiveFieldSamples32=g32['fieldCorrelation']['effectiveSamples'],boxModeFraction32=g32['boxModeFractionMean'],
                lowestShellFraction32=g32['lowestShellFractionMean'],finalLocalMachMax=float(local_mach.max()),
                finalSupersonicMassFraction=float(m[local_mach>1].sum()/m.sum()),
                finalMassRatioRange=[float(m.min()/scale),float(m.max()/scale)],
                peakAllocatedMiB=result['peakTorchAllocatedBytes']/2**20,peakReservedMiB=result['peakTorchReservedBytes']/2**20,
                shockSwitchC=resolved['solverConfig'].get('diffusionParams',{}).get('shockSwitchC'),
                integrator=resolved['integrator'],sourceDigest=provenance['sourceSHA256'])
            rows.append(row);reports[name]=report
            raw_evidence[name]=dict(historyRows=len(history),samples=len(data['t']),
                inputSHA256=digest(path/'input.json'),resultSHA256=digest(path/'result.json'),
                historySHA256=digest(path/'history.jsonl'),reportSHA256=digest(root/'analysis'/name/'summary.json'),
                initialQSHA256=hashlib.sha256(initial['Q'].tobytes()).hexdigest(),
                finalQSHA256=hashlib.sha256(final['Q'].tobytes()).hexdigest())
            print(f'audited {name}: {result["nSteps"]} steps, budget={row["energyResidual"]:.3g}',flush=True)
    audit.update(acceptedSteps=sum(r['steps'] for r in rows),rhsChecks=2*sum(r['steps'] for r in rows),
                 loopWallSeconds=sum(r['wallSeconds'] for r in rows),sourceDigests=sorted({r['sourceDigest'] for r in rows}),
                 individualReports=len(reports),topLevelIndexEntries=list(read(root/'analysis/summary.json')),
                 evidence=raw_evidence,forcing=forcing_audit(root,list(reports)))
    comparisons={}
    for scheme in SCHEMES:
        a,b=[reports[f'weak{nx}-{scheme}-s1234'] for nx in (64,128)]
        comparisons[scheme]=dict(symmetricDifferences={key:difference(a['means'][key],b['means'][key]) for key in
            ('rmsMachMass','kineticEnergy','logDensityStdMass')},
            pdfTotalVariation={key:float(.5*np.abs(np.array(a['meanPDFs'][key])-b['meanPDFs'][key]).sum()) for key in a['meanPDFs']})
        for grid in ('16','32'):
            x,y=[np.array(r['grid'][grid]['means']['velocityEnergy'])[1:int(grid)//4+1] for r in (a,b)]
            comparisons[scheme]['spectralSymmetricL1Grid'+grid]=float(2*abs(x-y).sum()/(abs(x).sum()+abs(y).sum()))
    timings={}
    for scheme in ('MFM','MFV'):
        by_skin={skin:[read(root/f'timing/timing-{scheme}-r{r}-skin{skin}/result.json') for r in range(3)] for skin in ('1.0','1.25')}
        means={k:float(np.median([v['meanStepMS'] for v in vals])) for k,vals in by_skin.items()}
        finals=[]
        for r in range(3):
            a,b=[arrays(root/f'timing/timing-{scheme}-r{r}-skin{k}/final.npz') for k in ('1.0','1.25')]
            common=[k for k in ('positions','Q','supports','masses','densities','velocities','pressures','internalEnergies','sourceEnergyBudget') if k in a and k in b]
            finals.append({k:dict(bitwise=bool(np.array_equal(a[k],b[k])),relativeL2=float(np.linalg.norm(a[k].astype(float)-b[k].astype(float))/max(np.linalg.norm(a[k].astype(float)),1e-300))) for k in common})
        timings[scheme]=dict(medianWindowMeanMS=means,wallReduction=1-means['1.25']/means['1.0'],
            individualWindowMeans={k:[v['meanStepMS'] for v in vals] for k,vals in by_skin.items()},finalFieldDifferences=finals,
            snapshotHashes=sorted({v['snapshotSHA256'] for vals in by_skin.values() for v in vals}),
            allCompleted=all(not v['diverged'] and not v['stopReason'] for vals in by_skin.values() for v in vals))
    repeats={}
    if args.baseline:
        for nx in (64,128):
            old=args.baseline/f'weak{nx}-s1234/final.npz'
            if old.exists():
                a,b=arrays(old),arrays(root/f'runs/weak{nx}-MFM-s1234/final.npz')
                repeats[str(nx)]={key:bool(np.array_equal(a[key],b[key])) for key in a if key in b}
    report=dict(audit=audit,rows=rows,resolution=comparisons,timing=timings,mfmBaselineFinalRepeat=repeats)
    (out/'cohort.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (out/'individual-reports.json').write_text(json.dumps(reports,indent=2,allow_nan=False)+'\n')
    figures(out,rows,reports,series)
    (out/'analysis-provenance.json').write_text(json.dumps(dict(driverSHA256=digest(Path(__file__)),numpy=np.__version__,
        commandCampaign=str(campaign),baseline=str(args.baseline),scope='CPU offline; no simulation or original output edits'),indent=2)+'\n')


def figures(out,rows,reports,series):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors=dict(zip(SCHEMES,plt.get_cmap('tab10').colors))
    labels={s:{'GSPH':'GSPH (c=0)','InutsukaGSPH':'Inutsuka (c=3)'}.get(s,s) for s in SCHEMES}
    fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True,layout='constrained')
    for j,nx in enumerate((64,128)):
        for scheme in SCHEMES:
            s=series[f'weak{nx}-{scheme}-s1234'];t=s['t']
            axes[0,j].plot(t,s['rmsMachMass'],color=colors[scheme],lw=1,label=labels[scheme])
            axes[1,j].plot(t,s['kineticEnergy'],color=colors[scheme],lw=1)
        axes[0,j].set_title(f'{nx}² particles, seed 1234');axes[1,j].set_xlabel('Physical time')
        for ax in axes[:,j]:
            for boundary in (10,20,30):ax.axvline(boundary,color='.75',lw=.7,ls='--')
            ax.grid(alpha=.2)
    axes[0,0].set_ylabel('Mass RMS Mach');axes[1,0].set_ylabel('Particle kinetic energy')
    axes[0,1].legend(ncol=2,fontsize=8,loc='upper left')
    fig.suptitle('Weak driven compressible flow: evolving statistics, not stationary samples')
    fig.savefig(out/'evolution.png',dpi=160);fig.savefig(out/'evolution.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    x=np.arange(len(SCHEMES));width=.36
    for nx,offset,color in [(64,-width/2,'#3676b5'),(128,width/2,'#db8b34')]:
        cohort=[next(r for r in rows if r['nx']==nx and r['scheme']==s) for s in SCHEMES]
        axes[0].bar(x+offset,[r['meanK'] for r in cohort],width,label=f'{nx}²',color=color)
        axes[1].bar(x+offset,[r['wallSeconds']/60 for r in cohort],width,color=color)
    for ax in axes[:2]:ax.set_xticks(x,[labels[s] for s in SCHEMES],rotation=55,ha='right');ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Mean K, time 10–40');axes[0].legend();axes[1].set_ylabel('Run loop wall time to t=40 (min)')
    axes[1].set_title('Includes observations/video; native timesteps differ',fontsize=9)
    for scheme in SCHEMES:
        for nx,style in [(64,'--'),(128,'-')]:
            e=np.array(reports[f'weak{nx}-{scheme}-s1234']['grid']['16']['means']['velocityEnergy'])
            axes[2].semilogy(np.arange(1,5),e[1:5],style,color=colors[scheme],lw=1.1,label=labels[scheme] if nx==128 else None)
    axes[2].set_xticks([1,2,3,4]);axes[2].set_xlabel('Rounded radial shell k, fixed H=0.125')
    axes[2].legend(ncol=2,fontsize=7,loc='lower left')
    axes[2].set_ylabel('Mean filtered velocity shell energy');axes[2].set_title('16² observation: dashed 64², solid 128²',fontsize=9)
    fig.suptitle('Method/resolution comparison: one realization, no accuracy ranking')
    fig.savefig(out/'comparison.png',dpi=160);fig.savefig(out/'comparison.pdf');plt.close(fig)


if __name__=='__main__':main()
