#!/usr/bin/env python3
"""CPU-only, correlation-aware assessment of retained driven-flow trajectories.

A failed block screen is not proof of nonstationarity. Correlation/bootstrap
estimates assume stationary fluctuations and are explicitly conditional when
applied to a candidate window; drift is retained, never removed by detrending.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,time
import numpy as np


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def window_clock(t,lo,hi):
    t=np.asarray(t,dtype=float)
    if not t[0]<=lo<hi<=t[-1] or np.any(np.diff(t)<=0):raise ValueError('strict times must cover complete assessment window')
    return np.r_[lo,t[(t>lo)&(t<hi)],hi]


def mean(t,v,lo,hi):
    clock=window_clock(t,lo,hi)
    v=np.asarray(v);flat=v.reshape(len(t),-1)
    values=np.stack([np.interp(clock,t,col) for col in flat.T],axis=-1)
    return (np.trapezoid(values,clock,axis=0)/(hi-lo)).reshape(v.shape[1:])


def power_rate(t,v,lo,hi):return float((np.interp(hi,t,v)-np.interp(lo,t,v))/(hi-lo))


def block_bootstrap(values,block_length,*,seed=1729,repeats=1000):
    """Circular moving-block mean bootstrap; caller supplies correlation block.

    No IID treatment of correlated samples. Quantiles have a stationarity
    assumption and are not a trend test or coverage guarantee for drifting data.
    """
    values=np.asarray(values,dtype=float);n=len(values)
    if values.ndim!=1 or not np.isfinite(values).all() or type(block_length) is not int or not 1<=block_length<=n or repeats<100:
        raise ValueError('need finite 1D samples, valid integer block length and >=100 replicates')
    rng=np.random.default_rng(seed);means=np.empty(repeats);offset=np.arange(block_length)
    for i in range(repeats):
        starts=rng.integers(n,size=int(np.ceil(n/block_length)))
        indices=((starts[:,None]+offset)%n).ravel()[:n]
        means[i]=values[indices].mean()
    return dict(meanCI95=np.quantile(means,[.025,.975]).tolist(),replicates=repeats,
                blockSamples=block_length,approximateBlocks=n/block_length,
                interpretation='Conditional moving-block bootstrap of a stationary candidate; not a stationarity/trend test.')


def estimate(t,v,lo,hi,utility,*,step=.1,seed=1729):
    clock=window_clock(t,lo,hi);values=np.interp(clock,t,v)
    avg=float(np.trapezoid(values,clock)/(hi-lo))
    variance=float(np.trapezoid((values-avg)**2,clock)/(hi-lo))
    regular=np.linspace(lo,hi,max(8,int(np.ceil((hi-lo)/step))+1))
    samples=np.interp(regular,t,v);corr=utility.correlationTime(regular,samples)
    out=dict(mean=avg,variance=variance,tau=corr['tau'],effectiveSamples=corr['effectiveSamples'],correlationCensored=corr['censored'],acfAssumption='Stationary candidate; drift can distort this diagnostic.')
    if variance<=np.finfo(float).eps**2*max(1.,avg**2):
        out['bootstrap']=dict(meanCI95=[avg,avg],interpretation='Numerically constant observable, not a stochastic stationarity certificate.')
    elif corr['tau'] is not None and not corr['censored']:
        block=max(1,int(np.ceil(5*corr['tau']/(regular[1]-regular[0]))))
        if len(regular)/block>=4:out['bootstrap']=block_bootstrap(samples,block,seed=seed)
        else:out['bootstrap']=dict(unavailable='Fewer than four blocks of five estimated correlation times',blockSamples=block)
    else:out['bootstrap']=dict(unavailable='Correlation estimate is censored')
    return out


def native_history(run):
    keys=('t','kineticEnergy','thermalEnergy','totalEnergy','rmsMachMass','logDensityStdParticle','forceWork','thermalExchange','relativeEnergyBudgetResidual','relativeMassChange','meanSpeed','instantaneousSourcePower','instantaneousBathPower')
    values=[]
    for line in (run/'history.jsonl').open():
        if line.strip():
            row=json.loads(line);values.append([row[k] for k in keys])
    data=np.asarray(values);return {k:data[:,j] for j,k in enumerate(keys)}


def analyze(run,output,utilities,*,blocks=(40,80,120,160),grid=32,radius=.0625,cadence=.1):
    start=time.perf_counter();output.mkdir(parents=True,exist_ok=False)
    u=load(utilities/'turbulence.py','stationarity_utils');b=load(utilities/'turbulenceBudget.py','observation_utils')
    config=json.loads((run/'input.json').read_text());result=json.loads((run/'result.json').read_text())
    if result['status']!='completed' or result['actualTime']<blocks[-1] or result['stageChecks']['bad'] or not result['acceptedPositiveFinite'] or not result['sourcesStable']:
        raise ValueError('Require a completed guarded trajectory covering the assessment window')
    native=native_history(run);np.savez_compressed(output/'native-series.npz',**native)
    t=native['t'];window_clock(t,blocks[0],blocks[-1]);params=config['params'];length=config['simulation']['L']
    paths=sorted((run/'samples').glob('*.npz'));sampling=json.loads((run/'sampling.json').read_text())
    sampleTimes=np.asarray(sampling['times']);
    if len(paths)!=len(sampleTimes):raise ValueError('snapshot/sampling inventory mismatch')
    schedule=b.AcceptedSampleClock(cadence);indices=[i for i,t in enumerate(sampleTimes) if schedule.due(t)]
    if indices[-1]!=len(paths)-1:indices.append(len(paths)-1)
    edges=np.linspace(-8,8,161);rows=[];pdfs=[];fields=[];spectra=[];evidence={};maxMassError=0.;maxExtensiveError=0.;nativeSupports=[]
    for counter,i in enumerate(indices):
        file=paths[i]
        with np.load(file) as z:s={k:z[k] for k in z.files}
        actual=float(s['t'])
        if actual!=sampleTimes[i]:raise ValueError('actual snapshot time differs from manifest')
        m,rho,v,energy=(s[k] for k in ('masses','densities','velocities','internalEnergies'))
        row=u.particleStatistics(m,rho,v,energy,gamma=params['gamma'],length=length);row['t']=actual
        logrho=np.log(rho/(m.sum()/length**2));logp=np.log(s['pressures']/(params['rho0']*params['soundSpeed']**2/params['gamma']))
        pdf={}
        for kind,weights in [('Mass',m),('Volume',m/rho)]:
            pdf['logDensity'+kind]=u.weightedPDF(logrho,weights,edges);pdf['logPressure'+kind]=u.weightedPDF(logp,weights,edges)
        observed=b.depositConservative(s['positions'],m,v,rho,s['pressures'],energy,np.zeros_like(v),gridSize=grid,length=length,radius=radius)
        if observed['metadata']['emptyCellFraction']>0:raise ValueError('empty observation cells; do not impute spectra')
        # The new map is ij; the established 2D spectral API is xy.
        velocity=observed['velocity'].swapaxes(0,1);density=observed['density'].T
        spectral=u.spectra2D(velocity,density);total=spectral['velocityEnergy'][1:].sum()
        row.update(boxModeEnergy=float(spectral['boxModeEnergy']),boxModeFraction=float(spectral['boxModeEnergy']/total) if total else 0.,velocitySpectralEnergy=float(total),logPressureStdMass=float(np.sqrt(np.average((logp-np.average(logp,weights=m))**2,weights=m))))
        rows.append(row);pdfs.append(pdf);fields.append(velocity.astype(np.float32));spectra.append(spectral);evidence[file.name]=digest(file)
        maxMassError=max(maxMassError,abs(observed['metadata']['relativeMassError']));maxExtensiveError=max(maxExtensiveError,observed['metadata']['maxExtensiveConservationError']);nativeSupports.append(float(s['supports'].max()))
        if counter%50==0 or counter==len(indices)-1:print(json.dumps(dict(run=run.name,sample=counter+1,total=len(indices),t=actual,wallSeconds=time.perf_counter()-start)),flush=True)
    times=np.asarray([r['t'] for r in rows]);series={k:np.asarray([r[k] for r in rows]) for k in rows[0]}
    for key in pdfs[0]:series['pdf_'+key]=np.asarray([p[key] for p in pdfs])
    for key in ('velocityEnergy','solenoidalEnergy','compressionalEnergy','densityPower'):series[key]=np.asarray([s[key] for s in spectra])
    np.savez_compressed(output/'statistics.npz',**series,pdfEdges=edges)
    metrics=('rmsMachMass','logDensityStdMass','logDensityStdVolume','logPressureStdMass','boxModeEnergy','boxModeFraction')
    windows=list(zip(blocks[:-1],blocks[1:]));reports=[]
    for j,(lo,hi) in enumerate(windows):
        record=dict(window=[lo,hi],native={k:estimate(t,native[k],lo,hi,u,seed=1729+j) for k in ('kineticEnergy','rmsMachMass','thermalEnergy','instantaneousSourcePower','instantaneousBathPower')},field={k:estimate(times,series[k],lo,hi,u,seed=1729+j) for k in metrics})
        record['budget']={k+'Rate':power_rate(t,native[k],lo,hi) for k in ('forceWork','thermalExchange','kineticEnergy','thermalEnergy','totalEnergy')}
        record['budget']['instantaneousSourceQuadratureDifference']=float(mean(t,native['instantaneousSourcePower'],lo,hi)-record['budget']['forceWorkRate'])
        record['budget']['instantaneousBathQuadratureDifference']=float(mean(t,native['instantaneousBathPower'],lo,hi)-record['budget']['thermalExchangeRate'])
        record['distributions']={key:mean(times,series['pdf_'+key],lo,hi).tolist() for key in pdfs[0]}
        record['spectra']={key:mean(times,series[key],lo,hi).tolist() for key in ('velocityEnergy','solenoidalEnergy','compressionalEnergy','densityPower')}
        chosen=(times>=lo)&(times<=hi);corr=u.fieldCorrelationTime(times[chosen],np.asarray(fields)[chosen]);record['fieldCorrelation']={k:corr[k] for k in ('tau','effectiveSamples','censored')}
        reports.append(record)
    distributionChange={key:[float(.5*np.abs(np.asarray(right['distributions'][key])-left['distributions'][key]).sum()) for left,right in zip(reports[:-1],reports[1:])] for key in pdfs[0]}
    screens={k:u.stationarityScreen(t,native[k],blocks=blocks) for k in ('kineticEnergy','rmsMachMass')}
    screens.update({k:u.stationarityScreen(times,series[k],blocks=blocks) for k in metrics})
    fullCorr=u.fieldCorrelationTime(times[(times>=blocks[0])&(times<=blocks[-1])],np.asarray(fields)[(times>=blocks[0])&(times<=blocks[-1])])
    turnovers=float(mean(times,series['rmsVelocityMass'],blocks[0],blocks[-1])*(blocks[-1]-blocks[0])/length)
    secondary=[]
    for lo,hi in [(10,20),(20,30),(30,40)]+[(lo,lo+20) for lo in range(40,160,20)]:
        if t[0]<=lo<hi<=t[-1]:
            secondary.append(dict(window=[lo,hi],kineticEnergy=float(mean(t,native['kineticEnergy'],lo,hi)),
                rmsMachMass=float(mean(t,native['rmsMachMass'],lo,hi)),
                boxModeEnergy=float(mean(times,series['boxModeEnergy'],lo,hi)),
                boxModeFraction=float(mean(times,series['boxModeFraction'],lo,hi)),
                inputPower=power_rate(t,native['forceWork'],lo,hi),
                bathPower=power_rate(t,native['thermalExchange'],lo,hi)))
    report=dict(secondaryWindows=secondary,run=str(run.resolve()),seed=params['seed'],blocks=blocks,windows=reports,distributionChangeTotalVariation=distributionChange,screens=screens,scalarScreenPassed=bool(all(s['passed'] for s in screens.values()) and turnovers>=10),boxCrossings=turnovers,fieldCorrelation={k:fullCorr[k] for k in ('tau','effectiveSamples','censored')},samples=len(times),requestedStatisticsCadence=cadence,missedStatisticsTargets=schedule.missedTargets,gridSize=grid,observationRadius=radius,maxNativeSupport=max(nativeSupports),maxMassMappingError=maxMassError,maxExtensiveMappingError=maxExtensiveError,
        maxNativeEnergyBudgetResidual=float(abs(native['relativeEnergyBudgetResidual']).max()),maxNativeMassDrift=float(abs(native['relativeMassChange']).max()),cost=dict(loopWallSeconds=result['loopWallSeconds'],msPerStep=1000*result['loopWallSeconds']/result['nSteps'],steps=result['nSteps']),
        limitations=['A failed screen does not prove nonstationarity; a pass does not prove distributional stationarity.','Conditional correlation/bootstrap estimates do not remove drift; mean-CI overlap is not a stationarity test.','Box spectra measure volume velocity variance, not mass kinetic energy.','Fixed observation filter; no stationary, inertial-range or physical viscosity claim follows automatically.','Deposition acceleration is a zero placeholder used only for field statistics; source budgets use dense native power/work.'],evidence=dict(inputSHA256=digest(run/'input.json'),historySHA256=digest(run/'history.jsonl'),resultSHA256=digest(run/'result.json'),sampleSHA256=evidence,utilities={p.name:digest(p) for p in (utilities/'turbulence.py',utilities/'turbulenceBudget.py')},driverSHA256=digest(Path(__file__))),wallSeconds=time.perf_counter()-start)
    (output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(run=run.name,scalarScreenPassed=report['scalarScreenPassed'],wallSeconds=report['wallSeconds'])),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--utility-directory',type=Path,default=Path(__file__).resolve().parents[1]/'src/warpSPH/utils')
    p.add_argument('--blocks',nargs='+',type=float,default=[40,80,120,160]);p.add_argument('--grid',type=int,default=32);p.add_argument('--radius',type=float,default=.0625);p.add_argument('--cadence',type=float,default=.1)
    args=p.parse_args()
    if len(args.blocks)<3 or not np.isfinite(args.blocks).all() or np.any(np.diff(args.blocks)<=0):p.error('need >=3 increasing finite assessment boundaries')
    analyze(args.run,args.output,args.utility_directory,blocks=args.blocks,grid=args.grid,radius=args.radius,cadence=args.cadence)
