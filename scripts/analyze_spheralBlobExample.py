"""Qualification evidence for the bundled Spheral blob execution, not Warp fitting."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import numpy as np
from compare_spheralMaterials import save


def analyze(root,output,verify=False):
    from warpsph_remote.sources import HDF5Source
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output.mkdir(parents=True,exist_ok=True)
    execution=json.loads((root/'execution.json').read_text());runtime=json.loads((root/'nativeConfiguration.json').read_text())
    paper=bool(runtime.get('paperPolicies'))
    h=json.loads((root/'diagnosticHistory.json').read_text());summary=json.loads((root/'summary.json').read_text())
    log=(root/'runtime.log').read_text()
    match=re.search(r'hydro  :  <[^\n]*\.(\w+) object',log)
    actualHydro=runtime.get('hydroClass',match.group(1) if match else '')
    generated=dict(np.load(root/'generatedState.npz'));initial=dict(np.load(root/'initializedState.npz'))
    for k in ('positions','velocities','densities','masses','materials','UIDs','internalEnergies'):
        np.testing.assert_array_equal(generated[k],initial[k],err_msg=k)
    supportDelta=initial['supports']/generated['supports']-1
    hIterations=re.findall(r'iterateIdealH: \(iteration, deltaH\) = \((\d+), ([\deE.+-]+)\)',log)
    source=HDF5Source(root/'trajectory.h5');tKH=json.loads((root/'settings.json').read_text())['tKH']
    nativeMass=np.loadtxt(next((root/'nativeOutputs').glob('**/cloud_mass_history.gnu')),ndmin=2)
    nativeDifference=[]
    for row in nativeMass:
        index=int(row[0]);nativeDifference.append(abs(h[index].get('originalCloudMassFraction',h[index]['cloudMassFraction'])-row[2]))
    assert max(nativeDifference,default=0)<1e-12
    qualification=dict(driverCompleted=execution['status']=='completed',reachedEnd=abs(summary['time']-runtime['goalTime'])<1e-8,
        finite=all(v is not None for row in h for v in row.values()),positiveEnergy=all(r['minEnergy'] is not None and r['minEnergy']>0 for r in h),
        boundedVelocity=all(r['maxVelocity'] is not None and r['maxVelocity']<=100 for r in h),
        energy=max(abs(r['relativeEnergyDrift']) for r in h if r['relativeEnergyDrift'] is not None)<1e-5,
        materialMass=max(abs(r['materialMassError']) for r in h)<1e-12,
        correctSolver='CRKSPH' in actualHydro)
    operatorAudits={}
    if paper:
        for name in ('operatorAudit','operatorAuditRHS'):
            path=root/(name+'.json')
            operatorAudits[name]=json.loads(path.read_text()) if path.exists() else {'passed':False,'error':'Audit missing'}
            qualification[name]=operatorAudits[name]['passed']
        calls=json.loads((root/'paperPolicyCalls.json').read_text())
        qualification['loadedPaperPolicies']=all(v>0 for v in calls.values())
    fig,axes=plt.subplots(1,3,figsize=(13,3.8),layout='constrained');t=np.array([r['tKH'] for r in h])
    for ax,field,title in zip(axes,('minEnergy','cloudMassFraction','relativeEnergyDrift'),('Minimum internal energy','Cold-cloud mass fraction (all gas)' if paper else 'Original cold-cloud mass fraction','Relative total-energy change')):
        ax.plot(t,[r[field] for r in h]);ax.set(xlabel=r'$t/\tau_{KH}$',title=title);ax.grid(alpha=.2)
    axes[0].axhline(0,color='k',lw=.7);axes[0].set_yscale('symlog',linthresh=.01)
    validEnergy=[r['minEnergy'] for r in h if r['minEnergy'] is not None]
    axes[0].set_ylim(min(0,min(validEnergy)*1.2),max(.2,max(validEnergy)*1.2))
    axes[0].set_ylabel('Symmetric log scale; linear near zero')
    axes[2].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    fig.suptitle('Paper-configured Spheral CRKSPH: W7 / nh1 / entropy work / corrected material density' if paper else 'Bundled CRKSPH example: actual native evolution; negative energy is flagged without repair')
    fig.savefig(output/'diagnostics.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,3),layout='constrained');snapshots=[]
    for ax,target in zip(axes,(1.,4.,8.) if paper else (0.,1.,2.5)):
        if target>source.times[-1]/tKH+1e-6:
            ax.text(.5,.5,f'{target:g} KH not reached',ha='center',transform=ax.transAxes);ax.set_axis_off();continue
        index=int(np.argmin(abs(np.asarray(source.times)/tKH-target)));a=source.read(index)
        ax.scatter(*a['positions'].T,c=a['materials'],cmap='coolwarm',s=.5,vmin=0,vmax=1)
        actual=float(source.times[index]/tKH);ax.set(xlim=(0,40),ylim=(0,10),aspect='equal',title=f'Actual t/KH={actual:.4f}')
        snapshots.append(dict(target=target,time=float(source.times[index]),tKH=actual))
    fig.suptitle('Original material labels; published snapshot times' if paper else 'Original material labels; reference remains unqualified if internal energy becomes negative')
    fig.savefig(output/'materials.png',dpi=180);plt.close(fig)
    if paper:
        fig,axes=plt.subplots(1,3,figsize=(14,3),layout='constrained')
        color=matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(.75,10),cmap='turbo')
        for ax,target in zip(axes,(1.,4.,8.)):
            if target>source.times[-1]/tKH+1e-6:
                ax.text(.5,.5,f'{target:g} KH not reached',ha='center',transform=ax.transAxes);ax.set_axis_off();continue
            index=int(np.argmin(abs(np.asarray(source.times)/tKH-target)));a=source.read(index)
            color=ax.scatter(*a['positions'].T,c=a['densities'],cmap='turbo',s=.5,vmin=.75,vmax=10,rasterized=True)
            ax.set(xlim=(0,40),ylim=(0,10),aspect='equal',title=f'Actual t/KH={source.times[index]/tKH:.4f}')
        fig.colorbar(color,ax=axes,label='Density; fixed paper Fig21 range 0.75–10',shrink=.7)
        fig.suptitle('Particle density samples, full periodic domain; no interpolation or repaired states')
        fig.savefig(output/'density.png',dpi=180);plt.close(fig)
    result=dict(execution=execution,nativeConfiguration=runtime,verifiedHydroClass=actualHydro,summary=summary,qualification=qualification,passed=all(qualification.values()),
        supportInitialization=dict(maxRelativeChange=float(abs(supportDelta).max()),medianRelativeChange=float(np.median(abs(supportDelta))),
            lastNativeIteration=int(hIterations[-1][0]) if hIterations else None,
            lastNativeDeltaH=float(hIterations[-1][1]) if hIterations else None,
            minGenerated=float(generated['supports'].min()),maxGenerated=float(generated['supports'].max()),
            minInitialized=float(initial['supports'].min()),maxInitialized=float(initial['supports'].max())),
        negativeEnergy=dict(first=next((r['time'] for r in h if r['minEnergy'] is not None and r['minEnergy']<0),None),
            affectedSteps=sum(r['minEnergy'] is not None and r['minEnergy']<0 for r in h),
            mostNegative=min(r['minEnergy'] for r in h if r['minEnergy'] is not None),
            maxNegativeParticles=max(r['negativeEnergyParticles'] for r in h) if all('negativeEnergyParticles' in r for r in h) else None),
        nativeMassDiagnostic=dict(samples=len(nativeMass),maxAbsoluteDifference=max(nativeDifference,default=0)),
        performance=dict(wallSeconds=summary['wallSeconds'],millisecondsPerStep=1000*summary['wallSeconds']/max(summary['steps'],1),
            scope='Native integration/diagnostics/publishing after initialization; excludes post-rendered video.'),
        initializedStateSHA256=hashlib.sha256((root/'initializedState.npz').read_bytes()).hexdigest(),snapshots=snapshots)
    if paper:
        result.update(operatorAudits=operatorAudits,paperPolicyCalls=calls,loadedOverlay=json.loads((root/'paperOverlayLoaded.json').read_text()),
            qualificationScope='Admissibility, conservation, runtime dispatch and formula checks. Literature mixing agreement is a separate comparison, not an automatic consequence of these gates.')
    if verify:
        import urllib.request,subprocess
        from warpsph_remote.daemon import descriptor
        info=descriptor();rid=json.loads((root/'registration.json').read_text())['remoteRunID']
        def get(path):
            request=urllib.request.Request(info['endpoint']+path,headers={'Authorization':'Bearer '+info['token']})
            with urllib.request.urlopen(request,timeout=20) as response:return json.load(response)
        run=next(r for r in get('/v1/runs')['runs'] if r['id']==rid);remote=get('/v1/runs/'+rid+'/history')
        np.testing.assert_allclose(remote['times'],source.times,rtol=0,atol=1e-12)
        assert run['status']==execution['status'],(run['status'],execution['status'])
        movie=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,nb_frames','-of','json',str(root/'output.mp4')]))['streams'][0]
        images=len(list((root/'images').glob('frame_*.png')));assert images==int(movie['nb_frames'])==len(source.times)
        result['daemonVerification']=dict(id=rid,status=run['status'],frames=remote['frames'],video=movie,
            qualificationIsSeparateFromDriverStatus=True)
    save(output/'qualification.json',result)
    print(json.dumps(dict(qualification=qualification,passed=result['passed'],summary=summary,negativeEnergy=result['negativeEnergy']),indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--verify-daemon',action='store_true');a=p.parse_args()
    analyze(a.input,a.output,a.verify_daemon)
