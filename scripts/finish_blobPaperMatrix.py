"""Offline comparison/recording verification; launches no physical trajectory."""
import argparse
import json
from pathlib import Path
import subprocess
import urllib.request
import numpy as np


def finish(root,report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from warpsph_remote.sources import HDF5Source
    from warpsph_remote.daemon import descriptor
    from warpSPH.runner.media import encodeFrames
    from study_blobPaperMatrix import analyze
    analyze(root,report)
    matrix=json.loads((report/'matrix.json').read_text());rows={r['name']:r for r in matrix['rows']};prov=matrix['provenance'];ref=Path(prov['reference'])
    histories={};finals={};verification={};d=descriptor()
    def get(path):
        request=urllib.request.Request(d['endpoint']+path,headers={'Authorization':'Bearer '+d['token']})
        with urllib.request.urlopen(request,timeout=15) as response:return json.load(response)
    remote={r['id']:r for r in get('/v1/runs')['runs']}
    def read_case(name,folder,native=False):
        data=folder if native else folder/'warp';history=json.loads((data/'diagnosticHistory.json').read_text());histories[name]=history
        final=data/('finalState.npz' if (data/'finalState.npz').exists() else 'failedState.npz')
        if final.exists():finals[name]=dict(np.load(final))
        path=data if native else Path(rows[name]['exportPath']);source=HDF5Source(path/'trajectory.h5')
        rid=rows[name]['remoteID'] if name in rows else json.loads((data/'registration.json').read_text())['remoteRunID']
        historyRemote=get('/v1/runs/'+rid+'/history');np.testing.assert_allclose(historyRemote['times'],source.times,rtol=0,atol=1e-12)
        if not (path/'output.mp4').exists():encodeFrames(str(path/'images'),str(path))
        movie=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=nb_frames','-of','json',str(path/'output.mp4')]))['streams'][0]
        count=len(list((path/'images').glob('frame_*.png')));assert count==int(movie['nb_frames'])
        recordingError=name in rows and rows[name]['classification']=='recording error'
        if not recordingError:assert count==len(source.times),(name,count,len(source.times))
        verification[name]=dict(id=rid,status=remote[rid]['status'],historyFrames=len(source.times),videoFrames=count,historyMatches=True,
            recordingFailure=recordingError,imageScope='Startup-only images; recorder failed after first step, not a usable trajectory comparison.' if recordingError else 'Stored vispy frames; invalid terminal state remains separate NPZ.')
        if name in rows:
            expected='completed' if rows[name]['passed'] else 'failed';assert remote[rid]['status']==expected,(name,remote[rid]['status'],expected)
    # Include the reused reference as a separate comparison anchor.
    read_case('Spheral reference',ref,True)
    for name,r in rows.items():
        folder=Path(next(x['folder'] for x in prov['rows'] if x['name']==name)) if not name.startswith('spheral-fresh') else root/name
        read_case(name,folder,name.startswith('spheral-fresh'))
    cohorts={
        'density-viscosity':['repo-default','matched','d-legacy_q-4_e-native','d-massReplacement_q-0_e-native','same-material','unit-Q'],
        'energy':['matched','d-massReplacement_q-4_e-entropy','native-compatible','entropy-compatible','node-dudt','dudt-compatible','node-dudt-half','dudt-compatible-half','one-sided','spheral-fresh','spheral-fresh-dudt','spheral-fresh-dudt-half-v2'],
        'scheduling':['matched','paper-density-v2','paper-density-entropy-v2','paper-work','paper-work-cached','cached-gradient','spheral-density-volume'],
        'numerics':['matched','small-fixed-dt','half-grid','float32','fixed-support','generated-H','quintic','mean-volume-v2','scalar-eos-compat']}
    for cohort,names in cohorts.items():
        fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
        for name in ['Spheral reference',*names]:
            if name not in histories:continue
            h=histories[name];valid=np.array([r['minEnergy'] is not None and r['minEnergy']>0 and r['relativeEnergyDrift'] is not None for r in h])
            for ax,field in zip(axes,('cloudMassFraction','minEnergy','relativeEnergyDrift')):
                values=[np.nan if field=='cloudMassFraction' and not ok else (r[field] if r[field] is not None else np.nan) for r,ok in zip(h,valid)]
                ax.plot([r['tKH'] for r in h],values,label=name,lw=1.1)
            if not valid.all():axes[0].axvline(h[-1]['tKH'],lw=.5,ls=':',alpha=.4)
        for ax,title in zip(axes,('Admissible cold-cloud mass','Minimum energy (stops retained)','Energy drift')):ax.set(xlabel='t/KH',title=title);ax.grid(alpha=.2)
        axes[1].axhline(0,color='k',lw=.5);axes[2].set_yscale('symlog',linthresh=1e-12);axes[0].legend(fontsize=6)
        residual=[r['relativeEnergyDrift'] for name in ['Spheral reference',*names] if name in histories for r in histories[name] if r['relativeEnergyDrift'] is not None]
        axes[2].set_ylim(min(-1e-12,1.2*min(residual)),max(1e-12,1.2*max(residual)))
        fig.savefig(report/(cohort+'.png'),dpi=180);plt.close(fig)
    selected=['Spheral reference','spheral-fresh','repo-default','matched','d-legacy_q-4_e-native','same-material','one-sided','paper-density-v2','paper-work-cached','spheral-density-volume','fixed-support','quintic']
    fig,axes=plt.subplots(4,3,figsize=(12,14),layout='constrained')
    for ax,name in zip(axes.flat,selected):
        if name not in finals or (name in rows and not rows[name]['passed']):ax.set_axis_off();continue
        a=finals[name];ax.scatter(*a['positions'].T,c=a['densities'],s=3,cmap='turbo',vmin=.75,vmax=10)
        ax.set(xlim=(3,12),ylim=(1,9),aspect='equal',title=name)
    fig.suptitle('Actual1-KH particle density; common color range. Coarse shock-dominated regime, not nonlinear validation.')
    fig.savefig(report/'endpoints.png',dpi=160);plt.close(fig)
    # Complete local metrics to both native reference integrators, preserving UIDs.
    metrics={}
    for name,r in rows.items():
        if not r['passed'] or name not in finals:continue
        a=finals[name];metrics[name]={}
        for anchor in ('Spheral reference','spheral-fresh'):
            b=finals[anchor];assert np.array_equal(a['UIDs'],b['UIDs']);cloud=b['materials']==1
            dx=a['positions']-b['positions'];L=np.array([40.,10.]);dx-=L*np.round(dx/L)
            metrics[name][anchor]=dict(cloudPositionRMS=float(np.sqrt(np.mean(dx[cloud]**2))),
                cloudEnergyRMS=float(np.sqrt(np.mean((a['internalEnergies'][cloud]-b['internalEnergies'][cloud])**2))),
                allPositionRMS=float(np.sqrt(np.mean(dx**2))))
    (report/'recordingVerification.json').write_text(json.dumps(verification,indent=2)+'\n')
    (report/'localMetrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    # Exact paired contrasts for the core factorial; no best-fit parameter search.
    def core(d,q,e):
        if (d,q,e)==('massReplacement',4,'native'):return 'matched'
        if (d,q,e)==('legacy',0,'native'):return 'repo-default'
        return f'd-{d}_q-{q}_e-{e}'
    contrasts=[]
    for e in ('native','entropy'):
        for q in (0,4):
            a,b=core('legacy',q,e),core('massReplacement',q,e);contrasts.append(dict(factor='density',before=a,after=b,coldMassChange=rows[b]['coldMass']-rows[a]['coldMass']))
        for density in ('legacy','massReplacement'):
            a,b=core(density,0,e),core(density,4,e);contrasts.append(dict(factor='Q length',before=a,after=b,coldMassChange=rows[b]['coldMass']-rows[a]['coldMass']))
    for density in ('legacy','massReplacement'):
        for q in (0,4):
            a,b=core(density,q,'native'),core(density,q,'entropy');contrasts.append(dict(factor='final partition',before=a,after=b,coldMassChange=rows[b]['coldMass']-rows[a]['coldMass']))
    (report/'pairedContrasts.json').write_text(json.dumps(contrasts,indent=2)+'\n')
    print('Verified',len(verification),'recordings; wrote cohort figures and paired/local comparisons',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();finish(a.input,a.output)
