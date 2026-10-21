"""Quantify standalone thermal Sedov results and verify daemon/video evidence."""
import argparse
import json
from pathlib import Path
import subprocess
import urllib.request

import numpy as np

from compare_spheralMaterials import save
from compare_spheralSedov import exact,radial_profile


def analyze(root,report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report.mkdir(parents=True,exist_ok=True)
    rows={};failed={};histories={};frames={}
    for folder in sorted([*root.glob('n64-dt*'),*(root/'entropy').glob('n64-dt*')]):
        if not folder.is_dir():continue
        policy='entropy' if folder.parent.name=='entropy' else 'paper'
        settings=json.loads((folder/'settings.json').read_text());expected=np.r_[0.,np.cumsum(settings['timeGrid'])]
        for solver in ('spheral','warp'):
            data=folder/solver
            if not (data/'registration.json').exists():continue
            key=f'{policy}-{folder.name}-{solver}'
            history=json.loads((data/'diagnosticHistory.json').read_text());unique=[]
            for row in history:
                if unique and row['time']==unique[-1]['time']:
                    assert row==unique[-1]
                else:unique.append(row)
            np.testing.assert_allclose([r['time'] for r in unique],expected[:len(unique)],atol=1.e-12,rtol=0)
            histories[key]=unique
            baseline=dict(np.load(folder/'input.npz'));initial=dict(np.load(data/'initialState.npz'))
            for field in baseline:np.testing.assert_allclose(initial[field],baseline[field],atol=1.e-12,rtol=0)
            failure=data/'failure.json'
            if failure.exists():
                entry=json.loads(failure.read_text());state=dict(np.load(data/'failedState.npz'))
                entry.update(final=unique[-1],finite={k:int((~np.isfinite(v)).sum()) for k,v in state.items()})
                if solver=='spheral':
                    i=int(np.argmin(state['internalEnergies']));entry.update(minimumEnergyUID=int(state['UIDs'][i]),initialEnergyAtMinimum=float(initial['internalEnergies'][i]))
                failed[key]=entry
            else:
                entry=json.loads((data/'comparisonRun.json').read_text());assert entry['passed']
                a=dict(np.load(data/'finalState.npz'));frames[key]=a
                radii=np.linalg.norm(a['positions'],axis=1);reference,_=exact(radii,.05);mask=(radii>.12)&(radii<.45);V=a['masses']/a['densities']
                entry['fixedCoreEnergyL1']=float(np.dot(V[mask],abs(a['internalEnergies'][mask]-reference['energy'][mask]))/V[mask].sum())
                rows[key]=entry
            if solver=='spheral':
                counts=json.loads((data/'overlayCalls.json').read_text());entry['overlayCalls']=counts
                assert counts['finalEnergy']==counts['rawPredictor']==0,counts
                assert counts['density']==entry['steps'] and counts['RHS']==2*entry['steps'],counts
                if (data/'thermalStageAudit.npz').exists():
                    a=np.load(data/'thermalStageAudit.npz')['node'];a=a[a[:,3]==1]
                    checks={}
                    for rhs in np.unique(a[:,0]):
                        z=a[a[:,0]==rhs];power=np.sum(z[:,4]*(np.sum(z[:,11:13]*z[:,34:36],axis=1)+z[:,31]))
                        scale=np.sum(z[:,4]*(abs(np.sum(z[:,11:13]*z[:,34:36],axis=1))+abs(z[:,31])))
                        checks[str(int(rhs))]=dict(powerResidual=float(power),scaledResidual=float(abs(power)/max(scale,1.e-100)))
                    entry['nativeStagePower']=checks
            else:
                audit=json.loads((folder/'thermalAudit.json').read_text());entry['thermalAudit']=dict(compatibleCalls=audit['compatibleCalls'],RHS=len(audit['rows']),
                    maxNormalizedPowerResidual=max(v['normalizedPowerResidual'] or 0 for v in audit['rows']))
                assert audit['compatibleCalls']==0 and len(audit['rows'])==2*entry['steps']
                entry['EOSAudit']=json.loads((folder/'eosAudit.json').read_text())
    contrasts={};timestep={}
    for solver in ('spheral','warp'):
        levels=[(int(k.split('-dt')[1].split('-')[0]),k) for k in rows if k.endswith(solver)]
        levels.sort();errors=[abs(rows[k]['final']['relativeEnergyDrift']) for _,k in levels]
        timestep[solver]=dict(levels=[n for n,k in levels],absoluteEnergyDrifts=errors,reductionFactors=[a/b for a,b in zip(errors[:-1],errors[1:])])
    for level in (1,2,4):
        keys=[f'entropy-n64-dt{level}-{s}' for s in ('spheral','warp')]
        if not all(k in rows for k in keys):continue
        s,w=[frames[k] for k in keys]
        contrasts[str(level)]={key:dict(spheral=rows[keys[0]]['final'][key],warp=rows[keys[1]]['final'][key]) for key in ('densityL1','pressureL1','velocityL1','density2Front','tangentialKineticFraction')}
        contrasts[str(level)]['UIDPositionRMS']=float(np.sqrt(np.mean(np.sum((s['positions']-w['positions'])**2,axis=1))))
    save(report/'comparison.json',dict(rows=rows,failures=failed,timestep=timestep,solverContrasts=contrasts))
    fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    rr=np.linspace(.001,.45,1500);ref,R=exact(rr,.05)
    for ax,field in zip(axes.flat[:3],('density','pressure','velocity')):
        ax.plot(rr,ref[field],color='k',label='Point-blast similarity')
        for name,a in frames.items():
            r,fields=radial_profile(a,64);ax.plot(r,fields[field],lw=1,label=name,ls='--' if 'dt1-' in name else '-',color='C0' if name.endswith('spheral') else 'C1')
        ax.set(xlabel='r',ylabel=field,xlim=(0,.45));ax.grid(alpha=.2)
    ax=axes.flat[3]
    for name,h in histories.items():
        if name.startswith('entropy'):ax.plot([v['time'] for v in h],[v['relativeEnergyDrift'] for v in h],label=name,lw=1)
    ax.set(xlabel='t',ylabel='relative total-energy drift');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=6);fig.suptitle('Standalone entropy-distributed du/dt; no compatible correction; 64²')
    fig.savefig(report/'profiles.png',dpi=175);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4.5),constrained_layout=True)
    for solver,ax in zip(('spheral','warp'),axes):
        for level in (1,2,4):
            key=f'paper-n64-dt{level}-{solver}'
            if key not in histories:continue
            h=histories[key];ax.plot([v['time'] for v in h],[v['minEnergy'] for v in h],'.-',label=f'dt/{level}')
        if solver=='warp':ax.text(.03,.08,'dt/1 ends nonfinite; last finite point shown',transform=ax.transAxes,fontsize=8)
        ax.axhline(0,color='k',lw=.6);ax.set(title=solver+' Eq.78 standalone',xlabel='t',ylabel='minimum specific energy',xlim=(0,.0001));ax.legend();ax.grid(alpha=.2)
    fig.savefig(report/'rawFailure.png',dpi=175);plt.close(fig)
    print(json.dumps(dict(timestep=timestep,contrasts=contrasts),indent=2))


def verify(root,report):
    from warpsph_remote.daemon import descriptor
    from warpsph_remote.sources import HDF5Source
    info=descriptor()
    def get(path):
        request=urllib.request.Request(info['endpoint']+path,headers={'Authorization':'Bearer '+info['token']})
        with urllib.request.urlopen(request,timeout=15)as response:return json.load(response)
    remote={r['id']:r for r in get('/v1/runs')['runs']};rows={}
    for registration in sorted(root.rglob('registration.json')):
        data=registration.parent;failed=(data/'failure.json').exists();r=json.loads((data/('failure.json' if failed else 'comparisonRun.json')).read_text())
        folder=data if data.name=='spheral' else next(p.parent for p in data.glob('*/trajectory.h5'))
        rid=r['remoteRunID'];source=HDF5Source(folder/'trajectory.h5');history=get('/v1/runs/'+rid+'/history')
        np.testing.assert_allclose(history['times'],source.times,rtol=0,atol=1.e-12)
        video=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=nb_frames','-of','json',str(folder/'output.mp4')]))
        n=int(video['streams'][0]['nb_frames']);expected='failed' if failed else 'completed'
        assert n==len(source.times) and remote[rid]['status']==expected,(data,n,len(source.times),remote[rid]['status'])
        if not failed:assert abs(source.times[-1]-.05)<1.e-8
        rows[str(data.relative_to(root))]=dict(id=rid,status=expected,videoFrames=n,historyFrames=len(source.times),matched=True)
    report.mkdir(parents=True,exist_ok=True);save(report/'recordingVerification.json',rows);print('Verified',len(rows),'daemon histories and videos')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('analyze','verify'));p.add_argument('--input',type=Path,required=True)
    p.add_argument('--report',type=Path,default=Path('reports/materialEOS/spheral/sedovThermal'));args=p.parse_args()
    (analyze if args.mode=='analyze' else verify)(args.input.resolve(),args.report.resolve())
