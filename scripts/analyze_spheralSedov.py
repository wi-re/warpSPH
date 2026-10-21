"""Similarity errors, shock/radial symmetry, solver contrasts and recordings."""
import argparse
import json
from pathlib import Path
import subprocess
import urllib.request
import numpy as np
from compare_spheralMaterials import save
from compare_spheralSedov import exact,radial_profile,oracle
from analyze_spheralKH import contrast


def analyze(root,report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report.mkdir(parents=True,exist_ok=True);provenance=json.loads((root/'provenance.json').read_text())
    rows={};contrasts={};initials={};frames={};histories={};timestep={};strict_failures={}
    variants=json.loads((root/'diagnosticVariants.json').read_text()) if (root/'diagnosticVariants.json').exists() else []
    paths=list(provenance['rows']);half=root/'n64-halfStrict'
    if (half/'spheral/comparisonRun.json').exists():paths.append(str(half))
    for path in paths:
        folder=Path(path);settings=json.loads((folder/'settings.json').read_text());times=np.r_[0,np.cumsum(settings['timeGrid'])];pair={}
        initial=dict(np.load(folder/'input.npz'))
        for solver in ('spheral','warp'):
            data=folder/solver
            if solver=='warp' and not (data/'comparisonRun.json').exists():
                candidate=next((v for v in variants if v['soundPolicy']=='spheralZero' and Path(v['reference']).parent==folder),None)
                if candidate is None:raise ValueError(('No completed comparison; strict result remains failed',folder))
                data=Path(candidate['folder'])/'warp'
            name=folder.name+'-'+solver;run=json.loads((data/'comparisonRun.json').read_text());history=json.loads((data/'diagnosticHistory.json').read_text())
            unique=[]
            for row in history:
                if unique and row['time']==unique[-1]['time']:
                    if row!=unique[-1]:raise ValueError(('Different diagnostics at repeated time',name))
                else:unique.append(row)
            run['duplicatePeriodicCallbackRows']=len(history)-len(unique);history=unique
            run['comparisonFolder']=str(data)
            run['EOSSoundPolicy']='nativeZero' if solver=='spheral' else run['settings'].get('eosSound','strict')
            if (data.parent/'eosAudit.json').exists():run['EOSAudit']=json.loads((data.parent/'eosAudit.json').read_text())
            if not run['passed']:raise ValueError(('Qualification failed',name))
            np.testing.assert_allclose([r['time'] for r in history],times,atol=1.e-12,rtol=0,err_msg='Stage time mismatch '+name)
            if solver=='spheral':
                counts=json.loads((data/'overlayCalls.json').read_text());n=run['steps']
                assert all(counts[k]==value for k,value in dict(density=n,finalEnergy=n,rawPredictor=n,RHS=2*n).items()),counts
                run['overlayCalls']=counts
            start=dict(np.load(data/'initialState.npz'));initials[name]={k:float(abs(start[k]-initial[k]).max()) for k in initial}
            if max(initials[name].values())>1.e-12:raise ValueError(('Startup mismatch',name,initials[name]))
            a=dict(np.load(data/'finalState.npz'));pair[solver]=a;frames[name]=a;rows[name]=run;histories[name]=history
            radii=np.linalg.norm(a['positions'],axis=1);reference,_=exact(radii,.05);mask=(radii>.12)&(radii<.45);V=a['masses']/a['densities']
            run['fixedCoreEnergyL1']=float(np.dot(V[mask],abs(a['internalEnergies'][mask]-reference['energy'][mask]))/V[mask].sum())
            rows[name]['finalSupportRange']=[float(a['supports'].min()),float(a['supports'].max())]
            # A finite-width density isosurface is only a front marker. Its
            # late-time fitted exponent is separate from exact profile errors.
            mask=np.array([r['time']>=.025 and r['density2Front']>0 for r in history]);tt=np.array([r['time'] for r in history])[mask]
            rr=np.array([r['density2Front'] for r in history])[mask]
            run['lateFrontExponent']=float(np.polyfit(np.log(tt),np.log(rr),1)[0])
        contrasts[folder.name]=contrast(pair['warp'],pair['spheral'])
    for failure in root.glob('*/warp/failure.json'):
        folder=failure.parent.parent;record=json.loads(failure.read_text());audit=folder/'eosAudit.json'
        if audit.exists():record['EOSAudit']=json.loads(audit.read_text())
        state=dict(np.load(failure.parent/'failedState.npz'));record['nonfiniteCounts']={k:int((~np.isfinite(v)).sum()) for k,v in state.items()}
        strict_failures[folder.name]=record
    for solver in ('spheral','warp'):
        name='n64-halfStrict-'+solver
        if name in rows:
            baseline=rows['n64-'+solver]['final'];current=rows[name]['final']
            timestep[solver]={key:dict(baseline=baseline[key],half=current[key],change=current[key]-baseline[key]) for key in ('densityL1','pressureL1','velocityL1','density2Front','tangentialKineticFraction')}
    answer,raw=oracle();lam=np.asarray(answer._lam);z,w=np.polynomial.legendre.leggauss(4)
    q=lam[:-1,None]+.5*np.diff(lam)[:,None]*(1+z);R=answer.shockState(.05)[1];ref,_=exact((q*R).ravel(),.05)
    kinetic=2*np.pi*R**2*np.sum((.5*ref['density']*ref['velocity']**2).reshape(q.shape)*q*.5*np.diff(lam)[:,None]*w)
    save(report/'comparison.json',dict(provenance=provenance,rows=rows,solverContrasts=contrasts,initialDifferences=initials,timestepContrasts=timestep,strictFailures=strict_failures,
        analytic=dict(alpha=answer.alpha,nativeAlpha=raw,shockState=answer.shockState(.05),kineticEnergy=float(kinetic)),
        scope='Final NPZ fields at identical .05; per-step histories checked against shared variable grid. Similarity oracle is a cold point blast; simulation uses 12-particle W7 spike plus tiny ambient pressure.'))
    rr=np.linspace(.001,.45,2000);ref,R=exact(rr,.05)
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for ax,key in zip(axes.flat,('density','pressure','velocity','supports')):
        if key!='supports':ax.plot(rr,ref[key],color='k',lw=1.5,label='Similarity, point blast')
        ax.axvline(R,color='k',lw=.6,ls=':')
        for name,a in frames.items():
            nx=rows[name]['settings']['nx'];r,fields=radial_profile(a,nx)
            if key=='supports':
                radii=np.linalg.norm(a['positions'],axis=1);order=np.argsort(radii);ax.scatter(radii[order],a['supports'][order],s=.3,alpha=.3,label=name)
            else:ax.plot(r,fields[key],label=name,lw=1.,color='C0' if name.endswith('spheral') else 'C1',ls=':' if 'halfStrict' in name else ('--' if nx==64 else '-'))
        ax.set(xlim=(0,.45),xlabel='r',ylabel=key);ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);fig.suptitle('2D Sedov — Warp zero-sound controls; dotted 64² half grid uses strict EOS')
    fig.savefig(report/'profiles.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,9),constrained_layout=True)
    for i,nx in enumerate((64,128)):
        for j,solver in enumerate(('spheral','warp')):
            ax=axes[i,j];a=frames[f'n{nx}-'+solver];p=a['positions']
            im=ax.scatter(*p.T,c=a['densities'],cmap='viridis',s=2.5 if nx==64 else .7,vmin=0,vmax=4,linewidths=0,rasterized=True)
            ax.add_patch(plt.Circle((0,0),R,fill=False,color='red',ls='--',lw=.8))
            label='Spheral' if solver=='spheral' else 'Warp, experimental c=0 for u<0'
            ax.set(xlim=(-.4,.4),ylim=(-.4,.4),aspect='equal',title=f'{label}\n{nx}² — t=.05',xlabel='x',ylabel='y')
    fig.colorbar(im,ax=axes,shrink=.7,label='density; dashed red = exact shock radius')
    fig.savefig(report/'density.png',dpi=190);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for name,h in histories.items():
        t=np.array([r['time'] for r in h]);style=':' if 'halfStrict' in name else ('--' if name.startswith('n64-') else '-')
        for ax,key in zip(axes.flat,('density2Front','densityL1','tangentialKineticFraction','relativeEnergyDrift')):
            ax.plot(t,[r[key] for r in h],label=name,color='C0' if name.endswith('spheral') else 'C1',ls=style,lw=1.)
            ax.set(xlabel='t',ylabel=key);ax.grid(alpha=.2)
    t=np.linspace(1.e-6,.05,300);axes[0,0].plot(t,[answer.shockState(v)[1] for v in t],color='k',label='Similarity radius');axes[0,0].legend(fontsize=7)
    axes[1,1].ticklabel_format(axis='y',style='sci',scilimits=(0,0));fig.savefig(report/'diagnostics.png',dpi=180);plt.close(fig)
    print(json.dumps(dict(rows={k:dict(final=v['final'],wall=v['wallSeconds'],msPerStep=v['meanStepMilliseconds'],exponent=v['lateFrontExponent']) for k,v in rows.items()},contrasts=contrasts,timestep=timestep,strictFailures=strict_failures),indent=2))


def verify(root,report):
    from warpsph_remote.daemon import descriptor
    from warpsph_remote.sources import HDF5Source
    info=descriptor()
    def get(path):
        request=urllib.request.Request(info['endpoint']+path,headers={'Authorization':'Bearer '+info['token']})
        with urllib.request.urlopen(request,timeout=15)as response:return json.load(response)
    remote={r['id']:r for r in get('/v1/runs')['runs']};rows={};provenance=json.loads((root/'provenance.json').read_text())
    for registration in sorted(root.glob('*/[sw]*/registration.json')):
            data=registration.parent;solver=data.name;failed=(data/'failure.json').exists()
            run=json.loads((data/('failure.json' if failed else 'comparisonRun.json')).read_text())
            folder=data if solver=='spheral' else (next(p.parent for p in data.glob('*/trajectory.h5')) if failed else Path(run['exportPath']))
            rid=run['remoteRunID'];source=HDF5Source(folder/'trajectory.h5');history=get('/v1/runs/'+rid+'/history')
            np.testing.assert_allclose(history['times'],source.times,rtol=0,atol=1.e-12)
            video=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=nb_frames','-of','json',str(folder/'output.mp4')]))
            count=int(video['streams'][0]['nb_frames'])
            expected='failed' if failed else 'completed'
            if count!=len(source.times) or remote[rid]['status']!=expected or (not failed and abs(source.times[-1]-.05)>1.e-8):raise ValueError(('Recording mismatch',data,count,len(source.times),remote[rid]['status']))
            rows[data.parent.name+'-'+solver]=dict(id=rid,status=remote[rid]['status'],historyFrames=len(source.times),videoFrames=count,historyMatches=True)
    if len(rows)!=8:raise ValueError(('Expected six completed and two failed controls',len(rows)))
    report.mkdir(exist_ok=True,parents=True);save(report/'recordingVerification.json',rows);print('Verified',len(rows),'daemon histories and videos')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('analyze','verify'));p.add_argument('--input',type=Path,required=True)
    p.add_argument('--report',type=Path,default=Path('reports/materialEOS/spheral/sedov'));a=p.parse_args()
    if a.mode=='analyze':analyze(a.input.resolve(),a.report.resolve())
    else:verify(a.input.resolve(),a.report.resolve())
