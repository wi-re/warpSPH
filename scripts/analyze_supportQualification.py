"""Offline support-only qualification, restart comparisons and recording checks."""
import argparse
import json
from pathlib import Path
import numpy as np
from compare_spheralMaterials import save
from analyze_blobThermal import material_contrast
from analyze_crkEnergySuite import contrast,verify


def repair_resume_recording(root,report):
    """Encode recorded post-step images, retaining the extra checkpoint preview."""
    import h5py
    import shutil
    import subprocess
    from warpSPH.runner.media import encodeFrames
    repaired={}
    for f in root.glob('*-resume/warp/comparisonRun.json'):
        run=json.loads(f.read_text());export=Path(run['exportPath'])
        if (f.parent/'videoRebuild.json').exists():continue
        with h5py.File(export/'trajectory.h5') as data:keys=sorted(data['times'],key=lambda k:int(k.split('_')[-1]))
        video=export/'output.mp4'
        n=int(json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=nb_frames','-of','json',str(video)]))['streams'][0]['nb_frames'])
        if n==len(keys):continue
        assert n==len(keys)+1
        images=export/'recordingImages';images.mkdir(exist_ok=False)
        for i,key in enumerate(keys):
            index=int(key.split('_')[-1]);shutil.copyfile(export/'images'/f'frame_{index:07d}.png',images/f'frame_{i:07d}.png')
        video.rename(export/'outputOriginal.mp4');encodeFrames(str(images),str(export))
        meta=dict(frames=len(keys),originalFrames=n,physicsRerun=False,
            reason='Resume video includes an initial checkpoint preview; trajectory stores only post-step frames. Re-encode exactly their existing vispy images; retain original video/images.')
        save(f.parent/'videoRebuild.json',meta);repaired[f.parent.parent.name]=meta
    report.mkdir(parents=True,exist_ok=True);save(report/'recordingRepairs.json',repaired)


def analyze(root,report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report.mkdir(parents=True,exist_ok=True)
    pr=json.loads((root/'provenance.json').read_text());rows={};states={};histories={}
    for name in pr['rows']:
        folder=root/name;data=folder/'warp'
        failure=data/'failure.json';path=failure if failure.exists() else data/'comparisonRun.json'
        if not path.exists():continue
        r=json.loads(path.read_text());s=json.loads((folder/'settings.json').read_text())
        a=dict(np.load(data/('failedState.npz' if failure.exists() else 'finalState.npz')))
        h=json.loads((data/'diagnosticHistory.json').read_text());initial=dict(np.load(folder/'input.npz'))
        for k in initial:np.testing.assert_array_equal(np.load(data/'initialState.npz')[k],initial[k])
        for k in ('masses','materials','UIDs'):np.testing.assert_array_equal(a[k],initial[k])
        start=s.get('resumeStepOffset',0)*s['dt']
        np.testing.assert_allclose([v['time'] for v in h],start+np.r_[0,np.cumsum(s['timeGrid'])][:len(h)],atol=2.e-11,rtol=0)
        sa=json.loads((folder/'supportAudit.json').read_text());da=json.loads((folder/'densityAudit.json').read_text())
        complete=not failure.exists() and r.get('passed',False)
        if complete:
            assert len(h)==s['steps']+1,(name,len(h),s['steps'])
            assert sa['rhs']==da['rhs']==2*s['steps']
            assert len(da['resets'])==s['steps'] and da['continuityUpdates']==2*s['steps'] and da['negativeEOSCalls']==0
        r.update(settings=s,completed=complete,maxima={k:max(abs(v[k]) for v in h if v[k] is not None) for k in h[0] if k!='time'},
            densitySummary=dict(rhs=da['rhs'],resets=len(da['resets']),negativeEOSCalls=da['negativeEOSCalls']),supportAudit=sa)
        rows[name]=r;states[name]=a;histories[name]=h
    blob={};parent=Path(pr['parent'])
    for split in (1,2):
        key=f'blob-dt{split}-material'
        if key not in states:continue
        native=Path(f'results/materialEOS/crk_heating_v2_2026-10-20/spheral-{"grid" if split==1 else "halfGrid"}-averaged')
        reference=dict(np.load(native/'finalState.npz'));base=dict(np.load(parent/f'blob-dt{split}-warp/warp/finalState.npz'))
        weighted=dict(np.load(parent/f'blob-dt{split}-weighted/warp/finalState.npz'))
        blob[key]=dict(vsNative=material_contrast(states[key],reference),vsWeighted=material_contrast(states[key],weighted),
            baselineVsNative=material_contrast(base,reference),weightedVsNative=material_contrast(weighted,reference))
    comparisons={};temporal={}
    for case in ('hydro','sod','kh'):
        for split in (1,2):
            base=f'{case}-dt{split}-warp'
            for policy in ('material','weighted'):
                name=f'{case}-dt{split}-{policy}'
                if name in states and base in states:
                    comparisons[name]=contrast(states[name],states[base],rows[name]['settings']['domain'])
                    comparisons[name]['velocityMaxAbsoluteDifference']=float(np.linalg.norm(states[name]['velocities']-states[base]['velocities'],axis=1).max())
        for policy in ('warp','material','weighted'):
            first,second=(f'{case}-dt{split}-{policy}' for split in (1,2))
            if first in states and second in states:temporal[case+'-'+policy]=contrast(states[first],states[second],rows[first]['settings']['domain'])
    restarts={}
    for name,r in rows.items():
        s=r['settings']
        if 'restartParent' not in s:continue
        ref=s['restartParent'];suffix=histories[ref][s['resumeStepOffset']:];h=histories[name]
        assert len(h)==len(suffix)
        errors={k:float(abs(states[name][k].astype(float)-states[ref][k].astype(float)).max()) for k in states[name]}
        diagnosticErrors={k:max(abs(a[k]-b[k]) for a,b in zip(h,suffix)) for k in h[0]}
        restarts[name]=dict(finalMaxAbsoluteErrors=errors,diagnosticSuffixMaxAbsoluteErrors=diagnosticErrors,
            bitwiseFinal=all(v==0 for v in errors.values()),passed=max(errors.values())<1.e-10,
            checkpoint=json.loads((root/ref/'checkpoint.json').read_text()),freshProcess=True,discardedSearchCaches=True)
    save(report/'comparison.json',dict(provenance=pr,rows=rows,blob=blob,policyComparisons=comparisons,timestep=temporal,restarts=restarts))
    lines=['# Support qualification costs and invariants','',
        '| Case/grid/policy | Complete | Max energy drift | Max momentum diagnostic | Min u | Wall s | ms/step |',
        '|---|---|---:|---:|---:|---:|---:|']
    for name,r in rows.items():
        h=histories[name];momentum=r['maxima'].get('momentumError',r['maxima'].get('relativeMomentumError'))
        lines.append(f"| {name} | {r['completed']} | {r['maxima']['relativeEnergyDrift']:.3g} | {momentum:.3g} | {min(v['minEnergy'] for v in h):.5g} | {r['wallSeconds']:.2f} | {1000*r['wallSeconds']/max(r['steps'],1):.2f} |")
    lines.extend(['','Momentum: simple cases use |p-p0|/total mass; blob retains its existing relativeMomentumError diagnostic.'])
    (report/'tables.md').write_text('\n'.join(lines)+'\n')
    colors=dict(warp='C0',material='C1',weighted='C2')
    labels=dict(warp='Existing Owen',material='Owen + material weights',weighted='Native lookup/bounds + weights')
    fig,axes=plt.subplots(3,3,figsize=(13,10),layout='constrained')
    for i,case in enumerate(('hydro','sod','kh')):
        keys={'hydro':('pressureError','displacementInSpacings','maxVelocity'),
            'sod':('densityL1','pressureL1','transverseVelocityRMS'),
            'kh':('modeAmplitude','entropyChangePerMass','pressureRMS')}[case]
        for split in (1,2):
            for policy in colors:
                name=f'{case}-dt{split}-{policy}'
                if name not in histories:continue
                h=histories[name]
                for ax,key in zip(axes[i],keys):
                    ax.plot([v['time'] for v in h],[v[key] for v in h],color=colors[policy],ls='--' if split==1 else '-',label=f'{labels[policy]} dt/{split}',lw=1.)
                    ax.set(xlabel='Time',ylabel=key,title=case);ax.grid(alpha=.2)
        if axes[i,0].get_legend_handles_labels()[0]:axes[i,0].legend(fontsize=7,ncol=2)
    fig.savefig(report/'diagnostics.png',dpi=150);plt.close(fig)
    for case in ('hydro','kh'):
        names=[f'{case}-dt2-{p}' for p in colors if f'{case}-dt2-{p}' in states]
        if not names:continue
        fig,axes=plt.subplots(1,len(names),figsize=(4*len(names),4),layout='constrained',squeeze=False)
        for ax,name in zip(axes.flat,names):
            a=states[name];lo,hi=rows[name]['settings']['domain'];x=(a['positions']-lo)%(hi-lo)+lo
            color=a['materials'] if case=='hydro' else a['velocities'][:,1]
            ax.scatter(*x.T,c=color,s=2,cmap='coolwarm',linewidths=0)
            ax.set(title=labels[rows[name]['settings']['supportPolicy']],aspect='equal',xlim=(lo,hi),ylim=(lo,hi))
        fig.suptitle(case+' at t=2, fine timestep')
        fig.savefig(report/(case+'Particles.png'),dpi=170);plt.close(fig)
    names=[f'sod-dt2-{p}' for p in colors if f'sod-dt2-{p}' in states]
    if names:
        from compare_spheralSod1d import exact
        xx=np.linspace(-.5,.5,1600);q,_=exact(xx,.2)
        fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
        for ax,key in zip(axes,('density','pressure','velocity')):
            ax.plot(xx,q[key],color='k',lw=1.3,label='Exact Riemann solution')
            for name in names:
                a=states[name];nx=3*rows[name]['settings']['nx'];ny=len(a['masses'])//nx
                x=a['positions'][:,0].reshape(ny,nx).mean(0);mask=abs(x)<.5
                field=a['velocities'][:,0] if key=='velocity' else a['densities' if key=='density' else 'pressures']
                value=field.reshape(ny,nx).mean(0);policy=rows[name]['settings']['supportPolicy']
                ax.plot(x[mask],value[mask],color=colors[policy],lw=1.,ls='--' if policy=='material' else '-',label=labels[policy])
            ax.set(xlabel='x',ylabel=key);ax.grid(alpha=.2)
        axes[0].legend(fontsize=8);fig.suptitle('2D Sod strip at t=.2, fine timestep; transverse-column averages')
        fig.savefig(report/'sodProfiles.png',dpi=160);plt.close(fig)
    key='blob-dt2-material'
    if key in states:
        native=Path('results/materialEOS/crk_heating_v2_2026-10-20/spheral-halfGrid-averaged')
        panels=[('Spheral reference',dict(np.load(native/'finalState.npz'))),
            ('Existing Owen support',dict(np.load(parent/'blob-dt2-warp/warp/finalState.npz'))),
            ('Existing Owen + material weights',states[key]),
            ('Native lookup/bounds + weights\nWarp before-RHS timing',dict(np.load(parent/'blob-dt2-weighted/warp/finalState.npz')))]
        fig,axes=plt.subplots(1,4,figsize=(17,4),layout='constrained')
        for ax,(title,a) in zip(axes,panels):
            order=np.argsort(a['materials'],kind='stable');x=a['positions']%np.array([40.,10.])
            ax.scatter(*x[order].T,c=a['materials'][order],s=4,cmap='coolwarm',vmin=0,vmax=1,linewidths=0)
            ax.set(title=title,xlim=(5,11),ylim=(2.5,7.5),aspect='equal')
        fig.suptitle('Same blob startup and 1050 timesteps, t = 1 KH')
        fig.savefig(report/'blobComparison.png',dpi=170);plt.close(fig)
    print('ANALYZED',len(rows),'physical executions; restarts',restarts.keys(),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('analyze','verify','repair'));p.add_argument('--input',required=True,type=Path)
    p.add_argument('--report',type=Path,default=Path('reports/materialEOS/supportQualification'));a=p.parse_args()
    dict(analyze=analyze,verify=verify,repair=repair_resume_recording)[a.mode](a.input.resolve(),a.report.resolve())
