"""Compare complete/failed Warp CRK energy-policy rows and recording evidence."""
import argparse
import json
from pathlib import Path
import subprocess
import urllib.request

import numpy as np

from compare_spheralMaterials import save
from study_crkEnergySuite import CASES,POLICIES,sod_exact,wave_fields,vortex_fields,sedov_exact
from compare_spheralSedov import radial_profile


def load_rows(root):
    rows={};histories={};states={};initials={}
    for folder in sorted(root.iterdir()):
        if not (folder/'settings.json').exists():continue
        data=folder/'warp';result=data/'comparisonRun.json';failure=data/'failure.json'
        if not result.exists() and not failure.exists():continue
        settings=json.loads((folder/'settings.json').read_text());name=folder.name
        row=json.loads((result if result.exists() else failure).read_text());row['settings']=settings
        row['numericalFailure']=failure.exists()
        if failure.exists():row['failure']=json.loads(failure.read_text())
        history=json.loads((data/'diagnosticHistory.json').read_text());histories[name]=history
        np.testing.assert_allclose([h['time'] for h in history],np.r_[0,np.cumsum(settings['timeGrid'])][:len(history)],atol=1.e-11,rtol=0)
        actual=dict(np.load(data/'initialState.npz'));reference=dict(np.load(folder/'input.npz'))
        for k in reference:np.testing.assert_allclose(actual[k],reference[k],rtol=0,atol=1.e-12)
        case=settings['case']
        if case not in initials:initials[case]=reference
        else:
            for k in reference:np.testing.assert_array_equal(reference[k],initials[case][k])
        state=dict(np.load(data/('failedState.npz' if failure.exists() else 'finalState.npz')))
        order=np.argsort(state['UIDs']);state={k:v[order] for k,v in state.items()};states[name]=state
        np.testing.assert_array_equal(state['UIDs'],reference['UIDs'])
        audit=json.loads((folder/'energyAudit.json').read_text());row['audit']={k:v for k,v in audit.items() if k!='stagePower'}
        values=np.array([[np.nan if v is None else v for v in item] for item in audit['stagePower']]);valid=np.isfinite(values[:,0])
        row['audit'].update(maxAbsolutePowerResidual=float(abs(values[valid,0]).max()),maxScaledPowerResidual=float(np.max(abs(values[valid,0])/np.maximum(values[valid,1],1.e-100))))
        if not settings['choice']['compatible']:assert audit['compatibleAccumulations']==audit['compatibleFinalizations']==0
        row['lastDiagnostic']=history[-1];row['completed']=bool(not failure.exists() and row.get('passed'))
        if row['completed']:
            assert len(history)==settings['exactSteps']+1,(name,len(history),settings['exactSteps'])
            if settings['choice']['compatible']:assert audit['compatibleFinalizations']>=row['steps']
            row['maxEntropyL1PerMass']=max(h['entropyL1PerMass'] for h in history)
            if case=='vortex':
                bg=np.array([.5,.25]);ke=lambda a:float(.5*np.dot(a['masses'],np.sum((a['velocities']-bg)**2,axis=1)))
                first,last=ke(reference),ke(state)
                row['vorticalKineticEnergy']=dict(initial=first,final=last,relativeChange=last/first-1)
                end=settings['end'];analytic=vortex_fields(state['positions'],end)
                relative=(state['positions']-bg*end+5)%10-5;mask=np.linalg.norm(relative,axis=1)<2.
                w=state['masses']/state['densities'];errors={}
                for field,key in [('densities','density'),('pressures','pressure'),('velocities','velocity'),('internalEnergies','energy')]:
                    diff=abs(state[field]-analytic[key]);diff=np.linalg.norm(diff,axis=1) if diff.ndim>1 else diff
                    errors[key+'L1']=float(np.dot(w[mask],diff[mask])/w[mask].sum())
                row['vortexCoreErrors']=dict(radius=2.,**errors)
            if case=='wave':
                row['waveNormalizedVelocityL1']=history[-1]['velocityL1']/1.e-4
                row['waveAmplitudeRatio']=history[-1]['waveVelocityAmplitude']/1.e-4
        rows[name]=row
    return rows,histories,states


def contrast(a,b,domain):
    length=domain[1]-domain[0];dx=(a['positions']-b['positions']+.5*length)%length-.5*length
    weight=b['masses']/b['densities'];out=dict(positionRMS=float(np.sqrt(np.mean(np.sum(dx*dx,axis=1)))))
    for field in ('densities','pressures','internalEnergies','velocities'):
        diff=abs(a[field]-b[field]);norm=abs(b[field])
        if diff.ndim>1:diff=np.linalg.norm(diff,axis=1);norm=np.linalg.norm(norm,axis=1)
        out[field+'RelativeL1']=float(np.dot(weight,diff)/max(np.dot(weight,norm),1.e-100))
    return out


def analyze(root,report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report.mkdir(parents=True,exist_ok=True);rows,histories,states=load_rows(root)
    comparisons={};timestep={};policy_pairs=[('raw','compatible'),('entropy','compatible'),('entropyCompatible','compatible'),('entropy','entropyCompatible')]
    for case,cfg in CASES.items():
        for split in (1,2):
            for first,second in policy_pairs:
                ka,kb=[f'{case}-dt{split}-{v}' for v in (first,second)]
                if ka in rows and kb in rows and rows[ka]['completed'] and rows[kb]['completed']:
                    comparisons[f'{case}-dt{split}-{first}_vs_{second}']=contrast(states[ka],states[kb],cfg['domain'])
        for policy in POLICIES:
            ka,kb=[f'{case}-dt{split}-{policy}' for split in (1,2)]
            if ka in rows and kb in rows and rows[ka]['completed'] and rows[kb]['completed']:
                a,b=[rows[k]['lastDiagnostic'] for k in (ka,kb)]
                delta={k:dict(dt=a[k],half=b[k],difference=b[k]-a[k]) for k in a if k!='time' and a[k] is not None and b[k] is not None}
                ea,eb=abs(a['relativeEnergyDrift']),abs(b['relativeEnergyDrift'])
                timestep[case+'-'+policy]=dict(finalDiagnosticChanges=delta,energyReductionFactor=ea/max(eb,1.e-100),
                    finalStateContrast=contrast(states[ka],states[kb],cfg['domain']))
    save(report/'comparison.json',dict(rows=rows,policyComparisons=comparisons,timestep=timestep,scope=json.loads((root/'provenance.json').read_text())['scope']))
    write_tables(rows,report)
    colors=dict(zip(POLICIES,['C0','C1','C2','C3']))
    fig,axes=plt.subplots(5,3,figsize=(14,17),constrained_layout=True)
    for i,case in enumerate(CASES):
        for policy in POLICIES:
            name=f'{case}-dt2-{policy}'
            if name not in rows:continue
            h=histories[name];t=[r['time'] for r in h]
            keys=('relativeEnergyDrift','entropyL1PerMass','modeAmplitude' if case=='kh' else 'velocityL1')
            for ax,key in zip(axes[i],keys):
                ax.plot(t,[r.get(key) for r in h],label=policy,lw=1,color=colors[policy]);ax.set(xlabel='t',ylabel=key,title=case);ax.grid(alpha=.2)
        axes[i,0].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
        if i==0:axes[i,0].legend(fontsize=8)
    fig.suptitle('Warp base CRK energy-policy suite — half timestep; failed rows end early')
    fig.savefig(report/'histories.png',dpi=150);plt.close(fig)
    for case in ('sod','wave','sedov'):
        fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
        if case=='sod':rr=np.linspace(-.5,.5,1500);reference,_=sod_exact(rr,.2)
        elif case=='wave':rr=np.linspace(0,1,1000);reference=wave_fields(rr[:,None],1.);reference['velocity']=reference['velocity'][:,0]
        else:rr=np.linspace(.001,.45,1500);reference,_=sedov_exact(rr,.05)
        for ax,key in zip(axes,('density','pressure','velocity')):
            ax.plot(rr,reference[key],color='k',label='Exact' if case!='wave' else 'Linear reference',lw=1.5)
            for policy in POLICIES:
                name=f'{case}-dt2-{policy}'
                if name not in rows or not rows[name]['completed']:continue
                a=states[name]
                if case=='sedov':r,fields=radial_profile(a,64);field=fields[key]
                else:
                    r=a['positions'][:,0];field=a[{'density':'densities','pressure':'pressures','velocity':'velocities'}[key]]
                    if field.ndim>1:field=field[:,0]
                    order=np.argsort(r);r,field=r[order],field[order]
                ax.plot(r,field,lw=.9,label=policy,color=colors[policy]);ax.set(xlim=(rr[0],rr[-1]),xlabel='r' if case=='sedov' else 'x',ylabel=key);ax.grid(alpha=.2)
        axes[0].legend(fontsize=7);fig.suptitle(case+' — final profiles, half timestep; completed rows only')
        fig.savefig(report/(case+'Profiles.png'),dpi=170);plt.close(fig)
    for case in ('kh','vortex'):
        fig,axes=plt.subplots(1,4,figsize=(15,4),constrained_layout=True);im=None
        for ax,policy in zip(axes,POLICIES):
            name=f'{case}-dt2-{policy}';ax.set_title(policy)
            if name not in states or not rows[name]['completed']:ax.text(.5,.5,'Failed / unavailable',ha='center',transform=ax.transAxes);continue
            a=states[name];lo,hi=CASES[case]['domain'];x=(a['positions']-lo)%(hi-lo)+lo
            if case=='kh':field=a['densities'];limits=(1,2);cmap='viridis'
            else:
                reference=vortex_fields(x,2.);field=np.linalg.norm(a['velocities']-reference['velocity'],axis=1);limits=(0,.15);cmap='inferno'
            im=ax.scatter(*x.T,c=field,s=1.5,vmin=limits[0],vmax=limits[1],cmap=cmap,linewidths=0);ax.set(aspect='equal',xlim=CASES[case]['domain'],ylim=CASES[case]['domain'],xlabel='x',ylabel='y')
        if im is not None:fig.colorbar(im,ax=axes,shrink=.6,label='Density' if case=='kh' else 'Velocity error magnitude')
        fig.suptitle(case+' — final half-step states');fig.savefig(report/(case+'States.png'),dpi=180);plt.close(fig)
    print('Analyzed',len(rows),'rows;',sum(r['completed'] for r in rows.values()),'completed,',sum(r['numericalFailure'] for r in rows.values()),'numerical failures')


def write_tables(rows,report):
    text=['# Quantitative energy-policy comparison','',
        'Columns follow the four policies defined in the report. Main accuracy/entropy tables use the half timestep. Failed rows are not compared as completed solutions.','']
    header='| Case / metric | compatible | raw | entropy | entropyCompatible |'
    separator='|---|---:|---:|---:|---:|'
    def cell(case,policy,field,split=2,scale=1.):
        row=rows.get(f'{case}-dt{split}-{policy}')
        if row is None:return 'pending'
        if not row['completed']:return f"FAIL t={row['lastDiagnostic']['time']:.6g}"
        value=row['lastDiagnostic'][field]
        return f'{value*scale:.6g}'
    text+=['## Final physical diagnostics','',header,separator]
    for case,field,scale in [('sod','densityL1',1),('sod','pressureL1',1),('sod','velocityL1',1),('sod','energyL1',1),
        ('kh','modeAmplitude',1),('kh','transverseKineticEnergy',1),('kh','pressureRMS',1),('wave','velocityL1',1.e4),
        ('wave','waveVelocityAmplitude',1.e4),('wave','waveVelocityPhase',1),('vortex','velocityL1',1),('vortex','densityL1',1),
        ('sedov','densityL1',1),('sedov','pressureL1',1),('sedov','velocityL1',1),('sedov','density2Front',1)]:
        suffix=' / initial amplitude' if case=='wave' and scale==1.e4 else ''
        text.append('| '+case+' / '+field+suffix+' | '+' | '.join(cell(case,p,field,scale=scale) for p in POLICIES)+' |')
    for title,field in [('Final relative total-energy drift','relativeEnergyDrift'),('Final particle entropy L1 per mass','entropyL1PerMass')]:
        text+=['','## '+title,'',header,separator]
        for case in CASES:text.append('| '+case+' | '+' | '.join(cell(case,p,field) for p in POLICIES)+' |')
    text+=['','## Vortex core and rotational energy','',header,separator]
    for label,group,field in [('Velocity L1 within r<2','vortexCoreErrors','velocityL1'),('Density L1 within r<2','vortexCoreErrors','densityL1'),
        ('Vortical kinetic-energy relative change','vorticalKineticEnergy','relativeChange')]:
        values=[]
        for policy in POLICIES:
            row=rows.get('vortex-dt2-'+policy)
            values.append(f'{row[group][field]:.6g}' if row and row['completed'] else 'pending/failed')
        text.append('| '+label+' | '+' | '.join(values)+' |')
    text+=['','## Simulation costs','',
        'Each cell is wall seconds / wall milliseconds per attempted step. Completed Warp loop timings are separately retained in comparison.json. These include recording, diagnostics and research hooks; failed-run wall includes startup/watchdog costs.','',header,separator]
    for case in CASES:
        for split in (1,2):
            values=[]
            for policy in POLICIES:
                row=rows.get(f'{case}-dt{split}-{policy}')
                if row is None:values.append('pending');continue
                cost=row.get('failure',row) if row['numericalFailure'] else row
                wall=cost['wallSeconds'];steps=cost['steps'];values.append(f'{wall:.3f} / {1000*wall/max(steps,1):.3f}'+(' failed' if row['numericalFailure'] else ''))
            text.append('| '+case+f' dt/{split} | '+' | '.join(values)+' |')
    (report/'tables.md').write_text('\n'.join(text)+'\n')


def verify(root,report):
    from warpsph_remote.daemon import descriptor
    from warpsph_remote.sources import HDF5Source
    info=descriptor()
    def get(path):
        request=urllib.request.Request(info['endpoint']+path,headers={'Authorization':'Bearer '+info['token']})
        with urllib.request.urlopen(request,timeout=15)as response:return json.load(response)
    remote={r['id']:r for r in get('/v1/runs')['runs']};rows={}
    for registration in sorted(root.glob('*/warp/registration.json')):
        data=registration.parent;failure=data/'failure.json';result=data/'comparisonRun.json'
        if not failure.exists() and not result.exists():continue
        r=json.loads((failure if failure.exists() else result).read_text())
        folder=next(p.parent for p in data.glob('*/trajectory.h5'));rid=r['remoteRunID'];source=HDF5Source(folder/'trajectory.h5');history=get('/v1/runs/'+rid+'/history')
        np.testing.assert_allclose(history['times'],source.times,atol=1.e-11,rtol=0)
        video=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=nb_frames','-of','json',str(folder/'output.mp4')]))
        n=int(video['streams'][0]['nb_frames']);expected='failed' if failure.exists() else 'completed'
        assert remote[rid]['status']==expected and len(source.times)==n,(data,remote[rid]['status'],len(source.times),n)
        settings=json.loads((data.parent/'settings.json').read_text())
        # Columnar recording labels are float32; physics/diagnostics use f64.
        if not failure.exists():assert abs(source.times[-1]-float(np.float32(settings['end'])))<1.e-10
        rows[data.parent.name]=dict(id=rid,status=expected,historyFrames=len(source.times),videoFrames=n,matched=True,
            finalRecordingTime=float(source.times[-1]),finalDiagnosticTarget=settings['end'],
            videoRebuiltFromSavedFrames=(data/'videoRebuild.json').exists())
    report.mkdir(parents=True,exist_ok=True);save(report/'recordingVerification.json',rows);print('Verified',len(rows),'daemon histories and videos')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('analyze','verify'));p.add_argument('--input',type=Path,required=True)
    p.add_argument('--report',type=Path,default=Path('reports/materialEOS/crkEnergySuite'));args=p.parse_args()
    (analyze if args.mode=='analyze' else verify)(args.input.resolve(),args.report.resolve())
