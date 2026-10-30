#!/usr/bin/env python3
"""Finite CPU-only analysis companion for a tracked serial turbulence campaign.

Waits for each validated terminal GPU result, analyzes it, then writes the
ensemble report. It never allocates CUDA, changes a running simulation, retries
a failed trajectory, or sends messages to another session.
"""
from pathlib import Path
import argparse,datetime,hashlib,json,os,subprocess,time,traceback
import numpy as np


def write(path,data):
    temporary=path.with_name(path.name+'.tmp');temporary.write_text(json.dumps(data,indent=2)+'\n');temporary.replace(path)


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_run(run,baseline,seed):
    result=json.loads((run/'result.json').read_text())
    if not (result['status']=='completed' and result['actualTime']==160 and result['acceptedPositiveFinite'] and result['sourcesStable'] and result['stageChecks']['bad']==0 and result['extraRHS']==0):raise ValueError('physical result validation failed')
    ca=json.loads((run/'input.json').read_text());cb=json.loads((baseline/'input.json').read_text())
    for c in (ca,cb):
        c['simulation'].pop('exportRoot');c['simulation'].pop('tLimit');c['params'].pop('seed')
    if ca!=cb:raise ValueError('undeclared configuration change')
    equalSources=json.loads((run/'sourceHashes.json').read_text())==json.loads((baseline/'sourceHashes.json').read_text())
    if not equalSources:raise ValueError('runtime source inventory differs from baseline')
    with np.load(run/'initial.npz') as x,np.load(baseline/'initial.npz') as y:
        initial={k:np.array_equal(x[k],y[k]) for k in x.files}
    if not all(initial.values()):raise ValueError('initial state differs from baseline')
    out=dict(runtimeInventoryEqual=equalSources,initialArrayEquality=initial,configurationEqualExceptEndpointSeedOutput=True)
    if seed==1234:
        count=0;keys=('t','kineticEnergy','thermalEnergy','totalEnergy','forceWork','thermalExchange','minEnergy','minDensity','minPressure')
        with (run/'history.jsonl').open() as a,(baseline/'history.jsonl').open() as b:
            for la,lb in zip(a,b):
                va,vb=json.loads(la),json.loads(lb)
                if vb['t']>=39.99:break
                if any(va[k]!=vb[k] for k in keys):raise ValueError('seed1234 prefix differs before old endpoint restriction')
                count+=1
        if count<49000:raise ValueError('unexpectedly short baseline prefix')
        out['bitwisePrefixRowsBefore39p99']=count
    return out


def video_audit(run,output):
    from PIL import Image,ImageDraw
    paths=list(run.glob('exports/**/*.mp4'));video=next(p for p in paths if p.parent.name!='images')
    info=json.loads(subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=nb_frames,width,height','-of','json',str(video)],capture_output=True,text=True,check=True).stdout)['streams'][0]
    n=int(info['nb_frames']);indices=[0,n//4,n//2,3*n//4,n-1];sheet=Image.new('RGB',(960,5*265),'white')
    for row,index in enumerate(indices):
        image=output/f'video-frame-{index}.png'
        subprocess.run(['ffmpeg','-v','error','-i',str(video),'-vf',f'select=eq(n\\,{index})','-frames:v','1','-y',str(image)],check=True)
        im=Image.open(image).convert('RGB');im.thumbnail((960,240));sheet.paste(im,(0,row*265+25));ImageDraw.Draw(sheet).text((8,row*265+5),f'{run.name}, frame {index}/{n-1}',fill='black')
    sheet.save(output/'video-contact-sheet.png')
    return dict(video=str(video),sha256=digest(video),inspectedAutomatically='Decoded frames and checked metadata; human/model visual review remains pending',frames=info,preparedFrameIndices=indices)


def ensemble(root,reports):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=root/'analysis';fig,axes=plt.subplots(3,2,figsize=(13,10),layout='constrained')
    rows=[]
    for report in reports:
        run=Path(report['run']);name=run.name;directory=output/name
        with np.load(directory/'native-series.npz') as n,np.load(directory/'statistics.npz') as s:
            for ax,key,title in [(axes[0,0],'kineticEnergy','Native kinetic energy'),(axes[0,1],'rmsMachMass','Mass RMS Mach')]:
                ax.plot(s['t'],np.interp(s['t'],n['t'],n[key]),label=f"seed{report['seed']}",lw=.9)
            for ax,key,title in [(axes[1,0],'boxModeEnergy','Absolute box-mode velocity energy'),(axes[1,1],'boxModeFraction','Box-mode velocity-energy fraction'),(axes[2,0],'logDensityStdMass','Mass-weighted log-density spread')]:
                ax.plot(s['t'],s[key],label=f"seed{report['seed']}",lw=.9)
            # Fixed five-unit means of signed input and bath exchange, not a
            # smoothed substitute for the retained instantaneous power ledger.
            edges=np.arange(0,161,5);power=[]
            for lo,hi in zip(edges[:-1],edges[1:]):power.append((np.interp(hi,n['t'],n['forceWork'])-np.interp(lo,n['t'],n['forceWork']))/(hi-lo))
            axes[2,1].plot((edges[1:]+edges[:-1])/2,power,label=f"seed{report['seed']}",marker='.',lw=.9)
        means=[w['native']['kineticEnergy']['mean'] for w in report['windows']]
        rows.append(dict(seed=report['seed'],kineticBlockMeans=means,kineticLastMinusFirst=means[-1]-means[0],scalarScreenPassed=report['scalarScreenPassed'],fieldCorrelation=report['fieldCorrelation'],cost=report['cost']))
    titles=['Native kinetic energy','Mass RMS Mach','Absolute box-mode velocity energy','Box-mode velocity-energy fraction','Mass-weighted log-density spread','Native input power: five-unit means']
    for ax,title in zip(axes.flat,titles):
        ax.set(title=title,xlabel='Time');ax.axvline(40,color='.4',ls='--',lw=.8);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Unchanged CRK128 model, three independent forcing seeds; primary assessment t40–160')
    fig.savefig(output/'ensemble.png',dpi=160);fig.savefig(output/'ensemble.pdf');plt.close(fig)
    late=np.array([r['kineticBlockMeans'] for r in rows]);result=dict(runs=rows,meanKAcrossSeeds=late.mean(axis=0).tolist(),betweenSeedKStandardDeviation=late.std(axis=0,ddof=1).tolist() if len(rows)>1 else None,completedRuns=len(rows),expectedRuns=3,allScalarScreensPassed=len(rows)==3 and all(r['scalarScreenPassed'] for r in rows),limitations='Three seeds give limited ensemble precision. Scalar pass/CI overlap does not certify full distributional stationarity. No physical viscosity or inverse-cascade certification.')
    write(output/'ensemble.json',result)
    return result


def report_markdown(root,ensembleResult,state,reportPath,roadmap,plans):
    rows=ensembleResult['runs'];lines=['# Long CRK128 stationarity investigation (2026-10-28)','',
        'The campaign requests three independent forcing seeds with the source-pinned verified model to t=160; the table lists successfully analyzed trajectories. Primary blocks are40–80,80–120 and120–160. No physical/numerical retuning or trajectory retries.','',
        '| Seed | Mean K40–80 | Mean K80–120 | Mean K120–160 | Scalar screen | Loop minutes | ms/step |','|---|---:|---:|---:|---|---:|---:|']
    for r in rows:
        k=r['kineticBlockMeans'];c=r['cost'];lines.append(f"|{r['seed']}|{k[0]:.6f}|{k[1]:.6f}|{k[2]:.6f}|{'pass' if r['scalarScreenPassed'] else 'fail'}|{c['loopWallSeconds']/60:.2f}|{c['msPerStep']:.2f}|")
    lines+=['',f"All scalar screens pass: {ensembleResult['allScalarScreensPassed']}. This is an assessment screen, not proof of stationarity. Inspect distributions, spectra, absolute box energy, within-window drift and field correlation before qualifying stationary data.",'',
        'Uncertainty estimates are conditional moving-block bootstraps using blocks of five estimated correlation times, only with at least four blocks. Drift is retained. Censored correlations and inadequate block counts are explicitly unavailable; no independent-particle error bars or trend removal. Three seeds leave limited ensemble precision.','',
        f"![Histories]({(root/'analysis/ensemble.png').resolve()})",'',
        f"Evidence: `{root}`. Per-run summary JSON includes native/field window moments, conditional uncertainty, PDF total-variation changes, spectra, field ACF and energy budgets. Native/fine snapshots and forcing history remain archived. Video contact sheets are prepared for subsequent model/human visual review.",'',
        'All successful trajectories passed raw/accepted positivity, finite endpoint, no-extra-RHS and source/provenance guards. Any skipped or failed run remains in the supervisor/analysis status; it must not be counted as stationary.']
    reportPath.write_text('\n'.join(lines)+'\n')
    today=datetime.date.today().isoformat();status=f"Long CRK128 t160 campaign: {len(rows)}/3 analyzed, scalar screens {sum(r['scalarScreenPassed'] for r in rows)}/{len(rows)}; [report](docs/{reportPath.name}). Distributional/ensemble stationarity requires interpretation."
    ptext=plans.read_text();pLines=ptext.splitlines();index=next(i for i,x in enumerate(pLines) if x.startswith('| [Driven compressible turbulence and multiphysics]'))
    original=pLines[index]
    for phase in ('authorized/launching','running'):
        marker=f'**Long campaign {phase}:** CRK128 seeds1234/5678/9012 to t160, source-pinned serial GPU,0.0125 output and dense power; primary40–160 stationarity/ensemble assessment.'
        pLines[index]=pLines[index].replace(marker,status)
    if pLines[index]==original and status not in original:
        cells=original.split('|');cells[4]=cells[4].rstrip()+' '+status+' ';pLines[index]='|'.join(cells)
    plans.write_text('\n'.join(pLines)+'\n')
    roadmapStatus=status.replace('docs/'+reportPath.name,os.path.relpath(reportPath,roadmap.parent))
    with roadmap.open('a') as file:file.write(f'\n\nLong campaign terminal analysis ({today}): {roadmapStatus}\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('manifest',type=Path);args=ap.parse_args();plan=json.loads(args.manifest.read_text());root=args.manifest.resolve().parent
    output=root/'analysis';output.mkdir(exist_ok=True);statusPath=root/'analysis-status.json'
    if statusPath.exists():raise ValueError('Analysis status already exists; refuse duplicate companion')
    state=dict(status='waiting',pid=os.getpid(),runs=[]);write(statusPath,state)
    reports=[]
    try:
        for job in plan['jobs']:
            while True:
                physical=json.loads((root/'status.json').read_text());row=next((x for x in physical['jobs'] if x['name']==job['name']),None)
                if row and row['status'] in ('passed','failed','skipped'):break
                if physical['phase'] in ('complete','complete with failures','stopped'):
                    row=dict(status='unavailable',reason='physical supervisor terminal without a result');break
                time.sleep(30)
            record=dict(name=job['name'],physicalStatus=row['status']);state['runs'].append(record)
            if row['status']!='passed':record.update(status='skipped',reason=row.get('reason',row.get('stopReason','physical job failed')));write(statusPath,state);continue
            for name,h in plan['sourceHashes'].items():
                if digest(Path(name))!=h:raise ValueError('Pinned analysis source changed: '+name)
            run=root/'runs'/job['name'];record['provenance']=verify_run(run,Path(plan['baseline']),job['seed'])
            directory=output/job['name'];logpath=output/(job['name']+'.log')
            state.update(status='analyzing',current=job['name']);record['status']='running';write(statusPath,state)
            command=[plan['python'],'-u',plan['driver'],str(run),'--output',str(directory),'--utility-directory',plan['utilities']]
            with logpath.open('w') as log:result=subprocess.run(command,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLCONFIGDIR='/tmp/warpsph-mpl'),stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:record.update(status='failed',reason='analysis subprocess failed',returncode=result.returncode);write(statusPath,state);continue
            report=json.loads((directory/'summary.json').read_text());record.update(status='completed',scalarScreenPassed=report['scalarScreenPassed'],video=video_audit(run,directory));reports.append(report);write(statusPath,state)
            print('ANALYZED',job['name'],flush=True)
        if reports:
            combined=ensemble(root,reports);report_markdown(root,combined,state,Path(plan['report']),Path(plan['roadmap']),Path(plan['plans']))
        state.update(status='completed' if len(reports)==3 else 'completed with failures',current=None,finished=datetime.datetime.now(datetime.timezone.utc).isoformat());write(statusPath,state)
    except Exception:
        state.update(status='error',error=traceback.format_exc());write(statusPath,state);raise


if __name__=='__main__':main()
