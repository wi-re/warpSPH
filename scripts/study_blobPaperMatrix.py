"""Short, sequential, daemon-visible CRK policy matrix with source provenance.

Main rows replay the qualified Spheral timestep grid. Numerical failures are
matrix observations; Remote Stop/STOP halts the queue. Production defaults
are unchanged. No full-resolution or overnight run is launched.
"""
import argparse
import copy
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import zipfile
import numpy as np
from compare_spheralMaterials import save,warp_run
from compare_spheralBlobPaper import context,initialize,PaperMeasures
from study_blobPartition import summary


def cases():
    rows=[]
    def add(name,description,solver=None,params=None,simulation=None,**kw):
        rows.append(dict(name=name,description=description,controls=dict(solver=solver or {},params=params or {},simulation=simulation or {}),**kw))
    add('matched','Successful working-tree policy selections; common reference timestep replay')
    add('repo-default','Default CRK density and AV-length policies on common paper geometry/kernel/Cl2/Cq1',
        {'crkDensityPolicy':'legacy','crkViscosityParams':{'viscosityKernelScale':0.}})
    for density,length,energy in itertools.product(('legacy','massReplacement'),(0.,4.),('native','entropy')):
        if energy=='native' and (density,length) in (('legacy',0.),('massReplacement',4.)):continue
        add(f'd-{density}_q-{int(length)}_e-{energy}','Core density × AV-length × final-partition factorial',
            {'crkDensityPolicy':density,'crkEnergyPartition':energy,'crkViscosityParams':{'viscosityKernelScale':length}})
    add('same-material','Same-material raw-kernel density only',{'crkDensityPolicy':'sameMaterial'})
    add('node-dudt','Current Spheral node-dudt final partition only',{'crkEnergyPartition':'spheral'})
    add('entropy-compatible','Final-work entropy partition plus compatible predictor',{'crkEnergyPartition':'entropy','crkEnergyPredictor':'compatible'},parent='d-massReplacement_q-4_e-entropy')
    add('dudt-compatible','Node-dudt partition plus compatible predictor',{'crkEnergyPartition':'spheral','crkEnergyPredictor':'compatible'},parent='node-dudt')
    add('native-compatible','Native fractions with compatible predictor only',{'crkEnergyPredictor':'compatible'})
    add('one-sided','One-sided raw viscous heating only',{'crkViscosityParams':{'oneSidedWork':True}})
    add('mean-volume','Arithmetic-mean squared pair volumes only',{'crkViscosityParams':{'meanVolumeWeights':True}})
    add('cached-gradient','Previous-RHS Q gradient only; current initial gradient seeds cache',{'crkViscosityGradient':'cached'})
    add('paper-density','Eq76 once/step + continuity stages, geometric volume every RHS',{'crkEvolution':'paper'})
    add('paper-density-entropy','Paper density timing plus final-work entropy; fresh Warp RHS/supports retained',{'crkEvolution':'paper','crkEnergyPartition':'entropy'},parent='paper-density')
    add('spheral-density-volume','Coupled CURRENT Spheral m/rho volume + same-material density schedule',{'crkEvolution':'spheral','crkDensityPolicy':'sameMaterial'},coupled=True)
    add('fixed-support','Freeze the common initialized supports; diagnostic control, not Spheral support emulation',params={'adaptiveSupportScheme':'NoScheme'})
    add('float32','Single-precision inputs/dynamics only',precision='float32')
    add('half-grid','Bisect every reference timestep',halfGrid=True)
    add('generated-H','Unrelaxed generated H rather than common relaxed H',generatedH=True)
    add('unit-Q','Cl1/Cq1 instead of Cl2/Cq1',{'diffusionParams':{'C_l':1.,'C_q':1.}})
    add('quintic','Quintic basis at identical compact radii/neighbors and Q length; not the bundled preset',simulation={'kernel':'QuinticSpline'})
    rows.append(dict(name='small-fixed-dt',description='Previously completed fixed dt<=.001 matched baseline',reuse=True))
    return rows


def prepare(args):
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    short=args.short.resolve();base=json.loads((short/'settings.json').read_text());ref=Path(base['reference'])
    history=json.loads((ref/'diagnosticHistory.json').read_text());grid=np.diff([r['time'] for r in history]);assert (grid>0).all()
    initial=dict(np.load(short/'input.npz'));rows=cases()
    for row in rows:
        if row.get('reuse'):row['folder']=str(short);continue
        folder=root/row['name'];folder.mkdir();a={k:v.copy() for k,v in initial.items()}
        if row.get('generatedH'):a['supports']=np.load(ref/'generatedState.npz')['supports']
        precision=row.get('precision','float64')
        if precision=='float32':a={k:v.astype(np.float32) if v.dtype.kind=='f' else v for k,v in a.items()}
        np.savez_compressed(folder/'input.npz',**a)
        dt=np.repeat(grid/2,2) if row.get('halfGrid') else grid
        settings=copy.deepcopy(base);settings.update(kernel='B8',referenceKernel='BSpline7',gammaVariant=row['name'],precision=precision,
            matrixControls=row['controls'],steps=len(dt),dt=float(dt[0]),end=float(dt.sum()),timeGrid=dt.tolist(),frameEvery=10,
            comparisonPolicy=row['description'],source='Working-tree CRK comparison matrix; exact common native startup, only declared row controls change.')
        save(folder/'settings.json',settings);row['folder']=str(folder)
        row['inputArraysSHA256']={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in a.items()}
    paths=set()
    for path in ('src/warpSPH/modules/crk','src/warpSPH/modules/compSPH','src/warpSPH/modules/eos','src/warpSPH/modules/dissipation/pi'):
        paths.update(Path(path).glob('*.py'))
    paths.update(map(Path,('src/warpSPH/systems/compSPH.py','src/warpSPH/configurations/crkSPH.py','src/warpSPH/runner/configuration.py',
        'src/warpSPH/schemes/compressible/crkSPH.py','scripts/compare_spheralBlobPaper.py','scripts/compare_spheralMaterials.py',
        'scripts/study_blobPaperMatrix.py','scripts/run_spheralBlobExample.py','benchmarks/eos/paperSpheral/paperPolicies.cc')))
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    with zipfile.ZipFile(root/'sourceSnapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(paths):z.write(p,str(p))
    patch=subprocess.check_output(['git','diff','--',*[str(p) for p in sorted(paths)]])
    (root/'workingTree.patch').write_bytes(patch)
    save(root/'provenance.json',dict(baseCommit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),sourceSHA256=hashes,
        originalMatchedRun=str(short),reference=str(ref),referenceTimeGrid=grid.tolist(),rows=rows,
        scope='Working-tree policies, not a pristine-HEAD build; source snapshot includes experimental controls and unchanged native paths. No closure solve applied.'))
    print('Prepared',len(rows),'Warp rows (one reused), plus native fresh-RK2 control',flush=True)


def run_row(args):
    settings=json.loads((args.output/'settings.json').read_text())
    return warp_run(args,settings,context,PaperMeasures,summary,'Blob matrix',initialize_factory=initialize)


def analyze(root,report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    provenance=json.loads((root/'provenance.json').read_text());ref=Path(provenance['reference'])
    hS=json.loads((ref/'diagnosticHistory.json').read_text());end=hS[-1]['time'];refFinal=dict(np.load(ref/'finalState.npz'))
    results=[];histories={};report.mkdir(parents=True,exist_ok=True)
    allRows=provenance['rows']+[dict(name='spheral-fresh',folder=str(root/'spheral-fresh'),kind='Spheral',description='Fresh native RK2; same recorded timestep grid/raw predictor/final entropy work')]
    if (root/'extraRuns.json').exists():allRows+=json.loads((root/'extraRuns.json').read_text())
    for row in allRows:
        folder=Path(row['folder']);native=row.get('kind')=='Spheral';data=folder if native else folder/'warp'
        if not (data/'diagnosticHistory.json').exists():continue
        h=json.loads((data/'diagnosticHistory.json').read_text());histories[row['name']]=h
        run=json.loads((data/('summary.json' if native else 'comparisonRun.json')).read_text()) if (data/('summary.json' if native else 'comparisonRun.json')).exists() else json.loads((data/'failure.json').read_text())
        tolerance=1e-5 if row.get('precision')=='float32' else 1e-8
        reached=abs(h[-1]['time']-end)<tolerance;positive=all(x['minEnergy'] is not None and x['minEnergy']>0 and x['minDensity'] is not None and x['minDensity']>0 for x in h)
        passed=reached and positive and (run.get('status')=='completed' if native else run.get('passed',False))
        t=np.array([r['time'] for r in h]);delta=np.array([r['cloudMassFraction'] for r in h])-np.interp(t,[r['time'] for r in hS],[r['cloudMassFraction'] for r in hS]);admissible=np.array([r['minEnergy'] is not None and r['minEnergy']>0 and r['relativeEnergyDrift'] is not None for r in h]);mask=(t>=.01*hS[-1]['time'])&admissible
        record=dict(name=row['name'],description=row['description'],controls=row.get('controls'),parent=row.get('parent','matched'),
            passed=bool(passed),reachedEnd=reached,endTimeTolerance=tolerance,timeKH=h[-1]['tKH'],coldMass=h[-1]['cloudMassFraction'] if admissible[-1] else None,
            coldCurveRMS=float(np.sqrt(np.mean(delta[mask]**2))) if mask.any() else None,
            minU=min(r['minEnergy'] for r in h if r['minEnergy'] is not None),minRho=min(r['minDensity'] for r in h if r['minDensity'] is not None),
            energyDrift=max(abs(r['relativeEnergyDrift']) for r in h if r['relativeEnergyDrift'] is not None),
            maxV=max(r['maxVelocity'] for r in h if r['maxVelocity'] is not None),wallSeconds=run['wallSeconds'],steps=run['steps'],
            millisecondsPerStep=1000*run['wallSeconds']/max(run['steps'],1),remoteID=(json.loads((data/'registration.json').read_text()))['remoteRunID'],
            exportPath=str(data) if native else run.get('exportPath',str(next(data.glob('blob_*')))) if list(data.glob('blob_*')) else None,reused=bool(row.get('reuse')),
            termination=run.get('failure',run.get('error')))
        # A recording exception is not an inadmissible fluid state.
        record['classification']='qualified' if passed else 'recording error' if 'crkMassDensities' in str(record['termination']) else 'numerical failure' if not positive else 'incomplete'
        final=data/('finalState.npz' if (data/'finalState.npz').exists() else 'failedState.npz')
        if reached and final.exists():
            a=np.load(final);dx=a['positions']-refFinal['positions'];L=np.array([40.,10.]);dx-=L*np.round(dx/L)
            record['positionRMS']=float(np.sqrt(np.mean(dx*dx)));record['densityRMS']=float(np.sqrt(np.mean((a['densities']-refFinal['densities'])**2)))
        if not native and record['exportPath']:
            resolved=json.loads((Path(record['exportPath'])/'resolvedConfiguration.json').read_text());record['resolvedConfiguration']=resolved
        results.append(record)
    save(report/'matrix.json',dict(provenance=provenance,rows=results,referenceFinal=hS[-1],
        scope='Controlled comparisons plus stated coupled rows; finite/admissible is not literature agreement. Failures remain observations. No full-resolution run.'))
    columns=('name','passed','timeKH','coldMass','coldCurveRMS','minU','energyDrift','positionRMS','wallSeconds','millisecondsPerStep')
    with (report/'matrix.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(results)
    lines=['# Short blob comparison matrix','',f'{len(results)} observed rows; base commit `{provenance["baseCommit"][:8]}` with snapshotted working-tree changes.','',
        'All main rows use the same4380 particles and recorded Spheral timestep grid to1KH. Full512 run remains deferred. See matrix.json for controls/provenance and per-row resolved configurations.','',
        '| Row | Qualified | Final KH | Cold mass | Curve RMS | Min u | Energy drift | Position RMS | Wall s | ms/step |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in results:
        number=lambda k:f'{r[k]:.4g}' if r.get(k) is not None else '—'
        lines.append('| '+' | '.join([r['name'],'yes' if r['passed'] else 'no',*[number(k) for k in columns[2:]]])+' |')
    lines+=['','Qualification means completion/positive state/conservation, not pointwise or literature agreement. Paired controls and exact settings are in matrix.json. Numerical failures retain terminal states; earlier final times are not compared as full-time solutions.','',
        'The prior successful Warp was working-tree native summation/Butcher energy plus experimental massReplacement density and R/4 Q length. Repo-default row means default density/Q-length policies at the common paper physical parameters, not pristine HEAD or the bundled blob preset.','',
        'Native force/energy defaults remain unchanged. New paper-density scheduling and independent midpoint energy combinations are explicitly experimental. Cached-gradient first call seeds from its current gradient; it is a timing control, not an exact reconstruction of native initial Q state. Frozen supports and quintic-at-fixed-radius rows are diagnostic controls.','']
    (report/'MATRIX.md').write_text('\n'.join(lines))
    if histories:
        fig,axes=plt.subplots(2,2,figsize=(14,8),layout='constrained')
        for name,h in [('Spheral reference',hS),*histories.items()]:
            for ax,field in zip(axes.flat,('cloudMassFraction','minEnergy','relativeEnergyDrift','maxVelocity')):
                values=[r[field] if field!='cloudMassFraction' or (r['minEnergy'] is not None and r['minEnergy']>0 and r['relativeEnergyDrift'] is not None) else np.nan for r in h]
                ax.plot([r['tKH'] for r in h],values,label=name,lw=1,alpha=.8)
        for ax,title in zip(axes.flat,('Cold cloud mass','Minimum energy','Relative energy change','Maximum speed')):ax.set(xlabel='t/KH',title=title);ax.grid(alpha=.2)
        axes[1,0].set_yscale('symlog',linthresh=1e-12);axes[0,1].axhline(0,color='k',lw=.5)
        axes[0,0].legend(fontsize=6,ncol=2);fig.savefig(report/'allDiagnostics.png',dpi=170);plt.close(fig)
    print('Updated matrix:',len(results),'observed rows;',sum(r['passed'] for r in results),'qualified',flush=True)


def study(args):
    root=args.output.resolve();report=args.report.resolve();prov=json.loads((root/'provenance.json').read_text());repo=Path.cwd()
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',MPLCONFIGDIR='/tmp/warpsph-material-mpl')
    manifest=dict(status='running',pid=os.getpid(),stages=[]);save(root/'study.json',manifest)
    def execute(name,command,extra=None):
        stage=dict(name=name,status='running',command=list(map(str,command)),log=str(root/(name+'.log')));manifest['stages'].append(stage)
        print('Starting',name,flush=True);start=time.perf_counter()
        with Path(stage['log']).open('w') as log:
            with subprocess.Popen(stage['command'],stdout=log,stderr=subprocess.STDOUT,env=extra or env,cwd=repo) as child:
                stage['pid']=child.pid;save(root/'study.json',manifest)
                while child.poll() is None:
                    if (root/'STOP').exists():child.send_signal(signal.SIGINT)
                    try:child.wait(timeout=5)
                    except subprocess.TimeoutExpired:pass
                rc=child.returncode
        stage.update(status='completed' if rc==0 else 'failed',returncode=rc,wallSeconds=time.perf_counter()-start);save(root/'study.json',manifest)
        print('Finished',name,rc,round(stage['wallSeconds'],2),'s',flush=True)
        if (root/'STOP').exists() or (rc and 'SimulationStopped' in Path(stage['log']).read_text(errors='replace')):raise KeyboardInterrupt('Matrix stopped by user')
        return rc
    try:
        # Fresh native predictor control, same time grid and native input/startup.
        library=(repo/'results/materialEOS/spheral-paper-policies/libSpheralPaperPolicies.so').resolve()
        folder=root/'spheral-fresh';native=repo/'results/materialEOS/spheral-build/install/.venv/bin/python'
        if not folder.exists():
            execute('spheral-fresh',[native,'-u','scripts/run_spheralBlobExample.py','run','--paper','--nx',128,'--tkh',1,'--strict-positive',
                '--integrator','fresh','--time-grid',str(Path(prov['reference'])/'diagnosticHistory.json'),'--output',str(folder)],dict(env,LD_PRELOAD=str(library)))
            shutil.copyfile(root/'spheral-fresh.log',folder/'runtime.log')
            if (folder/'trajectory.h5').exists():execute('spheral-fresh-render',[sys.executable,'scripts/run_spheralBlobExample.py','render','--output',folder])
        for row in prov['rows']:
            folder=Path(row['folder'])
            if row.get('reuse') or row.get('setupOnly'):continue
            if (folder/'warp/comparisonRun.json').exists() or (folder/'warp/failure.json').exists():continue
            execute(row['name'],[sys.executable,'-u',__file__,'row','--output',folder])
            execute('analyze-'+row['name'],[sys.executable,__file__,'analyze','--output',root,'--report',report])
        execute('final-analyze',[sys.executable,__file__,'analyze','--output',root,'--report',report])
        manifest['status']='completed'
    except BaseException as error:
        manifest.update(status='stopped' if isinstance(error,KeyboardInterrupt) else 'failed',error=repr(error));raise
    finally:save(root/'study.json',manifest)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('prepare','row','study','analyze'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--short',type=Path,default=Path('results/materialEOS/blob_paper_short_2026-10-20'))
    p.add_argument('--report',type=Path,default=Path('reports/materialEOS/spheral/blobMatrix'))
    args=p.parse_args()
    if args.mode=='prepare':prepare(args)
    elif args.mode=='row':raise SystemExit(run_row(args))
    elif args.mode=='study':study(args)
    else:analyze(args.output,args.report)
