"""Sequential daemon-visible AV gradient controls; offline operator/trajectory analysis."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import numpy as np


def save(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def study(root, audit=False):
    root.mkdir(parents=True, exist_ok=False)
    reference = Path('results/materialEOS/blob_matrix_2026-10-20/spheral-fresh').resolve()
    original = Path('results/materialEOS/blob_paper_2026-10-20/qualify128rhs/diagnosticHistory.json').resolve()
    history = json.loads(original.read_text())
    if audit: history=history[:9]
    grid = np.diff([r['time'] for r in history])
    save(root/'grid.json', history)
    half = np.concatenate(([0.], np.cumsum(np.repeat(grid/2, 2))))
    save(root/'halfGrid.json', [dict(time=float(t)) for t in half])
    library = Path('results/materialEOS/'+('spheral-av-audit-policies' if audit else 'spheral-av-policies')+'/libSpheralPaperPolicies.so').resolve()
    provenance = dict(reference=str(reference), originalGrid=str(original),
        library=str(library), librarySHA256=hashlib.sha256(library.read_bytes()).hexdigest(),
        scope='Only AV gradient choice varies within each time grid. Fresh native RK2, paper overlay, raw predictor, entropy final partition, native initialization.')
    snapshot = root/'sources'; snapshot.mkdir()
    for name in ('benchmarks/eos/paperSpheral/paperPolicies.cc','scripts/run_spheralBlobExample.py',__file__):
        path = Path(name); shutil.copy2(path, snapshot/path.name)
    save(root/'provenance.json', provenance)
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='4', MPLCONFIGDIR='/tmp/warpsph-av-mpl')
    manifest = dict(status='running', pid=os.getpid(), stages=[])
    def execute(name, command, native=False):
        stage = dict(name=name, command=list(map(str, command)), status='running', log=str(root/(name+'.log')))
        manifest['stages'].append(stage); save(root/'study.json', manifest)
        print('START', name, flush=True); start = time.perf_counter()
        with Path(stage['log']).open('w') as output:
            child = subprocess.Popen(stage['command'], stdout=output, stderr=subprocess.STDOUT,
                env=dict(env, LD_PRELOAD=str(library)) if native else env)
            stage['pid'] = child.pid; save(root/'study.json', manifest)
            try:
                while child.poll() is None:
                    if (root/'STOP').exists(): child.send_signal(signal.SIGINT)
                    try: child.wait(timeout=5)
                    except subprocess.TimeoutExpired: pass
                    lines = Path(stage['log']).read_text(errors='replace').splitlines()
                    latest = next((s for s in reversed(lines) if s.startswith(('BLOB ', 'Remote run registered:', 'Traceback'))), '')
                    if latest: print(name+': '+latest[:220], flush=True)
                rc = child.wait()
            except BaseException:
                child.send_signal(signal.SIGINT); child.wait(); raise
        stage.update(status='completed' if rc==0 else 'failed', returncode=rc, wallSeconds=time.perf_counter()-start)
        save(root/'study.json', manifest)
        print('END', name, stage['status'], round(stage['wallSeconds'], 2), 's', flush=True)
        if (root/'STOP').exists() or (rc and 'SimulationStopped' in Path(stage['log']).read_text(errors='replace')):
            raise KeyboardInterrupt('Stopped by user')
        return rc
    try:
        for gridName in (('grid',) if audit else ('grid', 'halfGrid')):
            for mode in ('lagged', 'crk', 'sph'):
                name = gridName+'-'+mode; folder = root/name
                execute(name, ['results/materialEOS/spheral-build/install/.venv/bin/python', '-u',
                    'scripts/run_spheralBlobExample.py', 'run', '--paper', '--nx', '128', '--tkh',
                    str(grid.sum()/json.loads((reference/'settings.json').read_text())['tKH']) if audit else '1',
                    '--strict-positive', '--integrator', 'fresh', '--partition', 'entropy',
                    '--av-gradient', mode, '--time-grid', root/(gridName+'.json'), '--output', folder], True)
                if (folder/'trajectory.h5').exists():
                    execute(name+'-render', [sys.executable, 'scripts/run_spheralBlobExample.py', 'render', '--output', folder])
                # A setup error or fresh-gradient wiring failure is not evidence
                # about an AV model. Stop rather than replicate it six times.
                if not (folder/'summary.json').exists():
                    raise RuntimeError('Setup failure; see '+str(root/(name+'.log')))
        manifest['status'] = 'completed' if all(s['status']=='completed' for s in manifest['stages']) else 'partial'
    except BaseException as error:
        manifest.update(status='stopped' if isinstance(error, KeyboardInterrupt) else 'failed', error=repr(error))
        raise
    finally:
        save(root/'study.json', manifest)


def analyze(root, report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import h5py
    report.mkdir(parents=True, exist_ok=True)
    provenance = json.loads((root/'provenance.json').read_text())
    initial = dict(np.load(Path(provenance['reference'])/'initializedState.npz'))
    results = {}; histories = {}; finals = {}; operator = {}
    for grid in ('grid', 'halfGrid'):
        for mode in ('lagged', 'crk', 'sph'):
            name = grid+'-'+mode; folder = root/name
            if not (folder/'summary.json').exists(): continue
            row = json.loads((folder/'summary.json').read_text())
            a = dict(np.load(folder/('finalState.npz' if row['status']=='completed' else 'failedState.npz')))
            h = json.loads((folder/'diagnosticHistory.json').read_text()); histories[name] = h; finals[name] = a
            expectedTimes=np.array([r['time'] for r in json.loads((root/(grid+'.json')).read_text())])
            np.testing.assert_allclose([r['time'] for r in h],expectedTimes,rtol=0,atol=1.e-10)
            startup = dict(np.load(folder/'initializedState.npz'))
            row['initialMaximumDifferences']={key:float(np.abs(initial[key]-startup[key]).max()) for key in initial}
            for key in initial:
                if key in ('H','supports'):
                    # Independent native startup H relaxation has OpenMP
                    # reduction roundoff; keep and quantify it rather than
                    # claiming bitwise-identical support tensors.
                    np.testing.assert_allclose(initial[key],startup[key],rtol=2.e-14,atol=1.e-14,err_msg=name+' initial '+key)
                else:np.testing.assert_array_equal(initial[key],startup[key],err_msg=name+' initial '+key)
            row['maxima'] = {k: max(abs(r[k]) for r in h) for k in
                ('relativeEnergyDrift','relativeMomentumError','maxVelocity','materialMassError')}
            row['minEnergy'] = min(r['minEnergy'] for r in h)
            row['coldMass'] = h[-1]['cloudMassFraction'] if row['minEnergy']>0 else None
            row['msPerStep'] = 1000*row['wallSeconds']/max(row['steps'], 1)
            row['remoteID'] = json.loads((folder/'registration.json').read_text())['remoteRunID']
            row['configuration'] = json.loads((folder/'nativeConfiguration.json').read_text())
            assert row['configuration']['avGradient']==mode
            assert row['configuration']['rigorousVelocityGradient']==(mode=='sph')
            gradient = np.load(folder/'avGradientAudit.npz')['rows']
            # The original observer is before the predictor but after state0
            # restoration for final energy work. Only phase1 is stage-aligned.
            # The independent 8-step audit below fixes the observation point.
            aligned=gradient[gradient[:,0]==1]
            diff = aligned[:, 5:9]-aligned[:, 9:13]
            row['gradientVsSameRHS'] = dict(maximum=float(np.abs(diff).max()), RMS=float(np.sqrt(np.mean(diff**2))),
                phases=[1], provenance='Beginning RHS only; original phase2 rows follow RK2 state restoration and are not compared.')
            if mode=='crk' and np.abs(diff).max()>1.e-9: raise ValueError('Fresh CRK does not match native same-RHS gradient')
            affine = np.load(folder/'avAffineAudit.npz')['rows']
            data = {}
            for pattern, label in enumerate(('compression', 'shear')):
                b = affine[affine[:,0]==pattern]
                # Zero cached gradient means phi=0. Compare the compiled Q to
                # independently evaluated paper Eq69, with identical physical h.
                dot = b[:,7]*b[:,9]+b[:,8]*b[:,10]
                r2 = b[:,7]**2+b[:,8]**2
                expected = np.column_stack([b[:,11+s]*(-2*b[:,13+s]*np.minimum(0., b[:,5+s]*dot/(r2+.01*b[:,5+s]**2))+
                    np.minimum(0., b[:,5+s]*dot/(r2+.01*b[:,5+s]**2))**2) for s in (0,1)])
                np.testing.assert_allclose(b[:,15:17], expected, rtol=2.e-12, atol=1.e-12)
                data[label] = dict(pairs=len(b), crossMaterialPairs=int((b[:,1]!=b[:,2]).sum()),
                    rawQMaximum=float(b[:,15:17].max()), exactGradientQMaximum=float(b[:,17:19].max()),
                    crkGradientQMaximum=float(b[:,19:21].max()), gradientErrorMaximum=float(b[:,21:23].max()),
                    nativeVsPaperQMaximum=float(np.abs(b[:,15:17]-expected).max()))
            operator[name] = data
            movie = json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
                '-show_entries','stream=nb_frames','-of','json', str(folder/'output.mp4')]))
            row['videoFrames'] = int(movie['streams'][0]['nb_frames'])
            with h5py.File(folder/'trajectory.h5') as f: row['historyFrames'] = len(f['states'])
            results[name] = row
    for name, a in finals.items():
        baseline = finals[name.split('-')[0]+'-lagged']
        dx = a['positions']-baseline['positions']; L = np.array([40.,10.]); dx -= L*np.round(dx/L)
        cloud = initial['materials']==1
        results[name]['vsLagged'] = dict(allPositionRMS=float(np.sqrt(np.mean(dx**2))),
            cloudPositionRMS=float(np.sqrt(np.mean(dx[cloud]**2))), densityRMS=float(np.sqrt(np.mean((a['densities']-baseline['densities'])**2))))
        # Common spatial bins assess fields rather than tracking a particle's
        # identity through rearrangement. Mass and momentum per cell are summed.
        def field(state):
            bins=[np.linspace(0.,40.,161),np.linspace(0.,10.,41)]
            return np.histogram2d(*state['positions'].T, bins=bins, weights=state['masses'])[0]/.25**2
        f, b = field(a), field(baseline)
        results[name]['vsLagged']['binnedDensityL1'] = float(np.abs(f-b).mean())
    old = dict(np.load(Path(provenance['reference'])/'finalState.npz'))
    results['grid-lagged']['previousBaselineMaximumDifferences'] = {k:float(np.abs(finals['grid-lagged'][k]-old[k]).max())
        for k in ('positions','velocities','densities','internalEnergies')}
    sensitivity = {}
    for mode in ('lagged','crk','sph'):
        a,b = finals['grid-'+mode],finals['halfGrid-'+mode]
        dx=a['positions']-b['positions']; L=np.array([40.,10.]); dx-=L*np.round(dx/L)
        sensitivity[mode] = dict(positionRMS=float(np.sqrt(np.mean(dx**2))),
            coldMassChange=results['halfGrid-'+mode]['coldMass']-results['grid-'+mode]['coldMass'])
    stageAudit={}
    for mode in ('lagged','crk','sph'):
        folder=root/'stageAudit'/('grid-'+mode)
        a=np.load(folder/'avGradientAudit.npz')['rows']
        stageAudit[mode]={str(int(phase)):dict(maximum=float(np.abs((b:=a[a[:,0]==phase])[:,5:9]-b[:,9:13]).max()),
            RMS=float(np.sqrt(np.mean((b[:,5:9]-b[:,9:13])**2)))) for phase in sorted(set(a[:,0]))}
        if mode=='crk' and max(r['maximum'] for r in stageAudit[mode].values())>1.e-9:
            raise ValueError('Corrected fresh CRK audit failed')
        final=dict(np.load(folder/'finalState.npz'))
        with h5py.File(root/('grid-'+mode)/'trajectory.h5') as f:
            reference=f['states']['frame_0000008']
            parity={k:float(np.abs(final[k]-reference[k][()]).max()) for k in final}
        if max(parity.values())>1.e-10:raise ValueError('Observer revision changed physical trajectory')
        stageAudit[mode]['observerRevisionStateParity']=parity
    warpPath=Path('results/materialEOS/blob_matrix_2026-10-20/matched/warp').resolve()
    warp=dict(np.load(warpPath/'finalState.npz'));warpComparison={}
    for name,a in finals.items():
        np.testing.assert_array_equal(warp['UIDs'],a['UIDs'])
        dx=warp['positions']-a['positions'];L=np.array([40.,10.]);dx-=L*np.round(dx/L);cloud=a['materials']==1
        warpComparison[name]=dict(allPositionRMS=float(np.sqrt(np.mean(dx**2))),cloudPositionRMS=float(np.sqrt(np.mean(dx[cloud]**2))))
    save(report/'comparison.json', dict(provenance=provenance, rows=results, affineChecks=operator, timeGridSensitivity=sensitivity,
        existingWarpComparison=dict(path=str(warpPath),positions=warpComparison,
            scope='Existing matched working-tree Warp trajectory; no new Warp run. Other density/energy/support differences remain, so this is not an AV-only solver comparison.'),
        stageAudit=stageAudit, stageAuditProvenance='Read-only finalizeDerivatives hook before RK2 state restoration; separate 8-step daemon replays, state parity checked against original trajectories.'))
    fig, axes = plt.subplots(1, 3, figsize=(14,4), constrained_layout=True)
    for name,h in histories.items():
        for ax,key in zip(axes, ('cloudMassFraction','minEnergy','relativeEnergyDrift')):
            ax.plot([r['tKH'] for r in h], [r[key] for r in h], label=name, ls='--' if name.startswith('half') else '-')
    for ax,label in zip(axes, ('Cold mass fraction','Minimum u','Relative energy drift')):
        ax.set(xlabel='t / KH', ylabel=label); ax.grid(alpha=.2)
    axes[0].legend(fontsize=8); fig.savefig(report/'diagnostics.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(13,7), constrained_layout=True)
    for ax,(name,a) in zip(axes.flat, finals.items()):
        ax.scatter(*a['positions'].T, c=a['materials'], cmap='coolwarm', s=2, vmin=0, vmax=1)
        ax.set(xlim=(3.,12.), ylim=(1.,9.), aspect='equal', title=name)
    fig.savefig(report/'interfaces.png', dpi=180); plt.close(fig)
    fig,axes=plt.subplots(1,4,figsize=(15,4),constrained_layout=True)
    for ax,(name,a) in zip(axes,[(n,finals['grid-'+n])for n in ('lagged','crk','sph')]+[('existing Warp',warp)]):
        ax.scatter(*a['positions'].T,c=a['materials'],cmap='coolwarm',s=3,vmin=0,vmax=1)
        ax.set(xlim=(3.,12.),ylim=(1.,9.),aspect='equal',title=name)
    fig.savefig(report/'interfacesVsWarp.png',dpi=180);plt.close(fig)
    print(json.dumps(dict(rows={n:{k:r[k] for k in ('status','coldMass','minEnergy','wallSeconds','msPerStep','gradientVsSameRHS','vsLagged')} for n,r in results.items()},
                          affine=operator['grid-crk'], timeGridSensitivity=sensitivity), indent=2))


def verify(root, report):
    import urllib.request
    from warpsph_remote.daemon import descriptor
    from warpsph_remote.sources import HDF5Source
    info=descriptor()
    def get(path):
        request=urllib.request.Request(info['endpoint']+path,headers={'Authorization':'Bearer '+info['token']})
        with urllib.request.urlopen(request,timeout=15) as response:return json.load(response)
    runs={r['id']:r for r in get('/v1/runs')['runs']}
    verification={}
    for registration in sorted(root.glob('*/registration.json'))+sorted((root/'stageAudit').glob('*/registration.json')):
        folder=registration.parent;rid=json.loads(registration.read_text())['remoteRunID'];run=runs[rid]
        source=HDF5Source(folder/'trajectory.h5');history=get('/v1/runs/'+rid+'/history')
        np.testing.assert_allclose(history['times'],source.times,rtol=0,atol=1.e-12)
        movie=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
            '-show_entries','stream=nb_frames','-of','json',str(folder/'output.mp4')]))
        frames=int(movie['streams'][0]['nb_frames'])
        if frames!=len(source.times) or run['status']!='completed':raise ValueError((folder,frames,len(source.times),run['status']))
        verification[str(folder.relative_to(root))]=dict(id=rid,name=run['name'],status=run['status'],
            historyFrames=len(source.times),videoFrames=frames,historyMatches=True)
    if len(verification)!=9:raise ValueError('Expected six trajectories and three stage-audit replays')
    save(report/'recordingVerification.json',verification)
    print('Verified',len(verification),'completed daemon histories and videos')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('run','audit','analyze','verify')); parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--report',type=Path,default=Path('reports/materialEOS/spheral/avGradient'))
    args=parser.parse_args()
    if args.mode=='run':study(args.output.resolve())
    elif args.mode=='audit':study(args.output.resolve()/'stageAudit',audit=True)
    elif args.mode=='verify':verify(args.output.resolve(),args.report.resolve())
    else:analyze(args.output.resolve(),args.report.resolve())
