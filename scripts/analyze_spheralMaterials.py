"""CPU-only analysis of recorded BoxTension/gas-gas Sod comparisons."""
import argparse
import json
from pathlib import Path

import numpy as np
from compare_spheralMaterials import Measurements, save


def verify_recordings(root, output):
    import subprocess
    import urllib.request
    from warpsph_remote.daemon import descriptor
    from warpsph_remote.sources import HDF5Source
    info = descriptor()
    def get(path):
        request = urllib.request.Request(info['endpoint']+path, headers={'Authorization': 'Bearer '+info['token']})
        with urllib.request.urlopen(request, timeout=20) as response:
            return json.load(response)
    runs = {run['id']: run for run in get('/v1/runs')['runs']}
    checks = []
    for path in sorted(root.glob('*/*/comparisonRun.json')):
        summary = json.loads(path.read_text()); rid = summary['remoteRunID']; run = runs[rid]
        source = Path(summary.get('exportPath', str(path.parent)))
        history = get('/v1/runs/'+rid+'/history')
        times = HDF5Source(source/'trajectory.h5').times
        video = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,nb_frames', '-of', 'json', str(source/'output.mp4')]))['streams'][0]
        assert run['status'] == ('completed' if summary['passed'] else 'failed')
        assert history['dimension'] == summary['settings']['dim']
        # The runner's time-based trajectory cadence is independent of plotting.
        # Reference video renders every stored state; native Warp video uses
        # plotInterval. Verify each against its own source, not each other.
        assert history['frames'] == len(times)
        image_frames = len(list((source/'images').glob('frame_*.png')))
        assert int(video['nb_frames']) == image_frames and image_frames > 1
        if summary['solver']=='Spheral': assert image_frames == len(times)
        assert abs(float(history['times'][-1])-times[-1]) < 1.e-12
        # Native Warp trajectory timestamps are stored in its float32 dtype.
        expected_time = float(np.float32(summary['time'])) if summary['solver']=='warpSPH' else summary['time']
        assert abs(times[-1]-expected_time) < 1.e-9
        checks.append(dict(case=path.parent.parent.name, solver=summary['solver'], id=rid,
            status=run['status'], recordedFrames=history['frames'], plottedFrames=image_frames,
            finalTime=times[-1], video=video))
    save(output/'daemonVerification.json', checks)
    print('Verified', len(checks), 'daemon histories and videos', flush=True)


def analyze(root, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output.mkdir(parents=True, exist_ok=True)
    comparisons = []
    for case in sorted(root.iterdir()):
        if not case.is_dir() or not all((case/s/'comparisonRun.json').exists() for s in ('warp', 'spheral')):
            continue
        settings = json.loads((case/'settings.json').read_text())
        runs = {s: json.loads((case/s/'comparisonRun.json').read_text()) for s in ('warp', 'spheral')}
        initial = {s: dict(np.load(case/s/'initialState.npz')) for s in runs}
        final = {s: dict(np.load(case/s/'finalState.npz')) for s in runs}
        history = {s: json.loads((case/s/'diagnosticHistory.json').read_text()) for s in runs}
        for key in ('UIDs', 'materials', 'positions', 'velocities', 'masses', 'densities', 'internalEnergies'):
            np.testing.assert_array_equal(initial['warp'][key], initial['spheral'][key], err_msg=case.name+' '+key)
        checks = {}
        for s in runs:
            for key in ('UIDs', 'materials', 'masses'):
                np.testing.assert_array_equal(initial[s][key], final[s][key], err_msg=s+' '+key)
            checks[s] = dict(identitiesAndMassesUnchanged=True)
        differences = {}
        local_sod = {}
        for key in ('positions', 'velocities', 'densities', 'pressures', 'internalEnergies', 'supports'):
            delta = final['warp'][key]-final['spheral'][key]
            if key == 'positions':
                length = settings['domain'][1]-settings['domain'][0]
                delta -= length*np.round(delta/length)
            differences[key] = dict(maxAbsolute=float(abs(delta).max()), rms=float(np.sqrt(np.mean(delta**2))))
        if settings['case'] == 'box':
            fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout='constrained')
            for ax, s in zip(axes[0, :2], ('warp', 'spheral')):
                a = final[s]; image = ax.scatter(*a['positions'].T, c=a['densities'], s=2, vmin=1, vmax=4, cmap='viridis')
                ax.set(title=s+' density', xlim=(-.5, .5), ylim=(-.5, .5), aspect='equal')
            fig.colorbar(image, ax=axes[0, :2], label='Density', shrink=.7)
            for ax, key in zip([axes[0, 2], *axes[1]], ['pressureError', 'maxVelocity', 'shapeDisplacement', 'densityRelativeL1']):
                for s in runs:
                    ax.plot([r['time'] for r in history[s]], [r[key] for r in history[s]], label=s)
                ax.set(title=key, xlabel='Time'); ax.grid(alpha=.2)
            axes[0, 2].legend()
        else:
            fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout='constrained')
            exact = Measurements(settings, initial['spheral']).exact
            grid = np.linspace(-.499, .499, 4097)
            _, v, _, rho, p, _, _ = exact.solution(settings['end'], x=grid)
            # The observed coarse reservoir initially has nx/4 particles in .5.
            halfwidth = 3.*(2./settings['nx'])
            for s in runs:
                a = final[s]; x = a['positions'][:, 0]
                contact = exact.vm*runs[s]['time']; tail = -exact.vt*runs[s]['time']; shock = exact.vs*runs[s]['time']
                mask = abs(x-contact)<halfwidth
                left = (x>tail+halfwidth)&(x<contact-halfwidth)
                right = (x>contact+halfwidth)&(x<shock-halfwidth)
                ids = initial[s]['materials']
                last_left = np.flatnonzero(ids==0)[-1]; first_right = np.flatnonzero(ids==1)[0]
                local_sod[s] = dict(contactWindowHalfwidth=halfwidth, contactWindowParticles=int(mask.sum()),
                    contactPressureRelativeMax=float(np.max(abs(a['pressures'][mask]/exact.Pm-1))),
                    contactVelocityAbsoluteMax=float(np.max(abs(a['velocities'][mask, 0]-exact.vm))),
                    materialInterfaceMidpoint=float(.5*(x[last_left]+x[first_right])),
                    exactInterfacePosition=contact,
                    leftPlateauDensityMedian=float(np.median(a['densities'][left])), exactLeftDensity=exact.rhom1,
                    rightPlateauDensityMedian=float(np.median(a['densities'][right])), exactRightDensity=exact.rhom2,
                    pressurePlateauMedian=float(np.median(a['pressures'][left|right])), exactPressure=exact.Pm,
                    velocityPlateauMedian=float(np.median(a['velocities'][left|right, 0])), exactVelocity=exact.vm)
            cross_code = {}
            for axis, key, reference in zip(axes, ('densities', 'pressures', 'velocities'), (rho, p, v)):
                axis.plot(grid, reference, 'k-', lw=1, label='Exact gas/gas')
                interpolated = {}
                for s in runs:
                    a = final[s]; x = a['positions'][:, 0]; mask = abs(x)<.5
                    value = a[key] if key!='velocities' else a[key][:, 0]
                    order = np.argsort(x)
                    interpolated[s] = np.interp(grid, x[order], value[order])
                    axis.plot(x[mask], value[mask], '.', ms=2, label=s+f" t={runs[s]['time']:.4f}")
                cross_code[key] = float(np.mean(abs(interpolated['warp']-interpolated['spheral'])))
                axis.set(title=key, xlabel='x'); axis.grid(alpha=.2)
            differences['commonGridMeanAbsolute'] = cross_code
            axes[0].legend(fontsize=8)
        fig.suptitle(case.name+'; compiled Spheral and warpSPH, matched initial state, native support/AV differences')
        figure = output/(case.name+'.png'); fig.savefig(figure, dpi=170); plt.close(fig)
        row = dict(case=case.name, settings=settings, runs=runs, invariants=checks,
            finalDifferences=differences, localSodDiagnostics=local_sod,
            source=str(case.resolve()), figure=figure.name)
        comparisons.append(row)
        print(case.name, {s: dict(passed=runs[s]['passed'], final=runs[s]['final']) for s in runs}, flush=True)
    save(output/'comparison.json', dict(comparisons=comparisons,
        scope='Actual compiled Spheral trajectories in explicit matching harness; no hydro patches. Independent equilibrium/Riemann gates. Native support, viscosity, force-volume weights and compatible work differ; trajectory equality is not assumed. Common-grid Sod comparison uses linear interpolation for diagnostics only.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify-daemon', action='store_true')
    args = parser.parse_args(); analyze(args.input, args.output)
    if args.verify_daemon: verify_recordings(args.input, args.output)
