"""Offline comparison of the standard hydrostatic density-policy A/B."""
import argparse
import json
from pathlib import Path
import subprocess

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--daemon-check', action='store_true')
    args = parser.parse_args()
    args.report.mkdir(parents=True, exist_ok=True)
    rows, states, histories, solvers, initial = {}, {}, {}, {}, {}
    for policy in ('legacy', 'massReplacement'):
        directory = args.root/('hydrostatic-standard-'+policy)
        matches = list(directory.glob('*/materialCaseValidation.json'))
        if len(matches) != 1:
            raise ValueError(f'Expected one completed recording: {directory}')
        path = matches[0].parent
        row = json.loads(matches[0].read_text())
        history = json.loads((path/'diagnosticHistory.json').read_text())
        first = dict(np.load(directory/'initialState.npz'))
        final = dict(np.load(path/'finalState.npz'))
        rows[policy] = row
        states[policy], histories[policy], initial[policy] = final, history, first
        solvers[policy] = json.loads((directory/'resolvedSolver.json').read_text())
        row['maxima'] = {k: max(abs(h[k]) for h in history) for k in
            ('maxVelocity', 'pressureError', 'shapeDisplacement', 'relativeEnergyDrift', 'densityRelativeL1')}
        row['maximumDisplacementInSpacings'] = row['maxima']['shapeDisplacement']*row['nx']
        row['initialDensityRelativeL1'] = float(np.mean(np.abs(first['densities']-
            first['masses']*row['nx']**2)/(first['masses']*row['nx']**2)))
        movie = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=nb_frames', '-of', 'json', str(path/'output.mp4')]))
        row['videoFrames'] = int(movie['streams'][0]['nb_frames'])
        trajectories = list(path.glob('*.hdf5')) + list(path.glob('*.h5'))
        row['recordings'] = []
        for recording in trajectories:
            with h5py.File(recording) as f:
                if 'times' not in f:
                    continue
                times = [float(np.asarray(f['times'][k][()]).item()) for k in f['times']]
                row['recordings'].append(dict(path=str(recording.resolve()), frames=len(times),
                                               firstTime=min(times), lastTime=max(times)))
        if not row['recordings'] or row['videoFrames'] < 2:
            raise ValueError('Missing trajectory or video')
    solverDifferences = {k: [solvers['legacy'][k], solvers['massReplacement'][k]]
                         for k in solvers['legacy'] if solvers['legacy'][k] != solvers['massReplacement'][k]}
    if set(solverDifferences) != {'crkDensityPolicy'}:
        raise ValueError(f'Unexpected solver differences: {solverDifferences}')
    initialComparison = {k: dict(equal=bool(np.array_equal(initial['legacy'][k], initial['massReplacement'][k])),
        maxAbsoluteDifference=float(np.max(np.abs(initial['legacy'][k].astype(float)-initial['massReplacement'][k]))))
        for k in initial['legacy']}
    for k in ('positions', 'supports', 'masses', 'materials', 'velocities', 'UIDs'):
        if not initialComparison[k]['equal']:
            raise ValueError(f'Initial geometry/mass differs: {k}')
    pressureIfUnchangedEnergy = (5/3-1)*initial['massReplacement']['densities']*initial['legacy']['internalEnergies']
    result = dict(runs=rows, solverDifferences=solverDifferences, initialComparison=initialComparison,
        initialPressureIfLegacyEnergyRetained=dict(minimum=float(pressureIfUnchangedEnergy.min()),
            maximum=float(pressureIfUnchangedEnergy.max()),
            maximumError=float(np.abs(pressureIfUnchangedEnergy-1).max()),
            provenance='Algebraic EOS check on saved initial arrays; no third trajectory.'))
    if args.daemon_check:
        import urllib.request
        from warpsph_remote.daemon import descriptor
        info = descriptor()
        def get(path):
            request = urllib.request.Request(info['endpoint']+path,
                headers={'Authorization': 'Bearer '+info['token']})
            with urllib.request.urlopen(request, timeout=15) as response:
                return json.load(response)
        remote = {r['id']: r for r in get('/v1/runs')['runs']}
        verification = {}
        for policy, row in rows.items():
            run = remote[row['remoteRunID']]
            history = get('/v1/runs/'+row['remoteRunID']+'/history')
            recording = row['recordings'][0]
            with h5py.File(recording['path']) as f:
                times = sorted(float(np.asarray(f['times'][k][()]).item()) for k in f['times'])
            np.testing.assert_allclose(history['times'], times, rtol=0, atol=1.e-12)
            if run['status'] != 'completed': raise ValueError(run['status'])
            verification[policy] = dict(id=run['id'], name=run['name'], status=run['status'],
                historyFrames=len(times), historyMatches=True, videoFrames=row['videoFrames'])
        result['daemonVerification'] = verification
    (args.report/'comparison.json').write_text(json.dumps(result, indent=2)+'\n')
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for policy, color in [('legacy', 'tab:blue'), ('massReplacement', 'tab:orange')]:
        history = histories[policy]
        t = [h.get('t', h.get('time')) for h in history]
        for ax, key, label in zip(axes[0],
                ('maxVelocity', 'pressureError', 'shapeDisplacement'),
                ('Maximum speed', 'Maximum |P − 1|', 'Maximum displacement / Δx')):
            values = np.array([h[key] for h in history])
            if key == 'shapeDisplacement': values *= rows[policy]['nx']
            ax.semilogy(t, np.maximum(values, 1.e-12), label=policy, color=color)
            ax.set(xlabel='Time', ylabel=label)
            ax.grid(alpha=.2)
        a = initial[policy]
        mask = np.abs(a['positions'][:, 1]-.5/rows[policy]['nx']) < 1.e-5
        order = np.argsort(a['positions'][mask, 0])
        axes[1, 0].plot(a['positions'][mask, 0][order], a['densities'][mask][order],
                        label=policy, color=color)
    axes[0, 0].legend()
    axes[1, 0].set(xlabel='x at y ≈ 0', ylabel='Initial density')
    axes[1, 0].legend()
    for ax, policy in zip(axes[1, 1:], ('legacy', 'massReplacement')):
        a = states[policy]
        mask = (np.abs(a['positions']) < .32).all(axis=1)
        ax.scatter(*a['positions'][mask].T, c=a['materials'][mask], cmap='coolwarm',
                   vmin=0, vmax=1, s=1, rasterized=True)
        ax.set(title=f'{policy}, t={rows[policy]["time"]:.4f}', xlabel='x', ylabel='y',
               xlim=(-.32, .32), ylim=(-.32, .32), aspect='equal')
    fig.savefig(args.report/'comparison.png', dpi=180)
    plt.close(fig)
    lines = ['# Standard hydrostatic density-policy comparison', '',
        f'Both runs completed to t={rows["legacy"]["time"]:.6f}. The historical degradation '
        'with cross-material mass replacement was not reproduced in this control.', '',
        'Standard runner case: 200² equal-spaced particles, periodic unit square, '
        'density 2 in the central square and 1 outside, initial P=1, gamma=5/3 in both materials. '
        'B7, nh=4, Owen support evolution, midpoint/Butcher native compatible energy, '
        'default CRK force and viscosity, CFL=.3, float32, eager execution. '
        'Only the density-policy solver setting differs; no Q-length, work partition, '
        'volume weighting or closure override.', '',
        '| Quantity | legacy | massReplacement |', '|---|---:|---:|']
    for label, key in [('Peak speed', 'maxVelocity'), ('Peak pressure error (P=1)', 'pressureError'),
                       ('Peak relative energy drift', 'relativeEnergyDrift'),
                       ('Peak density relative L1', 'densityRelativeL1')]:
        lines.append(f'| {label} | {rows["legacy"]["maxima"][key]:.6g} | {rows["massReplacement"]["maxima"][key]:.6g} |')
    for label, key in [('Maximum displacement / spacing', 'maximumDisplacementInSpacings'),
                       ('Initial density relative L1', 'initialDensityRelativeL1'),
                       ('Steps', 'steps'), ('Runner wall seconds', 'wallSeconds'),
                       ('Mean step milliseconds', 'meanStepMilliseconds')]:
        lines.append(f'| {label} | {rows["legacy"][key]:.6g} | {rows["massReplacement"][key]:.6g} |')
    lines += ['', 'Both remain essentially stationary; neither develops visible square deformation. '
        'The replacement policy retains a much sharper density contact. '
        'This supports this static, equal-spacing test only; it does not qualify '
        'unequal-spacing contacts or explain the dynamic blob discrepancy.', '',
        'Initialization matters: saved positions, supports, masses, labels, UIDs and velocities '
        'are bitwise identical, and both start at exactly P=1. The standard sampler initializes '
        'internal energy from the selected density rule, so initial rho/u differ. '
        f'Keeping legacy u while replacing rho would instead give initial pressure '
        f'{pressureIfUnchangedEnergy.min():.6f}–{pressureIfUnchangedEnergy.max():.6f}, '
        'already out of equilibrium. This is an algebraic check, not a third run or '
        'an established explanation of the historical failure.', '',
        '![Diagnostics and particle distributions](comparison.png)', '',
        'Both runs have daemon registrations, stored trajectories, and vispy videos; '
        'the comparison JSON records paths, frame counts, full diagnostic maxima '
        'and the exact solver/initial-state comparison. Daemon verification, when enabled, '
        'checks completion and every recorded history time against the HDF5 trajectory.', '',
        f'Raw campaign: `{args.root.resolve()}`. '
        'The full-duration campaign also retains source hashes, source snapshots and a tracked-source patch.', '',
        'Reproduce the physical pair:', '', '```bash',
        'python scripts/study_materialCases.py --phase hydrostaticPolicy --output <fresh-output-directory>',
        '```', '', 'Analyze without rerunning:', '', '```bash',
        f'python scripts/analyze_hydrostaticDensityPolicies.py {args.root} --report {args.report} --daemon-check',
        '```', '']
    (args.report/'README.md').write_text('\n'.join(lines))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
