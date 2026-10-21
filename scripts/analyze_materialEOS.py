"""Compare completed RM study runs without importing the simulation runtime.

python scripts/analyze_materialEOS.py results/materialEOS/study_* --output /tmp/rm-comparison
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np


def compare(roots, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import runpy
    reference = runpy.run_path(str(Path(__file__).resolve().parents[1] /
        'src/warpSPH/caseUtils/compressible/richtmyerMeshkov/reference.py'))['weakShockReference']
    output.mkdir(parents=True, exist_ok=True)
    paths = sorted({p for root in roots for p in root.rglob('materialValidation.json')})
    if not paths:
        raise ValueError('No completed materialValidation.json files found')
    figure, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
    rows = []
    for path in paths:
        summary = json.loads(path.read_text())
        history = json.loads((path.parent/'diagnosticHistory.json').read_text())
        name = path.parents[1].name
        a0 = summary['initialAmplitude']
        if summary['mode'] != 'rm' or 'circulationUpper' not in history[0]:
            continue
        # The campaign uses the same two-gamma ICs and wavelength in every run.
        # Refuse silently normalizing unrelated data with these constants.
        if not math.isclose(summary['shock']['speed'], 1.4198591479439078, rel_tol=1e-10):
            raise ValueError(f'{path}: this comparison expects the campaign Mach-1.2 shock')
        ref = reference(1, 3, 1, 1.4, 5/3, 1.2, a0)
        times = np.asarray([r['t'] for r in history])
        arrival = 1/summary['shock']['speed']
        elapsed = times-arrival
        amplitude = np.asarray([r['interfaceCosAmplitude'] for r in history])
        measuredTimes, unique = np.unique([r['circulationTime'] for r in history], return_index=True)
        sample = [history[i] for i in unique]
        circ = np.asarray([r['circulationUpper'] for r in sample])
        group = 0 if a0 < .05 else 1
        axes[group, 0].plot(elapsed, 2*np.pi*amplitude, label=name)
        axes[group, 1].plot(measuredTimes-arrival, circ/ref['circulationUpper'], label=name)
        axes[group, 2].semilogy(measuredTimes-arrival, np.maximum([r['transverseKineticEnergy'] for r in sample], 1e-20), label=name)
        axes[group, 3].plot(measuredTimes-arrival, [r['interactionAlphaMean'] for r in sample], label=name)
        fit = (times >= 1) & (times <= 1.8)
        growth = float(np.polyfit(times[fit], amplitude[fit], 1)[0]) if fit.sum() >= 3 else None
        # Derive directional extents from the raw coordinates, independent of
        # historical labels: light material 0 penetrates toward +x (bubble).
        final = history[-1]
        rows.append(dict(name=name, source=str(path.resolve()), nx=summary['nx'],
            densityPolicy=summary.get('densityPolicy', 'legacy'),
            switch=summary.get('viscositySwitch'), time=summary['time'],
            steps=summary['steps'], initialAmplitude=a0, finalAmplitude=summary['finalAmplitude'],
            growthRate=growth, growthOverWeakShearReference=growth/ref['growthRate'] if growth is not None else None,
            finalCirculationTime=measuredTimes[-1], finalCirculationUpper=float(circ[-1]),
            circulationOverWeakShearReference=float(circ[-1]/ref['circulationUpper']),
            finalTransverseKineticEnergy=sample[-1]['transverseKineticEnergy'],
            markerLayerWidth=final['markerLayerWidth'],
            bubbleExtent=final['interfaceMax']-final['interfaceMean'],
            spikeExtent=final['interfaceMean']-final['interfaceMin'],
            initialMarkerBandContribution=summary['initialMarkerLayerWidth']-2*a0,
            relativeEnergyDrift=summary['relativeEnergyDrift'],
            wallSeconds=summary['wallSeconds'], meanStepMilliseconds=summary['meanStepMilliseconds'],
            stoppedEarly=summary['stoppedEarly'], stopReason=summary['stopReason'], reference=ref))
    for group in range(2):
        axes[group, 1].axhline(1, color='black', linestyle='--', linewidth=1, label='weak-shock shear estimate')
        for index, title in enumerate(('Fourier marker amplitude\n(k × amplitude)',
                                       'Half-wave circulation\n/ weak-shock estimate',
                                       'Transverse kinetic energy',
                                       'Mean viscosity alpha\nin interaction window')):
            ax = axes[group, index]
            ax.set(title=title,
                   xlabel='Time since nominal shock impact')
            ax.grid(alpha=.25)
        axes[group, 0].set_ylabel('Initial amplitude / wavelength ≈ '+('.01' if group == 0 else '.1'))
        axes[group, 0].legend(fontsize=7)
    figure.suptitle('RM study — weak-shock normalization is approximate, not an exact error target', fontsize=11)
    figure.savefig(output/'comparison.png', dpi=160)
    plt.close(figure)
    result = dict(referenceLimit='Weak-shock irrotational model only; ratios are not errors against an exact solution.',
        circulationEstimator='Corrected SPH Jacobian, physical volumes m/rho; moving x window extending two max supports beyond marker band.',
        markerLimit='Two adjacent initial particle rows, finite O(dx) band width; not a molecular mixing measure.', runs=rows)
    (output/'comparison.json').write_text(json.dumps(result, indent=2)+'\n')
    for row in rows:
        print(f'{row["name"]}: a={row["finalAmplitude"]:.7g}, growth={row["growthRate"]:.7g}, '
              f'Gamma={row["finalCirculationUpper"]:.7g}, energy={row["relativeEnergyDrift"]:.3g}, '
              f'{row["wallSeconds"]:.2f}s, {row["meanStepMilliseconds"]:.3f}ms/step')
    print(output/'comparison.png')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('roots', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    compare(args.roots, args.output)
