"""CPU-only report and scientific figures for completed material controls."""
import argparse
import json
from pathlib import Path
import numpy as np


def analyze(roots, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted({p for root in roots for p in root.rglob('materialCaseValidation.json')}):
        row = json.loads(path.read_text())
        row['source'] = str(path.resolve())
        data = np.load(path.parent/'finalState.npz')
        pos, ids = data['positions'], data['materials']
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
        if row['case'] == 'twoGasSod':
            order = np.argsort(pos[:, 0])
            mask = np.abs(pos[order, 0]) < .5
            x = pos[order, 0][mask]
            for ax, name, exact in zip(axes, ('densities', 'pressures', 'velocities'), ('density', 'pressure', 'velocity')):
                value = data[name][order][mask]
                if value.ndim > 1:
                    value = value[:, 0]
                ax.plot(x, data['exact_'+exact][order][mask], color='black', lw=1.2, label='exact two-gas solution')
                ax.scatter(x, value, s=6, label='CRKSPH')
                ax.set(xlabel='x', title=exact)
                ax.grid(alpha=.2)
            axes[0].legend(fontsize=7)
        else:
            mask = np.ones(len(pos), dtype=bool)
            if row['case'] == 'shockBubble':
                mask = (pos[:, 0] < 1.1) & (pos[:, 1] < .4)
            fields = (ids, data['densities'], data['pressures'])
            for ax, value, label in zip(axes, fields, ('Material', 'Density', 'Pressure')):
                limits = {}
                if row['case'] == 'shockBubble':
                    lo, hi = {'Material': (0, 1), 'Density': (.15, 3.5), 'Pressure': (1, 10)}[label]
                    limits = dict(vmin=lo, vmax=hi)
                dots = ax.scatter(pos[mask, 0], pos[mask, 1], c=value[mask], s=1.8, cmap='viridis', rasterized=True, **limits)
                fig.colorbar(dots, ax=ax, shrink=.75)
                ax.set(title=label, xlabel='x', ylabel='y', aspect='equal')
        fig.suptitle(f'{row["name"]} — t={row["time"]:.5g}, nx={row["nx"]}, {row["wallSeconds"]:.1f}s wall')
        figure = output/(row['name']+'.png')
        fig.savefig(figure, dpi=160)
        plt.close(fig)
        row['figure'] = str(figure.resolve())
        rows.append(row)
        f = row['final']
        keys = ('pressureError', 'maxPeculiarVelocity', 'densityL1', 'pressureL1', 'velocityL1', 'bubbleWidth', 'circulationUpper')
        if row['case'] == 'shockBubble' and row['params'].get('driverPressure', 10.) != 1.:
            keys = tuple(k for k in keys if k != 'pressureError')
        measures = ', '.join(f'{k}={f[k]:.5g}' for k in keys if k in f)
        print(f'{row["name"]}: {"pass" if row["passed"] else "FAIL"}, {measures}, {row["wallSeconds"]:.2f}s, {row["meanStepMilliseconds"]:.2f}ms/step')
    (output/'comparison.json').write_text(json.dumps(dict(runs=rows,
        limits='Static/translated contacts and Sod have exact targets. ShockBubble gates cover completion/conservation only; shape agreement requires an independent matched reference.'), indent=2)+'\n')
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('roots', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    analyze(args.roots, args.output)
