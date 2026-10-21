"""Render corrected-curl maps from completed RM trajectories, without advancing.

Run sequentially after the simulation campaign releases the GPU. Input files
are opened read-only; derived observations are written to a separate directory.
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('roots', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    from warpSPHBootstrap import bootstrap
    bootstrap(precision='float32')
    import numpy as np
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from warpSPH.configurations import dictToCRKSPHConfig
    from warpSPH.io.importIO import importConfigs, loadTrajectory, loadTrajectoryFrame
    from warpSPH.modules.reconstruction.gradient import computeVelocityJacobian
    from warpSPHCore import SupportScheme
    paths = sorted({p for root in args.roots for p in root.rglob('materialValidation.json')})
    args.output.mkdir(parents=True, exist_ok=True)
    for path in paths:
        summary = json.loads(path.read_text())
        if summary['scheme'] != 'CRKSPH' or not (path.parent/'trajectory.h5').is_file():
            continue
        cfg, scfg = importConfigs(path.parent/'config.json', dictToCRKSPHConfig)
        file, meta = loadTrajectory(path.parent, 'cuda', extraFields=('internalEnergies', 'supports'))
        try:
            system, time = loadTrajectoryFrame(file, meta, -1, schemeConfig=scfg)
        finally:
            file.close()
        J = computeVelocityJacobian(system.state, cfg, SupportScheme.KernelMeanSymmetric, corrected=True)
        omega = J[:, 1, 0]-J[:, 0, 1]
        history = json.loads((path.parent/'diagnosticHistory.json').read_text())
        center = history[-1]['interfaceMean']
        state = system.state
        mask = (state.positions[:, 0]-center).abs() < .75
        fields = {name: getattr(state, name)[mask].detach().cpu().numpy()
                  for name in ('positions', 'velocities', 'densities', 'pressures', 'materials')}
        fields['vorticity'] = omega[mask].detach().cpu().numpy()
        fields['time'] = time
        name = path.parents[1].name
        np.savez_compressed(args.output/(name+'.npz'), **fields)
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
        limit = max(float(np.quantile(np.abs(fields['vorticity']), .995)), 1e-6)
        for ax, field, cmap, bounds in zip(axes, ('materials', 'densities', 'vorticity'),
                ('viridis', 'viridis', 'RdBu_r'), ((0, 1), (None, None), (-limit, limit))):
            x = fields['positions']
            dots = ax.scatter(x[:, 0], x[:, 1], c=fields[field], s=1, linewidths=0,
                              cmap=cmap, vmin=bounds[0], vmax=bounds[1], rasterized=True)
            ax.set(title=field, xlabel='x', ylabel='y', xlim=(center-.75, center+.75), ylim=(-.5, .5), aspect='equal')
            fig.colorbar(dots, ax=ax, shrink=.7)
        fig.suptitle(f'{name}, t={time:.4f}; corrected SPH curl (99.5% display range)')
        fig.savefig(args.output/(name+'.png'), dpi=160)
        plt.close(fig)
        print(f'Observed {name} at t={time:g}: {args.output/(name+".png")}', flush=True)


if __name__ == '__main__':
    main()
