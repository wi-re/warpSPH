#!/usr/bin/env python
"""Analyse a GIZMO KH snapshot series (and, optionally, our final.pt states) with the SAME metrics:
max |v|, fast-particle count (|v| > 1.5), KE, density min / max, fraction of particles outside [0.85, 2.35].

    python .tmp/gizmo_kh_analyze.py results/gizmo_compare_2026-10-12/kh256_mfm/output [--ours LABEL=final.pt ...]
"""
import glob
import sys

import h5py
import numpy as np


def stats(rho, vel, m, t, label):
    sp = np.linalg.norm(vel, axis=-1)
    ke = 0.5 * (m * sp ** 2).sum()
    out = ((rho < 0.85) | (rho > 2.35)).mean()
    return (f'{label:22s} t={t:5.2f} maxV={sp.max():7.3f} nFast={int((sp > 1.5).sum()):5d} KE={ke:.5f} '
            f'rho[{rho.min():.3f},{rho.max():.3f}] p0.1/p99.9 {np.percentile(rho, 0.1):.3f}/{np.percentile(rho, 99.9):.3f} '
            f'outside {100 * out:6.3f}%')


def main():
    outdir = sys.argv[1]
    for f in sorted(glob.glob(outdir + '/snapshot_*.hdf5')):
        with h5py.File(f, 'r') as h:
            t = float(h['Header'].attrs['Time'])
            p = h['PartType0']
            v = p['Velocities'][:, :2]
            print(stats(p['Density'][:], v, p['Masses'][:], t, 'GIZMO ' + f.split('_')[-1][:3]), flush=True)
    import torch
    for a in sys.argv[2:]:
        if '=' not in a or a.startswith('--'):
            continue
        label, path = a.split('=', 1)
        d = torch.load(path, map_location='cpu', weights_only=False)
        v = d['velocities'].double().numpy()
        rho = d['densities'].double().numpy()
        print(stats(rho, v, np.full(len(rho), 1.5 / len(rho)) * 0 + 1.0, float(d['t']), label).replace('KE=', 'KE(unit m)='), flush=True)


if __name__ == '__main__':
    main()
