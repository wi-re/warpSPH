import re
from pathlib import Path
import h5py, numpy as np, torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
R = Path('/home/lu26029/dev/warpSPH/results')
def gz(v):
    with h5py.File(R / f'gizmo_compare_2026-10-12/kh256_{v}/output/snapshot_013.hdf5', 'r') as h:
        p = h['PartType0']
        return dict(pos=p['Coordinates'][:, :2], rho=p['Density'][:], v=np.linalg.norm(p['Velocities'][:, :2], axis=1))
def ours(sch):
    d = torch.load(R / f'e2_kh256_defaults_2026-10-12/{sch}_t26_default/final.pt', weights_only=False)
    return dict(pos=d['positions'].numpy(), rho=d['densities'].numpy(), v=np.linalg.norm(d['velocities'].numpy(), axis=1))
arms = [('GIZMO MFM (fp64 CPU)', gz('mfm')), ('ours MFM, new defaults', ours('MFM')), ('GIZMO MFV (fp64 CPU)', gz('mfv')), ('ours MFV, new defaults', ours('MFV'))]
fig, ax = plt.subplots(2, 4, figsize=(21, 10.6), facecolor='#fcfcfb')
for k, (lab, d) in enumerate(arms):
    out = ((d['rho'] < 0.85) | (d['rho'] > 2.35)).mean() * 100
    for row, (key, vmin, vmax, cm, name) in enumerate((('rho', 0.8, 2.2, 'RdBu_r', 'density'), ('v', 0.0, 1.0, 'viridis', '|v|'))):
        a = ax[row, k]
        sc = a.scatter(d['pos'][:, 0], d['pos'][:, 1], c=d[key], s=0.5, cmap=cm, vmin=vmin, vmax=vmax, rasterized=True)
        a.set_aspect('equal'); a.set_xticks([]); a.set_yticks([]); a.set_xlim(0, 1); a.set_ylim(0, 1)
        a.set_title(f'{lab}\n' + (f'density [{d["rho"].min():.2f}, {d["rho"].max():.2f}], {out:.2f}% outside [0.85,2.35]' if row == 0 else f'|v| max {d["v"].max():.2f}'), fontsize=10, loc='left', color='#0b0b0b')
        if k == 3: fig.colorbar(sc, ax=ax[row, :], fraction=0.012, pad=0.01, label=name)
fig.suptitle('Kelvin-Helmholtz 256, identical IC, t = 2.6: GIZMO vs ours (new defaults: closure none, cfl 0.1 on L = V^(1/d))', x=0.02, ha='left', fontsize=13)
out = R / 'gizmo_compare_2026-10-12' / 'kh256_t2.6_mfm_mfv_gizmo_vs_ours.png'
fig.savefig(out, dpi=85, facecolor=fig.get_facecolor(), bbox_inches='tight'); print(out)
