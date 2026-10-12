import sys
from pathlib import Path
import h5py, numpy as np, torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
R = Path('/home/lu26029/dev/warpSPH/results')
snap = R / 'gizmo_compare_2026-10-12/kh256_mfm/output/snapshot_013.hdf5'
with h5py.File(snap, 'r') as h:
    t = float(h['Header'].attrs['Time']); p = h['PartType0']
    g = dict(pos=p['Coordinates'][:, :2], rho=p['Density'][:], v=np.linalg.norm(p['Velocities'][:, :2], axis=1))
def ours(tag):
    d = torch.load(R / f'e2_kh256_cap_2026-10-12/MFM_t26_{tag}/final.pt', weights_only=False)
    return dict(pos=d['positions'].numpy(), rho=d['densities'].numpy(), v=np.linalg.norm(d['velocities'].numpy(), axis=1))
arms = [('GIZMO MFM (fp64, CPU)', g), ('ours: closure = none (fp32)', ours('none')), ('ours: closure = project, shipped (fp32)', ours('project'))]
fig, ax = plt.subplots(2, 3, figsize=(16, 10.4), facecolor='#fcfcfb')
for k, (lab, d) in enumerate(arms):
    out = ((d['rho'] < 0.85) | (d['rho'] > 2.35)).mean() * 100
    for row, (key, vmin, vmax, cm, name) in enumerate((('rho', 0.8, 2.2, 'RdBu_r', 'density'), ('v', 0.0, 1.0, 'viridis', '|v|'))):
        a = ax[row, k]
        sc = a.scatter(d['pos'][:, 0], d['pos'][:, 1], c=d[key], s=0.6, cmap=cm, vmin=vmin, vmax=vmax, rasterized=True)
        a.set_aspect('equal'); a.set_xticks([]); a.set_yticks([]); a.set_xlim(0, 1); a.set_ylim(0, 1)
        ttl = f'{lab}\n{name}' + (f': range [{d["rho"].min():.2f}, {d["rho"].max():.2f}], {out:.2f}% outside [0.85, 2.35]' if row == 0 else f': max {d["v"].max():.2f}, {int((d["v"] > 1.5).sum())} particles > 1.5')
        a.set_title(ttl, fontsize=10, loc='left', color='#0b0b0b')
        if k == 2: fig.colorbar(sc, ax=ax[row, :], fraction=0.015, pad=0.01, label=name + (' (clipped to the colour range)' if row == 0 else ''))
fig.suptitle(f'Kelvin-Helmholtz 256, same IC, t = {t:.1f}: GIZMO MFM vs our MFM closures', x=0.02, ha='left', fontsize=13)
out = R / 'gizmo_compare_2026-10-12' / 'kh256_t2.6_gizmo_vs_ours.png'
fig.savefig(out, dpi=95, facecolor=fig.get_facecolor(), bbox_inches='tight')
print(out)
