"""Tight side-by-side of the mDBC ghost sampling at the Marrone 3.4 wedge's
convex tip (the sharp 45 deg apex, where P1 sits): gridsnap vs lattice.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import probe_deltaSPHMarrone34 as m34
from warpSPH.cases.dambreak import dambreakCase
from warpSPH.runner import run
from warpSPH.rigidBody.ghostParticles import _gridSnapGhostOffsets, _latticeGhostOffsets
from warpSPH.configurations.region import RegionType
from warpSPH.caseUtils.weaklyCompressible import buildObstacleSDF

OUT = os.path.join('scratchpad', 'openchecks', 'm34_apex_cmp')
os.makedirs(OUT, exist_ok=True)

H = 1.0
params = m34._params('deltaSPH', cornerOnly=False)
params['machTarget'] = 1.95 / 28.3
r = run(dambreakCase, scheme='deltaSPH', L=m34.TANK_L, nx=256, nSteps=1,
        tLimit=1e9, quiet=True, store=False, progress=False, params=params)

st = r.state.state
dx = float(r.ctx.config.dx)
hMean = float(st.supports.mean())
pos, kinds = st.positions, st.kinds
sdfs = [reg.sdf for reg in r.ctx.schemeConfig.regions if reg.type == RegionType.Boundary]
bpos = pos[kinds == 1]

offsets = {
    'gridsnap (before)': _gridSnapGhostOffsets(bpos, sdfs, dx, hMean),
    'lattice (now)': _latticeGhostOffsets(bpos, sdfs, dx, hMean),
}

bed = -m34.TANK_L / 2.0
apex = np.array([-m34.TANK_W / 2.0 + 5.0 * H, bed + H])   # convex 45 deg tip
rad = 7.0 * dx

# true surface outline for context
sdfPlot = buildObstacleSDF('marroneSharpEdge', 0, 0, 1, 1, 0, r.ctx.config, None,
                           m34.TANK_L, m34.TANK_W)
gx = np.linspace(apex[0] - rad, apex[0] + rad, 400)
gy = np.linspace(apex[1] - rad, apex[1] + rad, 400)
GX, GY = np.meshgrid(gx, gy)
P = torch.tensor(np.stack([GX.ravel(), GY.ravel()], 1), dtype=torch.float32,
                 device=pos.device)
D = sdfPlot(P).reshape(-1).detach().cpu().numpy().reshape(GX.shape)

bn = bpos.detach().cpu().numpy()
sel = (np.abs(bn[:, 0] - apex[0]) < rad) & (np.abs(bn[:, 1] - apex[1]) < rad)

fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))
for ax, (name, off) in zip(axes, offsets.items()):
    offN = off.norm(dim=1).detach().cpu().numpy()
    gh = (bpos - off).detach().cpu().numpy()
    zero = offN < 1e-9

    ax.contour(GX, GY, D, levels=[0], colors='k', linewidths=1.6)
    ax.scatter(bn[sel & ~zero, 0], bn[sel & ~zero, 1], s=46, c='#666666',
               label='boundary (placed)', zorder=3)
    ax.scatter(bn[sel & zero, 0], bn[sel & zero, 1], s=46, c='#000000',
               marker='x', label='boundary (zero-offset fallback)', zorder=4)
    ax.scatter(gh[sel & ~zero, 0], gh[sel & ~zero, 1], s=46, c='#d62728',
               label='ghost node', zorder=5)
    for i in np.where(sel & ~zero)[0]:
        ax.plot([bn[i, 0], gh[i, 0]], [bn[i, 1], gh[i, 1]], '-',
                c='#d62728', lw=0.8, alpha=0.55, zorder=2)

    nsel = int(sel.sum())
    nz = int((sel & zero).sum())
    gsel = gh[sel & ~zero]
    ndistinct = len(np.unique(np.round(gsel / (0.25 * dx)).astype(int), axis=0)) if len(gsel) else 0
    ax.set_title(f'{name}\n{nsel} bnd in view, {nz} zero-offset, '
                 f'{ndistinct} distinct ghost sites', fontsize=10)
    ax.set_xlim(apex[0] - rad, apex[0] + rad)
    ax.set_ylim(apex[1] - rad, apex[1] + rad)
    ax.set_aspect('equal')
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc='lower left')
    print(f'{name}: {nsel} bnd in view, {nz} zero-offset, {ndistinct} distinct ghost sites, '
          f'offset/dx med {np.median(offN[sel & ~zero])/dx:.2f} max {offN[sel & ~zero].max()/dx:.2f}')

fig.suptitle(f'Marrone 3.4 convex wedge tip (45 deg apex, P1) -- mDBC ghost sampling '
             f'(nx=256, dx={dx:.4g})', fontsize=12)
fig.tight_layout()
p = os.path.join(OUT, 'apex_gridsnap_vs_lattice.png')
fig.savefig(p, dpi=130)
print('->', p)
