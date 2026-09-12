"""Where does the hybrid rule actually differ from gridsnap? Marks every
boundary particle whose gridsnap node landed in the solid and therefore got the
lattice snap instead.
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
from warpSPH.rigidBody.ghostParticles import (_gridSnapGhostOffsets,
                                              _latticeGhostOffsets)
from warpSPH.configurations.region import RegionType

OUT = os.path.join('scratchpad', 'openchecks', 'm34_hybrid_rescue')
os.makedirs(OUT, exist_ok=True)

params = m34._params('deltaSPH', cornerOnly=False)
params['machTarget'] = 1.95 / 28.3
r = run(dambreakCase, scheme='deltaSPH', L=m34.TANK_L, nx=256, nSteps=1,
        tLimit=1e9, quiet=True, store=False, progress=False, params=params)

st = r.state.state
dx = float(r.ctx.config.dx)
hMean = float(st.supports.mean())
pos, kinds = st.positions, st.kinds
sdfs = [reg.sdf for reg in r.ctx.schemeConfig.regions if reg.type == RegionType.Boundary]
sdf = sdfs[0]
bpos = pos[kinds == 1]

base = _gridSnapGhostOffsets(bpos, sdfs, dx, hMean)
alt = _latticeGhostOffsets(bpos, sdfs, dx, hMean)
inSolid = (sdf(bpos - base)[0].reshape(-1) < 0).detach().cpu().numpy()
altZero = (alt.norm(dim=1) < 1e-9).detach().cpu().numpy()

bn = bpos.detach().cpu().numpy()
rescued = inSolid & ~altZero          # gridsnap declined, lattice placed it
stillZero = inSolid & altZero         # both decline (deep interior, > 2h)
kept = ~inSolid                       # gridsnap's node was fine

bed = -m34.TANK_L / 2.0
wall = m34.TANK_W / 2.0

fig, axes = plt.subplots(1, 2, figsize=(17, 6.5))
views = {
    'whole tank': (0.0, bed + 1.2, max(wall, 3.0) + 0.3),
    'obstacle + fillet': (1.8, bed + 0.7, 3.6),
}
for ax, (name, (cx, cy, rad)) in zip(axes, views.items()):
    for m, c, s, lbl in ((kept, '#bbbbbb', 3, f'gridsnap kept ({kept.sum()})'),
                         (rescued, '#d62728', 7, f'lattice-rescued ({rescued.sum()})'),
                         (stillZero, '#1f77b4', 7, f'both decline, >2h ({stillZero.sum()})')):
        ax.scatter(bn[m, 0], bn[m, 1], s=s, c=c, label=lbl, zorder=3)
    ax.set_xlim(cx - rad, cx + rad)
    ax.set_ylim(cy - rad, cy + rad)
    ax.set_aspect('equal')
    ax.grid(alpha=0.25)
    ax.set_title(name, fontsize=10)
    ax.legend(fontsize=8, loc='upper right', markerscale=2)

fig.suptitle('Marrone 3.4 -- which boundary particles the hybrid rule rescues '
             f'(nx=256, dx={dx:.4g})', fontsize=12)
fig.tight_layout()
p = os.path.join(OUT, 'hybrid_rescue_map.png')
fig.savefig(p, dpi=130)
print('->', p)
print(f'kept {kept.sum()}  rescued {rescued.sum()}  stillZero {stillZero.sum()}  total {len(bn)}')
