#!/usr/bin/env python
"""Dump the Marrone 2011 Sec. 3.1 dam-break boundary sampling + mDBC ghost-node
placement at t = 0 -- no time stepping. Same tool as `dump_sloshing_sampling.py`,
pointed at this case: renders a whole-tank overview plus a zoom on the
downstream-wall corner (where P1/P2/P3 sit, and where the 2026-09-11 scheme-fix
sweep showed the run-up jet fragmenting) and the upstream/reservoir corner for
comparison.

    python scripts/dump_marrone31_sampling.py --nx 72
    python scripts/dump_marrone31_sampling.py --nx 72 --ghost geometric

`--ghost` sets WARPSPH_GHOST_PLACEMENT for the build (default: the tree
default, 'gridsnap' on HEAD).
"""
from __future__ import annotations

import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument('--nx', type=int, default=72)
ap.add_argument('--ghost', default=None,
                help="WARPSPH_GHOST_PLACEMENT for the build (gridsnap|geometric|bodynode|simple)")
ap.add_argument('--out', default=os.path.join('scratchpad', 'marrone31_sampling'))
args = ap.parse_args()

if args.ghost:
    os.environ['WARPSPH_GHOST_PLACEMENT'] = args.ghost

from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from warpSPH.cases.dambreak import dambreakCase
from warpSPH.runner import run

os.makedirs(args.out, exist_ok=True)

# -- same reference tank as probe_deltaSPHMarrone.py --------------------------
H = 0.60
TANK_L = 1.00
TANK_W = 5.366 * H
COL_W = 2.0 * H
G = 9.81
U_MAX = 1.95 * (G * H) ** 0.5
SENSOR_Z = [0.160, 0.584, 1.000]   # P1, P2, P3 heights above the bed [m]

params = dict(
    W=TANK_W, fillRatio=H / TANK_L, fluidWidth=COL_W / TANK_W,
    gravityMagnitude=G, referenceVelocity=U_MAX, machTarget=1.95 / 40.0,
    pressureProbeHeights=SENSOR_Z, pressureProbeInset=0.0,
    pressureProbeDiscRadius=0.045,
)
r = run(dambreakCase, scheme='deltaSPH', L=TANK_L, nx=args.nx, nSteps=1,
        tLimit=1e9, quiet=True, store=False, progress=False, params=params)

st = r.state.state
pos = st.positions.detach().cpu().numpy()
kinds = st.kinds.detach().cpu().numpy()
goff = (st.ghostOffsets.detach().cpu().numpy() if getattr(st, 'ghostOffsets', None) is not None
        else np.zeros_like(pos))
dx = float(r.ctx.config.dx)
mode = os.environ.get('WARPSPH_GHOST_PLACEMENT', '(tree default)')

fluid = pos[kinds == 0]
bnd = pos[kinds == 1]
ghost = pos[kinds == 2]
print(f'nx={args.nx}  dx={dx:.5g}  H/dx={H/dx:.1f}  ghost placement = {mode}')
print(f'fluid {len(fluid)}  boundary {len(bnd)}  ghost {len(ghost)}')

domain = r.ctx.config.domain
print(f'domain (padded, periodic wrapper): [{domain.min[0].item():.4g}, '
      f'{domain.min[1].item():.4g}] to [{domain.max[0].item():.4g}, {domain.max[1].item():.4g}]')
interior = r.ctx.scratch.get('interiorDomain')
if interior is not None:
    bedY = float(interior.min[1].item())
    halfW = float(interior.max[0].item())
    lowW = float(interior.min[0].item())
else:
    bedY = -TANK_L / 2.0
    halfW = TANK_W / 2.0
    lowW = -halfW
print(f'interior wall box: bed y={bedY:.4g}, downstream wall x={halfW:.4g}, '
      f'upstream wall x={lowW:.4g}')

goffN = np.linalg.norm(goff[kinds == 1], axis=1) / dx
nz = goffN > 1e-6
print(f'ghost offset |r_b - r_g| / dx  (nonzero {nz.sum()}/{len(goffN)}): '
      f'min {goffN[nz].min():.2f}  median {np.median(goffN[nz]):.2f}  '
      f'max {goffN.max():.2f}   zero-offset (Shepard/rest fallback): {(~nz).sum()}')


def _scatter(ax, cx, cy, rad):
    for arr, c, s, lbl in ((fluid, '#7fb3ff', 6, 'fluid'),
                           (bnd, '#666666', 12, 'boundary'),
                           (ghost, '#d62728', 12, 'ghost node')):
        m = ((arr[:, 0] > cx - rad) & (arr[:, 0] < cx + rad)
             & (arr[:, 1] > cy - rad) & (arr[:, 1] < cy + rad))
        ax.scatter(arr[m, 0], arr[m, 1], s=s, c=c, label=lbl, zorder=3)
    bm = ((kinds == 1) & (pos[:, 0] > cx - rad) & (pos[:, 0] < cx + rad)
          & (pos[:, 1] > cy - rad) & (pos[:, 1] < cy + rad))
    for i in np.where(bm)[0]:
        gp = pos[i] - goff[i]
        ax.plot([pos[i, 0], gp[0]], [pos[i, 1], gp[1]], '-', c='#d62728',
                lw=0.6, alpha=0.6, zorder=2)
    for xw in (lowW, halfW):
        ax.axvline(xw, c='k', lw=0.8, alpha=0.5)
    ax.axhline(bedY, c='k', lw=0.8, alpha=0.5)
    for zh in SENSOR_Z:
        y = bedY + zh
        if cy - rad < y < cy + rad:
            ax.axhline(y, c='#2ca02c', lw=0.8, ls='--', alpha=0.7)
    ax.set_xlim(cx - rad, cx + rad); ax.set_ylim(cy - rad, cy + rad)
    ax.set_aspect('equal'); ax.grid(alpha=0.2)


# overview
fig, ax = plt.subplots(figsize=(13, 5))
_scatter(ax, 0.0, 0.0, max(halfW, TANK_L * 0.5) + 3 * dx)
ax.set_title(f'Marrone 3.1 dam-break IC -- nx={args.nx}, dx={dx:.4g}, ghosts={mode}   '
             f'fluid {len(fluid)} / bnd {len(bnd)} / ghost {len(ghost)}')
ax.legend(fontsize=8, loc='upper right')
fig.tight_layout()
p0 = os.path.join(args.out, f'overview_nx{args.nx}_{mode.strip("()").replace(" ", "")}.png')
fig.savefig(p0, dpi=120); plt.close(fig)
print('->', p0)

# corners: downstream (impact) wall bottom, downstream wall at P1 height,
# upstream (reservoir) wall bottom, a flat-bed reference band
rad = 14 * dx
spots = {
    'downstream wall, bed corner (P1 / blow-up site)': (halfW, bedY),
    'downstream wall at P1 height': (halfW, bedY + SENSOR_Z[0]),
    'upstream wall, bed corner (reservoir)': (lowW, bedY),
    'flat bed mid-span': (0.0, bedY),
}
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
for ax, (title, (cx, cy)) in zip(axes.ravel(), spots.items()):
    _scatter(ax, cx, cy, rad)
    ax.set_title(f'{title}\n(dx={dx:.4g})', fontsize=9)
    ax.legend(fontsize=6, loc='upper right')
fig.suptitle(f'Marrone 3.1 boundary + mDBC ghost sampling  (nx={args.nx}, '
             f'ghosts={mode})  -- red segment = extrapolation stencil', fontsize=11)
fig.tight_layout()
p1 = os.path.join(args.out, f'corners_nx{args.nx}_{mode.strip("()").replace(" ", "")}.png')
fig.savefig(p1, dpi=120); plt.close(fig)
print('->', p1)
