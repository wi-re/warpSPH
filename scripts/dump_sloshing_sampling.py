#!/usr/bin/env python
"""Dump the sloshingTank (SPHERIC TC10) boundary sampling + mDBC ghost-node
placement at t = 0 -- no time stepping. Renders a whole-tank overview plus a
zoom on each of the four interior corners, with the boundary -> ghost-node
connector drawn for every wall particle in view (that segment *is* the mDBC
extrapolation stencil: the ghost node is where the fluid field is sampled and
linearly extrapolated back to the wall particle).

    python scripts/dump_sloshing_sampling.py --nx 225
    python scripts/dump_sloshing_sampling.py --nx 225 --ghost geometric

`--ghost` sets WARPSPH_GHOST_PLACEMENT for the build (default: the tree default,
'gridsnap' on HEAD).
"""
from __future__ import annotations

import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument('--nx', type=int, default=225)
ap.add_argument('--ghost', default=None,
                help="WARPSPH_GHOST_PLACEMENT for the build (gridsnap|geometric|bodynode|simple)")
ap.add_argument('--out', default=os.path.join('scratchpad', 'sloshing_sampling'))
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

from warpSPH.cases import importAll
importAll()
from warpSPH.runner import getCase, run

os.makedirs(args.out, exist_ok=True)
case = getCase('sloshingTank')
r = run(case, scheme='deltaSPH', nx=args.nx, nSteps=1, tLimit=1e9,
        quiet=True, store=False, progress=False)

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
print(f'nx={args.nx}  dx={dx:.5g}  ghost placement = {mode}')
print(f'fluid {len(fluid)}  boundary {len(bnd)}  ghost {len(ghost)}')

# interior tank geometry
L = float(r.ctx.spec.L)
halfW = 0.5 * L
tankH = float(r.ctx.param('tankHeight'))
fill = float(r.ctx.param('fillDepth'))
print(f'tank: [-{halfW:.3g}, {halfW:.3g}] x [0, {tankH:.3g}]   fill depth {fill:.4g}  '
      f'(H/dx = {fill/dx:.1f})')

goffN = np.linalg.norm(goff[kinds == 1], axis=1) / dx
nz = goffN > 1e-6
print(f'ghost offset |r_b - r_g| / dx  (nonzero {nz.sum()}/{len(goffN)}): '
      f'min {goffN[nz].min():.2f}  median {np.median(goffN[nz]):.2f}  '
      f'max {goffN.max():.2f}   zero-offset (Shepard/rest fallback): {(~nz).sum()}')

# ghost nodes landing back inside the solid wall (interior sdf < 0 == wall)
from warpSPH.regions import sampleDomainSDF
interior = r.ctx.scratch['interiorDomain']
gnode = pos[kinds == 1] - goff[kinds == 1]
_dev = st.positions.device
gd = sampleDomainSDF(torch.tensor(gnode, dtype=torch.float32, device=_dev), interior, invert=False)
if isinstance(gd, (tuple, list)):
    gd = gd[0]
gd = gd.detach().cpu().numpy()
print(f'ghost nodes inside the wall solid (interior sdf<0): {(gd < -1e-6).sum()} / {len(gnode)}'
      f'  (min {gd.min():.3g})')


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
    for xw in (-halfW, halfW):
        ax.axvline(xw, c='k', lw=0.8, alpha=0.5)
    for yw in (0.0, tankH):
        ax.axhline(yw, c='k', lw=0.8, alpha=0.5)
    ax.axhline(fill, c='#1f9e4a', lw=0.8, ls='--', alpha=0.7)  # still-water line
    ax.set_xlim(cx - rad, cx + rad); ax.set_ylim(cy - rad, cy + rad)
    ax.set_aspect('equal'); ax.grid(alpha=0.2)


# overview
fig, ax = plt.subplots(figsize=(11, 7))
_scatter(ax, 0.0, tankH * 0.5, max(halfW, tankH * 0.5) + 3 * dx)
ax.set_title(f'sloshingTank IC -- nx={args.nx}, dx={dx:.4g}, ghosts={mode}   '
             f'fluid {len(fluid)} / bnd {len(bnd)} / ghost {len(ghost)}')
ax.legend(fontsize=8, loc='upper right')
fig.tight_layout()
p0 = os.path.join(args.out, f'overview_nx{args.nx}_{mode.strip("()").replace(" ", "")}.png')
fig.savefig(p0, dpi=120); plt.close(fig)
print('->', p0)

# four corners + a flat-wall reference band, zoomed
rad = 14 * dx
spots = {
    'bottom-left corner (wetted)': (-halfW, 0.0),
    'bottom-right corner (wetted)': (halfW, 0.0),
    'left wall at still-water line': (-halfW, fill),
    'right wall at still-water line': (halfW, fill),
    'top-left corner (dry)': (-halfW, tankH),
    'flat bottom wall mid-span': (0.0, 0.0),
}
fig, axes = plt.subplots(2, 3, figsize=(17, 10))
for ax, (title, (cx, cy)) in zip(axes.ravel(), spots.items()):
    _scatter(ax, cx, cy, rad)
    ax.set_title(f'{title}\n(dx={dx:.4g})', fontsize=9)
    ax.legend(fontsize=6, loc='upper right')
fig.suptitle(f'sloshingTank boundary + mDBC ghost sampling  (nx={args.nx}, '
             f'ghosts={mode})  -- red segment = extrapolation stencil', fontsize=11)
fig.tight_layout()
p1 = os.path.join(args.out, f'corners_nx{args.nx}_{mode.strip("()").replace(" ", "")}.png')
fig.savefig(p1, dpi=120); plt.close(fig)
print('->', p1)
