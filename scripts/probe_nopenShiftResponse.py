#!/usr/bin/env python
"""Characterise `computeMdbcNoPenShift` on a synthetic flat wall.

One fluid particle is swept over a grid of
  * **normal** distance to the wall surface, `n` from +2 dx (in the fluid) to
    -2 dx (inside the solid), and
  * **tangential** phase relative to the nearest boundary particle, `t` from
    -dx/2 to +dx/2,
at three velocity directions (into the wall / tangential / out of the wall).
The computed correction is recorded at every point and rendered as images, so
what the term actually *does* is visible rather than inferred from the branch
structure (`DELTASPH_VALIDATION_PLAN.md` 5.9).

The whole grid runs in ONE kernel launch: each sample gets its own patch of the
same long wall, spaced far enough apart in x that the samples cannot see each
other.

Expected, from DualSPHysics `ComputeNoPenVel`:
  * zero everywhere the particle is moving *away* from the wall (`vfc >= 0`);
  * zero beyond `r > 1.25 dp` of a first-layer boundary particle;
  * for an approaching particle, a correction opposing the normal velocity,
    scaled by `factor = -4*ratio + 3`, `ratio = max(|dr/norm|, 0.25)`;
  * **averaged** over the contributing boundary particles.

    python scripts/probe_nopenShiftResponse.py [--nt 41] [--nn 49] [--out DIR]
"""
from __future__ import annotations
import argparse, os

ap = argparse.ArgumentParser()
ap.add_argument('--nt', type=int, default=41, help='tangential samples over [-dx/2, dx/2]')
ap.add_argument('--nn', type=int, default=49, help='normal samples over [-2dx, 2dx]')
ap.add_argument('--nh', type=float, default=4.0, help='support ratio h/dx')
ap.add_argument('--layers', type=int, default=3, help='boundary layers')
ap.add_argument('--out', default=os.path.join('scratchpad', 'nopen_response'))
args = ap.parse_args()

from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from warpSPHCore import (KernelFunctions, DomainDescription, SupportScheme,
                         buildVerletList, sphKernelScale)
from warpSPH.modules.mdbc import computeMdbcNoPenShift
from warpSPH.modules.mdbc.wp_nopenshift import computeMdbcNoPenShiftWarp
from warpSPHCore import OperationProperties, WarpOperation, OperationDirection

dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
DT = torch.float32
dx = 0.01
h = args.nh * dx / sphKernelScale(KernelFunctions.Wendland2.value, 2) * 0 + args.nh * dx
# support radius actually used by the kernel; keep it generous so the 1.25dp
# and 1.75dp gates -- not the kernel cutoff -- are what limits the stencil.
support = 2.5 * dx

tangents = np.linspace(-0.5 * dx, 0.5 * dx, args.nt)
normals = np.linspace(2.0 * dx, -2.0 * dx, args.nn)     # +2dx (fluid) -> -2dx (solid)
SPACING = 16 * dx                                        # patch pitch; >> 2*support

VEL_CASES = {
    'into wall  v=(0,-1)': (0.0, -1.0),
    'tangential v=(1, 0)': (1.0, 0.0),
    'out of wall v=(0,+1)': (0.0, 1.0),
}

def buildState(velDir):
    """One long wall + one fluid sample per (t, n) grid point, patches spaced
    `SPACING` apart in x so samples are mutually invisible."""
    pos, vel, kinds, goff, gidx = [], [], [], [], []
    samples = []
    nSamplesX = args.nt * args.nn
    # wall spans every patch; lattice phase is global so `t` is a true phase
    xMax = nSamplesX * SPACING + 8 * dx
    nWall = int(xMax / dx) + 1
    wallX = np.arange(nWall) * dx
    for k in range(args.layers):                         # y = -(k+0.5) dx
        y = -(k + 0.5) * dx
        for wx in wallX:
            pos.append([wx, y]); vel.append([0.0, 0.0]); kinds.append(1)
            # r_g = r_b - offset, mirrored through the surface at y = 0
            goff.append([0.0, -(2 * k + 1) * dx])
    nBnd = len(pos)
    for bi in range(nBnd):                               # ghost per boundary
        gidx.append(bi)
    ghostStart = nBnd
    for bi in range(nBnd):
        b = pos[bi]; o = goff[bi]
        pos.append([b[0] - o[0], b[1] - o[1]]); vel.append([0.0, 0.0]); kinds.append(2)
        goff.append([-o[0], -o[1]])
    # fluid samples
    idx = 0
    fluidStart = len(pos)
    for jn, n in enumerate(normals):
        for it, t in enumerate(tangents):
            cx = (idx + 1) * SPACING
            # snap the patch centre onto the wall lattice, then add the phase
            cx = round(cx / dx) * dx + t
            pos.append([cx, float(n)]); vel.append(list(velDir)); kinds.append(0)
            goff.append([0.0, 0.0])
            samples.append((jn, it))
            idx += 1
    n = len(pos)
    ghostIndices = torch.full((n,), -1, dtype=torch.int64, device=dev)
    ghostIndices[ghostStart:ghostStart + nBnd] = torch.arange(nBnd, device=dev)

    class S: pass
    st = S()
    st.positions = torch.tensor(pos, device=dev, dtype=DT)
    st.velocities = torch.tensor(vel, device=dev, dtype=DT)
    st.kinds = torch.tensor(kinds, device=dev, dtype=torch.int32)
    st.ghostOffsets = torch.tensor(goff, device=dev, dtype=DT)
    st.ghostIndices = ghostIndices
    st.supports = torch.full((n,), support, device=dev, dtype=DT)
    st.masses = torch.full((n,), dx * dx, device=dev, dtype=DT)   # dp = sqrt(m) = dx
    st.densities = torch.ones(n, device=dev, dtype=DT)
    return st, samples, fluidStart

class C: pass
cfg = C()
cfg.kernel = KernelFunctions.Wendland2
cfg.dim = 2
cfg.verletScale = 1.0

results = {}
counts = {}
for label, vd in VEL_CASES.items():
    st, samples, fluidStart = buildState(vd)
    lo = st.positions.min(dim=0).values - 8 * dx
    hi = st.positions.max(dim=0).values + 8 * dx
    cfg.domain = DomainDescription(lo, hi,
                                   torch.zeros(2, dtype=torch.bool, device=dev), 2)
    adj = buildVerletList(st, cfg.domain, verletScale=1.0,
                          supportMode=SupportScheme.SuperSymmetric,
                          priorNeighborhood=None, verbose=False)
    raw = computeMdbcNoPenShiftWarp(
        st,
        operationProperties=OperationProperties(
            kernel=cfg.kernel, operation=WarpOperation.Interpolate,
            supportMode=SupportScheme.Gather,
            operationMode=OperationDirection.BoundaryToFluid),
        domain=cfg.domain, adjacency=adj,
        queryOffsets=st.ghostOffsets, queryVelocities=st.velocities)
    shift = raw[0].detach().cpu().numpy()
    count = raw[1].detach().cpu().numpy()
    grid = np.full((args.nn, args.nt, 2), np.nan)
    cgrid = np.full((args.nn, args.nt, 2), np.nan)
    for k, (jn, it) in enumerate(samples):
        grid[jn, it] = shift[fluidStart + k]
        cgrid[jn, it] = count[fluidStart + k]
    results[label] = grid
    counts[label] = cgrid
    nz = np.abs(grid).sum(axis=2) > 0
    print(f'{label}: non-zero at {nz.sum():5d}/{nz.size} grid points, '
          f'|shift| max = {np.abs(grid).max():.4g}, '
          f'contributing-neighbour count max = {np.nanmax(cgrid):.0f}', flush=True)

os.makedirs(args.out, exist_ok=True)
ext = [tangents[0] / dx, tangents[-1] / dx, normals[-1] / dx, normals[0] / dx]
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for c, (label, grid) in enumerate(results.items()):
    for r, comp in enumerate(('normal (y)', 'tangential (x)')):
        g = grid[:, :, 1 - r]
        m = np.nanmax(np.abs(g)) or 1.0
        ax = axes[r, c]
        im = ax.imshow(g, extent=ext, aspect='auto', origin='upper',
                       cmap='RdBu_r', vmin=-m, vmax=m)
        ax.axhline(0.0, color='k', lw=1.2)                 # wall surface
        ax.axhline(-0.5, color='0.4', lw=0.8, ls='--')     # 1st boundary row
        ax.set_title(f'{label}\nshift {comp}   (max |.| = {m:.3g})', fontsize=9)
        ax.set_xlabel('tangential phase  t/dx'); ax.set_ylabel('normal  n/dx')
        fig.colorbar(im, ax=ax)
fig.suptitle('computeMdbcNoPenShift response -- flat wall, single fluid particle\n'
             'black line = wall surface (n=0); dashed = first boundary row (n=-0.5dx)',
             fontsize=11)
fig.tight_layout()
p = os.path.join(args.out, 'nopen_response.png')
fig.savefig(p, dpi=130); plt.close(fig)
print('->', p)

# normal-direction profile at zero tangential phase, the cleanest 1-D read
mid = args.nt // 2
print(f'\n normal profile at t = {tangents[mid]/dx:+.2f} dx:')
print(f'{"n/dx":>7} ' + ' '.join(f'{k:>26}' for k in results))
for jn, n in enumerate(normals):
    if abs((n / dx) * 4 - round((n / dx) * 4)) > 1e-6:
        continue
    cells = []
    for label in results:
        s = results[label][jn, mid]
        c = counts[label][jn, mid]
        cells.append(f'({s[0]:+.3f},{s[1]:+.3f}) n={c[1]:.0f}')
    print(f'{n/dx:7.2f} ' + ' '.join(f'{c:>26}' for c in cells))
