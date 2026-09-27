#!/usr/bin/env python
"""What the mDBC no-penetration correction does at an orthogonal domain corner.

On a flat axis-aligned wall the correction is a clean velocity reversal. At a
corner the boundary particles' ghost normals swing smoothly through 90deg -- the
"fan" visible in `dump_sloshing_sampling.py` / `dump_marrone31_sampling.py`,
because the nearest surface point for a particle in the diagonal band is the
apex itself. This sweeps a fluid particle over the fluid quadrant around such a
corner and records what the correction does to its velocity.

Geometry: fluid = {x > 0, y > 0}; solid elsewhere; `layers` rows of boundary
particles on each wall plus the diagonal block. Ghost offset = -2*phi*n_hat with
(phi, n_hat) the true distance/normal to the L-shaped surface, so the corner
block gets radial normals exactly as the real ghost placement does.

Reported per grid point, for a particle driven into the corner:
  * |v_new| / |v_in|   -- 1 = elastic, 0 = stopped dead
  * the turn angle between v_new and the specular reflection about the nearest
    surface normal.

    python scripts/probe_nopenShiftCorner.py [--n 61] [--vdir diag|x|y]
"""
from __future__ import annotations
import argparse, os

ap = argparse.ArgumentParser()
ap.add_argument('--n', type=int, default=61, help='grid samples per axis')
ap.add_argument('--extent', type=float, default=3.0, help='grid half-extent in dx')
ap.add_argument('--layers', type=int, default=3)
ap.add_argument('--vdir', default='diag', choices=('diag', 'x', 'y'),
                help='incoming velocity direction (unit speed)')
ap.add_argument('--out', default=os.path.join('scratchpad', 'nopen_corner'))
args = ap.parse_args()

from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')

import numpy as np, torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from warpSPHCore import (KernelFunctions, DomainDescription, SupportScheme,
                         buildVerletList)
from warpSPH.modules.mdbc import computeMdbcNoPenShift

dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
DT = torch.float32
dx = 0.01
support = 2.5 * dx
SPACING = 16 * dx

VDIR = {'diag': (-1 / np.sqrt(2), -1 / np.sqrt(2)), 'x': (-1.0, 0.0), 'y': (0.0, -1.0)}[args.vdir]

def surface(bx, by):
    """(phi, n_hat) to the L-shaped surface of the solid {x<0 or y<0}."""
    if bx >= 0.0 and by < 0.0:
        return -by, (0.0, 1.0)
    if by >= 0.0 and bx < 0.0:
        return -bx, (1.0, 0.0)
    r = np.hypot(bx, by)                      # diagonal block -> radial to apex
    if r < 1e-12:
        return 0.0, (0.7071, 0.7071)
    return r, (-bx / r, -by / r)

gs = np.linspace(0.25 * dx, args.extent * dx, args.n)      # fluid quadrant only
pos, vel, kinds, goff, samples = [], [], [], [], []
idx = 0
for jy, gy in enumerate(gs):
    for ix, gx in enumerate(gs):
        cx, cy = idx * SPACING, 0.0
        # corner walls local to this patch
        for k in range(args.layers):
            for m in range(-args.layers, 14):
                bx, by = m * dx + 0.5 * dx, -(k + 0.5) * dx        # bottom wall
                phi, nh = surface(bx, by)
                pos.append([cx + bx, cy + by]); vel.append([0., 0.]); kinds.append(1)
                goff.append([-2 * phi * nh[0], -2 * phi * nh[1]])
                bx, by = -(k + 0.5) * dx, m * dx + 0.5 * dx        # left wall
                phi, nh = surface(bx, by)
                pos.append([cx + bx, cy + by]); vel.append([0., 0.]); kinds.append(1)
                goff.append([-2 * phi * nh[0], -2 * phi * nh[1]])
        samples.append((jy, ix, len(pos), gx, gy))
        pos.append([cx + gx, cy + gy]); vel.append(list(VDIR)); kinds.append(0)
        goff.append([0., 0.])
        idx += 1

nAll = len(pos)
kindsT = torch.tensor(kinds, device=dev, dtype=torch.int32)
ghostStart = nAll
bIdx = [i for i, k in enumerate(kinds) if k == 1]
for bi in bIdx:                                # one ghost per boundary particle
    b, o = pos[bi], goff[bi]
    pos.append([b[0] - o[0], b[1] - o[1]]); vel.append([0., 0.]); kinds.append(2)
    goff.append([-o[0], -o[1]])
n = len(pos)
gi = torch.full((n,), -1, dtype=torch.int64, device=dev)
gi[ghostStart:ghostStart + len(bIdx)] = torch.tensor(bIdx, device=dev)

class S: pass
st = S()
st.positions = torch.tensor(pos, device=dev, dtype=DT)
st.velocities = torch.tensor(vel, device=dev, dtype=DT)
st.kinds = torch.tensor(kinds, device=dev, dtype=torch.int32)
st.ghostOffsets = torch.tensor(goff, device=dev, dtype=DT)
st.ghostIndices = gi
st.supports = torch.full((n,), support, device=dev, dtype=DT)
st.masses = torch.full((n,), dx * dx, device=dev, dtype=DT)
st.densities = torch.ones(n, device=dev, dtype=DT)

class C: pass
cfg = C(); cfg.kernel = KernelFunctions.Wendland2; cfg.dim = 2; cfg.verletScale = 1.0
lo = st.positions.min(dim=0).values - 8 * dx
hi = st.positions.max(dim=0).values + 8 * dx
cfg.domain = DomainDescription(lo, hi, torch.zeros(2, dtype=torch.bool, device=dev), 2)
adj = buildVerletList(st, cfg.domain, verletScale=1.0,
                      supportMode=SupportScheme.SuperSymmetric,
                      priorNeighborhood=None, verbose=False)
shift = computeMdbcNoPenShift(st, cfg, None, adj).detach().cpu().numpy()

N = args.n
speed = np.full((N, N), np.nan); turn = np.full((N, N), np.nan)
fired = np.zeros((N, N), bool)
v_in = np.array(VDIR)
for (jy, ix, fi, gx, gy) in samples:
    s = shift[fi]
    vNew = v_in + s
    speed[jy, ix] = np.linalg.norm(vNew)
    fired[jy, ix] = np.abs(s).sum() > 0
    phi, nh = surface(gx, gy) if (gx < 0 or gy < 0) else (0, None)
    # specular reflection about the nearest WALL normal for this quadrant point
    nx_, ny_ = (1.0, 0.0) if gx < gy else (0.0, 1.0)
    nh2 = np.array([nx_, ny_])
    spec = v_in - 2 * np.dot(v_in, nh2) * nh2
    if np.linalg.norm(vNew) > 1e-9:
        c = np.dot(vNew, spec) / (np.linalg.norm(vNew) * np.linalg.norm(spec))
        turn[jy, ix] = np.degrees(np.arccos(np.clip(c, -1, 1)))

print(f'incoming v = {VDIR}  |v| = {np.linalg.norm(v_in):.3f}')
print(f'correction fired at {fired.sum()}/{fired.size} grid points')
ok = fired & np.isfinite(speed)
if ok.any():
    print(f'|v_new|/|v_in| over fired points: min {np.nanmin(speed[ok]):.3f}  '
          f'median {np.nanmedian(speed[ok]):.3f}  max {np.nanmax(speed[ok]):.3f}')
    print(f'  (1.0 = elastic, 0.0 = stopped dead)')
print(f'\n diagonal cut (x = y), distance from apex:')
print(f'{"r/dx":>7} {"|v_new|":>9} {"vx_new":>9} {"vy_new":>9}  fired')
for k in range(0, N, max(1, N // 14)):
    fi = [s for s in samples if s[0] == k and s[1] == k]
    if not fi: continue
    _, _, f_, gx, gy = fi[0]
    s = shift[f_]; vN = v_in + s
    print(f'{np.hypot(gx,gy)/dx:7.2f} {np.linalg.norm(vN):9.4f} {vN[0]:9.4f} '
          f'{vN[1]:9.4f}  {bool(np.abs(s).sum()>0)}')

os.makedirs(args.out, exist_ok=True)
ext = [gs[0] / dx, gs[-1] / dx, gs[0] / dx, gs[-1] / dx]
fig, ax = plt.subplots(1, 2, figsize=(13, 5.4))
im = ax[0].imshow(speed, extent=ext, origin='lower', cmap='viridis', vmin=0, vmax=1.2)
ax[0].set_title(f'|v_new| / |v_in|   (incoming {args.vdir})\n'
                '1 = elastic,  0 = stopped dead'); fig.colorbar(im, ax=ax[0])
im = ax[1].imshow(turn, extent=ext, origin='lower', cmap='magma', vmin=0, vmax=180)
ax[1].set_title('angle between v_new and specular reflection [deg]')
fig.colorbar(im, ax=ax[1])
for a in ax:
    a.set_xlabel('x from corner apex  [dx]'); a.set_ylabel('y from corner apex  [dx]')
fig.suptitle('mDBC no-penetration correction at an orthogonal domain corner '
             '(ghost normals fan through 90 deg)', fontsize=11)
fig.tight_layout()
p = os.path.join(args.out, f'nopen_corner_{args.vdir}.png')
fig.savefig(p, dpi=130); plt.close(fig)
print('->', p)
