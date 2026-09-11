#!/usr/bin/env python
"""Trajectory of a single ejected particle bouncing off a wall under the mDBC
no-penetration correction.

`probe_nopenShiftResponse.py` measures the correction at one instant; this
integrates it. One fluid particle with no fluid neighbours -- the regime an
ejected droplet is actually in, where there is no pressure force and no
artificial viscosity, so the wall treatment is the *only* thing acting besides
gravity -- is launched at a flat wall and stepped forward with the solver's
`finalize` update:

    shift = computeMdbcNoPenShift(state)
    v = where(shift != 0, v_pre + shift, v + g*dt)
    x = x + v*dt

Reported: bounce heights and the tangential distance travelled. A physical wall
absorbs energy; a perfectly elastic one does not, and the particle skips along
the wall forever -- which is what a free-slip wall plus this correction gives,
since neither supplies any dissipation (DELTASPH_VALIDATION_PLAN 5.10).

    python scripts/probe_nopenShiftTrajectory.py [--steps 4000] [--vx 3] [--vy -2]
"""
from __future__ import annotations
import argparse, os

ap = argparse.ArgumentParser()
ap.add_argument('--steps', type=int, default=4000)
ap.add_argument('--vx', type=float, default=3.0, help='tangential launch velocity [m/s]')
ap.add_argument('--vy', type=float, default=-2.0, help='normal launch velocity [m/s]')
ap.add_argument('--dt', type=float, default=1e-4)
ap.add_argument('--g', type=float, default=-9.81)
ap.add_argument('--layers', type=int, default=3)
ap.add_argument('--y0', type=float, default=0.10, help='launch height [m]')
ap.add_argument('--out', default=os.path.join('scratchpad', 'nopen_trajectory'))
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
xSpan = 4.0                                   # long enough wall for the skip

pos, vel, kinds, goff = [], [], [], []
wallX = np.arange(int(xSpan / dx) + 1) * dx
for k in range(args.layers):
    y = -(k + 0.5) * dx
    for wx in wallX:
        pos.append([wx, y]); vel.append([0.0, 0.0]); kinds.append(1)
        goff.append([0.0, -(2 * k + 1) * dx])
nBnd = len(pos)
for bi in range(nBnd):
    b, o = pos[bi], goff[bi]
    pos.append([b[0] - o[0], b[1] - o[1]]); vel.append([0.0, 0.0]); kinds.append(2)
    goff.append([-o[0], -o[1]])
FLUID = len(pos)
pos.append([0.05, args.y0]); vel.append([args.vx, args.vy]); kinds.append(0)
goff.append([0.0, 0.0])

n = len(pos)
ghostIndices = torch.full((n,), -1, dtype=torch.int64, device=dev)
ghostIndices[nBnd:2 * nBnd] = torch.arange(nBnd, device=dev)

class S: pass
st = S()
st.positions = torch.tensor(pos, device=dev, dtype=DT)
st.velocities = torch.tensor(vel, device=dev, dtype=DT)
st.kinds = torch.tensor(kinds, device=dev, dtype=torch.int32)
st.ghostOffsets = torch.tensor(goff, device=dev, dtype=DT)
st.ghostIndices = ghostIndices
st.supports = torch.full((n,), support, device=dev, dtype=DT)
st.masses = torch.full((n,), dx * dx, device=dev, dtype=DT)
st.densities = torch.ones(n, device=dev, dtype=DT)

class C: pass
cfg = C(); cfg.kernel = KernelFunctions.Wendland2; cfg.dim = 2; cfg.verletScale = 1.0
cfg.domain = DomainDescription(
    torch.tensor([-0.2, -0.2], device=dev, dtype=DT),
    torch.tensor([xSpan + 0.2, 0.6], device=dev, dtype=DT),
    torch.zeros(2, dtype=torch.bool, device=dev), 2)

g = torch.tensor([0.0, args.g], device=dev, dtype=DT)
traj, nCorr = [], 0
for step in range(args.steps):
    adj = buildVerletList(st, cfg.domain, verletScale=1.0,
                          supportMode=SupportScheme.SuperSymmetric,
                          priorNeighborhood=None, verbose=False)
    shift = computeMdbcNoPenShift(st, cfg, None, adj)
    vPre = st.velocities.clone()
    vFree = st.velocities + g * args.dt
    active = (shift != 0) & (st.kinds == 0).unsqueeze(-1)
    if bool(active[FLUID].any()):
        nCorr += 1
    st.velocities = torch.where(active, vPre + shift, vFree)
    st.positions = st.positions + st.velocities * args.dt
    p = st.positions[FLUID]
    traj.append([float(p[0]), float(p[1]), float(st.velocities[FLUID, 0]),
                 float(st.velocities[FLUID, 1]), float(shift[FLUID, 1])])
    if -2.0 * dx < float(p[1]) < 0.35 * dx and step % 5 == 0:
        print(f'   [band] step {step:6d}  y={float(p[1])/dx:+7.3f} dx  '
              f'vy={float(st.velocities[FLUID,1]):+8.4f}  '
              f'shift_y={float(shift[FLUID,1]):+8.4f}', flush=True)
    if float(p[0]) > xSpan - 0.1:
        break

traj = np.array(traj)
x, y, vx, vy = traj[:, 0], traj[:, 1], traj[:, 2], traj[:, 3]
print(f'closest approach to wall surface: {np.min(np.abs(y))/dx:.3f} dx; min y = {np.min(y)/dx:.2f} dx')
# bounce apexes: local maxima of y
apex = [(x[i], y[i]) for i in range(1, len(y) - 1) if y[i] >= y[i-1] and y[i] > y[i+1]]
print(f'launch  v = ({args.vx}, {args.vy})  g = {args.g}  dt = {args.dt}')
print(f'steps {len(traj)}   correction fired on {nCorr} steps')
print(f'travelled {x[-1]-x[0]:.3f} m tangentially in {len(traj)*args.dt:.3f} s')
print(f'\n{"bounce":>7} {"x [m]":>9} {"apex y [dx]":>12} {"vs launch":>10}')
y0 = args.y0
for i, (ax_, ay) in enumerate(apex[:12]):
    print(f'{i+1:7d} {ax_:9.3f} {ay/dx:12.2f} {ay/y0:10.3f}')
if len(apex) >= 2:
    r = apex[-1][1] / apex[0][1]
    print(f'\napex ratio last/first = {r:.4f} over {len(apex)} bounces '
          f'-> per-bounce restitution ~ {r**(1.0/max(1,len(apex)-1)):.4f}')
    print('1.000 = perfectly elastic: the particle never settles.')

os.makedirs(args.out, exist_ok=True)
fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                       gridspec_kw=dict(height_ratios=[2, 1]))
ax[0].plot(x, y / dx, lw=1.0)
ax[0].axhline(0, color='k', lw=1.2); ax[0].axhline(-0.5, color='0.5', lw=0.8, ls='--')
if apex:
    ax[0].plot([a[0] for a in apex], [a[1] / dx for a in apex], 'r.', ms=6,
               label='bounce apex')
    ax[0].legend(fontsize=8)
ax[0].set_ylabel('height  y/dx'); ax[0].grid(alpha=0.3)
ax[0].set_title(f'Ejected particle vs a free-slip wall under the mDBC no-penetration '
                f'correction\nlaunch v=({args.vx}, {args.vy}) m/s, g={args.g}  --  '
                f'{len(apex)} bounces, no visible decay = no dissipation')
ax[1].plot(x, vx, lw=1.0, label='v tangential')
ax[1].plot(x, vy, lw=1.0, label='v normal')
ax[1].set_xlabel('tangential position x [m]'); ax[1].set_ylabel('velocity [m/s]')
ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
fig.tight_layout()
p = os.path.join(args.out, 'nopen_trajectory.png')
fig.savefig(p, dpi=130); plt.close(fig)
print('->', p)
