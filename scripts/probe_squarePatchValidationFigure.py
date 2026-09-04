"""Validation figure for the rotating square patch (`docs/historic_plans/WCSPH_SHIFTING_PLAN.md`
§3/§4): a grid of particle snapshots — rows are shift treatments, columns are
time instants — coloured by pressure, in the style of Sun et al. 2019 Fig. 13.

The physical answer is an area-conserving rigid rotation whose free surface
grows four arms from the corners; a working surface shift keeps the bulk
regular and the arms clean without inflating the footprint.

Modes: `shiftOff`, `surfaceZeroed` (legacy `mat` hard-zero), `surfaceNormal`
(the Sun 2019 Eq. 20-21 treatment, now the default).

Usage:
  python scripts/probe_squarePatchValidationFigure.py [--nx 400]
      [--times 0.25 0.5 0.75 1.0] [--modes shiftOff surfaceZeroed surfaceNormal]
      [--field pressure|density] [--out <path.png>]
"""
from __future__ import annotations

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--nx', type=int, default=400)
parser.add_argument('--times', type=float, nargs='+', default=[0.25, 0.5, 0.75, 1.0])
parser.add_argument('--modes', nargs='+',
                    default=['shiftOff', 'surfaceZeroed', 'surfaceNormal'])
parser.add_argument('--field', default='pressure', choices=['pressure', 'density'])
parser.add_argument('--omega', type=float, default=4.0)
parser.add_argument('--out', default=None)
parser.add_argument('--markerSize', type=float, default=1.0,
                    help='scatter marker size in points^2 (small: individual particles visible)')
parser.add_argument('--clim', type=float, default=None,
                    help='fixed symmetric colour limit (pressure) / half-range; '
                         'default = p90 of |field| over the non-fragmented frames')
parser.add_argument('--dpi', type=int, default=220)
parser.add_argument('--fromCache', default=None,
                    help='replot from a previously written <out>.npz instead of re-running the sims')
args = parser.parse_args()

from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from warpSPH.cases.rotatingSquarePatch import rotatingSquarePatchCase as case
from warpSPH.runner import run
from warpSPH.configurations.moduleConfigurations.shifting import ShiftingProjectionScheme

SCRATCH = os.environ.get('CLAUDE_SCRATCH', '.')
OUT = args.out or os.path.join(SCRATCH, f'squarePatch_validation_nx{args.nx}_{args.field}.png')


def _configure(mode):
    base = case.configureScheme

    def wrapped(ctx):
        base(ctx)
        sc = ctx.schemeConfig
        if mode == 'shiftOff':
            sc.shiftProperties.active = False
        elif mode == 'surfaceZeroed':
            sc.shiftProperties.projectionScheme = ShiftingProjectionScheme.mat
        elif mode == 'surfaceNormal':
            sc.shiftProperties.projectionScheme = ShiftingProjectionScheme.surfaceNormal
        elif mode == 'surfaceNormalDeltaU':
            sc.shiftProperties.projectionScheme = ShiftingProjectionScheme.surfaceNormal
            sc.shiftProperties.correctdrhodt = True

    return wrapped


def _snapshot(state, field):
    p = state.state
    fluid = (p.kinds == 0).detach().cpu().numpy()
    xy = p.positions.detach().cpu().numpy()[fluid]
    if field == 'pressure' and getattr(p, 'pressures', None) is not None:
        c = p.pressures.detach().cpu().numpy()[fluid]
        if not np.any(np.abs(c) > 0):
            c = p.densities.detach().cpu().numpy()[fluid]
    else:
        c = p.densities.detach().cpu().numpy()[fluid]
    return xy, c


def run_mode(mode):
    targets = sorted(args.times)
    frames = {}                       # t -> (xy, c)
    pending = list(targets)

    def postStep(ctx, state, i):
        t = float(state.t)
        while pending and t >= pending[0] - 1e-9:
            frames[pending.pop(0)] = _snapshot(state, args.field)

    _origPost, _origCfg = case.postStep, case.configureScheme
    case.postStep = postStep
    case.configureScheme = _configure(mode)
    try:
        r = run(case, params={'shape': 'box', 'omega': args.omega},
                nx=args.nx, tLimit=max(targets) * 1.05, nSteps=None,
                store=False, plot=False, quiet=True, progress=False)
    finally:
        case.postStep, case.configureScheme = _origPost, _origCfg
    # the step-limited loop can stop a hair short of the last target
    for t in pending:
        frames[t] = _snapshot(r.state, args.field)
    return frames


CACHE = (args.fromCache or (os.path.splitext(OUT)[0] + '.npz'))

if args.fromCache:
    blob = np.load(args.fromCache, allow_pickle=True)
    data = blob['data'].item()
    args.modes = blob['modes'].tolist()
    args.times = blob['times'].tolist()
    if 'nx' in blob:
        args.nx = int(blob['nx'])
    cachedField = str(blob['field']) if 'field' in blob else None
    if cachedField and cachedField != args.field:
        print(f'NOTE: cache holds the "{cachedField}" field; --field {args.field} is '
              f'ignored on a replot (re-run without --fromCache to switch fields).')
        args.field = cachedField
    print(f'replotting from {args.fromCache}')
else:
    data = {m: run_mode(m) for m in args.modes}
    np.savez(CACHE, data=np.array(data, dtype=object),
             modes=np.array(args.modes), times=np.array(sorted(args.times)),
             nx=args.nx, field=args.field)
    print(f'wrote {CACHE}  (--fromCache it to retune the figure without re-simulating)')

# shared colour scale. The fragmented tω=4 frame is a pressure-noise firehose
# that would wash out every earlier panel, so the auto scale is taken from the
# earlier (still-coherent) frames only.
tmax = max(sorted(args.times))
coherent = np.concatenate([c for m in data for (t, (_, c)) in data[m].items() if t < tmax]) \
    if len(args.times) > 1 else np.concatenate([c for m in data for (_, c) in data[m].values()])
if args.field == 'pressure':
    lim = args.clim if args.clim is not None else float(np.percentile(np.abs(coherent), 90))
    vmin, vmax, cmap = -lim, lim, 'RdBu_r'
else:
    vmin, vmax, cmap = np.percentile(coherent, 1), np.percentile(coherent, 99), 'viridis'

times = sorted(args.times)
nr, nc = len(args.modes), len(times)
fig, axes = plt.subplots(nr, nc, figsize=(3.1 * nc, 3.1 * nr), squeeze=False)
ext = 0
for m in data:
    for xy, _ in data[m].values():
        ext = max(ext, np.abs(xy).max())
ext *= 1.05

for r, mode in enumerate(args.modes):
    for cc, t in enumerate(times):
        ax = axes[r][cc]
        if t in data[mode]:
            xy, c = data[mode][t]
            ax.scatter(xy[:, 0], xy[:, 1], c=c, s=args.markerSize, vmin=vmin, vmax=vmax,
                       cmap=cmap, linewidths=0, rasterized=True)
        ax.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], 'k-', lw=0.5, alpha=0.4)
        ax.set_xlim(-ext, ext)
        ax.set_ylim(-ext, ext)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        if r == 0:
            ax.set_title(f'tω = {t * args.omega:g}   (t = {t:g})', fontsize=10)
        if cc == 0:
            ax.set_ylabel(mode, fontsize=11)

fig.suptitle(f'Rotating square patch — {args.field}, nx = {args.nx} '
             f'(L/Δx ≈ {int(2 / (6.0 / args.nx))})', fontsize=12)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.6, label=args.field)
fig.savefig(OUT, dpi=args.dpi, bbox_inches='tight')
print(f'wrote {OUT}')
