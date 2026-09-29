"""CEILING_STICKING_PLAN.md §1: Marrone 3.1 delta+ nx67 fourtakas2019, jitter
seed 3 (the overnight worst seed), re-run with restartable checkpoints
(storeMode='states') and a per-step trace of every fluid row near the ceiling
or moving fast -- so the sticking and the ejected particles can be followed
by UID and then resumed in front of an event.

  python scratchpad/ceiling_run.py --out scripts/out_ceiling/seed3 [--resumeFrom ... --resumeStepOffset N --tLimit T]
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, os.path.join(ROOT, 'src'))

ap = argparse.ArgumentParser()
ap.add_argument('--out', default=os.path.join(ROOT, 'scripts', 'out_ceiling', 'seed3'))
ap.add_argument('--tLimit', type=float, default=1.9)
ap.add_argument('--seed', type=int, default=3)
ap.add_argument('--storeInterval', type=int, default=500)
ap.add_argument('--band', type=float, default=4.0, help='trace fluid rows within this many dx of the ceiling')
ap.add_argument('--vFast', type=float, default=6.0, help='... or faster than this [m/s]')
ap.add_argument('--plotInterval', type=int, default=20)
ap.add_argument('--resumeFrom', default=None)
ap.add_argument('--resumeStepOffset', type=int, default=0)
ap.add_argument('--tag', default='trace')
ap.add_argument('--patch', default=None, help='python file exec()d before the run (e.g. patch_timeCentredContinuity.py)')
args = ap.parse_args()

from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')
import numpy as np
import torch
from warpSPH.cases.dambreak import dambreakCase
import probe_deltaSPHMarrone as P

os.makedirs(args.out, exist_ok=True)
rec = {k: [] for k in ('step', 't', 'uid', 'x', 'y', 'vx', 'vy', 'rho', 'p', 'fs')}
geo = {}
_diag = dambreakCase.diagnostics
_step = [args.resumeStepOffset - 2]  # the initial (step -1) call comes first


def traceDiag(ctx, system, _d=_diag):
    row = dict(_d(ctx, system)) if _d is not None else {}
    state = system.state
    k = state.kinds
    pos = state.positions
    if 'yCeil' not in geo:
        dx = float(ctx.config.dx)
        # ceiling rows: boundary rows above the centre line and away from the
        # side walls; the wall surface is half a spacing below the first layer
        # (its ghost sits one spacing below it)
        b = pos[k == 1]
        top = b[(b[:, 1] > 0) & (b[:, 0].abs() < 0.95 * float(b[:, 0].abs().max()) - 6 * dx)]
        geo['yCeil'] = float(top[:, 1].min()) - 0.5 * dx
        geo['dx'] = dx
        print(f'[ceiling] yCeil={geo["yCeil"]:.5f} dx={dx:.5f}', flush=True)
    fl = k == 0
    speed = state.velocities.norm(dim=-1)
    sel = fl & ((pos[:, 1] > geo['yCeil'] - args.band * geo['dx']) | (speed > args.vFast))
    idx = torch.nonzero(sel).squeeze(-1)
    n = int(idx.numel())
    _step[0] += 1
    if n:
        t = float(system.t)
        rec['step'].append(np.full(n, _step[0], np.int32))
        rec['t'].append(np.full(n, t, np.float64))
        rec['uid'].append(state.UIDs[idx].cpu().numpy().astype(np.int64))
        p = pos[idx].cpu().numpy(); v = state.velocities[idx].cpu().numpy()
        rec['x'].append(p[:, 0]); rec['y'].append(p[:, 1])
        rec['vx'].append(v[:, 0]); rec['vy'].append(v[:, 1])
        rec['rho'].append(state.densities[idx].cpu().numpy())
        rec['p'].append(state.pressures[idx].cpu().numpy() if state.pressures is not None else np.full(n, np.nan))
        rec['fs'].append(state.surfaceIndicators[idx].cpu().numpy().astype(np.int8)
                         if state.surfaceIndicators is not None else np.full(n, -1, np.int8))
    row['nCeilTrace'] = n
    return row


dambreakCase.diagnostics = traceDiag
if args.patch:
    exec(open(args.patch).read())
watch = dict(store=True, storeMode='states', storeInterval=args.storeInterval,
             velocityAlarmPlotInterval=1, stallProgress=1e-3)
if args.resumeFrom:
    watch.update(resumeFrom=args.resumeFrom, resumeStepOffset=args.resumeStepOffset)
try:
    P._runOne(67, 40.0, args.tLimit, args.out, True, args.plotInterval,
              scheme='sun2017DeltaSPH', shifting='default',
              densityDiffusionTerm='fourtakas2019', jitter=1e-3, seed=args.seed,
              show=False, watch=watch)
finally:
    out = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in rec.items()}
    np.savez(os.path.join(args.out, f'{args.tag}.npz'), yCeil=geo.get('yCeil', np.nan),
             dx=geo.get('dx', np.nan), **out)
    print(f'[ceiling] trace -> {args.out}/{args.tag}.npz  rows={out["uid"].size}', flush=True)
