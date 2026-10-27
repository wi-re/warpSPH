#!/usr/bin/env python
"""E2 wave 2 D confirmation runs (COMPRESSIBLE_SPH_PLAN Track E): KH 256 f32, the case defaults, one
variant per call, video on (vispy), progress streamed.

    python .tmp/e2_kh256_confirm.py MFM base
    python .tmp/e2_kh256_confirm.py MFM guard1 faceGuard=1
    python .tmp/e2_kh256_confirm.py MFV prim mfvUpdate=primitive

`key=value` arguments are set on the scheme config (ints / floats / strings parsed); `--tLimit`,
`--nx`, `--no-video`, `--plotInterval` as usual. Per-row diagnostics add `nFast` (|v| > FAST), the
fliers count; a flushed `[confirm]` line every STATUS_EVERY rows gives max |v|, nFast, KE and dt.
"""
import argparse
import dataclasses
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import torch

torch.set_grad_enabled(False)

from warpSPH.cases.compressible.kelvinHelmholtz import kelvinHelmholtzCase
from warpSPH.runner import run

FAST = 1.5          # 3x the bulk shear speed (0.5); the sound speed is ~2
STATUS_EVERY = 200


def parse(v):
    for f in (int, float):
        try:
            return f(v)
        except ValueError:
            pass
    return {'True': True, 'False': False, 'None': None}.get(v, v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('scheme', choices=['MFM', 'MFV'])
    ap.add_argument('tag')
    ap.add_argument('cfg', nargs='*')
    ap.add_argument('--tLimit', type=float, default=3.2)
    ap.add_argument('--nx', type=int, default=256)
    ap.add_argument('--plotInterval', type=int, default=50)
    ap.add_argument('--no-video', dest='video', action='store_false')
    ap.add_argument('--out', default=str(REPO / 'results' / 'e2_kh256_confirm_2026-10-12'))
    ap.add_argument('--alarmScale', type=float, default=0.5)
    ap.add_argument('--alarmFactor', type=float, default=10.0)
    args = ap.parse_args()
    cfg = dict(kv.split('=', 1) for kv in args.cfg)
    cfg = {k: parse(v) for k, v in cfg.items()}

    counter = [0]
    base = kelvinHelmholtzCase

    def configureScheme(ctx):
        base.configureScheme(ctx)
        for k, v in cfg.items():
            if not hasattr(ctx.schemeConfig, k):
                raise SystemExit(f'unknown scheme option {k!r}')
            setattr(ctx.schemeConfig, k, v)
        print(f'[confirm] schemeConfig overrides: {cfg}', flush=True)

    def diagnostics(ctx, state):
        d = base.diagnostics(ctx, state)
        speed = torch.linalg.norm(state.state.velocities, dim=-1)
        d['nFast'] = float((speed > FAST).sum().item())
        counter[0] += 1
        if counter[0] % STATUS_EVERY == 1:
            print(f'[confirm] row {counter[0]} maxV={d["maxVelocity"]:.3f} nFast={int(d["nFast"])} '
                  f'KE={d.get("kineticEnergy", float("nan")):.5f}', flush=True)
        return d

    case = dataclasses.replace(base, configureScheme=configureScheme, diagnostics=diagnostics)
    runRoot = os.path.join(args.out, f'{args.scheme}_{args.tag}')
    os.makedirs(runRoot, exist_ok=True)
    kw = dict(scheme=args.scheme, nx=args.nx, tLimit=args.tLimit, precision=os.environ.get('warpSPHCore_PRECISION', 'float32'), quiet=True,
              progress=True, velocityScale=args.alarmScale, velocityAlarmFactor=args.alarmFactor,
              velocityAlarmPlotInterval=1, stallProgress=1e-3, exportRoot=runRoot)
    if args.video:
        kw.update(plot=True, video=True, plotInterval=args.plotInterval, plotBackend='vispy')
    print(f'[confirm] {args.scheme} {args.tag} nx={args.nx} tLimit={args.tLimit} video={args.video}', flush=True)
    t0 = time.perf_counter()
    r = run(case, **kw)
    wall = time.perf_counter() - t0
    rows = r.trajectory
    mv = [x['maxVelocity'] for x in rows]
    nf = [x['nFast'] for x in rows]
    te = [x['totalEnergy'] for x in rows]
    n = len(rows) - 1
    imax = max(range(len(mv)), key=mv.__getitem__)
    print(f'[confirm] DONE {args.scheme} {args.tag}: t_end={rows[-1]["t"]:.4f} steps={n} wall={wall:.1f}s '
          f'({1e3 * wall / max(n, 1):.2f} ms/step incl. video) diverged={r.diverged} stop={r.stopReason}', flush=True)
    print(f'[confirm] maxV peak={mv[imax]:.3f} at t={rows[imax]["t"]:.4f}; maxV(t>=0.75)>2 on '
          f'{sum(1 for x, y in zip(mv, rows) if y["t"] >= 0.75 and x > 2)} rows; '
          f'nFast peak={int(max(nf))} final={int(nf[-1])}; energyDrift='
          f'{abs(te[-1] - te[0]) / abs(te[0]):.3e}; video={getattr(r, "videoPath", None)}', flush=True)
    if True:
        rho = r.state.state.densities
        out = ((rho < 0.85) | (rho > 2.35)).float().mean().item()
        print(f'[confirm] density: min {rho.min().item():.3f} max {rho.max().item():.3f} p0.1/p99.9 '
              f'{torch.quantile(rho.float()[:2_000_000], 0.001).item():.3f}/{torch.quantile(rho.float()[:2_000_000], 0.999).item():.3f} '
              f'frac outside [0.85,2.35] = {100 * out:.3f}%', flush=True)
        torch.save(dict(positions=r.state.state.positions.cpu(), densities=rho.cpu(), velocities=r.state.state.velocities.cpu(),
                        t=rows[-1]['t']), os.path.join(runRoot, 'final.pt'))
    # coarse time series (t, maxV, nFast) every ~0.2 of simulated time
    nextT = 0.0
    for x in rows:
        if x['t'] >= nextT:
            print(f'[confirm]   t={x["t"]:.3f} maxV={x["maxVelocity"]:.3f} nFast={int(x["nFast"])} '
                  f'KE={x.get("kineticEnergy", float("nan")):.5f}', flush=True)
            nextT += 0.2


if __name__ == '__main__':
    main()
