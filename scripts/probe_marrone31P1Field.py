"""Marrone 3.1 P1 plateau: is the near-wall fluid really at higher pressure,
or is the on-wall MLS probe misreading a disordered neighbourhood?

The default delta-SPH no-PST config reads a P1 plateau of 0.77-0.82 rho g H
over 4 realisations (Buchner ~0.55), while delta+ (sun2017DeltaSPH, PST on)
reads 0.46 (`scripts/out_baseline_2026-09-26/`). The case probe is a
first-order MLS fit of the *fluid* pressures evaluated AT the wall (inset 0,
clamped >= 0, 7-point disc quadrature), i.e. a one-sided extrapolation.

This runs `probe_deltaSPHMarrone._runOne` unchanged (video on) and, from t* =
3.4, every `--every` steps records around P1 (z = 160 mm, disc radius 45 mm):
  - the case probe value (pProbe0Star);
  - single-point MLS at the wall and 1 / 2 / 4 dx into the fluid (+ Shepard,
    neighbour count, well-conditioned flag, fitted dp/dx);
  - fluid particle pressure / density stats in two bands at the disc height:
    0-4 dx and 4-8 dx from the wall;
  - the wall particles' own (mDBC) pressure in the disc height range;
  - the water height at the wall (hydrostatic reference rho0 g (y_s - z_P1)).
All pressures in rho0 g H. Rows stream to stdout and to <out>/<tag>_p1field.json.

Usage:
  python scripts/probe_marrone31P1Field.py --scheme deltaSPH --shifting off
  python scripts/probe_marrone31P1Field.py --scheme sun2017DeltaSPH --shifting default
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scheme', default='deltaSPH')
    ap.add_argument('--shifting', default='off', choices=('default', 'off', 'on'))
    ap.add_argument('--nx', type=int, default=67)
    ap.add_argument('--tStarEnd', type=float, default=6.0)
    ap.add_argument('--tStarStart', type=float, default=3.4)
    ap.add_argument('--every', type=int, default=25)
    ap.add_argument('--out', default=os.path.join(HERE, 'out_baseline_2026-09-26', 'p1field'))
    args = ap.parse_args()

    from warpSPHBootstrap import bootstrap
    bootstrap(precision='float32')
    import torch
    import probe_deltaSPHMarrone as M
    from warpSPH.cases.dambreak import dambreakCase
    from warpSPH.modules.liu import interpolateLiuLiu
    from warpSPHCore import OperationDirection

    os.makedirs(args.out, exist_ok=True)
    tag = f'{args.scheme}_nx{args.nx}_pst-{args.shifting}'
    rows = []
    baseDiag = dambreakCase.diagnostics
    counter = {'n': 0}

    def diagnostics(ctx, state):
        d = baseDiag(ctx, state)
        counter['n'] += 1
        tStar = d.get('tStar', 0.0)
        if tStar < args.tStarStart or counter['n'] % args.every:
            return d
        with torch.no_grad():
            P = state.state
            interior = ctx.scratch['interiorDomain']
            dx = float(ctx.config.dx)
            rho0 = float(ctx.schemeConfig.fluid.restDensity)
            ref = rho0 * M.G * M.H
            xWall = float(interior.max[0].item())
            yBed = float(interior.min[1].item())
            zP1 = yBed + M.PROBE_HEIGHTS[0]
            R = M.PROBE_DISC_RADIUS
            pos, p, rho, kinds = P.positions, P.pressures, P.densities, P.kinds
            fluid = kinds == 0
            dist = xWall - pos[:, 0]
            inZ = (pos[:, 1] - zP1).abs() <= R

            def stats(mask, q):
                n = int(mask.sum())
                if n == 0:
                    return dict(n=0, mean=float('nan'), med=float('nan'), std=float('nan'))
                v = q[mask].float()
                return dict(n=n, mean=float(v.mean()), med=float(v.median()),
                            std=float(v.std()) if n > 1 else 0.0)

            near = fluid & inZ & (dist >= -0.5 * dx) & (dist < 4 * dx)
            mid = fluid & inZ & (dist >= 4 * dx) & (dist < 8 * dx)
            # wall rows facing P1: boundary particles outside the interior, level with the disc
            wallRows = (~fluid) & inZ & (pos[:, 0] > xWall) & (pos[:, 0] < xWall + 3 * dx)
            # water height at the wall
            col = fluid & (dist < 4 * dx)
            ySurf = float(pos[col, 1].max()) if bool(col.any()) else float('nan')

            # single-point MLS into the fluid, same call the case probe makes
            import copy as _copy
            cfg = _copy.copy(ctx.config)
            dom = ctx.config.domain
            cfg.domain = type(dom)(dom.min, dom.max, torch.zeros_like(dom.periodic), dom.dim)
            offs = [0.0, 1.0, 2.0, 4.0]
            pts = torch.tensor([[xWall - k * dx, zP1] for k in offs],
                               device=pos.device, dtype=pos.dtype)
            v, g, nn, A_g, b, wc = interpolateLiuLiu(
                pts, referenceParticles=P, referenceQuantities=p, config=cfg,
                neighbor_threshold=4, direction=OperationDirection.FluidToFluid, supportScale=1.0)
            shep = torch.where(A_g[:, 0, 0] > 0, b[:, 0] / A_g[:, 0, 0].clamp_min(1e-12),
                               torch.zeros_like(A_g[:, 0, 0]))

            row = dict(
                step=counter['n'], tStar=tStar, probe=d.get('pProbe0Star'),
                hydro=rho0 * M.G * (ySurf - zP1) / ref, ySurfOverH=(ySurf - yBed) / M.H,
                nearP={k: (x / ref if k in ('mean', 'med', 'std') else x)
                       for k, x in stats(near, p).items()},
                midP={k: (x / ref if k in ('mean', 'med', 'std') else x)
                      for k, x in stats(mid, p).items()},
                nearRho=stats(near, rho), midRho=stats(mid, rho),
                wallP={k: (x / ref if k in ('mean', 'med', 'std') else x)
                       for k, x in stats(wallRows, p).items()},
                mls=[float(x) / ref for x in v.cpu()],
                mlsWC=[bool(x) for x in wc.cpu()],
                mlsN=[int(x) for x in nn.cpu()],
                mlsDpdx=[float(x) / ref * M.H for x in g[:, 0].cpu()],
                shep=[float(x) / ref for x in shep.cpu()],
            )
        rows.append(row)
        if len(rows) % 10 == 1:
            print(f"[p1] t*={tStar:.2f} probe={row['probe']:.3f} "
                  f"mls@0/1/2/4dx={'/'.join(f'{x:.3f}' for x in row['mls'])} "
                  f"wc={''.join('1' if x else '0' for x in row['mlsWC'])} "
                  f"near p mean/med/std={row['nearP']['mean']:.3f}/{row['nearP']['med']:.3f}/"
                  f"{row['nearP']['std']:.3f} (n={row['nearP']['n']}) "
                  f"mid={row['midP']['mean']:.3f} wall={row['wallP']['mean']:.3f} "
                  f"hydro={row['hydro']:.3f} rho near/mid={row['nearRho']['mean']:.4f}/"
                  f"{row['midRho']['mean']:.4f}", flush=True)
            with open(os.path.join(args.out, tag + '_p1field.json'), 'w') as f:
                json.dump(rows, f)
        return d

    dambreakCase.diagnostics = diagnostics
    tLimit = args.tStarEnd / (M.G / M.H) ** 0.5
    M._runOne(args.nx, 40.0, tLimit, args.out, True, 20, scheme=args.scheme,
              shifting=args.shifting)
    with open(os.path.join(args.out, tag + '_p1field.json'), 'w') as f:
        json.dump(rows, f)
    print(f'[p1] wrote {len(rows)} rows', flush=True)


if __name__ == '__main__':
    main()
