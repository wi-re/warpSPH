"""Does the surface pressure probe read the right pressure ON the boundary?

A still-water hydrostatic column has the exact answer `p(z) = rho0 g (H - z)`
everywhere, `z` measured up from the bed. So this isolates the
`surfacePressureProbes` machinery (`dambreak.diagnostics`, added for Marrone
2011 Sec. 3.4 -- `DELTASPH_VALIDATION_PLAN.md` Sec. 5.2.2) from any scheme /
free-surface / impact confound: place probes exactly on the tank floor and the
two side walls (and a column of interior control points), let the pool sit, and
compare the probe reading to the analytic line.

Two things it separates:

  * **the field vs the probe** -- the per-particle `p` is scattered behind the
    probe markers, so if the bulk pressure field itself is low the probe is not
    to blame;
  * **MLS vs Shepard tier** -- `pSurf{k}WC` records whether each probe got the
    first-order MLS fit or fell back to the 0th-order Shepard gather. Shepard is
    a plain average of a *one-sided* neighbourhood, so against a real pressure
    gradient it reads biased low by ~ the offset from the wall point to the
    neighbour centroid (~1-2 dx of head). A probe exactly on the boundary is
    the worst case for conditioning; `--inset` (in dx, along the into-fluid
    normal) backs it off.

Drives `dambreak` as a still pool (`fluidWidth = 1.0`, no obstacle,
`hydrostaticInit = True`, plain delta-SPH, no PST) -- the same vehicle
`probe_englishWedge.py` uses for English Sec. 4.1.

Usage
-----
  python scripts/probe_hydrostaticPressureProbe.py --dp 0.02
  python scripts/probe_hydrostaticPressureProbe.py --dp 0.02 --inset 0 0.25 0.5 1.0
  python scripts/probe_hydrostaticPressureProbe.py --report

Default --out: scripts/out_hydrostaticPressureProbe/
"""
from __future__ import annotations

import argparse
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, 'out_hydrostaticPressureProbe')

# -- geometry (SI-ish, non-dimensional density), matching probe_englishWedge ----
TANK_W = 2.4
TANK_H = 1.2           # `L` in dambreak terms
WATER_H = 0.5          # still-water depth H
G = 9.81
C0_RATIO = 20.0        # c0 = C0_RATIO * sqrt(g H); still water needs no headroom


def _probeLayout(nx):
    """(name, group, x, y, nx, ny) rows in centred-domain coords (x from tank
    centre, y from tank centre; bed at y = -TANK_H/2, still surface at
    y = -TANK_H/2 + WATER_H). Normals point into the fluid."""
    import numpy as np
    dx = TANK_H / nx
    xw = TANK_W / 2.0
    bed = -TANK_H / 2.0
    surf = bed + WATER_H
    # depths sampled up each wall: a bit off the very corners, up to near the surface
    zs = np.array([0.02, 0.08, 0.16, 0.26, 0.36, 0.46]) * (WATER_H / 0.5)
    xs = np.array([-0.7, -0.35, 0.0, 0.35, 0.7]) * xw

    rows = []
    for x in xs:                                   # floor
        rows.append(('floor@x=%+.2f' % x, 'floor', x, bed, 0.0, 1.0))
    for z in zs:                                   # left wall (x = -W/2)
        rows.append(('leftWall@z=%.2f' % z, 'leftWall', -xw, bed + z, 1.0, 0.0))
    for z in zs:                                   # right wall (x = +W/2)
        rows.append(('rightWall@z=%.2f' % z, 'rightWall', xw, bed + z, -1.0, 0.0))
    for z in zs:                                   # interior control column (x = 0)
        rows.append(('interior@z=%.2f' % z, 'interior', 0.0, bed + z, 0.0, 1.0))
    # the two bottom corners, exact
    rows.append(('cornerBL', 'corner', -xw, bed, 0.7071, 0.7071))
    rows.append(('cornerBR', 'corner', xw, bed, -0.7071, 0.7071))
    return rows


def _analyticStar(y):
    """p / (rho0 g H) for the hydrostatic column at centred-domain height y."""
    bed = -TANK_H / 2.0
    z = y - bed
    return max(0.0, (WATER_H - z) / WATER_H)


def _runOne(dp, insets, tLimit, c0Ratio, out):
    from warpSPHBootstrap import bootstrap
    bootstrap(precision='float32')
    import numpy as np
    from warpSPH.cases.dambreak import dambreakCase
    from warpSPH.runner import run

    os.makedirs(out, exist_ok=True)
    nx = int(round(TANK_H / dp))
    sqrt_gH = (G * WATER_H) ** 0.5
    layout = _probeLayout(nx)
    surfProbes = [[float(x), float(y), float(a), float(b)]
                  for (_n, _g, x, y, a, b) in layout]

    for inset in insets:
        tag = f'dp{dp:g}_inset{inset:g}'
        params = dict(
            W=TANK_W,
            fillRatio=WATER_H / TANK_H,
            fluidWidth=1.0,
            gravityMagnitude=G,
            disableGravity=False,
            referenceVelocity=sqrt_gH,
            machTarget=1.0 / c0Ratio,
            shifting='off',
            hydrostaticInit=True,
            pressureProbeHeights=[],
            surfacePressureProbes=surfProbes,
            surfacePressureProbeInset=float(inset),
        )
        kw = dict(scheme='deltaSPH', L=TANK_H, nx=nx, tLimit=tLimit,
                  quiet=True, store=False, progress=True, params=params)
        print(f'[{tag}] dp={dp} nx={nx} inset={inset} dx  '
              f'c0={c0Ratio:g}sqrt(gH)={c0Ratio * sqrt_gH:.1f}  -> t={tLimit}s ...',
              flush=True)
        r = run(dambreakCase, **kw)

        rows = [x for x in r.trajectory if x.get('step', -2) >= -1]
        np_ = len(layout)
        keys = ['step', 't', 'kineticEnergy', 'maxVelocity', 'maxDensity', 'minDensity']
        for k in range(np_):
            keys += [f'pSurf{k}', f'pSurf{k}Nnbr', f'pSurf{k}WC']
        cols = {k: np.array([row.get(k, np.nan) for row in rows], float) for k in keys}

        dx = float(r.ctx.config.dx)
        c0 = float(getattr(r.ctx.schemeConfig.fluid, 'fixedSoundSpeed', 0.0) or 0.0)
        rho0 = float(getattr(r.ctx.schemeConfig.fluid, 'restDensity', 1.0) or 1.0)
        pRef = rho0 * G * WATER_H

        # final per-particle field, for the scatter behind the probes
        st = r.state.state
        pos = st.positions.detach().cpu().numpy()
        kinds = st.kinds.detach().cpu().numpy()
        pres = (st.pressures.detach().cpu().numpy() if st.pressures is not None
                else np.full(len(pos), np.nan))
        fl = kinds == 0
        bed = -TANK_H / 2.0

        meta = dict(
            tag=tag, dp=float(dp), nx=nx, dx=dx, inset=float(inset),
            c0=c0, c0Ratio=float(c0Ratio), rho0=rho0, pRef=pRef,
            G=G, TANK_W=TANK_W, TANK_H=TANK_H, WATER_H=WATER_H, bed=bed,
            tLimit=float(tLimit), tReached=float(rows[-1].get('t', 0.0)) if rows else 0.0,
            diverged=bool(r.diverged), nSteps=int(r.nSteps),
            names=[n for (n, *_r) in layout],
            groups=[g for (_n, g, *_r) in layout],
            px=[x for (_n, _g, x, *_r) in layout],
            py=[y for (_n, _g, _x, y, *_r) in layout],
            analyticStar=[_analyticStar(y) for (_n, _g, _x, y, *_r) in layout],
        )
        npz = os.path.join(out, tag + '.npz')
        np.savez(npz, meta=json.dumps(meta),
                 fx=pos[fl, 0].astype('f4'), fy=pos[fl, 1].astype('f4'),
                 fp=pres[fl].astype('f4'),
                 **{f's_{k}': v for k, v in cols.items()})
        _summarise(meta, cols)
        print(f'[{tag}] -> {npz}  diverged={r.diverged}  steps={r.nSteps}', flush=True)


def _lateMean(v, t, frac=0.2):
    import numpy as np
    v = np.asarray(v, float)
    if v.size == 0:
        return float('nan')
    n = max(1, int(round(v.size * frac)))
    return float(np.nanmean(v[-n:]))


def _summarise(meta, cols):
    import numpy as np
    t = cols['t']
    names, groups = meta['names'], meta['groups']
    aStar = meta['analyticStar']
    pRef = meta['pRef']
    print(f"  {'probe':<20} {'grp':<10} {'z/H':>5}  {'meas*(t~0)':>11}  "
          f"{'meas*(late)':>12}  {'exact*':>7}  {'err%':>6}  {'nnbr':>4} {'WC':>3}")
    worst = {}
    for k, (nm, gp, a) in enumerate(zip(names, groups, aStar)):
        p0 = cols[f'pSurf{k}'][0] / pRef if cols[f'pSurf{k}'].size else float('nan')
        pl = _lateMean(cols[f'pSurf{k}'], t) / pRef
        nn = _lateMean(cols[f'pSurf{k}Nnbr'], t)
        wc = _lateMean(cols[f'pSurf{k}WC'], t)
        zH = (meta['py'][k] - meta['bed']) / meta['WATER_H']
        err = (pl - a) / a * 100.0 if a > 1e-6 else float('nan')
        print(f"  {nm:<20} {gp:<10} {zH:5.2f}  {p0:11.3f}  {pl:12.3f}  "
              f"{a:7.3f}  {err:6.1f}  {nn:4.0f} {wc:3.1f}")
        worst[gp] = max(worst.get(gp, 0.0), abs(err) if np.isfinite(err) else 0.0)
    print('  worst |err%| by group:', {k: round(v, 1) for k, v in worst.items()})


def _report(out):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    runs = []
    for npz in sorted(glob.glob(os.path.join(out, '*.npz'))):
        d = np.load(npz, allow_pickle=True)
        meta = json.loads(str(d['meta']))
        cols = {k[2:]: d[k] for k in d.files if k.startswith('s_')}
        runs.append((meta, cols, d))
    if not runs:
        print(f'no runs in {out}'); return

    gcol = {'floor': '#d62728', 'leftWall': '#1f77b4', 'rightWall': '#2ca02c',
            'interior': '#7f7f7f', 'corner': '#9467bd'}
    n = len(runs)
    fig, axes = plt.subplots(1, n, figsize=(1 + 5.2 * n, 6.2), squeeze=False)
    for ax, (meta, cols, d) in zip(axes[0], runs):
        H = meta['WATER_H']; bed = meta['bed']; pRef = meta['pRef']
        t = cols['t']
        # per-particle field scatter
        zf = (d['fy'] - bed) / H
        ax.scatter(d['fp'] / pRef, zf, s=3, c='#cccccc', alpha=0.5,
                   label='fluid particles', zorder=1)
        # analytic
        zz = np.linspace(0, 1.05, 50)
        ax.plot(np.clip(1 - zz, 0, None), zz, 'k-', lw=1.2, label='rho0 g (H - z)', zorder=2)
        # probes: late-time mean, coloured by group
        seen = set()
        for k, (gp, a) in enumerate(zip(meta['groups'], meta['analyticStar'])):
            pl = _lateMean(cols[f'pSurf{k}'], t) / pRef
            zH = (meta['py'][k] - bed) / H
            wc = _lateMean(cols[f'pSurf{k}WC'], t)
            lab = gp if gp not in seen else None
            seen.add(gp)
            ax.scatter([pl], [zH], s=70, c=gcol.get(gp, 'k'),
                       marker=('o' if wc > 0.5 else 'X'),
                       edgecolors='k', linewidths=0.6, label=lab, zorder=4)
        ax.set_title(f"dp = {meta['dp']:g}  (H/dx = {H / meta['dx']:.0f}),  "
                     f"inset = {meta['inset']:g} dx", fontsize=10)
        ax.set_xlabel('p / (rho0 g H)'); ax.set_ylabel('z / H')
        ax.set_xlim(-0.05, 1.15); ax.set_ylim(-0.02, 1.1)
        ax.grid(alpha=0.25); ax.legend(fontsize=7, loc='upper right')
    fig.suptitle('Hydrostatic still-water column -- surface pressure probe vs the analytic line\n'
                 '(o = first-order MLS tier, X = Shepard fallback)', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = os.path.join(out, 'hydrostatic_probe.png')
    fig.savefig(p, dpi=130)
    print('->', p)

    lines = ['# Hydrostatic still-water column -- surface pressure probe check', '']
    for meta, cols, d in runs:
        t = cols['t']
        lines.append(f"## `{meta['tag']}`  (H/dx = {meta['WATER_H'] / meta['dx']:.0f}, "
                     f"inset = {meta['inset']:g} dx, c0 = {meta['c0']:.1f}, "
                     f"diverged = {meta['diverged']}, t = {meta['tReached']:.2f}/{meta['tLimit']:g}s)")
        lines.append('')
        lines.append('| probe | group | z/H | meas* (t~0) | meas* (late) | exact* | err % | nnbr | WC |')
        lines.append('|---|---|--:|--:|--:|--:|--:|--:|--:|')
        gerr = {}
        for k, (nm, gp, a) in enumerate(zip(meta['names'], meta['groups'], meta['analyticStar'])):
            p0 = cols[f'pSurf{k}'][0] / meta['pRef']
            pl = _lateMean(cols[f'pSurf{k}'], t) / meta['pRef']
            nn = _lateMean(cols[f'pSurf{k}Nnbr'], t)
            wc = _lateMean(cols[f'pSurf{k}WC'], t)
            zH = (meta['py'][k] - meta['bed']) / meta['WATER_H']
            err = (pl - a) / a * 100.0 if a > 1e-6 else float('nan')
            gerr.setdefault(gp, []).append(abs(err) if err == err else 0.0)
            lines.append(f"| {nm} | {gp} | {zH:.2f} | {p0:.3f} | {pl:.3f} | {a:.3f} | "
                         f"{err:+.1f} | {nn:.0f} | {wc:.1f} |")
        lines.append('')
        lines.append('worst |err %| by group: '
                     + ', '.join(f'{g} {max(v):.1f}' for g, v in gerr.items()))
        lines.append('')
    md = os.path.join(out, 'REPORT.md')
    open(md, 'w').write('\n'.join(lines) + '\n')
    print('->', md)
    print('\n'.join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dp', type=float, default=0.02, help='particle spacing (dx = L/nx)')
    ap.add_argument('--inset', type=float, nargs='+', default=[0.0, 0.5],
                    help='probe insets in dx along the into-fluid normal (one run each)')
    ap.add_argument('--tLimit', type=float, default=2.0, help='settle time (s)')
    ap.add_argument('--c0Ratio', type=float, default=C0_RATIO)
    ap.add_argument('--out', default=DEFAULT_OUT)
    ap.add_argument('--report', action='store_true')
    args = ap.parse_args(argv)

    if args.report:
        _report(args.out); return
    _runOne(args.dp, args.inset, args.tLimit, args.c0Ratio, args.out)
    _report(args.out)


if __name__ == '__main__':
    main()
