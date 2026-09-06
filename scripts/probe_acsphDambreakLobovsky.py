"""Probe (`ACSPH_PLAN.md` Part 8 step 9 / Part 7 §4.5): the `dambreak` case at
Lobovsky et al. 2014's own tank scale, with wall pressure probes at that
paper's five-sensor matrix, scored against its Table 2 peak-pressure
percentiles (100 experimental runs, H = 300 mm series).

Reference geometry (`literature/lobovsky2014_experimental-dam-break-pressure-loads.pdf`)
------------------------------------------------------------------------------
- Tank inner box 1610 mm long x 600 mm tall (2D; the 150 mm breadth was chosen
  so the flow stays quasi-2D).
- Dam gate 600 mm from the left wall -> reservoir 600 mm wide, 1000 mm of dry
  bed downstream.
- Water column 600 mm wide x 300 mm tall (H = 300 mm).  g = 9.81.
- Sensors on the vertical downstream wall, centreline, at z = 3 / 15 / 30 /
  80 mm above the bed (sensor 2L is sensor 2 off-centre -- 3D only, dropped).
- Non-dimensionalisation: t* = t sqrt(g/H),  P* = P / (rho0 g H).

Modelling choices, all standard for this benchmark and noted in the report:
- Instantaneous dam removal (no gate model).  Lobovsky measure a ~0.069 s
  (t* ~ 0.4) gate motion and argue it is fast enough to disregard.
- 2D.  No-slip tank walls (the case's own default).
- The case runs in code units with rho0 = 1; P* divides that out, so the
  comparison is unit-consistent without setting a physical density.

Usage
-----
  # one scheme -> writes <out>/<scheme>_nx<N>.npz (+ video if --video)
  python scripts/probe_acsphDambreakLobovsky.py --scheme deltaSPH --nx 128 --video
  python scripts/probe_acsphDambreakLobovsky.py --scheme artificialCompressible --nx 128 --video

  # combine whatever .npz runs exist under <out> into plots + REPORT.md
  python scripts/probe_acsphDambreakLobovsky.py --report

Default --out: scripts/out_acsphDambreakLobovsky/
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, 'out_acsphDambreakLobovsky')

# --- Lobovsky et al. 2014, Table 2 (H = 300 mm, 100 runs) -------------------
# z (mm) ; median / 2.5% / 97.5% peak pressure in mbar.  rho g H = 997*9.81*0.3
# = 2934.17 Pa = 29.3417 mbar is the reservoir-bottom hydrostatic used for P*.
_RHO_G_H_MBAR = 997.0 * 9.81 * 0.3 / 100.0
_LOBOVSKY_MBAR = [
    # z_mm, median, p2.5, p97.5
    (3.0,  91.44, 75.06, 130.98),
    (15.0, 57.49, 53.26,  63.31),
    (30.0, 43.76, 39.79,  49.62),
    (80.0, 30.15, 24.18,  36.32),
]
SENSORS = [
    dict(name=f'P{i+1}', z_mm=z,
         median=m / _RHO_G_H_MBAR, lo=lo / _RHO_G_H_MBAR, hi=hi / _RHO_G_H_MBAR)
    for i, (z, m, lo, hi) in enumerate(_LOBOVSKY_MBAR)
]

# --- reference tank, in metres --------------------------------------------------
TANK_W = 1.61          # tank length  (case param W)
TANK_L = 0.60          # tank height  (case spec L, the reference length)
H_FILL = 0.30          # water depth  -> fillRatio = H_FILL / TANK_L
RES_W = 0.60           # reservoir width -> fluidWidth = RES_W / TANK_W
G = 9.81
PROBE_HEIGHTS = [s['z_mm'] / 1000.0 for s in SENSORS]


def _runOne(scheme: str, nx: int, tLimit: float, out: str, video: bool,
            plotInterval: int, accelConstraint: bool = True,
            targetDt: float = None, wallPeriodic: bool = False,
            machTarget: float = 0.1):
    from warpSPHBootstrap import bootstrap
    bootstrap(precision='float32')
    import numpy as np
    from warpSPH.cases.dambreak import dambreakCase
    from warpSPH.runner import run

    os.makedirs(out, exist_ok=True)

    # Sound speed: `machTarget` (default 0.1) sets `c0 = sqrt(2 g H) / Ma` per
    # Sun et al. 2017 Eq. (2) -- genuinely weakly compressible at any Dx, and
    # dt then adapts each step (Sun Eq. 5). `--targetDt` overrides with the
    # legacy back-solve (`c0 ~ 1/dx`, Mach drifts past 0.1 at fine Dx). See
    # `DELTASPH_VALIDATION_PLAN.md`.
    explicitDt = targetDt is not None

    tag = f'{scheme}_nx{nx}' + ('_periodic' if wallPeriodic else '')
    if explicitDt:
        tag += '_dt' + f'{targetDt * 1e4:g}'.replace('.', 'p') + 'e-4'
    else:
        tag += f'_Ma{machTarget:.2f}'.replace('.', 'p')
    runRoot = os.path.join(out, tag + '_run')

    params = dict(
        W=TANK_W, fillRatio=H_FILL / TANK_L, fluidWidth=RES_W / TANK_W,
        gravityMagnitude=G,
        pressureProbeHeights=PROBE_HEIGHTS, pressureProbeInset=0.0,
        acAccelConstraint=accelConstraint, wallPeriodic=wallPeriodic,
    )
    if explicitDt:
        params['targetDt'] = targetDt
    else:
        params['machTarget'] = machTarget

    kw = dict(
        scheme=scheme, L=TANK_L, nx=nx, tLimit=tLimit,
        quiet=True, store=False, progress=True, params=params,
    )
    if video:
        kw.update(plot=True, video=True, plotBackend='matplotlib',
                  plotInterval=plotInterval, exportRoot=runRoot)

    print(f'[{tag}] running to t={tLimit}s ...', flush=True)
    r = run(dambreakCase, **kw)

    rows = [x for x in r.trajectory if x.get('step', -2) >= -1]
    keys = ['step', 't', 'tStar', 'kineticEnergy', 'maxVelocity',
            'minDensity', 'maxDensity', 'nPenetrating', 'maxPenetrationDx']
    for k in range(len(SENSORS)):
        keys += [f'pProbe{k}', f'pProbe{k}Star', f'pProbe{k}Nnbr']
    cols = {k: np.array([row.get(k, np.nan) for row in rows], dtype=float)
            for k in keys}

    dx = float(r.ctx.config.dx)
    tReached = float(rows[-1].get('t', 0.0)) if rows else 0.0
    c0 = float(getattr(r.ctx.schemeConfig.fluid, 'fixedSoundSpeed', 0.0) or 0.0)
    meta = dict(scheme=scheme, nx=nx, tLimit=tLimit, tReached=tReached,
                dx=dx, HdxRatio=H_FILL / dx,
                targetDt=float(targetDt) if targetDt is not None else None,
                machTarget=float(machTarget) if targetDt is None else None,
                c0=c0, machAtImpact=(2.0 * G * H_FILL) ** 0.5 / c0 if c0 else None,
                accelConstraint=bool(accelConstraint), wallPeriodic=bool(wallPeriodic),
                diverged=bool(r.diverged), nSteps=int(r.nSteps),
                wallTime_s=float(r.wallTime or 0.0),
                dtFixed=float(r.ctx.config.dt) if scheme != 'artificialCompressible' else None,
                sensors=[{k: s[k] for k in ('name', 'z_mm', 'median', 'lo', 'hi')}
                         for s in SENSORS])

    npz = os.path.join(out, tag + '.npz')
    np.savez(npz, meta=json.dumps(meta), **cols)
    print(f'[{tag}] -> {npz}   diverged={r.diverged}  steps={r.nSteps}  '
          f'wall={r.wallTime:.1f}s', flush=True)

    if video and r.videoPath and os.path.exists(r.videoPath):
        for f in ('output.mp4', 'out.gif'):
            src = os.path.join(os.path.dirname(r.videoPath), f)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(out, f'{tag}_{f}'))
                print(f'[{tag}] video -> {out}/{tag}_{f}', flush=True)


def _rollingMedian(x, w):
    """Odd-window centred rolling median, edge-replicated."""
    import numpy as np
    w = max(1, int(w) | 1)
    if w == 1 or x.size < w:
        return x.copy()
    pad = w // 2
    xp = np.pad(x, pad, mode='edge')
    return np.median(np.lib.stride_tricks.sliding_window_view(xp, w), axis=-1)


def _peakStats(tStar, pStar, nnbr, smoothTStar=0.10):
    """Peak statistics for one sensor trace.

    Returns a dict:
      tArr        first t* with P* > 0.05 and a real neighbourhood
      rawPeak     max P* over the whole run (single-sample; catches spikes)
      smoothPeak  max of the rolling-median-filtered trace (window ~smoothTStar
                  in t*) -- the spike-robust "peak" comparable to an
                  experimental transducer + peak-detector
      impactPeak  smoothPeak restricted to [tArr, tArr + 3] (the primary
                  impact event, before the secondary wave / void closure)
      tRawPeak / tImpactPeak  the t* of those
    """
    import numpy as np
    ok = np.isfinite(pStar) & np.isfinite(tStar) & (nnbr > 4)
    out = dict(tArr=float('nan'), rawPeak=float('nan'), smoothPeak=float('nan'),
               impactPeak=float('nan'), tRawPeak=float('nan'),
               tImpactPeak=float('nan'))
    if ok.sum() < 3:
        return out
    ts, ps = tStar[ok], pStar[ok]
    order = np.argsort(ts)
    ts, ps = ts[order], ps[order]
    dts = np.median(np.diff(ts)) if ts.size > 1 else 1.0
    w = max(1, round(smoothTStar / max(dts, 1e-9)))
    psm = _rollingMedian(ps, w)
    arr = ps > 0.05
    out['tArr'] = float(ts[arr][0]) if arr.any() else float('nan')
    ir = int(np.argmax(ps)); out['rawPeak'] = float(ps[ir]); out['tRawPeak'] = float(ts[ir])
    ism = int(np.argmax(psm)); out['smoothPeak'] = float(psm[ism])
    if arr.any():
        win = (ts >= out['tArr']) & (ts <= out['tArr'] + 3.0)
        if win.any():
            j = int(np.argmax(psm[win]))
            out['impactPeak'] = float(psm[win][j])
            out['tImpactPeak'] = float(ts[win][j])
    return out


def _report(out: str):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    runs = []
    for npz in sorted(glob.glob(os.path.join(out, '*.npz'))):
        d = np.load(npz, allow_pickle=True)
        meta = json.loads(str(d['meta']))
        runs.append((meta, {k: d[k] for k in d.files if k != 'meta'}))
    if not runs:
        print(f'no .npz runs under {out}', file=sys.stderr)
        sys.exit(1)

    _base = {'deltaSPH': 'δ-SPH', 'artificialCompressible': 'ACSPH'}

    def runLabel(meta):
        s = _base.get(meta['scheme'], meta['scheme'])
        s += f" nx{meta['nx']}"
        if meta.get('wallPeriodic'):
            s += ' periodic'
        m = meta.get('machAtImpact')
        if m:
            s += f' Ma{m:.2f}'
        return s

    _palette = ['#c1121f', '#0353a4', '#2a9d8f', '#e76f51', '#6a4c93', '#8d99ae']
    colours = {runLabel(m): _palette[i % len(_palette)]
               for i, (m, _) in enumerate(runs)}
    labels = _base  # kept for any legacy references below

    # --- per-sensor P*(t*) panels --------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for k, (ax, s) in enumerate(zip(axes.flat, SENSORS)):
        ax.axhspan(s['lo'], s['hi'], color='0.85',
                   label='Lobovsky 2014  2.5–97.5 %')
        ax.axhline(s['median'], color='0.4', ls='--', lw=1,
                   label='Lobovsky median')
        for meta, col in runs:
            lbl = runLabel(meta)
            c = colours.get(lbl, 'k')
            ts = col['tStar']; ps = col[f'pProbe{k}Star']
            dts = np.median(np.diff(ts[np.isfinite(ts)]))
            psm = _rollingMedian(np.nan_to_num(ps), max(1, round(0.10 / max(dts, 1e-9))))
            ax.plot(ts, ps, color=c, lw=0.5, alpha=0.30)
            ax.plot(ts, psm, color=c, lw=1.4,
                    label=f"{lbl}  (raw + t*≈0.1 median)")
        ax.set_title(f"{s['name']}  (z = {s['z_mm']:.0f} mm, z/H = {s['z_mm']/300:.3f})")
        ax.set_ylabel('P*  =  P / (ρ₀ g H)')
        ax.set_xlim(0, min(11, max(np.nanmax(c['tStar']) for _, c in runs)))
        ax.grid(alpha=0.25)
        if k == 0:
            ax.legend(fontsize=7, loc='upper right')
    for ax in axes[-1]:
        ax.set_xlabel('t*  =  t √(g/H)')
    fig.suptitle('Dam break — downstream-wall pressure vs. Lobovsky et al. 2014 '
                 '(H = 300 mm)', fontsize=13)
    fig.tight_layout()
    p1 = os.path.join(out, 'pressure_sensors.png')
    fig.savefig(p1, dpi=110)
    plt.close(fig)

    # --- kinetic energy ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4))
    for meta, col in runs:
        lbl = runLabel(meta)
        ax.plot(col['tStar'], col['kineticEnergy'], color=colours.get(lbl, 'k'),
                label=lbl)
    ax.set_xlabel('t*  =  t √(g/H)'); ax.set_ylabel('kinetic energy [code units]')
    ax.set_title('Dam break — fluid kinetic energy'); ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p2 = os.path.join(out, 'kinetic_energy.png')
    fig.savefig(p2, dpi=110)
    plt.close(fig)

    # --- markdown --------------------------------------------------------------
    lines = []
    L = lines.append
    L('# Dam break vs. Lobovský et al. 2014 — downstream-wall pressure\n')
    L('`scripts/probe_acsphDambreakLobovsky.py` · `ACSPH_PLAN.md` §4.5 / Part 8 step 9\n')
    L('## Setup\n')
    L('| | |')
    L('|---|---|')
    L(f'| Reference | Lobovský, Botia-Vera, Castellana, Mas-Soler, Souto-Iglesias, '
      '*J. Fluids Struct.* **48** (2014) 407–434, H = 300 mm series (100 runs) |')
    L(f'| Tank (2D) | {TANK_W*1000:.0f} mm long × {TANK_L*1000:.0f} mm tall |')
    L(f'| Reservoir | {RES_W*1000:.0f} mm wide × {H_FILL*1000:.0f} mm water (H) |')
    L('| Dam removal | instantaneous (no gate model) |')
    L('| Walls | no-slip |')
    L(f'| Non-dim | t\\* = t √(g/H), P\\* = P / (ρ₀ g H), ρ₀ g H = {_RHO_G_H_MBAR:.2f} mbar |')
    L(f'| Probe | first-order MLS (Liu–Liu) interpolation of fluid pressure at '
      'the sensor points on the impact wall, every step |')
    L('')
    L('| run | H/Δx | Δx | c₀ | Mach √(2gH)/c₀ | t reached | steps | wall time | diverged |')
    L('|---|---|---|---|---|---|---|---|---|')
    for meta, _ in runs:
        mach = meta.get('machAtImpact')
        c0 = meta.get('c0') or 0.0
        c0s = f'{c0:.1f}' if c0 else '—'
        machs = f'{mach:.2f}' if mach else '—'
        L(f"| {runLabel(meta)} | "
          f"{meta['HdxRatio']:.0f} | {meta['dx']*1000:.2f} mm | {c0s} | {machs} | "
          f"{meta.get('tReached', 0):.2f} / {meta['tLimit']:.1f} s | {meta['nSteps']} | "
          f"{meta['wallTime_s']:.0f} s | {'**yes**' if meta['diverged'] else 'no'} |")
    L('')
    L('## Peak pressure vs. experimental band\n')
    L('Lobovský\'s Table 2 is the 2.5–97.5 % percentile band of the per-run '
      '*peak* pressure across 100 experiments. A single simulation gives one '
      'sample, and its raw single-step maximum is dominated by acoustic / '
      'free-surface spikes; `impact peak` is the maximum of the '
      't\\*≈0.1-wide rolling-median-filtered trace within the primary impact '
      'window `[t*_arrival, t*_arrival + 3]` (spike-robust, comparable to a '
      'real transducer), `raw peak` the unfiltered whole-run max for '
      'reference. `verdict` scores `impact peak` against the band.\n')
    for meta, col in runs:
        L(f'### {runLabel(meta)}  (H/Δx = {meta["HdxRatio"]:.0f})\n')
        L('| sensor | z/H | exp. median | exp. 2.5–97.5 % | impact peak P\\* | t\\* @ impact | raw peak P\\* | t\\* arrival | verdict |')
        L('|---|---|---|---|---|---|---|---|---|')
        for k, s in enumerate(SENSORS):
            st = _peakStats(col['tStar'], col[f'pProbe{k}Star'],
                            col[f'pProbe{k}Nnbr'])
            pk = st['impactPeak']
            verdict = '—'
            if pk == pk:
                verdict = ('✅ in band' if s['lo'] <= pk <= s['hi']
                           else ('▲ above' if pk > s['hi'] else '▼ below'))
            L(f"| {s['name']} | {s['z_mm']/300:.3f} | {s['median']:.2f} | "
              f"{s['lo']:.2f} – {s['hi']:.2f} | {pk:.2f} | {st['tImpactPeak']:.2f} | "
              f"{st['rawPeak']:.2f} | {st['tArr']:.2f} | {verdict} |")
        L('')
    L('## Figures\n')
    L('![pressure sensors](pressure_sensors.png)\n')
    L('![kinetic energy](kinetic_energy.png)\n')
    vids = sorted(glob.glob(os.path.join(out, '*_output.mp4')))
    if vids:
        L('## Video\n')
        for v in vids:
            L(f'- `{os.path.basename(v)}`')
        L('')
    L('## Notes\n')
    L('- Peak wall pressure in a dam break is resolution- and '
      'compressibility-sensitive; a single simulation produces one sample, '
      'the experimental band is 100. Landing in-band is encouraging, not a '
      'pass/fail in the statistical sense.')
    L('- `t* arrival` is the first sample with P\\* > 0.05; Lobovský quote a '
      '~t* = 0.07 spread in occurrence time across their 100 runs.')
    L('- Sensor 4 (z/H = 0.267) records the secondary wave rather than a clean '
      'primary impact in the experiment, so its peak occurs late.')

    md = os.path.join(out, 'REPORT.md')
    with open(md, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'-> {md}')
    print(f'-> {p1}')
    print(f'-> {p2}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scheme', choices=['deltaSPH', 'artificialCompressible'])
    ap.add_argument('--nx', type=int, default=128)
    ap.add_argument('--tLimit', type=float, default=2.0)
    ap.add_argument('--video', action='store_true')
    ap.add_argument('--plotInterval', type=int, default=20)
    ap.add_argument('--out', default=DEFAULT_OUT)
    ap.add_argument('--report', action='store_true',
                    help='(re)build plots + REPORT.md from existing .npz runs')
    ap.add_argument('--no-accel-constraint', dest='accelConstraint',
                    action='store_false',
                    help="ACSPH: drop Eq. (46)'s non-paper acceleration "
                         "constraint (keeps dt at the advective limit through "
                         "the wall impact; the paper's literal constraint set)")
    ap.add_argument('--machTarget', type=float, default=0.1,
                    help='Sun 2017 Eq. (2): c0 = sqrt(2 g H) / machTarget (default 0.1)')
    ap.add_argument('--targetDt', type=float, default=None,
                    help='override with the legacy back-solve (c0 ~ 1/dx)')
    ap.add_argument('--periodic', dest='wallPeriodic', action='store_true',
                    help='restore the periodic domain wrap (diagnostic for '
                         'near-wall neighbour-search artefacts)')
    args = ap.parse_args()

    if args.report or args.scheme is None:
        _report(args.out)
        return
    _runOne(args.scheme, args.nx, args.tLimit, args.out, args.video,
            args.plotInterval, accelConstraint=args.accelConstraint,
            targetDt=args.targetDt, wallPeriodic=args.wallPeriodic,
            machTarget=args.machTarget)


if __name__ == '__main__':
    main()
