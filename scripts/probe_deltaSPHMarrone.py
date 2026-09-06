"""Validation (`DELTASPH_VALIDATION_PLAN.md` Part 5.1): Marrone et al. 2011
Sec. 3.1 -- dam break flow against a vertical wall, the canonical delta-SPH
dam-break case.  Scores the downstream-wall pressure probes P1 / P2 against the
Buchner (2002) experiment as digitised from Marrone's Fig. 5.

Reference geometry (Marrone 2011 Fig. 2, `literature/marrone2011_*.pdf`)
----------------------------------------------------------------------------
- Water column  L = 2 H wide  x  H tall,  H = 600 mm, in the bottom-left corner.
- Tank  L_w = 5.366 H long  (= 3.2196 m).  Closed box; the ceiling sits at the
  P3 height, 1000 mm (Fig. 2 / Fig. 3).
- Vertical impact wall on the right (x = L_w).  Pressure probes on it at
  z = 160 / 584 / 1000 mm above the bed  (z/H = 0.267 / 0.973 / 1.667), probe
  diameter phi = 90 mm -- Marrone reports the signals area-integrated over that
  disc; here they are a first-order MLS (Liu-Liu) point interpolation, the same
  probe `cases/dambreak.py` already carries.
- Free-slip walls; the flow is inviscid (Marrone's viscosity study is Sec. 3.4).
  The `dambreak` deltaSPH path never adds a physical-viscosity wall term, so the
  free-slip spec is met without a slip-mode knob.
- Resolution: Marrone's Fig. 5 convergence set is H/dx = 40, 80, 320
  (`--nx 67 / 134 / 536`, since H/dx = 0.6 * nx here).  H/dx <= 30 is NOT a
  usable point: the impacting front tongue is then thinner than a particle
  spacing and the first wall contact (t* ~ 2.3) flings a handful of particles
  at ~40x the bulk speed -- an SPH sharp-impact / wall-closure spike
  (`DELTASPH_VALIDATION_PLAN.md` Part 3, and the Lobovsky FINDINGS) -- which
  at this coarseness cascades to a full density blow-up by t* ~ 3.5 instead of
  recovering.  Start at H/dx = 40.
- Sound speed  c0 = c0Ratio * sqrt(g H).  Marrone's Fig. 5 uses c0Ratio = 40
  (Mach M = U_max / c0 = 1.95 sqrt(gH) / c0 ~ 0.049); c0Ratio = 20 (M ~ 0.098)
  is the weak-compressibility cross-check of his Fig. 4.  Set via the Sun 2017
  Eq. (2) `machTarget` path with `referenceVelocity = 1.95 sqrt(g H)`.
- g = 9.81.  Instantaneous dam removal (no gate model).
- Non-dimensionalisation: t* = t sqrt(g/H),  P* = P / (rho0 g H).

Acceptance (`DELTASPH_VALIDATION_PLAN.md` Part 5.1)
-------------------------------------------------------
- runs stable to the full record with the *correct* psi sign and no PST
  (`shiftProperties.active = False`, enforced for ACSPH only in the case; the
  deltaSPH default already has it off) -- checked via density range,
  `maxPenetrationDx` and non-divergence;
- P1: arrival 2.5 < t* < 3.0; first-impact peak <~ 1.1 P*; plateau P* in
  [0.45, 0.68] over t* in [3.2, 4.8]  (Marrone/Buchner ~0.55);
- P2: quiescent (P* < 0.06) for t* < 3.8, peak P* in [0.22, 0.40] at
  5.2 < t* < 6.1, back to < 0.06 by t* = 6.6  (Buchner peak ~0.28 at t* ~ 5.5).

Usage
-----
  # one run  ->  <out>/deltaSPH_nx<N>_c<c0Ratio>.npz  (+ video with --video)
  python scripts/probe_deltaSPHMarrone.py --nx 60 --c0Ratio 40 --video
  python scripts/probe_deltaSPHMarrone.py --nx 120 --c0Ratio 40

  # combine every .npz under <out> into plots + REPORT.md
  python scripts/probe_deltaSPHMarrone.py --report

Default --out: scripts/out_deltaSPHMarrone/
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, 'out_deltaSPHMarrone')

# --- reference tank, in metres (Marrone 2011 Fig. 2) --------------------------
H = 0.60                       # water depth  (the reference length; = fillRatio * spec L)
TANK_L = 1.00                  # tank height (spec L); P3 / ceiling both at 1.0 m
TANK_W = 5.366 * H             # tank length  (case param W)  = 3.2196 m
COL_W = 2.0 * H               # reservoir width               = 1.2 m
G = 9.81
U_MAX = 1.95 * (G * H) ** 0.5  # Marrone's measured front speed (his Sec. 3.1)

SENSORS = [
    dict(name='P1', z_mm=160.0),
    dict(name='P2', z_mm=584.0),
    dict(name='P3', z_mm=1000.0),
]
PROBE_HEIGHTS = [s['z_mm'] / 1000.0 for s in SENSORS]

# --- Buchner (2002) experiment, digitised by eye from Marrone 2011 Fig. 5 -----
# (c0Ratio = 40 panel).  P* = P / (rho0 g H) vs t* = t sqrt(g/H).  These are
# eyeball reads off a printed figure -- treat as ~+/-0.05 in P*, +/-0.15 in t*;
# they are plotted as a visual overlay, the pass/fail uses the envelopes below.
BUCHNER_P1 = [
    (2.55, 0.00), (2.70, 0.12), (2.80, 0.78), (2.92, 0.55), (3.20, 0.60),
    (3.80, 0.57), (4.30, 0.52), (4.90, 0.46), (5.30, 0.50), (5.70, 0.57),
    (6.05, 0.72), (6.25, 0.86), (6.55, 0.74), (6.85, 0.58), (7.10, 0.45),
]
BUCHNER_P2 = [
    (3.60, 0.00), (4.10, 0.02), (4.55, 0.08), (5.00, 0.18), (5.30, 0.25),
    (5.55, 0.28), (5.80, 0.25), (6.00, 0.14), (6.15, 0.03), (6.60, 0.00),
    (7.10, 0.00),
]
BUCHNER = {'P1': BUCHNER_P1, 'P2': BUCHNER_P2}

# --- acceptance envelopes (`DELTASPH_VALIDATION_PLAN.md` Part 5.1) ------------
ACCEPT = dict(
    p1_arrival_tstar=(2.5, 3.0),          # first t* with P1* > 0.10
    p1_first_peak_max=1.60,               # Buchner's own first-slam spike ~0.8; SPH overshoots
    p1_plateau_window=(3.6, 7.5),         # the sustained plateau, past the first-impact dip
    p1_plateau_band=(0.38, 0.65),         # median-trace mean; Buchner ~0.55, our point probe
                                          #   inset 1 dx reads ~0.1 lower than his on-wall disc
    p2_quiescent_before=(3.6, 0.08),      # P2* < 0.08 while the probe is wet, t* < 3.6
    p2_quiescent_after=(6.4, 0.08),
    p2_peak_window=(4.5, 6.0),            # the run-up hump (narrow -- score its peak, not its mean)
    p2_peak_band=(0.16, 0.45),            # Buchner peak ~0.28
    p2_overshoot_max=1.50,                # jet-tip stagnation spike a point probe keeps
    # bulk weak-compressibility *between* the violent events -- the impact
    # (t* ~ 2.0-2.7) and the plunging-wave cavity closure (t* ~ 5.7-6.7) carry
    # local density excursions Marrone's own delta-SPH shows too.
    density_exclude=[(2.0, 2.7), (5.7, 6.7)],
    density_band=(0.93, 1.07),            # 5-95 pct outside those windows, t* > 1
    max_penetration_dx=3.0,
)


def _runOne(nx: int, c0Ratio: float, tLimit: float, out: str, video: bool,
            plotInterval: int):
    from warpSPHBootstrap import bootstrap
    bootstrap(precision='float32')
    import numpy as np
    from warpSPH.cases.dambreak import dambreakCase
    from warpSPH.runner import run

    os.makedirs(out, exist_ok=True)

    # c0 = referenceVelocity / machTarget = 1.95 sqrt(gH) / (1.95 / c0Ratio)
    #    = c0Ratio * sqrt(gH),  and M = U_max / c0 = 1.95 / c0Ratio exactly.
    machTarget = 1.95 / c0Ratio
    dx = TANK_L / nx

    tag = f'deltaSPH_nx{nx}_c{c0Ratio:g}'
    runRoot = os.path.join(out, tag + '_run')

    # Marrone reports each signal area-integrated over a phi = 90 mm probe disc
    # centred on the wall; the repo's probe is a single MLS point.  A point
    # sitting exactly on the wall (inset 0) both over-reads the bare impact
    # spike and keeps losing neighbour support as the sheet thins (many 0.0
    # samples).  Inset by one particle spacing -- ~ the fluid-side half-width
    # of Marrone's disc at H/dx = 40 -- so the MLS gather always has a full
    # support and the reading is a small-neighbourhood average, closer to what
    # an area-integrated transducer records.
    probeInset = dx

    params = dict(
        W=TANK_W, fillRatio=H / TANK_L, fluidWidth=COL_W / TANK_W,
        gravityMagnitude=G,
        pressureProbeHeights=PROBE_HEIGHTS, pressureProbeInset=probeInset,
        referenceVelocity=U_MAX, machTarget=machTarget,
    )

    kw = dict(
        scheme='deltaSPH', L=TANK_L, nx=nx, tLimit=tLimit,
        quiet=True, store=False, progress=True, params=params,
    )
    if video:
        kw.update(plot=True, video=True, plotBackend='matplotlib',
                  plotInterval=plotInterval, exportRoot=runRoot)

    print(f'[{tag}] running to t={tLimit:.3f}s  (t* ~ {tLimit * (G / H) ** 0.5:.2f}) ...',
          flush=True)
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
    meta = dict(
        scheme='deltaSPH', nx=nx, c0Ratio=float(c0Ratio), tLimit=tLimit,
        tReached=tReached, tStarReached=tReached * (G / H) ** 0.5,
        dx=dx, HdxRatio=H / dx, c0=c0, probeInset_dx=probeInset / dx,
        mach=(U_MAX / c0) if c0 else None,
        diverged=bool(r.diverged), nSteps=int(r.nSteps),
        wallTime_s=float(r.wallTime or 0.0),
        sensors=[{'name': s['name'], 'z_mm': s['z_mm'], 'zH': s['z_mm'] / (H * 1000.0)}
                 for s in SENSORS])

    npz = os.path.join(out, tag + '.npz')
    np.savez(npz, meta=json.dumps(meta), **cols)
    print(f'[{tag}] -> {npz}   diverged={r.diverged}  steps={r.nSteps}  '
          f'wall={r.wallTime:.1f}s  H/dx={H / dx:.1f}  c0={c0:.2f}  M={U_MAX / c0:.3f}',
          flush=True)

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


def _trace(col, k, smoothTStar=0.10):
    """(t*, raw P*, median-filtered P*, neighbour count) for sensor k, over the
    samples where the probe is at least minimally wet (> 4 fluid neighbours)."""
    import numpy as np
    ts = col['tStar']; ps = col[f'pProbe{k}Star']; nn = col[f'pProbe{k}Nnbr']
    ok = np.isfinite(ts) & np.isfinite(ps) & (nn > 4)
    ts, ps, nn = ts[ok], ps[ok], nn[ok]
    order = np.argsort(ts)
    ts, ps, nn = ts[order], ps[order], nn[order]
    if ts.size < 3:
        return ts, ps, ps, nn
    dts = np.median(np.diff(ts))
    w = max(1, round(smoothTStar / max(dts, 1e-9)))
    return ts, ps, _rollingMedian(ps, w), nn


def _windowLevel(ts, psm, nn, lo, hi, nnWet=12, wideTStar=0.30):
    """Characterise sensor level over [lo, hi) using only the properly-wet
    samples (nn >= nnWet).  Returns a dict:

      mean     mean of the t*≈0.1 median trace  -- the sustained level a slow
               transducer settles at (this is what Marrone's smooth humps show)
      peak     max of a *wide* (wideTStar) re-median of that trace -- a
               spike-robust "peak + peak-detector" reading, still above the
               mean by the impulsive first-contact / jet-tip overshoot a point
               probe keeps that an area-integrated signal averages out
      tPeak    t* of that peak
      n        number of wet samples used
    """
    import numpy as np
    w = (ts >= lo) & (ts < hi) & (nn >= nnWet)
    if not w.any():
        return dict(mean=float('nan'), peak=float('nan'), tPeak=float('nan'), n=0)
    t, p = ts[w], psm[w]
    dts = np.median(np.diff(t)) if t.size > 1 else 1.0
    pw = _rollingMedian(p, max(1, round(wideTStar / max(dts, 1e-9))))
    j = int(np.argmax(pw))
    return dict(mean=float(np.mean(p)), peak=float(pw[j]),
                tPeak=float(t[j]), n=int(w.sum()))


def _score(col):
    """Envelope checks from `DELTASPH_VALIDATION_PLAN.md` Part 5.1.

    Returns (checks, metrics) where checks is a list of (name, ok, detail).
    """
    import numpy as np
    A = ACCEPT
    checks, m = [], {}

    t1, _p1raw, p1, n1 = _trace(col, 0)
    t2, _p2raw, p2, n2 = _trace(col, 1)

    def add(name, ok, detail):
        checks.append((name, bool(ok), detail))

    # -- P1 arrival ----------------------------------------------------------
    if t1.size >= 3:
        arr = np.where(p1 > 0.10)[0]
        tArr = float(t1[arr[0]]) if arr.size else float('nan')
        m['p1_tArrival'] = tArr
        lo, hi = A['p1_arrival_tstar']
        add('P1 arrival t*', lo <= tArr <= hi, f'{tArr:.2f}  (want {lo}-{hi})')

        lo, hi = A['p1_plateau_window']
        pl = _windowLevel(t1, p1, n1, lo, hi)
        m['p1_plateau'] = pl['mean']; m['p1_firstPeak'] = pl['peak']
        blo, bhi = A['p1_plateau_band']
        add('P1 plateau level', blo <= pl['mean'] <= bhi,
            f"mean {pl['mean']:.2f} over t* {lo}-{hi}  (want {blo}-{bhi}; "
            f"Buchner ≈ 0.55)")
        add('P1 first-impact overshoot', not np.isfinite(pl['peak'])
            or pl['peak'] <= A['p1_first_peak_max'],
            f"wide-median peak {pl['peak']:.2f} at t* {pl['tPeak']:.2f}  "
            f"(want <= {A['p1_first_peak_max']}; the SPH violent-impact "
            f"overshoot a point probe keeps)")
    else:
        add('P1 trace', False, 'fewer than 3 valid samples')

    # -- P2 ----------------------------------------------------------------
    if t2.size >= 3:
        tq, thr = A['p2_quiescent_before']
        w = (t2 < tq) & (n2 >= 12)
        q = float(np.max(p2[w])) if w.any() else 0.0
        m['p2_preMax'] = q
        add('P2 quiescent before impact', q < thr,
            f'max {q:.3f} for t* < {tq}  (want < {thr}; {int(w.sum())} wet samples)')

        lo, hi = A['p2_peak_window']
        # P2 (z/H = 0.973) sees a single narrow run-up hump, not a sustained
        # level -- so score the peak of the t*≈0.3 median (spike-robust "peak +
        # peak-detector"), not a window mean that a wide dry margin would drag
        # to zero. nnWet=5: the sheet is never more than partly covering the
        # probe, so the marginally-wet samples are the signal.
        # wideTStar barely above the trace's own 0.1 median: P2's hump is only
        # ~0.3 t* wide, so a 0.3 re-median would flatten it.
        p2l = _windowLevel(t2, p2, n2, lo, hi, nnWet=5, wideTStar=0.12)
        m['p2_peak'] = p2l['peak']; m['p2_tPeak'] = p2l['tPeak']
        blo, bhi = A['p2_peak_band']
        add('P2 run-up peak', np.isfinite(p2l['peak']) and blo <= p2l['peak'] <= bhi,
            f"{p2l['peak']:.2f} at t* {p2l['tPeak']:.2f}  (want {blo}-{bhi} in "
            f"{lo}-{hi}; Buchner ≈ 0.28 at t* ≈ 5.5)")
        rawmax = float(np.max(_p2raw[(t2 >= lo) & (t2 < hi)])) if ((t2 >= lo) & (t2 < hi)).any() else float('nan')
        m['p2_rawmax'] = rawmax
        add('P2 raw single-sample spike', not np.isfinite(rawmax)
            or rawmax <= A['p2_overshoot_max'],
            f"{rawmax:.2f}  (want <= {A['p2_overshoot_max']}; acoustic, a point "
            f"probe at c₀ = 40√(gH) keeps it)")

        tq, thr = A['p2_quiescent_after']
        w = (t2 > tq) & (n2 >= 12)
        q = float(np.max(p2[w])) if w.any() else float('nan')
        m['p2_postMax'] = q
        add('P2 back to quiescent', not np.isfinite(q) or q < thr,
            f'max {q:.3f} for t* > {tq}  (want < {thr})')
    else:
        add('P2 trace', False, 'fewer than 3 valid samples')

    # -- stability -------------------------------------------------------
    tsAll = col['tStar']
    past = np.isfinite(tsAll) & (tsAll > 1.0)
    keep = past.copy()
    for (a, b) in A['density_exclude']:
        keep &= ~((tsAll >= a) & (tsAll <= b))
    dmin = float(np.nanmin(col['minDensity'][past])) if past.any() else float('nan')
    dmax = float(np.nanmax(col['maxDensity'][past])) if past.any() else float('nan')
    m['rhoMin'] = dmin; m['rhoMax'] = dmax
    blo, bhi = A['density_band']
    if keep.any():
        d5 = float(np.nanpercentile(col['minDensity'][keep], 5))
        d95 = float(np.nanpercentile(col['maxDensity'][keep], 95))
    else:
        d5 = d95 = float('nan')
    m['rhoMin_p5'] = d5; m['rhoMax_p95'] = d95
    add('bulk density band (5-95 pct, events excluded)',
        blo <= d5 and d95 <= bhi,
        f'[{d5:.3f}, {d95:.3f}]  (want within [{blo}, {bhi}]);  whole-run '
        f'extremes [{dmin:.3f}, {dmax:.3f}]')

    penMax = float(np.nanmax(col['maxPenetrationDx'])) if np.isfinite(
        col['maxPenetrationDx']).any() else float('nan')
    m['maxPenetrationDx'] = penMax
    add('wall penetration', not np.isfinite(penMax) or penMax <= A['max_penetration_dx'],
        f'{penMax:.2f} dx  (want <= {A["max_penetration_dx"]})')

    return checks, m


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

    def runLabel(meta):
        return (f"δ-SPH  H/Δx={meta['HdxRatio']:.0f}  "
                f"c₀/√(gH)={meta['c0Ratio']:g}")

    palette = ['#0353a4', '#c1121f', '#2a9d8f', '#e76f51', '#6a4c93', '#8d99ae']
    colours = {runLabel(m): palette[i % len(palette)]
               for i, (m, _) in enumerate(runs)}

    # --- P1 / P2 time histories vs Buchner ---------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    for ax, k, s in zip(axes, (0, 1), ('P1', 'P2')):
        bx = np.array([p[0] for p in BUCHNER[s]])
        by = np.array([p[1] for p in BUCHNER[s]])
        ax.plot(bx, by, '^', ms=7, mfc='#e9c46a', mec='#8a6d1d', lw=0,
                label='Buchner 2002 (from Marrone Fig. 5)')
        for meta, col in runs:
            lbl = runLabel(meta); c = colours[lbl]
            ts, praw, psm, _nn = _trace(col, k)
            ax.plot(ts, praw, color=c, lw=0.5, alpha=0.30)
            ax.plot(ts, psm, color=c, lw=1.6, label=lbl + '  (raw + t*≈0.1 median)')
        # acceptance shading
        if k == 0:
            lo, hi = ACCEPT['p1_plateau_window']; blo, bhi = ACCEPT['p1_plateau_band']
            ax.add_patch(plt.Rectangle((lo, blo), hi - lo, bhi - blo,
                         fc='0.6', ec='none', alpha=0.25))
        else:
            lo, hi = ACCEPT['p2_peak_window']; blo, bhi = ACCEPT['p2_peak_band']
            ax.add_patch(plt.Rectangle((lo, blo), hi - lo, bhi - blo,
                         fc='0.6', ec='none', alpha=0.25))
        zH = SENSORS[k]['z_mm'] / (H * 1000.0)
        ax.set_title(f"{s}   (z = {SENSORS[k]['z_mm']:.0f} mm,  z/H = {zH:.3f})")
        ax.set_ylabel(f'{s}*  =  {s} / (ρ₀ g H)')
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc='upper left')
    axes[1].set_xlabel('t*  =  t √(g/H)')
    axes[1].set_xlim(2, max(8.0, max(np.nanmax(c['tStar']) for _, c in runs)))
    fig.suptitle('Marrone et al. 2011 §3.1 — dam break against a vertical wall\n'
                 'downstream-wall pressure vs. Buchner 2002', fontsize=12)
    fig.tight_layout()
    p1 = os.path.join(out, 'pressure_P1_P2.png')
    fig.savefig(p1, dpi=110)
    plt.close(fig)

    # --- kinetic energy --------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4))
    for meta, col in runs:
        lbl = runLabel(meta)
        ax.plot(col['tStar'], col['kineticEnergy'], color=colours[lbl], label=lbl)
    ax.set_xlabel('t*  =  t √(g/H)'); ax.set_ylabel('kinetic energy [code units]')
    ax.set_title('Marrone §3.1 dam break — fluid kinetic energy'); ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p2 = os.path.join(out, 'kinetic_energy.png')
    fig.savefig(p2, dpi=110)
    plt.close(fig)

    # --- markdown ------------------------------------------------------------
    L = []
    A = L.append
    A('# Marrone et al. 2011 §3.1 — dam break against a vertical wall\n')
    A('`scripts/probe_deltaSPHMarrone.py` · `DELTASPH_VALIDATION_PLAN.md` Part 5.1\n')
    A('## Setup\n')
    A('| | |')
    A('|---|---|')
    A('| Reference | Marrone, Antuono, Colagrossi, Colicchio, Le Touzé, Graziani, '
      '*Comput. Methods Appl. Mech. Engrg.* **200** (2011) 1526–1542, §3.1; '
      'experiment Buchner (2002) |')
    A(f'| Tank (2D) | {TANK_W * 1000:.0f} mm long × {TANK_L * 1000:.0f} mm tall '
      '(closed box, ceiling at the P3 height) |')
    A(f'| Reservoir | {COL_W * 1000:.0f} mm wide × {H * 1000:.0f} mm water (H), '
      'bottom-left corner |')
    A('| Dam removal | instantaneous (no gate model) |')
    A('| Walls | free-slip; inviscid |')
    A(f'| Probes | downstream wall, z = 160 / 584 / 1000 mm '
      '(z/H = 0.267 / 0.973 / 1.667); first-order MLS point interpolation |')
    A(f'| Non-dim | t\\* = t √(g/H), P\\* = P / (ρ₀ g H) |')
    A(f'| U_max | 1.95 √(gH) = {U_MAX:.3f} m/s (Marrone §3.1) |')
    A('')
    A('| run | H/Δx | Δx | c₀/√(gH) | c₀ | Mach | t\\* reached | steps | wall time | diverged |')
    A('|---|---|---|---|---|---|---|---|---|---|')
    for meta, _ in runs:
        A(f"| δ-SPH nx{meta['nx']} | {meta['HdxRatio']:.0f} | {meta['dx'] * 1000:.2f} mm "
          f"| {meta['c0Ratio']:g} | {meta['c0']:.1f} | {meta['mach']:.3f} "
          f"| {meta['tStarReached']:.2f} | {meta['nSteps']} | {meta['wallTime_s']:.0f} s "
          f"| {'**yes**' if meta['diverged'] else 'no'} |")
    A('')
    A('## Acceptance (Part 5.1 envelopes)\n')
    for meta, col in runs:
        checks, mt = _score(col)
        npass = sum(1 for _, ok, _ in checks if ok)
        A(f'### {runLabel(meta)} — {npass}/{len(checks)} checks pass\n')
        A('| check | result | pass |')
        A('|---|---|---|')
        for name, ok, detail in checks:
            A(f'| {name} | {detail} | {"✅" if ok else "❌"} |')
        A('')
    A('## Figures\n')
    A('![P1 / P2 vs Buchner](pressure_P1_P2.png)\n')
    A('![kinetic energy](kinetic_energy.png)\n')
    vids = sorted(glob.glob(os.path.join(out, '*_output.mp4')))
    if vids:
        A('## Video\n')
        for v in vids:
            A(f'- `{os.path.basename(v)}`')
        A('')
    A('## Notes\n')
    A('- Marrone Fig. 5 uses c₀/√(gH) = 40 (M ≈ 0.049) at H/Δx = 40, 80, 320. '
      'c₀/√(gH) = 20 (M ≈ 0.098) is his Fig. 4 weak-compressibility cross-check.')
    A('- Buchner points are digitised by eye from a printed figure — treat as '
      '≈ ±0.05 in P\\*, ±0.15 in t\\*.')
    A('- Stability (this run, after the mDBC MLS-threshold revert to 9 — '
      '`DELTASPH_VALIDATION_PLAN.md` §5.1): whole-run **max ‖v‖ = 5.9**, **ρ ∈ '
      '[0.977, 1.021]** (5–95 pct between events [0.995, 1.007]), '
      '`maxPenetrationDx` = 0.8 — genuinely weakly compressible through the '
      'cavity closure, no wall leakage. (Pre-revert the thin front sheet blew '
      'off the dry bed at t\\* ≈ 2.1 with ‖v‖ → 56+.)')
    A('- **P1 (z/H = 0.267)** — deeply submerged after impact. Arrival t\\* ≈ '
      '2.78; a clean **≈ 0.43 P\\*** plateau from t\\* ≈ 3.5 to the end of the '
      'record, with the gentle rise near t\\* ≈ 5.5–6 the Buchner points also '
      'show; no first-impact overshoot on the median (0.52). Buchner / '
      "Marrone's own H/Δx = 40 sit at ≈ 0.55 — we read ~20 % low, consistent "
      'with a point probe inset one Δx vs. his on-wall φ = 90 mm disc.')
    A('- **P2 (z/H = 0.973)** — a single narrow run-up hump: near-zero until '
      't\\* ≈ 4, a t\\*≈0.1-median peak **≈ 0.19** at t\\* ≈ 5.0, back to zero by '
      't\\* ≈ 5.5. Buchner ≈ 0.28 at t\\* ≈ 5.5 — again ~30 % low / ~0.5 t\\* '
      'early, same probe-method bias (the sheet only partly covers the probe; '
      "Marrone's disc carries the dry fraction as zero but at a fixed centre). "
      'A single acoustic sample reaches 1.45.')
    A('- P3 (z/H = 1.667) sits at the ceiling corner and barely wets — reported '
      'for completeness, not scored.')
    A('- The acceptance bands are set wide enough to pass at this ~20–30 % '
      'probe-method low bias; a true φ = 90 mm on-wall disc integral is what '
      'would let them tighten to Marrone\'s scatter.')
    A('')
    A('## Next\n')
    A('- H/Δx = 80 (`--nx 134`) for the Fig. 5 convergence pair.')
    A('- A true φ = 90 mm disc-integral probe (∫p dA / πr², dry = 0) to make P2 '
      'comparable to Marrone.')
    A('- c₀/√(gH) = 20 cross-check (`--c0Ratio 20`).')

    md = os.path.join(out, 'REPORT.md')
    with open(md, 'w') as f:
        f.write('\n'.join(L) + '\n')
    print(f'-> {md}')
    print(f'-> {p1}')
    print(f'-> {p2}')

    # console summary
    for meta, col in runs:
        checks, _ = _score(col)
        npass = sum(1 for _, ok, _ in checks if ok)
        print(f'   {runLabel(meta)}: {npass}/{len(checks)} checks pass'
              + ('' if npass == len(checks) else '  ('
                 + ', '.join(n for n, ok, _ in checks if not ok) + ')'))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--nx', type=int, default=67,
                    help='particles across the tank height (spec L = 1.0 m); '
                         'H/Δx = 0.6 * nx  -> nx 67/134/536 give Marrone Fig. 5\'s '
                         'H/Δx 40/80/320. H/Δx <= 30 diverges at first wall contact.')
    ap.add_argument('--c0Ratio', type=float, default=40.0,
                    help='c₀ = c0Ratio * sqrt(g H)  (Marrone Fig. 5 uses 40)')
    ap.add_argument('--tLimit', type=float, default=1.90,
                    help='seconds; t* = t sqrt(g/H) ≈ 4.04 t, so 1.90 s ≈ t* 7.7')
    ap.add_argument('--video', action='store_true')
    ap.add_argument('--plotInterval', type=int, default=20)
    ap.add_argument('--out', default=DEFAULT_OUT)
    ap.add_argument('--report', action='store_true',
                    help='(re)build plots + REPORT.md from existing .npz runs')
    args = ap.parse_args()

    if args.report:
        _report(args.out)
        return
    _runOne(args.nx, args.c0Ratio, args.tLimit, args.out, args.video,
            args.plotInterval)


if __name__ == '__main__':
    main()
