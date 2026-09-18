"""Validation (`DELTASPH_VALIDATION_PLAN.md` Part 5.3): Sun et al. 2017
Sec. 4.2 -- the oscillating droplet under a central conservative force field,
benchmark no. 2 of the delta+-SPH suite.  Reproduces that paper's **Table 1**
(momenta conservation vs `R/dx`) and **Figs. 10/13** (the semi-axis history
against the analytic solution).

Why this case
-------------
It is the suite's *conservation* test.  The flow is periodic and unforced apart
from the central potential, so the exact answer is known for all time, and both
momenta are identically zero throughout -- which makes every recorded value an
error with no reference-reading uncertainty at all.  Sun's Table 1 then prints
those errors at three resolutions, so the convergence *rate* is checkable too,
not just a level.  It is also the first free-surface case in this plan: unlike
Sun 2019's Taylor-Green (`scripts/probe_deltaPlusTGV.py`), the PST here has to
run against a surface, so it exercises the free-surface shift treatment that
the periodic benchmark cannot reach.

Reference setup (Sun et al. 2017 Sec. 4.2,
`literature/sun2017_delta-plus-sph-model.pdf`)
------------------------------------------------------------------------------
- Central conservative force `f = -B^2 r`; **inviscid** fluid; the drop starts
  circular with radius `R` and velocity `u = A0 x`, `v = -A0 y`.
- `A0 / B = 1`, `A0 = 1`, so the oscillation period is `T ~ 4.827 / A0` -- the
  `DROPLET_PERIOD` already encoded in `cases/oscillatingDroplet.py`.
- `c0 = 15 A0 R`, i.e. Mach 1/15 against the straining field's edge speed.
- Artificial viscosity `alpha = 0.01` (Eq. 1).  Sun's Fig. 13 comparison
  against Antuono et al.'s delta-SPH sets `alpha = 0` instead; `--alpha 0`
  reproduces that leg.
- `R/dx = 50, 100, 200`, run for **15 oscillation periods**.

Table 1 (the numbers this scores against), maximum error recorded over those
15 periods:

    R/dx   linear momentum        angular momentum
           (rho A0 R^3)           (rho A0 R^4)
     50    1.5e-3                 4.6e-4
    100    3.3e-4                 1.6e-4
    200    3.3e-5                 5.7e-6

Both momenta are exactly zero in the continuum -- the straining field is
symmetric about the drop centre and irrotational -- so the recorded magnitude
*is* the error.  `cases/oscillatingDroplet.diagnostics` emits them already
normalised (`linearMomentumStar` / `angularMomentumStar`).

Not reproduced here: Figs. 11-12's energy budget.  Sun's "total energy" is
constant only because it includes the energy the artificial viscosity and the
density-diffusion term (`Q_delta`) have dissipated, and neither is recoverable
from the state a diagnostic sees -- both are per-step integrals of terms
`schemes/deltaSPH.py` folds into `dvdt`/`drhodt` and does not hand back.  The
mechanical/potential/elastic split *is* recorded (`mechanicalEnergy`,
`elasticEnergy`, `totalEnergy`), and the report prints its drift, but that
drift is not comparable to Fig. 12's 1.2 % without the dissipation ledger, so
it is reported and not scored.

Usage
-----
  # one run -> <out>/droplet_Rdx<N>[_a<alpha>].npz
  python scripts/probe_deltaPlusDroplet.py --Rdx 50
  python scripts/probe_deltaPlusDroplet.py --Rdx 100 --periods 15
  python scripts/probe_deltaPlusDroplet.py --Rdx 50 --alpha 0     # Sun's Fig. 13 leg

  # combine every .npz under <out> into plots + REPORT.md
  python scripts/probe_deltaPlusDroplet.py --report

Default --out: scripts/out_deltaPlusDroplet/
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, 'out_deltaPlusDroplet')

# --- reference setup (Sun et al. 2017 Sec. 4.2) ------------------------------
R = 1.0            # initial drop radius, the reference length
A0 = 1.0           # initial strain rate; also the reference time scale 1/A0
B = 1.0            # central force strength, A0/B = 1
RHO0 = 1.0
MACH = 1.0 / 15.0  # c0 = 15 A0 R
#: Box side.  The drop reaches `1.932 R` at maximum elongation, so this leaves
#: about a full radius of clearance on every side -- the domain is periodic
#: (`WEAKLY_COMPRESSIBLE_DEFAULTS`), and a tighter box would let the kernel
#: support wrap onto the drop's far side at maximum elongation.
BOX = 6.0
PERIOD = 4.827     # `cases/oscillatingDroplet.DROPLET_PERIOD`, in units of 1/A0

#: Sun et al. 2017 Table 1: max momenta errors over 15 oscillation periods,
#: normalised as `rho A0 R^3` (linear) and `rho A0 R^4` (angular).
SUN_TABLE1 = {
    50:  dict(linear=1.5e-3, angular=4.6e-4),
    100: dict(linear=3.3e-4, angular=1.6e-4),
    200: dict(linear=3.3e-5, angular=5.7e-6),
}

#: Sun et al. 2017 Fig. 11 (`R/dx = 200`, `alpha = 0.01`), read off the figure:
#: the mechanical energy `(E_M - E_M^0)/E_M^0` falls to about **-4.8 %** over 14
#: periods, while the total energy and the elastic energy stay at ~0 and the
#: diffusive term's numerical heating `Q_delta` reaches ~0.5 % of `E_M^0` (the
#: text states 0.45 %).  So the delta+-SPH *does* lose mechanical energy on this
#: benchmark -- to the artificial viscosity, as the text says -- and the drop's
#: oscillation amplitude decays with it.  That is shared physics, not an error
#: to score against zero.
SUN_FIG11_MECHANICAL_LOSS_PCT = -4.8

ACCEPT = dict(
    # Table 1's numbers are printed to two significant figures and span two
    # decades across the three resolutions, so a *factor* is the meaningful
    # distance; x5 still separates adjacent rows of the table (they are 4.5x
    # and 10x apart in the linear column).
    momentum_factor_tol=5.0,
    # Fig. 10 is the *first* ~10 periods of a(t) against the analytic solution.
    # Scored over the first three, where accumulated phase error has not yet
    # dominated the pointwise difference; the amplitude is ~1.4 R peak-to-peak,
    # so 5 % of R is a visible-on-the-plot disagreement.
    #
    # A pointwise RMSE over all 15 periods is NOT the right measure and was
    # tried first: at `R/dx = 50` it reads 0.126 R almost entirely because a
    # 1 %-short period accumulates ~0.15 of a period of phase by the end, and
    # *no* figure in the paper constrains the phase at 15 periods (Fig. 10 stops
    # at 10, Fig. 13 zooms the last oscillation of a `R/dx = 200` run).  Period
    # and amplitude are scored directly instead -- they are what those two
    # figures actually show, and they are phase-robust.
    semi_axis_rmse_max=0.05,
    semi_axis_rmse_periods=3.0,
    # Fig. 13: the last simulated oscillation still lines up with the analytic
    # one.  A period error of this size is ~0.15 periods of drift over 15.
    period_rel_error_max=0.02,
    # The amplitude decay that goes with Fig. 11's mechanical-energy loss.  Sun
    # loses 4.8 % of E_M at `R/dx = 200`; a coarser run is more dissipative
    # (the artificial viscosity is `nu = alpha c0 h / (2(n+2))`, linear in `h`),
    # so this is a generous ceiling, not a match.
    amplitude_decay_max=0.10,
    mechanical_loss_pct_max=20.0,
    # The drop must survive: no divergence, and the free surface must not
    # fragment (Sun's Fig. 9 shows an intact ellipse at t/T = 8.45).
    density_band=(0.90, 1.10),
)


def _runOne(Rdx: int, periods: float, out: str, scheme: str, alpha: float,
            kernel: str, video: bool, plotInterval: int):
    from warpSPHBootstrap import bootstrap
    bootstrap(precision='float32')
    import numpy as np
    from warpSPH.cases.oscillatingDroplet import oscillatingDropletCase
    from warpSPH.runner import run

    os.makedirs(out, exist_ok=True)
    nx = int(round(BOX * Rdx / R))
    tLimit = periods * PERIOD / A0
    tag = f'droplet_Rdx{Rdx}' + (f'_a{alpha:g}' if alpha != 0.01 else '')
    runRoot = os.path.join(out, tag + '_run')

    params = dict(
        R=R, A=A0, B=B, rho0=RHO0, alpha=alpha,
        # Sun's c0 = 15 A0 R, through the shared `setupTimestep`'s Eq. (2) path
        # rather than the legacy `targetDt` back-solve.
        machTarget=MACH, referenceVelocity=A0 * R,
    )
    kw = dict(scheme=scheme, L=BOX, nx=nx, tLimit=tLimit, kernel=kernel,
              integrationScheme='rungeKutta4',
              quiet=True, store=False, progress=True, params=params)
    if video:
        kw.update(plot=True, video=True, plotBackend='matplotlib',
                  plotInterval=plotInterval, exportRoot=runRoot)

    print(f'[{tag}] running {periods:g} periods (t = {tLimit:.2f}) at nx = {nx} ...',
          flush=True)
    r = run(oscillatingDropletCase, **kw)

    rows = [x for x in r.trajectory if x.get('step', -2) >= 0]
    keys = ['step', 't', 'semiAxisA', 'semiAxisB', 'kineticEnergy',
            'linearMomentumStar', 'angularMomentumStar', 'potentialEnergy',
            'elasticEnergy', 'mechanicalEnergy', 'totalEnergy',
            'minDensity', 'maxDensity', 'densityP05', 'maxVelocity']
    cols = {k: np.array([row.get(k, np.nan) for row in rows], dtype=float)
            for k in keys}
    cols['tStar'] = cols['t'] / PERIOD          # t/T, Sun's own abscissa

    c0 = float(getattr(r.ctx.schemeConfig.fluid, 'fixedSoundSpeed', 0.0) or 0.0)
    meta = dict(
        scheme=scheme, Rdx=int(Rdx), nx=nx, alpha=alpha, kernel=kernel,
        dx=float(r.ctx.config.dx), RdxAchieved=R / float(r.ctx.config.dx),
        c0=c0, mach=(A0 * R / c0) if c0 else None, periods=periods,
        tLimit=tLimit, tReached=float(rows[-1].get('t', 0.0)) if rows else 0.0,
        sun2017Eq7Shift=bool(getattr(r.ctx.schemeConfig.shiftProperties,
                                     'sun2017Eq7Shift', False)),
        diverged=bool(r.diverged), nSteps=int(r.nSteps),
        wallTime_s=float(r.wallTime or 0.0),
    )

    npz = os.path.join(out, tag + '.npz')
    np.savez(npz, meta=json.dumps(meta), **cols)
    print(f'[{tag}] -> {npz}   diverged={r.diverged}  steps={r.nSteps}  '
          f'wall={r.wallTime:.0f}s  R/dx={meta["RdxAchieved"]:.0f}  c0={c0:.2f}',
          flush=True)


#: `analyticSolution` integrates a stiff-ish ODE at `rtol=1e-10` with a capped
#: step, which over 15 periods and ~10^5 recorded samples is seconds rather than
#: milliseconds -- and `_report` asks for the same grid several times.
_ANALYTIC_CACHE = {}


def _analytic(ts):
    """`(a, b, KE)` of the exact elliptical-drop solution at times `ts`."""
    from warpSPH.cases.oscillatingDroplet import analyticSolution
    key = (int(ts.size), float(ts[0]), float(ts[-1]))
    if key not in _ANALYTIC_CACHE:
        _ANALYTIC_CACHE[key] = analyticSolution(ts, A=A0, B=B, R=R, rho0=RHO0)
    return _ANALYTIC_CACHE[key]


def _score(meta, col):
    import numpy as np
    A = ACCEPT
    checks, m = [], {}

    def add(name, ok, detail):
        checks.append((name, bool(ok), detail))

    ts = col['t']
    ok = np.isfinite(ts)
    ts = ts[ok]
    if ts.size < 8:
        add('run produced a record', False, f'{ts.size} samples')
        return checks, m

    m['periodsReached'] = float(ts[-1] / PERIOD)

    # -- Table 1: momenta ----------------------------------------------------
    ref = SUN_TABLE1.get(int(meta['Rdx']))
    for key, refKey, unit in (('linearMomentumStar', 'linear', 'ρ A₀ R³'),
                              ('angularMomentumStar', 'angular', 'ρ A₀ R⁴')):
        vals = col[key][ok]
        worst = float(np.nanmax(np.abs(vals))) if np.isfinite(vals).any() else float('nan')
        m[f'{refKey}MomentumMax'] = worst
        if ref is None:
            add(f'{refKey} momentum', True,
                f'{worst:.2e} {unit}  (no Sun Table 1 row for R/Δx={meta["Rdx"]})')
            continue
        factor = max(worst, ref[refKey]) / min(worst, ref[refKey]) if worst else float('inf')
        m[f'{refKey}MomentumFactorVsSun'] = factor
        add(f'{refKey} momentum vs Sun Table 1',
            np.isfinite(factor) and factor <= A['momentum_factor_tol'],
            f'{worst:.2e} vs Sun {ref[refKey]:.1e} {unit} → ×{factor:.2f} '
            f'(want <= ×{A["momentum_factor_tol"]}), over '
            f'{m["periodsReached"]:.1f} periods')

    # -- Fig. 10: a(t) against the analytic solution, first periods ----------
    aExact, _bExact, _ke = _analytic(ts)
    aMeas = col['semiAxisA'][ok]
    err = aMeas - aExact
    early = ts <= A['semi_axis_rmse_periods'] * PERIOD
    m['semiAxisRmseEarly'] = float(np.sqrt(np.nanmean(err[early] ** 2))) \
        if early.any() else float('nan')
    m['semiAxisRmseAll'] = float(np.sqrt(np.nanmean(err ** 2)))
    add('semi-axis a(t) vs analytic (Fig. 10)',
        m['semiAxisRmseEarly'] <= A['semi_axis_rmse_max'],
        f"RMSE {m['semiAxisRmseEarly']:.4f} R over the first "
        f"{A['semi_axis_rmse_periods']:g} periods (want <= "
        f"{A['semi_axis_rmse_max']}); {m['semiAxisRmseAll']:.4f} over all "
        f"{m['periodsReached']:.1f}, which is dominated by accumulated phase")

    # -- Fig. 13: period, and the amplitude decay of Fig. 11 -----------------
    tPeaks, aPeaks = _peaks(ts, aMeas)
    m['nPeaks'] = int(len(tPeaks))
    if len(tPeaks) >= 3:
        periods = np.diff(tPeaks)
        # Median, not mean -- see `_peaks`: one missed peak doubles a single
        # interval, and the mean carries that straight into the answer.
        m['periodMedian'] = float(np.median(periods))
        m['periodFirst'] = float(periods[0])
        m['periodLast'] = float(periods[-1])
        m['periodRelError'] = float(m['periodMedian'] / PERIOD - 1.0)
        add('oscillation period (Fig. 13)',
            abs(m['periodRelError']) <= A['period_rel_error_max'],
            f"T = {m['periodMedian']:.4f} vs analytic {PERIOD} → "
            f"{m['periodRelError']:+.2%} (want <= "
            f"{A['period_rel_error_max']:.0%}); drifts "
            f"{m['periodFirst']:.4f} → {m['periodLast']:.4f} over "
            f"{len(tPeaks)} peaks")
        m['amplitudeDecay'] = float(1.0 - aPeaks[-1] / aPeaks[0])
        add('amplitude decay (Fig. 11 dissipation)',
            m['amplitudeDecay'] <= A['amplitude_decay_max'],
            f"a_peak {aPeaks[0]:.4f} → {aPeaks[-1]:.4f} = {m['amplitudeDecay']:.2%} "
            f"over {m['periodsReached']:.1f} periods (want <= "
            f"{A['amplitude_decay_max']:.0%}); analytic is flat at 1.9319")
    else:
        add('oscillation period (Fig. 13)', False,
            f'only {len(tPeaks)} peaks found in a(t)')

    # -- Fig. 11: the mechanical-energy loss ---------------------------------
    eM = col['mechanicalEnergy'][ok]
    e0 = eM[0]
    m['mechanicalLossPct'] = float((eM[-1] - e0) / e0 * 100.0) if e0 else float('nan')
    m['elasticEnergyEnd'] = float(col['elasticEnergy'][ok][-1])
    m['totalEnergyDriftPct'] = float(
        (col['totalEnergy'][ok][-1] - col['totalEnergy'][ok][0]) / e0 * 100.0) \
        if e0 else float('nan')
    add('mechanical energy loss (Fig. 11)',
        abs(m['mechanicalLossPct']) <= A['mechanical_loss_pct_max'],
        f"{m['mechanicalLossPct']:+.2f} % of E_M⁰ over {m['periodsReached']:.1f} "
        f"periods (Sun Fig. 11 reads {SUN_FIG11_MECHANICAL_LOSS_PCT:+.1f} % at "
        f"R/Δx = 200 — this run is R/Δx = {meta['Rdx']}, and the artificial "
        f"viscosity ν = α c₀ h / (2(n+2)) is linear in h; want |·| <= "
        f"{A['mechanical_loss_pct_max']:.0f} %)")

    # -- health --------------------------------------------------------------
    m['rhoMin'] = float(np.nanmin(col['minDensity'][ok]))
    m['rhoMax'] = float(np.nanmax(col['maxDensity'][ok]))
    lo, hi = A['density_band']
    add('drop stays intact', not meta['diverged'] and lo < m['rhoMin'] and m['rhoMax'] < hi,
        f"ρ ∈ [{m['rhoMin']:.4f}, {m['rhoMax']:.4f}], diverged={meta['diverged']}")
    return checks, m


def _peaks(t, y, minSeparation=0.5 * PERIOD):
    """`(times, values)` of the maximum-elongation peaks of `y(t)`.

    Deliberately not `scipy.signal.find_peaks` -- three lines of comparison
    keep `scipy.signal` off the dependency surface -- but all three guards
    below are load-bearing, and each replaces a failure actually observed on
    this signal:

    * **plateau-tolerant** (`>=` on the left, `>` on the right).  A strictly
      two-sided `>` silently dropped the `t = 20.49` peak of the `R/dx = 50`
      run, where two consecutive float32 samples tie at the top; one missing
      peak turns a 4.82 interval into a 9.62 one and dragged the mean period
      from 4.80 to 5.17 (+7 %), which reads as a scheme error and is not one.
    * **prominence floor**.  `a(t)` carries small-scale noise that produces
      genuine local maxima near the trough; without the floor those enter as
      half-amplitude "peaks" and halve the inferred period.
    * **median, not mean, of the intervals** (in the caller).  One spurious or
      missing peak then cannot move the answer at all.
    """
    import numpy as np
    if y.size < 3:
        return np.array([]), np.array([])
    floor = 0.5 * (float(np.nanmin(y)) + float(np.nanmax(y)))
    interior = np.where((y[1:-1] >= y[:-2]) & (y[1:-1] > y[2:])
                        & (y[1:-1] > floor))[0] + 1
    times, values = [], []
    for i in interior:
        if times and t[i] - times[-1] < minSeparation:
            # Same peak, sampled twice through noise or a plateau: keep the
            # higher sample.
            if y[i] > values[-1]:
                times[-1], values[-1] = float(t[i]), float(y[i])
            continue
        times.append(float(t[i]))
        values.append(float(y[i]))
    return np.array(times), np.array(values)


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

    def label(meta):
        return (f"δ⁺-SPH  R/Δx={meta['Rdx']}" +
                (f"  α={meta['alpha']:g}" if meta['alpha'] != 0.01 else ''))

    palette = ['#c1121f', '#0353a4', '#2a9d8f', '#e76f51', '#6a4c93']
    colours = {label(m): palette[i % len(palette)] for i, (m, _) in enumerate(runs)}

    # --- Fig. 10 / 13: semi-axis history -----------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    tMax = max(float(np.nanmax(c['t'])) for _, c in runs)
    tGrid = np.linspace(0, tMax, 4000)
    aExact, bExact, _ke = _analytic(tGrid)
    for ax, window in zip(axes, ((0.0, min(tMax, 3 * PERIOD)),
                                 (max(0.0, tMax - PERIOD), tMax))):
        sel = (tGrid >= window[0]) & (tGrid <= window[1])
        ax.plot(tGrid[sel] / PERIOD, aExact[sel], 'k--', lw=1.4, label='analytic a(t)')
        for meta, col in runs:
            ax.plot(col['t'] / PERIOD, col['semiAxisA'], lw=1.2,
                    color=colours[label(meta)], label=label(meta))
        ax.set_xlim(window[0] / PERIOD, window[1] / PERIOD)
        ax.set_xlabel('t / T')
        ax.set_ylabel('a / R')
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc='best')
    axes[0].set_title('Fig. 10 — horizontal semi-axis, first oscillations')
    axes[1].set_title('Fig. 13 — the last simulated oscillation')
    fig.suptitle('Sun et al. 2017 §4.2 — oscillating droplet', fontsize=12)
    fig.tight_layout()
    p1 = os.path.join(out, 'semi_axis.png')
    fig.savefig(p1, dpi=110)
    plt.close(fig)
    print(f'-> {p1}')

    # --- Table 1: momenta vs resolution ------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    rdx = sorted(SUN_TABLE1)
    ax.loglog(rdx, [SUN_TABLE1[r]['linear'] for r in rdx], 'o--', color='#2a9d8f',
              mfc='none', label='Sun Table 1: linear')
    ax.loglog(rdx, [SUN_TABLE1[r]['angular'] for r in rdx], 's--', color='#4361ee',
              mfc='none', label='Sun Table 1: angular')
    for meta, col in runs:
        if meta['alpha'] != 0.01:
            continue
        _c, m = _score(meta, col)
        ax.loglog([meta['Rdx']], [m.get('linearMomentumMax', np.nan)], 'o',
                  color='#2a9d8f', label='this repo: linear')
        ax.loglog([meta['Rdx']], [m.get('angularMomentumMax', np.nan)], 's',
                  color='#4361ee', label='this repo: angular')
    handles, labels = ax.get_legend_handles_labels()
    seen = dict(zip(labels, handles))
    ax.legend(seen.values(), seen.keys(), fontsize=8)
    ax.set_xlabel('R / Δx')
    ax.set_ylabel('max momentum error')
    ax.set_title('Sun et al. 2017 §4.2 Table 1 — momenta conservation')
    ax.grid(alpha=0.25, which='both')
    fig.tight_layout()
    p2 = os.path.join(out, 'table1_momenta.png')
    fig.savefig(p2, dpi=110)
    plt.close(fig)
    print(f'-> {p2}')

    lines = ['# Sun et al. 2017 §4.2 — oscillating droplet', '',
             f'![semi_axis.png]({os.path.basename(p1)})', '',
             f'![table1_momenta.png]({os.path.basename(p2)})', '',
             '## Runs', '',
             '| run | R/Δx | α | c₀ | Eq.(7) shift | periods | steps | wall (s) | diverged |',
             '|---|---|---|---|---|---|---|---|---|']
    for meta, _col in runs:
        lines.append(
            f"| {label(meta)} | {meta['RdxAchieved']:.0f} | {meta['alpha']:g} | "
            f"{meta['c0']:.1f} | {meta['sun2017Eq7Shift']} | "
            f"{meta['tReached'] / PERIOD:.1f} | {meta['nSteps']} | "
            f"{meta['wallTime_s']:.0f} | {meta['diverged']} |")
    lines.append('')

    allOk = True
    for meta, col in runs:
        checks, m = _score(meta, col)
        ok = all(c[1] for c in checks)
        allOk = allOk and ok
        lines += [f"## {label(meta)} — {sum(c[1] for c in checks)}/{len(checks)} checks",
                  '', '| check | result | detail |', '|---|---|---|']
        for name, good, detail in checks:
            lines.append(f"| {name} | {'✅' if good else '❌'} | {detail} |")
        lines += ['', '```', json.dumps(m, indent=2), '```', '']

    md = os.path.join(out, 'REPORT.md')
    with open(md, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'-> {md}   {"ALL CHECKS PASS" if allOk else "SOME CHECKS FAILED"}')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--Rdx', type=int, default=50,
                    help='R/dx; Sun Table 1 has rows at 50, 100 and 200')
    ap.add_argument('--periods', type=float, default=15.0,
                    help="oscillation periods to run; Sun's Table 1 is over 15")
    ap.add_argument('--alpha', type=float, default=0.01,
                    help='artificial viscosity; Sun uses 0.01, and 0 for the Fig. 13 leg')
    ap.add_argument('--scheme', default='sun2017DeltaSPH')
    ap.add_argument('--kernel', default='Wendland2')
    ap.add_argument('--video', action=argparse.BooleanOptionalAction, default=True,
                    help='export a video (vispy); on by default so every run is '
                         'observable, not just recorded metrics -- pass --no-video '
                         'to opt out')
    ap.add_argument('--plotInterval', type=int, default=200)
    ap.add_argument('--out', default=DEFAULT_OUT)
    ap.add_argument('--report', action='store_true')
    args = ap.parse_args(argv)

    if args.report:
        _report(args.out)
        return
    _runOne(args.Rdx, args.periods, args.out, args.scheme, args.alpha,
            args.kernel, args.video, args.plotInterval)


if __name__ == '__main__':
    main()
