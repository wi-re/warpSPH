"""Validation (`DELTASPH_VALIDATION_PLAN.md` Part 5.3): Sun et al. 2019
Sec. 3.1 -- the periodic Taylor-Green flow, benchmark no. 1 of the delta+-SPH
suite.  Scores the kinetic-energy decay, the pressure at the centre of the box
and the global volume drift against that paper's Figs. 6, 7 and 9.

Why this case
-------------
It is the only case in either the delta-SPH or the delta+-SPH validation suite
with **no boundary of any kind** -- no wall, no free surface, no inflow.  Every
other case this repo has scored (Marrone Sec. 3.1's dam break, the sloshing
tank, the hydrostatic column) measures the scheme *and* its wall closure at the
same time, and when the answer comes out 20 % low there is no way to tell which
half is responsible.  Here there is only the scheme, and there is a closed-form
answer to hold it against, so a disagreement is unambiguous.

Reference setup (Sun et al. 2019 Sec. 3.1,
`literature/sun2019_consistent-particle-shifting-delta-plus-sph.pdf`)
------------------------------------------------------------------------------
- Four counter-rotating vortices on a **periodic** `[0, L] x [0, L]` box,
  `u = U cos(kx) sin(ky)`, `v = -U sin(kx) cos(ky)`, `k = 2 pi / L`.
- `Re = U L / nu` = **100** (record to `tU/L = 1`) and **1000** (to `tU/L = 10`).
- Eq. (22): both the kinetic energy and the pressure at a fixed point decay as
  `f(t) = f(0) exp(-16 pi^2 nu t)` -- which is `4 nu k^2` at `k = 2 pi / L`, so
  `cases/tgvWeaklyCompressible.analyticDecayRate` is the same number.
- Resolutions `L/dx = 50, 200, 400, 800`; Figs. 6/7/9 all compare at
  **`L/dx = 400`**, which is therefore the resolution the acceptance bands
  below are written for.
- Initial distribution from Colagrossi's packing algorithm.  We use the case's
  own `shuffleParticles` relaxation instead -- same purpose (break the Cartesian
  lattice symmetry), different algorithm.
- Wendland C2, `h/dx = 2` (`n_h = 4.0`, support `= 4 dx`), RK4.
- `c0 = 10 U` (Sun et al. 2017 Eq. (2), `machTarget = 0.1`); note
  `sqrt(p_max/rho0) = sqrt(U^2/2) < U`, so `U` is the binding scale.

**Which curve is the target.**  Sun's Figs. 6-9 each plot *three* models: the
delta-SPH, the "delta+-SPH by Sun et al. [5]" (= Sun et al. 2017, the
`literature/sun2017_delta-plus-sph-model.pdf` scheme) and the "present
delta+-SPH" (the 2019 paper's own consistent-PST scheme, which folds the
shifting transport back into the continuity and momentum equations).  **This
repo implements the middle one** -- `schemes/deltaSPH.py` +
`modules/shifting/delta.py`'s Sun 2017 Eq. (7) law, with `correctdrhodt` /
`correctdvdt` off -- so the green `delta+-SPH (Sun et al. 2017)` curve is what
a correct implementation here has to reproduce, *including its known errors*:
a centre pressure that sits progressively above the analytic solution and a
volume error that cumulates rather than settling.  Reproducing the red "present
delta+-SPH" curve is `PST_ALE_PLAN.md`'s job, not this script's -- it needs the
ALE correction terms, and this benchmark is exactly the measurement that would
show them working.

Acceptance (`L/dx = 400`, `Re = 100`, `tU/L` in [0, 1])
-------------------------------------------------------
- kinetic energy within 6 % of Eq. (22) over the whole record (Fig. 6 shows all
  three models sitting on the analytic line at log scale);
- centre pressure decaying, and ending in `p/p(t0)` in [0.35, 0.60] at
  `tU/L = 1` -- *above* the analytic 0.206, which is the Sun-2017 delta+ signature
  Fig. 7 shows (green ~0.47);
- volume error `eps_V` in [0.05 %, 0.30 %] at `tU/L = 1` (Fig. 9 green ~0.125 %)
  and **cumulating**: `eps_V(1) / eps_V(0.3) > 1.5`;
- the delta-SPH A/B leg (`--noShifting`) instead **plateaus**
  (`eps_V(1) / eps_V(0.3) < 1.3`, Fig. 9 blue) at a higher absolute level --
  Sun's own diagnosis, that delta-SPH's volume error is a one-off initial
  rearrangement while the 2017 PST's accumulates.

Usage
-----
  # one run -> <out>/<scheme><PST>_nx<N>_Re<R>.npz
  python scripts/probe_deltaPlusTGV.py --nx 400 --Re 100 --tLimit 1.0
  python scripts/probe_deltaPlusTGV.py --nx 400 --Re 100 --noShifting   # δ-SPH leg
  python scripts/probe_deltaPlusTGV.py --nx 400 --Re 1000 --tLimit 10.0

  # combine every .npz under <out> into plots + REPORT.md
  python scripts/probe_deltaPlusTGV.py --report

Default --out: scripts/out_deltaPlusTGV/
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, 'out_deltaPlusTGV')

# --- reference flow (Sun et al. 2019 Sec. 3.1) -------------------------------
L = 1.0          # box side; makes tU/L == t and the decay exponent literally 16 pi^2 nu
U = 1.0          # max initial speed, the Re velocity scale
RHO0 = 1.0
K = 2.0 * math.pi / L                 # four vortices in the box
K_PARAM = 2.0 * K                     # `cases/tgvWeaklyCompressible.wavenumber` halves it
#: Sun's layout, not the case's default: their Fig. 8 pressure profile along
#: `y = 0.5 L` peaks at `x/L = 0, 0.5, 1` and troughs at `0.25, 0.75`, so the
#: **centre of their box is a stagnation point** (`p(t0) = +P0`) and the vortex
#: cores sit at the quarter points.  The quarter-period-shifted field is the one
#: that puts that geometry under the centre probe.  It matters: what Figs. 7-9
#: measure is a drift of the *mean* pressure, an additive offset, and dividing
#: it by a `p(t0)` of the wrong sign flips which side of the analytic curve it
#: lands on.  Measured directly -- with `phase = 0` (a vortex core under the
#: probe) this repo's own δ⁺ run reads `p/p(t0) = -0.33` at `tU/L = 1` where
#: Sun's δ⁺-2017 reads `+0.47`, purely from that.
PHASE = math.pi / 2
P0 = 0.5 * RHO0 * U ** 2              # Sun's normalising max|p| at t=0
MACH = 0.1                            # c0 = U / MACH = 10 U  (Sun et al. 2017 Eq. 2)


def decayRate(Re: float) -> float:
    """Eq. (22)'s exponent, `16 pi^2 nu` at `L = 1` -- i.e. `4 nu k^2`."""
    return 4.0 * (U * L / Re) * K ** 2


# --- Sun et al. 2019 Figs. 7 and 9, digitised by eye at L/dx = 400 -----------
# Read off the printed figures; treat as ~+/-10 % in the ordinate.  Plotted as a
# visual overlay -- the pass/fail uses the envelopes in ACCEPT.
#
# Fig. 7: p(centre) / p(centre, t=0)  vs  tU/L.
SUN_FIG7 = {
    100: {
        'δ⁺-SPH (Sun et al. 2017)': [
            (0.00, 1.00), (0.10, 0.95), (0.20, 0.88), (0.25, 0.85), (0.30, 0.80),
            (0.40, 0.72), (0.50, 0.66), (0.60, 0.60), (0.70, 0.56), (0.80, 0.53),
            (0.90, 0.50), (1.00, 0.47)],
        'δ-SPH': [
            (0.00, 1.00), (0.10, 1.05), (0.20, 1.16), (0.28, 1.21), (0.35, 1.12),
            (0.45, 1.00), (0.55, 0.92), (0.65, 0.85), (0.75, 0.80), (0.85, 0.75),
            (1.00, 0.68)],
        'present δ⁺-SPH (Sun 2019)': [
            (0.00, 1.00), (0.25, 0.70), (0.50, 0.46), (0.75, 0.33), (1.00, 0.28)],
    },
    1000: {
        'δ⁺-SPH (Sun et al. 2017)': [
            (0.0, 1.00), (0.5, 1.25), (1.0, 1.45), (2.0, 1.75), (3.0, 1.95),
            (4.0, 2.10), (6.0, 2.25), (8.0, 2.35), (10.0, 2.40)],
        'δ-SPH': [
            (0.0, 1.00), (0.3, 2.05), (1.0, 1.60), (2.0, 1.42), (4.0, 1.10),
            (6.0, 0.75), (7.0, 0.50), (8.0, 0.42), (10.0, 0.30)],
        'present δ⁺-SPH (Sun 2019)': [
            (0.0, 1.00), (2.0, 0.73), (4.0, 0.53), (6.0, 0.39), (8.0, 0.29),
            (10.0, 0.22)],
    },
}

# Fig. 9: eps_V (%)  vs  tU/L  (log ordinate).
SUN_FIG9 = {
    100: {
        'δ⁺-SPH (Sun et al. 2017)': [
            (0.05, 0.012), (0.10, 0.025), (0.20, 0.045), (0.40, 0.075),
            (0.60, 0.095), (0.80, 0.110), (1.00, 0.125)],
        'δ-SPH': [
            (0.05, 0.050), (0.10, 0.130), (0.20, 0.220), (0.30, 0.210),
            (0.50, 0.215), (0.70, 0.230), (1.00, 0.220)],
        'present δ⁺-SPH (Sun 2019)': [
            (0.05, 0.013), (0.20, 0.008), (0.50, 0.007), (1.00, 0.006)],
    },
    1000: {
        'δ⁺-SPH (Sun et al. 2017)': [
            (0.1, 0.02), (0.5, 0.09), (1.0, 0.15), (2.0, 0.28), (4.0, 0.50),
            (6.0, 0.70), (8.0, 0.88), (10.0, 1.00)],
        'δ-SPH': [
            (0.3, 0.40), (1.0, 0.35), (2.0, 0.30), (4.0, 0.21), (6.0, 0.15),
            (8.0, 0.11), (10.0, 0.09)],
        'present δ⁺-SPH (Sun 2019)': [
            (0.1, 0.012), (1.0, 0.009), (4.0, 0.007), (10.0, 0.005)],
    },
}

#: The label this repo's own delta+-SPH runs are compared against.
TARGET = 'δ⁺-SPH (Sun et al. 2017)'

# --- acceptance envelopes ----------------------------------------------------
# Written against the digitised curves rather than as fixed numbers at a fixed
# time, so a short or long record scores against the reference *at its own
# t\**.  The reference curves are all `L/dx = 400`; a coarser run is compared
# against them anyway (there is nothing else to compare against) and the report
# prints the resolution next to every number, so a miss at `L/dx = 50` is read
# as a resolution statement, not as a scheme defect.
ACCEPT = dict(
    # Fig. 6: every model sits on the analytic line at log scale.  6 % is what
    # "indistinguishable on that plot" is worth.
    #
    # NOT widened for Re = 1000, deliberately.  The `L/dx = 200`, Re = 1000 run
    # fails this at +11 % around `tU/L = 2-3` -- but Fig. 6's *right* panel was
    # checked before touching the number, and the green delta+-2017 curve
    # overlays the analytic there for the whole record (the ~7 % offset near
    # `tU/L = 10` belongs to the red "present delta+-SPH").  So the failure is a
    # real open finding, recorded in `DELTASPH_VALIDATION_PLAN.md` Sec. 5.3.1,
    # not a band that was set too tight.
    ke_rel_error_max=0.06,
    # Fig. 7 / Fig. 9: how far from the digitised δ⁺-2017 curve counts as a
    # match.  The digitisation itself is worth ~10 %; the pressure tolerance is
    # a relative error, the volume one a factor (that ordinate is logarithmic
    # and spans two decades across the three models, so a *factor* is the
    # meaningful distance, and a factor of 2.5 still separates the three curves
    # cleanly -- they are an order of magnitude apart).
    p_centre_rel_tol=0.30,
    volume_error_factor_tol=2.5,
    # The shape of that volume curve, which is the actual physics claim: the
    # 2017 PST's error *cumulates*, δ-SPH's is a one-off initial rearrangement
    # that then plateaus or decays.
    volume_growth_min=1.5,             # eps_V(tEnd) / eps_V(0.3 tEnd), PST on
    volume_growth_max_noShift=1.3,     # ... same ratio, PST off
)


def _runOne(nx: int, Re: float, tLimit: float, out: str, scheme: str,
            shifting: bool, kernel: str, shuffleIters: int, video: bool,
            plotInterval: int):
    from warpSPHBootstrap import bootstrap
    bootstrap(precision='float32')
    import numpy as np
    from warpSPH.cases.tgvWeaklyCompressible import (
        analyticCentrePressure, tgvWeaklyCompressibleCase, viscosityScales)
    from warpSPH.runner import run

    os.makedirs(out, exist_ok=True)
    nu = U * L / Re
    tag = f"{scheme}{'_pst' if shifting else '_noPst'}_nx{nx}_Re{Re:g}"
    runRoot = os.path.join(out, tag + '_run')

    params = dict(
        k=K_PARAM, phase=PHASE, uMag=U, rho0=RHO0,
        inviscid=False, nu=nu,
        # Sun et al. 2017 Eq. (2) rather than the legacy targetDt back-solve, so
        # the Mach number is 0.1 at every resolution instead of a function of nx.
        machTarget=MACH, referenceVelocity=U,
        # Start on the analytic pressure field; a uniform-rho0 start would spend
        # the first ~0.1 tU/L radiating it into existence and the acoustic
        # transient is the same size as the signal being measured.
        initialPressure=True,
        shifting=shifting,
        shuffleIters=shuffleIters,
    )
    kw = dict(scheme=scheme, L=L, nx=nx, tLimit=tLimit, kernel=kernel,
              integrationScheme='rungeKutta4',
              quiet=True, store=False, progress=True, params=params)
    if video:
        kw.update(plot=True, video=True, plotBackend='matplotlib',
                  plotInterval=plotInterval, exportRoot=runRoot)

    print(f'[{tag}] running to tU/L = {tLimit:g}  (nu = {nu:.5g}, '
          f'decay rate {decayRate(Re):.4g}) ...', flush=True)
    r = run(tgvWeaklyCompressibleCase, **kw)

    rows = [x for x in r.trajectory if x.get('step', -2) >= -1]
    keys = ['step', 't', 'kineticEnergy', 'maxVelocity', 'minDensity',
            'maxDensity', 'pCentre', 'pCentreStar', 'pCentreNnbr', 'pMean',
            'totalVolume', 'volumeError', 'pairedFraction', 'nnDistP01']
    cols = {k: np.array([row.get(k, np.nan) for row in rows], dtype=float)
            for k in keys}
    # tStar == t here (L = U = 1), but carry it explicitly so the report never
    # has to re-derive the non-dimensionalisation from the metadata.
    cols['tStar'] = cols['t'] * U / L

    scales = viscosityScales(r.ctx, r.state)
    pRef = analyticCentrePressure(r.ctx)
    meta = dict(
        scheme=scheme, shifting=bool(shifting), nx=nx, Re=float(Re), nu=nu,
        kernel=kernel, tLimit=tLimit, LdxRatio=L / float(r.ctx.config.dx),
        dx=float(r.ctx.config.dx), c0=scales['c0'], mach=U / scales['c0'],
        decayRate=decayRate(Re), pCentreRef=pRef, p0=P0,
        shuffleIters=shuffleIters,
        sun2017Eq7Shift=bool(getattr(r.ctx.schemeConfig.shiftProperties,
                                     'sun2017Eq7Shift', False)),
        tReached=float(rows[-1].get('t', 0.0)) if rows else 0.0,
        diverged=bool(r.diverged), nSteps=int(r.nSteps),
        wallTime_s=float(r.wallTime or 0.0),
    )

    npz = os.path.join(out, tag + '.npz')
    np.savez(npz, meta=json.dumps(meta), **cols)
    print(f'[{tag}] -> {npz}   diverged={r.diverged}  steps={r.nSteps}  '
          f'wall={r.wallTime:.1f}s  L/dx={meta["LdxRatio"]:.0f}  '
          f'c0={meta["c0"]:.2f}  M={meta["mach"]:.3f}', flush=True)


def _series(meta, col):
    """(t*, KE/KE0, p/p(t0), eps_V%) with the pre-run row and any NaNs dropped.

    `pCentre` is normalised by the *analytic* `p(centre, 0)` rather than by the
    run's own first sample: the diagnostic's own t=0 sample is taken before the
    scheme has ever evaluated the equation of state, so it reads a flat zero and
    would make the ratio infinite.  The analytic value is what Sun's `p(t0)`
    means anyway.
    """
    import numpy as np
    ts = col['tStar']
    ok = np.isfinite(ts) & (col['step'] >= 0) & np.isfinite(col['kineticEnergy'])
    ts = ts[ok]
    ke = col['kineticEnergy'][ok]
    keRatio = ke / ke[0] if ke.size and ke[0] > 0 else ke * np.nan
    pRef = float(meta.get('pCentreRef') or 0.0)
    pRatio = col['pCentre'][ok] / pRef if pRef else col['pCentre'][ok] * np.nan
    return ts, keRatio, pRatio, col['volumeError'][ok]


def _at(ts, ys, t):
    """`ys` at `t*`, linearly interpolated; NaN outside the record."""
    import numpy as np
    good = np.isfinite(ts) & np.isfinite(ys)
    if good.sum() < 2 or t < ts[good][0] or t > ts[good][-1]:
        return float('nan')
    return float(np.interp(t, ts[good], ys[good]))


def _rollingMedian(x, w):
    """Odd-window centred rolling median, edge-replicated."""
    import numpy as np
    w = max(1, int(w) | 1)
    if w == 1 or x.size < w:
        return x.copy()
    pad = w // 2
    xp = np.pad(x, pad, mode='edge')
    return np.median(np.lib.stride_tricks.sliding_window_view(xp, w), axis=-1)


def _score(meta, col):
    """Envelope checks against Sun et al. 2019 Figs. 6/7/9.

    Returns (checks, metrics) where checks is a list of (name, ok, detail).
    """
    import numpy as np
    A = ACCEPT
    checks, m = [], {}
    ts, keRatio, pRatio, volErr = _series(meta, col)

    def add(name, ok, detail):
        checks.append((name, bool(ok), detail))

    if ts.size < 8:
        add('run produced a record', False, f'{ts.size} samples')
        return checks, m

    tEnd = float(ts[-1])
    analytic = np.exp(-meta['decayRate'] * ts)
    m['tEnd'] = tEnd

    # -- Fig. 6: kinetic energy vs Eq. (22) ---------------------------------
    relErr = np.abs(keRatio - analytic) / np.maximum(analytic, 1e-12)
    worst = int(np.nanargmax(relErr))
    m['ke_relErrorMax'] = float(relErr[worst])
    m['ke_relErrorMaxAt'] = float(ts[worst])
    m['ke_end'] = float(keRatio[-1])
    m['ke_endAnalytic'] = float(analytic[-1])
    add('KE decay vs Eq. (22)', relErr[worst] <= A['ke_rel_error_max'],
        f"max rel. error {relErr[worst]:.3f} at t*={ts[worst]:.2f}  "
        f"(want <= {A['ke_rel_error_max']}); end {keRatio[-1]:.4f} vs "
        f"analytic {analytic[-1]:.4f}")

    # The same information as a fitted viscosity, which is the form the rest of
    # this case reports dissipation in (`effectiveViscosity`).
    good = keRatio > 0
    if good.sum() > 4:
        slope = np.polyfit(ts[good], np.log(keRatio[good]), 1)[0]
        m['nuEffOverNu'] = float(-slope / meta['decayRate'])

    # -- Fig. 7: centre pressure --------------------------------------------
    # A t*≈0.02 median takes out the acoustic ripple (period ~ h/c0) without
    # touching the hydrodynamic decay (timescale ~ 1/decayRate >= 0.6 here).
    dts = float(np.median(np.diff(ts))) if ts.size > 1 else 1.0
    pSmooth = _rollingMedian(pRatio, max(1, round(0.02 * tEnd / max(dts, 1e-12))))
    m['p_end'] = _at(ts, pSmooth, tEnd)
    m['p_endAnalytic'] = float(analytic[-1])
    # A run with the PST off is δ-SPH, so it is Sun's *blue* curve it has to
    # match, not the δ⁺ one.
    refLabel = TARGET if meta.get('shifting', True) else 'δ-SPH'
    m['refCurve'] = refLabel
    m['p_endSun'] = _at(*_refCurve(SUN_FIG7, meta['Re'], refLabel), tEnd)
    pErr = (abs(m['p_end'] - m['p_endSun']) / m['p_endSun']) if m['p_endSun'] else float('nan')
    m['p_relErrorVsSun'] = pErr
    add(f'centre pressure vs Sun Fig. 7 {refLabel}',
        np.isfinite(pErr) and pErr <= A['p_centre_rel_tol'],
        f"p/p(t0) = {m['p_end']:.3f} at t*={tEnd:.2f} vs Sun {m['p_endSun']:.3f} "
        f"→ {pErr:+.0%} (want <= {A['p_centre_rel_tol']:.0%}); "
        f"analytic {analytic[-1]:.3f}, this run L/Δx={meta['LdxRatio']:.0f} "
        f"vs Sun's 400")

    # -- Fig. 9: volume drift ------------------------------------------------
    m['volErr_end'] = _at(ts, volErr, tEnd)
    m['volErr_mid'] = _at(ts, volErr, 0.3 * tEnd)
    m['volErr_max'] = float(np.nanmax(volErr))
    m['volErr_endSun'] = _at(*_refCurve(SUN_FIG9, meta['Re'], refLabel), tEnd)
    growth = (m['volErr_end'] / m['volErr_mid']) if m['volErr_mid'] else float('nan')
    m['volErr_growth'] = growth
    volFactor = (max(m['volErr_end'], m['volErr_endSun'])
                 / min(m['volErr_end'], m['volErr_endSun'])) \
        if (m['volErr_end'] and m['volErr_endSun']) else float('nan')
    m['volErr_factorVsSun'] = volFactor
    add(f'volume error vs Sun Fig. 9 {refLabel}',
        np.isfinite(volFactor) and volFactor <= A['volume_error_factor_tol'],
        f"ε_V = {m['volErr_end']:.4f} % at t*={tEnd:.2f} vs Sun "
        f"{m['volErr_endSun']:.4f} % → ×{volFactor:.2f} (want <= "
        f"×{A['volume_error_factor_tol']}); this run L/Δx={meta['LdxRatio']:.0f} "
        f"vs Sun's 400")
    if meta.get('shifting', True):
        add('volume error cumulates (PST on)', growth > A['volume_growth_min'],
            f"ε_V(t*={tEnd:.2f}) / ε_V(t*={0.3 * tEnd:.2f}) = {growth:.2f}  "
            f"(want > {A['volume_growth_min']}; Sun: the 2017 PST's error "
            f"accumulates in time)")
    else:
        add('volume error plateaus (PST off)', growth < A['volume_growth_max_noShift'],
            f"ε_V(t*={tEnd:.2f}) / ε_V(t*={0.3 * tEnd:.2f}) = {growth:.2f}  "
            f"(want < {A['volume_growth_max_noShift']}; Sun: δ-SPH's error is "
            f"the one-off initial rearrangement)")

    # -- health --------------------------------------------------------------
    m['rhoMin'] = float(np.nanmin(col['minDensity']))
    m['rhoMax'] = float(np.nanmax(col['maxDensity']))
    add('weakly compressible', not meta['diverged']
        and 0.9 < m['rhoMin'] and m['rhoMax'] < 1.1,
        f"ρ ∈ [{m['rhoMin']:.4f}, {m['rhoMax']:.4f}], diverged={meta['diverged']}")

    return checks, m


def _refCurve(table, Re, label):
    """(t array, y array) for one digitised reference curve; empty if absent."""
    import numpy as np
    pts = table.get(int(Re), {}).get(label, [])
    if not pts:
        return np.array([]), np.array([])
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])


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
        # Runs recorded before `sun2017Eq7Shift` existed were all the historical
        # 1/8-of-Eq.-(7) shift, so the missing field reads as False, which is
        # what they were.
        pst = ('δ⁺-SPH' if meta.get('sun2017Eq7Shift') else 'δ⁺-SPH [⅛ Eq.7]') \
            if meta.get('shifting', True) else 'δ-SPH'
        return f"{pst}  L/Δx={meta['LdxRatio']:.0f}  Re={meta['Re']:g}"

    # Colour by *family* (δ-SPH / δ⁺ at Eq. (7) / δ⁺ at the historical ⅛), not
    # by enumeration order, so the three legs stay visually separable however
    # many resolutions of each happen to be in the directory; resolution is
    # then the shade within a family.
    families = {False: ['#4361ee', '#0353a4', '#023047'],          # δ-SPH
                True: ['#e07a5f', '#c1121f', '#6a040f'],           # δ⁺, ⅛ Eq. (7)
                'eq7': ['#52b788', '#2a9d8f', '#14532d']}          # δ⁺, Eq. (7)
    counters = {}
    colours = {}
    for meta, _ in runs:
        key = 'eq7' if (meta.get('shifting', True) and meta.get('sun2017Eq7Shift')) \
            else bool(meta.get('shifting', True))
        i = counters.get(key, 0)
        counters[key] = i + 1
        colours[label(meta)] = families[key][i % len(families[key])]
    reynolds = sorted({int(m['Re']) for m, _ in runs})

    # --- one 3-panel column per Reynolds number ----------------------------
    figs = []
    for Re in reynolds:
        sub = [(m, c) for m, c in runs if int(m['Re']) == Re]
        fig, axes = plt.subplots(3, 1, figsize=(9, 11))
        tMax = max(float(np.nanmax(c['tStar'])) for _, c in sub)
        tGrid = np.linspace(0, tMax, 200)
        analytic = np.exp(-decayRate(Re) * tGrid)

        # KE (log)
        ax = axes[0]
        ax.semilogy(tGrid, analytic, 'k--', lw=1.4, label='analytic, Eq. (22)')
        for meta, col in sub:
            ts, ke, _p, _v = _series(meta, col)
            ax.semilogy(ts, ke, color=colours[label(meta)], lw=1.6, label=label(meta))
        ax.set_ylabel('$E_k / E_{k0}$')
        ax.set_title(f'Fig. 6 — kinetic energy,  Re = {Re}')

        # centre pressure
        ax = axes[1]
        ax.plot(tGrid, analytic, 'k--', lw=1.4, label='analytic, Eq. (22)')
        for name, style in (('δ⁺-SPH (Sun et al. 2017)', dict(marker='D', c='#2a9d8f')),
                            ('δ-SPH', dict(marker='v', c='#4361ee')),
                            ('present δ⁺-SPH (Sun 2019)', dict(marker='o', c='#8d99ae'))):
            rx, ry = _refCurve(SUN_FIG7, Re, name)
            if rx.size:
                ax.plot(rx, ry, ls=':', lw=1.1, ms=4, mfc='none',
                        marker=style['marker'], color=style['c'],
                        label=f'Sun Fig. 7: {name}')
        for meta, col in sub:
            ts, _ke, p, _v = _series(meta, col)
            dts = float(np.median(np.diff(ts))) if ts.size > 1 else 1.0
            ax.plot(ts, p, color=colours[label(meta)], lw=0.5, alpha=0.30)
            ax.plot(ts, _rollingMedian(p, max(1, round(0.02 * tMax / max(dts, 1e-12)))),
                    color=colours[label(meta)], lw=1.8, label=label(meta))
        ax.set_ylabel('$p / p(t_0)$  at the box centre')
        ax.set_title(f'Fig. 7 — centre pressure,  Re = {Re}')

        # volume error (log)
        ax = axes[2]
        for name, style in (('δ⁺-SPH (Sun et al. 2017)', dict(marker='D', c='#2a9d8f')),
                            ('δ-SPH', dict(marker='v', c='#4361ee')),
                            ('present δ⁺-SPH (Sun 2019)', dict(marker='o', c='#8d99ae'))):
            rx, ry = _refCurve(SUN_FIG9, Re, name)
            if rx.size:
                ax.semilogy(rx, ry, ls=':', lw=1.1, ms=4, mfc='none',
                            marker=style['marker'], color=style['c'],
                            label=f'Sun Fig. 9: {name}')
        for meta, col in sub:
            ts, _ke, _p, v = _series(meta, col)
            ax.semilogy(ts, np.maximum(v, 1e-4), color=colours[label(meta)],
                        lw=1.6, label=label(meta))
        ax.set_ylabel(r'$\epsilon_V$ (%)')
        ax.set_title(f'Fig. 9 — total volume drift,  Re = {Re}')

        for ax in axes:
            ax.set_xlabel('$tU/L$')
            ax.set_xlim(0, tMax)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=7, loc='best')
        fig.suptitle('Sun et al. 2019 §3.1 — Taylor-Green flow', fontsize=12)
        fig.tight_layout()
        path = os.path.join(out, f'tgv_Re{Re}.png')
        fig.savefig(path, dpi=110)
        plt.close(fig)
        figs.append(path)
        print(f'-> {path}')

    # --- markdown -----------------------------------------------------------
    lines = ['# Sun et al. 2019 §3.1 — Taylor-Green flow', '',
             'Target curve: **δ⁺-SPH (Sun et al. 2017)** — the scheme this repo '
             'implements. See `scripts/probe_deltaPlusTGV.py`’s docstring for why '
             'that is the right curve and not the paper’s own "present δ⁺-SPH".',
             '']
    for path in figs:
        lines += [f'![{os.path.basename(path)}]({os.path.basename(path)})', '']

    lines += ['## Runs', '',
              '| run | L/Δx | Re | c₀ | M | steps | wall (s) | diverged |',
              '|---|---|---|---|---|---|---|---|']
    for meta, _col in runs:
        lines.append(
            f"| {label(meta)} | {meta['LdxRatio']:.0f} | {meta['Re']:g} | "
            f"{meta['c0']:.2f} | {meta['mach']:.3f} | {meta['nSteps']} | "
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
    ap.add_argument('--nx', type=int, default=400,
                    help='particles per side; L/dx. Sun compares at 400.')
    ap.add_argument('--Re', type=float, default=100.0, help='U L / nu (100 or 1000)')
    ap.add_argument('--tLimit', type=float, default=None,
                    help='record length in tU/L; default 1.0 at Re=100, 10.0 at Re=1000')
    ap.add_argument('--scheme', default='sun2017DeltaSPH',
                    help="'sun2017DeltaSPH' (frozen diffusion across RK4 sub-stages, "
                         "Sun et al. 2017 §2) or 'deltaSPH' for the un-frozen A/B")
    ap.add_argument('--noShifting', action='store_true',
                    help='PST off — the δ-SPH leg of Sun’s three-model comparison')
    ap.add_argument('--kernel', default='Wendland2',
                    help='Sun/Marrone/DualSPHysics all use Wendland C2')
    ap.add_argument('--shuffleIters', type=int, default=128,
                    help="initial-distribution relaxation (stands in for Sun's packing)")
    ap.add_argument('--video', action='store_true')
    ap.add_argument('--plotInterval', type=int, default=60)
    ap.add_argument('--out', default=DEFAULT_OUT)
    ap.add_argument('--report', action='store_true',
                    help='combine every .npz under --out into plots + REPORT.md')
    args = ap.parse_args(argv)

    if args.report:
        _report(args.out)
        return
    tLimit = args.tLimit if args.tLimit is not None else (1.0 if args.Re <= 500 else 10.0)
    _runOne(args.nx, args.Re, tLimit, args.out, args.scheme, not args.noShifting,
            args.kernel, args.shuffleIters, args.video, args.plotInterval)


if __name__ == '__main__':
    main()
