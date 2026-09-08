"""English et al. 2022 §4.1 -- still water on a bed with a sharp-cornered wedge.
(`DELTASPH_VALIDATION_PLAN.md` §5.2.1.)

The cleanest mDBC discriminator in the suite: still water, so the exact answer
is the analytic hydrostatic profile `p = rho0 g (H - z)` and *every* departure
is the wall closure alone -- no scheme/wall confound like the dam break.

Paper spec (English et al. 2022, Comp. Part. Mech. 9:911-925):

  * 2D closed tank, 2.4 m x 1.2 m
  * trigonal wedge, bottom-centre, height 0.24 m, sharp apex into the fluid
  * initial water height H = 0.5 m
  * h/dp = 2 ; dp = 0.02 m and 0.01 m
  * 4 s of physical time (noise checks at 20 s / 200 s)
  * delta-SPH + mDBC, NO particle shifting (still water)

Measures (their Figs. 4-5):

  a) per-particle `p/(rho0 g H)` vs `z/H` at t = 4 s -- must stay on the
     hydrostatic line down to the solid surface, INCLUDING at the wedge
     corners;
  b) `sum 1/2 m |v|^2` vs t, log scale -- must stay small and not grow;
  c) noise onset time.

This script drives `dambreak` (gravity on, `fluidWidth = 1.0` so the tank is a
still pool with no column to collapse, `shifting = off`), optionally with the
`equilateralBottom` triangle preset as the wedge.

Sequencing (`DELTASPH_VALIDATION_PLAN.md` §5.2.1):
  (a) --no-wedge         flat-wall still-water baseline           <- START HERE
  (b) --wedge            current ghost placement, record failure
  (c) fix (A) surface-projected fixed-depth ghost, re-run
  (d) fix (B) corner rule, re-run
  (e) --tilt DEG         a tilted flat plate: isolated test of (A)

Usage:
  python scripts/probe_englishWedge.py --dp 0.02 --no-wedge
  python scripts/probe_englishWedge.py --dp 0.01 --wedge --tLimit 4.0 --video
  python scripts/probe_englishWedge.py --report            # rebuild plots/REPORT
"""
from __future__ import annotations

import argparse
import json
import os

# -- English §4.1 geometry (SI, metres / seconds) -------------------------------
TANK_W = 2.4
TANK_H = 1.2          # `L` in dambreak's terms (the vertical length unit)
WATER_H = 0.5         # initial still-water depth
WEDGE_H = 0.24        # wedge apex height above the bed
G = 9.81
RHO0 = 1.0            # the case is non-dimensional in density; p is scaled by rho0 g H

#: `equilateralBottom` preset (`sdEquilateralTriangle`, `aspectRatio = 2`)
#: numerically probed: apex height above the bed scales ~linearly with
#: `maxExtent`, apex z = 0.248 at maxExtent = 0.30. So for English's 0.24 m:
WEDGE_MAX_EXTENT = 0.30 * (WEDGE_H / 0.248)

#: c0 = c0Ratio * sqrt(g H). Still water needs no violent-impact headroom;
#: 20 sqrt(gH) (Mach ~ 0.05 on any wave that does appear) is the stiff end of
#: the "10 sqrt(gH)-class" the plan spec calls for and gives the sharpest
#: hydrostatic gradient. Overridable.
C0_RATIO = 20.0

DEFAULT_OUT = os.path.join(os.path.dirname(__file__), 'out_englishWedge')


def _runOne(dp, wedge, tilt, tLimit, c0Ratio, out, video, plotInterval, scheme):
    from warpSPHBootstrap import bootstrap
    bootstrap(precision='float32')
    import numpy as np
    from warpSPH.cases.dambreak import dambreakCase
    from warpSPH.runner import run

    os.makedirs(out, exist_ok=True)
    nx = int(round(TANK_H / dp))                 # dp = L / nx
    sqrt_gH = (G * WATER_H) ** 0.5
    machTarget = 1.0 / c0Ratio                   # c0 = referenceVelocity / machTarget

    tag = (f'{scheme}_dp{dp:g}'
           + ('_wedge' if wedge else '_flat')
           + (f'_tilt{tilt:g}' if tilt else ''))
    runRoot = os.path.join(out, tag + '_run')

    params = dict(
        W=TANK_W,
        fillRatio=WATER_H / TANK_H,              # still pool, depth H
        fluidWidth=1.0,                          # full tank width -> no column, no collapse
        gravityMagnitude=G,
        disableGravity=False,
        # Sun Eq. (2) c0 pick: c0 = referenceVelocity / machTarget = c0Ratio*sqrt(gH)
        referenceVelocity=sqrt_gH,
        machTarget=machTarget,
        # English §4.1 is plain delta-SPH + mDBC, no PST (still water).
        shifting='off',
        # Start on the hydrostatic density profile -- without this the pool
        # rings for seconds while it compresses into its own pressure field,
        # and the ring damps slower at finer dx (delta-SPH diffusion ~ delta h
        # c0). See `DELTASPH_VALIDATION_PLAN.md` §5.2.1.
        hydrostaticInit=True,
        pressureProbeHeights=[],                 # per-particle profile, not wall probes
    )
    if wedge:
        params.update(
            obstacleActive=True,
            obstacleType='equilateralBottom',
            offsetX=0.0,                          # bottom-centre
            aoa=float(tilt),
            # `maxExtent` calibrated so the apex reaches WEDGE_H above the bed
            # (numeric SDF probe -- see WEDGE_MAX_EXTENT).
            maxExtent=WEDGE_MAX_EXTENT,
        )

    kw = dict(scheme=scheme, L=TANK_H, nx=nx, tLimit=tLimit,
              quiet=True, store=False, progress=True, params=params)
    if video:
        kw.update(plot=True, video=True, plotBackend='matplotlib',
                  plotInterval=plotInterval, exportRoot=runRoot)

    print(f'[{tag}] dp={dp} nx={nx} c0={c0Ratio:g}sqrt(gH)={c0Ratio*sqrt_gH:.2f}  '
          f'running to t={tLimit}s ...', flush=True)
    r = run(dambreakCase, **kw)

    # -- per-step scalars (KE time series, Fig. 5) -----------------------------
    rows = [x for x in r.trajectory if x.get('step', -2) >= -1]
    series = {k: np.array([row.get(k, np.nan) for row in rows], dtype=float)
              for k in ('step', 't', 'kineticEnergy', 'maxVelocity',
                        'minDensity', 'maxDensity', 'nPenetrating', 'maxPenetrationDx')}

    # -- final per-particle state (p vs z, Fig. 4) ---------------------------
    st = r.state.state
    pos = st.positions.detach().cpu().numpy()
    kinds = st.kinds.detach().cpu().numpy()
    dens = st.densities.detach().cpu().numpy()
    pres = (st.pressures.detach().cpu().numpy() if st.pressures is not None
            else np.full(len(pos), np.nan))
    fluid = kinds == 0
    # tank bed is at y = -L/2; z measured up from the bed
    bed_y = -TANK_H / 2.0
    z = pos[fluid, 1] - bed_y
    final = dict(
        x=pos[fluid, 0].astype(np.float32), z=z.astype(np.float32),
        p=pres[fluid].astype(np.float32), rho=dens[fluid].astype(np.float32),
    )

    dx = float(r.ctx.config.dx)
    c0 = float(getattr(r.ctx.schemeConfig.fluid, 'fixedSoundSpeed', 0.0) or 0.0)
    tReached = float(rows[-1].get('t', 0.0)) if rows else 0.0
    meta = dict(
        scheme=scheme, dp=dp, nx=nx, dx=dx, wedge=bool(wedge), tilt=float(tilt),
        c0Ratio=float(c0Ratio), c0=c0, tLimit=tLimit, tReached=tReached,
        WATER_H=WATER_H, TANK_W=TANK_W, TANK_H=TANK_H, WEDGE_H=WEDGE_H, G=G,
        rho0=RHO0, bed_y=bed_y,
        # wedge preset params, so `_score` / `_report` can rebuild the obstacle
        # SDF and localise residuals to its faces / apex / base corners.
        wedgeMaxExtent=float(WEDGE_MAX_EXTENT), wedgeAspectRatio=2.0,
        wedgeOffsetX=0.0,
        shiftActive=bool(getattr(r.ctx.schemeConfig.shiftProperties, 'active', False)),
        diverged=bool(r.diverged), nSteps=int(r.nSteps),
        wallTime_s=float(r.wallTime or 0.0),
    )

    npz = os.path.join(out, tag + '.npz')
    np.savez(npz, meta=json.dumps(meta), **{f's_{k}': v for k, v in series.items()},
             **{f'f_{k}': v for k, v in final.items()})
    print(f'[{tag}] -> {npz}  diverged={r.diverged}  steps={r.nSteps}  '
          f'wall={r.wallTime:.1f}s  c0={c0:.2f}', flush=True)

    if video and r.videoPath and os.path.exists(r.videoPath):
        import shutil
        for f in ('output.mp4', 'out.gif'):
            s = os.path.join(os.path.dirname(r.videoPath), f)
            if os.path.exists(s):
                shutil.copy(s, os.path.join(out, f'{tag}_{f}'))

    _score(meta, series, final, verbose=True)
    return npz


def _wedgeGeometry(meta):
    """`(sdf_fn, apex_xy, (leftCorner_xy, rightCorner_xy))` for the wedge.

    Rebuilds the `equilateralBottom` obstacle SDF from the params stored in
    `meta`, and locates the apex (highest solid point) and base corners (solid
    meets the bed) by a one-off coarse grid evaluation. `sdf_fn` takes an
    (N, 2) world-xy array and returns the signed distance (>0 outside the
    wedge). Torch under the hood; the wrapper is numpy in/out.
    """
    import numpy as np
    import torch
    from warpSPH.caseUtils.weaklyCompressible import buildObstacleSDF
    L, W, bedY = meta['TANK_H'], meta['TANK_W'], meta['bed_y']
    off = -L / 2 + meta['wedgeMaxExtent'] / 4                 # preset's offsetY
    sdf_t = buildObstacleSDF('equilateralTriangle', meta['wedgeOffsetX'], off,
                             meta['wedgeMaxExtent'], meta['wedgeAspectRatio'],
                             meta.get('tilt', 0.0), None, None, L, W)

    def sdf_fn(xy):
        t = torch.as_tensor(np.asarray(xy), dtype=torch.float32)
        return sdf_t(t).detach().cpu().numpy()

    gx = np.linspace(-0.8, 0.8, 1201)
    gy = np.linspace(bedY - 0.02, bedY + meta['WEDGE_H'] + 0.1, 400)
    XX, YY = np.meshgrid(gx, gy, indexing='ij')
    inside = sdf_fn(np.stack([XX.ravel(), YY.ravel()], axis=1)).reshape(XX.shape) < 0
    ys, xs = YY[inside], XX[inside]
    apex = (0.0, float(ys.max()))
    at_bed = inside & (np.abs(YY - bedY) < (gy[1] - gy[0]))
    bx = XX[at_bed]
    hw = float(np.abs(bx).max()) if bx.size else meta['wedgeMaxExtent']
    return sdf_fn, apex, ((-hw, bedY), (hw, bedY))


def _score(meta, series, final, verbose=False):
    """Grade against English Figs. 4-5. Returns (checks, metrics)."""
    import numpy as np
    H, g, rho0 = meta['WATER_H'], meta['G'], meta['rho0']
    dx = meta['dx']
    pgH = rho0 * g * H

    z, p = final['f_z'] if 'f_z' in final else final['z'], \
        final['f_p'] if 'f_p' in final else final['p']
    x = final['f_x'] if 'f_x' in final else final['x']
    rho = final['f_rho'] if 'f_rho' in final else final['rho']

    # analytic hydrostatic, gauge zero at the free surface z = H
    p_exact = rho0 * g * np.clip(H - z, 0.0, None)
    resid = (p - p_exact) / pgH

    # bulk = at least 2 dx below the surface and 2 dx off any wall/wedge.
    wall_margin = 2.0 * dx
    bulk = (z < H - wall_margin) & (z > wall_margin) \
        & (np.abs(x) < meta['TANK_W'] / 2 - wall_margin)
    near_wall = (z <= wall_margin) | (np.abs(x) >= meta['TANK_W'] / 2 - wall_margin)

    m = dict(
        nFluid=int(len(z)),
        rmseBulk=float(np.sqrt(np.nanmean(resid[bulk] ** 2))) if bulk.any() else float('nan'),
        rmseNearWall=float(np.sqrt(np.nanmean(resid[near_wall] ** 2))) if near_wall.any() else float('nan'),
        maxAbsResidNearWall=float(np.nanmax(np.abs(resid[near_wall]))) if near_wall.any() else float('nan'),
        rhoMin=float(np.nanmin(rho)), rhoMax=float(np.nanmax(rho)),
    )

    # -- wedge-localised residuals ----------------------------------------
    # The global near-wall RMSE averages the wedge's ~20 bad particles across
    # the ~4000 near-wall particles of the whole tank, so it cannot see a
    # corner hot spot. Rebuild the obstacle SDF and score bands ON the wedge.
    # RMSE, not max |resid|: each corner band is only ~16 particles at
    # dp = 0.02, so the worst single particle is a noisy statistic (a first
    # A/B showed 0.02 run-to-run scatter in the corner max). RMSE over the band
    # is what the face and bulk checks already use.
    m['rmseWedgeFace'] = float('nan')
    m['rmseApex'] = m['maxResidApex'] = float('nan')
    m['rmseBaseCorner'] = m['maxResidBaseCorner'] = float('nan')
    if meta.get('wedge'):
        try:
            sdfW, apex, corners = _wedgeGeometry(meta)
            xy = np.stack([x, z + meta['bed_y']], axis=1)   # back to world y
            dW = sdfW(xy)                                    # signed dist to wedge
            face = (dW > 0.3 * dx) & (dW < 3.0 * dx)         # a band just off the faces
            if face.any():
                m['rmseWedgeFace'] = float(np.sqrt(np.nanmean(resid[face] ** 2)))
            R = 4.0 * dx
            dApex = np.hypot(xy[:, 0] - apex[0], xy[:, 1] - apex[1])
            near_apex = (dApex < R) & (dW > 0)
            if near_apex.any():
                m['rmseApex'] = float(np.sqrt(np.nanmean(resid[near_apex] ** 2)))
                m['maxResidApex'] = float(np.nanmax(np.abs(resid[near_apex])))
            dc = np.minimum(
                np.hypot(xy[:, 0] - corners[0][0], xy[:, 1] - corners[0][1]),
                np.hypot(xy[:, 0] - corners[1][0], xy[:, 1] - corners[1][1]))
            near_corner = (dc < R) & (dW > 0)
            if near_corner.any():
                m['rmseBaseCorner'] = float(np.sqrt(np.nanmean(resid[near_corner] ** 2)))
                m['maxResidBaseCorner'] = float(np.nanmax(np.abs(resid[near_corner])))
        except Exception as exc:                                # noqa: BLE001
            m['wedgeGeometryError'] = f'{type(exc).__name__}: {exc}'

    ke = series['s_kineticEnergy'] if 's_kineticEnergy' in series else series['kineticEnergy']
    t = series['s_t'] if 's_t' in series else series['t']
    ok = np.isfinite(ke) & np.isfinite(t)
    ke, t = ke[ok], t[ok]
    # The fluid is released from rho0 / zero pressure at t=0, so there is always
    # an O(0.1 s) settling transient whose KE peak is not what English Fig. 5
    # grades -- that figure is a log-scale history whose *level* (mDBC << DBC)
    # and *trend* (flat or decaying, not growing) are the point. So score the
    # settled tail (last 25 % of the record) and the second-half slope, and
    # only *report* the transient peak.
    m['keEnd'] = float(ke[-1]) if ke.size else float('nan')
    m['keMax'] = float(np.nanmax(ke)) if ke.size else float('nan')
    tail = t > 0.75 * t[-1] if t.size else np.zeros(0, bool)
    m['keSettled'] = float(np.nanmean(ke[tail])) if tail.any() else float('nan')
    if ke.size > 10:
        half = t > 0.5 * t[-1]
        m['keTrendSecondHalf'] = float(np.polyfit(t[half], ke[half], 1)[0])
    else:
        m['keTrendSecondHalf'] = float('nan')

    pen = series['s_maxPenetrationDx'] if 's_maxPenetrationDx' in series else series['maxPenetrationDx']
    m['maxPenetrationDx'] = float(np.nanmax(pen)) if np.isfinite(pen).any() else float('nan')

    checks = [
        ('hydrostatic profile, bulk', m['rmseBulk'] <= 0.03,
         f"RMSE {m['rmseBulk']:.4f} of rho g H  (want <= 0.03)"),
        ('hydrostatic profile, near wall/bed', m['rmseNearWall'] <= 0.08,
         f"RMSE {m['rmseNearWall']:.4f}, max |resid| {m['maxAbsResidNearWall']:.3f}  (want RMSE <= 0.08)"),
        ('settled kinetic energy small', m['keSettled'] <= 1e-4 * pgH * m['nFluid'] * dx * dx,
         f"KE tail-mean {m['keSettled']:.3e}  (want <= {1e-4 * pgH * m['nFluid'] * dx * dx:.3e}); "
         f"transient peak {m['keMax']:.3e}"),
        # Growth only counts as a failure if the settled KE is also
        # meaningfully above the numerical floor -- otherwise this fires on a
        # +1e-5 drift while KE itself is ~1e-5, which is rest, not instability.
        # Floor: 10 % of the "settled KE small" ceiling.
        ('kinetic energy not growing',
         not (m['keTrendSecondHalf'] > 0
              and m['keSettled'] > 0.1 * (1e-4 * pgH * m['nFluid'] * dx * dx)
              and m['keTrendSecondHalf'] * meta['tReached'] > 0.5 * m['keSettled']),
         f"2nd-half dKE/dt {m['keTrendSecondHalf']:.3e}, settled KE {m['keSettled']:.3e}"),
        ('no wall penetration', m['maxPenetrationDx'] <= 1.0,
         f"{m['maxPenetrationDx']:.2f} dx  (want <= 1.0)"),
        ('weakly compressible', (not meta['diverged']) and m['rhoMin'] > 0.9 and m['rhoMax'] < 1.1,
         f"rho in [{m['rhoMin']:.4f}, {m['rhoMax']:.4f}], diverged={meta['diverged']}"),
    ]
    if meta.get('wedge'):
        # English Fig. 3/4: the hydrostatic pressure must be accurate at the
        # wedge corners, not just on average. The target is corner accuracy
        # *comparable to the flat wall* -- the flat still tank reads near-wall
        # RMSE ~1.5 % and max |resid| ~3 %, so a corner at 8-11 % (the current
        # ghost placement, `out_englishWedge_before/`) is a real 3-4x penalty
        # and should fail. Face band <= 5 %, point-wise corner residuals <= 6 %
        # (2x the flat-wall max, allowing some corner penalty but not 4x).
        # These are the numbers fix (A) / (B) have to move -- see
        # `DELTASPH_VALIDATION_PLAN.md` §5.2.1.
        checks += [
            ('hydrostatic profile, wedge faces', m['rmseWedgeFace'] <= 0.05,
             f"RMSE {m['rmseWedgeFace']:.4f} in a 3-dx band off the sloped faces  (want <= 0.05)"),
            ('hydrostatic profile, wedge apex', not (m['rmseApex'] > 0.03),
             f"RMSE {m['rmseApex']:.4f} (max {m['maxResidApex']:.3f}) within 4 dx of the apex  (want RMSE <= 0.03)"),
            ('hydrostatic profile, wedge base corners',
             not (m['rmseBaseCorner'] > 0.03),
             f"RMSE {m['rmseBaseCorner']:.4f} (max {m['maxResidBaseCorner']:.3f}) within 4 dx of a base corner  (want RMSE <= 0.03)"),
        ]
    if verbose:
        npass = sum(c[1] for c in checks)
        print(f"  {npass}/{len(checks)} checks")
        for n, okk, det in checks:
            print(('  PASS ' if okk else '  FAIL '), n, '|', det)
    return checks, m


def _report(out):
    import glob
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    runs = []
    for npz in sorted(glob.glob(os.path.join(out, '*.npz'))):
        d = np.load(npz, allow_pickle=True)
        meta = json.loads(str(d['meta']))
        series = {k[2:]: d[k] for k in d.files if k.startswith('s_')}
        final = {k: d[k] for k in d.files if k.startswith('f_')}
        runs.append((meta, series, final, os.path.basename(npz)[:-4]))
    if not runs:
        print(f'no runs in {out}'); return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for meta, series, final, name in runs:
        H, g, rho0 = meta['WATER_H'], meta['G'], meta['rho0']
        pgH = rho0 * g * H
        z, p = final['f_z'], final['f_p']
        axes[0].scatter(p / pgH, z / H, s=2, alpha=0.3, label=name)
    zz = np.linspace(0, 1, 50)
    axes[0].plot(1 - zz, zz, 'k--', lw=1.5, label='hydrostatic  p = rho g (H-z)')
    axes[0].set_xlabel('p / (rho0 g H)'); axes[0].set_ylabel('z / H')
    axes[0].set_title('English Fig. 4 -- pressure vs depth at t = 4 s')
    axes[0].legend(fontsize=7); axes[0].grid(alpha=0.25)

    for meta, series, final, name in runs:
        t = series['t']; ke = series['kineticEnergy']
        ok = np.isfinite(t) & np.isfinite(ke)
        axes[1].semilogy(t[ok], np.maximum(ke[ok], 1e-30), lw=1.2, label=name)
    axes[1].set_xlabel('t  [s]'); axes[1].set_ylabel('sum 1/2 m |v|^2')
    axes[1].set_title('English Fig. 5 -- fluid kinetic energy')
    axes[1].legend(fontsize=7); axes[1].grid(alpha=0.25, which='both')

    fig.tight_layout()
    fig.savefig(os.path.join(out, 'pressure_and_KE.png'), dpi=110)
    print('->', os.path.join(out, 'pressure_and_KE.png'))

    lines = ['# English 2022 §4.1 -- still water on a wedge bed', '']
    for meta, series, final, name in runs:
        checks, m = _score(meta, series, final)
        npass = sum(c[1] for c in checks)
        lines.append(f'## `{name}`  ({npass}/{len(checks)})')
        lines.append(f"- dp = {meta['dp']}, nx = {meta['nx']}, c0 = {meta['c0']:.1f}, "
                     f"wedge = {meta['wedge']}, tilt = {meta['tilt']}, "
                     f"tReached = {meta['tReached']:.2f} s, diverged = {meta['diverged']}")
        for n, ok, det in checks:
            lines.append(f"  - {'PASS' if ok else 'FAIL'}  **{n}** -- {det}")
        lines.append('')
    md = os.path.join(out, 'REPORT.md')
    open(md, 'w').write('\n'.join(lines) + '\n')
    print('->', md)
    print('\n'.join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dp', type=float, default=0.02, help='initial particle spacing (m)')
    w = ap.add_mutually_exclusive_group()
    w.add_argument('--wedge', dest='wedge', action='store_true', help='place the wedge (step b+)')
    w.add_argument('--no-wedge', dest='wedge', action='store_false', help='flat tank (step a)')
    ap.set_defaults(wedge=False)
    ap.add_argument('--tilt', type=float, default=0.0,
                    help='wedge angle of attack in degrees; with --no-wedge, ignored '
                         '(a tilted flat plate variant is step e)')
    ap.add_argument('--tLimit', type=float, default=4.0, help='physical seconds (English: 4)')
    ap.add_argument('--c0Ratio', type=float, default=C0_RATIO)
    ap.add_argument('--scheme', default='deltaSPH')
    ap.add_argument('--video', action='store_true')
    ap.add_argument('--plotInterval', type=int, default=40)
    ap.add_argument('--out', default=DEFAULT_OUT)
    ap.add_argument('--report', action='store_true')
    args = ap.parse_args(argv)

    if args.report:
        _report(args.out)
        return
    _runOne(args.dp, args.wedge, args.tilt, args.tLimit, args.c0Ratio,
            args.out, args.video, args.plotInterval, args.scheme)


if __name__ == '__main__':
    main()
