"""Validation (`DELTASPH_VALIDATION_PLAN.md` Sec. 5.2.2): Marrone et al. 2011
Sec. 3.4 / Fig. 19 -- dam break flow against a **sharp-edged obstacle**, with a
concave quarter-circle fillet rounding the tank's downstream bottom corner.

The point of this case in the plan: it merges everything the boundary handling
has to get right in one geometry -- a convex 45deg edge that ejects a violent
jet, two re-entrant (concave) corners at the obstacle toe and roof/back-face,
and a genuinely *curved* concave wall (the fillet, radius H). Marrone uses it to
show the fixed-ghost-particle technique copes with "curvilinear parts and
convex/concave angles". The acceptance goal here is a method that runs it
**stably, without significant boundary penetration, and converges** with
resolution -- pressure-trace scoring against Colicchio's Level-Set solver and
Wagner's P1 peak (~36.7 rho g H) is a later refinement.

Reference (Marrone 2011 Fig. 19, `literature/marrone2011_*.pdf`), lengths in the
obstacle height H:

  * tank 10 H x 8 H, closed box, free-slip walls, inviscid (Sec. 3.4.1);
  * water column 3 H wide x 2.4 H tall in the upstream bottom corner;
  * obstacle on the floor: 45deg edge apex at x = 5 H (height H), toe at 6 H,
    vertical back face at 7 H;
  * concave fillet radius H from x = 9 H (floor) to x = 10 H (height H, at the
    downstream wall);
  * probes P1-P3 on the edge, P4-P6 on the roof, P7-P9 on the fillet;
  * space resolutions H/dx = 33.5 / 67 / 134 (Marrone Fig. 20). This script
    uses H/dx = 32 / 64 / 128 (`--nx 256 / 512 / 1024`, H/dx = nx / 8) so H is
    an integer number of spacings and the obstacle / fillet reference points
    land on the lattice;
  * sound speed c0 = c0Ratio * sqrt(g H); Marrone's Fig. 21 pair is
    c0Ratio = 28.3 (M ~ 0.085) and 56.6 (M ~ 0.043). Set through the Sun 2017
    Eq. (2) `machTarget` path with referenceVelocity = 1.95 sqrt(g H)
    (dam-break front speed, as in `probe_deltaSPHMarrone.py`);
  * non-dimensional: H = 1, g = 9.81, so t* = t sqrt(g/H) = t / 0.319 s.

Geometry lives in `caseUtils/weaklyCompressible.py` as the `marroneSharpEdge`
obstacle preset (`_marroneSharpEdgeSDF`): the obstacle polygon **unioned with**
the fillet lens, fed through `dambreak`'s normal single-obstacle SDF path.

Usage
-----
  # stability / penetration run  ->  <out>/<tag>.npz  (tag encodes scheme/nx/c0/...)
  python scripts/probe_deltaSPHMarrone34.py --nx 256 --tStar 4 --video
  python scripts/probe_deltaSPHMarrone34.py --nx 512 --tStar 7.4

  # the rounded tank corner on its own -- drop the sharp-edged obstacle
  python scripts/probe_deltaSPHMarrone34.py --nx 256 --tStar 7 --cornerOnly

  # delta+-SPH with particle shifting (the dataset-generation config)
  python scripts/probe_deltaSPHMarrone34.py --nx 256 --scheme sun2017DeltaSPH --shifting default

  # sampling check: render boundary + mDBC ghost particles at the edge, the two
  # re-entrant corners and the fillet (no time stepping)
  python scripts/probe_deltaSPHMarrone34.py --nx 256 --initdump

  # combine every .npz under <out> into plots + REPORT.md
  python scripts/probe_deltaSPHMarrone34.py --report

Default --out: scripts/out_deltaSPHMarrone34/

The reusable dataset configs (delta+ + shifting) are
`examples/sweeps/marrone34_sharp_edge.yaml` / `marrone34_rounded_corner.yaml`
-- see `datagen/README.md`.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, 'out_deltaSPHMarrone34')

# --- Marrone 2011 Fig. 19, non-dimensional (H = 1) ---------------------------
H = 1.0
TANK_L = 8.0 * H              # tank height  (spec `L`)
TANK_W = 10.0 * H             # tank length  (case param `W`); fillet needs W = 10 H
COL_W = 3.0 * H              # reservoir width
COL_H = 2.4 * H              # reservoir height
G = 9.81
U_MAX = 1.95 * (G * H) ** 0.5
SQRT_H_G = (H / G) ** 0.5    # t = tStar * SQRT_H_G

#: Marrone 2011 Fig. 19's nine surface pressure probes as
#: (name, x, y, nx, ny): a point in centred-domain coords (x from tank centre,
#: y from tank centre; bed at y = -TANK_L/2) and the outward unit normal
#: pointing into the fluid.  Six on the obstacle -- P1-P3 down the 45deg edge
#: from the apex, P4-P6 along the roof apex->back -- and three on the concave
#: fillet arc, P7 (floor) -> P9 (downstream wall).  The paper gives no
#: coordinates, so the positions are read approximately from the sketch; the
#: normals are analytic.  Fig. 24 compares P1 / P5 / P7 against Colicchio's
#: Level-Set solver; Wagner's method gives P1 ~ 36.7 rho g H near the edge.
def _probePoints():
    import numpy as np
    wall = TANK_W / 2.0
    bed = -TANK_L / 2.0
    apex = np.array([-wall + 5.0 * H, bed + H])
    toe = np.array([-wall + 6.0 * H, bed])
    backTop = np.array([-wall + 7.0 * H, bed + H])
    discC = np.array([wall - H, bed + H])                   # fillet arc centre
    s2 = 2.0 ** -0.5

    out = []
    for i, f in enumerate((0.15, 0.5, 0.85)):              # P1 (apex) .. P3 (toe)
        p = apex + (toe - apex) * f
        out.append((f'P{i + 1}', p[0], p[1], -s2, -s2))    # 45deg face -> up-left
    for i, f in enumerate((0.17, 0.5, 0.83)):              # P4 .. P6  (P5 = mid roof)
        p = apex + (backTop - apex) * f
        out.append((f'P{i + 4}', p[0], p[1], 0.0, 1.0))    # obstacle top -> up
    for i, deg in enumerate((90.0, 45.0, 5.0)):            # P7 (floor) .. P9 (wall)
        a = np.deg2rad(deg)
        p = discC + H * np.array([np.cos(a), -np.sin(a)])
        out.append((f'P{i + 7}', p[0], p[1], -np.cos(a), np.sin(a)))  # -> disc centre
    return out


#: `[x, y, nx, ny]` rows for `params['surfacePressureProbes']`.
_SURF_PROBES = [[float(v) for v in row[1:]] for row in _probePoints()]
_SURF_NAMES = [row[0] for row in _probePoints()]


def _params(scheme, cornerOnly=False, shifting='off', Re=None):
    p = dict(
        W=TANK_W,
        fillRatio=COL_H / TANK_L,
        fluidWidth=COL_W / TANK_W,
        gravityMagnitude=G,
        disableGravity=False,
        obstacleActive=True,
        # `marroneRoundedCorner` drops the sharp-edged floor obstacle, leaving
        # only the concave fillet -- an isolated test of the smoothed corner
        # (the surge runs the full 10 H and impacts it directly).
        obstacleType='marroneRoundedCorner' if cornerOnly else 'marroneSharpEdge',
        referenceVelocity=U_MAX,
        # `shifting`: 'off' is Marrone Sec. 3's plain delta-SPH (the validation
        # config); 'default' leaves the scheme's own PST setting (ON for
        # `--scheme sun2017DeltaSPH` = delta+-SPH), 'on' forces it. The
        # delta+ + PST run is the dataset-generation config -- more robust on
        # the violent sharp-edge jet. `DELTASPH_VALIDATION_PLAN.md` Sec. 5.2.2.
        shifting=None if shifting == 'default' else (shifting == 'on' or shifting is True),
        # violent free-surface jet off the sharp edge -> keep surface detection
        pressureProbeHeights=[],
        # Marrone Fig. 19's P1-P9 on the 45deg edge / roof / fillet arc, queried
        # exactly on the wall point (inset 0 -- the first-order MLS fit handles
        # the one-sided stencil; validated on a hydrostatic column by
        # `scripts/probe_hydrostaticPressureProbe.py`). `dambreak.diagnostics`
        # emits `pSurf{k}` (Pa) + `pSurf{k}Nnbr` + `pSurf{k}WC` (MLS vs Shepard).
        surfacePressureProbes=[list(p) for p in _SURF_PROBES],
        surfacePressureProbeInset=0.0,
    )
    # Marrone §3.4.2: the same flow with real viscosity. Re = √(gH)·H / ν, so
    # ν = √(gH)·H / Re (H = 1, g = 9.81). Walls stay free-slip here -- the
    # no-slip half of §3.4.2 is separate. Re=None -> inviscid (§3.4.1).
    if Re:
        p['inviscid'] = False
        p['nu'] = (G * H) ** 0.5 * H / float(Re)
    return p


def _runOne(nx, c0Ratio, tStar, out, video, plotInterval, scheme,
            cornerOnly=False, shifting='off', Re=None, plotBackend=None):
    from warpSPHBootstrap import bootstrap
    bootstrap(precision='float32')
    import numpy as np
    from warpSPH.cases.dambreak import dambreakCase
    from warpSPH.runner import run

    os.makedirs(out, exist_ok=True)
    machTarget = 1.95 / c0Ratio                       # c0 = c0Ratio * sqrt(gH)
    tLimit = tStar * SQRT_H_G
    tag = (f'{scheme}_nx{nx}_c{c0Ratio:g}'
           + ('_cornerOnly' if cornerOnly else '')
           + ('' if shifting == 'off' else f'_pst-{shifting}')
           + ('' if not Re else f'_Re{Re:g}'))
    runRoot = os.path.join(out, tag + '_run')

    params = _params(scheme, cornerOnly=cornerOnly, shifting=shifting, Re=Re)
    params['machTarget'] = machTarget

    kw = dict(scheme=scheme, L=TANK_L, nx=nx, tLimit=tLimit,
              quiet=True, store=False, progress=True, params=params)
    if video:
        # Unset backend -> the runner picks vispy (headless EGL) for 2D, which
        # renders large particle counts far faster than matplotlib Agg (the
        # matplotlib frame render was dominating video-run wall time). Pass
        # `--plotBackend matplotlib` to force the old path.
        kw.update(plot=True, video=True, plotInterval=plotInterval, exportRoot=runRoot)
        if plotBackend:
            kw['plotBackend'] = plotBackend

    print(f'[{tag}] H/dx={H / (TANK_L / nx):.1f}  c0={c0Ratio:g}sqrt(gH)  '
          f'-> t={tLimit:.3f}s (t*={tStar:g}) ...', flush=True)
    r = run(dambreakCase, **kw)

    rows = [x for x in r.trajectory if x.get('step', -2) >= -1]
    keys = (['step', 't', 'tStar', 'kineticEnergy', 'maxVelocity',
             'minDensity', 'maxDensity', 'densityP05', 'densityP99',
             'nPenetrating', 'maxPenetrationDx', 'nObstaclePen', 'maxObstaclePenDx']
            + [f'pSurf{k}' for k in range(len(_SURF_PROBES))]
            + [f'pSurf{k}Nnbr' for k in range(len(_SURF_PROBES))])
    cols = {k: np.array([row.get(k, np.nan) for row in rows], dtype=float) for k in keys}
    # `diagnostics` now emits its own `tStar` (surface probes are active) but it
    # non-dimensionalises with `fillRatio * L` = the reservoir height, not the
    # obstacle height H = 1 this case scales by -- always recompute from t.
    cols['tStar'] = cols['t'] / SQRT_H_G

    dx = float(r.ctx.config.dx)
    c0 = float(getattr(r.ctx.schemeConfig.fluid, 'fixedSoundSpeed', 0.0) or 0.0)
    rho0 = float(getattr(r.ctx.schemeConfig.fluid, 'restDensity', 1000.0) or 1000.0)
    dp = r.ctx.schemeConfig.diffusionParams
    tReached = float(rows[-1].get('t', 0.0)) if rows else 0.0
    meta = dict(
        scheme=scheme, nx=nx, dx=dx, HdxRatio=H / dx, c0Ratio=float(c0Ratio),
        c0=c0, mach=(U_MAX / c0) if c0 else None, tLimit=tLimit, rho0=rho0,
        Re=(None if Re is None else float(Re)),
        inviscid=bool(getattr(dp, 'inviscid', True)),
        nu=float(getattr(dp, 'viscidNu', 0.0)),
        pRef=rho0 * G * H,                                 # rho0 g H (H = obstacle height = 1)
        tStarLimit=float(tStar), tReached=tReached, tStarReached=tReached / SQRT_H_G,
        H=H, G=G, TANK_L=TANK_L, TANK_W=TANK_W, COL_W=COL_W, COL_H=COL_H,
        surfProbeNames=list(_SURF_NAMES),
        shiftActive=bool(getattr(r.ctx.schemeConfig.shiftProperties, 'active', False)),
        sun2017Eq7Shift=bool(getattr(r.ctx.schemeConfig.shiftProperties, 'sun2017Eq7Shift', False)),
        shiftingArg=str(shifting), cornerOnly=bool(cornerOnly),
        diverged=bool(r.diverged), nSteps=int(r.nSteps),
        wallTime_s=float(r.wallTime or 0.0),
    )
    npz = os.path.join(out, tag + '.npz')
    np.savez(npz, meta=json.dumps(meta), **{f's_{k}': v for k, v in cols.items()})
    print(f'[{tag}] -> {npz}  diverged={r.diverged}  steps={r.nSteps}  '
          f'wall={r.wallTime:.1f}s  c0={c0:.2f}  M={(U_MAX / c0) if c0 else float("nan"):.3f}',
          flush=True)

    if video and r.videoPath and os.path.exists(r.videoPath):
        for f in ('output.mp4', 'out.gif'):
            s = os.path.join(os.path.dirname(r.videoPath), f)
            if os.path.exists(s):
                shutil.copy(s, os.path.join(out, f'{tag}_{f}'))
                print(f'[{tag}] video -> {out}/{tag}_{f}', flush=True)

    _score(meta, cols, verbose=True)
    return npz


def _score(meta, cols, verbose=False):
    """Stability / penetration / no-blow-up grading (the Sec. 5.2.2 goal).
    Convergence is graded across runs in `_report`, not here."""
    import numpy as np
    ts = cols['tStar']
    ke = cols['kineticEnergy']; vmax = cols['maxVelocity']
    rlo = cols['densityP05']
    rhi99 = cols.get('densityP99', cols['maxDensity'])
    rhiMax = cols['maxDensity']
    penW = cols['maxPenetrationDx']; penO = cols['maxObstaclePenDx']
    ok = np.isfinite(ts)
    ts, ke, vmax, rlo, rhi99, rhiMax, penW, penO = (
        a[ok] for a in (ts, ke, vmax, rlo, rhi99, rhiMax, penW, penO))

    # after t* > 1 the column has hit the floor and is running out. The
    # fragmenting sharp-edge jet makes the *pointwise* extremes (`maxDensity`,
    # `maxVelocity`) sharpen with resolution -- Marrone says this case is not
    # converged in those even at H/dx = 234 -- so the stability gate bands the
    # BULK: `densityP05` / `densityP99` (99-pct, one jet-tip particle excluded)
    # and treats the literal max / v_max as report-only.
    settled = ts > 1.0
    m = dict(
        tStarReached=float(meta['tStarReached']),
        diverged=bool(meta['diverged']),
        keMax=float(np.nanmax(ke)) if ke.size else float('nan'),
        keEnd=float(ke[-1]) if ke.size else float('nan'),
        vmaxMax=float(np.nanmax(vmax)) if vmax.size else float('nan'),
        rhoLoMin=float(np.nanmin(rlo[settled])) if settled.any() else float('nan'),
        rhoHi99Max=float(np.nanmax(rhi99[settled])) if settled.any() else float('nan'),
        rhoHiMax=float(np.nanmax(rhiMax[settled])) if settled.any() else float('nan'),
        maxWallPenDx=float(np.nanmax(penW)) if np.isfinite(penW).any() else float('nan'),
        maxObstaclePenDx=float(np.nanmax(penO)) if np.isfinite(penO).any() else float('nan'),
    )
    # KE trend over the second half (should not be running away)
    if ke.size > 20:
        half = ts > 0.5 * ts[-1]
        m['keTrend2ndHalf'] = float(np.polyfit(ts[half], ke[half], 1)[0])
    else:
        m['keTrend2ndHalf'] = float('nan')

    uScale = U_MAX
    keScaleGuess = m['keEnd'] if np.isfinite(m['keEnd']) and m['keEnd'] > 0 else 1.0
    checks = [
        ('runs to target t*', m['tStarReached'] >= 0.98 * meta['tStarLimit'] and not m['diverged'],
         f"reached t* {m['tStarReached']:.2f} / {meta['tStarLimit']:g}, diverged={m['diverged']}"),
        ('weakly compressible (bulk)', m['rhoLoMin'] > 0.90 and m['rhoHi99Max'] < 1.10,
         f"rho [P05, P99] in [{m['rhoLoMin']:.4f}, {m['rhoHi99Max']:.4f}]  (t* > 1); "
         f"pointwise max {m['rhoHiMax']:.3f}"),
        ('no velocity divergence', m['vmaxMax'] < 12.0 * uScale,
         f"max|v| {m['vmaxMax']:.2f} = {m['vmaxMax'] / uScale:.1f} U_max  "
         f"(jet-tip, report-only below 12 U_max)"),
        ('no tank-wall penetration', m['maxWallPenDx'] <= 3.0,
         f"{m['maxWallPenDx']:.2f} dx past the tank AABB  (want <= 3.0)"),
        ('no obstacle / fillet penetration', m['maxObstaclePenDx'] <= 3.0,
         f"{m['maxObstaclePenDx']:.2f} dx into the solid  (want <= 3.0)"),
        ('kinetic energy not running away',
         not (np.isfinite(m['keTrend2ndHalf']) and m['keTrend2ndHalf'] > 0
              and m['keTrend2ndHalf'] * m['tStarReached'] > 1.0 * keScaleGuess),
         f"dKE/dt* (2nd half) {m['keTrend2ndHalf']:.3e}, KE_end {m['keEnd']:.3e}"),
    ]
    if verbose:
        npass = sum(c[1] for c in checks)
        print(f"  {npass}/{len(checks)} checks")
        for n, okk, det in checks:
            print(('  PASS ' if okk else '  FAIL '), n, '|', det)
    return checks, m


def _initdump(nx, out, cornerOnly=False):
    """Build the system (no stepping) and render the sampled boundary + mDBC
    ghost particles zoomed on the four hard spots: the 45deg edge, the toe
    re-entrant corner, the roof/back-face re-entrant corner, and the fillet."""
    from warpSPHBootstrap import bootstrap
    bootstrap(precision='float32')
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from warpSPH.cases.dambreak import dambreakCase
    from warpSPH.runner import run
    from warpSPH.caseUtils.weaklyCompressible import buildObstacleSDF

    os.makedirs(out, exist_ok=True)
    params = _params('deltaSPH', cornerOnly=cornerOnly)
    params['machTarget'] = 1.95 / 28.3
    r = run(dambreakCase, scheme='deltaSPH', L=TANK_L, nx=nx, nSteps=1,
            tLimit=1e9, quiet=True, store=False, progress=False, params=params)

    st = r.state.state
    pos = st.positions.detach().cpu().numpy()
    kinds = st.kinds.detach().cpu().numpy()
    goff = (st.ghostOffsets.detach().cpu().numpy() if st.ghostOffsets is not None
            else np.zeros_like(pos))
    fluid = pos[kinds == 0]
    bnd = pos[kinds == 1]
    ghost = pos[kinds == 2]

    dx = float(r.ctx.config.dx)
    obType = 'marroneRoundedCorner' if cornerOnly else 'marroneSharpEdge'
    sdf = buildObstacleSDF(obType, 0, 0, 1, 1, 0, None, None, TANK_L, TANK_W)
    gx = np.linspace(-TANK_W / 2 - 0.4, TANK_W / 2 + 0.4, 900)
    gy = np.linspace(-TANK_L / 2 - 0.4, -TANK_L / 2 + 3.2, 320)
    import torch
    GX, GY = np.meshgrid(gx, gy)
    D = sdf(torch.tensor(np.stack([GX.ravel(), GY.ravel()], 1), dtype=torch.float32)
            ).detach().numpy().reshape(GX.shape)

    wall = TANK_W / 2.0
    bed = -TANK_L / 2.0
    spots = {
        'sharp 45deg edge': (-wall + 5.0 * H, bed + H, 1.6 * H),
        'toe re-entrant corner': (-wall + 6.0 * H, bed, 1.6 * H),
        'roof / back-face corner': (-wall + 7.0 * H, bed + H, 1.6 * H),
        'concave fillet': (wall - 0.5 * H, bed + 0.5 * H, 1.8 * H),
    }
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (title, (cx, cy, rad)) in zip(axes.ravel(), spots.items()):
        ax.contour(GX, GY, D, levels=[0], colors='k', linewidths=1.0)
        for arr, c, s, lbl in ((fluid, '#7fb3ff', 5, 'fluid'),
                               (bnd, '#888888', 8, 'boundary'),
                               (ghost, '#d62728', 8, 'ghost')):
            msk = ((arr[:, 0] > cx - rad) & (arr[:, 0] < cx + rad)
                   & (arr[:, 1] > cy - rad) & (arr[:, 1] < cy + rad))
            ax.scatter(arr[msk, 0], arr[msk, 1], s=s, c=c, label=lbl, zorder=3)
        # boundary -> ghost connectors
        bmask = (kinds == 1) & (pos[:, 0] > cx - rad) & (pos[:, 0] < cx + rad) \
            & (pos[:, 1] > cy - rad) & (pos[:, 1] < cy + rad)
        for i in np.where(bmask)[0]:
            gp = pos[i] - goff[i]
            ax.plot([pos[i, 0], gp[0]], [pos[i, 1], gp[1]], '-', c='#d62728',
                    lw=0.4, alpha=0.5, zorder=2)
        # P1-P9 surface pressure probes: the marker is the query point (on the
        # wall, inset 0); the short line shows the into-fluid normal direction.
        for pn, px, py, pnx, pny in _probePoints():
            if not (cx - rad < px < cx + rad and cy - rad < py < cy + rad):
                continue
            ax.plot([px, px + 1.5 * dx * pnx], [py, py + 1.5 * dx * pny],
                    '-', c='#2ca02c', lw=1.0, zorder=4)
            ax.scatter([px], [py], s=34, marker='s', facecolors='none',
                       edgecolors='#2ca02c', linewidths=1.2, zorder=5)
            ax.annotate(pn, (px, py), fontsize=7, color='#1a611a',
                        xytext=(3, 3), textcoords='offset points', zorder=6)
        ax.set_title(f'{title}   (dx = {dx:.4g}, H/dx = {H / dx:.0f})', fontsize=9)
        ax.set_xlim(cx - rad, cx + rad); ax.set_ylim(cy - rad, cy + rad)
        ax.set_aspect('equal'); ax.legend(fontsize=7, loc='upper right')
        ax.grid(alpha=0.2)
    fig.suptitle('Marrone 2011 Fig. 19 -- boundary + mDBC ghost sampling '
                 f'(nx = {nx})', fontsize=11)
    fig.tight_layout()
    p = os.path.join(out, f'initdump_nx{nx}{"_cornerOnly" if cornerOnly else ""}.png')
    fig.savefig(p, dpi=110)
    print('->', p)

    # quick numbers
    print(f'fluid {len(fluid)}  boundary {len(bnd)}  ghost {len(ghost)}')
    goffN = np.linalg.norm(goff[kinds == 1], axis=1) / dx
    print(f'ghost offset |r_b - r_g| / dx : min {goffN.min():.2f}  '
          f'median {np.median(goffN):.2f}  max {goffN.max():.2f}  '
          f'(>{4.0:g}: {(goffN > 4.0).sum()})')
    # any ghost placed inside the solid?
    gpos = torch.tensor(ghost, dtype=torch.float32)
    if len(ghost):
        gd = sdf(gpos).detach().numpy()
        print(f'ghosts inside obstacle/fillet solid: {(gd < 0).sum()} / {len(ghost)}'
              f'  (min sdf {gd.min():.3g})')


def _runLabel(meta):
    """Short legend label for a run; None to leave it out of the trace figure
    (the `--cornerOnly` runs have no obstacle probes)."""
    if meta.get('cornerOnly'):
        return None
    lab = ('delta+-SPH + PST' if (meta.get('scheme') == 'sun2017DeltaSPH'
                                  or meta.get('shiftActive')) else 'delta-SPH')
    if not meta.get('inviscid', True):
        lab += f", Re={meta.get('Re') or float('nan'):g}"
    lab += f"  (H/dx {meta['HdxRatio']:.0f})"
    return lab


#: P1-P9 -> (region label, is this one of Marrone Fig. 24's three).
_PROBE_REGION = {
    'P1': ('45 deg edge', True), 'P2': ('45 deg edge', False), 'P3': ('45 deg edge', False),
    'P4': ('obstacle roof', False), 'P5': ('obstacle roof', True), 'P6': ('obstacle roof', False),
    'P7': ('fillet / bottom-right wall', True), 'P8': ('fillet / bottom-right wall', False),
    'P9': ('fillet / bottom-right wall', False),
}


def _traceFigure(out):
    """A clean 3x3 of the nine surface pressure traces P/(rho g H) vs t*, one
    line per run, across every `*.npz` in `out`. Wagner's P1 ~ 36.7 rho g H is
    drawn on the edge panels; P1 / P5 / P7 (Marrone Fig. 24's three) are boxed.
    """
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    runs = []
    for npz in sorted(glob.glob(os.path.join(out, '*.npz'))):
        d = np.load(npz, allow_pickle=True)
        meta = json.loads(str(d['meta']))
        lab = _runLabel(meta)
        if lab is None:
            continue
        cols = {k[2:]: d[k] for k in d.files if k.startswith('s_')}
        runs.append((meta, cols, lab))
    if not runs:
        print(f'no obstacle runs in {out}'); return

    names = runs[0][0].get('surfProbeNames') or _SURF_NAMES
    cmap = plt.get_cmap('tab10')

    def _smooth(y, t):
        """centred moving average over a ~0.15 t* window -- the point probe on a
        violent thin sheet is spiky; the running mean is the readable signal
        (Marrone's transducer area-integrates, which does the same)."""
        y = np.nan_to_num(np.asarray(y, float))
        if y.size < 5:
            return y
        dt = np.median(np.diff(t)) if t.size > 1 else 1.0
        w = max(3, int(round(0.15 / max(dt, 1e-9))) | 1)
        ker = np.ones(w) / w
        return np.convolve(y, ker, mode='same')

    fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=True)
    for k, ax in enumerate(axes.ravel()):
        pn = names[k] if k < len(names) else f'P{k + 1}'
        region, isFig24 = _PROBE_REGION.get(pn, ('', False))
        anyWet = False
        for i, (meta, cols, lab) in enumerate(runs):
            raw = cols.get(f'pSurf{k}')
            if raw is None or not np.isfinite(raw).any() or np.nanmax(raw) <= 0:
                continue
            anyWet = True
            ts = cols['tStar']
            star = np.nan_to_num(raw / float(meta.get('pRef') or (1000.0 * G * H)))
            c = cmap(i % 10)
            ax.plot(ts, star, lw=0.5, color=c, alpha=0.30)
            ax.plot(ts, _smooth(star, ts), lw=1.6, color=c, label=lab)
        if region.startswith('45'):
            ax.axhline(36.7, color='k', ls=':', lw=1.0)
            ax.text(0.02, 0.94, 'Wagner 36.7', transform=ax.transAxes,
                    fontsize=7, va='top', color='#444')
        if not anyWet:
            ax.text(0.5, 0.5, 'never wetted', transform=ax.transAxes,
                    ha='center', va='center', fontsize=10, color='#999')
            ax.set_ylim(-1, 1)
        ax.set_title(f'{pn}  --  {region}' + ('   [Fig. 24]' if isFig24 else ''),
                     fontsize=9, fontweight=('bold' if isFig24 else 'normal'))
        ax.grid(alpha=0.25)
        if isFig24:
            for s in ax.spines.values():
                s.set_edgecolor('#1f77b4'); s.set_linewidth(1.6)
        if k // 3 == 2:
            ax.set_xlabel('t*  =  t sqrt(g/H)')
        if k % 3 == 0:
            ax.set_ylabel('P / (rho g H)')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='lower center', ncol=min(4, len(handles)),
                   fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle('Marrone 2011 Sec. 3.4 / Fig. 19 -- surface pressure probes P1-P9\n'
                 'probe positions read approximately from the figure sketch; '
                 'first-order MLS fluid-pressure interpolation 0.5 dx off the wall',
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    p = os.path.join(out, 'surface_pressure_traces.png')
    fig.savefig(p, dpi=130, bbox_inches='tight')
    print('->', p)
    return p


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
        runs.append((meta, cols, os.path.basename(npz)[:-4]))
    if not runs:
        print(f'no runs in {out}'); return

    # P1 / P5 / P7 are the three probes Marrone Fig. 24 compares against the
    # Level-Set solver (P1 near the sharp edge, P5 on the roof, P7 on the
    # fillet floor). `surfProbeNames` -> column index.
    def _pcol(meta, cols, pname):
        names = meta.get('surfProbeNames') or _SURF_NAMES
        if pname not in names:
            return None
        k = names.index(pname)
        raw = cols.get(f'pSurf{k}')
        if raw is None or not np.isfinite(raw).any():
            return None
        return raw / float(meta.get('pRef') or (1000.0 * G * H))

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    for meta, cols, name in runs:
        ts = cols['tStar']
        axes[0, 0].plot(ts, cols['kineticEnergy'], lw=1.1, label=name)
        axes[0, 1].plot(ts, cols['maxVelocity'] / U_MAX, lw=1.1, label=name)
        axes[1, 0].plot(ts, cols['densityP05'], lw=1.0)
        axes[1, 0].plot(ts, cols.get('densityP99', cols['maxDensity']), lw=1.2, label=name)
        axes[1, 0].plot(ts, cols['maxDensity'], lw=0.6, ls=':', alpha=0.6)
        axes[1, 1].plot(ts, cols['maxObstaclePenDx'], lw=1.1, label=f'{name} obstacle')
        axes[1, 1].plot(ts, cols['maxPenetrationDx'], lw=0.8, ls='--', label=f'{name} tank')
        p1 = _pcol(meta, cols, 'P1')
        if p1 is not None:
            axes[0, 2].plot(ts, p1, lw=1.1, label=name)
        for pname, ls in (('P5', '-'), ('P7', '--')):
            pv = _pcol(meta, cols, pname)
            if pv is not None:
                axes[1, 2].plot(ts, pv, lw=1.0, ls=ls, label=f'{name} {pname}')
    axes[0, 0].set_title('kinetic energy'); axes[0, 0].set_xlabel('t*')
    axes[0, 1].set_title('max |v| / U_max  (jet tip, sharpens with dx)'); axes[0, 1].set_xlabel('t*')
    axes[1, 0].set_title('density: P05 / P99 (solid) / pointwise max (dotted)'); axes[1, 0].set_xlabel('t*')
    axes[1, 0].axhspan(0.90, 1.10, color='green', alpha=0.08)
    axes[1, 1].set_title('penetration [dx]'); axes[1, 1].set_xlabel('t*')
    axes[0, 2].set_title('P1 pressure / rho g H  (edge; Wagner ~ 36.7)'); axes[0, 2].set_xlabel('t*')
    axes[0, 2].axhline(36.7, color='k', ls=':', lw=0.9, label='Wagner P1 = 36.7')
    axes[1, 2].set_title('P5 (roof, solid) / P7 (fillet, dashed)  pressure / rho g H'); axes[1, 2].set_xlabel('t*')
    for ax in axes.ravel():
        ax.legend(fontsize=6); ax.grid(alpha=0.25)
    fig.suptitle('Marrone 2011 Sec. 3.4 -- stability / penetration / convergence / surface pressure')
    fig.tight_layout()
    p = os.path.join(out, 'stability_and_convergence.png')
    fig.savefig(p, dpi=110)
    print('->', p)

    lines = ['# Marrone 2011 Sec. 3.4 -- dam break vs a sharp-edged obstacle', '']
    for meta, cols, name in runs:
        checks, m = _score(meta, cols)
        npass = sum(c[1] for c in checks)
        lines.append(f'## `{name}`  ({npass}/{len(checks)})')
        visc = ('inviscid' if meta.get('inviscid', True)
                else f"Re = {meta.get('Re') or float('nan'):g} (nu = {meta.get('nu', 0.0):.2e})")
        lines.append(f"- H/dx = {meta['HdxRatio']:.1f}, c0 = {meta['c0']:.1f} "
                     f"({meta['c0Ratio']:g} sqrt(gH)), M = {meta.get('mach') or float('nan'):.3f}, "
                     f"{visc}, "
                     f"t* reached = {meta['tStarReached']:.2f} / {meta['tStarLimit']:g}, "
                     f"diverged = {meta['diverged']}, steps = {meta['nSteps']}, "
                     f"wall = {meta['wallTime_s']:.0f}s")
        for n, ok, det in checks:
            lines.append(f"  - {'PASS' if ok else 'FAIL'}  **{n}** -- {det}")
        names = meta.get('surfProbeNames') or _SURF_NAMES
        peaks = []
        for k, pn in enumerate(names):
            raw = cols.get(f'pSurf{k}')
            if raw is None or not np.isfinite(raw).any():
                continue
            star = raw / float(meta.get('pRef') or (1000.0 * G * H))
            peaks.append(f"{pn} {np.nanmax(star):.1f}")
        if peaks:
            lines.append(f"  - surface pressure peaks P/(rho g H): {', '.join(peaks)}  "
                         f"(Wagner P1 ~ 36.7)")
        lines.append('')
    md = os.path.join(out, 'REPORT.md')
    open(md, 'w').write('\n'.join(lines) + '\n')
    print('->', md)
    print('\n'.join(lines))
    _traceFigure(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--nx', type=int, default=256, help='lattice resolution; H/dx = nx / 8')
    ap.add_argument('--c0Ratio', type=float, default=28.3, help='c0 = c0Ratio * sqrt(gH)')
    ap.add_argument('--tStar', type=float, default=4.0, help='t sqrt(g/H) to run to (Marrone Fig. 20: 7.32)')
    ap.add_argument('--scheme', default='deltaSPH',
                    help="'deltaSPH' (Marrone Sec. 3 validation) or 'sun2017DeltaSPH' "
                         "(delta+-SPH, the dataset-generation config)")
    ap.add_argument('--shifting', choices=('off', 'default', 'on'), default='off',
                    help="particle shifting / PST: 'off' = plain delta-SPH (validation); "
                         "'default' keeps the scheme's own setting (ON for sun2017DeltaSPH); "
                         "'on' forces it")
    ap.add_argument('--cornerOnly', action='store_true',
                    help='drop the sharp-edged obstacle; keep only the rounded tank corner '
                         '(isolated fillet impact test)')
    ap.add_argument('--Re', type=float, default=None,
                    help='Marrone Sec. 3.4.2 viscous case: Re = sqrt(gH) H / nu. '
                         'Unset -> inviscid (Sec. 3.4.1). Marrone runs 1000 and 10000.')
    ap.add_argument('--video', action='store_true')
    ap.add_argument('--plotInterval', type=int, default=25)
    ap.add_argument('--plotBackend', default=None,
                    help="video backend; unset = vispy (headless EGL, fast). "
                         "Pass 'matplotlib' only if vispy misbehaves.")
    ap.add_argument('--out', default=DEFAULT_OUT)
    ap.add_argument('--initdump', action='store_true', help='render boundary/ghost sampling, no stepping')
    ap.add_argument('--report', action='store_true')
    ap.add_argument('--traceFigure', action='store_true',
                    help='just the clean 3x3 of the P1-P9 surface pressure traces')
    args = ap.parse_args(argv)

    if args.traceFigure:
        _traceFigure(args.out); return
    if args.report:
        _report(args.out); return
    if args.initdump:
        _initdump(args.nx, args.out, args.cornerOnly); return
    _runOne(args.nx, args.c0Ratio, args.tStar, args.out, args.video,
            args.plotInterval, args.scheme, args.cornerOnly, args.shifting, args.Re,
            args.plotBackend)


if __name__ == '__main__':
    main()
