"""The rotating square patch as a **PST discriminator** — plan item 1
(`DELTASPH_VALIDATION_PLAN.md` §5.1.1 pt 3 / Open-next 1).

Part 2.1 nominates this case as the one where "PST is validated separately":
plain δ-SPH is supposed to develop the tensile (pairing) instability at the
rotating corners, and the δ⁺ shift is supposed to cure it. That A/B has never
actually been run here — every recorded square-patch result had the shift on
(`ShiftProperties.active` defaults `True`), and since §5.3.2 the shift magnitude
changed 8×, so even the "on" leg needs re-checking.

Three legs, same as `probe_deltaPlusShiftSweep.py`:
  off     shiftProperties.active = False              -- plain δ-SPH (the control)
  eighth  active, sun2017Eq7Shift = False             -- the historical ⅛·Eq.(7)
  eq7     active, sun2017Eq7Shift = True              -- Sun 2017 Eq. (7)

The discriminator is the *geometric* signature (`particleDistributionMetrics`,
which a density-only check is blind to), reported at a pre-fragmentation time
`--twProbe` (default tω = 2) and at the last frame:
  paired    fraction of fluid particles with a neighbour < ½ the median spacing
            -- the clumping / tensile-instability signature
  nnP01     1st-pct nearest-neighbour distance / median  -- collapses to 0 under pairing
  void      fraction with nearest neighbour > 1.5× median  -- de-densification
  nCV       neighbour-count coefficient of variation  -- layering / voids
  rhoMed    bulk (median) density
  rhoMax    peak density
  sf        surfaceFraction -- if ~1 the patch has fragmented and the bulk
            metrics above no longer describe a bulk (arm fragmentation is
            under-resolution, `probe_squarePatchFragmentation.py`, not PST)

Usage:
  python scripts/probe_squarePatchPSTControl.py                       # nx 192, tω to 4
  python scripts/probe_squarePatchPSTControl.py --nx 192 288 --tLimit 1.0
  python scripts/probe_squarePatchPSTControl.py --nx 288 --figure     # + snapshot grid
"""
from __future__ import annotations

import argparse
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, 'out_squarePatchPSTControl')

LEGS = ('off', 'eighth', 'eq7')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--nx', type=int, nargs='+', default=[192])
    ap.add_argument('--tLimit', type=float, default=1.0, help='ω = 4, so tLimit 1.0 = tω 4')
    ap.add_argument('--twProbe', type=float, default=2.0, help='tω at which to also read the metrics')
    ap.add_argument('--legs', nargs='+', default=list(LEGS), choices=LEGS)
    ap.add_argument('--kernel', default='Wendland2')
    ap.add_argument('--omega', type=float, default=4.0)
    ap.add_argument('--figure', action='store_true', help='also write a particle snapshot per leg at tw=twProbe and t1')
    ap.add_argument('--out', default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    from warpSPHBootstrap import bootstrap
    bootstrap(precision='float32')
    import numpy as np
    from warpSPH.cases.rotatingSquarePatch import rotatingSquarePatchCase as case
    from warpSPH.runner import run

    os.makedirs(args.out, exist_ok=True)

    def _cfg(leg):
        base = case.configureScheme

        def wrapped(ctx):
            base(ctx)
            props = getattr(ctx.schemeConfig, 'shiftProperties', None)
            if props is None:
                return
            props.active = leg != 'off'
            props.sun2017Eq7Shift = leg == 'eq7'
        return wrapped

    cols = ['paired', 'nnP01', 'void', 'nCV', 'rhoMed', 'rhoMax', 'sf']
    hdr = (f"{'nx':>5} {'leg':>7} {'tw':>5} | "
           + ' '.join(f'{c:>8}' for c in cols) + '   note')
    print(hdr)
    print('-' * len(hdr))

    figData = {}
    for nx in args.nx:
        for leg in args.legs:
            _orig = case.configureScheme
            _origPost = case.postStep
            case.configureScheme = _cfg(leg)
            # Capture particle snapshots at the trace times *during* the run
            # (store=False leaves no intermediate states otherwise) -- the
            # `probe_squarePatchValidationFigure.py` pattern.
            frames = {}
            snapTw = sorted({args.twProbe, args.tLimit * args.omega})
            if args.figure:
                pending = list(snapTw)

                def postStep(ctx, state, i, _pending=pending, _frames=frames):
                    tw = float(state.t) * args.omega
                    while _pending and tw >= _pending[0] - 1e-6:
                        _frames[_pending.pop(0)] = _grabSnapshot(state)
                case.postStep = postStep
            try:
                r = run(case, params={'omega': args.omega}, nx=nx, tLimit=args.tLimit,
                        nSteps=None, kernel=args.kernel, store=False, plot=False,
                        quiet=True, progress=False)
            finally:
                case.configureScheme = _orig
                case.postStep = _origPost
            if args.figure:
                for t in pending:                          # any target the loop stopped short of
                    frames[t] = _grabSnapshot(r.state)

            rows = [row for row in r.trajectory
                    if 'pairedFraction' in row and row.get('step', -1) >= 0]
            if not rows:
                print(f"{nx:>5} {leg:>7}   no rows"); continue
            tw = np.array([row['t'] * args.omega for row in rows])

            def at(twTarget):
                k = int(np.argmin(np.abs(tw - twTarget)))
                row = rows[k]
                return dict(
                    paired=row.get('pairedFraction', np.nan),
                    nnP01=row.get('nnDistP01', np.nan),
                    void=row.get('voidFraction', np.nan),
                    nCV=row.get('neighbourCountCV', np.nan),
                    rhoMed=row.get('densityMedian', np.nan),
                    rhoMax=row.get('maxDensity', np.nan),
                    sf=row.get('surfaceFraction', np.nan),
                ), tw[k]

            # A trace, not two points: the pairing signature of a real
            # instability GROWS with time, while a startup transient off the
            # freshly-rotated lattice heals (the `tgv-wc` eq7 caution in
            # `probe_deltaPlusShiftSweep.py`'s report).
            traceTw = [t for t in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0) if t <= tw[-1] + 1e-6]
            for twTarget in traceTw:
                vals, twAt = at(twTarget)
                note = ''
                if r.diverged:
                    note = 'DIVERGED'
                elif vals['sf'] > 0.85:
                    note = 'fragmented'
                print(f"{nx:>5} {leg:>7} {twAt:>5.2f} | "
                      + ' '.join(f"{vals[c]:>8.4f}" for c in cols) + f'   {note}')
            print()

            if args.figure:
                figData[(nx, leg)] = [frames[k] for k in sorted(frames)]

    if args.figure and figData:
        _writeFigure(figData, args.nx, args.legs, args.omega, args.out)


def _grabSnapshot(state):
    """`(tω-less t, fluid xy, fluid pressure)` from a live system state."""
    import numpy as np
    p = state.state
    kinds = p.kinds.detach().cpu().numpy()
    fluid = kinds == 0
    pos = p.positions.detach().cpu().numpy()[fluid]
    pres = (p.pressures.detach().cpu().numpy()[fluid]
            if getattr(p, 'pressures', None) is not None else np.zeros(fluid.sum()))
    return float(state.t), pos, pres


def _writeFigure(figData, nxs, legs, omega, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    for nx in nxs:
        cols = [figData[(nx, leg)] for leg in legs if (nx, leg) in figData]
        if not cols:
            continue
        ncol = len(cols[0])
        fig, axes = plt.subplots(len(cols), ncol, figsize=(4.2 * ncol, 4.0 * len(cols)),
                                 squeeze=False)
        for i, leg in enumerate([l for l in legs if (nx, l) in figData]):
            for j, (t, pos, pres) in enumerate(figData[(nx, leg)]):
                tw = t * omega
                ax = axes[i][j]
                clim = np.percentile(np.abs(pres), 95) if len(pres) else 1.0
                ax.scatter(pos[:, 0], pos[:, 1], s=2, c=pres, cmap='RdBu_r',
                           vmin=-clim, vmax=clim)
                ax.set_aspect('equal'); ax.set_title(f'{leg}  tω={tw:.1f}', fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f'rotating square patch — PST discriminator (nx={nx})')
        fig.tight_layout()
        p = os.path.join(out, f'snapshot_nx{nx}.png')
        fig.savefig(p, dpi=130)
        print('->', p)


if __name__ == '__main__':
    main()
