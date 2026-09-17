#!/usr/bin/env python
"""Per-particle instrumentation for `BOUNDARY_DENSITY_PLAN.md` §9.7/§9.8's
remaining sloshingTank+`english2025` excursion (t~4.0-6.5s, now isolated from
the sign bug and the Shepard precision hole, both fixed -- see commit
`e2b97bd`). User-identified from the rendered video: particles sticking to
the ceiling then falling/being ejected -- suspected to be the SAME mechanism
[[antuono-pressure-switch-bug]]/[[sph-symmetric-pressure-truncation-artifact]]
already diagnosed on an EARLIER checkpoint (UID 4104, pre-dating every mDBC
scheme change this investigation has made): a genuinely-isolated free-surface
film, near a wall/ceiling, whose nearest boundary/ghost pair's mDBC pressure
is an under-conditioned placeholder that the Antuono TIC switch (`sun2018`
Eq. 9) has no way to distinguish from a real measurement -- pinning the film
in place until it's kicked free. This script confirms or refutes that this
IS the same mechanism under the current (fixed) code and scheme, rather than
assuming it, by tracing the actual first-extreme particle the way §8.5/§9.4
did for the earlier bugs.

Runs the exact `run_sloshingTank.py --mdbcDensityScheme english2025
--densityDiffusionTerm fourtakas2019` reproduction, truncated to `--nSteps`
(no `--tLimit` wall-clock/video cost), with capture enabled only inside
`[--windowLo, --windowHi]`:

  1. `english2025.py`'s `_CAPTURE` hook (DIAGNOSTIC ONLY block, see that
     file) -- every mDBC call's per-ghost-row `Mb`/`alpha`/`hasAny`/`P_g`/
     `P_b`/`rho_b`, boundary UID and position.
  2. A monkeypatch on `computePressureForceSurfaceAware` -- per-call, ALL
     particles' `kinds`/`positions`/`densities`/`pressures`/`UIDs`/
     `surfaceIndicators` (the Antuono switch's `mask_i` input,
     `currentState.surfaceIndicators`) and the resulting pressure-force
     magnitude.
  3. A monkeypatch on `enforceDirichlet` to recover real simulation time `t`.

Usage
-----
    python scripts/probe_ceilingHover.py --nSteps 40150 --windowLo 39950 \
        --windowHi 40150 --tag event1
    python scripts/probe_ceilingHover.py --nSteps 64300 --windowLo 64000 \
        --windowHi 64300 --tag event2 --resumeFrom out_ceilingHover/event1_last.pt

Writes `<out>/<tag>_mdbc.npz`, `<out>/<tag>_pressure.npz`, plus an inline
first-extreme trace and (if the extreme is a fluid particle) a search for the
nearest boundary/ghost pair feeding it, printing whether that pair's mDBC
value was a fallback/marginal one (matching the diagnosed mechanism) or a
well-conditioned one (refuting it for this event).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

from warpSPHBootstrap import bootstrap

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, 'out_ceilingHover')

HALF_W = 0.45
TANK_H = 0.508


def classifyWall(pos: np.ndarray, tol: float) -> np.ndarray:
    x, y = pos[:, 0], pos[:, 1]
    dLeft = np.abs(x - (-HALF_W))
    dRight = np.abs(x - HALF_W)
    dFloor = np.abs(y - 0.0)
    dCeil = np.abs(y - TANK_H)
    d = np.stack([dLeft, dRight, dFloor, dCeil], axis=1)
    names = np.array(['left', 'right', 'floor', 'ceiling'])
    order = np.argsort(d, axis=1)
    primary = names[order[:, 0]]
    dSorted = np.take_along_axis(d, order, axis=1)
    nearCorner = dSorted[:, 1] < tol
    labels = np.where(nearCorner,
                      np.char.add(np.char.add(primary, '+'), names[order[:, 1]]),
                      primary)
    return labels


def parseArgs(argv):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--tag', type=str, required=True)
    p.add_argument('--nSteps', type=int, required=True)
    p.add_argument('--windowLo', type=int, required=True)
    p.add_argument('--windowHi', type=int, required=True)
    p.add_argument('--nx', type=int, default=None)
    p.add_argument('--out', type=str, default=DEFAULT_OUT)
    p.add_argument('--storeInterval', type=int, default=1000,
                   help='write a full HDF5 particle-state checkpoint every N '
                        'steps to <exportPath>/trajectory/state_<step>.h5 '
                        '(runner.py storeMode="states"), so a later window '
                        'or a crash can resume from the nearest checkpoint '
                        'instead of re-simulating from step 0. 0 disables.')
    p.add_argument('--extremeAccel', type=float, default=1.0e3,
                   help='|dvdt_pressure| threshold [m/s^2] to flag as extreme '
                        'in the inline scan (lower than the divergence probe '
                        'default -- we want the PRECURSOR, not just NaN).')
    return p.parse_args(argv)


def main(argv=None):
    args = parseArgs(sys.argv[1:] if argv is None else argv)
    os.makedirs(args.out, exist_ok=True)

    bootstrap(precision='float32')
    import torch
    from warpSPH.cases import importAll
    importAll()
    from warpSPH.runner import getCase, run, CaseSpec
    from warpSPH.enumTypes import DensityDiffusionScheme
    import warpSPH.modules.mdbc.english2025 as eng2025mod
    import warpSPH.schemes.deltaSPH as deltaSPHmod

    if not hasattr(eng2025mod, '_CAPTURE'):
        print('WARNING: english2025.py does not carry the §10 _CAPTURE hook '
              '-- mDBC per-row records will be empty.')

    case = getCase('sloshingTank')
    spec = CaseSpec(caseName=case.name, scheme=case.scheme,
                    params=dict(case.params)).merged(**case.defaults)
    overrides = dict(
        caseName=f'diag-ceilingHover-{args.tag}',
        nSteps=args.nSteps,
        plot=False, video=False, show=False, progress=True,
        store=args.storeInterval > 0,
        storeMode='states', storeInterval=max(1, args.storeInterval),
        # Must match run_sloshingTank.py's buildSpec exactly -- shifting=True
        # is what lets this case survive the wall slam at all.
        params=dict(shifting=True, correctdrhodt=False),
    )
    if args.nx is not None:
        overrides['nx'] = args.nx
    spec = spec.merged(**overrides)

    _prevCfg = case.configureScheme
    def _cfgMdbc(ctx, _p=_prevCfg):
        _p(ctx)
        ctx.schemeConfig.mdbcDensityScheme = 'english2025'
        ctx.schemeConfig.diffusionParams.densityDiffusionTerm = DensityDiffusionScheme.fourtakas2019
    case.configureScheme = _cfgMdbc

    captureEnabled = [False]
    lastT = [None]
    pressureRecords = []

    _origEnforceDirichlet = deltaSPHmod.enforceDirichlet
    _origPressureForce = deltaSPHmod.computePressureForceSurfaceAware

    def _wrappedEnforceDirichlet(system, t, dt, *a, **kw):
        tv = float(t.item()) if torch.is_tensor(t) else float(t)
        lastT[0] = tv
        return _origEnforceDirichlet(system, t, dt, *a, **kw)

    def _wrappedPressureForce(*a, **kw):
        result = _origPressureForce(*a, **kw)
        if captureEnabled[0]:
            currentState = a[0]
            surfInd = getattr(currentState, 'surfaceIndicators', None)
            pressureRecords.append(dict(
                t=lastT[0],
                kinds=currentState.kinds.detach().cpu().numpy(),
                positions=currentState.positions.detach().cpu().numpy(),
                densities=currentState.densities.detach().cpu().numpy(),
                velocities=currentState.velocities.detach().cpu().numpy(),
                pressures=(currentState.pressures.detach().cpu().numpy()
                          if currentState.pressures is not None else None),
                UIDs=(currentState.UIDs.detach().cpu().numpy()
                     if currentState.UIDs is not None else None),
                surfaceIndicators=(surfInd.detach().cpu().numpy()
                                   if torch.is_tensor(surfInd) else None),
                dvdtPressure=result.detach().cpu().numpy(),
                dvdtPressureNorm=result.norm(dim=-1).detach().cpu().numpy(),
            ))
        return result

    deltaSPHmod.enforceDirichlet = _wrappedEnforceDirichlet
    deltaSPHmod.computePressureForceSurfaceAware = _wrappedPressureForce

    _prevPostStep = case.postStep
    def _windowGate(ctx, state, step):
        _prevPostStep(ctx, state, step)
        wasEnabled = captureEnabled[0]
        captureEnabled[0] = (args.windowLo <= step <= args.windowHi)
        if not wasEnabled and captureEnabled[0]:
            print(f'  [probe] capture window opened at step {step}')
        if wasEnabled and not captureEnabled[0]:
            print(f'  [probe] capture window closed at step {step}')
    case.postStep = _windowGate

    # `_CAPTURE` is a plain list appended to on every mDBC call, whole run --
    # replace it with a proxy that only actually stores while the window is
    # open, so a long `--nSteps` doesn't grow it unbounded.
    class _GatedList(list):
        def append(self, item):
            if captureEnabled[0]:
                item['t'] = lastT[0]
                super().append(item)
    gated = _GatedList()
    eng2025mod._CAPTURE = gated
    mdbcRecords = gated

    print(f'== probe_ceilingHover: tag={args.tag} nSteps={args.nSteps} '
         f'window=[{args.windowLo},{args.windowHi}] '
         f'storeInterval={args.storeInterval if args.storeInterval > 0 else "off"} ==')
    t0 = time.perf_counter()
    result = run(case, spec)
    wall = time.perf_counter() - t0
    print(f'-- done in {wall:.1f}s ({result.nSteps} steps, diverged={result.diverged}) --')
    if result.exportPath:
        print(f'   checkpoints: {result.exportPath}/trajectory/state_*.h5 '
             f'(every {args.storeInterval} steps)')
    print(f'   mDBC records: {len(mdbcRecords)}  pressure records: {len(pressureRecords)}')

    np.savez(os.path.join(args.out, f'{args.tag}_mdbc.npz'),
            records=np.array(list(mdbcRecords), dtype=object), allow_pickle=True)
    np.savez(os.path.join(args.out, f'{args.tag}_pressure.npz'),
            records=np.array(pressureRecords, dtype=object), allow_pickle=True)

    # -- Inline analysis: find the first/worst extreme pressure-force particle,
    # then look up what fed it (own surface flag + own pressure sign, nearest
    # boundary/ghost UID's mDBC state at the SAME step). --------------------
    if pressureRecords:
        dx = float(getattr(result.ctx.config, 'dx', 0.0045) or 0.0045)
        print('\n-- pressure-force extremes (fluid particles only, window) --')
        worst = None
        for i, rec in enumerate(pressureRecords):
            norm = rec['dvdtPressureNorm']
            kinds = rec['kinds']
            fluidMask = kinds == 0
            nf = norm.copy()
            nf[~fluidMask] = -1.0
            nf = np.where(np.isfinite(nf), nf, 1e18)
            j = int(np.argmax(nf))
            val = float(norm[j]) if np.isfinite(norm[j]) else float('inf')
            if worst is None or (val > worst[0] and fluidMask[j]):
                worst = (val, i, j)
        if worst is not None:
            val, i, j = worst
            rec = pressureRecords[i]
            pos = rec['positions'][j]
            print(f'  worst fluid-particle |dvdt_p| in window: {val:.3e} at '
                 f't={rec["t"]:.6f} (call {i}), UID='
                 f'{int(rec["UIDs"][j]) if rec["UIDs"] is not None else "?"} '
                 f'pos={pos.tolist()} '
                 f'surfaceIndicator={rec["surfaceIndicators"][j] if rec["surfaceIndicators"] is not None else "?"} '
                 f'own pressure={rec["pressures"][j] if rec["pressures"] is not None else "?"} '
                 f'own density={rec["densities"][j]:.4f}')

            # Nearest boundary/ghost particle at the SAME call, and (if the
            # mDBC capture window overlaps) that ghost's captured Mb/alpha/rho_b.
            bg = rec['kinds'] != 0
            if bg.any():
                d2 = ((rec['positions'][bg] - pos[None, :]) ** 2).sum(axis=1)
                nearestIdx = np.where(bg)[0][np.argmin(d2)]
                nearestDist = float(np.sqrt(d2.min()))
                nearestUID = int(rec['UIDs'][nearestIdx]) if rec['UIDs'] is not None else None
                nearestKind = int(rec['kinds'][nearestIdx])
                print(f'  nearest boundary/ghost (kind={nearestKind}): UID='
                     f'{nearestUID} dist={nearestDist:.5f} ({nearestDist/dx:.2f} dx) '
                     f'pressure={rec["pressures"][nearestIdx] if rec["pressures"] is not None else "?"} '
                     f'density={rec["densities"][nearestIdx]:.4f}')

                if mdbcRecords:
                    # Find the mDBC call closest in time to this pressure call.
                    mt = np.array([r['t'] for r in mdbcRecords])
                    mi = int(np.argmin(np.abs(mt - rec['t'])))
                    mrec = mdbcRecords[mi]
                    bUID = mrec.get('boundaryUID')
                    if bUID is not None and nearestUID is not None and nearestKind == 1:
                        hit = np.where(bUID == nearestUID)[0]
                        if hit.size:
                            k = int(hit[0])
                            print(f'  -> mDBC row for that boundary UID at t={mrec["t"]:.6f}: '
                                 f'Mb={mrec["Mb"][k]:.3e} alpha={mrec["alpha"][k]:.6f} '
                                 f'hasAny={bool(mrec["hasAny"][k])} nNbFluid={int(mrec["nNbFluid"][k])} '
                                 f'P_g={mrec["P_g"][k]:.3e} P_b={mrec["P_b"][k]:.3e} '
                                 f'rho_b={mrec["rho_b"][k]:.6f}')
                        else:
                            print(f'  -> no mDBC row found for boundary UID {nearestUID} '
                                 f'at t={mrec["t"]:.6f} (ghost may lack fluid neighbours '
                                 f'entirely, or UID mismatch -- check manually)')
        else:
            print(f'  no fluid-particle extremes found above threshold in window')

        # Free-surface occupancy context for the worst particle, if found.
        if worst is not None:
            val, i, j = worst
            rec = pressureRecords[i]
            pos = rec['positions'][j]
            matter = rec['kinds'] <= 1  # fluid(0) + boundary(1), not ghost(2)
            d = np.linalg.norm(rec['positions'][matter] - pos[None, :], axis=1)
            for mult in (1, 2, 4, 6):
                within = int((d < mult * 2.0 * dx).sum()) - 1  # exclude self
                print(f'  matter (fluid+boundary) within {mult}x support: {max(within,0)}')

    print(f'\nwrote {args.out}/{args.tag}_mdbc.npz and {args.tag}_pressure.npz')
    return 1 if result.diverged else 0


if __name__ == '__main__':
    raise SystemExit(main())
