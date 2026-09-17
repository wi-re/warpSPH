#!/usr/bin/env python
"""Per-particle instrumentation for `BOUNDARY_DENSITY_PLAN.md` §8's step-14135
divergence: confirm or refute the "wall-wide dot=0 collapse" hypothesis with
direct per-particle numbers, not aggregate-trace inference.

Background (see the plan §8 in full -- not re-derived here): `english2025.py`'s
analytic hydrostatic term `dot(g_b, relPos)` was clamped at `>=0` (a 90 degree
cutoff: no push when gravity and the wall's outward normal are >90 degrees
apart) to suppress it at a ceiling. That clamp, tested on `sloshingTank`
(which **rolls**, so gravity's direction in the tank frame is time-varying),
caused a reproducible divergence at step 14135 / t=1.4135s -- 4+ seconds
before the cascade the fix targets, with no visible precursor. §8.3's
(unproven) diagnosis: because the tank rolls, EVERY wall's dot product sweeps
through zero over a roll cycle, not just the ceiling's -- and because gravity
is spatially uniform (`modules/gravity/directional.py` broadcasts one vector
to every particle), a flat wall's entire particle set crosses `dot=0`
*simultaneously* (same angle, only `relPos` magnitude/layer-depth differs),
which could collapse the hydrostatic term wall-wide in a single step.

This script runs the exact `run_sloshingTank.py --mdbcDensityScheme
english2025 --densityDiffusionTerm fourtakas2019` reproduction, truncated to
`--nSteps` via `warpSPH.runner.run`'s own `spec.nSteps` (no `--tLimit`
wall-clock cost), with two layers of instrumentation:

  1. A `_DEBUG_HOOK` on `warpSPH.modules.mdbc.english2025` (installed only
     when that module has been edited to call it -- see
     `BOUNDARY_DENSITY_PLAN.md` §8's re-applied clamp) that captures every
     mDBC-density call's *raw* `dot(g_b, relPos)`, the clamped
     `hydrostaticTerm` actually used, `P_g`/`P_b`/pre-fallback `rho_b`, and
     each ghost/boundary particle's position and UID.
  2. Monkeypatches (namespace-level, not import-time-bound -- see the module
     docstring below for why this specific pair of functions was chosen) on
     `warpSPH.schemes.deltaSPH.enforceDirichlet` (to recover the real
     simulation time `t`, which lives on `currentSystem`, not the
     `currentState` object `computeMdbcDensityEnglish2025` receives) and
     `.computePressureForceSurfaceAware` (downstream pressure-force
     magnitude, all particles, same window).

Both hooks are gated by a `case.postStep`-driven window (`--windowLo`/
`--windowHi`, default around step 14135) so capture overhead is zero outside
it -- this is a truncated, instrumented run, not a full 7s pass.

Supports three modes via `--clampDeg`:
  ''    baseline, byte-identical to the unmodified `english2025.py` (no env
        var branch taken at all)
  90    the original §8.1 hard cutoff (`clamp_min(dot, 0)`)
  135   generalised cutoff via `cosTheta = dot/(|g||r|)`, clamped at
        `cos(135deg)` -- gives a wall at its 90 degree rest orientation
        ~45 degrees of roll headroom before the term clamps at all (user
        follow-up request to BOUNDARY_DENSITY_PLAN.md §8)

Usage
-----
    # requires src/warpSPH/modules/mdbc/english2025.py to carry the §8
    # diagnostic clamp/hook block (see that file's DIAGNOSTIC ONLY comments)
    python scripts/probe_englishClampDivergence.py --clampDeg ''   --tag baseline
    python scripts/probe_englishClampDivergence.py --clampDeg 90   --tag clamp90
    python scripts/probe_englishClampDivergence.py --clampDeg 135  --tag clamp135

Writes `<out>/<tag>_mdbc.npz` (per-call mDBC records, stacked) and
`<out>/<tag>_pressure.npz` (per-call pressure-force records), plus prints an
inline analysis: first non-finite/extreme value (function, particle, step),
whether it's concentrated on one wall or spread across many, and whether it
lines up with a dot=0 sign flip.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

from warpSPHBootstrap import bootstrap

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, 'out_englishClampDivergence')

# Tank geometry (sloshingTank case defaults: `L=0.9`, `tankHeight=0.508`) --
# walls are STATIC in the simulation frame (only gravity rotates, per that
# case's own docstring), so any snapshot of a boundary particle's position is
# also its rest position; wall membership can be read off directly.
HALF_W = 0.45
TANK_H = 0.508


def classifyWall(pos: np.ndarray, tol: float) -> np.ndarray:
    """Bucket each boundary-particle position into a wall label.

    `tol` (a few dx) decides whether a particle counts as "near a corner"
    (within `tol` of two walls at once) rather than purely one wall.
    """
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
    p.add_argument('--clampDeg', type=str, default='',
                   help="'' (baseline, unclamped), '90', or '135' -- sets "
                        "WARPSPH_ENGLISH_CLAMP_DEG for english2025.py's "
                        "diagnostic clamp block.")
    p.add_argument('--tag', type=str, default=None,
                   help="output file prefix; default derived from --clampDeg.")
    p.add_argument('--nSteps', type=int, default=14145)
    p.add_argument('--windowLo', type=int, default=14100)
    p.add_argument('--windowHi', type=int, default=14145)
    p.add_argument('--nx', type=int, default=None)
    p.add_argument('--out', type=str, default=DEFAULT_OUT)
    p.add_argument('--extremeRho', type=float, default=0.5,
                   help='|rho_b - rho0| threshold to flag as "extreme" (not '
                        'just non-finite) in the inline scan.')
    p.add_argument('--extremeAccel', type=float, default=1.0e4,
                   help='|dvdt_pressure| threshold [same units as gravity, '
                        'i.e. multiples of ~9.81] to flag as extreme.')
    return p.parse_args(argv)


def main(argv=None):
    args = parseArgs(sys.argv[1:] if argv is None else argv)
    tag = args.tag or (f'clamp{args.clampDeg}' if args.clampDeg else 'baseline')
    os.makedirs(args.out, exist_ok=True)

    # Read at call-time inside english2025.py (`os.environ.get(...)` in the
    # function body, not at import time), so setting it here before `run()`
    # -- even though import already happened -- is sufficient.
    os.environ['WARPSPH_ENGLISH_CLAMP_DEG'] = args.clampDeg

    bootstrap(precision='float32')
    import torch
    from warpSPH.cases import importAll
    importAll()
    from warpSPH.runner import getCase, run, CaseSpec
    from warpSPH.enumTypes import DensityDiffusionScheme
    import warpSPH.modules.mdbc.english2025 as eng2025mod
    import warpSPH.schemes.deltaSPH as deltaSPHmod

    if not hasattr(eng2025mod, '_DEBUG_HOOK') and 'DIAGNOSTIC ONLY' not in open(eng2025mod.__file__).read():
        print('WARNING: src/warpSPH/modules/mdbc/english2025.py does not look '
              'like it carries the §8 diagnostic hook block -- per-particle '
              'mDBC records will be empty. See BOUNDARY_DENSITY_PLAN.md §8.')

    case = getCase('sloshingTank')
    spec = CaseSpec(caseName=case.name, scheme=case.scheme,
                    params=dict(case.params)).merged(**case.defaults)
    overrides = dict(
        caseName=f'diag-englishClamp-{tag}',
        nSteps=args.nSteps,
        plot=False, video=False, store=False, show=False, progress=True,
        # `run_sloshingTank.py`'s `buildSpec` re-asserts these for the wcsph
        # scheme specifically -- the case's own bare `params` default
        # (`shifting=False`) is the `divergenceFree`-only default; WITHOUT
        # this override WCSPH runs with the delta+-SPH free-surface shift
        # OFF, which this case's own docstring says is "the reason this case
        # survives the wall slam at all" -- confirmed the hard way (a probe
        # run without it diverged at step 384, long before anything
        # roll/clamp-related). Must match `run_sloshingTank.py` exactly.
        params=dict(shifting=True, correctdrhodt=False),
    )
    if args.nx is not None:
        overrides['nx'] = args.nx
    spec = spec.merged(**overrides)

    # Same override pattern as `run_sloshingTank.py` -- chain onto
    # `configureScheme` so it still runs the case's own setup first.
    _prevCfg = case.configureScheme
    def _cfgMdbc(ctx, _p=_prevCfg):
        _p(ctx)
        ctx.schemeConfig.mdbcDensityScheme = 'english2025'
        ctx.schemeConfig.diffusionParams.densityDiffusionTerm = DensityDiffusionScheme.fourtakas2019
    case.configureScheme = _cfgMdbc

    # -- Instrumentation ------------------------------------------------------
    captureEnabled = [False]
    lastT = [None]
    pendingMdbc = []          # records from computeMdbcDensityEnglish2025 not yet t-tagged
    mdbcRecords = []          # t-tagged, ready to save
    pressureRecords = []

    def _mdbcHook(data):
        if not captureEnabled[0]:
            return
        rec = {k: (v.detach().cpu().numpy() if torch.is_tensor(v) else v)
              for k, v in data.items()}
        pendingMdbc.append(rec)

    eng2025mod._DEBUG_HOOK = _mdbcHook

    _origEnforceDirichlet = deltaSPHmod.enforceDirichlet
    _origPressureForce = deltaSPHmod.computePressureForceSurfaceAware

    def _wrappedEnforceDirichlet(system, t, dt, *a, **kw):
        tv = float(t.item()) if torch.is_tensor(t) else float(t)
        lastT[0] = tv
        # `pendingMdbc` holds records from THIS SAME `deltaSPH_step` call
        # (enforceDirichlet is step 4, right after the mDBC density call at
        # step 3 -- deltaSPH.py's own numbering) -- tag and flush them now
        # that `t` is known. Single-threaded synchronous code: safe FIFO.
        while pendingMdbc:
            rec = pendingMdbc.pop(0)
            rec['t'] = tv
            mdbcRecords.append(rec)
        return _origEnforceDirichlet(system, t, dt, *a, **kw)

    def _wrappedPressureForce(*a, **kw):
        result = _origPressureForce(*a, **kw)
        if captureEnabled[0]:
            currentState = a[0]
            pressureRecords.append(dict(
                t=lastT[0],
                kinds=currentState.kinds.detach().cpu().numpy(),
                positions=currentState.positions.detach().cpu().numpy(),
                densities=currentState.densities.detach().cpu().numpy(),
                pressures=(currentState.pressures.detach().cpu().numpy()
                          if currentState.pressures is not None else None),
                UIDs=(currentState.UIDs.detach().cpu().numpy()
                     if currentState.UIDs is not None else None),
                dvdtPressureNorm=result.norm(dim=-1).detach().cpu().numpy(),
            ))
        return result

    deltaSPHmod.enforceDirichlet = _wrappedEnforceDirichlet
    deltaSPHmod.computePressureForceSurfaceAware = _wrappedPressureForce

    _prevPostStep = case.postStep
    def _windowGate(ctx, state, step):
        _prevPostStep(ctx, state, step)
        captureEnabled[0] = (args.windowLo <= step <= args.windowHi)
        if step == args.windowLo:
            print(f'  [probe] capture window opened at step {step}')
        if step == args.windowHi:
            print(f'  [probe] capture window closed at step {step}')
    case.postStep = _windowGate

    print(f'== probe_englishClampDivergence: tag={tag} clampDeg={args.clampDeg!r} '
         f'nSteps={args.nSteps} window=[{args.windowLo},{args.windowHi}] ==')
    t0 = time.perf_counter()
    result = run(case, spec)
    wall = time.perf_counter() - t0
    print(f'-- done in {wall:.1f}s ({result.nSteps} steps, diverged={result.diverged}) --')
    print(f'   mDBC records captured: {len(mdbcRecords)}  '
         f'pressure records captured: {len(pressureRecords)}')

    np.savez(os.path.join(args.out, f'{tag}_mdbc.npz'),
            records=np.array(mdbcRecords, dtype=object), allow_pickle=True)
    np.savez(os.path.join(args.out, f'{tag}_pressure.npz'),
            records=np.array(pressureRecords, dtype=object), allow_pickle=True)

    # -- Inline analysis --------------------------------------------------------
    if mdbcRecords:
        dx = float(getattr(result.ctx.config, 'dx', 0.0045) or 0.0045)
        print('\n-- per-wall hydrostatic-term summary (mDBC calls in window) --')
        firstExtreme = None
        for i, rec in enumerate(mdbcRecords):
            walls = classifyWall(rec['boundaryPos'], tol=3.0 * dx)
            rhoB = rec['rho_b_preFallback']
            hydroRaw = rec['hydroRaw']
            hydroTerm = rec['hydroTerm']
            nonFinite = ~np.isfinite(rhoB)
            extreme = nonFinite | (np.abs(rhoB - 1.0) > args.extremeRho)
            if extreme.any() and firstExtreme is None:
                idx = np.argmax(extreme)
                firstExtreme = dict(
                    callIndex=i, t=rec['t'], wall=walls[idx],
                    UID=int(rec['boundaryUID'][idx]) if rec.get('boundaryUID') is not None else None,
                    hydroRaw=float(hydroRaw[idx]), hydroTerm=float(hydroTerm[idx]),
                    P_g=float(rec['P_g'][idx]), P_b=float(rec['P_b'][idx]),
                    rho_b=float(rhoB[idx]), relPos=rec['relPos'][idx].tolist(),
                    g_b=rec['g_b'][idx].tolist(), nExtremeThisCall=int(extreme.sum()),
                    nParticlesThisCall=int(rhoB.shape[0]),
                    wallsAffected=sorted(set(walls[extreme].tolist())),
                )
            uniqueWalls = sorted(set(walls.tolist()))
            byWall = {w: dict(
                n=int((walls == w).sum()),
                hydroRawMin=float(hydroRaw[walls == w].min()), hydroRawMax=float(hydroRaw[walls == w].max()),
                hydroTermMin=float(hydroTerm[walls == w].min()), hydroTermMax=float(hydroTerm[walls == w].max()),
                rhoBMin=float(np.nanmin(rhoB[walls == w])) if np.isfinite(rhoB[walls == w]).any() else float('nan'),
                rhoBMax=float(np.nanmax(rhoB[walls == w])) if np.isfinite(rhoB[walls == w]).any() else float('nan'),
                nNonFinite=int((~np.isfinite(rhoB[walls == w])).sum()),
            ) for w in uniqueWalls}
            print(f'  call {i:3d}  t={rec["t"]:.6f}  ' +
                 '  '.join(f'{w}: n={v["n"]} hydroRaw[{v["hydroRawMin"]:+.3f},{v["hydroRawMax"]:+.3f}] '
                            f'rho_b[{v["rhoBMin"]:.4f},{v["rhoBMax"]:.4f}]'
                            + (f' NONFINITE={v["nNonFinite"]}' if v['nNonFinite'] else '')
                          for w, v in byWall.items()))

        print('\n-- first extreme/non-finite rho_b (mDBC) --')
        if firstExtreme is not None:
            for k, v in firstExtreme.items():
                print(f'   {k}: {v}')
        else:
            print(f'   none found (threshold |rho_b-1|>{args.extremeRho})')

    if pressureRecords:
        print('\n-- pressure-force magnitude summary (all kinds, window) --')
        firstExtremeP = None
        for i, rec in enumerate(pressureRecords):
            norm = rec['dvdtPressureNorm']
            nonFinite = ~np.isfinite(norm)
            extreme = nonFinite | (np.abs(norm) > args.extremeAccel)
            kinds = rec['kinds']
            if extreme.any() and firstExtremeP is None:
                idx = np.argmax(extreme)
                pos = rec['positions'][idx]
                wallLabel = classifyWall(pos[None, :], tol=3.0 * 0.0045)[0] if kinds[idx] != 0 else 'fluid'
                firstExtremeP = dict(
                    callIndex=i, t=rec['t'], kind=int(kinds[idx]), wall=wallLabel,
                    UID=int(rec['UIDs'][idx]) if rec.get('UIDs') is not None else None,
                    position=pos.tolist(), dvdtPressureNorm=float(norm[idx]),
                    density=float(rec['densities'][idx]),
                    nExtremeThisCall=int(extreme.sum()), nParticlesThisCall=int(norm.shape[0]),
                )
            print(f'  call {i:3d}  t={rec["t"]:.6f}  '
                 f'max|dvdt_p|={np.nanmax(np.abs(norm[np.isfinite(norm)])) if np.isfinite(norm).any() else float("nan"):.3e}  '
                 f'nNonFinite={int(nonFinite.sum())}')
        print('\n-- first extreme/non-finite pressure-force magnitude --')
        if firstExtremeP is not None:
            for k, v in firstExtremeP.items():
                print(f'   {k}: {v}')
        else:
            print(f'   none found (threshold |dvdt_p|>{args.extremeAccel:g})')

    print(f'\nwrote {args.out}/{tag}_mdbc.npz and {tag}_pressure.npz')
    return 1 if result.diverged else 0


if __name__ == '__main__':
    raise SystemExit(main())
