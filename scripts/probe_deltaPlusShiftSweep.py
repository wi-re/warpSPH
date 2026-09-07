"""Sweep (`DELTASPH_VALIDATION_PLAN.md` Sec. 5.3.2): what does the Sun 2017
Eq. (7) shift scaling do to every case that actually uses it?

`ShiftProperties.sun2017Eq7Shift` multiplies the delta+ PST by 8 (see
`modules/shifting/delta.py`).  It is opt-in, and `Sun2017DeltaSPHConfig` turns
it on, so `--scheme sun2017DeltaSPH` already runs it.  The open question is
whether it can become the shared default, and
`scripts/probe_deltaPlusShiftBlastRadius.py` says that decision touches **11**
of the 35 registered cases -- every weakly compressible case with the shift
active, `dambreak` and `squarePatch` included.

Three legs per case, because the interesting comparison is not two-way:

* `off`    -- `shiftProperties.active = False`, i.e. plain delta-SPH. This is
              the *reference* leg, and it is the one Marrone 2011 Sec. 3
              specifies for the dam-break cases (`DELTASPH_VALIDATION_PLAN.md`
              Part 2.1's own acceptance gate: "a correct delta-SPH dam break
              must be stable with `shiftProperties.active = False`").
* `eighth` -- the shift active at the historical 1/8-of-Eq.-(7) scaling. This
              is what every one of those 11 cases has actually been running,
              including the Marrone Sec. 3.1 validation, whose spec table in
              the plan claimed `active = False` was "already the deltaSPH
              default" -- it is not; the field defaults to `True`.
* `eq7`    -- the shift active at Eq. (7)'s own scaling.

Each leg runs the same bounded number of steps and reports the health columns
that distinguish a stable weakly compressible run from a degrading one: the
density band, the peak speed, and the *geometric* metrics
(`particleDistributionMetrics`) that a density-only check is blind to --
`pairedFraction` (clumping / tensile instability, which is what the PST exists
to prevent) and `voidFraction` (de-densification, which an over-strong PST can
cause at a free surface).

This is a smoke sweep by design: enough steps to separate the legs, not a
production run.  Cases that look marginal here earn a full run.

Usage:
  python scripts/probe_deltaPlusShiftSweep.py                     # every affected case
  python scripts/probe_deltaPlusShiftSweep.py --cases droplet squarePatch
  python scripts/probe_deltaPlusShiftSweep.py --nSteps 600
  python scripts/probe_deltaPlusShiftSweep.py --report            # rebuild the table

  # one (case, leg) in this process -- what the driver spawns
  python scripts/probe_deltaPlusShiftSweep.py --one droplet eq7 --nSteps 300

Default --out: scripts/out_deltaPlusShiftSweep/
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DEFAULT_OUT = os.path.join(HERE, 'out_deltaPlusShiftSweep')

#: The blast radius, from `scripts/probe_deltaPlusShiftBlastRadius.py`.  Listed
#: rather than recomputed so a run of this script is reproducible from the file
#: alone; rerun that probe if the case registry changes.
AFFECTED = ['dambreak', 'drivenSquare', 'droplet', 'impact', 'kolmogorov',
            'ldc', 'movingObstacle', 'openFlow', 'randomFlow', 'squarePatch',
            'tgv-wc']

LEGS = ('off', 'eighth', 'eq7')

#: Columns worth carrying out of a run, and whether a *larger* value is worse.
METRICS = ['minDensity', 'maxDensity', 'densityP05', 'densityMedian',
           'maxVelocity', 'kineticEnergy', 'pairedFraction', 'voidFraction',
           'nnDistP01', 'neighbourCountCV']


def _runOne(caseName: str, leg: str, nSteps: int, out: str, scheme: str):
    from warpSPHBootstrap import bootstrap
    bootstrap(precision='float32')
    import numpy as np
    from warpSPH.cases import importAll
    importAll()
    from warpSPH.runner import getCase, run

    os.makedirs(out, exist_ok=True)
    case = getCase(caseName)

    # The three legs differ only in `shiftProperties`, and a case sets that in
    # its own `configureScheme`, so the override has to run *after* it -- wrap
    # the case's hook rather than trying to pass the settings in.  Restored in
    # the `finally` even though this process is about to exit, so that the
    # single-process `--one` path is not quietly order-dependent.
    original = case.configureScheme

    def configureScheme(ctx):
        if original is not None:
            original(ctx)
        props = getattr(ctx.schemeConfig, 'shiftProperties', None)
        if props is None:
            return
        props.active = leg != 'off'
        props.sun2017Eq7Shift = leg == 'eq7'

    case.configureScheme = configureScheme
    try:
        r = run(case, scheme=scheme, nSteps=nSteps, quiet=True, store=False,
                progress=False)
    finally:
        case.configureScheme = original

    rows = [x for x in r.trajectory if x.get('step', -2) >= 0]
    summary = {}
    for k in METRICS:
        vals = np.array([row.get(k, np.nan) for row in rows], dtype=float)
        if not np.isfinite(vals).any():
            continue
        summary[k] = dict(first=float(vals[np.isfinite(vals)][0]),
                          last=float(vals[np.isfinite(vals)][-1]),
                          min=float(np.nanmin(vals)), max=float(np.nanmax(vals)))
    result = dict(case=caseName, leg=leg, scheme=scheme, nSteps=int(r.nSteps),
                  requestedSteps=nSteps, diverged=bool(r.diverged),
                  wallTime_s=float(r.wallTime or 0.0),
                  tReached=float(rows[-1].get('t', 0.0)) if rows else 0.0,
                  metrics=summary)
    path = os.path.join(out, f'{caseName}__{leg}.json')
    with open(path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'[{caseName}/{leg}] steps={r.nSteps} diverged={r.diverged} '
          f'wall={r.wallTime:.1f}s -> {path}', flush=True)


def _drive(cases, nSteps, out, scheme, timeout):
    """One subprocess per (case, leg).

    Out-of-process for the same reason `scripts/run_sweep.py` is: a case that
    crashes, leaks warp state, or exhausts device memory must not take the rest
    of the sweep with it, and the legs have to be independent of each other.
    """
    os.makedirs(out, exist_ok=True)
    for caseName in cases:
        for leg in LEGS:
            cmd = [sys.executable, os.path.abspath(__file__), '--one', caseName,
                   leg, '--nSteps', str(nSteps), '--out', out, '--scheme', scheme]
            print(f'--- {caseName} / {leg}', flush=True)
            try:
                proc = subprocess.run(cmd, cwd=REPO, timeout=timeout,
                                      capture_output=True, text=True)
            except subprocess.TimeoutExpired:
                _writeFailure(out, caseName, leg, scheme, nSteps, 'timeout',
                              f'exceeded {timeout}s')
                print(f'[{caseName}/{leg}] TIMEOUT after {timeout}s', flush=True)
                continue
            tail = (proc.stdout + proc.stderr).strip().splitlines()
            if proc.returncode != 0:
                _writeFailure(out, caseName, leg, scheme, nSteps, 'crashed',
                              tail[-1] if tail else f'exit {proc.returncode}')
                print(f'[{caseName}/{leg}] FAILED (exit {proc.returncode}): '
                      f'{tail[-1] if tail else ""}', flush=True)
            else:
                for line in tail:
                    if line.startswith('['):
                        print(line, flush=True)


def _writeFailure(out, caseName, leg, scheme, nSteps, status, detail):
    with open(os.path.join(out, f'{caseName}__{leg}.json'), 'w') as f:
        json.dump(dict(case=caseName, leg=leg, scheme=scheme,
                       requestedSteps=nSteps, status=status, detail=detail,
                       diverged=True, metrics={}), f, indent=2)


def _report(out: str):
    runs = {}
    for path in sorted(glob.glob(os.path.join(out, '*__*.json'))):
        d = json.load(open(path))
        runs.setdefault(d['case'], {})[d['leg']] = d
    if not runs:
        print(f'no runs under {out}', file=sys.stderr)
        sys.exit(1)

    def cell(d, key, field='max'):
        if d is None or d.get('status'):
            return d.get('status', '-') if d else '-'
        m = d['metrics'].get(key)
        return f"{m[field]:.4f}" if m else '-'

    lines = ['# δ⁺ shift scaling — smoke sweep over the blast radius', '',
             'Legs: `off` = plain δ-SPH (no PST) · `eighth` = the historical '
             '⅛-of-Eq.(7) shift, which is what every one of these cases has '
             'actually been running · `eq7` = Sun et al. 2017 Eq. (7).', '',
             'See `scripts/probe_deltaPlusShiftSweep.py` for what the columns '
             'mean and why three legs.', '']

    lines += ['Geometric columns are reported **max (last)** and '
              '`nnDistP01` **min (last)** deliberately. The worst value alone '
              'cannot tell a startup transient from a degradation, and on this '
              'very sweep that distinction was the whole answer for `tgv-wc`: '
              'its `eq7` leg peaks at `paired = 0.15` on the *first* step, off '
              'the freshly-shuffled lattice, and then heals monotonically — the '
              'full-length validation runs in `scripts/out_deltaPlusTGV/` end at '
              '`paired = 0.0000` and `nnDistP01 = 0.877`, a **better** '
              'distribution than the ⅛ leg reaches (0.814). A max-only table '
              'reported that as the one case the fix breaks. It is the opposite.',
              '']
    header = ('| case | leg | steps | diverged | ρ min | ρ max | max ‖v‖ | '
              'paired max (last) | void max (last) | nnDistP01 min (last) |')
    lines += [header, '|' + '---|' * 10]
    for caseName in sorted(runs):
        for leg in LEGS:
            d = runs[caseName].get(leg)
            if d is None:
                continue
            if d.get('status'):
                lines.append(f"| `{caseName}` | {leg} | – | **{d['status']}** | "
                             + ' | '.join(['–'] * 6) + ' |')
                continue
            lines.append(
                f"| `{caseName}` | {leg} | {d['nSteps']} | "
                f"{'**yes**' if d['diverged'] else 'no'} | "
                f"{cell(d, 'minDensity', 'min')} | {cell(d, 'maxDensity', 'max')} | "
                f"{cell(d, 'maxVelocity', 'max')} | "
                f"{cell(d, 'pairedFraction', 'max')} ({cell(d, 'pairedFraction', 'last')}) | "
                f"{cell(d, 'voidFraction', 'max')} ({cell(d, 'voidFraction', 'last')}) | "
                f"{cell(d, 'nnDistP01', 'min')} ({cell(d, 'nnDistP01', 'last')}) |")
    lines.append('')

    # The headline the decision actually needs: did any case get *worse* going
    # from what it runs today (`eighth`) to `eq7`?
    lines += ['## eighth → eq7, per case', '',
              '`pairedFraction` is the tensile-instability signature the PST '
              'exists to suppress, so a *fall* there is the PST working; '
              '`voidFraction` rising is the failure mode an over-strong shift '
              'would produce. Compared at the **last** step, not the worst — '
              'see the note above.', '',
              '| case | diverged (⅛ → Eq.7) | paired last | void last | '
              'nnDistP01 last | ρ band |',
              '|---|---|---|---|---|---|']
    for caseName in sorted(runs):
        a, b = runs[caseName].get('eighth'), runs[caseName].get('eq7')
        if not a or not b or a.get('status') or b.get('status'):
            lines.append(f"| `{caseName}` | "
                         f"{(a or {}).get('status', '?')} → {(b or {}).get('status', '?')}"
                         f" | – | – | – | – |")
            continue
        lines.append(
            f"| `{caseName}` | {a['diverged']} → {b['diverged']} | "
            f"{cell(a, 'pairedFraction', 'last')} → {cell(b, 'pairedFraction', 'last')} | "
            f"{cell(a, 'voidFraction', 'last')} → {cell(b, 'voidFraction', 'last')} | "
            f"{cell(a, 'nnDistP01', 'last')} → {cell(b, 'nnDistP01', 'last')} | "
            f"[{cell(a, 'minDensity', 'min')}, {cell(a, 'maxDensity')}] → "
            f"[{cell(b, 'minDensity', 'min')}, {cell(b, 'maxDensity')}] |")
    lines.append('')

    md = os.path.join(out, 'REPORT.md')
    with open(md, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'-> {md}')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--one', nargs=2, metavar=('CASE', 'LEG'), default=None,
                    help='run a single (case, leg) in this process')
    ap.add_argument('--cases', nargs='*', default=None,
                    help=f'cases to sweep (default: the blast radius, {len(AFFECTED)})')
    ap.add_argument('--nSteps', type=int, default=300)
    ap.add_argument('--scheme', default='deltaSPH',
                    help="scheme to run every case under; 'deltaSPH' keeps the legs "
                         'differing only in the shift settings this script sets')
    ap.add_argument('--timeout', type=int, default=1800)
    ap.add_argument('--out', default=DEFAULT_OUT)
    ap.add_argument('--report', action='store_true')
    args = ap.parse_args(argv)

    if args.report:
        _report(args.out)
        return
    if args.one:
        caseName, leg = args.one
        if leg not in LEGS:
            ap.error(f'leg must be one of {LEGS}')
        _runOne(caseName, leg, args.nSteps, args.out, args.scheme)
        return
    _drive(args.cases or AFFECTED, args.nSteps, args.out, args.scheme, args.timeout)
    _report(args.out)


if __name__ == '__main__':
    main()
