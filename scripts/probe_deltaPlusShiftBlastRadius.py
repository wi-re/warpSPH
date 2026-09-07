"""Probe (`DELTASPH_VALIDATION_PLAN.md` Part 2 / Sec. 5.3.2): which registered
cases would actually change if `ShiftProperties.sun2017Eq7Shift` became the
shared default?

The Sun 2017 Eq. (7) fix multiplies the delta+ shift by 8. That is a real
physics change for any case that runs `ShiftingScheme.deltaSPH` with the shift
*active*, and no change at all for any case that does not -- so before the
default can be flipped, the first question is not "is it better" but "who is in
the blast radius". Grepping for `shiftProperties.active` answers that badly:
the field defaults to `True` (`buildDefaultShiftProperties`), so a case that
never mentions it is *in* the radius, and a case that sets it from a param is
in or out depending on that param's default.

So this resolves it the only reliable way -- build every registered case's
context exactly as `runner.run` would, and read the settings back off the
config it produced. No stepping, so it is seconds for the whole registry.

Reports per case: the scheme, whether shifting is active, which
`ShiftingScheme` it resolved to, and the verdict:

* `AFFECTED`     -- active, and `ShiftingScheme.deltaSPH`. Needs an A/B before
                    the default flips.
* `already-Eq(7)` -- affected, but the case already selects a config that turns
                    `sun2017Eq7Shift` on (i.e. `--scheme sun2017DeltaSPH`), so
                    flipping the shared default changes nothing for it.
* `shift-off`    -- shifting inactive; the scaling is never reached.
* `other-scheme` -- a different `ShiftingScheme` (`michel2022`, `implicit`,
                    ...), which does not use this scaling at all.
* `no-shifting`  -- the scheme config has no `shiftProperties` (compressible /
                    ACSPH families).

Usage:
  python scripts/probe_deltaPlusShiftBlastRadius.py
  python scripts/probe_deltaPlusShiftBlastRadius.py --cases droplet tgv-wc
"""
from __future__ import annotations

import argparse
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cases', nargs='*', default=None,
                    help='only these registered case names (default: all)')
    args = ap.parse_args(argv)

    from warpSPHBootstrap import bootstrap
    bootstrap(precision='float32')
    # `warpSPH.cases` registers lazily -- importing the package is not enough,
    # `importAll()` is what populates the registry (`scripts/run_sweep.py`
    # does the same thing to enumerate).
    from warpSPH.cases import importAll
    importAll()
    from warpSPH.runner import CaseSpec, buildContext, getCase, listCases

    names = listCases() if args.cases is None else args.cases

    rows = []
    for name in names:
        try:
            case = getCase(name)
            spec = CaseSpec(caseName=case.name, scheme=case.scheme,
                            params=dict(case.params)).merged(**case.defaults)
            spec = spec.merged(quiet=True, verbose=False)
            ctx = buildContext(case, spec)
            # `configureScheme` is where a case turns shifting on or off, so the
            # answer is only trustworthy after it has run.
            if case.configureScheme is not None:
                case.configureScheme(ctx)
            props = getattr(ctx.schemeConfig, 'shiftProperties', None)
            if props is None:
                rows.append((name, spec.scheme, '-', '-', 'no-shifting'))
                continue
            active = bool(props.active)
            schemeName = props.scheme.name
            if not active:
                verdict = 'shift-off'
            elif schemeName != 'deltaSPH':
                verdict = 'other-scheme'
            elif getattr(props, 'sun2017Eq7Shift', False):
                verdict = 'already-Eq(7)'
            else:
                verdict = 'AFFECTED'
            rows.append((name, str(spec.scheme), str(active), schemeName, verdict))
        except Exception as exc:                                # noqa: BLE001
            rows.append((name, str(case.scheme), '?', '?',
                         f'build failed: {type(exc).__name__}: {exc}'[:70]))

    w = max(len(r[0]) for r in rows) + 1
    print(f'{"case":<{w}} {"scheme":<22} {"active":<7} {"shiftScheme":<13} verdict')
    print('-' * (w + 60))
    for r in sorted(rows, key=lambda r: (r[4] != 'AFFECTED', r[0])):
        print(f'{r[0]:<{w}} {r[1]:<22} {r[2]:<7} {r[3]:<13} {r[4]}')

    affected = [r[0] for r in rows if r[4] == 'AFFECTED']
    print(f'\n{len(affected)} of {len(rows)} cases in the blast radius: '
          f'{" ".join(affected) if affected else "(none)"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
