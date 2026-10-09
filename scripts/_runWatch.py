"""Run-watching flags for probe scripts: the velocity alarm and the stall
watchdogs (`src/warpSPH/runner/velocityAlarm.py`, README "Watching a run").

Deliberately free of `warpSPH` imports: a probe builds its parser before it
bootstraps the precision, and importing `warpSPH` first would fix it too
early.

Use::

    from _runWatch import addWatchArguments, watchOverrides
    addWatchArguments(ap)                 # after the script's own flags
    args = ap.parse_args()
    ...
    kw.update(watchOverrides(args))       # into run(case, **kw)
"""

from __future__ import annotations

from typing import Any, Dict


#: Defaults of the `--velocityAlarm*` / `--stall*` flags `addWatchArguments`
#: gives a probe script. Probes record video (CLAUDE.md), so the alarm draws
#: every step while active; a stall watchdog is on, since a runaway that pins
#: dt otherwise burns hours at the floor (FREESLIP_DAMBREAK_FINDINGS.md §1).
#: `stallProgress` 1e-3 = "at this rate the run needs > 1e6 more steps". 1e-4
#: is too lax for a hover: the ACSPH Marrone nx=70 runaway (2026-09-28) sat at
#: dt 2.6e-7 for ~750 steps, 2.6e-4 of tLimit per 1000 steps, which 1e-4 never
#: trips. Healthy runs advance >= 1e-2 of tLimit per 1000 steps (Marrone,
#: sloshing, toys). OPEN_PROBLEMS.md §10 has the run.
PROBE_WATCH_DEFAULTS = dict(velocityAlarmFactor=100.0, velocityScale=None,
                            velocityAlarmPlotInterval=1, velocityAlarmStopRatio=None,
                            stallProgress=1e-3, stallDtSteps=None)


def addWatchArguments(parser, **defaults):
    """Add the run-watching flags to a probe script's `argparse` parser.

    `velocityAlarmFactor`, `velocityScale`, `velocityAlarmPlotInterval`,
    `velocityAlarmStopRatio`, `stallProgress`, `stallDtSteps` -- the same names
    as the `CaseSpec` fields (and as `warpsph-run`'s flags). `defaults`
    overrides `PROBE_WATCH_DEFAULTS` for this script. Pass the parsed args
    through `watchOverrides` into `run(...)`."""
    d = dict(PROBE_WATCH_DEFAULTS, **defaults)
    group = parser.add_argument_group('run watching (runner/velocityAlarm.py, README "Watching a run")')
    group.add_argument('--velocityAlarmFactor', type=float, default=d['velocityAlarmFactor'],
                       help='flag (not stop) when max |v| > this x the expected velocity; '
                            '0 disables (default %(default)s)')
    group.add_argument('--velocityScale', type=float, default=d['velocityScale'],
                       help="expected velocity; unset = the case's own, else an estimate")
    group.add_argument('--velocityAlarmPlotInterval', type=int,
                       default=d['velocityAlarmPlotInterval'],
                       help='frame every N steps while the alarm is active (needs video); '
                            '0 leaves frames alone (default %(default)s)')
    group.add_argument('--velocityAlarmStopRatio', type=float, default=d['velocityAlarmStopRatio'],
                       help='stop once max |v| > this x expected; unset = never')
    group.add_argument('--stallProgress', type=float, default=d['stallProgress'],
                       help='stop once 1000 steps advance sim time by < this x tLimit; '
                            '0 disables (default %(default)s)')
    group.add_argument('--stallDtSteps', type=int, default=d['stallDtSteps'],
                       help='stop once dt sits at minDt for this many steps; unset = never')
    return group


def watchOverrides(args) -> Dict[str, Any]:
    """The `run(...)` keyword overrides for the flags `addWatchArguments` added.
    `0` switches the factor, the plot interval and `stallProgress` off; an
    unset optional flag is left out, so a script's own value for it (e.g. a
    `stallDtSteps` it always passes) survives a `kw.update(...)`."""
    overrides = dict(velocityAlarmFactor=args.velocityAlarmFactor or None,
                     velocityAlarmPlotInterval=args.velocityAlarmPlotInterval or None,
                     stallProgress=args.stallProgress or None)
    for name in ('velocityScale', 'velocityAlarmStopRatio', 'stallDtSteps'):
        if getattr(args, name) is not None:
            overrides[name] = getattr(args, name)
    return overrides
