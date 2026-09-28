# warpSPH — working rules for Claude

## Plans: keep PLANS.md current (core rule)

`PLANS.md` is the index of every plan: one row each, with **Last worked** and
**Last surveyed** dates. This is how the user keeps track of what is actually
going on, so it is not optional bookkeeping:

1. **Working on a plan** (code, runs, decisions, a results section) — in the
   same session, update that plan's row: *Last worked* = today, and a
   one-phrase *Where it stands* / next step. One row per plan; never add a
   second entry for the same plan.
2. **Reviewing without working** (a status pass) — update *Last surveyed*.
3. **Before chasing a new problem**, put it somewhere first: an existing
   plan's row, a short entry in `OPEN_PROBLEMS.md`, or a new plan file with
   a new row. Say which in the reply. Don't open an investigation that has
   no row. Resolved `OPEN_PROBLEMS.md` entries move to
   `docs/historic_plans/RESOLVED_PROBLEMS.md` (one-line stub left behind).
4. The **Focus** line is the user's; suggest, don't set it. User decisions
   go in `PLANS.md`'s decisions log as well as in the plan they concern.
5. **Branches:** work on `dev`; merge it back into `main` periodically.

## Running simulations (probes, validation runs, A/Bs, reference checks)

These apply to **every** run, including throwaway inline probes that call
`warpSPH.runner.run(...)` directly — not only the `scripts/probe_*.py` /
`scripts/run_*.py` CLIs, which already default `--video` on.

1. **Video on.** Pass `plot=True, video=True` (vispy backend — never
   matplotlib, ~50-80x slower encode) and an `exportRoot`. `CaseSpec.video`
   defaults to `False` for the test suite's sake, so a direct `run()` call does
   **not** get video unless asked. Scalar diagnostics say *that* something
   broke; the frames show *where* and *how*. Only skip video for a tight loop
   of many tiny A/B runs, and say so.
2. **Stream progress while running.** `progress=True` (the runner's
   `| t=... | maxVelocity=...` rows, ~10 a second; an explicit `progress=True`
   keeps them even under `quiet=True`), plus any extra
   diagnostics via a flushed print or a `case.diagnostics` hook. Never write a
   probe that only prints a summary after `run()` returns. `stallDtSteps` only
   catches dt pinned *exactly* at `minDt`; check that simulated time is
   advancing from the log.
3. **Watch it, and let it stop itself when frozen.** The runner's velocity
   alarm is on by default: a `[warpSPH] VELOCITY ALARM` line when `max |v|`
   passes 100x the expected velocity (it flags, never stops). Probes built on
   `scripts/_runWatch.py` (`probe_deltaSPHMarrone.py`, `probe_contactLine.py`)
   also draw every step while the alarm is active and stop a frozen run
   (`--stallProgress 1e-3`); direct `run()` calls should pass
   `velocityAlarmPlotInterval=1, stallProgress=1e-3` themselves (`CaseSpec`
   has the stall watchdog off). Log to a file through `stdbuf -oL tr '\r' '\n'`
   (plain `tr` block-buffers, and the log stays empty), and arm a `Monitor` on
   `grep -E --line-buffered "VELOCITY ALARM|stopping|stops the run|non-finite|Traceback"`.
   README "Watching a run" has the fields, the probe helper and the
   `onVelocityAlarm` callback.
4. **One at a time.** GPU runs are kernel-launch bound; parallel runs slow
   each other down. Chain them (`a; b; c`) in one tracked background task —
   no detached `( ... &)` children. Use `setsid nohup` only when a run must
   survive a session restart, and say so.

Details and history: the `export-video-on-validation-runs` and
`run-simulations-sequentially` memories; `FREESLIP_DAMBREAK_FINDINGS.md` §9.6.

## Before re-investigating a hard WCSPH/ACSPH/mDBC issue

Check `OPEN_PROBLEMS.md` first.

## Python environment

`/home/lu26029/miniconda3/envs/warp/bin/python` (conda env `warp`); run
scripts from the repo root so `warpSPHBootstrap` resolves.
