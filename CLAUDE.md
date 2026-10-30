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

## Numerical-model changes: establish the reason for each choice

User decision (2026-10-20): for each change to the numerical model, verify the
formula against its paper, cross-check the corresponding Spheral implementation
where applicable, then use a case that explicitly measures that choice's
influence. Distinguish the published method, current reference-code behavior,
and any local experimental variant. Reference-code agreement alone does not
justify adoption.

- Change one model choice at a time, with a declared baseline and quantitative
  targets. Record initialization differences needed to maintain the same physical
  state (for example, pressure-balanced u when changing a density estimator).
- Report accuracy, dissipation/entropy, positivity and relevant conservation
  properties, including costs and limitations. Conserved total energy or a
  visually appealing interface alone is insufficient. Check timestep/resolution
  sensitivity when it could change the conclusion.
- Document the evidence and a keep/reject/unresolved verdict before promoting a
  choice to a default. Preserve failed cases as evidence; do not tune to a picture.
- In the CRKSPH/Spheral comparison, keep the selected Butcher integrator and fresh
  stage evaluations fixed. Do not pursue CheapRK2 derivative reuse or cached
  values that change kick/drift/stage semantics as model improvements. Existing
  such comparison rows remain diagnostic records, not adoption candidates.

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
   advancing from the log. A long run with no output or tracking until it
   finishes is the same disease as piping its output to tail -- and a monitor
   tailing a silent log is the same symptom: with no new lines you cannot
   tell a run stalled at a 1e-27 timestep from one progressing normally.
   The cost of *some* tracking, even a rudimentary step number / t / dt, is
   marginal against the cost of waiting hours on a run that is a halting
   problem. For long runs, always pass an explicit `progress=True` (it
   survives `quiet=True`) or at least a periodic flushed step/t/dt line.
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
5. **Register long runs with the viewer daemon by default (user, 2026-10-30).**
   Use the native `warpsph_remote.publisher` completed-step adapter, preserving
   existing hooks, video and diagnostics. Register each trajectory as it starts;
   retain its non-secret run ID/endpoint. Short numerical fixtures need not
   register unless testing the adapter. Daemon pauses must suspend external
   wall/stall budgets. Preserve frozen sources, including the viewer adapter;
   never inject into a running solver. An explicit opt-out is allowed when the
   user requests it. Report registration failures rather than silently omitting
   the viewer on a long campaign.
6. **Look at performance.** Report wall time and ms/step with every run (the
   progress rows and the completion report give both) and compare with the last
   figure for the same case; reference: Marrone 3.1 nx 67 to t\* 7.68, video on,
   defaults, 108.7 s (5.6 ms/step), `docs/historic_plans/SMALL_PROBLEM_PERFORMANCE.md` §1.
   A step that falls out of the CUDA graph, adds a host sync (`.item()`, a
   convergence read) or slows by more than run-to-run noise is a regression to
   say so about; `benchmarks/marrone31/bench_step.py` splits a step into
   integrator / hook / diagnostics with GPU-busy time, launches and syncs.
   **ACSPH:** `CaseSpec.cudaGraph` defaults to `False`; a direct `run()` call of an
   ACSPH case must pass `cudaGraph=True` (the probes and the `dambreak` case default it on): bitwise, and
   Marrone nx 70 to t = 2.45 goes 1475 s -> 333 s (OPEN_PROBLEMS §33). Also watch
   `pseudoIterations`: a step at the 200 cap every step is a non-converging
   dual-time loop, not a slow one.
   **Find host syncs and torch-op storms with the call audit:** `WARPSPHCORE_AUDIT=1 python scripts/...`
   (off by default; every existing `record_function` section then reports torch ops, warp launches,
   host syncs with their Python call site, per call and per step, and a worst-steps table at exit;
   `warpSPHCore/callAudit.py`). Use a short run: the dispatch hook slows it ~3x, only the counts matter.
   For any case: `python scripts/audit_case.py <case> --scheme S --nx N` (the `call-audit` skill).

Details and history: the `export-video-on-validation-runs` and
`run-simulations-sequentially` memories; `docs/historic_plans/FREESLIP_DAMBREAK_FINDINGS.md` §9.6.

## Before re-investigating a hard WCSPH/ACSPH/mDBC issue

Check `OPEN_PROBLEMS.md` first.

## Python environment

`/home/lu26029/miniconda3/envs/warp/bin/python` (conda env `warp`); run
scripts from the repo root so `warpSPHBootstrap` resolves.
