# warpSPH — working rules for Claude

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
2. **Stream progress while running.** `progress=True, quiet=False` (the
   runner's per-step `| t=... | maxVelocity=...` rows), plus any extra
   diagnostics via a flushed print or a `case.diagnostics` hook. Never write a
   probe that only prints a summary after `run()` returns. `stallDtSteps` only
   catches dt pinned *exactly* at `minDt`; check that simulated time is
   advancing from the log.
3. **One at a time.** GPU runs are kernel-launch bound; parallel runs slow
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
