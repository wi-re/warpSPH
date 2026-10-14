---
name: module-microbench
description: Micro-benchmark ONE warpSPH module call (or a family of interchangeable ones, e.g. the eleven viscosity switches) -- torch ops, warp kernel launches, host syncs with their Python call site, and host-enqueue / wall / GPU-busy time per call -- on a realistic state, without running a whole case through the call audit. Use when comparing implementations of the same operator, when a module is suspected of host syncs or op storms, before/after optimising one module, or to roll a module's cost up per RK step.
---

# Module micro-benchmark

`call-audit` shows what a whole step issues, section by section. This isolates one call: a table row per
implementation with **ops / launches / syncs per call** (exact, from `warpSPHCore.callAudit`) and
**host us / wall us / GPU us / device kernels per call** (indicative: one GPU, one N, warm cache).

## Pieces

* `src/warpSPH/utils/microBench.py` -- `microBench(fn, label, warmup=3, calls=10, timedCalls=50)` -> `MicroResult`;
  `formatTable(rows)`, `formatSyncSites(rows)`. Two passes, never at once: counting (audit hooks on) and timing
  (hooks off: host enqueue with no sync between calls, wall with one sync at the end, torch-profiler GPU busy time
  and device launches). The section tree of the counting pass is kept in `result.tree`.
* `scripts/microbench_switches.py` -- the worked example: every `ViscositySwitch` (terms, update, step hook),
  state from a short real run of a case with that switch selected, per-call and per-step (stages) tables, sync call sites,
  report in `scripts/out_microbench/`. Evidence copy: `docs/microbench/`.

```
python scripts/microbench_switches.py                                    # sod2d CompSPH, every switch, default N
python scripts/microbench_switches.py --nx 100 200 400                   # N scaling (counts at the first)
python scripts/microbench_switches.py --switches CullenHopkins Wadsley2017 --case sod --nx 2000
```

## Benchmarking another module

1. Get a realistic state: a short real `run(case, nSteps=20, plot=False, video=False, params=...)` with the
   variant selected, then read `r.state` (system), `r.state.state`, `r.state.adjacency`, `r.ctx.config`,
   `r.ctx.schemeConfig`, `r.ctx.config.dt`. (A setup run only: no video; the micro-benchmark is the measurement.)
2. Wrap the call in a no-argument closure that is repeatable: same inputs, nothing growing.
   `microBench(lambda: op(state, config, schemeConfig, adjacency), 'name op')`.
3. Anything that mutates state or returns at once on a repeat (a step hook that returns when time did not advance,
   a cache) must be driven so each call does the real work -- check `ops` is not suspiciously 0-1 (the first
   switch-hook table measured the early return).
4. Copy `microbench_switches.py` as `microbench_<module>.py`; keep the per-step roll-up (`perStep`) if the module runs
   every RK stage.

## Caveats

* The ops column counts every aten dispatch, including torch's own `_record_function_enter/exit` if a module imports
  `torch.profiler.record_function` instead of `warpSPHCore.profiling.record_function` (the gated one) -- those two ops per section
  are overhead you can remove, not work: grep the module's imports when `top ops` shows them.
* After the ops are gone the host time is the Python around the launches. `cProfile` 200 calls of the closure and read the
  cumulative column: the operator framework (`launchOperator`, autograd `Function.apply`, array views) was ~70 % of the Wadsley detector.

## Reading it

* `ops` ~ torch dispatches (fuse into a warp kernel if > ~30 for a small elementwise job); `launch` = warp kernels;
  `dev k` = ALL device kernels/memcpys the profiler saw (torch ops each launch one or more).
* `sync > 0` is a finding: `formatSyncSites` names the file:line. `.item()`, `bool(t)`, `.nonzero()`, a boolean-mask
  index, `bincount`, `segment_reduce(lengths=...)`, `adjacency.livePairs()` all read the host.
* `host us` ~ `wall us` and `gpu us` much smaller = host bound (launch/op overhead); a graph or fusing fixes it.
  `gpu us` flat across N while other rows grow = a latency-bound kernel (serial per-thread loop, low occupancy:
  look at its lanes/tiled variant).
* Per-step rows multiply per-stage calls by the stage count; a step hook counts once.
* Time columns are for the GPU in use and drift with other jobs on it (one GPU job at a time, CLAUDE.md); the counts do not.
