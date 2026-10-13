---
name: call-audit
description: Find where a warpSPH step spends host time -- torch-op storms, warp kernel launches and host syncs (`.item()`, `bool(tensor)`, `.cpu()`, `torch.cuda.synchronize`) per named section, with the Python call site of every sync -- for ANY registered case and scheme. Use when GPU utilisation is low, a step is slower than its GPU time, a CUDA graph silently falls back, after touching a scheme/module on the hot path, or before and after a performance change to show the delta. Produces the section tree with per-call counts, syncs per step, coverage (how much happens outside any section), the heaviest sections' top ops / kernels / sync sites, and the worst steps.
---

# Call audit

Counts, per `record_function` section (all ~300 sections in warpSPH / warpSPHCore, plus the
`[runner] ...` ones around the integrator, timestep hook, diagnostics, timer and speed read):

* **ops** every aten op dispatched (the "thousands of tiny torch calls"),
* **launches** warp kernels through `wp.launch` / `wp.launch_tiled`,
* **syncs** host synchronisations with the Python file:line that asked for them,
* **replays** CUDA-graph replays (a replay issues no ops: ops under a graph are the *capture*).

Off by default and free when off. Implementation: `warpSPHCore/callAudit.py`, riding on
`warpSPHCore.profiling.record_function`.

## Run it

```
python scripts/audit_case.py <case> [--scheme S] [--nx N] [--steps 60] [--warmup 10] \
       [--set key=value ...] [--cudaGraph | --no-cudaGraph] [--no-torch-ops] [--out file]

python scripts/audit_case.py dambreak --scheme artificialCompressible --nx 70
python scripts/audit_case.py sloshingTank --scheme artificialCompressible --nx 100 --set acBounceRamp=0.5
python scripts/audit_case.py hydrostaticColumn --scheme deltaSPH --nx 64
```

It prints the report and writes it to `scripts/out_callAudit/<case>_<scheme>_<time>.txt`.
Any other script: `WARPSPHCORE_AUDIT=1 [WARPSPHCORE_AUDIT_WARMUP=10] python scripts/...` prints
the same at exit; in code `callAudit.enable(warmupSteps=10)`, `callAudit.treeReport()`,
`callAudit.report()`, `callAudit.stepReport()`, `callAudit.reset()`, `callAudit.markStep()`
(the runner calls it once per step).

Rules: this is a **counting run** -- no video, short (30-100 steps), the aten dispatch hook slows
it ~3x, so read counts, never timings. One GPU job at a time (CLAUDE.md). Discard warm-up
(first-step compilation, graph capture, cache fills) with `--warmup`.

## Reading the table

* `calls`, `ops/call`, `launch/call`, `sync/call`: inclusive of nested sections. `self ops`,
  `self sync`: only what the section itself issued (its children excluded) -- the unattributed
  remainder of a parent, i.e. where to add a finer `record_function`.
* `sync/step`: the share of all syncs per marked step. Each sync drains the GPU queue; ~20+
  per step is why utilisation sits at 25-40 %.
* `coverage` line: percent of ops / launches / syncs inside a section; anything under ~99 %
  means a phase is not wrapped. Its `OUTSIDE any section` lines list the sync sites and ops
  that are not attributed -- add `with record_function("[scheme - NN] - name"):` around
  that call (it is free when the audit is off).
* Heaviest sections list their top torch ops (`mul x138` = 138 per call), top kernels and top
  sync sites (`file.py:line function xN`).
* The worst-steps table names the steps that spike (a Verlet rebuild, a graph re-capture) and
  which sections they spiked in.

Typical findings: a `float(tensor)` / `bool(mask.any())` / `.item()` in per-step code (move it to
a device scalar or compute it once per run), a host read per RK stage, a convergence test read
every iteration, an eager first iteration or per-step graph capture, per-step setup in many tiny
torch ops (fuse into one warp kernel).

## Before / after

Run the same command on the same case before and after the change, keep both report files, and
quote `sync/step`, the section's `ops/call`, and the coverage line. A change that adds a sync
to a graphed step is a regression (CLAUDE.md rule 5).

## Adding sections

`from warpSPHCore.profiling import record_function` then
`with record_function("[myScheme - NN] - what it does"):`. Names are the table's rows; keep them
stable and descriptive, number scheme phases in order.
