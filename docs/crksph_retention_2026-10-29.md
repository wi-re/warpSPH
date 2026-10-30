# CRKSPH obsolete-work retention and CUDA allocator capture (2026-10-29)

The runner now updates `RunResult.state` after each accepted step in both ordinary and pipelined loops. Previously the result kept the original system alive throughout integration, including its first-step pair-work vector. This is an ownership fix: the numerical state, scheme, integrator, fresh stages, forcing, precision and neighbour representation are unchanged. COO i-index removal remains deferred by user decision.

At 96³ (884,736 particles), the short-window peak live allocation falls from **16.006 to 10.333 GiB**. The matched ownership profiles contain two distinct 5.601 GiB work storages before and one afterward. The current cross-stage W remains necessary. The total 5.673 GiB peak reduction also includes small obsolete state buffers.

## Matched measurements

All timings use recompute storage, 36 accepted steps, first six discarded for steady timings, two fresh-process-internal repeats. 32³ resumes the retained t4 checkpoint and ends at t4.136861; 96³ starts from rest and ends at t0.0523248. These are short windows, not a full-t4 peak-memory bound.

|Resolution / ownership|Steady wall ms/step, repeats|Run wall seconds, repeats|Peak allocated GiB|Peak reserved GiB|
|---|---:|---:|---:|---:|
|32³ before|36.251 / 35.820|4.815 / 2.994|0.903|1.070|
|32³ after|36.145 / 36.046|3.601 / 3.085|0.693|0.857|
|96³ before|525.880 / 531.265|21.515 / 21.272|16.006|21.639|
|96³ after|531.172 / 536.722|21.870 / 21.495|10.333|13.396|

After-fix reserved peaks vary between 11.355 and 13.396 GiB across the two 96³ repeats. Sampled own-process memory peaks at 14.186 GiB versus 22.479 GiB before. These are different measurements and their maxima need not coincide. At this short-window allocated high-water, memory is about 12.25 KiB/particle.

Every job launched with no other compute user visible. Brief unrelated Python compute (~1.9 GiB) subsequently appeared during timing32-after, timing96-after and near trace96-after teardown; exact samples are retained. No other session was changed. Own allocator measurements remain interpretable, but the approximately 1% 96³ timing difference cannot establish a throughput regression. Do not attribute a speed change to this lifetime fix.

Separate three-step synchronized ownership profiles cost 6.094 / 6.176 s (2031 / 2059 ms/step); allocation-history traces cost 22.037 / 22.335 s (541 / 549 loop ms/step). These runs are instrumented and must not be used as production throughput figures. Total supervisor job wall times, including process startup/export/analysis overhead, are in status.json.

![Matched short-window allocator measurements](../.tmp/crk-retention-20261029/analysis/retention-memory.png)

## What the reserved/live gap contains

Torch memory history records allocations, free requests, completed frees and Python stacks. Native memory-viz timeline and allocator-state pages are saved locally alongside raw snapshots. Capture uses the installed Torch `_record_memory_history` with all contexts and Python stacks, 400,000-entry capacity; no per-section synchronization in the history runs. No allocator setting changes or integration-time cache flushes were introduced.

|Same-instant snapshot|Live GiB|Inactive GiB|Reserved GiB|Awaiting-free GiB|
|---|---:|---:|---:|---:|
|Before, accepted step36|15.623|6.016|21.639|0|
|After, accepted step36|9.951|3.446|13.396|0|
|Before, run returned|9.552|12.087|21.639|0|
|After, run returned|9.550|3.846|13.396|0|

All 28,056 free requests in each recorded trace have adjacent free-completed events; no observed pending frees remain. This rules out delayed stream frees as the explanation in these captures, without claiming asynchronous frees never occur in general. The original extra W was genuinely live during the loop. It was released at run exit and then became inactive cache. The approximately 6 GiB reservation gap during the old loop was inactive space, not an additional live tensor and not pending deallocation.

One observational `empty_cache()` call **after integration** leaves live storage unchanged. Before the fix it releases 11.465 GiB of wholly inactive segments, reducing reserved memory to 10.174 GiB. After the fix it releases only 0.057 GiB, leaving 13.340 GiB reserved: 3.790 GiB is free space inside segments that still contain live allocations.

For example, one 5.602 GiB segment contains two live neighbour-index arrays of 1.867 GiB each, a small live scalar array, and a large 1.821 GiB hole. Another 1.867 GiB segment contains only 0.122 GiB of live small arrays but a 1.300 GiB hole. Such holes cannot be returned as whole CUDA segments while their neighbours remain live. Allocator layout depends on allocation order; the cache-release result is not a portable fixed memory budget. No further lifetime leak is inferred from reserved bytes alone.

The live budget now consists chiefly of one padded W (5.601 GiB), neighbour indices/rows/geometry metadata (3.781 GiB), particle-state and remaining coefficients/stages/workspaces. The corresponding synchronized ownership profile peaks at 10.260 GiB, compared with 15.895 GiB before. This profile is distinct from the less intrusive 10.333 GiB timing-window high-water. Padded slots remain 501,202,944; the deferred COO i-index array alone costs 1.867 GiB.

## Numerical and lifecycle checks

- All eight finite queue jobs passed; no duplicate or full-length trajectory was launched.
- Initial/final fields and sampled live W match bitwise across both timings and repeats. All five paired forcing/RNG checkpoints also match exactly.
- Raw stage validity, positivity, source guards and no-extra-RHS guards pass. Maximum measured relative energy-budget residual is 8.61e-9.
- 2,362 archived source/driver/helper SHA256 pins match. The two 1,179-file runtime trees differ only in runner.py's two ownership assignments and comments.
- 27 CPU controls pass, including zero/one/multiple-step ownership, collection of obsolete work, intentional early termination, and shared-work controls. The CUDA pipelined ownership test passes.
- Existing graph replay validation fails capture and falls back to eager; the sealed before-fix runtime reproduces the same failure. The CRK graph-route check passes, but actual pipeline graph replay equivalence remains unavailable. Initial failed assertions and baseline evidence are retained, not relabelled as passes. The ownership test specifically checks accepted-state ownership through the pipeline, even when capture falls back.
- All timing/profile/trace recordings were reviewed through before/after contact sheets. Patterns match; 96³ is a tiny startup window and density auto-scaling magnifies small fluctuations, not a developed-turbulence qualification.

## Artifacts and reuse

Artifact root: `.tmp/crk-retention-20261029/`. Exact inputs, source trees, launch commands, logs, per-process GPU telemetry, videos and ownership addresses are retained.

- `analysis/comparison.json`: fields/W comparisons, allocator states, ownership and telemetry.
- `analysis/integrity.json`: source pins and forcing-checkpoint equality.
- `analysis/{before,after}-memory-timeline.html`: native Torch allocation timelines.
- `analysis/{before,after}-allocator-state.html`: native Torch segment/block views.
- `trace96-{before,after}/repeat0/allocator.pickle`: raw snapshots usable by Torch memory-viz; compressed copies retained as well.
- `analysis/contact32.jpg`, `analysis/contact96.jpg`: reviewed recordings.
- `cpu.xml`, `gpu-ownership.xml`, `gpu-lifecycle.xml`, `gpu-baseline-graph.xml`: successful and failed test evidence.

`benchmark_crkWork.py --memory-history` enables these optional captures. `analyze_crkRetention.py ROOT` generates comparisons and native HTML offline. History capture is separate from timing qualification; snapshots only observe Torch's allocator and do not by themselves account for every CUDA allocation owned by other libraries. The loopback viewer is local; snapshots have not been uploaded.

Verdict: keep the ownership fix. It removes the obsolete vector without numerical changes. Retain the present neighbour representation and allocator policy; graph capture remains a separately reproduced limitation.
