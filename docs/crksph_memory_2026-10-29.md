# CRKSPH memory ownership measurement (2026-10-29)

Follow-up: the obsolete RunResult owner has now been removed. Matched short-window
96³ peak live allocation is 10.333 GiB; the historical measurements below retain
their original scope. [Ownership fix and Torch memory-viz captures](crksph_retention_2026-10-29.md).

Particle-state storage is small. Pair-sized outputs, expression temporaries and retained compatible-work vectors dominate. The native X6 cross-stage optimization is active. A separate first-step vector is retained through the runner result object; no solver optimization has been applied.

|Measurement|32³ near t4|96³ startup|
|---|---:|---:|
|Particle fields stored once|2.500MiB|67.500MiB|
|Physical list: indices/rows|0.068GiB|1.681GiB|
|Verlet list: indices/rows, no padding|0.136GiB|3.401GiB|
|Verlet list: indices/rows, padded|0.139GiB|3.741GiB|
|One padded pair vector|0.207GiB|5.601GiB|
|Stage pair outputs ap/av/f|0.484GiB|13.070GiB|
|Maximum stage live Torch allocation|1.308GiB|34.493GiB|
|X6 transient above larger section boundary|0.415GiB|11.203GiB|

Full state here means the actually stored particle fields (80bytes/particle plus16bytes source ledger), excluding adjacency, derived Q output property, stages and grids. Unique storage addresses deduplicate tensors/views/shared references; this is not the64byte/particle export payload.

96³ physical-support directed pairs: 224,802,211; buffered pairs: 455,639,040; padded capacity: 501,202,944. Native indices are int32. The table includes both COO index arrays and CSR count/offset arrays. A single-index CSR representation would have a different cost and is not measured as an implementation change. The physical list uses exact current-position/support filtering, capacity=None; the trajectory keeps its unchanged2**(1/3) Verlet factor and capacity policy. Native list including stored geometry/hash metadata is 3.781GiB; shared geometry is not a second particle state.

## Peak and retained ownership

The primary96³ probe peaks at 34.493GiB during stage1 X6 accumulation. The ordered allocator trace reconstructs 34.490GiB requested storage at its peak; allocator block rounding makes this slightly lower than the Torch live-allocation high-water mark.

|Live at reconstructed peak|GiB|
|---|---:|
|Other live allocations|0.882|
|Neighbour indices|3.734|
|Retained first-step X6 vector|5.601|
|Stage pressure + viscosity vectors|11.203|
|Stage partition fractions|1.867|
|X6 expression temporaries|11.203|

At the end of the peak section, live Torch memory is 15.821GiB, and reachable unique storage accounts for 15.722GiB. Ownership group sums overlap and must not be added. The current PairWork work tensor and runnerResult.state.pairWork.work are distinct 5.601GiB storages. The latter is owned exclusively by runnerResult in the audited roots; it traces to first-step accumulation.

The runner initializes RunResult.state with the original system before the loop and replaces it at completion. That reference retains the first system and its first-step PairWork.work during later steps. It is additional retention beyond the current running work vector and does not indicate a per-stage history was reintroduced. A candidate fix is to update/release the obsolete result-state reference or retire unused pair-work scratch after finalization, with separate lifecycle/numerical/graph checks.

The X6 expression f*(ap+av), followed by stage weighting, materializes full pair-vector temporaries. Two such vectors coexist at the observed peak. Other measured costs include native stage pressure/viscosity vectors, f_ij, adjacency, small particle-state copies and output/plot allocations. These are concrete optimization targets without changing neighbour support or fresh stage semantics.

## Scope and checks

Four runs were serial:32³ four native steps resuming the saved t4 checkpoint;96³ four startup steps, then three/two-step observer repeats. The first observer missed the runner frame name; the second omitted the RunResult root. All original scalar peaks/traces are retained. Observer repairs and final complete roots are recorded; no numerical changes or failed physical trajectories. Each run has VisPy z=0 video, progress, fresh native RHS, finite/positive checks and source hashes.

The physical-list filtering is after each trajectory and never replaces its evolving neighbour policy. The32³ resumed step endpoint is not a bitwise comparison against a continuous restart control.96³ measures startup geometry; it does not reproduce the historical t0–4 peak42.10GiB or explain every later rebuild transient. Synchronization instrumentation can change asynchronous allocation lifetimes; no instrumented time is a production-throughput estimate.

Tensor-storage ownership excludes allocator rounding/cached blocks, foreign-library allocations, and Python/host memory. Remaining small live allocations are visible in allocator histories; this is not a byte-exact whole-device ledger. Original physical runtime pins remain unchanged.

|Probe|Steps|Wall seconds|Instrumented ms/step|RHS valid|
|---|---:|---:|---:|---:|
|probe32|4|5.722|1430.4|8|
|probe96|4|7.504|1876.1|8|
|probe96-ownership|3|6.027|2008.9|6|
|probe96-fullroots|2|4.913|2456.7|4|

Four CPU storage controls pass (view aliases, independent clones, cycles/computed properties/device filtering). Initial/final contact sheets for all four probes and the memory chart were visually reviewed. No blow-up or ejection in these representative slices; startup density colour limits amplify tiny lattice/interpolation fluctuations.

![Memory breakdown](/home/lu26029/.codex/worktrees/meshless-boundary-balance/warpSPH/.tmp/crk-memory-20261029/analysis/memory-breakdown.png)

Evidence: /home/lu26029/.codex/worktrees/meshless-boundary-balance/warpSPH/.tmp/crk-memory-20261029


Additional retained stage/update storage first reached through integratorRetained
and runnerPreviousResult is326.53MiB at the audited post-accumulation boundary,
after deduplicating current state/list/work and local RHS scratch in the declared
root order. This is much smaller than each5.601GiB pair-work vector. Group totals
in raw JSON overlap; incremental totals depend on root order. The unexplained
gap between reachable storage and live allocation at that boundary is about
0.099GiB; library workspaces and allocator granularity are outside the ownership
walk. The additive peak chart instead uses all requested allocation events.

Final integrity audit:1,178 original physical-runtime/source pins unchanged;
all four executed driver/helper archives match their pre-run hashes.26/26
RHS evaluations are finite/positive, with no extra evaluations; maximum observed
energy-ledger residual7.61e-9. Total instrumented run wall24.17seconds, plus
process startup/source archival and separate physical-list filtering. The table
reports each actual instrumented cost; these are not production step timings.

The next implementation targets are the obsolete first-step result owner and
full-vector expression intermediates. Preserve compatible energy and every
fresh stage, with source-paired output/energy and graph/lifecycle controls before
claiming a reduction or promoting a change. No solver fix was applied here.
