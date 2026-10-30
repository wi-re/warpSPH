# Whole-row neighbour chunks and multi-GPU execution

Plan index: [PLANS.md](PLANS.md). Parent physics programme: [driven compressible turbulence](docs/roadmap/driven_compressible_turbulence.md).

Status, 2026-10-30: **Phase 2 single-device construction/lifecycle qualified; Phase 3 next**.
The count/scan/ownership design is agreed in discussion. Core/main and frontend/dev
are integrated; see the [integration report](docs/chunked_integration_2026-10-30.md).
The planner and opt-in standalone single-device chunk construction/lifecycle are
qualified on CPU/CUDA. Native solver dispatch, compatible pair work, distributed
execution and physical multi-GPU qualification remain open.
This document is the canonical implementation/handoff plan for both repositories;
the core carries a short companion pointer, `CHUNKED_NEIGHBORS_MULTI_GPU_PLAN.md`.

## Start here / next action

1. The user authorized integration at ACSPH checkpoint `3d80e5e`. The paired
   merge and focused CPU/CUDA gates are complete; read the integration report.
2. Use the active paired workspace below. Chunked neighbours first, then
   multi-GPU execution continue sequentially in these same checkouts.
3. Phase1/2 are qualified within the limits below. Start Phase3 by threading
   row ownership through shared operators and all query/reference field bundles;
   bring up support/density/corrections/gradients and a simple eager scheme before
   native CRK/CompSPH compatible work. Preserve lanes and fresh stages.
   Use explicit source paths; do not change global editable installs.
4. Keep the existing turbulence worktree available for ordinary single-GPU
   physics through128^3, subject to a short readiness/memory check. No such
   campaign is scheduled or launched by this integration.

## Active implementation workspace (2026-10-30)

One paired workspace, used for the full sequential effort:

|Repository|Checkout|Branch|Integrated base|
|---|---|---|---|
|Frontend|`/home/lu26029/.codex/worktrees/chunked-neighbor-planner/warpSPH`|`codex/chunked-neighbor-planner`|`8b58008`|
|Core|`/home/lu26029/.codex/worktrees/chunked-neighbor-planner/warpSPHCore`|`codex/chunked-neighbor-planner`|`04e126e`|

The originally prepared `chunked-neighbors-multigpu` workspace is owned by
another chat and was left intact. The user requested a new worktree; this chat
created the active managed frontend workspace and its paired core checkout from
the same integrated commits. The frontend managed worktree is attached to this
chat. Both imports resolve to
these paths when their `src` directories are supplied explicitly in `PYTHONPATH`;
core reports0.6.3. Bootstrap does not select checkouts. For restricted read-only
import checks, use writable `NUMBA_CACHE_DIR`/`MPLCONFIGDIR` under`/tmp` rather than
requiring cache writes inside a new checkout. No editable installs were changed.
Set int64 offsets before imports when starting the wide-offset implementation;
the ordinary integration baseline retained its default int32 offsets.

The previous `meshless-boundary-balance` frontend/core worktrees remain intact
on their existing branches for the smaller-turbulence physics track. No worker
was launched there during this integration. The two phases do not get separate
implementation worktrees.

## Pre-integration repository snapshot (2026-10-29; historical)

|Repository|Current feature checkout / branch|Last code/scope commit when surveyed|Shared checkout|
|---|---|---|---|
|warpSPH|`/home/lu26029/.codex/worktrees/meshless-boundary-balance/warpSPH`, `codex/meshless-boundary-balance`|`6ad5d2b` (with this subsequent documentation handoff)|`/home/lu26029/dev/warpSPH`, `dev` at`5d4f41d`; `main` at`66492a2`|
|warpSPHCore|`/home/lu26029/.codex/worktrees/meshless-boundary-balance/warpSPHCore`, `codex/compressible-csr-only`|`3c56e2f` (plus companion handoff)|`/home/lu26029/dev/warpSPHCore`, `main` at`98081c1`|

At that snapshot the shared frontend had uncommitted ACSPH code/docs/tests. Its `dev`
and the turbulence branch have diverged (2 versus18 commits at the survey).
The shared core was clean and its feature branch was one commit ahead before
adding the companion plan. Re-evaluate all of this when executing Phase 0.
Frontend requires core>=0.6.3. Do not merge the frontend against the old core0.6.2.
The 2026-10-30 integration supersedes these checkout/head observations. The
user's Focus and ACSPH follow-up text are preserved.

Evidence already committed: CSR-only neighbour allocation and core0.6.3,
native CRK/CompSPH recomputation, retained-state lifetime cleanup, memory sweeps,
and a bounded CPU tile-view ABI probe. Read the reports rather than repeating
those investigations:

- [CSR-only implementation/parity](docs/compressible_csr_neighbors_2026-10-29.md)
- [CompSPH work/lifetime results](docs/compsph_work_lifetime_2026-10-29.md)
- [CRK retention/capture results](docs/crksph_retention_2026-10-29.md)
- [Seven-scheme memory sweep](docs/compressible_memory3d_2026-10-29.md)
- [Storage scope and row-chunk discussion](docs/tiled_edge_storage_scope_2026-10-29.md)

## User decisions and boundaries

- Integrate the current paired work into the main development checkouts after
  ACSPH reaches a mergeable checkpoint; develop this architecture separately.
- Keep whole particle rows together; partition by stored edge/byte budget.
  Each GPU holds the full particle/reference fields and only its owned J/W.
  Never gather the complete neighbour list or CRK pair-work tensor on one GPU.
- Reuse the existing separate count pass: distributed counting, synchronized
  int32 counts, replicated int64 prefix scan and deterministic partitioning,
  followed by direct local allocation/collection. Lists stay resident until
  the existing validity rules call for a coordinated rebuild.
- Preserve the numerical method: fresh stages, integrator/tableau, compatible
  work, forcing, EOS, support/Verlet policy and acceptance rules. No stage reuse
  or reduced RHS evaluations as a side effect of partitioning.
- Ordinary turbulence up to128^3 remains a separate physics track in the current
  worktree.96^3 has measured startup evidence;128^3 is a candidate, not a completed
  fit/performance/stationarity qualification.192^3/256^3 are later capacity gates.

Terminology: **row chunk** is a storage/ownership partition. Existing **neighbour
lanes** (`autograd/lanes.py`, `setNeighborLanes`) are CUDA threads cooperating on
one particle's reduction. They are independent axes. Freeze the existing lane
setting for chunk parity, then qualify combinations; do not rename or replace
that setting. Uniform tile/slot views remain optional, not a prerequisite.

## Data and execution contract

At rebuild generation g, all ranks agree on particle ordering, reference state,
supports, domain/periodicity, grid/search configuration and acceptance rule.
Each rank counts its disjoint counting interval. All-gather count slices into
C:int32[N]; do not sum duplicated full counts. Every GPU computes P:int64[N+1],
with P[0]=0, P[1:]=cumsum(C), and the same deterministic partition/owner map.
Validate a generation/config/partition signature before allocating pair buffers.

For a chunk owning particles[a,b):

```text
rowStart=a; rowCount=b-a; liveEdges=P[b]-P[a]
counts=C[a:b]; localOffsets=P[a:b]-P[a]
J[capacity] holds global reference particle IDs
W[capacity,dim] holds only this chunk's retained compatible work
local row r -> global particle i=a+r
```

Chunks contain variable particle counts for an edge budget; any optional row cap
is an additional bound. Include empty rows, non-divisible rank counts and empty
ranks. Define an explicit oversized-row error/fallback; never truncate neighbours.
Capacity headroom, byte strides and all associated buffers must fit the per-chunk
Warp limits. A global edge count may exceed INT_MAX without any allocation doing
so. Keep global offsets/prefixes int64 and IDs/counts int32; do not narrow a global
address before subtracting its chunk base. Account for the stored Verlet list,
not only physical-support neighbours. Count and collect use the same generation.

J and W reside together on their owner; no implicit flatten/concatenate is allowed
in production dispatch. Source fields use global indices. Query corrections,
extras and outputs require explicit local/global mapping: slicing positions alone
is insufficient. Single-GPU chunks write disjoint output ranges, without an
additional sum across chunks. Distributed particle outputs are gathered between
dependent passes, while pair data stays local. Reuse capacities until invalidated.

At256^3, replicated C/P cost64MiB/~128MiB per GPU. Derived particle matrices,
stage states, grids and communication staging must also be counted. Resident
chunking alone does not reduce total single-GPU J/W bytes; scratch lifetimes can
be bounded chunk-by-chunk, and distribution reduces owned pair bytes per GPU.

## Phase 0 — paired integration and isolation

- [x] Commit the scope/handoff in both feature repositories. Core companion is
  `f6c5900`; frontend handoff is the commit containing this plan. Production
  baselines remain `3c56e2f` (core) and `6ad5d2b` (frontend).
- [x] Obtain a mergeable committed ACSPH checkpoint through the existing user/task
  workflow. A clean tree is necessary but not evidence that numerical work is done.
  Preserve known failures and the ACSPH qualification record; do not message other
  sessions or invent a readiness signal.
- [x] Inspect fresh status/diffs and freeze paired baseline provenance. At the
  clean checkpoint merge `codex/compressible-csr-only` into the core's actual
  integration branch (currently`main`) first; run the bounded core gates.
- [x] Merge `codex/meshless-boundary-balance` into frontend`dev`. Reconcile shared
  runner/config, PLANS, roadmap and ACSPH changes deliberately; preserve both
  histories and avoid wholesale file replacement. Run paired import/config,
  CSR, shared-work and affected ACSPH gates. Promote frontend`dev` to`main` only
  according to the existing main/dev integration cadence.
- [x] Record merged frontend/core SHAs, core version, test results and remaining
  failures. Do not change global editable installs under active physics jobs.
- [x] Create one paired implementation workspace from those integrated commits. Suggested
  branch names are `codex/chunked-neighbors-multigpu` in both repositories; choose
  one frontend checkout and one sibling core checkout; record their absolute paths.
  Use both sequentially for chunks and multi-GPU, not one worktree per phase.
  Use managed frontend worktree creation/attachment and a paired core checkout;
  these names/paths are proposals, not already-created resources.
- [x] Verify imports explicitly: `warpSPH.__file__`, `warpSPHCore.__file__`,
  versions, offset dtype and both source hashes. Use explicit per-process import
  paths or an isolated environment; bootstrap selects precision, not repo paths.
  Keep plotting/integrator dependencies pinned as well.

Exit: committed merged baseline, bounded integration results, isolated paired
implementation checkouts, and the existing turbulence checkout still available.
Do not reverse-sync or archive it while a source-pinned physics worker is active.

## Phase 1 — deterministic planner (core, CPU first)

- [x] Implement immutable generation/partition metadata and deterministic
  whole-row boundary selection from C/P, with separate counting and compute owners.
  Ownership policy/version, edge/byte budget and capacity rules are explicit.
- [x] Test empty populations/rows, zero-work ranks, strongly uneven counts,
  exact-budget boundaries, oversized rows, capacity padding and deterministic
  replica agreement. Exercise total prefixes>INT_MAX with metadata-only fixtures,
  without allocating billions of edges. Test exact local/global reconstruction.
- [x] Planner-local `RowChunkPolicy` exposes edge/byte budgets, exact rational
  headroom, alignment, scalar work extents and optional row caps. Compute owners
  are logical ranks. Production backend/device selection remains a Phase2/4 task;
  the flat API, global defaults and existing neighbour lanes are unchanged.

Exit: exhaustive row coverage once, no row split, bounded local capacity,
identical partition on all replicas. This phase is independently reviewable.

### Qualified implementation and limits (2026-10-30)

Core `src/warpSPHCore/radiusSearch/rowChunks.py` provides `RowChunkGeneration`,
`RowChunkPolicy`, `RowChunkPlan` and `buildRowChunkPlan`. Rebuild configuration
signatures are supplied by the caller and must cover ordering, reference state,
supports, domain, grid/search settings and stored-list acceptance. The plan digest
also includes every count/prefix, generation, capacity policy and ownership map;
`validateSignature` rejects disagreement before future local allocation.

The CPU reference consumes an explicitly assembled CPU int32 count vector,
snapshots its prefix into immutable Python integers and materializes CPU int64
global/local offsets regardless of the flat default. It allocates no edge arrays.
Counting ownership balances contiguous row intervals; compute ownership balances
contiguous chunk counts (`contiguous-balanced-chunks-v1`), with empty owners
allowed. This is deterministic, not a measured optimal load balancer. Greedy
chunks retain zero-neighbour rows and never split an oversized row; such a row raises an
explicit error. Budgets include ceil-rounded rational headroom and alignment,
and conservatively bound the largest flattened scalar work array by INT_MAX.
Callers must account for every colocated pair buffer in bytesPerEdge. Particle
fields, row metadata, grids and communication staging need separate accounting.

At the Phase1 checkpoint this was a host reference for the ownership contract.
Phase2 adds standalone search and resident adjacency below. The production
replicated device planner, large-N memory/performance, native solver dispatch,
graph/AD and distributed transport remain open; physical CUDA accesses above
INT_MAX remain unqualified.

Validation: 59 planner controls (including 1024 exhaustive small count layouts)
pass in both offset configurations. Combined with the existing index-width,
CSR-only, split-collection, query-order and Verlet-validity suites, the int64 run
passes 160/skips 89 in 2.75 s; default int32 passes 159/skips 90 in 2.76 s, zero failures.
Skips are CUDA controls and int64-only cases. These are unit fixtures, not
simulations; no video or GPU workload was started. Imports are explicitly pinned
to this paired workspace; core remains 0.6.3, with shared installs unchanged.

Reproduce from the core checkout with `/home/lu26029/miniconda3/envs/warp/bin/python`,
`PYTHONPATH` set to the two active checkout `src` paths, `CUDA_VISIBLE_DEVICES=''`,
`OMP_NUM_THREADS=2`, `PYTHONDONTWRITEBYTECODE=1`, writable `NUMBA_CACHE_DIR` and
`MPLCONFIGDIR`, and `warpSPHCore_OFFSET_DTYPE=int32` or `int64` set before import:

```bash
python -m pytest -q -p no:cacheprovider \
  tests/radiusSearch/test_rowChunks.py tests/radiusSearch/test_indexDtypes.py \
  tests/radiusSearch/test_csrOnly.py tests/radiusSearch/test_splitCollection.py \
  tests/radiusSearch/test_queryOrder.py tests/radiusSearch/test_verletValidity.py
```

Local evidence/provenance: frontend `.tmp/row-chunk-planner-20261030/`.
Core implementation commit: `9a1057e` (on integrated base `04e126e`). Paired
source pins are recorded in `workspace-imports.json`; frontend's handoff commit
follows this core commit. The next direct count/collect and lifecycle milestone
is qualified below.

## Phase 2 — direct chunk collection and lifecycle (core, one GPU)

- [x] Add owned-row/global-row-base inputs to count/collect without changing
  neighbour acceptance or traversal order. Count first; allocate local buffers;
  collect directly. No monolithic intermediate J or reconstructed I.
- [x] Add a chunked adjacency container/iterator and connect validity, filters,
  capacity reuse, rebuild flags and generation tracking. Preserve the flat API
  for existing callers. Explicit diagnostics may request bounded/chunked export;
  do not hide a giant concatenation behind a compatibility property.
- [x] Force tiny budgets on ordinary2D/3D cases. Compare every live J/count/offset
  against the existing flat oracle, including periodic seams, unequal query/ref
  populations, all support modes and both stored-I/CSR-only contexts where supported.
  Use allocation/ownership checks to prove local construction and stable reuse.

Exit: exact order/content and count/collect consistency, correct invalidation,
no full-edge allocation, visible temporary/retained memory accounting.

### Qualified standalone API and limits (2026-10-30)

Core implementation: `31bc041`, following Phase1 `9a1057e`. New module
`radiusSearch/chunked.py` exports `ChunkedAdjacency`, `OwnedRowChunk`,
`radiusSearchRowChunks[_]`, `buildChunkedVerletList[_]`,
`filterChunkedVerletList[_]` and `chunkedVerletStaleFlag_`. These APIs and the
planner types are exported from the core root; they are explicitly opt-in.
There is no frontend backend switch or automatic solver dispatch yet.

The existing count/J-only collect/filter kernels receive query-position,
query-support and CSR-row slices together; references and J IDs stay global.
Host row bases define each owned interval. The only new Warp kernel fills optional
stored I with global query IDs. Traversal, acceptance, periodic wrapping, support
rules and per-row J ordering remain unchanged. Each generation freezes private
particle/domain snapshots before counting and collection. Counts synchronize to
the host reference planner at rebuild; the device also scans C into int64 P.
Local offsets subtract their global base in int64 before conversion to the
configured local offset width. Qualification uses int32 particle IDs/counts,
int32/int64 local offsets, FP32 CPU/CUDA and FP64 CPU.

`ChunkedAdjacency` has no global paddedJ/paddedI or implicit flattening property.
Iterate its owned chunks explicitly. Chunk adjacency row metadata is local;
source IDs and optional stored I are global. Do not pass these small adjacency
objects to unmodified global-row operators. `numOwners`/`owner` on explicit
search provide a single-device logical-owner fixture: all count intervals are
evaluated on that device, and only the selected owner's J is allocated. This
does not implement distributed counting or communication. Resident Verlet
lifecycle and filtering currently require one complete owner.

Resident lists return the same object while the existing motion/support validity
reduction accepts them. Forced rebuilds, motion/support changes beyond the validity buffer, and changes
to domain, support mode, scale, population, layout, budget or ordering key advance
generation/configuration metadata. Default rebuilds preserve old buffers;
explicit inPlace reuses pair/local-offset storage only for identical row intervals,
capacities and compatible layouts. It obsoletes old pair/CSR views; snapshots are
fresh. There is no maximal-capacity retention/shrink heuristic yet. Native W
accumulation/replay/migration across a rebuild is still Phase3 work.

Filtering checks validity first, then counts directly over source chunks and
collects each source/destination row overlap into the destination's local J.
It handles both merged and split chunk boundaries without assembling global pairs
or reconstructing CSR-only I. Filtered views record the source signature and
cannot be reused as resident Verlet supersets. A stale source fails before collect.
Callers must advance orderingKey when externally changing particle identity/order.
AD, deferred checks and CUDA graph capture fail explicitly; no flat fallback is
selected. Tensor-valued Verlet scales and distributed lifecycle remain unsupported.

Memory diagnostics report owned live/capacity pair bytes, policy-reserved pair
bytes, replicated/per-chunk row metadata, unique snapshots, host prefix bytes and
the grid/wrapped-query scratch frontier. These are logical storage measurements,
not allocator/process peaks or every intermediate of grid construction. Allocation
spies prove direct bounded J allocation and absent CSR-only I; resident and forced
rebuild controls prove reuse. Single-GPU resident chunking still retains all its
owned pair bytes; large-resolution fit and throughput remain Phase6 gates.

Validation (zero test failures/errors in the final gates):

|Gate|Passed|Skipped|Wall seconds|
|---|---:|---:|---:|
|Full core radiusSearch, CPU FP32/int32 offsets|487|521|5.10|
|Full core radiusSearch, CPU FP32/int64 offsets|488|520|5.01|
|New chunk search/lifecycle, CPU FP64/int64 offsets|186|187|7.19|
|CUDA chunk matrix, int32 offsets|173|1|21.91|
|CUDA chunk matrix, int64 offsets|173|1|40.80|
|Final fresh-field/wrapper extension, CPU+CUDA/int32|28|0|4.12|
|Final fresh-field/wrapper extension, CPU+CUDA/int64|28|0|3.84|
|Frontend CSR/native-work/lanes/driving compatibility, CPU/int64|73|2|383.86|

The CUDA matrix plus final extension cover 186 distinct CUDA cases per offset
setting (the extension repeats one wrapper control). CPU-only suites skip CUDA
controls; the CUDA matrix also selects the CPU variant of the CUDA-stream test,
which intentionally skips. All six support modes, 2D/3D, periodic seams and
out-of-domain images, unequal populations, empty rows/populations/owners,
headroom, global row identity, input freezing, signature disagreement, source
invalidity, capacity changes, fresh physical filter fields and nondefault CUDA
streams are exercised. Existing flat operator/gradient contexts remain covered
by the frontend compatibility controls. The new kernel is integer I fill; no
floating numerical kernel or model formula changed. Chunked AD is rejected.

These are bounded automated fixtures, without videos; no new flow campaign or
solver throughput benchmark was run. CUDA jobs were serial and the GPU was shared
with an existing workload; wall times include JIT/cache/test overhead and are
not comparable solver performance measurements.

Runtime import verification passes all 561 frontend modules against the paired
sources. The combined/static scanner still crashes at
`warpSPH.modules.incompressible.wp_alpha.computeAlpha` (`find_spec` on a module
without __path__). The identical crash reproduces with untouched core `04e126e`;
it is an inherited validation limitation, not a passing static check or a fix in
this change. Its baseline traceback and the initial fixture-construction failure
are retained with the final evidence.

Reproduce with explicit paired source PYTHONPATH, the conda Python above,
`OMP_NUM_THREADS=2`, `PYTHONDONTWRITEBYTECODE=1`, a writable `WARP_CACHE_PATH`,
`NUMBA_CACHE_DIR` and `MPLCONFIGDIR`. Select offsets/precision before imports;
set `CUDA_VISIBLE_DEVICES=''` for CPU-only gates. Main core commands:

```bash
python -m pytest -q -p no:cacheprovider tests/radiusSearch
python -m pytest -q -p no:cacheprovider tests/radiusSearch/test_chunkedSearch.py -k cuda
python -m pytest -q -p no:cacheprovider tests/radiusSearch/test_chunkedSearch.py \
  -k 'fresh_particle_fields or particle_wrapper'
```

Frontend gates: `test_compressibleCSR.py`, `test_crkSharedWork.py`,
`test_compSharedWork.py`, `test_compSPHPairWork.py`, `test_neighborLanes.py` and
`test_drivenTurbulence3D.py`, plus `scripts/check_imports.py --runtime`.
Evidence/provenance/commands: frontend `.tmp/chunked-search-20261030/`.
Shared editable installs and the physics worktrees remain unchanged by this work.
Next: shared operator dispatch with full query/reference mapping, then simple
eager solver parity and native compatible W with pinned pair identity/stages.

## Phase 3 — shared operators and native compatible work (both repos)

- [ ] Thread row ownership through `adjacencyData`, range helpers, canonical
  argument extraction/state bundles and prepared launchers. Handle grid sortIndex
  as a particle permutation, not an edge list. Map all query/reference fields.
- [ ] Bring up shared support/density/corrections/gradients and a simpler eager
  gather scheme (Monaghan or GSPH), then CRKSPH/CompSPH. Freeze lanes/tableau.
- [ ] Allocate/accumulate W per owned chunk across stages. First complete force
  outputs for all rows; then form stage half velocities and run work sweeps.
  Final energy reduction uses local W and global final half velocities.
- [ ] Pin pair identity/ordering and ownership during W accumulation. At changed
  lists or repartition, preserve existing replay semantics; explicitly migrate
  matching work if needed. Never silently discard work or use stale fields.
- [ ] Force mid-stage list changes, multiple integrator weights and chunk-layout
  changes; compare full live W and all stored fields against the flat baseline.
  Require bitwise per-row parity where arithmetic order is unchanged. Keep earlier
  recomputation/lane roundoff failures visible; do not relax screens to force a pass.

Exit: native fresh-stage correctness, energy/source budgets, positivity, no extra
or missing RHS, and independently measured scratch/retained-memory changes.

## Phase 4 — replicated-state multi-GPU runner (two GPUs, then four)

- [ ] Implement one rank per GPU with a transport abstraction selected after
  topology checks; a single-rank reference and multiple logical owners on one
  GPU can check sequencing but do not qualify physical distributed execution.
- [ ] All-gather counts, replicated scan, deterministic assignment, local collect
  and resident reuse. Replicate reference fields/grid; build identical grids or
  synchronize their particle ordering. Exchange owned particle results, not pairs.
- [ ] Document the dependency graph for each scheme: support/density -> corrections
  and gradients -> forces -> half velocities -> W -> accepted update. Add required
  collectives before every cross-owner read. Exchange source/limiter fields too.
- [ ] Global min timestep, global rebuild/overflow decisions, positive-state and
  failure flags, conservation/source reductions. Count each particle once; do not
  reduce duplicated replicas as if they were independent contributions. Keep the
  OU clock/coefficients/seed history synchronized and mean-force removal global.
- [ ] Test cross-GPU neighbours, uneven/empty owners, asynchronous-stream ordering,
  capacity changes and deliberate mismatched generations. A failed rank must
  terminate the job with diagnostics rather than deadlock its peers.
- [ ] Compare physical two-/four-GPU short steps against the same single-GPU
  configuration. Declare collective floating-point tolerances in advance; log
  partition/generation, bytes transferred and per-pass wait/compute times.

Exit: no full J/W on any rank, correct synchronization and failure handling,
per-device memory measured, physical conservation/positivity gates passed.

## Phase 5 — remaining schemes and compatibility

- [ ] Inutsuka list variants and MFM/MFV geometry/kappa/closure workspaces, including
  the independent closure reserve and iterative matvec/vector exchanges/global
  reductions. Do not claim meshless support from flux-kernel parity alone.
- [ ] Checkpoint/restart across chunk budgets and GPU counts at accepted-step
  boundaries: canonical particle/forcing state, reconstructed local adjacency,
  old COO/CSR checkpoints and int32-to-int64 offset import. Persist generation
  metadata only where needed; never create a global pair tensor for checkpointing.
- [ ] Legacy flat and all scheme families remain covered. Qualify graph cache keys,
  pointer stability, rebuild recapture, backward/JVP/HVP and prepared launches
  separately. Initially reject unsupported chunked AD/graph/wall contexts clearly
  or retain a safe explicitly selected flat backend; no silent giant allocation.
- [ ] Combine chunks with existing1/32 neighbour lanes after chunk-only parity;
  lane changes alter reduction order and are a separate comparison.

Exit: an explicit support matrix (scheme, eager/graph/AD, boundaries, lane count,
restart) and preserved old-path regressions; unsupported paths are visible.

## Phase 6 — capacity/performance and Isambard handoff

- [ ] Short fixed-step controls at32^3/64^3/96^3 before larger allocation gates;
  compare rebuild/steady time, kernels/host syncs, communication and memory against
  the integrated flat baseline. Separate allocator capture from timing runs.
- [ ] Validate real physical device accesses for a total edge count>INT_MAX spread
  across bounded chunks. Metadata-only/aliased probes do not establish this.
- [ ] Size128^3, then160^3/192^3 and ultimately256^3 from measured per-rank terms,
  leaving headroom for developed-flow neighbour growth and overlapping generations.
  Do not assume four GPUs make every live allocation exactly one quarter as large.
- [ ] On Isambard inspect actual node peer topology and measure transport/collective
  bandwidth. NVLink-C2C names a CPU-GPU link, not proof of a four-GPU NVLink domain.
  Record Arm64, CUDA/Warp/Torch, device/HBM, compiler/JIT, ranks and paired source
  pins. Qualify on GH200/Hopper rather than extrapolating local Blackwell timings.
- [ ] Require numerical/energy/positivity/source gates and a bounded memory run
  before any long trajectory. Report wall time, steady ms/step, edges/particle,
  live/reserved/process peaks, imbalance and communication fractions per rank.

Exit: reproducible node submission/run manifests and measured fit/throughput;
256^3 is a target, not a promised capacity or stationary turbulence result.

## Existing entry points and checks for pickup

Core anchors: `radiusSearch/csr.py`, `radiusSearch/compactHash/{search,wp_collectNeighbors}.py`,
`radiusSearch/verlet/{build,filter}.py`, `dataTypes/adjacency_t.py`,
`autograd/{arg_extract,prepared,operator_spec,lanes}.py`, `util/stateBundle.py`,
`mfm/{wp_closure,warpBackend}.py` (all under`src/warpSPHCore/`).
Frontend anchors: `modules/{crk,compSPH}/{accel,work}.py`,
`modules/compSPH/{pairWork,workReduction}.py`, `modules/meshless/flux.py`,
`utils/cudaGraph.py`, runner ownership and source/restart paths (under`src/warpSPH/`).

Useful existing focused suites (verify environment/device requirements first):
core`tests/radiusSearch/{test_csrOnly,test_splitCollection,test_indexDtypes,test_queryOrder,test_verletValidity}.py`;
frontend`tests/{test_compressibleCSR,test_crkSharedWork,test_compSharedWork,test_compSPHPairWork,test_neighborLanes,test_drivenTurbulence3D}.py`;
actual graph coverage`test_crkGraph.py`/`test_compSPHGraph.py` plus affected ACSPH
checkpoint tests selected by that owner. Extend these where useful; proposed new
planner/chunk/distributed tests do not exist yet.

Use fresh processes with`warpSPHCore_OFFSET_DTYPE=int64` before imports for the
wide-offset backend. Explicit source paths and provenance must be checked, not
assumed from an editable install. `scripts/profile_compressibleMemory.py` is the
current short memory driver (needs runtime/configuration/output); it and
`scripts/benchmark_crkWork.py` may need explicit chunk/transport instrumentation.
Do not use the older CRK profiler unchanged: it still inspects `paddedI.numel()`.

Every simulation follows the existing video/progress/watchdog/source/positivity
rules, with a documented exception only for tiny A/B fixtures. No long run is
needed to qualify an addressing/ownership change. Keep failed artifacts, frozen
inputs/source hashes, per-rank logs/status and reviewed videos. Respect shared
GPU workloads; contention invalidates throughput comparisons even when pass/fail
or per-process memory measurements are permitted concurrently.

## Parallel physics track and handoff completion

The current turbulence checkout remains the place for smaller single-GPU physics
runs (up to128^3 after readiness checks), with sparse full-state exports, recorded
flow and richer scalar/source diagnostics. Keep that campaign's source runtime
frozen; development of chunks or package installation must not modify it mid-run.
The physics programme's time horizon/seed choices remain independent decisions.
No turbulence run is launched by writing or picking up this infrastructure plan.

For each phase, update this checklist, the one PLANS row, paired commit IDs,
commands/configuration, passed/failed gates and the exact next action. First
reviewable implementation deliverable: Phase1 planner plus Phase2 direct
single-GPU chunk collection; first useful solver deliverable includes Phase3
CRK retained work. Full multi-GPU completion requires real multiple GPUs.
