# Whole-row neighbour chunks and multi-GPU execution

Plan index: [PLANS.md](PLANS.md). Parent physics programme: [driven compressible turbulence](docs/roadmap/driven_compressible_turbulence.md).

Status, 2026-10-30: **paired integration qualified; implementation workspace ready**.
The count/scan/ownership design is agreed in discussion. Core/main and frontend/dev
are integrated; see the [integration report](docs/chunked_integration_2026-10-30.md).
No chunked production backend, distributed runner or new simulation has been started.
This document is the canonical implementation/handoff plan for both repositories;
the core carries a short companion pointer, `CHUNKED_NEIGHBORS_MULTI_GPU_PLAN.md`.

## Start here / next action

1. The user authorized integration at ACSPH checkpoint `3d80e5e`. The paired
   merge and focused CPU/CUDA gates are complete; read the integration report.
2. Use the prepared paired workspace below. Chunked neighbours first, then
   multi-GPU execution continue sequentially in these same checkouts.
3. Imports have been verified with explicit source paths. Start Phase1 with the
   count/prefix/chunk planner and forced-small-chunk controls. Do not change
   global editable installs.
4. Keep the existing turbulence worktree available for ordinary single-GPU
   physics through128^3, subject to a short readiness/memory check. No such
   campaign is scheduled or launched by this integration.

## Prepared implementation workspace (2026-10-30)

One paired workspace, used for the full sequential effort:

|Repository|Checkout|Branch|Integrated base|
|---|---|---|---|
|Frontend|`/home/lu26029/.codex/worktrees/chunked-neighbors-multigpu/warpSPH`|`codex/chunked-neighbors-multigpu`|`b9df6a6` plus documentation follow-up|
|Core|`/home/lu26029/.codex/worktrees/chunked-neighbors-multigpu/warpSPHCore`|`codex/chunked-neighbors-multigpu`|`04e126e`|

The frontend managed worktree is attached to this task. Both imports resolve to
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

- [ ] Implement immutable generation/partition metadata and deterministic
  whole-row boundary selection from C/P, with separate counting and compute owners.
  Ownership policy/version, edge/byte budget and capacity rules are explicit.
- [ ] Test empty populations/rows, zero-work ranks, strongly uneven counts,
  exact-budget boundaries, oversized rows, capacity padding and deterministic
  replica agreement. Exercise total prefixes>INT_MAX with metadata-only fixtures,
  without allocating billions of edges. Test exact local/global reconstruction.
- [ ] Proposed configuration roles: backend flat|rowChunks, edge/byte budget,
  device assignment and existing neighbour-lane setting. Final names follow
  repository conventions; flat/default behaviour remains unchanged initially.

Exit: exhaustive row coverage once, no row split, bounded local capacity,
identical partition on all replicas. This phase is independently reviewable.

## Phase 2 — direct chunk collection and lifecycle (core, one GPU)

- [ ] Add owned-row/global-row-base inputs to count/collect without changing
  neighbour acceptance or traversal order. Count first; allocate local buffers;
  collect directly. No monolithic intermediate J or reconstructed I.
- [ ] Add a chunked adjacency container/iterator and connect validity, filters,
  capacity reuse, rebuild flags and generation tracking. Preserve the flat API
  for existing callers. Explicit diagnostics may request bounded/chunked export;
  do not hide a giant concatenation behind a compatibility property.
- [ ] Force tiny budgets on ordinary2D/3D cases. Compare every live J/count/offset
  against the existing flat oracle, including periodic seams, unequal query/ref
  populations, all support modes and both stored-I/CSR-only contexts where supported.
  Use allocation/ownership checks to prove local construction and stable reuse.

Exit: exact order/content and count/collect consistency, correct invalidation,
no full-edge allocation, visible temporary/retained memory accounting.

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

## Paired multi-GPU / particle-layout integration (2026-11-03)

User authorizes checkpointing the current repository states, then merging the
qualified multi-GPU core into core/main and frontend into dev. Qualify the
combined fixed-count layout, restart/ownership and current integrator contracts
locally before merging the core baseline into codex/native-cluster-pairs.
Accepted Isambard artifacts remain frozen; this integration starts no site jobs.

Starting pins: frontend dev 6a067df, multi-GPU frontend 2b57b59; core/main
04e126e, multi-GPU core 179df45; integrators/main cdbbc52; native backend
430aef7 plus the existing untracked operator survey to checkpoint separately.
Native source and survey artifacts are preserved; the unrelated Isambard archive
in the frontend checkout is not part of the code checkpoint.
