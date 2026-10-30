# Whole-row neighbour chunks and multi-GPU execution

Plan index: [PLANS.md](PLANS.md). Parent physics programme: [driven compressible turbulence](docs/roadmap/driven_compressible_turbulence.md).

Status, 2026-10-30: **Phase 3 bounded native CRKSPH/CompSPH compatible work qualified; Phase 4 next**.
The count/scan/ownership design is agreed in discussion. Core/main and frontend/dev
are integrated; see the [integration report](docs/chunked_integration_2026-10-30.md).
The planner and opt-in standalone single-device chunk construction/lifecycle are
qualified on CPU/CUDA. Explicit row-mapped shared operators and a bounded eager
Monaghan path pass flat parity. Native CRKSPH/CompSPH now retain compatible work
in owned chunks and preserve changed-list replay in bounded CPU/CUDA fixtures.
Distributed execution and physical multi-GPU qualification remain open.
This document is the canonical implementation/handoff plan for both repositories;
the core carries a short companion pointer, `CHUNKED_NEIGHBORS_MULTI_GPU_PLAN.md`.

## Start here / next action

1. The user authorized integration at ACSPH checkpoint `3d80e5e`. The paired
   merge and focused CPU/CUDA gates are complete; read the integration report.
2. Use the active paired workspace below. Chunked neighbours first, then
   multi-GPU execution continue sequentially in these same checkouts.
3. Phase1/2 and the bounded Phase3 native compatible-work checkpoint are qualified
   within the limits below. Next implement Phase4 process groups, distributed
   count/collect, and particle-field exchanges between dependent passes. Use
   torch.distributed/NCCL, with Gloo CPU and the isolated two-process/one-GPU NCCL
   runtime for debugging. Pair J/W must remain on their owners. Current operator
   outputs/reduction still require a complete single owner, so distributed
   scheduling must be explicit before relaxing that guard.
   There is no runner chunk-budget CLI option yet; select the Python case hook.
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
That Phase2 handoff led to the shared operators and native compatible-work
checkpoints recorded below.

## Phase 3 — shared operators and native compatible work (both repos)

- [x] Map row ownership through host contexts before canonical extraction/state
  bundles and prepared launchers. Reuse unchanged local-row adjacency/range/kernel
  ABIs. Slice all query fields; keep references and J IDs global. Grid sortIndex
  remains a particle permutation. Reject unported specs explicitly.
- [x] Bring up shared support/density/corrections/gradients and an explicit eager
  Monaghan path. Fixed lanes/tableau and full stored-field parity qualify the
  bounded checkpoint below, not every scheme/boundary/graph configuration.
- [x] Bring up native CRKSPH/CompSPH solver dispatch and compatible work. A CRK
  correction-construction pass is not a CRKSPH compatible-energy solver gate.
- [x] Allocate/accumulate W per owned chunk across stages. First complete force
  outputs for all rows; then form stage half velocities and run work sweeps.
  Final energy reduction uses local W and global final half velocities.
- [x] Pin pair identity/ordering and ownership during W accumulation. At changed
  lists or repartition, preserve existing replay semantics; explicitly migrate
  matching work if needed. Never silently discard work or use stale fields.
- [x] Force mid-stage list changes, multiple integrator weights and chunk-layout
  changes; compare full live W and all stored fields against the flat baseline.
  Require bitwise per-row parity where arithmetic order is unchanged. Keep earlier
  recomputation/lane roundoff failures visible; do not relax screens to force a pass.

Exit: native fresh-stage correctness, energy/source budgets, positivity, no extra
or missing RHS, and independently measured scratch/retained-memory changes.

### Shared-operator and Monaghan checkpoint (2026-10-30)

Core `autograd/row_dispatch.py` implements an explicit host adapter, without
changing floating-point kernels or the canonical Warp struct ABI. For each
resident chunk, positions, supports, masses, densities and kinds are sliced as
query fields. Reference particle arrays remain global. Volumes, CRK A/B/gradA/
gradB, grad-H and renormalization fields receive the same role mapping. Self
aliases resolve before slicing. A single query CRK state is permitted for
query-only consistency density; operators that read reference corrections
validate their reference population independently.

`OperatorSpec.rowChunked` opts a particle-output kernel into this contract.
`ExtraSpec.rowRole` declares QUERY, REFERENCE or REPLICATED, optionally through a
resolver following an actual enabled/disabled kernel branch. It never guesses
identity from population size. Disabled reconstruction dummies stay replicated;
active velocity Jacobians, Balsara factors and reconstructed state gradients
are mapped as particle fields. Pair-shaped outputs, unannotated extras, partial
owners, AD and deferred/captured execution are rejected. Per-query outputs are
assembled by disjoint copies into O(N) particle arrays, without cross-chunk sums
or a concatenated pair list. Local output scratch is released after each chunk.
This host loop is a correctness implementation, not a throughput qualification.

Prepared launchers hold a canonical prefix per chunk and use the same mapping.
They must be rebuilt when state tensor bindings change, as for flat prepared
prefixes. A private storage lease now invalidates old operator/prepared/filter/
Verlet consumers when an in-place rebuild actually reuses CSR/J buffers. Default
rebuilds leave old independent buffers valid. This makes buffer obsolescence
explicit; it does not implement W migration or replay.

Qualified core consumers: density, interpolation, gradient, divergence, curl,
Brookshaw Laplacian, covariance (including count output), CRK volume/moments/
consistency density and renormalization construction. Tests include self queries,
distinct equal-sized and unequal populations, 2D/3D, full correction roles,
periodic support/direction masks, empty populations, prepared extras refreshed
between calls, unsafe-reference validation and obsolete-buffer failures. Scalar
lanes are fixed for the full matrix; density/covariance also pass fixed 32-lane
chunk-versus-flat checks on a nondefault CUDA stream. This does not qualify every
Laplacian scheme, lane combination or AD path for chunks.

Frontend enables Omega/Psi0 support operators and classic Monaghan dissipation
operators. The ParticleState `buildVerletList` wrapper preserves an explicitly
seeded chunked representation/policy through subsequent validity checks and
rebuilds, rejecting flat capacity/grid-factory overrides. For an eager, wall-free
Monaghan case, select the path in its Python `buildSystem` hook:

```python
from warpSPHCore.radiusSearch import buildChunkedVerletList, RowChunkPolicy
config.compute.neighborOmitI = True
system.adjacency = buildChunkedVerletList(
    system.state, config.domain,
    policy=RowChunkPolicy(edgeBudget=1_000_000, bytesPerEdge=4),
    verletScale=config.verletScale, storeI=False)
```

No default or runner CLI selection changes. The fixtures use `store=False` and
`storeMode='trajectory'` to avoid the checkpoint serializer. Chunked native
checkpoint export raises explicitly; checkpoint/restart remains Phase5 work.
Monaghan wall closure and per-side pressure reject at entry. The latter has a
direct `i == j` check needing global query identity before it can be opted in.
Other unported solver/spec contexts remain gated.

Five two-step eager trajectory fixtures cover fixed support/RK2, Monaghan support
with grad-H/RK4, 3D Owen support/RK2, active velocity/Balsara reconstruction, and
active Riemann-state reconstruction. They compare every stored tensor exactly,
time, expected 4/8 fresh RHS calls, positivity and the existing source/energy
budget gate. Monolithic pair search is forbidden after the explicit seed. These
are tiny unit A/B fixtures: no video or long flow campaign, with progress/stall
monitoring enabled. They do not establish stationary turbulence or scaling.

Qualification, using source-pinned paired worktrees and fresh offset processes:

| Gate | Result |
| --- | --- |
| New core operators, CPU float32/int32 | 53 passed, 55 skipped |
| New core operators, CPU float32/int64 | 53 passed, 55 skipped |
| New core operators, CPU float64/int64 | 53 passed, 55 skipped |
| New core operators, CUDA float32/int32 | 54 passed, 1 skipped; CPU cases deselected |
| New core operators, CUDA float32/int64 | 54 passed, 1 skipped; CPU cases deselected |
| Monaghan CPU float32, int32/int64 | 5 trajectories passed per offset |
| Monaghan CUDA float32, int32/int64 | 5 trajectories passed per offset |
| Existing core CPU/int64 focused regressions | 834 passed, 849 skipped |
| Existing frontend CPU/int64 focused regressions | 73 passed, 2 skipped |
| Frontend runtime import checker | 561 modules passed |

The CUDA selection also matches the CPU parametrization of the test whose name
contains "cuda"; that one CPU stream-test case skips as intended. All 54 actual
CUDA parametrizations pass. CPU-only processes skip those 54 CUDA cases plus
their CPU stream-test case.
The existing static import-scanner crash from Phase2 remains recorded; runtime
imports pass. One initial import attempt omitted NUMBA_CACHE_DIR and failed on a
non-writable source cache; the corrected, source-pinned environment passes.
Development also corrected a test return-type assumption, then declared CRK
consistency density's genuinely query-only correction role; numerical tolerances
were not relaxed. The initial native checkpoint AttributeError was replaced by
the explicit unsupported export gate above.

Evidence/commands/source provenance: frontend `.tmp/chunked-operators-20261030/`.
The bounded tests run GPU suites sequentially. Timings include JIT and shared GPU
contention; no throughput, peak-memory saving, large-edge addressing or physical
multi-GPU claim is made. Single-device resident chunks retain total J storage;
this checkpoint bounds per-chunk addressing and operator scratch, not total J.
The next step at that checkpoint was chunk-owned native compatible work; it is
now qualified in the following section. Paired implementation provenance: core c0b8cf7 (frontend is the commit containing this checkpoint).

### Native chunk-owned compatible work (2026-10-30)

This checkpoint adds eager, wall-free native recomputation for CRKSPH and CompSPH,
with the original compatible-energy formula and fresh Butcher stages. It does
not change forcing, support iteration/acceptance, EOS or integrator weights.
The old flat path remains the default. Native special energy/material/AD/graph/
wall controls are gated at entry for chunks rather than allocating expanded pairs.

Core `radiusSearch/pairField.py` introduces `ChunkedPairField`: a frozen tuple of
dedicated bounded buffers associated with one adjacency object/generation and
its storage lease. It validates capacity, dtype, device, stride, disjoint storage
and policy extents. Hidden larger backing allocations and shared buffers across
chunks are rejected. There is no flat tensor view. `ExtraKind.CHUNK_TENSOR` and
`RowRole.CHUNK` bind each work input to its matching chunk; `ROW_START` supplies
global query identity for reads of replicated stage/final half velocities.
Prepared launches use the same mapping. Only explicit one-element unread ABI
scratch outputs are allowed beside query outputs; the first output must still
define the query thread count.

Force-only dispatch reuses the existing scalar/tiled force kernels with bounded
one-element pair placeholders. It completes all query-force outputs before
forming stage half velocities and starting the work sweep. Each chunk then
accumulates its own W with the same pair formula, fraction and stage weight.
Final reduction walks each local CSR row in order, reading global final half
velocities and writing its disjoint global particle result. It never assembles W
or edge-level velocity gathers on a device. Allocation/order audits in the test
fixtures check these statements against actual launches and tensor storage.

A changed adjacency identity or repartition triggers the existing `RerunRequest`:
the first attempt computes the global final half velocity, then the integrator
repeats all fresh stages and reduces each stage on its current list directly.
The replay retains no W, only a one-element unread placeholder and particle
increments. No implicit work migration or silent discard-and-accept was added.
Tests force a generation/layout change at stage 1, including a row cap of 5 -> 2.
Each two-step RK4 fixture performs the same 16 RHS calls as its flat replay;
normal RK2/RK4 controls perform exactly 4/8 RHS calls.

Policies must budget both J and W, including the scalar work extent. Example
for a native compatible scheme in a Python case build hook:

```python
from warpSPHCore import SupportScheme
from warpSPHCore.radiusSearch import RowChunkPolicy, buildChunkedVerletList
dim = system.state.positions.shape[1]
policy = RowChunkPolicy(
    edgeBudget=1_000_000,
    bytesPerEdge=4 + dim * system.state.positions.element_size(),
    maxElementsPerEdge=dim)
config.compute.neighborOmitI = True
system.adjacency = buildChunkedVerletList(
    system.state, config.domain, policy=policy,
    verletScale=config.verletScale, supportMode=SupportScheme.SuperSymmetric,
    storeI=False)
```

The four-byte ID term assumes omitted I; include another four bytes for stored I.
A J-only policy fails before work allocation. This is a single-owner selection;
partial-owner execution remains gated until the distributed exchange schedule
exists. Native checkpoint/HDF5 export also fails explicitly, before creating a
partial output file. Use the qualified `store=False, storeMode='trajectory'`
fixture configuration until Phase5 implements a chunk-local checkpoint format.

The new CUDA nondefault-stream control initially exposed an existing flat-path
dependency bug: CRK terms and flat work reductions launched outside Torch's
current stream. The first screen differed by one ULP; an isolated run could
read uninitialized values and fail positivity even on the flat control. Those
helpers now use the existing `torchWarpStream` context. Their floating kernels
are unchanged. Strict 32-lane chunk/flat parity then passes for both schemes.
Delayed-producer tests independently exercise terms, flat/chunk work reduction
and reference entropy work; repaired flat helpers also capture/replay correctly.
Chunked capture remains explicitly rejected. Failed initial logs/XML are kept;
no numerical tolerance was relaxed to remove the failure.

Qualification (fresh offset/precision processes, paired source paths):

| Gate | Result |
| --- | --- |
| Core pair-field/row-operator matrix, CPU float32 int32/int64 | 59 passed per offset, 61 expected skips |
| Core pair-field/row-operator matrix, CPU float64/int64 | 59 passed, 61 expected skips |
| Core pair-field/row-operator matrix, CUDA float32 int32/int64 | 60 passed per offset, 1 CPU stream-test skip |
| Native scalar-lane trajectory/work/replay cases, CPU int32/int64 and FP64/int64 | 8 cases pass per configuration |
| Native trajectory/work/replay plus fixed 32 lanes/nondefault stream, CUDA int32/int64 | 10 cases pass per offset |
| Padding/empty/generation/policy/unsupported-context/export/stream contracts | 15 CPU checks plus 16 CUDA checks per offset; capture/export included |
| Existing core CPU/int64 focused regressions | 834 passed, 849 expected skips |
| Existing frontend work/pair-work/Monaghan/graph-selection CPU controls | 57 passed, 18 CUDA skips |
| Existing native flat work CUDA controls | 33 passed |
| Runtime frontend imports | 562 modules pass |

Every live W vector, its J identity/order, stage force/heating/increment and all
stored final tensor fields match the native flat recomputation control bitwise.
The native recomputation/expanded-control regressions retain their existing
tolerances; they were not changed. All bounded accepted steps pass positivity
and the existing source/energy-budget gate. The largest flow fixture is 216
particles; these are two-step unit A/Bs with progress/stall monitoring, with the
documented video exception. No new developed-flow campaign was launched.

Measured storage example, 64-particle 2D RK4: both schemes retain 36,040 bytes of
flat W capacity versus 36,048 bytes across chunks (padding), with identical live
work. Largest force-kernel output storage is 784 bytes flat versus 76 bytes per
chunk at the debug row cap of 5. The chunk adapter additionally assembles the
full O(N) particle output, so this is **not a peak-memory saving**. Single-device
resident chunks keep total J/W; distribution will reduce ownership per GPU.
The allocation audit reports actual tensor storage separately from allocator/
native/JIT/process memory. Tiny CPU RK4 examples took roughly 24 -> 112 ms/step
(CRK) and 19 -> 82 ms/step (Comp); host-loop overhead is visible and expected at
this debug chunk size. Logs include all wall/step times. JIT and shared workload
contention prevent throughput conclusions; no performance default is promoted.

Commands, test XML/logs, allocation/timing audits, failure records and paired
source fingerprints: frontend `.tmp/chunked-compatible-work-20261030/`.
The existing static-scanner issue remains recorded in Phase2. No physical
>INT_MAX edge-buffer access, large-memory fit or physical multi-GPU claim follows
from these fixtures. Next action is the Phase4 replicated-state process backend.
Implementation provenance: core 85b5ef9; frontend is the commit containing this checkpoint.

## Phase 4 — replicated-state multi-GPU runner (two GPUs, then four)

- [ ] Implement `torch.distributed` (c10d) process groups with NCCL for production
  CUDA collectives, one process/rank per GPU. Use torchrun or the Isambard Slurm
  launch environment; inspect topology and site network configuration before
  transport tuning. Keep a Gloo CPU-tensor debug transport behind the same
  communication interface. Single-rank/logical-owner fixtures do not qualify
  physical distributed execution.
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

### Transport decision and local debugging (2026-10-30)

User preference: NCCL is the preferred communication library on Isambard and
PyTorch provides the intended interaction layer. Production execution will use
`torch.distributed`/c10d with NCCL, one process per GPU. Warp continues to compute
through Torch-backed device buffers, with explicit stream/event ordering before
communication and dependent kernels. Collectives exchange counts and owned
particle outputs plus global timestep/rebuild/failure/source reductions; pair
J/W stay on their owners. No distributed runner has been implemented yet.

Shared base package inspection: Torch 2.13.0+cu130, NCCL 2.29.7, both NCCL
and Gloo backends available. Torch pins the NCCL wheel exactly to 2.29.7; that
base does not support sharing one GPU among multiple ranks of the communicator.
The user subsequently authorized an NCCL upgrade if useful for debugging.
An isolated NCCL 2.32.3 runtime overlay is now installed and smoke-qualified below;
the shared conda environment and existing physics jobs retain their original
libraries. Production retains one process per GPU.

NCCL **2.30+** documents an experimental exception:
`NCCL_MULTI_RANK_GPU_ENABLE=1`. Each separate process must explicitly select the
shared device (e.g. cuda:0), instead of selecting LOCAL_RANK as a device ordinal.
NVIDIA expects up to two ranks per GPU with the default resource configuration;
more/larger ranks may require lower NCCL_MAX_CTAS and can exhaust resources or
hang. NVLS is incompatible with same-GPU ranks; the default auto setting disables
it, while forcing NVLS on fails initialization. Separate rank threads in one
process sharing that GPU are not supported by this feature. The isolated
2.32.3 runtime override below passes the bounded PyTorch/Warp smoke; this is not
a general compatibility qualification of every PyTorch/NCCL operation. No shared
package installation is changed.

Local debug choices now include the tested two-process/one-GPU NCCL 2.32.3
runtime and a future Gloo CPU transport for device-independent process/failure
tests. A single-rank NCCL smoke checks API plumbing only. Real two-/four-GPU NCCL
qualification is still required for device separation, transport, stream ordering
and performance on the target topology.

Sources: [PyTorch distributed process groups](https://docs.pytorch.org/docs/stable/distributed.html),
[NVIDIA same-GPU rank support and limits](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-multi-rank-gpu-enable).
The specific experimental switch qualifies the otherwise general prohibition in
the communicator/PyTorch documentation; do not assume it exists in older stacks.

### Isolated NCCL upgrade and shared-GPU smoke (2026-10-30)

Installed the official `nvidia-nccl-cu13==2.32.3` wheel with `--no-deps --target`
into frontend `.tmp/nccl-shared-gpu-20261030/overlay`. No shared Python or editable
package was replaced. `with-nccl-2.32.sh` prepends that overlay to PYTHONPATH and
preloads its `libnccl.so.2` for the launched Python process and its children.
Both settings matter: PyTorch eagerly preloads native libraries by searching
sys.path, so preloading alone also mapped the original NCCL. The final wrapper
verifies/uses a single mapped NCCL library from the overlay. It selects the native
runtime only; continue to set the paired frontend/core source paths explicitly
in PYTHONPATH when running SPH code.

`torch.cuda.nccl.version()` still reports the **build-time** 2.29.7 for this Torch
binary. The real loaded runtime was independently verified with `ncclGetVersion`
(23203), /proc/self/maps (one overlay library), and NCCL's communicator-init log
(2.32.3). A fresh process without the wrapper still loads only base NCCL 2.29.7.
The shared package dependency pin remains satisfied; this is an explicitly
selected experimental native-runtime override, not a replacement Torch build.

Ran torchrun with two separate OS processes, both explicitly selecting cuda:0.
Distinct PIDs and the same physical GPU UUID are recorded. Settings were
`NCCL_MULTI_RANK_GPU_ENABLE=1`, `NCCL_MIN_CTAS=1`, `NCCL_MAX_CTAS=2`, and
`NCCL_NVLS_ENABLE=0`, with a 20-second process-group timeout and 60-second outer
timeout. No MPS configuration or system library changes were made.

Both ranks passed all seven check groups:

- Uneven int32 count all-gather and replicated int64 prefix reconstruction.
- Padded owned particle-field all-gather.
- Global MIN timestep, MAX rebuild flag, SUM over unique owned particles.
- Broadcast.
- Int64 reduce-scatter.
- Batched point-to-point send/receive.
- Warp producer -> asynchronous NCCL all-reduce -> Warp consumer on a nondefault
  Torch CUDA stream, five exact repetitions.

Per-rank probe time was about 1.38 seconds excluding Python/torchrun startup;
this is a correctness fixture, not a benchmark. Torch allocator counters in the
reports exclude NCCL/native/driver allocations and are not process memory peaks.
Local optional IB/RMA plugin warnings fell back successfully; this run does not
test Isambard networking. No SPH solver or new flow campaign was run. The Phase4
runner/collective scheduling and physical multi-GPU gates remain unimplemented.

Re-run using the executable frontend
`.tmp/nccl-shared-gpu-20261030/run-nccl-shared-gpu-smoke.sh`.
The same directory contains the wrapper, smoke/verifier/summary scripts,
smoke.log, per-rank JSON, base/overlay provenance and summary.json with a library
SHA256. This runtime is now available for the subsequent distributed backend work.
The package source is [NVIDIA NCCL CUDA13 on PyPI](https://pypi.org/project/nvidia-nccl-cu13/2.32.3/).

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
