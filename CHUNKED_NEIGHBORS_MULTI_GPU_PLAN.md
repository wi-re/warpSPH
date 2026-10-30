# Whole-row neighbour chunks and multi-GPU execution

Plan index: [PLANS.md](PLANS.md). Parent physics programme: [driven compressible turbulence](docs/roadmap/driven_compressible_turbulence.md).

Status, 2026-10-30: **Phase 4 bounded process proof of concept qualified; local L1 restart and L2 launcher qualified; L3 control/hash and bounded particle packing qualified locally; L4 filtering/redistribution qualified locally; L5 bounded stress qualified; L6 compatibility active**.
The count/scan/ownership design is agreed in discussion. Core/main and frontend/dev
are integrated; see the [integration report](docs/chunked_integration_2026-10-30.md).
The planner and opt-in standalone single-device chunk construction/lifecycle are
qualified on CPU/CUDA. Explicit row-mapped shared operators and a bounded eager
Monaghan path pass flat parity. Native CRKSPH/CompSPH now retain compatible work
in owned chunks and preserve changed-list replay in bounded CPU/CUDA fixtures.
The bounded replicated-state Gloo/NCCL process path is now qualified locally;
physical multi-GPU, Isambard and capacity/performance gates remain open.
This document is the canonical implementation/handoff plan for both repositories;
the core carries a short companion pointer, `CHUNKED_NEIGHBORS_MULTI_GPU_PLAN.md`.

### Static import scanner issue resolved (2026-10-30)

The previously recorded wp_alpha.computeAlpha `find_spec` traceback is fixed.
Missing symbols on regular modules now produce normal diagnostics; only packages
are searched for child modules, and child-lookup errors do not abort the scan.
Five focused scanner regressions pass. The full check now imports566 modules
and checks6063 first-party imports across1314 files with no failures.
The five real stale imports exposed after fixing the exception were resolved:
two old pressure-solver notebooks are clearly labeled archival source listings
(their retired numerical APIs were not revived), and the Kidder regression calls
the accepted-boundary helper now used by finalizers. Five affected Kidder checks
also pass. Original notebook files are retained in ignored L3 evidence. Do not
re-investigate the old scanner traceback; older notes below describe history.
L3 control/hash and particle-packing work is now qualified locally. Evidence: `.tmp/l3-communication-20261030/` and `.tmp/l3-payload-20261030/`.

## Start here / next action

1. The user authorized integration at ACSPH checkpoint `3d80e5e`. The paired
   merge and focused CPU/CUDA gates are complete; read the integration report.
2. Use the active paired workspace below. Chunked neighbours first, then
   multi-GPU execution continue sequentially in these same checkouts.
3. Phases1/2, bounded eager Phase3 and the Phase4 process proof of concept are
   qualified within the limits below. Use an explicit RowChunkProcessGroup with
   Gloo CPU or NCCL CUDA tensors; the isolated two-process/one-GPU runtime is
   available. J/W stay owned, with particle exchanges between dependent passes.
   Continue with L6 compatibility expansion and the support matrix below;
   physical devices/topology/Isambard and capacity/performance remain open.
   Use `scripts/run_distributed.py` for the explicit experimental distributed
   launcher; ordinary runner setup still uses the Python neighborhood hook.
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

User scope (2026-10-30): start the process proof of concept with two separate
ranks sharing the existing GPU using isolated NCCL 2.32.3. ACSPH continues to
use the device; keep fixtures bounded, qualify correctness/stability and treat
all timings as diagnostic. Do not wait for an idle device or infer performance,
physical interconnect behavior or multi-device memory fit from these runs.

- [x] Explicit c10d process-group binding with NCCL CUDA and Gloo CPU under the
  same interface; torchrun process correctness/failure tests. Production retains
  one process per GPU; shared-GPU NCCL is an experimental debug exception.
- [x] Disjoint local counts -> replicated int32 counts / int64 scan -> identical
  plan/signature -> allocate/collect owned chunks only; coordinated resident
  reuse/rebuild and reference-grid ordering agreement.
- [x] Eager Monaghan/native CRKSPH/CompSPH dependent particle-field exchanges,
  owned compatible W, global half velocities and changed-generation replay.
- [x] Global timestep/rebuild/positive/failure and unique-owner source/work
  reductions; exact replicated OU clock/history/mean-removal contract.
- [x] Uneven/empty owners, unequal populations, nondefault streams, local capacity
  changes, deliberate mismatches, two-/three-process CPU tests and two real
  shared-GPU NCCL ranks. Fatal rank termination is tested with torchrun/Gloo.
- [ ] Physical two-/four-GPU execution, actual cross-device neighbours and memory,
  topology/Isambard Slurm bootstrap/network configuration, device-fault failure
  behavior, conservation/positivity and short-step reference comparisons.
- [ ] Large physical edge storage/capacity fit and target-network performance;
  declare physical collective tolerances and measure wait/compute/device memory
  on the target topology without shared-workload timing assumptions.

Exit: no full J/W on any rank, correct synchronization and failure handling,
per-device memory measured, physical conservation/positivity gates passed.

### Transport decision and local debugging (2026-10-30)

User preference: NCCL is the preferred communication library on Isambard and
PyTorch provides the intended interaction layer. Production execution will use
`torch.distributed`/c10d with NCCL, one process per GPU. Warp continues to compute
through Torch-backed device buffers, with explicit stream/event ordering before
communication and dependent kernels. Collectives exchange counts and owned
particle outputs plus global timestep/rebuild/failure/source reductions; pair
J/W stay on their owners. The bounded eager process checkpoint below is now
implemented; physical multi-GPU/site qualification remains open.

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
runtime and a qualified Gloo CPU transport for device-independent process/failure
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

### Bounded replicated-state process checkpoint (2026-10-30)

The opt-in eager process proof of concept now runs separate OS processes with
Gloo CPU or NCCL CUDA tensors. Two real NCCL ranks sharing the existing GPU
pass solver correctness and repeated rebuild/replay checks under ACSPH
contention. This qualifies the process protocol locally; the physical two-/four-
GPU and Isambard gates above remain open.

Core `warpSPHCore.distributed.RowChunkProcessGroup` binds an existing, explicitly
initialized c10d group to a device. The caller owns group initialization with a
finite timeout, device selection and destruction. Construct `communication = RowChunkProcessGroup(dist.group.WORLD, device)`
and pass `communication=communication` to `radiusSearchRowChunks` / `buildChunkedVerletList`; subsequent resident builds
infer it from the prior adjacency. Logical owners without a group remain test
fixtures and still cannot produce distributed particle results.

Each rank counts only its balanced counting interval. Disjoint broadcasts
assemble the replicated int32 counts; every rank builds its int64 prefix and
the same deterministic plan. Configuration, generation, complete plan signature
and reference-grid ordering agree **before owned pair allocation/collection**.
Only owned flat J/I and W buffers exist in the distributed path; no pair gather
or flattened compatibility buffer was added. Changing capacity/row boundaries
rebuilds directly on the new owners. Filtering an existing partial-owner list
remains explicitly unsupported; it would need a separate pair-migration design.

The existing row-dispatch ABI runs only local chunks, then broadcasts disjoint
particle-output intervals to every replica before returning. Query/reference
roles, canonical corrections, output layouts, scalar controls and operation
sequence are explicit; reference inputs are hashed exactly in this debug path.
Prepared launches follow the same protocol. Unread one-element force scratch
and chunk-local work buffers never enter particle collectives.

The qualified eager dependency schedule is:

1. Support/density outputs are exchanged before the next support iteration or
   correction pass. CRK volume -> moments -> density, gradients/limiters and
   optional grad-H fields therefore have complete reference inputs. Pointwise
   correction algebra runs on complete replicas and is checked before use.
2. Force/heating outputs are exchanged before either global half-velocity reads
   or the compatible-work sweep. Native CRKSPH/CompSPH accumulate W only on their
   owned chunks, in the existing row/edge/stage order.
3. Final half velocities are complete replicas. Each owner reduces its W directly
   to particle energy increments, which are exchanged before accepted updates.
   A midstage generation change coordinates a rebuild and preserves the existing
   full fresh-stage replay, including repartitioning.
4. Global MIN selects the timestep, MAX coordinates rebuild/invalid-state flags,
   and FP64 SUM counts uniquely owned source/work contributions once. Replicated
   FourierOU fields retain the original global mean-removal summation order;
   their clock, coefficients, generator and resulting fields must agree exactly.
   No duplicated full-replica SUM is used. Distributed body sources currently
   qualify `ConservedSources` with `FourierOU`; custom sources remain gated.

The ordinary eager runner recognizes an explicitly bound adjacency. It validates
configuration/initial replicas, picks a global timestep, and checks positive,
finite, equal accepted states and time. An optional `neighborhoodFactory(state,
config)` in the compressible sampler and driven case's `buildSPHSystem` lets the
initial density/support setup use owned chunks too. The process solver fixture
forbids monolithic search from before that setup. There is no default/CLI change,
automatic process-group initialization, output/checkpoint support or promotion
of unqualified AD/graph/material/wall/scheme contexts.

Control handshakes propagate local exceptions and detect differing sequence,
operation, layout, generation or replicated reference data before payload
collectives. Hard process death/device faults rely on c10d's finite timeout and
torchrun peer termination. Real two-/three-process CPU tests include fatal rank
termination; two-rank NCCL tests cover eight recoverable contract failures.

The longer run found a real time-precision bug in the first process adapter:
converting a float32 tensor dt to Python float changed accumulated time after
three steps (0.0005999999848427251 versus 0.0005999999702908099), with the same
24 RHS evaluations. The global MIN now preserves the original tensor dtype and
scalar representation. The three-step regression, longer run and forcing-clock
crossings retain exact time/stage/particle/W/history checks; no tolerance was
relaxed. Failed longer-run logs and the short CPU reproduction are retained.

An existing source-replay regression still patched only legacy `accumulateStage`
even though native recomputation already bypassed it before this work. Its hook
now spoils both storage paths, so the default actually performs the requested
replay. Both 20-step source controls and all 57 focused existing native/source
checks pass; this repair changes the test injection, not solver arithmetic.

Qualified controls and evidence:

|Control|Result|
|---|---|
|Separate-process CPU regression drivers, FP32/int32 and FP64/int64|5 pass each: two-/three-rank core, fatal peer death, two-/three-step solver comparisons|
|Existing chunk/planner/operator/native/Monaghan CPU controls, each local offset width|332 pass, 251 expected CUDA skips per width|
|Extended eager solver matrix|8 fixtures / 68 fresh RHS per rank, with 2D/3D, RK2/RK4, adaptive supports, source forcing and forced replay; Gloo FP32/FP64 and shared-GPU NCCL offset controls recorded|
|NCCL 32-lane controls|Native work/replay remains bitwise against the corresponding single-process lane configuration|
|Longer NCCL control, int64 FP32|16 accepted steps and 128 RHS per native scheme per rank; 16 forced rebuilds per scheme; 15 OU knots at interval 0.00025|
|Existing native/source CPU compatibility|57 pass, including repaired source replay injection|
|Runtime imports|563 frontend modules pass; inherited static scanner `wp_alpha.computeAlpha` failure remains unchanged|

Particle fields, live owned W/J order, fresh stage rates, replay increments,
simulated time and complete OU knot/generator histories match the serial oracle
bitwise. Only regrouped FP64 global source-budget sums use the predeclared
`rtol=atol=2e-12`. Accepted positivity and the existing relative energy-budget
gate of 2e-6 pass. Longer-run residuals were approximately 5.04e-9 (CRKSPH) and
3.51e-9 (CompSPH), with both ranks identical. The largest fixture is 216 particles;
videos are omitted for these tiny A/B loops, with progress/stall monitoring kept.

For all 16 accepted samples, per-rank Torch allocated bytes were constant within
each scheme (about 64.1 MiB). These counters exclude native/driver allocations
and are not process peaks or large-memory-fit evidence. Logical J/W ownership
is recorded separately: in the 64-particle initial CRK fixture, rank 0 owns 35
rows and 9,856 J-capacity bytes / 19,712 W-capacity bytes; the other rank owns the
remaining rows. Aggregate same-GPU memory still includes every owner's pairs
plus extra process/communication resources. No capacity scaling claim follows.

Telemetry includes cumulative logical particle payload/control bytes, generation,
partition and a bounded last-512-event ring of host local/enqueue and control/
collective wall times. Exact replica hashes add CPU transfers and synchronization;
the debug broadcasts prioritize correctness and bounded particle scratch. These
times are diagnostic under ACSPH contention, not GPU compute/transport benchmarks.

Maintained regressions live in core `tests/distributed/` and frontend
`tests/{test_distributed.py,distributed/_solverWorker.py}`. Tiny flat references
are explicit test oracles, released before the distributed solver phase; only
small CPU oracle snapshots remain. Logs, per-rank manifests, source fingerprints,
runtime-library provenance, failure records and reusable launch commands are in
frontend `.tmp/distributed-poc-20261030/`. This includes the isolated NCCL 2.32.3
wrapper used without changing the shared environment or ACSPH sources.

Implementation provenance: core `44bccb9`; frontend is the commit containing this checkpoint.

Next: physical multi-GPU/device/topology and Isambard launch/network qualification,
then larger storage/capacity and performance gates. Distributed filtering,
checkpoint/restart, other schemes, AD/graphs and walls/materials remain gated.


## Local continuation sequence (user-approved, 2026-10-30)

Complete these in the existing paired workspace while physical multi-GPU access
is unavailable. Keep CPU/Gloo and bounded two-process shared-GPU NCCL controls;
ACSPH contention means timing remains diagnostic. The ordering is deliberate.

1. [x] **L1 — accepted-step checkpoint/restart (bounded native/legacy v1 scope qualified).** Save canonical particle,
   exact time/timestep, forcing/generator/budget state and sufficient frozen search
   history to reconstruct owned J without exporting global J/W. Qualify changed
   chunk budgets and 1/2/3 CPU ranks, plus 1/2 NCCL ranks on one GPU. Compare
   uninterrupted versus restarted fresh stages, physical fields, pair order and
   forcing history; cover interrupted/corrupt writes and physical-policy mismatch.
   The versioned native recipe and changed int32-to-int64 local offsets now pass.
   Version-1 COO/CSR import and ordering/offset conversion now pass within the
   qualified scope below; other legacy/search policies and physical devices remain gated.
2. [x] **L2 — local launcher/output (site validation open).** Explicit rank/device/chunk configuration, rank-zero
   diagnostics/checkpoint output, source-pinned run manifests, meaningful failure
   reporting and locally checked Isambard submission templates. Final site launch
   validation still requires Isambard. Keep storage and solver code independent
   of the communicator. NCCL is implemented; optional MPI should require adapter/
   capability/failure tests rather than SPH kernel changes. The current Torch
   build has no MPI backend. User decision (2026-10-30): defer MPI-enabled and
   ROCm-enabled Torch builds; continue the local sequence on the existing stack.
   Distinguish MPI rank launching with NCCL payloads
   from an MPI data backend; do not replace/build shared packages for this task.
3. [x] **L3 — communication overhead (bounded local scope qualified).** Separate exhaustive replica hashing from
   normal validation, combine exchanges where dependencies allow, reuse metadata/
   particle scratch and reduce unnecessary host waits. Prove correctness, buffer
   lifetimes and fewer operations/allocations locally; defer throughput claims.
   Cached controls/metadata, explicit exact/contract validation, compatible output
   packing and one bounded particle buffer now pass the qualification below.
4. [x] **L4 — distributed filtering/redistribution (bounded local scope qualified).** Filter resident partial-owner
   lists and handle changed boundaries without a global pair intermediate. Preserve
   pair identity during accumulation or coordinate the existing fresh-stage replay.
5. [x] **L5 — stronger stability/failure controls (bounded local scope qualified).** Randomized partitions, skewed
   counts, empty owners, repeated capacity changes, longer forcing histories,
   failures during solver/IO phases and bounded memory checks; use more CPU ranks.
6. [ ] **L6 — compatibility expansion.** Explicit support matrix for other schemes,
   boundaries/materials, graphs and differentiation; qualify each as a separate
   change with existing native CRKSPH/CompSPH controls retained.
7. [ ] **L7 — actual wide-edge device access (when memory is free).** A total edge
   count above INT_MAX across bounded chunks is testable on one physical GPU.
   Require a memory/headroom check and actual device accesses; metadata/aliases
   do not qualify it. Do not compete with ACSPH for a large allocation now.

Physical inter-device transport/topology, aggregate multi-GPU capacity and
scaling remain separate gates. This sequence does not claim to complete them.

### Accepted-step native restart checkpoint (2026-10-30)

The first L1 slice now implements an explicit native checkpoint format for eager
driven Monaghan/CRKSPH/CompSPH with resident row chunks and ConservedSources /
FourierOU. The same API works with no process group, Gloo CPU, or NCCL CUDA.
Legacy flat-file import, automatic output cadence and physical multi-GPU restart
remain separate gates. No public defaults, CLI or shared packages changed.

Every rank calls `warpSPH.io.chunkCheckpoint.writeChunkedCheckpoint(path,
result.ctx, result.state)` after an accepted runner step. A weak identity/time/
completed-step token rejects stage/intermediate saves without retaining the
previous system or W. Rank zero writes a versioned HDF5 file to a temporary sibling,
checksums all datasets/metadata, closes/fsyncs it and atomically publishes it.
Existing files reject by default; `overwrite=True` explicitly replaces atomically.
Recoverable IO errors propagate through the existing local-phase protocol. A hard
process kill may leave an unpublished temporary file; no fault-tolerant filesystem
or physical-device failure claim follows from the injected exception tests.

The file stores canonical particle fields, accepted step count, exact time and next
dt representation/dtype, initial mass/energy reference, source energy budget, and
OU parameters/knots/generator. It also stores a versioned deterministic search
recipe: frozen query/reference positions and scaled supports, domain, int32 counts,
generation and ordering metadata. Frozen inputs matter: reconstructing only from
the current particle state could change a still-valid resident list and replay
semantics. Query/reference aliases are preserved. No J, I, W, saved local offset
array, communicator or device ordinal is serialized.

`radiusSearch.rowChunkCheckpointState` produces an independent inert CPU recipe;
`restoreRowChunkCheckpointState` validates it and recounts the frozen geometry.
It derives the target plan from the saved counts and a newly selected policy/
owner count, agrees on the complete plan before pair allocation, then collects
only locally owned J/I. Prefixes are reconstructed in int64; local offsets use
the target process's configured int32/int64 type. Particle precision must match.
The traversal algorithm is explicitly versioned; arbitrary older search ordering
is not assumed compatible. Restore rejects checksum, algorithm, precision,
shape/count, budget and recounted-plan mismatches.

For a continuation, seed the target runner with the existing explicit chunked
`neighborhoodFactory` and set `simulation['resumeFrom'] = path`. The target factory
selects the new policy and process group; the runner loads this format before
distributed startup, restores source/time/dt/context, discards transient pair work,
and rebuilds local adjacency. This bounded path still initializes a seed list
before restore, so seed and restored storage overlap until installation; remove
that startup duplication before large-capacity qualification. Physical/numerical configuration must match; chunk
budget, rank count and offset width may differ. The old flat checkpoint branch
and automatic-output guards remain in place. There is no monolithic fallback.

Qualification (tiny 64-particle, four-step uninterrupted versus two-step save +
two-step continuation controls; video omitted for this bounded A/B matrix,
progress/stall/velocity controls enabled):

|Control|Result|
|---|---|
|Separate Gloo jobs, CPU FP32|3 scheme drivers pass: CRKSPH 2→3 ranks, CompSPH 1→2, Monaghan 3→1; all change maxRows 3→1 and int32→int64 offsets|
|Separate Gloo jobs, CPU FP64|Same 3 drivers pass plus 14 core recipe cases; 22 independent rank reports / 168 RHS evaluations, with 14 expected CUDA skips|
|Shared-GPU NCCL 2.32.3, FP32|10 serial jobs / 14 rank reports / 136 RHS evaluations pass: CRKSPH 2→1 and 2→2, CompSPH 1→2, Monaghan 2→1|
|Empty owners|CRKSPH large-budget restore creates one chunk with two empty Gloo owners or one empty NCCL owner, including empty W storage|
|Failure/publication controls|Mid-stage save, existing-file overwrite, injected write failure/cleanup/retry, corrupt state checksum and changed forcing configuration reject; live peers remain coordinated|
|Core recipe CPU FP32/int32, CPU FP64/int64|14 pass each, 14 expected CUDA skips; self/unequal/empty populations, alias/order/reuse and malformed/re-signed recipes before pair allocation|
|Core recipe CUDA FP32/int32 and int64|14 pass each; non-CPU cases selected|
|Existing search/operators/native-work CPU FP32/int64|344 pass, 263 expected skips; includes the initial 12 CPU recipe cases, before two additional recipe guards|
|Legacy native checkpoint regressions|7 pass, preserving the old runner branch|
|Runtime imports|564 modules pass; static scanner still aborts on the inherited wp_alpha.computeAlpha `find_spec` issue documented in Phase2|

Continuation compares all particle tensors, exact accepted time/next dt, each
fresh RHS dvdt/dudt, per-row J and compatible W hashes, complete OU history and
generator. Only the two-component globally reduced source budget uses the existing
2e-12 tolerance; other checks are exact. Test oracles contain particle fields and
per-row hashes, never pair arrays. All three schemes retain positive finite states.
The FP32 process-driver aggregate is 100.25 seconds and the FP64 driver/recipe
aggregate is 100.03 seconds, including launches; GPU logs
retain per-step timings. Hashing, host waits, process startup and ACSPH contention
dominate these fixtures, so they establish correctness rather than throughput.

Evidence, XML/logs, files/oracles, failure history, reproduction scripts and paired
source fingerprints are in frontend `.tmp/chunked-restart-20261030/`. The first
standalone recipe test failed because its fixture had not called `wp.init()`;
the corrected fixture and the raw failure are retained. No floating kernel or
numerical formula changed. Next L1 work is legacy COO/CSR import with explicit
ordering/offset conversion; L2–L7 and physical multi-GPU/site/capacity gates remain
open. This checkpoint does not qualify cross-device-type restarts, AD/graphs,
walls/materials or schemes beyond the stated native driven scope.
Implementation provenance: core `e740b76`; frontend is the commit containing this checkpoint.

### L1 legacy import and L2 local launcher checkpoint (2026-10-30)

User decision: MPI-enabled and ROCm-enabled Torch builds are deferred. The local
sequence continues on the existing CPU/Gloo and isolated NCCL 2.32.3 stack; no
shared packages or ACSPH sources/installs changed.

L1 now imports version-1 driven SPH COO and CSR files into owned row chunks.
`rowChunkCheckpointRecipe` encodes explicit frozen inputs without accepting pair
arrays. The frontend legacy adapter reads O(N) snapshots/counts/offsets, validates
the offset scan in int64, reconstructs under the target policy/group, and compares
each owned J slice with the file before installing the state. Optional COO row IDs
and padding sentinels are checked too. No global pair tensor, flat compatibility
list or pair migration is constructed. Raw whole-file fingerprints use fixed
host blocks to detect cross-rank/file changes during import; legacy files have
no original checksum, so this does not detect arbitrary finite particle-state
corruption that occurred before import. Expanded per-edge stage-state files,
missing history, unknown formats and incompatible reconstructed ordering reject.

These old files lack mode/Verlet-scale metadata. The caller must supply the
matching original physical/search configuration, as in the existing flat restore
path; target config supplies mode/scale and saved scaled supports remain frozen.
Qualification uses the default search policy in bounded 2D controls. Other legacy
policies/versions, physical multiple-device restart and broad 3D import remain
separate qualifications. The seed-list startup overlap noted above remains.

L2 adds `warpSPH.distributedRun.runDistributed(configuration, communication,
outputDirectory=..., edgeBudget=..., maxRows=..., checkpointEvery=...)` and
`scripts/run_distributed.py`. The Python API accepts an existing explicit group;
the CLI initializes/destroys c10d with a finite timeout. Ordinary runner defaults
and existing run documents remain unchanged. The adapter currently selects eager
driven Monaghan/CRKSPH/CompSPH, with owned initialization/search from the start.
Native numerical/support choices come from the run document, including the
default Owen support policy; the adapter does not force a numerical preset.

An optional appended `Case.acceptedStep` output hook runs after the case's
postStep/timestep hooks and distributed accepted-state validation. Periodic and
final native checkpoints use the existing accepted-boundary writer. Rank zero
owns diagnostics JSONL and the global manifest; ranks write separate runtime,
communication and failure reports. Output directories must be fresh and shared.
Manifests include active frontend/core/integrator Python file hashes, available
Git heads, input configuration, rank/device/PID/host placement, dtype/offset width,
Torch/Warp/CUDA and actual loaded NCCL provenance. Git heads alone do not assert
that an uncommitted working tree matches that commit; active file hashes identify
what ran. The actual entrypoint path/hash is recorded in rank runtime metadata.
Accepted-step hooks explicitly reject pipelined graph output rather than being
silently skipped. Logical adjacency storage and sampled Torch allocation are not native/
driver/process peaks. Hard rank death still relies on c10d timeouts and torchrun;
no new collectives are entered in the failure-reporting exception path.

Example local CPU launch, with paired `src` paths in PYTHONPATH and a versioned
JSON RunConfiguration that disables ordinary store/plot/video and selects eager:

```bash
python -m torch.distributed.run --standalone --nproc-per-node=3 \
  scripts/run_distributed.py --configuration case.json --backend gloo \
  --output /absolute/fresh/shared-output --edge-budget 1048576 \
  --max-rows 512 --checkpoint-every 100 --offset-dtype int64
```

Production NCCL maps LOCAL_RANK to a distinct visible GPU. The explicit
`--shared-gpu` debug exception selects cuda:0 and requires NCCL >=2.30, the
experimental enable environment variable and at most two local ranks. Continue
to use the isolated overlay and low CTA/NVLS settings above for local shared-GPU
checks. Set the input document's `simulation.precision` before launch (float32 or
float64); the CLI selects it and the offset width before importing the stack.
Set `simulation.resumeFrom` to a checkpoint and use a fresh output directory to
continue with another budget/rank count. Launcher output overrides use its own
checkpoint path and rank-zero progress, keeping the legacy serializer disabled.

`scripts/slurm/chunked_driven.slurm` is a template with explicit environment
inputs for paired source paths, target Python, configuration/output, GPU count
and edge budget. It launches one torchrun parent per allocated node, quotes paths
and derives rendezvous from the Slurm node list/job ID. It passes bash syntax and
mocked remote argument-expansion checks including paths with spaces. No submission,
site module/resource/network setup or Arm64/GH200 execution has been qualified.

|Control|Result|
|---|---|
|Legacy COO/CSR CPU FP32|6 scheme/layout drivers pass; serial legacy files -> 3 owned ranks, int32/int64 source offsets; malformed scan/J/I reject|
|Legacy CPU FP64|CRKSPH both layouts and 14 recipe checks pass; 14 expected CUDA skips|
|Legacy shared-GPU NCCL FP32|6 serial jobs / 8 rank reports / 80 RHS evaluations pass; CRK CSR int32→int64 and Comp COO int64→int32, 1→2 ranks|
|Recipe CUDA after encoder refactor|14 pass per offset width|
|Native-format restart nonregression|3 scheme drivers + 14 core cases pass, 14 CUDA skips|
|Launcher CPU FP32|3 scheme lifecycle drivers pass: 1-rank uninterrupted, 2-rank save, 3-rank resume under changed budgets; exact final particles/time/dt/OU, one diagnostics row per accepted step plus initial row|
|Launcher shared-GPU NCCL FP32|9 serial jobs / 12 rank reports pass for all 3 schemes: 1-rank uninterrupted, 2-rank save, 1-rank resume; exact final particles/time/dt/OU; runtime23203 and shared UUID recorded|
|Output/failure controls|Periodic accepted checkpoints, final save, preserved existing directory and coordinated capacity failure with rank error reports and no final checkpoint pass|
|Existing chunked search/operators/native work|332 CPU passes, 251 expected skips|
|Final hook / legacy old-runner regression|8 pass (launcher CRKSPH lifecycle and 7 old checkpoint tests); Case hook appended to preserve existing positional fields|
|Runtime imports|566 modules pass; inherited static-scanner issue remains documented above|

Tiny 64-particle A/Bs omit video, retain progress/stall/velocity controls and run
serially on the GPU. CPU legacy drivers took154.88s total, launcher drivers94.68s;
native restart/core82.23s and legacy FP64/core54.22s are overlapping controls.
NCCL launcher rank-zero wall times are2.85–31.24s including setup/IO; all timings
are diagnostic under contention, hashing and process/IO overhead. No throughput
or physical capacity claim follows.

L3 baseline: the two-step shared-GPU launcher fixtures recorded980 CRKSPH,
740 CompSPH and1069 Monaghan control handshakes per rank at the report snapshot,
including initialization, exhaustive hashing and checkpoint/output controls.
Counters precede final report publication. This confirms substantial control/
hash overhead to address next; timings must not be used to rank transports.
Next: separate exhaustive replica hashing from normal validation, then combine
metadata/particle exchanges and reuse buffers while retaining exact-mode failure
and numerical controls. L4–L7 and physical-device/site gates remain open.

Evidence and reproducible scripts: frontend `.tmp/local-continuation-20261030/`.
Implementation provenance: core `7c09724`; frontend is the commit containing this checkpoint.

### L3 control/validation checkpoint (2026-10-30)

The first L3 slice is qualified on CPU/Gloo and bounded shared-GPU NCCL. It
reduces debug/coordination work without changing numerical kernels, payload
broadcast order, scalar reduction arithmetic, Butcher stages or owned J/W.

`RowChunkProcessGroup(..., replicaChecks='exact')` remains the default. Exact
mode retains the existing operator-input fingerprints (owned pair buffers and
unread scratch remain excluded). Explicit
`replicaChecks='contract'` (CLI `--replica-checks contract`, Slurm
`REPLICA_CHECKS=contract`) validates operator tensor layout/device/AD and operation
contracts without copying every field to CPU for a content hash. Both modes
force exact agreement at search/rebuild, initial/accepted-state and checkpoint
boundaries, plus the existing source clock/generator contract. Select the same
mode on every rank when constructing the group; the mode is part of every
control digest. Checkpoints can resume under a different mode in a fresh job.

Contract mode assumes the dependency schedule maintains replica contents during
an RHS evaluation. It deliberately does not detect every transient corruption
of a reference/correction field before owner outputs are mixed; exact mode is
the debugging choice for that guarantee. Exact boundary checks are not a claim
of equivalent fault detection inside each stage. AD/capture and other previously
unsupported contexts remain rejected.

Control send/receive storage is allocated once per group; CUDA also retains
pinned host staging. Gathered packets share a contiguous receive tensor, with
cached peer views and one explicit blocking device-to-host packet copy, replacing
one allocating host copy per peer. The host read completes before a buffer can
be reused on another current stream. This is a serial eager interface, not a
concurrent-thread/async-control API. Persistent packet storage is1.5KiB on CPU or
3KiB across CUDA/host for two ranks (native/backend workspace excluded).

A fused local-validation/JSON-contract handshake preserves coordinated exceptions
while removing redundant validation acknowledgements. Scalar validation and
clone allocation now precede one handshake; operator metadata is similarly
combined. All particle outputs from an operator are validated in one ordered
batch before any broadcast. Payload broadcasts themselves retain their previous
field/owner order; no floating reduction, dtype conversion or pair gather was
introduced. An invalid last output cannot leave earlier fields partly exchanged.

Matched measurement uses a frozen core source tree from `7c09724`, the same
frontend numerical code/fixtures, two ranks, 8 solver variants / 68 RHS per rank.
The frozen adapter shim only accepts the new exact keyword and counts existing
hash input bytes; its original control/search/dispatch implementation runs.
Explicit Python Torch factories inside control calls are instrumented in both
implementations. Timings include instrumentation/contention and are diagnostic.

|Per-rank quantity, FP32 CPU and NCCL (same counts)|Frozen exact|New exact|New contract|
|---|---:|---:|---:|
|Control handshakes|6,756|3,288|3,288|
|Replica bytes hashed|7,860,720|7,860,720|222,528|
|Explicit control Tensor factory calls after construction|20,268|0|0|
|Logical particle/scalar payload bytes|608,960|608,960|608,960|

This is51.3% fewer controls in either mode and97.2% less hashed data in contract
mode. The factory count excludes native/backend allocations and other solver
Tensors; the new group has two initial packet storage allocations on CPU/four on
CUDA. CPU FP64/int64 repeats the same handshake/factory counts, with hashed bytes
14,727,472 ->411,008 in contract mode. No throughput speedup is inferred.

Qualification (tiny A/Bs omit video, preserve progress/stall/velocity controls;
GPU jobs run serially under the existing isolated NCCL 2.32.3 runtime):

|Control|Result|
|---|---|
|Real CPU process drivers|5 pass: exact/contract × 2/3 ranks, plus fatal-peer termination|
|Core protocol fixtures|Self/unequal/empty populations, prepared density, covariance/gradients/CRK corrections, owned sums/rebuild/repartition and 8 existing recoverable failures pass; contract reference-content failure explicitly uses a forced exact boundary|
|Added packet/batch failures|Forced exact mismatch, differing live-rank modes, invalid scalar and invalid last batch output reject coherently; next controls recover with stable buffer addresses|
|Allocation/lifetime audit|12 warm controls invoke no explicit Tensor factories; cached addresses remain stable; GPU int64 cases alternate two CUDA streams for12 further controls in both modes|
|Frozen/new solver comparisons|CPU FP32/int32, CPU FP64/int64 and NCCL FP32/int32: 3 variants of implementation/check policy × 8 solver fixtures pass exact stages/owned W/particles/time/OU; existing2e-12 source-budget tolerance only|
|Long NCCL forcing/replay|16 accepted steps /128 fresh RHS per native scheme per rank, 256 RHS combined; every step forced changed-list replay, OU crossings, exact history/fields/work/time pass|
|NCCL 32-lane control|3 steps /48 RHS per rank across native schemes, forced replay and nondefault stream pass|
|CPU solver/launcher regressions|7 pass: two-/three-step exact/contract solver controls plus all3 scheme launcher lifecycles, 1/2/3 ranks; exact→contract→exact across jobs|
|Native checkpoint drivers|3 pass: changed budgets/ranks/int32→int64 offsets, contract continuation, exact fresh RHS/J/W hashes/forcing and coordinated IO/corruption/policy controls|
|NCCL launcher/checkpoint lifecycle|9 serial jobs /12 rank reports, all3 schemes exact→contract→exact with changed budgets/1/2 ranks and exact final particle/time/dt/OU parity|
|Existing focused chunk regressions|349 CPU passes, 265 expected skips, including the initial3 scanner tests|
|Static scanner follow-up|5 scanner regressions pass, including parent finder exceptions and installed-optional missing children; full runtime/static check passes566 modules /6063 imports /1314 files|
|Slurm template|Syntax/mock remote quoting passes with explicit contract mode; actual submission/site qualification remains open|

Core CPU process-driver wall total23.23s, solver/launcher134.79s, native checkpoint
77.80s and overlapping focused regression15.31s. Raw progress and per-step times
remain in the logs; the shared GPU and debug instrumentation invalidate transport
performance conclusions. All numerical/protocol controls pass. Failed pre-fix
scanner output and original retired notebooks are retained as evidence.

At this control/hash checkpoint, particle payload packing/workspace reuse remained
open; the following checkpoint qualifies them. These earlier changes batch output
contracts, not data payloads. Avoid
removing exact boundaries or failure controls as a shortcut. L4 filtering,
L5 stability expansion, L6 compatibility, L7 actual wide-edge accesses and physical
multi-GPU/site/performance remain separate gates. MPI/ROCm builds remain deferred.
Evidence/reproduction/source snapshots: `.tmp/l3-communication-20261030/`.
Implementation provenance: core `3742f10`; frontend is the commit containing this checkpoint.

### L3 particle packing/workspace checkpoint (2026-10-30)

L3 is qualified locally in the existing bounded eager process scope. Compatible
nonempty operator outputs share dtype-specific row packets; particle fields keep
their own storage and J/W remain owned. Packing/unpacking uses only same-dtype
copies, preserves signed-zero/NaN bits and changes no floating kernels, scalar
reduction arithmetic, fresh Butcher stages or exact-boundary policy. Exact input
checks remain the default, with the same explicit contract-mode limits above.

`RowChunkProcessGroup(..., particleWorkspaceBytes=4*1024*1024)` lazily allocates
one fixed-capacity byte buffer for packing. The capacity is read-only after
construction and included in the output contract on every rank; zero disables
packing. Whole-row tiles fit the capacity. Singleton fields, rows wider than the
buffer and groups whose tiling would not reduce broadcast calls use the direct
path. No per-shape/tag/dtype cache, buffer growth or borrowed output storage is
introduced. The default costs 4 MiB per process after first packed exchange, zero
before it; a 127-byte audit exercises mixed types, alignment and multi-tile owners.
This is additional communication scratch, separate from computational outputs,
owned pair buffers and backend/allocator workspace. Releasing the group releases
its retained scratch; the Torch caching allocator can retain reserved memory.

Whole-batch layouts, exact-alias topology, capacity and allocation agree before
any payload. Disjoint views of one storage and matching exact aliases are valid;
partial/cross-row overlap, retyped overlap and different alias topology across
ranks reject coherently. Allocation failure rejects before any field exchange;
subsequent controls/exchanges recover. CUDA records an event after the final
unpack and waits on it before reusing scratch, including another current stream.
`record_stream` additionally registers allocator lifetime. This adds no explicit
host synchronization; callers still own external producer/consumer dependencies.
The interface remains serial eager, not concurrent-thread/async/capture/AD safe.

Matched two-rank fixtures freeze the preceding control/hash core `3742f10`, use the
same frontend and 8 solver variants / 68 fresh RHS per rank, then run packed exact
and packed contract modes. A wrapper counts actual `dist.broadcast` calls; it
changes no solver/transport logic. Both ranks show the same counts:

|Per-rank quantity|Frozen 3742f10 exact|Packed exact|Packed contract|
|---|---:|---:|---:|
|Payload broadcasts, CPU FP32/FP64 and NCCL FP32|1,424|968|968|
|Control handshakes|3,288|3,288|3,288|
|Logical payload bytes, FP32|608,960|608,960|608,960|
|Logical payload bytes, FP64|1,202,464|1,202,464|1,202,464|
|Packed batches|0|116|116|
|Persistent packing storage / allocations|0 /0|4 MiB /1|4 MiB /1|

This removes 456 broadcasts per rank (32.0%) on these fixtures, without reducing
logical data volume. Logical bytes include each rank's own interval and scalar
payloads; they are not wire traffic. The earlier control/hash reductions remain
in place. No native/backend allocation or throughput reduction is inferred.

|Qualification|Result|
|---|---|
|CPU final process drivers|5 pass: exact/contract × 2/3 ranks plus fatal-peer termination; existing 8 recoverable faults and packet/batch audit retained|
|New packing audit|All 5 payload dtypes, scalar/vector/matrix tails, exact aliases/disjoint storage views, signed zero/NaN bits, zero-channel fields, zero populations, empty owners, tiled rows and oversized/direct fallbacks pass bitwise|
|Packing failures|Injected allocation failure, partial overlap, differing alias topology, retyped overlap and rank-budget mismatch reject before any payload; following controls recover|
|Bounded storage/lifetime|One 127-byte storage allocation across changing shapes/types and repeated warm calls, no explicit Tensor storage factories in warmed exchanges, stable pointer and retained outputs; disabled/uneconomical packing allocate zero bytes|
|Matched solver controls|CPU FP32/int32, CPU FP64/int64 and NCCL FP32/int32: frozen/exact/contract all pass exact stages, owned J/W, fields/time/OU; source-budget tolerance remains 2e-12|
|NCCL streams/offsets|Both check modes and both offset widths pass 12 alternating streams with delayed producers/unpacks, retained outputs and event-fenced scratch reuse; actual NCCL 23203 /same shared GPU UUID verified|
|NCCL replay/lanes|16-step native forced-replay controls retain 128 fresh RHS per scheme/rank and OU crossings; 3-step 32-lane controls also pass; 7 cohorts /14 rank reports /1,016 distributed RHS in the main GPU sequence|
|CPU solver/launcher/native restart|10 final passes, including all 3 schemes,1/2/3 ranks, changed budgets/int32→int64 offsets and exact→contract→exact mode changes|
|Broader first pass|16 solver/launcher/native/legacy driver passes (includes 6 COO/CSR imports); 351 focused chunk/scanner passes and 265 expected skips. These began before the final alias/schedule guard; final transport drivers/cohorts above qualify that guard|
|NCCL launcher/restart|9 serial jobs /12 rank reports, all 3 schemes, changed budgets/1/2 ranks, exact→contract→exact; exact final particle/clock/dt/OU parity and source hashes pass|
|Full import scanner|566 modules /6063 first-party imports /1314 files, no failures|

CPU final process suite 24.78s and solver/launcher/native drivers 209.08s; focused
chunk/scanner first pass 15.23s. Tiny A/Bs omit video and retain progress, stall and
velocity controls. Timings include debugging/control/contended execution and are
diagnostic only. For the 8 matched cases (16 accepted steps per rank), rank-zero
aggregate wall time /ms per accepted step was:

|Runtime|Frozen exact|Packed exact|Packed contract|
|---|---:|---:|---:|
|CPU FP32|3.66s /228.8ms|3.64s /227.4ms|3.30s /206.1ms|
|CPU FP64|3.45s /215.8ms|3.40s /212.7ms|3.34s /208.7ms|
|Shared-GPU NCCL FP32|17.34s /1083.8ms|16.35s /1021.8ms|15.65s /978.2ms|

These figures do not establish a transport speedup or a regression under ACSPH
contention. Physical devices/topology, aggregate capacity and scaling remain open.
L3 does not recycle computational output tensors retained by later stages, reduce
checkpoint seed/restored adjacency overlap, or migrate pair arrays. Next is L4
distributed filtering/redistribution; L5 stability expansion, L6 compatibility and
L7 actual wide-edge accesses remain ordered gates. MPI/ROCm builds stay deferred.

Evidence/reproduction/frozen sources: `.tmp/l3-payload-20261030/`; draft results
before the final alias/schedule guard are retained separately. Final core/front
heads are recorded in its `commits.json` after the paired checkpoint commits.
Implementation provenance: core `0513736`; frontend is the commit containing this checkpoint.

### L4–L6 continuation (2026-10-30, active)

User authorizes L4, then L5 and L6 in order if unblocked while away. L4 will count
filtered rows on existing owners, agree the target partition, and stream bounded
filtered J intervals directly into target chunk storage through c10d point-to-point
transfers. One old-chunk temporary at a time; no global pair tensor. Generate I
locally, preserve source leases/J/W, and qualify existing fresh-stage compatible
replay on list changes. Each local allocation/kernel/transfer contract must agree
before payload, including empty/skewed owners, changed boundaries and failures.
L5 extends randomized partition/capacity, memory, forcing and fault controls.
RHS fault probes also cover native W/half-velocity/dt/material preparation; local
allocation errors must agree before the work operator enters a payload.
L6 records and tests an explicit support/rejection matrix, then expands compatible
contexts as separately qualified changes. Physical devices/site/scaling, large
L7 allocations and MPI/ROCm builds remain deferred. Tiny serial GPU controls only;
no shared install/source changes and no throughput conclusions under ACSPH load.

### L4 distributed filtering/redistribution checkpoint (2026-10-30)

Distributed resident filtering now counts only existing owner rows, exchanges
O(N) counts, agrees a new whole-row plan and allocates only new owned J/I.
Existing count/filter kernels preserve source neighbour order. Each source owner
filters one original chunk into a bounded live-J temporary, then `transferOwned`
copies its contiguous row intersections directly into target chunks, with c10d
send/recv between changed owners or a local copy for unchanged owners. Stored
I is generated locally. No global pair tensor, pair gather or W transfer exists.

Every live rank validates each transfer's source/target/count/dtype/plan before
payload. Allocation, filtering and view failures propagate before their payload;
zero-edge intervals skip traffic. Source lists and work buffers remain valid,
independent and unchanged. New filtered adjacency identity triggers the existing
native fresh-stage replay when used during compatible-work accumulation. It does
not remap retained W or change integrator stages/reduction arithmetic. Filtering
is explicit; ordinary eager native steps still use resident acceptance kernels.

Peak storage includes the old and newly allocated owned lists, replicated current
snapshots/counts/prefix, plus at most one original source chunk's filtered J and
local offsets. `searchScratchBytes` reports that logical temporary bound, not
allocator/backend peaks; scan/cast temporaries and reserved memory remain separate.
Local filtering retains its original linear interval walk. Dynamic group size,
physical multi-GPU transfer and asynchronous recovery remain unqualified.

Qualification: final CPU core/search suite191 pass /187 expected skips (including
5 process drivers:2/3 ranks ×exact/contract and fatal-peer termination);4 new
frontend replay drivers pass across2/3 ranks ×exact/contract. Each core worker
adds27 source/filter fixtures across all support modes,2D/3D, stored-I changes,
source/target chunk splits/merges, periodic fields, zero populations/references
and empty owners. Tiny flat filters verify exact counts, prefix, each owned J/I
and padding. Retained J/W and leases survive; old W rejects the new list identity.
Temporary logical bytes stay below one original owned chunk's live J/offset bound.

Six added recoverable failures cover mismatched policy, capacity, target/source
allocation, stale source and malformed transfer layout; no payload occurs after
invalid preparation, and a subsequent valid filter recovers. CPU FP64/int64 and
NCCL FP32/int32 plus FP32/int64 each run both modes for core/filter and3-step
native filtering/replay:8 rank reports /108 filtering fixtures /192 distributed
RHS per configuration. Actual shared-GPU NCCL23203 verified. Native CRKSPH and
CompSPH each retain8 fresh RHS per accepted replay step and exact work/fields/
time/OU parity, with the existing2e-12 source-budget tolerance only.

Final core suite28.29s, frontend replay suite33.80s; rank-zero contract-mode filter
matrix0.65s CPU FP64 and6.86/6.89s NCCL int32/int64. Contract solver totals for
6 accepted steps/rank:1.92s CPU (320ms/step),9.90/10.00s NCCL (1649/1667ms/step).
These tiny A/Bs omit video, retain progress/stall/velocity controls and run GPU
jobs serially. Timings are diagnostic under ACSPH contention, with no speedup or
physical-device claim. Full scanner passes566 modules /6065 imports /1314 files.

An initial broad run failed only the obsolete single-owner rejection-message
assertion; the corrected clean run passes. The pre-fix XML/log and the restored
linear local-walk source variant are retained in evidence. Distributed code paths
are unchanged by that local-walk correction. Evidence/reproduction/frozen files:
`.tmp/l4-filtering-20261030/`; final paired heads in `commits.json`. L4 is locally
qualified. Proceed to L5 seeded partition/capacity/lifetime and forcing/fault
stress, then L6 compatibility. MPI/ROCm and large-memory L7 remain deferred.

### L5 bounded stability/failure checkpoint (2026-10-30)

A seeded process worker now changes populations (including zero/unequal sets),
2D/3D, all support modes, skewed/isolated rows, periodicity, headroom/alignment,
chunk capacities/row caps, stored-I layout and ownership repeatedly. Guaranteed
transitions move all rows into one owner and split them back across owners.
Every cycle compares exact filtered counts/prefix/owned J/I to a tiny independent
flat filter, retains and checks old W/J, exercises safe in-place lease invalidation
and bounded packet reuse, then uses weak references/GC to require that obsolete
pair tensors have no remaining Python owners. One source-chunk scratch bound is
checked each cycle. Packet scratch is a single stable 4 KiB allocation.

Injected failures after target allocation but before a transfer contract's payload
reject coherently, including an error on a nonparticipating rank. Retrying keeps
source leases/work valid. Fatal sender and receiver probes exit after the transfer
contract and require torchrun to terminate peers; ordinary Gloo peer death and
all earlier protocol/filter/IO failure controls remain covered. No arbitrary
post-device-fault recovery is promised; fatal/asynchronous faults still rely on
process-group deadlines and launcher termination. NCCL fatal-participant probes
were deliberately not added on the shared busy GPU.

The RHS fault probe found native W/half-velocity/dt/material preparation outside
a coordinated phase. `workInputs` now runs that local preparation behind one
`localPhase` before either native work operator starts. A work allocation failure
on only the last rank therefore reaches all live ranks before work payload, and
a fresh solver attempt with the same communicator recovers with exact flat parity.
This adds one control handshake per prepared weighted native work stage (also for
replay), with no floating-function/formula, stage, reduction or hash-policy change.
AST fingerprints of every existing Warp kernel/function in both touched work files
are unchanged. Flat execution runs the same local preparation without a collective.

|Qualification|Result|
|---|---|
|CPU process drivers|9 pass: prior2/3-rank controls plus4/5 ranks ×exact/contract stress,32 cycles/cohort,2 seeds,18 stress rank reports; fatal sender/receiver adds2 passing drivers|
|CPU FP64/int64|5 ranks ×48 seeded cycles pass;3-rank native RHS allocation failure/retry passes both native schemes|
|RHS/replay/compatible-work regressions|14 pass,2 expected skips:3-rank exact/contract work failure/retry,2/3-rank filtering replay, retained single-owner native controls|
|Shared-GPU NCCL stress|Exact/contract ×2 ranks ×48 generations; no obsolete pair tensors, one stable4 KiB packet buffer,24 KiB span in sampled allocated memory across changing tiny populations|
|Shared-GPU RHS faults|Both modes and both native schemes reject one-rank work allocation, then recover exact RHS/fields/time/forcing|
|Shared-GPU long forcing/replay|64 steps each CRKSPH/CompSPH,512 fresh RHS per scheme/rank,54 OU knots; exact stage/W/fields/clock/history parity and declared2e-12 source-budget tolerance; steady allocated-memory span0 after step8|
|Expanded checkpoint/fault lifecycle|Reference1 rank →save4 ranks with midstage/overwrite/interrupted-write cleanup and retry →resume5 ranks/int64 with empty owners; exact fresh rows/work/clock/forcing continuation passes,10 rank reports|
|Full imports|566 modules /6066 first-party imports /1314 source files, all resolve|

CPU process driver total81.83s, added fatal drivers and FP64 controls retain their
raw wall logs; frontend RHS/replay/work suite66.01s. NCCL48-cycle stress rank-zero
wall19.27s exact /21.27s contract. The64-step forced-filter/replay controls take
108.00s CRKSPH (1687.5ms/step) and92.80s CompSPH (1450.0ms/step) on rank zero.
The GPU sequence has10 rank reports and2,112 successful distributed solver RHS,
plus deliberately rejected attempts and flat reference RHS. These are correctness
controls under ACSPH contention, with tiny A/B video exceptions and progress/stall/
velocity guards; no throughput claim follows. Reserved memory is sampled, not
required to shrink. Logical bounds, weak-reference lifetime and Torch allocations
do not measure all backend/driver/process memory or establish multi-hour stability.

L5 is locally qualified within this bounded scope. L6 will expand compatible
Godunov contexts and record/test supported and rejected schemes, walls/materials,
AD/graphs and lane/prepared behavior, keeping native controls. Physical devices,
site/topology/scaling, MPI/ROCm builds and large-memory L7 remain separate gates.
Evidence/reproduction/frozen sources: `.tmp/l5-stability-20261030/`; paired heads
are recorded in its `commits.json` after checkpoint commits.

## Phase 5 — remaining schemes and compatibility

- [ ] Inutsuka list variants and MFM/MFV geometry/kappa/closure workspaces, including
  the independent closure reserve and iterative matvec/vector exchanges/global
  reductions. Do not claim meshless support from flux-kernel parity alone.
- [x] Native accepted-step checkpoint/restart across chunk budgets and local
  process counts: canonical particle/forcing/time state, reconstructed owned
  adjacency and int32-to-int64 local offsets, within the L1 scope above.
- [x] Bounded version-1 COO/CSR import with owned pair-order verification and
  int32/int64 local offsets. Other legacy policies/versions remain unqualified.
- [ ] Physical GPU-count changes and larger/broader restart qualification; persist
  only needed history metadata, never create a global pair tensor for checkpointing.
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
checkpoint tests selected by that owner. New focused suites now include core
`tests/radiusSearch/test_{rowChunks,chunkedSearch,chunkedState}.py`,
`tests/operations/test_chunked{PairField,_operators}.py`, both repositories’
`tests/distributed/` workers, and frontend `tests/test_chunkedCheckpoint.py` /
`test_distributed.py` / `test_chunkedCompatibleWork.py`. Extend these where useful.

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
