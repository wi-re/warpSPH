# Tiled edge-storage scope (2026-10-29)

User requests scope and breakage assessment, not a production migration. This document follows frontend ebfec9a and paired core3c56e2f. Production solver/storage code is unchanged in this investigation. A small CPU-only ABI probe and a source inventory are retained.

## Assessment

Tile/slot storage is feasible at the array-descriptor level and is a medium core refactor, substantially wider than the COO-I omission. The main task is a shared edge-access and conversion contract, not a numerical-method change. A naive reshape breaks fixed-rank Warp argument types, vector/matrix inference, raw pair reads/writes and cached closure buffers. A careful adapter can preserve the flat Torch/public-data interface, CSR row/pair ordering and particle states.

AST inventory finds40 direct neighbour lookup sites in25 core files and56 sites in51 frontend files:96 sites /76 files. These are syntactic lookup candidates, not a complete dependency count or a count of demonstrated bugs. Writers, tensor conversion, launch/output plumbing, graph metadata and I/O add more files. Plan on roughly90–120 production files for broad adoption, plus tests; an isolated one-scheme forward prototype could be closer to25–40. These are planning estimates with overlap, not exact diff sizes.

Rough engineering effort: about3–5 focused working days for a useful eager compressible implementation, and roughly1–2 working weeks including broad old-path, AD, graph, restart and large-address qualification. This is an effort estimate rather than a wall-clock promise. Scope is much smaller if the first backend explicitly supports only one forward scheme; it is larger if every public operator and differentiation path must support giant arrays from the first commit. Dispatch/AD discoveries could extend the range.

## Recommended representation

Keep Torch ownership and logical shapes: J remains `(capacity,)`, W remains `(capacity,dim)`, scalar pair fields remain `(capacity,)`. Create explicit zero-copy Warp edge views of those storages. Use a power-of-two slot width, initially2**16 as a testable candidate:

```text
q = int64(global_edge)
tile = int32(q >> tile_shift)
slot = int32(q & tile_mask)
value = warp_view[tile, slot]
```

The casts to int32 occur only after splitting a wide address. Particle IDs and per-row counts stay int32; CSR offsets and cumulative edge addresses must be int64 for large lists. The existing offset_t configuration and wide lane-range arithmetic are already useful, but selecting offset width remains a pre-import decision. The old flat capacity guard must be replaced by representation-specific checks, not removed.

An explicit index-view descriptor can carry the2D array plus shift/mask. Legacy small/flat buffers can be viewed as one row, with shift31 and an unused outer stride of zero; tiled buffers use shift16 and normal tile strides. The single-row zero outer stride avoids overflowing the descriptor’s signed-int32 byte-stride field for a very long row. The prototype exercises this descriptor through one concrete kernel ABI. Distinct scalar/vector edge view roles are also needed for W, expanded pair arrays and MFM kappa.

For multi-row tiled buffers, round capacity to a slot-width multiple; keep numEdges/live CSR counts independent of the physical tail. With65536 slots, extra final-tile padding is less than256KiB for one int32 array and768KiB for one3D FP32 vector. The measured96³ capacity501,202,944 would gain16,384 slots:64KiB J and192KiB W. This is additional to, rather than a substitute for, the existing Verlet/capacity policy. A two-view tail could avoid rounding but adds complexity; it is unnecessary for first bring-up.

Warp1.17.0 stores each shape extent and byte stride in int32, while native array indexing multiplies strides into size_t byte offsets. Therefore both per-dimension extent and per-dimension byte stride need validation.65536-slot strides are256KiB for int32 and768KiB for FP32 vec3, comfortably within that ABI. This does not by itself qualify large CUDA memory access.

Grid sortIndex should retain its particle-index meaning and order. Wrap it as an edge-index view where the shared neighbour-loop API needs one, rather than changing the physical grid permutation into CSR data. Most loops already receive a wide begin plus a small per-row count. A row can cross a tile boundary without splitting, duplicating or reordering it.

## What needs changing or would break

|Area|Concrete assumptions / required work|Risk|
|---|---|---|
|Search construction and filtering|Compact-hash collection, COO row fill, Verlet count/filter/update and capacity reuse currently allocate/pass flat arrays. Add edge allocation/views and replace each write/read; retain ordering, tails and in-place addresses. The small all-pairs oracle can remain explicitly flat-limited initially.|Medium|
|Shared CSR/kernel ABI|adjacencyData.neighborList is a fixed1D array. Range functions accept offsetArray as1D and reuse them for grid sortIndex. Replace with explicit accessors/views and update both branches. The inventory includes14 core primitive files,5 CRK core files,2 polyfit files and2 MFM files.|Broad mechanical change; high compile-compatibility exposure|
|Conversions, registry, prepared launches|castTorchToWarpAsBuiltins interprets2D as a vector and3D as a matrix. Simply reshaping J/W goes through the wrong conversion. Add explicit edge roles and views across arg_extract, build_fn/stateBundle, cached fields, null fields and PreparedLauncher; keep actual particle vectors/matrices unchanged.|High|
|Compatible/expanded work|CRKSPH/CompSPH direct W writes/accumulation and final reduction are flat. ap/av/f expanded routes also write by edge address. Edge-sized OutputSpec/ExtraSpec and allocation must distinguish logical Torch shapes from Warp views; .numel() on scalar vector components is not an edge count.|High; otherwise indices can silently refer to wrong pairs|
|MFM/MFV geometry and closure|Kappa writes/reads, direct matvec and cached neighbour/weight buffers are flat, including an independent1.25x closure-workspace reserve. Prepared launches bypass ordinary conversion. Multigrid particle arrays are a different index domain and should not be widened indiscriminately.|High; tiling J alone is insufficient|
|Graphs/caches and rebuilds|Clone keys currently use Torch addresses/shapes. Include view format/tile metadata; reuse capacities must remain tile-compatible. Update scalar metadata on every bundle refresh. Captures must not retain an old view layout after a rebuild; explicit overload/module warmup remains necessary.|High for stale metadata/lifetime bugs|
|AD/JVP/HVP|Hand-built JVP prefixes and reverse-mode seeds assume old array ranks. Preserve the exact taped Warp-array identity and reshape its gradient seed to the same physical view; avoid accidental detach/copy. HVP raw-edge launches also need tile/slot or bounded chunks when E exceeds one launch dimension.|Highest compatibility risk; can initially be a declared unsupported large-array path|
|Restart, export, diagnostics and tests|Flat Torch buffers allow most output schemas to stay intact. Old int32 CSR offset checkpoints need widening on load into an int64 runtime. Persist/reconstruct layout metadata and capacity policy. Raw COO reconstruction, gathers and adjacency export can allocate huge temporaries; stream/chunk where needed. Force tiny tiles in tests instead of relying on small cases that never cross a boundary.|Medium functional change; potentially high transient memory|

The frontend inventory reaches shared adaptive support, dissipation, shock switches and reconstruction as well as compressible solvers. It also includes IISPH/incompressible kernels, WCSPH/delta-SPH, shifting, surface detection, mDBC and sample utilities. A universal adjacency ABI migration therefore touches these even if their physical lists remain small and their public defaults stay unchanged. A separate opt-in backend can isolate initial support, at the cost of maintaining two ABIs until consolidation.

Numerical formulas, forcing, EOS, boundary conditions, integration coefficients, fresh-stage evaluation and compatible pair identity should not change. Neither should particle-field tensor shapes or CSR neighbour order. Preserve the same sequential accumulation order; do not combine this with atomics, pair sorting, a new Verlet skin or stage reuse. This makes bitwise old-size parity a reasonable gate for a pure representation change.

## Probe evidence and rejected shortcut

`scripts/probe_tiledEdgeABI.py` passes six CPU checks on the installed Warp1.17.0: flat and tiled index reads through one explicit descriptor, vector writes crossing a65536-slot boundary, scalar/vector reads at logical offsets around2**31, and rejection of the corresponding flat dimension. The contiguous test buffers total2MiB. The wide logical views have2,147,549,184 entries but alias4-byte/12-byte backing storage with zero strides. They prove shape/address-splitting acceptance, not physical access beyond8GiB, CUDA behaviour, throughput, gradients or a solver trajectory.

An earlier helper overloaded only on1D versus2D array rank segfaulted on the small2D case; declaring both overloads first did not repair it. Retained generated CPU code dispatched the2D case through the flat-reading helper. An explicitly named2D kernel worked; a single concrete edge-view descriptor then also worked for both representations. Treat the rank-overloaded shortcut as rejected/unqualified. This investigation does not establish a general upstream Warp bug or prove all generic function patterns unsafe. Failure sources and logs are retained alongside the successful probe.

Reproduce the bounded probe without CUDA allocation:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 /home/lu26029/miniconda3/envs/warp/bin/python scripts/probe_tiledEdgeABI.py --out .tmp/tiled-edge-abi
```

## Suggested implementation gates

1. Introduce explicit edge views and representation-aware capacity/stride checks. Prove flat and tiled loads/stores, zero/partial tiles, forced small tile widths, wide CSR scans, no-copy identity, in-place reuse and exact list ordering. Test real physical byte offsets beyond INT_MAX separately from the aliasing probe.
2. Bring up one eager forward scheme plus shared support/density/gradient operations. MFM has no compatible W but requires kappa and the cached closure workspace; GSPH/Monaghan avoid W and are simpler first addressing controls. Retain the old backend until unsupported paths are explicit.
3. Add CRKSPH/CompSPH W and expanded fallbacks, then all seven compressible paths. Require exact fields, full live-W/pair-ID checks, forced changed-list replay, positive native RHS and unchanged energy budgets at midpoint and multi-weight tableaux.
4. Qualify graph recapture/stale-list fallback, the MFM closure cache, checkpoint32→64-bit offset conversion and old-file import. Run reverse/forward/second-order gradient and deferred-backward tests at tile boundaries before claiming general compatibility.
5. Exercise existing small cases with a deliberately tiny tile width so every family crosses boundaries. Then run bounded real CUDA storage tests above2**31 logical entries with enough physical backing, followed by a short160³ flow. Only then try192³; include both Blackwell and GH200/Hopper qualification before claiming portability/performance.

Expected cost is extra integer address splitting and view/metadata plumbing, without a device copy or a change to contiguous memory order. A power-of-two width avoids integer division in production accessors. Occupancy, generated code and launch-host overhead remain unmeasured. Do not promise a speedup or assume the arithmetic cost is negligible; benchmark steady kernels and complete rebuilds separately.

## Relation to the256³ goal

Tiling removes an addressing limit; it does not reduce the current pair count or per-edge bytes. Extrapolating the latest post-I-removal live peaks gives approximately157GiB for CRKSPH,110GiB for MFM and83GiB for GSPH at256³, before allocator/context headroom or developed-flow growth. CRK W alone remains approximately106GiB. Thus this work can unlock192³-class qualification but cannot, by itself, make256³ fit for CRKSPH or MFM. GSPH is closer by live bytes, but its reserved-memory behaviour and physical qualification still prevent a fit guarantee.

The earlier ~156³ addressing threshold refers to the buffered SPH layout. MFM uses a different physical-neighbour list and its closure workspace adds its own25% reserve, so its first failing allocation must be calculated separately rather than inheriting that threshold.

Evidence: `.tmp/tiled-storage-scope-20261029/inventory.json`, `final/abi-probe.json`, successful and failed probe sources/logs. Source locations below are the surveyed anchors; counts include alternative/differentiation paths and are not a runtime coverage claim.

- Core: `dataTypes/adjacency_t.py`, `radiusSearch/{csr,compactHash/search,compactHash/wp_collectNeighbors,verlet/filter}.py`, `util/{cast,fieldRegistry,stateBundle}.py`, `autograd/{arg_extract,launcher,operator_spec,prepared,stateAwareWarpFunction}.py`, `coreOperations/_jvpCommon.py`, `mfm/{wp_mfm,wp_closure,warpBackend}.py`.
- Frontend: `modules/{compSPH,crk}/{accel,work}.py`, `modules/compSPH/{balance,workReduction}.py`, `modules/meshless/flux.py`, `modules/reconstruction/meshless.py`, `utils/cudaGraph.py`, `caseUtils/drivenTurbulence.py`, `io/hdf5.py`.


## Particle-row chunks and replicated-state multi-GPU proposal (2026-10-29)

The user proposes partitioning complete CSR rows by an edge budget, and retaining
only each GPU's owned rows of both J and compatible W. This is feasible in the
surveyed gather kernels and is a stronger architectural candidate than uniform
slot tiling alone. It remains a proposal, not an implemented or qualified backend.
The earlier file counts describe a global direct-access inventory, largely
mechanical edits; they do not imply rewriting flat Torch/CPU/output consumers.

At a coordinated neighbour-list rebuild, each GPU counts neighbours for its
assigned particle range against the same synchronized reference state/search
configuration. All-gather those disjoint int32 count slices into one complete
count vector on every GPU (not an all-reduce sum of duplicated full counts).
Each GPU independently computes the same int64 prefix scan and deterministic
chunk boundaries/device assignments from this shared source of truth. Exchange
or verify a compact partition/version signature before allocation. At256^3,
the replicated count vector is64MiB and an N+1 int64 prefix is approximately128MiB;
no global edge tensor is needed. Count and collect must use the same positions,
supports, periodic conventions, stored-neighbour acceptance and ordering contract.

For a chunk starting at a and ending before b, its live edge count is P[b]-P[a]
and its local row offsets are P[a:b]-P[a]. Allocate J and associated pair fields
only for locally assigned chunks, including bounded capacity headroom, and
collect their rows directly. Keep those lists resident and reuse them under the
existing validity rules until the next coordinated rebuild; do not re-count or
rebalance on each SPH operation. Multiple chunks may be assigned to one GPU.
Any reassignment while compatible W is live must transfer matching J/W or use
the existing scientifically equivalent replay path, rather than discarding work.

Choose contiguous particle intervals
whose summed stored row counts fit a configurable edge/byte budget. Each chunk
has a global row start/count, local CSR offsets/counts, global J particle IDs,
and ordinary flat Torch/Warp edge buffers. Rows do not straddle chunks. Equal
edge work generally means unequal particle counts; equal particle counts cannot
also guarantee a hard edge cap in an inhomogeneous flow. Account for Verlet
storage, capacity headroom and the largest associated pair field, not just live
physical neighbours. Empty rows still have owners; a row exceeding the budget
needs an explicit oversized-row policy. Never silently truncate it.

A launch over a chunk maps local row r to global particle i=rowStart+r; its local
CSR offset addresses its flat J/W buffers. Particle fields remain globally
indexed. This can preserve flat kernel array ranks and avoid per-edge tile/slot
arithmetic, but the query/global-row mapping must be propagated explicitly:
simply slicing query positions is insufficient for corrections, extras and outputs.
Neighbour construction must collect owned rows directly; splitting a giant list
after building it would retain the allocation limit and memory peak. Many logical
chunks can belong to one GPU, and the scheduling/granularity is independent of
the storage budget. Ten million edges is an example, not a selected default.

For a first multi-GPU backend, replicate particle/stage/reference fields and the
search grid, while distributing row-owned J, W and other edge workspaces. Gather
owned particle outputs between dependency passes, not pair arrays. CRK force
completion requires globally current stage half velocities before its work sweep;
final compatible reduction needs globally current final half velocities. Each GPU
retains and reduces only W for its owned rows. W need never be assembled on one
GPU. Row ownership, pair identity and ordering stay stable throughout accumulation;
rebalance at safe step boundaries, or explicitly migrate matching J/W together.
Fresh-stage and changed-list replay semantics remain unchanged.

Density, correction/gradient, force and source dependencies determine communication
barriers. Global timestep/conservation reductions and MFM closure iterations need
explicit collective handling; MFM is not just an independent final gather.
Replicated-state communication scales with particle count rather than edge count,
but includes derived fields (including matrices), not only the ~80-byte primary
state. A global FP32 3D vector field at256^3 is192MiB before collective traffic
accounting. Replication is a useful first single-node design, with spatial halo
exchange a later scalability improvement.

On one GPU, splitting resident J/W does not reduce their total bytes. It bounds
scratch only when temporary lifetimes are actually processed chunk-by-chunk.
Distributed retention does reduce pair bytes per GPU: CRK's current rough157GiB
single-GPU256^3 live estimate suggests approximately39GiB of one-quarter work plus
replication/communication/workspace overhead, not a measured fit or speedup.
Do not divide the entire live estimate by four as a precise model: some terms
replicate, others shard, and capacities/overlap change with execution.

Recommended next scope is a row-chunk abstraction and single-GPU forced-chunk
parity gates before a replicated-state multi-GPU runner. Include neighbour list
construction, compatible W, rebuild/replay and memory accounting in that design.
Uniform tile views remain an optional storage mechanism, not a prerequisite.
Bitwise per-row parity and exact pair identity are appropriate first gates;
distributed collective reductions can differ in rounding. No production changes,
GPU allocation or additional simulation were made for this proposal.

Hardware qualification: Isambard documents four96GB GH200 GPUs per node and
NVLink-C2C, which is the Grace CPU/Hopper GPU link. That alone does not establish
a four-GPU NVLink/NVSwitch domain. Inspect allocated-node topology and measure
actual peer/collective bandwidth before predicting transfer cost. Sources:
https://docs.isambard.ac.uk/specs/ and
https://docs.nvidia.com/dccpu/grace-perf-tuning-guide/index.html .


## Pickup plan and integration dependency

The user selected separate-worktree development after merging the current paired
work at a mergeable ACSPH checkpoint. The canonical [implementation and handoff
plan](../../CHUNKED_NEIGHBORS_MULTI_GPU_PLAN.md) now owns phase checklists,
integration/worktree logistics and acceptance gates. This scope remains the
source inventory/design record; its earlier tile-only effort estimates are not
an estimate for the expanded multi-GPU programme. Smaller turbulence investigations
remain in the existing checkout; this update launches no production work.
