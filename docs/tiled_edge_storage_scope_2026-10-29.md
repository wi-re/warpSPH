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
