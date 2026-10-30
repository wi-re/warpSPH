# Compressible CSR-only neighbour allocation (2026-10-29)

Optional core omission of COO I is implemented and enabled by default for compressible runner contexts. Other scheme families retain stored I pending their consumer/performance audit. The Verlet factor, padding, support solve, integrator, physical operators and forcing are unchanged.

## Paired implementation

Core prerequisite: `warpSPHCore`0.6.3, branch `codex/compressible-csr-only`, commit `3c56e2f`; persistent sibling checkout `/home/lu26029/.codex/worktrees/meshless-boundary-balance/warpSPHCore`. The shared `/home/lu26029/dev/warpSPHCore` checkout was not modified. The frontend dependency now requires this core version. The paired feature has not been merged into either shared dev checkout.

`buildVerletList` and compact-hash searches accept `storeI=False`. The existing J-only collection kernel allocates J only and skips I filling. `AdjacencyList.paddedI` is None; counts/offsets/J retain exact order and capacity. Filters inherit representation. In-place rebuilds reuse buffers only for matching representations; a requested switch forces a rebuild. The core default remains `storeI=True`.

Frontend `compute.neighborOmitI` / `simulation.neighborOmitI` is nullable: None resolves to true for compressible runner contexts and false elsewhere. Explicit False restores COO. A bare low-level `buildConfig` cannot infer a family and retains COO unless explicitly enabled. Pair-array capacity now comes from J. MFM/MFV forward the option through timestep geometry, ordinary geometry and optional Verlet caching.

Explicit `cooRows()` / `livePairs()` compatibility access reconstructs temporary rows without caching them. Wall energy fractions, experimental entropy/Spheral controls, Torch AD fallbacks and raw-pair diagnostics can still allocate I temporarily. Normal seven-scheme periodic trajectories are tested with reconstruction forbidden. Graph keys/clones accept None. Checkpoints omit I, restore old COO history, and reconstruct/drop rows when switching storage on restart.

## Matched96³ memory controls

Eight finite controls ran serially against one frozen paired runtime: COO/CSR for CRKSPH, GSPH and MFM for12 accepted steps, followed by an MFM36-step pair. Each has an independent three-step warmup, video/progress, native fresh stages, raw-RHS/accepted positivity and source/energy guards. The36-step extension was required because the12-step MFM window misses the earlier known high water. B7/n_h4, the original Verlet factor/headroom and seed1234 forcing/thermal settings are unchanged. These are startup memory controls, not developed-turbulence or throughput qualification.

|Scheme/window|COO live GiB|CSR live GiB|Saved GiB|COO reserved GiB|CSR reserved GiB|
|---|---:|---:|---:|---:|---:|
|CRKSPH/12steps|10.159|8.292|1.867|10.266|8.398|
|GSPH/12steps|8.124|4.390|3.734|9.801|6.066|
|MFM/36steps|10.391|5.782|4.609|10.568|5.953|

The12-step MFM pair instead measures8.504→4.787GiB live. CRK saves one1.867GiB padded I allocation at its peak; GSPH saves two. MFM saves several overlapping row arrays, so its peak reduction exceeds the size of one accepted geometry. Reserved memory is allocator capacity, not additional live tensors. Sampled process residency includes setup/warmup and is available separately in GPU telemetry; neither measure is a guaranteed developed-flow ceiling.

All paired final fields are bitwise identical:18 stored fields for CRK/GSPH and17 for MFM, including time and source ledgers. The CRK4096-entry live work sample is bitwise identical; full live-W parity is covered separately by the small tests. Every paired forcing state is exact. All288 observed native RHS evaluations are valid with no extra RHS; all144 accepted steps are finite/positive. Maximum absolute relative energy-ledger residual is 5.323e-08. Both campaign manifests verify all1,198 source/input/test pins. No physical GPU control failed or was retried.

All eight recordings were decoded and initial/final contact sheets visually inspected. COO/CSR slices agree; startup lattice/interpolation roundoff is magnified by the narrow density colour scale. No ejection or blow-up appears. The recordings do not establish later turbulent behaviour.

|Run|Measured wall s|Loop ms/step|Steady ms/step|
|---|---:|---:|---:|
|CRKSPH-96-coo|10.551|539.0|530.7|
|CRKSPH-96-csr|8.291|527.2|519.9|
|GSPH-96-coo|6.107|344.6|339.6|
|GSPH-96-csr|6.110|344.8|340.7|
|MFM-96-coo|3.665|138.8|136.9|
|MFM-96-csr|3.602|134.9|133.1|
|MFM-96-coo-36steps|6.951|137.8|137.7|
|MFM-96-csr-36steps|6.826|134.1|134.0|

Wall time includes measured-run setup/output; separate warmup, imports/JIT and supervisor overhead are excluded. The supervisor records complete child wall time. Steady timing discards3 or6 initial steps; these are single short samples with output. They show no material slowdown but do not establish a speedup.

## Qualification and retained failures

- Core CPU neighbour suite:242 passed /334 CUDA skips; int64-offset suite:36 passed /22 skips. New CUDA allocation/order/filter/in-place tests:12 passed. An allocation spy verifies no edge-sized I allocation rather than merely dropping a reference afterward.
- Bounded frontend CPU suite:113 passed /7 skips. CompSPH storage/Heun/grad-h/replay/AD controls:15 passed. FP64 storage/restart/cache suite:16 passed.
- Initial CUDA frontend suite:52 passed and2 CRK graph bitwise failures. An unchanged HEAD plus untouched shared core reproduces the same Sod2D graph failure. Eager-grid CRK used recomputed work while graph mode used expanded work, unlike the existing CompSPH restrictions. CRK now also keeps expanded work for supportSolveOnGrid/inPlaceAdjacency. Corrected complete CRK graph suite:9 passed, including forced stale-list fallback. This routing repair changes no default native trajectory.
- Public configuration/catalogue plus cached meshless CUDA checks:106 passed. An earlier CPU invocation had10 failures because these legacy tests explicitly request CUDA; those artifacts remain. Early test-fixture mistakes and two interrupted oversized CPU suites are retained separately and are not counted as completed qualification.

## Capacity and remaining scope

Dropping I halves the index-pair part of one list; it does not compact the3-float compatible-work buffer. Its padded CRK W remains5.601GiB at96³. Smaller Verlet skins remain a separate rebuild-frequency/performance tradeoff; no skin was tuned here. Other schemes can have lower persistent storage, while their peak still depends on search/geometry lifetimes.

Correction to the earlier VRAM extrapolation: the core checks a signed-int32 flat-array dimension limit, including padded capacity, even with64-bit CSR offsets. For the current buffered B7/n_h4 layout, the limit is reached around156³. Thus176³–192³ were memory-only projections, not currently executable targets. Chunked adjacency/work arrays or a different backend addressing strategy are needed before those sizes, independently of VRAM savings. No large-capacity run was attempted.

Evidence: `.tmp/csr-neighbor-20261029/manifest.json`, `status.json`, `extension/`, frozen `runtime/`, `sources.tar.gz`, and `analysis/comparison.json`. All finite workers are terminal. The new core commit is source-identical to the frozen core for the implemented API; the frontend snapshot includes the final CRK grid routing repair.
