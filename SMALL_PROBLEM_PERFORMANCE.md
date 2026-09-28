# Small-problem performance (Marrone 3.1 and similar) — findings and changes

> **Progress marker:** this plan's row in [PLANS.md](PLANS.md). Update its *Last worked* date and *Where it stands* whenever you work on this plan.

2026-09-27. Branch `perf-small-problems` in **warpSPH** and **warpSPHCore**.
RTX PRO 6000 Blackwell (188 SMs), warp 1.17, torch 2.13.

Target workload: `scripts/probe_deltaSPHMarrone.py` (Marrone 2011 §3.1 dam
break, `sun2017DeltaSPH`, symplectic Euler, Wendland2, freeSlip, english2025
mDBC, the probe's post-step mDBC hook and per-step wall-probe diagnostics),
nx = 67–70, i.e. **~11k particles (3.5k fluid), ~101 neighbours each**.

## 1. Headline

| Marrone 3.1, nx = 67, c0Ratio 40, t = 1.9 s (t* 7.68), 19 551 steps, video on | wall |
|---|---|
| baseline batch 2026-09-26 (`out_baseline_2026-09-26/m31`) | 702 s (35.9 ms/step) |
| control: this branch with `WARPSPHCORE_NEIGHBOR_LANES=1 --no-cudaGraph` | 662 s |
| this branch after round 2 (§9), defaults | 270.5 s (13.8 ms/step) — 2.6x |
| round 3 (§10: whole-step graph), `--no-pipeline` | 187 s (9.6 ms/step) — 3.8x |
| §11: + pipelined outputs (live window) | 152 s (7.8 ms/step) — 4.6x |
| **this branch, defaults** (§12: graphed diagnostics/probes, render thread; `python scripts/probe_deltaSPHMarrone.py --nx 67 --c0Ratio 40 --video --no-show`) | **108.7 s (5.6 ms/step) — 6.5x** |
| same + `WARPSPHCORE_COMPILE_GLUE=1` (opt-in, not bitwise) | 101.4 s — 6.9x |

Per step at nx = 70 (probe configuration, `benchmarks/marrone31/bench_step.py`,
400-step window near t = 0; median):

| | wall ms/step | GPU busy ms/step | kernel launches/step | host syncs/step |
|---|---|---|---|---|
| original code | 32.1 | 20.3 | 1349 | 164 |
| + multi-lane kernels (eager) | 19.1 | 3.8 | 1349 | 164 |
| + CUDA-graph RHS + sync-free mDBC + batched probes + gated markers | **9.45** | 3.9 | 1104 | 100 |
| same, bare integrator (no probe hook/diagnostics) | **5.63** (from 21.0) | 2.7 | 680 | 43 |

## 2. What the time was actually spent on (measured, not assumed)

The starting assumption was "CPU overhead dominates". Half right:

1. **The GPU kernels were latency-bound, not idle.** 14 neighbour-loop warp
   kernels (44 launches/step) took **16–18 ms/step of GPU time**, and each
   kernel's duration was *independent of particle count*: the interpolation
   kernel ran 294 / 296 / 282 µs at 4.7k / 10.9k / 28.5k particles. Every
   kernel is one thread per particle walking ~100 neighbours serially — a
   chain of dependent, uncoalesced gathers (~1 µs per neighbour). At these
   sizes the whole problem fits one wave, so run time = that chain's latency.
   A minimal hand-written neighbour sum reproduced it (106 µs thread-per-
   particle vs 13 µs with 32 lanes per particle).
2. **The CPU side was the other half**: ~1350 kernel launches/step (1300 of
   them tiny torch ops), 164 host syncs/step, and ~140 µs of Python wrapper
   cost per warp operator call (`launchOperator` → `extractStateInfo` →
   `StateAwareWarpFunction` → `wp.launch` generic overload resolution),
   spread thinly over hundreds of small costs — no single hotspot.
3. With syncs everywhere, CPU and GPU time largely **added** rather than overlapped.

## 3. Changes

### 3.1 Multi-lane ("tiled") neighbour loops — warpSPHCore `autograd/lanes.py`

An `OperatorSpec` may carry a `tiledKernel`: same physics, launched
`dim=[N, lanes]`, `block_dim=lanes`; each lane walks a contiguous slice of
particle i's neighbour range (`getIndexRangeLane` / `laneSubRange` in
`radiusSearch/grid_util.py`) and the partials are combined with a
deterministic `wp.tile_sum` (`laneSum`). The physics `Func_i` functions are
untouched: each `Func_Adjacency` gained `lane, lanes` arguments (the original
kernel passes `0, 1` and is unchanged).

Ported (the 13 kernels of the delta-SPH step): core Interpolate, Gradient,
Divergence, Covariance; warpSPH Liu matrices, countNeighbors, Barecasco
surface detection (two passes, lane-reduced cover vector in between),
velocity + density diffusion, surface-aware pressure, surface-mask dilation,
delta shift, mDBC no-penetration shift.

| interpolation kernel, 32 lanes | 10.9k ptcl | 85k | 283k |
|---|---|---|---|
| thread per particle | 535 µs | 591 µs | 1138 µs |
| 32 lanes per particle | 30 µs | 201 µs | 666 µs |

Faster at every size measured, so it is the default (`DEFAULT_NEIGHBOR_LANES = 32`,
CUDA only). Per RHS evaluation the 14 kernels went **7.18 ms → 0.52 ms**.
Numerics: identical up to float summation order (densities 3.6e-7 rel. after
one RHS; one full step: positions 7.6e-10 dx). Gradients through the tiled
kernels match (tests below). Found on the way: reading a tile value inside
`if lane == 0:` deadlocks the *adjoint* kernel (the tile adjoint has a block
barrier) — `laneSum` returns the value to every lane; read it before branching.

Opt out: `WARPSPHCORE_NEIGHBOR_LANES=1` or `warpSPHCore.setNeighborLanes(1)`.

### 3.2 CUDA-graph replay of the RHS — warpSPH `utils/cudaGraph.py`

`deltaSPH_step` is split into an eager preamble (the Verlet-list update: may
sync, may resize) and `_deltaSPH_rhs`, which `GraphedStateFunction` captures
into a `torch.cuda.CUDAGraph` (warp launches on torch's capture stream via
`wp.stream_from_torch` + `ScopedStream(sync_enter=False)`) and replays:
state tensors copied into static inputs (one `_foreach_copy_`), replay,
clones of the static outputs handed back exactly as the eager function would
have assigned them. Keyed on adjacency identity (a rebuilt Verlet list evicts
old graphs) + state tensor shapes/dtypes.

* **Bitwise identical to eager** — enforced, not hoped for: the first capture
  of each state signature is replayed and compared bit-for-bit against a fresh
  eager evaluation (every state tensor, so an unannounced in-place input
  mutation is caught too); any mismatch or capture error disables graphs for
  the run with a warning and falls back to eager.
  `WARPSPH_CUDAGRAPH_VALIDATE=always` re-validates every capture.
* Only used where the RHS is a pure function of the state
  (`_rhsIsGraphable`): no Dirichlet/forcing/update BC hooks (they bake `t`,
  `dt`), no `'derivative'` no-pen shift (`/ config.dt`), no frozen diffusion
  under a stage-indexed integrator. The scheme config must not change after
  the first step (its values are baked in at capture).
* Per operator call: 137 µs eager → 18 µs replay. Opt-in via
  `CaseSpec.cudaGraph` (default False in the library); **on by default in
  `probe_deltaSPHMarrone.py`** (`--no-cudaGraph` to run eagerly).

Prerequisite — **sync-free mDBC path** (`utils/syncFree.py`): boolean-mask
gathers/scatters on the ghost subset (`x[kinds == 2]`, `out[bIdx] = y[mask]`)
replaced by full-length index remapping (ghost-source gather; trash-slot
scatter), cached device constants instead of `torch.tensor(list, 'cuda')`,
a dead `torch.unique` removed (`mdbc/velocity.py`, `mdbc/english2025.py`,
`gravity/directional.py`, `deltaSPH/densityDiffusion.py`). Bitwise identical
to the masked form (60-step state + trajectory comparison).

### 3.3 Smaller CPU trims

* Wall-probe diagnostics: both probe columns (at the wall and 1 dx in) in
  **one** MLS call instead of two (one hash-map build + Liu launch fewer per
  step): diagnostics 4.7 → 2.9 ms/step, trajectory bitwise identical.
* warpSPH's 79 `torch.profiler.record_function` imports now use warpSPHCore's
  gated version (`WARPSPHCORE_PROFILING=1` restores the real markers — needed
  for region-level torch traces).
* Video frames (vispy): PNG `compress_level=1` for the ffmpeg-intermediate
  frames (lossless either way; encode 38 → 26 ms of a ~83 ms frame);
  `pumpEvents` skipped when `spec.show` is False (no window to repaint).

## 4. Where a full run's time goes now (nx = 67, t → 1.9, video on)

Per-step timeline of a whole run (`timeline.npy` under
`scripts/out_perf_validation/timeline_graph2/`):

| | cost | frequency | ms/step |
|---|---|---|---|
| normal step (graph replay + finalize + probe diag + hook) | ~10.3 ms | every step | 10.3 |
| video frame (vispy render ×2–3 + PNG) | ~61–66 ms extra | every 20 steps | ~3.1 |
| Verlet rebuild + graph re-capture | ~15 ms extra | every ~17 steps (1189 captures) | ~0.9 |

Captures themselves are cheap once warm (1189 captures = 6.6 s total; only the
first per signature is validated).

## 5. Remaining costs, ranked, with the next step for each

1. **`finalize` (delta shifting + no-pen shift), ~2.5–3 ms/step, eager.** It
   uses `dt` in torch arithmetic and passes it into `solveShifting`, calls
   `buildVerletList` inside the shift loop and reads two `.item()`s. Next:
   carry `dt` as a device scalar through the shifting path, hoist the Verlet
   check out, then graph it like the RHS.
2. **Video frames, ~3 ms/step at `plotInterval=20`.** Inside warpSPHPlotting:
   `updateQuantities(redraw=True)` repaints and pumps events, then `export`
   renders the scene again; the PNG encode of a 2688×768 frame is still
   ~26 ms. Next: render once per frame (redraw=False + a single offscreen
   render), and/or pipe raw frames straight to ffmpeg instead of PNGs.
3. **Probe diagnostics, ~2.9 ms/step.** ~15 separate `.item()` syncs, two
   `torch.quantile`s, a compact-hash-map build over all particles for the 42
   probe points, a masked batched `pinv`. Next: one stacked device→host
   transfer per step; probe query against the step's own adjacency.
4. **Verlet rebuild, ~15 ms every ~17 steps.** Eager radius search with syncs.
5. **mDBC post-step hook, ~0.9 ms/step.** Now sync-free, so it can be graphed
   with the same machinery.
6. **Eager operator wrapper cost (~140 µs/call)** for everything not in a
   graph — diffuse; graph capture is the better lever than micro-trimming.

## 6. Validation

* warpSPHCore: full suite 583 passed; new `tests/operations/test_neighbor_lanes.py`
  (tiled vs thread-per-particle, forward on adjacency and grid traversal,
  reverse-mode gradients, covariance). All gradcheck scripts pass.
* Pre-existing flake, not from this work: `test_implicitShiftingComparison.py::
  test_threeWayShiftComparison_allRelaxTowardUniformDensity` failed once in a
  full-suite run and passes in isolation. Its implicit-shift path is
  run-to-run non-deterministic on the **untouched original code** too (git
  stash of both repos; 4 repeats: step-8 density std 0.0083-0.0096), and an
  unlucky realisation occasionally misses the test's 0.7x bound.
* warpSPH: full suite passes apart from that flake; new `tests/test_cudaGraph.py` (sync-free
  helpers vs masked indexing; Marrone run graph vs eager **bitwise**, state and
  every diagnostic, with ≥1 capture and no fallback). One existing fingerprint
  test relaxed: `test_relaxedJacobiRegression`'s `pmean` (a 5.7e-6 cancellation
  residual of a field with mean |p| 358 — below float32 resolution of the sum)
  now bounded at float32 resolution of the field; the 8-iterate error
  fingerprint is unchanged.
* End-to-end bitwise: graph vs eager over 250 steps (3 captures); the full
  19 551-step run with validate-first-only matches the validate-every-capture
  run (see `scripts/out_perf_validation/`).
* Physics (the multi-lane kernels change float summation order, the only
  numerical change): Marrone 3.1 nx=67 to t* 7.68 with video
  (`scripts/out_perf_validation/`). The control run reproduces the baseline
  exactly; the optimised run passes every acceptance check and tracks the
  baseline P1/P2 through first impact, the P1 plateau and the P2 run-up; it
  departs in the plunging-wave phase (t* > 5.5), which is chaotic — the
  density band and "P2 back to quiescent" differ, but by less than the spread
  of the baseline batch's own jittered realisations (`m31_seeds`: band
  [0.977,1.023] … [0.997,1.004]; P2-quiescent FAIL 0.39 … PASS). Frames
  (`frames_baseline_vs_optimized.png`) show the same flow and no new fliers.

## 7. Tooling

* `benchmarks/marrone31/bench_step.py` — per-configuration subprocess, one at
  a time: mean/median wall per step, integrator / hook / diag split, and from
  a profiler window GPU busy time, kernel launches, host syncs, top warp
  kernels. Variants `lanes1` / `lanes32` / `graph`; `--video`, `--progress`,
  `--no-diag --no-hook`. Read the **mean** — periodic costs vanish in the
  median, and a 400-step window at t ≈ 0 under-samples rebuilds.

## 8. Other schemes and periodic domains (nx = 128, 16 384 particles, 2D)

`benchmarks/marrone31/bench_case.py --case <name> --nx 128` (mean wall per step):

| case (scheme) | lanes 1 | lanes 32 | + graph | GPU busy (lanes 32) | vispy frame |
|---|---|---|---|---|---|
| tgv-wc (deltaSPH, periodic) | 9.52 ms | 8.69 ms | **6.29 ms** | 2.5 ms | 31.9 ms (matplotlib 776 ms) |
| tgv (divergenceFree, periodic) | 12.15 ms | 11.25 ms | n/a | 10.3 ms | 34.6 ms |
| gresho (CRKSPH, periodic) | 15.25 ms | 15.46 ms | n/a | 10.7 ms | 31.6 ms |

* **tgv-wc**: same picture as the dam break, already CPU-bound (no mDBC, no
  probe diagnostics), so the graph is what helps. Its capture initially
  *failed* (falling back to eager, as designed): the legacy `targetDt`
  back-solve left `fixedSoundSpeed` a 0-d CUDA tensor, read back to the host
  at every RHS evaluation. Now stored as a float
  (`modules/timestep/weaklyCompressible.py`; can move tgv-wc results by ~1 ulp,
  as `c0**2` etc. now round in double first).
* **tgv (divergenceFree)**: only the core Gradient/Divergence operators are
  shared, and those went 11 ms -> 3.2 ms per step each (called once per
  pressure-solver iteration). Wall barely moves because the iterative solver
  checks convergence on the host (~310 syncs/step). Graph capture does not
  apply (data-dependent iteration count); next lever is fewer convergence
  checks (e.g. every k iterations) or a device-side loop.
* **CRKSPH (gresho and the compressible cases)**: no gain yet -- its hot kernels
  (`computeCrkSPHAccel` 1.8 ms, `computeCrkSPHdudt` 1.6 ms, `computeCRKMoments`,
  `computePsi0`, `computeCRKDensity`) are its own module kernels, not ported to
  lanes. It is the most GPU-bound of the three (10.7 of 15.3 ms), so porting
  those five kernels is the clear next step there (expected ~4 ms/step off);
  its step would then also need the sync-free treatment to be graphable.
* **Rendering**: a single-panel vispy frame is ~32-35 ms at 16k particles
  (the two-panel 2688x768 dam-break frame ~61-66 ms), i.e. ~1.6-3 ms/step at
  `plotInterval=20` -- 10-25 % of these now-fast runs. Matplotlib is ~24x
  slower per frame (776 ms).

## 9. Second round: ACSPH, CRKSPH, iterative-solver syncs, rendering

### 9.1 ACSPH (Marrone 3.1, nx = 70, `artificialCompressible`) -- 2065 -> 187 ms/step (11x)

| | wall ms/step | GPU busy | launches/step | syncs/step |
|---|---|---|---|---|
| thread-per-particle kernels | 2065 | 1800 | 53 860 | 2891 |
| + lanes | 1138 | 333 | 53 864 | 2892 |
| + adaptive eps_v checks, cached viscosity scalars | 1035 | 317 | 52 490 | 115 |
| + CUDA graph of the pseudo-iteration | 322 | 316 | (replayed) | 119 |
| + `computeWallMoment` on lanes (166 -> 8 ms/step) | **187** | 168 | (replayed) | 119 |

* The dual-time loop runs its full 200 pseudo-iterations every step here
  (eps_v ~ -3.7 against the -6 target), ~260 launches each: launch-bound.
* Syncs: `convergenceMetric` read eps_v back every iteration (5 syncs), and
  `_viscosity` read rho0 / mean(nu) / std(nu) on every RK stage (3 each) --
  constants within a real step, now cached against the step's own tensors.
  `convergenceMetricDevice` computes eps_v on the device, bitwise equal to
  the host metric (log10 in float64 of the float32 ratio: CUDA's `log10f` is
  one ulp off the CPU's; 0/2000 mismatches, test added), and it is read only
  at the check schedule's checkpoints (`acParams.convergenceCheckSchedule`).
* Graph: the pseudo-iteration body is fixed for a real step (dtau, BDF
  coefficients, k1/k2, nu, adjacency), so iteration 1 runs eagerly and the
  rest replay a graph captured once per step (`GraphedTensorFunction`,
  `utils/cudaGraph.py`); the run's first capture is checked bitwise against
  eager. With `CaseSpec.cudaGraph`; bitwise identical over a 5-step, ~1000-
  iteration comparison.
* Remaining: GPU-bound (168 of 187 ms) across the delta-SPH-family operators.

### 9.2 CRKSPH (gresho, nx = 128) -- 15.4 -> 12.6 ms/step

Ported: CRK moments, CRK density, SPH density (core), CRK accel, CRK dudt,
psi0. Warp kernel time 6.3 -> 2.2 ms/step. **Bug found and fixed on the way:**
`computePsi0_Func_i` returns `sum**(1/dim)` -- a per-lane root of a partial
sum is not the root of the sum, so the first port was wrong (19 % velocity
difference after 20 gresho steps). Split into a raw-sum function + root after
the lane reduction (grid traversal keeps its per-cell-root semantics via lane
0). Every other ported `Func_i` was audited for a non-additive return (none).
New `tests/test_neighborLanes.py` runs whole gresho / dam-break steps at lanes
1 vs 32 -- the kernel-level tests could not have caught this. Next: 
`computeCRKVolume`, `computeCompSPHBalanceTerm` (not ported) and ~4.7 ms/step
of non-warp GPU work.

### 9.3 Iterative-solver convergence checks (`convergenceCheckSchedule`)

`'adaptive'` (default): read the convergence test back only at ceil(0.50 P),
ceil(0.75 P), ceil(0.85 P) and then every other iteration, P = the iteration
at which the previous solve of the same kind first converged (recovered from
the device-side history, so overshoot does not feed back); first solve and
verbose runs check every iteration; `'every'` restores the old behaviour.
Applied to the two relaxed-Jacobi divergence-free loops
(`modules/incompressible/divergenceFree.py`), the omniIncompressible solver
used by `tgv`, and the ACSPH dual-time loop. Per-iteration statistics stay on
the device (gathered through a once-per-solve fluid index, bitwise the old
`x[fluidMask]` values) and are read once after the solve.

On `tgv` it does not move the wall clock (10.7 vs 10.8 ms): with lanes the
step is GPU-bound (~9.8 of 10.7 ms: the gradient/divergence operators applied
in every solver iteration), so the syncs were waiting on real work. It pays
where a loop is launch/sync-bound -- ACSPH above.

### 9.4 Rendering (vispy, two-panel 2688x768 dam-break frame)

| per frame, main thread | before | after |
|---|---|---|
| live window (`show=True`, the default) | ~83 ms | **~39 ms** |
| no window (`show=False`) | ~61 ms | **~30 ms** |

* The scene was drawn three times per frame: `updateQuantities(redraw=True)`,
  the case's `pumpEvents`, and `export`'s offscreen render. Now
  `redraw=False` (the export renders offscreen anyway; `pumpEvents` repaints a
  live window) -- and `pumpEvents` is skipped when `show=False`.
* PNG encode (~26 ms at zlib level 1) moved to a writer thread
  (warpSPHPlotting `backends/asyncExport.py`: `export(..., background=True)`,
  `waitForExports()`; bounded queue); the runner drains it when the step loop
  ends, before the video encode.
* Frames are pixel-identical to before except the run-start timestamp in the
  title. What is left is vispy itself: the offscreen render, about half of it
  re-laying-out the title text that changes every frame.

### 9.5 Two fragile tests found (not caused by these changes, now robust)

* `test_incompressibleKrylov::test_krylovFp64DoesNotWorsenResidual` compared
  one BiCGStab run in fp64 vs fp32 bookkeeping. BiCGStab stagnates on that
  operator, and a single float32 ulp in the densities moves the plateau by up
  to 30x and flips the comparison -- with the original kernels too. Now
  compares medians over the state and its +-1/2-ulp density perturbations.
* `test_implicitShiftingComparison::test_threeWayShiftComparison` is
  run-to-run non-deterministic on the untouched original code (section 6).

**Found later (2026-09-28): garbage collection during a capture.** A
collection that lands inside a graph capture finalises leftover warp
`Stream`s (`wp.stream_from_torch`), whose finaliser unregisters the stream --
a CUDA call the capturing thread may not make. The capture fails (warp error
901 / 710 in `wp_cuda_stream_unregister`), that call falls back to eager, and
`test_renderThreadFramesMatchMainThread` failed in most full
`test_runner`+`test_cudaGraph` sessions once a few more tests ran before it
(0/5 at the previous HEAD, which ran fewer). Fix: `utils/cudaGraph.py:
_captureGuard` keeps the collector off for every capture (0/10 failures
since). A render-thread/capture lock was tried first and did not help.
Also: the runner's progress row is now redrawn at most every 0.1 s -- a
redraw per step cost ~0.8 ms on the ~5 ms Marrone step, which only surfaced
once the probes' bar (`quiet=True, progress=True`) actually showed.

## 10. Third round (WCSPH): whole-step graph -- 9.7 -> 6.0 ms/step, bitwise

Marrone 3.1 probe configuration, nx = 70, `cudaGraph=True`. Every change in
this round is **bitwise identical** to the eager run: 1000-step comparisons
(state and every trajectory column, 9 captures and 13 Verlet rebuilds in the
window) against the fully eager reference.

| per step | before | after |
|---|---|---|
| whole step (RHS x2, integrator updates, `finalize`) | ~5.2 ms CPU | ~0.3 ms (one replay) |
| mDBC post-step hook | 0.95 ms | 0.05 ms (graph replay) |
| diagnostics | 2.94 ms | ~3.2 ms, now mostly *waiting* for the step's GPU work |
| **wall** | **9.7 ms** | **6.0 ms** (GPU busy 3.2 ms) |

### 10.1 Whole-step graph (`utils/cudaGraph.py: GraphedIntegratorStep`)

Captures the whole `integrator.function(...)` call; used by the runner with
`CaseSpec.cudaGraph` for the delta-SPH family (not with trajectory/state
storing). What made the step capturable:

* **Verlet checks** (three per step: before each RHS, inside shifting) --
  `warpSPHCore.deferVerletChecks()`: inside it a check records its device
  "rebuild?" flag and keeps the prior list. After each replay the OR of the
  flags is read (with the new time, one transfer); a set flag means an eager
  step would have rebuilt somewhere, so the replay is discarded and the step
  runs eagerly (the caller's state was never touched). ~1 step in 70 early in
  a run, ~1 in 17 during violent flow.
* **Time/dt** -- `dt` was already a 0-d device tensor; only the integrator's
  `float(t + dt)` stage times and a `bool(dt)` test read it back.
  `warpSPHIntegrators.util.deferHostTime()` skips those inside a capture; the
  wrapper sets the final time with the integrator's own float32 expression.
* **finalize / shifting syncs** (38 per step -> 0): the step statistics
  (`stepDiagnostics`) are computed lazily from stashed raw tensors when the
  diagnostics read them; the no-pen branch, the shift velocity cap and clamp,
  the finite-velocity maxima are branchless `torch.where` forms; `spacing`
  and the clamp bounds stay on the device (bounds formed in float64, as the
  host did); `dx` is read once per run; two dead host reads removed (`dt_c`,
  and `c_max`, which the delta-shift kernel never uses).
* The first capture of a run is compared bitwise against an eager step
  (state, last-stage update, step statistics, time).

### 10.2 Smaller exact savings

* mDBC post-step hook: running it only on output steps was **rejected** --
  it does affect the dynamics (boundary densities persist between steps and
  `finalize`'s shifting reads them; 250 steps without it: fluid positions
  differ by 7e-7, pressures by 3e-3). Instead its english2025 recompute is
  replayed from a graph per Verlet generation.
* Grid-traversal tiled kernels: lanes take whole cells round-robin instead of
  splitting each (few-particle) cell 32 ways with every lane repeating the
  hash lookup; english2025's neighbour-count query is back to the ghost rows
  only (cached ghost index). Liu kernels 0.99 -> 0.38 ms/step.
* Verlet check: when query and reference are the same particle set, the
  second (identical) half is reused. Diagnostics: fluid-fluid edge pairs
  cached per adjacency; scalar reads batched into one transfer.
  `buildCompactHashMap`: grid scalars read in one transfer, the unused
  reference hash and an explicit `wp.synchronize()` removed.

### 10.3 What is left, and why fusing the glue (item 6) is not next

GPU work is 3.2 ms of the 6.0 ms step: ~1.4 ms in ~1040 small torch kernels,
~1.4 ms in the 41 warp kernels, 0.3 ms sorts. But the GPU is idle ~2.8 ms per
step, because the diagnostics (probe hash build, MLS fit, quantiles; ~1.5-2 ms
of CPU) run only after waiting for the step to finish. Fusing glue kernels
(Triton or warp) would shave <=0.5-1 ms of GPU time -- which is not on the
critical path yet -- and would change rounding (FMA contraction), ending the
bitwise identity. The larger, exact lever is **overlap**: launch step n+1,
then compute step n's diagnostics on a side stream while it runs (values
unchanged, only concurrent) -- expected toward ~3.5 ms/step.

## 11. Overlapping each step's outputs with the next step -- 6.3 -> 4.7 ms/step, bitwise

`runner.py:_runPipelined`, used whenever the whole-step graph is
(`CaseSpec.pipelineOutputs`, default on; `--no-pipeline` on the Marrone probe,
variant `graphSeq` in the bench scripts for the old loop). With step n just
finished, per iteration:

1. post-step hook, next `dt`, stall watchdogs, stop conditions, the
   non-finite check -- the things that decide whether and how step n+1 runs;
2. **launch** step n+1 (`GraphedIntegratorStep.launch`: inputs copied into
   the graph's buffers, replay enqueued; the rebuild flag and new time are
   only stacked into one device tensor);
3. step n's diagnostics row, progress text and video frame, on a side CUDA
   stream (warp scoped to the same stream) that waits only on an event
   recorded *before* the launch -- so their GPU work and their host reads
   do not wait for step n+1;
4. **finish** step n+1 (`finish`: one read of flag + time; on a set Verlet
   flag the step is redone eagerly from step n's untouched state, as before).

Step n's state is clones the replay never writes, so the outputs see exactly
what the sequential loop showed them. A step that ends the run (tLimit,
nSteps, non-finite) gets its row and final frame before the loop exits; a
stall stop records the row without a frame -- both as before.
`stepTime_ms` becomes the per-step wall throughput (launch -> finish spans
the overlapped outputs), not the isolated step time.

**Checks.** 1000 steps (9 captures, 13 rebuild fallbacks), video on:
final state bitwise, all 1001 trajectory rows identical, all 22 frames
pixel-identical except the run-start timestamp in the title.
`tests/test_cudaGraph.py::test_pipelinedOutputsMatchSequentialLoop`
(time-limited run: same rows, same stopping step, same state). Full suite
passes. Full probe run (nx = 67 to t* 7.68, 19 551 steps, video on;
`scripts/out_perf_validation/full_{no-pipeline,pipeline}/`): every column of
all 19 552 rows identical, all 979 frames pixel-identical apart from the
title's run-start time; **187 s -> 152 s**.

**Timing** (nx = 70, 1000 steps after 100 warm-up, `bench_step.py`):

| | sequential (`graphSeq`) | pipelined (`graph`) |
|---|---|---|
| no video | 6.28 mean / 5.92 median | **4.69 / 4.36** ms/step |
| video every 20 steps + progress rows | 7.90 / 6.45 | **6.15 / 4.69** ms/step |

GPU busy is 3.5 ms/step, so the median step is now within ~0.9 ms of
GPU-bound. What remains on the wall clock:

* **Video frames** (mean - median, ~1.5 ms/step at every 20 steps): a frame
  is ~30 ms of host work (vispy render), and only the ~3.5 ms of step n+1
  can hide under it. Hiding the rest needs rendering off the loop thread
  (a render thread holding the GL context, fed host copies) -- possible,
  not done.
* **GPU time** itself: now the critical path. That is where fusing the
  elementwise glue (item 6: ~1040 small torch kernels, ~1.4 ms) pays --
  at the price of the bitwise identity with eager.

## 12. Fourth round: the outputs were the critical path -- 4.7 -> 3.1 ms/step exact, 2.8 with compiled glue

With step n+1 overlapped (§11), the loop runs at the pace of whichever is
longer: the step's GPU work (~2.2 ms) or step n's outputs on the host. The
outputs were: diagnostics 3.4 ms per step (wall probes 1.4, distribution
metrics 0.8, step statistics 0.6, the rest 0.6). So "fuse the step's GPU
glue" (item 6) could not move the wall clock yet -- measured: compiling two
glue regions cut 190 kernels and 0.27 ms of GPU time per step and left the
wall time unchanged. Order of work therefore:

### 12.1 Diagnostics replayed from a graph (exact)

* `cases/dambreak.py:_diagnosticsDevice` -- every per-step scalar as a device
  tensor with no host sync; `utils/cudaGraph.py:GraphedDiagnostics` replays
  it per Verlet generation (inputs: the state's tensors and the step
  statistics' raw tensors; first call on a new neighbour list eager, to warm
  the caches; first replay of a run validated bitwise), read back in one
  transfer, grouped by dtype. Set up by the runner with the whole-step graph.
* Exact sync-free replacements: fluid rows through the cached kind index
  (`index_select`, bitwise the mask gather); "finite values only" as NaN mask
  + `nanquantile` (same sorted prefix, same rank formula); fractions as exact
  integer counts x (1 / count), which is how torch's mean divides; the no-pen
  magnitude block (data-dependent size) eager only on steps where the
  correction touched a particle.
* `utils/syncFree.py:sortedQuantiles` -- several quantiles from one sort,
  op-for-op ATen's `quantile_compute` (random tests incl. NaN/inf: bitwise);
  a positive rescaling of a sorted array is still sorted, so the
  distribution metrics' three quantiles share one sort. 12 -> 6 sorts.
* Found on the way: `_fluidPairs`' per-adjacency cache never stored anything
  (`AdjacencyList` is slotted; the `setattr` failed silently), so every call
  masked the full edge list with a sync. Now a module-level cache keyed on
  the edge tensors.

### 12.2 Pressure probes on the Verlet list's hash map

The probes built their own hash map over all particles every step (several
host syncs, ~60 kernels). A fixed query point's fluid neighbours are all
found through the Verlet list's hash map while the list is valid (a
reference within h now was within verletScale x h at the build) -- the
argument the mDBC ghost query already relies on -- so the probes use it
(`_probeHashMap`; not for a widened gather or a periodic domain) and the fit
runs sync-free (`interpolateLiuLiu(syncFree=True)`: `inv_ex` + `where`
instead of a masked `pinv`, which cannot be captured) inside the diagnostics
graph. The only change in numbers: the probe columns, at float rounding
(candidate order, inv vs pinv) -- see the full-run check below. The
fresh-hash path is unchanged and bitwise the old code.

### 12.3 Compiled glue (`WARPSPHCORE_COMPILE_GLUE=1`, opt-in, not bitwise)

`warpSPHCore.compileGlue`: a decorator that runs a pure-torch helper through
`torch.compile(fullgraph=True)` when the switch is on (pass-through when off;
any compile failure -> eager with a warning); `markDynamic` for edge-sized
arguments (else the first Verlet rebuild recompiles inside the timed loop).
Applied to: the Verlet validity metrics (24 kernels -> 1), english2025's
boundary-density tail, the slip-velocity combination, three shifting pieces
(tangential restriction, lambda gate, cap + clamp), and the diagnostics'
distribution/step-statistics cores. Compiled kernels capture into the CUDA
graphs like any other; one trap: a host-float `dt` (eager step) vs a device
`dt` (captured step) specializes -> recompiles inside the capture -> the
capture fails; the shifting cap now always receives a tensor when compiling.

### 12.4 Frames

* Colorbar: `cmap` / `clim` were re-assigned every frame (a new colormap
  object regenerates the colorbar shader and creates textures). Now set only
  when changed (warpSPHPlotting). Pixel-identical; 29.7 -> 24.2 ms/frame.
* Render thread (`CaseSpec.asyncPlot`, default on; only without a live
  window, `show=False`, vispy): the plot hooks run on one worker thread that
  owns the plot through EGL; each frame renders a device snapshot of the
  state; <= 2 frames queue. Pixel-identical frames (EGL vs glfw too). It
  recovers ~40 % of a frame's cost -- the frame's work is Python (vispy), and
  the GIL, not the switch interval (tried 5 ms .. 50 us: no change), limits
  the overlap. A render *process* would recover the rest but needs a
  data-only plotting protocol. `--no-show` on the Marrone probe.
  Trap found by the first full run (not by the short tests, where no frame
  happened to coincide with a capture): torch captures in `'global'` mode by
  default, which forbids unsafe CUDA calls from every thread while a capture
  is underway -- a frame's device->host copy on the render thread failed and
  invalidated the loop thread's capture. All captures here now use
  `capture_error_mode='thread_local'` (`utils/cudaGraph.py:_CAPTURE_MODE`);
  the render thread only touches its own non-blocking stream.

### 12.5 Results (nx = 70, 1000 steps after 100 warm-up, mean / median ms/step)

| | no video | video every 20 steps |
|---|---|---|
| §11 (pipelined) | 4.69 / 4.36 | 6.15 / 4.69 (window) |
| + graphed diagnostics | 3.95 / 3.59 | 5.62 / 4.08 (window) |
| + probes on the Verlet hash, shared sorts, packing | **3.09 / 2.74** | 3.98 / 2.77 (window), 3.85 (no window), **3.62** (no window, render thread) |
| + compiled glue (opt-in) | **2.77 / 2.44** | |

Host syncs per step: 59 -> 9.

### 12.6 Full-run checks (nx = 67 to t* 7.68, 19 551 steps, video, `--no-show`)

`scripts/out_perf_validation/full_r4_{exact,compile}/`.

* **Exact mode, 152 s -> 108.7 s.** Every non-probe column of all 19 552
  rows bitwise the §11 run; all 979 frames pixel-identical (render thread +
  EGL vs the main thread's glfw window) apart from the title's start time.
  Probe columns (Verlet-hash fast path): neighbour counts identical
  everywhere; max |diff| in rho0 g H units P0 6e-6 (peak 1.08), P1 2.5e-4
  (peak 0.29), P2 3.5e-4 (peak 14.9) -- the first-order wall fit amplifies
  rounding (the near-wall MLS extrapolation is the sensitive reading already
  noted for P1), still far below any acceptance band.
* **Compiled glue, 101.4 s.** Tracks the exact run through t* 2.5 (P0 within
  1e-3, maxVelocity within 5e-4), then diverges as a chaotic flow does: same
  bulk flow in the frames, same ceiling/wall-hovering fragments. In this
  realisation one near-wall fragment is kicked late (maxVelocity 28.5 m/s at
  t* 6.1, a P1 spike of 84) -- the isolated-fragment kick of OPEN_PROBLEMS.md;
  the jittered baseline realisations show the same mechanism (late maxV
  11-21 m/s, P1 spikes to 13), but this one is above that range. One
  realisation cannot tell variance from a compile effect, so compiled glue
  stays opt-in until a jittered batch compares both.

Headline now: Marrone 3.1 nx 67 full run **702 s -> 108.7 s (6.5x)**, exact
(bitwise simulation; probe columns at rounding), 101 s with compiled glue.
