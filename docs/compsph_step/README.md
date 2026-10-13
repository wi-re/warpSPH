# CompSPH / CRKSPH step cleanup -- state at 2026-10-13, with the evidence

Written for the person (or the next session) who continues this work: what was done, why, what was measured, which
bugs it found along the way, and what is open. Everything below is committed; commit hashes are given per item.
Evidence files are in `evidence/` (call-audit tables, the scripts that produced the numbers). Re-run an audit with

    python scripts/audit_case.py sod2d --scheme CompSPH --steps 40 --warmup 10 [--scheme-set stepMode='graph']

All numbers are `sod2d` (2,500 particles, 2D, B7, RK2) unless a table says otherwise, 30 marked steps after 10 warm-up,
CUDA, float32. "ops" = aten calls, "launches" = warp launches, "syncs" = host reads; a CUDA-graph replay hides the ops it
contains (one replay counts as one launch).

## 0. Where it stands

CompSPH and CRKSPH are the proving ground for the ACSPH step (user, 2026-10-13: "get X6 working here, then take the lessons
back to ACSPH"). Per integrator step on `sod2d` CompSPH:

| Variant (audit file in `evidence/audits/`) | ops | launches | syncs |
|---|---|---|---|
| start of the day: rebuild every stage, per-stage pair storage (`compsph_sod2d_before`) | 975 | 22.0 | 24.0 |
| X6: one running pair sum, list reused across stages (`..._after_pairwork`) | 647 | 15.2 | 15.8 |
| validity check as two warp launches (`..._fusedValidity`) | 484 | 26.9 | 15.4 |
| support solve validates the list once, device early stop, no host reads (`..._lean`) | 497 | 21.2 | 5.4 |
| closing summation density dropped for the in-step caller (`..._noFinalDensity`) | 487 | 19.2 | 5.4 |
| support solve on the fixed-shape grid, explicit check per stage (`..._gridSolve`) | 466 | 18.9 | 5.5 |
| + deferred list check, one host read per step (`..._gridDeferred2`) | 446 | 21.8 | 3.9 |
| + step-end rebuild decision learns the buffer usage (`..._gridAdaptive`) | 443 | 21.6 | 3.7 |
| + host syncs of the runner loop removed (`hostTime.txt`; integrator step only: 441 ops, 1.73 syncs) | 441 | 21.6 | **1.73** in the step, **4.8** in the whole loop (16.1 before) |
| + the whole step replayed from one CUDA graph (`stepGraph.txt`, **current**, `--cudaGraph`) | 22 launches / step recorded, the step's own ops are inside the graph | | 3.35 in the whole loop: the replay's one read (1), the rebuilds (0.55 x 2), diagnostics (1), time + max speed (1) |

Wall time is NOT better in the grid mode at this size: 300 steps take 2.62 s (grid, deferred) against 2.52 s (list) at N = 2,500.
The step is launch-bound; the grid mode adds two graph replays and a few small ops per stage. The saving is in syncs and in what
a whole-step graph can later contain, not in eager time. `supportSolveOnGrid` is therefore **off by default**.

Host-sync pass (2026-10-13, `evidence/audits/hostTime.txt`, bitwise identical to the previous commit on sod2d CompSPH and CRKSPH, ACSPH dambreak eager and graph:
`evidence/scripts/hashCase.py`):

* The eager runner call runs under `deferHostTime()` (the mechanism the whole-step graph already used): the integrator no longer reads
  `float(t + c dt)` back per stage (2 syncs per step); the runner reads the step time once anyway.
* **Every system's `initializeNewState` formatted `f'... [t={self.t}]'` before `verbosePrint` looked at `verbose`** -- once `t` is a device
  tensor that is a sync per call (3 per step for a 2-stage step), silently, even with `verbose` off. Now `if verbose:` in all nine systems.
* `compressibleDiagnostics`: eight `.item()` reads became one stacked transfer (8.3 -> 1 per step).
* The step time and `max |v|` are read in one transfer (`_scalars`) instead of two.

Remaining syncs per step, whole loop 4.8: 1.0 step-end list flags, 0.6 rebuilds (edge-total read), 0.13 grid-radius flag (every 8 steps), 1.0 case
diagnostics, 1.0 time + max speed, 1.0 step timer (a measurement, `cuda synchronize` for the CUDA events). The integrator's own state copies and
stage combines are ~108 ops per step (25 % of all ops), in no compSPH section.

## 1. Commits (trail)

warpSPH (`dev`): 2d379a8 adjacency rename (frontend) / b1872e3 X6 pair sum / ffe6d2c list chained across stages + early stage-0
rebuild / 948b84a support solver validates once / f95f65b NoScheme fix / ae222f2, 03b2c81, 2cb8662 Psi0 / 825ccc7 no closing density +
out-of-bounds fix / c5fefb8 wall clamp module + grid solve / 700f8cf deferred check / f5b03a9 forceRebuild / 339e0b5 learned usage.
warpSPHCore (`dev`): d733192 padded adjacency / d2cd60b `inPlace` / 09b34f0 `rebuildSlack` / 07403b5 fused validity kernel /
3ea04fc fixed-shape hash map (slots) / f853f97 `verletStaleFlag`, `gridFactory` / 958ec05 `forceRebuild` / 8a67521 `verletUsage`
(+ plan commits: c266663, 67e27d7, c28a378, 3a12423, b249e02, f24ce3b).
warpSPH, later the same day: 9b1d312 host-sync pass / 4edc3e1 padded per-edge arrays / fb9e4e8 the CompSPH step graph (+ the Gather-grid fix).
warpSPHCore, later: a0b75bd `GraphedBuild` inline under a capture, `WARPSPHCORE_AUDIT_OPSITES` / bd103d0 in-place rebuild writes the reference snapshot too.
warpSPHIntegrators (`main`): b9cbacb `RungeKuttaB` stage weights + second pass / c5daded `update_position` no truth test of a tensor dt.
NOTE: the integrators repo has other uncommitted work that is not part of this (new tableaus, docs); only the two hunks above were committed.

## 2. Changes, by theme, with evidence

### 2.1 Adjacency contract (core d733192, d2cd60b; frontend 2d379a8)
`AdjacencyList.i/.j` renamed `paddedI/paddedJ` so that every use fails loudly; `numEdges` (host int), `livePairs()` (exact views),
`CapacityPolicy(headroom=0.1, shrinkBelow=0.5, inPlace=False)`, padding `-1`, default on in the compact-hash search, `buildVerletList`,
`filterVerletList`. `inPlace=True` makes a rebuild that keeps the capacity overwrite and share the replaced list's buffers (same
addresses for a captured graph), opt-in and not used anywhere yet. ~30 consumers in both repos moved.
Evidence: warpSPHCore 830 -> 847 tests pass (`tests/radiusSearch/test_adjacencyCapacity.py`, 19 tests incl. inPlace pointers and
contents against an exact build); warpSPH full suite passes (~19 min wall; the README's "2 minutes" was stale and is corrected);
bitwise identical to the previous commit on ACSPH `dambreak` nx 70, 55 steps, eager and graph (hashes in `evidence/scripts/bitwiseHash.py`).
Hidden-failure sites the rename could not catch (string attribute access) were found by grep and fixed: `hasattr(adjacency, 'i')` in the
distribution diagnostics (it would have silently returned no metrics) and `getattr(adjacency, 'i', None)` in the nopen boundary normal.
The identity-keyed `_fluidPairs` cache in the diagnostics was removed (on-device masking, `tests/test_distributionMetricsPadding.py`).

### 2.2 Fused validity check (core 07403b5)
`_verlet_validity_metrics` was ~44 torch ops per call (180 calls per 30 CompSPH steps = 38 % of all ops). Now two warp launches
(`radiusSearch/verlet/wp_validity.py`), 6 ops. 24 test variants against the torch reference (`tests/radiusSearch/test_verletValidity.py`),
200-step CompSPH/CRKSPH trajectories hash-identical. Ops 636 -> 484 per step.

### 2.3 X6: compatible energy without per-stage pair storage (frontend b1872e3, ffe6d2c; integrators b9cbacb)
`modules/compSPH/pairWork.py`. The update `deltaU_i = sum_s k_s sum_j f_ij v_ji . (ap+av)` with the half-step velocity
`v_half = v0 + dt/2 sum_s k_s dvdt_s` (known only after the last stage) factors as `sum_j v_ji . W_ij`, `W_ij = sum_s k_s f_ij(ap+av)`:
each stage folds its pair vector into one running `(E, dim)` buffer, `energyIncrement` closes it. Frontiere et al. 2017 Sec. 3.3
(p. 170) names the two options, retain or recompute the pair accelerations, and derives neither. The running sum needs one pair list;
a stage on another list sets `consistent=False` and `finalize` returns a `RerunRequest`: `RungeKuttaB` runs the whole step again
from the untouched initial state and each stage then adds its own share with the known `v_half` (no pair storage). User decisions: freeze the
list over a step, accumulate for every stage with `k_s != 0`, no recompute option, mid-step rebuild = the second pass.
Findings: (a) stage systems are copied from the step's INITIAL state, so a stage-0 rebuild was repeated at stage 1 -> `startFromLatest`;
(b) midpoint has `k_0 = 0` and stores nothing across stages, so it never needed a second pass (that hid (a) at first); Heun / SSPRK3 needed
one in ~20 % of `sod2d` steps; (c) an early rebuild at stage 0 (`rebuildSlack` in `buildVerletList`) removed them.
Evidence: 17 `tests/test_compSPHPairWork.py` tests (forced second pass reproduces the running sum to 1e-4, energy conserved, no per-stage
arrays retained, all support schemes valid, sedovWalls smoke, grid vs list equivalence); `evidence/scripts/heunRerun.py`.

### 2.4 Support solve (frontend 948b84a, f95f65b, 825ccc7, c5fefb8)
`evaluateOptimalSupport(validateOnce=True)`: the Owen iterations run on the incoming list unchecked (only `h` changes) and the list is
validated ONCE afterwards against the largest support any iteration used (rebuild + redo if short); the early stop is a device flag, not a
host read per iteration; the wall volume clamp no longer needs a host read. `computeDensity=False` skips the closing summation density
for the in-step callers (they recompute density in their own sense right after).
Cost of dropping the closing density: -10 ops, -1 launch per stage, ~8 % wall in a 200-step run, results bitwise identical where the clamp is
off. Where the clamp is on (wall cases) it uses the current density instead of the freshly summed one: bulk quantities move <= 0.2 %,
the maximum support in `sedovWalls` moves 0.2495 -> 0.2105 (`evidence/scripts/clampCompare.py`). So the clamp became its own step:
`modules/compressibleWall/supportClamp.py`, called by the CompSPH/CRKSPH/GSPH steps after the list (it only reduces h, so the list stays
valid); it computes the summation density it needs itself, bitwise equal to the old in-solver behaviour (`tests/test_supportClamp.py`).
`supportVolumeClamp` modes unchanged (`'walls'` default).

### 2.5 Fixed-shape grid and the grid solve (core 3ea04fc; frontend c5fefb8, 700f8cf, f5b03a9, 339e0b5)
`warpSPHCore/radiusSearch/compactHash/slots.py`: `HashMapSlots` (grid built for a support RADIUS fixed at construction, host-constant cell
counts, radius-1 stencil, everything sized N, padded tail sorts last), `buildFixed` (108 torch ops, 0 syncs, no data-dependent shapes),
`GraphedBuild` (captured once, replayed). Same grid as the eager build gives the same CSR order row by row (9 tests, CPU+CUDA, incl. a graph
capture/replay across moving particles). Measured: N = 2.5k eager build 0.87 ms -> graphed replay 0.19 ms; N = 100k 0.94 -> 0.37 ms;
163 ops + 4 syncs -> 0 syncs. (`evidence/scripts/benchSlots.py`)
`supportSolveOnGrid=True` (CompSPH, Owen only): per run a `SupportGrid` holder (shared by reference through `initializeNewState`, like the
pair work); every stage copies positions / supports into the slots and replays the build; the solver limits every iterate to the grid radius
(`info['outgrown']` device flag, read every 8 steps; the next grid gets a larger radius); no list is looked at inside the solve.
The pair list: per stage a device flag `verletStaleFlag` (no host read, appended to `pairWork.flags`); at `finalize` ONE host read returns
the in-step flags, the "list grid was too small" flags and the buffer usage `verletUsage` together; any in-step flag -> `RerunRequest(
conservative=True)` (second pass validating every stage); otherwise the list is rebuilt NOW if `usage + 1.25 x per-step usage >= 1`
(`rebuildNeeded`, the per-step usage is learned), so the next step starts on a list that needs no check. Rebuilds search a second graph-
replayed fixed grid (`pairListGridFactory`, built for the Verlet-scaled supports) instead of the eager hash-map build: 228 ops + 6 syncs ->
69 ops + 2 syncs. `buildVerletList(forceRebuild=True)` rebuilds without a second check and keeps the prior list's capacity.
Rebuild count (300 steps): fixed half-buffer margin 102; learned usage 75 (~7.5 per 30 steps, the list-solve rate); margin 0 -> 44 second
passes. Zero second passes for RK2, Heun, SSPRK3, RK4 (`evidence/scripts/slackScan.py`, `countGrid.py`). Equivalence to the list solve: mean
h, u, rho agree to 8 digits on `sod2d`, `sod`, `sod3d`; `tests/test_compSPHPairWork.py::test_gridSolveGivesTheListSolveResult` (1e-3,
round-off grows over 40 steps because the neighbour order differs).

### 2.6 The whole CompSPH step from one CUDA graph (core a0b75bd, bd103d0; frontend 4edc3e1, fb9e4e8)

Opt-in through ONE switch, `CompSPHConfig.stepMode = 'graph'` (2.7; the case spec's `cudaGraph` flag is not used for CompSPH). The mode turns
on the grid solve, `inPlaceAdjacency` and the graph together, and refuses (eager + warning) with wall rows, a viscosity switch with a
step-boundary host time, stored trajectories or no CUDA (`modules/compSPH/stepMode.py`). `GraphedIntegratorStep` is the machinery the ACSPH / delta-SPH steps already
used; what CompSPH needed:

1. **Fixed shapes.** The per-edge arrays (`ap_ij`, `av_ij`, `f_ij`) and the X6 pair sums use the PADDED capacity (`paddedI/paddedJ`, the `-1` tail
   is never read: `segment_sum` takes the first `numNeighbors` entries of each row), not `livePairs()`. Bitwise identical to before.
2. **The grid build is recorded inline** inside an outer capture (`GraphedBuild.replay`: a graph cannot replay inside a capture).
3. **No host read inside the step.** `finalize` registers a `StepDecision` (`utils/stepDecisions.py`) instead of `.tolist()`: the device values
   (in-step flags, list-grid-too-small flags, a support-grid limit was binding, buffer usage). The wrapper reads them with the step time in
   ONE transfer per replay and then (a) `redo`: a flag says a stage ran on a pair list that was too short: the replay is discarded and the step
   runs eagerly from the untouched input (the second-pass machinery), (b) `apply`: learns the usage, enlarges the support grid if the limit was
   binding, and REBUILDS THE PAIR LIST for the next step eagerly (rebuild stays outside the graph, as decided).
4. **In-place rebuild.** `inPlaceAdjacency` (frontend switch, set by the runner) makes the rebuild write the old list's buffers; the core now also writes
   the list's reference snapshot (`queryPositions` ... `referenceSupports`, what the validity check compares against) in place, because a captured
   check reads them at fixed addresses (`test_capturedValidityFollowsAnInPlaceRebuild`). The capture is keyed on the ADDRESSES and shapes of the
   list's arrays (and the grids' identities), not on the list object: a rebuild that keeps its capacity does not recapture; one that outgrows it does.
5. **The first replay is compared with an eager step** (as before), now from a COPY of the list (the eager step may rebuild in place) with the
   host-side learned state saved and restored (`StepDecision.hostState`).

Evidence: `sod2d` CompSPH, 300 steps, ~75 in-place rebuilds, 2 captures, 299 replays, 0 fallbacks, final state bitwise equal to the eager grid run
(`tests/test_compSPHGraph.py`, also with the rebuild margin removed so replays are discarded and redone eagerly: still bitwise equal). ACSPH
`dambreak` and delta-SPH graph runs unchanged against the previous commit (`evidence/scripts/hashCase.py`).

| `sod2d` CompSPH, ms per step (200 steps) | list solve (default) | grid, eager | grid, graph |
|---|---|---|---|
| N = 2,500 | 2.70 | 2.91 | **1.22** |
| N = 5,000 | 2.64 | 3.12 | 2.20 |
| N = 7,500 | 3.17 | 3.49 | 3.01 |

The saving is the host overhead (small N); with N the kernels dominate and the graph no longer matters. The loop around the step (diagnostics,
time + max-speed read, the replay's own read) is still 3 syncs per step.

Two things that bit on the way, both from state the capture cannot see: **(a)** a captured reduction over flag tensors that earlier EAGER steps had
left in `holder.pending` read freed memory after the Python references were dropped (alternating 0/1 'support grid outgrown' -> the grid was
regrown every other step; fixed: a recorded step clears the pending lists first and only reads flags it computes itself); **(b)** sizing a grid reads
a device value, so the pre-capture hook `prepareCapture` makes it before the recording starts.

### 2.7 One switch: `stepMode` = eager | graph | auto (2026-10-13, user)

The graph is only worth it while the host is the bottleneck. `sod2d` / `sod` / `sod3d`, CompSPH, ms per step averaged over steps 100-1100 (same
GPU, RTX PRO 6000 Blackwell; list = default eager list solve):

| dim | N | list | graph |
|---|---|---|---|
| 1D | 625 / 10k / 40k / 160k / 640k | 4.9 / 5.0 / 4.9 / 5.1 / 8.6 | 0.8 / 1.3 / 2.0 / 5.2 / 18.3 |
| 2D | 2.5k / 5k / 7.5k / 10k / 15k | 5.3 / 5.4 / 5.6 / 5.8 / 6.0 | 4.0 / 5.1 / 5.9 / 6.4 / 8.0 |
| 3D | 3.0k / 4.5k / 6k / 8k / 10k | 14.8 / 18.1 / 19.8 / 21.7 / 21.9 | 22.9 / 33.8 / 39.0 / 47.0 / 29.1* |

(*short window.) The first 300 steps of the 2D case are faster than the later ones (steps 100-1100 include the developed shock): at N = 2.5k the
graph is 2.2x the list over the first 300 steps and 1.3x over 100-1100. The grid solve is the cost: it visits a 3^dim cell stencil per
iteration where the list solve walks a precomputed neighbour list, so the graph's saving (host overhead, constant per step) is overtaken as soon as
the kernels dominate, and in 3D (27 cells, more neighbours) the step never gets there. Rebuilds and redo steps cost O(N) each but happen at a
rate set by particle motion, not by N, so they do not move the crossover.

Therefore a single switch, `stepMode`: `'eager'` (default; no deferred flags, no grid, no in-place buffers), `'graph'` (all of them together),
`'auto'` (graph below `GRAPH_MAX_PARTICLES[(scheme, dim)]`: CompSPH 1D 100k, 2D 5k, 3D never; override `stepModeMaxParticles`). The
crossovers are for this GPU and these cases; they are first estimates. Only CompSPH has a graph mode; CRKSPH and the others are always eager.
The grid-eager combination (`supportSolveOnGrid=True` with `stepMode='eager'`) stays available as a diagnostic and as the bitwise reference of
the graph tests; it is slower than both.

### 2.8 Eager-path savings in the comp path and what the graph mode still costs (2026-10-13)

* **Multi-lane kernels for CompSPH** (`computeCompSPHAccel`, `dudt`, `BalanceTerm` got `_KernelTiled` variants, ported from the CRK ones: lane-sliced
  neighbour ranges + `laneSum`; `WARPSPHCORE_NEIGHBOR_LANES=1` restores the old kernels). They ran one thread per particle: in 3D at N = 6k the accel
  kernel took 5.4 ms per launch (~8 ns per pair, 2 % occupancy). Eager list mode, ms per step: `sod3d` N = 6k **25.3 -> 7.6**, N = 8k 27.1 -> 9.5;
  `sod2d` 2.5k 5.5 -> 5.0, 15k 6.3 -> 5.8; `sod` unchanged (launch bound). Results equal to round-off (sum order).
* **Graph mode refuses Dirichlet boundary conditions**: `rayleighTaylor` / `kidder` / `yee` evaluate boolean-mask indexing in the BC (host-sized). Worse, a
  capture that fails at the CUDA level (not a Python exception) leaves the next eager step non-finite -- not understood, only avoided; a forced Python
  failure inside a capture recovers cleanly (`captfail3`). Treat 'capture failed' as untrusted until that is understood.
* **Support-grid headroom 1.5 -> 1.15** (`supportGrid.RADIUS_HEADROOM`): sod2d graph N = 2.5k 1.9 -> 1.3 ms, N = 10k 5.1 -> 3.1 ms; the number of regrows
  (recaptures) on sod2d, gresho, KH, noh, sedov, triplePoint did not change in a way that matters (sedov 6 -> 7 at 1.1). 3D is unaffected by the headroom.
* **3D graph mode is dominated by `computePsi0` on the grid**: 12 of 17.8 ms at N = 3k (5.7 ms per call vs ~0.2 ms on the list). Not the headroom (tried 1.1-1.5,
  no change); the cell stencil / the kernel's traversal is the suspect. Open.
* Sweep (`graph` request vs the grid-eager run, 40 steps): gresho, hydrostatic, KH, linearWave, noh, shearBox, shearingNoh, triplePoint, woodwardColella
  bitwise equal; sedov equal to 6e-12 in positions (last-bit rounding) with 5 captures; bowShock, closedBox, forwardStep, piston, shockCylinder,
  shockReflection (walls), kidder (Dirichlet) run eagerly with a warning.
* The crossover table in 2.7 was measured before these changes, on one GPU: the 2D graph limit has moved up (graph still ahead at 10k with the new kernels)
  and the numbers are hardware specific. Treat `GRAPH_MAX_PARTICLES` as a rough, conservative default, not a result.

## 3. Bugs found on the way (each with its test)

1. **Psi0 on a grid / no adjacency was 1.6-1.9x the list value** (`wp_psi0.py`): `computePsi0_Func_Adjacency` took the dim-th root of each
   CELL's partial sum and added the roots; the root is not additive. The production Owen solver passes a Verlet list (one call per
   particle) and was unaffected, any `adjacency=None`/`CompactHashMap` call was silently wrong. Fixed with the root taken once, in a wrapper
   AROUND the raw loop function (warp cannot differentiate a non-linear step on a value accumulated in a dynamic loop of the same function;
   `scripts/gradcheck_adaptiveSupport.py` passes). `tests/test_psi0Traversal.py` (4 variants fail without the fix).
2. **Psi0 grid path ran on lane 0 only** (tiled kernel): 0.43 ms vs 0.14 ms list at N = 2.5k, 10.5 ms vs 0.29 ms at 100k. Now lane-parallel:
   0.13 ms / 1.9 ms. No other tiled kernel has this pattern (scanned).
3. **NoScheme handed back a stale or missing list** after the validate-once change (CompSPH went non-finite at step 46, CRK energy drifted):
   `evaluateOptimalSupport(validateOnce)` now guarantees a list valid for the returned supports for every scheme.
4. **sedovWalls out-of-bounds** (illegal memory access) after the same change: wall rows are appended after the setup built its list, the
   skipped per-iteration check was what rebuilt a list of another size. Found by bisecting commits; fixed (`_listFits`), smoke test added.
   All 17 CompSPH/CRKSPH cases run 6 steps (`evidence/scripts/smokeCompressible.py`). Lesson: nothing in the suite covered a wall case.
5. **The pair list built through the fixed-shape grid was searched as a Gather list** (found 2026-10-13 while comparing graph and eager steps:
   the rebuilt list had 246,724 edges, a from-scratch build 255,618, 3.6 % more). The grid holder made its slots in `Gather` mode for the support solve
   and the pair-list factory reused that mode; the search takes its mode from the grid, so pairs only the NEIGHBOUR's larger support reaches were
   missing from the row of the smaller one. The pairs sit at the edge of the support (kernel ~ 0), which is why the grid-vs-list comparison
   passed at 1e-3 and hash checks were accepted as 'round-off'. Fixed (`ensureSupportGrid(mode=SuperSymmetric)` for the list grid);
   `tests/test_compSPHGraph.py::test_pairListGridFindsTheSymmetricPairs` fails without it. Every earlier grid-mode number in this document that
   depended on list contents (rebuild counts, the loose 1e-3 tolerance) predates the fix; the rebuild-count scan was not repeated.
6. `tests/test_implicitShiftingComparison.py` is flaky on the pre-change code too (2 of 3 runs fail on the base worktree); not caused here.

## 4. Measurements that decided things

* Why lists rebuild (`whyRebuild.py`): 42 of 43 rebuilds in `sod2d` need motion AND support change together, motion dominates (median 81 %
  of the buffer vs 21 %); `noh` 53 % / 48 %, `kelvinHelmholtz` 99 % / 3 %. So moving the solver onto a grid to avoid support-driven rebuilds
  gains little by itself.
* Psi0 on list vs grid per call, B7 2D (after the lane fix): N = 2.5k 0.145 vs 0.13 ms; N = 100k 0.31 vs 1.9 ms (cells >= 1.2 h examine
  ~2x the candidates of a Verlet list, plus a hash lookup per stencil cell). `benchPsi.py`.
* Per-stage solver cost, list (check + 2 psi0 + density, + 12 % rebuilds) vs fresh graphed grid (+ same on the grid): N = 2.5k 0.82 vs
  0.43 ms; 10k 0.86 vs 0.78; 25k 0.88 vs 1.45; 50k 0.88 vs 2.53; 100k 1.32 vs 5.20. The grid wins below ~10k particles, the list above
  ~15-25k. `benchStage.py`.
* Slack / margin scan: see 2.5.

## 5. Open items and the plan from here (user, 2026-10-13)

1. **The remaining syncs**: DONE for `hostTime`, the diagnostics and the time / max-speed read (see 2.x above; 4.8 per step in the whole loop). What is left
   is structural: the step-end flag read (1), the rebuilds' edge-total read (0.6), the grid-radius flag (0.13), the diagnostics read (1), the timer (1).
2. **The integrator's 110 self ops per step** (state copies, stage combines: `mul`, `clone`, `detach`, `view`, `add`, `zeros`).
3. **Full step graph**: DONE for CompSPH in grid mode (2.6). The original list of requirements follows for reference: `CapacityPolicy.inPlace` exists in the core but nothing in the frontend turns it
   on; it needs (a) a frontend switch, (b) every holder of an old list audited (the compSPH per-stage retention is gone with X6; the
   fluid-pair cache is gone), (c) kernels inside the graph reading the PADDED arrays with the `-1` guard, never `livePairs()`/`numEdges`, (d) the
   rebuild eager between replays (decided: not graphed), the validity/usage flags as the loop condition or the step-end read, (e) the
   second-pass machinery as the exact fallback. ADJACENCY_CAPACITY_PLAN.md (warpSPHCore) has the two-case graph contract.
4. Wire the grid mode and its deferred check into `crkSPH_step` (only CompSPH has it) behind the same `stepMode`; its threshold table entry follows from a measurement.
5. Carry the lessons back to ACSPH (ACSPH_REFACTOR_PLAN): stage systems copy the initial state, rebuild decisions at step end, tableau weights
   decide what is stored across stages, deferred device flags + one read per step.
6. Never observed in a real run: a NATURAL conservative second pass (only forced by tests) and a natural energy second pass.
7. Tuning constants are first guesses from one case: `USAGE_GUARD = 1.25`, grid radius headroom 1.5 (solver grid) , `CHECK_EVERY = 8`.
8. Core 'raw-pair' consumers still use `livePairs()` (eager only); retiring `adjacency.i` is the longer-term goal.
