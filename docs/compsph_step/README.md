# CompSPH / CRKSPH step cleanup -- state at 2026-10-13, with the evidence

Written for the person (or the next session) who continues this work: what was done, why, what was measured, which
bugs it found along the way, and what is open. Everything below is committed; commit hashes are given per item.
Evidence files are in `evidence/` (call-audit tables, the scripts that produced the numbers). Re-run an audit with

    python scripts/audit_case.py sod2d --scheme CompSPH --steps 40 --warmup 10 [--scheme-set supportSolveOnGrid=True]

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
| + step-end rebuild decision learns the buffer usage (`..._gridAdaptive`, **current**) | 443 | 21.6 | **3.7** |

Wall time is NOT better in the grid mode at this size: 300 steps take 2.62 s (grid, deferred) against 2.52 s (list) at N = 2,500.
The step is launch-bound; the grid mode adds two graph replays and a few small ops per stage. The saving is in syncs and in what
a whole-step graph can later contain, not in eager time. `supportSolveOnGrid` is therefore **off by default**.

Remaining sync sources per step (current table): 1.0 step-end list flags, 0.6 rebuilds (9 per 30 steps, 2 syncs each),
2.13 integrator self (`hostTime` = the `float(t + dt)` reads, 2 per step, plus the grid-radius flag every 8 steps), and outside the
step the runner's own 8.3 (case diagnostics) + 1.0 (max-speed read) + 1.0 (step timer). The integrator's own state copies
and stage combines are 110 ops per step (25 % of all ops), in no compSPH section.

## 1. Commits (trail)

warpSPH (`dev`): 2d379a8 adjacency rename (frontend) / b1872e3 X6 pair sum / ffe6d2c list chained across stages + early stage-0
rebuild / 948b84a support solver validates once / f95f65b NoScheme fix / ae222f2, 03b2c81, 2cb8662 Psi0 / 825ccc7 no closing density +
out-of-bounds fix / c5fefb8 wall clamp module + grid solve / 700f8cf deferred check / f5b03a9 forceRebuild / 339e0b5 learned usage.
warpSPHCore (`dev`): d733192 padded adjacency / d2cd60b `inPlace` / 09b34f0 `rebuildSlack` / 07403b5 fused validity kernel /
3ea04fc fixed-shape hash map (slots) / f853f97 `verletStaleFlag`, `gridFactory` / 958ec05 `forceRebuild` / 8a67521 `verletUsage`
(+ plan commits: c266663, 67e27d7, c28a378, 3a12423, b249e02, f24ce3b).
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
5. `tests/test_implicitShiftingComparison.py` is flaky on the pre-change code too (2 of 3 runs fail on the base worktree); not caused here.

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

1. **The remaining syncs** (3.7 per step): `hostTime` (the integrator's `float(t + dt)` reads, 2 per step, in `warpSPHIntegrators/util.py:299`),
   the grid-radius flag every 8 steps, the rebuilds' edge-total read (1 per rebuild), the step-end flag read (1). Outside the step: the case
   diagnostics (8.3 per step), the max-speed read, the step timer.
2. **The integrator's 110 self ops per step** (state copies, stage combines: `mul`, `clone`, `detach`, `view`, `add`, `zeros`).
3. **Towards the full step graph with reused adjacency tensors**: `CapacityPolicy.inPlace` exists in the core but nothing in the frontend turns it
   on; it needs (a) a frontend switch, (b) every holder of an old list audited (the compSPH per-stage retention is gone with X6; the
   fluid-pair cache is gone), (c) kernels inside the graph reading the PADDED arrays with the `-1` guard, never `livePairs()`/`numEdges`, (d) the
   rebuild eager between replays (decided: not graphed), the validity/usage flags as the loop condition or the step-end read, (e) the
   second-pass machinery as the exact fallback. ADJACENCY_CAPACITY_PLAN.md (warpSPHCore) has the two-case graph contract.
4. Wire `supportSolveOnGrid` and the deferred check into `crkSPH_step` (only CompSPH has it); decide its default after the graph work.
5. Carry the lessons back to ACSPH (ACSPH_REFACTOR_PLAN): stage systems copy the initial state, rebuild decisions at step end, tableau weights
   decide what is stored across stages, deferred device flags + one read per step.
6. Never observed in a real run: a NATURAL conservative second pass (only forced by tests) and a natural energy second pass.
7. Tuning constants are first guesses from one case: `USAGE_GUARD = 1.25`, grid radius headroom 1.5 (solver grid) , `CHECK_EVERY = 8`.
8. Core 'raw-pair' consumers still use `livePairs()` (eager only); retiring `adjacency.i` is the longer-term goal.
