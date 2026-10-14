# ACSPH scheme refactor: modules, fused kernels, one graph per step

> **Progress marker:** this plan's row in [PLANS.md](PLANS.md). Opened 2026-10-13 on the user's instruction ("clean up the scheme
> first": the ACSPH run is at 25-40 % GPU utilisation and every comparison is impractical). Related: OPEN_PROBLEMS §33,
> [ACSPH_PLAN.md](ACSPH_PLAN.md), `docs/acsph/DYNGPUSPH_COMPARISON.md` ("Launch overhead").
>
> **2026-10-14:** the cleanup recipe, the shared helpers and the lessons that came out of applying it to the other schemes (CompSPH, Monaghan, CRKSPH, GSPH, Inutsuka,
> meshless, delta-SPH) are in [SCHEME_CLEANUP_PLAN.md](SCHEME_CLEANUP_PLAN.md); phase 6 below is done there. ACSPH's own phases 2-5 are still open: start from that plan's section 1
> (baselines in eager AND graph mode first) and its section 5 (graph-vs-eager comparison, address-read hazards).

## 1. What is wrong (measured / read from the code, 2026-10-13)

`schemes/incompressible/artificialCompressible.py` is 906 lines. `artificialCompressible_step` is ~350 lines and contains, as nested closures
and inline torch, everything the repo otherwise puts in `modules/`:

| Inline in the scheme | Belongs in |
|---|---|
| `pseudoIteration` (the dual-time body: BDF source, surface flags, wall pressure, diffusion, RK stage loop, `project`, convergence metric) | a dual-time solver module |
| `wallBounceSetup` / `wallBounce` (D5: pair filtering, scatter_reduce, index_add: ~10 tiny torch launches per call, called up to 4x per iteration) | a wall-contact module, as a warp kernel |
| `wallPressures`, `_wallAdjacent`, `_unilateralWall*` | `modules/incompressible/wallPressure.py` (already holds the extrapolation) |
| `diffusionTaper`, `taperOf`, `surfaceFlags`, `project` (mask arithmetic) | the pressure-smoothing / surface-detection modules |
| `_spatialResidual`, `_viscosity`, `_ViscosityShim` | a residual module |
| the RK stage axpy/accumulate (`xs = xs + dtau*a*kx[l]` ... `torch.where(fluid, ...)`), per iteration ~100 elementwise torch launches | one fused warp kernel per stage |
| `convergenceMetricDevice` + a host `bool(epsT < target)` read at schedule points | a device-side stop flag |

Consequence (measured): ~260 launches per pseudo-iteration, the iteration body is captured into a CUDA graph **once per real step**
(one eager iteration + one capture, ~14 of 37 ms on a cheap Marrone step), per-step setup (Verlet, boundary velocities, surface
detection, bounce setup, forcing) is eager, and the stop test is a host sync. GPU utilisation 25-40 %. The whole-step graph that
delta-SPH has (`runner._stepGraphable`, `GraphedIntegratorStep`) is not available to ACSPH at all.

The same drift is under way in `schemes/weaklyCompressible/deltaSPH.py` (`_surfaceMaskAcrossStages`, `_graphedRHS`, `_rhsIsGraphable`, section-by-section
inline calls in `_deltaSPH_rhs`). Part of the inline ACSPH code (wall bounce, taper, the surface-flag closure) was added in the
2026-10-12/13 sessions on top of the existing pattern; the pattern itself is older.

## 2. Target design

* `modules/artificialCompressible/dualTime.py`: a frozen `DualTimeProblem` and free functions `iterate(problem, x, v, p)` / `solveDualTime(...)` (no solver object: nothing mutable to hold; user's call 2026-10-13 -- dependencies explicit, maps onto kernel arguments and static graph buffers). Later: persistent device buffers (x, v, p
  iterates, stage slopes, the BDF history, the masks) and a small device array of per-step scalars (`dtau`, `alphaT/betaT/gammaT`,
  `k1`, `nu`, tableau coefficients). The step becomes: build the adjacency, fill the scalars, `solver.solve(...)`.
* Fused warp kernels (`wp_*.py`, the repo convention: `warpOperation`-style, gradcheck-covered where differentiable):
  RK stage combine + fluid mask, projection (`clamp_min` / zero masks), wall bounce (one kernel over wall-fluid pairs, no
  scatter_reduce), the convergence metric writing a device flag.
* **One graph launch per real step, no host syncs**: the loop `while not converged and m < max` runs inside the graph with
  `wp.capture_while` (warp 1.17 has it; conditional graph nodes), the per-step scalars are graph inputs, so the graph is captured
  once per Verlet-list generation (the existing `GraphedIntegratorStep` keying) instead of once per step.
* Then wire ACSPH into the whole-step graph (pre-loop setup and `finalize` shifting are the remaining eager parts).
* Same treatment for the delta-SPH scheme afterwards (phase 5), no behaviour change.

## 3. Phases (REORDERED 2026-10-13, user: optimise the eager code first, worry about graph capture afterwards)

Graph capture hides the cost it should be measured against (a replay issues no ops, a per-step capture adds work) and demands
a sync-free, fixed-shape step, which is the same work as removing the syncs. So: **all development and measurement is eager
(`--no-cudaGraph`, `audit_case.py --no-cudaGraph`); graphs are a bolt-on at the end** (case defaults stay `cudaGraph=True`
meanwhile so ordinary runs stay fast). Each phase ends bitwise-checked, audited and committed; GPU jobs one at a time.

| # | Phase | Check |
|---|---|---|
| 0 | Eager baseline (below) | numbers recorded |
| 1 | **Done.** Extract into modules, no behaviour change | 64 tests, bitwise vs previous commit |
| 2 | Remove host syncs from the step (device-resident scalars: viscosity, bounce setup, parameters/BDF, forcing, solver setup, timestep hook, Verlet validity, shifting; one batched stop-test read) | audit `sync/step` per section; bitwise |
| 3 | Collapse the torch-op storms: fused warp kernels for the RK stage combine + fluid mask, projection, wall bounce (pair kernel, no scatter_reduce), convergence metric, surface-flag/taper arithmetic | audit `ops/step`, `launch/step`; per-kernel tests vs the torch reference; gradcheck script if differentiable; bitwise or tolerance stated |
| 4 | Per-step setup (contact masks, bounce setup, parameters, solver construction) as one or two kernels / cached per Verlet generation | audit |
| 5 | Only then graphs: persistent graph per Verlet generation, device-side stop test (`capture_while`), whole-step graph | utilisation >= 80 %, ms/step vs the eager phase-4 number |
| 6 | **Done 2026-10-14** (SCHEME_CLEANUP_PLAN.md): same extraction + cleanup in `schemes/weaklyCompressible/deltaSPH.py` | its tests + Marrone 3.1 reference timing (108.7 s) |

Acceptance: Marrone 3.1 nx 70 and sloshing nx 100 with graphs OFF at high GPU utilisation and low `sync/step`, then graphs ON as a bonus.

**Phase 0, eager baseline (2026-10-13, graph off, forensics off, video off, idle GPU):** Marrone 3.1 nx 70, t <= 0.3 (183 steps):
**127.5 ms/step, GPU 23 %** (graphed: ~42 ms/step). Audit (`dambreak`, nx 70, 50 steps after 10 warm-up; `scripts/out_callAudit/acsph_eager_baseline.txt`):
**21,058 torch ops, 907 warp launches, 25.7 syncs per integrator step** (dual-time loop 20,413 ops / 886 launches / 10.1 syncs at ~42 pseudo-iterations per step,
i.e. ~483 ops and 21 launches per iteration; contact masks + bounce setup 5.0 syncs; Verlet 2.8; stop test 6.1; timestep hook 9.2; step
timer, speed read, diagnostics ~1-1.6 each). Coverage 100 % of ops / launches, 99.9 % of syncs.

## 3a. Status

**Phase 1 done (2026-10-13).** `schemes/incompressible/artificialCompressible.py` 906 -> 133 lines (59 of them the module docstring): the step is
calls into `modules/artificialCompressible/` and nothing else. New modules: `parameters.py` (acParameters, convergence metric), `surfaceFlags.py` (`SurfaceFlags`, `detectSurface`, taper), `wallClosure.py` (Eq. 61 wall pressure,
unilateral variants, `WallBounce`/`makeWallBounce`, `ContactMasks`/`contactMasks`), `residual.py` (the spatial residual and viscosity),
`forcing.py` (`bodyForces`); configuration facts and checks live in `configurations/artificialCompressible.py` (`RK_TABLEAUS`, `RECOMMENDED_CFL_TAU`, `EXACT_DELTA_INTEGRATORS`, `ArtificialCompressibilityParams.validate()`, `ArtificialCompressibleSPHConfig.validate(simulationConfig)`; the step makes one guard call) -- user's rule, 2026-10-13: configuration and its validity checks go in the configurations, not in a module folder called from the step, `workingState.py`, `dualTime.py` (`DualTimeProblem`, `iterate`, `solveDualTime`, `exactDeltaUpdate`, `zeroNonFluidRows`,
`bdfHistory`). The same code moved, only renamed (underscores dropped) and split; no arithmetic changed. Checked: 64 tests green
(`tests/test_artificialCompressible.py` now imports from the modules), and Marrone 3.1 nx 70 to t = 0.3 against the previous commit in
a worktree: same 183 steps, identical per-step `maxVelocity` / `kineticEnergy` rows (66 compared), 42.3 vs 42.2 ms/step. `probe_contactLine.py`'s
`--wallMode` monkeypatches now target `dualTime` / `residual`. **Phase 0 baseline** (GPU idle apart from the other user's llama-server,
which was not running at the time; graph on, forensics off, video off): Marrone nx 70 early window **42 ms/step**; sloshing nx 100 first 3 s
**88.6 ms/step**, GPU 70 % by `nvidia-smi`, 200-iteration cap hit for the first 0.5 s at rest and 90 / 72 mean iterations in t 0.5-1.5 (separate
convergence issue, not the utilisation cause); aligned sloshing nx 100 31.7 ms/step (other session, t = 2.23). The per-iteration body is still the
same ~260 torch launches: phases 2-4 are the actual speed-up.

## 3b. Where the host time goes (call audit, 2026-10-13)

`WARPSPHCORE_AUDIT=1` (new, `warpSPHCore/callAudit.py`, rides on `record_function`) on Marrone 3.1 nx 70, graph on, 68 steps: about **22 host
syncs per step**, each a pipeline drain: ~8 dual-time convergence checks (`dualTime.py` `solve`), 3 `viscosityScalars` host reads
(`residual.py`; cached per tensor object, so once per step), 2 in `wallBounceSetup` (`bool(ghost.any())`, an `int(sum)` for the one-time message),
1 `torch.cuda.synchronize` in the per-step graph capture, ~2.6 each in the Verlet validity check and in shifting, ~5 in the timestep
(`computeTimestep`, `asTensor`), 1 diagnostics read. The dual-time section issues ~1150 torch ops per step even with graphs on
(the eager first iteration and the capture). These are phase 3's target list: device-resident scalars (viscosity, bounce setup), a persistent
graph, a device-side stop test, and then the shared pieces (Verlet validity, shifting, timestep).

## 3c. Adjacency capacity and raw pair lists (design notes, user decisions 2026-10-13)

* **Capacity-padded adjacency** (prerequisite for the phase-5 persistent graph; opt-in `capacity` on the Verlet build in
  warpSPHCore, default off): allocate the edge arrays ~10 % over the real count, rebuild in place while the count fits, reallocate
  only on overflow (a device flag, read at the Verlet validity check that already syncs) or when the real count falls below
  ~50 % of the capacity. This also removes the per-rebuild `torch.sum(edge_count).cpu().item()` allocate sync
  (`radiusSearch/compactHash/search.py`). Per-particle CSR kernels (`edgeOffsets`/`numNeighbors`) never touch the tail.
* **Padding convention: `i = j = -1`, no dummy particle.** A dummy row would leak an internal convention (a dead particle with no physics)
  into exported traces. Consumers of raw `i`/`j` check `0 <= i < n` before use (careful: a torch gather with `-1` silently wraps to the last
  element, so mask before indexing). The count is never taken from `adjacency.i.numel()`.
* **`adjacency.i` is the velocity-Verlet list, not the active pairs**: it is a superset, so consumers of raw pairs must already test
  actual interaction (`dist < h`) or Verlet membership. Nothing may treat its length as a pair count.
* **Raw `i`/`j` interactions are a development convenience, to be avoided in new code** (the dense `i` tensor itself is a candidate
  for removal later; per-particle CSR kernels are the target). Known raw users to retire: `wallBounceSetup` / `makeWallBounce`
  (the fluid-wall pair filter, phases 3-4: make it a CSR wall-neighbour kernel).
* Inlets / outlets (changing particle count) are not supported here; out of scope.
* **The core change has its own plan: `~/dev/warpSPHCore/ADJACENCY_CAPACITY_PLAN.md`** (sync sites, design, consumer inventory, steps). The consumer-side guard (`0 <= i < n` before using a raw pair) is independent of it and a no-op on today's adjacency, so it can land here at any time.

## 3d. CompSPH / CRKSPH as the proving ground (2026-10-13, user) -- lessons to bring back

The user redirected the work: get the same cleanup working on CompSPH / CRKSPH first (they have the same stage structure and are
cheaper to audit), then apply what holds to ACSPH. Full account with evidence: **`docs/compsph_step/README.md`** (commits, audit tables,
the scripts that produced every number, bugs found). Per-step on `sod2d` CompSPH: 975 ops / 22 launches / 24 syncs -> 443 / 21.6 / 3.7.
Lessons that carry over to the ACSPH phases above:

* **Stage systems are copies of the step's INITIAL state**: anything a stage learns (the pair list it rebuilt, a flag) must be carried
  explicitly (`PairWork`, `startFromLatest`); otherwise stage 1 repeats stage 0's work on stale input.
* **Tableau weights decide what must be stored across stages**: `RungeKuttaB` now passes `stageWeights` to the system's `initialize`
  (opt-in) and supports one optional second pass (`rerunRequest`); midpoint (`b = [0, 1]`) stores nothing across stages, Heun / SSPRK3 /
  RK4 do. A step whose `finalize` cannot complete from what the stages stored asks for the step again, the second pass uses what the
  first found out. ACSPH's pseudo-time loop is the same shape (BDF2 history is the "pair work").
* **Decide list rebuilds at step END, read one flag per step**: per-stage device flags (`verletStaleFlag`), the buffer usage `verletUsage`,
  ONE host read in `finalize`; rebuild then if `usage + 1.25 x (learned per-step usage) >= 1`; a failing in-step flag -> conservative second
  pass. A fixed margin was wrong for some tableau (scan in the README): learn it.
* **Fused / fixed-shape pieces first, graphs after** (the user's order held): the validity check was 38 % of all ops as 44 torch ops
  (now 2 warp launches); the hash-map build is 108 ops / 0 syncs with fixed shapes (`HashMapSlots`) and graph-replayable; rebuilds are
  eager and outside any graph (user decision), writing into the padded buffers when `inPlace` is on.
* **Responsibilities explicit in the step, not hidden in a helper**: the wall support clamp is its own module called after the list; the
  support solve returns supports (and optionally density) and does not own the adjacency (`supportSolveOnGrid`).
* **Hidden-failure checklist for refactors that rename or reorder**: `hasattr`/`getattr` with string names, identity-keyed caches, code paths
  no test covers (wall cases, `NoScheme`): both regressions of this day were found by a smoke run over all registered cases
  (`docs/compsph_step/evidence/scripts/smokeCompressible.py`) and by bisecting commits against a baseline worktree.

ACSPH items that this changes: phase 2 (host syncs) can reuse the step-end single-read pattern for the dual-time stop test; phase 5's
persistent graph can rely on `CapacityPolicy.inPlace` + the flags above; the `iterate` loop's per-iteration validity is not needed once
the list is decided at step end.

## 4. Risks / open questions

* `wp.capture_while` with torch tensors and torch ops inside the body (conditional nodes need CUDA >= 12.4; driver here is 13.3 /
  toolkit 12.9). Fallback: a fixed-count unrolled graph with the stop flag masking updates.
* Phase 3 changes when the loop stops (first converged iteration instead of the next checkpoint): not bitwise with today's runs,
  only with the every-iteration eager reference.
* Two sessions edit this file today (the user's call who owns the refactor); no edits to `schemes/incompressible/artificialCompressible.py` by a
  second session while a phase is open.
