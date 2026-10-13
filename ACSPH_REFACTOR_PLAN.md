# ACSPH scheme refactor: modules, fused kernels, one graph per step

> **Progress marker:** this plan's row in [PLANS.md](PLANS.md). Opened 2026-10-13 on the user's instruction ("clean up the scheme
> first": the ACSPH run is at 25-40 % GPU utilisation and every comparison is impractical). Related: OPEN_PROBLEMS §33,
> [ACSPH_PLAN.md](ACSPH_PLAN.md), `docs/acsph/DYNGPUSPH_COMPARISON.md` ("Launch overhead").

## 1. What is wrong (measured / read from the code, 2026-10-13)

`schemes/artificialCompressible.py` is 906 lines. `artificialCompressible_step` is ~350 lines and contains, as nested closures
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

The same drift is under way in `schemes/deltaSPH.py` (`_surfaceMaskAcrossStages`, `_graphedRHS`, `_rhsIsGraphable`, section-by-section
inline calls in `_deltaSPH_rhs`). Part of the inline ACSPH code (wall bounce, taper, the surface-flag closure) was added in the
2026-10-12/13 sessions on top of the existing pattern; the pattern itself is older.

## 2. Target design

* `modules/artificialCompressible/dualTime.py`: `DualTimeSolver`, a per-run object owning persistent device buffers (x, v, p
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

## 3. Phases (each ends bitwise-checked and committed; GPU runs one at a time, utilisation reported with every run)

| # | Phase | Check |
|---|---|---|
| 0 | Baseline: one profiled slot (Marrone nx 70 t <= 0.45, sloshing nx 100 t <= 1): ms/step, launches, syncs, GPU util, with the current code, graph on/off | numbers recorded here |
| 1 | Extract, no behaviour change: `pseudoIteration`, residual, wall closure, bounce, taper, flags move to modules as plain functions with explicit arguments | `tests/test_artificialCompressible.py` unchanged and green; Marrone / sloshing short runs bitwise identical |
| 2 | Fused warp kernels for the stage combine, projection, bounce, convergence metric | per-kernel tests vs the torch reference; gradcheck script if a kernel is differentiable; bitwise or <= 1 ulp (state the tolerance) |
| 3 | Device-side stop test + `capture_while`; per-step scalars as device inputs; one graph per Verlet generation | stops at the first converged iteration (the old schedule overshoots up to the next checkpoint, so the reference for the comparison is eager with a check every iteration) |
| 4 | Extend to the whole step (`_stepGraphable` for ACSPH) | a step is one graph launch; utilisation and ms/step vs phase 0 |
| 5 | Same extraction in `schemes/deltaSPH.py` | its tests + Marrone 3.1 reference timing (108.7 s, CLAUDE.md rule 5) unchanged |

Acceptance for the plan: Marrone 3.1 nx 70 and sloshing nx 100 at >= 80 % GPU utilisation with video off, ms/step well below the
phase-0 numbers, and every phase's results bitwise (or tolerance-stated) against the previous.

## 4. Risks / open questions

* `wp.capture_while` with torch tensors and torch ops inside the body (conditional nodes need CUDA >= 12.4; driver here is 13.3 /
  toolkit 12.9). Fallback: a fixed-count unrolled graph with the stop flag masking updates.
* Phase 3 changes when the loop stops (first converged iteration instead of the next checkpoint): not bitwise with today's runs,
  only with the every-iteration eager reference.
* Two sessions edit this file today (the user's call who owns the refactor); no edits to `schemes/artificialCompressible.py` by a
  second session while a phase is open.
