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

## 3a. Status

**Phase 1 done (2026-10-13).** `schemes/artificialCompressible.py` 906 -> 133 lines (59 of them the module docstring): the step is
calls into `modules/artificialCompressible/` and nothing else. New modules: `parameters.py` (acParameters, convergence metric), `surfaceFlags.py` (`SurfaceFlags`, `detectSurface`, taper), `wallClosure.py` (Eq. 61 wall pressure,
unilateral variants, `WallBounce`/`makeWallBounce`, `ContactMasks`/`contactMasks`), `residual.py` (the spatial residual and viscosity),
`forcing.py` (`bodyForces`); configuration facts and checks live in `configurations/artificialCompressible.py` (`RK_TABLEAUS`, `RECOMMENDED_CFL_TAU`, `EXACT_DELTA_INTEGRATORS`, `ArtificialCompressibilityParams.validate()`, `ArtificialCompressibleSPHConfig.validate(simulationConfig)`; the step makes one guard call) -- user's rule, 2026-10-13: configuration and its validity checks go in the configurations, not in a module folder called from the step, `workingState.py`, `dualTime.py` (`DualTimeSolver.iterate/solve`, `exactDeltaUpdate`, `zeroNonFluidRows`,
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

## 4. Risks / open questions

* `wp.capture_while` with torch tensors and torch ops inside the body (conditional nodes need CUDA >= 12.4; driver here is 13.3 /
  toolkit 12.9). Fallback: a fixed-count unrolled graph with the stop flag masking updates.
* Phase 3 changes when the loop stops (first converged iteration instead of the next checkpoint): not bitwise with today's runs,
  only with the every-iteration eager reference.
* Two sessions edit this file today (the user's call who owns the refactor); no edits to `schemes/artificialCompressible.py` by a
  second session while a phase is open.
