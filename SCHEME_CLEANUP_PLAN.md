# Scheme step cleanup sweep: regions, shared helpers, no host syncs, graph where it pays

> **Progress marker:** this plan's row in [PLANS.md](PLANS.md). Opened 2026-10-13 (CompSPH/CRKSPH as the proving ground for [ACSPH_REFACTOR_PLAN.md](ACSPH_REFACTOR_PLAN.md)),
> widened 2026-10-14 on the user's instruction to every compressible scheme and delta-SPH ("a similar sweep ... a more sensible basis for the more iterative fluid solvers").
> Evidence: `docs/microbench/` (audits before / after per scheme, README there), `docs/compsph_step/README.md` (the CompSPH graph route in detail), OPEN_PROBLEMS §33-§35.

## 1. What the sweep does to a scheme (the recipe, in order)

1. **Baselines first**, from the commit you start at: `python scripts/scheme_baseline.py dump --scheme S --out DIR --steps N [--graph] [--temporal ..] [--switches ..]` (final states of every case that runs
   under the scheme, optionally the CUDA-graph runs) and an audit: `WARPSPHCORE_AUDIT=1 python scripts/audit_case.py <case> --scheme S --steps 60 --warmup 10 --out FILE`
   (ops / launches / syncs per section and per step, with the Python call site of every sync; graph runs: `--cudaGraph` or `--scheme-set stepMode='graph'`).
2. **Split the step into single-purpose regions**: one `record_function("[warpSPH] - [<scheme> - NN] what")` per operation, numbered in CALL order, nothing else in the body (the file reads as the
   algorithm). Remove dead code (commented blocks, unused ops: each is a launch) and move self-contained pieces out. Names other scripts patch (`deltaSPH.enforceDirichlet`, ...) stay as module-level
   imports of the scheme module that CALLS them.
3. **Use the shared helpers** instead of per-scheme copies: `adaptiveSupport.solveStageSupport` (the support solve, ONE Verlet validity check per stage; `computeDensity=True` when the step builds on
   the closing density, as Inutsuka does), `utils.pairList.stagePairList` (the pair list in grid mode), operator wrappers `(state, config, schemeConfig, adjacency, ...)` instead of
   `OperationProperties` + flattened tensors, `godunov.godunovReconstruction`, `schemes/exactDelta.py` (exact-delta integrator contract).
4. **Turn host reads into device values**: stage times deferred in the eager runner (`runner.deferStageTimes`), sync-free timestep modules, device scalars (the Verlet scale may be a 0-d tensor),
   padded-pair / CSR kernels instead of `livePairs()` / `bincount` / `segment_reduce` / `torch.linalg.pinv` (`warpSPH.linalg`), torch-op chains that only exist because a kernel is missing -> one warp kernel
   (CRK terms, Wadsley detector). Keep the torch version as the fallback for inputs that need an autograd tape.
5. **Verify**: `scheme_baseline.py compare OLD NEW` (bitwise where the arithmetic is unchanged; otherwise the difference must be explained: the Owen solve's device-side convergence test moves results by
   its tolerance, ~4e-5 in 1D supports), eager AND graph, graph against eager of the NEW tree; the scheme's tests; the full suite (`pytest tests -q`, ~12 min) before each commit; then the audit again and
   `scheme_baseline.py time` against a worktree of the old commit (3 alternating runs; this GPU is shared, single runs are noisy).
6. **Record**: audits into `docs/microbench/` (+ README line), PLANS row, this file's table, memory. Commit per scheme, no attribution lines.

## 2. Status (2026-10-14; `sod2d` unless noted, integrator figures per RK step, "loop" adds the runner's ~3 syncs: diagnostics, time + max-speed read, step timer)

| scheme | state | commits | regions | integrator ops / syncs, before -> after | ms/step | notes |
|---|---|---|---|---|---|---|
| CompSPH | done (earlier + today) | 30fb72b, 212ae1b | 20 | 447 ops, 3.3 syncs (was 975 / 24 on 2026-10-13); graph ~110 ops, 3.9 syncs/step | eager ~6.0, graph 1.2-2.2 | graph route: `docs/compsph_step/README.md` 2.6-2.8; bitwise |
| Monaghan | done | 7e4e024 | 21 | syncs 10.0 -> 5.2 (loop 13.1 -> 8.2) | 8.76 -> 7.59 | results move by the Owen convergence tolerance; stage times deferred |
| CRKSPH | done, incl. graph mode | 1c5c4a7, a1e9e06 | 20 | 904 -> 504 ops, 9.2 -> 3.2 syncs; graph ~110 ops | eager 9.4 -> 7.6; graph 2.8 (2.5k), 5.0 (10k) | CRK terms = 1 kernel (`modules/crk/terms.py`); `auto` limits 100k / 20k / never(3D) |
| GSPH | done | c2bed4d | 16 | syncs 10.4 -> 4.2 | 7.05 -> 6.15 | |
| Inutsuka GSPH | done, one item left | c2bed4d, 9bcd65f (+ core 54dfc08) | 12 | syncs 12.2 -> 6.2 | ~7.1 unchanged | still 2 validity checks per stage (Owen list, then the larger Gaussian list) + rebuilds |
| MFM / MFV | structure done, syncs open | 50b9a79 | 4 (`step.py`) | unchanged: MFM ssprk 542 ops / 10 syncs, MFV hancock 159 / 4; timestep hook 320 ops / 9.3 syncs | - | `schemes/compressible/meshless/` package; bitwise on 38 runs; G1 + D4 parity pass |
| delta-SPH (+ delta+) | done | 487186a, 04c7426 | 21 | eager syncs 5.0 -> 3.0 (loop 12.2 -> 6.1); graph 7.2 -> 3.1 | eager 19.2 -> 18.2, graph 4.7 | delta+ = same step, 2 config defaults; found + fixed the sloshingTank graph bug (§35) |
| ACSPH | phase 1 only | (earlier) | modules extracted | 21k ops, 25.7 syncs per step on 2026-10-13 | 127 (eager) | `ACSPH_REFACTOR_PLAN.md` phases 2-5 are the work; use this recipe |
| divergenceFree, DFSPH / IISPH refs, omni, band2018pb | not started | | | | | the "iterative fluid solvers"; no graph route today except ACSPH |

Structure done alongside (all bitwise): `schemes/` split by family (`compressible/`, `weaklyCompressible/`, `incompressible/`, `pde/`, `experimental/`; `builder.py` stays the dispatcher; scheme modules are
imported by their full path so patching a module name patches the step), `modules/shockCapturing/` split into one subpackage per switch (`X/switch.py` + `X/kernels.py`).

## 3. Building blocks that exist now

* Tools: `scripts/scheme_baseline.py` (dump / compare / time), `scripts/audit_case.py` + skill `call-audit`, `warpSPH/utils/microBench.py` + `scripts/microbench_switches.py` + skill `module-microbench` (one module call:
  ops / launches / syncs with call sites, host / wall / GPU time), `scripts/microbench_wadsleyDetector.py`.
* `warpSPH/linalg/` (`pinvSymmetric`, `applyPinv`, `symmetricPinv3`): sync-free symmetric 1-3D pseudo-inverse, torch's rtol; torch fallback when a tape is needed. Not the core's `pinv2x2`
  (a renormalisation policy, identity below 4 neighbours, no 3x3).
* `modules/crk/terms.py` (`computeCRKTerms`, `crkFactors`), `modules/crk/operators.py` (`CRKStage`, `computeCRKAccel/dudt`), `modules/compSPH/operators.py`, `modules/godunov/inputs.py`,
  `modules/shockCapturing/Wadsley2017/kernels.py`, `modules/adaptiveSupport/stage.py`, `utils/pairList.py`, `schemes/exactDelta.py`, `schemes/weaklyCompressible/{graphRoute,surfaceMask,analyticWall}.py`.
* Core (warpSPHCore, unpushed): the Verlet validity kernels take `verletScale` as a float or a 0-d device tensor (54dfc08); the gated `record_function` is used in `operations` / `wp_gradient` (5baf30e: torch's
  was imported over it, 2 spurious ops and a C++ RecordFunction per section).
* Step modes: `CompSPHConfig.stepMode` eager | graph | auto for CompSPH and CRKSPH (`modules/compSPH/stepMode.py`, `GRAPH_MAX_PARTICLES`); delta-SPH / ACSPH use `cudaGraph` on the case.

## 4. Bugs found on the way (all fixed, tests added)

* A CUDA-invalidated capture left torch on the capture's side stream -> later eager steps raced and went non-finite (`utils/cudaGraph._captureGuard`, ca80611; README docs/compsph_step 5.10). The `captureAborted` hook sat in the wrong class.
* The delta-SPH graph replayed a stale gravity direction in `sloshingTank` (24-98 % velocity error vs eager, since before today; 487186a, OPEN_PROBLEMS §35). The graph key now includes the direction's address.
* `computeCRKTermsWarp` was 290 torch ops and 3 host reads per stage; the Wadsley detector took 16 syncs per step (4 sites); the weakly compressible timestep 4 per step. All replaced (OPEN_PROBLEMS §34).

## 5. Lessons (read before the next scheme)

* **Compare graph against eager of the same tree**, not only old against new: an unused `.clone()` removed in a refactor changed a graph result and exposed the stale-tensor bug above. Anything a graph reads by
  ADDRESS (config tensors, caches) must be updated in place or be part of the capture key.
* **A case that is nondeterministic in the old code carries no signal** (`tgv-wc`); run old twice before reading a difference. The setup's `Iter k/16 Sampling Error` lines match a grep for "Error".
* **A helper must keep what the caller reads**: Inutsuka's Gaussian h comes from the support solve's CLOSING density, so `solveStageSupport` has `computeDensity`; the first version moved results by percent.
* **`validateOnce` is Owen-only**: the Monaghan / NumberDensity support solvers ignore it, so `solveStageSupport` validates the list itself for them.
* Removing a host sync usually does NOT change wall time at ~7 ms/step (Inutsuka's reach, delta-SPH's graph mode): it shortens the critical path for the graph work and helps multi-GPU / large queues; the
  wall gains came from fewer ops and launches (CRK terms, CompSPH multi-lane kernels) and from the graph itself.
* Do not edit `src/` while a full-suite run is in progress (a package directory shadows a module of the same name mid-run); wait on the PID, never `pgrep -f` the command line.
* Never `git checkout <file>` to undo an experiment in a file with other edits; bisect with a worktree (`git worktree add $SCRATCH/wt <commit>`, `PYTHONPATH=$SCRATCH/wt/src`).

## 6. Open items, in the order to take them (the user chooses; nothing here is started)

1. **Decisions pending from the user:**
   * Meshless geometry hand-over (`schemes/compressible/meshless/common.py takeGeometry`) costs 2 host `torch.equal` syncs per step. Proposal: a system-owned hand-over keyed on tensor versions (no value compare),
     gated by a debug check. Risk: it trusts that the integrator's copy of the state equals the stashed source.
   * Inutsuka: merge the Owen list and the Gaussian list (one validity check per stage): pass the previous step's reach as the solver's scale + a device guard for growth. A redesign, not a patch.
2. **ACSPH** (`ACSPH_REFACTOR_PLAN.md` phases 2-5): apply this recipe; adopt `schemes/exactDelta.py` (its config keeps a copy of the tuple); baselines in eager AND graph. Phase 6 of that plan (delta-SPH) is done.
3. **The incompressible solvers** (`divergenceFree`, `dfsphReference` / `iisph`, `omniIncompressible`, `band2018pb`): not started. Same recipe; they share `schemes/incompressible/omniIncompressible` helpers
   (`_solve`, `_xsphFilter`, ...), so baseline all of them first (`hashCase.py` in `docs/compsph_step/evidence/scripts/` was used for these and was bitwise at the family move).
4. **`launchOperator` no-tape fast path** (user: its own change, small problems only): ~70 % of a small kernel call's host time is `Function.apply` + array views + per-launch overload resolution (OPEN §34).
5. Smaller, from OPEN §34 / §35 / `docs/compsph_step/README.md` section 5: 3D grid `computePsi0` slowness (item 9, why 3D never uses the graph), per-GPU `auto` thresholds (11), integrator clones (14: a 'shared'
   field behaviour would save ~9 of 51, in `warpSPHIntegrators`), wall / Dirichlet cases in graph mode (15), a general guard that keys a capture on every tensor attribute of the scheme config, the Hopkins
   update as one kernel (73 torch ops per stage), Balsara / ReadHayfield / CullenDehnen detector kernels whose GPU time does not scale with N (§34).
6. **Housekeeping:** `dev` is not merged into `main` and warpSPHCore has unpushed commits (54dfc08, 5baf30e, ...); `docs/compsph_step/README.md` section 5 should be re-read against this table before it is trusted.

## 7. How to resume

```
git log --oneline f1b5cc6..HEAD                     # the 2026-10-14 trail (warpSPH); warpSPHCore: git -C ~/dev/warpSPHCore log --oneline -4
python scripts/scheme_baseline.py dump --scheme <S> --out $SCRATCH/old --steps 20 [--graph]      # step 1 of the recipe, BEFORE any edit
WARPSPHCORE_AUDIT=1 python scripts/audit_case.py <case> --scheme <S> --steps 60 --warmup 10 --out $SCRATCH/audit_before.txt
# edit; then
python scripts/scheme_baseline.py dump --scheme <S> --out $SCRATCH/new --steps 20 [--graph] && python scripts/scheme_baseline.py compare $SCRATCH/old $SCRATCH/new
python -m pytest tests -q -p no:cacheprovider        # full suite, ~12 min, in the background; one GPU job at a time
```
Conda env `warp` (`/home/lu26029/miniconda3/envs/warp/bin/python`), repo root as the working directory. Memory: `compsph-step-cleanup-state`, `viscosity-switch-microbench`.
