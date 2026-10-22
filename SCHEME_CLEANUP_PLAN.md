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
| MFM / MFV | staged/module cleanup qualified 2026-10-21 | 50b9a79 + working changes | package time driver + spatial lifecycle/modules | final FP32 full-step launches MFM 594→522, MFV 326→278; stream syncs 23→20 / 17→12 | 65k FP32 MFM 7.256→7.249, MFV 4.346→4.066 | [architecture](docs/meshless_gizmo_validation/ARCHITECTURE_2026-10-21.md), [qualification/performance](docs/meshless_gizmo_validation/PERFORMANCE_2026-10-21.md); original numerical policies retained, focused tuning deferred |
| delta-SPH (+ delta+) | staged/module sweep qualified 2026-10-21 | 487186a, 04c7426 + working changes | 21 RHS regions + lifecycle modules | unused shifting count/reduction removed; shared transport divergence; capture-only pointwise fusion | 72.8k FP32 graph delta 13.661→13.505; delta-plus 13.720→13.481 | 35 source pairs and 12 requested-graph/eager pairs bitwise; source/controls/configuration and small/noisy timing bounds in [report](docs/delta_step/PERFORMANCE_2026-10-21.md) |
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

* **`SimulationConfig.compute`** (`configurations/compute.py`, `docs/compute/README.md`): `execution` eager / graph / auto (per-scheme particle limits, measured), the Jacobi check schedule and the Krylov check interval in one place; `stepMode`, `CaseSpec.cudaGraph` and a solver's `convergenceCheckSchedule` stay as overrides.

## 4. Bugs found on the way (all fixed, tests added)

* A CUDA-invalidated capture left torch on the capture's side stream -> later eager steps raced and went non-finite (`utils/cudaGraph._captureGuard`, ca80611; README docs/compsph_step 5.10). The `captureAborted` hook sat in the wrong class.
* The delta-SPH graph replayed a stale gravity direction in `sloshingTank` (24-98 % velocity error vs eager, since before today; 487186a, OPEN_PROBLEMS §35). The graph key now includes the direction's address.
* `computeCRKTermsWarp` was 290 torch ops and 3 host reads per stage; the Wadsley detector took 16 syncs per step (4 sites); the weakly compressible timestep 4 per step. All replaced (OPEN_PROBLEMS §34).

## 5. Lessons (read before the next scheme)

* **Compare graph against eager of the same tree**, not only old against new: an unused `.clone()` removed in a refactor changed a graph result and exposed the stale-tensor bug above. Anything a graph reads by
  ADDRESS (config tensors, caches) must be updated in place or be part of the capture key.
* **A case that is nondeterministic in the old code carries no signal** (`tgv-wc`); run old twice before reading a difference. The periodic cases jitter their lattice with an unseeded `torch.rand_like`: `scheme_baseline.py` now seeds torch before every run (old vs old bitwise). `--set` takes several `K=V` after ONE flag and repeated flags now accumulate (they used to overwrite each other silently, which looked like a port bug for an hour). When a refactor changes a config default (the adaptive check schedule), force the old value on BOTH trees for the comparison. The setup's `Iter k/16 Sampling Error` lines match a grep for "Error".
* **A helper must keep what the caller reads**: Inutsuka's Gaussian h comes from the support solve's CLOSING density, so `solveStageSupport` has `computeDensity`; the first version moved results by percent.
* **`validateOnce` is Owen-only**: the Monaghan / NumberDensity support solvers ignore it, so `solveStageSupport` validates the list itself for them.
* Removing a host sync usually does NOT change wall time at ~7 ms/step (Inutsuka's reach, delta-SPH's graph mode): it shortens the critical path for the graph work and helps multi-GPU / large queues; the
  wall gains came from fewer ops and launches (CRK terms, CompSPH multi-lane kernels) and from the graph itself.
* Do not edit `src/` while a full-suite run is in progress (a package directory shadows a module of the same name mid-run); wait on the PID, never `pgrep -f` the command line.
* Never `git checkout <file>` to undo an experiment in a file with other edits; bisect with a worktree (`git worktree add $SCRATCH/wt <commit>`, `PYTHONPATH=$SCRATCH/wt/src`).

## 6. Open items, in the order to take them (the user chooses; nothing here is started)

1. **Decisions pending from the user:**
   * Meshless geometry hand-over resolved 2026-10-21: state-owned versioned tickets and explicit clone lineage replace GPU equality reads in the normal runner; mutations/settings changes invalidate reuse. Legacy direct callers retain the comparison adapter.
   * Inutsuka: merge the Owen list and the Gaussian list (one validity check per stage): pass the previous step's reach as the solver's scale + a device guard for growth. A redesign, not a patch.
2. **ACSPH** (`ACSPH_REFACTOR_PLAN.md` phases 2-5): apply this recipe; adopt `schemes/exactDelta.py` (its config keeps a copy of the tuple); baselines in eager AND graph. Phase 6 of that plan (delta-SPH) is done.
3. **The incompressible solvers** (`divergenceFree`, `dfsphReference` / `iisph`, `omniIncompressible`, `band2018pb`): not started. Same recipe; they share `schemes/incompressible/omniIncompressible` helpers
   (`_solve`, `_xsphFilter`, ...), so baseline all of them first (`hashCase.py` in `docs/compsph_step/evidence/scripts/` was used for these and was bitwise at the family move).
   **The linear solvers come first (user, 2026-10-14):** `docs/linalg/SOLVER_INVENTORY.md` inventories every iterative and small dense solve in the three packages (inputs / outputs / syncs per iteration, measured),
   and proposes the unified `warpSPH/linalg` (operators, preconditioners, device-resident results and stopping, Krylov with device predicates, one stationary solver replacing the five relaxed-Jacobi loops) with a
   migration order. **Decided 2026-10-14 (user):** the layer lives in `warpSPHCore/linalg` (pinv moves there; `warpSPH.linalg` re-exports), pure-warp solvers with a `warp.optim.linear` branch (scoped, §8 of the document: works with our torch-op operators on a shared stream, 6-7x faster than the present host loops on the real IISPH operator), `adaptive` check schedule for all Jacobi loops, BiCG kept for comparison, `JFNKSolver(linear_solver=...)` hook for the integrators. On TGV the integrator syncs per step are `divergenceFree` 31, `omniIncompressible` 16, `dfsphReference` 310.
   `scripts/bench_solver_inventory.py` reproduces the per-iteration table.
   **Step 1 done (2026-10-14):** `warpSPHCore.linalg` holds `pinvSymmetric` and the new stationary driver (`runStationary`); all six relaxed-Jacobi loops run on it with `adaptive` as default, bitwise on all five schemes with `every`; syncs per step on TGV `divergenceFree` 31 -> 21, `dfsphReference` 310 -> 21. Next in `docs/linalg/SOLVER_INVENTORY.md` §6: operators / `Restricted`, Krylov with device predicates.
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

## Core search/index consumer qualification (2026-10-15)

[Frontend simulation results](docs/microbench/frontend_search_2026-10-15.md): 90 serial trials / 18,000 steps. Index width, collection and query scheduling preserve final tensors/trajectories bitwise, including default output pipelining. Large eager cases use 20–23% less peak Torch memory; timestep gains are small and spatial scheduling costs about 1% in ordered 3D. Different grid construction has small quantified float32 differences. Per-stage consumers and developed-flow locality are the next performance measurements; full pressure/MFM trajectory batteries remain open.

## CRKSPH / CompSPH performance pass (2026-10-21)

User requested evaluation and optimization after the paper-matching nx=512
blob took 250+ ms/step, with NCU now available; also requested an FP32 rerun.
The matching setup explicitly used FP64. NCU confirms FP64 arithmetic
saturation on the RTX PRO 6000 Blackwell, whose peak FP64 rate is 1/64 of
FP32. Sharing fresh force/heating traversal and reducing compatible work
straight over CSR rows improves both schemes without changing their formulas
or RK stages. The research IdealH control now accepts FP32 for comparison.
Measurements, qualification and commands are in
[the performance report](docs/compsph_step/PERFORMANCE_2026-10-21.md).
The frozen numerical blob investigation is not reopened by this performance
work; no precision or physical-policy default changes.

Completed in this pass: matching CRKSPH FP64 233.14→148.71 ms/step;
standard CRK blob FP64 137.49→89.46; CompSPH 70k FP64 75.23→53.71,
FP32 11.76→6.87. NCU finds ~87% FP64-pipeline utilization in both force
kernels. Validation: 76 FP32 + 16 FP64 checks passed. The daemon-connected
FP32 matching blob completed 4 KH, 23,339 steps / ~376 s, averaging
16.15 ms/step with video and retained history. It stayed positive, energy
change ~7e-6; cold/dense remnant 5.65% versus prior FP64 3.17%, so precision
sensitivity is retained explicitly and no precision default changes.

### Shared core register pass — 2026-10-21

User priority: pause solver-specific tuning to examine register pressure across core SPH operations. Reuse the core operator inventory and qualify shared dimension/traversal/lifetime changes in both precisions with compiler resources, numerical/derivative checks and actual GPU throughput. MFM/MFV modular cleanup remains active; AC is deferred. Evidence belongs in warpSPHCore `docs/regression/core_register_pass_2026_10_21.md`.

Shared register pass result: static tensor extents retained in core outer products, divergence and raw CRK moments. Repeated moment CSR timings improve 17–33% in 2D FP32 and 12–16% in 3D FP64; NCU attributes the gain to reduced local traffic despite higher registers. The ptxas parser now preserves entry spill statistics. Broad caps/specializations and the static-distance candidate are rejected as defaults because some paths regress. Final core gates: 1,558 FP32 checks, 50 focused FP64 checks, 77 derivative checks and 15 benchmark contracts pass; 32 source-paired outputs are bitwise identical. Full evidence and scope are in warpSPHCore `docs/regression/core_register_pass_2026_10_21.md`. MFM/MFV end-to-end qualification is recorded below.

### MFM/MFV resumed — 2026-10-21

User priority: correctness → modularity and structure → general performance → focused optimization. Limit tuning effort while numerical correctness remains uncertain. Close staged/compatibility and shared-context checks, qualify the current MFM/MFV implementation against saved source/state references, run short recorded physical controls, and report bounded throughput gains. Cache-policy experimentation remains deferred.

Completed MFM/MFV resumed pass: package-owned temporal driver and shared spatial lifecycle/modules qualified with 238 FP32 + 72 FP64 frontend checks, 125 core contracts (12 skipped), the full 8,792-pass integrator CPU gate, and 60 saved-state pairs. Daemon-connected FP32 old/new KH controls reach t=0.3 positively, MFM per-particle mass unchanged, energy drift <4e-8. Three alternating 65k trials show MFM neutral and MFV FP32 6.4% faster (4.346→4.066 ms); FP64 gains are small/noisy. Metadata-only plot-title reruns are bitwise identical. Raw source/evidence archive and exact bounds are in [the report](docs/meshless_gizmo_validation/PERFORMANCE_2026-10-21.md). General cleanup is complete; long-time physical questions, AC and focused cache/register tuning remain deferred.

### Delta / delta-plus resumed — 2026-10-21

User requested the same sweep after MFM/MFV. Capture current-source eager/graph and FP32/FP64 baselines, inspect lifecycle/cache ownership and post-integration physics, move reusable parts into modules, preserve integrator and numerical policies, then measure bounded general performance with recorded physical controls. Long-time free-surface/boundary model investigations and AC remain deferred.

Serialization audit found a preservation issue to close before completion:
the legacy weakly-compressible exporter omits Sun's Eq. (7) selector and other
ShiftProperties options, plus several explicit contact/continuity policies.
Round-tripping a delta-plus configuration can therefore lose the selected
method. Add a reusable shifting codec and preserve explicit weakly-compressible
policy fields, with legacy missing-key defaults unchanged. This is configuration
preservation, not adoption of a new model choice.

Completed delta/delta-plus sweep: 136 FP32 + 28 FP64 focused contracts, 104 configuration/schema checks and four idempotence checks pass. All 35 supported source pairs, 12 requested-graph/eager pairs and four post-codec default checks are bitwise. Four daemon/video dam-break controls reach t=.15 in 300 steps, positive with exact mass preservation and old/new state parity. Timing improvements are modest (mostly 1–2%); delta-plus FP32 eager is noisy with a 1.1% slower median, explicitly retained in the report. Native eager marshalling was rejected; fused assembly is capture-only. The reusable shifting codec fixes lost Sun/transport/contact selectors while retaining legacy missing-key defaults. The existing named delta-plus analytic-wall guard remains unchanged after automatic review rejected expanding that boundary-policy contract. Long-time numerical investigations, AC and focused optimization remain deferred. Evidence is in [the report](docs/delta_step/PERFORMANCE_2026-10-21.md).

### Incompressible projection family — 2026-10-21 (architecture qualified; numerical follow-ups open)

User requested inventory and proper reorganization of DFSPH, IISPH, omni and the related projection schemes after comparing source terms, boundary pressure and operator formulations against literature and existing implementations. Include divergenceFree/VD+PS and Band pressure boundaries. AC remains deferred. Establish a pinned source inventory and source-paired baselines first; move reusable physics out of schemes, temporal recurrences into the integrators package, and retain explicit distinctions in pressure units, sources, wall closures, clamps, gauges and convergence metrics. Numerical corrections require separate evidence and a declared adoption decision; structural cleanup alone must preserve current results.

Completed the requested projection-family inventory and architecture pass. Sources, pressure operators and distinct iterations/policies, force inclusion, wall loads and carry/update stages are reusable modules; interpolation XSPH and VD+PS shift/resample are outside schemes. ProjectedStep and kick/drift live in the integrators package; full-map RK repetition is corrected while supported single-evaluation old/new maps remain bitwise. IISPH has its own entry; ProjectedSystem retains the old class alias. Shifting/check schedules and established region/body codecs preserve explicit settings.

Qualification: 91 bitwise source-state pairs, including the identical recorded DFSPH failure; 203 FP32 passes/one baseline IISPH failure, 47 FP64 passes/four source-confirmed budget or fingerprint failures, 8,797 integrator CPU passes/639 skips. Precision-correct wall fixtures resolve seven test-only dtype errors; thresholds are retained. Ten recorded controls connect to the daemon with verified final metadata. TGV 65k medians are mostly neutral (VD+PS −1.9%; other changes −.7% to +1.1%); audit removes two unused host syncs from VD+PS and one unused CD neighbor traversal from the reference path.

The inventory identifies preserved numerical differences, especially physical-pressure versus volume-factor normalization, static-wall neighbor weights, source/stop units, reference ordering and boundary pressure/load closures. DFSPH/IISPH remain unreliable in the recorded column; this pass does not establish their physical correctness. New numerical choices require isolated paper/reference-matched validation. AC and focused tuning remain deferred. [Inventory](docs/incompressible/INVENTORY_2026-10-21.md), [qualification and limitations](docs/incompressible/QUALIFICATION_2026-10-21.md), raw archive in `warpSPHDevelopment/core/experiments/incompressible-cleanup-2026-10-21`.

## Classic-fluid comparison matrix — 2026-10-21

User requests a broad recorded capability survey after the cleanup: weakly compressible and incompressible schemes, including ACSPH as a comparison even where it fails; gas/compressible schemes excluded. TGV must use ≥32k particles; sloshing runs to t=7. Include supported analytic and particle walls, videos and exported diagnostics. Use historical stability settings and preserve failures without solver tuning.

Inputs, preset provenance, resolutions, stopping criteria and comparison limits are in [docs/fluid_matrix/README.md](docs/fluid_matrix/README.md); executable manifest and sequential resumable harness accompany it. Raw evidence stays outside Git under `warpSPHDevelopment/core/experiments/fluid-matrix-2026-10-21`. Status: sequential queue launched (PID 3425015), 120 entries / 86 runnable / 34 unsupported; all 86 configuration-only CPU checks pass. First TGV videos, CSV, particle HDF5 and daemon terminal metadata verified. VD+PS and reference finish t=5; IISPH and omni hit declared velocity stops with evidence retained. Automatic quantitative comparisons update after each row. No solver/default changes and no outcome-driven retries.

### Classic-fluid matrix finished — 2026-10-22

All 86 supported combinations attempted sequentially; finished 03:41 Europe/London. 63 reached the target time, 23 numerical stops, 34 unsupported matrix entries; no timeout/process crash. All 86 have valid video, CSV/NPZ, FP32 HDF5, tensor snapshots and matching final daemon status/time/step. Numerical sources match the launch snapshot. [Final comparison](docs/fluid_matrix/README.md) and curated evidence summarize usable coverage and accuracy rather than interpreting completion as a physics pass.

Delta-SPH completes 13/13; Sun delta-plus 8/8 and TGV final energy error +0.022%; VD+PS 12/13 but over-dissipative TGV. Reference DFSPH 2/12 and IISPH 6/12 show substantial failures. Sloshing pressures remain inaccurate across completed cases; delta alone completes the tested Couette setup, profile amplitude 0.985. AC comparison 4/7, including a sloshing stop at t=4.148. No solver fixes or outcome-driven reruns.

Harness qualification found 13 floating-point remainder steps at nominal endpoints, four corrupting the final shear state. Raw data preserved; derived metrics/plots explicitly exclude the negligible final step and use the preceding sample already at the endpoint within rounding error. Future harness endpoint labels now snap within the declared rounding tolerance; seven CPU regression checks pass. This is an orchestration repair, not a numerical scheme change. Other numerical questions remain separate.

### Per-case quantitative ranking figure — 2026-10-22

User requests one figure with per-case best-to-worst colors and explicitly chooses less vortex energy decay as better, with energy gain penalized. `scripts/plot_fluidMatrixRanking.py` produces one two-panel PNG/PDF/SVG: per-case quality/retention and separate completion wall time, both with numeric ranks and raw metrics. Band vortex excursions (+30.7% / +43.6%) rank behind nongrowing runs. Analytic/experimental errors are used where available; static-water composites and coarse Marrone reference limits are explicit. All 63 completed combinations ranked, failures/unsupported entries unscored, no overall aggregate and no reruns. Figure and metric definitions linked from the final comparison README/HTML.

### Baseline preservation — 2026-10-22

User explicitly requests durable retention for future improvements. Preserve report, static figure, metrics, manifest and scripts in a scoped Git commit; keep the entire raw matrix (videos/HDF5/states/logs/source bundles/reference/runtime) in a versioned archive under `warpSPHDevelopment/core/archives`, with per-file hashes, complete streamed verification and restore instructions. Original raw root marked KEEP; future runs must use distinct output directories/versions. This is local preservation, with no implied cloud upload.

Preservation verification complete: 16,625 payload files, 11,657,141,225 uncompressed bytes, 7,212,926,353-byte read-only zstd archive; every member hash checked. SHA-256 `edc06911a3c1054d21a4d4c12f004ccb84a4bb3d85ce6e1273b2fa3617b94921`. Curated receipt in `docs/fluid_matrix/evidence/archive_2026-10-22.json`; original matrix has a KEEP marker and restore instructions. No cloud synchronization performed.

Curated baseline committed as `582b032da9512d3dd61070c94f0cc64c56ad6aaa` (16 scoped files; unrelated solver WIP left unchanged). A separately verified report capsule from that commit is retained alongside the full raw archive, so the report does not depend on this checkout surviving.
