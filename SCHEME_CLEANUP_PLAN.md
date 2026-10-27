# Scheme step cleanup sweep: regions, shared helpers, no host syncs, graph where it pays

> **Progress marker:** this plan's row in [PLANS.md](PLANS.md). Opened 2026-10-13 (CompSPH/CRKSPH as the proving ground for [ACSPH_REFACTOR_PLAN.md](ACSPH_REFACTOR_PLAN.md)),
> widened 2026-10-14 on the user's instruction to every compressible scheme and delta-SPH ("a similar sweep ... a more sensible basis for the more iterative fluid solvers").
> Evidence: `docs/microbench/` (audits before / after per scheme, README there), `docs/compsph_step/README.md` (the CompSPH graph route in detail), OPEN_PROBLEMS §33-§35.

### DFSPH follow-up (2026-10-25)

**Current gate (user, 2026-10-25): plain hydrostatic tank first.** The observed
controls are not acceptable physical validation. Stop expanding the matrix;
compare native IISPH and DFSPH on one identical, simple at-rest tank and inspect
the pressure solves and integration in isolation. Other cases wait until that
tank behaves properly. Implementation is provisional pending this gate.

**Baseline correction (user, 2026-10-25):** the new inviscid tank was not the
established IISPH comparison. It used the correct one-time mass normalization,
but changed nu=.01 to zero, omega=.25 to .3, and the tank geometry. Its failures
must not be used to invalidate the earlier IISPH result or qualify DFSPH.
All unmatched probes are stopped; the new case and session-specific tests are
withdrawn from the public package into `.tmp/dfsph-tank-2026-10-25/`.
The wall-gravity experiment was never adopted. Recover and reproduce the
established saved tank first, then change only the solver.

The recovered C4/nh4 sampled-wall control is
`.tmp/incompressible-investigation/wall-treatment/runs/shepard-contact-wendland-c4/`:
512 fluid + 832 fixed wall samples, dx=1/32, fluid depth .5 m, tank [-.5,.5] x [0,2],
zero initial p/v, rho0=1000, nu=.01, two-sided free-slip viscosity, contact-Shepard
pressure with the existing gravity extension, omega=.25, min2/max100 pressure
iterations and the existing signed density stopping statistic. The common mass
factor is .9995844223215761, giving mass .9761566624234141 to every fluid/wall
particle; raw kernels are used thereafter. The saved run reaches 5 s. Reuse its
initial fields and settings before introducing any further variants.

User authorized baseline reproduction on 2026-10-25. Work is isolated in
`.tmp/iisph-baseline-reproduction-2026-10-25/`. Replay loads the saved full
initial state rather than resampling geometry, uses current native Warp IISPH,
and selects the existing fluid-only viscosity mode to match the archived
control's zero viscous wall drag. Its historical “free slip” label is not the
newer mirrored-wall viscosity model. Preflight requires bitwise initial-field
identity, unchanged already-calibrated masses, matched initial density and
matched first pressure source/force before advancing the saved 5 s case.
No DFSPH or wall-law variant runs are authorized by this baseline step.

Baseline replay completed: saved particle fields bitwise identical on input;
current Core mass factor exactly matches .9995844223215761 and masses remain
unchanged. Initial fluid density agrees within 2.22e-16 in rho/rho0. First native
pressure/acceleration agree within 2.44e-9 Pa / 1.49e-11 m/s²; archived states
through step 100 match positions within 5e-16 m and velocities within 1.23e-13
m/s. The current native replay reaches 5 s / 5000 steps with zero crossings
and zero pressure-budget terminations. Wall positions/velocities/masses remain
fixed. CPU cost 128.35 s wall / 20.23 ms per measured step.

Long-time identity is not claimed: mean RMS speed over 4.5–5 s is 10.723 mm/s
versus 8.984 in the archive, with different pressure-burst timing after ~1.3 s.
Frozen density, viscosity predictor and pressure checks at six archived states
including step 5000 agree to FP64 roundoff and match iteration counts. This
reproduces the calibrated physical setup and bounded settling behavior, not
the exact archived trajectory. Raw comparisons, movie, limits and source
snapshots are in `.tmp/iisph-baseline-reproduction-2026-10-25/README.md`.
No additional model variants or DFSPH trajectories were run for this step.

User now authorizes DFSPH on the same saved setup (2026-10-25). Run on the
GPU in FP32, with the saved fields cast once, calibrated masses retained,
recorded fluid-only two-sided nu=.01 viscosity, unchanged contact/gravity
wall law and physical timestep limits. Match the IISPH density policy
(omega=.25, min2/max100, .01 percent); use explicit native DF policy
(min1/max100, .1 percent, omega=.25), warm starts and deficiency guard on.
A sequential GPU FP32 IISPH control isolates device/precision differences
from the new divergence stage. No wall or initialization experiment is added.
Artifacts remain in `.tmp/iisph-baseline-reproduction-2026-10-25/`.

Matched GPU FP32 runs completed: both 5 s / 5000 steps, identical initialized
fields and preserved calibrated masses, no wall crossings and no pressure
budget terminations. Late RMS speed (4.5–5 s) is 7.43 mm/s DFSPH vs 10.05 IISPH;
raw hydrostatic pressure L2 is 15.68% vs45.38%. DFSPH suppresses intermittent
kinetic bursts, but corner motion and the detached top row remain. It costs
104.18 s wall / 15.27 ms per step vs75.43 s / 10.06 ms. Both 1001-frame movies
decode; all sampled-wall fields and particle masses remain unchanged. No
model/default changed or tuning sweep was done. This establishes a matched
sampled viscous-tank comparison, not exact equilibrium or broader qualification.
See `gpu-comparison.png/json` and the local evidence README.


User now requests the same DFSPH tank with analytic boundaries (2026-10-25).
Retain the exact saved fluid lattice, calibrated masses, GPU FP32, viscosity,
pressure policies and physical timestep limits. Replace sampled walls with one
outside-solid analytic box at the original planes; apply the same uniform mass
factor to continuum wall moments. Measure and retain the initial wall-density
residual, with no particle shifts, wall offsets or calibration fitting. Record
5 s with Vispy and compare against the completed sampled-wall DFSPH run.

Analytic GPU FP32 run completed 5 s / 5000 steps. Initial continuum wall
rho/rho0 excess peaks at 1.016677; fluid interior remains 1. The first CD
solve hits max100 iterations and injects a 1.971 m/s peak (sampled .0835).
Two particles briefly cross, deepest .441 mm; none remain outside at 5 s.
Late RMS4.938 mm/s vs7.429 sampled, pressure L2 17.16% vs15.68%. Cost111.45 s
wall /14.65 ms per measured step vs104.18 s /15.27 ms. Calibrated fluid masses
and initial fluid fields are verified unchanged; no numerical/default change
or tuning. Startup and wall coupling remain unqualified despite later damping.
The 1001-frame movie decodes; local comparison/evidence are in the same ignored
baseline-reproduction folder. Detailed results are in its README.



Implementation authorized by the user on 2026-10-25: register native `dfsph`
on IISPH's spatial operators; use reference-start divergence placement,
explicit density/divergence policies and dedicated pressure histories.
Retain `dfsphReference` as a named legacy comparison. Qualify algebra,
stage ownership, serialization and available trajectory checks before reporting
the implementation complete.

User-requested investigation: use the improved native IISPH as the basis for
DFSPH; cross-check the published constraints and SPlisHSPlasH's operators,
boundary terms, pressure policies and integration order against the current
`dfsphReference` and the other incompressible schemes. This is a source and
architecture audit before numerical implementation; AC remains deferred.

Audit complete: [DFSPH method and migration contract](docs/incompressible/DFSPH.md).
The native force and exact diagonal reproduce the single-phase equal-volume
reference form after converting pressure unknowns. The published end-of-step
divergence placement agrees with the local sequence; the pinned code moves it
to the next step's start. Current-volume weighting, static-wall squares,
pressure-history units, tolerances and deficient-row source handling remain
distinct. Adaptive reference timestep reads also need an explicit comparison
contract. Verification: 21 focused CPU checks passed, 8 CUDA checks skipped;
fixed-geometry algebra and actual native kernel/diagonal checks passed. Raw
audit checks and logs are in `.tmp/dfsph-audit-2026-10-25/`; that source audit
did not change numerical behaviour.

Native `dfsph` is now implemented with separate density/divergence policies,
native IISPH operators and exact diagonal, a shared pre-drift assembly,
reference-start divergence placement and dedicated checkpointed pressure
histories. Existing scheme names/numeric IDs are retained. Both stages use
.3 relaxation: the native wide-support spectrum and matched recorded failures
show that .5 exceeds the n_h=4 C2/C4 stability limits. Static geometry and
mass calibration follow native IISPH; a shared contact-distance precision
mismatch is corrected. Standalone IISPH's numerical defaults remain unchanged.

Recorded CPU controls cover TGV with dt/resolution sensitivity, matched IISPH,
sampled/analytic columns, contact pressure and a short dam break. The controls
complete but wall equilibrium remains noisy; no long-time/GPU qualification
is claimed. Raw evidence, videos and the rejected density-seed A/B are in
`.tmp/dfsph-native-2026-10-25/`. Broader tests and final status are recorded in
the method documentation and evidence index.


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

Preservation verification complete: 16,625 payload files, 11,657,141,225 uncompressed bytes, 7,212,926,353-byte read-only zstd archive; every member hash checked. SHA-256 `edc06911a3c1054d21a4d4c12f004ccb84a4bb3d85ce6e1273b2fa3617b94921`. Local receipt retained with the Git-ignored investigation archive; original matrix has a KEEP marker and restore instructions. No cloud synchronization performed.

Curated baseline committed as `582b032da9512d3dd61070c94f0cc64c56ad6aaa` (16 scoped files; unrelated solver WIP left unchanged). A separately verified report capsule from that commit is retained alongside the full raw archive, so the report does not depend on this checkout surviving.


### Incompressible follow-up

General fixes cover pressure-unit scaling, projection initialization and physical
timestep limits, unified attached obstacles, and optional Band viscosity.
The experimental IISPH/reference-boundary investigation remains unpromoted.
Its drivers, frozen checks, source snapshot, notes and generated figures live
under the Git-ignored `.tmp/incompressible-investigation/`; raw runs remain
under `.tmp/`. See CONTRIBUTING.md for repository layout requirements.


User correction (2026-10-25): match IISPH's analytic initial-condition
approach. The original-half-cell analytic run used the earlier adverse control,
not the later established C4 inset fluid. Recover
`.tmp/incompressible-investigation/fluid-inset/run/initial_state.pt`:465 fluid,
31x15 exact dx spacing, dx-clearance from original walls, same calibrated
per-particle mass, no initial compression, zero p/v. Reduced rest volume465/512
is explicit. Run native IISPH and DFSPH sequentially on GPU FP32 with identical
saved inputs, verify actual initial density against that recorded calibration,
and retain all solver/viscosity settings. No C2-only wall fitter or new IC fit.


Corrected analytic IC comparison complete: both native solvers5s/5000steps
on GPU FP32; all12 initialized fields bitwise identical between schemes,
archived initial density matched within1.705e-7, max rho/rho0=1, masses
unchanged. Both first-step RMS9.60989mm/s; no1.97m/s initial impulse. Late RMS
10.4407mm/s IISPH vs6.21906 DFSPH; hydrostatic pressure L2 49.75% vs19.40%.
IISPH crosses2 sidewall particles transiently (depth1.012mm); DFSPH has zero
crossings and minclearance3.279mm. Neither solver hits a pressure budget.
Cost83.54s wall/11.77ms step IISPH vs96.57s/14.40ms DFSPH. Both movies decode
1001frames. Reduced fluid rest volume465/512 is explicit. Surface detachment,
corner motion, density deficiency and pressure excursions remain; no numerical
model/default changes. Evidence:`inset-analytic-comparison.png/json` in the
ignored baseline-reproduction directory, with complete per-solver artifacts.


User requests late pressure curves split by density/divergence solve
(2026-10-25). Saved endpoint has no divergence-pressure array, and previous
trajectory diagnostics contain no particle pressure profiles. Replay the same
465-particle analytic GPU FP32 case without numerical changes; record both
last-stage solve pressure outputs over4.5–5s, at every step, with pre-drift
solve positions. Plot height profiles and late temporal statistics. Keep all
capture/plot artifacts in the ignored baseline-reproduction folder. The absent
persisted history fields need separate qualification; do not alter numerical
carry while producing the requested observational replay.


Late-pressure capture complete:501frames over4.5–5s, all12 saved endpoint
fields bitwise equal to previous native DFSPH run. Direct density/divergence
solve outputs plotted as height profiles at4.50/4.75/5.00s and temporal mean,
5–95% spread and maximum. Late mean pressures2227.91Pa density vs3.75269Pa
DF; peaks7189.59/181.436Pa. Oscillations primarily density projection.
Replay97.42s wall/14.53ms step, no crossings/budget hits,1001movie frames
verified. Artifacts:`dfsph-inset-analytic-pressure-tail-gpu-fp32/`.

New qualification defect recorded before follow-up: boundary finalization
reconstructs DFSPHState in updateBodyParticlesWCSPH without the three dedicated
pressure/history fields, wiping them even on an analytic body with empty
particle masks. Earlier state-based 'sum' diagnostics actually contain CD
pressure only. Direct stage-pressure plots are valid; warm-history persistence
and its numerical influence remain unresolved. Keep this observational replay
unchanged; any fix requires isolated before/after verification.


User asks whether the fast late pressure oscillations occur in SPlisHSPlasH
(2026-10-25). Existing archived native reference uses4m x1.5m fluid and
30Hz pressure snapshots, with an invalid original pressure_max CSV decoder;
those cannot establish timestep-scale parity. Record an unmodified native
DFSPH reference on the same1m-wide tank and465 initial particle positions,
fixed1ms timestep, every-step CD/DF carriers and physical pressure conversion.
Keep native cubic kernel/support(2dx), mass convention(.8 cell volume), volume
maps, relaxation/warm starts and standard-viscosity coefficient explicit:
this is a geometric/timestep reference, not an operator-matched A/B against
local C4/4dx/calibrated masses/contact-gravity walls. Disable native vorticity
for this plain tank. Preserve source hashes and a Vispy movie; no solver fix.


Native SPlisHSPlasH reference complete:5001steps to5.00082s using fixed1ms
and the identical465 initial position set. Native kernel/support/mass/walls
explicitly differ. Native first-step density is deficient but not compressed
(max.9123rho0). Every-step late CD pressure also chatters: std/mean13.05%
vs15.42%local, dominant bin479 vs499Hz, close to Nyquist500Hz. DF mean varies
more slowly upstream(std/mean28.9% vs73.7% local). This is not evidence of
acceptable pressure or operator parity. Native3.80sCPU/.760ms step,149Vispy
frames verified. Source and binding hashes, direct carriers/decoded pressures,
figures/CSVs and all limits in`.tmp/dfsph-splish-pressure-2026-10-25/`.

Clarification: local fixed-step density warm start survives through the
previous-physical-pressure fallback even when densityPressureImpulse isNone;
the dropped divergence impulse does disable DF carry. Do not attribute CD
chatter to lost density history without an isolated check. Both upstream warm
macros are enabled and carriers persist. No numerical change made here.


User asks whether native SPlisHSPlasH has the same detached surface gap
(2026-10-25). Read recorded tail positions only; no further simulation. Align
central free surfaces for a physical-scale particle-layout figure and quantify
the largest particle-free horizontal interval within the upper0.1m for
abs(x)<.35m. At endpoint native14.50mm vs local44.62mm; late median16.87 vs
44.98mm. Native has a dense flat surface band but no similarly large persistent
void. Local separation is visibly stronger, not just a lower overall native
water level. Measure definition and kernel/support/mass/wall differences remain
explicit; this geometry observation does not isolate a numerical cause.
Artifacts:`surface_gap.py`, `surface-gap.json`, `surface-gap-comparison.png`
in the existing ignored native-reference folder.


User authorizes broader follow-up after tank comparison (2026-10-25): run
native DFSPH through the same IISPH dam-break setup for particle and analytic
walls, then TGV with the same IISPH initialization. Reuse the latest recorded
Marrone3.1 shared-viscosity IISPH inputs:7200fluid, dx=.01, C4/nh4, calibrated
masses, nu=.01 two-sided mirrored free-slip walls, contact-Shepard plus gravity,
analytic outward offset.0005m, dtceiling.001/cfl.2, t=2.45s and retained stops.
TGV reuses the exact saved4096-particle IISPH production initialization after
32 native IISPH relaxation passes with seed1234/jitter.01; C4/nh4, nu=.01,
signed pressures, omega.25, dtceiling.001/cfl.3, t=2.001s. No source-centering
experiment is imported. All runs GPU FP32, sequential, Vispy/progress and
full initial-field parity checks. Retain the current solver, including the
known boundary history-loss limitation, without mixing a fix into these runs.
Evidence isolated in`.tmp/dfsph-followup-2026-10-25/`.


Requested three-case DFSPH follow-up complete on GPU FP32. Saved full
Marrone states match after one cast, and saved relaxed TGV positions/v/m/h
and initial density match exactly; all masses unchanged, physical dt caps
obeyed. Particle/analytic dam breaks reach2.45s in3847/5140steps,117.01/152.17s
wall and24.02/23.59ms per measured step. Crossings peak2/3particles, deepest
1.806/8.629mm; CD budgets3/3, DF0/0. Front extent is close to archived IISPH,
but pressure remains unqualified: DF outliers6.995e17Pa sampled/1.195e9Pa
analytic with bounded velocities. Known boundary-history loss retained.

TGV reaches2.001s/2001steps in47.18s/17.43ms step. KE fraction.47128 vs
analytic.92308 and saved IISPH.47007; excessive decay remains (~−48.94% vs
analytic). CD reaches budget251times, DFnever; native source/gauge policy is
unchanged. All3videos decode771/1029/401frames. No solver/default tuning or
additional model change. Complete data, exact reference choices and limits in
`.tmp/dfsph-followup-2026-10-25/README.md` and `comparison.png/json`.


User challenges the near-identical IISPH/DFSPH follow-ups and identifies TGV
symmetry breaking/distortion as preceding rapid energy loss (2026-10-25).
Audit setup/dispatch, periodic initial-state symmetry, actual per-stage
velocity changes and their survival through finalization before interpreting
similarity as method agreement. Analyze recorded TGV symmetry over time, then
short recorded instrumented prefixes using the identical saved inputs: log
mode call counts, configured/actual tolerances, DF correction, kinetic work
and final-state velocity/position closure. No source centering, pressure
retuning, history fix, or long rerun is mixed into this audit. Evidence stays
in`.tmp/dfsph-setup-audit-2026-10-25/`.


Setup audit complete (2026-10-25): logged actual DFSPHState and one DF+CD
call per step in recorded0.4s TGV and0.15s wall prefixes. DF velocity kick and
finalized x/v match the staged map exactly; no alias to IISPH or cancellation.
The observational TGV replay matches the prior0.4s state bitwise. Frozen signed
CD solves on identical predictors agree with IISPH to~8e-4Pa/~3.4e-4m/s²;
CD algorithms share the same linear system and iteration algebra. The added
DF correction is small under this setup: TGV median7.79e-5m/s vs flow~.704m/s,
stop statistic max.0087 of tolerance. Closeness is a weak qualification test.

TGV reflection-velocity mismatch exceeds.1 of initial RMS at.21s; energy
deficit exceeds1% at.22s. The pressure stage extracts KE during.20–.25s;
thereafter nonpressure/viscous work dominates as distortion grows. Do not
interpret this merely as viscosity overdamping. Jitter=.01 is a fraction of
support, not dx: per-coordinate std=.04dx atnh4. Native relaxation leaves
position reflection RMS.06046dx, from.08117dx in the jittered input. Earlier
claim that relaxation amplifies it was corrected. Source-centered IISPH
control has identical initial four fields and KE.8590, unlike production.4701;
the old DFSPH comparison reproduced the shared failing signed-pressure path.

Confirmed and repaired lifecycle bug: rigidBody/update.py now preserves the
three dedicated DFSPH pressure/history fields during reconstruction. Added
sampled+analytic-body regression coverage in test_particleIdentity.py.37
focused DFSPH/body checks and the complete14-test identity suite pass. Short
isolated before/after wall replays confirm histories survive every step and
match p_div, p_CD*dt²,p_div*dt exactly, with identical initial fields. Endpoint
velocities change by up to~.025m/s at.15s, so old full dam-break runs remain
pre-fix evidence and require requalification. TGV has no body update; its
histories already survived, so this fix does not solve its symmetry problem.
No pressure tolerances, source policy or initialization changed. All5short
movies decode. Costs, measurements, source snapshots and plots recorded in
`.tmp/dfsph-setup-audit-2026-10-25/README.md` and `audit-summary.json`.


Audit baseline clarification: recovered source-centered IISPH TGV control has
bitwise-identical initial x/v/m/h and retains.8590KE, whereas production and
DFSPH follow-up retain~.470/.471. The close production curves were not the
strongest available IISPH comparison. No source-centering policy was promoted
in this audit; work now needs to target the shared signed-pressure/symmetry
failure before broader method qualification.


User correction (2026-10-25): a velocity-divergence projection must be tested
as the standalone incompressibility mechanism; a negligible correction is
not sufficient evidence that the DF solve works. Check source/operator units,
reference integration and stopping policy; test its response to a known
compressive velocity field and record a divergence-only TGV isolation on
identical saved initial fields. Keep experimental integration variants in
`.tmp/dfsph-divergence-isolation-2026-10-25/`, with explicit force/projection/
drift order and density drift measurements. No tolerance/default promotion
without this evidence.


Independent divergence isolation complete (2026-10-25). Corrected the earlier
interpretation: small DF kicks in the coupled run did not qualify independent
pressure response. On identical GPU FP32 saved TGV fields, disabling only CD
yields KE .85849 / velocity error .07280 at2.001s, vs coupled .47128/.30734;
analytic KE .92308. At.4s standalone retains .98078 vs analytic .98413 and
its pressure shape correlates .9981 with analytic pressure. Density drift
remains [.98525,1.04173]rho0 and later symmetry loss persists, so divergence-
only is diagnostic, not a production scheme. Fixed-geometry pressure-gradient
input converges from C(v) RMS .0982/s to1.34e-6/s. Reference dt/scaling/order
checked again: units agree; default DF threshold is permissive for this TGV.
Tightening DF error .1→1e-5percent alone fails to rescue coupled motion
(.80337KE at.4s vs .81011original) and hits100iterations every step. Tight
standalone also accumulates spurious mean pressure and396/400budgets. The
shared signed-pressure density/mean-mode correction remains the repair target.
Added3analytic Fourier-mode regressions for independent DF incompressibility,
solenoidal preservation, momentum and dt/rho units; all36focused DFSPH checks
pass. All4accepted new movies decode. Full standalone22.65s wall/5.115ms step;
short default/tight standalone/tight coupled7.90/29.70/41.88s wall and
5.33/62.34/92.59ms step. A decimal-endpoint extra1e-12step extension retained
as harness evidence and rerun at exact saved endpoint. Data, costs and limits:
`.tmp/dfsph-divergence-isolation-2026-10-25/README.md`. No numerical defaults
or production integration changed.


User interpretation (2026-10-25): distinguish physical velocity projection
from density-driven particle regularization, as in VD+PS. The anisotropic
divergence-only distribution may be the familiar unshifted particle-sampling
problem, not failure of the pressure dynamics. Audit the implemented VD+PS
position shift/velocity transport and its primary source; quantify directional
sampling in the saved TGV trajectories. Treat a CD-as-shift extension as a
separate method choice, rather than silently relabel standard DFSPH. Evidence
in `.tmp/dfsph-vdps-interpretation-2026-10-25/`.


VD+PS interpretation audit complete (2026-10-25): Cornelis2019Eqs12–19
explicitly separate physical VD pressure from a DI position correction and
Eq17velocity resampling. Existing local VD+PS finalize implements this
separation (legacy case/source policies differ); nativeDFSPH currently puts
CD into velocity and bypasses that hook. Anisotropy is not itself evidence
of failed pressure dynamics: exact TGV advection of the same initial samples
also forms strongly stretched strings. Read-only RK4 reference is converged
to1.5e-13m; finalP05neighbor spacing .232dx analytic-unshifted/.414dx DF-only
vs .799dx coupled, from .913dx initially. P95C4gradient-moment anisotropy
.667/.113/.00308 respectively. Paired reflection diagnostics also mix sample
motion with field error: analytic velocity evaluated at final DF-only sample
positions has .0833paired mismatch vs .1001actual. Recorded this qualification;
the independently measured7.28%velocity error remains. Any native VD+PS-style
control must apply CD as displacement plus field resampling, measure geometry
and dynamics separately, and avoid predicting advection twice. No new SPH
run or numerical/default change. Evidence and plots:
`.tmp/dfsph-vdps-interpretation-2026-10-25/README.md`.


Source-order clarification (2026-10-25, user question): rechecked original
2015Algorithm1 and2017extension. Both put NP prediction→CD velocity
correction→drift→rebuild/density/factor→DF velocity correction inside one
paper iteration. Across the periodic loop DF feeds the next NP/CD stage.
Pinned SPlisHSPlasH and nativeDFSPH place DF at the next iteration start,
the cyclic phase convention already documented. UpstreamomniSPH timestep
and localomni use NP predictor→DF→CD→kick/drift (upstream adds XSPH before
integration). Both corrections feed velocity; no VD+PS position-only density
correction. Correct prior implication: adding CD to velocity is standard
DFSPH and is not by itself the diagnosed defect. VD+PS is a separate coupling
variant; the ablation identifies a failing CD path without proving that the
published DFSPH coupling must be replaced. No solver/code changes.


User requests ordering matrix (2026-10-25): test all three compositions on
saved TGV and calibrated hydrostatic inputs: (1) DF→NP→CD→drift,
(2) NP→DF→CD→drift, (3) NP→DF→drift→CD position shift and velocity
resample. Include a matched NP→DF→drift hydrostatic control to test whether
divergence alone supports gravity. Retain original solver policies, mass
renormalization and initial fields, GPU FP32, sequential video/progress/caps.
Use sampled tank and saved inset analytic tank, declare post-drift geometry
and position-only source, measure physical/resampling work separately and
keep variants in `.tmp/dfsph-order-matrix-2026-10-25/`. No default promotion.


Three-order matrix complete (2026-10-25):12recorded GPUFP32 runs with exact
saved TGV and calibrated sampled/analytic tank inputs. Private variants only,
shared C/F/diagonal, original solver settings. First-step native-order parity
verified; all12input/mass/static-wall/stage/history/finalization checks pass;
7986video frames decode, including the recovered numerical-failure prefix.
TGV2.001s: orders1/2 KE .47128/.47100 and velocity error .30734/.30767;
order3shift/resample KE .91221 vs analytic .92308 and error .01009. Matched
DF-only control .85853/.07276. P95gradient-moment anisotropy .00468shift
vs .11212unshifted. Shift solver still hits budget1826/2001times; no default
promotion. TGV wall/step costs52.80s/18.95ms,50.71/18.94,138.03/62.39;
DFcontrol22.78/4.78.

Hydro5s: sampled late RMS7.408/7.545/191.315mm/s for orders1/2/3; pressure
errors16.89/15.08/7.51percent, zero crossings. Analytic6.992/6.582/161.553
mm/s and23.30/22.61/8.41percent; order3two transient crossings, deepest
2.97mm, orders1/2none. PS maintains geometry while opposing large downward
physical velocities, so its apparently good pressure/density does not qualify
hydrostatic rest. PureDFsampled fails contact support after.953s/953steps
(16crossings, deepest.307m); analytic stops at.918s/918steps at24crossings
(>5%, deepest.120m). Original exception/data retained and valid-prefix
Vispy frames encoded without changing numerical failure. All physical wall
gravity extrapolation forces counted once in the predictor; PS source after
drift uses zero velocity and no physical gravity.

Frozen initial gravity projection diagnoses both tolerance and boundary/setup
limitations: defaultDFleaves8.914/9.384mm/s sampled/analytic. Tight1e-5percent
with cap1000converges in250/188iterations and leaves.202/7.832mm/s. Analytic
clamped convergence still allows expansion mean.001929/s and raw RMS.006986/s
vs mean compression9.91e-5/s. Hence tightening alone does not solve the
analytic/inset equilibrium response. No long-time tight-tolerance qualification
claimed. Hydrostatics and shifted-TGV require separate correctness gates.
All costs, explicit equations/gradient/wall-force placement, validation and
plots in `.tmp/dfsph-order-matrix-2026-10-25/README.md` and `RESULTS.md`.


User authorizes long-time tight DF-only hydrostatics (2026-10-25): repeat
NP→DF→drift for both exact saved sampled and analytic/inset tank inputs,
changing only DF maxErrorPercent .1→1e-5 and cap100→1000. Target5s,
GPUFP32, original physical/warm-start/deficiency policies, recordings and
watchdogs. Record raw compression, expansion, guarded convergence, velocity,
column geometry, budgets and cost. This resolves the missing long-time test;
frozen one-step convergence alone does not establish hydrostatic support.
Private drivers/data in `.tmp/dfsph-order-matrix-2026-10-25/tight_*`.


Tight DF-only hydrostatics completed (2026-10-25): saved calibrated inputs,
predictor→DF→drift, GPUFP32, error1e-5percent, cap1000, original warm/
clamped/deficiency/physical policies. Both reach5s/5000steps, zero crossings.
Sampled5000/5000solves converge,231.05mean iterations, late RMS7.199mm/s,
height loss.593mm, late pressure error7.805percent;1004.46s wall/195.30ms
step. Analytic4983/5000converge,17cap hits only0.032–0.048s,398.77mean
iterations, late RMS41.611mm/s,14.892mm height loss,27.222percent pressure
error;1685.66s wall/331.22ms step. Worst analytic reported residual1.795x
tolerance; not an all-converged trajectory. Tight DF supports both columns;
sampled residual speed comparable to coupled, analytic remains active with
circulation and poorer equilibrium. Compression convergence alone does not
establish hydrostatic rest. No model/default promotion; settings very costly.

Combined execution session terminated externally at analytic1.877s. Driver
lacked periodic particle saves; preserved prefix/video and replayed original
inputs with checkpoints in an independent session. All1878scalar prefix rows
exactly match; no numerical stop. Interrupted extra prefix cost~834.15s
wall/~444.41ms per valid step including startup (filesystem estimate). Two
completed videos decode2002frames; exact input fields, calibrated masses,
static walls, stage/history/finalization/dt checks pass. Evidence, costs,
controls, limitations and plots:
`.tmp/dfsph-order-matrix-2026-10-25/TIGHT_RESULTS.md`.


User requests analytic wall placement calibration (2026-10-25): the
DF-only analytic run forms a dilute-looking bottom layer. Fit the bottom
and side wall planes against rho/rho0=1 in the nearest initial lattice
layer, using flat regions outside corner/free-surface support. Preserve
all465saved fluid positions, masses, support, zero velocity/pressure seed,
wall packing factor, viscosity .01 and tight DF policies. Leave the remote
ceiling fixed. Evaluate actual C4 summation plus analytic solid integral;
do not invoke the existing C2-only helper that also changes masses.
Then repeat5s GPUFP32 with video/checkpoints/watchdogs and compare bottom
sampling, density, dynamics, convergence and cost against the fixed-wall
control. Private work: `.tmp/dfsph-wall-fit-2026-10-25/`.


Density-fitted analytic wall run complete (2026-10-25): actual GPUFP32 C4
summation fit, masses/packing/all465saved particles fixed. Floor moves up
14.824702mm, sides inward same amount; bounds(-.485175298,.014824702) to
(.485175298,2)m. Nearest flat-layer clearance.52560954dx; rho/rho0 changes
.854816→1exactly at fitted sites. Second row remains+.5752percent and
corner maximum+1.0882percent; only nearest flat layers are fitted. Ceiling
and free-surface deficiency preserved. Existing C2 mass-changing helper
not used.

Same tight DF-only policies, nu=.01fluid-only, GPUFP32,5s/5000steps: all
solves converge, zero crossings. Late RMS7.946mm/s vs41.611control, height
loss1.913mm vs14.892, pressure error12.515percent vs27.222; mean236.19
iterations vs398.77. Wall1030.27s/200.07ms step vs1685.66s/331.22ms.
Initial field/mass/ID/native-first-step/stage/history/finalization/dt checks
pass, both1001-frame videos decode. No production/default changes.

Fresh post-drift densities show old original-bottom IDs largely preserve
initial deficit:.854816→.855582rho0, whereas fitted1→1.000489rho0. Central
lowest17clipped-Voronoi geometric packing at5s improves.735913→.983176;
lowest-to-next17height-band gap1.36707→1.02584dx. This supports the user's
dilute-bottom-layer diagnosis as an initialization/wall-compatibility issue.
The shifted physical container is explicitly recorded; corner/general-IC
qualification and remaining residual motion/pressure errors remain open.
Evidence, formulas, masks, limitations, recordings and plots:
`.tmp/dfsph-wall-fit-2026-10-25/README.md`.


User requests fitted-tank ordering comparison (2026-10-25): run
DF→NP→CD→drift, NP→DF→CD→drift and NP→DF→drift→CD shift/resample
on the same465particle density-fitted analytic tank. Include completed
NP→DF→drift as fourth reference, preserving its direct per-step CD/DF
pressure trace and measured costs. Keep masses/fields/wall planes/nu=.01,
GPUFP32, DF1e-5percent/cap1000 and CD.01percent/cap100 fixed. Record
direct pressure fields every step in the new runs, physical mean/max and
late1ms waveforms for all four, final height profiles, solver budgets,
velocity/height/density and wall/step costs. Auxiliary PS multiplier stays
separate from physical pressure. Sequential checkpointed runs, video,
progress and watchdogs; no numerical/default changes. Private evidence:
`.tmp/dfsph-fitted-order-matrix-2026-10-25/`.


Fitted-tank three-order comparison complete (2026-10-25): three new
GPUFP32 runs plus retained matched DF-only, all5s/5000steps, zero DF/CD/
shift budgets and no crossings. Exact initial fields including density,
masses/IDs, fitted planes, numerical configurations, production source
snapshots and shared stage implementation match; stage order/history/
finalization/impulse scaling/dt checks pass. Three new direct pressure
archives match scalar traces and final fields; all4004video frames decode.

Late4.5–5s RMS mm/s / pressure instantaneous-relative-L2 percent / signed
height change mm / detrended sigma(mean physical p) Pa / wall s / ms step:
- DF→NP→CD→drift:3.786 /21.907 /-12.465 /412.868 /433.59 /80.64.
- NP→DF→CD→drift:17.326 /29.449 /+.966 /1.079 /1011.27 /196.07.
- NP→DF→drift→shift/resample:6.077 /10.041 /-1.526 /3.737 /1030.04 /199.86.
- DF-only reference:7.946 /12.515 /-1.913 /5.687 /1030.27 /200.07.

All physical pressure means~2.3kPa; DF-first oscillates~399Hz with mean
CD2182.13Pa and DF126.13Pa. Predictor-first CD is physically tiny in the
late mean(.042Pa), DF supplies~2293Pa. Shift variant mean physical DF
2304.26Pa; auxiliary shift multiplier2.989Pa is excluded. Other curves
peak near499Hz but have far smaller absolute fluctuation amplitudes. A
smooth global mean does not imply an accurate spatial pressure field.
Shift/resampling improves late RMS23.5%, pL2error19.8%, sigma(mean p)34.3%
over DF-only at equivalent measured cost. DF-first is cheaper/quieter in
velocity but noisier in pressure and loses more height. No default chosen;
remaining initial second-row/corner density excess is explicitly preserved.

New pressure capture overhead~.155ms/step, included in runner wall time
but outside the engine step timer; baseline
uses its original every-step direct mean/max pressure and final fields,
without synthesizing a dense particle history. Full-rate global curves,
50ms details, split physical CD/DF and auxiliary PS, absolute spectra,
instantaneous central height profiles, final fields, CSVs and all costs:
`.tmp/dfsph-fitted-order-matrix-2026-10-25/README.md`.

### Native SPlisHSPlasH on the fitted tank (2026-10-25, user)

User requests the native reference on these same initial conditions. Preparing a private recorded native run in `.tmp/dfsph-fitted-splish-2026-10-25/` from the exact saved particle positions/masses and fitted planes, matching support and viscosity/tolerances where available. Native cubic/volume-map operators and CPU FP32 remain explicit differences; initialization and effective fluid volume must be checked, not just reported particle masses. No production/default changes authorized by this comparison.

Native fitted-tank reference completed: saved465particle positions/velocities/masses bitwise equal; effective native neighbor volume matches m/rho0 within2.48e-8 relative (setting mass alone would be incorrect). Same fitted mesh planes, requested DF/CD tolerances, nu=.01, zero boundary viscosity and1ms; CPUFP32 native cubic, volume maps at scene resolution30x30x20 and fixed0.5 relaxation retained. Strict4dx comparison fails first CD step/cap100 with maxspeed8.14e13m/s. Initial central first/second layer density1.422/1.463rho0 (max1.672); the local C4 fit is not a native-compatible rest state. CachedW(0) refresh control reproduces failure. Usual2dx control retaining mass/planes also caps100 and stops at25.19m/s, above fixed1ms CFL6.25m/s; initial closest layer1.503rho0 (max1.633). This control shows a large startup kick, not proof of iterative instability.

Offline reconstruction matches native factor/acceleration <6.4e-7 relative and actual100 clamped pressure iterations1.04e-5/1.29e-7. Strict4dx largest normalized eigenvalue6.4905 exceeds linear stability bound4 for omega=.5 (bound omega<.3081); usual2dx eigenvalue2.2191 passes. Native source unchanged. No late-time native pressure samples exist and no late comparison fabricated. Native first-step timers1.203/.852ms, driver walls1.089/1.053s include map construction, encode.815/.806s separately. These failed startup CPU measurements cannot rank full-run GPU cost. Both two-frame Vispy videos decode, sources/input identities/mass-volume preservation verified. An earlier duplicate4dx empty-tail export error is retained and video recovered from saved fields. All work private and sequential; report `.tmp/dfsph-fitted-splish-2026-10-25/README.md`.

### Native low-density initializer replay (2026-10-25, user)

Reverse comparison requested: use native SPlisHSPlasH low-density particle masses and original tank geometry as the shared physical initial condition, then compare native and local DFSPH orderings forward. Private evidence in `.tmp/dfsph-native-init-2026-10-25/`; preserve reduced mass, zero seed and recomputed density per actual operator without fitting walls or normalizing density back to rest. Native kernel/support compatibility is being checked before the recorded sequential run matrix.

User selected native0.1% DF tolerance for both implementations after the1e-5% replay required~950local iterations/step. Preserve completed tight native and interrupted local prefix/checkpoints; run fresh native/local outputs under the selected tolerance, retaining all other policies.

Native low-density comparison completed under the user-selected0.1% DF bound. Exact465native particles, mass0.78125kg/particle (total363.28125kg), cubic/support2dx, original walls x±.5/floor0, zero seed, no renormalization/wall fit. Central bulk starts0.800689rho0 for both; closest flat native/local density0.879579/0.703136 because volume maps and continuum analytic walls remain different. Shared nu input.01; local rho0/meanrho viscosity factor and near-zero regularizer differ from native constant-nu/.01H² stencil, explicitly retained.

Native/DF-first/predictor-first/shift complete5000steps (~5s; native FP32 clock−.180ms); late RMS mm/s21.918/2.634/6.940/309.983; physical mean pressure Pa1916.76/1759.73/1758.58/1776.76; detrended sigma(meanp)237.30/52.49/40.61/663.30; pL2%24.31/15.90/17.01/34.75. NoDFcap hits. Shift4781PS caps and auxiliary mean multiplier63.8kPa excluded from physical pressure; guarded compression~.989/s is near selectedDFbound1/s. Local3orders have2peak crossing particles; native0. DF-only stops at.341s/341steps with25crossings (>5%), finalRMS234.65mm/s and8.94mm maximum signed penetration; no late curve/ranking synthesized.

Native/local coupled height loss95.82/133.78mm at exactly equal mass. Native width.94561m/bottom17gap29.925mm versus local width~1.002m/gap~8.6–8.9mm; supports boundary/sampling origin of remaining height/pressure difference. Central upper empty gaps native18.59mm versus local15.94/14.41mm (shift26.29mm), not the prior very large separation. Native driver7.65s/engine.462ms; local walls/engine152.17s/24.545ms,150.93s/24.345ms,511.78s/96.313ms, stoppedDF-only8.43s/11.253ms. DifferentCPU/GPUtiming and recording cadences declared. Exact inputs/mass/IDs/initial local densities, source snapshots, direct fields versus traces/final state, stages/history/finalization/dt and3220selectedvideo frames verified. Tight native5s and interrupted local.717s prefix/checkpoint.700s/144Vispyframes retained separately at user tolerance change. Full report/curves/videos: `.tmp/dfsph-native-init-2026-10-25/default-tol/README.md`. No production/default change.

User interpretation (2026-10-25): classic ordering is preferred on this hydrostatic comparison by RMS speed, with predictor-first/omni-like ordering a close alternative on speed/noise. Shift/resampling is not worth its cost here (4781density-shift cap hits also qualify that measurement); no default change requested. Read-only periodic support audit of pinned native reference: no periodic/TGV scene or configuration path in fluid/Simulator/Python sources; neighborhood wrapper has radius and point sets but no periodic box interface, fluid kernel macros use raw xi-xj and drift does not wrap. Official scene parameter documentation likewise lists no periodic-domain setting. A periodic TGV reference needs an explicit extension for seam neighbors/minimum-image interactions and wrapping, or correctly tied periodic images refreshed within each pressure iteration; wrapping positions alone is insufficient. No new simulation or reference-source modification performed.

Closed random-flow decay candidate (2026-10-25, user proposal): both frameworks support enclosing static walls with gravity disabled. Existing warpSPH `randomFlowIncompressible` has a bounded variant, but its default `divergenceFree` and scheme-specific hook are legacy, so a new native-DFSPH comparison must wire the actual `dfsph` stage explicitly. Its current sampler rotates an SPH gradient of a boundary-ramped random scalar potential; this is not proof of exact discrete divergence-free or wall-normal compatibility in both operators. A shared analytic band-limited sine streamfunction with deterministic random coefficients and psi=0 on all box walls gives continuum-divergence-free, zero-normal velocity without per-method noise sampling. Proposed baseline: fully filled rest-compatible box, g=0, no forcing/wall drag/filters, nu=0 first for numerical energy loss/gain, then a matched-viscosity control. Save one actual particle/mass/velocity state; measure initial discrete divergence/wall flux and first-projection work separately, plus K/K0, enstrophy, density drift, wall crossings and cost. Do not blindly reuse the0.8rho0 free-surface tank initializer: density repair would confound decay. Boundaries remain part of the closed-box comparison; no periodic native extension needed. This turn verifies capabilities/design only; no simulation or source change launched. Official native scene/material parameter docs verified at https://splishsplash.readthedocs.io/en/stable/file_format.html.

Closed decay setup authorized (2026-10-25, user): use existing bulk/particle-wall mass configuration and analytic-domain sizing. Private fixture/run/plots in `.tmp/dfsph-closed-decay-2026-10-25/`. Initial comparison scope: native SPlisHSPlasH plus classic and predictor-first local DFSPH on particle and analytic walls, inviscid/no gravity/no wall drag/filters; validate zero-motion density-compatible starts before random-flow trajectories. Reuse current native0.1% DF and0.01% CD tolerances. Exclude previously rejected shift/resample branch. Keep changes private; no numerical default promotion.


### Closed inviscid random-flow comparison completed (2026-10-25)

Private evidence: `.tmp/dfsph-closed-decay-2026-10-25/README.md`, `decay.png`, `pressures.png`, metrics and source/video checks. Reused ParticleRegion initial mass/velocity controls; no production feature/default changes. Shared1024-particle32² fluid,420 sampled solids, calibrated rest-volume0.000975721574, cubic2dx explicit native-matching override, zero gravity/viscosity/wall drag/filter/shifting, seeded analytic-curl random field RMS0.2m/s with a zero-velocity wall collar. Native Akinci samples/volumes match the saved lattice and effective fluid volume exactly, not just mass; CPUFP32AVX4OMP versus localGPUFP32. All flat analytic planes fitted atfixed fluid mass to±0.502376071m, closest clearance0.576034dx; full initial density0.999082–1.001796.

Classic/predictor-first frozen outputs bitwise identical because the zero nonpressure predictor is identity; duplicate trajectories unnecessary. Existing IISPH contactShepard pressure closure causes energy injection and CD caps in the closed moving-flow fixture: sampled stops0.959s, peakKE50.08K0,903CDcaps; analytic stops0.768s,peak39.77K0,380caps; no crossing, actual CFL watchdog. Retained prefixes/videos are failures, not extrapolated5s decay estimates.

Native Akinci wall pressure uses only fluid p_i; changing only the existing local boundaryPressure setting to self establishes a matched boundary-pressure control. All5000steps complete: K/K0 native0.205565, local sampled0.219422, local analytic0.149088; no DF/CDcaps or crossings. Density RMS final0.128/0.146/0.441%. Native clock4.999820s due FP32 cumulative timing, local5s; same1ms integrator steps. Native/local sampled energy curves and final particle patterns are close. DF stages all1iteration yet account for73%/82% of local sampled/analytic energy loss; CD handles the remainder, zero external work. Stage/finalization/mass checks pass and cumulative stable-run work closure<3.7e-5J.

Native/local sampled rest controls stay exactly motionless with zero pressure. Analytic5s self-pressure rest RMS0.002346m/s, final energy0.01376% of random initial energy; peak0.151% and minimum density0.81475 near second diagonal corner rows show that flat wall fitting alone is not a complete corner equilibrium. Analytic decay remains an imperfect wall comparison. Native engine0.476ms/step, driver3.23s plus1.54s encode; local sampled12.346ms/73.24s wall, analytic10.150ms/62.00s, recording included in local wall timing. Every physical run sequential;12Vispy videos fully decoded, input/mass/volume/source identities and work diagnostics verified. Retain sampled native-style baseline; closed-domain contact-pressure compatibility and analytic corner sampling remain unresolved. No ordering ranking or default promotion from this zero-predictor fixture.


### Nominal-volume closed sampled-wall replay (2026-10-25, user)

User identifies mass removal by initial density calibration: prior fluid rest-volume sum0.999138892 differs from physical box area1. Restore nominal Δx² fluid and particle-boundary volumes, so1024fluid volumes sum exactly1, mass totals1000 per unit depth; keep positions/velocities/support, raw cubic density estimator, 1ms/tolerances and self wall-pressure baseline unchanged. No initial relaxation, kernel renormalization, density clipping or projection of the physical initial state. Replay native and local sampled-wall baseline, retaining the original calibrated evidence; private subdirectory `.tmp/dfsph-closed-decay-2026-10-25/nominal-volume/`.


Nominal-volume replay complete (2026-10-25): exact common positions/velocities/support and physicalbox1m²; fluid and420sampled-solid volumesΔx²=1/1024, totalfluidmass1000, Σm/ρ0=1. Raw initial density1.0008618–1.000862rho0 left unchanged; no kernel/mass correction, density assignment/clipping or physical-state projection. Initial inferredΣm/ρ_i≈0.999139 reflects estimator bias and is separately recorded. Native effective neighbor volume is oneFP32ulp above nominal (relative1.192e-7), physicalmass exact; no reference source patch.

Native/local sampled self-pressure both reach5000steps, no wallcrossings/DFcaps: K/K0=0.349695/0.360278 versus calibrated0.205565/0.219422. Mass restoration is consequential but higher endenergy contains a large startup transient: native firststep+47.56%/peak2.253K0; local+18.86%/peak1.752K0; CDcaps27/39. Initial meanCDp55.12/31.92kPa. No post-kick renormalization. Local CD network+3.86055J versus calibrated−4.25214J; DF−16.65503J, workclosure<5e-5J. Late4.5–5s density RMS improves native0.16439→0.11890%, local0.14435→0.09982%; endpointlocalmin0.97542→0.98672 but lifetime minimum0.88083 duringstartup, so gaps remain. Native endpointminimum0.96407 despite improved late-averageRMS.

Native driver3.934s+1.527sencode/engine0.61688ms, local97.853s/17.227ms recording included; higherpressureiterationcost than calibrated73.24s/12.346ms. New complete native/local videos fullydecode; exactmass/state, scalar histories andwork verified; all runs sequential GPUFP32 locally. First native observer attempt aborted after1step on an overlystrict float32 comparison (2.44e-7 mismatch vs2e-7 diagnostic tolerance); physicalsettings unchanged in successful fresh5s replay, retained explicitly labeled initial-only recording, missing terminal arrays not fabricated. Private report/plots/sources: `.tmp/dfsph-closed-decay-2026-10-25/nominal-volume/README.md`. Original calibrated evidence retained. No default promotion; physicalmass consistency and quiet discrete initialization are distinct requirements highlighted by this replay.


### Damped jittered preparation of a nominal-mass closed IC (2026-10-25, user)

Keep physical particle-wall planes±0.5, nominal fluid/boundary rest volumesΔx² and totalfluidmass1000. Start from zero velocity and deterministic isotropic jitter bounded by0.01dx, fixed sampled solids. Use current local DFSPH self-wall operator with a separately logged exact linear drag v←exp(−20dt)v during preparation only. Do not calibrate mass, kernel or wall geometry. Run to a measured quietness criterion or explicit duration cap, then validate fresh zero-velocity/zero-pressure-history undamped hold before sampling the same saved analytic-curl random field at the relaxed positions. Share one resulting fixture between native and local recorded decays, damping absent. Characterize density, displacement, separation and anisotropy; small jitter may remain crystalline, so do not claim an isotropic glass from damping alone. Preserve all attempts privately in `.tmp/dfsph-glass-ic-2026-10-25/`; no production/default change.


Damped jittered IC preparation completed (2026-10-25): seeded uniform-disk max0.01dx jitter (RMS0.007071dx), zero velocities, physicalwalls±0.5 fixed, nominalfluid/solidvolumesΔx², totalmass1000 and Σm/ρ0=1 unchanged. Private exactlinearprestepdragγ20/s onlyduringlocalDFSPHpreparation. Settles0.754s/754steps after0.25s ofRMS<1e-5,maxspeed<1e-4,densityRMS<.001rho0,nocaps. FinalRMS3.22e-6m/s,24earlyCDcaps;11.83s wall/10.209msstep. Drag−15.9462J,DF−5.3197,CD+21.2659,closure<4.3e-6J.

Freshresetzero-velocity/zero-pressure-history **undamped**1s holds pass bothcodes: RMSnative1.93e-5/local1.63e-5m/s,zero caps/crossings,wall.546/12.10s,engine.376/7.736ms. Holds are independent validation, not extra geometry conditioning. Densityinitialmin0.978819 (diagonalnearcorner),RMS0.07818% persists; compression-only projection does not guarantee densityuniformity. RMS/maxparticlemovement0.1541/0.9088dx; nearestspacingmin0.6369dx, Q4 drops1→0.6168 so this is partiallyordered relaxation, not a fullyisotropicglass.

Same saved analytic-curl field coefficients/phases/amplitude resampled at these positions; no per-fixturevelocityrenormalization. SampledRMS0.1998878m/s, initialK19.977569J; mass/walls/viscosity0/gravity0/support2dx/timestep/tolerances unchanged. Native/local use bitwiseidenticalnewinputs, no preparatoryvelocity/pressurehistory or damping in flow. Both5000steps, no overshoot/caps/crossings: K/K0native0.281043/local0.293395. FirststepΔK−0.003170%/−0.001919% versusrawnominal+47.56%/+18.86%; initialmeanCDp7.48/3.92Pa versus55.12/31.92kPa. Grossstartupenergyimpulse removed while retainingphysicalmass. Late densityRMSnative0.08712%/local0.08946% versusrawnominal0.11890/0.09982%; earlyflowRMSstillpeaks~0.707%,minrho~0.898, so spatial/discretevelocitycompatibility remains. InitiallocalC(v)RMS0.025996/s versuslattice0.003999/s; continuumcurl doesnotenforcediscretedivergence.

LocalflowDFwork−10.05546J,CD−4.06082J,drag/external0,closure2.59e-5J. Nativeflow3.260s/.484msstep plusencode, local80.69s/13.857ms (privatewrapper adds2energyreductions/readbacks perstep evenatγ0); no hardware-normalized timingrank. All5newrecordings plus2retainedbaselinevideos fullydecode; initialfield/mass/wall/source, histories, noflowdrag andstage/finalization checks verified. Experiment/report/commonfixture: `.tmp/dfsph-glass-ic-2026-10-25/README.md`. Retain as a cleaner mass-preserving IC candidate; no production/default change and no isotropicglass or exact-density claim.


### Periodic nominal-mass TGV preparation (2026-10-25, user)

User requests the same zero-velocity/0.01dx-jitter/damped relaxation in periodic space, then sample TGV; warpSPH only. Preserve the established64²,WendlandC4/support4dx,2πperiodicdomain,nu.01 TGV flow settings; nominalΔx² masses, no lattice or kernel normalization. Private evidence `.tmp/dfsph-tgv-glass-2026-10-25/`. Track uniform raw-quadrature density bias separately (static C4 lattice1.00041575rho0 exceeds0.01% density tolerance) from fluctuations/velocity quietness. Begin with unchanged signed DFSPH source/operators, dragγ20/s in preparation only; retain cap/failure evidence rather than silently projecting away mean density. Fresh undamped rest check before claiming a quiet IC; nominal lattice flow control isolates IC changes. No native SPlisHSPlasH or production/default change.


Periodic preparation audit: direct raw signed nominal-mass projection repeatedly caps on mean density error (~0.04%) above0.01% tolerance and regenerates motion despite drag. Preserve this failed preparation prefix/full run. Add a separately named **preparation-only** source-mean removal control, relaxing fluctuations about the actual mean; no mass/kernel/rho field calibration. Main TGV source remains raw, so IC effects are not confounded with changing the evolution solver. Raw/centered fresh rest controls will distinguish mechanical quietness from convergence of the absolute-density constraint.


Fresh periodic rest diagnostic: the same prepared nominal-mass positions remain atRMS0.000135m/s under mean-free density source but raw source regeneratesRMS0.262m/s in0.25s/250CDcaps. Keep requested raw-source TGV lattice/relaxed pair unchanged. Add separately named mean-free-density-source **evolution** controls to distinguish an IC improvement from the known periodic mean-mode issue; this is an explicit solver-source variant, not an initializer effect or production change. All masses/raw density fields retained.


Geometric qualification before the final IC choice: jitter→mean-free preparation leaves Q4=0.999754 and only0.00506dxRMSoffset, i.e. a near-lattice rather than a glass. Direct raw2s preparation rearranges to Q4≈0.067/Q6≈0.642 but remains moving. Add a named two-stage candidate selected on geometry, not TGV energy: take the raw-rearranged positions, reset velocity/history, cool for1s with the mean-free preparation source andγ20; preserve masses. Validate raw/mean-free fresh rest and run a separately labeled mean-free-source TGV on this candidate. Keep the near-lattice and all raw-source evidence.


Complete the IC/source isolation for the two-stage rearranged candidate with a raw-source2sTGV replay as well as its mean-free-sourceTGV. This yields the same raw/mean-free evolution comparison for lattice, near-lattice relaxed and rearranged/cooled inputs; no IC claim relies only on the changed-source control.


Periodic no-mass-calibration TGV investigation completed (2026-10-25), warpSPH only.64²/[0,2π)²/WendlandC4/support4dx, nominalmass0.009638285264... each, sum39.47841644 vsFP32box39.47841980 (−8.51e-8relative rounding), fixedsupport/mass, no kernel normalization. Bypassed builtinTGV manualmasscorrection and scratchrelaxation, validated physicalinputs unchanged.0.01dx-boundedjitter/zeroV/γ20 drag duringpreparation only; flowsnu.01,1ms,rawsignedDFSPHpolicies .01%CD/.1%DF unchanged unlessexplicitlymean-sourcecontrol.

Direct raw2spreparation remains moving(RMS.02332,max.2965m/s) andcaps2000/2000CDsolves: rawnominalC4 densitymeanbias~.04% exceeds.01%tolerance, nearuniformperiodicconstantmodecannotbereliablyrepairedbyrelaxationalone. Separate source-mean-free5spreparation removesfluctuations(stdrho3.81e-7), terminalRMS8.27e-5, remainsnear-latticeQ4=.999754; strict1e-5RMSstopcriterion notmet, honestlyduration-limited. FreshrawholdRMS.2623 vsmeanfree.000135m/s/0CDcaps. Additionalgeometricallyselectedtwo-stageIC keepsraw2srearrangement and cools1s withsource-mean-freepreparation: Q4=.067165,Q6=.641434,RMSoffset.4877dx, nearestspacingmin.8976dx, notanisotropic-freeglass. Nomass/kernel/density/boxcalibration. TerminalcoolRMS.000110, freshrawhold.09545 vsmeanfree.000216m/s. All rawrestholds cap250/250CDsteps; centerednone.

Six2000stepTGVruns isolateIC×density-sourcechoice (rawvsb−meanb; divergence unchanged). Lattice/near-lattice/rearranged-cooled with rawsource retainKE3.97097/7.34922/7.17996%, velocityrelativeL2≈100–105%, all2000CDcaps: failed physicalbenchmark, not legitimate decayranking. Withexplicit source-mean controls,KE72.43251/77.85032/87.09919%, analytic92.31163%; velocityL2errors18.3607/14.3119/9.48975%, nocaps. Reflectionasymmetry4.036/5.094/5.419% so notallmetricsimprove. Final sampledmeanrho1.019409/1.014129/1.006077 showsremainingmean/samplingdrift; sourcecenteringisnotqualifiedasfinalincompressibilityfix. No claim that87.10%comesfromICalone. RearrangedinitialgeometrychosenbeforeTGVenergytest; allcandidate/failureevidenceretained.

Flowwall/enginems lattice/near/rearranged raw134.58s/63.868,134.07/63.738,131.81/62.618; source-meancontrols20.10/7.032,20.23/7.113,20.12/7.068. Two-stageone-timeprep142.96s; all13physicalrunssequentialGPUFP32withVispy/progress/watchdogs andper-runallcostsreported.13videosfullydecode; nominalmass/support/inputs, pressure-historyreset, noflowdrag,stage/finalization andenergyworkverified. Independentminimum-imageC4densitysummaxseamerror1.54e-6, translationinvariance~5e-15 rulesoutmissingperiodicseamcontribution; flowworkclosure<1.43e-5J. Retained keyword-adapter setupfailure tookzero physicalsteps. Private reproduciblefixtures`glass.npz`/`relaxed.npz`,drivers,source snapshots,table,plotsandreport `.tmp/dfsph-tgv-glass-2026-10-25/README.md`. Strategyreadyforexperimentation, no production/default change; periodicdensitymean/source compatibility anddensitysamplingdriftremainopen.


### Prepared incompressible comparison matrix (2026-10-25, user)

User authorizes updating the reusable comparison/tests to use nominal-mass jitter/damped preparation in closed/periodic domains, explicit periodic density-source compatibility controls, and calibrated free-surface sampling with analytic wall alignment. Include native IISPH, all three IISPH-based DFSPH compositions and the existing projection schemes; retain legacy results and report completion, physical accuracy and cost separately. Implement immutable shared fixtures and meaningful invariants before sequential GPU FP32 matrix execution. No outcome-driven tuning or promotion of experimental solver defaults.

Prepared matrix implementation: version6opt-in profile retains full historical resolutions/endpoints (256²TGV),8methods/120slots/101supported. Shared nominal-mass periodic/closed fixtures, explicit native periodicMeanFree evolution, calibrated C4free surfaces, static sampled contact contours, allnative pressure histories/physical pressure decomposition. Added C4flat-wall quadrature calibration; defaults remain C2/absolute/DF-first. Setup smoke failures retained separately; early missing contours and C2-only calibration corrected before full matrix. Legacy sampled closed walls must not double-count analytic solids and retain their ghost-node closure. Accuracy/runtimes report caps and source policy; shared preparation is a separate cost. Contract suites pass, including nominal-volume/ghost geometry and alternate uniform periodic transport. Full run awaits final setup checks; no new physical ranking yet.

Full prepared matrix launched sequentially on GPUFP32: `.tmp/incompressible-prepared-full-2026-10-25/`,120slots/101supported, native IISPH +3native compositions +4legacy routes.65,536-particle TGV, historical endpoints/resolutions retained; C2exception for the existing compact rotating-wall Couette closure. Corrected setup checks complete; native routes pass, one legacy Band sampled-box velocity stop remains a numerical result. Contract qualification131passes/13skips, plus14harness checks. All numerical sources frozen before full execution. Common preparation/physical fields/actual routing/source policy recorded; cap and volume/density diagnostics accompany ranking. Completion and physical agreement remain distinct; no production-default promotion.

Measurement qualification during the fixed matrix: legacy projection routes carry component pressures with different storage conventions. Added a read-only observer at the existing load-bookkeeping call; exact original arguments/results forwarded, physical components combined only during diagnostics, ambiguous legacy slosh component aliases omitted. Six observer identity/nonmutation tests pass. No pressure iterations/forces/timesteps/IC policies changed; diagnostic-only source receipt and exact patch are in `verification/diagnostic_source_changes.json` and `pressure_diagnostic.patch`, accepted separately by the evidence auditor. All numerical package sources remain pinned.

Full-resolution periodic TGV currently mixed: native DF-first/predictor-first reach5s at67.37/67.43%KE (analytic81.87%) but raw mean density reaches1.1093/1.1087; IISPH stops4.781s, native shift0.62558s. No native DF/CDcaps in the shift failure. Common physical IC hashes/mass/domain volume and independent seam density reconstruction verified(FP32-distance emulation error2.50e-7); no hidden mass calibration or missing seam neighbors. Retain fixed matrix unchanged. A separate4native-route,65,536-particle TGV control with the already-tested2sraw-packing+1smean-free-cool preparation is queued afterward in `.tmp/incompressible-prepared-tgv-rearranged-2026-10-25/`. It isolates near-lattice versus rearranged preparation at fixed mass/flow/source/tolerances; it is not pooled into the main ranking.

Before the first gravity-case execution, fixed another input-adapter contract: the existing hydrostaticInit profile had been written into pressures while zero DFSPH impulse histories suppressed it. The first physically limited timestep now encodes the same profile in CD p*dt²for coupled routes or physical DF p*dtfor shift/resample(auxiliary zero); IISPH retains its ordinary physical-pressure carry. No numerical operator/policy/time limit changes. `pressure_seed.json`/`first_step_state.pt` record exact first-solve inputs. Six dt/stage/nonmutation contracts added; initialization-source receipt/patch distinguish this input fix from diagnostic-only changes. Periodic/closed rows(hydrostaticInitfalse)are bitwise unaffected.

### Prepared incompressible matrix completed — 2026-10-26

Completed the full GPU FP32 profile at the original resolutions/endpoints: 120 slots, 101 supported attempts, 88 endpoints, 11 numerical alarm stops and two timestep-floor errors; 19 unsupported combinations remain explicit. Native classic and predictor-first DFSPH reach 13/13 endpoints, IISPH 12/13, native VD+PS 11/13. Legacy divergenceFree reaches 14/14, omni 12/14, reference 5/13 and Band 8/8. Completion is distinct from physical accuracy. All 101 evidence audits pass: source bundles, shared fixtures, recordings and 614,592 recorded timestep checks; no timestep-limit violation or measured fluid-mass drift. Two error rows retain partial trajectories/videos and lack terminal tensors. All 18 native first-pressure encodings independently match their actual first timestep and physical stage.

The declared per-case metric favors IISPH in analytic hydrostatics and both wedge representations; classic DFSPH leads sampled hydrostatics. The coupled native tanks have roughly 0.5–1.6 mm/s late RMS motion without crossings. IISPH generally costs less than native DFSPH in these static tests. Legacy divergenceFree leads the dam-break/sloshing pressure metrics, but sloshing error remains about 53–56%, and close analytic sloshing ranks against IISPH/omni are not significant. Every shear result has a large energy-curve error; a best color is not a physical pass. Native VD+PS at the shared tolerances retains about 0.37–0.38 m/s static residual motion, almost universal auxiliary density caps and some wedge penetrations. Legacy reference remains unreliable in periodic/dynamic cases and has static motion/height-growth defects.

Initialization remains qualified. The primary periodic preparation preserves almost square order (local Q4 .9998); the closed preparation reaches its duration cap, and its saved sampled/analytic fixtures already contain 2/26 particles outside the theoretical walls by .0175/.1889 dx. These are recorded separately from subsequent crossings. Main TGV coupled trajectories reach 5 s but retain only 67.37/67.43% energy versus analytic 81.87%, with about 8.5% of uniform periodic probes farther than 1.5 dx from a particle. Native periodic rows explicitly use the mean-free density source; this changes evolution as well as ICs and is not evidence that initialization alone repairs the absolute-density path. Historical common-slot completion improves IISPH from 4/12 to 11/12 in the retained v4 root, but changed kernel, physical presets, sampling and source policies prevent a solver-only old/new inference.

A separate full-resolution native TGV control used the prior 2 s raw-source rearrangement + 1 s mean-free cooling recipe at identical nominal mass, field and evolution settings. Classic/predictor retain 73.91/73.98% energy and reduce probe coverage gaps to about 3.9%; IISPH now reaches 5 s at 70.50% energy but still has substantial sampled-density drift and 11.5% coverage gaps. VD+PS stops at 1.621 s rather than .626 s. Preparation costs 141 s versus 32 s; both preparations exceed the strict quietness target. The control has strong local sixfold order and residual global square order, so no isotropic-glass claim is made. Its four evidence audits, independent periodic density/IC checks and actual private-helper hashes all pass. Control outcomes are separate from the primary ranking.

Six static closing-step responses are explicitly flagged and retained. The analytic coupled wedge runs end with a legitimate .153–.154 ms step after 1 ms steps, causing pressure jumps about 22× and elevated final velocities despite quiet tail medians. Native density history is stored as p*dt² and read using the current dt² with .5 damping, which amplifies its initial pressure guess on a smaller step; IISPH carries physical pressure and has a much smaller closing response. No warm-start policy, endpoint, tolerance or score was changed. Pressure-stage fluctuation plots, absolute pressure/RMS curves and terminal-step tables expose this difference.

Measurement corrections are explicitly receipted: common physical-pressure observation at existing bookkeeping/solve returns, and the first-step hydrostatic impulse encoding corrected before gravity rows. A providerless-reference observation gap affects six earlier raw pressure records; these have machine-readable qualifications and none completed a pressure-ranked case. Inputs/results are forwarded unchanged, with no numerical operator or force changes. The final harness suite passes 28 tests; the earlier numerical contracts pass 131 with 13 skips. No production default was promoted.

The reusable opt-in profile and protocol are in [docs/fluid_matrix/README.md](docs/fluid_matrix/README.md). Local evidence/report: `.tmp/incompressible-prepared-full-2026-10-25/README.md`, per-case quality/cost figure in `ranking/`, `STATIC_WATER.md`, `TGV_PREPARATION_CONTROL.md`, source receipts and independent audits in `verification/`. Raw tensor states, HDF5, videos, logs and source archives remain retained. The separate control is `.tmp/incompressible-prepared-tgv-rearranged-2026-10-25/`.

### Prepared results integrated into the normal matrix — 2026-10-26

User requests the normal classic-fluid comparison with the prepared results.
`scripts/compare_preparedFluidMatrix.py` selects the entire eight-method prepared
matrix and retains the original dated Delta-SPH, Sun delta-plus and ACSPH
controls. The separate TGV preparation control stays separate. The resulting
11-scheme comparison has165declared slots:113endpoints,14numerical stops,
2errors and36unsupported; the figure shows14case/wall columns (all11sampled
rotating-Couette slots are unsupported). Exact metrics, source roots, settings,
dates, old endpoint exclusions and incomplete-pressure qualifications remain
attached. No physics reruns, source dataset changes or default promotions.

The combined ranking retains Sun's TGV/shear lead and Delta's dam-break
pressure lead. Prepared legacy divergenceFree leads both sloshing pressure
metrics; native DFSPH leads sampled hydrostatics, IISPH analytic hydrostatics
and the sampled wedge; Delta narrowly leads the analytic wedge and leads
Couette. Different IC/kernel/viscosity/source presets prevent a solver-only
interpretation. Case curves include partial trajectories with explicit labels.
Output: `.tmp/fluid-matrix-2026-10-26-prepared-comparison/` with HTML, ranking
PNG/PDF/SVG, CSV/JSON, source hashes and reusable replay instructions.

### Prepared shear/sloshing startup defect — 2026-10-26

User flags volcanic sloshing particle ejection and shear startup shocks.
Saved-state forensics identifies a concrete shared setup problem: all23recorded
prepared shear/sloshing attempts use Monaghan–Gingold viscosity; the prepared
closed-case adapter skips the original shear configuration that selects and
calibrates Morris. GPUFP32 frozen fluid-fluid viscosity diagnostics on the
saved IISPH initial states give explicit-mode amplification1.4827at0.6221ms
(shear) and2.4524at0.8583ms(analytic slosh), with approximate stability ceilings
0.5011/0.4972ms. Thus the declared projection viscosity cap is insufficient
for the actual operator, despite the earlier compliance audit passing.
All8shear methods gain10%energy within15.6–19.3ms; IISPH peaks928xinitial
energy at46.9ms while the desired mode remains0.985. Slosh has1.97m/s particles
by55ms before a substantial bulk wave. These shear/slosh solver comparisons
are unqualified; earlier apparent slosh pressure leaders are not acceptable
physical results. The normal report is annotated, metrics/raw evidence retained.
No numerical/default changes or flow reruns. Next: short stage-work controls
on exactly the saved fixtures to verify viscosity selection/calibration and
an operator-appropriate step bound before considering a matrix rerun. The
frozen substep proves a setup defect but does not isolate every nonlinear
wall/projection artifact. Local evidence:
`.tmp/prepared-matrix-startup-audit-2026-10-26/README.md`.

### Startup stability sample controls — 2026-10-26

User requests sample replays of both affected cases. Seven sequential full-resolution
GPUFP32 controls are declared in `.tmp/prepared-matrix-stability-samples-2026-10-26/`:
IISPH and classic DFSPH shear to0.2s and both analytic/particle sloshing to1s, preserving MG
viscosity but reducing dt/maxDt to0.2ms; then IISPH shear with the Morris selector
at the original timestep ceiling and unchanged nominalnu. This final selector-only
control is explicitly uncalibrated for C4. Exact saved physical fixtures are copied
without preparation reruns. Pressure policies/mass/wall fit/forcing remain fixed.
Read-only stage energy observations, Vispy video, particle trajectories and source
receipts accompany the controls. Primary matrix and production code remain fixed.

Completed all7startup controls (22322steps,391.87scombined run wall):
MG at0.2ms removes shear energy explosions (old classic525x/IISPH928x;
new1.0xpeak) and the early particle/analytic sloshing jets. MG shear final
energy76.016% versus analytic74.666%, effectiveν.017366; Morris at the
original timestep also stays stable,75.095%energy/effectiveν.018137. The
remaining viscosity calibration error is explicit. Slosh peak speeds over1s
fall8.954→.162m/sclassic /7.612→.156IISPHsampled,8.637→.373classic /
8.603→.361IISPHanalytic. Analytic maxima are first-step compression kicks,
which are larger at smallerdt, so IC pressure compatibility remains open.
All7initial physical field sets are bitwise equal to their historical source,
fixture hashes match, mass drift/caps/dt-cap violations are zero, actual
pressure/gravity policies match, and all videos fully decode. IISPH sampled
viscosity energy work is nonpositive; classic DFSPH bypassed the composition
observer, so its stage journals are explicitly missing. Per-run observer
copies/hash receipts and the fluid-mask diagnostic correction are retained.
No production code/default or full-matrix row changed. Results/frames/videos:
`.tmp/prepared-matrix-stability-samples-2026-10-26/README.md`.

### Native VD+PS TGV failure audit —2026-10-26

User flags the surprising native VD+PS TGV failure. Both primary and rearranged
runs used exactly1DFiteration on every step, no DF/CDcaps, with failure at.626s
and1.621s. The shift converges only in a global mean statistic; local correction
size and resampling validity need measurement. Two short GPUFP32/65536particle
stage replays are declared in `.tmp/vdps-tgv-stage-audit-2026-10-26/`: primary
fixture to.7s at native DF0.1%, and a diagnostic0.001%divergence tolerance with
all other policies/iteration budget/ICs unchanged. Read-only stage energy,
shift-size, true density-residual and resampling observations plus ordinary
video/trajectory output. No numerical/default changes or primary-rank edits.

Short TGV stage replay exactly reproduces the primary.625579s stop. The first
surge at.600s occurs inDF(maxspeed1.05→1.30m/s); resampling changes only.020m/s
and the largest shift is.059dx then. Tightening onlyDF0.1→0.001% reaches.7s
with maxspeed.982m/s and endpoint shifts.00045dx; no caps. Prospectively extend
the unchanged tight setting/primary fixture to5s, retaining both short controls.
At1ms the two density-per-step tolerances correspond to meanDFresidual bounds
1/s and.01/s. This is a diagnostic pressure-accuracy change, not a default.

Completed TGVstage audit: native replay is bitwise identical in all retained
time/energy/speed/density-mean samples and terminal physical fields/pressure
histories. Its divergence kick is the main energy source (+6.764solverenergy;
resample+1.250; nonpressure−1.486). Tight0.001%DF reaches5s,5000steps,
129.13swall/25.825msstep/22.555msengine, with0DF/CDcaps, mean8.816DFiters,
maxshift.00371dx. No explosive stop, but E/E0=.720219vs.818731analytic,
velocityL2=.112858,meanrho1.04347,3.40%coveragegaps>1.5dx(max10.48dx):
clear vortex-center holes and excess decay remain. Frozen initial viscosity
hasdtλ=.0972andamplification.9028, so the shear/slosh explicit-viscosity
instability is absent on the initial TGVfixture. All3ICfield/hash checks,
zero mass drift, timestep-cap compliance and fully decoded videos pass.
Original and diagnostic results remain separate; no production policy or
matrix replacement. Tolerance accuracy is role-dependent: VD+PSdepends on
DFfor physical momentum, whereas coupled DFSPHalso hasCDmomentum correction.
Evidence `.tmp/vdps-tgv-stage-audit-2026-10-26/README.md`.

### Corrected full-endpoint matrix rerun —2026-10-26

User requests rerunning affected fields and updating the normal matrix.
Declared24supported entries:8shear(2s),15slosh(7s;bothwallrepresentations),
1nativeVD+PSTGV(5s). Reuse4exact saved physical fixtures; no preparation, mass,
geometry or initial-velocity changes. Prepared profile revisionstability-v1
(version7) records explicit row overrides: C4kernel-moment-calibrated Morris
for shear at targetν=.0185; sloshdt/maxDt=.2ms retainingMGν=.01; native
VD+PSTGVDFtolerance=.001%. Other pressure policies/iteration budgets fixed.
Existingversion6rows without overrides retain their prior behavior. Production
solver/timestep defaults unchanged. All24run sequentially GPUFP32 with video,
progress and particle exports; retain failures without tuning. Raw newroot
`.tmp/incompressible-corrected-rows-2026-10-26/`; afterward overlay these24
whole rows on the120slot primary prepared set, retain unaffected rows and
historicDelta/Sun/ACcontrols, and generate a new normal comparison version.


Stopped / rejected closeout — 26 October 2026:

**Corrected matrix rerun stopped at user request (2026-10-26):** 18 endpoints, one numerical stop, native VD+PS analytic sloshing interrupted at 3.8328 s, and four remaining analytic rows cancelled before starting. The stabilized sloshing is physically underactive: suppressed wave dynamics and speeds below expectation accompany pressure errors of roughly 50–79%. These results are rejected as a qualified comparison. Scheduler, simulation and automatic finalizer are stopped; GPU has no active compute processes. Completed and partial evidence retained in .tmp/incompressible-corrected-rows-2026-10-26. No corrected normal matrix published.

Removing the explicit-viscosity instability establishes numerical stability only. The near-planar surfaces and low speeds in the coupled particle-wall runs are not acceptable sloshing dynamics. Pressure scores and endpoint completion must not be presented as physical success. The user has cancelled further runs; diagnose the suppressed dynamic response before proposing another sweep. Native VD+PS also costs about 47 density-shift iterations per step (72.52 ms/step on sampled sloshing), with every-iteration host convergence synchronization and extra geometry/resampling work; no performance changes were made during this queue.

Four physical fixtures and all completed/partial recordings are preserved. Only the earlier 11-run interim evidence audit has completed; no 24-run audit or new normal-matrix publication is claimed. The full queue was not completed. Stop receipt: `.tmp/incompressible-corrected-rows-2026-10-26/stop_receipt.json`.


Saved dam-break energy/configuration audit — 26 October 2026:

User asks whether excessive incompressible dissipation is also visible in dam breaks. Saved primary prepared and retained WCSPH records were analyzed only; no runs resumed. Fluid K/M at 2.45 s is 0.5601 Delta / 0.6733 Sun versus 0.03508 native DFSPH and 0.03369 IISPH with sampled walls. Peak native DFSPH K/M is 0.6353 versus Delta 1.4560 (43.6%); integrated K is 36.5% of Delta. Analytic native DFSPH peaks at 1.3585 versus Delta 1.4397 (94.4%), ends at 0.1965 versus 0.5645 (34.8%), integrated K85.1%. Native VD+PS ends at0.3283 sampled/0.7168 analytic, demonstrating composition dependence. Mass normalization removes the0.04% initial mass difference. Saved-position/velocity frames also show greater K+gravitational-PE loss; compressible internal/elastic energy is not reconstructed.

Two concrete configuration/implementation confounds are present in the recorded inputs and archived source: the prepared adapter overrides inviscid dam-break parameters with physical nu0.01 (nominal Re~146), whereas WCSPH uses artificial alpha0.01. These are different branches and coefficients. Sampled prepared records contain5184ghosts: the viscosity boundary helper bypasses reflection when ghosts exist, assuming mDBC velocities have already been filled. WCSPH and legacy divergence-free explicitly compute those reflected velocities; native DFSPH/IISPH/composition do not. Thus stationary sampled wall velocities enter native viscosity, adding tangential drag; analytic walls use a free-slip mirror. Sampled body metadata is zeros versus analytic freeSlip. This is not a valid isolated measurement of intrinsic projection dissipation. Force/work attribution and corrected physical comparisons remain unmeasured; no fix or rerun is claimed.

Evidence: `.tmp/dambreak-energy-audit-2026-10-26/README.md`, `energy.png`, `metrics.csv`, `boundary_path_evidence.json` and source hashes. Existing matrices and stopped runs remain unchanged.


Consistency repair / single-case gate — 26 October 2026:

User authorizes fixing case-viscosity overrides and missing reflected ghost velocities in native viscosity. Declare one full2.45s, nx100, particle-wall native DFSPH dam-break replay on the exact saved primary fixture, GPU FP32, unchanged C4kernel/timestep/pressure policies. Match the case inviscid/alpha settings and supply viscosity-only reflected wall references. Compare energy, pressure, geometry and cost with the failed physical preset and retained WCSPH; audit inputs/runtime settings/video. No matrix continuation before this result is analyzed.


Consistency repair / representative outcome — 26 October 2026:

Implemented a viscosity-only mDBC reference refresh for native IISPH/DFSPH compositions and unprepared Omni/Band force routes; existing prepared WCSPH/reference paths are unchanged. The returned boundary reference does not overwrite pressure/rigid-body velocity state. Free-surface prepared comparison revision8/consistency-v1 preserves case viscosity; dam-break alpha0.01*c0=47.3091059 matches WCSPH, while sloshing preserves its case nu1e-6 instead of the blanket nu0.01. Original/stability viscosity revisions remain selectable. Native sampled contour metadata now follows wallBC. No pressure formula, tolerance, iteration budget or physical timestep bound was altered.

127 focused unit checks pass with12 CUDA-only skips (CPU tests include real ghost interpolation and a uniform tangential-flow/zero-drag regression). An initial sandbox-cache write limitation was resolved by using a writable test cache; it was not a numerical failure. The sole GPUFP32 replay uses exactly the saved7200fluid/17568total particles, unchanged positions/velocities/masses/supports/kinds/densities/pressures; no IC preparation or calibration rerun.

The repaired run restores wave dynamics and early energy: K/M at0.8s1.44209 versus Delta1.44842/Sun1.45621 and previousDFSPH0.63526. Prefix peak before0.8s1.44298 versus Delta1.45596 (99.1%, previously43.6%). Geometry and wall-impact front agree much more closely with WCSPH. However it stops at0.818580s/1766steps on a135.864m/s velocity alarm. Wall52.286s/29.607msstep; integrator21.676ms. Costs cover different durations from the old full endpoint and are not a speedup benchmark.

The final DFpressuremean jumps0.029→13890 whileCDmean stays1.17; the terminal step returns without a cap hit (1DF/2CDiterations). Across the recorded prefix there are0DF/3CD cap hits. Fluid UID478 has divergence pressure100,007,768 and sampled density0.179; reconstructed pre-drift geometry has3total neighbours including self, only1fluid (itself). Start geometry is reconstructed as x_final-dt*v_final and has float32 roundoff; no exact pre-step tensor was recorded. This is consistent with the documented deficient-row policy masking initial history/error but retaining raw pressure updates. The maximum-speed victim is UID119, not the high-pressure row itself. This local pressure robustness defect needs diagnosis before a new sweep. No tolerance adjustment, retry or matrix continuation was performed.

Evidence audit1/1 passes: all7initial fields bitwise original, no mass drift or timestep-cap violations, runtime viscosity/c0 and source archives verified, full video decode. Completed-prefix KE is not a full-run pressure/accuracy pass. Saved raw failure, diagnostic curves, pressure probes, particle snapshots and local details: `.tmp/dfsph-dambreak-consistency-2026-10-26/README.md`. Broader matrix remains stopped.


Deficient-neighbourhood robustness — 26 October 2026 (user follow-up):

Track the single-particle divergence-pressure explosion as an open robustness
blocker, not an acceptable free-surface artifact. The user identifies diagonal
clamping as a possible workaround; no regularization is selected or implemented.
Next diagnostic: record exact stage neighbourhood/diagonal/source/update/kick
at the saved failure onset, then evaluate a dimensionally consistent diagonal
magnitude floor / inverse-factor bound and consistent deficient-row policies
one choice at a time. Acceptance must cover sparse-particle kicks, bulk DF
response and hydrostatic support; passing a global mean error is insufficient.
Details in `docs/incompressible/DFSPH.md`, “Open robustness defect”. The failed
replay remains unchanged, and the wider matrix stays stopped. No new runs.


Deficient-neighbourhood investigation started — 26 October 2026 (user): observer-only GPU FP32 replay of the exact saved consistency case, with stage input/iterate/output capture after0.815s; no pressure or timestep changes. Use frozen failing-stage experiments to distinguish weak-diagonal amplification, adjacency/operator discrepancies and deficient-row stopping behavior before choosing any regularization. Preserve all failed evidence and keep the wider matrix stopped. Evidence root `.tmp/dfsph-deficiency-investigation-2026-10-26`.


Deficient-neighbourhood investigation outcome — 26 October 2026:

Observer replay reproduces all shared initial/final tensors and9physical/solver
histories bitwise, failing at the same0.81858002s/1766steps. Exact stage shows
UID478 first entering two wall samples with no other fluid neighbour; diagonal
-2.564641e-11, source-3.282894e-6/s, omega.25 yields pressure1.000078e8.
Basis-column diagonal agrees bitwise; rebuilding adjacency changes acceleration
only3.8e-6m/s², ruling out stale-list amplification at this event. Its pressure
travels through contact-Shepard interpolation (contact pressure12524.997Pa) to
UID119, entirely accounting for136.791m/s DFkick; no direct fluid pair connects
them. UID119 is also fluid-isolated but31wall-inclusive neighbours bypass the
deficiency guard. Input/compression-only stopping misses an expanding victim;
100forced sweeps lower the statistic while growing the kick to547.167m/s.

Published Eq.11 follow-up clamps mass denominator below1e-6; pinned reference
scalar/AVX instead disables volume factors<=1e-5. Native exact-zero-only policy
omits protection. Frozen translated floor/cutoff reduce remote kick to.0001124/0
m/s, with.119771m/s maximum remaining elsewhere. No numerical policy was changed
or full guarded trajectory qualified. Next: unit/resolution-aware weak-factor
policy, sparse wall-contact regression, bulk/hydrostatic controls and short
same-IC continuation before any wider matrix. Report and exact evidence in
`.tmp/dfsph-deficiency-investigation-2026-10-26/README.md`.

First observer attempt reached.81506026s then hit snapshot serialization on a
geometry callback; retained as diagnostic failure. Successful lean snapshot
replay: GPUFP32,77.199s/43.714msstep,31.064msintegrator, versus uninstrumented
52.286s/29.607msstep. Separate existing meshless GPU work and observer I/O
confound timing; no speed ranking. Saved-stage parity/operator/counterfactual
assertions pass. Matrix stays stopped.

Extended frozen check:100sweeps with the floor accumulate UID478 pressure8215.619Pa and remote kick.01124m/s; the cutoff keeps both zero. Both have maxkick.39601m/s elsewhere. A denominator floor changes the Jacobi preconditioner, not the exact pressure system, so iteration-budget sensitivity remains material. Successful replay input/runtime/source/video integrity audit1/1 passes; no mass drift or timestep-limit violations.


Weak-factor cutoff implementation / replay — 26 October 2026 (user): implement the pinned reference rest-volume denominator cutoff1e-5 in both native DFSPH projections, with zero pressure/history contribution on disabled rows. Apply it before dt and density-ratio scaling, using the base factor rather than contact interpolation terms. Keep case initialization, viscosity, pressure tolerances and iteration budgets fixed; add sparse contact/unit/timestep regression coverage, then rerun the exact saved particle-wall dam break from time zero to2.45s with video/progress and analyze before a wider sweep. Evidence root `.tmp/dfsph-cutoff-dambreak-2026-10-26`.


Weak-factor cutoff implementation / completed replay — 26 October 2026:

Native DFSPHConfig.weakFactorThreshold defaults1e-5 (inverse simulation length
squared);0 restores exact-zero-only behavior. Both projections test the base
rest-volume factor before density-ratio/dt scaling and exclude contact diagonal
contributions from eligibility. Disabled seeds/updates/output pressure are zero;
active exact operators, error statistic, tolerances and iterations unchanged.
Contact paths add a base-diagonal evaluation without new host synchronization.
142focused checks pass,12CUDA-only skips: sparse two-fluid/wall contact kick,
1/100sweeps, rho0/dt and density-ratio scaling, both-stage warm history and config
round-trip/validation; existing dense/bulk/viscosity/matrix checks still pass.

Same7ICfields bitwise, GPUFP32 full2.45s/5628steps completed, no velocity alarm.
Nine physical/solver histories remain bitwise identical through.81826003s;
nextstepvmax6.25026m/s versus135.86386 beforecutoff. Peak K/M1.44298 versus
Delta1.45596, integratedK99.14%Delta, endpoint.64534 versusDelta.56005/Sun.67328.
Overallvmax7.097m/s,0DF/3CDbudgets (allCD insharedstartup), no mass drift or
physical-step violations. Source/input/runtime/video audit1/1 passes; numerical
launch differences only solver/config, plus postprocessing metadata forwarding.

Pressure error42.245% remains worse thanDelta28.209/Sun32.705; P1 plateau low
and P2 lasts too long. At most4particles penetrate simultaneously/max.81052dx, noneat
endpoint; fragmentation/cavity evolution differ. Successful single-case explosion
regression, not a pressure/wall-accuracy qualification. No other matrix rows run.
Wall134.359s/23.873msstep/integrator18.376ms; separate existing GPU work shared
device, and endpoint/prefix workloads differ, so no speedup claim. Evidence
`.tmp/dfsph-cutoff-dambreak-2026-10-26/README.md`, video and archives retained.


Persistent surface-layer / re-merging investigation — 26 October 2026 (user): detached particles stack above and ride the bulk even after volcano/explosion repairs. Start with saved cutoff dam-break frames and archived sloshing evidence: quantify spacing/connectivity, density, physical DF/CD pressure and relative velocity/persistence. Distinguish ejection suppression from remeshing/re-merging and the weak-factor cutoff from existing unilateral/free-surface operators. No new simulation or numerical model change declared yet. Evidence `.tmp/dfsph-surface-layer-2026-10-26`.

User clarification: the hovering cohort is near self-density0.179rho0; sampling recovery necessarily increases summation density and may be suppressed by the DF rate constraint. Investigate this as sampling restoration at a free surface, distinct from physical compression of a fully sampled fluid.


Persistent surface-layer / saved-data outcome — 26 October 2026:

C4self-density is0.178974935rho0. UID1438 stays0.178978–0.179076rho0 through
1.918–2.45s while moving.740m horizontally/.037m downward; endpoint CDpressure
and CDacceleration0, reconstructed DFupward acceleration47.315m/s² (self47.225,
neighbour.090) versusgravity-9.81. Basefactor.007871 is787x cutoff; nearest
fluid3.385dx/support4dx, totalcount10 but nonselfdensityonly.0001015rho0.
Several other returning particles show the same positive DFsupport with CD0.
Sparse support graph connects1799→1199→1438→6244(bulk), all links3.6–3.9dx;
first3nodes remain almostself-density. This is a pressure-carrying sparse layer,
not proof of missing physical mass or a weak-factor-disabled free fall.

For51selectedlowdensityparticles, allownCDpressure0/20ownDFpositive at endpoint;
11haveDFay>g/2 and none CDay>g/2. Cohort also includes ballistic spray; do not
claim every lowdensityparticle is supported at every instant. At1438 removing
DFkick fromfinalvelocity raisesC(v).003728→.006084/s on reconstructed geometry,
confirming suppression of sampling-density growth. Forces use exactsavedp/rho,
pre-driftx reconstruction(FP32roundoff), localG=0; no exact pre-DFsource claimed.
Saved trajectory/forces only, CPUFP32postprocessing, no simulation or model edits.

Interpretation: returning-particle sampling recovery increases summationrho;
rawDF constraint treats that as material compression and suppresses entry at
kernel-fringe separation. Deficiency seed/error masks do not pin new pressure,
and count7 is not weighted completeness forC4/4dx. Keep cutoff explosion fix;
next separate quiet-surface re-entry/source-active-set experiment must permit
sampling recovery while preserving bulk DFresponse, with no ad hoc mass/density
reset. Sloshing analogy is user-observed, not independently force-attributed.
Evidence `.tmp/dfsph-surface-layer-2026-10-26/README.md`. Matrix stays stopped.


Native dam-break re-entry comparison declared — 26 October 2026 (user): run pinned SPlisHSPlasH native DFSPH from the cutoff replay’s exact7200fluid positions/zero velocities/masses and5184kind-1wall samples/volumes, rho0=1, gravity9.81, to2.45s. Primary cubic2dx plus declared cubic4dx support control (native C4 unavailable). Match CD.01%/DF.1%, caps100, native relaxation.5/carrier/deficiency behavior retained. External per-step CFL.2dx/vmax with1msceiling and native adaptiveCFL off avoids mixed-dt source contract. Native standard viscosity coefficient matches the warp artificial projected-term coefficient; two-sided action, density weighting/regularizer and kernel differ, wallviscosity0. No extra mass/density fitting or source patch. Record pressure, sparse trajectories/UIDs, energy, low-density residence and re-entry gaps with video/progress/cost/source archive. Evidence `.tmp/dfsph-native-dambreak-2026-10-26`. No broader matrix.


Native dam-break re-entry comparison completed — 26 October 2026:

PinnedSPlisHSPlasH cubic2dx completes2.45s/4772steps, maxspeed6.503m/s,
0DF/79CDcap hits, at most2simultaneous plane penetrations/.0377dx. Same7200
fluidx/v/m and5184wallx/volumes checked bitwise, nativeeffectiveV matchesm/rho0
within7.3e-8relative. Initialrawcubicrho median1.000447 retained; independent
first-input density agrees within2.4e-7absolute. CPUFP32AVXOMP4 wall24.762s/
5.189msstep, engine20.397s/4.274msstep, video1.021s, init.160s. No hardware-
normalized rank versuswarpGPU134.359s/23.873msstep. Native standardnu.00108938
matches projectedcoefficient only; two-sided/weighting/regularizer and Akinci
boundary differ, explicitly recorded. Both videos/source/input audits pass.

The same re-merging failure is observed: nativeUID719 stays.4544847–.4560856
rho0 through1.915786–2.45s (cubicself=.454484687), moving.47837m horizontally
and.02269m downward. Endpointoneotherfluidneighbour, DFpressure511.026/CD0;
reconstructedDFay314.085m/s², CD0, gravity-9.81. Its kerneldenominator.0404571
agrees with storednativefactor after dt²conversion to5.9ppm. Neighbour guard
therefore does not prevent pressure on the deficient returning particle, as
source audit predicted. Saved field/trajectory attribution, not a changed model.

Common observable selection (recovery<.02 for80%late samples, near rho>=.85bulk
within1.1support/interior50%late samples) finds5native/7warp persistentparticles;
median endpointbulkparticle gap1.908dx/3.892dx, about.954/.973ofsupport. Both
therefore hover near their kernel fringe; support, kernel, relaxation, boundary
and viscosity differences prevent a solver-only severity attribution. Native
K/Mpeak1.40388/end.46441/integral2.01142 versuswarp1.44298/.64534/2.20177.
No pressure-noise/error rank from sparse nativeframes. Pressureplots use the
same MLSdisc recipe on saved nativefields; rawsimpledisc estimates excluded.

Declaredcubic4dx control failsat.007s/7steps inCD(cap100), speed3.43e18m/s;
no relaxation tuning/retry or latecomparison. Wall.0579s/8.270msstep,
engine7.081msstep; failure/video retained separately. Native projection source
unchanged, externalconsistentdt CFLadapterdeclared. No new matrix/production
changes. Observer-onlycorrections (rawtStar sqrt2factor, unmatchedsimpledisc
means, finalpressureIDordering) preserveoriginalexports; correctedderiveddata
and fixed-IDsnapshots verified. Evidence `.tmp/dfsph-native-dambreak-2026-10-26/README.md`.


Single-particle hydrostatic re-entry trials declared — 26 October 2026 (user):

Reuse the calibrated465-fluid C4/4dx hydrostatic tank and fitted analytic walls,
GPUFP32, rho0=1000, dx=1/32, nu=.01, gravity9.81, classic DF→NP→CD→drift,
DF.1%/CD.01%, relaxation.25, caps100. Damped preparation (gamma20/s) then
undamped hold/drop from one common saved state. Add one equal-mass particle
above the centre; preserve all bulk masses/positions/velocities/pressure history.
Do not recalibrate mass or reset density to the target. Video/progress/watchdogs,
explicit exportRoot and source archives for every run; sequential execution.

Published DFSPH Eq9–11 removes positive density rate irrespective of sampling
headroom; pinned native deficiency masks only initialization/statistics. Trial
local variants separately: (1) pressure unknown pinned to zero when rho/rho0<.9
(weighted-completeness active set, pair-force operator unchanged); (2) allow
C(v_out)<=(1-rho/rho0)_+/tau in DF, tau=dt (one-step recovery) or .02s (finite
recovery horizon). Density constraint, exact diagonal, weak-factor cutoff, warm
start, force symmetry and all bulk physical settings retained. Gates are
experimental, not a published-method attribution. Baseline first; if it does
not reproduce hovering, report that and adapt diagnostic geometry explicitly.
Compare drop rho/nearest-bulk gap/relative motion, DF/CD impulses and work,
bulk RMS velocity/hydrostatic pressure error/overdensity, mass and cap hits;
check a promising choice at half dt before any recommendation. No production
default promotion or broad matrix launch. Evidence .tmp/dfsph-reentry-2026-10-26.

Re-entry preparation/control: gamma20/s5s preparation reaches RMS3.02mm/s,
max13.20mm/s, so the strict1/5mm/s rest criterion was NOT met. Retain as
quasi-relaxed diagnostic IC, not exact hydrostatic equilibrium or isotropic glass.
Two top side-wall samples lie0.043dx beyond fitted planes; bottom is inside.
Undamped1s no-drop hold ends RMS6.59mm/s/max21.76mm/s, peak49.52mm/s,
no solve caps and no final crossings. This background is measured explicitly
against the expected1.57m/s drop scale; do not attribute its pressure noise or
surface pairing to the added particle/remedy. Preparation107.74s/17.69msstep,
hold20.63s/15.20msstep. Private driver signature failure advanced zero steps,
retained under setup_failure_relax. Seven private algebra/bulk gates pass.


Single-particle re-entry remedies completed — 26 October 2026:

Production DFSPH delays re-entry (2s rho=.276rho0/gap2.10dx; late DF/CDay
7.56/2.16m/s²), while active09 and headroom/dt,20ms reach sustained.85rho0
at.294/.287/.286s and rho≈1/gap≈.71dx by2s. Headroom avoids the hard.9
threshold plateau; finite20ms at half dt merges.2835s/gap.707dx. All5drop
inputs match bitwise, fixed mass/positive density/nonnegative pressure, no caps;
7private algebra/bulk tests and all9completed video/source/timestep audits pass.
No production solver/default changed; trials private in .tmp/dfsph-reentry-2026-10-26.

Counterweight: matched no-drop2s controls show RMS6.15→6.54mm/s (+6.4%),
pressureL2error10.70→12.55% and pressure noise105→195Pa (+85.8%), meanpressure
+1.22Pa. Drop pressure chatter283→4.28Pa at half dt remains unqualified.
At most2corner crossings; headroom impact sampledmaxpenetration.262dx versus
.043dx baseline. Initial quasi-rest/surface pairing explicitly retained.
Primary2s drop wall42.3–43.6s/engine16.7–17.3ms; half-step76.3s/15.2ms;
no-drop controls47.2/45.5s,17.7/17.4ms. No remedy cost regression observed.

Keep density-headroom as preferred experimental re-entry hypothesis, hardgate
as mechanism control; do not promote defaults or restart matrix. Pressure/
wall/resolution/particle-boundary and representative impact qualification open.
Fullreport `.tmp/dfsph-reentry-2026-10-26/README.md`, durable interpretation
`docs/incompressible/DFSPH.md`.


Dam-break re-entry remedies declared — 26 October 2026 (user):

Run active09 (pin DF unknown for rho/rho0<.9), headroom/dt and headroom/20ms
using the private source/active-set rules from the completed single-particle
control. Reuse the exact saved cutoff7200-fluid sampled-wall Marrone fixture,
C4/4dx, alpha.01/c0=47.3091, rho0=1, classic ordering, DF.1%/CD.01%,
omega.25/cap100, cutoff1e-5, adaptive CFL.2 and1msceiling, target2.45s.
No mass fitting/IC preparation rerun, damping, density reset, force masking or
production-default change. Sequential GPUFP32, video/progress/watchdogs and
explicit exports, per-run source archive and unchanged physical-input check.
Observe UID-ordered x/v/rho/DFp/CDp at10ms plus exact endpoint, existing
full-rate scalar pressure/energy/cap/wall diagnostics, existing matched probe
recipe and physical timestep audit. Compare same sparse-rider criterion as
reference report (normalized sampling recovery<.02 for80%lateframes, near
rho>=.85bulk/interior for50%lateframes), overall sampling distribution, KE/M,
pressure error, positivity/mass/wall containment and wall/step cost. Preserve
failures without tuning/retry; no broader matrix restart. Output
.tmp/dfsph-remedy-dambreak-2026-10-26.

Dam-break remedy interim: active09 and headroom/dt complete2.45s, both0
persistent near-self-density riders versus7baseline. Endpoint rho<.3 counts
2/20 versus82; pressure errors40.70/40.78% versus42.25%. IntegratedK/M
2.188/2.164 versus2.202baseline, no gross dynamics suppression. Both retain
the3startup CDcaps at.004/.005/.006s,0DFcaps. Full evidence audits1/1 each
and private source/dispatch/pressure/endpoint checks pass. Pairing rises
5.08→7.89/8.39%, voidfraction falls2.03→.319/.417%; maxwallpenetration
.832/.847dx versus.811dx. Wall456.9/404.7s,81.7/74.3msstep, engine46.9/42.5ms;
contention prevents isolated timing rank. Third finite20ms run in progress.

Dam-break remedy trials completed — 26 October 2026:

All3 reach2.45s: persistentnearselfcohort7→0each; endpoint rho<.3 counts
82→2active09/20headroom-dt/7headroom-20ms. Broader persistent rho<.5
cohort43→0/0/2, so finite20ms leaves a moderately deficient remainder.
Pressureerrors42.25→40.70/40.78/39.72%; integratedK/M2.202→2.188/2.164/2.177,
endpointK/M.645→.591/.611/.631. Three common startup CDcaps only,0DFcaps,
fixedmass,positive sampled rho/nonnegativep;7gates and3/3fullsource/input/
video/timestep/dispatch audits pass. Production code hashes unchanged.
Pairing5.08→7.89/8.39/8.82%, maxwall excursion.832/.847/.823dx; physical
pressure/wall/isotropy qualification remains open. Endpoint nPenetrating0
counts only excursions>.5dx and is not perfect containment.
Wall456.9/404.7/243.3s (81.7/74.3/42.3msstep), engine46.9/42.5/27.9ms;
sharedGPUloadpreventsisolatedcostranking, directcopyoverhead1.67/1.18/.46s.
Keepall3experimental, no default promotion/matrix restart. Full durable
interpretation docs/incompressible/DFSPH.md; rawreport/figures/videos/receipts
.tmp/dfsph-remedy-dambreak-2026-10-26/README.md.


Four-way sloshing re-entry comparison declared — 26 October 2026 (user):

Run cutoffbaseline/active09/headroom-dt/headroom20ms sequential GPUFP32 through
full7s recorded sloshing excitation, using the saved4200-fluid sampled-wall
fixture, C4/4dx, calibrated masses, hydrostatic seed, classic ordering,
omega.25/DF.1%/CD.01%/cap100/cutoff1e-5. Corrected consistency-v1 viscosity
is the case physicalnu1e-6 (not old blanket.01), with current viscosity-only
reflected wall references. Retain the existing.2msceiling/CFL.2, no outcome
tuning, additional IC preparation, mass fitting, shifts or damping. Baseline
uses unchanged production projection; each remedy changes only its declared DF
constraint. Initial fields/first-step hydrostatic impulse are checked against
shared fixture. Keep videos/progress/alarms/stallguards, source/private-formula
archives, physical timestep audits and UID pressure/rho/x/v snapshots at10ms.
Compare forcing/wave/kinetic motion, sensor pressure, late lowdensity cohorts,
pairing/voids, wall containment, positivity/mass/caps and wall/step cost; retain
failures. This is the requested sloshing comparison, no unrelated matrix rows
or production-default promotion. Output .tmp/dfsph-remedy-sloshing-2026-10-26.

Sloshing baseline interim (fresh repaired physicalnu1e-6): full7s/35000steps
completed,0DF/0CDcaps, fixedmass, fullinput/source/video/seed/timestep audit1/1.
Peakv3.757m/s, peakK/M.30664, integratedK/M.73705; COMx peak-to-peak.3025m
confirms substantial wave dynamics. Full matched10ms sensortraceNRMSE96.03%;
impactpeaks2.868/2.284/2.405kPa versusmeasured1.800/2.322/1.171kPa, with
timingoffsets. Pressure is not qualified by plausible peak amplitude.
Persistentnearselfcohort8, broader rho<.3/.5cohorts39/62; endpoint199particles
under.3rho0. Pairing6.75%, voids5.37%, maxwallexcursion.776dx/4simultaneous
>.5dxcrossings, noendpointcrossings. Wall832.93s/23.80msstep, engine16.92ms;
snapshotcopies.55s. Active09 now running; remaining2headroom rows queued.

Sloshing active09 interim: full7s/35053steps,0caps, fixedmass and fullaudit1/1;
actual source pinning and endpoint pressure/UID parity verified. Persistent
nearself/.3/.5 cohorts8/39/62→0/0/0; endpoint rho<.3 count199→56. Integrated
K/M.72767 (-1.27%baseline), peakK/M.30879, peakv5.337m/s; no gross dynamics
suppression. Matched sensortraceerror96.03→71.50%; impactpeaks2.060/2.160/
2.557kPa, timings closer to measured first3impacts but thirdovershoot remains.
Pairing6.75→7.38%, voids5.37→2.17%, maxwall excursion.848dx/7simultaneous
>.5dx crossings versus.776/4baseline. Wall650.56s/18.56msstep, engine14.48ms.
Remainingheadroom/dt and20ms controls are in progress/queued, no default change.


Four-way sloshing re-entry comparison — 26 October 2026

All four variants complete the same saved 4,200-fluid particle-wall tank to
7 s on GPU FP32. Calibrated masses, hydrostatic pressure seed, C4/4dx, classic
DF → nonpressure → CD → drift ordering, omega=0.25, DF=0.1%/CD=0.01%,
cap=100 and weak-factor cutoff=1e-5 are shared. The repaired consistency-v1
case uses physical nu=1e-6 m²/s, reflected free-slip viscosity references,
and a 0.2 ms ceiling with CFL=0.2. No new IC preparation or additional damping.
These are local experimental DF constraints; CD and production defaults are unchanged.

| DF rule | Persistent near-self / rho<.3 / rho<.5 | Final rho<.3 count | Integrated K/M, relative baseline | Pressure trace error [%] | Pairing final [%] | Wall [s] / ms step |
|---|---|---:|---:|---:|---:|---|
| Cutoff baseline | 8/39/62 | 199 | 100.00% | 96.03 | 6.75 | 832.93 / 23.80 |
| DF off below 0.9 rho0 | 0/0/0 | 56 | 98.73% | 71.50 | 7.38 | 650.56 / 18.56 |
| Headroom / dt | 0/0/0 | 35 | 95.46% | 83.07 | 10.81 | 623.41 / 17.77 |
| Headroom / 20 ms | 0/0/0 | 67 | 98.27% | 66.93 | 8.83 | 604.46 / 17.27 |

The persistent cohort is observational: after 6.2 s, low density for at least
80% of saved samples and near dense interior fluid for at least 50%. Near-self
means normalized recovery (rho-rho_self)/(rho0-rho_self)<0.02; rho_self≈0.179rho0.
Global endpoint sparse counts also include spray. All three remedies eliminate
the measured persistent layer, including the broader rho<0.5 cohort. Common
geometry/color snapshots show the baseline's separate low-density surface row
and its removal by each remedy. This does not prove that every airborne
particle has merged.

Driven wave motion remains substantial: integrated K/M is 98.73%, 95.46% and
98.27% of baseline. COM x ranges are 0.3025/0.2864/0.2708/0.2803 m in table
order. Higher energy alone
is not an accuracy criterion for this externally driven problem.

Pressure error is the standard full-record relative L2/NRMSE against measured
Sensor1, with matched 10 ms smoothing and no phase alignment; it includes
pulse timing and shape. The 20 ms rule scores best overall, but its third
smoothed peak is 3.045 kPa versus 1.171 kPa measured, worse than baseline
2.405 kPa or hard-gate 2.557 kPa. Its first two peaks are 1.966/2.080 kPa
versus measured 1.800/2.322 kPa, with timings closer to the reference. Thus
the lower full-trace score does not establish accurate impact amplitudes.

Pairing rises from 6.75% to 7.38/10.81/8.83%; void fraction falls from 5.37%
to 2.17/2.94/1.86%. Peak rho/rho0 is 1.0022/1.0088/1.0185/1.0066. Maximum
wall excursions are 0.776/0.848/0.891/0.827dx; simultaneous counts more than
0.5dx outside the tank AABB reach 4/7/8/11. Endpoint counts by that half-dx
criterion are zero, but strict crossing counts are 0/0/6/1. Wall contact
therefore remains unresolved. Peak speeds are 3.757/5.337/5.627/4.645 m/s.

All four retain fixed mass, positive finite sampled density and nonnegative
pressure unknowns, with zero DF/CD iteration caps and no timestep violations.
Four full evidence audits pass, as do seven private operator/constraint gates.
Initial physical fields match bitwise, pressure impulses match the shared
hydrostatic seed and actual timestep, stage call counts match steps, final
UID-sorted snapshots match final tensors, and every video fully decodes. All
1,021 current production source hashes match the archived baseline. Engine
costs are 16.92/14.48/14.11/13.92 ms per step; snapshot copies total
0.55/0.26/0.23/0.19 s (exact receipts retained). Shared GPU contention means
wall/step differences are not an isolated algorithm-cost ranking.

**Verdict:** keep the simple density gate and finite-horizon headroom as
experimental re-entry remedies. Both recover sampling with little change in
bulk kinetic motion. The gate has less pairing; 20 ms gives the lowest full
pressure-trace error but a larger third impact overshoot. One-step headroom
has the largest pairing, density peak and motion reduction here. Pressure,
wall contact and resolution/timestep sensitivity remain open; no production
default or normal-matrix promotion.

[Full report, common-scale figures, recorded videos and retained evidence](.tmp/dfsph-remedy-sloshing-2026-10-26/README.md).


### Candidate selection and proposed qualification — 26 October 2026

User selects **DF active only at rho>=0.9rho0** as the candidate to carry
forward. Sloshing first-impact amplitude, second-impact width/ringing, bulk
kinetic motion and sampling quality take precedence over the single aggregate
pressure score. The headroom experiments remain retained comparison evidence;
this selection does not promote a production default or restart the matrix.

Proposed evaluation sequence (not launched by this planning discussion):

1. Freeze the exact experimental gate semantics: pin the clamped DF pressure
   unknown, including warm start, below threshold; retain symmetric neighbour
   pressure forces, CD, weak-factor cutoff and signed periodic policy. Before
   replacing the private adapter, expose an opt-in configuration and establish
   numerical parity with saved trials plus checks for activation, warm starts,
   force symmetry and convergence statistics.
2. Add the missing active09 no-drop hydrostatic control and a half-timestep
   active09 drop control. Measure hydrostatic pressure slope/error/noise, RMS
   motion, merger density/spacing and the observed pause near 0.9rho0. Track
   activation transitions and pressure/velocity kicks around the threshold.
3. Run active09 sloshing at half the timestep from the identical saved IC;
   then one finer spatial resolution using the same physical state and
   preparation policy. Compare all three impact amplitudes, arrival times,
   widths/impulses and post-impact ringing separately; retain raw and matched
   smoothed pressures, wave displacement, K/M, pairing, wall excursions,
   pressure-stage work, solve failures and measured runtime. No time alignment
   or threshold fitting to the experimental trace.
4. Verify boundary transfer with analytic hydrostatic/drop and sloshing cases,
   using established density-compatible wall placement. Verify that the gate
   consumes the same boundary-inclusive density ratio as the DF operator.
5. Test bulk dynamics on prepared TGV and nominal-mass relaxed closed-box
   decay, with matched IISPH controls. Check velocity-field error where a
   reference exists, energy/enstrophy decay, divergence, density and symmetry.
   Record the inactive fraction: well-sampled bulk should not routinely rely
   on the surface gate. The signed periodic policy remains unchanged.

Proceed to focused matrix comparison only after these controls show stable
re-entry and hydrostatic support without materially worsening pressure, bulk
dynamics or wall behaviour. Threshold sensitivity (e.g.0.85/0.95) is a later
robustness check if activation diagnostics warrant it, not an immediate tuning
sweep. Costs remain part of every run; this is a proposed sequence, not new
qualification evidence.


Active09 qualification started — 26 October 2026 (user approves sequence):
Implement an opt-in clamped-DF density gate with default zero, preserve the
trial's pressure-unknown/warm-start/statistic semantics and symmetric forces.
Check native/trial parity and existing projection regressions before GPU runs.
First controls: same saved quasi-relaxed analytic tank, active09 no-drop hold
and half-timestep drop, plus matched native baseline/1ms replay for provenance.
Retain established IC/viscosity for this control, distinguish its nu=.01
fluid-only tank from the case-nu=1e-6 sloshing replay. Watch threshold switching
and near-threshold residence; no hysteresis or threshold fitting. If sound,
continue half-dt sloshing, then declared transfer/refinement tests.

Qualification controls interim: native baseline hold and active09 drop endpoint
fields match their archived trajectories bitwise; 119 projection regressions
pass, 12 optional tests skipped, 12 frozen-operator parity checks pass. No-drop
active09: RMS6.152 vs6.145mm/s, noise104.688 vs104.687Pa, L2error10.720 vs10.701%.
One gate activation in each drop; no chatter. Half-dt drop reaches rho=.986
at2s versus1.000 at1ms, and pressure noise268.4→3.35Pa, with physical initial
pressure held fixed. This demands matched half-dt no-drop controls before
attributing pressure change to impact/gating; these two controls precede the
planned half-dt sloshing replay. No threshold/hysteresis tuning or promotion.


### Initial active09 qualification completed — 26 October 2026

Six matched 2s tank controls and one full7s half-step sloshing replay complete
on GPUFP32. The selected gate is opt-in and remains off by default. All six
control source/input/seed/endpoint/video audits and the normal sloshing audit
pass. Seven trajectories have fixed mass, positive sampled density, nonnegative
pressure, zero DF/CD caps and no timestep violations. Native baseline hold and
active09 1ms drop reproduce archived endpoint fields bitwise; 119 regressions
and12 frozen-operator parity checks pass (12 optional tests skipped).

No-drop hydrostatics are essentially invariant to the gate at each timestep:
1ms RMS6.145/6.152mm/s and pressure noise104.687/104.688Pa for original/gated;
.5ms RMS6.116/6.114mm/s and noise1.011/.970Pa. The large pressure-noise change
with dt occurs with the gate off as well; it is not threshold chatter. Drop
activation happens once per trajectory. Sustained .85rho0 recovery is.294/.288s
and final nearest spacing.709/.714dx at1/.5ms. Complete recovery toward1rho0
is slower at half-step (.986rho0 at2s, versus1.000), so early merger alone does
not demonstrate convergence. Physical pressure seed is fixed across dt by
rescaling DF/CD impulses with dt/dt², not preserving the old impulse magnitude.

| Sloshing ceiling | Peak1 / Peak2 / Peak3 [kPa] | Full trace error [%] | Integrated K/M | Pairing final [%] | Wall [s] / ms step |
|---|---|---:|---:|---:|---|
| .2ms, selected archived trial | 2.060 / 2.160 / 2.557 | 71.50 | .727675 | 7.38 | 650.56 / 18.56 |
| .1ms, equivalent native setting | 1.535 / 2.428 / 2.687 | 74.20 | .713944 | 8.18 | 1328.30 / 18.98 |
| Measured, same10ms smoothing | 1.800 / 2.322 / 1.171 | — | — | — | — |

Bulk motion remains close: integratedK/M−1.89%, COMx range.2864→.2795m.
No persistent near-self/.3/.5 sparse cohort at either dt. Endpoint rho<.3
counts56→49 include spray. Peakdensity1.00880→1.00066; maxwallexcursion
.848→.783dx with7 simultaneous>.5dx crossings in each, noendpointcrossings.
Both remain unqualified for exact containment/isotropy. Sloshing post-second
impact residual RMS after a10ms trend changes sensor115.49→140.88Pa, mean
DF3.89→5.56Pa, meanCD35.09→27.22Pa: unlike the quiet tank, this does not
uniformly improve. Residuals include physical fast transients and probe effects.
Second-pulse width66→57ms versus42ms measured, but window impulse457→488Pa·s
versus310Pa·s; separate amplitude, width, phase and impulse conclusions.

| Control | Wall [s] / ms step | Engine ms step |
|---|---|---:|
| Original hold1ms | 74.63 / 37.32 | 26.93 |
| Gate hold1ms | 69.71 / 34.86 | 25.00 |
| Gate drop1ms | 70.28 / 35.14 | 25.58 |
| Gate drop.5ms | 171.49 / 42.87 | 31.42 |
| Original hold.5ms | 68.11 / 17.03 | 13.44 |
| Gate hold.5ms | 69.43 / 17.36 | 13.63 |

Shared-device load and differing diagnostic/video work prevent an isolated
performance ranking. Sloshing engine14.48→15.31ms/step,70k vs35,053steps;
direct field copies.228s. Full runtime and all retained videos are in report.

**Verdict:** the gate remains the selected experimental candidate and passes
the initial rest/re-entry robustness checks. Impact pressure has a material
timestep dependence; the remaining evaluation programme is not complete.
Establish temporal/solve-accuracy convergence before proceeding to the costly
spatial, analytic-sloshing and bulk transfer sweeps or promoting a default.
Native percentage tolerances imply dt-dependent rate/pressure residual bounds,
so this two-level test does not isolate integration error. A further timestep
level and a separately declared solve-accuracy control are the next numerical
gate; no threshold fitting/hysteresis change. Finer spatial IC must preserve
actual fluid mass/volume, avoiding silent water-depth changes from rounding.
The sloshing reference also retains its existing gravity-only rotating-frame
approximation; no forcing model is changed here.

[Full report, per-impact metrics, source/input audits and recordings](.tmp/dfsph-active09-qualification-2026-10-26/README.md).


Pressure convergence study started — 26 October 2026 (user continues):
Fixed .9 gate; first-impact trajectories through2.9s only. Native.2ms replay
with frozen stage probes (pure solves, original trajectory returned), third
time level.05ms with native bounds, and fixed-.2ms accuracy control with both
native percentage tolerances×.01/cap100/miniters unchanged. Keep source/IC/
physical pressure seed/viscosity fixed; allGPUFP32/video/progress/guards and
sequential. Frozen probes compare tightened stopping bounds and fixed iteration
budgets at declared quiet/pre-impact/impact/post-impact times; measure actual
post-compression/complementarity residuals, pressure and velocity-kick changes.
A pressure reference requires iteration-refinement evidence, not just a native
converged label. No threshold/default/physical-model changes or matrix restart.
Output .tmp/dfsph-pressure-convergence-2026-10-26.


### Pressure convergence controls — 26 October 2026

Completed the declared first-impact study through 2.9 s on the identical
4,200-fluid-particle GPU FP32 sloshing fixture, with particle walls, the 0.9ρ₀
gate, physical ν = 1e-6 and matched hydrostatic pressure initialization.
Three new runs: native 0.2 ms with 112 frozen stage/variant records, native
0.05 ms, and 0.2 ms with both tolerances divided by 100. The archived 0.1 ms
trajectory supplies the middle timestep level. Mass, IC, forcing, threshold
and production solver code are unchanged.

With matched 10 ms filtering, native first peaks are non-monotonic:
2.060 → 1.535 → 1.888 kPa, versus 1.800 kPa measured. Times of 2.369, 2.371
and 2.373 s remain early versus 2.390 s. Through 2.85 s, integrated K/M is
0.126170, 0.127894 and 0.128613; centre-of-mass excursion is 0.237553,
0.237111 and 0.237088 m. Bulk motion changes much less than peak pressure.
Native median density residual scales roughly with dt² while DF/CD mean
iterations remain 1/~2. No temporal order is inferred.

The tighter control raises the peak to 2.396 kPa. FWHM is 35 ms versus
40 ms native and 49 ms measured. Pressure error over 2.2–2.85 s grows from
45.99% to 50.37%; positive pressure integral changes from 407.72 to
408.77 Pa·s versus 363.97 measured. Integrated K increases 1.59%; peak K
changes only 0.057%. Both stage tolerances changed together, so their
individual contributions remain unresolved. CD averages 99.316 iterations,
with 11,035/14,521 steps capped (75.99%). Reported CD residual/bound has
median 1.121, 99th percentile 1.584 and maximum 1.837. DF averages 1.148
iterations with no caps. This is **not a converged reference at the requested
tolerance**; retain its finite-budget result without retuning.

On frozen native peak geometry, DF takes one iteration and CD two, satisfying
their native global compression bounds. At 512 iterations, maximum velocity
kick changes reach 0.319/0.295 m/s. Pressure differences between 128 and 512
iterations remain 30%/35% relative to 512, so 512 itself is an unqualified
pressure reference. Tightly projecting an accumulated density error in one
small step can create very large pressure; this is not evidence that such
pressure is physical. The advancing tighter run maintains normal mean
pressure. Observer purity and bitwise replay parity are verified.

Noise does not improve uniformly: native→tight post-impact raw sensor
residual is 91.55→91.77 Pa, and the common 1 ms-minus-10 ms band changes
64.10→48.99 Pa. Mean DF residual grows 2.15→10.79 Pa while mean CD falls
28.22→19.75 Pa. At native 0.05 ms, raw sensor residual is 122.28 Pa and the
common-band residual 65.91 Pa. These statistics contain physical transients
and probe effects. Maximum pairing remains 8–10%, with wall excursions of
0.63–0.83 dx. This short prefix does not requalify late rider cohorts.

All three terminal, source, initialization, physical-timestep, mass and video
audits pass. All 1,029 source hashes per run and endpoint observer tensors
match. The instrumented native replay matches the earlier trajectory bitwise
before 2.85 s. Costs with video:

| New run | Wall seconds | Wall ms/step | Engine ms/step |
|---|---:|---:|---:|
| Native 0.2 ms replay | 278.40 | 19.20 | 15.59 |
| Native 0.05 ms | 1007.70 | 17.37 | 14.20 |
| Tolerances /100 at 0.2 ms | 1775.52 | 122.27 | 118.25 |

Replay includes 10.87 s of frozen projections. Tight accuracy effort costs
about 6.4× per step. The archived 0.1 ms cost of 1328.30 s / 18.98 ms per step
covers its full 7 s, not this prefix. Shared-device load limits timing isolation.

**Verdict:** retain the 0.9 gate as the experimental candidate. Pressure
convergence and solve cost remain unresolved. Establish stage-specific,
adequately converged pressure/velocity corrections before expensive spatial,
analytic-wall or matrix expansion and default promotion. Do not select
tolerances to fit a measured peak. Existing gravity-only rotating-frame and
fluid-probe/reference-transducer limitations remain unchanged. This closes
the declared convergence experiment, not the entire qualification programme.

[Full report, figures, recordings and evidence](.tmp/dfsph-pressure-convergence-2026-10-26/README.md).


Analytic-wall transfer started — 26 October 2026 (user): the shared native
projection already accepts analytic pressure geometry. Expose the existing
`df_sampling_gate.minDivergenceDensityRatio` row setting in the prepared matrix
adapter, keeping absent settings unchanged. Validate gate selection on analytic
wall-completed density, then run matched baseline (gate 0) and candidate (0.9)
sloshing through 7 s on the exact saved fitted-wall analytic fixture. Native
DF 0.1% / CD 0.01%, cap 100, dt ceiling 0.2 ms, GPU FP32, physical viscosity
1e-6, video/progress/guards, sequential. Shared IC/mass/wall fit/forcing, no
retuning or default promotion. Report all impacts, stage pressure, energy,
sparse residence, sampling/walls, caps and cost. Initial first-impact sample
is retained as an interior analysis window within the full records.


### Analytic-wall density-gate transfer — 26 October 2026

The prepared matrix adapter now consumes
`df_sampling_gate: {minDivergenceDensityRatio: 0.9}` for either wall
representation. The shared native solver already handles analytic pressure
geometry; no separate gate or force mask is introduced. A new analytic
projection test verifies that wall-completed density keeps the dense bottom
layer active when its fluid-only sum is below the threshold. Six matrix
configuration cases cover absent/zero/0.9 settings on both representations.
All 47 focused checks pass.

Completed matched baseline (gate 0) and candidate (0.9) sloshing runs through
7 s on GPU FP32, native DF 0.1% / CD 0.01% bounds, cap 100, 0.2 ms step ceiling,
C4/4dx, physical ν = 1e-6 and the exact saved fitted-wall analytic fixture.
This user-directed transfer proceeds with the acknowledged chaotic sensitivity;
no tolerance or threshold fitting, new relaxation or default promotion.

| Analytic result | Baseline | Gate 0.9 |
|---|---:|---:|
| Persistent near-self / rho<0.3 / rho<0.5 cohorts | 6 / 44 / 64 | 0 / 0 / 0 |
| Integrated K/M | 0.725202 | 0.722891 |
| Filtered peaks [kPa] | 1.789 / 1.509 / 2.178 | 1.801 / 2.614 / 2.410 |
| Peak times [s] | 2.361 / 4.057 / 5.623 | 2.380 / 4.030 / 5.660 |
| Pressure trace error [%] | 74.45 | 71.65 |
| Final paired fraction [%] | 5.77 | 8.09 |
| Saved-frame max excursion beyond fitted faces [dx] | 0.857 | 0.866 |
| Saved-frame max count beyond 0.5dx | 4 | 9 |
| DF / CD caps | 0 / 0 | 0 / 0 |
| Wall seconds / ms per step | 797.07 / 22.77 | 803.65 / 22.96 |
| Engine ms per step | 19.40 | 19.54 |

Persistent cohorts use the same late 6.2–7 s near-bulk residence criterion as
the sampled-wall tests. Ordinary sparse spray remains; the gate removes the
persistent surface layer. Baseline cohort samples have positive own DF pressure
93.68% of the time, while own CD pressure is zero. Every saved gated sample
below 0.9ρ₀ has zero own DF pressure. Integrated kinetic energy changes only
−0.32%, preserving driven motion. No mass drift, timestep violations, solver
caps or speed alarms; both source, initialization, terminal-field and video
audits pass.

Pressure is compared with the same 10 ms filter, without alignment. Measured
peaks are 1.800 / 2.322 / 1.171 kPa at 2.390 / 4.066 / 5.697 s. First/third
phase improves in this realization, second phase worsens, and the third peak
remains too large. All-fluid mean DF post-second-impact residual falls
348.78→4.09 Pa, but sensor residual rises 79.55→133.93 Pa and mean CD residual
29.80→37.29 Pa. Large raw sensor transients remain (12.08→19.27 kPa maxima).
These signal residuals include physical transients and probe effects. Pairing
and transient wall crossings worsen, despite no endpoint crossing in the gate
run. Saved frames can miss excursions between frames; nominal every-step
half-dx crossing counts reach 4→12.

Analytic baseline/gate initial tensors and first-step scalar diagnostics match
exactly. Both retain the existing corner density defect (initial maximum
1.01088ρ₀) and first-step speed 0.3727 m/s. Across wall representations, fluid
positions, velocities, supports, UIDs and physical pressure match exactly;
fluid mass differs by only 0.000036%, while wall-completed density differs.
The direct analytic A/B isolates the gate; the cross-boundary curves include
these discretization/startup differences. Existing gravity-only rotating-frame
and fluid-probe/reference-transducer limitations are unchanged.

**Verdict:** the selected gate transfers successfully for re-merging on analytic
walls at essentially the same measured cost. Pressure precision, packing and
containment remain follow-up issues. No statistical ranking or pressure
convergence claim is made from one realization per variant.

[Report, figures, videos and source/field audits](.tmp/dfsph-analytic-sloshing-2026-10-26/README.md).


### Native incompressible consolidation and adoption — 26 October 2026

User requested documenting the solution, aligning all new paths and committing.
Native clamped DF now defaults to the selected 0.9ρ₀ sampling-recovery gate.
Classic, predictor-first and VD+PS share its implementation with either wall
representation; signed divergence and density projection remain unchanged.
IISPH shares operators, wall completion, preparation and force contracts, while
its density-only constraint already permits under-sampled fluid to recover.
The weak-factor cutoff and gate remain distinct protections.

Alternate physical-stage coefficients now follow Dirichlet density evaluation,
matching classic DFSPH/IISPH. New prepared plans use version 9,
`sampling-recovery-v1`, explicitly enabling recovery in all native DF orderings.
Canonical closed/periodic preparation has its own common gate policy independent
of the evolution method or which row first creates the fixture. Historical
revisions and saved manifests lacking a gate field retain zero; explicitly
serialized zero also round-trips. Existing matrix results are preserved.

Validation: 493 broad FP32 CPU/CUDA checks passed; the two precision-specific
gradient checks passed separately in FP64 on CPU/CUDA. Final configuration and
canonical-preparation suite: 63 passed. Extended density-gate suite: 23 passed,
including actual physical/position stage routing for all three orderings and
both wall types on CPU/CUDA. Suites overlap. The normal CLI generates the
expected 60-row version-9 screen plan for the four native comparison methods.
No additional full matrix replay or performance/accuracy ranking is claimed.

[Sampling recovery](docs/incompressible/SAMPLING_RECOVERY.md) is the current
policy guide; two curated figures and a portable metric summary accompany it.
Earlier experiment-time defaults/failures remain a historical record. The
selected classic-flow remedy transfers, while spatial accuracy, pressure
transients, pairing and containment remain qualification topics.
Validation logs: `.tmp/incompressible-consolidation-2026-10-26`.


### Native DFSPH performance investigation — 26 October 2026

User requested a checkpoint then investigation of the runtime gap versus CPU
SPlisHSPlasH. Checkpoint is `529f431`; tracked tree was unchanged at task start.
Profile the adopted native solver, separating step/observer/output costs and
declaring CPU-reference workload differences. Compilation/audit overhead are
excluded from throughput timings. Short counting/timing windows omit video
under the tiny-A/B exception; full evolving controls retain it. Numerical
defaults, stage order, stopping rules and physical initialization stay fixed.

Performance investigation complete: current local GPU solver takes 21.89 ms/step
on the shared 465-particle native tank inputs, versus 0.297–0.313 ms CPU reference.
Initial geometry/velocity/mass/support sets are bitwise equal after native
spatial reorder, but wall laws, relaxation and CD counts differ (11.46 local
versus 5.25 native). Bare current sloshing still costs 13.97 ms sampled /
17.09–17.15 ms analytic at DF 1 / CD approximately 2, and 18.61 ms for the
resumed developed analytic state.

No output-only explanation: the engine audit has 1,259 Torch operations /
38 syncs analytic and 876 / 30.25 sampled; a separate trace has 968 CUDA kernels /
7.15 ms device activity versus 26.44 ms profiled integration wall. Major causes
are per-step contact-search rebuilding, repeated exact wall integrals, eight
rigid-transform scalar transfers, small-op iteration bookkeeping, duplicated
physical timestep reads and recomputation of geometry/gradients. Native
Release/AVX caches Vj grad Wij.

Temporary in-step integral reuse and tensor-only transform assembly preserve
every saved final tensor bitwise after 330 steps. The latter removes eight
integration syncs; total measured gains are small/noisy and no production change
is adopted. Next implementation targets: shared geometry/workspace/contact
index, eliminate round trips with correct stream ordering, then fuse/capture
iteration work while preserving stopping/nonfinite/stage semantics across all
three orders and both walls.
See [full performance report](docs/incompressible/PERFORMANCE_2026-10-26.md) and
portable receipts. Full traces/fields/failures retained in
`.tmp/dfsph-performance-2026-10-26`. No full matrix rerun.


### Particle-wall DFSPH startup scaling — 26 October 2026

User requested a resolution sweep of dam-break startup compute cost against
CPU SPlisHSPlasH, measuring where the GPU launch plateau ends and whether the
curves intersect. Boundary implementation is under simultaneous development:
do not edit it; freeze a read-only benchmark dependency snapshot if needed.
Use declared common fluid/wall geometry, support and timestep scaling, report
iterations and neighbour work, separate warmed solver cost from output/setup,
and distinguish work-matched probes from production-policy differences. Short
startup timing windows omit video under the tiny-A/B exception, retain progress
and watchdogs, and run GPU cases sequentially. No numerical tuning to force a
crossover; report an observed bracket or the measured range without one.

Scaling design is frozen before measurement: 3.2×1 m particle tank, 1.2×0.6 m
left column, six physical wall layers plus normal local ghost samples; FP32,
common cubic support 4dx with one-time lattice mass normalization, C4/4dx GPU
overlay, native pressure policies (DF0.1%/CD0.01%, caps100, local omega0.25 /
native0.5). Repeated first integration resets x/v/rho/pressure histories outside
the timer, retains the normal main-list reuse and native neighbour search.
Both solvers converge at DF1/CD2 at every completed size. Timestep is kept
below common viscosity limits, with negligible single-step displacement.
Boundary snapshot copies116Python sources, verified unchanged; no live boundary
source edits. CPU uses four/eight P cores on the Ultra9 285K, affinity declared.
Initial sweep450–1,843,200fluid particles; refine nx225/250around the observed
four-core crossing and nx500around departure from the fixed-cost plateau.
First native setup failed to resolve an absolute wall-file path; preserved,
corrected to scene-relative before any valid CPU measurement.

Particle-wall scaling complete: 44 workers at ten resolutions, 3,168 timed
integrations plus 528 warmups; every solve uses DF 1 / CD 2 without caps.
GPU fixed-cost floor is approximately 12 ms through 28,800 fluid particles;
the stable four-P-core CPU curve crosses between 28,800 and 36,450 (roughly
34,000 by local interpolation). Block ranges are separated at both endpoints.
A 20%-above-floor departure criterion brackets the GPU knee at 115,200–180,000.
At 1,843,200 fluid particles GPU cubic/C4 costs 95.9/97.1 ms, versus CPU four /
eight / all 24 cores 851.5/662.8/594.1 ms; GPU is 6.2 times faster than full CPU.
More-thread CPU configurations are variable at smaller sizes; raw values and
block ranges retained, not replaced by selected best timings. C4 follows cubic.

Shared FP32 fluid/wall IC audits and effective-volume checks pass on all workers;
all GPU imports use the unchanged pinned 116-file boundary snapshot. The live
boundary repository was not edited. Local ghosts/contact Shepard/padded lists
differ from native Akinci/self pressure: this is their actual implementation
cost at common geometry/support/iteration counts, not identical wall operators
or a developed-flow accuracy ranking. Local sqrt(2) Verlet padding produces
about 100 candidate neighbours per bulk fluid row, versus native roughly 48
true fluid neighbours; 199.3 million candidate entries at the largest size.
No production optimization or numerical policy change in this measurement.

[Report and scaling figure](docs/incompressible/STARTUP_SCALING_2026-10-26.md),
[benchmark](scripts/bench_dfsphScaling.py); portable 44 receipts/IC audits and CSV
under `docs/incompressible/startup_scaling_2026-10-26`. Full fixtures/dependency/
logs/setup failure preserved in `.tmp/dfsph-startup-scaling-2026-10-26`.
Next target is the roughly 12 ms fixed overhead for existing small cases, then
redundant traversal/geometry work at larger sizes.

User decision: retain the startup scaling benchmark and its current receipts as
the baseline for subsequent performance work. Preserve the shared fixtures,
pinned boundary dependency and original measurements; future optimizations
should add comparable results rather than overwrite these. The approximately
12 ms plateau supports fixed dispatch/launch/synchronization overhead; device
saturation itself was not measured by this sweep.


### Clean checkpoint before the next matrix — 26 October 2026

User requested committing the repository to a clean state before any new
comparison run. Retain the startup benchmark, performance reports and curated
measurements in this checkpoint. A fresh matrix of the current solver and
boundary-handling configurations is the next candidate; execution is explicitly
deferred. Before a later run, record the then-current dependency revisions and
initialization/settings so concurrent boundary development is attributable.
No new simulation or benchmark runs were launched for this checkpoint.
An unrelated untracked preview image was preserved with a SHA-256 receipt under
`.tmp/preserved-scratch-2026-10-26` rather than included in the commit.


### Overnight native incompressible matrix — 26 October 2026

User authorized starting the deferred row runs with exclusive overnight GPU
access. Queue the four current native solver compositions at full resolution
using version-9 sampling-recovery preparation (60 slots, 52 supported). Fresh
shared fixtures per case/wall configuration; preserve nominal-volume relaxed
closed/periodic preparation and calibrated/fitted free surfaces. Freeze current
frontend/core/integrator/boundary/plotting source dependencies before launch;
record live Git revisions and content hashes. Run sequentially with VisPy video,
progress, finite/velocity/stall guards and existing per-row wall budgets.
No tolerance/model tuning or automatic failed-row retries. Existing historical
WCSPH/legacy results remain unchanged. Start with hydrostatic rows, then dam
break/sloshing, followed by periodic/closed and obstacle rows.

Launched under a detached supervisor (PID 3086687; queue PID 3086856) in
`.tmp/incompressible-native-matrix-2026-10-26-night`. Frozen revisions:
frontend `56023e87`, core `98081c1d`, integrators `1fea2a3d`, boundaries
`c624db74`, plotting `05caeb33`, remote `b549e134`; full file hashes and runtime
import checks accompany the manifest. The first particle-wall DFSPH hydrostatic
row is advancing, with recorded frames/trajectory, finite fields, no penetration
and no iteration caps in the initial checks. Live status and per-row reports are
updated by the durable supervisor; final analysis, evidence audit and comparison
assembly are queued after the rows finish. These are launch checks, not final
matrix results.


### Overnight matrix rebuilt — 27 October 2026

The frozen queue finished at 04:29 UK: 51/52 attempted rows reached their
endpoints; IISPH periodic TGV stopped at t=4.781/5 s on the velocity guard
(72.5 times the reference scale). All 52 attempts pass the evidence audit,
with 464,889 steps, no setup mismatches, missing terminal tensors, measured
mass drift or timestep-limit violations. Summed simulation wall time is
5.076 h (39.3 ms/step averaged over heterogeneous cases, not a solver benchmark).

The automatic combined-report step failed because its composer required all
eight historical prepared methods. Add explicit `--native-only` admission for
exactly the four authorized native methods, retaining the strict default and
case/wall slot checks. Preserve the failed IISPH outcome and all historical
control rows; no simulations rerun. Rebuilt seven-method report:
`.tmp/fluid-matrix-native-comparison-2026-10-27/comparison.html`, with quality
and cost figure, curves, exact metrics, video links and hashed provenance.
The complete comparison has 105 slots: 76 endpoints, four failures (including
three retained ACSPH failures), 25 unsupported. All completed plotted cells
have their expected metrics. Legacy projection experiments remain separate.

At the existing case metrics, predictor-first DFSPH leads sloshing with both
walls (64.78% / 65.79% pressure-trace error), followed by classic DFSPH
(71.50% / 71.65%). Coupled hydrostatic losses are below 0.6%; native VD+PS
has roughly 23% hydrostatic loss and frequent caps. Native TGV errors are
6.63–6.93%, above retained WCSPH controls; native periodic vortex energy losses
are about 99.3–99.4%, versus about 87% for retained controls. Different physical
presets and initialization prevent interpreting this as a solver-only A/B.
These rankings do not promote a default or qualify capped solves.


### TGV failure and native pressure-boundary follow-up — 27 October 2026

User requests investigation of the IISPH periodic TGV velocity stop and weak
TGV performance of the three native DFSPH compositions, plus rebuilding the
pressure-boundary formulation for native DFSPH. Start from the frozen overnight
records, separate initialization/source/solver/transport effects, and reproduce
the failure with focused instrumentation before changing a numerical choice.
Review the published pressure-boundary method and current legacy implementation;
implement an explicit shared native option and qualify with recorded tank and
flow controls. Preserve the matrix and its failed evidence; no broad rerun or
default promotion is implied. Track accuracy, conservation, caps and cost.


TGV follow-up results: instrumented IISPH and DFSPH work replays match frozen
matrix endpoint fields bitwise. Actual D^-1 A spectrum rises from 4.272 to
8.10469 before IISPH's 4.781 s failure; omega .25 requires lambda<8. Half dt
fails at4.8405s (lambda8.45976); omega .1 delays the stop to4.9893s but does
not repair dense clustering/mean sampled-density drift. Do not loosen guards
or promote lower relaxation as a physical fix. Error paths retain numerical
prefixes/videos; half-dt and interrupted-budget cleanup subsequently segfault.

The prepared periodic fixture remains near crystalline (Q4 .999740); the
saved rearranged/cooled control has Q4 .256752, fixed nominal mass, and its
original duration-limited preparation qualification. Exact TGV advection of
near-lattice samples alone yields mean kernel rho1.14069 at5s. A four-cell
IC x DF-tolerance study plus intermediate tolerance isolates both effects.
Near-lattice DF .1% gives velocity error21.28%, energy-curve error6.669%; at
.0001% these become4.66%/4.493%. Rearranged .1% gives13.58%/3.077%; rearranged
.0001% gives2.67%/1.481%, rho_mean1.00491, but527 DF caps. Runtime45.09s/6.50ms
engine perstep baseline vs178.86s/33.20ms best candidate. Raising the latter
budget to1000 still caps118 times by0.651s; explicitly stopped at0.786s,
not a qualified completed solution. Other orderings not requalified at this
bound. DF .1% atdt1ms allows mean normalized divergence1/s; rate-error scaling
and conditioning need further work.

Work accounting: loose near-lattice DF/CD corrections +6.709/-6.149 J/m,
viscosity -3.781; rearranged/tight -0.042/-0.008/-2.007. Fixed-geometry
viscosity damping of recorded loose velocity is1.63x analytic, but the exact
TGV field at those same positions gives.92x: changing nu would mask the
numerical state error. See docs/incompressible/TGV_2026-10-27.md.

Rebuilt pressure boundaries as explicit DFSPHConfig.pressureBoundary='band2018'
(volume-centric, coupled fluid/static sampled-solid unknowns) in all three
native orders. Reuses bare pair kernels, exact diagonal and native histories/
policies; no legacy Tikhonov or automatic gauge. Local full-band volume and
fluid-contact masks are declared deviations from the single-layer paper.
Ghost queries excluded, clamped fluid DF uses the shared .9rho0 recovery gate.
Analytic/moving-wall forms are explicitly rejected. Dense 2D/3D force/C/diagonal,
Jacobi, ghost/nonmutation/config/gate tests pass in FP32/64;182 FP32 regression
checks pass with one skip, plus12 FP64 new-path checks. Default native results
remain unchanged.

At declared1000 budget the classic tank completes4s, zero caps/penetration,
tail median pressure gradient ratio1.00128, RMS speed.000245m/s,235.67s wall /
54.41ms engine (native346.25s/80.74ms). Predictor-first .25s is quiet. VD+PS
.25s still hasRMS.388m/s and219 shift caps: pressure slope alone is not a pass.
The1s dam-break prefix completes with zero caps/penetration in36.55s/14.17ms,
but integrated K/M is85.15% of native and84.00% of retained Delta. Solved wall
pressure spread is also unphysical despite a good bulk tank slope. Keep the
rebuilt full-band variant experimental; next boundary work is conditioning and
faithful surface sampling, not default promotion. Documented in
 docs/incompressible/PRESSURE_BOUNDARIES.md; reusable
scripts/probe_preparedProjection.py selects saved fixtures and explicit overrides.
All raw evidence, plots, sources and error receipts are retained under
.tmp/tgv-pressure-investigation-2026-10-27/ with KEEP. The normal matrix is intact.

Final evidence check: all16 attempt source archives verified against their
recorded hashes, all16 videos fully decoded, terminal fields finite and
particle masses unchanged wherever terminal tensors exist. Normal-matrix
manifest/metric hashes remain unchanged. Final tests:182 FP32 passes/one skip
and12 FP64 passes, including internal pressure momentum/torque conservation.
Report, failed/partial runs and provenance retained with KEEP.


### Native incompressible retirement — 27 October 2026

User defers further pressure-boundary investigation and requests retiring the
legacy projection implementations in favour of the three native DFSPH stage
orders and IISPH. Remove redundant runtime drivers, migrate active cases and
matrix generation, retain historical evidence/readers, and test routing,
initialization, stage ownership, histories and numerical contracts. This is
code consolidation, not numerical default tuning. Preserve the experimental
pressure-boundary option without promoting it. Old names/configuration must
fail with migration guidance rather than silently selecting different physics.
The replacement is not merely a Krylov-to-Jacobi change: legacy paths also
used Jacobi by default, with different operators, policies and wall closures.


Retirement completed: supported runtime entries are native dfsph / dfsphPredictor /
dfsphShift / iisph. Removed four legacy scheme drivers, their pressure/shift/
Krylov adapters and obsolete switches/configuration; retained generic Krylov,
ACSPH wall primitives, performance benchmark, historical reports and source
checkpoint18d069d. Case/CLI/example/matrix routing uses the current cohort;
old names and IDs cannot silently select different physics. Experimental native
Band option preserved without further physics work or default promotion.

Native initialization is shared across named variants; analytic tank/channel/
array wall mass calibration now applies once. Sloshing and matrix diagnostics
read dedicated stage fields and exclude the VD+PS shift potential from physical
pressure, without mutating solver carry. Legacy analytic per-body load booking
is not ported; native channel/array diagnostics omit load/drag fields explicitly.
Moving bodies/compact preset remain unsupported, with tested refusal. Native
Morris analytic walls require their explicit no-slip moment adapter.

Validation:406 FP32 checks pass /2skip;215 FP64 contracts pass after correcting
2 dtype-only expected vectors (213initial +2rerun). Adapter131checks overlap.
All12 GPUFP32 pre/post tensor sets and current changed-dt HDF5 continuations
match bitwise across4methods ×3setups. Three-step timed windows0.628/0.600s
before/after,4.30–69.45/4.35–67.34ms perstep; too short/cache-sensitive for a
performance claim. Raw periodic smoke:9pass/2strict expected failures. The raw
absolute-density TGV injects energy identically in the preserved native solver
and cleanup (20-step seeded energy/time series bitwise, decayratio−18.4228).
Kept original physical assertions as strict xfails; no equation/tolerance tuning
to make the cleanup appear physically qualified. Existing matrix limitations
remain. Full collection succeeds; unrelated full physics battery not run.

Migration/operators/policies/retired probes and limitations are documented in
docs/incompressible/MIGRATION.md. Local receipts/source A/Bs retained with KEEP
under.tmp/native-retirement-2026-10-27; no existing matrix or failure evidence
overwritten. Further pressure-boundary work remains deferred by the user.
