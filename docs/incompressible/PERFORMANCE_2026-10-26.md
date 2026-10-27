# Native DFSPH performance investigation — 26 October 2026

Checkpoint: **529f431**, `Add native DFSPH variants and shared sampling recovery`.
The tracked implementation was unchanged at the start of this investigation.
No production solver, numerical policy or default was changed by this work.

The user's concern is confirmed. On the original shared 465-particle tank IC,
current local GPU DFSPH costs **21.89 ms/step**, compared with **0.297–0.313 ms**
for CPU SPlisHSPlasH: approximately **70–74×** in this short window. This is a
comparison of implementations on shared physical inputs, **not identical wall
operators or convergence trajectories**. The reference needs 5.25 density
iterations on average; local needs 11.46. Nevertheless, the current sloshing
solver also costs 14–19 ms with only 1 DF / approximately 2 CD iterations.
Implementation overhead remains substantial independently of iteration count.

## Measurements and scope

Sequential local GPU windows; FP32 simulation state, RTX PRO 6000 Blackwell.
CPU reference: local Release build at `eccce86155776f6ac52d5080b1f720a52bf29450`,
AVX and `USE_PERFORMANCE_OPTIMIZATION` enabled, OpenMP 4 threads, FP32.
One GPU process was present at the process-inventory check. Clocks were not
locked; short timing differences below approximately 1 ms need repeats.

Each throughput window excludes 30 warmup steps and measures 300 steps.
The prepared-case harness selects 299 integration rows from that window for
its engine mean; loop and iteration means use all 300 intervals.
The call audit uses 10 warmup + 20 counted steps; its timings are excluded.
The CUDA trace covers a separate 10-step window after throughput measurement;
profiling increases CPU overhead, so its section durations are not throughput
measurements. Tiny counting/timing A/B windows omit rendering and storage,
with explicit progress and existing finite/velocity/stall checks. No long
accuracy trajectory or matrix replay was run.

| Window | Fluid / total particles | Step ms | Full loop ms/step | Mean DF / CD iterations |
|---|---:|---:|---:|---:|
| SPlisHSPlasH tank, CPU, two repeats | 465 / 465 fluid, volume-map walls | 0.297 / 0.313 | Not measured independently | 1 / 5.25 |
| Current local tank, GPU | 465 / 465 | 21.89 | 22.32 | 1.19 / 11.46 |
| Sloshing, sampled walls, bare | 4,200 / 10,660 | 13.97 | 14.88 | 1 / 2.02 |
| Sloshing, analytic walls, bare, two repeats | 4,200 / 4,200 | 17.09 / 17.15 | 17.58 / 17.77 | 1 / 2.05 |
| Analytic sloshing, resumed at saved t=7 s | 4,200 / 4,200 | 18.61 | 19.19 | 1 / 2.003 |
| Analytic sloshing, standard case diagnostics | 4,200 / 4,200 | 18.73 | 20.59 | 1 / 2.05 |

Local step time is the runner's integration event interval; loop time is the
synchronized post-step interval. Periodic progress and hooks remain included.
For the CPU reference, `timeStepNoGUI()` is timed directly, without field
observation inside the timed interval. Full-run wall receipts, including first
step/module cache work, are retained in the JSON; do not substitute those for
warmed throughput.

The tank uses identical initial fluid positions, zero velocities, masses,
supports, cubic kernel, 1 ms timestep and rest density. Initial tensor sets match
bitwise after accounting for the native spatial reorder. Both use DF 0.1% / CD
0.01%, maximum 100 iterations, minimum DF1 / CD2. Local retains its selected
0.9rho0 DF gate and omega=0.25, versus native neighbourhood guarding and
omega=0.5. Local exact solid integrals/contact Shepard differ from native
Bender2019 volume maps; fluid viscosity closures also differ. Both tank probes
use fluid-only viscosity, matching native zero wall viscosity. No relaxation,
wall law or stopping-tolerance tuning was done to improve the timing ratio.

The sloshing windows retain the established calibrated fixtures, C4/4dx,
0.2 ms ceiling, case viscosity 1e-6 and selected gate. The developed window
restores the saved full state and resumes its physical time at 7 s. These are
cost probes, not new sloshing validation results.

The prior recorded full analytic gate run cost 19.54 ms/step in the engine and
22.96 ms in the full loop over 35,000 steps (803.65 s wall). Its extra gate/work
observers, HDF output, daemon and video differ from these bare windows. Removing
output clearly does **not** reduce the engine to reference-like throughput;
those historical/full and current/short numbers are not a controlled A/B.

## Where the time goes

Lightweight host section timers for the bare analytic window:

| Stage | Host ms/step |
|---|---:|
| Main Verlet-list validity/reuse | 0.30 |
| Summation density + analytic wall density | 1.70 |
| Pressure geometry/contact weights/diagonals | 4.60 |
| Divergence projection | 2.18 |
| Viscosity including analytic wall closure | 3.97 |
| Density projection including known wall pressure | 3.23 |

The remaining interval includes the integration driver, rigid-body finalization,
other small stages and synchronization of pending work. Host sections include
launch/dispatch/waiting; these are not additive GPU-busy timings.

The traced analytic window launches **968 CUDA kernels per step**, with only
**52 Warp launches** in the startup call audit; most launches come from small
Torch operations. Summed device activity in the trace is **7.15 ms/step**,
versus **26.44 ms** of profiled integrator wall duration. This diagnoses long
host/launch/synchronization gaps; summed device activity is not a standalone
prediction of achievable runtime. Five box-table evaluations alone use about
**3.34 ms** of device time. These exact box lookup kernels and the analytic
wrapper outputs explicitly use FP64 even with FP32 fluid state; the simulation
has not fallen back to CPU or globally switched to FP64.

The audit counts, inside the integration call:

| Walls | Torch operations/step | Warp launches/step | Synchronization calls/step |
|---|---:|---:|---:|
| Analytic | 1,259 | 52 | 38 |
| Sampled | 876 | 37.8 | 30.25 |

The following are concrete sources, not inferred from GPU utilization alone:

1. **Contact geometry builds a second neighbour search every step.**
   `computeIISPHContactWeights` calls `buildVerletList` without prior contact
   adjacency, over all particle contact queries, even though only near-wall
   fluid rows need pressure support. The main Verlet list is reused in these
   startup windows. This extra Gather search accounts for six synchronization
   calls each step and is shared by native IISPH and DFSPH.
2. **Repeated analytic geometry and synchronous wrappers.** Density and pressure
   geometry each request the same lean integral at the same positions. Viscosity
   requests the larger three-group layout, including channels it does not all
   consume. The box/fused wrappers explicitly synchronize; the analytic step
   audit sees nine calls through `edge/precision.py:sync`.
3. **Rigid-body finalization extracts eight GPU scalars every step.**
   `getTransformationMatrix` assembles nested `torch.tensor` literals from
   GPU cosine/sine/center tensors. It runs even for these stationary bodies,
   including the analytic body with no sampled particle rows.
4. **Projection iterations are many small operations.** Each iteration builds
   contact pressure, traverses neighbours for F(q), traverses them again for
   C(F(q)), then performs Torch updates, masks, reductions and generic
   stationary-driver nonfinite/stopping/history bookkeeping. Convergence is
   read each eligible iteration, with a separate final statistics transfer.
   Both solves also recompute the base diagonal for the weak-factor cutoff
   when contacts exist, although geometry already computed that base.
5. **Timestep reductions are duplicated.** The registered sloshing case already
   calls `projectionTimestep` via the Kolmogorov timestep helper, while the
   runner also wraps it with `limitProjectionCase`. The audit attributes approximately eight scalar/shape
   synchronization calls per step to the timestep hook, including dynamic
   fluid indexing and h/vmax reads. This is outside the integration counts.

In contrast, the pinned SPlisHSPlasH source performs a neighbour search then
precomputes **Vj grad Wij once per step**. Its AVX loops consume those cached
weights during factor, pressure-acceleration and density-rate work. It runs
coarse C++/OpenMP loops with reductions in CPU memory, avoiding the per-Torch-op
GPU launches and device-to-host convergence transactions. Its wall maps also
avoid our repeated exact box integrations. The comparison therefore identifies
both execution overhead and additional/repeated arithmetic.

## Bounded, temporary ablations

These probes alter only the profiling harness. They are not production fixes
or new defaults, and their short-window parity is not a general qualification.

- **Reuse identical lean wall integrals within each static step:** all saved
  final tensor fields remain bitwise identical after 330 steps. Aggregate
  requests fall from three actual evaluations per step to two (two equal lean
  requests share one result; the viscosity layout remains separate). Paired
  geometry-stage time falls by about 0.9 ms in the first pair. Whole-step
  measurements are noisy: baseline 17.09/17.15 ms, reuse 17.63/15.68 ms. Their
  two-run means suggest only approximately 2.7% benefit, without a robust speedup
  claim. This removes one duplication, not the whole wall cost.
- **Assemble the same rigid transform with device tensor stacks:** every saved
  final tensor remains bitwise identical after 330 steps. Synchronizations
  inside the analytic integration call fall **38 -> 30**, with no iteration
  changes. One timing is 17.03 ms versus baseline 17.09/17.15: no convincing
  total speedup established. Eight avoidable scalar transfers are still real,
  but they are not the whole performance gap.

The first profiler harness had an off-by-one stop index and exited with an
active profiler, causing a segmentation fault at teardown. It was corrected
and repeated successfully; its original throughput receipt is excluded from
claims. The initial cubic-tank attempt hit unsupported exact cubic wall
viscosity before step one; the declared fluid-only reference policy completed.
Both failures remain in the scratch evidence.

## Recommended implementation sequence

1. **Shared step geometry/workspace first:** retain base/contact diagonals,
   contact support and raw wall integrals once per fixed geometry; restrict
   contact work to relevant queries and reuse the reference spatial index.
   Invalidate on drift, shift, wall pose/support changes, and keep VD+PS's
   second geometry evaluation after its drift. Preserve Dirichlet ordering.
2. **Remove host round trips and fuse eager bookkeeping:** tensor-only body
   transforms; one physical timestep reduction/transfer; boundary kernels on
   the active Torch stream with explicit interop ordering rather than global
   synchronization. Simply deleting boundary synchronization is unsafe.
3. **Cache neighbour-weighted geometry and reduce iteration launches:** use a
   common IISPH/DFSPH pressure workspace, fuse vector update/masking/check
   bookkeeping, then capture the stable iteration work where beneficial.
   Preserve the current residual definition, minimum iterations, stopping test,
   nonfinite handling and returned-iterate semantics. Do not mask the problem
   by loosening tolerance or changing pressure relaxation.

Every optimization should be tested against the checkpoint on frozen inputs
and a short recorded evolving control, including all three orderings and both
wall types. There is no evidence yet for an achievable final speedup; the CPU
reference's time is a target, not a promised GPU result.

[Portable receipts, source fingerprints and audit reports](performance_2026-10-26/summary.json).
The three archived profiling scripts require the declared local fixtures;
run them from the repository root with `PYTHONPATH=.` in the `warp` environment.
Full traces, saved fields, repeated timings and failures are preserved in
`.tmp/dfsph-performance-2026-10-26` (`KEEP` present).


## Particle-wall scaling follow-up

The [startup scaling sweep](STARTUP_SCALING_2026-10-26.md) separates the fixed
roughly 12 ms particle-wall overhead from large-size growth. The GPU crosses
the four-core CPU reference around 34k fluid particles, and at 1.84M costs
95.9 ms versus 594.1 ms with all 24 CPU cores. The live boundary package was
left untouched and a pinned dependency copy was used for all measurements.
