# Driven MFM/MFV performance investigation — 2026-10-27

Status: short-window profiling and an opt-in search optimization. The existing
long-flow physical/statistical qualification remains in
[the qualification report](driven_turbulence_qualification_2026-10-27.md).

## Measurement protocol

The RTX PRO 6000 is occupied by an independent simulation throughout these
measurements. That job is left untouched. Report observed repeated, interleaved
A/B comparisons under contention; do not rank hardware, extrapolate uncontended
throughput, or use these timings to qualify GH200 capacity.

`profile_drivenTurbulence.py` holds B7, n_h=4, finite gamma=5/3, native CFL=0.1,
fresh two-stage SSPRK, forcing seed 1234 and the thermal/source policy fixed.
For this timing cohort the thermal model is relaxation, cooling time 0.2, OU
correlation time 0.5 and update interval 0.05. Acceleration inputs are 0.3/6.
The main windows start from the saved seed-1234 weak/strong MFM states near t=25
at 64² and 128², discard 20 warmup steps, and measure 80 accepted steps. Both
methods receive the identical snapshot for a given input. The MFV windows are
therefore controlled local workloads initialized from MFM, not mature MFV
trajectories or a comparison of long-run method accuracy.

Three repetitions alternate search-path order. Additional runs measure 256²
startup (20 warmup + 60 measured steps) and the cost of full diagnostics / every
RHS observation. These extra points have one repetition and are exploratory.
Setup, compilation, warmup, final NPZ writing and video encoding are excluded.
Video is omitted for these tight timing A/B loops. Step/time progress remains.
The measured native runner includes adaptive timestep, accepted-state refresh,
speed/finite watchdog reads, its integrator timer synchronization, and minimal
trajectory bookkeeping. Full diagnostics and RHS checks are separate modes.
Snapshot output/compression and video overhead from the long qualification
harness are not included in these timings.

Profiler-instrumented runs explain call sites and launch/synchronization counts;
use unprofiled repetitions for wall time. CUDA event spans also include launch
gaps and scheduling delays, and are not a measure of active kernel time.

## First optimization: exact neighbors from a padded candidate list

`solver.neighborSkin=1` preserves the existing rebuild-every-geometry search.
Values greater than one opt into the core's existing Verlet machinery:

1. Build a candidate list with expanded support.
2. Check current particle motion and support change against its buffer before
   every use; rebuild whenever needed.
3. Filter at the **current, unexpanded support** for every fresh geometry.
4. Recompute moments, gradients, limiters and fluxes at every native stage.

Only this search work is reused. State-owned caches follow explicit integrator
clone lineage, not a process-global cache. Domain identity/tensor versions,
particle shape/device/precision and skin changes invalidate reuse. The saved
reference positions/supports are independent clones, so in-place state edits
cannot erase motion from the validity test. Restart reconstructs this scratch
cache; no cached physics is stored.

Neighbor sets are unchanged, but list ordering changes floating-point summation.
This is not a bitwise-equivalent trajectory optimization. It requires field,
budget and statistical comparisons appropriate to the use case. The default
remains 1; the measured candidate is 1.25, not a claimed globally optimal value.
A larger skin trades more candidate storage/filter work for fewer rebuilds.

## Observed timings

Medians of three unprofiled 80-step window means, in the shared-GPU phase:

| Input | Method | Rebuild ms/step | Skin 1.25 ms/step | Wall reduction |
| --- | --- | ---: | ---: | ---: |
| weak 64² | MFM | 38.64 | 27.77 | 28.1% |
| weak 64² | MFV | 38.47 | 27.58 | 28.3% |
| strong 64² | MFM | 39.35 | 31.21 | 20.7% |
| strong 64² | MFV | 39.10 | 31.48 | 19.5% |
| weak 128² | MFM | 38.65 | 28.04 | 27.4% |
| weak 128² | MFV | 38.65 | 28.00 | 27.6% |
| strong 128² | MFM | 40.24 | 32.89 | 18.3% |
| strong 128² | MFV | 39.86 | 32.28 | 19.0% |

These correspond to about **1.22–1.40× observed throughput** for this workload.
MFM and MFV have similar local step costs; the evolved timestep distribution
and variable-mass dynamics still need their own long MFV campaign. The current
results do not rank their total cost per unit of developed physical flow.

Single-repeat 256² startup: MFM 38.47→26.08 ms/step, MFV 38.94→25.61 ms/step.
This is a launch/search-dominated startup probe, not evidence that turbulent
256² costs the same as 64² or that capacity/scaling is qualified.

At strong 128², full scalar diagnostics added roughly 1–2 ms/step, and full
diagnostics plus the every-RHS stage checks added roughly 4–5 ms/step over the
minimal runner in these exploratory observations. Qualification snapshot
compression, history-file writing, progress rendering and video remain outside
these comparisons. Source/runner costs dominate the measured step, so removing
qualification observers alone would leave substantial solver overhead.

An attempted second cohort began when the first external compute job ended,
but a new ACSPH job appeared. That cohort was interrupted deliberately, after
nine complete windows. Its `free-*` directory names record the original intent;
**they are not isolated measurements and are excluded from the table**.
No external process was paused, stopped or modified. Uncontended throughput
remains an open measurement, not an inferred speedup from mixed-load runs.

## Where the time goes

Early 64² Python profiles placed about 59–60% of instrumented elapsed time under
fresh geometry construction and about 47–49% under compact-hash neighbor search
(the latter is included in the former). These are host call spans including
waits and profiler overhead, not additive exclusive GPU-time shares.

The 128² strong-state call audits are a cleaner mechanism check:

| Integrator observation, 80 accepted steps | Rebuild | Skin 1.25 |
| --- | ---: | ---: |
| Fresh geometry builds | 160 | 160 |
| Full compact-hash rebuilds | 160 | 46 |
| Torch operations per step | 881 | 634 |
| Recorded host syncs per step | 18 | 13.45 |
| Warp launches per step | 23 | 23.875 |

The filtered path launches slightly more kernels overall while avoiding many
small Torch operations and host round trips. The native runner adds its timer,
timestep and speed-check syncs beyond the integrator counts above. Remaining
cost includes validity checks, CSR edge-count reads, moment-inverse checks,
host stage-time conversion and the per-stage CPU-to-device Fourier coefficients.
The baseline CUDA trace also records the real Riemann and neighbor kernels;
range annotations in its profiler summary overlap and must not be summed as
active kernel time.

The next performance work should reduce search/CSR allocation synchronization,
then stage/forcing transfers and small pointwise launch overhead. Preserve cold
predictor validation, accepted-state refresh, native timesteps and stage
semantics. A fixed-capacity or graphed path needs overflow/rebuild handling and
independent numerical controls; it is not implemented by this change.

## Numerical scope and validation

Across the completed 60-window campaign (4,720 measured steps plus warmup),
all runs finished with finite positive final primitives. The four every-RHS
observer windows checked 800 stages including warmup: zero invalid/nonpositive
stages, minimum raw stage internal energy 0.5562. The maximum final absolute
energy-budget residual relative to initial energy was 3.638e-6; maximum total
mass change was 3.257e-8. These windows inherit the original snapshot budgets.

Matching baseline/cached final times were identical in the eight primary A/B
inputs. Conserved-state relative L2 differences ranged from 7.76e-7 to 2.05e-5
after warmup plus 80 measured steps; the maximum relative RMS-Mach difference
was 1.98e-7. Pair order changes explain the small numerical divergence; the
opt-in path has not had a new long chaotic/statistical qualification campaign.
It is available for controlled experiments, not promoted to the default.

Tests exercise exact current pair sets in 1D/2D/3D on CPU and GPU, heterogeneous
support growth/shrinkage, in-place motion, periodic wrapping and changed flags,
domain changes, count changes and clone lineage. MFM/MFV relaxation/adiabatic
short integrations compare fields and budgets. Native restart reconstructs the
cache and preserves forcing exactly, while allowing roundoff-level field
differences from a rebuilt list order. The default skin=1 retains its existing
bitwise restart contract and version-1 checkpoint signatures. The final
checkpoint-signature compatibility adjustment reproduced both measured strong
128² MFM paths bitwise in all saved fields.

Validation completed on the final runtime sources:

- **74 targeted CPU/GPU checks passed** (16.82 s).
- **334 broader CPU checks passed** across driven sources/restart, meshless
  modules/integration, configuration, runner and PESPH integration (142.48 s
  for that invocation). Hiding CUDA caused ten existing hard-coded CUDA tests
  to raise “No CUDA GPUs are available”; three CUDA-only checks were skipped.
  Those ten plus the three skips were then run with CUDA visible: **13 passed**
  (13.72 s). These were environment restrictions, not solver failures.
- **41 focused CPU FP64 checks passed** (16.40 s), including much tighter
  field/restart agreement (relative tolerance 1e-11).

The broad invocation covered `test_drivenTurbulence`, `test_meshlessNeighbors`,
`test_meshlessModules`, `test_meshlessSplit`, `test_configuration_catalogue`,
`test_configuration`, `test_runner` and `test_pesphIntegration`. Logs and final
source hashes are retained with the raw evidence. No new kernels or physical
model parameters were added.

## Reproduce and evidence

From the worktree, for example:

```bash
PYTHONPATH=src WARP_CACHE_PATH=/tmp/warpsph-warp-cache \
/home/lu26029/miniconda3/envs/warp/bin/python scripts/profile_drivenTurbulence.py \
  --scheme MFV --nx 128 --acceleration 6 --skin 1.25 \
  --snapshot .tmp/driven-qualification-20261027/strong128-s1234/samples/00250.npz \
  --warmup 20 --steps 80 --out .tmp/new-mfv-profile
```

Use a fresh output directory. `--profile`, `--audit` and `--trace` are separate
instrumented controls; do not compare their wall time to uninstrumented rows.
Use `--diagnostics full` or `rhs` to include those observers. Omitting the
snapshot starts the uniform fluid. The harness records actual advanced time,
final fields, physical diagnostics, input, snapshot/driver hashes and Git head.

Ignored raw evidence is retained in `.tmp/driven-performance-20261027/`:
`summary.json`, `summary-driver.py`, every run's input/result/final fields and
GPU-load records, campaign driver/source hashes/source archive/input snapshots,
Python profiles and detailed call audits. `campaign-source-audit.json` confirms
all recorded Python sources stayed unchanged during the complete timing cohort.
`post-signature-field-check.json` records the final compatibility-only change.
The interrupted second cohort and its source audit remain separate.
