# CRKSPH and CompSPH performance evaluation, 2026-10-21

Tracked by the SCHEME_CLEANUP_PLAN row in [PLANS.md](../../PLANS.md).

User request: investigate the reported 250+ ms timestep at approximately
70,000 particles, using Nsight Compute, and rerun the paper-matching blob in
FP32. This is a scheme performance pass; the frozen blob numerical-model
investigation remains frozen. No integrator, stage reuse, physical formula,
support policy, or precision default is changed.

## Reproduction and hardware diagnosis

The reported case is `results/materialEOS/blob_paper_long_2026-10-21`, nx=512,
70,181 particles, B8, explicit midpoint, FP64, paper density schedule, custom
native IdealH replacement, and the native final-work entropy partition.
Its saved launch configuration explicitly selected FP64. The standard blob
uses QuinticSpline and a different density/support schedule, and is cheaper.

GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition. NCU 2026.3, CUDA driver
13.3, Warp 1.17.0. The first NCU report is
`results/performance_2026-10-21/crk_accel_before.ncu-repz`.
NCU's measured acceleration kernel took 17.53 ms (standard QuinticSpline
blob), with FP64 pipeline utilization **87.32%**, DRAM throughput **1.23%**,
and L2 hit rate **96.45%**. NCU reports a **64:1 peak FP32:FP64 ratio** on this
GPU. This workload is limited by double-precision arithmetic, not memory
bandwidth or primarily Python launch overhead. CUDA graphs cannot remove
that arithmetic cost.

The final CompSPH force/heating kernel confirms the same diagnosis: **10.45 ms**
per launch, **87.04% FP64 pipeline utilization**, **1.50% DRAM throughput**. Its
report is `results/performance_2026-10-21/comp_accel_after.ncu-repz`.

A Torch/CUDA trace of the matching B8 setup assigns approximately 134 ms per
step to its two force and heating traversals, 28 ms to geometric volumes,
22 ms to moments, and 14 ms to velocity gradients. The standard CompSPH
70k-particle shock trace assigns 21 ms to forces, 9 ms to heating, 14 ms to
support iteration, and about 12 ms to gathered pair velocities and batched
GEMV for compatible work.

## Changes

* CRKSPH now evaluates the raw heating predictor in the same **fresh RHS**
  traversal as acceleration. It reuses corrected pair gradients, limited pair
  velocities and Q values. A per-stage result carries heating to its existing
  caller. Both averaged and one-sided work, and both volume-weight variants,
  retain their existing algebra. The standalone heating operator remains an
  independent reference and callable API.
* CompSPH similarly shares force/heating evaluation, retaining the **gather**
  gradient for heating before constructing the mean force gradient. It does
  not substitute the force gradient for the predictor.
* Compatible work reduces directly over each particle's CSR row. It avoids
  materializing two edge-sized gathered velocity arrays, their difference,
  and a batched dot product. Rows use deterministic serial accumulation; no
  floating-point atomics. The differentiable path retains Torch operations
  and excludes padded entries from their gradient calculation.
* The matching script's native entropy partition uses a similar row kernel,
  retaining its strict `1e-15` thresholds, denominator floor and half split.
  This is distinct from the production entropy policy. Unused RHS-estimated
  fractions are skipped when the selected controller weights final work.
* The research IdealH table cache includes precision, allowing the same
  table coefficients and formulas to be evaluated in FP32 as well as FP64.
  Its FP64 default remains unchanged.

## Final measurements

Sequential processes, eager mode, 10 warmup + 30 timed steps, no rendering
inside these short performance loops. CPU/CUDA profiling uses a separate
three-step window **after** timing. The benchmark synchronizes at its step
boundary and retains normal case diagnostics. Numbers below are synchronized
wall means, including the occasional neighbor rebuild. First-use compilation
is excluded. These are short throughput measurements, not developed-flow
accuracy qualification.

| Workload | Particles | Precision | Before, ms/step | After, ms/step | Cost reduction |
|---|---:|---|---:|---:|---:|
| Standard CRKSPH blob, QuinticSpline | 70,181 | FP64 | 137.49 | 89.46 | 34.9% |
| CompSPH Sod 2D slab, B7, nx=2800 | 70,000 | FP64 | 75.23 | 53.71 | 28.6% |
| Same CompSPH slab | 70,000 | FP32 | 11.76 | 6.87 | 41.6% |
| CRKSPH paper-matching blob, B8 | 70,181 | FP64 | 233.14 | 148.71 | 36.2% |
| Same matching blob | 70,181 | FP32 | — | 13.64 | — |

The CompSPH FP32 baseline uses the harness's independent force/heating and
Torch work-reduction reference. Earlier intermediate force/heating-only
measurements were 167.64 ms (matching FP64), 19.05 ms (matching FP32), and
101.87 ms (standard CRK FP64); the final table includes all reductions and
skips the unused matching fractions.

All six evolving fields at the final 43-step FP64 endpoints agree with the
saved baseline to roundoff: largest normalized deviations are 4.1e-16 for the
matching blob, 1.5e-14 for the standard blob, and 1.7e-13 for CompSPH. At
t=0.0215, the first FP32 versus FP64 matching comparison differs by at most
7.04e-5 of each field's maximum magnitude; density gives that maximum.
These short comparisons do not establish long-time FP32 parity.

The matching case's preexisting `relativeEnergyDrift` diagnostic references
the generated setup before loading the native initialization; its initial
FP64 value is -4.529e-8. Energy comparisons here normalize by the recorded
initial value, rather than treating that fixed initialization offset as drift.

Raw JSON, saved states and traces are under
`results/performance_2026-10-21/` (gitignored).

## Reproduction

Use `/home/lu26029/miniconda3/envs/warp/bin/python` from the repository root.
Run configurations sequentially. `--separate` retains independent heating
and Torch compatible-work reductions as a diagnostic reference.

```bash
python scripts/bench_compressible.py --scheme CompSPH --case sod2d --nx 2800 --profile 3 --out /tmp/comp
python scripts/bench_compressible.py --matching results/materialEOS/blob_paper_long_2026-10-21 --precision float32 --profile 3 --out /tmp/matching32
/usr/local/NVIDIA-Nsight-Compute/ncu --profile-from-start off --kernel-name regex:computeCrkSPHAccel_KernelTiled --launch-count 1 --set full -o /tmp/crk_ncu python scripts/bench_compressible.py --warmup 3 --steps 2 --profile 1 --ncu --out /tmp/ncu_run
```

`--ncu` uses the CUDA profiler API to delimit the window and disables Torch's
CUPTI profiler, since NCU and Torch cannot subscribe to CUPTI simultaneously.

## Validation

The final FP32 batch passed **76 tests**: fused raw predictors in 1D/2D/3D
with one and 32 lanes, velocity gradients through automatic differentiation,
padded and empty CSR rows, differentiable reduction fallback, exact native
entropy threshold semantics, compatible-energy conservation, forced list
reruns, graph capture/replay and rebuilds, deterministic replay, existing
energy controls and the native fraction checks. A further **16 FP64 tests**
passed on the target 2D predictor/gradient path, reductions, conservation and
reruns. First-use compilation of the dimension-specialized kernels dominated
validation wall time; it is excluded from warmed timestep measurements.

An initial test exposed a compatibility issue with lightweight legacy
adjacency objects that expose row counts but not CSR offsets. Those and
non-runtime tensor precisions retain the Torch implementation. The final
batch was rerun after that correction. Existing Gather-support warnings in
the Sod tests are unchanged.

## Recorded FP32 comparison

The 4-KH rerun uses the same native initialization and matching physical
policies as the previous FP64 run. Its name in the existing daemon is
`Paper blob nx512 | float32 | performance comparison`, run ID
`2d408604ea2a42a6bd893402ad9693d3`. Live measurements and retained trajectory
recording are enabled; local video and quarter-KH snapshots are also saved.
Quarter-KH times are rounded to representable FP32 values before clipping
timesteps, avoiding a clock stuck just below a nonrepresentable target.

This is a precision/performance comparison against the previous FP64
trajectory, not a new paper reproduction or a precision-default change.
Particle trajectories begin separating during the nonlinear flow. At 0.5 KH,
all-gas cold/dense mass normalized by initial cloud mass is 0.828208 (FP32)
versus 0.827435 (FP64). At 1.5 KH it is 0.648097 versus 0.656610. Finite energy,
positivity and aggregate similarity do not imply pointwise parity.

The run completed **23,339 steps to 4 KH in approximately 376 seconds** with
local video, daemon streaming and trajectory recording enabled: warmed
wall mean **16.15 ms/step**, median **14.11 ms**, integrator mean **14.86 ms**.
The daemon reports completion at t=11.612396240234375 and exposes **235**
retained frames reaching that endpoint. Recording and measurements were
read back through its API. Local artifacts are:

* `results/performance_2026-10-21/matching32_4KH/result.json`
* `results/performance_2026-10-21/matching32_4KH/precision_comparison.json`
* `results/performance_2026-10-21/matching32_4KH/precision_comparison.png`
* `results/performance_2026-10-21/matching32_4KH/video/blob_2026-10-21_15-42-20/trajectory.h5`
* `results/performance_2026-10-21/matching32_4KH/video/blob_2026-10-21_15-42-20/output.mp4`

At 4 KH, all-gas cold/dense mass fractions are **5.6491% FP32 versus 3.1728%
FP64**, a 2.476 percentage-point difference. The largest difference at the
quarter-KH checkpoints is **6.539 percentage points**. The density fields show
similar overall breakup with different filament details; particle density
RMS deviation is 34.3%, so pointwise parity is not claimed. With double
accumulation of saved FP32 fields, relative energy change is **7.10e-6** and
relative momentum change **4.60e-6**. Minimum density is **0.4168**, minimum
internal energy **0.1947**; no divergence was detected. Late remnant mass is
precision-sensitive. Keep FP64 for quantitative comparisons that require
that diagnostic, and choose FP32 explicitly for faster exploratory runs.

![FP64 versus FP32 density and cold-mass comparison](../../results/performance_2026-10-21/matching32_4KH/precision_comparison.png)

Reproduce the recorded comparison (the companion daemon must be running):

```bash
python scripts/bench_compressible.py --matching results/materialEOS/blob_paper_long_2026-10-21 --precision float32 --warmup 100 --steps 30000 --end 11.61239581763877 --video --plot-interval 100 --daemon --out /tmp/matching32_4KH
```

## Remaining measured costs

FP64 still pays heavily for repeated geometric volume, moment and support
calculations. The core CRK volume operator currently has no multi-lane
variant. These remain potential targets for a subsequent pass; this change
does not claim that FP64 is now suitable for real-time 70k-particle use.
No precision or graph threshold default was changed.
