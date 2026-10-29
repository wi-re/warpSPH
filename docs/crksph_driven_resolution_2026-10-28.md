# CRKSPH driven turbulence: 192²/256² cost and mean trend

2026-10-28. Plan index: [driven compressible turbulence and multiphysics](../PLANS.md).

**Both requested runs reached t=40 with valid checked states.** The local GPU
cost is modest for this scope: 13.50 min at 192² and 23.47 min at 256². The
finite-window means do not approach a clear limiting value across these four
resolutions, and none passes the existing stationarity screen. This is numerical
viability and cost evidence, not stationary or resolution-converged data.

## Matched setup and sources

Reuse the frozen source/runtime and physical inputs of the
[overnight seven-scheme study](driven_turbulence_overnight_analysis_2026-10-28.md).
Only the particle resolution and output root change. The unit periodic 2D box
starts from rest: seed1234, rho0=1, gamma=5/3, initial sound speed1/u0=0.9;
OU acceleration0.3, correlation time0.5, knot spacing0.05, solenoidal integer
modes1–2, actual mass-mean acceleration removal and relaxation bath time0.2.
This is finite-gamma compressible flow with cooling, not exact-isothermal flow.

Native CRKSPH midpoint, CFL0.15, B7/n_h4, support/dissipation and compatible-work
choices remain unchanged. There is no graph-mode change, derivative reuse,
CFL tuning, initial mass calibration, added drag or forcing feedback. The
192² run preceded256²; both used the same device and recording/observation
harness as the retained64²/128² runs. Those smaller runs are reused as evidence,
not silently rerun or counted as new realizations.

All four runs share the recorded source digest
`e834516dc49825e6774840b64153ea2d78a6c43f47a541065e0b5b3e30beea8a`
and 803 identical raw forcing knots/parameters, covering the full observation
window. All 2,107 source-pinned files are unchanged after the new jobs. As in
the earlier campaign, the Git label discovered from the enclosing primary
checkout is not the frozen physical revision: use the original8832040 launch
snapshot, full package archives and file hashes together.

## Cost and means

Means use physical-time quadrature over t=10..40, with the same mass-weighted
Mach and particle K definitions as before. The sparse observations agree with
independently integrated every-step histories to within 1.9e-6 in Mach and
1.4e-6 in K. Costs include accepted/RHS observation, snapshots, progress and
software-rendered recording, excluding the independent ten-step warmup/setup.
They are whole-harness costs, not exclusive GPU-kernel time.

| Resolution | Particles | Steps | Run wall to t=40 | ms/step | Mean RMS Mach | Mean K | Mean mass log-density std |
|---|---:|---:|---:|---:|---:|---:|---:|
| 64² | 4,096 | 23,583 | 3.76 min | 9.58 | 0.6790 | 0.2326 | 0.1870 |
| 128² | 16,384 | 49,550 | 8.05 min | 9.75 | 0.6735 | 0.2230 | 0.2236 |
| 192² | 36,864 | 71,326 | 13.50 min | 11.36 | 0.6296 | 0.1977 | 0.1903 |
| 256² | 65,536 | 102,606 | 23.47 min | 13.73 | 0.7162 | 0.2549 | 0.2244 |

256² costs **2.92× the 128² runtime**, with 4× the particles. That combines **2.07x as many
steps** and **1.41x the cost per step**. Relative to192² it costs 1.74×. This
measures the actual adaptive-step workload for these flows; do not extrapolate
a general scaling exponent, matched-error efficiency or GH200 timing from it.
The two new loops total 36.97 min; serial CPU analyses add 40 s and 70 s, respectively.

The queue waited for an unrelated ACSPH probe before192² and never stopped or
modified it. Prelaunch checks found no competing compute users when each run
started. A 30 s read-only monitor recorded 73 physical-run samples, with no query
errors or other compute PID. Monitoring begins near t=4.58 for192² and before
GPU initialization for256²; this is sampled load context, not isolation from
every transient graphics/system activity. The monitor is outside solver timing
instrumentation and performs no GPU computation.

![Cost and mean trend](../.tmp/crksph-driven-resolution-20261028/comparison-final/resolution.png)

## What the mean tends toward

There is **no identified asymptotic mean** in these runs. Over the fixed window,
mean RMS Mach spans 0.630–0.716 and K spans 0.1977–0.2549. The192²→256² shift is
+12.9% in Mach and+25.3% in K using the signed symmetric difference
2(b-a)/(a+b), rather than a steadily shrinking resolution correction.

The first ten-unit measurement block looks encouraging in isolation:
192²/256² K is0.2081/0.2063 and Mach0.6469/0.6490. The later blocks separate:

| Resolution | K, t=10–20 | K, t=20–30 | K, t=30–40 | Relative block spread | Effective field samples |
|---|---:|---:|---:|---:|---:|
| 64² | 0.1566 | 0.2604 | 0.2807 | 53.4% | 6.43 |
| 128² | 0.1913 | 0.2165 | 0.2612 | 31.3% | 6.06 |
| 192² | 0.2081 | 0.1722 | 0.2127 | 20.5% | 5.74 |
| 256² | 0.2063 | 0.2850 | 0.2736 | 30.9% | 6.18 |

Block spread is(max-min)/mean of the three block means; these are not confidence
intervals. At256², blocks2/3 have higher K than block1, while192² has a lower
middle block. One common forcing seed with only about6 effective field samples
per window cannot separate systematic resolution bias from finite-window
chaotic/stochastic variation. The plots expose that variation; no Richardson
fit or error bar implying independent ensembles is applied. All four fail the
predeclared <=10% scalar stationarity screen in the measured scope.

The density statistic is non-monotonic too. Using the common16² observation
filter (constant radiusH=0.125 in all four runs), the adjacent128²→192² and
192²→256² velocity-spectrum symmetric L1 differences over shells1–4 are 13.9%
and 26.1%; mass log-density PDF total variations are 6.8% and 7.3%. The effect
persists under a matched broad observation filter and is not explained solely
by unequal deposition radii. Box-wavelength modes still carry about 85–89% of
the32² filtered velocity spectral power. No inertial-range or transfer-direction
claim is made.

![Time evolution](../.tmp/crksph-driven-resolution-20261028/comparison-final/evolution.png)

## Numerical and recording checks

The new192²/256² runs take 173,932 accepted steps and 347,864 fresh RHS evaluations,
with no compatible-work replay required and no invalid/nonpositive checked u/P.
Minimum raw u is0.66830/0.61922 and minimum P is0.23510/0.16970. Maximum accepted
relative energy-budget residuals are3.69e-8/7.19e-8 of native initial energy;
these do not measure turbulence accuracy. Both endpoints are exactly40.

At192², the initial FP32 diagnostic Q energy sums to0.9000000134110451 while
native primitive sum(m*u) is0.9000000432133657. This exposes a non-power-of-two
mass/energy reconstruction rounding offset of3.31e-8 relatively. The first
comparison-harness assertion incorrectly used diagnostic Q as the SPH budget
reference and failed; that attempt is retained in
[initial-accounting-failure.json](../.tmp/crksph-driven-resolution-20261028/comparison/initial-accounting-failure.json).
The final analyzer uses the native primitive initial energy and records both
numbers. The residual now reproduces the case diagnostic without changing any
tolerance, physical source or trajectory. No measured means are changed by this
accounting correction.

Peak Torch allocations are366MiB at 192² and639MiB at 256²; reserved peaks are
404/676MiB. Sampled process memory including the CUDA runtime was about1.2–1.5GiB.
These are separate measurements and not a whole-device capacity forecast. Both
cases fit well within this GPU's available memory.

Both VisPy recordings completed: 73/104 decoded frames,960×480 at50fps. Mid-run
and final saved frames were inspected; endpoint headers show t=40 and the
expected particle counts. Density/speed plots are intact with coherent flow
structures. Frame colour limits vary, so image colour alone is not a quantitative
resolution comparison. The endpoint dt=0 in the header is endpoint clipping,
not a stalled trajectory. [Video metadata and hashes](../.tmp/crksph-driven-resolution-20261028/video-audit.json)
retain the two recording paths and final frame references.

## Verdict and next step

**Keep as measured:** CRKSPH runs through this192²/256² weak-flow family with
small measured budget error, positive checked states and a 23.5 min 256² cost.
192² is a useful cost midpoint; it is not already an asymptotic solution.
**Open:** a stationary or resolution-converged mean. Further evidence should
use longer physical-time sampling and independent seeds on a small selected
resolution cohort, retaining the declared physical model. These runs can still
be useful labelled transient/chaotic realizations; they do not by themselves
qualify stationary reference training data. No additional runs are started by
this analysis and no solver default is promoted.

## Evidence and reproduction

[Campaign manifest](../.tmp/crksph-driven-resolution-20261028/manifest.json),
[completed status](../.tmp/crksph-driven-resolution-20261028/status.json),
[GPU/progress observations](../.tmp/crksph-driven-resolution-20261028/gpu-progress.jsonl)
and [source postflight](../.tmp/crksph-driven-resolution-20261028/source-postflight.json)
retain execution, inputs, sources and outcomes. Each run keeps its initial/final
NPZ,401 snapshots, every accepted diagnostic, forcing checkpoint, source archive,
resolved configuration, driver copy and recording. Individual analyses remain
under analysis/weak{192,256}-CRKSPH-s1234; their top-level index is overwritten by
the last single-run invocation, so the merged comparison reads individual files.

The [four-resolution analyzer](../scripts/analyze_crksphResolution.py) produces
[comparison.json](../.tmp/crksph-driven-resolution-20261028/comparison-final/comparison.json),
[merged reports](../.tmp/crksph-driven-resolution-20261028/comparison-final/individual-reports.json)
and PNG/PDF figures. It checks native integration/CFL, actual endpoints, source/
forcing identity, raw-history budgets and GPU sample context. Reproduce into a
fresh output root, without running simulations:

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/warpsph-mpl \
/home/lu26029/miniconda3/envs/warp/bin/python -u scripts/analyze_crksphResolution.py \
  --baseline /home/lu26029/dev/warpSPH/.tmp/overnight-acsph-driven-2026-10-27/turbulence \
  --campaign .tmp/crksph-driven-resolution-20261028 \
  --output .tmp/crksph-driven-resolution-20261028/reproduction
```
