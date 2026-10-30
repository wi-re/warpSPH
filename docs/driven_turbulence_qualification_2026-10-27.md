# Driven 2D compressible turbulence: first qualification, 2026-10-27

Numerical viability is established for this cohort; **stationary training-data
qualification and resolution convergence remain open**. This is a finite-gamma
ideal gas with an explicit relaxation bath, separate from an exact-isothermal
reference. See the [case guide](driven_turbulence.md) for the model, commands and
measurement definitions, and the [roadmap](roadmap/driven_compressible_turbulence.md)
for later work.

## Physical cohort

Eight MFM SSPRK runs reached t=40: three independent 64² seeds and a matched
128² seed at each of two driving strengths. All 885,475 accepted steps and
1,770,950 RHS observations remained positive and finite. The largest absolute
energy-budget residual was 7.89e-06 of initial energy. Mass is conserved;
force work and signed bath exchange are accounted for separately.

Settings: gamma=5/3, u0=0.9, relaxation time=0.2, B7, n_h=4,
solver.cfl=0.1, FP32 physical state and FP64 budgets. OU correlation time=0.5,
knot interval=0.05, integer modes 1≤|k|≤2, solenoidal acceleration,
instantaneous mass-mean acceleration removed. No drag or Mach feedback was
introduced. Inputs 0.3 and 6 are acceleration amplitudes, not requested Mach
numbers. The following means use physical-time quadrature over t=10..40.

| Run | Mean RMS Mach | K block spread | Effective field samples | Scalar screen |
| --- | ---: | ---: | ---: | --- |
| weak64-s1234 | 0.552 | 31.2% | 8.74 | not established |
| weak64-s5678 | 0.656 | 25.8% | 5.07 | not established |
| weak64-s9012 | 0.764 | 17.6% | 4.21 | not established |
| weak128-s1234 | 0.673 | 43.5% | 4.37 | not established |
| strong64-s1234 | 2.385 | 108.7% | 12.25 | not established |
| strong64-s5678 | 1.897 | 43.7% | 18.96 | not established |
| strong64-s9012 | 2.655 | 113.0% | 7.18 | not established |
| strong128-s1234 | 2.875 | 90.2% | 8.77 | not established |

The predeclared screen compares three ten-time-unit blocks, requiring ≤10%
relative spread in Mach, kinetic energy and mass/volume log-density spread,
uncensored correlation estimates, blocks ≥5 integrated correlation times, and
≥10 box crossings. A failed screen does not distinguish secular evolution
from finite-sample uncertainty. Even passing scalar screens would require
spatial, seed and resolution checks before production certification.

Effective samples use the connected velocity-field autocorrelation of the
separately declared adaptive-kernel observation on a 32² grid; they do not
count particles or adjacent saved frames as independent. Keep matched forcing
seed families together across resolution, method and amplitude when constructing
training/validation/test partitions. The record contains
401 snapshots per run, including startup. The coarse weak-run seed range
contains the matched fine-run mean, so a single seed pair cannot isolate
resolution bias. No inertial-range slope or direction of spectral transfer is
claimed.

CIC leaves uncovered cells in strong flows, invalidating those spectra. Those
failures are retained. A separately labelled conservative Wendland C2 filter
(radius=max(support,2 grid spacings), per-particle discrete normalization)
covers the observation grids in this cohort and records filtering loss and
broadening. Comparisons retain 16²/32² grids and keep the two observation operators
separate. Native support-dependent filter radii change with resolution; a common
observation grid alone does not isolate fluid-resolution bias. Measurement
version2 distinguishes exact box-wavelength k²=1 power
from the first rounded radial shell, which also includes diagonal k²=2 modes.

## Matched resolution pairs

These compare seed 1234 over the same physical assessment window. Differences
combine trajectory sampling and numerical resolution; one fine seed cannot
identify a systematic bias or establish convergence.

| Forcing / observable | 64² mean | 128² mean | Symmetric fractional difference |
| --- | ---: | ---: | ---: |
| weak / RMS Mach | 0.5525 | 0.6728 | 19.6% |
| weak / kinetic energy | 0.1533 | 0.2276 | 39.0% |
| strong / RMS Mach | 2.385 | 2.875 | 18.6% |
| strong / kinetic energy | 6.118 | 9.055 | 38.7% |

Full PDF, spectrum and structure-function comparisons are retained in `resolution-comparisons.json`; adaptive-filter comparisons are separately prefixed.

## Failure found and fixed

The cold-gas control (initial sound speed 0.01, acceleration amplitude 6, 12² particles, t=0.02)
completed two accepted steps with positive saved energies while a predictor
had **u=−0.006136**, activating the existing pressure floor. Endpoint checks
alone missed this failure. No pressure/energy floor was added to repair it.

The case now bounds future acceleration across forcing knots and limits the
force-only Euler predictor's thermal-energy loss to at most min(u)/16. The
same GPU control takes 239 steps/478 RHS checks, with minimum raw predictor
u=8.95627×10⁻⁵ and pressure=5.97083×10⁻⁵. Controls cover MFM/MFV,
relaxation/adiabatic gas, an analytic uniform-force temperature case, and a
pulse crossing a forcing knot. This costs smaller steps in cold, strongly
driven gas. The integrator retains its stage sequence; no retry or forcing
retuning is introduced.

The accepted warm trajectories were audited against the new bounds with a
conservative FP32 time-rounding allowance. Both extra restrictions are inactive
throughout all eight runs. The retained warm GPU pilot is also bitwise unchanged.

## Short timestep sensitivity

Sixteen CPU FP64 controls compare CFL 0.1/0.05/0.025/0.0125 at 16² particles, t=0.2,
with both MFM/MFV and both forcing strengths. All raw RHS states stay positive,
and energy-budget residuals remain below 6.2×10⁻¹⁶. Velocity differences
decrease at every refinement. The finest run is a numerical control, not an
analytic reference; this short-window check does not establish long supersonic
or spatial convergence.

| Method / acceleration | Baseline velocity RMS difference from finest |
| --- | ---: |
| MFM-a0.3 | 0.0265% |
| MFM-a6.0 | 0.0338% |
| MFV-a0.3 | 0.0268% |
| MFV-a6.0 | 0.0374% |

Controls take 33–406 accepted steps and 0.8–9.9s each, including observation/progress overhead. Raw fields, inputs, diagnostics and checks are retained in `temporal-refinement/`. Video was omitted for these tiny A/B controls.

## Output, restart and validation

Native checkpoints restore OU knots/generator, accepted budgets and references,
physical next timestep and absolute step count, rejecting incompatible physical
or numerical settings. GPU controls reproduce uninterrupted warm and late
supersonic continuation bit for bit, including MFM and MFV. An endpoint-clipped
segment can validly continue but need not share the step sequence of a longer
uninterrupted run; the guide explains this distinction.

Native trajectory output records conserved/thermal fields, budgets, actual
FP64 timestamps and every intervening forcing knot, with chunk continuity and
coefficient-reconstruction checks. These are observation files; native restart
uses state checkpoints. Output cadence and conditional inputs still need
production-scale checks before training-data deployment.

Validation includes 306 affected FP32 integration checks, 51 focused FP64 checks,
CPU/GPU cold-source controls, and exact restart/forcing-output controls. Runtime
imports resolve for all 544 modules. The full static import audit has a
pre-existing computeAlpha module-path error; an unrelated runner divergence
test assumes FP32 overflow and also fails on unchanged sources in FP64. Neither
is reported as passing.

## Retained evidence and provenance

Raw inputs, resolved settings, source/driver archives, every accepted diagnostic,
RHS minima/counts, snapshots, original outcomes and forcing checkpoints remain
under `.tmp/driven-qualification-20261027`. Field videos replay the same saved
observations with software VisPy and fixed colour limits; no fluid state is
filled or modified for display.

Some original worker outcomes say failed because offline measurement helpers
were added or changed while the workers ran. These administrative flags are
preserved. The sealed source audit excludes exactly the two declared offline
measurement files and verifies all physical Python sources in all four
repositories unchanged. The cohort's common physical source digest is:

`b5e8217f82e2440ca2da22a2ea0f3ad84c3e7e2ce1b9916e078d2aef9a31d8d5`

Runtime fixes were developed in an isolated source copy and applied after the
cohort and its analyses finished. Scheduling included a context-contention
trial and pauses of the same trajectories; reported wall times include those
conditions and observation overhead and are unsuitable for solver speed rankings.
Torch allocator peaks alone do not establish total VRAM capacity.

[Full retained report](../.tmp/driven-qualification-20261027/RESULTS.md) ·
[Histories](../.tmp/driven-qualification-20261027/qualification-history.png) ·
[PDFs and spatial statistics](../.tmp/driven-qualification-20261027/qualification-statistics.png) ·
[Cold control](../.tmp/driven-qualification-20261027/cold-forcing-validation.png)

Next gates: decide whether the desired dataset is developing flow or a
stationary regime; obtain more independent observations and resolution coverage;
qualify any explicit large-scale sink as a distinct physical model if needed;
then measure larger-run capacity and production output locally. Exact-isothermal
physics and GH200 reproduction remain separate work. No HPC job was dispatched.
