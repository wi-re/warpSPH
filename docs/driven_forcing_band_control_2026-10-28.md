# CRKSPH128² forcing-band control

2026-10-28. [Plan](roadmap/driven_compressible_turbulence.md),
[preceding periodicity audit](driven_forcing_audit_2026-10-28.md).

**Removing direct box-scale acceleration reduces the flow amplitude but does
not remove box-scale velocity dominance.** The new modes2–4 run reaches t=40
with valid checked states. Mean K falls36.4% and mean mechanical input power
61.7%, while exact |n|=1 modes still carry76.6% of the matched filtered velocity
energy, compared with84.9% in the retained modes1–2 run. Both fail the existing
stationarity screen. CUDA checks also support the earlier finding that periodic
forcing evaluation errors are small.

## Controlled setup and scheduling

The user authorized this one experiment after the other session's ACSPH run.
That queue and its export/comparison jobs completed successfully at10:56UTC;
its supervisor and child were absent and the GPU had no compute users before
this experiment allocated CUDA. The source-pinned serial supervisor started
11:16:59UTC, ran the small CUDA field check, one physical trajectory, then CPU
analysis. No other-session process was interrupted, modified or messaged.

Reuse the retained overnight CRKSPH128² baseline, without rerunning it. The new
run uses exactly the same frozen package sources: launch snapshot8832040 and
source inventory digest
`e834516dc49825e6774840b64153ea2d78a6c43f47a541065e0b5b3e30beea8a`.
All2,111 manifest-pinned files remain unchanged. As in the earlier campaign,
the parent-checkout Git label rendered in the video is not the frozen source
revision; the archived files and their hashes establish provenance.

All nine initial fluid arrays match bitwise. Exact input comparison finds only
`kMin:1→2`, `kMax:2→4`, and the output destination changed. The copied probe
adds these two CLI parameters and points at the frozen runtime; its small patch
is retained. There is no physical package source edit.

Both runs use the unit periodic2D box at128², initial rest/rho1, gamma5/3,
sound speed1/u0=0.9, accelerationRMS0.3, OU correlation time0.5 and interval0.05,
solenoidal projection and actual mass-mean acceleration removal. The relaxation
bath has time0.2; B7/n_h4, native fresh midpoint/CFL0.15, maxDt0.01, positivity
observation, eager execution, progress and VisPy recording are unchanged.
No drag, physical viscosity, constant-power feedback or additional seed was added.

The real Fourier basis grows from6 to20 independent modes. The same seed1234
with a different mode count changes the mapping of random draws: these are not
identical coefficient histories. The acceleration ensemble normalization stays
0.3; the realized acceleration amplitude need not match exactly. Changing the
forcing length also changes ratios involving the flow turnover time at the
fixed OU clock. This is one forcing-configuration control, not an isolated
measurement of each of those effects.

## Means, input work and cost

Physical-time means use t=10..40. The force RMS below is the square root of
mean continuum acceleration variance, evaluated by exact segment integration.
Mechanical power comes from the native accepted compatible-work budget,
including its half-kick semantics. It is not inferred from sparse snapshots.

| Quantity | Modes1–2 baseline | Modes2–4 control |
|---|---:|---:|
| Mean mass RMS Mach | 0.67345 | 0.53651 |
| Mean particle K | 0.22300 | 0.14178 |
| Mean mass log-density std | 0.22361 | 0.19134 |
| Mean mass internal energy | 0.90593 | 0.90217 |
| Mean mechanical input power | 0.031709 | 0.012146 |
| Mean signed bath power | -0.029640 | -0.010859 |
| Continuum acceleration RMS | 0.29769 | 0.28694 |
| Direct box-mode fraction of force variance | 36.28% | 0% |
| Exact box-mode filtered velocity-energy fraction | 84.89% | 76.60% |
| Estimated effective field samples | 6.06 | 8.79 |
| Scalar stationarity screen | fail | fail |
| Accepted steps | 49,550 | 46,334 |
| Observed RHS states | 99,100 | 92,668 |
| Physical loop wall | 8.050min | 7.807min |
| ms/step | 9.748 | 10.110 |
| Peak Torch allocated memory | 183.69MiB | 183.80MiB |

The mathematical mode-norm notation in the table refers to integer box wave
numbers. The box-mode fraction uses a32² volume-velocity field, with the same
physical filter radius0.0625 and no empty cells in either run. It is the mean
instantaneous fraction at exact n²=1, not the first rounded radial shell, which
also includes diagonal modes. This filtered kinematic velocity-energy fraction
is distinct from the full mass-weighted particle kinetic-energy budget.

The realized acceleration RMS differs by3.6%, while mean mechanical work
differs by61.7%. This reinforces the distinction between prescribed acceleration
and flow-dependent input power. It does not isolate the cause: forcing spectrum,
flow correlations and numerical dissipation all enter the response.

The new run has6.5% fewer steps; each step costs3.7% more, and total wall time
is3.0% lower. Twenty modes require three bounded Fourier chunks instead of one,
but there is no isolated forcing profiler here to assign the timing difference
to that operation. These are one-run whole-harness costs, including recording,
RHS/accepted-state observation and snapshots, excluding independent warmup.
The CUDA field check took approximately5s and the single-run CPU analysis20s.
Fifteen GPU-user samples from t≈6.93 through queue completion found no unrelated
compute process. This is sampled load evidence, not a guarantee for every instant.

## Block behavior and qualification

Dense-history block means and source-work differences are:

| Band | Interval | Mean K | Mean input power | Mean box-mode fraction |
|---|---|---:|---:|---:|
| 1–2 | 10–20 | 0.19128 | 0.021304 | 84.34% |
| 1–2 | 20–30 | 0.21651 | 0.031909 | 81.17% |
| 1–2 | 30–40 | 0.26120 | 0.041914 | 89.15% |
| 2–4 | 10–20 | 0.13501 | 0.011681 | 78.99% |
| 2–4 | 20–30 | 0.13815 | 0.009331 | 77.11% |
| 2–4 | 30–40 | 0.15219 | 0.015427 | 73.70% |

The new K block spread falls from31.35% to12.12%, but its scalar correlation
time is about2.06, making ten-time-unit blocks shorter than the existing
five-correlation-time requirement. The new mass/volume density-spread screens
also fail. Its box-fraction screen alone passes. Smaller K fluctuations do not
qualify this window as a stationary training-data realization.

All92,668 new RHS states are finite with positive checked thermodynamics, all
accepted-state guards pass, no extra RHS evaluations occur, and t=40 is reached
exactly. Minimum raw RHS u=0.66328 and pressure=0.15363. Maximum relative energy
budget residual is9.60e-8, compared with1.09e-7 in the baseline; mass remains1.
These checks do not supply an analytic turbulent-flow error or a calibrated
physical dissipation/entropy rate. No such physical transport is specified.

![Matched flow and input-power evolution](/home/lu26029/.codex/worktrees/meshless-boundary-balance/warpSPH/.tmp/driven-forcing-band-20261028/comparison/evolution.png)

## Periodic CUDA checks and recorded flow

The preflight compares the unchanged CUDA evaluator against an independent
NumPy FP64 Fourier sum and analytic spatial Jacobian. It tests both bands and
both precisions at ten times, signed seams, random positions, coordinates
spanning24 periods, and64²..256² quadrature grids. Torch2.13.0+cu130 used
highest FP32 matmul precision and TF32 off on the RTX PRO6000 Blackwell.
All four band/precision combinations pass the predeclared thresholds.

For the new band in FP32, maximum opposite-edge acceleration discrepancy is
3.65e-7 and Jacobian discrepancy5.25e-6. Independent in-box value error is
3.09e-7; the deliberately unwrapped-coordinate test reaches1.68e-5.
The retained CRKSPH256 endpoint, evaluated on CUDA with its original forcing,
has maximum field error6.05e-6 and instantaneous input-power error-1.24e-8.
FP64 discrepancies remain around machine rounding. No substantial periodic
forcing discontinuity is demonstrated on the production backend.

The native recording has48 H.264 frames at960×480/50fps. Initial, middle
(t≈21.19) and final(t=40) frames decode; the middle/final frames and independent
matched-color endpoint plots were inspected. Large coherent density/speed
structures remain, with finer modulation. There is no visible collapse or
obvious seam break in the inspected views. The video is also copied under
`images/`; both files are byte-identical and preserved.

![Force variance and filtered velocity spectrum](/home/lu26029/.codex/worktrees/meshless-boundary-balance/warpSPH/.tmp/driven-forcing-band-20261028/comparison/spectra.png)

![Matched endpoint density and speed](/home/lu26029/.codex/worktrees/meshless-boundary-balance/warpSPH/.tmp/driven-forcing-band-20261028/comparison/endpoints.png)

## Interpretation and remaining work

The band change matters substantially to the flow amplitude and energy balance.
Direct forcing of the box modes is not necessary for their dominance here:
the modes2–4 acceleration has exactly zero |n|=1 coefficients, yet the flow
develops strong |n|=1 velocity content. This is consistent with coupled transfer
to larger spatial scales in this2D compressible system. It is not a measured
inverse energy cascade; no nonlinear spectral transfer/flux decomposition was
performed. Compression, pressure work and numerical effects remain part of the
unresolved mechanism.

This control supports investigating the2D finite-box energy balance rather
than treating the observed common structure as a demonstrated forcing seam bug.
It does not establish a converged resolution trend, universal stationary mean
or solver-independent result: there is one scheme, resolution and stochastic
realization per band. The forcing length, mode population and realization differ,
and there is no prescribed physical viscosity or large-scale momentum drag.
Further discrimination needs a transfer budget and independently justified
physical/statistical controls. No additional trajectories or default changes
are included in this authorization.

## Reproduction and retained diagnostic failures

Raw campaign: `.tmp/driven-forcing-band-20261028/`. It contains exact planned
and actual inputs, baseline hashes, frozen manifest, driver-only patch, upstream
completion receipt, supervisor status/logs, CUDA metrics/thresholds, raw states,
dense histories, forcing checkpoint, analysis, video audit and GPU load samples.
The [offline comparison helper](../scripts/analyze_drivenForcingBand.py) checks
input changes, bitwise initialization, native integration, source digest,
endpoints and all pinned hashes before generating `comparison/comparison.json`
and figures. No physical rerun was needed.

Two offline harness assumptions were corrected and retained as evidence:

- A quadrature agreement threshold failed because the preceding0.005-spacing
  trapezoid rule overestimates the convex squared-linear forcing variance by
  2.88364e-5 (about0.033%). The exact segment integral and independently derived
  trapezoid error agree; `quadrature-check.json` records both. Neither trajectories
  nor mechanical work budgets depend on this offline choice.
- The video selector initially assumed a single `output.mp4`; it found the
  recording and its standard identical `images/` copy. The initial helper is
  preserved and the final selector checks both hashes before decoding the
  canonical recording. No video or physical output was replaced.
