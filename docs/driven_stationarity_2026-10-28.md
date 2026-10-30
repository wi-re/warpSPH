# Long CRK128 stationarity investigation (2026-10-28)

The campaign requests three independent forcing seeds with the source-pinned verified model to t=160; the table lists successfully analyzed trajectories. Primary blocks are40–80,80–120 and120–160. No physical/numerical retuning or trajectory retries.

| Seed | Mean K40–80 | Mean K80–120 | Mean K120–160 | Scalar screen | Loop minutes | ms/step |
|---|---:|---:|---:|---|---:|---:|
|1234|0.252125|0.154144|0.297649|fail|36.58|10.99|
|5678|0.205911|0.230500|0.251176|fail|35.82|10.92|
|9012|0.284329|0.278524|0.218092|fail|36.32|10.94|

All three trajectories complete numerically, but none passes the predeclared
stationarity screen. The longer record does **not** support a shared monotonic
energy-growth trend: seed1234 falls then rebounds, seed5678 rises between block
means, and seed9012 falls. The observed behavior is consistent with strong,
slow, box-scale intermittency; it does not prove either statistical stationarity
or persistent secular drift. Stationary training-data qualification remains open.

Uncertainty estimates are conditional moving-block bootstraps using blocks of five estimated correlation times, only with at least four blocks. Drift is retained. Censored correlations and inadequate block counts are explicitly unavailable; no independent-particle error bars or trend removal. Three seeds leave limited ensemble precision.

![Histories](/home/lu26029/.codex/worktrees/meshless-boundary-balance/warpSPH/.tmp/driven-stationarity-20261028/analysis/ensemble.png)

Evidence: `/home/lu26029/.codex/worktrees/meshless-boundary-balance/warpSPH/.tmp/driven-stationarity-20261028`. Per-run summary JSON includes native/field window moments, conditional uncertainty, PDF total-variation changes, spectra, field ACF and energy budgets. Native/fine snapshots and forcing history remain archived. The ensemble plot, PDF/spectrum plot and five representative decoded frames per run have been visually reviewed. The inspected flow has box-scale coherent structures that reorient and change amplitude; no blow-up or boundary-ejection pattern is visible. This is representative inspection, not a review of every video frame.

All successful trajectories passed raw/accepted positivity, finite endpoint, no-extra-RHS and source/provenance guards. Any skipped or failed run remains in the supervisor/analysis status; it must not be counted as stationary.


## What changed our interpretation

The three-seed means in the successive40-unit windows are0.24745,0.22106 and
0.25564, with between-seed sample standard deviations0.03942,0.06273 and0.03997.
These are three finite realizations, not a precise ensemble confidence interval.
The ensemble block spread is14.3%; the first-to-last difference is only0.00818.
The single-seed short-window increase therefore does not establish an
unbounded accumulation process. Conversely, three seeds cannot exclude a
slower shared drift outside the measured duration.

| Seed | K block spread | Whole-window field ACF tau | Effective field samples | Box-energy fractions in the three windows |
|---|---:|---:|---:|---|
|1234|61.2%|7.82|7.67|88.2%,83.1%,87.1%|
|5678|19.7%|9.83|6.10|88.6%,85.0%,87.9%|
|9012|25.4%|5.35|11.21|85.9%,89.2%,87.1%|

The exact box-wavelength modes are |n|=1, separately measured from the rounded
first shell that also contains diagonal modes. Their fraction passes the10%
block-spread screen for every seed, while their **absolute** energy fails it
for every seed. A fairly stable spectral fraction can coexist with strongly
varying total amplitude. The spectra retain broadly similar shapes, but that
is not evidence for an inertial-range flux plateau or a uniquely inverse-cascade
origin: this forcing band directly includes box modes.

Longer time coverage reveals longer memory. Compared with the old approximately
six effective field samples over t10–40, t40–160 still gives only6.1–11.2 under
the connected Eulerian field-ACF estimator. Correlation estimates themselves
are conditional and can be distorted by drift. None is censored here, but the
40-unit blocks are not long relative to every estimate; notably five times the
whole-window field tau for seed5678 is49.2. K-window bootstrap intervals are
withheld in7 of9 windows because fewer than four five-tau blocks fit. Available
intervals are retained as conditional uncertainty, not evidence of stationarity.

Adjacent-window density/pressure PDF total-variation distances are approximately
3–12%, depending on seed and weighting. These descriptive histogram differences
are not significance tests with independent-particle sample counts. Density
spread, pressure spread, Mach and spectral amplitude also vary, so a claim based
on the box-energy fraction alone would miss relevant changes.

![PDF and spectrum comparison](/home/lu26029/.codex/worktrees/meshless-boundary-balance/warpSPH/.tmp/driven-stationarity-20261028/analysis/distributions-spectra.png)

## Forcing and energy accounting

Force variance was independently reconstructed from the full saved coefficient
history, using the exact integral of squared piecewise-linear coefficients.
The continuum acceleration RMS in the three late windows stays0.290–0.305,
while mechanical power varies much more:

| Seed | Mean input power40–80 | Mean input power80–120 | Mean input power120–160 |
|---|---:|---:|---:|
|1234|0.01655|0.01503|0.04389|
|5678|0.02298|0.02126|0.03175|
|9012|0.03892|0.02376|0.02317|

Mechanical input depends on velocity–acceleration alignment; the forcing
amplitude parameter is not a constant-power prescription. Input also becomes
negative in some five-unit averages. These observations support investigating
the coupled large-scale response and work correlations; they do not establish
that forcing amplitude drift causes the energy excursions. The three saved
coefficient histories are distinct (late coefficient cosines0.0027–0.0232),
and seed1234's retained coefficient prefix matches the old trajectory exactly.
Small cross-correlations are a provenance diagnostic, not a proof of stochastic
independence.

The bath holds mean thermal energy close to0.9 (late block means0.903–0.908),
while K varies substantially. Native total-energy rate minus signed forcing
and bath exchange is below3e-9 in magnitude in every40-unit window. Accepted
instantaneous-power quadrature differs from native mechanical work by at most
2.36e-6 in mean rate; the bath difference reaches8.97e-6. These small finite-step/
source-accounting differences cannot be used to assign a physical viscosity.
Clean total-energy conservation does not establish a stationary kinetic-energy
balance or accuracy of the scheme's numerical transport.

The weak-driving label also does not mean strictly subsonic at every instant:
maximum mass RMS Mach is1.077/1.040/1.065 for seeds1234/5678/9012. This campaign
retains those transonic excursions; it uses no target-Mach feedback.

## Numerical completion, provenance and cost

All3 endpoints are exactly160. There are595,572 accepted steps and1,191,144 raw
RHS checks, zero invalid RHS states, no extra RHS, positive accepted states,
unchanged total mass, and maximum relative energy-budget residual1.59e-7.
The frozen runtime inventory and initial arrays match the baseline for every
seed. Seed1234 matches49,539 accepted rows bitwise before the old t40 endpoint
restriction. Both finite workers have terminal successful statuses and their
matching processes have exited; completion is established by results/logs,
not process disappearance alone.

All2108 pinned executable/source files pass postflight hashes. All4803 analyzed
snapshot hashes match their recorded evidence. The38,403 fine snapshots
(12,801 per seed), exact inputs, forcing histories, source archives, stage/
accepted histories and videos remain retained. Fixed observation radius is
0.0625 on32² cells; this is a declared measurement filter, not a new spatial
convergence study. Local native supports can exceed that radius in rare states;
no deconvolution or resolved-inertial-range claim follows from these spectra.

Total physical-loop cost is108.72min at10.92–10.99ms/step, approximately3% below
the11.32ms/step short dense-output replay; different flow histories and startup
amortization prevent treating this as an algorithmic speedup. The three CPU
statistical analyses cost56.58/56.24/54.37s (2.79min total) and overlapped the
following GPU runs. Six statistical/failure-guard controls pass, and the retained
401-state analysis control reproduces the earlier K blocks and field ACF.
`postflight-review.json` and the reusable `audit_drivenStationarity.py` preserve
the final integrity, force-variance and visual-review evidence.

## Verdict

The model is numerically robust over the measured duration and produces
intermittent, coherent large-scale dynamics. A common ongoing energy-growth
trend is not supported by these three records; stationary, ensemble-precise
statistics are still unqualified. These trajectories can be described as
finite-window evolving-flow data, with seed/time/forcing state retained, rather
than independent stationary snapshots.

The most direct next analysis can use the archived fine states to examine
box-mode work and transfer during high/low-energy episodes. A defined large-scale
sink, fixed physical transport, changed scale separation or3D evolution would
be separate physical/numerical studies. Do not silently add drag or prescribe
constant power to make this screen pass. No further physical run is launched
by this report.

The completion heartbeat is paused after final review. The reusable companion
received a documentation-only repair for matching the running plan marker and
writing roadmap-relative links; the exact executed source remains archived.
No numerical analysis or physical trajectory changed.


A CPU-only rendering of the applied seed1234 forcing at actual retained states
near t40–44 is saved in `.tmp/driven-stationarity-20261028/forcing-view-s1234-t40/`.
It uses the full saved Fourier history and actual particle mass-mean subtraction,
fixed arrow/color scales, and no per-frame normalization. Static snapshots and
an acceleration-only video show the prescribed field, not fluid velocity.


## Persistent late pattern: t120–160, seed1234

The user's observation is supported by a separate CPU-only decomposition. The
late flow, especially after t130, is dominated by broad alternating flow bands.
An optimized periodic translation plus one amplitude factor fits t160 to t130
with vector correlation0.9203, explaining84.69% of mean-subtracted, filtered
velocity variance. The remaining15.31% includes internal deformation and
changing residual modes. Across t130–160, that reference's aligned correlation
has minimum0.758 and median0.913. This describes observed Eulerian similarity,
not a chaos or laminarity test. Mass-weighted bulk speed stays below5.85e-8;
the apparent drifting is pattern motion, not net bulk momentum.

Reference choice matters: t120 is a transition with only51.1% instantaneous
box-mode fraction, so its match to later fields is much weaker. All five
references120/130/140/150/160 are retained in `reference-sensitivity.json` to
avoid selecting only the most similar pair. The late-window mean box fraction
remains87.1%. The residual-speed and vorticity panels show internal evolution
that a full-speed color plot makes less prominent; its energetic weight is small.
The original native movie displays40 simulated units in roughly one video
second. A new fixed-scale, slower t120–160 clip shows full speed, speed excluding
exact box modes and observed vorticity at approximately2 simulation units per
video second.

A known subgrid periodic shift and gain is recovered to roundoff, and an
independent orthogonal-mode perturbation gives the expected unexplained energy
fraction. Some real BFGS fits retain precision-loss flags at gradient norms
below2.34e-8; flags are preserved. The reported t130→160 fit succeeds. One
reference and one common amplitude are limited descriptors; high alignment
can conceal residual dynamics, and decorrelation caused by translation can
exaggerate structural diversity for a training dataset. Use aligned residuals
and mode-resolved diagnostics when assessing that diversity.

The forcing directly drives box-scale modes, with only six independent wave
vectors and no scale interval larger than the box to develop. A coherent
few-mode response is plausible. Published2D studies also find wandering
long-lived coherent flows ([Parfenyev](https://arxiv.org/abs/2311.03065)) and,
under different compressible/intermediate-scale forcing conditions, interacting
vortices and shocks ([Falkovich and Kritsuk](https://arxiv.org/abs/1709.00814)).
Those models motivate interpretation, not a quantitative reference for this
finite-gamma bath/implicit-transport setup. Direct box forcing means the visual
pattern cannot by itself be assigned to an inverse cascade. Remaining limits
include numerical transport and inadequate forcing/domain scale separation.
This is a useful verification case with limited structural diversity; richer
training data would require a separately qualified choice of forcing scales,
resolution/transport, or a3D pilot.

Evidence: `.tmp/driven-stationarity-20261028/late-pattern-s1234/` and
`late-pattern-s1234-caption-review/`. Initial and corrected-caption analyses
have identical numerical series/controls; the caption now correctly states
coordinates are wrapped into[0,1), not translated by half a box. No new physical
trajectory or CUDA allocation was used.
