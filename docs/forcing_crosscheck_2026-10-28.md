# Forcing and observation cross-checks (2026-10-28)

This continues the [scale-budget investigation](turbulence_scale_budget_2026-10-28.md).
The controlled question is whether apparent energy-budget discrepancies arise
from force evaluation/integration, sparse output, or the observation map before
attributing them to hydrodynamic transport. Production forcing and solver
defaults are unchanged. All evidence is retained under
`.tmp/driven-forcing-crosscheck-20261028/`.

## Independent known-answer force

For a periodic unit box, choose constant density, pressure and specific internal
energy with

\[
 v_y=A\sin(\omega t)\sin(kx),\quad v_x=0,\qquad
 a_y=A\omega\cos(\omega t)\sin(kx),\quad a_x=0.
\]

Here k=2pi times an integer mode divided by box length. The derivation is by
substitution into compressible Euler: divergence is zero; transverse velocity
has no y dependence, so advection vanishes; density and pressure remain uniform.
Thus the momentum source is rho*a and total-energy source is rho*v dot a.
These are the acceleration/energy conventions in [Aluie, equations1–5](https://arxiv.org/html/1012.5877#S2.SS1).
The resulting integrated oracle is

\[
 K(t)=MA^2\sin^2(\omega t)/4,\quad
 P_{in}(t)=MA^2\omega\sin(2\omega t)/4,\quad
 W(t)-W(0)=K(t)-K(0).
\]

Particle displacement in y is A*(1-cos(omega*t))*sin(k*x0)/omega.
The force both supplies and extracts work, returns the continuum solution to
rest after one period, and is independent of the OU generator/interpolation.
It exercises the same actual-mass mean subtraction and body-source paths.
This deliberately simple verification is not a test of all nonlinear Euler
terms or a statistically turbulent flow. It remains on the compressible path.
No viscosity is added to claim a viscous steady shear result.

`OscillatingShear` supports two or three dimensions with distinct coordinate
and velocity-component axes. CPU controls independently check its spatial and
temporal formula, periodicity, Euler residual, filtered power, signed source
work, midpoint convergence and the production compatible source-work finish.
Spheral's `ConstantAcceleration.cc:64–83` adds a body acceleration to DvDt;
no matching oscillating-shear reference was found in the searched functional
Hydro tests or ExternalForce sources. Reference-code agreement is therefore
not claimed. The oracle here is the explicit continuum derivation.

Native controls use A=0.2, period2, mode1, an adiabatic gamma5/3 gas, initially
rho1/u0.9, B7/n_h4 and the existing fresh native stages/CFL. CRK64/128 and
MFM64/MFV64 use the same frozen hydrodynamic runtime as the turbulence cohort.
The independently prescribed force and disabled relaxation bath distinguish
this verification problem from the stochastic turbulence campaign. Video,
progress, raw RHS/accepted positivity, conservation and source hashes are
retained. Native transport can produce shear error/heating; endpoint completion
alone does not qualify the analytic accuracy.

## Observation stress

The original adapter retained mass and momentum but discarded the deposited
second velocity moment. It now also records

\[
 C_{ij}=\sum_p m_p v_{p,i}v_{p,j}W_p/\Delta V,\qquad
 \tau^H_{ij}=C_{ij}-\rho_H v_{H,i}v_{H,j}.
\]

For the continuously filtered momentum equation, the missing momentum term
is -div(tauH). Its periodic kinetic work after the subsequent Gaussian filter
is +mean(grad(vFavre):G(tauH)). This signed term is exposed separately as
`observationStressWork`; `predictedObservedRate` adds it to the previous
prediction. Both original and corrected residuals are retained. It is a
measured kinematic covariance, not a supplied physical viscous stress. The
particle map uses discretely normalized finite stencils, so this correction
does not certify exact native particle momentum closure.

An independent manufactured momentum-flux control checks the sign against
-div(tau), and checks boost invariance. A3D deposition control verifies every
integrated second-moment component and that half the covariance trace equals
the previously retained unresolved kinetic energy.

At ell0.125 on the64² observation grid, t10–40 means are:

| Band | Observation radius | Original residual | Observation stress work | Corrected residual |
|---|---:|---:|---:|---:|
|1–2|0.0625|0.000689945|+0.000354472|0.000335473|
|1–2|0.09375|0.000853809|+0.000520396|0.000333414|
|2–4|0.0625|0.001087030|+0.000966472|0.000120558|
|2–4|0.09375|0.001765582|+0.001646700|0.000118882|

This explains about51%/89% of the narrower-radius residual and nearly removes
the radius dependence at this coarse scale. These are still the original0.1
snapshot cadence; temporal quadrature and pressure/native transport effects
remain. The two CPU analyses cost107.54s/112.00s. No particle trajectory was
changed by this analysis.

## Measurement-only turbulence replay

One CRK128 band1–2 seed1234 replay retains every physical/numerical setting and
the exact original frozen runtime, adds accepted-state instantaneous source
power, and saves accepted-state snapshots at requested0.0125 spacing. The
observer does not restrict timesteps, interpolate states or duplicate them.
Configuration, full accepted histories, initial/final arrays and forcing
checkpoint parity are checked before treating this as a measurement-only
comparison. Both FP32 per-particle work reduced in FP64 and the FP64 dot of
native momentum source with velocity are recorded, distinguishing arithmetic
from quadrature differences.

`analyze_forcingCadence.py` compares dense accepted power and independently
selected physical clocks at0.00625/0.0125/0.025/0.05/0.1/0.2/0.4 against native
work over the same t10–40 window. The finest snapshot budget uses32² cells and
H0.0625, with coarse/fine time sensitivity retained. Spatial sensitivity is
reported separately; changing sampling does not qualify an inertial range or
stationarity. Native compatible half-kick work and continuous instantaneous
power need not agree exactly at finite timestep.

## Completed results

The replay is **bitwise identical** to the baseline: all49,551 accepted rows
for time, kinetic/thermal/total energy, source/bath work and raw minima; every
initial/final array; and all OU parameters, knots and generator state. Only
output/observation changed. There are49,550 steps/99,100 valid RHS evaluations,
zero extra RHS, no invalid accepted state, and maximum relative total-energy
budget residual1.09e-7. Loop cost is560.76s (9.35min),11.317ms/step, compared
with482.996s/9.748ms originally:16.1% extra recording/observation cost. The
tracked supervisor's570.06s also includes process startup/finalization.

Mean native forcing work over t10–40 is0.0317090558. Reintegrating accepted
source powers gives:

| Physical output interval | Work difference versus native |
|---:|---:|
|Every accepted state|−0.000834%|
|0.00625|+0.00200%|
|0.0125|+0.00655%|
|0.025|+0.0218%|
|0.05|+0.0749%|
|0.1|−1.9685%|
|0.2|+0.4719%|
|0.4|+4.2542%|

Thus sparse sampling explains nearly all of the old1.97% work discrepancy.
The remaining dense accepted-power difference is consistent with a small
finite-timestep/discrete-work difference; this experiment does not split those
contributions. CPU-reconstructed FP64 field power and native-device accepted
power agree at the coarse sampling discrepancy's scale. Do not infer this
result from a generic Nyquist argument alone: work is velocity dotted with
force, and its realized time dependence matters.

The3201-state fine budget costs127.12s on one CPU thread. At ell0.125,H0.0625,
including observation stress yields:

| Snapshot interval | Stress-corrected residual |
|---:|---:|
|0.0125|−0.0000155623|
|0.025|−0.0000173927|
|0.05|−0.0000233965|
|0.1|+0.000335467|

The0.1 times selected from the fine output exactly equal the original saved
times. Comparing32² with64² at those identical times changes this residual
by only−5.50e-9; grid differences in individual mean rates are below3.1e-7 at
this scale. This separates the large time-cadence effect from grid effects.
The0.0125-to0.025 residual difference is1.83e-6, so even the small remaining
residual has a temporal uncertainty that matters if interpreted in isolation.
At ell0.0625/0.25 the fine corrected residual is−0.0003131/+0.00000906.
Its remaining scale dependence and sign rule out treating it wholesale as a
unique positive viscous loss. Native transport, pressure representation and
observation/stencil commutation remain in the ledger. The band2–4 trajectory
has only the original coarse time sampling; no finer time claim is made for it.

The known-answer native controls all finish one period with positive finite
RHS and accepted states, zero extra RHS, videos inspected, and stable archived
runtime sources. Errors below are normalized by exact peak RMS velocity
A/sqrt(2) and exact peak K=0.01; peak normalization avoids dividing by the
vanishing answer when the shear returns to rest.

| Method | Steps | Loop wall | ms/step | Maximum velocity error | Maximum work error / peak K |
|---|---:|---:|---:|---:|---:|
|CRK64²|905|11.26s|12.44|0.400%|0.237%|
|CRK128²|1809|20.66s|11.42|0.114%|0.0273%|
|MFM64²|1295|14.13s|10.91|0.467%|0.166%|
|MFV64²|1295|14.01s|10.82|0.467%|0.166%|

Maximum relative total-energy residuals are1.14e-7–2.29e-7. Global momentum
norm stays below1.24e-8. Total mass is unchanged for fixed-mass schemes;
MFV's maximum relative mass drift is1.96e-6. Final net internal-energy increases
are2.14e-5/2.75e-6/1.53e-5/1.72e-5 for CRK64/CRK128/MFM64/MFV64.
Maximum density changes are below1.78e-4 and pressure changes below1.01e-4.
These small numerical shear errors/heating are retained, not repaired by force
normalization, drag, cooling or coefficient tuning. CRK refinement improves
velocity and work errors; this single pair is not an asymptotic order study,
and no MFM/MFV resolution convergence is claimed. Loop timings include the
probe's initialization/plot setup after build; these short, unwarmed runs are
verification costs, not method performance rankings.

![Analytic shear response and forcing-work cadence](/home/lu26029/.codex/worktrees/meshless-boundary-balance/warpSPH/.tmp/driven-forcing-crosscheck-20261028/figures/forcing-crosscheck.png)

## Verdict and next use

No forcing defect was exposed. The previous suspicious residual combined
missing kinetic observation stress with sparse time quadrature; measuring both
substantially closes the large-scale budget on an unchanged trajectory.
The analytic alternate-force check supports the shared source path for CRK,
MFM and MFV at the measured settings. It does not validate chaotic statistics,
all schemes, arbitrary Mach numbers or3D evolution.

Keep the OU turbulence model and numerical defaults. For the next budget-focused
campaign, explicitly enable `--instantaneous-power` and choose/test fine output
cadence;0.0125 is supported here, not a universal qualified default. Preserve
second moments in2D/3D observations. The measured negative transfer at selected
scales persists, but finite-box condensate behavior and stationary/ensemble
qualification remain open. Physical transport requires its own declared
coefficients and separate tests; do not tune a viscosity to this residual.

Reproduction tools: `qualify_drivenTurbulence.py --instantaneous-power`,
`analyze_forcingCadence.py`, `qualify_prescribedShear.py`,
`analyze_turbulenceBudget.py`, and `plot_forcingCrosscheck.py`.
The analytic Euler/source controls and observation tests pass33 CPU checks.
Manifests, exact sources, input hashes, parity results, scalar/scale series,
time/grid sensitivity and video audits are retained in the evidence directory.

Video titles inherit the old parent-checkout Git discovery label5d4f41d-dirty;
that label is not the frozen runtime identity. The physical source inventory
matches the baseline exactly, and all2106 files pinned across this audit's
manifests pass the postflight hash check. Source archives and manifests are the
provenance authority. The replay and all four shear videos were inspected;
peak shear is smooth and periodic, while replay frames retain the expected
large-scale vortical structures. Auto-scaled density colors in the shear videos
show variations of order1e-4 or smaller, not order-unity density changes.
