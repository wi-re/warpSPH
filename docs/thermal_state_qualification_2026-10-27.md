# Prescribed thermal states: repair and CPU qualification — 2026-10-27

The measured prescribed-state failures are repaired. The five ordinary
compressible SPH schemes now retain prescribed internal energy after
initialization and accepted integration; PESPH entropy solves its coupled
closure rather than treating a local entropy conversion as a temperature
constraint. Meshless driven particles can prescribe internal energy directly
as geometry changes. Accepted projection exchanges have explicit restartable
counters. **Physical conducting walls and convection remain unqualified.**

Baseline is `2ee802b`, following the [audit](thermal_boundary_audit_2026-10-27.md)
and [small controls](thermal_boundary_toys_2026-10-27.md). This work stays in
`codex/meshless-boundary-balance`. Turbulence campaigns remain deferred; all
new measurements use CPU while the independent ACSPH GPU process continues.

## Cause and implementation

An RHS sees a stage copy. Overwriting that copy's u does not constrain the
accepted combination assembled from the initial state and derivatives. CRKSPH
and CompSPH additionally replace integrated thermal energy with compatible
pair work. RT also installed preliminary band values before its density/support
solve and initial pressure reconstruction, which subsequently replaced them.

The shared accepted-state hook now runs after successful compatible work and
volume sources, before accepted switch history. It enforces prescribed values,
refreshes the appropriate EOS, and makes public total energy consistent with
accepted u and velocity. A replay returns before committing a projection.
The runner enforces the initial constraint before diagnostics and derives an
adaptive startup timestep from that corrected state. This can lower the step
for a hotter target but cannot raise the initializer's chosen ceiling; RT's
2/3 startup factor is retained. RT restores its primitive
band values after constructing its initial EOS state. No CFL factor, tableau,
support prescription, limiter or hydrodynamic force formula is changed.

Kidder's case-specific post-step overwrite is removed: the native lifecycle
owns the same task for all such cases. This also avoids applying the ordinary
mass-density EOS over PESPH's accepted pressure-volume closure. Thermal
trajectories now include u, support and the projection budget, including after
checkpoint continuation; native state checkpoints retain the counter too.
Ordinary `loadTrajectory` callers should request `extraFields=('internalEnergies',
'supports')` as usual; the new budget field is discovered when present.

These hooks activate for explicit `internalEnergies` Dirichlet callbacks.
Without a thermal prescription, the ordinary-SPH path is a no-op. Meshless
projection accounting activates only for driven particles. Existing material
EOS restrictions and PESPH's single-gas restrictions remain in force.

## Entropy constraint: solve the quantity actually prescribed

[Hopkins (2013), Eqs. 19–22](https://arxiv.org/html/1206.5006#S2.SS2.SSS3)
defines pressure-entropy weights and their coupled kernel sum. In the current
implementation, writing w=m A^(1/gamma), y=Ww gives

    u = (w/m) y^(gamma-1) / (gamma-1).

Thus setting A=(gamma-1) u_target rho_target^(1-gamma) once need not recover the
requested u after refreshing y. A thermal callback now constrains u and solves
only the selected entropy values; unconstrained A values stay bitwise fixed.
Auxiliary prescribed density is not an independent pressure-volume constraint.
Density-only prescriptions retain their previous local conversion.

The positive multiplicative iteration is

    w_next = w (u_target/u_current)^(1/gamma).

The damping follows uniform scaling: scaling every w by s scales u by s^gamma.
Every residual uses the actual stored-A roundtrip, native pressure sum and EOS.
Relative corrections use FP64 intermediates even with FP32 operators/storage;
large absolute-log subtraction in the first attempt stalled an FP32 gamma=3,
cooling control. That failure is retained. The tolerance was **not loosened**:
`max(1e-11, 8*eps(storage_dtype))`, with an explicit failure after 128 iterations.
Targets must be positive. This is a local boundary inversion, not a published
new force formulation or a proven globally convergent solver for arbitrary
kernels, state contrasts and gamma.

Spheral `116c71f2ea559d569456ba694194e5a6de3c75f7` was checked in
`Boundary/ConstantBoundary.cc`, `SPH/PSPH.cc`, and
`tests/functional/Hydro/RayleighTaylor/RT-2d_Hopkins.py`. Its RT bands become
fixed ghost states whose fields are restored, rather than integrated fluid
particles. Its PSPH evolves thermal energy and does not supply a matching
entropy-boundary inversion. Our fix is an explicit driven-particle extension;
no matched Spheral trajectory or reference-code equivalence is claimed.
[Frontiere et al. §3.3](https://arxiv.org/html/1605.00725#S3.SS3) distinguishes
interim thermal predictors from the final compatible-work update. Accordingly,
external prescribed-state resets are applied after that update, preserving its
hydro pair-work calculation.

## What the budget measures

`state.boundaryProjectionBudget` is an optional FP64 two-entry tensor:

1. Change of total thermal energy during accepted projection.
2. Change of total kinetic energy during accepted projection.

It is zeroed when defining a new initial state, added only at acceptance, copied
through native stage lineage, and retained by checkpoint and thermal-trajectory
I/O. Source work/cooling keeps its separate `sourceEnergyBudget`.
For PESPH entropy, changing constrained A can change neighboring u. The counter
therefore measures energy over the **whole state**, not only selected particles.
Meshless counters use authoritative Q rather than retained public stage fields.

This is a **projection-exchange counter**, not a physical wall heat-flux
integral or a complete mechanical boundary-reaction ledger. Continuous forcing,
reservoir fluxes and imposed acceleration constraints may require further work
terms. Uniform manufactured thermostat and source-coupling controls close their
known discrete budgets; that does not certify the full RT or moving-wall budget.

## Meshless fixed-u prescription

`DrivenParticles` now accepts the optional callback `specificInternalEnergy`.
Its existing primitive callback supplies velocity and may return `None` for
pressure when u is prescribed. The constrained conserved energy is
`E=m*u_target + |m*v|^2/(2m)`; pressure follows the current geometric volume.
Without that callback, the existing `P*V/(gamma-1)+KE` prescription is unchanged.
The driver also applies its prescribed initial trajectory/conserved state
before the first diagnostic. Invalid explicit u values are rejected rather than
clipped. This API prescribes specific internal energy; mapping it to a physical
temperature requires defined caloric units/heat capacity.

Native MFM/MFV tests use a smooth periodic deformation with changing geometry,
prescribed velocity and u, under SSPRK, Hancock and KDK. All six hold the target
and account for the accepted conserved-energy projection. An algebraic volume
control confirms fixed u under volume changes, whereas the old pressure mode
continues to produce u proportional to volume as intended.

## Quantitative controls

The primary current-source toy cohort repeats the original 92 controls in each
precision, with 2,640 integrated steps across both. The full conductivity section
is unchanged and remains a characterization of artificial diffusion, not a
physical-diffusion qualification. No execution errors occur. Source inventories
are unchanged during each measured cohort.

| Control | Baseline | Repaired |
|---|---|---|
| Ordinary-SPH ramp from u=1 to 1.02 | Accepted relative error 1.960784%, despite exact stage values | FP64 accepted error zero; FP32 <=1.87e-8 across RK2/RK4, both timesteps and resolutions |
| Ordinary-SPH uniform ramp heat, total mass 2 | Missing accepted heat 0.04 | Accepted projection supplies 0.04; FP64 caloric residual <=4.45e-16 |
| PESPH entropy, smooth-density ramp | Relative u error 2.61e-4 at nx32 | Maximum accepted error across PESPH E/A toy controls <=6.54e-16 FP64 and <=3.71e-7 FP32 |
| Meshless fixed-geometry thermostat | Accepted state held, initial state not generally reapplied | Initial error included in the all-method bound below; accepted error <=1.12e-11 FP64 and <=1.36e-7 FP32 |

Across all toy initialization/RHS/accepted observations, maximum relative error
is <=1.12e-11 FP64 and <=7.16e-7 FP32. The five ordinary-SPH FP32 caloric
residuals are <=3.82e-8; PESPH's maximum is <=4.89e-7. These small differences
include representable target and initial-state rounding. Tested states remain
positive. The original physical-diffusion gaps and resolution/amplitude-dependent
artificial-conductivity measurements remain unchanged.

Focused tests cover heating/cooling with gamma=1.4, 5/3 and 3, partial bands,
nonlocal entropy energy changes, explicit nonconvergence, source cooling plus
uniform acceleration, per-material EOS refresh for the existing CRKSPH/CompSPH/
Monaghan material paths, forced native replay, state checkpoint continuation,
thermal trajectory reconstruction, and native meshless temporal modes.

Validation records:

- Broad CPU suite: **440 passed, 12 GPU skips, 10 CUDA-only deselections**,
  115.57 s. This preceded the final startup/output integration refinements.
- Final runner/catalogue/thermal integration suite: **170 passed**, 7 CUDA-only
  deselections, 26.41 s.
- Final thermal/PESPH/source checks: **81 FP32 checks** included in the
  170-test suite above, and **81 passed FP64**, 11.66 s. These include the
  conservative startup-ceiling and hot-state timestep controls.
- FP64 before/after parity: nine saved-field controls bitwise equal against
  `2ee802b`: source-free Sod for five ordinary SPH schemes and PESPH E/A,
  plus tiny MFM/MFV driven-source regression controls. Those tiny source checks
  do not resume a turbulence campaign. No-prescription behavior is unchanged
  in the measured controls; this is not a universal proof over every case.

## Short coupled checks and recordings

Twenty-three successful FP64 coupled controls cover RT, Kidder and Yee for all
seven SPH variants, plus MFM/MFV pressure-reservoir Kidder. They are startup
checks with positive raw u/P, not full case accuracy or convergence campaigns.
Ordinary-SPH selected-band u errors are at roundoff; PESPH entropy stays within
its 1e-11 relative solve tolerance. The meshless Kidder controls check pressure,
not an independent analytic density/u prescription.

Two RT pilots at nx32 (2,432 particles) record density, u and speed with software
VisPy/EGL rendering. CRKSPH completes 80 steps to t=0.0533333, minimum u=1.05329,
P=0.842634, maximum selected-band u error zero. PESPH entropy completes 80 steps
to t=0.08, minimum u=1.05329, P=0.867112, maximum target error 9.992e-12.
Their actual endpoints differ; they are not a matched-physical-time accuracy
comparison. Both videos contain 11 decoded frames; final frames were inspected
and show intact prescribed bands and small, coherent interior perturbations.
The remaining controls run ten steps and intentionally omit video as a small
A/B-style loop.

Recorded wall times include setup, diagnostics, software video and encoding:
25.09 s / 313.68 ms per step for CRKSPH and 37.53 s / 469.11 ms for PESPH entropy.
They are not solver throughput rankings. Small coupled-check timings and
ms/step are retained per result; the entropy constraint adds iterative pressure
sums and stopping reads only where it is explicitly used. No GPU throughput or
large-N entropy-boundary cost is qualified. The repaired toy sweeps cost about
15 s per precision including their timed setup/observations, comparable to the
previous ~14 s startup-inclusive controls; this is not an isolated performance
measurement. The source-free path adds no tensor work for this feature.

## Verdict and next gap

**Keep** the prescribed-state lifecycle, native-precision entropy residual
check, direct meshless-u prescription, and explicit projection exchange. Their
measured CPU contracts pass the small controls without changing no-prescription
saved fields or native integrator/force choices. GPU eager equivalence, graph
execution and long thermal-flow validation remain open.

This closes the measured temperature-state handling failure. It does not add
physical Fourier conduction, dimensional heat capacities/transport coefficients,
conducting/insulated solid-wall fluxes, calibrated physical viscosity, a complete
boundary-reaction budget, or the MFM/MFV RT adapter. In particular, the old RT
meshless refusal is reworded to identify that missing case adapter rather than
incorrectly saying conserved source support does not exist.

The compressible comparison matrix has not been rerun; its earlier thermal-band
case results remain historical baselines.

Next is a separate physical-transport change: define caloric units and physical
kappa/chi and viscosity; validate heat diffusion and insulated exchange in 1D,
then hot/cold plates with boundary heat-flux balance, and only then a gravity-
supported conductive layer and convection onset. Keep turbulence campaigns
back while these thermal foundations are being investigated; the GPU can be
used later when it is available and useful for the next qualification scope.

Independent follow-up [reference scaffolding](thermal_transport_scaffolding.md)
now supplies analytic periodic/insulated/plate solutions, caloric conventions
and signed heat budgets for those future transport controls (22 CPU oracle
checks). It does not add physical transport or extend solver qualification.

## Reproduction and raw evidence

Use the [toy probe](../scripts/probe_thermalBoundaries.py) and commands in the
[small-controls report](thermal_boundary_toys_2026-10-27.md), with fresh roots.
New regression tests are [thermal state](../tests/test_thermalDirichlet.py) and
[source/material coupling](../tests/test_thermalSources.py).

`.tmp/thermal-repair-20261027/` retains the baseline source archive, parity
inputs/arrays, current `release-toys-{f64,f32}` and earlier development cohorts,
source hashes/archives, copied drivers, fields and histories. Coupled evidence
uses the two recordings in `coupled-final` and all 21 `coupled-short-final`
rows, summarized in `coupled-release-summary.json`. Earlier development
cohorts remain separate.
Earlier pilot schema refusals (explicit None/read-only tLimit) and loader-harness
attempts remain recorded and are excluded from numerical results. The initial
FP32 entropy-solve failure and its source are preserved under `initial-failure`.
`validation.json` identifies selected cohorts, final source inventory and logs.
