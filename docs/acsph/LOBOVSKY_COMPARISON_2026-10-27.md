# Lobovsky: exact-IC ACSPH comparison, 2026-10-27

Plan: [ACSPH_PLAN.md](../../ACSPH_PLAN.md). User: start the warpSPH comparison
with the supplied dynGPUSPH Lobovsky preset. This is a startup diagnosis,
not a completed 2 s experimental impact validation.

## Comparison contract

Reference: `dynGPUSPH@2dbc27a`, `tosend/dambreak_Lobovsky_400_2L`, CUDA FP64
base executable from [the reproducible preset recipe](REFERENCE_PRESETS.md).
warpSPH runs on the same GPU in FP64. The two engines run sequentially.

The adapter imports the reference's binary initial state directly: 320,000
fluid and 32,176 wall particles, dx=0.00075 m, constant rho=1000 kg/m³,
fluid mass 180 kg / sum(m/rho)=0.18 m² per unit breadth. The initial x/v/p/rho/m
arrays agree **bitwise**, including walls and the supplied hydrostatic pressure
(maximum 2935.6425 Pa). No mass calibration, relaxation, jitter, lattice
resampling or physical wall-particle motion is applied. warpSPH additionally
constructs 32,176 internal mDBC ghost rows, excluded from physical-mass and
field comparisons. They are not extra fluid mass or reference particles.

Common settings: Wendland C2, compact radius 0.003 m, no kernel normalization
correction, g=(0,−9.81), free slip, physical effective viscosity 0.0001608375
m²/s, 2L pressure smoothing, SSPRK3, nominal CFL_tau=0.75, dt/dtau=2,
log velocity target −8, cap 200 and every-iteration stopping checks. Existing
D1/D2/D3 and hard position-row wall bounce options are enabled to follow the
reference choices. The per-run controls do not change production defaults.

Timesteps replay the first three recorded reference steps exactly:
1.2e−5, 1.44e−5, 1.728e−5 s, ending at 4.368e−5 s. This isolates other
solver differences from adaptive timestep selection. Reference snapshot,
source-module hashes, effective coefficients, controls and audit receipts are
retained by the comparison script.

The mDBC geometry uses nominal tank planes [0,0]–[1.61,1.4] from the supplied
rectangles, without moving/calibrating the supplied particles. The reference
uses particle-derived normals/pair wall closures. Equal particle coordinates
do not make these operators equivalent. Other remaining differences include:

- warpSPH starts with BDF1; the reference seeds duplicate histories and
  dt_prev=1e−5 and uses BDF2 from the start.
- The reference halves dtau for the first two pseudo-iterations. The local
  iteration uses a constant dtau within a real step.
- Point-implicit formulas differ (including the reference's integer `(3/2)`);
  the local scheme retains its existing implementation.
- The reference has a pressure-residual gate/truncated fallback, velocity JST
  and an acoustic damper. These are not present in the local spatial residual.
- Shifting implementations differ; the reference transports p/v to shifted
  positions, while the local finalize applies its existing displacement/BDF
  correction without that same p/v remapping.

## Confirmed pressure-acceleration units defect

The first physical-density baseline exposed an extra density division in
`modules/artificialCompressible/residual.py`. The shared
`computePressureForceSurfaceAware` already returns
`−sum_j V_j (p_i+p_j) grad W_ij / rho_i`; both its ordinary and tiled kernels
explicitly divide by query density. ACSPH divided that acceleration by rho_i
again, weakening pressure acceleration by 1000 in this preset. Its previous
comment incorrectly described the shared helper as a force-density operator.
The unilateral branch combines the same helper and had the same error.

De Courcy et al. Eq. (25), checked in the local primary paper, contains one
factor 1/rho_i. The reference computes `irhoA * PGti` once
(`HashFunctions_cuda.cuh` momentum residual). The correction removes only the
second division, in both ACSPH branches; the shared pressure/boundary modules
are unchanged. rho0=1 tests and runs cannot expose this error.

A regression scales rho, particle mass and pressure together by 1000 while
holding geometry/volume and p/rho fixed: physical pressure acceleration must
stay invariant. It failed by a factor of 1000 before the correction in both
ordinary and unilateral branches and passes afterwards in FP32 and FP64.
The nominal three-step baseline is retained before this fix.

The FP64 bring-up also required converting `sphKernel_xi`'s host Warp scalar
to a Python float before Torch tensor arithmetic; otherwise the step setup
raised `div(Tensor, float64)` rather than running. This is a type conversion,
not a coefficient/model change.

## Separate smoothing-length convention mismatch

The core's Wendland-C2 xi is 2.821384729 in 2D. The current ACSPH parameter
setup uses h=compact_radius/xi=0.001063307662 m, while the paper/reference's
Wendland h is half the compact radius, 0.0015 m. The paper specifies h/dx=2;
the reference kernel's support test is distance<2h.

Thus equal nominal constants do not give equal physical pseudo-coefficients.
At the first dt (outside the reference's two warm-up iterations):

| Quantity | Local nominal | Reference |
|---|---:|---:|
| beta | 132.9134578 | 187.5 |
| k1=beta² | 17666 (approximately) | 35156.25 |
| k2 | 0.0141327898 | 0.028125 |

A second **probe-only** variant multiplies local CFL_tau and k2Factor by
xi/2. This maps beta and k2 exactly to the reference's main-iteration values
without changing the physical kernel, viscosity, timesteps or production
parameter conversion. It does not reproduce warm-up, wall, history or
point-implicit differences. The trial is evidence for a later parameter-model
correction, not a new recommended CFL/default.

## Execution cost limitation

CUDA graphs were requested for every local run. They fall back to eager:
`computeViscosityBoundaryVelocities` performs a host `bool((kinds==1).any())`
inside capture. The first real error is retained using
`WARPSPH_CUDAGRAPH_TRACE=1`; the final capture-invalidated exception alone
would obscure the cause. Shared boundary code is left unchanged.

Local run costs include setup/compile/video; integrator step times and graph
fallback status are retained separately. No hardware-speed conclusion should
be drawn from the compile-heavy first baseline.

## Reproduction and receipts

```bash
conda activate warp
python scripts/acsph_reference/compare_lobovsky.py \
  --reference .tmp/dyngpusph-reference-2026-10-27/runs/lobovsky-startup \
  --out .tmp/lobovsky-local-nominal --steps 3
# Same corrected solver, isolate the h convention:
python scripts/acsph_reference/compare_lobovsky.py \
  --reference .tmp/dyngpusph-reference-2026-10-27/runs/lobovsky-startup \
  --out .tmp/lobovsky-local-mapped --steps 3 --coefficients mapped
```

Local evidence root: `.tmp/acsph-lobovsky-2026-10-27/`. Physical snapshots use
original row order so errors pair the same material particles, rather than
silently nearest-neighbor rematching displaced particles. Initial/final mass
invariance is bookkeeping; it does not establish geometric volume conservation.
The reference's faulty `VOL` and one-step-late terminal probe timestamp are
not used as physical-volume or time truth.

## Results at the common three-step endpoint

At t=4.368e−5 s, with exactly equal physical ICs and timestep sequence:

| Variant | p_max [Pa] | Mean p [Pa] | Pressure RMS error [Pa] | KE [J / unit breadth] | Peak speed [m/s] |
|---|---:|---:|---:|---:|---:|
| dynGPUSPH | 2936.05 | 1343.37 | — | 8.09813e−6 | 0.00258609 |
| Original local pressure units | 9259.89 | 1503.00 | 507.89 | 1.64701e−5 | 0.00043981 |
| Remove extra density division | 3221.54 | 1393.05 | 110.35 | 1.55470e−5 | 0.00300981 |
| Above + coefficient mapping | 3399.46 | 1368.09 | 115.92 | 1.16220e−5 | 0.00209779 |
| Above + source startup history | 3275.98 | 1385.26 | 108.39 | 9.99021e−6 | 0.00225577 |

The original corner pressure spike largely disappears after fixing units.
Pressure RMS error drops from 17.26% to 3.75% of rho g H. At the middle
column, most of the corrected pressure profile follows the reference; the
largest remaining discrepancies sit near the bed/contact regions and at the
free vertical face. The coefficient/history controls improve KE agreement,
but they do **not** uniformly improve pressure: mapped coefficients alone
slightly worsen pressure RMS and p_max. Do not choose a default by the lowest
single number in this startup table.

The source-history control seeds x/v at both earlier levels with the initial
state and dt_prev=1e−5, just as upstream, and uses BDF2 immediately. Its
first-step alpha_t is 128787.8788 s⁻¹, versus 83333.3333 s⁻¹ for the local
BDF1 startup. This is an explicit **fixture option**, not a change to normal
warpSPH startup. Add `--history source` to reproduce that control.

Relative to reference, the last control has +23.36% KE, −12.77% peak speed,
+11.58% peak pressure, pressure RMS 3.68% of rho g H, and velocity RMS error
0.000124717 m/s. The absolute energies/velocities here are tiny: the fluid is
only just leaving rest. This does not measure developed dam-break dissipation
or establish a full-flow ranking. RMS particle-position difference is about
0.00314 dx, including the different shifting implementations.

All physical particle counts and masses remain constant. Every local solve
and each reference solve used all 200 iterations. Local final log velocity
metrics are −7.973 (original units), −6.703 (density fix), −6.589 (mapped) and
−6.489 (mapped + source history), against −8. These are cap-limited states,
not converged equivalence checks. The source-history control also produces
p_min=−46.82 Pa versus reference +0.0747 Pa; it is not a uniform pressure
improvement. The reference's own criterion/fallback is also different; a nominally equal epsilon is not a common residual norm.

| Run | Real steps | Wall seconds | Wall ms/step | Integrator step ms |
|---|---:|---:|---:|---|
| Reference | 3 | 25.395 | 8464.8 | native event 8035 / 8259 / 8287 |
| Original local units | 3 | 136.284 | 45427.9 | 62166 / 35310 / 35358 |
| Density fix | 3 | 107.223 | 35740.9 | see result.json |
| Fix + coefficients | 3 | 118.648 | 39549.5 | see result.json |
| Fix + coefficients + history | 3 | 138.857 | 46285.5 | see result.json |

Every run has a video; initial/final fields and the generated comparison
figure were visually checked. Native event timing and local integrator timing
have different inclusion boundaries. First-use compilation, capture attempts,
video and setup contribute to wall cost. These timings establish the current
practical cost, not a controlled GPU kernel performance ratio.

Validation: full FP32 ACSPH test file **66 passed**; FP64 parameter/scaling
selection **3 passed**. The two density-scaling cases fail before the fix and
pass afterwards. No shared boundary code changed. The physical rho fix has no
effect at rho0=1, so it does not by itself resolve the older density-one
matrix failures. The h conversion and BDF startup defaults are unchanged.

## Initial verdict and next discriminator

**Keep the pressure-units correction** and FP64 host-scalar conversion.
**Retain coefficient/history mappings as diagnostics**, without promoting
them as defaults. A full 2 s Lobovsky / experimental sensor-pressure
comparison is not started: the initial three steps already expose a remaining
wall/driver gap and exhausted iteration budgets. Next compare the actual
initial spatial pressure/divergence residuals and the rows controlling the
stopping norm, before porting the reference's extra dissipation or extending
an expensive campaign. Keep paper, reference-code quirks and local choices
separate.

The analysis can be regenerated with `analyze_lobovsky.py --reference
<reference-3> --local 'label=<local-run>' ... --out <analysis-dir>`. It checks
exact initial arrays and equal endpoint times, then produces `summary.json`
and `comparison.png` with material-row pressure profiles, KE, peak speed and
pressure-error comparisons. Artifacts are under the evidence root's
`analysis/`; videos are in each local run's `recording/.../output.mp4`.

## Follow-up: actual spatial operators and RK stages

User follow-up: the ICs are identical, so continue tracing the remaining
implementation differences. This audit keeps the density fix, mapped beta/k2
and source-seeded history as the baseline. All changes below are **probe-only**
options of `compare_lobovsky.py`; production solver and shared boundary code
are unchanged.

`audit_reference.py` copies the qualified isolated build, adds observations
around `ResidualHash3`, and records the exact instrumentation diff, source
hashes and new executable hash. It saves the first real step's iterations
0, 1 and 199, all three stages, in both sorted and original UID order as
appropriate. The local `--audit-stages` hook saves matching stage inputs and
unscaled spatial residuals. It uses eager evaluation explicitly because host
snapshots cannot be captured; the ordinary fixture already falls back to eager.
The reference observation changes final floating-point rounding very slightly:
max position 1.11e-16 m, velocity 2.35e-17 m/s, pressure 6.31e-12 Pa versus the
qualified uninstrumented run. It is **not bitwise identical**. The local audited
baseline is bitwise identical in x/v/p to its prior unaudited counterpart.

### Wall-neighbour pressure smoothing breaks linear consistency

At the first stage, both implementations have **identical fluid x/v/p**, with
lambda agreeing to 1.11e-15. The reference pressure residual is at most
8.11e-8 Pa/s: its AC-2L flux annihilates the initially linear hydrostatic field.
The local residual instead reaches **7.8449e6 Pa/s**, RMS 202178 Pa/s over the
fluid, concentrated at solid walls. At this stage v=0 everywhere, so this is
pure pressure smoothing, not velocity divergence or physical acceleration.
Reference k2 is twice the mapped local k2 in these first two iterations due
to the reference's half-dtau warmup; rescaling that negligible reference
residual cannot explain the discrepancy.

The reason is the wall gradient in Eq. (33). Local `_biLaplacian` computes
`gradP_wall` over the wall's own support and uses `gradP_i + gradP_wall` on
fluid-wall pairs. Outer solid rows without fluid support have pressure zero;
although the fluid's own wall neighbours carry the correct hydrostatic
pressure, those neighbours' *gradient estimates* reach the unloaded outer
rows. The resulting wall gradients contaminate an otherwise linear fluid
field. The reference explicitly uses **2 gradP_i** for a solid neighbour
(`sph.cuh`, `getContrib2`), and zero wall gradients are never read in that flux.
The paper's Eqs. (33)-(34) require linear cancellation; they do not prescribe
this particular solid-gradient extension. This is a boundary extension choice,
not a change to the interior Antuono operator.

The probe implements that difference without editing a shared kernel:

```
D_source_wall = D_current + Divergence_BoundaryToFluid_Difference(gradP)
```

The added term is precisely `sum_wall (gradP_wall - gradP_i).gradW V_wall`.
On the flat bottom strip (initial y < 4 dx), initial max |rP| falls from
7.8449e6 to **2.35e-8 Pa/s**. A small left/top contact-corner residual remains
(max 64326 Pa/s over the whole fluid): local initial wall-pressure attenuation
is already active, whereas the reference closes wall pressures before its
first surface detection. Only eight fluid gradients differ visibly at that
corner; this is separate from the false bed diffusion. Initial momentum
residual RMS difference is only 0.00221 m/s², and is 2.26e-13 away from walls.

Two focused invariants pass in both FP32 and FP64: a linear fluid pressure
next to unloaded outer solid layers is preserved by the proposed flux, and
its fluid-only operator remains bitwise unchanged. These tests exercise the
opt-in probe; they do not change the production default.

### The reference divergence damper is real, but does not close the gap alone

The reference adds `beta*h * sum_fluid (divF_i + divF_j) gradW V_j` to the
momentum residual, where `divF` itself sums **fluid neighbours only**. This is
separate from its physical viscosity and momentum JST terms. A local replay
of that operator on the actual reference RK states agrees to max 1.59e-12
m/s² (RMS 1.75e-14) at the first iteration's second stage, and to max 5.33e-15
at iteration 199's third stage. Thus the transcription is checked on identical
fields, rather than inferred from different trajectories.

At the first iteration's second stage, the reference horizontal component
RMS is 58.83 m/s² for pressure, 30.54 for the acoustic damper, 0.0557 for
physical viscosity and 0.1314 for momentum JST. By iteration 199's last stage,
the damper's vector-component RMS is 0.0515 m/s²; physical viscosity is
0.000550 and JST 0.00116. The large initial contribution is transient.
The paper discusses divergence damping in Section 2, but this exact
fluid-only operator is an extra in the executable source, absent from its
published Eq. (25). It is retained as an explicit diagnostic option.

Adding only this damper to the wall-flux trial changes final pressure RMS
75.92 -> 78.55 Pa and KE/reference 1.1657 -> 1.1701. It does **not** fix the
remaining difference, despite its sizeable first-stage contribution.

### RK source placement differs structurally

Both use the same SSPRK3 a/b tableau. Local code scales each source stage's
slope by `1 + c_next*dtau*alpha_BDF` (last stage uses c=1). The reference
instead scales each **destination stage row** by `1 + b_destination*dtau/dt`,
and each final weight by `1 + b_source*dtau/dt`. Its C++ `(3/2)` evaluates to
one, so this is not the paper's BDF coefficient. At dtau/dt=1/2 the reference
reciprocal factors are 12/13, 12/13 and 3/4. Local first-step slope factors
with source history are 22/39, 44/61 and 22/39. These are different iteration
maps even before considering warmup, wall closure or momentum damping.

`--rk-treatment source` reproduces the actual placement and integer-division
quirk only inside the comparison harness. Agreement with a reference-code
quirk is not grounds to promote it as the numerical model. The first two
half-dtau pseudo-iterations, stopping norm/fallback, wall pair velocity
closure and final shifting remain separately declared differences.

### Frozen-field replay localizes a second wall difference

The wall velocity is not equivalent between implementations. In the
reference, the extrapolated fluid velocity is sampled **at the physical wall
particle**, then reflected with a normal derived from wall-particle geometry.
Local mDBC samples at a **reflected ghost location** and uses its SDF normal.
Both are called free-slip, but they produce different wall velocities even
when the fluid fields are identical.

Replaying the exact first-step reference iteration 199 / last-stage fields:

| Operator on identical native fields | Max difference | Component RMS difference |
|---|---:|---:|
| Pressure acceleration, local kernel vs native | 2.45e-12 m/s² | 2.57e-13 m/s² |
| Divergence with local mDBC closure vs native | 0.3505 s⁻¹ | 0.003556 s⁻¹ |
| Divergence with native wall values fed to local kernel | 2.43e-17 s⁻¹ | 9.08e-19 s⁻¹ |
| Independently computed physical-wall Shepard/mirror | 9.71e-17 s⁻¹ | 1.02e-18 s⁻¹ |

The native divergence itself has max 0.04039 s⁻¹ / RMS 0.002599 s⁻¹. Thus the
wall-closure difference is larger than the residual being solved, rather than
a tiny rounding discrepancy. At the first iteration's second stage the same
local closure error reaches 2.912 s⁻¹, while the physical-wall implementation
agrees to 1.78e-15. With beta=187.5 and rho=1000, the last-stage difference
alone corresponds to **12.32 MPa/s maximum / 125 kPa/s RMS** in the pressure
residual. This is a genuinely different incompressibility boundary operator.

`--wall-velocity source` is scoped to the static Lobovsky fixture; it computes
the wall-location Shepard value and particle normal with existing operations.
It does not change mDBC or general moving-wall behavior. Both frozen replay
fixtures use only one pseudo-iteration to obtain the initialized system;
their advanced states are not convergence/trajectory evidence. Each has video.

### Shifting is a separate discrepancy

A second observational build records `ShiftHash` input/output. All first-step
RK/operator dumps are **bitwise equal** to the first observational build, so
adding shift observations leaves those evaluations unchanged. During the first
reference step, the shift has RMS magnitude **7.28e-7 m** and max **6.01e-6 m
(0.00801 dx)**. The local RK-aligned trial's Michel shift has RMS 3.78e-12 m /
max 1.49e-9 m (0.00000199 dx). The reference's dt-independent surface-cohesion
shift, wall collision shift and differing Michel coefficient are not the same
particle regularization as the local implementation.

Native remapping in this first shift changes pressure by RMS 0.0574 Pa / max
1.877 Pa, and velocity by RMS magnitude 5.41e-9 m/s / max 3.68e-7 m/s.
Therefore it is not responsible for the large initial wall pressure-flux
error or the first-step wall-velocity discrepancy. Its geometry/history effects
still matter to subsequent steps. A reference-matching shift and a physically
qualified shifting model are separate decisions; neither is silently promoted.

## Follow-up outcome: startup parity after matching the wall operator

The completed static wall-velocity trial closes the remaining gap by roughly
two orders of magnitude. Successive rows add **one choice at a time**, retaining
exact physical ICs, viscosity, mass, FP64 precision and reference timesteps.
All rows below finish three real steps at t=4.368e-5 s, with 200 iterations
per real step. The starting row already has the pressure-unit fix and explicit
h/coefficient/history mappings from the earlier section.

| Successive probe | Pressure RMS vs reference [Pa] | KE/reference | Peak speed [m/s] | p_min / p_max [Pa] |
|---|---:|---:|---:|---:|
| Native reference | — | 1 | 0.00258609 | 0.07473 / 2936.04685 |
| Local mapped/history baseline | 108.387 | 1.233645 | 0.00225577 | -46.817 / 3275.979 |
| + fluid-side wall diffusion gradient | 75.917 | 1.165742 | 0.00215215 | -45.017 / 2936.113 |
| + native divergence damper | 78.551 | 1.170089 | 0.00211714 | -57.644 / 2936.127 |
| + native RK factor placement | 106.470 | 1.015709 | 0.00178854 | -24.585 / 2936.204 |
| + native physical-wall velocity sampling | **1.084** | **1.001891** | **0.00259236** | **0.07056 / 2936.04563** |

Final KE differs by +0.1891%, peak speed +0.2423%, mean pressure +0.3775 Pa,
and velocity RMS error is 6.33e-7 m/s. Pressure RMS is 0.03684% of rho g H.
The first-step comparison is still closer: KE +0.00682%, peak speed +0.0632%,
mean pressure -0.00248 Pa, pressure RMS 0.05770 Pa and velocity RMS 1.06e-8 m/s.
The first-step pressure RMS is nearly the reference's own shift-remapping RMS
(0.05741 Pa); the much larger 108 Pa error is explained without changing ICs.

The wall-gradient change is independently justified by linear consistency.
The velocity change instead matches a *different discrete wall operator*:
wall-point Shepard and ghost-point Shepard are distinct choices. The staged
trajectory controls establish their combined contribution with RK/damper
alignment; this is not a factorial attribution of every interaction. The
near-exact first step and same-field replays give stronger localization than
matching total KE alone. The reference's extra damping, integer RK quirk and
dt-independent shift remain source-code choices, not automatically recommended
production models.

All runs retain 320000 fluid particles / 180 kg / 0.18 m² per unit breadth.
Initial array identity, common endpoint times and finite fields are checked by
`analyze_lobovsky.py`. Every video was decoded and the main trajectory final frames inspected;
the early geometry barely changes, so material-row pressure curves and direct
operator residuals are the useful diagnostics. The new wall-flux invariant
checks pass **2 FP32 + 2 FP64**, CPU. No production solver or shared boundary
implementation changed in this follow-up.

| Follow-up run | Real steps | Wall seconds | Wall ms/step |
|---|---:|---:|---:|
| Instrumented native | 3 | 32.035 | 10678.2 |
| Audited local baseline | 3 | 122.621 | 40873.6 |
| + wall-gradient extension | 3 | 127.079 | 42359.6 |
| + acoustic damper | 3 | 162.694 | 54231.3 |
| + source RK placement | 3 | 159.108 | 53035.9 |
| + source wall-velocity closure | 3 | 163.007 | 54335.8 |
| Native shift observation | 1 | 11.573 | 11572.6 |
| Frozen closure-value replay fixture (1 pseudo-iteration) | 1 | 4.695 | 4694.9 |
| Frozen closure-implementation replay fixture (1 pseudo-iteration) | 1 | 4.607 | 4607.2 |

These costs include stage dumps/setup/video. Local graph capture is explicitly
disabled for the observational hook; the ordinary fixture's shared-boundary
capture blocker remains. The unfused damper/closure probe is not a performance
implementation, and native/local timing inclusion differs. The 200-iteration
caps remain unresolved in both implementations. This establishes **bounded
startup agreement**, not convergence or a validated wall-impact solution.

**Verdict / next work:** retain these reproducible diagnostic controls. A clean
ACSPH-local wall-flux/velocity closure implementation, preserving the shared
boundary code, should be qualified with linear hydrostatic and static slip
invariants before promoting it. Keep the reference RK integer quirk diagnostic;
compare a justified point-implicit model and a common, fully reported stopping
criterion. Then extend the exact-IC comparison toward impact, while measuring
shifting/field transport separately. No full impact campaign was started.

Regenerate the new analyses using `analyze_operators.py --root <evidence-root>`
and `analyze_lobovsky.py` with the five successive local directories listed
above. Retained artifacts: `operators-analysis/{summary.json,operators.png}`,
`operators-trajectory-analysis/{summary.json,comparison.png}`, native raw
operator/shift dumps, and each trial's video/config/log/snapshots. Final probe
source copies/hashes are in `operator-probe-sources`; the comparison harness
also fingerprints its own and the hook's source for future runs.

Performance follow-up: [retained increments and fused residual results](PERFORMANCE_2026-10-27.md#implemented-retained-increments-fused-residual-and-production-graph-capture) qualify the production optimization separately from these diagnostic operator choices. Final runs use production graph capture without the process-local boundary benchmark patch.
