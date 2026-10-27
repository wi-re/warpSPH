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

## Verdict and next discriminator

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
