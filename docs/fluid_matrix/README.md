# Fluid comparison tools

The comparison tools run declared cases sequentially and keep measurements,
source snapshots and failure evidence beside each run. Generated output is
local work under the Git-ignored `.tmp/` directory; it is not documentation.

## Generate and run a comparison

```bash
python scripts/run_fluidMatrix.py plan --root .tmp/fluid-matrix
python scripts/run_fluidMatrix.py run --root .tmp/fluid-matrix
python scripts/analyze_fluidMatrix.py --root .tmp/fluid-matrix
python scripts/audit_fluidMatrix.py --root .tmp/fluid-matrix
python scripts/plot_fluidMatrixRanking.py --root .tmp/fluid-matrix
```

`plan` writes `ROOT/manifest.json`. Inspect the declared support, initialization,
viscosity, boundary representation, timestep limits and endpoints before
running. Use `--schemes` to select a subset and `--manifest` for an explicit
saved plan. Use a fresh output root for a new comparison; preserve prior
measurements when changing a numerical preset. Reports, ranking figures,
metrics, particle data and movies remain under the selected output root.

The runner respects physical timestep limits from the first step. Fixed-step
requests provide a ceiling. The velocity stop is 20 times the case's declared
reference scale. Unsupported combinations and failed or partial attempts
remain visible in the results.

## Read the results

Completion is separate from accuracy. A stopped run cannot be ranked as an
endpoint result, and missing measurements must not receive a numerical score.
When combining runs, preserve each row's date, source root and numerical
settings. Pressure probes, viscosity and sampling conventions can differ
between schemes; inspect those declarations before interpreting close ranks.

The ranking tool writes its metric definitions alongside the figure. Decaying
vortex retention is a relative energy criterion, with growth penalized.
Hydrostatic pressure, wall loads, viscous decay and impact traces have distinct
metrics. Runtime includes initialization, diagnostics and output work; it is
not an isolated solver benchmark.

To bring a completed full prepared profile into the normal matrix, retain the
original Delta-SPH, Sun delta-plus and ACSPH runs and replace all projection
rows together:

```bash
python scripts/compare_preparedFluidMatrix.py \
  --baseline /path/to/original-full-matrix \
  --prepared .tmp/incompressible-prepared \
  --out .tmp/fluid-matrix-prepared-comparison
```

Use an empty destination. This produces a comparison-only manifest, dated
11-scheme quality/runtime figures, case curves, HTML, CSV and hashed source
receipts without running physics or modifying either input. All declared
failures and unsupported slots remain. Per-row `data_root` resolves the actual
record, and the original specs/settings remain in the combined manifest.
Historical endpoint exclusions and pressure-observation qualifications carry
through unchanged. The separate rearranged TGV experiment is not substituted
into the primary matrix. Retained and prepared runs have different ICs,
kernels and presets; the combined ranking is a capability survey, not a
controlled solver-only comparison. Do not pass its comparison-only manifest
to the simulation runner.

Figure legends and titles describe the physical settings and use explicit
units. Keep run identifiers in machine-readable metadata.

## Preserve a baseline

```bash
python scripts/preserve_fluidMatrix.py --root .tmp/fluid-matrix \
  --archive .tmp/archives/fluid-matrix.tar.zst
```

The preservation tool checks archive contents and writes receipts beside the
archive and in the run's preservation directory. It does not write generated
receipts into `docs/`.


## Prepared incompressible profile

```bash
python scripts/run_fluidMatrix.py plan --profile prepared --scale full \
  --root .tmp/incompressible-prepared
python scripts/run_fluidMatrix.py run --root .tmp/incompressible-prepared
python scripts/analyze_fluidMatrix.py --root .tmp/incompressible-prepared
python scripts/audit_fluidMatrix.py --root .tmp/incompressible-prepared
python scripts/plot_fluidMatrixRanking.py --root .tmp/incompressible-prepared
```

This opt-in profile includes native IISPH, native DF-first DFSPH,
predictor-first DFSPH, post-drift density shift with velocity resampling, and
all four legacy projection routes. The legacy reference divergence-pressure
carrier is explicitly zeroed independently of its historical soundspeed slot.
The latter retain their operator/source
policies and have explicit legacy labels. `--schemes` selects profile method
IDs; `--scale screen` caps resolution at64 while retaining the endpoints.
Screen and full results belong in separate roots and must not be pooled.
Full scale retains the original resolutions, including256²TGV particles.
Unsupported moving-wall and Band analytic-wall combinations remain visible.

The 26 October full rerun of `stability-v1` was **stopped and rejected** by the user: sloshing remains physically underactive, with suppressed wave motion, unexpectedly low speeds and large pressure errors. Removal of startup eruptions is not a physical qualification. Completed and partial evidence is retained under `.tmp/incompressible-corrected-rows-2026-10-26`; the normal matrix has not been replaced by these results. The preset remains experimental.

New prepared plans use version 9, revision `sampling-recovery-v1`. All three
native DFSPH orderings explicitly enable the shared 0.9ρ₀ recovery gate for
clamped divergence with either wall representation. Signed periodic and
IISPH density constraints keep their existing semantics. Historical revisions
and saved manifests without a gate entry use zero; no saved result is replaced.
See [Sampling recovery](../incompressible/SAMPLING_RECOVERY.md) for the rule,
evidence and remaining tradeoffs.

Version 8, `--prepared-revision consistency-v1`, retains the pre-gate policy. Dam break and sloshing preserve their case viscosity settings; the inviscid dam break uses alpha=0.01 and c0=47.3091059, matching the retained WCSPH control. Native sampled viscosity obtains a fresh reflected mDBC reference field without changing pressure/rigid-body velocities. The sampled contour also retains the physical wall BC.

The pre-cutoff representative replay recovers the expected early energy rise but fails at 0.81858 s with a localized divergence-pressure spike. The repaired setup is not a qualified matrix result, and the wider matrix remains stopped. [Representative report](../../.tmp/dfsph-dambreak-consistency-2026-10-26/README.md). The user-authorized weak-factor cutoff follow-up now completes the exact same particle-wall dam break through2.45s: integrated K/M99.14% of Delta, pressure error42.245%, up to four simultaneous wall penetrations. This single-case robustness result does not update the broader matrix or qualify pressure accuracy. [Cutoff replay](../../.tmp/dfsph-cutoff-dambreak-2026-10-26/README.md).

The earlier version7, revision `stability-v1`, remains selectable for historical viscosity presets. Its explicit per-row overrides select Morris viscosity for shear, calibrated with the actual C4
kernel's square-lattice long-wave moment (targetν=.0185; configured
ν=.0188705359 at4dx support). Sloshing retains its MG viscosity and uses a
0.2ms step ceiling. Native VD+PS TGV uses0.001%divergence tolerance because
that stage supplies the physical momentum correction. Other rows retain the
previous settings. `--prepared-revision stability-v1` selects version7; `--prepared-revision original` reproduces the version6
plan for historical comparisons; old saved manifests without these overrides
retain their recorded viscosity overrides. Exact reproduction of the old wall-velocity bug requires the archived source, since the native viscosity path is now repaired. The viscosity choices and general projection timestep formula are unchanged;
the current recovery policy is selected separately by version 9.

| Native method | Step order |
|---|---|
| DFSPH | Divergence → nonpressure velocity prediction → density pressure → drift |
| Predictor-first DFSPH (omni-like order) | Nonpressure velocity prediction → divergence → density pressure → drift |
| VD+PS with native operators | Nonpressure velocity prediction → divergence → drift → density shift and velocity resampling |
| IISPH | Nonpressure velocity prediction → pressure → drift |

The native variants share IISPH operators. The separately labeled legacy omni
route retains its own operators and policies; it is a distinct comparison row.

| Domain | Common initialization | Density constraint |
|---|---|---|
| Fully periodic | Nominal cell masses, bounded0.01dx jitter, zero velocity, damped native DFSPH preparation, fresh undamped rest check; sample the analytic field afterward | Native IISPH/DFSPH explicitly select `periodicMeanFree`; legacy policies remain declared separately |
| Closed static box | Nominal fluid/solid cell masses, theoretical fixed walls, jitter and damped preparation, fresh rest check; sample an analytic streamfunction curl afterward | Absolute density with compression-only pressure |
| Free surface | One bulk mass calibration; analytic flat-wall lattice alignment; scheme-independent saved physical arrays | Absolute density, compression-only pressure |

Preparation uses a fixed seed and drag rate20/s, with a5speriodic or2sclosed
limit and a0.25sfresh undamped hold. Quietness and iteration caps are recorded;
a duration-limited preparation is never relabeled converged. Every tested
method consumes the exact same saved positions, masses, supports and sampled
velocity for that case/boundary. Density is recomputed with its actual operator.
Mass stays fixed during preparation and evolution. No flow uses preparation
drag. Fixtures, preparation journals/snapshots and metadata live under
`ROOT/fixtures`; hashes are repeated in each result directory.

`periodicMeanFree` subtracts the arithmetic density-source mean; it changes
the evolution constraint and leaves measured density and physical mass alone.
It is a diagnostic variant with remaining mean-density drift, not an established
absolute-density remedy. The unchanged raw-source periodic failure is retained
in the dedicated absolute-source controls and earlier evidence; do not attribute
an improvement requiring source centering solely to better initial positions.

The common numerical preset uses WendlandC4/support4dx(C2for the existing
compact Couette closure),1msmaximum steps (0.2msfor corrected sloshing),
physical timestep limits, native pressure relaxation0.25, density tolerance
0.01% and divergence tolerance0.1% (0.001%for corrected native VD+PS TGV).
For closed/static controls the physical viscosity preset is0.01 except the
shear target and Couette controls(0.0185); the shear coefficient includes the
explicit lattice calibration above. Version8 free-surface rows preserve case
viscosity: dam break uses artificial alpha0.01 at c0=47.3091059, and sloshing
uses physical nu1e-6. Historical version6/7 free-surface rows used physical
nu0.01. No XSPH filter is enabled.
Legacy projection algorithms retain their declared iteration/source policies.
The four native routes differ in composition, not pressure geometry.

The recorded25October prepared shear/sloshing comparisons are unqualified:
runtime metadata shows projected Monaghan–Gingold viscosity, and fixed-geometry
GPU diagnostics find unstable explicit viscous modes at the declared timestep
cap. The shear adapter also skipped the case's Morris selection/calibration.
The compliance audit checked the declared cap, not its spectral adequacy.
Retain those trajectories as failure evidence; verify the actual operator and
its step bound in short controls before treating their ranks as solver quality.
Physical pressure for the two coupled DFSPH routes is the sum of density and
divergence pressures. VD+PS physical pressure excludes the shift multiplier.

Ranking retains the established case metrics and reports native projection
caps explicitly; legacy routes do not expose equivalent cap counters here.
A completed trajectory and a numerical rank do not establish convergence or
physical correctness. Fixture creation cost is listed separately and subtracted
from comparison wall cost so the first method does not pay everyone’s shared
preparation. Engine step timing remains available. The legacy matrix is kept
as historical evidence; changed initial states, viscosity and numerical policies
prevent interpreting old/new ranks as a single controlled solver-only change.
The evidence audit also records initial and final closed-box containment.
Small rest velocities do not imply a wall-contained or isotropic fixture;
initial boundary crossings remain visible independently of subsequent decay.

The prepared-profile pressure probes observe the actual legacy stage pressures
at the existing wall-load bookkeeping call and combine only the components
applied to velocity. The observer forwards the original call unchanged and
does not modify state or force arrays. This avoids treating a density-only
carry or a historical soundspeed carrier as total physical pressure. The
reference also observes its returned solve pressures, since its wall-load
bookkeeping is skipped when an analytic provider is absent. That observer
forwards the exact original inputs and returned tuple unchanged. The
old ambiguous sloshing `sensorPressureCD/DF` aliases are omitted by this
profile; `sensorPressure` is the physical total used by its ranking.

Where `hydrostaticInit` is enabled, every native route receives the same
physical pressure profile. At the actual first timestep the adapter encodes
that profile as density pressure timesdt²for coupled DFSPH, or physical
divergence pressure timesdtfor VD+PS(with zero auxiliary shift history).
IISPH consumes its physical pressure carry directly. `pressure_seed.json`
and `first_step_state.pt` retain the exact converted input; this avoids
accidentally erasing a pressure seed by passing a zero impulse history.


The saved dam-break energy audit also exposes specific preset and wall differences: prepared runs force physical `nu=0.01` while retained WCSPH uses artificial `alpha=0.01`; native sampled viscosity assumes mDBC ghost velocities have been prepared but its step does not perform that update. These records therefore cannot isolate projection dissipation. See [energy curves and archived call-path evidence](../../.tmp/dambreak-energy-audit-2026-10-26/README.md). No simulations were resumed for this audit.
