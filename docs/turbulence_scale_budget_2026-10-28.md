# Shared 2D/3D compressible scale-energy diagnostics

2026-10-28. [Roadmap](roadmap/driven_compressible_turbulence.md),
[forcing-band control](driven_forcing_band_control_2026-10-28.md).

**A reusable offline diagnostic core and accepted-state sampling interface are
implemented and checked in both2D and3D.** Eighteen CPU controls pass, including
an end-to-end3D retained-state fixture. Applied to802 saved2D states, the tools
find negative mean transfer at two examined Gaussian widths, substantial
pressure conversion, and observation/time-sampling sensitivity. These are
measurements of the reconstructed fields, not a qualified inertial-range cascade
or a uniquely identified numerical viscosity. No new physical trajectory,
production transport term, timestep change or default forcing change was made.

## Implemented components

- [PeriodicBudget and conservative observation mapping](../src/warpSPH/utils/turbulenceBudget.py):
  dimension-generic2D/3D Gaussian filtering, physical-length spectral derivatives,
  density-weighted resolved kinetic energy, deformation and baropycnal transfer,
  pressure dilatation, forcing work and optional supplied viscous-stress work.
  A collocation Euler RHS/kinetic chain rule checks the discrete budget algebra.
- [Offline CLI](../scripts/analyze_turbulenceBudget.py): reads retained particle
  NPZ snapshots and the full forcing checkpoint, reconstructs applied acceleration
  at actual timestamps, compares native dense work/bath budgets, and reports
  spatial-grid, observation-radius and time-decimation sensitivity.
- [Plot CLI](../scripts/plot_turbulenceBudget.py): rate decomposition and cumulative
  resolved-energy ledger from retained scalar summaries; no particle reread.
- `AcceptedSampleClock` and the [qualification probe](../scripts/qualify_drivenTurbulence.py):
  requested physical sample cadence through `--sample-interval`, actual accepted
  times and missed-target counts in `sampling.json`. No timestep restriction,
  repeated synthetic state or time interpolation. Existing0.1 default remains.
  The previously tested forcing-band parameters are now exposed as `--k-min`
  and `--k-max`, with defaults1 and2 unchanged.

The analysis CLI loads only the NumPy utility, avoiding hydrodynamic runtime or
CUDA initialization. The production case remains2D; this work qualifies the
analysis/sampling components for3D data, not a3D flow solver or production run.
The direct grid API also supports rectangular periodic boxes. The particle CLI
currently supports cubic2D/3D boxes in the retained qualification NPZ format.

## Definitions and signs

The scale budget follows [Aluie, *Scale decomposition in compressible turbulence*,
equations9 and13–19](https://arxiv.org/html/1012.5877). With a linear filter G,
use density-weighted velocity `v_tilde=G(rho*v)/G(rho)` and resolved energy density
`K_l=G(rho)*|v_tilde|²/2`. Deformation transfer Π and baropycnal transfer Λ are
kept separate from pressure dilatation P_l div(v_bar). Periodic spatial averaging
gives

`d<K_l>/dt = -<Π+Λ> + <P_l div(v_bar)> + <v_tilde · G(rho*a)> - <D_l>`.

Positive Π+Λ removes resolved energy; positive pressure dilatation feeds kinetic
energy. Stress is caller-supplied with momentum source `+div(sigma)`, and
`D_l = grad(v_tilde):G(sigma)`. The diagnostic chooses no viscosity coefficient.
Rates are spatial means per volume, not shell sums. Multiply by box volume for
total powers. Gaussian ell is a physical standard deviation, not a sharp
wavenumber cutoff. The core uses coordinate-order `ij` arrays, including in3D;
the older2D spectrum/deposition adapter uses `xy`, so its axes are not reused
without an explicit transpose.

Products use grid collocation; the tool neither claims nor silently applies
de-aliasing. Smooth filtering, grid refinement and an independent collocation
identity residual expose this numerical limitation. Even-grid real derivatives
zero their own-axis Nyquist mode; the Gaussian filter retains its normal
attenuation there.

## Validation

The18 CPU checks cover:

- Physical derivative units and analytic Gaussian response in nonunit rectangular
  2D/3D boxes, plus the declared Nyquist convention.
- Uniform force work and zero spurious transfer; analytic pressure-dilatation
  magnitude/sign; a divergence-free three-mode interaction with independently
  derived modal energy rates, testing both forward and reversed transfer signs.
- Fully compressible2D/3D manufactured fields, including nonzero baropycnal work,
  filtered Euler-budget closure and Galilean invariance of the transfer fields.
- Supplied Newtonian shear stress with its analytic viscous work and energy loss.
- Conservative mass/momentum, pressure-volume, internal-energy, forcing-work and
  heat mapping; kinetic resolved/unresolved identity; legacy2D axis agreement.
- An end-to-end3D fixture with512 particles and nine synthetic observations,
  run through the same CLI. It is not a hydrodynamic trajectory.
- Sampling missed-target accounting, idempotent repeated queries and rejection
  of invalid cadence before runtime/output creation.

Replaying both real dense accepted-time histories reproduces all401 existing
sample timestamps bitwise at the default0.1 cadence. A requested0.025 cadence
would select1,601 distinct accepted states in each history, with no missed
clock targets. This demonstrates clock behavior; it does not recover particle
states that were never saved. `sampling-replay.json` retains the receipts.

One initial3D pointwise invariance test missed a3e-14 absolute floor at a
near-zero value by6.6e-15. The corrected floor is256 FP64 eps times the global
term scale; analytic work/sign tolerances are unchanged. Original test output
is retained. A standalone full-package import also hit an unrelated Numba cache
write under read-only execution. The NumPy-only CLI avoids that runtime import.

## Application to the retained forcing-band runs

Use the same frozen CRKSPH128² band1–2 and band2–4 trajectories as the preceding
control. All401 states per run are retained, with actual spacing about0.1.
The observation grid is32² or64². Every particle uses the same compact
Wendland radius H=0.0625 or0.09375, held constant across time and runs.
Additional Gaussian widths are ell=0.0625,0.125,0.25. Means cover10..40;
three10-unit blocks and time decimation to approximately0.2/0.4 are also saved.

Each particle stencil is normalized. Mass, momentum, pressure-volume integral,
internal energy, external work and bath heat conserve their extensive sums to
within8.6e-15 absolute in this study. The mapping separately records kinetic
energy and forcing correlations lost by averaging; it does not turn missing
resolved kinetic energy into heat or fill empty cells. All observation cells
are covered. Pressure is mapped from particle volume times pressure; no new EOS
or thermal model is imposed by the analysis.

At64², H=0.0625 and ell=0.125, mean rates per volume are:

| Term | Band1–2 | Band2–4 |
|---|---:|---:|
| Deformation transfer Π | +0.006855 | -0.001113 |
| Baropycnal transfer Λ | -0.007894 | -0.002213 |
| Total transfer Π+Λ | **-0.001039** | **-0.003326** |
| Pressure dilatation | -0.011331 | -0.003844 |
| Coarse forcing work | +0.010852 | +0.000115 |
| Observed resolved-energy rate | +0.001250 | +0.000683 |
| Unclosed residual | +0.000690 | +0.001087 |

At this width, looking only at Π would give the wrong sign for total transfer
in the baseline. Both reconstructed flows have mean negative total transfer at
ell0.0625/0.125 for both observation radii. This is evidence for upscale transfer
in the measured filtered fields, subject to the limitations below. It is not
proof of a stationary inverse-cascade range. The baseline total transfer becomes
positive at ell0.25; there is no demonstrated scale-independent flux plateau.
A positive residual cannot simply be called positive viscous dissipation,
which has the opposite sign in this ledger.

![Rate decomposition and sensitivity](/home/lu26029/.codex/worktrees/meshless-boundary-balance/warpSPH/.tmp/driven-scale-budget-20261028/figures/rates.png)

### Measurement limitations exposed by the tool

**Grid refinement is small here.** Changing32² to64² at fixed H changes the
mean total transfer by at most6.8e-7 across the tested widths/runs. The broader
H case is even less sensitive. This tests evaluation of the observed fields,
not particle-resolution convergence.

**Observation filtering is consequential.** At ell0.125, raising H from0.0625
to0.09375 changes total transfer from-0.001039 to-0.000476 in the baseline,
and from-0.003326 to-0.002482 in the control. Their signs persist at this width,
but magnitudes are not observation-independent. The tool's flux is computed
from reconstructed grid density/velocity. It does not include all stresses
below the particle-to-grid observation scale.

**Time sampling needs a tighter control.** Integrating reconstructed instantaneous
particle forcing power at the saved0.1 cadence gives0.031085 versus native
0.031709 in the baseline (-1.97%), and0.012210 versus0.012146 in the control
(+0.52%). These discrepancies include snapshot quadrature and native compatible
half-kick work semantics; the cadence alone is not proven to explain the whole
difference. They are much larger than the measured Fourier evaluation error.
Decimation also changes the coarse forcing budget and residual, so the smaller
transfer balances cannot be treated as tightly closed. The signed bath-power
difference is only about1e-6.

**The residual is not a viscosity measurement.** It combines missing observation
stress/source correlations, temporal sampling, pressure reconstruction, native
scheme effects and collocation error. The collocation identity residual is near
roundoff for these smooth reconstructed fields, which validates their budget
algebra but does not close the native particle equations. In particular, neither
a unique effective viscosity nor pure physical dissipation can be fitted to
this gap. The cumulative ledger makes that distinction visible.

![Observed and predicted resolved energy changes](/home/lu26029/.codex/worktrees/meshless-boundary-balance/warpSPH/.tmp/driven-scale-budget-20261028/figures/ledger.png)

The two analyses cost125.81s and131.69s on one CPU thread, about4.3min together.
Mapping dominates. No GPU was allocated, and no new flow or stationarity claim
was produced. The original dense native total-energy budgets remain unchanged.

## Dissipation and future3D output contract

The present case has no prescribed physical viscosity, bulk viscosity, thermal
conductivity or mechanical drag. It combines scheme-dependent numerical
transport with a finite-gamma internal-energy relaxation bath. Cooling is not
an explicitly specified momentum sink, and no common Reynolds number is fixed
across schemes/resolutions. Keep this case labeled with that model rather than
silently adding drag or viscosity to flatten its statistics.

The next measurement control is finer output cadence with unchanged physical
parameters, checking e.g.0.025 against0.0125 before selecting a production
cadence. Those values are candidates, not newly qualified defaults. The new
probe option makes that control possible without changing timestep selection.
A future fixed-transport case requires explicit dimensional coefficients and
separate shear/acoustic/conduction controls; the supplied-stress budget already
has an analytic work control for that future implementation. Native per-stage
work/dissipation instrumentation would be needed to identify discrete scheme
losses beyond what these archived observation fields can establish.

For3D, preserve masses, positions, all velocity components, density, pressure,
internal energy, actual timestamps and supports, full forcing parameters/past
knots, plus dense signed work/bath ledgers. Record the requested and achieved
sample cadence and any missed targets. Use identical physical observation
filters in resolution comparisons. Field memory scales with the number of3D
cells; the adapter bounds temporary particle/cell pairs but has not been
benchmarked on production-size3D/HPC data. No3D trajectory or MHD budget is
qualified by these controls.

## Reproduction and evidence

Typical analysis of saved runs:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python scripts/analyze_turbulenceBudget.py RUN_A RUN_B --output BUDGET_OUTPUT
python scripts/plot_turbulenceBudget.py BUDGET_OUTPUT --output FIGURE_OUTPUT
```

The default physical radii assume the present unit box; choose explicit
`--grids`, `--radii`, `--scales` and `--window` for another problem. The CLI rejects
invalid coverage/density, domain/forcing dimension mismatches and incomplete
time windows. The direct grid API accepts an explicitly supplied viscous stress;
no stress is inferred when absent.

Raw evidence: `.tmp/driven-scale-budget-20261028/`, including test XML/logs,
per-frame input hashes, ledgers, grid/scale series, block/time sensitivity,
source snapshots, source reconciliation, figures and sampling replay.
The sampling-clock API was added after the two analyses completed; exact prior
utility source is archived by its recorded SHA256 and an AST check verifies
the budget and deposition algorithms are unchanged. The CLI metadata was also tightened afterwards to avoid inferring transport
from an absent stress input; an exact text comparison verifies this changes
metadata only. Failed initial diagnostic checks remain in the same evidence
directory.
