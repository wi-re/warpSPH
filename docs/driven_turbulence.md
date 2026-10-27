# Driven compressible turbulence

The drivenTurbulence case is a periodic **2D compressible ideal-gas flow**, using
MFM by default. A fixed-clock Ornstein–Uhlenbeck process drives a small set of
Fourier modes. The first supported thermodynamics are finite-time relaxation to
a prescribed specific internal energy, or an explicitly adiabatic gas.

This is the first developing-flow implementation in the
[longer horizon roadmap](roadmap/driven_compressible_turbulence.md). It is separate
from WCSPH/incompressible Kolmogorov flow. It does not yet implement the exact
isothermal Mach 0.3/8.4 benchmark or establish stationary turbulence statistics.

## Run the case

```bash
warpsph-run drivenTurbulence --config examples/configuration/drivenTurbulence.yaml
warpsph-run drivenTurbulence --nx 128 --tLimit 4 --no-plot --no-video
```

The example declares the thermal model, forcing clock and solver policy.
Increase accelerationRMS only as an explicit change to the driving strength;
RMS Mach is measured rather than held by feedback. The default accelerationRMS
is an ensemble RMS at OU knots before removal of the instantaneous mean force.
Finite-clock interpolation and mean removal change the realized RMS.

MFM uses its native two-stage SSPRK route. MFV is also supported with an explicit
solver.temporalScheme=ssprk override. MFV's default single-stage Hancock route
is refused for this case because the external sources have no Hancock source
predictor. Existing source-free solver defaults are unchanged. The case currently
supports eager execution, a single ideal gas and two spatial dimensions. The
native meshless Courant parameter is `solver.cfl`; the example explicitly uses
0.1. The general simulation.cflFactor does not replace that meshless setting.

## Physical model

The gas retains a finite ratio of specific heats greater than one and the
standard ideal-gas fluxes and acoustic signal speeds. Initially, velocity is
zero, nominal density is rho0, and the prescribed adiabatic sound speed fixes
`u0 = soundSpeed² / [gamma (gamma - 1)]`. Physical masses
come from periodic cell volumes; energy and pressure use the initial meshless
geometry, without relaxing masses or particle positions.

For thermalModel=relaxation the source is du/dt = (u0-u)/coolingTime. This is a
heat bath: the signed thermal exchange may heat expanding fluid or cool heated
fluid. For thermalModel=adiabatic that source is absent. Neither mode is an
exact isothermal EOS, and cooling energy is reported rather than hidden in an
energy reset. There is no physical viscosity/conductivity model, large-scale
drag or target-Mach feedback in this first case; the hydrodynamic solver supplies
its native numerical dissipation. A developing 2D flow need not be statistically
stationary.

## Reproducible forcing

Integer wavevectors satisfy kMin <= magnitude <= kMax in box units, with one
representative from each opposite pair. Independent real sine/cosine
coefficients follow exact OU transitions at forcingInterval, with linear
interpolation between those physical times. solenoidalWeight=1 projects onto
transverse acceleration; zero selects longitudinal acceleration. Intermediate
values are an amplitude mixture, with ensemble normalization accounting for
the projection. Solenoidal forcing does not make the gas incompressible.

The OU generator is local and seeded independently of the global random state.
Knots are generated chronologically and retained, so repeated, earlier or
extra RHS/diagnostic queries cannot change the forcing at a given time. The
field evaluator stores at most eight modes per particle in its temporary phase
array, rather than allocating a full particle-by-all-modes tensor.
FourierOU.accelerationBound supplies a conservative spatial bound over the
upcoming knot interval; the cached bound respects knot mutations and does not
change the realized noise process.

FourierOU.state_dict and from_state_dict preserve knots and generator state for
forcing checkpoint continuation, including future and past queries. Native
`resumeFrom` supports versioned `storeMode=states` checkpoints containing the
complete forcing state, accepted budgets, initial budget references, physical
next timestep and completed-step count. Resume with the same physical/numerical
configuration; output settings and the endpoint may change. The case restores
timestep-growth history and infers the absolute step offset. Earlier outputs
without that context are refused rather than restarted with a new driver.

```bash
warpsph-run drivenTurbulence --config examples/configuration/drivenTurbulence.yaml \
  --store --storeMode states --storeInterval 1000 \
  --exportRoot .tmp/driven-checkpoints --tLimit 40
warpsph-run drivenTurbulence --config examples/configuration/drivenTurbulence.yaml \
  --resumeFrom /path/to/trajectory/finalState.h5 --tLimit 80
```

Exact continuation checks use interior checkpoints or step-count splits. A
time-limited segment can shorten its final step; extending that segment is
a valid continuation but has a different step sequence from a run that never
stopped at that endpoint. Same-source/device/precision controls cover MFM and
MFV on CPU and GPU, including a late clustered supersonic state. Cross-device
statistical reproduction and production-scale restart remain separate gates.
Trajectory files are observation output, not native restart checkpoints.
The generic runner derives its trajectory output stride from the initial
timestep; adaptive timesteps therefore produce nonuniform physical output
intervals. Consume the recorded timestamps and retain the model timestep.
The qualification probe instead samples the first accepted state crossing
each fixed 0.1-time threshold and also retains its actual timestamp.

## Conservative sources and budgets

The meshless RHS adds mass times acceleration to momentum and velocity dotted
with that force to total energy at each SSPRK stage. No mass source is added.
By default, the instantaneous mass-weighted mean acceleration is removed,
including when MFV changes individual masses.

Force work and signed bath heat are diagnostic integrals in sourceEnergyBudget.
They accumulate in FP64 using the accepted integrator's stage weights in finalization, without
changing stage selection or duplicating the physical time driver. The energy
budget residual compares the actual conserved total energy against the initial
energy plus those two exchanges. Cooling is therefore not mistaken for a
conservation defect.

The first and subsequent steps obey native signal limits, with additional
forcing-clock, acceleration and explicit-cooling caps. The forcing bound uses
all enclosing knots over a possible step, including growth across a knot. An
additional available-thermal-energy limit prevents a force update from making
an SSPRK predictor cold before compatible full-step work is accumulated. Force
alone consumes at most one sixteenth of the minimum specific internal energy
in that predictor. Cold gas with strong driving can therefore need much smaller
steps even when its acoustic CFL limit is loose. This is a local safety
restriction on the existing SSPRK method, not a change taken from the Hopkins
isothermal model. For the force-only Euler predictor, v′=v+aΔt and
E′=E+v·aΔt give u′=u−|aΔt|²/2. With a conservative bound A on
the mean-removed acceleration, Δt≤√(2 min(u))/(4A) limits that loss to
min(u)/16. Native signal restrictions still control the hydrodynamic update;
this source bound alone is not a proof of positivity for an arbitrary coupled
flow. Time-limited runs cap
the final step and snap only time roundoff. No thermal clipping, support
reduction or failure-driven CFL retry is introduced by this case.

Diagnostics include mass-weighted local RMS Mach after subtracting the mean
velocity, bulk speed, density/pressure/internal-energy minima, force work,
thermal exchange and the relative energy-budget residual. logDensityStdParticle
is explicitly an unweighted particle diagnostic. Spectra, mass/volume PDFs,
correlation times and their resolution dependence remain part of the statistical
qualification before reference training-data production.

## Output and validation

State exports include the accepted budget and versioned restart context.
Trajectory frames retain actual Q, masses, particle IDs, supports and budget
integrals alongside the primitives. Physical timestamps are stored in FP64,
without rounding a FP64 simulation clock to FP32.

Initial file attributes describe the initial state. Time-varying conditional
inputs live in `frameData/frame_NNNNN`: forcing parameters, integer modes, the
applied Fourier coefficients and forcing time. Each frame also writes new raw
OU knots and their projected coefficients, with forcingKnotStart identifying
the first index. Empty chunks are allowed. Concatenating the chunks retains
every knot, including updates between stored fluid frames, without duplicating
the complete history at every frame. The analysis command checks continuity,
parameters, timestamps and reconstruction of the applied field. The seed alone
is not a substitute for exposing this driving to a learned model.

Focused regressions in tests/test_drivenTurbulence.py cover query-order and
checkpoint reproducibility, periodicity, Fourier projection and Parseval
normalization, OU variance/correlation, mass-weighted force centering, compatible
force work, exponential cooling, both declared meshless routes, repeated runs
stored forcing/budget metadata, exact native continuation and complete
between-frame driving history. Cold-gas checks inspect every RHS and include
a uniform-force analytic temperature control and a force ramp across a knot. Existing meshless and typed-case checks
protect integration and registration.

GPU developing-flow pilots and the independent per-RHS observer live under
Git-ignored .tmp/driven-turbulence-20261027 in the implementation worktree.
They retain physical inputs, initial/final arrays, all accepted diagnostics,
RHS admissibility, exact runtime sources and source hashes. Their timings include
extra observations/synchronization; peak Torch memory is not the complete device
residency needed for the HPC feasibility gate. A pilot is not a stationary or
resolution-converged dataset.

## Statistical qualification protocol

`warpSPH.utils.turbulence` provides offline FP64 measurements, separate from the
physical update. `scripts/analyze_drivenTurbulence.py` analyzes retained particle
snapshots without running a simulation:

```bash
python scripts/analyze_drivenTurbulence.py .tmp/campaign/weak64-seed1234 \
  .tmp/campaign/weak128-seed1234 --output .tmp/campaign/analysis \
  --grids 16 32 --blocks 10 20 30 40
```

The command accepts a native run directory containing `trajectory.h5` and
`caseSpec.json`, or an investigation directory with `input.json` and
chronological `samples/*.npz`.
Each snapshot supplies scalar `t`, positions/velocities of shape N by 2, and
masses, densities, pressures and internalEnergies of shape N. The qualification
probe also retains Q and supports for physical audits. Optional history.jsonl
and result.json supply all-step budgets and the runtime/admissibility outcome.
Keep experiment drivers, snapshots and generated analysis under `.tmp/`.
Native budget checks cover saved snapshots only; an independent RHS/accepted-step
observer is still needed for an all-stage qualification. Missing source manifests
and runtime results are explicitly identified rather than inferred from HDF5.
Output includes per-snapshot particle statistics/PDFs, filtered spectra,
structure functions, correlation estimates and a machine-readable summary.

The measurement conventions are:

- Particle means and PDFs use either actual masses or meshless volumes m/rho,
  with each set of weights explicitly normalized. Report the difference between
  the summed meshless volume and physical box area. Log density is relative to
  sum(m)/L²; pressure PDFs are relative to the initial nominal pressure. Common
  PDF edges retain underflow and overflow as separate probability bins.
- Periodic, cell-centred cloud-in-cell deposition conserves mass and momentum.
  Grid velocity is deposited momentum divided by deposited mass. Record the
  kinetic energy removed by this measurement filter. Empty cells retain NaN
  velocity and invalidate spectra; they are never filled for plotting or fits.
- `--deposition kernel` selects an explicit adaptive measurement filter for sparse
  regions: a compact Wendland C2 shape with radius max(particle support, twice
  the grid spacing). Each particle's discrete stencil is normalized to conserve
  mass and momentum. Report the filter radii, fraction of particles with
  cell-scale broadening, empty cells and unresolved kinetic energy. This requires
  supports in the snapshots. It changes the observation operator, not the fluid
  model, and does not recover unresolved structure. Retain CIC failures and
  compare low modes at multiple grid resolutions rather than joining spectra
  from different filters. Uncovered cells still invalidate kernel spectra.
  Native supports depend on fluid resolution and state, so matching grid size
  alone does not match the physical smoothing radius. Use the recorded radii
  and filtering loss when interpreting cross-resolution differences.
- Velocity spectra subtract the grid volume mean and use the full 2D FFT divided
  by the number of cells. Shell sums of half the squared transform obey Parseval.
  The longitudinal and transverse projections describe velocity, not a
  variable-density kinetic-energy spectrum. Density power uses rho/mean(rho)-1.
  There is no CIC deconvolution. Retain all shells for the normalization check,
  but compare at k <= gridSize/4 and check sensitivity to deposition resolution.
  Measurement version2 separates exact box-wavelength power (kx²+ky²=1)
  from the rounded first radial shell, which also includes diagonal k²=2 modes.
  A few resolved modes do not establish an inertial-range slope.
- Second-order longitudinal/transverse increments average the two grid-axis
  directions; an isotropic angular average is not assumed.
- Scalar and connected Eulerian-velocity autocorrelations use actual physical
  timestamps, interpolated to a uniform clock. Integrate the initial positive
  correlation lobe, including half the zero-lag bin. Report right censoring when
  no zero crossing is observed within half the record, and report effective
  sample counts. These are finite-record diagnostics, not confidence guarantees.

The first declared long-flow screen uses t=0..40, discards t<10, and compares
three blocks [10,20], [20,30], [30,40]. Time means use piecewise-linear quadrature,
not an average over adaptive step counts. For each of RMS Mach, kinetic energy,
and mass/volume log-density spread, require block-mean spread <=10% of its mean,
blocks at least five estimated integrated correlation times long, uncensored
correlation estimates, and at least ten accumulated box crossings in the
assessment window. A failed screen means stationarity is **not established**;
it does not by itself distinguish secular evolution from sampling uncertainty.
A passing scalar screen still needs spatial-correlation, PDF/spectrum,
resolution and independent-seed checks before any stationary-data claim.

Use whole independent trajectories as training/validation/test partitions.
Keep all runs sharing a forcing seed/realization in the same partition,
including matched resolutions, methods and amplitude-scaled realizations.
The saved 0.1-time snapshot cadence supports correlation measurement; it does
not declare adjacent frames independent. Choose a decorrelated sampling cadence
from the slowest relevant measured field/observable, and keep temporally dense
trajectories labelled as correlated sequences with their forcing inputs. A
dynamics dataset also needs a cadence-refinement check for its fastest resolved
structures; the global velocity autocorrelation does not establish that. Model
adapters must consume thermal/conserved fields and interval forcing history,
rather than treating positions, velocity and density alone as a closed state.

Reproduce one observed realization with the maintained harness (run GPU jobs
sequentially):

```bash
python scripts/qualify_drivenTurbulence.py --nx 64 --time 40 --seed 1234 \
  --acceleration 0.3 --root .tmp/driven-qualification-new/weak64-s1234
python scripts/analyze_drivenTurbulence.py .tmp/driven-qualification-new/weak64-s1234 \
  --output .tmp/driven-qualification-new/cic
python scripts/analyze_drivenTurbulence.py .tmp/driven-qualification-new/weak64-s1234 \
  --output .tmp/driven-qualification-new/kernel --deposition kernel
```

The harness refuses an existing output directory, archives source/input/driver
provenance, checks every RHS, logs every accepted state, and retains observations
at fixed physical thresholds and the final forcing checkpoint. VisPy video and
live progress are enabled; its ten-step independent warmup omits video. Run the
other seeds and resolutions as separately named attempts. It reports Torch
allocator peaks, which exclude context/graphics/other-process memory and do not
qualify total device capacity. Wall time includes observation/video overhead.
The original cohort drivers are archived unchanged with their runs; those runs
retained snapshots for software-VisPy replay instead of drawing during stepping.

The first long-flow campaign is retained under
`.tmp/driven-qualification-20261027`: weak and strong acceleration inputs 0.3 and
6, three 64² seeds, and a matched 128² seed, without changes to thermal closure,
forcing or solver policy. It tests the finite-time relaxation model. The
isothermal Hopkins reference remains a different physical problem
([section 4.4.4](https://arxiv.org/html/1409.7395#S4.SS4.SSS4)); neither changing
gamma to one nor interpreting a short cooling time as an exact EOS qualifies it.

The [first qualification report](driven_turbulence_qualification_2026-10-27.md)
records the cohort, cold-source failure, corrected controls and remaining gates.

Qualification currently establishes numerical source/output/restart controls
and supplies an explicit statistical screen. It does not declare the default
flow stationary or resolution-converged. Long-flow studies retain source
archives and all failures under `.tmp/driven-qualification-20261027`; the
maintained measurement protocol above separates numerical admissibility,
finite-record sampling uncertainty and spatial-resolution evidence.
