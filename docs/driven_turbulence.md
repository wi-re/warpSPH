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
supports eager execution, a single ideal gas and two spatial dimensions.

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

FourierOU.state_dict and from_state_dict preserve knots and generator state for
forcing checkpoint continuation, including future and past queries. Whole-run
resumeFrom is currently refused for this case: restoring the forcing, the
native timestep history and the complete run lifecycle together is a later
qualification gate. Output states are still available for inspection and
future restart development.

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
forcing-clock, acceleration and explicit-cooling caps. Time-limited runs cap
the final step and snap only time roundoff. No thermal clipping, support
reduction or failure-driven CFL retry is introduced by this case.

Diagnostics include mass-weighted local RMS Mach after subtracting the mean
velocity, bulk speed, density/pressure/internal-energy minima, force work,
thermal exchange and the relative energy-budget residual. logDensityStdParticle
is explicitly an unweighted particle diagnostic. Spectra, mass/volume PDFs,
correlation times and their resolution dependence remain part of the statistical
qualification before reference training-data production.

## Output and validation

State exports include the accepted budget. Case metadata include the forcing
parameters, integer modes, the applied Fourier coefficients and their physical
time. They identify the actual conditional input to a trajectory; the seed alone
is not a substitute for exposing the driving to a learned model.

Focused regressions in tests/test_drivenTurbulence.py cover query-order and
checkpoint reproducibility, periodicity, Fourier projection and Parseval
normalization, OU variance/correlation, mass-weighted force centering, compatible
force work, exponential cooling, both declared meshless routes, repeated runs
and stored forcing/budget metadata. Existing meshless and typed-case checks
protect integration and registration.

GPU developing-flow pilots and the independent per-RHS observer live under
Git-ignored .tmp/driven-turbulence-20261027 in the implementation worktree.
They retain physical inputs, initial/final arrays, all accepted diagnostics,
RHS admissibility, exact runtime sources and source hashes. Their timings include
extra observations/synchronization; peak Torch memory is not the complete device
residency needed for the HPC feasibility gate. A pilot is not a stationary or
resolution-converged dataset.
