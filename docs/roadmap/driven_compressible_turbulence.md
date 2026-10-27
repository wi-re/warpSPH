# Driven compressible turbulence and multiphysics roadmap

Plan status and priorities: [PLANS.md](../../PLANS.md).

Build a reproducible source of chaotic fluid trajectories, beginning with driven
**2D compressible turbulence**, followed by local GPU feasibility studies and
larger HPC campaigns. Extend the validated hydrodynamic foundation to thermal
convection, then ideal MHD and magnetoconvection. Open boundaries and injection
form a separate development track.

Status: Stage A implementation started. The drivenTurbulence case now provides
fixed-clock Fourier OU acceleration, stage-consistent meshless force work and
named finite-time cooling/adiabatic modes. Simple controls and initial GPU
developing-flow pilots are covered in [the case guide](../driven_turbulence.md).
Exact-isothermal dynamics, stationary statistics, full runner restarts and HPC
production qualification remain open. Later stages below retain their own gates.

## Scope and dependencies

The first study is a periodic, fully compressible driven flow in two spatial
dimensions. Kolmogorov flow in the WCSPH and incompressible paths is an
orthogonal study. Existing Kolmogorov cases can inform shared forcing and
measurement infrastructure, but extending or validating them is not a
prerequisite for this roadmap. The first compressible study does not substitute
a sinusoidally driven incompressible case for compressible turbulence.

| Stage | Work | Gate before proceeding |
| --- | --- | --- |
| A | Driven 2D compressible turbulence, subsonic then supersonic | Reproducible forcing, explicit thermodynamics, valid budgets and resolved statistical behaviour |
| B | Local GPU proof of feasibility, including larger 2D runs and a 3D bridge | Measured memory, physical time advanced, stable long integration and restart fidelity |
| C | Larger GH200 HPC campaigns within the same per-GPU VRAM limit | Reproducible environment, measured capacity and validated production output |
| D | Thermal Rayleigh–Bénard convection | Conduction, gravity balance, onset and heat transport qualified before chaotic runs |
| E | Focused Richtmyer–Meshkov revisit and triple-point qualification | Shock/interface measurements and an uncontaminated observation window |
| F | Sustained supersonic external flow and injection | Validated inflow, outflow, wall interaction and material transport |
| G | Ideal MHD on the compressible core | Magnetic waves, shocks, advection and divergence control pass independent tests |
| H | Driven MHD turbulence and magnetic driving | Magnetic energy transfers, forcing and statistically stable dynamics qualified |
| I | Rayleigh–Bénard magnetoconvection | Thermal convection and MHD work separately, then coupled boundaries and transport pass |

Stages D and E follow the hydrodynamic foundation and can progress independently
of a large production campaign. Stage F is a separate boundary programme; ideal
MHD in a periodic domain does not need to wait for open boundaries.
Magnetoconvection depends on validated thermal and magnetic physics, rather than
on completion of every external-flow case.

## Stage A Driven 2D compressible turbulence

Use MFM/MFV as the initial conservative architecture, with selected CompSPH or
CRKSPH comparisons where useful. Choose one baseline method before broad method
sweeps so forcing and thermodynamics are qualified independently of solver
selection. Preserve differences in numerical dissipation in the metadata.

The reference is Hopkins section 4.4.4: driven, isothermal, periodic turbulence
with RMS Mach numbers approximately 0.3 and 8.4, and correlated forcing at
wavelengths between half and one box length. Begin with a 2D adaptation, then
establish the corresponding 3D setup. The paper's 3D spectra are a later
reference; they are not acceptance curves for the 2D adaptation.

### Forcing and thermodynamics

1. Implement a periodic Fourier acceleration field with an Ornstein–Uhlenbeck
   time process. Expose the seed, driven modes, amplitude, correlation time and
   solenoidal/compressive mixture. Solenoidal acceleration does not constrain
   the compressible fluid velocity to be divergence free.
2. Define the forcing process on physical time. Repeated RHS evaluations,
   warmup, rejected work and diagnostic callbacks must not draw a new forcing
   realization. Save mode coefficients, forcing time and random-generator state
   for exact restart. A fixed forcing-update clock with defined interpolation
   is one candidate implementation.
3. Add stage-consistent momentum and total-energy sources to the conserved
   meshless update, including the work of the applied force. State whether net
   acceleration is removed, and use the actual particle masses when measuring
   momentum and power, including MFV's variable masses.
4. Choose an explicit isothermal formulation or a separately named thermal
   model with controlled cooling. Match pressure, sound speed, Riemann states
   and energy treatment consistently. The present ideal-gas energy relations
   contain gamma minus one; assigning gamma equal to one is not an implementation
   of an isothermal compressible solver.
5. Record forcing power, thermal removal and all intentional sinks separately.
   Adiabatic driven gas heating, isothermal removal and finite-time cooling
   define different physical problems and must have distinct presets.

Start with the subsonic case, then increase to the supersonic target using the
same declared numerical policy. Check cold-gas thermal accuracy, positivity,
large density contrasts and timestep collapse explicitly. Preserve failed runs;
do not repair them through undisclosed floors, forcing changes or CFL retries.

### Two dimensional dynamics and acceptance

Decide whether the target is a finite-time developing flow or a statistically
stationary state. In 2D, transfer towards the largest scales can accumulate
energy in the box. If an explicit large-scale sink is needed, define and measure
it as part of the model. Do not silently add drag merely to hold a target Mach
number, or assume the 3D cascade applies to the 2D run.

Use a reproducible, declared startup and perturbation. Distinguish changes to
initial conditions from changes to the driving process. First verify the source
update on simple uniform or prescribed-force states, then qualify the turbulent
case with:

- Mass and momentum budgets, positive density/internal energy, finite RHS and
  accepted states, and energy balance including forcing and thermal terms.
- RMS Mach history with an explicit mass- or volume-weighting convention,
  density/pressure PDFs, velocity and density spectra, structure functions and
  correlation times. Define particle-to-grid deposition and spectral
  normalization so interpolation noise is not mistaken for a physical cascade.
- Multiple resolutions and independent seeds. Compare converged statistics and
  resolved scales, rather than requiring identical late chaotic trajectories.
- Repeated runs and checkpoint/restart tests that preserve the same forcing
  process. Exclude startup transients from any claimed stationary dataset.

Proceed when the source terms and thermal closure pass their simple controls
and the 2D flow has a documented, physically interpretable statistical regime.
Turbulent-looking images alone do not establish this gate.

## Stages B and C Local feasibility and GH200 campaigns

Use the local GPU for substantial feasibility runs, not only small routing
checks. Increase the 2D particle count, integration duration and ensemble size
until the actual cost and memory envelope are known. Add modest 3D pilots before
planning a production 3D campaign: a successful 2D run does not qualify a 3D
cascade, density distribution or magnetic dynamo.

Treat the GH200 allocation as having the **same per-GPU VRAM ceiling** as the
local machine. Its hardware may change throughput and the dominant bottlenecks;
it does not provide a larger device-memory budget for this plan. Choose larger
HPC work through measured combinations of resolution within that limit, physical
duration and independent trajectories. Multiple independent jobs can expand the
ensemble; a single problem exceeding one device's budget needs a separately
scoped distributed/domain-decomposition capability.

Record the following for each feasibility point:

- Actual particles and neighbour counts, dimension, precision, kernel, support,
  method, forcing and thermal parameters. Retain n_h=4 unless a separate,
  explicitly identified support study justifies another choice.
- Peak device residency including Warp allocations, adjacency, geometry,
  reconstruction, all temporal stages and diagnostic scratch. Torch's allocator
  statistics alone are not a complete device-memory measurement. Leave measured
  headroom for transient allocations and density clustering.
- End-to-end seconds per unit of physical time, timestep distributions and
  limiting criteria, plus setup/compilation, stepping, diagnostics and output
  costs separately. Short step-count throughput is insufficient for a chaotic
  run with an evolving CFL limit.
- Output volume, host/device transfers, checkpoint size and restart time.
  Profile forcing-mode evaluation and synchronization at scale; do not assume
  that the CPU noise interpolation in the orthogonal Kolmogorov cases is an
  appropriate implementation for large compressible runs.

Keep native physical timestep limits on the first and every subsequent step,
and cap the endpoint. Do not lower support or alter the forcing spectrum simply
to make the memory benchmark fit.

Before a large HPC submission, reproduce a qualified pilot on the GH200 with
pinned sources, configuration, precision and seeds. Check Torch/Warp/CUDA and
compiled-extension compatibility, restart portability, device residency and I/O.
Require budget consistency and matched statistics; cross-device bitwise identity
of late chaotic states is not the acceptance criterion. Package immutable input
manifests, sources, forcing state, checkpoints and the measurement protocol with
each campaign. Offloading or unified-memory spill is a separate measured study,
not assumed capacity for a device-resident production run.

## Stage D Thermal Rayleigh–Bénard convection

Use a fully compressible ideal-gas formulation with defined physical viscosity,
thermal conductivity/diffusivity and heat capacity. Build a hydrostatic,
conductive reference state under gravity, then choose the superadiabatic driving,
thermal boundaries and velocity boundary conditions. Specify Rayleigh and
Prandtl numbers, stratification and the thermodynamic scaling.

The current equal-pressure dense-square contact has no gravity. It does not
validate hydrostatic gravity balance. Artificial shock conductivity likewise
does not establish a calibrated physical heat flux.

Qualify diffusion against a simple conduction solution, preservation of a stable
stratification, and gravity/energy work before convection onset. Then test onset
for the selected boundaries, heat-flux balance and Nusselt diagnostics. Resolve
thermal and viscous boundary layers before extending to strongly chaotic
convection. Keep any Boussinesq/WCSPH comparison on its own named physical path.

## Stage E Interface and shock instability cases

Revisit Richtmyer–Meshkov with a short audit of the incident shock, transmitted
and reflected states, interface amplitude and growth, material treatment, and
boundary contamination. The current case allows selected SPH schemes and uses
a padded periodic domain with a causal observation limit. Extending its MFM/MFV
routes or its useful nonlinear observation window is explicit work.

Use triple-point interactions as a complementary case for shocks, slip lines
and roll-up. Qualify sampling choices, symmetry, conservation, interface
measurements and the time window before broad parameter sweeps. Neither case
becomes training-grade solely by completing an endpoint.

Add conservative tracers or mass fractions when the intended labels represent
material mixing. Initial particle labels alone do not track conservative tracer
transport under MFV mass exchange. Distinguish geometric interleaving from a
specified physical mixing/diffusion model.

## Stage F Open boundaries supersonic objects and injection

Treat inlets/outlets as their own qualification effort. The existing bow-shock
case moves a cylinder through a closed gas box, providing a finite transient
starting point. Sustained flow past a stationary object requires consistent
inflow states, outflow/reflection control and a domain that does not feed
boundary artifacts back into the interaction.

Begin with uniform advection and acoustic/shock exit controls. Then qualify
stationary and moving objects using shock position, stagnation pressure,
wall penetration and force/work budgets. Separate startup transients from any
claimed sustained wake or statistically stationary flow.

For injection, first define whether the target is a gas clump/jet or a physical
liquid droplet in gas. Liquid breakup requires appropriate material EOS,
surface tension and physical transport as well as a validated injector and
mass/momentum/energy accounting. Prescribe the relevant Reynolds, Mach, Weber
and density-ratio parameters where applicable. Build the simplest injector and
single-object controls before interacting droplets or complex geometries.

## Stages G and H MHD and magnetic driving

Start with ideal MHD as an independent extension of the conservative compressible
architecture. Add magnetic induction, magnetic stress and energy, appropriate
MHD Riemann fluxes, fast-wave timestep limits and divergence control. Carry the
magnetic and cleaning state through every temporal stage and checkpoint.
Hopkins and Raives provide a directly relevant MFM/MFV formulation; divergence
control beyond Powell transport alone is central to its qualification.

The initial test sequence is zero-field hydro recovery, linear Alfvén and
magnetosonic waves, magnetic shock tubes, field-loop advection, rotor and
Orszag–Tang. Check positivity, conservation/source budgets, field diffusion and
dimensionless divergence errors before long chaotic runs. Add difficult
low-beta states only after simpler controls work.

First reuse the periodic mechanical forcing with a seed or mean magnetic field.
Measure kinetic/magnetic spectra, energy transfers, amplification and saturation.
Separate this from genuinely magnetic driving: an externally imposed induction
source, reconnection-powered flow and MRI/shear each need their own equations,
energy supply and boundary setup. Specify which mechanism is intended before
choosing its case. Add explicit resistivity when claiming controlled magnetic
Reynolds or magnetic Prandtl numbers; numerical dissipation alone does not set
those physical parameters.

## Stage I Rayleigh–Bénard magnetoconvection

Combine qualified thermal convection with the tested MHD solver. Specify field
orientation and strength, magnetic boundary conditions, transport coefficients
and dimensionless controls. Preserve the zero-field convection limit and test
magnetic modification of onset and heat flux before entering chaotic regimes.
Track magnetic, kinetic and thermal transfers together with divergence errors
and boundary power/heat flux. Stratification, thermal walls and magnetic walls
must work in combination, not just in isolated tests.

## Training data requirements

Save conservative and primitive fields with particle identities, geometry and
physical time. Include material/tracer fields where appropriate, the applied
forcing or its sufficient state, and all thermal/magnetic source parameters.
Choose whether the learning target is particle-based or grid-based; if both are
exported, retain the deposition operator and its conservation/error checks.

Use independent seeds and parameter families. Split training, validation and
test sets by whole trajectories or independent realizations, rather than random
adjacent frames. Report temporal correlations and sampling cadence. Preserve
source/configuration hashes, precision, solver/dissipation choices, resolution,
restart state and quality flags with every trajectory. Treat different thermal
closures, sinks and forcing mixtures as different conditional problems.

Require statistical convergence, trustworthy budgets and adequate resolved
scales before labelling output as reference data. Use explicit physical
transport where controlled Reynolds numbers are needed; otherwise identify the
dataset's numerical-dissipation regime. Preserve failures and unqualified runs
as evidence outside the accepted training cohort.

## References and existing entry points

- [Hopkins 2015 section 4.4.4](https://arxiv.org/html/1409.7395#S4.SS4.SSS4),
  driven subsonic and supersonic turbulence and dimensional differences.
- [Hopkins and Raives ideal MHD](https://arxiv.org/abs/1505.02783), meshless
  magnetic fluxes, divergence control and validation problems.
- [Verhoeven et al compressible convection](https://arxiv.org/abs/1501.01237),
  fully compressible and anelastic Rayleigh–Bénard formulations.
- [Compressible comparison](../compressible_matrix/README.md) and
  [Godunov qualification](../compressible_matrix/godunov_shock_audit_2026-10-24.md).
- [Meshless methods and boundaries](../meshless.md).
- [Meshless stage RHS](../../src/warpSPH/schemes/compressible/meshless/step.py),
  [Richtmyer–Meshkov](../../src/warpSPH/cases/compressible/richtmyerMeshkov.py),
  [triple point](../../src/warpSPH/cases/compressible/triplePoint.py) and
  [bow shock](../../src/warpSPH/cases/compressibleWalls/bowShock.py).
- Orthogonal classical-fluid cases:
  [WCSPH Kolmogorov](../../src/warpSPH/cases/periodic/kolmogorov.py) and
  [incompressible Kolmogorov](../../src/warpSPH/cases/periodic/kolmogorovIncompressible.py).

Keep temporary drivers, measurements, trajectories and generated reports under
Git-ignored .tmp directories. Commit supported reusable functionality, focused
regressions and maintained documentation as each stage is qualified. This
roadmap records future scope; HPC jobs and solver implementations are separate
execution tasks.
