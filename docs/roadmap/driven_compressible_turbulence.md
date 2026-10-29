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
Offline mass/volume statistics, conservative observation filters, explicit
stationarity screens, native checkpoint context and complete per-frame driving
history are implemented. Cold-gas controls exposed and now guard the explicit
force predictor with an available-energy timestep limit. Long-flow studies are numerically healthy while
stationary/reference-data precision and resolution convergence remain open.
Exact-isothermal dynamics, larger/3D feasibility and HPC production qualification
also remain open. Later stages below retain their own gates.

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
  Use same-forcing perturbed twins when distinguishing intrinsic sensitivity
  from variation caused by the random driving itself.

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

Thermal-boundary work (2026-10-27): the [initial audit](../thermal_boundary_audit_2026-10-27.md)
identified initialization/accepted-u and entropy-closure failures. The
[CPU-qualified repair](../thermal_state_qualification_2026-10-27.md) now retains
prescribed u, supports geometry-independent meshless u, and records restartable
projection exchange. The meshless RT case adapter, physical conductivity,
conducting/insulated walls and physical viscosity remain open.
[Reference scaffolding](../thermal_transport_scaffolding.md) defines caloric
units, distinct c_v/c_p diffusivities, analytic conduction profiles and signed
face heat budgets for the next small controls. No physical thermal-wall model
has been promoted.

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

Independent [MHD references and implementation contracts](../mhd_scaffolding.md)
are now available; production magnetic state, fluxes and cleaning remain open.

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

## Performance investigation (2026-10-27)

User requested profiling and tuning MFM/MFV before longer driven campaigns.
Separate setup, fresh SSPRK spatial stages, timestep, accepted-state refresh,
forcing, diagnostics and export. Use short sequential windows from startup and
saved weak/strong flow states, and compare both methods at the same physical
input. Preserve CFL, support, thermodynamics, stage semantics and safety checks.
Profiler counts explain bottlenecks; instrumented or GPU-contended wall times
are not throughput rankings. Require repeatable untimed field/budget agreement
and relevant regressions before adopting optimizations. The existing long-run
qualification wall times include observers and contention.

Performance pass: [report](../driven_turbulence_performance_2026-10-27.md).
Opt-in exact-neighbor caching reduces observed shared-GPU step wall time by
18–28% across repeated 64²/128² weak/strong MFM/MFV windows. The original
search remains default; uncontended throughput and long cached-flow
statistics remain open.

## Stage A scheme extension (2026-10-27)

Implement CRKSPH and CompSPH compatible source work, plus fixed-mass
compressible adapters where the same interface applies. Preserve each native
Runge–Kutta tableau and hydro/viscosity formulas. Establish uniform-force,
thermal-only, combined source, temporal-refinement, positivity and restart
controls on CPU while GPU contention continues. Keep this implementation
qualification distinct from long chaotic statistics and performance rankings.

Scheme-extension result: [CPU implementation qualification](../driven_turbulence_scheme_support_2026-10-27.md).
Five native SPH routes now share the driver/thermal model, with compatible
source work, exact tested CPU checkpoint continuation, recorded developing
pilots and timestep controls. GPU and long statistical qualification remain
separate gates.

## Thermal-boundary audit decision (2026-10-27)

User requested checking all schemes against RT-style Dirichlet thermal bands
while GPU work remains contended. Audit completed on CPU in both precisions;
the findings and implementation order are linked in Stage D. Solver changes
and thermal-convection qualification are subsequent work.

## Small thermal controls (2026-10-27)

Measure the audited gaps with small CPU problems: uniform prescribed heating
through stage/accepted state, a frozen-lattice thermal Fourier mode for the
existing artificial conductivity, and an expanding pressure reservoir. Record
explicit unsupported physical-diffusion/convection gates rather than treating
a fitted artificial diffusion rate as calibrated physical heat transport.

Results: [small thermal controls](../thermal_boundary_toys_2026-10-27.md).
184 CPU diagnostics (92 per precision) and nine tests per precision complete.
The initial boundary-target defects persisted under timestep/tableau refinement;
the subsequent state repair is recorded below. Artificial heat-operator response
varies with resolution/amplitude and some formulations switch off at equal
pressure. Physical slab/insulation/gravity gates still need a dimensional
transport operator and their case adapters. No transport model was promoted.

## Thermal implementation investigation (2026-10-27)

Prioritize the measured thermal constraints and exchange budgets before further
turbulence runs. Start with initialization and accepted-state enforcement,
then isolate entropy/geometry conversion and dimensional diffusion. Preserve
native fresh integration stages and qualify each change on the small controls
before any larger coupled convection run.

Thermal-state result: [repair and CPU qualification](../thermal_state_qualification_2026-10-27.md).
Prescribed u now survives startup/acceptance, PESPH entropy is inverted against
its closure, meshless bands can prescribe u independently of volume, and
projection exchanges survive replay/restart/output. CPU toys and short coupled
controls pass; no-prescription parity is bitwise in nine saved controls.
Physical heat transport/wall fluxes, calibrated physical viscosity and GPU/long
convection qualification remain open. Turbulence campaigns remain deferred.

## Thermal reference scaffolding (2026-10-27)

Before physical transport, add independent periodic/insulated/plate conduction
oracles, explicit caloric units and signed face-flux energy references.
[Scaffolding and measurement contract](../thermal_transport_scaffolding.md):
22 CPU oracle checks pass; no new solver physics or GPU campaign. The stationary
fixed-density controls must be mechanically frozen, and convection's common
c_p diffusivity must be distinguished from the c_v heat-equation diffusivity.

## MHD reference scaffolding (2026-10-27)

[MHD scaffolding and contracts](../mhd_scaffolding.md) now supplies independent
magnetic-energy/finite-source-work, wave-speed, Alfvén-wave, vector-potential,
force-free resistive-decay and divergence-summary references; 26 CPU checks pass.
The audit identifies hydro conserved-vector slicing and spatial/vector dimension
coupling as prerequisites for magnetic integration. State/restart, divergence
control, magnetic driving, resistivity, wall and magnetoconvection contracts are
recorded. No production MHD algorithm or scheme qualification is introduced;
turbulence campaigns remain deferred.

## Overnight GPU qualification resumed — 2026-10-27

User explicitly authorizes these previously deferred runs from the ACSPH chat:
GPU source/restart controls, isolated MFM/MFV search timing, then common seed1234
weak-driving trajectories to40 at64²/128² for MFM, MFV, CRKSPH, CompSPH, Monaghan,
GSPH and InutsukaGSPH. All use the existing native methods and forcing/thermal
settings. Runs follow the corrected Lobovsky ACSPH replay in one persistent
serial queue, with separate frozen runtimes and no shared solver edits.
The physical source snapshot is8832040 plus the recorded dependency sources.
One seed and two resolutions do not qualify stationary/ensemble convergence.
[Campaign, exact inputs and live status](/home/lu26029/dev/warpSPH/docs/acsph/OVERNIGHT_2026-10-27.md).

This supersedes the earlier turbulence deferral for the declared GPU campaign;
physical transport and production MHD remain separate open implementation work.

User ordering correction: ACSPH → all14 turbulence comparisons/analysis → GPU source/restart checks and timing controls. GPU checks assess the recorded cohort afterwards; their later placement does not certify passing methods in advance. Physical guards remain active. The live ACSPH trajectory was retained without restart.

Overnight completion reviewed2026-10-28: all14 weak-driving trajectories and
analyses reached40; all1,108,534 observed RHS states were valid/positive. Twelve
MFM/MFV timing windows finished. Fifty GPU source controls passed directly;
five export assertions initially compared CPU and CUDA tensors. A separate
test copy moves only the expected tensor to CPU with zero tolerances unchanged;
all five then pass. Preserve original failures, source pins and trajectories.
The ACSPH campaign's `completion-review.json` reconciles the original supervisor
status with this postflight evidence. Stationary/ensemble qualification remains
open. No physical solver changes or new long trajectories in this review.

Independent [cohort analysis (2026-10-28)](../driven_turbulence_overnight_analysis_2026-10-28.md):
all fourteen runs/554,267 steps and 2,106 pinned files reconcile. All viable,
none passes the stationarity screen; single-seed resolution differences remain
large. Short launch-screened neighbor-cache windows save19–21%, still opt-in.
Resolved GSPH uses c=0 and Inutsuka c=3; these do not relabel the matrix variants.
No new trajectories, physical changes or default promotions in this analysis.

## CRKSPH resolution extension (2026-10-28)

User authorizes192² and256² weak driven runs to t=40, seed1234, preserving
all prior physical parameters, midpoint/CFL0.15, B7/n_h4 and source/positivity
guards. Run serially with recordings from the frozen overnight runtime; compare
physical-time cost and means over10..40 with64²/128². One seed and the existing
nonstationary window limit any convergence interpretation.
Raw manifest/status: `.tmp/crksph-driven-resolution-20261028/`.

CRKSPH extension [results](../crksph_driven_resolution_2026-10-28.md): both new
endpoints pass (173,932 steps/347,864 RHS checks).192² costs13.50min and256²
23.47min, at11.36/13.73ms per step. Means over10..40 are Mach0.6296/0.7162
and K0.1977/0.2549. The four-resolution trend is non-monotonic; none passes
stationarity. Sources/forcing match, no numerical defaults change, and no new
campaign is started. The native primitive-energy reference resolves a tiny
192² initial diagnostic-Q rounding offset in the offline harness.

## Forcing/domain audit (2026-10-28)

Test the suspected shared cause directly: periodic value/gradient matching,
independent Fourier evaluation and projection, fixed-clock knot behavior and
realized forcing power in the retained runs. Separate numerical seam defects
from properties of acceleration-driven2D periodic flow. Preserve the physical
model and raw trajectories while diagnosing; no long reruns initially.

Forcing audit [results](../driven_forcing_audit_2026-10-28.md): all16 retained
OU histories match. Independent CPU FP32/FP64 value/Jacobian checks find only
roundoff at periodic seams; 15 field/OU controls pass. Unwrapped SPH positions
increase phase roundoff, with sampled instantaneous power errors at most3.02e-8.
The six low modes directly force the box; stochastic acceleration does not fix
input power (CRK192²0.0160 versus256²0.0348 over10..40). This does not isolate
forcing from numerical dissipation or finite-window variability. Production
sources stay unchanged; no GPU trajectory launched during contention. Next
controls: CUDA field evaluation when idle, then a declared forcing-band A/B
and additional seed before treating resolution means as a convergence trend.

## Forcing-band control deferred behind ACSPH (2026-10-28)

User authorizes this control after the other session's ACSPH run finishes.
The identified queue is `.tmp/acsph-lobovsky-ramp05-2026-10-28/` in the primary
checkout (supervisor3509247, physical child3558862 at scheduling time).
A thread heartbeat checks hourly (user revised the interval2026-10-28), confirms terminal status/logs and a free
GPU before allocation, and prevents duplicate launches. It must not interrupt
or message the other session.

Prepared contract: `.tmp/driven-forcing-band-20261028/experiment.json`.
Run one new CRKSPH128² seed1234 trajectory to40 with band2–4, compared with
the frozen overnight band1–2 baseline. Only kMin/kMax and output destination
change; all other source, thermal, integration and observation choices match.
The same seed with different mode count does not imply identical coefficient
draws. First close the CUDA field-check gap; retain video, progress, raw-state
and energy guards. Compare block input/bath power, K, Mach and matched-filter
box-mode energy over10..40. No extra seeds or drag/viscosity/feedback sweep in
this authorization. Automation `run-forcing-band-control-after-acsph` reports
actionable progress/completion and disables itself after delivery.

2026-10-28 launch: ACSPH ramp05 and its export/comparison jobs all passed;
queue terminal10:56UTC, verified absent supervisor/children and no GPU users.
CUDA field controls passed for both bands in FP32/FP64 (highest matmul
precision, TF32 off). New-band seam acceleration max3.66e-7, gradient5.25e-6;
retained CRK256 endpoint power error1.25e-8. Source-pinned serial supervisor
PID3808217 started11:16:59UTC; physical CRK128² child3808396 follows the passed
field preflight. The supervisor is persistent so this finite experiment survives
a chat/session restart. No parallel GPU job or other-session modification.

Forcing-band control [complete](../driven_forcing_band_control_2026-10-28.md):
CRK128² modes2–4 reaches40,46,334 steps/92,668 valid RHS checks, max relative
energy residual9.60e-8. Cost7.81min/10.11ms per step. Mean K0.1418 versus0.2230,
Mach0.5365 versus0.6735, mean input power0.01215 versus0.03171. Direct |n|=1
acceleration vanishes but its matched filtered velocity-energy fraction remains
76.60% (baseline84.89%). Both fail stationarity; box-scale dominance does not
require direct box-mode forcing in this realization. CUDA field controls pass;
all2,111 pinned files unchanged and initial fluid arrays bitwise equal. No
extra seed, trajectory, transport/feedback change or default promotion. Next:
discriminate scale-transfer, compression and numerical dissipation before
claiming stationary/converged2D training data. One-shot heartbeat is disabled
on delivery of this comparison.

## Shared2D/3D scale-energy diagnostics (2026-10-28)

User endorses the next diagnostic/qualification work and its reuse for3D.
Implement dimension-generic periodic Favre coarse-graining with separate
resolved kinetic energy, deformation/baropycnal transfer, pressure dilatation
and forcing work. Validate signs, dimensions, Galilean invariance, analytic
pressure work and inviscid filtered-budget closure in manufactured2D/3D fields
before applying it to saved runs. Use conservative observation deposition and
explicit spatial/time sampling sensitivity; an unclosed residual must not be
labeled measured physical viscosity or uniquely numerical dissipation. Preserve
native dense work/bath budgets, thermodynamics and raw trajectories. Document
the current implicit dissipation model and later physical transport choices;
no production solver, forcing, drag or viscosity change and no new campaign.

Shared scale-budget [toolchain results](../turbulence_scale_budget_2026-10-28.md):
2D/3D Favre core, conservative observation adapter, CLI and plots implemented;
18 CPU controls pass, including a complete synthetic3D retained-state fixture.
Both real401-frame2D controls analyzed on32²/64² grids, two observation radii,
three Gaussian widths and time-decimation controls. At ell0.125,H0.0625,
mean total transfer is-0.001039/-0.003326 (bands1–2/2–4), but observation filtering
and residuals remain consequential. Baseline deformation/baropycnal terms nearly
cancel; a velocity-only flux would give the wrong total-transfer sign there.
Snapshot forcing-work quadrature differs from native dense work by-1.97%/+0.52%.
The qualification probe now exposes sampling cadence and both band bounds,
retaining default0.1 and1–2. Accepted-time replay exactly reproduces all default
samples and verifies a finer clock without imputation or timestep changes.
No new flow, GPU allocation, physical viscosity/drag, stationary qualification
or production3D claim. Next: controlled finer-cadence measurement and native
work/stress accounting before attributing an unclosed residual to dissipation.


## Forcing and observation cross-check continuation (2026-10-28)

User requests continued investigation and a different force with a known
answer. Before attributing residuals to native dissipation, measure accepted-state
instantaneous input power and cadence sensitivity in one unchanged CRK128
seed1234 band1–2 replay, preserving all native stages/CFL/physics and checking
trajectory parity against the retained baseline. Retain finer physical snapshots
and compare decimated ledgers. Independently derive and test a prescribed
transverse periodic shear: v_y=A sin(omega t) sin(kx),
a_y=A omega cos(omega t) sin(kx), uniform rho/P/u. Advection and dilatation
vanish, so K=M A² sin²(omega t)/4 and external work equals delta K.
Use source-only controls to isolate force integration, then native hydro to
measure additional shear error/heating. This is a local analytic verification
problem, not a replacement turbulence model or a stationary viscous Kolmogorov
reference. Inspect missing observation-stress terms separately; no new drag,
physical viscosity, forcing feedback or default numerical changes.


Continuation [results](../forcing_crosscheck_2026-10-28.md): analytic oscillating
shear and observation-stress controls pass33 CPU checks. Four native shear
controls (CRK64/128, MFM64, MFV64) finish with maximum velocity errors below0.47%
of exact peak RMS; CRK refinement improves0.400%→0.114%. One CRK128 band1–2
measurement replay is bitwise identical in all49,551 accepted states, initial/
final arrays and forcing checkpoint. Dense accepted power reduces the old
1.97% work discrepancy to0.000834%;0.0125 snapshots give0.00655%. Missing
observation stress explains51%/89% of original narrower-radius residuals in
bands1–2/2–4 and nearly removes radius dependence. At ell0.125, finer baseline
sampling plus stress reduces the residual from+0.000690 originally to−0.0000156;
matched32²/64² grid residual difference5.5e-9. No forcing defect exposed, no
solver/OU/default change, and no stationary/inertial-range certification.
Keep accepted work and second moments in future2D/3D output; remaining native
transport/pressure/stencil effects require separate attribution. Replay9.35min,
11.32ms/step; four short shear loops total60.06s,10.82–12.44ms/step.


## Long-duration stationarity campaign (2026-10-28)

User authorizes one longer CRK128 run and two additional independent seeds.
Run seeds1234/5678/9012 from the shared resting IC to t160 with the exact frozen
runtime used in the verified replay. Keep all physical and native numerical
choices; retain0.0125 accepted-state snapshots, every-accepted-step source power,
video/progress, raw/accepted positivity and source guards. Jobs run serially and
check for unrelated GPU compute before launch. The finite tracked supervisor
survives chat/session changes, detects stalled physical progress and saves
terminal results/failures without numerical retries. Source/manifest/status:
`.tmp/driven-stationarity-20261028/`.

Predeclare primary blocks40–80/80–120/120–160 and retain startup10–40 plus
20-unit subdivisions. Compare K/Mach/density statistics, distributions and
spectra, absolute box-mode energy and fraction, power/bath/energy balances,
correlation times and uncertainty. A failed10% screen is insufficient evidence
of persistent nonstationarity; ACF/uncertainty for drifting records is diagnostic,
not certified independent sampling. Assess all seeds, retaining failures and
between-seed variation; no default/model or stationarity promotion is implied.

Launch validated: first seed passes the old t40 horizon with a bitwise baseline
prefix before its endpoint cap; raw states remain valid. CPU analysis companion
and quiet hourly completion heartbeat are active. Six statistical/failure-guard
controls pass. [Live report](../driven_stationarity_2026-10-28.md).


Long campaign terminal analysis (2026-10-28): Long CRK128 t160 campaign: 3/3 analyzed, scalar screens 0/3; [report](../driven_stationarity_2026-10-28.md). Distributional/ensemble stationarity requires interpretation.


Long-campaign interpretation complete (2026-10-28): all3 seeds numerically pass
to160,595,572 steps/1,191,144 valid RHS, no extra RHS or source changes. Total
loop108.72min,10.92–10.99ms/step. K late blocks are0.252/0.154/0.298,
0.206/0.230/0.251,0.284/0.279/0.218 (seeds1234/5678/9012): no common monotonic
rise, but0/3 stationarity screens pass. Slow coherent box modes carry83–89%
of velocity energy; their fractions pass while absolute energy varies.
Whole-window field ACF gives only6–11 effective samples; K-window confidence
is unavailable in7/9 windows under the declared block requirement. Ensemble
block means0.247/0.221/0.256 have substantial between-seed variation. Fixed-clock
force RMS remains0.290–0.305 while input work fluctuates much more. All4803
analyzed snapshot hashes and2108 executable pins match; representative videos,
PDF/spectrum and ensemble plots reviewed. Data remain finite-window evolving
trajectories, not qualified independent stationary training snapshots. Keep
archived fine states for mode-work/transfer analysis before choosing a separate
physical transport/sink/3D study. [Final report](../driven_stationarity_2026-10-28.md).

Final integrity/visual review saved; completion heartbeat paused to prevent repeats.


Late-flow interpretation check (2026-10-28): user observes an almost unchanged
pattern drifting during t120–160. Separate rigid periodic translation/amplitude
changes from internal deformation in the archived seed1234 flow; compare the
box-mode velocity and remaining velocity/vorticity using the same fixed filter.
CPU-only analysis, no new trajectory or model change.

Late-pattern check complete: seed1234 t130→160 is84.7% explained by a periodic
shift plus one gain in filtered velocity variance (correlation0.920). Persistent
box-scale bands dominate, with evolving residual/vorticity. t120 is a distinct
transition; reference dependence and optimizer flags are retained. Slow video
and decomposition saved; field decorrelation from pattern drift is insufficient
to establish structural training-data diversity. No new physical run.


## Proposed 96³ feasibility envelope (2026-10-28)

User asks whether96³ is a practical next pilot with sparse particle output.
No new trajectory is authorized or launched by this capacity discussion.96³
is884,736 particles. The local RTX PRO6000 Blackwell reports97,887MiB total
VRAM. Capacity is plausible but unmeasured: with initial support radius4dx, a
uniform3D sphere contains about268 neighbours versus about50 in2D; pair
storage, CRK correction/stage scratch and evolving clustering prevent a simple
particle-count extrapolation from the measured256² cost. The existing case
explicitly restricts dim to2 and constructs FourierOU(dim=2); the source module
itself supports3D. Extend case/driver/visualization and qualify the3D source and
restart path before a short, watched GPU capacity/throughput probe. Do not quote
a long-run wall-time forecast until measured milliseconds per step and physical
time advancement are available. Preserve B7/n_h4 and native timestep policy.

For the current qualification snapshot fields extended to3D, float32 payload
is64bytes/particle:56.62MB/frame before compression (metadata negligible).
At0.0125 intervals, t40 would retain3,201 frames/about181GB; t160 would retain
12,801/about725GB. Position/velocity/density/internal-energy fields alone need
28.31MB/frame. Compression and real I/O cost require measurement.
Prefer accepted-step scalar source/bath/conservation guards, online3D spectra
and filtered budgets at a measured affordable cadence, sparse full states and
restart checkpoints, and slice/projection video. For example0.25 full-state
spacing to40 needs about9.12GB uncompressed; it does not resolve all temporal
statistics or qualify training cadence. Use short dense recording bursts for
measurement checks instead of repeating the2D dense archive throughout.


## Authorized 3D performance pilot (2026-10-28)

User authorizes serial CRKSPH32³,64³ and96³ runs to t4 with full exports every
0.25 simulated seconds. Extend the case/qualification driver to3D while
preserving all physical/numerical inputs: gamma5/3, OU band1–2 RMS0.3/tau0.5/
knot0.05, seed1234, solenoidal mass-mean removal, u0.9 bath/tau0.2, B7/n_h4,
native fresh midpoint/CFL0.15. Perform CPU3D periodic/projection/Parseval/source
controls before CUDA; slice video, live progress and raw/accepted guards remain
on. Record source/input hashes, peak Torch/device residency, step and simulated-
time throughput, export volume and positivity/conservation. Wait for unrelated
GPU work to finish; do not interrupt it. This short startup campaign measures
capacity/performance and does not qualify stationary turbulence or convergence.


3D pilot queued under persistent supervisor809027, source-pinned
`.tmp/driven-3d-pilot-20261028/manifest.json`; no CUDA allocation while
ACSPH713827 is active. Exact frozen3D CPU controls pass7/7 (19 initial working-
runtime controls pass); packaging failure repaired/retained before launch.
Automatic CPU audit/report follows the three finite serial runs.
[Pending performance report](../driven_3d_pilot_2026-10-28.md).


3D pilot review (2026-10-29): all three trajectories reached t4 with valid
raw/accepted states, no extra RHS and stable sources. CPU analysis did not
launch because the supervisor used duplicate keyword environment merges.
Original failure/source/status retained; repair only offline analysis and
complete recording/provenance/video audit. No new CUDA allocation or trajectory.


3D pilot terminal: **3D startup pilot terminal:**3/3 endpoints audited; [performance report](../driven_3d_pilot_2026-10-28.md), runtime/memory/exports recorded. Video review and3D restart continuation remain pending; no stationarity claim.


3D pilot completed/audited2026-10-29:32³/64³/96³ all reach t4, loop35.12/
376.19/1919.04seconds (35.65/187.07/631.26ms/step); sampled process peaks
2.56/14.99/48.01GiB.51 particle exports and12,072 RHS valid, no extra RHS,
energy residual≤2.29e-8; sources/configuration/common OU histories match.
Representative slices and history plots reviewed.96³ snapshots705.44MB;
I/O22.80seconds/1.19% of loop. Fits the current VRAM limit;~8wallmin per
simulated second is a startup measurement. Original supervisor environment
merge failure retained; only offline analysis repaired, no GPU reruns. Complete
3D restart continuation and long/production physics remain open.
[Final report](../driven_3d_pilot_2026-10-28.md). No larger run launched.


## Authorized CRK memory ownership measurement (2026-10-29)

User requests full-state-once cost, exact physical versus Verlet neighbour-list
cost, temporary storage in one stage and retained data. Use the frozen3D
pilot runtime; count unique device storages and aliases, not only tensor sizes.
Record allocated/reserved/device memory separately, stage-local allocator
peaks and tensor ownership at stage/step boundaries, including X6 pairWork
and integrator-retained states. Begin with saved32³ native state and short
serial32³/96³ probes with existing physical model/fresh midpoint. A unit-
Verlet list is a separate read-only geometry measurement, not a trajectory
with changed neighbour policy. Profile instrumentation is excluded from
throughput claims; report actual probe cost and positivity/source guards.
No long trajectories or numerical-model changes.


Memory ownership audit complete2026-10-29:96³ stored particle fields80bytes/
particle (67.5MiB). Exact physical-support/Verlet-unpadded/Verlet-padded index
and row costs1.68/3.40/3.74GiB; stage ap/av/f outputs13.07GiB; X6 expression
transient11.20GiB. Full-root96³ startup peak34.493GiB is explained by allocation
events; it does not reproduce the historical t0–4 high-water42.10GiB. Found
RunResult.state retains its original system and5.601GiB first-step PairWork.work
until completion, distinct from the current X6 work vector. Stage/update-only
incremental retention is~326.5MiB. All13 short native steps/26 RHS checks valid;
4 CPU accounting controls, source/archive hashes and all videos/figure reviewed.
Observer root omissions repaired and original traces retained. No physical
solver change; obsolete result ownership and vector intermediates are next
implementation targets with numerical/lifecycle/graph parity gates.
[Report](../crksph_memory_2026-10-29.md).


## Authorized CRK shared force/work kernel (2026-10-29)

User requests a shared CRK pair kernel producing acceleration sums and compatible
W_ij without expanded ap/av/f arrays or Torch vector temporaries. Preserve native
forces, entropy estimator and fresh midpoint/RK stages. First establish the
partition dependency: native f uses half-step velocities that require completed
per-particle acceleration at that native point, including scheme forcing and
enforceUpdates. Case-local OU/bath sources still run afterward, as before. A fused path may
therefore require a force-only sweep then a shared pair-physics work sweep, not
an incorrect single sweep using incomplete acceleration. Source-paired CPU/GPU
operator/stage/trajectory controls, positivity/energy, multi-material/wall/replay
compatibility and memory/throughput measurements determine adoption. Keep legacy
path available for unqualified cases. Scope memory changes separately from the
RunResult retention fix; do not combine their benefits in a single comparison.


Shared force/work implementation2026-10-29: opt-in `crkPairStorage=recompute`
uses one force-only sweep and one register-local pair recomputation/work sweep
inside each fresh native RHS. Zero-weight stages omit work; weighted W updates
in place, changed-list replay reduces directly. Seven FP32/FP64 CPU controls
pass, including live W, replay, specialized-energy fallback and checkpoint
continuation. Independent old-source controls match all17 fields bitwise for
2D/3D midpoint and2D RK4;1,178 frozen source pins and348 dependency files match.
Expanded is still default. CUDA memory/runtime qualification waits for unrelated
compute to release the GPU. No measured GPU saving/speedup yet; separate obsolete
RunResult W retention remains unchanged. [Report](../crksph_shared_work_2026-10-29.md).


2026-10-29 user authorizes short GPU performance comparisons now, without full t4 trajectories. Source-paired frozen/refactored-expanded/recompute windows at32³ evolved checkpoint and96³ startup; matching recordings/guards, separate timing and allocator instrumentation. Keep physical policy and RunResult retention unchanged. Artifact root `.tmp/crk-shared-gpu-20261029/`.


Short CRK shared-work GPU comparison complete2026-10-29: two36-step windows
per path at32³ evolved/t4 checkpoint and96³ startup. Recompute versus expanded
32.27→35.43ms (+9.8%) at32³;559.99→535.20ms (−4.4%) at96³. Peak allocated
1.589→0.903GiB and34.545→16.006GiB (−43%/−54%). Separate3-step profiles confirm
ap/av/f13.07GiB→0 and X6 transient11.20GiB→0.073GiB.876 observed RHS valid,
no extra RHS, budget≤8.61e-9; sources/forcing pinned.96³ all endpoint fields/W
bitwise;32³ primary fields close but strict auxiliary-divergence screen fails
by a small FP32 amount and changed-list W samples are incomparable. Recordings
reviewed; copied profiler helper-path failures retained and instrumentation-only
recovery succeeds. Keep recompute opt-in, default expanded; no full t4 rerun,
no RunResult retention fix. [GPU report](../crksph_shared_work_gpu_2026-10-29.md).


2026-10-29 default follow-up: user questions opt-in solely for the small-case cost. Promote recomputation for qualified native eager compatible-energy contexts after focused default/fallback and accuracy controls; retain explicit expanded reference and automatic legacy routes for unqualified wall/graph/AD/material/experimental policies. Launch-overhead attribution is not established by the timing comparison.


Default adopted2026-10-29 (user preference): recompute is normal for native eager
single-gas compatible-energy steps; retain expanded reference/override and
automatic wall/graph/AD/material/experimental-policy fallbacks.77 distinct CPU
and26 GPU controls pass, including replay/Heun/support/thermal/checkpoint/late
differentiable-source routing. Recorded native Sod/Gresho existing accuracy
gates pass unchanged (269/1051steps;31.55/87.98s including compilation/video).
Strict32³ divergence screen remains failed historically; primary-state parity
and physical accuracy gates justify storage adoption without relaxing it.
No physics/timestep/precision or RunResult-retention change.
[Performance/default evidence](../crksph_shared_work_gpu_2026-10-29.md).


2026-10-29 user authorizes removing obsolete RunResult work ownership, while
deferring COO i-index elimination. Isolate the runner lifetime fix, test normal/
stop/replay/graph/pipelined lifecycles and bitwise source-paired short32³/96³
controls. Use Torch CUDA memory snapshots/memory-viz to distinguish live
blocks, stream-pending frees and inactive allocator cache; report matched-time
metrics and cache-release observations without per-step allocator flushing.


Obsolete RunResult ownership removal complete2026-10-29: accepted state now
replaces the original owner in ordinary and pipelined loops. Matched96³ peak
live16.006→10.333GiB; reserved11.355–13.396GiB observed after versus21.639
before, own-process peak14.186 versus22.479GiB in short windows. Fields/W and
all five paired forcing checkpoints bitwise.27 CPU controls and CUDA pipeline
ownership pass; graph capture failure also reproduces against sealed baseline,
so actual replay equivalence remains unavailable. Native Torch memory-viz
captures show inactive cache, no pending frees: post-run cache release leaves
3.790GiB inactive split blocks pinned within occupied segments. Brief unrelated
compute makes the ~1% timing difference inconclusive. No full t4 rerun, neighbour
representation change or allocator tuning. [Retention report](../crksph_retention_2026-10-29.md).


2026-10-29 user authorizes cross-scheme3D memory scaling after committing results
(commit931aca9). Planned seven supported schemes: CRKSPH control, CompSPH,
Monaghan, GSPH, InutsukaGSPH, MFM, MFV. Serial32³/64³/96³36-step startup
windows, independent warmup, then separate six-step96³ allocator/ownership
captures. Preserve native scheme/kernel/support/timestep policies, same physical
OU/bath setup; this measures implementation memory, not equal-error throughput
or developed-turbulence high-water bounds. GPU-free gate, progress/source/RHS/
positivity/energy guards, recorded slices and finite tracked supervisor.


Cross-scheme sweep prepared2026-10-29: all seven tiny CPU controls and CRK/MFM
capture-path controls pass after a meshless observer-only repair (failures
retained).28-job serial source-pinned GPU supervisor and finite CPU-only analysis
companion are tracked under `.tmp/compressible-memory3d-20261029/`. Initial
launch waits for unrelated ACSPH compute; no CUDA allocation into that work.
[Protocol/status](../compressible_memory3d_2026-10-29.md).


Cross-scheme3D memory sweep complete/reviewed2026-10-29:28/28 GPU jobs pass,
1,596 raw RHS valid, no extra RHS, source pins intact, all7 capture/native step6
stored fields bitwise.32³/64³/96³ scaling and all recordings reviewed.96³ live
peaks: CRK10.333, CompSPH28.791, Monaghan8.333, GSPH/Inutsuka8.316,
MFM/MFV10.533GiB. CompSPH retains expanded force/work transients; other SPH
peaks include overlapping lists, meshless peaks include multiple geometry/list
lifetimes. No pending frees in captures. Short startup viability/memory evidence,
not turbulent accuracy or evolved-state memory bounds. User allows future
accuracy/memory probes alongside other compute; label timing and shared-headroom
limitations separately. Existing queue finished before that policy update, no
reruns. [Results and limitations](../compressible_memory3d_2026-10-29.md).
