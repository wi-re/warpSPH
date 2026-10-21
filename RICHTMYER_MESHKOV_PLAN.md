# Per material equations of state and Richtmyer Meshkov instability

Progress and priorities: [PLANS.md](PLANS.md).

Status: implementation authorized and in progress, 2026-10-19.

The first deliverable is a reproducible two dimensional Richtmyer–Meshkov
(RM) case under CRKSPH, with two ideal gases that can have different
adiabatic indices. A material list will compile to a persistent Warp table;
each particle selects its EOS through `state.materials`. Completing this
requires changes to initialization, timestep evaluation, compatible energy,
configuration, and saved-state reconstruction as well as the EOS call.

## Scope and completion criteria

The initial scope includes CRKSPH, CompSPH and Monaghan with independently
configured ideal gases. CRKSPH is the acceptance target; the other two share
enough of the thermodynamic path to migrate and regression test together.
The material representation will carry an EOS type so that adding another
closure has a defined dispatch point. Supporting a new physical EOS requires
its thermodynamic conversions and tests, not just another pressure formula.

A completed first version must provide:

- A material list accessible from Python and the existing run configuration
  format, validated before stepping, serialized with the run, and restored
  when loading trajectories.
- Material-dependent pressure, sound speed, internal energy conversions and
  entropic function, consistent during initialization, every stage and CFL
  calculation. Material identity follows particles through integration.
- A stationary two-gas contact and a planar shock/interface interaction
  that pass quantitative checks before the perturbed case is accepted.
- A registered `richtmyerMeshkov` case, a small smoke variant, a small
  amplitude validation variant, and a visibly developing demonstration.
  Outputs include material and density video, pressure, interface growth,
  conservation histories, resolved settings, and performance.
- Evidence of resolution convergence and agreement with an independent
  reference for the same initial and boundary conditions. Reaching the
  requested end time and producing billows alone is not sufficient.

Species diffusion, molecular mixing, chemistry, temperature-dependent gases,
3D turbulent RM, reshock, and general multi-EOS Riemann solvers are separate
extensions. Weakly compressible/barotropic closures remain on their current
path in this iteration. Material labels describe pure Lagrangian parcels;
they do not supply a mixture EOS when gases mix below particle resolution.

## Existing implementation and implications

Paths below are relative to this repository unless marked otherwise.

| Area | Current implementation | Consequence |
|---|---|---|
| Particle identity | `systems/baseState.py`: `materials` is a constant particle field. Sod already uses distinct labels. `initializers/weaklyCompressible.py` enumerates fluid and boundary regions separately; boundary material IDs also identify rigid bodies. | Reuse the field for fluid EOS selection, preserve existing labels, and keep boundary identity separate from the fluid EOS contract. Do not globally renumber particles. |
| EOS | `modules/eos/idealGas.py` is the active Torch ideal-gas conversion. `modules/eos/gas.py` and `props.py` contain an unused list/mask implementation and a second `fluidProperties` type. | Build one supported material API. Do not wire the Python masking loop into the timestep or introduce a third ambiguous `fluidProperties`. |
| Configuration | `configurations/compressibleConfig.py` contains scalar `gamma`; CRKSPH and CompSPH inherit it. Dict import/export explicitly enumerate fields. | Add material configuration and backward-compatible serialization in the common base. Keep device objects out of serialized config. |
| Scheme EOS | `schemes/compressible/{crkSPH,compSPH,monaghan}.py` call `idealGasEOS` with `schemeConfig.gamma`. CRK pressure/viscosity and work kernels already consume particle pressure and sound speed. | Replace the closure centrally; the CRK force law does not need a new EOS-specific derivation just to use different ideal gases. |
| Compatible energy | `modules/compSPH/operators.py` forwards one gamma to `balance.py`; `EnergyScheme.CRK` compares `P_i/rho_i**gamma` and `P_j/rho_j**gamma`. `systems/compSPH.py` applies the accumulated pair work. | Specify an energy-partition policy for different EOS parameters. Substituting two gamma values in this comparison is not sufficient: the two entropic functions have different units/normalizations. |
| CFL | `modules/timestep/compressible.py` independently recomputes the ideal-gas sound speed with global gamma. | Route through the material closure using the current internal energy. Stored sound speeds can be from the last RK stage and must not simply be assumed current. |
| Initialization | `sample/compressible.py`, `sample/regions2D.py` and individual `caseUtils/compressible/*` builders compute energies with global gamma. Triple-point helpers also restore prescribed densities after calculating energies. | Add a common material-aware initialization routine and use the density actually used by the selected scheme when constructing pressure-balanced initial energies. Audit any builder that accepts a material table. |
| Density at contacts | `modules/crk/terms.py` uses the core CRK density operator. In `../warpSPHCore/src/warpSPHCore/crk/crk_density.py`, neighbors contribute without a material filter. | A material boundary is not automatically a sharp density boundary. Contact equilibrium and sampling are a separate numerical gate. |
| Diagnostics and reload | `cases/compressible/common.py` reconstructs entropy with global gamma; `io/importIO.py::loadTrajectoryFrame` reconstructs pressure/sound speed/entropy the same way. `io/export.py` stores material labels, currently as floats. | Update diagnostics and reconstruction, validate/cast loaded IDs, and include the material list in metadata. Report entropy changes by material rather than interpreting a sum of unlike entropic functions as thermodynamic entropy. |
| Other physics options | Wadsley2017 and ReadHayfield2012 use global-gamma formulas; `compressibleWall/wall.py` has a single-gas wall closure. GSPH/Inutsuka and Riemann artificial viscosity use single-gamma Riemann solves; MFM/MFV have their own global-gamma primitive/flux path. | These options must either be adapted and validated or rejected for heterogeneous EOS. Changing their pressure call does not establish support. |

The runner/configuration interface was committed in `0f1b2d3` during this
scoping session. Implement against that versioned contract. The plan index also
records a related triple-point requirement: three thermodynamic regions
can represent only two materials. RM has the same distinction: shocked and
unshocked regions of the incident gas share one material.

## Material configuration and Warp representation

Use a serializable `MaterialConfig` with a descriptive name and an EOS
description. The initial supported description is `IdealGas(gamma)` with
finite `gamma > 1`. List position is the dense, zero-based material ID.
Explicit tables require fluid IDs in `[0, len(materials))`; unused entries
are allowed. Two disjoint regions can select the same entry.

Proposed run-document shape, to add to the existing solver descriptors:

```yaml
schemaVersion: 1
case: richtmyerMeshkov
solver:
  scheme: CRKSPH
  materials:
    - name: incidentGas
      eos: {type: idealGas, gamma: 1.4}
    - name: transmittedGas
      eos: {type: idealGas, gamma: 1.6666666666666667}
```

This is the proposed interface, not an executable case configuration yet.
Material 0 selects the first entry in both shocked and unshocked regions.
Register `solver.materials` as an array of validated objects, using the
existing `Parameter.items` and `Parameter.properties` mechanism. Convert
these plain objects to host config types before building the runtime table.

Recommended device layout, with names provisional:

```python
@wp.struct
class EOSParameters:
    eosType: wp.int32
    gamma: scalar_t

@wp.struct
class MaterialTable:
    entries: wp.array(dtype=EOSParameters)
    count: wp.int32
    uniform: wp.int32
```

Build the list of parameter structs and upload it once for the configured
device and precision. Each EOS thread reads
`table.entries[materials[i]]`, or entry zero in compatibility mode. Warp
supports structs and arrays initialized from lists of structs; see the
[official runtime documentation](https://nvidia.github.io/warp/latest/user_guide/runtime.html#structs).
Confirm the actual installed/minimum supported Warp version in phase 1.

Keep the table on a runtime owner reachable by the RHS and timestep path,
outside integrated particle fields. The owner must preserve its lifetime
through every stage, rerun, and captured graph. Build it before graph capture;
invalidate the runtime cache and recapture if the material configuration,
device or precision changes. Never rebuild/upload it on each EOS call.

Compatibility rules:

- `materials=None` on the scheme config retains scalar `gamma` as a uniform
  EOS for every fluid particle, regardless of its existing material label.
  Existing Sod region labels must not cause out-of-range table access.
- An explicit material list is authoritative; scalar `gamma` remains a
  legacy fallback and is marked inactive in this mode. Reject ambiguous
  conflicting user overrides rather than silently selecting one.
- Preserve the public `idealGasEOS` function for existing callers. Add a
  material-aware API with an explicit source quantity and stable result
  ordering. Initialization uses pressure as source, the RHS uses internal
  energy. Support entropic-function and sound-speed conversion explicitly
  if retaining the old helper's full capability.
- For multi-material cases, validate material IDs once after construction
  and on load/insertion, before capture. Wrong shape, nonintegral values,
  negative IDs and missing entries produce a clear error. Add a device-side
  guard/error flag so invalid runtime IDs cannot cause an unchecked lookup.
- `kinds != 0` rows need an explicit thermodynamic/boundary policy. Initially
  reject heterogeneous EOS with solid-wall rows; do not index the fluid table
  with rigid-body IDs. The first RM case uses fluid buffer bands, not solids.

For each ideal-gas material, use the same closure in all paths:

```text
p = (gamma_m - 1) rho u
c^2 = gamma_m p / rho
A = p / rho^gamma_m
u = p / ((gamma_m - 1) rho)
```

`A` is the entropic function, not physical specific entropy. Require valid
positive initial thermodynamic states for RM and monitor positivity during
the run. The current `abs(u)`/`abs(p)` sound speed must not hide an invalid
state in acceptance diagnostics. Any floors or legacy behavior must be
explicit and counted rather than introduced silently during this refactor.

Use one fused particle kernel with reusable outputs in the forward path.
Maintain differentiation through `rho` and the source quantity via the
repository's autograd/JVP conventions or a material-gathering Torch reference
fallback. A raw `wp.from_torch` launch alone does not connect Torch autograd;
see [Warp interoperability guidance](https://github.com/NVIDIA/warp/blob/main/docs/user_guide/faq.rst).
Optimizing material coefficients themselves is outside the first scope.

## Compatible energy and material contacts

Preserve the original CRK energy partition for particles with identical EOS
parameters. For a pair with different EOS parameters, start with an explicit
equal-work partition, using the existing `EnergyScheme.equalWork` algebra
for that pair only. This is a proposed conservative baseline, not a claim
that it preserves entropy monotonicity or positive internal energy.

The implementation must show that the two pair orientations partition the
same total work, including unequal particle masses; test pair reversal,
`0 <= f <= 1`, conservation, and internal-energy positivity. Compare this
policy with the existing monotonic/hybrid energy schemes on the contact and
planar interaction. If the equal-work baseline produces unacceptable contact
heating or negative energy, resolving that is required work before 2D RM.
Do not resolve it by averaging gamma or comparing unlike raw `A` values.

Keep the default CRK viscosity/velocity limiter as the first baseline, with
the conservative `KernelMeanSymmetric` support mode. Initially allow the
no-switch/fixed-viscosity path; qualify Cullen–Dehnen separately. Rosswog's
per-particle relative entropy history can be qualified with local `A`, but
Read–Hayfield entropy diffusion, Wadsley’s pressure detector, and Riemann
viscosity require separate material-aware work. Reject unqualified options
at setup. Do not add thermal smoothing across unlike materials as an
unexamined cure for a pressure blip.

Construct both a uniform-spacing/variable-mass contact and an equal-mass/
variable-spacing contact. Measure the density reconstruction and pressure
error before stepping and after several sound crossing times. If neither
holds an acceptable contact, inspect the CRK density/support formulation
before continuing. A correction must preserve reproducing conditions,
pair symmetry and conservation; simply removing cross-material neighbors
from existing sums is not a justified fix. This branch may require changes
in warpSPHCore as well as this frontend.

## Richtmyer Meshkov initial conditions

Use a planar shock moving in positive x through gas 0 toward gas 1. The
unshocked gases start at equal pressure and zero velocity, separated by

```text
x_interface(y) = x0 + a0 cos(2 pi y / wavelength)
```

There are three initial states and two materials:

| Region | Material | State |
|---|---|---|
| Behind the incident shock | 0 | Compressed incident gas with positive streamwise velocity |
| Between shock and interface | 0 | Incident gas at rest |
| Beyond the perturbed interface | 1 | Second gas at rest at the same initial pressure |

Generate the post-shock density, pressure and velocity from the incident
material's Rankine–Hugoniot relations; never choose an arbitrary high-pressure
patch and call its emerging waves an incident shock. With incident Mach
number `M`, `c0 = sqrt(gamma0 p0/rho0)` and shock speed `D = M c0`:

```text
r = rho_post/rho0 = (gamma0 + 1) M^2 / (2 + (gamma0 - 1) M^2)
p_post/p0 = 1 + 2 gamma0 (M^2 - 1) / (gamma0 + 1)
vx_post = D (1 - 1/r)
```

These are the initialization relations used by
[Tritschler et al.](https://arxiv.org/html/1402.1433#S3.SS2).
That paper's multimode, diffusive, reshocked problem is a later benchmark,
not the first single-mode inviscid acceptance case.

Proposed engineering parameters, not a published benchmark: nondimensional
`wavelength=1`, `p0=1`, `rho0=1`, `rho1=3`, `gamma0=1.4`,
`gamma1=5/3`, and `M=1.2`. Use `a0/wavelength=0.01` for linear growth
and `0.05` for a demonstration. Also retain equal-gamma and `a0=0` controls.
Place the shock at least several kernel radii upstream of the nearest
interface point. Expose shock position, interface position, domain lengths,
material list, densities, pressure, Mach number and amplitude in the case
parameters; derive post-shock quantities rather than duplicating them.

Use periodic y and fluid Dirichlet reservoir bands at distant x ends,
following the existing compressible buffer mechanism. Size the domain and
bands for the maximum support and observation interval; keep the interface
and all measured waves outside their causal influence. Check with a longer
domain and wider bands. These bands are not established nonreflecting
boundaries, and prescribed reservoirs exchange momentum/energy. Record that
exchange or restrict conservation checks to a closed periodic control and
an unaffected observation region. For later nonlinear growth, extend the
domain/run together and repeat the boundary-independence check.

Sample physical mass/volume consistently with each region's density. Solve
supports and CRK volume/density before calculating `u` from prescribed
pressure with local gamma; quantify how the discrete density differs from
the intended plateau/contact. Recompute total particle energy as
`m (u + |v|^2/2)`. A setup that resets density after this calculation can
inject a pressure transient at the first RHS evaluation.

For the small-amplitude case, prefer a smooth displacement of a planar
particle arrangement that conforms to the interface, fades away from it,
and leaves the upstream shock planar. Check the mapping's cell volumes and
resulting masses/densities. A binary material mask on an undeformed lattice
can turn a small sinusoid into a staircase or remove its seed entirely.
Measure the actual initial Fourier amplitude and unwanted harmonics at
each resolution; increase resolution if geometry or reconstruction cannot
resolve the prescribed perturbation. A coarse smoke run does not establish
linear-growth accuracy.

## Implementation sequence and gates

Estimates are focused engineering days, including local validation, and are
conditional on the contact gate. They are not run completion forecasts.

| Phase | Work and likely files | Exit gate | Estimate |
|---|---|---|---|
| 1 Material contract and closure | New material config/device types and EOS kernel under `configurations/moduleConfigurations/` and `modules/eos/`; exports; common compressible config dict converters. Retain the old scalar helper. | Mixed/permuted labels match analytic conversions and the scalar reference; invalid IDs/states fail clearly; f32/f64 and gradients work; runtime table survives repeated calls without uploads. | 1–2 days |
| 2 Complete the thermodynamic path | Three SPH RHS functions, timestep, common initialization helper, compatible-energy operator/kernel, diagnostics, setup guards. Recompute derived outputs from finalized `rho,u` when required, including loaded/output states. | Single-gas regression, explicit identical materials vs uniform EOS, unequal-gamma pair-work conservation/positivity, and eager/graph parity. Table buffers remain valid across RK stages and step reruns. | 2–3 days |
| 3 Configuration and saved runs | `runner/configuration.py` and parameter descriptors; config validation/overrides; `io/{export,importIO}.py`; runtime owner rebuild on restore. Python and YAML use the same list schema. | Config and trajectory round trips reconstruct material-specific pressure/sound speed; old files still load; unsupported solver/options are rejected. Integrate with the versioned runner contract committed in `0f1b2d3`. | 0.5–1 day |
| 4 Contact and planar shock gate | New focused EOS/material physics tests and a probe; both particle samplings; stationary/advected contact; single-material incident shock; zero-amplitude two-gas interaction with a two-gamma reference. | Pressure equilibrium, unchanged labels/material mass, shock jump/speed and transmitted/reflected waves agree within fixed tolerances and improve with resolution. No unexplained positivity failures. | 1–2 days if the existing contact treatment passes |
| 5 Registered RM case | New `caseUtils/compressible/richtmyerMeshkov/` sampler, `cases/compressible/richtmyerMeshkov.py`, case registration, plotting, parameter metadata, example wrapper/config. | Smoke run finishes; flat-interface control stays flat; perturbed interface shows shock-induced transverse motion and growth; no boundary contamination during the measured interval. | 1–2 days |
| 6 Qualification and example | Sequential resolution runs, independent reference, linear growth fit, material/density/pressure video and saved diagnostics, performance and documentation. | Quantitative criteria below pass; a complete reproducible demonstration runs to its chosen time with resolved configuration and evidence. | 1–2 days plus GPU runtime |

Expected uncomplicated scope is about **7–12 focused days**, across roughly
15–25 production files plus tests/examples. A failure of the contact gate
opens a numerical-method task whose cost cannot be inferred from EOS
plumbing; allow a separate investigation before promising the RM date.
Riemann-based schemes and general multi-material walls are not included in
that estimate.

Phase 5 is the runnable RM milestone after the contact and planar checks;
phase 6 supplies quantitative qualification. Select and obtain the independent
2D reference during phase 4, before spending time on long refinement runs.
It must match the two gamma values, interface geometry, shock strength and
observation window, and demonstrate its own convergence. If producing that
reference requires a new solver or a separate installation project, estimate
that work separately and label the phase 5 result as a validated shock/contact
implementation with qualitative RM growth until phase 6 is complete.

## Validation and acceptance evidence

Set numerical tolerances before accepting results. The following are
proposed starting gates, to calibrate against dtype and resolution during
phase 4 without relaxing them to conceal a failure:

1. **EOS and identity:** `u -> p -> u`, `A -> u -> A`, and sound speed
   conversions match an independent analytic/Torch reference. Test arbitrary
   particle ordering, duplicate EOS entries, empty input, invalid IDs, dtype
   and device. Material labels and material masses survive RK stages,
   sorting/rebuilds and reload. Add a float64 gradient check for the new
   kernel or the documented differentiation fallback.
2. **Compatibility and energy:** repeat representative Sod, linear-wave,
   Sedov and KH checks. A uniform table and the legacy global configuration
   agree within roundoff, including when existing labels exceed zero.
   In closed controls, retain the existing scheme conservation tolerance
   (`tests/test_physics.py`, CRKSPH currently allows relative drift `1e-4`
   in its short scheme battery) and seek roundoff-level compatible energy
   in f64. Equal-material behavior must not change just because labels differ.
3. **Contact:** after one wavelength sound crossing, target pressure error
   below 1% of the imposed pressure and spurious velocity below 1% of the
   smaller sound speed in the measured contact region, with improvement on
   refinement. Test both a stationary and uniformly advected contact and
   more than one density/gamma ratio. Failure is a blocker for RM qualification.
4. **Planar shocks:** compare the incident shock with its analytical jump
   and speed, then compare the two-gas interaction to an independent exact
   two-gamma Riemann solution. Target 1–2% plateau pressure/velocity and wave
   position errors away from smeared discontinuities at the qualification
   resolution. The existing Sod solver has a single gamma and cannot serve
   unchanged as this oracle. A small validation-only two-gamma exact solver
   does not require extending the production Riemann schemes.
5. **Linear RM:** start at 64, 128 and 256 particles per wavelength in y,
   refining further if the small initial amplitude is underresolved. Report
   actual discrete amplitude, unwanted harmonics, local support and interface
   resolution. Track the signed
   fundamental Fourier amplitude of a reconstructed material interface,
   its mean translation, shock arrival and the post-shock amplitude. Fit
   early growth only after shock passage and before nonlinear deformation
   or boundary influence. Compare to the impulsive estimate
   `da/dt ≈ k A_post delta_v_interface a_post`, with post-shock densities
   and interface velocity from the planar reference. This estimate has
   compressibility/transient limitations; use an independently converged
   solution with identical ICs for a quantitative acceptance target.
   Target less than 10% change in fitted growth between the two finest
   resolutions and agreement within 10% of that reference. The
   [impulsive-model study by Vandenboomgaerde et al.](https://www.lsce.ipsl.fr/wp-content/uploads/2013/01/Vandenboomgaerde_et_al_PhysRevE_1998.pdf)
   explains why shock and amplitude conventions matter.
6. **Controls and diagnostics:** the zero-amplitude case must not develop
   a transverse signal comparable to the seeded case. Include an equal-gamma
   control and a uniform-density/identical-EOS control. Plot material identity,
   density, pressure and vorticity; save bubble/spike positions after roll-up,
   momentum/energy changes, per-material mass, minima of `rho,u,p,c²`, and
   timestep progress. Reconstruct volume fractions with particle volumes,
   not particle counts when masses or spacing differ. Apparent interleaving
   is geometric entrainment, not a measurement of molecular mixing.
7. **Reproducibility and cost:** compare eager and captured paths, including
   at least one adjacency rebuild. Profile the EOS and complete step against
   the uniform baseline; confirm no per-stage host sync, upload or Python
   material loop. Run GPU cases sequentially with progress, stall detection,
   video and an export root. Record wall time, ms/step, particle count and
   device/precision. Run the repository's relevant tests, import checker,
   affected-case sweep and gradient checks before calling the feature done.

## Follow on work

After the first RM acceptance, use the same material API to qualify the
two-gamma triple-point case, fixing region-to-material mapping and the
equal-mass sampler's hardcoded split positions as needed. Extend shock
switches individually with a demonstrated cross-material policy.

Stiffened gas is a reasonable next compressible EOS, but needs pressure,
energy, sound speed and entropy definitions plus new reference tests.
Barotropic WCSPH material lists can reuse the device-table mechanism while
retaining their own pressure-only contract. GSPH/Inutsuka, MFM/MFV and solid
walls each require separate multi-material thermodynamics and flux/boundary
work; reject unsupported configurations until those paths are qualified.

## Implementation status on 2026 10 19

The user authorized implementation after scoping. The material dataclasses,
persistent Warp table, fused EOS, source conversions, Torch differentiation
fallback, configuration schema and dict round trips are implemented.
CRKSPH, CompSPH and Monaghan now use material-dependent thermodynamics in
the RHS, CFL calculation, finalized outputs and trajectory reconstruction.
Legacy configurations retain their scalar EOS and existing region labels.
The new material option is keyword-only to preserve positional constructors.

CRK compatible energy uses equal-work partition across unlike gamma values
and retains the previous entropy weighting for equal gamma. Material-aware
grid runs grow the support grid as soon as its radius binds, matching graph
replay instead of the legacy eager delay. Captured material arrays have an
explicit lifetime owner; changing the table invalidates the step capture.

The registered RM case uses the proposed pressure-balanced displaced lattice
and an analytic incident shock. The implementation uses a padded periodic
domain in both directions instead of Dirichlet reservoirs. This keeps the
validation closed and permits graph replay; setup checks a conservative
signal-return time, including a kernel-width margin. Only the central
interaction before that return time is the intended RM problem. The live
video follows the interface and leaves the distant seam out of view.

Completed numerical checks include the stationary and advected contacts,
the planar two-gamma interaction, and RM growth at 32, 64 and 128 particles
per wavelength. At t = 2 the interface amplitudes are 0.01400, 0.01492 and
0.01544 from an initial amplitude near 0.01. Growth fitted over t = 1–1.8
changes by 6.7% between the two finest runs. The planar plateau pressure
errors are below 0.06%; CRK total-energy drift remains below 4e-7 in the
small-amplitude runs. Artifacts and final verification are recorded under
`reports/materialEOS/` and `results/materialEOS/`.

Remaining qualification includes an independently converged 2D reference,
resolution 256 or higher, an equal-mass sampling comparison, broader
density/gamma ratios, and general multi-material boundary/switch support.
These remain separate from the implemented and running ideal-gas RM case.

## Remote head integration on 2026 10 19

The user requested launching RM through the warpSPH runner and Remote head.
The registered `richtmyerMeshkov` case uses the Remote companion's dynamic
installed-case discovery, including all five native variants, the three
supported SPH solvers and the nested `solver.materials` schema. No separate
Remote preset or browser build is needed. A companion launch request is now
under `examples/remote/`; the README documents the case-list refresh and
explicit variant end time. Discovery, schema description and recipe validation
pass for all three solvers, and every variant validates. The active local
companion's case cache was refreshed and advertises the case as available.

The sibling warpSPHRemote publisher now includes the optional integer
`materials` field, enabling live material coloring and profile selection.
Its focused configuration/transfer/buffering/video regression bundle passes
26 tests. An isolated managed CRKSPH worker completed five steps on the GPU
with 8,192 particles, served an initial material snapshot while paused, resumed
and recorded a trajectory with `combinedMaterials`. This tiny integration
check used live snapshots rather than video; it is not another physics or
performance qualification. See the material-EOS report for the measurements.

## Nonlinear benchmark comparison on 2026 10 19

The user reports that the t=6 demonstration remains much less dynamic than
the nonlinear RM example in Dzanic and Martinelli, *High-order limiting
methods using maximum principle bounds derived from the Boltzmann equation
I: Euler equations*, section 6.8 / figure 10
([author-hosted paper](https://tarikdzanic.org/docs/jcp_boltzmann_euler_limiter.pdf)).
This is a valid limitation of the demonstration's scope: completion, energy
conservation and early growth do not establish nonlinear roll-up fidelity.

The supplied reference prescribes three initially stationary primitive states
`(rho,u,v,p)`: bottom `(0.25,0,0,1)`, middle `(1,0,0,1)`, and top
`(8/3,0,0,4.5)`. The lower discontinuity is flat at y=0.5; the upper
pressure/density discontinuity is y=2-0.5*cos(2*pi*x). It uses a half-wavelength
domain with symmetry walls transversely and Dirichlet conditions streamwise.
The figure reflects the half-domain and shows a cropped subregion at t=2.
These are not the current planar Rankine–Hugoniot shock / pressure-balanced
sinusoidal two-material contact initial conditions. The reference's zero
initial velocities also mean its pressure ratio must not simply be relabeled
as the Mach number of our incident shock.

Our demonstration has amplitude/wavelength 0.1 (default 0.01), Mach 1.2,
incident pressure ratio 1.5133 and density ratio 3 across the unshocked
material contact. Its two gases have gamma 1.4 and 5/3. Our nx=64 particle
resolution is also not comparable to the reference's P4 element counts.
The difference in forcing and geometry explains why directly comparing t=6
with that figure's t=2 is not a useful accuracy test. It does not establish
how much additional roll-up our artificial viscosity or kernel resolution
may suppress; that needs controlled measurements.

The user subsequently clarified that this image was illustrative, not a request
to adopt this particular setup. The reference choice and next qualification
sequence below supersede making that three-state case the immediate target.

Candidate qualification work for that particular setup:

1. Preserve the weak-shock case for contact, planar and early-growth tests.
2. Add a separately named runner/Remote benchmark matching the reference's
   three-state ICs, orientation, EOS and boundaries. Establish the EOS value
   from the reference's reproducible input before calling the setup exact.
   The current sampler's compact deformation, Mach-only pressure assignment
   and periodic-boundary assumptions cannot reproduce it through overrides.
3. Compare density, material interface, vorticity/circulation, layer width,
   energy and minima at the same physical time and crop. Fourier marker
   amplitude ceases to be a sufficient diagnostic after overturning.
4. Refine particle resolution and compare the supported viscosity policies
   with identical ICs, retaining shock/contact controls. Establish large-scale
   roll-up before using the finest secondary curls as an acceptance target.

## Reference selection and next qualification sequence

Publication correction: the author-hosted PDF above is a preprint copy of a
peer-reviewed article published in *Journal of Computational Physics* 529,
113895 (15 May 2025), DOI
[10.1016/j.jcp.2025.113895](https://doi.org/10.1016/j.jcp.2025.113895).
[The authors' institutional record](https://collaborate.princeton.edu/en/publications/high-order-limiting-methods-using-maximum-principle-bounds-derive/)
confirms that status. Its publication status is not a reason to discard it;
its different ICs and lack of a verified reference-data package remain
practical considerations for this project.

Recommended references, rather than a single image as ground truth:

- **Linear verification:** Wouchuk, *Physical Review E* 63, 056303 (2001),
  [growth rate when a shock is reflected](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.63.056303).
  This compressible linear theory accommodates arbitrary shock strength,
  density ratio and fluid compressibilities within its reflected-shock
  regime. Use it beyond the existing approximate impulse-model comparison;
  do not extrapolate it to nonlinear roll-up or reflected-rarefaction cases.
- **Primary single-mode validation family:** Collins and Jacobs, *JFM* 464,
  113–136 (2002), [air/SF6 experiments](https://doi.org/10.1017/S0022112002008844),
  together with Latini, Schilling and Don, *Physics of Fluids* 19, 024104
  (2007), [experimental/model comparison](https://doi.org/10.1063/1.2472508),
  and their *JCP* 221, 805–836 (2007),
  [WENO order/resolution study](https://www.sciencedirect.com/science/article/abs/pii/S0021999106003196).
  The latter examines orders 3/5/9 and 128/256/512 grid points per wavelength.
  Start with the pre-reshock interval. Physical interface thickness, EOS,
  experimental conditions and numerical regularization must be specified;
  a pure-parcel sharp interface cannot be declared an exact reproduction of
  a diffuse miscible experiment.
- **Later 3D multimode target:** Thornber et al., *Physics of Fluids* 29,
  105107 (2017), [theta-group collaboration](https://doi.org/10.1063/1.4993464),
  with an [open author manuscript](https://arxiv.org/abs/1706.09991).
  Eight independent codes provide a more useful uncertainty context for
  large-scale turbulent statistics than a single solver image. This is a
  later target, not a replacement for current 2D single-mode verification.

The Mach-number explanation needs qualification: the Collins/Jacobs
experiments use Mach 1.11 and 1.21 and do observe instability development.
Mach 1.2 alone cannot diagnose our weak late-time dynamics. Initial/post-shock
amplitude, wavelength, time since interaction, circulation deposition,
resolution and dissipation must be separated.

Recommended execution order (proposals, not newly completed validation):

1. Extend the existing case's diagnostics with separate bubble/spike
   positions, geometric layer width, signed circulation in each half-wave,
   transverse kinetic energy and maps of artificial-viscosity activation.
   Compare dimensionless amplitudes/time using one independent reference
   growth rate, not a separately fitted normalization for each solver.
2. Verify the deposited circulation and early growth against compressible
   linear theory for the actual two-gamma ICs. Include equal-gamma,
   equal-state and planar controls; retain existing EOS/contact tests.
3. Run a controlled 64/128/256 particles-per-wavelength study and a timestep
   refinement. Compare supported viscosity policies while holding ICs fixed,
   then investigate particle mass/spacing treatment as a separate change.
   Test streamwise padding sensitivity before investing in new boundaries.
4. Freeze a published single-mode benchmark specification and independent
   numerical reference. Archive source/version, ICs/EOS, boundary and
   transport choices, reference resolution, arrays, normalization and
   uncertainty. No downloadable raw field archive for the single-mode
   reference has yet been verified; obtaining data or generating a converged
   independent solution is real work, not something supplied by a citation.
5. Qualify nonlinear bubble/spike growth, circulation and layer width before
   secondary curls. For a specified finite interface thickness/physical
   transport model, refine those physical scales consistently. Separate
   geometric entrainment from molecular mixing, which the current pure
   material IDs and ideal-gas table do not model.
6. Add reshock/wall handling and eventually 3D multimode studies once the
   single-shock baseline is understood. Compare experimental data within
   its applicable time window and uncertainty; do not require late 2D
   patterns to match turbulent 3D laboratory structure point by point.

## Qualification execution and boundary question

The user authorized this sequence and requested that longer runs register with
the resident Remote companion. `probe_materialEOS.py --live` now wraps the
case with `SimulationPublisher`, while preserving local video, trajectory and
progress. `study_materialEOS.py` runs children sequentially, records a campaign
manifest/logs and stops the queue on a failed or remotely stopped run. The
initial pilot compares NoneSwitch and Cullen–Dehnen at nx=64 for both small
amplitude to t=2 and amplitude 0.1 to t=6. Refinement cases are explicitly
listed separately, including resolution, timestep and padding controls.

New observations are corrected-SPH curl integrated with particle volume m/rho,
signed half-wavelength circulations, enstrophy, transverse kinetic energy,
interaction-weighted viscosity alpha and marker-band envelopes. Expensive
measurements have an explicit `circulationTime` and physical-time cadence.
The marker band has finite O(dx) width; it is not an exact material surface.
Pilot raw rows initially used the opposite directional bubble/spike labels;
the analysis derives both from interfaceMin/Max/Mean, and the helper is now
corrected (light material 0 bubbles toward +x). Fourier amplitudes and curl
measurements are unaffected by that label correction.

The user asked why the t=6 domain is so large and noted the RT Dirichlet BC.
The 32-wavelength domain is a conservative periodic-seam isolation choice,
not a physical RM requirement. RT already uses fluid Dirichlet bands,
prescribing rho/u/velocity and zero dvdt/dxdt in the hydrostatic reservoirs.
The framework is reusable, but RM requires moving/flowing reservoir targets,
material-consistent EOS and adequate particle coverage; fixed zero-velocity
RT bands would change this problem. Add a compact reservoir case and compare
its central growth/circulation to the padded control before reducing the
default domain. Dirichlet values alone do not establish non-reflecting
boundary behavior. The padded runs continue as boundary-independent controls.

### First study results (2026-10-19)

The four nx=64 pilot runs and four selected refinement controls completed.
[The report](reports/materialEOS/README.md#controlled-diagnostics-study)
contains source paths, timings and observations. At small amplitude, extra
padding changes the final Fourier amplitude by less than 0.0001%, half CFL
by 0.011%, and doubling nx by 3.4% (growth fit changes 7.2% relative to nx=64).
Cullen–Dehnen changes small-mode growth by less than 1% at nx=64 and
negligibly at nx=128; changing switches alone is not the roll-up remedy.
Final t=6 maps show the expected counter-rotating shear but no pronounced
roll-up. Material sampling/regularization and nonlinear convergence remain
open. The sequential `linear256`, `nonlinear128`, `nonlinear128-cullen` queue
has started and is visible in Remote; do not count its results as complete.

The weak-shock estimate now implemented in `reference.py` is only the
irrotational interface contribution (Sano et al., ApJ 758, 126 (2012),
[equations 6–9](https://doi.org/10.1088/0004-637X/758/2/126)). It omits the
full acoustic/bulk-vorticity history. Its normalized curves are context,
not exact relative errors or a direct measure of numerical damping.

A stronger, reproducible linear-reference lead is Napieralski, Cobos,
Velikovich and Huete, *Richtmyer–Meshkov instability when a shock is reflected
for fluids with arbitrary equation of state*, JFM (2024),
[DOI 10.1017/jfm.2024.1010](https://doi.org/10.1017/jfm.2024.1010).
The primary article supplies a fully compressible linear theory and links
author Mathematica notebooks for base flow and perturbations. That is a
better route to transient and asymptotic verification than treating the
impulse formula as exact. The article was inspected, but its supplementary
ZIP returned HTTP 403 to the download client, and no `wolframscript`/`math`
runtime was found here. No author-code solution has been executed or ported;
this is a documented next step, not completed reference validation.

For an independent nonlinear solver, MFC ships an RM example at
[pinned source commit 42c44fdc2b1077ec5cb0a67bcf6fd1b869d1328c](https://github.com/MFlowCode/MFC/tree/42c44fdc2b1077ec5cb0a67bcf6fd1b869d1328c/examples/2D_richtmyer_meshkov).
Its case and hardcoded IC 208 were inspected. It uses a five-equation
two-fluid model, physical gammas 1.093 and 1.4, an error-function mixture
profile whose thickness depends on grid spacing, and viscosity. This is a
candidate reproducible solver, not a matched reference for our sharp pure
parcels. Freeze consistent ICs/EOS/transport and establish its convergence
before adopting any arrays as a nonlinear acceptance target. MFC has not
been built or run in this study.

### Compact-domain comparison before changing defaults

Reuse the fluid Dirichlet-band dispatch rather than adding solid walls.
Keep the present padded periodic run as a control. Introduce a separately
named compact variant with transverse periodicity, streamwise reservoirs,
material-specific prescribed thermodynamics and flow-compatible particle
motion/coverage. Decide how waves leave the domain before choosing reservoir
targets: a constant primitive-state reset can reflect an outgoing shock.
Validate stationary and advected contacts, then the planar interaction,
then the seeded case at the same nx and physical observation times. Compare
central pressure/velocity plateaux, Fourier growth, circulation and envelope
widths while moving each boundary independently. Include a run long enough
for an outgoing wave to reach the treatment, since an early-time match alone
does not validate it. Track mass/energy flux for open boundaries rather than
requiring the padded closed-domain conservation diagnostic to stay zero.
Only then reduce the default domain. The user's priority is sound physics;
the current runtime does not justify rushing an unqualified boundary change.

### Remote interruption investigation (2026-10-19)

The user reports repeated daemon disconnects and frozen live updates during
the study. Initial observation: the active `linear256` solver continues to
advance, localhost run-list latency is about 6 ms, and a 23.6 MB dynamic
snapshot remains pending viewer acknowledgement for over two minutes.
Diagnose the transport/viewer path without stopping or taking over the run;
do not equate a stalled live frame with a stalled integrator.
The user confirms the published viewer at `https://fluids.dev/warpSPHRemote/`
is talking to the local daemon. Its deployed bundle already disables the
total-duration deadline for snapshot/static downloads. A brief explicit
viewer takeover retrieved and validated the 23,593,592-byte pending frame
in 17.6 ms and its 10,486,048-byte static packet in 9.9 ms, acknowledged it,
then retrieved/validated/acknowledged fresh sequence 489 in 13.8 ms. The
diagnostic viewer detached immediately. Solver execution was unchanged.
The user saw the takeover notification and is refreshing the browser;
recovery in their viewer is not yet confirmed. No daemon restart or network
security change was performed, and the original cause remains unproven.
The user subsequently confirmed that refreshing restored normal operation.
Treat this as a recovered viewer session, not a proven daemon fault or a
permanent fix. If it recurs, collect browser Console/Network failures before
refreshing to distinguish a failed request from a rendering/ACK deadlock.
The user requested connection/timeout console diagnostics be tracked in the
Remote repository rather than implemented in the fluid solver. Added that
follow-up, evidence and verification requirements to
`warpSPHRemote/reports/ROADMAP.md` (practical gaps, item 9); resume observation
of this campaign without changing physics or restarting its processes.

### Linear nx=256 completion (2026-10-19)

The run completed at t=2.00007 after 5,767 steps: fitted growth 0.00649184,
final Fourier amplitude 0.01568107, energy drift -5.96e-7, wall 903.34 s and
mean step 150.19 ms. Relative to nx=128, the growth change is 3.0% and the
amplitude change 1.6%; the preceding changes were 7.2% and 3.4%. Archive the
updated comparison in `reports/materialEOS/studyComparison.json` and the
figure under `results/materialEOS/study_2026-10-19_highres/analysis/`.
This establishes decreasing resolution sensitivity of these observations,
not full compressible-theory validation. The next `nonlinear128` run has
registered and is delivering fresh acknowledged frames in Remote; the
Cullen–Dehnen counterpart remains queued. No concurrent GPU postprocessing
is run while these simulations occupy the GPU.

### Viewer memory regression isolated (2026-10-19)

The user subsequently reported Chrome increasing roughly 50 MB per frame,
reaching 15.4 GB and reconnecting again. A full-viewer synthetic reproduction
isolated a historical callback/parent-state retention chain in Remote. The
viewer-only fix uses stable callback slots: controlled baseline 942 MiB JS
heap at 50 frames versus fixed 87 MiB at 200 frames. Tests, explicit GC check
and production build pass. Findings and reproducible fixture are recorded in
`warpSPHRemote/reports/LIVE_VIEW_MEMORY.md`; hosted deployment remains pending.
A tested static build is served at `http://127.0.0.1:5174/` for pairing with
the existing daemon. No simulation or daemon restart was performed. The run
continues normally at t=4.41/6. The user's Chrome Local Network Access Issues
entries are tracked separately; no browser security setting was weakened.
The user requested deployment of the tested fix. Remote commit `90521fe`
was pushed to `main`, triggering its normal Pages workflow. The user also
reaffirmed localhost access, including SSH port forwarding, as the intended
connection path; direct-network TLS/certificate work remains parked.
Pages build and deployment succeeded for `90521fe`; the public page returned
HTTP 200 and served `index-C_xxvnyT.js`, verified to include the stable
callback-slot fix and explicit target address-space annotation. Existing
hosted browser tabs need a refresh to acquire it; the daemon and solver do
not need restarting. The separate broader CI job also completed successfully.

### Nonlinear nx=128 completion and viewer feedback (2026-10-19)

The constant-alpha nonlinear run completed at t=6.0005 in 10,287 steps,
1454.87 s wall, 132.66 ms/step, energy drift -1.79e-7. Final Fourier amplitude
is 0.219299 (7.5% below nx=64), marker width 0.568675 (2.2% above nx=64),
and upper-half circulation 0.0458697 (-0.6%). The final material image still
shows a broad lobe without pronounced roll-up; the differing Fourier and
envelope trends require nonlinear shape/reference qualification. The
`nonlinear128-cullen` run is now active (t=1.12 at the status check), with
the same sequential execution and Remote registration.

The user reports that the deployed memory fix appears effective. Their new
Chrome Issues screenshot explicitly marks 1,793 affected requests Blocked;
this stronger evidence is recorded in Remote's roadmap/findings as a separate
connection follow-up. No further networking or security changes were made.

### Spheral multi-material verification candidates (2026-10-19)

The user asked for useful Spheral cases beyond TriplePoint and questioned the
need for an added viscosity switch. Inspected local Spheral commit
`116c71f2ea559d569456ba694194e5a6de3c75f7`; the public pinned versions of
BoxTension, Sod-planar-1d, ShockBubble-Variant-2d and CRKSPHHydros.py match
the local files byte-for-byte. No dedicated Richtmyer/Meshkov driver was
identified in the functional Hydro directory.

Retain `NoneSwitch` as the baseline: it means the built-in reconstructed,
limited CRK viscosity remains active, with no additional evolving alpha
switch. Spheral's CRKSPH factory defaults to LimitedMonaghanGingoldViscosity.
Cullen–Dehnen is an optional extra multiplier in both frameworks, useful as
a controlled comparison, not a prerequisite or proposed cure for RM.

Recommended matched-test order:

1. `Hydro/BoxTension/BoxTension.py`: square or elliptical inclusion at equal
   pressure, density 4:1 by default, independently configured gamma values,
   optionally a common translation. Start with the default equal gammas,
   then 1.4 and 5/3. Measure pressure equilibrium, spurious kinetic energy,
   interface-shape drift and Galilean consistency. This extends the existing
   planar contact check to a closed 2D interface and corners/curvature.
2. `Hydro/Sod/Sod-planar-1d.py` with two NodeLists and unequal gamma values:
   the driver provides SodSolutionGasGas for an analytic wave/plateau
   comparison. Check pressure, velocity, density, wave locations and energy,
   with the built-in limiter and no additional alpha switch. Two NodeLists
   are essential: the one-list mode forces gamma2 to equal gamma1.
3. `Hydro/ShockBubble/ShockBubble-Variant-2d.py`: pressure-driven air/helium
   bubble, separate gamma 1.4 and 1.648 EOS, a curved density interface and
   shock-generated circulation. Match its actual ICs, sampling and reflecting
   walls before comparing bubble deformation, circulation and vorticity.
   Treat it as a numerical cross-code reference, not an analytic solution.

These are test specifications to reproduce, not newly completed runs or
guaranteed ready-to-run current drivers. BoxTension's CRK constructor omits
the currently required W argument; ShockBubble uses legacy CRKSPHHydro names.
Their driver interfaces need checking/refreshing before running current
Spheral. The gas–gas Sod driver uses the current W=WT CRK construction.
BlobTest-2d is a useful later wind/cloud shear control, but its two EOS use
the same gamma by default; it does not independently test unequal-gamma
dispatch. Water/gas variants require stiffened-gas support not supplied by
the current ideal-gas material table.

Important interface audit before claiming parity: Spheral's
`computeCRKSPHSumMassDensity.cc` includes pair contributions only when
`nodeListi == nodeListj` (line 66), and CRKSPHBase calls it for RigorousSumDensity.
Our `crkFactors` / core density path currently has no material-ID selection.
Matching per-material EOS alone therefore does not match all interface
discretization choices. Compare density/volume summation and mass/spacing
rules explicitly in the box/Sod controls before changing them. This is a
specific numerical difference to investigate, not a demonstrated cause of
the weak nonlinear RM roll-up. No physics or density policy was changed in
this survey.

### Recover the existing hydrostatic contact control (2026-10-19)

The user recalls that the hydrostatic multi-density test needed careful
choices of density when communicating across the interface. Recover and
preserve that behavior before introducing a new material-dependent density
policy. The existing `hydrostatic` runner case already supplies the square
inclusion geometry proposed above; extend it rather than duplicating it.

The current builder (`caseUtils/compressible/hydrostatic/sample.py`) assigns
different masses on one regular lattice, relaxes supports, computes the CRK
density, and initializes internal energy from that density at uniform pressure.
This density/energy consistency is part of the control. It currently inherits
all-zero material IDs and uses the scalar gamma, so its historical success
does not qualify the new two-EOS dispatch. First preserve the legacy result,
then label the two regions and initialize through the material EOS with equal
gammas, then unequal gammas. Add common translation and an ellipse afterward.
Track pressure error, spurious velocity/kinetic energy, material mass and shape.

Historical evidence in `docs/historic_plans/RESOLVED_PROBLEMS.md`, section 15,
records that an Eq. (76) cross-material mass substitution degraded this box
and made the then-unfixed Sod shock fail earlier. That experiment predates
the coincident-pair viscosity fix and is not a definitive rejection of every
interface policy; it does rule out treating that substitution as an untested,
obvious repair. The same record reports max velocity about 1e-5 through t=3
for the legacy static box. These are historical measurements, not fresh runs.
No existing density substitution or pair operator was changed in this audit.

### High-resolution campaign complete (2026-10-19)

The last queued run, `nonlinear128-cullen`, reached t=6.00052 in 10,259 steps,
1417.94 s wall at 131.18 ms/step. Final Fourier amplitude is 0.2139003,
upper-half circulation 0.04344795 and marker width 0.607654. Relative to the
same-resolution baseline, amplitude is 2.46% lower, signed circulation 5.28%
lower and marker width 6.85% higher. Energy drift is zero at the stored
float32 diagnostic precision. All three high-resolution campaign entries
completed successfully. The CPU comparison and report snapshot now include
all eleven controlled runs. These results do not establish an extra viscosity
switch as a solution to the nonlinear reference discrepancy.

### Material verification cases authorized (2026-10-19)

User: implement and run the new cases against literature/reference behavior,
with every run registered with the resident daemon. Extend `hydrostatic`
without changing its legacy defaults, add `twoGasSod` with an exact two-gamma
profile, and add `shockBubble` matching Spheral's pressure-driven variant.
Use reflected initial-condition tiling for its four slip walls. Record the
unit conversion (its CGS t=0.003 becomes nondimensional t=0.3), uniform-lattice
sampling choice and any difference from Spheral's default composite seeding.
The variant is not the Quirk–Karni Mach-1.22 experiment. Static contact/Sod
have exact targets; shock–bubble requires a matched numerical reference before
claiming quantitative agreement. Baseline limiter and density policy stay fixed.
Run recorded pilots/refinements sequentially with Remote controls, progress,
stop-on-failure, durable manifests and quantitative summaries.

### Material controls implemented and run (2026-10-19)

The user-authorized control suite is now registered in the runner: legacy
`hydrostatic` extended with explicit material EOS/ellipse/boost variants,
`twoGasSod` with exact profiles and a causally padded 1D domain, and
`shockBubble` using reflected periodic copies of Spheral's pressure-driven
variant. YAML configurations and Remote launch request bodies are provided.
`study_materialCases.py` supplies sequential daemon-visible pilots,
refinements and no-driver contacts; `analyze_materialCases.py` is CPU-only.
Catalogue refresh confirms all cases/variants are available to Remote.

Seventeen full controls ran through the resident daemon; sixteen pass their
stated gates and one fails only its pressure/quiescence qualification. Three
five-step setup controls and twelve CPU wave/geometry/configuration/queue
tests pass. An early diagnostic serialization failure was corrected before
the campaigns. Recordings, summaries, histories and final arrays are retained.
The exact Sod profile agrees with Spheral's independent analytic routine to
<8.5e-5 for both gamma choices (limited by its scalar root tolerance).

At nx=100, t≈7, the equal/unequal-gamma square has maximum velocity
9.89e-6/1.35e-5 and shape drift 3.34e-6/9.42e-6. Runtime is 70.46/67.45 s,
8.69/8.59 ms per step. The translated ellipse remains near its exact motion.
However, density is smoothed: relative particle L1 error 3.54% at nx=64,
2.28% at nx=100, with local relative errors about 82% near the light side.
This does not reproduce the publication's exact sharp density reconstruction;
pressure-balanced initialization alone conceals that distinction.

Two-gamma Sod density L1 errors are .005555/.002858/.001460 at nx=200/400/800;
pressure .006551/.003342/.001683; velocity .010930/.005610/.002978. The
approximately first-order global convergence supports EOS dispatch on this
shock/contact problem. Wall time is 8.99/14.65/25.92 s, 8.39/8.38/8.27 ms/step.
The equal-gamma published-state nx=400 control also passes its exact-profile
checks (14.54 s, 8.30 ms/step).

Both nx=200 shock–bubble initializations complete with 110,000 particles and
energy drift <1e-6: 120.54/108.51 s at 25.61/25.74 ms/step. Width and centroid
are sensitive to initialization/resolution, so these are not independent
nonlinear-reference agreement claims. No Spheral hydro executable was run.

**New isolated failure / next density audit:** the no-driver bubble initialized
with prescribed density develops final speed .3801 and pressure error .4315
at t≈.02 (peaks .7247 and 1.7895; min pressure .5558). The same geometry
initialized from the summed density stays at speed 7.07e-6, pressure error
1.41e-5. Both are finite, positive, energy-conserving and have unchanged
material masses; the prescribed case is retained as failed in Remote. This
isolates the density/energy initialization mismatch before any shock. The
first-stage EOS necessarily gives P=P0 rho_summed/rho_prescribed when density
is replaced while u retains the prescribed-density initialization. Summed
initialization maintains numerical pressure equilibrium but still smooths the
physical contact. Audit the cross-material density/volume policy against
Spheral and the earlier rejected mass substitution, using these regressions;
do not assume this alone explains weak RM roll-up or tune viscosity to hide it.
No pair force or density policy was changed in this control suite.

Detailed definitions, source links, results, limits and figures:
`reports/materialEOS/controls/README.md`. Full manifests are under
`results/materialEOS/controls_{pilot,refinement,sod_literature,contact}_2026-10-19`.

### Material density policy audit authorized (2026-10-19)

User approved continuing with the isolated contact-density correction and
regressions. Preserve the legacy density path as the default. Compare an
explicit same-material density sum, using Spheral's positive uncorrected
scatter-kernel ratio, with the paper's cross-material mass-replacement rule.
Keep geometric volumes, reproducing moments and force/energy neighbours across
all materials; filtering density must not decouple material pressure forces.
Use one configuration policy in initialization and every RHS, retain it in
configuration/reload and graph cache keys, and test list/grid/AD behavior.
First qualify the unforced bubble and sharp static density, then box/advection,
two-gas Sod, the historical lattice-shock regression, and shock–bubble/RM
only after those pass. All full runs remain sequential and daemon-visible.

### Material density audit results (2026-10-19)

Implemented opt-in `solver.crkDensityPolicy= sameMaterial | massReplacement`,
retaining `legacy` as the default. Initialization and RHS share the policy;
configuration/reload and CUDA graph invalidation retain it. All-material
geometric volumes, reproducing moments, pressure forces and the existing
viscosity limiter are preserved.

Both rules remove the prescribed-density bubble's startup pressure error
(43% → about 1.1e-5) and pass sharp static/advected contacts, the published
gamma=1.5 square through t≈7, exact two-gas Sod and the historical 220-step
lattice-shock regression. Twenty-seven numerical/configuration/queue tests,
eight graph tests, three trajectory roundtrip tests and four case contract
checks pass. The Remote managed recipe validates, its catalogue is refreshed,
and all physical campaigns were recorded and registered with the daemon.
The t≈7 square costs 9.23/9.59 ms per step versus 8.69 for legacy; RM timings
remain comparable to their recorded baseline.

The nx=64 RM comparison changes linear growth by only +0.61% for sameMaterial.
The nonlinear t=6 run still has a broad lobe without pronounced roll-up:
amplitude -2.53%, upper-half circulation +5.20% relative to legacy. The
contact correction is necessary for the regular sampler but does not explain
the full nonlinear discrepancy.

**Remaining qualification:** stationary equal-mass contacts with an abrupt
spacing change fail local pressure/quiescence under both rules (pressure
errors 27.7%/38.1%), despite finite, conservative trajectories. Keep these
failed records. The all-material number-density volumes smooth the sampling
transition; Spheral instead uses continuity-volume evolution. Next isolate
physical volume initialization/evolution and the conservative RK force's
consistency on irregular sampling, then compare an actual Spheral run. No
compiled Spheral hydro executable is installed, so source/analytic agreement
must not be presented as a cross-code simulation comparison. Do not promote
either density rule to a general default or tune viscosity to hide this issue.

Evidence, formulas, figures, reproduction commands and limits:
[density audit](reports/materialEOS/density/README.md). Explicit runner recipe:
`examples/configuration/richtmyerMeshkovSameMaterial.yaml`; managed launch:
`examples/remote/richtmyerMeshkovSameMaterial.json`. Independent nonlinear RM
reference agreement and general unequal-spacing material handling remain open.

### Separate particle provenance from physical material (2026-10-19)

User requested distinguishing emitters from actual materials. An emitter/source
label is passive provenance; material IDs select EOS and material-interface
rules. One sampled region may assign multiple materials by position, and
multiple source regions may share material 0. Separate rigid-body membership
as well: it currently also occupies `materials` on boundary/ghost rows.

Implement independent optional `emitters` and `bodyIDs` particle fields,
explicit material assignment in region initial conditions (default 0), and
retain both through state cloning, boundary ghost/body reconstruction and
trajectory/dataset reload. Move impact/source diagnostics and body lookup off
material IDs. Preserve legacy boundary-state loading through an explicit
fallback, without guessing new physical materials from old provenance labels.
Verify source relabeling leaves EOS/density/forces unchanged and material
assignment works within a single source. Keep this separate from the pending
volume-evolution correction.

Implemented and verified the separation. All nine particle-state classes
inherit keyword-only optional provenance/body fields, preserving their old
positional constructors. Region initial conditions accept integer `materials`
as a scalar, per-particle tensor or position-dependent callable; their
serialization now retains that map (including the callable, rather than
dropping initial conditions). Standard Sod source labels are passive; the
historical two-material shock wrapper assigns physical IDs explicitly.
Rigid-body construction, ghost creation and velocity BC selection use body
membership, and impact gap diagnostics use emitter IDs. State cloning,
trajectory fields and both dataset reconstruction APIs preserve the new IDs.
Old material labels are never automatically reinterpreted as physical species.

Thirty-eight CPU checks pass, including eleven identity-specific cases and the
existing density/reference/queue checks. Seven focused GPU checks pass
(three emitter-relabeling stage comparisons, three trajectory roundtrips,
one eager/graph comparison), plus four boundary/configuration regressions.
Relabeling emitters leaves CRK acceleration, energy rates, density and EOS
outputs bit-for-bit unchanged under all three density policies.

The daemon-visible recorded contact rerun reproduces the previous speed and
pressure errors exactly (4.16 s, 12.94 ms/step vs 4.11 s, 12.97 ms/step).
The two-source impact smoke run has one physical material, two source labels
and a working gap diagnostic (80 pre-contact steps, 4.29 s, 12.64 ms/step).
Both dataset reload APIs recover those identities. The lattice-shock
regressions have bitwise identical final physical arrays and conserved energy
(5.38/5.43 s, 9.92/10.08 ms/step, about 4–5% above the earlier recorded timings).
All records are under `results/materialEOS/identity_*_2026-10-19`.

This completes the identity separation; the equal-mass contact volume issue
and independent nonlinear RM reference remain the next qualification work.

### Spheral build and matched volume/timing experiment authorized (2026-10-19)

User requested building Spheral's compiled Python interface and, concurrently,
trying its volume evolution and density-evaluation timing on the pressure
contact. Keep the reference source pinned and build/dependencies isolated from
the simulation environment. Record build progress and any unresolved failure.
Audit RKCorrections startup/pre-step volume initialization as well as
ContinuityVolumePolicy before claiming a matched update. Implement an opt-in
CRK path preserving the legacy defaults; validate the equal-spacing and
equal-mass stationary contacts through the resident daemon, with video,
conservation/local-pressure gates and performance measurements.

Implemented `crkEvolution=spheral` as an opt-in explicit-midpoint experiment.
The controller audit shows a pre-step volume reset to m/rho, then one rigorous
material density sum, followed by continuity density and bounded volume
updates using updated rho inside stages. RK moments use evolved volumes.
Legacy summation remains default; graph requests fall back to eager and
Dirichlet functions are not qualified. Volume/density fields persist through
recording/reload. Support evolution and conservative force/energy are still
warpSPH's, so this is not full Spheral parity.

Four full daemon-visible recorded runs complete. Equal spacing, square and
unforced bubble pass. Equal-mass contact initial density error is now zero,
but final pressure error is 6.71% and peculiar speed .07715 at t≈.02, versus
27.66%/.04588 previously. It remains failed in Remote; energy drift is -2.38e-7.
A stage-only audit finds initial max acceleration 88.14 at exactly uniform
pressure/sharp density, isolating a remaining force-consistency issue before
any pressure disturbance. Do not tune viscosity or calibrate density to hide
it. The next actual Spheral comparison must match kernel and H conventions:
warpSPH B7 is degree 6 whereas the published BoxTension NBSpline defaults to
order 7; a common cubic spline is a straightforward shared choice.

44 CPU and five focused GPU checks pass, including updated-density volume
algebra/bounds, pre-step sum counting, generic initializer field allocation,
trajectory roundtrip, legacy graph and passive-emitter invariance. Recorded
wall time/step cost: equal-spacing contact 4.86 s/9.51 ms; unequal-spacing
contact 4.93 s/9.54 ms; square 14.04 s/9.59 ms; bubble 4.12 s/13.79 ms. These
were measured with a concurrent CPU reference build, not an isolated benchmark.

The pinned Spheral source/submodules, TPLs and isolated Python environments
are prepared, and the eight-job C++ compilation is running. All generated
files live under `results/materialEOS/spheral-build`; the original source and
hydro/RK/controller code are unchanged. A resumable build driver records
commands, logs and build-only fixes. No completed Spheral hydro trajectory is
claimed. See [volume evidence](reports/materialEOS/volume/README.md).

The long command session expired during C++ compilation. It was resumed
incrementally as a locked persistent background process (PID/driver log under
the build root), with an automatic import check after installation. Do not
launch a duplicate build; inspect `build.json` and `process.json` first.

The initial 1D/2D C++ library built, but install failed on unresolved 3D
symbols with 3D disabled. The reference is being rebuilt with upstream
1D/2D/3D enabled; TPLs and isolated Python environments are reused. This
remains a build in progress, not an installed/qualified reference yet.


### Actual Spheral contact comparison (2026-10-19)

The isolated pinned build and automatic 1D/2D Python/CRKSPH import check
completed successfully. User authorized proceeding with actual trajectories.
Use a common cubic spline, identical particle positions/masses/densities/EOS
and initial smoothing lengths; compare equal spacing and equal mass. First
hold support fixed to isolate force consistency, then compare normal support
evolution with its differences stated. All trajectories register with the
daemon, preserve recordings and report local pressure/speed, energy and cost.
An expected failed equilibrium gate remains failed; inspect it before launching
the next case. No force or viscosity tuning to obtain agreement.


Actual comparison complete: eight recorded trajectories to t=.02, all verified
in the daemon with readable recorded history. Common cubic kernel, identical
saved particle state, prescribed density/two gammas and midpoint dt=4e-5.
Equal-spacing fixed/adaptive contacts pass in both codes; unequal-spacing
equal-mass contacts fail local equilibrium gates in both and remain failed.
With fixed supports, initial acceleration is 161.4881/161.4885 (warp/Spheral),
relative field L2 difference 4.32e-5. Initial force response is therefore
validated against actual Spheral for this setup; full trajectory parity is
not claimed. Peak pressure errors are 7.48%/12.09% with fixed supports and
4.24%/11.93% with native adaptive support. Native supports already diverge
at the first RHS. Material masses/IDs stay unchanged and all energy drifts
remain below 1e-4. No production force/viscosity/support sources changed.

Reference startup found that generic SPHSmoothingScale(FixedH) evaluates
unregistered derivative fields; the dedicated upstream FixedSmoothingScale
works without patching hydro. Python source/build qualification is now
complete and the reference is executable. All videos rendered through vispy,
including postprocessing of reference CPU trajectories. Warp runner costs
7.88/7.96 ms fixed and 9.18/9.10 ms adaptive (spacing/mass), walls6.84/6.89
and7.52/7.48s; reference one-thread integration walls4.39/2.89 and4.56/2.98s.
Those workloads/timing boundaries differ, so no cross-code speedup is claimed.

Next: isolate moving-stage AV smoothing-length/coefficient/reconstruction
conventions and density/compatible-energy updates at shared support, then
actual published BoxTension/Sod and nonlinear RM. Unequal-spacing equilibrium
is not a passing oracle for either current conservative CRK implementation.
[Full report and reproducible scripts](reports/materialEOS/spheral/README.md).


### Support/volume consistency audit requested (2026-10-19)

The user points out directional gather/scatter consistency at a resolution
jump and asks to investigate volume symmetrization. First distinguish the
custom planar contact from Spheral's bundled BoxTension: no bundled
BoxTension trajectory has been run yet. Audit legacy number-density volumes
(gather), RK moments (scatter), force/energy pair antisymmetrization, and
the experimental m/rho volume path separately. The failed actual comparison
bypasses the legacy number-density estimator, so that estimator alone
cannot explain the observed initial acceleration. Evaluate constant/linear
RK reproduction separately from conservative force closure; do not change
only the kernel in the force or silently alter the production derivation.


Audit findings: the pinned bundled BoxTension default uses equal spacing
(dx=.01 inside and outside), whereas our failed planar control has a 4:1
spacing/support jump. Do not call this a failed bundled Spheral test.
Legacy apparent volumes are gather; moments/corrections and material density
are scatter. Generic kernel primitives accept symmetric modes, but the CRK
pipeline hard-codes these conventions and explicitly combines both directed
corrected gradients in its conservative force. Pair reciprocity is distinct
from uniform-pressure equilibrium. The matched m/rho-volume path bypasses
the legacy apparent-volume estimator and still exhibits the residual.

A dense float64, initial-only audit reproduces actual Spheral initial
acceleration to 9.51e-8 relative L2. All eight controlled volume/base-kernel
variants preserve constant/linear RK reproduction and exact pair-force
antisymmetry, but none removes the equilibrium residual. Physical volumes
with consistent symmetric base kernels in moments/gradients reduce max
acceleration 161.488→66.952; volume-only symmetry gives 141.489 versus 121.308
for gather number volumes under scatter moments. No dynamic fix is claimed.
The directed gradient row sum vanishes; the reciprocal/transpose sum in the
conservative pressure force does not. Next examine quadrature and dual closure
constraints together before AV/energy matching; preserve conservation and
linear reproduction. No simulation advanced or production solver changed
in this audit. [Detailed audit](reports/materialEOS/spheral/README.md#follow-up-directional-support-volumes-and-force-closure).


### Continue with the 1D reduction (2026-10-19)

The user requests the simpler 1D equal-mass/variable-spacing versus
equal-spacing/variable-mass comparison. The actual trajectories and closure
probes above are already 1D (500 and 800 particles respectively). Both
samplings reproduce affine fields to roundoff, but changing quadrature points
and supports need not leave the correction coefficients or higher-order error
identical. Hold geometry/volumes/supports fixed to distinguish passive mass
factors from geometric reconstruction; additionally compare native support
jumps against uniform support on the same irregular 1D geometry.


Fresh exact-double 1D geometry controls added to the stage-only probe:
equal-spacing/variable-mass acceleration is below 7e-12. Unequal-spacing
geometry gives 161.491 with spacing-scaled support, including at uniform
density (masses then vary with cell volume, removing the material jump).
A common coarse cutoff reduces max acceleration to 2.595 for the density-jump
case and 1.020 for uniform density. It also increases fine-side neighbor count,
so this does not isolate support asymmetry at fixed local resolution and is
not a fix. All retain affine reproduction and total pair-force conservation.
Reconstruction depends on geometry/volume/support, not mass or material at
fixed volume; coefficients need not be identical across changed quadrature
points. Stay in 1D for the next closure derivation. No trajectories advanced.


### Intermediate 1D operator qualification requested (2026-10-19)

Before further force changes, the user requests actual interpolation/gradient
accuracy for constant and linear fields across resolution jumps. Exercise
production warpSPH correction/operation calls, not only dense algebra.
Separate interface/bulk errors, float32/float64 precision, physical/legacy
volumes and gather/scatter/symmetric correction/operator conventions.
Use deliberately mismatched support conventions as a sensitivity control.
These are operator evaluations only, not simulation trajectories.


Production operator qualification complete. All 48 coherent configurations
pass: four 1D spacing ratios (1/2/4/8), two volume choices (physical/legacy
gather), three coherent support conventions (scatter/gather/kernel-mean),
float32 CUDA and float64 CPU. Actual crkFactors/computeCRKTerms/warpOperation
interpolation and direct/difference gradients are exercised. Constant/affine
maximum errors are collected separately near the interface and in the bulk.
Worst interpolation errors are 2.38e-7 float32 and 5.55e-16 float64; worst
direct-gradient errors 8.40e-5 and 5.69e-14; affine difference-gradient
errors 6.92e-6 and 9.11e-15. Difference gradients of constants are exactly zero.
Scatter correction coefficients/derivatives remain bitwise identical under
mass/density reassignment at fixed geometry/support/explicit volumes.

Independent dense evaluations agree to precision. Deliberately inconsistent
scatter corrections with gather evaluation produce order-one interpolation
and large gradient errors, so the harness has a working sensitivity control.
No production operator fix is indicated by these affine checks. Continue
with the reciprocal-gradient contribution in the conservative force, retaining
the directed operator reproduction tests as gates. No trajectory advanced;
probe walls 3.68 s GPU float32 and 10.31 s CPU float64 (including first compilation).
[Operator report/plots/reproduction](reports/materialEOS/spheral/reconstruction1D/README.md).


### Reciprocal force closure continuation authorized (2026-10-19)

Continue from qualified directed operators to pressure-force edge geometry.
Define antisymmetric edge weights q_ij=0.5 V_i V_j(G_ij-G_ji). Uniform
pressure requires sum_j q_ij=0; an affine pressure requires
sum_j q_ij(x_j-x_i)=V_i in 1D, with wrapped local displacements on the
periodic domain. Test these independently, including physical versus
Voronoi cell volumes, before changing production force/energy. Any candidate
must retain pair conservation, directed reconstruction and compact support.


Literature qualification corrected: section 3.1 and Appendix B of
Frontiere/Raskin/Owen 2017 explicitly describe the consistency/conservation
tradeoff. Fig. B.1 shows exactly the 1D affine-pressure discrepancy between
RK and conservative CRK; Fig. B.2 shows equal-mass BoxTension deformation.
The default bundled Spheral BoxTension remains equal-spacing; the paper does
include a failing exact-equilibrium equal-mass variant. Treat the observed
residual as a known formulation limitation unless a distinct implementation
disagreement is demonstrated.

Seven static geometry/refinement/volume controls (21 cases) now test compact
antisymmetric edge coefficients q. Enforcing only sumq=0 removes constant
pressure forces but leaves affine-pressure error. Enforcing sumq=0 and
sumq*r=V simultaneously gives constant/affine acceleration errors below 1e-9
on the tested grids while conserving pair force and instantaneous pair
pressure work. At 128 left/32 right particles, standard force max a0=51.677;
constant closure reduces it to 7.39e-13 but leaves affine error .316; the
two-constraint projection gives 1.86e-10/2.60e-10 respectively. Changing to
Voronoi volumes alone does not fix the standard force. Smooth periodic
pressure L1 errors decrease .007306→.001012→.0001488 for left counts 64/128/256
with the combined projection, versus near-constant 2.16 in the baseline.

This projection is a research construction, not a production patch: global
coefficient coupling, 39.7% relative edge changes, bulk changes, 2216 iterations
at 128 left and 13098/.264 s at 256 left. It retains compact interaction edges
and original directed reconstruction; force and instantaneous total power
residuals remain below 1.5e-15/2.3e-16. Time-discrete energy, stable density/
volume evolution and shock behavior are unqualified, so no trajectory is
advanced from it yet. Next investigate structured/preconditioned 1D edge
geometry before a coherent daemon-recorded dynamic candidate.
[Research report/derivation/results](reports/materialEOS/spheral/forceClosure1D/README.md).


### Practical 1D solve and later Spheral portability (2026-10-19)

User requests finding a solve, potentially testing it in Spheral later.
Investigate a periodic 1D cycle-space construction: repair divergence using
nearest-neighbor ring flux, then enforce affine moments with divergence-free
triangle currents. The triangle moment equation reduces to a weighted 1D
Poisson equation solvable with prefix sums and a periodic flux gauge. Keep
the interface independent of Warp/Spheral and retain compact support;
qualify geometric robustness and coherent pair work before dynamics.


A non-iterative periodic 1D solve is implemented as the independent NumPy
module scripts/crkClosure1D.py. Repair row closure using nearest-ring flux;
repair affine moment using divergence-free nearest/next-nearest triangle
cycles. The moment equation is weighted 1D Poisson, solved by prefix sums
and periodic gauges. Mapping/sorting adds logarithmic overhead; no iterative
global projection remains. All original compact edges are retained, but
coefficient changes are larger than the full least-norm projection (169%
relative L2 vs 40% at 4:1). Seven static controls have force errors below 1.2e-13.
The representative 160-particle solve is .118 ms versus 43 ms previously;
standalone synthetic-coefficient benchmark costs .141/.244/.963/3.95/23.35ms
at 320/1280/5120/20480/81920 particles.

A separate CPU 1D research model uses the same corrected q for pressure
force, volume divergence and local adiabatic P dV work, with rho=m/V and
explicit midpoint. It is not production warpSPH/Spheral CRKSPH and contains
no viscosity/shock handling. Ten tests pass, including jitter, permutations,
edge orientations, periodic translation, affine/constant force, momentum,
volume-rate, energy-power and entropy derivative checks.

Six full daemon-recorded trajectories pass with vispy videos: stationary
and advected equal-mass/two-gamma contacts, small-amplitude sound, and
finite-amplitude compression at three timesteps. Contacts retain P/rho
exactly; stationary speed<5.69e-15. Acoustic pressure/velocity L1 errors
against linearized targets<8.47e-9/7.20e-9. At finite amplitude .01 to t=.2,
energy drift decreases 4.52e-13→1.14e-13→2.88e-14 under timestep halving;
state difference ratios are 3.998–4.001, confirming second-order time
convergence in this smooth control. All histories/videos/statuses verified
in daemon. A stationary postprocessing error was repaired without rerunning
the physics; original failure record retained and terminal status annotated.
Integration walls 2.96/2.89/3.01s for the three 1000-step pilots; finite
sequence 2.98/5.99/11.92s (1000/2000/4000 steps), about 3 ms/step, excluding
post-integration video. Full geometry construction dominates the NumPy RHS.

The exact same closure module imports alongside installed Spheral1d and
reproduces saved static coefficients within 8.27e-14; this is interface-only,
not Spheral hydro evolution. A later Spheral test should share this module,
map unique internal/ghost pairs, and replace force plus density/volume/work
coherently; stale compatible-energy pair data must not survive a force change.
Next integrate the experimental stage/update path into a real solver and
repeat smooth/compressive controls before conservative shock dissipation
and multidimensional/RM qualification. Production solver defaults unchanged.
[Structured solve/derivation/dynamic evidence](reports/materialEOS/spheral/structuredClosure1D/README.md).


### 2026-10-19: two-dimensional force-closure qualification

User requested testing the construction in 2D. The ordered-ring prefix solve
is specific to 1D. Investigate a general compact-edge vector correction with
zero row sum and full first moment tensor V I. Measure feasibility, constant
and affine force accuracy, directed RK reproduction, pair power/momentum,
volume balance and solve cost on equal-spacing and unequal-spacing geometry.
Only advance a candidate after static checks; physical contact/acoustic
pilots must register with the daemon and retain trajectories and vispy video.
This remains an independent research model, not a production CRKSPH change.


The 2D research extension now has seven static controls, normalized jittered
circular two-gas contacts, oblique wave refinement and finite-compression
timestep studies. Constant/affine tensor constraints are feasible on compact
edges, but the untapered least-change projection has a finite neighbor-cutoff
jump: 4.12% coefficient change under arbitrarily small position perturbations.
It loses temporal order (1.11–1.35). Optional compact cubic column weighting
smooths corrections to zero at support boundaries, removing the jump and
restoring observed time orders 1.981–2.032. Weighted correction changes are
larger (76–110% relative L2 on planar static controls), so this is not a
production-ready correction. 19 focused 1D/2D tests pass.

Untapered small-wave pressure/velocity L1 error relative to amplitude improves
9.63/6.36%→6.08/3.59%→3.01/1.65% at fine n=24/32/48. The 1440-particle
CPU prototype costs 1.99 s/step, 95% in projection. Tapered n=24 finite
compression costs 137–141 ms/step versus 110 untapered. Tapered advected
circle preserves P to 2.39e-12 and peculiar speed to 7.70e-12, costing
138 ms/step versus 75.8 untapered. A longer n=24, A=.05 compression at t=.2
fails the pressure-reference gate (22.6% amplitude error); its failed status,
history and video remain recorded. Initial circle controls with volume sum
.972222 were preserved and repeated after normalizing quadrature volume to1.
All physical trajectories register with the daemon and retain vispy video.
The same 2D module imports under installed Spheral2d and reproduces saved
coefficients to 5.81e-15, an interface-only check. Production Warp/Spheral
hydro remains unchanged. Next: efficient multidimensional geometry, full
tapered spatial refinement and vortical controls before stage integration,
compatible work bookkeeping and conservative shock treatment.
[2D evidence, limitations and reproduction](reports/materialEOS/spheral/closure2D/README.md).


### 2026-10-19: GPU solve and complex-flow controls

User asks for a faster GPU 2D closure solve, with Kelvin–Helmholtz or a 2D
Sod tube as targets; RM is explicitly later. Implement an opt-in independent
GPU solve preserving tapered constant/full-tensor identities and pair work.
Measure residuals, coefficient agreement and real end-to-end step costs, not
just device solve time. First preserve existing contact/acoustic controls,
then use a daemon-recorded KH pilot (smooth flow), or add coherent conservative
shock dissipation before claiming a Sod result. Production defaults unchanged.


### 2026-10-19: research qualification takes priority

The user challenges the KH particle distribution and asks for founded observations,
including conservation tradeoffs, rather than pushing a correction to completion.
A reproducible case demonstrating remaining work is an acceptable result. The
GPU projection is an independent experimental model, not a qualified CRKSPH fix.
Its passing algebraic, positivity and energy gates do not validate the KH flow.
Assess saved states and manufactured geometry for packing and pressure torque;
retain failures and compare with the published CRKSPH limitations and the actual
Spheral contact results. No further trajectories or algorithm tuning in this
assessment. In a periodic box, distinguish minimum-image pair torque from the
angular momentum of wrapped coordinates and its seam transport.


### 2026-10-19: GPU results and qualification outcome

The GPU projection reproduces CPU coefficients and trajectories: the matched
576-particle acoustic coarse solve is 11.53x faster end to end (38.46 versus
443.46 ms/step). The coarse-preconditioned 4096-particle KH reaches t=1 in
333.80 integration seconds, but remains physically unqualified. The FFT variant
fails its unchanged residual gate at t=.263; retain that failure and partial video.
All nine daemon histories and video frame counts are verified. The real-CUDA
focused closure suite passes 24 tests. Production hydro remains unchanged.

The user's particle-distribution concern is measurable: minimum nearest distance
normalized by sqrt(evolved V) falls from 1 to .0781; 1.5625% of particles end below
.25, first appearing in the stored frames at t=.94. An exact unseeded shear map
retains nearest spacing 1, though it is not an exact seeded KH reference. Voronoi
area/evolved V ranges .568–1.547; this is a quadrature diagnostic, not a required
material-volume identity. Existing algebra/energy gates miss this crowding.

Static pressure torque does not support attributing this pattern to net angular
momentum loss. Full moment closure implies aggregate minimum-image pair torque
zero for arbitrary scalar P. Measured final torque is -1.02e-13; wrapped-coordinate
torque -6.30e-5 is explained by the periodic seam term. Individual pairs remain
noncentral, and the projection changes coefficients 69.1% in relative L2. These
checks do not qualify angular transport or nonperiodic angular-momentum evolution.
The crowding mechanism remains unresolved. The global periodic construction is
not a boundary-independent counterexample to the published CRK tradeoff.

The founded outcomes are the published/actual-Spheral unequal-spacing contact
limitation and a separate experimental KH case exposing insufficient qualification
gates. Retain the method exploration and reproducible evidence; defer production
integration and RM. Further work, if pursued, should be a controlled crowding
mechanism study, quantitative KH refinement and boundary-aware rotational test,
not tuning solely to pass current gates. No new trajectories were run for this
audit. [Observations, derivation and reproduction](reports/materialEOS/spheral/gpuClosure2D/README.md).


### 2026-10-19: resume published multi-material baselines

User authorizes resuming actual compiled Spheral BoxTension and two-gas Sod,
with equal/unequal-gamma controls, instead of further global-projection tuning.
Use matched saved initial arrays, explicit provenance for sampler/kernel/support
and stage-policy differences, independent equilibrium/exact-Sod targets, and
sequential daemon-visible trajectories with vispy videos. Retain any failed
qualification. Begin with a checked Spheral harness, then full square t=7 and
Sod t=.15 controls. Shock–bubble remains subsequent work.


The explicit `compare_spheralMaterials.py` harness and sequential
`study_spheralMaterials.py` driver are implemented. Identical prescribed states
are shared across solvers, with quintic NBSpline(5), nPerh=1.35, synchronous
midpoint/fixed dt and native adaptive-support differences recorded. Four pairs
are prepared: 100² BoxTension equal/unequal gamma to t=7 and two-gas Sod with
400/100 observed particles equal/unequal gamma to t=.15. The full reference
BoxTension baseline is now running in the daemon. Eight focused checks pass,
including actual 1D/2D kernel value/gradient agreement against compiled Spheral
samples and independent analytic Sod checks. Setup failures in boundary-object
ownership and summary serialization are retained; no Spheral hydro patched.
Remaining study stages are sequential and stop on failure/Remote Stop. Report:
[compiled multi-material baselines](reports/materialEOS/spheral/materialBaselines/README.md).


### 2026-10-19: first baseline results and volume-selection correction

Actual compiled Spheral and Warp both pass equal-gamma 100² BoxTension to t=7.
Peak P errors are 9.15e-7 / 1.86e-4, speeds6.08e-7 / 1.06e-5 (Spheral/Warp).
The first equal-gamma Sod harness used controller-default MassOverDensity;
both codes complete physical evolution but fail density/pressure L1 (.049/.042
and .051/.043). The source audit then found that the bundled Sod controller
explicitly selects RKSumVolume. This mismatch is corrected in separate new
cases, not retroactively in the failed recordings. Spheral's isolated volume
change reduces errors to .00274/.00368; Warp's native geometric-volume/summed-
density RHS also passes (.00313/.00388). Unequal gamma passes as well, with
.00281/.00369 Spheral and .00311/.00385 Warp. Native stage/support/AV differences
remain, and Warp's switch also changes within-stage density scheduling.

Local pressure errors still reach 11.3–11.5% in Spheral, 5.4–7.7% in Warp at
this resolution in the contact window ±.015, despite close plateau medians and
passing global L1. Document these observations without tuning viscosity or
claiming full trajectory parity/convergence. Eight completed trajectories
(including two failed mass/rho controls) have verified daemon histories/videos;
ten focused geometry/EOS/kernel/reference/summary checks pass. Unequal-gamma
100² square remains running, followed by its Warp counterpart. The opt-in
mass/rho volume/stage path is not qualified for this compressive setup.


### 2026-10-19: compiled multi-material baseline campaign complete

All four primary Spheral/Warp pairs complete: equal/unequal-gamma 100² squares
to t=7, equal/unequal-gamma two-gas Sod (400/100 observed particles) to t=.15
within one fixed step. All eight baseline trajectories pass their declared
independent targets. The two equal-gamma mass/rho-volume Sod controls remain
failed and are separately documented; ten trajectories/videos are verified
through the daemon. The study is complete with no physical worker left running.

Unequal-gamma square peak pressure errors are 7.94e-7 / 1.21e-4 and peak speeds
6.24e-7 / 1.26e-5 (Spheral/Warp), with unchanged material masses. Reference wall
939.73s at134.25ms/step (4 CPU threads); Warp wall75.77s, runner step9.24ms.
The earlier equal-gamma reference is1098.14s at156.88ms (1 thread), Warp80.96s,
step9.38ms. Geometric-volume Sod reference walls15.37/15.61s (10.24/10.40ms),
Warp17.00/17.19s (step8.49/8.61ms). Different timing boundaries and threads are
explicit; no cross-code performance claim. Ten focused tests and Python/
whitespace checks pass. No production hydro/viscosity changes were needed.

The founded conclusion is agreement with stationary-contact and bulk Riemann
behavior, with remaining local pressure errors5–11% at the Sod material contact
and a demonstrated dependence on volume policy. No full trajectory parity,
convergence, arbitrary-sampling equilibrium or nonlinear RM qualification is
claimed. Next original-suite case is an actual Spheral ShockBubble comparison;
local Sod refinement and controlled moving-stage audits remain separate work.
[Complete definitions, results, failures, videos and reproduction](reports/materialEOS/spheral/materialBaselines/README.md).


### 2026-10-19: actual shock–bubble comparison authorized

Proceed with compiled Spheral versus Warp for the pressure-driven air/helium
source variant. Audit cubic kernel/nPerh, mass/rho volume default, reflecting
walls, native XSPH and the separate upstream air NodeLists before dispatch.
Use common saved primary input; Spheral native reflecting walls and Warp's
reflected periodic copies. Start uniform lattice controls at two resolutions,
sequential, daemon-visible, with videos and durable failures. Measure bubble
shape/motion and a common offline circulation estimator; completion/conservation
is not quantitative agreement. Preserve differences and avoid force/viscosity
tuning to obtain a visual match. RM remains later.


### 2026-10-19: shock–bubble reveals retained positivity and boundary failures

Actual compiled source-variant controls at dx=1/80 and 1/160 are complete,
with common cubic/lattice inputs, two physical material NodeLists, XSPH off,
source MassOverDensity default, and documented native stage/support/AV
differences. The coarse reference stops at t=.2844 with negative local u/P
while total energy closes; the finer reference stops at t=.2195, so matched
refinement does not remove the failure. The latter actual failed state is
saved. These are harness results, not failures of the bundled radial/three-
NodeList/XSPH default driver. Independent time-step controls remain open.

Coarse Warp reaches t=.3 with positive thermodynamics but loses mirror-tile
symmetry and crosses the nominal slip wall. The existing case description
now states the actual symmetry assumption rather than claiming explicit wall
enforcement. This is a reproducible boundary qualification failure, retained
as failed; no coordinate folding, force projection or viscosity retuning
conceals it. Near .28, centre differs .00464, height 7.47%, circulation 24.66%,
and both particle sets show crowding. Finer Warp also reaches t=.3 while
failing the wall gate (first escape .1609, peak penetration .00471). At common
t=.2 the centre agrees within .0009, circulation within about 11%, but the
reference circulation changes over 50% with resolution: no nonlinear
convergence claim. Four full and two setup histories/videos are verified
through the daemon; 17 focused tests pass. No physical worker remains.
Explicit reflecting-boundary qualification and independent dt/source-default
controls remain; do not tune forces/viscosity to hide these failures. [Definitions and durable evidence](reports/materialEOS/spheral/shockBubble/README.md).


### 2026-10-19: interface separation observation takes priority

User points out that Warp's rolled interface forms clean separated particle
layers resembling surface tension, absent from the irregular Spheral
interface. Bulk centroid/circulation agreement understates this discrepancy.
Inspect saved-state interface mixing/gaps and pressure at earlier times, then
audit pressure versus cross-material artificial-viscosity/reconstruction
terms. Treat surface tension as an appearance/hypothesis, not a diagnosed
force. Boundary failures remain separate evidence and do not explain away
the interface difference. No force/viscosity fitting or new physical run is
needed for the first saved-state/source audit.

The saved-state/source audit confirms visible ordering by t=.15, before the
coarse wall crossing. Tracked same-material spacing is .451 Warp vs .642
Spheral then; unlike spacing is 1.140 vs1.065 and is smaller in Warp later,
so a larger capillary gap is not established. Both use ViVj pressure weights
and include cross-material force neighbours. Native Q differs in smoothing
length (R/1.778002 vs R/2) and gradient time level (current-stage Warp vs
cached Spheral by default). Compiled viscosity defaults exported. These are
concrete candidates, not a diagnosed cause. Prioritize a fixed-geometry
pressure/viscosity force/work comparison with complete native stage exports,
then controlled differences; do not infer interface accuracy from bulk
centroids or assume that boundary crossing explains the contrast. No new
physical runs or production hydro changes were made in this audit.


### 2026-10-19: viscosity parity controls authorized

User asks to try the identified viscosity-length and gradient-timing
differences. Run a 2×2 control at coarse resolution, identical saved inputs
and fixed step, current vs cached AV gradients and native vs spline-extent
viscous length, without modifying pressure forces or native defaults.
Sequential daemon-visible trajectories with videos and failure retention.
Evaluate matched early times before nominal-wall crossings; compare interface
geometry/pressure, bubble motion and common circulation, not a scalar pass
alone. Verify the cache schedule against compiled Spheral's midpoint source
before attributing a result to gradient lag. Refine if a material effect is
found; retain null results and no visual fitting.


### 2026-10-19: viscosity 2×2 and refinement controls complete

Four coarse controls (native/extent Q length × current/previous-RHS gradient)
reach t=.2 and pass operational gates; native/combined fine controls also reach
.2 but fail mirror-wall qualification. All six histories/videos verified in
the daemon, no physical worker remains. Opt-in settings preserve the native
default; native runs reproduce 1001/2001 diagnostic rows and 40 common stored
frames at each resolution bitwise. 48 selected tests pass; one CUDA-only
trajectory/reload check is skipped in the sandbox.

Neither change alone or together removes the interface layering. Coarse
combined circulation improves near .15 (12.1% error to1.17%) but worsens by
.2 (11.5% to17.2%). Fine pre-crossing .1 error worsens9.18% to13.09%; later
fine data are confounded by combined wall escape starting .124 versus .1609
native (peak penetration .00893 versus .00227). These two details affect
dissipation/circulation but are not a sufficient explanation of the observed
morphology within these controls. Do not promote them as a correction.

Implementation: opt-in viscosityKernelScale in Q only, cached AV gradient
with current continuity gradient; shared midpoint holder seeded from initial
state on every retry, final cache persisted, matching the audited Spheral
postStateUpdate chronology for zero initial velocity. Pressure-force formula,
coefficients and compatible energy remain fixed. Full native operator parity,
nonzero-velocity startup, restart/graph qualification and nonlinear accuracy
are not claimed. Next is the fixed-geometry pressure/AV pair-force/work audit
with complete stage fields. [Evidence/reproduction](reports/materialEOS/spheral/shockBubble/viscosityControls/README.md).


### 2026-10-19: common-state pressure/viscosity force audit authorized

Proceed with complete stage-field exports and static cross-code operator
replays on early Spheral- and Warp-derived states. Compare like/unlike
pressure and viscosity forces, pair work and normal/tangential components.
Use identical geometry, support, volume and thermodynamics for the controlled
comparison; keep native-stage results distinct from reconstructed inputs.
Verify pair sums against native total derivatives, and compare both donors
before assigning the difference to accumulated particle disorder. New physical
trajectories, if needed for complete stage capture, must remain sequential,
daemon-visible with video. Static operator evaluations do not advance time.

### 2026-10-19: native stage force/work audit complete

Three sequential daemon-visible coarse captures reach .1504 with operational
passes and 32-frame videos: Spheral native stages, Warp native stages, and an
independent Spheral half-step energy validation. Passive exports preserve all
numeric fields in 31 common stored frames per solver bitwise; Warp diagnostic
prefix is bitwise, Spheral scalar reductions differ <9.5e-14. An earlier
operator-interrupted native pilot is retained separately with its partial
video, not classified as a numerical failure. A no-timestep JSON rejection
exposed default viscosityKernelScale scalar serialization; float conversion
and untouched-default regression coverage added. No dynamics default changed.

Four complete native RHS stages per donor (t=0,.05,.1,.15), including actual
volumes/corrections/Q gradients/pairs, support controlled float64 GPU replays.
Native pair sums/formulas recover Spheral acceleration/dudt to roundoff.
On identical Spheral inputs, actual Warp pressure with rebuilt corrections
agrees to 1.6e-8–9.6e-6 relative L2 at nonzero times; Q agrees <3.8e-8 with
native Q length/gradient held fixed. Imported/rebuilt operators also close on
Warp donor data within float32 cancellation/correction error. Both use ViVj;
the stale config comment associating pinned Spheral with mean-volume weights
is corrected. Cross-material pressure forces largely cancel like-material
forces; a uniform-P static residual exists on both evolved geometries. There
is no evidence here for an extra Warp-only pressure-pair surface-tension term.

Raw Q heating differs (one-sided Spheral versus symmetric Warp), producing
only .13–.99% thermal-rate differences here. The much larger difference is
compatible-energy partition: Spheral's sign-filtered node raw-dudt weights
versus Warp entropy weights/equal unlike-gamma work. At fixed fields the
instantaneous compatible thermal rates differ ~15%/.05, ~40%/.1, ~60%/.15
on either donor. Unlike-only weight changes give ~2%, ~21–23%, ~18–21%;
like-material weighting also matters. This is not a finite-step energy error
or a causal trajectory result. Spheral's bridge is validated against the
actual compiled half-step update (max energy error 3.6e-15); Warp fractions
match captured native fractions to 1.1e-7. Native ghost raw-dudt BC fills are
required in Spheral weights. Predictor and final compatible-work chronology
also differ. 24 focused configuration/reference/bridge tests pass.

Next: isolate the full compatible-energy policy, including midpoint predictor
and native one-sided Q heating, in recorded trajectory controls with fixed
support/volume/Q settings. Keep conservation, packing, pressure, circulation
and deformation gates; do not accept closer pictures alone. Boundary and
reference positivity limitations remain open. [Audit and reproduction](reports/materialEOS/spheral/shockBubble/forceAudit/README.md).

### 2026-10-19: compatible-energy trajectory controls authorized

User requests the full energy-policy/predictor control and flags the arbitrary
Butcher timestepper migration as a possible source of thermodynamic divergence.
Audit stage construction, weights and rerun semantics; isolate policy versus
predictor effects without changing defaults. Validate compiled-reference stage
increments and pair conservation before sequential daemon-visible bubble runs.

### 2026-10-19: compatible-energy controls and temporal coupling evidence

Implemented opt-in crkEnergyPartition=native/spheral, crkEnergyPredictor=
raw/compatible and oneSidedWork. Native defaults remain selected; new
partition/predictor controls are restricted to compatible CRK explicit midpoint
and the existing periodic-fluid Spheral evolution mode. Only the latest RHS
is retained until each state update, then released; trial work uses the
trial multiplier and initial velocity, not the rerun whole-step half velocity.
Production helper reproduces compiled Spheral half-step energies within 1e-12;
actual GPU one-sided dudt matches native reference to <7e-9 relative L2.

Audit of historical pre-X6 per-stage sums and current running sums finds
roundoff agreement for Euler/midpoint/Heun/RK4/signed weights and list reruns,
with unequal-mass pair energy conservation. Earlier inspected source already
used raw predictors. No algebraic X6 regression is established. However, a
smooth two-mass counterexample through the real Butcher midpoint driver shows
raw prediction converging at order 1.008 to the compatible ODE, versus 1.999
with compatible prediction; both conserve energy to ~4e-15. This demonstrates
a thermodynamic stage-coupling mechanism, not a full-SPH order result or a
historical attribution. Actual nonlinear midpoint/final-reset wiring is tested.

Ten sequential daemon-visible GPU runs and videos complete: six coarse to .2,
a native repeat, three fine controls to .1. Coarse weights alone change tracked
interface spacing near .15 from .451 to .558 (Spheral .642); predictor-only and
raw-Q-only remain .451. Weights+compatible predictor reduce position RMS by
40% near .15 and 34% at .2, but circulation ranking changes with time. Fine
full energy control improves packing .541→.576 (reference .607) and .1
circulation error 6.33%→2.25%, while position RMS worsens ~18%. No correction
is qualified. Strict wall crossings occur for coarse weights-only (.1956,
peak .000407) and fine policy (.0692, peak .001484); older quarter-spacing
operational gates pass all ten. Mirror symmetry is lost even without crossing,
and full coarse minimum energy falls to .0396 despite global drift <1.01e-6.
Keep these limitations; closer pictures do not qualify dynamics.

Fresh native repeats 1001 diagnostic rows / 41 frames bitwise. Adding the raw
work struct/branch changes compilation: old native recordings are not bitwise
reproduced (first stored velocity discrepancy 7.6e-10, later amplified).
All A/Bs use the fresh baseline; inferred float32 code-generation sensitivity
is documented, not hidden. Warm steps ~10.1 ms coarse, 12.1 ms fine; energy
controls cost ~2% coarse / 5–7% fine. 44 distinct focused tests pass and one
CUDA-only roundtrip skips in sandbox. No physical worker remains.

Next priorities: explicit reflecting-wall qualification, smooth compressible
spatial-scheme temporal-order test, then combined energy/Q-length/cache controls.
Do not change defaults or claim arbitrary-Butcher thermodynamic order from
energy conservation. [Full evidence/reproduction](reports/materialEOS/spheral/shockBubble/energyControls/README.md).

### 2026-10-19: paper clarification and blob partition comparison authorized

User correctly notes that the CRKSPH paper explicitly uses the derivative
energy equation for interim states and compatible work for final energies
(section 3.3, after equation 63). Retract the implication that this differs
from the paper or is by itself a demonstrated implementation defect. Current
pinned Spheral and the paper differ in predictor and partition choices; the
algebraic counterexample concerns a different compatible ODE, not a validation
failure of the published CRKSPH method. Correct report framing accordingly.
User requests comparative blob runs: implement the published 2D periodic
wind–cloud geometry and hold the raw predictor, force/Q/support/volume settings
fixed while changing entropy versus sign-filtered dudt partition. Daemon and
video required; assess cloud mass fraction, conservation and thermodynamics.

### 2026-10-19: blob partition controls completed

Added the `blob` runner case/configuration: periodic [0,40] x [0,10], Mach-2.7
wind over a density-10 cloud, common gamma=5/3, rounded near equal-mass cloud
lattice from the pinned Spheral driver. Eight sequential CUDA trajectories
use identical 17,524-particle arrays, daemon registrations, progress/watchdogs
and vispy recordings; daemon histories/statuses and videos verified.

Raw predictor fixed: native entropy and entropy with common final-work sign
both reach 8 KH times (9290 steps, 122.46/120.87 s, 11.23/11.10 runner ms/step),
max energy drift 1.90/1.94e-6, momentum <1.19e-6, identities/material masses
unchanged. Transplanted current-Spheral dudt fractions fail with negative u
at t=.095. Half/quarter dt repeat failures at .09625/.09875; captured pair
work cools a cloud particle past zero while entropy fractions on that same
work give net heating. Compatible predictor plus one-sided Q work also fails
non-finite at .08; it changes multiple policies and is not full Spheral parity.
Failures retained without floors; exception video teardown segfaults after
state retention, so captured frames encoded offline. No worker remains.

This is a comparison of partition formulas inside Warp, not actual Spheral
integration. Both entropy runs also lose the thresholded cold-cloud mass
before 1 KH while retaining a compact material clump. Half published ambient
resolution and remaining support/Q/volume differences preclude literature
agreement claims. Defaults retained. Next decisive check: actual compiled
Spheral blob on these inputs, then isolate remaining operators and refine.
57 focused tests pass, one CUDA test skips; whitespace checks pass.
[Evidence, formulas and reproduction](reports/materialEOS/spheral/blobPartition/README.md).

### 2026-10-20: compiled blob reference authorized

User requests actual compiled Spheral on the saved blob inputs, noting the
persistent surface-tension-like Warp morphology. Preserve the previous Warp
trajectories, copy inputs byte for byte, keep physical duration and dt, and use
native compiled Spheral CRKSPH with recorded policies. Compare cloud thresholds
and interface shape/packing at matched times, without inferring a physical
surface tension coefficient from the visual observation. Register with daemon,
retain failures and render recorded trajectories via vispy.

### 2026-10-20: compiled blob references stop before nonlinear evolution

Six actual compiled Spheral references on byte-identical saved 17,524-particle
inputs are complete as failed partial trajectories, with daemon histories and
vispy videos verified. Cl=Cq=1 dt=.0025 fails at t=.015 (6 steps); high-cadence
replay is bitwise identical. Half/quarter dt fail at .015/.01375, dt=.0001 at
.0138. Current CRK constructor coefficients Cl=1.5/Cq=.5625 also fail at .014375.
Max global energy drift <8e-15, identities unchanged; per-particle u becomes
negative. Wall/step ~228–237 ms on four CPU threads; 1.37 s original through
31.43 s small-dt control, excluding startup/post-rendering. No worker remains.
This is a matched-input harness, not the bundled BlobTest default initialization:
initial H iteration is disabled to preserve supports, fixed dt and SynchronousRK2
are explicit. Do not claim Spheral's bundled test or published method is disproved.

At common t=.0125, shapes nearly identical, but selected outer-cloud pressure
peaks differ: Spheral 7.3849, Warp entropy 16.6828, Warp dudt 7.9420. Minimum cloud
u=.02059/.14584/.09807 respectively. This is a concrete thermodynamic difference,
not proof of the mechanism behind later surface-tension-like packing. No completed
Spheral reference exists here for late-time morphology. Keep that observation
unresolved; do not promote the dudt transplant, which also fails in Warp.
Next isolate startup support equilibration/native energy behavior or a separately
labeled bundled-driver initialization control to obtain a qualified reference.
25 focused tests and whitespace/compile checks pass. No dynamics defaults changed.
[Evidence, figures, raw paths and reproduction](reports/materialEOS/spheral/blobReference/README.md).

### 2026-10-20: reference-first reset agreed

User agrees to establish the bundled Spheral CRKSPH blob with native initialization,
integrator and timestep policy first, then export its initialized state/settings
for Warp. Suspend further matched-input policy ablations until that reference is
qualified. Instrument the shipped driver for daemon/video/state capture; preserve
its source and record any API compatibility repairs separately from numerical
changes. Earlier failures remain facts about the custom harness only.

### 2026-10-20: bundled driver audited; no qualified reference obtained

Executed the actual bundled 2D BlobTest driver via an explicit adapter, without
injecting Warp input. The source has stale API calls/imports and an independent
FSI/GSPH/SPH branch that overwrites its CRKSPH selection. Minimal API and exclusive
solver-dispatch repairs are recorded; the upstream source is untouched and actual
CRKSPH2d is asserted/logged. No numerical policy tuning. Native startup H iteration,
CheapSynchronousRK2, adaptive dt/cfl=.25 and requested 2.5 KH are retained. H startup
hits 50 iterations at deltaH~.01109, changes supports up to 34.6%, and preserves
positions/mass/rho/u. Generated and initialized states/full H tensors exported.

Strict added positivity observer stops at t=.01434265, 18 steps. To distinguish
that from native continuation, a second execution flags negative u without
repair or early positivity stop. It runs 1450 steps to t=3.26471234 (1.12456 KH),
then hits the native dtMin=1e-5 and a velocity spike to 1789; the runaway guard
halts it. Minimum u=-3.09e6, despite energy error 2.02e-14 and fixed material mass.
This is concrete runaway behavior, not merely an extra positivity criterion.
It does not establish a defect in published CRKSPH or identify the underlying
cause in this version. No working reference is claimed and Warp is not rerun.

Continuation wall304.69s (~210.13ms/step), strict trace3.64s (~202.36ms/step),
excluding startup/post-video; both are four-thread CPU runs. All 154/11 recording
frames/videos and failed statuses verified against daemon. Native and observer
cold-mass diagnostics agree at all 144 common samples to <1e-12. Fourteen focused
tests pass; compile/whitespace checks pass. No physical worker remains.
Next obtain a verified upstream version/setup before further Warp energy/volume
ablations or interpreting its surface-tension-like interface morphology.
[Full source repairs, raw paths, figures and reproduction](reports/materialEOS/spheral/blobExample/README.md).

### 2026-10-20: user retains 2D; audit the paper specification itself

User prefers 2D and asks about the published setup. Re-read arXiv1605.00725v2
sections3.3/3.4/4/4.5.1 directly. The 2D paper is explicit: periodic40x10,
R1 cloud centered(5,5), rho10/1, P1, gamma5/3, ambient Mach2.7 +x,
ambient512x128 uniform lattice with mass-matched cloud, snapshots1/4/8 KH,
tKH~2.903. More importantly, its numerical method differs from the bundled
example: seventh-order B-spline W7 (supporteta4), four radial neighbors,
Cl2/Cq1, inverse number-sum volumes Eq75 and corrected material-aware density
Eq76, entropy-compatible final work with raw interim energy, modified RK2
and CFL.25. Current runs matched geometry or current-code defaults, not this
complete published algorithm. Paper itself cautions that clipping-seeded
perturbations limit strict late-time convergence/visual comparisons.

Keep 2D. A deliberately paper-configured reproduction is the relevant target;
list and qualify each unsupported policy instead of treating latest Spheral
factory/driver defaults as the paper. This audit launches no new trajectory
and does not claim any listed setting fixes the observed runaway.

### 2026-10-20: paper-matched 2D reproduction authorized

User requests the paper-matching approach. Build an isolated Spheral 2D paper
policy overlay (entropy partition, inverse number-sum volume, corrected
material-aware Eq76 density), explicitly configure W7/nh1/Cl2/Cq1/CFL.25 and
raw-interim compatible-final RK2, and verify actual loaded policies before
physical qualification. Keep original binary/source baseline intact. Run
daemon-visible with video; qualify a reduced resolution first, then512x128
to8KH only if the short checks pass. No unreported fallback settings or floors.

### 2026-10-20: paper-configured overlay qualifies coarse case; published resolution running

Implemented isolated strong 2D policy specializations against the pinned native
Spheral ABI; original source/objects/libraries and production Warp dynamics
remain untouched. W7/extent4/nh1, Cl2/Cq1, CFL.25, raw CheapRK2 prediction,
entropy-compatible final work, Eq75 uncapped kernel-sum volume and corrected
material-aware Eq76 density are explicit. Volume is refreshed at **every RHS**,
including midpoint, not only the full-step reset. Cache latest RHS entropy
before native integrator state restore for final pair work. Runtime class,
loaded library maps/hash and five policy counters verify dispatch. Hash-named
DSOs are immutable and builds record source/base-library hashes and command.

128x32 ambient (4380 particles) completed8KH in1344 steps:109.62s integration/
diagnostics/publishing,81.56ms/step on4 threads, plus12.61s video. Minimum
u=.04594 and rho=.19883; maxspeed4.238. Maximum relative energy drift1.09e-8,
momentum7.36e-9; material masses fixed. Native startup H converges in7 iterations
(deltaH3.10e-5). Independent analytic B7/KD-tree Eq75/Eq76 and constant/linear
checks on2254 real particles agree to~1e-8. The second/moving midpoint changes
volume9.77%, still passing volume/reconstruction checks. Nine compiled entropy
algebra cases agree exactly with independent positive-entropy fractions and
complementary shares. All145 HDF5/daemon/video frames verified; seventeen focused
tests pass. Original-tag and paper all-gas cold-mass measurements are retained
separately. Coarse cold mass vanishes at3.948KH, earlier than paper~6.5KH:
operational qualification is **not** literature agreement.

The sequential foreground study has advanced to the published512x128 ambient
case (70181 particles),8KH, daemon name **Paper policies Spheral CRKSPH blob
n512**, ID9a245ef2645d44a88f62c37bfa488a66. It has passed the native-default
failure time with positive energy; full nonlinear agreement remains pending.
User flags the viewer's8.5-hour estimate. Measured recent full-resolution cost
is1235ms/step near.149KH, dt≈.00088: fixed-rate extrapolation gives8.8h remaining.
Reference overlay volume/density/final-work reductions are serial CPU loops;
native force/RK retain4 threads. Coarse dt grows substantially later, so the
early-rate ETA may shrink, but the full runtime is not yet known. Run unchanged.
Initial and moving-midpoint audits on4073 real points already pass(<1.2e-8).
An attached progress watchdog interrupts on <.001 simulated advance/minute or
120s without log output; numerical guards preserve failed states. Study owns
its child and automatically renders/analyzes/verifies after completion or
failure. No further physical cases are launched concurrently.

Raw artifacts/stages/PIDs/logs:
`results/materialEOS/blob_paper_2026-10-20/`; overview/provenance, limitations,
coarse density/material/diagnostic figures and qualification:
[paper-configured report](reports/materialEOS/spheral/blobPaper/README.md).
Earlier per-step-volume prototypes and recording-finalizer failures have videos
and remain explicitly superseded. Retained native support, table, timestep and
viscosity implementation choices are listed; this is not a verified historical
2017 binary. Compare full-resolution density/shock structure, cloud mass-loss
history and fragmentation before interpreting Warp's interface discrepancy.

### 2026-10-20: user redirects to short paired comparison before long reference

The user correctly prioritizes comparing Warp against the cheap, qualified
reference before spending hours on the512 case. Sent SIGINT to the owned native
process; stopped at600 steps, t=.5148795781=.177355KH, wall744.64s (~1241ms/step).
This is an intentional interruption, not a numerical failure; userStop.json
distinguishes it from the driver's generic failed/KeyboardInterrupt status.
Study rendered and analyzed the partial trajectory before exiting. Defer the
published-resolution case to a later user-selected window; no scheduled job.

Reuse the completed128x32/1KH reference and its actual initialized arrays/H.
Run current Warp on the identical arrays with seventh-degree B spline (Warp
name B8, Spheral W7), nominal compact support4, matched Cl2/Cq1 and physical AV
length R/4. Keep existing Warp dynamics; explicitly list remaining density
reset timing, raw/compatible energy and support/integrator differences. Audit
initial geometric operators before advancing. Both trajectories daemon-visible,
with videos and side-by-side thermodynamics, cloud mass and interface plots.

Completed this short comparison:4380 particles including316 cloud; actual
Spheral initialized arrays copied exactly. Static Warp volume/density agree
with native Eq75/76 to1.1e-8 and corrected coefficients to1.83e-7. Current Warp
uses existing summation/massReplacement/native entropy-Butcher options with
B8(seventh-degree W7), nh4, Cl2/Cq1 and Q lengthR/4. No production force or
energy formulas changed. Native density/stage/support policy differences
remain explicit, so no single cause is inferred from trajectory discrepancies.

At1KH, cold mass fraction is.518987 Warp vs.512658 Spheral (two cloud particles).
Curve RMS difference1.43 percentage points, max3.16 after.01KH (linear native
history interpolation onto Warp times). Both positive; Warp minimum u.05608
and rho.2186, maxrelative energy2.44e-15, material mass fixed. Early shock/cloud
structure broadly similar; local thermodynamic/support and particle-position
differences remain (per-coordinate final position RMS.0911). This is not proof
of nonlinear or literature agreement. Compare cheap controls before escalation.

Warp2904 fixed-dt steps to exact1KH:119.38s wall including startup/compilation/
video,41.11ms/step wall and36.57ms runner mean. Reused adaptive Spheral525 steps:
41.05s integration,78.19ms/step, different timing scope. Both daemon histories
and videos verified (60 Warp/63 Spheral frames). Seventeen focused tests and
compile/whitespace checks pass. No physical worker remains. Full512 run remains
deferred with no overnight schedule; native restart output was disabled, so a
later full reference starts fresh rather than resuming an unsupported state.
[Paired report and figures](reports/materialEOS/spheral/blobPaperShort/README.md).

### 2026-10-20: provenance and short comparison matrix requested

User asks how "Warp" the prior result was. It ran modified working-tree
CRKSPH, not pristine HEAD0f1b2d3 or default policy settings: native summation,
native Butcher-compatible energy, experimental massReplacement density and
AV lengthR/4, paper B8/W7/nh4/Cl2/Cq1 parameters. Spheral evolution, alternative
energy partitions, cached Q gradient and closure corrections were disabled.
Equal-gamma material EOS plumbing is active; no unlike-gamma branch is tested.

Build a durable matrix/provenance report at fixed128x32/1KH. Include native
policy defaults, density×Q-length×final-energy core factorial, predictor/Q-work/
pair-volume/support/gradient/precision/initial-H controls, timestep replay and
half-grid, and volume/density timing alternatives. Existing validation couples
energy controls to an unrelated Spheral density mode; make those opt-in knobs
independent for midpoint-only ablations instead of changing multiple policies
and attributing the result to one. Add explicit experimental paper density
schedule (Eq76 once/step, continuity within stages, geometric Eq75 every RHS).
Keep defaults unchanged, test the algebra/configuration, snapshot relevant
working-tree sources/hashes and mark new experimental combinations clearly.

Use the qualified reference's525-step grid for the main matrix; retain the
previous fixed-small-dt result and test half-grid sensitivity. Sequential
daemon-visible physical runs with video/progress/watchdogs. Individual numerical
failure is an observation, not a reason to discard independent matrix rows;
Remote Stop stops the queue. Include actual times, mass/conservation/minimum
thermodynamics, local particle/interface differences, configuration and timing.
Full512 remains deferred. Native cheap-vs-fresh integrator comparison must
preserve raw predictor and entropy final work, otherwise mark it confounded.

### 2026-10-20: comparison matrix completed; provenance and unresolved differences

Completed37 result rows including one reused previous run and native overlay
equivalence:29 operationally qualified,6 thermodynamic failure rows(3 controls
and half-grid confirmations),2 first-step recorder-error attempts. Two rejected
setup launches had no steps and remain separately documented. All38 recordings
including the reference anchor verified against daemon histories/videos; no
physical worker or overnight automation remains. Full512 remains deferred.

Prior success used working-tree native summation/Butcher energy plus
massReplacement and AV lengthR/4, paper kernel/Cl2/Cq1, copied reference H and
material EOS table; it was not pristine/default CRKSPH. Closure corrections,
Spheral stage mode, cached gradient and alternative partition controls were off.
Working-tree/core/integrator source hashes/snapshots and native loaded-library
hashes are retained. New experimental paper density schedule and independent
midpoint energy/gradient combinations are explicitly matrix-only selections;
defaults unchanged. HDF5 startup-field and public-option schema repairs did
not change the successful native flow; their failed attempts are preserved.

Default density/Q-length policies are stable at the common B8/nh4/Cl2/Cq1
setup, cold mass.4810. Matched selections on the native replay grid give.5253,
versus reference.5127; smaller fixed dt.5190 and half-grid.5127 show residual
time sensitivity. AV length moves endpoint mass more than density/final-entropy
partition in the core8-case factorial. No parameter was fitted to the reference.
The coupled current-Spheral m/rho-volume/same-material density stage bundle
gives zero threshold cold mass at1KH despite positive states and conserved
energy; frozen supports give.3797. This is not evidence of physically correct
mixing. Mean pair-volume weights preserve the matched endpoint cold mass but
increase global position RMS from.0899 to.1325; bulk mass is insufficient.

Node-dudt controls fail the positive-energy criterion: Warp raw/non-finite
at.20666KH and compatible/non-finite at.20833KH; half-grid raw fails at.20666
and compatible reaches negative u at.20750. Spheral-side fresh RK2 node-dudt
control reaches u=-2.31e-6 at.07794KH; half-grid remains negative at.07857KH.
Native entropy counterpart on the **identical compiled overlay/input/grid**
completes positive. Native is stopped by our strict observer, not an assertion
that unmodified Spheral crashes. EOS invalid-state handling differs; these are
admissibility observations, not a universal partition-failure theorem. The
native entropy overlay revision check agrees to~3e-11 in density,~3e-12 position.

Combined paper density/entropy/one-sided/cached-Q controls do not eliminate
local disagreement: cloud-position RMS remains.242 versus native reference
(.236 against fresh-native), similar to matched Warp.247/.244. Scalar-EOS
compatibility dynamics match material-table counterpart to~4.6e-10 in density
on the prescribed grid, but its unrefreshed stored pressure differs by up
to.0247; this does not cover adaptive timestep feedback. Unlike-gamma EOS,
all switch interactions, exact native support evolution and nonlinear mixing
remain unqualified. No angular conservation claim for this periodic box.

Typical float64 replay runs25–29s (~48–55ms/step on525steps), half-grid43.18s/
1050 (~41.12ms/step); original fixed-small-dt119.38s/2904(~41.11ms/step), scopes
include startup/video unlike native integration-only timings. Float32 has
~1.1e-7 energy drift and cold compilation overhead. All48 focused checks pass;
compile/whitespace checks pass. Useful next experiment freezes common later
states/geometry/H and compares support/RHS/pressure/AV forces before another
expensive reference. [Full matrix, source provenance, figures and CSV](reports/materialEOS/spheral/blobMatrix/README.md).

### Standard hydrostatic mass-rule A/B authorized (2026-10-20)

User recalls the historical hydrostatic degradation with m_i substitution and
requests a fresh isolated comparison. Run standard hydrostatic nx=200, rho=1/2,
gamma=5/3, default B7/nh=4/Owen/midpoint/native compatible energy/default AV,
CFL=.3 through t=3. Label both phases with identical gamma EOS entries in both
runs; vary only crkDensityPolicy legacy versus massReplacement. Preserve the
standard sampler's pressure-balanced initialization using each selected density
rule. Thus geometry, masses and supports match, but initial density and u may
differ by policy; retain both initial states. No volume, work partition, Q-length
or support experiments. Both physical trajectories registered with the daemon,
video/progress/storage/watchdog enabled, sequential. Harness phase:
`scripts/study_materialCases.py --phase hydrostaticPolicy`.

Initial t=3 pair completed: both 2735 steps, max speeds 1.7681e-5 legacy
and 1.7428e-5 replacement; pressure errors 3.1269e-4 and 2.6226e-4.
Only solver difference is density policy; initial x/h/m/IDs/velocity match
exactly, P=1 exactly, policy-consistent rho/u differ. Historical degradation
not reproduced in this window. Extend the same pair to the case default t=10
to avoid relying on a shortened window. Early results preserved in
`reports/materialEOS/hydrostaticDensityPolicies/t3/`.

### Hydrostatic density-only A/B complete (2026-10-20)

The historical degradation is not reproduced in the current standard case.
Both nx=200, rho1/2, gamma5/3 runs reach actual t=9.998414 (default tLimit10),
9116 steps; default B7/nh4/Owen/RK2/native compatible energy/default AV/CFL.3,
float32/eager. Resolved solver configs differ only in crkDensityPolicy.
Peak speed: legacy1.93165e-5, replacement2.34722e-5. Peak pressure error:
1.95467e-3 vs1.27339e-3; maximum particle displacement .01199 vs.01580 spacings.
Peak density relative L1 .00459984 vs4.55438e-5. Energy drift remains below3e-7.
Both pass every existing qualification gate, with no visible square deformation.
Legacy134.16s/11.51ms per step; replacement133.57s/11.56ms, no material cost increase.
Full trajectories each9116 frames and videos each366 frames verified against
the daemon (both completed); earlier t3 controls each2735 frames/111 video
frames also verified. No workers remain.

Important scope: both phases use equal-gamma EOS labels; initial x/h/m/IDs/v
match bitwise and P=1 exactly. The standard sampler constructs u from the
selected density policy, so initial rho/u differ. Algebraically retaining
legacy u with replacement rho would start at P=.785457–1.309265; that is not
a pressure-balanced contact. This is not a third trajectory nor a proven
explanation for the older result. Static equal-spacing success does not
qualify unequal-spacing or dynamic blob behavior. No solver defaults changed.

Report: `reports/materialEOS/hydrostaticDensityPolicies/README.md`; early
window retained under t3/. Raw full pair:
`results/materialEOS/hydrostatic_density_ab_t10_2026-10-20/`, with source
hashes/snapshots and tracked-source patch. Reproducible harness phase
hydrostaticPolicy plus offline scripts/analyze_hydrostaticDensityPolicies.py.

### Model-choice validation standard and remaining priorities (user, 2026-10-20)

For every model change: (1) trace the exact paper equations, definitions and
normalizations; (2) cross-check the corresponding pinned Spheral source and
behavior, explicitly separating current-code choices from the paper; (3) run
an isolated case designed to measure that choice; (4) justify keep/reject/open
with accuracy, physical admissibility, conservation and cost evidence. A
reference-code match, global energy closure or a preferred interface picture
is not a sufficient adoption criterion. Preserve initialization consistency
and make any necessary changes in rho/u explicit, as in the hydrostatic A/B.
Apply this to existing experimental controls as well as future additions.

The user explicitly deprioritizes CheapRK2/FSAL-like derivative reuse and reuse
of precomputed values that changes kick/drift or stage behavior. Retain those
matrix rows as diagnostic history; they are not candidate model corrections.
Use the existing fresh-stage Butcher RK2 for Warp and the already available
fresh Spheral RK2 control where a reference trajectory is needed. Compare
instantaneous formulas on identical fresh physical states before dynamics;
do not import scheduling differences simply to improve agreement. No integrator
cache/scheduling changes or new physical runs are made in this decision turn.

Next evidence packages, in order:

1. Artificial viscosity. Audit limited pair velocity, approach criterion,
   h/compact-support normalization, epsilon units, coefficients and each side's
   Q in force/work against Frontiere2017 Eqs48–53/69–74 and the pinned source.
   First measure same-state pair Q/acceleration/heating; then isolate Q-length
   conversion and coefficient choices individually on the existing planar Sod
   (exact wave profiles, shock/contact widths, overshoot and entropy production)
   and a smooth-flow control such as linearWave/Gresho (amplitude/phase or vortex
   decay, excess viscosity). Match units/kernel conventions before treating a
   normalization correction as a tunable parameter. Hold density, energy policy,
   support treatment and integrator fixed in each A/B. Include a timestep/grid
   check where needed; avoid selecting a setting from the coarse blob mass alone.
2. Thermal work / compatible energy. Treat raw interim du/dt (Eq78), pressure
   versus viscous work, and final compatible pair-work partition (Eqs65–68) as
   separate choices. Explain which expression supplies stage prediction and
   which supplies final work. Check same-state pair work and per-particle
   increments, including symmetry/sum identities. Use the existing Kidder
   isentropic control for reversible compression/expansion against its analytic
   state (after confirming matching reference setup); then a shock for
   irreversible heating and the saved short blob as a regression/stress case.
   Measure local u/P/entropy, positivity and actual global energy/momentum
   errors, not only total energy. Keep predictor fixed while comparing
   partitions, and partition fixed while comparing predictors. Existing
   node-dudt failures are evidence on the recorded configurations, not a
   universal rejection of every implementation or proof of a repair.
3. Re-run static hydrostatic contact after any proposed adopted change; retain
   established sharp-contact and unequal-spacing limitations. Only combine
   individually justified choices afterward and return to nonlinear blob/RM
   after those controls pass. All trajectories remain sequential, daemon-visible,
   recorded and watched, with failed states retained.

Deliverable per choice: a short equation/source comparison, explicit input and
initialization contract, dedicated A/B measurements with reference and numerical
sensitivity, and an adoption verdict with limits. Defaults remain unchanged
until supported by that evidence.

### Clarify what the existing Spheral fresh variant means (2026-10-20)

Source audit in response to the user's question: `spheral-fresh` selects native
SynchronousRK2Integrator, which evaluates beginning and midpoint derivatives
for each step; the overlay preserves the raw energy predictor and final
compatible partition. It does NOT disable the separate AV gradient lag.
ArtificialViscosity defaults rigorousVelocityGradient=false and copies hydro
DvDx in postStateUpdate; no harness/driver override set that flag in these runs.
Therefore label the existing variant 'fresh integrator RHS', not 'all fields
fresh' or 'cache-free Spheral'. Existing partition comparisons still hold this
AV behavior fixed across their two alternatives.

The pinned implementation exposes `Q.rigorousVelocityGradient=True`; this
recomputes an AV gradient during initialize from the current state. However,
updateVelocityGradient uses a linearly corrected SPH gradient, rather than
the CRK hydro gradient otherwise copied, so that switch changes both timing
and spatial estimator. A native flag comparison must declare that coupling.
If isolating temporal reuse itself, evaluate the same CRK gradient freshly
before Q (explicit additional reference implementation), not a blanket
assumption that switching the integrator already accomplished it. This
read-only source audit launches no trajectory and changes no model setting.

### AV freshness comparison authorized (2026-10-20)

User requests the AV comparison after clarifying that fresh RK2 still retains
lagged AV gradients. Execute a short, isolated three-way Spheral comparison:
lagged CRK gradient (baseline), native rigorous fresh corrected-SPH gradient
(coupled estimator/timing change), and fresh CRK Eq54 gradient (timing-only
relative to the baseline). All use the existing paper-configured128x32 short
blob, identical initialized arrays and replayed525-step time grid, fresh
SynchronousRK2, raw predictor and entropy-compatible final energy. Rebuild
only a hash-named opt-in overlay; installed native binaries untouched.

Before interpreting dynamics, validate the fresh CRK gradient against native
hydro DvDx on the same RHS, and add fixed-geometry affine compression/shear
checks of linear reproduction and native limited pair Q against Eqs48–54/69.
These are static operator checks, not unrecorded physical trajectories.
Compare actual local interface/field behavior, positivity, energy/momentum,
cold-cloud diagnostics and cost; use no default/policy tuning. Add half-grid
checks if temporal sensitivity makes the comparison ambiguous. All physical
runs sequential, daemon-visible, with progress/watchdogs and video.

AV observer correction: the initial same-RHS audit was sampled inside the
energy update; SynchronousRK2 restores state0 before final energy work, so
its midpoint Q-gradient snapshot was from the beginning state while hydro
derivatives were midpoint. These midpoint audit rows are not stage-aligned
and must not be interpreted as a fresh-gradient defect. Beginning-stage
comparison agrees at7e-15. Move observation to native finalizeDerivatives
before state restoration, preserving its boundary operations. The six running
physical trajectories use one immutable existing library and are unaffected
by this read-only observer. Validate both stages in separate8-step daemon
replays with the corrected observer, checking state parity against saved
early frames; retain both overlay versions and qualify the audit provenance.

### AV gradient freshness comparison complete (2026-10-20)

Report: reports/materialEOS/spheral/avGradient/README.md; raw:
results/materialEOS/spheral_av_2026-10-20/. Six full Spheral128x32 paper-overlay
runs to1KH: lagged CRK, fresh CRK Eq54, native rigorous fresh corrected SPH,
on525-step replay and1050-step bisection. All retain fresh RK2/raw predictor/
entropy final work and all other model choices. Startup physical arrays match
bitwise; relaxed H differs only4.44e-16, support2.22e-16 (recorded, not hidden).

Fresh CRK versus lagged coordinate RMS .003735/.004308 on the two grids;
fresh SPH .077810/.075234. Cloud-only RMS .01313/.01575 vs.23627/.22556.
Cold mass lagged .512658/.512658; fresh CRK .512658/.531646; native freshSPH
.525316/.525316. Temporal sensitivity itself is .00788–.01047 coordinate RMS,
so no convergence/accuracy ranking follows from two grids or cold mass.
All states remain positive, relative energy drift<2.3e-15, normalized momentum
error<7.3e-14, material mass exact. Existing matched Warp cloud-coordinate RMS
.24410 vs lagged, .24470 vs fresh CRK: gradient lag alone does not explain the
solver difference. Native rigorous is a coupled spatial/timing variant, not
a cache-only toggle; no evidence justifies adopting it to fit a picture.

Dedicated fixed-geometry affine compression/shear:864 pairs each,79 cross-
material, outside intentional close-pair fold. CRK gradient reproduces linear
field within1.56e-15; native limited Q is zero with exact gradient and<3.2e-15
with computed CRK gradient, vs raw Q maxima2.919/1.293. Native raw Q matches
independently evaluated Eq69 within1.34e-15. Corrected stage observer proves
freshCRK vs native hydro DvDx within7.11e-15 at both RK2 stages. Three8-step
daemon replays match the original trajectories within5.4e-15; original
misaligned midpoint observer rows are retained and excluded.

Integration/diagnostics/publishing costs (exclude startup/post-video): lagged
44.18s/84.15ms, freshCRK49.41s/94.12ms, freshSPH45.42s/86.51ms on525steps;
half-grid88.60s/84.38ms,100.07s/95.30ms,90.97s/86.64ms. Fresh CRK uses an
extra serial pass; overhead~12–13% is specific to this unoptimized reference
overlay, not an intrinsic cost prediction for Warp. All nine completed
daemon histories match HDF5 times and all videos match frame counts. Twelve
focused adapter/viscosity tests pass, plus operator/observer/parity checks.

Disposition: retain current-stage CRK gradient in Warp and use explicitly
fresh-CRK Spheral for subsequent model comparisons. No solver defaults or
installed Spheral sources/binaries changed. AV h normalization, coefficients,
per-side heating, and du/dt/final partition remain separate unqualified choices.
No workers or schedules remain; full512 run stays deferred.

### Limiter and threshold audit authorized (2026-10-20)

User recalls substantial one-sided/averaged interim heating effects in earlier
code and indexing ambiguity shared with CompSPH. Preserve a separate distinction
between an intra-stage predictor du/dt and an energy update used without final
compatible balancing. Later work must trace i/j, gradient orientation, Q-side,
rho/volume/mass factors and pressure/viscous work in both CRKSPH and CompSPH;
validate pair power balance separately from the predictor's stage accuracy.
Do not assume a good predictor is automatically a conservative standalone rate,
or infer that final energy closure validates interim thermodynamics.

Current request: inspect paper/native Spheral/Warp limiter shapes, denominator
guards, close-pair onset/width, h/R conversions, epsilon, approach gates and
coefficient defaults before changing any policy. Use compiled fixed-pair
sweeps to quantify differences. No physical trajectories or defaults need
change for this source/operator audit. Initial finding: Warp and native
Spheral scale etaFold by nominal spacing; the paper states fixed .2 in h units.
They agree in the existing matched nPerh=1 blob, not for arbitrary nPerh.

### Limiter/threshold source and fixed-pair audit complete (2026-10-20)

Report: reports/materialEOS/spheral/limiterThresholds/README.md. Raw:
results/materialEOS/limiter_threshold_audit_2026-10-20/. Actual Warp CPU
limiter/Q kernels compared with an isolated compiled wrapper calling native
Spheral QPiij:3196 fixed pairs,3168 nondegenerate projected-slope pairs,
including unequal supports/densities/c_s, signed slopes and random2D inputs.
Max ordinary phi delta5.0e-16; after matching Q h, max abs(deltaQ)/(1+abs(Q))
4.65e-13. Static checks only, no physical trajectory or daemon run.

Findings: onset agrees (one nominal spacing), and ordinary slope limiter
4r/(1+r)^2 agrees. Warp/current Spheral fold width is0.2 nominal spacings;
literal paper is0.2 spline smoothing lengths. With R=K*h_s and n_h=K*nPerh,
these are0.2/n_h vs0.2/K in r/R, equal only at nPerh=1. Matched W7 blob
already agrees (K4,n_h4), so this is not its discrepancy. At nPerh1.35 and
0.9 nominal spacings in an affine compression, current versus literal-paper
phi .778801/.871818 gives Q_i .219257/.126343 (ratio1.7354), all other
pair inputs fixed. Physical adoption requires a dedicated evolved case at
nPerh !=1, not a change to existing defaults based on this one pair.

Default B8 Q uses R/2.481044 (shape-derived2sigma convention), versus R/4
for spline-h in the matched native W7 case: length ratio1.61222. Epsilon²
.01 agrees dimensionlessly but physical .01*h² differs unless h agrees.
The override changes Q, not reconstruction thresholds. Default WarpCl/Cq1/1
differs from paper2/1; native factory scales Cl=2*Kdb/4 and Cq=(Kdb/4)^2.
Record actual configurations, including NodeList versus kernel extent.

Near-zero slope guards differ: native floors each signed denominator at
1e-30; Warp substitutes1 for a missing ratio. Both-zero phi0 vs1, but
actual Q agrees for the tested zero gradients (no reconstruction correction).
An extreme-unit affine test exposes native guard unit dependence; no observed
force defect in the existing dimensionless blob follows from this. Conditional
source difference with external alpha switches: native pair-averages fCl/fCq,
Warp Frontiere uses side alpha; inactive with no-switch alpha1.

Corrected comments/docstrings about fold units and taper direction (it
suppresses reconstruction and restores AV); no executable numerical code or
default changed. Historical limiter plan has a correction without rewriting
its measured outcomes. Heating audit stays separate for CRKSPH and CompSPH:
interim predictor accuracy, instantaneous pair-power identity, standalone
finite-step energy integration, and final compatible work are separate tests.
Do not assume that differing local work partitions necessarily differ in
global pair power, or that compatible energy closure validates stage u/P.

### Heating and compatible-work audit authorized (2026-10-20)

User: proceed after establishing that ordinary limiter/AV differences largely
reduce to conventions and declared defaults. Trace CRKSPH and CompSPH pair
orientations, pressure/viscous heating, interim energy predictions and final
work partitions independently. First use fixed-state compiled operator checks,
then short daemon-visible controls with fresh RK2 and timestep sensitivity.
Keep numerical defaults unchanged and distinguish instantaneous conservation,
finite-step energy closure, positivity and physical accuracy.

### Heating/compatible-work audit complete (2026-10-20)

Report: reports/materialEOS/spheral/heating/README.md. Raw trajectories:
results/materialEOS/crk_heating_v2_2026-10-20/; fixed-pair/common-state data:
results/materialEOS/heating_operators_2026-10-20/. Native controls use isolated,
hash-named heating/sign-observer DSOs; installed Spheral remains untouched.

Paper Eq.78/native CRK use neighbour P_j+Q_j for du_i. Warp pressure agrees;
default viscous heating averages Q_i/Q_j. Existing oneSidedWork=True selects
the paper policy. Both local shares balance instantaneous force pair power:
delta du_i=B/(2m_i)(Q_i-Q_j)v_ij.G and m_j delta du_j=-m_i delta du_i.
Eighteen compiled 2D pairs (support/density jumps and compression/expansion)
confirm CRK pair-power error<2.12e-15 and local rate changes up to.46257.
Native observer reproduces derivative sums within1.48e-15 scaled; actual
Warp kernels on native beginning/midpoint states and imported RK coefficients
agree within1.64e-8 relativeL2 (native1000-entry table vs direct spline).

CompSPH is distinct: own-side pressure/viscosity in gather heating agrees with
Eq.E.3 indexing; Warp force instead averages the two gradients before applying
both pressure coefficients, while paper Eq.E.2/native SPH retain separate
gradients. Raw pressure-power residual equals
m_i*m_j/2*(A_i-A_j)*v_ij.(g_i-g_j), A=P/(Omega*rho²). It vanishes for equal
supports in the audit but reaches6.19017 in the prescribed unequal-support
states. Final compatible energy residual remains<1.34e-15. This is a concrete
predictor/standalone-rate distinction, not a promoted force correction or an
explanation of the CRK common-state algebra. Misleading model comments about
Eq.E.2/identical gradients corrected; executable formulas unchanged.

Eight full1-KH heating controls (six new, two reused) hold freshCRK gradients,
freshRK2, raw predictor and final-work entropy fixed on525/1050step grids.
Heating contrast cloud-coordinate RMS native.02425/.03841, Warp.04972/.00701;
same-policy cross-solver RMS remains.219–.245. All positive, energy drift
<2.45e-15, normalized momentum<7.01e-14, materialmass exact. Time-grid changes
themselves are comparable; two grids/316 cloudparticles do not rank accuracy.
Other density scheduling and support evolution still differ between solvers.

Final-partition-only controls, with one-sided raw heating fixed: native
sign-filtered node-specific dudt weights lose positivity atKH.079193 on both
grids (80/160steps), Warp atKH.182320/.183079 (155/311steps). Energy still
closes to<2.45e-15. Corresponding entropy controls complete. These are declared
paper-overlay controls, not a claim that stock Spheral fails its bundled test.
Reject adopting node-rate weighting for this setup; retain entropy work here.
Five sampled native stepcounts show entropy-vs-node fractions RMS.33–.46;
RHS-estimated vs endpoint-work sign changes0–1 sampled activepairs/step, with
maximum entropy-fraction difference.00859. Sign choice and fraction policy
remain separate questions; no exhaustive sign ablation was performed.

Fourteen daemon histories/videos verified:12new recordings (including short
observer, later full observer and4positivity stops) plus2reused AV references.
Eight-step observer parity<5.33e-15; full observer parity<4.22e-11. Both failed
Warp processes subsequently exitSIGSEGV after retaining watchdog failure and
closing daemon status; cause unestablished, retained vispy frames encoded
offline. Native policy-label render adapter repaired without rerunning physics.
All finite failed endpoints retained separately. Initial adapter/build failures
are not scientific outcomes.29 focused tests pass (26float64,3float32).

Native one-sided49.41/100.07s (94.12/95.30ms), averaged64.98/130.23s
(123.78/124.03ms): extra serial observer/control pass ~30% implementationcost.
Warp one-sided26.60/44.27s, averaged54.82/44.37s: first averaged run includes
~28s JIT compilation; other mean step times35.25–37.09ms. Full wall scopes,
all stopped-run costs, limitations and provenance are in the report.

Disposition: use one-sided work in paper comparisons, preserve mean work as
declared alternative; no model/default change justified by interface matching.
Keep entropy final work for these controls. Next isolate density scheduling
and support evolution on common states; do not tune more AV/work choices to
pictures. No workers/schedules remain; full512 still deferred.

### Single-material 1D Sod control requested (2026-10-20)

User notes the pronounced Spheral blob nose persists across heating choices
and requests a simpler single 1D Sod comparison. Prioritize this before further
2D nose/support decomposition. Use one physical material and one native
NodeList (not two equal-gamma materials), classical Sod states, common saved
float64 samples, fresh RK2, matched kernel/Q/entropy work and density timing.
Start with equal spacing and fixed supports to remove resolution/support
jumps; then test each code's adaptive-support policy and spatial refinement.
Compare rho/P/v/u with the exact Riemann solution, early RHS/state parity,
conservation/positivity and timings. Physical runs remain sequential and
daemon/video recorded. Agreement should be judged quantitatively, not assumed
from the simpler geometry. No production default changes are authorized by
an agreement metric alone.

### Single-material 1D Sod comparison complete (2026-10-20)

Report: reports/materialEOS/spheral/singleSod1d/README.md; raw:
results/materialEOS/single_sod1d_2026-10-20/. Classical gamma1.4 states
(rho,P)=(1,1)/(.125,.1), equal spacing/variablemass, one materialID and one
nativeNodeList. Commonfloat64input, W7/nh4/Cl2/Cq1/QhR4, freshCRK gradients,
freshRK2/raw neighbour-Q predictor, entropy finalwork, Eq75 perRHS volume and
Eq76 perstep density/continuity stages. No production model/default changed.
Native force/raw heating/RK2/support remain compiled code; separate1D DSO
provides the explicitly matched paper volume/density/energy/Q-gradient choices.

Eight selected runs:600/1200particles (200/400 perunit), fixed/adaptive supports
in both codes,800/1600 matched steps to t.2. Initialphysicalarrays agree except
EOS pressure roundoff2.22e-16. NativefirstRHS snapshots retained; DSOprestep
counts match selected physicalstepcounts. Two initialfine native advance(t)
runs took an extra~8e-15 terminal step, which still resets density/support;
retained as diagnostics, replaced by exact1600step references for comparison.
All10daemon histories and videos verified, including playback descriptors for
the archived extra-step versions. No active workers/schedules remain.

AdaptiveSod agrees closely and improves with refinement. DensityL1:
native.01113294->.00545856, Warp.01116251->.00546723; pressureL1
.01258991->.00580315 vs.01261738->.00581100; velocityL1
.02636900->.01095524 vs.02643285->.01096318. Fine common-grid code differences
rho3.75e-5/P5.16e-5/v1.51e-4; centralUIDpositionRMS4.57e-6. Both retain
localized contact overshoot and smaller fan/shock errors. This establishes
matched scalar1D agreement, not exact nonlinear or multi-material accuracy.

Frozen support coarse runs agree nearly to roundoff (densityRMS7.46e-10,
velocityRMS2.98e-8), but share a strong rarefaction distortion. Errors worsen
with refinement (density~.0321->.036, velocity~.0797->.088), so fixed supports
are unqualified as accurateSod here. Fine frozen-support velocityRMS~.008
shows sensitivity in the distorted fan; no cause assigned to tabulation or
roundoff without a separate test. Conservation/positivity alone would miss it.

All selected runs positive, energyerror<1.78e-15, normalizedmomentum<4.84e-17,
mass exact.13focused tests pass (exact Riemann jump/fan identities, pinned
nativeanalytic oracle comparison and existing comparison/kernel checks).
Native walls coarsefixed/adaptive3.83/4.00s, fine12.91/13.16s; Warp64.59/19.39s
and18.53/22.14s. FirstWarp fixed/adaptive include54.26/8.00s JIT steps; cached
fine loopmeans8.98/11.26ms. Timing scopes/video costs in report.

The blob's larger disagreement remains. This1D control matches the density
schedule that still differs between blob codes, so it cannot isolate dimension
from material coupling or scheduling. Proposed next discriminating control:
equal-mass/variable-spacing Sod, keeping one EOS and matched stage policies;
then add a second material/NodeList before returning to2D. No default changes
or expensive reference runs scheduled.


### Smooth single-material 2D KH requested (2026-10-20)

User accepts the adaptive Sod agreement and requests a slightly more complex
case that both solvers support. Select the bundled Spheral McNally smooth KH
initial conditions, imported from the same saved equal-spacing/variable-mass
arrays in both codes: gamma5/3, rho1/2, vx+/-.5, P2.5, smoothing .025,
vy=.01sin(4pi x), periodic unit square, one EOS and one native NodeList.
This differs from the Warp case's default localized transverse seed; use an
explicit common-input harness, not a claim of unchanged bundled-driver runs.
Keep matched fresh RK2, fresh CRK AV, entropy final work, raw neighbour-Q
predictor, paper volume/density schedule and B8 support convention from Sod.
Start adaptive64^2, then128^2 to t2 with a timestep scaled by resolution.
Quantify native mixing-mode amplitude, transverse kinetic energy, pressure,
entropy, positivity/conservation and periodic UID displacement. No exact
nonlinear KH reference is assumed. Preserve daemon trajectories and vispy
videos, run sequentially and assess cost/progress. No model defaults changed.


Smooth KH pilot: both64^2 runs reach t2, positive and conservative. Mode
amplitudes .043272 native/.053913 Warp (24.6% difference), versus .75% at t.4
and4.17% at t1. Add a64^2 half-timestep pair after refinement to distinguish
time sensitivity from unresolved spatial/support effects. Keep the native
binaries and both model configurations unchanged; conservation is not the
accuracy verdict. Native77.34s/1000steps, Warp31.51s (27.03ms runner loop).


### Smooth single-material 2D KH comparison complete (2026-10-20)

Report: reports/materialEOS/spheral/smoothKH/README.md; raw:
results/materialEOS/smooth_kh_2026-10-20/. Six sequential, daemon-visible
positive trajectories with vispy videos:64^2/128^2 and64^2 halfdt in bothcodes,
one EOS/NodeList, identical smooth McNally physical arrays. Keep declared
fresh/paper controls from Sod; native force/heating/support remain compiled
code with the existing immutable opt-in2D DSO. No defaults/modelchanged.

At128^2, t2 mode native.175648/Warp.178927 (1.87% gap), transverse KE
.0073963/.0077171 (4.34%), pressureRMS.083615/.084109 (.59%). Maximum mode
absolute difference overhistory.00635. Same broad rolls; individualUIDs differ
more (periodicpositionRMS.01524, velocityRMS.06157; maxima.219/.668), so no
pointwise-agreement claim. Fine common-grid L1 rho.01396/P.01085/vx.00662/
vy.00891/u.02568. Native/Warp entropyproxy.003355/.003141 (6.39% difference).

At64^2 final modes.043272/.053913 (24.6% gap). Halfdt yields.044033/.053647
(21.8% gap); within-code changes+1.76%/-.49%. This bounded timestep reduction
does not close coarsegap. Refinement increases modes4.06x/3.32x; bothcodes
remain strongly resolution-sensitive, so128^2 is not qualified asconverged.
Admissibility:energy<2.23e-16,momentum<7.04e-15,masspreserved;startup<4.45e-16.
16focused tests pass. Native DSO counters verify2freshRHS/step and1rawpredictor/
finalwork/densityreset perstep. All6daemon histories and51frame videos match.

Native walls77.34/602.53/155.48s (64/128/halfdt); Warp31.51/131.23/56.80s,
runnermeans27.03/61.25/25.66ms. Native timings include qualification-overlay
serial loops and Python diagnostics/publishing, excludepostrender; Warp includes
plot/storage/encode. No workers/schedules remain. RegularWarp column frames
startafterstepone, havefloat32 exporttimes andomitdynamic P/u/supports;
finalfieldcomparisons use completeNPZ atmatchedt2, curvesperstepdiagnostics.
Offlineanalysis assumptions corrected without rerunning/modifyingphysics.

Verdict:retain KH regression/control, no modelpromotion. Some2D divergence
occurs without a material/EOS discontinuity, but causeandblobnose remainopen.
Proposed next is1DSod split into2materialIDs/nativeNodeLists withsameEOS,
then changegamma with matchedstage/densityschedule. Equalmass/variable-spacing
controlandfull512remain deferred; noexpensive runs scheduled.


### Matched 2D cylindrical Sedov requested (2026-10-20)

After KH user asks for Sedov. Both Warp and pinned native Spheral include2D
cylindrical Sedov; native source also supplies a similarity solution. Use common
float64 full-square periodic arrays, rho1/gamma5/3/Edeposit1, tiny ambientP1e-6,
normalized W7 kernel spike with support2dx (native smoothSpikeScale.5 for
W7/H=1/dx). Keep existing freshCRK/freshRK2, paper volume/density schedule,
raw neighbour-Q predictor and entropy finalwork; no defaults changed. Start
64^2/128^2 to t.05 in[-.5,.5]^2, before boundaries affect blast. Shared
quadratically spaced times t_n=.05(n/N)^2, N300/600, resolve faststartup and
keep physical stage times matched. Add optional bounded timestepcontrol if
needed. Compare radial rho/P/velocity with similarity, frontmarkers and
broadening, tangentialKE/symmetry, positivity/energy/momentum andcost. Explicitly
account for finite initial deposition and tiny ambient pressure when interpreting
point-blast errors. Global angular momentum alone cannot qualify a rotational
operator on this mirror-symmetric test. All physical runs sequential/daemon/
vispy recorded; publish finite source/input/library provenance.


Sedov strict-EOS control found a failure: native64 completes t.05, while Warp
becomes nonfinite at step27/t.000405. Actual stoparrays retained (finite x/rho,
156nonfinite velocityrows,492nonfinite thermal/pressurerows). A subsequent
vispy teardown exitsSIGSEGV after the numerical watchdog/retention, as in prior
failedheating trajectories; distinguish teardown from the initiating failure.
Sourcecheck: nativeGammaLawGas uses sqrt(max(0,gamma*(gamma-1)*u)), permits
negative pressure; explicitmaterial Warp EOS uses sqrt(gamma*(gamma-1)*u),
returnsNaN fornegativeu (legacyuniformEOS usedabs). Investigate firstnegative
midpointu using nonmutatingEOSobserver, strict halfdt, then opt-inresearch
soundzero continuation that changesc only onnegativeu. Keepu/Pandforce/work
unchanged, retain strictdefaults/failures and all daemonrecordings. Complete
native128 reference before Warpdiagnostics. These are explicit diagnostic
variants, not production fixes or an assertion of physical negative-energyEOS.


### Matched cylindrical Sedov diagnostics complete (2026-10-20)

Report: reports/materialEOS/spheral/sedov/README.md; raw:
results/materialEOS/sedov_matched_2026-10-20/. Six completed plus two retained
failed daemon/video trajectories, all8histories/framecounts verified; no workers.
Strict64 original/observer both failstep27 t.000405. Observer preserves preceding
history exactly; midpoint RHS EOS sees8negativeu, minimum-.00700323, and8NaNcs.
This is beforefinalcompatiblework, not an observed energy-balance violation.
Original/observer processes also exitSIGSEGV in vispyteardown afterwatchdog;
actualnonfinite stopstates, invalid EOS inputs and fiveframevideosretained.

Native scalarclosure compiledcheck confirmsc=sqrt(max(0,gamma*(gamma-1)*u)),
negativePallowed; strictmaterialWarpc=sqrt(...) givesNaN. Research-onlywrapper
changesc to0atu<0, preservesu/P. Full64/128WarpzeroSound finishpositive; only
2EOS callspergrid haveu<0, worst-.009912/-.039654. NoEOSmodule/defaultchange.
64halfgrid strictWarp finishes with2401observedEOS calls and0negativeu;
matchednativehalfgridalsofinishes. A physicallyadmissiblestepsize therefore
alreadyworks; nativecontinuationisnotautomaticallyanadoptioncandidate.

ExactpointblastreferenceR(.05)=.25799460, alpha.5642837406 afterindependent
energy normalization ofnativehelper table (rawalpha.5643574122; constructornbins
passedtotolerance; radiusdifferenceonly.0033%). Strongshock identities/physical
energyintegral pass. Finite12particleW7depositE1plusambientP1e-6declared.
Uniform64/128square[-.5,.5]^2; matchedt_n=.05(n/N)^2 N300/600, freshCRK/RK2,
existingpaperpolicies. Nativecounters andperstep timesverifyactualschedule.
Nativefinalduplicateperiodiccallbackrowsidenticalanddeduplicatedonlyoffline.

Native/Warp researchzeroSoundrho2frontmarkers64.259334/.259548,
128.265255/.265305. Finesolvergap.0195%exactR, butbothmarkers2.8%aheadexact;
finitewidthmarkeriscalenotexactfront. Lateexponents.5110/.5106vsexact.5.
Halfstrict64front.259289/.259511. Fine densityL1.10715/.10628, pressure
.19614/.19843, radialv.09541/.09521, uL1fixed .12<r<.45 .18806/.19704.
Coarse rhoL1.10980/.10699,P.25353/.25424,v.14400/.13820; pressure/vimprove,
rhoalmostflat. Bothsharebroadenedshock/radialoscillations/sparsecavity;
no pointwise/generalconvergenceclaim. Finepeaks3.396/3.442vsexact4.
ExactKE.2665454; fine.261513/.261356(-~1.9%, crosssolver.060%).
TangentKE64.459%/.582%,128.247%/.338%; globalsymmetrycannotqualifygeneral
angularmomentumconservation. Energywithin5e-14,momentum3.25e-16,masspreserved,
startupidentical. Full/half64finalprofiledifferencesmodestoncestepsadmissible.

Native64/128/half64wall23.70/178.79/47.06s; Warpzero64/zero128/strictHalf64
17.96/56.39/30.09s, loop45.72/81.00/41.53ms. EOSobserveradds4syncs/step;
qualificationcostsnotoptimizedthroughput. Strictpartialwatchdogwalls4.96/5.00s.
Allphysicalrunsdaemon/vispyrecorded; no installednativecode/modeldefaultsmodified.

19focusedreference/regressionchecks pass. Extendedsuite32pass/6CUDA-skips/
1independentlyreproducibleexistingRMconfigurationSchema np.bool JSONserialization
failure, evenwithoutSedovharnessimport; numericalassertionsnotreached. Tracked
inreportratherthanunrelatedschemafix. StrictnegativeEOScontracttestpasses.

Verdict:Sedovprovidesisolatedrawpredictor-admissibilityevidence. Nextcompare
stage-rejection/retry or separatelyjustifiedcompatiblepredictor; do not silently
adoptnativeEOScontinuation. Materialsplit/variable-spacing/full512remainopen;
noexpensiverunsscheduled.


### Standalone thermal-energy baseline first (2026-10-20)

User decision: establish simulation with a suitable standalone du/dt update
without compatible correction, then return to integration of compatible terms.
First test the published Eq.78 and native CRKSPH IncrementState route with
compatibleEnergyEvolution=False and evolveTotalEnergy=False. Hold the shared
Sedov initial state, force, AV, density/volume schedule and fresh midpoint RK2
fixed. Measure semidiscrete power balance, finite-step energy error, positivity,
similarity errors and timestep sensitivity. Do not require roundoff energy
conservation of plain RK2. Any alternative thermal partition needs its own
derivation and isolated evidence; no model defaults changed by this study.

### Standalone thermal baseline results (2026-10-20)

Report: `reports/materialEOS/spheral/sedovThermal/README.md`.
Raw: `results/materialEOS/sedov_thermal_2026-10-20/`.

With compatible=False and total-energy evolution off, published/native Eq78
loses positivity in both codes:300/600/1200 planned steps stop at13/25/50,
t=9.38889e-5/8.69444e-5/8.69444e-5. Native min u=-.09856/-.01101/-.01103.
The initial energy of those outer spike particles was+.517929. Energy error
reduces~4x per dt halving while the positivity failure remains. A production
pressure-only expanding-pair test also verifies du_i<0 at u_i=P_i=0 with P_j>0;
total pair power still balances. Thus semi-discrete conservation does not imply
positive thermal evolution, independent of finite-step compatible correction.

Derived research alternative: dt->0 limit of Eq61/63 entropy pair work.
q_ij=-(vi-vj) dot Fij; du_i+=f_i*q/mi. Same-gamma ideal gas: heating
f_i=sj/(si+sj), cooling f_i=si/(si+sj), both-zero fallback.5. No timestep in
RHS, no dt*a velocity correction, no final balance/overwrite. Exact nonzero
entropy ratio in Torch and isolated native overlay;48 compiled fraction tests
agree exactly. This is neither Eq78 nor unchanged native Spheral behaviour.

Both complete t.05 on all3 time grids with strict Warp EOS, no negative calls,
min u=1.5e-6. Native final energy errors=-2.51709e-4/-6.46172e-5/-1.54012e-5;
Warp=-2.63841e-4/-6.98159e-5/-1.78923e-5. Reduction factors3.895/4.196 and
3.779/3.902 establish approximately second-order temporal energy behaviour.
Native compatible energy/predictor counters=0; RHS=2N/density=N; Warp guards
record0 compatible calls. Actual RHS power closes near roundoff. At finest
level density/P/radial-v L1 native=.110678/.255365/.143535 versus
Warp=.108143/.256158/.138532. Both share coarse broadened shock/oscillations;
no spatial convergence or interface-fix claim. Momentum/mass roundoff; symmetric
blast cannot qualify general angular momentum conservation.

Native wall30.25/60.38/121.61s (~101ms/step,4threads); Warp18.56/30.85/55.59s
(loop46.46/42.01/40.14ms). Qualification observers add syncs; native research
rate reevaluates pair force serially.24 focused tests pass; independent ODE
solution/energy convergence, cold boundary, production counterexample,
configuration, old energy/heating and similarity controls.12 daemon histories
and videos verified (6complete51frames,6failed2-3frames), all workers exited.

Runner now exposes existing CRKSPH compatibleEnergy=False. Production Eq78 and
all defaults unchanged; entropy RHS remains research harness/isolated DSO.
Verdict: raw Eq78 is not a robust standalone cold-blast rate; retain it as the
published/native control. The derived rate is a working standalone baseline,
requiring adiabatic/entropy checks, spatial refinement, other shocks and
material contacts before promotion. Establish those before adding compatible
terms to this baseline. No expensive/full512 run scheduled.

### Warp-only compatible/noncompatible case suite (2026-10-20)

User requests Sod, Kelvin–Helmholtz and other representative WarpSPH cases.
Use a four-way matrix: repository compatible update with existing raw RHS;
only disable compatible correction; experimental continuous entropy thermal
RHS without correction; same experimental RHS with the repository compatible
correction. This separates changing the thermal derivative from applying the
finite-step correction. Hold fresh midpoint, CRK summation/legacy density,
current gradient and default AV/limiter fixed; no matched-Spheral stage controls.
Use common saved one-gas inputs and explicit strict material EOS. Include
Sod1D, smooth KH2D, acoustic1D, smooth isentropic vortex2D and cold Sedov2D.
Accuracy/reference diagnostics, energy/momentum, particle entropy, positivity,
KE and timing are measured; use half timesteps for sensitivity. All physical
runs remain sequential, daemon-visible and recorded with vispy video. No
production default/formula promotion from this suite alone.

### Warp-only energy-policy suite complete (2026-10-20)

Report `reports/materialEOS/crkEnergySuite/README.md`, quantitative tables and
comparison.json; raw `results/materialEOS/crk_energy_suite_2026-10-20/`.
Four policies x five cases x two dt grids =40 physical controls. Repository
CRK summation/legacy density, current AV gradient, Cl=Cq=1, averaged Qi/Qj,
native partition/raw predictor retained; B8/n_h4/fresh RK2/strict one-gas EOS
are common declared runtime controls. No Spheral stage overlays or caching.
Four policies are base-compatible, base-raw without correction, experimental
instantaneous entropy RHS without correction, same RHS with native correction.

38 complete positive; two raw Sedov failures at t8.694444e-5,25/50steps,
min EOS u~-.018406/-.018405. Other Sedov policies complete600/1200steps;
standalone entropy E drift -5.49310e-5 -> -1.42021e-5 (3.868x), compatible
within7e-15. Profile errors shared, fine density/P/v ~.1116/.2438/.144.

Half-dt Sod vL1 compatible/raw/entropy/entropyCompatible:
.027273/.028607/.027473/.027452. Raw modestly worse exact profiles; entropy
noncompatible/compatible close. KH modes .062804/.074948/.062756/.062786:
raw19.3% above base-compatible (coarse22.4%), despite tiny energy drift;
raw particle entropy L1 lower(.01176 vs.03854), pressure RMS/TKE higher.
No exact nonlinear64² KH reference to rank growth accuracy. Wave all retain
99.9692% amplitude, vL1/A~.00416; shared entropy L1~1.01e-7. Vortex all core
vL1~.0222, rotational KE loss3.23%, entropy L1~3.5e-4; no material gain from
energy policy. Refinement changes these shared smooth errors little.

Actual stage power balanced near roundoff and noncompatible correction counters
zero. Compatible update changes local work sharing as well as total-energy
error; base raw is not necessarily the same local ODE in the dt->0 limit.
Entropy-rate noncompatible and compatible use closer continuous sharing.
No universal accuracy/adiabatic/torque claim or model promotion.

33 focused tests pass; compilation/whitespace checks pass.40 daemon histories,
status labels, trajectory times and videos verified.32 complete recordings41
frames; six complete Sedov recordings44/43frames; two failed recordings2frames.
Float truncation in runner int(exportInterval/dt) caused Sedov HDF5 strides
14/29 versus plotting15/30: six videos reconstructed from saved trajectory,
original videos retained, no physics rerun. Repro driver automatically repairs.
Diagnostic/NPZ times f64, HDF5 frame labels f32 explicitly checked.

Physical harness walls sum906.19s; half-KH compatible/raw/entropy/entropyComp
52.17/50.52/53.97/55.18s,26.09/25.26/26.98/27.59ms per step; every run's costs
in tables.md. Eager GPU, observers and videos included; offline repair excluded.
Verdict: retain compatible baseline; raw standalone not robust for cold blast;
continuous entropy remains a viable experimental standalone option with no
universal improvement demonstrated. Sharp contacts, variable spacing and
spatial convergence open. All suite workers exited; no full512 run scheduled.

### Blob test of the continuous thermal rate (2026-10-20)

User asks how the experimental rate performs on the blob. Reuse the exact
qualified n128 short input,4380 particles,1KH, and original short Warp matched
configuration (massReplacement density every RHS, geometric volume, Cl2/Cq1,
R/4 Q length, averaged-Q work, current CRK gradient, raw predictor, native
compatible fraction). Only vary continuous thermal rate and final compatible
correction, four combinations, on the recorded525-step grid and its bisected
1050-step version. Do not confuse original native compatible fraction with the
later final-work-sign entropy control. Keep the stored fresh-CRK averaged-Q
Spheral reference as a declared reference, not a pointwise exact solution.
Measure material/cloud metrics, interface geometry, local entropy/pressure,
energy, positivity and cost; verify daemon/video. No full-resolution run,
new reference build, default adoption or parameter tuning is authorized/needed.


### Blob thermal-rate comparison complete (2026-10-20)

Report `reports/materialEOS/blobThermal/README.md`; raw
`results/materialEOS/blob_thermal_2026-10-20/`. Eight physical controls on exact
stored n128/4380-particle input, 525/1050 steps to 1 KH. Baseline coarse final
fields replay bitwise. Six corrected/experimental runs complete positive;
base raw fails at .02517/.02461 KH, attempted EOS u=-4.79e-4/-1.96e-4.

Fine cold mass fractions: compatible .512658, experimental .512658,
experimental+compatible .506329, stored fresh-CRK Spheral .518987. Native x95
9.49989 vs all Warp ~9.2677; y-span90 2.11812 vs 2.01459–2.02902. Interface
gap remains; new RHS is not a surface-tension/interface remedy. Local cloud
pressure differs by ~3.94% L1 from baseline with new RHS+compatible, although
bulk geometry is close. Compatible atop new RHS differs from standalone by
~.91% pressure and .40% u L1; energy returns to roundoff (<2.67e-15).
Experimental standalone max energy drift 3.00e-7 -> 7.97e-8 on half-dt.
Coarse survival changes do not persist with same sign on fine dt; no universal
accuracy or disruption improvement. Density/support/final-sign differences
remain explicit in the native comparison; short coarse spatial sampling.

Nine focused tests pass; eight daemon/HDF5/video checks pass, including two
partial failures. Successful videos 54 frames, failed videos four. Pair power
residual <1.65e-14; no correction calls without compatible energy. Fine walls
43.88/44.90/45.86s, 41.79/42.76/43.67ms per step (baseline/new/new+compatible),
including observers/video; ~2.3%/~4.5% overhead. Coarse 25.53/26.18/26.46s.
Stored native reference only, no new Spheral or full512 run. All workers exited.
Verdict: retain baseline default, experimental thermal option remains promising
for standalone robustness, interface issue remains independently unresolved.


### Density reset / continuity-stage qualification (2026-10-20, in progress)

User prioritizes density evolution. Isolate every-RHS Eq76 replacement versus
full-step reset followed by Eq79 continuity at RK stages using the existing
opt-in paper path. Geometric Eq75 volume remains recomputed every RHS, with
Owen support, fresh midpoint, AV and native compatible energy held fixed.
Audit actual stage densities, reset sizes, EOS consistency and counter paths;
check smooth acoustic wave, classical Sod and exact short blob on two temporal
grids. Contrast stored compiled native paper/fresh controls where available.
Separate reset-at-beginning accepted density from recomputed endpoint density.
No coupled m/rho-volume mode, cached derivatives or default promotion.


### Density reset / continuity-stage comparison complete (2026-10-20)

Report `reports/materialEOS/densityEvolution/README.md`; raw
`results/materialEOS/density_evolution_2026-10-20/`. Twelve wave/Sod/blob
physical controls, every-RHS summation vs paper reset/continuity, two dt grids.
Same Eq75 volume rule, Owen H, AV, original thermal RHS/native compatible
energy and fresh midpoint. Existing paper implementation retained; graph
resolver/direct wrapper now both guard its eager pre-step reset. Midpoint
state, reset EOS/unchanged-u and graph guards independently tested.

All12 complete positive, exact IDs/material mass, max E deviation2.45e-15;
unchanged coarse blob bitwise replay. Actual reset count N and density
updates2N with no alias/update residual. Native beginning/midpoint recorded
CRK gradient replay gives Eq79 relativeL2 7.97e-9/1.09e-8; same-state Eq76
versus evolved rho differs2.86e-10 at beginning,4.93e-4 at midpoint. Target
is trace of actual native fresh CRK gradient; native DrhoDt not exported.

Fine wave actual entropyL1 1.01e-7 vs2.05e-8 becomes2.055e-8 vs2.054e-8
under common offline endpoint Eq76 evaluation. Apparent improvement is mostly
output time centering, not dynamics. Same amplitude ~99.961%, phase~.0049.
Fine Sod densityL1 .0112497/.0111626, pressure .0127378/.0126204, velocity
.0267214/.0264368. Changes modest and dt-sensitive; no universal ranking.

Fine blob coldmass .512658/.525316 vsnative .518987; x95 9.26765/9.26730
vs9.49989, yspan90 2.01459/2.04697 vs2.11812. Local cloud P/u changes
6.26%/2.46% L1, position RMS .0812; native gap remains, cloud displacement
error vsnative .3104->.3342. Coarse survival policy contrast opposite sign;
threshold metric only316cloud particles. Offline endpoint rho leaves blob
coldmass unchanged. No interface remedy or accuracy adoption demonstrated.

Density resets at fixed u change P and entropy. Blob cumulative signed
log-entropy reset jumps -.002890/-.002848; absolute .10160/.10203. Only reset
budget, not net entropy production. Initial input density reset differs .06071
L1; first RHS also exposes pre-reset H vs following Owen support solve, giving
.006011 L1/~11% pointwise initial stage discrepancy on both dt grids. Later
sampled discrepancy .0006003->.0003021. Preserve startup/support ordering as
an explicit confound before inferring a dt limit or changing model defaults.

31 focused CPU tests pass; compile/whitespace clean;12 daemon history/HDF5/video
checks match (four blob42frames, eight wave/Sod41frames). One0step EOS diagnostic
signature error retained under setupAttempts, not a numerical failure. Raw
provenance distinguishes prepare snapshot, corrected physical execution source
and later analysis/provenance improvements. No full512 or new native evolution.

Coarse sum/paper walls: wave8.89/9.43s (22.22/23.57ms/step), Sod9.09/9.70s
(22.72/24.26), blob25.24/29.63s (48.08/56.43). Fine wave12.12/13.56s
(15.15/16.94), Sod12.51/14.02s (15.64/17.53), blob43.61/51.87s
(41.54/49.40). Observer/shadow/reset host syncs included, not intrinsic model
throughput;12–19% measured fine-grid overhead. Static endpoint/native work
additional. Workers exited. Retain default; paper is qualified opt-in for
periodic fluid midpoint only, not pure continuity or entropy-preserving reset.
Hydrostatic boundaries, variable-spacing, general Butcher and H/reset ordering
remain separate qualification before broader adoption.


### Owen / native IdealH support comparison (2026-10-20, in progress)

User identifies smoothing-length model as next priority and notes the Owen
lineage of Spheral. Trace native SPHSmoothingScale, TableKernel lookup and
ReplaceBoundedState scheduling against Warp psi_H, lookup calibration,
cross-material weighting and iteration. Fixed-state compiled native operator
checks first; then controlled daemon/video blob trajectories with density,
energy, fresh midpoint and remaining numerical choices fixed. No FSAL/cache
change, anisotropic ASPH experiment or default promotion.


### Owen / IdealH comparison complete: support gap isolated (2026-10-20)

Report `reports/materialEOS/idealH/README.md`, raw
`results/materialEOS/idealH_2026-10-20/`. The two are the same Owen-style
strategy and relaxation formula, not unrelated algorithms. Actual differences:
Warp lattice vs native shell mapping; unweighted vs cross-NodeList relative
m/rho weighting; old bounded lookup range vs native Hermite table; case absolute
hmax=.5 (Rmax2) vs Warp relative bounds; before-RHS update vs following state
replacement. Runtime Warp uses one relaxation per RHS, not full convergence.
No derivative reuse, CheapRK2 or position/velocity kick/drift change.

C++ wrapper calls installed SPHSmoothingScale::evaluateDerivatives directly.
Seven identical static 2D states (lattices, volume jump, actual blob fields)
agree with research GPU exact exported native interpolation tables to1.8e-15
in target R. Equal-volume labels irrelevant at roundoff; manufactured unequal
volume target difference up to10.3%, actual blob startup34%. Copied native
initial supports are weighted IdealH fixed point (nPerh .999981–1.000014).
Unweighted initial effective compact n_h reaches8.19;20particles exceed old
LUT upper6, target saturation error8.16%. On later Warp endpoint none exceed
lookup range but22 unbounded native targets exceed R2; bound distinct from
lookup. Shell/lattice remains a deliberate calibration choice, not a universal
bug verdict. Case hmax is not intrinsic to IdealH or imposed by the paper.

Twelve sequential physical controls (6policies x525/1050 grids) to1KH onexact
4380particle input. Paper reset/continuity density, geometric volumes, current
CRK AV/Cl2Cq1/R4 length/averaged work, raw thermal RHS/native compatible energy
and fresh midpoint fixed. Chain: baseline, cap, shell, native lookup, material
weight, replacement timing. Not a full factorial. Both baseline reruns bitwise
match preceding density study. Native physical reference reused, not rerun.

Half-grid cloud RMS displacement vs native: .334241 baseline; .344836 bound;
.308689 shell; .309076 native lookup; .0419997 weighted before RHS;
8.68905e-5 same weighted target at following state updates. Coarse last
contrast .333585 -> .000242254. Final replacement fine cloud density/P/u/v
relativeL1 vsnative7.85e-5/1.24474e-4/7.95e-5/8.19e-5. Fine coldmass exactly
.518987, x95 9.499887vs9.499892, yspan2.118303vs2.118123, meanu .40713449
vs.40713845, nearest-ambient fraction exactly .0775316. The support policy
accounts for almost all the earlier surface-tension-like gap on this short
matched case; remaining kernel/sign/reduction differences not individually
attributed. This is reference agreement, not proof of nonlinear accuracy or
resolution convergence and not removal of all CRK interface limitations.

All12 positive, maxE drift2.67e-15, exact material mass/IDs. Runtime fresh RHS2N,
resetN, continuity2N; replacement controls target updates2N/final overridesN.
Independent manufactured test exercises the real Butcher midpoint and survives
finalizer H copying, with no derivative cache.33 focused tests pass, compile
and whitespace clean.12daemon/HDF5/video checks pass, all42frames. No setup or
numerical failures. Native audit DSO/build under spheral-support-audit; source
and exact table hashes retained. Unused audit error placeholder removed from
reproduction code after runs; it is explicitly not used as validation evidence.

Coarse baseline/cap/shell/nativeLookup/weighted/replacement wall s:
29.80/32.10/34.87/29.45/30.23/29.89; ms/step56.77/61.14/66.41/56.09/57.58/56.92.
Fine52.28/57.71/61.21/51.45/53.19/52.68s;
49.79/54.96/58.30/49.00/50.66/50.17ms/step. Observers, resets, daemon and video
included; no tuned throughput claim. Static native/range audits extra.
All workers exited; no full512 run started.

Disposition: keep production defaults unchanged. Native weighted IdealH plus
explicit support-state replacement is now a justified opt-in candidate.
Integrate/qualify auxiliary H semantics, hydrostatic and unequal-spacing
contacts,2D Sod/KH and smooth-flow/conservation before broader adoption. Keep
absolute bounds and lattice/shell calibration independently selectable.

Figure clarification (2026-10-20): replaced internal dt2/run IDs with descriptive
solver/support-policy titles in the main IdealH comparison and policy gallery.
All main panels use the shared 1050-step grid; density/AV/energy choices within
Warp are fixed. This is plot relabeling only; no physical trajectories rerun.

User follow-up (2026-10-20): requested Warp's original before-RHS timing with
the other support changes applied. This is exactly the completed weighted
control at both dt grids. Reused its verified daemon/video trajectories rather
than duplicate identical physics. New timingComparison.png places it beside
native and same-target replacement. Fine native cloud contrast: position
RMS .0419997, P4.4854%, v2.8945%, coldmass .531646, versus replacement
.00008689/.01245%/.00819%/.518987. Coarse positionRMS .090111; both positive
and roundoff energy, measured costs30.23/53.19s (57.58/50.66ms).

### Preferred support timing and explicit state (user, 2026-10-20)

Select the `weighted` before-RHS variant as the preferred research baseline:
native lookup, separately configurable case bounds and cross-material volume
weights, with fresh support adjustment before each RHS. The user accepts the
single relaxation's dependence on incoming support. Keep support in the
explicit state; no geometry-only converged support reconstruction is required.
The desired ML contract is reproducible fresh evaluation from that state,
without omitted stage history or derivative caches. Including support makes
its dependence explicit rather than introducing hidden memory into a reduced
state. This preference supersedes replacement as the adoption candidate above;
replacement remains the useful native reference-match control.

Before production adoption, qualify saved-state continuation and the precise
meaning of reported endpoint support (currently retained from the midpoint
RHS), then the existing hydrostatic/Sod/KH regressions. Do not introduce extra
endpoint relaxation or a fixed-point solve solely to remove the accepted
incoming-support dependence. This records a policy decision; no numerical
defaults or trajectories changed.

User steering (2026-10-20): include the simpler material-aware original Owen
solver, preserving all other support choices and 1D/2D/3D capability. Implement
an explicit opt-in cross-material moment weight only; test dimension-generic
operators, reuse the exact blob input/time grids, and add this third policy to
the hydro/Sod/KH qualification. Keep all numerical defaults unchanged.

### Weighted before-RHS qualification authorized (2026-10-20)

Run a fresh-process continuation from a saved accepted state, keeping incoming
support and physical fields while discarding neighbor/search caches. Compare
the resumed suffix and final fields against the uninterrupted run. Then
compare original lattice support against weighted/native-lookup before-RHS
support on a 2D square pressure contact, a 2D Sod strip and smooth McNally KH,
at two timestep sizes. The audited research operator is 2D-specific, so the
Sod control uses a periodic transverse strip rather than silently transplanting
a 2D lookup into 1D. Keep fresh midpoint, paper density, geometric volumes,
original compatible energy and AV settings fixed within each pair. Record
accuracy, positivity, entropy, momentum, costs and state/export semantics;
retain failures as evidence. Physical runs remain sequential, daemon-visible
and recorded. No production default promotion is authorized by agreement alone.

### Before-RHS restart and material-only qualification complete (2026-10-20)

Report: `reports/materialEOS/supportQualification/README.md`; raw data:
`results/materialEOS/support_qualification_2026-10-20/`. All 21 physical runs
completed and have matching daemon histories/HDF5/video counts. Eighteen are
original/material-only/native-lookup weighted contact/Sod/KH controls on two
dt grids; two are exact-input material-only blob controls; one is the separate
process KH restart suffix. Numerical sources stayed unchanged throughout.
Total reported runner wall time 776.87 s, with per-row costs in the report.

Added opt-in `owenMaterialWeights=False` with config persistence/validation.
Only the psi moment changes: same-material weight 1, cross-material weight
(m_j/rho_j)/(m_i/rho_i), using physical material IDs, never emitter IDs.
Existing lattice lookup/range, relative bounds, relaxation and timing remain.
The generic operator covers 1D/2D/3D list/grid and scalar/tiled reductions.
Ten new operator/config checks plus the prior 33 focused checks pass (43).

The weighted/native-lookup KH checkpoint after step500 (t=1) reloads every
tensor exactly. Discarding topology/grids/pair work/derivatives and repeating
500 steps in a new process gives maximum final field difference 6.52e-12,
position7.81e-13, support2.32e-13. This qualifies accepted full-step restart
for fresh midpoint/no switch/paper density; it does not claim bitwise identity
or partial-RK restart. Endpoint support remains the last midpoint-RHS support.

Material-only blob cloud position RMS vs native is .23936/.25317 coarse/fine,
compared with original .33358/.33424 and full weighted before-RHS .09011/.042.
Weights alone retain only part of the improvement. Supports reach R2.947/2.929
vs the native cap2; bounds/calibration are still jointly different in this
weighted comparison. Both runs positive, exact mass/IDs, energy drift<2.45e-15;
costs35.01/54.65s (66.69/52.05ms per step, coarse includes first compilation).

Across all21, max energy drift2.89e-15; no watchdog failures. Contact maxspeed
7.46e-15, displacement<1.3e-13 spacings, uniform initial-density-reset pressure
offset3.21e-6 original/material-only vs4.81e-6 native lookup. Material-only
single-material Sod agrees with original at ~1e-15, KH positionRMS<1.9e-12.
Sod retains common overshoots/transient negative mean entropy, documented as
baseline limitations. Fine density/P/v L1 .01859/.02130/.04434 original vs
.01876/.02141/.04490 native lookup; fine KH mode .0536477 vs .0440141,
with the policy difference retained under timestep bisection.

Disposition: keep the material-only option as a dimension-generic control,
without claiming equivalence to the user's preferred full weighted before-RHS
variant. Production defaults unchanged. Next: split bounds/calibration under
weighting, then resolution/unequal-spacing qualification and any desired
full1D/3D physical tests. One extra restart checkpoint-preview frame was
excluded in a re-encode of existing vispy images to match recorded history;
original movie/images retained and no physics rerun for that repair.

### Locally generated Hermite support tables (2026-10-20, complete)

Generate coefficients from Warp's own kernel functions for each dimension.
Source contract: pinned TableKernel shell sums, quadratic gradient sampling,
Han--Guo tridiagonal derivative fit, inverse construction before forward
monotonic limiting, and Fritsch--Carlson slope limiting. Express the runtime
table/moment in compact-radius units, leaving case bounds and before-RHS
timing separate. Verify against retained native 2D B8 tables/operators, then
check common hydrodynamic kernels in 1D/2D/3D and short daemon-visible blob
controls at both saved timestep grids. Keep defaults and cached-stage policies
unchanged; explicitly reject unsuitable singular/non-invertible kernel tables.

Completed report: `reports/materialEOS/generatedOwen/README.md`; raw controls:
`results/materialEOS/generated_owen_2026-10-20/`. Opt-in `owenTable: hermiteShell`
generates from the actual core kernel/normalization in compact-radius units.
Material weighting remains independent; existing bounds, relaxation and
before-RHS timing remain. No exported native tables or Spheral runtime is
needed. Production lattice default unchanged.

The native-compatible 1000-point uniform inverse reproduces retained native
B8 operators within 7.0e-8 relative across seven common states. The general
4096-point logarithmic inverse deliberately limits the forward curve before
inversion; it differs by at most 9.76e-5 on those states and resolves narrow
Gaussian tails missed by the uniform inverse. All 12 hydro kernels generate
in each of 1D/2D/3D, both precisions; worst inverse roundtrip 1.472e-4 on
n_R=[3,12]. Singular surface/viscosity kernels and non-invertible data reject
explicitly. Tests: 89 float64, 55 float32, one expected native-float64 skip.

Four exact-input blob controls, native-compatible/refined at 525/1050 steps,
completed sequentially with verified daemon/HDF5/videos (42 frames each).
Fine refined cloud position RMS vs imported-table Warp 1.846e-5, vs Spheral
.0420045 (imported .0419997); cold mass .531646. Positive, exact mass/IDs,
energy drift <2.45e-15; fresh RHS and density-reset counts correct. Fine
runner cost 61.59s/58.65ms per step versus imported 53.19s/50.66ms, about
16% overhead, including diagnostics/storage/publishing/video. No hidden
physics history introduced.

A float32 Gaussian zero-lower-endpoint fix followed the physical runs; all
executed float64 B8 coefficient arrays and all 36 float64 roundtrip metrics
are bitwise unchanged. Preserve executed source snapshots and the explicit
postfix proof in `validation.json`/`zeroEndpointFixVerification.json`; do not
claim every current source hash was used for those runs. Full 1D/3D flows,
other kernels/float32 flows, bounds/calibration separation and performance
optimization remain separate work.

### Generated Owen full-flow dimensional qualification (2026-10-21, spot checks complete)

User requests a snapshot commit after adding generated reports to .gitignore,
then 1D/3D physical checks of hermiteShell. Use the same fresh midpoint,
material-aware support and compatible-energy policies as the preferred path;
run sequentially with the daemon and vispy recordings. Prefer a planar Sod
problem in both dimensions with an analytic reference, plus a pressure contact
if needed to isolate cross-material weighting. Record dimensional symmetry,
positivity, conservation, errors and wall/ms-per-step; retain failures.

Snapshot commit: `db84b94` on dev (213 files); generated reports ignored,
existing tracked report documents retained. The initial qualification attempt
failed before sampling because the runner schema omitted owenTable and
owenMaterialWeights. Added both public descriptors for compressible SPH and
12 save/load/prepare/schema/invalid-input regression checks; all pass. Preserve
that startup log under results/materialEOS/owen_dimensions_2026-10-21_startup_schema_failure.
The physical cases use the public options directly, not a support replacement
overlay. Numerical sources are hashed/snapshotted before the sequential queue.

Four sequential controls complete at t=.2: generated hermiteShell / lattice,
both material-aware, in 1D (300 particles, dx=.01) and genuine 3D
(120x16x16=30,720 particles, dx=.025). Two material labels share gamma1.4
to retain the classical Sod analytic reference. Periodic reservoirs on
x=[-1.5,1.5], compare |x|<.5; 3D transverse periods .4 exceed twice max
support .15116. No added support clamp, no forces constrained to be planar.
Fresh midpoint, paper density, geometric volumes, original compatible energy
and AV held fixed; 200steps at dt=.001, 400 fresh RHSs and 200 resets each.
Report: reports/materialEOS/owenDimensions/README.md.

Generated density/P/v/u L1 errors: 1D .019726/.021501/.037282/.042745;
3D .041100/.050345/.083526/.105904. 1D generated/lattice density/P relative
L1 differences ~1e-8; 3D .000701/.000886, positionRMS .0001605, support
relativeL1 .01179. Different resolutions across dimensions; not a convergence
comparison. Positive accepted states, no negative EOS calls, exact masses/
material IDs/UIDs. Max energy drift across all4 <8.55e-14; generated3D max
transverse speed7.05e-15, density-plane spread2.33e-14, momentum error per
mass<7.2e-15. All4 daemon histories/HDF5/videos verified, 41frames each.

Retain common numerical limitations: contact overshoots visible in both
lookups; generated3D global mean delta log(P/rho^gamma) ends -.009898 vs
lattice -.009997, minima -.010842/-.010900. Significant particle entropy
loss (threshold-.01) has the same .12222 mass fraction for both. 1D also
has a transient negative mean, ending +.0000465. These periodic-box measures
include the reverse Riemann problem at the outer seam. Do not infer entropy
monotonicity from compatible total energy conservation or attribute this
shared defect to local table generation. Density/energy isolation remains
a future model task, alongside spatial/dt/unequal-spacing qualification.

Costs (wall seconds / wall ms per step): generated1D16.17/80.84, lattice1D
7.86/39.31, generated3D193.16/965.79, lattice3D124.67/623.37. First generated
rows bear first-load/JIT costs. Sampled warm median after step10: 1D11.05/
12.60ms, 3D647.40/595.75ms (generated/lattice), about9% 3D overhead including
changed support/neighbors, diagnostics, shadow evaluations and recording.
No hidden-history optimization introduced. Focused Owen/material/runner
regressions:109passed, two legacy two-step physical parity tests deselected.
All snapshotted execution source hashes unchanged. Defaults unchanged;
accept dimensional operation as a spot check, keep full qualification open.
