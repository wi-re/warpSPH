# Per material equations of state and Richtmyer Meshkov instability

Progress and priorities: [PLANS.md](PLANS.md).

Status: scoped, 2026-10-19. Implementation and numerical runs are pending.

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

The checkout already has ongoing runner/configuration changes. Implement
against that interface, preserving unrelated work. The plan index also
records a related triple-point requirement: three thermodynamic regions
can represent only two materials. RM has the same distinction: shocked and
unshocked regions of the incident gas share one material.

## Material configuration and Warp representation

Use a serializable `MaterialConfig` with a descriptive name and an EOS
description. The initial supported description is `IdealGas(gamma)` with
finite `gamma > 1`. List position is the dense, zero-based material ID.
Explicit tables require fluid IDs in `[0, len(materials))`; unused entries
are allowed. Two disjoint regions can select the same entry.

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

## Implementation sequence and gates

Estimates are focused engineering days, including local validation, and are
conditional on the contact gate. They are not run completion forecasts.

| Phase | Work and likely files | Exit gate | Estimate |
|---|---|---|---|
| 1 Material contract and closure | New material config/device types and EOS kernel under `configurations/moduleConfigurations/` and `modules/eos/`; exports; common compressible config dict converters. Retain the old scalar helper. | Mixed/permuted labels match analytic conversions and the scalar reference; invalid IDs/states fail clearly; f32/f64 and gradients work; runtime table survives repeated calls without uploads. | 1–2 days |
| 2 Complete the thermodynamic path | Three SPH RHS functions, timestep, common initialization helper, compatible-energy operator/kernel, diagnostics, setup guards. Recompute derived outputs from finalized `rho,u` when required, including loaded/output states. | Single-gas regression, explicit identical materials vs uniform EOS, unequal-gamma pair-work conservation/positivity, and eager/graph parity. Table buffers remain valid across RK stages and step reruns. | 2–3 days |
| 3 Configuration and saved runs | `runner/configuration.py` and parameter descriptors; config validation/overrides; `io/{export,importIO}.py`; runtime owner rebuild on restore. Python and YAML use the same list schema. | Config and trajectory round trips reconstruct material-specific pressure/sound speed; old files still load; unsupported solver/options are rejected. Resolve the currently in-progress runner API before editing it. | 0.5–1 day |
| 4 Contact and planar shock gate | New focused EOS/material physics tests and a probe; both particle samplings; stationary/advected contact; single-material incident shock; zero-amplitude two-gas interaction with a two-gamma reference. | Pressure equilibrium, unchanged labels/material mass, shock jump/speed and transmitted/reflected waves agree within fixed tolerances and improve with resolution. No unexplained positivity failures. | 1–2 days if the existing contact treatment passes |
| 5 Registered RM case | New `caseUtils/compressible/richtmyerMeshkov/` sampler, `cases/compressible/richtmyerMeshkov.py`, case registration, plotting, parameter metadata, example wrapper/config. | Smoke run finishes; flat-interface control stays flat; perturbed interface shows shock-induced transverse motion and growth; no boundary contamination during the measured interval. | 1–2 days |
| 6 Qualification and example | Sequential resolution runs, independent reference, linear growth fit, material/density/pressure video and saved diagnostics, performance and documentation. | Quantitative criteria below pass; a complete reproducible demonstration runs to its chosen time with resolved configuration and evidence. | 1–2 days plus GPU runtime |

Expected uncomplicated scope is about **7–12 focused days**, across roughly
15–25 production files plus tests/examples. A failure of the contact gate
opens a numerical-method task whose cost cannot be inferred from EOS
plumbing; allow a separate investigation before promising the RM date.
Riemann-based schemes and general multi-material walls are not included in
that estimate.

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
   reporting local support and interface resolution. Track the signed
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

## Next implementation step

Start phase 1 with the material schema, uniform fallback, fused ideal-gas
kernel and analytic round-trip tests. Continue through the complete CRKSPH
thermodynamic path before adding the RM case. Keep phases 4 and 6 as
acceptance gates, not optional demonstrations.
