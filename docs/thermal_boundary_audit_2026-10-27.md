# Thermal boundary support audit — 2026-10-27

The schemes do **not** yet share a qualified thermal-wall boundary contract.
Rayleigh–Taylor (RT) does prescribe internal energy, density and velocity in
fluid buffer bands. That is useful existing infrastructure, but accepted-state
persistence is incomplete, and prescribed gas reservoirs do not establish
physical wall heat transport for Rayleigh–Bénard convection.

This is an audit of `a237cfa` on `codex/meshless-boundary-balance`. No solver,
case initialization, boundary or numerical defaults were changed. All new
controls used CPU while the independent GPU work continued.

## Capability inventory

| Scheme | Existing prescribed thermodynamic state | Limitation for thermal walls |
|---|---|---|
| CRKSPH, CompSPH | RT callbacks set rho/u/v before each stage EOS | Accepted compatible-work update can replace prescribed u; RT has no accepted-state restoration. No physical conduction operator in these RHSs. |
| Monaghan | Same stage rho/u/v dispatcher | RT accepted u is integrated from the initial state and hydro rates, not replaced by the target. Accepted pressure/entropy diagnostics retain the last stage. Separate artificial conductivity is present. |
| GSPH, Inutsuka GSPH | Same stage rho/u/v dispatcher | Same accepted-state gap. Riemann pressure work and shock dissipation are present; explicit physical heat diffusion is absent. |
| PESPH PressureEnergy | Converts prescribed u/v to authoritative total E, then refreshes the pressure-volume closure; repeats at acceptance | Tested fixed u is held to roundoff. Prescribed density is not retained as the pressure-volume or summation density. Solid-wall closure and physical conduction remain unqualified. |
| PESPH PressureEntropy | Converts prescribed rho/u to authoritative A, then refreshes the closure, including at acceptance | Prescribed A is held, but reconstructed u need not equal the requested u. This is an entropy prescription rather than an exact temperature boundary. |
| MFM, MFV | `DrivenParticles` prescribes trajectory, fixed mass, velocity and pressure at fresh stages and acceptance; Kidder uses it. Noh has a separate analytic inflow quadrature. | RT is excluded by its case schema. Fixed pressure at geometry-dependent density does not fix temperature. No physical heat diffusion operator. |
| WCSPH, incompressible families, ACSPH, acoustic/wave systems | Mechanical and pressure/density boundary machinery | Their current state contracts have no evolved temperature/internal-energy transport field. Thermal scalar transport and buoyancy coupling would be an additional model. |

A `thermalConductivity` configuration field alone does not establish heat
transport. CRKSPH and CompSPH pressure/viscosity pair work does not invoke the
separate conductivity operator. Monaghan and dissipative PESPH do invoke it;
its coefficient feeds signal-speed/pressure/viscosity-based artificial
conductivity, not a specified dimensional Fourier conductivity or diffusivity.
PESPH entropy dissipation additionally depends on the selected heating policy.

The compressible solid-wall path gathers fluid u, optionally reconstructs a
wall Riemann state, and freezes wall rates. It has no prescribed wall temperature
or normal heat-flux contract. These mechanics are distinct from RT's fluid bands.

## RT controls and reproduced failures

Used the registered case with CPU, nx=24, band=3, three accepted steps,
maxDt=0.001, unchanged native time integration and other case defaults.
Measured the relative error `max_band |u-u_target|/|u_target|`, after every
native RHS and at the endpoint. These are deliberately coarse callback/lifecycle
controls, **not** a resolved RT benchmark or an assessment of bulk instability.
Tiny controls omitted video; progress was streamed and retained.

| Scheme | FP32 endpoint u error | FP64 endpoint u error |
|---|---:|---:|
| CRKSPH | 21.4543% | 21.4541% |
| CompSPH | 21.4543% | 21.4541% |
| Monaghan | 21.7743% | 21.7740% |
| GSPH | 21.8840% | 21.8838% |
| Inutsuka GSPH | 21.8226% | 21.8223% |
| PESPH PressureEnergy | 1.17e-5% | 2.13e-14% |
| PESPH PressureEntropy | 7.61308% | 7.61300% |

All five ordinary SPH schemes have **zero measured target error inside every
sampled RHS**, in both precisions. Their endpoint is t≈0.002. Both PESPH
variants reach t≈0.003; PressureEnergy holds u to roundoff at RHS and acceptance,
while PressureEntropy already differs after each RHS closure refresh. Every
completed control remains positive and advances normally. MFM and MFV are
rejected by RT's supported-schemes schema, before the older explicit refusal
inside `buildSystem` can run.

The ordinary-SPH 21% number is largely an **initialization mismatch**, not a
new 21% thermal instability: the sampler first installs the boundary values,
then relaxes density/support and reconstructs u from the initial pressure,
overwriting the boundary prescription. At build return, the coarse control
already has a 21.454% maximum u discrepancy in its buffer. The compatible
schemes retain that initial u in these nearly static bands; ordinary integrated-u
schemes additionally accumulate hydro rates. Their accepted mass-density EOS
residual reaches approximately 27–28%, although their stage EOS is consistent.
The location and magnitude depend on the coarse finite buffer/support geometry;
these percentages must not be extrapolated to the default nx=128/band=20 run.

A second control removes that initial mismatch explicitly and changes only the
boundary target to `u_target(y,t)=u_RT(y)*(1+10*t)`. Initial u/rho/EOS errors are
zero. Every ordinary-SPH RHS again matches the target, while endpoint errors are:

| Scheme | FP32 endpoint u error | FP64 endpoint u error |
|---|---:|---:|
| CRKSPH | 1.96107% | 1.96107% |
| CompSPH | 1.96107% | 1.96107% |
| Monaghan | 2.59755% | 2.59756% |
| GSPH | 2.38840% | 2.38840% |
| Inutsuka GSPH | 2.32723% | 2.32722% |

This isolates the accepted-state gap from RT initialization. The accepted
state is assembled from the initial state and derivative combination; changing
an RHS copy's u does not enforce a Dirichlet constraint on that combination.
Kidder already addresses this for ordinary SPH with a case `postStep` hook;
RT has no corresponding hook. A general fix should handle accepted thermodynamic
state consistently rather than rely on every case to rediscover this lifecycle.

PESPH requires separate interpretation: its EOS uses **thermodynamic volume
density**, not the ordinary mass-density sum. The raw audit's
`massDensityEOSRelativeError` is not a valid PESPH consistency criterion. A
separate closure check gives pressure-volume EOS residuals below 1.14e-7 in
FP32 for both variants. PressureEntropy's prescribed A error is exactly zero,
but its u error is 7.613%; resetting A from target rho/u once does not solve the
coupled pressure-volume temperature constraint. The existing two band unit
checks pass because they assert the translated authoritative E/A **before**
closure refresh, rather than the resulting boundary u afterward.

## Meshless reservoir controls

Ran Kidder at nx=32, band=4, three steps for both MFM and MFV under SSPRK,
Hancock and KDK, in both precisions. All twelve controls finish at
`t≈5.10835e-5`, with positive raw u. Across these controls:

- prescribed position and mass errors are zero;
- velocity error is at most 3.73e-9 in FP32;
- relative pressure error is at most 1.25e-7 in FP32 and 2.42e-16 in FP64.

This confirms the existing accepted reservoir prescription. It does not validate
an impermeable thermal wall. Reservoir energy is
`E=P*V_geometry/(gamma-1)+m*v²/2`; when V changes at fixed mass and prescribed P,
u and hence temperature can change. An exact fixed-u/T prescription needs the
current geometry to set the conserved energy, rather than simply reusing a
fixed-pressure callback. The current callback receives time and positions.
The RT refusal's claim that the entire meshless core has no forcing is stale:
conserved body-force sources and driven bands now exist, but the RT adapter,
gravity-balanced initialization and thermal qualification still need work.

## Implementation order before thermal convection

1. Define a shared boundary contract separating driven gas reservoirs from
   impermeable walls, and fixed temperature/u from fixed entropy, fixed pressure
   and specified normal heat flux. Define heat capacity/caloric units; u is
   proportional to T only for the chosen caloric EOS.
2. Enforce targets after initialization and at fresh stage and accepted times.
   Reconcile authoritative u, E or A, density/volume and derived EOS diagnostics.
   Account explicitly for boundary energy/momentum exchange, after successful
   compatible-work replay. For entropy formulations, a nonlocal closure can
   change neighboring thermal energies, so auditing only the clamped rows is
   insufficient. Preserve existing source-free/default paths.
3. Add a geometry-aware meshless u/T boundary adapter; port RT only with matched
   gravity/source work and initialization controls. Keep a prescribed density
   reservoir separate from a thermal wall whose density follows pressure and
   the mechanical boundary condition.
4. Add a dimensional physical heat-transport operator with a diffusion timestep
   bound and defined coefficients. Compatible-energy schemes must retain its
   accepted heat increment rather than overwrite it with hydro pair work.
5. Qualify fixed/time-varying boundary temperatures through multiple tableaux,
   restart and native replay; measure boundary heat exchange and closed-system
   energy balance. Then validate analytic 1D diffusion, hot/cold plates, a
   gravity-supported conductive equilibrium, and convection onset/heat flux.

No thermal-wall capability is promoted by this audit. The next useful repair
is the initialization/accepted-state contract; a long RT or convection run
would not resolve these implementation gaps.

## Evidence and costs

Raw drivers, JSON measurements, progress and selected existing tests are retained
under `.tmp/thermal-boundary-audit-20261027/`. `provenance.json`, source archives
and SHA-256 checksums identify the measured source and environment. The three
primary drivers repeat in FP32/FP64; the closure-specific check uses FP32.
There are 36 successful primary three-step controls, two auxiliary closure
controls and four expected RT schema rejections: 114 accepted CPU steps.

Four existing PESPH band/conductivity tests pass (19 deselected, 0.82 s).
They establish their narrower current contracts, not full thermal-wall support.
The independent endpoint controls above expose gaps not covered by those tests.

Control costs include Python/run setup, cache loading/compilation and observers:
ordinary-SPH RT/ramp wall times 0.34–3.23 s (114–1075 ms/accepted step including
setup); PESPH RT 0.50–6.32 s (167–2107 ms/step); meshless 0.016–2.68 s
(5.3–893 ms/step). These tiny startup measurements are not throughput rankings
or GPU performance results. No GPU was used by these controls.

Code entry points: `cases/compressible/rayleighTaylor.py`,
`caseUtils/compressible/rayleighTaylor/{sample,bcs}.py`,
`modules/boundaryConditions/bcs.py`, `systems/{compSPH,compressibleMonaghan}.py`,
`modules/pesph/{boundaries,stage,thermodynamics}.py`, `systems/pesph.py`,
`modules/meshless/{boundary,reservoir,stages}.py`, `systems/meshless.py`,
`modules/compressibleWall/wall.py`, `modules/dissipation/{rates,wp_conductivity}.py`
and `modules/dissipation/pi/terms.py`, all beneath `src/warpSPH/`.
