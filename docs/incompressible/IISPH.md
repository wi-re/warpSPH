# Native IISPH

The `iisph` scheme uses a single IISPH density projection per timestep. It
replaces the former wrapper around `dfsphReference` with rest-volume
continuity and density-dependent pressure coefficients. `dfsphReference`
retains its composed DFSPH formulation.

## Functional API and ownership

`schemes/incompressible/iisph.py` follows the numbered module-call layout
of delta-SPH: Core adjacency, density, Dirichlet conditions, forces,
boundary geometry, pressure solve, derivative assembly and update boundary
conditions. It does not advance positions or velocities.

`modules/iisph` exposes ordinary functions:

- `computeIISPHBoundaryIntegrals(state, config, schemeConfig)` returns the
  analytic provider's raw per-body geometry, or `None` for sampled walls.
- `computeIISPHContactPoints(state, schemeConfig)` returns closest surface
  points when contact pressure is selected, independently of the integrals.
- `computeIISPHDensityRate(state, field, coefficients)` applies continuity.
- `computeIISPHPressureAcceleration(state, normalizedPressure, coefficients)`
  applies the linear pressure force; `computeIISPHPressureOperator` composes
  force and continuity.
- `solveIISPH(state, config, schemeConfig, acceleration, dt, adjacency,
  previousPressure=None, boundaryIntegrals=None, contactPoints=None, gravity=None)`
  assembles the pressure coefficients, contact-pressure weights, exact Jacobi
  diagonal, and known gravity extrapolation inside the solve. It returns an
  `IISPHResult` without modifying input fields. The supplied nonpressure
  acceleration forms a local predicted velocity inside the solver; it does
  not advance the simulation velocity. Omit previous pressure for a
  cold start. The gravity extension enters both the source and final pressure
  acceleration. The provider's `Cov` is the first spatial moment of the kernel
  gradient, `integral_solid (x_solid-x_i) tensor grad_i W dV`.
- The scheme calls `modules.deltaSPH.computeVelocityDiffusion` directly,
  passing the selected boundary representation and the physical particle fields.

`IISPHCoefficients` and contact/kernel data are plain
records of tensors. They retain no simulation state, configuration, callable,
or solver instance. Coefficients belong to the supplied positions and must
be recomputed after particles move. The Warp kernels in `wp_operators.py`
use Core neighbor lists, kernel functions and precision. Core's stationary
driver controls iteration. The standard `semiImplicitEuler` driver advances
velocity from the combined nonpressure and pressure acceleration, then
advects positions with that updated velocity. IISPH selects this registered
single-evaluation driver through its scheme bundle. `IISPHSystem` uses normal
derivative updates and carries only summed density, pressure, and other
recomputed fields at finalization. It does not run the VD+PS shifting hooks.
The RHS has no kick/drift calls or zero-update full-map adapter.
Configuration is explicit in
`IncompressibleSPHConfig.iisph` and uses the existing codecs.

Density uses the shared `modules.density.computeDensities` function. Its
`boundaryRepresentation` and `wallMassDensity` arguments select sampled
walls or a geometry-only solid integral in physical density units. The
scheme does not construct wall density from pressure boundary data.
Dirichlet conditions run after this summation. Pressure coefficients read
that resulting density; they neither recompute it nor overwrite prescribed
values. Initial density reporting uses the same summation without building
pressure coefficients or evaluating gravity.

There is no SciPy dependency or assembled pressure matrix in this path.
Geometry and iterative fields stay on the state's CPU or CUDA device.

## Equation and pressure units

Let `V=m/rho0`, `s=rho/rho0` and `q=p/rho0`. For static walls,

```
C(v)_i = sum_fluid_j V_j (v_i-v_j).gradW_ij + G_i.v_i
F(q)_i = -sum_fluid_j V_j (q_i/s_i² + q_j/s_j²) gradW_ij
         -G_i (q_i/s_i² + q_contact_i)
b = 1-s-dt C(v_predictor)-dt² C(a_gravity_extension)
A = C F
```

The diagonal is the exact diagonal of this coupled `A`, including the
nonlocal contact interpolation. Each relaxed Jacobi step evaluates `A q`
through force and continuity reductions. The scheme stores physical `p`,
and starts the next solve from half the previous pressure.

The stopping statistic follows the IISPH reference policy: signed density
error on pressure-active rows, evaluated using the new diagonal and old
off-diagonal iterate, averaged over all fluid rows. It is neither an
absolute residual norm nor a divergence-free convergence test. Checks occur
each iteration, after the configured minimum.

## Configuration

`IncompressibleSPHConfig.iisph` is an `IISPHConfig`, serialized under `iisph`.
It is independent of `solverConfig`'s DFSPH and shifting solver settings.

| Field | Default | Meaning |
| --- | --- | --- |
| `minIterations` | 2 | Minimum pressure iterations |
| `maxIterations` | 100 | Iteration budget |
| `maxErrorPercent` | 0.01 | Signed mean density error, percent of `rho0` |
| `relaxationFactor` | 0.5 | Fixed Jacobi relaxation; wider supports may require a smaller value |
| `boundaryRepresentation` | `auto` | `analytic` with a provider; otherwise `particles` |
| `boundaryPressure` | `self` | `self` or `contactShepard` |
| `contactGravity` | `True` | Affine gravity extension for `contactShepard` |

`self` gives static solids no pressure unknown of their own. It retains the
fluid's self-pressure wall force. `contactShepard` reconstructs a nonnegative
Shepard pressure value at the closest solid contact, inside every pressure
iteration. With gravity enabled it extends that value into the solid as
`q(x)=q(contact)+g.(x-contact)`. The known affine acceleration enters both the
pressure source and final acceleration.

For sampled contact walls, supply their contour through `boundaryProvider`
and explicitly select `boundaryRepresentation='particles'`. The provider
then supplies contact geometry only; density, gradients and gravity
extension use the actual wall samples. Analytic walls use solid integrals
and their first moments. Their dimensionless density contribution is
multiplied by physical `rho0`.

Viscosity uses the shared delta-SPH operator on physical masses and densities.
`diffusionParams.approachOnly` selects approaching pairs only (`True`) or
two-sided diffusion (`False`, the default). The shared projected operator
converts the configured kinematic viscosity to reference dynamic viscosity
`mu0 = rho0 * nu`; IISPH has no viscosity adapter or normalized state copy.

Free slip uses `u_mirror = u_wall + w_t - w_n`, where `w = u_fluid - u_wall`.
The normal component is reflected while the tangential component is retained.
Sampled walls interpolate fluid velocity at reflected sample positions into
an independent viscous reference field; stored rigid-body velocities remain
unchanged. Existing mDBC ghost rows reuse their established boundary field.
Analytic walls integrate the same local mirror closure through the solid
Laplacian, using geometric surface normals without evaluating wall pressure.
The selected boundary representation prevents counting analytic and sampled
walls together. Wall velocity conditions come from the configured `BCType`;
use `freeSlip` for the Marrone setup. Omitting walls from diffusion is a
separate control and does not impose this boundary condition.
There is no shifting, divergence pass or no-penetration position correction.

## Supported scope and checks

The particle/periodic operator supports 2D and 3D, one fluid, common fixed
support, kind-0 fluid and kind-1 stationary solids. Analytic solid integrals
currently support 2D Wendland C2/C4. Moving bodies, ghost rows, adaptive
supports and autograd are rejected. CUDA graph capture is not implemented.

Initial masses and geometry must be calibrated before stepping. Preserve
free-surface kernel deficiency; do not calibrate it away. IISPH initialization
consumes `config.calibrateNormalization`: it applies the lattice factor once
to every particle mass and to analytic solid packing, then sets that flag
to false. All subsequent IISPH operations use ordinary kernels and gradients.
The applied factor is recorded in `ctx.scratch['initialMassCalibration']`.
Already calibrated inputs keep `config.calibrateNormalization=False` before
initialization. The first density report uses the shared density function.
The returned timestep density is the summation at the solve's pre-drift
positions, as in the IISPH recurrence.

`IISPHResult` exposes pressure, acceleration, iterations, error, termination
and the corrected source. Scheme results are available on the last stage
update's `iisphResult`; runner hooks access that update through
`ctx.scratch['lastStageUpdate']`. A budget termination is reported without
silently increasing the configured cap. Nonfinite pressure raises an error.

`tests/test_projectionModules.py` checks independent dense force/continuity
oracles, the exact coupled diagonal, iteration policy, density-unit scaling,
configuration round trips, policy isolation and CPU/CUDA parity. It also
checks RHS input ownership, standard integrator selection, derivative boundary
conditions, fixed sampled walls, summed-density carry and pressure warm starts.
`tests/test_analyticIisph.py`
exercises the standard scheme/driver on an analytic tank.
