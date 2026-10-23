# Native IISPH

The `iisph` scheme uses a single IISPH density projection per timestep. It
replaces the former wrapper around `dfsphReference` with rest-volume
continuity and density-dependent pressure coefficients. `dfsphReference`
retains its composed DFSPH formulation.

## Functional API and ownership

`schemes/incompressible/iisph.py` follows the numbered module-call layout
of delta-SPH: Core adjacency, density, Dirichlet conditions, forces,
pressure boundary geometry and coefficients, wall pressure extrapolation,
pressure solve, Integrators kick/drift, and update.
The scheme orders those calls and assigns their results to the current state.

`modules/iisph` exposes ordinary functions:

- `computeIISPHBoundaryIntegrals(state, config, schemeConfig)` returns the
  analytic provider's raw per-body geometry, or `None` for sampled walls.
- `computeIISPHContactPoints(state, schemeConfig)` returns closest surface
  points when contact pressure is selected, independently of the integrals.
- `computeIISPHCoefficients(state, config, schemeConfig, adjacency,
  boundaryIntegrals=..., contactPoints=...)` consumes the density already
  stored in the state and builds the linear pressure operator and diagonal.
- `computeIISPHBoundaryPressureAcceleration(state, config, schemeConfig,
  coefficients, boundaryIntegrals, contactPoints, gravity)` evaluates the
  known hydrostatic extension separately from the linear pressure operator.
  It applies the calibrated wall mass factor at the physical equation. The
  provider's `Cov` is the first spatial moment of the kernel gradient,
  `integral_solid (x_solid-x_i) tensor grad_i W dV`, not a statistical covariance.
- `computeIISPHDensityRate(state, field, coefficients)` applies continuity.
- `computeIISPHPressureAcceleration(state, normalizedPressure, coefficients)`
  applies the linear pressure force; `computeIISPHPressureOperator` composes
  force and continuity.
- `solveIISPH(state, config, schemeConfig, velocity, dt, coefficients,
  previousPressure=None, boundaryAcceleration=None)` returns an `IISPHResult` without modifying inputs.
  Omit previous pressure for a cold start. The known boundary acceleration
  enters both the pressure source and the final acceleration through the
  explicit `boundaryAcceleration` argument.
- `computeIISPHViscosity(state, config, schemeConfig, adjacency)` evaluates
  the configured fluid viscosity.

`IISPHCoefficients` and contact/kernel data are plain
records of tensors. They retain no simulation state, configuration, callable,
or solver instance. Coefficients belong to the supplied positions and must
be recomputed after particles move. The Warp kernels in `wp_operators.py`
use Core neighbor lists, kernel functions and precision. Core's stationary
driver controls iteration; the package `ProjectedStep` driver owns the
full timestep lifecycle. Configuration is explicit in
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

Viscosity uses the shared configured delta-SPH operator with a normalized
density view, two-sided fluid interactions and zero wall drag (free slip).
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
configuration round trips, policy isolation and CPU/CUDA parity.
`tests/test_analyticIisph.py`
exercises the standard scheme/driver on an analytic tank.
