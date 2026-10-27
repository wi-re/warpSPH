# Meshless case initialization and boundaries

MFM and MFV integrate `Q = (m, m v, E)` with density `m / V`, where `V` is the
number-volume partition returned by `MeshlessWarp`. The active integrator is
`ConservedLagrangian`: MFM defaults to SSPRK and MFV to Hancock. The usual SPH
primitive-field boundary hooks do not update this conserved state.

## Pressure equilibrium (`hydrostatic`)

This case is a periodic dense square or ellipse at uniform pressure, without
gravity. Both meshless methods have a native regular-lattice initializer. Cell
masses carry the requested density jump, supports come from the number density,
and thermal energy is `P V / (gamma - 1)` using the first RHS's actual geometry.
This prevents the initial primitive refresh from introducing a pressure jump.
No SPH mass-weighted support adaptation or CRK density renormalization is used.
No equilibrium correction is applied during evolution.

Both the common-gamma case and the `twoGas` variant (gamma 1.4 and 5/3) are
supported. `advectedEllipse` combines two material EOS entries, an elliptical
inclusion and a common translation. Each particle's gamma is used consistently
for initial energy, primitive refresh, sound speeds, prediction and Riemann
states. The scalar-gamma path remains available without a material table.

Material IDs denote pure fluids. MFM retains zero mass flux everywhere. MFV
uses a contact-moving, zero-mass-flux face between different material IDs, while
retaining its mass exchange within each fluid. Pair momentum and pressure work
remain antisymmetric; each material's total mass is conserved up to arithmetic
roundoff. This is an immiscible-material interface treatment, not a mixed-species
EOS or a species-mixing model. Negative thermal energy on the explicit material
path remains visible rather than being repaired by a pressure floor.

`closure=none` remains the general default; the graph projection is not required
to balance this regular contact.

The separate gravity-driven `hydrostaticColumn` tank is still outside meshless
support. It requires gravity in the conserved momentum and energy sources,
consistent source prediction/reconstruction, wall face fluxes and free-surface
pressure treatment. Adding its case name to the meshless scheme list would not
supply those numerical operators.

## Noh

The native periodic 1D Noh setup already runs with both methods. Use float64 for
the cold inflow: subtracting kinetic from total energy in float32 can lose the
small thermal energy and yield nonpositive internal energies even when the
shock-density metric looks accurate. The matrix uses the same explicit
`uAmbient=1e-8` and float64 for all Noh methods. Exactly cold runs remain possible,
but positive thermal energy is not a meaningful invariant for their unshocked gas.

`outerBoundary=open` is supported for MFM/MFV with `closure=none`. The native
cell-centred initial particles, masses and IDs are retained, and the execution
domain becomes nonperiodic. Exterior quadrature particles complete the number
volumes and supply the analytical incoming density, velocity and thermal
energy at every temporal stage. Their positions follow the incoming streams;
they are not integrated, exported or counted as physical fluid particles.
Resolved configurations with `periodic=false` reproduce the same initial lattice.

This is an analytic Lagrangian inflow reservoir, labeled as such in exported
`boundaryTreatment` metadata. It differs from the SPH open-edge route, which
only removes periodic seam interactions. The physical domain has finite
Lagrangian mass: stop before the shock reaches its material edge,
`t < (L/2)/(1+v_s)`. The default L=2 and t=0.6 satisfy this limit; the setup
rejects longer requests unless L is enlarged. This route does not implement
ongoing particle injection at a fixed Eulerian boundary.

The reservoir supplies exterior pressure work and, for MFV, numerical mass and
energy exchange. Full-fluid energy is therefore not an exact closed-system
invariant. The native periodic route remains available for retained comparisons.

## Planar Kidder

The meshless sampler uses physical shell masses and a number-density support.
The first and last `band` particles follow the analytical Lagrangian trajectory.
At every RHS stage their positions are prescribed before geometry construction;
then momentum and thermal energy are prescribed on that geometry. The accepted
state repeats this operation after temporal combination. Boundary particle
masses stay fixed in MFM and MFV; MFV interior particles may exchange mass.
The remaining interior evolves through the ordinary reconstructed Riemann flux.

The prescribed bands are external reservoirs: they supply momentum and work,
so the complete shell's total energy is not a closed-system invariant. Near the
exposed ends, number-volume deficiency means `m/V` differs from analytical
boundary density; comparisons exclude the driven bands, as in the SPH case.
Use sufficiently wide bands to keep this edge error outside the fluid interior.
The native controls (`n_h=4`, `band=10`) provide this buffer.

Only `dim=1` and `closure=none` are supported for this route. A global zero-face-
sum projection on a graph with exposed ends cannot represent the prescribed
boundary pressures. Cylindrical/spherical shells need their own sampling and
geometric terms. The timestep uses the meshless signal-speed dispatcher, rather
than the SPH acoustic timestep previously attached to Kidder.

Examples (run from an environment with the sibling packages installed):

```bash
warpsph-run hydrostatic --scheme MFM --nx 100 --tLimit 7
warpsph-run hydrostatic --scheme MFV --nx 100 --tLimit 7
warpsph-run noh --scheme MFM --nx 256 --precision float64
warpsph-run kidder --scheme MFV --nx 200 --precision float64
```

Versioned examples for the unequal-gamma contact and open-edge inflow:

```bash
warpsph-run --config examples/configuration/hydrostaticMeshless.yaml
warpsph-run --config examples/configuration/nohMeshlessOpen.yaml
```

Override `--scheme MFM` or `--scheme MFV` to compare the methods. The open Noh
example chooses float64 before imports; its configured reservoir is retained
when rerunning a saved nonperiodic configuration.

## Prescribed thermal state in driven bands

`DrivenParticles` retains its trajectory and `(velocity, pressure)` primitive
callback. Passing an additional `specificInternalEnergy(time, positions)`
callback instead prescribes u, forms `E=m*u+KE`, and lets pressure follow the
fresh geometric volume. The primitive callback may return `(velocity, None)`
in this mode. u must be finite and positive. It is a driven fluid reservoir,
not an impermeable conducting-wall or physical heat-diffusion model.

The prescribed initial state is applied before the first diagnostic. Accepted
projections accumulate `state.boundaryProjectionBudget` (FP64 thermal and
kinetic energy changes); stage calls do not advance it. The counter records
projection exchanges, not all reservoir flux or mechanical-reaction work.
[CPU controls and limits](thermal_state_qualification_2026-10-27.md).
