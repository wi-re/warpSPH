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

`shape=ellipse` and a common `boost` also work with the single ideal-gas EOS.
The two-material variants require material EOS support in the meshless flux
operator and remain unavailable. `closure=none` remains the general default;
the graph projection is not required to balance this regular contact.

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

`outerBoundary=open` remains unsupported for MFM/MFV. Dropping seam neighbors
without supplying reservoir/wall fluxes leaves an unclosed particle domain.
A future inflow route must prescribe incoming states and account for boundary
mass, momentum and energy; it must be labeled separately from the existing
SPH open-edge comparison. Periodic seam rarefaction limits the useful time and
spatial comparison window even when the central shock is accurate.

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
