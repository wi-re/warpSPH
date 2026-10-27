# Native DFSPH pressure-boundary experiment

`DFSPHConfig.pressureBoundary='band2018'` selects explicit pressure unknowns
on fluid and static sampled solids. All three native step orders use the same
geometry, stage solver, timestep-scaled histories and 0.9ρ₀ sampling-recovery
gate. The existing native self/contact closure remains the default.

This is a **volume-centric extension of Band et al. 2018 to two DFSPH
constraints**, not a claim that the paper publishes this DFSPH composition.
It changes the fluid force discretization as well as the wall closure. The
paper's single-layer cubic-wall calibration is not used for our sampled volume
band. Nominal rest volumes and full kernel volume sums remain explicit.

## Equations and implementation

Let V₀=m/ρ₀, sᵢ=ΣⱼV₀ⱼWᵢⱼ, Vᵢ=V₀ᵢ/sᵢ and q=p/ρ₀. For static solids,

```
Fᵢ(q) = -(Vᵢ/V₀ᵢ) Σⱼ Vⱼ(qᵢ+qⱼ) ∇Wᵢⱼ   (fluid)
Fᵢ(q) = 0                                    (solid)
Cᵢ(v) = Σⱼ Vⱼ(vᵢ-vⱼ)·∇Wᵢⱼ

dt² CF(q_density) = 1-s-dt C(v*)
dt  CF(q_div)     = -C(v)
```

Fluid and solid rows both participate in the constraints. The solid diagonal
contains only the adjacent fluid reaction term; it has no solid self-motion
term. The bare pair primitive computes the exact diagonal, with no analytic
provider contribution. Pressure impulses remain p·dt² and p·dt. Forces act
only on fluid particles; pressure unknowns also exist on fluid-facing solids.
Interpolation ghosts never enter neighbour sums or the unknown set.

The state is not temporarily overwritten: a geometry-local shallow proxy
provides apparent volume weights to the pair operators. The solver uses the
native relaxation, pressure clamp, minimum iterations and tolerance units. Its
error average includes the fluid and fluid-facing solid rows. There is no
legacy Tikhonov shift, median-diagonal cutoff, residual floor or automatic
pressure gauge. The absolute weak-factor threshold remains explicit.
Solid unknowns require kernel-weighted fluid contact greater than 0.02;
deeper band samples retain zero pressure. This is a declared local support
mask, not the paper's surface-sampling rule.

Clamped DF pressure is pinned below the configured fluid sampling threshold;
neighbour forces remain active. The density stage and signed constraints are
not gated. The known IISPH contact-gravity force is omitted: wall pressure is
already solved as an unknown, so adding that extension would count it twice.

Only uniform-support, static sampled walls are supported. Analytic walls lack
the required solid pressure degrees of freedom and are rejected. Moving solids
are also rejected. Their existing native self/contact route remains available.

## Enable and reproduce

In a case configuration hook:

```python
sc = ctx.schemeConfig
sc.iisph.boundaryRepresentation = 'particles'
sc.iisph.boundaryPressure = 'self'  # contact interpolation is not used here
sc.dfsph.pressureBoundary = 'band2018'
sc.dfsph.density.maxIterations = 1000
```

The larger budget is an explicit experimental setting, not a hidden override.
The matched tank needed up to 272 density iterations during startup. A budget
of 100 capped and produced a biased short-run pressure field.

Use an already generated shared fixture to avoid changing the physical IC:

```bash
python scripts/probe_preparedProjection.py \
  --source .tmp/incompressible-native-matrix-2026-10-26-night \
  --out .tmp/my-pressure-boundary-control \
  --row hydrostatic__dfsph__particles --pressure-boundary band2018 \
  --max-iterations 1000 --t-limit 4
```

The probe requires a fresh destination, records overrides and current source
snapshots, and uses the matrix's video, progress and failure guards. The
source matrix and fixture are preserved.

## Qualification, 27 October 2026

Independent dense 2D/3D checks cover force, continuity and every basis-column
diagonal with unequal masses; signed density/divergence Jacobi, density units,
input ownership, ghost exclusion and the sampling gate are tested. Twelve
tests pass in FP32 and twelve in FP64. Native-route regression tests also pass.

The four-second classic-order tank completes with zero caps or penetration.
Its last-10%-time median bulk pressure-gradient ratio is 1.00128, RMS speed
0.000245 m/s and pressure-fit RMS 0.057 m²/s². Wall/engine cost is 235.67 s /
54.41 ms per step, versus the native matched tank's 346.25 s / 80.74 ms.
These are single recorded runs with output, not isolated solver benchmarks.

Predictor-first also completes a quiet 0.25 s tank control. VD+PS at the same
loose DF tolerance remains physically underconstrained: RMS speed 0.388 m/s
and 219/250 shift caps, despite a plausible pressure slope. A pressure fit
alone does not qualify that composition.

The one-second dam-break prefix completes without caps or penetration, but
its integrated specific kinetic energy is 85.15% of the native DFSPH control
and 84.00% of retained Delta-SPH over the same interval. That extra loss is a
material drawback. The full dam break and sloshing are not qualified here.
The plotted solved solid pressures also show a broad spread around the
hydrostatic profile, including bottom-row values well above the expected
pressure. Correct fluid bulk gradients do not validate these wall unknowns.
The full-band extension needs further conditioning/sampling work; a faithful
single-layer boundary discretization from the paper is a separate next step.
This option stays experimental; it does not replace matrix results or defaults.

Local videos, source receipts, pressure curves and all failed attempts are in
`.tmp/tgv-pressure-investigation-2026-10-27/index.html`.

Reference: [Band et al., Pressure Boundaries for Implicit Incompressible SPH](https://cg.informatik.uni-freiburg.de/publications/2018_TOG_pressureBoundaries.pdf),
Eqs. 8–10 and 16–20. Spheral has no corresponding incompressible pressure-boundary
solver to use as a reference; its compressible formulations are a different model.
