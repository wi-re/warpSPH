# Analytic boundaries — exact kernel integrals over the solid instead of boundary particles

> **Progress marker:** this plan's row in [PLANS.md](PLANS.md). Update its *Last worked* date and *Where it stands* whenever you work on this plan.

Opt-in alternative to boundary particles for the weakly compressible δ⁺-SPH scheme (`deltaSPH`): a wall or obstacle is a `RigidBody` whose
`representation` is an analytic body of the optional package **warpSPHBoundaries** (surface loops, boxes, implicit / SDF solids; the SPH kernel integrals
over the solid are evaluated exactly by edge line integrals, with no wall particles, no ghost layer and no packing error). Without a representation
nothing changes: every hook is guarded by `schemeConfig.boundaryProvider is not None`. Derivations, the reference solver `DeltaSPH2D` and the oracle tests live in
warpSPHBoundaries (repository `wi-re/warpSPHBoundaries`; `docs/plan-upstreaming.md` there has the history of this change set, `docs/audit-warpsph-boundary-hooks.md` the hook audit).

## Using it

```
pip install -e warpSPHBoundaries/        # or: pip install -e "warpSPH/[boundaries]"
warpsph-run dambreak --wallRepresentation analytic                       # tank walls analytic
warpsph-run dambreak --wallRepresentation analytic --obstacleActive --obstacleType circleBottom --maxExtent 0.5 --offsetX -0.4
```
`--obstacleRepresentation particles` keeps the obstacle as boundary particles next to the analytic tank (a mixed scene); `--obstacleDynamic` makes an analytic
obstacle a free body (density `obstacleDensity`); `--shiftScheme` / `--shiftProjection` select Michel 2022 or the implicit / dynamic shifting; `--analyticWallPressure normal` replaces
the hydrostatic wall-pressure term by its normal part only (as the boundary particles' ghost extrapolation does); `--cudaGraph` captures the whole step for static walls.

## What was added (stage numbers of `_deltaSPH_rhs`)

| piece | where |
|---|---|
| `RigidBody.representation`, `buildAnalyticRigidBody` (mass / inertia from the representation), `ParticleRegion.representation` (not sampled, no ghost layer) | `rigidBody/build.py`, `regions/region.py`, `initializers/weaklyCompressible.py` |
| provider protocol + adapter (`bindBodies` writes the integrated pose into the provider's bodies every right-hand side) | `boundary/provider.py` |
| wall continuity (05b), wall pressure force (12), wall viscosity (11, inside the frozen-diffusion tuple), wall loads (13b) | `modules/analyticBoundary/wallTerms.py`, `schemes/deltaSPH.py` |
| free-surface detection: Barecasco as separate partial-sum passes, the wall continuum added between them, normals / lMin from fluid + wall matrix | `modules/analyticBoundary/detector.py`, `modules/surfaceDetection/wp_barecasco{Cover,Cone}.py` |
| shifting: the wall's share of the raw δ⁺ sum (`wp_deltaShiftRaw`, warpSPH's `W0` convention), of grad C-tilde and `U_char` (Michel), of grad C (implicit) | `modules/analyticBoundary/shifting.py`, `modules/shifting/` |
| no-penetration impulse from the provider's signed distance | `modules/analyticBoundary/noPenetration.py`, `systems/weaklyCompressible.py` |
| **the wall flux in `WeaklyCompressibleSystem.drift_rates`** (time-centred continuity): without it the wall's compression is missing from the density update and the dam break diverges within 0.06 s | `systems/weaklyCompressible.py` |
| free analytic bodies through the exact integrator (`RigidBody.dynamic`, `load`) | `systems/weaklyCompressible.py` |
| whole-step CUDA graph with analytic walls (host constants `_analyticSupport` / `_analyticMass` / `_analyticDx` fixed at initialisation) | `schemes/deltaSPH.py` (`_rhsIsGraphable`), `initializers/weaklyCompressible.py` |

## Refused (loudly), by design of this change set

3D; `BCType.extended`; the `exactHessian` implicit operator with walls (needs the wall Hessian integral); obstacle presets other than circle, box and equilateral triangle
(an ellipse, `aspectRatio != 1`); the right-hand-side graph and the whole-step graph for moving or dynamic bodies, implicit / dynamic shifting and mixed scenes; the incompressible
and ACSPH consumers (not hooked). Each is a separate next step, not a precondition. *(Until 2026-10-09 the last item was refused in the text only: the incompressible and ACSPH schemes ran with analytic walls and no wall at all. `initializeSimulation` now raises `NotImplementedError` for every scheme but the weakly compressible δ⁺, `tests/test_analyticRefusal.py`; a scheme leaves the list when it is ported.)*

## Verified (A/B against the boundary-particle path, `dambreak`)

| comparison | result |
|---|---|
| dam break 0.5 s, nx 64, delta+, analytic vs particle walls | fastest particle 5.340 / 5.345, kinetic energy 3.567 / 3.712 (4 %), density bounds equal, no penetration |
| Michel 2022, 0.3 s | 1.9 % (fastest particle), 4.8 % (kinetic energy) |
| implicit shifting vs the analytic δ⁺ run (the particle path diverges there, OPEN_PROBLEMS §26) | 1.1 % / 0.04 % |
| circular / box / triangular obstacle | kinetic energy within 3-4 %, fastest particle 6-11 % lower (a spray particle); mixed scene stable, kinetic energy within 1.1 % |
| free cylinder, density 0.5 / 2.0 | accelerates at 3.0 (float64) / 2.5 (float32) m/s² against the ideal 3.27 (added-mass limit); the float32 load of a buoyant body is 3 % low (small difference of large numbers) |
| whole-step graph | bitwise the eager step (tank, tank + obstacle, Michel), about 3x faster |

The term-by-term equalities with the reference solver (wall continuity / pressure / viscosity 1e-13 in float64, detector masks exactly, shifting 1.4e-4 relative in float32, wall loads) are
tests of warpSPHBoundaries (`tests/warpsph/`, run with this tree on `PYTHONPATH`, in both precisions): warpSPH does not import that package's solver layer.

## Tests and gates

`tests/test_analytic*.py` (skip without warpSPHBoundaries), `scripts/gradcheck_analyticBoundary.py` (cover vector and raw shift sum gradchecked; the wedge count is discrete and checked by value),
`scripts/check_imports.py` (`warpSPHBoundaries` is an optional package: its imports are only verified when it is installed).

## Next

Incompressible / ACSPH consumers (hook table in warpSPHBoundaries' audit); moving bodies in a graph; 3D; the wall Hessian integral for `exactHessian`; wall particles of the `ParticleBoundary` provider as a
cross-representation convergence test inside warpSPH.

**Direction (user, 2026-10-09):** the analytic-boundary support moves into the front end's solver schemes, aligned with all capabilities there (δ⁺ first, then DFSPH / ACSPH / ...), and the solver code in warpSPHBoundaries (`DeltaSPH2D`, `DFSPH2D`) is retired afterwards. For δ⁺ that waits on the boundaries repo's δ⁺ giving usable results: its validation is internal and was not run on full cases such as sloshing. First full-run check: `sloshingTank --wallRepresentation analytic` (δ⁺, Wendland C2, Sensor 1 = fluid probe) against the particle-wall run, 7 s, `examples/sloshingTank/output/analytic_ab_2026-10-09`.

Survey of what the boundaries repo's solvers do, term by term against warpSPH's modules, and the proposed module-level shape: [ANALYTIC_BOUNDARIES_PORT_SURVEY.md](ANALYTIC_BOUNDARIES_PORT_SURVEY.md).
