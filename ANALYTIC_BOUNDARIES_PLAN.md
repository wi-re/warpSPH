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

## Stage 2 plan (owner's order, 2026-10-09; sized for several sessions, none started)

State at the start (survey §6c): δ⁺, `omniIncompressible`, `divergenceFree` on analytic walls, validated on a still tank, the dam break, the English wedge and 7 s sloshing; the MLS wall pressure, XSPH and boundary friction ported.
Open decisions of the owner this plan rests on: the omni relaxation limit and the early sloshing arrival are known DFSPH-style limitations (the latter improves with resolution), not work items; ACSPH is for another time (it is experimental).
Order: **A** the remaining `DFSPH2D` variants, led by the compact CG projection; **B** the `finalize` position shift; **C** `iisph` and `dfsphReference`; **D** the δ⁺ sloshing comparison that decides retiring the boundaries repo's solvers. Each phase ends with its gate, its evidence folder in `docs/analytic_boundaries/`, the PLANS row and a commit; GPU runs one at a time, videos on, frames looked at.

### A. The remaining `DFSPH2D` variants (oracle: `DFSPH2D` in the boundaries repo, identical lattice, term by term as for the omni port: `scripts/probe_omni*Oracle.py`)

| # | variant (boundaries repo) | lands in warpSPH as | check |
|---|---|---|---|
| A1 | **compact CG projection** (`projection='compact'`, `sim/projection.py`: the compact Morris / Brookshaw Laplacian `w_ij = 2 Vt_i Vt_j (x_ij . grad W) / (r^2 + eta^2 h^2) / cal`, Jacobi-preconditioned CG to `projectionTol`, warm start, wall row `D_i` (Robin), wall flux source `projectionWallFlux`, `projectionWall` robin / mirror, `projectionGradient` symmetric / difference / renormalised) | `modules/incompressible/compactProjection.py` (the operator and the CG over the Verlet edge list, torch first, then the Warp / CUDA-graph form, keeping the step sync-free: SMALL_PROBLEM_PERFORMANCE rule), the wall pieces in `modules/analyticBoundary/` (`wallCompactRow`, flux). `omniIncompressible` / `divergenceFree` take `schemeConfig.projection = 'jacobi' (default) / 'compact'` for the **divergence** stage | oracle terms (weights, rhs, D, converged p, corrected velocity); TGV decay 1.00 against 1.38 at 100 composed iterations; hydrostatic tank; dam break to omniSPH |
| A2 | `closedPreset` / `closedDomain` / `pressureGauge` / `divergenceGauge` (mean / min removal), `densitySolve=False`, `densityMode='continuity'` | schemeConfig fields on the two schemes; the gauge as a small module next to `compactProjection` | the preset reproduces `CLOSED_PRESET`; closed analytic box stays at rest, no drift |
| A3 | `shifting='fickian'` / `'fixed'` (`D = shiftA h_s |v| dt` / `shiftA h_s^2`, concentration gradient with the wall completing it, tangential part in the Lind surface layer, cap `shiftCap dx`) | first check what exists: `modules/shifting/delta.py` and `analyticBoundary/shifting.py` (`wallConcentrationGradient`) already carry a wall-aware concentration shift; port only the missing form (fixed D, surface-layer tangential projection) as a mode of that module | oracle shift vector per particle on the lattice and after a few steps |
| A4 | free-surface Dirichlet of the projection (`freeSurface`, `surfaceRho`: P = 0 on surface rows) | the existing `surfaceIndicators` (detector) instead of `rho < surfaceRho`; decision below | hydrostatic tank pressure; sloshing impact without surface noise |
| A5 | moving bodies in the divergence solve (`boundaryInDivergence`, `wallDivergenceClamp`, rigid wall velocity in the source) | `_solve` already has `analyticWallInDivergence` and `wallDivergence(bodyVelocity=True)`; add the clamp of the wall pressure and a moving-body case (rolled tank in the lab frame; the sloshing case rotates gravity instead) | a driven moving wall pushes the fluid, no suction at the ceiling |
| A6 | force / torque bookkeeping per body (`recordForces`, `loads`) | `wallLoads` for the omni loop (slice 3 of the survey: pressure + friction + no-penetration impulse, momentum balance as in `DFSPH2D.balance`) | momentum balance to round-off on the still tank and the dam break |
| A7 | `bodyForce` (periodic driver) | **not in A**: with the channel-flow group (decision of 2026-10-09: after the free-surface cases); decide at the end of A whether the free-surface cases are done enough to open it | - |

Dependencies: A1 first (A2 and A4 are options of it), A3 independent, A5 / A6 independent. Cost guess: A1 two to three sessions (the CG on the graph path is the bulk), A2 - A6 one session each.

**Do the compact projection help the Jacobi limit? An experiment, not an assumption** (A1b). The relaxation limit lives in the relaxed Jacobi of both solves; the CG replaces it only where the operator is symmetric and linear: the **divergence** solve (the compact Laplacian is; the composed `div(grad)` of the scheme is not symmetric at the wall) and, with `densitySolve=False` (the `closedPreset`), the constant-density correction is dropped for a particle shift. The omni **density** solve is clamped (`p >= 0`), nonlinear, and a Krylov solver there failed already (`CD_SOLVER`, DFSPH_IMPROVEMENT_PLAN Part 43). So: measure the largest stable `OMEGA` of the remaining Jacobi iterate with the compact divergence stage at n_h 2.57, 4, 5 on the hydrostatic tank and the dam break; if the density solve stays Jacobi-limited, say so and consider an active-set CG (`p > 0` rows) as a separate item, not part of A1. Also from the boundaries repo's own findings (`docs/plan-next-steps.md`): **free surfaces stayed on the omniSPH-style path there** (the mirror projection gave the right hydrostatic pressure but surface noise and splash clusters), the compact projection is the closed / periodic-flow tool; A1 reproduces that finding on warpSPH's sloshing before anything is made the default.

### B. The `finalize` position shift (VD+PS) on analytic walls

`IncompressibleSystem.finalize` runs `solveIncompressible` (the pressure-Poisson solve `dvdt_incomp`), the `gradVel` projection and `solveShifting` with no wall: skipped today when gravity is on (`_RESTORE_PS_SHIFT='auto'`), so a gravity-free analytic scene would run it wall-blind. Steps: (B0, first, 15 minutes) refuse it loudly when a boundary provider is present and the shift would run; (B1) `solveIncompressible` takes `wall=` through the same modules the omni solve uses (`computeAlpha(wall=)`, `_divergence(wall=)`, `_pressureAccel(wall=)`, `wallDensity` in `rho*`), the `gradVel` operator gets the wall term `sum_b (f_b - f_i) G_b` of `wallDivergence`, the surface mask stays; (B2) the `'delta'` mode (`solveShifting`) is wall-aware already through the shifting modules, check it; (B3) lift the refusal and let `_RESTORE_PS_SHIFT='auto'` keep its meaning. Oracle: `DFSPH2D(densityShift=True)` (VD+PS with the wall in the density solve). Gate: a closed gravity-free analytic box (TGV-like decay in a walled box, no drift) and `hydrostaticColumn` with a body force on analytic walls; the existing particle-wall results unchanged (bitwise fingerprint of short runs).

### C. `iisph` and `dfsphReference` (`schemes/dfsphReference.py`: `dfsphReference_step`, `iisph_step`)

Both build on `applyConsistentCoupling` and the Akinci boundary band (boundary particles as static fluid), so the wall enters the same places as in the omni loop: `computeDensities`, `computeAlpha`, the source (`wallDivergence`), the pressure acceleration (`wallPressureAccelerationOmni` or the δ⁺-form variant: pick per scheme, the density factors differ) and the final pressure force. No `DFSPH2D` oracle exists for them (it is the omniSPH loop), so the checks are internal: term-level equality with the omni loop where the formulas coincide, a still tank, the dam break against the omni analytic run, and the same scheme on boundary particles. `iisph` first (the simpler loop), then `dfsphReference`; lift each from `ANALYTIC_WALL_SCHEMES` only with its test (`tests/test_analyticRefusal.py` shrinks accordingly).

### D. δ⁺ sloshing against `DeltaSPH2D`, then retiring the solvers

1. Rerun the warpSPH δ⁺ analytic 7 s sloshing in the exact setup of `~/dev/curvatureBoundaries/results/deltasph/slosh_B_nx200_series.npz` (nx 200, C4 vs C2 as the repo had, probe, smoothing; its README `docs/deltasph-validation.md` s.6 states the setup; the stored series is the one full-run result of `DeltaSPH2D`: 4.3 / 3.8 / 7.5 kPa).
2. Compare like with like (10 ms Gaussian on both and on the measurement): impact times and peaks, kinetic energy, density range, frames. A difference is attributed (kernel, probe, shifting, the calibrated-lattice-free setup) before it is called a defect; the gate is "warpSPH δ⁺ is at least as close to the measurement as `DeltaSPH2D`, and the remaining difference is explained".
3. Retirement checklist (written, not executed, until the owner agrees): which files of the boundaries repo go (`sim/deltasph2d.py`, `sim/dfsph2d.py`, their kernels / graph step / cases / tests), which stay (the integral machinery, `scene/`, the oracle probes that need `DFSPH2D`: so the oracle must be frozen first, e.g. as stored term tables, or retired last), the tests that move to warpSPH, the `EXPERIMENTAL` flags to lift (all three analytic schemes) and the `ANALYTIC_WALL_SCHEMES` documentation.

### Housekeeping to fold in
OPEN_PROBLEMS 30 (the flaky test) before the next full-suite gate; the third sloshing impact overshoot and the n_h scaling of the peaks stay in OPEN_PROBLEMS 31 (a resolution study, nx 200 / 300, is the owner's expected remedy for the early arrival, a cheap addition to D); keep the oracle probes working (two broke silently today when `dambreak` gained the automatic calibration: add a one-line self-check, each prints its own equality at the start).

## Next

Incompressible / ACSPH consumers (hook table in warpSPHBoundaries' audit); moving bodies in a graph; 3D; the wall Hessian integral for `exactHessian`; wall particles of the `ParticleBoundary` provider as a
cross-representation convergence test inside warpSPH.

**Direction (user, 2026-10-09):** the analytic-boundary support moves into the front end's solver schemes, aligned with all capabilities there (δ⁺ first, then DFSPH / ACSPH / ...), and the solver code in warpSPHBoundaries (`DeltaSPH2D`, `DFSPH2D`) is retired afterwards. For δ⁺ that waits on the boundaries repo's δ⁺ giving usable results: its validation is internal and was not run on full cases such as sloshing. First full-run check: `sloshingTank --wallRepresentation analytic` (δ⁺, Wendland C2, Sensor 1 = fluid probe) against the particle-wall run, 7 s, `examples/sloshingTank/output/analytic_ab_2026-10-09`.

Survey of what the boundaries repo's solvers do, term by term against warpSPH's modules, and the proposed module-level shape: [ANALYTIC_BOUNDARIES_PORT_SURVEY.md](ANALYTIC_BOUNDARIES_PORT_SURVEY.md).
