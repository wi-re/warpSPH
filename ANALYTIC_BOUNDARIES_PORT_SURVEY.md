# Analytic boundaries — survey of the boundaries repo's solvers, for moving them into warpSPH's modules

> Part of [ANALYTIC_BOUNDARIES_PLAN.md](ANALYTIC_BOUNDARIES_PLAN.md) (direction of 2026-10-09: the solver code in `warpSPHBoundaries` is retired once its schemes live in warpSPH). Row in [PLANS.md](PLANS.md).
> Survey date 2026-10-09. Source: `~/dev/curvatureBoundaries` (package `warpSPHBoundaries`, `src/warpSPHBoundaries/sim/`). Read-only; nothing here is changed yet.

**Read in full:** `DeltaSPH2D.rhs` (`sim/deltasph2d.py:372-596`), `DFSPHConfig` and `DFSPH2D.step`, `DeltaSPHConfig`, the porting notes and upstreaming plan, the warpSPH side (`modules/analyticBoundary/`, `schemes/deltaSPH.py` hooks, `boundary/provider.py`).
**Only skimmed (names and docstrings):** `DeltaSPH2D.shift` / `no_penetration` / `_project` / `settle`, `DFSPH2D._solve*` / `_solve_compact`, `wallmoments.py`, `wallclosure.py`, `projection.py`, `graphstep.py`. Anything below about those is marked *(unverified)*.

## 1. What the sim layer is

| file | lines | role |
|---|---|---|
| `sim/deltasph2d.py` | 1031 | `DeltaSPH2D` + a 60-field `DeltaSPHConfig`: rhs, shift, no-penetration, step, loads, packing tools |
| `sim/dfsph2d.py` | 1054 | `DFSPH2D` + a 45-field `DFSPHConfig`: summation density, divergence and density solves (Jacobi), a `compact` CG projection, shifting, friction, forces |
| `sim/{wallmoments,wallclosure,projection,fluidwarp,graphstep}.py` | ~1000 | no-slip moment closures, compact Laplacian projection, glue to warpSPH's fluid modules, whole-step CUDA graph |
| `sim/{solvers,cases,validation,probes,references}.py` | ~700 | a common `make_solver("delta"\|"dfsph")` description, cases (tank, wedge, Marrone, sloshing), probes |
| `scene/` (not sim) | ~4500 | the representations and exact integrals: stays in the package |

Two facts shape the port:

1. **`DeltaSPH2D` already delegates the fluid–fluid terms to warpSPH.** `fluidWarp=True` (default) routes continuity, fourtakas2019 DDT, the Antuono pressure force and the α-viscosity of fluid pairs through `sim/fluidwarp.py` onto warpSPH's own modules and a warpSPHCore Verlet list. What is inline in `rhs` is the **wall half** of each term, plus options the front end does not have.
2. **warpSPH already hosts a first version of the δ⁺ wall half** (the 0.6.0 series): the math sits in `modules/analyticBoundary/` and `_deltaSPH_rhs` calls it behind `if wall is not None`. That is an *adjacent* structure (wall terms in a side package, glue in the scheme), not yet the target of "the analytic boundary is part of each module".

## 2. δ⁺ term by term

`wall` = what a module has to add for the wall; `now` = where that lives in warpSPH today.

| stage | boundaries-repo `rhs` | warpSPH module (fluid part, exists) | wall part now | wall part in the module (target) | gap to close first |
|---|---|---|---|---|---|
| wall state | `_wall_state`: λ, ∇λ per body, hydrostatic term `A = ∫P_b∇W` with `P_b = P_i + ρ(g−a_w)·(x'−x_i)`, cached per position set | — | `evaluateWall` → `WallState` | one object the modules take (see §5) | none |
| density sum / EOS | EOS only (continuity-based) | `modules/eos` | unchanged | none | none |
| free-surface detection | `_detect_surface`, wall as exact cover vector + cone area (`cover.py`, `cone_area.py`) | `modules/surfaceDetection` (`barecascoDetection`) | `detectFreeSurfaceAnalytic` (split Barecasco passes + wall between) | `detectFreeSurface(..., wall=)` in the same module | thresholds were tuned on particle walls; check on Marrone |
| continuity | `kinematic(v)`: `2ρ (v−v_b)·n̂ |G|` per body (free-slip mirror), re-evaluated with the mean velocity for time-centred continuity | `modules/momentum/inconsistent` | `wallContinuity`; wall flux also in `drift_rates` | term of the continuity module | none (finding: without it in `drift_rates` the dam break diverges) |
| density diffusion | fluid–fluid only | `modules/deltaSPH/densityDiffusion` | none needed | none | none |
| pressure force | `-(1/ρ)[(P⁺+sP)G + A_eff]`, `A_eff = A − excess(A,G,P⁺)·G` (clamp ≥ 0), optional `pressureConsistent` (static-residual removal) | `modules/pressure/surfaceAware` | `wallPressureAcceleration` (clamp) | pressure module takes `wall` | `pressureConsistent`, `wallPressureViscous`, `bodyForceAtWall` are options not ported |
| viscosity (fluid pairs) | α form via warpSPH; Morris path inline | `modules/deltaSPH/velocityDissipation` | — | — | Morris + wall: refused in warpSPH (`viscousPrefactor` raises) |
| viscosity (wall) | six forms: `laplacian` (free-slip, exact `∫∇²W`), `pairwise` (polar quadrature), `noslip`, `noslipMirror`, `noslipMoment` (Morris), `noslipCurv` (labelled experimental and empirical) | same module | `wallViscousAcceleration` (free-slip exact Laplacian, and a no-slip variant) | velocity-dissipation module takes `wall` | decide which forms survive; `noslipCurv` should not be ported; `noslipMoment` needs `wallmoments.py` + `wallclosure.py` *(unverified)* |
| gravity, bodyForce | `+ g`, `+ bodyForce` (kept out of the hydrostatic DDT, in the wall pressure) | `modules/gravity` | gravity only | forcing module | `bodyForce` + `pinned` band + periodic images are channel-flow options, absent in warpSPH's analytic path |
| shifting | `shift()`: Sun 2017 Eq. 7 + Sun 2019 surface treatment, wall tensile term exact (`tensile.py`) | `modules/shifting` (`deltaShift`, Michel 2022, implicit) | `wallShiftRaw`, `wallUChar`, `wallConcentrationGradient` | shifting modules take `wall` | warpSPH is *ahead* here (Michel, implicit); the C2-only restriction of the wall tensile kernel is the limit |
| no-penetration | `no_penetration()` impulse, `f(d) = 3 − 4 clip(½ + d/dp, ¼, 1)` | `modules/mdbc/wp_nopenshift` | `analyticNoPenShift` | mdbc module offers a provider-based variant | none |
| wall loads | `loadsAt` / `_load`: pressure + viscous per body, torque | `RigidBody.load` | `wallLoads` | none | viscous/shift/no-pen bookkeeping incomplete on both sides |
| integrator, dt | warpSPHIntegrators by name; Sun-2017 dt | same | same | none | none |
| pressure solver `projection` | CG on the compact Laplacian, shared with `DFSPH2D` | no counterpart in `δ⁺` | not hooked | belongs to the incompressible track (§3) | out of scope for δ⁺ |
| whole-step graph | `graphstep.py` | `schemes/deltaSPH.py` `_graphedRHS` | done for static walls | — | moving bodies |

**Not in warpSPH, present in `DeltaSPH2D`, and worth deciding on:** the particle flavour of the provider (`wallParticleSpacing`: wall as a lattice of particles summed pairwise, the cross-representation convergence test), `settle` / `pack` (body-fitted initial packing — the porting notes call this *open* for analytic obstacles: a regular lattice is inconsistent with a sloped wall at first order), `pinned` bands, periodic domains, exact momentum/torque booking of viscous/shift/no-pen terms.

## 3. DFSPH: which warpSPH scheme it corresponds to

`DFSPH2D` is **omniSPH on analytic walls** (its docs: validated against the compiled omniSPH on tank, dam break, rotating obstacle; "omniSPH conventions": summation density with the wall continuum λ, a divergence solve of 3-4 Jacobi iterations, a density solve of 4-256, `p_b ≥ 0` clamp, ω = 0.5, XSPH, boundary friction). warpSPH's counterpart is **`omniIncompressible`** ("direct port of the omniSPH incompressible solver loop"), which already calls `wallPressureExtrapolation` per Jacobi iterate (`modules/incompressible/wallPressure.py`: the particle analogue of `DFSPH2D`'s per-iterate `p_b`).

**This is not the scheme the sloshing runner's `--scheme dfsph` runs.** That preset is `divergenceFree` (VD+PS: divergence-free projection plus a density-driven *position shift*). `DFSPH2D` has `densityShift` (Cornelis VD+PS) as an option, so the idea exists there, but the default path and everything validated is the omniSPH loop. So "DFSPH on analytic walls" is two different jobs:

| target | boundaries-repo source | warpSPH hooks to replace (current) | notes |
|---|---|---|---|
| `omniIncompressible` (+ `iisph`, `dfsphReference`, which share `wallPressure.py` / `consistent.py`) | `DFSPH2D._prepare` (wall λ, ∇λ, first moments), `_boundary_accel` (hydrostatic `p_b`, clamp), `_wall_div` (flux into the divergence source), `_a1_part`, `_alpha` (wall in the diagonal) | `applyConsistentCoupling` (Akinci band mass), `wallPressureExtrapolation` per iterate, `computeAlpha(includeBoundaryReaction)`, `computeBoundaryVelocities` | the wall appears in **four** places: density, diagonal α, source term (flux), pressure acceleration. All four are λ/∇λ/moment integrals the provider returns. |
| `divergenceFree` (VD+PS) | `DFSPH2D` with `densityShift=True` *(unverified how far it was validated)* | `computeDensities`, `computeMdbc*`, `computeBoundaryVelocities`, `wallMoment` kernels (`modules/incompressible/wp_wallMoment.py`) | the wall enters through the mDBC-corrected density and the wall-moment kernels, not through a per-iterate extrapolation; needs its own pass |
| the `compact` CG projection | `sim/projection.py`, `_solve_compact` | no counterpart (`krylov.py` is a Krylov solve of the IISPH-style operator) *(unverified)* | a different pressure solve; not a port, a new warpSPH module |

The boundaries repo's own finding: `dfsph` free-surface behaviour carries a ~10 % pressure error near the surface (the Dirichlet-surface/density-solve combination), so a still-water/sloshing comparison of the omni loop is expected to show that regardless of walls.

## 4. What is messy, concretely (why it is a rewrite, not a copy)

* **Everything is one 1000-line class with a 60-field config.** `rhs` interleaves fluid pairs, wall terms, six viscosity forms, the projection pressure solver, `pinned`, `pressureConsistent`, load bookkeeping.
* **Three code paths per term:** fused provider (`WallAggregate`), per-body `scene.*` oracle, and the `near`-index list with `index_add_` (host syncs; the eager path). warpSPH needs only the first.
* **Torch on float64 around Warp kernels,** `torch.nonzero`/host reads in places; warpSPH's modules are Warp-first, dtype-generic, graph-capturable.
* **Config fields that are experiments:** `noslipCurv` (empirical, "wrong on c…"), `cornerWedgeTables` ("EXPERIMENTAL"), `shiftMachFrame`, `wallFrictionLever`. These are research switches and should not be ported as fields.
* **Fluid parts are duplicated:** `rhs` still carries a non-`fluidWarp` torch copy of every fluid–fluid term (`cfg.fluidWarp=False`) as an oracle.
* **Gains worth keeping:** the exact wall integrals (all in `scene/`, not here), `pressureConsistent`, the complement-moment no-slip closure, momentum bookkeeping, `settle`/`pack`.

## 5. Proposed shape in warpSPH (for discussion; nothing built)

1. **One wall object, passed to the modules.** `WallState` (λ, G per body, hydrostatic `A`, Cov, cover, lap, tens) already exists. Each affected module gets an optional `wall=None` argument: `detectFreeSurface`, the continuity/momentum term, `computePressureForce*`, `computeVelocityDiffusion`, `solveShifting` (`computeDeltaShift`, Michel, implicit), `computeMdbcNoPenShift` (provider variant). With `wall=None` the function is the current one, bit for bit (the existing guard, `boundaryProvider is not None`, becomes this argument). The wall term is added in the module that owns the fluid–fluid term, next to it, using its coefficients (`fac`, `h`, the Antuono switch, kernel) so the two cannot drift apart.
2. **Incompressible modules take the same object.** `applyConsistentCoupling`, `computeAlpha`, `wallPressureExtrapolation`, the divergence source and `computePressureAccelIISPH` get the analytic contribution in place of the Akinci band/boundary particles. Start with `omniIncompressible` (it maps 1:1 to `DFSPH2D`), then decide on `divergenceFree`.
3. **Drop, don't port:** `noslipCurv`, `cornerWedgeTables`, the torch oracle copies, the `near`-index paths, the polar-quadrature `pairwise` form (needs a decision: it is the warpSPH-faithful free-slip form, the exact Laplacian one is the default there).
4. **Validation gate (the user's condition for retiring `DeltaSPH2D`):** full-length cases, not term tests: sloshing 7 s (running now, §6), Marrone dam break (to the t* of the stored series), English wedge, tank. Compare the warpSPH-hosted analytic run with (a) warpSPH's particle-wall run, (b) the boundaries repo's stored series (`results/deltasph/slosh_B_nx200_series.npz`, `slosh_warpSPH_series.npz`, `dambreak_{A,B}_nx67_series.npz`).

## 6. Evidence so far

* `sloshingTank` 7 s, nx = 200, `examples/sloshingTank/output/analytic_ab_2026-10-09` (2026-10-09; videos and Sensor-1 plots there, all three runs finished, no velocity alarm):
  * δ⁺, particle walls (Wendland4): smoothed Sensor-1 peaks 5060 / 5309 / 5448 Pa at t = 2.343 / 4.005 / 5.557 s; KE max 0.0340; density [0.82, 1.33].
  * δ⁺, **analytic walls** (new `--wallRepresentation analytic`, Wendland2 as the wall shifting requires, Sensor 1 = Gaussian fluid probe): 4746 / 5224 / 4959 Pa at 2.351 / 4.016 / 5.623 s; KE max 0.0288 (rms 14 % from the particle run); density [0.75, 1.34]; extra short probe spikes between impacts (t ~ 3.0, 4.7, 6.4 s; isolated near-wall particles in the frames, not checked further). Kernel, wall treatment and sensor definition all differ from the particle run, so this is not a single-factor A/B. Wall time not comparable (a parallel GPU job ran during it).
  * `divergenceFree` (DFSPH, particle walls): finished (15 555 steps, 237 s) but poor: smoothed impacts ~1.8 / 4.6 / 3.8 kPa as broken trains of spikes, density down to 0.33, sparse stretched fluid layer in the frames. Same signature as the recorded nx = 200 DFSPH (free-surface de-densification grows with resolution, DFSPH_IMPROVEMENT_PLAN Part 23; raw peak 16.2 kPa both times), not a regression. GPU utilisation 22-29 % with one CPU core pegged (host-bound), see SMALL_PROBLEM_PERFORMANCE.md.
  * Boundaries repo's own nx = 200 7 s δ⁺ (Wendland4, Sun-2017 shifting): 4.9 / 5.2 / 5.3 kPa.
* Boundaries repo, nx = 200, 7 s, C4, Sun-2017 shifting: impacts 4.9 / 5.2 / 5.3 kPa (Gaussian probe) vs measured 1.8 / 2.3 / 1.2 kPa; KE overlaps warpSPH through 7 s; 5439 s against warpSPH's 2050 s (its HANDOFF).

## 6b. Port status (2026-10-09)

**Convention (all modules that take a wall):** `wall=None` means "resolve it": `resolveWall(state, config, schemeConfig, adjacency, wall)` (`modules/analyticBoundary/wallTerms.py`) returns the argument if given, else the wall of `schemeConfig.boundaryProvider` at these positions (shared through the `evaluateWall` cache), else `None` (boundary particles). `None` never means "skip the wall on a scene that has analytic bodies". A passed wall is valid for the input positions only: iterating modules (`computeDeltaShift`, `computeMichelShift`) use it in the first iteration and re-resolve after.

**Slice 1+2 (done, bitwise):** the wall term moved into the module that owns the fluid–fluid term: `computeMomentum` (continuity, `wallContinuity`), `computePressureForceSurfaceAware` (`wallPressureAcceleration`, with `antuonoSwitch` now an exported helper of that module), `computeVelocityDiffusion` (`wallViscousAcceleration`), `detectFreeSurface` (-> `detectFreeSurfaceAnalytic`), `computeDeltaShift` / `computeMichelShift` / `computeImplicitShift` (their provider lookups replaced by `resolveWall`). `_deltaSPH_rhs` now calls the modules with `wall=wall`, `WeaklyCompressibleSystem.drift_rates` lost its private wall block. Verification: a fingerprint of 300 / 200 / 300 / 200 steps of dam break (analytic), dam break + circular obstacle (analytic), sloshing (analytic, C2) and dam break (particle walls): positions, velocities, densities **bitwise identical** to the tree before the change; analytic tests, `check_imports.py` pass.

**A trap found on the way:** the first version of `computeMomentum(wall=None)` resolved the wall while `drift_rates` still added the wall flux itself: a double count (dam break differed at 10 m/s level). The fingerprint caught it. Any caller that adds a wall term next to a module that now adds it is a double count: when moving a term into a module, move *every* caller's private copy at the same time (grep the module's callers).

**Slice 4a (omniIncompressible, EXPERIMENTAL, 2026-10-09, not working yet):** the wall terms of the omni loop are in (density, alpha, divergence / source, pressure acceleration, Shepard surface source), the scheme is on `runner.ANALYTIC_WALL_SCHEMES` and flagged experimental; hydrostatic tank at rest: unstable at the first rows. Findings and the plan (term-by-term against `DFSPH2D`, calibrated lattice) are OPEN_PROBLEMS 31. **Dam break (omniSPH's own case): the scalars (front, mean height, max |v|) are within 1.2-1.7x of omniSPH with its wall convention, but the frames are NOT physical (hollow arch at t = 0.31, owner's review): the omni analytic port does not work yet; `docs/analytic_boundaries/omni_dambreak_2026-10-09/` (conclusion retracted there).**

**Next slices:** (3) the no-penetration impulse and the loads (`analyticNoPenShift`, `wallLoads`) behind the same convention; (4) the incompressible track, `omniIncompressible` first: per-iterate wall pressure, wall flux in the divergence source, wall in the diagonal `alpha`, wall density, against `DFSPH2D`; (5) `divergenceFree`.

## 7. Decisions (owner, 2026-10-09)

1. **DFSPH order:** `omniIncompressible` first (the raw C++ omniSPH works well on dam breaks with analytic boundaries, so it is the reference), then `divergenceFree`. Targets: sloshing and the dam breaks.
2. **Wall-viscosity forms:** all survive in some form if any case uses them (not a fixed list: the form a case needs decides). Experimental switches are kept only where a case uses them, as a named variant, not as a free config field.
3. **Body-fitted packing (`settle` / `pack`):** not part of this port.
4. **Channel-flow options:** see below; this follows from the rule in 2: ported when a case in the front end uses them.

### The channel-flow options, specifically

All from `DeltaSPHConfig` / `DFSPHConfig`; used by the periodic-channel, Stokes-array, Couette, corner and cylinder studies and notebooks 02-05, 09 (none by the sloshing / dam-break / wedge cases).

| option | what it does | used by (boundaries repo) | nearest thing in warpSPH |
|---|---|---|---|
| `bodyForce` | uniform acceleration in the momentum equation only (the pressure-gradient driver of a periodic channel). Unlike gravity it is not in the hydrostatic part of the density diffusion | `periodic_channel`, `periodic_cylinder_array`, `corner_rig`, `dfsph_viscous`, notebooks 02, 04 | `GravityType.Directional` (also enters the DDT hydrostatic term); `kolmogorov` cases (body-force driven) |
| `bodyForceAtWall` | whether that force also enters the wall pressure condition `dp/dn = ρ(g + f − a_w)·n` (default on) | `periodic_cylinder_array` | — (follows the gravity treatment) |
| `periodic` | minimum-image pair geometry for fluid pairs; the wall integrals take their own periodic image shifts (cylinder array) | most studies, notebooks 01-05 | fluid side exists (periodic domains); the **wall side (image shifts in the provider) is the gap** |
| `pinned` | a band of fluid particles whose velocity is prescribed (free stream of a flow past a body in a periodic box): acceleration zeroed, velocity reset to the stream after the step, no shift inside | `cylinder_wake`, notebook 09 | `meanFlowForcingBC` (`movingObstacle`, `drivenSquare`) and the freestream forcing band of `channelFlow` |
| `pressureConsistent` | removes the force a *uniform* pressure would exert through the lattice-vs-wall inconsistency (cut lattice at a curved wall); makes the load independent of the pressure level | `cylinder_wake`, `periodic_cylinder_array`, `corner_rig`, `uniform_pressure_wall` | none (particle walls have the same residual, never corrected) |
| `wallPressureViscous` | the wall pressure condition carries the viscous term, `∇p = ρ(g + f − a_w) + ρν∇²u` (matters where pressure is viscous-dominated, Stokes flows) | `periodic_cylinder_array`, `corner_rig` | none |
| `fluidViscosity="morris"` + `noslipMoment`/`noslipMirror`/`complementMoments`/`morrisCalibration` | physical viscosity (not the α form) with a no-slip wall closure; needed for Couette, Stokes arrays, cylinder flows | studies above, notebooks 02-05, 09 | Morris exists on the fluid side; the wall term does not (refused in `viscousPrefactor`) |

So: for sloshing and the dam breaks none of these are needed, they are the *viscous, low-Reynolds, curved-wall* group. Inside warpSPH the cases that could use them are `channelFlow`, `movingObstacle`, `drivenSquare`, `kolmogorov*`, `shearBox`, `lidDrivenCavity` once those get analytic walls; the driving and freestream parts already exist there as forcing bands.

### Sequencing (owner, 2026-10-09)
The table above waits until the free-surface cases (sloshing, dam breaks) work with both δ⁺ on analytic walls and the DFSPH-style solvers (`omniIncompressible`, then `divergenceFree`). The channel paths are the next stage after that; which warpSPH cases take them (`channelFlow`, `movingObstacle`, `drivenSquare`, `kolmogorov*`, ...) is decided then.
