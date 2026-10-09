# Compact CG projection (stage 2, A1) — 2026-10-09

Evidence for [ANALYTIC_BOUNDARIES_PLAN.md](../../../ANALYTIC_BOUNDARIES_PLAN.md) "Stage 2", item A1. Code: `modules/incompressible/compactProjection.py` (the compact Morris / Brookshaw Laplacian, the CG, `solveCompactProjection`),
`wallPressureAccelerationOmni(clampPressure=)`, the switch `schemeConfig.projection = 'jacobi' (default) | 'compact'` and `densitySolve` in `omniIncompressible._solve` / the step and in `divergenceFree`'s step; case params
`projection`, `projectionTol`, `densitySolve` (`sloshingTank`, `dambreak`), CLI `--projection compact --noDensitySolve`. Tests `tests/test_compactProjection.py`. Port of `warpSPHBoundaries/sim/projection.py` + `DFSPH2D._solve_compact`
(robin wall, symmetric gradient, free-surface Dirichlet, closed-domain mean removal, gauges, wall-flux factor).

## Equality with the oracle (`scripts/probe_omniCompactOracle.py`, identical dam-break lattice, tol 1e-10)

| case | CG iterations (oracle / warpSPH) | pressure rel. rms | total acceleration rel. rms |
|---|---|---|---|
| robin wall + free-surface Dirichlet | 150 / 150 | 5.8e-5 | 5.6e-6 |
| wall flux x2 | 150 / 150 | 8.0e-5 | 1.0e-5 |
| robin, no Dirichlet | 210 / 210 | 6.6e-5 | 1.8e-5 |

(float32 against float64; `cal = 0.958` of the lattice is computed by `morrisCalibration` from the actual spacing and handed to the oracle.)

## What it does for the schemes

* **Inviscid Taylor-Green vortex** (periodic, no walls; `scripts/probe_tgvCompactProjection.py`, nx 48, `divergenceFree`; the energy of a flow that should lose none), KE / KE0 at t = 2: jacobi 0.9914, **compact 0.9982** (4.8x less loss), compact without the density solve 0.9935 (the warpSPH VD+PS shift of `finalize` instead of DFSPH2D's `fixed` shift: A3). Frames `tgv_t3_jacobi_over_compact.png`: regular lattices, both stable.
* **Does it lift the relaxation limit? No, not where it matters** (`scripts/probe_omniOmegaLimit.py`, still analytic tank, 150 steps, `omega_limit.log`): at n_h 4, omega 0.5 diverges with either divergence stage; omega 0.4 blows up with the Jacobi divergence stage (9e5 m/s) and stays bounded (1.7 m/s) with the compact one. So part of the limit sits in the three divergence sweeps (the compact stage removes that part), the rest in the clamped density solve, which stays Jacobi: the p >= 0 clamp makes it nonlinear and a Krylov solver there failed already (DFSPH_IMPROVEMENT_PLAN Part 43). At n_h 2.57 all of omega 0.3 - 0.5 are stable with the Jacobi stage.
* **Free surfaces: not an improvement, as in the boundaries repo.** Still sloshing tank with the compact divergence stage: peak |v| 1.0 - 2.4 m/s (Jacobi 0.2 - 0.5), rising at n_h 4. The oracle's own free-surface column study behaves the same: `scripts/studies/dfsph_freesurface.py column --variant compact` (robin wall, free surface): near-wall pressure error 2.9, mean-height drift -22 dx, vmax 7.3; its `mirror` and `compact-diff` variants stay bounded in the short run but have a bulk pressure error of 0.58 (mirror) and the difference variant crashes (NaN) in the study. The oracle's own notes keep free surfaces on the omniSPH-style path ("the mirror projection has the right hydrostatic pressure but surface noise / splash clusters", item 6 open), so the `mirror` wall and the `difference` / `renormalised` gradients are **not ported** here: they are an unfinished research branch of the oracle. The compact projection is the closed / periodic-flow tool.
* **Cost** (eager torch CG, sloshing tank N ~ 4200): 65 CG iterations per step at tol 1e-8, 4.8 s for 100 steps against 5.2 s for the Jacobi stage, so the Warp / graph CG of the oracle is not needed at this size (revisit for larger N).
* Boundary-particle scenes refuse it (`NotImplementedError`); only analytic walls and periodic / closed domains (no walls) are supported.

Next within A: A2 (closed-domain presets / gauges as scheme options), A3 (the `fixed` / Fickian shift: needed for the closed preset), A4 - A6 (see the plan).

## A2 / A3: closed preset, gauges, Fickian shift

* `modules/shifting/fickian.py`: `particleShift = 'fixed' | 'fickian'` (schemeConfig `shifting`, `shiftA`): `dx = -D grad C`, `grad C = sum V_j grad W + wall G`, `D = shiftA h_s^2` or `shiftA h_s |v| dt`, tangential part only in the free-surface layer (surface particles and neighbours), cap 0.25 spacing; equal to `DFSPH2D`'s shift to 3e-5 (`scripts/probe_omniFickianOracle.py`; with the layer; without it 2 % from the cap, which the oracle takes as 0.25 PACKING h and warpSPH as 0.25 dx). The Verlet list carries pairs beyond the support: the layer must count only pairs inside it (that was the whole difference between 20 % and 3e-5). It replaces the VD+PS shift of `IncompressibleSystem.finalize` (switched off when the Fickian shift is on) and is refused with boundary particles.
* `closedPreset` (`True` | `'auto'` | default off) fills `projection='compact'`, `densitySolve=False`, `particleShift='fixed'`, `shiftA 0.5`, `divergenceGauge='min'` where they are still at their default (`resolveClosedPreset`). **Off by default here** (DFSPH2D turns it on for fully periodic domains): the shipped TGV would change behaviour.
* **The preset does not carry over to warpSPH's TGV.** Inviscid TGV nx 48: compact + density solve + VD+PS 0.9982 at t = 2 (best), compact, no density solve, VD+PS 0.9935, the full preset (fixed shift, no VD+PS) 0.9543 and max |v| 1.6. Isolated at t = 1: compact without density solve and without any shift 0.9960 (VD+PS on), the same with the `fixed` or `fickian` shift (VD+PS off) 0.984. The Fickian shift is a weak regulariser at warpSPH's time step (dt ~ 1e-3, thousands of steps; DFSPH2D's `fixed` D is per step) and removing VD+PS costs energy. The preset stays opt-in until a closed walled case (the channel group) measures it; the TGV recipe is `projection='compact'` with the density solve and the shipped VD+PS shift.
* A4 (free-surface Dirichlet of the projection, `rho < surfaceRho`) came with A1; it is the oracle's rule, not the detector's.
