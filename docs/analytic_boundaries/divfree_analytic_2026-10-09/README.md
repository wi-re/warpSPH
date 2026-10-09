# `divergenceFree` (DFSPH) on analytic walls, and the DFSPH2D extras (MLS wall pressure, XSPH, boundary friction) — 2026-10-09

Evidence for [ANALYTIC_BOUNDARIES_PORT_SURVEY.md](../../../ANALYTIC_BOUNDARIES_PORT_SURVEY.md) §6c. Code state: commit 4f871c6 (the extras) plus the `divergenceFree` hook-up of the next commit
(`schemes/divergenceFree.py`: analytic no-penetration, friction, XSPH; the runner / `sloshingTank` / `dambreak` allow the scheme). Raw output stays in `results/divfree_analytic_2026-10-09/` (gitignored).

## The extras (omniIncompressible and divergenceFree)

| piece | where | check against `DFSPH2D` (identical lattice, float32 vs float64) |
|---|---|---|
| MLS wall pressure (`wallPressure='linear'`, option `analyticWallPressure='mls'`) | `modules/analyticBoundary/wallPressureMLS.py` (the anchored W-weighted fit, per Jacobi iterate), `wallPressureAccelerationOmni(gradient=)` | fit gradient 1e-6, wall force 3e-5 relative (`scripts/probe_omniMlsOracle.py`) |
| boundary friction (`boundaryFriction`) | `modules/xsph/xsph.py: computeBoundaryFriction` (tangential drag by `min(c lambda, 1)`; moving walls through the provider's first moment) | 2e-5, static, rotating and translating wall (`scripts/probe_omniFiltersOracle.py`) |
| XSPH (`xsphCoefficient`) | `modules/xsph/xsph.py: computeXSPH` (its own module; fluid smoothing and the wall-particle drag) | 1.8 % rms: the weights are `m_j / rho_j` (Interpolate) where `DFSPH2D` uses `2 V_j / (rho_i + rho_j)` |

* The MLS closure is the **noisier** one, as in `DFSPH2D` itself: on a still column `linear` gives 1.5-2x the velocity and ~4x the kinetic energy of the hydrostatic closure (oracle: KE tail 3.1e-4 vs 8.7e-5, v 0.17 vs 0.11; warpSPH tank: peak 1.1 vs 0.49 m/s at n_h 2.57, 0.57 vs 0.26 at n_h 4, bounded, no growth in 300 steps). The hydrostatic closure stays the default.
* Boundary friction and XSPH (1e-4, 5e-3: the `DFSPH2D` defaults) move the omni dam break toward the boundaries repo's stored analytic run (`scripts/probe_omniMatchOriginal.py 0.6 calib 1e-4 5e-3`: rms position difference 4.50 vs 4.93 dx at t = 0.5, 8.2 vs 8.9 dx at 0.6, 1.32 vs 1.44 dx at 0.3): consistent with that run having used them, small next to the chaotic splash divergence. They do nothing on still water. Both are off by default (the validated runs of the day used neither).
* Found on the way: `probe_omniOracleTerms.py` and `probe_omniMatchOriginal.py` had silently broken when `dambreak` gained `calibratedLattice='auto'` (18 dx instead of 0.03 dx at t = 0.05); they now pin `calibratedLattice=False`.

## `divergenceFree` on analytic walls

What the step needed: nothing new for density, detector, velocity diffusion (they already take `wall=`) or the pressure solves (the omni `_solve` it calls); the analytic no-penetration shift in place of `computeMdbcNoPenShift` (the boundary-particle correction is kept for mixed scenes), the friction filter at the start of the step and XSPH folded into `dvdt`. The VD+PS position shift in `IncompressibleSystem.finalize` is skipped with gravity on (`_RESTORE_PS_SHIFT='auto'`) and is not wall-aware, so a gravity-free analytic scene would run it without a wall: not a case in scope.

| test | result |
|---|---|
| still tank, 300 steps | peak max \|v\| 0.39 (n_h 2.57), 0.23 (n_h 4); density max 1.00 |
| dam break to t = 1 (`scripts/probe_incompressibleAnalyticDambreak.py`), front / mean height / max \|v\| against compiled omniSPH | t = 0.2: 0.314/0.266/2.78 (omniSPH 0.303/0.263/2.82); 0.4: 0.934/0.088/3.71 (0.918/0.087/3.67); 0.6: 1.283/0.085/6.51 (1.297/0.079/6.97). `omniIncompressible` on the same setup: 0.313/0.266, 0.935/0.088, 1.283/0.085: the two schemes agree to 0.1 % on the dam break |
| English wedge, t = 1…4 s (`probe_omniEnglishWedge.py --scheme dfsph`) | KE 1.7e-3 → 1.2e-4 (omni: 1.9e-3 at t = 4), p / hydrostatic 1.03 with rms 4 % (omni 5.6 %); density 1.000 – 1.002 |
| sloshing 7 s, nx = 200 | below |

Sloshing, smoothed Sensor-1 peaks (10 ms Gaussian) against the measurement smoothed the same way (1800 @ 2.390 s, 2322 @ 4.066, 1171 @ 5.697):

| run | impact 1 | impact 2 | impact 3 | wall time |
|---|---|---|---|---|
| **DFSPH analytic, n_h 2.57** | 2251 @ 2.371 (+25 %, −19 ms) | 2660 @ 4.047 (+15 %, −19 ms) | 3287 @ 5.657 (+181 %, −40 ms) | 487 s (13 010 steps) |
| omni analytic, n_h 2.57 (earlier today) | 1897 @ 2.370 | 2219 @ 4.056 | 3082 @ 5.645 | 248 s |
| **DFSPH analytic, n_h 4** | 2971 @ 2.375 (+65 %, −15 ms) | 3450 @ 4.051 (+49 %, −15 ms) | 4413 @ 5.657 (+277 %, −40 ms) | 282 s (11 327 steps) |
| omni analytic, n_h 4 | 2884 @ 2.378 | 4071 @ 4.047 | 4570 @ 5.658 | 224 s |
| DFSPH, particle walls, n_h 4 (earlier today) | 1911 @ 2.787 (late) | 4631 @ 4.173 | 3779 @ 5.795 | 237 s |

* No alarm, no divergence; kinetic energy of all runs the same through 7 s. The two DFSPH analytic runs follow the omni analytic runs closely (same wall, same pressure solve); the particle-wall DFSPH run is late and shows the train of spikes after each impact (the free-surface de-densification of the boundary-particle DFSPH, DFSPH_IMPROVEMENT_PLAN Part 23) that the analytic runs do not have: the sensor trace after the impacts is smooth and follows the measurement's shape.
* Frames (`sloshing_frames_nh2.57.png`, t = 3.7, 4.9, 6.1, 6.8 s by the video clock; `dambreak_*`, `english_wedge_dfsph_t4.png`): a compact body, smooth surface, no holes; spray droplets at the impact walls only (a few isolated particles per jet), at the dam break t = 1.0 the spray is a set of small tight clumps in both `divergenceFree` and `omniIncompressible` (the same as the compiled omniSPH shows).
* Open, shared with the omni scheme (OPEN_PROBLEMS 31): the third impact overshoots (+181 % at n_h 2.57), every impact is 15-40 ms early, the peaks scale with n_h (the user: earlier arrival improves with resolution; the relaxation limit of the DFSPH-style Jacobi solver is a known limitation).
