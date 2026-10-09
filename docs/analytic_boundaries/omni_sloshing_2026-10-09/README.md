# SPHERIC TC10 sloshing, 7 s, nx = 200: omniIncompressible on analytic walls (calibrated lattice) — 2026-10-09

`examples/sloshingTank/run_sloshingTank.py --scheme omni --nh 2.57 [--wallRepresentation analytic] --tLimit 7` (omni preset: `semiImplicitEuler`, Wendland C2, n_h = 2.57 = omniSPH's h, dt <= 2e-3; the analytic run applies the
calibrated lattice automatically, `calibratedLattice='auto'`; Sensor 1 = Gaussian fluid probe within 2 cm for the analytic run, the nearest wall particle for the particle run). `scripts/plot_sloshingOverlay.py`.

| run | smoothed Sensor-1 peaks [Pa] at t [s] | wall time |
|---|---|---|
| measured (SPHERIC, raw record) | ~3700 @ 2.375, ~3900 @ 4.05, ~2700 @ 5.68 | |
| **omni, analytic walls** | 2319 @ 2.372, 3246 @ 4.039, 5106 @ 5.648 | 832 s (12 891 steps) |
| omni, boundary particles | 1784 @ 2.374, 1711 @ 4.045, 2086 @ 5.652 | 233 s (14 415 steps) |
| delta+, analytic walls (n_h 4, C2; `analytic_ab_2026-10-09`) | 4549 @ 2.351, 5041 @ 4.016, 4835 @ 5.623 | not comparable |

* Both omni runs finish (no divergence flag, no velocity alarm; analytic: max |v| 4.9, density [0.37, 1.05], spray). Impact times within 3-30 ms of the measurement (delta+: 24-57 ms early); the pressure after each impact follows the measured plateau;
  kinetic energy of all runs agrees through 7 s. The first two analytic peaks are 0.6x / 0.8x of the measured raw peaks, the third is 1.9x (2.7 kPa measured).
* Frames (`frames_omni_analytic_...png`, the analytic run at the impacts): wave run-up with a jet and spray at the right wall (2.35 s), a broken wave over the tank (2.6), a high run-up (4.05) and its bore (4.3), the return wave at the left wall (5.65),
  run-up again (6.5): a compact body, smooth surface, spray as small clusters, no hollow arches or stuck particles.
* Not a single-factor comparison: scheme (omni vs delta+), n_h (2.57 vs 4), kernel (C2 for both analytic runs), sensor definition (probe vs wall particle for the particle run). n_h = 4 with the omni analytic walls is unstable (OPEN_PROBLEMS 31, being investigated).

> **Superseded (2026-10-09, later):** these two runs were made before the divergence-solve fix (commit 91c9da8: the diagonal carried a wall the operator left out) and compared smoothed simulated peaks with the raw measured ones; the corrected runs and the like-with-like comparison are in `../omni_sloshing_fixed_2026-10-09/`.
