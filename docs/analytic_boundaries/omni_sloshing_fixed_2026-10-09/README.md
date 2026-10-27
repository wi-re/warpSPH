# SPHERIC TC10 sloshing, 7 s, nx = 200: omniIncompressible on analytic walls after the divergence-solve fix — 2026-10-09

Runs: `run_sloshingTank.py --scheme omni --wallRepresentation analytic --nh {2.57, 4} --tLimit 7` (omni preset, Wendland C2, calibrated lattice automatic, divergence solve without the wall and with the matching diagonal, commit 91c9da8); boundary-particle omni and the delta+ analytic run (n_h 4, C2) from
earlier the same day for reference. `scripts/plot_sloshingOverlay.py` (now also plots the **measured record smoothed the same way** as the runs, 10 ms Gaussian).

**Compare like with like.** Earlier READMEs and messages of the day compared the 10 ms-smoothed simulated peaks with the *raw* measured peaks (3.7 / 3.9 / 2.7 kPa): a sharp raw peak and a smoothed one are not comparable. The measured record smoothed the same way peaks at 1800 Pa @ 2.390 s,
2322 Pa @ 4.066 s, 1171 Pa @ 5.697 s.

| smoothed Sensor-1 peak [Pa] @ t [s] | impact 1 | impact 2 | impact 3 | wall time |
|---|---|---|---|---|
| measured, same smoothing | 1800 @ 2.390 | 2322 @ 4.066 | 1171 @ 5.697 | |
| **omni, analytic, n_h 2.57** | 1897 @ 2.370 (+5 %, -20 ms) | 2219 @ 4.056 (-4 %, -10 ms) | 3082 @ 5.645 (+163 %, -52 ms) | 248 s (12 637 steps) |
| omni, analytic, n_h 4 | 2884 @ 2.378 (+60 %, -12 ms) | 4071 @ 4.047 (+75 %, -19 ms) | 4570 @ 5.658 (+290 %, -39 ms) | 224 s (11 625 steps) |
| omni, boundary particles, n_h 2.57 | 1784 @ 2.374 (-1 %, -16 ms) | 1711 @ 4.045 (-26 %, -21 ms) | 2086 @ 5.652 (+78 %, -45 ms) | 233 s |
| delta+, analytic, n_h 4, C2 | 4549 @ 2.351 (+153 %, -39 ms) | 5041 @ 4.016 (+117 %, -50 ms) | 4835 @ 5.623 (+313 %, -74 ms) | |

* All runs finish, no velocity alarm. The kinetic energy of all four agrees through 7 s. The first two impacts of the n_h 2.57 analytic omni run are within 5 % of the measurement; the third overshoots 2.6x; every run is 10-55 ms early.
* The peak depends strongly on the support: n_h 4 gives 1.5-1.8x the n_h 2.57 peaks, and the n_h 4 omni run is close to the delta+ run at the same n_h (not a coincidence of the scheme: the same support and the same wall). The n_h 4 run was unstable before the divergence-solve fix (OPEN_PROBLEMS 31).
* Sensor 1 is the Gaussian fluid probe (2 cm) for the analytic runs, the nearest wall particle for the particle run: not the same estimator.
* Frames (`frames_omni_analytic_nh2.57.png`, `frames_omni_analytic_nh4.png`, t = 2.35, 2.6, 4.05, 4.3, 5.65, 6.5 s): wave run-up with a jet and spray at the right wall, a broken wave over the tank, a high run-up and its bore, the return wave at the left wall: a compact body, a smooth surface, no hollow arch, no stuck particle. At n_h 4 the wall jets are thin, particle-wide strings.
