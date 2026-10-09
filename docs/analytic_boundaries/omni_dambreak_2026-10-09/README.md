# omniIncompressible dam break, analytic walls vs boundary particles vs compiled omniSPH — 2026-10-09

`scripts/probe_omniAnalyticDambreak.py --particles --offsets 0.5 0.05` (nx = 100, n_h = 2.57 = omniSPH's h = sqrt(20 V / pi), 2093 particles = omniSPH's 23 x 91 lattice, free slip,
`semiImplicitEuler`, dt <= 1e-3, **particle mass 0.9715 x rho0 dx^2** = omniSPH's V = pi r^2). Reference: the compiled omniSPH of the boundaries repo (`~/dev/curvatureBoundaries/.tmp/omni/cmp_dam.npz`:
particle centres on the block edges, walls one spacing outside the block) and that repo's `DFSPH2D` (hydrostatic wall closure), which agrees with it within 1 % (front) / 0.4 % (mean height).
Columns: front displacement [m] / mean height above the wall plane [m] / max |v| [m/s]. Analytic offset = the wall plane moved outward by that many dx (0.5: the first row one spacing from the wall,
omniSPH's convention; 0.05: the calibrated lattice of `DFSPH2D`, first row 0.55 dx).

| t | omniSPH | DFSPH2D | warpSPH omni, analytic, offset 0.5 | analytic, offset 0.05 | warpSPH omni, boundary particles |
|---|---|---|---|---|---|
| 0.1 | 0.081 / 0.366 / 1.93 | 0.082 / 0.367 / 1.99 | 0.083 / 0.371 / 1.79 | 0.088 / 0.369 / 1.77 | 0.099 / 0.368 / 1.93 |
| 0.2 | 0.303 / 0.263 / 2.82 | 0.299 / 0.264 / 2.94 | 0.298 / 0.267 / 2.81 | 0.310 / 0.267 / 2.79 | 0.323 / 0.264 / 2.82 |
| 0.3 | 0.587 / 0.150 / 3.37 | 0.580 / 0.151 / 3.43 | 0.587 / 0.153 / 3.47 | 0.586 / 0.154 / 3.57 | 0.609 / 0.150 / 3.46 |
| 0.4 | 0.918 / 0.087 / 3.67 | 0.913 / 0.088 / 3.78 | 0.929 / 0.088 / 3.80 | 0.925 / 0.089 / 3.77 | 0.954 / 0.086 / 3.84 |
| 0.5 | 1.289 / 0.061 / 4.07 | 1.279 / 0.061 / 4.16 | 1.287 / 0.061 / 5.15 | 1.284 / 0.062 / 5.79 | 1.285 / 0.060 / 9.02 |
| 0.6 | 1.297 / 0.079 / 6.97 | 1.298 / 0.079 / 6.27 | 1.288 / 0.084 / 6.55 | 1.286 / 0.084 / 6.95 | 1.285 / 0.086 / 6.70 |

Frames: `frames_analytic_omni_t0.05_to_0.55.png` (t = 0.05, 0.10, 0.20, 0.30, 0.40, 0.55): the column slumps from its base, the foot runs along the floor, at t = 0.55 the wave throws a jet up the right wall; one isolated
particle at the upper left at t = 0.55. Compact, evenly spaced body, as the reference video of the boundaries repo.

## What went wrong the first time, and the cause

The first run of the day (particle mass = rho0 dx^2, the warpSPH default) agreed with the table above to 1.2-1.7x in the scalars and was reported as "close to omniSPH" and "physical". **It was not**: at t = 0.31 the
fluid was a hollow arch (`FAILED_heavy_mass_frame_t0.31.png`), found by the owner on the frames, not by the numbers. Cause, found by running warpSPH and `DFSPH2D` on the identical lattice and comparing the terms
(`scripts/probe_omniOracleTerms.py`: density, alpha, both source terms, wall gradient, fluid and wall pressure acceleration all equal to <= 1e-5 relative once the wall pressure term uses omniSPH's form
`-(p/rho^2 + p/rho0^2) G - A rho/rho0` instead of the delta+ one, which differed by the density factors on the under-dense wall rows):

* omniSPH's particle area is `V = pi r^2` = 7.85e-5, 2.9 % **smaller** than the lattice cell (8.08e-5): its rest lattice measures rho = 0.983, the density solve (`p >= 0`) is inactive in the bulk and the pressure comes from the divergence solve.
* With mass = rho0 dx^2 the lattice measures 1.001, the density solve is active, and its relaxed Jacobi iteration is **unstable at the free surface** (the free-surface particles have rho = 0.6 and a large positive source it cannot satisfy):
  the pressure on a free-face particle grows with the iteration count (57 after 61 iterations in `DFSPH2D`, 252 after 79 in warpSPH, same particle, same terms), so which value is reached depends on when the stopping criterion fires. Both codes do this on the same state; it is a property of the omni iteration, not of the wall.
* The boundary-particle `omniIncompressible` of warpSPH has the same problem: with the heavy mass its front was 5x too fast at t = 0.1; with 0.9715 it is within 10-20 % (last column).

## Matching the boundaries repo's original analytic result (2026-10-09, later the same day)

The target is the boundaries repo's analytic run behind its `1_dambreak_N2k_omni_vs_analytic.mp4` (`.tmp/omni/runs/F_surf_r5.npz`: `DFSPH2D`, exact edge integrals, **calibrated lattice**), per particle (the three codes start from the identical 2093-particle lattice, snapshots index-aligned). `scripts/probe_omniMatchOriginal.py 1.0 calib` reproduces its setup in warpSPH:
omniSPH's lattice, `lattice_calibration` (particle mass 7.967e-5, wall mass 0.986, the wall plane 0.55 dx from the first row), h = sqrt(20) r, hydrostatic closure, and **the wall not in the divergence solve** (omniSPH's convention, `DFSPH2D`'s default for static walls; the port had it in both solves, now `schemeConfig.analyticWallInDivergence`, default False).

RMS position error [dx = 0.0088 m] (and velocity error [m/s]); the last column is the boundaries repo's own analytic run against the compiled omniSPH, for scale:

| t | warpSPH vs boundaries-repo analytic | warpSPH vs compiled omniSPH | boundaries-repo analytic vs compiled omniSPH |
|---|---|---|---|
| 0.05 | 0.028 (0.014) | 0.386 | 0.394 (0.106) |
| 0.10 | 0.059 (0.026) | 0.653 | 0.664 (0.079) |
| 0.15 | 0.199 (0.067) | 0.886 | 0.912 (0.111) |
| 0.20 | 0.453 (0.122) | 1.160 | 1.227 (0.143) |
| 0.30 | 1.414 (0.235) | 1.882 | 2.046 (0.232) |
| 0.40 | 2.908 (0.255) | 3.179 | 3.332 (0.246) |
| 0.50 | 4.802 (0.379) | 4.889 | 5.097 (0.327) |
| 0.60 | 8.922 (0.767) | 8.614 | 8.542 (0.657) |

Agreement with the boundaries repo's analytic run to 0.03-0.06 dx up to t = 0.1 (float32 against float64; warpSPH has no XSPH / boundary friction, which `DFSPH2D` has by default), then the growth of chaos; warpSPH is as close to the compiled omniSPH as that run is, or closer, at every time.
Noise and wall metrics at t = 0.6 / 1.0 (spacing irregularity, 5th percentile of the nearest-neighbour distance, particles within 0.35 dx of a wall): omniSPH 0.206 / 0.299, 0.00383 / 0.00196; boundaries-repo analytic 0.221 / 0.271, 0.00352 / 0.00270, 53 / 45; **warpSPH 0.217 / 0.291, 0.00371 / 0.00252, 26 / 27**, no isolated particle. The earlier probe (wall in both solves, heavy mass, my own lattice) had 0.243 and 0.00315.
Frames of the three-panel video in the boundaries repo's format (omniSPH | boundaries-repo analytic | warpSPH): `match_original_frame_t0.32.png`, `match_original_frame_t1.0.png`.
