# omniIncompressible dam break, analytic walls vs boundary particles vs compiled omniSPH — 2026-10-09

`scripts/probe_omniAnalyticDambreak.py --particles --offsets 0.5 0.05` (nx = 100, n_h = 2.57 = omniSPH's h = sqrt(20 V / pi), 2093 particles = omniSPH's 23 x 91 lattice, free slip, `semiImplicitEuler`, dt <= 1e-3).
The reference is the compiled omniSPH of the boundaries repo (`~/dev/curvatureBoundaries/.tmp/omni/cmp_dam.npz`, particle centres on the block edges, **walls one spacing outside the block**); the
boundaries repo's own `DFSPH2D` (hydrostatic wall closure) is in the same file and agrees with it within 1 % (front) / 0.4 % (mean height). Columns: front displacement [m] / mean height above the
wall plane [m] / max |v| [m/s]. Analytic offset = the wall plane moved outward by that many dx (0.5: first row one spacing from the wall, omniSPH's convention; 0.05: the calibrated lattice of `DFSPH2D`, 0.55 dx).

| t | omniSPH | DFSPH2D | warpSPH omni, particle walls | warpSPH omni, analytic, offset 0.5 | analytic, offset 0.05 |
|---|---|---|---|---|---|
| 0.1 | 0.081 / 0.366 / 1.93 | 0.082 / 0.367 / 1.99 | 0.411 / 0.392 / 4.76 | 0.142 / 0.376 / 2.54 | 0.403 / 0.398 / 4.82 |
| 0.2 | 0.303 / 0.263 / 2.82 | 0.299 / 0.264 / 2.94 | 0.887 / 0.298 / 5.10 | 0.393 / 0.275 / 3.49 | 0.883 / 0.305 / 5.20 |
| 0.3 | 0.587 / 0.150 / 3.37 | 0.580 / 0.151 / 3.43 | 1.283 / 0.187 / 8.79 | 0.720 / 0.165 / 4.23 | 1.287 / 0.192 / 7.58 |
| 0.4 | 0.918 / 0.087 / 3.67 | 0.913 / 0.088 / 3.78 | 1.284 / 0.133 / 7.07 | 1.067 / 0.101 / 4.42 | 1.287 / 0.134 / 7.36 |
| 0.5 | 1.289 / 0.061 / 4.07 | 1.279 / 0.061 / 4.16 | 1.285 / 0.138 / 6.97 | 1.289 / 0.082 / 7.88 | 1.285 / 0.151 / 6.79 |
| 0.6 | 1.297 / 0.079 / 6.97 | 1.298 / 0.079 / 6.27 | 1.286 / 0.186 / 5.53 | 1.291 / 0.098 / 6.21 | 1.284 / 0.209 / 7.05 |

* All three warpSPH runs complete to t = 0.6, no divergence. Density minimum 0.34-0.38 by t = 0.5-0.6 (spray / isolated particles, as in the frames), maximum 1.01-1.09.
* The wall distance decides the early dynamics: with the first row at 1.0 spacing (omniSPH's convention) the front is 1.2-1.7x omniSPH's before the wall impact and the wall impact (t >= 0.5) matches; with the wall at 0.5-0.55 dx (boundary particles continue the lattice; the boundaries repo's calibrated lattice) the front is 5x too fast at t = 0.1 (initial overpressure) in the particle and the analytic runs alike. omniSPH itself blows up with the wall half a spacing from the first row (dfsph-validation.md of the boundaries repo, s.2).
* Remaining gap to omniSPH with the right convention: front +75 % / +30 % / +23 % / +16 % at t = 0.1-0.4, mean height +3 % ... +34 %, max |v| 2.5 vs 1.9. Not investigated: omniSPH's `boundaryFriction` (5e-3) and XSPH (1e-4) are off here (`omniIncompressible`'s `XSPH_FLUID` / `XSPH_BOUNDARY` / `DAMPING` code is commented out), and the first-row density (0.87 at offset 0.5, the lattice is under-dense at the wall, as in omniSPH).
* Frames (`frame_t0.31_analytic_omni.png`, `frame_t0.55_analytic_omni.png`): collapsing column, wave along the floor, splash up the right wall; two isolated particles at the left wall at t = 0.55.
* The hydrostatic tank at rest (OPEN_PROBLEMS 31) is still unstable with this scheme; the dam break, the case omniSPH is known to run well, is stable.
