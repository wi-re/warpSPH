# omniIncompressible on analytic walls with the calibrated lattice — hydrostatic tank and English wedge, 2026-10-09

The calibrated lattice is ported (`modules/analyticBoundary/latticeCalibration.py`, equal to the boundaries repo's `lattice_calibration` to 6 digits, `tests/test_analyticLattice.py`) and applied by the cases
(`boundary/calibration.py`, `calibratedLattice='auto'`: analytic walls under an incompressible scheme): particle mass `V'`, wall mass `mu`, and the tank walls the fluid touches moved so the first row sits at the
calibrated distance (dam break: 0.55 dx; the sampled `sloshingTank` first row was at 0.458 dy from the wall, 2 % over-dense without it). Wendland C2, n_h = 2.57 (omniSPH's h). Nothing is set by hand any more.

**Dam break** (`probe_omniAnalyticDambreak.py --massScale 1 --offsets 0`, automatic calibration): front 0.092 / 0.315 / 0.597 / 0.930 / 1.282 against the compiled omniSPH's 0.081 / 0.303 / 0.587 / 0.918 / 1.289 at t = 0.1 ... 0.5.

**Hydrostatic tank** (`sloshingTank`, zero roll, nx = 100, 300 steps, p against g (h - y)):

| | max \|v\| at t = 0.12 / 0.24 / 0.6 | p / hydrostatic, bulk / bottom rows |
|---|---|---|
| analytic, n_h 2.57 | 0.22 / 0.13 / 0.088 (decaying) | 1.017 / 1.024 |
| boundary particles, n_h 2.57 | 0.071 / 0.062 / 0.106 | 0.951 / 0.897 |
| analytic, n_h 4 (sloshing default) | 0.95 / 1.11 / 1.96 (growing) | 1.64 / 0.98 |

`DFSPH2D` on a still column: 0.2 m/s. The analytic omni is quiet and hydrostatic at n_h = 2.57 (video `examples/sloshingTank/output/omni_hydro_2026-10-09/`, `hydrostatic_tank_t1.0_frame.png`: flat layer, incoherent noise below 0.1 m/s);
**at n_h = 4 it is still unstable** (OPEN_PROBLEMS 31: the relaxed Jacobi of the omni solve and the support size).

**English wedge** (`probe_omniEnglishWedge.py`, 2.4 x 1.2 m tank, 0.5 m of water, 0.24 m sharp wedge at the bottom centre, dp = 0.02, tank and wedge both analytic; the wedge side is not a calibrated lattice: the lattice is cut in steps against the slope):

| t [s] | max \|v\| | KE | p / hydrostatic, bulk (rms err) | near walls | within 3 dx of the wedge |
|---|---|---|---|---|---|
| 1 | 0.931 | 1.4e-2 | 0.926 (0.147) | 0.962 (0.093) | 1.030 (0.169) |
| 2 | 0.468 | 5.4e-3 | 1.018 (0.122) | 1.041 (0.076) | 0.999 (0.126) |
| 3 | 0.240 | 2.9e-3 | 1.084 (0.102) | 1.095 (0.103) | 1.088 (0.101) |
| 4 | 0.236 | 1.9e-3 | 0.982 (0.079) | 0.983 (0.057) | 0.989 (0.056) |

The same case with boundary particles (omni, mass 0.9715): max \|v\| 1.1 and KE 2.0e-2 at t = 4 (not decaying), bulk p / hydrostatic 0.46. The flat tank without the wedge: 1.019 bulk, 1.023 near walls at t = 2 s, KE falling.
Frames (`english_wedge_frames_t0_t1_t4.png`): the stepped lattice at the wedge; a hump of fluid over the apex at t ~ 1 s while it fills the cut gaps; a flat free surface and a regular lattice along the floor and the slopes at t = 4 s, velocity speckle ~0.1-0.2 m/s. Accepted by English et al.: p on the hydrostatic line down to the wedge corners, KE small and not growing: both met.
