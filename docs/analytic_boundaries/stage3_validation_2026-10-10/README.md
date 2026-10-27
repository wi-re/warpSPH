# Stage 3 validation (E4 – E6), 2026-10-09 / 10

Evidence for the channel-flow group ported into warpSPH (ANALYTIC_BOUNDARIES_PLAN.md, stage 3): the Stokes arrays, Taylor–Couette flow and the cylinder wake on analytic walls, each against the boundaries
repo's oracle (`DeltaSPH2D` / `DFSPH2D`) **on the identical lattice**, plus the exact / literature value where there is one. Raw result lines: `results_extract.txt`; frames: the PNGs here; the runs and probes:
`scripts/probe_{stokesArray,taylorCouette,cylinderWake,periodicChannel}.py`, `scripts/probe_taylorCouetteOracle.py` (term-by-term). Units: rho0 = 1, mass = volume, nu = the discretisation's shear-wave viscosity.

## Taylor–Couette (r1 = 0.2, r2 = 0.5, inner cylinder rotating at 1, t = 25, `scripts/probe_taylorCouette.py`)

The lattice decides the torque. Two things were wrong in the first runs and are fixed in the case (`taylorCouette.py`): the regular sampler fits its lattice to the domain extent (spacing 1 / 48.8 at nx = 48, so the
calibrated particle mass was wrong by 3 % in density), and particles sat 0.0013 from the wall while the oracle cuts the lattice next to each wall (`wallCut`: the calibrated first-row distance for the incompressible
loops, half a spacing for delta+). With the lattice identical to the oracle's (N = 1384 at n = 48, 592 at n = 32):

| scheme, setup | profile / exact | inner torque / exact | outer torque / exact | oracle (same lattice) |
|---|---|---|---|---|
| omniIncompressible, closed preset, n = 48, H = 2.505 dx (the oracle's) | 0.9994 | **0.9733** | 0.888 | DFSPH2D: 1.0004, **0.9746** |
| omniIncompressible, closed preset, n = 48, H = 4 dx | 1.0000 | 0.9975 | 0.898 | (not run) |
| delta+, Morris + no-slip moment closure + complement, n = 32, H = 4 dx | 0.9943 | **0.9356** | **0.762** | DeltaSPH2D: 0.9999, **0.9428**, 0.766 (0.812 of the inner) |
| delta+, same, + `pressureConsistent` | 0.9944 | 0.9355 | 0.762 | (identical: no net pressure torque on a centred disk) |

Torque ratios are the booked wall loads (reaction to the wall accelerations, plus the Morris traction `-2 mu Omega A` of the rotating wall) over `T = -4 pi mu omega r1^2 r2^2 / (r2^2 - r1^2)`; the outer wall on this
cut lattice transmits less than the exact torque in the oracle too (0.77), the curved no-slip closure is not exactly torque-conserving there. Static check on identical particles with the exact profile
(`probe_taylorCouetteOracle.py`): densities identical, wall terms equal to 1.5 % rms of the first rows (float32 vs float64), static inner torque 0.646 vs 0.649.

Frames: `taylorCouette_delta_n32_t25.png`, `taylorCouette_omni_n48_nh2.505_t25.png` (intact rings, smooth gradient, no fliers).

## Cylinder wake (`cylinderWake`, pinned-velocity frame, box 30 x 15 D, blockage 8.3 %, D/dx = 10, delta+ Morris, t = 150, `scripts/probe_cylinderWake.py`)

The `pinned` band (`modules/boundaryConditions/pinned.py`): fluid particles in a frame around the seams keep the free stream, the rest of the box is the flow.

| Re | | C_D | C_L amplitude | St | L_w / D |
|---|---|---|---|---|---|
| 20 | this port | 2.153 | – (steady) | – | 0.921 |
| 40 | this port | 1.710 | – (steady) | – | 2.135 |
| 40 | oracle, same lattice | 1.7225 | – | – | 2.138 |
| 100 | this port | 1.520 | 0.426 | 0.1705 | (unsteady) |
| 100 | oracle, same lattice | 1.5278 | 0.376 | 0.1717 | (unsteady) |

The unconfined literature (C_D 2.0–2.09 / 1.50–1.55 / 1.33–1.35, L_w/D 0.91–0.94 / 2.2–2.35, St 0.164–0.166) is lower in C_D by the blockage, as the oracle's own numbers are. Re 40 agrees with the oracle to 0.7 %
(C_D) and 0.1 % (L_w), Re 100 to 0.5 % (C_D) and 0.7 % (St); **the Re 100 lift amplitude is 13 % above the oracle's (0.426 vs 0.376): open** (the oracle's own amplitude moves 0.376 -> 0.333 from D/dx = 10 to 20).
Frames: `wake_Re{20,40,100}_t150.png` (steady wakes; a regular vortex street at Re 100; a thin density speckle at the pinned frame's periodic seam, not compared with the oracle).

## Stokes arrays (`stokesArray`, disk R = 0.2 in the unit cell, body force f = 0.03, n = 48)

| scheme | K / K_ref (K_ref = (1 - c) K_SA = 26.49) | load / body force | history K/K_ref at t/4, t/2, t |
|---|---|---|---|
| omniIncompressible, closed preset, t = 20 | 1.0035 | 0.9992 | |
| omniIncompressible, closed preset, square a = 1/3 (K_ref 25.91) | 0.9863 | 0.9991 | 1.048, 0.996, 0.988 |
| divergenceFree, closed preset, t = 20 | 0.991 | 0.9817 | 1.048, 1.003, 0.995 |
| delta+, Morris + `pressureConsistent`, t = 16, eager (lattice uncut: particles down to the wall) | 1.025 | 1.0016 | 1.112, 1.041, 1.025 |
| delta+, Morris, no `pressureConsistent`, t = 16, eager (uncut lattice) | 0.906 | 0.886 | 0.665, 0.788, 0.920 |
| **delta+, Morris + `pressureConsistent`, t = 40, lattice cut at r >= R + dx/2 (the oracle's; N = 1988)** | **1.0029** | **0.9983** | steady from t = 25: U = 0.05292 |
| oracle `periodic_cylinder_array.py` (DeltaSPH2D, same lattice, `--wall noslipMoment --consistent --visc morris --cal 0.985`) | 0.9981 | 0.9985 | U = 0.0531 (t = 30) |

`pressureConsistent` is needed on the cut lattice (without it the disk loses 11 % of the balance to the uniform-pressure force of the lattice-vs-wall residual). The lattice matters as in Taylor–Couette: the oracle cuts the delta+
disk at `r >= R + dx / 2` (`wallCut`, now the case's default for delta+); on that lattice the port is within 0.5 % of the oracle in K and equal in the momentum balance. (An earlier t = 40 pair of runs, K 1.017 and 1.007 with
balances 0.9996 and 1.025, were read through a diagnostics race of the pipelined graph loop: the graph rewrote the bodies' load tensors in place while the previous step's diagnostics read them; the loads of a replayed step
are now cloned behind the replay and handed to that step's outputs, `tests/test_analyticGraph.py` compares the booked load of every step against the eager run.)

## Performance (E8)

`benchmarks/analyticWalls/` (`bench_step.py`, `count_ops.py`): the analytic-wall step was CPU-launch-bound (4600 kernel launches, 86 host syncs per step for 592 particles; GPU busy 8.9 of 35 ms). Replayed from a CUDA
graph and with the no-slip closure compiled (opt-in `WARPSPH_COMPILE_WALLS=1`, on in the probes): delta+ Taylor–Couette n = 32 31.0 -> 7.8 -> **4.8 ms/step**, Stokes array n = 48 23.4 -> 5.6 ms, bitwise equal to the eager
step with the graph alone (`tests/test_analyticGraph.py`); the 125 000-step Taylor–Couette run takes ~10 minutes instead of over an hour.
