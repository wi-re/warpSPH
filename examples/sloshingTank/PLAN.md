# Sloshing-tank verification (SPHERIC Test Case 10) — step-by-step plan

**Goal.** Verify that the two free-surface schemes in this repo — the weakly
compressible `deltaSPH` ("WCSPH") and the incompressible `divergenceFree`
("DFSPH") — reproduce the classic laterally-excited sloshing-tank experiment,
by computing the pressure history at wall Sensor 1 and comparing it to the
measured record.

The reference implementation is diffSPH's
`examples/weaklyCompressible/16_SloshingTank.ipynb`; this plan ports its setup
onto warpSPH's `Case`/`runner` framework.

---

## STATUS (2026-09-03)

**Tooling: done and verified.** `warpSPH.cases.sloshingTank` (registered),
`run_sloshingTank.py` (`--scheme wcsph|dfsph`, `--replot`, `--video`,
`--targetDt`/`--alpha`), `16-sloshing-tank.ipynb`. Geometry, sensor, roll
excitation and wave timing all check out against the measured record.

**Verdict per scheme (nx = 100–200, `t → 7 s`):**

- **DFSPH (`divergenceFree`)** — *reproduces the pressure signal.* Needed one
  code change: `schemes/dfsph.py` now **persists both projection pressures**
  (`pressures` ← constant-density solve, `soundspeeds` ← divergence-free
  solve; carrier convention shared with `dfsphReference`). First impact ~3.6 kPa
  sim vs ~3.6 kPa measured, on time. The wave then runs **2–5 % fast** — the
  known SPH wave-celerity artifact, and it **converges** (per-cycle lag halves
  from nx=100 → nx=200). The **free surface does not converge** (`minDensity`
  0.48 → 0.24 with refinement) — the Part-23 quiescent-free-surface weakness;
  needs the scheme-level fix, not resolution.
- **WCSPH (`deltaSPH`)** — *diverges at the first wave impact* (`t ≈ 3.5 s`),
  and **`c_s` is not the cause**: 3× sound speed + 3× artificial viscosity fixed
  the impact-pressure magnitude and the local density excursion but the run
  still blows up at the same time, with particles ejecting. It is a
  **free-surface particle-distribution / tensile instability** — `deltaSPH`
  runs with no effective particle shift at the free surface.

**Next work is spun out to `docs/historic_plans/WCSPH_SHIFTING_PLAN.md`** (now COMPLETE): make the
weakly-compressible particle shift work *near free surfaces* without the volume
growth that made it get switched off there. Target case: `rotatingSquarePatch`.
The sloshing tank surviving the first impact is that plan's transfer test.

---

## 0. The experiment (from `SPHERIC_TestCase10/`)

`SPHERIC_TestCase10_Fig1.png` gives the geometry exactly:

| quantity | value | notes |
|---|---|---|
| tank internal width `B` | 0.900 m | x ∈ [−0.45, +0.45] |
| tank internal height | 0.508 m | y ∈ [0, 0.508] |
| still-water depth `h` | 0.093 m | "lateral impact, water, H93" configuration |
| coordinate origin | bottom-centre of the tank floor | roll angle φ about the origin |
| **Sensor 1** | left wall, `(−0.45, 0.093)` | at the still-water line — the lateral-impact gauge |
| fluid | water, ρ ≈ 1000 kg/m³ | |
| gravity | 9.81 m/s² | |

Excitation: harmonic **roll** about the origin, amplitude ≈ ±4°, period ≈ 1.7 s
(near the first sloshing mode). The full prescribed motion is tabulated in
`SPHERIC_TestCase10/data_files/lateral_water_1x.txt` (Δt = 5·10⁻⁵ s, 0 … 8.35 s):

```
Time[s]  Pressure[mbar]  Position_smooth_splines[deg]  Velocity[deg/s]  Acceleration[deg/s2]  Position_original[deg]
```

`Pressure[mbar]` is the measured Sensor-1 signal (1 mbar = 100 Pa; peaks ≈ 71 mbar ≈ 7.1 kPa).
`Repeatability_Files/Water_4first_peak_lateral_impact_tto_0_85_H93_B1X.txt` lists the
first four impact-pressure peaks over many repeats — the run-to-run scatter band.

## 1. Modelling approach — rotate gravity in the tank frame

Following diffSPH: instead of physically rotating the tank walls, solve in the
**tank-fixed frame** and rotate the gravity vector by the roll angle θ(t):

```
g_dir(t) = R(−θ(t)) · (0, −1) = (−sin θ, −cos θ),   θ(t) = spline roll angle
```

`schemes/modules/gravity/directional.py` reads
`schemeConfig.gravityConfig.direction` fresh every step, so a `postStep` hook
that rewrites it each step is all that is needed. Walls and fluid stay static
in this frame.

**Known limitation (documented, not fixed in v1):** the tank-fixed frame is
non-inertial, so the exact body force also carries the Euler term
`−dω/dt × r` and the centrifugal term `−ω × (ω × r)`. The pure gravity-rotation
model drops both. For ±4° at ~1.7 s over a 0.45 m half-width they are small
next to `g`, and dropping them is exactly what the diffSPH reference does, so
v1 matches the reference. A `--inertialForces` extension can add them later via
a per-step `BoundaryCondition` forcing function.

## 2. Deliverables

1. `src/warpSPH/cases/sloshingTank.py` — a registered `Case`:
   - `configureScheme`: rectangular 0.9×0.508 interior domain inside a periodic
     wrapper widened by `band` layers; free-surface detection on; directional
     gravity on; scheme-aware dissipation (`deltaSPH`: artificial viscosity
     `alpha`; `divergenceFree`: physical `nu`, optional shift/XSPH).
   - `buildSystem`: fluid = box `x∈[−0.45,0.45], y∈[0,h]`; boundary = interior
     complement (`sampleDomainSDF`). Reuses `fluidRegion`/`boundaryRegion`/
     `buildRegionSystem` from `cases/weaklyCompressible.py`.
   - `initialConditions`: `v = 0`; for `deltaSPH`, pick `c_s` + `dt` from
     `targetDt` via `setupTimestep`; stamp the t=0 gravity direction.
   - `postStep`: interpolate θ(t) from the roll table, rewrite the gravity
     direction.
   - `timestep`: `config.dt` for `deltaSPH` (fixed acoustic dt, like `dambreak`);
     `kolmogorovIncompressibleTimestep` for `divergenceFree`.
   - `diagnostics` (recorded every step → `RunResult.trajectory`):
     `sensorPressureTait`, `sensorPressureLinear` (WCSPH, from the nearest
     **boundary** particle's density via EOS, ×`rho0Physical` to get Pa),
     `sensorPressure` (DFSPH, the solver's carried `pressures[idx]` ×`rho0Physical`),
     `sensorDensityRatio`, `sensorPressureProbe` (Shepard-smoothed over fluid
     particles within `probeRadius` of the sensor — a spray-robust cross-check),
     plus `rollAngleDeg`, `maxVelocity`, `kineticEnergy`, `maxDensity`,
     `minDensity`. The sensor particle index is located once and cached on
     `ctx.scratch`.
   - `setupPlot`/`updatePlot`: `particlePlot` of velocity + density.
   - Register in `src/warpSPH/cases/__init__.py::CASE_MODULES`.

2. `examples/sloshingTank/run_sloshingTank.py` — script runner:
   - `--scheme {wcsph,dfsph}` (prepends the right integrator/kernel/CFL preset),
     `--nx`, `--tLimit`, `--nSteps` (smoke), `--smoothSigma`, `--no-plot`,
     `--out`.
   - Runs the case via `warpSPH.runner.run`, pulls the sensor series with
     `RunResult.series(...)`, loads `lateral_water_1x.txt`, Gaussian-smooths the
     simulated pressure, and writes to `examples/sloshingTank/output/`:
     - `<scheme>_sensor_pressure.{png,pdf}` — simulated vs measured pressure,
       plus the roll angle and the repeatability peak band;
     - `<scheme>_series.npz` — raw `t`, pressures, roll angle, health metrics;
     - a printed summary (peak pressure, first-impact time, wall-clock).

3. `examples/sloshingTank/16-sloshing-tank.ipynb` — the notebook: markdown
   description of the experiment and the tank-frame method; explicit `CaseSpec`;
   `buildContext`→`configureScheme`→`buildSystem`→`initialConditions`; a region
   plot with the sensor marked and the fill line; the prescribed roll history;
   the unrolled step loop with a live field plot and per-step sensor recording;
   the final measured-vs-simulated comparison; a density-bounds / KE health
   panel. Mirrors the structure of `examples/weaklyCompressible/12-dambreak.ipynb`.

## 3. Parameters (case defaults)

```
L (= tank width B) = 0.9      nx = 150  ->  dx = 0.006  (h ≈ 15.5 dx)
tankHeight = 0.508            fillDepth = 0.093
band = 5                      n_h = 4.0
gravityMagnitude = 9.81       rho0Physical = 1000.0   (Pa conversion only; scheme runs at restDensity = 1)
sensorPos = [-0.45, 0.093]    probeRadius = 3 * dx
rollDataFile = SPHERIC_TestCase10/data_files/lateral_water_1x.txt
rollStartTime = 0.0
tLimit = 7.0                  targetDt = 2e-4
WCSPH: kernel Wendland4, rungeKutta2, KernelMeanSymmetric, cflFactor 0.3, inviscid AV alpha 0.02
DFSPH: kernel Wendland2, semiImplicitEuler, SuperSymmetric, cflFactor 0.2, dt 1e-3, maxDt 2e-3, nu ~1e-6, shifting off
```

## 4. Execution order

1. `PLAN.md` (this file). ✅
2. Write `cases/sloshingTank.py`; register it.
3. Smoke test: `python -m warpSPHRun sloshingTank --nSteps 20 --no-plot --no-store`
   (WCSPH) and `... --scheme divergenceFree --integrationScheme semiImplicitEuler
   --kernel Wendland2 --cflFactor 0.2 --dt 1e-3 --nSteps 20` (DFSPH). Fix wiring
   until both step without error and the sensor is located in the fluid corner.
4. Write `run_sloshingTank.py`; short run (`--tLimit 0.5`) for each scheme; check
   the plot/`.npz` are produced and the pressure trace is finite and O(kPa).
5. Write the notebook; execute top-to-bottom at reduced `nx`/`tLimit`.
6. Full runs `--tLimit 7` for both schemes; save figures under `output/`.
7. Write up: does each scheme track the measured Sensor-1 pressure (phase of the
   sloshing period, magnitude of the impact peaks within the repeatability
   band)? Record where each one departs. `divergenceFree` is known to struggle
   with quiescent free-surface-under-gravity states
   (`DFSPH_IMPROVEMENT_PLAN.md` Part 23 / `hydrostaticColumn`), so a divergence
   or heavy over-dissipation here is a plausible finding, not a wiring bug —
   report it as a result.

## 5. Acceptance

- Both schemes run to `t = 7 s` (or the failure mode is characterised).
- `output/wcsph_sensor_pressure.pdf` and `output/dfsph_sensor_pressure.pdf`
  exist, overlaying simulated and measured Sensor-1 pressure.
- The notebook runs end-to-end.
- A short findings section (in the notebook's final markdown cell and echoed
  here) states, per scheme, whether the sloshing experiment is reproduced.

---

# RESULTS (nx = 100, dx = 9 mm, h/dx ~ 10, `t -> 7 s`)

Re-draw either figure without re-running: `python run_sloshingTank.py --replot
--scheme {wcsph,dfsph}`. Raw series are in `output/<scheme>_series.npz`.

## The case wiring is correct

- The **applied roll** overlays the prescribed roll exactly in both runs -- the
  spline interpolation and the per-step `gravityConfig.direction` rewrite work.
- Under WCSPH the sensor first responds at **t ~ 2.35 s**, the exact time of the
  measured first impact, and there is a matching ~300 Pa bump near t ~ 0.8 s in
  both traces. The sloshing-wave kinematics and the tank-frame coupling are
  right.

## WCSPH (`deltaSPH`) -- does NOT pass; diverges at the first slam

`c_s = 16.6 m/s`, fixed `dt = 2e-4 s`, `alpha = 0.02`, free-surface detection on.

- Quiescent phase (t < 2.3 s) is clean: `rho in [1.000, 1.003]`, sensor pressure
  order 100 Pa.
- The first sloshing wave hits Sensor 1 at t ~ 2.35 s and drives a pressure
  **spike to ~5.7 kPa smoothed / ~50 kPa raw** (measured first peak ~3.6 kPa,
  band 2.2-13 kPa), immediately followed by a large **undamped acoustic
  ring-down** (+-10 kPa oscillation at the sensor).
- The run then **diverges at t = 3.41 s** (step 17055): `maxVelocity -> 6e10`,
  `maxDensity -> inf`.
- Diagnosis: a weakly-compressible free-surface *impact* instability -- the
  low sound speed and the artificial-viscosity-only dissipation cannot absorb
  the slam, and the mDBC wall + free surface at the sensor corner amplify it.
  Mitigation to try (not done here -- this is a verification, not a scheme fix):
  smaller `targetDt` / higher `c_s`, a background pressure or `delta+`-SPH
  particle shift, stronger density diffusion, `--wallBC noSlip` with `nu > 0`,
  or lower `nx` sensitivity study.

## DFSPH (`divergenceFree`) -- does NOT pass; no wall-pressure observable

`semiImplicitEuler`, Wendland2, `cflFactor 0.2`, `dt ~ 1.2e-3 s` adaptive.

- The run **completes all 7 s without diverging** and tracks the roll.
- But the VD+PS `divergenceFree` scheme **carries no stored pressure field** --
  it enforces incompressibility with a momentum-neutral position shift, not a
  pressure that persists on the particles. `state.pressures` is all zeros, so
  `sensorPressure` and the fluid-probe are identically 0: **there is no
  Sensor-1 signal to compare.** (The `dfsphReference` IISPH-style scheme does
  carry a non-negative pressure and could be probed instead; it has its own
  documented late-time failure -- `DFSPH_IMPROVEMENT_PLAN.md` Part 29.)
- The **free surface de-densifies badly**: `sensorDensityRatio` and fluid
  `minDensity` fall to **~0.48-0.70 rho0** over the run -- the quiescent
  free-surface-under-gravity weakness `hydrostaticColumn` / Part 23 document,
  on display here too. `maxVelocity` reaches ~4.9 m/s (vs `sqrt(g h) ~ 0.95`),
  i.e. the surface layer is being flung, not sloshing.

## Verdict

The tooling reproduces SPHERIC Test Case 10 faithfully (geometry, sensor,
roll excitation, wave timing). **Neither scheme currently passes the
quantitative test**: WCSPH gets the physics right for ~2 roll periods then the
first wave impact destabilises it; DFSPH-VD+PS is stable but has no wall
pressure to read and its free surface collapses. Both failures are consistent
with limitations already recorded in `DFSPH_IMPROVEMENT_PLAN.md`.

---

# FOLLOW-UP (same session)

## DFSPH now stores both projection pressures

`schemes/dfsph.py` computed the divergence-free projection pressure and the
constant-density solve pressure every step and **threw both away** (only the
non-default `DIVERGENCE_SOLVER='vdps'` branch ever wrote `state.pressures`).
Now, in every branch, after the solves:

    currentState.pressures   <- constant-density / particle-shift solve p  (pRho)
    currentState.soundspeeds <- divergence-free projection p               (pDiv)

Raw solver iterates, fluid rows only (non-fluid held at 0). This is the same
carrier convention `schemes/dfsphReference.py` already uses
(`st.pressures`=CD kappa, `st.soundspeeds`=DF kappaV) -- the DFSPH family has
no acoustic sound speed, so `soundspeeds` is a free slot. Incompressible /
DFSPH test subset stays green (`tests/... -k "dfsph or incompressible or
hydrostaticColumn or columnCollapse"`, 25 passed; the lone
`test_incompressibleKrylov::test_optimalStepRejectedForConstantDensitySolver`
failure is pre-existing on `main`).

`sloshingTank.diagnostics` reads them through a fluid-particle probe near the
sensor (DFSPH fills fluid rows only): `sensorPressureCD`, `sensorPressureDF`.
On a 40-step smoke, `sensorPressureCD` = 1..1000 Pa (settling transient),
`sensorPressureDF` = -36..83 Pa -- i.e. the constant-density solve carries the
physical pressure and the DF projection is a small correction, as expected.
CD is the DFSPH analogue of the compression pressure a wall gauge measures.

## DFSPH re-run with the pressure field wired (nx = 100, t -> 7 s)

`output/dfsph_sensor_pressure.pdf`, regenerated. Now that
`sensorPressure` reads the constant-density solve pressure, DFSPH **does
produce a Sensor-1 signal, and it is a good one**:

- **First impact (t ~ 2.37 s): magnitude and timing both match.** Simulated
  peak ~3.6 kPa raw / ~2.2 kPa smoothed vs measured ~3.6 kPa -- inside the
  2.2-13 kPa repeatability band. The small pre-impact bump near t ~ 0.25 s is
  also reproduced.
- **The wave arrives progressively earlier than measured.** Cross-correlation
  lag of the simulated vs measured pressure, per sloshing cycle:

  | cycle | window [s] | sim-vs-measured lag |
  |---|---|---|
  | 1 | 2.0-3.4 | **-31 ms** (sim early) |
  | 2 | 3.4-5.0 | **-37 ms** |
  | 3 | 5.0-6.6 | **-78 ms** |

  ~1.9 % of the ~1.64 s period on cycle 1, growing to ~4.7 % by cycle 3. This
  is the known SPH free-surface **wave-celerity / dispersion error**: a small
  error in the gravity-wave phase speed, which in this *resonant* case (roll
  period ~ first sloshing eigenperiod) accumulates as a growing phase lead
  instead of staying bounded. Aggravated here by (a) the DFSPH free surface
  degrading over the run (`minDensity -> 0.48`, `sensorRho -> 0.70`), which
  shifts the effective wave speed -- consistent with the lead *growing*, not
  being constant; and (b) only ~10 particles across the 93 mm depth -- SPH
  sloshing-period convergence is resolution-sensitive. **Next check: an nx
  150 / 200 study should shrink the per-cycle lead if it is the celerity
  artifact.**
- **Later impacts overshoot.** The 3rd impact spikes to ~12 kPa (band top
  13 kPa) with heavy post-impact ringing, and the inter-impact pressure holds
  a ~500-1000 Pa plateau the measured signal does not -- the surface piling
  against the wall and not draining back, tied to the density collapse.

**Revised DFSPH verdict:** with the pressure field wired, `divergenceFree`
*does* reproduce the sloshing-tank pressure -- first impact on the money -- but
the wave phase runs 2-5 % fast over three cycles (SPH celerity artifact +
resonance + coarse/degrading surface) and the late impacts overshoot as the
free-surface density collapses.

## DFSPH resolution / convergence study (nx 100 vs 200, `output/dfsph_nx200/`)

`--video` on the runner now renders velocity+density field frames (vispy/EGL,
headless) and encodes `<scheme>_field.{mp4,gif}`. nx=200 -> dx 4.5 mm, ~20
particles across the 93 mm depth, 31.8k particles, `t -> 7 s` in ~15 min.

**The two errors converge in opposite directions.**

*Phase drift -- CONVERGENT (it is the SPH wave-celerity artifact).*
Cross-correlation lag of simulated vs measured pressure, and impact-peak times:

| | nx = 100 | nx = 200 | measured |
|---|---|---|---|
| lag cycle 1 (2.0-3.4 s) | -31 ms | -28 ms | -- |
| lag cycle 2 (3.4-5.0 s) | -37 ms | **-19 ms** | -- |
| lag cycle 3 (5.0-6.6 s) | -78 ms | **-35 ms** | -- |
| lag global (2-7 s) | -45 ms | **-27 ms** | -- |
| impact-peak times [s] | 2.37 / ~4.3 / 5.63 | **2.37 / 4.05 / 5.67** | 2.40 / 4.07 / 5.71 |

The phase lead roughly **halves** per 2x refinement (global -45 -> -27 ms;
cycle 3 -78 -> -35 ms), and the nx=200 impact-peak times land within ~40 ms of
measured. So the "waves arrive early" is numerical dispersion in the
gravity-wave celerity, and it is resolution-convergent -- nx ~300-400 would
likely bring it inside ~1 % of the period.

*Free-surface collapse -- DIVERGENT (a method defect, not a discretisation
error).*

| metric | nx = 100 | nx = 200 |
|---|---|---|
| `minDensity` over the run | 0.481 | **0.242** |
| `maxVelocity` | 4.89 | **7.36** |
| `maxDensity` | 1.030 | 1.043 |
| peak sensor pressure (t > 2 s) | 12.0 kPa | **16.2 kPa** (over the 13.1 kPa band top) |

Refining the grid makes the free-surface skin de-densify *worse*, the spray
faster, and the impact overshoot larger -- the field video shows the surface
layer tearing into spray with ejected particles in the "air". This is the
documented DFSPH quiescent free-surface-under-gravity weakness
(`DFSPH_IMPROVEMENT_PLAN.md` Part 23 / `hydrostaticColumn`): a finer grid just
gives the instability finer structure. **Resolution will not fix this; it
needs the scheme-level free-surface treatment.**

**Takeaway:** DFSPH's sloshing *kinematics* converge (wave speed, impact
timing) -- the case is correctly posed and the method is consistent -- but its
*free surface* does not, and that caps how far the pressure amplitude can be
trusted past the first impact.

## Sound speed

diffSPH's `16_SloshingTank.ipynb` computes `c_s` from the acoustic CFL
(`0.3 * h / Kernel_Scale / targetDt`), prints it, then **hardcodes
`c_s = 20 m/s`** (`rho0 = 1000`, gravity magnitude `10`, EOS `stiffTait`,
gamma 7). Our `setupWeaklyCompressibleTimestep` derived **`c_s = 16.6 m/s`**
from `targetDt = 2e-4` -- close to diffSPH, slightly lower.

Both are marginal *for the impact*, not the bulk sloshing: the bulk wave speed
`sqrt(g h) ~ 0.96 m/s` gives `Ma ~ 0.06` (fine), but the diverged WCSPH run
showed a ~18% density excursion at the sensor during the first slam
(`p ~ 50 kPa = rho0 c_s^2 * 0.18`), i.e. local `Ma ~ 0.42` -- well outside
weakly-compressible validity. Raising `c_s` needs a matching `dt` cut, so the
lever is `targetDt`: `c_s ~ 1/targetDt`. The runner now takes `--targetDt` and
`--alpha`.

### Stiffer WCSPH re-run: `--targetDt 1e-4 --alpha 0.06` (c_s = 49.7)

`output/wcsph_hics/`. **c_s is not the fix.** 3x the sound speed and 3x the
artificial viscosity:

- **helped the impact peak:** first-impact *smoothed* peak dropped 5.7 kPa ->
  ~3.5 kPa (measured ~3.6), and the sensor-local density excursion went 18% ->
  0.6% -- the EOS/compression side is now healthy;
- **did not help the instability:** the run still **diverges at t = 3.49 s**
  (baseline t = 3.41 s -- essentially unchanged), after a growing `+-5-8 kPa`
  post-impact ring-up from ~2.4 s. At the blow-up a handful of particles
  eject (`maxVelocity -> 2191` in a 0.9 m box, `maxDensity -> inf`,
  `minDensity -> 0`) while the sensor density is still `1.000 +- 0.006`.

So the WCSPH failure is a **free-surface particle-distribution / tensile
instability** triggered by the repeated wall slam at the sensor corner, not a
weak-compressibility (sound-speed) problem. `deltaSPH` here runs with
artificial viscosity + delta-SPH density diffusion + the mDBC no-pen shift but
**no fluid particle shifting**; the likely fixes are a delta+-SPH particle
shift, a background pressure, or Monaghan's artificial-stress tensile
correction -- scheme work, out of scope for this verification.
