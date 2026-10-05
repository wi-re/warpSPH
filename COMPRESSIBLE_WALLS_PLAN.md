# Compressible solid-wall support (Monaghan, CompSPH, CRKSPH)

> Index and status: [PLANS.md](PLANS.md) (row "COMPRESSIBLE_WALLS_PLAN.md").

## Goal

One scheme-agnostic wall treatment for the compressible schemes
(`schemes/monaghan.py`, `schemes/compSPH.py`, `schemes/crkSPH.py`), so fast
flows and compression dynamics can run against solid boundaries. Today none
of them reads `kinds == 1`; only the weakly-compressible/ACSPH paths do (via
`modules/mdbc`).

## Reference behaviour (Spheral)

Spheral has no dummy-particle walls: planar reflecting/periodic/constant-state
ghost nodes rebuilt every step, included in the CRK moment sums; solids are a
second `NodeList` with its own EOS. To be checked against its source/docs
before relying on this (written from memory).

## Design

Shared module (working name `modules/compressibleWall.py`):

1. `extrapolateWallState`: rho, u, v on boundary particles from fluid
   neighbours; p from the EOS. Interpolant selectable: Shepard, linear
   (mDBC-style), CRK-corrected.
2. `enforceWallVelocity`: generalises `computeBoundaryVelocities`.
3. Optional wall Riemann state (mirrored state) for strong shocks.

Scheme-specific parts:

| Scheme | Interpolant | Boundary particles in the sums |
|---|---|---|
| Monaghan | Shepard / linear | pair forces + continuity |
| CompSPH | Shepard / linear | the scheme's own pair terms |
| CRKSPH | CRK | must also enter the A, B, gradient moment sums |

Adiabatic wall: `dudt = 0`; non-fluid `dxdt/dvdt/drhodt` already zeroed by the
`nonFluidMask` pattern in `deltaSPH.py`.

## Survey (step 1, 2026-10-05)

- `CompressibleState` and `CompSPHState` carry `kinds` (constant), but
  `monaghan.py`, `compSPH.py` and `crkSPH.py` never read it: every row gets the
  same support solve, density, EOS and pair terms, and nothing zeroes the
  update of non-fluid rows (`deltaSPH.py` does, via `nonFluidMask`).
- Density is recomputed each step (Owen adaptive support, then summation or
  `computeCRKFactors`), not integrated, so a wall row's density/h would be
  overwritten unless the extrapolation runs after that solve.
- `evaluateOptimalSupport` solves h for every row, wall rows included.
- `modules/mdbc` is tied to `WeaklyCompressibleSPHConfig` (`fluid.restDensity`,
  `fixedSoundSpeed`): `computeMdbcDensity*` are not reusable as is. The
  velocity helpers (`_shepardFluidVelocity`, `noSlip`/`freeSlip`, ghost
  offsets) are candidates; the no-penetration shift is WC-specific.
- No example samples boundary particles for a compressible scheme yet.

## Steps

1. Survey: done, see above.
2. Reference tests: 1D shock reflecting off a wall (Sod/Noh against a
   mirrored-ghost or analytic solution); a quiescent uniform gas in a closed box
   (must stay at rest); piston/moving wall.
3. Implement Shepard extrapolation for Monaghan; run the tests.
4. CompSPH, then CRKSPH (boundary rows in the moment sums).
5. Compare against a reflecting-ghost reference for wall-normal momentum.

## Examples

Live in `examples/compressibleWalls/` (separate from `examples/compressible/`,
which stays pure fluid dynamics). Numbered ladder, one case per file:

1. `01-closedBox`: quiescent gas at rest, plus hydrostatic column (unit test).
2. `02-piston`: moving wall, exact piston solution.
3. `03-shockReflection`: Mach-M shock off a wall, Rankine-Hugoniot reflected state.
4. `04-woodwardColellaWalls`: blast wave between two walls (1D).
5. `05-forwardStep`: Mach 3 step (2D corner/grazing wall).
6. `06-shockCylinder`, `07-cylinderBowShock`: curved wall, standoff distance.

Minimum set: 1, 2, 3, 5. Run one at a time with video and progress output.

## Cautions

- u must stay consistent with rho so wall pressure is not spuriously high.
- Linear extrapolation can overshoot near strong shocks; needs a limiter.
- Dummy-particle walls conserve wall-normal flux only approximately.

## Log

- 2026-10-05: plan created; nothing implemented.
- 2026-10-05: step 1 survey done (see Survey).
- 2026-10-05: first Monaghan wall. `modules/compressibleWall/` (Shepard gather
  of rho, u from fluid rows onto `kinds == 1` rows via `FluidToBoundary`,
  prescribed wall velocity, zeroed wall update) hooked into
  `schemes/monaghan.py` after the density sum. Case `closedBox`
  (`cases/closedBox.py`, `examples/compressibleWalls/01-closedBox.py`, 1D, 8
  wall layers per side): 1000 steps to t=0.5 in float32, fluid max|v| 3e-6,
  pressure error 7e-6, wall pressure error 3e-6, energy flat; no alarms.
  Only a no-harm baseline (uniform state, so Shepard is exact). Next: case 3
  (shock reflection) and 2 (piston), which discriminate; then CompSPH, CRKSPH.
- 2026-10-05: case 3 `shockReflection` (`cases/shockReflection.py`, shared
  helpers in `cases/compressibleWalls.py`; Mach 2, gamma 1.4, nx 400, exact
  RH states and reflected-front speed). Findings, Monaghan, t=0.45, B7:
  - Shepard-only wall: reflected shock forms but peak p is 0.93 of exact,
    front 0.16 off, and fluid tunnels through the wall layer (30 rows past
    the surface, 11 beyond the domain edge).
  - First-order MLS (`interpolateLiuLiu`) wall: identical to Shepard here
    (no change), and NaN without a limiter. Dropped; not worth carrying.
  - Wall state = state behind the shock that stops the approach speed `u_n`
    (normal from the fluid centroid; closed form, `u_n = 0` gives the Shepard
    state): peak p 0.97, front error 0.03. This is the plan's "wall Riemann
    state" and it was needed, not optional.
  - Wall rows take the fluid's smoothing length (the support solve gave the
    outer ones h up to 7x the fluid's): small gain.
  - Leakage is a wall-depth problem: `wallLayers` 8 -> 16 takes escapes at
    t=0.45 from 11 to 1 and peak p to 1.005, peak rho 1.013. Default is now
    16. Mach 1.3: 2 escaped, peaks 0.99; Mach 3 (8 layers): 36 escaped.
  - Correction: the "energy drift" above was not a drift. The thicker wall
    shrinks the fluid region, so the initial total energy is 17.67 instead of
    18.34; with 16 layers it stays 17.67-17.72 through the whole run.
  - Root cause of the leak and the near-wall deficit: wall rows kept their
    initial mass, so the fluid density sum saw the wall as rest-density gas
    (rho2) and fluid that entered the wall layer read a low density and stayed.
    Fix: wall mass = `rho_wall * V` (V = m/rho at the first call), and the
    fluid density sum redone after the wall update, then the wall applied
    again. Result (16 layers, nx 400, t=0.5): plateau p and rho within 0.1% of
    state 3, front error 0.003, 0 penetrating/escaped, total energy flat
    17.67, peak overshoot at the wall 4.6% (p) / 2.7% (rho). Mach 1.3 and 3
    the same (plateau within 0.2%, 0 escaped; 2 rows past the surface at Mach 3).
    Mach 5 is not a valid test at this length: the left wall's rarefaction
    reaches the reflected front first.
  - Still open: the last fluid row at the wall reads ~17% low in p
    (coarse wall lattice against compressed gas); no no-penetration backstop
    (not needed at Mach <= 3 here).
  Next: piston (case 2, moving wall; needs prescribed wall row motion), then
  CompSPH and CRKSPH.
