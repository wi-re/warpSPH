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
which stays pure fluid dynamics). Numbered ladder, one case per file; every
case runs under `--scheme Monaghan|CompSPH|CRKSPH` (CRKSPH with
`--supportMode KernelMeanSymmetric`):

1. `01-closedBox` (`cases/closedBox.py`): quiescent gas at rest.
2. `02-piston` (`cases/piston.py`): moving wall, exact push (shock) and
   withdraw (rarefaction) solutions, piston-work energy balance.
3. `03-shockReflection` (`cases/shockReflection.py`): Mach-M shock off a wall,
   Rankine-Hugoniot reflected state.
4. `04-woodwardColellaWalls` (`cases/woodwardColellaWalls.py`): the blast waves
   between two walls (1D); the periodic mirror-symmetric `woodwardColella` is
   its exact reference (`scripts/compare_wallMirror.py`).
5. `05-forwardStep` (`cases/forwardStep.py`): Mach 3 step, run in the gas frame
   (the step moves, no inflow/outflow needed).
6. `06-shockCylinder` (`cases/shockCylinder.py`): piston-driven Mach-2 shock on
   a fixed cylinder; stagnation pressure p3 at impact, then the steady value.
7. `07-bowShock` (`cases/bowShock.py`): cylinder at Mach 3 through gas at rest;
   standoff vs Billig, pitot pressure.
- `sedovWalls` (`cases/sedovWalls.py`): Sedov in a walled box (1/2/3D); the
  periodic `sedov` on [-1, 1]^d is its exact reference.

2D cases share `buildWalledBox` (`cases/compressibleWalls.py`): a cell-centred
lattice, fluid = box minus a solid SDF, walls = the solid within `wallLayers`
spacings of the fluid, wall velocity a function of position.

## Reference: how the WCSPH path moves walls (surveyed 2026-10-05)

Rigid bodies (`rigidBody/`, `systems/weaklyCompressible.py`); cases
`movingObstacle` (spin), `drivenSquare` (oscillation); `lidDrivenCavity` is the
other route (a Dirichlet function on the wall velocity, not a body).

- Wall motion is prescribed, never integrated from forces: `finalize` calls
  `integrateRigidBody` (explicit Euler on pose with `linearVelocity`/
  `angularVelocity`; `dudt = dwdt = 0` at every call site) then
  `updateBodyParticlesWCSPH`, which rewrites wall and ghost positions from the
  new pose, writes the rigid-body velocity into the wall rows' `velocities`
  (for every BC type), and writes `boundaryAccelerations` (centripetal term
  only). A case with a time law re-sets the velocity in `postStep`
  (`drivenSquare`): a one-shot assignment would translate at constant speed.
- Wall rows have zero `dxdt/dvdt/drhodt` in the update; the pose update in
  `finalize` is the only thing that moves them. Non-fluid positions are
  restored from the initial state every step before that (a nonzero BC
  velocity on a wall row otherwise drifts it under `symplecticEuler`: the
  half-step position update reads the raw velocity, `DELTASPH_VALIDATION_PLAN.md`
  §9.2, lid cavity drifted 1.5 domain widths by t=3).
- The wall's own velocity has to enter the pair terms, not just advect the
  wall: mDBC slip conditions are written on the relative velocity
  `w = Shepard(u_f) - u_body` (`u_g = u_body + w_t - w_n` free slip,
  `u_body - w_t` no slip), so continuity sees the wall's normal motion
  (`modules/mdbc/velocity.py`; validation plan 5.2, 5.7). `u_body` must come
  from the rigid body, not from the wall row's stored velocity, or the BC
  compounds on its own output (normal slope -3 instead of -1; validation plan,
  "u_body must come from a rigid body", near line 3070).
- Order per step: adjacency, fluid density, mDBC wall density, wall velocity
  BC, EOS, forces; wall pose advanced last in `finalize`, so every stage of a
  step sees the wall at its start-of-step pose and velocity (the wall is
  frozen across RK stages, and the velocity is the one for the step start).
- A wall that starts at its peak speed shocks the fluid (`drivenSquare`:
  density -7%/+6% versus -0.3%/+1.8% for the spinning body); it starts from
  rest instead.
- Open there: constant-omega only (no tangential/linear acceleration terms);
  the wall is not coupled to the fluid force.

Implications for the compressible wall: the piston needs (1) prescribed wall
row motion with the wall velocity as the `target` in `applyCompressibleWall`
(already a parameter, and the approach speed uses it), (2) wall rows kept out
of the integrated update and restored/advanced in the system's `finalize`, (3)
the velocity at the step start used for every stage, (4) a start from rest or
a ramp, (5) `wallVolumes` re-checked once the wall moves (a moving wall
changes the fluid's local spacing but not the wall row spacing).

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
  - Near-wall deficit, dissected 2026-10-05 (`scripts/probe_wallPressureDeficit.py`,
    `.tmp/probe_wallRefinement.py`): the last fluid row reads -17% in p / -16% in
    rho, decaying to 0 over a ~7-row boundary layer. Cause: the wall sits on the
    *initial* fluid lattice (spacing dx) while state 3 is compressed 6x (spacing
    dx/6), so the last row's kernel (width h ~ 0.66 dx) under-samples the wall
    side of the density sum -- only 1 wall particle sits in the kernel at the
    default resolution. Refining the wall to spacing dx/ref is best (p -11%) when
    ref matches the compression ratio (dx_w = dx/6); coarser under-samples and
    finer makes the support solve shrink the last row's h (0.0064 -> 0.0027),
    which degrades it. A ~11% floor remains (the dummy wall is a uniform lattice,
    not the exact mirror of the local fluid). Confined to the last ~6 spacings;
    the plateau diagnostic already excludes that band, so the bulk solution is
    unaffected. General fix = ghost/exact-mirror wall (a uniform wall cannot stay
    matched to a compressing fluid). No no-penetration backstop (not needed at
    Mach <= 3 here).
  Next: piston (case 2, moving wall; needs prescribed wall row motion), then
  CompSPH and CRKSPH.
- 2026-10-05: residual near-wall deficit dissected further (`.tmp/probe_owen_psi.py`):
  the previous entry attributed it to wall under-sampling, but the dominant term
  is a *frozen over-large h* on the 1-2 outermost fluid rows. Owen's psi fill
  (wp_psi0.py) IS scale-invariant once a kernel is well-sampled, so the LUT
  returns n_h=4.00 (= nhTarget) for *every* row and the relaxation
  `h_new=(1-a+a*s)*h` runs with s=1 -> a no-op; h can grow (under-sampled) but
  never shrink (one-way ratchet). The outermost row's h froze at 0.00640 (~1.8x
  its neighbour) *despite identical m/rho*. Owen *does* count the wall
  (psi_H with wall = live = 12.778 vs fluid-only 10.595) -- not a wall-counting
  bug. Fix: global spacing clamp in the adaptive-support dispatch
  (`_clampSupportsToVolume` in modules/adaptiveSupport/optimalSupport.py):
  `h = min(h, n_h_target*(m/rho)^(1/d))` for fluid rows, applied to the h every
  scheme's `evaluateOptimalSupport` returns (Monaghan + CRKSPH). No-resampling
  (corrects a derived quantity on existing rows). Validated 2026-10-05:
  - shockReflection (Mach 2, t=0.45): outermost-row pressure deficit -15.3% ->
    -8.8%, density -13.7% -> -9.3%; wall stays fixed (innermost row 0.9225);
    small 0.145dx transient penetration. Residual ~9% is the sparse wall (1
    layer at dx/2) under-filling the boundary kernel -- separate, harder issue
    (the ghost/exact-mirror wall from the prior entry).
  - sedov 1D (nx 800, t=0.5556, full goal time): peak shock rho unchanged
    (4.0093 clamp vs 4.0080 no-clamp) so the 4.0-vs-6.0 is pre-existing (hat
    init / SPH smearing), not a clamp artifact; h grows naturally (0.232 in the
    rarefaction); over-grown h capped (max h/spacing 5.14 -> 4.01); run healthy
    (diverged=False, 0 alarms, total energy conserved at 1.0).
  Cleanup: removed the unused `applyMirrorDensityCorrection` (and its `_b7`/
  `_b7Cdim` helpers) from compressibleWall -- the static mirror correction
  over-corrected (+37%) and was never the right mechanism.
  **2026-10-06 (later): the clamp is now wall-only** (`SimulationConfig.supportVolumeClamp='walls'` default, user): it
  degraded smooth pure-fluid flow (OPEN_PROBLEMS §20, resolved). shockReflection near-wall rows are identical under
  'walls' and 'always' (outermost -0.9 % in p; 'off' -10.2 %). The clamp's root cause (Owen ratchet) is OPEN_PROBLEMS §21.
- 2026-10-06 (overnight session, user away): module pass, moving walls, CompSPH
  and CRKSPH.
  - **API**: `modules/compressibleWall` is now a per-evaluation object,
    `wall = beginCompressibleWall(state, config)` (None without wall rows),
    `wall.apply(...)` (twice, around the density redo), `wall.finishUpdate(update)`,
    and for the compatible-energy schemes `wall.balanceFractions(f_ij, ...)`.
  - **Moving walls**: a wall row's *stored* velocity is its prescribed velocity.
    `beginCompressibleWall` captures it before anything overwrites the row (RK
    stage states are clones, `butcher.py:62`, so the step's own state keeps it);
    `finishUpdate` sets `dxdt = prescribed`, `dvdt = 0`, so walls advect with it.
    Bug this exposed: `buildWalledSystem` gave wall rows the nearest *fluid*
    velocity, so shockReflection's left wall started at `u2` -- harmless while the
    wall velocity was hard-coded to 0, a piston once it is not (fluid energy
    17.67 -> 20.99). Walls now get `wallVelocity=(left, right)`, default 0.
  - **Wall state, two-sided**: `wallRiemannState` is the star state of the
    mirrored Riemann problem -- the existing shock branch on approach, plus an
    isentropic rarefaction branch on recession (`u_n < 0`; was the Shepard state).
  - **Free slip** (`CompressibleSPHConfig.wallSlip`, default 'freeSlip'): wall
    rows show the pair terms the prescribed normal velocity plus the fluid's
    tangential one (no AV shear against a wall); 'noSlip' = prescribed. No
    effect in 1D. `wallRiemannState` (default True) switches the Riemann state.
  - **Compatible energy (CompSPH, CRKSPH)**: the pair work is split `f_ij`/`f_ji`
    and the wall's share was lost (its state is overwritten every evaluation).
    `balanceFractions` sets `f_ij = 1` on fluid->wall pairs: the fluid gets all
    of it, so a fixed wall does no work and a moving one exactly `F . v_wall`.
    `CompSPHSystem.finalize` now carries the wall masses (as Monaghan's does) and
    keeps the wall rows' u out of the compatible increment.
  - **CRKSPH**: `computeCRKFactors` is redone after the wall update (CRK density
    is `sum m_j V_j W^R / sum V_j^2 W^R`, so the wall mass `rho_w V` carries
    over). First run blew up at the shock's impact (t=0.31, negative u on the
    outermost row): wall rows with the compressed fluid's h (below their own
    spacing) have singular moment matrices. Fix: `apply(..., latticeSupport=True)`
    floors the wall h at `n_h (V)^(1/d)`; CRK only -- it costs Monaghan's
    outermost row -0.9% -> -32% in p, CompSPH -15% -> -34% in rho.
  - **Near-wall deficit, cause found** (supersedes the "sparse uniform wall /
    needs an exact-mirror wall" reading above): wall rows no fluid reached were
    given the fluid's *median* h, which reached back into the compressed fluid
    with a stale state. Rows no fluid reaches now keep the support solve's h.
    Monaghan outermost row at t=0.45: -8.8% -> **-0.9%** in p (rho -9.3% -> -2.2%);
    A/B: restoring the median fallback gives -8.8% again. CompSPH outermost row
    +13% p / -15% rho, CRKSPH +8% / -19%; second row in within 2-7% for all.
  - **Results**, shockReflection (Mach 2, nx 400, t=0.5), plateau p / peak p /
    penetrating / energy: Monaghan 0.9998 / 1.036 / 2 / 17.667 flat; CompSPH
    0.9998 / 1.12 / 1 / 17.667 exact; CRKSPH 0.9997 / 1.22 / 0 / exact.
    closedBox stays at rest under all three (|v| ~ 3e-6, float32).
  - **Case 2 `piston`** (`cases/piston.py`, `examples/compressibleWalls/02-piston.py`):
    left wall moving at `pistonSpeed` from t=0 into gas at rest; exact face state
    (RH for a push, isentropic for a withdrawal, which needs empty cells behind
    the piston: `buildWalledSystem(gaps=...)`), and `energyRatio` = fluid energy
    gain / piston work `p_face u_p t`. u_p = +1 / -1, t=0.6, plateau p ratio /
    energyRatio: Monaghan 1.0000 / 0.9989, 1.0011 / 0.9944; CompSPH 1.0000 /
    1.0003, 0.9996 / 0.9993; CRKSPH 1.0000 / 1.0003, 0.9991 / 1.0052. No
    penetration, fronts within 2 spacings. Frames checked.
  - **Step 5, mirror references** (`scripts/compare_wallMirror.py`): a periodic
    run whose initial state is mirror-symmetric about the wall planes is a run
    with exactly reflecting walls, so `woodwardColella` (periodic [-1, 1]) is the
    reference for case 4 `woodwardColellaWalls`, and `sedov` (periodic
    [-1, 1]^d) for `sedovWalls` (gas in [-1, 1]^d, walls outside; built on the
    reference's own lattice plus appended wall rows). Fields binned onto a grid
    (cell means), L1 relative to the reference's mean magnitude:
    - Woodward-Colella, 500 per tube, t=0.038: CRKSPH density / pressure / speed
      L1 0.9% / 0.6% / 0.95%, peaks rho 6.473 vs 6.472, p 410.3 vs 410.5, energy
      exact (275.2 = half the reference's 550.4), 0 penetrating. Monaghan 1.1% /
      1.6% / 2.9%, peaks within 0.1%, but fluid energy +1.7% (275.2 -> 279.9):
      the Monaghan wall is not energy-balanced (no compatible split there).
    - Sedov 2D, nx 101, until the free shock would reach r = 1.4 (t=1.47, after
      the wall reflections): Monaghan L1 5.1% / 4.2% / 6.1%, peaks low (p 0.82 vs
      1.22, rho 7.7 vs 8.6), energy -0.4%; CompSPH 9.6% / 8.9% / 21%, energy
      exact, min fluid u 0.08. CRKSPH: NaN at t=1.016 (energy exact until the
      step before, no alarm) -- see next item.
    - The WC runs are slow for a physical reason, not a wall one: dt ~ 1e-6 is
      set by the blast region's compressed edge (min h 4e-4 at the p=1000/0.1
      contact, a fluid row) under the global min(h)/max(c) CFL.
    - Note: `WOODWARD_REGIONS` uses p = 0.1 in the middle region; Woodward &
      Colella (1984) use 0.01. Not changed (both runs share it); flagged here.
  - **Wall Riemann state NaN guard**: Sedov's ambient pressure is exactly 0, so a
    round-off-negative u gives `p + b < 0` and the shock branch took sqrt of a
    negative. The gathered p is now clamped at 0 and the radicand at 0.
    Confirmed on the CRKSPH rerun (`scripts/probe_sedovWallRows.py`): fluid u
    dips to -0.0014 mid-run; with the guard the run finishes (t=1.47), energy
    exact, 0 penetrating, corner wall rows p 2.9 next to corner fluid 3.1.
  - **2D cases** (`buildWalledBox`; Monaghan unless noted). Findings in order:
    - *Builder bug*: `volumeToSupport` takes a volume and got `dx` (not `dx^d`),
      so every 2D row started at h = 0.447 (36 spacings) and the first dt was
      ~8x too large (the support solve repaired h on step 1, dt was already
      taken). It made the impulsive Mach 3 forward step NaN in 4 steps; fixed.
      The 2D results below that predate the fix are marked.
    - *Piston/floor seam leak* (shockCylinder, pre-fix): the corner fluid row
      was squeezed into the seam between the piston's bottom row (moving) and
      the floor's top row (fixed), then got behind the piston and expanded into
      the vacated region at |v| ~ 25 (`scripts/probe_pistonLeak.py`). Cause: the
      floor rows next to the piston had a *diagonal* centroid normal (the fluid
      fills only part of their neighbourhood), read the fluid streaming away
      along it as a rarefaction (rho 2.0, p 2.1 vs 4-6 around) and sucked the
      corner row in. Fix: **geometric wall normals** -- a `wallNormals` constant
      field on both compressible states (zero on fluid rows), written by the
      builders from the fluid region's SDF gradient (rigid with the rows, so
      right for translating walls); the module falls back to the centroid where
      none is given. No leak afterwards. Also: the piston is only the left
      wall's rows within the channel's height (the corner blocks belong to the
      fixed top and bottom walls).
    - *Flat 2D walls hold*: the same channel without a cylinder (shock reflects
      off the right wall at t=0.68, the reflection meets the piston): no
      penetration (max 0.017 dx), |v| bounded, energy follows the piston work
      (9.67 -> 16.32 at t=1).
    - *Curved wall, stagnation point -- OPEN (OPEN_PROBLEMS §19)*: shockCylinder
      is right through the first impact (stagnation pressure 0.80 p3 after
      impact, relaxing to 1.17x the steady value by t=0.37; penetration < 0.3
      dx), and the second shock (the bow shock reflected off the piston,
      arriving t~0.40, p* 2.4 p3 -- physical) is right too; then the stagnation
      region compresses without bound (22 p3 by t=0.445, penetration 9 dx).
      `scripts/probe_cylinderFront.py`: once the fluid is compressed ~10x its
      spacing is ~0.3 of the wall lattice's, and fluid rows sit *inside* the
      surface ((r-R) ~ -0.1 dx) exactly half-way between wall rows, as
      coincident pairs: the lattice wall's pressure field is bumpy at its own
      spacing and the compressed fluid falls into the valleys. Not the exact
      mirror symmetry (off-axis cylinder fails the same), not wall porosity in
      the wall's own kernel (the lattice h floor, `wallLatticeSupport on`, fails
      the same: the fluid's own small kernel still sees the valleys).
    - bowShock (pre-builder-fix run): bow shock clean, pitot pressure within
      1-2% of Rayleigh up to t~0.3; standoff reaches Billig (ratio 0.94-1.0) by
      t~0.3, then drifts to 1.15 with the pitot falling to 0.91 by t=0.6 and
      ~18 fluid rows inside the cylinder -- likely the same valley mechanism,
      slower. Deep cylinder rows reach p ~ 160 (gathering far fluid with a wide
      h at approach ~U): visible in the plots until the walls were hidden.
    - New knobs: `CompressibleSPHConfig.wallLatticeSupport` (None = scheme
      default), case params `wallLatticeSupport` ('auto'/'on'/'off') and
      `wallSlip`; 2D case params are scalars (`centreX/Y`, `startX/Y`).
    - *After the builder fix* (all reruns): bowShock Monaghan unchanged (standoff
      0.99 Billig at t~0.27 -> 1.14 at 0.6, pitot 0.96 -> 0.91); shockCylinder
      Monaghan fails at the same time. So the initial-h bug was not behind either.
    - *CompSPH / CRKSPH in 2D*: bowShock standoff / pitot at t=0.6: CompSPH 1.09 /
      0.98, CRKSPH **1.006 / 0.95** (Monaghan 1.14 / 0.91); 12 / 30 rows inside the
      cylinder. shockCylinder to t=0.38 (before the §19 window): peak stagnation
      p / p3 CompSPH 1.04, CRKSPH 0.95 (exact: 1.0 at impact; Monaghan 0.80),
      penetration 0.24 / 1.0 dx.
    - *Sedov 2D, CRKSPH vs mirror* (after the NaN guard): L1 density / pressure /
      speed 12% / 19% / 16%, peaks low (rho 10.3 vs 12.4), energy exact, 0
      penetrating -- worse than Monaghan's 5/4/6%; CRK's lattice-h floor on the
      wall rows is the obvious suspect (not checked).
    - *Experiment, §19*: fluid<-wall pressure force through the wall's own kernel
      (`Scatter`, lattice-floored wall h; fluid-fluid unchanged), Monaghan
      shockCylinder: fewer rows end up inside the cylinder (32 vs 235) but the
      stagnation pressure still runs away at the same time (271 p3 by t=0.47).
      The gap-filling is a symptom; the runaway compression at the stagnation
      point is not caused by it alone. Reverted (not kept as an option).
    - *forwardStep, Monaghan, classic resolution (80 per unit), t=4*: 6733
      steps, 5.5 min. The step block now reaches past the tunnel's end by its
      whole travel (`buildWalledBox(keepSolid=..., extendHi=...)`: body rows kept
      outside the box, normals from the body alone), so no vacuum opens behind
      it (before: gas fell into the room behind the block's back end, went
      through the floor there and inflated h -- 450 ms/step). Max penetration
      into the step 0.41 dx (transient), at most 1 row out of the tunnel, bow
      standoff at y=0.5 0.159. Frames (window in the step frame, matplotlib)
      show the Woodward-Colella structure at t=4: bow shock, Mach stem with the
      triple point near (0.65, 0.8), reflected shock on the step top at x~1.35.
    - *Test regression, not from this session*: `test_shockCapturing::
      test_sod1d_readHayfieldSuppressesContactOvershoot` fails since 013a22f
      (passes at d0f4774): R&H contact entropy overshoot 3.43% vs NoneSwitch
      3.24%. The support clamp is the change; it is still needed for the walls
      (without it the outermost row is -10% in p again, with a frozen 1.8x h and
      a coincident pair). OPEN_PROBLEMS §20, your call.
    - *§19 data points*: CompSPH and CRKSPH full shockCylinder runs fail in the
      same window as Monaghan (CompSPH NaN at t=0.497, CRKSPH at 0.433); Monaghan
      at nx 400 has the same onset (19 p3 at t=0.439 vs 22 p3 at 0.446 at nx 200).
      Scheme- and resolution-independent, triggered by the second shock's
      arrival. Next discriminating runs: `wallRiemannState` off; a flat wall
      facing the same second shock.
    - `wallRiemannState` off (new case param `--no-wallRiemannState`): same
      onset (11 p3 at t=0.444), penetration 9.5 dx -- not the wall state either.
