# mDBC contact-line suction — plan

Status: **step 1 done; fix implemented (opt-in) but NOT validated** — see §7,
§8 and §9 (2026-09-24: the videos show every variant spraying high-speed
fliers through the whole domain; the §7/§8 checks did not measure this). Tracks `OPEN_PROBLEMS.md` §8. Background and all
numbers: `FREESLIP_DAMBREAK_FINDINGS.md` §9 (commits `fcaa998`, `2738e21`,
`cfcd796`). Run rules: `CLAUDE.md` (video on, stream progress, one run at a
time).

## 1. The problem

A **thinly supported free-surface particle next to a solid wall** gets
negative pressure. The symmetric `(p_i + p_j)` pressure force then pulls it
toward the wall. Closer to the wall it has fewer fluid neighbours, so the
suction deepens. The growth is geometric: one ACSPH particle went p = −20 →
−26 → −36 → −54 → −91 → −180 → … → −1.8e6 over ~20 steps (×1.2–2.5 per step)
while it moved through the floor. The particle then crosses the wall, is kicked
to 1e2–1e6 m/s, and the adaptive dt collapses until simulated time stops.

**Where it shows up**

| Scheme | Instance |
|---|---|
| ACSPH | a particle pulled diagonally into a tank corner (before the `fcaa998` fix); a particle riding the **ceiling** at constant speed against gravity for ~1 s (every nx=24 dam-break run); the **run-up jet** at the far wall, 1–2 particles thick, scattering at the first impact (nx=70, t* ≈ 2.8) |
| δ-SPH | the sloshingTank **ceiling hover** (UID 4104, `OPEN_PROBLEMS.md` §1); the english2025 low-`alpha` wall leak that the neighbour-count ramp fixed |

**What it is not**

- **Not walls in general.** TGV (periodic), bounded random flow (fluid pressed
  against all four walls) and a spinning moving obstacle are clean in both
  schemes at nx=64 (`FREESLIP_DAMBREAK_FINDINGS.md` §9.7). A free surface
  meeting a wall is required.
- **Not solver non-convergence (ACSPH).** The pseudo-time loop reaches its
  target every step while the mode grows. Its metric only measures the velocity
  residual, so it does not see pressure.
- **Not fixed by** the corner fillet, `noPenetrationShift` (it parks the
  particle at the wall instead), the Antuono pressure switch (no change), or
  clamping surface-particle pressure to ≥ 0 (worse: the runaway moves one layer
  inward).
- **Held off, not fixed, by** the paper's viscosity together with a Michel shift
  whose U_char includes wall particles. With both, nx=24 survives to t* 8.1;
  nx=70 still fails at the first impact.

**The open question** that decides the fix: does the negative pressure come
from the **wall closure** (the extrapolated wall pressure: ACSPH Eq. (61)
Shepard, δ-SPH mDBC ghost fit), or from the **fluid–fluid truncation
artifact** (`OPEN_PROBLEMS.md` §1: symmetric pressure sum over a truncated
kernel, with the Antuono mask chatter)? Probably both contribute, but in
different proportions per instance.

## 2. Step 1 — measure first (decides the fix)

Goal: for the particles that fail, split the pressure force into its wall and
fluid contributions, and see where the negative pressure originates.

1. **Reproducible case with a checkpoint.** Use the nx=24 ACSPH dam break
   (all fixes, Michel + viscosity), which shows the ceiling rider from
   t ≈ 0.87 s, and the nx=70 run-up jet (failure at t ≈ 0.64–0.685 s). Store
   `storeMode='states'` checkpoints shortly before each event and resume from
   them (`CaseSpec.resumeFrom`, `warpsph-resume-checkpoint-support` memory), so
   each probe is minutes, not an hour.
2. **Per-particle trace** for the ceiling rider, the run-up-jet particles, and a
   healthy near-wall control particle, every step through the event:
   - its own pressure, neighbour counts (fluid / wall), λ, surface flag;
   - the pressure force split into the sum over **wall** neighbours and the sum
     over **fluid** neighbours;
   - the extrapolated pressures of the wall particles it sees, and **how many
     fluid neighbours each of those wall particles extrapolated from**.
3. **Same trace in δ-SPH** on the sloshingTank ceiling hover (`sloshing-tank-case`,
   `antuono-pressure-switch-bug` memories — UID 4104 has a resume point), to
   confirm the two schemes share the mechanism rather than just the symptom.

Outcome: a clear "wall term dominates" / "fluid term dominates" / "both"
verdict, which selects step 2 or step 3 (or both).

## 3. Step 2 — if the wall closure drives it

**Conditioning ramp on the wall pressure** — the analogue of english2025's
`alpha` neighbour-count ramp (`english2025-alpha-neighbour-ramp` memory):
when a wall particle's fluid support is a few sparse surface particles, blend
its extrapolated pressure toward a non-attracting value (the hydrostatic
reference, or 0) instead of trusting the Shepard value.

- ACSPH: `schemes/artificialCompressible.wallPressures` →
  `modules/incompressible/wallPressure.wallPressureExtrapolation`.
- δ-SPH: the matching step in the mDBC density schemes, if step 1 shows the
  same driver there.
- Gate on the smallest signal that separates the failing rows from the healthy
  ones in the step-1 data (fluid-neighbour count, Shepard denominator, or λ);
  do not pick a threshold before seeing the data.
- Acceptance: nx=24 **and** nx=70 ACSPH dam break past t* 8 with no wall
  crossing; the ceiling rider falls; the §9.7 no-free-surface checks unchanged;
  `hydrostaticColumn` still holds (pressureSlopeRatio ≈ 1); tests green.

## 4. Step 3 — if the fluid-side truncation drives it

This is `OPEN_PROBLEMS.md` §1's problem, whose next steps are already listed
there, cheapest first:
1. freeze the Antuono surface mask across RK stages;
2. a continuous coverage blend instead of the hard surface/bulk mask;
3. Barecasco et al.'s Eq. (8), not yet implemented;
4. a force clamp keyed to the renormalisation eigenvalue λ, inside the
   free-surface population only.

ACSPH uses the plain `(p_i + p_j)` form, so check whether an ACSPH-side variant
of (4) is needed separately from the δ-SPH switch work.

## 5. Step 4 — resolution and precision

Our runs: H/Δx 14–42, float32, ε_v = −5, RK2, Δt/Δτ = 5, CFL_t = 0.2. De
Courcy et al.: H/Δx 400, float64, ε_v = −8, RK3, Δt/Δτ = 2, CFL_t = 0.4. Run
the nx=70 ACSPH dam break once in **float64 with ε_v = −8** (and optionally the
paper's RK3 settings) to see whether part of the failure is numerical noise
rather than the mechanism. This is independent of steps 2–3 and cheap to set
up.

## 6. Smaller follow-ups (independent)

- ~~**Runner stall watchdog on simulated-time progress.**~~ **Done
  2026-09-23**: `CaseSpec.stallProgress` -- abort once the last 1000 steps
  advanced simulated time by less than this fraction of `tLimit` (relative, so
  one value fits every case); `tests/test_runner.py`. `stallDtSteps` stays for
  the exact-floor case.
- **ACSPH wall acceleration in Eq. (61):** `a_b` is not tracked for moving
  walls. It matters for `movingObstacle` and any moving-body ACSPH case.
- **ACSPH divergence always uses the no-penetration mirror** (De Courcy §3.3);
  only the viscous Laplacian should follow the no-slip/free-slip choice. Today
  one wall velocity feeds both; this only matters for no-slip ACSPH cases.
- **Eq. (58) Taylor correction** of p and v after the shift (paper: little
  influence).
- **ACSPH step cost:** the pseudo-time loop often runs 100–200 iterations
  (~1–2 s/step, launch-bound). Profile before using ACSPH at scale.
- **Re-run the six §9.7 reference checks with video** for visual confirmation.

## 7. Results (2026-09-23)

Tooling: `scripts/probe_contactLine.py` (toys + `--toy marrone31`, streamed
per-step forensics: the wall/fluid split below, the most-negative fluid row's
flags / lambda / neighbour counts / wall distance, a ceiling-rider count, and a
sim-time stall watchdog). Outputs under `scripts/out_contactLine/`, all with
video.

### 7.1 Literature — what the physics says

- **Batty, Bertails & Bridson 2007 §4** (`literature/batty2007_…`): fluid
  "crawling up walls and even along ceilings" comes from the *bilateral* wall
  condition `u.n = v_s.n`. The physical one is the complementarity
  `0 <= p ⊥ (u - v_s).n >= 0` (their Eq. 15): the fluid may leave the wall, and
  where it does, the gap is air, p = 0 -- "we rule out suction from keeping
  the fluid stuck".
- **Every AC-family SPH code does the wall half of this**: EDAC (Ramachandran &
  Puri 2019, dam break: "the only change … is a clamping of the boundary
  pressure to non-negative values so as to prevent the fluid from sticking to
  the walls"; PySPH `ClampWallPressure`), dual-time SPH (Ramachandran 2021,
  citing Hughes & Graham 2010), DualSPHysics' m2dbc `rho_b >= rho0` guard.
- **Why not a blanket clamp here**: this repo already measured that a blanket
  `rho_b >= rho0` erases legitimate negative wall pressure on a *submerged*
  receding wall and ratchets sloshingTank (`density2025.py` docstring). The
  distinction is physical: tension at a wall is legitimate where no air can
  reach the gap (submerged wall, sealed box -- the gap would be vapour), and
  unphysical where air can. Linear potential flow says the same: for a wall
  accelerating away from the fluid, the attached-flow wall pressure is
  `p(0,s) = rho s [g + a((2/pi) ln s + c)]` (s = depth below the contact
  point), negative near the contact line for **any** `a > 0` -- a contact line
  never holds tension; hydrostatics (`a = 0`) never needs it.

Criterion, no tuned threshold: **a row may carry tension only if no
free-surface particle lies inside its kernel support** -- exactly the
surface *vicinity* (the raw Marrone set dilated by one support), which
`dilateSurface` (AllToAll) already evaluates at wall rows too.

### 7.2 Toys (ACSPH nx=32, dam-break configuration: Michel + alpha_nu + eps_v -5)

`hydrostaticColumn` with gravity turned (fixed a latent bug on the way:
`hydrostaticColumn` and `dambreak` wrote `gravityDirection` into
`gravityConfig.origin`, which directional gravity never reads -- the param was
dead; the `[0,-1]` default hid it).

- `drop` — a block under the ceiling, gravity toward the open side: exact
  answer free fall. **Baseline: it drips** -- the contact layer stays pinned
  (|v| ~ 0), the block stretches into a stalactite, the neck thins to 1-2
  particles and its pressure diverges (-8 … +6); the tension on the last
  contact rows grows like weight/contact-area (-3 -> -9 -> -19). This is the
  dam-break runaway in 1.5k particles.
- Force split on the wall-contact rows (units of g, from the linear
  `(p_i+p_j)` operator): wall p_j **+0.80**, own p_i against the wall **+0.52**,
  fluid-fluid -0.55. At t = 0, with p = 0 in the fluid, Eq. (61)'s hydrostatic
  term alone already gives the ceiling wall p_w = -0.92.

| drop, t = 0.2 s | fallRatio | contact tension |
|---|---|---|
| Eq. 61 (baseline) | 0.64 | -1.3 -> **-19** |
| no hydrostatic term | 0.64 | -1.6 -> -8 |
| blanket `p_w >= 0` (EDAC/PySPH) | 0.69 | **own p_i: -2.9 -> -8.8, holds 0.87 g** |
| p >= 0 on raw surface set + wall clamp | 0.70 | -3 … -11, runaway one layer in |
| p >= 0 on **vicinity** (fluid + wall rows) | 0.78 | interior only, <= 7, released by 0.15 s |
| p >= 0 everywhere (IISPH-style bracket) | 0.87 | 0 (remaining 10%: viscosity x shifting drag) |

(free fall reads ~0.97 on this metric -- one-step integrator lag.)

Readings:
1. **The wall closure alone is not the driver in ACSPH.** Clamping the wall
   moves the load onto the fluid row's *own* pressure: the pseudo-time loop
   converges `div v = 0` with the mirrored wall velocity, i.e. it enforces the
   bilateral constraint exactly, and separation shows up as `div v > 0` ->
   `p_i < 0`. That is why EDAC gets away with the wall clamp (weakly
   compressible, the constraint is soft) and ACSPH does not.
2. **The raw surface set is not enough.** The rows carrying the tension sit
   against the wall with lambda 0.6-0.86; the wall counts as support, so
   Marrone does not flag them -- this is the previously recorded "clamping
   surface particles moves the runaway one layer inward". They *are* in the
   vicinity.
3. Hence the fix is Batty's complementarity on both sides of the contact:
   **`acParams.cavitationProjection = 'vicinity'`** projects `p >= 0` after every
   RK stage (a projected pseudo-time iteration) on vicinity fluid rows, and on
   the Eq. (61) value of vicinity wall rows -- plus exactly-isolated rows
   (`detectIsolated`, §7.5: Marrone reads an empty support as bulk).

Controls (both nx=32, t = 0.5):
- `sealed` (fillRatio 1, gravity up; legitimate bottom-wall tension, no free
  surface): `vicinity` is **bit-identical** to `off` (empty set). Side
  observation, pre-existing and identical in both: the sealed box drifts to
  vmax ~ 1 and an all-negative pressure field -- the `(p_i+p_j)` tensile
  instability under uniform tension (OPEN_PROBLEMS candidate).
- `column` (the at-rest hydrostatic column): `pressureSlopeRatio` 1.001 both,
  residual equal; `off` has wall rows at **-0.65** (side-wall rows above the
  waterline -- Eq. 61 extrapolates the column upward: Batty's "crawling up
  walls", at rest); `vicinity` 0. Residual motion slightly up (vmax 0.012 vs
  0.009, dispMax 0.3 dx vs 0.16 dx).

### 7.3 Marrone 3.1 dam break, ACSPH

Final code (vicinity + isolated rows; nx=24 is step-for-step identical to
the vicinity-only run, the isolated rows never fire there):

| | nx=24 off | nx=24 vicinity | nx=70 off | nx=70 vicinity |
|---|---|---|---|---|
| reached | t* 8.5 | t* 8.5 | **fails t* 2.8** (§9.5) | **t* 9.91** in one run (7505 steps, 62 min) |
| rows under a wall, t* > 4 (geometry-free count) | present every step, median \|v_up\| **0.000** (pinned riders) | present (roof hit repeatedly), median \|v_up\| 0.19, mean v_up **-0.23** (falling away) | – | 17 at t* 3.75 (jet hits the roof -- physical); after t* 5.5 moving, median \|v_up\| 0.30 |
| wall p < -0.5 | every step (1065/1065), min -18.5 | 3 steps (t* 4.66, 6.04), min -5.5 | – | 544/7505 steps, min -23 (t* 4.5-4.9, see below) |
| penetration at end | 1 ptcl / 1.55 dx | 2 / 1.38 dx | – | 0 at end (max 2 ptcl / 1.0 dx) |
| P1 plateau (Buchner ~0.55) | 0.567 | 0.570 | – | 0.532 |
| P2 peak 4.5-6 (Buchner 0.28) | 0.446 | 0.340 | – | 0.307 |
| steps | 1065 | 1240 | – | 7505 |

(The spiky first P1 peak (~5) fails the <= 1.6 check in both nx=24 runs --
H/dx = 14. At nx=70 P1 reads low around the first impact (0.15-0.35); the
2026-09-21 nx=70 ACSPH run without the projection read the same (0.18-0.37),
so that is pre-existing ACSPH probe behaviour, not this change.)

nx=70 with `vicinity` (first run: vicinity only, to t* 8.49 + a resume to
9.9; the final-code run above reproduces it in one go) passes the first far-wall run-up that kills `off`
(t* 2.8: vmax 117 -> 1e6): vmax 5.6 there, no tension anywhere through t* 4.4.
What remains, classified by the per-step forensics (most-negative row: flags,
lambda, neighbour counts, wall distance):

- **t* 4.5-4.9, wall-adjacent** (1.5-3.5 dx off a wall, lambda 0.39-0.8):
  fluid to -40, wall rows to -23, *not* in the vicinity set -- lambda < 0.5 is a
  badly incomplete support, yet no raw-surface particle is detected within one
  support. The set is only as good as Marrone's raw detection.
- **t* 6.0-6.8, 7.7, 8.4, interior** (6-30 dx off every wall, lambda
  0.003-0.8): -30 … -136, the plunging jet's entrapped air cavity (Marrone
  3.1 closes it at t* ~ 6). Rows round a small pocket are not flagged (the
  scan cone sees fluid across the bubble). Fluid-fluid: `OPEN_PROBLEMS.md` §1's
  population, not a wall contact.
- **t* 8.49, vmax 6.6 -> 23.5 in 2 steps**: one particle ejected by a
  *positive*-pressure kick 7.7 dx from any wall, then ballistic (0 neighbours,
  p frozen at +119). A free-surface flyer (§1), not a contact line.

### 7.4 Verdict on the plan's open question (§1)

**Both, and they are one mechanism: the wall is bilateral.** The wall closure
(Eq. 61, including its hydrostatic term, which is negative at any ceiling and
above any waterline) supplies tension the moment p_f ~ 0; but in ACSPH the
larger share is the fluid row's own pressure, because the converged
`div v = 0` with the mirrored wall velocity *is* the no-separation constraint.
A fix on the wall pressure alone (plan step 2 as written, and the literature's
blanket clamp) provably does not release the contact (7.2, row 3). The fix is
the complementarity on both sides, restricted to where air can reach -- no
threshold, no case parameter. It holds for ACSPH; for delta-SPH it does not
transfer as a clamp (§7.5). Plan step 3 (the fluid-side truncation) is still
open for the flyer/cavity population above, which is what the remaining
nx=70 tension and the one ejection are.

### 7.5 delta-SPH: the same complementarity does NOT transfer naively

sloshingTank (the case's own defaults, nx=200, english2025 + fourtakas2019,
t = 6.5, `scripts/probe_contactLine.py --toy sloshing`). Baseline `off`:
survives; rows hanging under a wall (geometry-free count: a wall row within
1.5 dx directly on the gravity-up side) all run long, mean 2-12, max 25; wall
p to -64, fluid to -91; vmax up to 7.7; smoothed sensor peaks 3.4-6.4 kPa
(TC10 measured band 2.2-13.1 kPa).

| variant | result |
|---|---|
| v1: clamp EOS `p >= 0` on vicinity fluid + wall rows | **diverges t = 2.90** |
| v2: v1 + floor the *carried* density `rho >= rho0` on those rows (in `finalize`) | survives, ceiling rows 0 after t = 3, but **ratchets**: bulk median density 1.001 -> 1.03, KE / 85 by t = 6, sensor peaks 17-41 kPa |
| v3: v1 + isolated rows in the set | **diverges t = 2.90** (same step as v1) |

Why, from the forensics:
- **v1/v3 — stored deficit.** delta-SPH's state is rho, not p. With the
  tension clamped away, a separating row's continuity equation books the
  separation as particle *expansion*: droplets reach rho = 0.21 (80 % deficit).
  When such a row sinks back out of the set -- or, in v1, simply becomes
  isolated (see below) -- `p = c^2 (rho - rho0)` releases it: p -174 … -320,
  interior (15-25 dx off every wall), a re-entering droplet (5.5 m/s, 0
  neighbours) triggers the runaway at t = 2.85. `off` survives because the
  tension keeps droplets from expanding that far in the first place.
- **v2 — acoustic rectification.** WCSPH's free surface reflects compression
  waves as rarefactions; truncating rho < rho0 in the surface layer rectifies
  them into a net compression -- the same ratchet this repo already recorded
  for a blanket `rho_b >= rho0` (`density2025.py` docstring).
- ACSPH has neither problem: it is incompressible (no acoustics) and its
  projected variable *is* its state.

So for delta-SPH the complementarity has to be built into the continuity
equation itself -- positive divergence at an air-reachable row booked as air
entering the gap, not as particle expansion -- and acoustically neutral. That
is a research item, not a clamp; the option was **removed** from the code
(no known-bad switch shipped). Open.

**Found on the way — detection bug (all schemes):** an isolated particle
(empty support) gets an identity renormalization fallback, `lambda = 1`, and
is classified as *bulk*. `modules/surfaceDetection/isolated.py`
(`detectIsolated`: exact, the trace `sum_j V_j (x_j - x_i).gradW_ij` is 0.0 only
for an empty support) is used by ACSPH's `vicinity` set. The shared detector
itself is unchanged -- fixing it there would change shifting and the Antuono
switch for every scheme; left as a decision (`OPEN_PROBLEMS.md` §8).

## 8. delta-SPH, second pass (2026-09-24)

### 8.1 What the CG solvers do differently (source, not memory)

- **omniSPH DFSPH** (`~/dev/omniSPH/simulation/fluidMechanics.cpp`): density
  is a fresh *summation* every step, boundary integral included (l. 5-21) --
  nothing is carried; the density solve is a one-sided constraint on **every**
  row, `p = max(p, 0)` (l. 316); the boundary pressure (MLS extrapolation) is
  clamped `>= 0` too, both where it is computed (l. 391) and where it enters
  the force (l. 464-465). The divergence solve is effectively inert (source
  term 0, 3 iterations).
- **SPlisHSPlasH** DFSPH: `densityAdv = max(densityAdv, 0)` and zeroed below
  20 (3D) / 7 (2D) neighbours; IISPH/DFSPH pressure `max(., 0)`. Its **WCSPH**
  is summation density with `density = max(density, density0)` on every row
  -- the exact floor that ratcheted delta-SPH in §7.5 v2. It is harmless there
  because summation density has no memory.
- So the CG methods never stick for two structural reasons: the constraint is
  unilateral (LCP, Batty's complementarity) everywhere, and density is
  re-derived from positions, so separation is never *booked* anywhere.

### 8.2 Literature (online, this pass)

- **SPHinXsys / Zhang, Fan, Zhang, Adams & Hu 2023** (arXiv:2310.11179, water
  entry/exit, WCSPH): wall pressure `p_d = p_j + rho_j r . max(0, (g - dv/dt).r)`
  (Eq. 2.11) -- the hydrostatic extrapolation is **one-sided**, never tension
  into a ceiling. Also Rezavand et al. 2022's density reinitialisation, every
  step, against summation density: `rho = rho_sum + max(0, rho - rho_sum) rho0/rho`
  (Eq. 2.8) -- bounds a continuity density below by its geometric value, so a
  stored deficit cannot outlive the configuration that produced it.
- **DualSPHysics** (`JSphCpu_mdbc.cpp`): pressure cloning is *not* clamped
  (`pressfinal = pghost + normforce*normpos`); english2025 matches it. Its
  guards are elsewhere: `max(RhopZero, .)` when the kernel sum < 0.1, unsubmerged
  boundary switched off (`BMODE_MDBC2OFF`, rho0), and fluid outside
  `[RhopOutMin, RhopOutMax]` is **deleted** -- the rho = 0.21 droplets of §7.5
  would simply be removed there. Free slip copies the ghost velocity (no
  reflection).
- **delta+-SPH group** (Lyu, Sun et al. 2021, Appl. Ocean Res. 117 102938;
  water entry/exit with TIC 2023): their concern is the opposite failure --
  spurious *voids* at a wall under legitimate negative pressure -- fixed with
  PST + TIC applied at boundaries. A fix here must keep tension where no air
  can reach.

### 8.3 Mechanism in delta-SPH: the reflected normal velocity

`freeSlip` reflects the fluid's normal velocity (`u_g = u_body + w_t - w_n`,
`DELTASPH_VALIDATION_PLAN.md` 5.7 -- right for *approaching* fluid). For a
*receding* row the wall particle then moves the opposite way, so the wall pair
books the separation as expansion at twice the relative speed in `drho/dt`:
the continuity-form bilateral constraint, and the source of §7.5's stored
deficit. Batty's complementarity in kinematic form is: reflect while
approaching, copy while receding (`normal = |w.n| n`).

Diagnostic switches (default off, bit-identical):
`velocity._UNILATERAL_NORMAL` and `english2025._ONESIDED_HYDRO`
(`'all'` / `'vicinity'`); `scripts/probe_contactLine.py --unilateralVel
[vicinity] --oneSidedHydro [vicinity]`.

### 8.4 Harness fix first

The delta-SPH toys (run 2026-09-23, not written up) were invalid: the
`hydrostaticColumn`-based WC toy ran `semiImplicitEuler` with physical nu = 0
(DFSPH case settings), and even the **at-rest column blew up at t = 0.1**.
The probe now runs the WC toys in sloshingTank's configuration (alpha = 0.02,
Michel shifting, `symplecticEuler`); column/sealed are then stable to t = 0.5.
(dt stays at the adaptive hook's value, Courant `c0 dt/dx` ~ 0.8 -- stable
with symplecticEuler; the pinned `targetDt` is overridden per step.)
Side finding: the runner reported `diverged=False` for runs at vmax 5e4 with
rho = 0 rows -- its divergence check misses a bounded-NaN-free blowup.

### 8.5 Toys (delta-SPH nx=32, fixed harness)

| drop, t = 0.3 | fallRatio | fluid p | contact |
|---|---|---|---|
| baseline | 0.67 | -23 … +13 | held, wall pulls 0.4-1.3 g |
| unilateral velocity (all) | 0.88 | ±0.6 | released by t 0.15 |
| one-sided hydrostatic (all) | 0.67 | -13 … +13 | held |
| **both (all)** | **0.984** (free fall reads ~0.97) | **±0.025** | released at once |
| both (vicinity) | 0.85 | -9 … +1.6 early, ±0.9 at end | peels from the contact lines inward |

Controls: `column` unchanged in all modes. `sealed` with **'all'**: pressure
ratchets monotonically, pfMin 0 -> +38.6, median rho 1.00 -> 1.02 by t = 0.5
(baseline oscillates ±7) -- the unilateral velocity rectifies acoustics at a
wall (compression booked, rarefaction dropped), the continuity-form twin of
§7.5 v2. With **'vicinity'** `sealed` is **bit-identical** to the baseline
(846/846 records): no free surface, empty set. So, as for ACSPH, the rule is
the complementarity restricted to where air can reach. The velocity half does
most of the work in delta-SPH (the fluid's own pressure is a state function
of the booked density); the hydrostatic half alone does nothing, but is
needed for the last ~10 %.

### 8.6 sloshingTank (case defaults, t = 6.5, both switches 'vicinity')

`scripts/probe_contactLine.py --toy sloshing --scheme default --nx 0
--unilateralVel vicinity --oneSidedHydro vicinity` (1 h 27 min, video in
`out_contactLine/sloshing_default_nx0_cavoff_uniVic`).

- **Survives the full 6.5 s** (v1/v3 of §7.5 diverged at t = 2.90); median
  density 1.001-1.002 throughout -- no v2-style ratchet.
- **Ceiling riders down**: mean 0.5-2.6 / max <= 6 through t = 5.5 (baseline
  1.7-17.5 / max 25); last second 5-6.7 / max 10 (baseline 6-9.3 / max 16).
- **But impacts 2 and 3 read 3-4x hot at Sensor 1**, above the TC10 measured
  band (2.2-13.1 kPa):

| sensor peak, 0.05 s mean (kPa) | 1.5-2.5 | 3.2-4.2 | 5.0-5.8 |
|---|---|---|---|
| baseline | 4.8 | 5.4 | 4.6 |
| unilateral (vicinity) | 5.3 | **15.6** | **20.9** |

  (0.01 s window: 13.5 / 9.6 baseline vs 56 / 39; raw steps > 20 kPa: 45 vs
  331.) The first impact is unchanged, so the effect builds after the first
  slam. Also a harder transient at t ~ 2.5 (vmax 9.0 vs 3.8, fluid p -97 vs
  -19, min rho 0.76 vs 0.94).

Not yet separated: whether the hot impacts come from the velocity half
(dropped rarefaction at a wall in the vicinity, i.e. §8.5's sealed-box
rectification, local to the contact layer) or the hydrostatic half, and
where the t ~ 2.5 transient sits. Next: the two halves separately on
sloshingTank (velocity-only / hydro-only), and the frames at t ~ 2.5 / 3.5 /
5.3. Status: **promising on sticking, not shippable** -- the switches stay
diagnostic, default off.

## 9. Retraction (2026-09-24): the fixes spray fliers, the checks missed it

The user reviewed the §8.6 sloshingTank video: every wall interaction with the
unilateral wall ('vicinity') launches high-speed fliers through the entire
domain; frames at t = 3.93 show ~20 isolated droplets (baseline 2-3).
(Correction: the video's density colour range -0.48 … 2.48 includes wall /
ghost rows; the logged *fluid* minimum is 0.76 vs 0.80 baseline -- no negative
fluid density.) The
Sensor-1 overshoots of §8.6 are those fliers hitting the wall, not a wall
pressure build-up.

Re-checking the ACSPH `vicinity` videos (§7.3) against that: **the same
disease**. nx=70 (the "reaches t* 9.91" run): after the far-wall run-up
(t ~ 1.26 s) a cloud of isolated single particles fills the air space, ~80-100
by t ~ 1.8-2.2 s, ballistic, some at the ceiling. nx=24: 3-6 fliers vs 2 with
`off`. §7.3's metrics (probe peaks, ceiling riders, penetration, survival,
worst-row forensics) contain no measure of isolated particles, so "validated"
was not supported. §7.4's verdict on the *mechanism* (bilateral wall) stands;
the *fix* is not shown to be acceptable for either scheme.

Required before any further claim:
1. A flier metric streamed every step: fluid rows with an (almost) empty
   fluid support (fragment size <= 3 particles), their count, max speed and
   height above the connected bulk; plus min density (negative = invalid).
2. The same metric on the `off` baselines (delta-SPH sloshingTank and Marrone
   3.1 production, ACSPH where it survives) -- physical spray is attached
   sheets/droplets; numerical fliers are single particles.
3. Where the fliers originate (step / row / wall contact) before choosing a fix.

### 9.1 Flier metric and baselines (2026-09-24)

`FlierTracker` in `scripts/probe_contactLine.py`: from the step's own
adjacency on the GPU (`countNeighborsWarp`; `FluidToBoundary` gates on the
query kind too and returns 0 for fluid rows, so nW = AllToAll - FluidToFluid;
both counts include the row itself). A fluid row with <= 2 fluid neighbours is
**wall-sparse** (nW > 0: the population the contact fix targets) or a **free
flier** (nW = 0: the population a bad fix creates). Each new free flier logs
`sinceWall` (time since its last wall neighbour). Calibrated on ACSPH nx=24:
off = 2-3 wall-sparse (the 2 pinned ceiling riders), 0 free; vicinity = 0
wall-sparse, 2-5 free (5 distinct UIDs, the run-up tip released at t ~ 1.0 s,
1-4 m/s ~ sqrt(gH) -- plausibly physical at this resolution).

| run | free fliers | wall-sparse | min fluid rho |
|---|---|---|---|
| delta-SPH sloshingTank off, t <= 4.2 | **0 every step** | <= 7 | 0.80 |
| delta-SPH sloshingTank unilateral (vicinity) | mean 10 -> 25, max 31, rising; <= 8.8 m/s | <= 8 | 0.76 |
| delta-SPH Marrone 3.1 nx=70 off, t <= 2.45 | **0 every step** | <= 9 | 0.96 |
| ACSPH Marrone 3.1 nx=70 off | 0 (diverges t ~ 0.62-0.69) | <= 1 | 1 |
| ACSPH Marrone 3.1 nx=70 vicinity, t <= 2.45 | mean 22 (t 1.0) -> 96 (t 2.25), max 105; up to 23.7 m/s | <= 5 | 1 |

Origins: delta-SPH unilateral -- 357 births / 63 particles, 150 within 0.02 s
and 127 within 0.2 s of their last wall contact: launched off the walls.
ACSPH vicinity -- 911 births / 134 particles, **432 never touched a wall**:
the ACSPH projection clamps p >= 0 on *every* surface-vicinity fluid row, not
only at walls, which removes the free surface's own cohesion -- spray breeds
from the jet itself, not just from wall contacts.

Verdict: neither fix is acceptable; the gate for any future contact-line fix
is free fliers at the `off` baseline (0 here) with wall-sparse reduced.
Videos (time-aligned, `scripts/out_contactLine/compare/`):
`sloshing_t4.2_off_vs_unilateral.mp4`,
`marrone31_nx70_3way_deltaSPHoff_ACSPHoff_ACSPHvicinity.mp4`.

### 9.2 Why the ACSPH fliers accumulate: an elastic contact (user observation)

User, from the nx=70 video: fliers fly back toward the fluid, dent a crater
as if merging, and are flung back out -- so the flier count only builds.
The births confirm it (911 births / 134 particles): median 6 re-ejections per
particle (max 23), median 0.046 s between them, successive ejection speeds
unchanged (median +0.01 m/s, 51 % faster -- restitution ~ 1), 72 % ejected
with p > 0 (median 151).

Mechanism: the flier (isolated) and the surface rows it hits are both in the
projected set, p >= 0. Approach: div v < 0, pressure builds and repels --
fine. Separation: div v > 0 would give the negative pressure that decelerates
the rebound (the inelastic, merging half of the contact); the clamp removes
it and the stored positive pressure is released like a spring. Droplet
coalescence is inelastic; this is a hard elastic collision, so no flier ever
merges back. Any complementarity-type fix must keep the dissipative
(separation-resisting) half *between fluid rows* -- Batty's unilateral
condition is for the solid wall only; applying it fluid-fluid (vicinity fluid
rows, isolated rows) is what breaks coalescence.

delta-SPH: the unilateral wall (§8) is worse than `off` on every measure;
both diagnostic switches were removed from `src` and the probe (never
committed). §8 stays as the record.

### 9.3 ACSPH fluid-solid-only unilateral force (`cavitationProjection = 'wall'`)

Minimal Batty-style change, one difference from `off`: the fluid-solid pair
terms of the pressure force use `max(0, .)` on both sides,
`-sum_wall V_j (p_i^+ + p_w^+) gradW`; the pressure solve, the divergence, the
wall extrapolation and every fluid-fluid pair are untouched (omniSPH's
boundary force does the same, fluidMechanics.cpp:464-465). Exact by linearity
of Eq. (25)'s `(p_i + p_j)`: `F' = A(p) + A((p^+ - p) 1_wall) + (p_i^+ - p_i)
A(1_wall)` (`_unilateralWallForce`; 4 unit tests: exact without tension, a
wall cannot pull, rows without a wall neighbour untouched, no-op without walls).

| check | result |
|---|---|
| `drop` (t = 0.2) | **fallRatio 1.02** (free fall ~0.97; off 0.64, vicinity 0.78); wall hold ~0, fluid p bounded (-8.7 transient -> ±1.4) |
| `column` | unchanged (slope ratio 1.00, same residual); vmax 0.012 vs 0.009 |
| `sealed` | pressure level grows exponentially (7 -> 1720 by t 0.44), collapse t ~ 0.47 |
| Marrone 3.1 nx=24 | **blows up t ~ 0.48**, before the far-wall impact |

`sealed` is a **pre-existing ACSPH instability**, not the fix: `off` started
from a uniform p = +10 grows identically (25 -> 1290, collapse t ~ 0.44). A
closed box's pressure is defined up to a constant and a positive level is
unstable (off without offset drifts *negative* and saturates ~ -16). 'wall'
only triggers it, by removing the bottom wall's legitimate -rho g H.
Separate open item.

Marrone nx=24 is the fix's own failure, and it is the inconsistency, measured:
from t ~ 0.40 the most negative row sits against a wall (1.1-1.6 dx, ~20 wall
neighbours, p = -12), then -2e3 -> -1.8e6 within 0.02 s. The force is
unilateral but the pressure equation is not: the divergence still carries the
wall pairs with the mirrored wall velocity, so a separating row keeps a
positive divergence the wall can no longer act on, the pseudo-time integrates
`-k1 div v` without bound, and the resulting p_i acts through the (unclamped)
fluid-fluid pairs. Batty's complementarity pairs the pressure sign with the
*kinematic* condition, `0 <= p ⊥ (u - v_s).n >= 0`; a unilateral force needs
the unilateral kinematic row with it. That is the one additional difference
the data argues for: the wall-pair contribution to the fluid row's divergence
only while it compresses (a separating wall pair contributes nothing).

### 9.4 Both halves of Batty's complementarity (`'wall'` + `_unilateralWallDivergence`)

Added the kinematic partner the §9.3 failure argued for: the wall pairs'
share of a fluid row's divergence (Difference form), split out by linearity
`divWall_i = D(v 1_wall)_i - v_i . G_i`, `G_i = sum_wall V_j gradW_ij`, kept
only while compressive (`min(divWall, 0)`), per row. 3 unit tests (the split
is exact away from the wall, an approaching wall is unchanged, a rigidly
separating row has no wall divergence); 46/46 ACSPH tests pass.

| check (video in `out_contactLine/*_bat`) | force only (§9.3) | force + divergence |
|---|---|---|
| `drop` fallRatio t = 0.2 | 1.02 | 0.90, fluid p within ±0.4, wall hold 0 |
| `column` vmax / dispMax / residual | 0.012 / 0.010 / 7e-4 | **0.07-0.27 / 0.045 / 5e-3-4e-2**, pfMax up to 8.1 (off 4.86) |
| `sealed` | gauge growth, collapse t ~ 0.47 | level +609 by t 0.06, **NaN t = 0.13** |
| Marrone 3.1 nx=24 | blowup t ~ 0.48 (p_i runaway at a wall) | **blowup t ~ 0.34: the whole fluid body lifts off the floor** |

Mechanism, from the forensics and the frames: the clamp is a **rectifier of
discretisation noise at every wall**. The SPH wall-pair divergence
`sum_wall V_j (v_w - v_i).gradW` is not the normal relative velocity: with a
disordered row, each pair's gradW has a component along the wall, so fluid
sliding along a floor (or a column at rest, jiggling) produces wall-pair
divergence of both signs. `min(., 0)` keeps only the compressive half, the
pseudo-time turns it into floor pressure above hydrostatic (t 0.14-0.16:
floor row, 22 wall neighbours, p +34 -> +60, v 3.4 -> 8.6), and since
separation is no longer resisted the fluid is pushed off the floor and
floats (frames t 0.22-0.33). In the sealed box the same bias violates the
pure-Neumann compatibility condition (net divergence over the box no longer
zero), so the pressure level grows without bound.

Batty's grid form does not have this: his kinematic row is the normal face
velocity `(u - v_s).n` exactly, no tangential leakage. The SPH analogue would
have to clamp a *normal relative velocity* per wall contact, not the
kernel-weighted wall-pair divergence; and a sealed container stays
incompatible with a unilateral wall unless the fluid can actually open a
(vacuum) gap. Neither is a small change. **Status: `'wall'` is not
acceptable in either form; the contact-line problem is open for ACSPH as for
delta-SPH.** The mechanism verdict (bilateral wall) stands.

## 10. Batty's wall term in detail, its follow-ups, and their test scenes (2026-09-25)

Read in full: Batty, Bertails & Bridson 2007 (`literature/batty2007_…`, §1.1,
§2, §4); Chentanez & Müller, SCA 2011 / TVCG 2012, "A Multigrid Fluid Pressure
Solver Handling Separating Solid Boundary Conditions" (the direct follow-up,
[pdf](https://matthias-research.github.io/pages/publications/separatingBoundaries.pdf));
Inglis, Eckert, Gregson & Thuerey 2017, "Primal-Dual Optimization for Fluids"
§5 ([arXiv 1611.03677](https://arxiv.org/abs/1611.03677)). Andersen, Niebe &
Erleben 2017 (GPU LCP solver) is solver speed only; its pdf link is dead.

### 10.1 What the term actually is

- Batty §2: the pressure projection **is** a kinetic-energy minimisation;
  pressure is the Lagrange multiplier of `div u = 0` and of `u.n = v_s.n`.
  §4 adds one thing: the bound `p >= 0` "on solid boundaries" (the pressure
  unknowns of the cells at the solid). The kinematic half of Eq. 15,
  `(u - v_s).n >= 0`, is **never imposed**: it is the KKT stationarity of
  the bound-constrained QP ("the complementarity is automatically enforced
  for us by the KKT conditions"). Concretely, at a bound-active row the
  row's *total* divergence residual is allowed one sign (expansion); at an
  inactive row it is zero as before.
- Chentanez & Müller make it concrete: `p_min = 0` only "if (i,j,k) is
  inside a solid", `-inf` everywhere else, i.e. the bound sits on the
  **solid-side** pressure unknowns next to liquid (after trying the exact
  per-edge interface version and finding it "hardly distinguishable").
  Fluid rows are unbounded. Solved by projected Gauss-Seidel,
  `p = max(p_min, GS update)`, as a multigrid smoother. Their second
  ingredient: a solid cell whose pressure lands at 0 is **re-labelled air**
  (level set set positive) so the neighbouring liquid sees a free surface
  (Dirichlet p = 0) there. "Only with this modification does the liquid
  peel-off freely from solids."
- Inglis et al.: same structure as an explicit **active set per wall face**:
  non-separating = Neumann (bilateral), separating = Dirichlet p = 0 (the
  face is treated as free surface, velocity left alone). They start from
  the naive velocity rule (zero the normal velocity only where `u.n < 0`)
  and report it "breaks down when solving up to finite accuracy … small
  separating motions where the liquid should only be standing still or
  moving tangentially … hydrostatic cases". Their fix is hysteresis on the
  classification tied to the CG tolerance (a solver-accuracy threshold, not
  a physical one).

### 10.2 What this says about §9.3 / §9.4

- §9.4's `_unilateralWallDivergence` (keep the wall-pair divergence only
  while compressive) is the **Foster & Fedkiw 2001 rule**, not Batty's:
  "if fluid velocities are found to be separating … that separation
  velocity is enforced". Batty §4 rejects it because "it becomes unstable …
  with oblique boundaries" and "completely fails for closed or nearly
  closed fluid-filled containers"; Inglis reproduces the hydrostatic
  failure. Our §9.4 results are exactly those two failures (column at rest
  lifts/floats; sealed box NaN). The §9.4 closing suggestion (a per-contact
  *normal-relative-velocity* criterion) is still a velocity-sign rule and
  falls under the same objection. **Withdrawn.**
- §9.3 (force-only) is half of an active set: the wall stops pulling, but
  the contact stays in the continuity row, so it is neither Neumann nor
  Dirichlet. All three papers switch a contact **as a whole**: it is
  either in both momentum and continuity (bilateral) or in neither (air).
- The switching variable in Batty and C&M is the **pressure sign** (the
  multiplier), not the velocity sign. At hydrostatic rest a floor's
  multiplier is `+rho g H` and is never released, so resting fluid cannot
  be rectified off the floor. That is why the method is robust in exactly
  the cases §9.4 failed.
- §9.2's fliers: Batty's §4 motivation is the analogy with **inelastic**
  rigid contact (KE minimisation). A projection that minimises KE cannot
  produce the restitution ~1 re-ejections measured in §9.1-9.2. So those
  measurements also say the 'vicinity' iteration was not behaving as a KE
  projection, and not only that the set was too large.
- Caveat on transfer: KKT does the kinematic half only if the pressure
  force is the negative adjoint of the divergence. For fluid-fluid pairs
  it is: `-sum_j V_j (p_i + p_j) gradW_ij` against the Difference
  divergence `sum_j V_j (v_j - v_i).gradW_ij` under the V-weighted inner
  product (checked by index swap). Through the wall it is **not**: the
  Eq. 61 wall pressure is a closure of the fluid pressure plus a
  hydrostatic term, not an independent multiplier with its own row. So
  ACSPH's projected pseudo-time is a projected Uzawa / Arrow-Hurwicz
  iteration on a proper QP in the bulk and an unproven iteration at the
  wall.

### 10.3 Two faithful SPH translations (not yet implemented)

- **A. Batty literal, cheap.** Bound set = wall rows + fluid rows with at
  least one wall neighbour (the SPH "cells at the solid boundary"; no
  tunable). Keep the full bilateral divergence and force, and project
  `p >= 0` in the existing loop (`project`), letting the pseudo-time's
  fixed point supply the kinematic half. This differs from 'vicinity'
  (which also bounded non-wall surface rows, breaking fluid-fluid
  coalescence, §9.2: 432/911 flier births never touched a wall) and from
  the wall-only clamp of §7 (which bounded only the closure, so the
  tension moved into p_i). It is a new `cavitationProjection` set. Known
  cost: it also removes tension at a submerged receding wall; Batty
  accepts that explicitly ("we rule out suction").
- **B. C&M / Inglis active set, structurally complete.** Wall pressure as
  its own unknown with its own continuity row (Band et al. 2018 pressure
  boundaries, `literature/band2018pb_…`, is the SPH precedent: boundary
  samples carry a PPE row and are clamped `>= 0`), `p_w >= 0` projected;
  a wall particle at its bound is air for its fluid neighbours in **both**
  the force and the divergence. Fluid rows unbounded. More work: an extra
  row per wall particle, and the Eq. 61 closure is replaced at the wall.

### 10.4 Their test scenes (all judged visually, sticky vs separating)

None of the three papers has a quantitative physics check. Batty: Fig. 6
only, plus the analytic hydrostatic claim for the variational solve (§2.1).
C&M: figures plus a timing table (LCP ~12 % slower). Inglis: figures plus
solver iteration counts. The scenes, with the observable we would attach:

| # | scene (source) | what separation must show | our observable |
|---|---|---|---|
| S1 | 2D ball of water thrown against a side wall (Batty Fig. 6; C&M Fig. 5, 3D) | fluid peels off the wall instead of creeping up it | wall-attached mass above the impact height vs t; peak rebound speed (restitution, should be ~0 by the inelastic-contact analogy); flier count |
| S2 | 2D ball moving diagonally up-right inside a circle (C&M Fig. 2) | no creeping along the curved wall/ceiling | mass in contact with the upper arc vs t; tests the curved-wall normal |
| S3 | dam break in an axis-aligned box with a ceiling (C&M Fig. 4; Inglis Fig. 14 "sticking to the ceiling and getting stuck in corners") | no ceiling film, no corner pockets | ceiling riders (probe), corner mass; flier count |
| S4 | dam break in a 45°-rotated box (C&M Fig. 3) | same, oblique walls | as S3 |
| S5 | dam break in a sphere/circle (C&M Fig. 1) | peels off the curved container | as S2 |
| C1 | fluid at rest in an **oblique** container (Batty §1.1 / §4, the Foster-Fedkiw failure; Inglis §5) | **must not change anything**: no spurious currents or lift-off | max |v|, KE vs t, floor contact fraction (vs 'off') |
| C2 | closed / nearly closed fluid-filled container (Batty §4) | **must not change anything** | pressure level / NaN (sealed toy; note OPEN_PROBLEMS 9 is pre-existing) |

Mapping onto what exists: S3 ~ `--toy marrone31` (plus the `drop` toy for
the ceiling), C2 = `sealed`, the axis-aligned half of C1 = `column`.
Missing: S1 (a variant of `cases/impact.py`'s one-body form with a wall
in the box, or the probe's `peel` with an initial velocity), S2/S5 (need a
circular container), S4 and the oblique half of C1 (a rotated box; the
sloshingTank's tilted-tank frame or a rotated SDF container). The C-rows
are the gates §9.4 failed; the S-rows are the gates §9.1's flier metric
already covers only partly.

### 10.5 Other work in this direction (SPH-side survey, 2026-09-25)

The question: is there a published, implementable SPH method for wall
separation, as opposed to adapting Batty (research)? Findings, beyond §7.1/§8.1-8.2:

- **Dual-time SPH** (Ramachandran, Muta & Ramakrishna, arXiv 1904.00861):
  the closest relative of our solver (AC pressure in pseudo-time, Adami
  2012 walls). Its entire treatment: "we ensure that the pressure is always
  positive on the solid walls to prevent particles from sticking to them",
  citing **Hughes & Graham 2010** (J. Hydraul. Res. 48:105-117), which is
  where the wall clamp comes from. EDAC / PySPH `ClampWallPressure` are the same.
  Judged visually on a dam break without a ceiling. Our §7 found this exact clamp does not release the
  contact in our ACSPH (tension moves into the fluid row's p_i). Nobody has
  explained that discrepancy yet. PySPH is open source, so it can be checked the
  way the diffSPH harness checked delta-SPH (Part 8).
- **De Courcy 2024** (our ACSPH paper): no sticking treatment at all;
  its dam break is free-slip, no clamp.
- **Globally unilateral solvers** (IISPH/DFSPH/PBF; Band et al. 2018
  pressure boundaries, local pdf): `p >= 0` on every row, boundary pressure
  as its own clamped unknown. No sticking, but no tension anywhere; the
  graphics codes accept that. This is our `'fluid'` bracket.
- **Constraint Fluids** (Bodin, Lacoursière & Servin 2012, local pdf): SPH
  with bilateral density constraints and **unilateral boundary constraints only**,
  solved as a mixed LCP by projected Gauss-Seidel (`z = max(0, -q/d)`). This is the one
  published SPH form of Batty's structure, but it needs its own boundary model
  (non-penetration constraint / boundary density field) and its SPOOK
  integrator. A different solver, not a plug-in.
- **Repulsive boundary forces** (Monaghan & Kajtar 2009): unilateral by
  construction, walls absent from the continuity equation; gives up the near-wall pressure accuracy the
  mDBC/Adami walls exist for (sloshing sensors).
- **One-sided hydrostatic extrapolation** (SPHinXsys, Zhang et al. 2023
  Eq. 2.11): already covered in §8.2, tried in delta-SPH.
- Not relevant on inspection: Zhang et al. 2026 well-balanced WCSPH
  (arXiv 2608.22269), with no wall-separation content.

Verdict: no SPH paper gives a principled, drop-in unilateral wall for an
AC/WC scheme with extrapolated ghost walls. The published practice is the
Hughes & Graham wall clamp, whose failure in our solver is unexplained, or a
globally unilateral solver.

## 11. Experiments: Batty set A (`'contact'`) and the dual-time SPH wall (2026-09-25)

Code: `acParams.cavitationProjection = 'contact'` (every wall row + every
fluid row with a wall neighbour, `_wallAdjacent`, exact, no threshold; p >= 0
projected in the existing loop; divergence and force stay two-sided). Tests: exact
set vs brute force, no-op without walls (48/48 ACSPH tests pass). Probe:
`--cav contact`; `--wallMode dtsph` = dual-time SPH's wall as in their source
(`gitlab.com/pypr/dtsph`, `dtsph.py`: `SetPressureSolid` = Adami Eq. 27 incl.
hydrostatic term then `max(0, p_w)`, every pseudo-iteration; `PredictPressure`
sees the wall's own velocity, 0, not the mirror); `--wallMode dtsphDiv` = same
but the static wall velocity only in the continuity row (viscosity keeps the
free-slip mirror). Logs `scripts/out_contactLine/{c,p,q}_*.log`, videos in
the run folders. Toys nx=32 `--paperAC`.

| check | off | contact (Batty A) | dtsph (faithful) | dtsphDiv |
|---|---|---|---|---|
| drop fallRatio t=0.2 (free fall ~0.97) / stuck | 0.64 / 6 % | **0.88 / 0** | 0.74 / 2 % | 0.73 / 2 % |
| column at rest vmax / dispMax / slope | 0.027 / 0.006 / 1.001 | 0.053 / 0.011 / 1.001 | 0.051 / 0.010 / 1.002 | 0.048 / 0.010 / 1.001 |
| sealed box | bounded, level drifts to ~-15 | level grows 0 -> 2000, vmax 32 by t=0.5 | NaN-scale by t=0.39 | NaN-scale by t=0.33 |
| Marrone nx24 to t=2.1: fliers (uids/births) | 0/0 | 3/10 | 0/0 | 0/0 |
| Marrone nx24: ceiling riders max / mean | 7 / 2.3 | 5 / 1.7 | 1 / 0.2 | 10 / 1.7 |
| Marrone nx24: KE t=0.6, peak probe p0* | 0.82, 4.74 | 0.83, 5.01 | **0.53, 1.75 (late t* 3.25)** | 0.82, 3.17 |
| Marrone nx70 | diverges t 0.62 (run-up) | passes run-up; fliers from t 0.69; **diverges t 1.26 (t* 5.1)** | -- | -- |

Findings:

- **Column at rest is clean for every clamp-type variant** (none rectifies the
  floor), unlike §9.4's velocity-sign clamp (vmax 0.31, slope 0.95). The
  literature's point (§10.2: switch on pressure, not velocity) holds.
- **The sealed box fails for all of them.** Clamping the walls removes the
  bottom wall's legitimate tension, the level has to sit positive, and the
  pre-existing positive-level instability (OPEN_PROBLEMS 9) runs away. Batty's QP
  has no such instability; ours does. Not fixable at the wall.
- **dtsph "works" on Marrone only by drag:** zeroing the wall velocity
  also enters ACSPH's viscous term (no-slip against a free-slip case): KE
  -35 % at t=0.6, impact late and 1/3 as strong. dtsphDiv (continuity only)
  restores the flow (KE matches off) and then does **not** remove ceiling
  riders (max 10). The dual-time SPH authors' dam break (open 4 m tank, 1x2 column,
  t <= 1 s, alpha = 0.25 artificial viscosity seeing the wall at rest)
  never tests a ceiling. **The Hughes & Graham clamp does not solve the
  contact line in our ACSPH, reproduced faithfully.**
- **contact** is the best of the family on the toys and survives nx24,
  but produces wall-launched fliers (nx24 3, nx70 16 births before the
  blowup) and diverges at nx70, t 1.26.

nx70 blowup forensics: 5 steps of a thin filament row far from walls
(nF 6, nW 0, 14.5 dx from a wall, lambda 0.001-0.003) going -84 -> -130,
then one step to -8.6e11 on an isolated row (nF = nW = 0). Same family as the
thin-filament / truncation-artifact items (OPEN_PROBLEMS 1), reached because
contact lets the run-up past t 0.62.

### 11.1 Fliers carry a frozen pressure (answers "how can fliers be re-ejected if fluid-fluid is unchanged")

Isolated rows cannot change their pressure: in Eq. (23) `-k1 rho div v` and
the diffusion are both empty sums, so p stays at its last contact value.
nx70 contact, uid 167: born t 0.703 with p = 0 (left the clamped band), re-born t 0.771 /
0.793 / 0.822 with p = +297 / +255 / +306; uid 83 p = +358 at t 0.99. A
particle that builds positive pressure against the wall or the clamped
sheet keeps it in flight and discharges it through `(p_i + p_j)` on the
next contact: stored elastic energy, matching §9.2's restitution ~1. Two
routes, not yet separated per event: (a) the partner rows are in the clamped
set (the band within one kernel radius of the walls); (b) the frozen p of the
flier itself. Also the flier metric's "birth" = entering nF <= 2 with nW = 0, which
filament particles can cross repeatedly without being thrown (nx24 uid 28, births with p < 0).

Physically exact candidate for (b): Dirichlet p = 0 on exactly isolated
rows (`detectIsolated`, no threshold; an isolated droplet is surrounded by
air). Does not cover rows with 1-2 neighbours. Untested. Per-event
tracking (the uid's neighbours' clamp membership, own p, approach vs departure
normal speed) would separate (a) from (b).

### 11.2 p = 0 on isolated rows, and per-contact flier traces (2026-09-25)

`acParams.isolatedZeroPressure` (dambreak `acIsolatedZeroPressure`, probe
`--isoZero`): Dirichlet p = 0 on exactly isolated fluid rows
(`detectIsolated`), applied in `project`. The adjacency is fixed per step, so it
acts on nothing within the step and only removes the pressure carried into the next
contact. No-op test (periodic box, bit-identical); 49/49 ACSPH tests pass.
Probe trace: `FlierTracker` follows every UID ever born each step
(`trace.json`); `scripts/analyze_flierTrace.py` splits it into contacts
(runs of nF > 2) and reports carried-in p, approach/departure normal speed
relative to the fluid neighbours (restitution e), peak p, and the fraction of
partners inside the clamped wall-adjacent band.

| Marrone 3.1 | tEnd | flier uids / births | free vmax | ceiling max / mean | penetration | maxV |
|---|---|---|---|---|---|---|
| nx24 off | 2.10 | 0 / 0 | - | 7 / 2.3 | 2 | 4.8 |
| nx24 contact | 2.10 | 3 / 10 | 5.0 | 5 / 1.7 | 2 | 5.1 |
| nx24 contact + iso0 | 2.10 | 3 / 10 (identical) | 5.4 | 5 / 1.7 | 2 | 5.4 |
| nx70 off | **0.62 diverged** | 0 | - | 0 | 0 | 6.4 |
| nx70 contact | **1.26 diverged** | 3528 (the blowup) | 2e11 | 58 | 730 | 2e11 |
| nx70 contact + iso0 | **2.45 (t* 9.9), finished** | 14 / 58 | 6.4 | 12 / 2.7 | 2 | 13.2 (one spike) |

For reference, nx70 vicinity (§9.1): 22 -> 96 fliers, up to 23.7 m/s.
Frames (nx70 contact + iso0, t 1.26 and 1.95): run-up sheet, then a plunging
breaker with an entrapped cavity; ~8-10 slow droplets in the air, no
domain-wide spray, nothing on the ceiling. A saw-tooth pattern along the top of the thin
floor layer (left of the tank) is visible; not yet compared with off.

Per-contact traces (`contact`, no iso0):

- **nx70: the frozen pressure (route b).** Both finished rebounds arrived with
  a stored positive p (+385, +309), their partners were **not** in the
  clamped band (band 0.00), peak p +1120 / +1210, restitution 0.93 / 0.75. Removing
  the stored p (iso0) removes the rebound and the run survives; nothing
  else changed.
- **nx24: the clamped band (route a).** All 4 finished rebounds had partners
  mostly in the wall-adjacent band (0.67-0.98 of neighbours), the flier
  itself away from the wall (nW 0), pressure building to +500...+1365 during
  contact, restitution 0.39-1.57 (mean 1.03). Two arrived with p < 0, so it is not
  stored pressure. Clamped partners can push but not hold, so the contact is
  one-sided (repel only) and elastic. iso0 cannot touch this:
  these particles are never exactly isolated (nF 1-2 before contact), and the
  nx24 results are identical with and without it.

Samples are small (4 and 2 finished contacts); vn is sampled on the steps
either side of the contact.

Status: `contact` + `isolatedZeroPressure` is the first variant that runs
Marrone 3.1 nx70 to t* 9.9 without domain-wide spray. Still open: (1) the
clamped band's one-sided fluid-fluid contacts (route a: nx24 fliers,
restitution ~1); (2) the sealed box (OPEN_PROBLEMS 9, all clamp variants);
(3) no ceiling-rider reduction at nx24 (5 vs 7), and nx70 off has no baseline
past t 0.62 to compare against. Not validated against the Marrone/experimental
pressure (peak probe p0* 2.35 at nx70 vs 5.0 at nx24 and vicinity 3.87).

## 12. delta-SPH: lone-particle density reset + one-sided hydrostatic wall term (2026-09-25)

Transfer of §11.1 to delta-SPH. A fluid row with no fluid neighbour has empty
continuity and DDT sums, so it carries its density (hence EOS pressure). Two
switches, default off, diagnostic:

- `WeaklyCompressibleSPHConfig.loneDensityReset`: `rho = rho0` at the end
  of every step (`WeaklyCompressibleSystem.finalize`, after shifting) for fluid
  rows with no fluid neighbour, detected on the last stage's state with its own
  adjacency (`detectNoFluidNeighbours`: exact integer count via
  `countNeighborsWarp`; a first version using a linearity split of wall and
  fluid sums never read exactly 0 with wall neighbours, because of float
  rounding). Physical reading (user): mass is unchanged, a low density means
  the parcel is smeared out, and surface tension would pull it into a compact
  blob at rest.
- `WeaklyCompressibleSPHConfig.mdbcOneSidedHydrostatic`: english2025
  `P_b = P_g + max(0, rho0 (g - a_b) . relPos)` (SPHinXsys Eq. 2.11). Needed
  because for a lone particle the english2025 trust ramp already sets the
  ghost value to rho0 (nNbFluid < 4), so the wall carries only the hydrostatic
  increment, which is negative under a ceiling: the ceiling pulls.
- Tests (`tests/test_loneDensityReset.py`): detector vs brute force; the reset
  is bit-identical with no lone rows (delta-SPH column); the one-sided term only
  ever raises wall densities and leaves the floor unchanged. NB: it is not a
  run-level no-op in an open column, because side-wall ghosts sit ~0.008 off
  level, giving small negative increments.
- Probe `--loneReset`, `--oneSidedHydro`; runs `scripts/out_contactLine/
  {d,e}_*.log`, sloshing with trace + video.

sloshingTank, delta-SPH case defaults, t = 0-4.2 (Sensor 1 = peak of the
0.05 s moving mean; the TC10 measured band is 2.2-13.1 kPa):

| | baseline | loneReset | loneReset + oneSidedHydro |
|---|---|---|---|
| fliers (uids / births), free vmax | 0 / 0 | 3 / 6, 3.2 m/s | 1 / 4, 2.8 m/s |
| ceiling riders max / mean | 24 / 3.78 | 11 / 2.76 | 16 / 2.55 |
| wall-sparse mean | 1.69 | 0.88 | 0.68 |
| lone-at-wall rows (particle-steps) | 34 584 | 23 226 | 11 987 |
| maxV / min rho | 7.66 / 0.803 | 6.91 / 0.840 | 5.85 / 0.878 |
| Sensor 1 impact 1 / impact 2 (kPa) | 4.77 / 5.39 | 4.77 / 7.09 | 5.13 / 8.83 |
| raw steps > 20 kPa | 22 | 37 | 72 |

- Frames (t 2.80, 3.50, baseline vs loneReset): the baseline's line of fluid
  on the right half of the ceiling at t 3.5 is gone; the loneReset fliers
  are riders released from the ceiling and falling at the speed they had
  while sliding (born with p = 0 straight off a wall, <= 3.2 m/s), not
  launches. §8's unilateral wall produced up to 8.8 m/s and 31 fliers.
- The baseline carries 8 lone-at-wall particles for up to 0.6 s each, with
  pressure ringing at -43...+51 (OPEN_PROBLEMS 1's population).
- **Open: the second impact reads hotter**, +32 % with the reset, +64 % with
  both (and first impact +8 %, raw spikes 3x), still inside the TC10 band. One
  realisation each; the first impact is unchanged with the reset alone, so it may be
  flow divergence after the first slam, but the one-sided term also raises
  wall densities directly and Sensor 1 reads at a wall, so part may be
  measurement bias. §8.6 (which also had the one-sided term) saw the same
  direction, 3x. Needs: a second realisation (perturbed IC) and the sensor
  read with and without the one-sided term on the same state.

Toys (delta-SPH nx32): drop, both switches, fallRatio 0.673 = identical to
§8's one-sided-only run (a block has no lone particles, so the reset is inactive);
column with one-sided term identical to baseline; sealed box with one-sided
term: median rho +0.0025 by t 0.25, then a plateau at 1.003 (baseline drifts to
1.0007): a one-off level shift from removing the lid's legitimate tension
increment, not a ratchet (0.5 s only).

Verdict: `loneDensityReset` is the promising half: it halves the wall-sparse
population, removes the t 3.5 ceiling film, and releases riders as slow
falling droplets, at the cost of a second-impact reading +32 % (unexplained).
The one-sided term adds little on top for sticking and doubles the
impact-spike count; keep it off pending the sensor-bias check.

## 13. Realisation batch for the second-impact question (launched 2026-09-25 23:10)

Sensor-bias check on the existing runs (no new run): `sensorPressure` (δ-SPH)
is the Tait pressure of the single wall particle nearest the sensor, so it
goes through the mDBC closure; `sensorPressureProbe` is a fluid-side Gaussian
probe that does not. Impact 2 (0.05 s mean peak, kPa):

| | wall sensor | fluid probe |
|---|---|---|
| baseline | 5.39 | 5.07 |
| loneReset | 7.09 | 6.72 (+33 %) |
| loneReset + oneSidedHydro | 8.83 | 7.26 (+43 %) |

So the reset's +32 % is in the flow (the fluid probe shows it too), not a
measurement artefact; the one-sided term adds wall-reading bias on top
(the wall sensor is 22 % above the probe there, against ~6 % otherwise).

Open question: is the reset's impact-2 rise systematic or run-to-run
variability after the first slam? Batch `scripts/run_loneBatch.sh`
(sequential, restartable: finished runs are skipped on relaunch): seeded IC
jitter `--jitter 1e-3 --seed S` (uniform fluid-position offset of at most
1e-3 dx, new probe option), sloshingTank delta-SPH defaults, t = 0-4.2, trace
+ video. Order: baseline/loneReset seeds 1, 2, then loneReset+oneSidedHydro
seed 1, then baseline/loneReset seed 3 (~31 min each). Status
`scripts/out_contactLine/lone_batch.status`; results
`scripts/out_contactLine/lone_batch_summary.md` (`scripts/summarize_loneBatch.py`,
rewritten after every run; includes the three unperturbed §12 runs as a
realisation).

Decision rule: if loneReset's impact-2 fluid-probe values sit inside the
baseline's spread over realisations, the rise is variability and the reset
is a candidate for default-on (next gates: Marrone 3.1/3.4 delta-SPH, flier
count, frames). If loneReset is above every baseline realisation, the rise is
systematic and needs a mechanism before going further.

### 13.1 Results (2026-09-26): the pressure rise is variability; the fliers are not

Batch complete (7 runs, all to t = 4.2, none diverged; ALLDONE 02:44).
4 realisations each for baseline and loneReset (seeds 1-3 + the unperturbed §12
run), 2 for loneReset + oneSidedHydro. Mean [min, max] over realisations:

| | baseline | loneReset | loneReset + oneSidedHydro |
|---|---|---|---|
| S1 fluid probe, impact 2 (kPa) | 6.09 [4.92, 7.85] | 5.84 [3.96, 6.72] | 7.06 [6.86, 7.26] |
| S1 wall sensor, impact 2 (kPa) | 6.57 [5.39, 8.01] | 6.00 [4.36, 7.09] | 8.19 [7.56, 8.83] |
| S1 fluid probe, impact 1 (kPa) | 5.56 [4.50, 6.85] | 5.56 (identical per seed) | 4.86 [4.63, 5.09] |
| free fliers (uids) | 0.75 [0, 3] | 4.75 [1, 9] | 3.0 [1, 5] |
| ceiling riders, mean | 3.22 [2.61, 3.78] | 2.95 [2.76, 3.26] | 2.69 [2.55, 2.84] |
| wall-sparse, mean | 1.27 [0.46, 1.79] | 1.05 [0.84, 1.33] | 0.91 [0.68, 1.13] |
| lone-at-wall rows | 24 162 [14 725, 34 584] | 15 406 [11 394, 23 226] | 17 672 [11 987, 23 356] |

**Decision rule: inside.** loneReset's impact-2 probe values (3.96-6.72) all lie
inside the baseline's range (4.92-7.85), and its mean is lower. The §12 "+32 %"
came from comparing against the lowest baseline realisation (the unperturbed
one, 5.07). Impact 1 is bit-identical per seed with and without the reset
(nothing is lone before the first slam). The impact-2 pressure question is
closed as run-to-run variability. oneSidedHydro: the probe is inside the
baseline range, but the wall sensor at 8.83 is above every baseline value (n = 2).
That is consistent with §13's wall-reading bias, so keep it off.

**But §12's benefits shrink to within the baseline spread.** Ceiling riders
and wall-sparse means overlap the baseline range (baseline seed 3 has wall-sparse
0.46, below every loneReset run). Only lone-at-wall rows drop consistently, by
about 36 % on the mean, with overlapping ranges.

**Fliers are up systematically.** Every loneReset realisation has at least one
free flier (1, 3, 6, 9 uids); 3 of the 4 baselines have none. That fails the
§9.1 gate (free fliers at the `off` baseline). Character, from births.json
and frames:
- Not spray. Speeds are at most 4.1 m/s (about sqrt(gH)); 27 of 33 loneReset
  births are nF = 2 fragments (2-3 particle droplets, p slightly negative)
  and 6 are single particles. §8's unilateral wall gave 31 fliers at up to 8.8 m/s.
- 7 of the 19 flier uids had been reset (lone at a wall) before birth. These
  are §12's released riders. The other 12 never were: they are pieces broken
  off crest tips or run-up sheets. Examples: seed 2, t = 3.86, three particles
  at about 3 m/s from the plunging crest near x ≈ 0.2 that never touched a wall
  (the baseline crest at the same instant is intact, tip 2.1 m/s); seed 3,
  t = 4.006, one particle at 4.1 m/s off the left-wall run-up tip, where the
  baseline tip is at 3.45 m/s and still attached.
- Frames at t = 2.80 (seed 2): the runs are indistinguishable. At t = 3.50,
  both show the right-wall splash with a few detached clusters; loneReset has
  fewer ceiling dots and a longer detached sheet near the upper right. There is
  no domain-wide spray at any checked time (2.80, 2.83, 3.50, 3.86, 4.01, 4.03).

Verdict: the second-impact rise is variability, so it does not block the reset.
However, at n = 4 the reset has not shown its sticking benefit beyond
variability, and it raises the free-flier count from 3 to 19 uids in total
(4/4 vs 1/4 realisations). Not a default-on candidate yet.

Next: find the flier mechanism before the Marrone gates.
1. For the 12 never-reset flier uids: were they inside the support of a
   reset particle within ~0.1 s before birth? Hypothesis (unverified): a reset
   lone particle that re-joins with p = 0, instead of its carried p < 0, removes
   the tensile pull that would have kept a thin tip attached. That would make
   the reset itself the cause. The alternative is just a diverged flow, in
   which case more realisations would close the gap.
2. Decide whether the §9.1 gate should count slow (<= sqrt(gH)) 2-3 particle
   crest fragments as numerical fliers at this resolution. This is a gate
   change, so the user decides it; it should not be adjusted to make the reset pass.
3. Only then run Marrone 3.1/3.4 delta-SPH (flier count + frames).
