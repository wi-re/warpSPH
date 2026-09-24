# mDBC contact-line suction — plan

Status: **step 1 done, fix implemented (opt-in), ACSPH validated** — see §7
(2026-09-23). Tracks `OPEN_PROBLEMS.md` §8. Background and all
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
