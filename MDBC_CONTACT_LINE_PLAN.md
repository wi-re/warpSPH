# mDBC contact-line suction — plan

Status: **open**, not started. Tracks `OPEN_PROBLEMS.md` §8. Background and all
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

- **Runner stall watchdog on simulated-time progress.** `stallDtSteps` only
  fires when dt sits exactly at `minDt`; ACSPH's [0.8, 1.2] step clamp lets dt
  hover just above it (one run sat frozen for 7 h). Port the scratchpad
  watcher's rule (sim time advanced < ε over a wall-clock window) into
  `runner.py`.
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
