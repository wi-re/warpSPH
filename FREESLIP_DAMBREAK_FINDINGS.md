# FreeSlip dam-break comparison — findings (2026-09-21)

Marrone 3.1 (H=0.6 m, TANK_L=1.0 m, W=3.2196 m, column 1.2 m, G=9.81),
nx=70 (H/dx=42), c0Ratio=40 (c0=97.04, M=0.049), after the 2026-09-21 wall-BC
default flip (`72f938c`, `constant` → `freeSlip`). All outputs in
`scripts/out_deltaSPHMarrone_freeslip/` (gitignored).

## 1. ACSPH leg (freeSlip, nx=70, t→5.0): passed t*≈6, then stalled — stopped

**Good news — the original question is answered:** the run passed the pre-flip
divergence point. At step 7840 (t=1.534, **t*≈6.12**) the flow is coherent and
healthy: run-up jet against the right wall, density colorbar 0.9999987–1.000003
(exactly 1.0, as ACSPH guarantees), max visible velocity 3.89 m/s. The pre-flip
(constant-wall) run went maxV 6.1→183 at t*≈5.5 and non-finite at t*≈5.97.
**The wall-BC hypothesis for the t*≈6 divergence is supported: freeSlip does
not produce the non-finite divergence.**

**Bad news — the run is numerically stalled, so it was stopped (bg_8f966d6e):**

| step | t (s) | t* | dt |
|---|---|---|---|
| 4000 | 1.481 | ≈5.9 | 1.57e-4 (healthy) |
| 6000 | 1.534 | ≈6.1 | **1e-08 (floor)** |
| 7840 | 1.534 | ≈6.1 | 1e-08 |
| 11000 | 1.534 | ≈6.1 | 1e-08, 3.9 s/step |

dt collapsed between steps 4000–6000 — exactly at the wall impact — and sim
time has been frozen at t≈1.534 since. Why it can never finish: ACSPH's dt is
`min(maxDt, cflT·h/max(1,|v|)/ks, 0.125h²/ν/ks)` (accel constraint inactive in
this path; viscous term inert at ν=1e-6 → ~14 s). The **only** term that can
reach 1e-8 is the advective one: with h≈0.0101 m (2dx support, Wendland2,
kernelScale 1.897) and cflT=0.2, that requires **|v|_max ≈ 1e5 m/s**. The
visible max is 3.89 m/s → there is a hidden particle at ~100 km/s **outside
the plotted domain** pinning the whole run's dt at minDt=1e-8. At the floor,
reaching t=5.0 would take ~3.5e8 steps ≈ **157 days**.

Two side findings from this:

- **A single runaway particle pins the entire run's dt** (global-max in the dt
  computation, no outlier isolation). That is a code-robustness issue in its
  own right.
- The in-memory trajectory (P1/P2/KE vs t to t=1.534) was lost with the
  process (npz is written at run end; `store=False`); the 550 PNG frames
  remain under `artificialCompressible_nx70_c40_pst-on_run/
  3-dambreak_2026-09-21_08-48-12/images/`.

## 2. Key correction: ACSPH doesn't use `mdbcDensityScheme` at all

The ACSPH step never calls any of the three boundary-density modules (fluid
density is exactly 1.0 by construction; ghost densities are never
re-extrapolated). Only `schemes/deltaSPH.py:107` reads `mdbcDensityScheme` —
so the `--mdbcDensityScheme` flag is a **no-op for ACSPH runs**, and the
ramped/re-baseline idea that worked for delta-SPH has no ACSPH analogue.

Consequently, the ACSPH escaper is **not** the english2025 issue. ACSPH's wall
enforcement is the freeSlip velocity mirror alone (Eq. 62) —
`noPenetrationShift=False` is the default and dambreak exposes no param to
enable it. Its escaper is a pure **wall-confinement failure** under the
violent impact. (The pre-flip constant-wall run also leaked — 183 m/s
escaper → non-finite — so ACSPH on this case needs more than a wall-BC
change.)

## 3. Mechanism correction: the noPen safeguard is NOT disabled by freeSlip

Checked in the live `wp_nopenshift.py`: the shift fires on
`(v_fluid − v_wall)·n_out < 0` (approaching) within r < 1.25 dx and the
0.75·norm band, with `n_out = −ghostOffset/|·|`. Under the freeSlip mirror the
relative normal velocity works out to **−2(v_f·n_b) — doubled, not zeroed**.
So the earlier "freeSlip mirror zeroes the relative normal velocity that
`computeMdbcNoPenShift` measures" attribution was wrong. More likely
(unproven): the delta-SPH freeSlip leaks happened because english2025's
negative near-wall pressure (§5.10's attracting-wall mechanism) pulls a
particle into the band over many steps — a force the proximity push-back
can't out-muscle.

## 4. Delta-SPH leak root cause (2×2 isolation grid at t*≈7.7)

| | mdbcRho=ramped | mdbcRho=english2025 |
|---|---|---|
| ddt=deltaSPH | **D: clean** (maxV 12.8) | **E: 2 escapers** (penDx 4243) |
| ddt=fourtakas2019 | **F: clean** (maxV 7.3, penDx 0.0) | **C: 1 escaper** (penDx 3930) |

A (fourtakas2019+english2025, **constant** wall, nx=70): clean →
single-factor verdict: **english2025 × freeSlip is the leak**; fourtakas2019
and ramped are benign. Mechanism: under violent impact english2025's wall
pressure goes negative (attracting wall, §5.10) → pulls a particle into the
wall band → it escapes at 30–78 m/s (delta-SPH's escapers don't pin dt, so the
run completes, contaminated). The 09-15 stable overnight batch was
**constant-wall** (caseSpec.json proof) — consistent. Prescribed fix per
§5.10: a cavitation-style floor on the english2025 wall pressure. Side note:
`298d52f`'s new default (english2025) is unsafe for walled freeSlip runs
until that floor exists.

Isolation npz: `scripts/out_deltaSPHMarrone_freeslip_isolation/`.

## 5. Delta-SPH freeSlip re-baseline (F's config) — completed, clean

bg_50a7df83: fourtakas2019 + ramped (keeping the new DDT default, reverting
only the boundary-density scheme), nx=70, t=5.0, video on. Completed 15:56,
EXIT=0: **diverged=False, t*=20.22, 53774 steps, wall 49.9 min**,
`_output.mp4` + `_out.gif` written.

Score:

- **nPenetrating = 0 for the entire run**; maxPenetrationDx 0.34 @ t*=7.53
  (impact transient, under the 0.5 dx threshold that counts as penetrating).
- minDensity 0.719 / maxDensity 1.305 / maxVelocity 25.2 — **all inside the
  impact window (t*≈7.5)**; no late-time excursions.
- Contrast, leaked english2025 run (same dir): maxV 78.7 @ t*=13.06,
  rhoMin 0.523 @ t*=13.06, rhoMax 2.40 @ t*=11.95, penDx → 13882 @ t*=20.22,
  nPen 5 — the escaper's late-time signature.

→ F's config validates at nx=70 through t*=20.2: the cleanest freeSlip
delta-SPH reference. `python scripts/probe_deltaSPHMarrone.py --report
--out scripts/out_deltaSPHMarrone_freeslip` now globs both delta runs (the
leaked english2025 one is kept as the documented-defect data point).

## 6. Delta-SPH leak fix — implemented and validated (2026-09-21, new session)

**Superseded by §8** (same day, later in this session): this magnitude floor
was reverted in favour of a neighbour-count ramp on `alpha` itself, which
fixes the same leak without the case-specific opt-in this section's own
sloshingTank check required. Left in place below as the record of how the
root cause was found.

Item 3's speculative mechanism ("english2025's negative near-wall pressure
pulls a particle into the band") is now confirmed directly and fixed.

**Direct measurement** (`.tmp`-era `_CAPTURE` hook on `english2025.py`, the
leaking C config reproduced at nx=35/t*≈8.9 for speed): `rho_b` collapses to
**0.525** (`P_b` = **-4472** against `c0**2 ~ 9417`) on a handful of ghost
rows at the worst step, while the other ~98% of the wall sits at
**0.9999-1.0000 the same step** — a localized outlier from a degenerate
Shepard fit (sparse/skewed fluid-neighbour sample under violent impact
spray), not a broad bias. This is `alpha` (the 0th-order Shepard value fit at
the ghost node) collapsing directly — `english2025.py`'s hydrostatic
correction term is `O(1e-5)` relative to `c0**2` at this scale, so `rho_b ~
alpha` almost exactly.

**Fix**: `WeaklyCompressibleSPHConfig.mdbcRhoBCavitationFloor` (new field,
`configurations/weaklyCompressible.py`), a lower bound on `english2025.py`'s
`rho_b` as a fraction of `rho0`. Tuning, on the same nx=35 config:

| floor | nPenetrating (peak) | maxPenetrationDx |
|---|---|---|
| off (unclamped) | 1-2 (run-to-run) | 3930-4243 (isolation grid) / 2017-4243 (this session) |
| 0.8 | 2 | 2017 — **still leaks** |
| **0.95** | **0** | **0.11** (sub-threshold, matches the clean 'ramped' reference) |

**Not a blanket default** — shared default is `0.0` (off; `rho_b` never
actually goes negative in practice, so this is a no-op everywhere except a
literal sign-flip pathology). Measured that `0.95` is not free on
sloshingTank: same case, nx=60/t=7.0, freeSlip, floor 0.95 vs. unclamped —
`maxDensity` 1.328→1.130, `minDensity` 0.754→0.918, KE last-quarter mean
0.0196→0.0123 (both runs `diverged=False`, KE still growing quarter-over-
quarter in both — not a collapse in this short/coarse check, but the same
*direction* as the historical ratchet, so promoting this floor beyond
dambreak needs its own check, not an assumption). **`dambreak.configureScheme`
opts in specifically** (`cases/dambreak.py`), exposed as
`--mdbcRhoBCavitationFloor` (default 0.95; pass `0.0` for the old leaking
behaviour, A/B-style, matching `--wallBC constant`'s reproducibility role).

**Validated at both resolutions**: nx=35 config C (ddt=fourtakas2019,
mdbcRho=english2025, freeSlip) now `nPenetrating=0` for the entire t*≈8.9
run, `maxPenetrationDx` <= 0.11 dx. Re-confirmed at nx=70 (H/dx=42, the
original isolation-grid resolution, t*≈7.68, `diverged=False`, 20435 steps,
wall 934.5s): `nPenetrating=0` for the entire run, `maxPenetrationDx` <=
0.096 dx — the same clean signature, not a resolution-specific fluke.
`tests/test_physics.py` and the existing mDBC/english2025 unit tests all
still pass with the change. Code: `configurations/weaklyCompressible.py`
(new field + docstring), `modules/mdbc/english2025.py` (the clamp itself),
`cases/dambreak.py` (the case-scoped opt-in + CLI param). All three
uncommitted as of this edit.

**Left undone, on purpose**: §5.10's own preferred fix — gate on neighbour
count/conditioning instead of raw magnitude (give `english2025.py` a smooth
ramp-to-fallback like `density2025.py`'s `'ramped'` scheme already has,
instead of its current binary `nNbFluid >= 1` trust-or-fallback) — was not
attempted. It should in principle only touch the genuinely degenerate-fit
rows this mechanism lives on and leave sloshingTank's well-supported ones
alone, which the magnitude floor cannot guarantee. Worth trying before
promoting `mdbcRhoBCavitationFloor` past dambreak.

## 8. Item 6's magnitude floor replaced with the principled neighbour-count ramp (2026-09-21, same session)

Item 6's `mdbcRhoBCavitationFloor` was a symptom-level hack: it clamped the
*output* `rho_b` magnitude, needed a case-specific opt-in (dambreak: `0.95`)
because a blanket value measurably damped sloshingTank's legitimate deep
negative-pressure transients, and item 6's own last paragraph already named
the better fix as untried. That better fix is now implemented and the floor
mechanism is fully reverted (all three files: `english2025.py`,
`configurations/weaklyCompressible.py`, `cases/dambreak.py` -- no
`mdbcRhoBCavitationFloor` field/param survives anywhere).

**Root cause, restated precisely**: `rho_b` here is `alpha` (the 0th-order
Shepard VALUE fit at the ghost node) plus a hydrostatic correction that's
`O(1e-5)` relative to `c0**2` at this scale -- so `rho_b ~ alpha` almost
exactly. `alpha` was trusted outright the moment `nNbFluid >= 1` (a binary
`hasAny` gate), with no discount for a marginal 1-3-neighbour stencil under
violent impact spray -- exactly the sparse/skewed sample that collapsed
`alpha` to 0.525 in item 6's capture. `density2025.py`'s 'ramped' scheme
already solves the analogous problem for its OWN 1st-order fit with a smooth
`wN` ramp on `numNeighbors` before blending down to its 0th-order fallback
(`_MDBC_NBR_FLOOR=4.0`, `_MDBC_NBR_RAMP=1.0`); english2025 has no gradient
block to blend away, so the same ramp is now applied one level down, directly
to `alpha` itself, blending it toward `rho0` as `nNbFluid` drops below 4:

    wAlpha = clamp((nNbFluid - 4.0) / 1.0, 0, 1)
    alpha  = wAlpha * alpha + (1 - wAlpha) * rho0

This discounts exactly the rows whose VALUE fit is under-supported (the
actual failure condition) and leaves every well-conditioned row -- including
sloshingTank's legitimately-low-but-well-supported wall readings -- alone,
unlike a magnitude floor which can't distinguish "genuinely degenerate fit"
from "well-conditioned deep negative-pressure reading that happens to be
low."

**Validated**, both re-run from scratch against the new code (not compared
to old npz's -- both are fresh runs):

- **Dambreak, the exact isolated leaking config** (config C: `ddt=
  fourtakas2019`, `mdbcRho=english2025`, freeSlip, nx=35, t*→9.3, same
  fast-repro window item 6 used): `nPenetrating = 0` for the ENTIRE run
  (previously peaked at 1-2), `maxPenetrationDx <= 0.084` dx (previously
  3930-4243) -- diverged=False, 12368 steps. The leak is gone.
- **sloshingTank stability check** (nx=30, t=3.0, `mdbcRho=english2025`,
  the *default* scheme -- `stability-check-run-length` convention): 30000
  steps, diverged=False, density range [0.991, 1.026] over the run, peak
  sensor pressure 5084 Pa (inside the measured 2213-13119 Pa impact band).
  No sign of the floor's density-damping regression -- consistent with the
  ramp only discounting genuinely under-supported rows, not sloshingTank's
  well-conditioned ones.
- `tests/test_physics.py`: all 75 tests pass (unchanged from before this
  edit). `scripts/gradcheck_mdbc.py`: passes (only covers
  `computeMdbcNoPenShiftWarp` currently -- englsih2025/density2025/densityBand
  have no gradcheck coverage of their own yet, pre-existing gap, not
  introduced by this change).

Not done: a full nx=70 / t*≈20 re-validation matching item 5's F-config
reference run, and a longer (t=7) sloshingTank run matching the historical
ratchet check length -- both would strengthen confidence further but weren't
run this session for time. The fast nx=35/t*≈9.3 dambreak repro is the exact
window item 6 used to first characterise and then "fix" the leak, so it's an
apples-to-apples improvement over that baseline.

## 7. Open decisions

1. **ACSPH leg** — options: (a) re-run with `noPenetrationShift=True` (the
   only ACSPH wall knob; the plan's hydrostaticColumn test shows it bounded
   that case) — short no-video pre-check to t*≈7.7 first, then the full video
   run; (b) accept the stalled result as the ACSPH answer; (c) noPen +
   constant walls. Not attempted this session — ACSPH runs are ~80x costlier
   per step than delta-SPH (nx=70 delta-SPH: ~0.03 s/step; nx=70 ACSPH:
   ~2.6 s/step, measured item 1), so even a cheap-nx test is expensive; the
   delta-SPH leak had a clear, cheaply-reproducible isolation and was
   prioritized instead.
2. Record the ACSPH escaper/dt-pin failure mode + this summary in
   ACSPH_PLAN.md (todo item 1 answer), and the OPEN_PROBLEMS.md entry
   (todo item 2) is now a *characterised* defect with a prescribed fix.
3. Uncommitted: the `wallBC` meta addition in `scripts/probe_deltaSPHMarrone.py`
   (still present, now alongside the `mdbcRhoBCavitationFloor` fix — both
   uncommitted as of this edit).
4. §5.10's neighbour-count-gated alternative (item 6, last paragraph) —
   untried; would remove the sloshingTank-promotion caveat if it works.

## 9. ACSPH stall — root causes, fixes, and what is left (2026-09-22/23)

Supersedes item 1's "hidden runaway particle" reading and open decision 1.
Code: `fcaa998`. Probes were throwaway (`scripts/_tmp_*`), numbers below are
from their logs.

### 9.1 Bug: wall rows integrated as fluid inside the pseudo-time loop (fixed)

`artificialCompressible_step` applied the RK stage increments to **every**
row, so inside one real step the wall/ghost particles moved (~1e-3 dx) and
their velocities grew from the step-start mirror to O(|v_fluid|) (0.03-0.06
m/s on the at-rest column) under gravity and pressure force; the mirror was
only an initial guess the loop overwrote. Proof it was dead: `freeSlip`,
`noSlip` and all three ghost-placement modes gave **bit-identical**
trajectories. Visible symptom: on the walled `hydrostaticColumn` the fluid
particle nearest each bottom corner is pulled diagonally into the corner from
step 1 and exits through the wall at 2-3.5 m/s (bulk stays <= 0.15) — the
"corners destabilise ACSPH" result. Fix: zero non-fluid increments and
re-apply `computeBoundaryVelocities` per stage (De Courcy Sec. 3.3: the
no-penetration mirror "is also set on the Lagrangian velocity in the velocity
divergence"). Column at rest now holds: vmax 0.054 -> 0.014 m/s,
`pressureSlopeRatio` 1.006 at t = 1.38 s, **no shifting needed**; sharp and
filleted corners indistinguishable (so the fillet was treating a symptom).

### 9.2 Paper-conformance gaps in the dam-break ACSPH config (fixed)

- **Shifting** was forced off; De Courcy Sec. 3.2/4.5 run Michel shifting
  (displacement form, outside the loop). Without it pairing grows from step 1
  (`pairedFraction` ~0.2, `nnDistP01` ~0.06 dx).
- **Viscosity**: the paper applies `alpha_nu = 0.01`, `nu = alpha_nu h c0/K`,
  `c0 = 50 sqrt(gH)` in every case; we ran `nu = 1e-6` (~4000x less at
  nx=70). Now the default (`acAlphaNu`).
- Still different from the paper (not changed): H/dx 400 vs our 14-42,
  float64 + eps_v = -8 vs float32 + -5, RK3 / dt/dtau = 2 / CFL_t = 0.4 vs
  RK2 / 5 / 0.2.

### 9.3 Michel implementation vs. Michel et al. 2022 (fixed / decided)

- Bug: `reuseNormals` fed the **dilated** surface set (what every scheme
  caches in `surfaceIndicators`) as Eq. (47)'s raw set -> d^FS = 0 across the
  whole vicinity (beta = 1 instead of up to 64, normal fully cancelled).
  michel2022 now re-detects (also fixed a latent bool-mask dtype error on that
  path). Affects every scheme using Michel, incl. delta-SPH `sloshingTank`.
- U_char including wall particles kept as a **deliberate deviation** from
  Sec. 6.1 — ablation below.
- De Courcy Eq. (59) (`bdfShiftCorrection`, default True) was never read;
  implemented. Neutral in the ablation. Eq. (58)'s Taylor correction of p/v
  after the shift is still not implemented (paper: little influence).
- Unchanged extras not in Michel: `maxShiftVelocityFraction` (global-Umax cap)
  and the per-component `threshold` clamp.
- Also found: `computeForcing` returns a force but ACSPH added it as an
  acceleration (every ACSPH forcing scaled by particle mass) — fixed.

### 9.4 nx=24 dam-break ablation (loop fix + paper viscosity throughout)

| shifting | U_char neighbours | Eq. 59 | result |
|---|---|---|---|
| Michel | fluid + walls | on | **survives to t* 8.09** (2/2 incl. pre-Eq.59 code) |
| off | – | – | wall leak -> stall t* 7.1 |
| Michel | fluid only (Sec. 6.1) | off | NaN t* 5.15 |
| Michel | fluid only (Sec. 6.1) | on | NaN t* 5.1 |

Without viscosity (loop fix only) the same runs failed at t* 2.6-4.9. Reading:
wall-inflated U_char strengthens the shift at the wall and, since grad C
includes the walls, that shift points away from them — extra wall repulsion.

### 9.5 The remaining failure: wall suction at wall/free-surface contacts

Every failure, before and after the fixes, is the same local event: a
**thinly-supported free-surface particle next to a wall** acquires negative
pressure, the symmetric `(p_i + p_j)` force pulls it toward the wall, it loses
fluid support, the suction deepens. Measured on one particle (nx=24, t*~2.6):
p = -20 -> -26 -> -36 -> -54 -> -91 -> -180 -> ... -> -1.8e6 over ~20 steps
(x1.2-2.5/step) while it moves through the floor, with the pseudo-time loop
**converged every step** (eps_v -5 reached; the velocity-only metric does not
see pressure). Instances: corner sink (column, pre-fix), a particle riding the
**ceiling** at constant speed against gravity for ~1 s (every nx=24 run), and
the run-up jet at the far wall. **nx=70 with all fixes still fails at the
first far-wall impact (t = 0.685, t* 2.8)**: the run-up jet is 1-2 particles
thick against the wall, scatters, 2 particles cross the wall, vmax 117 ->
1e6. Pairing is only ~3% there, so this is the contact-line suction itself.
Tried and rejected: Antuono pressure switch (no change), clamping surface
particles to p >= 0 (worse: t* 1.2-1.3, runaway moves one layer inward).

Same mechanism as delta-SPH's ceiling hover (OPEN_PROBLEMS.md 1) — see
OPEN_PROBLEMS.md 8, which tracks it as one mDBC contact-line item.

### 9.6 Tooling

- `stallDtSteps` only fires when dt sits *exactly* at `minDt`; the ACSPH
  [0.8, 1.2] step-ratio clamp lets dt hover just above it indefinitely (one
  probe ran 7 h frozen). Probes must stream progress; a wall-clock watcher
  (sim-time advance < 0.005 s over 5 min -> kill) caught each stall in ~5 min.
  A sim-time-progress watchdog in the runner would be the proper fix.
- ACSPH step cost is launch-bound (~1-2 s/step at nx=24-70 when the pseudo
  loop runs 100-200 iterations).

### 9.7 No-free-surface checks (isolating the contact-line issue)

nx=64, streamed logs, post-`fcaa998` code (`scripts/_tmp_check.py`):

| case | deltaSPH | ACSPH |
|---|---|---|
| `tgv-wc` (periodic, no walls), t=2 | clean: vmax <= 1.017, rho [0.9997, 1.001], paired 0 | clean: vmax <= 1.048, paired 0 |
| `randomFlow --bounded` (4 walls, no FS), t=3 | clean: vmax 1.0 -> 0.53, rho [0.9998, 1.002], paired 0 | clean: vmax <= 1.22 -> 0.80, paired 0, nnDistP01 0.78 |
| `movingObstacle` (spinning hexagon, driven), t=3 | clean: vmax <= 1.94, rho [0.996, 1.009], paired 0 | clean to t=0.87 (vmax 1.63, paired 0.003) when the session ended; re-run pending |

**Walls without a free surface are stable in both schemes**, including the
fixed ACSPH wall treatment with fluid pressed against all four walls of the
bounded random flow. Supports the isolation in 9.5: the failures need a
free surface meeting a wall. ACSPH's slight vmax overshoot above the initial
value (1.048 TGV, 1.22 random flow) is consistent with its near-zero
`nu = 1e-6` in those cases, not an instability.
