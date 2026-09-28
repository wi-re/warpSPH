# Plans — index, status and priorities

The one place that says which plan is active, what each one is waiting on,
and what matters most. Detail lives in the plan files; this file is the map.

## How to use this file (the progress marker)

- **One row per plan** in the table below, and only one. Its two dates are the
  progress markers:
  - **Last worked** — the last day real work was done on that plan (code, runs,
    decisions, a results section). Update it, and the row's *Where it stands*
    cell, **in the same session** as the work — before or with the commit.
  - **Last surveyed** — the last day someone reviewed the plan *without*
    working on it (a status pass like the one that produced this file). A plan
    whose *Last worked* is recent but *Last surveyed* is old has drifted from
    view; one where both are old is parked, whether or not it says so.
- **Before chasing a new problem**, decide where it belongs: an existing
  plan (update its row), `OPEN_PROBLEMS.md` (a short entry — hard problems and
  small loose ends alike), or a new plan file (add a row). Then chase it. A
  problem investigated with no row to show for it is the chaos this file
  exists to prevent. Resolved `OPEN_PROBLEMS.md` entries move to
  [docs/historic_plans/RESOLVED_PROBLEMS.md](docs/historic_plans/RESOLVED_PROBLEMS.md),
  leaving a one-line stub under the same number.
- A **new plan file** gets a row the day it is created; a finished one moves
  to `docs/historic_plans/` and its row to *Finished* (one line).
- The **Focus** line is the user's call. Claude may suggest, not set it.
- Every plan file carries a one-line pointer back here under its title.

**Focus:** *not set* — the 2026-09-28 loose-ends batch is done (see the
`OPEN_PROBLEMS.md` row). Work happens on branch `dev`, merged back into
`main` periodically.

## Plans

Kind: **practical** = a current limitation of something people run today;
**ambitious** = new capability, nothing is broken without it;
**reference** = findings/tracking, no steps of its own.

| Plan | State | Kind | Where it stands / next step | Last worked | Last surveyed |
|---|---|---|---|---|---|
| [MDBC_CONTACT_LINE_PLAN.md](MDBC_CONTACT_LINE_PLAN.md) | **active**, stuck | practical | ACSPH unusable at wall/free-surface contacts at nx≥70 (fails at step 1211, t\*≈2.8; confirmed 2026-09-28). Every fix tried so far fails. Next (ACSPH): Batty per-contact normal velocity `(u−v_s)·n`. Next (δ-SPH `loneDensityReset`): flier mechanism (§13.1 item 1), gate question (§13.1 item 2, user decides) | 2026-09-26 | 2026-09-28 |
| [ACSPH_PLAN.md](ACSPH_PLAN.md) | open, blocked on the contact line | practical | Scheme built and validated on the non-free-surface checks; dam break blocked by the contact-line plan. `impact` reference decided (Marrone, aspectRatio 0.5). Left: Lobovsky run, paper-scale tables, closed-box pressure growth (OPEN §9). §0.1 TODO list is out of date | 2026-09-28 | 2026-09-28 |
| [CRKSPH_LIMITER_PLAN.md](CRKSPH_LIMITER_PLAN.md) | open | practical | Gresho spin-up located (non-central pressure forces), limiter constants in the wrong units; nothing changed in code yet. Next: P1 parameter-stability map (η_crit × η_fold) gated by the Sod ripple probe | 2026-09-26 | 2026-09-28 |
| [SMALL_PROBLEM_PERFORMANCE.md](SMALL_PROBLEM_PERFORMANCE.md) | mostly done | practical | δ-SPH Marrone 702 → 109 s (6.5×), bitwise. Left (§5): graph `finalize` (~2.5–3 ms/step), ACSPH still ~187 ms/step, compiled glue needs a jittered batch before default-on | 2026-09-28 | 2026-09-28 |
| [OPEN_PROBLEMS.md](OPEN_PROBLEMS.md) | open (batch done) | practical | 2026-09-28 batch done: §7 Morris (no-slip column at rest; DFSPH viscous-dt bug fixed; default-on is your call), §8 isolated rows (classification fix, bitwise no-op for δ-SPH), §5 englishWedge corners *worse* under new defaults (open), §6.3 re-runs clean, §1 step 1 moot (mask chatter is step-to-step → step 2 next). New: §13 `iisph` column (parked), §14 viscosity + free-slip flip. §10–§12 resolved | 2026-09-28 | 2026-09-28 |
| [PST_ALE_PLAN.md](PST_ALE_PLAN.md) | parked (user) | ambitious | Stage A (Michel PST) done and validated. Stages B′ δ-ALE-SPH → B Riemann/MUSCL → C Parshikov → D Vila on hold until the ACSPH validation is done | 2026-09-06 | 2026-09-28 |
| [AV_PLAN.md](AV_PLAN.md) | not started | ambitious | M0–M8 all open. Its input (C&D 2010 + R&H 2012 on Monaghan, `phase6.md`) is done. First step: M0 baseline lock. Gates PESPH | 2026-09-19 | 2026-09-28 |
| [PESPH_PLAN.md](PESPH_PLAN.md) | blocked on AV_PLAN M8 | ambitious | Not started; ~3 weeks estimated; the missing fourth scheme of the Frontiere comparison | 2026-09-19 | 2026-09-28 |
| [COUPLED_INCOMPRESSIBLE_NEWTON_PLAN.md](COUPLED_INCOMPRESSIBLE_NEWTON_PLAN.md) | waiting on decisions | ambitious | Scoping only. Unblocked since DFSPH was retired (2026-09-19); needs answers to its open questions 2 and 3. Its banner still calls DFSPH the current priority | 2026-09-04 | 2026-09-28 |
| [FREESLIP_DAMBREAK_FINDINGS.md](FREESLIP_DAMBREAK_FINDINGS.md) | findings, superseded | reference | ACSPH stall root causes (§9) feed the contact-line plan; the §7 "open decisions" are superseded by §9 | 2026-09-23 | 2026-09-28 |
| [examples/weaklyCompressible/MIGRATION_PLAN.md](examples/weaklyCompressible/MIGRATION_PLAN.md) | open, housekeeping | practical (low) | Notebook slots 01–12 done; 13 (`13-open-flow.ipynb`, the largest) open | 2026-09-04 | 2026-09-28 |

### Finished (reference only)

- [phase6.md](phase6.md) + [phase6_shock_capturing_log.md](phase6_shock_capturing_log.md) —
  C&D 2010 + R&H 2012 wired into Monaghan, Sod 1D/2D/3D + Gresho validated. Done 2026-09-19.
- [examples/sloshingTank/PLAN.md](examples/sloshingTank/PLAN.md) — SPHERIC TC10 port; done.
- [examples/compressible/01-sod/BACKPROP_PLAN.md](examples/compressible/01-sod/BACKPROP_PLAN.md) — done 2026-08-12.
- `docs/historic_plans/` — BOUNDARY_DENSITY, CLEANUP, DELTASPH_VALIDATION, DFSPH_FINDINGS,
  DFSPH_IMPROVEMENT, DFSPH_TODO, INCOMPRESSIBLE_SOLVER, LATTICE_DENSITY, WAVE_EQUATION,
  WCSPH_DEFAULT_CLOSEOUT, WCSPH_SHIFTING. Closed; what survived them is in `OPEN_PROBLEMS.md`.

## Priorities

Ordered by how much it matters now: is the feature unusable, what does fixing
it gain, and at what effort. Last ranked 2026-09-28.

| # | Item | Home | Unusable now? | Gain / effort | Kind |
|---|---|---|---|---|---|
| 1 | Wall suction at wall/free-surface contact lines | contact-line plan, OPEN §8 | **Yes, ACSPH** (Marrone 3.1 nx=70 fails at t\*≈2.8). δ-SPH defaults are fine (0 free fliers) | High gain; high, uncertain effort — research-grade now | practical |
| 2 | Isolated-fragment kicks / corner fliers in δ-SPH | OPEN §1 (no plan) | No — runs finish, but late kicks (maxV up to 28 m/s) and sensor spikes | Medium gain; **cheapest step never tried**: freeze the Antuono bulk/surface mask across RK sub-stages | practical |
| 3 | ACSPH pressure level grows exponentially in a closed box | OPEN §9 (ACSPH) | Yes for ACSPH in sealed/filled containers | Medium; the discriminating test (uniform level, periodic box) is cheap and unrun | practical |
| 4 | CRKSPH Gresho spin-up; limiter constants in wrong units | CRKSPH_LIMITER_PLAN | No — but silent energy injection in vortical flows | Medium; moderate (P1 is mostly a sweep) | practical |
| 5 | Missing Morris shear viscosity (no tangential stress at no-slip walls) | OPEN §7 (no plan) | Partly — viscous wall-bounded flows are incomplete | Good ratio: new kernel + gradcheck + `hydrostaticColumn` A/B | practical |
| 6 | ACSPH validation leftovers (Lobovsky, paper-scale tables) | ACSPH_PLAN | No | Low–medium; Lobovsky will hit #1 first | practical |
| 7 | Remaining performance (ACSPH ~187 ms/step; `finalize` eager) | SMALL_PROBLEM_PERFORMANCE §5 | No — but ACSPH's step cost slows every attempt at #1 and #3 | Medium | practical |
| 8 | Cheap rechecks: englishWedge concave corner (OPEN §5), 8 cases never re-run after the sampler mass fix (OPEN §6.3), probe toys' pinned dt (OPEN §11) | OPEN_PROBLEMS | No | Very cheap | housekeeping |
| 9 | `'mls'` wall pressure (OPEN §2), `'band'` mDBC (OPEN §3), LDC density ramp (OPEN §4) | OPEN_PROBLEMS | No — non-default modes / expected behaviour | Nothing queued | accepted |
| — | Surface detection: isolated particle reads λ=1 (`detectIsolated` only in ACSPH) | OPEN §8 | No | Nice to fix, **not** the cause of surface blowups (that was the DDT choice; `fourtakas2019` helped) — user, 2026-09-28. In the current batch | practical (low) |
| ✓ | Runner missed finite blowups | RESOLVED_PROBLEMS.md (was OPEN §10) | — | **Done 2026-09-28**: velocity alarm + probe stall defaults (README "Watching a run") | — |

Suggested order: #2 step 1 (cheap, improves the default scheme) → #3's
periodic-box test (narrows #1 cheaply) → #1's next ACSPH idea, with #7's ACSPH
step cost worked on alongside since it limits how fast #1 can move. The
ambitious plans can wait: none of them fixes something broken today.

## Plan notes (the survey behind the table)

### MDBC_CONTACT_LINE_PLAN.md — active, stuck
A thinly-supported free-surface particle next to a wall gets negative
pressure from the wall closure; the symmetric `(p_i + p_j)` force pulls it
in; with fewer fluid neighbours the suction deepens geometrically (×1.2–2.5
per step) until it crosses the wall and the kick collapses `dt`. Walls alone
are fine (6/6 no-free-surface checks clean). Tried and failed: ACSPH
`vicinity` (sprays elastic fliers), `wall` (separating row runs away),
`wall` + unilateral divergence (fluid floats, blows up), δ-SPH unilateral
wall and density floors (removed), `loneDensityReset` (pressure rise is
variability, but free fliers up 4/4 realisations — not default-on).
Re-run 2026-09-28 under the velocity alarm: raised at step 1211 (t=0.6893)
with the offending particle already below the floor, `dt` at the floor by
step 2135. Open: a per-contact normal-velocity criterion; the δ-SPH
complementarity must live in the continuity equation; rows with λ<0.5 not in
Marrone's raw surface set.

### ACSPH_PLAN.md — open, blocked
De Courcy et al. 2024 dual-time ACSPH: built, gradchecked, the five paper
cases wired. Hydrostatic column, droplet, patch and impact run;
no-free-surface checks clean. The dam-break "t\*≈6 divergence" in §0.1 is
superseded: after the wall-row loop fix and paper viscosity (`fcaa998`) the
remaining failure is the contact line (above). Decided 2026-09-28: the
Marrone configuration is the reference for `impact` (and generally); De
Courcy's H=2L only for final validation. Open: Lobovsky run (harness built,
never run), Table 1/2 error cliff at paper resolution, Fig. 15/16 needs
digitisation, closed-box level growth (OPEN §9). Doc hygiene: §0.1 items 1,
2 and 5 are done or superseded; §9.2's table still calls the t\*≈6 run the
headline.

### CRKSPH_LIMITER_PLAN.md — open
From the warpSPHCore higher-order harness: CRKSPH's Gresho vortex gains
kinetic energy (non-central pair pressure forces), and the limiter constants
(1/3, 0.2) are in units of H where the paper's are in Δx — the derived
values trade one case against another (14-case sweep), with knife-edge
sensitivity at the nearest-neighbour spacing. Plan: P1 stability map over
(η_crit, η_fold) with seeds and glass ICs, P2 local-spacing / softer limiter
variants, P3 spin-up source, P4 acceptance before changing defaults, P5
regression guards. Rule: no viscosity switch; every change also checked on
Sod against the paper's Fig. 4.

### SMALL_PROBLEM_PERFORMANCE.md — mostly done
Multi-lane neighbour kernels, whole-step CUDA graphs, pipelined outputs,
graphed diagnostics: Marrone 3.1 nx=67 full run 702 → 109 s, bitwise. ACSPH
2065 → 187 ms/step but still ~60× δ-SPH per step. 2026-09-28: garbage
collection during a capture could fail it (fixed, `_captureGuard`); the
progress row is throttled (a per-step redraw cost ~0.8 ms). Left: graph
`finalize`, one-transfer probes, Verlet rebuild, compiled glue behind a
jittered batch.

### PST_ALE_PLAN.md — parked by the user
Michel 2022 PST (Stage A) implemented and validated: first-order interior
convergence (slope 0.949), free-surface rate 1.811, sloshingTank default.
The ALE route is δ-ALE-SPH (antuono2021, primitive variables, no Riemann
solver) before Riemann/MUSCL → Parshikov → Vila. On hold until the ACSPH
validation cases are done.

### AV_PLAN.md — not started
Turns the dissipation stack into swappable detector / coefficient /
velocity-pair / operator parts; phases M0 (baseline lock) → M8 (AV
bake-off), covering Rosswog 2020, García-Senz & Cabezón 2026, Chen & Nixon
2025, Sphenix, Gasoline2. Large. Its Phase-1 refactor also fixes two state
tag bugs PESPH_PLAN §2.1 found. Nothing current is broken without it.

### PESPH_PLAN.md — blocked on AV_PLAN M8
Pressure-energy / pressure-entropy SPH (Frontiere App. G, Hopkins 2013),
one scheme with a switch; PESPH-A first. ~3 weeks estimated; risk item is the
number-density h-solve with grad-h terms. §7 (MFM outlook) is not a
commitment.

### COUPLED_INCOMPRESSIBLE_NEWTON_PLAN.md — waiting on decisions
Scoping: the linear pressure-Poisson part is already done by the Krylov
solvers; what remains is a genuinely coupled nonlinear pressure–velocity
Newton solve with no reference implementation. Proposes a small Phase 1
(projected/semismooth Newton for the `p ≥ 0` clamp). Needs answers to its
questions 2 (unify with the WC/EOS family or pressure-only) and 3 (is Phase 1
worth it on its own).

### FREESLIP_DAMBREAK_FINDINGS.md — reference
2026-09-21/23: the free-slip wall flip, the δ-SPH leak fix (neighbour-count
ramp on english2025's alpha), the ACSPH stall root causes (wall rows
integrated as fluid, paper-conformance gaps, Michel U_char), and the
isolation of the contact-line failure. Feeds the contact-line plan.

## Decisions log

- 2026-09-28 — Velocity alarm flags, does not stop; a stop is the stall
  watchdogs' job or an explicit ceiling (user).
- 2026-09-28 — The Marrone configurations are the reference for every
  scheme; paper-specific setups (De Courcy H=2L) only for final validation (user).
- 2026-09-28 — `detectIsolated` in the shared surface detector: worth doing,
  low priority; surface blowups were the DDT choice, not detection (user).
- 2026-09-28 — Loose ends live in `OPEN_PROBLEMS.md` as short entries, not
  in plan files of their own; resolved ones move to
  `docs/historic_plans/RESOLVED_PROBLEMS.md` (user).
- 2026-09-28 — Development happens on a `dev` branch, merged back into
  `main` periodically (user).

## Doc hygiene (open)

- `ACSPH_PLAN.md` §0.1 and §9.2 still treat the t\*≈6 divergence as the headline.
- `COUPLED_INCOMPRESSIBLE_NEWTON_PLAN.md`'s banner calls DFSPH the current priority (retired 2026-09-19).
- `AV_PLAN.md` / `phase6.md` say the `acsph-plan` branch is not pushed (merged into `main` 2026-09-27).
- `OPEN_PROBLEMS.md` §6 points to `retired_plans/` (it is `docs/historic_plans/`).
