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

**Focus:** *not set* — the englishWedge item (§5) is resolved: a sign bug in
the default `fourtakas2019` DDT; the default-combo re-validation is done
(2026-09-29, recommendation: keep — see the `OPEN_PROBLEMS.md` row). Work happens on branch
`dev`, merged back into `main` periodically.

## Plans

Kind: **practical** = a current limitation of something people run today;
**ambitious** = new capability, nothing is broken without it;
**reference** = findings/tracking, no steps of its own.

| Plan | State | Kind | Where it stands / next step | Last worked | Last surveyed |
|---|---|---|---|---|---|
| [MDBC_CONTACT_LINE_PLAN.md](MDBC_CONTACT_LINE_PLAN.md) | **active**, stuck | practical | ACSPH unusable at wall/free-surface contacts at nx≥70 (fails at step 1211, t\*≈2.8; confirmed 2026-09-28). Every fix tried so far fails. Next (ACSPH): Batty per-contact normal velocity `(u−v_s)·n`. Next (δ-SPH `loneDensityReset`): flier mechanism (§13.1 item 1), gate question (§13.1 item 2, user decides) | 2026-09-26 | 2026-09-28 |
| [ACSPH_PLAN.md](ACSPH_PLAN.md) | open, blocked on the contact line | practical | Scheme built and validated on the non-free-surface checks; dam break blocked by the contact-line plan. `impact` reference decided (Marrone, aspectRatio 0.5). Left: Lobovsky run, paper-scale tables, closed-box pressure growth (OPEN §9). §0.1 TODO list is out of date | 2026-09-28 | 2026-09-28 |
| [CRKSPH_LIMITER_PLAN.md](CRKSPH_LIMITER_PLAN.md) | mostly done | practical | Closed out 2026-10-01: CRK nondeterminism fixed; limiter default now (1/n_h, 0.2/n_h) (user-decided); Gresho spin-up traced to a slowly converging pressure pump the viscosity cancels (intrinsic, no fix: cusps, pair weights, support consistency all ruled out); regression guards added. Follow-ups done same day: Noh + 2D shock cases at the new default stable (note (h)), AV_PLAN CRK baseline re-taken as results/av_M0f_crk; new OPEN_PROBLEMS §18 (video leaks GPU memory per frame). Parked: local-spacing / softer limiter, glass ICs. 2026-10-02: triple point re-run at nx 256, tLimit 7 with video (`scripts/probe_crkShockVideos.py --nx/--tLimit`, results/crk_triplePoint_nx256): finished, 8803 steps, dE/E -8e-7, no alarms, §18 leak not seen in 2D; probe overrides committed 45375e8 | 2026-10-02 | 2026-10-01 |
| [SMALL_PROBLEM_PERFORMANCE.md](SMALL_PROBLEM_PERFORMANCE.md) | mostly done | practical | δ-SPH Marrone 702 → 109 s (6.5×), bitwise. Left (§5): graph `finalize` (~2.5–3 ms/step), ACSPH still ~187 ms/step, compiled glue needs a jittered batch before default-on | 2026-09-28 | 2026-09-28 |
| [CEILING_STICKING_PLAN.md](CEILING_STICKING_PLAN.md) | **parked 2026-09-30** (kicks fixed + default; sticking open, cosmetic) | practical | **Done:** the ceiling *kicks* were the time integrator (explicit-midpoint growth on the stiff wall-contact mode); `timeCentredContinuity` (ρ as a drift field) + noPen `impulse` fixed them and are defaults (dev c55a209; §3-§7). **Parked (user):** the *sticking* (bilateral wall: english2025 hydrostatic tension + continuity-booked tension). Unilateral wall contact and its gates (§8), and density re-init (§9, experiment) were tried, do not generalise, and their code lives on branch `parked/ceiling-uniwall`. **Finding (§8.4):** a released row is a sub-support fragment, which this model treats as an elastic compressible ball (impact KE ratio 0.68, single-particle P 150-760); that is a model question (cohesion out of scope, isolated-particle mass/volume with the δ-ALE derivation), not a contact switch. | 2026-09-30 | — |
| [OPEN_PROBLEMS.md](OPEN_PROBLEMS.md) | open | practical | Eleven open items: §1 flyers / free-surface pinning (next: freeze the Antuono mask across RK sub-stages), §2 `'mls'` wall pressure, §3 `'band'` mDBC, §4 LDC density ramp (expected behaviour, nothing queued), §8 contact-line suction (own plan), §9 ACSPH closed-box level growth, §13 `iisph` free-slip column (re-checked 2026-10-01: still blows up at step 433, deterministic), §14 DFSPH viscous free-slip alternation (re-checked 2026-10-01: projected term still flips every step, Morris term ~50x smaller), §15 CRKSPH residual + `C_l`/`C_q`/dt follow-ups, §19 compressible lattice walls give way at a curved wall's stagnation point (new 2026-10-06, COMPRESSIBLE_WALLS_PLAN), Resolved: §5, §6, §7, §10–§12, §16–§18, §20 (2026-10-06: support clamp now wall-only by default), §21 (2026-10-06: Owen table fixed, SUPPORT_SOLVER_PLAN closed) (RESOLVED_PROBLEMS.md). **Your decisions pending:** keep the `fourtakas2019` default (recommendation: keep); Morris viscosity default-on; merge `dev`→`main` | 2026-10-06 | 2026-10-01 |
| [COMPRESSIBLE_WALLS_PLAN.md](COMPRESSIBLE_WALLS_PLAN.md) | active | ambitious | Solid walls for Monaghan/CompSPH/CRKSPH. **1D done for all three schemes** (2026-10-06): closedBox, piston (moving wall, push/withdraw), shockReflection, Woodward-Colella between walls (CRKSPH within 1% of its exact mirror reference, energy exact), near-wall deficit cause found (Monaghan -9% -> -1%). Sedov 2D in a walled box vs its mirror twin: Monaghan 4-6% L1. Geometric wall normals (fixed a piston-seam leak). 2D: flat walls hold; forward step (gas frame) and bow shock run; **open: a curved wall's stagnation point gives way under ~10x compression (OPEN_PROBLEMS §19)**. Next: §19's route (mirror ghosts or a per-pair wall kernel), PLANS decision on it. 2026-10-06: the near-wall h clamp is now `supportVolumeClamp='walls'` (applies only with wall rows; shockReflection unchanged, -0.9 %); its root cause is OPEN_PROBLEMS §21 | 2026-10-06 | 2026-10-05 |
| [PST_ALE_PLAN.md](PST_ALE_PLAN.md) | parked (user) | ambitious | Stage A (Michel PST) done and validated. Stages B′ δ-ALE-SPH → B Riemann/MUSCL → C Parshikov → D Vila on hold until the ACSPH validation is done | 2026-09-06 | 2026-09-28 |
| [AV_PLAN.md](AV_PLAN.md) | Phases 0-2 done, **3-6 built + swept, decisions pending** | ambitious | Phase 1 done 2026-10-07 (tag `milestone/dissipation-abstraction`): pair-velocity argument, `pi` is a package (one function per formulation), Pi audited against the papers (5 dead-branch formulations fixed), CRKSPH `Q` goes through it (`Frontiere2017`); the CRK viscosity switch was a no-op (ignored by `Q`, `alpha0s` never advanced) -- fixed, CRK reference now `results/av_crk3_*`. Phase 2: Rosswog KH failure traced to the sharp-IC contact transient + resolution; nx 256 A(1.5) 0.131 vs C&D 0.137 (gate missed by 4.6 %, smooth IC), default CRK best (0.131-0.145) (`docs/av/kh256_2026-10-07/`). Reference for Monaghan/CompSPH is M0g. **2026-10-07 (user: build first, sweep once): Phases 3-6 built** (reconstruction module + policies, García-Senz Table 1 configs, shear box + Chen & Nixon ratio, Sphenix, Wadsley with h re-derived), unit tests + gradchecks green; OPEN_PROBLEMS §22 (C&D corrected-gradient path, opt-in) found and fixed, §18 (video memory) mitigated. **Sweep done 2026-10-08** (41 pass / 21 fail, classified in AV_PLAN's status board; evidence `docs/av/sweep_2026-10-08/`): Wadsley's uniform-compression claim reproduced, Rosswog + limited reconstruction best on KH (A(1.5) 0.151 > CRK), García-Senz SLR helps but AV-energy cut only 2.1x, Sphenix / Wadsley under-dissipate at Sod shocks (suspect the step-boundary alpha lag). **Next: user decisions** (C_q default, chase shock under-dissipation?, Phase 7 default, commits) | 2026-10-08 | 2026-10-08 |
| [PESPH_PLAN.md](PESPH_PLAN.md) | blocked on AV_PLAN M8 | ambitious | Not started; ~3 weeks estimated; the missing fourth scheme of the Frontiere comparison | 2026-09-19 | 2026-09-28 |
| [COUPLED_INCOMPRESSIBLE_NEWTON_PLAN.md](COUPLED_INCOMPRESSIBLE_NEWTON_PLAN.md) | waiting on decisions | ambitious | Scoping only. Unblocked since DFSPH was retired (2026-09-19); needs answers to its open questions 2 and 3. Its banner still calls DFSPH the current priority | 2026-09-04 | 2026-09-28 |
| [FREESLIP_DAMBREAK_FINDINGS.md](FREESLIP_DAMBREAK_FINDINGS.md) | findings, superseded | reference | ACSPH stall root causes (§9) feed the contact-line plan; the §7 "open decisions" are superseded by §9 | 2026-09-23 | 2026-09-28 |
| [examples/weaklyCompressible/MIGRATION_PLAN.md](examples/weaklyCompressible/MIGRATION_PLAN.md) | open, housekeeping | practical (low) | Notebook slots 01–12 done; 13 (`13-open-flow.ipynb`, the largest) open | 2026-09-04 | 2026-09-28 |

### Finished (reference only)

- [SUPPORT_SOLVER_PLAN.md](docs/historic_plans/SUPPORT_SOLVER_PLAN.md) — compressible adaptive support solve: Owen's psi_H table fixed (default), new AV reference M0g, Newton / per-side pressure / clamp-everywhere measured and kept opt-in; clamp stays wall-only (user). Evidence (figure, tables, frames): [docs/support_solver/](docs/support_solver/README.md). Done 2026-10-06.
- [phase6.md](docs/historic_plans/phase6.md) + [phase6_shock_capturing_log.md](docs/historic_plans/phase6_shock_capturing_log.md) —
  C&D 2010 + R&H 2012 wired into Monaghan, Sod 1D/2D/3D + Gresho validated. Done 2026-09-19.
- [examples/sloshingTank/PLAN.md](examples/sloshingTank/PLAN.md) — SPHERIC TC10 port; done.
- [examples/compressible/01-sod/BACKPROP_PLAN.md](examples/compressible/01-sod/BACKPROP_PLAN.md) — done 2026-08-12.
- `docs/historic_plans/` — BOUNDARY_DENSITY, CLEANUP, DELTASPH_VALIDATION, DFSPH_FINDINGS,
  DFSPH_IMPROVEMENT, DFSPH_TODO, INCOMPRESSIBLE_SOLVER, LATTICE_DENSITY, WAVE_EQUATION,
  WCSPH_DEFAULT_CLOSEOUT, WCSPH_SHIFTING. Closed; what survived them is in `OPEN_PROBLEMS.md`.

## Priorities

Ordered by how much it matters now: is the feature unusable, what does fixing
it gain, and at what effort. Last ranked 2026-09-28; resolved rows removed 2026-10-01.

| # | Item | Home | Unusable now? | Gain / effort | Kind |
|---|---|---|---|---|---|
| 1 | Wall suction at wall/free-surface contact lines | contact-line plan, OPEN §8 | **Yes, ACSPH** (Marrone 3.1 nx=70 fails at t\*≈2.8). δ-SPH defaults are fine (0 free fliers) | High gain; high, uncertain effort — research-grade now | practical |
| 2 | Isolated-fragment kicks / corner fliers in δ-SPH | OPEN §1 (no plan) | No — runs finish, but late kicks (maxV up to 28 m/s) and sensor spikes | Medium gain; **cheapest step never tried**: freeze the Antuono bulk/surface mask across RK sub-stages | practical |
| 3 | ACSPH pressure level grows exponentially in a closed box | OPEN §9 (ACSPH) | Yes for ACSPH in sealed/filled containers | Medium; the discriminating test (uniform level, periodic box) is cheap and unrun | practical |
| 4 | CRKSPH Gresho spin-up; limiter constants in wrong units | CRKSPH_LIMITER_PLAN | No — but silent energy injection in vortical flows | Medium; moderate (P1 is mostly a sweep) | practical |
| 5 | ACSPH validation leftovers (Lobovsky, paper-scale tables) | ACSPH_PLAN | No | Low–medium; Lobovsky will hit #1 first | practical |
| 6 | Remaining performance (ACSPH ~187 ms/step; `finalize` eager) | SMALL_PROBLEM_PERFORMANCE §5 | No — but ACSPH's step cost slows every attempt at #1 and #3 | Medium | practical |
| 7 | `'mls'` wall pressure (OPEN §2), `'band'` mDBC (OPEN §3), LDC density ramp (OPEN §4) | OPEN_PROBLEMS | No — non-default modes / expected behaviour | Nothing queued | accepted |
| — | Surface detection: isolated particle reads λ=1 (`detectIsolated` only in ACSPH) | OPEN §8 | No | Nice to fix, **not** the cause of surface blowups (that was the DDT choice; `fourtakas2019` helped) — user, 2026-09-28. In the current batch | practical (low) |

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

### CRKSPH_LIMITER_PLAN.md — mostly done
Closed out 2026-10-01. CRKSPH runs were not reproducible (atomic scatter in the compatible-energy sum): fixed, bit-identical now. The limiter
constants (1/3, 0.2) were in units of H where the paper's / Spheral's are in particle spacings: the default is now derived per step as
(1/n_h, 0.2/n_h) (user-decided). The Gresho vortex's KE spin-up is a slowly converging (~nx^-0.4) tangential pressure pump that the viscosity
cancels to a few percent, so the net grows with resolution (-9 % / +2 % / +8 % at nx 48 / 64 / 96): ruled out the cusps, Spheral's pair weights
and the support consistency, so it is intrinsic to CRKSPH as formulated; no limiter constant removes it across resolution (P1 map: a smooth viscosity knob).
Regression guards added. Parked: local-spacing / softer-threshold limiter, glass ICs, Noh + 2D shocks at the new default, re-take of the AV_PLAN CRK baseline.

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

### AV_PLAN.md — Phases 0–1 done, Phase 3 next (see its status board)
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

- 2026-10-07 — OPEN_PROBLEMS (user): fix what can be fixed while the AV sweep runs. Done: §22 (C&D / Cullen-Hopkins corrected gradient: volume-weighted M, `Vs M^-1`) resolved; §18 (video GPU-memory growth) mitigated with `runner.collectFrameGarbage`. Both developed in a scratch copy and installed mid-sweep only after showing the sweep's code paths bit-identical / unaffected.
- 2026-10-07 — AV_PLAN (user): **build first, sweep once.** Implement Phases 3-6 back to back, each gated only by its cheap checks (unit tests, gradcheck, bit-identical smoke `--compare` for refactors, a short no-divergence run); the expensive validation runs of all phases (KH nx 256, full-profile ladders, Phase 4/5A/5B/6 matrices) go into one overnight sweep instead of one wait per subtask. Phase 7's matrix is that sweep's natural superset.
- 2026-10-07 — AV_PLAN (user): sweep the Pi formulations against the papers, split `pi` into a package and route CRKSPH's `Q` through it (single source of truth); fix CRKSPH's viscosity switch; tag `milestone/dissipation-abstraction`; run KH at nx 256 to t = 3 for Rosswog / C&D / fixed alpha and default CRK. **Reverses the 2026-09-30 S2 decision to delete CRKSPH's `updateViscositySwitch` call**: that call is what lets the switch have any effect under CRK (evidence `docs/av/crk_switch_2026-10-07/`). Smooth-density KH IC tried at the user's suggestion (CRKSPH paper Eq. 100), opt-in `smoothDensity`.
- 2026-10-01 — AV_PLAN: adopt the S4 CRK `dudt.py` sign fix; work OPEN_PROBLEMS §17; pull in the Rosswog 2000 paper and check that switch against it; move the Read-Hayfield pair loops to warp kernels (minimum image, core kernel functions) (user).
- 2026-10-06 — Support solver closed: keep `supportVolumeClamp='walls'` (pure Owen for pure fluid); `'always'` (min(Owen, h(rho))), Newton h and `pressureFormulation='perSide'` stay opt-in (user; docs/historic_plans/SUPPORT_SOLVER_PLAN.md).
- 2026-10-06 — Support solver: make the corrected Owen table (`owenTable='lattice'`) the default now; then build the Hamiltonian grad-h form (Price 2012 Eq. 44) in warpSPHCore and test Newton + grad-h before choosing the Monaghan host's solver (user; SUPPORT_SOLVER_PLAN option A then C).
- 2026-10-06 — Fix the adaptive support solve now, before more compressible work builds on it (SUPPORT_SOLVER_PLAN.md) (user).
- 2026-10-06 — The adaptive-support volume clamp (013a22f) degrades smooth pure-fluid flow, so it applies only when the state has wall rows: `SimulationConfig.supportVolumeClamp='walls'` default ('always' / 'off' selectable). Rule: a fix for a boundary issue that degrades fluid behaviour is off for pure-fluid cases (user; OPEN_PROBLEMS §20 → RESOLVED, root cause §21).
- 2026-10-01 — OPEN_PROBLEMS §18 (Monaghan Sedov energy drift) closed as expected behaviour: RK2 time-integration error that converges with dt; no CFL advice needed (user).
- 2026-09-30 — AV_PLAN order: tag M0 as-is → `av_report --compare` → S2 (limitXi/xi,
  state tags, fixed β, registry) → S3 stubs, S4 CRK sign any time → OPEN_PROBLEMS §16
  *last*, before Phase 2's Sedov/Noh rows (user).
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
- 2026-09-29 — WCSPH defaults: `timeCentredContinuity=True` (density as a
  drift field) and `mdbcNoPenShiftMode='impulse'` (noPen as a restitution
  impulse on the integrated velocity), "generally improvements and avoid real
  instabilities with a principled fix"; warpSPHIntegrators `drift-fields`
  merged into its `main` after its full suite passed (user;
  `CEILING_STICKING_PLAN.md` §7.2).

- 2026-09-30 — `densityReinit` (summation-density floor) is an experiment only, never a default: WCSPH density stays a purely integrated quantity (temporal memory, ρ-free/ML view); any configuration-derived ρ correction is opt-in and labelled (user; `CEILING_STICKING_PLAN.md` §9).

- 2026-09-30 — `unilateralWallContact` (+ its refinements `unilateralWallAirOwnPressure`, `unilateralWallBodyForceGate`) parked, opt-in and default off, not to be extended: releasing a contact by a chain of gates (traction sign → own-pressure clamp → gravity/orientation gate → corner gate …) is not a principled fix and does not generalise (moving boundary, g_eff); the sub-support fragments it releases (elastic lone particles) are a model problem to be solved in the physics, not by more switches (user; `CEILING_STICKING_PLAN.md` §8.5). Code moved off `dev` to branch `parked/ceiling-uniwall` the same day: the models stay free of these fragments (user).
- 2026-09-30 — Sub-support fragments (elastic lone particles, `CEILING_STICKING_PLAN.md` §8.4): surface tension / cohesion (Akinci 2013) is out of scope here (proper research work); the ballistic-isolated-particle treatments (Schechter & Bridson 2012, Ihmsen 2012) are the same idea as DualSPHysics' exclusion of out-of-range fluid, not pursued; Michel et al. 2022's treatment of isolated particles' mass/volume belongs to the δ-ALE-SPH derivation already on the list (user).

## Doc hygiene (open)

- `ACSPH_PLAN.md` §0.1 and §9.2 still treat the t\*≈6 divergence as the headline.
- `COUPLED_INCOMPRESSIBLE_NEWTON_PLAN.md`'s banner calls DFSPH the current priority (retired 2026-09-19).
