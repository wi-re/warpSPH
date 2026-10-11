# warpSPH — free-surface / wall contact plan (contact-line suction · ceiling sticking · fliers)

> **Progress marker:** this plan's row in [PLANS.md](PLANS.md). Update its *Last worked* date and *Where it stands* whenever you work on this plan.

Opened 2026-10-11 (user). One plan for what goes wrong where a free surface meets a solid wall in the weakly compressible
and artificial-compressibility schemes (δ-SPH / δ⁺-SPH with mDBC, and ACSPH). It replaces `MDBC_CONTACT_LINE_PLAN.md`,
`CEILING_STICKING_PLAN.md` and `FREESLIP_DAMBREAK_FINDINGS.md` (now in `docs/historic_plans/`, text intact). Not part of it:
[`ACSPH_PLAN.md`](ACSPH_PLAN.md) (the scheme and its paper validation, which is blocked on this plan), and `OPEN_PROBLEMS.md`
§1 (fluid-fluid truncation / Antuono mask), §2-§4 (non-default wall modes), which keep their own homes and are only linked.

## 0. What went where

| Old plan | Now | Carried over | Dropped |
|---|---|---|---|
| `MDBC_CONTACT_LINE_PLAN.md` (1107 lines, active, stuck) | `docs/historic_plans/` | §1-§2 below (problem, mechanism), §3 (the attempt catalogue), the open items W1-W7 | The toy-by-toy tables (§7.2, §8.5, §11), the Batty set-B design (kept as one line, W1), the literature digests (§7.1, §8.2, §10; the pointers are in §6) |
| `CEILING_STICKING_PLAN.md` (855, parked) | `docs/historic_plans/` | The integrator finding as a settled result, the sticking as a parked item (W6), the elastic-fragment observation | The forensics logs (§3, §7.1), the unilateral-wall and density-reinit experiments (code is on `parked/ceiling-uniwall`; outcome in §3) |
| `FREESLIP_DAMBREAK_FINDINGS.md` (383, findings) | `docs/historic_plans/` | The δ-SPH leak fix and the ACSPH stall root causes (settled, §2), the no-free-surface controls | The §7 "open decisions" (superseded by its own §9), the measurement logs |

## 1. The problem

A **thinly supported free-surface particle next to a wall** gets negative pressure from the wall closure and from its own
pressure. The symmetric `(p_i + p_j)` force then pulls it into the wall; nearer the wall it has fewer fluid neighbours, so
the suction deepens geometrically (one ACSPH particle: p = −20 → −26 → −36 → −54 → … → −1.8e6 in ~20 steps, ×1.2-2.5 per
step) until it crosses the wall and the kick collapses `dt`.

| Scheme | Where it shows up |
|---|---|
| ACSPH | the far-wall run-up jet, 1-2 particles thick, at the first impact (Marrone 3.1 nx 70, t\* ≈ 2.8); a particle riding the ceiling against gravity (every nx 24 run); sloshingTank nx 100 blows up at t = 4.917 (uid 958 escapes through the right wall during a run-up) |
| δ-SPH / δ⁺-SPH | the sloshingTank ceiling hover (UID 4104; also OPEN §1); **ceiling sticking**: 22-25 single particles or 2-3-particle clumps held under a ceiling in Marrone 3.1 (cosmetic: no dynamics); the english2025 low-α wall leak (fixed) |

**What it is not** (all measured): walls in general (periodic TGV, bounded random flow with fluid against all four walls and a
spinning obstacle are clean in both schemes at nx 64, `docs/historic_plans/FREESLIP_DAMBREAK_FINDINGS.md` §9.7, re-run with
video still owed); ACSPH solver non-convergence (the pseudo-time loop converges every step, its velocity-only metric just does
not see pressure); the corner fillet, `noPenetrationShift`, the Antuono switch, clamping surface-particle pressure to ≥ 0.

## 2. Settled (do not re-open)

1. **The wall is bilateral, and both halves matter.** The wall closure supplies tension when `p_f ≈ 0` (english2025's
   hydrostatic extension `ρ₀ g·(x_b − x_g)` is negative at any ceiling), and the row's own small tension is booked by
   continuity against the mirrored wall velocity (a separating row *expands* at twice its speed). In ACSPH the converged
   `div v = 0` with the mirrored wall velocity *is* the no-separation constraint. A wall-pressure clamp alone (the literature's
   EDAC / PySPH / DualSPHysics fix) moves the tension onto the row's own pressure and does not release the contact.
2. **The δ-SPH velocity kicks at the ceiling were the time integrator**, not the wall model: explicit-midpoint growth of the
   stiff wall-contact acoustic mode. Fixed, and now the defaults (dev c55a209): `timeCentredContinuity` (ρ advanced as a drift
   field with the time-centred velocity) and noPen `'impulse'` (the old `'finalize'` replacement pumped energy at wall contacts
   and launched a lone ceiling rider). Gates run on Marrone 3.1 / 3.4, sloshingTank to t = 7; the late-run-up P2 pressure
   difference between the modes is realisation spread (6 seeds), not an `impulse` effect.
3. **The δ-SPH leak was the english2025 low-α fit**, fixed by a neighbour-count ramp on α (not a magnitude floor); the
   `fourtakas2019` DDT sign bug was separate and is fixed (68a9a6d, re-validated 2026-09-29).
4. **ACSPH stall root causes** (`fcaa998`): wall rows integrated as fluid inside the pseudo-time loop; viscosity 4000× below the
   paper's; Michel shifting off; `reuseNormals` fed the dilated surface set to Eq. (47) (also affects δ-SPH sloshingTank); the
   `computeForcing` force added as an acceleration. With these nx 24 survives to t\* 8.1; nx 70 still fails at the first
   far-wall run-up. `stallProgress` (sim-time watchdog) is in the runner.
5. **The isolated-particle detection bug** (`λ = 1` classed it as bulk) is fixed in the shared detector
   (`SurfaceDetectionConfig.flagIsolated`, default on, merged 2026-09-28). δ-SPH is bitwise unchanged: a classification fix, not
   a dynamics one.
6. **The "fliers" are an elastic-fragment model property.** A sub-support fragment behaves as an elastic compressible ball
   (impact KE ratio 0.68; restitution ≈ 1 in the ACSPH contacts; single-particle P 150-760). Two routes feed it: a frozen
   pressure carried into the contact (removed by `isolatedZeroPressure`), and the clamped wall-adjacent band, which can push but
   not hold. Surface tension / cohesion (Akinci 2013) is **out of scope** (user, 2026-09-30); ballistic isolated particles
   (Schechter & Bridson 2012, Ihmsen 2012) is DualSPHysics' exclusion of out-of-range fluid, not pursued; Michel 2022's treatment
   of isolated particles' mass / volume belongs with the δ-ALE-SPH derivation (PST_ALE_PLAN).

## 3. The attempt catalogue (every fix tried for the sticking / contact line)

None is on by default except where stated. "Flier gate": free fliers at the `off` baseline are 0-3, so a fix must stay there.

| Variant | Scheme | Outcome |
|---|---|---|
| wall-only pressure clamp | ACSPH | does not release the contact (§2.1) |
| `vicinity` (p ≥ 0 on surface+wall rows within one support of the raw surface set, isolated rows) | ACSPH | releases riders, reaches t\* 9.9 at nx 70, but ~100 elastic fliers (each contact restitution ≈ 1); 432/911 flier births never touched a wall |
| `wall` (Batty on fluid-solid pairs, force only) | ACSPH | clean ceiling release on the toy, but a separating row's pressure runs away; Marrone nx 24 blows up t 0.48 |
| `wall` + kinematic half (wall-pair divergence kept only while compressive) | ACSPH | clamp rectifies discretisation noise at every wall, the body floats off the floor; blow-up t 0.34; sealed box NaN t 0.13 |
| `contact` (Batty set A: wall rows + fluid rows with a wall neighbour, p ≥ 0 projected) | ACSPH | nx 70 diverges t 1.26 alone |
| `contact` + `isolatedZeroPressure` | ACSPH | **best so far:** nx 70 finishes to t\* 9.9, 14 flier uids / 58 births, max v 6.4 (one 13.2 spike); nx 24 still 3 fliers (the clamped band, route a); sealed box still fails |
| dual-time SPH's wall (`dtsph`, `dtsphDiv`) | ACSPH | partial: toy drop fall ratio 0.74 (off 0.64, free fall 0.97), column at rest fine, but the sealed box is NaN-scale within t = 0.4 |
| p ≥ 0 clamp on vicinity rows (v1, v3) | δ-SPH | diverges t = 2.9 (separation booked as expansion, ρ down to 0.21; released later as p −174…−320) |
| v1 + density floor `ρ ≥ ρ₀` (v2) | δ-SPH | survives but **ratchets** (acoustic rectification: bulk ρ 1.001 → 1.03, KE/85); removed from the code |
| unilateral freeSlip velocity + one-sided hydrostatics | δ-SPH | fliers launched off every wall; removed |
| `loneDensityReset` (ρ = ρ₀ for rows with no fluid neighbour) | δ-SPH | second-impact pressure rise = realisation variability (inside the baseline range); riders/wall-sparse benefits within spread; **free fliers up in 4/4 realisations vs 1/4**, not default-on |
| `mdbcOneSidedHydrostatic` | δ-SPH | wall sensor 8.83 kPa above every baseline value (n = 2); keep off |
| `unilateralWallContact` (+ two refinements) | δ-SPH | needs a new gate per failure (traction sign → own-pressure clamp → gravity/orientation → corner gate → damping); not principled, assumes static wall and fixed `g`; **parked, code on branch `parked/ceiling-uniwall`** (user, 2026-09-30) |
| `densityReinit` (summation-density floor) | δ-SPH | experiment only, never a default: WCSPH density stays a purely integrated quantity (user, 2026-09-30) |
| Antuono mask frozen across RK stages | δ-SPH | moot: the mask changes between *steps* (45 % of steps), not stages (0.2 %) |

## 4. State per scheme (2026-10-11)

| Scheme | State |
|---|---|
| δ-SPH / δ⁺-SPH (defaults) | **usable.** Zero free fliers on the baselines; kicks fixed; the sticking is cosmetic and open (W6) |
| ACSPH | **unusable at wall / free-surface contacts above nx ≈ 70** (Marrone 3.1 fails at step 1211, t\* ≈ 2.8, 2026-09-28; sloshingTank nx 100 blows up at t = 4.917). Best partial: `contact` + `isolatedZeroPressure` |

## 5. Open items

Worth: **H** blocks ACSPH use, **M** improves a result, **L** nice to have.

| # | Item | Scheme | Worth | Note |
|---|---|---|---|---|
| W1 | **Per-contact normal relative velocity** `(u − v_s)·n` instead of the kernel-weighted wall-pair divergence (Batty 2007's actual kinematic half) | ACSPH | H | The next idea, not tried. The kernel-weighted wall-pair divergence is not the normal velocity, which is why `wall` + divergence rectified noise. Alternative, more work: Chentanez & Müller / Inglis active set with the wall pressure as its own unknown (Band 2018 pressure boundaries are the SPH precedent). A sealed container stays incompatible with any unilateral wall unless a gap can open |
| W2 | sloshingTank nx 100 blow-up at t = 4.917, uid 958 | ACSPH | H | No forensics yet: which term launches the particle? First: rerun with `--cav contact --isoZero` (the best config); `probe_contactLine.py --toy sloshing` now has the ACSPH case branch |
| W3 | ACSPH closed-box pressure level grows exponentially (OPEN §9) | ACSPH | H | Cheap discriminating test, unrun: a uniform level in a **periodic** box (no walls). Growth there points at the solve; none at the wall closure. Any gauge fix (pin the mean or one row in a closed region) follows. Needed by any unilateral wall, which shifts a closed region's level |
| W4 | Complementarity inside the continuity equation, acoustically neutral | δ-SPH | M | Positive divergence at an air-reachable row booked as air entering the gap, not as particle expansion. Research, not a clamp (§3 shows why clamps fail). Only worth it if the sticking (W6) is ever to be fixed in δ-SPH rather than by the incompressible route |
| W5 | `loneDensityReset` flier mechanism | δ-SPH | L | For the 12 never-reset flier uids: were they inside a reset particle's support ≤ 0.1 s before birth? (Hypothesis: a reset particle re-joining with p = 0 instead of its carried p < 0 removes the tension that held a thin tip.) **Gate question for the user:** should slow (≤ √(gH)) 2-3-particle crest fragments count as numerical fliers at this resolution? The gate is not to be adjusted to make the reset pass |
| W6 | The sticking itself (22-25 ceiling rows, no dynamics) | δ-SPH | L | **Parked (user, 2026-09-30).** A fix has to come from the model (cohesion / sub-support fragments, or the incompressible route), not another gate. Small residues noted: a row past a wall row's centre can get a noPen factor < 0 (ratio clamp [0.25, 1] → factor ∈ [−1, 2], |e| > 1 under `impulse`); the mirrored velocity books 2 v_n against a force that carries `P_i` once |
| W7 | Rows with λ < 0.5 that Marrone's raw detection does not flag | both | L | The `vicinity` / `contact` sets are only as good as the raw set (nx 70, t\* 4.5-4.9, wall p to −23) |
| W8 | Small ACSPH follow-ups | ACSPH | L | `a_b` of moving walls in Eq. (61); the divergence should always use the no-penetration mirror while only the viscous Laplacian follows the slip choice (matters for no-slip ACSPH); Eq. (58) Taylor correction after the shift; re-run the six no-free-surface controls with video; step cost (~187 ms/step, OPEN_PROBLEMS §33) |
| W9 | Fluid-fluid half: continuous coverage blend for the Antuono switch (OPEN §1 step 2) | δ-SPH | M | Stays in `OPEN_PROBLEMS.md` §1; listed here because it is the other source of the flier population |

## 6. Method and rules

- **Principled fixes only** (user): from the equations being integrated, literature first, toys, physics-derived criteria,
  never a case threshold or another gate.
- **Look at the frames, and count fliers.** Scalar metrics missed domain-wide spray twice (2026-09-24). Every claim needs the
  streamed flier metric (`scripts/probe_contactLine.py`: fluid rows with an almost-empty support, fragments ≤ 3, their count,
  max speed, height above the bulk, min density) on the `off` baseline too, and the per-contact traces
  (`FlierTracker`, `scripts/analyze_flierTrace.py`; `scripts/align_run_videos.py` for time-aligned videos).
- **Realisations.** The flow is chaotic: n ≥ 4 seeds before calling a difference real (the loneReset batch showed a "+32 %" was
  the lowest baseline realisation).
- **Resume, don't re-run:** `storeMode='states'` checkpoints + `CaseSpec.resumeFrom` put the run in front of the event.
- Run rules from CLAUDE.md (video on, streamed progress, stall watchdog, one GPU run at a time).
- **Literature.** On disk (`literature/`): Batty 2007, Band 2018 (pressure boundaries, two papers), Schechter & Bridson 2012,
  Akinci 2013, Michel 2022, Dehnen & Aly 2012 (the corner clump is tension-driven, not a zero-gradient effect), Foster &
  Fedkiw 1996/2001, Losasso 2008. Cited by the old plans but not on disk: Chentanez & Müller 2011 and Inglis 2017 (both
  switch a contact as a whole, by the sign of the contact pressure and never the velocity sign), Ihmsen 2012. No paper
  found that quantifies the restitution of a lone SPH particle.

## 7. Decisions needed (user)

| # | Decision | Recommendation |
|---|---|---|
| D1 | The flier gate (W5): do slow 2-3-particle crest fragments count? | Yours; recorded so nobody edits the gate to make a fix pass |
| D2 | Priority of ACSPH contact work against the compressible plan | W3 first (cheap, narrows everything else), then W1; W4-W7 wait |

## 8. Where the evidence is

Archived plans in `docs/historic_plans/` carry the toy tables, forensics and the per-variant numbers. Videos and traces in
`results/` and `scripts/out_contactLine/`, `scripts/out_ceiling/` (gitignored). Parked code: branch `parked/ceiling-uniwall`.
Tests: `tests/test_timeCentredContinuity.py`, `tests/test_loneDensityReset.py`, `tests/test_artificialCompressible.py`.

## Status log

- 2026-10-11: plan opened by consolidating the three boundary plans. Read against the three plans, OPEN_PROBLEMS §1 / §8 / §9,
  PLANS.md. No code changed, no runs.
