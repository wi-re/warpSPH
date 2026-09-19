# DFSPH / incompressible — pickup TODO

**RETIRED 2026-09-19.** Moved to `docs/historic_plans/` alongside
`DFSPH_IMPROVEMENT_PLAN.md`/`DFSPH_FINDINGS.md` — the track reached a stable,
documented recommendation (`divergenceFree` default, `band2018pb` as a
deliberate trade-off) and every item that was still open here is either
closed (see below), was checked and found to target dead/superseded code
(also below), or — the one item worth carrying forward, the Morris et al.
1997 viscosity term — is now `OPEN_PROBLEMS.md` item 7. The smaller
"Known-open"/"Blocked" bullets further down were not individually promoted;
they stay here for the record, not because they were forgotten.

Distilled from `DFSPH_IMPROVEMENT_PLAN.md` (452 lines) + `DFSPH_FINDINGS.md`
(1551 lines) so picking this back up doesn't mean re-reading ~2000 lines
first. This file is the punch list only — mechanism, evidence, and the full
part-by-part narrative stay in those two files; each item below names where
to go for the detail once you've picked one.

Current recommended default is unchanged: `divergenceFree` for general use,
`band2018pb` only for closed wall-bounded flows where tight density matters
more than particle-distribution smoothness (`DFSPH_IMPROVEMENT_PLAN.md`
"Recommendation").

## Closed this session (2026-09-19)

- **`columnCollapse`'s kernel `Wendland2` -> `Wendland4` — decided against.**
  The 400-step win (0.099→0.063 final `pairedFraction`) does not survive a
  3x-longer run (gap closes, slightly reverses: 0.174 vs 0.191 at 1200 steps)
  and does not transfer to `impact`, the other pairing-prone case (Wendland4
  is *worse* there: 0.183 vs 0.221 final). A real but transient early-post-
  impact effect, not a persistent reduction — not promoted anywhere, no code
  changed. Also surfaced in passing: at 1200 steps `columnCollapse` under
  `Wendland2` shows one `nPenetrating` crossing where the 400-step
  characterisation called wall integrity exactly correct — that claim was
  scoped to 400 steps, not indefinitely. `DFSPH_FINDINGS.md` §9 Part 60, §2
  negative-results table.
- **`relaxationFactor=0.3`'s stability margin — measured, a real bug found
  and fixed along the way, default NOT changed.** `--mode spectrum` confirms
  the margin shrinks with resolution on a bounded wall subproblem (12.7% at
  nx=32 down to 7.1% at nx=128, vs the ~15% the docstring quoted from the
  un-walled TGV family); `boundaryOperatorTerms='full'` (not shipped) is
  already outside the window at every resolution measured, a latent trap if
  ever selected without also changing `relaxationFactor`. Along the way,
  found and fixed a real bug: the constant-density solve hardcoded
  `omega = 0.3` inside its loop, silently ignoring
  `relaxationFactor` regardless of what it was configured to (the exact
  "Known-open" item `DFSPH_IMPROVEMENT_PLAN.md`'s pre-merge cleanup log had
  flagged and deliberately left for its own validation) — fixed,
  bit-identical at the unchanged default, full suite green. With the bug
  fixed, a real A/B of `0.3` vs `0.27` shows a genuine trade (density bounds
  tighten slightly; peak `maxVelocity` gets worse by a margin that *grows*
  with resolution, +21% at nx=128) rather than the "free" margin the first,
  broken A/B attempt seemed to show — not adopted, `relaxationFactor` stays
  `0.3`. `DFSPH_FINDINGS.md` §9 Part 61.
- **`band2018pb`'s `columnCollapse`/`sloshingTank` smoke-profile FAILs —
  settled, not a resolution artifact, but not new regressions either.**
  `--scheme band2018pb --cases hydrostaticColumn-64 columnCollapse
  sloshingTank --profile full --video`: `hydrostaticColumn-64` flips to PASS
  (same story as `divergenceFree`'s, Part 58) but `columnCollapse` and
  `sloshingTank` don't — both still fail the harness's automated grade at
  full resolution. Neither is new information, though: `columnCollapse`'s
  pairing growth (0.026→0.084) converges to the same magnitude as
  `divergenceFree`'s own already-accepted, cosmetic post-impact clumping on
  this case (Part 58/59); `sloshingTank`'s failure (voids 0.3%→3.0%, peak
  4.4%) is the same free-surface void gap already documented on
  `dambreak`/`randomFlow` (Part 51), just on a third case. `DFSPH_FINDINGS.md`
  §9 Part 62. Videos in `sweeps/validate-band2018pb-full-20260919-125858/`.
- **`DensityEvolution.hybrid` mDBC hypothesis — closed as moot, not
  investigated.** The "cheap test" this item proposed (re-sum for the
  extrapolation only) turned out untestable: `DensityEvolution` is dead code
  on the production `divergenceFree`/`omniIncompressible` path (both branches
  that used to read it are commented out since the 09-02 `band2018pb` rewrite,
  `71a8ae7`, and no other scheme reads it either) — confirmed by code reading
  and by `probe_densityEvolution.py` giving identical results across all three
  settings. Part 10's `hybrid`-dies-at-286-steps finding was real for the code
  that existed then; nothing today reads the config field it was made
  through. Reviving it is a design decision (re-wire into the current step
  function, decide whether `hybrid`'s semantics still make sense
  post-rewrite), not a bug fix — not attempted here. `DFSPH_FINDINGS.md` §9
  Part 63.

## Worth picking up, roughly cheapest/most-concrete first

Empty — the one item that was here (the real Morris et al. 1997 laminar
viscosity term, biggest lift in this file) moved to `OPEN_PROBLEMS.md` item 7
when this track retired (2026-09-19). See that entry for the full mechanism
and next step.

## Blocked / low priority — don't start here

- **`shearWave` vs [C] Fig. 3/4** — blocked on literature access
  (`literature/MANIFEST.md`). Plan item 5.
- **`'mirror'` wall-pressure mode's rough edges** — low priority, `'shepard'`
  is the shipped default and neither is needed. Plan item 6.
- **Re-scope before running, not just re-run as written**: the
  divergence-free half-state contraction under `minShift`, and warm-starting
  the divergence-free solve — both predate the Part 46-58 rewrite
  (`c637785`, Parts 46-47's shift gate) and may not mean what they used to
  against `omniIncompressible._solve`. Check relevance before running.
- **Dam-break dissipation mechanism, `nx` convergence study — blocked, not
  "can run any time" as it looked.** `probe_dambreakEnergyBudget.py` (the
  source of the "net −8.5, 85% of the loss" figures) crashes against current
  code (`KeyError: 'a_DF'`): it captures `DIVERGENCE_SOLVER='vdps'`'s
  `solveDivergenceFree`, but the shipped default is `'omni'`
  (`_omniPass`/`omniIncompressible._solve`), which never calls it. Needs the
  probe's capture points re-scoped to the `'omni'` path's own force terms
  before a convergence sweep is worth running. `DFSPH_FINDINGS.md` §9 Part 64,
  Plan item 3.
- **Likely moot, drop unless someone has a specific reason**: a from-scratch
  two-solve `dfsph` scheme — `band2018pb` + `divergenceFree` already cover
  the practical need.

## Known-open, lower priority (not blocking, not scheduled)

- `rotatingSquarePatch` corner density loss is a documented method
  limitation ([BK] §5), not a bug — but the case is missing a `Case.timestep`
  hook (`dt` never adapts) and inherits `rungeKutta2` instead of
  `semiImplicitEuler`; both are easy, unrelated fixes.
- Nothing enforces `semiImplicitEuler`, which the PPE derivation actually
  requires. Candidate: one-line assert/warn in `divergenceFree_step`.
- Two mDBC slip defects in `modules/mdbc/velocity.py`: `freeSlip`'s comment
  says `-2*` but the code computes `-1*` (fixing it measures worse, so leave
  code as-is and fix the comment); `noSlip`'s `2 u_wall` term is dead code,
  latent until the first moving no-slip wall.
- Two-way coupling ([BWJ23] Eq. 35's `f_{k<-i}`) is never applied — fine
  today since all walls are static, but a moving-body case would silently
  get one-way coupling.
- `DensityEvolution` + `BoundaryPressureMode.plain` is a trap (skips both the
  mDBC extrapolation and the re-sum, so `kind != 0` rows never update) —
  documented in the enum, not guarded.
- Ghost particles (`kind == 2`) are lumped in with boundaries by the `kj==0`
  test — correct for this scheme family, but nothing measures them
  separately.
- `wp_dfsph_factor.py`'s `ki == 0` change (Parts 35-38) was gradcheck'd at
  the time (Part 37) but not re-verified since.
- **`omniIncompressible`'s `'mls'` wall-pressure regression is already
  tracked in `OPEN_PROBLEMS.md` item 2** (confirmed a genuine Jacobi-solver
  amplification instability, not a sign bug, and independent of the
  `ghostOffsets` sign fix) — don't re-diagnose this from scratch here.

## Full-suite flakes unrelated to this track

- `test_implicitShiftingComparison.py::test_automaticImplicitShift_convergesLikeHandBuilt`
  — flaky (~1 run in 3 as of Part 48), not re-verified this session.
- `test_incompressibleKrylov.py::test_optimalStepRejectedForConstantDensitySolver`
  — already fixed (09-04), all 20 tests in that file pass.
- `test_runner.py::test_everyCaseDeclaresItsParamsAsScalarsOrLists` — not
  flaky, deterministically fails: `dambreak.surfacePressureProbes` is a tuple,
  not a scalar/list/dict. Confirmed pre-existing (predates this session,
  found via Part 61's git-stash A/B), unrelated to DFSPH/incompressible.

## Where the detail lives

- `DFSPH_IMPROVEMENT_PLAN.md` — current status tables, the case-by-case pass
  grade, pre-merge cleanup log.
- `DFSPH_FINDINGS.md` §9 — one paragraph per part (64 parts), fastest way to
  find *when/why* something changed. §1.1-§1.20 — durable physics lessons.
  §2 — negative results (don't re-try these). §7 — a bug table.
- `git log -p DFSPH_IMPROVEMENT_PLAN.md` — full prose narrative, part by
  part, for anything the above doesn't answer.
