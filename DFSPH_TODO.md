# DFSPH / incompressible — pickup TODO

Distilled from `DFSPH_IMPROVEMENT_PLAN.md` (452 lines) + `DFSPH_FINDINGS.md`
(1551 lines) so picking this back up doesn't mean re-reading ~2000 lines
first. This file is the punch list only — mechanism, evidence, and the full
part-by-part narrative stay in those two files; each item below names where
to go for the detail once you've picked one.

Current recommended default is unchanged: `divergenceFree` for general use,
`band2018pb` only for closed wall-bounded flows where tight density matters
more than particle-distribution smoothness (`DFSPH_IMPROVEMENT_PLAN.md`
"Recommendation").

## Worth picking up, roughly cheapest/most-concrete first

1. **Promote `columnCollapse`'s kernel to `Wendland4`, or decide not to.**
   Already measured to substantially reduce post-impact pairing (0.099→0.063
   final `pairedFraction`, no measured cost) at one resolution, one case.
   Before promoting: run longer (does the gap widen or plateau?) and check
   whether it transfers to `impact` (the other pairing case). Plan item 2,
   Part 59.
2. **`relaxationFactor=0.3`'s stability margin is ~4% on a bounded state**,
   not the ~15% its docstring quotes from the TGV family — cheap, isolated
   robustness win. `probe_boundaryOperatorTerms.py --mode spectrum`. Plan
   item 6.
3. **Cheap test for the `DensityEvolution.hybrid` mDBC hypothesis**:
   `computeMdbcDensity` runs on carried (not re-summed) density under
   `hybrid`, possibly why it dies at 286 steps at a wall but not
   periodically. Try re-summing for the extrapolation only. Plan item 5.
4. **One run settles whether `band2018pb`'s `columnCollapse` smoke-profile
   FAIL is a resolution artifact.** `--scheme band2018pb --cases
   hydrostaticColumn-64 columnCollapse sloshingTank --profile full --video`
   — `divergenceFree` already flipped 2/3 of this table from smoke-FAIL to
   full-PASS: unknown whether `band2018pb` does too. Plan item 3.
5. **Dam-break dissipation mechanism, `nx` convergence study.** Isolated to
   the incompressibility cycle (DF projection / Eq. 17 resample, net −8.5,
   85% of the loss) but not explained: discretization error that vanishes
   as `nx` grows, or a structural cost of the constraint? Independent of
   everything else, can run any time. Plan item 4.
6. **The real Morris et al. 1997 laminar viscosity term** — biggest lift
   here. `hydrostaticColumn`'s `noSlip`+`viscidNu` bounds the free-slip
   slosh but roughens the surface because the stock term
   (`wp_viscosityDelta.py`) is normal-projected (`mu_ij` along `x_ij`), no
   tangential stress. The approach-only-clamp half of this already landed
   (2026-09-05, `ACSPH_PLAN.md` step 5); what's left is the full `v_ij`
   vector as a new `DiffusionParameters`-wired option, gradcheck'd, with its
   own `deltaSPH` regression pass. Plan item 1, `DFSPH_FINDINGS.md` §1.14.

## Blocked / low priority — don't start here

- **`shearWave` vs [C] Fig. 3/4** — blocked on literature access
  (`literature/MANIFEST.md`). Plan item 7.
- **`'mirror'` wall-pressure mode's rough edges** — low priority, `'shepard'`
  is the shipped default and neither is needed. Plan item 8.
- **Re-scope before running, not just re-run as written**: the
  divergence-free half-state contraction under `minShift`, and warm-starting
  the divergence-free solve — both predate the Part 46-58 rewrite
  (`c637785`, Parts 46-47's shift gate) and may not mean what they used to
  against `omniIncompressible._solve`. Check relevance before running.
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

## Where the detail lives

- `DFSPH_IMPROVEMENT_PLAN.md` — current status tables, the case-by-case pass
  grade, pre-merge cleanup log.
- `DFSPH_FINDINGS.md` §9 — one paragraph per part (58 parts), fastest way to
  find *when/why* something changed. §1.1-§1.20 — durable physics lessons.
  §2 — negative results (don't re-try these). §7 — a bug table.
- `git log -p DFSPH_IMPROVEMENT_PLAN.md` — full prose narrative, part by
  part, for anything the above doesn't answer.
