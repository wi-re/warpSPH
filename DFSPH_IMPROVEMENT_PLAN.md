# warpSPH — Incompressible (VD+PS / DFSPH) Improvement Plan

Working document for the incompressible SPH path: `divergenceFree`
(`schemes/divergenceFree.py`, renamed from `dfsph.py` in the pre-merge
cleanup pass, 09-04), `dfsphReference` (troubleshooting artifact), `iisph`
(Part 33 baseline), `omniIncompressible` (Part 35 omniSPH-loop port), and
`band2018pb` (Part 45, Pressure Boundaries).

**This file is the current state and the actionable list only.** Durable
physics lessons, negative results, the literature/config/tooling reference,
and a one-line-per-part session index are in **`DFSPH_FINDINGS.md`**. The
full part-by-part investigation narrative (58 parts, 09-04 and earlier) is in
`git log -p DFSPH_IMPROVEMENT_PLAN.md` and `DFSPH_FINDINGS.md` §9 — this file
no longer carries it inline (cut in the Part 58 cleanup pass; nothing was
lost, `DFSPH_FINDINGS.md` §1.1–§1.20 and §9 already had it).

**Branch note.** This work landed on `dfsph-shift-pressure-gauge` and was
fast-forward-merged into `main` on 09-04 (commit `13b2c96`) after the
pre-merge artifact cleanup pass — see "Pre-merge cleanup" below for what that
covered (the omniSPH/SPlisHSPlasH comparison tooling, committed videos, dead
code, `mdbcMlsPressure`, the `dfsph.py` rename). That section's own list was
fully closed before the merge; nothing scoped there is still outstanding.

**Units note.** `cflFactor` on the incompressible cases multiplies the
particle **diameter** `dx`. Numbers in git history before commit `9a9bfe7`
are in the old `h`-based units (4x more travel per step for the same
number). `cflFactor = 0.4` today == the published Bender & Koschier constant.

---

## Recommendation (current, Part 56–58)

**`divergenceFree` is the right general-purpose default and needs no change**
(it already is one). As of Part 56 it does not diverge on any case in the
suite, including every wall-bounded one — the catastrophic failure mode this
whole track chased (`randomFlowIncompressible --bounded` NaNing at every
resolution) is fixed. Where it still has quality caveats, they are modest and
non-fatal: a closed box's density sits ~5–8% high against a strict 5% target,
and two cases (`impact`, `columnCollapse`) develop growing particle pairing
after an impact while the walls themselves hold exactly (`nPenetrating`
stays 0).

**`band2018pb` is a real trade-off, not a strict upgrade — reach for it
deliberately, not by default.** It tightens density control on closed
wall-bounded cases (~4–5% vs `divergenceFree`'s ~7–8%) at the cost of more
particle pairing/voids there, and it has no free-surface treatment at all (the
paper's method assumes a closed tank), so it inherits real problems on violent
free-surface cases (`dambreak`) and fails free-space bodies outright
(`staticBlob`). Use it for closed, wall-dominated flows where incompressibility
matters more than particle-distribution smoothness.

This is now codified where a future reader will actually see it:
`src/warpSPH/enumTypes.py`'s `IncompressibleSPHScheme` docstrings on
`divergenceFree` and `band2018pb`. Evidence: Part 56 (the fix), Part 57
(smoke-profile head-to-head across the whole boundary-case table), Part 58
(the two open questions from Part 57 re-verified at full resolution with
video — `DFSPH_FINDINGS.md` §9 rows 56–58).

---

## Status

**Suite: `tests/test_physics.py` + `tests/test_runner.py` = 92 passed / 0
failed / 0 xfail.** (Two pre-existing failures in *other* test files —
`test_implicitShiftingComparison.py`, `test_incompressibleKrylov.py` — are
recorded in "Known-open" below and have **not** been re-verified this
session; don't assume they're still exactly as described without a re-run.)

**Case status at the shipped defaults (`divergenceFree`):**

| case | status |
|---|---|
| `tgv`, `shearWave`, `staticBlob`, `randomFlowIncompressible` periodic, `kolmogorovIncompressible` (periodic / free space, no walls) | **pass**, clean. |
| `randomFlowIncompressible --bounded` / `--obstacle` | **holds at every resolution** (Part 56, `_SHIFT_WALL_PRESSURE = 'shepard'`), but density sits ~7–8% high against the harness's strict 5% band — a real, resolution-independent property of the closure, not a bug (`DFSPH_FINDINGS.md` §1.18–§1.19). |
| `hydrostaticColumn` | **holds** at its validated resolution (nx=64/128, 400 steps): `pressureSlopeRatio` ~1.0, `densityP05` ~0.94–1.0. A bounded undamped free-slip limit cycle remains in the top few rows (opt-in `XSPH_SCALE` decays it at a dissipation cost). Fails at smoke-profile nx=32/100 steps — that's a resolution artifact, not a real failure (Part 58). |
| `dambreak`, `columnCollapse` | **hold** — walls do not leak (`nPenetrating` 0). `columnCollapse` develops post-impact particle pairing (0.3%→9% over 400 steps under `Wendland2`, its default) that is cosmetic, not structural (Part 58); substantially reduced (→6.3%) by `kernel='Wendland4'` at otherwise-identical settings, not yet promoted (Part 59) — open item below. |
| `impact` | **pass** physically, but has a pre-existing pairing issue unrelated to any of this track's work. |
| `squarePatch` | **runs** — stable rotation; corner density loss is a documented method limitation ([BK] §5), not a bug. |
| `sloshingTank` | **holds** the full 7s SPHERIC TC10 run (nx=200); Sensor-1 in range. Fails at smoke-profile nx=60/150 steps — too short to build real pressure, not a real failure (Part 58). |

**`band2018pb`, graded across the same table (Part 57/58):** holds
`hydrostaticColumn`, `dambreak` (wall integrity), `randomFlowIncompressible`
(all variants, tighter density than `divergenceFree`) — but with growing
pairing/voids on the `randomFlow` variants, and `dambreak`'s free-surface
voids/spray are worse than `divergenceFree`'s. `staticBlob` fails outright
(Part 51 — no free-surface treatment in the method). `columnCollapse` at
**smoke** profile shows *worse* pairing growth than `divergenceFree`
(0.138 vs 0.033–0.066) — **not yet re-verified at full resolution**, open
item below.

---

## What's realistically open

Ordered roughly by how concrete/actionable each is, not by importance.

1. **The shear-carrying Morris viscosity term.** `hydrostaticColumn`'s
   `wallBC=noSlip` + `viscidNu` bounds the free-slip slosh but roughens the
   surface, because the stock `viscidNu` term (`wp_viscosityDelta.py`) is
   normal-projected — no tangential stress. Needs a real Morris et al. 1997
   laminar term (full `v_ij` vector, no approach-only clamp) as a new
   `DiffusionParameters`-wired option, gradcheck'd, with its own `deltaSPH`
   regression pass. Well-specified, not started. (`DFSPH_FINDINGS.md` §1.14.)
2. **`columnCollapse`'s post-impact particle pairing — partially triaged,
   Part 59.** Wall integrity is exactly correct (`nPenetrating` stays 0
   throughout); the clumping itself (`pairedFraction` 0.3%→9.0% after the
   collapse impact, Part 58) is **substantially reduced by switching the
   case's kernel from `'Wendland2'` to `'Wendland4'`** at otherwise-identical
   settings (nx=64, 400 steps, `cflFactor 0.2`, `n_h = 4` unchanged): final
   `pairedFraction` 0.099→0.063, peak 0.099→0.076, a real and widening trend
   from the impact onward, not sampling noise — with density, KE, and wall
   integrity all as good or better, and no measured cost (`DFSPH_FINDINGS.md`
   §9 row 59). **Not (yet) proposed as the case default** — one case, one
   resolution, one run each; open before that: does the gap keep widening or
   plateau on a longer run; does it transfer to `impact` (the other pairing
   case); and note it is not quite testing Dehnen & Aly 2012's own headline
   claim (any Wendland kernel is pairing-*stable* at any `N_H`; the Wendland2-
   vs-Wendland4 gap here is more likely a smoother-estimate-at-matched-`N_H`
   effect than a stability difference — see the paper, now in `literature/`
   as `dehnen2012`, promoted to the core set for exactly this question).
3. **`band2018pb` on `hydrostaticColumn-64`/`columnCollapse`/`sloshingTank` at
   full resolution — untested.** Part 58 re-verified `divergenceFree` on
   these three (2 flipped to PASS, `columnCollapse` didn't) and `band2018pb`
   on the `randomFlow`/`dambreak` trio, but never crossed the two: whether
   `band2018pb`'s smoke-profile `columnCollapse` FAIL (pairing 0.138, worse
   than `divergenceFree`'s) is a resolution artifact like `hydrostaticColumn`/
   `sloshingTank` turned out to be, or a real regression, is unknown. One
   `--scheme band2018pb --cases hydrostaticColumn-64 columnCollapse
   sloshingTank --profile full --video` run would settle it.
4. **The dam break's dissipation mechanism** (Ranked queue, formerly item 1).
   Isolated to the incompressibility cycle (DF projection / Eq. 17 resample,
   net −8.5, 85% of the loss) but not explained: discretization error that
   vanishes as `nx` grows, or a structural cost of the constraint? Needs an
   `nx` convergence study; independent of everything else, can run any time.
5. **The mDBC hypothesis for `DensityEvolution.hybrid`.** `computeMdbcDensity`
   runs on the carried (not re-summed) density under `hybrid`, which may be
   why `hybrid` dies at 286 steps at a wall but not periodically. Cheap test:
   re-sum for the extrapolation only.
6. **`relaxationFactor = 0.3`'s stability margin is ~4% on a bounded state**,
   not the ~15% its docstring quotes from the TGV family. Cheapest
   robustness win available, independent of everything else.
   `probe_boundaryOperatorTerms.py --mode spectrum`.
7. **`shearWave` vs [C]'s Fig. 3/4** — blocked on literature access
   (`literature/MANIFEST.md`), not actionable right now.
8. **The `'mirror'` wall-pressure mode's rough edges** (Part 56, low
   priority since `'shepard'` is the shipped default and neither is needed):
   (a) `'mirror'` + `_CLOSED_DOMAIN_GAUGE='always'` destabilises, unexplained;
   (b) its Adami body-force correction term is unimplemented (needs a vector
   *moment* of the neighbourhood, which `Interpolate` doesn't return) — moot
   until something calls this solve under gravity, which nothing does today.

**Two items to re-scope before picking up, not just re-run as written** —
they predate the Part 46–58 architecture changes (`c637785`'s rewrite,
Parts 46–47's shift gate) and may no longer apply as stated:
- Re-measuring the divergence-free half-state's contraction under `minShift`
  (`probe_boundaryOperatorTerms.py --mode diag`) — written against the old
  `solveDivergenceFree`; check it still means something against
  `omniIncompressible._solve`'s divergence pass before running it.
- Warm-starting the divergence-free solve — same caveat.

**Likely moot, recommend dropping unless someone has a specific reason to
want it:** a from-scratch two-solve `dfsph` scheme (the old ranked-queue
item 9, "stays last") — `band2018pb` + `divergenceFree` already cover the
practical need this was aimed at, and the queue itself never prioritized it.

---

## Pre-merge cleanup

Started 09-04. None of this blocks continued physics work; it blocks merging
to main cleanly.

**Done:**
- **`scripts/omnisph_compare/` and `scripts/splishsplash_compare/` removed**
  (23 files total, including the three committed `.npz` binaries) — both were
  explicitly "throwaway" per their own commit history and needed an external
  checkout (`~/dev/omniSPH`) or install (`pysplishsplash`) that isn't part of
  this repo. What each found is preserved in `DFSPH_FINDINGS.md` §1.12/§1.13/
  §1.15/§8 regardless; git history has the scripts themselves if ever needed
  again.
- **Committed video files removed** (`scripts/videos_band2018pb/`,
  `scripts/videos_dfsph_columnCollapse/`, `scripts/videos_dfsph_dambreak/`,
  ~28MB) — regenerable by their `make_video_*.py` scripts, which still exist.
  Stops further growth; does not shrink existing git history (would need a
  rewrite, not done and not proposed).
- **`scripts/probe_*.py` review — DONE, by the only bar that's actually
  checkable rather than a judgment call.** 8 of the original 59 removed
  across two passes, all confirmed cited *nowhere*: 7 DFSPH-track scripts on
  the first pass (`probe_band2018pbNearWall.py`,
  `probe_dambreakSurfaceGauge.py`, `probe_dfsphFactorCheck.py`,
  `probe_dfsphReferenceStaticBlob.py`,
  `probe_hydrostaticColumnDfsphSurfaceSource.py`,
  `probe_hydrostaticColumnDfsphTune.py`,
  `probe_hydrostaticColumnDfsphXsph.py` — several superseded early
  iterations of a sibling script that *is* cited, e.g.
  `probe_hydrostaticColumnDfsphSurface.py`) plus
  `probe_mdbcMlsPressureInstability.py` on the `mdbcMlsPressure` removal
  pass, since its whole purpose was probing the function that pass deleted.
  **The remaining 51 are all cited by at least one `.md` file somewhere in
  the repo** (checked repo-wide, not just the two DFSPH docs — six of them
  turned out to belong to `WCSPH_SHIFTING_PLAN.md`, a different, apparently
  still-active track, not DFSPH scratch work at all). Zero-references-
  anywhere was always the actual removal bar this session used, precisely
  because it needs no judgment call about whether a *cited* script's finding
  is "fully superseded" — that reopens editorial questions (is the citing
  doc itself current, would removing the tool make the citation
  unverifiable) this pass isn't positioned to answer for tracks it isn't
  working on. Nothing further to do here under this bar; going further would
  mean picking a new, more subjective one.
- **`modules/incompressible/incompressible.py`'s `solveIncompressible`
  cleaned up**: removed the ~150 lines of commented-out dead code left from
  the `c637785` rewrite (the `ShiftPressureGauge`/`minShift` gauge-selection
  path, superseded by `_CLOSED_DOMAIN_GAUGE`; the `BoundaryPressureMode.
  consistent` special-casing, handled externally by `applyConsistentCoupling`
  already; the commented-out Krylov-solver dispatch, since re-landed
  properly, see below; duplicate imports), and gated every remaining
  `print()` behind the function's own `verbose` parameter instead of firing
  unconditionally. Verified behaviour-preserving: full suite green,
  `gradcheck_incompressible` green, and `randomFlowIncompressible
  --bounded`'s KE trace bit-identical before/after.
  **Found and fixed along the way, not purely cosmetic:** the commented-out
  block included the `JacobiRelaxationMode.optimal` guard the "Known-open"
  list below already flagged as regressed (a real ValueError this solver is
  supposed to raise, silently dropped by the rewrite) — restored it, which
  fixes `test_incompressibleKrylov.py::test_optimalStepRejectedForConstantDensitySolver`
  (was failing consistently, now passes). **Left alone, on purpose:** the
  loop hardcodes `omega = 0.3`, silently shadowing the
  `relaxationFactor`-configured `omega` computed above it — clearly worth a
  look, but changing which relaxation factor a numerical solve actually uses
  is a behaviour change needing its own validation, not something to fold
  into a dead-code cleanup.
- **`wp_viscosityDelta.py`'s docstring fixed** — no longer claims a
  "Morris-style Laplacian" for the `inviscid=False` branch; states plainly
  that the same approach-only clamp that stabilises the artificial-viscosity
  branch also normal-projects this one, so it isn't a real Morris term
  (open item 1 above still needs the real one).
- `randomFlowIncompressible.py`'s and `incompressible.py`'s stale
  Part-48-era comments (calling the case's divergence "current" and the
  regression test "xfailed") corrected during the Part 56 session; already
  landed, noted here for completeness.

**Still open: nothing.** Every item this section originally listed —
`mdbcMlsPressure`, the probe-script review, and the `dfsph.py` naming pass —
closed the same day; see below for each.

**Done, 09-04 (later the same day):**
- **`schemes/dfsph.py` renamed to `schemes/divergenceFree.py`,
  `dfsph_step` to `divergenceFree_step`** — matches every other scheme
  file's convention of being named after its `IncompressibleSPHScheme`
  value (`omniIncompressible.py`, `band2018pb.py`, `dfsphReference.py`); the
  old "rename to `vdps.py`" idea is dropped (it hasn't been VD+PS-only since
  `c637785` either, so that name would have been wrong too). All call sites
  fixed: the two files that actually wire scheme dispatch
  (`schemes/__init__.py`, `schemes/builder.py`), every other `src/`/`tests/`
  reference, and every `scripts/probe_*.py`/`make_video_*.py` import that
  touched the module (aliases like `dfsphmod`/`D` were left as the scripts'
  own local names — only the module path changed, so
  `import warpSPH.schemes.divergenceFree as dfsphmod` still reads naturally
  where it's used). Verified: full suite green (112/112 including
  `test_incompressibleKrylov.py`), every touched script still parses.
  Historical prose in `DFSPH_FINDINGS.md` / `git log` keeps saying
  `dfsph.py`/`dfsph_step` on purpose — that was the name at the time; the new
  file's own docstring says so.

**`BoundaryPressureMode.mdbcMlsPressure` — REMOVED, 09-04 (later the same
day).** What was deprecated about it: it isn't just "measured worse", its
whole architecture was the wrong shape for the problem.
`computeMdbcPressure` (`modules/mdbc/pressure2025.py`, now deleted)
extrapolated the boundary pressure **once per step**, before the pressure
solve ran, and *carried* that value into the next step under relaxation
(`mdbcPressureRelaxation`, default 0.3) — which existed only to damp a
feedback loop the once-per-step lag itself created: a larger boundary
pressure pushes the fluid harder, which projects to an even larger boundary
value next step. Undamped (`relaxation = 1.0`) it doubled every step and
NaNed within 7-8 steps; over a real 900-step run at the published CFL it was
the worst configuration measured (`|rho-1|` 1.86e-1, worse than the shipped
baseline's 1.78e-1) and separately NaNed at t=0.21 under `inStepVelocity`
(`DFSPH_FINDINGS.md` §3 item 1). Part 41's `wallPressureExtrapolation`
(`modules/incompressible/wallPressure.py`, its own `'mls'` mode — a
**different** thing despite the shared name) closes the same gap properly:
it recomputes the wall pressure from the *current* fluid iterate **every
Jacobi sweep**, with no carried state, so there's no lag and nothing to feed
back — which is also why it's the thing Part 56 this session used to
actually fix `randomFlowIncompressible --bounded`, where `mdbcMlsPressure`
was never even a candidate.

**What actually got removed, once a coverage check confirmed it was safe:**
zero tests reference `mdbcMlsPressure`, `computeMdbcPressure`, or
`mdbcPressureRelaxation` (checked, not assumed), and the mode's only real
call site was already dead code (commented out, presumably since
`c637785`) — selecting it did exactly what `mdbcDensity` does today, a
silent no-op. Removed: the `BoundaryPressureMode.mdbcMlsPressure` enum value
and the now-pointless `mdbcPressureRelaxation` config field (with its
`incompressibleConfigToDict`/`dictToIncompressibleSPHConfig` round-trip
entries), `modules/mdbc/pressure2025.py` itself, the dead call-site comment
block in `schemes/divergenceFree.py`, and `scripts/
probe_mdbcMlsPressureInstability.py` (its entire purpose was probing the
now-deleted function; its finding is preserved in `DFSPH_FINDINGS.md`).
Fixed four other probe scripts that swept `mdbcMlsPressure` as one of
several `BoundaryPressureMode` arms (`probe_boundedIncompressibleBlowup.py`,
`probe_hydrostaticColumn.py`, `probe_consistentCoupling.py`,
`probe_randomFlowIncompressibleBoundaryModes.py`) to drop that arm instead
of erroring on it. Verified: full suite green (112/112),
`gradcheck_incompressible` green, a live `hydrostaticColumn` run plus a
config round-trip both work.

**Not a repo-cleanliness issue but worth knowing:** `sweeps/` (582MB) and
`export/` (23GB) are already gitignored — no action needed before merge,
just don't let anyone `git add -f` into them.

---

## Known-open, lower priority

- **`rotatingSquarePatch` corner density loss.** Resolution-independent
  `rho ≈ 0.506` at the four convex free-surface corners; `--scheme deltaSPH`
  holds 0.9998. **[BK] §5 documents this as a known method limitation** with
  a published remedy (ghost particles, Schechter & Bridson 2012, in
  `literature/` as `schechter2012`) — not an implementation bug. Two
  independent minor bugs on the case still want fixing: no `Case.timestep`
  hook (so `dt` never adapts) and it inherits `integrationScheme='rungeKutta2'`.
- **Nothing enforces `semiImplicitEuler`.** The PPE derivation is specific to
  it; `CaseSpec`'s default is `rungeKutta2` and no path checks. Candidate: a
  one-line assert/warn in `divergenceFree_step`.
- **`solveIncompressible`'s `JacobiRelaxationMode.optimal` guard — FIXED in
  the pre-merge cleanup pass (09-04).** The guard (`optimal` + the
  constant-density solver is a combination both papers rule out, since this
  solve's non-negativity clamp breaks the exact residual recurrence `optimal`
  needs) existed in the code as a commented-out block, dead since the
  `c637785` rewrite; restored while cleaning up the surrounding dead code.
  `test_incompressibleKrylov.py::test_optimalStepRejectedForConstantDensitySolver`
  now passes (was failing consistently).
- **`solveIncompressible`'s Krylov dispatch — RESTORED, same day, after the
  cleanup pass turned out to have regressed it.** The commented-out block the
  `optimal` guard came from also carried the dispatch itself
  (`solverType != relaxedJacobi -> solvePressureKrylov(...)`), already dead
  since `c637785` — removing it as "dead code" during the cleanup silently
  broke `test_incompressibleVariantRuns` (it still passed only because its
  assertions, "pressure is finite and non-negative", hold for the ordinary
  Jacobi fallback too; it never actually checked that Krylov ran). Restored
  to match `divergenceFree.py`'s own live Krylov dispatch exactly (same
  primitives, `gauge='nonnegative'` here vs that solve's `gauge='center'`,
  since only this one has a tensile/free-surface clamp to match). This is
  **not** re-opening the Parts 43-44 near-wall finding — Krylov still breaks
  down on every wall-bounded case tried, and the restored dispatch changes
  nothing about that; it only restores the ability to *select* a Krylov
  solver at all; a caller doing so on a wall-bounded case still hits the
  documented breakdown, unguarded (matching `divergenceFree.py`'s own
  dispatch, which has no such guard either). `test_incompressibleKrylov.py`'s
  20 tests build their case on periodic `tgv` (wall-free, well-conditioned),
  which is why they were never in a position to catch the regression from the
  physics side — only a "did the code path actually run" check would have,
  and now does again. All 112 tests green;
  `test_minresGivensMatchesDenseLstsq` is a pre-existing floating-point flake
  (fails ~1 run in a few, passes on retry, unrelated to this).
- **Two mDBC slip defects** (`modules/mdbc/velocity.py`). `freeSlip` computes
  `u_f - 1*(u_f.n)n` while its comment says `- 2*`; `noSlip` applies the same
  projection undocumented. Fixing measures *worse*, so comment and code
  should agree rather than the code changing. Separately, `noSlip`'s
  `2 u_wall` term is dead code (reads a ghost row nothing but
  `BCType.constant` writes) — latent until the first moving no-slip wall.
- **Two-way coupling is absent.** [BWJ23] Eq. 35's `f_{k<-i}` is never
  applied. Fine today (all walls static); a moving-body case would silently
  get one-way coupling.
- **`DensityEvolution` + `BoundaryPressureMode.plain` is a trap** — `plain`
  skips the mDBC extrapolation and the non-summation modes skip the re-sum,
  so `kind != 0` rows would never update. Documented in the enum, not
  guarded.
- **Ghost particles (`kind == 2`) are lumped in with boundaries** by the
  `kj == 0` test. Correct for this scheme family, but no measurement
  separates them.
- **Full-suite failures unrelated to this track:**
  - `test_implicitShiftingComparison.py::test_automaticImplicitShift_convergesLikeHandBuilt`
    — a flake (~1 run in 3 as of Part 48; passed on a single re-run 09-04, not
    enough to call it fixed).
  - `test_incompressibleKrylov.py::test_optimalStepRejectedForConstantDensitySolver`
    — **fixed, 09-04** (all 20 tests in that file pass now); see the
    Krylov-guard item above.
- **Parts 35–38's `wp_dfsph_factor.py` `ki == 0` change** touches a
  `@wp.kernel` and was gradcheck'd at the time (Part 37) but not re-verified
  since.

---

## History

The full part-by-part narrative (58 parts as of 09-04) lives in
`DFSPH_FINDINGS.md`:
- **§9** — one-paragraph-per-part session index, the fastest way to find
  when/why something changed.
- **§1.1–§1.20** — durable physics lessons, written to outlive any one part
  (e.g. §1.18–§1.20 are this session's: the Robin-consistency requirement for
  a near-wall Jacobi, the pressure-projection-closure family, and the
  density-vs-structure trade-off implicit boundary treatment makes).
- **§2** — negative results, so they don't get re-tried.
- **§7** — a bug table.
- **git log -p DFSPH_IMPROVEMENT_PLAN.md** — the actual prose, part by part,
  for anything the above doesn't answer.

Headline sequence, for orientation: Parts 1–32 built and debugged the
original VD+PS scheme and the `iisph`/`dfsphReference` baselines. Parts
35–41 ported omniSPH's loop, cross-checked it against SPlisHSPlasH and
omniSPH's own Python bindings, and landed the per-iterate wall-pressure
closure (`wallPressureExtrapolation`) that Part 56 later reused. Parts 42–45
chased the near-wall constant-density operator's rank deficiency to its root
and landed `band2018pb`, the extended-PPE scheme that removes it structurally.
Commit `c637785` ("tmp commit") then rewrote `dfsph_step` onto
`omniIncompressible._solve`, which Parts 46–49 characterised and repaired
(the `tgv`/bounded-box regression, the dropped no-penetration shift). Parts
50–53 fixed two real bugs in the `band2018pb` port and a lattice-density
sampling bug that had been misdiagnosed as several unrelated case failures.
Parts 54–58 are this session: the sampler fix's real scope, its interaction
with a since-fixed-differently wall-pressure bug, the actual fix
(`_SHIFT_WALL_PRESSURE`), and the resulting scheme-selection recommendation
at the top of this file. Part 59, a follow-up session: `columnCollapse`'s
post-impact pairing responds to a kernel-order change (Wendland2→Wendland4),
and `literature/dehnen2012` (the Wendland-for-SPH origin paper) was promoted
from the background set to the curated core for it.
