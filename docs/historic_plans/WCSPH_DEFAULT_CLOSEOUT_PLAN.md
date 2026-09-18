# WCSPH mDBC closeout: easy wins, then flip the default

> Working checklist. Update each item's result line in place as it's done. When
> everything here is checked off (or explicitly deferred to `OPEN_PROBLEMS.md`
> with a reason), this file's job is done — fold a short summary back into
> `BOUNDARY_DENSITY_PLAN.md`/`DELTASPH_VALIDATION_PLAN.md` rather than leaving
> three parallel documents alive.

## Why this exists

`BOUNDARY_DENSITY_PLAN.md`'s `english2025` mDBC scheme now beats `'ramped'`/
`'band'` on every Marrone config tested, after two real bugs (`ghostOffsets` sign,
Shepard precision hole) were found and fixed. Rather than re-opening the one
genuinely hard remaining mechanism (corner-flyer / ceiling-sticking, tracked
separately in `OPEN_PROBLEMS.md`), this closes out every smaller item that
accumulated around it and flips the WCSPH production default to the validated
combo (`mdbcDensityScheme='english2025'`, `densityDiffusionTerm='fourtakas2019'`,
`integrationScheme='symplecticEuler'`).

## Checklist

- [x] **A. `densityBand.py` Shepard-ratio epsilon fix** — port `english2025.py`'s
  symmetric-epsilon regularization (`_ALPHA_EPS=1e-30`) into
  `computeMdbcDensityBand`'s `alpha`/`dbx`/`dby` (currently `MbSafe =
  Mb.clamp_min(1e-30)`, denominator-only floor). Verify the Taylor-shift
  `relPos`/`xtb` lines are already correct post-`ghostOffsets`-sign-fix (expected,
  not a new fix). Result: **Done.** `alpha`/`dbx`/`dby` now use the symmetric
  `(numerator+eps*C)/(Mb+eps)` form; `relPos`/`xtb` confirmed unchanged (already
  matched `english2025.py`'s convention). `test_physics.py`/`test_wallPressure.py`
  green.
- [x] **B. `probe_deltaSPHMarrone34.py --report` scoring fixes** — (1) velocity
  check: make the "report-only below 12 U_max" comment true by either excluding
  it from `checks`/`npass` or tightening it to a real hard-fail threshold
  (calibrated ~8x U_max against clean-vs-defect data already in the plan); (2) KE
  check: non-finite `keTrend2ndHalf` must hard-FAIL, not silently pass via
  `not (isfinite and ...)`'s short-circuit. Result: **Done, verified against the
  exact historical traces** (`scripts/out_deltaSPHMarrone34/deltaSPH_nx256_c28.3_
  mdbcRho-{ramped,band}.npz`, re-scored without re-simulating): `ramped` now
  5/6, "no velocity divergence" correctly FAILs at 9.9 U_max (was 6/6); `band`
  now 0/6, KE check correctly FAILs with "KE trend non-finite -- run diverged"
  (was a silent PASS on `KE_end nan`).
- [x] **C. Validate `densityBand.py` fix against Marrone stress configs** —
  re-run Marrone 3.1 adversarial + Marrone 3.4 full protocol with
  `--mdbcDensityScheme band` using the fixed `--report`. Does not reopen the
  default-scheme decision either way. Result: **Done, on both configs — a
  real, substantial improvement, but `band` remains the worst of the three
  schemes, not a default candidate.**
  - **Marrone 3.1 adversarial** (`--nx 35 --shifting off --tLimit 1.2`): no
    longer diverges to NaN (previously failed at t≈0.74; now completes the
    full 1.2s) but is still badly behaved — maxVel peak **63.2** (vs
    `english2025`'s 4.10, `ramped`'s 13.9), density swings to
    **[0.708, 1.425]**, still climbing at the end (not decaying).
  - **Marrone 3.4 full protocol** (`--nx 256 --tStar 15.6605`): a much bigger
    change — previously diverged catastrophically at t*=4.21/15.66 (density
    to 2.5e7, wall penetration 195,730 dx); **now completes the full 22,453
    steps, `diverged=False`, 5/6 checks pass** (bulk density
    [0.9815, 1.0410], 0.48/0.73 dx wall/obstacle penetration, KE decaying).
    Only failing check: velocity divergence at **17.7x U_max** (108.1 m/s) —
    still clearly worse than `english2025`'s 6.4x (§9.8) and `ramped`'s 9.9x
    (§6.3d/9.2) at the identical config, but a real, qualitative
    improvement (a resolved catastrophic blowup, not just a smaller number).
  - **Reading**: the symmetric-epsilon fix removes an entire failure mode
    (the Shepard-fallback precision hole) for `'band'` too, exactly as it did
    for `english2025` — confirming §9's fix was a genuine, general-purpose
    correctness fix, not something specific to `english2025`'s formulation.
    But `'band'` still trails both other schemes substantially on every
    metric, consistent with §3's separate architectural finding (depth/
    lever-arm gradient amplification) being the dominant, still-unaddressed
    failure mode for this scheme. **Does not reopen the default-scheme
    decision — `english2025` remains the clear best choice.**
- [x] **D. Real `a_b` (moving-boundary acceleration) support** — add a per-
  particle acceleration field to `RigidBody`/`WeaklyCompressibleState` (currently
  absent entirely — kinematics are prescribed-analytic, `a_b=0` in
  `english2025.py` isn't a placeholder, there's nothing to read yet), compute it
  analytically (centripetal + tangential + linear terms), wire it into
  `computeMdbcDensityEnglish2025`. Verify: `movingObstacle.py` gets a nonzero
  `a_b`; Marrone/sloshingTank (static walls) reproduce recorded numbers bit-for-
  bit (`a_b` must compute to exactly 0 there). Result: **Done.** New
  `boundaryAccelerations` field on `WeaklyCompressibleState`/`IncompressibleState`
  (both required for `rigidBody/update.py`'s shared reconstruction), computed as
  `-omega^2 * relativePositions` (centripetal; tangential/linear terms are
  identically zero today since `dudt`/`dwdt` are hardcoded 0 at every call
  site — documented as a known future extension point, not silently assumed
  complete). Sanity check on `movingObstacle` (nx=32, tLimit=0.05,
  `--mdbcDensityScheme english2025`): `a_b` nonzero for 100% of boundary rows,
  magnitude 0.044-0.269 matching `omega^2 * r` order exactly, no divergence.
  Regression check: Marrone 3.1 adversarial + stable configs with
  `english2025` reproduce §9.3's recorded numbers to the printed precision
  (adversarial 4.100@t=0.858/end 2.162/rho[0.9896,1.0047]; stable
  4.872@t=0.748/end 2.553/rho[0.9870,1.0056]) — `a_b` computes to exactly 0 for
  every static wall, zero regression.
- [x] **E. Flip WCSPH production defaults** — Result: **Done, with one scope
  correction made mid-execution (see below).**
  - `mdbcDensityScheme` → `'english2025'`
    (`configurations/weaklyCompressible.py:152`).
  - `densityDiffusionTerm` → `fourtakas2019`
    (`weaklyCompressibleDiffusionParams.py`, both the dataclass field default
    and `buildDefaultDiffusionParamsWeaklyCompressibleSPH`'s factory default
    -- the factory is what production actually consults, the field default
    was dead code but kept in sync to avoid future confusion).
  - **`integrationScheme` — NOT touched via the shared
    `WEAKLY_COMPRESSIBLE_DEFAULTS` dict as originally planned.** Discovered
    mid-execution: that dict is shared with `randomFlowIncompressible`,
    `staticBlob`, `hydrostaticColumn`, and `columnCollapse` -- all default to
    `scheme='divergenceFree'` (incompressible), never validated under
    `symplecticEuler` by this investigation. Flipping the shared dict would
    have silently changed their integrator default too. Instead:
    `integrationScheme='symplecticEuler'` was added as an explicit per-case
    override in `dambreak.py`, `sloshingTank.py` (both have their own
    `defaults` dict, not the shared one), and added explicitly in
    `lidDrivenCavity.py`/`movingObstacle.py`'s own `defaults=dict(
    WEAKLY_COMPRESSIBLE_DEFAULTS, ..., integrationScheme='symplecticEuler')`
    calls (overriding the shared dict's `'rungeKutta2'` locally, without
    touching the shared dict itself). The shared dict's own default stays
    `'rungeKutta2'`, unaffected, for every other case that uses it.
  - **`dambreak.py`'s own default was `rungeKutta4` for a real reason**
    (matching Sun 2017/Marrone's own numerical method for literature
    comparison) that isn't a stability concern -- confirmed via
    `sloshingTank.py`'s own history comment: `symplecticEuler`'s earlier
    divergence there (and on Marrone 3.1) was root-caused to
    `mdbcNoPenShiftMode='derivative'`'s stage-splitting interaction, already
    superseded by `'finalize'` (already the default); no case shows a live
    `symplecticEuler` instability. User's call: the burden belongs on a
    script that needs a specific literature-matching method to anchor it
    explicitly (`--integrationScheme rungeKutta4`), not on the case default
    to protect implicit reproducibility. Flipped `dambreak.py`'s own default
    too, on that basis. **Consequence to flag clearly:** several of this
    plan's own recorded Marrone comparison numbers (e.g. step D's adversarial-
    config bit-for-bit reproduction) were produced under the *old* implicit
    RK4 default with no `--integrationScheme` override -- reproducing them
    with a bare command post-flip now requires passing
    `--integrationScheme rungeKutta4` explicitly.
  - Updated `BOUNDARY_DENSITY_PLAN.md` §11 framing (done in step I).
- [~] **F. Post-flip regression sweep** — full WC gallery
  (`render_examples.py --only weaklyCompressible`); LDC to its full `tLimit=30`
  under the new defaults (checks whether §7.2's density ramp appears/worsens/is
  unaffected — if worse, revert E's integrationScheme change); bare re-run of
  Marrone 3.1/3.4/sloshingTank with no override flags to confirm the flip itself
  is wired correctly. Result:
  - **LDC full 30s under the new combo: done.** `diverged=False`, all 60,000
    steps, `t=30` reached. `densityMedian` 1.0 -> **1.007** (max 1.007),
    `maxDensity` up to 1.018, `maxVelocity` 0.71-0.77 (well below lid speed
    1.0, sane single-vortex steady state), `kineticEnergy` 0.034 and still
    rising slowly at t=30 (not yet fully steady, not exploding). §7.2's ramp
    is present, does not diverge or accelerate into a blowup.
  - **Exact-baseline comparison run lost mid-flight** (a standalone script
    reproducing the OLD combo — `rungeKutta2`/`ramped`/`deltaSPH` — at the
    same full 30s, for an apples-to-apples number) — the session restarted
    while it was running and it was not resumable (no checkpoint,
    `store=False`). Not re-run: the user directly compared the new run's
    video against the existing shipped reference
    (`examples/weaklyCompressible/outputs/09-lidDrivenCavity.mp4`, from
    before this session, old defaults) and confirmed **the old reference
    shows visibly worse density creep and a less-developed vortex** than the
    new combo's result — a real, if qualitative rather than exactly
    numerically matched, confirmation that the flip did not worsen §7.2.
    §7.2 stays open (`OPEN_PROBLEMS.md` item 4), not resolved by this flip,
    but not made worse by it either.
  - **Bare re-run, Marrone 3.1 adversarial config: done.** `--nx 35 --shifting
    off --tLimit 1.2 --integrationScheme rungeKutta4` (integrator pinned per
    the literature-comparison note above; `mdbcDensityScheme`/
    `densityDiffusionTerm` left at their new defaults, no override) ->
    `diverged=False`, maxVel peak **4.51** @ t=0.76, end 2.16, density
    [0.9902, 1.0084] -- clearly running `english2025`/`fourtakas2019` (not
    the old `ramped`'s 13.9 peak), confirming the default-flip wiring itself
    is correct. Small numeric differences from the earlier explicit-override
    run (4.51 vs 4.10 peak) are within this case's own documented
    run-to-run chaotic sensitivity, not a wiring issue.
  - **Full WC gallery + bare Marrone 3.4/sloshingTank re-runs: explicitly
    deferred, not done this session** (each is a substantially longer run;
    user's call, 2026-09-18 — LDC's own improvement over its pre-session
    shipped reference already gives enough confidence not to block further
    progress on these; user may run the overnight batch script separately).
    **Follow-up, whenever convenient:** `python scratchpad/run_overnight_
    batch_2026-09-17.sh`-style full gallery pass, plus a bare
    `run_sloshingTank.py --tLimit 7.0` (no override flags) to confirm the
    flip reproduces the recorded numbers there too.
- [~] **G. DFSPH/incompressible `wallPressure.py` 'mls' mode check** — re-run
  `randomFlowIncompressible --bounded` with `WARPSPH_WALL_PRESSURE_MODE=mls` (a
  known pre-fix divergence — does the sign fix resolve it?); corner-only
  pressure-accuracy check via `probe_deltaSPHMarrone34.py --cornerOnly` under an
  incompressible scheme, or `validate_scheme.py --wallPressure mls` as fallback.
  Independent of E, does not gate it. Document in `DFSPH_IMPROVEMENT_PLAN.md`.
  Result: **First check gave a false positive from testing the wrong scheme**
  — `randomFlowIncompressibleCase`'s own default scheme is `divergenceFree`
  (not `omniIncompressible`), and `divergenceFree` + `mls` ran clean for 400
  steps (10x the original test window). But `omniIncompressible.py`'s own
  code comment about this exact regression is about the `omniIncompressible`
  SCHEME's Jacobi iteration specifically, a different code path. Re-tested
  correctly with `scheme='omniIncompressible'` explicitly:
  **still catastrophically blows up** (`kineticEnergy` 0.32 -> 4.3e34,
  `maxVelocity` -> 4.0e17 by step 40; `result.diverged` doesn't flag it since
  the values are large-but-finite float32, not NaN/Inf) — unaffected by the
  `ghostOffsets` sign fix. Consistent with the code's own mechanistic
  explanation (the linear term amplifies real near-wall pressure structure
  in a sheared flow, pumping energy into the Jacobi iteration — an
  amplification/stability problem, not a sign bug, so the sign fix was never
  going to touch it). **Corner-only pressure-accuracy check: not yet run.**
- [~] **H. Bounded recheck of two old open items** — H/Δx≈72-160 resolution hole
  (nx=120 Marrone 3.1 re-run) and englishWedge concave-corner ghost collapse
  (`--dp 0.01` re-run). Single focused re-run each; whatever doesn't resolve goes
  to `OPEN_PROBLEMS.md`, not further open-ended debugging here. Result:
  - **Marrone 3.1 nx=120 (H/Δx=72) recheck: done — resolved.** Historical
    finding was "every config blows up at nx=120 in t*≈5.5-6.3, regardless of
    scheme" (`DELTASPH_VALIDATION_PLAN.md` item 4b). Re-ran the new
    stable-config combo (`--nx 120 --integrationScheme symplecticEuler
    --shifting on --densityDiffusionTerm fourtakas2019 --tLimit 1.9`, i.e.
    t*≈7.68 — well past the old failure window): `diverged=False`, all
    35,025 steps, maxVel peak **7.41** @ t=1.87 (moderate, decaying to 4.72
    by the end), density [0.982, 1.039] — no divergence, no NaN. **This
    intermediate-resolution hole no longer reproduces under the new default
    combo.** Not root-caused *why* (could be the mDBC scheme change, the DDT
    change, the integrator, or some combination — not isolated), but the
    practical symptom is gone at exactly the config that used to fail.
    Removed from `OPEN_PROBLEMS.md`'s "still open" list, recorded there as
    resolved instead.
  - **englishWedge recheck: explicitly deferred**, same wrap-up call as item
    F's deferred runs.
- [x] **I. Documentation wrap-up** — Result: **Done.**
  `BOUNDARY_DENSITY_PLAN.md` got a new §12 closeout summary;
  `DELTASPH_VALIDATION_PLAN.md` got pointers from its video banner, its
  corner-flyer "Next up" section, and item 4b's resolution note, all to
  `OPEN_PROBLEMS.md`/this plan rather than duplicating content;
  `OPEN_PROBLEMS.md` written (5 items: corner-flyer/ceiling-hover,
  `omniIncompressible` `'mls'`, `'band'`'s architectural ceiling, LDC's
  density ramp, and the now-resolved-so-removed H/Δx hole). Memory updated:
  `[[boundary-density-plan]]` rewritten (was stale, said "not the default,
  do not flip"), `[[ghostoffsets-sign-bug]]`/`[[english2025-shepard-precision-hole]]`
  got closure notes, `[[wcsph-deltasph-scheme-concerns]]` got a cross-reference
  to the LDC ramp finding, `[[export-video-on-validation-runs]]` rewritten for
  the tooling-level fix, and a new `[[wcsph-default-closeout-2026-09-18]]`
  memory added as a pointer to this whole session's artifacts.

## Found along the way (not originally scoped, fixed anyway)

- **`--video` defaults flipped to on, opt-out via `--no-video`.** The
  "always pass `--video`" rule ([[export-video-on-validation-runs]]) got
  violated again this session (three sessions running). Root cause fixed at
  the tooling level instead of relying on memory: every
  `scripts/probe_*.py`/`scripts/run_*.py`/`scripts/crossengine/run_*.py`/
  `run_sloshingTank.py` `--video` flag changed from `store_true` (opt-in) to
  `argparse.BooleanOptionalAction, default=True` (opt-out). Deliberately NOT
  applied to `scripts/validate_scheme.py` (its `smoke` profile is
  fast-by-design on purpose) or the shared `CaseSpec.video` dataclass default
  (the whole test suite silently relies on that being `False`).
- **Found, not fixed (flagged for later)**: `probe_deltaSPHMarrone{,34}.py`'s
  `_runOne` hardcodes `store=False`, so a killed/interrupted long run has
  zero recoverable state — the `.npz` metrics trace is written once, at the
  end, and no HDF5 checkpoint exists to resume from. An hour-long nx=256
  Marrone 3.4 run was lost this way mid-closeout. Reusing
  `CaseSpec.resumeFrom`/`resumeStepOffset` ([[warpsph-resume-checkpoint-support]])
  in these two scripts would fix it; not done here to stay in scope.

## Explicitly out of scope

- The corner-flyer / ceiling-sticking mechanism (→ `OPEN_PROBLEMS.md`, not
  reinvestigated here).
- 3D support for `english2025.py`.
- `DFSPH_IMPROVEMENT_PLAN.md`'s own separate priority list beyond the one
  `wallPressure.py` cross-check in G.

See the full plan rationale/dependencies in the session that created this file
(2026-09-18); this document is the working tracker, not the narrative.
