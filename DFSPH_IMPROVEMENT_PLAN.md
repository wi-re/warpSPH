# warpSPH — Incompressible (VD+PS / DFSPH) Improvement Plan

Working document for the incompressible SPH path (`schemes/dfsph.py`,
registered as `IncompressibleSPHScheme.divergenceFree`), the `dfsphReference`
troubleshooting scheme, the `iisph` baseline (Part 33), the `omniIncompressible`
omniSPH-loop port (Part 35), and `band2018pb` (Part 45).

**Since commit `c637785` (the "tmp commit"), `dfsph_step` no longer runs the
VD+PS `solveDivergenceFree` + `finalize` position shift. It runs
`omniIncompressible._solve` for both the divergence-free and constant-density
passes** — i.e. `divergenceFree` and `omniIncompressible` are now nearly the
same code, and much of the pre-Part-46 prose below (VD+PS, `solveDivergenceFree`,
the `finalize` resample) describes a path that is no longer live. See
**"Status — Parts 46–47"** immediately below; `DFSPH_FINDINGS.md` §1.16–§1.17
+ §9 rows 46–49 have the detail. Part 48 fixed the `kolmogorovIncompressible`
forcing regression and added regression tests; `randomFlowIncompressible
--bounded` is **confirmed unfixable without the near-wall CD-solve work** and
stays open (xfailed in the suite). Part 49 re-wired the `mdbcNoPenetrationShift`
the rewrite had dropped (FINDINGS §1.6) — wall crossing on `dambreak` /
`columnCollapse` impacts back to ~0.

**This file is the current state and the actionable list only.** The durable
findings — the physics lessons, the negative results, the literature notes, the
config surface, the bug list, the tooling index, and the one-line session
index — are in **`DFSPH_FINDINGS.md`**. The full part-by-part investigation
narrative (Parts 1–38) is in `git log -p DFSPH_IMPROVEMENT_PLAN.md`.

**Units note.** `cflFactor` on the incompressible cases multiplies the particle
**diameter** `dx`. Numbers in git history before commit `9a9bfe7` are in the
old `h`-based units and mean 4x more travel per step for the same number
(`n_h = 4`). `cflFactor = 0.4` today == the published Bender & Koschier
constant.

---

## Current state — post-`c637785`, Parts 46–47

`c637785` swapped `dfsph_step`'s body to `omniIncompressible._solve` for both
passes and gutted `IncompressibleSystem.finalize`'s VD+PS shift. That bought
`hydrostaticColumn` and cost the periodic KE tests; Parts 46–47 characterised
and closed the gap.

**What `divergenceFree` is now:** `_solve` divergence projection + `_solve`
constant-density pass, then — gated on `schemeConfig.gravityConfig.active` —
either fold the CD result into `dvdt` (`INSTEP_CD='auto'`, body-force cases)
**or** apply a VD+PS position shift in `finalize` (`_RESTORE_PS_SHIFT='auto'`,
everything else). Never both. Plus opt-in knobs: `SURFACE_SOURCE`,
`DIVERGENCE_SOLVER` (`'omni'`/`'vdps'`), `XSPH_SCALE` (per-case via
`schemeConfig.xsphFilterScale`), `SOLVE_ORDER`, `GRAVITY_OSC` (ablation).

**Suite: `tests/test_physics.py` + `tests/test_runner.py` = 91 passed / 0
failed / 1 xfail** — green since Part 47; Part 48 added `kolmogorovIncompressible`
forcing + `randomFlowIncompressible` periodic regression tests, and one
`xfail(strict)` marking `randomFlowIncompressible --bounded` as a known
divergence that flips to a suite failure the moment it starts holding. Part 50
added the first two `band2018pb` tests. (Two pre-existing full-suite failures,
unrelated — see "Known-open".)

**Case status at the shipped defaults (`divergenceFree`, verified 09-03;
`randomFlow`/`kolmogorov` rows updated Part 48):**

| case | status |
|---|---|
| `tgv`, `shearWave`, `staticBlob` (periodic / free space) | **pass** — `tgv` KE ×0.9996 monotone (Part 47 gate); `staticBlob` `\|v\| ≡ 0`. |
| `randomFlowIncompressible` **periodic** | **holds** — KE decays cleanly (0.45 → 0.38 / 200 steps, `\|v\|` ~1). |
| `randomFlowIncompressible --bounded` | **FIXED (Part 54) — root cause was NOT the CD operator this row blamed.** `nx=96` (the suite's fixture resolution) went from XFAIL to reproducible XPASS the moment `sample/regular.py`'s mass bug (Part 54) was fixed, with **no other change** — this case never calls `calibrateRestDensity`. Re-read in that light, the "density excursion builds near the wall over many steps" this row describes was seeded by the sampler assigning particle mass inconsistent with the cell it actually occupied (a uniform, small IC defect that reads as noise everywhere and is worst at a wall, where support is already asymmetric); the near-singular pure-Neumann near-wall CD operator did not manufacture the excursion, it failed to absorb one that was already there. The exhaustive stopgap sweep below (`maxDt`, `cflFactor`, `DIVERGENCE_SOLVER`, `boundaryPressureMode`, `nu`) is preserved as a record of what does NOT fix a *seeded* density excursion — it says nothing about whether the CD operator has its own, independent near-wall weakness, which is now untested since the confound is gone. Suite promoted (`tests/test_physics.py::test_randomFlowBoundedDoesNotDiverge`, xfail removed). **Not yet re-verified at the other resolutions this row lists** (nx=32/48/56/64/128) — only nx=96 has been re-run since the fix. Original characterisation, preserved: diverges at every resolution (Part 48 characterised it; no stopgap works). A density excursion builds near the wall over many steps (ρ spreads [0.98, 1.02] → [0.2, 3.4] while KE / `\|v\|` stay flat) then the pressure response detonates `\|v\|`. Onset time scales with `nx`: nx=32 t≈0.025, nx=48 t≈0.16, nx=56 t≈0.15, **nx=64 t≈0.95** (longest-lived — the one a `maxDt = 1e-3` cap "held" for 400 steps, a false positive), nx=96 t≈0.9, **nx=128 (shipped default) step 4**. Isolated (Part 48) to the **constant-density `solveIncompressible` inside the VD+PS shift** (`systems/incompressible.finalize`) — B: `_RESTORE_PS_SHIFT=False` delays it (nx=64 → step 90) but cannot hold it and can't be global (tgv needs the shift, §1.17); the shift's velocity-resample term detonates fastest, its position-move term alone rots ρ to [0.14, 4.97]. Nothing fixes it: `maxDt` cap / `--cflFactor 0.2` / `DIVERGENCE_SOLVER='vdps'` (worse — step-1 blow) / `boundaryPressureMode` ∈ {density, mlsPressure, plain} (identical) / `nu = 0.01` (worse). Same near-singular pure-Neumann near-wall CD operator as Parts 42–45; `iisph` / `omniIncompressible` fail this same case. Pre-`c637785` `divergenceFree` (VD+PS, convergent `solveDivergenceFree`) held it at band 4.48e-3. Periodic variant holds. |
| `kolmogorovIncompressible` | **FIXED (Part 48).** `dfsph_step` had `forcing = computeForcing(...) * 0` (a `c637785` mid-experiment leftover — `f401e4a` had it un-multiplied); dropped it. `computeForcing` returns 0 for every case without a forcing function, so only this case moves: KE 0 → 2.06 / 40 steps at nx=32, density [0.99, 1.003]. In the suite. |
| `hydrostaticColumn` (quiescent column under gravity) | **`divergenceFree` HOLDS it** (`c637785`'s point; confirmed nx=64 & nx=128 this session): `pressureSlopeRatio` 1.001, `densityP05` 0.94, `\|v\|` ~0.07–0.10. Residual is a bounded undamped free-surface limit cycle in the top ~3 rows (§1.16); the per-case `xsphScale` knob (`XSPH_SCALE=1.0`) decays it (KE ↓55×/2250× at nx=64/128) at a dissipation cost, so it is opt-in. The old note "position shift cannot sustain a body force" applied to the pre-`c637785` VD+PS path and is superseded for the shipped scheme. `iisph` also holds it (Part 33/34); `dfsphReference` does not (Part 37); `omniIncompressible` holds nx=128 (Part 41); `band2018pb` holds nx=32/64/128 (Part 50 — `embeddedMinDensity` 0.9998 and `pressureSlopeRatio` 1.015 at nx=128). |
| `impact` (collision) | **pass** (Part 23; `\|v\|` ~1.5, no NaN). |
| `squarePatch` (`rotatingSquarePatch`) | **runs** — stable rotation; corner density → 0.50. [BK] §5 documents this as a method limitation, not an implementation bug. |
| `dambreak --scheme divergenceFree` | **runs** (Part 19) — the only working free surface — but half `deltaSPH`'s run-out speed and most of the flow's KE dissipated on impact; needs its own `--cflFactor 0.2` (Part 20), not 0.4. Wall-crossing at the surge impact went 5 → **0** once the no-pen shift was re-wired (Part 49). |
| `columnCollapse` (released water column in a closed tank) | **runs** (Part 49, new case) — gravity-gated in-step CD path, `cflFactor 0.2`. Collapse impact used to push ~6 fluid particles a full spacing past the wall; with the restored no-pen shift `nPenetrating` stays 0–1. `test_columnCollapseWallHoldsOnImpact` (nx=32, 260 steps) locks it. |

### Immediate — `c637785` / Parts 46–47 fallout (A2/A3/A5 closed, A1 open, Parts 48–49)

- **A1. `randomFlowIncompressible --bounded` diverges — INVESTIGATED (Part 48),
  NOT fixable as a stopgap; folded into the Active track.** Full
  characterisation in the case-status row above. Short version: it is the
  constant-density `solveIncompressible` in the VD+PS shift losing density
  control at the wall (the Parts 42–45 near-singular pure-Neumann near-wall
  operator), it diverges at **every** resolution (nx=64 merely lasts longest,
  which made the initial `maxDt = 1e-3` cap look like a fix — it holds nx=64
  and nothing else), and no lever tried (`maxDt`, `cflFactor`,
  `DIVERGENCE_SOLVER`, `boundaryPressureMode`, `nu`, disabling the shift)
  holds it. The `boundedMaxDt` param was reverted. The case is now `xfail`ed
  in `tests/test_physics.py` (nx=96, `strict` → flips to a suite failure when
  it starts holding). **Real fix = the near-wall CD solve, Active track item
  1 / `band2018pb`.**
- **A2. `kolmogorovIncompressible` inert — DONE (Part 48).** The `*0` on
  `dfsph_step`'s `computeForcing` was a `c637785` mid-experiment leftover
  (`f401e4a` had it un-multiplied); dropped. `computeForcing` returns 0 for
  every incompressible-path case without a forcing function (only
  `kolmogorovIncompressible` has one), so the change is bit-identical
  everywhere else; `hydrostaticColumn` takes gravity through `computeGravity`.
  Verified: KE 0 → 2.06 / 40 steps, density [0.99, 1.003].
- **A3. Regression tests — DONE (Part 48).** In `tests/test_physics.py` (not
  `test_runner.py` — that file is runner-infra only): `kolmogorovIncompressible`
  forcing drives the flow + stays incompressible; `randomFlowIncompressible`
  **periodic** does not diverge + stays incompressible; `randomFlowIncompressible
  --bounded` `xfail(strict)`. Suite 86 → 89 passed / 1 xfail.
- **A4 (perf, lower) — still open.** `_RESTORE_PS_SHIFT='auto'` adds a
  `solveIncompressible` + adjacency rebuild per step on every non-gravity case
  — measurable at `tgv`'s default nx=256. Cache the adjacency / gate the shift
  to every N steps if it bites.
- **A5. `mdbcNoPenetrationShift` dropped by the rewrite — DONE (Part 49).**
  `c637785` commented `computeMdbcNoPenShift`'s call out of `dfsph_step`, so
  post-rewrite `divergenceFree` had only the pressure projection for
  no-penetration and a violent wall impact crossed the band. Re-wired at the
  pre-rewrite spot (fold `nopenshift/dt` into `dvdt` before the pressure
  solves, as `schemes/deltaSPH.py`), gated by `dfsph.NOPEN_SHIFT` (`'config'`
  default → obey `schemeConfig.solverConfig.mdbcNoPenetrationShift`, itself
  `True`). A/B (`scripts/probe_dfsphWallPenetration.py`, nx=64): `dambreak`
  wall crossing 5 → 0, `columnCollapse` 6 → 1, impact `|v|`max −30 % on both,
  `hydrostaticColumn` quiescent untouched, `randomFlowIncompressible
  --bounded` **unchanged** (still A1 — that is the near-wall CD operator, a
  different failure). `dambreak.diagnostics` gained an `nPenetrating` /
  `maxPenetrationDx` watch; `test_columnCollapseWallHoldsOnImpact` added.
  `c637785`-era unconditional `print`s in `dfsph_step` / `omniIncompressible.
  _solve` gated behind verbose (`omniIncompressible.VERBOSE_SOLVE`). FINDINGS
  §1.6 / §2 / §9 row 49.

---

## Active track — the near-wall CD solve on `randomFlowIncompressible --bounded`

**Note (Parts 46–47):** `divergenceFree` now *is* the `omniIncompressible._solve`
two-solve loop, so "grade `omniIncompressible` vs `divergenceFree`" below is
largely one scheme against itself + the shift gate. The live question this
track still owns: the **near-wall constant-density solve** — the same thing
that NaNs `randomFlowIncompressible --bounded` at graded `dt` (Immediate A1)
and that `band2018pb` / `CD_TIKHONOV` target.

**Status (Part 50): `band2018pb` now holds `hydrostaticColumn` at nx=32/64/128
*and* `randomFlowIncompressible --bounded` at nx=64/128 — the case that NaNs
`divergenceFree` and detonates `iisph` / `omniIncompressible`. The two
remaining blockers were both bugs in the port, not physics: `bandRelaxation`
applied the `n_h = 4` relaxation detune to the fluid rows only (leaving
boundary rows a ~100× larger Jacobi step), and a fully enclosed domain had no
gauge for the constant null mode of the summation-gradient operator. Defaults
now `OMEGA_FLUID = 0.1`, `DIAG_TIKHONOV = 0.05`, `CLOSED_DOMAIN_GAUGE =
'auto'`. See "Next, in order" item 1.** Part 45 landed the scaffold (operator
verified, wall rank-deficiency removed). Part 44 ruled out the "symmetrise `A`
+ MINRES" interim probe — `A` is *already* symmetric; the blocker is rank
deficiency, and only full `band2018pb` addresses it. Part 42 fixed the
divergence
(`WALL_PRESSURE_MODE = 'shepard'`) and the closed-box compatibility
(`CD_SOURCE_PROJECT = 'auto'`). Part 43 characterised the
remaining free-surface CD stall: the constant-density operator
`A p = -dt²·div(a_p(p))` is **near-singular**, not merely slow —
`hydrostaticColumn` nx=128's CD Jacobi hits its 256-iteration cap every step
(omniSPH's floored `mean(max(·,-1e-3))` metric reads "converged"; the true
`|r|₂/|s| ≈ 0.94`; the per-iterate `p ≥ 0` clamp is what makes the run
hold). `randomFlowIncompressible --bounded` nx=64 is the same — capped every
step even *with* the compatibility projection.
- **Landed (opt-in, default-inert): `omniIncompressible.CD_TIKHONOV`** — a
  uniform absolute diagonal shift `tik·median(|alpha_fluid|)` on the
  density-mode operator, applied only where `CD_SOURCE_PROJECT` did not fire.
  On the Jacobi path, `tik = 0.1` takes `hydrostaticColumn` nx=128 off the
  cap (mean 210 → ~75 iters, holds 400 steps, `embeddedMinDensity`
  0.984 → 0.991, `slope` → 1.005, `maxRho` 1.012 → 1.027). Strict wash on
  `dambreak` (its CD solve already converges in the 3-iter minimum). `tik = 0`
  (the default) is bit-identical to pre-Part-43 on every case.
- **Negative: `CD_SOLVER ∈ {bicgstab, gmres}`** — a non-symmetric Krylov
  solve of the same `A p = s` **breaks down** on every wall-bounded case
  (free surface *and* closed box), returning tiny-residual / `|p| ~ 1e9`
  iterates along the near-null space. Uniform Tikhonov 0.1 is not enough; a
  reject guard stops the detonation but the run then loses density control.
  So the near-wall block needs `band2018pb` / an explicit symmetrisation of
  `A` (`computePressureShiftIISPH`), not a diagonal patch — ranked-queue
  item 0. The `CD_SOLVER` plumbing + guard are landed as the scaffold for
  that; `'jacobi'` stays the default and only usable setting.
- **Part 44 — the cheaper interim probe ("symmetrise `A` via
  `computePressureShiftIISPH`, then MINRES") is a dead end, for a reason that
  redirects item 1.** `scripts/probe_omniIncompressibleCDSymmetry.py` captures
  the healthy density-mode system *inside* `applyConsistentCoupling` and, on
  that one system, measures the symmetry defect of three operator forms and
  runs relaxed Jacobi / MINRES / CG / BiCGStab on each. Findings
  (`hydrostaticColumn` nx=64 s30, `randomFlowIncompressible --bounded` nx=64
  s5, `dambreak` nx=64 s30):
  - **`A` is already symmetric.** `A_plain` (boundary `p ≡ 0`, BWJ23 Eq. 33)
    and `A_krylov` (`krylov.buildIISPHMatvec`, `staticBoundary`, `dt²`) have a
    relative symmetry defect `|⟨Ax,y⟩−⟨x,Ay⟩| / (‖Ax‖‖y‖)` of 2.7e-5
    (bounded box) to 8e-3 (dambreak) — i.e. fp32 + kernel-asymmetry noise. The
    per-iterate `wallPressureExtrapolation` Robin closure (`A_wall`) adds only
    ~4e-3 more defect and a ~10 % operator perturbation. **Symmetrising buys
    nothing that is not already there.**
  - **MINRES / CG / BiCGStab all diverge on the un-regularised system,
    symmetric form included** — `|x|` → 1e4–1e7, status −14 / −16 / −10 — on
    every wall-bounded case. The operator is **rank-deficient** (near-null
    space from kernel deficiency at the free surface *and* the wall corners;
    `median|alpha_fluid|` as low as 2.5e-5 on `dambreak`), so a symmetric
    Krylov has nothing to converge to. Only a uniform Tikhonov shift bounds
    `|x|`: on the closed box `tik = 0.1` gets every method to `|r|/|s| ≈ 0.1`
    in 20–70 iters; on the free-surface column it bounds `|x|` (~23, physical)
    but still `|r|/|s| ≈ 2.8` — the shifted (slightly-compressible) problem,
    not the PPE.
  - **`dambreak`'s CD "converges in the 3-iter minimum" is the floored
    omniSPH metric, not a solved system** — the captured `A p = s` there has
    `|r|/|s| = 1.000` after 2000 relaxed-Jacobi iterations (`Ax ≈ 0`; tiny
    diagonal). Same mechanism as `hydrostaticColumn` (Part 43), just even more
    kernel-deficient.
  So `computePressureShiftIISPH`-symmetrisation is struck from item 1: the
  operator's symmetry was never the blocker, its **near-singularity at the
  wall** is, and the only listed fix that removes that is the *full*
  `band2018pb` — boundary samples as PPE unknowns give the near-wall rows
  their own non-trivial equation + diagonal, so `A` is no longer
  rank-deficient there. (The free-surface near-null space is separate and
  stays handled by the `p ≥ 0` clamp / `CD_TIKHONOV`.) Item 1 below is
  rewritten around that.
See **"Next, in order"** at the end of this section.

**Why.** `omniIncompressible` + the Part 41 MLS wall pressure holds the
*quiescent* `hydrostaticColumn` at nx=128, but **diverges on the wall-bounded
*sheared* flow** `randomFlowIncompressible --bounded` (Part 42): it detonates on
step 1 (KE 0.35 → ~1.4e3, `|v|max` → 83 from an enormous near-wall pressure
impulse — density stays fine at [1.001, 1.001], so a solve overshoots), briefly
recovers over steps 2–7, then re-detonates by step ~10 → `|v|max` 1e15 by
step 14. `iisph` also fails this case (KE → 5e9); `dfsphReference` untested
here; the *pre-`c637785`* VD+PS `divergenceFree` held it (KE ratio 0.896, `|v|max` 1.06) — **the current `_solve`-based `divergenceFree` NaNs it at graded `dt`** (Immediate A1), so that reference point is gone. So the
DFSPH-family composed operators + the Akinci-band / MLS wall closure are not
robust to a real wall-bounded shear flow — this is the gap to close before
`omniIncompressible` (or any `iisph`+divergence-pass scheme, ranked queue
item 4/9) can be a general incompressible solver.

**Diagnosis so far** (`scratchpad` probe adapting `transient.py`'s solve/step
spies; nx=64):
1. **It is the constant-density Jacobi, not the divergence solve.** Step 1:
   the 3-iter divergence solve *converges* (`errDiv` ~1e-10, `max|a_p|` ~6);
   the constant-density solve **hits its 256-iter cap without converging**
   (`errRho` above tol, `max|p|` ~4e4, `max|a_p|` ~3e5) → `|v|max` ~27–84 on
   step 1. This is §1.7 ("the CD solve does not converge — it integrates") at
   a fully-walled box: with `div v* ≈ 0` (the divergence solve just made it
   so) and `rho ≈ rho0`, the source is near the operator's null space, so
   the Jacobi increment is ~constant and `p` grows ~linearly in the iterate
   count until the cap.
2. **`WALL_PRESSURE_MODE = 'mls'` (the Part 41 default) makes step 1 much
   worse here** — `errRho` 3.4e-2 vs 9.6e-4 without it, `|v|max` 84 vs 27.
   The MLS `p_b = alpha + beta*x + gamma*y` assumes a locally-linear
   near-wall pressure (exact for the hydrostatic column, Part 41); a sheared
   flow has real near-wall pressure structure, so the linear term amplifies
   and pumps energy into the Jacobi. `'mls'` is **regime-dependent**: it
   rescues the quiescent column and breaks the sheared box.
3. **`WALL_PRESSURE_MODE = None` HOLDS `randomFlowIncompressible --bounded`**
   (120 steps confirmed: rough step 1 at `|v|max` 27, then decays — `|v|max`
   ~1.0 by step 45, ~0.68 by step 120, density 0.97–1.02). The un-converged
   step-1 impulse happens to decay rather than compound. `'shepard'` (0th
   order, no linear term): promising on step 1 (`|v|max` ~2 vs mls's 84),
   longer-run TBD.

**Resolved for now (Part 42): `WALL_PRESSURE_MODE = 'shepard'` threads both.**
The split was `'mls'` needed for the quiescent column (Part 41, `None`
diverges there) vs `None` needed for the sheared bounded box (`'mls'`
diverges here). `'shepard'` — the zero-order mirror, omniSPH's MLS `alpha`
term with **no linear `beta*x + gamma*y`** — is stable in both regimes
(there is no linear term to amplify the sheared flow's real near-wall
pressure structure): `hydrostaticColumn` nx=128 holds (`|v|max` ~0.5, KE
~1e-3, the exact hydrostatic gradient, 350 steps); `randomFlowIncompressible
--bounded` holds (`|v|max` decays 2 → 0.4, density 0.99–1.01, 300 steps).
`omniIncompressible.WALL_PRESSURE_MODE` default changed `'mls'` → `'shepard'`;
`'mls'` kept as an option (Part 41 measured it recovers a slightly better
near-wall density on the quiescent free-surface column). The `dfsphReference`
/ `iisph` flag stays `None`. **Regression clean:** `tgv` / `kolmogorov`
bit-identical `'shepard'` vs `'mls'` (no `kind == 1`, wall pressure a no-op);
`dambreak` nx=64 identical and preserved (200 steps, `maxRho` 1.000).
`gradcheck_incompressible` + physics green.

**Chased the CD solve (Part 42): the closed-box divergence is an incompatible
source, fixed by a compatibility projection.** Captured `omniIncompressible`'s
constant-density linear system `A p = s` on `randomFlowIncompressible
--bounded` step 1 (`scratchpad` probe: monkeypatch `_solve`, dump the operator
+ RHS, run offline solvers). Findings:
- **The Jacobi stalls, it does not diverge.** `|r|_2`: 8.06e-2 → 8.10e-2 →
  8.18e-2 over 256 iters — flat, from `|s|_2 = 7.72e-2`. The residual *cannot*
  drop below `|s|`.
- **The source is 99.98 % its own mean.** `mean(s_fluid) = -1.2e-3`,
  `|s - mean(s)|_2 = 1.16e-5` (vs `|s|_2 = 7.7e-2`). `randomFlowIncompressible
  --bounded` is a **fully closed box, no free surface** → the pressure
  operator is pure-Neumann → `A·1 ≈ 0` → the constant part of `s` (which the
  `n_h = 4` lattice density bias, §1.1, makes non-zero) is in `null(A)` and
  has **no solution**. The Jacobi's residual floor is exactly that
  incompatible component; `p` ramps linearly (§1.7 — "it integrates").
  MINRES / CG break down immediately on the inconsistent system.
- **Compatibility projection fixes it.** Subtract `mean(s_fluid)`; the Jacobi
  on the residual (with `p` kept mean-zero, no `p ≥ 0` clamp — a closed box
  has no tensile-instability free surface) converges: `|r|_2` 1.05e-5 →
  6e-6 → 1.2e-6 over 256 iters (bulk of it in < 8). The residual
  `mean(ρ) ≠ ρ0` is a rest-density calibration offset the solve legitimately
  ignores.

**Landed: `omniIncompressible.CD_SOURCE_PROJECT = 'auto'`** — projects the
density source's mean only when `1 - |s - mean(s)| / |s| > CD_PROJECT_THRESHOLD`
(= 0.7), i.e. when the source is mean-dominated. A free surface makes the
spatial part of `s` large (`hydrostaticColumn` step 1: `frac_uniform ≈ 0.09`,
step 30 `≈ 0.006`), so it is a **strict no-op there** and on `dambreak` /
`tgv` / `kolmogorov`. Full matrix (`auto`, nx=64/128, 120–150 steps):
`randomFlowIncompressible --bounded` holds, KE 0.34 → 0.10 (decays, density
0.996–1.007 — the CD Jacobi now *converges*, not just survives);
`hydrostaticColumn` nx=128 holds, `slope` 0.995, `|v|max` 0.52;
`dambreak` / `tgv` / `kolmogorov` unchanged. `gradcheck_incompressible` +
physics green.

**Part 43 — the free-surface CD solve: characterised, Jacobi made affordable,
Krylov ruled out.** `scripts/probe_omniIncompressibleCDSolver.py` +
`scripts/probe_omniIncompressibleCDSystem.py` (offline capture of the healthy density-mode
`A p = s` at a chosen step).

- **The operator is near-singular, not slow.** On `hydrostaticColumn` nx=128
  the CD Jacobi hits its 256-iteration cap every step; omniSPH's convergence
  metric `mean_i(max(residual_i, -1e-3))` floors the (large, negative)
  residual and reads "converged", but the true `|r|₂/|s| ≈ 0.94`. The
  quiescent column has `div v* ≈ 0` and `ρ ≈ ρ0`, so the source sits near the
  operator's near-null space — the §1.7 mechanism, only *partly* removed by
  the free surface (`frac_uniform ≈ 0.99` at nx=128 step 30, so
  `CD_SOURCE_PROJECT` correctly does not fire). The per-iterate `p ≥ 0` clamp
  is what regularises the *run*: drop it (as a linear Krylov solve needs) and
  the minimum-residual `p` blows up to `|p| ~ 1e9` (vs the clamped Jacobi's
  physical `~23`). `randomFlowIncompressible --bounded` nx=64 is the same —
  capped every step *with* the compatibility projection.
- **Landed: `omniIncompressible.CD_TIKHONOV` (default `0.0`, opt-in).** A
  uniform absolute diagonal shift `tik·median(|alpha_fluid|)` — solving a
  nearby slightly-compressible problem `(A − eps|D|) p = s`. Uniform, *not*
  per-row `∝ alpha`: the kernel-deficient near-surface rows have tiny
  `|alpha|` and that is where the near-null space lives. Applied only for the
  density solve when `CD_SOURCE_PROJECT` did not fire (closed / periodic need
  no shift). On the Jacobi path, `tik = 0.1`: `hydrostaticColumn` nx=128
  density solve **mean 210 → ~75 iters** (off the cap), holds 400 steps,
  `embeddedMinDensity` 0.984 → 0.991, `slope` 0.999 → 1.005, `maxRho`
  1.012 → 1.027, KE band unchanged ~2e-3. `dambreak` nx=64: **strict wash**
  (that CD solve already converges in the 3-iter minimum — dynamic surface,
  well-conditioned source). `tik = 0` is bit-identical to pre-Part-43 on
  every case (`gradcheck_incompressible` + tgv/shearWave/dambreak physics
  green).
- **Negative: non-symmetric Krylov (`CD_SOLVER ∈ {bicgstab, gmres}`).** A
  BiCGStab / GMRES solve of the same `A p = s` (matvec built from the
  existing `accel` / `_divergence` closures; Jacobi-diagonal preconditioner;
  wall-pressure closure run `clampNonNeg=False` for linearity; zero warm
  start) **breaks down on every wall-bounded case** — free surface *and*
  compatibility-projected closed box. BiCGStab bails at iteration ~1–3 with a
  tiny residual and `|p| ~ 1e9` along the near-null space; GMRES the same.
  Uniform Tikhonov 0.1 does not rescue it. A reject guard (`|r| ≥ |s|`, or
  `|x|₂ > 1e3·|s|`, or `|x|max > 1e5`, or non-finite → the step takes no
  density correction) stops the 1e9 detonation, but the run then loses
  density control (`maxRho ~ 1.4`, `slope 0`). So the composed operator
  `-dt²·div(a_p_IISPH(p_with_wall))` is too non-symmetric / ill-conditioned
  at the wall for a matrix-free Krylov method — MINRES already breaks down
  here (`status -13`), and BiCGStab/GMRES do too. The `CD_SOLVER` flag +
  guard are landed as scaffolding; `'jacobi'` is the default and only usable
  setting.

**The real fix (ranked-queue item 0):** make the near-wall block *consistent*
— `band2018pb` (boundary samples as solve unknowns). Part 44 struck the
"symmetrise `A` via `computePressureShiftIISPH`, then MINRES" alternative:
`A` is *already* symmetric to fp32, and MINRES/CG diverge on it anyway
because it is **rank-deficient** at the wall, not asymmetric — only the
extra boundary equations of `band2018pb` remove that. A diagonal patch
(`CD_TIKHONOV`) buys the Jacobi its iteration budget back but does not make
the operator Krylov-solvable.

Not a blocker: `hydrostaticColumn` / `dambreak` *hold* — this is a §1.7
quality issue (open since Part 15), not a failure.

**Next, in order.**

1. **Full `band2018pb` for the free-surface / wall CD solve — the live
   thread, and now the *only* listed path.** Parts 43–44 ruled out every
   cheaper option: the diagonal patch (`CD_TIKHONOV`) only restores the
   *Jacobi's* iteration budget; non-symmetric Krylov (BiCGStab / GMRES)
   breaks down; and (Part 44) `A` is *already* symmetric to fp32, so
   symmetrising it via `computePressureShiftIISPH` + MINRES changes nothing —
   MINRES / CG / BiCGStab all diverge anyway because the operator is
   **rank-deficient at the wall corners** (near-null space; `median|alpha|`
   → 2.5e-5). **`band2018pb`** (Band, Gissler, Ihmsen, Cornelis, Peer,
   Teschner 2018, ACM TOG 37(2):14, `literature/band2018pb_…pdf`) removes the
   rank deficiency at its source: boundary samples enter the PPE as their own
   unknowns, so the near-wall rows get a non-trivial equation + diagonal
   instead of a near-zero one. Concretely, from the paper's §2.3
   (volume-centric, cubic-spline `2h`; the derivation ports onto the DFSPH/
   IISPH step this codebase already runs):
   - **Unknown vector** `p = [p_f ; p_b]` over fluid *and* `kind == 1`
     boundary rows (ghosts excluded).
   - **Pressure accel, unified (Eq. 8):** `a^p_f = -(V_f/m_f) Σ_j V_j
     (p_f + p_j) ∇W_fj` over *all* neighbours `j` (fluid + boundary) —
     replaces `computePressureAccelIISPH`'s BWJ23-Eq.33 form (which drops the
     boundary `p`). Linear-momentum preserving; no mirroring assumption.
   - **Fluid rows (Eq. 9):** `dt² Σ_ff V_ff (a^p_f − a^p_ff)·∇W_fff
     + dt² Σ_fb V_fb a^p_f·∇W_ffb = 1 − V^0_f/V_f + dt ∇·v*_f`.
   - **Boundary rows (Eq. 10):** `-dt² Σ_bf V_bf a^p_b·∇W_bbf
     = 1 − V^0_b/V_b + dt ∇·v*_b`, with `a^p_b = 0` (one-way static wall) —
     so the LHS is `-dt² Σ_bf V_bf a^p_b·∇W_bbf` evaluated from the *fluid*
     `a^p`, i.e. the boundary row couples back to `p_f` through `a^p`.
   - **Volumes:** `V^0_f = h^d`; `V^0_b = γ / Σ_bb W_bbb` (γ = 0.7);
     `V_f = V^0_f / (Σ_ff V^0_ff W_fff + Σ_fb V^0_fb W_ffb)`;
     `V_b = V^0_b / (Σ_bf V^0_b W_bbf + γ + β)` (β = 0.15·h^d, one-sided
     planar wall).
   - **Divergences:** `∇·v*_f` = Eq. 16 (over fluid + boundary neighbours,
     `v*_b` = 0 static); `∇·v*_b` = Eq. 17 (fluid neighbours only).
   - **Diagonals** `a_ff` (Eq. 19) and `a_bb` (Eq. 20) — closed forms in the
     same volume sums; per-sample relaxation `ω_i = 0.5 V^0_i/h^d`
     (`ω_f = 0.5`, `ω_b = 0.5 γ/(h^d Σ W_bbb)`).
   - **Convergence:** mean over *all* fluid+boundary rows of the residual
     `(Ap)_i − s_i` (a relative volume error), not the fluid-only floored
     omniSPH metric.
   **Part 45 — scaffold landed as a fresh module + a distinct scheme;
   operator verified, wall block healthy, nx=32 holds, nx≥64 unstable.**
   - **`modules/incompressible/pressureBoundaries.py`** — the whole operator
     set (`bandRestVolumes` / `bandActualVolumes` / `bandBoundaryUnknownMask`
     / `bandVelocityDivergence` / `bandPressureAccel` / `bandApplyOperator`
     / `bandDiagonal` / `bandRelaxation` + a `bandInjectVolumes` context
     manager), composed **entirely from existing `warpOperation` primitives +
     `computeAlpha`** — no new `@wp.kernel`, so no gradcheck burden. Band
     volumes are injected by temporarily setting `state.densities :=
     state.masses / V` so `warpOperation`'s `m_j/ρ_j` weight becomes `V_j`
     (the `applyConsistentCoupling` trick).
   - **`schemes/band2018pb.py`** + **`IncompressibleSPHScheme.band2018pb = 4`**
     + `builder._band2018pb` — a distinct IISPH-style single-solve step over
     `p = [p_f ; p_b]`, reusing `DFSPHReferenceSystem`.
   - **Adaptations from the paper (one-layer cubic-spline `2h` → this
     codebase's Wendland2 / `n_h = 4` / five-layer Akinci band):** `γ` (Eq. 12)
     and `β` (Eq. 14) are **dropped** — they inflate `V_b` to ~1.4× `V0_b`
     here, injecting a spurious +0.3 boundary source; boundary rows use the
     nominal `V0 = m/ρ0` and the same full-neighbour `V_f = V0 / Σ_j V0_j W`
     as the fluid. Only `kind == 1` rows with real fluid contact
     (`Σ_bf V0_f W_bf > MIN_FLUID_CONTACT`) are unknowns — the 3–4 band
     layers behind the interface have a zero Eq. 20 diagonal and are
     excluded. For static walls Eq. 16 = Eq. 17 and Eqs. 9/10 collapse to one
     difference-divergence of the (fluid-masked) Eq. 8 accel.
   - **Verified (`scripts/probe_band2018pbSystem.py`):** the Eq. 8 accel on
     the analytic hydrostatic `p` gives bulk `a^p_y ≈ +9.805` (cancels
     gravity — correct scale + sign); `A` is symmetric to ~3e-3; **the wall
     interface rows now have healthy non-singular diagonals** (`dt²·a_bb`
     median 5.9e-4, *all* correct sign) — the rank deficiency this whole
     item exists to remove **is removed**.
   - **The nx≥64 divergence was the near-surface diagonal, fixed by a
     Tikhonov floor.** The relaxed Jacobi diverged at nx≥64 even at step 1
     (`|x| → ∞`). Cause (probed, not swept): the band diagonal is
     `computeAlpha`-**exact** (matches the numerically-probed true `A_ii` to
     round-off, fluid *and* boundary), but kernel-deficient near-surface rows
     have `|dt²·a_ii| ~ 1e-6`, so the Jacobi step `omega / a_ii` on those
     rows (which also carry `s > 0`) detonates. **Landed:
     `band2018pb.DIAG_TIKHONOV`** (default `0.3`, non-zero — unlike
     `omniIncompressible.CD_TIKHONOV`, `band2018pb` needs it) — solve the
     nearby `(A − eps·median|dt²a_ii|·I) p = s`, the same device as Part 43,
     applied to both the diagonal and the operator (`Ap − shift·p`).
   **Part 50 — the two blockers were both bugs in the port, not physics.
   nx=128 holds, and `randomFlowIncompressible --bounded` holds.**
   - **Bug 1 — `bandRelaxation` scaled only the fluid.** The paper's
     `ω_i = 0.5·V0_i/h^d` applies *one* base constant to every row; a boundary
     sample takes a smaller step only because its rest volume is smaller. The
     port detuned the fluid base to `OMEGA_FLUID` (0.05) for `n_h = 4` but left
     boundary rows at a hardcoded `0.5`. Since `bandRestVolumes` gives
     `V0_b = V0_f = m/ρ0` here, that was a flat **10× larger ω on rows whose
     Eq. 20 diagonal is itself ~10× *smaller*** than the fluid's Eq. 19 one
     (it lacks the first term) — a Jacobi step `ω/a_ii` ≈ **100×** the fluid's.
     Measured at nx=128: `p_b` ran to 96 against a peak fluid `p` of 11.8,
     Eq. 8 turned that into `|a_p| ~ 1.3e3` on the near-wall fluid, the column
     was kicked apart, and — because the source then went 96 % positive and
     every row clamped to `p = 0` — the floored metric read `err < 0` and the
     solve **exited at the 8-iteration minimum doing nothing**. Fixed:
     `ω_i = OMEGA_FLUID·V0_i/V0_f` for all rows.
   - **Bug 2 — no gauge for a closed domain.** Eq. 8 is a *summation* gradient,
     so a uniform `p` accelerates nothing wherever the kernel support is
     complete; in a fully enclosed domain the constant is a null mode of `A`,
     and the Eq. 18 `max(·,0)` clamp — which can only push a row *up* — ratchets
     it away instead of pinning it. This is **not** Part 42's incompatible
     source: `fracUniform ≈ 0.03` here (threshold 0.7), so
     `CD_SOURCE_PROJECT`'s test would never fire. The *source* is fine; the
     *solution* is unpinned. Landed **`band2018pb.CLOSED_DOMAIN_GAUGE`**
     (`'auto'`), keyed on a direct null-mode test
     `bandConstantModeRatio = rms|A·1| / rms|a_ii|` (nx=64 step 1: closed box
     **0.032**, free-surface column **1.29** — a 40× separation, threshold
     0.25); when it fires, each iterate is projected to zero mean over the
     solve rows and the non-negativity clamp is dropped (a closed domain has no
     free surface, so there is no tensile instability for it to suppress —
     the same pairing Part 42 reached).
   - **Retuned on the fixed operator:** `OMEGA_FLUID` 0.05 → **0.1** (0.2+ still
     detonates nx=128), `DIAG_TIKHONOV` 0.3 → **0.05** (0.05 and 0.1 both hold
     nx=32/64/128; 0.05 over-compresses least — bulk ρ 1.013 vs 1.023 at
     nx=128 — and the paper has no Tikhonov term at all). `tik = 0` still
     fails (`embMin` 0.44 at nx=64), so the floor is genuinely needed.
   - **What holds now** (`hydrostaticColumn`, 200 steps): **nx=64** bulk ρ
     1.0035, `embMin` 0.995, `slope` 1.004, `|v|max` 0.12, 0 spray; **nx=128**
     bulk ρ 1.013, `embMin` 0.9998, `slope` 1.015, `|v|max` 0.16, **2** spray
     (was: ρ 0.934, `embMin` 0.14, `slope` 0.020, `|v|max` 2.7, **1695** spray).
     nx=32 unchanged. The nx=128 numbers are bit-identical with the gauge on
     and off — it correctly does not fire at a free surface.
   - **`randomFlowIncompressible --bounded` — the acceptance test — HOLDS
     under `band2018pb`.** nx=64: KE decays 0.446 → 0.213 monotonically over
     200 steps, `|v|max` ~1.0 (peak 1.37), ρ ∈ [0.991, 1.017]. nx=128: KE
     0.322 → 0.302 / 150 steps, `|v|max` 1.0–1.36, ρ ∈ [0.995, 1.018]. This is
     the case that NaNs `divergenceFree` by step ~21 and detonates `iisph` /
     `omniIncompressible`. **nx=96 is the exception and it is a case-IC
     artifact, not the scheme:** its step-0 state is already `KE 9.23` /
     `|v|max 10.2` against ~0.3 / ~1.0 at nx=64 and nx=128, i.e. the random-flow
     initial condition is anomalous at that resolution *before any solve runs*;
     from there the run stays bounded but shows recurring KE injection
     (0.36 → 35 → 26 → 38). Worth chasing in the case, separately.
   - **Suite:** two new regression tests
     (`test_band2018pbColumnStaysQuiescent`, nx=64/200 steps;
     `test_band2018pbBoundedRandomFlowDecays`, nx=64/120 steps) lock both
     fixes. `gradcheck_incompressible` green (no `@wp.kernel` touched — the
     module is still composed from existing primitives).
   - **Also fixed on the way in:** `literature/band2018pb_…pdf` was
     **corrupt on disk** — its header read `%PDF-1.7` followed by `EF BF BD`
     bytes, i.e. the file had been round-tripped through a text decode and
     every binary stream destroyed (`pdftotext`/`mutool` both extracted 0
     bytes; all 11 pages rendered blank). Re-fetched from the authors' copy at
     `cg.informatik.uni-freiburg.de/publications/2018_TOG_pressureBoundaries.pdf`.
     It is the only corrupt file in `literature/`, and it is gitignored, so
     nothing was committed. **The equation transcription in this plan was
     nonetheless correct** — re-checked line by line against the clean text.
   **Part 51 -- graded across 14 cases by the new
   `scripts/validate_scheme.py`: 13 pass, `staticBlob` is the one real
   failure.** PASS includes `hydrostaticColumn` nx=64 (slope 1.002) and nx=128
   (slope 1.016), `dambreak` (600 steps, `nPenetrating` 0), `columnCollapse`,
   `impact`, `squarePatch`, all three `randomFlow` variants, `tgv`,
   `kolmogorov`, `shearWave`, and **`sloshingTank` -- the full 8500 steps / 7 s
   of SPHERIC TC10, Sensor-1 peak 7.0 kPa, inside the measured 2.2-13 kPa
   band.** `tgv` passes at nx=64 but *fails* at nx=32, so Part 50's energy
   injection is a coarse-grid artifact. **`staticBlob` fails for real:** a
   free-space blob that should sit at `|v| = 0` reaches `|v|max` 1.02 with
   `dispMax` > 0.1 while `centroidDrift` passes -- local surface distortion,
   not a spurious net force. That is coherent with the paper, which is a
   *boundary-pressure* method with no free-surface treatment at all (every
   scenario in it is a wall-bounded tank): `band2018pb` fixes walls and
   inherits the free-surface problem, masked on `hydrostaticColumn` by the
   `p >= 0` clamp + `DIAG_TIKHONOV` and unmasked on a pure free-space body.
   Also new: `sloshingTank.sensorPressureWall` reads the solved pressure at the
   wall sample directly (the paper's Sec. 3.3 selling point); it peaks at
   25.8 kPa, ~3.7x the smoothed fluid probe, so it needs its own calibration
   before it is a validated figure. Detail + the harness's grading corrections:
   FINDINGS Sec. 9 row 51.
   **Next on this sub-thread:** the `staticBlob` free-surface defect is now the
   clearest open `band2018pb` item. Then point
   `scripts/probe_omniIncompressibleCDSymmetry.py`'s operator builders at the
   `band2018pb` `A` — with the wall rows consistent *and* the constant mode
   gauged, a symmetric Krylov (MINRES / CG) may now apply where Part 44 found
   only divergence. Then grade `dambreak` / `columnCollapse` (free surface +
   violent impact) under `band2018pb`, which is the regime it has not been
   tried in. The paper's own levers are still unused and are the cheap wins if
   iteration count starts to matter: **warm start `λ = 1`** (the paper measures
   λ=1 as best for Pressure Boundaries, vs 0.5 for IISPH — this port uses 0.5)
   and the **unfloored all-row volume-error metric** of Eq. 21 (the floored
   omniSPH metric is what let bug 1 exit at 8 iterations pretending to
   converge). Whatever lands here is also the contractive divergence-pass any
   `iisph`-based general scheme needs (item 4/9).
2. **`randomFlowIncompressible --bounded` under `divergenceFree`** — Immediate
   A1, **still open; still xfailed.** Part 50 showed `band2018pb` *does* hold
   this case (nx=64 and nx=128, clean decay), so the acceptance test for item 1
   is met — but `test_randomFlowBoundedDoesNotDiverge` runs the case's **default
   scheme**, which is `divergenceFree`, and that path is untouched. The xfail
   therefore correctly stays. Two ways to close it, and they are different
   pieces of work: (a) port the Part 50 findings into the VD+PS shift's
   constant-density `solveIncompressible` — the closed-domain gauge is the
   likelier of the two to transfer, since the shift solve has the same
   summation-gradient null mode; or (b) decide `band2018pb` is the wall-bounded
   scheme and re-point the case's default. Detail: the case-status row + §9
   rows 48 and 50.
   (The pre-Parts-46/47 "grade `omniIncompressible` vs `divergenceFree`"
   framing is moot — they are the same code; what remains is the shift gate +
   `XSPH_SCALE` dissipation trade, and the closed-box KE budget.)
3. **`omniIncompressible` full `dambreak` grade** (item 0 leftover) and
   **`dfsphReference` on `randomFlowIncompressible --bounded`** (untested —
   does the two-solve path diverge there like `iisph`/`omniIncompressible`
   did before Part 42?).
4. **The shear-carrying Morris viscosity term** (item 0b TODO, independent of
   the CD-solve work) — a real `DiffusionParameters`-wired laminar viscosity
   that carries tangential stress, so `hydrostaticColumn`'s `wallBC=noSlip` +
   `nu` gives a clean no-slip wall (Part 42 found the stock `viscidNu` term is
   normal-projected). New `ViscosityTerms` value + `deltaSPH` regression pass;
   fix `wp_viscosityDelta.py`'s docstring.
5. **Full-suite validation of Parts 34–48 — mostly done (Part 48).**
   `bash scripts/run_tests.sh` green bar 2 pre-existing failures (see
   "Known-open"); `gradcheck_incompressible` green; `run_sweep.py --nSteps 25`
   **33/33**; a deeper nx=48 / ~200-step pass of every `divergenceFree` case
   holds (gravity gate routes correctly). Left: `run_sweep.py --full`. Parts
   34–45's
   omni/band flags are default-inert, but Part 47's `INSTEP_CD` /
   `_RESTORE_PS_SHIFT` `'auto'` **change the `divergenceFree` default path**
   for every non-gravity case — this one needs the full sweep + `run_sweep.py`
   before it can be called landed.

---

## Earlier track — a velocity-coupled incompressible scheme for the column

**Why (pre-`c637785`).** The *old* VD+PS `divergenceFree` could not hold a
quiescent, wall-bounded, free-surface-under-gravity state (`hydrostaticColumn`,
Part 23): the density-invariance correction was a momentum-neutral position
shift (`DFSPH_FINDINGS.md` §1.2/§1.3), and a position shift cannot sustain a
body force. **This is resolved for the shipped scheme** — `c637785` moved the
CD correction to velocity (`_solve` in the step) and Part 47's `INSTEP_CD=
'auto'` gate keeps it there only where a body force needs it. The `iisph`
work below stands as an independent baseline / cross-check.

**Outcome (Part 33): `IncompressibleSPHScheme.iisph` — plain IISPH ([I],
Ihmsen et al. 2014) — holds it, and is landed as the [I] baseline.** One
constant-density projection per step, applied as a velocity impulse; no
divergence-free pass, no VD+PS shift. It is `dfsphReference`'s step body with
the divergence solve switched off (`schemes/dfsphReference.py::iisph_step`,
`skipDivergence=True`; reuses `DFSPHReferenceSystem` and the incompressible
codecs). Measured on `hydrostaticColumn` nx=32: stable geometry (surface
height flat over 2000 steps), stable bulk density, `pressureSlopeRatio` → ~1.0
(the correct hydrostatic gradient builds over a few hundred steps). It also
holds `staticBlob` nx=64/60 where the two-solve `dfsphReference` diverges —
not a regression, the two-solve baseline diverges there too. Full suite green,
`gradcheck_incompressible.py` green (no new kernel — the CD Jacobi already
carries `computeAlpha`'s IISPH `a_ii`).

**What the "late-time free-surface degradation" (Parts 26/30/31/32) actually
was — two compounding artifacts, both now removed:**

1. **The divergence-free Jacobi.** It was the catastrophic-instability source:
   the inf-velocity / uniform-`rho`-0.139 soup of Parts 29/30. Dropping it
   (`SKIP_DIVERGENCE_SOLVE` / the `iisph` scheme) gives **0 / 12** blow-ups
   across the four IISPH arms of a 1500-step × 3 batch, vs 1/3–3/3 with the DF
   solve in the loop. Part 29 got the linear DF Jacobi *contracting* on the
   quiescent column, but it never survived a woken-up free surface.
2. **The `minDensity` figure of merit.** The surface probe
   (`probe_dfsphReferenceColumnSurface.py`) shows the low readings are 1–3
   fluid particles thrown **1–3 dx above** the free surface by the bulk slosh
   — isolated (10–25 neighbours vs ~50), reading `rho` ~0.14–0.3 by kernel
   deficiency, then falling back. Never the same particles two samples running
   (`persist` = 0). Cosmetic ballistic spray, not structural loss; the
   embedded column min density stays ~0.76–0.92 for 2000 steps.

**What is left, and it is not a divergence:** the bulk carries a **bounded,
undamped free-slip slosh** (Part 32's counter-rotating vortex pair — KE
plateaus at ~0.12 and neither grows nor decays; the scheme family has no
tangential stress and the case runs `nu = 0`). It does not fail the run; it
keeps the surface spray alive. Decaying it needs a real viscosity / XSPH
choice, not a pressure-solve lever.

**Measured negatives from Part 33** (see `DFSPH_FINDINGS.md` §2):

- **Wall-XSPH `ε_b = 0.1`** (Part 32's n=1 lead) — **not confirmed at n=3.**
  Marginal onset delay (~50–100 steps), end-state inside the baseline spread,
  1/3 to the inf-soup. Part 32's single-run win was a lucky draw both sides.
- **Rest-density calibration** (`hydrostaticColumn` `calibrateRestDensity`
  param, default off — normalise fluid mass so the at-rest bulk reads `rho0`
  instead of the ~0.95 the `n_h = 4` lattice integrates to). It *does* stop
  the Part 31 IC-seed self-destruct, but with the DF solve in the loop it
  wakes the CD→DF coupling and detonates (3/3 immediate blow-up); paired with
  the damped warm start it degrades the late-time surface *earlier and
  deeper* than baseline. In `iisph` mode it speeds the gradient build but is
  not needed (plain IISPH gets there on its own).
- **The damped warm start under the single solve** — the Part 31 gated/capped
  seed *starves* the accumulating `kappa`, so the hydrostatic gradient never
  forms (`pressureSlopeRatio` stays ~0). Full-carry warm start (or none) is
  what lets IISPH build the profile.

**Next on this track — the validation ladder:**

1. **Done (Part 34).** `hydrostaticColumn --scheme iisph` at the default
   `nx = 128` **holds**: runs clean to `tLimit` (438 steps) and past it —
   1500 steps / t = 2.9, `diverged = False` — with `pressureSlopeRatio`
   late-run ~0.99 (the exact hydrostatic gradient) and the column body at
   `embeddedMinDensity` ~0.92–0.96 throughout. The plain `minDensity`
   still swings 0.2–0.6 — the ballistic spray, exactly as Part 33 diagnosed
   at nx=32. Two spray-robust FOMs landed in `hydrostaticDiagnostics`:
   `densityP05` (5th-percentile fluid density) and `embeddedMinDensity`
   (min over fluid rows > 1 dx below the 95th-percentile surface). Two
   caveats recorded: (a) a startup transient at t ≈ 0.14–0.22 where the raw
   hydrostatic IC seed relaxes to the nx=128 discretization —
   `embeddedMinDensity` dips to ~0.59, `pressureResidual` spikes to ~1.0 —
   deeper than at nx=32, recovered by t ≈ 0.25; (b) the bounded free-slip
   slosh is not flat-plateaued at nx=128 as it was at nx=32 — KE creeps
   ~0.025 → 0.050 over 1500 steps (still bounded, no blow-up).
   `probe_hydrostaticColumnIisph.py`.
2. `dambreak` A/B: `iisph` vs `divergenceFree` vs `deltaSPH` — run-out speed
   and the energy budget (also a second data point for queue item 1: IISPH
   has no Eq. 17 resample).
3. **Done for the periodic cases (Part 42): `iisph` fails `tgv` outright.**
   `tgv --scheme iisph` nx=64 *injects* energy — KE 9.86 → 1876 by t=0.1,
   `|v|max` 1.0 → 29, then a wrong bounded plateau (KE ~1200, `|v|max` ~25)
   for 300 steps; density stays near-perfect (`rho` [0.995, 1.007],
   `rhoStd` 1.3e-3) throughout. `divergenceFree` holds `tgv` (KE ratio ~1.0,
   monotone — pre-`c637785` via VD+PS, now via the Part 47 shift gate) and so
   does `dfsphReference` (KE ratio 0.81, bounded) — the
   *only* difference from `iisph` is the divergence-free pass. So plain IISPH
   (CD-only velocity-impulse) is **not a general incompressible scheme**: with
   the density-invariance constraint enforced but `div v = 0` not, the
   pressure impulses spin the vortices up while keeping `rho` flat (§1.2 from
   the other side — the correction is now on velocity and there is no
   divergence pass to keep it consistent). `iisph` is viable only in the
   near-quiescent regime (`hydrostaticColumn`, `staticBlob`). **`omniIncompressible`,
   which *has* a (3-iter) divergence pass, holds `tgv`** (KE ratio 0.79,
   bounded — over-dissipative from its `XSPH_FLUID = 0.05` default) and
   `kolmogorovIncompressible` — but **diverges on `randomFlowIncompressible
   --bounded`** (KE → ~1e31), so it is not general either; Part 41's MLS wall
   pressure holds the quiescent column but not a wall-bounded shear flow.
   **`randomFlowIncompressible --bounded` confirms it** — `iisph` `|v|max`
   → 1.3e6 by step 40 where `divergenceFree` holds at 1.06.
   `kolmogorovIncompressible` (forced, from rest) is milder — `iisph` `|v|max`
   3.78 vs `divergenceFree` 2.51 at matched KE, a rougher field not a blow-up.
4. Only then: whether a full two-solve DFSPH (`dfsphReference` hardened) is
   worth finishing, or `iisph` + a divergence pass added later is the path.
   **Part 41 shifted the odds toward the latter, and Part 42 makes the
   divergence pass non-optional:** `iisph` alone fails `tgv` (item 3), so any
   general scheme built on it *must* add the divergence solve back. The
   per-iterate wall pressure that fixed `omniIncompressible`'s constant-density
   Jacobi is `band2018pb`-lite, so a `band2018pb` boundary + `iisph`'s CD
   solve + `omniIncompressible`'s exactly-3-iter divergence pass is the
   concrete candidate. `dfsphReference`'s two-solve structure still
   under-builds the gradient (`pressureSlopeRatio` ~0.7) independent of the
   wall.

`dfsphReference` stays a troubleshooting artifact (its toggles all ship off);
no `divergenceFree` default changed by any of Parts 24–42 — the one shipped
default change from Parts 35–42 is `omniIncompressible.WALL_PRESSURE_MODE`
(`'mls'` in Part 41, `'shepard'` since Part 42; a new scheme, nothing
regresses), plus Part 42's `hydrostaticColumn.wallBC` param (default
`freeSlip` = no-op). Full suite + `gradcheck_incompressible.py` green.

---

## Parts 35–41 — the omniSPH port, the SPlisHSPlasH cross-check, the free-slip wall diagnosis, the viscosity path, the omniSPH bindings, the operator diff + wall-pressure fix

Full narrative: `DFSPH_FINDINGS.md` §9 (Parts 35–41), §1.12/§1.13/§1.14/§1.15, §2, §8.
The short version and what it changes:

**Part 41 (operator-by-operator diff + the wall-pressure closure).** The
`omniIncompressible` / composed-Jacobi departure from omniSPH is now located
and fixed. Rest-state interior operators (`operator_diff.py`): density, source,
`alpha` (modulo the `ρ0` convention), `a_p` all match omniSPH to round-off.
Transient (`transient.py`): warpSPH's density Jacobi does **not contract at the
wall band** — it had the wall in `alpha` only (`applyConsistentCoupling`'s
Akinci band) with boundary `p ≡ 0` (BWJ23 Eq. 33), where omniSPH's
`densitySolve` recomputes an MLS wall pressure from the current fluid `p`
*every iterate* and feeds its gradient into `a_p` / `A·p` / `alpha` (a Robin
closure). **Fix, landed as the `omniIncompressible` default:**
`WALL_PRESSURE_MODE = 'mls'` — `modules/incompressible/wallPressure.py`
(`wallPressureExtrapolation`, reusing `modules/liu`'s `interpolateLiuLiu`, the
same first-order fit `computeMdbcPressure` uses; **no relaxation / no carried
state**, so *not* the `mdbcMlsPressure` feedback instability). `hydrostaticColumn`
nx=128: was diverging → **holds 400+ steps at `pressureSlopeRatio` 0.99–1.00**.
Two loose ends tied off: `omega = 0.5` still detonates *with* the fix, so the
`OMEGA = 0.3` window is a **separate, pre-existing** `n_h = 4` bulk-conditioning
issue (§ deviations table), not the wall — it is what leaves the residual
`nRho ~180` vs omniSPH's 4. Wall pressure is **density-solve-only** (into the
divergence Jacobi → detonates step ~110; omniSPH's `divergenceSolve` has none).
On `iisph` / `dfsphReference` the same flag is default `None` — wash-to-negative
for `iisph` (already holds), inconclusive for the two-solve path.

**What landed.**

- **`IncompressibleSPHScheme.omniIncompressible = 3`** (`schemes/omniIncompressible.py`,
  `schemes/builder.py::_omniIncompressible`, `enumTypes.py`) — a faithful
  transcription of omniSPH's `SPHSimulation::timestep`: two Jacobi solves
  (3-iter divergence, ≤256 constant-density) on ONE neighbourhood, all
  pressure accels accumulated into a single semi-implicit Euler step; no
  gauge / guard / mask. `OMEGA = 0.3` (omniSPH's `0.5` detonates — Part 29's
  window); `XSPH_FLUID = 0.05` / `XSPH_BOUNDARY = 0.0` (omniSPH's `XSPH` +
  `BXSPH`, a post-solve velocity filter; `0/0` = the faithful no-dissipation
  loop). Reuses `DFSPHReferenceSystem`.
- **`wp_dfsph_factor.py`: the back-reaction sum-of-squares gate is now `ki == 0`**
  (query kind) instead of `kj == 0` (neighbour kind) — a fluid query
  accumulates it over fluid + boundary neighbours; a non-fluid query gets 0.
  A deliberate departure from SPlisHSPlasH / [BWJ23] Eq. 32 (fluid-only): it
  softens the near-wall `alpha`, same intent as `akinciBoundaryVolumeScale`.
  Bulk-identical to the reference. Affects `iisph` / `dfsphReference`.
- **`scripts/splishsplash_compare/`** — the SPlisHSPlasH bindings work in both
  conda envs now (base `.pth`; warp env `.so` rebuilt for py3.13). The kit:
  matched scene, initial-state diff, the exact-state import driver, the
  `n_h`/kernel sweep, the semi-periodic isolation variant, the video batch.

- **The viscosity path (Part 39, then the Part 42 cleanup)** — Part 39
  hand-rolled a `_physViscosity` helper + three `PHYS_VISCOSITY_*` module
  globals on `schemes/dfsphReference.py`. **Part 42 removed them:** the Adami
  no-slip *mirror* is `computeBoundaryVelocities` with `BCType.noSlip`
  (`modules/mdbc/velocity.py`, the call `schemes/deltaSPH.py` / `schemes/dfsph.py`
  already make — `dfsphReference_step` now makes it in step 2), and the
  physical viscosity is the stock `computeVelocityDiffusion` /
  `schemeConfig.diffusionParams.viscidNu` term already in that step.
  `hydrostaticColumn` got a `wallBC` param (default `freeSlip` = the
  historical no-op). **But the stock `viscidNu` term is normal-projected
  (`μ_ij·∇W`, approach-only) — no tangential stress — so `wallBC=noSlip` + `nu`
  bounds the slosh KE (4× down) and holds the gradient but roughens the
  surface (embMin 0.94 → 0.60) and spikes `|v|`.** A clean viscous no-slip
  wall needs a shear-carrying Morris laminar term in the module layer — a
  TODO, see the ranked queue. `probe_hydrostaticColumnViscosity.py` +
  `make_videos_viscosity.py` removed.

**What it establishes (and redirects the track).**

1. **The `hydrostaticColumn` failure is not the density solve.** Part 38's
   semi-periodic test (x wraps, floor wall only, native Wendland2 / `n_h = 4`):
   `omniIncompressible` holds the column at `|v|max ≈ 0.22`, KE ≈ 1e-4,
   `embeddedMinDensity` ≈ 0.99, `pressureSlopeRatio` ≈ 1.00. The vertical
   physics is sound. **The walled-case slosh, the ~30 % column drop, and the
   spray are the free-slip side walls** — the missing ingredient is a wall
   no-slip, not a pressure-solve lever (confirms Part 32 from the other side).
2. **SPlisHSPlasH holds the matched case, and the setup matches for ~0.3 s.**
   Importing SPlisHSPlasH's exact fluid state into warpSPH reproduces its
   slump transient (`|v|max` peak ~2.4 at t ≈ 0.14) for ~150 steps; then
   `dfsphReference` detonates (~step 50), `omniIncompressible`'s Jacobi
   detonates (t ≈ 0.4), `iisph`'s surface thins. **warpSPH's operator-composed
   divergence-free Jacobi is not contractive at SPlisHSPlasH's `h = 2·dx` /
   cubic discretization** where the dedicated `TimeStepDFSPH.cpp` at
   `omega = 0.5` is.
3. **Do not chase SPlisHSPlasH's `h = 2·dx`.** It blows up in warpSPH even in
   the trivial semi-periodic case — the operators / density estimator need
   `n_h ≳ 3`. warpSPH's native `n_h = 4` is the right discretization for it.
4. **Neither the `ki==0` factor nor XSPH dissipation closes the DFSPH path at
   nx=128.** `dfsphReference` + `ki==0` holds ~940 steps then hits the same
   Parts 26/30/31 late-time degradation; XSPH is a wash there (heavier XSPH
   diverges sooner). The divergence-free Jacobi remains the blocker (§9 item).
5. **The free-slip slosh on `iisph` at nx=128 decays under a viscous no-slip
   wall (Part 39) — but cleanly only with a *shear-carrying* viscosity term
   (Part 42).** Part 39's hand-rolled `_physViscosity` (Adami mirror + a full
   Brookshaw vector Laplacian `ν∇²v` + a mirror-velocity clamp) took the slosh
   `|v|max` 1.7 → ~0.8, KE 0.04 → ~0.012 flat, `embeddedMinDensity` /
   `pressureSlopeRatio` unchanged. Part 42 removed it and re-did it through the
   stock path (`computeBoundaryVelocities`/`BCType.noSlip` + `viscidNu`): that
   bounds the slosh KE 4× and holds the gradient, but the stock `viscidNu`
   term is normal-projected so it roughens the surface (embMin 0.94 → 0.60)
   and spikes `|v|`. XSPH is null-to-negative; fluid-only viscosity roughens
   the surface (embMin → 0.43). The *wall* is the right place for the stress
   (item 1 / Part 38), and it wants a shear-carrying Morris term the module
   layer does not have — ranked-queue TODO.
6. **omniSPH itself, run through its Python bindings, holds the matched
   column — and shows the warp port's real gap is the boundary model, not
   the missing viscosity (Part 40).** `omnySPH` (`~/dev/omniSPH/omnySPH`)
   now builds against the warp env (`scripts/omnisph_compare/build_omnysph.sh`)
   and exposes `timestep` + every substep + all `fluid*` buffers. On a
   matched hydrostatic column (`column.yaml`, its native `n_h ≈ 1.8`):
   omniSPH's shipped loop (DFSPH + `XSPH` 0.01 + `BXSPH` **wall no-slip**
   0.50) **decays** `|v|max` 0.6 → 0.05 and KE 3e-3 → 8e-5 with `ρ` pinned
   at 1.002 and the surface flat to ±0.003; **fully inviscid** (both filters
   off) it is still bounded and near-rest (`|v|max` ~0.45, `ρ` 1.002, KE
   flat, no divergence) — where warp's `omniIncompressible` / `iisph` at
   `n_h = 4` sloshes at `|v|max ~1.7` or needs `OMEGA = 0.3` to not diverge.
   The difference: omniSPH's walls are **analytic solid triangles** —
   `density()` adds the boundary kernel integral `k` (near-wall `ρ ≈ ρ0` at
   rest → constant-density source ≈ 0 at the wall), every solve operator
   takes the analytic wall gradient `gk`, and `computeBoundaryPressure` (MLS)
   runs **inside every Jacobi iteration**. The warp family uses a 5-layer
   Akinci particle band wrapped once by `applyConsistentCoupling` — no
   analytic fill, no per-iteration wall pressure. Part 39 was not wrong
   (omniSPH's `BXSPH` is a shipped wall no-slip and the Adami mirror
   re-derives it — and omniSPH is stable at `boundaryViscosity = 0.01`, not
   only its `0.5` default, so a *light* no-slip suffices). **Direction
   (user's call):** do **not** port the analytic triangle path; the interior
   fluid physics matter more first, and when the boundary is addressed use
   **`band2018pb`** (Band et al. 2018, *Pressure Boundaries for Implicit
   Incompressible SPH*) — the extended PPE where boundary samples are solve
   unknowns in the same Jacobi loop — not the triangle geometry, not the
   Akinci-volume coupling, not the `band2018` MLS extrapolation. See
   ranked-queue item 0.

---

## Ranked queue

0. **The fluid physics vs omniSPH, operator by operator — largely resolved
   (Part 41).** Tooling: `scripts/omnisph_compare/` (`operator_diff.py` =
   rest-state operator diff, `transient.py` = 3-arm per-step/per-solve A/B,
   `wallp_ab.py` = the `iisph`/`dfsphReference` cross-check, `make_video.py`).
   **What it found:**
   - **The interior operators match omniSPH.** On omniSPH's exact rest lattice
     with matched `h` / Wendland2: density `~5e-6`, constant-density source
     exact, `alpha` exact modulo the `ρ0` convention (omniSPH carries
     `fluidRestDensity = 998`; `alpha` / `a_p` / `p` all scale with `1/ρ0` and
     `a_p = −∇p/ρ` is invariant — a unit choice, not a bug),
     `computePressureAccelIISPH` on the analytic `p = ρ0 g(H−y)` gives
     `a_p_y ≈ +9.4` (SPH-gradient-of-a-linear-field error). The density solve
     holds the column on the *pristine* rest state in **neither** code (bulk
     `ρ/ρ0 ≈ 0.999` → `p ≥ 0` clamp zeros it); the hydrostatic gradient is a
     transient build-up in omniSPH too.
   - **The composed density Jacobi does not contract at the wall band.**
     omniSPH: `nRho = 4` every step. warpSPH nx=128: 256-iter cap, `errRho`
     *rising* (a positive overshoot), blow-up at the **bottom corners** by
     step ~10. Side walls removed (semi-periodic): still never converges but
     the column holds — so it is a real solver property, fatal only when the
     corners amplify it.
   - **Cause (read from both loops):** omniSPH's `densitySolve` recomputes an
     MLS wall pressure `p_b` from the current fluid `p` *every iterate* and
     feeds it into `a_p` + `A·p` + `alpha` (a Robin closure). warpSPH had the
     wall in `alpha` only (`applyConsistentCoupling`'s Akinci band, wrapped
     once) with boundary `p ≡ 0` (BWJ23 Eq. 33) — the near-wall iteration
     matrix is inconsistent (`D` carries a wall term `A·p` does not), the
     classic non-contraction (§1.8; §2 `diagonalOnly`).
   **What landed (Part 41):** `omniIncompressible.WALL_PRESSURE_MODE = 'mls'` —
   `modules/incompressible/wallPressure.py`'s `wallPressureExtrapolation`
   reuses `modules/liu`'s `interpolateLiuLiu` (the `computeMdbcPressure` fit,
   `p_b = α + β·x_b + γ·y_b`, **no relaxation / no carried state** so *not* the
   `mdbcMlsPressure` feedback instability). `hydrostaticColumn` nx=128: was
   diverging → **holds 400+ steps, `pressureSlopeRatio` 0.99–1.00,
   `densityP05` 1.000, `|v|max` ~0.56**. **Part 42 changed the default to
   `'shepard'`** (the same file's 0th-order mirror, no linear term): `'mls'`
   diverges the sheared `randomFlowIncompressible --bounded`, `'shepard'`
   holds both it and the column. Suite (102) + Krylov + runner +
   `gradcheck_incompressible` green. Shared with `dfsphReference` (→ `iisph`)
   behind `WALL_PRESSURE_MODE` (default `None` there — wash-to-negative for
   `iisph`, inconclusive for the two-solve path). Wall pressure is
   **density-solve-only** (into the divergence Jacobi → detonates).
   **What is left:**
   - **`omega = 0.5` still detonates *with* the fix** → the `OMEGA = 0.3`
     window is a **separate, pre-existing** bulk-operator conditioning issue
     (`n_h = 4` → ~50 nbrs → `ρ(D⁻¹A) ≈ 5.6`, § deviations table), not the
     wall. It is what leaves the residual `nRho ~180` vs omniSPH's 4. Options
     if the convergence cost matters: a Krylov solve of the same `A p = s`
     (cf. item 10), or `p_b` also fed into `alpha` (currently only `a_p`).
   - **The undamped free-slip slosh** (`|v|max` ~0.56) still wants a viscous
     no-slip wall — see item 0b and the shear-carrying-Morris-term TODO.
   - **`omniIncompressible` on the other cases (Part 42):** *periodic* — holds
     `tgv` (KE ratio 0.79 over 200 steps vs `divergenceFree`'s 0.996 —
     bounded but over-dissipative from the `XSPH_FLUID = 0.05` default) and
     `kolmogorovIncompressible` (KE → 1.38 vs 2.51 at matched forcing). Its
     3-iter divergence pass is what keeps it bounded where `iisph` blows up.
     *Wall-bounded sheared* — `randomFlowIncompressible --bounded`
     **diverges** (KE → ~1e31 within a few steps, `|v|max` → 1e15), *with*
     `WALL_PRESSURE_MODE = 'mls'` in the loop. So Part 41's MLS wall pressure
     holds the *quiescent* column but not a wall-bounded shear flow — the
     `OMEGA = 0.3` / 3-iter-divergence / Akinci-band combination is not
     robust there. dambreak nx=64 still preserved (200 steps, `maxRho`
     1.000); `staticBlob` marginally noisier (`|v|max` 0.14 vs 0.12). A full
     dambreak grade is untried.
   - **Boundary, the principled version: `band2018pb`** — the `'mls'` closure
     is `band2018pb`-lite (`p_b` *extrapolated*, not a solve unknown). See
     the sub-item below.
   - **Boundary, when it is time: `band2018pb`** — Band, Gissler, Ihmsen,
     Cornelis, Peer, Teschner 2018, *Pressure Boundaries for Implicit
     Incompressible SPH* (ACM TOG 37(2):14, `literature/`). The **extended
     PPE**: boundary samples enter the solve **as unknowns**, with their own
     source term and diagonal, iterated in the *same* Jacobi loop as the
     fluid — i.e. it maps directly onto the DFSPH/IISPH step this codebase
     already runs. Volume-centric (not density-centric) discretization. Its
     abstract's stated benefits are exactly this codebase's open problems:
     "reduced pressure oscillations, improved solver convergence, and larger
     possible time steps". **Not** the analytic triangle path, **not** the
     current Akinci-volume `applyConsistentCoupling`, and **not** the MLS
     extrapolation `band2018` / `[B]` (that is what `mdbcMlsPressure`
     already is, and it is the worst boundary mode measured — §2). Adami 2012
     §3.2 is the outside-the-loop extrapolation `band2018pb` supersedes.
   - **On the wall no-slip magnitude:** omniSPH is stable with
     `boundaryViscosity = 0.01` too, not just its `0.5` default — so a
     *light* wall no-slip suffices; any wall no-slip should aim low.

0b. **The viscosity path — Part 39 measured it, Part 42 folded it into the
   stock machinery.** `iisph` / `dfsphReference` now extend the boundary
   velocity field with `computeBoundaryVelocities` (per each boundary
   region's `BCType`) in step 2, exactly like `schemes/deltaSPH.py` /
   `schemes/dfsph.py`, and the physical viscosity is the ordinary
   `computeVelocityDiffusion` / `schemeConfig.diffusionParams.viscidNu` term
   already in that step. `hydrostaticColumn` exposes `wallBC` (default
   `freeSlip`). Part 39's `_physViscosity` + `PHYS_VISCOSITY_*` globals + the
   two throwaway probes are gone.
   - Graded (`iisph`, nx=128, 1200 steps): `wallBC=freeSlip` (default) is a
     strict no-op on the landed baseline (identical to `constant`/wall-v=0,
     matches Part 34). `wallBC=noSlip` + `viscidNu=0.01` bounds the slosh KE
     4× and holds the gradient (`slope` 1.02) but roughens the surface
     (`embMin` 0.94 → 0.60) and spikes `|v|` — the stock `viscidNu` term is
     `μ_ij·∇W`, normal-projected + approach-only, so a no-slip mirror only
     adds noisy *normal* wall damping. `wallBC=freeSlip` + `viscidNu` roughens
     the surface worse (`embMin` 0.43); `wallBC=extended` (MLS) is unstable.
   - **TODO (the shear-carrying Morris term).** A clean viscous no-slip wall
     needs a laminar viscosity that carries tangential stress: Morris et al.
     1997, `a_visc_i = Σ_j (m_j (μ_i+μ_j)/(ρ_i ρ_j)) (x_ij·∇W_ij /
     (r_ij²+ηh²)) v_ij` — the **full `v_ij` vector**, **no approach-only
     clamp** (that clamp is an artificial-viscosity device). Neither
     `wp_viscosityDelta.py`'s `inviscid=False` branch (which is projected
     despite its "Morris-style Laplacian" docstring) nor `dissipation/pi.py`'s
     `computePi_actual` (all Monaghan-family, projected) provides it. Add it
     as a real `DiffusionParameters`-wired option — a new `ViscosityTerms`
     value or a non-projected branch — gradcheck'd, with its own `deltaSPH`
     regression pass (`inviscid=False` is shared by tgv / kolmogorov /
     shearWave / staticBlob / impact / randomFlowIncompressible). Then
     `wallBC=noSlip` + `nu` gives Part 39's clean result through the stock
     path. Also fix `wp_viscosityDelta.py`'s docstring either way.
1. **Explain the dam break's dissipation** (Part 19). The channel is
   identified (Part 22): not the walls (Part 21 ruled out the free-surface
   clamp; Part 22's budget puts the no-pen shift at negligible), not
   viscosity alone — it is the incompressibility cycle (DF projection −35.8 +
   Eq. 17 resample +27.3, net −8.5, 85% of the loss), Monaghan viscosity
   secondary (−6.4). Both cycle channels are `divergenceFree`-only, which
   carries the cross-scheme gap against `deltaSPH`. **What remains is the
   mechanism:** that −8.5 is the residual of two ~30-magnitude terms — a
   discretization error that vanishes as `nx` grows, or a structural cost of
   the constraint? The natural instrument is an `nx` convergence of the
   cycle's net on this case — but the default `nx = 128` diverges mid
   free-fall, so this is a stability study as much as a dissipation one.
   Independent of the active track; can run in parallel.
2. **Re-measure the divergence-free half-state's contraction under
   `minShift`.** Under the clamp, `staticBoundary` on the DF solve alone
   removed only ~20% of the incoming residual per solve (vs ~50% under
   `full`) and eventually detonated; Part 14 removed the observable (it runs
   all 901 steps under `minShift`) but did not re-measure whether the
   per-sweep contraction is still halved. If it is, the mechanism is live and
   merely survivable, and worth understanding before any third solve is
   added. `probe_boundaryOperatorTerms.py --mode diag`.
3. **Warm-start the divergence-free solve only.** It genuinely converges, so
   this is the ordinary [BK] optimisation (~3x in iteration count).
   **Do not warm-start the constant-density solve** — it integrates, so a
   warm start carries a linear ramp across steps (Part 15, §1.7).
4. **`relaxationFactor = 0.3` has only a ~4% stability margin on a bounded
   state**, not the ~15% its docstring quotes from the TGV family. Cheapest
   robustness win available; independent of everything else.
   `probe_boundaryOperatorTerms.py --mode spectrum`.
5. **Move `computeMdbcPressure` inside the solver iteration** ([B] Alg. 1
   recomputes `p_b` from the current iterate every sweep — no state, no lag).
   **Part 41 did exactly this for the `omniIncompressible` / `dfsphReference`
   family** — `modules/incompressible/wallPressure.py`'s
   `wallPressureExtrapolation` is the no-relaxation, no-carried-state,
   recompute-every-iterate version, and it holds `hydrostaticColumn` nx=128.
   What is still open here is the (historical) `divergenceFree`
   `computeMdbcPressure` call in `schemes/dfsph.py` (once per step, carried as
   under-relaxed state) and [B]'s **SVD-safe inversion** on the MLS gradient
   system (the codebase falls back on a neighbour-count threshold of 9 with no
   conditioning guard). *But:* `mdbcMlsPressure` is the worst `divergenceFree`
   boundary mode measured and should be **deprecated rather than repaired** —
   do this only if it is kept.
6. **Test the mDBC hypothesis for `DensityEvolution.hybrid`.**
   `computeMdbcDensity` runs at the top of the step on the *carried* density,
   so under `hybrid` the boundary rows are extrapolated from a drifted field;
   the periodic case has no mDBC, which is exactly the difference between
   `hybrid` working there and dying at 286 steps at a wall. Cheap test:
   re-sum for the extrapolation only.
7. **Grade `shearWave` against [C]'s Fig. 3 and Fig. 4.** Blocked on the
   paper — `literature/MANIFEST.md`. Everything else about the case is done
   (Part 16).
8. **Rename `dfsph.py` / `dfsph_step` → `vdps.py`** (§1.3). Zero-risk; the
   registered scheme name already needs no change.
9. **The scheme split — a real two-solve `dfsph` scheme.** Fully specified by
   [BK] Alg. 1: density solve → integrate → divergence solve; no position
   shift; full warm start; tolerances 1e-4 / 1e-3; [B]'s MLS boundaries
   recomputed inside each iteration. Part 33 landed the density half of this
   as `iisph` (it holds `hydrostaticColumn`); the open piece is the
   divergence pass, which is exactly what `dfsphReference`'s Jacobi has never
   made survive a woken-up free surface (Parts 26/28/29/33/37), and what
   `omniIncompressible`'s 3-iter divergence Jacobi cannot hold at nx=128
   either (Part 35). Part 36 sharpened the target: SPlisHSPlasH's dedicated
   `TimeStepDFSPH.cpp` Jacobi at `omega = 0.5` **is** contractive on the
   imported exact state where warpSPH's operator-composed one is not — the
   gap is the composed operator's conditioning, not the algorithm. The path
   is `iisph` + a *contractive* divergence solve (dedicated kernels, or a
   Krylov solve of the same `A p = -Drho/Dt` — cf. item 10), not the
   `dfsphReference` / omniSPH two-solve structure hardened. **Stays last.**
10. **`MINRES` without the non-negativity clamp** — 2.15x on periodic density
    error for 1.73x wall time. Needs (a) a free-surface test (a signed
    shifting potential can *pull* particles together — Part 21 showed forcing
    the clamp off NaNs `dambreak` in 4 steps) and (b) an equal-cost
    comparison against `relaxedJacobi` at `maxIterations = 128`. Opt-in at
    best.

---

## Known-open, lower priority

- **`rotatingSquarePatch` corner density loss** (Part 3). Resolution-
  independent `rho ≈ 0.506` at the four convex free-surface corners; `--scheme
  deltaSPH` holds 0.9998. **[BK] §5 documents this as a known method
  limitation** with a published remedy (ghost particles, Schechter & Bridson
  2012, now in `literature/` as `schechter2012`) — not an implementation bug.
  Two independent minor bugs on the case still want fixing: it has **no
  `Case.timestep` hook** (so `dt` is never adapted) and it inherits
  `integrationScheme='rungeKutta2'`. The commented-out
  `pressureB[surfaceIndicators == 1] = 0.0` in `divergenceFree.py` is
  untestable as-is (`detectFreeSurface` flags 96/100 particles) and is not
  [BK]'s remedy anyway (Part 21).
- **Nothing enforces `semiImplicitEuler`.** The PPE derivation is specific to
  it; `CaseSpec`'s default is `rungeKutta2` and no path checks. Candidate: a
  one-line assert/warn in `dfsph_step`.
- **`solveIncompressible` should raise on the Krylov path**, the way
  `JacobiRelaxationMode.optimal` already does — it routes through
  `solvePressureKrylov(..., gauge='nonnegative')`, the clamped-solve
  combination both papers rule out.
- **Two mDBC slip defects** (`modules/mdbc/velocity.py`). `freeSlip` computes
  `u_f - 1*(u_f.n)n` while its comment says `- 2*`; `noSlip` applies the same
  projection undocumented. Fixing measures *worse* (§2), but comment and code
  should agree. Separately, **`noSlip`'s `2 u_wall` term is dead code** — it
  reads the ghost row's velocity, which `rigidBody/update.py` refreshes only
  for `BCType.constant`. Latent today; would bite the first moving no-slip
  wall not backed by a Dirichlet.
- **Two-way coupling is absent.** [BWJ23] Eq. 35's `f_{k<-i}` is never
  applied. Fine today (all walls static); a moving-body case would silently
  get one-way coupling.
- **`DensityEvolution` + `BoundaryPressureMode.plain` is a trap** — `plain`
  skips the mDBC extrapolation and the non-summation modes skip the re-sum,
  so `kind != 0` rows would never be updated. Documented in the enum, not
  guarded.
- **Ghost particles (`kind == 2`) are lumped in with boundaries** by the
  `kj == 0` test. Correct for this scheme (`dfsph_step` freezes them too),
  but no measurement here separates them.
- **Full-suite failures unrelated to this track (pre-existing on a clean
  tree, verified Part 48 by `git stash` + re-run):**
  - `test_implicitShiftingComparison.py::test_automaticImplicitShift_convergesLikeHandBuilt`
    — the historical `implicitShiftAutomatic` flake (~1 run in 3; passed on
    retry Part 48).
  - `test_incompressibleKrylov.py::test_optimalStepRejectedForConstantDensitySolver`
    — **fails consistently, not a flake.** It expects `solveIncompressible`
    with `JacobiRelaxationMode.optimal` on the CD solver to raise
    `ValueError('optimal')`; it no longer does. Related to the known-open
    "`solveIncompressible` should raise on the Krylov path" item above — that
    guard has regressed at some earlier point. Not touched by Part 48.
  (`test_minresGivensMatchesDenseLstsq`, the previously-noted Krylov flake,
  passed Part 48.)
- **Parts 35–38 are not suite/gradcheck-validated.** The `wp_dfsph_factor.py`
  `ki == 0` change (Part 37) alters `computeDFSPHFactor` for `iisph` /
  `dfsphReference` and touches a `@wp.kernel` — run `gradcheck` (the
  `wp_dfsph_factor` module) and the incompressible physics suite before
  relying on it. `omniIncompressible`'s `XSPH_FLUID = 0.05` default ships
  **non-inert** (it is a new scheme, so nothing regresses, but the "faithful
  omniSPH loop" is `0/0`). Everything under `scripts/splishsplash_compare/` is
  throwaway comparison tooling, not part of the suite.

---

## Done — for the record

- **The four-way default change** (Parts 12–14): `cflFactor` units, `minShift`
  on bounded solves, `staticBoundary` on both solvers landed;
  `BoundaryPressureMode.consistent` and `akinciBoundaryVolume` rejected
  (inert / diverges). `boundaryOperatorTerms` moved per-solver; the PS-only
  split is rejected by measurement (`both` is 1.45x better).
- **The stopping criterion** (Part 15): measured, not a defect. Periodic
  cases converge in 3 iterations; the constant-density solve does not
  converge in any norm (it integrates — `maxIterations` is a shift gain);
  the shipped iteration budget is on the accuracy/cost frontier. Landed one
  configurable `JacobiConvergenceCriterion` + `rtol` disjunct, both inert.
- **`shearWave` ported** (Part 16); **`ShiftApplication` settled on
  `positionShift`** at a pinned `dt` (Part 18) — the velocity modes cost
  2.1x the kinetic energy for 2x lower density error, identical wall
  behaviour.
- **`dambreak` incompressible `timestep` hook** (Part 20): `dambreakTimestep`,
  `--scheme divergenceFree` only, `--cflFactor 0.2` (0.4 diverges here).
- **The three baseline cases** (Part 23): `staticBlob`, `impact`,
  `hydrostaticColumn` landed + the `relaxLattice` free-surface guard.
- **The dam-break energy budget** (Part 22): `probe_dambreakEnergyBudget.py`,
  closes `dKE` exactly.
- **`IncompressibleSPHScheme.iisph`** (Part 33): plain IISPH ([I]),
  velocity-impulse CD solve only. First scheme in the codebase to hold
  `hydrostaticColumn`; also holds `staticBlob` where `dfsphReference`
  diverges. `iisph_step` shares `dfsphReference`'s body with
  `skipDivergence=True`. The Parts 26/30/31/32 "late-time degradation" was the
  divergence Jacobi (instability) + `minDensity` reading ballistic surface
  spray (FOM artifact); both gone. Wall-XSPH `ε_b` and rest-density
  calibration both measured negative here.
- **`IncompressibleSPHScheme.omniIncompressible`** (Part 35): a faithful
  transcription of omniSPH's two-solve loop (both solves on one neighbourhood,
  accumulate-then-integrate). Does not hold `hydrostaticColumn` at nx=128;
  Part 38 showed the vertical physics is sound and the residual is the
  free-slip side walls. `OMEGA = 0.3`, `XSPH_FLUID`/`XSPH_BOUNDARY` toggles
  (omniSPH's `XSPH` + `BXSPH`, default `0.05`/`0.0`). Reuses
  `DFSPHReferenceSystem`.
- **The SPlisHSPlasH cross-check** (Parts 36–38): `pysplishsplash` bindings
  installed in both conda envs; SPlisHSPlasH's DFSPH holds the matched
  hydrostatic column; importing its exact fluid state into warpSPH reproduces
  the transient for ~0.3 s, then warpSPH's composed divergence-free Jacobi
  loses it (not contractive at `h = 2·dx` / cubic). `scripts/splishsplash_compare/`.
- **`wp_dfsph_factor.py` `ki == 0` back-reaction gate** (Part 37): the
  sum-of-squares term is query-kind gated (fluid query → all neighbours;
  non-fluid → 0). A deliberate departure from the fluid-only reference; softens
  the near-wall `alpha`. Affects `iisph` / `dfsphReference`.
- **The viscosity path for the incompressible scheme family** (Part 39,
  reworked in Part 42): Part 39 hand-rolled `_physViscosity` +
  `PHYS_VISCOSITY_*` module globals on `schemes/dfsphReference.py` and
  measured that a real (shear-carrying) Adami no-slip wall decays the
  `hydrostaticColumn` nx=128 free-slip slosh with the surface + gradient
  intact, while fluid-only `nu·∇²v` roughens the surface and XSPH is
  null-to-negative. **Part 42 removed the bespoke path:** the mirror is
  `computeBoundaryVelocities` / `BCType.noSlip` (step 2, like `deltaSPH.py`),
  the viscosity is the stock `viscidNu` term, `hydrostaticColumn` gets a
  `wallBC` param (default `freeSlip` = no-op on the baseline). The stock
  `viscidNu` term is normal-projected, so `noSlip` + `nu` bounds the slosh
  KE but roughens the surface — a shear-carrying Morris term is a TODO
  (ranked queue item 0b). `gradcheck_incompressible` + tgv/shearWave/dambreak
  physics green.
- **The omniSPH Python bindings + cross-check** (Part 40):
  `scripts/omnisph_compare/` — `build_omnysph.sh` builds `~/dev/omniSPH/omnySPH`'s
  `_core` against the warp env (the repo ships a py3.14-only `.so`), exposing
  `SPHSimulation.timestep` + every substep (`computeAlpha`, `computeSourceTerm`,
  `divergenceSolve`, `densitySolve`, …) + all `fluid*` buffers. `run_omnisph.py`
  + `column.yaml` run a matched hydrostatic column; `ablate_xsph.py` toggles
  `XSPH` / `BXSPH`. Established that omniSPH holds the column fully inviscid
  and that the warp port's gap is the **analytic triangle wall boundary**
  (ranked-queue item 0), not the viscosity. Throwaway comparison tooling.
- **The operator diff + the per-iterate wall-pressure closure** (Part 41):
  `scripts/omnisph_compare/{operator_diff,transient,wallp_ab,make_video}.py`
  (throwaway). Rest-state interior operators match omniSPH; the composed
  density Jacobi does not contract at the wall band because it had the wall in
  `alpha` only where omniSPH recomputes an MLS wall pressure every iterate.
  **Landed:** `modules/incompressible/wallPressure.py` (`wallPressureExtrapolation`,
  `'shepard'` / `'mls'`, reuses `modules/liu`, no relaxation / no carried
  state) + `omniIncompressible.WALL_PRESSURE_MODE` **default** — the first
  setting that holds `hydrostaticColumn` at nx=128 under `omniIncompressible`
  (was diverging, Part 35): 400+ steps, `pressureSlopeRatio` 0.99–1.00. Part 41
  shipped `'mls'`; **Part 42 changed the default to `'shepard'`** (`'mls'`'s
  linear term diverges the sheared `randomFlowIncompressible --bounded`;
  `'shepard'` holds both). Same flag on `dfsphReference` / `iisph` (default
  `None`, A/B negative-to-inconclusive). `omega = 0.3` is unchanged — the
  `omega`-window is `n_h = 4` conditioning, not the wall.
- **The free-surface CD solve — characterised + `CD_TIKHONOV`** (Part 43):
  `scripts/probe_omniIncompressibleCDSolver.py` (jacobi/bicgstab/gmres ×
  Tikhonov sweep on `hydrostaticColumn` / `dambreak` / `randomFlowBounded`) +
  `scripts/probe_omniIncompressibleCDSystem.py` (offline `A p = s` capture from a healthy run).
  The free-surface constant-density operator is **near-singular** — the
  quiescent column's source sits near its near-null space, the Jacobi hits
  its 256-iter cap every step (omniSPH's floored metric hides it), and
  removing the `p ≥ 0` clamp blows the solution to `|p| ~ 1e9`. **Landed
  (default-inert): `omniIncompressible.CD_TIKHONOV`** (`0.0`), a uniform
  diagonal shift `tik·median(|alpha_fluid|)` applied where `CD_SOURCE_PROJECT`
  did not fire; on the Jacobi path `0.1` takes `hydrostaticColumn` nx=128 off
  the cap (210 → ~75 iters, holds 400 steps, quality neutral-to-better) and
  is a strict wash on `dambreak`. **Also landed (scaffold): `CD_SOLVER ∈
  {jacobi, bicgstab, gmres}`** + a reject guard + `wallPressureExtrapolation`'s
  `clampNonNeg=False` — but non-symmetric Krylov **breaks down** on the
  composed wall operator on every wall-bounded case (returns tiny-residual /
  `|p| ~ 1e9` iterates), so `'jacobi'` stays the only usable setting and the
  real fix is `band2018pb` (ranked-queue item 0 / "Next" 1).
  `gradcheck_incompressible` + tgv/shearWave/dambreak/incompressible physics
  green; `tik = 0` / `'jacobi'` bit-identical to pre-Part-43.
- **`A` is already symmetric — symmetrising it is not the fix** (Part 44):
  `scripts/probe_omniIncompressibleCDSymmetry.py` measures the CD operator's
  relative symmetry defect at **2.7e-5–8e-3** (fp32 noise) for the
  boundary-`p ≡ 0` and `krylov.buildIISPHMatvec` forms; the per-iterate
  wall-pressure Robin closure adds only ~4e-3 / ~10 %. MINRES / CG / BiCGStab
  all still diverge (`|x|` → 1e4–1e7) because the operator is
  **rank-deficient** at free surfaces + wall corners (`median|alpha_fluid|`
  → 2.5e-5 on `dambreak`), which only `band2018pb`'s extra boundary equations
  remove. Also: `dambreak`'s "3-iter" CD is the floored omniSPH metric — the
  captured system has `|r|/|s| = 1.0` after 2000 Jacobi iters. Analysis only,
  no code changed.

Full detail for all of the above: `git log -p DFSPH_IMPROVEMENT_PLAN.md`,
indexed one line per part in `DFSPH_FINDINGS.md` §9.
