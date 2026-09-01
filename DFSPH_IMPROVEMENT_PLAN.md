# warpSPH — Incompressible (VD+PS / DFSPH) Improvement Plan

Working document for the incompressible SPH path (`schemes/dfsph.py`,
registered as `IncompressibleSPHScheme.divergenceFree`), the `dfsphReference`
troubleshooting scheme, the `iisph` baseline (Part 33), and the
`omniIncompressible` omniSPH-loop port (Part 35).

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

## Current state

The incompressible path is **VD+PS** (Cornelis et al.), faithfully
implemented, registered as `divergenceFree`. Three shipped-default changes
landed from this work, all measured first; every other switch is opt-in and
default-inert. Three further schemes are now registered as baselines:
`dfsphReference` (DFSPH-proper troubleshooting artifact, Parts 24–32),
**`iisph`** (plain IISPH, Ihmsen et al. 2014 — Part 33, the first scheme to
hold `hydrostaticColumn`; but **CD-only, so not a general scheme** — Part 42
found it injects energy on `tgv`, viable only near-quiescent), and
**`omniIncompressible`** (omniSPH's two-solve loop, Part 35 — **now holds
`hydrostaticColumn` at nx=128**, Part 41, via a per-iterate MLS wall pressure).

- **`cflFactor = 0.4` against the particle diameter** (Part 12, committed).
- **`ShiftPressureGauge.minShift` is the default** (Part 4) and now reaches
  wall-bounded solves (Part 14); still falls back to `nonNegativeClamp` where
  there is a free surface.
- **`BoundaryOperatorTerms.staticBoundary` on both solvers** (Part 14). A
  strict no-op on any case with no `kind != 0` particles.
- The last two are one fix at two points: together they take bounded
  `randomFlowIncompressible` at nx=128 from a density band of 1.78e-1 to
  **4.48e-3** (40x, and 5.4x better than they compose independently).
- Several real bugs fixed (`DFSPH_FINDINGS.md` §7), the largest being the
  Eq. 17 resample, the boundary-row masking, and the `drhodt` pre-projection
  evaluation.
- Full suite passes (241 passed / 1 skipped), `gradcheck_incompressible.py`
  passes, `run_sweep.py` 30/30. Two known pre-existing flakes
  (`DFSPH_FINDINGS.md` §4-known-open list, carried below).

**Case status at the shipped defaults.**

| case | status |
|---|---|
| `tgv`, `kolmogorovIncompressible`, `shearWave` (periodic) | `divergenceFree` healthy. `iisph` **fails `tgv`** (injects energy, KE ×145 — CD-only, no divergence pass, Part 42); `omniIncompressible` holds them (over-dissipative from `XSPH_FLUID = 0.05`). |
| `randomFlowIncompressible --bounded` | `divergenceFree` best it has ever been (band 4.48e-3); still where all remaining wall error lives. `iisph` **diverges** (`\|v\|max` → 1e6, Part 42); `omniIncompressible` **holds with `WALL_PRESSURE_MODE = 'shepard'`** (`\|v\|max` decays 2 → 0.3, 300 steps — Part 42; `'mls'` diverged here, step-1 CD-Jacobi blow-up). |
| `staticBlob` (free space), `impact` (collision) | **pass** (baseline cases, Part 23) |
| `hydrostaticColumn` (quiescent column under gravity) | `divergenceFree` **fails** (position shift cannot sustain a body force); **`--scheme iisph` holds it** — Part 33 at nx=32 / 2000 steps, **Part 34 confirmed at the default nx=128** (clean to `tLimit`, and to t=2.9 / 1500 steps): stable geometry, `embeddedMinDensity` ~0.92–0.96, `pressureSlopeRatio` ~0.99, with a bounded undamped free-slip bulk slosh (KE creeps ~0.025→0.05 at nx=128) and cosmetic surface spray (the graded FOMs `densityP05` / `embeddedMinDensity` now exclude it). `dfsphReference` (two-solve DFSPH) still does not — its divergence Jacobi is the instability (Part 37: `ki==0` factor buys ~940 steps, then the same late-time surface degradation). **`omniIncompressible` (omniSPH port, Part 35) now holds nx=128 (Part 41)** — the composed density Jacobi was not contracting at the wall band (blow-up at the bottom corners by step ~10) because it had the wall in `alpha` only, boundary `p ≡ 0`, where omniSPH recomputes a wall pressure every iterate. A per-iterate wall-pressure closure (`WALL_PRESSURE_MODE`, built on `modules/liu` / a Shepard mirror) closes it: 350+ steps clean, `pressureSlopeRatio` 0.99–1.00, `densityP05` 1.000, `|v|max` ~0.56 (the undamped slosh, still wants a shear-carrying no-slip wall to damp — Part 42 TODO). **Default is `'shepard'` (Part 42): `'mls'` (Part 41's first default) holds this case but diverges the sheared `randomFlowIncompressible --bounded`; `'shepard'` — no linear term — holds both.** **Part 38 nailed the interior: side walls removed (x-periodic / floor-only), `omniIncompressible` sits at vmax ~0.2, `ρ≡ρ0`, slope 1.00 — the vertical density-solve physics is sound.** **Part 39 / 42: a viscous no-slip wall takes the `iisph` nx=128 slosh from vmax ~1.7 toward ~0.8; Part 39's hand-rolled shear Laplacian did it with the surface intact, but the stock `viscidNu` term (Part 42's cleanup) is normal-projected and roughens the surface — a shear-carrying Morris term is the TODO.** `dfsphReference` (two-solve DFSPH) still under-builds the gradient (`pressureSlopeRatio` ~0.7) and its divergence Jacobi is GPU-stochastic near step ~940 (Part 37); the wall pressure helps its slosh but is inconclusive there (Part 41). SPlisHSPlasH's own DFSPH holds the matched case; importing its exact state reproduces the transient for ~0.3 s, then warpSPH's composed Jacobi loses it (Part 36). |
| `dambreak --scheme divergenceFree` | **runs** (Part 19) — the only working free surface — but half `deltaSPH`'s run-out speed and most of the flow's KE dissipated on impact; needs its own `--cflFactor 0.2` (Part 20), not 0.4 |
| `rotatingSquarePatch --scheme divergenceFree` | broken; [BK] §5 documents it as a method limitation, not an implementation bug |

---

## Active track — `omniIncompressible` on `randomFlowIncompressible --bounded`

**Why.** `omniIncompressible` + the Part 41 MLS wall pressure holds the
*quiescent* `hydrostaticColumn` at nx=128, but **diverges on the wall-bounded
*sheared* flow** `randomFlowIncompressible --bounded` (Part 42): it detonates on
step 1 (KE 0.35 → ~1.4e3, `|v|max` → 83 from an enormous near-wall pressure
impulse — density stays fine at [1.001, 1.001], so a solve overshoots), briefly
recovers over steps 2–7, then re-detonates by step ~10 → `|v|max` 1e15 by
step 14. `iisph` also fails this case (KE → 5e9); `dfsphReference` untested
here; `divergenceFree` (VD+PS) holds it (KE ratio 0.896, `|v|max` 1.06). So the
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

**Still open — the deeper issue.** `'shepard'` *makes the run survive*, it does
not make the constant-density Jacobi *converge*: it still hits the 256-iter
cap on many steps (§1.7 — "the CD solve does not converge, it integrates"),
and the run only holds because the un-converged impulse decays rather than
compounds. A genuinely robust `omniIncompressible` (and any `iisph` + a
divergence pass, ranked queue item 4/9) still wants:
- a **contractive constant-density solve** — a Krylov solve of the same
  `A p = s` (item 10), or real preconditioning;
- **`band2018pb`** (boundary samples as solve unknowns, ranked-queue item 0
  sub-item) — the principled boundary that fixes the near-wall conditioning
  the wall-pressure mirror only patches;
- feeding `p_b` into `alpha` (not just `a_p`).
Next concrete steps: grade `'shepard'` on `dambreak` (has walls) and the
periodic cases for regressions (wall pressure is a no-op with no `kind == 1`,
but confirm), then a longer `randomFlowIncompressible --bounded` run vs
`divergenceFree`'s density band.

---

## Earlier track — a velocity-coupled incompressible scheme for the column

**Why.** `divergenceFree` cannot hold a quiescent, wall-bounded,
free-surface-under-gravity state (`hydrostaticColumn`, Part 23): the VD+PS
density-invariance correction is a momentum-neutral position shift
(`DFSPH_FINDINGS.md` §1.2/§1.3), and a position shift cannot sustain a body
force. The correction has to be applied to *velocity*.

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
   `rhoStd` 1.3e-3) throughout. `divergenceFree` holds it (KE ratio 0.996,
   monotone) and so does `dfsphReference` (KE ratio 0.81, bounded) — the
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
   What is still open here is the `divergenceFree` (VD+PS) scheme's own
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
- **Two intermittent test flakes**, both pre-existing, neither a regression:
  `test_implicitShiftingComparison.py`'s `implicitShiftAutomatic` assertions
  (~1 run in 3) and
  `test_incompressibleKrylov.py::test_minresGivensMatchesDenseLstsq`.
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

Full detail for all of the above: `git log -p DFSPH_IMPROVEMENT_PLAN.md`,
indexed one line per part in `DFSPH_FINDINGS.md` §9.
