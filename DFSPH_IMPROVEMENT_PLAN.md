# warpSPH — Incompressible (VD+PS / DFSPH) Improvement Plan

Working document for the incompressible SPH path (`schemes/dfsph.py`,
registered as `IncompressibleSPHScheme.divergenceFree`), the `dfsphReference`
troubleshooting scheme, and the `iisph` baseline (Part 33).

**This file is the current state and the actionable list only.** The durable
findings — the physics lessons, the negative results, the literature notes, the
config surface, the bug list, the tooling index, and the one-line session
index — are in **`DFSPH_FINDINGS.md`**. The full part-by-part investigation
narrative (Parts 1–33) is in `git log -p DFSPH_IMPROVEMENT_PLAN.md`.

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
default-inert. Two further schemes are now registered as baselines:
`dfsphReference` (DFSPH-proper troubleshooting artifact, Parts 24–32) and
**`iisph`** (plain IISPH, Ihmsen et al. 2014 — Part 33, the first scheme to
hold `hydrostaticColumn`).

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
| `tgv`, `kolmogorovIncompressible`, `shearWave` (periodic) | healthy |
| `randomFlowIncompressible --bounded` | best it has ever been (band 4.48e-3); still where all remaining wall error lives |
| `staticBlob` (free space), `impact` (collision) | **pass** (baseline cases, Part 23) |
| `hydrostaticColumn` (quiescent column under gravity) | `divergenceFree` **fails** (position shift cannot sustain a body force); **`--scheme iisph` holds it** (Part 33) — stable geometry + bulk density + hydrostatic gradient over 2000 steps, with a bounded undamped free-slip bulk slosh and cosmetic surface spray. `dfsphReference` (two-solve DFSPH) still does not — its divergence Jacobi is the instability. |
| `dambreak --scheme divergenceFree` | **runs** (Part 19) — the only working free surface — but half `deltaSPH`'s run-out speed and most of the flow's KE dissipated on impact; needs its own `--cflFactor 0.2` (Part 20), not 0.4 |
| `rotatingSquarePatch --scheme divergenceFree` | broken; [BK] §5 documents it as a method limitation, not an implementation bug |

---

## Active track — a velocity-coupled incompressible scheme for the column

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

1. `hydrostaticColumn --scheme iisph` clean to `tLimit` at the default
   `nx = 128` (Part 33 is nx=32); add a spray-robust FOM to
   `hydrostaticDiagnostics` (5th-percentile or embedded-min density) so the
   case grades the column, not the spray.
2. `dambreak` A/B: `iisph` vs `divergenceFree` vs `deltaSPH` — run-out speed
   and the energy budget (also a second data point for queue item 1: IISPH
   has no Eq. 17 resample).
3. `iisph` on `randomFlowIncompressible --bounded` and the periodic cases —
   does the velocity-impulse single solve beat VD+PS's position shift on the
   wall band, or cost the periodic cases dissipation (§1.2)?
4. Only then: whether a full two-solve DFSPH (`dfsphReference` hardened) is
   worth finishing, or `iisph` + a divergence pass added later is the path.

`dfsphReference` stays a troubleshooting artifact (its toggles all ship off);
no `divergenceFree` default changed by any of Parts 24–33; full suite +
`gradcheck_incompressible.py` green.

---

## Ranked queue

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
   recomputes `p_b` from the current iterate every sweep — no state, no lag)
   and add [B]'s **SVD-safe inversion** on the MLS gradient system (the
   codebase falls back on a neighbour-count threshold of 9 with no
   conditioning guard). *But:* `mdbcMlsPressure` is the worst boundary mode
   measured and should be **deprecated rather than repaired** — do this only
   if it is kept.
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
   made survive a woken-up free surface (Parts 26/28/29/33). The path is
   `iisph` + a *contractive* divergence solve added on top — not the
   `dfsphReference` two-solve structure hardened. **Stays last.**
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

Full detail for all of the above: `git log -p DFSPH_IMPROVEMENT_PLAN.md`,
indexed one line per part in `DFSPH_FINDINGS.md` §9.
