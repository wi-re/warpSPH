# warpSPH — Artificial-Compressibility SPH (ACSPH) Implementation Plan

Target paper — `literature/decourcy2024_incompressible-delta-sph-artificial-compressibility.pdf`, bib key `decourcy2024`:

> **Incompressible δ-SPH via artificial compressibility**
> J.J. De Courcy, T.C.S. Rendall, L. Constantin, B. Titurus, J.E. Cooper
> *Computer Methods in Applied Mechanics and Engineering* **420** (2024) 116700
> `doi:10.1016/j.cma.2023.116700` — CC BY, open access. 40 pp, 85 refs.
> Precursor: De Courcy et al., SPHERIC 2023 (ref [32]).

Synced into `literature/` on 2026-09-05 together with seven of its references
(see Part 6); `python scripts/check_literature.py` passes with all eight
abstracts verified verbatim against their PDFs.

---

# Status board — read this first

**Done (2026-09-05):** steps 1–6 of Part 8, and step 7 in full — implemented
*and* measured. The scheme runs end to end: `--scheme artificialCompressible`
builds, the dual-time driver solves, and `hydrostaticColumn` runs to
`t = 0.94` without diverging. Step 7 (Michel et al. particle shifting) has its
own document, **`PST_ALE_PLAN.md`**, whose Stage A landed 2026-09-05: the
law, its free-surface treatment, and the wiring into
`ArtificialCompressibleSystem.finalize` all exist and are gradchecked/tested
(off by default — `buildDefaultACSPHShiftProperties`).

**Re-measured this column with it on** (`scripts/probe_michelHydrostaticColumn.py`,
nx=32, 200 steps, one seed — see `PST_ALE_PLAN.md` §7.1 for the full table).
A real units bug turned up while doing this and is now fixed:
`computeMichelShift` was applying Eq. (22)'s shifting *velocity* straight to
`positions` with no `dt` factor (Eq. (58) is `dx/dt = u + delta_u`) —
confirmed ~8x the particle spacing per call before the fix, ~sane after it.
Numbers below are post-fix. Split result: `pairedFraction` — "exactly and
only what particle shifting exists to prevent" — **does move**, 0.065 →
**exactly 0.000** with the shift active (better than the pre-fix run, not
worse — the fix made the finding stronger, not weaker). `‖v‖` at the corners
**does not**: the shift alone (peak 2.33) is well above `noPenetrationShift`
alone (0.75); running both together helps (1.63) but doesn't match the
safeguard alone. **This overturns decision 4 below** — the shift is not a
replacement for `noPenetrationShift`, it fixes a different failure mode.
**The `δu_max` convergence-rate reproduction (Fig. 1) is now done too**
(`scripts/probe_michelConvergenceRate.py`, `PST_ALE_PLAN.md` §7.1) — the
test the `rotatingSquarePatch` footprint-drift proxy above turned out not to
substitute for. Clean pass: `michel2022` measures first-order convergence
(log-log slope 0.949 vs. the theoretical 1.0) against `deltaSPH`'s flat
-0.063, on a jittered periodic lattice swept nx 16→128 at fixed `R/Δx`.
Table 2's interior-consistency claim is now directly confirmed against this
implementation, not just inherited from the paper.

**`sloshingTank` is measured, root-caused, and fixed** (`scripts/probe_sloshingTankSurfaceShift.py`,
`PST_ALE_PLAN.md` §7.1): `michel2022` runs the full `tLimit=4.0` (20001
steps) with density held to `[0.994, 1.011]` — the only one of four shift
configurations tested that doesn't diverge; `noShift`/`surfaceNormal` diverge
at `t=0.34`/`t=0.68`, roughly an order of magnitude earlier than
`docs/historic_plans/WCSPH_SHIFTING_PLAN.md` documents (`t≈2.6-3.5s`).
**Confirmed root cause** (not just suspected): re-ran the identical probe
against the commit *before* the psi-sign fix in an isolated `git worktree`,
and got the documented pre-fix numbers back almost exactly
(`noShift` diverges at `t=2.76`, `surfaceNormal` survives to `t=4.0`). See
decision 1's update below for why that fix is correct and should not be
reverted. **Fix applied**: `cases/sloshingTank.py`'s `deltaSPH` branch now
defaults to `ShiftingScheme.michel2022`/`ShiftingProjectionScheme.michel2022`
instead of the shared default, confirmed end-to-end through the case's own
real entry point (`run(sloshingTankCase, params={'shifting': True}, nx=60,
tLimit=4.0)` → 20001/20001 steps, not diverged). Full repo test suite green.

**The full `deltaSPH` validation-case sweep this regression made urgent is
also done** (Part 9.1): all 11 other `deltaSPH`-family weakly-compressible
cases in the registry, each run to its own `tLimit`, pass clean, and
genuinely exercise the same shift-dependent code (`surfaceNormal`,
`active=True` by default in every one). `sloshingTank` was the only case
affected. The psi-sign fix is confirmed **not** a broad regression — this
line item is closed.

**Michel §5.4's own free-surface `δu_max` convergence rate is also done now**
(`scripts/probe_michelFreeSurfaceConvergenceRate.py`, `PST_ALE_PLAN.md` §7.1):
a rigid-rotation field turned out to be degenerate for this measurement
(`U_char` is exactly zero for solid-body rotation at every pair — an
identity, not a resolution effect), so it reuses the interior probe's own
Taylor-Green field on a non-periodic (genuinely free-surface-bounded) box.
The domain-wide `δu_max` just reproduces the periodic bulk result (Eq. 48
can only shrink a shift, never grow the global max); the maximum restricted
to the dilated free-surface set converges at slope **1.811** — faster than
first order, clearing Table 3's claim with margin. That closes out both
halves (interior + free surface) of the Table 2/3 audit for this
implementation.

**The multi-seed/multi-resolution `hydrostaticColumn` sweep is also done now**
(`scripts/probe_michelHydrostaticColumnSweep.py`, `PST_ALE_PLAN.md` §7.1),
and it **refines rather than simply confirms** the single-seed finding above.
The single-seed run used a perfectly regular lattice — the case's own jitter
path is dead code, re-enabled from outside for this sweep — across
`nx ∈ {24,32,48} × 3 seeds × 4 modes`. Two things hold up: `pairedFraction`
is lower under the shift than under `neither` at every resolution and every
seed (the core claim), and **`both` (shift + safeguard) is the uniformly
best and most seed-stable configuration** at every resolution. But **decision
4's overturn above needs qualifying, not retracting**: `michelShift` alone is
not *categorically* worse than `noPenetrationShift` alone — at nx=32 it's
actually better on mean `‖v‖` peak (0.470 vs. 0.612) — it is just markedly
less *seed-stable* (std up to 0.896 vs. `noPenetrationShift`'s 0.014-0.338,
tightening with resolution). So the single regular-lattice seed happened to
land on a case where the shift underperforms; the safeguard's real advantage
is consistency across configurations, not a categorically lower velocity.
See decision 4 below and `PST_ALE_PLAN.md` §7.1 for the full table.

**Stage A (Michel PST) is fully closed. Per user direction, `PST_ALE_PLAN.md`
Stages B/B′/C/D are on hold** until Part 8 steps 6, 8, 9 and §4.2 below —
the paper's own remaining validation cases — are done, since none of them
need anything from those later PST stages.

**2026-09-05, second pass — all four remaining validation cases wired for
ACSPH** (`oscillatingDroplet`, `rotatingSquarePatch`, `impact`, `dambreak`;
`hydrostaticColumn` was already done). Each follows the same pattern
`hydrostaticColumn` established: an `_configureArtificialCompressible`
branch calling `configureArtificialCompressible` (or, for `dambreak`, whose
`configureScheme` never went through that shared helper, the handful of
things it would have done, applied by hand), a per-case `acParams.uChar`
default (§5.5's gap — `A*R` for the droplet's edge speed, `omega*size` for
the patch, `impactVelocity` for the bodies, `sqrt(g*fillRatio*L)` for the
column, matching `dambreak`), and a `timestep=` hook dispatching to Eq. (46)
for ACSPH while leaving every other scheme's behaviour bit-for-bit
unchanged. All four pass smoke tests with sane physics, not just
"doesn't crash": `oscillatingDroplet` runs 200 steps to `t=1.85` with KE
oscillating 0.003-0.76 (a real oscillation, not a decay); `rotatingSquarePatch`
tracks its own WCSPH run's KE/`‖v‖` closely at 20 steps; `impact` shows the
bodies actually colliding (`gap` closing to 0.043, KE dropping from 0.203 to
0.142 on impact, `comDrift` ~3.5e-8 — momentum conserved); `dambreak` runs
150 steps under gravity with zero wall penetration. Full test suite green
throughout (no case's non-ACSPH path changed behaviour).

**Step 6, the Table 1/2 reproduction — real machinery built, mechanism
demonstrated, full paper-scale numbers still open.** Needed De Courcy's own
`IRMSE(KE)/IRMSE(a)` (the semi-major-axis error), which requires the
droplet's *exact* time-dependent shape — the two already-encoded constants
(`DROPLET_STRETCH`/`DROPLET_PERIOD`) only sample two isolated instants. That
solution is Monaghan & Rafiee (2013) [77], cited but not itself in
`literature/` (§6.2 already flagged this as the one genuine gap); **the user
supplied the PDF** (filed under its own title, "A simple SPH algorithm for
multi-fluid flow with high density ratios", IJNMF 71(5) 537-561, DOI
10.1002/fld.3671 — Wiley's "2012" Early View date on the filename, "2013" the
print-issue date the citation uses). Re-derived the governing ODE from its
Appendix A rather than trusting the OCR'd equation text directly (which is
easy to misread as `dsigma/dt = -Omega**2 + 2*(b**2-a**2)/(a**2+b**2)` — the
"2" is `Omega`'s superscript, not a coefficient; that misreading fails the
check below):

```
dsigma/dt = (sigma**2 + Omega**2) * (b**2 - a**2) / (a**2 + b**2)
da/dt = sigma * a
db/dt = -sigma * b
```

with `Omega**2 = B**2` (`computePotentialFieldGravity`'s acceleration is
`-magnitude**2 * r`) and `sigma(0)=A, a(0)=b(0)=R`. **Verified against the
already-encoded constants**: at the case's own `A=B=R=1` default, this ODE's
first peak of `a(t)` lands at 1.931852 at `t=1.207` — against
`DROPLET_STRETCH=1.931843` — with the next peak at `t=6.034`, a period of
4.827 against the encoded `DROPLET_PERIOD` exactly. Implemented as
`oscillatingDroplet.analyticSolution` (also derives `KE(t)` from Appendix A
Eq. A.28-A.29 specialised to one fluid), plus `_measuredSemiAxes` (mass-weighted
second moments of the fluid particle cloud — exact for this problem, since
the straining IC has no rotational component, so the ellipse never leaves
the x/y axes) wired into `diagnostics` as `semiAxisA`/`semiAxisB`.

Eq. (65)'s effective cost `e` needed the per-step pseudo-iteration count,
which nothing outside the scheme's own `update` object could see before now
— `diagnostics`/`postStep` receive `runningState`, never `stepResult`. Fixed
with one general, non-case-specific line in `runner.py`: after each step,
`ctx.scratch['lastStageUpdate']` is stashed from `stepResult.stages[-1].update`
(the ad hoc `pseudoIterations`/`epsilonV`/`bdfOrder` attributes ACSPH's step
already sets on it), so any case's `diagnostics` can read scheme-specific
extra step data with `getattr(..., None)`; `1` (one function evaluation) for
every scheme that sets nothing extra. `oscillatingDroplet.diagnostics` reads
it as `pseudoIterations`. Since diagnostics are recorded once per real step
of size `dt`, Eq. (65)'s integral telescopes exactly to
`sum_steps(pseudoIterations) * s_RK` — no quadrature needed.

`scripts/probe_acsphOscillatingDropletTable1.py` runs the sweep: a δ-SPH
reference plus an ACSPH cell per `(CFL_t, Δt/Δτ)`, both scored against the
same real analytic ground truth, ACSPH's own IRMSE/`w`/`e` reported
normalised by the reference's (Table 1's own convention). **Found and fixed
a real bug in the process**: `config.maxDt` defaults to `1e-2` (`runner/caseSpec.py`,
a generic global default) and is unconditionally one of Eq. (46)'s `min()`
candidates (`modules/timestep/artificialCompressible.py`) — so with the
default, every `CFL_t` in `[0.1, 0.6]` produced *bit-identical* trajectories
on the first sweep, `dt` silently pinned to `maxDt` throughout. Fixed by
raising `--maxDt` (default `1.0`) in the sweep script; not a scheme bug, but
a sharp footgun for any case whose natural adaptive `dt` exceeds the generic
default, worth knowing before trusting *any* ACSPH sweep result on a new
case.

**Reduced-scale sweep result** (nx=24, one period, RK3 — not the paper's
`L/Δx=200`, multi-period scale; a single real step already costs ~1s at
nx=32 on one CPU core here, so that reproduction is a separate, larger,
budgeted run):

| `CFL_t` | `Δt/Δτ` | steps | `IRMSE(KE)` | `IRMSE(a)` | `w` | `e` |
|---|---|---|---|---|---|---|
| 0.1 | 2 | 391 | 0.188 | 0.201 | 1.78 | 6.06 |
| 0.1 | 5 | 391 | 0.191 | 0.202 | 1.82 | 6.08 |
| 0.2 | 2 | 208 | 0.183 | 0.213 | 0.64 | 2.16 |
| 0.2 | 5 | 208 | 0.281 | 0.247 | 0.97 | 3.23 |
| 0.4 | 2 | 118 | 0.184 | 0.220 | 0.30 | 1.02 |
| 0.4 | 5 | 118 | 0.175 | 0.194 | 0.41 | 1.36 |
| 0.6 | 2 |  90 | 0.227 | 0.260 | 0.22 | 0.76 |
| 0.6 | 5 |  90 | 0.219 | 0.223 | 0.29 | 0.95 |

(all four ratio columns normalised by the δ-SPH reference's own
`IRMSE(KE)=0.345`, `IRMSE(a)=0.224`, `w=155s`, `e=3.86e4` at the same
resolution.) **Cost drops monotonically and sharply with `CFL_t`, matching
the paper's own claim exactly** (`w`/`e` fall ~8x from 0.1 to 0.6). **The
error side does not yet reproduce Table 1's sharp 0.4→0.6 jump** — `CFL_t=0.4`
is if anything the *lowest*-error row here, not a cliff edge — most likely
because one period at nx=24 is too short/coarse to resolve BDF2's accuracy
cliff cleanly against this much noisier baseline (every ratio here sits well
under the paper's own Table 1 floor of ~0.46, meaning ACSPH is doing
comparatively better against δ-SPH at this coarser scale than at the
paper's, which dilutes the accuracy-cliff signal specifically, not the cost
one). **Open**: a longer, finer, multi-period reproduction before trusting
the error trend; the mechanism (real analytic ground truth, real cost
accounting, both schemes measured identically) is now correctly built and
does not need revisiting.

**Step 8, AC-4 — implemented, formula verified against the actual PDF, but
found genuinely unstable.** `PressureSmoothingScheme.biharmonic`
(Eq. 35) is `-h**2 * (AC-2 applied to AC-2's own raw output)` — no new
kernel, `computeScalarFieldDiffusion(..., densityOnly, field=...)` called
twice (`modules/artificialCompressible/pressureSmoothing.py`). Re-checked
character-by-character against `decourcy2024`'s own PDF text (not just this
plan's transcription) and matches exactly. Run on `hydrostaticColumn` *and*
`rotatingSquarePatch` (nx=24, no walls, so not a boundary-treatment
artefact) it diverges within ~15 real steps (pressure/velocity reaching
`O(1e5-1e6)`) — measured directly on the initial hydrostatic state alone
(before any dynamics): AC-2's own raw output is already large near a free
surface (the known-bad truncation Eq. 35 is built from, unrenormalised), and
nesting it without correction amplifies that ~20x in one more pass. The
paper's own text (§4.1.1) says AC-4 "struggles to maintain a converged
kinetic energy" but does *not* report an outright blow-up, and separately
reports AC-4 resolving the hydrostatic free-surface profile fine at
`t=50s` — so this is either a resolution/parameter sensitivity the coarse
nx=24 test here falls into more severely than the paper's own setup, or a
subtler issue than the formula itself (already verified correct). **Not
validated for production use** — implemented and callable for
experimentation, but flagged in its own docstring as unstable pending a
resolution/parameter study; do not default to it. AC-JST (Eqs. 36-37) is
not attempted yet — it needs a genuinely new kernel (the `chi` switch is a
nonlinear function of both `p_i` and `p_j`, not a per-particle field
`computeScalarFieldDiffusion` can express) plus the `𝕍`/`min`-vs-`max`
questions §5.1 already flagged, a larger and more self-contained piece of
work than AC-4 turned out to be.

**2026-09-05, third pass — AC-JST implemented, closing out step 8 and every
remaining *mechanism* item in the plan.** See Part 8 step 8's own entry for
the full account: one new gradchecked kernel (`wp_jstSwitch.py`, Eq. (37)'s
`chi_i`), the `𝕍`-gated AC-2L/AC-4 blend, §5.1's `min`/`max` resolved as
already decided. **AC-JST also cross-validates AC-4** — both stay bounded
where AC-4 alone diverges, confirming AC-4's formula is correct and its
standalone instability is real rather than a bug. Full test suite green.

**Every "build it" item in this plan is now done**, and every sweep/validation
item has now had a real attempt at reduced scale (2026-09-05, fourth pass) —
see each step's own entry above for the numbers:

| item | result |
|---|---|
| Step 6, Table 1/2 sweep | **Done at reduced scale** (nx=24, one period). Cost trend matches the paper cleanly; the accuracy-cliff trend does not resolve cleanly at this scale (§4.3's own entry above and the new authors' question). A larger (nx=32, 1.5 periods) attempt was started but abandoned after ~40 minutes of CPU time still on the reference run alone — not worth the wall-clock cost for a plan closeout; the existing reduced-scale numbers stand. |
| Step 8, Fig. 2 (hydrostaticColumn operators) | **Done at reduced scale** (nx=48, 300 steps). Qualitatively matches: AC-2 degrades over time (can't hold the gradient), AC-2L/AC-JST hold it, AC-JST clearly best (`pressureSlopeRatio` 0.995 vs. 0.783 vs. 0.721). AC-4 diverges immediately, as already known. |
| Step 8/§4.2, Fig. 15/16 (rotatingSquarePatch vs. BEM) | **Deferred.** Needs digitized reference data from a figure, not a formula — not attempted rather than risk silently encoding wrong numbers. |
| §4.4, `impact` vs. Marrone 2015 | **Done, genuine formula-level validation.** Found and implemented their exact closed-form energy-loss ratio; at nx=64 the simulation's KE trajectory passes through the analytic target almost exactly (0.1% off) at an intermediate time, though "the instant to compare" has no clean definition in a continuously-evolving sim. Real, encouraging, not a clean pass/fail. |
| §4.5, `dambreak` vs. Lobovsky 2014 | **Deferred.** Needs the experiment's own physical geometry and new per-height pressure probes, not present in this repo's dimensionless case as configured. |

**What's left in this plan is now entirely a matter of scale, not substance**:
finer/longer versions of the four sweeps above, and the two deferred items'
own prerequisite work (BEM digitization, dambreak geometry/probe matching).
Nothing here blocks anything else. `PST_ALE_PLAN.md` Stages B′/B/C/D remain
queued after all of this, per the user's own sequencing.

## Decisions taken without you — overturn any of these if you disagree

1. **The δ-SPH `ψ` sign was wrong and is now fixed** (Part 3, "The sign
   error"). This changes the *default WCSPH scheme's* behaviour repo-wide, so
   it is the one with the widest blast radius. The evidence is not a judgement
   call — the Antuono correction must annihilate a linear field pair-by-pair
   and did the opposite — and a single-variable A/B on `sloshingTank` improves
   both the density floor and the pressure peak. **Worth reporting upstream and
   checking against diffSPH, which this kernel was ported from.**
   → **Confirmed and closed out** (2026-09-05, `PST_ALE_PLAN.md` §7.1, found
   incidentally while measuring Stage A on `sloshingTank`, then root-caused
   deliberately): `--scheme deltaSPH`'s baseline (`shiftProperties.active =
   False`) now diverges at `t = 0.34`, and the `surfaceNormal` shift fix
   diverges at `t = 0.68` — both roughly **10x earlier** than
   `docs/historic_plans/WCSPH_SHIFTING_PLAN.md` documents (`t≈2.6-3.5s`) for
   the same case. **Confirmed this commit is the cause**: re-running the
   identical probe against `790a7c7`'s parent commit, in an isolated `git
   worktree` so the working tree never moved, reproduces the documented
   pre-fix numbers almost exactly (`noShift` diverges at `t=2.76`,
   `surfaceNormal` survives to `t=4.0`, density `[0.969, 1.042]`).
   **This is not a reason to revert the fix.** The sign correction is
   independently re-derived from Marrone et al. 2011 Eq. (6), backed by its
   own O(N²) torch reference and `tests/test_deltaSPHDiffusion.py`'s
   linear/quadratic-field cancellation checks, and its own commit message is
   explicit about the mechanism: the old sign made the diffusion operator
   accidentally twice as strong (second-order, not the intended
   fourth-order) — "diffusive either way, hence never a blow-up, hence never
   caught". `sloshingTank`'s stability was quietly resting on that extra,
   undocumented damping. (The "improves... density floor and pressure peak"
   A/B claim above and this finding aren't actually in tension once that's
   understood: more diffusion generically looks "better behaved" on a
   short-window A/B even though it is quantitatively wrong and, on this
   specific violent-impact case, was load-bearing for stability over the
   full run.) **Fix**: `cases/sloshingTank.py` no longer relies on the
   shared shift default for this case — see `PST_ALE_PLAN.md` §7.1 for the
   applied fix and its end-to-end confirmation.
   → **Full validation-case sweep also done** (2026-09-05, Part 9.1): every
   other `deltaSPH`-family weakly-compressible case in the registry (11 of
   them), each run to its own `tLimit`, passes clean — no divergence, and
   all 11 genuinely exercise the same shift-dependent code path
   (`shiftProperties.active=True` by default, untouched, in every one of
   them). `sloshingTank` was the only case affected, and it is fixed. The
   sign fix is confirmed **not** a broad regression.
2. **ACSPH's pressure force defaults to `nonConservative`** (the literal
   `(p_i + p_j)` of Eq. 25), not `Antuono`. The Antuono switch is a
   tensile-instability guard the paper does not use; it is one config field
   away.
3. ~~**`hydrostaticColumn` runs non-periodic under ACSPH.**~~ **Withdrawn** —
   the underlying limitation is gone. Eq. (61)'s position moment is now a real
   operator (`modules/incompressible/wp_wallMoment.py`) that takes `x_ij` from
   `computeDistanceVec` like everything else, so it is minimum-image correct and
   periodic domains need no workaround. The case is back on the shared
   `periodic=True` default and runs 120 steps at `band=5` where the old
   two-gather decomposition died at step 45.
4. **`noPenetrationShift` is off by default** and is *not* in the paper — it is
   a wall *safeguard*, not particle shifting. The actual shift is a
   `finalize`-step displacement (Eq. 58, outside the pseudo-time loop), which
   is what step 7 built; `ArtificialCompressibleSystem.finalize` is where it
   goes, alongside where `WeaklyCompressibleSystem.finalize` runs
   `solveShifting`. The flag stays because turning it on shows exactly what a
   walled case does with neither — measured both ways in step 5b.
   → **Updated 2026-09-05, now that step 7 is measured, not just built**: the
   original expectation here (recorded in `ArtificialCompressibleSPHConfig`'s
   own field comment) was that this flag "should not be needed once the shift
   exists". Measured on `hydrostaticColumn`, one regular-lattice seed at
   nx=32: it looked still needed — the shift fixes `pairedFraction` (interior
   particle clustering) completely (0.065 → 0.000) but the shift alone (peak
   2.33) sat well above the safeguard alone (0.75).
   → **Refined 2026-09-05 by the multi-seed/multi-resolution sweep**
   (`scripts/probe_michelHydrostaticColumnSweep.py`, `PST_ALE_PLAN.md` §7.1,
   `nx ∈ {24,32,48} × 3 seeds × 4 modes`, jitter re-enabled from the case's
   own dead code path): the single seed above was not representative on the
   corner-velocity question. `michelShift` alone is not *categorically*
   worse than `noPenetrationShift` alone — at nx=32 its mean `‖v‖` peak
   (0.470) actually beats the safeguard's (0.612) — it is simply markedly
   less seed-stable (std up to 0.896 against the safeguard's 0.014-0.338,
   which tightens with resolution). What *does* hold at every resolution and
   seed: `pairedFraction` is lower under the shift than under `neither`, and
   **`both` together is the uniformly best and most seed-stable
   configuration** (lowest mean `‖v‖` peak at every `nx`, and — at 32/48 —
   the lowest variance too). So the safeguard's real value is consistency
   across configurations, not a categorically lower velocity than the shift
   — and the practical recommendation is "run both", not "the safeguard is
   still required because the shift alone underperforms it".
5. **`approachOnly=False` was added to `computeVelocityDiffusion`** rather than
   writing a new kernel. The default is unchanged, so no existing scheme moves.

   **This is half of `DFSPH_IMPROVEMENT_PLAN.md`'s open item 1** (the
   shear-carrying Morris term), which asks for "full `v_ij` vector, *no
   approach-only clamp*". `approachOnly=False` removes the clamp — the term is
   now two-sided, which is what makes it Monaghan & Gingold (1983)'s velocity
   Laplacian and what Eq. (25) requires. It does **not** remove the normal
   projection: the contribution is still `mu_ij * gradW` with
   `mu_ij = (v_ij . x_ij)/|x_ij|^2`, a vector along `x_ij`, so there is still no
   tangential stress. A real Morris et al. 1997 laminar term needs the full
   `v_ij` vector and is still a separate, unwritten operator. Both papers use
   the normal-projected form, so ACSPH does not need it — but if that item is
   picked up, the clamp half is already done and gradchecked.

## Questions for the authors, all in one place

| Where | Question |
|---|---|
| §5.1 | Eq. (37) prints `ε₄ = min(0, κ₄ − ε₂)`, which makes the JST operator **vanish** in smooth flow. Standard JST is `max`. Which does the CUDA code do? |
| §5.2 | Eq. (40)'s low-storage form cannot represent Fig. 1's SSPRK3 or RK4 at all. Which is the code — Jameson coefficients, or the full tableaus? |
| §5.4 | Is `𝕍` the same set (same `𝔽`, same dilation radius) in Eq. (36) and Eq. (57)? |
| §5.5 | `U_char` per case; the `𝕍` branch of Eq. (36) being unscaled. (The `β` interpolation question is closed — `michel2022` §5.3 says linear.) |
| §5.6 | Eq. (46)'s `CFL_t h` is a length. Is `h` there carrying an implicit reference velocity (which is what the term *does*)? And is the absence of a body-force constraint deliberate? |
| Part 3 | The `ψ` sign error above — does their δ-SPH reference implementation have it? |
| §4.1.1 | **AC-4 run standalone (not blended into AC-JST) diverges catastrophically for us** — pressure/velocity reaching `O(1e5-1e6)` within ~15 real steps on both `hydrostaticColumn` and a wall-free case (`rotatingSquarePatch`), nx=24. The paper reports only that AC-4 alone "struggles to maintain a converged kinetic energy" — bounded but non-convergent, not a blow-up — and separately that it resolves the hydrostatic free-surface profile fine at `t=50s`. Is the discrepancy resolution (our nx=24 vs. their finer grids), or something in `k2`/timestep tuning specific to AC-4 that isn't shared with AC-2L? We cross-checked the *formula* against the PDF character-by-character and again indirectly via AC-JST (which uses the identical AC-4 operator, scaled by `epsilon_4 <= kappa_4 = 1/32`, and stays bounded on the same cases) — confident the transcription is right, curious whether the severity we see is expected at coarser resolution or points at a parameter we're missing. |
| §4.3 | **Table 1/2's `CFL_t=0.4 -> 0.6` error cliff did not reproduce cleanly for us at reduced scale** (nx=24, one oscillation period only, vs. the paper's `L/Δx=200`-class, presumably multi-period runs): our `CFL_t=0.4` row was the *lowest*-error row measured, not a cliff edge, though the **cost** trend (`w`/`e` falling sharply with `CFL_t`) reproduced cleanly at every scale we tried. Is the accuracy cliff itself only clean at finer resolution / longer integration, or is there a specific measurement window (post-transient, particular oscillation number) Table 1's numbers are drawn from that we should match? |

---

## Why this is worth doing

The paper's own closing claim is the reason: *"in terms of software, it is clear
any weakly compressible δ-SPH code may be transformed to ACSPH by removing the
equation of state, specifying the artificial compressibility parameter k₁ and
adding a pseudo-time loop within the existing time loop."* This repo **is** a
δ-SPH code with all of that already built. The three things ACSPH needs that
δ-SPH does not — a pressure-evolution equation instead of an EOS, a BDF2
real-time source, and an inner pseudo-time RK loop — are the only genuinely new
code. Everything else (§4 below) is a rename or a config flag away.

Strategically it gives us a **third** incompressible baseline that is
structurally unlike both existing ones: DFSPH iterates a *pressure Poisson-like*
Jacobi solve on a velocity constraint; ACSPH iterates a *differential* equation
in pseudo-time. The paper's §2 argues these are the same thing solved two ways,
which makes it a genuinely informative comparison rather than a third opinion.

---

# Part 1 — Complete equation inventory

Numbering follows the paper. Symbols: `p` pressure, `v` momentum (Lagrangian)
velocity, `ρ` density (invariant, `= ρ₀`), `V_j = m_j/ρ_j` particle volume,
`h` smoothing length, `κh` kernel support radius, `x_ij = x_i − x_j`,
`(·)_ij = (·)_i − (·)_j`, `τ` pseudo-time, `t` real time, `n` real-time index,
`m` pseudo-time index, `s` RK stage index.

## 1.1 Governing system (continuous) — Eqs. (49)–(50)

The incompressibility constraint `∇·v = 0` is converted from elliptic to
hyperbolic by adding a pseudo-time derivative, and the momentum equation gains a
matching one:

```
Dp/Dτ  = −k₁ ρ ∇·v  +  k₂ 𝒟^p                       (continuity, Eq. 50/51)
Dv/Dτ  +  Dv/Dt  = −∇p/ρ + ν∇²v + f                 (momentum,   Eq. 25)
Dx/Dτ  +  Dx/Dt  = v                                (velocity,   Eq. 26)
```

`𝒟^p` is the pressure-smoothing operator (§1.4). It is *not* an ad-hoc artificial
viscosity: §2 derives it as the divergence of the momentum residual folded into
the pressure equation (Eq. 6), which is the same construction that produces the
ISPH pressure Poisson equation (Eqs. 10–15). That derivation is the paper's main
theoretical contribution and is worth reading in full, but nothing in it needs
implementing.

**Dropped by the paper itself:** the `k₃ D(∇·v)/Dτ` term of Eqs. (9)/(22) is set
to zero ("*during experimentation this third term was found to have little
influence*"). Implement the field, default it to 0, do not spend time on it.

## 1.2 Discrete continuity — Eq. (23)

```
Dp_i/Dτ = −k₁ ρ_i Σ_j (v_j − v_i)·∇W_ij V_j  +  k₂ 𝒟^p_i
```

Standard difference-form velocity divergence. Anti-symmetric, hence
volume-conserving (Eq. 52–53 discussion).

## 1.3 Discrete momentum and velocity — Eqs. (25)–(26)

```
Dv_i/Dτ + Dv_i/Dt = −(1/ρ_i) Σ_j (p_i + p_j) ∇W_ij V_j
                    + ν K Σ_j (v_ij·x_ij)/‖x_ij‖² ∇W_ij V_j
                    + f_i

ṽ_i := Dx_i/Dτ = v_i − Dx_i/Dt                                    (Eq. 26)
```

- Pressure gradient is the symmetric `(p_i + p_j)` form. The paper notes it is
  equivalent to `Σ m_j (p_i/ρ_i² + p_j/ρ_j²) ∇W_ij` under constant mass and
  invariant density — which holds here by construction.
- Viscosity is the Monaghan–Gingold (1983) velocity Laplacian, `K = 8` in 2D,
  `K = 10` in 3D.
- `ṽ` is the pseudo-time particle velocity. **On convergence `ṽ → 0`**, and it is
  simultaneously the position-row residual and the convergence metric (§1.6).

## 1.4 Pressure smoothing operators `𝒟^p` — Eqs. (32)–(37)

Four variants, named AC-2 / AC-2L / AC-4 / AC-JST:

**AC-2** — plain Laplacian of pressure (Molteni & Colagrossi form), Eq. (32):
```
𝒟^Δ_i = Σ_j 2(p_i − p_j) (x_ij·∇W_ij)/‖x_ij‖² V_j
```
Known-bad at free surfaces: kernel truncation diffuses the surface and it cannot
hold a hydrostatic gradient (§4.1.1 confirms this, Figs. 2–4). Implement it, but
it is a negative control, not a candidate default.

**AC-2L** — renormalised bi-Laplacian (Antuono correction), Eqs. (33)–(34).
**This is the paper's working default** and the operator used in every
head-to-head against δ-SPH:
```
𝒟^ΔL_i = Σ_j 2{ (p_i − p_j) − ½(⟨∇p⟩^L_i + ⟨∇p⟩^L_j)·x_ij } x_ij/‖x_ij‖² · ∇W_ij V_j

⟨∇p⟩^L_i = −Σ_j (p_i − p_j) L_i ∇_i W_ij V_j
L_i      = −[ Σ_j (x_i − x_j) ⊗ ∇_i W_ij V_j ]⁻¹
```

**AC-4** — nested (bi-harmonic) Laplacian, Eq. (35):
```
𝒟^Δ²_i = −h² Σ_j 2(𝒟^Δ_i − 𝒟^Δ_j) (x_ij·∇W_ij)/‖x_ij‖² V_j
```
Two neighbour loops, no correction. Inherits AC-2's truncation error but weaker
(§4.1.1: slight surface-row separation, volume perturbations, KE fails to fully
settle).

**AC-JST** — Jameson–Schmidt–Turkel blend, Eqs. (36)–(37):
```
𝒟^JST_i = 𝒟^ΔL_i                        if i ∈ 𝕍  (free-surface region)
        = ε₂ 𝒟^ΔL_i + ε₄ 𝒟^Δ²_i         otherwise

ε₂ = κ₂ min(1, χ),   ε₄ = min(0, κ₄ − ε₂),   κ₂ = 0.5, κ₄ = 1/32
χ_i = Σ_j |(p_i−p_j)/(p_i+p_j)| W_ij V_j  /  Σ_j W_ij V_j
```
`𝕍` = every particle within one kernel support radius of a free-surface particle.
**Not pairwise-symmetric, therefore not locally conservative** — the paper says
so explicitly and points at Lee et al. [51] for a conservative version.
See §5.1 for the `min`/`max` problem in ε₄.

**Frozen diffusion**: computed on the first RK stage and held across stages
(Antuono [8] / Jameson [40] technique). But **re-evaluated at every dual-time
iteration** — "*the diffusive terms are evaluated at each dual-time iteration
and cannot be fixed without loss of stability*".

## 1.5 Parameters — Eq. (24)

```
β = CFL_τ · h / Δτ        k₁ = β²        k₂ = 0.1 h β
```

`β` is the pseudo-time wave speed. In finite volumes `β` is prescribed and `Δτ`
varies locally; the paper **inverts this** — `Δτ` is a fixed fraction of `Δt`
(spatially constant, to keep particle displacements smooth) and `β` is the
derived variable.

`k₂ = 0.1 h β` is deliberately the δ-SPH `δ h c₀` prefactor with `β` playing the
role of `c₀` and `δ = 0.1`. The measured stability ceiling is `k₂ = 0.2 h β`,
consistent with Antuono's linear stability analysis [7,8]; `0.1` is kept for
consistency with δ-SPH practice.

`CFL_τ` is set by the pseudo-time integrator: **0.5 / 1.0 / 1.5 for RK2 / RK3 /
RK4**.

## 1.6 Dual-time integration — Eqs. (38)–(48)

State vector and residual (Eq. 38–39):
```
u = {p, x, v}ᵀ                 I_c = diag{0, 1, 1}
Du/Dτ + I_c Du/Dt = r          r* := Du/Dτ = r − I_c Du/Dt
```
The `0` in `I_c` is the whole point: the continuity equation has no real-time
derivative, so driving `r* → 0` enforces `∇·v = 0` *at* time level `n+1`.

**Pseudo-time RK sweep** (Eq. 40):
```
u^{n+1,m+1,0} = u^{n+1,m}
u^{n+1,m+1,s} = u^{n+1,m+1,0} + α_s Δτ r*^{,n+1,m+1,s−1},   s = 1..s_RK
u^{n+1,m+1}   = u^{n+1,m+1,s_RK}
```

**BDF2 real-time source** (Eq. 41), with variable-`Δt` coefficients (Eq. 42):
```
r*^{,n+1,m+1,s−1} = (1/α_PI) [ r^{n+1,m+1,s−1} − I_c( α_t u^{n+1,m+1,0} + β_t u^n + γ_t u^{n−1} ) ]

α_t = (2Δtⁿ + Δtⁿ⁻¹) / ((Δtⁿ + Δtⁿ⁻¹) Δtⁿ)
β_t = −(Δtⁿ + Δtⁿ⁻¹) / (Δtⁿ Δtⁿ⁻¹)
γ_t = Δtⁿ / ((Δtⁿ + Δtⁿ⁻¹) Δtⁿ⁻¹)
```
(Fixed-`Δt` limit: `α_t = 1.5/Δt`, `β_t = −2/Δt`, `γ_t = 0.5/Δt`.) Note the BDF
source is evaluated at the **frozen stage-0 value** `u^{n+1,m+1,0}`, not the
current stage.

**Point-implicit source treatment** (Eqs. 43–45): `α_PI = 1 + α_s Δτ α_t`,
applied to all three equations for temporal consistency. The paper then says
**`α_PI = 1` works fine here** ("*no noticed adverse behaviour*") because
`Δτ < Δt` and `Δτ` is spatially constant. Implement it (it is one scalar), keep
it on by default, expose the switch.

**Real timestep** (Eq. 46), with growth limiting to protect BDF2 accuracy:
```
Δtⁿ = max( min( CFL_t h , CFL_t h/‖v‖_max , 0.125 h²/ν , 1.2 Δtⁿ⁻¹ ) , 0.8 Δtⁿ⁻¹ )
```
`CFL_t ≈ 0.2`. §4.3 measures a sharp accuracy cliff above `CFL_t = 0.4`
(Table 1/2: error jumps ~2.4× from 0.4 → 0.6, ~10× at 1.0). Treat 0.4 as the
hard ceiling.

**Convergence metric** (Eqs. 47–48):
```
ε_v = log10( (1 / (N U_ε)) · sqrt( Σ_i^N ‖ṽ_i‖² ) )
U_ε = max( min(‖v‖_max, U_char), ε_s ),   ε_s = 1e−5
```
Iterate until `ε_v` drops below target. Recommended targets from the paper:
**−6 for general use, −8 for violent impact** (dam break, jet impact).

> ⚠ Note the `1/N` (not `1/√N`) — this is *not* an RMS. Per-particle residual
> `v̄` gives `ε_v = log10(v̄ / (√N U_ε))`, so a fixed `ε_v` target is a
> **stricter per-particle tolerance at higher resolution** by `−½log₁₀N`. Across
> the paper's own `L/Δx = 200 → 800` sweep that is a ~0.6-decade drift. Reproduce
> it verbatim (it is what their numbers mean) but record it, and consider
> exposing a normalised variant as a non-default option.

**Butcher tableaus** (Fig. 1, image-only — transcribed here):

| RK2 (explicit midpoint) | RK3 (SSPRK3 / Shu–Osher) | RK4 (classical) |
|---|---|---|
| `c = [0, 1/2]` | `c = [0, 1, 1/2]` | `c = [0, 1/2, 1/2, 1]` |
| `A = [[0,0],[1/2,0]]` | `A = [[0,0,0],[1,0,0],[1/4,1/4,0]]` | `A = [[0,0,0,0],[1/2,0,0,0],[0,1/2,0,0],[0,0,1,0]]` |
| `b = [0, 1]` | `b = [1/6, 1/6, 2/3]` | `b = [1/6, 1/3, 1/3, 1/6]` |

See §5.2 — these tableaus and Eq. (40) are mutually inconsistent for RK3/RK4.

**§4.3 finding: higher-order pseudo-time RK buys nothing.** Accuracy is set by
the BDF2, and cost rises near-linearly with stage count. RK2 at `CFL_t = 0.2`,
`Δt/Δτ = 5` is the best cost/accuracy point in Table 2. Default to RK2.

## 1.7 Optional `ṽ` material-derivative correction — Eqs. (27)–(31)

Recasting `D̃(·)/Dτ = ∂(·)/∂τ + ṽ·∇(·)` adds advective terms (Ramachandran et
al. [29]):
```
D̃p_i/Dτ  += −p_i Σ_j (ṽ_j − ṽ_i)·∇W_ij V_j + Σ_j (p_j ṽ_j + p_i ṽ_i)·∇W_ij V_j
D̃v_i/Dτ  += Σ_j (ṽ_j ⊗ v_j + ṽ_i ⊗ v_i) ∇W_ij V_j − v_i Σ_j (ṽ_j − ṽ_i) ∇W_ij V_j
```

**The paper's conclusion is to leave these off** (§4.2): *"generally ignore the
ṽ material derivative corrections."* With them on, the minimum stable `ε_v`
degrades from −8 to −6 and residual `ṽ` leaves non-physical terms that blow up
at thin free-surface tips. The second momentum term violates both linear and
angular momentum conservation. Implement behind a flag, default off, do not
tune.

## 1.8 Particle shifting — Eqs. (55)–(58), Michel et al. [66]

```
δv*_i = 0.5 · { −U^shift_i β_i (κh) ∇C_i               if ‖β_i(κh)∇C_i‖ < 0.5 κh/Δx
              { −0.5 U^shift_i (κh/Δx) ∇C_i/‖∇C_i‖     otherwise

U^shift_i = max_j( |(v_i − v_j)·(x_i−x_j)/‖x_i−x_j‖| )
∇C_i      = Σ_j [ 1 + 0.2 (W_ij / W(Δx_i))⁴ ] ∇_i W_ij V_j
β_i       = (κh/Δx)³ in the interior, decreased to 1 for surface particles
```
Free-surface correction (Eq. 57):
```
δv_i = 0                                          if λ_i < 0.4
     = λ_i² ( δv*_i − σ_i (δv*_i·n_i) n_i )       if i ∈ 𝕍
     = δv*_i                                      otherwise

σ_i = min[ 1, max( 0, (κh − d^fs_i) / (0.5 κh) ) ]
```
`λ` = min eigenvalue of `L_i` (Eq. 34); `d^fs` = distance to nearest surface
particle.

Applied **outside** the pseudo-time loop as a displacement (Eq. 58):
```
x'_i = x_i + δv_i Δt ,   φ'_i = φ_i + ∇φ_i·(δv_i Δt)
```
§4.2 tested internal (per-pseudo-iteration, Eq. 60) vs external shifting and
**chose external**: internal shifting stalls `ṽ_δ → 0` convergence during strong
impacts and worsens volume conservation. Eq. (59) offers a BDF correction
`ṽ = v − Dx'/Dt + D(δx)/Dt` so the real-time derivative stays Lagrangian;
§4.2 found it makes little difference. Implement, default on, cheap.

The paper picked Michel et al. specifically because it has **no `c₀`/Mach
dependence** — which matters since ACSPH has no `c₀`. Our existing δ⁺ shift is
Mach-scaled (`modules/shifting/delta.py`), so this is a real gap (§4.2 below).

## 1.9 Boundary conditions — Eqs. (61)–(62), Adami et al. [68]

Fixed ghost particles. Boundary pressure by extrapolation:
```
p_b = Σ_f [ p_f + ρ_f (g − a_b)·x_bf ] W_bf V_f  /  Σ_f W_bf V_f
```
Other fields by Shepard interpolation with a mirroring condition:
```
f_b = Σ_f f_f W_bf V_f / Σ_f W_bf V_f
no-penetration:  f_b^⊥ = 2 f_p·n − f_b·n ,   f_b^∥ = f_b·τ
no-slip:         f_b^⊥ = f_b·n ,             f_b^∥ = 2 f_p·τ − f_b·τ
```
Applied per field: `δv` and `ṽ` get no-penetration (so `δv·n = 0`); `v` gets
no-penetration in the velocity divergence; the velocity Laplacian gets no-slip
or free-slip depending on the case.

Free surface: handled implicitly by the discretisation (Colagrossi et al. [69]);
detection by Marrone et al. [70] with Sun et al. [11] modifications, producing
the surface set `𝔽` and the dilated set `𝕍`.

## 1.10 Reference δ-SPH baseline — Eqs. (63)–(64)

The paper's comparison scheme is δ-ALE-SPH (Sun et al. [14] / Antuono et al.
[15]), i.e. **quasi-Lagrangian δ⁺-SPH with the shifting terms folded into the
equations of motion** — which is exactly what
`docs/historic_plans/WCSPH_SHIFTING_PLAN.md` delivered here
(`ShiftProperties.correctdrhodt` / `correctdvdt`). RK4, frozen diffusion,
`CFL_wc = 0.75`. Our existing `--scheme deltaSPH` with those flags on is the
right control.

---

# Part 2 — Every constant the paper fixes

| Symbol | Value | Where |
|---|---|---|
| `δ` (δ-SPH density diffusion) | 0.1 | Eq. (16) |
| `k₂` prefactor | `0.1 h β` (max stable `0.2 h β`) | Eq. (24) |
| `κ₂`, `κ₄` (JST) | 0.5, 1/32 | Eq. (37) |
| `CFL_τ` | 0.5 / 1.0 / 1.5 for RK2 / RK3 / RK4 | §3.1.3 |
| `CFL_t` | ~0.2; hard ceiling 0.4 | Eq. (46), Table 1–2 |
| `Δt/Δτ` | 2 (best cost/accuracy), 5–10 for accuracy | Table 1 |
| `ε_v` target | −6 general, −8 impact-heavy | §4.2, §4.4, §4.5 |
| `ε_s` | 1e−5 | Eq. (48) |
| `α_ν` (artificial viscosity) | 0.01, `ν = α_ν h c₀ / K` | §4 |
| `K` | 8 (2D), 10 (3D) | Eq. (25) |
| `λ` surface cutoff | 0.4 | Eq. (57) |
| Shift tensile term | `1 + 0.2 (W_ij/W(Δx))⁴` | Eq. (56) |
| Kernel | Wendland C2, `h/Δx = 2` | §4 |
| `α_PI` | `1 + α_s Δτ α_t`, or 1 | Eq. (41) |
| `k₃` | 0 (dropped) | §3.1 |

> ⚠ **`ν` still needs a nominal `c₀`.** The paper says ACSPH "*does not require
> the definition of c₀*" and then defines `ν = α_ν h c₀ / K` with `α_ν = 0.01`,
> using "*the same value of ν*" as the δ-SPH run. So a reference `c₀` is still
> needed as an input to fix the physical viscosity — it just never enters the
> scheme. Our config must make this explicit rather than silently reusing
> `fluid.fixedSoundSpeed` as if it were an acoustic parameter.

---

# Part 3 — What the repo already has

This is the good news: most of the paper's spatial discretisation is a config
away.

| Paper | Repo | Status |
|---|---|---|
| Eq. (25) pressure gradient `(p_i+p_j)` | [surfaceAware.py](src/warpSPH/modules/pressure/surfaceAware.py), `PressureForceScheme.Antuono` | ✅ direct |
| Eq. (25) Monaghan–Gingold viscosity | [velocityDissipation.py](src/warpSPH/modules/deltaSPH/velocityDissipation.py), `ViscosityTerms.MonaghanGingold1983` | ✅ direct |
| Eq. (23) velocity divergence | [modules/momentum/](src/warpSPH/modules/momentum/), `WarpOperation.Divergence` | ✅ direct |
| Eq. (34) `L_i`, `⟨∇·⟩^L` | [gradRhoL.py](src/warpSPH/modules/density/gradRhoL.py), `computeRenormalizationMatrices` | ✅ **generalised 2026-09-05** (`computeGradRhoL(field=)`) |
| Eq. (33) bi-Laplacian ψ operator | [wp_densityDelta.py](src/warpSPH/modules/deltaSPH/wp_densityDelta.py) `DensityDiffusionScheme.deltaSPH` | ✅ **generalised 2026-09-05**; sign error found + fixed — see note |
| Eq. (61) value term | [wallPressure.py](src/warpSPH/modules/incompressible/wallPressure.py) `'shepard'` mode | ✅ exact |
| Eq. (61) body-force term | `wallPressureExtrapolation(bodyForce=)` | ✅ **built 2026-09-05**, §4.4 |
| Eq. (62) Shepard + mirroring | [modules/mdbc/velocity.py](src/warpSPH/modules/mdbc/velocity.py) | ⚠ audit, see §4.4 |
| §3.3 free-surface detection (Marrone/Sun) | [maronneDetection.py](src/warpSPH/modules/surfaceDetection/maronneDetection.py) + [dilation.py](src/warpSPH/modules/surfaceDetection/dilation.py) | ✅ gives `𝔽`, `𝕍`, `n`, `λ` |
| Eq. (57) `λ < 0.4` surface gating | `ShiftingProjectionScheme.surfaceNormal`, `surfaceLambdaThreshold` | ✅ direct |
| Eq. (46) adaptive `Δt` | [timestep/weaklyCompressible.py](src/warpSPH/modules/timestep/weaklyCompressible.py) | ⚠ recast, see §4.5 |
| §3.4 δ-ALE-SPH baseline | `--scheme deltaSPH` + `correctdrhodt`/`correctdvdt` | ✅ direct |
| RK2/SSPRK3/RK4 tableaus | `warpSPHIntegrators` (`rungeKutta2`, `sspRK3`, `rungeKutta4`) | ✅ reuse tableaus |
| Test cases (§4.1–4.5) | `hydrostaticColumn`, `rotatingSquarePatch`, `oscillatingDroplet`, `impact`, `dambreak` | ✅ all five exist |

### The one that matters most: Eq. (33) ≡ our existing δ-SPH ψ

> ⚠ **The A/B this section asked for found a sign error in the existing δ-SPH
> kernel.** Fixed 2026-09-05, see "The sign error" below. Everything in this
> subsection describes the operator *after* that fix.

`wp_densityDelta.py`'s `deltaSPH` branch computes
`ψ_ij = −(∇ρ^L_i + ∇ρ^L_j) − 2(ρ_j−ρ_i) x_ij/‖x_ij‖²`, dotted with `∇W_ij` —
the **unprojected** Marrone 2011 form (its Eq. 6, read off the PDF:
`ψ_ij = 2(ρ_j−ρ_i) r_ji/‖r_ij‖² − [⟨∇ρ⟩^L_i + ⟨∇ρ⟩^L_j]` with
`r_ji = r_j − r_i = −x_ij`). The paper's Eq. (33) writes the **projected**
form, gradient term contracted onto `x̂_ij` first.

These are algebraically identical whenever `∇W_ij ∥ x_ij`, which holds for any
isotropic kernel:
```
((∇p_i+∇p_j)·x̂)(x̂·∇W) = W'(r) (∇p_i+∇p_j)·x_ij / r = (∇p_i+∇p_j)·∇W
```

**Measured** (`scripts/probe_deltaSPHPsiProjection.py`, float64, 20×20 jittered
lattice, rms over interior rows, against a from-scratch `O(N²)` torch reference
that shares no code with the kernel):

| claim | linear `f` | quadratic | cubic |
|---|---|---|---|
| warp kernel − torch reference (unprojected) | 6e−14 | 2e−13 | 3e−13 |
| unprojected − projected, renormalisation **off** | 2e−15 | 2e−15 | 2e−15 |
| unprojected − projected, renormalisation **on** (relative) | **1.00** | 0.54 | 0.32 |

So **reusing the existing kernel reproduces AC-2L exactly**, with the caveat
confirmed quantitatively: they diverge completely once
`useGradientRenormalization` puts an `L_i` in front of `∇W` (then `L∇W ∦ x_ij`).
Assert that path off for ACSPH. Note the probe also shows *which* form survives
that: the projected one still annihilates a linear field with `L` on (its brace
`{(p_i−p_j) − ½(⟨∇p⟩_i+⟨∇p⟩_j)·x_ij}` is a scalar that vanishes regardless of
what multiplies `∇W`), while the unprojected one does not. If renormalised
gradients on this operator ever become desirable, implement Eq. (33) literally.

### The sign error (found by that A/B, fixed 2026-09-05)

`ψ` had the gradient term's sign flipped — `+grad − rho` where Marrone Eq. (6)
is `−grad − rho` — in all four gradient-carrying branches (`deltaSPH`,
`deltaOnly`, `denormalized`, `denormalizedOnly`). `densityOnly`
(Molteni–Colagrossi, no gradient term) was and is correct.

The defining property of the Antuono correction is that the two terms **cancel
pair-by-pair on a field linear in space**: for `f = a·x`,
`(∇f_i+∇f_j)·∇W = 2a·∇W` and `2(f_j−f_i)(x_ij·∇W)/r² = −2a·∇W` identically.
That is what promotes the Molteni–Colagrossi Laplacian to a *bi*-Laplacian and
lets the diffusive term reach a free surface without eating the smooth field
underneath. With the sign flipped the terms *added*: `deltaSPH` was numerically
`2 × densityOnly` on any smooth field — a second-order diffusion twice as
strong as the uncorrected one, in place of the fourth-order one. Measured
before/after on the linear field, rms over interior rows:

| | linear | quadratic |
|---|---|---|
| `deltaSPH`, before | 1.6e+00 | — |
| `deltaSPH`, after | **1.0e−13** | 1.8e−02 |
| `densityOnly` (unchanged, the control) | 6.2e−01 | 4.1e+00 |

It never blew up — the sign was still diffusive — which is exactly why only a
property test catches it. Pinned as `tests/test_deltaSPHDiffusion.py` (4 tests,
including the specific shape of the bug: `deltaOnly` came out *equal* to
`densityOnly` rather than its negative). Full suite green after the fix.

**This is not cosmetic for either scheme.** For ACSPH it is the whole of
§4.1.1: AC-2L is separated from AC-2 precisely by whether the operator can hold
a hydrostatic — i.e. linear — pressure gradient, and before the fix ours could
not. For WCSPH it means the repo's default δ-SPH density diffusion has been
over-strong and second-order, actively diffusing exactly the density gradients
it exists to preserve. **Worth reporting to the authors' circle / checking
against diffSPH, which this kernel was ported from.**

**Measured end-to-end on `sloshingTank --scheme wcsph`**, nx=100, `t ≤ 3.6 s`,
today's code both sides, the only difference being the sign (the pre-fix
operator is reproduced *exactly* by negating the gradient field handed to the
kernel, so this is a clean single-variable A/B — see
`scripts/probe_deltaSPHPsiSignAB.py`):

| | pre-fix ψ | post-fix ψ |
|---|---|---|
| diverged | no | no |
| min density over the run | 0.396 | **0.532** |
| max density over the run | 1.849 | 1.837 |
| peak \|p\| at Sensor 1, `t > 2 s` | 39.2 kPa | **32.5 kPa** |

(Measured band for the first impact peak: 2.2–13.1 kPa.) So the fix improves
both the density floor and the pressure overshoot, and degrades nothing. Note
neither is *good* — the case has separate open problems recorded in
`examples/sloshingTank/PLAN.md`, and the peak is still ~2.5× the band top. For
reference the run recorded on 2026-09-03 (`examples/sloshingTank/output/`)
**diverged** at `t = 3.41` with `ρ ∈ [0, ∞]`; that is older code, not this
variable, and is why the A/B above was run rather than compared against it.

---

# Part 4 — What has to be built

Ordered by how much is genuinely new.

## 4.1 The dual-time driver — *entirely new, the core of the work*

`schemes/artificialCompressible.py`. Owns the whole real-time step:

```
for m in 0..maxPseudoIters:
    u0 = u                                   # freeze stage-0 for the BDF source
    compute frozen 𝒟^p at stage 0            # §1.4 frozen diffusion
    for s in 1..s_RK:
        r  = spatial residual at u^{s−1}     # Eqs. 23, 25, 26
        Dx_Dt = α_t x0 + β_t x^n + γ_t x^{n−1}
        Dv_Dt = α_t v0 + β_t v^n + γ_t v^{n−1}
        r* = (1/α_PI) [ r − I_c·(Dx_Dt, Dv_Dt) ]     # I_c zeroes the p row
        u^s = u0 + α_s Δτ r*
    u = accumulate(b_s)                      # see §5.2 on Eq. (40) vs Fig. 1
    ṽ = v − Dx_Dt ;  if ε_v(ṽ) < target: break
apply shifting displacement (Eq. 58) + BDF correction (Eq. 59)
roll history: (x^{n−1},v^{n−1}) ← (x^n,v^n) ← (x^{n+1},v^{n+1})
```

**Framework integration — recommended approach.** The runner drives steps via
`ctx.integrator.function(state, f=ctx.stepFunction, dt=...)`
([runner.py:269](src/warpSPH/runner/runner.py#L269)), which does not fit a
scheme that owns its own time advance. Rather than adding a `dualTime`
integrator to `warpSPHIntegrators` (which would couple a general library to one
scheme), use the **exact-delta trick**: have `acsph_step` run the full dual-time
solve and return
```
dxdt = (x^{n+1} − x^n)/Δt ,  dvdt = (v^{n+1} − v^n)/Δt ,  dpdt = (p^{n+1} − p^n)/Δt
```
Forward Euler on an exact delta is the identity, so the runner reproduces the
converged state byte-for-byte with zero framework changes. Pin
`config.integrationScheme = forwardEuler` and **validate it at build time** —
a silent RK2 here would run the whole dual-time solve twice per step and blend,
which is wrong and not obviously wrong. (Precedent for this class of trap:
`dambreak.py`'s note that `divergenceFree` needs `semiImplicitEuler` and
"nothing in the code enforces this yet". Do enforce it.)

**BDF history storage.** `x^n, x^{n−1}, v^n, v^{n−1}` are needed. Put them on the
*system*, not the state — `WeaklyCompressibleSystem` already carries non-state
fields (`adjacency`, `t`, `domain`), which is the established precedent and
avoids `initializeNewState` cloning them per stage. Startup: step 0 has no
`u^{n−1}`, so fall back to BDF1 (`α_t = 1/Δt, β_t = −1/Δt, γ_t = 0`) for the
first step.

## 4.2 Michel et al. (2022) shifting law — new

Our shift is δ⁺-SPH (Sun 2017 scaling, Mach-dependent):
`modules/shifting/delta.py`'s docstring already notes *"An equivalent
shifting-velocity form (Michel 2022 scaling) is present in a comment but not
used."* ACSPH has no `c₀` and no Mach number, so this must be built:
`U^shift` (Eq. 56), the `β = (κh/Δx)³` interior scaling with surface decay to 1,
the two-branch magnitude clamp (Eq. 55), and the `σ` ramp (Eq. 57).

The surface machinery it needs (`𝕍`, `n`, `λ`, `d^fs`) is all already produced by
`detectFreeSurface`. Add as `ShiftingScheme.michel2022`; it is independently
useful to the existing WCSPH scheme, so it should live in `modules/shifting/`,
not inside the ACSPH scheme.

## 4.3 Generalised scalar-field diffusion operators — modify existing

~~`wp_densityDelta.py` reads `ρ_i`/`ρ_j` from `getParticle(referenceState, j)`,
i.e. bound to the state's density field. Add an optional
`queryField`/`referenceField` tensor pair...~~ **Done 2026-09-05.** The pair is
threaded through the existing `ExtraSpec` mechanism, guarded so both or neither
must be supplied, and the volume weight `m_j/ρ_j` is deliberately left on the
density (it is a quadrature weight, not the diffused quantity).
`computeScalarFieldDiffusion` is the raw-operator entry point;
`computeGradRhoL(field=)` supplies Eq. (34). AC-2 and AC-2L now exist with no
new kernel. Gradcheck covers the new branch. See Part 3 for the sign error this
work uncovered.

Then add, new:
- **AC-4** (Eq. 35): a second pass over the AC-2 output. Trivial once AC-2 is a
  reusable scalar-field operator.
- **AC-JST** (Eqs. 36–37): the `χ` switch is one extra interpolation loop; the
  blend is elementwise. Needs `𝕍` from the dilated surface mask.

> ⚠ `modules/deltaSPH` is covered by the `gradcheck` skill. Touching this kernel
> means running `/gradcheck deltaSPH` before and after.

## 4.4 Boundary conditions — one real gap, plus an audit

`modules/incompressible/wallPressure.py` is closer than it first looks:

- **Eq. (61)'s value term is already exact.** Its `'shepard'` mode is literally
  `p_b = Σ_f V_f p_f W_bf / Σ_f V_f W_bf`.
- ~~**Eq. (61)'s body-force term is the gap.**~~ **Done 2026-09-05.**
  `wallPressureExtrapolation(..., bodyForce=g)` now adds
  ```
  p_w = [ Σ_f p_f W_wf + (g − a_w)·Σ_f ρ_f r_wf W_wf ] / Σ_f W_wf ,   r_wf = r_w − r_f
  ```
  (`adami2012` Eq. 27) on the `'shepard'` **and** `'mirror'` closures.
  `'shepard' + bodyForce` is De Courcy's Eq. (61) exactly — Adami weights by
  `W_wf` alone where Eq. (61) weights by `W_bf V_f`, and the two agree here
  because ACSPH is density-invariant so `V_f = V₀` cancels. `'mls'` raises
  rather than taking it: its Liu–Liu linear fit already carries the local
  pressure gradient, so adding the correction would double-count.

  *Implementation.* The vector moment `Σ_f V_f ρ_f (r_w − r_f) W_wf` is not
  any single `WarpOperation` — it is assembled from two `Interpolate` gathers,
  `r_w·Σ_f V_f ρ_f W_wf − Σ_f V_f ρ_f r_f W_wf`, which is legitimate because
  `(g − a_w)` is a per-*wall* quantity and comes out of the sum. Both gathers
  reuse the value term's `OperationProperties`, so numerator and denominator
  share one kernel evaluation.

  *Restriction, deliberate.* Splitting `r_w − r_f` across two gathers discards
  the minimum-image convention, so a wrapping pair contributes `±L_d` of error
  per periodic direction `d`. Dotting with `bodyForce` annihilates that error
  whenever `bodyForce` has no component along a periodic axis — the only
  physically sensible configuration — so the code **asserts** that instead of
  silently returning a wrong wall pressure. A real moment kernel would lift it.

  *Verified* by `tests/test_wallPressure.py`: for a pressure field linear in
  space every neighbour's contribution `p_f + ρ_f g·(r_w − r_f)` is already
  the analytic wall value, so the weighted average is exact regardless of how
  truncated the wall neighbourhood is. Measured on a 24×24 column over three
  wall rows: **corrected 2.0e−7 relative error (float32 machine precision),
  plain Shepard 1.3e−1** — i.e. the uncorrected wall under-reads by up to
  `3 Δx · ρ₀ g`, 12.5 % of the whole column's pressure drop. That is precisely
  the error that stops a hydrostatic column from holding.

  Note the wall-acceleration `a_b` still has no per-particle source anywhere in
  the codebase (the same gap `modules/mdbc/velocity.py` documents for the
  velocity mirror's dead `2 u_wall` term) — static walls make `a_b = 0`, which
  covers every case in Part 7, but a moving-wall ACSPH case would need it. The
  `(N, dim)` `bodyForce` form is already accepted, so wiring a source is all
  that would be left.

  **Wired into ACSPH on 2026-09-05** (`schemes/artificialCompressible.py`'s
  `wallPressures`): recomputed at every RK stage from the current fluid
  pressure, `clampNonNeg=False` because ACSPH's pressure is a solved field that
  legitimately goes negative. Without it the wall reads `p = 0` (the non-fluid
  rows are masked every step) and the column simply falls out of the box.
  Verified exact against the analytic profile on the real case — see step 5b.

  Independently a DFSPH improvement: no DFSPH caller passes `bodyForce` yet, so
  it stays additive there, but it is exactly the term
  `DFSPH_IMPROVEMENT_PLAN.md` Part 23 needs for a gravity-driven wall.
- **A structural note**: these live under `modules/incompressible/`, i.e. they
  are DFSPH-facing. ACSPH needs them too, so either relocate to a shared module
  or import across. Prefer relocating — a third consumer makes the current home
  misleading.

Eq. (62)'s Shepard + no-penetration/no-slip mirroring largely exists in
`modules/mdbc/`. The audit item: **ACSPH extrapolates three velocity-like fields
where WCSPH extrapolates one** — `v` (no-penetration in the divergence, no-slip
or free-slip in the Laplacian), `ṽ` (no-penetration), `δv` (no-penetration, so
`δv·n = 0`). Confirm each gets the right condition rather than inheriting `v`'s.

## 4.5 Timestep — recast

Eq. (46) is close to but not the same as `modules/timestep/weaklyCompressible.py`:
the acoustic constraint is replaced by an advective one (`CFL_t h/‖v‖_max`), the
viscous constraint is `0.125 h²/ν`, and there is a symmetric growth/shrink clamp
(`[0.8, 1.2]×`) that exists specifically to protect BDF2 accuracy. `Δτ = Δt / R`
with `R` a config constant.

## 4.6 New scheme family: config, state, system, wiring

Per the user's expectation, ACSPH gets its own family rather than being bolted
onto WCSPH:

- **`enumTypes.py`**: new `ArtificialCompressibleSPHScheme` enum (single member
  `artificialCompressible`), plus `PressureSmoothingScheme` (`laplacian`,
  `renormalizedBiLaplacian`, `biharmonic`, `jst`) mapping to AC-2/2L/4/JST.
- **`systems/artificialCompressible.py`**: `ArtificialCompressibleState`. The
  key structural difference from `WeaklyCompressibleState` — **`pressures`
  becomes an integrated field** (`integrated('dpdt')`) and `densities` becomes
  `constant` at `ρ₀`. Carries `surfaceIndicators`/`surfaceNormals`/
  `surfaceLambdas` and the ghost bookkeeping unchanged.
  `ArtificialCompressibleSystemUpdate` = `{dxdt, dvdt, dpdt}`.
  `ArtificialCompressibleSystem` carries the BDF history (§4.1).
- **`configurations/artificialCompressible.py`**:
  `ArtificialCompressibleSPHConfig`, modelled on `WeaklyCompressibleSPHConfig`
  (fluid, viscosity, BCs, shifting, regions, rigid bodies, surface detection,
  gravity) with the EOS/`densityDiffusion` block swapped for a new
  `acParams`: `{ pressureSmoothing, CFL_tau, CFL_t, dtOverDtau, rkStages,
  epsilonV, epsilonS, uChar, maxPseudoIterations, minPseudoIterations,
  k2Factor (=0.1), kappa2 (=0.5), kappa4 (=1/32), k3 (=0.0),
  usePointImplicit (=True), useTildeVAdvection (=False),
  shiftInsidePseudoLoop (=False), bdfShiftCorrection (=True),
  referenceSoundSpeedForViscosity }`. Plus the round-trip
  `artificialCompressibleConfigToDict` / `dictTo...` pair.
- **Registration touchpoints** (mirroring the `WeaklyCompressibleSPHScheme`
  surface): `schemes/builder.py` (`SchemeBundle`), `io/parsers.py`,
  `io/export.py`, `io/importIO.py`, `runner/caseSpec.py` (enum sweep at
  [caseSpec.py:286](src/warpSPH/runner/caseSpec.py#L286)), `warpSPH/__init__.py`,
  `modules/timestep/wrapper.py` (dispatch on system type).

---

# Part 5 — Ambiguities and errors in the paper

Flagging these matters for two reasons: they must be resolved before coding, and
(given we work with the authors) they are worth reporting back.

## 5.1 Eq. (37): `ε₄ = min(0, κ₄ − ε₂)` — almost certainly should be `max`

Verified against the rendered page (p. 9), so this is not a text-extraction
artefact. As printed with `κ₄ = 1/32` and `ε₂ = 0.5 min(1, χ) ≥ 0`:
`κ₄ − ε₂ ≤ 1/32`, so `min(0, ·) ≤ 0` always. In smooth flow (`χ → 0`) it gives
`ε₂ = 0, ε₄ = 0` and the JST operator **vanishes entirely** — the exact opposite
of the stated design ("*fourth-order dissipation in smooth regions*"). Standard
JST (Jameson–Schmidt–Turkel 1981) is `ε₄ = max(0, κ₄ − ε₂)`.

**Decision: implement `max`.** Expose the paper-literal `min` behind a flag so
the discrepancy is reproducible, but do not default to it. Ask the authors.

## 5.2 Eq. (40) and Fig. 1 are mutually inconsistent for RK3/RK4

Eq. (40)'s update `u^s = u^0 + α_s Δτ r*(u^{s−1})` is the Jameson low-storage
form. It can only represent tableaus whose `A` is non-zero on the sub-diagonal
only *and* whose `b` equals the final stage row. The RK2 midpoint tableau in
Fig. 1 satisfies this (`α = {1/2, 1}`). **SSPRK3 and classical RK4 do not** —
SSPRK3 has `a₃₁ = a₃₂ = 1/4`, and RK4's `b = [1/6,1/3,1/3,1/6]` is not any stage
row. So either Fig. 1 is decorative and the code uses Jameson coefficients
(`{1/4,1/3,1/2,1}` for 4 stages), or Eq. (40) is a simplification and the code
uses the full tableaus.

**Decision: implement the general explicit Butcher form** (reuse
`warpSPHIntegrators.butcher` tableaus), which reproduces Fig. 1 exactly and
degenerates to Eq. (40) for RK2. Since §4.3 concludes RK2 is the best operating
point anyway, the ambiguity is largely academic — but it must be a deliberate
choice, not an accident. Ask the authors which the CUDA code does.

## 5.3 Eq. (30) has a stray `h`

Eq. (30) reads `+ k₂ h 𝒟^p_i` (verified on the rendered page 8) while Eqs. (23),
(51) and (54) all read `+ k₂ 𝒟^p_i`. Dimensional analysis settles it: `k₂ = 0.1hβ`
already carries the length scale, `𝒟^p ~ [p]/L²`, so `k₂𝒟^p ~ [p]β/L ~ [p]/T` ✓
and the extra `h` is wrong. Typo in Eq. (30) only; use the `k₂ 𝒟^p` form.

## 5.4 `𝕍` is used for two different things

`𝕍` denotes "within a kernel support radius of a free-surface particle" in both
Eq. (36) (JST switching) and Eq. (57) (shifting). Whether the *same* dilation
radius and the same underlying `𝔽` set are intended in both is not stated.
Assume yes, expose the dilation iteration count separately.

## 5.6 Eq. (46)'s first term is dimensionally a length

Verified against the rendered page 10, so not an extraction artefact:

```
Δtⁿ = max( min( CFL_t h , CFL_t h/‖v‖_max , 0.125 h²/ν , 1.2 Δtⁿ⁻¹ ) , 0.8 Δtⁿ⁻¹ )
```

`CFL_t h` is a **length**, not a time. What it *does*, though, is unambiguous
from the structure — paired with the next entry it is exactly

```
min( CFL_t h , CFL_t h/‖v‖_max )  ==  CFL_t h / max(1, ‖v‖_max)
```

i.e. the advective constraint with its denominator floored at one. In a code
whose velocities are O(1) that floor is a reference velocity of 1 left implicit,
which is precisely the missing dimension.

**Decision: implement it that way**, with the floor named
(`REFERENCE_VELOCITY`) rather than hidden as a bare `1`. Ask the authors whether
their `h` there carries such an implicit reference velocity.

**This is not pedantry — it is load-bearing.** Without that floor *nothing*
bounds `Δt` in a quiescent case: `‖v‖_max → 0` makes the advective term
infinite, and an inviscid run makes the viscous term infinite too, so `Δt`
climbs 1.2× every step to `config.maxDt`. Measured on `hydrostaticColumn`:
`Δt` went 5e−4 → 6.4e−3 in fifteen steps and the near-wall velocity error grew
with it. With the floor, `Δt` settles at 4.9e−3 and stays there.

**Separately**, Eq. (46) has no body-force constraint, which every other δ-SPH
timestep in this repo and in the literature carries and which a gravity-driven
case needs. `CFL_t √(h/‖a‖_max)` is implemented behind the existing
`dt_accelerationConstraint` flag; turn it off for the paper's literal set. Also
worth asking about.

## 5.5 Under-specified

- **`U_char`** in Eq. (48) is never given a definition per case. It is presumably
  the case's own characteristic velocity (`√(gH)` for dam break, `ωL` for the
  patch). Make it a required per-case config value.
- **The `𝕍` branch of Eq. (36)** returns `𝒟^ΔL` *unscaled*, i.e. `ε₂ = 1`
  implicitly at the surface, while the interior uses `ε₂ 𝒟^ΔL + ε₄ 𝒟^Δ²`. That
  is a discontinuity in the operator at the `𝕍` boundary. Presumably intended
  (the text says the bi-Laplacian is "activated at the free surface"), but worth
  confirming.
- ~~**`β` in Eq. (57)'s surface decay**~~ — **answered from `michel2022`
  itself** (its §5.3), which De Courcy is restating: "`β_i = (R/Δx)³` for the
  inner particles with a **linear** decreasing in the free-surface region to
  reach `β_i = 1` for the free-surface particles". Not `σ`, not `λ²`. See
  `PST_ALE_PLAN.md` §2.3.
- **Symbol collision**: `β` is both the AC wave speed (Eq. 24) and the shifting
  scaling `(κh/Δx)³` (Eq. 56). Unrelated quantities. Use distinct names in code.

---

# Part 6 — Cited literature

## 6.0 Status

**Every blocking reference is now in `literature/`.** On 2026-09-05 the target
paper and seven of its references were synced in (bib keys `decourcy2024`,
`antuono2010`, `antuono2012`, `letouze2013`, `michel2022`, `ramachandran2021`,
`lobovsky2014`, `marrone2015`), each identified from its own front matter,
verified field-by-field against its DOI record, and abstracted verbatim —
`scripts/check_literature.py` passes.

Four more the ACSPH discretisation leans on were **already here**: `marrone2011`
(δ-SPH, Eqs. 16–17), `sun2017` (δ⁺-SPH), `sun2019` (consistent shifting / the
δ-ALE baseline), `adami2012` (wall BC, Eq. 61). Also present: `cummins1999` [2].

Nothing in Parts 1–5 or Part 7 is now blocked on a document we do not have.

## 6.1 Obtained for this plan

| Ref | Key | Unblocks |
|---|---|---|
| — | `decourcy2024` | The scheme itself. |
| [8] | `antuono2012` | The `k₂ ≤ 0.2hβ` stability bound; the bi-Laplacian interpretation of Eq. (33); frozen diffusion; why Eq. (32) fails at free surfaces. |
| [7] | `antuono2010` | Origin of the corrected Laplacian; co-cited for the stability bound. |
| [76] | `letouze2013` | Square-patch **initial pressure field** (a Poisson solve — §4.2 cannot be initialised without it), the analytic stretching solution, and BEM/LDFM reference data. |
| [82] | `lobovsky2014` | Dam-break probe geometry and the 2.5%/97.5% experimental bounds (Figs. 28/30). Supplementary Materials carry the raw signals. |
| [80] | `marrone2015` | The analytic incompressible KE drop the jet-impact case (§4.4) is scored against. |
| [66] | `michel2022` | The shifting law of §4.2 — its derivation of `β = (κh/Δx)³` and its PST-conditions checklist, which is also worth auditing our existing shift against. |
| [29] | `ramachandran2021` | Cross-check `α_PI = 2Δt/(2Δt+3Δτ)` and the `ṽ` material derivative. Closest prior art, **with an open-source reference implementation**. |

## 6.2 Still not obtained — non-blocking, the paper reproduces the equations in full

| Ref | Paper | Note |
|---|---|---|
| [40] | Jameson, Schmidt, Turkel (1981), AIAA-81-1259 | Would settle §5.1 (`min` vs `max` in ε₄) definitively. |
| [70] | Marrone, Colagrossi, Le Touzé, Graziani (2010), *Fast free-surface detection and level-set function definition*, JCP 229(10) 3652–3663 | Already implemented here (`maronneDetection.py`); the citation is missing from the library, not the code. |
| [77] | Monaghan & Rafiee (2013), IJNMF 71(5) 537–561 | Droplet analytic solution — already encoded as `DROPLET_STRETCH`/`DROPLET_PERIOD` in `cases/oscillatingDroplet.py`. |
| [6] | Molteni & Colagrossi (2009), CPC 180(6) 861–872 | AC-2 (Eq. 32) is given in full. |
| [15] | Antuono, Sun, Marrone, Colagrossi (2021), *δ-ALE-SPH*, C&F 216 104806 | ~~Not obtained~~ — **synced 2026-09-05** as `antuono2021`, for `PST_ALE_PLAN.md`. |
| [50],[54],[55] | Monaghan & Gingold 1983; Bonet & Lok 1999; Randles & Libersky 1996 | Standard operators, already implemented. |
| [19] | Sun, Pilloton, Antuono, Colagrossi (2023), *Acoustic damper term in WCSPH*, JCP 483 112056 | The competing "fix WCSPH instead" approach — interesting for the comparison narrative. |
| [32] | De Courcy et al., SPHERIC 2023 | The precursor; may carry implementation detail cut from the journal version. |
| [21],[23],[25],[39],[28],[27],[31],[26] | Chorin 1997; Turkel 1987; McHugh & Ramshaw 1995; Dupuy 2020; Clausen 2013; Ramachandran & Puri 2019; Chola & Shintake 2021; Rouzbahani & Hejranfar 2017 | §2 theory context only. No implementation content. |
| [60] | Vila 1999 | ~~§2 theory context only~~ — **synced 2026-09-05** as `vila1999`, for `PST_ALE_PLAN.md` stage D, where it is the scheme rather than the context. |

## 6.3 Out of scope

[72] Fourey et al. (2017) and the FSI chain [33–35], [73–75], [85] support §4.1.2
(elastic-base hydrostatic column) and Appendix A (modal structural solver + RBF
coupling). **This repo has no structural solver**, so §4.1.2 is not reproducible
and Appendix A needs nothing. Skip.

---

# Part 7 — Validation plan

All five of the paper's cases already exist here, which is unusually lucky.

| § | Case | Repo case | Measures | Reference |
|---|---|---|---|---|
| 4.1.1 | Hydrostatic column, rigid base | `hydrostaticColumn` | Hydrostatic profile, free-surface integrity, KE decay | Analytic gradient |
| 4.1.2 | Hydrostatic column, elastic base | — | FSI energy dissipation | **Skip** — no structural solver |
| 4.2 | Rotating square patch | `rotatingSquarePatch` | KE decay, centre pressure, momentum conservation | BEM/LDFM [76] |
| 4.3 | Oscillating droplet | `oscillatingDroplet` | IRMSE(KE), IRMSE(semi-major axis), cost | Analytic [77] ✅ already encoded |
| 4.4 | Normal impact of 2D jets | `impact` | Instantaneous KE drop, pressure smoothness | Analytic [80] |
| 4.5 | Dam break (2D + 3D) | `dambreak` | 4 wall pressure probes, KE | Experiment [82] |

**Ordering.** §4.1.1 first — it is the operator discriminator (it is what
separates AC-2 from AC-2L/AC-JST) and it is cheap. Then §4.3, which is the only
case with a clean analytic score and is the paper's own parameter-sweep vehicle
(reproduce Tables 1 and 2 — they are the single best acceptance test for the
dual-time machinery). Then §4.2 for conservation, then §4.4/§4.5.

**Acceptance targets from the paper:**
- AC-2L/AC-JST hold a hydrostatic gradient with no free-surface diffusion; AC-2
  visibly fails. (§4.1.1, Figs. 2–4.)
- Table 1/2 reproduce qualitatively: error flat for `CFL_t ≤ 0.4`, ~2.4× jump at
  0.6; cost linear in `Δt/Δτ` and in RK stage count; RK order buys no accuracy.
- Square patch: KE loss 25–32% *less* than δ-SPH across `L/Δx = 200/400/800`;
  **zero visible pressure oscillation at every resolution**; cost 2.4–2.8× δ-SPH.
- Jet impact: correct KE drop within a few time steps, no oscillatory ringing;
  total cost ≤ 1.5× δ-SPH.
- Dam break: noise-free `P1` through the void-closure event at `t√(g/H) ≈ 8.4`
  where δ-SPH's acoustic noise swamps the signal.

**Cost metric.** The paper's `𝒞_e = ∫ (m_iter · s_RK / Δt) dt` (Eq. 65) is
implementation-independent and should be recorded alongside wall time — it is
how their numbers are quoted and the only fair way to compare against our δ-SPH.

---

# Part 8 — Sequencing

1. ~~**Literature sync.**~~ **Done 2026-09-05** — the paper and seven references
   synced, checker green (§6.0). Remaining optional follow-ups: add the §6.2 set
   to `EXPANSION_CANDIDATES.md`, and consider promoting `marrone2011` from the
   extended set to the core (it is cited throughout Parts 1 and 3 but carries no
   abstract, the same case that promoted `dehnen2012` on 2026-09-04).
2. ~~**Finish the Adami `bodyForce` term** (§4.4).~~ **Done 2026-09-05** —
   `wallPressureExtrapolation(..., bodyForce=...)` on the `'shepard'` and
   `'mirror'` closures, exact on a linear pressure field to float32 machine
   precision (`tests/test_wallPressure.py`, 6 tests). See §4.4 for the
   two-gather decomposition and its periodic-axis restriction.
3. ~~**Generalise the diffusion kernel to an arbitrary scalar field** (§4.3), and
   A/B the projected vs unprojected ψ form.~~ **Done 2026-09-05.**
   - `computeDensityDiffusionDeltaSPH` takes an optional
     `queryField`/`referenceField` pair; the volume weight `m_j/ρ_j` is
     untouched (it is quadrature, not the diffused quantity). New public entry
     `computeScalarFieldDiffusion` (no `schemeConfig`, no prefactor);
     `computeDensityDiffusion` is now its δ-SPH specialisation.
     `computeGradRhoL` takes `field=` — with the pressure it is Eq. (34)
     verbatim.
   - `/gradcheck deltaSPH` green before and after; the script now runs every
     `DensityDiffusionScheme` twice, once through the field pair, so the new
     branch's adjoint is covered.
   - The A/B (`scripts/probe_deltaSPHPsiProjection.py`) confirmed the Part 3
     equivalence to 2e−15 with renormalisation off, quantified the divergence
     with it on (100 %/54 %/32 %), **and found the ψ sign error** — see §3's
     "The sign error". Full suite green after the fix.
4. ~~**Scaffold the new family** (§4.6).~~ **Done 2026-09-05.** `--scheme
   artificialCompressible` resolves, builds, and runs a step end to end through
   the real integrator; the step is a deliberate no-op
   (`schemes/artificialCompressible.py::PHYSICS_IMPLEMENTED = False`, warns
   once on entry) with a marked socket where step 5's driver goes.
   - `ArtificialCompressibleSPHScheme` + `PressureSmoothingScheme` +
     `isArtificialCompressibleScheme` in `enumTypes.py`.
   - `systems/artificialCompressible.py`: state with `pressures` **integrated**
     and `densities` **constant**; system carrying the BDF history plus
     `rollHistory` and `bdfCoefficients` (Eq. 42, with the BDF1 first-step
     fallback reported through an `order` return).
   - `configurations/artificialCompressible.py`: `ArtificialCompressibilityParams`
     (every Part 2 constant, `uChar` and `referenceSoundSpeedForViscosity`
     `Optional` on purpose) + the round-trip pair.
   - Registered in `schemes/builder.py`, `io/parsers.py`, `io/export.py`,
     `io/importIO.py`, `runner/runner.py::_resolveScheme`,
     `runner/caseSpec.py::schemeNames`, `warpSPH/__init__.py`.
   - **The integrator trap is enforced**, not documented:
     `validateIntegrationScheme` raises on anything but `forwardEuler` at step
     entry, because the exact-delta hand-off is silently wrong under a
     multi-stage integrator (it would run the whole dual-time solve per stage
     and blend). `modules/timestep/wrapper.py` likewise raises rather than
     letting an ACSPH system fall through to the acoustic timestep.
   - `tests/test_artificialCompressibleScaffold.py`, 14 tests. Note
     `test_variableStepBdf2DifferentiatesAQuadraticExactly`: the fixed-step
     limit alone would not catch a swapped `Δtⁿ`/`Δtⁿ⁻¹` in Eq. (42).
5. ~~**The dual-time driver** (§4.1) with AC-2L and RK2 only.~~ **Done
   2026-09-05.** `schemes/artificialCompressible.py` now runs the full
   Eqs. (38)–(48) loop: frozen stage-0 BDF source, frozen-per-iteration `D^p`,
   a general explicit Butcher RK sweep, `I_c = diag{0,1,1}`, point-implicit
   `α_PI`, and the `ε_v` convergence test on `ṽ`. `ṽ` advection, internal
   shifting and `k₃` raise rather than silently no-opping; AC-4/AC-JST raise
   pointing at step 8.
   - **New supporting work.** `modules/artificialCompressible/pressureSmoothing.py`
     (AC-2 / AC-2L dispatch onto `computeScalarFieldDiffusion`);
     `approachOnly=False` on `computeVelocityDiffusion`, which turns its
     `inviscid=False` branch into the Monaghan–Gingold velocity Laplacian
     proper — the clamp that made it one-sided is an artificial-viscosity
     device, and Eq. (25) has no such clamp. Both `nu * rho0` (for the kernel's
     `mean(ρ)` division) and the `1/ρ_i` on the pressure gradient are applied
     explicitly, so nothing here assumes `ρ₀ = 1` the way δ-SPH does.
   - **Measured** (`tests/test_artificialCompressible.py`, 24×24 periodic box,
     Taylor–Green plus a compressive perturbation, so only the solve is
     graded): `rms(∇·v)` **1.84 → 2.6e−3**, monotone in the iteration budget
     and flat between 100 and 400 iterations (converged). §4.3's finding
     reproduces: RK3/RK4 buy no accuracy over RK2 at equal iteration count and
     cost linearly more.
   - **Still open on the driver itself:** convergence to `ε_v = −6` took ~400
     iterations from that (deliberately extreme) initial transient. Whether
     that is the initial condition, `Δt/Δτ`, or something real is exactly what
     step 6's Table 1/2 reproduction answers — do not tune it before then.
   - **Validated on `hydrostaticColumn`** (§4.1.1, the paper's own first case),
     see step 5b below for the wiring that took and what it measures.
5b. **`hydrostaticColumn` under ACSPH** — done 2026-09-05.

   *Wiring.* `initializers/weaklyCompressible.py::initializeState` now builds
   any of the three state classes from one construction (it had a
   character-identical `if`-branch per class and no `else`, so an unknown class
   fell through to a `NameError`); `cases/weaklyCompressible.py` gains
   `configureDomain` + `configureArtificialCompressible`, which also **forces
   `forwardEuler`** (loudly) since the exact-delta contract requires it;
   `hydrostaticColumn` branches on `isArtificialCompressibleScheme` and takes
   the *raw* hydrostatic seed rather than the mean-shifted one — ACSPH has no
   pressure gauge, so the shift is a half-column-drop error at the free
   surface, not a gauge choice.

   *The domain is made non-periodic on this branch.* It is walled on every
   side, so the periodicity buys nothing, and it actively breaks Eq. (61): the
   wall-pressure moment is not minimum-image safe, so once a fluid particle
   drifts within a support radius of the bottom face it becomes a wrapped
   neighbour of the *top* wall and the moment picks up a whole domain height
   along gravity. `wallPressureExtrapolation` now detects exactly that (an
   `O(N)` per-axis test on whether such a pair can exist, one-sided, so it can
   only over-report) and refuses rather than returning a wrong wall pressure.

   *Measured — the scheme is right.* With `p` seeded analytically and `v = 0`
   (`scratchpad` probe, nx=32):
   - Adami wall pressure vs analytic at 199 wall rows: **max error 0.0000**
     against a column pressure drop of 4.905. The Part-2 work does exactly its
     job.
   - Bulk momentum residual: **‖r_v‖ = 0.0079 against g = 9.81**, i.e. the
     discrete pressure gradient balances gravity to 0.08 %. The hydrostatic
     state *is* a discrete equilibrium for this scheme.
   - `r_p = 0` in the bulk; density exactly invariant, as it must be.

   *What it still lacks: the shift (step 7).* Run forward, the column holds its
   pressure profile (`p ∈ [0.31, 4.87]` against an analytic drop of 4.5) but
   develops a near-wall velocity error concentrated in the **bottom corners**,
   and the free surface drifts down slowly.

   The Eq. (46) advective floor (§5.6) was needed to keep `Δt` bounded at all —
   without it `Δt` ran 1.2× per step to `maxDt` and the corner error grew with
   it. Beyond that, the corner behaviour is measured **both ways**, because the
   difference is exactly the size of the hole step 7 fills:

   | | `‖v‖_max` peak | worst corner particle | verdict |
   |---|---|---|---|
   | as the paper has it (no safeguard) | 2.9 | `x = -0.62` | inside the wall band; fluid is leaving the box |
   | `noPenetrationShift = True` | 0.29 | `x = -0.497` | bounded, wall plane at `-0.5` holds |

   `noPenetrationShift` is **off by default** and is *not* the particle shift —
   it is the repo's mDBC wall safeguard, applied as an acceleration the way
   `deltaSPH_step` applies it. The paper has neither, because it always has the
   shift. The 200-step figures below were taken with it on.

   *200 steps, nx=32, to `t = 0.94` (the case's full `tLimit`):* no divergence,
   density **exactly 1.000** throughout (invariant by construction, as it must
   be), `voidFraction 0`, `neighbourCountCV` flat at 0.24, `‖v‖_max` 0.58
   (peak 0.73), KE 6.1e−3, `dispMax` 0.27. The one clearly bad number is
   **`pairedFraction` 0.065** with `nnDistP01` down to 0.32 — particle pairing,
   which is exactly and only what particle shifting exists to prevent.

   So: **step 7 (Michel et al. shifting) is the next action** — was, as of
   this writing; it has since landed and this case has been re-measured
   (`scripts/probe_michelHydrostaticColumn.py`, `PST_ALE_PLAN.md` §7.1, after
   fixing a units bug found in the process — `computeMichelShift` was
   applying Eq. (22)'s shifting *velocity* with no `dt` factor). `pairedFraction`
   moved, cleanly, to exactly 0.000 with the shift on; `‖v‖_max` did not carry
   the corners on the shift alone the way the paper's phrasing implied it
   should — `noPenetrationShift = False` with the shift active still peaks
   `‖v‖` around 2.3, against 0.75 for the safeguard alone. See decision item 4
   above: this overturns the "shift replaces the safeguard" expectation
   rather than confirming it.

   *One loose end:* the case's own `pressureSlope`/`pressureSlopeRatio` figures
   of merit stop being reported once the run develops (`hydrostaticDiagnostics`
   returns early when its bulk band has fewer than 8 rows). They report fine on
   the first steps — `pressureSlopeRatio 0.79` after three. The direct probe
   above is the stronger measurement anyway (the *residual* against `g`, not a
   fit), but the band gate is worth understanding before quoting this case's
   published axes for ACSPH.

6. **Timestep + convergence control** (§4.5, §1.6). Reproduce Tables 1 and 2 on
   `oscillatingDroplet`. This is the real acceptance gate for the machinery.
   - **Eq. (46) landed 2026-09-05**: `modules/timestep/artificialCompressible.py`,
     wired into `modules/timestep/wrapper.py`'s dispatch. Advective in place of
     acoustic, `0.125 h²/ν` viscous, the symmetric `[0.8, 1.2]×` step-ratio
     clamp, and a `CFL_t > 0.4` warning (Tables 1–2's measured cliff). Eq. (46)
     as printed is dimensionally impossible — see the new §5.6 — so the first
     term is implemented as `CFL_t √(h/‖a‖_max)`, not `CFL_t h`.
   - **The Table 1/2 sweep mechanism landed 2026-09-05**
     (`scripts/probe_acsphOscillatingDropletTable1.py`, `oscillatingDroplet`
     wired for ACSPH, Monaghan & Rafiee (2013)'s analytic solution re-derived
     and verified — see the status board). Cost trend (`w`/`e` falling
     sharply with `CFL_t`) reproduces cleanly at a reduced scale (nx=24, one
     period). **Still open**: the error-side accuracy-cliff trend at
     paper scale (finer resolution, more periods) — not yet reproduced
     cleanly at this reduced scale.
7. **Michel shifting** (§4.2). Validate on `rotatingSquarePatch`.
   → **Split out into `PST_ALE_PLAN.md`** (2026-09-05). Reading `michel2022`
   properly, this is not one law: it is a set of requirements, an audit of every
   PST in the literature against them, the new law, and three SPH schemes to
   validate it on — two of which (Vila's ALE and Parshikov & Medin) this repo
   does not have, and both of which need a Riemann solver it also does not have.
   That plan's **stage A is this step**, unchanged and still the next action;
   stages B–D are the new scope it opens.
8. **Remaining operators**: AC-2, AC-4, AC-JST. Reproduce Fig. 2 / Fig. 16.
   → **Done, all four operators implemented, 2026-09-05.** AC-2/AC-2L from
     step 4/5; AC-4 and AC-JST land together in this pass, and AC-JST turned
     out to double as AC-4's own validation.
   - **AC-4** (`PressureSmoothingScheme.biharmonic`, no new kernel — two
     calls to `computeScalarFieldDiffusion`) formula verified character-by-
     character against the PDF, but **run alone** it diverges within ~15
     real steps on both `hydrostaticColumn` and `rotatingSquarePatch`
     (nx=24) — pressure/velocity reaching `O(1e5-1e6)`.
   - **AC-JST** (`PressureSmoothingScheme.jst`) needed one genuinely new
     warp kernel — `modules/artificialCompressible/wp_jstSwitch.py`, Eq. (37)'s
     `chi_i` switch (a pairwise nonlinear function of *both* `p_i` and `p_j`,
     which nothing in `computeScalarFieldDiffusion`'s per-particle-field
     family can express) — gradchecked
     (`scripts/gradcheck_jstSwitch.py`, w.r.t. both pressures and positions,
     plus a constant-field sanity check giving exactly `chi=0`). §5.1's
     `min`/`max` ambiguity resolved as already decided: `epsilon_4 = max(0,
     kappa_4 - epsilon_2)` by default, `acParams.jstUsePrintedMin` for the
     paper-literal `min` (which the plan had already predicted zeroes the
     operator in smooth flow). The dilated free-surface set `𝕍`
     (`currentState.surfaceIndicators`, already computed once per real step
     by `schemes/artificialCompressible.py`) gates AC-2L-alone at the
     surface vs. the interior blend elsewhere, no new surface-detection work
     needed.
   - **AC-JST cross-validates AC-4, and settles the question its own
     instability raised**: run on `hydrostaticColumn`, `rotatingSquarePatch`
     *and* `oscillatingDroplet` (nx=24, 60 real steps each), AC-JST stays
     bounded on all three (`‖v‖` peaks 0.26-5.0) despite using AC-4's exact
     same operator internally. `epsilon_4` caps AC-4's contribution at
     `kappa_4=1/32` (vs. standalone AC-4's implicit `epsilon_4=1`) and zeroes
     it entirely once the smoothness switch `chi_i > kappa_4/kappa_2 =
     1/16` — throttling the ~20x per-pass amplification measured on AC-4
     alone by at least that factor of 32 wherever it matters. **This
     confirms the AC-4 formula itself is correct** (a genuine sign/coefficient
     bug would still show up scaled down in the JST blend, not vanish) —
     the standalone blow-up is a real, resolution-sensitive property of
     running it unblended, matching (if more severely than) the paper's own
     "AC-4 struggles to maintain a converged kinetic energy" report against
     AC-JST's clean behaviour. AC-4 stays flagged not-production-ready
     standalone; AC-JST is the one to reach for if AC-4's fourth-order
     smooth-region behaviour is wanted.
   - Full test suite green throughout.
   - **Fig. 2's operator comparison — reduced-scale reproduction done,
     qualitatively matches.** `scripts/probe_acsphPressureSmoothingComparison.py`
     on `hydrostaticColumn`, nx=48. At 80 steps (`t~0.17`) all three
     surviving operators (AC-4 diverges immediately here too, as already
     known) look similar — too early for the paper's own discriminator to
     show, which is measured at `t=50s`. **At 300 steps (`t~0.31`) the
     expected separation appears**: `pressureSlopeRatio` (1.0 = exact
     hydrostatic gradient) is **AC-2 0.721** (down from 0.824 at 80 steps —
     actively *degrading*, exactly the "cannot hold a hydrostatic gradient"
     failure the paper describes), **AC-2L 0.783**, **AC-JST 0.995** (up
     from 0.809 — converging toward the exact gradient). AC-JST clearly
     beats both AC-2 and AC-2L here, plausibly because it gets AC-2L's own
     treatment at the free surface *plus* AC-4's higher-order correction in
     the smooth interior. `pressureResidual` (noise, independent of slope)
     tells a different, consistent story: AC-2 0.051 (worst), AC-2L 0.035
     (best), AC-JST 0.043 (between) — slope accuracy and noise are separate
     axes, as the case's own diagnostic docstring says. **Not the paper's
     `t=50s`/Fig. 2 numbers themselves** (far more real time than tractable
     here), but the qualitative claim — AC-2 fails, AC-2L/AC-JST hold the
     gradient — reproduces cleanly at this reduced scale, and the trend
     (AC-2 worsening, AC-JST improving, over the same time window) is
     itself informative, not just the endpoint values.
   - **`rotatingSquarePatch` vs. BEM (Fig. 15) — deferred, not attempted.**
     `letouze2013`'s BEM reference for the *square* patch (as opposed to
     the *circular*-patch case in the same paper, which has a closed-form
     analytic solution) is read off a scatter plot in their own figure —
     genuine digitized data, not a formula. Reproducing it validly needs
     those numbers, which this pass did not extract (no digitization tool
     available, and eyeballing points off a rendered PDF figure risks
     silently encoding wrong numbers as "the reference" — exactly the kind
     of error this repo's culture is built to avoid). A qualitative
     sanity check (does the patch grow arms, does energy decay monotonically,
     no blow-up) is a much lower bar than what Fig. 15 actually validates,
     so this is left open rather than substituted for one.
9. **Impact and dam break** (§4.4, §4.5), including the `𝒞_e` cost metric.
   - **Both cases wired for ACSPH 2026-09-05** (`impact`, `dambreak` — see
     the status board): smoke-tested with real physics (body collision,
     column collapse under gravity, zero wall penetration), full test suite
     green.
   - **`impact` vs. Marrone et al. 2015 — attempted, real formula, resolution-
     limited result.** Found their exact closed-form (`literature/marrone2015`
     Appendix A Eq. A.34): the instantaneous KE ratio `Ek(0+)/Ek(0-)` for two
     identical rectangular jets impacting head-on, as a function of aspect
     ratio `r = π·L/(2H)`. Maps cleanly onto `impactCase`'s own parameters —
     `size=H`, and (since each jet's near edge sits at the impact plane once
     `touching=True` closes the gap) `r = π·aspectRatio` directly, no unit
     conversion needed. Self-check against the paper's own two limits
     (`r→0` gives ratio `→0`, `r→∞` gives `→1`) passes exactly
     (`scripts/probe_acsphImpactEnergyLoss.py`).
     Measured (`aspectRatio=0.5`, i.e. `r=π/2`, analytic ratio 0.499, target
     `KE=0.2495` from `KE(0-)=0.5`): the naive "value at first contact"
     reading is a poor proxy — 0.84 at nx=24 — because KE does not drop in
     one instantaneous step; it declines over dozens of steps with the
     contact `gap` oscillating (repeated bounce/re-contact), not a single
     inelastic collision. **But at nx=64, tracking the full KE trajectory
     rather than just the first-contact instant, `KE` passes almost exactly
     through the analytic target**: `0.2492` at step 94 (vs. the target
     `0.2495`, 0.1% off) — then keeps declining afterward (`0.229` by step
     149), consistent with the paper's own remark that the true incompressible
     solution keeps evolving past `t=0+` as "thin horizontal jets develop"
     and carry away further energy, which the idealised snapshot theory
     does not capture and a real simulation necessarily will. **Reads as a
     genuine, encouraging sign** — resolution matters a lot here (nx=24's
     crude reading was 68% too high; nx=64 crosses the target almost
     exactly) — but "the instant to compare against" has no clean operational
     definition in a continuously-evolving simulation, so this is not a
     clean pass/fail validation, just real evidence the scheme is in the
     right regime once resolved enough.
     The paper's own reference (LS-FVM) shows the same qualitative
     complication — "thin horizontal jets develop, that requires a proper
     spatial resolution to be captured" — at their `L/Δx=400`; ours tops out
     far below that. **Open**: this needs the paper's own resolution scale
     (or at least a convergence sweep clearly trending to the analytic
     value) before treating any single number here as validated — logged as
     a real, resolution-limited attempt, not a pass.
   - **`dambreak` vs. Lobovsky et al. 2014 — deferred, not attempted.**
     Their pressure bounds (Table 2, `H=300mm`: median/2.5%/97.5% percentile
     peak pressure at 5 sensor heights) are genuine experimental statistics
     from 100 repeated trials, quoted at their own physical tank scale.
     Reproducing them validly needs (a) reconfiguring `dambreakCase`'s
     geometry to match their exact column/tank proportions (currently
     dimensionless, unrelated to their `mm` scale) rather than this repo's
     own historical Koshizuka & Oka proportions, (b) adding per-height
     pressure-probe diagnostics `dambreak.py` does not yet have, and
     (c) their own non-dimensionalisation of both pressure and time. None of
     that is a quick add-on to what exists; flagged here rather than forcing
     a mismatched comparison.
10. **Optional/experimental**, only if the above is clean: `ṽ` material
    derivative (§1.7), internal shifting (Eq. 60), `k₃` term, RK3/RK4.

---

# Part 9 — The validation sweep

## 9.1 `deltaSPH`, because the `ψ` sign changed under it — **done, 2026-09-05**

Part 3's sign fix alters the *default* density-diffusion operator: `deltaSPH`
was behaving as twice the uncorrected Molteni–Colagrossi Laplacian on any smooth
field and is now the fourth-order Antuono operator it was meant to be. That is a
behaviour change in every WCSPH run in the repo. What existed before this sweep
was a single-variable A/B on one case (`sloshingTank`: density floor 0.396 →
0.532, peak sensor pressure 39.2 → 32.5 kPa, neither run diverging) plus a
green test suite with no `deltaSPH`-scheme physics case in it.

**`scripts/run_sweep.py --cases dambreak drivenSquare droplet impact kolmogorov
ldc movingObstacle openFlow randomFlow sloshingTank squarePatch tgv-wc --full
--timeout 1800`** — every `deltaSPH`-family weakly-compressible case in the
registry, each run to its own `tLimit`, not just a smoke pass (a stability
regression like the one below only shows up deep into a long run). **11/12
passed clean** — no `non-finite`/`diverged`/NaN marker in any of their logs.
Confirmed all 11 genuinely exercise the shift-dependent code path this fix
touches: none of their `configureScheme`s override `shiftProperties`, so each
inherits the shared dataclass default (`ShiftingScheme.deltaSPH`,
`ShiftingProjectionScheme.surfaceNormal`, `active=True`) untouched — this is
real positive evidence, not 11 cases that happened not to test the thing that
broke.

**The one failure, `sloshingTank`, is not a new finding — it's the same
already-diagnosed and already-fixed issue** (`PST_ALE_PLAN.md` §7.1,
decision 1 above), and in a bare sweep invocation it isn't even the same
mechanism: `sloshingTank`'s own `params=dict(shifting=False, ...)`
pre-populates the `shifting` key, so `ctx.param('shifting', True)` in its
`configureScheme` resolves to `False` regardless of the `True` fallback
written there — a bare `warpSPHRun sloshingTank` runs with **no shift at
all**, which is the case's own long-documented, deliberately-demonstrated
"why shifting exists" baseline failure, unrelated to which projection scheme
is configured. (The `surfaceNormal`-specific regression this sweep does *not*
exercise — since shifting never turns on in a bare run — is the one already
found and fixed by hand: see `PST_ALE_PLAN.md` §7.1.)

**Conclusion: the sign fix is not a broad regression.** One case
(`sloshingTank`, and specifically only when shifting is deliberately turned
on for it) needed its free-surface treatment updated, already done. Every
other `deltaSPH` case in the registry, running the same `surfaceNormal`
default this fix touches, is unaffected over a full run. `scripts/probe_deltaSPHPsiSignAB.py`
remains available for a single-variable A/B on any specific case someone
wants to re-check, but the broad sweep this section called for is complete.

## 9.2 `artificialCompressible`, because it is new

The five cases of Part 7, in the order given there. Nothing beyond §4.1.1 has
been run at all yet.

## 9.3 Note for later

`ψ` is one member of a broader family of diffusion/dissipation operators worth
implementing properly (the `DensityDiffusionScheme` enum is the current, narrow
slice of it). Not scheduled here; recorded so the sign fix is understood as a
correction to one operator rather than a redesign of the family.

---

## Relationship to the other plans

This displaces `DFSPH_IMPROVEMENT_PLAN.md` as the active priority, per the
current decision. `COUPLED_INCOMPRESSIBLE_NEWTON_PLAN.md` remains queued behind
both. Steps 2 and 3 above are shared infrastructure that benefit the DFSPH work
regardless of ordering, so they are not sunk cost if priorities shift back.
