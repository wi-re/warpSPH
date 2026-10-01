# Open problems — known, hard, not part of routine closeout work

> **Progress marker:** this plan's row in [PLANS.md](PLANS.md). Update its *Last worked* date and *Where it stands* whenever you work on this plan.

This file is a deliberate dumping ground for exactly one kind of item: something
that has been traced to a real, understood mechanism, investigated seriously
(often across multiple sessions), and is **not fixed and not a quick fix**. It
exists so these items don't get silently re-absorbed into `BOUNDARY_DENSITY_PLAN.md`/
`DELTASPH_VALIDATION_PLAN.md`'s ever-growing history and re-discovered from
scratch every few sessions, and so a routine closeout pass (like
`WCSPH_DEFAULT_CLOSEOUT_PLAN.md`) has a clear place to point instead of
re-opening the investigation.

**If you're picking one of these up:** read the cross-referenced memory entry
and plan section first — each has a specific next-step already identified, not
just a description of the symptom.

**Lifecycle.** Short, self-contained problems without a plan of their own live
here too (a few lines each). When one is resolved, move its text to
[docs/historic_plans/RESOLVED_PROBLEMS.md](docs/historic_plans/RESOLVED_PROBLEMS.md)
and leave a one-line stub under the same number, so references stay valid.
Progress on this file is tracked by its single row in [PLANS.md](PLANS.md).

## 1. Corner-flyer / free-surface-pinning instability (the "flyers" and
"ceiling-sticking" phenomena)

> **2026-09-29:** the δ⁺-SPH ceiling-sticking half is now worked in
> [CEILING_STICKING_PLAN.md](CEILING_STICKING_PLAN.md) (Marrone 3.1, checkpoint forensics).

**What it is:** isolated fluid fragments near a solid corner or free-surface
edge — a few particles nearly disconnected from the bulk — undergo a
self-sustained, non-physical pressure oscillation ("numerical surface
tension" ringing from a truncated symmetric SPH pressure-pair sum at severely
under-supported kernel stencils), then collide with another fragment or the
bulk once geometry brings them back into mutual kernel support, producing a
sharp, violent kick. Manifests differently depending on where/how it's
triggered:

- **Marrone 3.4 (sharp-edged obstacle)**: scattered roof debris at re-impact,
  a bright floor-hugging band approaching the fillet, a sharp overturning-jet
  tip as the wave breaks, a persistent standing crest as the first jet crosses
  the obstacle apex. `DELTASPH_VALIDATION_PLAN.md` §8.18 + its roof-crest
  follow-up, and the "Next up" section right after it.
- **sloshingTank**: a particle pinned at the free surface near the ceiling/
  corner, held there by an under-conditioned mDBC fallback pressure, that
  eventually collides with a second isolated fragment and gets violently
  ejected. `BOUNDARY_DENSITY_PLAN.md` §10 traces this particle-by-particle.

**Root cause, confirmed independently from two directions:**
[[sph-symmetric-pressure-truncation-artifact]] (the truncated-kernel pressure
ringing itself) and [[marrone31-truncation-artifact-vs-pst]] (the Antuono
pressure-switch mask chattering continuously near this same population —
checked against Barecasco 2013's own §5, which admits this exact instability
is open and unsolved in the source paper, not a porting bug). Lives entirely
in `computePressureForceSurfaceAware` (the Antuono/TIC switch), upstream of
and orthogonal to mDBC density scheme, DDT term, integrator, and PST choice —
confirmed by `BOUNDARY_DENSITY_PLAN.md` §11 item 1: the completeness gate
tried against the exact reproduced sloshingTank kick had **zero effect**
(`DELTASPH_VALIDATION_PLAN.md` §5.38 — mechanistically guaranteed to do
nothing, since the gate deliberately never touches free-surface-flagged
particles, which is exactly where this population lives). Cross-implementation
confirmation: diffSPH's own independent reference (no PST, different mDBC
entirely) shows the same pinning pattern on its own SPHERIC TC10 reproduction.

**Why it's hard, not just unaddressed:** the switch is load-bearing (forcing
either branch unconditionally, or removing it, makes Marrone 3.1 measurably
worse — `DELTASPH_VALIDATION_PLAN.md` §5.31), so any fix has to work *around*
or *alongside* it, not remove it. `letouze2025` (a free-surface-methods
review) independently confirms this switch "alone... drastically reduces
robustness" without a PST alongside it — the same conclusion this codebase
reached via its own instrumentation.

**Concrete next steps, not yet attempted (cheapest first):**
1. Freeze the Antuono switch's bulk/surface classification across RK
   sub-stages (same pattern already used for `sun2017DeltaSPH`'s frozen
   diffusion) — addresses only the intra-step half of the chatter.
2. A continuous coverage-fraction blend instead of the switch's current hard
   boolean classification (the "scan circle" idea Barecasco's own paper left
   unfinished).
3. Implement Barecasco et al.'s own Eq. (8) faithfully — a real, missing
   piece of the paper that may specifically help the degenerate near-corner
   population, even though it doesn't touch the general bulk/surface chatter.
4. A force-magnitude plausibility clamp keyed to a continuous conditioning
   signal already computed every step (`currentState.surfaceLambdas`, the
   renormalization tensor's minimum eigenvalue) — applied *inside* the
   free-surface population specifically, a genuinely different lever from
   "renormalize or don't." `DELTASPH_VALIDATION_PLAN.md` §10.4 item 2.

The Antuono-mask flip-rate diagnostic (`DELTASPH_VALIDATION_PLAN.md`
§5.18/§5.35) is the right tool to quantify whether any of the above actually
helps, rather than eyeballing more frame grids. **It is no longer in the
code** (checked 2026-09-28: nothing in `src/` or `scripts/`) — it has to be
rebuilt before step 1 can be judged.

**Step 1 measured, 2026-09-28 — moot under the current defaults.** Rebuilt
as `schemeConfig.surfaceMaskDiagnostics` (+ `freezeSurfaceMaskAcrossStages`
for step 1 itself; `probe_deltaSPHMarrone.py --surfaceMaskDiagnostics /
--freezeSurfaceMask`). Marrone 3.1 δ⁺ nx67, seed 1, to t* 7.7: the mask
**does not change between symplectic Euler's two stages** (0.2 % of steps,
max 3 rows, all before t* 1), but changes **step to step** in 45 % of steps
(up to 38 rows, growing through the run). So freezing across stages cannot
help — it was an RK4-era idea — and the chatter to attack is step to step:
**step 2 (a continuous coverage blend) is the next lever.** The frozen-mask
option stays available for RK integrators. Diagnostics do not perturb the
physics (bitwise the graphed run).

## 2. `omniIncompressible`'s `'mls'` wall-pressure mode — genuine numerical
instability, not a sign bug

`modules/incompressible/wallPressure.py`'s `'mls'` mode catastrophically
blows up under the `omniIncompressible` scheme's own Jacobi iteration on
`randomFlowIncompressible --bounded` (kinetic energy 0.32 -> 4.3e34 by step
40; confirmed still present after the `ghostOffsets` sign fix,
`WCSPH_DEFAULT_CLOSEOUT_PLAN.md` item G / `DFSPH_IMPROVEMENT_PLAN.md`'s
"Known-open" section). The mechanism is understood and documented in
`omniIncompressible.py`'s own code comment: the mode's linear term assumes a
locally-linear near-wall pressure (exact for a quiescent hydrostatic column,
wrong for a sheared flow), amplifying real near-wall pressure structure and
pumping energy into the iterative solver — an amplification/stability
problem in the Jacobi iteration, not a wrong-sign value. `'shepard'` (the
current default) sidesteps it by having no linear term to amplify. Not
attempted: a damped/relaxed version of the linear term, or a completeness-
style gate that falls back to `'shepard'` specifically where the local flow
is sheared rather than quiescent.

## 3. `'band'` mDBC scheme — the Shepard-precision-hole fix is real but the
architectural depth/lever-arm problem remains

After porting `english2025.py`'s symmetric-epsilon fix
(`WCSPH_DEFAULT_CLOSEOUT_PLAN.md` item A/C), `densityBand.py` no longer
diverges catastrophically on either Marrone 3.1 (adversarial) or Marrone 3.4
(full protocol, nx=256/t*=15.66) — a real, substantial improvement. But it
remains clearly the worst of the three mDBC schemes on every metric (Marrone
3.1: maxVel peak 63.2 vs `english2025`'s 4.1; Marrone 3.4: velocity check
fails at 17.7x U_max vs `english2025`'s 6.4x). This is consistent with
`BOUNDARY_DENSITY_PLAN.md` §3.1's separate, unaddressed finding: a Taylor-
shifted fitted gradient's error grows with the boundary layer's depth/lever
arm, independent of conditioning — `english2025`'s whole design point is
replacing that fitted gradient with a known, noise-free analytic term, which
`'band'` does not do and structurally cannot without becoming a different
scheme. Not a bug to fix; `'band'` stays a research/comparison tool.

## 4. §7.2 — lid-driven cavity density ramp / standing-wave amplification

A slow density ramp / amplifying standing pressure wave in the fully enclosed
LDC box, first reported under the shipped `rungeKutta2` integrator
(`DELTASPH_VALIDATION_PLAN.md` §7.2). Distinct from the (already-fixed)
`symplecticEuler` boundary-position-drift bug (§9.2) — this one is slower,
present regardless of integrator, and reads as "not enough dissipation for a
resonant acoustic mode in a domain with no free-surface energy sink." First-
pass instrumentation ruled out the no-penetration-shift mechanism as the
primary driver (`nopenshiftNActive` stays flat/negligible while the ramp
keeps growing). Next lever, not yet tried: whether the DDT/artificial-
viscosity coefficients are strong enough to damp an acoustic mode with no
free-surface sink at all — LDC may simply need more dissipation than a
free-surface case does, independent of any mDBC/mesh choice.
`WCSPH_DEFAULT_CLOSEOUT_PLAN.md` item F re-ran LDC to its full 30s duration
under the new default combo (`symplecticEuler`+`english2025`+`fourtakas2019`)
specifically to check this: `densityMedian` still creeps (1.0 -> 1.007 by
t=30), so the ramp is present and **not resolved** by the flip, but the run
completes cleanly with no divergence or acceleration into a blowup. A
qualitative comparison against the shipped pre-session reference video
(`examples/weaklyCompressible/outputs/09-lidDrivenCavity.mp4`, old defaults)
showed the OLD combo actually had visibly *worse* creep and a less-developed
vortex -- so the flip did not worsen this item either, on the evidence
available. (An exact numeric same-duration old-vs-new comparison run was
attempted but lost to a session restart before completing; not re-run given
the qualitative video comparison already available.)

**Status: noted, not active.** A slowly-growing standing acoustic mode in a
fully enclosed WCSPH box with no free-surface sink is expected behavior for
this class of scheme, not a mystery to keep chasing — logged here so it
isn't re-discovered as a surprise, but there is nothing queued to fix it.

## 8. mDBC wall suction at wall/free-surface contact lines — both schemes

**What it is:** a thinly-supported *free-surface* particle next to a solid
wall (a run-up jet 1-2 particles thick, a corner, a particle on the ceiling)
acquires negative pressure from the wall closure, the symmetric `(p_i + p_j)`
pressure force pulls it into the wall, it loses fluid support, and the suction
deepens — geometric growth (x1.2-2.5 per step on one ACSPH particle, p from
-20 to -1.8e6 over ~20 steps) ending in a wall crossing and a velocity kick
that collapses dt. Same event, two schemes:

- **ACSPH** (`FREESLIP_DAMBREAK_FINDINGS.md` §9.5): Eq. (61) Shepard wall
  pressure. After `fcaa998` (dual-time wall-row fix, paper viscosity, Michel
  shifting) the nx=24 dam break survives to t* 8.1, but **nx=70 still fails at
  the first far-wall run-up (t* 2.8)**; a ceiling-riding particle (constant
  speed against gravity, ~1 s) appears in every nx=24 run.
- **delta-SPH**: the ceiling hover of item 1 (sloshingTank UID 4104, "pinned by
  an under-conditioned mDBC fallback pressure"), and the english2025 low-alpha
  leak the neighbour-count ramp fixed (`FREESLIP_DAMBREAK_FINDINGS.md` §8) —
  the same under-supported-extrapolation-at-a-wall family.

Item 1 attributes its population to the truncated-kernel pressure-pair
artifact and the Antuono switch; this entry is the wall half of it: the
*wall closure* supplies the negative pressure, not just the fluid-fluid sum.

**Ruled out (ACSPH):** Antuono pressure switch (no change); clamping
surface-particle pressure to >= 0 (worse — the runaway moves one layer in);
solver non-convergence (the pseudo-time loop converges every step; its
velocity-only metric just does not see pressure).

**Isolation:** the no-free-surface control (`FREESLIP_DAMBREAK_FINDINGS.md`
§9.7 — periodic TGV, bounded random flow, moving obstacle) is clean in both
schemes, so walls alone are fine; the free-surface contact is required.

**Plan:** `MDBC_CONTACT_LINE_PLAN.md`.

**Status 2026-09-23 — mechanism identified, fix implemented (opt-in).
2026-09-24: NOT validated -- the videos show the ACSPH fix (and the delta-SPH
unilateral wall) spraying isolated high-speed fliers through the domain;
the metrics used did not measure this (plan §9)** (`MDBC_CONTACT_LINE_PLAN.md` §7). The wall is *bilateral*:
Eq. (61)'s hydrostatic term is negative at any ceiling / above any waterline,
and ACSPH's converged `div v = 0` with the mirrored wall velocity is itself a
no-separation constraint -- clamping the wall pressure alone (the literature's
EDAC/PySPH/DualSPHysics fix) moves the tension onto the fluid row's own
pressure and does not release the contact. Fix: Batty et al. 2007's
complementarity `0 <= p ⊥ separation`, applied only where air can reach -- the
surface *vicinity* (raw surface set dilated by one support, fluid AND wall
rows, plus fully isolated rows): `acParams.cavitationProjection = 'vicinity'`
(projected pseudo-time iteration). No threshold. **ACSPH only** -- three
delta-SPH variants failed on sloshingTank (p-only clamp: stored density
deficits released as kicks, diverges t 2.9; density floor: acoustic
rectification ratchet, +3 % bulk density, KE / 85), so the delta-SPH option
was removed; see the plan's §7.5.
Marrone 3.1 ACSPH: nx=24 ceiling riders gone; **nx=70 passes the t* 2.8
run-up and reaches t* 9.91** (one run, 7505 steps) (was: fails at 2.8). Default still 'off'.

**2026-09-24/25 -- every fix tried so far fails** (plan §8-§9, flier metric
in `scripts/probe_contactLine.py`, time-aligned videos via
`scripts/align_run_videos.py`). Both scheme defaults produce **zero** free
fliers (delta-SPH sloshingTank, delta-SPH and ACSPH Marrone 3.1); the
contact is the only defect. ACSPH `vicinity` (p >= 0 on vicinity fluid+wall
rows): ~100 free fliers at nx=70, each droplet-surface contact elastic
(restitution ~1), so spray only accumulates. delta-SPH unilateral freeSlip
velocity + one-sided hydrostatics: fliers launched off every wall, removed
from the code. ACSPH `'wall'` (Batty on the fluid-solid pairs only, force
alone): clean ceiling release on the toy, but a separating row's pressure
runs away (the divergence stayed bilateral), Marrone nx=24 blows up t 0.48.
Adding the kinematic half (wall-pair divergence kept only while
compressive): the clamp rectifies discretisation noise at every wall (the
SPH wall-pair divergence is not the normal relative velocity), the fluid
body is pushed off the floor and floats, blowup t 0.34; sealed box NaN at
t 0.13 (pure-Neumann compatibility violated). **Next idea, not tried:** a
per-contact *normal relative velocity* criterion (Batty's `(u - v_s).n`)
instead of the kernel-weighted wall-pair divergence; a sealed container
stays incompatible with any unilateral wall unless a gap can open.

**Still open here:** (0) delta-SPH (plan §7.5): the complementarity has to
live in the continuity equation, acoustically neutral; (a) rows next to a wall with a badly deficient support
(lambda < 0.5) that Marrone's raw detection does not flag -- the vicinity set
is only as good as the raw set (nx=70, t* 4.5-4.9, wall p to -23); (b) the
entrapped-cavity / flyer population is §1's (fluid-fluid, positive-pressure
ejections), not a wall contact. Side observation: a *sealed* box (no free
surface) under gravity drifts to vmax ~ 1 with an all-negative ACSPH pressure
field -- the `(p_i+p_j)` tensile instability under uniform tension,
independent of this fix. Detection bug found on the way (all schemes): an
isolated particle reads `lambda = 1` and is classified as bulk;
`detectIsolated` is the exact fix, used by ACSPH's set only -- changing the
shared detector would touch shifting / the Antuono switch everywhere.
**2026-09-28:** checked per detector on an isolated row: the default every
case runs (Barecasco + lambda-gradient normals), ColorField and
ColorFieldGrad miss it; Marrone's detection flags it (its lambda reads 1.0,
but another criterion catches it). Fixed in the shared detector on branch
`isolated-surface` (`SurfaceDetectionConfig.flagIsolated`, default on),
merged into `dev` after the baseline re-runs. **Before/after (2026-09-28):**
δ-SPH is **bitwise unchanged** — Marrone 3.1 δ⁺ nx67, three jittered
realisations each with the flag on and off (`scripts/run_isolatedFlagAB.sh`),
and sloshingTank to t = 7 — as expected, since an isolated row has no pair
interactions. englishWedge / squarePatch / impact identical too; only DFSPH's
staticBlob moved (surface min density 0.472 -> 0.454). So this is a
classification fix (diagnostics, row-local consumers), not a dynamics one.
(The same batch's sloshingTank Sensor 1 peak, 78 kPa vs 25 kPa in the
2026-09-26 run, is therefore not the flag: code changes since then — the
multi-lane kernels change summation order — plus a chaotic flow; seed 2 of
the Marrone A/B shows the spread, P2 peak 91 and late |v|max 19.)
**Decision (user, 2026-09-28):** worth fixing in the shared detector
eventually, but low priority -- it is *not* why free surfaces blow up: that
was investigated before and traced mainly to the density-diffusion choice
(`fourtakas2019` DDT is what helped), not to surface detection.

See also: [[sph-symmetric-pressure-truncation-artifact]],
[[antuono-pressure-switch-bug]], [[english2025-alpha-neighbour-ramp]].

## 9. ACSPH: a positive pressure level in a closed box grows exponentially

**What it is:** in a fully sealed box (`probe_contactLine.py --toy sealed`,
fillRatio 1, gravity up, nx=32, `--paperAC`), unmodified ACSPH (`off`)
started from a uniform p = +10 grows the level exponentially -- 25 -> 75 ->
201 -> 513 -> 1290 over t 0.06-0.38 (doubling ~ every 0.05 s) -- then
collapses into disorder at t ~ 0.44 (`--p0 10`, 2026-09-24). Started from
p = 0 the same box drifts the other way, to an all-negative field saturating
near -16 with vmax ~ 1 (MDBC_CONTACT_LINE_PLAN.md §7.2). A closed box's
pressure is defined only up to a constant (pure-Neumann null space); nothing
in the dual-time solve anchors the level, and a positive level is unstable
under the Eq. (61) wall closure + `(p_i + p_j)` force.

**Why it matters:** any treatment that shifts a closed region's level
(any unilateral wall does, by removing legitimate tension) triggers it;
enclosed air-free regions are not exotic (a filled tank, a trapped pocket).

**Next step:** a gauge fix (pin the mean or one row's pressure in a closed
fluid region) vs. whether the growth is the truncation artefact of the
symmetric sum near walls ([[sph-symmetric-pressure-truncation-artifact]])
acting on a uniform level -- test with a uniform level in a periodic box (no
walls): growth there would point at the solve, none at the wall closure.

## 13. `hydrostaticColumn` under `iisph` blows up in its default free-slip configuration

**What it is (2026-09-28, found by the §7 Morris A/B):** `iisph`, nx=128,
`semiImplicitEuler`, `wallBC=freeSlip`, `nu=0` — the configuration
`docs/historic_plans/DFSPH_FINDINGS.md` §1.14's post-Part-41 table graded as
stable over 1200 steps (|v|max 1.94, embMin 0.94) — now runs away: |v|max 24
by t = 0.147 (step 238), velocity alarm at step 433 (particle near the bottom
wall), 1.25e6 by step 1200. Blocks the §7 A/B, whose other arms are judged
against this one. Next: same arm on `main` (is it today's work?), then bisect
back to 2026-09-04.

**Narrowed the same day:** also on clean `main` (not today's work), and
**`iisph` only** — the case default `divergenceFree` holds the same column
(|v|max 0.3-0.7, embMin 0.95-0.98 over 1200 steps / 8 s). A single-run bisect
is unreliable: before 2026-09-02 the case jittered its lattice with an
unseeded RNG, so the same commit gave |v|max 1.5-4.2 over 300 steps; `71a8ae7`
("tmp commit", 2026-09-02) commented that jitter out (still out today), which
made runs deterministic. Re-enabling it on today's code helps (|v|max 6.6-21
vs 39) but does not restore the old behaviour, so more than one change since
2026-09-01 is involved (dt also halved: t = 0.31 vs 0.69 after 300 steps).
**Parked** (`iisph` is the retired DFSPH track's reference variant); the §7 A/B
runs on `divergenceFree` instead. To resume: bisect with ~3 jittered runs per
commit; decide whether the `71a8ae7` jitter removal was meant to stay.

**Re-checked 2026-10-01 (dev `991e4fa`): still broken, identical to the 09-28
record.** Same arm (`iisph`, nx 128, `semiImplicitEuler`, `freeSlip`, `nu = 0`,
1200 steps): velocity alarm at **step 433** (t = 0.160, |v|max 239, particle
636 at x = [0.54, -0.69], i.e. the bottom wall), dt collapses to 1e-8, t stalls
at 0.1616, |v|max 1.25e6 at step 1200, embedded density 0.17. Deterministic
(same step as before). The `divergenceFree` control on the same arm is fine
(|v|max 2.06, embMin 0.89, t = 7.96, no alarm). Nothing merged since 09-28
touches the incompressible schemes (the 09-29 `timeCentredContinuity` /
noPen-`impulse` defaults are weakly-compressible-only), so no change was
expected. One untested lead from the ceiling-sticking work: the blow-up starts
on a wall-contact particle under an explicit/semi-implicit integrator, the same
signature as the explicit-midpoint wall-contact mode `CEILING_STICKING_PLAN.md`
§3-§6 found for WCSPH; whether IISPH's pressure solve has an analogous
integrator-side issue is not known. Videos/script:
`scratchpad/hydroRecheck/`, `probe_hydroRecheck.py` (session scratchpad).

## 14. DFSPH + physical viscosity + free-slip walls: the column's velocity alternates sign every step

**What it is (2026-09-28, found by §7's A/B):** `hydrostaticColumn`,
`divergenceFree`, `wallBC=freeSlip`, `nu=0.01`: the mean vertical velocity of
the whole column flips sign every step (±0.015-0.019), with either viscosity
term (projected or Morris); late on a near-wall horizontal band does the same.
Not the viscous timestep limit — halving dt only shrinks it (±0.005-0.01).
Absent with `nu = 0` (free slip) and with no-slip walls. Suspects: the
free-slip boundary-velocity mirror (`computeBoundaryVelocities`) feeding the
viscous term, against the divergence-free solve. Not investigated further.

**Re-checked 2026-10-01 (dev `991e4fa`): half still there, half not.**
`divergenceFree`, nx 128, 1200 steps, `freeSlip`, `nu = 0.01`:
- projected term (`monaghanGingold`): mean vertical velocity flips sign **every
  step** (flip fraction 1.0), amplitude 0.022 — reproduced. The column is
  otherwise very quiet (|v|max 0.07, embMin 0.99).
- **Morris 1997 term: amplitude 4e-4, ~50x smaller** (|v|max 0.17, embMin 0.98),
  the same as the no-slip Morris control (4e-4). So "with either viscosity
  term" no longer holds: the alternation is tied to the *projected* viscous
  term (a bulk Laplacian, §7), not to the free-slip mirror feeding any viscous
  term. That narrows the suspect list away from `computeBoundaryVelocities`.
- Caveat: the `nu = 0` control (|mean vy| up to 0.05, flip fraction 0.77) is not
  "absent" by this crude metric — its real slosh swamps it, so the original
  "absent with `nu = 0`" needs a detrended alternation measure (e.g.
  |v_k - (v_{k-1} + v_{k+1})/2| of the mean) to confirm; not yet done.
Impact is small while Morris is the opt-in alternative and the column stays
bounded. Videos in `scratchpad/hydroRecheck/` (session scratchpad).

## 15. CRKSPH on a lattice shock — blow-up RESOLVED 2026-09-30; residual and follow-ups open

The blow-up (a CRK artificial-viscosity regulariser `1e-7 h^2` added to a dimensionless `eta.eta`, unbounded `mu` on a
near-coincident approaching pair) is fixed and moved to [RESOLVED_PROBLEMS.md](docs/historic_plans/RESOLVED_PROBLEMS.md)
with its mechanism, A/B and reproduction. What stays open:

**Residual (not a blow-up).** With a sharp same-lattice mass step the first light column is kicked forward within
four steps (0.56 m/s), the light columns then settle into an irregular spacing, and one pair of columns per
interface merges (coincident from t ~ 0.18 and stays, 20 pairs per interface, min spacing 0.0004 dx); equal-mass
sampling never does this (min spacing 0.6 dx). It does not hurt accuracy by the L1 measure (Sod 2D same-lattice
1.4e-2 vs equal-mass 1.9e-2 at t = 0.25, exact Riemann). The paper's Sod / triple point are equal-mass (Sod 3D:
"to maintain mass matching"), so this is an initial-condition property of our extra same-lattice preset.

**Follow-ups for the AV / limiter work.** (1) [2026-09-30: `C_l` = 2 tried on Gresho / Sod, worse on Gresho (+8 % -> +11.5 % KE at nx 64, +12.6 % -> +21.9 % at nx 96), 70 % less Sod ringing; kept at 1, to be revisited with the limiter, CRKSPH_LIMITER_PLAN note (b)] `C_l, C_q`: Frontiere Table D.1 makes them kernel-dependent
(mu scales with h): 7th-order B-spline, our default, `C_l = 2.0, C_q = 1.0`, Wendland `0.5 / 0.25`; our
`buildDefaultDiffusionParamsCRKSPH` uses 1 / 1 for every kernel (unverified whether `smooth = H/xi` equals the
paper's h for B7, CRKSPH_LIMITER_PLAN O2 is the same units question for the limiter). (2) The compressible dt is
acoustic only (`cfl h_min / (xi c_s)`), no approach/viscous term, so it cannot see a column closing (a smaller cfl
only hid the pile-up). (3) [2026-10-01: Gresho spin-up traced to an intrinsic, slowly converging pressure pump; the limiter default is now (1/n_h, 0.2/n_h); CRKSPH_LIMITER_PLAN notes (c)-(g)].


## 18. Video runs leak GPU memory in proportion to the per-pair state (CRKSPH 3D Sedov OOMs at nx 40)

Found 2026-10-01 re-taking the CRK AV baseline (`scripts/av_report.py --config crk --profile full`): `crkNone / sedov` (3D, nx 40, video on, velocity alarm drawing every step) went 8.5 -> 21.5 GB
allocated over 110 steps and hit CUDA OOM (the GPU is shared with other processes; ~28 GB free). Not the solver: the same run **without video** holds a flat 3.2 GB (nx 40; 1.4 GB at nx 30) over 160 steps. With
`plot=True, video=True, velocityAlarmPlotInterval=1` at nx 30 allocated memory grows ~56 MB per step (1.4 -> 6.7 GB in 120 steps), i.e. about one 15M-entry float array per drawn frame.
Suspect: `_RenderThread.submitFrame` (`runner/runner.py`) snapshots the state with `utils/cudaGraph._cloneState`, which clones **every** tensor attribute -- including the pair-sized `ap_ij` / `av_ij` / `f_ij` a CRK state carries (tens of
millions of entries in 3D) -- and something keeps the snapshots alive after the frame is drawn (the queue is capped at 2 pending, so a retained reference, not the queue, is the leak). Two independent fixes: clone only particle-sized tensors that
the plot hook reads, and find what retains the snapshot. Workaround: `--noVideo` for large 3D CRK runs (scalar metrics are unaffected). Not investigated beyond the measurement above (`scratchpad/sedmem.py` pattern: diagnostics hook printing
`torch.cuda.memory_allocated()`).

## Resolved (details in [RESOLVED_PROBLEMS.md](docs/historic_plans/RESOLVED_PROBLEMS.md); numbers kept so references stay valid)

- **§5** englishWedge concave-corner residual -- sign bug in the `fourtakas2019` hydrostatic correction, fixed `68a9a6d`; 2026-09-29 re-validation: keep the default combo.
- **§6** lattice-density kernel calibration -- `onResidual='raise'` path now tested (`tests/test_restDensityCalibration.py`); wiring the normalisation into every operator is deliberately not done (not worth it), documented on `SimulationConfig.calibrateNormalization`.
- **§7** missing shear-carrying laminar viscosity -- Morris (1997) term implemented, opt-in (`viscousTerm = morris1997`). Whether to make it the default for viscous no-slip cases is the user's call.
- **§10** runner divergence detection -- velocity alarm + probe stall defaults.
- **§11** `probe_contactLine` delta-SPH toys' pinned dt -- adaptive dt capped at the pinned acoustic Courant 0.3.
- **§12** render thread + grid-interpolated plots crashing CRKSPH cases -- plot hooks that run warp kernels stay on the loop thread.
- **§16** Monaghan viscous heating lacked the 1/2 -- fixed, `tests/test_monaghanEnergy.py`.
- **§17** Read-Hayfield entropy dissipation did not conserve energy -- Eq. (33) transcription error, fixed; pair loops moved to warp kernels (`wp_readHayfield.py`).
- **§18** Monaghan Sedov energy drift 5e-4 -- RK2 time-integration error, expected behaviour, converges at second order in dt.

See also: [[boundary-density-plan]], [[wcsph-deltasph-scheme-concerns]],
[[sph-symmetric-pressure-truncation-artifact]],
[[marrone31-truncation-artifact-vs-pst]], [[antuono-pressure-switch-bug]],
[[incompressible-plan-sequencing]].
