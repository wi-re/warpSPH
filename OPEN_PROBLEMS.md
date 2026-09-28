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

**Current batch** (user, 2026-09-28; on branch `dev`):
1. §5 / §6.3 / §11 re-runs as a baseline, in the background;
2. §7 Morris shear viscosity (opt-in; DFSPH `hydrostaticColumn` no-slip A/B
   against Part 39, δ-SPH unchanged when off);
3. §8's `detectIsolated` in the shared surface detector (changes a default:
   shifting and the Antuono switch everywhere) — then repeat the step-1
   re-runs plus Marrone 3.1 and sloshingTank as the before/after check;
4. §1 step 1, frozen Antuono mask across RK sub-stages — dev work: the
   flip-rate diagnostic has to be rebuilt first.

## 1. Corner-flyer / free-surface-pinning instability (the "flyers" and
"ceiling-sticking" phenomena)

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

## 5. One older mDBC/sampling item, still open (the other resolved)

- **`_gridSnapGhostOffsets`'s concave-corner ghost collapse**: ~40 boundary
  particles at a concave corner (e.g. englishWedge's base corner) collapse
  their ghost node onto one interior point, a degenerate near-coplanar
  stencil no fit handles well. Not rechecked as of 2026-09-18 (deferred, see
  `WCSPH_DEFAULT_CLOSEOUT_PLAN.md` item H).

  **TODO:** re-run `englishWedge` dp=0.01 under the current default combo
  (`english2025`+`fourtakas2019`+`symplecticEuler`) to check whether the
  base-corner residual (RMSE 0.0431 last measured) has moved at all, before
  deciding whether this still needs its own fix.

  **Re-run 2026-09-28** (default combo, dp = 0.01, t = 4 s, video:
  `scripts/out_looseEnds/baseline/englishWedge/`): stable, bulk / near-wall /
  wedge-face bands pass, but the **base corners got worse — RMSE 0.0674 (max
  0.096)**, against 0.0431 under the old defaults; the apex just fails too
  (0.0312 vs 0.03). The frames show a weak spurious circulation along the
  wedge faces (|v| up to 0.04 m/s) and a disturbed lattice around the apex.
  So this still needs its own fix; which part of the default flip moved it
  (english2025 / fourtakas2019 / symplecticEuler) is not isolated.

(The H/Δx≈72-160 resolution hole that used to share this item was resolved
2026-09-18: [RESOLVED_PROBLEMS.md](docs/historic_plans/RESOLVED_PROBLEMS.md).)

## 6. Lattice-density kernel calibration — low-priority polish, not urgent

`retired_plans/LATTICE_DENSITY_PLAN.md`'s core design (correct the kernel
normalization for a finite-support lattice, `calibrateNormalization`) landed
and is verified. Its original migration plan (a mass-vs-kernel split, then
flip the default) turned out to be **moot**: the real bug was a mass/cell
mismatch at the sampler, fixed directly at the source
(`sample/regular.py:62`), which is why `calibrateRestDensityMasses` was
retired in favor of `calibrateRestDensity`. With mass fixed at the sampler,
the residual the kernel term still corrects is tiny (≤0.04% on
sloshingTank) — **this is genuinely a "do if you've got time to burn" item,
not something worth going back into the core operator machinery for.**
Three loose ends, all optional:

1. No test exercises `calibrateRestDensity`'s `onResidual='raise'` path —
   nothing deliberately corrupts an IC (mismatched mass, an overlapping
   region) to confirm it actually raises.
2. Whether wiring the remaining ~131 `OperationProperties(...)` call sites
   (gradient/divergence/laplacian/curl/interpolate) is worth it at all, now
   that the number it would remove is this small. If the answer is ever
   yes, build the `propsFromConfig` helper the plan's §3.9 already flags
   rather than editing them by hand.
3. The sampler mass fix itself is unverified outside the cases actually run
   that session — `kelvinHelmholtz`, `rayleighTaylor`, `kidder`,
   `yeeVortex`, `triplePoint`, `staticBlob`, `impact`, `squarePatch` all use
   the same sampler and now get a slightly different (more correct) mass,
   but none are in `tests/test_physics.py`'s fixture set and none were
   re-run by hand.

## 7. Missing shear-carrying laminar viscosity term (Morris et al. 1997) —
no tangential stress at a no-slip wall

**What it is:** the stock velocity-diffusion term
(`computeVelocityDiffusion`'s `inviscid=False` branch, `wp_viscosityDelta.py`)
is `mu_ij * gradW` with `mu_ij = (v_ij . x_ij)/|x_ij|^2` — a scalar built by
contracting the relative velocity along the separation vector `x_ij`, not the
full `v_ij` vector. That makes it a normal-projected diffusion: it damps the
approach/separation component of relative velocity but carries no
tangential/shear stress at all. A real Morris et al. (1997) laminar viscosity
term needs the full vector Laplacian, which neither `viscidNu` nor
`viscosityParams` currently has.

**Why it matters:** `hydrostaticColumn`'s free-slip side walls leave a
bounded, undamped limit-cycle slosh (documented as non-fatal — DFSPH_FINDINGS.md
§1.12/§1.20). Switching to `wallBC=noSlip` + `viscidNu` through the existing
(normal-projected) term does bound the slosh KE (~4x down) and hold the
hydrostatic gradient, but roughens the free surface
(`embeddedMinDensity` 0.94 -> 0.60, `|v|max` spikes to ~3.7-4.1) — a no-slip
mirror through a normal-only diffusion term adds noisy *normal* wall damping
with no tangential component, not the physically-correct shear stress a real
no-slip wall should apply. A hand-rolled full-vector Brookshaw Laplacian
(DFSPH_FINDINGS.md Part 39, since removed in the Part 41 cleanup) *did* hold
the surface (embMin 0.94-0.97) while damping the slosh at the same time,
confirming the mechanism — the module layer just doesn't have it as a real,
supported option today.

**Why it's not a quick fix:** it needs a new kernel term (the full `v_ij`
vector, not the scalar `mu_ij` reduction), wired as a `DiffusionParameters`
option, gradcheck'd (it touches a `@wp.kernel`), and given its own `deltaSPH`
regression pass so it doesn't silently change WCSPH's diffusion behaviour
too. Half of the groundwork already landed (2026-09-05, `ACSPH_PLAN.md` step
5 — `computeVelocityDiffusion(approachOnly=False)` lifts the approach-only
clamp, turning the `inviscid=False` branch into the Monaghan & Gingold
(1983) Laplacian, De Courcy et al. 2024 Eq. (25), gradchecked in all four
`inviscid` x `approachOnly` combinations, default unchanged) — what remains
is the actual full-vector term, a new kernel, not a flag flip.

**Concrete next step:** implement the full `v_ij` Morris Laplacian as a new
`DiffusionParameters`-wired option alongside the existing `viscidNu` scalar
term, gradcheck it, and re-run the `hydrostaticColumn` `wallBC=noSlip` A/B to
confirm it reproduces Part 39's numbers (embMin held, KE damped) through the
stock machinery instead of the since-removed bespoke path.

**2026-09-28 — implemented, A/B pending.** `diffusionParams.viscousTerm =
ViscosityTerm.morris1997` (opt-in; default `monaghanGingold` unchanged),
gradchecked, `tests/test_morrisViscosity.py`. **Correction to "carries no
tangential/shear stress at all" above:** on a divergence-free shear wave
`v = (sin ky, 0)` in the bulk, the projected term reproduces `nu lap(v)` too
(4.8 % / 4.3 % error at n = 32 / 64, vs Morris 2.6 % / 1.8 %) — its
`K = 2(d+2)` normalisation makes it a consistent Laplacian away from
boundaries. So whatever Part 39's term did better has to come from the
truncated supports at walls and the free surface, which is what the
`hydrostaticColumn` A/B (`scripts/probe_morrisNoSlipColumn.py`) measures. `DFSPH_IMPROVEMENT_PLAN.md`'s
ranked-queue item 1, `DFSPH_FINDINGS.md` §1.14 (both now retired to
`docs/historic_plans/` — the incompressible/DFSPH track itself reached a
stable, documented recommendation (`divergenceFree` default, `band2018pb` as
a deliberate trade-off) and this is the one item that survived it).

See also: [[boundary-density-plan]], [[wcsph-deltasph-scheme-concerns]],
[[sph-symmetric-pressure-truncation-artifact]],
[[marrone31-truncation-artifact-vs-pst]], [[antuono-pressure-switch-bug]],
[[incompressible-plan-sequencing]].

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
`isolated-surface` (`SurfaceDetectionConfig.flagIsolated`, default on), not
yet merged — it changes a default, so it waits for the baseline re-runs.
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

## 10. Runner divergence detection misses bounded-NaN-free blowups — RESOLVED 2026-09-28

Moved to [RESOLVED_PROBLEMS.md](docs/historic_plans/RESOLVED_PROBLEMS.md) (velocity
alarm + probe stall defaults; README "Watching a run").

## 11. probe_contactLine delta-SPH toys: pinned dt is overridden

**What it is (minor, probe-level):** the probe sets `soundSpeed` and
`targetDt = 0.3 dx / c0` for the WC toys, but the per-step adaptive
`computeTimestep` hook overrides dt (the runs still step at ~5.9e-4,
acoustic Courant ~0.8). Stable with `symplecticEuler`, so results stand, but
the toys do not run at the Courant number the probe claims.

**Next step:** make the toys' timestep hook return the pinned dt (or report
the achieved Courant number) before relying on dt-sensitive toy results.

## 12. Render thread + grid-interpolated plots crash or hang CRKSPH cases — RESOLVED 2026-09-28

Moved to [RESOLVED_PROBLEMS.md](docs/historic_plans/RESOLVED_PROBLEMS.md)
(found by the §6.3 re-runs; plot hooks that run warp kernels now stay on the
loop thread).

