# Open problems — known, hard, not part of routine closeout work

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
§5.18/§5.35) is already instrumented and is the right tool to quantify
whether any of the above actually helps, rather than eyeballing more frame
grids.

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

**RESOLVED, 2026-09-18** (moved out of this list): the **H/Δx≈72-160
resolution hole** (`DELTASPH_VALIDATION_PLAN.md` item 4b) — every Marrone
3.1 config used to blow up specifically at nx=120 (H/Δx=72), in t*≈5.5-6.3,
"regardless of scheme." Rechecked under the new default combo
(`english2025`+`fourtakas2019`+`symplecticEuler`) at the exact same
resolution, run to t*≈7.68 (past the old failure window):
`diverged=False`, maxVel peak 7.41 (moderate, decaying), density
[0.982, 1.039] — no divergence. Not root-caused *why* (which of the
combo's components fixed it, or whether it's a combination, wasn't
isolated), but the practical symptom is gone. `WCSPH_DEFAULT_CLOSEOUT_PLAN.md`
item H has the full result.

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
stock machinery instead of the since-removed bespoke path. `DFSPH_IMPROVEMENT_PLAN.md`'s
ranked-queue item 1, `DFSPH_FINDINGS.md` §1.14 (both now retired to
`docs/historic_plans/` — the incompressible/DFSPH track itself reached a
stable, documented recommendation (`divergenceFree` default, `band2018pb` as
a deliberate trade-off) and this is the one item that survived it).

See also: [[boundary-density-plan]], [[wcsph-deltasph-scheme-concerns]],
[[sph-symmetric-pressure-truncation-artifact]],
[[marrone31-truncation-artifact-vs-pst]], [[antuono-pressure-switch-bug]],
[[incompressible-plan-sequencing]].
