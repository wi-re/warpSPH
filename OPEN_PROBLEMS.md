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

**Batch of 2026-09-28 — done** (on branch `dev`; outcomes in each item):
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

## 5. englishWedge concave-corner residual — RESOLVED 2026-09-28

Moved to [RESOLVED_PROBLEMS.md](docs/historic_plans/RESOLVED_PROBLEMS.md): the
regression was a sign bug in the `fourtakas2019` hydrostatic correction
(fixed `68a9a6d`). **Follow-up, re-validation done 2026-09-29 (recommendation
pending, decision is the user's):** the default combo (english2025 +
fourtakas2019 + symplecticEuler) re-run with the fixed term against the
`deltaSPH` DDT, everything else default (batch
`scratchpad/run_overnight_batch_2026-09-28.sh`, tables in
`scripts/out_overnight_2026-09-28/SUMMARY.md`, videos looked at):

- **Marrone 3.1**: P1 plateau (0.46 delta+, 0.83-0.88 PST-off — the PST-off
  value is the old on-wall-probe extrapolation, In1/Shepard read 0.62) and P2
  peak (0.23-0.33) are **identical between the two DDTs and to the pre-fix
  09-26 baseline**; the sign bug never moved them. Checks passed: delta+ nx67
  x3 seeds fourtakas 27/27 vs deltaSPH 24/27 (fails are P2 "quiescent"
  flags = a flier crossing the probe); nx134 8/9 vs 7/9; PST-off 8/9 vs 7/9.
  Bulk density: PST-off [0.992, 1.007] vs [0.977, 1.023]. **One signal against
  fourtakas:** delta+ nx67 max|v| 16.1 / 11.4 / 19.7 vs deltaSPH 10.3 / 10.1 /
  10.2 (buggy baseline 10.9) and wider rho extremes ([0.81, 1.20] vs
  [0.93, 1.08] worst seed); the fast ones are ceiling-pinned isolated
  particles at t\*~5.3-5.7 (the §1 kick, rhoMin/rhoMax and vmax coincide in
  time), present in both DDTs' frames. Opposite at nx134 (23.0 vs 30.1) and
  PST-off (10.2 vs 21.4). n=3 seeds: not a separation.
- **Marrone 3.4** nx256 (delta-SPH and delta+): all 6/6 PASS, both DDTs;
  penetration 0.64-0.76 dx; fixed fourtakas indistinguishable from the buggy
  baseline (vmax 40.9 vs 39.5, same rho extremes 0.63/1.8). Pointwise rho max
  is higher with fourtakas (1.79 / 1.58 vs 1.40 / 1.33; P99 bulk band equal).
- **sloshingTank** t=7: both DDTs survive; Gaussian-10 ms wall-sensor peaks
  2.9 / 3.8 / 7.3 kPa (fourtakas, fixed) vs 3.9 / 4.8 / 5.3 (deltaSPH) vs
  2.4 / 3.5 / 5.0 (buggy baseline), all inside the measured 2.2-13.1 kPa band;
  rho extremes [0.82, 1.32] (fourtakas) vs [0.73, 1.50] (deltaSPH). Raw
  sensor spikes 25-50 kPa in both (known).
- **Gallery**: 33/34 ran; weakly-compressible examples healthy (impact,
  LDC, open-flow, moving obstacle etc. look at least as good as the
  2026-09-13 stored images, which pre-date the current defaults); the one
  failure is compressible and unrelated, see §15.
- **Verdict for the user:** no evidence to change the default; fixed
  `fourtakas2019` is equal or slightly better on the Marrone/sloshing
  checks and density band, with one mild counter-signal (nx67 delta+ kicks
  in 2/3 seeds) that belongs to §1, not to the DDT. A 10-seed nx67 delta+
  A/B would settle that counter-signal if wanted.
  **2026-09-29: counter-signal explained** ([CEILING_STICKING_PLAN.md](CEILING_STICKING_PLAN.md) §3):
  those kicks are symplecticEuler's explicit-midpoint (ρ, v) update
  amplifying the wall-contact acoustic mode of a ceiling-pinned cluster;
  with `timeCentredContinuity` the three seeds read max|v| 10.1 / 9.9 / 9.9
  and ρ [0.96, 1.03], tighter than either DDT before.

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
   **Done 2026-09-28** (`scripts/run_looseEndsBaseline.sh baseline`, video,
   `scripts/out_looseEnds/baseline/`): all eight reach their tLimit, no
   velocity alarm, total energy conserved to the printed digits in the
   compressible ones (kidder is driven, RT's total excludes potential energy).
   Frames match the stored references where comparable (squarePatch vs the
   2026-09-18 gallery, kelvinHelmholtz, rayleighTaylor plumes at t≈3.8). The
   first pass crashed three CRKSPH cases — not the sampler fix, the render
   thread (§12, resolved). Item 3 closed; items 1 and 2 stay optional.

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
`hydrostaticColumn` A/B (`scripts/probe_morrisNoSlipColumn.py`) measures.

**A/B, 2026-09-28** (`scripts/probe_morrisNoSlipColumn.py`, `divergenceFree`
— `iisph` no longer holds even the free-slip arm, §13). First pass: projected
+ no-slip was already clean on `divergenceFree` (embMin 0.956, slope 1.003 —
Part 41's embMin 0.60 was `iisph`-specific); Morris + no-slip showed a
near-wall band at |v| ~0.17 whose velocity **flipped sign every step** while
the particles stayed put — an explicit-diffusion instability. Cause: the
shared DFSPH timestep (`kolmogorovIncompressibleTimestep`) divided the viscous
limit by `kernelScale` once instead of squared, allowing nu dt / hs^2 = 0.24
(hs = smoothing length) instead of Morris' 0.125. Fixed; at the corrected
limit Morris + no-slip settles the column to rest (|v|max <= 0.01, near-wall
and bulk rows ~1e-4 by t = 0.2). The projected term tolerated the old limit
because it couples more weakly. Final table (`scripts/out_morrisNoSlipColumn/SUMMARY.md`,
`divergenceFree`, nx=128, 1200 steps, tail = last quarter, corrected dt):

| arm | \|v\|max mean | KE mean | embMin | slope |
|---|---|---|---|---|
| free-slip, nu = 0 | 0.541 | 3.1e-4 | 0.889 | 0.966 |
| no-slip, projected | 0.0127 | 1.5e-7 | 1.000 | 1.004 |
| no-slip, Morris | 0.0055 | 4.3e-8 | 1.000 | 1.006 |
| free-slip, Morris | 0.171 | 5.3e-5 | 0.982 | 1.006 — see §14 |

**Verdict:** with a no-slip wall both terms now bring the column to rest;
Morris damps ~3.5x harder (KE) with smaller peaks. The Part 39 gap is closed
through the stock machinery. Morris stays opt-in (`viscousTerm`); whether to
make it the default for viscous no-slip cases is a separate decision (user).
**Resolved** except that decision — the free-slip + viscosity row is §14. `DFSPH_IMPROVEMENT_PLAN.md`'s
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

## 10. Runner divergence detection misses bounded-NaN-free blowups — RESOLVED 2026-09-28

Moved to [RESOLVED_PROBLEMS.md](docs/historic_plans/RESOLVED_PROBLEMS.md) (velocity
alarm + probe stall defaults; README "Watching a run").

## 11. probe_contactLine delta-SPH toys: pinned dt is overridden — RESOLVED 2026-09-28

Moved to [RESOLVED_PROBLEMS.md](docs/historic_plans/RESOLVED_PROBLEMS.md) (the
toys' adaptive dt is capped at the pinned acoustic Courant 0.3).

## 12. Render thread + grid-interpolated plots crash or hang CRKSPH cases — RESOLVED 2026-09-28

Moved to [RESOLVED_PROBLEMS.md](docs/historic_plans/RESOLVED_PROBLEMS.md)
(found by the §6.3 re-runs; plot hooks that run warp kernels now stay on the
loop thread).

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

## 14. DFSPH + physical viscosity + free-slip walls: the column's velocity alternates sign every step

**What it is (2026-09-28, found by §7's A/B):** `hydrostaticColumn`,
`divergenceFree`, `wallBC=freeSlip`, `nu=0.01`: the mean vertical velocity of
the whole column flips sign every step (±0.015-0.019), with either viscosity
term (projected or Morris); late on a near-wall horizontal band does the same.
Not the viscous timestep limit — halving dt only shrinks it (±0.005-0.01).
Absent with `nu = 0` (free slip) and with no-slip walls. Suspects: the
free-slip boundary-velocity mirror (`computeBoundaryVelocities`) feeding the
viscous term, against the divergence-free solve. Not investigated further.


## 15. CRKSPH on a lattice shock: light-gas columns pile up at the contact, blow-up when a pair reaches r = 0 — blow-up RESOLVED 2026-09-30

**Status: blow-up RESOLVED 2026-09-30 (cause found, fixed, regression test); a benign residual and follow-ups
below.** The mechanism sections further down were written before the cause was known; the "what it is not" list
there stays valid (kernel, `n_h`, limiter constants, gradB, CUDA graphs were all innocent).

**Cause (2026-09-30).** The blow-up is the CRK **artificial viscosity** on a near-coincident approaching pair,
not the pressure force. `modules/crk/accel.py` / `dudt.py` regularised the switch as
`mu = v_hat . eta / (eta . eta + 1e-7 h^2)` with `eta = x/h` **dimensionless**: the regulariser was ~1e-11, so
`mu ~ v/|eta|` was unbounded as a pair closed and `Q = rho (-C_l c mu + C_q mu^2)` gave a pair acceleration
~1e4-6e4 in one right-hand-side evaluation (trapped: the second stage of step 163, pair at r = 0.0026 dx,
|a_visc| = 6.2e4 vs |a_pressure| = 0.12; then u -> -5e5). Frontiere et al. 2017 Eq. (69) has
`eps^2 = 1e-2` in eta^2 units (their standard parameter set). Fix: exactly that (`scalar_t(1.0e-2)`), so the
viscosity vanishes smoothly as `eta -> 0`.

**Result.** Sod 2D same-lattice: blow-up at step 164 -> clean to t = 0.6 (268 steps, total energy 0.35385
exact); the shipped `triplePoint_equalSpacing.py`: blow-up at step 88 -> t = 10.0 in 4700 steps, energy exact
(48.203), vortex roll-up as in the paper's Fig. 25 (videos in `scripts/out_crk2d/examples/export/`); a regression
test (`tests/test_crkCoincidentPair.py`) fails on the old code (non-finite state) and passes now; gradcheck
(`scripts/gradcheck_crk.py`) and `tests/test_physics.py` etc. pass (110). A/B of the change over the 13
compressible cases of the limiter-plan harness (old vs new, `scripts/out_crk2d/sweep/`): Sod L1 rho -0.1 %,
Sedov 0.0 %, Noh -1.7 %, Kidder -2.7 %, linear wave -3.6 %, Yee -0.9 %, hydrostatic max|v| +0.5 %, KH rebound
-34 %, RT +0.7 %, nothing diverges; the one worse number is Gresho (+12-15 % velocity error, KE spin-up
+7.3 % -> +8.2 %), a case CRKSPH_LIMITER_PLAN O4 shows is a knife-edge (2 % threshold shift = 7x spin-up).

**Tested and refuted on the way.** (a) The sampling ratio is not the cause (it only decides when). (b) Frontiere
Eq. (76)'s multi-material mass rule for the density (`m_ij = m_i` across materials; our kernel always uses `m_j`,
`warpSPHCore/crk/crk_density.py`; the core has no material tag) was built by linearity and tried: it makes Sod 2D
blow up *earlier* (step 83), and the user reports it degraded the hydrostatic 2D box, so it stays out. (c) The
equal-spacing hydrostatic box (static mass jump at uniform pressure) is essentially perfect under CRKSPH
(max|v| 1e-5 over t = 3), so the density treatment across a jump is not the problem; the mass-jump trouble is
dynamic, at start-up.

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
only hid the pile-up). (3) Gresho spin-up (+7-8 % KE) and the limiter knife-edge are unchanged
(CRKSPH_LIMITER_PLAN P1).

**What it is.** `compressible/14-triplePoint/triplePoint_equalSpacing.py`
(CRKSPH, nx 256, shipped preset, `dev` @ a28aeb1) is clean for 86 steps
(max|v| 1.2, energy exact) and one step later |v| ~ 1e5, dt collapses, NaN
in the hash-map build (gallery `F_gallery` rc=1). The same thing happens in
the **Sod 2D slab under CRKSPH on the same-lattice sampling** once it runs
past the shipped tLimit (0.15): a 4000-particle reproducer.

**Replicate** (`scripts/probe_triplePointPileup.py`, prints dt / max|v| /
extrema per step and the smallest nearest-neighbour spacing; dumps the last
clean + blown snapshot to an npz and stops at the blow-up):

```
python scripts/probe_triplePointPileup.py --nnEvery 8                    # triple point, cfl 0.3: blows at step 88
python scripts/probe_triplePointPileup.py --case sod2d --nnEvery 20      # Sod 2D, CRKSPH, --no-equalMass: blows at step 164
python scripts/probe_triplePointPileup.py --case sod2d -- --equalMass    # control: clean to t = 0.6 (269 steps)
python scripts/probe_triplePointPileup.py --case sod1d                   # control: clean to t = 0.6 (2157 steps)
python scripts/probe_triplePointPileup.py --nnEvery 8 -- --cflFactor 0.05 --nSteps 600   # survives, same pile-up
```
Everything after `--` goes to the case CLI. The shipped-preset views:
`examples/compressible/14-triplePoint/triplePoint_equalSpacing.py` and
`scripts/out_sodCRKSPH_2026-09-29/run.sh` (Sod 1D / 2D equal-mass / 2D
equal-spacing under CRKSPH at the shipped tLimit 0.15 — all three finish
there: 539 / 67 / 67 steps, too short to reach the collapse; videos under
`export/`, git-ignored).

**Mechanism (measured).** The nearest-neighbour spacing on the light side of
the contact starts shrinking around step 16 (triple point) and roughly halves
every 8 steps; by step 87 there are 228 coincident pairs (456 particles), one
pair per row, both particles *light* (m = 6.9e-5, same v, different u), at
x = 1.3125 and its periodic mirror 12.6875. The blow-up is the step a pair
reaches r = 0 (kernel gradient -> 0; the CRK-corrected gradient keeps a
W(0)(grad A + A B) term). CRK terms are benign until then (cond(m2) <= 2.3,
|B|h <= 2.9, no singular-matrix warning). Sod 2D shows the same thing:
coincident light pairs (m = 2.5e-5) at the contact, x = +-0.7303, with the
dense side (m = 1e-4) expanded behind it.

**Where it forms / what drives it.** Always at the contact discontinuity, on
the light side. The particle-spacing jump across the contact (in the
compressed direction) is what the sampling controls: with the same lattice
on both sides the dense-side gas is 4x (Sod) / 8x (triple point) heavier per
particle, so its x-spacing at the contact ends up ~2.5x the light side's;
with equal-mass sampling (light side coarser from the start) it is ~1.6x and
the run is clean. Triple-point A/B on rho_II: 0.5 / 0.25 / 0.125 blow at step
194 / 136 / 89 -- **the sampling ratio sets *when*, but even rho_II = 1.0
(equal masses everywhere) still blows (step 208, pairs at the entropy
contact x ~ 1.41)**, so the ratio is an amplifier, not the whole cause.

**What it is not (A/B'd, 5-10 s each):** kernel or n_h (B7, Wendland4,
QuinticSpline, n_h 2-5 all blow, at different steps); the CRK viscosity limiter
(`enableCRKLimiter=0`, eta_crit 0 / 0.1 / 1 -- none cure it, eta_crit = 1 is
earlier); the 09-26 gradB fix (reverted in a scratch core: same); CUDA graphs /
pipelined outputs; `calibrateNormalization`.

**Not a clean regression.** The 08-13 code (stored gallery image) and the
08-31 / 09-04 core pairs develop the same pile-up as a hot cluster (rho 8,
umax 10-16, h down to 0.014, c_s 5) that shrinks the adaptive dt to ~3e-4 and
keeps pairs from touching; the 09-04 pair blew at t = 0.248 in one run and
survived to t = 1.9 in a re-run. From 09-19 on it blows every time.

**A smaller dt only hides it.** `--cflFactor 0.05` survives to t = 1.87 and
its state through t = 0.29 matches cfl 0.3 (rho_max 3.37 vs 3.35, umax 3.07,
h_min 0.050), but it carries the same pile-up (484 particles < 0.1 dx from a
neighbour, min separation 0.009 dx vs 0). cfl 0.15 blows at step 168. The
compressible dt is `cfl * h_min / (xi c_s)`: acoustic only -- no bulk/approach
velocity, no separation term -- and h adapts to density, so it cannot see a
column collapse.

**Next, when picked up:** the pair force between the first two light columns
at r -> 0 (CRK gradient at small separation) and the first-light-column
acceleration (a_i ~ m_heavy (P_i/rho_i^2 + P_j/rho_j^2)); then test a
mass-jump-smoothing remedy on the Sod 2D reproducer (cheap: 4000 particles).
Interim for the example (user's call): leave failing, ship `--cflFactor 0.05`
(runs but carries the pairs), or drop the variant (equal-mass is the case's
shipped default and passes). No videos of the collapse itself (crash probes,
~90-160 steps); the frames to compare are the equal-mass gallery output vs a
cfl-0.05 run.

## 16. Monaghan host does not conserve total energy on a strong shock (Sedov) — RESOLVED 2026-09-30

Moved to [RESOLVED_PROBLEMS.md](docs/historic_plans/RESOLVED_PROBLEMS.md). Cause: the Monaghan
viscous heating (`modules/dissipation/wp_dissipation.py`) lacked the 1/2 of
`du_i/dt = 1/2 sum m_j Pi_ij v_ij . gradW_ij`, so it was exactly twice the kinetic energy the
viscous force removes. Fixed; regression test `tests/test_monaghanEnergy.py`.

## 17. Read-Hayfield 2012's entropy-dissipation term does not conserve total energy

Found 2026-09-30 by the AV_PLAN M0c baseline, right after §16's heating fix made the `none` and
Cullen-Dehnen columns conserve to round-off. Not investigated yet.

Monaghan host, `ReadHayfield2012` (alpha 0.2-1.0), total-energy drift `|E - E0| / E0`, vs 0.0000 for
`none` / Cullen-Dehnen on the same runs: Sod 1D 1.8e-3, Sod 2D 6.3e-3, Sod 3D 1.5e-2, Sedov 3D 1.3e-2
(`C_q = 2`: 2.8e-2), KH 1.6e-3, RT 1.2e-3, and **Noh at `C_q = 2`: +23 %** (its post-shock density is
-16 % vs +-0.1 % for `none` / C&D). Gresho and Yee: 0. The only term R&H adds to the energy equation
is `dudt_diss` (`ReadHayfield2012.py`, eqs. 33-35, the SPHS "entropy dissipation"); it is computed per
particle from `(A_i - A_j)` with `rho_j / rho_i` and `rho_ij` prefactors that are not symmetric in
`(i, j)`, so its energy sum `sum_i m_i dudt_i` need not vanish. Whether that is a transcription error
or (as in the original SPHS) a non-conservative design is the first question -- check the paper's
statement on energy conservation before changing anything.

First check: `scripts/probe_monaghanEnergy.py` with `--switch ReadHayfield2012` (extend `pieces()` with
the `dudt_entropy` term; the net `sum m dudt_diss` over a mid-run state is the number).
