# warpSPH — boundary (mDBC) density extrapolation plan

> **ALWAYS run validation cases with `--video` (vispy).** It opens a live window,
> not just a recorded file — the only way to see *where* and *how* a boundary
> extrapolation scheme fails, not just *that* it did. See
> [[export-video-on-validation-runs]] (memory).

## Why this exists

The δ-SPH validation plan (`DELTASPH_VALIDATION_PLAN.md`) has a long-standing
open item: **few-particle, large/thin fluid sheets near a boundary can blow
up.** This document tracks a specific investigation into whether the
MLS-vs-Shepard *hard switch* in the wall-density extrapolation is a plausible
cause, the prototype and production work that came out of it, and where it
currently stands. It is a narrower, code-focused companion to that plan, not
a replacement for it.

## 1. The hypothesis and the toy-problem confirmation

**Hypothesis:** this codebase's boundary-density extrapolation
(`modules/mdbc/density2025.py`, English et al. 2022 Eq. 12) reduces to a
1st-order MLS fit with a 0th-order Shepard fallback for thin/sparse stencils.
Historically (`DELTASPH_VALIDATION_PLAN.md` ~line 2001, `afb6e59`/`80eabb9`)
this was a **hard** `where(wellConditioned, MLS, Shepard)` switch that visibly
stepped at the free-surface contact line; it was fixed there with a smooth
`w = wDet * wN` ramp on `numNeighbors`/`|det(A_g)|`.

Built `scripts/probe_boundaryExtrapolationSwitchDiscontinuity.py`: synthetic
single/pair/triangle/square/line-N boundary clusters, with and without
scatter, sweeping a continuous "how far off perfectly flat is this sheet"
control through the actual `wellConditioned` gate. Findings:

- The hypothesis is **confirmed but refined**: the existing smooth ramp helps
  when the crossing is driven by smoothly-varying conditioning at *fixed*
  neighbour count, but gives **zero protection** when the crossing coincides
  with a *discrete* neighbour entering/leaving support — for a sparse
  stencil, one new neighbour can move `det(A_g)` by 3.5x the ramp's entire
  blend width in a single infinitesimal geometric step.
- `scripts/probe_mdbcSparseFluid.py` (the existing regression guard for this
  exact pathology) has a real ghost-mirror sign bug — its ghosts land deeper
  in the solid instead of mirrored into the fluid, so `nNbMax=0` always and
  its "PASS" is vacuous. **Not yet fixed** — flagged, not actioned.

## 2. Band et al. 2018 as a candidate replacement

`literature/band2018_mls-pressure-boundaries.pdf` ("MLS pressure boundaries
for divergence-free and viscous SPH fluids") formulates the same
value+gradient boundary fit differently: a basis transform around the
neighbourhood's own kernel-weighted centroid decouples the 0th-order Shepard
term (exact, always well-posed) from the gradient block algebraically, so
degrading to Shepard under ill-conditioning is built into the linear algebra
rather than a separate branch.

Prototyped in `scripts/probe_bandMlsPressureBoundary.py` against the same toy
geometries, with **real per-particle noise added** (a clean/noise-free field
is misleading here — a near-singular inversion recovers an exact answer with
no noise to amplify, hiding the actual failure mode). Findings:

- The paper's own literal prescription (plain `torch.linalg.pinv`, "safe
  inversion via SVD") is **worse than the current ramp** — 10-20x worse
  jumpiness across every tested geometry — because `pinv`'s own rtol cutoff
  is itself a hard per-eigenvalue threshold.
- Replacing that with a **Tikhonov-damped eigendecomposition** (smooth
  per-eigenvalue degradation, self-normalized to the block's own eigenvalue
  scale) gives a consistent 1.5-17x improvement over the ramp, across a
  robust λ range (3e-3 to 3e-2).

## 3. Production implementation and what production actually found

Implemented `src/warpSPH/modules/mdbc/densityBand.py`
(`computeMdbcDensityBand`), wired behind
`WeaklyCompressibleSPHConfig.mdbcDensityScheme` (`'ramped'` default,
`'band'` opt-in; `schemes/deltaSPH.py` dispatches on it, and
`scripts/_mdbcDensityHook.py`'s postStep diagnostic hook was updated to match
so it doesn't silently overwrite an A/B run's boundary density with the other
scheme's value). `--mdbcDensityScheme band|ramped` added to
`scripts/probe_deltaSPHMarrone.py` for A/B testing.

Validating this directly against the real dam-break case (not just the toy
problem) found **three more real, distinct bugs/gaps**, each fixed in turn,
before running into what looks like a genuine architectural mismatch:

1. **Float32 catastrophic cancellation via a too-permissive gate.** The
   original `Mb > 1e-12` "has real neighbours" gate is a continuous,
   scale-sensitive test — a boundary particle whose stencil barely grazes the
   kernel's outer edge has `Mb ~ 1e-10` (should read as "no real neighbours,"
   reads as "valid"), and the subsequent `Sxx - dbx*Sx`-style closed forms
   then cancel two already-noise-floor numbers, producing `rho_b` up to 1.40
   at rest (should be exactly 1.0; confirmed pure precision — float64
   reproduces 1.0 to 1e-13 everywhere). **Fixed**: gate on the exact integer
   fluid-neighbour count (`computeLiuMatricesWarp`'s own `neighCounts`)
   instead.
2. **Tikhonov floor collapsing with the row it's meant to protect.** The
   self-normalizing damping scale (`evals.abs().mean()`) assumes at least one
   healthy eigenvalue — true in the toy tests, false for a genuinely
   collinear-in-both-directions 3-particle stencil, where the floor collapses
   right along with the eigenvalues it's supposed to guard against. First fix
   (`Mb * dx^2`, this row's own Shepard weight) was **insufficient** — `Mb`
   for a marginal stencil can itself collapse for the same reason. **Fixed**:
   floor on the *median* `Mb` across the call's gradient-eligible rows, which
   a minority of degenerate outliers cannot drag down.
3. **Reintroduced the m2dbc anti-attraction clamp**, blended smoothly by the
   gradient block's own conditioning (not a hard switch) rather than dropped
   entirely — the unclamped scheme still diverged (t≈0.42 on the nx=35 no-PST
   stress config) specifically running *into tension* (`rho_b` 0.96 → 0.88 →
   ...), exactly what the clamp guards against. This pushed the survival
   point from t≈0.42 to t≈0.56-0.70 depending on config, a real improvement,
   but did not fully fix it.

### 3.1 The architectural tension (user-identified, confirmed)

Band's own claimed advantage — fit directly at the boundary particle, no
ghost node, no separate Taylor-correction transport step — **only holds for
Band's own single-layer (Akinci-style) boundary sampling**, where every
boundary particle sits at the fluid interface. Confirmed directly against a
rendered dam-break frame: this codebase's boundary bands have real depth
(multiple layers into the wall), and particles more than ~1 layer deep have
poor-to-nonexistent fluid overlap when fit at their own position.

Tried moving the fit to the ghost node instead (mirrored near the surface at
every layer, matching `density2025.py`'s own convention) and Taylor-shifting
to the boundary particle via the existing `ghostOffsets`/`ghostIndices`
machinery. This is **not a free win** — it is a real either/or:

- Fit at the boundary particle: no extrapolation error, but poor/absent
  support at depth (the original problem).
- Fit at the ghost + extrapolate: good support everywhere (confirmed —
  neighbour counts jumped from 3-13 to 16-30 at the previously-starved rows),
  but now a well-conditioned, correctly-computed gradient gets multiplied by
  a lever arm that grows with depth (`xtb ~ 0.25`, roughly 9 particle
  spacings, at the failure point) — amplifying a modest, real gradient into
  an unphysical extrapolated value. Confirmed directly: this version
  diverged *earlier* (t≈0.54) than the at-particle version, with a
  perfectly-conditioned matrix (`wCond=1.0`) at the failure — conditioning
  was never the problem for this failure mode, distance was.

This is also not unique to Band: each boundary *layer* has its own ghost
mirrored to an *increasing* depth into the fluid, so `density2025.py`
inherits the same depth-dependent extrapolation-distance exposure — it
already lives with it, and evidently copes with it noticeably better in
practice (see §3.2), likely through some combination of its different
(asymmetric, kernel-value/kernel-gradient-mixed) moment matrix and its
anti-attraction clamp engaging earlier — not because the depth problem is
actually solved there either.

### 3.2 Head-to-head result (nx=35, symplecticEuler, shifting on,
fourtakas2019 DDT — the current recommended/stable configuration, not the
deliberately-adversarial no-PST/un-frozen one)

| scheme | result |
|---|---|
| **ramped** (production) | Full 0.9 s run, clean throughout. `maxVel` peaks at **4.74** at t≈0.73 (the second-impact event), ends at 2.6; density stays in [0.995, 1.003] almost the whole run. |
| **band** (at-boundary-particle fit) | Same event, `maxVel` peaks at **73** — 15x larger — before partially relaxing. |
| **band** (at-ghost fit + extrapolation) | Diverges *earlier*, t≈0.54, at a perfectly-conditioned stencil (`wCond=1.0`) — the extrapolation-distance failure mode from §3.1, not a conditioning one. |

**Current verdict:** Band's centroid-decoupled/Tikhonov-damped moment
construction is a real, validated improvement over the existing ramp *for
the toy problem's failure mode* (thin, marginally-conditioned stencils), but
it does not currently outperform `density2025.py`'s production scheme on the
real dam-break stress test, because the dominant failure mode there turned
out to be a different mechanism (extrapolation-distance amplification at
depth) that neither variant's conditioning-based safeguards address.

## 4. Next session: the missing piece may already be in the literature folder

`literature/1-s2.0-S0045793025003305-main.pdf` — English, Vacondio et al.
2025, *Computers and Fluids* 303:106870 (river flows past bridges) — is
**cited in `DELTASPH_VALIDATION_PLAN.md`'s reference table as a secondary
"application" paper but has never been synced into
`literature/{ABSTRACTS,MANIFEST}.md`/`references.bib`, and its actual
methodology content has not been read until now.** It turns out to be
directly relevant, not secondary:

Its mDBC boundary pressure/density procedure (Eqs. 8-11, its "pressure
cloning" step) is structurally exactly the fix this investigation has been
converging on without knowing it:

1. Eq. (8): fit the density **value** at the ghost node via a corrected
   kernel-sum matrix (same idea as `interpolateLiuLiu`/Band), with a plain
   Shepard fallback when `dp^2 * ||A_g||_inf * ||A_g^-1||_inf > 50`
   (a different conditioning metric than this codebase's `det(A_g)` gate —
   worth comparing).
2. Eq. (9): `P_g = c0^2 (rho_g - rho0)` — pressure at the ghost via EOS.
3. **Eq. (10): `P_b = P_g + rho0 * (g - a_b).n_b * (x_g - x_b).n_b`** — the
   boundary pressure is the ghost pressure **plus an ANALYTIC hydrostatic
   term along the boundary normal**, using the true gravity/acceleration
   field, not a fitted spatial gradient.
4. Eq. (11): `rho_b = rho0 + P_b / c0^2` — back to density via inverse EOS.

This is the key structural difference from everything tried so far: the
ghost→boundary *extrapolation* step uses a **known, noise-free physical
quantity** (gravity) instead of a **fitted, noise-prone spatial gradient**.
It sidesteps both failure modes found in this investigation at once —
there's no fitted gradient to be poorly conditioned (§3, item 1-2), and
extrapolating a known constant over a large lever arm doesn't amplify
anything (§3.1) the way extrapolating a fitted, possibly-noisy gradient does.
`(x_g - x_b).n_b` maps directly onto this codebase's existing
`ghostOffsets` field (`n_b` in the paper = `x_g - x_b` = `-relPos` in this
codebase's sign convention), so this would reuse existing infrastructure,
not require new geometry.

Separately worth noting: `density2025.py`'s own docstring already records
that a hydrostatic-gravity term (from the *original* English et al. 2022
"m2dbc" formulation) **used to be part of the fallback and was deliberately
dropped** (`afb6e59`/`80eabb9`) because it "adds error near a churning
surface." That removal was scoped to the *fallback share only* under the old
hard-switch architecture; it has not been reconsidered since, and the 2025
paper's version (an *additive* analytic term on top of a still-fitted ghost
*value*, not a replacement for the whole fallback) is a different, more
targeted formulation than whatever was dropped — worth distinguishing rather
than assuming they're the same thing.

**Action item for next session (DONE 2026-09-15, see §5):** read Eqs. 8-11
and their surrounding derivation in full, sync the paper
(`english2025`, `literature/english2025_river-flows-past-bridges.pdf`), and
prototype the analytic-gravity extrapolation as a toy problem the same way
Band's centroid decoupling was prototyped in §2 — confirmed below to remove
the depth/lever-arm amplification failure from §3.1, combined with (not
replacing) the already-validated Tikhonov-damped value fit at the ghost
from §2-3.

## 5. §4's prototype, and a real sign bug caught against DualSPHysics' own source

`literature/1-s2.0-S0045793025003305-main.pdf` synced as `english2025`
(`literature/ADDING.md` procedure; `python scripts/check_literature.py`
clean). Full derivation read (Eqs. 1-13, `pdftotext -layout`, since Elsevier
deposits no abstract and OCR'd equations need the surrounding prose to
disambiguate subscripts).

### 5.1 The paper's printed Eq. (10), read literally, has the wrong sign

`pdftotext`'s extraction of Eq. (10) reads (subscripts as rendered):

    P_b = P_g + rho0 * dot(g - a_b, n_b) * dot(x_g - x_b, n_b)

Integrating the paper's own governing equation (Eq. 2, `du/dt = -grad(P)/rho +
Gamma + g`) at hydrostatic equilibrium along the boundary normal gives
`dP/dy = rho0 * g_y`, which — for a ghost strictly further from the wall than
the boundary particle along `+n_b` — requires `P_b > P_g` when gravity
opposes `n_b` (deeper = higher pressure). The literally-extracted Eq. (10)
gives the opposite sign. Rather than trust a hand-parsed two-column PDF
equation over a physical derivation, cross-checked directly against the
paper's own reference implementation: `~/dev/DualSPHysics/src/source/
JSphCpu_mdbc.cpp`'s `Mdbc2PressClone` (the exact "m2dbc pressure cloning"
function this paper's Eqs. 8-11 describe). Its actual code:

    dpos      = boundaryPos - ghostPos                    // = this codebase's relPos
    normal    = boundnor / |boundnor|                     // boundnor = ghostPos - boundaryPos
    normpos   = dot(dpos, normal)                          // = -|dpos|, by construction
    normforce = rho0 * dot(gravity - motionAccel, normal)
    P_b       = P_g + normforce * normpos

Substituting `normpos = -|dpos|` and `normal*|dpos| = -dpos` collapses this
to, with no unit-normal left over:

    P_b = P_g + rho0 * dot(g - a_b, relPos),   relPos = x_b - x_g

which matches the hydrostatic-ODE derivation exactly, and reuses this
codebase's OWN existing `relPos` convention (`density2025.py`'s
`relPos = -currentState.ghostOffsets[ghostMask]`) verbatim — no new geometry,
no unit-normal computation needed, because the ghost already sits exactly
along the true normal by construction (both here and in DualSPHysics).
**Likely cause:** a subscript pair transposed by the PDF's equation renderer
(a documented `pdftotext` failure mode for stacked/subscripted dot products,
not a paper error — never checked against the typeset PDF image itself, only
the text layer). Recorded here rather than silently fixed, since anyone
re-deriving this from the PDF text alone will hit the same wrong sign.

### 5.2 Depth-sweep prototype (`scripts/probe_englishAnalyticGravityExtrapolation.py`)

One flat-wall geometry, 16 boundary layers (ghost depth into the fluid grows
linearly with layer index, exactly §3.1's mechanism — lever arm swept 1-31
`dx`), dense fluid block, exact hydrostatic field `rho(y) = rho0 + rho0 g_y/
c0^2 y` (so the analytic term's own assumption is exactly satisfied) plus IID
per-particle noise. Four variants at the SAME boundary particles across the
SAME depth sweep, mean `|rho_b - rho_true|` per layer:

| variant | shallow-quartile err | deep-quartile err | ratio |
|---|---|---|---|
| `english2022` (production, fitted grad, Liu/MLS value) | 6.2e-4 | 4.0e-3 | **6.5x** |
| `band` (production, fitted grad, Tikhonov value) | 8.0e-4 | 4.0e-3 | **5.1x** |
| analytic Eq. (10) + Liu/MLS ghost value | 1.06e-1 | 1.05e-1 | 1.0x (flat but **bad**) |
| analytic Eq. (10) + Band's Tikhonov ghost value | 2.1e-4 | 2.6e-4 | **1.2x** |

Confirms the hypothesis directly: both existing fitted-gradient paths grow
error 5-6.5x from shallow to deep layers (reproducing §3.1's mechanism on a
clean synthetic case, not just the Marrone stress config); the analytic
term, paired with Band's already-validated Tikhonov value fit, stays flat
(1.2x) at roughly the noise floor — confirmed at `fieldNoise` in {0, 0.001,
0.005}, and the fitted-gradient methods still grow 10.9-11.2x even at
**zero** injected noise (a correctly-computed gradient's own discretization
error, not just noise, is what the lever arm amplifies — matches §3.1's
"a well-conditioned, correctly-computed gradient" wording exactly).

`analytic + Liu/MLS value` is flat but ~400x worse in absolute terms than
`analytic + Band value` — diagnosed, not a depth effect: a consistent 10.5%
of ghosts get `rho_interp = 0` at EVERY depth (`wellConditioned` fraction is
depth-independent), including rows with 7-10 fluid neighbours, well past
`neighbor_threshold=4`. This is `interpolateLiuLiu`'s coupled `(dim+1)x
(dim+1)` matrix hitting a lattice-conditioning degeneracy on this synthetic
grid's exact-`dx` regularity, evaluated here RAW (no production ramp/blend
applied) — it reinforces §2's finding that Band's centroid-decoupled Shepard
term is unconditionally well-posed given >=1 neighbour, in a new setting
(regular-lattice ghost stencils, not just adversarial scatter).

**Net conclusion:** English 2025's analytic-hydrostatic extrapolation,
combined with Band's Tikhonov-damped value fit at the ghost, is a validated
fix for the §3.1 depth/lever-arm failure mode in the toy problem — the
specific failure mode that stopped the Band prototype from beating
production on the real dam-break stress test. **Not yet validated on the
real dam-break case itself** (§3.2's head-to-head); that is the next step,
not this one.

### 5.3 What was NOT yet resolved at end of session (superseded, see §6)

- The analytic term only ever cancels `n_b` because the ghost lies exactly
  along the true normal by construction; a boundary geometry where that is
  not exact (curved surfaces, corner ghosts) was not tested.
- No production wiring yet — this is a standalone toy script, not a third
  `mdbcDensityScheme` option. Wiring it needs: (a) a real per-particle
  gravity/boundary-acceleration field (`configurations/moduleConfigurations/
  gravity.py`, already imported by `density2025.py` but unused there), (b) a
  fallback ladder for the <10% of ghosts with zero neighbours (the analytic
  term still needs SOME ghost value estimate to extrapolate from), (c) a
  decision on whether `a_b` should ever be nonzero here (only matters for
  moving/rigid-body boundaries, not the static-wall dam-break case).
- Still needs the §3.2 head-to-head against `density2025.py`'s production
  `'ramped'` scheme on the actual Marrone dam-break stress config (nx=35,
  no-PST, un-frozen) that stopped Band's own gradient-based candidate at
  t≈0.42-0.70 — this toy result predicts it should survive materially
  longer, but that is a prediction, not yet a measurement.

## 6. Production wiring and real-case validation (2026-09-15)

Implemented `src/warpSPH/modules/mdbc/english2025.py`
(`computeMdbcDensityEnglish2025`): Band's decoupled 0th-order Shepard value
fit at the ghost (no gradient block -- this scheme fits none), `a_b = 0`
(static walls only -- no per-particle rigid-body boundary acceleration is
tracked anywhere in this codebase yet), gravity read via the real
`modules/gravity/wrapper.computeGravity` (handles `Directional`/
`PointSource`/`PotentialField`/inactive generically, not hardcoded to a
constant vector like the toy script). Wired as `mdbcDensityScheme='english2025'`
end to end: `WeaklyCompressibleSPHConfig.mdbcDensityScheme`,
`schemes/deltaSPH.py`'s dispatch, `scripts/_mdbcDensityHook.py`'s postStep
diagnostic hook, and `--mdbcDensityScheme english2025` on both
`scripts/probe_deltaSPHMarrone.py` (already had the `ramped`/`band` switch)
and `scripts/probe_deltaSPHMarrone34.py` (added the same override pattern,
previously missing entirely).

### 6.1 Marrone 3.1, adversarial stress config (nx=35, `--shifting off`, un-frozen)

The exact config that made `'band'` diverge at t≈0.42-0.70 (§3, item 3).
All three schemes run head-to-head, `--tLimit 1.2`:

| scheme | outcome | maxVel peak | maxVel end | density range (whole run) |
|---|---|---|---|---|
| `ramped` (production) | survives to t=1.2 | **13.9** @ t=0.66 | 2.61 | [0.887, 1.109] |
| `band` | **diverges at t=0.74** | NaN | NaN | blows to [0.05, 2.6e11] |
| `english2025` | survives to t=1.2 | **4.83** @ t=0.78 | 2.17 | **[0.994, 1.006]** |

Confirms `band`'s documented divergence exactly (t=0.74, inside the
recorded 0.42-0.70 window's tail). `english2025` not only survives the full
run where `band` diverges -- it is MORE stable than the current production
default here: maxVel peaks at under a third of `ramped`'s spike (4.8 vs
13.9), and the density excursion over the whole run is roughly 15x tighter
than `ramped`'s ([0.994,1.006] vs [0.887,1.109]). This is the first time in
this investigation that a candidate scheme has beaten `'ramped'` on its own
adversarial stress test rather than merely matching it.

### 6.2 Marrone 3.1, stable/recommended config (nx=35, symplecticEuler, shifting on, fourtakas2019 DDT)

Extends §3.2's table (whose `ramped`/`band` rows are unchanged, reused
as-is):

| scheme | result |
|---|---|
| `ramped` (production) | maxVel peaks **4.74** @ t≈0.73, ends 2.6; density in [0.995, 1.003] |
| `band` | maxVel peaks **73** |
| `english2025` | maxVel peaks **4.48** @ t=0.78, ends 2.56; density in [0.992, 1.004] |

On the config where production was already stable, `english2025` matches it
essentially exactly (a marginally lower peak) -- no regression on the easy
case, in addition to the large win on the hard one in §7.1.

### 6.3 Marrone 3.4 (nx=128, tStar=3.0, `deltaSPH`, shifting off)

All three schemes: 6/6 PASS (`scripts/probe_deltaSPHMarrone34.py --report`)
-- runs to target t*, weakly-compressible bulk density, no velocity
divergence, no tank/obstacle penetration, kinetic energy not running away.
No stability differentiation at this resolution/config (unsurprising: the
depth/lever-arm mechanism §3.1 targets is most exposed on a violent,
fast-moving front like 3.1's, not this obstacle case's milder overtopping
at t* <= 3). Surface pressure peaks differ somewhat between schemes
(`english2025` P1 116.5 vs `ramped` 76.8 vs `band` 137.5, Wagner reference
~36.7) but this is a `t*=3` snapshot, not the calibrated `t*=15.66`/`nx=256`
protocol the existing `sun2017DeltaSPH` reference rows in `REPORT.md` use --
not a like-for-like accuracy comparison, only a stability smoke test.
**Open:** repeat at the full calibrated resolution/duration if a real
accuracy comparison (not just stability) is wanted here.

### 6.3b Marrone 3.4 at 3x the duration (nx=128, tStar=9.0, `ramped`/`english2025` only)

At `t*=3` the two schemes looked interchangeable (§6.3) -- user-requested
`t*=9` (`band` skipped) uncovered a real difference the short run hid.
`scripts/probe_deltaSPHMarrone34.py --report`:

| scheme | checks | maxVel peak | maxVel end | density (whole run) | dKE/dt* (2nd half) | P7/P8/P9 peaks |
|---|---|---|---|---|---|---|
| `ramped` | **5/6 -- FAIL kinetic energy** | 53.3 @ t*=7.9 | 20.9 | [0.688, 1.388] | **+2.66** (growing) | 39.7 / 134.4 / 210.8 |
| `english2025` | **6/6** | 32.7 @ t*=2.1 (same early jet-tip event as the short run) | 5.7 | [0.744, 1.113] | -0.45 (decaying) | 6.1 / 8.8 / 9.9 |

`ramped` develops a large LATE secondary-impact event (t*≈7.9, well past
the `t*=3` window §6.3 checked) that the short run could not see: a bigger
velocity spike than even its own early jet-tip, a wider density excursion,
growing (not decaying) kinetic energy in the second half -- an actual
`--report` FAIL, not just a worse number -- and probes P7-P9 (the
downstream/lee-side taps) overshooting the Wagner reference (~36.7) by
4-6x. `english2025` shows no such event: its peak stays the same early
jet-tip already seen at `t*=3`, kinetic energy decays normally throughout,
and P7-P9 stay within a few multiples of the reference. Wagner's own
reference (`sun2017DeltaSPH`, `nx=256`) surface-pressure table (§6.3) also
shows P7-P9 in the tens, not hundreds, of `P/(rho g H)` -- `ramped`'s
values here look like a genuine artifact of this specific
resolution/duration combination, not the physical signal.

This reframes §6.4's "at least as good as ramped everywhere tested" --
at longer duration, on this case, `english2025` is not just competitive
but avoids a failure mode `ramped` actually has.

### 6.3c Both cases at 2x resolution and full duration (user-requested robustness check)

`english2025` only, doubling resolution AND extending duration together --
does the fix hold up away from the exact settings it was validated at?

- **Marrone 3.1** (nx=70, was 35; `--tLimit 5.0`, was 0.9s, same stable
  config: symplecticEuler, shifting on, fourtakas2019 DDT). `diverged=False`
  through the full 5 s. maxVel peaks **5.64** @ t=0.70 (same order as the
  4.5-4.8 seen at nx=35), ends at 1.39, fully settled by t=5
  (`rho` in [0.9999, 1.0003]). Whole-run density range widens to
  [0.977, 1.022] (vs [0.992, 1.004] at the shorter nx=35/t=0.9s window) --
  expected, since this run now covers 5.5x more of the later-time dynamics
  the shorter snapshot never reached, not a sign of degradation.
- **Marrone 3.4** (nx=256, was 128; `--tStar 15.6605`, i.e. `t=5s` -- this
  conversion lands almost exactly on the SAME full duration already used
  by the pre-existing `sun2017DeltaSPH_nx256` reference rows in
  `REPORT.md`, letting this run reuse that established protocol rather
  than inventing a new one). `--report`: **6/6 PASS**, `dKE/dt*` (2nd half)
  **-1.10 (decaying)** -- the same check `ramped` FAILED at nx=128/this
  exact duration in §6.3b. Doubling resolution did not reintroduce
  `ramped`'s late-time growing-energy failure mode at the same duration;
  surface pressure peaks (P1 48.2, P2 62.3, P3 106.5, P7-P9 in the 20s-30s)
  are all moderate, no 100+ overshoots like `ramped` showed at nx=128.

Confirms the fix is not narrowly tuned to the specific resolutions/durations
validated so far -- both cases remain stable and well-behaved at 2x
resolution and full/extended duration simultaneously.

### 6.3d Paired `ramped`/`band` re-run at the SAME 2x resolution (closing the §6.3c gap)

§6.3c ran `english2025` alone at nx=70/nx=256. User-requested: re-run
`ramped` and `band` at the identical settings for a real three-way
comparison, not `english2025`-at-2x against the others' original-resolution
numbers.

**Marrone 3.1 (nx=70, t=5s, stable config):**

| scheme | maxVel peak | density range (whole run) |
|---|---|---|
| `english2025` (§6.3c) | 5.64 @ t=0.70 | [0.977, 1.022] |
| `ramped` | 11.66 @ t=0.74 | [0.915, 1.091] |
| `band` | 22.20 @ t=2.91 (late, not the jet-tip) | [0.812, 1.076] |

The ordering from nx=35 (§6.1/6.2) holds exactly at nx=70: `english2025` <
`ramped` < `band`, and the gaps are, if anything, larger in absolute terms.

**Marrone 3.4 (nx=256, t*=15.66):**

| scheme | outcome | maxVel peak | dKE/dt* (2nd half) |
|---|---|---|---|
| `english2025` (§6.3c) | 6/6 PASS | 32.4 | -1.10 (decaying) |
| `ramped` | 6/6 PASS | **60.6** (9.9x U_max) | -1.46 (decaying) |
| `band` | **diverges at t*=4.21/15.66** | 1.76e6 | NaN |

`band` catastrophically fails here -- density blows to 2.5e7 (pointwise
2.2e31), wall penetration 195,730 dx. Doubling resolution did not just fail
to fix `band`'s nx=128 weakness (§6.3), it made it categorically worse:
survives-but-worst at nx=128, outright diverges at nx=256. The frame at the
divergence (`deltaSPH_nx256_c28.3_mdbcRho-band_output.mp4`) shows the exact
mechanism this investigation has targeted since §3.1, visually, for the
first time: a handful of boundary/ghost particles near the left wall eject
violently, and because SPH is local, their now-garbage state poisons
neighbouring particles' kernel sums the next step -- a cascade, not an
isolated bad sample. A fitted, noisy/marginal gradient at a deep boundary
layer is exactly the kind of error `english2025`'s analytic term has no
mechanism to produce.

**`ramped`'s §6.3b failure does NOT reproduce at nx=256** -- it now PASSES
the kinetic-energy check (dKE/dt* -1.46, was +2.66 at nx=128) at the SAME
t*=15.66 duration. That specific failure mode looks like a resolution
artifact of nx=128, not a durable property of `'ramped'`'s mDBC scheme --
§6.3b/6.4's framing of it as a scheme-inherent advantage for `english2025`
overstated the case; **retracted below**.

That said, `ramped` is not clean at nx=256 either: `maxVel=60.6` (9.9x
`U_max`) is real, and a user-provided frame from this exact run shows a
stray ejected particle and a visibly disordered "wormy" cluster at a
corner -- a local blow-up signature bulk kinetic energy does not have to
reflect (a few fast particles carry little of the total KE). **The
`--report` velocity check is currently `report-only below 12 U_max`, not a
hard fail** -- `ramped`'s 9.9x reading passes that threshold nominally but
is visibly a real defect; this check is too loose and should be tightened
(§7) rather than trusted at face value. Also found in this run: the KE
check on `band`'s diverged trace printed **PASS** with `dKE/dt* nan, KE_end
nan` -- a NaN comparison silently defaulting to "not failing" rather than
being treated as a hard fail. Both are `scripts/probe_deltaSPHMarrone34.py`
`--report` scoring bugs, not modelling issues; not yet fixed (§7).

### 6.4 sloshingTank, first attempt with the full combo (english2025 + PST + fourtakas2019 DDT)

User-requested: does this combination -- functionally close to running
DualSPHysics' own defaults inside this codebase (`fourtakas2019` IS
DualSPHysics' `DDT_DDT2`; `english2025`'s mDBC term was verified line-by-line
against DualSPHysics' actual `Mdbc2PressClone`) -- look like a strong
default candidate on a case outside the Marrone dam-break family?

`examples/sloshingTank/run_sloshingTank.py` did not have `--mdbcDensityScheme`/
`--densityDiffusionTerm` overrides; added them (same pattern as the Marrone
scripts), plus a file-naming tag so an A/B run doesn't overwrite the plain
scheme's own recorded output. Run at the case default `nx=200`, full
reference duration `--tLimit 7.0` (shifting/PST is already on by default in
this script), `--mdbcDensityScheme english2025 --densityDiffusionTerm
fourtakas2019`.

**Result: `diverged=False`, ran the full 7 s -- but NOT a clean pass.**
Through most of the run (t=0-5.5 s) density and sensor pressure look
reasonable and physically sensible: density in roughly [0.99, 1.03], sensor
pressure peaks growing gradually with each roll cycle's impact (189 -> 717
-> 975 -> 1546 Pa), consistent with a resonant tank building amplitude over
cycles. From t≈5.66 s a real cascade unfolds -- user-identified from the
rendered frames, then confirmed by binning the trajectory at 0.05s
resolution:

  1. **t=5.65-5.70 s:** an isolated sensor-pressure spike, 98,177 Pa (7.5x
     the measured impact-peak band's own upper bound of 13,119 Pa) --
     the wall/corner region visibly shows a particle pinned at the free
     surface near the corner (the ceiling-hover pattern from
     [[antuono-pressure-switch-bug]]/[[sph-symmetric-pressure-truncation-artifact]]:
     an isolated particle held by an under-conditioned mDBC fallback, a
     different mechanism than the depth/lever-arm one this scheme
     targets -- `english2025` was never going to suppress this).
  2. **t=5.70-6.20 s:** moderate wall-region churn (density dips to ~0.89,
     pressure fluctuating in the 1000s-2000s of Pa) as an upwash sheet
     climbs toward that pinned particle.
  3. **t=6.25-6.40 s:** a second precursor spike (18,598 -> 3,738 Pa) as
     the sheet reaches it.
  4. **t=6.40-6.70 s:** rapid escalation -- minRho falls 0.86 -> 0.66,
     maxVel climbs to **16.4** at t≈6.67 -- the collision catapults the
     pinned particle out into the bulk ([[sloshing-rightwall-thin-sheet-kick]]'s
     mechanism, here triggered at the corner/ceiling rather than the
     right wall).
  5. **t=6.70-6.90 s:** minRho sits FROZEN at exactly 0.6509 for ~0.2s --
     the now-ejected particle, disconnected from the fluid bulk, has no
     SPH neighbours left to update its density estimate from as it
     flies through empty space.
  6. **t=6.90-7.00 s:** relaxes back toward normal (density [0.989, 1.011]
     by t=7.0).

So this is not a runaway divergence, and not two disconnected events --
it is one real, large-amplitude cascade this scheme combination did not
avoid, running the already-documented wall-pinning/thin-sheet-kick
mechanism to an unusually violent conclusion.

**Not yet resolved:** no `'ramped'` (or `'band'`) baseline has been run at
these exact settings, so it is not yet known whether this cascade is
(a) specific to `english2025`, or (b) a known sensitivity of this exact
case regardless of mDBC scheme -- [[sloshing-rightwall-thin-sheet-kick]]
(memory) already documents a *different*, previously-adjudicated
right-wall pressure-force spike (100-1000x, near a wall, as fluid-neighbour
count drops) on this same case as "genuine sensitivity, not a resume
defect," under whatever scheme was default at the time. Whether THIS event
is the same phenomenon, a new one, or an `english2025`-specific issue is
open. Also notable: [[stability-check-run-length]] (memory) records that
this case's stability checks normally stop at t=3 ("2nd impact resolved"),
short of both this event (t=5.66-6.8) and Marrone 3.4's own late failure
window (§6.3b, t*≈7.9) -- the pattern of "a longer run than the usual
stability-check window surfaces a real event" is now three-for-three across
Marrone 3.1 (implicitly, via the resolution/duration sweep), 3.4, and
sloshingTank.

**Action item, done (2026-09-15, for posterity -- user's own expectation
going in: these events are timing/coincidence-sensitive, "unlikely to line
up" between schemes).** Ran `'ramped'` at identical settings (nx=200,
t=7s, PST on, fourtakas2019 DDT): `diverged=False`, ran the full 7s, but
**not clean either -- a different failure character, not a cleaner run:**

| | `ramped` | `english2025` |
|---|---|---|
| pattern | many isolated pressure spikes, nearly every wave impact (t≈2.3-3.0, 3.9-4.1, 4.5-4.6, 5.5-5.6, 6.1-6.2) | mostly quiet; ONE escalating cascade (t=5.66-6.9) |
| peak pressure | **158,822 Pa** @ t=6.17 (higher than english2025's) | 98,177 Pa @ t=5.66 |
| density range (whole run) | [0.857, 1.288] -- bounded, no ejection | [0.651, 1.715] -- an actual particle ejection |
| maxVel peak | ~5.1 | 16.4 |

So `'ramped'` shows MORE frequent large-magnitude pressure-sensor spikes
(consistent with [[sloshing-rightwall-thin-sheet-kick]]'s already-adjudicated
"genuine sensitivity, not a defect" pattern -- brief high-magnitude spikes
without a density catastrophe) but NEVER escalates into a sustained
density/velocity blowup the way `english2025`'s single event did. Neither
scheme is a clean pass on this case; the CHARACTER of the instability
differs (frequent-but-contained vs. rare-but-severe) rather than one
scheme being straightforwardly worse. This is consistent with the case
itself carrying inherent, timing-sensitive free-surface sensitivity
regardless of `mdbcDensityScheme` -- neither confirms nor rules out
`english2025`-specific involvement in ITS particular cascade, but it does
rule out "`'ramped'` is clean here and `english2025` broke something" --
`'ramped'` was never clean on this case either.

**A separate, smaller discrepancy, unrelated to the above (Sensor-1
pressure-history comparison, `t < 5.6s`, before the cascade):** overall
agreement with the measured record is good, with one exception --
smoothed-peak comparison (`scipy.signal.find_peaks` on a Gaussian-smoothed
trace, matching the comparison plot's own smoothing) gives:

| impact | measured | simulated |
|---|---|---|
| 1st (first significant impact) | t=2.39, 1805 Pa | t=2.36, 3056 Pa |
| 2nd (the mismatch) | t=2.71, **838 Pa** (decays) | t=2.93, **2992 Pa** (does not decay), ~0.22s late |
| 3rd | t=4.07, 2346 Pa | t=4.06, 8984 Pa |

User-diagnosed from the rendered frame at this second peak: the splash
that climbed the wall after the first impact falls back down and, given
how the bulk was moving at that moment, collides with an oncoming sheet
head-on rather than sliding smoothly back along the boundary -- a
phase/timing-sensitive kinematic collision between two bulk fluid masses,
not a wall-density-extrapolation effect. Unlike the t=5.66-6.9s cascade
(a particle *pinned at the boundary* by an under-conditioned mDBC
fallback, §6.4 above), this is a bulk-flow collision in free space that
`mdbcDensityScheme` has no real mechanism to influence either way --
assessed as plausibly a fragile coincidence of surface friction/viscosity/
resolution rather than an mDBC effect, and out of scope for this
investigation to chase further.

### 6.5 Net conclusion (revised)

`english2025` remains the most consistently well-behaved scheme across
every test run so far: lowest peak velocities and tightest density/pressure
excursions on both Marrone cases at both resolutions tested (§6.1, §6.2,
§6.3c, §6.3d), and the only one of the three that never diverged anywhere
in this investigation, including where `band` now catastrophically fails
at 2x resolution (§6.3d) -- direct visual confirmation of the depth/lever-
arm-amplification-into-a-cascade mechanism §3.1/§5 theorized. However,
two findings from this session materially calibrate that picture down from
the prior "substantially better on two separate axes" framing (§6.4,
retracted):

1. `'ramped'`'s §6.3b kinetic-energy FAILURE at nx=128/t*=15.66-equivalent
   duration did not reproduce at nx=256 -- likely a resolution artifact,
   not a durable scheme defect `english2025` uniquely avoids. (`'ramped'`
   still shows a real, un-caught-by-`--report` blow-up signature at
   nx=256 -- elevated `maxVel`, visible particle ejection -- so the case
   for `english2025`'s superior robustness on Marrone 3.4 stands, just not
   for the specific reason previously claimed.)
2. sloshingTank with the full candidate-default combo is **not a clean
   pass** -- a real, large-amplitude transient at t=5.66-6.8s, un-compared
   against any baseline yet.

**Verdict on "strong contender to be the default":** the Marrone evidence
(§6.1-6.3d) is now quite strong and holds across two resolutions each --
`english2025` has not lost a single head-to-head there. sloshingTank
(§6.4) is the first case where the picture is genuinely unclear, and is
exactly the kind of gap (a case outside the family the fix was developed
against) this investigation has been burned by before (`band` looked good
in isolation too). **Do not flip the default** until the sloshingTank
`'ramped'` baseline closes that gap.

## 7. Also found along the way (real, unrelated bugs)

- **`scripts/probe_deltaSPHMarrone34.py --report`'s velocity-divergence
  check is `report-only below 12 U_max`, not a hard fail** -- a run with a
  visible local particle-ejection blow-up (§6.3d, `'ramped'` at nx=256,
  `maxVel` 9.9x `U_max`) still scores a nominal PASS. The kinetic-energy
  check on the SAME script also scored **PASS on a NaN trace**
  (`dKE/dt* nan` from `'band'`'s divergence, §6.3d) -- a NaN comparison
  silently not triggering the FAIL branch. Neither fixed yet; both should
  be tightened (a real threshold on the velocity check; NaN treated as an
  explicit FAIL) before trusting a 6/6 `--report` score at face value on a
  borderline run.

- **`warpSPHPlotting`'s vispy `_auto_point_size` used total particle count
  over domain area to estimate marker diameter** — systematically wrong
  whenever boundary+ghost particles (perimeter-distributed, often
  outnumbering fluid several-to-one) dominate the count while filling far
  less area, giving markers ~1.4x too large in length (~2x in on-screen
  area — matching the reported "twice as large" visual impression).
  **Fixed**: use the real particle spacing (`sqrt((masses/densities).mean())`,
  the same quantity the matplotlib backend already threads through) instead
  of estimating it from particle count.
- Density field colormap switched from `RdBu` (white midpoint, hard to see
  against small markers) to `managua` (dark midpoint) for
  `cases/dambreak.py`'s density field, per direct user feedback.

## 8. Ceiling-normal-vs-gravity clamp attempt (2026-09-16): a confirmed regression, not a fix

§6.4's sloshingTank cascade (t=5.66-6.9s) starts with a particle pinned at the
free surface near the ceiling/corner -- an under-conditioned mDBC state
"treated as real" pressure, the same family as
[[antuono-pressure-switch-bug]]'s ceiling-hover pattern. User-proposed
hypothesis: `english2025`'s Eq. (10) analytic term,
`rho0 * dot(g - a_b, relPos)` (`relPos = x_b - x_g`, pointing along the
wall's outward normal by construction), is signed exactly by
`cos(angle(gravity, outward normal))` -- positive for a floor (gravity
presses fluid against the wall, a real hydrostatic column justifies the
term) and negative for a ceiling/overhang (gravity pulls fluid AWAY from the
wall, no column exists to justify it). A thin residual film at a ceiling has
no bulk column behind it; applying the raw negative term there manufactures
spurious tension that could be exactly what pins it in place. Proposed fix:
don't apply the hydrostatic push when wall normal and gravity are more than
90 degrees apart.

### 8.1 Implementation

`.clamp_min(0.0)` on the `dot(g_b, relPos)` term before use (code reverted
after the finding below; was `src/warpSPH/modules/mdbc/english2025.py`
~lines 123-129). Chosen over a boolean mask specifically because a hard
switch was expected to chatter -- `BOUNDARY_DENSITY_PLAN.md`'s own history
(§1) already found a hard `wellConditioned` switch visibly stepping at a
free-surface contact line, and the reasoning at the time was that
`clamp_min` is continuous through `dot==0`, and that side/vertical walls
have `dot≈0` "already, by construction" so the clamp would be a pure no-op
for them and only change the ceiling.

**That "no-op for side walls" assumption is the bug** (§8.2) -- it silently
assumed a static tank. `sloshingTank` rolls.

### 8.2 Validation: deterministic early divergence, not the targeted fix

Ran the exact §6.4 comparison command twice, independently:
```
python examples/sloshingTank/run_sloshingTank.py --tLimit 7.0 \
  --mdbcDensityScheme english2025 --densityDiffusionTerm fourtakas2019 --video
```
Both runs **diverged at the identical step 14135, t=1.4135s** -- traces
bit-identical to ~15 significant figures across the two independent runs
(one under heavy unrelated GPU contention from two external `llama.cpp`
processes at 100% util, one solo), ruling out hardware/driver noise as the
cause. This is **4+ seconds before** the cascade window the fix targets --
the run never reaches t=5.66s, so the original question (does the clamp
suppress the cascade?) is untested and untestable with this implementation.

The failure itself has no visible precursor: the 5 timesteps immediately
before are fully nominal (`maxVel` climbing smoothly 0.5965->0.5970, density
and pressure unremarkable), then an instantaneous jump to `maxVel=NaN` with
`minDensity` snapped to exactly 1.0 (looks like a NaN silently clamped
upstream) at the very next recorded step. The last rendered frame before
failure (t=1.41s) shows a calm, shallow sloshing layer -- no pinned
particle, no ejection, nothing resembling the original cascade's character
(which built up visibly over ~1.2s). This reads as a discrete numerical
edge case triggered at a specific instant, not a physical instability.

Baseline at the identical step is completely nominal and sails on to t=7:
```
step 14135, t=1.4135s:
  baseline (pre-fix): minD=1.00003  maxD=1.00272  maxVel=0.599  sensorP=23.4 Pa
  fixed (both runs):  minD=1.00000* maxD=1.00335  maxVel=NaN    sensorP=81.4 Pa
```

Marrone 3.1 regression check (§6.1's adversarial no-PST config,
`--nx 35 --shifting off --tLimit 1.2`) showed **no regression** --
`diverged=False`, maxVel peak 4.29 vs the recorded 4.83, density
[0.989,1.010] vs the recorded [0.994,1.006] (small, plausibly this
codebase's documented run-to-run chaotic sensitivity). Consistent with the
diagnosis below: Marrone 3.1's tank is open-top and does not roll, so
`dot(g, relPos)` is >=0 by construction there and the clamp genuinely is a
no-op on that case.

### 8.3 Diagnosed mechanism: the clamp's gate is in the wrong reference frame

`sloshingTank` prescribes a time-varying roll angle -- confirmed directly
from rendered-frame metadata that the gravity vector in the tank/simulation
frame rotates over time (`g=(0.423,-9.80)` at t=1.16 vs `g=(0.693,-9.79)` at
t=1.41). Over a roll cycle, EVERY wall's outward-normal-to-gravity dot
product sweeps through zero, not just the ceiling's -- a side wall sits
exactly at the `dot=0` threshold in the tank's rest orientation, so any roll
at all pushes it to one side or the other, and it crosses back as the roll
reverses. The `clamp_min` fix's own justification (§8.1, "no-op for side
walls... by construction") implicitly assumed the wall/gravity geometry is
static; it silently breaks the moment the geometry rotates relative to
gravity, which is `sloshingTank`'s entire premise as a case.

Per-particle, `clamp_min` is still continuous in time (no jump in the
per-particle value). The likely amplification: because roll angle changes
nearly uniformly along the boundary, many boundary/ghost pairs along the
SAME wall cross `dot=0` in close temporal proximity, so the hydrostatic
term's contribution along an entire wall segment can collapse from a
meaningful value to zero within one or a few timesteps -- effectively a
near-simultaneous, wall-wide change in boundary pressure that a calm,
low-dynamic-pressure regime (t~1.4s, well before the first wave impact at
~t=2.36s per the baseline record) has no comparable signal to absorb. This
is a hypothesis, not root-caused to a specific particle/NaN source --
**not yet instrumented at step 14135 to find the first non-finite value.**

### 8.4 Disposition

**Code reverted** (`git checkout -- src/warpSPH/modules/mdbc/english2025.py`)
-- the clamp as implemented is a confirmed regression, not a candidate fix,
and must not ship. Evidence preserved for a future attempt:
- Baseline (pre-fix, unchanged): `examples/sloshingTank/output/baseline_precap/wcsph_mdbcRho-english2025_ddt-fourtakas2019_series.npz`
- Diverged run 1 (backed up before being overwritten by run 2, same failure): `examples/sloshingTank/output/fixed_run1_diverged_t1.41/` (series.npz, field.mp4/gif, run.log)
- Diverged run 2 (current contents of the main output path at time of writing, identical failure): `wcsph_mdbcRho-english2025_ddt-fourtakas2019_series.npz` / `_field.mp4` / `_field.gif`

The underlying hypothesis (§8's opening paragraph -- a ceiling has no bulk
column to justify the analytic hydrostatic term) is not refuted by this
result; only THIS implementation of it is. **Next attempt should gate on a
tank/body-frame-fixed notion of "which wall is the ceiling"** (e.g.
classify each boundary particle's wall membership once, from its rest-state
geometry, rather than recomputing `dot(g, relPos)` fresh every step against
the instantaneously-rotated `relPos`) so a few degrees of roll cannot flip a
side wall's classification and chatter it wall-wide -- or, before building
that, instrument step 14135 directly (dump per-particle `hydrostaticTerm`,
`P_b`, `rho_b`, and downstream pressure-force magnitude for the boundary
layer at that exact step, both runs) to confirm this diagnosis rather than
inferring it from the aggregate trace and rendered frame alone.

### 8.5 Direct instrumentation (2026-09-16, same day): §8.3's hypothesis refuted; a different, more specific mechanism confirmed; a 135-degree cutoff survives the tested window

Instrumented step 14135 directly per §8.4's action item, with a real
per-particle dump (`scripts/probe_englishClampDivergence.py`, kept as a
reusable diagnostic; degrades gracefully with a warning if run against the
unmodified, unclamped file). Also tested, per a user suggestion mid-session,
a **wider-angle cutoff**: rather than `dot(g,relPos).clamp_min(0)` (a hard
90-degree cutoff -- gravity exactly perpendicular to a wall's outward normal
already sits AT the threshold, so any roll at all flips a side wall's sign),
generalize to an arbitrary cutoff angle by clamping the *cosine*:
`cosTheta = dot(g,relPos) / (|g|*|relPos|)`, `cosTheta.clamp_min(cos(theta))`,
`hydrostaticTerm = cosTheta_clamped * |g| * |relPos|` -- reduces to the
original 90-degree clamp when `theta=90`, gives side walls headroom before
clamping in for `theta=135`.

**Methodology pitfall, caught and fixed before trusting any result:** the
first instrumentation attempt hand-built the case config instead of reusing
`run_sloshingTank.py`'s own `buildSpec`, and missed that `buildSpec`
explicitly re-asserts `params['shifting']=True` for the WCSPH scheme (the
case's own bare default is `shifting=False`, a `divergenceFree`-only
default) -- `cases/sloshingTank.py`'s own comment states shifting/PST is
*"the reason this case survives the wall slam at all."* Without it, even
the **unclamped baseline** diverged at step 384. Fixed by matching
`run_sloshingTank.py` exactly (`params=dict(shifting=True,
correctdrhodt=False)`); all results below are from the corrected harness.

**Three-way comparison, `--nSteps 14145` (t≈1.41s), window 14100-14145:**

| variant | result |
|---|---|
| baseline (unclamped, current production) | `diverged=False`, ran the full window |
| **clamp90** (§8.1's original clamp) | `diverged=True` at step 14134, NaN velocity at 14135 -- reproduces §8.2 exactly |
| **clamp135** (cosine-generalized, 135-degree cutoff) | `diverged=False`, ran the full window |

**The causal chain for clamp90's failure, directly traced (not inferred):**

1. Boundary/ghost pair UID 5475, right wall, shallowest layer, just above
   the still-water line -- a marginal, intermittently-wetted location.
2. 66 consecutive captured calls: zero real fluid neighbours (`hasAny=False`),
   `hydroRaw` pinned at a steady, negligible -0.0031 in all three runs alike.
3. **Call 67 (real step 14133), in clamp90 only:** `hasAny` flips `True` --
   one fluid particle enters this ghost's kernel support for the first time,
   with a vanishingly small (float32-underflowing) weight. The Shepard
   ratio `alpha = Sq/Mb` collapses toward 0 instead of ~1, giving
   `P_g = c0^2*(alpha-rho0) ~ -400` -- a ~400x too-negative ghost pressure.
   **The hydrostatic term itself is negligible here** (`hydroTerm ~ 2.7e-18`)
   -- this is a pre-existing precision hole in the Shepard `hasAny` branch
   (protects the exact `N=0` case, per §3 item 1's fix, but not `N=1` with a
   near-zero weight), **not an arithmetic consequence of the clamp**.
4. `rho_b = rho0 + P_b/c0^2 = 1 + (-400)/400 = 0.0` exactly, written to the
   boundary particle.
5. **Same call, same step:** `computePressureForceSurfaceAware` immediately
   produces NaN for 4 nearby fluid particles (all within ~0.01m of UID 5475)
   plus 99 boundary/ghost particles (harmless -- masked downstream). Call 66
   had zero non-finite particles; call 67 has 103 -- the NaN is born at this
   exact instant, in this function, not a delayed accumulation.
6. Step 14134's velocities go NaN -- the divergence the run reports.

**Answers to §8.4's three questions:**

- **Concentrated or wall-wide?** The *triggering event* is a single
  particle, one wall -- not a synchronized multi-particle collapse (§8.3's
  literal "many pairs cross zero in close proximity" mechanism is
  **refuted**). But the clamp's background perturbation genuinely *is*
  wall-wide and continuous: at t~1.41s (near a roll-angle turning point,
  barely sweeping) ~80% of the floor's particles sit at `hydroRaw` around
  -0.3 to -0.4 and are clamped toward 0 continuously -- true since t~0, not
  a rare sweep event (see §9 for why the sign is negative here at all).
- **Dot-sign flip at that instant, or a coincidental marginal-neighbour
  event?** **The marginal-neighbour event**, confirmed -- `hydroTerm` is
  unchanged from 66 calls earlier at the failing call; no sign flip is
  happening at that instant.
- **Origin in `rho_b`/`P_b` here, or downstream?** **In this function**
  (the Shepard-fallback branch), not the clamp arithmetic. Downstream
  `computePressureForceSurfaceAware` is the confirmed propagation path to a
  fluid-particle NaN, same step.

**Net mechanism:** not a direct arithmetic bug in the clamp, and not §8.3's
literal hypothesis -- a genuine, pre-existing precision hole in the
zero-neighbour Shepard fallback (an `N=1`-with-underflowing-weight gap next
to the already-fixed `N=0` gate), that clamp90's continuous, wide-scope
perturbation of the boundary pressure field over the preceding 14133 steps
was evidently enough to nudge into triggering at this exact step. Clamp135
never reaches its (much larger) clamp threshold on the floor's actual
`hydroRaw` range (max magnitude ~-0.4, far short of
`cos(135deg)*|g|*|relPos| ~ -0.7`), so its floor `rho_b` stays close to
baseline's (`[1.0000,1.0020]` vs `[0.9997,1.0019]` at call 60, vs. clamp90's
visibly shifted `[1.0015,1.0028]`) -- consistent with staying close enough
to baseline's trajectory to avoid the marginal-contact transition within
the tested window.

**Caveats:** only tested to t~1.41s, not the full 7s -- clamp135's survival
here says nothing yet about the actual t=5.66-6.9s cascade it targets; that
needs a separate full run. One loose end not chased: 5 boundary (kind=1,
harmless/masked) particles on the far left wall also showed NaN
`dvdt_pressure` at the same call 67 -- likely a coincidental second marginal
event, not verified further. Code reverted to pristine after this session
(`git diff` clean on `english2025.py`).

## 9. A separate, orthogonal bug found while instrumenting §8: `ghostOffsets`'s sign is flipped at the ghost particle's own row

Found incidentally while interpreting the per-particle dump above (needed
`relPos`'s sign to make sense of "which wall is floor-like"), **not yet
independently re-verified beyond this session's own tracing** -- flagged
here at the same confidence level the discovering session gave it, pending
a dedicated audit.

**The bug:** `english2025.py:88`'s `relPos = -currentState.ghostOffsets[ghost]`
is documented (module docstring, §5.1) as computing `x_b - x_g`. Checked
empirically at step 0 (before any stepping): it does not -- it comes out as
`x_g - x_b`, backwards from the documented/intended convention.

**Root cause, traced in code:** `addBoundaryGhostParticles`
(`src/warpSPH/rigidBody/ghostParticles.py:639-641`) deliberately negates the
offset it writes at a ghost particle's own row, so that `ghostOffsets` reads
as the intended `x_g - x_b` there. But
`src/warpSPH/initializers/weaklyCompressible.py:254-255` auto-registers a
(static) `RigidBody` for *every* boundary material and calls
`updateBodyParticlesWCSPH` on it immediately at init, and again every step
from `finalize`. That function
(`src/warpSPH/rigidBody/update.py:61-65`) computes one
`offsets = particlePositions - ghostParticlePositions` and writes the SAME,
un-negated value to *both* the boundary row and the ghost row -- silently
overwriting `addBoundaryGhostParticles`'s deliberate negation at the ghost
row. Net effect: `ghostOffsets` read at a ghost particle's own row ends up
`x_b - x_g` instead of the intended `x_g - x_b`, so `relPos` in
`english2025.py` computes to `x_g - x_b` -- the opposite of what its own
docstring and §5.1's DualSPHysics-cross-checked derivation assume.

**Blast radius:** every mDBC module reading `ghostOffsets[ghostMask]` with
this same pattern is affected: `english2025.py:88`, `densityBand.py:113`,
`velocity.py:210`. `density2025.py:106-107` happens to carry an *extra*
`-` elsewhere in its own formula (`drho = -einsum(relPos, grad)`) that, by
algebraic inspection (not independently re-verified end-to-end), cancels
this bug -- so production's default `'ramped'` scheme is *likely*
unaffected, but that inference rests on one file's algebra, not a run-time
check, and should not be trusted without a dedicated audit.

**Why this matters for everything §5-7 already concluded about
`english2025`:** §5.1's sign correction was verified against DualSPHysics'
`Mdbc2PressClone` *as a formula* (`P_b = P_g + rho0*dot(g,relPos)`,
`relPos = x_b - x_g`) -- that derivation is not in question. What's in
question is whether the *running code*, this whole time, has actually been
evaluating that formula with the correct-sign `relPos`, or with this bug's
flipped sign. If flipped, `english2025` has been computing `P_b = P_g -
rho0*dot(g, x_b-x_g)` in every one of §5-7's validation runs (Marrone
3.1/3.4 at both resolutions, the sloshingTank baseline) -- the opposite
hydrostatic sign from what the docstring, the derivation, and the DualSPHysics
cross-check all claim. This does **not** invalidate the empirical
performance numbers themselves (they measured whatever the actual code did,
correctly), but it means the *causal story* for why `english2025` outperformed
`'ramped'`/`'band'` there (§3.1/§5's "known, noise-free hydrostatic term vs.
a fitted, lever-arm-amplified gradient") may not be the operative mechanism
in the runs that produced those results, and it is not yet known which sign
was actually in effect for §6.4's specific sloshingTank cascade this whole
§8 investigation has been chasing.

### 9.1 Follow-up audit (2026-09-16, same day): independently confirmed, full consumer table, and the fix is not a one-line patch

Re-verified from scratch, independent of the instrumenting session's own
harness: a standalone one-step script
(`scripts/probe_englishClampDivergence.py`'s case-building pattern, minimal
version not kept in the repo) compares `ghostOffsets` directly against
ground-truth `positions[boundary] - positions[ghost]` via `ghostIndices`
pairing, on sloshingTank at t=0. Result, exact to float32 precision (errors
either `0.000e+00` or the genuine per-particle offset magnitude, nothing in
between):

```
boundary row: ghostOffsets[b] vs (x_b - x_g)                        max err = 0.000e+00
ghost row:    ghostOffsets[g] vs (x_g - x_b)  [intended convention] max err = 8.395e-02
ghost row:    ghostOffsets[g] vs (x_b - x_g)  [the alleged bug]     max err = 0.000e+00
```

Confirms §9's finding exactly: the boundary row is always correct; the ghost
row is always flipped. Also confirms the bug is universal, not
roll/sloshingTank-specific -- `buildRigidBody`
(`src/warpSPH/rigidBody/build.py:17-70`) wraps **every** boundary material
with particles into a `RigidBody` unconditionally (`angularVelocity`/
`linearVelocity` default to zero -- a wall only moves if a case script sets
a nonzero rate afterward), and `updateBodyParticlesWCSPH` runs for every
registered body regardless of whether it moves, both once at init
(right after `addBoundaryGhostParticles` sets the correct value) and every
step from `finalize` thereafter. So the corruption applies to every WC/
incompressible case with boundary particles -- Marrone 3.1/3.4 included --
not only rolling geometries; sloshingTank's roll only made the SYMPTOM
(§8's clamp work) depend on gravity's direction, not the underlying data bug.

**Full consumer audit** (every `ghostOffsets[ghostMask]`-pattern read found
via `grep -rn ghostOffsets src/warpSPH`), determining whether each is
actually wrong under the confirmed flipped sign, or coincidentally immune:

| consumer | pattern | actual effect under the bug |
|---|---|---|
| `density2025.py:106-107` (`'ramped'`, production default) | `relPos=-ghostOffsets[ghost]`; `drho=-dot(relPos,grad)` | **Correct, by accidental cancellation** -- the function's own extra `-` in the `drho` line cancels the bug's sign flip exactly (verified algebraically: two negations of the same flipped quantity restore the intended `dot(x_b-x_g, grad)`). |
| `modules/incompressible/wallPressure.py:255-256` (`'mls'` mode) | `relPos=-ghostOffsets[ghost]`; `dP=-dot(relPos,grad)` | **Correct**, same structure/cancellation as `density2025.py`. |
| `english2025.py:88` | `relPos=-ghostOffsets[ghost]`, used directly in `dot(g,relPos)` | **Wrong -- sign flipped.** No compensating negation; matches §8/§9's original finding. |
| `densityBand.py:113` | `relPos=-ghostOffsets[ghost]`; `xtb=xtg+relPos` (a **position**, not a scalar dot) | **Wrong, and NOT just a sign flip on the answer** -- the Taylor-shift evaluation point works out to `2*x_g - x_b` (algebra: intended shift is `+(x_b-x_g)`, the bug delivers `-(x_b-x_g)`, i.e. the same magnitude in the opposite direction from the ghost) instead of `x_b`. This is the mirror point of the boundary particle reflected through the ghost -- **not on the codebase's boundary at all**, and the resulting extrapolation error grows with the SAME lever arm (`x_b-x_g`, i.e. layer depth) that §3.1 already identified as the mechanism behind Band's depth/lever-arm divergence. This bug is a plausible, previously-unidentified CONTRIBUTOR to (not necessarily the sole cause of) that failure mode -- not yet re-validated with the bug fixed to confirm how much of §3's "architectural mismatch" verdict it actually explains. |
| `velocity.py`'s `noSlip`/`freeSlip` (`_wallNormals`, line 130-132) | `n_b = ghostOffsets / \|ghostOffsets\|`, used only as `dot(w,n_b)*n_b` | **Mathematically immune regardless of sign** -- `n (n^T)` is invariant under `n -> -n`, so a flipped normal changes nothing here. |
| `velocity.py`'s `extendedVelocity`, line 210 | `relPos=ghostOffsets[ghost]` (**no** negation, unlike every other consumer above); `vel=u_interp-dot(-relPos,grad)` | **Correct only because of the bug** -- this function's own formula omits the negation the other four apply, so it happens to land on the right answer only while `ghostOffsets[ghost]` is flipped. This is the mirror image of `density2025.py`'s situation: not independently correct, coincidentally correct, and would BREAK if the root cause were fixed without also adding a negation here. |
| `wp_nopenshift.py:560` | `queryOffsets=ghostOffsets` (full array; kernel indexes at the **boundary** row for `BoundaryToFluid`, confirmed by tracing `checkDirectionality_i`) | **Unaffected** -- reads the boundary row, never corrupted. |
| `systems/weaklyCompressible.py:96` (`_meanBoundaryNormal`) | `off=ghostOffsets[jj[sel]]` where `sel` selects `kinds[jj]==1` (boundary row) | **Unaffected**, same reason. |
| `rigidBody/build.py:62-65` (`particleBoundaryNormals`/`Distances`, both rows) | snapshot taken once, **before** `updateBodyParticlesWCSPH` first runs (`build.py`'s call precedes the corrupting loop in `initializers/weaklyCompressible.py:249-255`) | Captures the **correct** pre-corruption value at both rows -- but only ever consumed by `io/export.py`'s `.h5` dataset writer (checked: no other reader in the tree). Inert for live physics; only affects exported diagnostic data, and even that is currently *correct* (pre-corruption snapshot), not wrong. |

**What a real root-cause fix requires -- not a one-line patch.** Simply
negating `update.py`'s ghost-row write (restoring the intended `x_g - x_b`)
would fix `english2025.py` and `densityBand.py` for free (no other change
needed -- they already use the documented convention), but would BREAK the
two currently-correct modules (`density2025.py` and `wallPressure.py`'s
`'mls'` mode both need their own compensating `-` removed to stay correct)
and `velocity.py`'s `extendedVelocity` (needs a negation added, to match
the convention it currently omits only because the bug supplies it). A
correct fix therefore touches **four files together**
(`rigidBody/update.py` + `density2025.py` + `wallPressure.py` +
`velocity.py`), not one, and needs `density2025.py` (production default)
and `wallPressure.py` (incompressible/DFSPH's `'mls'` wall-pressure mode)
re-validated after the change, since both currently-correct paths would be
silently exercised with the wrong sign otherwise -- a regression risk in
production code, not just the two known-broken opt-in schemes.

### 9.2 Fixed (2026-09-16, same day) and re-validated against production

User elected the full fix (not a narrower patch confined to the two broken
opt-in schemes). Applied all four coordinated changes identified in §9.1:

1. `rigidBody/update.py:66-68` -- the root cause: `updatedOffsets[rigidBody.ghostParticleIndices]`
   now gets `-offsets` instead of `offsets`.
2. `mdbc/density2025.py:106-107` -- removed the now-redundant compensating
   `-` (`drho = torch.einsum(...)`, was `-torch.einsum(...)`).
3. `modules/incompressible/wallPressure.py:255-256` -- same removal
   (`dP = torch.einsum(...)`, was `-torch.einsum(...)`).
4. `mdbc/velocity.py:210` (`extendedVelocity`) -- added the negation it was
   previously missing (`relPos = -currentState.ghostOffsets[ghostMask]`, was
   unnegated).

`english2025.py`/`densityBand.py` needed no changes -- confirmed correct
automatically now that the source data is fixed.

**Verified the fix took effect:** re-ran the same standalone one-step
empirical check from §9.1 after the change -- ghost row now matches the
intended `x_g - x_b` convention exactly (`0.000e+00` error), flipped
convention now the one with nonzero error. Root cause confirmed fixed, not
just patched around.

**Production regression check, adversarial config** (`scripts/
probe_deltaSPHMarrone.py --nx 35 --shifting off --tLimit 1.2 --video`, no
`mdbcDensityScheme` override -- the true production default, `density2025.py`):

| | §6.1 recorded (pre-fix) | post-fix |
|---|---|---|
| diverged | False | **False** |
| maxVel peak | 13.9 @ t=0.66 | 13.925 @ t=0.657 |
| maxVel end | 2.61 | 2.607 |
| density range | [0.887, 1.109] | [0.8868, 1.1093] |

Matches to the precision the original table was recorded at -- expected,
not just hoped for: `density2025.py`'s old (buggy-input + compensating `-`)
and new (fixed-input + no compensation) expressions are the same closed
-form algebraic identity either way, so this result is a near-exact
reproduction by construction, and it is one. Production (`'ramped'`) is
confirmed unaffected.

`wallPressure.py`'s `'mls'` mode (incompressible/DFSPH scheme) has the
identical algebraic structure to `density2025.py`'s fix -- same reasoning
applies, but not yet separately run through an incompressible-scheme
regression case (this Marrone probe is weakly-compressible/`deltaSPH`
only); flagged if incompressible-scheme confidence specifically is wanted
later.

**Production regression check, stable/recommended config** (`--nx 35
--integrationScheme symplecticEuler --shifting on --densityDiffusionTerm
fourtakas2019 --tLimit 0.9 --video`):

| | §6.2 recorded (pre-fix) | post-fix |
|---|---|---|
| diverged | False | **False** |
| maxVel peak | 4.74 @ t~0.73 | 4.738 @ t=0.733 |
| maxVel end | 2.6 | 2.629 |
| density range | [0.995, 1.003] | [0.9838, 1.0157] |

`maxVel` (the more failure-predictive metric elsewhere in this plan)
matches almost exactly. `density range` is measurably wider -- roughly 2x
the old excursion band. Not attributed to this fix, for two reasons: (1)
`density2025.py`'s change is a pure double-negation identity (both the old
buggy-input+compensating-`-` and the new fixed-input+no-compensation paths
reduce to the identical `torch.einsum` call on the identical floating-point
`relPos` tensor -- sign-flip is exact in IEEE754, introduces no rounding),
so this specific computation should be bit-identical; (2) §6.2's own text
notes that row is reused verbatim from an EARLIER §3.2 measurement, which
predates several unrelated fixes since landed in this codebase (the
Antuono pressure sign fix, mdbc ghost-refresh changes, etc.) -- not a clean
isolated baseline for just this change, and no same-codebase
"before-this-exact-fix" run was preserved to isolate it (the files were
edited in place before validating). This system is also documented
elsewhere in this plan/memory as run-to-run chaotic-sensitive. Recorded
here rather than waved away -- an open, low-priority observation, not a
confirmed regression; a repeat run would sharpen this if wanted later.

**Status: fixed and validated on both Marrone 3.1 configs (adversarial:
clean match; stable: maxVel clean match, density range wider but not
attributed to this fix -- see above).**

### 9.3 `english2025` re-validated under the corrected sign (2026-09-16): edge intact, not an artifact of the bug

User's follow-up, correctly prioritized: `english2025.py` has no
compensating negation anywhere (unlike `density2025.py`/`wallPressure.py`'s
accidental self-cancellation) -- if the sign bug were actually consequential
to this scheme's validated stability advantage, THIS is the case where it
would show. Re-ran both Marrone 3.1 configs with `--mdbcDensityScheme
english2025` under the now-fixed sign.

**Adversarial config** (`--nx 35 --shifting off --tLimit 1.2`):

| | §6.1 recorded (wrong sign) | post-fix (correct sign) |
|---|---|---|
| diverged | False | **False** |
| maxVel peak | 4.83 @ t=0.78 | **4.10** @ t=0.86 |
| maxVel end | 2.17 | 2.16 |
| density range | [0.994, 1.006] | [0.9896, 1.0047] |

Lower peak velocity, essentially unchanged end velocity, marginally wider
density band but still an order tighter than `'ramped'`'s post-fix 13.925
peak / [0.8868,1.1093] range (§9.2). If anything modestly BETTER under the
genuinely-correct formula, not worse.

**Stable config** (`--nx 35 --integrationScheme symplecticEuler --shifting
on --densityDiffusionTerm fourtakas2019 --tLimit 0.9`):

| | §6.2 recorded (wrong sign) | post-fix (correct sign) | `'ramped'` post-fix (§9.2, for reference) |
|---|---|---|---|
| diverged | False | **False** | False |
| maxVel peak | 4.48 @ t=0.78 | **4.87** @ t=0.75 | 4.74 @ t=0.73 |
| maxVel end | 2.56 | 2.55 | 2.63 |
| density range | [0.992, 1.004] | [0.987, 1.006] | [0.984, 1.016] |

Still a clean pass, no divergence -- but a more nuanced result than the
adversarial config. On the config where production was already stable, the
corrected sign brings `english2025` almost exactly into line with
`'ramped'` (4.87 vs 4.74 peak, both close together) rather than staying
modestly ahead of it the way the old (wrong-sign) recorded row showed (4.48
vs `'ramped'`'s then-baseline 4.74). Unlike `'ramped'`'s own §9.2
comparison (an algebraic identity, so its shift is attributed to
codebase drift, not the fix), THIS comparison is a clean, direct
before/after of the same scheme with only the sign changed -- so this
narrowing is a real, attributable effect of the fix, not noise.

**Net reading:** `english2025`'s stability edge over `'ramped'` is
confirmed NOT to be an artifact of the wrong sign on the adversarial
config, where it remains clearly and substantially ahead (4.10 vs 13.9
`maxVel` peak) -- the mechanism claim (analytic hydrostatic term beats a
fitted, lever-arm-amplified gradient) holds up there under the real
formula. On the easier, already-stable config, though, the genuine edge
over `'ramped'` **narrows to roughly parity** once the sign is corrected --
the previously-recorded modest advantage there (§6.2) was partly (not
entirely -- both still pass cleanly) an artifact of the wrong sign
happening to help on this specific easy case. Both configs still pass
cleanly with no divergence either way.

### 9.4 sloshingTank under the corrected sign (2026-09-16): diverges almost immediately -- same pre-existing precision hole as §8.5, triggered much earlier

User's follow-up, correctly sequenced after the Marrone checks passed: re-ran
§6.4's exact sloshingTank reproduction (`examples/sloshingTank/
run_sloshingTank.py --tLimit 7.0 --mdbcDensityScheme english2025
--densityDiffusionTerm fourtakas2019 --video`, no clamp code present --
`english2025.py` itself carries zero diff from the original, only the sign
fix upstream of it) to see whether the t=5.66-6.9s ceiling-hover cascade
changes under the genuinely-correct formula.

**Result: `diverged=True` almost immediately -- t=0.02s (164 steps), vs.
the original wrong-sign run's full 7s survival.** Reproducible exactly
(re-ran twice, identical step/time). `maxVelocity` was small and smoothly
varying (0.03-0.05, a barely-started roll, nowhere near violent) for the
preceding 20+ steps, then jumps straight to NaN with zero precursor --
the same abrupt, precursor-free signature as §8.5's clamp90 divergence, not
a gradual physical blowup.

**Instrumented directly** (`scripts/probe_englishClampDivergence.py`'s
pattern, ad hoc, not kept in the repo -- the case-building/monkeypatch
technique from §8.5, reused): traced the exact particle and cause, not just
inferred it.

- Boundary particle UID 5493, left wall, `y=0.110` -- the same
  marginal, near-the-still-water-line profile as §8.5's UID 5475 (a
  different wall this time, same mechanism).
- `nNbFluid=1` (passes the `hasAny=nNbFluid>=1` gate -- the "exact integer
  count" fix from §3 item 1 protects against `N=0`, not this).
- `Mb=9.59e-42` -- the single neighbour's kernel weight has essentially
  underflowed to nothing (this is not merely "small," it is deep inside
  float32 denormal territory).
- `alpha = Sq/Mb = 9.6e-12` -- collapses toward 0 instead of ~1 (should
  read as "no usable neighbour," reads as "valid" because the gate is on
  neighbour COUNT, not on the resulting weight's magnitude).
- `hydro term = 5.9e-6` -- negligible; **not the cause**, confirming this is
  the identical mechanism §8.5 found, unrelated to the hydrostatic sign
  itself.
- `P_g = c0^2*(alpha-rho0) ~ -400`, `rho_b = rho0 + P_g/c0^2 = 0.0`
  EXACTLY -- the same `Mb`-underflow-into-Shepard-collapse signature as
  §8.5's UID 5475.
- Same step: ~100 nonfinite pressure-force values across both left- and
  right-wall boundary/ghost particles plus 3 nearby fluid particles -- the
  same order of magnitude and propagation pattern as §8.5's finding.

**This is not a new bug the sign fix introduced.** It is the SAME
pre-existing Shepard-fallback precision hole §8.5 already documented
(`hasAny` protects the exact `N=0` case but not `N=1`-with-an
-underflowing-weight) in `english2025.py` (and, per §9.1's audit,
`densityBand.py` inherits the identical pattern). The wrong sign didn't
avoid this hole -- it just perturbed the boundary pressure field
differently enough that THIS specific fragile particle happened not to get
pushed over the edge for 5+ seconds, long enough to reach a different,
already-documented failure (§6.4's ceiling-hover cascade) first. The
corrected sign perturbs the same fragile field from t=0 in a way that
triggers the identical hole almost immediately instead. Neither the sign
nor the ceiling-clamp question is the real blocker here -- this precision
hole is.

**Net implication:** `english2025.py` cannot be meaningfully validated on
sloshingTank (ceiling-hover cascade or otherwise) until this Shepard
-fallback fragility is fixed -- a genuine numerical-robustness gap, not a
sign or clamp-direction issue. The fix shape is already sketched by
`density2025.py`'s own analogous protection (§ "1st-order (English Eq. 12)
-> 0th-order (Shepard) blend ramps," `_MDBC_NBR_FLOOR`/`_MDBC_NBR_RAMP`):
require more than a bare `>=1` neighbour count, and/or floor/ramp on `Mb`
itself (the same class of fix `densityBand.py`'s docstring already
describes doing for ITS `Mb`, §3 item 1 -- but that fix protects the
*gradient* block there, not this *value* term, which both `english2025.py`
and `densityBand.py` still gate on bare `hasAny`). Not yet implemented --
flagged, not actioned, pending a scope decision (this affects both opt-in
schemes' shared Shepard-value term, a small but real piece of numerical
-robustness work, not a revert of anything done today).

### 9.5 Sign re-verified against the DualSPHysics source directly (2026-09-16), and why sloshingTank hits this but Marrone 3.4 doesn't

**Sign re-verification.** User's follow-up challenge: does `english2025.py`'s
`relPos` need the "surface normal" convention (`r_g - r_b`) for the gravity
dot product specifically, rather than the `r_b - r_g` it uses (and that its
own docstring, correctly per the user, describes as right for the
extrapolation)? Re-derived directly from
`~/dev/DualSPHysics/src/source/JSphCpu_mdbc.cpp` (not the transcription in
§5.1 -- read the actual file this time): `boundnor = x_g - x_b` (line 49,
`gposp1 = pos[p1] + boundnor[p1]`, i.e. the ghost position is built by
adding `boundnor` to the boundary position); `dpos = -boundnor = x_b - x_g`
(line 338). `Mdbc2PressClone` (lines 182-205) computes
`normal = boundnor/|boundnor|`, `normpos = dot(dpos, normal)`,
`normforce = rho0*dot(gravity, normal)`, `pressfinal = pghost +
normforce*normpos`. **`normal` appears twice, multiplicatively** -- flip
its sign and both `normforce` and `normpos` flip, so their product (and the
whole formula) is invariant under `normal -> -normal`. Substituting through
algebraically (and confirmed with a concrete floor-case numeric check: `x_b=(0,0)`,
`x_g=(0,0.1)`, `g=(0,-9.8)` -> `pressfinal = pghost + 0.98*rho0`, matching
`pghost + rho0*dot(g, x_b-x_g)` exactly) collapses the whole thing to
`pressfinal = pghost + rho0*dot(gravity, x_b - x_g)` -- no normal needed at
all, `english2025.py` doesn't compute one, and dotting `relPos` (=`x_b-x_g`
post-fix) directly with gravity is exactly this reduced form. Also
confirmed `computeGravity` (`modules/gravity/wrapper.py`) returns the plain
physical acceleration vector -- no hidden normal-related transform that
could reintroduce a normal-direction dependency. **The fix stands**;
`r_g - r_b` in the gravity dot product would reintroduce the original
flipped sign, not correct anything.

**Marrone 3.4, corrected sign** (`--nx 128 --tStar 3.0 --mdbcDensityScheme
english2025 --shifting off`, matching §6.3's original config): **6/6 PASS,
`diverged=False`** -- `maxVel` 29.27 (4.8 U_max), density [0.996,1.026],
no penetration, KE decaying. Confirms the precision hole (§9.4) genuinely
does not manifest on this case, not just "hasn't been observed yet."

**Why sloshingTank but not Marrone 3.4 -- traced, not guessed.** Tracked
the minimum Shepard weight (`Mb`) among ghost rows with exactly one fluid
neighbour (`nNbFluid==1`, the exact class of row §9.4's failure came from),
every step, for the first ~164-300 steps of each case (`english2025`,
otherwise matching settings). Both geometries produce razor's-edge
near-zero `Mb` values -- this is not unique to sloshingTank, it is an
inherent side effect of `gridsnap` ghost placement against a regular
-lattice fluid IC, wherever a ghost's kernel-support boundary happens to
graze a single fluid particle almost exactly. The difference is temporal:

- **sloshingTank**: a marginal (`N=1`) row appears in only 19 of 164
  captured steps, but when it does, its weight decays SMOOTHLY AND
  MONOTONICALLY across 13 consecutive steps from t=0
  (`2.7e-20 -> 2.5e-20 -> ... -> 1.8e-23 -> 4.3e-25 -> 2.4e-31`) -- nothing
  disturbs that one particle (confirmed independently by §9.4's
  frozen-gravity test: this spot is at rest, no roll needed to reproduce
  it), so it just keeps drifting toward genuine float32 underflow with
  nothing to interrupt it.
- **Marrone 3.4**: marginal rows appear far MORE often (136 of 300 steps,
  including one 84-consecutive-step run) -- the violent splashing
  constantly creates new marginal contacts -- but the weight FLUCTUATES
  around a stable order of magnitude (~1e-8) rather than monotonically
  collapsing (e.g. `1.9e-8 -> 3.9e-8 -> 7.1e-8 -> 1.2e-7 -> ...`, even
  across that 84-step run) -- the fast-moving fluid keeps perturbing each
  marginal particle's neighbourhood before it can drift all the way to the
  edge.

So the fragile geometric coincidence exists in both cases (this is a
property of `gridsnap` ghost placement generally, not a sloshingTank
-specific defect); Marrone's violent dynamics never let a marginal particle
sit still long enough to fully underflow, while sloshingTank's calm free
surface does. This directly explains why Marrone's extensive validation
(§6.1-6.3d, §9.3) never surfaced §9.4's precision hole despite sharing the
exact same underlying fragility.

### 9.6 The Shepard precision hole, fixed (2026-09-16): why it doesn't cancel, and a symmetric-epsilon regularization

**Why the numerator/denominator don't cancel (user's question, worth
recording precisely).** In exact arithmetic, `alpha = Sq/Mb = (Sigma w_j
rho_j)/(Sigma w_j)` reduces to `rho_j` for a single neighbour regardless of
how small its kernel weight `w` is -- the SAME `w` appears in both sums, so
it cancels. It didn't cancel here because of an asymmetric safety clamp:
`Mb = M[ghost].clamp_min(0.0)`, `MbSafe = Mb.clamp_min(1e-30)`,
`alpha = Sq[ghost]/MbSafe` (the pre-fix code). `Sq` (the numerator) was
NEVER floored; only `Mb` (the denominator) was, and only once it dropped
below `1e-30`. For UID 5493's failing row, true `Mb ~ 9.6e-42` (below the
floor) but `Sq ~ Mb*rho_j ~ 9.6e-42` (never touched) -- so
`alpha = Sq/MbSafe = 9.6e-42/1e-30 = 9.6e-12`, matching §9.4's captured
value exactly, not an approximation. The `1e-30` floor was meant only to
avoid literal `0/0`; applied to just one side of the ratio, it silently
converts "technically valid but vanishingly small" into "wildly wrong but
finite," which is worse than the `NaN` it was meant to prevent (a `NaN`
gets caught by `nan_to_num`; a wrong-but-finite value doesn't).

**The fix (user-proposed): add the SAME epsilon to both sides.**
`alpha = (Sq[ghost] + eps*rho0) / (Mb + eps)`. This preserves the
cancellation at every scale instead of breaking it at one: `Mb=0` (no
neighbours) -> `alpha=rho0` exactly (matching the existing explicit
fallback, now redundant for this term but left in place as a second
guarantee on the final `rho_b`, unchanged); `Mb` comparable to or below
`eps` -> `alpha` blends smoothly toward `rho0` instead of the wrong
near-zero value; `Mb >> eps` -> `alpha ~= Sq/Mb`, unaffected.

**Calibrating `eps`:** measured this case's own `Mb` distribution directly
(not the `density2025.py`-docstring's Marrone-oriented "healthy ~1e-3"
number) -- at nx=200, a genuinely healthy row (`nNbFluid=4`) has
`Mb ~ 7.3e-4`; even a non-degenerate `nNbFluid=1` row sits around
`Mb ~ 1e-20` already (single-neighbour rows are inherently marginal in
this setup, independent of underflow). `eps = 1e-8` sits 4-5 orders of
magnitude below the healthy floor (<0.002% perturbation there) while
fully dominating every `nNbFluid=1` case and the catastrophic underflow
range -- meaning single-neighbour rows now uniformly fall back toward
`rho0`, which is the intended, conservative behaviour, not an accident of
where the old `1e-30` constant happened to sit.

**Implemented** in `english2025.py` only (the file under active
investigation). `densityBand.py` has an analogous `alpha = Sq[ghost] /
MbSafe` pattern (`MbSafe = Mb.clamp_min(1e-30)`) that likely has the same
issue -- not yet fixed there, flagged for a follow-up pass.

**Validated:**

- **sloshingTank** (`--tLimit 7.0 --mdbcDensityScheme english2025
  --densityDiffusionTerm fourtakas2019 --video`): **`diverged=False`, runs
  the full 7s** -- was diverging at t=0.02s before this fix. The precision
  hole is resolved. The run now reaches a DIFFERENT, worse excursion later:
  density [0.518, 2.889] at t~5.53-5.77s, sensor pressure peak 102,347 Pa
  at t=4.65s, onset ~t=2.36-2.46s. This timing lines up closely with §6.4's
  ALREADY-DOCUMENTED ceiling-hover cascade (originally t=5.66-6.9s, peak
  98,177 Pa under the wrong sign) -- almost certainly the same pre-existing,
  separately-tracked free-surface-pinning mechanism
  ([[antuono-pressure-switch-bug]]), now larger in magnitude under the
  genuinely-correct sign and no longer masked by the earlier Shepard-hole
  crash. Not yet root-caused further this session -- flagged as the next
  open item, not a new bug from today's work.
- **Marrone 3.4** (`--nx 128 --tStar 3.0 --mdbcDensityScheme english2025
  --shifting off`): **6/6 PASS, diverged=False, maxVel=29.25** (pre-fix:
  29.27) -- no regression, essentially identical. Worth checking despite
  §9.5's finding that Marrone doesn't hit the OLD `1e-30` floor: the new
  `eps=1e-8` is comparable to or larger than some of Marrone's own observed
  marginal `Mb` values (`1e-8` to `1e-25` range, §9.5's trajectory), so it
  could plausibly have touched legitimate marginal-but-real Marrone
  stencils differently -- confirmed it didn't, empirically.
- **Marrone 3.1** adversarial-config re-check: deferred (user redirected
  effort toward the longer-duration 3.4 check below instead, since a
  short-`tStar` run doesn't probe the thin-sheet/boundary interaction the
  investigation actually cares about -- §6.3b/§6.3c already established
  that pattern: short runs missed failures a longer duration exposed).
- **Marrone 3.4 at 3x duration** (`--nx 128 --tStar 9.0`, matching §6.3b's
  protocol exactly -- the config that originally exposed `'ramped'`'s late
  secondary-impact/kinetic-energy failure a short run couldn't see):
  **6/6 PASS, `diverged=False`, `maxVel=29.25`, `dKE/dt*` (2nd half)
  -0.616 (decaying)** -- no regression at the longer duration either.

### 9.7 `eps=1e-8` overcorrected; recalibrated to `eps=1e-30` -- substantially better than both

**User's observation, correct:** the full-7s sloshingTank run with
`eps=1e-8` (§9.6) looked WORSE than the original wrong-sign baseline, not
just different -- density swung to `[0.518, 2.889]` (wider than the
baseline's `[0.65, 1.72]`) and the smoothed sensor-pressure trace itself
went fully off-scale (peak 102,347 Pa vs the baseline's 98,177 Pa, but
recurring across at least three separate off-scale events instead of one,
with onset shifted much earlier, ~t=2.4s vs ~t=5.66s).

**Root cause: `eps=1e-8` sits ABOVE this case's own typical `Mb` scale for
a non-degenerate single-neighbour row (`Mb ~ 1e-20`, §9.5/§9.6), not just
above the catastrophic-underflow range (`Mb ~ 1e-42`).** So it wasn't
narrowly patching the one pathological particle -- it was silently
overriding EVERY ordinary `nNbFluid=1` boundary row's raw Shepard value
toward `rho0` throughout the whole run, discarding real information the
OLD (buggy) code's unclamped `Sq/Mb` had actually been using successfully
at that scale. A much bigger, more indiscriminate behaviour change than
intended.

**Recalibrated to `eps=1e-30`** -- reusing the OLD code's own denominator
-only floor threshold, just applied symmetrically instead of
asymmetrically. This sits far below the normal `nNbFluid=1` scale
(negligible perturbation, near-reproduces the old code's behaviour there)
while still dominating genuine underflow (`Mb~1e-42`) enough to fall back
to `rho0` instead of the wrong `9.6e-12`.

**Full three-way comparison, sloshingTank + `english2025`, full 7s
(`--tLimit 7.0 --mdbcDensityScheme english2025 --densityDiffusionTerm
fourtakas2019 --video`):**

| | original (wrong sign) | `eps=1e-8` | `eps=1e-30` |
|---|---|---|---|
| diverged | False (full 7s) | False (full 7s) | **False (full 7s)** |
| onset of worst excursion | ~t=5.66s | ~t=2.4s | ~t=4.0-6.5s |
| density range (whole run) | [0.65, 1.72] | [0.518, 2.889] | **[0.725, 1.436]** |
| sensor pressure peak | 98,177 Pa | 102,347 Pa | **50,031 Pa** |
| maxVel peak | 16.4 | ~11.0 | **7.25** |

`eps=1e-30` beats BOTH the `eps=1e-8` attempt and the original baseline on
every metric -- tighter density range than either, roughly half the peak
pressure, lower peak velocity. The worst excursion's timing (t~4.0-6.5s,
peaking t~6.4-6.5s) also sits much closer to the original baseline's
already-documented t=5.66-6.9s window than `eps=1e-8`'s much-earlier onset
did -- consistent with `eps=1e-30` reproducing the old code's (better
-behaved) dynamics almost everywhere except the one genuinely pathological
case, rather than broadly changing behaviour. The smoothed sensor-pressure
trace now tracks the measured record reasonably well across all five roll
-cycle impacts, unlike `eps=1e-8`'s fully-off-scale smoothed peaks.

**`_ALPHA_EPS` updated to `1e-30` in `english2025.py`** (was `1e-8`,
briefly, not shipped as final). Marrone re-validation at `1e-8` (§9.6)
should be treated as superseded by this recalibration -- not yet re-run at
`1e-30` specifically, though since `1e-30` is a strict subset of what the
old (already-Marrone-validated) code did, no regression is expected there;
flagged if anyone wants that explicit re-check.

**Remaining excursion likely the pre-existing ceiling-hover mechanism, not
a new bug.** User directly observed (watching the rendered video) particles
sticking to the ceiling that then randomly fall or get ejected into the
fluid, causing pressure spikes -- matching
[[antuono-pressure-switch-bug]]/§6.4's already-documented "ceiling-hover"
pattern (a particle pinned at the free surface near a corner/ceiling by an
under-conditioned mDBC fallback pressure, eventually ejected
catastrophically) exactly. This is a separate, pre-existing failure mode
that neither today's sign fix nor the Shepard-hole fix was ever going to
address -- §8's original ceiling-clamp attempt was aimed at exactly this
mechanism before getting derailed into the sign-bug/Shepard-hole
investigation. **Next step, if picking this back up:** now that the
Shepard hole and sign are both fixed and sloshingTank runs cleanly to full
duration otherwise, this ceiling-hover mechanism is finally isolated enough
to investigate directly (previously masked by earlier, more severe
failures) -- likely where §8's clamp135 angle-cutoff idea (or a
differently-targeted fix for the free-surface pinning itself) belongs.

### 9.8 Marrone 3.4 at full resolution/duration (nx=256, tStar=15.6605), recalibrated `eps=1e-30`

User-requested: the highest-resolution, longest-duration config this
investigation has (§6.3c's protocol -- `t=5s`, matching the pre-existing
`sun2017DeltaSPH_nx256` reference rows), specifically to see thin-sheet/
boundary interaction behaviour more clearly than a shorter or coarser run
can. `--nx 256 --tStar 15.6605 --mdbcDensityScheme english2025 --video`
(shifting off, default):

**6/6 PASS, `diverged=False`, reached t*=15.66/15.6605.** `rho` [P05,P99]
in [0.9971, 1.0222] (pointwise max 1.212); `maxVel` 39.07 (6.4x U_max,
report-only threshold is 12x, well clear); no wall or obstacle
penetration; `dKE/dt*` (2nd half) -1.168 (decaying). Comparable to the
pre-Shepard-fix §6.3c baseline at the same config (`maxVel` 32.4,
`dKE/dt*` -1.10 decaying) -- both clean, `maxVel` modestly higher here but
still far under the report threshold and kinetic energy still properly
decaying, not the growing-energy failure `'ramped'` showed at nx=128/this
duration (§6.3b). No regression from the recalibrated Shepard fix at this
combined highest-resolution/longest-duration config.

## Status

Open, actively worked 2026-09-15 through 2026-09-17. This section is a
current-state summary; §§1-9.8 above are the full history/evidence trail.

**`mdbcDensityScheme='english2025'`** is implemented, toggle-able, **still
not the default**. Two significant, unrelated bugs were found and fixed
this session, both re-validated against Marrone 3.1 and 3.4 (multiple
resolutions, including the full `nx=256`/`tStar=15.66` protocol) with no
regression:

1. **`ghostOffsets` sign bug** (§9-9.2) -- a `RigidBody` init/step hook was
   silently overwriting a deliberate sign convention at every ghost
   particle's own row, universal across every WC/incompressible case with
   boundary particles. Fixed across the four files that needed coordinated
   changes (`rigidBody/update.py`, `density2025.py`, `wallPressure.py`,
   `velocity.py`'s `extendedVelocity`); `english2025.py`/`densityBand.py`
   needed no changes, they become correct automatically. Production's
   `'ramped'` default re-verified clean on both Marrone 3.1 configs.
2. **Shepard precision hole** (§9.4-9.7) -- `english2025.py`'s ghost-value
   Shepard ratio floored only its denominator, breaking a cancellation that
   should hold at any scale; fixed via symmetric epsilon regularization
   (`alpha = (Sq+eps*rho0)/(Mb+eps)`). First calibration (`eps=1e-8`)
   overcorrected -- caught directly from the rendered video/pressure plot,
   not just aggregate pass/fail -- and was recalibrated to `eps=1e-30`,
   which beats both that attempt and the original buggy baseline on every
   metric. `densityBand.py` has the identical pattern, **not yet fixed
   there**.

Together these let sloshingTank survive its full 7s reference duration for
the first time under the genuinely-correct formula (previously it either
diverged almost immediately from the Shepard hole, or "survived" only
because the wrong sign happened to avoid triggering it for 5+ seconds).

**The one item still open on sloshingTank:** a smaller excursion around
t=4-6.5s survives both fixes -- density [0.725,1.436], pressure peak
50,031 Pa, much closer in timing and magnitude to the original baseline
(t=5.66-6.9s, 98,177 Pa) than either intermediate attempt was. User
directly identified this in the rendered video as particles sticking to
the ceiling and later falling/being ejected into the fluid -- matching
[[antuono-pressure-switch-bug]]/§6.4's already-documented ceiling-hover
mechanism exactly. This is what §8's original clamp attempt targeted,
before that investigation got derailed into the sign-bug/Shepard-hole
chain. **This is now the most directly actionable next step** -- for the
first time, it is isolated enough (no larger failure masking it) to
investigate on its own.

**`mdbcDensityScheme='band'`** unchanged this session -- still a validated
research tool for a narrow failure mode, not a `'ramped'` replacement. Its
own Taylor-shift bug (§9.1, extrapolates to the boundary particle's mirror
reflection through the ghost) was discovered as a side effect of this
session's audit but not yet fixed or re-validated -- worth doing, since it
may explain part of §3's depth/lever-arm divergence verdict that was
previously attributed entirely to Band's architecture.

**Do not flip the default away from `'ramped'`.**

**Next steps, in priority order:**
1. Investigate the ceiling-hover mechanism itself, now that it's isolated
   -- likely where §8's clamp135 angle-cutoff idea belongs, but that idea
   was validated under the buggy sign convention and needs re-checking
   under the now-correct one.
2. Apply the same symmetric-eps fix to `densityBand.py` and re-validate it
   against §3's stress configs, to see how much of its depth/lever-arm
   divergence verdict this explains vs. its own architecture.
3. `probe_deltaSPHMarrone34.py --report`'s velocity/NaN-handling looseness
   (§7) -- still unfixed, still means a 6/6 score can hide a real blow-up.
4. A 3D implementation and a moving-boundary test (`a_b` is still
   hardcoded to zero in `english2025.py`) before any default-scheme change
   is considered.
