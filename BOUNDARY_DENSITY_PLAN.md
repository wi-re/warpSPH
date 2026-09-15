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

## Status

Open. `mdbcDensityScheme='band'` is implemented and toggle-able but should
**not** be treated as a drop-in replacement for `'ramped'` yet — it is a
validated improvement for the narrow thin-sheet/marginal-conditioning
failure mode, and a live A/B research tool, but not yet more robust than
production on the actual dam-break stress test. Do not flip the default.

`mdbcDensityScheme='english2025'` (English 2025 Eqs. 8-11, corrected sign,
§5-6) is implemented and toggle-able. On every Marrone 3.1/3.4 config tested
across two resolutions each (§6.1, §6.2, §6.3c, §6.3d), it is the most
consistently well-behaved of the three schemes and the only one that never
diverged — including where `'band'` now catastrophically fails at 2x
resolution on 3.4 (§6.3d: diverges at t*=4.21/15.66, visually confirming the
depth/lever-arm-into-a-cascade mechanism §3.1/§5 theorized). One earlier
claim is **retracted**: §6.3b's finding that `'ramped'` fails 3.4's
kinetic-energy check does NOT reproduce at 2x resolution (§6.3d) — that
looks like an nx=128 resolution artifact, not a durable property of
`'ramped'`'s mDBC scheme. (`'ramped'` still shows a real blow-up signature
at nx=256 that the `--report` script's velocity check is too loose to
catch — §7 — so `english2025`'s edge there stands, just not for the reason
originally claimed.)

**sloshingTank, tested for the first time this session (§6.4) with the full
candidate-default combo (`english2025` + PST + `fourtakas2019` DDT,
functionally close to running DualSPHysics' own defaults): NOT a clean
pass, but neither is `'ramped'` at the same settings.** `english2025`
survives the full t=7s reference duration with one real, large-amplitude
cascade at t=5.66-6.9s (sensor pressure 98,177 Pa; density swinging to
[0.65, 1.72], an actual particle ejection). The `'ramped'` baseline
(run this session, §6.4) also survives the full 7s, but shows a DIFFERENT
failure character: many isolated pressure spikes at nearly every wave
impact throughout the run (peak 158,822 Pa -- higher than `english2025`'s),
while density/velocity stay bounded ([0.857, 1.288], no ejection). Neither
is straightforwardly "worse" -- frequent-but-contained (`'ramped'`, matching
[[sloshing-rightwall-thin-sheet-kick]]'s already-documented sensitivity) vs.
rare-but-severe (`english2025`). This rules out "`english2025` broke a
clean case" -- `'ramped'` was never clean here either -- but does not
cleanly exonerate `english2025`'s specific cascade as pure case-inherent
noise vs. partly scheme-influenced. Given the small (n=1 per scheme) sample
and known timing-sensitivity of these events, not pursued further this
session.

**Still not the default. Do not flip it** — the Marrone evidence is strong
across two resolutions each and only strengthens `english2025`'s case there;
sloshingTank shows both candidate schemes have case-inherent rough edges
that are a separate investigation from this one's scope (mDBC boundary
density extrapolation specifically). This investigation has been burned
before by a candidate that looked good until a case outside the original
test family exposed a gap (`'band'` on the real dam-break, §3) — sloshingTank
did surface real behaviour neither scheme handles cleanly, just not in a
way that discriminates between them. **Next session:** (1) tighten
`probe_deltaSPHMarrone34.py --report`'s velocity check and NaN handling
(§7) so a borderline run can't score a misleading 6/6; (2) a 3D
implementation and a moving-boundary test (`a_b` is still hardcoded to
zero) before any default change is considered; (3) the sloshingTank
free-surface pinning/collision sensitivity (§6.4, both the cascade and the
second-peak mismatch) is real but appears orthogonal to `mdbcDensityScheme`
-- likely belongs in a separate investigation (surface tension/friction/
resolution), not this plan.
