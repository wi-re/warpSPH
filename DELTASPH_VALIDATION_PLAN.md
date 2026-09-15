# warpSPH — δ-SPH / δ⁺-SPH conformance audit + validation plan

> **ALWAYS run validation cases with video output (vispy).** For a WCSPH run the
> vispy encode overhead is negligible next to being able to *watch* the run and
> see where and how it goes wrong — which the scalar diagnostics alone never
> show (a density ratchet, a corner leak, a jet fragmenting all read the same in
> a `minDensity` column). It needs no HDF5 trajectory. Every regression in this
> plan that took a bisect to localise would have been obvious from the field
> video. Pass `--video`; do **not** force `--plotBackend matplotlib` (that path
> is ~50–80× slower — `probe_englishWedge.py`'s `--video` still hardcodes it and
> needs fixing). See [[export-video-on-validation-runs]] (memory).

## Why this exists

The Lobovský-scale dam break (`ACSPH_PLAN.md` §4.5) would not run cleanly under
`--scheme deltaSPH`. Tracing it turned up **several independent deviations from
the schemes the code claims to implement**, any one of which can drive a violent
free-surface impact unstable:

- the density blew up (ρ → 10¹¹) — masked, not fixed, by reverting a density-
  diffusion sign change (`790a7c7`) whose sign is in fact *correct* per both
  source papers;
- the wall leaked hundreds of particles — the weakly-compressible sound speed
  was Mach ≈ 0.5, and separately the non-periodic domain under-pads the
  near-wall neighbour search;
- the case runs RK2 with a fixed timestep and no adaptive constraints, where
  the papers specify RK4 with frozen diffusion and a 3-term adaptive `Δt`.

So before building validation cases it is worth **auditing the δ-SPH, δ⁺-SPH and
mDBC implementations against their papers and against DualSPHysics**, then
standing up the papers' own validation suites — starting with δ-SPH, whose suite
is four dam-break configurations.

Reference material now on disk:

| what | file / path |
|---|---|
| δ-SPH | `literature/marrone2011_delta-sph-violent-impact-flows.pdf` |
| δ-SPH diffusive term origin | `literature/antuono2010_*`, `literature/antuono2012_*` |
| δ⁺-SPH | `literature/sun2017_delta-plus-sph-model.pdf` |
| δ⁺-SPH shifting | `literature/sun2019_consistent-particle-shifting-delta-plus-sph.pdf` |
| δ-ALE-SPH | `literature/antuono2021_delta-ale-sph-model.pdf` |
| **mDBC (primary)** | `literature/s40571-021-00403-3.pdf` — English, Domínguez, Vacondio et al. (2022), *Comp. Part. Mech.* 9:911–925 |
| mDBC pressure cloning (Eqs. 8-11) | `english2025` (`literature/english2025_river-flows-past-bridges.pdf`) — English, Vacondio et al. (2025), *Computers & Fluids* 303:106870. Synced 2026-09-15; see `BOUNDARY_DENSITY_PLAN.md` §4 — its analytic-hydrostatic ghost-to-boundary extrapolation is a candidate fix for the depth/lever-arm amplification found there, not merely a secondary application reference. |
| DualSPHysics source | `~/dev/DualSPHysics/src/source/` — `JSphCpu_mdbc.cpp` (mDBC + m2dbc), `DualSphDef.h` / `JSph.h` (DDT variants), `examples/main/01_DamBreak` |
| existing local reference | `~/dev/diffSPH` (the kernel `wp_densityDelta.py` was ported from — plan's own note) |

`literature/ratios.pdf` is a 0-byte-text scan; ignore or re-fetch.

---

# Current state & how to resume  (as of 2026-09-10)

> **Superseded in part -- read §5.13 (2026-09-12) first.** Marrone 3.4 is now
> **6/6 over the full record** under `'hybrid'` mDBC ghost placement, and
> englishWedge's base-corner residual (the "NEXT" item below) is **fixed**
> (RMSE 0.0431 -> 0.0178). sloshingTank's divergence is root-caused to
> `mdbcNoPenShiftMode = 'derivative'` and runs the full 7 s under `'finalize'`.
> §5.13 also lists four scheme changes **rejected on measurement** -- including
> the `rho_b` clamp work §5.10 points at -- so check it before retrying any of
> them. §5.10's causal chain for the 3.4 blow-up is superseded.

The audit (Parts 1–4), the original `c₀` / integrator / mDBC-determinant-gate
work, **and** the 2026-09-10 boundary-sampling session (`sample/regular.py`,
`caseUtils/weaklyCompressible.py`, `cases/dambreak.py`,
`rigidBody/ghostParticles.py`, the `marroneSharpEdge` polygon SDF — the block
below) are all **committed** (`01060d4`). `test_physics` + `test_latticeDensity`
(219 tests) green on that commit. Explicit-`dx` acceptance-band re-runs after the
sampler change are in progress — see "Open / still to do".

## Where the δ-SPH validation stands

**The boundary handling is sound.** Root cause of the whole "won't run cleanly"
history was the *sampling*, not the scheme: nothing pinned the tank box to a
sensible lattice phase, so a lattice row landed on every flat wall and the mDBC
ghost placement was papering over a bad sample with geometry heuristics. Fixed
in three places (`dn ≤ dx` sampler + `alignInteriorDomainToLattice` +
`'gridsnap'` ghost placement — full detail in the block below), plus the
`marroneSharpEdge` obstacle SDF's internal-seam bug.

**On the fixed sampling, with the un-frozen defaults and no case-specific
tuning** (no `hydrostaticInit` / `alpha` / `nu` / frozen diffusion), every
Marrone dam-break case now runs the full record stably with essentially no
boundary leakage — see the Verified table below. Highlights:

- **§3.1 flat wall (nx=72):** δ-SPH no-PST *and* δ⁺-SPH PST both survive to
  t\*≈7.7 with **0 wall penetration**. (δ-SPH's free surface fragments to spray
  by t\*≈5 — a real no-PST method limit, not a boundary artefact.)
- **§3.4 rounded corner (nx=256):** δ-SPH no-PST and δ⁺ both stable to t\*=7,
  **0 tank + 0 fillet penetration**, indistinguishable from each other.
- **§3.4 full wedge (nx=256):** δ-SPH no-PST **and** δ⁺ both stable to t\*=7,
  **0 tank penetration**; obstacle penetration 1.81 dx (δ-SPH) / 0.77 dx (δ⁺),
  both well inside the ≤ 3 dx gate. **Marrone's own δ-SPH carries its own §3.4
  cases** — PST tightens the jet and roughly halves the obstacle pen but is not
  load-bearing for stability. cf. the toe-leak history: 30 → 4.66 → **0.77 dx**.

**mDBC ghost placement (2026-09-10):** `'gridsnap'` (`_gridSnapGhostOffsets`) is
the sole live path; nothing sets `WARPSPH_GHOST_PLACEMENT`. `_bodyNodeGhostOffsets`
(the `1e145a1` polyline + item-4 `nodeDepthCap`/`layerReach` fix) and
`_geometricGhostOffsets` are **parked** — kept in-file as troubleshooting aids,
not reached in any run. Item 2b and the item-4 regression fix are therefore
**superseded**: gridsnap has no `capMax` and does not consult the polyline, so
neither the §3.1 flat-wall regression nor the M34 toe leak they fought exists on
the live path.

## Open / still to do

- ~~**Commit the session's work**~~ **DONE** (`01060d4`).
- **Convergence** for §3.1 and §3.4 — the runs so far are the *stability +
  penetration* gate at one resolution each; the pressure-trace scoring vs
  Buchner (§3.1 P1/P2) and Colicchio/Wagner (§3.4 P1–P9), and the H/dx
  convergence pairs, are still to do.
- **§3.1 P2** median 2× low / phase-early at H/Δx=322 — probing-methodology gap
  (item 3), not a scheme error.
- **Explicit-`dx` tuned-acceptance re-runs after `01060d4` (2026-09-10) — two
  bugs found and FIXED (uncommitted, on top of `01060d4`):**

  **(A) `_gridSnapGhostOffsets` snapped mDBC ghost nodes to the wrong lattice
  phase.** `nodeDepth = ceil(d/dx)·dx` put the node an integer number of `dx`
  *past the wall surface*. But the sampler (post `alignInteriorDomainToLattice`)
  straddles a flat wall cleanly at ±0.5 dx, so fluid rows sit at `(k+0.5)·dx`
  and every gridsnap ghost node landed **0.5 dx off the fluid lattice, in the
  interstitial gap** (measured: dist-to-nearest-fluid 0.4999 dx for every
  flat-bottom-wall node at nx = 225; **0.0000 dx** for the plain mirror
  `'geometric'` uses) → one-sided/biased MLS at every wall particle every step.
  This was the `englishWedge` dp = 0.01 **9/9 → 8/9** regression (wedge
  base-corner hydrostatic RMSE 0.0150 → 0.0316, gate ≤ 0.03). **Fix
  (`rigidBody/ghostParticles.py`):** `nodeDepth = (floor(d/dx) + 0.5)·dx` — the
  fluid-particle lattice phase. Verified: gridsnap flat-wall nodes now coincide
  with fluid particles (dist 0.0000), offset stats back to the plain-mirror
  baseline (1.0 / 5.0 / 12.4 dx). Dump: `scripts/dump_sloshing_sampling.py`,
  `scratchpad/sloshing_sampling{,_fixed}/`.

  **(B) `sloshingTank` (TC10) density ratchet — culprit `80eabb9`, FIXED by
  removing its clamp.** Bulk ρ ratcheted to ≈ 1.09–1.15 and the sloshing
  motion collapsed (KE 0.028 → 7e-4, maxVel → 0.4 by t = 7 s) vs the
  2026-09-09 reference. **Not the sampler** (IC byte-identical: 12,415 ptcls,
  domain, c_s 11.05; the `9c7f878` worktree re-run reproduces the reference to
  every digit) and **not the ghost default** (ratchets under `geometric` too).
  Bisect (nx = 225, --tLimit 4): `b584170` ratchets → `01060d4` cleared;
  `d494d7f` (= `1e145a1~1`) ratchets → `1e145a1` cleared; `80eabb9` ratchets,
  **`afb6e59` clean** (densityMedian 1.003, KE 0.025, sensorRho min 0.876).
  `80eabb9` added `rho_b = torch.clamp(rho_b, min=rho0)` to the *whole* blended
  `computeMdbcDensity` result — the m2dbc anti-attraction guard. On the rolling
  free surface it pins `p_b ≥ 0` where the physics wants neutral/negative → the
  wall does net positive work each roll cycle → ρ ratchets, KE bleeds into
  compression, motion locks; and it deletes the negative-pressure wall
  transients §5.2.3 exists to validate. `80eabb9`'s own message concedes the
  revert "does not actually move Marrone 3.4's tank-wall penetration" (no
  measured upside); the `0.38 → 0.027` p95 win is `afb6e59`'s smooth blend.
  **Fix (`modules/mdbc/density2025.py`): drop the clamp line.** English 2022
  Eq. (12) has no such clamp.

  **HEAD + (A) + (B), sloshingTank nx = 225, t = 7, un-frozen, vispy video:**
  | scheme | result |
  |---|---|
  | **δ⁺-SPH (michel2022 PST)** | **full 7 s, `diverged=False`. Ratchet gone:** densityMedian 1.008 (vs 1.005 baseline / 1.09–1.14 pre-fix), KE 0.024 (vs 0.028 baseline / 7e-4 pre-fix), maxVel 2.6 — still actively sloshing. ρ over run [0.581, 1.634] ≈ baseline [0.623, 1.585]. **sensorRho [0.900, 1.243]** — negative wall transients restored (pre-fix floored at exactly 1.000 by the clamp); raw sensor min −9.1 kPa. First impact t ≈ 2.34 s (meas 2.40). Back to baseline quality. |
  | δ-SPH (no PST) | **diverges at t = 2.19 s** — a ~2 ms spatial-operator blowup as the first slam builds (maxVel 11 → 56, maxρ → 3). *Expected*: un-frozen no-PST cannot do a violent first impact (item 4c, [[validation-scheme-defaults]]). The fixes push it from the docstring's old t = 0.68 all the way to the impact, but not through it. |

  `test_physics` + `test_wallPressure` green with both fixes.
  Videos: `e42_HEADfixed_{delta,deltaplus}_vid/`, plus
  `e42_nx225_{gridsnap,geometric,9c7f878}_vid/` and `e42_bisect_*/` from the hunt.

  **Open-check re-runs on the dam-break family (HEAD + A + B + the new
  post-step mDBC-density-to-state hook, `scripts/_mdbcDensityHook.py`):**
  - **Marrone §3.1 nx = 72 (H/dx = 43), δ⁺ PST, un-frozen — CLEAN, no
    regression.** Full record t\* = 7.68, `diverged=False`, **0 wall pen / 0
    penetrating**, ρ ∈ [0.9905, 1.013] (M = 0.049 — weakly compressible held
    tight). P1 plateau 0.47 (Buchner 0.55, in the [0.45, 0.68] band); P1 raw
    peak 2.99 P\* and P2 peak 0.271 at t\* = 4.82 (band 5.2–6.1) are the
    known point-probe ring / phase-early issues (item 3), unchanged.
    `scratchpad/openchecks/m31_deltaplus/` (+ video).
  - **Marrone §3.4 nx = 256 (H/dx = 32), δ⁺ PST, un-frozen — 6/6, no
    regression.** tStar = 5, `diverged=False`, **0 tank-wall pen**, obstacle
    pen **0.72 dx** (gate ≤ 3; was 0.77 dx pre-fix — unchanged), ρ [P05, P99]
    [0.997, 1.027], KE 27 → 9 decaying. The production M34 config is unaffected
    by (A)/(B)/the hook. `scratchpad/openchecks/m34_deltaplus/` (+ video).
  - **englishWedge dp = 0.01 wedge — still 8/9, and the wedge base corner got
    *worse*: RMSE 0.0316 → 0.0431** (max\|resid\| 0.062 → 0.081; gate ≤ 0.03).
    Everything else passes (bulk 0.014, near-wall 0.019, apex 0.015, faces
    0.033, 0 pen, ρ ∈ [1.0000, 1.0026]). The phase fix (A) only touches flat
    walls; the base-corner residual is **gridsnap's corner rule** — ~40 corner
    boundary particles collapse their ghost nodes onto one interior point
    (unchanged by A) — and removing the clamp (B) made that already-bad
    concave corner worse (still water wants ρ_b ≥ ρ0 there; the unclamped
    noisy extrapolation now dips below). History: bodynode 0.0150 (9/9) →
    gridsnap+clamp 0.0316 → gridsnap+phasefix+noclamp 0.0431.
    `scratchpad/openchecks/wedge_dp01/`.

  **Verdict on (A)+(B):** they fix sloshingTank cleanly (δ⁺ back to baseline,
  ratchet gone, negative transients restored) and leave both Marrone dam-break
  configs untouched (§3.1 0 pen ρ[0.99,1.01]; §3.4 6/6, 0.72 dx). The only cost
  is englishWedge's *already-failing* concave base corner, 0.0316 → 0.0431.

  **(B) REFINED — clamp moved to the Shepard-fallback share only, tested.**
  `density2025.py`: `shepardDensity = torch.clamp(shepardDensity, min=rho0)`
  right after it's computed; the final blend
  `rho_b = w·rho_proj + (1−w)·shepardDensity` is **not** re-clamped. Re-ran both
  confirmations (nx = 225 δ⁺ t = 7 video; dp = 0.01 wedge t = 4 video):
  - **sloshingTank δ⁺ — still clean.** densityMedian 1.0053 final (vs 1.008
    blanket-removal, 1.005 baseline), KE 0.023 (vs 0.024 / 0.028), sensorRho
    min 0.932 (vs 0.900 blanket-removal — the fallback clamp bites slightly
    harder, but negative transients are still there: raw sensor min −6.8 kPa).
    Ratchet fix holds.
  - **englishWedge dp = 0.01 — base-corner RMSE 0.0431, byte-identical (4+
    sig figs, every check) to the blanket-removal run.** So the clamp's
    *placement* was never the corner's problem: at the corner `w` is not near
    0 (a clean 0/1 split would make the two clamp forms coincide) but
    intermediate, and the badly-conditioned `rho_proj` component (from
    gridsnap's collapsed corner ghost stencil) leaks through the blend
    unclamped either way. The *old* blanket clamp's 0.0316 came from brute-force
    flooring the whole blend, catching that leak as a side effect — at the cost
    of also flooring sloshing's legitimate sub-ρ0 readings. **Conclusion: the
    englishWedge base-corner regression is not a clamp question at all — it is
    `_gridSnapGhostOffsets`' concave-corner ghost placement**, a separate open
    item (target the fluid lattice at corners too, or fall back to the bodynode
    polyline mirror there). The fallback-only clamp is kept as the more
    principled form (matches the guard's actual intent, zero cost vs full
    removal on every case tried) even though it doesn't move englishWedge.
    `examples/sloshingTank/output/e42_fallbackclamp_deltaplus_vid/`,
    `scratchpad/openchecks/wedge_dp01_fallbackclamp/`.

  **(C) IMPLEMENTED (2026-09-11, uncommitted) — both changes made, sweep run,
  a real corner regression found. Not committed pending the bisect below.**
  `systems/weaklyCompressible.py`: the exponential override deleted, density
  now left at the RK integrator's own combined update (only the non-fluid
  band's post-mDBC carry-forward + a defensive NaN/negative floor added back).
  `modules/deltaSPH/densityDiffusion.py`: `computeDensityDiffusion`'s own call
  now passes `operationMode = OperationDirection.FluidToFluid` (the shared
  `computeScalarFieldDiffusion`, also used by ACSPH, keeps its `AllToAll`
  default — only δ-SPH's density-diffusion caller changed). `test_physics` /
  `test_wallPressure` / `test_deltaSPHDiffusion` (ψ_ij cancellation pin) all
  green; a quick englishWedge smoke came back 9/9 including the base corner.

  **Sweep (nx=225 δ⁺ sloshing, nx=72 δ⁺ Marrone 3.1, both full record, both
  fixes together):**
  - **sloshingTank δ⁺ — clean, arguably the best result yet.** `diverged=False`,
    densityMedian 1.009 (no ratchet), **KE final 0.031** (higher/healthier than
    every earlier fix, close to the pre-regression baseline's 0.028 — less
    over-damped). **First impact t=2.34 s (meas 2.40 s), smoothed peak 6.89 kPa
    (meas 3.6 kPa, band 2.2–13.1)** — the best match of any run this session.
    Raw sensor spiked to 156 kPa (vs ~55–70 kPa before) but it is a brief
    needle the 10 ms smoothing erases cleanly; not a global effect.
  - **Marrone §3.1 nx=72 δ⁺ — still 0 wall penetration, but a real localized
    regression: the downstream-wall run-up jet fragments at the corner.**
    maxVelocity peak 6.0 → **29.6** (5×), KE *final* 0.21 → **0.92** (4.4×, not
    just a transient), ρ range [0.99,1.01] → [0.85,1.16], P1 raw peak 2.99 →
    **85.4 P\*** at t\*=3.13. **Confirmed visually**, not just in the scalars: a
    matched-sim-time frame pair (t\*≈3.13, `ffmpeg`-extracted from both videos)
    shows the pre-fix run-up as a clean coherent jet (velocity colour-scale max
    3.5) climbing the downstream corner smoothly, and the post-fix run at the
    *identical instant* as a scattered, fragmenting cluster at the same corner
    (colour-scale max **17.4**) — the user's own read ("that's definitely a
    blow up in the corner") from the video, confirmed frame-by-frame.
    **CORRECTION (checked, not just eyeballed): the static corner ghost
    sampling is NOT obviously degenerate, so the "same as englishWedge" claim
    above doesn't hold up.** Measured at t=0 in a 12 dx box around this exact
    corner: 180 boundary particles' ghost nodes span **11 dx in both x and y**
    (149/180 distinct even at 0.1 dx bins) — `_gridSnapGhostOffsets` mirroring
    nearby particles toward a common corner *vertex* is English/Marrone's own
    prescribed rule for a plain convex corner (π/2 ≤ θ ≤ π), not a bug; the
    dense look in a zoomed crop is expected, not collapse. And a conditioning
    check (`numNeighbors`/`det(A_g)` at the ghost nodes) is uninformative at
    t=0 either way: the downstream corner is dry pre-dam-break, so
    `numNeighbors=0` there regardless of placement quality. Also: englishWedge's
    base corner is a genuinely different geometry (a re-entrant feature from a
    *merged* wedge+bed SDF) than M31's tank corner (a plain single-primitive
    box corner) — asserting they share a root cause was unjustified pattern-
    matching, not a checked claim.
    **So what we actually have, no more:** the fragmentation is real (video,
    both mine and the user's read), it is spatially at the corner, and it
    appears only after (C). Whether that's a corner-geometry defect at all, or
    just "the flow is most violent/thinnest there so a general loss of
    dissipation shows up there first," is open. **Not yet proven which of the
    two (C) changes (or both) drives it** — bisect running
    (`scratchpad/openchecks/m31_{ddtonly,densityonly}/`, each with only one of
    the two changes on top of the committed `ea55e91`); its result plus a
    conditioning check *during* the impact (not at t=0) are what would actually
    settle this, not the static dump.
  Videos: `examples/sloshingTank/output/e42_schemefix_{deltaplus,delta}_vid/`,
  `scratchpad/openchecks/{m31,m34,wedge_dp01}_schemefix/`.

  **(C) original notes, for the record:**
  - **The density state update bypasses the RK integrator.**
    `systems/weaklyCompressible.py:219` commits density as
    `ρ^{n+1} = ρ^n · exp(Δt · drhodt_lastStage / ρ_lastStage)` — using only the
    *final* RK stage's `drhodt` and `ρ`, from the *initial* `ρ^n`. Positions and
    velocities get the full RK4 stage combination; **density does not** — it is a
    single-evaluation exponential-Euler step (exact flow of `dρ/dt = ρ·g` with
    `g` frozen), consistent with a symplectic / semi-implicit Euler (KDK) step,
    not with a multi-stage RK where combining the stage rates *is* the method.
    Marrone/Sun integrate the continuity eqn with the same RK4 as the momentum
    eqn. Two consequences: (i) density is ~1st-order while v/x are 4th; (ii) the
    convex `exp` **rectifies** an oscillatory `drhodt` — `exp(+x)` grows more
    than `exp(−x)` shrinks — so acoustic ringing in `drhodt` integrates to a net
    density *gain* (a ratchet), invisible under the commented-out linear /
    Padé(1,1) form. `epsilon` is computed and clamped on the line above but
    **unused** (the Padé line is commented out) — dead code. Try: the plain
    RK-consistent `ρ^{n+1} = ρ^n + Δt·(RK-combined drhodt)`, or at least the
    Padé form, and A/B the acoustic hash on sloshingTank / Marrone 3.1.
  - **δ-SPH density diffusion runs fluid↔boundary; DualSPHysics does not.**
    `modules/deltaSPH/densityDiffusion.py` calls `computeDensityDiffusionDeltaSPH`
    (and `computeGradRhoL`, `computeGradRho`) with `OperationDirection.AllToAll`
    — so a fluid particle's DDT sum includes boundary neighbours carrying the
    **mDBC-extrapolated** wall density. DualSPHysics `JSphCpu.cpp` (~L925–939)
    sets the DDT accumulator to the `FLT_MAX` "disable" sentinel for **every**
    boundary neighbour under **all** DDT variants (`DDT_DDT`, `DDT_DDT2`,
    `DDT_DDT2Full`) — its own comment: DBC neighbours "makes it boil". The mDBC
    wall density is a phase-lagged extrapolation of the fluid's own field;
    diffusing the fluid *toward* it is a delayed self-coupling that can sustain
    / pump near-wall oscillations instead of draining them (compounds with the
    `exp` rectification above). Try: restrict the DDT / `gradRhoL` / `gradRho`
    sums to fluid↔fluid (exclude `kinds != 0` as sources) and A/B.
- **§5.2.3 sloshingTank (TC10)** grading vs the experiment — unblocked (δ⁺
  runs clean on HEAD + the two fixes). Remaining pre-existing open item: the
  raw first-impact peak overshoots the repeatability band (~62 kPa vs ≤ 13 kPa)
  because the case has no `machTarget` (c_s falls with resolution → local
  Ma ≈ 0.75 at impacts) — the soft-EOS item already noted below.
- Other explicit-`dx` cases (dambreak / M34) were re-verified inside `01060d4`;
  periodic boxes (tgv-wc / randomFlow / hydrostaticColumn) are byte-identical.
- **NEXT: `_gridSnapGhostOffsets`' concave-corner ghost collapse** — the
  englishWedge base-corner 8/9 residual (RMSE 0.0431, stable across the clamp
  A/B, so it's isolated to this). At a corner ~40 boundary particles collapse
  their ghost node onto one interior point (a degenerate, near-coplanar
  stencil `interpolateLiuLiu` can't fit well); the flat-wall phase fix (A)
  doesn't touch this path. Try: target the fluid lattice at a corner the way
  (A) does for a flat wall, or fall back to `_bodyNodeGhostOffsets`' polyline
  mirror specifically at a detected corner (bodynode gave 0.0150 / 9/9 there).

**Boundary lattice alignment + dn≤dx sampler + pristine ghost placement
(2026-09-10, item 4a):** root-caused the "abysmal" dam-break wall sampling —
nothing pinned the tank box to a sensible phase in the fixed sampling lattice,
so at nx = 72 a lattice row lands **on** every flat wall (innermost boundary
−0.95 dx, first fluid +0.06 dx against the −0.5 / +0.5 ideal), and the
ghost-placement heuristics above were compensating for a bad *sample*. Three
changes:
 * `sample/regular.py` `buildPointCloud` — the explicit-`dx` non-periodic path
   picked the *point* count as `round(l/dx)` and spanned `l` with `points−1`
   gaps, so realised `dn = l/(round(l/dx)−1) > dx` always (1.0034× x / 1.0124×
   y at nx = 72), gapping every wall band. Now (both periodic + non-periodic
   explicit-`dx`) `nCells = ceil(l/dx − 1e-3)` *intervals*, `dn = l/nCells ≤ dx`;
   points = `nCells+1` non-periodic (endpoint per face), `nCells` periodic. The
   `−1e-3` absorbs float32's ~1e-4 error on `l/dx` (nx=134: `l/dx`=144.0000076
   → 145 cells, `dn/dx`=0.993 with `−1e-6`). Canonical periodic box (`l/dx==nx`
   exact) is byte-identical to old `round`; `dx=None` untouched. Verified:
   tgv-wc / randomFlow / hydrostaticColumn counts identical; sloshingTank +10
   bnd (~1 %, fluid unchanged); dambreak ± M34 gain the endpoint layer (bnd
   +~20 %, fluid unchanged), `dn/dx`→≤1.
 * `alignInteriorDomainToLattice` (`caseUtils/weaklyCompressible.py`, called from
   `dambreak.configureScheme`, param `alignBoundaryLattice` default on) — snap
   the interior wall box onto the **nearest** lattice mid-gap (ties inward),
   move ≤ dn/2; `domain` / the lattice untouched. Moving the wall never gaps —
   the boundary/fluid rows straddling it are adjacent lattice points `dn` apart.
   **Periodicity-aware**: the periodic sampler's half-cell wrap offset flips the
   `nCells` parity that puts a mid-gap on centre; helper handles both, no longer
   skips periodic axes (still skips a genuinely unpadded / fullyPeriodic axis).
   nx = 72 → all four flat walls straddle at exactly ±0.5 dx in **both**
   `wallPeriodic` modes, ghost offsets exact `(2k+1)`, NN spacing 0.9992 dx.
   **`wallPeriodic=True` now samples virtually identically to non-periodic**
   (identical fluid count at every nx 67–400, same `dn`; differs only by a
   physically-irrelevant global dn/2 lattice stagger + one inert endpoint row).
 * `WARPSPH_GHOST_PLACEMENT` env / `_GHOST_PLACEMENT_DEFAULT` in
   `rigidBody/ghostParticles.py` — **`'gridsnap'` is the default**
   (`_gridSnapGhostOffsets`): distance `d` behind + inward normal `n` from the
   *merged* solid SDF (all boundary/obstacle SDFs `min`-combined so a wall
   running into an obstacle is one surface — `_mergedSurface`), normal a
   **central difference over `~dx`** (mollifies the `min`/`max` gradient
   discontinuity at the wedge apex / re-entrant toe), node placed at
   `ceil(d/dx)*dx` **past the surface** (layer-1 node exactly `dx` in, layer-2
   `2 dx`, … — independent of the sub-`dx` lattice phase), depth capped at
   `1.5 h`, deep layers (`d > 2 h`) → zero offset, retract `dx/2` at a time if
   inside another solid. It also covers 3D (was `_geometricGhostOffsets`).
   `'simple'` (raw capped mirror, no retract), `'bodynode'` (the `1e145a1`
   polyline + item-4 fix) and `'geometric'` (`∇sdf` + magnitude retract) are
   **parked** — reachable only via `WARPSPH_GHOST_PLACEMENT`, which nothing
   sets; kept in-file as troubleshooting aids, each isolating one piece of the
   placement, not as modes to select in normal use.

**Verified (2026-09-10, nx used as noted; aligned grid + gridsnap ghosts +
c₀-per-Marrone, no `hydrostaticInit`/`alpha`/`nu`/frozen-diffusion tuning):**
| case | scheme (unfrozen) | result |
|---|---|---|
| §3.1 flat wall, nx=72 | δ-SPH no-PST | stable to t\*≈7.7, **0 wall pen**; free surface fragments to spray by t\*≈5 (method limit, not a boundary artefact); bulk ρ [0.984, 1.033] |
| §3.1 flat wall, nx=72 | δ⁺-SPH PST | stable to t\*≈7.7, **0 wall pen**; clean plunging breaker + air cavity; ρ [0.98, 1.015] whole run; heavily dissipative (KE end 0.22 vs δ-SPH 3.9) |
| §3.4 rounded-corner-only, nx=256 (H/dx=32) | δ⁺-SPH PST | stable to t\*=7, **0 tank pen, 0 fillet pen**; ρ bulk [0.999, 1.004]; KE 42→21 |
| §3.4 rounded-corner-only, nx=256 | **δ-SPH no-PST** (Marrone §3's own config) | stable to t\*=7, **0 tank pen, 0 fillet pen**; ρ bulk [0.999, 1.004]; KE 42→21 — **indistinguishable from δ⁺** on the smooth corner |
| §3.4 full wedge, nx=256 | δ⁺-SPH PST | stable to t\*=7, **0 tank pen, obstacle pen max 0.77 dx** (gate ≤3; → 0.23 dx/5 ptcls at end); sharp-edge jet ejects cleanly, bulk overtops without leaking through. cf. the toe leak history: 30 dx (SDF-grad) → 4.66 dx (bodynode) → **0.77 dx (gridsnap + one-polygon obstacle SDF, none of the nodeDepthCap/layerReach machinery)** |
| §3.4 full wedge, nx=256 | **δ-SPH no-PST** (Marrone §3's own config) | stable to t\*=7, **0 tank pen, obstacle pen max 1.81 dx** (gate ≤3; → 0.77 dx/36 ptcls at end); same bulk flow as δ⁺, jet sheet more ragged (max\|v\| 15.6 vs 10.1 U_max·3.3), ρ bulk [0.999, 1.023]. **Marrone's own δ-SPH carries its own §3.4 cases** — PST tightens the jet + halves the obstacle pen but is not load-bearing for stability. |

**`marroneSharpEdge` obstacle SDF was `min(triangle, box)` sharing an internal
edge at `x = toe_x = -W/2 + 6H`** — an integer number of `dx` from the centre,
so a lattice column lands exactly on it, `min ≈ 0` all up the seam, the `< 0`
interior test drops that column, and fluid slides into the one-particle slot
(the 1.25 → 0.77 dx of the two wedge runs above). Fixed: the wedge+block is one
convex quadrilateral `[apex, (back_x, top), (back_x, bed), toe]` as a single
`_convexPolygonSDF` (`max` over the 4 edge half-planes; only `+ - * @ maximum`,
no new primitive). Same geometry, +31 boundary particles (the filled seam
column), 0 misplaced ghosts.

Runners in `scripts` scratch (`run_dambreak_marrone31.py`,
`run_dambreak_marrone34.py`), videos in `marrone3{1,4}_out/`. Sampling dump:
`dump_dambreak_sampling.py`.

## Probe scripts — the entry points

| script | case | key flags |
|---|---|---|
| `scripts/probe_deltaSPHMarrone.py` | Marrone 2011 §3.1 dam break | `--shifting default\|off\|on`, `--scheme`, `--nx`, `--c0Ratio`, `--cflFactor` (halve → half Δt), `--report`, `--video` (vispy by default now; `--plotBackend matplotlib` to force) |
| `scripts/probe_hydrostaticPressureProbe.py` | still-water column — validates the `surfacePressureProbes` MLS probe on the analytic profile | `--dp`, `--inset` (dx, along the wall normal) |
| `scripts/probe_deltaPlusTGV.py` | Sun 2019 §3.1 Taylor–Green | `--nx`, `--Re`, `--tLimit`, `--report` |
| `scripts/probe_deltaPlusDroplet.py` | Sun 2017 §4.2 oscillating droplet | `--Rdx`, `--periods`, `--report` |
| `scripts/probe_englishWedge.py` | English 2022 §4.1 still-water wedge | `--dp`, `--wedge` / `--no-wedge`, `--tilt`, `--tLimit`, `--report` |
| `scripts/probe_deltaSPHMarrone34.py` | Marrone 2011 §3.4 / Fig. 19 sharp-edged obstacle + rounded tank corner | `--nx` (H/dx = nx/8), `--c0Ratio`, `--tStar`, `--Re` (§3.4.2 viscous), `--initdump`, `--report`, `--traceFigure` (P1–P9 surface pressure), `--video` |
| `scripts/probe_deltaPlusShiftMagnitude.py` | measures the δ⁺ shift vs Sun Eq. (7) | — |
| `scripts/probe_deltaPlusShiftBlastRadius.py` | which registered cases use `ShiftingScheme.deltaSPH` | — |
| `scripts/probe_deltaPlusShiftSweep.py` | 3-leg (`off`/`eighth`/`eq7`) smoke sweep over the 11 affected cases | `--cases`, `--nSteps`, `--report` |
| `scripts/probe_squarePatchPSTControl.py` | PST discriminator on the rotating patch (§5.3.2a) | `--nx`, `--legs`, `--twProbe`, `--figure` |
| `scripts/probe_squarePatchValidationFigure.py` | square patch vs Sun 2019 Fig. 13 (§5.3.2b) | `--nx`, `--scheme`, `--times`, `--modes`, `--poissonInit`/`--noPoissonInit` |

## Done

- **§5.1 / §5.2.2 boundary sampling + mDBC ghost placement (2026-09-10, uncommitted)** —
  root-caused the dam-break "won't run cleanly" history to the *sampling*: `dn ≤ dx`
  sampler (`sample/regular.py`), `alignInteriorDomainToLattice` snapping the tank
  box onto lattice mid-gaps (`caseUtils/weaklyCompressible.py`, param
  `alignBoundaryLattice`), `'gridsnap'` mDBC ghost placement on the merged
  mollified surface (`rigidBody/ghostParticles.py`), and the `marroneSharpEdge`
  obstacle as a single seam-free `_convexPolygonSDF`. On the un-frozen defaults,
  no tuning: §3.1 flat wall (nx=72) and §3.4 corner + wedge (nx=256) all run to
  the full record with **0 tank-wall penetration** under **both** δ-SPH no-PST
  and δ⁺-SPH PST; §3.4 obstacle pen 1.81 dx (δ-SPH) / 0.77 dx (δ⁺), well inside
  the ≤ 3 dx gate. See "Current state". Convergence + pressure-trace scoring
  still to do.
- **Part 1–4 audit** — ψ sign (`790a7c7`), `c₀` = Sun Eq. (2) via `machTarget`,
  RK4, adaptive Δt, two-sided viscosity, mDBC determinant gate with a
  per-kernel `determinantThreshold`, `dambreak` on Wendland C2. All landed.
- **§5.3.1 TGV (Sun 2019 §3.1)** — DONE. Found the δ⁺ shift was **⅛ of Sun 2017
  Eq. (7)** (`fac6d4f`). Fixed opt-in as `ShiftProperties.sun2017Eq7Shift`
  (default on only for `--scheme sun2017DeltaSPH`). Re = 100 **and** Re = 1000
  both **5/5 at Sun's own L/Δx = 400**, including the δ⁺-2017 pathology
  (`3a51f61`). The Re = 1000 KE excursion at L/Δx = 200 was under-resolution.
- **§5.3.3 droplet (Sun 2017 §4.2)** — **7/7 at all three of Table 1's
  resolutions** (`06482e7`). Momenta beat Table 1 at R/Δx = 50/100 and converge
  at a steady ~1.4 order; Sun's Table 1 has an unexplained order jump (→ 3.3 /
  4.8) at its last refinement. Not chased.
- **§5.3.2 blast-radius sweep** — `sun2017Eq7Shift` swept clean on 10 of 11
  cases that run `ShiftingScheme.deltaSPH` (`c465983`); `impact` −4 % on
  `nnDistP01` is the one unexplained cost. **Default NOT flipped.**
- **§5.3.2a PST discriminator + shift-magnitude decision** (Open/next 1 & 2,
  resolved). `probe_squarePatchPSTControl.py` (new): plain δ-SPH develops the
  pairing/tensile instability on the rotating patch (paired 0.03 → 0.21 over
  tω, `nnP01` collapses), both shift magnitudes suppress it, `eq7` best — so
  the square patch **is** a working PST discriminator, prefers Eq. (7). The
  2500-step free-surface re-sweep (`out_deltaPlusShiftSweep_fs/`): `droplet`
  neutral, `squarePatch` better at `eq7`, `impact` a real small regression;
  **`dambreak eq7` under un-frozen `deltaSPH` breaks** (21 % voids, localised
  ρ 1122 / ‖v‖ 90762), and **frozen diffusion (`sun2017DeltaSPH`) fully
  rescues it** (paired 0.028, void 0.003). **Decision: `sun2017Eq7Shift` stays
  coupled to `sun2017DeltaSPH`, not a scheme-agnostic default** — the global
  flip would give un-frozen `deltaSPH` the 8× shift and diverge `dambreak`.
  The frozen-`eighth` cell completed the 2×2: **freezing** is the load-bearing
  factor, `eq7`-vs-`eighth` a wash once diffusion is frozen (`7f658c6`).
- **§5.2.3 English §4.2 sloshing tank (SPHERIC TC10)** — found & fixed two
  IC-generation bugs (`9c7f878`): `buildPointCloud` was taking the lattice
  spacing from the shortest domain axis (→ `sloshingTank` at 0.6× `config.dx`,
  no dp/2 stagger, c_s regressed 16.6→11); and mDBC ghost placement keyed off
  runtime fluid position (`_fluidDirectedGhostOffsets`) → scattered nodes.
  Now: sampler honours `config.dx`; ghosts placed from the boundary geometry
  alone (later superseded by `_bodyNodeGhostOffsets`, `1e145a1`). sloshing wcsph
  runs the full 7 s (was NaN at t=0.6). First Sensor-1 impact matches (t 2.37 vs
  2.40 s, 3.8 vs 3.6 kPa); negative-pressure transients captured. Open:
  `sloshingTank` needs a `machTarget` (c_s falls with resolution → later-impact
  overshoot).
- **§5.3.2b square patch vs Sun 2019 Fig. 13** — `probe_squarePatchValidationFigure.py`
  gained `--scheme` + ε_M/ε_E conservation errors (`12e5126`); added opt-in
  **Poisson pressure init** (square torsion function, `f718b78`) which replaces
  the undamped acoustic core mode with a stable negative pressure core. δ⁺ +
  Poisson at L/Δx = 200 matches Fig. 13 through **tω ≈ 3** (ε_M < 0.2 %,
  ε_E < 0.7 %, volume < 0.04 %, 4 coherent arms, noise-free field); tω = 4 arm
  fragmentation is under-resolution (needs L/Δx = 400), not scheme/init.
- **§5.1 Marrone §3.1** — §5.1.1: the case had always run δ⁺-SPH, not δ-SPH
  (`shiftProperties.active` defaults `True`); added the `shifting` param
  (`c14dd06`). §5.1.2: re-run at **H/Δx = 322** — **P1 converges to Buchner**
  (plateau 0.556 vs 0.55; the deficit was under-resolution, `2138c0f`). Eq. (7)
  *regresses* this case 9/9 → 8/9 (`7fb6222`). **2026-09-10: `1e145a1` broke
  this case catastrophically at H/Δx = 40** (its ghost `capMax` zeros deep
  flat-wall mDBC layers); an uncommitted `ghostParticles.py` fix restores it —
  δ⁺+PST un-frozen now the **best Buchner match yet** (P1\*/P2\* 0.51 / 0.25).
  Two residual holes are separate & pre-existing: H/Δx ≈ 72–160 blows regardless
  of scheme, and `no-PST + un-frozen` can't do the first impact at any Δt. Full
  account: **item 4**.
- **§5.2.1 English §4.1 wedge** — steps (a)–(d). `dambreak` gained
  `hydrostaticInit` (`5d3fe07`); mDBC ghost placement rewritten to
  fluid-directed + corner-gated (**fix A**, `17290ff`). **Wedge 9/9 at
  dp = 0.01 (H/dx = 50, English's resolution)**; RMSE corner gates
  (`982375d`). Fix (B) attempted and reverted as a no-op.
- **§5.2.2 Marrone §3.4 / Fig. 19 sharp-edged obstacle + rounded tank corner**
  — geometry added (`marroneSharpEdge` = obstacle polygon ∪ concave fillet;
  `marroneRoundedCorner` = fillet only), `probe_deltaSPHMarrone34.py`,
  obstacle/fillet-SDF penetration + `densityP99` metrics in `dambreak.diagnostics`.
  **Bulk converges H/dx = 32→64** (KE overlies, `densityP05` flat); pointwise
  jet-tip ρ/‖v‖ grow with Δx (fragmenting jet, as Marrone notes). **Rounded
  corner in isolation: 6/6 to t\* = 7, ZERO fillet penetration.** **δ⁺-SPH +
  PST 6/6** (bulk ρ P99 ≤ 1.02 vs ~1.14 for plain δ-SPH); reusable configs
  `examples/sweeps/marrone34_*.yaml`. **Plain δ-SPH leaked ~30 dx by t\* = 5 at
  nx = 256** (NOT a regression) — leak-tracked to fluid sinking through the bed
  at the **obstacle-toe re-entrant corner** where the composed-SDF-gradient
  ghost placement failed. **FIXED (`_bodyNodeGhostOffsets`, Marrone App. A /
  English §3, `1e145a1`): penetration 29.8 → 4.66 dx to t\* = 5 (≤ 1 dx through
  t\* = 3.9); englishWedge still 9/9, re-entrant-corner RMSE 10× better; all
  tests pass.** The 4.66 dx is one particle slow-creeping through the toe *apex*
  after t\* ≈ 4.5 — the fluid wedge there is too thin for any single-ray node;
  Marrone's continuity fill (borrow a neighbour's node) is the real fix,
  deferred (§5.2.3). δ⁺-SPH + PST seals M34 fully (0.58 dx) — the production
  config. `mdbcGhostRefreshEvery` removed (`76af473`). **Item 7 (2026-09-09):**
  (b) surface probes P1–P9 done (`surfacePressureProbes` + `--traceFigure`;
  plain δ-SPH edge P2 = 36.4 ρgH ≈ Wagner 36.7); (c) viscous wiring done
  (`--Re`; Re = 1000/10000 both 5/6, flow ≈ inviscid, walls still free-slip);
  (a) H/dx = 32/64 `densityP99` re-run — bulk P99 1.025 → 1.013 converges,
  wall leak *worsens* 4.66 → 7.83 dx. Open: H/dx = 128 to t\* ≈ 7.4; Colicchio
  Fig. 24 curve digitising; §3.4.2 no-slip walls. See §5.2.2.
- **`runner/media.py`** — frame-ordering bug (glob sort breaks past 100k
  steps) fixed (`0810491`).

## Open / next — in rough priority order

1. ~~**`rotatingSquarePatch` no-PST control**~~ **DONE (§5.3.2a).** Plain
   δ-SPH develops the pairing instability on the patch; the shift suppresses
   it; `eq7` best. The square patch + TGV are now a clean PST-needed /
   PST-works pair. `scripts/probe_squarePatchPSTControl.py`.
2. ~~**`sun2017Eq7Shift` as the shared default**~~ **DECIDED (§5.3.2a): no —
   it stays coupled to `sun2017DeltaSPH`.** The 2500-step free-surface
   re-sweep showed `dambreak eq7` diverges under un-frozen `deltaSPH` (21 %
   voids, localised singularity); frozen diffusion rescues it. Flipping the
   global default would hand un-frozen `deltaSPH` the 8× shift for no benefit.
   `impact`'s −4 % `nnDistP01` confirmed as a real (small) cost. Remaining
   micro-question: frozen-`eighth` run to isolate whether frozen diffusion or
   `eq7` carries the good frozen result.
2b. **Body-node mDBC ghost placement (Marrone 2011 App. A / English §3) —
   SUPERSEDED (2026-09-10) by `'gridsnap'`.** `_bodyNodeGhostOffsets` is now a
   parked path (see "Current state"); the M34 toe-leak and englishWedge
   re-entrant-corner gains it delivered are subsumed by gridsnap on the fixed
   sampling. The original 2026-09-09 note follows, for the record:
   DONE (2026-09-09), `_bodyNodeGhostOffsets` (`1e145a1`). Was the M34
   wall-leak fix (§5.2.3): the composed-`min`/`max`-SDF gradient mis-placed the
   mDBC ghosts at the obstacle-toe re-entrant corner. Replaced (2D) by a
   marching-squares polyline of `region.sdf` as the body-node set + mirror
   through the nearest polyline point. **M34 nx = 256 plain δ-SPH tank-wall
   penetration 29.8 → 4.66 dx to t\* = 5** (≤ 1 dx through t\* = 3.9; was 5.3
   climbing to 30) — 5/6; the 4.66 dx is **one particle** slow-creeping through
   the toe after t\* ≈ 4.5, still 1.7 over the ≤ 3 gate (§5.2.3). **englishWedge
   9/9** and *better* — base-corner hydrostatic RMSE 0.024 → 0.0022, wedge-face
   0.0013, apex 0.0020, wall penetration 0. All `test_physics` /
   `test_wallPressure` / `test_deltaSPHDiffusion` pass. Bisector second pass
   for θ > π **attempted and backed out** (§5.2.3) — a bisector still can't get
   a useful node at the sharp toe *apex*; that needs Marrone's continuity fill
   (borrow a neighbour's node), deferred. Left: an analytic per-primitive
   normal (exact on curves, no marching-squares grid); the continuity fill.
   The 4.66 dx / 5/6 is accepted — δ⁺+PST is the production config for M34.
3. **§5.1 Marrone P2** — median 2× low **and** ~1 t\* phase-early at H/Δx = 322.
   The user's read (2026-09-08): a **probing-methodology** gap, not a scheme
   error — P1's *raw* trace is dominated by weak-compressibility acoustic
   ringing that Marrone's φ = 90 mm disc *area integral* low-passes and a
   point/small-disc probe cannot. "Mostly good enough for now." A true
   on-wall disc integral matching Marrone's transducer is the test.
4. **Marrone §3.1 mDBC wall — the `1e145a1` regression FOUND & FIXED, plus a
   separate resolution hole (2026-09-10).** **SUPERSEDED (2026-09-10, same day)
   by the boundary-sampling fix + `'gridsnap'` — see "Current state".** The
   `1e145a1` `capMax` reject and the `nodeDepthCap`/`layerReach` fix below both
   live in `_bodyNodeGhostOffsets`, now a parked path; gridsnap has no `capMax`
   and reproduces §3.1 at nx=72 (both schemes, un-frozen, 0 wall pen) without
   any of it. The "resolution hole" (H/Δx ≈ 72–160 blow-up) and the no-PST
   un-frozen first-impact blow-up noted here were pre-fix; re-check on the live
   path is part of the §3.1 convergence work. Investigation kept below for the
   record; artefacts in `scratchpad/m31ab/`.

   **(a) `1e145a1` (body-node mDBC ghost placement) is a real regression — FIXED.**
   Its `_bodyNodeGhostOffsets` rejected any ghost offset with `|r_b − r_g| >
   1.5·hMean ≈ 6 Δx` (`tooFar`); the retract loop then can't satisfy
   `|off| ≤ capMax` **and** "node stays in fluid" for a deep boundary layer, so
   the offset **zeros → the deep 2–3 layers of a `band = 5` flat wall read rest
   density → wall pressure collapses at depth**. Under a fast near-wall flow
   (the §3.1 downstream-wall run-up jet, t\* ≈ 4.4) the wall gives ~half its
   back-pressure and the jet penetrates → monotone runaway (866 Δx at nx = 67).
   Latent: englishWedge + `test_physics`/`test_wallPressure` never caught it
   (still water / no violent impact). Confirmed by a 2×2 A/B (nx = 67, HEAD vs
   `1e145a1~1` ghostParticles.py, both schemes): pre-`1e145a1` clean both
   schemes (pen ≤ 1.1 Δx), HEAD diverges. The ghost-offset dump is Δx-dependent
   — byte-identical at nx = 120 (the `capMax` doesn't bind there) but at nx = 67
   HEAD caps at 6 Δx where pre-`1e145a1` spreads to 13 Δx — which is why it
   first read as a pure "resolution hole".
   **Fix (`rigidBody/ghostParticles.py`):** the offset-magnitude reject becomes
   two *node-position* bounds — `nodeDepthCap = 1.5·hMean` slides a too-deep
   mirror node inward along its own ray (matching `_geometricGhostOffsets`), and
   `layerReach = 2·hMean` zeros only genuinely inert deep-interior ghosts (fluid
   can't reach them). A clean, deep, correctly-mirrored flat-wall node is kept
   regardless of offset length. Result at nx = 67 (H/Δx = 40):

   | config | pre-fix (`1e145a1`) | with the fix |
   |---|---|---|
   | δ⁺-SPH ⅛-Eq.(7) PST, **un-frozen** | (plan §5.1.2: 6/9) | **clean, pen 0.69, P1\*/P2\* 0.51 / 0.25** (Buchner 0.55 / 0.28) — best match yet |
   | δ-SPH frozen | blows up, pen 866 | clean, pen 0.87, P1\* 0.43 |
   | δ-SPH **no PST, un-frozen** | — | **first-impact blowup (see (c))** |

   *Cost:* M34 (§5.2.2) plain-δ-SPH obstacle-toe leak ≤ 1 Δx → **~6–7 Δx
   bounded** (5/6, stable, plateaus — not the pre-body-node 30 Δx). δ⁺-SPH + PST,
   M34's production config, is unaffected. Two discriminators to keep both
   clean were tried and shelved (`smooth = ∠(mirror, ∇sdf)`; an extrapolation-
   lever cap in `computeMdbcDensity`) — "is the near-wall density field linear
   enough for a deep 1st-order extrapolation" is flow-dependent, not geometric.

   **(b) A SEPARATE hole at H/Δx ≈ 72–160.** With the fix, nx = 67 (40) and the
   plan's nx = 536 (322, δ⁺-Eq.(7) frozen, survived to t\* ≈ 7.7 at 5 Δx) are
   OK, but **every config blows up at nx = 120 (H/Δx = 72)** in t\* ≈ 5.5–6.3 —
   full Eq.(7) frozen, ⅛-Eq.(7) un-frozen, no-PST. The fix only *delays* it
   (t\* ≈ 3 → 6). **Not a sampling artefact:** the nx = 120 IC is a perfectly
   regular lattice (NN spacing 1.003 Δx, zero scatter), correct 2H × H column, 5
   clean boundary layers, sane ghost offsets, 0 ghosts outside the domain —
   structurally identical to nx = 67 (`m31_initsampling_nx{67,120}.png`). So
   it's genuinely in the mDBC-wall *dynamics* at that Δx band. This is the plan's
   old item-4 "scaled the wrong way / non-monotonic" — real, still open, likely
   `modules/mdbc/` (density extrapolation / determinant gate at intermediate Δx).

   **(c) `δ-SPH, no PST, un-frozen` cannot do the §3.1 first impact — and it is
   NOT a CFL issue.** Blows up at t\* ≈ 2.5–2.8 regardless of the ghost fix;
   **halving Δt (`--cflFactor 0.15`) makes it worse** (blows at t\* ≈ 2.4, pen
   3054 vs 2035). A fundamental spatial-operator instability at the
   near-discontinuous impact — PST *or* frozen diffusion is required to survive
   it. Marrone's own δ-SPH *is* frozen (RK4 + frozen diffusive terms), so their
   "plain δ-SPH no PST" is the frozen leg; the un-frozen no-PST combination
   doesn't exist in their setup. Tension with [[validation-scheme-defaults]]'s
   "no frozen default" — for violent-first-impact cases the un-frozen default is
   δ⁺-SPH + PST, which does work.

   **(d) latent, at every resolution:** the sampler gives **no dp/2 fluid↔wall
   stagger** — the lowest fluid particle sits ~0.04 Δx *on* the bed surface and
   the first boundary layer is a full ~1 Δx below it, not the English/Marrone
   dp/2-each-side arrangement. Not the (b) trigger (same at 40 and 72) but a
   plausible mDBC-roughness contributor worth a separate look. Also present at
   every resolution: ~25 boundary particles per box corner mirror their ghost
   onto the single corner vertex (a degenerate cluster; `1e145a1`'s cap culled
   it, the fix keeps them).

   *Also this session:* video runs were forced onto the matplotlib backend by
   the probe scripts, overriding the runner's own default (`display.py`: "2D
   goes to vispy"). For an nx = 67 §3.1 video that was **~3000 s of a 3760 s
   wall time**. `probe_deltaSPHMarrone{,34}.py` no longer force it (vispy via
   headless EGL works here); `--plotBackend matplotlib` forces the old path.
   `probe_deltaSPHMarrone.py` also gained `--cflFactor`. Per
   [[validation-scheme-defaults]], its `--scheme sun2017DeltaSPH` (frozen)
   default should change to an un-frozen default.
5. **§5.2.1 (e)** — a tilted flat plate in still water: the isolated test of
   fix (A) with no corner. `probe_englishWedge.py --tilt DEG` (wire the plate
   geometry — currently `--tilt` only rotates the wedge).
6. **§5.2.1 fix (B) faithful version** — reflect the corner ghost through the
   obstacle-polygon *vertex* (English Fig. 1c/d); needs the obstacle geometry
   plumbed into `addBoundaryGhostParticles`. Deferred — (A) clears the wedge at
   English's resolution, and the dp = 0.02 residual is under-resolution.
7. **§5.2.2 Marrone §3.4 — probes + viscous + fine Δx.** *(b) DONE* — general
   surface-point MLS probe (`surfacePressureProbes`) + Fig. 19's P1–P9 +
   `--traceFigure`: edge P2 = 36.4 ρgH on Wagner's 36.7 for plain δ-SPH (δ⁺+PST
   damps it below), roof P5 ≈ 12.7 at the t\* ≈ 5.5 re-impact, fillet barely
   wets by t\* = 7.4. Positions read approximately from the sketch; Colicchio
   Fig. 24 curve-for-curve = digitising, not done. *(c) DONE (wiring)* —
   `inviscid`/`nu`/`alpha` plumbed to `diffusionParams`; `--Re` sets
   ν = √(gH)·H/Re; Re = 1000 / 10000 both 5/6, flow within noise of inviscid
   (as Marrone states). Walls still free-slip (no-slip half deferred). *(a)
   PARTIAL* — H/Δx = 32/64 `densityP99` re-run: bulk P99 **1.025 → 1.013**
   converges; **H/Δx = 128 to t\* ≈ 7.4 (nx = 1024, multi-GPU-hour) still
   open.** New finding: plain δ-SPH tank-wall leak *worsens* with resolution
   (4.66 → 7.83 dx) even with body-node ghosts — the toe-apex creep needs
   Marrone's continuity fill (§5.2.3). δ⁺+PST unaffected (6/6, wall pen 0.83
   dx), stays production.
8. **§5.2 the rest** — English §4.2 (`sloshingTank` / TC10 — **§5.2.3 first
   pass done**: 2 IC bugs fixed, first impact matches, transients captured;
   left: `sloshingTank` `machTarget`, dp = 0.002, the mDBC-vs-DBC three-row
   contrast, §3.4.2 viscous); Marrone §3.2 / §3.3; English §4.3 (3D dam break
   vs a cuboid).
9. **Marrone §3.1 loose ends** — `c₀ = 20√(gH)` cross-check; DualSPHysics
   `01_DamBreak` cross-validation; H/Δx = 80 for the Fig. 5 convergence pair.
11. **Perf** — sustained step cost ran ~1.85× the 40-step benchmark on the
    H/Δx = 322 runs. Separate investigation, not a validation blocker.
12. **Frozen diffusion** — still only the `sun2017DeltaSPH` default, no general
    path. Perf only (Sun says correctness doesn't need it).

The detailed narrative for each is in §5.1 / §5.1.1 / §5.1.2 / §5.2.1 / §5.2.2 /
§5.3.1–3 below. The section immediately after this one is the **historical**
first-session status (mDBC determinant gate, kernel choice) — kept for the
reasoning trail, not a to-do list.

---

# Status — first-session mDBC / c₀ / kernel work  (historical, all committed)

Steps 2–4 and part of 5 of Part 6, on the `dambreak` case's δ-SPH path:

| change | file(s) | effect |
|---|---|---|
| **Sun Eq. (2) sound speed** — `machTarget` param → `c₀ = √(2gH)/Ma` (default path 0.1), `dt` from the acoustic CFL. Legacy back-solve kept behind `--targetDt`. | `cases/dambreak.py`, `modules/timestep/weaklyCompressible.py` (`setupWeaklyCompressibleTimestep` gains `cSound`/`uMaxExpected`/`machTarget`) | Mach 0.5 → 0.1; runs genuinely weakly compressible at any Δx |
| **adaptive Δt on the δ-SPH path** — `dambreakTimestep` now dispatches deltaSPH to `computeTimestep` (Sun Eq. 5: min of viscous / acoustic / **acceleration**). Fixed a real bug: `computeTimestepWeaklyCompressible` read `systemUpdate.velocities` where the update carries `.dvdt`, so the acceleration term never engaged. | `cases/dambreak.py`, `modules/timestep/weaklyCompressible.py` | acceleration limit now active through the impact |
| **RK4** — `integrationScheme` default `rungeKutta2` → `rungeKutta4` | `cases/dambreak.py` | Sun §2 integrator. (Frozen diffusion not done — perf only.) |
| **two-sided viscosity** — δ-SPH scheme calls `computeVelocityDiffusion(approachOnly=False)` | `schemes/deltaSPH.py` | Marrone (5b) / Sun (1) `π_ij` is every pair, not just approaching; **blast radius: every deltaSPH case** — regression sweep running |
| ~~**mDBC MLS threshold** — English Eq. (12) bulk path lowered `numNeighbors > 9` → `> 4`~~ **REVERTED to 9** | `modules/mdbc/density2025.py` | The MLS path has no conditioning guard; at 5–9 one-sided neighbours (thin dam-break front over the dry bed) it blows up `∇ρ` → explosive `P_b` at c₀ = 40√(gH). **This was the Marrone §3.1 pre-impact blow-up** (§5.1). Reinstating 9 makes H/Δx = 40 quiet through first impact. |
| **ψ sign un-reverted** — back to `790a7c7`'s correct `−grad_ij − rho_ij` (Marrone Eq. 6 = Sun Eq. 4) | `modules/deltaSPH/wp_densityDelta.py` | `tests/test_deltaSPHDiffusion.py` 4/4 green again |

**First result** (physical Lobovský geometry, nx=80, correct ψ + all of the
above, to t\* ≈ 6.9 through the impact + run-up): **no divergence**, ρ ∈
[0.976, 1.035] at every 10 % sample point (genuinely weakly compressible),
`vmax` 2–6. Over-time extremes carry a brief transient (ρ dip to 0.49, one
`vmax` ~30 spike, 6 particles leaked ~17 Δx) at the sharp first-impact instant,
which recovers (ρ back to [0.998, 1.004] by t\* = 4.6). So the **correct**
δ-SPH operator is stable here once the scheme is set up per the papers — the
`# PSI-REVERT` over-diffusion band-aid is not needed. Residual transient is
next (mDBC extrapolation sign — Part 3 — and/or the SPH sharp-impact spike).

**mDBC extrapolation sign — audited, CORRECT** (`scripts/probe_mdbcExtrapolationSign.py`):
- `interpolateLiuLiu` returns the standard `+∇f` (a synthetic linear field
  `f = a·x + b` recovers `∇f = a` to 2e-6; `|∇f + a| = 4.7`), **not**
  DualSPHysics' negated-in-the-solve convention.
- ghost placement is `r_g = r_b − ghostOffset` (verified exactly), so English's
  `(r_b − r_g)` is `+ghostOffset`.
- `density2025.py`'s `relPos = −ghostOffset; drho = −relPos·∇ρ; rho_proj =
  rho_interp + drho` assembles to `ρ_g + (r_b − r_g)·∇ρ_g` — **English et al.
  2022 Eq. (12) exactly** (stored boundary density tracks the `+ghostOffset`
  prediction to 3.6e-3; 0.47 from the flipped form). `wallPressure.py`'s
  `p_proj` uses the identical pattern → also correct.
- (The probe's *absolute* disagreement with the analytic hydrostatic profile
  is the `c₀`-too-soft / uncalibrated-bulk problem of the `hydrostaticColumn`
  deltaSPH path, not the sign — that's the English §4.1 wedge validation, Part 5.)

**mDBC determinant/condition gate — implemented, replacing the `threshold = 9`
hack** (`modules/liu/interp.py` `interpolateLiuLiu`): the DualSPHysics
`determlimit` idea (`JSphCpu_mdbc.cpp`, `|det(A_g)| >= 1e-3`, else 0th-order
Shepard, else rest value) is now the function's own gate, combined with the
existing neighbour-count floor into a `wellConditioned` mask it returns
alongside `res`/`A_g`/`b`. This is a systemic fix, not a one-off: the same
hand-tuned `numNeighbors > 9` heuristic had been copy-pasted into three call
sites — `modules/mdbc/density2025.py` (this plan's own hack), `modules/
incompressible/wallPressure.py` (`_LINEAR_MIN = 9`), and `modules/mdbc/
velocity.py`'s `extendedVelocity` (`BCType.extended`) — all three now consume
`wellConditioned` instead. Fixed a real bug found while editing `velocity.py`:
its low-neighbour fallback branch read `vel[ghostMask]` (always zero — ghost
rows are never written elsewhere in that function) instead of `vel[bIndices]`,
silently discarding the Shepard fallback value just computed on the previous
line for every point that didn't clear the old threshold.

**H/Δx = 15 improved, but H/Δx = 40 REGRESSED — this change is NOT landed.**
Confirmed via a same-args A/B (`git stash` the changes, rerun, `git stash
pop`), not a fluke or reporting artifact:

- **H/Δx = 15** (`--nx 25 --c0Ratio 40 --tLimit 3.2`, `t* ≈ 12.9` reached,
  `scripts/old/out_deltaSPHMarrone_gatecheck/`): previously the recorded "worst
  case" — no violent explosion, but *"a milder disturbance still builds by
  t\* ≈ 2.7."* With the gate: no divergence to t\* ≈ 12.9, density settles to
  `[0.9998, 1.0007]` by t\* ≈ 4 and stays there (a brief `ρ ∈ [0.86, 1.15]` /
  `vmax ≈ 28` transient at t\* ≈ 2.3 — the known separate SPH sharp-impact
  spike). Genuine improvement.
- **H/Δx = 40** (`--nx 67 --c0Ratio 40 --tLimit 0.9`, identical args run
  against this session's gate code and, separately, a `git stash`-reverted
  baseline): the baseline reproduces the plan's previously-recorded whole-run
  `max ‖v‖ = 5.86` almost exactly and stays smooth (`v` 3–5,
  `ρ ∈ [0.979, 1.021]`) through t\* = 3.64. **The gate-code run instead spikes
  to `vmax = 38`, `P1* = 45`, `ρ ∈ [0.85, 1.22]` starting at t\* ≈ 3.10** and
  had not recovered by the end of the (short) window.

**Root cause:** DualSPHysics' own mDBC has *no* separate neighbour-count
floor — it gates purely on `|det(a_corr)| >= determlimit`. This session's
first attempt collapsed the repo's existing two-stage guard
(`interpolateLiuLiu`'s internal `neighbor_threshold` *and then* each caller's
separate `> 9` check) into one `wellConditioned = (count > 4) & (|det(A_g)|
>= 1e-3)` — net *looser* than before, since 4 < 9. Points with 5–9 neighbours
near the impact front that happen to clear the borrowed `1e-3` determinant
threshold now get promoted to the full MLS extrapolation, which the old `> 9`
floor had been silently protecting against. DualSPHysics' constant — tuned to
their own kernel/precision/unit convention — is not strict enough here to
substitute for that margin on its own.

A provably-safe fallback exists but doesn't reach the original goal: AND the
new `wellConditioned` with the old `count > 9` floor (`(count > 9) &
wellConditioned`). That can only ever *remove* points relative to the old
`count > 9` gate (catching the coplanar-but-numerous-neighbour case Part 3
also flagged) and can never admit a low-count point the old code rejected, so
it cannot reproduce this regression — but it also does not extend the MLS
path below 9 neighbours, i.e. it would not have fixed H/Δx = 15. Landing it
would be a small, low-risk win over the status quo without solving what this
part of the plan set out to solve; a real fix needs an empirically-calibrated
`determinantThreshold` for this codebase's kernel/precision, not 
DualSPHysics' borrowed constant. **Not implemented yet — decision pending.**

**Search-radius parity checked and ruled out as the cause.** DualSPHysics'
mDBC ghost-node gather does *not* widen the interpolation stencil: it reuses
`KernelSize = KernelK·h`, the identical radius every ordinary particle
interaction (density sum, forces, viscosity) already uses — `KernelK = 2.0`
for Wendland (`FunSphKernel.h`), so `KernelSize = 2h`, one normal compact
support. `interpolateLiuLiu`'s default `supportScale = 1.0` does the same
here — it reuses `referenceParticles.supports` unmodified, not an enlarged
radius. And the two setups match in absolute size: this case's `n_h = 4.0` is
literally `support / Δx` (`modules/adaptiveSupport/optimalSupportOwen.py`'s
`n_h_to_nH`), giving support `= 4Δx`; DualSPHysics' own Wendland-C2-at-`h/dp
= 2` setup gives support `= 2h = 4dp` — the same 4·spacing. So the
neighbourhood size and expected neighbour count are matched to the reference
almost exactly; a wider/narrower search radius is not what's driving the
scale mismatch.

What *is* still mismatched, and was already an open Part 1 item before this
determinant-gate work started: this case runs **Wendland C4** (`Wendland4`),
DualSPHysics/Sun run **Wendland C2**. Same support radius, same neighbour
count, but a different kernel falloff shape (C4 sits flatter near the centre,
sharper at the cutoff) — which changes the actual magnitude of the
kernel-weighted moment matrix `A_g` even at matched geometry, so
`|det(A_g)| >= 1e-3` does not mean the same thing in both codebases. That
kernel-shape mismatch, not the search radius, is the more likely reason
DualSPHysics' literal constant doesn't transfer; recalibrating
`determinantThreshold` (or switching this case to Wendland C2 to match the
reference kernel outright, which Part 1 already flagged as a deviation worth
revisiting) are the two live options.

**Kernel-shape hypothesis CONFIRMED** — same A/B (`--kernel Wendland2` on
`probe_deltaSPHMarrone.py`, matching support radius via `n_h=4.0` both ways):
under Wendland C2, `determinantThreshold=1e-3` works cleanly at *both*
resolutions simultaneously — H/Δx=40 (`vmax=6.63`, `ρ∈[0.990,1.018]`,
`maxPen=0.37`) matches the old C4 baseline (`vmax=5.86`) with **no** spike
near t\*≈3.1, and H/Δx=15 (`vmax=6.80`, `ρ∈[0.953,1.042]`) is cleaner even
than the earlier C4+gate result (`vmax=28.3`). Confirms this is a kernel-shape
effect, not a search-radius one, and independently validates the Part 1 audit
item that this case should be on Wendland C2 to match Marrone/Sun/DualSPHysics
in the first place.

**Deriving a Wendland C4 `determinantThreshold`** (`scripts/
probe_mdbcKernelDeterminantScale.py`, new): computes `det(A_g)` for the same
synthetic particle geometry under both kernels at matched support (no
timestepping — a few ms). Two fixed reference points (a filled half-disc
"healthy" point, a maximally coplanar "degenerate" band) gave inconsistent
C4/C2 ratios (1.09 vs 2.68) because both sit 15–35x *below* 1e-3 for either
kernel — too degenerate to be diagnostic of behaviour *at* the cutoff.
Sweeping the neighbour band's vertical half-width instead (parametrising "how
thin is the sheet," the actual failure axis) finds a stable **C4/C2 ratio of
≈ 1.8** across the whole well-resolved range (16–56 neighbours; ratio decays
toward 1.0 only as the geometry approaches a full, isotropic disc). The
crossing-point method (locate where C2's own `det` hits `1e-3`, read C4's
`det` at that same geometry) lands close to the same order (`1.47e-3`) but is
noisier — the crossing falls at only 8–13 neighbours, where a coplanar
lattice's determinant is jitter-dominated, not a clean smooth function of
geometry. **Recommended: `determinantThreshold ≈ 1.8e-3` for Wendland C4**
(the robust sweep-region value, `1e-3 * 1.79`).

**Validated — CONFIRMED.** Monkeypatched `interpolateLiuLiu`'s default (no
source changes) to `1.8e-3` and reran both resolutions on the case's default
Wendland C4:

| case | C4, threshold=1e-3 (broken) | **C4, threshold=1.8e-3 (derived)** | C2, threshold=1e-3 (reference) |
|---|---|---|---|
| H/Δx=40 | vmax=38.0, ρ∈[0.855,1.221] | **vmax=7.66, ρ∈[0.970,1.023], no spike near t\*≈3.1–3.6** | vmax=6.63, ρ∈[0.990,1.018] |
| H/Δx=15 | vmax=28.3, ρ∈[0.86,1.15] | **vmax=9.49, ρ∈[0.960,1.049], maxPen=0** | vmax=6.80, ρ∈[0.953,1.042] |

Both resolutions are simultaneously clean under the derived threshold,
tracking the C2 reference numbers closely while keeping the case's original
kernel. `determinantThreshold ≈ 1.8e-3` is the recommended Wendland C4 value
at `n_h = 4.0` (support = 4·Δx) — **not yet wired into the actual call sites**
(`density2025.py` / `wallPressure.py` / `velocity.py` still take
`interpolateLiuLiu`'s bare `1e-3` default; only a scratch monkeypatch was
used for this validation, no tracked source file changed since the last
Status entry).

Two open decisions before this is ready to land:
1. **How to wire the threshold in** — a per-caller `determinantThreshold`
   keyed off `config.kernel` (a small `{KernelFunctions.Wendland2: 1e-3,
   KernelFunctions.Wendland4: 1.8e-3, ...}` lookup, extended as other kernels
   are exercised through mDBC), vs. hardcoding `1.8e-3` as `interpolateLiuLiu`'s
   new bare default (simpler, but wrong for any future Wendland C2 caller,
   and silently wrong for any other kernel nobody has calibrated).
2. **Keep Wendland C4 (with the derived threshold) or switch this case to
   Wendland C2 outright** (already an independent Part 1 audit item — Sun/
   DualSPHysics/Marrone all use C2). The C2 numbers were marginally cleaner
   at both resolutions in every A/B run this session; C4 was never a
   deliberate choice recorded anywhere in the plan, so there's no known
   reason to prefer it over matching the reference kernel directly.

**LANDED.** Both decisions resolved (kernel-keyed lookup; switch to C2):

- `modules/liu/interp.py`: `determinantThreshold` is now `Optional[float] =
  None`, resolved per-call from `_DETERMINANT_THRESHOLDS[config.kernel]`
  (`{Wendland2: 1e-3, Wendland4: 1.8e-3}`, falling back to the C2 value for
  any uncalibrated kernel) when the caller doesn't pass one explicitly —
  `density2025.py` / `wallPressure.py` / `velocity.py` / the `dambreak.py`
  pressure probe / `probe_mdbcExtrapolationSign.py` all take the lookup
  automatically, no call-site changes needed.
- `cases/dambreak.py`: default kernel `Wendland4` → `Wendland2`, matching
  Marrone/Sun/DualSPHysics (support radius unchanged — `n_h=4.0` already gave
  the same `4·Δx` as DualSPHysics' `h/dp=2`).
- New: `scripts/probe_mdbcKernelDeterminantScale.py` (the calibration sweep,
  kept for calibrating any future kernel through this gate) and a `--kernel`
  override on `scripts/probe_deltaSPHMarrone.py` (kept for future kernel
  A/Bs).

**Final end-to-end confirmation**, production code path, no monkeypatch or
override — the case's plain new defaults:

| case | vmax | ρ range | maxPen |
|---|---|---|---|
| H/Δx=40 (`--nx 67 --tLimit 0.9`) | 6.63 (t\*≈2.80) | [0.990, 1.018] | 0.37 |
| H/Δx=15 (`--nx 25 --tLimit 3.2`) | 6.80 (t\*≈2.52) | [0.953, 1.042] | 0.02 |

Both exactly reproduce the earlier explicit `--kernel Wendland2` A/B numbers
— the wiring is correct, not a fluke of the override path. `tests/
test_wallPressure.py`, `tests/test_deltaSPHDiffusion.py`, and `tests/
test_physics.py -k dambreak` all green under the new defaults. Uncommitted —
7 modified files (`interp.py`, `density2025.py`, `wallPressure.py`,
`velocity.py`, `cases/dambreak.py`, `probe_deltaSPHMarrone.py`,
`probe_mdbcExtrapolationSign.py`) + 1 new (`probe_mdbcKernelDeterminantScale.py`).

**Not done:** the `c₀` rework for *other* WCSPH cases (only `dambreak` wired);
frozen diffusion; the validation cases themselves (Part 5) beyond §3.1's first
pass; the c₀ = 20√(gH) cross-check; the DualSPHysics cross-validation run;
H/Δx = 80 (`--nx 134`) for
the Fig. 5 convergence pair.

---

# Part 1 — δ-SPH (Marrone et al. 2011): equation-by-equation

Marrone 2011 Eqs. (5)–(7). `r_ji = r_j − r_i = −x_ij` in the paper's convention.

```
Dρ_i/Dt = ρ_i Σ_j (u_j − u_i)·∇_iW_ij V_j  +  δ h c₀ Σ_j ψ_ij·∇_iW_ij V_j        (5a)
Du_i/Dt = −(1/ρ_i) Σ_j (p_j + p_i) ∇_iW_ij V_j  +  f_i
          +  α h c₀ (ρ₀/ρ_i) Σ_j π_ij ∇_iW_ij V_j                                (5b)
Dr_i/Dt = u_i ,   p_i = c₀²(ρ_i − ρ₀)                                            (5c)

ψ_ij = 2(ρ_j − ρ_i) r_ji/|r_ij|²  −  [⟨∇ρ⟩^L_i + ⟨∇ρ⟩^L_j]                       (6)
⟨∇ρ⟩^L_a = Σ_b (ρ_b − ρ_a) L_a ∇_aW_ab V_b
L_a      = [ Σ_b (r_b − r_a) ⊗ ∇_aW_ab V_b ]^{-1}
π_ij     = (u_j − u_i)·r_ji / |r_ij|²                                           (from Sun Eq. 3; Marrone's a·h·c₀·π form)
p_G      = Σ_{j∈fluid} p_j W^MLS(r_j) V_j  +  2 d ρ f·n                          (7)  fixed-ghost wall pressure, Neumann
```

Constants: **δ = 0.1** (not tunable — narrow validity range, Antuono 2012);
**α = 0.02** (Marrone's stated minimum stable value; Sun 2017 later uses 0.01);
Gaussian kernel, **h = 1.32 Δx** (Marrone) — Sun/DualSPHysics use Wendland C2 at
h/Δx = 2. Free-slip walls for the impact cases; no-slip only for the viscosity
sub-study (§3.4).

## 1.1 What to check in the repo

| paper item | repo location | check |
|---|---|---|
| (5a) continuity divergence | `modules/momentum/` (`WarpOperation.Divergence`), `computeMomentum` | sign, `ρ_i` prefactor, volume weight `V_j = m_j/ρ_j` |
| (5a) δ-term prefactor `δ h c₀` | `modules/deltaSPH/densityDiffusion.py` | is it `δ h c₀` exactly, with `h` the smoothing length (not support radius)? `c₀ = fluid.fixedSoundSpeed`? |
| (5b) pressure gradient `(p_i+p_j)` symmetric | `modules/pressure/surfaceAware.py`, `computePressureForceSurfaceAware` | the `PressureForceScheme` default this case uses; the `−1/ρ_i` vs `Σ m_j(p_i/ρ_i²+p_j/ρ_j²)` form (equivalent only at constant mass + invariant density) |
| (5b) artificial viscosity `α h c₀ (ρ₀/ρ_i) π_ij` | `modules/deltaSPH/velocityDissipation.py` + `wp_viscosityDelta.py` | ⚠ **two issues found.** (a) `computeVelocityDiffusion(..., approachOnly=True)` is the default and it **clamps `μ_ij ≤ 0`** — i.e. only *approaching* pairs are damped. Marrone Eq. (5b) / Sun Eq. (1) `π_ij = (u_j−u_i)·(r_j−r_i)/|r_j−r_i|²` is **two-sided, no clamp**. Approach-only is Monaghan-1992 *shock* viscosity, not the δ-SPH linear term — it under-damps exactly the tensile/shear regions a violent impact grows. `approachOnly=False` already exists (`ACSPH_PLAN.md` decision 5) — the δ-SPH path should use it. (b) confirm the coefficient is `α c_s h_i / kernelXi` with the `kernelXi` giving the *smoothing length* `h` consistent with the `δ h c₀` density term, and check whether the `ρ₀/ρ_i` factor is present (`wp_viscosityDelta.py` line ~127 shows `factor = alpha*c_s*hi/kernelXi` — no obvious `ρ₀/ρ_i`). |
| `inviscidAlpha` default = **0.01** | `moduleConfigurations/weaklyCompressibleDiffusionParams.py` | Sun 2017 value, fine; Marrone 2011 says 0.02 is the floor for *its* setup. Note, don't necessarily change. |
| (6) ψ operator | `modules/deltaSPH/wp_densityDelta.py` `DensityDiffusionScheme.deltaSPH` | **the sign** — see Part 4. Also: `L_a` renormalisation must be **off** on this operator's `∇W` (the projected/unprojected equivalence breaks with `L` in front — `ACSPH_PLAN.md` Part 3, `scripts/probe_deltaSPHPsiProjection.py`). Confirm `useGradientRenormalization=False` on the δ-term's own `∇W`. |
| (6) `⟨∇ρ⟩^L`, `L_a` | `modules/density/gradRhoL.py`, `computeGradRhoL` / `computeRenormalizationMatrices` | matches `L_a = [Σ (r_b−r_a)⊗∇W V_b]^{-1}`; the `field=` generalisation from `790a7c7` did not change the density path |
| (5c) EOS `p = c₀²(ρ−ρ₀)` | `modules/eos/weaklyCompressible.py` | ✅ **verified** — default `EquationOfState.isoThermal` is exactly `c_s²(ρ−ρ₀)` (Marrone/Sun form); `dambreak` does not override `eosType`. (Note DualSPHysics DBC uses Tait γ=7 — matters only if cross-checking pressures against DSPH; a γ=7 Tait at Mach 0.5 explodes far worse than linear, so keep isoThermal.) |
| (5c) `c₀` selection | `modules/timestep/weaklyCompressible.py` `setupWeaklyCompressibleTimestep` | **Sun Eq. (2): `c₀ ≥ 10 max(U_max, √(p_max/ρ₀))`.** The repo instead *back-solves* `c₀` from a fixed `targetDt` via the acoustic CFL, giving Mach ≈ 0.5 at small Δx. **This is a real bug for any physical-scale case** — rework so `c₀` is set from the expected velocity and `Δt` follows. |
| (7) fixed-ghost wall pressure | `modules/mdbc/*` | see Part 3 — the repo's `computeMdbcDensity` resembles DualSPHysics **m2dbc** (pressure clone), not Marrone Eq. (7) MLS nor English 2022 density-extrapolation mDBC |
| integrator | `cases/dambreak.py` `integrationScheme='rungeKutta2'` | **Sun 2017 §2: RK4 with frozen diffusive terms** (diffusion + viscosity evaluated once per real step, frozen across RK sub-steps — Antuono/Jameson technique). Repo uses RK2 and (grep) has **no frozen-diffusion path**. Both deviations. |
| adaptive `Δt` | `cases/dambreak.py` `dambreakTimestep` returns `config.dt` unchanged for deltaSPH | **Sun Eq. (5): `Δt = min(Δt_visc, Δt_acc, Δt_acoustic)`** with `Δt_visc = 0.125 min(h²/ν)`, `Δt_acc = 0.25 min(√(h/‖a‖))`, `Δt_acoustic = CFL·min(h/c₀)`, **CFL = 1.5** for Wendland. The repo runs a *fixed* `Δt = targetDt` on the deltaSPH path — no acceleration constraint at all, which a gravity-driven impact needs. |
| kernel | `cases/dambreak.py` `kernel='Wendland4'`, `n_h=4.0` | Sun/DualSPHysics: **Wendland C2** (`Wendland2`), h/Δx = 2, support 2h = 4Δx (~50 nbrs in 2D). Confirm what `n_h=4` means here (support/Δx = 4 ⇒ h/Δx = 2, OK; or h/Δx = 4 ⇒ 2× too wide). The C4 vs C2 choice changes `W(0)/W(Δx)` (5.3 for C2 h/Δx=2) which the δ⁺ tensile term (Part 2) is calibrated against. |

**Deliverable for Part 1:** a probe `scripts/probe_deltaSPHConformance.py` that,
on a jittered lattice with an analytic field, checks each RHS term of (5) against
a from-scratch O(N²) torch reference (the pattern
`scripts/probe_deltaSPHPsiProjection.py` already established for ψ), plus a short
written table of every constant/kernel/integrator deviation with a
keep-or-change call.

---

# Part 2 — δ⁺-SPH (Sun et al. 2017): what it adds, and is it wired

**Audited — two of the four Eq. (7) constants were wrong, together making the
shift 1/8 of the paper's. Root-caused and fixed in §5.3.2; the table in §2.1
below carries the verdicts.**

δ⁺-SPH = δ-SPH (Part 1) **plus**:

1. **Particle-shifting technique (PST)**, Sun Eq. (7), applied to positions
   *outside* the RK sub-steps:
   ```
   δr_i = −CFL·Ma·(2 h_ij)² Σ_j [ 1 + R (W_ij/W(Δx_i))^n ] ∇_iW_ij · 2 m_j/(ρ_i+ρ_j)
   ```
   with **R = 0.2, n = 4** (Monaghan tensile-control values); the `2 m_j/(ρ_i+ρ_j)`
   volume (not `V_j`) is what keeps it momentum-conserving (XSPH-antisymmetric).
   Equivalent shifting-velocity form (Sun Eq. 9): `δu_i = −U_max (2h) Σ_j [1 + R(...)^n] ∇_iW_ij V_j`.
2. **Free-surface PST correction**, Sun §3.1 — remove the shift component normal
   to the surface inside the dilated surface region; switch PST off entirely for
   `λ_i < ` threshold (min eigenvalue of `L_i`).
3. **(optional) multi-resolution** `h_ij`, `φ_ij` — out of scope here (uniform Δx).
4. δ-ALE / quasi-Lagrangian variant (Sun 2019 / Antuono 2021): fold the shift
   transport into the continuity and momentum equations (`correctdrhodt`,
   `correctdvdt`) — this is what `PST_ALE_PLAN.md` / `WCSPH_SHIFTING_PLAN.md`
   already targeted.

## 2.1 Repo status

`modules/shifting/` already exists (`delta.py`, `michel.py`, `wrapper.py`), wired
through `WeaklyCompressibleSystem.finalize` → `solveShifting`, with
`ShiftingScheme` / `ShiftingProjectionScheme` enums (`surfaceNormal`,
`michel2022`, …) and the Sun 2019 `maxShiftVelocityFraction` limiter
(`moduleConfigurations/shifting.py`). `WCSPH_SHIFTING_PLAN.md` reports the
δ⁺ free-surface shift as landed and default.

**Checks:**

| Sun 2017 item | repo | check |
|---|---|---|
| Eq. (7) shift magnitude `CFL·Ma·(2h_ij)²` | `modules/shifting/delta.py` | ❌ **WRONG — was `2h²`, i.e. half of `(2h)² = 4h²`.** Fixed under `ShiftProperties.sun2017Eq7Shift`; see §5.3.2. |
| tensile term `1 + R (W_ij/W(Δx))^n`, R=0.2 n=4 | `modules/shifting/` + `moduleConfigurations/shifting.py` | present; `n = 4` ✅, but `R` was `computeDeltaShiftWarp`'s default **0.25**, not Sun's 0.2. `delta.py` now passes 0.2 under `sun2017Eq7Shift`. |
| volume weight `2 m_j/(ρ_i+ρ_j)` not `V_j` | shifting kernel | ❌ **WRONG — `wp_deltaShift.py` uses `0.5 m_j/(ρ_i+ρ_j)`, a quarter of it.** Not changed in the kernel (`modules/shifting/michel.py` shares it); compensated in `delta.py`'s scaling. §5.3.2. |
| free-surface normal removal + `λ` cutoff | `ShiftingProjectionScheme.surfaceNormal`, `surfaceLambdaThreshold` | already claimed working (`ACSPH_PLAN.md` Part 3). Re-audit against Sun §3.1 vs Michel 2022 (they differ — `PST_ALE_PLAN.md` chose Michel). |
| applied *outside* RK sub-steps | `WeaklyCompressibleSystem.finalize` | confirm it is post-step, not per-stage |
| **dam break needs PST?** | — | **No.** Marrone 2011 §3 dam-break cases are plain δ-SPH (no shifting). Sun 2017's harder cases (square patch, bluff-body wakes) need it. So a correct δ-SPH dam break must be stable with `shiftProperties.active = False`. Use that as the Part 5 acceptance gate; PST is validated separately on the square patch. |

---

# Part 3 — mDBC: English 2022 vs DualSPHysics vs the repo

## 3.1 English et al. 2022 — the method (their §3, Eqs. 8–12)

For each boundary particle `b`: a **ghost node** `g` at `dp/2` inside the fluid
from the nearest boundary layer, mirrored across the interface along the boundary
normal. Solve a first-order-consistent SPH system at `g` over **fluid neighbours
only**:
```
A_g · [ρ_g, ∂xρ_g, ∂yρ_g, ∂zρ_g]ᵀ = b_g                                         (8)
A_g  = moment matrix, rows [ W_gj , (x_j−x_g)W_gj , (y_j−y_g)W_gj , (z_j−z_g)W_gj ]
       and the ∂·W_gj rows, each · V_j                                          (9)
b_g  = [ Σ W_gj m_j , Σ ∂xW_gj m_j , Σ ∂yW_gj m_j , Σ ∂zW_gj m_j ]              (10)
```
Ill-conditioned (`< 3–4` fluid neighbours) → **Shepard fallback**
`ρ_g = Σ ρ_j W_gj V_j / Σ W_gj V_j`  (11).
Then **linear extrapolation back to the boundary particle**:
```
ρ_b = ρ_g + (r_b − r_g)·[∂xρ_g, ∂yρ_g, ∂zρ_g]                                   (12)
```
Boundary velocity ≡ 0 (so `u·n = 0` by definition; only first-order for velocity
near the wall). Pressure from the EOS on `ρ_b`. No clamp to `ρ₀` in the paper.

## 3.2 DualSPHysics — two variants in `JSphCpu_mdbc.cpp`

- **`InteractionMdbcCorrectionT2`** = English 2022 exactly: ghost node at
  `pos + boundnor`, accumulate `ρ = Σ m_j W`, `∇ρ = Σ m_j ∇W`, the 2D 3×3 / 3D
  4×4 `a_corr` moment matrix (`·volp2`); `determlimit = 1e-3`; if invertible,
  `ρ_ghost = (A⁻¹·[ρ,∇ρ])₀`, `∇ρ_ghost = −(A⁻¹·[ρ,∇ρ])_{1..d}`, then
  `ρ_final = ρ_ghost + ∇ρ_ghost·dpos` with `dpos = −boundnor`; else 0th-order
  `ρ_final = ρ/a11`; else `ρ_final = RhopZero`. `velrho.w = ρ_final`; velocity 0
  (SlipMode `SLIP_Vel0`).
- **`Mdbc2PressClone`** (`<vs_m2dbc>`, newer) = **pressure cloning**:
  `p_final = c₀²(ρ_ghost − ρ₀) + ρ₀ (g − a_motion)·n̂ (dpos·n̂)` — the ghost EOS
  pressure plus a hydrostatic Neumann correction along the wall normal. Also
  carries slip modes other than `Vel0`.

## 3.3 Repo — `modules/mdbc/`

`computeMdbcDensity` (`density2025.py`): `interpolateLiuLiu` at each `kind==2`
ghost point (`FluidToGhost`), then — per its own docstring — *"converts that to a
hydrostatic pressure correction along the ghost-offset normal, including a
gravity term, clamped to at least rest density"*, with *"one deviation … the
ghost-normal normalization … matching DualSPHysics rather than the paper."*

**This description matches DualSPHysics `Mdbc2PressClone` (m2dbc), not English
2022's density-extrapolation mDBC nor Marrone Eq. (7).** Open questions for the
audit:

| # | question |
|---|---|
| **sign** | ✅ **audited CORRECT** (`scripts/probe_mdbcExtrapolationSign.py`) — see the Status section. `interpolateLiuLiu` returns `+∇f`; ghost is `r_g = r_b − ghostOffset`; the assembled path is `ρ_g + (r_b − r_g)·∇ρ_g` = English Eq. (12). `wallPressure.py` identical. |
| 1 | Which target — English 2022 mDBC (density extrapolation Eq. 12) or DualSPHysics m2dbc (pressure clone)? The repo does English Eq. (12) for `numNeighbors > 4` (bulk) and the m2dbc Shepard-density + hydrostatic-normal term as the ill-conditioned fallback. Confirm this hybrid is intended vs. one or the other pure. |
| 2 | If m2dbc: is the `p_final = c₀²(ρ_g−ρ₀) + ρ₀(g−a)·n̂ (dpos·n̂)` formula reproduced exactly? The repo's "clamp to ≥ ρ₀" is a DualSPHysics DBC anti-attraction guard — English 2022 has no such clamp; m2dbc clamps pressure ≥ 0, equivalent. Confirm it clamps *pressure*, not density. |
| 3 | `interpolateLiuLiu`'s `b` vector: English/DualSPHysics use `Σ m_j W`, `Σ m_j ∇W` (mass-weighted); `computeMdbcDensity` passes `referenceQuantities = densities` → `Σ ρ_j W_gj V_j = Σ m_j W_gj` (same) but the **gradient rows** must be `Σ m_j ∇W_gj`, and the moment matrix `A` must be volume-weighted `·V_j`. Verify row-by-row against `a_corr2`/`a_corr3`. |
| 4 | `determlimit = 1e-3` and the exact fallback ladder (invertible → 0th-order `ρ/a11` → `ρ₀`). `interpolateLiuLiu` uses a `neighbor_threshold` and a pinv; align the thresholds and the fallback order. |
| 5 | ghost-node **placement**: `dp/2` inside from the nearest boundary layer, mirrored along the analytic boundary normal. Where does the repo get `ghostOffsets` / `ghostIndices` / the normals, and are they `dp/2`? (`geometry/` / region sampling.) A corner ghost has an ill-defined normal — English averages; DualSPHysics `boundnor` handles it upstream. |
| 6 | **velocity BC**: `computeBoundaryVelocities` — English/DualSPHysics mDBC v1 is strictly `u_b = 0` (free-/no-slip both realised through the fluid-side viscous stencil, not the boundary velocity). The repo has `zero/constant/no-slip/free-slip/extended` modes — which does the dam break use, and does "no-slip" here mean the Marrone/Sun ASM mirror or something else? |
| 7 | no-penetration shift `computeMdbcNoPenShift` (added to `dvdt`) — **not in English 2022, not in Marrone/Sun**. It is a DualSPHysics-DBC-lineage repulsion crutch (cf. `DFSPH_IMPROVEMENT_PLAN.md`'s `mdbcNoPenetrationShift` A/B). With a correct mDBC it should be unnecessary; keep it as an off-by-default A/B, not a silent always-on term (it is currently unconditional in `deltaSPH_step`). |

## 3.4 English 2022 validation suite (their §4)

| § | case | measures | repo case |
|---|---|---|---|
| 4.1 | Still water tank + triangular wedge (sharp corner), H = 0.5 m, h/dp = 2, dp = 0.02 / 0.01 | hydrostatic `p/(ρgH)` vs `z/H` down to the wall; KE decay (log scale); noise onset time | ≈ `hydrostaticColumn` + a wedge obstacle |
| 4.2 | Sloshing tank, SPHERIC benchmark (moving boundary) | wall pressure sensors vs experiment; mDBC captures negative-pressure transients DBC misses (Fig. 9) | `sloshingTank` (TC10) exists, already runs mDBC — **scoped §5.2.3** |
| 4.3 | 3D dam break impacting a cuboid (Kleefsman/MARIN 2005) | pressure on the obstacle face | `dambreak` (3D + box obstacle) |
| 4.4 | 3D fish pass with baffles | turbulent 3D structure | — (skip; no turbulence model) |

---

# Part 4 — the ψ-sign question, resolved

**Both source papers give `ψ_ij = −rho_ij − grad_ij`** in this kernel's variables
(`rho_ij = 2(ρ_j−ρ_i) x_ij/|x_ij|²`, `grad_ij = ⟨∇ρ⟩^L_i + ⟨∇ρ⟩^L_j`):

- Marrone 2011 Eq. (6): `ψ_ij = 2(ρ_j−ρ_i) r_ji/|r_ij|² − [⟨∇ρ⟩^L_i+⟨∇ρ⟩^L_j]`,
  `r_ji = −x_ij` ⇒ `= −rho_ij − grad_ij`.
- Sun 2017 Eq. (4) (projected form): `ψ_ij = (ρ_j−ρ_i) − ½(⟨∇ρ⟩^L_j+⟨∇ρ⟩^L_i)·(r_j−r_i)`,
  contracted into `D_i = 2 Σ ψ_ij (r_ji·∇W)/|r_ji|² V_j` — same operator, same
  sign, for an isotropic kernel with no `L` on the `∇W`.

Commit `790a7c7` changed the code from `psi = grad_ij − rho_ij` (gradient term
sign-flipped vs both papers) **to** `psi = −grad_ij − rho_ij` (**correct**), and
pinned the property with `tests/test_deltaSPHDiffusion.py` (linear field →
annihilation, ~1e-13). That commit is right.

The dam-break blow-up under the correct sign is therefore **not** the ψ term —
it is the Part 1 deviations (Mach ≈ 0.5, γ=7 Tait vs linear EOS?, RK2 vs
RK4-frozen, fixed vs adaptive `Δt`, kernel). The old sign "worked" only because
`psi = grad_ij − rho_ij` degenerates on a smooth field to **2× the uncorrected
Molteni–Colagrossi Laplacian** — extra numerical density diffusion that papered
over the other problems.

**Plan:** keep `790a7c7`'s correct sign. Revert the local `# PSI-REVERT` hack in
`wp_densityDelta.py`. Fix the real causes in Part 1. Re-run Part 5's dam break
with the correct operator and confirm stability comes from the scheme being
right, not from over-diffusion. `tests/test_deltaSPHDiffusion.py` stays green.
(If a genuinely stronger-diffusion operator is ever wanted for a specific case,
add it as a distinct `DensityDiffusionScheme` member — do not overload
`deltaSPH`.)

---

# Part 5 — validation cases: δ-SPH first

Marrone 2011 §3 is four dam-break configurations. Do §3.1 first (it is the
canonical one and already has a case skeleton), then reuse the machinery for the
rest.

## 5.1 Marrone 2011 §3.1 — dam break against a vertical wall  ← start here

Spec **verified against the paper's figures** (Fig. 2 geometry, Fig. 5 the P1/P2
comparison) — the earlier row here was guessed and several values were wrong:

| item | spec (Marrone 2011 §3.1) |
|---|---|
| geometry | Fig. 2: water column **2H wide × H tall** (H = 600 mm), bottom-left corner of a closed tank **L_w = 5.366 H long**; ceiling at the P3 height (1000 mm). Downstream vertical impact wall at x = L_w. |
| probes | on the downstream wall at **z = 160 / 584 / 1000 mm** (z/H = 0.267 / 0.973 / 1.667); Marrone area-integrates over a φ = 90 mm disc |
| resolution | **H/Δx = 40, 80, 320** (Fig. 5's convergence set — *not* 15/45/75) |
| sound speed | **c₀ = 40 √(gH)** (Fig. 5, M = U_max/c₀ ≈ 0.049); c₀ = 20 √(gH) (M ≈ 0.098) is the Fig. 4 weak-compressibility check. Set via the Part 1 `machTarget` path with U_max = 1.95 √(gH) (Marrone's measured front speed). |
| walls | free-slip; **inviscid** (viscosity is §3.4). The δ-SPH `dambreak` path adds no physical-viscosity wall term, so free-slip is met without a slip-mode knob. |
| scheme | plain δ-SPH, `shiftProperties.active = False`, RK4, adaptive Δt (Sun Eq. 5). Frozen diffusion still not implemented (perf only). ⚠ **This row used to read "(already the deltaSPH default)". That was false and the case has never met this line of its own spec — see §5.1.1.** |
| reference | Buchner (2002) P1/P2 traces, digitised by eye from Marrone Fig. 5 |
| acceptance | P1 arrival 2.5 < t\* < 3.0, first-impact peak ≲ 1.1 P\*, plateau P\* ∈ [0.45, 0.68] over t\* ∈ [3.2, 4.8]; P2 quiescent then peak 0.22–0.40 at 5.2 < t\* < 6.1; density band + `maxPenetrationDx` ≲ 3; stable to the full record with the correct ψ sign and no PST |

Implemented as **`scripts/probe_deltaSPHMarrone.py`** (sibling of
`probe_acsphDambreakLobovsky.py`; drives `dambreakCase` with the Buchner
geometry via `W`/`fillRatio`/`fluidWidth`/`pressureProbeHeights`). Added
`referenceVelocity` (default `None`) to `cases/dambreak.py` so U_max can be
passed for the Sun Eq. (2) c₀ pick. No separate geometry preset in the case —
the probe supplies the numbers.

### 5.1 status — root-caused an mDBC regression; not yet stable

**The dam break is not stable, and it is not a resolution or an impact problem.**
The thin fluid layer at the *front* of the collapse explodes off the **dry
downstream bed** at **t ≈ 0.54 s (t\* ≈ 2.1)** — *before* the front reaches the
end wall (P1 stays dry until t\* ≈ 2.9; `maxPenetrationDx` ≈ 0.6 and *falling*
throughout, so it is not wall penetration). vMax goes 4 → 6 → 14 → 76 over
~0.03 s and KE triples; the flung sheet particles then coast ballistically
(vMax pinned at ~55, then ~38, …). Every resolution shows the identical event
at the identical time — H/Δx = 15/25/30 cascade to a full ρ → 1e11 blow-up,
H/Δx = 40 "survives" only in that it never NaNs (the sheet is gone, the bulk
limps on). The earlier "H/Δx ≥ 40 is stable, start there" note here was wrong —
it mistook a non-NaN run for a stable one.

**Cause: this session's uncommitted mDBC change in
`modules/mdbc/density2025.py` — the English Eq. (12) MLS extrapolation
threshold dropped 9 → 4.** That path has *no* conditioning guard (English §3
and DualSPHysics both gate on a `determlimit`; `modules/liu/interp.py` only
`pinv`s `A_g`, which passes a near-singular direction straight through). A bed
boundary particle under the thin fast front sees ~5–9 fluid neighbours all in
a shallow horizontal band → the vertical moment of `A_g` is tiny → the
extrapolated `∇ρ` blows up → `ρ_proj` is wild → `P_b = c₀²(ρ_proj − ρ₀)` with
c₀ = 40√(gH) (c₀² ≈ 9400) is an explosive repulsion. This is also why Lobovský
"worked": at c₀ ≈ 14√(gH) the same ρ error is ~8× weaker and only produced the
~5-particle transient ejection the Lobovský FINDINGS noted.

**Reverted to `threshold = 9`** (reasoning in-code). **Full run, H/Δx = 40,
c₀ = 40√(gH), to t\* = 7.7** (`scripts/old/out_deltaSPHMarrone/`, 9/9 acceptance
checks, video):

- **Stable and weakly compressible the whole way**, through the plunging-wave
  cavity closure: whole-run **max ‖v‖ = 5.9** (was 56+ pre-impact pre-revert),
  **ρ ∈ [0.977, 1.021]** (5–95 pct between events [0.995, 1.007]),
  `maxPenetrationDx` = 0.8 — no wall leakage.
- **P1 (z/H = 0.267):** arrival t\* ≈ 2.78; a clean **≈ 0.43 P\*** plateau from
  t\* ≈ 3.5 to the end of the record, tracking the Buchner points including
  their gentle rise near t\* ≈ 5.5–6; no median overshoot (0.52).
- **P2 (z/H = 0.973):** near-zero until t\* ≈ 4, a single narrow hump peaking
  **≈ 0.19** (t\*≈0.1-median) at t\* ≈ 5.0, back to zero by t\* ≈ 5.5.
- Both P1 and P2 read **~20–30 % below** Buchner / Marrone's own H/Δx = 40
  (P1 ≈ 0.55, P2 ≈ 0.28) and P2 ~0.5 t\* early — consistent with a point probe
  inset one Δx vs. Marrone's φ = 90 mm disc centred **on** the wall. The
  acceptance bands are set wide enough to pass at that bias; an on-wall disc
  integral is what tightens them.

H/Δx = 15 (worst case): the violent explosion is gone (‖v‖ < 10 through
t\* ≈ 2.1); a milder disturbance still builds by t\* ≈ 2.7 — the front sheet is
~1 particle thick at that Δx.

**Principled fix (Part 3) — DONE:** a determinant / condition gate on the MLS
path (DualSPHysics `determlimit ≈ 1e-3`, fall back to 0th-order then ρ₀), now
`interpolateLiuLiu`'s own `wellConditioned` return value, with a per-kernel
`determinantThreshold` lookup (`Wendland2: 1e-3` DualSPHysics' own value,
`Wendland4: 1.8e-3` derived from it — see the Status section above and
`scripts/probe_mdbcKernelDeterminantScale.py`) after an initial attempt with
the bare `1e-3` regressed H/Δx = 40 (fixed by the lookup, then independently
by moving this case's default kernel to Wendland C2 to match Marrone/Sun/
DualSPHysics outright). H/Δx = 15 — previously the case that still "built a
milder disturbance" under the `threshold = 9` hack — is now clean to
t\* ≈ 12.9, and H/Δx = 40 confirms no regression against its prior baseline.

Probe tooling: `probe_deltaSPHMarrone.py` (Buchner geometry + P1/P2-vs-Fig.5
scoring, plus a `--kernel` override added this session for kernel A/Bs);
`referenceVelocity` and `pressureProbeSupportScale` params on
`cases/dambreak.py` (both default to the old behaviour).

**True φ = 90 mm on-wall disc-integral probe — implemented.**
`cases/dambreak.py`'s `diagnostics` gained `pressureProbeDiscRadius`: a
7-point Gauss-Legendre quadrature over the probe's vertical chord (the 2D
reduction of a disc-area integral — no out-of-plane extent to integrate over,
so `∫∫_disc P dA = ∫_{-R}^{R} P(s)·2√(R²−s²) ds`), replacing the single MLS
point inset one Δx into the fluid. Validated standalone (a linear field
recovers its exact centre value; a quadratic field matches the analytic disc
moment `R²/4` to < 0.5%). `probe_deltaSPHMarrone.py` now sets
`PROBE_DISC_RADIUS = 0.045` (Marrone's actual radius) with `probeInset = 0.0`
(flush on the wall, matching the real transducer — the disc's own vertical
averaging supplies the neighbour support the inset hack used to).

**Full H/Δx = 40 rerun, disc probe, to t\* = 7.7 — 9/9 checks pass**:

- Stability unchanged: whole-run `max ‖v‖ = 6.8`, `ρ ∈ [0.983, 1.020]`,
  `maxPenetrationDx = 0.43`.
- **P1 plateau moved from ≈ 0.43 to a mean 0.63** over t\* 3.6–7.5 (Buchner ≈
  0.55) — the ~20% *low* bias the inset point probe carried is gone; it now
  reads slightly *high* instead, well inside the `[0.38, 0.65]` band.
- **P2 peak moved from ≈ 0.19 to 0.33** at t\* ≈ 5.24 (Buchner ≈ 0.28 at
  t\* ≈ 5.5) — much closer in both level and timing than the point probe was.
- **P1 first-impact overshoot, previously invisible, now surfaced.** Wide-median
  peak now 1.94 at t\* ≈ 3.82; the old point probe never showed this at all
  (median 0.52) — the disc probe, sitting flush on the wall instead of one Δx
  into the fluid, reads the documented SPH violent-impact acoustic transient
  (the plan's own "First result" note above records the same class of event, a
  brief `vmax ≈ 28-30` spike at first contact) directly, where the inset point
  probe happened to smooth it away.

**The "1.94 is a real, benign transient, loosen the band" read above was
wrong — it was a real bug, caught by inspection of the raw trace, not by
trusting the smoothed/scored numbers.** Three rounds, each disproved by
actually looking at the raw per-step signal rather than the report's
rolling-median summary:

1. **Disc-average masking bug.** The 7-point quadrature weighted every sample
   by its fixed chord factor regardless of whether `interpolateLiuLiu`
   actually trusted that sample (`wellConditioned`); an untrusted sample
   silently contributed a hard `0` at full weight. The raw trace showed
   `P1* = 0.451 (nnbr=10)` collapsing to a hard `0.000` one sample later and
   staying pinned there for a long stretch while `nnbr` slowly decayed
   `10→4` — a square-wave artefact, not a physical signal. **Fixed**: mask
   the chord weights by `wellConditioned` and renormalise over the valid
   subset.
2. **Missing 0th-order fallback.** Fixing (1) did not fix the trace: a
   `t* ≈ 3.02–3.28` stretch stayed at a hard `0.000` even with the
   best-supported quadrature sample carrying 9–17 neighbours — every sample
   was failing the *determinant* check (a thin run-up sheet along the wall is
   near-coplanar however many neighbours it has) with no fallback tier, where
   `modules/mdbc/density2025.py` already has one. **Fixed**: `dambreak.py`'s
   probe now uses the same two-tier ladder (first-order MLS where
   `wellConditioned`, else 0th-order Shepard `b[:,0]/A_g[:,0,0]`, matching
   `density2025.py`/`wallPressure.py`) instead of a hard zero.
3. **Still not smooth after both fixes — genuinely investigated, not
   band-aided.** The disc-averaged trace *still* wasn't smooth: values up to
   `5.9` (inset 1Δx: `11.7`, i.e. *worse* away from the wall, ruling out
   "near-wall mDBC layer noise"). The stiff-EOS arithmetic (`c₀²/(gH) ≈ 1600`
   here, so an ordinary `0.5%` density fluctuation reads as `P* ≈ 8`) looked
   like a clean explanation until Marrone's own Fig. 5 (smooth, no such
   excursions, same `c₀`) ruled it out — a stiff EOS could explain *some*
   point noise, but not this. A properly-derived temporal filter (window
   physically bounded between the acoustic timescale `h/c₀ ≈ 0.0025` and the
   hydrodynamic timescale `~1`, both in `t*` units — matches the existing
   ad hoc `0.10`/`0.30` windows) *also* failed to clean it up: the elevated,
   multi-humped signal persisted for `~1.4 t*` (`t* ≈ 2.8–4.3`), far too long
   for any noise filter to remove without destroying real dynamics elsewhere
   — this was a real, extended simulated transient, not point noise.

**Root cause: RK4 without Sun 2017's frozen-diffusion companion technique.**
Part 1's audit table flagged this back at the start of this plan ("Sun 2017
§2: RK4 with frozen diffusive terms... Repo uses RK4 and has no
frozen-diffusion path") and it sat unactioned until this investigation forced
the question. Each of RK4's 4 sub-stages evaluates the density-diffusion and
artificial-viscosity terms at a different intermediate state; during a
violent, near-discontinuous impact those states diverge enough that the
diffusive term itself becomes a source of spurious feedback. A single-stage
integrator (DualSPHysics' symplectic Euler) structurally cannot have this
problem — there is no cross-stage inconsistency to freeze against in the
first place, which is why "DualSPHysics doesn't freeze diffusion either" is
not a counter-example, it is the reason the technique is RK-specific.

**Fix — `freezeDiffusionAcrossStages`, and a new named scheme.** Implemented
as a minimal, opt-in extension rather than a rewrite of `deltaSPH_step`'s
architecture (`schemes/artificialCompressible.py` freezes its own diffusion
term the same way, Antuono/Jameson, but owns a self-contained internal RK
loop; `deltaSPH_step` is called 4 separate times *by* the external
`warpSPHIntegrators.RungeKuttaB`, and is shared by 12 cases, so restructuring
it was too wide a blast radius for what started as an experiment):

- `warpSPHIntegrators.butcher.RungeKuttaB` now injects `stageIndex=0,1,2,...`
  into the scheme's step function call — **only** if that function's own
  signature declares a `stageIndex` parameter (checked via
  `inspect.signature`). Every other scheme is untouched; nothing is added to
  `kwargs` unless the callee asks for it. Full physics suite (74 tests) green
  before and after.
- `schemes/deltaSPH.py`'s `deltaSPH_step` gained `stageIndex=None`: on the
  first stage of a real step it computes gradRho/gradRhoL/`drhodt_diss`/
  `dvdt_diss` as before and, if `schemeConfig.freezeDiffusionAcrossStages`,
  caches them (`schemeConfig._frozenDiffusionCache`); later stages reuse the
  cache instead of recomputing at their own (intermediate) state.
- `WeaklyCompressibleSPHConfig.freezeDiffusionAcrossStages` (new field,
  default `False`) — every existing `deltaSPH` case is unaffected.
- **Per the user's explicit call**: rather than flipping that default for the
  generic `deltaSPH` scheme (which would silently change physics for all 12
  cases that use it), a genuinely separate, registered scheme —
  `WeaklyCompressibleSPHScheme.sun2017DeltaSPH` / `--scheme
  sun2017DeltaSPH` (`schemes/builder.py`) — reuses the identical state/step/
  update classes and just swaps in `Sun2017DeltaSPHConfig`, a
  `WeaklyCompressibleSPHConfig` subclass overriding one field default:
  `freezeDiffusionAcrossStages=True`. Selecting a specific paper's exact
  prescription is now an explicit choice, not a default anyone inherits
  silently. `dambreak.py`'s own `freezeDiffusionAcrossStages` case param
  defaults to `None` (don't touch it — let the selected scheme's own default
  stand) rather than `False`, so the two compose correctly.

**Validated — confirmed fixed**, same nx=67/H/Δx=40 case, `--scheme
sun2017DeltaSPH` (now `probe_deltaSPHMarrone.py`'s default):

| | `deltaSPH` (un-frozen) | **`sun2017DeltaSPH`** | Buchner |
|---|---|---|---|
| P1 first-impact overshoot | 1.94 | **0.60** | ~0.78 |
| P1 plateau | 0.63 | 0.46 | 0.55 |
| P2 peak | 0.33 | 0.22 | 0.28 |
| whole-run `max ‖v‖` | 6.8 | 5.4 | — |
| `ρ` range | [0.983, 1.020] | [0.983, 1.017] | — |
| raw trace, t\* 2.8–4.3 | elevated/multi-humped throughout | one brief hump (~0.2 t\* wide), then settles | — |

`p1_first_peak_max` restored `2.2 → 1.60` (the temporary loosening was
compensating for the bug above, not a genuine physical necessity) — **9/9
checks pass** under `sun2017DeltaSPH`. `scripts/old/out_deltaSPHMarrone/` now
holds both runs side by side (`REPORT.md` shows both score tables) as the
record of the fix. Not yet investigated: P1 plateau / P2 peak both sit a
little below Buchner under the frozen scheme (0.46/0.22 vs 0.55/0.28) —
comfortably in-band, but not chased further this session.

**Next:** H/Δx = 80 (`--nx 134`) for the Fig. 5 convergence pair, now under
`sun2017DeltaSPH`; the c₀ = 20√(gH) cross-check; why the frozen-diffusion
plateau/peak read a touch low.

**Cross-validate against DualSPHysics** `examples/main/01_DamBreak` on matched
geometry/resolution: same c₀, same Δt rule, DBC vs mDBC — its output is a
ready-made second reference and isolates "our δ-SPH" from "our mDBC".

### 5.1.1 This case has been running δ⁺-SPH, not δ-SPH — every result above

Found while scoping §5.3.2's shift fix. The spec table above says
`shiftProperties.active = False` and Part 2.1's acceptance gate says *"a
correct δ-SPH dam break must be stable with `shiftProperties.active = False`.
Use that as the Part 5 acceptance gate; PST is validated separately on the
square patch."* **The case has never done that.** `ShiftProperties.active`
defaults to **`True`** (`buildDefaultShiftProperties`), `cases/dambreak.py`'s
only `active = False` sits inside `_configureArtificialCompressibleExtra` — the
ACSPH branch, never reached on the δ-SPH path — so every run recorded in §5.1,
including the 9/9 `sun2017DeltaSPH` result, was **δ⁺-SPH with the PST on**.
Marrone 2011 §3 is plain δ-SPH.

Verified directly rather than by reading the source: build the case exactly as
`probe_deltaSPHMarrone.py` invokes it and read `shiftProperties` back off the
resolved config — `active=True, scheme=deltaSPH, projection=surfaceNormal`
under both `--scheme deltaSPH` and `--scheme sun2017DeltaSPH`.

**The grep that produced the "(already the deltaSPH default)" claim is the
trap.** `grep shiftProperties.active cases/*.py` shows `dambreak`, `impact` and
`rotatingSquarePatch` all setting it `False`, and all three lines are inside
ACSPH-only branches. A field that defaults to `True` cannot be audited by
looking for the places that mention it — the cases that never mention it are
the ones that have it on. `scripts/probe_deltaPlusShiftBlastRadius.py` exists
because of this: it builds every registered case's context the way `runner.run`
does, runs its `configureScheme`, and reads the resolved value. Answer: **11 of
35 cases** run `ShiftingScheme.deltaSPH` with the shift active — `dambreak`,
`drivenSquare`, `droplet`, `impact`, `kolmogorov`, `ldc`, `movingObstacle`,
`openFlow`, `randomFlow`, `squarePatch`, `tgv-wc`.

Consequences, in order of how much they matter:

1. **§5.1's numbers are δ⁺-SPH numbers.** They are not wrong as measurements,
   but they do not test what the section says they test, and the plan's own
   PST-free acceptance gate has never been exercised. The P1 plateau reading
   0.46 against Buchner's 0.55 is now a candidate for *this* rather than for
   anything about the frozen diffusion.
2. **§5.3.2's fix changed this case**, since `probe_deltaSPHMarrone.py` runs
   `--scheme sun2017DeltaSPH`, which now selects Eq. (7) — an 8× shift on a
   violent free-surface impact. The §5.1 re-run is required, not optional.
3. `rotatingSquarePatch` is in the same position, and Part 2.1 nominates it as
   the case where *"PST is validated separately"* — so the PST discriminator
   has also never been run against its own no-PST control.

`scripts/probe_deltaPlusShiftSweep.py` runs all 11 under three legs (`off` /
`eighth` / `eq7`) to bound the damage before anything else is decided.

### 5.1.2 Four-leg re-run + the H/Δx = 322 convergence — the P1 deficit was resolution

All four legs, one code state, on `dambreak` with the `shifting` knob from
§5.1.1 (`scripts/old/out_deltaSPHMarrone_pst/`, H/Δx = 40, to t\* = 7.7):

| leg | P1 plateau | P1 1st peak | P2 peak | checks |
|---|---|---|---|---|
| δ⁺ ⅛·Eq.(7), frozen — *the recorded 9/9* | 0.46 | 0.60 | 0.22 | 9/9 |
| δ⁺ Eq.(7), frozen — *current `sun2017DeltaSPH` default* | 0.396 | 0.49 | 0.126 | 8/9 |
| δ-SPH, no PST, frozen — *Marrone §3's actual spec* | 0.405 | 0.46 | 0.079 | 8/9 |
| δ⁺ ⅛·Eq.(7), un-frozen | 0.651 | 1.94 | 0.366 | 6/9 |
| Buchner 2002 | 0.55 | ~0.78 | 0.28 | — |

Two results here run *against* §5.3.1–5.3.3's direction:

1. **Eq. (7) regresses this case**, 9/9 → 8/9, confirmed against a freshly-run
   frozen+⅛ baseline (not the plan's recorded number). It is right on the TGV
   and the droplet and wrong here.
2. **The §5.1.1 spec violation is not the explanation for the gap.** Running
   Marrone §3's actual no-PST configuration gives the *worst* P2 of any leg,
   0.079 vs Buchner 0.28. The candidate raised in §5.1.1 point 1 is closed:
   not it.

At H/Δx = 40 the P1 plateau sat at ~0.40–0.46 in every frozen leg regardless
of PST — "invariant to everything varied", which pointed at the wall or the
probe. **It was under-resolution.** Re-run at **H/Δx = 322** (nx = 536, 251k
particles, Marrone's own finest Fig. 5 resolution; `scripts/
out_deltaSPHMarrone_hires/`, ~4.8 h/leg):

| H/Δx = 322 | δ⁺ Eq.(7), frozen | δ⁺ ⅛, un-frozen | Buchner |
|---|---|---|---|
| P1 plateau (t\* 3.6–7.5) | **0.556** | 0.547 | 0.55 |
| P1 first-impact peak | **0.72** | 0.80 | ~0.78 |
| P2 run-up peak | 0.126 @ t\*≈4.5 | 0.141 @ t\*≈5.9 | 0.28 @ t\*≈5.5 |
| wall penetration | 5.08 Δx ❌ | 5.36 Δx ❌ | (≤ 3 Δx gate) |
| checks | 7/9 | 6/9 | — |

- **P1 is converged.** Plateau 0.556 vs 0.55, first peak 0.72 vs ~0.78 — both
  on Buchner, up from ~0.40 at H/Δx = 40. The scheme reproduces the lower-probe
  pressure; it just needs the resolution the paper used.
- **P2 does not converge with resolution** and is now the isolated
  discrepancy: still ~2× low (0.13 vs 0.28) **and** the run-up pulse arrives
  ~1 t\* unit early (peak at t\* ≈ 4.2–4.3 vs Buchner's 5.5). This is a *phase*
  error in the plunging-wave / run-up jet reaching the upper probe, not only an
  amplitude one — a different failure than "under-diffused" or "wrong PST".
- **New at H/Δx = 322: wall penetration fails**, 5 Δx vs the ≤ 3 Δx gate
  (0.43 Δx at H/Δx = 40). 130 of 251k particles leak, starting at the first
  wall impact t\* ≈ 2.5 and growing through the run. Present in *both* legs, so
  it is the mDBC wall at fine resolution, not the PST — a Part 3 item, and it
  scaled the wrong way with Δx.
- **P1's *raw* trace is acoustically dominated — a probing-methodology gap, not
  a scheme error.** The rolling median tracks Buchner (0.556 vs 0.55), but the
  un-averaged P1 signal swings through a range several times the hydrodynamic
  pressure it sits on, growing from t\* ≈ 4 to the end (worse at H/Δx = 322
  than at 40). This is genuine weak-compressibility ringing: P1 sits deep under
  the settled pool (z/H = 0.267, ~0.44 H of water above), where acoustic waves
  reflecting off the end wall and bed are trapped and resonate at
  c₀ = 40√(gH); P2, near the free surface, has no trap and its raw trace stays
  clean. **Marrone's Fig. 5 shows none of this because his probe is a
  φ = 90 mm disc *area integral*** (≈ 1.5 Δx across at H/Δx = 40), which is a
  spatial low-pass that annihilates the short-wavelength acoustic mode; a
  point/small-disc probe at this c₀ cannot. So the P1 raw swings are not a
  defect to fix in the scheme — they are the cost of reading a weakly
  compressible field at a point, and the honest comparison against Marrone is
  the disc-integrated median, which passes. (This does **not** apply to the
  separate P2 finding above: P2's median, not its raw trace, is what is 2× low
  and early, and P2 shows no acoustic ringing.)

**Video export bug found and fixed while doing this** (`runner/media.py`):
frames are `frame_{step:05d}.png`, a 5-wide pad that overflows past 100k steps,
and ffmpeg's `-pattern_type glob` then orders `frame_100100.png` before
`frame_99840.png` — so at H/Δx = 322 (156k steps) the entire back half of every
video played out of order. `encodeFrames` now sorts frames by their embedded
step number and feeds ffmpeg a gapless `f%06d.png` symlink sequence, so
filename width no longer matters.

**Not yet run:** δ-SPH no-PST at H/Δx = 322 (deferred — the two legs above
took 9.6 h between them at ~1.85× the benchmarked step cost, a perf regression
to chase separately).

**2026-09-10 update — supersedes the "wall penetration failed at H/Δx = 322"
line above.** That 5 Δx leak, and a much worse H/Δx ≈ 72–160 catastrophe, are
now understood: `1e145a1` (body-node ghost placement, committed *after* the
H/Δx = 322 runs here) regressed the wall by zeroing deep flat-wall mDBC layers;
an uncommitted `ghostParticles.py` fix restores H/Δx = 40 (δ⁺+PST un-frozen is
now the **best Buchner match**, P1\*/P2\* 0.51 / 0.25). Separate & still open:
H/Δx ≈ 72–160 blows up regardless of scheme (not a sampling artefact — the IC
is identical across 40/72/322); `no-PST + un-frozen` cannot do the first impact
at any Δt. **Full account: Open/next item 4.** The H/Δx = 322 legs should be
*re-run* on the fixed code, not treated as current.

## 5.2 Then

| next | case | notes |
|---|---|---|
| Marrone §3.2 | dam break vs a tall thin column | Yeh & Petroff force + LDV velocity; 3D effects — 2D first |
| Marrone §3.3 | dam break vs a rectangular step | 3D; obstacle preset already in `buildPresetObstacles` |
| Marrone §3.4 | dam break vs a **sharp-edged obstacle**, rounded tank corner (Fig. 19) | **§5.2.2 — geometry + stable baseline done**; boundary sampling / stability / convergence. §3.4.2 adds no-slip `ν` (Re = 1000 / 10000) |
| English 2022 §4.1 | still water + wedge | mDBC discriminator — hydrostatic profile to the wall, KE decay. **Scoped in §5.2.1 — next up.** |
| English 2022 §4.2 | sloshing tank | `sloshingTank` (TC10) — **§5.2.3 scoping done**: mDBC is already the shipped wall treatment; the work is grading Sensor 1 vs the experiment (incl. the negative-pressure transients) at English's dp, + an optional DBC A/B |

### 5.2.1 English 2022 §4.1 — still water on a bed with a sharp-cornered wedge — SCOPING

**Why this case, now.** It is the cleanest possible mDBC discriminator: still
water, so the exact answer is the analytic hydrostatic profile `p = ρg(H−z)`
and *every* departure is the wall closure alone — no scheme/wall confound like
the dam break has. It also directly follows up two open items: Part 3
question 5 (the corner ghost has an ill-defined normal), and §5.1.2's
finding that wall penetration failed at H/Δx = 322 in *both* legs, which
pointed at ghost generation.

**Paper spec** (English et al. 2022 §4.1, *Comp. Part. Mech.* 9:911–925):

| item | value |
|---|---|
| tank | 2D, 2.4 m × 1.2 m, closed |
| wedge | trigonal, bottom-centre, **height 0.24 m**, sharp apex pointing up into the fluid |
| water | initial height **H = 0.5 m** |
| kernel / resolution | `h/dp = 2`; **dp = 0.02 m and 0.01 m** (H/dp = 25 and 50) |
| duration | 4 s physical (noise checks at 20 s and 200 s) |
| scheme | δ-SPH + mDBC; **no PST** (still water) — so `--scheme deltaSPH --shifting off`, and this is also a clean test of the §5.1.1-corrected default |
| EOS / c₀ | weakly compressible, `c₀ = 10√(gH)`-class (Mach ≈ 0.03 for still water); the Sun Eq. (2) `machTarget` path |
| measures | (a) per-particle `p/(ρgH)` vs `z/H` at t = 4 s — Fig. 4, must stay on the hydrostatic line **down to the solid surface, including at the wedge corners**; (b) `Σ ½m‖v‖²` vs t, log scale — Fig. 5, mDBC ≪ DBC and must not grow; (c) noise onset time (paper: negligible to 20 s, small noise ~200 s) |
| acceptance (proposed) | profile RMSE off `p = ρg(H−z)` ≤ 3 % of `ρgH` over the bulk **and** ≤ 8 % within 2 dp of the wedge faces/apex; KE per unit mass stays below ~1e-4·gH for the 4 s record and is not trending up; no particle penetrates the wedge or tank wall by > 1 dp |

**Case construction — mostly free.** `caseUtils/weaklyCompressible.buildPresetObstacles`
already has an `equilateralBottom` preset: an equilateral triangle sitting on
the tank floor centre, which *is* the English wedge up to the apex angle
(`maxExtent` / `aspectRatio` set the height and half-angle). `dambreak`
already wires `obstacleActive` / `obstacleType` through to a boundary region.
So the geometry is: `dambreak` with `disableGravity = False`,
`fluidWidth = 1.0` (fill the whole box width → still pool, no column to
collapse), `fillRatio = 0.5/1.2 ≈ 0.417`, `obstacleActive = True`,
`obstacleType = 'equilateralBottom'` with `maxExtent`/`aspectRatio` tuned to
0.24 m height. A thin `scripts/probe_englishWedge.py` (sibling of
`probe_deltaSPHMarrone.py`) supplies the numbers and scores (a)–(c). **Open
choice:** a dedicated `stillWaterWedge` case vs. a `dambreak` preset + probe.
Lean preset+probe — the dam-break hooks already do everything, and a second
still-water case would duplicate `hydrostaticColumn`'s diagnostics machinery
(note `hydrostaticColumn` itself is a *DFSPH* case and a known-failing
baseline — not a usable base here).

**The actual work — ghost placement in `rigidBody/ghostParticles.py`
(`addBoundaryGhostParticles`). Two coupled defects, and the second is the more
fundamental.**

Today: `sdfValues, sdfNormals = region.sdf(boundaryPos)`;
`offset = (s − min(|s|, 1.5h))·∇φ`; `ghost = boundaryPos − offset`. For
`|s| < 1.5h` this is `ghost = boundaryPos − 2s·∇φ` — the boundary particle
**reflected across the surface by twice its own signed distance `s`**.

**(A) The `dp/2` assumption does not hold for a general obstacle — the more
important fix.** `regions/sample.py` builds the boundary layer by laying a
*regular lattice* over the domain and keeping the points with `sdf < 0`. Only
when the obstacle face is axis-aligned *and* grid-registered do those points
sit at `s = −dp/2, −3dp/2, …`; for a tilted, curved or offset face each
boundary particle's `s` is wherever the lattice happened to cut the geometry,
anywhere in `(−dp, 0)`. The `ghost = boundaryPos − 2s·∇φ` reflection then puts
the ghost node at depth `≈ |s|` *past the true surface* — so its penetration
into the fluid, and hence the fluid support the Liu–Liu interpolation sees,
**varies particle-to-particle from ~0 (ghost buried in the boundary/fluid
transition, interpolation is garbage) to ~dp**. That is why nothing caught
this until now: every case scored so far (`dambreak` tank, `hydrostaticColumn`
box, the sloshing tank) has grid-aligned walls where `s ≡ −dp/2` and the
reflection is uniform and correct. The wedge — and every obstacle/FSI case
after it — is not aligned.

Fix: decouple the ghost's fluid-side depth from the particle's own `s`.
1. **project to the true surface**: `foot = boundaryPos − s·∇φ/|∇φ|²`
   (a Newton step; iterate 1–2× because `∇φ` at the particle ≠ `∇φ` at the
   foot on a curved face). The `/|∇φ|²` matters for any obstacle built with
   `op_smooth_union` / `op_smooth_*` where the composite SDF is not a true
   distance and `|∇φ| ≠ 1`; guard `|∇φ| → 0`.
2. **place the ghost at a fixed fluid-side depth**: `ghost = foot + d_g·n_foot`,
   `n_foot = ∇φ(foot)` re-normalised, `d_g` a new `ghostFluidDepth` param
   defaulting to English's `dp/2` and sweepable for interpolation
   conditioning the way `determinantThreshold` was in the Marrone work.
3. **`ghostOffset = boundaryPos − ghost`** — an arbitrary vector now, which is
   fine: `density2025.py` / `wallPressure.py` apply English Eq. (12)
   `ρ_b = ρ_g + (r_b − r_g)·∇ρ_g` over the *actual* offset, so no downstream
   change.
   **Safety property:** on a grid-aligned flat wall (`s ≡ −dp/2`, `|∇φ| ≡ 1`)
   this reproduces the current ghost position exactly, so no already-scored
   case moves — only misaligned geometry changes.

**(B) Corner-aware placement — rides on (A).** At a convex apex the SDF is a
`min`/`max` of two half-plane fields and `∇φ` is discontinuous across the
medial axis, so step (A)'s projection lands on a face or oscillates across the
seam. English's remedy (their Fig. 1c/d): mirror the corner ghost **through
the corner point**. So: **detect** corner particles (∇φ at `boundaryPos`
disagrees with ∇φ at its `foot`; or `|∇φ| < 1 − ε` from the combine; or
distance to the nearest obstacle-polygon vertex < ~1 dp — `sdPolygon` already
carries the vertex list), and for those set `ghost = vertex + d_g·bisector`
(convex apex) or the re-entrant mirror for the concave wedge-base / floor
corners — check both signs.

`interpolateLiuLiu`'s `wellConditioned` gate (from the Marrone work) already
falls back to Shepard when a ghost still lands with poor fluid support, so the
failure mode stays graceful while (A) and (B) are iterated.

**Sequencing:**
- (a) probe on the *flat-wall* still-water tank (no wedge) — confirms mDBC
  holds still water at all, and is the baseline for the "grid-aligned cases
  don't move" safety property of fix (A);
- (b) add the wedge with the **current** ghost placement and record the
  failure — expect both the apex (defect B) *and* a general noise band along
  the two sloped wedge faces from defect (A), since those faces are not
  grid-aligned;
- (c) implement fix (A) — surface-projected, fixed-depth ghost — and re-score;
  the sloped-face noise should clear even before the corner rule;
- (d) implement fix (B) — corner rule — and re-score the apex;
- (e) a tilted flat plate in still water is the cheapest isolated test of (A)
  alone (no corner), worth adding as a probe variant.

`rotatingSquarePatch`'s no-PST control (§5.1.1 point 3) can run in parallel —
it needs no new code.

### 5.2.1 status — steps (a)–(d) done; fix (A) landed, wedge 9/9 at dp = 0.01

**(a) flat still-water tank — mDBC holds it, once the fluid starts on its own
pressure field.** `scripts/probe_englishWedge.py` (new): drives `dambreak` as a
still pool (`fluidWidth = 1.0`, gravity on, `shifting = off`), scores English
Fig. 4 (per-particle `p/(ρgH)` vs `z/H` against the hydrostatic line, split
bulk / near-wall / — with the wedge — face-band / apex / base-corner) and
Fig. 5 (fluid KE history, graded on the settled tail not the release
transient). First run failed at the *finer* resolution: dp = 0.02 clean by
t ≈ 1 s (bulk RMSE 1.4 %), dp = 0.01 still ringing at t = 4 s (bulk RMSE
**21 %**, a 23 %-too-steep hydrostatic gradient). Cause: uniform-`ρ₀`
initialisation — the pool has to compress into its hydrostatic pressure field,
and the δ-SPH diffusion that damps that transient scales as `δ h c₀`, so it
damps ~2× slower per second at half the `dx`. **`dambreak` gained
`hydrostaticInit`** (default off; stamps `ρ(z) = ρ₀(1 + g max(H−z,0)/c₀²)`,
same pattern as `tgv-wc`'s `initialPressure`). With it, dp = 0.01 flat →
**1.1 %**, both resolutions 6/6.

**(b) wedge with the legacy ghost placement — the predicted corner failure,
localised.** Global near-wall RMSE passed (the ~20 bad particles are diluted
across ~4000), so `_score` gained wedge-localised checks. Those show:
base-corner max |resid| **0.135** (dp = 0.02) / 0.078 (dp = 0.01), wedge-face
RMSE 0.074 / 0.025 — against a flat-wall baseline of ~0.03. **The failure is at
the two *concave base corners* (sloped face meets bed), not the apex** — the
apex reads ~0.02.

Root-caused by inspecting the init `ghostOffsets` directly: `dambreak` runs
`band = 5` boundary layers, and the legacy placement mirrors each particle
across the boundary SDF by `2·(its own signed distance)` along `∇(sdf)`. So a
4th/5th-layer particle 0.1 below the bed beside the wedge gets a ghost mirrored
**straight up by 10·dp** into open fluid, and English Eq. (12)
(`ρ_b = ρ_g + (r_b−r_g)·∇ρ_g`) then extrapolates over that 10·dp lever arm.
Flat wall: `∇ρ` linear, exact. Re-entrant corner: `∇ρ` bends, and
`error ~ arm² · ∇²ρ` → ~13 % of `ρgH`. `interpolateLiuLiu`'s determinant gate
does **not** catch these — 59 of 66 corner ghosts are `wellConditioned`
(30+ fluid neighbours); they are well-supported, just far.

**(c) fix (A) — `_fluidDirectedGhostOffsets` rewritten, corner-gated
(`17290ff`).** Direction and interface distance from the nearby *fluid*
particles (not `∇(sdf)`); ghost mirrored across the fluid-facing interface with
the offset **capped at `2·dp`** so the extrapolation stays in clean near-wall
fluid; applied **only where the fluid direction disagrees with the legacy SDF
normal by > ~10°** — i.e. at corners. On a flat grid-aligned wall the two are
collinear, the gate stays shut, and every mid-wall boundary particle keeps its
legacy offset byte-for-byte (of 1880, the 196 that move are all at tank corners
/ the waterline). Two earlier iterations failed and are recorded in the commit:
capping *every* wetted ghost pulled flat-wall inner-layer ghosts back inside
the solid (dp = 0.02 flat near-wall 0.03 → 0.11); the misalignment gate fixed
that.

Scored on **RMSE** over each band (≤ 0.03 for the apex / base corners = the
flat-wall near-wall level; ≤ 0.05 for the face band) — each corner band is only
~16 particles at dp = 0.02, so the worst single particle carries ~0.02
run-to-run scatter and is the wrong statistic (two legacy runs of the same
config gave corner max 0.108 and 0.135):

| still-water wedge, t = 4 s | legacy | **fix (A)** | gate |
|---|---|---|---|
| dp = 0.01 base-corner RMSE | 0.033 | **0.015** | ≤ 0.03 |
| dp = 0.01 wedge-face RMSE | 0.025 | **0.010** | ≤ 0.05 |
| dp = 0.01 checks | 8/9 | **9/9** | |
| dp = 0.02 base-corner RMSE | 0.058 | **0.044** | ≤ 0.03 ❌ |
| dp = 0.02 wedge-face RMSE | 0.049 | **0.037** | ≤ 0.05 |
| dp = 0.02 checks | 8/9 | **8/9** | |
| dp = 0.02 flat near-wall RMSE | 0.020 | 0.025 | ≤ 0.08 |

**At English's validation resolution (dp = 0.01, H/dx = 50) the wedge is 9/9**
— corner RMSE 2.2× better, face RMSE 2.5× better. The coarse `dp = 0.02` case
stays 8/9: the acute base corner (16 particles at H/dx = 25) reads corner RMSE
0.044 against the 0.03 flat-wall target — ~1.5× better than legacy, still a
real under-resolution residual, not eliminated.

**(d) fix (B) — the corner-point mirror (English Fig. 1c/d) — attempted and
reverted.** The cheap realisation (mirror through the *nearest* fluid
particle's direction rather than the fluid-weighted mean, since the mean is
face-count biased at an acute corner) moved the dp = 0.02 corner RMSE by
< 0.001 — a no-op. The faithful version reflects the corner ghost through the
obstacle-polygon *vertex*, which needs the obstacle geometry plumbed into
`addBoundaryGhostParticles` (today it only gets `regions` + `particleState`).
**Deferred** — fix (A) already clears the wedge at the resolution English
validates at, and the residual is coarse-`dp` under-resolution that a better
ghost rule will not fix. (e) the tilted-plate isolated-(A) test still stands.

### 5.2.2 Marrone 2011 §3.4 / Fig. 19 — dam break vs a sharp-edged obstacle with a rounded tank corner

**Why this case, now.** The user's pick after §5.2.1. Marrone §3.4 merges every
hard part of the boundary handling into one geometry: a **convex 45° edge**
that ejects a violent jet, **two re-entrant (concave) corners** at the obstacle
toe and the roof/back-face junction, and a genuinely **curved concave wall**
(the fillet, radius H) rounding the tank's downstream bottom corner. Marrone
uses it to show the fixed-ghost technique copes with "curvilinear parts and
convex/concave angles" (Appendix A). Goal here: **a method that runs it
stably, without significant boundary penetration, and converges** — the
pressure traces vs Colicchio's Level-Set (their Fig. 24) and Wagner's
P1 ≈ 36.7 ρgH are a later refinement.

**Geometry** (Fig. 19, all lengths in the obstacle height H): tank 10 H × 8 H,
closed, free-slip, inviscid (§3.4.1). Water column 3 H × 2.4 H in the upstream
bottom corner. Obstacle on the floor: 45° edge apex at x = 5 H (height H), toe
at 6 H, vertical back face at 7 H. Concave fillet radius H from x = 9 H (floor)
to x = 10 H (height H, at the downstream wall). c₀ = c₀Ratio·√(gH); Marrone's
Fig. 21 pair is 28.3 (M ≈ 0.085) and 56.6. Space resolutions H/dx = 33.5 / 67 /
134; the probe uses **H/dx = 32 / 64 / 128** (`--nx 256/512/1024`, H/dx = nx/8)
so H is an integer number of spacings and the reference points land on the
lattice.

**Construction — a preset + probe, like §5.2.1.** `caseUtils/weaklyCompressible.py`
gained a `marroneSharpEdge` obstacle: `_marroneSharpEdgeSDF` returns the
obstacle polygon (a `sdTriangle` wedge ∪ `sdBox` block) **unioned with** the
fillet lens (`max(cornerBox, −disc)`), built straight from the domain `L`/`W`
(so `maxExtent`/`offsetX`/`aoa` don't apply), fed through `dambreak`'s normal
single-obstacle SDF path. `dambreak.diagnostics` gained `nObstaclePen` /
`maxObstaclePenDx` from that SDF — the interior-AABB penetration watch cannot
see a solid island in the flow or a fillet inside the tank AABB.
`scripts/probe_deltaSPHMarrone34.py` (`--initdump` renders boundary + ghost
sampling at the four hard spots with no stepping; `--report` builds the
stability/penetration/convergence plots).

**Boundary sampling — the ghost placement is init-only, and the obstacle is dry
at t = 0.** `--initdump` at H/dx = 32, straight after `addBoundaryGhostParticles`:
**784 of 7608 mDBC ghost particles are placed *inside* the obstacle/fillet
solid** (min SDF ≈ −6 Δx), ghost-offset lever arms up to 21.75 Δx. Cause:
`addBoundaryGhostParticles` runs **once** at init, and §5.2.1 fix (A)'s
`_fluidDirectedGhostOffsets` only re-places a boundary particle's ghost when
fluid is *already within 6 Δx*. In the English wedge the wedge is submerged from
t = 0 so the correction always applies; on a **dam break** the obstacle is dry
at init, so every obstacle-surface ghost keeps the legacy `2·s·∇(sdf)`
reflection — wrong at the 45° edge, both re-entrant corners, and worst at the
concave fillet (there `∇` of `max(box, −disc)` points into the wall/floor, not
the fluid). Nothing re-places them when the jet arrives.

**Baseline result — stable anyway (H/dx = 32, δ-SPH + mDBC, no PST,
c₀ = 28.3√(gH), to t\* = 5, 7114 steps, ~15 min):**

| check | result |
|---|---|
| divergence | **none**, ran the full t\* = 5 |
| obstacle / fillet penetration | **0.00 Δx** into the solid, whole run |
| tank-wall penetration | 0.6 Δx past the AABB (sub-Δx), flat after t\* ≈ 3.5 |
| ρ (5-pct proxy / max), t\* > 1 | [0.996, **1.144**] — spikes at the two impact events |
| max ‖v‖ | 26.2 = **4.3 U_max** — jet-tip / thin-sheet spike |
| KE | smooth rise to a t\* ≈ 1.9 peak, then monotone decay — no runaway |
| flow | reproduces Fig. 20 / 22 — sharp-edge jet ejection at t\* ≈ 2, fragmenting sheet arcing back over the reservoir, roof re-impact at t\* ≈ 3.9 |

So **the user's stated goal is already met at H/dx = 32**: it runs stably with
no meaningful boundary penetration, despite the 784 mis-placed init ghosts —
`interpolateLiuLiu`'s `wellConditioned` → Shepard fallback and the always-on
`computeMdbcNoPenShift` are the graceful-failure path §5.2.1 relied on, and here
they hold the wall. What's left is **quality**: the ρ_max ≈ 1.14 / v_max ≈ 4.3
U_max excursion at the sharp-edge ejection and the roof re-impact — localised
jet-tip / thin-sheet under-resolution (Marrone says this case is not converged
even at H/dx = 234), not a bulk instability.

**Convergence — H/dx = 32 vs 64 (init-only ghosts, c₀ = 28.3√(gH), to t\* = 5):**

| | H/dx = 32 | H/dx = 64 |
|---|---|---|
| diverged | no | no |
| KE history | — | **overlies H/dx = 32 through the collapse, tracks in decay** |
| `densityP05` (bulk-low), t\* > 1 | 0.996 | 0.999 — no bulk drift with Δx |
| tank-wall penetration | 0.6 Δx ⚠ | 1.4 Δx ⚠ |
| obstacle / fillet penetration | 0.00 Δx | 0.06 Δx |
| ρ pointwise max (1 jet-tip particle) | 1.14 | **1.27** |
| max ‖v‖ | 4.3 U_max | **7.1 U_max** |

⚠ **The tank-wall row predates the ghost-placement fix.** With the old
composed-SDF-gradient ghosts, plain δ-SPH nx = 256 leaked **~30 Δx** by t\* = 5
(15–20 particles sinking through the bed at the obstacle toe — a re-entrant
corner the `∇(sdf)` mirror mishandled; not a code regression). Fixed by
`_bodyNodeGhostOffsets` (§5.2.3): now ≤ 1 dx. Bulk convergence unaffected.

The **bulk converges** — KE curves overlie, `densityP05` is flat. The **pointwise** jet-tip extremes (`maxDensity`, `maxVelocity`)
*grow* with resolution: the fragmenting sharp-edge jet resolves thinner and
faster with less numerical smoothing, exactly the non-convergence Marrone notes
at H/dx = 234. `dambreak.diagnostics` gained `densityP99` and the probe's
`_score` now bands the bulk (`[densityP05, densityP99]`) and demotes the
pointwise max / v_max to report-only — the stability gate was reading a jet tip.

**Rounded corner in isolation — `marroneRoundedCorner` (`--cornerOnly`), the
fillet without the obstacle, H/dx = 32, to t\* = 7: 6/6.** The dam-break surge
runs the full 10 H with nothing upstream to bleed it (KE peaks ~1.5× the
sharp-edge case) and impacts the concave fillet directly. Result: **no
divergence, ZERO fillet penetration the whole run**, ρ bulk [0.999, 1.05],
max ‖v‖ = 1.9 U_max, tank penetration 0.9 Δx, KE clean. The fillet turns the
horizontal surge into a smooth vertical wall run-up jet with no separation and
no penetration — despite 218/6088 init ghosts placed inside the lens solid
(`--initdump --cornerOnly`; the legacy SDF reflection scatters them where the
box-face and disc-arc normals conflict). Same graceful-fallback path as the
full case. **The smoothed corner works.**

**Ghost-offset refresh (`mdbcGhostRefreshEvery`) — tried, then REMOVED (`76af473`).**
A blanket per-step recompute of every ghost destabilises (**109 Δx wall
penetration** at `= 5`); a *targeted* `postStep` (re-place only a ghost whose
node currently sits inside the solid, using the live fluid direction) was a
small quality gain (ρ pointwise max 1.14 → 1.11) and passed the nx = 256
penetration gate at 0.68 dx. **But it violates the design rule** (user,
2026-09-09): mDBC ghost positions are a pure function of the boundary geometry,
fixed once at init; a static boundary that discretises differently as the flow
arrives is a bug. The whole path is gone — `dambreak.postStep`, the param, the
probe's `--ghostRefresh`, and `_fluidDirectedGhostOffsets`. The concave-region
ghost placement must be fixed on the **sampling** side instead (§5.2.3).

**δ⁺-SPH + particle shifting — the dataset config.** `--scheme sun2017DeltaSPH
--shifting default` (PST on, Sun 2017 Eq. (7) shift magnitude), H/dx = 32:
sharp-edge **6/6 to t\* = 5**, rounded-corner **6/6 to t\* = 7**. The shift
regularises the distribution so the **bulk** ρ excursion drops sharply vs plain
δ-SPH — P99 [0.999, **1.022**] (sharp-edge) / [0.999, **1.004**]
(rounded-corner), against ~1.14 for plain δ-SPH; only isolated jet-tip
particles still reach ρ ≈ 1.24. Penetration stays sub-Δx. The reusable
configs are `examples/sweeps/marrone34_sharp_edge.yaml` /
`marrone34_rounded_corner.yaml` (`warpsph-run dambreak --config …`) — a dataset
source for violent free-surface / solid-boundary interaction; see
`datagen/README.md`.

**Item 7 (2026-09-09) — surface probes (b), viscous wiring (c), a real
bulk-max convergence number (a, partial). All in `scripts/old/out_deltaSPHMarrone34_item7/`.**

*(b) Surface pressure probes — DONE.* `dambreak.diagnostics` gained a general
surface-point probe: `surfacePressureProbes` = `[x, y, nx, ny]` rows (a
centred-domain point + the into-fluid unit normal); the query point is pushed
`surfacePressureProbeInset` = 0.5 dx off the wall along the normal and the fluid
pressure is read there by the **same** `_mlsPressure` helper the axis-aligned
wall probe now shares (first-order MLS fit → 0th-order Shepard → 0). Emits
`pSurf{k}` / `pSurf{k}Star` / `pSurf{k}Nnbr`. `probe_deltaSPHMarrone34.py` sets
Marrone Fig. 19's P1–P9 (P1–P3 down the 45° edge from the apex, P4–P6 roof,
P7–P9 fillet arc) with analytic normals; `--traceFigure` (also emitted by
`--report`) writes `surface_pressure_traces.png` — a 3×3 of P/(ρgH) vs t\*,
~0.15 t\* running mean over faint raw, P1/P5/P7 (Fig. 24's three) boxed, Wagner
36.7 on the edge panels. **Probe positions are read approximately from the
figure sketch — the paper gives no coordinates.**
- **Edge (P1–P3), first impact t\* ≈ 1.9–2.1.** Plain δ-SPH H/dx = 32: the
  mid-edge probe **P2 peaks 36.4 ρgH — on Wagner's 36.7 ρgH**; the apex probe
  P1 reads 11.6, the near-toe P3 overshoots (84, a point probe on a thin
  under-resolved sheet). This reproduces Marrone's own "SPH close to Wagner".
  δ⁺-SPH + PST damps the point peaks *below* Wagner (P2 20.6) — the PST
  regularises the jet. H/dx = 64: P2 65.8 (the mean is resolution-robust, the
  raw peak sharpens).
- **Roof (P4–P6), δ⁺ t\* = 7.4:** wet from t\* ≈ 5.5 (the roof re-impact —
  earlier than the plan's t\* ≈ 3.9, which was when the sheet *starts* arcing).
  P4 ≈ 18.7, **P5 ≈ 12.7**, P6 ≈ 1.6 ρgH.
- **Fillet (P7–P9):** barely wet even at t\* = 7.4 — P7 first wets at t\* ≈ 7.0
  (peak ≈ 19 at the very end), P8 grazed, **P9 never** — the sharp-edge case
  sends little flow that far downstream inside Marrone's own window, so their
  Fig. 24 P7 trace is short too.
Curve-for-curve against Colicchio's Level-Set (Fig. 24) still needs their
traces digitised — not done.

*(c) Viscous sub-case §3.4.2 — wiring DONE, runs stable.* `dambreak.configureScheme`
now plumbs `inviscid` / `nu` / `alpha` into `schemeConfig.diffusionParams` on
the non-ACSPH path (`inviscid=True` default → every existing run byte-identical;
`alpha=None` → don't touch the scheme's own artificial-viscosity coefficient).
`probe_deltaSPHMarrone34.py --Re` sets ν = √(gH)·H / Re. **Walls stay free-slip
— the no-slip half of §3.4.2 is a separate piece.** Re = 1000 (ν = 3.13e-3) and
Re = 10000 (ν = 3.13e-4), H/dx = 32, t\* = 5: both **5/6, no divergence**, bulk
ρ P99 1.025 / 1.020, KE and v_max within noise of the inviscid run, edge P2
34.1 / 22.6 ρgH — i.e. **viscosity barely moves the flow at these Re**, exactly
as Marrone states.

*(a) Bulk-max convergence — DONE for the H/dx = 32/64 pair.* Re-ran both with
`densityP99` + body-node ghosts (plain δ-SPH, t\* = 5): bulk ρ [P05, P99]
tightens **[0.9969, 1.0251] → [0.9992, 1.0127]** — the **bulk converges**; the
pointwise jet-tip max still diverges (1.11 → 1.20). This is the real bulk-max
number the old table lacked (it read `maxDensity`, 1.14 → 1.27). **H/dx = 128 to
t\* ≈ 7.4 (nx = 1024, multi-GPU-hour) still open.**

*Wall-leak residual scales the wrong way.* Plain δ-SPH tank-wall penetration
**4.66 dx (H/dx = 32) → 7.83 dx (H/dx = 64)** to t\* = 5 even with
`_bodyNodeGhostOffsets`; the viscous legs are the same order (8.06 / 7.33 dx).
Still the one-particle toe-apex creep, still needs Marrone's continuity fill
(§5.2.3). δ⁺ + PST is untouched by any of this — 6/6, wall pen 0.83 dx,
obstacle/fillet pen 0.87 dx to t\* = 7.4 — and remains the production config;
plain δ-SPH stays 5/6.

**2026-09-10 — SUPERSEDED by the boundary-sampling fix + `'gridsnap'` (see
"Current state").** All of the above (`_bodyNodeGhostOffsets`, the `capMax` /
`nodeDepthCap` / `layerReach` tuning, the 4.66 → 7.83 dx wall-leak scaling, the
one-particle toe-apex creep needing a continuity fill) was on the parked
body-node path. On the live path — aligned grid, `dn ≤ dx`, gridsnap ghosts on
the merged mollified surface, the wedge as a single seam-free `_convexPolygonSDF`
— **M34 nx = 256, t\* = 7, both schemes un-frozen, no tuning:**

| | tank-wall pen (whole run) | obstacle/fillet pen (whole run) |
|---|---|---|
| δ-SPH, no PST | **0.00 dx** | 1.81 dx → 0.77 dx / 36 ptcls at t\*=7 |
| δ⁺-SPH, PST | **0.00 dx** | 0.77 dx → 0.23 dx / 5 ptcls at t\*=7 |
| rounded-corner-only, either | **0.00 dx** | **0.00 dx** |

Both inside the ≤ 3 dx gate; the toe-apex continuity fill is no longer needed.
The `min(triangle, box)` obstacle SDF was itself a bug: the two primitives
touch along `x = toe_x` (on a lattice column), `min ≈ 0` up the seam, the
sampler drops that column, fluid slides in — that seam gap *was* the residual
obstacle penetration (1.25 → 0.77 dx). Runner: `run_dambreak_marrone34.py`
(`--mode {corner,wedge} --pst {on,off}`), videos in `scratchpad/marrone34_out/`.

### 5.2.3 English 2022 §4.2 — sloshing tank (SPHERIC TC10) under mDBC

**Result: first impact matches, negative-pressure transients captured; later
impacts overshoot (soft EOS — the case has no `machTarget`).** Getting there
took fixing two IC-generation bugs the case had been hiding behind.

**Two IC bugs, fixed in `9c7f878`:**
1. **`buildPointCloud` picked the lattice spacing from the *shortest* domain
   axis** (`min over d of span_d/nx`, `shortEdge=True`). `sloshingTank` is
   wide-and-shallow, so its short (y) axis won: particles landed at 0.0054
   while `config.dx` stayed 0.009 — a lie that still fed `dt`/`c_s` (c_s
   16.6 → 11) and the mDBC ghost offsets, with the lattice phase-anchored to
   `domain.min` so there was **no dp/2 fluid–wall stagger**. Fix: `buildPointCloud`
   takes an explicit `dx`; `sampleParticles` passes `config.dx` as
   authoritative, each axis snapping to `round(span/dx)` cells. `dambreak`
   et al. unchanged (their domains aren't axis-lopsided).
2. **mDBC ghost placement keyed off where the fluid currently was**
   (`_fluidDirectedGhostOffsets`, added `17290ff` for the §5.2.1 corner fix):
   in an 18-particle-deep layer the whole lower wall reads as "near fluid", the
   distance-weighted direction estimate isn't the wall normal, and the
   misalignment gate opens everywhere → ghost nodes scattered into the bulk,
   above the surface, and inside the wall (270–728 of them). The user's point:
   boundary discretisation must be a function of the geometry, not runtime
   state — an empty tank or a fill-in case would sample differently. Fix:
   **`_geometricGhostOffsets`** — English reflection, then a geometry validity
   pass (retract where the node crosses into any solid, collapse to Shepard/rest
   where no clean node exists). No fluid consulted. Byte-identical to the
   reflection on flat grid-aligned walls. (`_fluidDirectedGhostOffsets` was
   kept at the time for `dambreak`'s opt-in refresh path; both removed in
   `76af473` — dynamic ghost placement disallowed.)

**Regressions clean:** Marrone §3.4 still 6/6, obstacle penetration
**1.4 dx → 0.06 dx** (the validity pass beats the fluid-directed guess at the
sharp edge); `dambreak` smoke healthy (80/5000 tank-corner ghosts collapse to
the Shepard fallback — correct there); sloshing IC has **0/3320** ghost nodes
in a solid (was 728).

**English §4.2 run** (`examples/sloshingTank/output/english42_nx225_fixed/`,
`--scheme wcsph --nx 225` = dp = 0.004, t → 7 s, `michel2022` shift, mDBC):

| | sim | measured (SPHERIC TC10) |
|---|---|---|
| 1st impact | **t = 2.37 s, 3.8 kPa** (smoothed) | t ≈ 2.40 s, ≈ 3.6 kPa (band 2.2–13.1) |
| 2nd impact | t = 4.02 s, 8.9 kPa smoothed / 25 kPa raw | t ≈ 4.07 s |
| 3rd impact | t = 5.63 s, 8.7 kPa smoothed | t ≈ 5.71 s |
| neg. transient | **min −16.5 kPa** at t = 4.28 s; 6.8 % of t>2 s samples < −0.5 kPa | experiment shows short −ve spikes at violent impacts (English's mDBC vs DBC discriminator) |

**First impact — timing within 30 ms, magnitude within 6 %, in the
repeatability band.** The **negative-pressure transients English calls the mDBC
benefit over DBC are present** (short-lived spikes right after the violent
impacts). Runs the full 7 s, no divergence — vs NaN at t = 0.6 s before the fix,
in *every* scheme/c_s/AV combination tried.

**Open — soft EOS:** the later-impact raw peaks overshoot ~2× and ring, and the
raw signal carries heavy acoustic hash — because `c_s ∝ dx/targetDt` with no
Mach target, so it *falls* with resolution (25 at nx=100 → **11 at nx=225**),
and the impacts run at local Ma ≈ 0.75 (ρ swings to 1.58 / 0.62 → slam ≈ 8.5 m/s,
partly self-inflicted spray). The fix is the `machTarget` / `referenceVelocity`
path `rotatingSquarePatch` and `dambreak` already have, sized off the *impact*
velocity (c_s ≈ 40–85). `examples/sloshingTank/PLAN.md`'s stiffer trial
(c_s ≈ 50) already showed it drops the first-impact peak onto the measured
value and kills the density excursion. After that: `dp = 0.002` (nx=450); the
mDBC-vs-DBC three-row contrast; English §3.4.2 viscous sub-case.

**mDBC extrapolation quality (`afb6e59`).** Chasing the raw-signal roughness led
to a controlled test of the mDBC MLS: freeze the config, overwrite the fluid
density with an analytic field, run `computeMdbcDensity` once, compare the
boundary result to the field at the boundary positions (a Liu-Liu-style unit
check). Result: **the MLS operator is exact** — it reproduces a linear field to
float32 eps on ~92 % of near-fluid boundary particles at *every* point of a dam
break (still column through violent run-up). The roughness is the 8–33 % that
fall to the fallback — all at the free-surface contact line, growing with the
flow (8 % / 0.03 err at rest → 33 % / 0.6 err at dam-break run-up). The
fallback was Shepard density **clamped to ≥ ρ₀** (a DualSPHysics DBC
anti-attraction guard, not in English 2022) plus an **Adami hydrostatic term**,
picked by a **hard `where(wellConditioned, …)` switch** — so adjacent boundary
particles snapped between the exact MLS value and ~ρ₀, a visible step right where
the surface meets the wall. Now (`afb6e59` + `80eabb9`): a smooth MLS→Shepard
ramp on `numNeighbors` / `|det(A_g)|` (`w = 0` exactly below
`interpolateLiuLiu`'s determinant floor, so the Marrone §3.1 sheet-fling
protection is intact by construction), the hydrostatic term dropped from the
fallback, and the ρ_b ≥ ρ₀ clamp kept (the m2dbc anti-attraction guard —
`afb6e59` dropped it, `80eabb9` restored it). Pristine-IC MLS-path error
unchanged (5e-7); dam-break wall-impact ALL-boundary p95 error vs a linear field
**0.38 → 0.027**. `determinantThresholdFor` added to the `liu` module.

**Regression gates:**
- **englishWedge 9/9 — PASS.** Fresh run (dp = 0.02, wedge, t = 4 s): near-wall
  hydrostatic RMSE 0.0069 (gate ≤ 0.08), base-corners RMSE 0.024 (gate ≤ 0.03),
  no penetration, ρ ∈ [1.0001, 1.0025]. English §4.1 is still water where every
  departure is the wall closure alone, so this clears the change.
- **Marrone §3.1** (`sun2017DeltaSPH`, nx = 60, to t\* ≈ 4) — `diverged=False`.
  The det-gate sheet-fling protection held.
- **Marrone §3.4 nx = 256 tank-wall penetration — FAIL, NOT a regression**
  (present since `8d5e22e`). Chased 2026-09-09; **root cause found AND FIXED:
  broken mDBC ghost placement at the obstacle-toe re-entrant corner** (fluid
  sank through the bed there once the sharp-edge jet formed) — replaced by
  `_bodyNodeGhostOffsets` (Marrone App. A / English §3; see below), leak now
  ≤ 1 dx. Trail:
  - **`9c7f878` ruled out directly.** A/B of the probe at HEAD vs `9c7f878^`
    (`2571265`, git worktree): the M34 nx = 256 init state is **byte-identical**
    — same `config.dx`, fluid count (7392), particle mass (10 sig figs), realised
    lattice spacing, bbox, boundary/ghost counts, ghost-offset distribution,
    `c₀`, `dt` — and the full t\* = 5 runs return **identical** numbers
    (pen 29.81 dx, ρ P99 1.0232, max‖v‖ 28.20, KE trend, 7139 steps). `9c7f878`'s
    `config.dx` sampler branch is inert here (domain axes are already integer ×
    `config.dx`, so `round == ceil`) and `_geometricGhostOffsets` reduces to the
    legacy reflection on M34's dry init.
  - **afb6e59 / 80eabb9 already ruled out** (the 76f61b4 revert test above).
    Those + `9c7f878` are the *only* behavioural src deltas on the plain-δ-SPH
    M34 path since `8d5e22e`; everything else is opt-in
    (`mdbcGhostRefreshEvery = 0`) or diagnostic (`densityP99`).
  - **The leak is a slow progressive escape, not a transient.** `maxPenetrationDx`
    grows monotonically `0.9 → 5.6 → 6.8 → 8.0 → 14.9 → 29.8 dx` across
    t\* = 1.5 → 5.0, with `nPenetrating` 5–21 particles; `maxObstaclePenDx`
    stays 0.10.
  - **LOCATION (leak-tracking run, 2026-09-09): fluid sinks through the *bed*
    (y = −L/2), 100 % on the −y face, at x ≈ 0 → 0.9 — i.e. right at the
    **obstacle toe**, the re-entrant corner where the 45° hypotenuse meets the
    floor** (fluid-side angle 5π/4). Onset t\* ≈ 2.1 as the sharp-edge jet
    forms and drives fluid down into that corner; `maxpen` 0.5 → 5.3 dx by
    t\* = 2.5, particles reaching y ≈ −4.17 (5 dx below the bed). NOT the
    fillet / downstream wall — the earlier "fillet run-up" reading was wrong.
  - **CAUSE: ghost placement IS broken here — the composed-SDF gradient fails
    at this re-entrant corner.** Layer-1/2 bed boundary particles in the leak
    footprint (x ∈ [−0.6, 1.4]): **28 % have a collapsed (zero) ghost offset →
    Shepard/rest density → ~no mDBC wall pressure**, vs **0 %** on a clean flat
    bed strip. Across the wider toe box, **42 % of ghost directions disagree
    with the true surface normal by > 45°** (a fine-`region.sdf`-contour
    nearest-point mirror; the same test on the *fillet* box gave 0 %
    disagreement — that region really is fine, which is why the first two
    checks mis-cleared "ghost placement"). Fluid piling into a wall of
    rest-density bed particles at the toe just sinks through.
  - This is a **Marrone 2011 Fig. A.33** re-entrant corner (θ > π). **Mostly
    fixed** by `_bodyNodeGhostOffsets` (`1e145a1`) — full write-up in the
    "IMPLEMENTED" block below. Net: M34 nx = 256 plain δ-SPH penetration
    **29.8 → 4.66 dx** to t\* = 5 (≤ 1 dx through t\* = 3.9), englishWedge 9/9 +
    better, all tests pass. The 4.66 dx is **one particle** slow-creeping
    through the toe *apex* after t\* ≈ 4.5 — 1.7 over the ≤ 3 gate. The bisector
    rule for θ > π was tried and backed out (IMPLEMENTED block): at the sharp
    apex the fluid wedge is too thin for any single-ray node; Marrone's
    continuity fill (borrow a neighbour's node) is the real fix, deferred.
    δ⁺+PST is production for M34 (0.58 dx). Earlier heuristics (CD normal,
    ascent, Newton) had been tested against the *wrong* (fillet / deep-interior)
    particle sets and so looked useless.
  - **Configs that pass** at HEAD, nx = 256, t\* = 5: `sun2017DeltaSPH` + PST
    (0.58 dx — the shift keeps particles off the wall regardless). The
    `deltaSPH` + `mdbcGhostRefreshEvery = 5` config that also passed (0.68 dx)
    is **removed** (`76af473`) — dynamic ghost placement for a static boundary
    is disallowed.

**Ghost placement — the method the papers actually specify** (Marrone 2011
App. A; English et al. 2022 §3, "a procedure similar to Marrone"):

The boundary is **discretised into body nodes with analytic normal + tangent
unit vectors** (the geometry is known parametrically — a polyline of wall
segments and arcs — *not* a composed SDF). Fixed ghost particles sit in layers
`ds/2` outside; each has an interpolation point (our "ghost node") placed by:

| fluid-side angle θ at the node | rule |
|---|---|
| flat wall / along one side of a corner | mirror across the interface **along the analytic normal**, `r_g = r_b − 2 φ n̂` |
| inside a convex solid wedge, π/2 ≤ θ ≤ π (English Fig. 1c) | **central symmetry**: reflect through the corner *point*, `r_g = 2 r_corner − r_b`. Keeps adjacent ghosts → adjacent interpolation points (needed for a smooth fluid response). |
| re-entrant / concave corner, θ > π (Marrone Fig. A.33) | interpolation points along the **angle bisector**; any point mirrored **> one support radius** past the vertex is dropped and the ghost takes the value of the furthest valid bisector point within a support radius |
| sharp convex, θ < π/2 (Marrone Fig. A.32) | central symmetry for nodes in the wedge; the uncovered solid sliver between `s'` and `s` is filled *by continuity* — each row takes the interpolation point of its intersection with the `s'` mirror line |
| θ ≈ 2π | two ghost sets + a visibility criterion |

DualSPHysics stores this as a precomputed per-boundary-particle vector
`boundnor` (`JSphCpu_mdbc.cpp`: `ghost = pos + boundnor`, so `boundnor` is the
*full* offset, not a unit normal), built once by `JSphBoundCorr` from the
boundary discretisation / `_Dp.xml` normals.

**IMPLEMENTED — `_bodyNodeGhostOffsets`, 2026-09-09** (`rigidBody/ghostParticles.py`).
warpSPH has no analytic body nodes (the geometry is one composed `min`/`max`
SDF; boundary particles are a lattice cut by `sdf < 0`). So the body-node
discretisation is taken from the SDF itself: a **fine marching-squares polyline
of `region.sdf`** (`_boundaryPolyline`, ~2.5 samples/dp, built once, no fluid).
Then per boundary particle:

* nearest point `r_s` on the polyline → **mirror through it**, `r_g = 2 r_s − r_b`.
  One rule for every Marrone case: flat wall → perpendicular foot → standard
  mirror; convex solid wedge → `r_s` collapses to the apex → central symmetry
  through the vertex; re-entrant corner → `r_s` on the nearest wall → mirror
  across it into the fluid.
* validity retract (shorten `r_g` toward `r_s`, never lengthen) so the node
  clears every solid; cap the offset at `1.5 h`; past that → zero offset →
  Shepard/rest (Marrone's "not considered" rule).

Wired in `addBoundaryGhostParticles`: **2D uses `_bodyNodeGhostOffsets`**; 3D
(no `find_contours`) keeps the old `∇(sdf)` `_geometricGhostOffsets`. No fluid
consulted, static for the run.

**Results:**
- **englishWedge 9/9 — PASS, and better than before.** Re-entrant wedge-base
  corner RMSE **0.024 → 0.0022** (~10×), near-wall/bed **0.0069 → 0.0045**,
  apex 0.0020, faces 0.0013, 0 dx penetration, ρ ∈ [1.0001, 1.0025].
- **Marrone §3.4 nx = 256 plain δ-SPH — the toe leak is gone.** Leak-tracking
  run: `maxPenetrationDx` stays **≤ 1.0 dx through t\* = 3.9** (was 5.3 dx by
  t\* = 2.5 and climbing to ~30 by t\* = 5), `nPenetrating` **1–2** (was 5–21),
  no divergence. **Full t\* = 5 acceptance run: 29.8 → 4.66 dx** (5/6).
  - The **4.66 dx residual is one particle** at the toe (x ≈ 0.87) that had
    been stable at ≈ 0.9 dx through t\* ≈ 4.5, then resumes a slow creep
    (~0.3 dx per 0.1 t\*, `w_rho ≈ 1.0`, near-zero velocity — not a jet, a
    quasi-static leak through the imperfect re-entrant corner). Plus 6–7
    particles at **~1–2 dx** on the *upstream* back wall (x = −W/2) from
    t\* ≈ 3.9 — the fragmenting sheet arcing back over the reservoir
    (Marrone Fig. 20/22) making contact; minor, flat wall.
  - So the toe is 6× better and no longer a runaway, but not perfectly sealed.
- `test_physics` / `test_wallPressure` / `test_deltaSPHDiffusion`: all pass.

**Bisector rule — attempted 2026-09-09, backed out.** Added a second pass:
detect a re-entrant particle (a second wall face — different segment tangent,
`|t̂·t̂| < 0.94` — comparably close, and the one-wall mirror only marginally
clears) and place the node along the bisector of the two wall directions,
walked out to the deepest clearance a support radius allows. It fires cleanly
on flat walls (no false trigger) and *did* deepen the near-toe ghosts, **but
does not fix the residual**: right at the toe *apex* the fluid wedge is
genuinely too thin for any single ray (mirror, bisector, retract) to reach a
useful depth — the best bisector node there is still only ~0.2–0.9 dx clear.
Marrone's real treatment of that (Fig. A.32/33) is the **continuity fill**:
those apex particles *borrow* the interpolation-point value of a neighbour
further along a clean wall, rather than getting their own node. That is a
bigger, separate piece (needs per-particle "borrow from" links). Deferred —
the payoff is one creeping fluid particle at t\* = 5, and **δ⁺-SPH + PST
already seals M34 fully (0.58 dx)** and is the production config.

Still a *quality* item, not done: an analytic per-primitive normal (vs the
polyline discretisation) would be exact on curved walls and cheaper than the
`O(n · segments)` marching-squares polyline. Plus the continuity-fill above.

The DFSPH wall-pressure path (`modules/incompressible/wallPressure.py`) has an
analogous structure and was not touched.

---
_Original scoping notes:_

### 5.2.3 (scoping) English 2022 §4.2 — sloshing tank (SPHERIC TC10) under mDBC

**The plan's "mDBC vs the current treatment" framing is wrong: mDBC *is* the
current treatment.** `sloshingTank` under `deltaSPH` already goes through
`addBoundaryGhostParticles` (every WC case with a `RegionType.Boundary` region
does) and `deltaSPH_step` calls `computeMdbcDensity` unconditionally — verified:
a 3-step run has 3984 kind==2 ghost rows and the Sensor-1 reading is the
MLS-extrapolated boundary density through the EOS. There is no toggle on the WC
path (the `BoundaryPressureMode` enum plain/mdbcDensity/consistent lives on
`IncompressibleSolverConfig`, DFSPH only).

**Prior work (`examples/sloshingTank/PLAN.md`, 2026-09-03/04):** TC10 verified
for wiring (applied roll overlays prescribed, sensor first responds at t ≈ 2.35 s
≈ measured). But that pass predates the §5.3.2 shift fixes — its WCSPH runs
**diverged at t ≈ 3.4 s** (free-surface tensile instability at the sensor
corner). The memory notes this was later fixed by making `surfaceNormal`
(now `michel2022` for this case) the default shift; the tracked PLAN.md still
shows the diverging behaviour. DFSPH reproduces the pressure (first impact on
the money; wave runs 2–5 % fast, converges with resolution; free surface
collapses, doesn't). None of it was framed as an English §4.2 mDBC validation.

**English §4.2 spec** (their §4.2, *Comp. Part. Mech.* 9:911–925): SPHERIC TC10
exactly — 0.9 × 0.508 m tank, H = 0.093 m, rolled per the benchmark table.
**dp = 0.004 m and 0.002 m**, h/dp = 2 (→ our `nx` = L/dp = 225 and 450).
First left-wall (Sensor 1) impact at t ≈ 2.47 s. Figure of merit is **Fig. 9**,
three rows: (1) DBC, gauge at the true Sensor-1 location → *erroneous* (an ~h
gap with no fluid particles); (2) DBC, gauge moved +h into the fluid → much
better; (3) **mDBC, gauge at the true location → very good** agreement with the
experiment. Plus: mDBC captures the **short-lived negative-pressure transients**
(O(1) time steps) at violent impacts that the experiment shows and DBC misses;
and mDBC wall-particle pressures are less spatially noisy (their Fig. 8).

**Proposed work:**
- **Phase 1 (no code).** Fresh WCSPH + mDBC run at dp = 0.004 and 0.002 with the
  *current* shipped defaults (post shift-fix) to t ≈ 6–7 s via
  `examples/sloshingTank/run_sloshingTank.py`. Confirm no divergence; grade
  Sensor 1 vs the TC10 experiment + repeatability band (peak timing/magnitude,
  already ~matched pre-fix at low res) and specifically check the
  **negative-pressure transients** appear at the violent impacts — that is the
  mDBC discriminator vs DBC, and the one thing the prior pass never looked for.
  Also try `--scheme sun2017DeltaSPH` (frozen + Eq. 7) as an alternative to
  `michel2022` now that it exists.
- **Phase 2 (small scheme hook), only if Phase 1 leaves the mDBC>DBC contrast
  unshown.** Add a WC-path boundary-density toggle (`plain` = skip
  `computeMdbcDensity`, read the boundary particle's own evolved density) and a
  gauge `+h` offset option, to reproduce Fig. 9's three rows directly.
- Skip English §4.2's `dp = 0.002` if the `nx = 450` run is too slow; `nx = 225`
  is already at their coarse resolution.

## 5.3 δ⁺-SPH suite (after δ-SPH is clean) — Sun 2017 §4 / Sun 2019 §3

`rotatingSquarePatch` (N°1, the PST discriminator — tensile instability without
it), `oscillatingDroplet` (N°2, conservation under a body force),
`impact` (N°3). N°4–7 are bluff-body wake flows needing an inflow/outflow and a
`movingObstacle`-class case — lower priority.

**Start with Sun 2019 §3.1's Taylor–Green flow instead**, ahead of all of
those. It is the only case in either suite with **no boundary of any kind** —
no wall, no free surface, no inflow — so it measures the scheme and nothing
else, and it has a closed-form answer (Sun 2019 Eq. 22,
`f(t) = f(0) exp[−16π²ν t]`, governing *both* the kinetic energy and the
pressure at a fixed point). Every case scored before it — Marrone §3.1's dam
break, the sloshing tank, the hydrostatic column — measures the scheme *and*
its wall closure together, so a 20 %-low answer there has two possible owners
and no way to separate them.

### 5.3.1 Sun 2019 §3.1 — Taylor–Green flow ← DONE, and it found the PST bug

`scripts/probe_deltaPlusTGV.py` (new), on `cases/tgvWeaklyCompressible`
(`tgv-wc`). Setup exactly as the paper: periodic `[0,L]²`, four counter-rotating
vortices, `Re = UL/ν = 100`, `L/Δx = 50…400`, Wendland C2 at `h/Δx = 2`, RK4,
`c₀ = 10 U` via the Sun 2017 Eq. (2) path, recorded to `tU/L = 1`.

**Which curve is the target.** Sun 2019's Figs. 6–9 each plot *three* models:
δ-SPH, "δ⁺-SPH by Sun et al. [5]" (= Sun 2017, which is what this repo
implements) and "present δ⁺-SPH" (Sun 2019's own consistent-PST scheme, which
folds the shifting transport back into the continuity and momentum equations —
that one is `PST_ALE_PLAN.md`'s target, not this plan's). So the middle curve
is what a correct implementation here must reproduce, **including its known
errors**: a centre pressure that drifts progressively above the analytic
solution, and a volume error that cumulates rather than settling.

**Case changes needed to run it at all** (all default-off, no existing run
changes):
- `cases/weaklyCompressible.setupTimestep` now honours `machTarget` /
  `referenceVelocity` (Sun Eq. 2), which had been wired only inside
  `dambreak.py`'s own `initialConditions` — this is Part 6 step 2's "c₀ rework
  for *other* WCSPH cases", done in the shared block so any case that declares
  the two params gets it;
- `tgv-wc` gained `shifting` (the δ vs δ⁺ A/B), `initialPressure` (start *on*
  the analytic pressure field — a weakly compressible scheme carries its
  pressure in the density, so a uniform-`ρ₀` start has to radiate the whole TGV
  pressure field into existence, and that acoustic transient is the same size
  as the signal being measured), `phase`, and the `pCentre` / `volumeError`
  (Sun Eq. 23) diagnostics.
- **`phase` matters and is not cosmetic.** Sun reads his pressure at the box
  centre; his Fig. 8 profile along `y = 0.5L` peaks at `x/L = 0, 0.5, 1`, so
  that point is a **stagnation point** (`p(t₀) = +P₀`), not a vortex core. The
  case's own even-`k` rule put a vortex core there instead. What Figs. 7–9
  measure is a drift of the *mean* pressure — an additive offset — so dividing
  it by a `p(t₀)` of the wrong sign flips which side of the analytic curve it
  lands on: measured, `p/p(t₀) = −0.33` against Sun's `+0.47`, purely from that.

**Result — the δ-SPH half is validated; the δ⁺ half was not doing its job.**
At `L/Δx = 400`, `tU/L = 1` (analytic `0.206`):

| | our δ-SPH | Sun δ-SPH | our δ⁺ (before) | Sun δ⁺-2017 |
|---|---|---|---|---|
| `p/p(t₀)` at `tU/L=1` | **0.632** | 0.680 | 0.675 | 0.470 |
| `ε_V` (Eq. 23) | **0.206 %** | 0.220 % | 0.227 % | 0.125 % |
| `ε_V` growth `t=1 / t=0.3` | 1.26 (plateau) | plateau | 2.72 | cumulates |
| KE max rel. error vs Eq. (22) | 0.037 | ~0 | 0.027 | ~0 |

The δ-SPH leg lands within **7 %** of Sun's on both metrics, and the KE decay is
within 3 % of Eq. (22) at every resolution (`ν_eff/ν = 0.983`). But the δ⁺ leg
sat **on top of our own δ-SPH leg** instead of improving on it — where Sun's PST
takes `ε_V` from 0.22 % to 0.125 %, ours moved it from 0.206 % to 0.227 %. The
pressure and volume errors also tracked each other exactly (ours were 1.78× and
1.82× Sun's), which is Sun §3.1's own claim — the mean-pressure offset *is* the
volume error, since `p = c₀²(ρ−ρ₀)`.

### 5.3.2 Root cause — the δ⁺ shift was 1/8 of Sun 2017 Eq. (7)

Eq. (7) verbatim (`literature/sun2017_*.pdf` p. 28), `φ_ij = 1`, `h_ij = h`:

```
δr_i := −CFL·Ma·(2h)² Σ_j [1 + R (W_ij/W(Δx_i))^n] ∇_iW_ij · 2 m_j/(ρ_i+ρ_j)
```

with **R = 0.2, n = 4**. Against that:

| Eq. (7) | the code | ratio |
|---|---|---|
| prefactor `(2h)² = 4h²` | `delta.py`: `2h²` | ½ |
| volume weight `2 m_j/(ρ_i+ρ_j)` | `wp_deltaShift.py`: `0.5 m_j/(ρ_i+ρ_j)` | ¼ |
| `R = 0.2` | `computeDeltaShiftWarp`'s default `R = 0.25` | — |

The missing factor of 2 on the prefactor carried an in-code justification —
"we include the 2 from the mean density term in the computation of the shift so
it's not `(2h)²`". That accounting does not hold: the `2` in
`2 m_j/(ρ_i+ρ_j)` is Eq. (7)'s own volume weight, a *separate* factor from the
`(2h)²` prefactor, and the raw kernel sum carries `0.5 m_j/(ρ_i+ρ_j)` rather
than either of them. Net: **1/8 of Eq. (7)**.

Measured, not argued — `scripts/probe_deltaPlusShiftMagnitude.py` (new)
recomputes the literal Eq. (7) from the same raw kernel sum on a running TGV
state and reports the ratio: **0.131× at `L/Δx` = 50, 100 and 200 alike** (the
residual over 1/8 is the `R = 0.25` vs `0.2` difference). It also checks Sun's
own Eq. (8), `|δr_i|/Δx_i < 0.05` "in all the simulations performed": the
historical shift reads `0.0007`, i.e. **70× below the scale the paper says its
own PST operates at**; Eq. (7)'s reads `0.022`, inside the bound and the right
order.

**Fix — `ShiftProperties.sun2017Eq7Shift`, opt-in.** Same precedent as
`freezeDiffusionAcrossStages`: default `False`, so every existing case using
`ShiftingScheme.deltaSPH` is byte-for-byte unchanged, and
`Sun2017DeltaSPHConfig` (`--scheme sun2017DeltaSPH`) defaults it `True`, so
"the paper's own prescription" stays an explicit choice. `tests/test_physics.py`
72/72 green before and after.

**Validated — the disagreement is gone.** Same probe, `Re = 100`, `tU/L = 1`,
at Sun's own compared resolution and at half of it:

| | `L/Δx` | before (1/8 shift) | **after (Eq. 7)** | Sun δ⁺-2017 |
|---|---|---|---|---|
| `p/p(t₀)` | 200 | 0.771 (+64 %) | **0.479 (+2 %)** | 0.470 |
| `ε_V` | 200 | 0.2737 % (+119 %) | **0.1293 % (+3.4 %)** | 0.125 % |
| KE max rel. error | 200 | 0.030 | **0.021** | — |
| `p/p(t₀)` | **400** | 0.675 (+44 %) | **0.458 (−2.6 %)** | 0.470 |
| `ε_V` | **400** | 0.2273 % (+82 %) | **0.1241 % (−0.7 %)** | 0.125 % |
| KE max rel. error | **400** | 0.027 | **0.019** | — |

At the matched resolution all three of Figs. 6, 7 and 9 agree with the paper to
within the digitisation error of reading them. `scripts/old/out_deltaPlusTGV/`
holds both legs side by side (`*_eighthEq7.npz` are the pre-fix runs) with
`REPORT.md` and the three-panel figure as the record.

**`Re = 1000` — the pathology reproduces too, and at Sun's own resolution the
whole benchmark is 5/5.** Sun's right-hand panels are the long-time run
(`tU/L = 10`), where his δ⁺-2017's pressure error becomes dramatic:

| at `tU/L = 10` | ours, `L/Δx = 200` | ours, `L/Δx = 400` | Sun δ⁺-2017 (`L/Δx = 400`) |
|---|---|---|---|
| `p/p(t₀)` | 2.79 (+16 %) | **2.56 (+6 %)** | 2.40 |
| `ε_V` | 1.27 % (×1.27) | **1.16 % (×1.16)** | 1.00 % |
| `ε_V` growth `t=10 / t=3` | 2.23 | **2.40** | cumulates |
| KE max rel. error | 0.112 ❌ | **0.045 ✅** | — |

The *characteristic Sun-2017 δ⁺ failure* — a centre pressure climbing
monotonically to ~2.4× its initial value while the volume error accumulates
past 1 % — is reproduced in shape and magnitude. That is the stronger of the
two agreements: it says the implementation matches the scheme including its
pathology, not merely its good behaviour.

**The KE excursion was under-resolution — resolved, not open.** At
`L/Δx = 200` the kinetic energy decayed *too slowly*, peaking at `+11 %` over
Eq. (22) around `tU/L = 2–3`, which Fig. 6's right panel does not license (the
green δ⁺-2017 curve overlays the analytic for the whole record there). Two
candidates were put up with opposite predictions — (a) plain under-resolution,
which the paper's own warning about low viscosity supports, predicting the
excursion roughly halves at `L/Δx = 400`; (b) kinetic energy leaking from the
smooth TGV mode into small-scale particle-disorder motion, which does not
dissipate at the mode's rate and would therefore largely persist. The
`L/Δx = 400` run settles it:

| `tU/L` | 0.5 | 1 | 2 | 3 | 5 | 7 | 9 |
|---|---|---|---|---|---|---|---|
| rel. err, `L/Δx = 200` | +2.1 % | +6.1 % | +11.1 % | +10.9 % | +9.6 % | +7.5 % | +4.4 % |
| rel. err, `L/Δx = 400` | +0.9 % | +1.7 % | **+1.8 %** | +1.7 % | +0.9 % | −0.6 % | −3.0 % |

The excursion collapses by ~6× for a 2× resolution increase — steeper than (a)
even predicted, and flatly incompatible with (b). It also **changes sign** by
`tU/L ≈ 7`, i.e. at the finer resolution the late-time error is ordinary
numerical *over*-dissipation rather than the under-dissipation that looked
anomalous. `Re = 1000` at `L/Δx = 200` is simply below the resolution this
benchmark needs; Sun compares at 400 for the same reason.

**Should `sun2017Eq7Shift` be the shared default? — swept, and the evidence
says yes, but the decision is not taken here.** It is a correctness fix against
the source equation, so the burden is on keeping the old value; the 8× shift is
still a real physics change for 11 cases, so it was swept first
(`scripts/probe_deltaPlusShiftSweep.py`, three legs — `off` / `eighth` / `eq7`
— 300 steps each). Comparing the **last** step of each run:

- **no case diverges**, and the density band is unchanged everywhere;
- `pairedFraction` — the tensile-instability signature the PST exists to
  suppress — falls or stays at zero in all 11: `drivenSquare` 0.0002 → 0,
  `movingObstacle` 0.0017 → 0.0003, `openFlow` 0.0027 → 0.0019;
- `nnDistP01` (higher is better) improves in 8 and is flat in 2: `dambreak`
  0.724 → 0.758, `movingObstacle` 0.798 → 0.843, `openFlow` 0.672 → 0.744,
  `tgv-wc` 0.845 → 0.882;
- two get slightly worse — `impact` 0.951 → 0.910 and `droplet` 0.953 → 0.942.
  `droplet`'s own 15-period runs (§5.3.3) are excellent under Eq. (7), so that
  one is noise at step 300; **`impact`'s −4 % is the one unexplained cost** and
  should be checked over a full record before the default flips.

**A startup transient, and the misreading it caused.** The first pass at this
sweep reported geometric metrics as their worst value over the run, and on that
table `tgv-wc` looked like the one case the fix breaks: `pairedFraction`
0 → 0.146, `voidFraction` 0 → 0.167, `nnDistP01` 0.845 → 0.131. It is the
opposite. The spike is at **step 0** and heals monotonically; the full-length
validation runs in `scripts/old/out_deltaPlusTGV/` — the same runs that match Sun
to 3 % — carry the identical `paired = 0.14` first sample and end at
`paired = 0.0000`, `nnDistP01 = 0.877`, a *better* distribution than the ⅛
leg's 0.814. A 300-step window caught the transient and nothing else.

The transient itself is real and worth its own item: **on a freshly-shuffled
lattice the Eq. (7) shift briefly violates Sun's own Eq. (8)** — measured
`|δr|/Δx` mean 0.063, max 0.212 at step 0, against his `< 0.05` — because the
raw kernel sum is large on a badly-relaxed distribution. By step ~40 of a
running flow it is back to 0.003/0.017. Sun does not meet this because his
§3.1 initial distribution comes from Colagrossi's **packing** algorithm, where
ours comes from `shuffleParticles(jitter=1.0)`. So this is a *sampling* gap,
independent of the shift scaling, and the fix is a better initial relaxation
rather than a smaller shift.

The sweep tool now reports `max (last)` for exactly this reason — a worst-value
column cannot separate a transient from a degradation, and here that
distinction was the entire answer.

### 5.3.2a The square-patch PST discriminator + the shift-magnitude decision

Two follow-ups from Open/next 1 & 2, resolved together.

**(1) The rotating square patch IS a working PST discriminator, and it had
never been run as one.** `scripts/probe_squarePatchPSTControl.py` (new): the
three legs (`off` / `eighth` / `eq7`) on `squarePatch`, scored on the
*geometric* pairing signature over a time trace (not two points — the §5.3.2
transient caution applies). At `nx = 384`, before the arms fragment
(fragmentation is under-resolution, `probe_squarePatchFragmentation.py`, a
different phenomenon):

| tω | `off` paired / nnP01 | `eighth` | `eq7` |
|---|---|---|---|
| 1.0 | 0.031 / 0.35 | 0.007 / 0.52 | 0.002 / 0.65 |
| 2.0 | 0.103 / 0.10 | 0.029 / 0.35 | 0.014 / 0.46 |
| 3.0 | **0.213 / 0.060** | 0.061 / 0.28 | **0.044 / 0.32** |

`off` (plain δ-SPH) pairing **grows monotonically** — 0.03 → 0.21, `nnP01`
collapses 0.35 → 0.06, `voidFraction` → 0.11 — a real tensile instability, not
a transient. Both shift magnitudes suppress it, monotonically in magnitude: at
tω = 3, paired is 0.213 (`off`) → 0.061 (`eighth`) → 0.044 (`eq7`), and `nnP01`
0.06 → 0.28 → 0.32. Snapshot (`out_squarePatchPSTControl/`): `off` is
salt-and-pepper pressure noise everywhere, `eq7` keeps a coherent field. So the
square patch + TGV (§5.3.1) are now a clean **PST-needed / PST-works** pair, and
the discriminator prefers the Eq. (7) magnitude.

**(2) The free-surface re-sweep — `eq7` is only safe with frozen diffusion.**
The 300-step smoke was extended to **2500 steps** on the four free-surface
cases (`scripts/old/out_deltaPlusShiftSweep_fs/`). `droplet` neutral, `squarePatch`
clearly better at `eq7` (paired 0.056 → 0.017), `impact` a small regression
(paired 0.015 → 0.018, `nnP01` 0.39 → 0.35 — the same −4 % the smoke flagged,
now confirmed as a genuine small cost, not noise). But **`dambreak` at `eq7`
under `--scheme deltaSPH` breaks** by t ≈ 1.25 s: 20 % paired, **21 % voids**,
a localised singularity (ρ_max 1122, ‖v‖ 90762, KE up 11 orders) — the bulk
(`densityP05` 0.993) survives but the run is destroyed. The `eighth` leg is
clean at the same point.

**Frozen diffusion (`--scheme sun2017DeltaSPH`) rescues it completely** —
identical `dambreak eq7`, 2500 steps: ρ_max 1.06 (1.37 over the run), ‖v‖ 8.4,
paired **0.028**, void **0.003**, `nnP01` 0.29 — a *better* distribution than
the un-frozen `eighth` leg (0.034). So the `eq7` blow-up is specifically the
8× shift interacting with per-sub-stage RK4 diffusion; freezing the diffusive
terms across sub-stages (Sun 2017 §2, which `sun2017DeltaSPH` already does)
removes it.

**Decision: `sun2017Eq7Shift` stays coupled to `sun2017DeltaSPH`, NOT promoted
to a scheme-agnostic default.** Flipping the global default would give
`--scheme deltaSPH` runs the 8× shift *without* frozen diffusion — which
diverges `dambreak` and slightly regresses `impact`, for no benefit: the cases
that want the bigger shift (square patch, TGV, violent free-surface impacts)
are run under `sun2017DeltaSPH` anyway, where it is already on. `eq7` is the
correct magnitude (§5.3.2), and it is correctly *scoped* to the scheme that
carries the rest of Sun's prescription with it.

**Which factor is load-bearing — frozen diffusion, not the shift magnitude.**
The 2×2 completed with a frozen-`eighth` run (`dambreak`, `--scheme
sun2017DeltaSPH`, ⅛ shift, 2500 steps): no divergence, paired **0.031**, void
**0.003**, `nnP01` 0.27 — statistically the same distribution as frozen-`eq7`
(0.028 / 0.003 / 0.29). So:

| | `eighth` (⅛·Eq.7) | `eq7` (full, 8×) |
|---|---|---|
| un-frozen (`deltaSPH`) | ✅ paired 0.034, ρ∈[0.80, 1.38], ‖v‖ 12.3 | ❌ **diverges** — ρ 1122, ‖v‖ 90762, 21 % void |
| frozen (`sun2017DeltaSPH`) | ✅ paired 0.031, ρ∈[0.59, 1.39], ‖v‖ 19.5 | ✅ paired 0.028, ρ∈[0.77, 1.37], ‖v‖ 15.4 |

Un-frozen + unscaled already works (it is the current default and nothing here
changes it); the *only* broken corner is un-frozen + scaled, and **either**
freezing the diffusion **or** keeping the ⅛ magnitude fixes it independently.
Freezing is what delivers the clean distribution — on top of frozen diffusion
the Eq. (7) magnitude is a marginal further gain (tighter ρ band: min 0.77 vs
0.59, peak ‖v‖ 15.4 vs 19.5), not the load-bearing factor. This reinforces the
scoping decision: the good result travels with `sun2017DeltaSPH`, and `eq7` is a
mild bonus there.

### 5.3.2b The square-patch validation figure vs Sun 2019 Fig. 13

`scripts/probe_squarePatchValidationFigure.py` gained `--scheme` (so the row
runs the production δ⁺ config, `sun2017DeltaSPH` = frozen diffusion + Eq. (7)
shift) and the Sun 2019 §3.3 conservation errors ε_M / ε_E (Eqs. 26–27) plus
footprint drift, added to `squarePatchAreaMetrics` (`epsAngularMomentum`,
`epsKineticEnergy`, `linearMomentumMag`). Run at L/Δx = 200 (`nx = 600`), times
tω = 1/2/3/4.

**Poisson pressure initialisation** (`squarePatch` param `poissonPressureInit`,
opt-in, `box` only). The patch is not an equilibrium; its t=0 pressure is the
solution of ∇²p = 2ρ₀ω² with p = 0 on the free surface — for the square, 2ρ₀ω²
times the Saint-Venant torsion function (Prandtl stress function), negative in
the interior with p(0,0) ≈ −0.589 ρ₀ω²a². The case previously seeded only the
rotating velocity, so p developed from zero through an **undamped acoustic core
mode** — blue (−5) → white → red (+5) → blue over tω = 1…4, nothing like
Fig. 13. Seeding the torsion function (imposed via density, ρ = ρ₀ + p/c₀²,
since the isothermal EOS recomputes p from ρ each step) gives a **stable
coherent negative core through tω = 1–3**, matching Sun's "negative pressure
core". Verified the series: ∇²v = 1 to machine precision, x↔y symmetric, v = 0
on the boundary.

**Result (nx = 600, δ⁺ + Poisson init):**

| tω | ε_M | ε_E | ‖p‖_lin/ρωL³ | vol drift | field |
|---|---|---|---|---|---|
| 2 | 0.06 % | 0.19 % | 2e-5 | −0.03 % | negative core, 4 coherent arms, noise-free |
| 3 | 0.19 % | 0.70 % | 8e-5 | −0.03 % | ″ |
| 3.5 | 5.1 % | 2.8 % | — | −0.04 % | arm tips begin to break |
| 4 | 36 % | 16 % | — | −0.06 % | fragmented |

Through **tω ≈ 3** this is a clean qualitative + quantitative match to Fig. 13:
area-conserving spiral rotation, four arms from the corners, smooth pressure
with the negative core, near-perfect conservation (ε_M < 0.2 %, ε_E < 0.7 %,
volume drift < 0.04 %, linear momentum ~1e-4 ρωL³). Poisson init does **not**
delay the fragmentation onset (identical to the p=0 run at both L/Δx = 100 and
200) — the arms are ~2 particles thick by tω ≈ 3.2 and break up regardless; the
seed is long forgotten by then. Holding the arms to tω = 4 as Sun does needs
L/Δx = 400 (`nx = 1200`, a multi-hour run); consistent with
`probe_squarePatchFragmentation.py`'s finding that arm break-up here is
under-resolution, not a PST or init failure. Figures:
`scratchpad/sp_validation_nx600_poisson.png`,
`scratchpad/sp_nx300_{poisson,nopoisson}.png` (the A/B).

### 5.3.3 Sun 2017 §4.2 — oscillating droplet

`scripts/probe_deltaPlusDroplet.py` (new). Sun's own spec: `f = −B²r`,
inviscid, `A₀/B = 1`, `c₀ = 15 A₀R`, `α = 0.01`, `R/Δx = 50/100/200`, **15
oscillation periods**. Scores his **Table 1** (maximum momenta-conservation
errors — both momenta are identically zero in the continuum, so the recorded
value *is* the error, with no reference-reading uncertainty at all) and
**Figs. 10/13** (the semi-axis history against `oscillatingDroplet.
analyticSolution`, which was already encoded).

| R/Δx | linear momentum (ρA₀R³) | angular momentum (ρA₀R⁴) |
|---|---|---|
| 50 | 1.5e-3 | 4.6e-4 |
| 100 | 3.3e-4 | 1.6e-4 |
| 200 | 3.3e-5 | 5.7e-6 |

Case changes: `machTarget`/`referenceVelocity` params (Sun's `c₀ = 15 A₀R`
through the shared Eq. (2) path) and a `_conservationMetrics` diagnostic block
emitting both momenta already in Table 1's normalisation, plus the
mechanical/potential/elastic energy split.

**Partly reproduced: Figs. 11–12's energy budget.** Sun's *total* energy is
constant only because it includes what the artificial viscosity and the
density-diffusion term (`Q_δ`) have dissipated, and neither is recoverable from
the state a diagnostic sees — both are per-step integrals of terms
`schemes/deltaSPH.py` folds into `dvdt`/`drhodt` and does not hand back.
Exposing `dvdt_diss`/`drhodt_diss` on `WeaklyCompressibleSystemUpdate` would
close it, and is the next piece of work on this case. His *mechanical* energy
is scoreable as is: **Fig. 11 (R/Δx = 200) falls to −4.8 % of `E_M⁰` over 14
periods**, so the δ⁺-SPH does lose mechanical energy here — to the artificial
viscosity, as his text says — and the drop's oscillation amplitude decays with
it. That is shared physics, not an error to score against zero.

**Result — 15 periods, `α = 0.01`, Eq. (7) shift: 7/7 checks at all three of
Table 1's resolutions.**

| | R/Δx = 50 | R/Δx = 100 | R/Δx = 200 |
|---|---|---|---|
| linear momentum (ρA₀R³) | 7.51e-4 | 1.26e-4 | 4.88e-5 |
| — Sun Table 1 | 1.5e-3 | 3.3e-4 | 3.3e-5 |
| angular momentum (ρA₀R⁴) | 1.41e-4 | 5.57e-5 | 1.98e-5 |
| — Sun Table 1 | 4.6e-4 | 1.6e-4 | 5.7e-6 |
| `a(t)` RMSE, first 3 periods | 0.0070 R | 0.0045 R | 0.0034 R |
| oscillation period | 4.800 (−0.55 %) | 4.810 (−0.35 %) | 4.816 (−0.22 %) |
| peak amplitude decay, 15 periods | 4.3 % | 2.7 % | 1.7 % |
| mechanical energy loss | −7.6 % | −4.8 % | **−3.2 %** (Fig. 11: −4.8 %) |

Every error converges monotonically, and the semi-axis, period and amplitude
columns are excellent throughout. **The momenta are the interesting column, and
they do not tell a simple story.** Ours are 2–3× *better* than Table 1 at
`R/Δx = 50` and `100`, then 1.5×/3.5× *worse* at `200` — not because ours stop
converging, but because Sun's last refinement does something ours does not:

| refinement | ours, linear | order | Sun, linear | order | ours, angular | order | Sun, angular | order |
|---|---|---|---|---|---|---|---|---|
| 50 → 100 | 5.96× | 2.58 | 4.55× | 2.18 | 2.53× | 1.34 | 2.88× | 1.52 |
| 100 → 200 | 2.58× | 1.37 | **10.0×** | **3.32** | 2.81× | 1.49 | **28.1×** | **4.81** |

Ours converge at a consistent ~1.4 order across both refinements (2.6 for the
first linear step). Sun's Table 1 agrees on the first refinement and then jumps
to order 3.3 and 4.8 on the second — a 28× drop in angular-momentum error for a
2× resolution change. That is very steep for a conservation error, and it is
inconsistent with his own 50 → 100 rates. Not investigated further here, and
**not** asserted to be an error in the paper: the honest statement is that our
momenta converge steadily and land between his 100 and 200 rows, and that the
entire disagreement at `R/Δx = 200` lives in that order jump rather than in our
errors growing.

The mechanical-energy loss now has a same-resolution comparison and comes out
**−3.2 % against Fig. 11's −4.8 %** — i.e. this implementation is ~30 % *less*
dissipative than the paper's at matched `R/Δx`, having been more dissipative at
the coarse end (as `ν = α c₀ h / (2(n+2))`, linear in `h`, predicts).

**Two scoring traps, both hit and both fixed** — recorded because each looked
like a scheme error and was not:
1. **A pointwise `a(t)` RMSE over all 15 periods is the wrong measure.** It
   reads 0.126 R, almost entirely because a 0.55 %-short period accumulates
   ~0.08 of a period of phase by the end; over the first 3 periods the same
   trace reads 0.0070 R. And no figure in the paper constrains the phase at 15
   periods — Fig. 10 stops at 10, Fig. 13 zooms the last oscillation of an
   `R/Δx = 200` run. Period and amplitude are scored directly instead.
2. **The peak detector.** A strictly two-sided `>` comparison silently dropped
   the `t = 20.49` peak, where two consecutive float32 samples tie at the top.
   One missing peak turns a 4.82 interval into a 9.62 one and dragged the mean
   period to 5.17 (**+7.1 %**) — a scheme-error-shaped number produced entirely
   by the analysis. Fixed with a plateau-tolerant comparison, a prominence
   floor (the signal has real small-scale noise near the trough, which a bare
   plateau-tolerant test admits as half-amplitude peaks) and the **median**
   rather than the mean of the intervals.

This is also **the first free-surface case run under the 8× Eq. (7) shift**,
and it is stable and accurate there — the first evidence bearing on whether
`sun2017Eq7Shift` can become the shared default.

---

# Part 6 — sequencing

**The original 7-step plan; steps 1–7 are all done. Live work is now tracked in
"Current state & how to resume" at the top of this file.**

1. ~~**Audit** (Parts 1–3)~~ — done.
2. ~~**`c₀` rework**~~ — Sun Eq. (2) via `machTarget`, wired into the shared
   `setupTimestep` (`3a51f61`), not just `dambreak`.
3. ~~**Integrator**~~ — RK4 + adaptive Δt done. Frozen diffusion is the
   `sun2017DeltaSPH` default only; no general path (perf only, open item 10).
4. ~~**Revert `# PSI-REVERT`**~~ — done; `tests/test_deltaSPHDiffusion.py` green.
5. ~~**mDBC**~~ — determinant gate + per-kernel threshold landed; English §4.1
   hydrostatic-to-the-wall validated (fix A, wedge 9/9 at dp = 0.01).
6. ~~**Validation — Marrone §3.1**~~ — P1 converges to Buchner at H/Δx = 322;
   P2 + wall-penetration + DualSPHysics cross-check are open items 3/4/8.
7. **δ⁺-SPH suite (§5.3)** — TGV and droplet done; `rotatingSquarePatch` (the
   PST discriminator) is open item 1; the bluff-body wakes (N°4–7) are unstarted.

## Relationship to the other plans

`ACSPH_PLAN.md` decision 1 currently reads *"this is not a reason to revert the
[ψ] fix"* — this plan agrees and supersedes the ad-hoc `# PSI-REVERT` made while
chasing the Lobovský dam break. `PST_ALE_PLAN.md` / `WCSPH_SHIFTING_PLAN.md` own
the δ⁺ shifting / δ-ALE work; Part 2 here is an audit of what they landed, not a
re-do. `DFSPH_IMPROVEMENT_PLAN.md` owns the incompressible wall closure; Part 3's
`computeMdbcNoPenShift` A/B overlaps its `mdbcNoPenetrationShift` item.

---

## Item 5 — diffSPH parity pass on `sloshingTank`  (2026-09-11)

Comparing this repo's δ⁺-SPH against `~/dev/diffSPH`'s (the notebook
`examples/weaklyCompressible/16_SloshingTank.ipynb`, which runs its sloshing
tank cleanly). Step 1 was to remove the *setup* differences so any remaining
gap is attributable to the solver.

### 5.1 Matched the reference discretisation (done)

| | diffSPH notebook | warpSPH before | warpSPH now |
|---|---|---|---|
| `nx` | 200 (dx = 0.0045) | 150 | **200** |
| `dt` | 1e-4, fixed | back-solved, 2e-4 | **1e-4, fixed** |
| `c_0` | 20, literal | back-solved from `dt` (11 at nx=225) | **20, literal** |
| integrator | RK2 | RK2 | **`symplecticEuler`** (13, 2-stage KDK — see 5.3) |

`caseUtils/weaklyCompressible.py:setupTimestep` grew a third route: with
`soundSpeed` set, **both** `c_0` and `dt` are pinned to what the case asked for,
no back-solve in either direction (the `machTarget` route and the legacy
back-solve are untouched, so every other case is unchanged). `dt` is reported
against the acoustic-CFL step and warned about rather than silently clamped, so
an over-long step cannot quietly invalidate an A/B.

### 5.2 The mDBC body-velocity path (done)

Two independent holes, both fixed; **numerically a no-op for every case in the
tree today**, verified by `tests/test_physics.py` (72/72 green):

1. `rigidBody/update.py` only wrote the body's velocity into
   `particleState.velocities` when `kind == BCType.constant`. A moving
   `freeSlip` / `noSlip` / `extended` wall therefore had no velocity anywhere in
   the state. How the wall's motion is imposed on the fluid is the BC's
   business; how fast the wall moves is not a boundary condition. Now written
   for every `BCType`. (`drivenSquare` and `movingObstacle` are the only cases
   with rigid bodies and both are `constant`, hence the no-op.)
2. `modules/mdbc/velocity.py` read its `u_body` at the **ghost** rows, where
   nothing writes it, so even a `constant` body degenerated to a stationary
   wall inside the BC. Both slip conditions are now written on the *relative*
   velocity `w = Shepard(u_f) - u_body` with the wall's velocity added back:
   `noSlip -> u_body - w_t`, `freeSlip -> u_body + w_t`. The boundary
   particle's normal component is now `u_body . n` instead of pinned to 0.

Why it matters beyond advection: a wall moving into the fluid has to enter
`div(v)` with its own normal velocity, or the density change the wall drives is
missing while gravity's forcing is not, and the two diverge. diffSPH reaches the
same place by a different route — it restores `boundaryBodyVelocities` before
`computeMomentum` (`schemes/deltaSPH.py:169`) rather than folding the body
velocity into the BC. At `u_body = 0` both new forms reduce **exactly** to what
this module computed before, so the measured "projects the fluid's normal
component out rather than reflecting it" deviation (DFSPH Part 9 addendum) is
untouched and still recorded.

### 5.3 `semiImplicitEuler` (21) integrates density explicitly — use `symplecticEuler` (13)

**There are two Euler-family entries and they are not the same scheme.**
`IntegrationSchemeType.symplecticEuler` (13, `verlet.py:66`) is the **2-stage
kick-drift-kick** scheme DualSPHysics runs; `IntegrationSchemeType.
semiImplicitEuler` (21, `util.py:66`) is a **1-stage** scheme. Only the latter
has the problem below. `symplecticEuler` advances both `v` and `rho` by
`dt * k1`, where `k1` is evaluated on the half state `state^n + (dt/2) k0` --
i.e. explicit midpoint on the `(rho, v)` pair, so the density rate *is*
staggered against velocity and the acoustic amplification is the benign RK2
one. That is the scheme this case should use.

#### The `semiImplicitEuler` (21) failure

**`semiImplicitEuler` (21) + the linear density update is unconditionally
unstable for the WCSPH acoustic mode.** `sloshingTank` at the matched config above
diverges at **t = 0.041 s (step 410)**, at rest, before any meaningful roll
(0.018 deg), with `voidFraction` 0.44 and `maxDensity` 7.7e33.

Mechanism, `warpSPHIntegrators/util.py:updateStateSemiImplicitEuler`:

```python
applyVelocityUpdate(..., explicit_step(dt), semiImplicit=True)   # v from state^n
applyPositionUpdate(..., semi_implicit_position_step(dt))        # x from v^{n+1}  <- staggered
applyQuantityUpdate(..., explicit_step(dt))                      # rho from state^n <- NOT staggered
```

`update_component` (`fields.py:660`) is a plain `rho^n + dt * drhodt^n`; there is
no hook to re-evaluate a quantity's rate against the updated velocity. So
position is correctly staggered against velocity — but for WCSPH the stiff
oscillator is **(rho, v)**, not (x, v): `drho/dt = -rho div(v)` and
`dv/dt = -grad(p(rho))/rho`. Leaving rho on the old velocity makes the acoustic
pair plain **explicit Euler**, whose amplification factor is
`|A| = sqrt(1 + (c k dt)^2) > 1` at every `dt`.

Measured per-step amplification of that mode
(`scratchpad/` oscillator check, `|A|` and `|A|^400`):

| `omega*dt` | `semiImplicitEuler` (21) | `symplecticEuler` (13) | `rungeKutta2` (1) | `rungeKutta4` (9) | Padé(1,1)/`exp` map |
|---|---|---|---|---|---|
| 0.05 | 1.00125 -> **1.65** | 1.0000008 -> 1.00 | 1.0000008 -> 1.00 | 1.0 -> 1.0 | 1.0 -> 1.0 |
| 0.2 | 1.01980 -> **2551** | 1.0002 -> 1.08 | 1.0002 -> 1.08 | 0.9999996 -> 1.00 | 1.0 -> 1.0 |
| 0.5 | 1.11803 -> **2.4e19** | 1.0078 -> 22.2 | 1.0078 -> 22.2 | 0.99989 -> 0.96 | 1.0 -> 1.0 |

`symplecticEuler` and `rungeKutta2` coincide because both reduce to explicit
midpoint on the linearised pair. Reproduce: `scratchpad/acoustic_amplification.py`.

This also explains, retroactively, why the `exp` / Padé density override was
load-bearing and why removing it (item C) was safe **only** under RK: the
Padé(1,1) map `(2-eps)/(2+eps)` is the Cayley transform, which has modulus
**exactly 1** for an imaginary `eps` — it is neutrally stable on the acoustic
mode by construction. `exp(-eps)` likewise. The linear map `1 - eps` is not.
Item C's reasoning (single-evaluation, 1st-order-vs-RK-order, `exp` rectifies an
oscillatory `drhodt` into net density gain) stands for the **RK** case, which is
what it was measured on; it does not transfer to a single-stage integrator.

**Bisect — the integrator is the sole cause.** Equal physical time (t = 0.15),
`scratchpad/slosh_bisect.py`:

| config | `rungeKutta2` | `semiImplicitEuler` |
|---|---|---|
| nx 150, dt 2e-4, back-solved c_s (**the old shipped default**) | clean | **diverged, step 251** |
| nx 200, dt 2e-4, back-solved c_s | clean | **diverged, step 253** |
| nx 200, dt 1e-4, c_s 20 (**matched, 5.1**) | clean | **diverged, step 440** |

So 5.1's discretisation change is exonerated: the *original* configuration also
diverges under `semiImplicitEuler`, and every configuration runs under RK2.
`symplecticEuler` (13) was **not** in this first matrix -- that was the error;
it is covered in the follow-up run (`scratchpad/slosh_bisect2.py`).

**Resolution:** the case uses `symplecticEuler` (13). No solver change needed --
the 2-stage form already staggers the density rate. `semiImplicitEuler` (21)
remains unsuitable for any weakly-compressible scheme in this tree and should
carry a warning at selection time (open, small); the only ways to make *it* work
would be to stagger the continuity term explicitly (re-evaluate `-rho div(v)` at
`v^{n+1}` in `finalize`, one extra divergence pass, and `drhodt_diss` would have
to be split out of `drhodt` so the stabiliser is not also re-evaluated), or to
restore the Padé/Cayley density map gated on single-stage integrators. Neither
is needed for this case.

`cases/sloshingTank.py` ships `integrationScheme='symplecticEuler'`;
`--integrationScheme` (a new `run_sloshingTank.py` flag) selects any other for
the A/B.

### 5.4 Ghost sampling at nx=200 — clean

`scripts/dump_sloshing_sampling.py --nx 200` (`scratchpad/sloshing_sampling_nx200/`):
3230/3230 boundary particles carry a non-zero ghost offset (1.00–12.36 dx,
median 5.00), **0** ghost nodes inside the wall solid. Flat walls mirror
correctly through all 5 layers. The only artifact is the already-tracked
`_gridSnapGhostOffsets` **concave-corner collapse** (the "NEXT" item above): at
each tank corner all 5 layers' nodes fan onto the apex. It was equally present
in the nx=225 baseline that ran clean, so it is not what 5.3 is about.

### 5.5 The fluid/boundary density step in the field video — **not** mDBC

Reported from the density video: a clear jump from the fluid to the boundary
rows, present from the start; visually "as if the extrapolation uses half the
distance, i.e. boundary-to-surface instead of boundary-to-ghost".

**Ruled out — the mDBC operator is exact.** `scratchpad/probe_mdbcDistance.py`
freezes the real `sloshingTank` configuration, overwrites the fluid density with
an exact linear field `rho = 1 + a*y`, runs `computeMdbcDensity` once, and grades
the boundary result against the analytic field. English Eq. (12) is exact on a
linear field, so any residual is the operator's own:

| boundary row `y/dx` | `\|r_b - r_g\|/dx` | `rho` got | `rho` want | err | implied `lambda` |
|---|---|---|---|---|---|
| -0.50 | 1.00 | 1.000114 | 1.000113 | +2e-9 | 1.001 |
| -1.50 | 3.00 | 1.000338 | 1.000338 | -3e-10 | 1.000 |
| -2.50 | 5.00 | 1.000563 | 1.000563 | -2e-9 | 1.000 |
| -3.50 | 7.00 | 1.000787 | 1.000788 | -4e-10 | 1.000 |
| -4.50 | 9.00 | 1.001012 | 1.001013 | -5e-10 | 1.000 |

`|r_b - r_g|/dx` is 1, 3, 5, 7, 9 — exactly `2 x depth`, the **full** mirror, not
the boundary-to-surface half; and `lambda = 1.000` at every layer, so the
extrapolation travels exactly `(r_b - r_g)` with English's sign.
`scripts/probe_mdbcExtrapolationSign.py` independently confirms the sign
(`interpolateLiuLiu` returns `+grad`; the assembled path tracks `+ghostOffset`,
mean|Δ| 6.0e-2 against 6.9e-1 for the flipped form).

**The actual cause is upstream — the fluid's own near-wall density profile.**
`scratchpad/probe_sloshWallDensity.py`, mid-span flat floor, 3000 steps at rest
(nx=200, c_s=20, hydrostatic `d(rho)/dx = rho0 g dx / c^2 = 1.104e-4`):

| fluid rows `y/dx` | measured `d(rho)/dx` | x hydrostatic |
|---|---|---|
| 0.5 -> 1.5 | 1.80e-05 | **0.16x** |
| 1.5 -> 2.5 | 1.22e-04 | 1.11x |
| 2.5 -> 3.5 | 2.34e-04 | **2.12x** |
| 3.5 -> 4.5 | 1.71e-04 | 1.55x |
| 4.5 -> 5.5 | 1.07e-04 | 0.97x |
| 5.5 -> 6.5 | 1.02e-04 | 0.92x |

The profile is only hydrostatic from row ~5 outward; the first four rows are
badly non-linear. The mDBC ghost node for wall layer `k` sits at `+k dx` —
inside that anomalous zone — so the MLS measures the locally-wrong gradient and
continues it linearly over `2 phi`, roughly doubling the error by the time it
reaches the wall particle. Hence a wall band whose slope is ~2.1x the fluid's
(measured), and a visible step. The interface step itself is small in the static
state (+5.3e-5, 2 % of the hydrostatic span); what the video shows is the band,
plus the dry/near-surface wall reading **exactly** `rho0` (the rest-density
fallback where the ghost node has no fluid) against a fluid at ~1.003.

**Open — what makes the near-wall fluid profile non-hydrostatic.**

*Not the DDT pair direction.* The obvious suspect was item C's uncommitted
`OperationDirection.FluidToFluid` restriction on the DDT outer sum (near a wall
roughly half a particle's stencil *is* boundary, so excluding it removes the
diffusion where the lattice error is largest). A/B refutes it, and inverts it
(`scratchpad/probe_nearWallDDT.py`, 3000 steps, same readout):

| fluid rows `y/dx` | `FluidToFluid` (current) | `AllToAll` |
|---|---|---|
| 0.5 -> 1.5 | 0.17x | **0.99x** |
| 1.5 -> 2.5 | 1.10x | 1.72x |
| 2.5 -> 3.5 | **2.12x** | **2.62x** |
| 3.5 -> 4.5 | 1.55x | 2.07x |
| 4.5 -> 5.5 | 0.97x | 1.70x |
| 5.5 -> 6.5 | 0.93x | 1.62x |
| 7.5 -> 8.5 | 1.73x | 1.28x |
| interface step | **+5.3e-5** | +1.13e-4 |

`AllToAll` fixes only the *first* gap and makes every row from 1.5 outward
worse, doubles the interface step, and is still at 1.28x out at row 8 where
`FluidToFluid` has converged by row 5. So `FluidToFluid` stays (item C's change
is not implicated) and the rows 2.5-4.5 anomaly, present in **both**, is
something else.

*Cause: the delta+ particle shift.* `scratchpad/probe_nearWallShift.py`,
3000 steps, same readout (`d(rho)/dx` as a multiple of hydrostatic, per fluid
row gap). All three legs are **pinned** to `symplecticEuler` in the script
(`INTEGRATOR`, not the case default -- re-run pinned reproduced the original
numbers to every printed digit, confirming the first run was already on it).
Comparable to each other, **not** to the DDT / wall-profile tables above, which
ran under `rungeKutta2`; the anomaly is qualitatively the same in both, so it is
not integrator-specific:

| row gap start `y/dx` | 0.5 | 1.5 | 2.5 | 3.5 | 4.5 | 5.5 | 6.5 | 7.5 | interface step |
|---|---|---|---|---|---|---|---|---|---|
| `michel2022` (default) | 1.39 | 1.60 | **2.33** | **2.11** | 1.15 | 1.07 | 1.46 | 1.80 | +1.19e-4 |
| **shift OFF** | 0.61 | 0.69 | **0.91** | **0.85** | 0.95 | 1.02 | 1.41 | 1.75 | **+4.8e-5** |
| shift + `correctdrhodt` | 0.72 | 1.36 | 2.28 | 2.00 | 1.29 | 1.25 | 1.35 | 1.51 | +7.0e-5 |

With the shift off, rows 2.5-5.5 read 0.91 / 0.85 / 0.95 / 1.02 — essentially
hydrostatic — and the interface step more than halves. With `michel2022` on they
read 2.33 / 2.11. **The near-wall density anomaly is the particle shift**, not
the DDT and not mDBC. Sun 2019 Eq. (9)'s volume-consistency term
(`correctdrhodt`) recovers part of it (first row 1.39 -> 0.72, step 1.19e-4 ->
7.0e-5) but leaves rows 2.5/3.5 at 2.28 / 2.00, so it is not the whole story.

Not actionable as "turn the shift off": `cases/sloshingTank.py` documents why
`michel2022` is the shipped default (the `deltaSPH`/`surfaceNormal` shift
diverges at t=0.68 post-`790a7c7`). The open question is why *this* shift
formulation drives a one-sided density error next to a wall, and whether the
free-surface/wall projection is the part at fault.

### 5.6 Three plotting bugs behind the "density field looks wrong" report

None of them are solver bugs; together they made the wetted wall look
discontinuous from the fluid when the data is continuous. All fixed.

1. **Independent normalisation, shared colormap, one colorbar.** The vispy and
   pyvista backends called `getBounds` separately on the fluid and on the
   boundary set, drew both through the same colormap, and labelled the single
   colorbar with the *fluid's* range only. A dry mDBC wall particle at exactly
   `rho0` is the minimum of the boundary set, so it saturated one end of the map
   against a fluid whose own minimum is a different number — a hard edge at the
   waterline that is not in the data. (`visualize.py`/`update.py`, the
   matplotlib path, already shared a norm.) Fixed: new
   `warpSPHPlotting.math.getSharedBounds`, used by both backends' create and
   update paths, and the colorbar now reports that shared norm.
2. **`midPoint` silently ignored.** `getBounds` only applies it on the
   `Symmetric`/`SymmetricLog` branch. Every case pairing `midPoint=1.0` with the
   default `scaling='Linear'` got a plain min-max stretch wearing a diverging
   colormap, so "white" landed at the data minimum rather than at `rho0`.
   `cases/sloshingTank.py`'s density `Field` now passes `scaling='Symmetric'`.
   **Still open for 7 other cases** that set `midPoint` with default scaling
   (`tgv`, `kolmogorovIncompressible`, `shearWave`, `staticBlob`, `dambreak`,
   `impact`, `weaklyCompressible`) — a presentation change, left alone.
3. **`CenteredNorm(halfrange=maxScale)`** — `halfrange` is a half-*width*, but
   `maxScale` is `midPoint + max|q - midPoint|`. Correct only for the default
   `midPoint == 0`, which is why it never surfaced; at `midPoint = 1.0` it made
   the halfrange 1.003 instead of 0.003, ~340x too wide, collapsing every value
   onto the colormap midpoint. Fixed to `maxScale - midPoint`. No case in the
   tree used `Symmetric` before, so this was a no-op on existing output.

Also widened the vispy colorbar labels from `.3g`/`.4g` to `.6g`: at three
significant figures both ends of `[0.997, 1.003]` render as "1".

Verified end-to-end on `sloshingTank` frame 50: wall band now sits at white
(`rho0`), wetted floor is continuous red with the fluid, colorbar reads
`0.994236 .. 1.00576`.

### 5.7 Marrone 3.1 wall boundary condition — the bed was effectively no-slip

Chasing a reported "the front interacts violently with the far wall way too
early" on `probe_deltaSPHMarrone.py --nx 67 --c0Ratio 40`, visible in every
scheme/integrator combination.

**What the frames actually show** (RK4, no PST, the *stable* baseline):

| t* | state |
|---|---|
| 1.26 | coherent tongue, clean tip |
| 2.51 | front at ~92 % of tank; last ~30 % of the tongue fragmented into singles |
| 2.73 | `maxVelocity` spikes to 17.4 m/s = **3.7 x U_max** |
| 3.14 | run-up jet climbing the far wall **with a gap behind it** -- bulk not arrived |
| 3.18 | P1 peaks at **P\* = 27.4** (Marrone Fig. 5: ~2-3) |
| 3.77 | detached cloud of fluid hanging in mid-air near the wall |

So it is not an invisible wall: the *tip* sheds fliers 7-8 dx ahead of the bulk
front while the tongue is only ~3 dx thick, and those arrive alone, slam, and
eject the airborne cloud. The RK4 baseline is **stable but not correct** -- the
density range and `nPenetrating` columns it was being graded on are blind to a
wall probe reading 10x the reference.

**Root cause found: the tank walls are `BCType.constant`.**
`caseUtils/weaklyCompressible.py:buildRegions` hardcoded it. `constantVelocity()`
returns the boundary velocities unchanged and nothing writes them (no rigid
body), so the wall sits at `v = 0` while `computeVelocityDiffusion` runs
`AllToAll` -- the artificial-viscosity term therefore drags the fluid against a
stationary bed. That is an effective **no-slip** wall; Marrone 2011 Sec. 3
specifies **free slip**. (`probe_deltaSPHMarrone.py`'s own docstring claimed the
free-slip spec was met "without a slip-mode knob" because no *physical*
viscosity term is added -- true, and beside the point: the artificial viscosity
is the one that is on.) Fixed by a new `wallBC` param on `dambreak`, default
`'constant'` so no recorded run changes.

**Free-slip A/B** (`scratchpad/probe_m31_front.py`, H/dx = 40, RK4, no PST):

| t* | v_front `constant` | v_front `freeSlip` | thick `constant` | thick `freeSlip` | tip lead `constant` | tip lead `freeSlip` |
|---|---|---|---|---|---|---|
| 1.60 | 3.44 | **3.88** | 5.4 dx | 4.6 dx | 6.4 dx | 7.6 dx |
| 2.00 | 3.41 | **4.13** | 3.9 dx | 3.5 dx | 7.7 dx | 9.2 dx |
| 2.20 | 3.43 | **4.14** | 3.5 dx | 3.0 dx | 7.4 dx | 7.8 dx |
| 2.40 | 3.37 | **4.04** | 2.8 dx | 3.5 dx | 7.5 dx | 6.1 dx |
| `pb@front` t*=2.6 / 2.8 | **-0.0137 / -0.0177** | +0.0007 / -0.0004 | | | | |

- **Bed drag was the late-arrival cause.** Free slip takes the front from 3.4 to
  **4.13 m/s** (Ritter 2 sqrt(gH) = 4.85, Marrone U_max = 4.73) and moves impact
  from t* ~ 2.8 to ~ 2.6, toward the reference 2.3.
- **Bed drag was the suction cause.** `pb@front` goes *negative* (-0.014 /
  -0.018) under the no-slip bed exactly as the front passes -- an attracting
  wall, which is what pulls a particle into the boundary band and poisons its
  neighbourhood. Free slip removes it.
- **Bed drag is NOT the fragmentation cause.** Tongue thickness (3.0-4.6 dx) and
  tip lead (6-9 dx) are the same either way. **Still open**; remaining
  candidates are under-resolution at H/dx = 40, the hydrostatic term missing
  from the mDBC Shepard fallback (5.5), and tensile instability in the sheet.

**Two corrections to earlier readings in this section, both from measurement
error, both caught by re-measuring:**

1. A first pass reported the front as "30 % slow" against Ritter's 4.85 m/s.
   Ritter is the ideal inviscid dry-bed front, not what this case should
   produce: the reference's own t* = 2.3 over the 2.004 m dam-to-wall distance
   implies ~3.5 m/s. The terminal speed was never the problem; the acceleration
   phase was.
2. The first free-slip A/B reported the tongue thickening to 10-14 dx and the
   tip lead halving, and the front speed *not* moving -- the opposite of the
   table above on both counts. That run went through the `_ghostBodyVelocity`
   compounding bug below, i.e. a wall ~3x stiffer than intended. Withdrawn.

### 5.8 mDBC slip conditions — `freeSlip` now matches the published form

`modules/mdbc/velocity.py`. Graded by
`scripts/probe_boundaryVelocityModes.py --mode verify` as least-squares slopes
of the boundary velocity against the fluid velocity at the ghost point,
decomposed on the wall normal:

| BCType | before | after | published |
|---|---|---|---|
| `noSlip` | (0, -1) | (0, -1) | (-1, -1) |
| `freeSlip` | (0, +1) | **(-1, +1)** | (-1, +1) |

`freeSlip` now **reflects** the fluid's normal component instead of projecting
it out: `u_g = u_body + w_t - w_n`, `w = Shepard(u_f) - u_body`. `noSlip` is
deliberately left projecting -- `DFSPH_IMPROVEMENT_PLAN.md` Part 9's addendum
measured the reflecting form as *worse* on the bounded DFSPH case, so that one
is revisited with that case in hand, not blind.

**Bug found by the grader, introduced in 5.2 and live for several runs:
`u_body` must come from a rigid body, not from the boundary row.** 5.2 changed
`_ghostBodyVelocity` to read the *boundary* rows (correct for a real moving
body, and the point of that fix). But for a wall with no rigid body those rows
hold **the previous step's BC output**, so the condition compounded on itself:
`u_g . n = 2(u_body . n) - f . n` with `u_body . n` already `= -f . n` gives
`-3 f . n`, and the grader measured exactly **-3.0063**. The term had always
been dead before (it read the ghost rows, which are identically zero), which is
why the compounding had never surfaced. Now `_ghostBodyVelocity` takes a
non-zero value **only** for particles owned by a `RigidBody`, zero elsewhere.
Re-graded: `freeSlip` (-1.0000, +1.0000), `noSlip` (0, -1.0000) unchanged,
`test_physics` green, and the `constant` front-probe leg reproduces
bit-for-bit (that path never calls `_ghostBodyVelocity`, which is why 5.3's
integrator comparison is unaffected).

Lesson for the audit: "no existing case is affected" was argued in 5.2 from
*which rigid bodies exist*, and never checked what non-rigid-body walls had
sitting in those rows. The unit grader caught in one run what 72 passing
physics tests did not.

### 5.9 mDBC no-penetration shift — two fixes, and a characterisation harness

`symplecticEuler` (enum 13) diverged on **both** active cases -- sloshingTank at
t = 0.727 and Marrone 3.1 at t* = 1.295 with **473** penetrating particles
against RK4's zero -- while RK2/RK4 ran clean. Ablating
`computeMdbcNoPenShift` made sloshingTank clean immediately
(rho [0.9972, 1.009] vs a 1e30 blow-up), which localised it to that term.

**Fix 1 -- stage `dt` -> `config.dt`.** `deltaSPH_step` had
`dvdt_nopenshift = nopenshift / dt`, where `dt` is the **stage** dt the
derivative function was called with. `nopenshift` is a *velocity* correction,
so `/dt` makes an acceleration the integrator multiplies by the step length
again; with a full step those cancel and the particle gets exactly the intended
correction. `symplecticEuler` calls the derivative with `dt/2` but updates with
`dt`, so the correction landed at **2x** in both stages (RK4 is wrong by a
different factor; a single-evaluation scheme is right by accident). Now divides
by `config.dt`, the real step length, which is already in scope.

**Fix 2 -- `wp.abs(normDist)` -> signed `normDist`.** DualSPHysics
(`JSphGpu_ker.cu`) gates on `normdist < 0.75f*norm`, a **signed** test that
stays true however deep a particle has penetrated. `wp.abs()` turned it into a
band, so a particle more than 0.75*norm *past* the first boundary row got
**zero** correction -- the anti-penetration term switched off exactly where
penetration was worst.

**Characterisation harness: `scripts/probe_nopenShiftResponse.py`.** A single
fluid particle swept over normal distance (+2 dx to -2 dx) x tangential phase
(-dx/2 to +dx/2) against a synthetic flat wall, at three velocity directions,
all in one kernel launch; output as images plus a 1-D normal profile. What the
term *actually does*, rather than what the branch structure suggests:

- a **hard on/off band**, not a graded ramp: constant `+2` over
  `n in (-1.17 dx, +0.2 dx]` before Fix 2, extending past `-1.5 dx` after;
- exactly **zero** for tangential or outward motion (the `vfc < 0` gate is
  correct);
- **no tangential-phase dependence** (no lattice artefact);
- magnitude `+2` for an approach velocity of `-1`, i.e. an elastic reflection --
  which is what DualSPHysics' `v_new = v_pre + nopenshift` produces, so with
  Fix 1 the magnitude now matches the reference.

**Verification.** `test_physics` green. sloshingTank t = 1.5: symplecticEuler
rho [0.9972, 1.009] vmax 0.5904, rungeKutta2 rho [0.9972, 1.009] vmax 0.5917 --
the two integrators now agree to 3 digits where one previously blew up.
Marrone 3.1 delta+ symplecticEuler: **full t* = 5.014, rho [0.835, 1.20],
vmax 29.3, nPen_max 1** (was: disintegrating, vmax 4959, nPen 473).

**Two audit claims retracted** -- both were read off the source and both were
wrong; the synthetic sweep is what settled them:

1. *"the counter is discarded, so the correction is never averaged"* -- false.
   The averaging is in the launch kernel
   (`avg_out[d] = ret_out[d] / ret_ctr[d]`) before the output is written;
   `return nopenshift[0]` returns the already-averaged value and the counter is
   a diagnostic. Measured: 3 contributing neighbours at `+2` each, output `+2`.
2. *"`ratio` is clamped at 1.0 where DualSPHysics has no upper bound"* -- moot.
   `ratio = |dr/norm|` divides a length by a dimensionless unit-vector
   component, so it is ~5e-3 and always hits the `0.25` floor; `factor` is the
   constant `2` in practice. **DualSPHysics has the identical expression**, so
   the distance-dependent ramp is dead in both -- a shared defect, not a
   deviation, and worth revisiting on its own.

**Still deviating from the reference, recorded not fixed:** the term is applied
as an acceleration summed into `dvdt` rather than DualSPHysics' post-integration
velocity *replacement* (`v_new = v_pre + nopenshift`, displacement recomputed),
and it is unconditional where DualSPHysics gates it on
`SlipMode >= SLIP_NoSlip` and makes it opt-in -- i.e. never applies it under
free slip, which is what Marrone 2011 Sec. 3 specifies.

**Consequence for earlier results in this document:** the RK4 Marrone baselines
in 5.3 and the free-slip A/B in 5.7 were all measured with the mis-scaled term,
so they shift under these fixes and need re-running before being compared
against anything new.

### 5.10 Normal-vector reflection — correct operator, exposes a pre-existing bug

The no-penetration correction was decomposed per Cartesian component
(`-factor * dv_d * n_d^2`), which is the right magnitude only when the wall
normal **is** an axis. Three synthetic harnesses were built to characterise it
rather than trace its branches, and each one killed a conclusion previously
reached by reading the source:

* `scripts/probe_nopenShiftResponse.py` -- one particle swept over normal
  distance x tangential phase against a flat wall, three velocity directions.
  Showed a hard on/off band, constant `+2` reflection, zero for tangential or
  outward motion, no lattice dependence.
* `scripts/probe_nopenShiftTrajectory.py` -- integrates a single ejected
  particle. Measured restitution ~0.70 on one firing, **not** the perfectly
  elastic bounce assumed.
* `scripts/probe_nopenShiftCorner.py` -- an orthogonal corner with the radial
  ghost-normal fan. Per-component form: fires at 25/1681 points, and where it
  fires the particle is **stopped dead** (`|v_new| = 0.000`).

An angle sweep pinned the defect: `shift_d = 2 v_d n_d^2`. At 45 deg each
component receives exactly `|v_d|`, which **cancels** the velocity instead of
reversing it.

**Change:** reflect along the wall normal as a vector, and make `ratio`
dimensionless (`|normDist| / norm_j`, where the old `|dr/norm|` divided metres
by a unit-vector component and so was pinned to the 0.25 floor -- the distance
ramp DualSPHysics intends was dead in both codes). `factor` now runs 0 (grazing,
smooth onset) -> 1 (absorb) -> 2 (deep, full specular).

Graded: flat wall unchanged (939 pts, tangential/outward exactly 0); tilted wall
angle-independent; corner `|v_new|` min 0.000 -> 0.998.

**Results (symplecticEuler + PST + `finalize` + freeSlip):**

| case | outcome |
|---|---|
| Marrone 3.1, t* 7.5 | ran clean, nPen 0, **P1\* 45.6 -> 7.7** (Fig. 5 ref ~2-3) |
| sloshingTank, t 4.5 | 45000 steps, rho [0.862, 1.250] |
| **Marrone 3.4, t\* 6** | **all four gates FAIL** -- rho to 46.5 (pointwise 2.4e8), max\|v\| 296897 U_max, 3.5e7 dx outside the AABB |

**Diagnosis of the 3.4 failure (from the field video, user):** particles are
**sucked into the wedge**, then massively accelerated through the boundary, then
fly out and destroy the fluid. It localises to the wedge toe -- the re-entrant
corner where the 45 deg face meets the floor. **The explosion is at t* ~ 0.9**,
so reproduction is cheap; no need to run the full record.

The causal chain:

1. **The wall attracts.** The uncommitted `density2025.py` change moved the
   `rho_b >= rho0` clamp to the *fallback share only*, so on the MLS path
   `rho_b` is unbounded below and `p_b` can go arbitrarily negative. Measured
   independently in 5.7: `pb@front` = -0.014 / -0.018 under the dam-break front.
2. **The per-component form was masking it.** Its 45 deg cancellation stopped
   *any* particle reaching the wedge face dead -- not a reflection, an
   absorption, and on a 45 deg geometry a hard backstop against penetration.
3. **The vector reflection removes the backstop** (correctly -- it returns the
   particle's speed instead of eating it), so attracted particles now reach the
   solid.
4. Inside the solid the stencil is one-sided and the density nonsense, hence the
   1e6 velocities and the ejected particles tearing up the fluid.

**So the reflection did not introduce a bug, it exposed one**, and what it
removed was a bug acting as a safety net. Reverting it would restore the mask,
not the fix. The fix is upstream: bound `rho_b` from below on the MLS path
without reintroducing the sloshingTank density ratchet that motivated removing
the blanket clamp (5.5) -- a cavitation-style floor, or clamping only where the
particle is being drawn inward. `probe_mdbcDistance.py` + the corner/response
sweeps can grade a candidate before it touches a case.

**Confound, not yet controlled:** the 3.4 run changed the reflection form *and*
the duration (t* 4 -> 6) together; the passing per-component run only reached
t* = 4. The decisive control is normal-vector at t* = 4 (or just past the
t* ~ 0.9 explosion). Until it runs, the normal-vector change is **unproven, not
validated**.

### 5.11 The velocity overshoot is thin-sheet collapse, not wall impact

On Marrone 3.1 (normal-vector, t* 7.5) `maxVelocity` peaks at **40.80 m/s =
8.6 U_max at t = 1.3582 s (t* = 5.492)** -- and the wall probe reads
**P1\* = 0.28** at that instant. An impact would spike both. It does not:

```
t = 1.3364   vmax =  4.17   rho_min = 0.9929
t = 1.3505   vmax = 18.73   rho_min = 0.9472
t = 1.3582   vmax = 40.80   rho_min = 0.9114   <- peak, P1* = 0.28
t = 1.3786   vmax = 11.04   rho_min = 0.9859
t = 1.3927   vmax =  4.47   rho_min = 0.9939
```

4 -> 40 -> 4 m/s in ~55 ms with `rho_min` dipping to 0.91 in lockstep. Observed
in the field video (user): when the fluid detaches from the top wall a thin
sheet stays attached, then is accelerated down hard once it is only a few
particles thick -- it loses kernel support, its density falls, and the remaining
particles are yanked.

**This reframes the metrics.** `vmax 40.8` and the widened `rho` range on the
normal-vector 3.1 run were reported as a possible regression; they are a single
free-surface event the longer run exposed, not wall behaviour. Like-for-like the
wall metrics improved unambiguously (P1\* 45.6 -> 7.7, nPen 0).

It also unifies the remaining open items: this, the 5.7 tongue fragmentation
(tongue thinning to 2.8 dx, fliers 7-8 dx ahead of the front) and 5.5's fallback
returning exactly `rho0` under thin fluid are **one problem -- SPH with too few
neighbours**, not SPH against a wall. Next instrument: a thin-sheet probe
(N-particle sheet, acceleration vs thickness), with a concrete target to
reproduce: `rho_min ~ 0.91`, `8.6 U_max`, onset over ~55 ms.

### 5.12 Gravity and the `finalize` velocity replacement -- orientation matters

A particle was observed flying **parallel to the ceiling without dropping**.
`finalize` applies `vNew = where(active, vPre + shift, vFree)` with `vPre` the
*start-of-step* velocity, so every step the correction fires, that component
loses the step's gravity.

**That discard is correct for a floor and wrong for a ceiling** (user), with
`n_hat` pointing into the fluid:

| wall | `n_hat` | `n_hat . g` | gravity | discard? |
|---|---|---|---|---|
| floor | (0,+1) | **< 0** | presses *into* the wall | **yes** -- else the particle accumulates downward velocity every step and sinks through |
| ceiling | (0,-1) | **> 0** | pulls *away* from the wall | **no** -- discarding is what makes it hover |

So the rule is `n_hat . g < 0` -> keep discarding; `> 0` -> let gravity act. A
blanket "use `vFree` instead of `vPre`" would fix the ceiling and start dropping
particles through floors. The grazing probe must cover **both orientations**
before any fix is accepted.

### 5.13 Marrone 3.4 fixed; sloshingTank root-caused to the no-pen mode  (2026-09-12)

Two independent boundary defects, found by instrumenting rather than inferring.
Marrone 3.4 is now **6/6 over the full record**; sloshingTank's divergence is
root-caused and has a one-flag fix. Four plausible-looking scheme changes were
tried and **rejected on measurement** -- recorded below so they are not retried.

#### (a) mDBC ghost placement -- `'hybrid'`, and what M34's blow-up actually was

5.10's account of the 3.4 failure (wall attracts -> per-component reflection was
masking it -> vector reflection exposed it) is **superseded**. The real cause was
ghost *placement*, and 5.10's own severity claim was overstated in one respect:
gridsnap never places a node inside the solid. Measured with a sign test on the
merged SDF at nx = 256, every node it *places* clears the solid (min sdf
+0.0038); the 1187 particles at issue are **declined** -- the retraction loop
exhausts, the offset collapses to zero, and the ghost then coincides with its own
boundary particle (which is inside the solid band by construction, which is what
an earlier "637 ghosts inside the solid" count was really seeing). So the defect
is lost *coverage*, not corrupted data: those particles fall back to Shepard /
rest density instead of getting an extrapolation.

Root cause of the decline, from the SDF field export
(`scratchpad/dump_m34_sdfField.py`): the merged solid is built entirely from
`min`/`max` CSG combinations, and every such combination is only C0 -- it has a
real gradient kink wherever two terms cross, and the kink locus can be a *curve*
with awkward topology, not a line. Isolating `dFillet = max(box, -disc)` alone
reproduces a spurious crescent kink sitting in open fluid, where `|grad|`
measures **0.005** instead of 1. `_mergedSurface`'s central difference
(`eps = 0.1 dx`) cannot survive that, so the normal is mis-aimed, the retraction
never finds a clearing node, and the offset zeroes. Pointwise the merged SDF is
*correct* everywhere tested (sign and value); only the derivative is unusable.

Two fixes were tried on the placement itself:

* **`'lattice'` (`_latticeGhostOffsets`, new)** -- sign-only, no gradient
  anywhere. Classify the sampling lattice fluid/solid by sign, take the nearest
  fluid site as the escape direction, step out to the mirror distance
  `2 |sdf(r_b)|`, snap back to the lattice. Flat walls reproduce `'gridsnap'`
  **exactly** (570/570 identical offsets), and at the convex 45 deg apex it is
  strictly better: **21/21 nodes land on a lattice site vs 0/21** for gridsnap,
  which averages 0.41 dx (up to 0.62 dx) off-lattice there because its
  `(floor(d/dx)+0.5) dx` arithmetic assumes an axis-aligned wall and cannot hit
  the lattice on a 45 deg face. But *as a wholesale replacement it is worse on
  the case* -- 3/6 gates, obstacle penetration unchanged at 16.24 dx -- because
  its longer levers (max 20.5 vs 14.0 dx) and its ghost sharing at the apex
  (16 distinct sites for 21 particles) cost more than the conditioning gains.
* **`'hybrid'` (`_hybridGhostOffsets`, the keeper)** -- gridsnap everywhere it
  produces a fluid-side node, lattice **only** for the particles it declines.
  Spatially the rescue is entirely at the wedge base and the fillet (737
  particles); the tank walls are untouched, so it is a no-op on flat geometry by
  construction.

| Marrone 3.4, nx=256, delta-SPH + PST, symplecticEuler, freeSlip | gridsnap | lattice | **hybrid** |
|---|---|---|---|
| gates | 1/6 | 3/6 | **6/6** |
| max\|v\| | 14090 (2307 U_max) | 93.2 | **27.7 (4.5 U_max)** |
| tank-wall penetration | 72706 dx | 6.89 dx | **0.00 dx** |
| obstacle penetration | 15.95 dx | 16.24 dx | **0.16 dx** |
| rho [P05, P99] | [0.717, 1.307] | [0.964, 1.079] | **[0.9983, 1.0184]** |
| rho pointwise max | 93.4 | 18.4 | **1.150** |
| KE_end | 2.04e5 | 58.4 | **22.5**, decaying |
| zero-offset fallbacks | 1187 | 450 | 450 (toe 440, **fillet 0**) |

**The onset does not move** -- maxv lifts off at t ~ 0.61 s in every variant.
That event is the front reaching the obstacle, a real flow feature; the boundary
treatment was amplifying it into a runaway, not causing it. Under hybrid it
peaks 27.7 at t\* = 1.97 and decays monotonically.

**Full record t\* = 7.5 holds 6/6** (10741 steps): max\|v\| still 27.7 (nothing
later exceeds the first impact), obstacle penetration 0.16 -> 0.21 dx over twice
the duration, tank walls 0.00 dx throughout. The P1-P9 probes fire in the right
order and match the documented phenomenology -- edge P1-P3 peak at t\* 1.84-1.98
(68 / 58 / 89 rho g H), roof P4-P6 at t\* 5.9-6.2 (the re-impact 5.2.2 predicts
at ~5.5), fillet P7-P9 at t\* 7.1-7.5 with only 61-573 wet samples and still
rising at the end ("barely wets by t\* = 7.4"). Raw edge peaks sit above
Wagner's 36.7 rho g H, which is item 3's point-probe gap, not a scheme error.

**englishWedge dp = 0.01 also improves, resolving the plan's open
concave-corner item:** base-corner RMSE **0.0431 -> 0.0178** (gate 0.03), better
than the old bodynode 0.0150/9-9 reference; faces 0.033 -> 0.012, apex
0.015 -> 0.008, 0 dx penetration. Still 8/9, but the failing check is now
"kinetic energy not growing" (2nd-half dKE/dt 5.3e-05 against a settled KE of
1.7e-04, itself under the 5.6e-04 gate) -- suspected an artifact of running
t = 8 s against English's 4 s, since that envelope compares halves of the
record. Unresolved.

#### (b) The solid/fluid partition had a hole -- `regions/filter.py`

`regions/sample.py:31` claims `sdf < 0` for the boundary band and
`regions/filter.py:20` claimed `sdf > 0` for fluid, leaving `== 0` claimed by
**neither**. That is not hypothetical: the 3.4 obstacle is built from exact
multiples (`H = W/10`, `toe_x` an integer number of dx from the centre), so its
top and back faces land exactly on lattice rows and lost a full row from *both*
bands -- a one-particle hole in the wall on the faces the jet hits hardest, the
same mechanism 5.2.2 blames for the old 1.25 dx obstacle penetration at the
`toe_x` seam. Fixed by `filter.py` taking `>= 0` (the zero goes to fluid, so the
solid stays exactly what the SDF calls negative; giving it to the boundary would
grow every lattice-coincident wall by a row). A no-op wherever nothing is
lattice-coincident. With it, wedge-top ghosts sit at exactly +1, +2, +3, +4, +5 dx
mirroring boundary rows at -1..-5, and the fallback count drops 932 -> 450.

A `dx/2` embedding of the obstacle/fillet bottom edges (and the fillet's right
edge, into the downstream wall) was tried first, to break the same coincidence
geometrically -- `_marroneSharpEdgeSDF` now takes `dx` for it. **It did not
help on its own**: the ghost fallback counts barely moved (toe 756 -> 759,
fillet 342 -> 336), and it is resolution- and geometry-specific where the
`filter.py` partition fix is universal. **Kept anyway, not reverted**, for two
reasons: it removes a genuine coincident-surface seam between the obstacle and
the tank's own SDF (two independent SDFs sharing an exact zero level, which is
what makes `_mergedSurface`'s central difference ambiguous there), and every
M34 result in this section -- including the 6/6 -- was measured with it in
place, so removing it now would mean shipping something other than what was
validated. Treat it as a neutral cleanup whose effect is bundled into those
numbers, not as an independently justified fix.

#### (c) sloshingTank -- `mdbcNoPenShiftMode = 'derivative'` is the root cause

The divergence at **t = 4.5726** (45,726 steps, NaN, rho -> 1.9e30, void
fraction 0.36) is **pre-existing on HEAD and unrelated to (a)/(b)**, which are
provable no-ops on this case: 0 fluid sites at `sdf == 0` (the tank walls
straddle at exactly +-0.5 dx), 0 of 3620 gridsnap nodes declined, hence 0
rescued and **bit-identical offset arrays**. 5.10's own results table lists only
`sloshingTank, t 4.5 | 45000 steps` for the committed normal-vector reflection --
the same numbers -- and warns that earlier baselines "need re-running before
being compared against anything new". This is that.

Found by checkpointing and resuming (below) rather than by inference. It is **one
particle**, UID 4804, sliding along the ceiling:

| step | y | v_norm | nopen_n | rho | fluid nb |
|---|---|---|---|---|---|
| 43500 | 0.50819 | -0.012 | +0.014 | 0.995 | 6 |
| 43580 | 0.50809 | -5.545 | +6.063 | 0.938 | 6 |
| 43700 | 0.50803 | -19.87 | +20.44 | 0.800 | 3 |
| 44000 | 0.50802 | -20.38 | +20.88 | 0.602 | **0** |

`v_norm` is the velocity on the inward normal (negative = into the ceiling).
`nopen_n ~ -v_norm` at **every** step: the correction cancels the *displacement*
exactly -- `y` is pinned to five decimals -- but the stored velocity keeps its
-21 m/s normal component indefinitely, because in `'derivative'` mode the term
is summed into `dvdt` as a force and a force can only oppose a velocity, never
replace it. The continuity equation then dots that phantom normal velocity into
the gradient sum and grinds rho from 0.995 down to 0.60 while the particle sheds
all six fluid neighbours. 5.9 already recorded this deviation -- "applied as an
acceleration summed into `dvdt` rather than DualSPHysics' post-integration
velocity replacement" -- without connecting it to a failure.

**The tangential story is a red herring, disproved by measurement** (user's
call, confirmed): `SUM_j (m_j/rho_j) grad W_ij` over the wall stencil is purely
normal to 1 part in 1e4, and the wall particles share one velocity exactly
(`spread = 0`, because the roll is applied as a *rotating gravity vector*, not
tank motion), so `(v_i - v_wall)` factors out and tangential sliding contributes
**+0.0085 of a total +1352**. A one-sided stencil does not manufacture density
error from tangential motion.

**`'finalize'` fixes it.** Swapping only the mode on a resume from the same
checkpoint:

| mode | v_norm over 300 steps | rho | y |
|---|---|---|---|
| `derivative` | -0.012 -> **-21.0** | 0.995 -> **0.707** | pinned 0.50803 |
| **`finalize`** | -0.012 -> **~0** | 0.995 -> **0.995** | 0.50819 -> **0.50704** (drops away) |
| `off` | stays small | healthy | 0.50819 -> **0.51205** (drifts up, nothing arrests it) |

Under `finalize` the velocity is genuinely replaced, the particle reflects and
**falls away from the ceiling**, rho recovers within a few steps, and
`nopen_n -> 0.000` because there is no longer any into-wall motion. Full 7 s
run: `diverged=False`, 70001 steps, densityMedian 1.0029 (baseline ~1.005), KE
final 0.0247 (~0.028), voidFraction max **0.00097** (vs 0.36-0.43),
pairedFraction max 0.080 (vs 0.29-0.36), maxDensity max 2.0056 (vs 1.9e30),
`sensorRho` [0.853, 1.249] so the negative-pressure transients 5.2.3 validates
are present. The raw sensor peak 213.8 kPa against the measured 2.2-13.1 kPa
band is the pre-existing `machTarget` item, not a no-pen question.

**Not flipped as the default.** `'derivative'` is labelled historical and
`'finalize'` is the reference behaviour, but 5.12's ceiling/gravity concern is
specifically about `finalize`'s `vPre` discarding the step's gravity, which
wants its own check first -- and see the open items below, where a ceiling
particle under `finalize` hovers without dropping.

#### (d) Rejected on measurement -- do not retry

All four looked right on a synthetic harness or from the source and failed on
the real case. All reverted; `density2025.py` and `wp_surfaceAware.py` are at
HEAD.

1. **Cavitation-style floor on `rho_b`** (`clamp(rho_b, min=rho_floor)` from an
   EOS-inverted ~1 atm tension limit). Engaged exactly as designed (min pinned
   at 0.9871 from t = 0.6 s) and changed nothing: 3.4 still failed 5/6 gates
   with the same onset. The floor was bounding the wrong side.
2. **Symmetric ceiling as well** (bound `rho_b` above too). Also failed, and
   is fragile on its own terms -- hydrostatic pressure exceeds a fixed 1 atm
   bound at scale (user's objection).
3. **Dropping the fallback `min=rho0` clamp** + relaxing `numNeighbors > 1` to
   `> 0`, so the wall can follow the fluid into tension. `probe_mdbcSparseFluid
   --rhoSweep` shows the clamp *is* one-sided and rectifying -- the wall tracks
   rho_fluid exactly above rho0 (dp = 0) but floors every sub-rho0 reading, so a
   tensioned near-wall particle faces `p_b = 0` and an unbalanced jump. But
   removing it made sloshing **worse** (died t=2.39 vs 4.57), because with
   `mask_i == 1` the Antuono switch takes `P_i + P_j` and a mirrored wall gives
   `2 P_i`: measured -6.06e3 -> **-1.04e4** on the harness.
4. **The three-part combination** -- clamp removal + restricting the Antuono
   free-surface override to fluid-fluid pairs + `supportScale 1.0 -> 2.0` for the
   ghost search (Marrone's larger boundary radius, which addresses the real
   inconsistency that a deep wall particle is inside the fluid's kernel support
   while its ghost reaches no fluid). On the harness this is excellent and the
   three are strictly coupled -- any two without the third is neutral or worse,
   and together the spurious attraction goes -6.06e3 -> **+1.9e-4** at 3.0x, i.e.
   zero to float noise. On the case it **diverged at t = 2.39 s**. Best read:
   at enlarged reach the ghosts start reading fluid that is not physically
   adjacent, including free-surface spray (`sensorRho` hit the 0.05 floor), and
   with `numNeighbors > 0` a single distant stray can set `rho_b` outright.
   The harness has one clean cluster and no spray, so it could not show this.

Note for (3)/(4): `supportScale` is silently capped by a precomputed adjacency,
which is built at fluid support -- the first two attempts to enlarge it measured
byte-identical until the call was made to fall through to `interpolateLiuLiu`'s
own grid search.

#### (e) Checkpoint / resume now works -- two bugs fixed

`storeMode='states'` was unusable, which is why this session burned runs on
hypotheses instead of reading the failing state:

* `io/export.py` -- a `None` in `extraData` (sloshingTank's unset
  `noPenShift`) maps to numpy object dtype, h5py rejects it, and the exception
  aborted the **whole** state write. Now recorded as the sentinel `'None'`.
* `io/hdf5.py` `loadState` -- `UIDcounter` is a scalar, so the writer never
  emits it and rebuilding the state dataclass failed on the missing required
  argument, making every checkpoint unloadable. Now derived from `UIDs.max()+1`,
  the reconstruction `importIO.py:210` already uses.
* `examples/sloshingTank/run_sloshingTank.py` gained `--storeInterval` /
  `--storeMode`; the flags were unreachable from the CLI.

Still broken, worked around: `schemeNameToSimulationScheme` loops
`CompressibleSPHScheme` before `WeaklyCompressibleSPHScheme`, so `deltaSPH`
resolves to the compressible variant and `importSimulationSystem`'s stage loader
builds the wrong Update class (`CompressibleSystemUpdate` missing `dudt`/`dEdt`).
Load the `state` group directly with `loadState` to sidestep it.

The resume loop that cracked (c): `scratchpad/probe_slosh_resume.py` grafts a
checkpoint into a live system (the integrator wants the *system* wrapper, with
particles at `.state`) and steps forward with per-step instrumentation on one
UID, including a `--noPenShift` override so modes can be A/B-ed from an
identical state in seconds rather than one 1.5 h run per hypothesis.

#### New probes

| script | what it answers |
|---|---|
| `scripts/probe_mdbcSparseFluid.py` | does the wall generate pressure against sparse fluid it shouldn't? `--rhoSweep` sweeps the cluster density above *and* below rho0 and reports the jump `p_fluid - p_b`; `--surfaceMask` flags the cluster as free-surface, which is the realistic case and the one that flips the Antuono branch |
| `scratchpad/probe_slosh_resume.py` | resume a checkpoint, step forward, per-step trace of one UID; `--noPenShift` A/B |
| `scratchpad/probe_slosh_gradsum.py` | recovers `SUM (m_j/rho_j) grad W_ij` from the real operator (zero the query velocity, set all others to a uniform `e`) to decompose drho/dt into normal vs tangential |
| `scratchpad/dump_m34_sdfField.py` | merged SDF value + central-difference gradient + zero isoline, whole domain / wedge / fillet -- how the CSG kinks were found |
| `scratchpad/cmp_apex_sampling.py`, `dump_hybrid_rescue_map.py` | apex ghost placement gridsnap vs lattice; which particles the hybrid rule rescues |

#### Reproductions -- both issues have a named particle and an exact checkpoint

Everything below is in **`export/16-sloshingTank-wcsph_2026-09-12_16-45-47`**
(nx = 225, `--noPenShift finalize`, hybrid ghosts, 7 s clean, 141 checkpoints
every 500 steps, 9.6 GB). Keep or regenerate with:

```
WARPSPH_GHOST_PLACEMENT=hybrid python examples/sloshingTank/run_sloshingTank.py \
  --scheme wcsph --nx 225 --tLimit 7 --noPenShift finalize \
  --store --storeMode states --storeInterval 500 --plot --video --out <dir>
```

| issue | reproduction |
|---|---|
| **hovering ceiling particle** | `UID 4104`, checkpoints `state_60000` (t = 6.0) onward. `python scratchpad/probe_slosh_resume.py --dir <dir> --from-step 60000 --steps 200 --every 25 --uid 4104 --noPenShift finalize` |
| **few-neighbour cluster kick** | `UID 3792 / 4673 / 4006`, checkpoints `state_33500`-`state_36000` (t = 3.35-3.60), peak at `state_34500` (t = 3.45). `python scratchpad/probe_slosh_hover.py --dir <dir> --what rightwall` for the band summary; the resume probe with `--uid 3792` for the per-step trace |

`scratchpad/probe_slosh_hover.py --what hover` lists the ceiling residents in
the final checkpoint and traces them back; `probe_slosh_residual.py` scans all
141 for the two signatures. Note the cluster event never touches
`maxVelocity` (1.66 m/s absolute), so **neither issue is visible in the
aggregate diagnostics** -- both were found from the field video and then
localised from checkpoints.

#### Open / next

1. **Under `finalize`, a ceiling particle hovers without dropping -- NOT 5.12's
   gravity discard.** The obvious attribution is wrong and was checked:
   resuming at `state_60000` with `finalize`, `nopen_n = 0.000` at **every**
   step and `v_norm ~ 0`, so the approach gate never fires, `active` is false,
   and `vPre` is never used. What pins it is `dvdt_n` running **-28 to -123**,
   i.e. a steady acceleration *into* the ceiling at 3-12x gravity, already net
   of gravity's +9.81 the other way. With rho = 0.9992 the particle carries
   p ~ -0.4 against a wall reading exactly rho0 (p_b = 0) -- the same
   one-sided tension asymmetry `probe_mdbcSparseFluid --rhoSweep` measures in
   (d)(3). It is mild but permanent, and it only has to beat 9.81 to pin a
   particle indefinitely. **So the hover and (c)'s density collapse are two
   different bugs**, and this one is in the pressure/wall-density path, not the
   no-pen path -- nothing in the no-pen term will fix it.

   5.12's rule itself still stands as a latent issue (a particle that *does*
   approach a ceiling loses its gravity). It is **implemented and gated off**
   behind `_RESTORE_GRAVITY_ON_CEILING_NOPEN` in `systems/weaklyCompressible.py`
   (mean inward boundary normal from the neighbours' ghost offsets, restore
   `dt (g . n_hat) n_hat` only where `g . n_hat > 0`). Left inactive on purpose:
   there is no case that exercises it, and an unvalidated behaviour change to
   `finalize` is how (d)'s four rejected fixes got expensive. Enable it
   alongside a case where a particle genuinely approaches a ceiling. Note the
   shift vector is *not* a usable proxy for the normal there -- its
   `factor = -4 ratio + 3` goes negative past ~1.25 dx, so the shift can point
   back into the wall.

2. **The right-wall cluster just after t = 3.4 s is a thin-sheet event, not a
   boundary one** -- an instance of 5.11's open class rather than a new
   mechanism. Traced (UIDs 3792 / 4673 / 4006, checkpoints 33500-36000):
   (bobbing at y ~ 0.05-0.16) until t ~ 5.8, is thrown to the ceiling, and then
   sits there for the **last full second of the run -- 10,000 steps**:

   | t | y | vy |
   |---|---|---|
   | 6.000 | 0.50707 | -0.0000 |
   | 6.400 | 0.50701 | +0.0021 |
   | 7.000 | 0.50700 | -0.0003 |

   Under gravity it should cross the whole tank in that second. Its velocity is
   purely tangential (0.1787, -0.0003), it has **0 fluid neighbours**, and
   `rho = 0.99966` -- perfectly healthy, which is why it hovers rather than
   exploding (contrast UID 4804 in (c), whose rho collapsed). So `finalize`
   trades a density-destroying phantom velocity for a kinematically stuck
   particle: strictly better, still wrong. This is 5.12's `vPre` discard, in
   the orientation 5.12 says must *not* discard (ceiling: `n_hat . g > 0`,
   gravity pulls away from the wall). Fix per 5.12's rule, and check a floor
   too so it does not start dropping particles through beds. **Blocks making
   `finalize` the default.**
2. **The right-wall cluster just after t = 3.4 s is a thin-sheet event, not a
   boundary one** -- an instance of 5.11's open class rather than a new
   mechanism. Traced (UIDs 3792 / 4673 / 4006, checkpoints 33500-36000):

   | t | \|v\| | rho | fluid nb |
   |---|---|---|---|
   | 3.350 | 1.22 | ~1.000 | **17** |
   | 3.450 | **1.66** | 1.004 | **9** |
   | 3.500 | **0.17** | 1.004 | 9 |
   | 3.600 | 0.86 | ~1.000 | **18** |

   Densities stay in [0.993, 1.009] throughout, so neither the wall reading nor
   the density is implicated; what tracks the event is the **neighbour count
   halving, 17 -> 9 -> 18**. The cluster thins away from the bulk, takes a ~2x
   kick, decelerates 10x on reattachment, and recovers inside 0.1 s. Absolutely
   it is mild (1.66 m/s) but it is **6.6x the local bulk median** (0.252),
   which is why it reads as violent on screen while never touching the
   aggregate maxima -- worth remembering when grading by `maxVelocity`. Same
   family as 5.11's `rho_min 0.91` / 8.6 U_max sheet collapse and 5.7's tongue
   fragmentation: "SPH with too few neighbours, not SPH against a wall". The
   instrument 5.11 asks for (a thin-sheet probe, acceleration vs thickness) is
   still the right next step, and this gives it a second, milder target to
   reproduce alongside 5.11's.

   **Term decomposition (2026-09-12, `scratchpad/probe_slosh_rightwallTerms.py`)
   -- it is specifically the pressure-force term, not diffusion.** Monkeypatched
   `computeVelocityDiffusion`/`computePressureForceSurfaceAware`/`computeForcing`/
   `computeGravity`/`computeMdbcNoPenShift` in-process (no file edits --
   `schemes.deltaSPH`'s bound names, matching the `_mdbcDensityHook.py` pattern)
   to capture each term's per-particle contribution to `dvdt`, then resumed
   `state_33000` and stepped through the event with all three UIDs traced. At
   the kick, `computePressureForceSurfaceAware`'s contribution is **2-3 orders
   of magnitude larger than `computeVelocityDiffusion`'s, every time** --
   e.g. one representative step: pressure 3.44e3 vs diffusion 56.8 for UID 3792;
   pressure 3.77e3 vs diffusion 23.3 for UID 4006. Gravity is the constant 9.81
   background; forcing and the no-pen shift are inactive here (`finalize` mode
   applies the no-pen correction outside this step function, so it never
   appears in this decomposition -- expected, not a gap). All three UIDs sit
   1.3-1.9 dx from the nearest boundary particle at the pre-event checkpoint
   (`state_33500`), so this is genuinely wall-adjacent, matching the "right-wall"
   name; checking the nearest boundary particles' mDBC ghost fits at that same
   checkpoint found them well-conditioned (`det` 3-8x the threshold, `numNeighbors`
   13-16) -- **not** the same under-conditioned-fallback mechanism §5.14 found
   for UID 4104's ceiling hover. This looks like a different failure mode
   sharing the same trigger (few real neighbours near a wall as a thin sheet
   passes), not the same bug recurring.

   **Caveat: the resumed re-simulation is measurably more violent than the
   archived run at the same absolute step -- one candidate cause checked and
   ruled out.** The archived checkpoints show all three UIDs' densities in
   [0.993, 1.009] the whole time; the resumed re-simulation from the identical
   `state_33000` checkpoint crashes to 0.94-0.98 with the pressure term in the
   thousands by the same step count.

   **Ruled out: this session's uncommitted §5.14 pressure-switch fix.** That
   fix was live in the working tree for every probe run so far, including this
   one -- a real methodological gap to control for, since it means every probe
   in this session ran under different physics than whatever originally
   produced the archived checkpoint if that checkpoint predates the fix.
   Checked directly: `git stash` on `modules/pressure/wp_surfaceAware.py`
   (back to the exact pre-fix code, `sw = P_i>=0 or P_j>=0 or mask_i==1`) and
   re-ran the identical resume. **Still diverges, differently but no less
   violently**: density spikes to 1.106 then crashes to 0.968 by step
   33800-34000 (pressure term 6.93e3), vs. the with-fix run's crash to
   0.94-0.98 (pressure 1.4e3-3.8e3). Neither matches the archived
   [0.993, 1.009]. So the fix is not the cause; restored (`git stash pop`)
   before continuing.

   Also checked: the archived `config.json` confirms `pressureForceTerm:
   Antuono` and the documented reproduction command's `WARPSPH_GHOST_PLACEMENT=hybrid`
   / `--noPenShift finalize` both match what every probe script in this
   session sets (hybrid is the code default, finalize is set explicitly) --
   not a config mismatch either.

   **Ruled out: stale pre-checkpoint adjacency reuse.** `running.adjacency`
   after `run(..., nSteps=1)` is built from t=0 (at-rest) positions and is
   never touched by grafting the checkpoint onto `running.state`; `deltaSPH_step`
   reads it as `priorNeighborhood` and only rebuilds if
   `_verlet_validity_metrics` says the Verlet buffer is exceeded --  computed
   via `_minimum_image_delta`, i.e. *minimum-image* (wrap-around) distance,
   and this domain is `periodic=[True,True]` (a closed tank, but the neighbour-search
   infrastructure treats the box as periodic regardless -- boundary particles
   fill the seam so it's a no-op physically, but it does change how this
   validity check measures distance). A real candidate: if a particle's t=0
   -> checkpoint displacement happened to look small under wrapping, the stale
   adjacency could be reused. Tested directly: explicitly set
   `running.adjacency = None` before stepping, forcing a from-scratch
   `radiusSearchCompactHashMap_` build on the first resumed step regardless of
   the validity check. **Byte-identical result** to leaving it alone -- same
   trajectory, same spike, same step. Not the cause.

   **Adjudicated: not a resume bug.** Three independent, direct mechanisms
   checked and ruled out (the §5.14 fix, config parity, stale adjacency) is
   enough to stop looking for a resume defect. What settles it further:
   comparing the resumed trace against the *archived* checkpoints directly
   (not just against each other) shows the two trajectories agree closely for
   the first ~700 steps after resuming --

   | step | archived rho (3792 / 4673 / 4006) | resumed rho |
   |---|---|---|
   | 33500 | 0.99962 / 0.99956 / 1.00037 | 1.00010 / 1.00002 / 1.00002 -- close |
   | 33700 | (no archived checkpoint this fine) | 1.00038 / 0.99989 / 1.00002 -- still mild |
   | 33900 | (no archived checkpoint this fine) | 0.96044 / 1.00094 / 0.97680 -- splitting |
   | 34000 | 1.00209 / 1.00234 / 0.99827 | 1.00243 / 0.99497 / **0.93766** -- UID 4006 clearly split |

   -- and only splits apart during the violent window itself (33700-34000),
   the same window the pressure-force term spikes in. A resume-fidelity bug
   would be expected to show divergence immediately on resuming (step
   33000-33100), not after 700 clean, matching steps. **This is the signature
   of a genuinely sensitive/marginal configuration**, not a resume defect:
   floating-point-level differences an independent resume cannot avoid (GPU
   hash/neighbour-search reduction order is not guaranteed bit-identical
   across separate kernel launches even on identical inputs) get amplified by
   the same too-few-neighbours pressure-sum mechanism the term decomposition
   already identified. The pressure-vs-diffusion magnitude finding stands
   confirmed on two independent trajectories now (fixed and pre-fix code, both
   diverging differently from the archive but agreeing that pressure, not
   diffusion, dominates); exact severity numbers from any single resumed run
   still should not be quoted as matching the archived event's numbers, since
   this is now understood to be inherent to the phenomenon, not fixable by
   getting the resume "more correct."
3. ~~Decide whether `'finalize'` becomes the default~~ **DONE -- both defaults
   switched**: `_GHOST_PLACEMENT_DEFAULT = 'hybrid'`
   (`rigidBody/ghostParticles.py`) and `mdbcNoPenShiftMode = 'finalize'`
   (`configurations/weaklyCompressible.py`). Item 1 is *not* a blocker after
   all -- it turned out not to be the `vPre` discard, so nothing in the no-pen
   path gates it.
3b. **`test_incompressibleKrylov.py::test_minresGivensMatchesDenseLstsq` is
   skipped** -- it fails intermittently in a full-suite run but passes
   standalone every time (3/3), so it is suite-ordering or RNG state rather
   than the MINRES core, and it was red on every commit. Low priority, but
   worth a look eventually: the random SPD/NSD systems in it are drawn with no
   explicit seed, which is the obvious suspect.
4. Marrone 3.1 under hybrid scored 5/9 with all four failures on P2, but the
   comparison is confounded -- the ghost mode, the integrator/wall BC
   (`symplecticEuler` + `freeSlip`, which postdate the last recorded M31
   numbers) and the duration all changed at once, and one failure ("P2 back to
   quiescent, max 3.034 for t\* > 6.4") is partly an artifact of running to
   t\* = 10.1 when that envelope assumes the record ends near t\* ~ 7.7. P1
   passes everything (arrival 2.51, plateau 0.47 vs Buchner 0.55, overshoot
   0.89); wall penetration 0.95 dx / 1 particle. Needs the gridsnap control at
   the identical config.
5. `_gridSnapGhostOffsets` still declines 440 particles at the 3.4 toe, where
   the fluid sliver between the wedge underside and the floor is genuinely
   thinner than dx. Accepted for now (obstacle penetration is 0.16 dx), but it
   is the remaining coverage gap.

### 5.14 The Tensile Instability Control paper found -- `PressureForceScheme.Antuono` has an extra branch Eq. (9) does not have  (2026-09-12)

The user supplied `literature/1-s2.0-S0010465517303995-main.pdf`, since renamed
and synced to the four literature-tracking files as **`sun2018`** (Sun,
Colagrossi, Marrone, Antuono, Zhang, *Multi-resolution Delta-plus-SPH with
tensile instability control*, Comput. Phys. Commun. 224:63-80, 2018,
`10.1016/j.cpc.2017.11.016`) -- ref [13] in `sun2019`, and the actual origin of
the pressure-symmetrization switch this codebase names `PressureForceScheme.Antuono`
(`modules/pressure/wp_surfaceAware.py`). Neither `sun2017` nor `sun2019`
(already on disk) states this switch; `sun2017` Eq. (1)'s momentum equation
always uses the plain symmetric `(p_i+p_j)` and controls tensile instability
entirely through the PST's `[1 + R(W_ij/W(Δx))ⁿ]` term (§5.13's shifting, not
the pressure force), and `sun2019` only paraphrases it ("the formula can be
modified locally by using `p_j - p_i`") without the free-surface exception.

**`sun2018` Eq. (9), exactly:**

```
F_ji = p_j + p_i    if  p_i >= 0  or  i in SF
     = p_j - p_i    if  p_i <  0  and  i not in SF
```

`SF` is the free-surface region (the free-surface particles and their
neighbours, §2.1). The switch is conditioned **only on the query particle i**
-- its own pressure sign and its own free-surface membership -- never on the
neighbour j's pressure.

**The code's version adds a second OR-branch the equation does not have:**

```python
# modules/pressure/wp_surfaceAware.py, PressureForceScheme.Antuono, before this session
sw = P_i >= 0.0 or P_j >= 0.0
sw = (sw) or (mask_i == 1)
p_ij = (P_j + P_i) if sw else (P_j - P_i)
```

`or P_j >= 0.0` switches to the symmetric (attractive-capable) form whenever
the **neighbour's** pressure is non-negative, regardless of `P_i`'s sign or
whether `i` is anywhere near a free surface. Every mDBC wall in this codebase
clamps `rho_b` at-or-above `rho0` in every path that matters in practice (the
fallback Shepard clamp kept by [[marrone31-mdbc-wall-regression]]'s (B)
REFINED fix, and the `numNeighbors` gate that returns rest density outright),
so **`P_j >= 0` at a solid wall essentially always**. The extra branch therefore
forces the symmetric form for *every* fluid particle adjacent to a wall the
instant that fluid goes into tension (`P_i < 0`) -- exactly the condition
`sun2018` Eq. (9) is designed to route to the antisymmetric form instead. This
is not a free-surface-only edge case: `PressureForceScheme.Antuono` is the
**default** `pressureForceTerm` for both `configurations/weaklyCompressible.py`
and `configurations/incompressible.py`, so the bug is live on every wall in
every WCSPH/incompressible run, whether or not `--surfaceMask`-style free-surface
flagging is in play.

**Verified with the existing harness, no new script needed.**
`scripts/probe_mdbcSparseFluid.py --rhoSweep` already assembles the real
operator (`computePressureForceSurfaceAware`) against a synthetic wall + sparse
fluid cluster and reports the *signed* normal acceleration, with a docstring
that already half-diagnosed this ("by pair force `(p_i+p_j)` a mirrored wall is
*more* attractive") but attributed the trigger only to the `--surfaceMask`
flag. Run **without** `--surfaceMask` at rest (`rho_fluid=0.90`, a tensioned
fluid particle 0.5 dx off a ceiling, `p_b` clamped to exactly `rho0` so
`P_j=0`):

| | `a_normal` (`+` = away from wall, `-` = pulled into it) |
|---|---|
| **before** (`P_i>=0 or P_j>=0 or mask_i`) | **-8.17e3** -- pulled *into* the ceiling |
| **after** (`P_i>=0 or mask_i`, matching Eq. 9) | **+8.17e3** -- correctly repelled |

Sign flips the same way across every shape (`single` / `pair-tangential` /
`block-2x2` / `block-3x3`), both wall orientations, and both probe distances
(0.5 / 1.0 dx) -- 36 configurations, all consistent, none regressed on the
compression side (`rho_fluid > rho0` stays repulsive, unchanged, since
`P_i >= 0` alone already covers it). The non-`--rhoSweep` mode (velocity x
distance x shape sweep, the `|p_b|` magnitude check) still reports
**PASS: every configuration produced `|p_b| <= 1e-4 rho0 c0^2`** -- the fix
does not touch the *density* extrapolation this mode checks, only the force
sign once a nonzero `p_b`/`P_i` exists.

**Fix applied (uncommitted):**

```python
sw = P_i >= 0.0 or (mask_i == 1)
p_ij = (P_j + P_i) if sw else (P_j - P_i)
```

`test_physics.py` + `test_wallPressure.py` green (exit 0, no failures) with the
fix in place.

**This is exactly the mechanism §5.13 item 1 (the sloshingTank ceiling-hover
particle) describes without naming it:** "`dvdt_n` running -28 to -123 ... p ~
-0.4 against a wall reading exactly rho0 (`p_b = 0`)" is precisely `P_i < 0`,
`P_j = 0 >= 0`, non-free-surface -- the old code's extra branch forces the
symmetric `(P_j+P_i) = P_i` (negative, attractive) where `sun2018` Eq. (9)
calls for the antisymmetric `(P_j-P_i) = -P_i` (positive, repulsive). It is
also the mechanism named in §5.13(d)(3)/(4)'s rejected-fix notes ("with
`mask_i == 1` the Antuono switch takes `P_i + P_j` and a mirrored wall gives
`2 P_i`") -- those attempts changed the density clamp to fight a force-sign bug
downstream of it, which is why (d)(3)/(4) made sloshing worse rather than
better: removing/relaxing the `rho_b` floor lets `P_j` swing negative too,
which coincidentally satisfies neither OR-branch and hides the symptom without
fixing the switch.

**Checkpoint reproduction run (2026-09-12): the §5.14 fix alone does NOT
resolve the ceiling-hover repro -- a second, independent bug does the actual
damage here.** Resumed `export/16-sloshingTank-wcsph_2026-09-12_16-45-47` at
`state_60000` (t=6.0) with the fix from this section already applied and
stepped UID 4104 forward 200 steps (`scratchpad/probe_slosh_hoverPressure.py`,
a `probe_slosh_resume.py` variant that also prints `P_i`, the nearest
boundary/ghost neighbour's density/pressure, and `mask_i`):

```
  step        t         y   P_i  maskI  nearestBnd_rho  nearestBnd_P  dvdt_n
 60000   6.0002   0.50707  -0.321   1        1.00000        0.0000  -28.36
 60175   6.0177   0.50701  -1.119   1        1.00000        0.0000  -135.3
```

`maskI = 1` throughout -- **the particle is flagged as free-surface**. Per
`sun2018` Eq. (9), `i in SF` alone is *sufficient* to select the symmetric
branch regardless of `p_i`'s sign -- so with the fix applied exactly as the
paper states it, this specific particle still gets `F_ji = p_j+p_i = p_i`
(negative, attractive) every step, and `dvdt_n` stays in the same -28..-135
range the pre-fix plan text recorded. The §5.14 fix is confirmed correct
against the equation and against the synthetic mDBC-wall harness (still true,
still worth keeping), but it is not sufficient for this reproduction because
**the free-surface flag itself is wrong here, not the switch that consumes it.**

**First analysis attempt was methodologically flawed -- caught by the user,
not self-corrected.** The first pass (`scratchpad/probe_slosh_surfaceDetect.py`)
found `numNeighbors (AllToAll, excludes ghost) = 30` against a 37.7 cutoff (52
ghost query points inside the support radius don't count), and a follow-up
angular-occupancy check appeared to confirm "no real gap" -- but that check
included ghost particles (kind 2) as if they were matter occupying a
direction. They are not: kind-2 points are pure MLS query locations for a
boundary particle's density fit, they exert no force on the query fluid
particle in any SPH sum, and the detector's own `AllToAll` mode already
excludes them (`checkDirectionality_j` opInt 9: `queryKind != 2`). Counting
them as "occupying" a sector was circular -- filling the picture with exactly
the particles that neither interact with UID 4104 nor are seen by anything
being audited. The user asked, correctly, why a particle genuinely at a
surface shouldn't read as one, which is what forced this back open.

**Redone with only fluid+boundary (the particles that actually exchange force
and that the detector actually sees):**

```
UID 4104, fluid+boundary only:  10/16 sectors occupied  (a real ~35% gap)
wave-crest particles (>6dx from any wall), same:  1-2/16 sectors occupied
```

A real gap, not the false 16/16. That still leaves it ambiguous by degree --
10/16 is meaningfully more enclosed than 1-2/16, but not "clearly not a
surface" either -- so the check was widened past one kernel support radius,
since a gap that closes up just outside it would mean "locally thin, but
still part of the bulk," while a gap that keeps widening means "genuinely cut
off":

```
fluid count within 1x/2x/3x/4x support radius (4/8/12/16 dx): flat at 5, no change
fluid count within 6x support radius (24 dx): jumps to 12 -- a spatially
  SEPARATE cluster ~20 dx below UID 4104, not a gradual thickening
```

**UID 4104 is a real, physically isolated film of fluid stuck to the ceiling,
cut off from the main body of water by a genuine ~20 dx span of open space.**
This is not a within-kernel-support artifact of temporarily thin sampling --
it is a real free surface. **Conclusion reversed from the first pass: the
free-surface flag is correct here, and so is the Antuono switch** -- per
`sun2018` Eq. (9), `i in SF` legitimately selects the symmetric branch for a
genuine free-surface particle regardless of `p_i`'s sign; that is exactly what
the paper's authors intend (§2.1: dropping this term near a real free surface
"leads to problems... where the pressure generally oscillates around the
zero value").

**So what is actually wrong sits one layer further down: the mDBC ghost fit
feeding `P_j` is under-conditioned, and its fallback value is being consumed
as if it were a real reading.** The boundary particle nearest UID 4104
(`scratchpad` ad-hoc check on `interpolateLiuLiu`'s own stencil):

```
ghost stencil: numNeighbors=6, det=1.711e-3, detFloor=1.800e-3  (BELOW the conditioning floor)
rho_b = 1.00000 exactly, P_j = 0.0000 exactly
```

`det` below `detFloor` means `computeMdbcDensity` cannot trust the MLS
extrapolation there (only 6 real neighbours to fit, because the real fluid
nearby genuinely is this sparse -- the same isolated-film fact just
established) and falls back to the Shepard/rest-density path, which floors at
exactly `rho0` -> `P_j=0`. **That `P_j=0` is a "not enough data" placeholder,
not a measured wall pressure**, but Eq. (9)'s switch has no way to
distinguish a trustworthy `P_j` from a fallback one -- it sees `p_i < 0`,
`i in SF`, and correctly (per the equation) returns the attractive symmetric
form, pinning a real, legitimately-tensioned free-surface film against the
wall instead of letting it fall.

**Not fixed, and the earlier two candidate fixes in this section (angular-gap
test, ghost-inclusive completeness count) are now understood to be treating
the wrong layer** -- the free-surface classification and the pressure switch
are both behaving as specified once the mDBC fallback is accounted for. The
open question is what the fallback density path should hand the pressure
switch when it does not have enough data to extrapolate: whether a starved
mDBC ghost fit should suppress the wall's contribution to the SF branch
specifically, report a value the switch can recognise as untrustworthy, or
something else, is a design question this session did not resolve.

**Scope: real but narrow, not systemic.** Scanning every fluid particle at
this checkpoint: 1250 of 5175 fluid particles are flagged free-surface (a
normal count for a sloshing wave crest); this specific compounding (a
genuinely isolated free-surface film also touching a wall, whose nearest
ghost fit is under-conditioned) is one plausible mechanism among however many
of those 1250 are also wall-adjacent, not scanned exhaustively this session.

**Also still open, unchanged by any of this:** the Marrone 3.1/3.4 +
englishWedge full-record regression sweep for the §5.14 pressure-switch fix
itself (`test_physics`/`test_wallPressure`/the synthetic harness are green,
but per [[validation-scheme-defaults]]'s practice a default-pressure-term
change this load-bearing needs the standard case sweep before being promoted
past "characterized"), and the §5.13 open-item-2 right-wall thin-sheet cluster
event, which this session did not touch.

### 5.15 The synthetic thin-sheet probe -- one mechanism explains 5.13 item 2, 5.14, and the "chaos" reading  (2026-09-12)

5.11 asked for this instrument two sessions ago ("a thin-sheet probe: N-particle
sheet, acceleration vs thickness") and it finally got built:
`scratchpad/probe_thinSheetPressure.py`. No wall, no mDBC, no resume, no case
machinery -- a bare periodic-in-x slab of regular particles with a UNIFORM
pressure field (so the physically correct force is exactly zero everywhere)
run straight through `computePressureForceSurfaceAware`, the same production
operator. It settles both open threads above from first principles.

**The mechanism, confirmed directly.** The symmetric pair term
`Sum_j V_j (P_i+P_j) grad_i W_ij` (`nonConservative`, and `Antuono` whenever
its switch picks the symmetric branch) reduces, for uniform `P0`, to
`2 P0 * Sum_j V_j grad_i W_ij` -- a real, nonzero force whenever a particle's
kernel support is truncated, proportional to the local pressure and to how
incomplete the support is, entirely independent of any actual pressure or
density gradient. `conservative` (`P_j-P_i`) is *exactly* zero for uniform `P`
at any truncation, term-by-term. This is textbook SPH kernel-consistency, not
new physics, and it is why `sun2018`'s TIC switch trades it in specifically
where it does: at a genuine free surface pressure is expected to be near zero,
so the artifact is usually small there; the switch does not (and, per Eq. (9)
as written, cannot) protect a particle whose local pressure is *not* small,
whether because it's compressive (`P0>=0`, which Eq. (9) always routes to the
symmetric form) or because it's a genuinely tensioned free-surface particle
(`i in SF`, also always symmetric per Eq. (9) -- §5.14's UID 4104 exactly).

**Measured directly** (support = 4 dx, a thick slab, edge row = depth 0):

| depth from edge (dx) | nNb | force, `nonConservative`/`Antuono` (P0 either sign, magnitude only) |
|---|---|---|
| 0 | 25 | **5450** |
| 1 | 32 | **2240** |
| 2 | 39 | **300** |
| 3+ | 44+ | ~0.001 (numerical noise) |

`conservative` and `Antuono` under `P0<0` + not flagged SF are 0 at every
depth, exactly. `Antuono` under `P0>=0` matches `nonConservative` at every
depth, exactly (Eq. (9) always symmetric there); flagging any depth's particle
free-surface (`mask_i=1`) makes `Antuono` match `nonConservative` there too,
regardless of `P0`'s sign -- reproducing the UID 4104 mechanism on a bare
lattice with no mDBC involved at all. For scale, depth-0's 5450 is **~550x
gravity**.

**On a perfect lattice this is a clean, 2-3 dx-deep, deterministic edge
effect** -- confirmed insensitive to sub-dx lattice phase (`--phaseShifts`
swept 0/0.1/0.25/0.5 dx: identical force at every depth, every shift, since a
perfect lattice is translation-invariant). That *rules out* "the edge sits at
an unlucky sub-dx phase" as an explanation for anything -- a real fluid's
disorder is what matters, not where the lattice happens to sit.

**With real disorder, the same mechanism reaches much deeper and becomes
unpredictable -- this is the resume-fidelity "chaos" from 5.13 item 2,
demonstrated on a bare lattice.** `--jitter 0.1` (each particle displaced by
an independent random offset up to 0.1 dx -- an order of magnitude coarser
than any floating-point-level difference a resume could introduce, chosen to
be visible, not realistic) turns depth 3-4's ~0 into 76-537, **varying 5-7x
between otherwise-identical random seeds**:

| seed | depth 2 | depth 3 | depth 4 |
|---|---|---|---|
| 0 | 418 | 355 | 75.9 |
| 1 | 285 | 87.4 | 372 |
| 2 | 412 | 134 | 309 |
| 3 | 79.6 | 397 | 362 |
| 4 | 537 | 216 | 360 |

Disorder does not just add noise on top of the deterministic depth profile --
it changes *how deep* the artifact reaches and makes its magnitude at a given
depth essentially a random draw. A real fluid near a wall or thinning into a
sheet is never a perfect lattice, so this is the realistic regime, not the
idealised one. This is why §5.13 item 2's resumed-vs-archived trajectories
agreed closely for ~700 steps and then split sharply exactly in the violent
window: not a resume bug (three independent candidates were checked and
ruled out there), but this exact sensitivity, operating on real position
differences too small to see directly.

**What this does and does not settle.** It confirms the pressure-vs-diffusion
magnitude finding (5.13 item 2) and the free-surface-flag mechanism (5.14)
share one root cause, and that the mechanism is real, well-understood SPH
theory rather than a codebase bug -- `PressureForceScheme.Antuono` is behaving
exactly as `sun2018` Eq. (9) specifies in every case tested this session; the
paper's own switch is what's exposed to this. It does **not** yet answer
whether this codebase should do anything differently: the options (an
explicit kernel-sum-completeness correction to the symmetric term, restricting
`Antuono`'s free-surface branch more tightly, accepting the artifact as
`sun2018` itself does, or something else) all have different blast radii
across every case using the default `pressureForceTerm`, and this session
did not pick one. That decision, and the still-open Marrone/englishWedge
regression sweep for the already-applied §5.14 fix, are the concrete next
steps.

### 5.16 Two full 7 s confirmation runs with the §5.14 fix -- clean, and 5.9's `symplecticEuler` divergence looks superseded  (2026-09-12)

With the §5.14 pressure-switch fix in the working tree, `examples/sloshingTank/run_sloshingTank.py --scheme wcsph --nx 225 --tLimit 7 --noPenShift finalize` (hybrid ghosts, the current defaults) run to the full record twice:

| integrator | diverged | densityMedian final | voidFraction max | pairedFraction max | raw Sensor-1 peak (measured band 2.2-13.1 kPa) |
|---|---|---|---|---|---|
| `rungeKutta2` (case default) | False | 1.0062 | 0.00097 | 0.073 | 233.7 kPa |
| `symplecticEuler` | **False** | 0.9997 | 0.00135 | 0.064 | **60.1 kPa** |

Both runs are healthy by every other diagnostic (density range, void/paired
fractions, KE still actively decaying, not frozen) -- consistent with the
baselines this plan already established, no regression from the §5.14 fix.

**`symplecticEuler` no longer diverges on this case.** The case file
(`cases/sloshingTank.py`) recorded it diverging at t = 0.727 s (§5.9),
prime-suspecting `deltaSPH_step`'s `nopenshift/dt` term, whose magnitude
depends on the integrator's stage splitting under
`mdbcNoPenShiftMode='derivative'`. That mode is no longer the default --
`'finalize'` (§5.13) applies the no-pen correction once per step outside
`deltaSPH_step` entirely, removing the suspected mechanism -- and under
today's defaults `symplecticEuler` ran the full 7 s clean, with a
meaningfully better Sensor-1 match than `rungeKutta2` (60.1 vs 233.7 kPa raw
peak against the same band). Case comment updated to record this as
superseded rather than current fact. **Not switched as the case default** --
this is one comparison run, not the Marrone/englishWedge sweep the plan's
other default changes went through before being adopted. `semiImplicitEuler`
(the other Euler-family scheme, diverges at t=0.041s at rest from a genuine
acoustic-mode instability unrelated to the no-pen mechanism) was not
re-tested and has no reason to behave differently.

Videos: `examples/sloshingTank/output/e44_fixedAntuono_full7s/` (RK2),
`e45_symplecticEuler_full7s/` (symplecticEuler) -- both are ordinary
`--out` run dirs (`**/output/` is gitignored, not durable). The field video
(mp4 + gif), Sensor-1 plot (png + pdf) and raw series (npz) for both runs are
additionally copied to
`examples/sloshingTank/output/reference_antuonoFix_2026-09-12/{rk2,symplecticEuler}/`
as a stable, clearly-named pair kept specifically for reuse in future
documentation (write-ups, presentations) -- still gitignored (these are
large binaries, ~50-75 MB each), so reference the path, don't expect `git
show` to find them.

### 5.17 The §5.14/5.15 artifact confirmed on Marrone 3.1 itself, not just sloshingTank  (2026-09-13)

An overnight batch (`scratchpad/run_overnight_batch.sh`) ran Marrone 3.1
(nx=67, plain `deltaSPH`, PST on) to t=5.0s (t*≈20.2) and flagged a visible
free-surface destabilisation around t≈1.5s that "doesn't look energy
conserving" -- the scalar trace (`kineticEnergy` in the run's npz) shows real
*increases*, not just noisy decay: e.g. 0.248→0.339→0.474→0.554 over 0.15s at
t≈2.05-2.25s with nothing external doing work.

Re-ran the identical config to t=2.6s with `store='states'`/`storeMode='states'`
(full per-particle export every 100 steps, `scratchpad/m31_dig/`) and diffed
consecutive stored frames for anomalous velocity jumps. Two clusters, both
matching the §5.15 signature exactly:

- **t≈1.50-1.65s (t*≈6.0-6.7)**, the reported event: particles on the
  left-side backwash sheet (x≈-1.07 near the original dam wall, and a second
  cluster at x≈0.05-0.22 spanning the surface down to ≈0.24 depth) show
  O(450-720 m/s²) (45-70 g) accelerations between consecutive stored frames
  (~0.01s apart).
- **t≈2.1-2.3s (t*≈8.6-9.2)**: the same signature spread across the *entire*
  free surface (x from -1.6 to 0), not one location.

In every case: `surfaceIndicators=1` (flagged free surface), density near
rest (0.97-1.04 -- mild disorder, not a torn sheet), and **pressure flipping
sign between consecutive stored frames** on the same particle (e.g. P = 333 →
-271; another: -161 → 151 → -176). Sign-flipping pressure under mild disorder
at truncated kernel support, on particles the Antuono switch (§5.14) routes
to the symmetric branch, is exactly the mechanism §5.15 characterised on a
bare synthetic slab -- not a resume/config/mDBC-wall defect, and not
something specific to sloshingTank.

**This raises the priority of §5.15's still-open decision.** It was left as
"understood, not fixed" because it had only been shown on a secondary case
(sloshingTank); it now demonstrably reaches Marrone 2011 §3.1 itself, at a
resolution (H/dx=40) and duration inside the paper's own tested range. The
next concrete step on this thread is picking one of §5.15's three options (a
kernel-sum completeness correction to the symmetric term, tightening when the
free-surface branch fires, or accepting the artifact as `sun2018` itself
does) rather than another characterisation pass.

Diagnostic scripts: `scratchpad/m31_dig_run.py` (the instrumented re-run),
`scratchpad/m31_dig_analyze.py` (the frame-diff anomaly finder) -- both
throwaway, not added to `scripts/`.

### 5.18 Per-step acceleration / no-pen-shift instrumentation  (2026-09-13)

Added because both this session's Marrone 3.1 dig (5.17) and the Marrone 3.4
re-investigation (Part 7 below) kept needing to answer "was `nopenshift`
involved in this spike, and how many particles did it touch?", which
previously meant writing a one-off probe script and re-running the case from
scratch each time.

`WeaklyCompressibleSystem.finalize` (`systems/weaklyCompressible.py`) now
computes, every step, on `self.stepDiagnostics` (a plain dict of Python
floats):

- `accelMag{Min,Max,Mean,P05,P95}` and `accel{X,Y}{Min,Max,Mean,P05,P95}` --
  the *net* fluid acceleration actually applied this step,
  `(v_after - v_before) / dt`, magnitude and per axis;
- when `mdbcNoPenShiftMode == 'finalize'` (the default): `nopenshiftNActive`
  (how many fluid particles the correction fired on) and
  `nopenshiftMag{Min,Max,Mean,P05,P95}` (the correction's magnitude among
  those particles). NaN-filled, not omitted, when nothing fired, so the
  trajectory's columns stay a fixed shape.
- `'derivative'`-mode's per-RK-stage no-pen term (`schemes/deltaSPH.py`,
  `dvdt_nopenshift`) is **not** separately instrumented yet -- it is summed
  into the stage acceleration alongside pressure/gravity/diffusion rather
  than applied as a separate, easily-isolated correction, and the currently
  active default is `'finalize'`.

`cases/weaklyCompressible.py`'s new `stepAccelerationDiagnostics(state)`
reads this dict back (returns `{}` before any step has run, or for a non-WC
scheme) and is wired into every case's `diagnostics()` that already reports
per-step scalars: `dambreak.py` (so every Marrone/englishWedge/openFlow run
gets it), `sloshingTank.py`, and `weaklyCompressibleDiagnostics` (the shared
helper `lidDrivenCavity`/`kolmogorov`/`randomFlow`/`impact`/`tgv`/
`oscillatingDroplet`/`drivenSquare`/`movingObstacle`/`rotatingSquarePatch`
already call) -- no per-case wiring needed beyond that one call. Every
existing probe script's npz export keeps its own hardcoded column list, so
picking these up in a saved `.npz` still needs that list extended (as done
ad hoc in `scratchpad/m34_dig_run.py`); reading `r.trajectory` directly (a
list of per-step dicts) already has them with no changes.

### 5.19 Quantitative driver of §5.17: the artifact scales with c0², and Marrone 3.1's own weak-compressibility spec is what exposes it  (2026-09-13)

The user noticed that `examples/weaklyCompressible/12-dambreak.py` (the plain
gallery `dambreakCase`, default params) shows none of §5.17's free-surface
destabilisation, and pointed at the one obvious difference: it runs at a much
lower sound speed. Checked directly: `12-dambreak`'s legacy back-solved `c0`
at its default `nx=128` is **19.76 m/s**; Marrone 3.1's validation config
(`c0Ratio=40`, matching Marrone's own weak-compressibility spec) is
**97.04 m/s** -- a 4.9x ratio.

**This EOS is exact, not approximate, about why that matters.**
`modules/eos/weaklyCompressible.py`'s `isoThermalEOS` (the scheme both cases
use) is literally `P = c0² * (rho - rho0)`. The density noise that drives the
§5.15/§5.17 symmetric-pressure-truncation artifact is a geometric/disorder
effect (kernel-sum incompleteness at a truncated or sparse support) --
nothing about *how much* a real particle's position is disordered depends on
`c0`. So for the same absolute density noise `Δrho`, a higher `c0` produces a
quadratically larger spurious pressure `P0 = c0² Δrho`, and the artifact's
own force (`~2 P0 Sum V_j grad W_ij`, §5.15) scales with it directly.
Marrone's validation protocol deliberately runs weakly-compressible (high
`c0`, low Mach) to stay physically accurate to real water -- which is exactly
what maximises this pre-existing artifact's absolute size. The two are not
separate concerns: **making the compressibility more physically correct makes
this artifact worse**, not better.

**Confirmed by direct A/B** (`scratchpad/m31_c0sweep_run.py`): the identical
Marrone 3.1 config (nx=67, PST on) run to t=1.9s at `c0Ratio=40` (c0=97.04,
the validation config) vs `c0Ratio=8` (c0=19.41 -- matching `12-dambreak`'s
own c0 almost exactly) --

| c0Ratio | c0 (m/s) | peak `accelMagMax`, t in [1,1.9]s | peak `accelMagP95` (typical, t=1.4-1.85s) |
|---|---|---|---|
| 40 | 97.04 | **19,632 m/s²** | 250-3900 |
| 8  | 19.41 | **1,146 m/s²**  | 45-75 |

A 5x drop in `c0` gave a **17x drop** in the peak spurious acceleration --
well above the 5x a linear relationship would give, and in the right
neighbourhood of the 25x (5²) an exact quadratic would give (not exact
because dt, substep count and the actual disorder realised also differ
between the two legs, not just `c0`). `minDensity`/`maxDensity` move the
*other* way (0.94/1.06 at `c0Ratio=40` vs 0.89/1.22 at `c0Ratio=8`) -- also
expected, not a contradiction: a softer EOS (lower `c0`) lets any given
velocity divergence, spurious or real, show up as a *larger* density swing
(compressibility ~ 1/c0²), so density range is the wrong metric to grade this
by; the force/acceleration columns (§5.18) are the ones that isolate the
artifact from the EOS's own response to it.

**Read for §5.15's still-open decision:** this doesn't change what the fix
should be, but it does mean the fix is not optional for any case that wants
to run genuinely weakly-compressible (high `c0Ratio`) *and* touch a free
surface or sparse/truncated region -- the artifact is not a fixed-size
nuisance, it scales directly with how physically accurate the compressibility
is set to be. A case that only ever runs at a low, back-solved `c0` (like the
gallery's default `12-dambreak`) is not "fine" in the sense of not having the
bug; it simply isn't running weakly-compressible enough to expose it at a
visible scale.

### 5.20 Ruled out: PST/shifting is not the cause (and §5.17's "free-surface" framing was incomplete -- it's compressive-bulk too)  (2026-09-13)

The user, from a video frame, flagged that the destabilisation looked like it
was starting *in the bulk*, a few particle layers under the free surface, and
asked whether particle shifting (PST) -- which Marrone 2011 Sec. 3 runs
without -- could be the cause or a contributing factor, since every
validation run so far (§5.17, §5.19's sweep included) left `shifting=None`,
i.e. the scheme default of PST **on**, never Marrone's own no-PST spec.

**Re-examining the existing §5.17 per-particle dump (with PST) first**:
counting every anomalous-jump event over the full t=0-2.6s run, not just the
hand-picked examples quoted in §5.17 --

| | count | share |
|---|---|---|
| total anomalous events | 29,617 | -- |
| flagged bulk (`surfaceIndicators=0`) both before and after | 10,823 | 36.5% |
| touches `surfaceIndicators=1` (before or after) | 18,794 | 63.5% |

So §5.17's write-up, which only quoted `surfaceIndicators=1` examples, was
**incomplete, not wrong about the mechanism**: over a third of the events are
on particles flagged bulk on both sides of the jump. This is not a surprise
once you re-read §5.14's own description of the Antuono switch -- `sun2018`
Eq. (9) picks the artifact-prone symmetric branch whenever the query
particle's own pressure is non-negative **or** it's flagged free-surface; the
`P_i >= 0` half of that condition fires for any compressive particle,
anywhere, bulk included. The user's video observation is real and is the
same mechanism, not a second bug.

**Then a direct test of the PST hypothesis**: re-ran the identical Marrone
3.1 validation config (nx=67, `c0Ratio=40`) with `shifting=False` -- Marrone's
actual spec, and *also* the config the 2026-09-10 memory
(`marrone31-mdbc-wall-regression`) recorded as unable to survive the first
impact at all (t*~2.5, a spatial-operator instability, "halving dt makes it
worse -- not CFL"). That finding no longer holds: this run went the full
t=2.6s (t*~10.5) `diverged=False`, past both the first impact and the whole
§5.17 destabilisation window -- consistent with the mDBC hybrid-ghost,
`nopenshift='finalize'`, and Antuono-switch fixes landed since then having
incidentally fixed it too (not confirmed further; a full no-PST regression
pass is separate future work).

**But turning PST off does not fix, or even clearly reduce, the
destabilisation:**

| | with PST (§5.17/5.19) | no PST (`shifting=False`) |
|---|---|---|
| peak `accelMagMax`, t in [1,2.6]s | 19,632 m/s² | **25,615 m/s²** (higher) |
| total anomalous events, t=0-2.6s | 29,617 | 23,650 (~20% fewer) |
| bulk-flagged share of those events | 36.5% | 38.7% (same, within noise) |
| `pairedFraction` at t=1.5-1.7s | 3-7% | **9-17%** (2-3x higher) |

PST is doing exactly what it's for -- `pairedFraction` (the clumping/tensile-
instability signature) is 2-3x lower with it on, confirming it isn't inert
here. But the peak spurious acceleration is *not* lower without PST (it's
somewhat higher), the event count only drops ~20% (not the order-of-magnitude
a primary cause being removed should give), and the bulk/surface split is
unchanged. **Read: PST suppresses one disorder channel (particle pairing)
that this codebase already knows can drive kernel-truncation artifacts
(§5.15), but removing it just lets a different disorder channel (natural
tensile-instability clumping, which is what PST exists to prevent) feed the
same mechanism instead.** PST is not the cause, and is not a viable fix on
its own -- it is a wash at best, and this specific run says slightly worse at
the peak.

**Net effect on §5.15's open decision: unchanged.** Both the c0² scaling
(§5.19) and this PST A/B point at the same place -- the mechanism is
`sun2018` Eq. (9)'s own switch condition acting on a *given* level of
particle disorder, and neither compressibility nor shifting are things this
plan should tune to avoid it; the fix has to be in the switch or the
truncated symmetric term itself.

Scripts: `scratchpad/m31_noPST_run.py` (the no-PST re-run, full per-particle
states + video); reused `scratchpad/m31_dig_analyze.py`'s method for the
recount, not the file itself.

### 5.21 Also ruled out: the DDT's L-renormalization is not the cause -- disabling it trades fewer/smaller spikes for a chronically noisier free surface and ~2x the retained energy  (2026-09-13)

Third candidate tested: `modules/deltaSPH/wp_densityDelta.py`'s density-
diffusion (DDT) flux normally uses the Antuono-corrected, L-renormalized
gradient `gradRhoL` (`DensityDiffusionScheme.deltaSPH`, the default) rather
than the plain kernel-sum `gradRho` (`DensityDiffusionScheme.denormalized`).
The renormalization matrix `L` comes from `detectFreeSurface`'s per-particle
covariance fit -- structurally the same kind of fit (an MLS/covariance solve
over the local neighbourhood) already found ill-conditioned at truncated/
sparse support elsewhere in this codebase (the mDBC ghost-density fallback,
`antuono-pressure-switch-bug`). Worth checking whether a bad `L` was itself
injecting the density noise that then feeds the c0²-amplified pressure
artifact (§5.19), on top of (or instead of) the Antuono pressure-switch's own
truncation behaviour (§5.14/5.15).

Same validation config as §5.17/5.19 (nx=67, `c0Ratio=40`, PST on) with only
`schemeConfig.diffusionParams.densityDiffusionTerm` switched to
`denormalized` (`scratchpad/m31_denorm_run.py`, via the same
`configureScheme`-monkeypatch pattern `probe_deltaSPHMarrone34.py` uses for
`noPenShift`). `diverged=False`, full t=2.6s.

| | renormalized (`gradRhoL`, default, §5.17/5.19) | no PST (§5.20) | **denormalized (`gradRho`)** |
|---|---|---|---|
| peak `accelMagMax`, t=[1,2.6]s | 19,632 | 25,615 | **15,853** (lowest) |
| total anomalous events, t=0-2.6s | 29,617 | 23,650 | **122,894** (4.1-5.2x more) |
| bulk-flagged share of those events | 36.5% | 38.7% | **6.8%** |
| touches free surface | 63.5% | 61.3% | **93.2%** |
| peak `kineticEnergy`, t=[1,2.6]s | 0.714 | 0.662 | **1.519** (~2.1x higher) |

**Worse overall, not better, despite the single worst spike shrinking.**
Denormalizing the DDT does knock down the peak acceleration (lowest of the
three configs tried), but at the cost of a **4-5x larger number** of
anomalous events, now overwhelmingly (93%) at the free surface rather than
split between bulk and surface, and roughly **double the sustained kinetic
energy** through the same window -- visible directly in the video
(`marrone31_denorm_clip_t1.0-2.5.mp4`): the free surface stays visibly
noisier throughout, not just at the two previously-flagged spike windows.
This is consistent with the renormalization's documented purpose (Antuono et
al. 2010/2012 -- promotes the plain Molteni-Colagrossi Laplacian to a
*bi*-Laplacian specifically so the diffusive term can reach a free surface
without eating the real hydrostatic gradient, `wp_densityDelta.py`'s own
docstring): removing it doesn't fail to help by being neutral, it measurably
degrades exactly the free-surface dissipation the renormalization exists to
provide. A smaller worst-case spike is a real effect, but comes from a
generally noisier, less-dissipated flow overall, not from fixing anything.

**Three for three now: neither PST (§5.20), the DDT renormalization
(this section), nor a lower, less physically-correct sound speed (§5.19, and
only as an artifact-hiding side effect, not a fix) address the root cause.**
All three are existing pieces of numerical machinery doing legitimate
regularising work (suppressing pairing, properly diffusing surface noise,
or -- inversely -- not being run in the regime that exposes the bug); turning
any of them off does not fix the mechanism and generally makes some other
metric worse. This continues to point at the Antuono/`sun2018` Eq. (9)
pressure switch itself (§5.14/5.15) as the place the actual fix belongs.

Scripts: `scratchpad/m31_denorm_run.py`.

### 5.22 Zeroing either dissipation term outright -- expected-null, confirms both are load-bearing, no new signal

Two more short elimination trials, flagged in advance by the user as
"probably destabilize... unlikely to really point to something" since both
terms are known stability requirements, not candidate root causes: same
validation config (nx=67, `c0Ratio=40`, PST on) to t=2.0s, with (1) density
diffusion fully off (`densityDelta=0`, not just denormalized -- §5.21 already
covers the renormalization question) and (2) artificial viscosity fully off
(`alpha=0`).

Neither hit the hard non-finite-velocity divergence check inside t=2.0s, but
both are severely degraded relative to baseline -- exactly the destabilisation
predicted, just short of a hard crash in the time tried:

| | baseline (§5.17/5.19) | no density diffusion | no artificial viscosity |
|---|---|---|---|
| peak `kineticEnergy` | 0.714 | **8.326** (11.7x) | 1.094 (1.5x) |
| peak `accelMagMax` | 19,632 | **141,906** (7.2x) | 55,085 (2.8x) |
| density range over the run | ~[0.94,1.06] | **[0.699, 1.433]** | [0.809, 1.231] |

`densityDelta=0` is the more severe of the two -- a 40%+ density excursion is
well outside anything "weakly compressible" should tolerate, effectively a
soft blow-up the hard divergence check doesn't catch inside this short a
window. Confirms both terms are load-bearing for stability, as expected;
does not implicate either as the destabilisation's root cause -- if anything,
consistent with §5.19-5.21, removing stabilising machinery makes the same
underlying artifact worse, not different.

Scripts: `scratchpad/m31_zeroterms_run.py`.

### 5.23 First candidate fix tried: renormalizing the pressure-force gradient -- makes the peak 8.8x WORSE, likely the renormalization matrix itself is ill-conditioned exactly where it would matter

With PST, the DDT's own renormalization, and both dissipation terms all
ruled out (5.20-5.22), tried an actual fix at the operator the artifact lives
in: `wp_surfaceAware.py`'s pressure-force kernel already had an unused
`useGradientRenormalization`/`Li` path (`gradw_ij = matmul(Li, gradw_ij)`,
applied before the `P_i`/`P_j` combination) -- the same kind of correction
`gradRhoL` already applies to the DDT term. Wired it up as an opt-in
`schemeConfig.pressureForceRenormalized` flag (default `False`, so every
existing case is unaffected):

- `configurations/weaklyCompressible.py`: new field + `toDict`/`fromDict`.
- `modules/pressure/surfaceAware.py`: `computePressureForceSurfaceAware`
  gained an optional `renormalizationState` parameter, forwarded to
  `computePressureSurfaceAwareWarp`.
- `schemes/deltaSPH.py`: passes the *same* `renormalizationState_` already
  computed once per step for `gradRhoL` (`detectFreeSurface`'s output) when
  the flag is set -- no new computation, reuses the existing per-step fit.

Full regression suite green, and the flag is off by default, so this is a
safe, permanent, opt-in addition regardless of the result below.

**Result: substantially worse, not better.** Same validation config
(nx=67, `c0Ratio=40`, PST on), full t=2.6s, `diverged=False`:

| | baseline (5.17/5.19) | **pressureForceRenormalized=True** |
|---|---|---|
| peak `accelMagMax`, t=[1,2.6]s | 19,632 | **172,386 (8.8x worse)** |
| peak `kineticEnergy` | 0.714 | **1.397 (~2x worse)** |
| density range | ~[0.94,1.06] | **[0.80, 1.18]** |
| total anomalous events, t=0-2.6s | 29,617 | 32,596 (~10% more, not dramatic) |
| bulk-flagged share of those events | 36.5% | 34.4% (essentially unchanged) |

The nearly-unchanged event *count* and bulk/surface *split* alongside a
massively worse peak is the tell: this isn't making the mechanism fire more
often or somewhere new, it's making a subset of the *same* events far more
extreme. The likely explanation is conditioning, not the correction's
mathematical form: `Li` comes from `detectFreeSurface`'s per-particle
covariance fit over the local neighbourhood -- the *same* sparse/disordered
neighbourhoods already identified as where this artifact lives are exactly
where that fit is most likely near-singular. Elsewhere in this codebase
(the mDBC ghost-density MLS fit) an ill-conditioned local fit is already
known to need an explicit `detFloor` fallback rather than being trusted
as-is; this pressure-force path has no such guard -- it applies whatever
`Li` `detectFreeSurface` produced unconditionally, so at the handful of
particles where that matrix is poorly conditioned, multiplying the kernel
gradient by it amplifies noise rather than correcting it.

**Not adopted -- flag stays off by default.** A follow-up worth trying
before abandoning gradient renormalization as a direction entirely: gate
`Li`'s use on `detectFreeSurface`'s own conditioning output (it already
returns `lMin`, currently unused by this call site) and fall back to the raw
gradient below some threshold, mirroring the mDBC `detFloor` pattern -- i.e.
the fix might be "renormalize, but only where the fit is trustworthy," not
"renormalize unconditionally" or "don't renormalize." Not attempted this
session.

Scripts: `scratchpad/m31_pforceRenorm_run.py`.

### 5.24 Redirected to the PST itself per the user: `ShiftProperties.correctdrhodt`/`correctdvdt` -- the Sun et al. 2019 "consistent shifting" terms this codebase already implements but has never turned on -- make it categorically worse: a hard NaN divergence, not just a worse artifact

The user pushed back on the operator-level fix attempts (5.14-5.23) and
asked to scrutinize the PST itself instead: a well-behaved PST should
*improve* particle regularity near the free surface, reducing the need for
any post-hoc renormalization, so it's worth checking whether this codebase's
PST is actually doing that job properly, and specifically flagged
`literature/`'s "consistent shifting for delta-plus SPH" paper (Sun,
Colagrossi, Marrone, Antuono, Zhang 2019, *A consistent approach to particle
shifting in the δ-Plus-SPH model*, CMAME 348:912-934,
`literature/sun2019_consistent-particle-shifting-delta-plus-sph.pdf`) as the
one to look at, versus the "Michel PST" the user believed was in current use.

**Checked what's actually configured for Marrone 3.1**, directly off the
live `schemeConfig`, not by re-reading source: `ShiftingScheme.deltaSPH`
(magnitude law: Sun et al. 2017 Eq. (7), historical 1/8 scaling) +
`ShiftingProjectionScheme.surfaceNormal` (the free-surface treatment, which
*is* Sun et al. 2019 Eqs. (20)-(21) per the code's own docstring -- not
Michel's). So the near-surface projection is already the right paper's
algorithm. But `ShiftingProjectionScheme` is only half of what Sun 2019 is
actually about.

**Reading the paper (pp. 913-918) surfaced the real gap.** Its central
contribution is not the free-surface projection (that's Section 2.4, a
secondary refinement) but Section 2.1-2.3: once particles are advected by
`u + δu` instead of `u`, the continuity equation picks up an extra
`div(ρ δu)` term (Eq. (9)) that must be included, or -- direct quote from the
paper's own highlights -- **"Unphysical drift of the solution is shown when
PST is not included in a consistent way."** The paper explicitly grades the
two new terms differently: the continuity-equation one is "of crucial
importance," while the momentum-equation counterpart (`div(u ⊗ δu)`)
"plays a minor role" and "does not seem to induce sensible differences."

Checked the actual `ShiftProperties` on the live Marrone 3.1 config:
`correctdrhodt=False`, `correctdvdt=False` -- **both off**. This codebase
already has the full machinery for both terms implemented and ready
(`systems/weaklyCompressible.py` `finalize`: `drhodt_shift` = `div(ρ·du) -
ρ·div(du)` for continuity, `dudt`+`duCross` for momentum), gated behind
these two flags, and every single run in this whole investigation
(5.17-5.23) had them both off -- i.e. every configuration tested so far,
including the "baseline," has been running the exact *inconsistent* PST
the paper's abstract warns against.

**Tested it directly** (`scratchpad/m31_consistentPST_run.py`): identical
Marrone 3.1 validation config (nx=67, `c0Ratio=40`, PST on) with
`correctdrhodt=True` and `correctdvdt=True`. Result: **a hard divergence**,
the only one of six configurations tried (baseline, no-PST, denormalized-DDT,
zero-dissipation x2, renormalized-pressure-force, this one) to actually hit
non-finite velocities rather than a bounded-but-severe artifact:

- Runs cleanly and comparably to baseline through t≈2.2s (`accelMagMax`
  staying in the same few-thousand range as baseline over that stretch).
- At t=2.25s (t*=9.10) -- **inside the same second flagged window
  (t*=8.5-9.3) already identified in §5.17** -- `maxVelocity` jumps to 16.0,
  density crashes to `[0.8655, 1.2059]`, `accelMagMax` to 33,685.
- From there it's an uncontrolled cascade, not a bounded spike: by step
  23865 (six steps later, same reported `t` at float32 resolution) density
  has gone **negative** (`minDensity = -4.2e18`), `kineticEnergy` reaches
  `2.6e14`, and step 23868 is all-NaN. `nSteps=23869` of a planned 26748;
  stopped by the runner's own non-finite-velocity check.

**Verdict: consistent shifting does not fix this, and makes the failure mode
categorically worse** -- a genuine, unrecoverable blow-up in the *same*
already-identified window, not a reduction of the existing artifact. This
doesn't mean Sun 2019's argument is wrong in general (their own benchmarks
are calmer flows without this codebase's specific truncation-artifact
background, §5.15/5.17/5.19); it means adding the continuity-equation
correction on top of a scheme that already has the symmetric-pressure-
truncation artifact live compounds rather than cancels it here -- plausibly
because `div(ρδu)` amplifies whatever density noise the artifact is already
injecting, rather than correcting a clean signal the way it would in the
paper's own (non-pathological) benchmarks.

**The `ShiftingProjectionScheme.surfaceNormal` regression flagged in the
code comments (5.20's PST discussion did not re-check this) is still an open
thread**, separate from the correctdrhodt/correctdvdt result here: a
2026-09-05 comment in `configurations/moduleConfigurations/shifting.py`
records that `surfaceNormal` (Marrone 3.1's active projection scheme) was
found to diverge early on `sloshingTank` after commit 790a7c7 fixed the
density-diffusion sign, and `sloshingTank.py` moved to
`ShiftingScheme.michel2022`/`ShiftingProjectionScheme.michel2022` instead --
but Marrone 3.1 was never re-checked against that same regression, and this
session didn't either (found while investigating this section, not chased
further). Whether swapping to the `michel2022` scheme+projection pair changes
anything on Marrone 3.1 specifically is the next concrete thing to try on
this thread, not `correctdrhodt`/`correctdvdt` again.

Scripts: `scratchpad/m31_consistentPST_run.py`.

### 5.25 `ShiftingScheme.michel2022`/`ShiftingProjectionScheme.michel2022` -- mixed: calmer in both previously-flagged windows, but a new, larger, later event appears

Tested the pairing `sloshingTank.py` moved to (5.24's closing question):
Michel et al. 2022's shift-magnitude law (a relative, Galilean-invariant
characteristic velocity, not Sun 2017 Eq. (7)'s Mach-scaled one) and its
Eq. (48) free-surface projection (inherited nearest-surface-particle normal,
not Sun 2019's own-normal Eqs. (20)-(21)). `correctdrhodt`/`correctdvdt` left
at their `False` defaults, per 5.24. Same validation config otherwise
(nx=67, `c0Ratio=40`), full t=2.6s, `diverged=False`.

| | baseline (5.17/5.19) | **michel2022/michel2022** |
|---|---|---|
| peak `accelMagMax`, t=[1,2.6]s | 19,632 | **111,475 (5.7x worse)** -- but see below |
| peak `kineticEnergy` | 0.714 | 0.859 (comparable) |
| density range | ~[0.94,1.06] | [0.89, 1.15] |
| total anomalous events, t=0-2.6s | 29,617 | **24,190 (~18% fewer)** |
| bulk-flagged share | 36.5% | 33.8% (essentially unchanged) |

**Genuinely calmer in both of §5.17's originally-flagged windows.** Sampling
`accelMagMax` through t*=5.9-6.9 and t*=8.1-9.3 (the two windows the whole
investigation has centred on), this run stays in the 1,000-5,000 range
throughout -- comparable to or better than baseline's 500-8,500 range over
the same windows, and total event count is down, not up.

**But the single worst spike of the whole t=[1,2.6]s window is a *new* event,
later and larger than anything seen before**: at t=2.47-2.52s (t*=9.98-10.15)
-- past both previously-flagged windows -- `accelMagMax` reaches 111,475,
the highest peak of any configuration tried except 5.24's outright
divergence. Per-particle trace: a spatially coherent cluster
(x drifting -0.78 -> -0.96 as t goes 2.469 -> 2.518, all `surfaceIndicators=1`
throughout), and the video frame at t=2.479s shows it as a small, sharp
splash right at the **left wall**, at bed height -- visually consistent with
a genuine second reflected-wave return impact (the front travels right,
hits the far wall, returns left, and by t*~10 is back at the origin wall),
not an obviously non-physical isolated-particle cluster like §7.4's Marrone
3.4 finding. Whether this is (a) a real secondary impact whose genuinely
sharp local dynamics get amplified by the same ever-present truncation
artifact, amplified differently because `michel2022` moved the bulk flow's
timing/shape enough to change where the second impact lands, or (b) the same
artifact simply relocated by the changed particle distribution, was not
resolved this session -- the video is consistent with either reading, and
distinguishing them needs the same pressure-sign-flip check §5.17 used, not
attempted here for lack of time.

**Net read: not a fix, but not strictly worse either -- a genuinely
different failure profile.** `michel2022` measurably calms exactly the
regions this investigation has been tracking, at the cost of a larger,
later event this investigation had not previously characterised. This is
the first tested change that improves *any* concrete metric in the
originally-flagged windows without an outright divergence (5.24) or a
board-wide degradation (5.21-5.23) -- worth pursuing further (a longer run
past t*=10 to see if a third window opens up too, and the pressure-sign
check on the new cluster to classify it), but not yet a result to adopt as
a default.

Scripts: `scratchpad/m31_michel_run.py`.

### 5.26 Double resolution + `symplecticEuler` on top of 5.25's `michel2022`/`michel2022` -- 5.25's late event converges away (~100x smaller), but a new, earlier one appears at comparable magnitude

Per the user: "the same event happens at 1.5 [in 5.25] but its kept in check
more, still not great" -- asked for double resolution (nx=134, H/dx=80.4 vs
nx=67's 40.2) with `symplecticEuler` in place of `rungeKutta4`, on top of
5.25's best-so-far `michel2022`/`michel2022` shifting. Rationale: `t*` is
resolution-independent, so if a flagged event shrinks at higher resolution
it's a genuine (if under-resolved) physical/numerical feature; if it stays
the same absolute size, that's further evidence of a resolution-independent
consistency defect. `symplecticEuler` evaluates the pressure force once per
step instead of RK4's four sub-stage evaluations, and already has a track
record on this case (5.16: full 7s sloshingTank record, better Sensor-1
match than `rungeKutta2`, once `mdbcNoPenShiftMode='finalize'` fixed its old
divergence). `scratchpad/m31_michel_hires_symplectic_run.py`: `diverged=False`,
full t=2.6s, 53,476 steps (~2x nx=67's step count, as expected from the
acoustic-CFL `dt ~ dx`), 1765s wall (symplecticEuler's single evaluation
per step roughly offset the 4x particle count from doubling resolution --
not the ~8x-longer run this was expected to need).

**5.25's worst event (t*~10) converges away almost completely.** At
t=2.47-2.50s (the same `t*` window that hit `accelMagMax=111,475` at nx=67),
this run reads **935-1,144** -- essentially two orders of magnitude smaller.
Strong evidence that specific event is a genuine, resolvable physical
feature (read in 5.25 as a second reflected-wave return impact at the origin
wall) that was simply under-resolved at nx=67, not a resolution-independent
defect.

**But the run's own worst event moved earlier, to t*~5.6, at comparable
magnitude to 5.25's original worst case.** Peak `accelMagMax` over
t=[1,2.6]s is 48,439, at t=1.386s (t*=5.60-5.61, sustained over dozens of
consecutive steps) -- a cluster near the impact wall (x~1.28-1.36,
y~-0.33 to -0.36, all `surfaceIndicators=1`), consistent with the falling/
re-impacting jet right after the first wall slam rather than either of the
two previously-tracked windows (t*=6-6.7, 8.5-9.3), whose own readings here
(1,700-5,400) are comparable to 5.25's nx=67 numbers -- neither clearly
better nor worse.

| | baseline (5.17/5.19) | michel2022 nx=67 (5.25) | **michel2022 nx=134 + symplecticEuler** |
|---|---|---|---|
| peak `accelMagMax`, t=[1,2.6]s | 19,632 | 111,475 (at t*~10) | **48,439 (at t*~5.6, new location)** |
| peak `kineticEnergy` | 0.714 | 0.859 | 0.883 |
| density range | ~[0.94,1.06] | [0.89, 1.15] | [0.85, 1.15] |

Per-particle-frame anomaly rate (normalizing for 4x the particles and 269
states either way): nx=134 run 0.0253 events per fluid-particle-frame, vs
nx=67 michel2022's raw 24,190 events over ~10,272 particles x 269 frames =
0.0087 -- **higher, not lower**, though the bulk/surface split flips
(65.9% bulk vs nx=67's 33.8%) which may partly reflect the fixed absolute
`dv > 1.0` detection threshold interacting differently with the finer `dx`
rather than a real change in character; not resolved further this session.

**Net read: genuinely informative, not a fix.** The ~100x shrinkage of
5.25's specific late event is a real, useful data point -- it says that
particular symptom is legitimate physics colliding with under-resolution,
consistent with how the rest of this investigation (5.15/5.17/5.19)
distinguishes "genuine but under-resolved" from "resolution-independent
consistency defect." But the whole exercise does not converge to a clean
result overall: the peak simply relocated to an earlier, comparably severe
event, and the two originally-tracked windows (t*=6-6.7, 8.5-9.3) did not
measurably improve. Whether the new t*~5.6 cluster is itself another
instance of genuine-but-under-resolved physics (a further doubling of
resolution would be the direct test) or the same background consistency
artifact was not determined this session.

Scripts: `scratchpad/m31_michel_hires_symplectic_run.py`.

### 5.27 The properly-scaled "full" Sun 2019 recipe diverges even sooner than 5.24's mismatched one -- the consistency corrections are ruled out on their own terms, not just on a scaling technicality

The user's screenshots of 5.26's video confirmed the same near-surface
"sudden upward push" as the fluid thins, just bounded now rather than
catastrophic -- still a surface artifact, not fixed. They then pointed out a
real gap in 5.24's test: `ShiftProperties.sun2017Eq7Shift` (literal Eq. (7)/
Eq. (13) shift magnitude) defaults to `False`, and `modules/shifting/
delta.py`'s own docstring measures the historical default at only **~0.131x**
(~1/8) of the literal value -- so 5.24 paired the continuity/momentum
consistency corrections with a shift an order of magnitude smaller than the
one those corrections were actually derived against. Worth checking whether
that mismatch, not the corrections themselves, was what blew it up.

Re-ran with all three together for the first time -- `sun2017Eq7Shift=True`,
`correctdrhodt=True`, `correctdvdt=True` -- on `ShiftingScheme.deltaSPH` /
`ShiftingProjectionScheme.surfaceNormal` (Sun 2019 on its own terms, not
mixed with 5.25/5.26's Michel results). `scratchpad/m31_fullConsistentPST_run.py`.

**Result: diverges again, and *sooner*, not later.** Clean through t=1.25s
(`accelMagMax` in the low hundreds, density held to [0.997,1.004] -- calmer
than baseline at the same times). Then at t=1.28-1.30s (**t\*=5.18-5.26,
before every window this investigation has tracked, including 5.26's new
t*~5.6 cluster**): `maxVelocity` 21.7 -> 30.1, density crashes to
`[0.75, 1.08]`, `accelMagMax` 3,840 -> 35,485 in two consecutive samples.
From there the same runaway cascade as 5.24 -- density negative, `KE` into
the billions, all-NaN by step 13,668 (t*=5.36, `nSteps=13,669` of a planned
26,748) -- reached in *roughly half* the steps 5.24 took to blow up
(13,669 vs 23,869).

**This rules out the consistency corrections on their own terms, not on a
scaling technicality.** If 5.24's failure had been an artifact of the
scale mismatch, using the shift magnitude the corrections were actually
derived for should have made it *more* stable, or at least fail later at a
comparable severity. Instead it failed categorically faster with a larger,
more "correct" `delta u` feeding the same `div(rho delta u)` /
`div(u tensor delta u)` terms -- consistent with those terms amplifying
whatever noise the pre-existing symmetric-pressure-truncation artifact
(5.14/5.15) puts into the density/velocity field in the first place, in
direct proportion to the shift magnitude: a bigger, more textbook-correct
`delta u` just means a bigger amplifier on the same underlying noise, not a
closer-to-correct physical increment.

**Where this leaves the PST thread:** two independent, internally-consistent
attempts at "the full delta+SPH recipe" (mismatched-scale 5.24, correctly-
scaled 5.27) both diverge, getting *worse* as the implementation gets
*more faithful* to the paper. Whether this points to a genuine bug in
`systems/weaklyCompressible.py`'s `drhodt_shift`/`dudt`/`duCross` warp-
operator calls (the sign/form matched the paper's Eq. (9) by inspection when
checked this session, but was not independently unit-tested against a
synthetic case the way `sun2018`'s pressure switch was in 5.15) or a real
incompatibility between Sun 2019's consistency assumption (mild `delta u`
relative to a *clean* field) and this codebase's specific truncation-
artifact background was not resolved. Either way, `correctdrhodt`/
`correctdvdt` are not a usable direction on Marrone 3.1 as it stands.
5.25/5.26's `michel2022`/`michel2022` pairing remains the only tested change
that helps at all (reduces event count and, at higher resolution, converges
away one specific late-window event) without diverging.

Scripts: `scratchpad/m31_fullConsistentPST_run.py`.

### 5.28 Found it: the Sun 2019 shift-correction terms have the SAME kernel-truncation-sum defect as the Antuono pressure switch (5.14/5.15) -- not a sign bug, the paper's own prescribed SUM discretization

The user, from 5.27's result, suspected a sign error in `drhodt_shift`/
`dudt`/`duCross` -- "it almost looks like the sign is wrong" -- and asked
for a line-by-line check against the paper.

**Hand-derivation + code inspection: no sign error.** Substituting the
quasi-Lagrangian derivative (Eq. (5)/(6)) into the plain Lagrangian
Navier-Stokes system reproduces Eq. (7) exactly, and the added terms beyond
the base continuity/momentum equations are `-rho*div(delta_u) +
div(rho*delta_u)` (continuity) and `div(u tensor delta_u) - u*div(delta_u)`
(momentum, as acceleration). Checked `warpSPHCore/coreOperations/
wp_divergence.py`'s actual kernel (not assumed): `GradientScheme.Difference`
computes `(fj-fi)`, `GradientScheme.Summation` computes `(fj+fi)`, per pair,
exactly Eq. (8)/(9)'s difference/sum conventions. `systems/
weaklyCompressible.py`'s `drhodt_shift = [Summation-div(rho*du)] -
rho*[Difference-div(du)]` and `momentum_extra = [Summation-div(u tensor du)]
- u*[Difference-div(du)]` match the paper's added terms sign-for-sign,
term-for-term. Not a bug by inspection.

**So a zero-physics synthetic probe instead** (`scratchpad/
probe_shiftCorrectionTruncation.py`, same slab/methodology as 5.15's
`probe_thinSheetPressure.py`: periodic-x, finite-y, row 0 = open top edge/
truncated support, real `buildVerletList` adjacency, no wall/mDBC/case
machinery). Uniform `rho`, `u`, `delta_u` fields -> the TRUE continuum
divergence of every added term is exactly zero *everywhere*, isolating pure
SPH-discretization artifact with zero real physics involved.

**First pass (`delta_u` tangential to the edge) read as clean -- `div(rho*
delta_u)` came back ~1e-7 (float noise) at every depth including the edge.**
That's a real result, not a null one: for a flat, translation-invariant edge
the kernel-gradient defect `B_i = sum_j V_j grad_i W_ij` has **zero
tangential component by symmetry** -- a `delta_u` aligned with the surface
picks up none of it. But PST does not shift particles tangentially; it
shifts them along the local density gradient, which right at a free surface
is the **normal** direction, into the missing-neighbour side. Re-run with
`delta_u` normal to the edge:

| depth | `div(rho*delta_u)` (true value: exactly 0) | `momentum_extra`_y (true value: 0) |
|---|---|---|
| 0 (edge) | **-6.81** | **-2.04** |
| 1 | **-2.80** | **-0.84** |
| 2 | **-0.375** | **-0.113** |
| 3+ | ~1e-6 (clean) | ~1e-8 (clean) |

Exactly 5.15's pressure-switch signature: large at the truncated edge, ~2-3
particle layers deep on a perfect lattice, decaying to float noise in the
bulk. With 0.1dx of jitter (disorder, not lattice phase -- 5.15's own
distinction) it reaches depth 3-4 and becomes seed-sensitive (`div(rho*
delta_u)` at depth 3 across three seeds: -0.433 / -0.074 / -0.055, a 6-8x
spread), the same "disorder makes it deep and unpredictable" reading 5.15
found for the pressure term.

**Root cause, precisely stated:** Sun 2019's own Eq. (9) *deliberately*
prescribes a summation form (`rho_j delta_u_j + rho_i delta_u_i`, not a
difference) for `div(rho delta_u)` and `div(u tensor delta_u)` -- for
momentum-conservation reasons, explicitly stated in the paper, the same
reason `<grad p>_i = sum_j(p_j+p_i) grad_i W_ij V_j` (Eq. (8)) uses a sum
instead of a difference. A summation-form SPH pair term is mathematically
exact only for complete, regular kernel support and produces a real,
field-value-proportional artifact wherever that support is truncated or
disordered -- 5.14/5.15's finding, now shown to apply identically here.
PST's own defining behaviour is to shift particles along the density
gradient, i.e. **normal** to a free surface -- precisely the direction this
defect is largest, and precisely where a violently-thinning free surface
(this session's whole investigation) is both truncated and disordered
essentially continuously. This is why turning the corrections on made things
worse (5.24) and why using the literal, ~8x larger Eq. (7)/(13) shift
magnitude made it fail categorically faster (5.27): a bigger `delta_u` is a
bigger amplifier on the same defect, not a closer-to-correct physical
increment. Not an implementation bug; a real consequence of the paper's own
discretization choice interacting with a background this codebase already
has (5.14/5.15), which the paper's own (much calmer) benchmark cases never
would have exposed.

**This unifies three separate findings from this session under one root
cause**: the Antuono pressure switch's symmetric branch (5.14/5.15), the
DDT's L-renormalization amplifying noise at ill-conditioned fits (5.23), and
now the PST consistency corrections -- all are summation/matrix-correction
-style SPH operators that are exact only under complete, well-conditioned
local support, and this codebase's violent free-surface regime does not
reliably provide that. `correctdrhodt`/`correctdvdt` are conclusively ruled
out as a fix direction, on their own mechanism now, not just empirically.

Scripts: `scratchpad/probe_shiftCorrectionTruncation.py`.

### 5.29 The free-surface projection (Eq. 20/21) itself checked against the paper -- two discrepancies, one matching the exact symptom this session has been chasing

Per the user: the paper should also cover the near-surface shift projection
itself (separate from 5.28's continuity/momentum corrections), and it may
not be implemented correctly either. Checked `modules/shifting/wrapper.py`'s
`ShiftingProjectionScheme.surfaceNormal` branch line-by-line against Sun
2019 §2.4 (Eqs. (18)-(21)) and its immediately-following §2.5.

**The branch structure and directionality are correct.** `outward = n . update`
(the raw shift's component along the outward normal), `restrict = inF &
(outward >= 0)` (shift points out of the fluid -> restrict to
`kappa * tangential`), else keep the full shift -- matches Eq. (20)'s two
`i in F` branches and the paper's own stated intent ("discriminates internal
particles moving towards the free surface... from those moving away... the
latter shift unconstrained... effective in preventing particle clusters on
the free surface"). `_curvatureGate` matches Eq. (21) exactly (`kappa = 0`
if any surface-neighbour's normal deviates by >= 15 deg, else 1).

**Two real discrepancies found:**

1. **Threshold value.** Eq. (20)'s hard-zero cutoff is `lambda_i < 0.55`;
   `ShiftProperties.surfaceLambdaThreshold` defaults to **0.4**. The paper
   itself caveats this isn't a universal constant ("this optimal threshold
   value depends on the kernel function, on the adopted ratio h/dx and on
   the algorithm of the free-surface correction"), so 0.4 isn't
   automatically wrong for this codebase's own kernel/h-dx choices -- but
   it's untested against the paper's own number, and a lower threshold is
   strictly *less* conservative (lets the shift compute in more
   marginally-conditioned territory than the paper's own tuning would).

2. **Missing the ghost-exclusion refinement from Sun 2019 §2.5 (the
   paragraph immediately after Eq. (21)) -- likely the more significant
   one.** Direct quote: *"the field lambda evaluated with the ghost
   particles cannot be used in (20), and it needs to be re-evaluated without
   considering the ghost particles... This numerical treatment is crucial
   for maintaining the simulation stable when thin liquid jets running on
   the solid wall occurs."* Checked `modules/surfaceDetection/wrapper.py`'s
   `computeNormals`: the renormalization matrix / `lambda` field is computed
   ONCE, via `operationMode=OperationDirection.AllToAll` (no kind
   filtering -- boundary and ghost particles included), and that same
   `lambda` is what `solveShifting` uses both for the normal (Eq. (19),
   where the paper wants ghosts included) *and* for Eq. (20)'s
   magnitude-zeroing gate (where the paper explicitly requires them
   excluded). Including ghosts makes a thin, wall-hugging layer of real
   fluid read as artificially well-supported (`lambda` inflated by the
   ghost padding), so the gate that's supposed to zero the shift there never
   fires, and shift computation proceeds exactly where the paper says it
   must not. This is not a mistuned default -- the second, ghost-excluded
   `lambda` pass the paper specifies does not appear to exist in this
   codebase at all.

**Why this is the more promising lead of the two:** the paper ties this
specific treatment to "thin liquid jets running on the solid wall" by
name -- precisely this session's recurring symptom (Marrone 3.1's thinning
free surface, §7.4's Marrone 3.4 wall-hugging ejected particles). Unlike
5.28's finding (a fundamental defect in the paper's own prescribed
discretization, not fixable by this codebase alone), this is a *specific
missing implementation piece* the paper itself already prescribes the fix
for.

**Implemented and tested (5.30 below)** -- item 1 (the threshold value)
remains untested and is cheap to A/B independently.

### 5.30 Item 2 implemented (fluid-only lambda gate) -- modest, not a fix: fewer total events, but the worst case is unchanged

Per the user: this should be simple to add, since `solveShifting` already
has the machinery to (re-)compute the free surface, and the fix is really
just "pass the right direction to the surface function." Implemented in
`modules/shifting/wrapper.py`'s `surfaceNormal` branch: a second
`computeRenormalizationMatrices` call with `operationMode=
OperationDirection.FluidToFluid` (excludes both boundary *and* ghost
kinds -- a superset of what the paper asks for, and the more conservative
reading of "is there enough genuine local fluid to trust a shift here"),
reusing the *same* adjacency already built earlier in the loop (a kind
filter evaluated per-pair inside the existing neighbour list, not a new
neighbour search) -- so the added cost is one more covariance-matrix solve
per shift iteration, not a new spatial query. `lMinGate` (fluid-only) now
feeds Eq. (20)'s magnitude gate; `n`/`lMin` (ghost-inclusive, from
`detectFreeSurface`) are untouched for Eq. (19)'s normal and the
outward/tangential direction test, matching the paper's own split. Full
regression suite green; smoke-tested clean before the full trial.

Same Marrone 3.1 validation config as the main line (nx=67, `c0Ratio=40`,
`ShiftingScheme.deltaSPH`/`ShiftingProjectionScheme.surfaceNormal`, PST on --
not Michel). `scratchpad/m31_ghostExcludedLambda_run.py`: `diverged=False`,
full t=2.6s.

| | baseline (5.17/5.19) | **fluid-only lambda gate** |
|---|---|---|
| peak `accelMagMax`, t=[1,2.6]s | 19,632 | 18,999 (~3% lower -- noise-level) |
| peak `kineticEnergy` | 0.714 | 0.857 (higher, not lower) |
| density range | ~[0.94,1.06] | [0.94, 1.07] (unchanged) |
| total anomalous events, t=0-2.6s | 29,617 | **23,459 (~21% fewer)** |
| bulk-flagged share | 36.5% | 41.4% |

**Modest, real, but not a fix.** The ~21% drop in total event count is a
genuine improvement of similar size to `michel2022`'s (5.25's 18% drop) --
this specific piece of missing paper machinery does measurably calm the
*typical* case. But the worst-case peak and the run's overall kinetic energy
are not better (KE is if anything higher), and the two originally-tracked
windows (t*=6-6.7, t*=8.5-9.3) still show the same order-of-magnitude spikes
as baseline when sampled directly (e.g. t=1.5s: 9,237; t=1.65s: 7,874) --
qualitatively indistinguishable from the unfixed run in the video. Given
this system's already-established sensitivity to tiny perturbations (a
resumed trajectory diverges sharply from its archive within one violent
window, per earlier sloshingTank work), a single run's peak is a noisy
statistic; the event-count drop, averaged over the whole t=0-2.6s record, is
the more trustworthy signal here, and it says "somewhat better on average,
not qualitatively different."

**Net position after 5.14-5.30**: three changes now show a genuine, repeatable
*reduction in event count* without diverging or regressing other metrics --
`michel2022`/`michel2022` (5.25, -18%), the fluid-only lambda gate (5.30,
-21%), and (partially) higher resolution (5.26, converges one specific event
away entirely). None individually fixes it. The two haven't been tried
together yet (`michel2022` doesn't use `ShiftingProjectionScheme.surfaceNormal`
at all, so this session's fluid-only-lambda fix has no effect under it) --
combining `michel2022`'s shift law with the fluid-only-lambda idea would
need porting the same fix into `modules/shifting/wrapper.py`'s `michel2022`
projection branch, not attempted this session.

Scripts: `scratchpad/m31_ghostExcludedLambda_run.py`.

### 5.31 Antuono-switch-removal test: forcing each branch unconditionally on Marrone 3.1 -- confirms the switch is load-bearing, not optional (2026-09-13)

Per the user, a direct test of whether the `PressureForceScheme.Antuono`
switch itself (rather than something adjacent to it, per 5.17-5.30's
exhaustive elimination) is the thing keeping Marrone 3.1 alive: added a
`--pressureForceTerm` override to `scripts/probe_deltaSPHMarrone.py`
(`conservative`/`nonConservative`/`Antuono`/... plumbed straight to
`schemeConfig.pressureForceTerm` via the same
`dambreakCase.configureScheme` monkeypatch pattern `--noPenShift` already
uses) and ran the identical §5.17 config (nx=67, plain `deltaSPH`, PST on,
tLimit=5.0s, t*≈20.2) twice, forcing each of Eq. (9)'s two branches
unconditionally instead of switching between them:

| forced term | diverged (flag) | density range | final maxVel | final maxDensity | KE-increase events (>5%/step) |
|---|---|---|---|---|---|
| `nonConservative` (always `P_i+P_j`, the switch's "symmetric" branch) | False | [0.935, 1.062] -- physical | -- | -- | 229 clustered events over the whole run |
| `conservative` (always `P_j-P_i`, the switch's "antisymmetric" branch) | **False (flag missed it)** | **[0.05, 8931]** | **976 m/s** | **3889x rest density** | 3 clustered (all in the first 0.03s) |

**`conservative` alone is not a mild degradation, it is instantaneous
catastrophic tensile instability** -- the textbook SPH failure mode Sun 2018
Eq. (9) exists to prevent, not a slow drift: `minDensity` collapses from
0.86 to 0.61 and `maxVelocity` rockets from 17 to 976 m/s within t*=0.03-0.04
(the first ~40 steps), i.e. essentially at t=0, nowhere near first wall
impact (t*≈2.3-2.7). The run's own `diverged` flag never trips (it checks
for NaN/Inf, not physically-absurd magnitudes) and the run completes all
56,908 steps to t=5.0s nominally "clean" while carrying a wildly unphysical
state the whole time -- **a second, independent finding: this case's
divergence check is too weak to catch this failure mode**, filed as a gap,
not fixed here.

**`nonConservative` alone is stable for the full 5s/t*=20.2 record** (density
never leaves [0.935, 1.062], no catastrophic blowup) but reproduces 5.17's
signature continuously: 229 distinct clustered KE-increase events spread
across the whole run (vs `conservative`'s 3, all in the opening
transient) -- consistent with 5.15/5.17's diagnosis that the always-symmetric
form's spurious pressure force from a truncated/disordered kernel sum is a
persistent, bounded nuisance, not the occasional/catastrophic thing
`conservative` alone is.

**Reading:** this rules out "drop the switch entirely" as a fix in either
direction. `conservative`-only reproduces exactly the tensile instability
the switch was added to prevent (worse: instantly, not eventually) --
confirms the switch is load-bearing, not incidental. `nonConservative`-only
is survivable but is literally 5.15/5.17's artifact running unconditionally
everywhere instead of only where Eq. (9)'s condition fires it, i.e. running
*more* of the mechanism already identified as the open problem, not a
smaller amount of it. This leaves 5.17's original conclusion standing
unchanged: **a fix has to live inside the switch's symmetric branch itself
(e.g. a kernel-sum completeness correction per 5.15, or tightening when it
fires) or be accepted as an inherent `sun2018` cost** -- it is not
achievable by removing the switch, in either direction.

Runtime note: each run took ~47-57 min wall time on an RTX PRO 6000 for
5s/t*=20.2 (`nonConservative` 3420s/51,420 steps, `conservative` 2794s/56,908
steps despite finishing "faster" -- the blown-up state took smaller,
CFL-limited steps once velocities hit hundreds of m/s, hence *more* steps in
*less* wall time per step, not evidence of being closer to correct).

Script change: `scripts/probe_deltaSPHMarrone.py --pressureForceTerm
{conservative,nonConservative,Antuono,i,j,symmetric}` (new flag, kept --
useful for any future A/B on this axis). Output:
`scratchpad/m31_antuonoAblation/` (not added to `scripts/out_*`, throwaway
per-run npz + video pair, gitignored).

### 5.32 What DualSPHysics itself actually does here -- its *default* WCSPH scheme has no pressure switch at all; the switch-like mechanism it does have is gated on kernel completeness, not pressure sign (2026-09-13)

Per the user, read `~/dev/DualSPHysics/src/source/` directly (checked out locally,
per the table at the top of this doc) rather than inferring from citations.
Two separate things live in that codebase, easy to conflate:

**(1) The classic/default WCSPH momentum equation -- what actually runs
Marrone 2011 / the standard DualSPHysics dam-break example -- has NO
pressure-sign switch whatsoever.** `JSphCpu.cpp::InteractionForcesFluid`,
the `!ncpress` branch (line ~859-864, the only branch that runs unless the
newer "advanced shifting" feature below is explicitly enabled):

```cpp
const float prs = (pressp1+press[p2])/(rhop1*velrhop2.w)
  + (tker==KERNEL_Cubic ? fsph::GetKernelCubic_Tensil(...) : 0);
const float p_vpm = -prs*massp2;
acep1 += p_vpm*grad;
```

`(P_i+P_j)/(rho_i*rho_j)` is applied **unconditionally, to every particle,
every step, everywhere** -- this is exactly `PressureForceScheme.
nonConservative` in this repo, run permanently, with no antisymmetric branch
at all. The only tensile-instability handling bolted onto this is Monaghan's
artificial-pressure repulsive term (`GetKernelCubic_Tensil`, the
`od_wdeltap` constant) -- and it is compiled in **only for the Cubic
spline kernel**; for Wendland (what Marrone/Sun/DualSPHysics' own dam-break
validation and this repo's `dambreak.py` all actually use) there is
**zero** tensile-instability term in the base momentum equation. DualSPHysics
relies entirely on its density-diffusion term (DDT) and basic (Lind-style)
shifting to keep Wendland-kernel WCSPH stable -- not a pressure-force
switch of any kind.

**This directly reframes §5.31's ablation.** `--pressureForceTerm
nonConservative` (always `P_i+P_j`, survived the full 5s record but carries
5.15/5.17's truncation artifact continuously) is not some degraded special
case -- **it is a faithful reproduction of what DualSPHysics itself runs by
default**, unconditionally, for its own Marrone-class validation. The
`Antuono` switch (this repo's current `pressureForceTerm` default, sun2018
Eq. 9's pressure-sign test) is not what makes DualSPHysics' baseline stable;
it isn't in DualSPHysics' baseline at all.

**(2) DualSPHysics DOES have a switch-like mechanism, but only in its newer,
optional "advanced shifting" / ALE-SPH extension** (`vs_advshift`, config
flag `ncpress`, off by default -- a Vacondio/Fourtakas-lineage consistent-
shifting scheme, a relative of `literature/sun2019_consistent-particle-
shifting-delta-plus-sph.pdf`, i.e. the same paper family
[[marrone31-truncation-artifact-vs-pst]] already traced this repo's
`correctdrhodt`/`correctdvdt` flags to). When enabled, `InteractionForcesFluid`
computes and stores **both** pressure forms every step without adding either
to the acceleration yet (`presssym[p1] = (P_i+P_j)/(rho_i rho_j)` sum,
`pressasym[p1] = (P_j-P_i)/(rho_i rho_j)` sum, plus a gradient-renormalization
matrix `lcorr[p1]` accumulated alongside them, same shape as this repo's own
`Li`/`useGradientRenormalization`). The actual blend happens once, after
the neighbour loop closes, gated on **kernel-sum completeness**, not on
pressure sign (`JSphCpu.cpp` ~1053-1078):

```cpp
poup1 += W(0)*m/rho;                  // Shepard-sum self term ("partition of unity")
if(fstype[p1]==0 && poup1>0.95){      // bulk classification AND ~complete kernel support
    acep1 += inverse(lcorr) * pressasymp1;   // renormalized ANTIsymmetric (conservative) form
}else{                                 // free surface (any fstype>0) OR incomplete support
    acep1 += presssymp1;               // raw, UNRENORMALIZED symmetric form
}
```

`fstype` itself (0=bulk,1=surface,2=candidate,3=isolated) comes from a
Marrone-style **divergence-of-position** test (`JSphCpu_preloop.cpp`
`InteractionComputeFSNormals`: `fs_treshold = -sum_j V_j (dr . gradW)`,
which equals the space dimension for a complete, regular neighbourhood and
drops below it exactly where support truncates), refined by a
neighbour-density ratio for the "isolated" tier -- a different free-surface
detector from `warpSPH`'s resolved default here (`SurfaceDetectionScheme.
Barecasco`, an angle-based coverage-vector test -- **correction, see
§5.35**: an earlier draft of this entry said `ColorField`, misreading the
`SurfaceDetectionConfig` dataclass's own per-field default instead of
`buildDefaultSurfaceDetectionConfig()`, the factory actually used to
construct it; `Barecasco` was never wired to gate the pressure term in
DualSPHysics either way, only PST).

**Reading, and how it changes the open Marrone 3.1 problem's shape:**
DualSPHysics' own switch-like mechanism switches on the exact quantity
[[sph-symmetric-pressure-truncation-artifact]] already root-caused this
repo's artifact to -- kernel-sum completeness -- not on pressure sign the
way `sun2018` Eq. (9) (and this repo's `Antuono` scheme) does. And critically,
it only trusts the *sharper* antisymmetric/conservative gradient where the
renormalization matrix is well-conditioned (`poup1>0.95` in the *bulk*,
`fstype==0`); everywhere else -- including every free-surface particle --
it falls back to the plain, unrenormalized symmetric sum, same as this
repo's `nonConservative` branch. That is precisely §5.15/§5.17's original
candidate direction (a) ("a kernel-sum completeness correction to the
symmetric term") and is *also* why this repo's own attempt 4
([[marrone31-truncation-artifact-vs-pst]], "pressure-force renormalization
made the peak 8.8x WORSE") failed the way it did: attempt 4 applied
`Li`-renormalization to the pressure gradient **unconditionally**, with no
completeness gate, right where `Li` is most ill-conditioned; DualSPHysics
gates the exact same idea behind `poup1>0.95 && fstype==0` for that reason.
**Not implemented here -- this is a concrete, already-proven-elsewhere
design for the fix §5.15/§5.30 said had to live inside the switch itself,
not a new hypothesis.** Next step, if picked up: port a `poup1`-equivalent
(Shepard-sum completeness, already computable from the existing
`Li`/renormalization machinery) as the gate on `pressureForceRenormalized`,
rather than applying it unconditionally or gating on `mask_i`/pressure sign.

See [[antuono-pressure-switch-bug]],
[[marrone31-truncation-artifact-vs-pst]],
[[sph-symmetric-pressure-truncation-artifact]].

### 5.33 DualSPHysics' two shifting implementations, and how each treats the free surface (2026-09-13)

Per the user, continuing §5.32's source read into shifting specifically.
DualSPHysics ships **two separate, independently-selected** shifting
objects, not one scheme with options:

**(1) `JSphShifting` -- the classic Lind et al. (2012)/Skillen et al. (2013)
shift, the one actually used unless advanced shifting is explicitly
enabled.** Direction and free-surface signal are both accumulated inline
during the *same* main force loop, no separate pass:
`JSphCpu.cpp` ~942-949 (the `shift` block already quoted in §5.32's
neighbour loop) does, per neighbour:

```cpp
shiftposfs.xyz += (m_j/rho_j) * gradW_ij;             // raw concentration gradient, unrenormalized
shiftposfs.w   -= (m_j/rho_j) * (dr . gradW_ij);      // divergence-of-position, same quantity as §5.32's fs_treshold
```

`JSphShifting::RunCpu` (`JSphShifting.cpp` ~390-420) then turns that into a
displacement once per step:

```cpp
umagn = dt * ShiftCoef * KernelH * |v_p1|;
if (ShiftTFS) {                          // free-surface threshold, OFF (0) by default
    if (rs.w < ShiftTFS) umagn = 0;                          // full cancellation, all 3 components
    else umagn *= (rs.w - ShiftTFS) / ((dim) - ShiftTFS);    // linear ramp back up to full strength
}
shift = clamp(rs.xyz * umagn, +/-0.1*Dp);
```

Three things worth naming: **(a) `ShiftTFS` defaults to 0 (disabled) in the
library itself** -- `sxml->ReadElementFloat(lis,"fsthreshold","value",true,0)`
-- so unless a case's XML explicitly sets `fsthreshold` (Lind's own
recommended 1.5 in 2D / 2.75 in 3D, the exact numbers reappearing as
literal constants in §5.32's ALE detector), classic DualSPHysics shifting
has **no free-surface awareness at all** and shifts every fluid particle at
full strength off the raw, unrenormalized gradient -- boundary-adjacency is
the only thing that can zero it (`SHIFT_NoBound`/`SHIFT_NoFixed`, a
sentinel `FLT_MAX` set inline, unrelated to `ShiftTFS`). **(b) When it *is*
enabled, the treatment is a single SCALAR gate on the whole displacement
vector** -- either shift at (a ramped fraction of) full strength in the raw
gradient direction, or don't shift at all. There is no directional
correction: it never tries to keep a tangential component while dropping
only the outward-normal one. **(c) No gradient renormalization anywhere in
this path** -- `gradW_ij` is used raw, same truncation exposure as the
`nonConservative` pressure branch (§5.31/5.32) and the pressure-force
artifact this plan is chasing, just for a different quantity (displacement,
not force).

**(2) The "advanced shifting" ALE extension (`ncpress`/`aleform`, off by
default, §5.32's other finding) treats the free surface completely
differently -- and matches this repo's own `surfaceNormal` idea.**
`JSphCpu_preloop.cpp::ComputeShiftingVel` (~448-500), using the 4-tier
`fstype` classification (0 bulk / 1 surface / 2 candidate / 3 isolated)
from a dedicated normal-and-divergence pass (`InteractionComputeFSNormals`,
already described in §5.32):

```cpp
theta = clamp((fsmindist - KernelSize) / (0.5*KernelSize - KernelSize), 0, 1);  // 0 far from FS, ->1 within half a kernel support
if (fstype==0)         shift_final = shift;                              // bulk: untouched
else if (fstype==1||2) shift_final = ale ? (shift - theta*n*(n.shift))   // project OUT only the along-normal component
                                          : 0;                            // (non-ALE sub-variant: full cancel instead)
else /* fstype==3 */   shift_final = 0;                                  // isolated particle: no shift, full stop
```

This -- keep the tangential component, remove only the outward-normal one,
ramped smoothly by distance to the free surface -- is directionally the
same correction this repo's `ShiftingProjectionScheme.surfaceNormal`
implements (Sun et al. 2019, [[wcsph-shifting-free-surface]]), not the
classic scheme's blunt all-or-nothing scalar gate. Isolated particles
(`fstype==3`) getting an unconditional zero-shift is also a distinct,
sharper rule this repo doesn't currently have an equivalent for.

**Reading:** the two DualSPHysics shifting implementations sit on either
side of this repo's own default. Our default (`surfaceNormal`, directional
projection, distance-ramped) is architecturally the ALE/advanced scheme's
answer, not the classic scheme's -- consistent with [[wcsph-shifting-free-surface]]
already citing Sun 2019 as the source. The classic scheme's binary
scalar-gate approach was not tried here and is a strictly cruder fallback,
not a candidate replacement. **This does not change §5.20's finding that
shifting/PST is not the cause of the Marrone 3.1 pressure artifact** -- that
was tested directly (no-PST run, destabilisation persisted) -- but it does
show DualSPHysics keeps its free-surface *shift* correction (directional,
distance-ramped) and its free-surface *pressure* correction (§5.32's
completeness-gated symmetric/antisymmetric blend) as two separately-tuned
instruments built on the same underlying classification, whereas this
repo's shift correction (`surfaceNormal`) and pressure correction
(`Antuono`, sign-gated) currently read *different* free-surface signals
(the shifting module's own renormalization-based normal/lambda fit vs.
`detectFreeSurface`'s `Barecasco` mask, per §5.35's correction) that are not
guaranteed to agree particle-by-particle -- a structural inconsistency
worth keeping in mind alongside [[sph-symmetric-pressure-truncation-artifact]]'s
"disorder makes it seed-sensitive" observation, though not established here
as a contributing cause, just noted. **§5.35 makes this concrete and much
more serious than "not guaranteed to agree": the mask itself is shown to
flip for the same particle within a single real timestep.**

See [[wcsph-shifting-free-surface]], [[marrone31-truncation-artifact-vs-pst]].

### 5.34 What diffSPH does here -- per the user's observation that it "never had these issues to this extent." Reading `~/dev/diffSPH` directly (the local reference this repo's DDT was ported from) turns up three concrete differences, one of them a strong candidate (2026-09-13)

**(1) diffSPH's `Antuono` pressure switch is the same equation, already bug-free -- not a differentiator.** `modules/pressureForce.py::computePressureForce`:
`switch = (p_i>=0) | surfaceMask` -- no `p_j>=0` clause, i.e. it never had
§5.14's extra-branch bug this repo had to fix. Same raw (unrenormalized)
kernel gradient otherwise, same lack of any completeness gate on the
pressure term itself. Not the explanation.

**(2) Free-surface *detector* -- correction, not actually a difference (see
§5.35).** diffSPH's `solveShifting`/pressure path defaults to
`surfaceDetection: 'Barecasco'` (angle-based coverage-vector test, Barecasco
et al. 2013). This entry originally reported this repo's own default as
`ColorField`, misread from `SurfaceDetectionConfig`'s bare dataclass field
default rather than `buildDefaultSurfaceDetectionConfig()`, the factory
actually used to build it -- **verified empirically (§5.35): this repo's
Marrone 3.1 case *also* resolves to `SurfaceDetectionScheme.Barecasco`**,
`barecascoThreshold=pi/3`, matching diffSPH's own value exactly. Both
codebases already run the identical detector choice here; this is not a
difference at all, let alone the explanation.

**(3) Free-surface *shifting* treatment is meaningfully more conservative in
diffSPH -- the strongest candidate.** `modules/particleShifting.py::solveShifting`,
default `projectionScheme='mat'` (matches `examples/weaklyCompressible/
scripts/15_damBreak.py`'s actual run, no override):

```python
M = I - n(x)n                              # tangential projector, same idea as this repo's surfaceNormal
update[fsMask>0.5]  = (M @ update)[fsMask>0.5]   # project OUT only the along-normal component (wide/expanded FS band)
update[lMin<0.4]    = 0                          # ADDITIONAL: zero outright wherever the local renorm/covariance
                                                  #   matrix's smallest eigenvalue says the support is too
                                                  #   ill-conditioned to trust ANY shift direction
update[fs>0.5]       *= surfaceScaling            # ADDITIONAL: uniformly damp to 10% (default) at the (narrower,
                                                  #   literal) free-surface flag, even after tangential projection
```

This is a three-layer defense this repo's `surfaceNormal` doesn't have:
this repo does the tangential projection (layer 1) but has no
`lMin`-style hard conditioning gate on the shift itself (§5.30's fluid-only
lambda gate is a related but different thing -- it gates Eq (20)'s
*magnitude* threshold, not a direct zero-below-threshold cutoff on the final
displacement), and has no analogue of `surfaceScaling`'s **unconditional 10x
damping** right at the free surface on top of the projection. Less PST-driven
motion at/near the surface means less disorder injected there per step,
which is the direct input to [[sph-symmetric-pressure-truncation-artifact]]'s
mechanism (disorder, not lattice phase, is what makes the truncation defect
"deep and unpredictable") -- a materially calmer free surface from shifting
alone would show up as fewer/smaller pressure-force spikes downstream, with
no change needed to the pressure switch itself.

**(4) Likely also relevant, unverified by an actual run: diffSPH's own
dam-break demo may not run at Marrone's weak-compressibility spec at all.**
`examples/weaklyCompressible/scripts/15_damBreak.py` back-solves its sound
speed from a small *fixed* CFL timestep --
`c_s = 0.3 * volumeToSupport(dx^2, targetNeighbors, 2) / Kernel_Scale(kernel,2) / targetDt`
with `targetDt=0.0005` -- rather than targeting a physical Mach number the
way this repo's Marrone 3.1 validation does (`c0 = c0Ratio * sqrt(g H)`,
`c0Ratio=40`). A hand calculation from that formula (`L=2, nx=64 -> dx=1/32`,
`n_h=4 -> targetNeighbors~50 -> support~4dx`) gives `c_s` corresponding to
an effective `c0Ratio` of roughly **8**, not 40 -- and this repo's own
§5.19 already measured a **17x drop** in peak spurious acceleration between
those exact two `c0Ratio` values on the identical Marrone 3.1 config. If
that estimate holds, diffSPH's own dam-break demo is likely running well
below the compressibility level that exposes this artifact at visible
scale, for the same reason `examples/weaklyCompressible/12-dambreak.py`
(this repo's own legacy back-solved-c0 gallery case) shows none of it
either. **Not confirmed by an actual diffSPH run this session** (its
package has an unrelated circular-import bug when importing submodules
directly outside its own top-level import order, blocking a quick script) --
flagged as the most likely single largest factor, but the shifting
difference (3) is the one directly verified by source inspection.

**Net reading:** nothing here suggests this repo's `Antuono` switch or free
surface *detector* choice is the problem (diffSPH runs materially the same
switch, already bug-free, over a different-but-comparable detector). The
concrete, verified difference is shifting's near-surface treatment being
markedly more conservative (conditioning gate + 10x damping neither this
repo nor DualSPHysics' explicit shift has) plus a plausible, unverified
difference in how hard diffSPH's own dam-break example is actually pushed on
compressibility. Neither is a fix for the pressure-switch mechanism itself
(§5.32's DualSPHysics-derived completeness-gate design is still the
concrete next step for that); (3) is a second, independent lever -- damping
`surfaceNormal`'s near-surface shift the way diffSPH does -- worth an A/B on
its own, since it touches the *input* disorder rather than the pressure
operator's *response* to it.

See [[marrone31-truncation-artifact-vs-pst]], [[wcsph-shifting-free-surface]],
[[sph-symmetric-pressure-truncation-artifact]], [[antuono-pressure-switch-bug]].

### 5.35 The user's hypothesis confirmed and quantified, at zero extra compute cost: the Antuono switch's free-surface flag is not just noisy, it visibly flips within a single RK4 timestep, and its branch changes for 15-60% of ALL fluid particles every ~0.01s throughout the ENTIRE run -- not only during the flagged anomaly windows (2026-09-13)

**First, a correction this entry depends on.** §5.32/§5.34 both stated this
repo's default free-surface detector is `SurfaceDetectionScheme.ColorField`.
That was wrong -- it read `SurfaceDetectionConfig`'s bare dataclass field
default, not `buildDefaultSurfaceDetectionConfig()` (the factory the config
actually calls via `field(default_factory=...)` in `configurations/
weaklyCompressible.py`), which sets `scheme=Barecasco, active=False,
barecascoThreshold=pi/3`; `dambreak.py` only flips `.active` to `True` and
never touches `.scheme`. **Verified empirically**, not just by reading
source, by constructing the actual Marrone 3.1 case (`nx=67`, `scheme=
deltaSPH`, tLimit=0.002s) and reading back `r.ctx.schemeConfig.
surfaceDetectionConfig`: `scheme=Barecasco, active=True,
barecascoThreshold=1.0472 (pi/3), normalSource=LambdaGrad`. This repo has
been running `Barecasco` on Marrone 3.1 all along -- the same detector
diffSPH defaults to, with the identical threshold constant. §5.32's and
§5.34's ColorField-vs-Barecasco discussion is retracted; both entries edited
in place to point here.

**The user's hypothesis, prompted by §5.34's diffSPH read**: if surface
detection is itself unstable frame to frame, particles could intermittently
or wrongly flip in/out of the flagged set, swapping which branch of the
Antuono switch they're on -- and that swapping, not disorder alone, could be
what's driving the instability. This is directly testable on data already
on disk from §5.17's dig run (`scratchpad/m31_dig/.../trajectory/`, 271
`state_*.h5` files spanning t=0-2.6s, saved every 100 real steps) --
**each file also stores all 4 RK4 sub-stage evaluations of `pressures` and
`surfaceIndicators`for that one step**, so both intra-step and cross-frame
switch stability are checkable with no new simulation run at all
(`scratchpad/m31_switch_flicker.py`, this session).

**Finding 1 -- the mask itself is recomputed fresh every RK sub-stage, and
visibly changes within a single step.** Direct check on one file
(`state_16000.h5`, t=1.555s): `surfaceIndicators` differs between RK
stage 0 and each of stages 1/2/3 for 5 of 3240 fluid particles -- confirming
the detector is not evaluated once per real step and held fixed (the way
`freezeDiffusionAcrossStages` already does for the diffusive terms), it is
re-run at every sub-stage's intermediate, possibly-jittered position, and
the *committed* `state/surfaceIndicators` for the step is simply whatever
the *last* sub-stage happened to produce.

**Finding 2 -- across the full run, this is not rare.** Computing the full
switch condition (`P_i>=0 OR surfaceIndicators`) at every RK sub-stage for
every one of the 271 saved steps: the fraction of fluid particles whose
switch value changes *within a single real timestep* runs **1-6%
continuously throughout the whole run**, from t=0 onward -- e.g. 4.14% at
t=0.02s (nothing violent has even happened yet), 5.90% at t=1.361s, 5.12% at
t=2.080s. This is a steady background rate, not something that switches on
only near the previously-flagged anomaly windows (t*~6-9, i.e. t~1.5-2.3s).

**Finding 3 -- across saved frames (~100 steps, ~0.01s apart), the
committed switch branch changes for a striking fraction of the whole fluid
domain, continuously.** Same computation using each file's final committed
`pressures`/`surfaceIndicators`, matched by UID: the fraction of fluid
particles that flip branch between consecutive saved frames runs
**15-60% for the entire run**, e.g. 22.3% at t=0.068s (still just the dam
starting to fall), 43.8% at t=1.497s (the reported onset), 57.7% at
t=2.177s (deep into the previously-reported window). This is not a
sometimes-noisy detector occasionally misfiring near a hard case -- a large
fraction of the fluid domain is swapping which term of Eq. (9) it evaluates
every ~0.01s, for the entire simulation, calm periods included.

**Finding 4 -- particles with a large pressure jump between frames are
disproportionately (not exclusively) the ones that flipped branch.**
`P(flip | |deltaP|>50)` vs `P(flip | |deltaP|<=50)` across the run: mostly
1.5-3x higher conditional on a big jump (e.g. t=2.177s: 80.8% vs 36.1%;
t=1.555s: 51.8% vs 47.8% -- closer together deep in the violent window,
where nearly everything is flipping anyway). Read with the obvious caveat
that switching which pressure-difference formula a particle evaluates can
itself directly cause a pressure jump, so this correlation doesn't by
itself distinguish cause from effect -- but it is consistent with, and does
not rule out, the user's mechanism.

**Reading.** This substantially reframes the artifact's shape. §5.15/§5.17
characterised the failure as *disorder at a truncated kernel support making
the symmetric branch's residual nonzero and unpredictable* -- correct, but
incomplete: the mechanism making a given particle *evaluate* the symmetric
branch at all is itself unstable at both the sub-step and multi-step
timescale, for a large, steady fraction of the whole fluid domain,
independent of whether that particle is anywhere near the violent window
this plan has been focused on. A particle whose classification and/or
`P_i>=0` sign genuinely toggles every step or two is not applying a
consistently-conservative or consistently-symmetric pressure force at
all -- it is applying whichever the last evaluation happened to land on,
which is a form of numerical noise injection on top of (and possibly
compounding) the truncation-sum residual itself.

**Concrete next steps, not implemented this session, roughly cheapest-first:**
1. **Freeze the free-surface mask (and arguably the switch decision itself)
   across RK sub-stages** -- evaluate `detectFreeSurface` once per real step
   at the committed t^n state, matching the existing
   `freezeDiffusionAcrossStages` pattern (`configurations/
   weaklyCompressible.py`), rather than recomputing it fresh (at a jittered
   intermediate position) every sub-stage. Cheap, mechanical, directly
   targets Finding 1; does not address the larger cross-frame (Finding 3)
   churn, which reflects genuine per-step evolution, not just intra-step
   recomputation noise.
2. **Replace the hard binary switch with a continuous blend**, echoing both
   reference codebases already read this session: DualSPHysics' ALE
   extension smoothly ramps its shift correction by `theta` (distance to the
   surface) and hard-gates its pressure blend on a Shepard-sum completeness
   scalar (§5.32) rather than a binary classification; diffSPH ramps its own
   shift's damping continuously via `surfaceScaling`/`lMin` (§5.34). A
   pressure term that blends symmetric and antisymmetric forms by a
   continuous function of (pressure sign, completeness) instead of an `OR`
   over two booleans would not have a discrete branch to chatter across in
   the first place -- addresses Findings 2 and 3 directly, not just 1, and
   is the more likely actual fix; bigger change than (1).
3. **Quantify how much of Findings 2/3 is `surfaceIndicators` re-flagging
   vs. genuine `P_i` sign crossing.** Not separated in this pass (both feed
   the same `OR`); re-run `m31_switch_flicker.py`'s Part A/B with the two
   contributions counted separately to know how much of the chatter (1) can
   fix outright vs. how much is inherent to the pressure-sign clause of Eq.
   (9) and needs (2).

Neither (1) nor (2) has been tried; this entry stops at diagnosis, per the
user's request to confirm the hypothesis first. Script:
`scratchpad/m31_switch_flicker.py` (throwaway, reused §5.17's existing dig
data, no new run).

See [[antuono-pressure-switch-bug]], [[sph-symmetric-pressure-truncation-artifact]],
[[marrone31-truncation-artifact-vs-pst]].

### 5.36 Found it (likely): Barecasco's own magnitude gate is checking the wrong vector -- it tests the norm of an already-normalized direction (~1 always), not the raw cover vector's relative magnitude, so the "isotropic neighbourhood -> definitely bulk" safety net never fires. Confirmed on real data: flip particles measurably weaker-signal than stable ones (2026-09-13)

Per the user: rather than paper over the chatter (§5.35's freeze-mask/continuous-blend proposals), debug the actual surface-detection code paths directly -- Barecasco's own kernel, and the lambda/renormalization-derived normal.

**`NormalSource.LambdaGrad` (this case's other resolved default, alongside
`scheme=Barecasco`) is a dead end for this specific bug -- ruled out by
reading `wrapper.py::detectFreeSurface`, not by running anything.**
`normalSource` only ever changes which vector gets attached as `normals`
(consumed downstream by PST/shifting); the boolean/scalar flag that feeds
the Antuono switch (`fsm`, later dilated once by
`expansionIterations=1` into `surfaceIndicators`) comes **strictly** from
`surfaceConfig.scheme` (`Barecasco`), independent of `normalSource`. Also:
`computeNormalsLambdaGrad` builds its normal from the SPH *gradient of the
scalar* `min(|eigenvalues|)`, not from an eigenvector directly, so it
sidesteps the classic eigenvector-sign-ambiguity bug that would otherwise
be the obvious first suspect in "lambda terms and their derivation." Not
where this bug lives.

**Barecasco's own magnitude gate is where it lives.** Both `~/dev/diffSPH`'s
reference (`modules/surfaceDetection.py::detectFreeSurfaceBarecasco`, the
literal source `wp_barecasco.py`'s own header comment copies) and this
repo's warp kernel (`wp_barecasco.py`,
`computeBarecascoSurfaceDetection_Func_i_second`) implement Barecasco et
al. (2013)'s two-part interior test identically:

```python
coverVector = sum_j( x_ij / |x_ij| )          # (negated in diffSPH, unnegated here -- a pure sign
                                               #  convention difference, doesn't affect this bug)
normalizedCoverVector = normalize(coverVector)
# a particle is INTERIOR (not surface) if, for ANY neighbour j:
#   (a) j falls within `threshold/2` of the cover-vector direction, OR
#   (b) |normalizedCoverVector| <= 0.5
```

**(b) is checking the norm of an already-unit-normalized vector.**
`torch.nn.functional.normalize`/`wp.normalize` return a vector of norm
(very close to) exactly **1.0** for any input whose magnitude is
meaningfully above float epsilon -- which is every real cover vector a
simulation will ever produce, isotropic bulk particle or not. So condition
(b) is, in practice, **only ever true in the literal machine-epsilon-exact-
zero case** -- it does not implement "the directional imbalance is too weak
to trust," which is what Barecasco's own magnitude criterion is for and
what the variable name `normalized` (rather than `coverVector`) should have
been the tell. The check should almost certainly be on the **raw**
`coverVector`'s magnitude (typically relative to neighbour count, i.e. how
close the vector sum is to fully cancelling for an isotropic neighbourhood)
-- not on a vector `normalize()` has already rescaled to 1. With (b) never
firing, EVERY particle's classification rests entirely on the single-neighbour
angular cone test (a), including particles whose cover vector is pure
positional noise with no real directional signal -- and that test's outcome
is inherently unstable under a tiny position perturbation (a marginal
neighbour a few degrees from the `threshold/2` boundary crosses it on the
next RK sub-stage or timestep).

**Confirmed directly on §5.17's existing dig data**
(`scratchpad/m31_barecasco_magnitude.py`, reusing the same per-step
adjacency + positions already stored, no new run): recomputed the raw,
unnormalized cover-vector magnitude per particle per frame and compared it
against §5.35's flip/stable labels and the stored `surfaceIndicators`.

- **Particles that flip their Antuono branch between saved frames have a
  ~40% smaller relative cover-vector magnitude than particles that don't**
  (median `|coverVector|/nNeighbours`: **0.113 (flipped)** vs. **0.185
  (stable)**, n=276,720 / 591,600 fluid-particle-frame samples across the
  whole t=0-2.6s run) -- chatter concentrates exactly where the directional
  signal is weakest, i.e. where a working magnitude gate should have
  overridden the noisy angular test and called it bulk.
- **The "flagged surface" and "flagged bulk" populations heavily overlap in
  cover-vector magnitude** (sampled every 20th frame): `fs==1` median
  relative magnitude 0.302, `fs==0` median 0.104, but `fs==0`'s own 90th
  percentile (0.219) sits well inside `fs==1`'s populated range -- a
  large fraction of particles flagged "surface" have a directional signal
  no stronger than an ordinary bulk particle's noise floor, exactly what a
  functioning magnitude gate exists to keep out.

**Reading.** This is a plausible, mechanistic, and now numerically
supported single root cause for a large share of §5.35's chatter: not a
sign bug, not a wrong-formula bug, but a threshold check applied to the
wrong (already-rescaled) vector, silently disabling half of Barecasco's own
two-part test. It is shared between diffSPH and this repo (inherited via
the port, not introduced here) -- consistent with the user's and this
session's earlier read that diffSPH "never had these issues to this
extent": diffSPH's own dam-break example (§5.34) very likely just doesn't
run compressible enough (low effective `c0Ratio`) to turn this same latent
instability into a visible pressure-force problem, not because its
detector avoids the bug.

**Proposed fix, not implemented this session:** replace condition (b) with
a check on the **raw** `coverVector`'s magnitude (the value already computed
in `_first`/`computeBarecascoSurfaceDetection_Func_Adjacency_first`, before
`wp.normalize` discards it) relative to neighbour count -- e.g.
`|coverVector| / nNeighbours <= someThreshold` -- restoring Barecasco's
actual two-part interior test instead of running the angular test
unconditionally on every particle. This directly targets the same
mechanism as §5.35's "continuous blend" idea, but fixes it at the
detector itself rather than downstream at the pressure switch -- the two
are complementary, not alternatives (a correct magnitude gate reduces how
often the mask flips at all; a continuous blend removes the consequence of
whatever flipping remains).

**Not yet done:** implementing the fix, and re-running §5.35's flicker
analysis (and, if it substantially reduces flip rates, a full Marrone 3.1
stability run) to confirm it actually calms both the switch chatter and
the downstream pressure artifact -- this entry stops at diagnosis per the
user's request to debug the actual path before changing anything. `lambda`
term derivation itself (renormalization-matrix eigenvalues,
`computeRenormalizationMatrices`) was not found to be implicated for this
specific bug and was not separately audited beyond ruling out its role via
`wrapper.py`'s dispatch logic above.

Scripts: `scratchpad/m31_barecasco_magnitude.py` (throwaway, reused
existing dig data).

### 5.37 Checked §5.36 against the actual paper (user supplied it, `literature/1309.4290v1.pdf`) -- the proposed fix was wrong; the real finding is sharper and worse: the paper's authors themselves flag this exact instability as open and unsolved (2026-09-13)

Read the paper term-by-term against both implementations rather than
inferring further.

**Eq. (6) (cover vector) and Eq. (7) (scan-cone test) are both implemented
correctly.** `b_i = sum_j (x_i-x_j)/|x_i-x_j|` (Eq. 6) and `arccos((x_j-x_i)/
|x_j-x_i| . b_i/|b_i|) <= theta_i/2 => interior` (Eq. 7, `theta_i = pi/3` in
the paper, matching this repo's resolved `barecascoThreshold=pi/3` exactly)
-- verified sign-by-sign against `wp_barecasco.py`'s `_first`/`_second`
kernels: `computeDistanceVec` gives `x_ij = x_i-x_j`, so `_first`'s
`out += n_ij` accumulates exactly Eq. (6), and `_second`'s
`dot = wp.dot(-n_ij, coverVector)` is exactly `(x_j-x_i)/|x_j-x_i| .
b_i/|b_i|` -- Eq. (7) verbatim. diffSPH's extra `-n_ij` negation in its own
`coverVector = scatter_sum(-n_ij, ...)` is a sign convention difference in
its `x_ij` (opposite of this repo's), not a bug -- both implementations
correctly compute the same two published equations.

**§5.36's proposed fix was wrong -- the paper has no cover-vector-magnitude
gate at all, so there's nothing to "restore."** The paper's only fallback
for degenerate/sparse neighbourhoods is Eq. (8): a **raw neighbour-count**
threshold (`n_th = 4` in 2D, `15` in 3D) checked *before* the cone test --
"a particle having the number of neighbour particles greater than `n_th`
must be tested for its boundary status [i.e. run Eq. 7], otherwise it is a
boundary particle [i.e. surface, unconditionally]" (Section 3.2/3.3, Step
3). Neither diffSPH nor this repo implements Eq. (8) at all -- the
`norm(normalize(coverVector))<=0.5` clause both share is not a broken port
of Eq. (8), it is an *invented* extra condition with no equation behind it,
and (per §5.36) numerically inert besides. **Checked whether restoring the
real Eq. (8) would matter here: it would not.** Re-ran §5.36's flip/stable
comparison using neighbour count instead of cover-vector magnitude (same
data, same method): median neighbour count is **109-153** for this Marrone
3.1 discretisation, three orders of magnitude above `n_th=4` -- Eq. (8)'s
guard is aimed at genuinely degenerate cases (thin single-particle jets,
isolated clusters), not the general bulk-vs-surface call, and would never
fire in this simulation regardless of which neighbour list backs it. (Note:
the `adjacency/numNeighbors` used for this check is a padded/Verlet spatial
list, not the exact kernel-support count Barecasco's own kernel filters via
`w_ij>0` -- immaterial here since both are far above 4, but worth flagging
for anyone reusing this specific number elsewhere.)

**The real finding: Section 5 of the paper itself is titled "Correction
Method: Scan Circle" and describes precisely the instability this plan has
been chasing, as an open problem the authors never solved.** Quoting
directly: *"the cover vector of a boundary particle sometimes aims too
close to another neighbour particle. The scan cone mistakenly regarded
this boundary particle as interior... a drawback exists in the boundary
detection accuracy because of random behavior of particle distribution. A
research on correction method using a scan circle coverage is on going."*
Figure 10 illustrates exactly the failure mode: a genuine surface particle
whose nearest "gap-closing" neighbour happens to sit near the `theta_i/2`
cone boundary gets classified interior, and the paper's own remedy (a
"scan circle" refining the test) is presented as future work with no
follow-up -- this 2013 conference-proceedings paper has no known published
successor implementing it. **A hard `arccos(...) <= theta/2` boolean is, by
the original authors' own account, inherently unstable for any particle
near the classification boundary** -- exactly the population §5.36 found
flips branch (weaker/more marginal directional signal, closer to the
decision surface). This is not a porting bug to fix; it is a limitation of
the published method itself when its output is consumed as a hard
classification, which is exactly what the Antuono switch does with it.

**Revised reading, replacing §5.36's fix proposal:** implementing Eq. (8)
faithfully is still worth doing for correctness/completeness (it is a real,
free, missing piece of the paper, and may matter for the genuinely
degenerate cases this plan has separately flagged elsewhere -- Marrone
3.4's bed-corner isolated-particle ejection, [[marrone34-sharp-edge-case]]
-- population is disjoint from the general bulk/surface call, so it is a
different, narrower fix than what §5.36 hoped for), **but it will not touch
the pervasive chatter §5.35 measured.** For that, per the paper's own
diagnosis of its method, the only sound directions are the ones §5.35
already proposed independent of this reading: don't consume a boolean this
unstable-by-nature as a hard switch input (a continuous blend, or the
unfinished "scan circle" idea itself -- a continuous coverage fraction
instead of a single yes/no cone test -- would be a from-scratch
implementation of research the original paper never finished, not a
one-line fix), or stabilise it in time (freeze across RK stages, §5.35 item
1) even though that only addresses the intra-step half.

Not implemented this session (still diagnosis, per the user's request).

See [[antuono-pressure-switch-bug]], [[sph-symmetric-pressure-truncation-artifact]],
[[marrone31-truncation-artifact-vs-pst]].

# Part 7 — 2026-09-13 overnight gallery batch

An overnight run (`scratchpad/run_overnight_batch.sh`: Marrone 3.1 + 3.4 at
extended time horizons, then `scripts/render_examples.py` over the full
`examples/weaklyCompressible` gallery) surfaced five separate failures.
Marrone 3.1 is §5.17 above; the other four are here.

## 7.1 Open-flow semi-periodic BC -- FIXED

`channelFlow.openFlowCase` sets `semiPeriodic=True`, but `dambreak.
configureScheme` unconditionally zeroed `domain.periodic` unless the
(unrelated) `wallPeriodic` diagnostic knob was set -- so the case lost the
periodicity `buildDomain` had already given it, and the neighbour search
never saw the minimum-image distance across x.

**First pass wrongly also added a `postStep` hook that rewrote particle
positions back into the box every step (reverted -- flagged immediately as
wrong).** Position wrap-around is already handled internally wherever it's
needed (the compact hash map / neighbour search take `domain.periodic`
directly and do the minimum-image wrap themselves, exactly as
`13-open-flow.ipynb` -- which builds its own domain with
`buildDomainDescription(..., True, ...)` and runs correctly with no core
changes and no position rewriting anywhere) -- mutating `state.positions` to
force them into `[min, max)` was redundant at best and risked breaking
whatever *does* rely on a continuous, unwrapped trajectory.

**Actual, minimal fix (`cases/dambreak.py` `configureScheme` only):**
`buildDomain` already returns `domain.periodic` all-True for every case,
`semiPeriodic`/`fullyPeriodic` included (only `domain.min`/`max` differ
between them) -- exactly what the notebook sets by hand. `configureScheme`'s
zeroing now only fires when the case is neither `semiPeriodic` nor
`fullyPeriodic`, i.e. it leaves `buildDomain`'s value alone instead of
stomping it. No `postStep`, no position mutation.

Verified: `domain.periodic == [True, True]` after the fix (matches the
notebook), run to t=1.5s with `diverged=False`. Fluid particles legitimately
sit outside `[domain.min, domain.max]` in x at the end (322/1148) -- this is
*expected*, not a bug, once `domain.periodic` is correctly set: positions
drift unwrapped and the engine's own minimum-image handling is what makes
that correct, the same as any other periodic case in this codebase.

## 7.2 Lid-driven cavity density ramp / standing-wave amplification -- OPEN, deferred

Reported: density keeps ramping up over a long run, with standing pressure
waves that amplify rather than damp -- in a fully enclosed box (no free
surface, no outlet) this reads as a lack of dissipation for a resonant
acoustic mode.

**First-pass hypothesis (from a triage pass, not yet the conclusion below):**
`mdbcNoPenShiftMode='finalize'` (default since §5.13/e10469a) rebuilds a
corrected particle's velocity from the *start-of-step* value, discarding that
step's pressure/viscosity contribution along the corrected component -- LDC
is entirely walled with a large near-wall population, so doing this
continuously could plausibly suppress exactly the dissipation that would
damp a reflecting wave.

**Quick check with the new §5.18 instrumentation weakens this hypothesis.**
`scratchpad/ldc_dig_run.py`, nx=64 to t=4s: `nopenshiftNActive` stays in the
single digits (0-6, out of thousands of fluid particles) the entire run, with
no growth trend, while `accelMagP95` clearly grows over the same window
(0.08 at t=0.2s to 1.7-2.1 by t≈3.6-4.0s -- roughly a 20x rise) and
`densityMedian` drifts up monotonically (1.00002 -> 1.00223). If `finalize`'s
force-discarding were the energy source, `nopenshiftNActive` should track a
meaningful, probably growing, fraction of the near-wall band; instead it is
flat and negligible while the thing it's supposed to explain keeps growing.
**This does not rule the mode out** (a handful of particles firing hard,
repeatedly, at the same wall location could still matter locally -- was not
checked), but it means "just turn `finalize` off and see" is not obviously
where this goes anymore.

**Left open, per explicit instruction -- not investigated further this
session.** Next step whenever this is picked up: since `nopenshiftNActive`
is off the table as the *primary* mechanism, look first at whether the
`deltaSPH` DDT (density diffusion) coefficient / the artificial-viscosity
`alpha` are strong enough to damp an acoustic mode in a domain with no
free-surface energy sink at all -- LDC may simply need more dissipation than
a free-surface case does, independent of any mDBC mechanism. The new
`accelMag*`/`nopenshiftN*` columns (§5.18) are already available on every run
of this case with no further instrumentation needed.

## 7.3 Rotating square patch -- FIXED

`rotatingSquarePatch.py`'s own docstring already named the cause:
`poissonPressureInit=False` (the old default) leaves p to develop from zero
through a documented ~2 tω acoustic start-up transient that "drives a
spurious first-period fragmentation of the arms." At this case's
`tLimit=1.0`/`omega=4.0` (tω=4), that transient spans essentially the whole
run. `poissonPressureInit` default flipped to `True`; `_seedPoissonPressure`
already no-ops (prints, leaves p=0) for any `shape` other than `box`, so
non-box presets are unaffected. Smoke-tested (nx=48, t=0.3s, `diverged=False`).

## 7.4 Marrone 3.4 -- REOPENED: not the sharp-edge jet tip, and not primarily `nopenshift`

The overnight run's own report labelled its `max|v| = 9.9 U_max` gate
"jet-tip, report-only" -- i.e. assumed it was Marrone's known unconverged
sharp-edge singularity. **The user's direct observation of the video
contradicts that**: the fast particles come off the flat bed/left-wall
region, as isolated single particles ejected straight up along the wall, not
off the obstacle's sharp edge.

Confirmed two ways:

1. **Visually**, on two independent runs (the original overnight
   `deltaSPH_nx256_c28.3_output.mp4`, and a fresh instrumented re-run to
   t=2.65s, `scratchpad/m34_dig_run.py`): both show, at t≈2.48s, several
   particles completely detached from the bulk fluid and from each other,
   scattered at increasing heights up the left wall -- consistent with
   individual particles being left behind, isolated, as the reservoir at
   that wall drains, rather than a coherent jet.
2. **Quantitatively**, via the new §5.18 instrumentation
   (`scratchpad/m34_dig/trace.json`): `accelMagMax` is already O(3x10^4) at
   *every* step through the whole t=2.470-2.482s window, including steps
   where `nopenshiftMagMax` is negligible (e.g. step 11123, t=2.4796s,
   `maxVelocity=60.6`, `accelMagMax=31088`, `nopenshiftMagMax=0.029`,
   `nopenshiftNActive=58` but with a tiny mean of 0.0011). `nopenshift`
   itself does spike hard at a few individual steps (step 11086: 20.7; step
   11104: 35.2; step 11124: 18.3) but these look like a large *reaction* --
   the mode's own "rebuild from `v_pre`" design is exactly a velocity snap,
   so a big value there means it detected and tried to arrest an
   already-extreme velocity, not that it manufactured one from a small
   input. The velocity at that location oscillates continuously between
   ~26 and ~60 m/s for the entire 50+-step window, whether or not
   `nopenshift` fires hard on a given step.

**Read: this is the same failure family as §5.17's Marrone 3.1 artifact
(a spurious SPH pressure force at severely truncated kernel support), not a
`nopenshift` bug and not the sharp-edge jet tip.** Where §5.17's free-surface
particles were merely disordered (still O(dozens) of fluid neighbours,
producing O(500 m/s²) accelerations), these bed-corner particles are nearly
*isolated* (visually, no fluid neighbour within several dx), which is a far
more severe truncation and produces accelerations two orders of magnitude
larger (O(3x10^4-1.6x10^5 m/s²)). `nopenshift`'s occasional large correction
rides on top of this and adds its own per-step non-smoothness but is not
the root driver -- the underlying pressure-force blow-up is present with or
without it firing hard.

**Not fixed this session.** Candidate next steps (not attempted): a minimum
neighbour-count / kernel-completeness floor that suppresses the pressure
force (or forces a Shepard fallback) below some support threshold, distinct
from -- but philosophically the same fix family as -- §5.15/§5.17's open
symmetric-pressure-truncation question; or preventing a fluid particle from
becoming this isolated in the first place (a sampling / drainage issue at
the flat-wall band). `scratchpad/m34_dig_run.py` is the reproduction (nx=256,
t=2.65s, ~15-20 min including GPU contention from an unrelated local
process); the ejection reproduced at the same t≈2.48s and the same wall
location on a completely independent run, so it is not a one-off.

## 7.5 Marrone 3.1 -- see §5.17

Documented above in the existing Antuono/pressure-artifact investigation
thread, since it is a direct continuation of that mechanism rather than a
separate case-specific bug.
