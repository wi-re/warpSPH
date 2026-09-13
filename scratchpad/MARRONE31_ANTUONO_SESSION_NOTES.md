# Marrone 3.1 / Antuono switch instability -- session resume notes (2026-09-13)

Condensed handoff for this investigation thread. Full detail, in order, is
`DELTASPH_VALIDATION_PLAN.md` §5.31-5.37 (search for those headers); this
file is the "read this first" digest, not a replacement.

Branch: `acsph-plan`. Nothing in this thread has been committed -- all
findings below are diagnosis only, no source changes made.

## Starting point

Marrone 2011 §3.1 dam-break (`scripts/probe_deltaSPHMarrone.py`, nx=67,
`c0Ratio=40`, plain `deltaSPH` scheme, PST on -- the case's own defaults)
destabilizes at the free surface around t*~6-9 (t~1.5-2.3s), previously
root-caused (§5.15/§5.17, earlier sessions) to a kernel-truncation-sum
defect in `PressureForceScheme.Antuono`'s symmetric branch. Six-plus fix
attempts before this session (PST off, DDT denorm, zero-dissipation,
pressure-force renorm, michel2022 shifting, fluid-only lambda gate) all
failed or gave only ~20% fewer events.

## What this session did, in order, and what's now confirmed

1. **§5.31 -- Removed the Antuono switch entirely, ran both branches
   unconditionally.** `--pressureForceTerm` flag added to
   `probe_deltaSPHMarrone.py`. `conservative`-only (always `P_j-P_i`):
   catastrophic instant blowup at t*~0.03, essentially at t=0. `nonConservative`-only
   (always `P_i+P_j`): survives the full 5s record but runs the known
   truncation artifact everywhere, continuously (229 events vs 3).
   **Conclusion: removing the switch is not a fix either direction; it's
   load-bearing.**

2. **§5.32/5.33 -- Read DualSPHysics' actual source**
   (`~/dev/DualSPHysics/src/source/`, checked out locally).
   - Its *default* WCSPH momentum equation has **no pressure switch at
     all** -- always `(P_i+P_j)/(rho_i rho_j)`, unconditionally, matching
     this repo's `nonConservative`. Tensile-instability handling (Monaghan's
     artificial pressure) only exists for the Cubic kernel, never for
     Wendland (what Marrone/Sun/this repo actually run).
   - Its optional, off-by-default "advanced shifting" (ALE) extension DOES
     have a switch, but gates it on **kernel-sum completeness**
     (`fstype==0 && poup1>0.95`, a Shepard-sum test), not pressure sign --
     bulk+complete uses a *renormalized* antisymmetric form, everything
     else (including every free-surface particle) uses the raw
     unrenormalized symmetric form. This is exactly candidate fix direction
     (a) from the earlier §5.15 investigation, never actually implemented.
   - Its shifting has the same split: classic default (`JSphShifting`,
     Lind 2012) uses a blunt scalar on/off gate (`ShiftTFS`, defaults to
     0/OFF in the library!); the ALE extension projects out only the
     along-normal shift component with a smooth distance ramp, same idea
     as this repo's `surfaceNormal`.

3. **§5.34 -- Read diffSPH** (`~/dev/diffSPH`, the local reference this
   repo's DDT was ported from), per the user's observation it "never had
   these issues to this extent."
   - Its Antuono switch is the same equation, already bug-free (no
     `p_j>=0` extra clause -- this repo *did* have that bug, fixed
     earlier, see [[antuono-pressure-switch-bug]] memory / §5.14).
   - Its shifting near the free surface is more conservative than this
     repo's: tangential projection (like `surfaceNormal`) PLUS a hard
     `lMin<0.4` conditioning cutoff PLUS an unconditional 10x damping
     (`surfaceScaling=0.1`) right at the surface. This repo's
     `surfaceNormal` only does the tangential projection.
   - Its own dam-break demo likely runs well below Marrone's actual
     weak-compressibility spec (back-solves `c0` from a small fixed
     `targetDt` rather than targeting a physical Mach number -- hand
     calc gives effective `c0Ratio`~8, not 40; recall §5.19's own finding
     that c0Ratio 8 vs 40 gives a 17x difference in peak spurious
     acceleration on this exact case). **Not confirmed by an actual
     diffSPH run** -- its package has an unrelated circular-import bug
     blocking a quick script.

4. **§5.35 -- Tested the user's specific hypothesis: is surface-detection
   instability itself swapping which Antuono branch particles evaluate?**
   First, a correction: this repo's Marrone 3.1 case does NOT default to
   `ColorField` (an earlier claim in §5.32/5.34, now fixed in place) -- it
   resolves to `SurfaceDetectionScheme.Barecasco`,
   `barecascoThreshold=pi/3`, verified by actually constructing the case
   and reading back `r.ctx.schemeConfig.surfaceDetectionConfig`. Same
   detector diffSPH defaults to.
   Using §5.17's existing per-particle dig dump (`scratchpad/m31_dig/
   3-dambreak_2026-09-13_08-02-37/trajectory/`, 271 `state_*.h5` files,
   t=0-2.6s, saved every 100 steps -- **each file also stores all 4 RK4
   sub-stage evaluations**, so this needed NO new simulation run):
   - The surface mask is recomputed fresh every RK sub-stage (not frozen
     like the diffusive terms are), and visibly changes within one step.
   - **1-6% of ALL fluid particles flip Antuono branch WITHIN a single
     real timestep**, continuously from t=0 (not just near the known
     anomaly windows).
   - **15-60% of ALL fluid particles flip branch between saved frames
     (~0.01s apart), for the ENTIRE run** including calm early periods
     (22% at t=0.068s, dam barely falling).
   - Particles with a big pressure jump between frames are 1.5-3x more
     likely to have flipped (could run either causal direction).
   Script: `scratchpad/m31_switch_flicker.py` (reused existing dig data).

5. **§5.36 -- First bug hypothesis (WRONG, see §5.37): the Barecasco
   kernel's own magnitude gate (`|normalize(coverVector)|<=0.5`) checks
   an already-unit-normalized vector, so it's numerically dead (fires only
   in a literal-zero edge case).** Confirmed empirically that flip
   particles have systematically weaker raw cover-vector signal than
   stable ones (~40% smaller median relative magnitude). Proposed fix:
   gate on the RAW cover-vector magnitude instead.
   Script: `scratchpad/m31_barecasco_magnitude.py`.

6. **§5.37 -- Checked §5.36 against the actual paper** (user supplied it:
   `literature/1309.4290v1.pdf`, Barecasco, Terissa & Naa 2013, "Simple
   free-surface detection in two and three-dimensional SPH solver").
   **§5.36's proposed fix was wrong -- overturned:**
   - Eq. (6) cover vector and Eq. (7) scan-cone test are BOTH implemented
     correctly in this repo (verified sign-by-sign against
     `wp_barecasco.py`). diffSPH's extra negation is a compensating
     sign-convention difference, not a bug.
   - The paper has **no cover-vector-magnitude gate at all**. Its only
     fallback is Eq. (8): a raw NEIGHBOUR-COUNT threshold (`n_th=4` in
     2D, `15` in 3D) checked BEFORE the cone test -- below it, a particle
     is surface unconditionally, no cone test run. Neither diffSPH nor
     this repo implements Eq. (8) at all; the shared `<=0.5` clause is an
     invented extra condition with no equation behind it, not a broken
     port of Eq. (8).
   - Checked whether restoring Eq. (8) would matter: **no** -- median
     neighbour count in this simulation is 109-153, three orders of
     magnitude above `n_th=4`. Eq. (8) targets genuinely degenerate cases
     (thin jets, isolated clusters -- same family as Marrone 3.4's
     bed-corner ejection, a separate open item), not the general
     bulk/surface call.
   - **The real finding: the paper's own Section 5, titled "Correction
     Method: Scan Circle," documents this EXACT instability as an open,
     unsolved problem the original authors never fixed.** Direct quote:
     "the cover vector of a boundary particle sometimes aims too close to
     another neighbour particle. The scan cone mistakenly regarded this
     boundary particle as interior... a drawback exists... because of
     random behavior of particle distribution." Their proposed "scan
     circle" refinement is presented as ongoing work; no known published
     follow-up exists. **This is not a porting bug -- it is a published,
     acknowledged limitation of a hard boolean cone test**, which this
     repo (and diffSPH) then feeds directly into a hard pressure-force
     switch -- the worst possible way to consume something the original
     authors themselves called unstable near the classification boundary.

## Where this leaves the investigation (nothing implemented yet)

The pervasive chatter (§5.35) is now understood at the mechanism level,
confirmed against the literature, not just diagnosed by inspection. Three
non-exclusive next steps, none tried:

1. **Implement Eq. (8) properly** (raw neighbour-count fallback, `n_th=4`
   in 2D) -- correctness/completeness fix, free, but per §5.37 will NOT
   touch the pervasive chatter. Might help the separate degenerate-particle
   family (isolated jets/clusters, cf. Marrone 3.4).
2. **Freeze the free-surface mask across RK sub-stages** (§5.35 item 1) --
   cheap, mechanical, same pattern as `freezeDiffusionAcrossStages`.
   Addresses only the intra-step half of the chatter (the cross-frame
   15-60% churn reflects genuine per-step evolution, not just intra-step
   recomputation noise).
3. **Replace the hard binary switch with a continuous blend** (§5.35 item
   2, reinforced by §5.37) -- echoing DualSPHysics' `poup1`-gated blend and
   diffSPH's `surfaceScaling`/`lMin` ramps, or literally finishing the
   paper's own unfinished "scan circle" idea (a continuous coverage
   fraction instead of one yes/no cone test). Most likely to actually fix
   it; the biggest implementation lift of the three.

None of (1)-(3) has been implemented or tested. The natural next action is
to pick one (probably (2) as a cheap first A/B, since it's mechanical and
would show how much of the chatter is intra-step vs genuine cross-step
churn) and re-run `m31_switch_flicker.py`'s analysis to quantify the
effect before touching a full Marrone 3.1 validation run.

## Key scripts/data (all throwaway, in `scratchpad/`, all still on disk)

- `scratchpad/m31_dig/3-dambreak_2026-09-13_08-02-37/trajectory/` -- the
  source dig data everything in this session re-used (271 `state_*.h5`,
  t=0-2.6s, per-100-step + full RK4 sub-stage detail per file). Predates
  this session (from §5.17).
- `scratchpad/m31_switch_flicker.py` -- intra-step + cross-frame Antuono
  switch flip-rate analysis (§5.35).
- `scratchpad/m31_barecasco_magnitude.py` -- raw cover-vector magnitude /
  neighbour-count vs flip-rate correlation (§5.36/5.37).
- `scratchpad/m31_antuonoAblation/` -- §5.31's two full runs (npz + video),
  `--pressureForceTerm nonConservative` / `conservative`.
- `scripts/probe_deltaSPHMarrone.py` -- gained a `--pressureForceTerm`
  CLI flag this session (kept, not throwaway).

## Reference sources read this session (not part of this repo)

- `~/dev/DualSPHysics/src/source/` -- `JSphCpu.cpp` (momentum eq.,
  `InteractionForcesFluid`), `JSphCpu_preloop.cpp` (ALE shifting, FS
  classification), `JSphShifting.cpp` (classic Lind shifting).
- `~/dev/diffSPH/src/diffSPH/` -- `modules/pressureForce.py`,
  `modules/surfaceDetection.py`, `modules/particleShifting.py`,
  `modules/shifting/implicitShifting.py`.
- `literature/1309.4290v1.pdf` -- Barecasco, Terissa & Naa (2013), the
  paper this repo's/diffSPH's `Barecasco` detector is named after.
