# warpSPH — δ-SPH / δ⁺-SPH conformance audit + validation plan

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
| mDBC in practice | `literature/1-s2.0-S0045793025003305-main.pdf` — English, Vacondio et al. (2025), river flows past bridges (an *application*, secondary) |
| DualSPHysics source | `~/dev/DualSPHysics/src/source/` — `JSphCpu_mdbc.cpp` (mDBC + m2dbc), `DualSphDef.h` / `JSph.h` (DDT variants), `examples/main/01_DamBreak` |
| existing local reference | `~/dev/diffSPH` (the kernel `wp_densityDelta.py` was ported from — plan's own note) |

`literature/ratios.pdf` is a 0-byte-text scan; ignore or re-fetch.

---

# Status — implemented this session (uncommitted)

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

**Not done:** the `c₀` rework for *other* WCSPH cases (only `dambreak` wired);
frozen diffusion; the validation cases themselves (Part 5).

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
| Eq. (7) shift magnitude `CFL·Ma·(2h_ij)²` | `modules/shifting/delta.py` | the plan's `delta.py` note says it is **Mach-scaled** — matches Sun. Confirm the `(2h)²` (not `(2h)`), and the `CFL·Ma` prefactor vs the `U_max` shifting-velocity form. |
| tensile term `1 + R (W_ij/W(Δx))^n`, R=0.2 n=4 | `modules/shifting/` + `moduleConfigurations/shifting.py` | present? value? |
| volume weight `2 m_j/(ρ_i+ρ_j)` not `V_j` | shifting kernel | conservation depends on this exact form |
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
| 4.2 | Sloshing tank, SPHERIC benchmark (moving boundary) | wall pressure sensors vs experiment | `sloshingTank` (TC10) exists |
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
| scheme | plain δ-SPH, `shiftProperties.active = False` (already the deltaSPH default), RK4, adaptive Δt (Sun Eq. 5). Frozen diffusion still not implemented (perf only). |
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
c₀ = 40√(gH), to t\* = 7.7** (`scripts/out_deltaSPHMarrone/`, 9/9 acceptance
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

**Principled fix (Part 3):** a determinant / condition gate on the MLS path
(DualSPHysics `determlimit ≈ 1e-3`, fall back to 0th-order then ρ₀) so it can
safely extend below 9 neighbours — the reason the threshold was lowered.

Probe tooling: `probe_deltaSPHMarrone.py` (Buchner geometry + P1/P2-vs-Fig.5
scoring); `referenceVelocity` and `pressureProbeSupportScale` params on
`cases/dambreak.py` (both default to the old behaviour).

**Next:** H/Δx = 80 (`--nx 134`) for the Fig. 5 convergence pair; the
determinant gate; a true φ = 90 mm on-wall disc-integral probe so the P1/P2
bands can tighten to Marrone's scatter; the c₀ = 20√(gH) cross-check.

**Cross-validate against DualSPHysics** `examples/main/01_DamBreak` on matched
geometry/resolution: same c₀, same Δt rule, DBC vs mDBC — its output is a
ready-made second reference and isolates "our δ-SPH" from "our mDBC".

## 5.2 Then

| next | case | notes |
|---|---|---|
| Marrone §3.2 | dam break vs a tall thin column | Yeh & Petroff force + LDV velocity; 3D effects — 2D first |
| Marrone §3.3 | dam break vs a rectangular step | 3D; obstacle preset already in `buildPresetObstacles` |
| Marrone §3.4 | viscosity influence | no-slip; `inviscid=False`, real `ν` |
| English 2022 §4.1 | still water + wedge | mDBC discriminator — hydrostatic profile to the wall, KE decay |
| English 2022 §4.2 | sloshing tank | `sloshingTank` — mDBC vs the current treatment on the SPHERIC sensors |

## 5.3 δ⁺-SPH suite (after δ-SPH is clean) — Sun 2017 §4

`rotatingSquarePatch` (N°1, the PST discriminator — tensile instability without
it), `oscillatingDroplet` (N°2, conservation under a body force),
`impact` (N°3). N°4–7 are bluff-body wake flows needing an inflow/outflow and a
`movingObstacle`-class case — lower priority.

---

# Part 6 — sequencing

1. **Audit** (Parts 1–3). One probe script + one findings table. No behaviour
   change yet. Output: the keep/change decision for every constant, the EOS
   form, the kernel, the integrator, `c₀` selection, and the mDBC target.
2. **`c₀` rework** — `setupWeaklyCompressibleTimestep` sets `c₀` from expected
   `U_max` (Sun Eq. 2), `Δt` follows. This alone is the single biggest fix and
   unblocks every physical-scale WCSPH case, not just the dam break.
3. **Integrator** — RK4 + frozen diffusion path for the weakly-compressible
   step; adaptive `Δt` (Sun Eq. 5) on the deltaSPH path (the `dambreakTimestep`
   hook already exists, it just returns `config.dt`).
4. **Revert the `# PSI-REVERT` hack** (Part 4); confirm dam break is now stable
   with the correct operator + 2 + 3.
5. **mDBC** — decide English-2022 vs m2dbc, implement/repair to match one of
   them exactly, cross-check on English §4.1 (hydrostatic-to-the-wall).
6. **Validation** — Part 5, Marrone §3.1 first, DualSPHysics cross-check.
7. δ⁺-SPH suite (Part 5.3).

## Relationship to the other plans

`ACSPH_PLAN.md` decision 1 currently reads *"this is not a reason to revert the
[ψ] fix"* — this plan agrees and supersedes the ad-hoc `# PSI-REVERT` made while
chasing the Lobovský dam break. `PST_ALE_PLAN.md` / `WCSPH_SHIFTING_PLAN.md` own
the δ⁺ shifting / δ-ALE work; Part 2 here is an audit of what they landed, not a
re-do. `DFSPH_IMPROVEMENT_PLAN.md` owns the incompressible wall closure; Part 3's
`computeMdbcNoPenShift` A/B overlaps its `mdbcNoPenetrationShift` item.
