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
  `scripts/out_deltaSPHMarrone_gatecheck/`): previously the recorded "worst
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
checks pass** under `sun2017DeltaSPH`. `scripts/out_deltaSPHMarrone/` now
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
§5.1.1 (`scripts/out_deltaSPHMarrone_pst/`, H/Δx = 40, to t\* = 7.7):

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
- The δ⁺ Eq.(7) P1 raw trace grows a visible acoustic-oscillation envelope
  over t\* ≈ 4–5.5 (stiff-EOS ringing at c₀ = 40√(gH)); the rolling median
  scores through it, but it is larger here than at H/Δx = 40.

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

## 5.2 Then

| next | case | notes |
|---|---|---|
| Marrone §3.2 | dam break vs a tall thin column | Yeh & Petroff force + LDV velocity; 3D effects — 2D first |
| Marrone §3.3 | dam break vs a rectangular step | 3D; obstacle preset already in `buildPresetObstacles` |
| Marrone §3.4 | viscosity influence | no-slip; `inviscid=False`, real `ν` |
| English 2022 §4.1 | still water + wedge | mDBC discriminator — hydrostatic profile to the wall, KE decay |
| English 2022 §4.2 | sloshing tank | `sloshingTank` — mDBC vs the current treatment on the SPHERIC sensors |

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
within the digitisation error of reading them. `scripts/out_deltaPlusTGV/`
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
validation runs in `scripts/out_deltaPlusTGV/` — the same runs that match Sun
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
