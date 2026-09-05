# warpSPH — Artificial-Compressibility SPH (ACSPH) Implementation Plan

Target paper — `literature/decourcy2024_incompressible-delta-sph-artificial-compressibility.pdf`, bib key `decourcy2024`:

> **Incompressible δ-SPH via artificial compressibility**
> J.J. De Courcy, T.C.S. Rendall, L. Constantin, B. Titurus, J.E. Cooper
> *Computer Methods in Applied Mechanics and Engineering* **420** (2024) 116700
> `doi:10.1016/j.cma.2023.116700` — CC BY, open access. 40 pp, 85 refs.
> Precursor: De Courcy et al., SPHERIC 2023 (ref [32]).

Synced into `literature/` on 2026-09-05 together with seven of its references
(see Part 6); `python scripts/check_literature.py` passes with all eight
abstracts verified verbatim against their PDFs.

---

# Status board — read this first

**Done (2026-09-05):** steps 1–6 of Part 8. The scheme runs end to end:
`--scheme artificialCompressible` builds, the dual-time driver solves, and
`hydrostaticColumn` runs to `t = 0.94` without diverging. **Next action:
step 7, Michel et al. particle shifting** — everything left on the column
(`pairedFraction 0.065`, a residual near-wall `‖v‖_max` of 0.58) is what
shifting exists to fix, and the paper never runs a walled case without it.

## Decisions taken without you — overturn any of these if you disagree

1. **The δ-SPH `ψ` sign was wrong and is now fixed** (Part 3, "The sign
   error"). This changes the *default WCSPH scheme's* behaviour repo-wide, so
   it is the one with the widest blast radius. The evidence is not a judgement
   call — the Antuono correction must annihilate a linear field pair-by-pair
   and did the opposite — and a single-variable A/B on `sloshingTank` improves
   both the density floor and the pressure peak. **Worth reporting upstream and
   checking against diffSPH, which this kernel was ported from.**
2. **ACSPH's pressure force defaults to `nonConservative`** (the literal
   `(p_i + p_j)` of Eq. 25), not `Antuono`. The Antuono switch is a
   tensile-instability guard the paper does not use; it is one config field
   away.
3. **`hydrostaticColumn` runs non-periodic under ACSPH** (§4.4 / step 5b). The
   box is walled on every side, so periodicity buys nothing, and it breaks
   Eq. (61)'s position moment. The DFSPH branch is untouched.
4. **`noPenetrationShift` (the repo's mDBC position correction) is on by
   default** and is *not* in the paper. It stands in for the particle shifting
   the paper always has; the flag exists so the paper's literal wall treatment
   is one setting away. Re-test with it **off** once step 7 lands — if the
   shift carries the corners on its own, this should go.
5. **`approachOnly=False` was added to `computeVelocityDiffusion`** rather than
   writing a new kernel. It lifts an artificial-viscosity clamp that was making
   the physical-viscosity branch one-sided; the default is unchanged, so no
   existing scheme moves.

## Questions for the authors, all in one place

| Where | Question |
|---|---|
| §5.1 | Eq. (37) prints `ε₄ = min(0, κ₄ − ε₂)`, which makes the JST operator **vanish** in smooth flow. Standard JST is `max`. Which does the CUDA code do? |
| §5.2 | Eq. (40)'s low-storage form cannot represent Fig. 1's SSPRK3 or RK4 at all. Which is the code — Jameson coefficients, or the full tableaus? |
| §5.4 | Is `𝕍` the same set (same `𝔽`, same dilation radius) in Eq. (36) and Eq. (57)? |
| §5.5 | `U_char` per case; the `𝕍` branch of Eq. (36) being unscaled; how `β` interpolates to 1 at the surface in Eq. (57). |
| §5.6 | Eq. (46)'s `CFL_t h` is a length. Is `h` there carrying an implicit reference velocity (which is what the term *does*)? And is the absence of a body-force constraint deliberate? |
| Part 3 | The `ψ` sign error above — does their δ-SPH reference implementation have it? |

---

## Why this is worth doing

The paper's own closing claim is the reason: *"in terms of software, it is clear
any weakly compressible δ-SPH code may be transformed to ACSPH by removing the
equation of state, specifying the artificial compressibility parameter k₁ and
adding a pseudo-time loop within the existing time loop."* This repo **is** a
δ-SPH code with all of that already built. The three things ACSPH needs that
δ-SPH does not — a pressure-evolution equation instead of an EOS, a BDF2
real-time source, and an inner pseudo-time RK loop — are the only genuinely new
code. Everything else (§4 below) is a rename or a config flag away.

Strategically it gives us a **third** incompressible baseline that is
structurally unlike both existing ones: DFSPH iterates a *pressure Poisson-like*
Jacobi solve on a velocity constraint; ACSPH iterates a *differential* equation
in pseudo-time. The paper's §2 argues these are the same thing solved two ways,
which makes it a genuinely informative comparison rather than a third opinion.

---

# Part 1 — Complete equation inventory

Numbering follows the paper. Symbols: `p` pressure, `v` momentum (Lagrangian)
velocity, `ρ` density (invariant, `= ρ₀`), `V_j = m_j/ρ_j` particle volume,
`h` smoothing length, `κh` kernel support radius, `x_ij = x_i − x_j`,
`(·)_ij = (·)_i − (·)_j`, `τ` pseudo-time, `t` real time, `n` real-time index,
`m` pseudo-time index, `s` RK stage index.

## 1.1 Governing system (continuous) — Eqs. (49)–(50)

The incompressibility constraint `∇·v = 0` is converted from elliptic to
hyperbolic by adding a pseudo-time derivative, and the momentum equation gains a
matching one:

```
Dp/Dτ  = −k₁ ρ ∇·v  +  k₂ 𝒟^p                       (continuity, Eq. 50/51)
Dv/Dτ  +  Dv/Dt  = −∇p/ρ + ν∇²v + f                 (momentum,   Eq. 25)
Dx/Dτ  +  Dx/Dt  = v                                (velocity,   Eq. 26)
```

`𝒟^p` is the pressure-smoothing operator (§1.4). It is *not* an ad-hoc artificial
viscosity: §2 derives it as the divergence of the momentum residual folded into
the pressure equation (Eq. 6), which is the same construction that produces the
ISPH pressure Poisson equation (Eqs. 10–15). That derivation is the paper's main
theoretical contribution and is worth reading in full, but nothing in it needs
implementing.

**Dropped by the paper itself:** the `k₃ D(∇·v)/Dτ` term of Eqs. (9)/(22) is set
to zero ("*during experimentation this third term was found to have little
influence*"). Implement the field, default it to 0, do not spend time on it.

## 1.2 Discrete continuity — Eq. (23)

```
Dp_i/Dτ = −k₁ ρ_i Σ_j (v_j − v_i)·∇W_ij V_j  +  k₂ 𝒟^p_i
```

Standard difference-form velocity divergence. Anti-symmetric, hence
volume-conserving (Eq. 52–53 discussion).

## 1.3 Discrete momentum and velocity — Eqs. (25)–(26)

```
Dv_i/Dτ + Dv_i/Dt = −(1/ρ_i) Σ_j (p_i + p_j) ∇W_ij V_j
                    + ν K Σ_j (v_ij·x_ij)/‖x_ij‖² ∇W_ij V_j
                    + f_i

ṽ_i := Dx_i/Dτ = v_i − Dx_i/Dt                                    (Eq. 26)
```

- Pressure gradient is the symmetric `(p_i + p_j)` form. The paper notes it is
  equivalent to `Σ m_j (p_i/ρ_i² + p_j/ρ_j²) ∇W_ij` under constant mass and
  invariant density — which holds here by construction.
- Viscosity is the Monaghan–Gingold (1983) velocity Laplacian, `K = 8` in 2D,
  `K = 10` in 3D.
- `ṽ` is the pseudo-time particle velocity. **On convergence `ṽ → 0`**, and it is
  simultaneously the position-row residual and the convergence metric (§1.6).

## 1.4 Pressure smoothing operators `𝒟^p` — Eqs. (32)–(37)

Four variants, named AC-2 / AC-2L / AC-4 / AC-JST:

**AC-2** — plain Laplacian of pressure (Molteni & Colagrossi form), Eq. (32):
```
𝒟^Δ_i = Σ_j 2(p_i − p_j) (x_ij·∇W_ij)/‖x_ij‖² V_j
```
Known-bad at free surfaces: kernel truncation diffuses the surface and it cannot
hold a hydrostatic gradient (§4.1.1 confirms this, Figs. 2–4). Implement it, but
it is a negative control, not a candidate default.

**AC-2L** — renormalised bi-Laplacian (Antuono correction), Eqs. (33)–(34).
**This is the paper's working default** and the operator used in every
head-to-head against δ-SPH:
```
𝒟^ΔL_i = Σ_j 2{ (p_i − p_j) − ½(⟨∇p⟩^L_i + ⟨∇p⟩^L_j)·x_ij } x_ij/‖x_ij‖² · ∇W_ij V_j

⟨∇p⟩^L_i = −Σ_j (p_i − p_j) L_i ∇_i W_ij V_j
L_i      = −[ Σ_j (x_i − x_j) ⊗ ∇_i W_ij V_j ]⁻¹
```

**AC-4** — nested (bi-harmonic) Laplacian, Eq. (35):
```
𝒟^Δ²_i = −h² Σ_j 2(𝒟^Δ_i − 𝒟^Δ_j) (x_ij·∇W_ij)/‖x_ij‖² V_j
```
Two neighbour loops, no correction. Inherits AC-2's truncation error but weaker
(§4.1.1: slight surface-row separation, volume perturbations, KE fails to fully
settle).

**AC-JST** — Jameson–Schmidt–Turkel blend, Eqs. (36)–(37):
```
𝒟^JST_i = 𝒟^ΔL_i                        if i ∈ 𝕍  (free-surface region)
        = ε₂ 𝒟^ΔL_i + ε₄ 𝒟^Δ²_i         otherwise

ε₂ = κ₂ min(1, χ),   ε₄ = min(0, κ₄ − ε₂),   κ₂ = 0.5, κ₄ = 1/32
χ_i = Σ_j |(p_i−p_j)/(p_i+p_j)| W_ij V_j  /  Σ_j W_ij V_j
```
`𝕍` = every particle within one kernel support radius of a free-surface particle.
**Not pairwise-symmetric, therefore not locally conservative** — the paper says
so explicitly and points at Lee et al. [51] for a conservative version.
See §5.1 for the `min`/`max` problem in ε₄.

**Frozen diffusion**: computed on the first RK stage and held across stages
(Antuono [8] / Jameson [40] technique). But **re-evaluated at every dual-time
iteration** — "*the diffusive terms are evaluated at each dual-time iteration
and cannot be fixed without loss of stability*".

## 1.5 Parameters — Eq. (24)

```
β = CFL_τ · h / Δτ        k₁ = β²        k₂ = 0.1 h β
```

`β` is the pseudo-time wave speed. In finite volumes `β` is prescribed and `Δτ`
varies locally; the paper **inverts this** — `Δτ` is a fixed fraction of `Δt`
(spatially constant, to keep particle displacements smooth) and `β` is the
derived variable.

`k₂ = 0.1 h β` is deliberately the δ-SPH `δ h c₀` prefactor with `β` playing the
role of `c₀` and `δ = 0.1`. The measured stability ceiling is `k₂ = 0.2 h β`,
consistent with Antuono's linear stability analysis [7,8]; `0.1` is kept for
consistency with δ-SPH practice.

`CFL_τ` is set by the pseudo-time integrator: **0.5 / 1.0 / 1.5 for RK2 / RK3 /
RK4**.

## 1.6 Dual-time integration — Eqs. (38)–(48)

State vector and residual (Eq. 38–39):
```
u = {p, x, v}ᵀ                 I_c = diag{0, 1, 1}
Du/Dτ + I_c Du/Dt = r          r* := Du/Dτ = r − I_c Du/Dt
```
The `0` in `I_c` is the whole point: the continuity equation has no real-time
derivative, so driving `r* → 0` enforces `∇·v = 0` *at* time level `n+1`.

**Pseudo-time RK sweep** (Eq. 40):
```
u^{n+1,m+1,0} = u^{n+1,m}
u^{n+1,m+1,s} = u^{n+1,m+1,0} + α_s Δτ r*^{,n+1,m+1,s−1},   s = 1..s_RK
u^{n+1,m+1}   = u^{n+1,m+1,s_RK}
```

**BDF2 real-time source** (Eq. 41), with variable-`Δt` coefficients (Eq. 42):
```
r*^{,n+1,m+1,s−1} = (1/α_PI) [ r^{n+1,m+1,s−1} − I_c( α_t u^{n+1,m+1,0} + β_t u^n + γ_t u^{n−1} ) ]

α_t = (2Δtⁿ + Δtⁿ⁻¹) / ((Δtⁿ + Δtⁿ⁻¹) Δtⁿ)
β_t = −(Δtⁿ + Δtⁿ⁻¹) / (Δtⁿ Δtⁿ⁻¹)
γ_t = Δtⁿ / ((Δtⁿ + Δtⁿ⁻¹) Δtⁿ⁻¹)
```
(Fixed-`Δt` limit: `α_t = 1.5/Δt`, `β_t = −2/Δt`, `γ_t = 0.5/Δt`.) Note the BDF
source is evaluated at the **frozen stage-0 value** `u^{n+1,m+1,0}`, not the
current stage.

**Point-implicit source treatment** (Eqs. 43–45): `α_PI = 1 + α_s Δτ α_t`,
applied to all three equations for temporal consistency. The paper then says
**`α_PI = 1` works fine here** ("*no noticed adverse behaviour*") because
`Δτ < Δt` and `Δτ` is spatially constant. Implement it (it is one scalar), keep
it on by default, expose the switch.

**Real timestep** (Eq. 46), with growth limiting to protect BDF2 accuracy:
```
Δtⁿ = max( min( CFL_t h , CFL_t h/‖v‖_max , 0.125 h²/ν , 1.2 Δtⁿ⁻¹ ) , 0.8 Δtⁿ⁻¹ )
```
`CFL_t ≈ 0.2`. §4.3 measures a sharp accuracy cliff above `CFL_t = 0.4`
(Table 1/2: error jumps ~2.4× from 0.4 → 0.6, ~10× at 1.0). Treat 0.4 as the
hard ceiling.

**Convergence metric** (Eqs. 47–48):
```
ε_v = log10( (1 / (N U_ε)) · sqrt( Σ_i^N ‖ṽ_i‖² ) )
U_ε = max( min(‖v‖_max, U_char), ε_s ),   ε_s = 1e−5
```
Iterate until `ε_v` drops below target. Recommended targets from the paper:
**−6 for general use, −8 for violent impact** (dam break, jet impact).

> ⚠ Note the `1/N` (not `1/√N`) — this is *not* an RMS. Per-particle residual
> `v̄` gives `ε_v = log10(v̄ / (√N U_ε))`, so a fixed `ε_v` target is a
> **stricter per-particle tolerance at higher resolution** by `−½log₁₀N`. Across
> the paper's own `L/Δx = 200 → 800` sweep that is a ~0.6-decade drift. Reproduce
> it verbatim (it is what their numbers mean) but record it, and consider
> exposing a normalised variant as a non-default option.

**Butcher tableaus** (Fig. 1, image-only — transcribed here):

| RK2 (explicit midpoint) | RK3 (SSPRK3 / Shu–Osher) | RK4 (classical) |
|---|---|---|
| `c = [0, 1/2]` | `c = [0, 1, 1/2]` | `c = [0, 1/2, 1/2, 1]` |
| `A = [[0,0],[1/2,0]]` | `A = [[0,0,0],[1,0,0],[1/4,1/4,0]]` | `A = [[0,0,0,0],[1/2,0,0,0],[0,1/2,0,0],[0,0,1,0]]` |
| `b = [0, 1]` | `b = [1/6, 1/6, 2/3]` | `b = [1/6, 1/3, 1/3, 1/6]` |

See §5.2 — these tableaus and Eq. (40) are mutually inconsistent for RK3/RK4.

**§4.3 finding: higher-order pseudo-time RK buys nothing.** Accuracy is set by
the BDF2, and cost rises near-linearly with stage count. RK2 at `CFL_t = 0.2`,
`Δt/Δτ = 5` is the best cost/accuracy point in Table 2. Default to RK2.

## 1.7 Optional `ṽ` material-derivative correction — Eqs. (27)–(31)

Recasting `D̃(·)/Dτ = ∂(·)/∂τ + ṽ·∇(·)` adds advective terms (Ramachandran et
al. [29]):
```
D̃p_i/Dτ  += −p_i Σ_j (ṽ_j − ṽ_i)·∇W_ij V_j + Σ_j (p_j ṽ_j + p_i ṽ_i)·∇W_ij V_j
D̃v_i/Dτ  += Σ_j (ṽ_j ⊗ v_j + ṽ_i ⊗ v_i) ∇W_ij V_j − v_i Σ_j (ṽ_j − ṽ_i) ∇W_ij V_j
```

**The paper's conclusion is to leave these off** (§4.2): *"generally ignore the
ṽ material derivative corrections."* With them on, the minimum stable `ε_v`
degrades from −8 to −6 and residual `ṽ` leaves non-physical terms that blow up
at thin free-surface tips. The second momentum term violates both linear and
angular momentum conservation. Implement behind a flag, default off, do not
tune.

## 1.8 Particle shifting — Eqs. (55)–(58), Michel et al. [66]

```
δv*_i = 0.5 · { −U^shift_i β_i (κh) ∇C_i               if ‖β_i(κh)∇C_i‖ < 0.5 κh/Δx
              { −0.5 U^shift_i (κh/Δx) ∇C_i/‖∇C_i‖     otherwise

U^shift_i = max_j( |(v_i − v_j)·(x_i−x_j)/‖x_i−x_j‖| )
∇C_i      = Σ_j [ 1 + 0.2 (W_ij / W(Δx_i))⁴ ] ∇_i W_ij V_j
β_i       = (κh/Δx)³ in the interior, decreased to 1 for surface particles
```
Free-surface correction (Eq. 57):
```
δv_i = 0                                          if λ_i < 0.4
     = λ_i² ( δv*_i − σ_i (δv*_i·n_i) n_i )       if i ∈ 𝕍
     = δv*_i                                      otherwise

σ_i = min[ 1, max( 0, (κh − d^fs_i) / (0.5 κh) ) ]
```
`λ` = min eigenvalue of `L_i` (Eq. 34); `d^fs` = distance to nearest surface
particle.

Applied **outside** the pseudo-time loop as a displacement (Eq. 58):
```
x'_i = x_i + δv_i Δt ,   φ'_i = φ_i + ∇φ_i·(δv_i Δt)
```
§4.2 tested internal (per-pseudo-iteration, Eq. 60) vs external shifting and
**chose external**: internal shifting stalls `ṽ_δ → 0` convergence during strong
impacts and worsens volume conservation. Eq. (59) offers a BDF correction
`ṽ = v − Dx'/Dt + D(δx)/Dt` so the real-time derivative stays Lagrangian;
§4.2 found it makes little difference. Implement, default on, cheap.

The paper picked Michel et al. specifically because it has **no `c₀`/Mach
dependence** — which matters since ACSPH has no `c₀`. Our existing δ⁺ shift is
Mach-scaled (`modules/shifting/delta.py`), so this is a real gap (§4.2 below).

## 1.9 Boundary conditions — Eqs. (61)–(62), Adami et al. [68]

Fixed ghost particles. Boundary pressure by extrapolation:
```
p_b = Σ_f [ p_f + ρ_f (g − a_b)·x_bf ] W_bf V_f  /  Σ_f W_bf V_f
```
Other fields by Shepard interpolation with a mirroring condition:
```
f_b = Σ_f f_f W_bf V_f / Σ_f W_bf V_f
no-penetration:  f_b^⊥ = 2 f_p·n − f_b·n ,   f_b^∥ = f_b·τ
no-slip:         f_b^⊥ = f_b·n ,             f_b^∥ = 2 f_p·τ − f_b·τ
```
Applied per field: `δv` and `ṽ` get no-penetration (so `δv·n = 0`); `v` gets
no-penetration in the velocity divergence; the velocity Laplacian gets no-slip
or free-slip depending on the case.

Free surface: handled implicitly by the discretisation (Colagrossi et al. [69]);
detection by Marrone et al. [70] with Sun et al. [11] modifications, producing
the surface set `𝔽` and the dilated set `𝕍`.

## 1.10 Reference δ-SPH baseline — Eqs. (63)–(64)

The paper's comparison scheme is δ-ALE-SPH (Sun et al. [14] / Antuono et al.
[15]), i.e. **quasi-Lagrangian δ⁺-SPH with the shifting terms folded into the
equations of motion** — which is exactly what
`docs/historic_plans/WCSPH_SHIFTING_PLAN.md` delivered here
(`ShiftProperties.correctdrhodt` / `correctdvdt`). RK4, frozen diffusion,
`CFL_wc = 0.75`. Our existing `--scheme deltaSPH` with those flags on is the
right control.

---

# Part 2 — Every constant the paper fixes

| Symbol | Value | Where |
|---|---|---|
| `δ` (δ-SPH density diffusion) | 0.1 | Eq. (16) |
| `k₂` prefactor | `0.1 h β` (max stable `0.2 h β`) | Eq. (24) |
| `κ₂`, `κ₄` (JST) | 0.5, 1/32 | Eq. (37) |
| `CFL_τ` | 0.5 / 1.0 / 1.5 for RK2 / RK3 / RK4 | §3.1.3 |
| `CFL_t` | ~0.2; hard ceiling 0.4 | Eq. (46), Table 1–2 |
| `Δt/Δτ` | 2 (best cost/accuracy), 5–10 for accuracy | Table 1 |
| `ε_v` target | −6 general, −8 impact-heavy | §4.2, §4.4, §4.5 |
| `ε_s` | 1e−5 | Eq. (48) |
| `α_ν` (artificial viscosity) | 0.01, `ν = α_ν h c₀ / K` | §4 |
| `K` | 8 (2D), 10 (3D) | Eq. (25) |
| `λ` surface cutoff | 0.4 | Eq. (57) |
| Shift tensile term | `1 + 0.2 (W_ij/W(Δx))⁴` | Eq. (56) |
| Kernel | Wendland C2, `h/Δx = 2` | §4 |
| `α_PI` | `1 + α_s Δτ α_t`, or 1 | Eq. (41) |
| `k₃` | 0 (dropped) | §3.1 |

> ⚠ **`ν` still needs a nominal `c₀`.** The paper says ACSPH "*does not require
> the definition of c₀*" and then defines `ν = α_ν h c₀ / K` with `α_ν = 0.01`,
> using "*the same value of ν*" as the δ-SPH run. So a reference `c₀` is still
> needed as an input to fix the physical viscosity — it just never enters the
> scheme. Our config must make this explicit rather than silently reusing
> `fluid.fixedSoundSpeed` as if it were an acoustic parameter.

---

# Part 3 — What the repo already has

This is the good news: most of the paper's spatial discretisation is a config
away.

| Paper | Repo | Status |
|---|---|---|
| Eq. (25) pressure gradient `(p_i+p_j)` | [surfaceAware.py](src/warpSPH/modules/pressure/surfaceAware.py), `PressureForceScheme.Antuono` | ✅ direct |
| Eq. (25) Monaghan–Gingold viscosity | [velocityDissipation.py](src/warpSPH/modules/deltaSPH/velocityDissipation.py), `ViscosityTerms.MonaghanGingold1983` | ✅ direct |
| Eq. (23) velocity divergence | [modules/momentum/](src/warpSPH/modules/momentum/), `WarpOperation.Divergence` | ✅ direct |
| Eq. (34) `L_i`, `⟨∇·⟩^L` | [gradRhoL.py](src/warpSPH/modules/density/gradRhoL.py), `computeRenormalizationMatrices` | ✅ **generalised 2026-09-05** (`computeGradRhoL(field=)`) |
| Eq. (33) bi-Laplacian ψ operator | [wp_densityDelta.py](src/warpSPH/modules/deltaSPH/wp_densityDelta.py) `DensityDiffusionScheme.deltaSPH` | ✅ **generalised 2026-09-05**; sign error found + fixed — see note |
| Eq. (61) value term | [wallPressure.py](src/warpSPH/modules/incompressible/wallPressure.py) `'shepard'` mode | ✅ exact |
| Eq. (61) body-force term | `wallPressureExtrapolation(bodyForce=)` | ✅ **built 2026-09-05**, §4.4 |
| Eq. (62) Shepard + mirroring | [modules/mdbc/velocity.py](src/warpSPH/modules/mdbc/velocity.py) | ⚠ audit, see §4.4 |
| §3.3 free-surface detection (Marrone/Sun) | [maronneDetection.py](src/warpSPH/modules/surfaceDetection/maronneDetection.py) + [dilation.py](src/warpSPH/modules/surfaceDetection/dilation.py) | ✅ gives `𝔽`, `𝕍`, `n`, `λ` |
| Eq. (57) `λ < 0.4` surface gating | `ShiftingProjectionScheme.surfaceNormal`, `surfaceLambdaThreshold` | ✅ direct |
| Eq. (46) adaptive `Δt` | [timestep/weaklyCompressible.py](src/warpSPH/modules/timestep/weaklyCompressible.py) | ⚠ recast, see §4.5 |
| §3.4 δ-ALE-SPH baseline | `--scheme deltaSPH` + `correctdrhodt`/`correctdvdt` | ✅ direct |
| RK2/SSPRK3/RK4 tableaus | `warpSPHIntegrators` (`rungeKutta2`, `sspRK3`, `rungeKutta4`) | ✅ reuse tableaus |
| Test cases (§4.1–4.5) | `hydrostaticColumn`, `rotatingSquarePatch`, `oscillatingDroplet`, `impact`, `dambreak` | ✅ all five exist |

### The one that matters most: Eq. (33) ≡ our existing δ-SPH ψ

> ⚠ **The A/B this section asked for found a sign error in the existing δ-SPH
> kernel.** Fixed 2026-09-05, see "The sign error" below. Everything in this
> subsection describes the operator *after* that fix.

`wp_densityDelta.py`'s `deltaSPH` branch computes
`ψ_ij = −(∇ρ^L_i + ∇ρ^L_j) − 2(ρ_j−ρ_i) x_ij/‖x_ij‖²`, dotted with `∇W_ij` —
the **unprojected** Marrone 2011 form (its Eq. 6, read off the PDF:
`ψ_ij = 2(ρ_j−ρ_i) r_ji/‖r_ij‖² − [⟨∇ρ⟩^L_i + ⟨∇ρ⟩^L_j]` with
`r_ji = r_j − r_i = −x_ij`). The paper's Eq. (33) writes the **projected**
form, gradient term contracted onto `x̂_ij` first.

These are algebraically identical whenever `∇W_ij ∥ x_ij`, which holds for any
isotropic kernel:
```
((∇p_i+∇p_j)·x̂)(x̂·∇W) = W'(r) (∇p_i+∇p_j)·x_ij / r = (∇p_i+∇p_j)·∇W
```

**Measured** (`scripts/probe_deltaSPHPsiProjection.py`, float64, 20×20 jittered
lattice, rms over interior rows, against a from-scratch `O(N²)` torch reference
that shares no code with the kernel):

| claim | linear `f` | quadratic | cubic |
|---|---|---|---|
| warp kernel − torch reference (unprojected) | 6e−14 | 2e−13 | 3e−13 |
| unprojected − projected, renormalisation **off** | 2e−15 | 2e−15 | 2e−15 |
| unprojected − projected, renormalisation **on** (relative) | **1.00** | 0.54 | 0.32 |

So **reusing the existing kernel reproduces AC-2L exactly**, with the caveat
confirmed quantitatively: they diverge completely once
`useGradientRenormalization` puts an `L_i` in front of `∇W` (then `L∇W ∦ x_ij`).
Assert that path off for ACSPH. Note the probe also shows *which* form survives
that: the projected one still annihilates a linear field with `L` on (its brace
`{(p_i−p_j) − ½(⟨∇p⟩_i+⟨∇p⟩_j)·x_ij}` is a scalar that vanishes regardless of
what multiplies `∇W`), while the unprojected one does not. If renormalised
gradients on this operator ever become desirable, implement Eq. (33) literally.

### The sign error (found by that A/B, fixed 2026-09-05)

`ψ` had the gradient term's sign flipped — `+grad − rho` where Marrone Eq. (6)
is `−grad − rho` — in all four gradient-carrying branches (`deltaSPH`,
`deltaOnly`, `denormalized`, `denormalizedOnly`). `densityOnly`
(Molteni–Colagrossi, no gradient term) was and is correct.

The defining property of the Antuono correction is that the two terms **cancel
pair-by-pair on a field linear in space**: for `f = a·x`,
`(∇f_i+∇f_j)·∇W = 2a·∇W` and `2(f_j−f_i)(x_ij·∇W)/r² = −2a·∇W` identically.
That is what promotes the Molteni–Colagrossi Laplacian to a *bi*-Laplacian and
lets the diffusive term reach a free surface without eating the smooth field
underneath. With the sign flipped the terms *added*: `deltaSPH` was numerically
`2 × densityOnly` on any smooth field — a second-order diffusion twice as
strong as the uncorrected one, in place of the fourth-order one. Measured
before/after on the linear field, rms over interior rows:

| | linear | quadratic |
|---|---|---|
| `deltaSPH`, before | 1.6e+00 | — |
| `deltaSPH`, after | **1.0e−13** | 1.8e−02 |
| `densityOnly` (unchanged, the control) | 6.2e−01 | 4.1e+00 |

It never blew up — the sign was still diffusive — which is exactly why only a
property test catches it. Pinned as `tests/test_deltaSPHDiffusion.py` (4 tests,
including the specific shape of the bug: `deltaOnly` came out *equal* to
`densityOnly` rather than its negative). Full suite green after the fix.

**This is not cosmetic for either scheme.** For ACSPH it is the whole of
§4.1.1: AC-2L is separated from AC-2 precisely by whether the operator can hold
a hydrostatic — i.e. linear — pressure gradient, and before the fix ours could
not. For WCSPH it means the repo's default δ-SPH density diffusion has been
over-strong and second-order, actively diffusing exactly the density gradients
it exists to preserve. **Worth reporting to the authors' circle / checking
against diffSPH, which this kernel was ported from.**

**Measured end-to-end on `sloshingTank --scheme wcsph`**, nx=100, `t ≤ 3.6 s`,
today's code both sides, the only difference being the sign (the pre-fix
operator is reproduced *exactly* by negating the gradient field handed to the
kernel, so this is a clean single-variable A/B — see
`scripts/probe_deltaSPHPsiSignAB.py`):

| | pre-fix ψ | post-fix ψ |
|---|---|---|
| diverged | no | no |
| min density over the run | 0.396 | **0.532** |
| max density over the run | 1.849 | 1.837 |
| peak \|p\| at Sensor 1, `t > 2 s` | 39.2 kPa | **32.5 kPa** |

(Measured band for the first impact peak: 2.2–13.1 kPa.) So the fix improves
both the density floor and the pressure overshoot, and degrades nothing. Note
neither is *good* — the case has separate open problems recorded in
`examples/sloshingTank/PLAN.md`, and the peak is still ~2.5× the band top. For
reference the run recorded on 2026-09-03 (`examples/sloshingTank/output/`)
**diverged** at `t = 3.41` with `ρ ∈ [0, ∞]`; that is older code, not this
variable, and is why the A/B above was run rather than compared against it.

---

# Part 4 — What has to be built

Ordered by how much is genuinely new.

## 4.1 The dual-time driver — *entirely new, the core of the work*

`schemes/artificialCompressible.py`. Owns the whole real-time step:

```
for m in 0..maxPseudoIters:
    u0 = u                                   # freeze stage-0 for the BDF source
    compute frozen 𝒟^p at stage 0            # §1.4 frozen diffusion
    for s in 1..s_RK:
        r  = spatial residual at u^{s−1}     # Eqs. 23, 25, 26
        Dx_Dt = α_t x0 + β_t x^n + γ_t x^{n−1}
        Dv_Dt = α_t v0 + β_t v^n + γ_t v^{n−1}
        r* = (1/α_PI) [ r − I_c·(Dx_Dt, Dv_Dt) ]     # I_c zeroes the p row
        u^s = u0 + α_s Δτ r*
    u = accumulate(b_s)                      # see §5.2 on Eq. (40) vs Fig. 1
    ṽ = v − Dx_Dt ;  if ε_v(ṽ) < target: break
apply shifting displacement (Eq. 58) + BDF correction (Eq. 59)
roll history: (x^{n−1},v^{n−1}) ← (x^n,v^n) ← (x^{n+1},v^{n+1})
```

**Framework integration — recommended approach.** The runner drives steps via
`ctx.integrator.function(state, f=ctx.stepFunction, dt=...)`
([runner.py:269](src/warpSPH/runner/runner.py#L269)), which does not fit a
scheme that owns its own time advance. Rather than adding a `dualTime`
integrator to `warpSPHIntegrators` (which would couple a general library to one
scheme), use the **exact-delta trick**: have `acsph_step` run the full dual-time
solve and return
```
dxdt = (x^{n+1} − x^n)/Δt ,  dvdt = (v^{n+1} − v^n)/Δt ,  dpdt = (p^{n+1} − p^n)/Δt
```
Forward Euler on an exact delta is the identity, so the runner reproduces the
converged state byte-for-byte with zero framework changes. Pin
`config.integrationScheme = forwardEuler` and **validate it at build time** —
a silent RK2 here would run the whole dual-time solve twice per step and blend,
which is wrong and not obviously wrong. (Precedent for this class of trap:
`dambreak.py`'s note that `divergenceFree` needs `semiImplicitEuler` and
"nothing in the code enforces this yet". Do enforce it.)

**BDF history storage.** `x^n, x^{n−1}, v^n, v^{n−1}` are needed. Put them on the
*system*, not the state — `WeaklyCompressibleSystem` already carries non-state
fields (`adjacency`, `t`, `domain`), which is the established precedent and
avoids `initializeNewState` cloning them per stage. Startup: step 0 has no
`u^{n−1}`, so fall back to BDF1 (`α_t = 1/Δt, β_t = −1/Δt, γ_t = 0`) for the
first step.

## 4.2 Michel et al. (2022) shifting law — new

Our shift is δ⁺-SPH (Sun 2017 scaling, Mach-dependent):
`modules/shifting/delta.py`'s docstring already notes *"An equivalent
shifting-velocity form (Michel 2022 scaling) is present in a comment but not
used."* ACSPH has no `c₀` and no Mach number, so this must be built:
`U^shift` (Eq. 56), the `β = (κh/Δx)³` interior scaling with surface decay to 1,
the two-branch magnitude clamp (Eq. 55), and the `σ` ramp (Eq. 57).

The surface machinery it needs (`𝕍`, `n`, `λ`, `d^fs`) is all already produced by
`detectFreeSurface`. Add as `ShiftingScheme.michel2022`; it is independently
useful to the existing WCSPH scheme, so it should live in `modules/shifting/`,
not inside the ACSPH scheme.

## 4.3 Generalised scalar-field diffusion operators — modify existing

~~`wp_densityDelta.py` reads `ρ_i`/`ρ_j` from `getParticle(referenceState, j)`,
i.e. bound to the state's density field. Add an optional
`queryField`/`referenceField` tensor pair...~~ **Done 2026-09-05.** The pair is
threaded through the existing `ExtraSpec` mechanism, guarded so both or neither
must be supplied, and the volume weight `m_j/ρ_j` is deliberately left on the
density (it is a quadrature weight, not the diffused quantity).
`computeScalarFieldDiffusion` is the raw-operator entry point;
`computeGradRhoL(field=)` supplies Eq. (34). AC-2 and AC-2L now exist with no
new kernel. Gradcheck covers the new branch. See Part 3 for the sign error this
work uncovered.

Then add, new:
- **AC-4** (Eq. 35): a second pass over the AC-2 output. Trivial once AC-2 is a
  reusable scalar-field operator.
- **AC-JST** (Eqs. 36–37): the `χ` switch is one extra interpolation loop; the
  blend is elementwise. Needs `𝕍` from the dilated surface mask.

> ⚠ `modules/deltaSPH` is covered by the `gradcheck` skill. Touching this kernel
> means running `/gradcheck deltaSPH` before and after.

## 4.4 Boundary conditions — one real gap, plus an audit

`modules/incompressible/wallPressure.py` is closer than it first looks:

- **Eq. (61)'s value term is already exact.** Its `'shepard'` mode is literally
  `p_b = Σ_f V_f p_f W_bf / Σ_f V_f W_bf`.
- ~~**Eq. (61)'s body-force term is the gap.**~~ **Done 2026-09-05.**
  `wallPressureExtrapolation(..., bodyForce=g)` now adds
  ```
  p_w = [ Σ_f p_f W_wf + (g − a_w)·Σ_f ρ_f r_wf W_wf ] / Σ_f W_wf ,   r_wf = r_w − r_f
  ```
  (`adami2012` Eq. 27) on the `'shepard'` **and** `'mirror'` closures.
  `'shepard' + bodyForce` is De Courcy's Eq. (61) exactly — Adami weights by
  `W_wf` alone where Eq. (61) weights by `W_bf V_f`, and the two agree here
  because ACSPH is density-invariant so `V_f = V₀` cancels. `'mls'` raises
  rather than taking it: its Liu–Liu linear fit already carries the local
  pressure gradient, so adding the correction would double-count.

  *Implementation.* The vector moment `Σ_f V_f ρ_f (r_w − r_f) W_wf` is not
  any single `WarpOperation` — it is assembled from two `Interpolate` gathers,
  `r_w·Σ_f V_f ρ_f W_wf − Σ_f V_f ρ_f r_f W_wf`, which is legitimate because
  `(g − a_w)` is a per-*wall* quantity and comes out of the sum. Both gathers
  reuse the value term's `OperationProperties`, so numerator and denominator
  share one kernel evaluation.

  *Restriction, deliberate.* Splitting `r_w − r_f` across two gathers discards
  the minimum-image convention, so a wrapping pair contributes `±L_d` of error
  per periodic direction `d`. Dotting with `bodyForce` annihilates that error
  whenever `bodyForce` has no component along a periodic axis — the only
  physically sensible configuration — so the code **asserts** that instead of
  silently returning a wrong wall pressure. A real moment kernel would lift it.

  *Verified* by `tests/test_wallPressure.py`: for a pressure field linear in
  space every neighbour's contribution `p_f + ρ_f g·(r_w − r_f)` is already
  the analytic wall value, so the weighted average is exact regardless of how
  truncated the wall neighbourhood is. Measured on a 24×24 column over three
  wall rows: **corrected 2.0e−7 relative error (float32 machine precision),
  plain Shepard 1.3e−1** — i.e. the uncorrected wall under-reads by up to
  `3 Δx · ρ₀ g`, 12.5 % of the whole column's pressure drop. That is precisely
  the error that stops a hydrostatic column from holding.

  Note the wall-acceleration `a_b` still has no per-particle source anywhere in
  the codebase (the same gap `modules/mdbc/velocity.py` documents for the
  velocity mirror's dead `2 u_wall` term) — static walls make `a_b = 0`, which
  covers every case in Part 7, but a moving-wall ACSPH case would need it. The
  `(N, dim)` `bodyForce` form is already accepted, so wiring a source is all
  that would be left.

  **Wired into ACSPH on 2026-09-05** (`schemes/artificialCompressible.py`'s
  `wallPressures`): recomputed at every RK stage from the current fluid
  pressure, `clampNonNeg=False` because ACSPH's pressure is a solved field that
  legitimately goes negative. Without it the wall reads `p = 0` (the non-fluid
  rows are masked every step) and the column simply falls out of the box.
  Verified exact against the analytic profile on the real case — see step 5b.

  Independently a DFSPH improvement: no DFSPH caller passes `bodyForce` yet, so
  it stays additive there, but it is exactly the term
  `DFSPH_IMPROVEMENT_PLAN.md` Part 23 needs for a gravity-driven wall.
- **A structural note**: these live under `modules/incompressible/`, i.e. they
  are DFSPH-facing. ACSPH needs them too, so either relocate to a shared module
  or import across. Prefer relocating — a third consumer makes the current home
  misleading.

Eq. (62)'s Shepard + no-penetration/no-slip mirroring largely exists in
`modules/mdbc/`. The audit item: **ACSPH extrapolates three velocity-like fields
where WCSPH extrapolates one** — `v` (no-penetration in the divergence, no-slip
or free-slip in the Laplacian), `ṽ` (no-penetration), `δv` (no-penetration, so
`δv·n = 0`). Confirm each gets the right condition rather than inheriting `v`'s.

## 4.5 Timestep — recast

Eq. (46) is close to but not the same as `modules/timestep/weaklyCompressible.py`:
the acoustic constraint is replaced by an advective one (`CFL_t h/‖v‖_max`), the
viscous constraint is `0.125 h²/ν`, and there is a symmetric growth/shrink clamp
(`[0.8, 1.2]×`) that exists specifically to protect BDF2 accuracy. `Δτ = Δt / R`
with `R` a config constant.

## 4.6 New scheme family: config, state, system, wiring

Per the user's expectation, ACSPH gets its own family rather than being bolted
onto WCSPH:

- **`enumTypes.py`**: new `ArtificialCompressibleSPHScheme` enum (single member
  `artificialCompressible`), plus `PressureSmoothingScheme` (`laplacian`,
  `renormalizedBiLaplacian`, `biharmonic`, `jst`) mapping to AC-2/2L/4/JST.
- **`systems/artificialCompressible.py`**: `ArtificialCompressibleState`. The
  key structural difference from `WeaklyCompressibleState` — **`pressures`
  becomes an integrated field** (`integrated('dpdt')`) and `densities` becomes
  `constant` at `ρ₀`. Carries `surfaceIndicators`/`surfaceNormals`/
  `surfaceLambdas` and the ghost bookkeeping unchanged.
  `ArtificialCompressibleSystemUpdate` = `{dxdt, dvdt, dpdt}`.
  `ArtificialCompressibleSystem` carries the BDF history (§4.1).
- **`configurations/artificialCompressible.py`**:
  `ArtificialCompressibleSPHConfig`, modelled on `WeaklyCompressibleSPHConfig`
  (fluid, viscosity, BCs, shifting, regions, rigid bodies, surface detection,
  gravity) with the EOS/`densityDiffusion` block swapped for a new
  `acParams`: `{ pressureSmoothing, CFL_tau, CFL_t, dtOverDtau, rkStages,
  epsilonV, epsilonS, uChar, maxPseudoIterations, minPseudoIterations,
  k2Factor (=0.1), kappa2 (=0.5), kappa4 (=1/32), k3 (=0.0),
  usePointImplicit (=True), useTildeVAdvection (=False),
  shiftInsidePseudoLoop (=False), bdfShiftCorrection (=True),
  referenceSoundSpeedForViscosity }`. Plus the round-trip
  `artificialCompressibleConfigToDict` / `dictTo...` pair.
- **Registration touchpoints** (mirroring the `WeaklyCompressibleSPHScheme`
  surface): `schemes/builder.py` (`SchemeBundle`), `io/parsers.py`,
  `io/export.py`, `io/importIO.py`, `runner/caseSpec.py` (enum sweep at
  [caseSpec.py:286](src/warpSPH/runner/caseSpec.py#L286)), `warpSPH/__init__.py`,
  `modules/timestep/wrapper.py` (dispatch on system type).

---

# Part 5 — Ambiguities and errors in the paper

Flagging these matters for two reasons: they must be resolved before coding, and
(given we work with the authors) they are worth reporting back.

## 5.1 Eq. (37): `ε₄ = min(0, κ₄ − ε₂)` — almost certainly should be `max`

Verified against the rendered page (p. 9), so this is not a text-extraction
artefact. As printed with `κ₄ = 1/32` and `ε₂ = 0.5 min(1, χ) ≥ 0`:
`κ₄ − ε₂ ≤ 1/32`, so `min(0, ·) ≤ 0` always. In smooth flow (`χ → 0`) it gives
`ε₂ = 0, ε₄ = 0` and the JST operator **vanishes entirely** — the exact opposite
of the stated design ("*fourth-order dissipation in smooth regions*"). Standard
JST (Jameson–Schmidt–Turkel 1981) is `ε₄ = max(0, κ₄ − ε₂)`.

**Decision: implement `max`.** Expose the paper-literal `min` behind a flag so
the discrepancy is reproducible, but do not default to it. Ask the authors.

## 5.2 Eq. (40) and Fig. 1 are mutually inconsistent for RK3/RK4

Eq. (40)'s update `u^s = u^0 + α_s Δτ r*(u^{s−1})` is the Jameson low-storage
form. It can only represent tableaus whose `A` is non-zero on the sub-diagonal
only *and* whose `b` equals the final stage row. The RK2 midpoint tableau in
Fig. 1 satisfies this (`α = {1/2, 1}`). **SSPRK3 and classical RK4 do not** —
SSPRK3 has `a₃₁ = a₃₂ = 1/4`, and RK4's `b = [1/6,1/3,1/3,1/6]` is not any stage
row. So either Fig. 1 is decorative and the code uses Jameson coefficients
(`{1/4,1/3,1/2,1}` for 4 stages), or Eq. (40) is a simplification and the code
uses the full tableaus.

**Decision: implement the general explicit Butcher form** (reuse
`warpSPHIntegrators.butcher` tableaus), which reproduces Fig. 1 exactly and
degenerates to Eq. (40) for RK2. Since §4.3 concludes RK2 is the best operating
point anyway, the ambiguity is largely academic — but it must be a deliberate
choice, not an accident. Ask the authors which the CUDA code does.

## 5.3 Eq. (30) has a stray `h`

Eq. (30) reads `+ k₂ h 𝒟^p_i` (verified on the rendered page 8) while Eqs. (23),
(51) and (54) all read `+ k₂ 𝒟^p_i`. Dimensional analysis settles it: `k₂ = 0.1hβ`
already carries the length scale, `𝒟^p ~ [p]/L²`, so `k₂𝒟^p ~ [p]β/L ~ [p]/T` ✓
and the extra `h` is wrong. Typo in Eq. (30) only; use the `k₂ 𝒟^p` form.

## 5.4 `𝕍` is used for two different things

`𝕍` denotes "within a kernel support radius of a free-surface particle" in both
Eq. (36) (JST switching) and Eq. (57) (shifting). Whether the *same* dilation
radius and the same underlying `𝔽` set are intended in both is not stated.
Assume yes, expose the dilation iteration count separately.

## 5.6 Eq. (46)'s first term is dimensionally a length

Verified against the rendered page 10, so not an extraction artefact:

```
Δtⁿ = max( min( CFL_t h , CFL_t h/‖v‖_max , 0.125 h²/ν , 1.2 Δtⁿ⁻¹ ) , 0.8 Δtⁿ⁻¹ )
```

`CFL_t h` is a **length**, not a time. What it *does*, though, is unambiguous
from the structure — paired with the next entry it is exactly

```
min( CFL_t h , CFL_t h/‖v‖_max )  ==  CFL_t h / max(1, ‖v‖_max)
```

i.e. the advective constraint with its denominator floored at one. In a code
whose velocities are O(1) that floor is a reference velocity of 1 left implicit,
which is precisely the missing dimension.

**Decision: implement it that way**, with the floor named
(`REFERENCE_VELOCITY`) rather than hidden as a bare `1`. Ask the authors whether
their `h` there carries such an implicit reference velocity.

**This is not pedantry — it is load-bearing.** Without that floor *nothing*
bounds `Δt` in a quiescent case: `‖v‖_max → 0` makes the advective term
infinite, and an inviscid run makes the viscous term infinite too, so `Δt`
climbs 1.2× every step to `config.maxDt`. Measured on `hydrostaticColumn`:
`Δt` went 5e−4 → 6.4e−3 in fifteen steps and the near-wall velocity error grew
with it. With the floor, `Δt` settles at 4.9e−3 and stays there.

**Separately**, Eq. (46) has no body-force constraint, which every other δ-SPH
timestep in this repo and in the literature carries and which a gravity-driven
case needs. `CFL_t √(h/‖a‖_max)` is implemented behind the existing
`dt_accelerationConstraint` flag; turn it off for the paper's literal set. Also
worth asking about.

## 5.5 Under-specified

- **`U_char`** in Eq. (48) is never given a definition per case. It is presumably
  the case's own characteristic velocity (`√(gH)` for dam break, `ωL` for the
  patch). Make it a required per-case config value.
- **The `𝕍` branch of Eq. (36)** returns `𝒟^ΔL` *unscaled*, i.e. `ε₂ = 1`
  implicitly at the surface, while the interior uses `ε₂ 𝒟^ΔL + ε₄ 𝒟^Δ²`. That
  is a discontinuity in the operator at the `𝕍` boundary. Presumably intended
  (the text says the bi-Laplacian is "activated at the free surface"), but worth
  confirming.
- **`β` in Eq. (57)'s surface decay** — "decreased to 1 for surface particles"
  gives the endpoints but not the interpolation. Assume it rides `σ` or `λ²`;
  ask.
- **Symbol collision**: `β` is both the AC wave speed (Eq. 24) and the shifting
  scaling `(κh/Δx)³` (Eq. 56). Unrelated quantities. Use distinct names in code.

---

# Part 6 — Cited literature

## 6.0 Status

**Every blocking reference is now in `literature/`.** On 2026-09-05 the target
paper and seven of its references were synced in (bib keys `decourcy2024`,
`antuono2010`, `antuono2012`, `letouze2013`, `michel2022`, `ramachandran2021`,
`lobovsky2014`, `marrone2015`), each identified from its own front matter,
verified field-by-field against its DOI record, and abstracted verbatim —
`scripts/check_literature.py` passes.

Four more the ACSPH discretisation leans on were **already here**: `marrone2011`
(δ-SPH, Eqs. 16–17), `sun2017` (δ⁺-SPH), `sun2019` (consistent shifting / the
δ-ALE baseline), `adami2012` (wall BC, Eq. 61). Also present: `cummins1999` [2].

Nothing in Parts 1–5 or Part 7 is now blocked on a document we do not have.

## 6.1 Obtained for this plan

| Ref | Key | Unblocks |
|---|---|---|
| — | `decourcy2024` | The scheme itself. |
| [8] | `antuono2012` | The `k₂ ≤ 0.2hβ` stability bound; the bi-Laplacian interpretation of Eq. (33); frozen diffusion; why Eq. (32) fails at free surfaces. |
| [7] | `antuono2010` | Origin of the corrected Laplacian; co-cited for the stability bound. |
| [76] | `letouze2013` | Square-patch **initial pressure field** (a Poisson solve — §4.2 cannot be initialised without it), the analytic stretching solution, and BEM/LDFM reference data. |
| [82] | `lobovsky2014` | Dam-break probe geometry and the 2.5%/97.5% experimental bounds (Figs. 28/30). Supplementary Materials carry the raw signals. |
| [80] | `marrone2015` | The analytic incompressible KE drop the jet-impact case (§4.4) is scored against. |
| [66] | `michel2022` | The shifting law of §4.2 — its derivation of `β = (κh/Δx)³` and its PST-conditions checklist, which is also worth auditing our existing shift against. |
| [29] | `ramachandran2021` | Cross-check `α_PI = 2Δt/(2Δt+3Δτ)` and the `ṽ` material derivative. Closest prior art, **with an open-source reference implementation**. |

## 6.2 Still not obtained — non-blocking, the paper reproduces the equations in full

| Ref | Paper | Note |
|---|---|---|
| [40] | Jameson, Schmidt, Turkel (1981), AIAA-81-1259 | Would settle §5.1 (`min` vs `max` in ε₄) definitively. |
| [70] | Marrone, Colagrossi, Le Touzé, Graziani (2010), *Fast free-surface detection and level-set function definition*, JCP 229(10) 3652–3663 | Already implemented here (`maronneDetection.py`); the citation is missing from the library, not the code. |
| [77] | Monaghan & Rafiee (2013), IJNMF 71(5) 537–561 | Droplet analytic solution — already encoded as `DROPLET_STRETCH`/`DROPLET_PERIOD` in `cases/oscillatingDroplet.py`. |
| [6] | Molteni & Colagrossi (2009), CPC 180(6) 861–872 | AC-2 (Eq. 32) is given in full. |
| [15] | Antuono, Sun, Marrone, Colagrossi (2021), *δ-ALE-SPH*, C&F 216 104806 | Baseline scheme; `sun2019` covers it for our purposes. |
| [50],[54],[55] | Monaghan & Gingold 1983; Bonet & Lok 1999; Randles & Libersky 1996 | Standard operators, already implemented. |
| [19] | Sun, Pilloton, Antuono, Colagrossi (2023), *Acoustic damper term in WCSPH*, JCP 483 112056 | The competing "fix WCSPH instead" approach — interesting for the comparison narrative. |
| [32] | De Courcy et al., SPHERIC 2023 | The precursor; may carry implementation detail cut from the journal version. |
| [21],[23],[25],[39],[28],[27],[31],[26],[60] | Chorin 1997; Turkel 1987; McHugh & Ramshaw 1995; Dupuy 2020; Clausen 2013; Ramachandran & Puri 2019; Chola & Shintake 2021; Rouzbahani & Hejranfar 2017; Vila 1999 | §2 theory context only. No implementation content. |

## 6.3 Out of scope

[72] Fourey et al. (2017) and the FSI chain [33–35], [73–75], [85] support §4.1.2
(elastic-base hydrostatic column) and Appendix A (modal structural solver + RBF
coupling). **This repo has no structural solver**, so §4.1.2 is not reproducible
and Appendix A needs nothing. Skip.

---

# Part 7 — Validation plan

All five of the paper's cases already exist here, which is unusually lucky.

| § | Case | Repo case | Measures | Reference |
|---|---|---|---|---|
| 4.1.1 | Hydrostatic column, rigid base | `hydrostaticColumn` | Hydrostatic profile, free-surface integrity, KE decay | Analytic gradient |
| 4.1.2 | Hydrostatic column, elastic base | — | FSI energy dissipation | **Skip** — no structural solver |
| 4.2 | Rotating square patch | `rotatingSquarePatch` | KE decay, centre pressure, momentum conservation | BEM/LDFM [76] |
| 4.3 | Oscillating droplet | `oscillatingDroplet` | IRMSE(KE), IRMSE(semi-major axis), cost | Analytic [77] ✅ already encoded |
| 4.4 | Normal impact of 2D jets | `impact` | Instantaneous KE drop, pressure smoothness | Analytic [80] |
| 4.5 | Dam break (2D + 3D) | `dambreak` | 4 wall pressure probes, KE | Experiment [82] |

**Ordering.** §4.1.1 first — it is the operator discriminator (it is what
separates AC-2 from AC-2L/AC-JST) and it is cheap. Then §4.3, which is the only
case with a clean analytic score and is the paper's own parameter-sweep vehicle
(reproduce Tables 1 and 2 — they are the single best acceptance test for the
dual-time machinery). Then §4.2 for conservation, then §4.4/§4.5.

**Acceptance targets from the paper:**
- AC-2L/AC-JST hold a hydrostatic gradient with no free-surface diffusion; AC-2
  visibly fails. (§4.1.1, Figs. 2–4.)
- Table 1/2 reproduce qualitatively: error flat for `CFL_t ≤ 0.4`, ~2.4× jump at
  0.6; cost linear in `Δt/Δτ` and in RK stage count; RK order buys no accuracy.
- Square patch: KE loss 25–32% *less* than δ-SPH across `L/Δx = 200/400/800`;
  **zero visible pressure oscillation at every resolution**; cost 2.4–2.8× δ-SPH.
- Jet impact: correct KE drop within a few time steps, no oscillatory ringing;
  total cost ≤ 1.5× δ-SPH.
- Dam break: noise-free `P1` through the void-closure event at `t√(g/H) ≈ 8.4`
  where δ-SPH's acoustic noise swamps the signal.

**Cost metric.** The paper's `𝒞_e = ∫ (m_iter · s_RK / Δt) dt` (Eq. 65) is
implementation-independent and should be recorded alongside wall time — it is
how their numbers are quoted and the only fair way to compare against our δ-SPH.

---

# Part 8 — Sequencing

1. ~~**Literature sync.**~~ **Done 2026-09-05** — the paper and seven references
   synced, checker green (§6.0). Remaining optional follow-ups: add the §6.2 set
   to `EXPANSION_CANDIDATES.md`, and consider promoting `marrone2011` from the
   extended set to the core (it is cited throughout Parts 1 and 3 but carries no
   abstract, the same case that promoted `dehnen2012` on 2026-09-04).
2. ~~**Finish the Adami `bodyForce` term** (§4.4).~~ **Done 2026-09-05** —
   `wallPressureExtrapolation(..., bodyForce=...)` on the `'shepard'` and
   `'mirror'` closures, exact on a linear pressure field to float32 machine
   precision (`tests/test_wallPressure.py`, 6 tests). See §4.4 for the
   two-gather decomposition and its periodic-axis restriction.
3. ~~**Generalise the diffusion kernel to an arbitrary scalar field** (§4.3), and
   A/B the projected vs unprojected ψ form.~~ **Done 2026-09-05.**
   - `computeDensityDiffusionDeltaSPH` takes an optional
     `queryField`/`referenceField` pair; the volume weight `m_j/ρ_j` is
     untouched (it is quadrature, not the diffused quantity). New public entry
     `computeScalarFieldDiffusion` (no `schemeConfig`, no prefactor);
     `computeDensityDiffusion` is now its δ-SPH specialisation.
     `computeGradRhoL` takes `field=` — with the pressure it is Eq. (34)
     verbatim.
   - `/gradcheck deltaSPH` green before and after; the script now runs every
     `DensityDiffusionScheme` twice, once through the field pair, so the new
     branch's adjoint is covered.
   - The A/B (`scripts/probe_deltaSPHPsiProjection.py`) confirmed the Part 3
     equivalence to 2e−15 with renormalisation off, quantified the divergence
     with it on (100 %/54 %/32 %), **and found the ψ sign error** — see §3's
     "The sign error". Full suite green after the fix.
4. ~~**Scaffold the new family** (§4.6).~~ **Done 2026-09-05.** `--scheme
   artificialCompressible` resolves, builds, and runs a step end to end through
   the real integrator; the step is a deliberate no-op
   (`schemes/artificialCompressible.py::PHYSICS_IMPLEMENTED = False`, warns
   once on entry) with a marked socket where step 5's driver goes.
   - `ArtificialCompressibleSPHScheme` + `PressureSmoothingScheme` +
     `isArtificialCompressibleScheme` in `enumTypes.py`.
   - `systems/artificialCompressible.py`: state with `pressures` **integrated**
     and `densities` **constant**; system carrying the BDF history plus
     `rollHistory` and `bdfCoefficients` (Eq. 42, with the BDF1 first-step
     fallback reported through an `order` return).
   - `configurations/artificialCompressible.py`: `ArtificialCompressibilityParams`
     (every Part 2 constant, `uChar` and `referenceSoundSpeedForViscosity`
     `Optional` on purpose) + the round-trip pair.
   - Registered in `schemes/builder.py`, `io/parsers.py`, `io/export.py`,
     `io/importIO.py`, `runner/runner.py::_resolveScheme`,
     `runner/caseSpec.py::schemeNames`, `warpSPH/__init__.py`.
   - **The integrator trap is enforced**, not documented:
     `validateIntegrationScheme` raises on anything but `forwardEuler` at step
     entry, because the exact-delta hand-off is silently wrong under a
     multi-stage integrator (it would run the whole dual-time solve per stage
     and blend). `modules/timestep/wrapper.py` likewise raises rather than
     letting an ACSPH system fall through to the acoustic timestep.
   - `tests/test_artificialCompressibleScaffold.py`, 14 tests. Note
     `test_variableStepBdf2DifferentiatesAQuadraticExactly`: the fixed-step
     limit alone would not catch a swapped `Δtⁿ`/`Δtⁿ⁻¹` in Eq. (42).
5. ~~**The dual-time driver** (§4.1) with AC-2L and RK2 only.~~ **Done
   2026-09-05.** `schemes/artificialCompressible.py` now runs the full
   Eqs. (38)–(48) loop: frozen stage-0 BDF source, frozen-per-iteration `D^p`,
   a general explicit Butcher RK sweep, `I_c = diag{0,1,1}`, point-implicit
   `α_PI`, and the `ε_v` convergence test on `ṽ`. `ṽ` advection, internal
   shifting and `k₃` raise rather than silently no-opping; AC-4/AC-JST raise
   pointing at step 8.
   - **New supporting work.** `modules/artificialCompressible/pressureSmoothing.py`
     (AC-2 / AC-2L dispatch onto `computeScalarFieldDiffusion`);
     `approachOnly=False` on `computeVelocityDiffusion`, which turns its
     `inviscid=False` branch into the Monaghan–Gingold velocity Laplacian
     proper — the clamp that made it one-sided is an artificial-viscosity
     device, and Eq. (25) has no such clamp. Both `nu * rho0` (for the kernel's
     `mean(ρ)` division) and the `1/ρ_i` on the pressure gradient are applied
     explicitly, so nothing here assumes `ρ₀ = 1` the way δ-SPH does.
   - **Measured** (`tests/test_artificialCompressible.py`, 24×24 periodic box,
     Taylor–Green plus a compressive perturbation, so only the solve is
     graded): `rms(∇·v)` **1.84 → 2.6e−3**, monotone in the iteration budget
     and flat between 100 and 400 iterations (converged). §4.3's finding
     reproduces: RK3/RK4 buy no accuracy over RK2 at equal iteration count and
     cost linearly more.
   - **Still open on the driver itself:** convergence to `ε_v = −6` took ~400
     iterations from that (deliberately extreme) initial transient. Whether
     that is the initial condition, `Δt/Δτ`, or something real is exactly what
     step 6's Table 1/2 reproduction answers — do not tune it before then.
   - **Validated on `hydrostaticColumn`** (§4.1.1, the paper's own first case),
     see step 5b below for the wiring that took and what it measures.
5b. **`hydrostaticColumn` under ACSPH** — done 2026-09-05.

   *Wiring.* `initializers/weaklyCompressible.py::initializeState` now builds
   any of the three state classes from one construction (it had a
   character-identical `if`-branch per class and no `else`, so an unknown class
   fell through to a `NameError`); `cases/weaklyCompressible.py` gains
   `configureDomain` + `configureArtificialCompressible`, which also **forces
   `forwardEuler`** (loudly) since the exact-delta contract requires it;
   `hydrostaticColumn` branches on `isArtificialCompressibleScheme` and takes
   the *raw* hydrostatic seed rather than the mean-shifted one — ACSPH has no
   pressure gauge, so the shift is a half-column-drop error at the free
   surface, not a gauge choice.

   *The domain is made non-periodic on this branch.* It is walled on every
   side, so the periodicity buys nothing, and it actively breaks Eq. (61): the
   wall-pressure moment is not minimum-image safe, so once a fluid particle
   drifts within a support radius of the bottom face it becomes a wrapped
   neighbour of the *top* wall and the moment picks up a whole domain height
   along gravity. `wallPressureExtrapolation` now detects exactly that (an
   `O(N)` per-axis test on whether such a pair can exist, one-sided, so it can
   only over-report) and refuses rather than returning a wrong wall pressure.

   *Measured — the scheme is right.* With `p` seeded analytically and `v = 0`
   (`scratchpad` probe, nx=32):
   - Adami wall pressure vs analytic at 199 wall rows: **max error 0.0000**
     against a column pressure drop of 4.905. The Part-2 work does exactly its
     job.
   - Bulk momentum residual: **‖r_v‖ = 0.0079 against g = 9.81**, i.e. the
     discrete pressure gradient balances gravity to 0.08 %. The hydrostatic
     state *is* a discrete equilibrium for this scheme.
   - `r_p = 0` in the bulk; density exactly invariant, as it must be.

   *What it still lacks: the shift (step 7).* Run forward, the column holds its
   pressure profile (`p ∈ [0.31, 4.87]` against an analytic drop of 4.5) but
   develops a near-wall velocity error concentrated in the **bottom corners**,
   and the free surface drifts down slowly. Two things were needed to keep it
   bounded, both marked in code:
   - the Eq. (46) advective floor (§5.6) — without it `Δt` ran to `maxDt` and
     the corner error grew with it;
   - the repo's mDBC no-penetration position correction, applied as an
     acceleration the way `deltaSPH_step` applies it
     (`acParams`-adjacent flag `noPenetrationShift`, default on). **Not in the
     paper** — Eq. (62) relies on the velocity mirror alone — but the paper
     never runs a walled case without particle shifting either. With it,
     `‖v‖_max` peaks at 0.29 instead of 2.9 and no particle leaves the box; the
     worst corner particle sits at `x = -0.497` against a wall plane at
     `-0.5`, where before it reached `-0.62`, well inside the wall band.

   *200 steps, nx=32, to `t = 0.94` (the case's full `tLimit`):* no divergence,
   density **exactly 1.000** throughout (invariant by construction, as it must
   be), `voidFraction 0`, `neighbourCountCV` flat at 0.24, `‖v‖_max` 0.58
   (peak 0.73), KE 6.1e−3, `dispMax` 0.27. The one clearly bad number is
   **`pairedFraction` 0.065** with `nnDistP01` down to 0.32 — particle pairing,
   which is exactly and only what particle shifting exists to prevent.

   So: **step 7 (Michel et al. shifting) is the next action**, and this case is
   the thing to re-measure after it — `pairedFraction` and `‖v‖_max` are the
   two numbers it has to move. Set `noPenetrationShift = False` then and check
   whether the shift alone carries the corners, which is what the paper implies.

   *One loose end:* the case's own `pressureSlope`/`pressureSlopeRatio` figures
   of merit stop being reported once the run develops (`hydrostaticDiagnostics`
   returns early when its bulk band has fewer than 8 rows). They report fine on
   the first steps — `pressureSlopeRatio 0.79` after three. The direct probe
   above is the stronger measurement anyway (the *residual* against `g`, not a
   fit), but the band gate is worth understanding before quoting this case's
   published axes for ACSPH.

6. **Timestep + convergence control** (§4.5, §1.6). Reproduce Tables 1 and 2 on
   `oscillatingDroplet`. This is the real acceptance gate for the machinery.
   - **Eq. (46) landed 2026-09-05**: `modules/timestep/artificialCompressible.py`,
     wired into `modules/timestep/wrapper.py`'s dispatch. Advective in place of
     acoustic, `0.125 h²/ν` viscous, the symmetric `[0.8, 1.2]×` step-ratio
     clamp, and a `CFL_t > 0.4` warning (Tables 1–2's measured cliff). Eq. (46)
     as printed is dimensionally impossible — see the new §5.6 — so the first
     term is implemented as `CFL_t √(h/‖a‖_max)`, not `CFL_t h`.
   - **Still to do:** the Table 1/2 sweep itself, which needs an ACSPH-aware
     case.
7. **Michel shifting** (§4.2). Validate on `rotatingSquarePatch`.
8. **Remaining operators**: AC-2, AC-4, AC-JST. Reproduce Fig. 2 / Fig. 16.
9. **Impact and dam break** (§4.4, §4.5), including the `𝒞_e` cost metric.
10. **Optional/experimental**, only if the above is clean: `ṽ` material
    derivative (§1.7), internal shifting (Eq. 60), `k₃` term, RK3/RK4.

## Relationship to the other plans

This displaces `DFSPH_IMPROVEMENT_PLAN.md` as the active priority, per the
current decision. `COUPLED_INCOMPRESSIBLE_NEWTON_PLAN.md` remains queued behind
both. Steps 2 and 3 above are shared infrastructure that benefit the DFSPH work
regardless of ordering, so they are not sunk cost if priorities shift back.
