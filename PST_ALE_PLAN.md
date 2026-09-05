# warpSPH — Particle Shifting and the ALE Formalism: analysis and plan

Target paper — `literature/michel2022_particle-shifting-techniques.pdf`, bib key
`michel2022`:

> **On Particle Shifting Techniques (PSTs): Analysis of existing laws and
> proposition of a convergent and multi-invariant law**
> J. Michel, A. Vergnaud, G. Oger, C. Hermange, D. Le Touzé
> *Journal of Computational Physics* **459** (2022) 110999
> `doi:10.1016/j.jcp.2022.110999`

Split out of `ACSPH_PLAN.md` step 7. That step read "implement the Michel
shifting law"; reading the paper properly, it is not one law but **a set of
requirements, an audit of every PST in the literature against them, a new law,
and three SPH schemes to validate it on — two of which this repo does not
have.** Hence its own document.

---

# The question this plan answers

**Is now the moment to implement the Vila and Parshikov & Medin ALE schemes?**

Short answer: **yes to an ALE now — but not via Vila, and not via a Riemann
solver.** The ordering is **PST → δ-ALE-SPH → Riemann/MUSCL → Parshikov →
Vila**, and the reason the second stage is not the one the question proposed is
a paper that was already sitting in `literature/` unclaimed:

> `antuono2021`, *The δ-ALE-SPH model* — "Differently from previous works on
> ALE, which generally adopt conservative variables (i.e. mass and momentum)
> and rely on the use of **Riemann solvers** inside the spatial operators, the
> proposed model is expressed in terms of **primitive variables** (i.e. density
> and velocity) and is written by using the **standard differential
> formulations of the weakly-compressible SPH schemes**."

That is a full ALE formalism this repo can reach with the operators it already
has, and its §5 constant-mass variant is *literally* what
`WCSPH_SHIFTING_PLAN.md` already built. Part 4.0 works through it. The rest of
the reasoning below still holds and is what puts Vila last rather than first:

- **The PST is the thing with a live consumer.** ACSPH needs it now
  (`ACSPH_PLAN.md` step 5b: `pairedFraction 0.065` on the hydrostatic column is
  the shift-shaped hole), and it is entirely scheme-independent — Eqs. (21),
  (22) and (48) reference nothing but positions, velocities and the free-surface
  set. It is also the *only* piece here that is unblocked by literature.
- **Both ALE schemes share one dependency this repo does not have: a Riemann
  solver with MUSCL reconstruction.** Michel Eqs. (26)–(27) and (29)–(31) are
  written in terms of `(ρ_E, u_E, P_E)`, the Riemann solution at the `i–j`
  midpoint, reconstructed with MUSCL + minmod. The repo has *related* machinery
  — CRKSPH's "Riemann-like pseudo-viscosity `Q`" with van Leer and eta-based
  limiters (`modules/crk/{accel,limiter}.py`) — but no actual Riemann solver.
  That subsystem is the real cost, it is shared, and it is independently useful.
- **Parshikov is cheap once that exists; Vila is not.** Parshikov keeps mass
  constant (`d(Vρ)/dt = 0`) and puts the PST in the position equation only, so
  it is structurally *today's* scheme plus a Riemann flux. Vila makes volume an
  integrated field and mass non-constant, and threads the PST through every
  equation.
- **And one thing genuinely does argue for scoping the ALE now, which is the
  reason to write this document rather than a one-line step:** Michel §3.1 says
  the PST's consistency requirement "can be theoretically alleviated when using
  an ALE formalism". A PST designed only against the Lagrangian case may be
  shaped wrong for the ALE one. The requirements in Part 2 are what keeps that
  from happening, and they cost nothing to respect up front.

Two further findings lower the cost of everything downstream:

- **The apparent-volume path is already threaded through every operator in the
  package** (Part 4.3), so an integrated `V_i` needs no kernel change at all.
- **`antuono2021` is already on disk** — it was in `literature/` unclaimed by
  the manifest, which `scripts/check_literature.py` caught. Now synced.

---

# Part 1 — What this repo has today, audited against the paper

Michel's Table 1 classifies every Fick's-law PST by four choices, and Table 2
scores them against three requirements. Our shift is the `Sun et al. [32]` row.

## 1.1 The generic form (Eq. 8)

Every Fick's-law PST is:

```
δu_i = U_char · d_char · ∇̃C_i                 if ‖U_char d_char ∇̃C_i‖ < U_lim
     = U_lim · ∇̃C_i/‖∇̃C_i‖                    otherwise
```

with four free choices: the pointing vector `∇̃C`, a characteristic length
`d_char`, a characteristic velocity `U_char`, and a limit `U_lim`.

| | `d_char` | `U_char` | `∇̃C_i` | `U_lim` |
|---|---|---|---|---|
| Lind et al. 2012 [17] | `h` | `h/Δt` | `−∇̃C_i` (tensile-corrected) | `0.2 h/Δt` |
| Oger et al. 2016 [25] | `R` | `Ma c₀` | `−∇C_i` (plain) | `0.25‖u_i‖` |
| Sun et al. 2019 [32] | `2h` | `Ma c₀` | `−∇̃C_i` | `0.5 U_max` |
| **Michel 2022** | `R` | `max_j\|(u_j−u_i)·x̂_ij\|` | `−∇̃C_i` | same as `U_char` |

`∇C_i = Σ_j ∇_iW_ij V_j` (Eq. 2); the tensile-corrected form is
`∇̃C_i = Σ_j [1 + 0.2 (W_ij/W(Δx))⁴] ∇_iW_ij V_j` (Eq. 3, Monaghan 2000).

## 1.2 Where we sit

`modules/shifting/delta.py` is the Sun 2017/2019 row: it rescales the raw
kernel-gradient term to `−CFL · Ma · 2h²` with a **per-call global** Mach number
`Ma = ‖v‖_max/c₀`. Its own docstring already records that "an equivalent
shifting-*velocity* form (Michel 2022 scaling) is present in a comment but not
used".

So, against Michel's Table 2, our current shift:

| requirement | ours | why |
|---|---|---|
| `lim_{Δx→0} δr = 0` | ✅ | true of every row in Table 1 |
| `lim_{Δx→0} δu = 0` | ⚠ only if `Δx/R → 0` | which is exactly the ratio SPH holds *fixed* |
| Galilean + local rotation invariance | ❌ | `U_lim` is `0.5 U_max`, a global velocity |
| independence from the global solution | ❌ | same `U_max` |
| compatible with incompressibility | ❌ | `Ma c₀` — there is no `c₀` in DFSPH or ACSPH |
| `lim_{Δx→0} δu^FS = 0` at the surface (Table 3) | ❌ | no row but Michel's satisfies this |

That last one is not academic. The Mach dependence is precisely why
`ACSPH_PLAN.md` §4.2 flagged this as a gap: ACSPH has no sound speed to form
`Ma` from, so the current law cannot be used by it *at all*.

**We do already have the surface machinery**, from
`docs/historic_plans/WCSPH_SHIFTING_PLAN.md`: `ShiftingProjectionScheme.surfaceNormal`
implements the normal-cancelling projection with the `λ < 0.4` gate, and
`detectFreeSurface` produces `𝔽`, the dilated set `𝕍`, normals `n`, and `λ`.
Eq. (48) is that projection with a different `σ` ramp and a `λ²` prefactor.

## 1.3 We also already have *part* of an ALE

`WCSPH_SHIFTING_PLAN.md` delivered `ShiftProperties.correctdrhodt` /
`correctdvdt` — the Sun 2019 δ-ALE convective corrections, folded into the
equations of motion in `WeaklyCompressibleSystem.finalize`:

```
drhodt_shift = ∇·(ρ δu) − ρ ∇·(δu)
dudt_shift   = −u ∇·(δu) + ∇·(u ⊗ δu)
```

That is a **quasi-Lagrangian** scheme — Michel §4.2.3's δ-SPH row *plus* the δu
terms in the continuity and momentum equations. It is not the full ALE of
§4.2.1: mass and volume are still not integrated, and the fluxes are not
Riemann-based. Worth being precise about, because "we already have ALE terms"
and "we have an ALE formalism" are different claims and only the first is true.

---

# Part 2 — The requirements, and the law

## 2.1 Three requirements (§3)

1. **Consistency (§3.1).** `lim δu_i = 0` as `Δx → 0` at *fixed* `R/Δx`. From
   Quinlan et al.'s truncation error, `R‖∇C_i‖ = O(1)` at fixed `R/Δx`
   (Eq. 12) — so the whole burden falls on `U_char`: the PST converges iff
   `U_char → 0` (Eq. 15) or `U_lim → 0` (Eq. 16). Every row in Table 1 uses a
   `U_char` that does **not** vanish under refinement, which is why they are all
   conditional.
2. **Galilean *and local rotation* invariance (§3.2).** `R∇C_i` is already
   invariant; `U_char`/`U_lim` must not break it. Any velocity built from the
   *absolute* `u` (local or global) fails. A *relative* one, `f(u_j − u_i)`,
   passes — and projecting it, `(u_j − u_i)·x̂_ij`, additionally gives local
   rotation invariance.
3. **Independence from the global solution (§3.3).** A dam break's `U_max` lives
   in the front; keying the PST to it over-shifts everywhere else.

The elegance is that (2) implies (1) for free: a first-order Taylor expansion
gives `u_j − u_i = O(Δx)`, so a relative `U_char` vanishes under refinement at
first order without any extra device.

## 2.2 The law (Eq. 22), interior

```
U_char_i = U_lim_i = max_j | (u_j − u_i) · x̂_ij |            (Eq. 20, ω ≡ 1)
β_i      = (R/Δx)³
∇̃C_i     = Σ_j [1 + 0.2 (W_ij/W(Δx))⁴] ∇_iW_ij V_j            (Eq. 3)

δu_i = −0.5 · U_char_i · β_i R ∇̃C_i                if ‖β_i R ∇̃C_i‖ < ½ (R/Δx)
     = −0.5 · U_lim_i  · ½(R/Δx) · ∇̃C_i/‖∇̃C_i‖     otherwise
```

`α = 0.5` throughout the paper. `β_i` counterbalances the lowest-degree term of
the truncation expansion (Eq. 11); the `½(R/Δx)` cap exists because `R‖∇̃C_i‖` is
`O(1)` but not predictable a priori.

> ⚠ Note `U_char` and `U_lim` are the *same quantity* here. That is deliberate
> (§3.2: "using in addition a similar `U_lim` allows the Galilean invariance to
> be recovered in any case") and is what makes the limited branch invariant too.

## 2.3 The law at a free surface (Eq. 48)

```
δu_i^FS = 0                                        if λ_i < 0.4
        = λ_i² ( δu_i − σ_i (δu_i · ñ_i) ñ_i )     otherwise

σ_i = min[ 1, max( 0, (d_i^FS − R)/(R/2 − R) ) ]
```

with three modifications, all of which matter:

- **`ñ_i` is not `n_i`** (Eq. 47). A free-surface particle uses its own normal;
  a *vicinity* particle inherits the normal of its **nearest free-surface
  neighbour**. Michel's Fig. 14 is the argument: the computed normal degrades
  away from the surface, and inheriting is cheaper and better than fixing it.
- **`β_i` decreases linearly** through the free-surface region, from `(R/Δx)³`
  in the interior to `1` at the surface. → *This answers `ACSPH_PLAN.md` §5.5's
  open question about the interpolation.* It is linear, not `σ`- or `λ²`-riding.
- `d_i^FS ≤ R/2` cancels the normal component entirely; beyond that `σ` ramps it
  back in linearly, reaching full freedom at `d^FS = R`.

Table 3 records that this is the only law in the survey with
`lim_{Δx→0} δu^FS = 0`.

## 2.4 What we can reuse

| paper | repo | status |
|---|---|---|
| Eq. (3) tensile-corrected `∇̃C` | `sample/wp_deltaShift.py` | ⚠ audit — is the `1 + 0.2(W/W(Δx))⁴` factor there? |
| `𝔽`, `𝕍`, `n`, `λ` | `modules/surfaceDetection/` | ✅ direct |
| Eq. (48)'s normal projection + `λ<0.4` gate | `ShiftingProjectionScheme.surfaceNormal` | ✅ shape matches, `σ` ramp differs |
| `d_i^FS` (distance to nearest surface particle) | — | ❌ new, and `ñ_i` needs the *identity* of that neighbour too |
| Eq. (20) `max_j |(u_j−u_i)·x̂_ij|` | — | ❌ new; one neighbour-loop reduction |
| δ-ALE convective terms | `correctdrhodt`/`correctdvdt` | ✅ direct |

The `ñ_i` inheritance is the one piece with no analogue here: it needs a
nearest-free-surface-*neighbour* search, not just a distance. `wp_dilate.py`
already walks the surface neighbourhood, so it is the natural place.

---

# Part 3 — The three schemes Michel validates on

All three share the Cole EOS, Wendland C2 at `R/Δx = 4` (2D) / `3` (3D), RK4,
and `Δt = CFL min_i(R_i/c₀)` with `CFL = 0.375`.

## 3.1 δ-SPH, PST in the motion equation only (§4.2.3) — **we have this**

```
dx_i/dt = u_i + δu_i
dρ_i/dt = −ρ_i Σ_j (u_j−u_i)·∇W V_j + 𝒟_i
d(V_iρ_i)/dt = 0                                       (mass constant)
ρ_i du_i/dt = Σ_j (P_i+P_j) ∇W V_j + Π_i + V_iρ_i g
```

This is `schemes/deltaSPH.py` exactly. With `correctdrhodt`/`correctdvdt` on we
are one step *beyond* it (the quasi-Lagrangian variant).

## 3.2 Parshikov & Medin, Riemann fluxes (§4.2.2) — **needs a Riemann solver**

```
dx_i/dt      = u_i + δu_i
dV_i/dt      = V_i Σ_j 2 (u_E − u_i) · ∇_iW_ij V_j
d(V_iρ_i)/dt = 0                                       (mass constant)
d(V_iρ_iu_i)/dt = −V_i Σ_j 2 P_E ∇_iW_ij V_j + V_iρ_i g
```

Only `(u_E, P_E)` are new — no density flux. Mass is still constant, so
`ρ_i = m_i/V_i` follows from the integrated volume. **The PST enters the
position equation only**, i.e. the same coupling we already have.

## 3.3 Vila, full ALE (§4.2.1) — **the real ALE**

```
dx_i/dt = u_i + δu_i
dV_i/dt = V_i Σ_j (u_j−u_i)·∇W V_j          + V_i Σ_j (δu_j−δu_i)·∇W V_j
d(V_iρ_i)/dt = −V_i Σ_j 2ρ_E(u_E−u_ij)·∇W V_j + V_i Σ_j ρ_E(δu_i+δu_j)·∇W V_j
d(V_iρ_iu_i)/dt = −V_i Σ_j [2(ρ_E u_E⊗(u_E−u_ij) + P_E I)]∇W V_j + V_iρ_i g
                  + V_i Σ_j ρ_E u_E ⊗ (δu_i+δu_j) ∇W V_j
```

with `u_ij = (u_i+u_j)/2`, and the interface at the `i–j` midpoint moving at
`u⁰_ij = (u⁰_i + u⁰_j)/2` where `u⁰ = u + δu`. MUSCL with a minmod limiter
reconstructs `(ρ, ρu)` at `x_ij`.

Three things are genuinely new here relative to §3.2:
1. **Mass is not conserved per particle** — `d(Vρ)/dt ≠ 0`, there are real mass
   fluxes between particles.
2. **The PST appears in every equation**, not just the position one. That is the
   "consistent ALE" of the title, and it is what lets §3.1's consistency
   requirement be relaxed.
3. The momentum flux carries the full `ρ_E u_E ⊗ (u_E − u_ij)` tensor.

---

# Part 4 — Assessment: what it would actually take

## 4.0 There are two routes to an ALE, and only one needs a Riemann solver

**Route 1 — conservative variables + Riemann fluxes.** Vila (§3.3) and
Parshikov & Medin (§3.2). `antuono2021` §2.3 states the constraint plainly:
"The formulation in conservative variables is, however, mandatory if Riemann
solvers are used to model the particle interactions." Blocked on §4.1.

**Route 2 — primitive variables, standard operators.** `antuono2021`'s
δ-ALE-SPH. The ALE volume equation is just

```
dV_i/dt = V_i ∇·(u + δu)_i                                    (its Eq. 8)
```

and the system is written in `(ρ, u)` with the same differential operators
weakly-compressible SPH already uses. **No Riemann solver, no MUSCL, no
conservative-variable state.**

Two things make route 2 the obvious next stage for *this* repo:

1. **Its §5 "constant mass" variant is already implemented here.** That variant
   reads
   ```
   dρ_i/dt = −ρ_i ∇·(u+δu)_i + ∇·(ρ δu)_i + 𝒟^ρ_i
   du_i/dt = −∇p_i/ρ + ∇·(T_v)_i + g + ∇·(u ⊗ δu)_i − u_i ∇·(δu)_i
   ```
   which is term-for-term `ShiftProperties.correctdrhodt` / `correctdvdt` in
   `WeaklyCompressibleSystem.finalize` (`WCSPH_SHIFTING_PLAN.md`). We got there
   from `sun2019`; `antuono2021` §5 is the same equations, and it says of them:
   *"strictly speaking, the above variants cannot be regarded as ALE schemes on
   their own."* So Part 1.3's claim is the paper's own, not an editorial one.

2. **The delta to a real ALE is small and named.** Make `V` (equivalently `m`)
   an integrated field — which needs *no kernel change*, see §4.3 — and add the
   two diffusion terms. Those terms are the paper's actual contribution and the
   thing that cannot be guessed:

   > "We show that the above-mentioned ALE-SPH equations are, however, unstable
   > when they are integrated in time. The instability appears in the form of
   > large volume variations in those fluid regions characterised by high
   > velocity strain rates. Nonetheless, the scheme can be stabilised if
   > appropriate diffusion terms are included in **both** the equations of
   > density and mass."

   `𝒟^ρ` is the δ-SPH density diffusion we already have (and just fixed, see
   `ACSPH_PLAN.md` Part 3). `𝒟^m` is new, and is the mass-equation counterpart.

Route 2 is therefore **stage B′**: a genuine ALE formalism, reachable from what
is already here, and it does not compete with route 1 — it is the primitive-
variable half of the same picture, and having both is what makes the comparison
Michel's paper is built around actually possible.

## 4.1 Route 1's shared blocker — a Riemann solver

Neither ALE scheme is implementable without `(ρ_E, u_E, P_E)`. Michel does not
specify the solver, only that MUSCL [36, van Leer 1979] with a minmod limiter
[30, Roe 1986] reconstructs the conservative variables. That is a **new
subsystem**, roughly:

- an approximate Riemann solver for the Cole/Tait EOS at each pair midpoint —
  the standard SPH-ALE choice is the acoustic (linearised) solver, with HLLC as
  the accurate option;
- MUSCL reconstruction of `(ρ, ρu)` from each particle to `x_ij`, which needs
  per-particle gradients (we have `computeGradRhoL` and the renormalisation
  matrices already);
- a minmod slope limiter (`modules/crk/limiter.py` has van Leer; minmod is
  three lines next to it).

**This is independently valuable.** CRKSPH's pseudo-viscosity is an
approximation to exactly this; a real Riemann flux would give the compressible
side of the repo something it currently fakes.

## 4.2 What Vila costs beyond Parshikov

Non-constant mass is the structural one. `masses` is `constant(...)` in
**every** state class in `systems/`, and the samplers, rigid-body update and
diagnostics all assume it. Vila needs `{x, V, Vρ, Vρu}` as the integrated set,
with `ρ` and `u` derived — a new state class in the same shape as
`ArtificialCompressibleState` (which is precedent: that one moved `pressures`
from constant to integrated and `densities` the other way).

## 4.3 The good news — the apparent-volume path is already universal

Every kernel in this package already carries `useVolume, V_i, referenceVolumes`
and computes `apparentVolume = m_j/ρ_j if not useVolume else referenceVolumes[j]`
(≈20 kernels: `wp_densityDelta`, `wp_viscosityDelta`, `wp_surfaceAware`,
`wp_alpha`, `wp_mat`, `wp_wallMoment`, the CRK and compSPH sets, …), threaded
from `Corrections(volumes=(queryVolumes, referenceVolumes))`.

**So an integrated `V_i` needs no kernel change at all** — it is passed as
`referenceVolumes` and every existing operator uses it. That removes what would
otherwise be the largest and most error-prone part of the Vila work.

## 4.4 Recommendation

| stage | what | blocked on | size |
|---|---|---|---|
| **A** | The Michel PST (Eqs. 21/22/47/48) | nothing | medium |
| **B′** | **δ-ALE-SPH** (`antuono2021`): integrated `V`, `𝒟^ρ` + `𝒟^m` | A (it needs a PST), nothing else | **small-medium** |
| **B** | Riemann + MUSCL + minmod subsystem | `vila1999`/`oger2016` would help | medium-large |
| **C** | Parshikov & Medin scheme | B | small-medium |
| **D** | Vila ALE scheme | B, C | medium |

**A first** — it has a waiting consumer (ACSPH), no dependencies, and no
literature gap. **Then B′**, which is the answer to "is now the time for an
ALE": yes, and sooner and cheaper than the question assumed, because
`antuono2021` reaches one without a Riemann solver and we are already two thirds
of the way there. B′ also gives stage A its sharpest test — Michel §3.1's point
that an ALE relaxes the PST consistency requirement is only checkable with an
ALE in hand.

**B, C, D remain worth doing** and are the second half of the picture: a
Riemann-flux SPH is something this repo genuinely lacks (CRKSPH's
pseudo-viscosity approximates one), and Vila + Parshikov are what make Michel's
central claim — that a good PST is *scheme-independent* — testable across four
schemes rather than asserted on one. But they are a bigger, separable
investment, and nothing upstream is blocked on them.

---

# Part 5 — What has to be built

## 5.1 Stage A — the PST itself

- **`modules/shifting/michel.py`**, new `ShiftingScheme.michel2022`. Lives in
  `modules/shifting/` because it is scheme-independent — δ-SPH, ACSPH and (later)
  Parshikov all consume it.
- **`U_char_i = max_j |(u_j−u_i)·x̂_ij|`** — a new neighbour-loop reduction. A
  `max` over a runtime-length loop is exactly the shape
  `scripts/gradcheck_incompressible.py`'s docstring warns about (`wp.max` in
  `computeVsigWarp`), so it gets its own gradcheck from the start, and the
  accumulator must be *returned* before any nonlinear read.
- **`∇̃C_i` audit** — confirm `sample/wp_deltaShift.py` carries Eq. (3)'s
  `1 + 0.2(W_ij/W(Δx))⁴` tensile factor, and that `W(Δx)` uses the *achieved*
  spacing (`achievedLatticeSpacing`), not nominal `dx`.
- **`d_i^FS` and `ñ_i`** — distance to, and normal of, the nearest free-surface
  particle. Extend `modules/surfaceDetection/wp_dilate.py`, which already walks
  that neighbourhood.
- **Eq. (48) projection** — a new `ShiftingProjectionScheme.michel2022`
  alongside `surfaceNormal`: same `λ<0.4` gate, but `σ` from `d^FS` rather than
  the current `surfaceScaling`, the `λ²` prefactor, and `ñ` instead of `n`.
- **`β_i` linear decay** from `(R/Δx)³` to 1 across the free-surface region.
- Wire into ACSPH's `ArtificialCompressibleSystem.finalize` as the Eq. (58)
  displacement, and retire `noPenetrationShift` if it carries the corners.

## 5.2 Stage B′ — δ-ALE-SPH

- **`volumes` becomes an integrated field** on a new `DeltaAleState`, with
  `dV_i/dt = V_i ∇·(u+δu)_i` and `m_i = ρ_i V_i` no longer constant. Per §4.3
  this needs **no kernel change** — the volume is passed as
  `queryVolumes`/`referenceVolumes` and every operator already reads it.
- **`𝒟^ρ`** is `computeScalarFieldDiffusion` (`ACSPH_PLAN.md` step 3 made it
  field-agnostic), already in place.
- **`𝒟^m`, the mass-equation diffusion, is the new physics.** Read it off
  `antuono2021` §4 rather than guessing; it is the paper's contribution and the
  thing without which the scheme is unstable.
- The Lagrangian limit (`δu = 0`) and the constant-mass limit (§5) must both
  reduce to schemes we already have — `deltaSPH`, and `deltaSPH` with
  `correctdrhodt`/`correctdvdt` respectively. **Those two reductions are the
  acceptance test**: they are exact, cheap, and they catch a sign or a factor in
  the volume equation immediately.

## 5.3 Stage B — Riemann and reconstruction

- **`modules/riemann/`**: an acoustic solver first (closed form, no iteration),
  HLLC behind an enum. Inputs are the MUSCL-reconstructed left/right states at
  `x_ij`; outputs `(ρ_E, u_E, P_E)`.
- **MUSCL reconstruction** to the pair midpoint using per-particle gradients.
  `computeGradRhoL` generalises (it already takes `field=`); the velocity needs
  a matching tensor gradient.
- **minmod** next to `computeVanLeer` in `modules/crk/limiter.py`.
- Gradcheck all of it. The limiter branches are the risk — `limiter.py`'s own
  docstring records that a zero-denominator branch must return *before*
  dividing, or reverse-mode AD differentiates the unguarded division.

## 5.4 Stages C and D — the two Riemann schemes

Both follow the `ACSPH_PLAN.md` step 4 pattern exactly: enum member, state,
system, config, round-trip, registration, then physics. C reuses
`WeaklyCompressibleState` with `volumes` added as integrated; D needs the
`{x, V, Vρ, Vρu}` state.

---

# Part 6 — Literature status

**Nothing in this plan is blocked on a document we do not have.** All eight
references it leans on were synced on 2026-09-05 (`literature/ADDING.md`
procedure; `scripts/check_literature.py` green), and every bibliographic field
below came from a DOI record rather than from the papers' own reference lists —
which caught one disagreement worth having: `michel2022` cites [27] as
*Quinlan, Lastiwka, Basa*, and the DOI record has *Quinlan, Basa, Lastiwka*.

| ref | key | what it unblocks |
|---|---|---|
| — | `michel2022` | The target: the requirements (Part 2), the law, and the three-scheme validation. |
| — | `antuono2021` | **Stage B′.** δ-ALE-SPH: an ALE in primitive variables, no Riemann solver, and the `𝒟^ρ`/`𝒟^m` diffusion without which it is unstable. Its §5 constant-mass variant is what we already have. |
| [25] | `oger2016` | Stage B/D context: the weakly-compressible ALE study, and Table 1's Mach-scaled row. |
| [37] | `vila1999` | **Stage D.** The ALE formalism itself, and *why* the conservative-variable route makes a Riemann solver mandatory. |
| [26] | `parshikov2002` | **Stage C.** The contact-algorithm scheme `michel2022` §4.2.2 recovers from Vila. |
| [17] | `lind2012` | Table 1's first row — the Fick's-law PST everything here descends from, including `delta.py`. |
| [27] | `quinlan2006` | The truncation-error analysis §3.1's consistency requirement and `β = (R/Δx)³` rest on. |
| [36] | `vanleer1979` | MUSCL, for stage B. |
| [30] | `roe1986` | minmod, and a survey of the approximate Riemann solvers stage B must choose between. |

Two provenance notes, both recorded in the entries themselves:

- **`vila1999`'s copy is an author's version**, paginated 1–48 rather than the
  published 161–209. Fields are the DOI record's.
- **`roe1986` has no abstract** and therefore no `ABSTRACTS.md` block — Annual
  Reviews articles of that vintage print none. OpenAlex *does* return an
  abstract for its DOI, but it belongs to a different paper (something about
  machine learning in fluid mechanics); it is not used. This is the only core
  row in the library without an abstract, and `MANIFEST.md` names the exception.

`antuono2021` also arrived with a corrupted text layer — a floating `δu`
overline glyph lands mid-word and one hyphen is lost. That is now a declared
`**text-layer:**` repair, a mechanism added to `check_literature.py` for it: the
repairs are applied to the extracted text before matching, so the verbatim check
still runs at full strictness against a transformation written down in the file
rather than being switched off. See `literature/ADDING.md`.

# Part 7 — Validation

Michel's own cases, in his order — all three already exist here:

| § | case | repo case | measures |
|---|---|---|---|
| 4.3.1 | Inviscid 2D Taylor–Green | `tgv` / `tgvWeaklyCompressible` | `δu_max` **convergence rate** (the headline: first order), pressure field, KE decay |
| 4.3.2 | Viscous Taylor–Green, Re=100 | same, `nu>0` | same, plus the analytic decay |
| 4.4 | Uniform translation invariance | `tgv` + a superposed constant velocity | `δu` must be **unchanged** — the sharpest single test of requirement 2 |
| 5.4 | Free surface | `rotatingSquarePatch` | `δu_max` convergence with a surface present |
| 6.2 / 7.3 | 2D and 3D jet on a flat plate | `impact` | surface + solid boundary together |

**The acceptance test is §4.4, and it is nearly free.** Add a uniform velocity
to a Taylor–Green initial condition and the PST field must not move. Our current
law will fail it (its `U_lim = 0.5 U_max` sees the offset); Michel's must pass
to machine precision. That is one probe script and it discriminates the whole
requirement-2 claim.

Second acceptance test: `δu_max` vs resolution at fixed `R/Δx`. Michel's law
converges at first order; ours should be flat. Fig. 1's bottom row is the
picture to reproduce.

---

# Part 8 — Sequencing

1. **Sync `oger2016`, `vila1999`, `antuono2021`** (`literature/ADDING.md`). Cheap,
   and stage B should not start without them.
2. **Stage A, the PST** (§5.1). Validate with the §4.4 translation-invariance
   probe and the `δu_max` convergence sweep on `tgv`, then on ACSPH's
   `hydrostaticColumn` (`ACSPH_PLAN.md` step 7's actual deliverable) and
   `rotatingSquarePatch`.
3. **Audit the existing shift against Table 2** and record it. If
   `ShiftingScheme.michel2022` wins on `tgv` and `sloshingTank`, propose it as
   the default — but as its own decision, with its own sweep, not folded into
   step 2.
4. **Stage B′, δ-ALE-SPH** (§5.2). Integrated volume, `𝒟^ρ` + `𝒟^m`, graded
   first by its two exact reductions (Lagrangian, and constant-mass = today's
   `correctdrhodt`/`correctdvdt`) and then on `antuono2021`'s own benchmarks —
   the inclined elliptical cylinder, the lid-driven cavity (`lidDrivenCavity`
   exists), and dam-break-on-a-wall (`dambreak` exists).
5. **Stage B, Riemann + MUSCL + minmod** (§5.3). Standalone, gradchecked,
   validated on `sod`/`sodND` against the analytic solution before any scheme
   consumes it.
6. **Stage C, Parshikov & Medin.** The cheap consumer of B, and the check that B
   is right in an SPH setting.
7. **Stage D, Vila ALE.** The `{x, V, Vρ, Vρu}` state, mass fluxes, PST in every
   equation. Then reproduce Michel Figs. 1–9 across **four** schemes — δ-SPH,
   δ-ALE, Parshikov, Vila — which is the point of the whole exercise: the PST is
   supposed to be scheme-independent, and we would finally have the schemes to
   show it on, including one from each side of the primitive/conservative
   divide.

## Relationship to the other plans

`ACSPH_PLAN.md` step 7 **is** stage A here, and is that plan's next action;
everything from stage B on is new scope this document opens. `sun2019`'s δ-ALE
terms (`correctdrhodt`/`correctdvdt`, `WCSPH_SHIFTING_PLAN.md`) are the partial
ALE stages C/D complete. `DFSPH_IMPROVEMENT_PLAN.md` is untouched by this,
except that a Mach-free PST is the first shifting law DFSPH could actually use.
