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
| **A** | The Michel PST (Eqs. 21/22/47/48) — **landed 2026-09-05** | nothing | medium |
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

**Landed 2026-09-05.** What actually shipped, against the checklist below:

- **`modules/shifting/michel.py`**, `computeMichelShift` /
  `ShiftingScheme.michel2022`. Consumes the tensile-corrected, pure-volume-
  weighted `∇̃C_i` and `U_char_i` below; the free-surface `β` decay is computed
  by the caller (`wrapper.py`) and passed in, since Eq. (48) requires it to
  apply *before* the interior law's norm clamp, not as a post-hoc projection.
- **`U_char_i = max_j |(u_j−u_i)·x̂_ij|`** — `modules/shifting/wp_michelUChar.py`,
  the two-pass forward-argmax + re-evaluate-outside-the-loop split
  `modules/shockCapturing/wp_vsig.py` established for exactly this `wp.max`
  hazard. Gradchecked (`scripts/gradcheck_michelUChar.py`, positions +
  velocities, plus an AdjacencyList-vs-grid consistency check) — clean.
- **`∇̃C_i` audit — done, and it found a real discrepancy.**
  `sample/wp_deltaShift.py` already had the `1 + R(W_ij/W(Δx))⁴` tensile
  factor and an *achieved* `W(Δx)` (derived from `m_j/rho0`, not nominal
  `dx`), but its weight was Sun's mean-density term
  `0.5*m_j/(rho_i+rho_j)`, not Michel's Eq. (2)-(3) plain volume `V_j` — a
  different formula, not just a naming difference. Fixed by adding a
  `volumeWeighted` flag (default `False`, so every existing caller is
  byte-identical) that selects the kernel's own already-computed but
  previously-discarded `apparentVolume` instead. Regression-gradchecked
  alongside the existing case.
- **`d_i^FS` and `ñ_i`** — `modules/surfaceDetection/wp_nearestSurfaceNormal.py`,
  the argmin variant of the same two-pass split, gated on the *raw* (undilated)
  free-surface mask. Gradchecked
  (`scripts/gradcheck_nearestSurfaceNormal.py`, positions + normals, plus a
  hand-computed forward-value sanity check) — clean.
- **Eq. (48) projection** — `ShiftingProjectionScheme.michel2022` in
  `modules/shifting/wrapper.py`: `λ<0.4` gate, `σ` from `d^FS`, the `λ²`
  prefactor, and `ñ` instead of `n`. No curvature gate (that's Sun 2019's
  addition, kept only in `surfaceNormal`).
- **`β_i` linear decay** from `1` at `d^FS=0` to `(R/Δx)³` at `d^FS=R` (the
  paper states the endpoints but not the decay's outer bound; `R` was chosen
  to match where `σ` itself bottoms out and where the dilated vicinity set's
  own extent naturally is).
- **Wired into ACSPH's `ArtificialCompressibleSystem.finalize`** as the
  Eq. (58) displacement (no `correctdrhodt`/`correctdvdt` — density is
  invariant there, nothing for them to correct). Found and fixed two real bugs
  surfaced by wiring this in: a `config` `NameError` (never fetched from
  `kwargs` in this method before), and `ArtificialCompressibleSPHConfig`'s
  shared `shiftProperties` default (`buildDefaultShiftProperties()`) being
  `active=True` with the Mach-based `deltaSPH` scheme — harmless while the
  call was unwired, but would have silently turned shifting on with the wrong
  law for every existing ACSPH config/test the moment it was wired up. Fixed
  with a dedicated `buildDefaultACSPHShiftProperties()` (`active=False`,
  `scheme=michel2022`) — no behavior change for anything that doesn't opt in,
  and the right scheme when something does. `noPenetrationShift` is untouched
  (still off by default, still not retired — that's a physics/validation call
  for the next pass, not this one).
- **Verified**: full repo test suite green; the §4.4 translation-invariance
  acceptance test (`scripts/probe_michelTranslationInvariance.py`) passes to
  `5.6e-16` for `michel2022` and, as the negative control, fails at `1.9e-2`
  for `deltaSPH` — reproducing Part 7's discriminating claim directly; an
  ACSPH smoke run (5-10 steps, `michel2022` active) stays finite with visibly
  nonzero shift on a jittered lattice.

**Deliberately not done in this pass** (next iteration, per Part 8's own
sequencing): the `δu_max` convergence sweep on `tgv`/`hydrostaticColumn`/
`rotatingSquarePatch`, re-measuring `ACSPH_PLAN.md` step 5b's
`pairedFraction 0.065` with the shift actually on, and the "propose as new
default" decision against Table 2 (Part 8 step 3) — none of those are
implementation work, they're validation runs against a law that now exists
to be measured.

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

## 7.1 Results so far (2026-09-05) — §4.4 passed; a units bug found and fixed along the way

**⚠ A first pass at this section reported a "michel2022 roughly halves the
`rotatingSquarePatch` footprint drift" finding. That finding was an
artifact of a real bug and is retracted below — read the "units bug" note
before anything else here.**

**§4.4, `scripts/probe_michelTranslationInvariance.py`.** Passes as designed,
unaffected by the bug below (an equality check is invariant to a shared
scalar factor): `max|δu(u) − δu(u+U0)|` is `5.6e-16` for `michel2022`
against `1.9e-2` for `deltaSPH` (the negative control) on a jittered
periodic lattice. Confirms Part 2's Galilean-invariance claim directly, not
just by construction.

**Units bug, found and fixed while running the measurements below.**
`computeMichelShift` was adding Eq. (22)'s `delta_u` straight into
`positions`. Eq. (22)'s `delta_u` is a **velocity**
(`ACSPH_PLAN.md` Eq. (58): `dx_i/dt = u_i + delta_u_i`), unlike
`computeDeltaShift`'s Sun-style law, whose `-CFL*Ma*2*h^2` scaling bakes in
an implicit acoustic timestep and so already returns a position-shaped
displacement (`delta.py`'s own comment on this). Confirmed numerically on a
representative jittered lattice: the un-fixed code produced a shift **~8x
the local particle spacing** per call, against Sun's ~0.07x — physically
absurd as a single displacement. Fixed by multiplying `delta_u` by the real
simulation `dt` before accumulating into positions (ACSPH has no acoustic
timestep to reuse — no sound speed — so the real `dt` is the only timescale
available, and it is exactly what Eq. (58) asks for over one real step).
`computeMichelShift` and `solveShifting`'s dispatch now both take `dt`
explicitly. Full repo test suite re-verified green after the fix.

**5.4, `rotatingSquarePatch` free surface — `scripts/probe_squarePatchAreaConservation.py --modes surfaceNormal michel2022`,
re-run after the fix.** Not yet the `δu_max`-convergence-rate reproduction
Michel's own §5.4 runs (that needs a resolution sweep reporting the rate
itself, still open) — a cheaper first cut reusing the area/footprint drift
metrics `docs/historic_plans/WCSPH_SHIFTING_PLAN.md` already built for
exactly this comparison. `box` is the benchmark (real arm growth expected);
`circle` is the null experiment (rigid rotation is an equilibrium, so any
drift there is pure shift artifact):

| shape | nx | mode | ΔhullArea% | ΔrmsRadius% |
|---|---|---|---|---|
| circle | 48 | `shiftOff` | +18.6% | +3.21% |
| circle | 48 | `surfaceNormal` | +18.7% | +3.43% |
| circle | 48 | `michel2022` | +18.3% | +2.98% |
| circle | 96 | `shiftOff` | +58.9% | +9.79% |
| circle | 96 | `surfaceNormal` | +72.2% | +10.11% |
| circle | 96 | `michel2022` | +68.6% | +9.05% |
| box | 48 | `shiftOff` | +221.4% | +48.95% |
| box | 48 | `surfaceNormal` | +255.2% | +50.82% |
| box | 48 | `michel2022` | +252.4% | +50.24% |
| box | 96 | `shiftOff` | +302.4% | +56.55% |
| box | 96 | `surfaceNormal` | +327.2% | +56.93% |
| box | 96 | `michel2022` | +325.3% | +56.52% |

**The corrected picture is much more muted than the retracted one:**
`michel2022` now tracks `surfaceNormal` and even `shiftOff` closely at every
cell — no resolvable win, on this metric, at this `tLimit=0.5`. Two
non-exclusive readings: (a) most of this benchmark's footprint growth is the
patch's own real arm-growth physics (`shiftOff` alone already shows nearly
the full drift `surfaceNormal`/`michel2022` do), which a PST only perturbs
at the margin regardless of which law; (b) now that `michel2022`'s shift is
correctly scaled by the real (small, Mach-CFL-limited) `dt` at this `mach=0.05`
setting, its per-step correction is small enough that its cumulative effect
over one `tLimit=0.5` run doesn't move a coarse area metric — the opposite
failure mode from the bug (too small to see, rather than artificially large).
**This area-drift metric does not appear to discriminate these two laws at
these settings; the real `δu_max`-convergence-rate reproduction (still open)
is the test that actually speaks to Table 2 here, not this proxy.**

**ACSPH `hydrostaticColumn`, `scripts/probe_michelHydrostaticColumn.py`,
nx=32, 200 steps, re-run after the fix.** The actual re-measurement
`ACSPH_PLAN.md` step 7 asked for. Caveat before the numbers: adaptive `dt`
means each mode's 200 steps land at a *different* final `t`, so this is
directional evidence from one seed/resolution, not a controlled comparison:

| mode | final t | `‖v‖` peak | pairedFraction (final) | nnDistP01 (final) |
|---|---|---|---|---|
| neither | 0.41 | 6.78 | 0.000 (particles scattered, not paired) | 0.759 |
| `noPenetrationShift` only | 0.91 | 0.75 | 0.043 | 0.275 |
| `michel2022` shift only | 0.62 | 2.33 | **0.000** | **0.862** |
| both | 0.80 | 1.63 | **0.000** | **0.849** |

Unlike `rotatingSquarePatch`, this case's finding **survives the fix, and
comes out cleaner**: `pairedFraction` was 0.008/0.004 (shift-only/both) under
the buggy oversized shift — already good — and is now exactly **0.000**
under the correctly-scaled one, with `nnDistP01` (higher is healthier —
closer to the ideal lattice spacing) up from 0.53-0.67 to 0.85-0.86, well
past even the safeguard-only run's 0.275. `‖v‖` peak also improved with the
fix (3.16 → 2.33 shift-only; 2.18 → 1.63 both) but is still well short of
`noPenetrationShift` alone (0.75). Two findings stand, the second
**correcting an assumption in this repo's own notes**:

1. **The shift does what a PST is for, cleanly.** `pairedFraction` —
   "exactly and only what particle shifting exists to prevent" (step 5b) —
   goes to exactly zero with the shift active, safeguard on or off, and the
   particle-spacing distribution (`nnDistP01`) is healthier than under the
   wall safeguard alone. Not a marginal effect, and not an artifact of the
   fixed bug — it got *better*, not worse, once the shift was properly
   dt-scaled down to a sane per-step magnitude.
2. **The shift does not carry the corners on its own.** `ACSPH_PLAN.md`
   Part "Decisions taken without you" item 4 speculated `noPenetrationShift`
   "should not be needed once the shift exists". Measured, both before and
   after the fix: it is still needed. `‖v‖` peak with the shift alone (2.33)
   is well above the safeguard alone (0.75); `both` together (1.63) helps
   further but still falls short of the safeguard's bound. The wall-corner
   problem and the interior-pairing problem are apparently two different
   failure modes, and the PST — faithfully implemented, and now cleanly
   fixing the one it targets — does not also fix the other.
   `noPenetrationShift` stays, and the paper's own Eq. (58)+(62) combination
   (shift plus the velocity mirror) is worth rereading against this before
   concluding anything stronger from a single seed/resolution.

**`δu_max` convergence rate, `scripts/probe_michelConvergenceRate.py` — the
real test, and it passes cleanly.** Michel's Fig. 1 headline claim,
reproduced directly rather than via the inconclusive `squarePatch` proxy
above: a jittered periodic Taylor-Green-vortex lattice, `nx` swept
16→128 at **fixed `R/Δx = 2.5`** and fixed jitter fraction (`0.3*Δx`, so the
disorder is self-similar across resolutions — essential for a fixed-`R/Δx`
claim to mean anything). Measures each law's own *velocity* form of
`delta_u` (Michel's Eq. (22), pre-`dt`, via a new `computeMichelShift(...,
returnVelocity=True)` opt-in return — `dt` is a real-timestep artifact
extraneous to the theoretical claim; Sun's `Table 1` velocity form,
`2h * Ma*c0 * grad(C)`, computed directly since `computeDeltaShift`'s public
contract returns a position delta with an implicit acoustic-timestep scaling
that would be the wrong quantity here):

| nx | Δx | `michel2022` `δu_max` | `deltaSPH` `δu_max` |
|---|---|---|---|
| 16 | 0.0625 | 0.498 | 0.352 |
| 32 | 0.0313 | 0.258 | 0.403 |
| 64 | 0.0156 | 0.137 | 0.396 |
| 128 | 0.0078 | 0.070 | 0.411 |

Log-log fit over all seven resolutions tested (16-128): **`michel2022`
slope = 0.949** — first order, matching Michel's own headline result almost
exactly. **`deltaSPH` slope = -0.063** — flat, exactly the "does not
converge" claim Table 2 makes about every pre-2022 row. This is the
strongest evidence yet that the Eq. (20)-(22) implementation itself
(`modules/shifting/wp_michelUChar.py`, `modules/shifting/michel.py`) is
correct, independent of whatever any given downstream case's `dt`/CFL
happens to do to it — which is exactly the confound that made the
`rotatingSquarePatch` proxy above inconclusive.

**`sloshingTank`, `scripts/probe_sloshingTankSurfaceShift.py --modes noShift shiftZeroed shiftSurfaceNormal shiftMichel2022`,
nx=60, tLimit=4.0 — Part 8 step 3's comparison target.** `michel2022` is a
clean, unambiguous pass, and turned up something else along the way:

| mode | steps | final t | diverged | minRho | maxRho | sensorP |
|---|---|---|---|---|---|---|
| `noShift` | 1696 | 0.34 | **yes** | 0.000 | 3.8e35 | -69832 |
| `shiftZeroed` (`mat`) | 3664 | 0.73 | **yes** | 0.000 | inf | 0 |
| `shiftSurfaceNormal` | 3407 | 0.68 | **yes** | 0.004 | inf | 3454 |
| **`michel2022`** | 20001 | **4.00** | **no** | 0.994 | 1.011 | 4019 |

`michel2022` runs the case **to completion** (the full `tLimit=4.0`, all
20001 steps) with density held to `[0.994, 1.011]` — the tightest bound of
any mode here by a wide margin, and the only one that doesn't diverge at
all.

**The regression is confirmed, root-caused, and fixed.** `docs/historic_plans/WCSPH_SHIFTING_PLAN.md`
and this very probe's own docstring record `noShift` diverging at
`t ≈ 3.5 s` (`ShiftProperties.projectionScheme`'s docstring says `t~2.6 s`)
and `surfaceNormal` *clearing* that divergence — the whole reason
`surfaceNormal` became the default. Measured here (current `HEAD`): `noShift`
diverges at `t = 0.34`, **an order of magnitude earlier** than documented,
and `surfaceNormal` — the fix — now diverges too, at `t = 0.68`.

**Root cause, confirmed directly, not just suspected:** re-ran this exact
probe (`noShift`, `shiftSurfaceNormal`) against commit `3d72163` — the parent
of `790a7c7` ("fix its psi sign"), in an isolated `git worktree` so nothing
in the working tree had to move — and got `noShift` diverging at `t = 2.76`
and `shiftSurfaceNormal` surviving the full `t = 4.0` run
(`minRho/maxRho = [0.969, 1.042]`), matching the documented pre-fix numbers
almost exactly. **`790a7c7` is the cause**, definitively.

That commit is not a bug to revert: it independently re-derives Marrone et
al. 2011 Eq. (6), ships its own O(N²) torch reference confirming the old
sign was wrong, adds `tests/test_deltaSPHDiffusion.py` pinning the
linear/quadratic-field cancellation the corrected sign restores, and its own
message is explicit that the old behavior was "diffusive either way, hence
never a blow-up, hence never caught" — i.e. the bug was silently supplying
extra numerical damping, not doing anything visibly wrong, which is exactly
why `sloshingTank`'s stability quietly depended on it. Fixing `psi`'s sign
correctly *weakens* `deltaSPH`'s density diffusion from an accidental
second-order operator back to the intended fourth-order one, and this
violent-impact case needed that accidental extra damping to survive under
`surfaceNormal`'s free-surface treatment.

**Fix applied**: `cases/sloshingTank.py`'s `deltaSPH` branch no longer
inherits the shared `buildDefaultShiftProperties()` scheme/projection pair
(`deltaSPH`/`surfaceNormal`) — it now sets
`ShiftingScheme.michel2022`/`ShiftingProjectionScheme.michel2022` explicitly
(new `--shift-scheme`/`shiftScheme` param added alongside the existing
`shiftProjection` knob, same pattern). Confirmed end-to-end with the
case's own real entry point, no test wrapper: `run(sloshingTankCase,
params={'shifting': True}, nx=60, tLimit=4.0)` → 20001/20001 steps, not
diverged, `minRho/maxRho = [0.994, 1.011]`. Full repo test suite green
(one `test_incompressibleKrylov.py` failure did not reproduce in isolation —
pre-existing test-order flakiness, unrelated to shifting). Also corrected:
`buildDefaultShiftProperties()`'s own docstring, which cited the now-false
"clears the sloshingTank NaN" claim for `surfaceNormal` — annotated with the
regression and the fix, shared default left unchanged (that is Part 8 step
3's own separate, larger decision, not something one case's fix should
force).

**§5.4, the free-surface `δu_max` convergence rate — done, `scripts/probe_michelFreeSurfaceConvergenceRate.py`.**
Same jittered Taylor-Green lattice and velocity field as the interior
convergence-rate probe above, at the same fixed `R/Δx = 2.5`, self-similar
jitter, `nx` 16→128 — except the domain is not wrapped, so the box edge is a
genuine free surface, and the measured quantity is Eq. (48)'s own
`δu_i^FS`, built from the same three real modules production code uses
(`detectFreeSurface`, `computeNearestSurfaceNormalWarp`,
`computeMichelShift(..., returnVelocity=True)`) with Eq. (48)'s projection
applied to the velocity form directly (legitimate: that projection is
homogeneous of degree 1 in the vector it's given).

A rigid-rotation field — the obvious first choice, matching
`rotatingSquarePatch`'s own `t=0` state — turned out to be **degenerate for
this specific measurement**: `U_char_i = max_j|(u_j-u_i)·x̂_ij|` is *exactly*
zero for solid-body rotation at every pair and every resolution (`v_j-v_i =
ω×x_ij` is exactly perpendicular to `x_ij`, an identity of rigid rotation, not
a resolution effect), giving an all-machine-epsilon table with no signal to
fit. Switched to the same Taylor-Green field the interior probe already
validated, which has real local strain.

The **global** `max_i` over every particle turned out to reproduce the
periodic probe's bulk numbers almost exactly — a real finding, not a bug:
Eq. (48) can only ever *shrink* a shift near the surface (project out a
normal component, scale by `λ² ≤ 1`), so the domain-wide maximum stays
pinned wherever the periodic case already had it, and never actually
speaks to the free-surface law. The metric that does is the maximum
**restricted to the dilated free-surface set `𝕍`** (`surfaceIndicator`,
i.e. exactly the particles Eq. (48) modifies):

| `nx` | `Δx` | `n` | `n` in `𝕍` | global `δu_max` | `𝕍`-restricted `δu_max` |
|---|---|---|---|---|---|
| 16 | 0.0625 | 256 | 156 | 0.498 | 0.416 |
| 24 | 0.0417 | 576 | 252 | 0.355 | 0.231 |
| 32 | 0.0313 | 1024 | 348 | 0.258 | 0.146 |
| 48 | 0.0208 | 2304 | 540 | 0.180 | 0.0606 |
| 64 | 0.0156 | 4096 | 732 | 0.137 | 0.0410 |
| 96 | 0.0104 | 9216 | 1116 | 0.0926 | 0.0177 |
| 128 | 0.0078 | 16384 | 1503 | 0.0698 | 0.0101 |

Log-log slope: **global 0.949** (reproducing the interior probe's own
0.949 almost exactly, as expected since it's measuring the same bulk
saddle point), **`𝕍`-restricted 1.811** — convergent, and *faster* than
first order. Consistent with the mechanism (the projection only ever
shrinks the shift, and does so more aggressively as a particle sits deeper
in the dilated band at fixed `R/Δx`), and clears Table 3's `lim_{Δx→0}
δu^FS = 0` bar with margin rather than merely meeting it. Confirms the
free-surface law converges; does not by itself distinguish "exactly first
order with a favourable prefactor" from "genuinely super-first-order" —
that would need a wider resolution range or a cleaner analytic test case
than a synthetic bounded Taylor-Green box, which is a possible follow-up
but not blocking.

**`hydrostaticColumn`, multi-seed/multi-resolution — done, `scripts/probe_michelHydrostaticColumnSweep.py`.**
The single-seed table above ran on a perfectly regular lattice — `hydrostaticColumn.initialConditions`'s
own jitter call is dead code, commented out, despite the case declaring a
`jitter` param nobody applies. A single regular configuration can hide (or
manufacture) symmetric artefacts a disordered one would not share, so this
re-enables that dead path from the outside (`shuffleParticles`, seeded per
run) and sweeps `nx ∈ {24, 32, 48} × seeds {0, 1, 2}` across the same four
modes, at 50 steps rather than 200 (measured cost ~1-2 s/step at this
particle count, so the full matrix at 200 steps would run hours; the
per-step diagnostics below stabilise well before 50 steps in the single-seed
run, so this trades run length for breadth):

| `nx` | mode | `‖v‖` peak (mean±std, 3 seeds) | `pairedFraction` (mean±std) |
|---|---|---|---|
| 24 | `neither` | 3.512±0.690 | 0.0637±0.0171 |
| 24 | `noPenetrationShift` | 0.857±0.338 | 0.0637±0.0016 |
| 24 | `michelShift` | 1.036±0.492 | 0.0347±0.0000 |
| 24 | `both` | **0.663±0.227** | 0.0417±0.0057 |
| 32 | `neither` | 3.306±0.801 | 0.0501±0.0160 |
| 32 | `noPenetrationShift` | 0.612±0.068 | 0.0495±0.0066 |
| 32 | `michelShift` | 0.470±0.036 | 0.0169±0.0074 |
| 32 | `both` | **0.443±0.042** | **0.0130±0.0049** |
| 48 | `neither` | 3.019±0.082 | 0.0556±0.0118 |
| 48 | `noPenetrationShift` | 0.552±0.014 | 0.0532±0.0029 |
| 48 | `michelShift` | 1.049±0.896 | 0.0240±0.0047 |
| 48 | `both` | **0.415±0.009** | 0.0229±0.0032 |

**This refines, and partly overturns, the single-seed conclusion rather than
just confirming it.** Two things hold up: the shift consistently lowers
`pairedFraction` relative to `neither` at every resolution and seed (the
core claim), and `both` is the uniformly best *and* most seed-stable
configuration at every resolution (lowest mean `‖v‖` peak, and — at nx 32/48 —
the lowest variance too). But the single-seed claim that "the shift alone
does not carry the corners the way the safeguard does" **does not
robustly replicate**: at nx=32, `michelShift` alone (0.470) actually beats
`noPenetrationShift` alone (0.612) on mean `‖v‖` peak; at nx=48 the two are
comparable outside one outlier seed (`michelShift` seed 0 hit 2.316, seeds
1-2 gave 0.42-0.43, matching `noPenetrationShift`'s 0.53-0.57 closely); only
at nx=24 is `michelShift` clearly the worse of the two. The pattern that
*does* hold at every resolution is a variance asymmetry:
`noPenetrationShift`'s `‖v‖` peak is tightly seed-stable (std 0.014-0.338,
shrinking with resolution) while `michelShift` alone is not (std 0.036-0.896,
non-monotone in resolution, driven by occasional bad seeds). So the
single-seed run's finding was directionally real (the regular lattice
happened to be a seed where the shift alone underperforms) but overstated as
a general claim — **the safeguard's real advantage is consistency across
configurations, not a categorically lower velocity**, and `both` together is
the actually-robust recommendation, not `noPenetrationShift` alone.

Also visible: `pairedFraction` under the shift is no longer *exactly* 0.000
here (0.0130-0.0347 vs. the single-seed run's exact zero) — most likely
because that run went to 200 steps on the regular lattice and this one stops
at 50 on a jittered one, and the metric is a slow-annealing steady-state
quantity, not because the earlier finding was wrong; still, at every
resolution and every seed, shift-active modes come in at roughly half of
`neither`'s value or better, so the core claim survives even without a long
run to confirm the exact-zero endpoint.

---

# Part 8 — Sequencing

1. **Sync `oger2016`, `vila1999`, `antuono2021`** (`literature/ADDING.md`). Cheap,
   and stage B should not start without them.
2. **Stage A, the PST** (§5.1). **Implemented and wired in, 2026-09-05;
   §4.4's translation-invariance probe passes, and so does the real
   `δu_max` convergence-rate reproduction** (Fig. 1's headline result:
   `michel2022` measures first order, slope 0.949, against `deltaSPH`'s flat
   -0.063 — see §7.1). A units bug was found and fixed along the way
   (`computeMichelShift` was applying Eq. (22)'s *velocity* as if it were
   already a position delta) — the `hydrostaticColumn` finding below is
   post-fix. `pairedFraction` does move — 0.065 → **exactly 0.000** — but
   `‖v‖` at the corners does not; the shift and `noPenetrationShift` turn out
   to fix two different failure modes, not one. **The free-surface `δu_max`
   rate (Michel §5.4) is also done now** (`scripts/probe_michelFreeSurfaceConvergenceRate.py`,
   §7.1): slope 1.811 on the dilated free-surface set, faster than first
   order. **The multi-seed/multi-resolution version of the column table is
   also done now** (`scripts/probe_michelHydrostaticColumnSweep.py`, §7.1):
   it confirms the `pairedFraction` claim and that `both` (shift +
   safeguard) is the uniformly best and most seed-stable configuration, but
   refines the single-seed corner-velocity claim — `michelShift` alone is
   not categorically worse than `noPenetrationShift` alone across
   resolutions/seeds, just less seed-stable. See §7.1 for the full table.
3. **Audit the existing shift against Table 2** and record it — **the real
   evidence is now in, and it is decisive on the interior claim**: the
   convergence-rate sweep (§7.1) confirms `michel2022` is first-order
   convergent and `deltaSPH` is not, exactly as Table 2 predicts. The
   `rotatingSquarePatch` footprint-drift proxy tried first was inconclusive
   (post-fix, `michel2022` tracks `surfaceNormal`/`shiftOff` closely) — not a
   contradiction, just the wrong metric for this claim. **`sloshingTank`
   (§7.1) is done and closed out**: `michel2022` is the only mode of four
   tested that completes the run, `noShift`/`surfaceNormal`'s early
   divergence is root-caused (confirmed by re-running against the pre-fix
   commit in an isolated worktree: `790a7c7`, a correct and independently
   verified fix, removed accidental extra diffusion this case's stability
   depended on — see `ACSPH_PLAN.md` decision 1), and the fix is applied:
   `cases/sloshingTank.py`'s `deltaSPH` branch now defaults to
   `michel2022`/`michel2022` rather than the shared default, confirmed
   end-to-end through the case's real entry point. **Michel §5.4's own
   free-surface convergence rate is now done too**
   (`scripts/probe_michelFreeSurfaceConvergenceRate.py`, §7.1): the
   `𝕍`-restricted `δu_max` converges at slope 1.811 (faster than first
   order) against the interior law's own 0.949, clearing Table 3's
   `lim_{Δx→0} δu^FS = 0` claim with margin. That closes out the interior
   *and* free-surface halves of the Table 2/3 audit; proposing a *shared*
   default for every WCSPH case remains its own separate decision.
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
