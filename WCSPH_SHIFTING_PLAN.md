# warpSPH — Weakly-Compressible Particle-Shifting Near Free Surfaces

Working document for the **δ⁺-SPH particle shift on the weakly-compressible
`deltaSPH` scheme** (`schemes/deltaSPH.py` /
`systems/weaklyCompressible.py::finalize`, gated by
`schemeConfig.shiftProperties.active`). This is **not** the DFSPH VD+PS shift —
that is `DFSPH_IMPROVEMENT_PLAN.md`'s. The two share `modules/shifting/` code
but are applied and graded separately.

Spun out of the sloshing-tank verification (`examples/sloshingTank/PLAN.md`),
where `deltaSPH` reproduced the sloshing kinematics but **diverged at the first
wave impact** (`t ≈ 3.5 s`, unchanged by 3× sound speed + 3× artificial
viscosity) — a free-surface particle-distribution / tensile instability that a
working surface shift would relieve.

---

## The problem

The δ⁺-SPH shift (`modules/shifting/delta.py`, Sun et al. 2017 scaling:
`δx = −CFL·Ma·2·h²·∇C`, `C` a number-density concentration) **is not
volume-preserving**. Applied every step near a free surface it produces a net
**outward drift of the surface particles**, so the fluid footprint grows
without bound while mass is conserved — an unphysical volume increase.

Why the drift: on the air side the kernel sum is truncated, so `∇C` at the
surface points *into the fluid* (toward the denser interior) and `−∇C` pushes
surface particles *outward* into the void. There is nothing at a free surface
to oppose it — no wall, and the deltaSPH pressure there is ≈ 0. Iterated, it
is a ratchet.

### What the code does about it today (`modules/shifting/wrapper.solveShifting`)

When `surfaceDetectionConfig.active`, the shift is suppressed near the surface,
by `shiftProperties.projectionScheme`:

- **`dot`** (default): surface particles (`fsm > 0.5`) keep only the
  *tangential* shift, scaled by `surfaceScaling` (**default 0.1**); any
  particle with `lMin < 0.4` (the min renormalisation-matrix eigenvalue, a
  kernel-deficiency proxy) has its shift **zeroed outright**.
- **`mat`**: `(I − n nᵀ)` projection scaled by `lMin²`, then surface particles
  zeroed anyway.
- fallback: surface + near-surface shift zeroed.

So the current answer to "shifting inflates the free surface" is **"don't shift
the free surface"**. That tradeoff was judged *somewhat worth it* — it keeps
the volume roughly bounded — but it leaves the surface region **under-regularised**:
particle clumping and voids form there with no shift to relieve them, which is
the mechanism behind the sloshing-tank impact divergence and the classic
`rotatingSquarePatch` corner erosion.

### Why the suppression is not enough

1. `dot`/`mat` remove the *instantaneous* normal component, not the
   *accumulated* outward displacement — drift still ratchets through
   normal-estimate error.
2. The surface normal is noisy exactly where it matters — **corners**.
3. `surfaceScaling = 0.1` on the tangential part still leaks outward motion via
   that normal-estimate error.
4. The hard `lMin < 0.4` cutoff is a sharp on/off boundary that itself seeds
   disorder one layer into the bulk.

**Goal: get the regularisation benefit at the free surface *without* the
volume growth, so the "just switch it off there" tradeoff no longer has to be
made.**

---

## Reference method — Sun et al. 2019 (`sun2019`, `literature/`)

`sun2019_consistent-particle-shifting-delta-plus-sph.pdf` — *A consistent
approach to particle shifting in the δ-Plus-SPH model*, CMAME 348 (2019)
912–934. This is the paper the root fix (§2 below) is taken from, and much of
it is **already coded in this repo but gated off and never validated** — see
"What is already implemented" at the end of this section.

### The idea: shift as an advection velocity, not a post-hoc displacement

δ⁺-SPH as originally published (Sun et al. 2017, ref [5] in the paper — the
`δx = −CFL·Ma·2h²·∇C` displacement `modules/shifting/delta.py` implements)
moves particles *after* the step with no feedback into the continuity or
momentum equations. That is the inconsistency that ratchets volume: the
density each particle carries was integrated for a trajectory it did not take.

Sun 2019 fixes it by defining a **modified advection velocity `u + δu`** (`δu`
= the shifting velocity) and a second Lagrangian derivative
`df/dt = ∂f/∂t + ∇f·(u + δu)`. Rearranging Navier–Stokes in this
"quasi-Lagrangian" frame — *not* ALE: particle masses stay fixed and
`V_i = m_i/ρ_i`, so mass is still conserved intrinsically — produces extra
divergence terms:

```
continuity:  dρ/dt  = −ρ div(u + δu) + div(ρ δu)            + δ h c₀ D_i
momentum:    ρ du/dt = −∇p + div V + ρ g + ρ div(u⊗δu) − ρ u div(δu)
trajectory:  dr/dt   = u + δu
```

The `δu`-terms (`div(ρ δu)`, `div(u⊗δu)`, `u div(δu)`) are what the 2017
scheme is missing. Paper's headline result: the **continuity `δu`-terms are
essential** in confined / periodic domains and long free-surface runs with
breaking; the **momentum `δu`-terms are negligible** for every benchmark they
tried *except* they improve angular-momentum conservation on the square patch
(Fig. 16).

### SPH form of the δu-terms (paper Eq. 9) — sign/scheme matters

```
⟨div(δu)⟩_i    = Σ_j (δu_j − δu_i)·∇_i W_ij V_j          ← difference form (as div(u))
⟨div(ρ δu)⟩_i  = Σ_j (ρ_j δu_j + ρ_i δu_i)·∇_i W_ij V_j  ← SUM form
⟨div(u⊗δu)⟩_i  = Σ_j (u_j⊗δu_j + u_i⊗δu_i)·∇_i W_ij V_j  ← SUM form
```

The sum form on the last two is deliberate: by antisymmetry
`Σ_i ⟨div(ρδu)⟩_i V_i = 0`, so the global continuity integral collapses
*exactly* to `Σ_i [dρ_i/dt + ρ_i⟨div(u+δu)⟩_i] V_i = 0` (paper Eq. 16). That
identity is the volume-conservation guarantee. Get the difference/sum split
wrong and you lose it.

### Shifting velocity (paper Eq. 13–14)

```
δū_i = −Ma (2h) c₀ Σ_j [1 + R (W_ij / W(Δx))ⁿ] ∇_i W_ij V_j ,   R = 0.2, n = 4
δu*_i = min(‖δū_i‖, U_max/2) · δū_i/‖δū_i‖                        (robustness limiter)
```

`Ma = U_max/c₀`. Proportional to `h` → `δu → 0` as `h → 0`, which is what keeps
`u + δu` a small perturbation of the true trajectory (the whole
quasi-Lagrangian argument depends on this). Same scaling family as
`delta.py`'s `−CFL·Ma·2h²·∇C` (that is `δū·dt_acoustic`).

### Free-surface treatment (paper §2.4, Eq. 20–21)

`λ_i` = min eigenvalue of the renormalisation matrix `L_i`; Marrone 2010
detection gives the free-surface particle set; `F` = particles interacting
with ≥ 1 surface particle; normal `n_i = ⟨∇λ⟩_i / ‖⟨∇λ⟩_i‖`. Then:

| condition | shift applied |
|---|---|
| `λ_i < 0.55` and `i ∈ F` | `δu_i = 0` |
| `λ_i ≥ 0.55`, `n_i·δu*_i ≥ 0`, `i ∈ F` (moving **toward** surface) | `δu_i = κ_i (I − n_i n_iᵀ) δu*_i` (tangential only) |
| `λ_i ≥ 0.55`, `n_i·δu*_i < 0`, `i ∈ F` (moving **away** from surface) | `δu_i = δu*_i` (**full, unconstrained**) |
| `i ∉ F` | `δu_i = δu*_i` |

`κ_i = 0` if `max_j arccos(n_i·n_j) ≥ 15°`, else `1` — a curvature gate: if the
surface radius of curvature is below the kernel radius, kill the shift. (15° is
calibrated for C² Wendland, `h = 2Δx`.)

The **toward/away discrimination and the κ gate are the new-in-2019 parts** and
are what stop particle clusters piling up on the surface. §2.5 adds the
free-surface / solid-wall corner fix: compute `λ` *with* ghost particles for the
normal, then recompute `λ` *without* ghosts for the Eq. (20) branch test so a
thin fluid tongue running along a wall is still recognised as surface.

### Validation paths (what to reproduce, and what each one actually grades)

| # | case | knobs | graded quantity | δ⁺-2017 vs consistent |
|---|---|---|---|---|
| **TC1** | **Taylor–Green** (periodic, no surface) — Figs 2–9 | Re ∈ {100, 1000}; `L/Δx ∈ {50, 200, 800}` (Fig. 5), `400` for the A/B (Figs 6–9) | centre pressure vs `f(t₀)e^{−16π²ν(t−t₀)}`; KE; **`ϵ_V(%) = |Σ_j V_j/L² − 1|·100`** (Eq. 23) | consistent keeps `ϵ_V < 1%` both Re; δ-SPH → ~10% (Re 1000), δ⁺-2017 plateaus at ≈ initial-volume error. **Mean-pressure drift tracks volume drift.** This is the **periodic-domain regression** (`tgv`/`kolmogorov` must not regress). |
| TC2 | flow past inclined ellipse, Re 1000 — Figs 10–12 | `L/Δx = 400` | `ϵ_V` (Eq. 24); drag/lift vs DVH | δ⁺-2017 does **not** drift here (`ϵ_V < 0.02%`) — an inflow BC pins the pressure. δu-terms negligible. Least relevant to us. |
| **TC3** | **rotating square patch** — **Figs 13–18** | `L/Δx = 400` (Figs 13–17); `100/200/400` (Fig. 18) | see below | — |
| **TC4** | **shallow-water sloshing** (gravity, breaking every cycle) — Figs 19–24 | Bouscasse "Series 5": `A/H = 2.333`, `H/L = 0.03`, `ω = 1.231 ω_r⁽¹⁾`, `H/Δx = 24` | wave height @ `x/L = 0.05` vs experiment; wall pressure @ `P1 (y = H)` | δ⁺-2017 free surface **rises non-physically** from cumulated breaking-impact volume error (visible by `t > 10T`); wall-impact pressure fails for `t/T ≥ 8`. Consistent: repeatable period-to-period. §3.4.1: 3D LNG tank, 6-DOF, ~700 oscillations — δ⁺-2017 diverges in a few once violent. |

**TC3, Fig. 13 onward, in detail** (this is `rotatingSquarePatch`, the plan's
own target case):

- **Fig. 13** — fluid config + pressure at 4 times, particle cloud overlaid
  with the LFDM free surface (ref [27]). Agreement good; pressure noise-free
  (density diffusion).
- **Fig. 14** — `tω = 4.0 / 6.0 / 8.0`; `tω = 4` vs MEL-BEM free surface. Arms
  become very thin; no reference past `tω = 4`.
- **Fig. 15** — centre pressure vs MPS [19] + BEM-MEL. **δu-terms do *not*
  change this** — the patch is bounded entirely by free surface at `p ≈ 0`, so
  there is little volume error to correct here in the first place.
- **Fig. 16** — **`ϵ_M(%)` angular-momentum error** (Eq. 26). δu-terms
  **strongly reduce it**. This is the consistent formulation's main *measured*
  payoff on a free-surface case.
- **Fig. 17** — `ϵ_E(%)` KE variation (Eq. 27). Both variants dissipative
  (diffusion); consistent slightly less.
- **Fig. 18** — **`δr/Δx` map at `L/Δx = 100/200/400`**. Shows
  `δr/Δx ≤ 0.01` everywhere and `→ 0` with resolution — the "is the shift
  still a small perturbation" check.

> **Consequence for the plan's metrics.** The paper is explicit (§3.3, and
> Fig. 15 vs Fig. 9): **a fully free-surface-bounded body barely shows volume
> drift**, because `p ≈ 0` on the surface leaves nothing to inflate the way a
> confined domain's mean pressure inflates. So on `squarePatch --box` the
> graded quantities are **angular momentum** and **`δr/Δx ≤ 0.01`** (Figs 16,
> 18), not `hullArea`. `hullArea`/`sphVolume` are still worth plotting, and the
> `--circle` null experiment is still a clean artifact probe (a rotating circle
> *is* an equilibrium, so any area drift there is pure shift error), but the
> **volume-drift signal lives in TC1 (`tgv`, periodic) and TC4 (`sloshingTank`,
> breaking)** — those two must anchor the "does the fix work" claim.

### What is already implemented in this repo

`ShiftProperties.correctdrhodt` and `ShiftProperties.correctdvdt`
(`configurations/moduleConfigurations/shifting.py`, **both default `False`**,
no test coverage) already compute the paper's Eq. (9)–(10) `δu`-terms in
`systems/weaklyCompressible.py::finalize` (lines ~150–227):

- `correctdrhodt` → `⟨div(ρ δu)⟩_i` (`GradientScheme.Summation`) `− ρ_i ⟨div(δu)⟩_i`
  (`GradientScheme.Difference`), added to `dρ/dt` — **exactly the Eq. 9
  continuity δu-terms, with the paper's sum/difference split**. This is the
  high-value one.
- `correctdvdt` → `−u_i ⟨div(δu)⟩_i` + `⟨div(u⊗δu)⟩_i` added to `du/dt` — the
  Eq. 10 momentum δu-terms. Paper says negligible; low priority.

`modules/shifting/wrapper.py`'s surface projection is a **lossy port of
Eq. (20)**:
- it removes the normal component for *every* `fsm > 0.5` particle — it is
  **missing the `n·δu* < 0` → full-shift branch** for particles moving away
  from the surface (the anti-clustering mechanism);
- it gates on a hardcoded `lMin < 0.4`, not the paper's `λ < 0.55`, and the
  threshold is not exposed via config;
- there is **no `κ` (15° curvature) gate** at all.

So the concrete first moves for §2 are: (a) turn `correctdrhodt` on and
validate against TC1 (`tgv` `ϵ_V`) + TC4 (`sloshingTank`); (b) port the full
Eq. (20)–(21) branch structure into `wrapper.py`. Neither is new code from
scratch.

---

## Target case — `rotatingSquarePatch` (`squarePatch`)

`src/warpSPH/cases/rotatingSquarePatch.py`, run as `--scheme deltaSPH`. A 2×2
fluid square, `omega = 4`, free surface on all four sides **and corners**, no
wall, no gravity. The cleanest controlled probe:

- **Area drift is directly measurable** and, for the square, the physical
  answer is *area-conserving rigid rotation* (the surface deforms into four
  growing arms — real physics, [BK] §5 / the case docstring — but the enclosed
  area is fixed).
- **The corners are where the surface normal is worst**, so it stresses the
  exact failure mode.
- `--shape circle` is the **null experiment**: a circle in rigid rotation *is*
  an equilibrium, so any area drift there is pure shift artifact.
- `--shape triangleIsosceles` / `--shape star5` are sharper corner versions.

### Case bug found while validating: the Mach number climbed with `nx`

`setupTimestep` fixed `targetDt` and let `c0` follow from the acoustic CFL, but
this case's `Umax = ω·R` is resolution-independent while `dx` shrinks — so a
fixed `targetDt` drove **`c0` *down* and the Mach number *up* with `nx`**:
Ma ≈ 0.055 at `nx = 64`, ≈ 0.16 at `nx = 192`, ≈ **0.34 at `nx = 400`** — far
outside the weakly-compressible regime, and the patch fragmented into blobs by
`tω = 4` *regardless of the shift treatment* (all of `shiftOff` /
`surfaceZeroed` / `surfaceNormal`; the nx=400 pressure figure shows it). The
early `nx = 64/96` validation was only clean because those happen to sit near
Ma 0.05.

**Fixed** (`rotatingSquarePatch.py::_setupTimestep`, new `mach` param, default
0.05): pick `targetDt` so `Umax/c0 = mach` at every resolution — `c0 ∝
1/targetDt`, so one probe call rescales it. `mach=None` restores the old
fixed-`targetDt` behaviour. Cost: `dt` now scales with `dx`, so `nx = 400`
needs ~13.5 k steps to `t = 1` (was ~2 k). All later square-patch numbers use
this.

### Second issue: the arms fragment at late `tω` — it is under-resolution, not a bug

Even at fixed Ma = 0.05 the arms are **coherent through `tω ≈ 3` then shatter
into a string of blobs** — for **all** of `shiftOff` / `surfaceZeroed` /
`surfaceNormal` alike (so not a shift regression), and it is **invisible to the
area metrics** (a beaded arm has nearly the same hull area / RMS radius /
per-particle density), which is why the visual check was needed.

`scripts/probe_squarePatchFragmentation.py` settles the cause. The shatter time
(the `tω` where `surfaceFraction ≥ 0.9`) rises steadily with resolution:

| `nx` | `L/Δx` | `tω` shatter, `shiftOff` | `tω` shatter, `surfaceNormal` |
|---|---|---|---|
| 72 | ~24 | 1.60 | 1.60 |
| 96 | ~32 | 2.09 | 2.09 |
| 144 | ~48 | 2.60 | — |
| 216 | ~72 | 3.19 | — |
| 288 | ~96 | 3.33 | **4.04** |
| 384 | ~128 | 3.58 | **4.58** |

Two findings:

1. **It is under-resolution.** The arms are pressureless SPH filaments with no
   surface-tension model; they hold only while a few particles thick. The
   paper's `L/Δx = 400` (`nx ≈ 1200`) holds them past `tω = 8` — which is what
   Fig. 14 shows. Nothing to fix in the scheme. Ruled out (all identical to
   ≥ 3 decimals): **kernel** (Wendland2 vs Wendland4 shatter within one output
   sample), **IC** (`samplingScheme` `regular` vs `jittered` byte-identical),
   the δ⁺ tensile term (present, `R = 0.25, n = 4`), the negative-pressure
   force switch (`pressureForceTerm = Antuono`, active).
2. **`surfaceNormal` extends arm coherence once the shift can bite.** Below
   `nx ≈ 216` the shift is too weak to change the shatter time; at `nx ≥ 288`
   `surfaceNormal` pushes it out by `Δtω ≈ 0.7–1.0` (3.33 → 4.04, 3.58 → 4.58)
   *and* keeps `maxρ` at ~1.02 where `shiftOff` spikes to 1.10 at `nx = 384`.
   A real payoff that the low-res runs masked.

**Consequence:** the case's default `nx = 192` (`L/Δx ≈ 64`) is fine for the
shift-work window (`tω ≲ 3`, `t ≲ 0.75`); a paper-faithful arm comparison to
`tω = 4-8` needs `nx ≳ 600`, where `surfaceNormal`'s advantage grows.

### Pressure figures at `nx = 192 / 288 / 384 / 600` (`scripts/probe_squarePatchValidationFigure.py`)

The visual A/B (shiftOff vs surfaceNormal, pressure, to `tω = 4-4.6`; `.npz`
cached for retuning) sharpens the picture at `nx ≥ 288`:

- **`surfaceNormal` regularises the bulk visibly.** The core stays smooth and
  compact through `tω = 3`; `shiftOff` develops tensile-instability speckle
  from `tω = 2` on. At `nx = 600` the pressure range is `~[−2, +3.5]` for
  `surfaceNormal` vs **`[−335, +1588]`** for `shiftOff` at the same instant —
  the base scheme's field is garbage well before the arms visibly break.
- **`surfaceNormal` arms at `nx = 600`, `tω = 3` look genuinely paper-like**:
  four clean thin coherent arms, compact core.
- **⚠ Core-pressure sign.** `surfaceNormal`'s core reads **mildly positive**
  (`p ≈ +3`, i.e. `ρ ≈ +0.02 %`) and near-constant, where a rotating patch's
  centrifugal core should be *negative* (paper §3.3 / Fig. 15). The value is
  tiny and the field is otherwise clean, but it points at the shift **mildly
  over-compressing the core** (packing particles into a region that wants to
  cavitate). ⬜ Check against the paper's Fig. 15 centre-pressure trace before
  trusting `surfaceNormal` for a quantitative pressure comparison — this is the
  one open question the figures raised.

## Metrics — `rotatingSquarePatch.diagnostics` ✅ done (2026-09-03)

Implemented in `cases/weaklyCompressible.py::squarePatchAreaMetrics`, wired into
`rotatingSquarePatch.diagnostics` (alongside `weaklyCompressibleDiagnostics`).
Emitted every step.

| metric | definition | healthy behaviour |
|---|---|---|
| `sphVolume` | `Σ_i m_i / ρ_i` (fluid) | flat (the SPH volume; drifts only with ρ error) |
| `hullArea` | convex-hull area of the fluid point cloud (monotone chain, no `scipy`) | flat modulo the physical arms; **the "inflation" number** |
| `rmsRadius` | `sqrt(Σ_i m_i ‖x_i − x_cm‖² / Σ_i m_i)` | grows if the patch spreads |
| `surfaceFraction` | `(surfaceIndicators == 1).sum() / N_fluid` | flat; grows if the surface frays; `nan` at t=0 (indicators not yet set) |
| `cornerRetention` | mean fluid extent along the 4 initial corner diagonals ÷ its t=0 value (baseline cached in `ctx.scratch`) | ~1 until the arms form; < 1 = corner erosion |

Plot each vs `t`. A working surface shift keeps `sphVolume`, `hullArea` and
`rmsRadius` flat (arms aside) with the shift **fully active at the surface**.
Per `sun2019` §3.3, `sphVolume` barely moves on a fully surface-bounded body
regardless — `hullArea`/`rmsRadius`/`cornerRetention` and the `--circle` null
are the discriminating signals here.

---

## Actionable list

> **Status (2026-09-03):** §1 (metrics + probe) done. §3 (`surfaceNormal` =
> real Sun 2019 Eq. 20–21) done, verified, and **now the default**. §4 mostly
> done — remaining: high-res `sloshingTank` (SPHERIC TC10 grid) + wall-pressure
> vs experiment, and watch `maxρ` at higher `nx`.
> Open items:
> - **§2d is blocked** on a `correctdrhodt` correctness audit — it collapses ρ
>   to 0.42 on the violent box flow (`surfaceNormalDeltaU`). Keep it off.
> - **§2a** (background pressure) wanted to make the `circle` null valid and to
>   address the tensile instability `surfaceNormal` doesn't touch.

### 1. Baseline & metrics — "how bad is it, does the switch-off help"

- ✅ Metrics added to `rotatingSquarePatch` (see above).
- ✅ Probe `scripts/probe_squarePatchAreaConservation.py`. Modes:
  `shiftOff` / `surfaceZeroed` (today's default) / `surfaceActive`
  (`surfaceDetectionConfig.active = False` — the raw un-suppressed drift) /
  `deltaU` (+ `correctdrhodt`) / `surfaceActiveDeltaU` (the §4 target). Sweeps
  `--shapes box circle` × `--nx …`, reports `d hullArea/dt`, `d sphVolume/dt`
  (fraction of initial per unit time) + per-metric start→end %.
- ⬜ Run the matrix at `nx ∈ {96, 192, 288}` to `t = 1` (~0.6 rev at
  `omega = 4`) and record the numbers here. Confirm `surfaceZeroed` actually
  bounds the drift `surfaceActive` shows, and quantify the residual it leaks
  (point 3 above). `circle` isolates the pure artifact.

#### First numbers — `nx = 64`, `t = 0.6` (~2.4 rad at `ω = 4`), float32

| shape | mode | steps | Δ sphVolume | Δ hullArea | Δ rmsRadius | maxρ | verdict |
|---|---|---|---|---|---|---|---|
| box | shiftOff | 1200 | +0.01 % | +365 % | +78 % | 1.001 | healthy (hull/rms growth is the physical arms; ρ tight) |
| box | surfaceZeroed *(default)* | 1200 | +0.01 % | +365 % | +78 % | 1.001 | **bit-identical to shiftOff** — the default shift does ~nothing on this case |
| box | deltaU (`+correctdrhodt`) | 1200 | +0.01 % | +365 % | +78 % | 1.001 | also identical — the `δu`-terms have nothing to correct on a surface-bounded body (`sun2019` §3.3) |
| box | surfaceActive *(no suppression)* | 1200 | **−12 %** | +3390 % | +267 % | **1.28** | shift runs riot at the surface: rms spreads 3× the physical rate, ρ error blows to 28 % |
| box | surfaceActiveDeltaU | **269 → NaN** | — | — | — | — | `correctdrhodt` does **not** rescue an unsuppressed surface shift — it diverges faster |
| circle *(null)* | shiftOff | 1200 | +0.02 % | **+46 %** | +15 % | 1.001 | the circle is **not** a clean equilibrium here — negative-pressure-core tensile instability spreads it even with no shift |
| circle | surfaceZeroed / deltaU | 1200 | +0.02 % | +46–48 % | +16 % | 1.001 | no better than shiftOff — suppressed shift can't regularise the surface it's suppressed on |
| circle | surfaceActive | 1200 | −14 % | +5400 % | +417 % | 1.25 | same blow-up as the box |
| circle | surfaceActiveDeltaU | **208 → NaN** | — | — | — | — | diverges |

**Second pass — with `ShiftingProjectionScheme.surfaceNormal` (§3), same grid:**

| shape | mode | steps | Δ sphVolume | Δ hullArea | Δ rmsRadius | minρ / maxρ | verdict |
|---|---|---|---|---|---|---|---|
| box | **surfaceNormal** | 1200 | **+0.011 %** | +444 % | **+80 %** | 0.998 / **1.001** | ✅ **stable, volume-bounded, surface shift active** — rms only 2 pts over the physical +78 %, ρ bounds *tighter* than shiftOff. The §3 win. |
| box | surfaceNormalDeltaU | 1200 | **+3.6 %** | +335 % | +75 % | **0.425** / 1.35 | ⚠️ `correctdrhodt` wrecks the density field (ρ → 0.42) even on top of a working surface shift |
| circle | surfaceNormal | 1200 | +0.016 % | +65 % | +16 % | 0.998 / 1.001 | ρ tight; hull spreads a bit *more* than shiftOff (+65 vs +46 %) — the tensile instability is unchanged, shifting isn't its fix (needs §2a) |
| circle | surfaceNormalDeltaU | 1200 | −0.002 % | +60 % | +18 % | 0.997 / 1.003 | fine here — the box ρ-collapse is specific to the more violent flow, but `correctdrhodt` is clearly fragile |

**Third pass — `nx = 96`, `t = 1.0` (~4 rad, arms fully formed):**

| shape | mode | Δ sphVolume | Δ hullArea | Δ rmsRadius | minρ / maxρ |
|---|---|---|---|---|---|
| box | shiftOff | +0.003 % | +1148 % | +204 % | 0.986 / 1.013 |
| box | surfaceZeroed | +0.025 % | +1174 % | +203 % | 0.980 / 1.015 |
| box | **surfaceNormal** | **−0.013 %** | +1249 % | +207 % | 0.982 / **1.020** |
| circle | shiftOff | +0.007 % | +534 % | +105 % | 0.995 / 1.003 |
| circle | surfaceZeroed | +0.005 % | +550 % | +107 % | 0.998 / 1.001 |
| circle | **surfaceNormal** | +0.011 % | +603 % | +106 % | 0.991 / 1.003 |

`surfaceNormal` holds at the higher resolution / longer time: `sphVolume`
within 0.03 % on every cell, `rmsRadius` within 4 pts of the shiftOff
(physical-arms) value, no divergence. It spreads `hullArea` a few % more than
the hard-zero — that is the surface shift *doing its job*, and `sphVolume`
(the real volume) stays flat. `maxρ` is the one number it loosens: 1.020 on the
box vs 1.013 for shiftOff — still well inside weakly-compressible, worth a
watch at higher `nx`.

Conclusions that change the plan:

1. **`surfaceZeroed` ≈ `shiftOff` for this case.** The δ⁺ shift's only real work
   is at the surface, and that is exactly what the default zeroes — so "does the
   switch-off help" is moot here: it isn't shifting the surface at all, and the
   arms/`rmsRadius` growth is physical. The *drift* question has to be asked on a
   case where the bulk shift matters (periodic `tgv`) or where the surface shift
   is actually allowed to act.
2. **`correctdrhodt` alone is inert on a free-surface-bounded body** — exactly
   what `sun2019` §3.3 predicts. §2d must be validated on **`tgv` (periodic
   `ϵ_V`)**, not here. It is *not* a standalone fix for the square patch /
   sloshing surface problem.
3. **`correctdrhodt` + unsuppressed surface shift diverges faster, not slower.**
   The `δu`-terms presume a shift that is already sane at the surface; Sun 2019
   pairs them with the *full* §2.4 algorithm (Eq. 20–21). So §3 (port the real
   surface treatment) is a prerequisite for §4, not an optional polish — §2d
   cannot substitute for it.
4. **The `circle` null is contaminated.** A rotating circle destabilises on its
   own here (tensile instability from the negative-pressure core: `+46 %`
   hull, `+15 %` rms with the shift off). "circle area drift < 0.5 %" is
   unreachable without a *working* regulariser, so it cannot be used as a
   shift-off baseline — only as an A/B between shift treatments once one of them
   works. Consider adding a background pressure (`§2a`) just to make the null
   valid.
5. **`surfaceNormal` (§3) is the working surface treatment.** On the box it
   keeps `sphVolume` flat (+0.011 %) and ρ in `[0.998, 1.001]` while letting the
   surface shift act (rms +80 % vs the +78 % physical floor, vs +267 % for
   unsuppressed `surfaceActive`). It is a strict improvement on both the legacy
   hard-zero (`surfaceZeroed`) and `surfaceActive`. **Next:** confirm at
   `nx ∈ {96,192,288}` / `t = 1`, then the `sloshingTank --scheme wcsph`
   transfer test, then flip the `buildDefaultShiftProperties` default (§4).
6. **`correctdrhodt` is not merely inert — it is *harmful* on a violent flow.**
   `surfaceNormalDeltaU` on the box collapses ρ to 0.42. See the audit below.

> The three tables above predate the Mach fix and the fragmentation finding.
> `surfaceNormal`'s volume/ρ numbers still hold for `t ≤ 0.75`; treat the
> `t = 1` row as inside the fragmentation regime.

#### `correctdrhodt` audit (`scripts/probe_correctdrhodtAudit.py`)

Instrumented `squarePatch --nx 96` (Ma 0.05, `surfaceNormal`, `correctdrhodt`
on), sampling every 150 steps: shift `|dx|/Δx` bulk vs surface, per-step
`Δρ/ρ`, and the global `Σᵢ Δρᵢ Vᵢ / Σᵢ ρᵢ Vᵢ` (Eq. 16 wants ~0).

| phase | `t` | `|dx|/Δx` surf p99 | `Δρ/ρ` surf p99 | global bias | `minρ` |
|---|---|---|---|---|---|
| pristine | ≤ 0.30 | ≤ 0.0007 | 0.0000 | ~1e-8 | 0.977 |
| arms forming | 0.38 | 0.0074 | 0.0026 | −8e-5 | 0.981 |
| beading | 0.47 | 0.059 | 0.043 | −1e-3 | 0.834 |
| runaway | 0.52 | 0.14 | 0.06 | −2e-3 | 0.44 |
| — | 0.57 | NaN at step ~1840 | | | |

Findings:

1. **The formula, sign, operators and Eq. 16 conservation are all correct.**
   While the flow is well-behaved (`t ≤ 0.35`) `correctdrhodt` is exactly the
   tiny volume-conserving nudge the paper describes: `Δρ/ρ` ≈ 0 and the global
   bias is ~1e-8. It was *not* mis-implemented.
2. **The failure is driven by the shift magnitude, not the correction.** Once
   the arms bead (the §"second issue" fragmentation), `∇C` at those surface
   particles explodes and `|dx|/Δx` runs 0.007 → 0.06 → 0.14 (the paper's stays
   ≤ 0.01 *because its arms never fragment*). `correctdrhodt` faithfully turns
   the oversized shift into an oversized `Δρ`, which destabilises ρ, which
   grows the shift — a positive-feedback runaway.
3. **warpSPH had no flow-scaled shift limiter.** The paper's Eq. (14) caps the
   shift velocity at `Umax/2`; warpSPH only had `wrapper.py`'s per-component
   `0.5·Δx` clamp, ~20–50× looser. **Added** `ShiftProperties.maxShiftVelocityFraction`
   (default 0.5) — an L2 cap at `frac·Umax·dt` in `solveShifting`. It holds the
   shift near the paper's `≤ 0.01 Δx` while the flow is sane, but once the flow
   itself diverges `Umax·dt` grows with it, so it slows the runaway rather than
   stopping it. `test_physics.py` 69 pass / 1 xfail with it on.

**Net:** `correctdrhodt` is sound but has **no headroom for a misbehaving
shift**. On `squarePatch` it can only be judged for `tω ≲ 3` (before the arms
shatter from under-resolution); the periodic box below is the real test.

#### `correctdrhodt` on a periodic box — it works (`scripts/probe_correctdrhodtPeriodic.py`)

`kolmogorov --scheme deltaSPH` (periodic, no free surface, no fragmentation),
volume drift `ε_V = |Σ Vᵢ / Σ Vᵢ(0) − 1|`, `nx = 64`, to `t = 3`:

| `t` | `ε_V` off | `ε_V` on |
|---|---|---|
| 1.0 | 0.009 % | 0.007 % |
| 2.0 | 0.57 % | 0.19 % |
| 2.4 | **1.17 %** | **0.23 %** |
| 3.0 | **1.74 %** | **0.16 %** |

With `correctdrhodt` **off** the drift grows monotonically past 1 %; **on**, it
**plateaus at ~0.15–0.23 %** — an ~11× reduction that stops accumulating. ρ
bounds stay tight both ways (`[0.993, 1.021]`), neither diverges. This is
`sun2019` Fig. 9 reproduced: **the implementation is correct and does its job
on the case it is for.**

**Verdict:** `correctdrhodt` is validated for periodic/confined WCSPH and
should be used there. It stays **off by default** (conservative — flipping it
touches every WCSPH case) and **off for violent free-surface flows** until the
shift is bounded to the paper's `≤ 0.01 Δx` there (needs the fragmentation
fix + the Eq. 14 limiter, which is now in but only bites while the flow is
sane). `correctdvdt` (momentum δu-terms) still untested — paper says negligible.

### 2. A volume-preserving shift (the root fix)

The drift exists because `−D∇C` is not divergence-free. Options, cheapest
first:

- **2d. Consistent δ⁺ `δu`-terms — `sun2019`. Audited: formula is correct;
  needs a bounded shift.** `ShiftProperties.correctdrhodt` feeds the shift into
  the continuity equation as `⟨div(ρ δu)⟩ − ρ⟨div(δu)⟩` (paper Eq. 9). The
  audit (§1) confirms the sign, operators and Eq. 16 global conservation are
  right — for `t ≤ 0.35` on the square patch it is the tiny nudge the paper
  describes (`Δρ/ρ ≈ 0`, global bias ~1e-8). It blows up only *after* the arms
  fragment, because the shift magnitude then explodes and `correctdrhodt`
  amplifies it. The Eq. (14) limiter (`maxShiftVelocityFraction`, now added)
  helps but cannot save an already-diverging flow. **Next:** validate on a
  periodic box (`kolmogorov` — no surface, no fragmentation) where the paper's
  `ϵ_V` drift is the target; only then consider it for free-surface cases (and
  only with `surfaceNormal` + the limiter). `correctdvdt` (momentum `δu`-terms)
  is the paper's "negligible" half — leave off.
- **2a. Background pressure** (≈ 2 lines). Add a small uniform `p_b` to the
  deltaSPH pressure (`p ← p + p_b`, or Antuono's form). A positive `p_b` at the
  free surface pushes particles *together*, directly opposing the shift's
  outward push, and needs no surface special-casing. Sweep `p_b` vs the area
  drift on `squarePatch`. Known cost: a spurious surface-tension-like force
  that rounds corners — the arms / `cornerRetention` are the tolerance test.
- **2b. Divergence-free projection of the shift field.** Before applying,
  project `δx` onto `∇·(δx) = 0` (a small least-squares / Poisson solve on the
  fluid). Volume-conserving by construction, no surface heuristic. Cost: one
  extra solve per shift iteration — reuse `modules/shifting/`'s Krylov drivers.
- **2c. Transport-velocity formulation** (Adami et al. 2013, `adami2013`).
  Builds the regularisation into the momentum equation with a constant
  background pressure, inherently conservative. Larger change; do only if
  2d/2a/2b fall short. (Sun 2019's `δu`-terms are the δ⁺-SPH-native version of
  the same consistency idea — 2d — so try that route first.)

### 3. Better surface treatment than hard-zero

- ✅ **`ShiftingProjectionScheme.surfaceNormal`** — the Sun 2019 Eq. (20)–(21)
  algorithm ported into `wrapper.py::solveShifting` (+ `_curvatureGate`). Opt-in
  (default still `mat`); flipping the default is §4.
  - `i ∈ F`, `λ ≥ threshold`, `n·δu* ≥ 0` (shift points **into** the surface,
    `n` outward) → tangential only, `κ`-gated.
  - `i ∈ F`, `λ ≥ threshold`, `n·δu* < 0` (shift points **away**) → **full,
    unconstrained** shift. This branch is the anti-clustering mechanism the old
    `dot`/`mat`/`zero` all lack.
  - `i ∈ F`, `λ < threshold` → zero. `i ∉ F` → full shift.
  - `κ` curvature gate (Eq. 21): `κ_i = 0` when any **F–F** neighbour normal
    deviates by `≥ surfaceCurvatureAngle` (default 15°). Restricted to F–F
    edges because `LambdaGrad` interior normals are ~0 and would otherwise gate
    every surface particle next to the bulk.
  - `surfaceLambdaThreshold` (default 0.4 — the old hardcoded `lMin < 0.4`
    constant, now exposed; paper's 0.55 is for a different `λ` normalisation)
    and `surfaceCurvatureAngle` are new `ShiftProperties` fields.
- ✅ Verified on the §1 probe (see the "second pass" table): `surfaceNormal`
  is stable on the box where `surfaceActive` blew up, keeps `sphVolume` flat
  and ρ in `[0.998, 1.001]`, and lets the surface shift act (rms +80 % vs the
  +78 % physical floor).
- ✅ **`sloshingTank` transfer test** (`scripts/probe_sloshingTankSurfaceShift.py`,
  `nx = 60`, `t → 4.2 s`):
  - `noShift` (the case default) → **NaN at `t ≈ 2.59 s`** (`ρ → 0`, `v → ∞`)
    — the divergence this plan exists to clear.
  - `shiftZeroed` (legacy `mat`) → survives to 4.2 s, `ρ ∈ [0.992, 1.005]`.
    (On sloshing the *bulk* shift `mat` keeps is enough — unlike the square
    patch, whose bulk is already ordered.)
  - `shiftSurfaceNormal` → survives to 4.2 s, `ρ ∈ [0.9936, 1.0067]`. **On par
    with `mat` here, and strictly better on the square patch** → safe to make
    the default.
- ✅ `nx = 96` / `t = 1` square-patch confirm (see "third pass" table):
  `surfaceNormal` holds — `sphVolume` within 0.03 %, no divergence, `maxρ`
  1.020 (vs 1.013 shiftOff, still weakly-compressible).
- ✅ **Default flipped**: `buildDefaultShiftProperties().projectionScheme`
  `mat → surfaceNormal`. `test_physics.py` **70 passed / 1 xfail** with the
  flip (covers `dambreak`, `columnCollapse`, `tgv`, `shearWave`).
  `test_implicitShiftingComparison.py` has a *pre-existing* flake when batched
  after `test_implicitShifting.py` — the failing test name varies run-to-run
  and it reproduces on a clean tree; the Krylov comparison path does not read
  `projectionScheme`.
- ⬜ (later) smooth taper in `λ` instead of the hard `surfaceLambdaThreshold`
  step; expose taper width.
- ⬜ (later, beyond the paper) cumulative normal-displacement damping so drift
  cannot ratchet through normal-estimate error. The paper's own answer to
  residual ratcheting is §2d's `δu`-terms, not a displacement tracker.

### 4. Re-enable the full surface shift + verify the payoff — ✅ mostly done

`surfaceNormal` (§3) *is* the re-enabled surface shift — it does not need
`surfaceScaling → 1.0` or a hard-zero drop, it replaces that whole mechanism.
Now the default (`buildDefaultShiftProperties`). Verified:
- ✅ `squarePatch --scheme deltaSPH` (nx 64 & 96, t up to 1): `sphVolume`
  conserved to < 0.03 %, no divergence, `rmsRadius` at the physical-arms rate.
- ✅ `sloshingTank` (nx 60, t → 4.2 s): clears the `noShift` NaN at `t ≈ 2.6 s`,
  `ρ ∈ [0.994, 1.007]`, on par with the old `mat`.
- ✅ `test_physics.py` 69 passed / 1 xfail with the flip.
- ⬜ Higher-res `sloshingTank` (nx 150, the SPHERIC TC10 grid) and the wall
  sensor-pressure trace vs experiment — the real payoff check.
- ⬜ Watch `maxρ`: `surfaceNormal` loosens it (1.020 vs 1.013 on the box at
  nx 96). If it grows with `nx`, revisit `surfaceLambdaThreshold` / add a
  `λ` taper (§3 later items).

---

## Success criteria

- `squarePatch --scheme deltaSPH`, shift fully active at the surface:
  `sphVolume` conserved to **< 1 %** over `t = 1`; `rmsRadius` growth back down
  to the `shiftOff` (physical-arms) rate, not the 3× of `surfaceActive`; corners
  sharp until the physical arms form. (`hullArea` is dominated by the arms and
  is only meaningful as an A/B between shift treatments, per `sun2019` §3.3.)
- `squarePatch --shape circle`: with a working shift, area/`rmsRadius` drift
  **below the `shiftOff` baseline** (which is itself ~+15 % rms / +46 % hull at
  `nx = 64`, `t = 0.6` — the tensile instability the shift is supposed to
  suppress). A `< 0.5 %` absolute target needs a background pressure (§2a) to
  kill the negative-pressure core first.
- Transfer: `sloshingTank --scheme wcsph` runs past the first impact without
  divergence.
- No regression on the periodic weakly-compressible cases (`tgv`,
  `taylorGreenVortex`, `kolmogorov`) — shifting there is already fine and must
  stay fine.

## Files

- `src/warpSPH/modules/shifting/wrapper.py` — `solveShifting`, the surface
  projection block. `projectionScheme` `dot`/`mat`/`zero` are the lossy legacy
  paths (hard-zero the surface set); **`surfaceNormal` is the real Sun 2019
  Eq. (20)–(21)** (§3, done), plus `_curvatureGate`.
- `src/warpSPH/modules/shifting/delta.py` — `computeDeltaShift` (the Sun 2017
  `−CFL·Ma·2h²·∇C` displacement).
- `src/warpSPH/systems/weaklyCompressible.py` — `finalize` (~L130–228), where
  the shift is applied and its `correctdrhodt` / `correctdvdt` `δu`-terms
  (Sun 2019 Eq. 9–10) are computed — both currently gated off by default.
- `src/warpSPH/configurations/moduleConfigurations/shifting.py` —
  `ShiftProperties`: `surfaceScaling`, `threshold`, `projectionScheme`,
  `correctdrhodt`, `correctdvdt`, and (new) `surfaceLambdaThreshold`,
  `surfaceCurvatureAngle`. Still to add: a `λ` taper width and a
  `backgroundPressure` (§2a).
- `src/warpSPH/cases/{rotatingSquarePatch,weaklyCompressible}.py` — the metrics
  (`squarePatchAreaMetrics`).
- `scripts/probe_squarePatchAreaConservation.py` — the §1 probe (modes incl.
  `surfaceNormal`, `surfaceNormalDeltaU`).

## Notes / prior art

- `DFSPH_FINDINGS.md` ("The published CFL is calibrated against a metric a free
  surface silences") documents the same *expanding-particle* asymmetry for the
  DFSPH stopping metric — related framing, different scheme.
- `rotatingSquarePatch --scheme divergenceFree` is separately broken
  (`DFSPH_IMPROVEMENT_PLAN.md`: corner density → 0.5, [BK] §5 method
  limitation). Keep the deltaSPH and divergenceFree `squarePatch` results
  distinct — this plan is deltaSPH only.
- The δ⁺-SPH scaling and the Michel 2022 shifting-velocity alternative are
  described in `modules/shifting/delta.py`'s docstring.
- The `surfaceNormal` default lives in `buildDefaultShiftProperties`, which
  `IncompressibleSPHConfig` also uses — but the DFSPH path only reads
  `projectionScheme` when `systems/incompressible.py::_PS_SHIFT_MODE == 'delta'`
  (default is `'cd'`, which routes through `solveIncompressible` instead), so
  the change is inert for DFSPH as shipped. `test_physics.py` full suite
  confirms no incompressible regression.
- `sun2019` (`literature/`, added 2026-09-03) is the reference for §2d and §3;
  its method and per-figure validation paths are summarised in "Reference
  method" above. The 2017 δ⁺-SPH it corrects (its ref [5], Sun et al.,
  *The δplus-SPH model*, CMAME 315 (2017) 25–49) is **not** in `literature/`;
  add it if §2/§3 work needs the original derivation.
