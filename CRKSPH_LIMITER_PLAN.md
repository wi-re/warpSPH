# warpSPH — CRKSPH limiter & Gresho spin-up: observations and plan (CRKSPH_LIMITER_PLAN)

> **Progress marker:** this plan's row in [PLANS.md](PLANS.md). Update its *Last worked* date and *Where it stands* whenever you work on this plan.

Observations from the warpSPHCore higher-order convergence harness
(2026-09-26) that turned out to be frontend CRKSPH questions, and a plan to
dig into the **parameter stability** of the CRKSPH viscosity limiter. Related
to [`AV_PLAN.md`](AV_PLAN.md) (the dissipation roadmap; its Phase 3 is the
Frontiere et al. 2017 slope-limited reconstruction this repo already has) and
to warpSPHCore's `higher_order.md` (Phases 1–2, "Gresho anomaly"), but the
work here is frontend work.

Reference paper: `frontiere2017` — Frontiere, Raskin & Owen (2017), *CRKSPH*,
JCP 332, 160 (`literature/frontiere2017_crksph.pdf`). Equation numbers below
are theirs.

**Ground rules for this plan** (from review of the first round):

* CRKSPH evaluations use **no viscosity switch** (`viscositySwitch=
  'NoneSwitch'`). A switch is not an avenue for fixing anything here.
* The limiter is delicate: tweaks that cure one case easily destabilise
  another or add ripple. **Every** limiter / viscosity / pair-force change is
  run on CRK Sod alongside (ripple probe below), and at the end Sod must match
  the paper's Fig. 4 as closely as possible.
* CRKSPH needs symmetric support (`KernelMeanSymmetric`) for exact energy
  conservation (Sod, Gather: −7.3e-6 total-energy drift vs 2.5e-16) — now a
  `CRKSupportWarning` (`6574d01`).

---

# Status board

*Updated 2026-10-01 (end of the limiter close-out session).*

| Item | State |
|---|---|
| Gresho spin-up: detection | **done** -- warpSPHCore `ke_rebound` check; also `tests/test_crkRegression.py` |
| Gresho spin-up: energy source located | **done** -- CRK pair *pressure* forces (non-central), §O1; per-radius budget §P3 note (g) |
| Gresho spin-up: fix | **no fix found; closed as intrinsic** (P3, notes (b), (e), (f), (g)): not the cusps, not the pair weights, not a support inconsistency; limiter / AV only set how much of a persistent, slowly converging pressure pump is cancelled |
| Limiter constants in the wrong units | **fixed 2026-10-01** -- default now (1/n_h, 0.2/n_h), derived per step (note (f)) |
| CRK run-to-run nondeterminism | **fixed 2026-10-01** (note (a)) |
| Parameter-stability map (P1) | **done** (reduced) -- a smooth viscosity knob, no robust optimum (note (c)) |
| Limiter variants (P2) | derive-from-n_h **done** (it is the default); local-spacing and softer-threshold variants **not done, parked** |
| Decision and defaults (P4) | **decided 2026-10-01 (user)** -- (1/n_h, 0.2/n_h), `C_l` = 1 |
| Regression guards (P5) | **done** -- `tests/test_crkRegression.py`, `test_crkReproducible.py`, `test_crkLimiterDefault.py` |

All changes are uncommitted on `dev` at the time of writing; the numbers in Part 1 (O1-O4, note (b)) were taken at the old (1/3, 0.2).

---

# Part 1 — Observations

## O1. The CRKSPH Gresho vortex spins up (silently)

`cases/greshoVortex.py`, CRKSPH, t = 3 (L = 1, n_h = 4, B7, RK2, no switch):
kinetic energy **gains +7.3 % at nx = 64 and +13 % at nx = 96** with total
energy exact — internal → kinetic transfer in a steady vortex
(anti-dissipation). The trajectory dips (viscous transient, −1.6 % by
t ≈ 0.5) then rises monotonically. CompSPH on the same case decays
monotonically (−62 … −90 %). Present both before and after the warpSPHCore
CRK gradient fix (`correctGradientCRK` gradB transpose, 2026-09-26).

It went unnoticed because the final KE drift nets the dip against the gain
(nx = 48 ends at −3.9 % but regained 4.7 % on the way) and a
finest-run self-convergence metric reported a clean order 2.06 (every rung
shares the growing error). Against the exact steady v_φ profile the error
*rises* at the finest rung.

Experiments at nx = 64 (KE at t = 3 unless noted):

| experiment | KE | reading |
|---|---|---|
| baseline | +7.3 % | gain is in the **mean** v_φ profile (+6.8 %), not noise (0.35 %) / radial (0.19 %) |
| CFL / 2 | +7.8 % | not the time integrator |
| AV × 0.5 / × 2 | +5.3 % / +9.0 % | net scales with AV … |
| AV off (nx = 48) | +199 %, vortex destroyed | … but without AV the scheme is violently unstable (CompSPH without AV: +37 %) |
| limiter φ forced 0 / 1 | −96 % / +52 % | reconstruction in Q shapes the net strongly |
| Q gated on raw compression | −36 % (φ=1: −25 %) | spin-up gone, vortex over-damped |
| velocity-gradient `.mT` removed | bit-identical | only x·Gx enters μ_ij — transpose-invariant |
| AV gradient projected radial | +5.4 % | AV tangential work is not the source |
| **power budget** P_i = m_i v_i·Σ_j a_ij per RK stage, t ∈ [0, 1.5] | — | **AV net dissipative (−0.37 KE₀/time); pressure injects +0.39 (φ=1: +0.52)**, in 0.22 < r < 0.38 |
| **pressure pair forces projected central** | −72 %, no spin-up | pressure power flips to −0.045 — **source confirmed**; but the vortex collapses (the non-central CRK components are needed for an accurate pressure gradient) → diagnostic, not a fix |

**Mechanism.** The CRK pair pressure force −½ V_i V_j (P_i + P_j)
(∂W^R_ij − ∂W^R_ji) (Eq. 38, code ≡ paper) is *non-central*
(W^R_ij ≠ W^R_ji — the paper notes this). In an exact equilibrium vortex the
pressure force is radial and does no work on the azimuthal flow; here the
tangential components do net positive work in the annulus and redistribute
angular momentum radially (|ΔL| stays small, converging 4.5e-3 → 3.9e-5
over nx 32 → 96, but the core and a band at r ≈ 0.26 rotate too fast,
r ≈ 0.2 / 0.36 too slow). The viscosity removes most of the injected
energy; the residue is the spin-up and grows with resolution.

Ruled out: viscosity switch (by design), Eq. 38 prefactor, single-support vs
kernel-mean pair gradients (Gresho supports are exactly uniform at t = 0,
0.8 % std at t = 3), time integrator, the `.mT`.

Open: inherent to CRKSPH or an implementation deviation? The paper shows
CRKSPH holding the *peak* v_φ through t = 5 at 64² (their Fig. 12) — our
peak also holds (0.94 vs 0.95 at t = 3) — but it does not report KE, so the
paper neither confirms nor excludes a gain.

## O2. The limiter constants are in the wrong units

`modules/crk/limiter.py` `crkLimiter`: factor
exp(−((η_ij − η_crit)/η_fold)²) for η_ij < η_crit with
η_ij = min(r/H_i, r/H_j) — **H = support radius**. Defaults
(`configurations/crkSPH.py`): (η_crit, η_fold) = (1/3, 0.2).

The paper (Eqs. 51–53) uses η in units of its smoothing scale h, with
(η_crit, η_fold) = (1/n_h, 0.2). Its CRKSPH kernel has extent η_max = 4 in h
with n_h = 1 (≈ 4 radial neighbours) — i.e. h = Δx, the same resolution as
our n_h = 4, H = 4Δx. So the paper's constants are, in length units,
**η_crit = 1 Δx** (the threshold sits *at* the nominal spacing: "in ordinary
smooth regions this second multiplier should never activate") and
**η_fold = 0.2 Δx** — in our r/H units **(0.25, 0.05)**.

With (1/3, 0.2) every nearest-neighbour pair sits below the threshold (factor
≈ 0.83 in perfectly smooth flow) and the fall-off is 4× too wide: extra
viscosity everywhere. (1/3 = 1/n_h for n_h = 3.)

Measured on all 14 compressible cases (their own sampled particles):
Δx/H = 0.245 – 0.2505 and the median nearest-neighbour η equals it. In units
of spacing the paper's constants are kernel-independent; the kernel enters
only via H/Δx. (The alternative reading of h as the Dehnen–Aly smoothing
scale gives η_fold = 0.2/ks ≈ 0.086–0.103 depending on kernel — reported by
the derivation script for comparison.)

## O3. Sweep: current (1/3, 0.2) vs derived (Δx/H, 0.2 Δx/H)

All 14 compressible cases under CRKSPH (budget resolutions):

| case | metric | current → derived |
|---|---|---|
| Kidder | L1 ρ vs exact | 2.96e-5 → **5.33e-6 (×0.18)** |
| linearWave | L2 v vs analytic / acoustic KE loss | 6.7e-9 → **3.0e-9** / −1.36 % → **−0.03 %** |
| Yee vortex | L1 v vs exact (core) / KE loss | 2.90e-3 → 2.73e-3 / −1.9 % → −1.6 % |
| Sod (paper setup, nx = 800) | L1 ρ / entropy err / plateau u std / TV excess | 2.95e-3 → 2.90e-3 / **−14 %** / **+18 %** / **+22 %** |
| Sedov (nx = 400) | L1 ρ vs exact | 6.09e-2 → 6.63e-2 (**+9 %**) |
| Noh | L1 ρ vs exact | 0.106 → 0.110 (+3 %) |
| Gresho (nx = 64) | KE(3) / ke_rebound / L1 v vs exact | +7.3 % → +4.8 % / 0.112 → 0.091 / +10 % |
| Kelvin–Helmholtz | KE loss / rebound | −5.8 % → −5.6 % / 2.1e-3 → 7.2e-4 |
| hydrostatic | spurious max |v| | 5.7e-7 → 6.3e-7 |
| Woodward–Colella, triple point, Rayleigh–Taylor, shearing Noh, 2D Sod | no reference | stable, energy at round-off; final-state Lagrangian Δρ 0.02 – 3.7 % |

Reading: removing the extra nearest-neighbour damping helps the smooth /
weakly nonlinear cases a lot and costs the shock cases (more Sod ringing,
Sedov +9 %). Sod stays visually the paper's Fig. 4 CRKSPH column (entropy
better, L1 unchanged).

## O4. Knife-edge sensitivity at the nearest-neighbour spacing

Gresho with η_fold = 0.05: η_crit = **0.25 → KE +0.7 %**, η_crit =
**0.2458** (the measured 2D nearest-neighbour η after the support solve)
**→ +4.8 %**. A 2 % shift of the threshold changes the spin-up ~7×; with
(0.25, 0.2) it is +27 %. On a lattice IC the paper's threshold sits exactly
on the nearest-neighbour distance and the sharp fold makes the result hinge
on which side of it the lattice pairs land. This is the parameter-stability
problem this plan is about.

---

# Part 2 — Tooling (warpSPHCore harness, `higherOrderSPH/harness/pde/`)

| tool | what |
|---|---|
| `run_sod_ripple.py` | the paper's Sod (400 / 100 per tube, t = 0.15): L1 vs exact Riemann, post-shock velocity ringing, rarefaction-tail overshoot, shocked-region entropy error, density TV excess, shock width, total-energy drift + a Fig.-4-style plot |
| `derive_crk_limiter.py` | measures Δx, H, nearest-neighbour η on each case's sampled particles; derives (η_crit, η_fold) in r/H units (spacing and D&A-σ conventions) |
| `run_crk_limiter_sweep.py` | all compressible cases, current vs derived constants; per-case accuracy vs exact solutions, KE / rebound / energy, Lagrangian effect size |
| `conservation.ke_rebound` + `run_pde.py` | KE rise above its running minimum; flags unforced steady / decaying cases (Gresho, TGV); `gresho` / `gresho-std` legs |

Scratch patterns that worked (no source edits): config tweaks via
`dataclasses.replace(case, configureScheme=wrapped)` setting
`ctx.schemeConfig.crkViscosityParams.*` / `diffusionParams.*`; per-stage
power budgets by wrapping `warpSPH.schemes.builder.crkSPH_step` (the stage
state carries `ap_ij` / `av_ij` and the adjacency — `postStep` only sees the
integrator's fresh state without them).

---

# Part 3 — Plan

## P1. Parameter-stability map (main item)

Goal: find whether a robust region of (η_crit, η_fold) exists, or whether
the knife-edge (O4) is intrinsic to a threshold placed at the lattice
spacing.

- [x] 2D scan η_crit ∈ {0.20, 0.22, 0.24, 0.245, 0.25, 0.26, 0.28, 0.30, **[reduced scan done, notes (c)]**
      1/3} × η_fold ∈ {0.03, 0.05, 0.08, 0.12, 0.2} (in r/H), metrics:
      Gresho `ke_rebound` + KE(3) at nx 64 **and** 96, Sod ripple (plateau u
      std, TV excess, tail overshoot, entropy), Sedov L1, Kidder L1,
      linearWave L2. Plot each metric as a heat map; mark the paper point
      (0.25, 0.05) and the current point (1/3, 0.2).
- [x] Separate genuine sensitivity from realisation noise: repeat the **[moot: runs are deterministic; jittered/glass ICs not run -- parked]**
      sensitive points with jittered ICs (several seeds) and with a
      relaxed/glass IC instead of the lattice — if the knife-edge is a
      lattice artefact (every pair at exactly Δx), it should soften.
- [x] Resolution dependence of any candidate region (Gresho spin-up grows **[nx 64 / 96 (and 48 in the budget), notes (c), (e), (g)]**
      with nx; Sod ringing may too).

## P2. Limiter formulation variants (each gated by the Sod ripple probe)

- [~] Express the limiter in units of the **local** nominal spacing **[parked: not needed once the constants are derived from n_h; revisit with adaptive supports]**
      (per-particle Δx_i = (m_i/ρ_i)^(1/dim)) instead of a global constant
      over H — removes the dependence on H/Δx (sampler, dimension, adaptive
      supports) that already shifts the 2D value to 0.2458.
- [x] Derive (η_crit, η_fold) from n_h / the kernel at config time instead **[done, default since 2026-10-01, note (f)]**
      of hard-coding (1/3, 0.2) in `buildDefaultCRKViscosityParams`.
- [~] Softer threshold shape (continuous at η_crit with a finite slope) **[parked]**
      as an alternative to moving the constants — test whether it removes
      the knife-edge without the Sod ripple cost.

## P3. Gresho spin-up source (pressure term)

- [x] Reproduce the paper's Gresho setup as closely as possible (64² **[done, notes (d), (e): the paper text and Spheral setups differ; neither reproduces a flat KE]**
      lattice, t = 3 and 5, their kernel/n_h) and measure KE, not just the
      peak profile — inherent to CRKSPH or not?
- [x] **Cusp test: done 2026-10-01, NEGATIVE** (see the note at the end). Gresho has slope discontinuities at r = 0.2 / 0.4 (the
      paper notes they complicate the test) and the energy injection sits
      next to them; Yee (smooth) shows no rebound. Run a smoothed-cusp
      Gresho variant — if the spin-up vanishes, it is a non-smooth-field
      interaction of the non-central forces.
- [x] Time-resolved, per-radius power budget (pressure vs viscosity) across **[done, note (g)]**
      nx to see how the injection scales with resolution.
- [x] Consistency of the correction: A, B are computed with Scatter-mode **[ruled out: O1 (single vs mean support) and note (g) (Spheral uses the same pairing and weights)]**
      moments while `accel.py` evaluates pair gradients with single supports
      (h_j, h_j) / (h_i, h_i); the paper's pair kernel is the kernel mean
      (Eq. 8). Second-order for Gresho (uniform h), but may matter for
      shocks with strong h variation (Sedov / Noh).

## P4. Decision and defaults

Acceptance before changing any default (**as decided 2026-10-01: met except the Gresho rebound line -- see note (g)**):
- Sod matches the paper's Fig. 4 (qualitatively), with plateau-u std and TV
  excess within a stated bound of today's (e.g. ≤ +10 %), entropy no worse;
- Sedov / Noh L1 vs exact no worse than today by more than a stated bound;
- Gresho `ke_rebound` < 1 % at nx = 64 **and** 96, robust over the P1
  neighbourhood (not a knife-edge point);
- smooth cases (Kidder, linearWave, Yee) no worse.

## P5. Regression guards (warpSPH `tests/`)

- [x] Sod ripple bounds (CRK, symmetric support, reduced resolution for CI). **[tests/test_crkRegression.py]**
- [x] Gresho KE non-increase over a short window (after the transient) — **[tests/test_crkRegression.py: dip/rebound window, not strict non-increase -- the accepted behaviour has a 4 % rebound]**
      the check that would have caught the spin-up.
- [x] CRK energy conservation with symmetric support (exists implicitly; **[tests/test_crkRegression.py, both cases]**
      make explicit next to `test_crkSupportWarning.py`).

## Note 2026-09-30 (OPEN_PROBLEMS §15)

The viscosity switch's regulariser was fixed to the paper's `eps^2 = 1e-2` (eta^2 units). Effect on the sweep's 13 cases is
small (Sod -0.1 %, Sedov 0 %, Noh -1.7 %, Kidder -2.7 %, KH rebound -34 %); **Gresho worsened 12-15 %** (L1 v 2.87e-2 ->
3.30e-2, KE(3) +7.3 % -> +8.2 %), consistent with the knife-edge of O4. Baselines for P1 should be re-taken on this code.
The `C_l, C_q` units question (paper Table D.1) is the same class as O2.

## Note 2026-09-30 (b): `C_l` = 2 (Frontiere Table D.1, B7) on Gresho and Sod — user asked "just to see"

`scratchpad/crk_cl_sweep.py` (harness metrics, float64, current limiter constants, `C_q` = 1, no video: metric A/B),
`scripts/out_crk2d/cl/`. **`C_l` = 2 is worse on Gresho and gets worse with resolution; it is much better on Sod's ringing.**

| Gresho | (C_l, C_q) | KE(3) | ke_rebound | L1 v |
|---|---|---|---|---|
| nx 64 | (0, 0) | +274 % | 2.75 | 0.713 |
| nx 64 | (0, 1) | +7.3 % | 0.074 | 0.0499 |
| nx 64 | (0.5, 1) | +5.5 % | 0.073 | **0.0309** |
| nx 64 | (1, 1) ours | +8.2 % | 0.120 | 0.0330 |
| nx 64 | (2, 1) paper | +11.5 % | 0.193 | 0.0363 |
| nx 96 | (1, 1) ours | +12.6 % | 0.146 | 0.0372 |
| nx 96 | (2, 1) paper | **+21.9 %** | 0.262 | 0.0517 (+39 %) |

Sod nx 800, C_l 1 -> 2: plateau u std 8.2e-3 -> 2.5e-3 (-70 %), max deviation 3.2e-2 -> 9.5e-3, density TV excess
0.077 -> 0.045 (-42 %), tail overshoot -12 %; but L1 rho +9 % (2.95e-3 -> 3.21e-3), entropy error +27 %, shock width
8 -> 12 points.

Reading: some viscosity is essential (none: KE x3.7), the Gresho optimum is near `C_l` ~ 0.5, and above it the linear
term *adds* spin-up (total energy is conserved, so it is pressure work on a vortex the viscous force has reshaped),
growing with nx. Shock cases want the paper's value, smooth vortical flow does not; in the paper the limiter
(`eta_crit`, `eta_fold`, Van Leer) is what switches the viscosity off in smooth flow, and ours is in the wrong units (O2),
so `C_l` cannot be judged on its own. **Decision (user, 2026-09-30): keep `C_l` = 1; revisit `C_l`/`C_q` together with the
limiter, later, in the higher-order / AV / Riemann-MUSCL work that will stress the limiter code as well.**

## Note 2026-10-01: CRK run-to-run nondeterminism fixed (prerequisite for P1)

CRKSPH results were not reproducible (AV_PLAN S4: Gresho L1(v_phi) +-5 %, angular momentum 4x swings = noise). Bisected by hooking every
stage of `crkSPH_step` and diffing two identical runs: all pair kernels, the support solve, the CRK factors and the Verlet list are
bit-identical; the first difference enters in `finalize` -> `compSPH_deltaU_multistep`, which summed per-pair energy terms with
`scatter_add_` (CUDA atomics, order-dependent). Replaced by `math.scatter.segment_sum` (`torch.segment_reduce` over the CSR
adjacency's `numNeighbors`; no host sync, same values to round-off). Gresho nx 48, 157 steps: velocities / u / rho / h / x identical
over 3 runs, also under C&D, R&H and Rosswog2000 switches. Test: `tests/test_crkReproducible.py`. **Consequence: P1 needs no repeats for
realisation noise** (jittered ICs / glass remain a separate question about the lattice, not about the run-to-run noise), and the earlier
single-run numbers in O3/O4/notes (and AV_PLAN's CRK Gresho rows) carried an unquantified noise of order 1e-3 in angular momentum /
~5 % in L1 -- re-take baselines before reading small differences.

## Note 2026-10-01 (b): P3 cusp test -- smoothing the cusps does NOT remove the spin-up

`greshoProfile(r, cuspWidth)` (`caseUtils/compressible/greshoVortex/sample.py`, case param `cuspWidth`, default 0 = unchanged): the standard
v_phi, odd-extended through r = 0, convolved with a cubic B-spline of half-width w (cusps become C^3; w < 0.1 so the two do not overlap), P
re-derived from dP/dr = v_phi^2/r anchored at the standard outer value (steady state exactly; P(0) shifts 5 -> 5.0016 / 5.0098 for w =
0.02 / 0.05). `scripts/probe_greshoCusp.py` (video on; `results/probes/gresho_cusp*/`), CRKSPH, no switch, nx 64, t = 3, float32,
deterministic runs, L1(v_phi) against each variant's own exact profile, r < 0.5:

| w | KE(3)/KE0 - 1 | ke_rebound | L1(v_phi) | peak speed (exact) |
|---|---|---|---|---|
| 0 (standard) | +7.7 % | 0.114 | 0.0313 | 0.990 (1.000) |
| 0.02 | +8.6 % | 0.123 | 0.0348 | 1.055 (0.977) |
| 0.05 | +7.9 % | 0.115 | 0.0290 | 0.996 (0.942) |
| 0.09 | +9.3 % | 0.128 | 0.0292 | 1.003 (0.895) |

The spin-up is the same (slightly larger) with a fully smooth profile -- at w = 0.09 the profile varies on ~2 kernel supports (H ~ 0.07
at nx 64), and the peak speed ends *above* the smoothed exact peak by 12 %. **The cusps are not the cause; the "non-smooth-field
interaction" hypothesis is refuted.** The mechanism of O1 (non-central pair pressure forces net-positive work on smooth rotation) stands.
It also means there is no better-cusp Gresho that reproduces the literature's flat KE; the smoothed variant stays available as a
smooth-vortex check (and as a better-posed convergence test, since the standard profile's cusps limit the order). Remaining P3 items: the
per-radius power budget across nx, and A, B (Scatter moments) vs the pair gradient's single supports. The standard-profile baseline on
this code (deterministic, before the P1 sweep): nx 64 KE +7.7 %, rebound 0.114, L1 0.0313.

## Note 2026-10-01 (c): P1 reduced map (passes 1-2) -- the constants are a viscosity knob; no robust optimum

`scripts/crk_limiter_p1.py` (harness metrics, float64, CRKSPH, no switch, `C_l` = `C_q` = 1, deterministic so no repeats; no video: metric A/B
over ~60 runs), raw rows in `results/crk_limiter_p1/p1.jsonl`. Gresho nx 64 / 96 (t = 3), Sod nx 800, Sedov nx 400, Kidder nx 100, linearWave nx 200.
r/H units as in O2 (paper point (0.25, 0.05); current (1/3, 0.2)).

| (eta_crit, eta_fold) | Gresho 64: KE / rebound / L1 | Gresho 96: KE / rebound / L1 | Sod: plateau u std / TV excess / L1 rho / entropy / width | Sedov L1 | Kidder L1 | linWave KE loss |
|---|---|---|---|---|---|---|
| (1/3, 0.2) current | +8.1 % / 0.119 / 0.034 | +12.0 % / 0.140 / 0.040 | 8.3e-3 / 0.078 / 2.945e-3 / 1.92e-3 / 8 | 0.0610 | 2.9e-5 | -1.3 % |
| (0.25, 0.05) paper | +1.6 % / 0.061 / 0.024 | +7.7 % / 0.100 / 0.026 | 9.8e-3 / 0.094 / 2.897e-3 / 1.66e-3 / 8 | 0.0629 | 5.3e-6 | -0.03 % |
| (0.26, 0.05) | -3.5 % / 0.028 / 0.026 | +2.6 % / 0.054 / 0.029 | 8.4e-3 / 0.083 / 2.911e-3 / 1.75e-3 / 8 | 0.0631 | 9.4e-6 | -0.35 % |
| (0.28, 0.05) | -14.6 % / 0.010 / 0.030 | -7.1 % / 0.011 / 0.021 | 5.4e-3 / 0.058 / 3.079e-3 / 2.67e-3 / 10 | 0.0655 | 5.2e-5 | -2.4 % |
| (1/3, 0.05) | -46.6 % / 0 / 0.095 | -34.2 % / 0 / 0.067 | 2.6e-3 / 0.039 / 3.679e-3 / 5.31e-3 / 16 | 0.0646 | 1.6e-4 | -7.1 % |
| (0.25, 0.03) | -8.6 % / 0.020 / 0.025 | pass 3 | 8.3e-3 / 0.085 / 2.896e-3 / 1.65e-3 / 8 | pass 3 | pass 3 | pass 3 |
| (0.26, 0.03) | -13.6 % / 0.011 / 0.034 | -- | 6.1e-3 / 0.067 / 2.979e-3 / 2.09e-3 / 10 | -- | -- | -- |
| (0.25, 0.08) | +11.6 % / 0.148 / 0.050 | -- | 1.04e-2 / 0.098 / 2.906e-3 / 1.69e-3 / 8 | -- | -- | -- |
| (0.26, 0.08) | +8.2 % / 0.119 / 0.029 | -- | 9.7e-3 / 0.093 / 2.901e-3 / 1.70e-3 / 8 | -- | -- | -- |

Full Gresho nx 64 grid, KE(3) at fold 0.05 / 0.2: eta_crit 0.22: +17 / +27 %; 0.24: +7.7 / +27 %; 0.25: +1.6 / +28.5 %; 0.26: -3.5 / +24.5 %;
0.28: -14.6 / +25.5 %; 1/3: -47 / +8 %. Sod at fold 0.2 is insensitive to eta_crit below 0.28 (the limiter is barely active; plateau std ~1.09e-2).

Findings:
1. **It is a smooth, steep, monotone viscosity knob, not a cliff.** KE(3) falls ~5 % per 0.01 of eta_crit (fold 0.05) and rises ~10 % per
   step in fold at fixed eta_crit (0.03 / 0.05 / 0.08 / 0.2 at 0.25: -8.6 / +1.6 / +11.6 / +28.5 %). The O4 "knife edge" was this slope plus
   noise; with deterministic runs it is a sensitivity (about one nearest-neighbour spacing x 0.01 per few %), not a discontinuity.
2. **The spin-up is compensated, not removed.** Zero rebound is reached only where Gresho is over-damped (eta_crit >= 0.28: KE -15 %) and that costs Sod shock
   width / entropy (+60 % at 0.28), Kidder (x10) and linearWave (-2.4 %). The P4 target "Gresho rebound < 1 % at nx 64 and 96" is met by no point that also keeps Sod
   within +10 % and the smooth cases no worse.
3. **No resolution-robust threshold.** At fixed constants the spin-up grows with nx (0.25/0.05: +1.6 -> +7.7 %; 0.26/0.05: -3.5 -> +2.6 %; current: +8.1 -> +12.0 %), except at
   0.28 where L1 improves (0.030 -> 0.021) at the price above. A constant tuned at one resolution will not hold at the next.
4. **The current default is the worst compromise on the smooth cases** (Kidder 5x-10x the paper point's error, linearWave -1.3 % vs -0.03 %) and worse on Gresho
   (+8 -> +12 %); its only edge is Sod ringing (-15 % plateau std vs the paper point) and Sedov (-3 %).
5. **Sod and the smooth cases are the constraint, Gresho the tie-break**: the paper point (0.25, 0.05) and the narrower-fold (0.25, 0.03) both keep Sod's L1 and entropy
   better than today with ringing within +17 % / +0 % of today's, and keep Kidder / linearWave 5-40x better; (0.25, 0.03) also gives Gresho L1 0.025 at nx 64 (-27 % vs today).

### P1 pass 3 -- the narrow-fold point (0.25, 0.03) on the full suite, and its neighbours

| (eta_crit, eta_fold) | Gresho 64: KE / rebound / L1 | Gresho 96: KE / rebound / L1 | Sod: plateau std / TV / L1 rho / entropy / width | Sedov L1 | Kidder L1 | linWave KE loss |
|---|---|---|---|---|---|---|
| (1/3, 0.2) current | +8.1 % / 0.119 / 0.034 | +12.0 % / 0.140 / 0.040 | 8.3e-3 / 0.078 / 2.945e-3 / 1.92e-3 / 8 | 0.0610 | 2.9e-5 | -1.3 % |
| **(0.25, 0.03)** | -8.6 % / 0.020 / 0.025 | -0.5 % / 0.027 / 0.022 | 8.3e-3 / 0.085 / 2.896e-3 / 1.65e-3 / 8 | 0.0669 (+9.7 %) | 5e-6 | -0.03 % |
| (0.24, 0.03) | -2.0 % / 0.031 / 0.022 | -- | 1.03e-2 / 0.097 / 2.898e-3 / 1.64e-3 / 8 | -- | -- | -- |
| (0.255, 0.03) | -10.4 % / 0.014 / 0.028 | -- | 7.2e-3 / 0.075 / 2.928e-3 / 1.79e-3 / 8 | -- | -- | -- |
| (0.25, 0.04) | -2.7 % / 0.037 / 0.026 | -- | 9.2e-3 / 0.091 / 2.892e-3 / 1.65e-3 / 8 | -- | -- | -- |

Reading against P4's acceptance list: (0.25, 0.03) meets Sod (ringing +0 %, TV +9 %, L1 and entropy better), Kidder / linearWave (6x / 40x better), Sedov
(+9.7 %, on the +10 % edge) and Gresho L1 (-27 % / -44 % vs today at nx 64 / 96, improving with nx, no net spin-up at either resolution), but **not** the
"rebound < 1 %" line (2-3 %: the KE dips then recovers by that much). The neighbourhood is forgiving only on one side: lowering eta_crit to 0.24 costs Sod ringing
(+24 %), raising it to 0.255 moves Gresho to -10 % (over-damped, Sod ringing better) -- so it is a ridge of about +-0.005 in eta_crit at this fold, not a plateau.
Untested: nx > 96 (the spin-up of the neighbouring points grew with nx; this one went -8.6 -> -0.5 %, i.e. the same direction), Noh, the 2D shock cases,
glass / jittered ICs. **No default has been changed** -- decision for the user: keep (1/3, 0.2), move to the paper point (0.25, 0.05), or to (0.25, 0.03).

## Note 2026-10-01 (d): what Spheral (the authors' code, `~/dev/spheral`) actually does for Gresho and the limiter

Sources: `tests/functional/Hydro/GreshoVortex/GreshoVortex.py`, `src/ArtificialViscosity/LimitedMonaghanGingoldViscosityViewInline.hh`,
`src/CRKSPH/CRKSPHVariant.cc` (`evaluateDerivatives`).

**Gresho regression test (CRKSPH legs).** `--CRKSPH True --cfl 0.25 --Cl 1.0 --Cq 1.0 --filter {0,0.01,0.1,0.2} --goalTime 3.0`: 64x64 lattice on
[-0.5, 0.5]^2, periodic, rho = 1, gamma = 5/3, standard profile (no cusp smoothing), `NBSplineKernel` order 5 (extent (order+1)/2 = 3), `nPerh = 1.51`
(support 4.53 dx -- NOT the paper's B7 / n_h = 1 / 4 dx), `IdealH`, `RigorousSumDensity`, `RKSumVolume`, `LinearOrder`, compatible energy, `CheapSynchronousRK2`,
`epsilon2 = 1e-2`, `Qlimiter = False` and Balsara off (those flags are not the Van Leer limiter, which is always on in `LimitedMonaghanGingoldViscosity`),
no viscosity switch. The test reports an L1 azimuthal-velocity error against the analytic profile; **it never measures KE**.

**Limiter constants.** `etaCrit = etaCritFrac / nPerh`, `etaFold = etaFoldFrac / nPerh`, defaults `(1.0, 0.2)`, with `eta = H x` (units of the smoothing scale h, `support = extent * h`),
`eta_ij = min(|eta_i|, |eta_j|)`, factor `exp(-((eta_ij - etaCrit)/etaFold)^2)` for `eta_ij < etaCrit`. So **both** thresholds scale with 1/nPerh: `etaCrit` is exactly the
nearest-neighbour distance on the initial lattice (`eta_nn = dx/h = 1/nPerh`) and `etaFold = 0.2 dx` -- confirms O2 (our (0.25, 0.05) in r/H for n_h = 4); in r/H units for any kernel it is
`(dx/H, 0.2 dx/H) = (1/n_h, 0.2/n_h)`. The factor equals 1 at `etaCrit` (continuous), so the lattice's nearest-neighbour pairs sit at the foot of the fold by construction.
The Van Leer `phi`, `vi1 = vi - phi DvDx_i xij`, `vj1 = vj + phi DvDx_j xij` (xij = (xi - xj)/2) and `mu = v.eta / (eta^2 + epsilon2)` all match our `accel.py`/`dudt.py` (S4 sign, eps^2 = 1e-2).

**`C_l`, `C_q`.** The test itself uses `Cl = 1, Cq = 1` (the constructor default for Cq is 0.75) -- not the paper's `Cl = 2`. This supports keeping our `C_l = 1` (note (b)).

**CRK pair force.** `force_ij = 0.5 w_ij^2 [(P_i + P_j)(gradW_j - gradW_i) + Qacc_ij]`, `w_ij = (V_i + V_j)/2` (arithmetic-mean volume **squared**), `gradW_j` = j's kernel (H_j) with i's A, B and
`gradW_i` = i's kernel (H_i) with j's A, B -- our `gradw_i` / `gradw_j` use the same pairing (h_j / h_i). Ours is `V_i V_j (P_i + P_j) 0.5 (gradw_i + gradw_j)` (paper Eq. 38): the only difference is
`V_i V_j` vs `((V_i + V_j)/2)^2` (second order in the volume contrast; both symmetric). Velocity gradient: Spheral `DvDx -= w_ij v_ij (x) gradW_j` with `w_ij` the mean volume, ours the Scatter/Difference operation with `V_j`.
Not tested whether either difference matters for the spin-up.

## Note 2026-10-01 (e): P3 item 1 -- the paper's / Spheral's Gresho setup in our code (`scripts/probe_greshoPaper.py`, video on, `results/probes/gresho_{paper,spheral,kernel,nh,nh96}/`)

nx 64, t = 5 unless noted (KE end / KE min / rebound, L1 of v_phi vs exact at t = 3 / 5, peak = bin-mean v_phi at t = 3 / 5):

| config | KE(5) | KE min | rebound | L1 t3 / t5 | peak t3 / t5 |
|---|---|---|---|---|---|
| `paper` text: B7, n_h 4, `C_l` 2, (0.25, 0.05) | -2.2 % | -8.6 % | 0.065 | 0.026 / 0.054 | 0.865 / 0.847 |
| `current`: B7, n_h 4, `C_l` 1, (1/3, 0.2) | **+15.6 %** | -3.8 % | 0.193 | 0.032 / 0.110 | 0.943 / 0.866 (fastest particle 1.00 / 1.14) |
| `candidate`: B7, n_h 4, (0.25, 0.03) | -11.7 % | -11.7 % | 0.020 | 0.027 / 0.065 | 0.850 / 0.796 |
| `spheral`: quintic, n_h 4.53, limiter (1/n_h, 0.2/n_h) | +2.4 % | -5.3 % | 0.079 | 0.022 / 0.034 | 0.888 / 0.891 |
| quintic, n_h 4.53, (1/3, 0.2) | -1.4 % | -6.0 % | 0.048 | 0.024 / 0.030 | 0.880 / 0.874 |
| B7, n_h 4.53, (1/3, 0.2) | -2.5 % | -5.6 % | 0.037 | 0.023 / 0.036 | 0.874 / 0.864 |
| quintic, n_h 4.0, (1/3, 0.2) | +14.6 % | -4.1 % | 0.188 | 0.032 / 0.053 | 0.940 / 0.958 |

n_h sweep, B7, `C_l = C_q = 1`, limiter scaled like Spheral `(1/n_h, 0.2/n_h)`, t = 3 (KE gain / rebound / L1):

| n_h | nx 64 | nx 96 |
|---|---|---|
| 3.5 | -0.7 % / 0.037 / 0.029 | -- |
| 4 | +1.7 % / 0.061 / 0.027 | +8.0 % / 0.103 / 0.034 |
| 4.5 | +1.9 % / 0.068 / 0.025 | +10.3 % / 0.127 / 0.034 |
| 5 | -2.5 % / 0.038 / 0.024 | +9.7 % / 0.123 / 0.034 |
| 6 | -13.2 % / 0.004 / 0.039 (over-damped) | -- |

Findings:
1. **The paper's text setup does not reproduce its "near theoretical peak"**: the paper-constant run (B7, n_h 4, `C_l` 2) holds a bin-mean peak of 0.85-0.87 (fastest particle 0.91-0.95). Our *current* defaults get a particle peak of 1.00 (t = 3) -- but with +15.6 % KE and L1 0.11 at t = 5: a peak-only
   metric rewards the spin-up. Fig. 12's "peak maintained" is not evidence against a KE gain; the paper never plots KE.
2. **Spheral's own test is a different setup from the paper's text** (note (d)): B5 / nPerh 1.51 (support 4.53 dx), `Cl = Cq = 1`. Run that way (`spheral`, and with our (1/3, 0.2)) there is no spin-up at nx 64 and the peak holds to t = 5 (0.888 -> 0.891), L1 0.034 / 0.030
   at t = 5 -- the closest to the literature picture of all runs. **The difference is mostly the limiter scaling, not the kernel**: quintic n_h 4.0 with (1/3, 0.2) spins up as much as B7 n_h 4.0 (+14.6 / +15.6 %), while (1/3, 0.2) at n_h 4.53 gives -1.4 / -2.5 % on either kernel
   (the nearest-neighbour factor is exp(-((1/n_h - 1/3)/0.2)^2) = 0.83 at n_h 4, 0.73 at 4.53: the unscaled constants limit more as n_h grows).
3. **Expressing the constants in dx (Spheral's `etaCritFrac / nPerh`, `etaFoldFrac / nPerh`) makes the limiter n_h- and kernel-independent**, and then n_h 3.5-5 all give +-2 % KE / L1 0.024-0.029 at nx 64 -- confirming O2 and P2's "derive from n_h" item.
4. **But the spin-up still grows with resolution** under that scaling: at nx 96, +8 to +10 % for n_h 4-5 (L1 0.034, flat; rebound 0.10-0.13). Spheral's test never measures KE or runs a resolution series (single 64x64, t = 3, L1 vs the profile), so its Gresho "looks right" at one resolution.
   So the pressure-force spin-up is intrinsic to our CRKSPH at fixed n_h and the limiter / n_h only offsets it at a given nx. Cusp smoothing (note (b)/(c) of P3) does not change this.
5. Cl = 1 is what the CRKSPH authors' own Gresho regression uses (supports the 2026-09-30 decision to keep it).

Next candidates (P3 remainder): (a) Spheral's pair force uses `w_ij^2` (mean volume squared) and a mean-volume velocity gradient where ours uses `V_i V_j` and `V_j` -- try it; (b) per-radius pressure-power budget vs nx; (c) whether `A`/`B` (Scatter moments) vs the pair gradients' single supports matters.

## Note 2026-10-01 (f): default changed to (1/n_h, 0.2/n_h); Spheral's `w_ij^2` pair weights tested -- no effect

**Default (user: "that sounds good for the defaults", 2026-10-01).** `CRKViscosity.eta_crit` / `eta_fold` now default to -1 = "derive from n_h": `resolveCRKLimiter(params, n_h)` (`configurations/crkSPH.py`, called once per
step in `schemes/crkSPH.py`) gives `eta_crit = 1/n_h`, `eta_fold = 0.2/n_h` in r/H (n_h = 4: (0.25, 0.05); n_h = 4.5: (0.222, 0.044)), the paper's / Spheral's definition in units of the particle spacing.
An explicit positive value still overrides (each independently); stored configs keep their explicit values. Verified bit-identical to the explicit values at n_h = 4 and 4.5; `tests/test_crkLimiterDefault.py`. **Consequence: every
CRKSPH number taken before this change at the old (1/3, 0.2) (AV_PLAN's M0 CRK rows, `results/av_M0d_*`, the CRK parts of the S4 notes) is on the old constants and will move; re-take them when AV work needs CRK numbers.**
Expected effect at n_h = 4 (P1 map, this default = the "paper point" (0.25, 0.05) with `C_l = 1`): Gresho KE at nx 64 / 96 +1.6 / +7.7 % (was +8 / +12 %), L1(v) 0.024 / 0.026 (was 0.034 / 0.040), Kidder 5e-6 (was 3e-5), linearWave -0.03 % (was -1.3 %),
Sod ringing +17 % plateau std / +21 % TV (entropy -13 %, L1 -2 %), Sedov +3 %. (0.25, 0.03) remains the tuned alternative (note (c) pass 3) and is not the default.

**Pair weights.** New opt-in `CRKViscosity.meanVolumeWeights` (default False = `V_i V_j`, Frontiere Eq. 38): True uses Spheral's `((V_i + V_j)/2)^2` in the pressure and viscosity pair terms of `accel.py` and `dudt.py`
(symmetric, so conservation is unaffected; `scripts/probe_greshoPaper.py --meanW`). Gresho, n_h 4 with the new default, t = 3 (KE / rebound / L1): nx 64 `V_i V_j` +1.6 % / 0.059 / 0.0259, mean-volume +2.2 % / 0.065 / 0.0259; nx 96 `V_i V_j` +8.0 % / 0.103 / 0.0342,
mean-volume +8.3 % / 0.106 / 0.0347. **No effect** (within the ~0.1-0.5 % KE spread that round-off-level reordering produces), so the weight difference is not the spin-up source; the flag is left in as an option. Spheral's velocity gradient (`w_ij` weights vs our `V_j`) was not changed.
Remaining P3: per-radius pressure-power budget vs nx; A, B (Scatter moments) vs the pair gradients' single supports.

## Note 2026-10-01 (g): P3 per-radius power budget, and the close-out

`scripts/probe_greshoPower.py` (video on; `results/probes/gresho_power/`): wraps `crkSPH_step`, accumulates per radial bin the KE each pair force injects,
`dKE_i = m_i v_i . sum_j a_ij * dt/2` per RK2 stage, for the pressure (`ap_ij`) and viscous (`av_ij`) pair accelerations. Default limiter, n_h 4, `C_l` = 1, t = 3. Closure of (pressure + viscous) against the measured
KE change: 3e-5 / 2e-5 / 2e-5 of KE0.

| nx | pressure work / KE0 | viscous work / KE0 | net = KE gain |
|---|---|---|---|
| 48 | +1.067 | -1.159 | -9.2 % |
| 64 | +0.998 | -0.983 | +1.6 % |
| 96 | +0.826 | -0.747 | +7.8 % |

Findings: (1) the pressure pair force pumps ~1 KE0 into the vortex over t = 3, **at a roughly constant rate** (cumulative +0.2 at t/3 = 0.2 ... +1.0 at 1; not a start-up transient), all in the shear annulus 0.2 < r < 0.4 (peak bins 0.3-0.35);
(2) the viscous force removes almost exactly the same bin by bin (e.g. nx 64, bin 0.30-0.35: +0.317 / -0.316), so the vortex sits in a pump-and-dissipate balance and the visible KE change is the small residual; (3) the pump converges
slowly, ~nx^-0.4 (rate 0.85 / 0.80 / 0.65 for nx 48 / 64 / 96), and the viscous removal falls faster (the limiter switches the viscosity off in smoother flow), so the residual grows from -9 % to +8 %. That is the whole picture of the spin-up
at the default constants: it is a slowly converging tangential pressure work from non-central pair forces on a lattice that the viscosity cancels to within a few percent, so no limiter constant removes it across resolution (note (c): any
choice trades damping for spin-up). Combined with the negative cusp test (note (b)), the pair-weight A/B (note (f)) and Spheral's identical pair structure (note (d)), I found no implementation deviation: **the Gresho spin-up is intrinsic to CRKSPH as formulated**, and
its resolution dependence is the part the paper's single-resolution, peak-only test does not show.

**What is left open (parked, not blocking):** a formulation that removes the tangential pressure work (e.g. a central/symmetrised pair force that keeps the linear reproduction -- pure research; the projected-central
variant of O1 collapsed the vortex), the local-spacing / softer-threshold limiter variants, glass / jittered ICs, Noh and the 2D shock cases at the new default, and re-taking the AV_PLAN CRK baseline at the new constants.

## Note 2026-10-01 (h): follow-ups run -- Noh + 2D shock cases at the new default, CRK baseline re-taken

**Old (1/3, 0.2) vs new (1/n_h, 0.2/n_h) = (0.25, 0.05)** (`scripts/crk_limiter_p1.py`, `results/crk_closeout/`, metric A/B no video; the new-default runs of the two cases av_report does not cover have video in `results/crk_closeout_videos/`):
all stable, energy at round-off. Noh L1 rho 0.1042 -> 0.1073 (+3 %); Yee core L1 v 2.88e-3 -> 2.72e-3 (-6 %); Kelvin-Helmholtz KE loss -5.7 -> -5.5 %; shearing Noh KE loss -18.2 -> -19.4 %; Rayleigh-Taylor
(KE x21, dE/E 1.9e-4), triple point, Sod 2D, Woodward-Colella, hydrostatic (max |v| 6e-7) unchanged. Final frames of triple point and shearing Noh: symmetric roll-ups / stripes, no fliers (triple point keeps its known slight ripple along the lower contact, OPEN_PROBLEMS §15).

**AV_PLAN CRK baseline re-taken** as `results/av_M0f_crk` (18 pairs; AV_PLAN updated). 3D Sedov and Noh ran without video: with video the CRK 3D Sedov leaks GPU memory per drawn frame and OOMs on the shared GPU -- logged as OPEN_PROBLEMS §18.
