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

| Item | State |
|---|---|
| Gresho spin-up: detection | **done** — warpSPHCore `ke_rebound` check flags it at nx = 48 / 64 / 96 |
| Gresho spin-up: energy source located | **done** — CRK pair *pressure* forces (non-central), §O1 |
| Gresho spin-up: fix | **open** (§P3) |
| Limiter constants in the wrong units | **found**, not changed (§O2) |
| Derivation of paper-consistent constants | **done** (script), §O2 |
| 14-case sweep current vs derived | **done**, §O3 — trade-off, not a clear win |
| Parameter-stability map | **open** — this plan's main item (§P1) |

Nothing in the frontend has been changed for these items except the support
warning. All experiment edits were reverted; the numbers below come from
config hooks and scratch step wrappers.

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

- [ ] 2D scan η_crit ∈ {0.20, 0.22, 0.24, 0.245, 0.25, 0.26, 0.28, 0.30,
      1/3} × η_fold ∈ {0.03, 0.05, 0.08, 0.12, 0.2} (in r/H), metrics:
      Gresho `ke_rebound` + KE(3) at nx 64 **and** 96, Sod ripple (plateau u
      std, TV excess, tail overshoot, entropy), Sedov L1, Kidder L1,
      linearWave L2. Plot each metric as a heat map; mark the paper point
      (0.25, 0.05) and the current point (1/3, 0.2).
- [ ] Separate genuine sensitivity from realisation noise: repeat the
      sensitive points with jittered ICs (several seeds) and with a
      relaxed/glass IC instead of the lattice — if the knife-edge is a
      lattice artefact (every pair at exactly Δx), it should soften.
- [ ] Resolution dependence of any candidate region (Gresho spin-up grows
      with nx; Sod ringing may too).

## P2. Limiter formulation variants (each gated by the Sod ripple probe)

- [ ] Express the limiter in units of the **local** nominal spacing
      (per-particle Δx_i = (m_i/ρ_i)^(1/dim)) instead of a global constant
      over H — removes the dependence on H/Δx (sampler, dimension, adaptive
      supports) that already shifts the 2D value to 0.2458.
- [ ] Derive (η_crit, η_fold) from n_h / the kernel at config time instead
      of hard-coding (1/3, 0.2) in `buildDefaultCRKViscosityParams`.
- [ ] Softer threshold shape (continuous at η_crit with a finite slope)
      as an alternative to moving the constants — test whether it removes
      the knife-edge without the Sod ripple cost.

## P3. Gresho spin-up source (pressure term)

- [ ] Reproduce the paper's Gresho setup as closely as possible (64²
      lattice, t = 3 and 5, their kernel/n_h) and measure KE, not just the
      peak profile — inherent to CRKSPH or not?
- [ ] Cusp test: Gresho has slope discontinuities at r = 0.2 / 0.4 (the
      paper notes they complicate the test) and the energy injection sits
      next to them; Yee (smooth) shows no rebound. Run a smoothed-cusp
      Gresho variant — if the spin-up vanishes, it is a non-smooth-field
      interaction of the non-central forces.
- [ ] Time-resolved, per-radius power budget (pressure vs viscosity) across
      nx to see how the injection scales with resolution.
- [ ] Consistency of the correction: A, B are computed with Scatter-mode
      moments while `accel.py` evaluates pair gradients with single supports
      (h_j, h_j) / (h_i, h_i); the paper's pair kernel is the kernel mean
      (Eq. 8). Second-order for Gresho (uniform h), but may matter for
      shocks with strong h variation (Sedov / Noh).

## P4. Decision and defaults

Acceptance before changing any default:
- Sod matches the paper's Fig. 4 (qualitatively), with plateau-u std and TV
  excess within a stated bound of today's (e.g. ≤ +10 %), entropy no worse;
- Sedov / Noh L1 vs exact no worse than today by more than a stated bound;
- Gresho `ke_rebound` < 1 % at nx = 64 **and** 96, robust over the P1
  neighbourhood (not a knife-edge point);
- smooth cases (Kidder, linearWave, Yee) no worse.

## P5. Regression guards (warpSPH `tests/`)

- [ ] Sod ripple bounds (CRK, symmetric support, reduced resolution for CI).
- [ ] Gresho KE non-increase over a short window (after the transient) —
      the check that would have caught the spin-up.
- [ ] CRK energy conservation with symmetric support (exists implicitly;
      make explicit next to `test_crkSupportWarning.py`).

## Note 2026-09-30 (OPEN_PROBLEMS §15)

The viscosity switch's regulariser was fixed to the paper's `eps^2 = 1e-2` (eta^2 units). Effect on the sweep's 13 cases is
small (Sod -0.1 %, Sedov 0 %, Noh -1.7 %, Kidder -2.7 %, KH rebound -34 %); **Gresho worsened 12-15 %** (L1 v 2.87e-2 ->
3.30e-2, KE(3) +7.3 % -> +8.2 %), consistent with the knife-edge of O4. Baselines for P1 should be re-taken on this code.
The `C_l, C_q` units question (paper Table D.1) is the same class as O2.
