# warpSPH — Godunov SPH plan (opened 2026-10-08, user)

> **Progress marker:** this plan's row in [PLANS.md](PLANS.md). Update its *Last worked* date and *Where it stands* whenever you work on this plan.

## Why

AV_PLAN Phase 7b built a Riemann solver module and a first-order Riemann *dissipation* term (the velocity-jump part of the pair's
star pressure, `ViscosityTerms.RiemannDissipation`). That keeps the symmetric SPH pressure gradient and only replaces the
viscosity. **Godunov SPH** replaces the whole pressure term: the force on a particle comes from the star pressure `p*` of the
Riemann problem between it and each neighbour, no artificial viscosity (Inutsuka 2002; Cha & Whitworth 2003). This plan is the
step from one to the other, in three layers, each a usable result on its own:

1. **Reconstruction (the old "stage 2")**: per-particle gradients of `rho`, `P` (and `v`, which exists), a limiter, and left / right
   states at the pair midpoint. Plugs into the Phase 7b term and is validated there first.
2. **First-order GPH** (Cha & Whitworth Cases 1-3): `p*` of the (reconstructed) pair replaces `P_i`, `P_j` in the SPH force.
3. **Inutsuka (2002) proper**: the kernel-convolution force with the effective `V_ij^2` and `s*`, the time-centred second-order Riemann
   states, the shock-surface first-order switch.

## Papers (all on disk since 2026-10-08; `literature/`)

`inutsuka2002` (the scheme), `cha2003` (the simplified variants), `murante2011` (the reference implementation and the sensitivity study),
`iwasaki2011` (Inutsuka's group's second-order recipe, MHD), `puri2014` (approximate solvers, AV equivalence, wall heating), `cha2010`
(why standard SPH fails at a density gradient), plus `toro2009`, `vanleer1979`, `hopkins2013/2015`, `parshikov2002`, `vila1999`.
All four of the 2026-10-08 arrivals are verbatim-checked in `ABSTRACTS.md`. Not on disk and cited as inputs: van Leer 1997 (the iterative
solver of `cha2003`), Sirotkin & Yoh 2013 (SPH with LLF / HLL fluxes, a different route), Dukowicz 1985, Rider 2000 (wall heating).

### What the papers settle

**Two schemes are called "GSPH", and they are not equivalent.**

| | Inutsuka 2002 (the real one) | Cha & Whitworth 2003 / Iwasaki Eq. (24) / Puri Eq. (15) ("simplified") |
|---|---|---|
| momentum | `a_i = -2 sum_j m_j p*_ij [V_ij^2(h_i) dW(x_ij, sqrt2 h_i) + V_ij^2(h_j) dW(x_ij, sqrt2 h_j)]` | `a_i = -sum_j m_j p*_ij [dW(h_i)/rho_i^2 + dW(h_j)/rho_j^2]` |
| energy | `du_i/dt = -2 sum_j m_j p*_ij (v*_ij - xdot*_i) . [same bracket]` | `du_i/dt = -sum_j m_j p*_ij (v*_ij - xdot*_i) . [same bracket as its momentum]` |
| kernel | Gaussian only (the integrals `int W(x-x_i) W(x-x_j) / rho^2 dx` close), neighbours within `sqrt2 * 3h` | any kernel |
| origin | convolve the equations with the kernel, integrate by parts (Eq. 23, 43) | replace `P_i`, `P_j` by `p*` in SPH |
| density-gradient consistency | exact: `a_i = 0` for constant `P` whatever `rho` (Inutsuka Eq. 23) | **lost**: for constant `P` it is *identically the standard SPH force*, so it keeps the spurious repulsion of Cha 2010 Fig. 1 |
| KH / blob (Murante 2011) | follows both | "exceedingly diffusive", wrong instabilities |

So the simplified form is a useful cheap baseline (it is Phase 7b's term with the whole pressure sum replaced, and Puri / Iwasaki show it matches
finite-volume codes on 1D / 2D shocks) but **it cannot be the end point**: the consistency Cha 2010 diagnoses needs the convolution form.
The plan's layer 2 is therefore the simplified form, as a baseline and a stepping stone, and layer 3 the real scheme.

**Details of the real scheme** (Inutsuka 2002 §3, Murante 2011 §3 and Appendix A):
- `s` axis along `x_i - x_j`, origin at the midpoint, `s_ij = |x_i - x_j|`; specific volume `V = 1/rho` interpolated along `s`:
  linear (`V = C s + D`, `C = (V_i - V_j)/s_ij`, `D = (V_i + V_j)/2`, `V_ij^2 = h^2 C^2 / 4 + D^2`, Inutsuka Eq. 52) or cubic spline (Eq. 60-65, which also uses
  the projected gradients `e . grad V` at both particles; **the cubic is the reference** (Murante), falling back to linear when the end gradients disagree in sign).
- Interface position `s*_ij = h^2 C D / (2 V_ij^2)` (linear, Eq. 57) / Eq. (65) (cubic); the Riemann problem is solved there with `p*`, `u*`.
- Variable `h`: `h_i` for the half of the integral containing `x_i`, `h_j` for the other (Eq. 79); the sums run over neighbours within `sqrt2 h_i` or
  `sqrt2 h_j`. Murante fixes `N_neigh` (Gaussian truncated at `3h`: 100 in 1D / 3D tests up to 442 for KH; fewer neighbours degrade KH).
- Density: **symmetrised** `rho_i = sum_{|x_j - x_i| < max(h_i,h_j)} m_j [W(x_ji, sqrt2 h_i) + W(x_ij, sqrt2 h_j)]/2` (Murante Eq. 15), an even function
  of the pair, removing the SPH density asymmetry (Cha 2010).
- States of the Riemann problem: **second-order**: `Q_R = Q_i - (1/2) DeltaQ_i`, `Q_L = Q_j + (1/2) DeltaQ_j` with the limited slope along `s`
  (`DeltaQ_i = limited(Q_i - Q_j, gradQ_i . n s_ij)`); Inutsuka evolves them by `C Delta t / 2` (the domain of dependence, Eq. 68; Iwasaki the same).
  Limiters: Inutsuka 2002 zero both slopes when the velocity gradients disagree in sign (Eq. 74) and drop to first order in a shock
  (`C_shock (v_j - v_i).n > min(c_i, c_j)`, `C_shock = 3`, Eq. 75); **Murante's reference** is van Leer's harmonic mean of the projected gradient and the
  finite difference, `2 Q1 Q2/(Q1 + Q2)` if `Q1 Q2 > 0` else 0 (Eq. 18-23), which beats Inutsuka's limiter on KH; Iwasaki's is van Leer 1979's monotonised
  slope, `min(2|Q_i - Q_j|, |Dbar|, 2|DeltaQ_i|) sgn`, `Dbar = ((Q_i - Q_j) + DeltaQ_i)/2`, zero if the signs disagree (App. C). The gradient is the
  plain difference gradient `n . sum_k (m_k/rho_k)(Q_k - Q_i) grad W` (what `computeStateGradients` computes). **First-order states are
  dramatically worse on KH and blob** (Murante), so the layer-1 reconstruction matters here.
- Energy: `v*` is the Riemann velocity along `n` plus the mean of the tangential parts, which drops out against the kernel gradient (Inutsuka Eq. 70-73);
  `xdot*_i = v_i + a_i Delta t / 2` (time-centred, needed for exact discrete energy conservation). Puri Eq. 36: the `Delta t / 2` term is a
  higher-order dissipation `-(Delta t/2) a_i^2`.
- Solvers: exact iterative (van Leer 1997) in the papers; **Puri 2014 finds Roe, Dukowicz and HLLC suitable replacements, LLF and HLLE too diffusive (they fail Noh
  with negative densities)**. Our `modules/riemann` has Acoustic / PVRS / TRRS / TSRS / Adaptive / HLLC. Puri's Lagrangian HLLC uses `S_l = min(v_l - c_l, -c_lr)`,
  `S_r = max(v_r + c_r, c_lr)` with Roe-averaged `c_lr`; ours uses Toro's PVRS-based wave speeds, which put Sod's contact speed at 0.61 against the exact 0.93
  (`docs/av/riemann_7b_2026-10-08/`): worth trying Puri's speeds.

**GSPH dissipation is signal-velocity artificial viscosity** (Puri §4): under a "centred + diffusive" solver, `p*_ab = pbar - (1/2) rhobar cbar (u_a - u_b)`,
the momentum dissipation is Monaghan's `alpha v_sig (v_ab . rhat) grad W` with `alpha_GSPH = rhobar^2 Vbar^2` and `v_sig = cbar`, **with no approach-only
switch** (the GSPH viscosity acts for receding pairs too) and no `beta` term (compensated by the larger `alpha` at density jumps). This is exactly what
AV_PLAN Phase 7b measured (Acoustic with equal states = Monaghan 1997a with alpha = 1); `monaghanSwitch = False` is the GSPH behaviour of that term.
Consequences Puri draws and the plan inherits: GSPH has a non-zero viscosity (which kills SPH's pressure blip) and therefore larger dissipation-induced
heating (Sjogreen); **wall heating at the origin of Noh persists for every solver** (the entropy error is made in the shock reflection and then advected;
the remedy in the literature is artificial thermal conduction; we saw the same dip, `docs/av/riemann_7b_2026-10-08/frames/`); a **pairing / clumping
instability for unequal-mass particles** appears in the 2D Riemann problem.

**Tests the papers define** (and the codebase has most of the cases): Sod (Murante, Cha 2003); the **pressure-equilibrium density-jump force test** of Cha 2010
Fig. 1 (acceleration must vanish: the single most discriminating test between the two schemes, new); KH with density contrast 2:1 and the Wengen KH (Murante,
Cha 2010; `kelvinHelmholtz` exists); the blob test (new); Noh 2D (Puri); Woodward-Colella blast wave and Sjogreen (Puri); the acoustic-wave diffusion and 1D/2D
accuracy tests (Puri §5.1-5.3).

## Design notes

- Pair geometry as in Inutsuka §3.1: `n = x_ij / r` (j to i), 1D Riemann problem on `n`, left state j, right state i, tangential velocity passive.
  The 7b convention (left j, right i) is the paper's (`u_r = v_a . rhat_ab`, `u_l = v_b . rhat_ab`, Puri Eq. 12).
- Layer 2 needs the whole pair loop `(dvdt, dudt)` in one kernel (both use `p*_ij`, `u*_ij`): a new `modules/godunov/` next to `modules/riemann`, a scheme
  entry beside `schemes/monaghan.py` reusing its state, EOS, time-step and switch plumbing, no viscosity switch and no AV (`viscosityTerm` unused).
- Integrator: the papers use a time-centred `xdot*_i = v_i + a_i Delta t/2` and `C Delta t/2` state evolution, both with an explicit `Delta t`. The repo's
  integrators call `f(state) -> update`; the first cut uses the instantaneous `v_i` and un-evolved states (RK2 supplies the temporal order), and measures
  total-energy drift against the papers' conservation claim before deciding whether a `Delta t`-aware variant is needed.
- Momentum is conserved because `p*_ij` is symmetric under `i <-> j` (our solvers are mirror-symmetric, `tests/test_riemann.py`) and `grad_i W_ij = -grad_j W_ji`.
- Layer 3 uses `KernelFunctions.Gaussian` (it exists, `h = 2 sigma`, 16-sigma truncation: check its support / `sqrt2 h` bookkeeping against the papers' `W = (pi h^2)^{-d/2} exp(-x^2/h^2)`).
- Not variational and not separable (`p*` depends on `v`): outside the geometric-integration work (PESPH_PLAN §7.4).

## Steps

- [x] **L1** reconstruction of `rho`, `P` to the pair midpoint (`computeStateGradients`, `pairState.py`); A/B in the AV term: null result (`docs/av/godunov_l1_2026-10-08/`)
- [x] **L1b** (2026-10-08, uncommitted) `StateLimiter`: `PairRatio` (old), `VanLeerHarmonic` (Murante Eq. 23), `VanLeerMonotonized` (Iwasaki App. C), `InutsukaSign` (Eq. 74) + first-order shock switch (Eq. 75, C = 3);
      `docs/av/godunov_l1b_2026-10-08/`. The papers' limiters beat the ratio limiter on Gresho / KH / Sedov, slightly worse on Sod 1D / Noh; the shock switch fixes Noh and the Sedov radius. Default still `PairRatio` (user's call)
- [x] **L2** simplified GSPH (2026-10-08, uncommitted): `modules/godunov`, `schemes/gsph.py`, `GSPHConfig`, `CompressibleSPHScheme.GSPH`; tests (conservation, standard-SPH identity at constant `P`, Sod) and
      gradcheck; Sod 1D/2D, Noh, Gresho, Sedov vs the AV rows: `docs/av/godunov_l2_2026-10-08/`. Pressure blip removed (6x), best 1D velocity error, second order essential; weaker on Gresho
      and the Sedov radius, 25x the AV energy drift on Sod 1D (time-centring omitted). Still to do for L2: the Cha 2010 density-jump force profile as a measured baseline (expected = standard SPH), KH.
- [x] **L2 follow-ups** (2026-10-08, uncommitted): density-jump force baseline (`docs/av/godunov_densityJump/`: simplified GSPH = standard SPH, as predicted), KH (`docs/av/godunov_l2_2026-10-08/`: first order does not grow, second order 0.089 -> 0.104-0.106 with the papers' limiters),
      time-centred energy (D, `timeCentredEnergy`): **negative result**, 30x the energy drift, off by default
- [x] **L3** Inutsuka 2002 built and validated (2026-10-08, uncommitted): `InutsukaGSPH` (`schemes/gsphInutsuka.py`, `modules/godunov/wp_inutsuka.py`), Gaussian density, cubic `V_ij^2` / `s*` (verified against 1D quadrature), symmetric
      pair operators (conservation tests), 4 gradchecks, tests `test_inutsuka*`, `test_densityJump`; evidence `docs/av/godunov_l3_2026-10-08/`. Gaussian width `h_G = eta (m/rho)^(1/d)`: eta <= 1 stable, wider pairs on cold shocks
      (Noh / Sedov diverge); eta = 1 default. Good on Gresho (0.076-0.091 vs 0.109) and Sod velocity (0.0030) with no AV, KH 0.098-0.102, but (a) the Fig. 1 density-jump force is **not** removed at the stable width and (b) energy is not conserved at
      strong shocks (Noh 7.5e-3, Sedov 4.5e-2) unless the first-order shock switch is on (2.5e-5, 6.6e-4)
- [x] **decision (user, 2026-10-08):** `InutsukaGSPH` defaults to the first-order shock switch (C = 3); `GSPHConfig` state limiter default `VanLeerHarmonic` (was `PairRatio`); applied in `configurations/gsph.py`
- [ ] open: blob test, Wengen KH, solver ablation (Murante Table 1) on the Inutsuka scheme; the density-jump consistency vs pairing trade-off (Inutsuka's own remedy for the pairing: Eq. 81 plus a smaller width, which kills the gain)
- [x] godunov rows runnable in the larger sweep: `scripts/av_sweep_overnight.py --godunov` (13 AV-free rows over 11 cases, 3 KH at nx 256; ~5.6 h), `av_report` groups `godunov`, `gsphLimiters`, `godunovL3`
- [ ] bake-off rows into the AV table (same metrics)

## Status log

- 2026-10-08: plan opened; papers inventoried; four missing papers added to TO_ACQUIRE.
- 2026-10-08: layer 1 built and validated (`docs/av/godunov_l1_2026-10-08/`). A/B on Sod 1D/2D, Noh, Gresho, Sedov: reconstructed states change the
  Riemann *dissipation* by < 1 % (null result, as the form `p*(w) - p*(0)` implies); the infrastructure is what layer 2 needs. Not committed.
  Next: L2, first-order GPH (Cha & Whitworth Cases 1-3): `p*` replaces `P_i`, `P_j` in the SPH force; open design point: with the whole pressure
  term replaced, the reconstructed states carry the pressure gradient, so the layer-1 limiter now matters for accuracy, not only robustness.
- 2026-10-08 (end of day): bake-off smoke pre-flight re-run after layer 1 (`results/av_bakeoff_smoke_afterL1`): all jobs rc 0, every row bit-identical (tol 0) to the pre-7b pre-flight; the tree is whole for the bake-off.
- 2026-10-08 (evening): the four papers are on disk and read (`literature/` synced, abstracts verbatim). Plan rewritten around two findings: (1) the simplified
  Cha & Whitworth / Puri Eq. 15 / Iwasaki Eq. 24 form reduces to the standard SPH force at constant pressure, so it keeps the density-gradient inconsistency Cha 2010
  diagnoses and Murante finds it too diffusive for KH / blob: layer 2 is a baseline, layer 3 (Inutsuka's convolution form, Gaussian kernel, `V_ij^2`) the scheme; (2) Puri
  proves GSPH dissipation = signal-velocity AV with `v_sig = c` and no approach switch, the formal statement of Phase 7b's result. New step L1b (the papers' limiters).
- 2026-10-08 (night): layer 2 built and validated (`docs/av/godunov_l2_2026-10-08/`): works without AV on all five cases; pressure blip at the contact 6x smaller than any AV row, Sod 1D velocity error the
  best of any row, entropy spike 4x larger; second-order states (and the layer-1 `(rho, P)` reconstruction, unlike in the AV term) essential; Gresho 0.13 vs 0.06-0.09, Sedov radius -0.048 vs -0.015..-0.023.
  The simplified form equals the standard SPH force at constant pressure (tested), so L3 is what the density-gradient inconsistency needs. Next: the density-jump measurement + KH for L2, L1b limiters, then L3.
- 2026-10-08 (late night): L1b, density-jump baseline, KH for L2, time-centred energy (negative), L3 Inutsuka built, stability-probed and validated (`docs/av/godunov_l3_2026-10-08/`, `docs/av/godunov_densityJump/`).
  Stability is a Gaussian-width question (eta <= 1; cubic `V`); the shock switch is what makes energy conserve at strong shocks. The headline consistency gain of the convolution form (Fig. 1) does not show at the stable width.
  Tests and gradchecks pass; the full suite and the bake-off smoke pre-flight are run once at the end of the day.
- 2026-10-08 (end of day, after L3): full pytest suite green (1 skip); bake-off smoke pre-flight re-run (`results/av_bakeoff_smoke_afterL3`): 17 jobs rc 0, all 15 row/compSPH/KH256 dirs bit-identical (tol 0) to the pre-7b pre-flight; the tree is whole for the bake-off. Uncommitted.
