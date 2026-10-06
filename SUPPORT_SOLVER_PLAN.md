# Adaptive support solver (SUPPORT_SOLVER_PLAN)

> **Progress marker:** this plan's row in [PLANS.md](PLANS.md). Update its *Last worked* date and *Where it stands* whenever you work on this plan.

Started 2026-10-06 (user: "everything else depends on the support and changing it could potentially invalidate
everything that comes after"). Home of OPEN_PROBLEMS §21. Goal: one support solve whose h is well defined everywhere
(walls, shocks, free surfaces, smooth flow), consistent with the density the step uses, and with no post-hoc clamp.
Do it *before* AV_PLAN Phase 3+ and the compressible-walls 2D work build more numbers on top of the current h.

## Scope

Only the compressible schemes run an adaptive support solve (`schemes/monaghan.py`, `compSPH.py`, `crkSPH.py` ->
`modules/adaptiveSupport/optimalSupport.py:evaluateOptimalSupport`). Weakly-compressible, ACSPH, DFSPH/incompressible
configs default to `AdaptiveSupportScheme.NoScheme` (fixed h): nothing here can move them. Case builders also call the
solve once at initialisation (16 iterations) with their own config (sod, sodND, sedov, linearWave, yee, triplePoint,
regions2D, compressibleWalls, ...).

## What the code does today (read 2026-10-06)

- **Runtime:** `cases/compressible.py` sets `adaptiveSupportScheme='Owen'`, `adaptiveSupportCorrections=False` (no grad-h
  Omega terms); `adaptiveSupportIterations` is the config default **1**, so each RHS evaluation takes one relaxation step.
- **Owen** (`optimalSupportOwen.py`): psi_H = (sum_j |f'(r_ij/h)|)^(1/d) with W = h^-d f(r/h), i.e. a gradient-weighted
  neighbour count that scales like h/dx on a lattice; a per-(kernel, dim) lattice LUT (`owenLUT.py`, n_h in [2, 6]) maps
  psi_H to the n_h a lattice would have; `h <- (1 - a + a s) h`, `s = n_h_target / n_h`, `a = 0.4(1+s^-3)` (grow) or
  `0.4(1+s^2)` (shrink). The density is summed at the final h (consistent).
- **Monaghan** (`optimalSupportMonaghan.py`): Newton on `h = volumeToSupport(m / rho(h))` with `dF/dh` from the Omega term.
  Present, not used by any case at runtime.
- **The clamp** (013a22f, `supportVolumeClamp`, wall-only since beb381a): after the solve, `h <= n_h (m/rho)^(1/d)` on fluid
  rows. (All three schemes re-sum rho at the final h afterwards -- `computeDensities` / `computeCRKFactors` -- so rho and h stay
  consistent; an earlier note here said otherwise and was wrong.)

## Known facts (OPEN_PROBLEMS §20/§21, COMPRESSIBLE_WALLS_PLAN 2026-10-05)

1. Wall outer rows: Owen's h froze at ~1.8-2x its neighbour's despite equal m/rho (shockReflection -10 % in p without the
   clamp). The walls plan called it a "one-way ratchet"; the relaxation formula itself can shrink h, so it is more likely a
   **fixed point at large h** for a row whose kernel holds a mixed configuration (sparse / different-spacing wall, empty
   side). Unverified.
2. 1D Sod: 31 % of rows above the volume bound, up to 1.32x, at the shock / contact. Capping them halved Sod's contact
   pressure spike and raised the Sedov peak -- over-large h at discontinuities costs shock quality.
3. 2D smooth flow: h = 1.02x the volume bound on ~100 % of rows. Likely **not** an Owen error but the lattice offset of the
   summation density (`sum_j m W != rho0` on a lattice, `SimulationConfig.calibrateNormalization`): the bound uses the biased
   rho. 1D shows no offset (p50 1.0000). To verify.

## Steps

1. **Toy problems first** (static solves, no time stepping; `scripts/probe_supportSolver.py`):
   a. uniform lattice 1D/2D/3D: Owen h, Newton h, volume bound; with h started at 0.5x and 2x target -- does each return?
      quantify the lattice offset vs the 2 % (fact 3);
   b. density step 1:8 (Sod-like), static: h profile across the jump per solver; where does h exceed the bound and why;
   c. half-space (free surface) and fluid lattice next to a wall lattice of different spacing: reproduce the frozen
      outer row (fact 1) with no dynamics; which term of psi_H holds it there;
   d. Newton robustness: isolated particle, the 1:8 jump, a coincident pair; iterations to converge from 1 iteration per step.
2. Decide the target relation from step 1 (the derivation decides, not case numbers): neighbour count (Owen) vs
   h = eta (m/rho)^(1/d) self-consistent (Newton, the relation the grad-h Lagrangian derivation assumes), and how walls enter.
3. Implement the fix in the solve; the clamp becomes unnecessary (remove or keep as an off-by-default diagnostic).
4. Regression harness (all must be run, numbers reported, before anything is called done):
   - AV report full baseline + baselineQ (Monaghan), CRK (`crk` group, av_M0f_crk), compared with `--compare` to M0e /
     M0f -- every moved number explained;
   - compressible-walls cases: shockReflection near-wall rows (probe_wallPressureDeficit, -0.9 % today), closedBox,
     piston, Woodward-Colella with walls, Sedov 2D walled vs mirror;
   - test suite (`tests/test_physics.py`, `test_shockCapturing.py`, switch tests) + gradcheck of `modules/adaptiveSupport`
     if a warp kernel changes;
   - then a new reference baseline (M0g) if numbers move, with the user's go-ahead.

## Log

- 2026-10-06: plan created; code read (above).
- 2026-10-06 **step 1 (toys, `scripts/probe_supportSolver.py`, B7, n_h 4, float32, 64 iterations):**
  - *1a lattice:* no ratchet -- Owen and Newton both return to the target from h = 0.5x and 2x. B7's lattice density is
    exact (rho/rho_lat 1.00001), so the 2D offset is **not** the density (fact 3 was wrong). **Owen settles at 1.0205 h_lat
    in 2D and 1.0088 in 3D on a perfect lattice (1D exact)**; Newton lands on h_lat in every dimension.
    Cause: `wp_psi.py:computePsi` sums the real lattice (`gradSum`) and then **does not use it** -- the table's psi_H is a
    continuum shell approximation `sum_k |f'(k/n_h)| 2 pi k` (2D) / `4 pi k^2` (3D), exact only in 1D (two points per
    radius). Runtime measures the real lattice sum, reads n_h ~ 3.92 in 2D, and grows h 2 %. A table bug.
  - *1b density step 1:8 (1D, equal masses):* Owen counts neighbours regardless of mass: dense side of the jump h = 1.12
    h_vol, sparse side 0.83 h_vol; Newton = h_vol by construction (1.55 / 0.50 h_lat). This is the over-large h at Sod's
    shock/contact that the clamp was capping (§21 fact 2).
  - *1c free surface (2D half-space):* outer row Owen 1.30 h_lat (1.07 h_vol), Newton 1.21 h_lat (= h_vol).
  - *wall (dynamic, shockReflection M2 t 0.45, `probe_wallPressureDeficit.py --adaptiveSupportScheme Monaghan`):* Newton has
    no frozen row (outer h/dx 0.62) and the clamp is a no-op on top of it; but rows 1-2 read **+7.6 % / +9.4 %** in p
    (Owen + clamp -0.9 % / +3.0 %; Owen alone -10.2 %), rows 1-2 too close (0.08 dx). The wall model (rho_w V, fallbacks)
    was tuned against Owen + clamp -- to be re-examined if Newton is adopted, not read as a solver defect yet.
  - Running: full AV baseline with Newton (`results/support_solver/av_newton_*`) vs M0e (Owen) against exact solutions.
- 2026-10-06 **table fix verified:** `owenTable='lattice'` (new `CompressibleSPHConfig` field; `wp_psi.py` now also returns the
  lattice-sum psi_H it always computed) is monotone in n_h in 1D/2D/3D, differs from the shell table by 0.9785 / 0.9910 at
  n_h 4 in 2D / 3D (= the observed 2.05 % / 0.88 %), and makes Owen land on h_lat exactly in every dimension. Default still
  'shell' until the AV comparison is in.
- 2026-10-06 **literature (step 2 input):** Price 2012 §2.3 (Eqs. 10-12): the standard for Hamiltonian / grad-h SPH is the
  self-consistent h = eta (m/rho)^(1/d) solved with rho (Newton-Raphson; Springel & Hernquist 2002, Monaghan 2002); a
  neighbour-count parameter "is only related to the true number of neighbours so long as the number density within the
  smoothing sphere is approximately constant" -- i.e. not at discontinuities (toy 1b). Frontiere 2017 §(nh) and Owen 2014
  (CRKSPH, CompSPH; Spheral) use Owen's ideal-H, "optimiz[ing] the total kernel weight sampled at each point rather than
  maintain a strict number of neighbors". Paper-faithful assignment: Newton for the Monaghan host, Owen (fixed table) for
  CompSPH / CRKSPH. Running: AV suite with Newton and with Owen-lattice (`results/support_solver/av_{newton,owenlat}_*`).
- 2026-10-06 **AV suite, three solvers vs M0e (Owen-shell), full profile, clamp off for pure fluid** (`results/support_solver/av_*`):
  - fixed alpha = 1 (`none`): Newton Sedov peak 2.53 -> **2.89**, shock radius err -3.2 % -> -0.25 %, linear-wave error
    0.0237 -> 0.0164, Sod L1 -9 %; Gresho / Yee / Sod 2D/3D within +-5 %. Owen-lattice = Owen-shell within 3 % everywhere,
    so Newton's gains come from the h *definition* in compressed / discontinuous regions, not from the 2 % scale.
  - C&D: Newton Sod contact P spike 0.134 -> 0.029, L1 -15 %, Sedov +3.5 %; but **Gresho L1 +13 %, angular-momentum loss
    +49 %, KH growth -35 %** (Owen + clamp had the same signature, smaller). Owen-lattice keeps C&D Gresho 0.0726 / KH 0.100,
    so the 2 %-scale hypothesis is falsified: the smooth-flow loss comes from h following the density.
  - C_q = 2: same pattern (Sod better, Noh / Sedov within gates, C&D Gresho +15 %).
  - Suspect: the compressible cases run with `adaptiveSupportCorrections=False` (no grad-h Omega terms). Price 2012 Eqs.
    43-45: with h = h(rho) the Hamiltonian equations carry Omega; without it every fluctuation of h (Newton's h follows the
    density noise) is a force error. Owen's h is not h(rho), so Omega is the wrong pairing for it and the right one for
    Newton. Running: Newton + `adaptiveSupportCorrections=True` (`results/support_solver/av_newtongh_*`).
- 2026-10-06 **Newton + grad-h (`adaptiveSupportCorrections=True`) breaks energy conservation**: `none` Sedov drift 5.2e-4 ->
  **5.0e-2**, Sod 5e-6 -> 8e-4, KH 5e-7 -> 7.5e-5 (no divergence). Cause (read, warpSPHCore `coreOperations/wp_gradient.py`
  Symmetric branch): Omega divides each side's term (`f / Omega`) but both sides use one symmetrised kernel gradient (the
  scheme calls it with `KernelMeanSymmetric`); Price 2012 Eq. 44 needs `P_i/(Omega_i rho_i^2) grad W_ij(h_i) +
  P_j/(Omega_j rho_j^2) grad W_ij(h_j)` (each side its own h), and the energy equation the matching i-side term. Omega with
  a mean kernel is not the Hamiltonian form, so the existing grad-h path is not a test of the hypothesis. A correct grad-h
  path is a core-kernel change (warpSPHCore): its own sub-step if the Newton route is chosen.
  C&D with this broken path: Gresho 0.083, KH 0.053, Sedov drift 2.7e-2 -- no recovery, as expected; not evidence either way.

## Where it stands (2026-10-06, end of step 1) -- decision needed

| Option | What changes | Measured |
|---|---|---|
| A. Owen, table fixed (`owenTable='lattice'`) | bug fix only; bulk h 2 % / 0.9 % smaller in 2D / 3D | = today within 3 % (fixed alpha); C&D Gresho / KH kept; walls still need the clamp |
| B. Newton (h = h(rho)), no grad-h | Monaghan host's h definition | shocks much better (C&D Sod spike -78 %, Sedov peak 2.53 -> 2.89 at alpha 1, linear wave -31 %); C&D Gresho +13 %, KH -35 %; walls: no frozen row, no clamp needed, but near-wall +8 % p (wall model tuned to Owen) |
| C. Newton + correct grad-h (Price 2012 Eq. 44) | B plus per-side kernels with Omega in warpSPHCore's symmetric gradient and the energy equation | not built; the textbook-consistent form; whether it recovers C&D's smooth flow is the open question |

Paper defaults: Newton for the Monaghan host (Price 2012), Owen ideal-H for CompSPH / CRKSPH (Owen 2014, Frontiere 2017).

**Decision (user, 2026-10-06):** A now (`owenTable='lattice'` default), then build C and compare before choosing the Monaghan default.
- 2026-10-06 **option A applied:** `owenTable='lattice'` is the default (configs stored without the field load as 'shell').
  Smoke vs M0e: 24/30 pairs moved -- all 2D/3D cases; 1D linearWave / Noh bit-identical, 1D Sod moved only at round-off
  (every physics metric equal to 4 digits; the Verlet neighbour count -0.5 %: ~1e-7 h changes flip lattice-aligned pairs
  across the Verlet cut). New reference M0g running: `results/av_M0g_{baseline,baselineQ,crk,s3,phase2}`.
- 2026-10-06 **option C built:** `modules/pressure/wp_perSidePressure.py` (`computePerSidePressureWarp`): one neighbour pass
  giving a_i = -sum_j m_j [f_i grad W_ij(h_i) + f_j grad W_ij(h_j)] and du_i/dt = f_i sum_j m_j v_ij . grad W_ij(h_i),
  f = P/(Omega rho^2), Omega optional. Selected by `CompressibleSPHConfig.pressureFormulation='perSide'` in the Monaghan
  scheme (default 'meanKernel', unchanged). `tests/test_perSidePressure.py` (8 pass): momentum and energy conserved to
  round-off in 1D/2D/3D with and without Omega for random h / m / rho / P / v; equals the mean-kernel force at equal h.
  Noted on the way: the default path's du/dt (`-(P_i/rho_i) sum_j (m_j/rho_j) v_ji . grad W`) is not the exact energy
  conjugate of its m_j (P_i/rho_i^2 + P_j/rho_j^2) force (volume m_j/rho_j vs m_j/rho_i), which is part of why the
  existing grad-h path drifts. Running: AV suite with C (Newton + perSide + Omega) and with Newton + perSide, no Omega.
- 2026-10-06 **M0g taken** (default `owenTable='lattice'`): `results/av_M0g_{baseline,baselineQ,s3,phase2}`, smoke ref
  `results/av_smoke_ref_M0g`. vs M0e every headline metric moves <= 3.5 % (Sod 2D/3D and Yee L1 -2..-3.5 %, Sedov peak +0.5 %,
  KH / Gresho +-2 %), 1D at round-off, nothing diverged; Phase 2 verdicts unchanged (KH 0.011, Gresho alphaMean 0.049).
  Tests: 119 pass. CRK group aborted on 3D Sedov (OOM: other GPU tenants); re-run queued (`av_M0g_crk`, `_crk_sedov`).
- 2026-10-06 **option C measured** (`results/support_solver/av_C_*`, `av_Cnoomega_cullenDehnen2010`), C&D host:

  | | Owen, fixed table (M0g) | Newton | C: Newton + perSide + Omega | Newton + perSide, no Omega |
  |---|---|---|---|---|
  | Sod contact P spike | 0.134 | **0.029** | 0.059 | 0.044 |
  | Sod L1(v_x) | 0.0052 | 0.0045 | 0.0066 | **0.0044** |
  | Sedov peak rho (exact 4) | 2.17 | 2.23 | 2.23 | **2.29** |
  | Gresho L1(v_phi) | **0.073** | 0.082 | 0.083 | 0.081 |
  | KH A(1.5) (McNally 0.148) | **0.100** | 0.064 | 0.015 | 0.0096 |
  | RT mixing width | 0.38 | 0.37 | 0.25 | 0.32 |
  | Sedov energy drift | 5.3e-4 | 5.4e-4 | 5.3e-4 | 5.3e-4 |

  The per-side form fixes the energy problem of the old grad-h path (Sedov 5e-2 -> 5.3e-4) -- the formulation is right --
  but it collapses Kelvin-Helmholtz (frames: dense-phase particles break off into the light phase, the interface diffuses by
  particle mixing instead of rolling up). Without Omega it is worse still (0.0096): the per-side kernels, i.e. h jumping by
  sqrt(2) across a 2:1 contact when h follows the density, drive it, not Omega. Same with fixed alpha = 1 (C: KH 0.020, Sod
  L1 0.0041, linear wave 0.027). **No variant that ties h to the density keeps the contact/shear behaviour**; Owen's
  neighbour-count h, smooth across a contact, does. The contact problem itself is standard SPH's (Price 2008 conductivity,
  pressure-entropy SPH -> PESPH_PLAN), not something the support solve should paper over.

## Outcome / recommendation (2026-10-06)

- **Default: Owen with the fixed table (A), all three hosts** -- best on shear / contact flows, now exact on lattices; M0g is
  the reference. Its cost: the dense side of a shock / contact over-grows h (+12 % vs h(rho), toy 1b), which is the Sod
  contact spike Newton removes (C&D 0.134 vs 0.029).
- **Opt-in, kept:** `adaptiveSupportScheme='Monaghan'` (Newton) for shock-dominated runs; `pressureFormulation='perSide'`
  (exactly conservative, Lagrangian with Omega) -- not for flows with density contrasts at shear layers.
- **Walls:** Owen still freezes the outer wall row; the wall-only clamp stays (rho is re-summed at the clamped h by the
  schemes, so there is no rho-h inconsistency to fix -- corrected 2026-10-06).
- **Open candidate:** with the fixed table the bulk h equals the volume bound, so `supportVolumeClamp='always'` would now act
  only where Owen exceeds h(rho) -- min(Owen, h(rho)): the dense side of shocks / contacts, surfaces, wall rows. Whether that
  keeps Newton's shock gain without its KH cost is a direct test (queued).
- 2026-10-06 **M0g complete:** CRK `results/av_M0g_crk_all` (merge of `av_M0g_crk` + `av_M0g_crk_sedov`; 3D Sedov needed the GPU
  freed). vs M0f <= 6 % (Gresho L1 +5.9 %, Yee -6.2 %, KH +4.4 %, Sod 2D -2 %, 3D Sedov peak 2.09 -> 2.12, energy exact), 1D
  unchanged, nothing diverged.
- 2026-10-06 **min(Owen, h(rho)) measured** (`supportVolumeClamp='always'` with the fixed table, `results/support_solver/av_minclamp_*`):
  with the table fixed the bulk h equals the bound, so the clamp now acts only where Owen over-grows (dense side of shocks /
  contacts, interfaces). vs M0g (Owen, wall-only clamp):

  | | alpha = 1 | C&D | C&D, C_q = 2 |
  |---|---|---|---|
  | Sod contact P spike | 0.060 -> **0.026** | 0.134 -> **0.039** | 0.082 -> 0.045 |
  | Sod L1(v_x) | -8 % | -13 % | -6 % |
  | linear-wave error | +12 % | 0.0085 -> **0.0037** | -- |
  | Sedov peak | 2.54 -> 2.75 | 2.17 -> 2.20 | 1.83 -> 1.84 |
  | Gresho L1 / ang.-mom. loss | +1 % / +3 % | **+9 % / +5 %** | +8 % / +29 % |
  | KH A(1.5) | 0.021 -> 0.024 | **0.100 -> 0.087** | -- |

  Most of Newton's shock gain (Newton: Gresho +13 % / +44 %, KH -36 %) at about a third of its shear-flow cost. It still caps h
  at the KH interface, which is where the cost comes from; separating "shock" from "contact" would need a detector, i.e. a
  case-dependent switch -- not done. **Decision needed: keep `'walls'` (pure Owen for pure fluid; best shear / contact) or
  make `'always'` the default (better shocks, smooth-shear cost above).**
