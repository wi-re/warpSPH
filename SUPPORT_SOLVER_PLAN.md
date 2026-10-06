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
  rows, *without* re-summing rho -- so a clamped row's density belongs to a larger h than the kernels then use.

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
