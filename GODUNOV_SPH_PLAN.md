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

## Papers

On disk (`literature/`), read for this plan 2026-10-08: `inutsuka2002` (Eqs. 51-57, 63-65 `V_ij^2` and `s*_ij` for linear / cubic `1/rho`;
66-67 the update with Riemann `P*`, `v*`; 68 the time-centred MUSCL states; 70-75 velocity projection and the two stability switches,
`Cshock = 3`; 79-85 variable `h`), `cha2003` (Eqs. 9-16: four GPH cases; 17-23 van Leer's iterative solver; 24-27 the isothermal solver;
28-30 the linearisation, `P* = C1 Pa + (1 - C1) Pb`: Case 1 reduces to SPH for `C1 = 1/2`), `toro2009`, `vanleer1979`, `hopkins2013`,
`hopkins2015`, `parshikov2002`, `vila1999`. That covers every equation of layers 1-3.

Not on disk (`literature/TO_ACQUIRE.md`, Godunov-SPH section, Crossref-verified): Murante et al. 2011, Iwasaki & Inutsuka 2011,
Puri & Ramachandran 2014, Cha, Inutsuka & Nayakshin 2010. These settle choices the 2002/2003 papers leave open (limiter variables,
kernel gradient in place of the Gaussian convolution, the time-centred step, solver comparison); until they are read, those choices
are made from the 2002 paper's own prescription and recorded as such here.

## Design notes

- Pair geometry as in Inutsuka §3.1: the `s` axis along `x_ij`, 1D Riemann problem on the normal, tangential velocity passive.
- Inutsuka's integral formulas need a Gaussian kernel (`W(x - x_i) W(x - x_j)` integrates to `W(x_i - x_j, sqrt(2) h)`); the repo has
  `KernelFunctions.Gaussian`. Layer 3 either uses it, or the ordinary kernel gradient as the later codes do (decide when the missing papers are read).
- The existing Phase 7b objects carry over: `riemannStarState` / `hllcFlux`, `LimiterType` (the limiter family), the velocity
  reconstruction, `RiemannSolver`.
- Not variational and not separable (Riemann `p*` depends on `v`): outside the geometric-integration work (PESPH_PLAN §7.4).

## Steps

- [x] **L1.1** scalar gradients of `rho`, `P` (`reconstruction.computeStateGradients`, difference gradient x `M^-1`; exact for linear fields)
- [x] **L1.2** `reconstructPairScalar` / `reconstructPairRiemannStates`: limited midpoint extrapolation, same limiter + taper as the velocity, clamped between the particle values
- [x] **L1.3** the Phase 7b Riemann term takes the reconstructed `(rho, P)` (`DiffusionParameters.riemannReconstruction`, default off); tests, gradcheck; A/B vs first order: null result (`docs/av/godunov_l1_2026-10-08/`)
- [ ] **L2** first-order GPH momentum + energy, `schemes/` entry, Sod 1D/2D, Noh, Sedov, Gresho, KH; no AV
- [ ] **L3** Inutsuka 2002 proper
- [ ] bake-off rows into the AV table (same metrics)

## Status log

- 2026-10-08: plan opened; papers inventoried; four missing papers added to TO_ACQUIRE.
- 2026-10-08: layer 1 built and validated (`docs/av/godunov_l1_2026-10-08/`). A/B on Sod 1D/2D, Noh, Gresho, Sedov: reconstructed states change the
  Riemann *dissipation* by < 1 % (null result, as the form `p*(w) - p*(0)` implies); the infrastructure is what layer 2 needs. Not committed.
  Next: L2, first-order GPH (Cha & Whitworth Cases 1-3): `p*` replaces `P_i`, `P_j` in the SPH force; open design point: with the whole pressure
  term replaced, the reconstructed states carry the pressure gradient, so the layer-1 limiter now matters for accuracy, not only robustness.
- 2026-10-08 (end of day): bake-off smoke pre-flight re-run after layer 1 (`results/av_bakeoff_smoke_afterL1`): all jobs rc 0, every row bit-identical (tol 0) to the pre-7b pre-flight; the tree is whole for the bake-off.
