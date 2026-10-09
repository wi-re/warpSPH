# GODUNOV_SPH_PLAN layer 1: reconstructed (rho, P) states in the Riemann dissipation, A/B (2026-10-08)

`riemannReconstruction = True`: the left / right states of the pair's Riemann problem are the limited extrapolations of `rho` and `P`
to the pair midpoint (`modules/reconstruction/pairState.py`, gradients from `computeStateGradients`: difference gradient times `M^-1`),
instead of the particle values. Same limiter family and close-pair taper as the velocity reconstruction; a reconstructed state is
kept between the two particle values. Configs `riemannMuscl*` in `av_report` (group `godunovL1`); the first-order references are the
`riemann*` runs of `docs/av/riemann_7b_2026-10-08/`. Full profile, video on, no velocity alarm in any run.

| case | metric | riemann | + states | riemannLimited | + states | riemannLimitedB2 | + states |
|---|---|---|---|---|---|---|---|
| Sod 1D | L1(v_x) | 0.003687 | 0.003663 | 0.003890 | 0.003852 | 0.003694 | 0.003666 |
| Sod 1D | P contact spike | 0.0685 | 0.0685 | 0.0626 | 0.0628 | 0.0684 | 0.0683 |
| Sod 2D | L1(v_x) | 0.01551 | 0.01727 | 0.01759 | 0.01750 | 0.01553 | 0.01726 |
| Noh | post-shock rho error | -0.00095 | -0.00102 | -0.00267 | -0.00297 | -0.00095 | -0.00102 |
| Gresho | L1(v_phi) | 0.2098 | 0.2100 | 0.09056 | 0.09055 | 0.09099 | 0.09007 |
| Sedov | peak rho ratio | 2.322 | 2.329 | 2.463 | 2.464 | 2.347 | 2.354 |
| Sedov | shock radius error | -0.0203 | -0.0206 | -0.0235 | -0.0262 | -0.0213 | -0.0216 |

## Reading

A clean null result, as expected from the form of the term: `Pi = (p*(w) - p*(0)) (1/rho_i^2 + 1/rho_j^2)` compares the star pressure with and
without the velocity jump, so the states enter only through the impedance and the weak quadratic term. Every metric moves by < 1 % except
Sod 2D L1(v_x) of the raw-velocity rows (0.0155 -> 0.0173, slightly worse) and the Sedov shock radius of the limited row (-0.0235 -> -0.0262).
The reconstruction is the infrastructure layer 2 needs (there `p*` replaces the whole pressure sum, so the states carry the pressure
gradient); it is not an improvement to the dissipation term, and `riemannReconstruction` stays off by default.

## Verification of the build

- `tests/test_pairState.py`: linear field with exact gradients gives the midpoint value from both sides for every limiter; step / sign
  change keeps the particle values; states bounded by the particle values; `i <-> j` symmetry; `computeStateGradients` exact for linear
  `rho`, `P` on a lattice and a jittered lattice (float32 and float64).
- `gradcheck_dissipation.py` with reconstructed states (gradients as inputs): passes for Acoustic, Adaptive, HLLC after breaking the
  mirror symmetry of the 5-particle line case: there `rho_1 = rho_3`, so the clamp of the reconstructed state to `[min, max]` of the pair
  has zero width and a kink at `s_i = s_j`, where the central difference averages two subgradients. A genuine property of the clamp
  (a uniform-density lattice sits on it), not an adjoint bug (isolated `reconstructPairScalar` gradcheck passes at random states).
