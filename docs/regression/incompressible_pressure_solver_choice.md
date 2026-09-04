# Incompressible (IISPH) pressure solver: what the Krylov options actually do (warpSPH)

**Date:** 2026-08-20. Records the measured character of the DFSPH incompressible
pressure operator and how the five opt-in Krylov solvers (BiCGStab, GMRES, CG,
BiCG, MINRES) behave on it, versus the shipped **relaxed-Jacobi** default. The
code's docstrings state the conclusion; this note keeps the evidence and the
per-method numbers so a future reader can follow without re-running the probe.
See `docs/historic_plans/INCOMPRESSIBLE_SOLVER_PLAN.md` for the full plan.

**Outcome:** the operator is **symmetric** and **negative-semi-definite** (with a
gauge mode), but **ill-conditioned** (κ ≈ 2.4e7) and **not diagonally
dominant**. **MINRES is the best all-round choice** (its design domain —
symmetric, NSD, gauge-singular — is exactly this operator: 9.7e-4 @200,
monotone, no divergence; it settles at the precision floor at long budgets).
BiCGStab and GMRES are the robust alternatives (BiCGStab diverges in fp32 at
long budgets — use `krylovFp64`); CG is viable (operand is symmetric) and
reaches the tightest long-budget residuals, but is the most fragile; BiCG is
the least robust. The relaxed-Jacobi default is unchanged (the Krylov branch
is skipped for it).

## The setup

`solveDivergenceFree` (the live `dfsph_step` path) drives a scalar pressure
field `p` so that `A p = b`:

- **operator** `A = dt · (IISPH pressure shift ∘ IISPH pressure accel)` — two
  matrix-free SPH passes, never assembled;
- **RHS** `b = sourceTerm = -divergence(predicted v)`;
- **preconditioner** `1/D` where `D = dt · computeAlpha(...)` (the IISPH
  diagonal, negative);
- **gauge** `p -= p.mean()` (the operator has a constant null space, `A·1 = 0`).

The Krylov solvers live in `modules/incompressible/krylov.py` (matvec/precond/
matvecT builders + `solvePressureKrylov` dispatch) and reuse the
`modules/shifting` solvers. CG and BiCG sign-flip the (negative) operator, so
they solve `-A p = -b` (a positive-semi-definite system). MINRES needs no sign
flip (it handles the NSD operator directly); instead it uses the `1/D`
diagonal as a **symmetrizing congruence** (`d = 1/√|precond|`, solve
`Ã ũ = c` with `Ã v := d⊙(A(d⊙v))`, `c = d⊙b`, then `x = d⊙ũ`).

## The operator, measured (fp32, 2D TGV lattice, `nx=24`, N=576)

| property | value | note |
|---|---|---|
| symmetry `‖A−Aᵀ‖/‖A‖` | 9.6e-07 | symmetric to fp32 precision |
| symmetric-part eigenvalues | min −4.7e-3, max +2.0e-10 | negative-semi-definite; the ~1e-10 top mode is the **gauge** null space |
| condition number (sym part) | 2.4e7 | dominated by the gauge mode |
| row diagonal dominance | −0.71 (min) | **not** diagonally dominant on every row |
| `computeAlpha` vs `Diag(A)` | rel-L2 3.3e-7 | the IISPH diagonal **is** the operator diagonal |

Two consequences worth stating:

- **The damped relaxed-Jacobi can diverge here** — the fixed-ω update has a
  hard stability window `ω < 2/ρ(D⁻¹A)`, measured `≈ 0.355` on this operator
  family, so any ω above it (e.g. the dataclass default 0.5) diverges
  regardless of the state. The full analysis, the measured μ, and the
  implemented `relaxationMode: optimal` stabilization are in the section
  below.
- **BiCG's `Aᵀ` is not the issue.** Because the operator is symmetric, the
  self-adjoint placeholder `buildIISPHMatvecT` (`Aᵀ = A`) is *exact*. BiCG's
  residual weakness is the indefinite/gauge-mode spectrum, not the adjoint.

## Relaxed Jacobi: the omega stability window and `relaxationMode: optimal`

(2026-08-20 addendum) The relaxed-Jacobi default's sensitivity to
`relaxationFactor` is now understood exactly. The update
`p ← p + ω D⁻¹ r` has error-iteration matrix `I − ω D⁻¹ A`; because the
operator is symmetric NSD with a gauge null space, `D⁻¹ A` is similar to the
symmetric `|D|^(−1/2)(−A)|D|^(−1/2) ≥ 0`, so its spectrum lies in `[0, μ]`
(gauge mode at 0) and the iteration is stable **iff** `0 < ω < 2/μ`, with
`μ = ρ(D⁻¹ A)`.

Measured on the dense operator (fp64, N=1024, `nx=32`): `μ ≈ 5.636` — the
spectral top is a *degenerate high-frequency lattice cluster* (top five
eigenvalues all ≈ 5.636), local in nature and essentially insensitive to
smooth grid deformation (5.6364 uniform → 5.6369 after 0.5-cell TGV
advection → 5.6379 after a full-cell advection; `nx=24`: 5.670), and
`dt`-invariant (μ is a ratio, so the dt/dt² variants share it). The window
is therefore `ω < 0.355` across the family: the dataclass default `ω=0.5`
**always diverges** (residual → 1e15 in 60 steps), the scheme-builder
default `ω=0.3` sits inside with ~15% margin, and **inside the window the
fixed-ω performance is flat** (0.3 vs `1/μ` vs `0.8·2/μ`: within ~3% at
every checkpoint) — there is no performance benefit to tuning ω, only the
stability margin matters. (The operator depends on particle
positions/densities, not velocities, so the felt "state dependence" is
really grid deformation; the smooth deformations measured above barely move
μ. Strongly deformed / free-surface states could push μ up and shrink the
window below a configured ω — that is the residual risk for fixed mode.)

Two stabilizations were evaluated:

- **Power-iteration spectral estimate + fixed ω** — *rejected*. μ must be
  estimated matrix-free (power iteration on the symmetric similar form), but
  the degenerate top cluster makes the Rayleigh quotient converge slowly:
  5 iterations underestimates μ by ~36% (→ ω = 0.44 → **divergence**),
  10 by ~9%, 20 by ~1%. That is 8–60% extra matvecs on the 32–64-step
  production budget for an estimate that is fatal when short and, when
  good, only beats ω=0.3 by <2%. (This is also why the earlier
  `modules/shifting/richardson.py` backtracking approach felt hacky: the
  power-iteration seed is unreliable on this spectrum and the trial/halving
  loop papered over it.)
- **`relaxationMode: optimal`** — *implemented*. Each step uses the exact
  1-D residual minimizer `ω_k = (r · A D⁻¹ r)/‖A D⁻¹ r‖²` in
  `p ← p + ω_k D⁻¹ r`, advancing the residual by the exact recurrence
  `r ← r − ω_k A D⁻¹ r` (re-verified against the true `b − A p` every 16
  steps to bound fp32 drift). It costs the same single accel+shift pair per
  step as a fixed step (`A(D⁻¹r)` replaces `A p`) — zero overhead — and the
  residual decreases monotonically by construction: no stability window, no
  tuning, no initialization. It works even with `relaxationFactor=0.5` set
  (outside the fixed window); the measured `ω_k` (0.21 → 0.43 over the run)
  even exceeds the fixed window edge 0.355 while still descending, because
  it minimizes the true residual rather than the error. Measured (N=1024,
  64 steps, zero start): final relative residual 4.8% vs 5.2% for in-window
  fixed ω=0.3, monotone throughout. Opt-in:
  `config.solverConfig.divergenceFreeSolver.relaxationMode = 'optimal'`
  (YAML: `relaxationMode: optimal`); the default `fixed` keeps the
  historical path byte-for-byte (the regression guard still pins it). It is
  defined only for the divergenceFree (IISPH) solver — the constant-density
  variant clamps pressures non-negative each step, which breaks the exact
  residual recurrence (it raises `ValueError`).

Practical guidance for the Jacobi path: keep `fixed` with
`relaxationFactor` ≤ 0.3 (the builder default) for historical behaviour;
use `optimal` whenever a state might push the window below the configured
ω, or if ω is raised toward 0.5. For actually *solving* (rather than
smoothing) the system the Krylov options above remain the right tool:
Jacobi in either mode is a smoother — the smallest non-gauge eigenvalue of
`D⁻¹A` is ~5e-8 (spread ~1e8), so no constant-ω Jacobi converges this
system fast.

Reproduce and extend these measurements with
`scripts/probe_relaxedJacobiOmega.py`: it builds the same TGV start state
as the unit tests, assembles the production operator densely, and sweeps
kernel, support radius (neighbor count), dimension (2D/3D), and resolution
(plus optional grid deformation), reporting the `D⁻¹A` spectrum (μ, window,
gauge mode, spread, the degenerate top cluster, power-iteration estimates)
and the fixed-ω / optimal convergence trajectories (tables + `--csv`).

Dimension dependence (first sweep, 2026-08-21): the same n_h=4 in 3D means
~4× more neighbors (422–515 vs 99 in 2D), and μ tracks that —
Wendland2 μ=13.5–14.3, window **≈ 0.14–0.15** (vs 0.355 in 2D), so the
builder default ω=0.3 *diverges* in 3D; across kernels the 3D windows are
B7 0.330 (only one that covers ω=0.3), Wendland4 0.214, QuarticSpline 0.166,
Wendland2 0.149. The window is roughly 40–55% of the 2D value at the same
n_h for every kernel, and μ is again nearly resolution-independent
(13.5 @ nx3=8 → 14.3 @ nx3=12).

## Per-method behaviour (divergence-free variant, N=1024, `‖b‖≈1.87`)

Relative residual `‖b − A p‖/‖b‖` of the returned iterate at 200 iters:

| method | rel. residual @200 | reading |
|---|---|---|
| relaxed-Jacobi (default) | n/a (diverged on this state) | see above; unchanged path |
| **MINRES** | **9.7e-4** | **best of all at 200**; monotone, no divergence; long budget settles at the precision floor (1.8e-5 @745 fp32, 1.6e-4 @478 fp64, then status −13 = Lanczos breakdown) |
| **BiCGStab** | 1.1e-3 | best of the *indefinite* methods at 200; but see the deep-dive below — that is its fp32 stagnation shoulder |
| **GMRES** | 2.8e-3 | most robust (no breakdown; fp32 == fp64); slowest of the good methods |
| CG | 4.8e-3 | much stronger than it looks here: 3.9e-5 @1200 fp32, 3.6e-6 @1200 fp64 (b is high-frequency on this state) |
| BiCG | ~1e2 (diverged) | least robust on the gauge-mode spectrum |

**Deep-dive (session 2) — what the 200-iter snapshot hides.** Full 1200-iter
true-residual trajectories on the same state: BiCGStab-fp32 *stagnates* at
~1.9e-3 by 400 iters and **diverges** (4e+04) by 1200; the cause is fp32
orthogonality loss in its shadow system, because `κ(M⁻¹A) ≈ 1.1e8 >
eps_fp32⁻¹` (on a uniform lattice `diag A` is constant, so the Jacobi
preconditioner is just a scalar and the methods face the raw κ). With
**`krylovFp64: true`** (opt-in; the recurrence runs in fp64, the SPH matvec
stays fp32) BiCGStab reaches 1.1e-4 @800 and CG 3.6e-6 @1200 at the same
matvec cost. CG itself reaches 3.9e-5 @1200 even in fp32 because
`b = −div(v*)` aligns with the large-|λ| part of the spectrum. GMRES(30)
holds 4.0e-4 @1200 with no breakdown (fp32 == fp64). **MINRES** — the minimum
residual method, whose design domain (symmetric, negative-semi-definite,
gauge-singular) is exactly this operator — is **implemented** (Phase 6,
`modules/shifting/minres.py`): a Givens-LQ Lanczos MINRES (one 2×2 rotation
per step, O(1) bookkeeping) preconditioned by the symmetrizing congruence,
verified per-iterate against a dense-`lstsq` reference
(`test_minresGivensMatchesDenseLstsq`). Measured on this state: 9.7e-4 @200
(fp32 == fp64), monotone, no divergence — better than any other method at the
production budget; at long budgets it settles at the precision floor
(1.8e-5 @745 fp32 / 1.6e-4 @478 fp64) and returns −13 (Lanczos breakdown,
i.e. the Krylov subspace is A-invariant at working precision — the residual
there is the best the method can do). One implementation subtlety worth
keeping: its per-step β/ρ tests run in the d-weighted congruence space, so
they compare against `dmin·atol` there, while the final verification checks
the original-space residual against `atol` (comparing the d-scaled residual
to an original-space floor reads as an instant false "breakdown" under this
preconditioner, whose scale d ≈ 0.038 puts the whole transformed operator
below the raw floor).

Practical guidance today: keep `relaxedJacobi` as the default — with
`relaxationMode: fixed` (historical behaviour; keep `relaxationFactor` ≤ 0.3,
see the ω-window section above) or `relaxationMode: optimal` (window-free,
monotone, same cost) — and for Krylov use `minres` (best residual per
iteration, monotone, no divergence risk) — with
`krylovFp64: true` for longer budgets; `bicgStab` with **`krylovFp64: true`**
or `gmres` are the robust alternatives; `cg` (with `krylovFp64: true`) wins
only when the budget is 1000+ iters and the tightest residual matters
(3.6e-6 @1200). The gauge floor of the RHS is negligible on this
state (`√n·|mean(b)|/‖b‖ ≈ 9e-8`), so the ~1e-3@200 numbers are not structural.

BiCGStab, GMRES and MINRES agree to within a small fraction of the pressure
scale (`test_krylovSolversAgree`, after removing the gauge), a strong check
that all three solve the same `A p = b`.

## Regression guard

`tests/test_incompressibleKrylov.py::test_relaxedJacobiRegression` fingerprints
an 8-iteration relaxed-Jacobi run on the seeded state (error sequence + pressure
mean) and re-runs it to confirm determinism. Because the Krylov branch is only
taken when `solverType != relaxedJacobi` (the default is `relaxedJacobi`), this
pins that the historical path was not altered by adding the opt-in solvers.
The optimal-step mode is pinned by
`test_optimalStepJacobiMonotoneAndWindowFree` (monotone over a full 64-step
budget with `relaxationFactor=0.5` set — outside the fixed window),
`test_optimalStepAtLeastAsGoodAsInWindowFixed`, and
`test_optimalStepRejectedForConstantDensitySolver`; the default stays `fixed`
(`test_relaxationModeDefaultAndRoundTrip`), so the fingerprint above is
unaffected.

## How to read / extend

- Change the solver with
  `config.solverConfig.divergenceFreeSolver.solverType`
  (`cg` / `bicg` / `bicgStab` / `gmres` / `minres`) and, for Krylov, raise
  `maxIterations` (the Jacobi default caps are too low) and tune `rtol`.
  Set `krylovFp64: true` on the same sub-config to run the recurrence in fp64
  (matvec stays fp32) — recommended for `minres`/`bicgStab`/`cg` at budgets
  beyond a few hundred iters on this operator.
- On the relaxed-Jacobi path, set
  `relaxationMode: optimal` on the same sub-config for the window-free
  per-step exact step (see the ω-window section above); `fixed` (default)
  needs `relaxationFactor` kept below the window edge `2/ρ(D⁻¹A) ≈ 0.355`.
- The operator probe is `test_operatorIsSymmetricNegativeSemiDefinite` (it
  assembles `A` densely on the small case via `_assembleA`); re-run it if the
  pressure operators (`pressure/iisph.py`, `incompressible/drift.py`) change,
  to re-confirm symmetry/definiteness before trusting CG or the BiCG matvec.