# warpSPH — Incompressible (IISPH) Pressure-Solver Krylov Plan

> ## ✅ COMPLETE
>
> All 7 phases (0–6) are implemented and verified, not just marked done in the
> table below: `krylov.py`, `cg.py`, `bicg.py`, the `PressureSolverType` enum,
> and the config round-trip all exist and are wired into both
> `modules/incompressible/divergenceFree.py` and `incompressible.py`'s enum
> branches; `tests/test_incompressibleKrylov.py`'s 20 tests are green (checked
> 09-04, alongside the `warp-lang` 1.17 upgrade) — `test_minresGivensMatchesDenseLstsq`
> is a known pre-existing fp32 tolerance flake (fails ~1 run in a few, passes
> on retry, confirmed again this pass; unrelated to this plan's own code). The
> per-phase `- [ ]` checklists below were never individually ticked off but are
> superseded by this — read the *Status* table as authoritative, not the
> checkboxes. Two small file-map deviations, neither a functional gap:
> `tests/test_incompressibleOperatorProbe.py` and
> `test_incompressibleSolverComparison.py` were never split out as separate
> files — their content (the symmetry/definiteness probe, the cross-solver
> agreement check, the regression guard) landed as sections of
> `test_incompressibleKrylov.py` instead. The shipped default is still
> byte-identical relaxed Jacobi, exactly as the "Invariant across all phases"
> below requires; Krylov is opt-in, and per `DFSPH_IMPROVEMENT_PLAN.md`'s
> "Known-open" section it's known to break down on wall-bounded cases — that
> was never in this plan's scope to fix, only to make selectable.

Plan for adding opt-in **CG / BiCG / BiCGStab / GMRES** Krylov pressure solvers
to the DFSPH incompressible scheme (today a matrix-free **relaxed Jacobi**),
reusing the existing matrix-free Krylov library from the implicit-shifting
work, with **relaxed-Jacobi kept as the byte-identical shipped default**.
Written up here (rather than only in an ephemeral plan) so it can be picked
up and followed structurally in a later session. **Implemented** — Phases 0–6
are in (see *Status* below); the relaxed-Jacobi default is byte-identical and
the incompressible Krylov tests are green.

The solver set is deliberately **one solver per phase (Phases 1–4)** so any
solver can be targeted independently whenever its time comes. **BiCG is
Phase 4 (last)** because it is the only method that needs the operator
**transpose** `Aᵀ` (a real derivation); CG / BiCGStab / GMRES only ever apply
the existing matrix-free forward `matvec`, so they slot in directly.

## Status (as of implementation)

**All phases are in.** `PressureSolverType` + config fields, `krylov.py`
(matvec/precond/matvecT builders + `solvePressureKrylov` dispatch), the enum
branches in `divergenceFree.py` / `incompressible.py`, `modules/shifting/cg.py`
and `bicg.py`, and `tests/test_incompressibleKrylov.py` (12 tests, all green).
The relaxed-Jacobi path is untouched (the branch is skipped for the default),
confirmed by the `test_relaxedJacobiRegression` fingerprint.

**Phase-0 operator probe (measured, fp32, 2D TGV lattice, `nx=24`, N=576):**

| property | value | implication |
|---|---|---|
| symmetry `‖A−Aᵀ‖/‖A‖` | **9.6e-07** | the operator **is symmetric** (to fp32) → the BiCG `Aᵀ=A` placeholder is *exact* |
| symmetric-part eigenvalues | min −4.7e-3, max +2.0e-10 | **negative-semi-definite** with a near-zero **gauge mode** |
| definiteness | indefinite only via the ~1e-10 gauge mode | CG/BiCG get a **sign-flip** (solve `−A p = −b`) in `krylov.py` |
| condition number (sym part) | **2.4e7** | ill-conditioned → Krylov needs many iters; ~1e-3 relative residual at 200 iters |
| row diagonal dominance | −0.71 (min) | **not** diagonally dominant → the damped relaxed-Jacobi can diverge here (pre-existing) |
| `computeAlpha` vs true `Diag(A)` | rel-L2 **3.3e-7** | the IISPH diagonal is the **exact** operator diagonal → the Jacobi preconditioner is well-founded |

This resolves the two open questions: the operator is **symmetric** (so CG is
viable with the sign-flip, and BiCG's self-adjoint matvec is exact — no
Phase-4 derivation needed for symmetry), and it is **negative-semi-definite**
(not SPD), so BiCGStab/GMRES remain the robust all-round choices and CG/BiCG
are the weaker ones (BiCG is the least robust on the gauge-mode spectrum).

**Session-2 update (BiCGStab deep-dive).** The "BiCGStab deep-dive" section
below refines the per-method picture with full 1200-iteration true-residual
trajectories: the Phase-1 "~1e-3 @ 200 iters" was BiCGStab-fp32's
*stagnation shoulder* (it diverges by 1200 iters), the cause is fp32
orthogonality loss at κ(M⁻¹A) ≈ 1.1e8 (not the gauge mode, not a matvec
limitation), and CG is much stronger than expected on this state. Landed: the
opt-in `krylovFp64` bookkeeping flag (matvec stays fp32; ~10x better residual
for BiCGStab/CG). **Phase 6 — MINRES** is now **implemented** (session 3):
Givens-LQ Lanczos MINRES (`modules/shifting/minres.py`), symmetrizing
congruence preconditioning, `PressureSolverType.minres = 5`, dispatch in
`krylov.py` (no sign flip), 9.7e-4 @200 (best of all methods), per-iterate
verification against a dense-`lstsq` reference
(`test_minresGivensMatchesDenseLstsq`). The full handoff spec remains in the
deep-dive section below (historical). Test module is now 16 tests, all green.

**Session-4 update (relaxed-Jacobi ω window).** The default path's
sensitivity to `relaxationFactor` is pinned down exactly: the fixed-ω update
`p ← p + ωD⁻¹r` is stable **iff** `ω < 2/ρ(D⁻¹A)`, and `D⁻¹A` is similar to
the symmetric `|D|^(−1/2)(−A)|D|^(−1/2) ≥ 0` (NSD operator + gauge null
space). Measured on the dense operator: `ρ ≈ 5.636` — a *degenerate
high-frequency lattice cluster*, robust to smooth grid deformation (5.636 →
5.638 across 0.5–1 cell advection) and `dt`-invariant — so the window is
`ω < 0.355` across the family: the dataclass default 0.5 always diverges,
the builder default 0.3 sits inside with ~15% margin, and fixed-ω
performance is flat inside the window (only the margin matters). Landed: an
opt-in `relaxationMode: optimal` (per-step exact residual minimizer
`ω_k = (r·AD⁻¹r)/‖AD⁻¹r‖²`; zero extra matvec cost; monotonically decreasing
residual — no window, no tuning, works even with ω=0.5 configured; default
`fixed` stays byte-identical, regression guard still green). Rejected: the
power-iteration spectral estimate (the degenerate top cluster makes 5 power
iters underestimate ρ by ~36% → the derived ω diverges). Full evidence in
"Relaxed Jacobi: the omega stability window" in
`docs/regression/incompressible_pressure_solver_choice.md`; the analysis is
reproducible/sweepable via `scripts/probe_relaxedJacobiOmega.py` (kernel,
support radius/neighbor count, dimension, resolution, deformation →
`D⁻¹A` spectrum + fixed-ω/optimal trajectories, CSV out). Test module is
now 20 tests, all green. First 3D sweep (n_h=4): the same n_h means ~4×
more neighbors, μ rises to 6.1–14.3 across kernels (2D: 3.2–5.7), and the
window shrinks to 40–55% of the 2D value for every kernel (B7 0.330,
Wendland4 0.214, QuarticSpline 0.166, Wendland2 0.149) — so ω=0.3 diverges
in 3D for all but B7, and ω=0.5 everywhere.

## BiCGStab deep-dive (session 2)

**Question.** BiCGStab (the recommended Krylov option) reached only ~1e-3
relative residual at 200 iters on the seeded TGV state. Is that conditioning,
a bug, or fixable? What other solvers fit this operator?

**Setup.** Probe (`/tmp/bicgstab_probe.py`, ephemeral — numbers recorded
below): the session's seeded state (TGV 2D, `nx=32`, N=1024, random `v*`,
`x0=0`, `b = -div(v*)`, `‖b‖ = 1.87`), `A` densely assembled, and hand-rolled
Krylov loops recording the **true** residual `‖b − A x‖/‖b‖` along the way
(repo solvers verify the true residual only on return). matvec ~0.9 ms (GPU),
so 1200-iter trajectories take seconds.

**Hypotheses and verdicts.**
- **H1 — structural gauge floor** (`A·1 = 0` ⇒ no iterate can beat
  `√n·|mean(b)|`): **refuted.** `mean(b) = −5.3e-9` → floor 9.1e-8 relative.
  (The commented-out `sourceTerm -= sourceTerm.mean()` at
  `divergenceFree.py:76` is not needed on this state; it may matter on
  non-periodic states — left untouched.)
- **H2 — fp32 recurrence precision loss**: **confirmed.** BiCGStab-fp32
  stagnates at ~1.9e-3 by 400 iters and *diverges* (4e+04) by 1200;
  BiCGStab with **fp64 bookkeeping over the same fp32 matvec** reaches 1.1e-4
  at 800. Mechanism: `κ(M⁻¹A) ≈ 1.1e8 > eps_fp32⁻¹ ≈ 8.4e6`, so BiCGStab's
  shadow orthogonality `r0ᵀrk = 0` (which `α = ρ/rv` relies on) is destroyed
  by round-off → garbage updates.
- **H3 — pure conditioning**: real but not the whole story. On this *uniform
  lattice* `D = diag(A) = −1.471e-3` is **constant**, so the Jacobi
  preconditioner degenerates to a scalar and `κ(M⁻¹A)` = the raw `κ(A) ≈
  1.1e8`. On deformed states the diagonal varies and the preconditioner does
  real work, but the operator is an elliptic discretization with a gauge mode,
  so large κ is expected anyway.

**Spectrum (measured, gauge mode excluded).**
- `A` (sym part): `[−8.29e-3, +2.75e-10]` — NSD, gauge mode slightly positive.
- `M⁻¹A` (M = D): quantiles `[4.97e-8, 4.49e-3, 4.21e-2, 1.17, 5.64]`.
  Optimal Jacobi `ω = 2/(λmax+λmin) ≈ 0.355` with `ρ_opt ≈ 1` (~7000 iters to
  1e-5). Production `ω = 0.5` → spectral radius of `I − ω M⁻¹A` =
  `|1 − 0.5·5.64| = 1.82 > 1` → **relaxed-Jacobi diverges on this state**
  (consistent with the fingerprint test where the errors grow).

**True-residual trajectories (rel to ‖b‖, x0 = 0, 1200-iter budget).**

| method | 200 | 400 | 800 | 1200 |
|---|---|---|---|---|
| BiCGStab fp32 | 1.0e-3 | 1.7e-3 | **1.9e-3 (stagnant)** | **4e+04 (diverged)** |
| BiCGStab fp64 (fp32 matvec) | 8.8e-4 | 3.5e-4 | **1.1e-4** | breakdown ~1200 |
| BiCGStab fp32 + residual deflation | 1.3e-3 | 6.1e-4 | 1.9e-3 | 1.9e-3 (no help — H1 refuted) |
| CG (repo left-precond, fp32) | 4.8e-3 | 2.6e-3 | 3.7e-5 | 3.9e-5 |
| CG fp64 | 4.8e-3 | 2.4e-3 | 3.1e-5 | **3.6e-6** |
| GMRES(30) fp32 = fp64 | 2.8e-3 | 1.2e-3 | 6.1e-4 | 4.0e-4 (still drifting down; most robust) |
| MINRES (prototype, unpreconditioned) | 9.7e-4 | 2.2e-4 | **3.5e-5 (fp32)** | **1.0e-5 (fp64); monotone, no breakdown** |

Take-aways:
- **CG is much better than the Phase-1 note suggested** on this state:
  `b = −div(v*)` is high-frequency (Rayleigh quotient on A = −5.9e-3, close to
  the largest |λ| = 8.3e-3), so PCG sees a far smaller effective κ than the
  worst-case 1.1e8. CG's real risk is states where `b` excites the
  small-eigenvalue part (or the slightly-positive gauge eigenvalue +2.75e-10
  makes `−A` marginally indefinite → `cgSolve`'s −16 bail).
- **GMRES(30) is the most robust** (no breakdown, fp32 == fp64) but the
  slowest of the good methods at 1200 iters.
- **BiCGStab-fp32 is the worst at long budgets**: it *looks* fine at 200 iters
  (1.0e-3, the number the Phase-1 tests recorded) but that is the shoulder of
  its stagnation curve; longer runs return a diverged iterate. Its fp32
  "floor" is precision, not conditioning.
- **MINRES is the structurally right method for this operator** (symmetric,
  NSD, gauge-singular — exactly MINRES's design domain): it minimizes the true
  residual every step, has no shadow orthogonality to lose, and was the best
  and cleanest of all runs. → **Phase 6, handoff below.**

**Implemented in this session (in the tree, tested):**
- `RelaxedJacobiSolverConfig.krylovFp64: bool = False` (opt-in; both solver
  sub-configs; dict round-trip in `configurations/incompressible.py`).
- `solvePressureKrylov` wraps all four Krylov branches in fp64 bookkeeping
  when set: `b`/`precond`/`x0` → fp64, `matvec = λ p: matvec(p.to(fp32)).double()`,
  and the returned iterate is cast back to the production dtype before the
  gauge fix / final accel. The relaxed-Jacobi path never touches this code
  (branch guard `solverType != relaxedJacobi` in both variants).
- Tests: `test_krylovFp64ConfigRoundTrip`,
  `test_krylovFp64DoesNotWorsenResidual` (asserts fp64 ≤ 1.5× fp32 residual at
  200 iters **and** the dtype cast-back). Full module **14 passed**;
  `test_runner`/`test_caseSpec`/`test_physics` green.
- The MINRES wiring that was started (enum value `minres = 5`, dispatch
  branch, import) was **reverted** so the tree is green without `minres.py`;
  a NOTE comment in the enum points here.

**Phase 6 handoff — MINRES.** *(Status: implemented in session 3 — see the
*Status* section above; the spec below is the historical handoff document.)*
Rationale: best measured method (table above); the operator is symmetric to
fp32, so MINRES's symmetry assumption holds (the same guarantee that makes
BiCG's `Aᵀ = A` placeholder exact); no sign flip needed (MINRES handles NSD);
robust in fp32 (no breakdown at 1200 iters; fp32/fp64 within ~3×).

Design (decided — implement as specified):
1. **Interface** (mirror the siblings in `modules/shifting/`):
   `minresSolve(matvec, b, x0=None, tol=0.0, rtol=1e-5, atol=0.0,
   maxiter=None, precond=None, verbose=False, threshold=None, dim=1) ->
   (x, status, convergence)`. `atol = max(atol, tol, rtol*‖b‖)` floor;
   `convergence` = per-step recurrence residual, final entry the **verified
   true residual** `‖b − A x‖` (the `finish()` pattern from `bicgstabSolve`).
   Status codes from the family: `>=0` converged at iter; `−12` per-particle
   `|x|` threshold; `−13` Lanczos breakdown / stagnation (β_k → 0 with the
   residual still above tolerance — the Krylov subspace is invariant);
   `−14` max-iter budget.
2. **Preconditioning — symmetrizing congruence (not similarity, not left
   preconditioning):** MINRES requires a symmetric operator. Given the flat
   diagonal `precond = 1/D` (`D = diag A`, negative here), set
   `d = sqrt(|D|) = 1/sqrt(|precond|)` (elementwise; `precond=None` ⇒ `d = 1`)
   and solve `Ã ũ = c` with `Ã v := d ⊙ (A (d ⊙ v))` (symmetric when A is;
   NSD is fine for MINRES), `c = d ⊙ b`, then `u = d ⊙ ũ` (verified:
   `Ãũ = c ⟹ A(d⊙ũ) = b`). `x0` enters as `ũ0 = x0 / d`. All bookkeeping
   (Lanczos vectors, LQ scalars) lives in the transformed space; the threshold
   check and the returned `x` are in the original space (`d ⊙ ũ`).
3. **MINRES core (Lanczos + LQ of the tridiagonal):**
   - Lanczos: `w = Ã v_k − β_{k−1} v_{k−1}`, `α_k = v_k·w`, `w −= α_k v_k`,
     `β_k = ‖w‖`, `v_{k+1} = w/β_k` (keep `v_prev`, the `V` list, `β_prev`).
   - Normal equation: `M_k = [T_k; β_k e_kᵀ]` is **(k+1)×k** (T_k tridiagonal:
     diag α_i, off-diag β_i; last row has β_k at col k−1);
     `y_k = argmin ‖M_k y − β1 e1'‖`; **`x_k = x0 + V_k y_k` is the FULL
     solution at step k — not an incremental `x += V_k y_k`** (`y_k[:k−1] ≠
     y_{k−1}`; the prototype's first bug). Recurrence residual `‖r_k‖ =
     ‖M_k y_k − β1 e1'‖` (exact in exact arithmetic; monotone
     non-increasing).
   - **Implementation choice:** the prototype used `torch.linalg.lstsq(M_k, c)`
     per step (O(k³)/step, O(k⁴) total) — verified correct (SPD self-test →
     1.6e-15; NSD+gauge self-test → 3.2e-10; monotone). Shipping that for v1
     is fine (converges in a few hundred iters; the matvec dominates), but a
     Givens-LQ version (two 2×2 rotations per step, O(k)/step, O(n) memory) is
     the production form. **Do NOT transcribe the Givens update from memory** —
     fetch Trefethen's `minres.m` (www.math.utah.edu/~trefethen/minres.m) and
     port it line-by-line, then assert against the dense-lstsq version on a
     random SPD and a random NSD+gauge 30×30 system in a test (both must agree
     to ~1e-10 per iterate and the residual must be monotone non-increasing).
4. **Wiring:** `PressureSolverType.minres = 5` in
   `configurations/moduleConfigurations/solver.py` (replace the NOTE comment);
   `from ..shifting.minres import minresSolve` + an `elif` branch in
   `krylov.py`'s `solvePressureKrylov` — **no sign flip** (unlike CG/BiCG),
   pass `precond` through (the congruence uses it). The enum round-trips by
   name (`PressureSolverType[v]` in `dictToIncompressibleSPHConfig`), so no
   serialization change is needed.
5. **Tests** (`tests/test_incompressibleKrylov.py`): extend the
   `test_pressureSolverTypeAndDefault` name list with `'minres'`;
   `test_minresReducesResidual` (finite pressure, `_relResid < 2e-3` at 200
   iters — measured 9.7e-4); add MINRES to the gauge-removed agreement check
   (`test_krylovSolversAgree`); the dense-SPD/NSD self-test if the Givens
   version is shipped.
6. **Docs:** update the per-method table + usage notes in
   `docs/regression/incompressible_pressure_solver_choice.md`; flip the
   Phase-6 row in the Status table below.

**Other findings (for the record).**
- The repo `gmresSolve` solves its (tiny) Hessenberg least-squares via the
  **normal equations** (`Hc.T @ Hc`) — conditioning² in fp32; a minor weakness
  (GMRES was still the most robust here). Optional follow-up: switch to
  `torch.linalg.lstsq`.
- `bicgstabSolve`/`gmresSolve` record the *recurrence* residual per iterate
  (true residual only on return). The Phase-1 "1.1e-3 @ 200" number was the
  verified return value; the trajectories above show what happens *after* 200
  (fp32 BiCGStab diverges). At long fp32 budgets the status code is the only
  reliable signal.
- Prototype bugs hit while investigating (all in throwaway scripts, **not** in
  the repo code — recorded so Phase 6 doesn't repeat them):
  1. MINRES `x` update must be `x0 + V_k y_k` (full), not incremental.
  2. `M_k` must be (k+1)×k — the prototype first assembled it transposed.
  3. `torch.linalg.qr` defaults to *reduced* mode (Q (k+1)×k, not (k+1)×(k+1));
     the LQ needs `mode='complete'` (or skip QR — use lstsq).
  4. In the LQ residual `z = Q̂ᵀ β1 e1`, the prototype used the wrong factor
     orientation (`Q.T @ e1` instead of `Q @ e1`) — silent and divergent.
  5. A "quick" PCG probe mixed the x/r preconditioning (x step `p`, r step
     `A M⁻¹p`) — inconsistent unless `M = I`; and the left-preconditioned CG
     α is `(r, M⁻¹r)/(p, Ap)`, **not** `(r, r)/(p, Ap)` (off by the
     preconditioner scale — a factor ~680 on this state). The repo `cgSolve`
     was correct all along: the probe CG matched it exactly once fixed
     (4.8e-3 @ 200).
- **Phase-6 implementation bug** (in the repo code, caught by
  `test_minresReducesResidual`, fixed): the per-step β/ρ tests compared the
  d-weighted (congruence-transformed) residual against the original-space
  `atol` floor (`rtol·‖b‖`). Under the IISPH preconditioner (d ≈ 0.038 on
  this state) the whole transformed operator (norm ~1.2e-5) sits *below* that
  floor (~1.9e-5), so MINRES read as an instant Lanczos breakdown (status −13
  at iteration 0, residual untouched). Fixed by testing the transformed-space
  quantities against `dmin·atol` (`‖r‖ ≤ ‖d⊙r‖/dmin`); the final convergence
  verification still checks the original-space residual against `atol`. General
  trap for any preconditioned symmetric Krylov method: the
  convergence/breakdown floor must live in the same space as the residual it
  is compared against.
- **Relaxed-Jacobi ω window** (session 4): the `relaxationFactor` sensitivity
  is a hard stability window `ω < 2/ρ(D⁻¹A)` with measured
  `ρ ≈ 5.636` (degenerate high-frequency cluster — robust to smooth
  deformation, dt-invariant), i.e. `ω < 0.355`: the dataclass default 0.5
  diverges, the builder default 0.3 is inside. Fixed-ω performance is flat
  inside the window, so only the margin matters; don't tune ω for
  performance. `relaxationMode: optimal` (per-step exact residual minimizer,
  zero extra matvecs, monotone) removes the window entirely. The
  power-iteration spectral estimate was rejected — the degenerate top cluster
  makes a short power run under-estimate ρ by ~36%, pushing the derived ω
  out of the window; that is also why the `richardson.py` backtracking felt
  hacky (unreliable seed, trial/halving compensating).

- `/tmp` artifacts (ephemeral, safe to delete): `bicgstab_probe.py`,
  `minres_selftest.py`, `cg_debug.py`, `linearity_debug.py`,
  `bicgstab_probe.log`.

## Why

- The live pressure solve (`solveDivergenceFree`) is **damped Jacobi**: it
  converges linearly and slowly (defaults up to 32–64 iterations, each a full
  2-pass SPH matvec). A Krylov method converges in a small multiple of
  `√κ` matvecs, and we already have a principled preconditioner — the IISPH
  diagonal `a_ii` that Jacobi is already dividing by.
- The operator is **already matrix-free** (two SPH neighbor-loop passes, no
  matrix ever built) and the diagonal is precomputed in `O(n)`, so a Krylov
  method preserves the memory property exactly — it just calls the same
  `matvec` closure.
- The **implicit-shifting work already built generic, pure-torch, matrix-free
  Krylov solvers** (`modules/shifting/bicgstab.py`, `gmres.py`,
  `richardson.py`) whose interface is *exactly* "caller-supplied `matvec`
  closure + flat diagonal `precond` vector". Reusing them (not rewriting the
  solve loops) is the whole point of this plan.
- **People have tried CG before.** This plan scopes that honestly: it adds an
  **operator probe** (Phase 0) that measures symmetry + definiteness so CG
  viability is a *measured* decision, and it makes **BiCGStab/GMRES** the
  robust defaults for a non-SPD operator (matching the shifting experience,
  where the same operator family was found nonsymmetric/indefinite).

## Current state (what runs today)

Live path: `schemes/divergenceFree.py:160` → `modules/incompressible/divergenceFree.py`
(`solveDivergenceFree`, the **relaxed Jacobi**). The commented-out
`modules/incompressible/incompressible.py` (`solveIncompressible`) is the
density-error variant (adds a `clamp(p, min=0)` nonlinearity); it is **not**
on the live `divergenceFree_step` path today.

`divergenceFree_step` (`schemes/divergenceFree.py:159-169`):
```python
currentState.pressures = torch.zeros_like(currentState.densities) if currentState.pressures is None else currentState.pressures
dvdt_pressure, pressure, errors, pressures = solveDivergenceFree(
    particles=currentState, config=config, schemeConfig=schemeConfig,
    adjacency=adjacency, dvdt=dvdt + dvdt_diss, dt=dt)
currentState.pressures = pressure
```

`solveDivergenceFree` solves the **linear system** `A_op · p = b` for the
**scalar** pressure field `p ∈ ℝⁿ`:
```python
predictedVelocities = v + dt*dvdt
b   = sourceTerm = -divergence(predictedVelocities)   # = -rho0*div(v*)  [momentum/incompressible.py:20]
D   = alphas = dt * computeAlpha(...)                 # IISPH a_ii, NEGATIVE, precomputed ONCE [wp_alpha.py:315]
for i in range(maxIters):
    a_p  = computePressureAccelIISPH(p)        # 1 SPH pass (Scatter Symmetric gradient) [pressure/iisph.py:17]
    dx_p = dt * computePressureShiftIISPH(a_p) # 1 SPH pass (Scatter Difference divergence) [incompressible/drift.py:22]
    residual = b - dx_p
    p = p + omega * residual / D               # damped Jacobi step, omega = relaxationFactor
    p = p - p.mean()                           # gauge fix (kills the constant null-space mode)
    if mean|residual| < tol and i >= minIters: break
# final: a_p = computePressureAccelIISPH(p)  -> returned as dvdt_pressure
```
The "matrix" is **never materialized**. The matvec is a two-kernel composition
and the only thing built is an `O(n)` diagonal.

## Key observations

### The linear system, in standard form

The unknown is the scalar pressure `p`; the operator, preconditioner, and RHS
are all already computed (or trivially computable) matrix-free:

| standard concept | in this codebase | note |
|---|---|---|
| unknown `x` | `pressure` (scalar `[n]`) | not a vector field |
| `A x` (matvec) | `p → dt_scale · computePressureShiftIISPH(computePressureAccelIISPH(p))` | 2 SPH passes; `dt_scale = dt` (div-free) or `dt**2` (incompressible variant) |
| preconditioner `M⁻¹` | `1/(dt_scale · computeAlpha(...))` | the IISPH `a_ii`, **negated** → negative diagonal |
| RHS `b` | `sourceTerm` | `-rho0·div(v*)` (div-free) or `rho0 − rho*` (incompressible variant) |
| current iteration | `p ← p + ω·M⁻¹(b − A p)` | damped Jacobi, `ω = relaxationFactor`, `M = D = diag(A)` |
| gauge | `p −= p.mean()` | constant null space, `A·1 = 0` |

**Preconditioner gotcha (critical):** both `bicgstabSolve` and `gmresSolve`
apply the flat vector as `psolve(v) = precond * v` (**multiply**, not divide).
So we must pass `precond = 1/D = 1/(dt_scale·computeAlpha)`, *not* `D`. This
matches exactly what the Jacobi does (`residual / alphas`). `computeAlpha`
returns a negative value (clamped `max=-1e-6` upstream), so `precond` is a
finite negative reciprocal — safe.

The state (`positions`, `densities`, `masses`, `adjacency`) is **fixed**
throughout the pressure iteration (only `p` updates), so `A` and `b` are
genuinely constant → a proper linear solve. The only nonlinearity is the
post-iteration gauge re-center (linear, harmless) and — in the inactive
incompressible variant — the `clamp(p, min=0)` (a true inequality, see Scope).

### What is already reusable (the big win)

`modules/shifting/` already contains generic, **pure-torch** (import only
`typing` + `torch`, zero SPH dependency → no circular-import risk when
imported from `modules/incompressible/`) matrix-free Krylov solvers with the
exact interface we need:

- `bicgstab.py:62  bicgstabSolve(matvec, b, x0, tol, rtol, atol, maxiter, precond, verbose, threshold, dim) → (x, iters, convergence)` — 2 matvec/iter, robust non-SPD.
- `gmres.py:41     gmresSolve(matvec, b, ..., precond, ..., restart) → (x, iters, convergence)` — 1 matvec/iter, no breakdown modes, O(m·n) memory.
- `richardson.py   richardsonSolve(...)` — bounded auto-tuned fallback (optional last resort).
- `solverDriver.py  solveImplicitSystem / runKrylov` — primary + BiCGStab↔GMRES(+Richardson) fallback chain, returns the best iterate by *stamped true residual* (`convergence[-1]`). **Mirror this dispatch/fallback pattern** in the incompressible glue.

Both solvers return a `convergence` list whose **last entry is always the
verified true residual `‖b − A x‖`** of the returned iterate — that is what we
report as solver quality and use for the comparison harness.

### Operator character → which methods

`A_op = dt·div( −grad(p)/ρ )` is a weighted graph-Laplacian.
- **Continuum:** a self-adjoint (symmetric) elliptic operator, PSD with a
  1-D constant null space.
- **Discrete SPH:** generally **nonsymmetric** (the Symmetric-gradient and
  Difference-divergence are not exact discrete adjoints; `/ρ` is applied on
  one side; BCs break symmetry) and can be **indefinite** — the *same operator
  family* the shifting work probed (`exactHessian`) and found
  nonsymmetric/indefinite with a translation null space.

  **Measured (Phase 0, this pressure operator, fp32):** `‖A−Aᵀ‖/‖A‖ ≈ 1e-6`
  (symmetric to precision) and **negative-semi-definite** with a gauge mode —
  the *pressure* operator turns out to be more benign than the `exactHessian`
  shifting operator. See *Status*.

Consequences (this drives the phase order):
- **CG** needs SPD. The null space is harmless (converges on the
  null-orthogonal complement; re-center the result). But nonsymmetry/
  indefiniteness breaks its guarantees → **fragile; gate on the Phase-0 probe.**
- **BiCGStab** — robust non-SPD, low memory → **best all-round default**.
- **GMRES(m)** — most robust (no breakdown), 1 matvec/iter, O(m·n) memory →
  **best fallback / ill-conditioned choice**.
- **BiCG** — robust in principle but **needs `Aᵀ`**; see next subsection.

### The BiCG-adjoint finding (why BiCG is Phase 4 / last)

BiCG (unlike BiCGStab/GMRES/CG) updates a **shadow residual**
`r̃ ← r̃ − α·Aᵀ p̃`, so it needs the **adjoint matvec `Aᵀ`** every iteration.
Investigation of `warpSPHCore` (`/home/lu26029/dev/warpSPHCore`):
- `enumTypes.py` has **no adjoint/transpose operator scheme** — only forward
  `Gradient`/`Divergence` with `GradientScheme` (Naive/Symmetric/Difference/
  Summation) and `SupportScheme` (Gather uses h_i, Scatter uses h_j,
  MeanSymmetric, KernelMeanSymmetric, SuperSymmetric, PartialSymmetric).
- `warpier_adjoint.md` is about Warp **autodiff** adjoints, *not* an explicit
  operator-transpose. So there is **no ready matrix-free `Aᵀ`**.

Therefore BiCG's `matvecT` must be **derived**, in descending preference:
1. **Verified discrete adjoint** — identify the transpose SPH ops and prove
   `⟨A x, y⟩ = ⟨x, Aᵀ y⟩` numerically against an explicitly assembled `A`
   (test-only, small `n`). Most work, but makes BiCG production-grade.
2. **Self-adjoint approximation** `matvecT = matvec` — valid when the Phase-0
   probe shows a small symmetry residual `‖A − Aᵀ‖/‖A‖` (plausible, since the
   continuum operator *is* self-adjoint). Cheapest; ship with a documented
   caveat and let the probe set the threshold.
3. **Explicit-assembly `Aᵀ = A.t()`** — test/benchmark only (breaks
   matrix-free for production).

Because (1) is real deriving work and (2)/(3) are the realistic near-term
paths, **BiCG is scheduled last** and can be delivered at whatever fidelity the
probe warrants. **Probe outcome:** this operator is symmetric to fp32, so (2)
`matvecT = matvec` is *exact* (not an approximation) and `buildIISPHMatvecT`
ships as the self-adjoint alias. BiCG's residual weakness is the
indefinite/gauge-mode spectrum, not the adjoint.

### Cost / memory (matrix-free property is preserved)

| method | matvec/iter | SPH passes/iter | new memory | breakdown modes |
|---|---|---|---|---|
| relaxed-Jacobi (default) | 1 | 2 | O(n) | none (but slow convergence) |
| CG | 1 | 2 | O(n) | none — **needs SPD** |
| BiCG | 2 | 4 | O(n) | **needs `Aᵀ`** + rho/alpha |
| BiCGStab | 2 | 4 | O(n) | rho/omega (already guarded in the port) |
| GMRES(m) | 1 | 2 | O(m·n) | none (most robust) |

No matrix is ever built in any path — only the `O(n)` diagonal and (GMRES) the
`[n, restart]` Arnoldi basis. Each matvec transiently allocates the `[n, dim]`
accel intermediate (the composition), never an `[n, n]` operator.

## Scope

**In scope (this plan):** all four Krylov methods as opt-in; the `PressureSolverType`
enum + config; the `krylov.py` glue (matvec/precond/RHS builders + dispatch);
the operator probe; the comparison harness; the legacy byte-identity regression
guard; docs.

**Out of scope / deferred (each is a real, separate piece of work):**
- A **fused single-kernel matvec** (`div(−grad(p)/ρ)` in one pass) to drop the
  `[n, dim]` intermediate + a launch — optimization only; the two-kernel
  composition is correct to start.
- A **true constrained solve** for the inactive `solveIncompressible`
  `clamp(p, min=0)` inequality — here we approximate with a linear solve +
  post-projection clamp, documented as such.
- **Production BiCG** if no verified discrete adjoint is established — it then
  ships at self-adjoint-approximation fidelity (with caveat) or test-only.

**Invariant across all phases:** the shipped default stays
`PressureSolverType.relaxedJacobi`, and that path stays **byte-identical** to
today's `solveDivergenceFree`/`solveIncompressible` output. Every phase is
additive behind the enum; nothing changes for existing users.

## Status

| Phase | Solver / work | Status |
|---|---|---|
| 0 | Foundations: glue, enum, operator probe, baseline capture | ✅ Done |
| 1 | **BiCGStab** (first solver; validates the glue) | ✅ Done |
| 2 | **GMRES** | ✅ Done |
| 3 | **CG** (gated on Phase-0 probe) | ✅ Done |
| 4 | **BiCG** (last — needs `Aᵀ`) | ✅ Done |
| 5 | Comparison harness, regression guard, docs | ✅ Done (14 tests green; `docs/regression/` note) |
| 6 | **MINRES** (symmetric minimum residual; session-2 finding) | ✅ Done (Givens-LQ core + symmetrizing-congruence preconditioning; 16 tests green; `docs/regression/` note updated) |

## Phases

Each solver is its own phase so it can be started/landed independently. Phases
1–4 are additive (each just adds a `PressureSolverType` case); Phase 0 is the
shared substrate and Phase 5 is the wrap-up.

### Phase 0 — Foundations & operator probe (no behavior change)

The shared substrate. Nothing user-visible changes; default stays relaxed-Jacobi.

- [ ] **0a — `modules/incompressible/krylov.py`** (new). Builders over the fixed
  state:
  - `buildIISPHMatvec(state, config, schemeConfig, adjacency, dt_scale, supportScheme)`
    → closure `matvec(p) = dt_scale·computePressureShiftIISPH(computePressureAccelIISPH(p))`
    (Scatter, same `adjacency` as the Jacobi).
  - `buildIISPHPrecond(state, config, schemeConfig, adjacency, dt_scale)` →
    `1.0/(dt_scale·computeAlpha(...))` (the **`1/D`** the solvers multiply by).
  - `buildIISPHMatvecT(...)` → **stub** (returns `matvec` self-adjoint approx for now;
    Phase 4 replaces with the derived adjoint).
  - `solvePressureKrylov(...)` → dispatch scaffold (raises `NotImplementedError`
    for solver types not yet added) + post-solve gauge re-center (`p −= p.mean()`)
    + final `a_p = computePressureAccelIISPH(p)` + return `(a_p, p, errors, pressures)`
    (maps the solver `convergence` list → `errors`). Signature mirrors the
    `solverDriver.solveImplicitSystem` pattern (primary + optional fallback).
- [ ] **0b — `configurations/moduleConfigurations/solver.py`**: add
  `PressureSolverType(Enum)`: `relaxedJacobi`(default) / `cg` / `bicg` / `bicgStab`
  / `gmres`; add `solverType`, `rtol`, `atol`, `restart` to the per-solver config
  (both `pressureSolver` + `divergenceFreeSolver`), reusing `maxIterations`→`maxiter`,
  `tolerance`→`tol`. Update `__all__`, `buildDefault*`.
  - Also update the round-trip in `configurations/incompressible.py`
    (`incompressibleConfigToDict` / `dictToIncompressibleSPHConfig`, ~line 200).
- [ ] **0c — `tests/test_incompressibleOperatorProbe.py`** (new). Build a small case
  (reuse the `_gradcheck_common` line-case helpers, `n≈20–40`, like
  `scripts/gradcheck_incompressible.py`), **assemble `A` explicitly** (apply
  `matvec` to each basis vector — test-only), then record:
  - symmetry residual `‖A − Aᵀ‖_F / ‖A‖_F`;
  - on the null-orthogonal subspace `Q = I − 11ᵀ/n`: `λ_min`, `λ_max`, condition
    number (eigvalsh of `Q·(A+Aᵀ)/2·Q`, or Rayleigh quotients).
  - **Emit/record a verdict** (a small helper the later phases can import):
    `cg_viable = (sym_residual < τ_sym) and (λ_min > 0)`;
    `bicg_selfadjoint_ok = (sym_residual < τ_sym)`. This is what Phase 3 and
    Phase 4 read — it turns "should we trust CG / BiCG" into a measured fact.
- [ ] **0d — baseline capture** for the regression guard: snapshot the
  relaxed-Jacobi `(a_p, pressure, errors, pressures)` on the small case (a
  fixture / recorded values) so Phase 5 can assert byte-identity of the default.

### Phase 1 — BiCGStab (first solver; proves the glue end-to-end)

Chosen first because it reuses the already-hardened `bicgstabSolve` and is the
robust non-SPD default — lowest risk way to validate `krylov.py`.

- [ ] Add the `bicgStab` case to `solvePressureKrylov` → call `bicgstabSolve(matvec, b, x0, tol=…, rtol=…, maxiter=…, precond=1/D, threshold=…, dim=1)`.
- [ ] Wire `modules/incompressible/divergenceFree.py` to branch on `solverType`:
  `relaxedJacobi` → **existing code unchanged**; `bicgStab` → `solvePressureKrylov`.
  Warm-start `x0 = currentState.pressures` (mirrors the Jacobi's `*0.75`).
- [ ] Tests: `matvec` closure == direct composition; convergence quality
  (stamped `‖A p − b‖` ≤ relaxed-Jacobi residual); gauge `mean(p) ≈ 0`; status
  codes on a forced-breakdown/budget case.
- [ ] Verify the **default path is unchanged** (run with `relaxedJacobi`, diff
  against the Phase-0d baseline).

### Phase 2 — GMRES

Reuses `gmresSolve`; the robust, breakdown-free alternative (ill-conditioned).

- [ ] Add the `gmres` case → `gmresSolve(matvec, b, x0, …, restart=cfg.restart, precond=1/D)`.
- [ ] Wire `divergenceFree.py` (`gmres` branch).
- [ ] Tests: convergence; restart sweep (small `m` vs larger) for the
  residual-vs-memory tradeoff; confirm no breakdown on the indefinite probe case
  where BiCGStab might struggle.

### Phase 3 — CG (gated on the Phase-0 probe)

CG is the cheapest (1 matvec/iter, low memory, no breakdown) *if* the operator
is SPD. The Phase-0 probe decides whether it is.

- [ ] **New `modules/shifting/cg.py`** → `cgSolve(matvec, b, x0, tol, rtol, atol,
  maxiter, precond, verbose, threshold, dim)`, matching the `bicgstabSolve`
  signature/status-code convention exactly (left-preconditioned, `psolve(v)=
  precond·v`; breakdown code if `‖p_k‖` collapses; `convergence[-1]` = stamped
  true residual). ~45 lines. (Optional: extend `solverDriver.runKrylov` to add a
  `cg` case, or keep the incompressible dispatch self-contained — decide in 0a.)
- [ ] Add the `cg` case to `solvePressureKrylov`; wire `divergenceFree.py`.
- [ ] **Gate:** read the Phase-0 verdict.
  - If `cg_viable`: test CG on the real operator, assert convergence + residual.
  - If **not** SPD: still ship `cgSolve` (it's correct and useful for SPD
    subcases / as a reference), but the real-operator test is `xfail`/skipped
    with the probe's numbers attached, and the regression note says so.
- [ ] Unit-test `cgSolve` itself on a **known-SPD** matvec (a small SPD matrix
  assembled explicitly) so the solver is validated independent of the SPH operator.

### Phase 4 — BiCG (LAST — the transpose)

The only method needing `Aᵀ`. Scheduled last so the direct methods land first
and the adjoint derivation is isolated.

- [ ] **New `modules/shifting/bicg.py`** → `bicgSolve(matvec, matvecT, b, x0, …,
  precond, …)` (~50 lines; same status-code convention; `convergence[-1]` =
  stamped true residual).
- [ ] **Derive/choose `matvecT`** per the preference list in "Key observations":
  - [ ] **(1) verified discrete adjoint** — find the transpose SPH ops, then
    assert `|⟨A x, y⟩ − ⟨x, Aᵀ y⟩| / |⟨A x, y⟩| < tol` on the probe case against
    an assembled `A`. If it holds, `buildIISPHMatvecT` returns the true adjoint.
  - [ ] **(2) self-adjoint approx** — if (1) is infeasible and the probe's
    `sym_residual` is small, `buildIISPHMatvecT` returns `matvec`, with a
    documented caveat + a runtime `warnings.warn` when `sym_residual` is large.
  - [ ] **(3) explicit-assembly `Aᵀ = A.t()`** — test/benchmark-only fallback.
- [ ] Add the `bicg` case to `solvePressureKrylov`; wire `divergenceFree.py`.
- [ ] Tests: adjoint-identity check (if (1)); convergence; **BiCG vs BiCGStab**
  head-to-head (BiCGStab is the stabilized BiCG — expect it to be at least as
  robust; record where plain BiCG wins/loses).

### Phase 5 — Comparison harness, regression guard, docs

- [ ] **`tests/test_incompressibleSolverComparison.py`** (new) — the "full
  comparison in one pass": run **relaxed-Jacobi + CG + BiCG + BiCGStab + GMRES**
  on the same case and print a table (iters, final `‖r‖`, matvec count,
  wall-time); assert all finite + the probe's viability verdicts.
- [ ] **Regression guard:** default (`relaxedJacobi`) output is byte-identical to
  the Phase-0d baseline, on both `divergenceFree.py` and `incompressible.py`.
- [ ] **`incompressible.py` (density-error variant):** wire the same enum
  branch; linear solve + `clamp(p, min=0)` as a **post-projection**, documented
  as an approximation (the inequality is not enforced inside the Krylov
  iteration). Lower priority (path is currently inactive in `divergenceFree_step`).
- [ ] **Docs:** update `modules/incompressible/` + `configurations/.../solver.py`
  docstrings; add `docs/regression/incompressible_pressure_solver_choice.md`
  (honest per-method findings — which converged, iteration/residual numbers, the
  probe verdict, and why BiCG's fidelity is what it is), mirroring
  `docs/regression/implicit_shifting_operator_choice.md`.

## File map

| file | role | phase |
|---|---|---|
| `modules/incompressible/krylov.py` | matvec/precond/matvecT builders + `solvePressureKrylov` dispatch | 0, then 1–4 |
| `configurations/moduleConfigurations/solver.py` | `PressureSolverType` + fields | 0 |
| `configurations/incompressible.py` | config dict round-trip (~line 200) | 0 |
| `modules/incompressible/divergenceFree.py` | active solver; enum branch | 1 (2,3,4 add cases) |
| `modules/incompressible/incompressible.py` | density-error variant; enum branch | 5 |
| `modules/shifting/cg.py` | `cgSolve` | 3 |
| `modules/shifting/bicg.py` | `bicgSolve` (needs `matvecT`) | 4 |
| `modules/shifting/{bicgstab,gmres,richardson,solverDriver}.py` | reused as-is | 1–2 |
| `tests/test_incompressibleOperatorProbe.py` | symmetry + spectrum probe, verdicts | 0 |
| `tests/test_incompressibleKrylov.py` | per-solver unit tests | 1–4 |
| `tests/test_incompressibleSolverComparison.py` | all-methods table | 5 |
| `docs/regression/incompressible_pressure_solver_choice.md` | findings | 5 |

## Risks / open questions

- **Is the discrete operator SPD?** **Resolved (Phase 0):** symmetric but
  **negative-semi-definite** (gauge mode), not SPD. CG is therefore viable
  (with the sign-flip in `krylov.py`) but slow on the ill-conditioned spectrum;
  BiCGStab/GMRES remain the robust choices.
- **BiCG adjoint fidelity** **Resolved (Phase 0):** the operator is symmetric,
  so the self-adjoint `matvecT = matvec` placeholder is exact. BiCG is still the
  least robust method (indefinite/gauge-mode spectrum), as anticipated.
- **Conditioning + fp32 precision** (refined in session 2): the raw
  `κ(A) ≈ 1.1e8` (on a uniform lattice the Jacobi preconditioner degenerates
  to a scalar, so the methods face the raw condition number). The Phase-1
  "1e-3 @ 200 iters" was **not** the conditioning limit — it was BiCGStab's
  fp32 orthogonality loss (κ > eps_fp32⁻¹). With the opt-in `krylovFp64`
  flag BiCGStab reaches 1.1e-4 at 800 iters, and CG reaches 3.9e-5 at 1200
  iters even in fp32 (b is high-frequency on this state). See the *BiCGStab
  deep-dive* section.
- **Warm start** (`x0 = previous pressure`) is wired through
  `solvePressureKrylov`; on the seeded single-shot test states it is a cold
  start (zero previous pressure).

## Verification (per phase)

- Every phase: `python -m py_compile` on touched files; the phase's tests green;
  **default-path byte-identity** re-checked against the Phase-0d baseline.
- Phase 0: probe numbers printed + recorded (drives 3 and 4).
- Phase 5: full comparison table; `docs/regression` note written; then run the
  broader incompressible test slice (and `scripts/run_tests.sh`) to confirm no
  cross-module regressions.




