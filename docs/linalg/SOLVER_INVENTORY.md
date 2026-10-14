# Linear solvers across warpSPH, warpSPHCore and warpSPHIntegrators: inventory and a unified design

2026-10-14. Read-only survey (no code changed). Prepares the incompressible-solver cleanup of `SCHEME_CLEANUP_PLAN.md` §6.3: the solvers come first, the schemes after.
Measured with `WARPSPHCORE_AUDIT=1 scripts/audit_case.py tgv ... --nx 40` (1,600 particles, 2D periodic, float32) and a synthetic-system benchmark
(`scripts/bench_solver_inventory.py`, one 2D 5-point Laplacian + shift, n = 65,536, Jacobi preconditioner, 30 iterations, RTX PRO 6000).

## 1. Summary

* There are **no shared linear-solver types**. The repo has 11 iterative solver implementations (6 Krylov/stationary in `modules/shifting`, a pair-list CG, a GMRES and an Arnoldi in the integrators, a graph-Laplacian CG with multigrid in the core and its torch reference) behind 2 dispatchers (`solverDriver`, `solvePressureKrylov`), **5 hand-written relaxed-Jacobi loops** in the incompressible code, and about a dozen small dense solves done with `torch.linalg`.
* The Krylov code lives in `modules/shifting/` (written for implicit shifting, reused for pressure). It has no `linalg` home, no operator type, no preconditioner type, no result type.
* **Host syncs are the cost, not the flop count.** Every Krylov solver in `modules/shifting` reads the device 2-10 times per iteration (python `if` on a device scalar). The core's closure CG is the one solver that does it right (device-side stop flags, lagged pinned-memory read, graph replay, multigrid) and is the design reference.
* Measured on TGV, integrator syncs per step: `divergenceFree` 31, `omniIncompressible` 16, `dfsphReference` **310**. All five Jacobi loops converge to a different statistic, three of them read the host every iteration.
* The five Jacobi loops are one algorithm (Richardson with a Jacobi preconditioner, fixed or optimal step) plus a per-scheme projection (clamp >= 0, gauge, bad-row zeroing, wall extrapolation). They collapse into one `stationary` solver with a projection hook.

## 2. Iterative solvers (the Krylov family and friends)

Common calling convention of the `modules/shifting` family: `solve(matvec, b, x0, tol, rtol, atol, maxiter, precond, verbose, threshold, dim) -> (x, status:int, convergence:list[0-d tensor])`; `precond` is a flat vector multiplied in (so `1/D`, not `D`) or a callable; status `>= 0` converged at that iterate, `-10/-11` breakdown, `-12` per-particle `|x|` threshold, `-13` stagnation (GMRES), `-14` budget, `-15` stagnation (Richardson), `-16` indefinite (CG); `convergence[-1]` is always the verified true residual. Syncs/iter and ops/iter are measured on the synthetic system (ops include ~13 for the synthetic matvec).

| # | Solver (file) | Algorithm | Operator form | Unknown | Preconditioner | Output | syncs / it | ops / it | Consumers |
|---|---|---|---|---|---|---|---|---|---|
| S1 | `cgSolve` (`shifting/cg.py`) | left-preconditioned CG, SPD | `matvec` closure | flat `(n,)` | flat `1/D` or callable | `(x, status, [norms])` | 5.1 | 55 | `solvePressureKrylov` (sign-flipped for the negative IISPH diagonal) |
| S2 | `bicgSolve` (`shifting/bicg.py`) | left-preconditioned BiCG | `matvec` **and** `matvecT` (placeholder `A^T := A`, "provisional") | flat | same | same | **10.1** | 95 | `solvePressureKrylov` |
| S3 | `bicgstabSolve` (`shifting/bicgstab.py`) | BiCGStab, relative breakdown guards | `matvec` | flat, `dim` for the threshold norm | same | same | 5.1 | 73 | implicit shifting (default), `solvePressureKrylov`, omni constant-density (`CD_SOLVER`) |
| S4 | `gmresSolve` (`shifting/gmres.py`) | restarted GMRES(m), MGS reorthogonalised, **normal equations** `torch.linalg.solve(H^T H, ...)` per cycle | `matvec` | flat | same | same | 2.2 | 57 | implicit shifting (fallback / option), `solvePressureKrylov`, omni CD |
| S5 | `minresSolve` (`shifting/minres.py`) | Lanczos + Givens LQ; keeps the whole `n x min(maxiter, n)` basis; O(k^2) `solve_triangular` per step | `matvec` | flat | symmetrising congruence `d = 1/sqrt|precond|` (not a left precond) | same | 7.2 | 129 | `solvePressureKrylov` |
| S6 | `richardsonSolve` (`shifting/richardson.py`) | `x += omega (b - Ax)`, power-iteration seed + backtracking, no preconditioner | `matvec` | flat | none by design | same | 2.8 | 43 | `solverDriver` last-resort fallback |
| S7 | `solveImplicitSystem` / `runKrylov` (`shifting/solverDriver.py`) | picks S3 or S4, optional fallback chain by stamped residual (`krylov`, `krylov_richardson`) | - | flat | - | best `(x, status, hist)` | - | - | `computeImplicitShift`, `...Automatic` |
| S8 | `solvePressureKrylov` (`incompressible/krylov.py`) | dispatch to S1-S5 on `PressureSolverType`; fluid-mask restriction, Dirichlet lifting of the frozen boundary pressure, optional fp64 recurrence, gauge (`center` / `nonnegative`), final `a_p` | IISPH `accel o shift` closure | `(n,)` scalar | `1/(dt_scale * computeAlpha)` | `(a_p, p, errors, pressures)` (python floats) | per S1-S5 + ~10 post-solve | - | `divergenceFree`, `incompressible` |
| S9 | `CompactCG` (`incompressible/compactProjection.py`) | Jacobi PCG, float64, rhs normalised, warm start kept on the object, Dirichlet rows eliminated symmetrically | COO pair list `(i, j, w)`, `index_add_` (atomics: not bitwise reproducible) + diagonal `D` | `(n,)` | Jacobi | `(P, it, rel)` floats | **0.2** (host check every 10th iteration) | 23 | `omniIncompressible` (`projection='compact'`) |
| S10 | `gmres` (`warpSPHIntegrators/jfnk.py`) | restarted GMRES, Givens, python-list Arnoldi, left/right preconditioner | `matvec` (FD or JVP of the step) | flat | 1-arg callable | `(x, iterations)` | 5.1 | **586** | `JFNKSolver` (DIRK/IMEX, acoustic core, wave equation) |
| S11 | `_arnoldi`, `krylov_phi` (`warpSPHIntegrators/exponential.py`) | Arnoldi for `phi_k(A) v` (a matrix function, not a solve) | `matvec` | flat | none | `(result, n_iter)` | `float()` per Arnoldi step | - | ETD2RK, EXPRB32 |
| S12 | `_solveClosure` (`warpSPHCore/mfm/warpBackend.py`) + `CellMultigrid` (`mfm/multigrid.py`) | PCG on the face-closure graph Laplacian; Jacobi or one cell-aggregation Galerkin V-cycle as preconditioner; best-iterate tracking, `lam_cap` guard | **warp CSR kernel** (`mfmClosureMatvecCSR`, deterministic, one launch) | `(N, dim)` vector field (norms over all columns) | Jacobi or multigrid (coarsest level: dense `cholesky_ex` + `bool(info)`) | `lam` plus attributes on the backend (`closureResidualRatio`, `multigridLevels`) | **~0** (device-side freeze, lagged pinned-memory flag read `_CG_LAG`) | - | MFM / MFV closure |
| S13 | `_closeFaces` (`warpSPHCore/mfm/geometry.py`) | same problem in plain torch (the oracle of S12) | torch ops | `(N, dim)` | Jacobi | `A + C` | ~4 (`bool()` per test) | - | reference only |

Operator providers (what `matvec` actually is) — nothing shared between them:

| Provider | Form | Cost note |
|---|---|---|
| IISPH `computePressureAccelIISPH` o `computePressureShiftIISPH` (+ `where` masks) | 2 warp operator calls + torch | `divergenceFree`, `incompressible`, `solvePressureKrylov` |
| omni `_pressureAccel` / `_divergence` / `applyA` (+ analytic wall, MLS fit, Tikhonov shift, `wallPressureExtrapolation`) | warp + torch | the wall closure must stay linear for Krylov (`clampNonNeg=False`) |
| dfsph reference `_pressureAccel` / `_drhodt` | warp + torch | |
| band2018pb `bandPressureAccel` / `bandApplyOperator` (injected volumes) | warp + torch | |
| compact `(i, j, w)` COO, `index_add_` | torch | nondeterministic float order |
| implicit shifting `_multiplyLaplacianBlock`: per-pair `H` blocks, `dim^2` `scatter_sum` calls per matvec | torch | 4 scatters (2D) / 9 (3D) per application where one CSR block kernel would do |
| MFM closure CSR kernel | one warp launch | the model to copy |
| integrators: finite-difference or forward-mode JVP of the step | torch / autograd | |

## 3. Stationary / relaxed-Jacobi loops

All iterate `p <- p + omega D^-1 (s - A p)` on the same IISPH-type operator, with different stop statistics, projections and sync behaviour.

| # | Loop | Step | Projection after each update | Stop statistic | Host reads / iteration (TGV measured) |
|---|---|---|---|---|---|
| J1 | `_solveIncompressibleImpl` (`incompressible/incompressible.py`, constant density) | fixed `omega` | clamp `>= 0` or closed-domain gauge; wall-pressure extrapolation inside | `flooredOneSided` | **4** (`error`, then `min`, `max`, `mean` of the pressure for a history list); **no check schedule**. 12 syncs/step in the `divergenceFree` audit |
| J2 | `_solveDivergenceFreeImpl` (`incompressible/divergenceFree.py`) | fixed `omega` | gauge `center`, boundary re-pin | `meanAbsolute` + `rtol` disjunct | scheduled (`ConvergenceCheckSchedule`: `every` or `adaptive`); history stacked on device |
| J3 | `_solveDivergenceFreeOptimal` (same file) | **optimal step** `omega_k = r.q / q.q` (minimal-residual), residual recurrence re-verified every 16 steps | gauge, re-pin | as J2 | scheduled |
| J4 | `omniIncompressible._solve` | fixed `omega`; `invAlpha` with Tikhonov shift | clamp or gauge, bad-row zeroing (`|alpha| < 1e-25`, non-finite, `|p| > 1e25`), analytic wall | `max(Ap - s, floor)` mean | scheduled; one stacked read at the end. Krylov branch (`CD_SOLVER`) adds ~6 host reads for its accept/reject test |
| J5 | `dfsphReference._jacobiSolve` | fixed `omega = 0.3` | clamp `>= 0`, pins, exempt rows | one-sided mean | **3** (`fluidMask.any()`, boolean-mask `mean`, `float`); **no schedule**. **310 syncs/step** in the audit (102 iterations x 3) |
| J6 | `band2018pb._solve` | fixed, Tikhonov + gauge | clamp or gauge, bad rows | `max(Ap - s, floor)` mean | **2** (`rows.any()`, `float(...)`); no schedule |
| J7 | dual-time pseudo-iteration (`modules/artificialCompressible/dualTime.py`) | explicit RK in pseudo time (not a linear solve; the same stopping problem) | wall bounce, pressure projection | `eps_v` on device | scheduled + `GraphedTensorFunction`; the model for "iteration inside a graph" |

Three different stop statistics (`flooredOneSided`, `oneSided`, `meanAbsolute`; `convergence.py` names them) and the Krylov norm test `||r|| <= atol + rtol ||b||`: the Jacobi tolerances are not interchangeable with the Krylov ones. Convergence-check scheduling exists three times: `ConvergenceCheckSchedule` (history-adaptive), the core's lagged flag read, and the dual-time stop test.

## 4. Dense small solves (batched, per particle)

| Where | Matrix | Method | Sync |
|---|---|---|---|
| `warpSPH/linalg/pinv.py` (`pinvSymmetric`, `applyPinv`) | `(n, d, d)`, d <= 3, symmetric | Jacobi eigen on a zero-padded 3x3, one launch | none (done) |
| `modules/analyticBoundary/wallPressureMLS.py` | `(n, 2, 2)` float64 | `torch.linalg.eigh` + inverse by eigenvalue ratio | cuSOLVER info flag |
| `modules/mdbc/densityBand.py` | `(Nb, 2, 2)` | `torch.linalg.eigh` + Tikhonov-damped inverse, `float(median)`, `bool(any)` | 3+ |
| `modules/liu/interp.py` (+ `mdbc/density2025.py`) | `(n, dim+1, dim+1)`: 3x3 (2D), **4x4 (3D)** | `det`, then masked `pinv` (sync) or `inv_ex` (`syncFree=True`) | yes / no |
| `modules/shifting/preconditioner.py` block Jacobi | `(n, dim, dim)`, **not** symmetric in general | `torch.linalg.inv` with ridge, `bool(isfinite)`, `pinv` fallback | 1-2 |
| `modules/shifting/gmres.py` least squares | `(m+1) x m`, m <= 31 | `torch.linalg.solve(H^T H, ...)` (normal equations: squares the condition number) | 1 per cycle |
| `modules/shifting/minres.py` | growing triangular `k x k` | `solve_triangular` per step | - |
| `modules/crk/terms.py` (torch fallback), `shockCapturing` reference detector | `(n, d, d)` | `torch.linalg.pinv` | yes (kept as references) |
| `warpSPHCore/mfm/geometry.py invertMoments` | `(n, d, d)` | `inv_ex`, then `pinv(hermitian)` on the bad subset | `bool(bad.any())` |
| `warpSPHCore/polyfit/polyfit.py` | `(N, n, n)`, n up to ~44 | `eigvalsh` + `pinv(hermitian)` | yes |
| `warpSPHCore/pinv/wrapper.py` (`pinv_warp`) | d = 1, 2 warp; d = 3 `eigvals` + `pinv` | | yes (d = 3) |
| `warpSPHCore/crk/crk_terms.py` | `(n, d, d)` | `torch.linalg.pinv` | yes (reference) |
| `warpSPHCore/mfm/multigrid.py` | dense `<= 512` coarsest operator | `cholesky_ex` + `bool(info)` | 1 per setup |

The core's `pinv2x2` / `wp_pinv3x3` and `warpSPH.linalg.pinvSymmetric` are two implementations of the same operation on opposite sides of the package dependency (core cannot import `warpSPH`).

## 5. Findings

1. **Host syncs per iteration** (measured above): CG 5, BiCGStab 5, MINRES 7, BiCG 10, GMRES 2, Richardson 3, integrators GMRES 5; the core CG 0. They come from `if rho < eps` / `bool(torch.isfinite(rz))` / `rnrm < atol` on device scalars. A 30-iteration solve is 150 syncs; at the tolerances the incompressible schemes use (up to 64 iterations, two solves) that is the whole sync budget of a step.
2. **Four return conventions**: `(x, status, [tensors])`, `(x, iterations)`, `(P, it, rel)`, and "result plus attributes on the backend".
3. **Five Jacobi loops, three stop statistics, three of five without a check schedule** (J1, J5, J6). J1 and J5 are the two worst offenders in the audit.
4. **The same masking logic five times**: restrict to fluid rows (`where(fluid, p, 0)` around the matvec), Dirichlet lifting of the frozen boundary pressure, bad-row zeroing, gauge (`center` / `nonnegative` / `min`), `fluidIndex` gathers.
5. **Preconditioners are ad hoc**: a flat vector that is multiplied (and is `1/D` with `D < 0` for IISPH, so a host read `float(precond.mean()) < 0` decides a sign flip in S1/S2), a congruence in MINRES, a callable for block Jacobi, a V-cycle only in the core.
6. **Operators have no common type**; a Krylov solver cannot ask an operator for its diagonal, its transpose or whether it is symmetric. BiCG's transpose is a placeholder.
7. **Three Arnoldi/GMRES codes** (S4, S10, S11), one of them at 586 torch ops per iteration.
8. **fp64 handled four ways**: `krylovFp64` wrapper, `CompactCG` hard-coded float64, `.double()` in the optimal-step reductions, nothing elsewhere.
9. **Warm starts are scattered**: previous pressure in `particles.pressures`, `schemeConfig._compactCG.prevP`, `x0` arguments, the core's `lam0`.
10. **Package layering**: the core (MFM closure, `pinv`, polyfit, `invertMoments`) and the integrators (JFNK) cannot import `warpSPH.linalg`.
11. **`warp.optim.linear` (warp 1.17)** offers `cg`, `cr`, `bicgstab`, `gmres` over `wp.array` / `BsrMatrix` / `LinearOperator`, left preconditioner `M`, `check_every`, `use_cuda_graph`, and device-side convergence with conditional graphs when `check_every=0`. It does not know about masks, gauges, projections or this repo's status codes, and every operator in section 2 mixes torch and warp calls. Prototyped 2026-10-14, see section 8: it drives our torch-op operators, including the real IISPH operator.

## 6. Proposed unified `linalg` (lives in `warpSPHCore`, decision 1 below; `warpSPH.linalg` re-exports it)

```
linalg/
  pinv.py            (exists)  pinvSymmetric, applyPinv
  dense.py           small batched solve / inverse / det / symmetric eigen, d <= 4 (warp kernels, no host read)
  operators.py       LinearOperator: apply(x), diagonal(), applyT(x)?, n, dtype, symmetric/definite hints
                     Restricted(op, mask, lift)   -- the fluid-row restriction + frozen boundary lifting (finding 4)
                     Shifted(op, mu), Scaled(op, s) -- Tikhonov shift, sign flip (replaces the host sign decision)
                     PairListOperator(i/j or CSR, w, D) -- CompactCG's operator as one deterministic warp kernel
                     BlockPairOperator(CSR, H blocks)   -- implicit shifting's dim x dim blocks, one kernel instead of dim^2 scatters
  preconditioners.py Identity, Jacobi(diag, sign handled once), BlockJacobi (pinvSymmetric), Multigrid (wraps CellMultigrid)
  stopping.py        Tolerances(atol, rtol, maxiter), residual statistics (the three Jacobi ones + norm), StatusCode enum,
                     CheckSchedule (every | adaptive | lagged), breakdown predicates evaluated on device
  result.py          SolveResult(x, status, iterations, residuals): all device tensors, .host() reads once
  krylov/            cg, bicgstab, gmres, minres, bicg  (device-side predicates, iterate frozen with torch.where from the step a test fires:
                     bitwise the stop-at-first-test loop, as the core CG proves; lagged flag read; graph-capturable fixed-count mode)
  stationary.py      richardson / jacobi(omega | optimal, project=callable)  -- replaces J1-J6
  gauge.py           center | nonnegative | min | none
```

Rules the design follows from the findings:

* **One result type, all device-resident**; the one host read happens at the end of the solve (or is lagged). Callers that need floats call `.host()`.
* **Stopping is data, not code in the solver**: `Tolerances` + a statistic + a `CheckSchedule`. The three Jacobi statistics stay available (they are not interchangeable), selected by the scheme config as today.
* **Execution modes of the same solver bodies**: (a) host-checked with lagged read (default, equals today's behaviour minus the syncs), (b) fixed-count with device freeze (graph-capturable; the pattern of S12), (c) a `capture_while`-style conditional graph (the dual-time loop shows the scaffolding). (c) was prototyped against `warp.optim.linear` (section 8): it works with torch-op operators on a shared stream.
* **Bitwise strategy**: the freeze-on-device formulation returns exactly the stop-at-first-test iterate, so S1-S6 can be rewritten with unchanged results (`test_incompressibleKrylov.py`, `test_implicitShiftingSolvers.py`, `test_compactProjection.py` plus `scripts/scheme_baseline.py` on the incompressible schemes in `every` schedule mode). Only the Jacobi `adaptive` schedule changes results (documented: it can overshoot the first converged iteration by up to the checkpoint gap); keep `every` as the baseline and compare `adaptive` separately.

### Migration order (each step bitwise unless noted, tests listed)

1. `result.py`, `stopping.py`, `gauge.py` and the `stationary.py` solver; port **J5** and **J1** first (310 and 12 syncs/step). Check schedule `every` is bitwise; `adaptive` is the new default only after A/B.
2. Port J2/J3/J4/J6 onto `stationary`; delete the five copies of the masking logic via `Restricted`.
3. `operators.py` + `Restricted`; `solvePressureKrylov` becomes a thin builder (operator, preconditioner, gauge) over `krylov/*`.
4. `krylov/*` with device predicates (S1-S6), removing the 2-10 syncs per iteration; replace the GMRES normal equations by Givens rotations (as MINRES) on the device.
5. `BlockJacobi` through `pinvSymmetric`; `BlockPairOperator` for implicit shifting (4-9 scatters per matvec -> 1 launch).
6. `PairListOperator` for `CompactCG` (deterministic, 23 ops/it -> a few launches).
7. `dense.py` (symmetric `eigh` and small solves for `wallPressureMLS`, `densityBand`, `liu/interp` 3x3 / 4x4, `invertMoments`).
8. JFNK: `warpSPHIntegrators` is torch-only (depends on neither warp nor the core), so it cannot import the layer: give `JFNKSolver` a `linear_solver=` callable (two call sites, `jfnk.py` lines 662 and 825; default stays its own `gmres`) and ship the adapter with `linalg` (section 8.4).
9. Core MFM closure CG: keep as is (it is the reference, perf-tuned and bitwise-baselined).

## 7. Decisions (user, 2026-10-14)

1. **The layer moves to `warpSPHCore`** (`warpSPHCore/linalg`), and `warpSPH/linalg/pinv.py` (`pinvSymmetric`, `applyPinv`, `tests/test_linalg.py`) moves with it as core functionality; `warpSPH.linalg` stays as a re-export so imports and the existing consumers (CRK terms, Wadsley, gradient) do not change. The core's own `pinv/` (`pinv_warp`: `eigvals` + `pinv` for d = 3), `invertMoments`, `polyfit` and the crk torch reference can then use it. Not done yet.
2. **Pure-warp solvers, plus a branch onto `warp.optim.linear`**: scoped in section 8.
3. **`adaptive` check schedule** becomes the default for the Jacobi loops too (J1, J5, J6 change results slightly; the baselines stay on `every`). Worth less once the loops are device-side warp code, but it still decides graph-ability and scheduling.
4. **BiCG stays** as an option, for comparison, as long as it solves what it is meant to solve (its transpose is the placeholder `A^T := A`, exact only for a symmetric operator; the symmetry probe in `test_incompressibleKrylov.py` covers that). It gets the same device-side treatment as the rest.

## 8. Scope: pure-warp solvers and a `warp.optim.linear` branch

### 8.1 What `warp.optim.linear` does (read from the source and exercised)

* `cg`, `cr`, `bicgstab`, `gmres` over `LinearOperator(shape, dtype, device, matvec(x, y, z, alpha, beta))` with `z = alpha A x + beta y`, an operator-valued left preconditioner `M`, batched subproblems, `restart`.
* The convergence test is **device-side**: `atol` is computed from `||b||` on the device, and once a residual passes it the update kernels use `alpha = where(resid > tol, rz/pAp, 0)`, i.e. they **freeze the iterate** (the same trick as the core's closure CG). `check_every = N` reads the residual on the host every N iterations; `check_every = 0` + `use_cuda_graph = True` wraps the loop in `wp.capture_while` (a conditional graph node), so the whole solve is one graph launch with a real early exit. `check_every = 0` without a graph just runs all `maxiter` iterations with frozen updates (no early exit).
* It has **no** breakdown detection (a non-positive `pAp` is not flagged), no status codes, no per-entry threshold bailout, no stagnation test, no true-residual stamp (the residual is the recurrence one), no MINRES, no BiCG, no Richardson/Jacobi, and no hooks for masks, gauges or projections (a projection between iterations is not expressible; a mask is part of the operator).

### 8.2 Measured (`scripts/probe_warpOptimLinear.py` (synthetic) and `scripts/probe_warpOptimLinearReal.py` (IISPH), RTX PRO 6000, warp 1.17, CUDA 12.9)

Synthetic 2D Laplacian, n = 65,536, Jacobi preconditioner, tolerance 1e-4 (all reach residual 9.45e-5 in 46-50 iterations):

| Configuration | Time |
|---|---|
| `shifting.cgSolve` (torch, 5 syncs / iteration) | 6.9 ms |
| `warp.optim.linear.cg`, pure-warp matvec, host check / 10, no graph | 9.8 ms |
| same, host check / 10, graph | 1.7 ms |
| same, `check_every=0`, conditional graph | 1.5 ms |
| same, nested inside a `torch.cuda.graph` capture (replay) | 0.9 ms |
| torch-op matvec (`wp.to_torch` views), host check / 10, no graph | 13.7 ms |
| torch-op matvec, conditional graph | 2.1 ms |
| torch-op matvec, nested in a torch graph (replay) | 1.2 ms |

Real IISPH pressure operator on the TGV state (N = 6,400, `(-A) p = b`, tolerance 1e-4, same iteration counts within 1-2 and residuals 9.8e-5 to 1.0e-4 in every pair):

| Configuration | Time |
|---|---|
| `shifting.cgSolve` | 114-174 ms (208-329 iterations) |
| `warp.optim.linear.cg`, host check / 10, no graph | 135 ms |
| same, conditional graph | 28 ms |
| same, nested in a torch graph (replay) | 25 ms |

So a device-side loop is 6-7x faster than the present host-checked torch loop on the real operator with identical convergence. This compares loop styles, not warp against torch arithmetic: a torch solver rewritten with device predicates would close part of the gap, and fused warp update kernels (one launch for `x += a p; r -= a Ap`, as warp's `_cg_kernel_1`) the rest. Not measured.

### 8.3 Hazards found

* **Stream.** Torch ops inside a warp-captured matvec run on torch's current stream, and CUDA refuses to capture ("operation would make the legacy stream depend on a capturing blocking stream"). Fix, verified: run the torch part under `torch.cuda.stream(torch.cuda.ExternalStream(wp.get_stream(dev).cuda_stream))`, as the adapter would. Inside our existing torch-graph capture (`utils/cudaGraph.py` already shares the stream with warp) it works unchanged after `wp.capture_begin(external=True)`.
* **Allocation.** Torch temporaries inside a graph that warp (not torch) captured come from the ordinary caching allocator, not a graph pool; their addresses can be reused after capture. Safe route: let warp's loop be captured **inside** the torch graph (verified), or make every temporary a preallocated buffer.
* **`maxiter` is baked into the captured graph** (and the tolerance is read at capture time from device memory, so it can change per replay; `maxiter` cannot): the capture key must include it, like the other config scalars (`graphRoute.liveConfigKey`).
* **No early exit without the conditional graph**; the conditional graph needs a CUDA >= 12.4 toolkit and driver (12.9 / 13.3 here).
* A fixed `warp.optim` workspace holds fixed-address buffers (`LinearSolverState`); reuse it across steps for graph replay.

### 8.4 Plan and rough size

Three engines behind one `linalg.krylov` interface, chosen by `engine=` (default `warp`; `torch` kept as the bitwise reference through the migration, then retired; `warpOptim` as a comparison/benchmark branch).

| Piece | Work | Size |
|---|---|---|
| `warpOptim` adapter: torch closure -> `LinearOperator` (shared stream, `from_torch` views), diagonal -> `M`, `SolveResult` (status from the residual vs `atol`; no breakdown codes), x0 copy | cg / bicgstab / gmres only; maps `restart`, tolerances | small (hours) |
| Pure-warp CG | fused update kernels as warp's `_cg_kernel_1/2` plus device flags for `pAp <= 0` (status -16), non-finite, threshold; status as a device int | small-medium |
| Pure-warp BiCGStab, BiCG | rho/omega breakdown flags on device; BiCG needs `applyT` (placeholder for now) | medium each |
| Pure-warp GMRES | Arnoldi on a preallocated basis, device least squares (warp has a tiled LS kernel to crib; replaces the normal equations), right preconditioning for JFNK | medium-large |
| Pure-warp MINRES | the 3-term Lanczos recurrence with Givens needs no stored basis (the present code stores `n x k` and does an O(k^2) triangular solve per step) | medium |
| Pure-warp Richardson / Jacobi / optimal step | cheap kernels, but the per-scheme projections (clamp, gauge mean, bad-row zeroing, wall extrapolation) must be kernels or torch ops on the same stream | small + per-scheme |
| Loop scheduler: host-lagged, fixed-count freeze, `capture_while` | one driver used by every solver (`_run_capturable_loop` is the template) | medium |
| Operators in warp (`PairListOperator`, `BlockPairOperator`, IISPH accel o shift as one operator) | most of the present operators are torch closures around warp operator calls; they work as they are on the shared stream | per operator |

The torch-closure operators need no rewrite for any of this (section 8.2 used the real one); pure-warp operators are an optimisation, not a prerequisite.

### 8.5 Integrators

`warpSPHIntegrators` depends on `torch` and `numpy` only (not warp, not the core), so it cannot import the layer. The seam is small: `JFNKSolver` calls its module-level `gmres` in two places (the Newton step, `jfnk.py:662`, and the implicit-differentiation adjoint, `jfnk.py:825`). Add `linear_solver: Callable = None` to `JFNKSolver` (default: its own `gmres`; a callable `(matvec, rhs, *, tol, maxiter, restart, preconditioner, preconditioning) -> (x, iterations)`), and ship an adapter `linalg.krylov.asJFNKLinearSolver(solver='gmres'|'bicgstab'|..., engine=...)`. That gives every scheme that runs through JFNK/DIRK/IMEX (acoustic core, wave equation, implicit WCSPH experiments) the shared solvers, the device-side loop and the 5 syncs/iteration -> 0, and replaces the 586 torch ops per GMRES iteration. Caveats: our GMRES needs the right-preconditioning option JFNK uses; the JVP matvec (forward-mode dual tensors) is not graph-capturable, so graph mode is for the FD matvec or operator-based JFNK only; the exponential integrators' Arnoldi (`krylov_phi`) is a matrix function, not a solve, and would need its own entry point.

## 9. Progress

**Step 1 (2026-10-14): the stationary driver and the five relaxed-Jacobi loops.** `warpSPHCore.linalg.stationary` (`runStationary`, `CheckSchedule`, `residualStatistic`; `inclusive` for the `<=` tests) owns the loop skeleton: the scheme supplies `step(x, i) -> (xNext, stat, rNorm)`, the driver keeps the statistic, the convergence flag and an optional tracked vector on the device, reads the flag only at the schedule's checkpoints and returns the histories in one transfer. J1 (constant density), J2, J3 (divergence free, fixed and optimal), J4 (omni), J5 (`dfsphReference` / `iisph`) and J6 (`band2018pb`) run on it; `modules/incompressible/convergence.py` keeps the old names as thin delegates. `adaptive` is now the default for all of them (decision 3): J1, J5 and J6 read `pressureSolver` / `divergenceFreeSolver.convergenceCheckSchedule`, whose config default is `adaptive`.

Verification: pre-change states of all five incompressible schemes on every case they run (19-20 each, 6 steps, torch seeded; old tree vs old tree is bitwise on the periodic cases that used to differ by ~100 %), new tree compared with `convergenceCheckSchedule=every` on the solvers whose old loop read the host each iteration (J1, J5, J6; J2-J4 compare at their old default): **`divergenceFree`, `dfsphReference`, `iisph`, `omniIncompressible`, `band2018pb` all bitwise identical on every case**; incompressible tests 78 passed.

Integrator syncs per step on TGV (1,600 particles, 2D periodic, float32, `audit_case.py`, default `adaptive`): `divergenceFree` 31 -> 21, `dfsphReference` 310 -> 21, `omniIncompressible` 16 -> 16 (already scheduled). What remains in `dfsphReference` is about 11 convergence checkpoints per step (the adaptive marks of two solves) plus 5 per-solve setup reads (`hasFluid`, the fluid index); a lagged flag read (the core CG's pattern) would remove the checkpoints.

Not done in step 1: the loops still call torch / warp operator closures and apply their update and projection as torch ops (no fused warp kernels yet), the Krylov family (S1-S6) is untouched, and `Restricted` / operators / preconditioners do not exist yet.

**Step 2a (2026-10-14): `linalg.krylov` and CG.** `warpSPHCore.linalg.krylov` has `Status` (the legacy codes as an `IntEnum`), `SolveResult` (device `x`, `status`, residual-norm history; `.legacy()` gives the old `(x, int status, list)`) and `cg`, device-predicated: the breakdown / convergence / threshold tests are evaluated on the device, the iterate is frozen from the iteration a test fires, the host reads one flag per `checkEvery` (default 8) iterations and the status once at the end. Bitwise the iterate, status and history of the host-branching loop (kept as `tests/linalg/_legacyKrylov.py`, the reference of 30 core tests: SPD in float32 / float64 at several check intervals, callable preconditioner, warm start, indefinite -> -16 with the stamped residual, budget, threshold, zero right-hand side, already converged, non-finite operator, and a host-read count). `shifting.cgSolve` is a thin wrapper. Synthetic 65k-unknown system, 30 iterations: **153 -> 6 host syncs per solve**; ops per iteration 55 -> 67 and wall 156 -> 184 us / iteration at this launch-bound size (the price of the where-freeze; the device-side loop is what makes graph capture and a deep GPU queue possible, section 8).

Where the rest of the family differs from CG (found while reading them for step 2b): **BiCGStab** verifies a would-be converged recurrence residual with an extra matvec *only when the test fires* and keeps iterating if the recurrence lied, so a device formulation cannot be bitwise without computing that matvec every iteration; the plan is to stop on the recurrence residual on the device, verify once after the loop and restart from the iterate if it was wrong (a documented change of the stopping iterate, residual-quality tests instead of bitwise). **BiCG** forms `beta` in Python doubles (`float(rho/rho_prev) * ...`), so the device version needs float64 scalars to stay bitwise. **GMRES** is naturally cycle-structured (device flags per cycle, the least squares moves to Givens rotations), **MINRES** keeps an `n x k` basis and a growing triangular solve (the 3-term recurrence with Givens needs neither), **Richardson** has host-side step-size backtracking.

**Step 2b (2026-10-14): BiCG and BiCGStab.** `linalg.krylov.bicg` and `bicgstab`, device-predicated like CG; `shifting.bicgSolve` / `bicgstabSolve` are thin wrappers.
* **BiCGStab**: the tests run on the recurrence residuals (full step and half step), the verified true residual comes once after the loop, and a solve that reported convergence but fails it restarts from the iterate (up to 2 times; fresh shadow residual). Bitwise the old loop wherever the recurrence was honest (unpreconditioned float32 / float64, 24 parametrisations x cpu / cuda); preconditioned float32 may converge one step earlier because the old half-step candidate used the raw direction `p` instead of `M p` (a bug for a preconditioned solve). A sweep over ill-conditioned float32 systems checks that a converged status always means the verified residual passed.
* **BiCG**: the loop it replaces was not BiCG. Three defects found while porting: `beta` carried BiCGStab's `alpha / alpha_prev` factor, it was lagged by one iteration (`rho_k / rho_{k-1}` instead of `rho_{k+1} / rho_k`), and the adjoint applied the preconditioner on the wrong side (`M^-1 A^T q` for `(M^-1 A)^T q = A^T M^-1 q`). On a 200-unknown SPD system it needed 14 iterations where CG needs 9. The core version is textbook BiCG and coincides with CG iteration for iteration for a symmetric operator (tested in float32 and float64), so the pressure solver's `bicg` option now means what it says. Consequence: results of `PressureSolverType.bicg` change (it converges faster).
* Host syncs per 30-iteration solve (synthetic system): CG 153 -> 6, BiCG 302 -> 6, BiCGStab 153 -> 8. Ops per iteration rise (CG 55 -> 67, BiCG 95 -> 129, BiCGStab 73 -> 112; wall per iteration at this launch-bound 65k-unknown size 156 -> 180, 334 -> 464, 265 -> 411 us): the where-freeze costs elementwise ops, which the fused warp update kernels of the pure-warp step remove.
* `scalar_t` / float64: the warp kernels (`pinvSymmetric`, `symmetricPinv3`) follow the process-wide `scalar_t` (`warpSPHCore_PRECISION=float64`: the core linalg suite, 49 tests, passes in float64); the torch-level solvers follow the tensor dtype (all Krylov tests run float32 and float64). A float64 recurrence around a float32 operator (`krylovFp64`) cannot be a `scalar_t` kernel: the pure-warp solvers will use dtype-generic arrays like `warp.optim.linear`.

Remaining in step 2: GMRES (cycle-structured: device flags per cycle, Givens rotations instead of the normal equations, the in-cycle `hNew < atol` exit with its least-squares update and true-residual check, stagnation across cycles, right preconditioning for JFNK), MINRES (3-term recurrence, no stored basis), Richardson (host-side step-size backtracking). Each is a redesign of its exits rather than a port; same method as BiCGStab (device predicates + a verified result), own tests.
