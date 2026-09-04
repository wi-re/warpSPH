# Sod differentiability tests: three backprop-through-trajectory cases

Status: **done**, drafted and landed 2026-08-12. Deliverable is
`sod_backprop.ipynb` in this directory. This file is the plan/design record,
kept for reference — see "Results" at the bottom for what the runs actually
showed, including where the plan's own expectations were wrong.

## Context

This is the backprop-over-trajectory demo flagged as deferred work when
`01-sod/` was built (see `docs/historic_plans/CLEANUP_PLAN.md`'s "Notebook simplification,
pilot" entry and the hook-point comments already sitting in
`sod_1d.ipynb`'s step loop). The per-kernel AD work this whole cleanup sweep
has been doing (`.claude/skills/gradcheck/SKILL.md`, 58 gradcheck tests)
verifies individual kernels' adjoints against finite differences; nothing in
the repo has yet verified that backprop actually works **chained across many
integrator steps** for a real case. That's what these three cases are for,
and they're the first thing to actually exercise it.

Per the user: these live as **notebooks** in `examples/compressible/01-sod/`
(interactive setup/visualization is the point), not wired into CI (500-step
BPTT is too expensive to run on every commit) — but "eventually useful as a
test," so the underlying logic should be written as clean, reusable
functions a future `scripts/gradcheck_sod_trajectory.py` could import
directly, not one-off notebook spaghetti.

## A real bug found while checking this is actually possible: `gamma`'s gradient is severed

Read directly (`src/warpSPH/modules/compSPH/balance.py`): `gamma` is CompSPH's
**only** kernel-facing use of the EOS constant (confirmed by grepping the
whole compSPH/shockCapturing/adaptiveSupport/timestep modules — `idealGasEOS`
itself is plain PyTorch and already differentiates fine). At the
`computeCompSPHBalanceTermWarp` call site (line 388):

```python
wp.int32(energyScheme.value), asScalarArg(dt, device=device), scalar_t(gamma)
```

`dt`, two arguments earlier in the *same call*, is correctly wrapped in
`asScalarArg` (the established fix from this sweep's Tier 2 work — see
`warpSPHCore/autograd/scalar_arg.py`'s docstring). `gamma` is not — it's cast
through `scalar_t(...)`, which per the same docstring "collapses a tensor to
a Python float," severing the graph right there. The kernel-side signature
(`computeCompSPHBalanceTerm_Kernel`, line 267) declares `gamma: scalar_t`
(by value) instead of `wp.array(dtype=scalar_t)` like `dt` does. This isn't
hypothetical: `gamma` is read inside the kernel's `EnergyScheme.CRK` branch
(line 138, `s_i = P_i / wp.pow(rhoi, gamma)`) — Sod's own default scheme —
feeding `f_ij`, the energy-partition weight `dudt` depends on downstream.
Case 2 (optimize gamma) would silently get a wrong-but-plausible gradient
without this fixed — not a crash, so `nbconvert`-only verification would
never catch it.

**Fix, mirroring `dt`'s pattern in the same file exactly:**
- `computeCompSPHBalanceTerm_Kernel` (line 267): `gamma: scalar_t` →
  `gamma: wp.array(dtype = scalar_t)`.
- Its body (line 287): pass `gamma[0]` down to `..._Func_Adjacency` instead
  of `gamma` (which keeps its plain `scalar_t` signature at the
  `_Func_Adjacency`/`_Func_i` layers, unchanged — only the outer `@wp.kernel`
  entry point needs the array form, exactly like `dt`).
- Call site (line 388): `scalar_t(gamma)` → `asScalarArg(gamma, device=device)`.
- `computeCompSPHBalanceTermWarp`'s own signature (line 300): annotate
  `gamma: Union[float, torch.Tensor]`, matching `dt`'s annotation just above it.

**Per the user: don't apply this fix before running case 2 — run case 2
*twice* and compare.** `gamma` still influences the trajectory through
`idealGasEOS`'s pressure/soundspeed path (plain PyTorch, already
differentiable) even with the kernel-arg bug in place — the bug only drops
the balance kernel's own entropy-based energy-partition contribution to
`gamma`'s gradient, it doesn't zero the gradient outright. That makes "does
it converge, and how fast, with the partial gradient" a genuinely
interesting empirical question, not just a bug to silently paper over:
1. Implement case 2 first, run it as-is (partial gradient, `balance.py`
   unfixed). Document the loss/parameter-error curve — does it converge at
   all, and if so, how does its rate compare to case 1 (masses, which has
   no known gradient gaps)?
2. Then apply the `balance.py` fix above.
3. Re-run case 2 unchanged otherwise (same seed/init/lr/iteration count).
4. Plot both convergence curves together and note the difference in the
   notebook — this is a real, useful data point about how much a missing
   kernel-argument gradient actually costs in practice for this case, not
   just a pass/fail check.

## Design decisions that make all three cases tractable

- **Fixed, non-adaptive `dt`** (`spec.adaptiveDt=False`, one explicit `dt`)
  for every run in this notebook. Adaptive `dt` is CFL-derived from sound
  speed, which depends on `gamma` (case 2) — if `dt` could change with the
  parameter being optimized, "N steps" would land at a different simulated
  time for every trial, making the target comparison ill-posed. This isn't
  a workaround, it's the only way to pose the experiment cleanly.
- **The IC's own `supports`/`densities` snapshot doesn't need to be
  differentiable, and neither does `evaluateOptimalSupport`'s fixed-point
  solver.** `compSPH_step` calls `evaluateOptimalSupport` itself as the
  *first thing it does every step* (`schemes/compSPH.py:42`) and immediately
  recomputes `densities` from current `masses`/`positions`/`supports` right
  after (line 63). Whatever `supports`/`densities` an IC starts with gets
  overwritten by real physics on step 1 regardless of how it was produced.
  So: build the reference IC once with the normal (non-differentiable)
  `buildSod1D` path, reuse its `positions`/`supports` as fixed/detached
  starting points in every case below, and only make the fields that
  actually carry the experiment's signal (`masses`, `internalEnergies`,
  `gamma`) into `requires_grad` leaves. This sidesteps needing to verify
  `evaluateOptimalSupportOwen`'s iterative solver is itself differentiable
  (unverified territory, and a dynamic-trip-count loop is exactly the shape
  `.claude/skills/gradcheck/SKILL.md`'s AD-bug catalog warns about) —
  case 3, which the user already flagged as "more involved," turns out not
  to need it at all once this is recognized.
- **Small `nx`/`N` by default** (e.g. `nx=100`, `N=5`) for interactive
  iteration; a commented-out `nx=800, N=500` block for the "real" run,
  consistent with the user's own "5 for debugging, 500 at the end" framing.
  At these particle counts (~125 for `nx=100`), retaining 500 steps of
  autograd graph is not a real memory concern on the dev GPU — no gradient
  checkpointing or truncated BPTT needed.

## `examples/compressible/01-sod/sod_backprop.ipynb` (new)

Shared setup (reused by all three cases, written as plain functions so a
future test script can import the logic directly):

- `buildReferenceRun(spec)` — `sodCase.buildSystem`, step `N` times with a
  fixed `dt`, return `(referenceIC, targetState)` with `targetState`'s
  tensors `.detach()`ed (it's a fixed target, never backpropagated into).
  Same `ctx.integrator.function(...)` call the other two notebooks already
  use, no new stepping code.
- `stepN(system, ctx, n)` — run `n` steps from `system.initializeNewState()`,
  return the final `state`. The one place a perturbation/gradient path gets
  threaded through; everything else is unchanged case machinery.
- `stateLoss(a, b)` — sum of per-field MSE over `positions`, `velocities`,
  `internalEnergies`, `densities` (the independent physical fields;
  `pressures`/`soundspeeds`/`entropies` are pure functions of these so
  including them adds no new signal).
- An Adam loop wrapper printing loss + recovered-vs-true parameter error
  per iteration, and a final plot of both curves — shared by cases 1 and 2.

**Case 1 — recover perturbed masses.** Clone the reference IC's `masses`,
add noise scaled to the mean mass, clamp to a small positive floor (never
literally zero — matches "make sure to not create negative masses").
`requires_grad_(True)`. Rebuild the state via `dataclasses.replace(state,
masses=noisyMasses)` (works generically on `CompSPHState`, a plain
`@dataclass`) reusing the reference's `positions`/`supports`/everything
else. Adam over `[noisyMasses]`; each iteration: rebuild the system from the
current `noisyMasses`, `stepN`, `stateLoss` vs. the detached target,
`.backward()`, `optimizer.step()`, then `noisyMasses.clamp_(min=floor)`
under `no_grad` (Adam has no built-in positivity constraint). Track
`||noisyMasses - trueMasses||` alongside the loss.

**Case 2 — recover `gamma`, run twice (before/after the `balance.py`
fix).** `gammaGuess = torch.tensor(2.0, requires_grad=True)` (true value is
Sod's default `5/3`). Reuse the reference IC completely unchanged (same
`internalEnergies` — those stay bound to the *true* physical IC; only
`schemeConfig.gamma` is wrong) — set `schemeConfig.gamma = gammaGuess`
each iteration before stepping, since `compSPH_step` recomputes
`pressures`/`soundspeeds`/`entropies` from `internalEnergies` + `gamma`
every step (`schemes/compSPH.py:108-114`), which is exactly where the
trajectory starts diverging from the target. Adam over `[gammaGuess]`;
clamp to `>1.01` after each step (unphysical/NaN-producing below that).
Written once as a function taking the same fixed seed/init/lr/iteration
count, called twice (pre-fix, post-fix) so the two runs are otherwise
identical — the `balance.py` fix is the only thing that changes between
them.

**Case 3 — fit IC to the analytic Riemann solution.** Optimize
`params = torch.tensor([left_rho, left_pressure, right_rho, right_pressure],
requires_grad=True)` (Sod's own case params, initialized off-true to give
the optimizer something to do). Each iteration: derive `masses`
differentiably from `params` the same way `buildSod1D` does (mass ∝ ρ),
derive `internalEnergies` via `idealGasEOS` from `params`' pressures/
densities — both plain differentiable tensor ops, no solver involved;
reuse the reference's fixed `positions`/`supports` per the design note
above. Step `N` times to reach `t = N*dt`. Call `sodSolution.solve(...)`
(already imported by `plotSod_`, `caseUtils/compressible/sod/sodSolution.py`)
once at that `t` to get the analytic `x`/`rho`/`u`/`p` arrays; `numpy.interp`
them onto the SPH particles' (fixed, non-optimized) x-positions once — the
analytic solution is a fixed target, never part of the backward graph, so a
plain numpy interpolation is fine, no need for a torch-differentiable
interpolator. Loss is `stateLoss`-style MSE between the SPH trajectory's
final `densities`/`velocities`/`internalEnergies` and the interpolated
analytic values. Track recovered vs. true `[left_rho, left_pressure,
right_rho, right_pressure]`.

Notebook structure: one markdown section per case (what's being tested, why
it's expected to work), shared setup cells first, then each case as its own
self-contained cell block (params → build → optimize loop → plot), matching
the visible/hookable/editable style already established in `sod_1d.ipynb`.
Case 2's section runs its optimize-loop function twice (pre-fix, then a
markdown note + the `balance.py` fix + post-fix) with a combined plot of
both convergence curves at the end, rather than one run silently using
whichever version of `balance.py` happens to be on disk.

## Verification

**Not-crashing is not sufficient evidence** — the `gamma` bug above is
exactly the kind of thing that produces a plausible-looking, silently wrong
gradient rather than an exception. Verification has to check actual
convergence, not just clean execution:
- `nbconvert --execute` on a small-`nx`/`N=5` copy for the headless
  "doesn't crash" check (matching this session's established pattern for
  the other two notebooks).
- For each of the three cases, actually run a small-but-real optimization
  (e.g. `nx=100`, `N=5-20`, a few dozen Adam steps) myself and confirm the
  recovered parameter's relative error decreases over iterations and lands
  within a reasonable tolerance of the true value — this is the only way to
  catch a case where gradients flow but are wrong. If any case fails to
  converge, that's a real finding to report, not something to paper over.
- For case 2 specifically: run and report on *both* the pre-fix and
  post-fix convergence behavior (does the partial gradient converge at all,
  slower, to a biased value, or not noticeably differently? — genuinely
  don't know the answer yet, that's the point), not just confirm the
  post-fix version eventually gets there.
- Full test suite (`scripts/run_tests.sh`) after the `balance.py` fix, to
  confirm the `gamma`-as-array kernel signature change doesn't regress
  `gradcheck_compSPH.py`'s existing (now correctly-gradient-checked)
  coverage or any other CompSPH case.
- `scripts/run_sweep.py --cases sod` unaffected (doesn't touch
  `storeMode`/`extraFields`/anything this plan changes).

## Results (2026-08-12)

All three cases converge; backprop chains correctly through the CompSPH
integrator for at least 200 steps. Two things came out differently from what
the plan above assumed, both recorded in the notebook itself.

**The `gamma` bug was real, and the fix was applied as specified** — but the
pre/post-fix comparison the plan designed had to be restructured, because a
notebook cannot reload a changed `@wp.kernel` signature mid-session. Instead
the pre-fix behaviour is re-created *in* the fixed source by detaching `gamma`
at the balance call site only (a `functools.wraps` patch of
`warpSPH.schemes.compSPH.computeCompSPHBalanceTermWarp`). That emulation was
validated against the real pre-fix source: case 2 was run for 60 iterations
against unfixed `balance.py`, then again after the fix with the emulation, and
the two agree **bit-for-bit** on loss, parameter error and gradient norm at
every iteration. So the notebook shows both curves live in one kernel session
and does not depend on which version of `balance.py` is on disk.

**What the missing gradient actually costs: nothing at short trajectories,
and it compounds.** Measured at `nx=100`, `gamma=2.0`, every run
bit-deterministic (three repeats agreed exactly):

| steps | severed vs. full | full vs. central FD | severed vs. FD |
|---|---|---|---|
| 5 | +0.003% | | |
| 20 | +0.044% | -0.009% | +0.035% |
| 50 | +0.226% | -0.010% | +0.216% |
| 100 | +0.673% | -0.009% | +0.663% |
| 200 | +2.350% | +0.010% | +2.359% |

At the 5-step size the two optimization runs are indistinguishable (`gamma`
recovered to 1.680127 vs. 1.680126 from a 2.0 start, true 5/3), so the plan's
"does it converge, and how fast" question resolves to "identically, at this
depth" — the bug would not have been caught by watching a short optimization.
The finite-difference columns are what make this a correctness claim rather
than a difference: the fixed gradient tracks FD to ~0.01% at every step size
tried, the severed one is off by a consistent 2.33-2.36% at 200 steps.

**Case 1 converges in loss much faster than in parameter error** (loss 7.9e-4
-> 3.2e-6 over 250 iterations, a factor of 250, while the relative mass error
only goes 4.8% -> 2.4%). That is the inverse problem's conditioning, not a
gradient defect: 125 free masses against a 4-field endpoint leaves a large
near-null subspace. Confirmed by a directional finite difference, which the
notebook keeps as a cell — the analytic mass gradient agrees with FD to
0.02-0.14%, and separately to ~0.01% at one step, at five steps, along a
random direction, and under `adaptiveSupportScheme='Monaghan'` as well as
Sod's default `'Owen'`. Worth noting because `masses` reach the physics partly
through `evaluateOptimalSupport`'s fixed-point solver, which the plan above
deliberately routed around rather than verifying.

**Case 3** recovers `[1.25, 0.80, 0.325, 0.135]` -> `[1.019, 1.009, 0.236,
0.170]` against a true `[1.0, 1.0, 0.25, 0.1795]` in 60 iterations. One
deviation from the plan: the analytic solution is interpolated onto the
particles' *final* (detached) positions rather than the IC's, since the
particles have moved by `t = N*dt`; it is still a fixed target outside the
backward graph, so `numpy.interp` is still all it needs.

**Scaling.** All the numbers above are at the notebook's interactive default
(`nx=100`, 5 steps). The user separately ran the gradient test at `nx=400` over
100 steps and reports it working, so nothing about the approach is tied to the
small size; the 200-step measurements in the table were made in the same
session at `nx=100`.

**Also landed:** `run_balance_gamma_gradcheck` in
`scripts/gradcheck_compSPH.py`, a regression guard for the fixed argument
(gradcheck of `gamma` itself under every `EnergyScheme`, plus an assertion
that the gradient is non-zero exactly under `CRK`). Verified to fail against
the pre-fix source and pass against the fixed one. Full suite: 57 passed.
