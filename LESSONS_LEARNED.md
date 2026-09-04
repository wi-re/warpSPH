# warpSPH — Lessons Learned (cleanup sweep, 2026-08)

Reusable, still-relevant lessons from the pre-AD cleanup sweep tracked in
`docs/historic_plans/CLEANUP_PLAN.md`. That file is a status tracker now; this file is the "why" and
"watch out for" that's worth keeping around. Not a chronological log — if something
here is superseded by a later fix, it's been removed rather than marked stale.

For the AD/gradcheck bug-class catalog specifically (nine bug classes, with fix
recipes, found rolling gradcheck out across 25 kernel files), see
**`.claude/skills/gradcheck/SKILL.md`** — not duplicated here. For the procedure
of porting an example onto the `Case`/runner style, and for taking a case to
2D/3D, see **`PORTING_EXAMPLES.md`** — this file keeps the general lessons, that
one keeps the recipe.

## AD-readiness / autograd bridge

- **Scalar (non-array) kernel arguments cannot carry a gradient through
  warpSPHCore's autograd bridge at all**, regardless of whether the caller detaches
  them first. `warpWrapper2` casts every scalar `additionalArguments` entry through
  `scalar_t(...)` before its tensor/non-tensor split runs, and a kernel declaring
  `x: scalar_t` by value (not `wp.array(dtype=scalar_t)`) severs the tangent right
  there — "stop calling `.item()`" on the Python side is a no-op fix, since the
  severing point is the kernel signature, one layer up. The working pattern (added to
  warpSPHCore, proven on `computeCompSPHBalanceTermWarp`'s `dt`): declare the
  parameter as `wp.array(dtype=scalar_t)`, read `param[0]` once inside the kernel,
  wrap the Python-side call in `asScalarArg`. Check whether a target kernel receives
  a value as a plain scalar type before assuming a differentiability fix is just
  "stop detaching."
- **Adaptive `dt` genuinely carries a tangent back to state** (via sound speed →
  density) whenever `config.adaptiveDt` is set. This was the one confirmed
  tangent-dropping bug found across ~200 `.detach()`/`.item()`/`.cpu()`/`.numpy()`
  call sites, in two full audit passes. Most such calls are genuinely fine — loop-
  control scalars in fixed-point solvers, print/reporting boundaries, dead code whose
  only consumer is commented out. Don't assume a bare `.item()` sighting is a bug;
  trace it to its actual consumer before flagging it.
- **`x.clone()` followed by `x_.requires_grad = True` crashes if the caller's `x`
  already required grad** — `clone()` of a tensor that requires grad is a non-leaf,
  and PyTorch refuses to set `requires_grad` on a non-leaf. Guard with `if not
  x_.requires_grad: x_.requires_grad = True`. Hit in `geometry/sdf.py` and
  `regions/domainSDF.py`; was latent until a caller actually passes a
  `requires_grad=True` position tensor, which is exactly what forward-mode AD needs
  to do.
- **A bare `ParticleState` (warpSPHCore's own minimal fixture) is not a safe
  stand-in for this repo's scheme-level state objects.** Several kernels here read
  `internalEnergies`/`pressures`/velocity-derived fields that nothing validates are
  present, so a bare fixture can reach a kernel launch with `None` for one of them
  and segfault rather than raise. `scripts/_gradcheck_common.py`'s
  `make_compressible_state`/`compute_crk_state` exist because of this — use them (or
  an equally real state object) instead of the bare fixture for new gradcheck
  coverage of this repo's scheme layer.

- **A minimal reproduction can be the bug.** The above cost weeks in the worst
  possible shape: `computeCompSPHBalanceTermWarp` segfaulted in a hand-built
  minimal case, and the investigation
  (`scripts/troubleshoot_balanceTerm_segfault.py`) eliminated the arithmetic, the
  `wp.static` dispatch, the CSR shapes, the domain size, the dimension, the warp
  version and the adjacency construction — every one of them a property of the
  *product*. It concluded the most likely lead was an out-of-bounds read that a
  cold process exposed and a warm one masked. It was none of that. The harness
  had never built its reference states properly, so the "minimal case" was not a
  stripped-down valid input but an invalid one, and the real defect was an
  `Optional` argument left as `None` (below). **When a repro is the only thing
  that crashes, suspect the repro before concluding the product hides a memory
  bug** — and be especially suspicious when eliminating product properties keeps
  not moving the outcome, which is the signature of the input being wrong.

- **An `Optional` `reference*` argument must fall back to its `query*`
  counterpart, not stay `None`.** The defect behind that segfault: when
  `referenceVolumes` was not passed and the reference state carried no volume
  member, it stayed `None` and reached the kernel as a null array. Several entry
  points shared it. Fixed in `warpSPHCore` (`120c4bf`, 2026-08-06) with the
  one-liner now at `operations.py:52`, alongside making the state path primary.
  This repo's `modules/compSPH/balance.py` already does the same for
  `referenceVelocities`/`referenceEnergies`/`referencePressures` — when adding a
  new paired `query*`/`reference*` argument, add the fallback in the same commit;
  the failure mode is a segfault, not a `TypeError`.

- **The damage from a severed scalar kernel argument compounds with BPTT depth,
  so short-trajectory verification hides it.** `balance.py`'s `gamma` (the
  by-value case above, next to the already-fixed `dt`) was worth 0.003% of
  `dL/dgamma` after 5 integrator steps and 2.35% after 200 — each step's missing
  contribution is also absent from every earlier step's adjoint, so the error
  grows roughly as the square of the trajectory length. Two optimization runs
  with and without it are literally indistinguishable at 5 steps. Measured in
  `examples/compressible/01-sod/sod_backprop.ipynb`; don't conclude a partial
  gradient is harmless from a short rollout.
- **A directional central difference is the practical gradcheck for a whole
  trajectory.** `torch.autograd.gradcheck` needs one full BPTT rollout per
  component and is hopeless here, but two forward rollouts along a single
  direction resolve a systematically wrong gradient just as well — that is what
  distinguished "the fix changed the gradient" from "the fix made the gradient
  correct" for the case above (fixed: ~0.01% off FD; severed: a consistent
  2.33-2.36% off). Keep `eps` at or above ~1e-3 of the parameter scale; below
  that, float32 cancellation in the difference makes agreement *worse*.

## Dimensionality

Nothing in this repo ran in 3D until `sod3d` (2026-08-12). Adding it found two
bugs in an afternoon, both of the same shape: a value that depends on the
dimension, computed or cached once, by code that only ever saw one.

- **A kernel's normalisation constant is per-dimension, and a wrong one is
  invisible in the wave structure.** `B7_C_d`'s 3D constant was 16x too small,
  so every 3D density came back at 1/16 of the mass it was built from — while
  velocities, pressures-as-a-fraction and shock/contact positions all still
  looked *right*, because a uniform factor on density largely cancels in the
  dynamics. Check a kernel by summing it over a uniform lattice of known
  density (`sum_j m_j W_ij` must return the density you built) rather than by
  looking at a result and judging whether it seems plausible. Doing that across
  all nine kernels and all three dimensions is ~20 lines and caught this in one
  run.
- **A cache keyed on nothing is a bug the moment a second variant exists.**
  Owen's psi LUT (`modules/adaptiveSupport/optimalSupportOwen.py`) is sliced by
  dimension, built into a module-level global on first use, and reused for
  every call after — so in a process that touched two dimensions, the second
  one relaxed its supports against the first one's table. No exception, no
  obviously wrong number, and invisible to every existing test because each ran
  one dimension per process. When caching anything derived from config, key it
  on the parts of the config it was derived from, even if today only one value
  is ever used.

## Testing / coverage gaps

- **Per-kernel gradcheck only covers a kernel's *declared* differentiable
  inputs.** `gradcheck_compSPH.py` had exercised `computeCompSPHBalanceTermWarp`
  for six `EnergyScheme` values without ever noticing its `gamma` argument was
  severed, because `gamma` was passed as a plain float and so was not one of the
  gradcheck inputs. A parameter that a call site happens to pass as a constant is
  exactly the one nothing checks; when adding a differentiable-scalar path, add
  the gradcheck case for it too (`run_balance_gamma_gradcheck` is the pattern,
  including asserting the gradient is zero on the branches that should not read
  it).
- **A scheme or branch only reachable via a flag rots unnoticed if no test passes
  that flag.** Monaghan (`--scheme Monaghan`) had two independent breaking signature
  drifts — a missing `t` parameter on three boundary-condition helpers, a removed
  `supportScheme=` keyword — invisible because every default case runs CRKSPH or
  CompSPH. Fixed by adding a Sod-under-all-three-solvers test. When a code path is
  flag-gated, the coverage gap *is* the flag; add a test that passes it, not just a
  test of the default path.
- **Runtime-only import checking misses notebook/function-level imports.** A plain
  `python -c "import warpSPH"` never executes a notebook cell's own `from x import
  y`. `scripts/check_imports.py` needed a second pass — an AST scan of every `.py`
  file and notebook code cell, checking module *and* symbol — to catch what the
  runtime pass alone couldn't. Don't treat a clean top-level import as sufficient
  verification in a repo with notebooks.
- **A planning doc can lag reality in both directions.** Two items were already
  fixed by an earlier commit while still listed open; separately, three gradcheck
  scripts existed, worked, and were committed but were never wired into the test
  suite or written up. Before trusting a status claim in a planning doc, diff it
  against the actual code and test list rather than re-deriving or re-trusting the
  prose.

## Python/Warp gotchas

- **A keyword argument in a call doesn't rebind the caller's local of the same
  name.** `someWarpCall(..., c_max = c_max.cpu().item(), ...)` reads as if it
  detaches the outer `c_max`, but it only binds the callee's parameter — the
  caller's own `c_max` local is untouched, and any later use of it in the same scope
  still carries a gradient. Verify by variable *identity*, not name, whenever a
  `.item()`/`.detach()` shows up inside a call's keyword arguments.
- **The installed warp-lang version can drift silently.** It moved three times
  across this sweep (1.12.0 → a local 1.17.0.dev3 dev checkout → 1.15.0 from PyPI)
  while `pyproject.toml` deliberately pinned nothing, waiting for the 1.17 stable
  release that fixes the same-array-ternary/`Interpolate` adjoint bug at the
  source (see `docs/historic_plans/CLEANUP_PLAN.md`). A script that passed last session could fail
  this one with zero code changes on either side — check `python -c "import warp;
  print(warp.__version__)"` against what a script last passed under before assuming
  a new failure is a real code regression. **Resolved 09-04**: 1.17.0 shipped,
  `pyproject.toml` now pins `warp-lang>=1.17.0`, so this specific drift mode is
  closed — the general lesson (check the installed version before trusting a
  failure) still applies to any future unpinned dependency.
- **A kernel argument can be computed, threaded through several call layers, and
  never actually read inside the kernel.** `shifting/delta.py` computes `c_max`/`dx`
  and passes them into `computeDeltaShiftWarp`'s scalar args; the kernel uses them to
  compute a `shiftScaling` value it then never applies (the next line, `out +=
  shiftAmount`, ignores it). Harmless, but worth a grep for the parameter name
  *inside* the kernel body — not just at the call site — before trusting that a
  value passed to a kernel actually matters.

## Process / cross-repo

- **Cross-repo fixes belong in the repo that owns the bug, made once, on purpose —
  not worked around locally as a side effect of an unrelated task.**
  `correctGradientCRK`'s wrong-axis contraction, the autograd bridge's
  `requires_grad` dtype gate, `access_optional`, and `asScalarArg` were all found
  while working in this repo but fixed in `warpSPHCore`. `volumeToSupport`'s
  scalar-only limitation is the same shape and now resolved end to end: `warpSPH/
  utils/support.py` carried a tensor-aware local wrapper until core grew the same
  `isinstance(volume, torch.Tensor)` dispatch internally (2026-08-12), at which point
  the wrapper collapsed to a plain re-export — collapsing it *before* the upstream
  fix landed would have silently reintroduced the 2D/3D Monaghan crash the wrapper
  existed to prevent. General rule: a local wrapper papering over an upstream gap is
  a marker of unfinished work, not dead weight — don't remove it until the upstream
  fix is actually in place, and check the marker's precondition, not just its
  presence, before removing it.
- **Measure before acting on an inherited assumption.** The original repo-weight
  plan assumed ~380 MB of repeatedly-recommitted media; measuring first found the
  `.mp4`/`.gif` files were each committed exactly once (nothing to reclaim there) and
  the real bloat was 312 blob versions of 42 notebooks — a different fix entirely
  (`nbstripout`, not LFS). Re-measure a stale plan's numbers before executing its
  prescribed fix.
- **A rename/restructure that touches star-imported names can create an invisible
  shadowing trap.** The pre-`SchemeBundle` 7-tuple unpack bound a scheme's config
  class to the name `SimulationConfig` in 33 notebooks — the same name `from
  warpSPH import *` also provides for the *global* simulation config — so
  `schemeConfig = SimulationConfig()` was silently reading the shadowed local
  instead of the intended global class. When flattening a star-import surface or
  introducing a named-bundle API, grep the outgoing names against the incoming ones,
  not just for import breakage.
