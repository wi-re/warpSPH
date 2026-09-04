---
name: gradcheck
description: Run this repo's torch.autograd.gradcheck regression scripts against its 25 custom warp-kernel modules (schemes/, modules/{compSPH,crk,dissipation,mdbc,shockCapturing,surfaceDetection,adaptiveSupport,deltaSPH,incompressible,liu,util}, geometry/, regions/, sample/) -- all of them at once, or a single one while iterating on that module's kernel. Use whenever touching a @wp.kernel/@wp.func in one of those directories, to catch silently-wrong gradients (adjoint-zeroing ternaries, loop-carried AD bugs, missing fallbacks, mixed-dtype multi-output bridge gaps) that forward-only physics tests (tests/test_physics.py) cannot see.
---

# Running the gradcheck scripts

15 scripts under `scripts/`, one (or a small family) per kernel-bearing module,
mirroring warpSPHCore's own methodology (`~/dev/warpSPHCore/.claude/skills/
gradcheck/SKILL.md`) but for this repo's scheme-specific physics layer, built
*on top of* warpSPHCore's already-gradchecked operators. Every one calls
`torch.autograd.gradcheck` directly against the module's real `compute*Warp`
entry point -- no manual Jacobian, no per-call workaround. docs/historic_plans/CLEANUP_PLAN.md's
Phase 4.1 has the full history; this file is the "how do I use/extend this"
reference, and the bug-class catalog below is the "what should I watch for"
one, distilled from every real bug the rollout found.

```
scripts/gradcheck_wp_surfaceAware.py
scripts/gradcheck_sdf.py
scripts/gradcheck_scalarArg_dt.py
scripts/gradcheck_compSPH.py
scripts/gradcheck_dissipation.py
scripts/gradcheck_crk.py
scripts/gradcheck_adaptiveSupport.py
scripts/gradcheck_deltaSPH.py
scripts/gradcheck_shockCapturing.py
scripts/gradcheck_mdbc.py
scripts/gradcheck_incompressible.py
scripts/gradcheck_liu.py
scripts/gradcheck_surfaceDetection.py
scripts/gradcheck_util.py
scripts/gradcheck_deltaShift.py
```

All of them set `warpSPHCore_PRECISION=float64` via `os.environ.setdefault`
*before* importing anything from `warpSPHCore`/`warpSPH` -- required, since
Warp bakes precision into every compiled kernel at first import and it can't
change mid-process. This repo has a **second, independent** reason a script
can't just call `warpSPHBootstrap.bootstrap()` and move on: `tests/conftest.py`
already calls `bootstrap(precision='float32')` at collection time, before any
test module is imported, so the main pytest process is locked to float32
before a gradcheck script would even get a chance to request float64. Both
reasons point the same way: gradcheck scripts run as **subprocesses**, each
with a fresh interpreter, never in-process.

## Run all of them (~1 min)

```bash
bash scripts/run_tests.sh          # full suite, gradcheck scripts included
pytest tests/test_gradcheck_scripts.py -v   # just these
```

Each script runs as its own subprocess, asserted to exit `0`
(`tests/test_gradcheck_scripts.py`). Use this as the default "did I break
gradients anywhere" check -- it's part of the same suite `run_sweep.py`/
`check_imports.py` sit alongside, per the README's verification section.

## Run one directly (fast iteration loop)

```bash
python scripts/gradcheck_crk.py
```

Swap in the module you're touching. Each prints one `PASSED`/`FAILED` block
per case (multiple scheme-enum values, correction-flag combinations, and
sometimes a dedicated regression guard -- see each script's own module
docstring for exactly what it covers and *why*; several document a real bug
that was found and fixed getting the script to pass in the first place, which
is worth reading before assuming a new failure is spurious).

## Adding a new module or a new case

Follow the existing pattern rather than writing one from scratch:

1. Import shared fixtures from `scripts/_gradcheck_common.py`: `make_domain`,
   `single_particle_case`, `line_case`, `grid_case_2d`, `build_adjacency`,
   `build_grid_adjacency`, `compute_densities` are vendored from warpSPHCore's
   own file of the same name (kept behaviorally identical on purpose -- don't
   add repo-specific logic to them). `make_compressible_state` and
   `compute_crk_state` are repo-local additions, for modules that need a real
   per-scheme state object (`.velocities`/`.internalEnergies`/`.pressures`/
   `.soundspeeds`/`.alphas`) rather than a bare `ParticleState` -- check first
   whether an existing helper covers what you need before writing a new one.
2. Set `warpSPHCore_PRECISION=float64` the same way every existing script
   does, before any `warpSPH`/`warpSPHCore` import.
3. Call `torch.autograd.gradcheck(f, inputs)` directly against the module's
   public `compute*Warp` (or equivalent) entry point, `eps=1e-6, atol=1e-5`,
   at every differentiable-flag combination the module exposes (grad-h on/off,
   CRK on/off, `individual_cs`/`explicitPressure` on/off, ...) -- this exact
   surface is what caught several silently-wrong-argument and missing-fallback
   bugs by hand before gradcheck coverage existed at all.
4. Register the new script's filename in `GRADCHECK_SCRIPTS` in
   `tests/test_gradcheck_scripts.py` so it's picked up automatically.
5. **If the module has a `if useAdjacency: ... else: checkOffset(...)`
   branch, test both traversal modes** -- `build_adjacency` for the
   `AdjacencyList` path every existing script exercises, and
   `build_grid_adjacency` for the grid/`CompactHashMap` path most don't. See
   bug class 5 below: a fix verified only against `AdjacencyList` can leave a
   second, unrelated bug live in the untested branch. A pure forward-value
   agreement check between the two modes (no gradients) is enough to catch
   this class -- see `gradcheck_shockCapturing.py`'s `_run_grid_consistency`
   for the pattern.

## Bug classes found so far, and their fix recipes

Nine real bug classes turned up rolling gradcheck out across all 25 files.
Roughly in the order you're likely to hit them again:

### 1. Same-array-vs-default ternary silently zeroes an adjoint

`X[j] if cond else default` (or `X[j] if cond else Y[j]`, or even a
non-array-read ternary) compiled inside a `@wp.func`/`@wp.kernel` can compile
fine, run the right branch at runtime, and still produce a **zero adjoint**
for the array read -- confirmed on the currently-installed warp-lang (1.15.0)
and broader than first suspected: not just same-array reads, *any* inline
`X if cond else Y` is suspect on this version. Fix: warpSPHCore's
`access_optional(arr, index, condition, defaultValue)` (`util/stateUtil.py`)
-- a real `@wp.func` with an explicit `if`/`else` block, not a compiled
ternary. Use it (or an explicit `if`/`else` for non-array-read cases) instead
of an inline conditional expression in any kernel code, full stop, until
warp-lang 1.17 ships and this is fixed at the source (see "Decisions already
made" in docs/historic_plans/CLEANUP_PLAN.md -- don't propose pinning to 1.15.0/1.16.0/1.12.0 as
a fix, that's an intentional wait, not an oversight). If a gradcheck starts
failing right after adding or touching a ternary in kernel code, check this
first.

### 2. Missing query→reference fallback reads stray memory

Several modules had `referenceVelocities = queryVelocities` as a fallback
when the caller doesn't supply `referenceParticles`, but were missing the
*equivalent* fallback for another field (`referenceCs`/`referenceAlphas`/
`referencePressures` never defaulting to their `query*` counterparts). A
caller that passes the query-only field explicitly -- exactly what a bare-
`ParticleState` gradcheck case does, and what several real call sites do too,
just via an attribute that happens to live on the same object -- falls
through to a **size-1 dummy tensor read out of bounds**, silently reading
stray memory (wrong *values*, not just wrong gradients). Found by gradcheck
because its fixtures are the first thing to actually exercise the query-only
path. Fix: add the missing fallback, mirroring whichever sibling field
already has one correctly.

### 3. Loop-carried nonlinear reduction *after* a dynamic loop, same scope

A **linear** accumulation over a dynamic (runtime, per-particle) neighbor
loop -- the safe "independently-computed-then-summed" pattern -- followed by
a **nonlinear reduction of the accumulated value** (squaring via `wp.dot`,
`wp.normalize`, ...) **in the same `@wp.func`/`@wp.kernel` scope as the
loop** silently zeroes that contribution's adjoint. `computeAlphaWarp`
(`modules/incompressible/wp_alpha.py`) is the worked example: `alpha =
areaI/mi * wp.dot(sumA, sumA) + areaI*sumB`, computed right after the loop
that built `sumA`, in the same function. **Fix**: split the function so it
*returns* the raw accumulator, and do the nonlinear reduction in the
*caller* instead (a different compiled scope) -- a pure textual relocation,
no math change. Confirmed with a from-scratch minimal Warp kernel with zero
SPH code (see `gradcheck_incompressible.py`'s docstring for the exact repro):
any number of nested dynamic loops is fine as long as the innermost one
whose accumulation directly feeds the nonlinear op finishes and *returns*
before that op reads it, in a different function.

### 4. Loop-carried nonlinear *reassignment* inside the loop, every iteration

A different, *not* directly fixable-by-relocation shape: `out = wp.max(out,
candidate)` reassigned nonlinearly on **every iteration** of the loop, not
just reduced once after it. `computeVsigWarp`
(`modules/shockCapturing/wp_vsig.py`) is the worked example -- this was
initially (wrongly) written up as needing an upstream fix, since the "move
the reduction to a different scope" recipe from class 3 doesn't apply here
(there's no single post-loop reduction to relocate). **Fix**: separate "which
candidate wins" from "what is that candidate's value". Do a forward-only
search that finds the winning *index* (its own gradient is never used, and
indices aren't differentiated by Warp anyway, so a wrong adjoint on this
search's discarded return value is harmless) via the same loop-carried
`wp.max` as before, then recompute the actual value for that *one
already-known* index, with no loop at all, in the caller -- an ordinary,
non-looped expression that Warp's automatic diff handles correctly like any
other per-neighbor SPH formula. See `computeVsig_Func_i_argmax`/
`computeVsig_valueAt` for the pattern. Use class 3's recipe when the
nonlinearity is a one-time reduction after the loop; use this one when it's a
repeated in-loop reassignment.

### 5. Missing compact-support filter: invisible on AdjacencyList, wrong on grid traversal

A neighbor loop with **no `w_ij > 0` (or equivalent radius) check at all**
-- every entry in the candidate list treated as a genuine neighbor
unconditionally, unlike every other module in the same family. Invisible when
`useAdjacency=True` (`radiusSearchCompactHashMap` already returns an exact,
pre-filtered list, so the missing filter is a no-op) but **wrong** for grid
traversal (`checkOffset` returns every particle in a nearby spatial *cell*,
coarser than the exact support radius) -- confirmed in `wp_vsig.py`: grid and
`AdjacencyList` results disagreed until a `computePairwiseSupport`-based
check was added, then matched exactly (and matched a from-scratch
brute-force reference). This is a forward-*value* bug, not an AD bug, but it
was found *because* fixing an AD bug in the same file prompted testing the
grid-traversal branch specifically -- worth checking any time you're already
in a module's neighbor loop for another reason. Fix: add the missing
`if r_ij >= computePairwiseSupport(hi, hj, kernelProperties.supportMode) and
i != j: continue`, matching the sibling modules that already have it.

A related, second-order version of this: an *outer* offset loop
(`for o in range(numOffsets)`) that **sums** each offset's own per-offset
reduction via `+=`, correct only when `numOffsets == 1`
(`AdjacencyList` mode) but silently wrong for grid traversal, where multiple
offsets (up to 27 in 3D) each contribute -- summing several maxes is a
different quantity than the max over their union. Fix: have the *outer*
function do the same compare-don't-accumulate argmax approach as the inner
one, and move the actual differentiable evaluation up to the primary caller
(the `@wp.kernel`), run once per particle rather than once per offset.

### 6. Self-interaction / absent-field division poisons the adjoint even when patched forward

A division that's `0/0` for the self-interaction pair (`i == j`, present in
every adjacency list) or when a field is legitimately absent (e.g. densities
defaulting to zero on a bare `ParticleState`) can be patched *forward* with a
value check (`if ri != ri: ri = 1.0`, i.e. a post-hoc NaN fixup) and still
poison the *backward* pass: reverse-mode AD differentiates the expression
that was actually evaluated, not the value it was replaced with, so the
singular local derivative (`1/0`) leaks into the adjoint regardless of the
later overwrite. Fix: guard the division itself with an `if`/`else` *before*
it happens, not after -- e.g. `modules/crk/limiter.py`'s `computeVanLeer`.

### 7. Multi-output kernels with mixed dtypes

A kernel launch with more than one output array, where at least one is a
non-float dtype (an `int32` neighbor-count array alongside a float
correction array) crashes the moment any input requires grad --
warpSPHCore's `launch_kernel` used to set `requires_grad` on *every* output
unconditionally, and `wp.to_torch` on the int32 one then rejects it. Fixed in
warpSPHCore via `_dtype_is_float` gating `requires_grad` on the output's own
dtype. If you add a new multi-output kernel with a mixed-dtype output, assert
the non-float output's `requires_grad` is `False` inside the gradcheck
closure itself (see `gradcheck_liu.py`'s and `gradcheck_util.py`'s
`nnbrs`/`countNeighborsWarp` assertions) as a standing regression check on
that fix, not just a one-time verification.

### 8. Unguarded float32 literal in a float64 kernel

A bare Python float literal (`+ 1e-12`) inside a `@wp.func`/`@wp.kernel`
infers as `wp.float32` regardless of the active precision -- a genuine
`float64 + float32` type mismatch under `warpSPHCore_PRECISION=float64`,
often the *first* time a given line is ever exercised at float64 (every
sibling `+ eps` site in the same file is usually already correct, making the
straggler easy to miss by eye). Fix: wrap it in `scalar_t(...)`.

### 9. A hand-written accumulation loop can just be wrong, not only hard-to-differentiate

Distinct from classes 3/4: a loop that manually contracts a tensor via
explicit `for row: for col: ...` accumulation can contract the *wrong axis*
and be wrong even forward, independent of any AD concern (`correctGradientCRK`
in warpSPHCore -- fixed by replacing the loop with a single `matmul`). Prefer
built-in linear-algebra ops (`matmul`, `wp.dot`, `wp.outer`) over hand-rolled
index loops for tensor contractions in new kernel code, both for correctness
and to sidestep this whole risk category.

## What's *not* in scope for gradcheck

Genuinely discrete/counting outputs -- a boolean surface-detection mask
(`wp_maronne.py`, `wp_barecasco.py`'s second output), a plain neighbor count
(`countNeighborsWarp`), or a lookup-table build that never touches
simulation-state tensors (`wp_psi.py`, the one file of the 25 using a raw
`wp.launch` instead of `warpWrapper2` -- no backward pass at all, and
correctly so). Their true derivative is zero almost everywhere and undefined
exactly at a condition boundary, so a finite-difference probe that crosses
one gives a spuriously huge mismatch that looks like a bug but isn't. Don't
write a `gradcheck` call against these -- instead verify the *absence* of
differentiability is deliberate: assert `requires_grad is False` on the
output (a comparison op breaking the graph in plain PyTorch, or the
`_dtype_is_float` gate on a non-float Warp output), or just print the
forward values with a one-line "not in scope" note, as
`gradcheck_surfaceDetection.py` and `gradcheck_util.py` both do.

## warp-lang version

**Resolved, 2026-09-04**: `pyproject.toml` now pins `warp-lang>=1.17.0` (see
`docs/historic_plans/CLEANUP_PLAN.md`). 1.17 fixes class 1's ternary bug at
the source -- confirmed directly via `warpSPHCore/scripts/
repro_ternary_adjoint_zeroing.py`, not just assumed from the changelog -- so
the `access_optional`/explicit `if`-`else` workaround is no longer required
for *new* kernel code, though existing uses of it are still correct and
don't need to be reverted. Before this pin landed, the installed version had
drifted three times with nobody changing `pyproject.toml` (1.12.0 → a
1.17.0.dev3 local checkout → 1.15.0 from PyPI) -- the general lesson still
applies to any future unpinned dependency: if a previously-clean script
starts failing with no code changes on your end, check `python -c "import
warp; print(warp.__version__)"` against what it last passed under before
assuming a new code bug.
