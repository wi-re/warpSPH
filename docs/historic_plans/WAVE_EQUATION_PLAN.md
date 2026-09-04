# warpSPH — Wave-Equation Case Salvage Plan

> ## ✅ COMPLETE — superseded by work that landed without a matching plan doc
>
> Despite the header below still saying "Not started," the scheme has been
> wired, tested, and taken well past this plan's own scope since it was
> written (2026-08-18): `f_wave_equation` is registered in
> `schemes/builder.py` and `waveEquation` in `cases/__init__.py`, both import
> cleanly and build via `buildScheme`. Test coverage now spans
> `tests/test_waveEquation.py`, `tests/test_physics.py` (3 dimension/energy/
> divergence cases), `tests/test_forwardModeWave.py` (the "explicitly not
> doing" forward-mode AD wiring this plan deferred), and
> `tests/test_implicitWaveEquation.py` (implicit DIRK/JFNK time integration,
> the other deferred item) — 20 tests, all green as of this cleanup pass
> (09-04). Nothing in this plan's "Steps" or "Verification" sections is
> outstanding. Left as a working record below; if picked back up, treat the
> test files themselves as the current spec rather than the steps here.

Plan for turning the wave-equation subsystem (`schemes/waveEquation.py`,
`systems/waveSystem.py`, `configurations/waveEquationConfig.py`,
`sample/waveSystem.py`, `caseUtils/waveEquation/`) from the unwired demo code
described in `README.md`'s "The wave system" section and `docs/historic_plans/CLEANUP_PLAN.md`
into a runnable, tested, convention-aligned `Case`. Written up here (rather
than only in an ephemeral plan file) so it can be picked up in a later
session.

## Why

`warpSPH` needs a very simple, narrow-surface-area toy case to validate a
forward-mode differentiable simulation pipeline and, later, to test
convergence/stability once hooked into an implicit time integrator. The
scalar wave equation (`d2u/dt2 = c^2 laplacian(u)`) is ideal for this: it
exercises only a neighbour search and a single SPH Laplacian operator, with
no pressure solve, no EOS, no artificial viscosity.

## Current state (why it doesn't run today)

- `f_wave_equation(system, dt, verbose=False)` uses the *old* calling
  convention — no `config`/`schemeConfig` params — so it cannot be driven by
  `runner.run()`, which always calls step functions as
  `f(state, dt, config=..., schemeConfig=..., verbose=...)`.
- It is not registered as a scheme (`schemes/builder.py`'s `_SCHEMES`) or as
  a `Case` (`cases/`), so there is no `warpsph-run waveEquation` entry point.
- `caseUtils/waveEquation/sample.py`'s `addNoise` calls `sampleVoronoi`
  without importing it — a real `NameError` whenever `noisyICs=True`.
- `caseUtils/waveEquation/damping.py`'s `'exponential'` damping profile
  hardcodes `.to('cuda')` — crashes on CPU. Not on the current default call
  path (`'cosine'` via `borderDamping_strong`), but worth a one-line fix.
- There is no test exercising any of this.

## Scope

**Explicitly out of scope for this pass** (per the user): implicit time
integration and forward-mode AD itself. Positions stay fixed — this drops
the "moving-neighbourhood" framing in the current docstrings but matches how
`finalizeWaveSystemSetup` already builds the Verlet list exactly once. This
pass only needs to produce a *runnable, tested, convention-aligned explicit*
wave case; the implicit/differentiable work builds on top of it later.

**Dimension is not out of scope.** The source/obstacle machinery
(`caseUtils/waveEquation/shape_generation.py`) is genuinely 2D-only (2x2
rotation matrices, 2-tuple offsets, 2D SDF primitives) and stays that way —
generalizing it is real, separate work. But the *base* pipeline with no
sources/obstacles (uniform wave speed, cosine border-damping absorption
instead of a hard wall) has no inherent dimension dependence and currently
fails in 1D/3D only because of a few hardcoded `2`s, not because of any real
2D-specific logic — the same shared `sampleParticles`/`buildVerletList`/
`warpOperation` infra already serves the 1D `linearWave` case and 3D `sod3d`
case. Generalizing that base path is in scope: it exercises the Laplacian
operator and neighbour search across dimensions with very little extra code,
directly useful for the differentiable/implicit work this is building
toward.

**Differentiability of source/obstacle parameters is also in scope**, even
though wiring up an actual forward-mode integrator is not: this case is meant
to validate that gradients can flow through the whole pipeline to a source's
position/intensity and an obstacle's position/speed, so those need to be
real leaf `torch.Tensor`s a caller can `requires_grad_()`, not Python floats
baked in during setup. See step 5.

## Steps

### 1. Fix the two standalone bugs, and generalize the base pipeline to N-D
- `caseUtils/waveEquation/sample.py`: import `sampleVoronoi` (it lives in
  `warpSPH.math.noiseFunctions.generator`, re-exported from `warpSPH.math`)
  so `addNoise(noiseType='perlin')` works. While touching it, replace the
  `nx = int(math.sqrt(N))` grid-size guess (assumes a 2D square grid) with
  `nx = round(N ** (1 / config.dim))`.
- `caseUtils/waveEquation/damping.py`:
  - `sampleDamping`'s `'cosine'`/`'polynomial'` branches hardcode `dist`/
    `rect_dist` to `positions[:, 0]`/`positions[:, 1]` and `dist_max =
    sqrt(2)`. Generalize to `torch.linalg.norm(positions, dim=-1)` /
    `positions.abs().amax(dim=-1)` / `sqrt(dim)` — dimension falls out of the
    tensor shape, no branching needed. This is the profile
    `borderDamping_strong` (the actual default/live path) uses, so this is
    the one that matters for the registered case.
  - The `'exponential'` branch hardcodes both `.to('cuda')` (crashes on CPU)
    and a 2-component `torch.tensor([l, l])` box half-extent. Fix both in the
    same edit (drop `.to('cuda')`, use `[l] * dim`) even though it's not on
    the default path — no reason to leave a landmine next to code being
    otherwise generalized.
- `caseUtils/waveEquation/gencase.py`: `genInitial` allocates `nx**2` and
  takes `nx` only for that. Replace with `particleState.positions.shape[0]`
  (the grids need one entry per sampled particle, however many dimensions
  produced it) — `nx` becomes unused and can be dropped from the signature.
- `systems/waveSystem.py`'s `computeDt`: replace the hardcoded 2D domain-area
  product (`(max[0]-min[0]) * (max[1]-min[1])`) with a product over
  `range(config.dim)`, and `dx = (domainVolume / n) ** (1 / config.dim)`.

Source/obstacle shapes (`shape_generation.py`, `generateDomainBox`) stay
2D-only, unchanged — the registered case uses `domainBox=False` for 1D/3D
variants and leans on the now-generalized cosine border damping for
absorption instead of a hard wall.

### 2. Modernize `f_wave_equation`'s calling convention
`src/warpSPH/schemes/waveEquation.py`: change the signature to
`f_wave_equation(system, dt, config, schemeConfig, verbose=False)`, matching
`compressibleSPH_Monaghan` (`schemes/monaghan.py:32`) and the
`SchemeBundle.stepFunction` contract (`schemes/builder.py`).

Since positions are fixed, drop the per-step `buildVerletList` call — reuse
`system.adjacency` (already built once by `finalizeWaveSystemSetup`) instead
of rebuilding it every step. This both matches the "ease implementation"
directive and removes a chunk of complexity/risk the moving case would need.
Still return the adjacency as the second tuple element (`return update,
system.adjacency`) — `WaveSystemv3.finalize` (`systems/waveSystem.py:78`)
reads `returnValues[-1][0]` for it, and `warpSPHIntegrators.util.split_return`
already tolerates a variable-length return, so this needs no integrator-side
change.

Replace the hardcoded `KernelFunctions.Wendland2` /
`LaplacianScheme.Brookshaw` / `GradientScheme.Difference` /
`SupportScheme.SuperSymmetric` in the `OperationProperties` with values
pulled from `config`/a new scheme config (next step), the way
`compressibleSPH_Monaghan` reads `config.kernel` — keep the current values as
defaults so behavior doesn't change.

### 3. Add a minimal scheme config and register the scheme
Add a small `WaveEquationConfig` dataclass (in
`configurations/waveEquationConfig.py`, alongside the existing
`WaveCaseConfig`) to fill the `SimulationConfig` slot of `SchemeBundle` —
holding just the kernel/laplacian/gradient/support-mode knobs now hardcoded
in `f_wave_equation`, plus the two `*ToDict`/`dictTo*` codec functions the
other four schemes have (mirror `compressibleConfigToDict` /
`dictToCompressibleConfig`).

Add a `WaveEquationScheme(Enum)` to `enumTypes.py` (mirroring
`CompressibleSPHScheme` etc.), and a `_waveEquation()` factory in
`schemes/builder.py` wired into `_SCHEMES`/`_ALIASES`:
```python
def _waveEquation() -> SchemeBundle:
    return SchemeBundle(
        SimulationSystem=WaveSystemv3,
        SimulationState=WaveSystemStatev3,
        SimulationConfig=WaveEquationConfig,
        SimulationUpdate=WaveSystemUpdatev3,
        stepFunction=f_wave_equation,
        exportFunction=waveEquationConfigToDict,
        importFunction=dictToWaveEquationConfig,
    )
```

### 4. Register a `Case`
New `src/warpSPH/cases/waveEquation.py`, following the `Case`/`RunContext`
contract (`runner/case.py`, modeled on `cases/linearWave.py`):
- `buildSystem(ctx)`: build a `WaveCaseConfig` directly from `ctx.param(...)`
  values (a couple of sources, optional obstacles, damping toggle) — the same
  way `linearWaveCase.buildSystem` pulls `ctx.param('A')`, etc. — rather than
  requiring a TOML casefile. This keeps one source of truth for defaults
  (`Case.defaults`/`params`) like every other registered case; leave
  `caseUtils/waveEquation/casefile.py`'s TOML loader as an optional,
  untouched alternate path for power users, not a requirement for the
  registered case. Internally this still runs the existing pipeline stages:
  `sampleParticles` → `genInitial` → (shape_generation to place any
  sources/obstacles, only reachable when `ctx.spec.dim == 2`) →
  `finalizeWaveSystemSetup`. Respect `ctx.spec.dim`: only pass `domainBox` /
  sources / obstacles through when `dim == 2`; for `dim in (1, 3)` sample a
  plain domain with a single point/region source placed via `WaveSource`'s
  existing shapeless magnitude field (or directly on the `u` grid, whichever
  is less code) and `domainBox=False`, relying on the now-generalized cosine
  border damping.
- `initialConditions(ctx, system)`: call the existing `computeDt` (already in
  `systems/waveSystem.py`) once to set `ctx.config.dt`, mirroring
  `linearWaveCase.initialConditions`. No `timestep` hook needed — wave speed
  and positions are static, so a fixed CFL-derived `dt` is exact for the run.
- `diagnostics(ctx, state)`: report a discrete wave-energy estimate
  (`0.5 * sum(m * v**2)` plus a gradient-based potential-energy term via the
  existing `WarpOperation.Gradient`), plus `u`/`v` max-abs, so tests can
  assert an energy-drift bound the way `test_physics.py`'s `_ENERGY_DRIFT`
  does for the fluid schemes.
- `defaults=dict(dim=2, nx=64, L=2.0, tLimit=...)` — `dim` becomes a normal
  `--dim` override (like every other case), so `warpsph-run waveEquation
  --dim 1` and `--dim 3` work via the branch above, not a separate case.
- Register the module name in `cases/__init__.py`'s `CASE_MODULES` so it's
  reachable as `warpsph-run waveEquation`.

### 5. Expose source/obstacle position and intensity as differentiable tensors

Today, source/obstacle parameters never survive as tensors that could carry a
gradient: `WaveSource.magnitude`/`WaveBoundary.speed` are plain Python
`float`s, `ShapeSpec.position` is a plain tuple, and
`populate_source_obstacle_grids_structured` actively detaches things that
*are* tensors (`torch.rand(...).item()` for randomized magnitude/speed). Then
`finalizeWaveSystemSetup` bakes a value into the `u`/`c` grids by
`torch.where(idGrid == id, torch.full_like(grid, magnitude), grid)` — this
step *is* differentiable w.r.t. `magnitude` (an autograd-tracked
`torch.where` value arg), so intensity is cheap to fix: stop converting to
Python floats, keep `magnitude`/`speed` as 0-d tensors end to end.

Position is the real fork. The existing 2D source/obstacle *shapes*
(`shape_generation.py`) place themselves by thresholding an SDF
(`torch.where(sdf(points) < 0, 1, 0)`) into a discrete id-grid — a step
function, so `d(mask)/d(position)` is zero almost everywhere no matter how
the position tensor is wired through. Making that genuinely differentiable
(soft/smoothed SDF masks) is real work and is **not** in scope here (see
"Explicitly not doing").

For the dimension-generic point source this plan is already adding for
1D/3D (step 4), use a smooth placement instead of an id-grid mask: contribute
to the initial `u` field as a kernel-weighted bump centered at a `position`
tensor, e.g. reusing the SPH kernel already available via `KernelFunctions`
(the same one `warpOperation` uses) evaluated at `|x_i - position|`, scaled
by `magnitude`. This is both simpler than painting an id-grid for a single
point source and gives a real, non-zero gradient w.r.t. `position` through
the kernel's shape.

Verification (backward-mode only — `ExecutionMode.FORWARD` is not
implemented in `warpSPHCore`, so only reverse-mode can be checked today):
add a small test/script that builds a 1D or 2D wave system with one smooth
point source, sets `requires_grad_()` on its `position` and `magnitude`
tensors, runs `f_wave_equation` for a handful of steps, sums a scalar probe
of the final `u` field, calls `.backward()`, and asserts both gradients are
finite and non-zero. This isn't a full `torch.autograd.gradcheck` of new
kernels (there are none — see step 6's note on `gradcheck_waveEquation.py`)
but a pipeline-level check that the exposed tensors actually receive
gradient, which is the concrete thing being asked for.

### 6. Tests
- `tests/test_physics.py`: add a `waveResult` fixture running the new case
  for ~`STEPS` steps (mirroring `tgvResult`), asserting `not diverged` and an
  energy-drift bound from the new `diagnostics`. Parametrize it over
  `dim in (1, 2, 3)` — this is the payoff of generalizing the base pipeline:
  the same scheme code now gets exercised by the neighbour search and
  Laplacian operator in three different dimensionalities, not just 2D.
- New `tests/test_waveEquation.py`: a small standalone convergence/accuracy
  check that does **not** go through the `Case`/CLI machinery — build a
  `WaveSystemStatev3` directly with a single sinusoidal standing-wave IC
  (`u(x,0) = sin(k.x)`, `v(x,0) = 0`, constant `c`, `damping = 0`, periodic
  domain), which has the closed form `u(x,t) = sin(k.x) cos(c |k| t)`. Run
  `f_wave_equation` at two or three resolutions via a plain time-stepping
  loop (no integrator machinery needed beyond what `computeDt` already
  gives), compute the L2 error against the analytic solution at a fixed
  physical time, and assert the error shrinks with resolution. Do this at
  least in 1D (simplest, cheapest, exact) and 2D; 3D is a nice-to-have if the
  cost is acceptable. This is intentionally a small, purpose-built harness
  (the repo has no reusable convergence-testing infra to build on) — it also
  becomes the baseline the future implicit-solver convergence/stability
  comparison will extend.

## Explicitly not doing
- No implicit time integration (separate follow-up step).
- No forward-mode AD wiring (`ExecutionMode.FORWARD` stays
  `NotImplementedError` in `warpSPHCore`; out of scope here).
- No N-dimensional source/obstacle shapes — `shape_generation.py` stays
  2D-only; 1D/3D case variants use a plain domain + point source instead of
  the SDF shape machinery.
- No new `gradcheck_waveEquation.py` — `f_wave_equation` calls existing,
  already-integrated `warpOperation`/`buildVerletList` primitives and defines
  no new `@wp.kernel`s of its own, so there's nothing scheme-specific for a
  gradcheck script to cover yet (the underlying Laplacian kernel lives in the
  sibling `warpSPHCore` repo).

## Verification
- `scripts/check_imports.py` — should report zero new errors.
- `pytest tests/test_physics.py -k wave` and `pytest tests/test_waveEquation.py`
  (needs a CUDA device, per repo convention).
- `warpsph-run waveEquation --nx 64 --tLimit 2.0` (2D, with a source/obstacle)
  and `--dim 1`/`--dim 3` variants all run end-to-end and produce a sane,
  bounded `u`/`v` field (spot-check via the case's diagnostics or a quick
  plot).
- The step-5 gradient-flow check passes: `.backward()` through a short
  `f_wave_equation` rollout produces finite, non-zero gradients on the smooth
  point source's `position` and `magnitude` tensors.
