# Porting an example, and taking a case to 2D/3D

Written 2026-08-12, after converting Sod (`examples/compressible/01-sod/`) and
then extruding it into `sod2d`/`sod3d`. `docs/historic_plans/CLEANUP_PLAN.md`'s open "Notebook
simplification, the rest" item — 33 notebooks, ~13k LOC — is the work this is
for. `LESSONS_LEARNED.md` holds the general "why" of the sweep; this file is
the procedure, the conventions that have a reason behind them, and the specific
things that cost hours.

Two independent jobs are described here. Porting a case to the runner (part 1
and 2) is mechanical once you know the shape. Taking one to 2D/3D (part 3) is
not — it is mostly a sampling problem, and it is where the real bugs were.

---

## 1. What "the current style" is

Three artefacts per case, and a hard split between what is generic and what is
the case's own physics:

| file | holds | size |
|---|---|---|
| `src/warpSPH/cases/<name>.py` | the `Case`: geometry/IC, diagnostics, plot hooks, defaults | ~100 lines |
| `examples/<family>/<name>.py` | a `caseMain(<name>Case, PRESET + sys.argv[1:])` wrapper | ~30 lines, mostly docstring |
| `examples/<family>/<name>.ipynb` | the same case, unrolled and editable | as fat as it needs to be |

Everything a pre-`Case` script spelled out — config construction, `buildScheme`
unpacking, the step loop, export, plotting, ffmpeg — is `warpSPH.runner` and
does not get copied into a case.

**The notebook does not shrink.** This is the correction recorded in
`docs/historic_plans/CLEANUP_PLAN.md` and it is easy to get backwards: a notebook that only
re-derives its `.py` sibling should *not* become another thin `caseMain`
wrapper. Notebooks are where new physics and new hooks get prototyped, so they
build ICs through the real case code but keep the step loop visible in a cell.
`sod_backprop.ipynb` exists because `sod_1d.ipynb` had that loop exposed with a
hook-point comment sitting in it.

## 2. The procedure

1. **Read the old notebook and sort it into three piles**: geometry/IC,
   diagnostics, plots. Everything left over is runner and gets deleted, not
   ported.
2. **Write the `Case`.** Only `buildSystem` is required; the rest are optional
   hooks (`configureScheme`, `initialConditions`, `diagnostics`, `setupPlot`,
   `updatePlot`, `extraData`, `postStep`, `timestep`), each taking the
   `RunContext`. Pull the shared block from `cases/compressible.py`
   (`COMPRESSIBLE_DEFAULTS`, `COMPRESSIBLE_PARAMS`, `configureCompressible`,
   `compressibleDiagnostics`, `compressibleTimestep`, `paramExtraData`) or
   `cases/weaklyCompressible.py` (`WEAKLY_COMPRESSIBLE_DEFAULTS`,
   `configureWeaklyCompressible`, `buildRegionSystem`, `fluidRegion`,
   `boundaryRegion`, `setupTimestep`, ...) rather than restating it. Plot hooks
   come from `cases/plotting.py`: `particlePlot(fields)` for 2D field views,
   `profilePlot(axes, shape)` for 1D profiles (see `noh.py`, `kidder.py`,
   `woodwardColella.py`).
3. **Register it** in `CASE_MODULES` (`cases/__init__.py`). `warpsph-run`,
   `scripts/run_sweep.py` and the CLI all discover it from there — the sweep
   needs no edit, it enumerates the registry.
4. **Script wrapper**: `PRESET` list plus `caseMain`. Put the "why this case is
   interesting" prose in its module docstring; that is where someone lands.
5. **Notebook**, following `sod_1d.ipynb` cell for cell:
   intro markdown → imports/bootstrap → **parameters cell** → IC build through
   the real case code → export/plot setup → **unrolled step loop with a
   `# <-- hook point` comment** → close/encode. Then add whatever analysis is
   genuinely notebook-shaped for that case.
6. **Test it** in `tests/test_physics.py`: a physical invariant (energy
   conserved, kinetic energy grows, nothing diverged), never a golden number.
   20 steps at a coarse resolution — the suite is a pre-commit check.
7. **Verify**, in this order:
   `bash scripts/run_tests.sh` · `python scripts/run_sweep.py --cases <name>` ·
   `python scripts/check_imports.py` (it AST-scans notebook cells, which a
   runtime import check never reaches) · `jupyter nbconvert --to notebook
   --execute` on the notebook.

## 3. Notebook conventions that have a reason

- **Parameters cell surfaces every knob** a `--flag` would set, merged over
  `case.defaults`/`case.params` so nothing is silently re-specified.
- **Do not use `setupPlot`/`updatePlot` in a notebook.** They go through
  `runner.display.openWindow`/`pumpEvents`, which does not live-update inside a
  Jupyter cell in this environment — confirmed by testing both side by side,
  cause not fully understood. Call the plotting function directly
  (`plt.subplots`, `ax.clear()`, `fig.canvas.draw()`/`flush_events()`), which
  every pre-`Case` notebook already did and which does update.
- **`%matplotlib widget` for a live-updating loop, `inline` for static
  end-of-run figures.** `inline` needs no ipympl and renders under
  `nbconvert`; `widget` produces widget-view outputs there, which is fine for a
  "doesn't crash" check but gives you no picture to look at.
- **Under `inline`, assign the figure** (`figure = plotConvergence(...)`) or it
  renders twice — once as the cell result, once by the inline backend's
  end-of-cell hook.
- **Ship notebooks with no stored outputs.** All five in `01-sod/` have none.
- **Filter library chatter rather than muting stdout.** `warpSPHCore`'s autograd
  bridge prints one line per grad-tracking launch (~14 per step), which buries
  everything else in a loop. `sod_backprop.ipynb`'s `_LineFilter` drops exactly
  that prefix and passes the rest through; a blanket `redirect_stdout` would
  also swallow the warnings you want.
- **Reference an animation from `outputs/`** in the intro cell
  (`![](outputs/<Case>.gif)`), produced by the notebook's own `encodeFrames`
  call.

## 4. Taking a case to 2D/3D

Order matters here: geometry, then sampling, then the constraint check, then
plotting. Sampling is the substance.

### 4.1 Read what the 1D geometry actually implies

Sod's 1D setup is periodic on `[-1, 1]` with the dense state on the middle half
and the light state wrapped around the outer quarters — *two* interfaces, at
`x = ±L/4`. Nothing is reflected explicitly: the mirror symmetry makes `x = 0`
and `x = ±L/2` behave as reflecting walls until a wave arrives. Extruding means
keeping that arrangement in x untouched and making the new directions plain
periodic slabs. Check what the analytic-solution overlay assumes before you
change any extent — `plotSod_` passes `geometry=(0., 1., 0.5)`, i.e. it is
hardcoded for `L=2`.

### 4.2 Equal mass, not equal spacing

The one thing that genuinely has to be redesigned. Giving both states the same
lattice leaves the dense side's particles `rho_l/rho_r` times heavier (4x for
Sod, a 75% mismatch), and a mass jump across a contact discontinuity is exactly
where SPH's density estimate misbehaves. Sample the light side **coarser by
`(rho_l/rho_r)**(1/dim)` in every direction** instead, so
`mass = cell volume * density` matches. In 1D this falls out of
`samplingRatio = 4` being the density ratio, which is why 1D masses were
already equal and why the issue only appears on the way up.
`sampleTriplePointEqualMass` makes the same trade in 2D with `sqrt(8)`.

Two integers have to come out of that and neither is free:

1. the transverse count, from the isotropic ideal `dx * ratio**(1/dim)`;
2. the x count, chosen so the mass matches given (1).

Then retry (1) one count either side and keep the best pair — a few lines, and
it regularly wins, because the first rounding otherwise dumps its error on the
second (at `dim=3, nx=100` it took the mass error from 1.4% to 0.29%).
**Bound the cell aspect while doing so** (`MAX_CELL_ASPECT = 1.2`): unbounded,
the same search takes 0.04% at that resolution by stretching cells to 1.37:1,
and an isotropic kernel feels the stretch more than the mass jump it bought off.

In 3D it cannot be exact — `4**(1/3)` is irrational, so no two commensurate
periodic lattices have equal masses. Measured for Sod over `nx` 25–400: exact
in 1D/2D where the counts divide evenly, ≤4.2% otherwise, ≤1.4% in 3D, cells
never worse than 1.17:1. Print it at build time (`sodSamplingReport`) and assert
it in a test — from the outside a badly rounded pair looks exactly like a good
one, and the failure mode is a quietly worse density estimate, not an error.

Keep the sampling arithmetic in a **pure function** taking counts and returning
counts (`sodSampling`) with the actual particle building separate. It makes the
notebook cell that inspects resolutions free, and it makes the test cheap.

### 4.3 The constraint that will bite you: slab width vs. support radius

A slab narrower than **twice the largest support radius** lets a particle
interact with its own periodic image. Nothing downstream detects it — you get
wrong densities and a plausible-looking run. The coarse (light) side sets it.

Two consequences:

- **Measure the slab in particle spacings, not in length.** The constraint is a
  multiple of the spacing, so a length that clears it at one `nx` fails at
  another. In spacings, the transverse count is constant and the particle count
  grows as `nx * spacings**(dim-1)` — linear in resolution, not quadratic or
  cubic. Sod needs ~16 spacings in 3D; 20 is the default and clears both sides.
- **Check it against the relaxed supports and raise.** `buildSodND` compares
  `h_optimal.max()` after `evaluateOptimalSupport` and refuses to return,
  naming the number to raise to. It caught a bad test config within minutes of
  being written.

### 4.4 The domain is cubic unless a case says otherwise

`buildDomainDescription(L, dim, ...)` takes a scalar and produces a cube.
For a slab, either mutate `ctx.config.domain.min`/`.max` in the case's
`configureScheme` (the precedent: `rayleighTaylor.py`, `triplePoint.py`) or have
the sampler write it back, which is what `buildSodND` does — it snaps the slab
to the lattice and then sets the domain to match, because a periodic box that
disagrees with the lattice by a fraction of a spacing has a wrong wrap-around
spacing.

### 4.5 Lattices, cell centres, and the mirror point

`buildPointCloud` derives one isotropic `dx` from the **shortest** edge and
fills each dimension with `ceil(l/dx)` points, cell-centred for periodic
dimensions — so `area = dx**dim` is only the true per-particle volume when the
box is commensurate. When you need exact control (two lattices, exact masses),
build the lattice explicitly with `linspace` and compute masses and supports
from the actual cell volume (`volumeToSupport(volume, targetNeighbors, dim)`).

A cell-centred lattice with an **odd** count puts a particle exactly on the
symmetry plane. That is correct and harmless on a mirror plane, but
`buildSod1D`'s "sample a block centred on the origin, then push the halves
apart" strands one light-state particle at `x = 0` — in the middle of the dense
state, carrying the wrong mass — whenever `nx // samplingRatio` is odd (`nx=100`
does it, the `nx=800` default does not). Laying the light state out as a
cell-centred lattice on the *periodic interval* `[L/4, 3L/4]` and wrapping it
back into the box gives the same mirror-symmetric arrangement with no special
case, and lets the count be odd, which halves the granularity of the equal-mass
rounding.

**Do not refactor the 1D builder to share the ND one** just because the latter
generalises it. `buildSod1D` carries an explicit `samplingRatio` knob rather
than deriving the count, and `sod_backprop.ipynb`'s recorded numbers are tied to
its exact output. Two builders with a test asserting where they agree — and
where they deliberately do not — is the cheaper arrangement.

### 4.6 Plotting: collapse, don't average; slice, don't project

- **Profile panels**: scatter *every* particle against its own `x`, with no
  averaging over the transverse directions. Because the domain is periodic
  there and the solution does not depend on them, all particles are directly
  comparable to the same 1D reference curve — and the **vertical spread at a
  given x is then a measurement**, not a plotting artefact: it is the symmetry
  breaking, and it appears first at the shock and the contact. This is what the
  papers do and it is why the existing 1D panels needed almost nothing: indexing
  the x column explicitly (`pos[indices, 0]`, `velocities[indices, 0]`) is a
  no-op in 1D and makes them work in any dimension.
- **Individual quantities**: use the particle visualizer the 2D examples use
  (`warpSPHPlotting.visualize`, via `runner.display.visualizeWithFallback` so a
  missing GL context degrades instead of taking the run down). Pass a windowed
  domain, not just windowed particles, or the axes will not zoom.
- **In 3D, slice — do not project.** `visualize` given a 3D state draws the whole
  depth on top of itself, and a coarse lattice then looks identical to a
  genuinely disordered front. Take `|z| < (less than one coarse spacing)` and
  say so in the title.
- A uniform colour map beats the examples' usual diverging one for a monotone
  field like Sod density: there is no midpoint to diverge about.

### 4.7 Dimension-dependent constants and caches are where the bugs are

Both real bugs found by running 3D at all were the same shape — a value that
depends on the dimension, computed or cached once by code that had only ever
seen one dimension.

- **B7's 3D kernel normalisation was 16x too small.** 1D and 2D were right.
  Densities came back at 1/16 of the mass they were built from, while
  velocities and wave *positions* still looked correct, because a uniform
  factor on density largely cancels in the dynamics. **Recipe:** sum the kernel
  over a uniform lattice of known density (`sum_j m_j W_ij` must return what you
  built) for every kernel and every dimension. ~20 lines, one run, and it is
  the only check that would have caught this.
- **Owen's psi LUT was cached in an unkeyed module global.** It is sliced by
  dimension, built on first use, and reused forever — so a process touching two
  dimensions relaxed the second one's supports against the first one's table.
  Every existing test ran one dimension per process, so nothing saw it; it
  surfaced as a test passing alone and failing in the suite. **Recipe:** when a
  cache holds anything derived from config, key it on what it was derived from,
  and add a test that builds A, then B, then A again and asserts the two A's
  agree.

More generally: **a test that passes alone and fails in the suite is reporting
global state, not flakiness.** Bisect by fixture order (`-k`) before assuming
anything else.

### 4.8 Budget

Particle count is `nx * spacings**(dim-1)` for the dense block. Sod's shipped
defaults land at 1,000 (1D, nx=800), 2,500 (2D, nx=100) and 20,032 (3D, nx=40)
— 3D pays the transverse count twice, so its resolution along the tube is
correspondingly lower for a comparable budget. A 3D notebook run to `tLimit` is
~25 steps and about a minute including the first-time kernel compile.

## 5. Verification, honestly

- `nbconvert --execute` proves a notebook does not crash. It proves nothing
  about whether the numbers are right — the two bugs above both produce clean
  runs.
- Assert **invariants**, not golden numbers: total energy conserved, thermal
  energy converting to kinetic, nothing diverged. They survive refactors and
  fail on real regressions.
- Look at a plot against the analytic solution once, deliberately, before
  declaring a case done. Both 3D bugs were visible there and in nothing else.
- After touching anything in a kernel or a sampler, run the whole suite rather
  than the case you changed: `scripts/run_tests.sh` is ~2 minutes and includes
  the gradcheck scripts as subprocesses.
