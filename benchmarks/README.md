# Benchmarks

For CPU/GPU operators, neighbour search, memory controls, RHS and real solver
steps, including input/Morton storage and compute-flag sweeps, see
[the single-device performance suite](performance/README.md).

## Wave-case integrator comparison

Benchmark suites that compare the **explicit and implicit time integrators**
of `warpSPHIntegrators` on the registered `waveEquation` case, across the
four axes the notebook
(`examples/wave/waveCase_implicit_vs_explicit.ipynb`) only sketches:

* **runtime** — ms/step, ms/RHS, measured RHS evaluations per step, and the
  log-log scaling exponent vs. particle count;
* **memory** — peak allocated/reserved GPU memory per run, plus the static
  state+adjacency footprint (MB and KB/particle);
* **accuracy** — relative L2 error against a converged finer-`dt` reference
  and the *measured* convergence order per `dt` level;
* **stability** — where each explicit scheme stops, and which implicit
  *internal-solver settings* (Picard count, JFNK matvec × tol × max
  iterations) stay finite and bounded past the explicit limit.

Run from the repo root in the `warp` conda env:

```sh
python benchmarks/wave/bench_accuracy.py    --help
python benchmarks/wave/bench_performance.py --help
python benchmarks/wave/bench_stability.py   --help
```

Each suite writes a timestamped directory (or `--out DIR`) containing
`results.json` (machine-readable, one record per measured run), `summary.md`
(the table you want to read), and one PNG (skip with `--no-plot`).

## Why a custom step loop instead of the frontend runner?

`warpSPH.runner` hides the integrator call inside its own loop, which is the
right thing for running a simulation but the wrong thing for benchmarking
an *integrator configuration*:

* the notebook's measured path is an unrolled
  `ctx.integrator.function(state=..., f=..., dt=..., config=...,
  schemeConfig=..., verbose=False, solver=...)` — only that call shape lets
  a benchmark pass a specific `NonlinearSolver` (Picard with *n* iterations,
  JFNK with a given matvec/tol/budget); the frontend always uses the
  registry default;
* a counting wrapper around `ctx.stepFunction` then sees **every** RHS
  evaluation, including the ones a `JFNKSolver` matvec hides inside its GMRES
  Krylov iterations — that is the number a cost comparison actually needs
  (a "2-iteration" JFNK step may cost ~30 RHS evaluations);
* a `RecordingSolver` wrapper (plain delegation over the `NonlinearSolver`
  protocol) captures each stage solve's `(converged, iterations)` verdict,
  which the DIRK driver otherwise discards. Without it, a JFNK that used up
  its `max_iterations` budget and returned an un-converged stage is
  indistinguishable from a converged one in the state alone.

What is deliberately *not* measured: per-step `case.diagnostics(...)` (the
notebook's trajectory loop pays for a Gradient operator per step; a pure
integrator benchmark must not). Energy is taken at the initial and final
state only, outside timed regions.

## Layout

```
benchmarks/
  common/
    schemes.py   # the scheme + internal-solver registry (data, one line per entry)
    runner.py    # buildWaveCase + runScheme: the instrumented step loop
    metrics.py   # relL2, effectiveOrder, loglogFit, fmt
    report.py    # results.json / summary.md / PNG (Agg backend, headless-safe)
    scaling.py   # post-hoc scaling graph set from stored results.json
  wave/
    bench_accuracy.py    # dt-refinement sweep vs. a converged reference
    bench_performance.py # particle-count sweep: time, memory, scaling
    bench_stability.py   # dt-multiplier sweep past the explicit limit
    bench_plots.py       # render the scaling graph set from stored run outputs
  results/               # timestamped output
tests/test_bench_wave.py # small smoke + consistency tests for the machinery
```

## The scheme registry (`common/schemes.py`)

Every entry pairs one `IntegrationSchemeType` member name (resolved through
the exact path `buildContext` uses, so a typo fails at import time, not at
run time) with, for the implicit entries, the `NonlinearSolver` that drives
its stage equations. Orders are read from the integrator registry itself,
never re-stated. 64 entries today:

* **24 explicit** — one-step RK (orders 1–5, embedded, TVD, SSP), symplectic
  (leapfrog, velocity-Verlet, PEFRL, ...), and the Adams-Bashforth /
  ABM multistep family;
* **40 implicit** — 4 DIRK tableaus (backward Euler, implicit midpoint,
  trapezoidal, SDIRK2) × 10 internal-solver configurations: Picard at fixed
  counts 2/4/8/16, and JFNK with matvec `fd`/`jvp` × `tol` 1e-4/1e-6/1e-8 ×
  matching `max_iterations` budgets. `*_jfnk_{fd,jvp}_1e-6`
  (`max_iterations=15`) is exactly the notebook's configuration.

Keys are stable: `<tableau>_<solver>`, e.g. `sdirk2_jfnk_jvp_1e-6`,
`be_picard8`, `ab4`, `rk4`. Adding a configuration is one line in
`_EXPLICIT`/`_SOLVERS`; a suite's default set is one line in
`*_DEFAULT`. `--schemes all` (or any explicit key list) reaches everything.

## Measuring conventions

* **RHS counting** wraps `ctx.stepFunction`; it counts the evaluations a
  stage solve makes internally, so `f/step` is the price a scheme actually
  pays. Multistep schemes are run with `history=` threaded between calls
  (`IntegrationResult.history` -> next call's `history=`), their documented
  correct convention — without it they re-run their Dormand-Prince starter
  every step and cost 6–7× their nominal one evaluation per step.
* **Timing** is CUDA-event based per step (the frontend `_Timer`'s
  convention), with untimed warmup steps first so one-time warp kernel
  compilation/loads never enter a number. On CPU it falls back to the wall
  clock.
* **Memory** is the CUDA allocator high-water mark for the run (reset per
  run), plus the static footprint of the built state + adjacency tensors
  (MB and KB/particle). On CPU it falls back to the process RSS
  high-water mark (`peakReserved` is then meaningless and reported as 0).
* **Divergence** is the notebook's own stop condition: a non-finite max|u|
  at any step ends the run early and is recorded as `diverged`.

## The internal-solver record (and a float32 caveat)

Implicit runs report, per stage solve: the count of solves, how many the
solver flagged `converged`, and the per-solve iteration min/max/mean. Two
properties of the shipped stack shape how to read them:

1. **Picard (fixed count) has no convergence signal by construction.**
   `FixedPointSolver` without a `tol` runs its exact schedule and reports
   `converged=True` — so `solver conv` for a Picard run is the schedule
   itself (e.g. `46/46`), and the only axis that matters for it is the
   count (2/4/8/16): accuracy and cost both move with it.
2. **Under float32, JFNK's `converged` flag is effectively unreachable.**
   The DIRK driver passes its own Hairer-Wanner weighted norm
   (`rtol=1e-3, atol=1e-6`, scaled by the reference state) to the solver,
   and `JFNKSolver` checks *that* norm against its `tol`. The weighted
   residual of a float32 solve floors near the float32 epsilon divided by
   the norm's rtol (≈1e-4), above any `tol` in 1e-4..1e-8 — so most stage
   solves run their full `max_iterations` budget and report
   `converged=False`: at coarse-to-moderate dt the column reads 0/N, and
   only at the finest dt, where the stage residual is small enough, do
   some solves dip under the threshold (25/326 in the default accuracy
   run). The solution is still good (for this linear problem Newton needs
   one correction; the remaining budget just re-solves the same linear
   system to the precision floor), but the **budget is now a direct cost
   knob**: `maxit=20` costs ~2× `maxit=5` at the same accuracy. Under
   **float64** the flag works as intended (residual floors near 1e-13)
   and discriminates. In float32 the honest columns are therefore `iters`
   (== max_iterations ⇒ budget exhausted) and the cost columns
   (`ms/step`, `f/step`) — the stability suite's summary says so.

## The three suites

### `bench_accuracy.py` — temporal error and measured order

One resolution, one horizon (`--tEnd`, default 0.5), a `dt` grid
`dt_CFL, dt_CFL/2, ...` (`dt_CFL` is the case's own CFL-derived dt — the
notebook's `DT_DEFAULT`). The reference is `--reference-scheme` (RK4 by
default) run **one extra refinement finer** than the finest tested dt, on
the same spatial discretization, so the shared spatial error cancels and
the measured error is temporal. The table's `order` column is the
log-ratio between successive dt levels. One case-specific caveat: at the
default horizon (`--tEnd 0.5`) the field still carries high-wavenumber
content that is time-under-resolved even at the finest tested dt, so the
*measured* order sits below the nominal one for every scheme (≈1 and
rising as dt is refined) — and all schemes track each other to within a
few percent at equal dt, which is the true correctness signal (the
notebook's RK4-vs-implicit comparison, quantified). `dE/E` is the
total-energy change over the run (the case is dissipative through its
absorbing border, so it is a drift indicator, not an error).

### `bench_performance.py` — runtime and memory vs. particle count

Sweeps `--nxs` (default 32 64 128 256; particle count = nx²) at each
scheme's own CFL dt for `--steps` timed steps after untimed warmup (default
50 — long enough to absorb both the one-time warp kernel loads and the GPU
clock ramp, so a cold GPU does not bias the smallest-N point and the fitted
slope), and fits the log-log slope of ms/step vs. N (≈1: linear, the
expectation for a fixed-support-radius SPH step with a cell list; ≈2:
all-pairs). `ms/RHS` is the fair cross-scheme number; `f/step` is the
measured RHS count. At the default resolutions (≤65k particles) the per-RHS
cost is still launch-overhead dominated, so a slope near 0 is the expected
reading; push `--nxs` higher to see the true neighbour-search scaling.

### `bench_stability.py` — past the explicit limit

Sweeps `dt = mult × dt_CFL` (default multipliers 1 2 4 8 16 32 64) for a
small `--nx` (the point is dt, not cost). Explicit schemes show the first
multiplier at which max|u| goes non-finite or leaves the bounded band; the
binding limit for the higher-order ones is the absorbing border's linear
damping (`dt < ~2.8/dampingStrength`; case file comment: cflFactor 0.3
blows up, 0.1 stays bounded). `bounded` = finite and peak max|u| ≤
`--bounded-factor` × initial throughout, so a run that stays finite but
runs away is `unbounded`, not `diverged`. Implicit schemes report the same
plus the internal-solver record — the table that answers "which solver
settings survive, and at what iteration cost".

## Scaling graph sets from stored outputs (`bench_plots.py`)

Each suite's `--plot` writes one summary PNG next to its tables. For the
scaling questions the suites exist to answer — *how does each quantity
scale, and how do the schemes compare on it* — `bench_plots.py` turns any
stored run output into a full set of graphs, without re-running anything
(no GPU needed, it only reads the JSON):

```sh
python -m benchmarks.wave.bench_plots /path/to/run_dir        # or its results.json
python -m benchmarks.wave.bench_plots runA runB --out /tmp/p  # overlay two runs
```

The x-axis is whatever that suite swept (particle count / dt / dt
multiplier) and each panel is one quantity with one curve per scheme —
plus a combined overview grid and a `scaling.md` index carrying the input
provenance and the fitted log-log scaling exponents per scheme per
quantity (the numbers behind the curves). The panel set per suite lives in
`common/scaling.py` (`PANELS`): performance gets ms/step, ms/RHS, f/step,
peak + static memory, build time; accuracy gets the error curves, measured
order, energy drift, and cost; stability gets the boundedness envelope,
cost, internal-solver iterations, and the converged fraction. Overlaying
several runs tags each run's curves in the legends, which is also how you
check run-to-run reproducibility visually.

## Extending to another case

`common/runner.py::buildWaveCase` is the only case-specific seam (it builds
the `waveEquation` case via the public `CaseSpec` → `buildContext` →
`configureScheme` → `buildSystem` → `initialConditions` path). A new case
means a sibling `build<Case>Case` plus a sibling suite package that
reuses `runScheme`/`metrics`/`report` — the measurement machinery is
case-agnostic by construction (the error metrics take the compared fields
explicitly; only the default `u`/`v` pair is case-specific).

