# `SimulationConfig.compute`: how a run uses the machine

2026-10-14. One place for the settings that decide *how* a step is computed, as opposed to what physics it computes. Code: [configurations/compute.py](../../src/warpSPH/configurations/compute.py).

## Why

The same decisions were spread over a case-spec flag (`CaseSpec.cudaGraph`), a per-scheme switch (`CompSPHConfig.stepMode` + `stepModeMaxParticles` + a table in `modules/compSPH/stepMode.py`), a per-solver field (`convergenceCheckSchedule`) and function defaults (`checkEvery`). They all depend on the same two things, the problem size and the hardware: below a few 10^4 particles a step is limited by the host (kernel launches, torch ops, host reads) and replaying it from a CUDA graph wins; above, it is limited by the GPU and the eager path wins because the graphed variants (fixed-shape grid support solve, deferred checks) cost kernel time.

## Fields

| Field | Default | Meaning |
|---|---|---|
| `execution` | `'eager'` | `'eager'`, `'graph'` (replay the step, or the scheme's right-hand side / dual-time iteration, from a CUDA graph where the scheme supports it) or `'auto'`: graph below a particle count, eager above |
| `graphMaxParticles` | `None` | `'auto'` limit; `None` = the table `GRAPH_MAX_PARTICLES[(scheme, dim)]` (an absent entry means eager) |
| `pipelineOutputs` | `True` | with a whole-step graph, run step n's diagnostics / frame while step n+1 replays |
| `iterationChecks` | `'adaptive'` | relaxed-Jacobi style loops (`warpSPHCore.linalg.CheckSchedule`) read the convergence test back after every iteration (`'every'`) or at checkpoints learned from the previous solve of the same kind (`'adaptive'`) |
| `krylovCheckEvery` | `None` (8) | Krylov solvers (`warpSPHCore.linalg.krylov`) read their still-running flag every this many iterations; they freeze the iterate when a test fires, so the interval costs wasted iterations (at most interval - 1), never accuracy |

From the case spec / CLI: `--execution`, `--graphMaxParticles`, `--iterationChecks`, `--krylovCheckEvery`, `--pipelineOutputs`; from Python `buildConfig(compute={...})` or `config.compute = ComputeConfig(...)`. Saved with the config (`configurationToDict`), and files written before the field existed load with the defaults.

## Precedence

    scheme config value (stepMode, stepModeMaxParticles)  >  CaseSpec legacy flag (cudaGraph)  >  config.compute  >  shipped default

`CompSPHConfig.stepMode`, `CaseSpec.cudaGraph` and a solver's `convergenceCheckSchedule` ('auto') default to "inherit", so existing cases and tests behave exactly as before. A graph request a run cannot honour (no CUDA, stored trajectories, solid wall rows, a viscosity switch with a step-boundary host time, Dirichlet boundary conditions) runs eagerly, with a warning only for an explicit `'graph'`.

## Measured crossovers behind `'auto'` (one GPU: RTX PRO 6000 Blackwell; hardware dependent, conservative)

| Scheme | dim | graph wins below | Evidence |
|---|---|---|---|
| CompSPH | 1 / 2 / 3 | 100k / 5k / never | `docs/compsph_step/README.md` 2.7 |
| CRKSPH | 1 / 2 / 3 | 100k / 20k / never | same, 2.9 |
| deltaSPH, sun2017DeltaSPH | 2 | 100k | `scheme_baseline.py time`, `dambreak`: graph 3.4x faster at 6k particles (4.5 vs 15.2 ms/step), 2.0x at 16k, 1.5x at 42k, 1.1x at 92k, tie at 200k |
| everything else | | not measured: eager | |

## Where an iterative solver's early exit fits (the design the check settings serve)

Regimes, from the measurement `check_interval` on the real IISPH operator (CG, 1,600 to 102,400 particles: ~0.78 ms per iteration at every size, i.e. host-launch-bound, so the interval is not yet measurable):

* **Large, GPU-bound**: eager with real early exits; check every iteration, ideally with a lagged flag read (the core closure CG's pattern) so the GPU never idles at a check.
* **Small, host-bound**: the whole solve in a graph. Conditional graph node (`wp.capture_while`, CUDA >= 12.4: a true in-graph early exit, 46 of 400 iterations in the prototype) or, without it, a fixed iteration count sized from the previous solve with the retrospective freeze.
* `krylovCheckEvery` and `iterationChecks` are the knobs of the first regime today; the graph regime arrives with the fused warp update kernels and a `mode` that `execution` selects.

## Adding a compute setting

Put it on `ComputeConfig` (validated in `validate()`), resolve it through one function in `configurations/compute.py` (so the precedence rule lives in one place), and add a test to `tests/test_computeConfig.py`.
