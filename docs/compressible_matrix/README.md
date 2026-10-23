# Fixed-resolution compressible comparison

This harness compares declared numerical presets on one fixed resolution per
case. It is an overview for finding outliers, not an equal-compute or convergence
study. It does not change solver implementations or tune settings after failures.

The entry point is `scripts/run_compressibleMatrix.py`. Planning and reporting
never run a simulation. Run commands execute selected cells **sequentially**, each
in a fresh subprocess. Use the normal Warp Python environment. Schedule measured
GPU runs when the device is undisturbed; the harness does not stop other GPU jobs
or claim that hardware was idle.

## Plan and inspect

```bash
python scripts/run_compressibleMatrix.py list
python scripts/run_compressibleMatrix.py plan --root .tmp/compressible-matrix
python scripts/run_compressibleMatrix.py check --root .tmp/compressible-matrix
```

`manifest.json` freezes the inputs for 90 cells: nine cases and ten presets.
`check` validates typed configurations on CPU without sampling or stepping.
Planning refuses to overwrite a manifest. Choose a fresh root when changing
inputs; runtime source versions are recorded separately for each attempt.

| Case ID | Fixed nx | Endpoint | Primary quality metric |
|---|---:|---|---|
| `linearWave` | 128 | t=1 | Velocity L1 / imposed wave amplitude |
| `sod` | 400 | t=.15 | Volume-weighted density L1 |
| `sod2d` | 160 | t=.15 | Volume-weighted density L1 |
| `noh` | 256 | t=.6 | Absolute relative post-shock density error |
| `sedov2d` | 96 | Shock reaches radius .8 | Volume-weighted density L1 within 1.15 shock radii |
| `sedov3d` | 40 | Shock reaches radius .8 | Same radial metric in 3D |
| `gresho` | 64 | t=3 | Tangential velocity L1 within radius .5 |
| `kelvinHelmholtz` | 128 | t=2.5 | Mode amplitude/mixing diagnostics, unranked |
| `khSmooth` | 128 | t=2.5 | Same diagnostics with smooth initial density |

`nx` is the case sampler's resolution, not always the actual particle count.
Native samplers can differ between methods. Initial physical arrays, their
hashes, actual particle counts, resolved inputs and endpoints are retained. The
linear wave explicitly uses the same analytic travelling-wave mass prescription
for every method, bypassing legacy mass relaxation. Other cases use native
initialization; identical particle arrays are not assumed from equal nx.

Presets: `Monaghan` (constant AV), `Monaghan-CD` (Cullen–Dehnen switch), `CompSPH`,
`CRKSPH`, `PESPH-E`, `PESPH-A`, `MFM`, `MFV`, `GSPH`, `InutsukaGSPH`.
The Monaghan rows differ only in the declared switch; remaining native diffusion
defaults are retained. The CD row uses the native switch defaults, including its
uncorrected detector gradient. Full effective settings are saved in `resolved.json`.

**PESPH-A explicitly opts into experimental `SwiftLocal` dissipation.** It is
labelled in the figure, and its energy defects remain measured. This does not
promote dissipative A to a recommended default. Noh and Sedov use a common
`uAmbient=1e-8` for all methods. Cold-state failures remain failures; no floors,
automatic CFL retries or replacement presets are introduced.

Common controls are B7, n_h=4, eager execution and FP32. Each method retains its
native timestep/support policy and package time driver; RK2 is requested where
the ODE-integrator contract applies. MFM/MFV retain their own CFL controls.
The first timestep is physically limited, and the final timestep is capped at
the endpoint. Use `--precision float64` or `--device cpu` **at plan time** for a
separate cohort. `--smoke` plans only three coarse steps and never generates
endpoint rankings. Resolutions and endpoints are not changed by run filters.

## Select what to execute

```bash
# One case across every planned method
python scripts/run_compressibleMatrix.py run --cases sod

# Every planned case for one method
python scripts/run_compressibleMatrix.py run --methods CRKSPH

# Intersection: two cases across three selected methods
python scripts/run_compressibleMatrix.py run \
  --cases noh gresho --methods Monaghan Monaghan-CD CRKSPH

# Full planned matrix: only run when the GPU is available for the whole study
python scripts/run_compressibleMatrix.py run
```

Pass `--root` to every command if using a nondefault root. `--schemes` aliases
`--methods`. Filters are checked against the saved manifest; typos fail loudly.
You may also filter `plan` to create a smaller study.

Existing attempts are skipped, including failures. `--rerun` appends a fresh
attempt for every selected cell and selects that new attempt for future reports:

```bash
python scripts/run_compressibleMatrix.py run --cases noh --methods CRKSPH --rerun
```

Every attempt gets an immutable `runs/<run-id>/` directory. No old attempt is
deleted. Workers write logs and partial diagnostic history, so failures and
timeouts retain evidence. The controller continues after a failed cell and
returns nonzero if a new attempt failed. `--timeout` sets a per-cell limit in
seconds (default 14400); termination includes child processes. Ctrl-C stops the
current worker and the queue. A root lock refuses concurrent controllers.
Inspect the recorded host/PID before manually removing a stale `.run.lock`.

## Report, rerun and compare parameter trials

```bash
python scripts/run_compressibleMatrix.py report
python scripts/run_compressibleMatrix.py report --cases sod noh gresho
```

Reports go to new `comparisons/<comparison-id>/` directories. Each contains:

- `matrix.png` / `matrix.svg`: quality and loop-cost panels, independently ranked
  within each column. Cells contain ranks/values/status, **no dates or commits**.
- `index.html`: figure, metric definitions and a separate provenance table.
- `metrics.json` / `metrics.csv`: all diagnostics, not only the primary score.
- `provenance.csv`: case/preset/run ID, date, GitHub commit URL, dirty state,
  Core/integrator commits, device, precision and links to exact inputs/evidence.
- `comparison.json`: explicit result paths and input hashes for every selected
  cell. Rebuilding this manifest never picks a best run or the newest run.

Pending cells remain visible. Stopped runs show the fraction of target physical
time reached and cannot win endpoint ranks. Smoke runs remain unranked. Runs
whose source files changed during execution are marked invalid for comparison.
Missing metrics are not converted to zeros. Raw errors have their documented
units; the figure does not imply a shared percentage scale or absolute pass.

KH mode amplitude is printed as **diagnostic, unranked**. The bundled KH initial
conditions are not a matched McNally reference; larger amplitude is not treated
as automatically better accuracy. Energy, mode history and grid mixing remain
available for investigating outliers. Add a separately validated reference
protocol before interpreting these columns as accuracy rankings.
Grid mixing uses initial element labels, not an evolved conservative tracer;
MFV's mass exchange therefore prevents interpreting it as physical tracer mixing.

Regenerate a frozen view into a new directory, or explicitly select an earlier
attempt for future views:

```bash
python scripts/run_compressibleMatrix.py report \
  --comparison .tmp/compressible-matrix/comparisons/COMPARISON_ID/comparison.json \
  --out .tmp/compressible-matrix/rebuilt-view
python scripts/run_compressibleMatrix.py select \
  --cases noh --methods CRKSPH --run-id RUN_ID
```

A parameter trial uses a distinct preset ID. Save a JSON file such as:

```json
{
  "CRK-lowCFL": {
    "base": "CRKSPH",
    "label": "CRKSPH CFL .075",
    "simulation": {"cflFactor": 0.075}
  }
}
```

```bash
python scripts/run_compressibleMatrix.py plan --root .tmp/compressible-trial \
  --presets /path/to/presets.json --methods CRK-lowCFL --cases sod noh gresho
python scripts/run_compressibleMatrix.py check --root .tmp/compressible-trial
python scripts/run_compressibleMatrix.py run --root .tmp/compressible-trial
python scripts/run_compressibleMatrix.py report --cases sod noh gresho \
  --include .tmp/compressible-trial
```

Overrides use the existing typed case/solver schema; unsupported controls fail
`check` or worker setup. Trial roots can be included as additional figure rows.
Duplicate preset IDs, changed case definitions, mixed precision and mixed
smoke/full plans are rejected. The exact overridden resolution, physical
parameters and numerical settings remain in input/resolved records; inspect
them before treating an intentionally changed trial as a matched comparison.
Presets cannot override nx, dimension, domain extent/periodicity, endpoint or
step count. Change the case catalog for a separate fixed-resolution study;
`--smoke` is the explicit exception for plumbing checks. Physical case-parameter
overrides, if used, must be distinguished from numerical-parameter experiments.

## Timing and retained evidence

No rendering, video, full trajectory export or daemon work runs in this
protocol. Every attempt retains initial/final NPZ arrays and diagnostic history.
Full runs first execute a separate ten-step warmup, then rebuild the measured
initial state with the same seed. Warmup time/outcome is retained separately;
the measured trajectory does not start from an advanced warmup state. This
warms initial code paths, not every branch a later shock/geometry may encounter.
Smoke runs omit that warmup.
Loop wall time excludes case construction but includes stepping, timestep
selection, diagnostics and the shared recording overhead. `totalRunWallSeconds`
includes measured-run initialization and any remaining first-use compilation;
independent warmup cost is in `warmup.wallSeconds`. Runner integrator times and
mean/median after ten warmup steps are reported separately. CUDA events inherit
the existing runner's timing convention. Peak allocated/reserved Torch GPU
memory includes initialization; it is not total device memory or all Warp
allocations. CPU GPU-memory fields remain null.

Per-attempt provenance includes Python/Torch/Warp versions, device, Git commits
and dirty flags for the frontend/Core/integrators/plotting checkouts. Python
source contents and hashes are archived in `sources.tar.gz` / `sourceHashes.json`;
tracked diffs are retained as patches. This preserves untracked Python modules
and dirty implementations alongside their base commit. Source hashes are checked
again after the run. Native installed binaries and the complete environment are
not bundled. Hardware contention is not automatically certified. Reports warn
when selected records mix hardware or Torch/Warp versions; their cost ranks
require inspection of that provenance.

Generated data stays under Git-ignored `.tmp/`. Preserve the whole root, including
its runs, when retaining or moving a study. Comparison manifests currently use
absolute result paths; update those paths explicitly if relocating the archive.
