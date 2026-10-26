# Fixed-resolution compressible comparison

Integration ownership and the accepted-state meshless diagnostic fix are
documented in [the integrator audit](integrator_audit_2026-10-24.md).
Godunov cold-shock failures and labeled controls are covered in
[the shock audit](godunov_shock_audit_2026-10-24.md).

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

`manifest.json` freezes the inputs for 150 cells: fifteen cases and ten presets (142 supported combinations).
`check` validates typed configurations on CPU without sampling or stepping.
Planning refuses to overwrite a manifest. Choose a fresh root when changing
inputs; runtime source versions are recorded separately for each attempt.

| Case ID | Fixed nx | Endpoint | Primary quality metric |
|---|---:|---|---|
| `linearWave` | 128 | t=1 | Velocity L1 / imposed wave amplitude |
| `sod` | 800 | t=.15 | Volume-weighted density L1 |
| `sod2d` | 512 | t=.15 | Volume-weighted density L1 |
| `noh` | 256 | t=.6 | Absolute relative post-shock density error |
| `nohOpen` | 256 | t=.6 | Same shock metric with explicit open boundaries |
| `kidder` | 200 | 0.99 analytic collapse time | Interior volume-weighted density L1 |
| `sedov1d` | 800 | Shock reaches radius .8 | Volume-weighted density L1 within 1.15 shock radii |
| `sedov2d` | 256 | Shock reaches radius .8 | Volume-weighted density L1 within 1.15 shock radii |
| `sedov3d` | 64 | Shock reaches radius .8 | Same radial metric in 3D |
| `gresho` | 256 | t=3 | Tangential velocity L1 within radius .5 |
| `hydrostatic` | 100 | t=7 | Maximum pressure error from P=1 |
| `hydrostaticTwoGas` | 100 | t=7 | Maximum pressure error from P=1; gamma=1.4 and 5/3 |
| `hydrostaticAdvectedEllipse` | 100 | t=1 | Same pressure metric after one periodic crossing |
| `kelvinHelmholtz` | 256 | t=2.5 | Mode amplitude/mixing diagnostics, unranked |
| `khSmooth` | 256 | t=2.5 | Same diagnostics with smooth initial density |

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

Both Noh columns and Kidder use FP64 for every method; other columns default to FP32.
Precision appears in column labels and provenance, never mixed within a column.
The hydrostatic case is the literature dense-square pressure-equilibrium test
(rho_high=4, gamma=1.5), without gravity. Native MFM/MFV setups initialize this
contact with their number-volume geometry and evolve Kidder with prescribed
moving bands in conserved variables. New plans execute those cells. Existing
saved plans and unsupported results remain historical evidence; regenerate a
plan to include the new routes. See [meshless cases](../meshless.md) for the
boundary and precision limitations.

`hydrostaticTwoGas` selects the native `twoGas` material variant; the advected
ellipse selects `advectedEllipse`, with a common x velocity of 1. Both use
rho=1 and 4, gamma=1.4 and 5/3, and equal initial P=1. These are periodic
pressure-equilibrium contacts without gravity or a free surface. Their material
tables are resolved from the named variant and retained in the effective inputs.
PESPH, GSPH and InutsukaGSPH do not support those material configurations and
are declared unsupported; the other six presets can execute them. The existing
`hydrostatic` column retains its common gamma=1.5 definition.

`noh` retains its native boundary protocol for historical reproducibility.
`nohOpen` explicitly requests open edges for every preset. Its meshless cells
record `analytic Lagrangian inflow reservoir` in comparison and provenance
metadata; SPH cells record `native open edges`. Treat differences between those
boundary implementations as part of the comparison protocol. The finite meshless
Lagrangian domain must remain outside the analytic shock until the endpoint.

Common controls are B7, n_h=4 and eager execution. Each method retains its
native timestep/support policy and package time driver; RK2 is requested where
the ODE-integrator contract applies. MFM/MFV retain their own CFL controls.
The first timestep is physically limited, and the final timestep is capped at
the endpoint. Use `--precision float64` or `--device cpu` **at plan time** for a
separate cohort (Noh/Kidder stay FP64). `--smoke` plans only three coarse steps and never generates
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

# Fresh full-length meshless validation cohort (use a new root)
python scripts/run_compressibleMatrix.py plan --root .tmp/meshless-matrix \
  --cases hydrostatic hydrostaticTwoGas hydrostaticAdvectedEllipse kidder nohOpen \
  --methods MFM MFV
python scripts/run_compressibleMatrix.py check --root .tmp/meshless-matrix
python scripts/run_compressibleMatrix.py run --root .tmp/meshless-matrix
python scripts/run_compressibleMatrix.py report --root .tmp/meshless-matrix

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

- `matrix.pdf` / `matrix.png` / `matrix.svg`: quality and loop-cost panels, independently ranked
  within each column. Cells contain ranks/values/status, **no dates or commits**.
- `index.html`: figure, metric definitions and a separate provenance table.
- `metrics.json` / `metrics.csv`: all diagnostics, not only the primary score.
- `provenance.csv`: case/preset/run ID, date, GitHub commit URL, dirty state,
  Core/integrator commits, device, precision and links to exact inputs/evidence.
- `comparison.json`: explicit result paths and input hashes for every selected
  cell. Rebuilding this manifest never picks a best run or the newest run.

An explicitly composed comparison can group boundary variants in one displayed
case without changing saved run evidence. Set a cell's `sourceCaseId` to its
original result case ID; its input hash and method are still checked against
the original result. Use `boundaryTreatment` to record the actual boundary setup,
`marker` to flag exceptional cells in both panels, and top-level `figureNotes`
to explain the marker in the PDF and HTML. These fields are also retained in the
metric/provenance exports. Retained consolidated Noh views use open edges for
SPH and mark the historical periodic MFM/MFV cells with `‡`. New meshless runs
can instead select `params.outerBoundary=open`, which uses an analytic
Lagrangian inflow reservoir. Keep that boundary treatment in new comparison
metadata and preserve the periodic labels on retained evidence.

Pending cells remain visible. Stopped runs show the fraction of target physical
time reached and cannot win endpoint ranks. Completed endpoints with nonpositive
or nonfinite states show INVALID in the quality panel; their measured endpoint
cost is retained with a dagger. They receive no accuracy rank. Endpoint
overshoots show ENDPOINT and receive neither rank. These distinctions also
interpret retained older records without rewriting their evidence. Smoke runs remain unranked. Runs
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
Duplicate preset IDs, changed case definitions, mixed precision within a column and mixed
smoke/full plans are rejected. The exact overridden resolution, physical
parameters and numerical settings remain in input/resolved records; inspect
them before treating an intentionally changed trial as a matched comparison.
Presets cannot override nx, dimension, domain extent/periodicity, endpoint or
step count or precision. Change the case catalog for a separate fixed-resolution study;
`--smoke` is the explicit exception for plumbing checks. Physical case-parameter
overrides, if used, must be distinguished from numerical-parameter experiments.

## Timing and retained evidence

No rendering, video, full trajectory export or daemon work runs in this
protocol. Every attempt retains initial/final NPZ arrays and diagnostic history.
Full runs first execute a separate ten-step warmup, then rebuild the measured
initial state with the same seed. Warmup time/outcome is retained separately;
the measured trajectory does not start from an advanced warmup state. A warmup
exception aborts that attempt with its original error; CUDA errors are never
followed by a measured run in a potentially poisoned process. This
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

## Revised cohort and previous endpoint audit

Sedov 3D was reduced from nx=128 to nx=64 at the user's request after the
nx=128 runs proved too slow. The original manifest and attempts stay intact;
nx=64 uses a separate manifest with identical other numerical settings. Complete
the remaining non-3D cases first, then run every Sedov 3D method at nx=64.
The resulting comparison selects the nx=64 column explicitly and retains
nx=128 attempts separately, without mixing resolutions within that column.

Use a new root (for example `.tmp/compressible-matrix-20261024-v2`) for the
revised resolutions and per-case precision. Existing manifests keep their exact
original inputs. The earlier PESPH endpoint wrapper was replaced by the native
runner callback, allowing small endpoint overshoots. The revised harness caps
the native proposed timestep in diagnostics, after that callback, and snaps only
time-dtype roundoff before the runner reads the new time. No solver policy or
physical state is repaired. Earlier overshoots remain excluded from ranks.

Regression checks: `python -m pytest tests/test_compressibleMatrix.py -q` includes
short multi-step CPU runs of both PESPH variants, exercising the native runner
callback replacement and the final endpoint cap.
