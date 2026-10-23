# Fluid comparison tools

The comparison tools run declared cases sequentially and keep measurements,
source snapshots and failure evidence beside each run. Generated output is
local work under the Git-ignored `.tmp/` directory; it is not documentation.

## Generate and run a comparison

```bash
python scripts/run_fluidMatrix.py plan --root .tmp/fluid-matrix
python scripts/run_fluidMatrix.py run --root .tmp/fluid-matrix
python scripts/analyze_fluidMatrix.py --root .tmp/fluid-matrix
python scripts/audit_fluidMatrix.py --root .tmp/fluid-matrix
python scripts/plot_fluidMatrixRanking.py --root .tmp/fluid-matrix
```

`plan` writes `ROOT/manifest.json`. Inspect the declared support, initialization,
viscosity, boundary representation, timestep limits and endpoints before
running. Use `--schemes` to select a subset and `--manifest` for an explicit
saved plan. Use a fresh output root for a new comparison; preserve prior
measurements when changing a numerical preset. Reports, ranking figures,
metrics, particle data and movies remain under the selected output root.

The runner respects physical timestep limits from the first step. Fixed-step
requests provide a ceiling. The velocity stop is 20 times the case's declared
reference scale. Unsupported combinations and failed or partial attempts
remain visible in the results.

## Read the results

Completion is separate from accuracy. A stopped run cannot be ranked as an
endpoint result, and missing measurements must not receive a numerical score.
When combining runs, preserve each row's date, source root and numerical
settings. Pressure probes, viscosity and sampling conventions can differ
between schemes; inspect those declarations before interpreting close ranks.

The ranking tool writes its metric definitions alongside the figure. Decaying
vortex retention is a relative energy criterion, with growth penalized.
Hydrostatic pressure, wall loads, viscous decay and impact traces have distinct
metrics. Runtime includes initialization, diagnostics and output work; it is
not an isolated solver benchmark.

Figure legends and titles describe the physical settings and use explicit
units. Keep run identifiers in machine-readable metadata.

## Preserve a baseline

```bash
python scripts/preserve_fluidMatrix.py --root .tmp/fluid-matrix \
  --archive .tmp/archives/fluid-matrix.tar.zst
```

The preservation tool checks archive contents and writes receipts beside the
archive and in the run's preservation directory. It does not write generated
receipts into `docs/`.
