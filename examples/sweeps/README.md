# Sweep configs

A run is a `CaseSpec`: a flat mapping of the fields in
[`warpSPH/runner/caseSpec.py`](../../src/warpSPH/runner/caseSpec.py), plus a
`params` block for the knobs the case itself declares. Anything omitted falls
back to the case's own default, so a sweep file only states what it changes.

```bash
warpsph-run sod      --config examples/sweeps/sod_highres.yaml
warpsph-run tgv      --config examples/sweeps/tgv_nu.yaml --nx 128
warpsph-run dambreak --config examples/sweeps/dambreak_obstacle.yaml
warpsph-run dambreak --config examples/sweeps/marrone34_sharp_edge.yaml       # Marrone 2011 Fig. 19
warpsph-run dambreak --config examples/sweeps/marrone34_rounded_corner.yaml   # ... just the rounded corner
```

CLI flags override the file, and the file overrides the case defaults — so one
config plus a varying `--nx` is a resolution study. `--saveConfig out.yaml`
writes the fully resolved spec back out, which is the easiest way to see what a
run actually used; every stored run also drops a `caseSpec.json` next to its
`config.json`.
