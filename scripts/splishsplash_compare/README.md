# SPlisHSPlasH ↔ warpSPH hydrostatic-column comparison

## Install (both fixed)

- **base env** (`/home/lu26029/miniconda3`, py3.14): SPlisHSPlasH's editable
  install only registered the `pySPlisHSPlasH` source package (an empty
  `__init__.py`); the real bindings are the compiled `pysplishsplash*.so` at the
  repo root, which was not on `sys.path`. Fixed with
  `site-packages/SPlisHSPlasH-local.pth` -> `/home/lu26029/dev/SPlisHSPlasH`.
- **warp env** (`/home/lu26029/miniconda3/envs/warp`, py3.13): no `.so` for
  3.13 existed. Reconfigured `/home/lu26029/dev/SPlisHSPlasH/build` with
  `-DPython_EXECUTABLE=<warp python>`, built the `pysplishsplash` target
  (link needs `LIBRARY_PATH=$CONDA_PREFIX/lib` so `-ldbus-1` resolves),
  copied `build/lib/pysplishsplash.cpython-313-*.so` into the warp env
  site-packages. `import pysplishsplash` now works in both envs.
  (The shared `build/` dir is now configured for py3.13; re-run
  `cmake . -DPython_EXECUTABLE=<base python>` + rebuild if the base editable
  install ever needs refreshing.)

## Driver note

`base.setTimeStepCB()` corrupts the sim in this build (shipped `DamBreakModel_2D`
explodes at step 1 with a callback attached, runs clean without). And touching
`Simulation.getCurrent()` / field buffers **after** `base.run()` returns
segfaults (run() tears the sim down). Reliable path used here:
`base.init(sceneFile=...)` -> `setValueFloat(STOP_AT, T)` -> `base.run()` with
`enableVTKExport`, then parse the VTK frames (`run_export.py`).

Full write-up: `DFSPH_IMPROVEMENT_PLAN.md` "Parts 35–38" + `DFSPH_FINDINGS.md`
§9 (Parts 35–38), §1.12/§1.13, §2, §8.

## Files

| file | what |
|---|---|
| `hydrostatic_2d.json` | matched 2D scene: L=1 box, particleRadius=1/256 (spacing 1/128 = warpSPH dx), bottom-half fill, DFSPH, inviscid, rho0=1, Bender2019 volume-map walls |
| `run_export.py` | SPlisHSPlasH run + VTK export + parse the `\|v\|` / density series |
| `run_splish.py` | SPlisHSPlasH initial-state read (no stepping) |
| `warpsph_initial.py` | dump warpSPH `hydrostaticColumn` (nx=128) initial state |
| `compare.py` | side-by-side of the two initial states + the SPlisHSPlasH `\|v\|` series |
| `warpsph_matched.py` | warpSPH sweep: `n_h` × kernel × `calibrateRestDensity` (the "match the setup" negatives) |
| `import_and_run.py` | **the controlled test** — overwrite warpSPH's fluid rows with SPlisHSPlasH's exact positions/mass/`h` from `splish_fluid8001.npz`, cubic kernel, zero IC; run a scheme |
| `semiperiodic.py` | x-periodic / floor-wall-only `hydrostaticColumn` variant — isolates vertical stability (Part 38) |
| `one_noslip.py` | walled `omniIncompressible` × `XSPH_BOUNDARY` (penalty no-slip) sweep |
| `make_videos.py` | render every scheme (native + imported-SP + semi-periodic) → `videos/<tag>/output.mp4` |
| `*_initial.npz`, `splish_fluid8001.npz` | dumped states |
| `videos/` | the mp4/gif set |
