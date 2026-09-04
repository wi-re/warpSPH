# Compressible examples: notebook-migration status

**Done.** All 13 compressible examples (14 cases, since `06-sedov` and
`14-triplePoint` each cover multiple variants in one directory) are in the
current style. This file is kept as a record of what that style is and how
each case got there, in case a future case in this family needs the same
treatment as a template. `../../PORTING_EXAMPLES.md` has the general
procedure and the *why* behind each convention; `../../docs/historic_plans/CLEANUP_PLAN.md` has
the wider cleanup context this was part of — the non-compressible notebooks
(`examples/weaklyCompressible/`, `examples/incompressible/`) are a separate,
still-open item there.

## Target end-state

Every case is:

- `src/warpSPH/cases/<name>.py` — the `Case` (geometry/IC, diagnostics, plot
  hooks, defaults).
- `examples/compressible/<slot>-<name>.py` (or `<slot>-<name>/<name>_<variant>.py`
  for a multi-variant case) — a thin `caseMain()` wrapper.
- `examples/compressible/<slot>-<name>.ipynb` (or the per-variant equivalent)
  — the *fat* notebook: parameters cell, IC built through the real case code,
  an **unrolled step loop** with a `# <-- hook point` comment, plots called
  directly rather than through `case.setupPlot`/`updatePlot`.

A case only gets its own `<slot>-<name>/` directory if it has more than one
variant worth showing side by side (different dimensions, different sampling,
a resume/backprop demo, ...). A single-variant case stays a flat numbered
file.

## The two notebook shapes

- **`profilePlot`** (1D scatter-against-`x`, matplotlib): `01-sod/`,
  `02-linear-wave.ipynb`, `03-kidder-isentropic-compression.ipynb`,
  `04-noh-implosion.ipynb`, `05-woodward-colella.ipynb`, `06-sedov/`. A
  notebook calls the case's exported `draw<Case>` (e.g. `drawKidder`,
  `drawWoodwardColella`) directly instead of `case.setupPlot`/`updatePlot`,
  which go through `openWindow`/`pumpEvents` and do not live-update reliably
  inside a Jupyter cell in this environment.
- **`particlePlot`** (2D field view, vispy via `visualizeWithFallback`):
  `08-hydrostatic.ipynb` through `13-rayleigh-taylor.ipynb`,
  `14-triplePoint/`. `08` is the pilot: `cases/plotting.py` gained
  `buildFieldPlotter`/`refreshFieldPlotter`, the window/event-loop-free core
  of `particlePlot`'s `setupPlot`/`updatePlot` (mirroring `profilePlot`'s
  `draw`). Each case exports its own `Field` list (`HYDROSTATIC_FIELDS`,
  `GRESHO_FIELDS`, `YEE_FIELDS`, `SHEARING_NOH_FIELDS`,
  `KELVIN_HELMHOLTZ_FIELDS`, `RAYLEIGH_TAYLOR_FIELDS`, `TRIPLE_POINT_FIELDS`)
  so the notebook passes it to `buildFieldPlotter`/`refreshFieldPlotter`
  rather than re-deriving it. `particlePlot`'s own public signature never
  changed, so no case file needed edits beyond exporting its list.

Whether the step loop is `while t < tLimit` or a fixed `range(nSteps)`
depends only on whether the `Case` has a `timestep` hook (adaptive dt,
time-limited) — nothing about 1D vs. 2D or `profilePlot` vs. `particlePlot`.

## Full case list

| slot | case(s) | shape | loop | notes |
|---|---|---|---|---|
| 01 | `sod`, `sod2d`, `sod3d` | `profilePlot` | `while` (`sod2d`/`3d` only) | the pilot — `01-sod/`. Also has `sod_resume.ipynb`/`.py` (trajectory-export demo) and `sod_backprop.ipynb`. |
| 02 | `linearWave` | `profilePlot` | `range` | |
| 03 | `kidder` | `profilePlot` | `while` (`timestep` + `postStep`) | boundary bands driven from the analytic solution every step |
| 04 | `noh` | `profilePlot` | `range` | |
| 05 | `woodwardColella` | `profilePlot` | `while` (`timestep`, no `postStep`) | `drawWoodwardColella` exported (was the unexported `_draw`) |
| 06 | `sedov` (`--dim 1/2/3`) | `profilePlot` (radial) | `while` (2D/3D goal-radius stop) | `06-sedov/` — one `Case`, three files differing only in resolution/`caseName` |
| 08 | `hydrostatic` | `particlePlot` | `range` | **the pilot for this shape**; see above |
| 09 | `greshoVortex` | `particlePlot` | `range` | steady state — velocities/pressures should not drift |
| 10 | `yeeVortex` | `particlePlot` | `range` | shell-sampled, not lattice; Dirichlet buffer rings installed once in `buildSystem` |
| 11 | `shearingNoh` | `particlePlot` | `range` | velocity panel plots the *x*-component, not magnitude — the shear is the point |
| 12 | `kelvinHelmholtz` | `particlePlot` | `range` | diverging `RdBu` density colour map |
| 13 | `rayleighTaylor` | `particlePlot` | `range` | fixed `vMin`/`vMax` so colour scale doesn't auto-range as the instability grows |
| 14 | `triplePoint` (`--equalMass`/`--no-equalMass`) | `particlePlot` | `while` (`timestep`, no `postStep`) | `14-triplePoint/` — one `Case`, merged from slots 14/15 the way `06-sedov` merged 06/07; `triplePoint_equalSpacing`/`triplePoint_equalMass` |

## Verification

Each case was checked with `jupyter nbconvert --to notebook --execute` on a
small-`nx`/short-`tLimit` copy (proves it runs, not that the numbers are
right — see `PORTING_EXAMPLES.md` §5) and `python scripts/check_imports.py
--static`. Whole-suite pass after the last case landed:
`bash scripts/run_tests.sh` and `python scripts/run_sweep.py` (full
registry, no `--cases`).
