# datagen

Dataset generation on top of the same cases the examples run. Where
`examples/` produces *one* run to look at, this produces *many* runs to train
on: batches of parameter-varied simulations, each archived as a single
trajectory file plus preview media.

Only the weakly compressible family has a generator so far
(`weaklyCompressible/`, built on `warpSPH.cases.dambreak`).

## The pipeline

```
cases/*.sh          batches of generator.py invocations, one file per family
   |
generator.py        runs the case, then archives the result
   |
compressed/         flat dataset directory, one set of files per run
   |
compressor.py       optional: downsample a trajectory to a coarser interval
```

### `generator.py`

The simulation itself is `warpSPH.cases.dambreak` — what lives here is only the
dataset-specific part: stamping the run directory with a timestamp and the
obstacle description, then collecting the results into `compressed/`.

It takes the **same flags as any other case** (it builds its parser with
`buildArgumentParser`, so `warpsph-run dambreak --help` documents them), plus
this directory's own geometry knobs (`--obstacleType`, `--fillRatio`,
`--fluidWidth`, `--maxExtent`, `--aoa`, `--offsetX`, `--W`, ...).

```bash
python generator.py --nx 128 --plot --store
python generator.py --config sweeps/obstacle.yaml
```

Each run archives into `compressed/` as a flat set, tagged
`<caseName>_<exportDirName>`:

| file | what |
|---|---|
| `trajectory_<tag>.hdf5` | the trajectory, **moved** out of the export tree |
| `video_<tag>.mp4` | the render, if `--video` produced one |
| `first_frame_<tag>.png`, `last_frame_<tag>.png` | first and last frames |

Note that the trajectory is *moved*, not copied — after archiving, the run's
own export directory no longer holds it.

### `cases/*.sh`

Flat lists of `generator.py` command lines, one file per case family — the
parameter sweep written out longhand, so a batch is reproducible by rerunning
the file. They are not scripts with logic in them; read them as data.

| file | runs |
|---|---|
| `examples.sh` | one representative run of each family — start here |
| `dambreak.sh` | 81 |
| `kolmogorov.sh` | 192 |
| `periodic_wObstacle.sh` | 96 |
| `periodic.sh` | 48 |
| `openChannel.sh` | 44 |
| `semiPeriodic.sh` | 26 |
| `fullyPeriodic.sh` | 17 |

Run them from **inside `datagen/weaklyCompressible/`** — the commands invoke
`python generator.py` by relative path. Each line is a full simulation, so a
whole file is many GPU-hours; take the lines you want rather than running the
file end to end.

### `compressor.py`

Re-writes an existing trajectory at a coarser export interval, for when a run
was stored more finely than the dataset needs.

```bash
python compressor.py --directory compressed --exportInterval 0.01
```

### Marrone 2011 §3.4 — sharp-edged obstacle + rounded tank corner

A dense source of violent free-surface / solid-boundary interaction — a jet
ejected off a convex 45° edge, two re-entrant corners, and a smooth concave
corner in one geometry (Marrone et al. 2011 Fig. 19). Two tracked configs under
[`examples/sweeps/`](../examples/sweeps/):

| config | geometry |
|---|---|
| `marrone34_sharp_edge.yaml` | tank 10 H × 8 H, column 3 H × 2.4 H upstream, sharp-edged floor obstacle (45° edge apex at x = 5 H, back face at 7 H) + a concave quarter-circle fillet (radius H) rounding the downstream bottom corner |
| `marrone34_rounded_corner.yaml` | same tank, obstacle removed — just the fillet; the surge runs the full 10 H and impacts the smoothed corner directly |

```bash
warpsph-run dambreak --config examples/sweeps/marrone34_sharp_edge.yaml --video
warpsph-run dambreak --config examples/sweeps/marrone34_sharp_edge.yaml --nx 512
```

- **Scheme: `sun2017DeltaSPH` (δ⁺-SPH) with particle shifting on** (Sun 2017
  Eq. (7) magnitude) — validated to run this case stably (bulk ρ P99 ≤ 1.02 vs
  ~1.14 for plain δ-SPH); set `params.shifting: false` for plain δ-SPH.
- **Resolution: `nx` → H/dx = nx/8.** Ships `nx = 256` (H/dx = 32, Marrone's
  coarsest, "almost converged"); raise `nx` for higher-fidelity samples.
- **c₀ = 28.3 √(gH)** via `params.machTarget` / `referenceVelocity`; H = 1,
  g = 9.81, so t\* = t √(g/H) = t / 0.319 s and `tLimit` ≈ t\* 7.4.
- To batch either through `run_sweep.py`, expand the terse YAML to a full
  `CaseSpec` first (`warpsph-run dambreak --config <f>.yaml --saveConfig
  <f>.json`) — `run_sweep.py` loads a spec literally and does not merge the
  case's own defaults.
- Geometry: `warpSPH.caseUtils.weaklyCompressible._marroneSharpEdgeSDF`
  (obstacle preset keys `marroneSharpEdge` / `marroneRoundedCorner`). The
  δ-SPH validation story — stability, boundary penetration, convergence — is
  `DELTASPH_VALIDATION_PLAN.md` §5.2.2 and
  `scripts/probe_deltaSPHMarrone34.py`.

## Notebooks

Exploratory, and older than the `examples/` ones — these are still on the
pre-`warpSPHBootstrap` style, so they set precision by hand rather than through
`bootstrap()`.

| notebook | what |
|---|---|
| `generator.ipynb` | `generator.py`'s pipeline, step by step |
| `dataset.ipynb` | assembling and inspecting a dataset |
| `compressedLoader.ipynb` | reading `compressed/` back |
| `compressedResume.ipynb` | restarting a run from an archived trajectory |
| `obstacle_init.ipynb` | the obstacle SDF presets, visualised |
| `quartzTest.ipynb` | scratch |

## `utils.py`, `export_util.py`

Thin compatibility shims. Both just re-export from `warpSPH` — the real
implementations moved to `warpSPH.caseUtils` and `warpSPH.io`. They exist so
the notebooks' `from utils import *` keeps working; new code should import
from `warpSPH` directly.

## Output directories

`export/`, `compressed/` and the per-case preview PNGs under `cases/*/` are all
gitignored except for the PNGs, which are small and are kept as a visual index
of what each obstacle preset looks like.
