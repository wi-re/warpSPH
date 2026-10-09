# Contributing

## Setting up

The four repositories are developed together, so editable installs from a common
parent directory are the supported setup. Install `warpSPHCore` before
`warpSPH` — see [Precision](README.md#precision-and-other-things-that-bite) for
why the ordering matters at *import* time.

```bash
mkdir warp-sph-stack && cd warp-sph-stack
git clone https://github.com/wi-re/warpSPHCore
git clone https://github.com/wi-re/warpSPHIntegrators
git clone https://github.com/wi-re/warpSPHPlotting
git clone https://github.com/wi-re/warpSPH

conda create -n warp_env python=3.14 && conda activate warp_env
pip install -e warpSPHCore/ warpSPHIntegrators/ warpSPHPlotting/
pip install -e "warpSPH/[dev]"
```

A CUDA device is required for anything beyond imports.

Optionally, for the analytic boundary tests and `--wallRepresentation analytic`, install `warpSPHBoundaries` as well (`git clone https://github.com/wi-re/warpSPHBoundaries`,
`pip install -e warpSPHBoundaries/`). The `tests/test_analytic*.py` files skip without it, and `check_imports.py` only verifies its imports when it is installed.

## Notebooks: set up nbstripout first

**Do this before committing any notebook.** `.gitattributes` declares a
`filter=nbstripout` for `*.ipynb`, but a git filter only runs if it is *also*
defined in git config — and that config lives in `.git/config`, which is not
cloned. In a fresh clone the filter silently does nothing, and committing a
notebook bakes in every stored output, including megabytes of embedded PNGs.

```bash
pip install nbstripout          # included in the [dev] extra
nbstripout --install            # writes the filter into this repo's .git/config
nbstripout --status             # should say "installed in repository ..."
```

`--install` is per-clone. Run it again in every clone, and after any
`.git/config` reset.

A handful of notebooks committed before this was set up still carry their
outputs; the filter is a `clean` filter, so those keep their outputs until the
file is next staged. That is expected, not a sign the filter is broken.

## Before committing

There is deliberately **no hosted CI** — the suite and the sweep both need a
CUDA device, which makes GitHub runners a poor trade. Run the checks locally
instead.

```bash
scripts/run_tests.sh                  # 89 tests, ~2 min
scripts/check_imports.py              # every module imports; every import resolves
```

`check_imports.py` is the one to run after any rename or refactor: it imports
each module for real, then AST-scans every `.py` **and notebook cell** in the
repo for `warpSPH*` imports and verifies both the module and the imported
symbol exist. Nothing else executes function-level or notebook imports.

For a bigger change, also sweep every case:

```bash
scripts/run_sweep.py                  # every case, 5 steps each, ~4 min
scripts/run_sweep.py --cases sod noh  # or just the affected ones
```

Each case runs in its own process, sequentially, so one crashing case cannot
take the sweep down with it. Results land in `sweeps/sweep_<timestamp>/`.

If you touched a `@wp.kernel` or `@wp.func`, the gradcheck scripts are what
catch silently-wrong gradients that the forward-only physics tests cannot see:

```bash
python -m pytest tests/test_gradcheck_scripts.py     # all 15, in subprocesses
python scripts/gradcheck_deltaSPH.py                 # or one, while iterating
```

They run as subprocesses because gradcheck needs float64 and precision is baked
in at first `warpSPHCore` import — see the docstring in
[`tests/test_gradcheck_scripts.py`](tests/test_gradcheck_scripts.py).

## Adding a case

A `Case` is a name, a scheme, and a set of hooks over a `RunContext`; only
`buildSystem` is required. See [Writing a case](README.md#writing-a-case) for
the hook set, and [PORTING_EXAMPLES.md](PORTING_EXAMPLES.md) for the procedure
and the things that bite when porting an existing example or taking a case up
to 2D/3D.

Register the module name in `CASE_MODULES`
([`src/warpSPH/cases/__init__.py`](src/warpSPH/cases/__init__.py)) and it
becomes reachable as `warpsph-run <name>`.

## Regenerating example media

The GIF/MP4/PNG under `examples/*/outputs/` are produced by
[`scripts/render_examples.py`](scripts/render_examples.py), which re-runs each
example's `.py` wrapper at its shipped settings. These are full-length runs, so
re-rendering everything takes hours:

```bash
scripts/render_examples.py --list             # what would run, and where it lands
scripts/render_examples.py --only dambreak
```

Use `scripts/run_sweep.py` — not this — to check that every case still runs.
