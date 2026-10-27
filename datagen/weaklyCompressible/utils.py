"""Compatibility layer for weakly-compressible datagen helpers.

The source-of-truth implementations now live in warpSPH:
- case utilities: warpSPH.caseUtils
- HDF5 config serialization helpers: warpSPH.io
- case running: warpSPH.runner

The sweep helpers below (``sampleCaseSystem``, ``renderCasePreview``,
``buildMatrixSpecs``, ``previewMatrix``, ``writeSweep``, ``loadSweepIndex``)
are shared by ``obstacle_init.ipynb`` (which generates sweeps) and
``sweep_browser.ipynb`` (which browses them), so the two notebooks render a
config exactly the same way and neither carries its own copy of the SDF/region
plotting code.
"""

import itertools
import json
import os
import shutil
from dataclasses import fields as _dataclassFields

import matplotlib.pyplot as plt

from warpSPH.caseUtils import (
    SimulationProperties,
    buildPresetObstacles,
    buildObstacleSDF,
    build_sdfs,
    buildDomain,
    buildRegions,
    sampleNoise,
    setupFreestream,
    setupKolmogorov,
)
from warpSPH.io.hdf5 import copy_dict_to_h5, restore_config_from_h5
from warpSPH.runner import CaseSpec, buildContext
from warpSPH.cases.freeSurface.dambreak import dambreakCase
from warpSPH.cases.freeSurface.impact import impactCase


# Backward-compatible legacy name.
def restoreConfig_from_h5(group, indent=0):
    return restore_config_from_h5(group, indent=indent)


#: Which registered Case builds the geometry for each sweep family. The first
#: six families are all `dambreakCase` runs that differ only in the params
#: (domain topology, freestream, noise, ...); `impact` is a different Case
#: entirely (free-surface bodies colliding, no obstacle/tank).
FAMILY_CASE = {
    'dambreak': dambreakCase,
    'semiPeriodic': dambreakCase,
    'fullyPeriodic': dambreakCase,
    'openChannel': dambreakCase,
    'periodicNoise': dambreakCase,
    'kolmogorov': dambreakCase,
    'impact': impactCase,
}

#: Scatter style per `state.kinds` value (0=fluid, 1=boundary, 2=ghost/rigid).
_KIND_STYLE = {
    0: dict(color='tab:blue', label='fluid'),
    1: dict(color='tab:red', label='boundary'),
    2: dict(color='tab:green', label='rigid'),
}

_CASESPEC_FIELDS = {f.name for f in _dataclassFields(CaseSpec)}


def baseSpecFor(case, **overrides):
    """A `CaseSpec` seeded with `case`'s own defaults/params, plus overrides.

    `overrides` may mix top-level `CaseSpec` fields (e.g. `nx`) and case
    params (e.g. `obstacleType`) freely -- each is routed to the right place.
    """
    spec = CaseSpec(caseName=case.name, scheme=case.scheme, params=dict(case.params))
    spec = spec.merged(**case.defaults)
    # `params` is itself a CaseSpec field, so a caller passing `params={...}`
    # (a nested dict) and a caller passing param names directly as kwargs both
    # need to land in the same place -- pop the nested form out first.
    nestedParams = overrides.pop('params', {}) or {}
    if overrides or nestedParams:
        topLevel = {k: v for k, v in overrides.items() if k in _CASESPEC_FIELDS}
        params = {k: v for k, v in overrides.items() if k not in _CASESPEC_FIELDS}
        params.update(nestedParams)
        spec = spec.merged(**topLevel, params=params)
    return spec


def sampleCaseSystem(case, spec):
    """Build the sampled particle system for `spec`: geometry + particles only.

    This runs exactly the first two hooks `warpSPH.runner.run` calls --
    `configureScheme` then `buildSystem` -- before it would stamp initial
    velocities/energies and start stepping. A preview built this way can never
    drift from what a real run actually samples, because it *is* a prefix of
    a real run.
    """
    ctx = buildContext(case, spec)
    if case.configureScheme is not None:
        case.configureScheme(ctx)
    return ctx, case.buildSystem(ctx)


def renderCasePreview(case, spec, ax, markerSize=2.0, title=None):
    """Render one CaseSpec's sampled particles onto `ax`, coloured by kind."""
    ctx, system = sampleCaseSystem(case, spec)
    positions = system.state.positions.detach().cpu().numpy()
    kinds = system.state.kinds.detach().cpu().numpy()
    for kind in sorted(set(kinds.tolist())):
        if kind == 2:
            continue
        style = _KIND_STYLE.get(kind, dict(color='gray', label=f'kind {kind}'))
        mask = kinds == kind
        ax.scatter(positions[mask, 0], positions[mask, 1], s=markerSize, **style)
    domain = ctx.config.domain
    ax.set_xlim(domain.min[0].item(), domain.max[0].item())
    ax.set_ylim(domain.min[1].item(), domain.max[1].item())
    ax.set_aspect('equal')
    ax.set_title(title or spec.caseName, fontsize=8)
    return ctx, system


def buildMatrixSpecs(baseSpec, axes):
    """The cartesian product of `axes` (name -> list of values) as
    `(label, CaseSpec)` pairs, built on top of `baseSpec`.

    `axes` keys may be top-level `CaseSpec` fields or case params. Only axes
    with more than one value show up in the label, so a single-axis sweep
    gets a short, readable name instead of restating every fixed param.
    """
    names = list(axes.keys())
    varying = [n for n in names if len(axes[n]) > 1]
    specs = []
    for combo in itertools.product(*(axes[n] for n in names)):
        overrides = dict(zip(names, combo))
        label = '_'.join(f'{n}{overrides[n]}' for n in varying) or 'case'
        topLevel = {k: v for k, v in overrides.items() if k in _CASESPEC_FIELDS}
        params = {k: v for k, v in overrides.items() if k not in _CASESPEC_FIELDS}
        spec = baseSpec.merged(**topLevel, params=params)
        spec = spec.merged(caseName=f'{baseSpec.caseName}/{label}')
        specs.append((label, spec))
    return specs


def previewMatrix(case, baseSpec, axes, ncols=4, figsize=None, markerSize=2.0):
    """Build every combination in `axes` and render them as one grid.

    Returns `(fig, specs)`, where `specs` is the `(label, CaseSpec)` list
    `writeSweep` expects.
    """
    combos = buildMatrixSpecs(baseSpec, axes)
    n = len(combos)
    ncols = max(1, min(ncols, n))
    nrows = -(-n // ncols)
    fig, axesGrid = plt.subplots(nrows, ncols, squeeze=False,
                                 figsize=figsize or (3.2 * ncols, 3.2 * nrows))
    for (label, spec), ax in zip(combos, axesGrid.flat):
        renderCasePreview(case, spec, ax, markerSize=markerSize, title=label)
    for ax in axesGrid.flat[n:]:
        ax.axis('off')
    fig.tight_layout()
    return fig, combos


def previewSample(case, specs, n=12, ncols=4, figsize=None, markerSize=2.0):
    """Render up to `n` evenly-spaced specs from `specs` (a `(label, CaseSpec)`
    list, as from `buildMatrixSpecs`) as a grid.

    For a sweep too large to preview in full (shapes x offsets x fill ratios
    can run into the hundreds), this is the sanity check: a representative
    sample of what actually got written, not a separately hand-picked subset
    that could drift from the real sweep.
    """
    if len(specs) > n:
        step = len(specs) / n
        specs = [specs[int(i * step)] for i in range(n)]
    n = len(specs)
    ncols = max(1, min(ncols, n))
    nrows = -(-n // ncols)
    fig, axesGrid = plt.subplots(nrows, ncols, squeeze=False,
                                 figsize=figsize or (3.2 * ncols, 3.2 * nrows))
    for (label, spec), ax in zip(specs, axesGrid.flat):
        renderCasePreview(case, spec, ax, markerSize=markerSize, title=label)
    for ax in axesGrid.flat[n:]:
        ax.axis('off')
    fig.tight_layout()
    return fig


def archive(result, caseName: str, destination: str = 'compressed') -> None:
    """Move the trajectory and copy the renders into a flat dataset directory.

    Shared by `generator.py` (dambreak-family runs) and `run_sweep.py` (every
    family) so there is one archiving step, not one per runner script.
    """
    exportPath = result.exportPath
    if exportPath is None:
        return
    os.makedirs(destination, exist_ok=True)
    tag = f'{caseName}_{os.path.basename(exportPath)}'

    trajectory = os.path.join(exportPath, 'trajectory.h5')
    if os.path.exists(trajectory):
        shutil.move(trajectory, os.path.join(destination, f'trajectory_{tag}.hdf5'))

    if result.videoPath and os.path.exists(result.videoPath):
        shutil.copy(result.videoPath, os.path.join(destination, f'video_{tag}.mp4'))

    imagePath = result.ctx.imagePath
    if imagePath and os.path.isdir(imagePath):
        frames = sorted((f for f in os.listdir(imagePath)
                         if f.startswith('frame_') and f.endswith('.png')),
                        key=lambda name: int(name.split('_')[1].split('.')[0]))
        if frames:
            shutil.copy(os.path.join(imagePath, frames[0]),
                        os.path.join(destination, f'first_frame_{tag}.png'))
            shutil.copy(os.path.join(imagePath, frames[-1]),
                        os.path.join(destination, f'last_frame_{tag}.png'))


def writeSweep(family, specs, root='sweeps', casesRoot='cases', script='run_sweep.py'):
    """Write `specs` (as from `previewMatrix`/`buildMatrixSpecs`) to
    `root/family/*.json`, an `index.json` describing which params actually
    vary (read by `sweep_browser.ipynb`), and a runner loop into
    `casesRoot/family.sh` (`run_sweep.py <config>`, not `generator.py
    --config <config>` -- `generator.py` only knows how to run `dambreakCase`,
    so it can't run the `impact` family; `run_sweep.py` dispatches on
    `FAMILY_CASE` and works for all of them).

    `CaseSpec.save` writes every field, including `store`/`storeMode` -- and a
    `--config` file's fields take precedence over a runner's own `store=True`
    default (see `warpSPH.runner.caseSpec.specFromArgs`), so a spec left at
    its unset `store=False` would silently produce a run that writes nothing,
    and `impactCase` (unlike `dambreakCase`) leaves `storeMode` at its
    CaseSpec default of `'states'` (one file per step) rather than
    `'trajectory'` (one growing `trajectory.h5`) -- and `archive`/the dataset
    loader only look for the latter. A saved sweep entry exists to be run
    later and land in the dataset, so both are forced here rather than left to
    whichever case or runner happens to touch the file.
    """
    outDir = os.path.join(root, family)
    os.makedirs(outDir, exist_ok=True)
    os.makedirs(casesRoot, exist_ok=True)

    index = []
    paths = []
    for label, spec in specs:
        spec = spec.merged(store=True, storeMode='trajectory')
        path = os.path.join(outDir, f'{label or "case"}.json')
        spec.save(path)
        paths.append(path)
        scalarParams = {k: v for k, v in spec.params.items()
                        if isinstance(v, (int, float, str, bool))}
        index.append({'file': path, 'label': label, **scalarParams})

    # Data-driven, not copied from the caller's `axes` dict: a param only
    # counts as an axis if it actually differs somewhere in this sweep.
    keys = {k for row in index for k in row} - {'file', 'label'}
    axes = sorted(k for k in keys if len({row.get(k) for row in index}) > 1)
    with open(os.path.join(outDir, 'index.json'), 'w') as f:
        json.dump({'family': family, 'axes': axes, 'rows': index}, f, indent=2)

    with open(os.path.join(casesRoot, f'{family}.sh'), 'w') as f:
        for path in paths:
            f.write(f'python {script} {path}\n')

    return outDir


def loadSweepIndex(family, root='sweeps'):
    """The `index.json` `writeSweep` wrote for `family`: `{family, axes, rows}`."""
    with open(os.path.join(root, family, 'index.json')) as f:
        return json.load(f)


__all__ = [
    "SimulationProperties",
    "buildPresetObstacles",
    "buildObstacleSDF",
    "build_sdfs",
    "buildDomain",
    "buildRegions",
    "sampleNoise",
    "setupFreestream",
    "setupKolmogorov",
    "copy_dict_to_h5",
    "restore_config_from_h5",
    "restoreConfig_from_h5",
    "FAMILY_CASE",
    "baseSpecFor",
    "sampleCaseSystem",
    "renderCasePreview",
    "buildMatrixSpecs",
    "previewMatrix",
    "previewSample",
    "archive",
    "writeSweep",
    "loadSweepIndex",
]
