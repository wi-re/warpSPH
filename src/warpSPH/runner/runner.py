"""The step loop, once.

`examples/compressible/01-sod/sod_1d.py`,
`examples/incompressible/01-taylor-green-vortex.py` and
`datagen/weaklyCompressible/generator.py` each carried their own copy of: build
config, unpack ``buildScheme``, initialize state, loop the integrator, time it,
accumulate diagnostics, plot every N, export every M, encode a video. This
module is that code, parameterised by a :class:`~warpSPH.runner.case.Case`.
"""

from __future__ import annotations

import itertools
import os
import sys
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from ..configurations import buildConfig
from ..enumTypes import (ArtificialCompressibleSPHScheme, CompressibleSPHScheme,
                         IncompressibleSPHScheme, WaveEquationScheme,
                         WeaklyCompressibleSPHScheme)
from ..io.hdf5 import createOutFile
from ..io.export import exportSimulationSystem, prepExport, writeFrame, writeInitialData
from ..schemes import buildScheme
from warpSPHIntegrators import get_tagged_attr
from ..utils import buildDomainDescription
from .case import Case, RunContext
from .caseSpec import CaseSpec
from .display import closeWindow, holdWindow
from .media import encodeFrames
from .report import describeRun, quietedWarp, reportRun

__all__ = ['RunResult', 'run', 'buildContext', 'resolveEnum']


@dataclass
class RunResult:
    """What a run produced. `trajectory` is one dict of diagnostics per step."""

    ctx: RunContext
    state: Any
    trajectory: List[Dict[str, float]] = field(default_factory=list)
    exportPath: Optional[str] = None
    videoPath: Optional[str] = None
    nSteps: int = 0
    diverged: bool = False
    #: Wall-clock seconds for the whole run, setup included.
    wallTime: float = 0.0

    def series(self, key: str) -> np.ndarray:
        """One diagnostic across the whole run, as an array."""
        return np.array([row[key] for row in self.trajectory if key in row])


def resolveEnum(enumClass, value):
    """Case-insensitive name lookup, passing through values already resolved."""
    if value is None or isinstance(value, enumClass):
        return value
    for member in enumClass:
        if member.name.lower() == str(value).lower():
            return member
    raise ValueError(
        f'Invalid {enumClass.__name__} {value!r}. Valid options are: '
        f'{[m.name for m in enumClass]}'
    )


def _resolveScheme(name: str):
    """Map a scheme name onto whichever of the five scheme enums owns it."""
    for enumClass in (CompressibleSPHScheme, WeaklyCompressibleSPHScheme, IncompressibleSPHScheme,
                      ArtificialCompressibleSPHScheme, WaveEquationScheme):
        for member in enumClass:
            if member.name.lower() == str(name).lower():
                return member
    raise ValueError(f'Unknown scheme {name!r}.')


def buildContext(case: Case, spec: CaseSpec) -> RunContext:
    """Resolve a spec into a config, a scheme, and a populated context."""
    # Idempotent, and cheap once done. Running a case module directly imports
    # warpSPH without going through warpSPHBootstrap, so warp would otherwise
    # still be uninitialized here and every kernel launch would fail.
    import warp as wp
    wp.init()

    from warpSPHCore.type_config import get_torch_precision
    from warpSPHIntegrators import IntegrationSchemeType
    from warpSPHCore import (GradientScheme, KernelFunctions, LaplacianScheme,
                             SupportScheme)
    from ..geometry import SamplingScheme

    device = torch.device(spec.device) if spec.device else (
        torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'))
    dtype = get_torch_precision()

    domain = buildDomainDescription(spec.L, spec.dim, spec.periodic, device, dtype)

    config, integrator = buildConfig(
        domain=domain,
        dim=spec.dim,
        kernel=resolveEnum(KernelFunctions, spec.kernel),
        n_h=spec.n_h,
        calibrateNormalization=spec.calibrateNormalization,
        supportMode=resolveEnum(SupportScheme, spec.supportMode),
        gradientMode=resolveEnum(GradientScheme, spec.gradientMode),
        laplacianMode=resolveEnum(LaplacianScheme, spec.laplacianMode),
        integrationScheme=resolveEnum(IntegrationSchemeType, spec.integrationScheme),
        samplingScheme=resolveEnum(SamplingScheme, spec.samplingScheme),
        verletScale=spec.verletScale,
        device=device,
        dtype=dtype,
        dt=spec.dt,
        minDt=spec.minDt,
        maxDt=spec.maxDt,
        adaptiveDt=spec.adaptiveDt,
        cflFactor=spec.cflFactor,
        nx=spec.nx,
        dx=spec.L / spec.nx,
    )

    scheme = _resolveScheme(spec.scheme or case.scheme)
    bundle = buildScheme(scheme)

    return RunContext(
        spec=spec,
        case=case,
        config=config,
        integrator=integrator,
        schemeConfig=bundle.SimulationConfig(),
        scheme=scheme,
        device=device,
        dtype=dtype,
        bundle=bundle,
    )




def _scalar(value) -> float:
    return value.detach().cpu().item() if isinstance(value, torch.Tensor) else float(value)


class _Timer:
    """CUDA-event timing where available, wall clock otherwise."""

    def __init__(self, device):
        self.cuda = torch.cuda.is_available() and device.type == 'cuda'

    def __enter__(self):
        if self.cuda:
            self._begin = torch.cuda.Event(enable_timing=True)
            self._end = torch.cuda.Event(enable_timing=True)
            self._begin.record()
        else:
            self._begin = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.cuda:
            self._end.record()
            torch.cuda.synchronize()
            self.elapsed_ms = self._begin.elapsed_time(self._end)
        else:
            self.elapsed_ms = (time.perf_counter() - self._begin) * 1000.0
        return False


def run(case: Case, spec: Optional[CaseSpec] = None, **overrides) -> RunResult:
    """Run `case` under `spec`, returning the trajectory and final state.

    Keyword `overrides` are applied on top of `spec` (or on top of the case's
    own defaults when no spec is given), so a test can say
    ``run(tgvCase, nx=32, nSteps=20)``.
    """
    if spec is None:
        spec = CaseSpec(caseName=case.name, scheme=case.scheme, params=dict(case.params))
        spec = spec.merged(**case.defaults)
    if overrides:
        spec = spec.merged(**overrides)

    startedAt = time.perf_counter()
    with quietedWarp(spec.quiet):
        return _run(case, spec, startedAt)


def _run(case: Case, spec: CaseSpec, startedAt: float) -> RunResult:
    ctx = buildContext(case, spec)

    if case.configureScheme is not None:
        case.configureScheme(ctx)

    system = case.buildSystem(ctx)
    if case.initialConditions is not None:
        case.initialConditions(ctx, system)

    # Setup may replace the spec -- Kidder only learns its time limit once the
    # analytic collapse time exists -- so the loop reads it back from the
    # context rather than from the local it started with.
    spec = ctx.spec

    runningState = system.initializeNewState()

    # --- output setup -------------------------------------------------------
    outFile = None
    groups = None
    if spec.store or spec.plot:
        ctx.exportPath = prepExport(spec.caseName, ctx.config, ctx.schemeConfig,
                                    ctx.scheme, ctx.exportFunction,
                                    exportRoot=spec.exportRoot)
        spec.save(os.path.join(ctx.exportPath, 'caseSpec.json'))

    if spec.plot and case.setupPlot is not None:
        ctx.imagePath = os.path.join(ctx.exportPath, 'images')
        os.makedirs(ctx.imagePath, exist_ok=True)
        ctx.scratch['plot'] = case.setupPlot(ctx, runningState)

    extraData = case.extraData(ctx, runningState) if case.extraData is not None else {}

    if spec.store and spec.storeMode == 'trajectory':
        outFile = createOutFile(ctx.exportPath)
        groups = writeInitialData(ctx.exportPath, outFile, ctx.scheme, ctx.config,
                                  ctx.schemeConfig,
                                  SimpleNamespace(exportInterval=spec.exportInterval),
                                  runningState, extraData=extraData,
                                  extraFields=case.extraFields)

    # `config.dt` is only final once the case has configured it -- weakly
    # compressible cases derive it from the sound speed during setup.
    if ctx.config.dt is None:
        raise ValueError(
            'config.dt is unset after case setup; set spec.dt or have the case '
            'derive it (e.g. via setupWeaklyCompressibleTimestep).'
        )
    dt = _scalar(ctx.config.dt)
    # A case with a `timestep` hook re-picks dt every step, so a step count
    # derived from the initial dt would be wrong -- those runs are bounded by
    # simulated time instead, exactly as the notebooks' `while t < tLimit` was.
    nSteps = spec.nSteps if spec.nSteps is not None else int(spec.tLimit / dt)
    timeLimited = spec.nSteps is None and case.timestep is not None
    storeSteps = max(1, int(spec.exportInterval / dt)) if spec.storeMode == 'trajectory' \
        else max(1, spec.storeInterval)

    if not spec.quiet:
        describeRun(ctx, runningState, nSteps, timeLimited)

    if spec.store and spec.storeMode == 'states':
        exportSimulationSystem(ctx.exportPath, 'initialState', ctx.scheme, runningState,
                               exportAdjacency=False, stages=None,
                               exportStagesAdjacency=False,
                               extraData=dict(extraData, frame_num=0))

    result = RunResult(ctx=ctx, state=runningState, exportPath=ctx.exportPath)
    if case.diagnostics is not None:
        result.trajectory.append(dict(case.diagnostics(ctx, runningState), step=-1, t=0.0,
                                      stepTime_ms=0.0))

    stepResult = None

    showProgress = spec.progress if spec.progress is not None else sys.stderr.isatty()
    steps, progress = _stepIterator(nSteps, spec.tLimit, timeLimited,
                                    showProgress and not spec.quiet)
    for i in steps:
        with _Timer(ctx.device) as timer:
            stepResult = ctx.integrator.function(
                state=runningState,
                f=ctx.stepFunction,
                dt=ctx.config.dt,
                config=ctx.config,
                # Forwarded to the step function *and* to `system.finalize`, so
                # this is what makes --verbose reach the scheme's own reporting.
                verbose=spec.verbose,
                schemeConfig=ctx.schemeConfig,
            )
        runningState = stepResult.state
        # Scheme-specific extra step data that doesn't fit the state (e.g.
        # ACSPH's `update.pseudoIterations`/`.epsilonV`, set ad hoc on the
        # `ArtificialCompressibleSystemUpdate` `schemes/artificialCompressible.py`
        # returns) rides on the last stage's `update` object, which `diagnostics`/
        # `postStep` cannot otherwise reach -- neither gets `stepResult`, only
        # `runningState`. Stashed here, read with `getattr(..., None)` by anything
        # that wants it; `None` for every scheme that sets nothing extra.
        ctx.scratch['lastStageUpdate'] = stepResult.stages[-1].update if stepResult.stages else None

        if case.postStep is not None:
            case.postStep(ctx, runningState, i)
        if case.timestep is not None:
            ctx.config.dt = case.timestep(ctx, runningState)

        t = _scalar(runningState.t)
        row = {'step': i, 't': t, 'stepTime_ms': timer.elapsed_ms}
        if case.diagnostics is not None:
            row.update(case.diagnostics(ctx, runningState))
        result.trajectory.append(row)

        if progress is not None:
            if timeLimited:
                progress.n = min(progress.total, int(t / spec.tLimit * progress.total))
            progress.set_description(
                _describeStep(i, None if timeLimited else nSteps, row, spec.tLimit))

        if timeLimited and t >= spec.tLimit:
            _plotAndStore(ctx, case, spec, runningState, stepResult, i, extraData,
                          groups, storeSteps, final=True)
            break

        _plotAndStore(ctx, case, spec, runningState, stepResult, i, extraData,
                      groups, storeSteps, final=(not timeLimited and i == nSteps - 1))

        # Read by tag rather than the `velocities` field name: every fluid
        # scheme's velocity field happens to be named that, but the wave
        # scheme's is `v` (tagged `'velocity'` so it rides the same
        # position/velocity integrator machinery under a different name).
        velocities = get_tagged_attr(runningState.state, tag='velocity')
        # `isfinite` (not `isnan`): the dfsphReference late-time failure can
        # collapse into a degenerate uniform-density state with inf velocities
        # (DFSPH_IMPROVEMENT_PLAN.md Part 29) where no NaN ever appears and
        # the run would otherwise report `diverged=False`.
        if torch.any(~torch.isfinite(velocities)):
            print(f'non-finite velocities detected at step {i}; stopping.')
            result.diverged = True
            break

    if progress is not None:
        progress.close()

    result.state = runningState
    result.nSteps = len(result.trajectory) - (1 if case.diagnostics is not None else 0)

    if spec.store and spec.storeMode == 'states' and stepResult is not None:
        exportSimulationSystem(ctx.exportPath, 'finalState', ctx.scheme, runningState,
                               exportAdjacency=False, stages=stepResult.stages,
                               exportStagesAdjacency=True,
                               extraData=dict(extraData, frame_num=result.nSteps))
    if outFile is not None:
        outFile.close()

    if spec.video and ctx.imagePath is not None:
        result.videoPath = encodeFrames(ctx.imagePath, ctx.exportPath)

    _teardownPlot(ctx)

    result.wallTime = time.perf_counter() - startedAt
    if not spec.quiet:
        reportRun(result, result.wallTime)

    return result


def _teardownPlot(ctx: RunContext) -> None:
    """Hold the final figure if asked, then release it.

    Releasing matters because the figures are pyplot-managed: a process that
    runs several cases would otherwise keep every one of their windows alive.
    """
    handle = ctx.scratch.pop('plot', None)
    if handle is None:
        return
    holdWindow(ctx, handle)
    closeWindow(handle)


def _plotAndStore(ctx: RunContext, case: Case, spec: CaseSpec, state, stepResult,
                  i: int, extraData: Dict[str, Any], groups, storeSteps: int,
                  final: bool) -> None:
    """The per-step output side of the loop: plot every N, export every M."""
    if spec.plot and case.updatePlot is not None and i > 0 and \
            (i % spec.plotInterval == 0 or final):
        case.updatePlot(ctx, state, ctx.scratch.get('plot'), i)

    if spec.store and (i % storeSteps == 0 or final):
        frameExtra = dict(extraData,
                          **(case.extraData(ctx, state) if case.extraData else {}),
                          frame_num=i)
        if spec.storeMode == 'trajectory':
            writeFrame(groups, i, stepResult.state, stepResult.stages,
                       config=ctx.config, schemeConfig=ctx.schemeConfig,
                       uniqueParticles=True, writeStages=False,
                       extraFields=case.extraFields)
        else:
            exportSimulationSystem(ctx.exportPath, f'state_{i:04d}', ctx.scheme,
                                   state, exportAdjacency=False,
                                   stages=stepResult.stages,
                                   exportStagesAdjacency=True, extraData=frameExtra)


def _stepIterator(nSteps: int, tLimit: float, timeLimited: bool, enabled: bool):
    """The loop's index source, plus the tqdm handle to drive (or `None`).

    A time-limited run cannot know its step count up front, so its bar is a
    fixed 1000-tick scale over simulated time -- the same trick the notebooks
    used -- while a step-limited run counts steps directly.
    """
    steps = itertools.count() if timeLimited else range(nSteps)
    if not enabled:
        return steps, None
    try:
        from tqdm.autonotebook import tqdm
    except ImportError:
        return steps, None
    if timeLimited:
        return steps, tqdm(total=1000, leave=True)
    bar = tqdm(total=nSteps, leave=True)
    return _counting(steps, bar), bar


def _counting(steps, bar):
    for i in steps:
        yield i
        bar.update(1)

import warp as wp
def _describeStep(i: int, nSteps: Optional[int], row: Dict[str, float],
                  tLimit: float) -> str:
    # A time-limited run has no meaningful step total, so it counts time.
    parts = [f'{i + 1}/{nSteps}' if nSteps is not None else f'{i + 1}',
             f't={row["t"]:.4f}' + ('' if nSteps is not None else f'/{tLimit:.4g}')]
    parts += [f'{k}={v:.4g}' for k, v in row.items()
              if k not in ('step', 't', 'stepTime_ms') and isinstance(v, (int, float))]

    current_memory_allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # in MB
    current_memory_reserved = torch.cuda.memory_reserved() / (1024 ** 2)

    warp_memory = wp.get_mempool_used_mem_current() / (1024 ** 2)  # in MB

    if i % 100 == 0:
        torch.cuda.empty_cache()


    # tq.n = min(1000, int(t / spec.tLimit * 1000))
    # tq_text = f"t: {t:.4f}, " + ", ".join(f"{k}: {v:.4f}" for k, v in row.items())
    mem_text = f"Mem Alloc: {current_memory_allocated:.2f} MB, Mem Reserved: {current_memory_reserved:.2f} MB Warp Mem: {warp_memory:.2f} MB"

    parts.append(f'{row["stepTime_ms"]:.1f}ms')
    parts.append(mem_text)
    return ' | '.join(parts)
