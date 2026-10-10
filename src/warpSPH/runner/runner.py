"""The step loop, once.

`examples/compressible/01-sod/sod_1d.py`,
`examples/incompressible/01-taylor-green-vortex.py` and
`datagen/weaklyCompressible/generator.py` each carried their own copy of: build
config, unpack ``buildScheme``, initialize state, loop the integrator, time it,
accumulate diagnostics, plot every N, export every M, encode a video. This
module is that code, parameterised by a :class:`~warpSPH.runner.case.Case`.
"""

from __future__ import annotations

import collections
import itertools
import math
import os
import sys
import time
import warnings
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
from ..utils import buildDomainDescription
from .case import Case, RunContext
from .caseSpec import CaseSpec
from .display import closeWindow, holdWindow
from .media import encodeFrames
from .report import describeRun, quietedWarp, reportRun
from .velocityAlarm import VelocityAlarm, maxSpeed

__all__ = ['RunResult', 'run', 'buildContext', 'resolveEnum']

#: Window of the `CaseSpec.stallProgress` sim-time watchdog, in steps.
STALL_WINDOW_STEPS = 1000


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
    #: Why the run stopped early (a watchdog, the non-finite check, or an
    #: `onVelocityAlarm` callback); `None` when it ran to the end.
    stopReason: Optional[str] = None
    #: Every raise / escalation / clear of the velocity alarm
    #: (`runner/velocityAlarm.py`), in order; empty when it never fired.
    velocityAlarms: List[Dict[str, Any]] = field(default_factory=list)
    #: Wall-clock seconds for the whole run, setup included.
    wallTime: float = 0.0

    def series(self, key: str) -> np.ndarray:
        """One diagnostic across the whole run, as an array."""
        return np.array([row[key] for row in self.trajectory if key in row])


#: the schemes whose step consumes the analytic boundary provider (a scheme leaves the refusal when it is ported)
ANALYTIC_WALL_SCHEMES = frozenset({'deltaSPH', 'omniIncompressible', 'divergenceFree', 'dfsphReference', 'iisph'})
#: ... of which not yet usable
EXPERIMENTAL_ANALYTIC_WALL_SCHEMES = frozenset({'omniIncompressible', 'divergenceFree', 'dfsphReference', 'iisph'})


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
        densityCorrection=spec.densityCorrection,
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


def run(case: Case, spec: Optional[CaseSpec] = None, *, onVelocityAlarm=None,
        **overrides) -> RunResult:
    """Run `case` under `spec`, returning the trajectory and final state.

    Keyword `overrides` are applied on top of `spec` (or on top of the case's
    own defaults when no spec is given), so a test can say
    ``run(tgvCase, nx=32, nSteps=20)``.

    `onVelocityAlarm(ctx, state, event)` is called whenever the velocity alarm
    (`spec.velocityAlarmFactor`, `runner/velocityAlarm.py`) is raised,
    escalates or clears; `event` is a dict (`kind`, `step`, `t`, `vmax`,
    `ratio`, `scale`, `dt`, and for a raise/escalation the fastest particle's
    `index`/`uid`/`position`). Return `True` to stop the run -- anything else
    lets it continue, e.g. after switching on closer monitoring
    (`ctx.spec.plotInterval = 1`).
    """
    if spec is None:
        spec = CaseSpec(caseName=case.name, scheme=case.scheme, params=dict(case.params))
        spec = spec.merged(**case.defaults)
    if overrides:
        spec = spec.merged(**overrides)

    startedAt = time.perf_counter()
    with quietedWarp(spec.quiet):
        return _run(case, spec, startedAt, onVelocityAlarm)


def _run(case: Case, spec: CaseSpec, startedAt: float, onVelocityAlarm=None) -> RunResult:
    ctx = buildContext(case, spec)

    if case.configureScheme is not None:
        case.configureScheme(ctx)
    if spec.cudaGraph:
        ctx.schemeConfig.cudaGraph = True

    system = case.buildSystem(ctx)
    if getattr(ctx.schemeConfig, 'boundaryProvider', None) is not None:
        schemeName = getattr(ctx.scheme, 'name', ctx.scheme)
        if schemeName not in ANALYTIC_WALL_SCHEMES:
            # a scheme that does not consume the boundary provider would run with no wall at all (ANALYTIC_BOUNDARIES_PORT_SURVEY.md)
            raise NotImplementedError(f"analytic boundaries (a region with a `representation`) are hooked into the schemes {sorted(ANALYTIC_WALL_SCHEMES)} only, not {schemeName!r}")
        if schemeName in EXPERIMENTAL_ANALYTIC_WALL_SCHEMES:
            import warnings
            warnings.warn(f"analytic walls in {schemeName!r} are EXPERIMENTAL: the wall terms are in and the hydrostatic, dam-break, wedge and sloshing runs are quiet, but the Jacobi relaxation limit, the free-surface density-solve instability and the sloshing impact overshoot are open (OPEN_PROBLEMS 31)", stacklevel=2)
    if case.initialConditions is not None:
        case.initialConditions(ctx, system)

    # Setup may replace the spec -- Kidder only learns its time limit once the
    # analytic collapse time exists -- so the loop reads it back from the
    # context rather than from the local it started with.
    spec = ctx.spec

    if spec.resumeFrom:
        # Overwrite the t=0 state `case.initialConditions` just built with a
        # checkpoint's -- domain/config setup above still ran normally, so
        # this only replaces particle state and simulated time, exactly what
        # `system.initializeNewState()` (below) reads off `self.state`/`self.t`.
        # The spec must describe the SAME configuration the checkpoint was
        # written under (nx, params, scheme, ...); a mismatch loads a state
        # whose shapes/geometry don't match what the rest of the run expects.
        import h5py
        from ..io.hdf5 import loadState
        with h5py.File(spec.resumeFrom, 'r') as f:
            system.state = loadState(f['state'], ctx.config.device, type(system.state))
            system.t = float(f.attrs['time'])
        if not spec.quiet:
            print(f'-- resumed from {spec.resumeFrom} '
                 f'(t={system.t:.6f}, step offset {spec.resumeStepOffset}) --')

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
        ctx.scratch['plot'] = _setupPlot(ctx, case, runningState)

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

    # Resolved before the banner, which reports it; reads the t=0 state.
    alarm = VelocityAlarm(ctx, runningState, callback=onVelocityAlarm,
                          write=lambda message: print(message, flush=True))
    ctx.scratch['velocityAlarmMonitor'] = alarm

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
    # `stallDtSteps` watchdog state: counts consecutive steps whose *next*
    # dt landed on the floor. Reset whenever dt lifts back off it.
    dtAtFloorStreak = 0
    # `stallProgress` watchdog state: simulated time at each of the last
    # `STALL_WINDOW_STEPS` steps.
    recentTimes = collections.deque(maxlen=STALL_WINDOW_STEPS + 1)

    # An explicit `progress=True` wins over `quiet`: the probes pass both, to
    # drop the banner/report but keep the per-step rows streaming (CLAUDE.md
    # run rule 2) -- `quiet` only switches off the *default* bar.
    showProgress = spec.progress if spec.progress is not None else (
        sys.stderr.isatty() and not spec.quiet)
    steps, progress = _stepIterator(nSteps, spec.tLimit, timeLimited, showProgress)
    # Everything the loop says mid-run (stop reasons, the velocity alarm) goes
    # through `say`: with a bar up, a bare print lands on the end of the bar's
    # unfinished line (and a log shows it buried in a progress row).
    say = _writer(progress)
    alarm.write = say
    # `spec.cudaGraph`: replay whole steps from a CUDA graph where the scheme
    # allows it (utils/cudaGraph.py `GraphedIntegratorStep`; bitwise the eager
    # step, self-checked, eager fallback on any Verlet rebuild). Not with
    # trajectory/state storing, which reads the per-stage data a replayed
    # step does not reconstruct.
    stepGraph = None
    if spec.cudaGraph and not spec.store and _stepGraphable(ctx):
        from ..utils.cudaGraph import GraphedDiagnostics, GraphedIntegratorStep
        stepGraph = GraphedIntegratorStep(ctx.integrator.function, ctx.stepFunction)
        if not isinstance(ctx.config.dt, torch.Tensor) and ctx.config.dt is not None and ctx.device != 'cpu':
            # a case with a fixed host-float step (`soundSpeed` + `targetDt`, the analytic-wall cases): the replay reads dt from the device, as the adaptive cases already do
            ctx.config.dt = torch.tensor(float(ctx.config.dt), dtype=ctx.dtype, device=ctx.device)
        ctx.scratch['stepGraph'] = stepGraph
        # cases whose diagnostics have a sync-free device part replay it too
        # (`cases/dambreak.py:_diagnosticsHost`); ignored by every other case
        ctx.scratch['graphedDiagnostics'] = GraphedDiagnostics()

    if stepGraph is not None and spec.pipelineOutputs:
        # Pipelined loop: step n+1 runs on the GPU while step n's diagnostics
        # and frame are computed (see `_runPipelined`); same values, same rows.
        runningState, stepResult = _runPipelined(
            ctx, case, spec, result, stepGraph, steps, progress, runningState, nSteps,
            timeLimited, extraData, groups, storeSteps, recentTimes, alarm, say)
        steps = ()

    for i in steps:
        with _Timer(ctx.device) as timer:
            if stepGraph is not None:
                stepResult = stepGraph(runningState, ctx.config.dt, ctx.config, ctx.schemeConfig,
                                       verbose=spec.verbose)
            else:
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

        # Absolute step number: equal to `i` for a fresh run (offset 0), or
        # continues from where `spec.resumeFrom`'s checkpoint left off -- used
        # everywhere a step number is externally visible (postStep,
        # diagnostics' `step` column, checkpoint filenames) so a resumed run
        # reads as a continuation, not a restart from 0. Purely-internal loop
        # arithmetic (progress-bar total, `i == nSteps - 1`) stays on `i`.
        absStep = i + spec.resumeStepOffset

        if case.postStep is not None:
            case.postStep(ctx, runningState, absStep)
        if case.timestep is not None:
            ctx.config.dt = case.timestep(ctx, runningState)

        t = _scalar(runningState.t)
        row = {'step': absStep, 't': t, 'stepTime_ms': timer.elapsed_ms}
        if case.diagnostics is not None:
            row.update(case.diagnostics(ctx, runningState))
        result.trajectory.append(row)

        # `stallDtSteps` watchdog: a healthy adaptive-dt run should never sit
        # at `minDt` for long. When it does, simulated time is effectively
        # frozen (a single outlier particle typically pins the global CFL
        # estimate -- see ACSPH_PLAN.md's dam-break stall) and every further
        # step burns wall-clock without progress, so stop rather than run out
        # the clock to tLimit/nSteps at the floor.
        if spec.stallDtSteps is not None and ctx.config.minDt is not None:
            if _scalar(ctx.config.dt) <= ctx.config.minDt * 1.0001:
                dtAtFloorStreak += 1
            else:
                dtAtFloorStreak = 0
            if dtAtFloorStreak >= spec.stallDtSteps:
                say(f'dt has been pinned at the minDt floor '
                    f'({ctx.config.minDt:g}) for {dtAtFloorStreak} consecutive '
                    f'steps as of step {absStep} (t={t:g}); stopping -- '
                    f'simulated time is effectively frozen.')
                result.diverged = True
                result.stopReason = 'stallDtSteps: dt pinned at minDt'
                break

        if spec.stallProgress is not None and spec.tLimit:
            recentTimes.append(t)
            if len(recentTimes) == recentTimes.maxlen and \
                    recentTimes[-1] - recentTimes[0] < spec.stallProgress * spec.tLimit:
                say(f'simulated time advanced {recentTimes[-1] - recentTimes[0]:.3g} '
                    f'over the last {STALL_WINDOW_STEPS} steps as of step {absStep} '
                    f'(t={t:g}), below stallProgress x tLimit = '
                    f'{spec.stallProgress * spec.tLimit:.3g}; stopping -- '
                    f'simulated time is effectively frozen.')
                result.diverged = True
                result.stopReason = 'stallProgress: simulated time frozen'
                break

        if progress is not None:
            _showStep(progress, i, None if timeLimited else nSteps, row, spec.tLimit,
                      force=(timeLimited and t >= spec.tLimit) or (not timeLimited and i == nSteps - 1))

        if timeLimited and t >= spec.tLimit:
            _plotAndStore(ctx, case, spec, runningState, stepResult, absStep, extraData,
                          groups, storeSteps, final=True)
            break

        _plotAndStore(ctx, case, spec, runningState, stepResult, absStep, extraData,
                      groups, storeSteps, final=(not timeLimited and i == nSteps - 1))

        # Read by tag rather than the `velocities` field name: every fluid
        # scheme's velocity field happens to be named that, but the wave
        # scheme's is `v` (tagged `'velocity'` so it rides the same
        # position/velocity integrator machinery under a different name).
        # `max |v|` propagates NaN *and* inf, so this one host read is both
        # the non-finite check and the velocity alarm's input. Inf matters:
        # the dfsphReference late-time failure can collapse into a degenerate
        # uniform-density state with inf velocities (DFSPH_IMPROVEMENT_PLAN.md
        # Part 29) where no NaN ever appears.
        vmax = _scalar(maxSpeed(runningState))
        if not math.isfinite(vmax):
            say(f'non-finite velocities detected at step {absStep}; stopping.')
            result.diverged = True
            result.stopReason = 'non-finite velocities'
            break
        if alarm.check(ctx, runningState, vmax, absStep, t):
            say(f'velocity alarm stops the run at step {absStep} (t={t:g}): '
                f'{alarm.stopReason}.')
            result.diverged = True
            result.stopReason = alarm.stopReason
            break

    if progress is not None:
        progress.close()
    if ctx.scratch.get('renderThread') is not None:
        ctx.scratch['renderThread'].drain()
    if spec.plot:
        _drainFrameWrites()

    result.state = runningState
    result.nSteps = len(result.trajectory) - (1 if case.diagnostics is not None else 0)
    result.velocityAlarms = list(alarm.events)
    alarm.restore(ctx)

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


def _runPipelined(ctx, case, spec, result, stepGraph, steps, progress, runningState, nSteps,
                  timeLimited, extraData, groups, storeSteps, recentTimes, alarm, say):
    """The step loop with the next step overlapped with this step's outputs.

    Per iteration, with step n just finished:

    1. post-step hook, next `dt`, stall watchdogs, stop conditions and the
       non-finite check -- everything that decides whether and how step n+1
       runs (these need step n, not its diagnostics);
    2. launch step n+1 (a graph replay: only enqueued);
    3. step n's diagnostics row and plot frame, on a side CUDA stream (and
       warp on the same stream) that waits only on an event recorded before
       the launch -- so the CPU work, and the GPU work that fits alongside,
       overlap step n+1 instead of following it;
    4. finish step n+1.

    Step n's state is a set of clones the replay of step n+1 never writes,
    so every row and frame holds exactly the values the sequential loop
    produced; a step that ends the run is diagnosed and plotted before the
    loop stops, as there. Only used with a whole-step graph (no storing).
    """
    import warp as wp
    # highest priority: the outputs' many small kernels are what the loop
    # waits on, while the step replaying alongside has slack
    side = torch.cuda.Stream(device=ctx.device, priority=torch.cuda.Stream.priority_range()[1])
    wside = wp.stream_from_torch(side)
    dtAtFloorStreak = 0
    stepResult = None
    handle = None
    ts = time.perf_counter()

    def outputs(i, absStep, state, stepRes, stepMs, ready, final):
        """Diagnostics row + progress + frame for one finished step."""
        with torch.cuda.stream(side), wp.ScopedStream(wside, sync_enter=False, sync_exit=False):
            if ready is not None:
                side.wait_event(ready)
            row = {'step': absStep, 't': _scalar(state.t), 'stepTime_ms': stepMs}
            if case.diagnostics is not None:
                row.update(case.diagnostics(ctx, state))
            result.trajectory.append(row)
            if progress is not None:
                _showStep(progress, i, None if timeLimited else nSteps, row, spec.tLimit,
                          force=final)
            _plotAndStore(ctx, case, spec, state, stepRes, absStep, extraData,
                          groups, storeSteps, final=final)
        # the next step (default stream) may reuse memory the side stream
        # read from; make the default stream wait for it
        torch.cuda.current_stream(ctx.device).wait_stream(side)

    for i in steps:
        if handle is None:
            handle = stepGraph.launch(runningState, ctx.config.dt, ctx.config, ctx.schemeConfig,
                                      verbose=spec.verbose)
        stepResult = stepGraph.finish(handle)
        # wall time launch -> finish: on this loop that spans the previous
        # step's (overlapped) outputs too, i.e. the per-step throughput
        stepMs = (time.perf_counter() - ts) * 1e3
        handle = None
        runningState = stepResult.state
        ctx.scratch['lastStageUpdate'] = stepResult.stages[-1].update if stepResult.stages else None
        absStep = i + spec.resumeStepOffset

        if case.postStep is not None:
            case.postStep(ctx, runningState, absStep)
        if case.timestep is not None:
            ctx.config.dt = case.timestep(ctx, runningState)
        t = _scalar(runningState.t)

        stop = None
        if spec.stallDtSteps is not None and ctx.config.minDt is not None:
            if _scalar(ctx.config.dt) <= ctx.config.minDt * 1.0001:
                dtAtFloorStreak += 1
            else:
                dtAtFloorStreak = 0
            if dtAtFloorStreak >= spec.stallDtSteps:
                stop = (f'dt has been pinned at the minDt floor '
                    f'({ctx.config.minDt:g}) for {dtAtFloorStreak} consecutive '
                    f'steps as of step {absStep} (t={t:g}); stopping -- '
                    f'simulated time is effectively frozen.')
                result.stopReason = 'stallDtSteps: dt pinned at minDt'
        if stop is None and spec.stallProgress is not None and spec.tLimit:
            recentTimes.append(t)
            if len(recentTimes) == recentTimes.maxlen and \
                    recentTimes[-1] - recentTimes[0] < spec.stallProgress * spec.tLimit:
                stop = (f'simulated time advanced {recentTimes[-1] - recentTimes[0]:.3g} '
                    f'over the last {STALL_WINDOW_STEPS} steps as of step {absStep} '
                    f'(t={t:g}), below stallProgress x tLimit = '
                    f'{spec.stallProgress * spec.tLimit:.3g}; stopping -- '
                    f'simulated time is effectively frozen.')
                result.stopReason = 'stallProgress: simulated time frozen'
        # one host read: the non-finite check (max propagates NaN/inf) and
        # the velocity alarm's input
        vmax = _scalar(maxSpeed(runningState))
        nonFinite = not math.isfinite(vmax)
        if stop is None and not nonFinite and alarm.check(ctx, runningState, vmax, absStep, t):
            stop = (f'velocity alarm stops the run at step {absStep} (t={t:g}): '
                    f'{alarm.stopReason}.')
            result.stopReason = alarm.stopReason
        if stop is not None:
            # as the sequential loop: the row is recorded, no frame, then stop
            row = {'step': absStep, 't': t, 'stepTime_ms': stepMs}
            if case.diagnostics is not None:
                row.update(case.diagnostics(ctx, runningState))
            result.trajectory.append(row)
            say(stop)
            result.diverged = True
            break

        last = (timeLimited and t >= spec.tLimit) or (not timeLimited and i == nSteps - 1)
        if last or nonFinite:
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(ctx.device))
            outputs(i, absStep, runningState, stepResult, stepMs, ready, final=last)
            if nonFinite and not (timeLimited and last):
                # (the sequential loop stops at tLimit before this check)
                say(f'non-finite velocities detected at step {absStep}; stopping.')
                result.diverged = True
                result.stopReason = 'non-finite velocities'
            break

        # step n+1 goes onto the GPU first, then step n's outputs overlap it
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(ctx.device))
        ts = time.perf_counter()
        handle = stepGraph.launch(runningState, ctx.config.dt, ctx.config, ctx.schemeConfig,
                                  verbose=spec.verbose)
        outputs(i, absStep, runningState, stepResult, stepMs, ready, final=False)

    if handle is not None:
        stepGraph.finish(handle)
    return runningState, stepResult


class _RenderThread:
    """Runs a case's plot hooks on one worker thread, off the step loop.

    A frame is ~20-30 ms of host work (vispy scene update, draw, pixel
    readback) that the loop otherwise waits for every `plotInterval` steps;
    here it overlaps the steps that follow. The worker owns the plot from
    `setupPlot` to teardown, so its GL context (EGL: offscreen and bound to
    no thread in particular) is only ever current on that thread. Each frame
    renders a snapshot of the state (device clones taken on the submitting
    stream; the worker's own CUDA stream waits for them), so later steps are
    free to reuse or modify theirs. At most `maxPending` frames queue up
    before a submit waits; a failed frame raises at the next submit or at
    `drain`, which the runner calls before encoding the video.
    """

    def __init__(self, device, maxPending: int = 2):
        import concurrent.futures
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix='warpSPH-render')
        self.device = device
        cuda = torch.cuda.is_available() and torch.device(device).type == 'cuda'
        self.stream = torch.cuda.Stream(device=device) if cuda else None
        self.pending = collections.deque()
        self.maxPending = maxPending
        self._switchInterval = sys.getswitchinterval()
        sys.setswitchinterval(float(os.environ.get('WARPSPH_RENDER_SWITCH', self._switchInterval)))

    def _job(self, fn, args, ready):
        if self.stream is None:
            return fn(*args)
        # never while the loop thread captures a graph: plot code syncs
        # (mask indexing, host copies), which a capture forbids
        from ..utils.cudaGraph import CAPTURE_LOCK
        with CAPTURE_LOCK, torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            if ready is not None:
                self.stream.wait_event(ready)
            return fn(*args)

    def call(self, fn, *args):
        """Run `fn(*args)` on the worker, wait, return its result."""
        self.drain()
        return self.executor.submit(self._job, fn, args, None).result()

    def submitFrame(self, updatePlot, ctx, state, plotter, step):
        """Queue `updatePlot(ctx, snapshot, plotter, step)`."""
        from ..utils.cudaGraph import _cloneState
        snapshot = _cloneState(state)
        ready = None
        if self.stream is not None:
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(self.device))
        self.pending.append(self.executor.submit(
            self._job, updatePlot, (ctx, snapshot, plotter, step), ready))
        while len(self.pending) > self.maxPending:
            self.pending.popleft().result()

    def drain(self):
        while self.pending:
            self.pending.popleft().result()

    def close(self):
        try:
            self.drain()
        finally:
            self.executor.shutdown(wait=True)
            sys.setswitchinterval(self._switchInterval)


def _setupPlot(ctx: RunContext, case: Case, state):
    """`case.setupPlot`, on a render thread when the run can use one
    (`_RenderThread`): no live window (`show=False`; a window's event loop
    belongs to the main thread) and the vispy backend, rendered through EGL.
    Anything else -- or a setup that fails there -- plots on the main thread
    as before."""
    spec = ctx.spec
    backend = spec.plotBackend or ('vispy' if spec.dim == 2 else None)
    opts = dict(spec.plotBackendOptions or {})
    # Plot hooks that launch warp kernels (grid-interpolated panels,
    # `cases/plotting.py:particlePlot`) stay on the loop thread: warp's
    # current stream and module loading are per process, so warp work on the
    # render thread races the step's (OPEN_PROBLEMS §12).
    usesWarp = getattr(case.updatePlot, 'usesWarp', False) or getattr(case.setupPlot, 'usesWarp', False)
    headlessVispy = (not spec.show and backend == 'vispy'
                     and opts.get('app_backend', 'egl') == 'egl')
    if headlessVispy and usesWarp:
        # still headless: render through EGL on this thread, not through the
        # default app backend, which opens a real (black) window
        original = spec.plotBackendOptions
        spec.plotBackendOptions = dict(opts, app_backend='egl')
        try:
            return case.setupPlot(ctx, state)
        except Exception as ex:  # noqa: BLE001 -- e.g. no EGL: default backend
            spec.plotBackendOptions = original
            warnings.warn(f'[warpSPH] EGL unavailable, plotting through the default backend '
                          f'({type(ex).__name__}: {str(ex).splitlines()[0][:200]})')
            return case.setupPlot(ctx, state)
    if not headlessVispy or not spec.asyncPlot or usesWarp:
        return case.setupPlot(ctx, state)
    original = spec.plotBackendOptions
    spec.plotBackendOptions = dict(opts, app_backend='egl')
    renderer = _RenderThread(ctx.device)
    try:
        handle = renderer.call(case.setupPlot, ctx, state)
    except Exception as ex:  # noqa: BLE001 -- e.g. no EGL: plot on the main thread
        renderer.close()
        spec.plotBackendOptions = original
        warnings.warn(f'[warpSPH] render thread unavailable, plotting on the main thread '
                      f'({type(ex).__name__}: {str(ex).splitlines()[0][:200]})')
        return case.setupPlot(ctx, state)
    ctx.scratch['renderThread'] = renderer
    return handle


def _stepGraphable(ctx) -> bool:
    """Whole-step graphs are only wired for the weakly-compressible delta-SPH
    family (its RHS, integrator path and `finalize` were made sync-free; see
    `schemes/deltaSPH.py:_rhsIsGraphable` for the per-configuration rules)."""
    from ..enumTypes import WeaklyCompressibleSPHScheme
    if not isinstance(ctx.scheme, WeaklyCompressibleSPHScheme):
        return False
    from ..schemes.deltaSPH import _rhsIsGraphable
    return _rhsIsGraphable(ctx.schemeConfig, None)


def _drainFrameWrites() -> None:
    """Wait for frames the vispy backend is still encoding on its writer
    threads (`cases/plotting.py:_fastPngOptions`), so every PNG is complete
    before the video encode -- or the caller -- reads it."""
    try:
        from warpSPHPlotting import waitForExports
    except ImportError:
        return
    waitForExports()


def _teardownPlot(ctx: RunContext) -> None:
    """Hold the final figure if asked, then release it.

    Releasing matters because the figures are pyplot-managed: a process that
    runs several cases would otherwise keep every one of their windows alive.
    """
    handle = ctx.scratch.pop('plot', None)
    renderer = ctx.scratch.pop('renderThread', None)
    if renderer is not None:
        if handle is not None:
            renderer.call(closeWindow, handle)
        renderer.close()
        return
    if handle is None:
        return
    holdWindow(ctx, handle)
    closeWindow(handle)


#: `collectFrameGarbage`: run a cycle collection once allocated GPU memory exceeds the post-collection baseline by
#: this factor (and by at least `_FRAME_GC_MIN_BYTES`). OPEN_PROBLEMS §18.
_FRAME_GC_GROWTH = 1.5
_FRAME_GC_MIN_BYTES = 256 * 2**20


def collectFrameGarbage(ctx: RunContext) -> None:
    """Free the step states a plotted run strands in reference cycles (OPEN_PROBLEMS §18).

    Step / stage states end up in reference cycles, which only Python's cycle collector frees. Without plotting it
    runs often enough; with plotting, matplotlib's mathtext parser leaves ~10^5 cyclic objects per frame (pyparsing
    exceptions holding traceback frames), the collector's full passes get rarer, and the stranded states -- with
    CRKSPH's pair-sized `ap_ij` / `av_ij` -- pile up: a CRK 3D Sedov at nx 16 reached 2.9 GB in 60 frames (flat
    255 MB without video, and flat 255 MB with video plus a collection per frame); at nx 40 it ran out of memory.
    The allocator's counter is host-side (no sync), so the check is free; a collection runs only once the
    allocated memory has grown by `_FRAME_GC_GROWTH` over its post-collection baseline."""
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated()
    baseline = ctx.scratch.get('_frameGcBaseline')
    if baseline is None:
        ctx.scratch['_frameGcBaseline'] = allocated
        return
    if allocated > max(_FRAME_GC_GROWTH * baseline, baseline + _FRAME_GC_MIN_BYTES):
        import gc
        gc.collect()
        ctx.scratch['_frameGcBaseline'] = torch.cuda.memory_allocated()
        ctx.scratch['_frameGcCollections'] = ctx.scratch.get('_frameGcCollections', 0) + 1


def _plotAndStore(ctx: RunContext, case: Case, spec: CaseSpec, state, stepResult,
                  i: int, extraData: Dict[str, Any], groups, storeSteps: int,
                  final: bool) -> None:
    """The per-step output side of the loop: plot every N, export every M."""
    if spec.plot and case.updatePlot is not None and i > 0 and \
            (i % spec.plotInterval == 0 or final):
        renderer = ctx.scratch.get('renderThread')
        if renderer is not None:
            renderer.submitFrame(case.updatePlot, ctx, state, ctx.scratch.get('plot'), i)
        else:
            case.updatePlot(ctx, state, ctx.scratch.get('plot'), i)
        collectFrameGarbage(ctx)

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


def _showStep(progress, i: int, nSteps: Optional[int], row: Dict[str, float],
              tLimit: float, force: bool = False) -> None:
    """Put this step's row on the bar -- at most every `progress.mininterval`
    (0.1 s) unless `force`d (the last step). Formatting and redrawing it every
    step cost ~0.8 ms/step on the ~5 ms Marrone step (2026-09-28 A/B); the
    rows still stream ~10 times a second."""
    now = time.monotonic()
    if not force and now - getattr(progress, '_warpSPHShownAt', -math.inf) < progress.mininterval:
        return
    progress._warpSPHShownAt = now
    if nSteps is None:  # time-limited: the bar is a 1000-tick scale over t
        progress.n = min(progress.total, int(row['t'] / tLimit * progress.total))
    progress.set_description(_describeStep(i, nSteps, row, tLimit))


def _writer(progress):
    """A flushed line to the terminal that does not collide with the bar."""
    if progress is not None:
        return lambda message: progress.write(message)
    return lambda message: print(message, flush=True)


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
