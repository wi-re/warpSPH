"""Instrumented build + step loop for the wave case, shared by all suites.

The step loop is the notebook's unrolled loop (`examples/wave/
waveCase_implicit_vs_explicit.ipynb`, cell 5) with three measurement
additions, each wrapping the notebook's exact call path rather than
replacing it:

* a counting wrapper around `ctx.stepFunction` -- every right-hand-side
  evaluation is counted, including the ones a `JFNKSolver` matvec hides
  inside its Krylov iterations, which is the number a cost comparison needs;
* a `RecordingSolver` wrapper around the `NonlinearSolver` -- records each
  stage solve's iteration count and convergence verdict (the "internal
  solver loop" axis: a non-converged JFNK can still return a finite state,
  and only this record tells the two apart);
* per-step CUDA-event timing (mirroring `warpSPH.runner.runner._Timer`)
  plus peak-memory statistics and a static state footprint.

Deliberately *not* timed: per-step `case.diagnostics(...)` (the notebook's
trajectory loop pays for a Gradient operator per step; a pure integrator
benchmark must not). Diagnostics are taken at the initial and final state
only, outside the timed region.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from warpSPH.cases.demos.waveEquation import waveEquationCase

from .schemes import MULTISTEP_SCHEMES, SchemeSpec


@dataclass
class RunRecord:
    """Everything one measured run produced; JSON-serializable via `toDict`."""

    # what was run
    key: str
    label: str
    kind: str
    integrationScheme: str
    solverDesc: str
    order: int
    nx: int
    nParticles: int
    device: str
    dt: float
    nSteps: int
    # measured cost
    buildSeconds: float = 0.0
    warmupSteps: int = 0
    stepsDone: int = 0
    stepSeconds: float = 0.0
    msPerStep: float = 0.0
    msPerRhs: float = 0.0
    fEvals: int = 0
    fEvalsPerStep: float = 0.0
    # internal solver (implicit only; zeros for explicit)
    solves: int = 0
    convergedSolves: int = 0
    itersMin: int = 0
    itersMax: int = 0
    itersMean: Optional[float] = None
    # memory
    peakAllocatedMB: float = 0.0
    peakReservedMB: float = 0.0
    staticStateMB: float = 0.0
    # numerics
    uMax0: float = 0.0
    uMaxFinal: Optional[float] = None
    uMaxPeak: Optional[float] = None
    diverged: bool = False
    energy0: Optional[float] = None
    energyFinal: Optional[float] = None
    # final fields (accuracy suites) -- kept on CPU, small
    uFinal: Optional[torch.Tensor] = field(default=None, repr=False)
    vFinal: Optional[torch.Tensor] = field(default=None, repr=False)
    # per-step max|u| (stability suite)
    uMaxTrajectory: List[float] = field(default_factory=list)
    # suite-specific extras (error vs. reference, order, ...)
    extra: Dict[str, Any] = field(default_factory=dict)

    def toDict(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        d.pop('uFinal', None)
        d.pop('vFinal', None)
        return d


class StepTimer:
    """CUDA-event timing where available, wall clock otherwise -- the same
    convention as `warpSPH.runner.runner._Timer`, so suite numbers are
    comparable to the frontend's own `stepTime_ms`."""

    def __init__(self, device: torch.device):
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


class RecordingSolver:
    """`NonlinearSolver` wrapper that records each solve's verdict and cost.

    `JFNKSolver`/`FixedPointSolver` return a `SolveResult(y, converged,
    iterations)` that the DIRK driver silently discards -- the driver only
    uses `y`. Without this wrapper a run where the solver hit its
    `max_iterations` budget and returned an un-converged stage is
    indistinguishable (in the state alone) from a converged one. The
    protocol is structural (one `solve` method), so plain delegation is all
    that is needed.
    """

    def __init__(self, inner):
        self.inner = inner
        self.solves = 0
        self.converged = 0
        self.iterations: List[int] = []

    def solve(self, step, y0, norm=None, **opts):
        result = self.inner.solve(step, y0, norm, **opts)
        self.solves += 1
        self.converged += int(bool(result.converged))
        self.iterations.append(int(result.iterations))
        return result

    def stats(self) -> Dict[str, Any]:
        iters = [i for i in self.iterations if math.isfinite(i)]
        return {
            'solves': self.solves,
            'converged': self.converged,
            'itersMin': min(iters) if iters else 0,
            'itersMax': max(iters) if iters else 0,
            'itersMean': (sum(iters) / len(iters)) if iters else None,
        }


def _peakMemoryMB(device: torch.device) -> Tuple[float, float]:
    """`(peak_allocated_MB, peak_reserved_MB)` since the last reset."""
    if device.type == 'cuda' and torch.cuda.is_available():
        return (torch.cuda.max_memory_allocated(device) / 2**20,
                torch.cuda.max_memory_reserved(device) / 2**20)
    # CPU: no per-run resettable allocator stats; the process high-water RSS
    # is the honest (if coarser) number. `reserved` is not meaningful there.
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 0.0


def _resetPeakMemory(device: torch.device) -> None:
    if device.type == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def tensorFootprintMB(obj: Any, _depth: int = 0, _seen: Optional[set] = None) -> float:
    """Total byte footprint of every `torch.Tensor` reachable from `obj`
    (its `__dict__`, recursively, cycle-guarded, depth-bounded) in MB.

    Used for the static state/adjacency footprint: the per-particle memory
    a case of this size holds before the integrator allocates anything.
    """
    if obj is None or _depth > 6:
        return 0.0
    if isinstance(obj, torch.Tensor):
        return obj.element_size() * obj.numel() / 2**20
    if isinstance(obj, (list, tuple, set, frozenset)):
        if _seen is None:
            _seen = set()
        return sum(tensorFootprintMB(v, _depth + 1, _seen) for v in obj)
    d = getattr(obj, '__dict__', None)
    if not d:
        return 0.0
    if _seen is None:
        _seen = set()
    oid = id(obj)
    if oid in _seen:
        return 0.0
    _seen.add(oid)
    return sum(tensorFootprintMB(v, _depth + 1, _seen) for v in d.values())



def buildWaveCase(nx: int, device: Optional[str] = None,
                  **paramOverrides) -> Tuple[Any, Any, float]:
    """Build the registered `waveEquationCase` exactly the way the notebook
    (and every `examples/*/0N-*.ipynb`) does, with the case's own defaults,
    and return `(ctx, system, buildSeconds)`.

    `dt` is deliberately left to the case's CFL derivation
    (`spec.dt=None`): `ctx.config.dt` then holds the case's own default dt
    for this `nx`, which the suites use as their `dt` base and
    stability-limit reference. `paramOverrides` are merged onto the case
    defaults the same way the notebook's `spec.merged(...)` calls do (e.g.
    `obstacleEnabled=False` for a bare-domain run).

    One build serves many runs: the integrator never mutates the built
    `system` in place (it works on `initializeNewState()` clones), so
    `runScheme` below re-seeds from the same `system` for every run.
    """
    # Imported here (not at module top) so importing the benchmark package
    # stays cheap and -- more importantly -- happens only *after* the calling
    # script has run `warpSPHBootstrap.bootstrap`, which is the only point at
    # which precision can still be chosen.
    from warpSPH.cases.demos.waveEquation import waveEquationCase
    from warpSPH.runner import CaseSpec
    from warpSPH.runner.runner import buildContext

    t0 = time.perf_counter()

    from dataclasses import fields as _dataclassFields
    _specFields = {f.name for f in _dataclassFields(CaseSpec)}

    spec = CaseSpec(caseName=waveEquationCase.name, scheme=waveEquationCase.scheme,
                    params=dict(waveEquationCase.params)).merged(**waveEquationCase.defaults)
    spec = spec.merged(nx=nx, quiet=True, progress=False,
                       integrationScheme='rungeKutta4')  # name only; runs pick their own
    if device is not None:
        spec = spec.merged(device=device)
    if paramOverrides:
        # Same split `specFromArgs` uses: known CaseSpec fields go as
        # overrides, anything else is a case parameter merged onto
        # `spec.params` (the case's `ctx.param(...)` surface).
        fieldOverrides = {k: v for k, v in paramOverrides.items() if k in _specFields}
        caseParams = {k: v for k, v in paramOverrides.items() if k not in _specFields}
        spec = spec.merged(**fieldOverrides, params=caseParams)

    ctx = buildContext(waveEquationCase, spec)
    waveEquationCase.configureScheme(ctx)
    system = waveEquationCase.buildSystem(ctx)
    waveEquationCase.initialConditions(ctx, system)  # derives dt from CFL (spec.dt is None)

    if ctx.device.type == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize(ctx.device)
    buildSeconds = time.perf_counter() - t0
    return ctx, system, buildSeconds


def _safeEnergy(ctx, system) -> Optional[float]:
    """`case.diagnostics` total energy, or `None` if the call fails (e.g. on a
    diverged state). Always called outside timed regions."""
    try:
        return float(waveEquationCase.diagnostics(ctx, system)['totalEnergy'])
    except Exception:
        return None


def runScheme(ctx, system, scheme: SchemeSpec, nSteps: int, dt: float,
              warmup: int = 3, trackU: bool = False, keepFields: bool = True) -> RunRecord:
    """Run `scheme` for `nSteps` steps of size `dt` from a fresh state seeded
    off the built `system`, exactly the way the notebook's unrolled loop does
    (`ctx.integrator.function(state=..., f=..., dt=..., config=...,
    schemeConfig=..., verbose=False, [solver=...])`), and measure it.

    * `warmup` untimed steps first -- this is where the one-time warp kernel
      compilation/loads for this device happen, so the timed region is the
      steady state;
    * `trackU` records the per-step max|u| trajectory (stability suite);
    * `keepFields` keeps the final u/v on CPU (accuracy suite);
    * multistep schemes get `history=` threaded between calls -- their
      documented correct calling convention, without which they re-run their
      high-order starter every step and cost 5-6x their nominal RHS count.

    A non-finite max|u| at any step stops the run early and is recorded as
    `diverged` (the notebook's own stop condition). Implicit runs additionally
    report their internal-solver record: a JFNK that used up its
    `max_iterations` budget without converging shows up as
    `convergedSolves < solves` even when the state itself stayed finite.
    """
    from warpSPHIntegrators import get_scheme

    device = ctx.device
    record = RunRecord(
        key=scheme.key, label=scheme.label, kind=scheme.kind,
        integrationScheme=scheme.integrationScheme, solverDesc=scheme.solverDesc,
        order=scheme.order, nx=int(ctx.spec.nx), nParticles=int(system.state.u.shape[0]),
        device=str(device), dt=float(dt), nSteps=nSteps,
        warmupSteps=warmup,
    )

    state = system.initializeNewState()
    record.uMax0 = float(state.state.u.abs().max().item())
    record.energy0 = _safeEnergy(ctx, state)

    # Every right-hand-side evaluation counted, including the ones a
    # JFNKSolver matvec hides inside its Krylov iterations.
    count = [0]
    original = ctx.stepFunction

    def fCounted(*args, **kwargs):
        count[0] += 1
        return original(*args, **kwargs)

    recorder: Optional[RecordingSolver] = None
    kwargs: Dict[str, Any] = dict(f=fCounted, dt=dt, config=ctx.config,
                                  schemeConfig=ctx.schemeConfig, verbose=False)
    if scheme.kind == 'implicit':
        recorder = RecordingSolver(scheme.makeSolver())
        kwargs['solver'] = recorder

    integrator = get_scheme(scheme.integrationScheme)
    isMultistep = scheme.integrationScheme.lower() in MULTISTEP_SCHEMES
    history = None

    _resetPeakMemory(device)
    staticMB = tensorFootprintMB(system.state) + tensorFootprintMB(system.adjacency)

    diverged = False
    stepsDone = 0
    stepMs = 0.0
    uMax = record.uMax0
    trajectory: List[float] = []

    for i in range(warmup + nSteps):
        if isMultistep:
            kwargs['history'] = history
        else:
            kwargs.pop('history', None)
        timed = i >= warmup
        if timed:
            with StepTimer(device) as timer:
                result = integrator.function(state=state, **kwargs)
            stepMs += timer.elapsed_ms
        else:
            result = integrator.function(state=state, **kwargs)
        state = result.state
        if isMultistep:
            history = result.history
        stepsDone = i + 1
        uMax = float(state.state.u.abs().max().item())
        if trackU:
            trajectory.append(uMax)
        if not math.isfinite(uMax):
            diverged = True
            break

    finiteTraj = [v for v in trajectory if math.isfinite(v)]
    record.stepsDone = stepsDone
    record.diverged = diverged
    record.fEvals = count[0]
    record.fEvalsPerStep = count[0] / max(1, stepsDone)
    record.stepSeconds = stepMs / 1000.0
    timedSteps = max(0, stepsDone - warmup)
    record.msPerStep = stepMs / timedSteps if timedSteps > 0 else 0.0
    record.msPerRhs = (record.msPerStep / record.fEvalsPerStep
                       if record.fEvalsPerStep > 0 else 0.0)
    if recorder is not None:
        s = recorder.stats()
        record.solves, record.convergedSolves = s['solves'], s['converged']
        record.itersMin, record.itersMax, record.itersMean = s['itersMin'], s['itersMax'], s['itersMean']
    record.peakAllocatedMB, record.peakReservedMB = _peakMemoryMB(device)
    record.staticStateMB = staticMB
    record.uMaxTrajectory = trajectory
    if diverged:
        record.uMaxFinal = None
        record.uMaxPeak = max(finiteTraj) if finiteTraj else None
    else:
        record.uMaxFinal = uMax
        record.uMaxPeak = max(finiteTraj + [uMax])
        record.energyFinal = _safeEnergy(ctx, state)
    if keepFields and not diverged:
        record.uFinal = state.state.u.detach().cpu()
        record.vFinal = state.state.v.detach().cpu()
    return record


