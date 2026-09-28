"""CUDA-graph replay of a scheme's right-hand-side evaluation.

For a small problem the weakly-compressible step is CPU-bound: each RHS
evaluation issues ~400 small torch ops and ~40 warp operator launches whose
Python-side cost (argument marshalling, overload resolution, torch dispatch)
dwarfs the GPU work -- measured on Marrone 3.1 at nx=70 (10.9k particles):
~13 ms of CPU for ~2 ms of GPU per step after the multi-lane kernels
(warpSPHCore ``autograd/lanes.py``). Capturing the RHS once and replaying it
removes all of that per-call CPU work.

:class:`GraphedStateFunction` wraps ``fn(system, *args, **kwargs) ->
(update, adjacency, state)`` (a scheme step function *after* its eager
adjacency update):

* **capture** -- the first call for a given key runs ``fn`` once eagerly (warm
  up: module loads, caches), then captures it into a ``torch.cuda.CUDAGraph``
  on a side stream shared with warp (``wp.stream_from_torch``) against private
  copies of the state's tensors (the *static inputs*). Every tensor attribute
  of ``system.state`` that ``fn`` reassigns, and every tensor field of the
  returned update, becomes a *static output*.
* **replay** -- copy the caller's state tensors into the static inputs (one
  ``torch._foreach_copy_``), replay, then hand back *clones* of the static
  outputs (the next replay overwrites them, and the integrator keeps earlier
  stages' updates alive), assigned onto the caller's state exactly as the eager
  ``fn`` would have assigned them.
* **key** -- adjacency object identity (the Verlet list is only rebuilt every
  so often; a rebuilt list is a new object with new buffers, which evicts all
  graphs of the old one) + the name/shape/dtype of every state tensor + any
  caller-supplied extra key.
* **validation** -- right after the first capture of each state signature
  (every capture with ``WARPSPH_CUDAGRAPH_VALIDATE=always``) the graph is
  replayed on the capture inputs and compared **bitwise** against a fresh
  eager evaluation of ``fn``, including every state tensor (so an in-place
  mutation of an input that ``fn`` does not reassign is caught too). Any mismatch, or any error
  during capture (e.g. a host sync inside ``fn``), disables graphs for this
  wrapper with a one-time warning and falls back to eager ``fn`` -- graph mode
  can only ever be exactly the eager result or eager itself.

What ``fn`` must satisfy to be capturable (the eager fallback covers the rest):
no host syncs (``.item()``, ``bool(tensor)``, boolean-mask indexing -- see
``utils/syncFree.py``), no data-dependent Python control flow, and no
dependence on Python scalars that change between calls (``t``, ``dt``), since
those are baked into the graph at capture time. Callers are responsible for
only routing such a configuration here (``schemes/deltaSPH.py``).
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import gc
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

__all__ = ['GraphedStateFunction', 'GraphedTensorFunction', 'GraphedIntegratorStep']

#: True while a whole integrator step is being captured: the nested per-RHS
#: graphs then run their function eagerly, so it is recorded into the outer graph.
_OUTER_CAPTURE = [False]


#: Capture mode for every graph here: 'thread_local', not torch's default
#: 'global', which forbids unsafe CUDA calls from *every* thread during a
#: capture -- the runner's render thread (`runner.py:_RenderThread`) copies
#: its frame data on a stream of its own and must be able to keep doing so
#: while the loop thread captures (under 'global' its copy failed and
#: invalidated the capture). Work on the capturing stream is unaffected.
_CAPTURE_MODE = 'thread_local'


@contextlib.contextmanager
def _captureGuard():
    """No garbage collection while a graph is captured.

    A collection mid-capture finalises whatever dead objects earlier work
    left behind -- among them warp `Stream`s from `wp.stream_from_torch`,
    whose finaliser unregisters the stream: a CUDA call the capturing thread
    may not make, so the capture fails (warp error 710/901 in
    `wp_cuda_stream_unregister` / `wp_cuda_launch_kernel`). Whether that
    happens depended on how much garbage the process had accumulated --
    `test_renderThreadFramesMatchMainThread` failed in most runs once the
    test session ran a few more cases before it, and in none of 10 with this
    guard (2026-09-28). `torch.cuda.graph`
    collects on entry; this keeps the collector off until the capture ends."""
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


def _tensorAttrs(obj) -> Dict[str, torch.Tensor]:
    if dataclasses.is_dataclass(obj):
        names = [f.name for f in dataclasses.fields(obj)]
    else:
        names = list(vars(obj))
    out = {}
    for n in names:
        v = getattr(obj, n, None)
        if isinstance(v, torch.Tensor):
            out[n] = v
    return out


_INT_VIEW = {torch.float16: torch.int16, torch.bfloat16: torch.int16,
             torch.float32: torch.int32, torch.float64: torch.int64}


def _bitwiseEqual(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bit-for-bit equality (NaN payloads included, unlike ``torch.equal``)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    iv = _INT_VIEW.get(a.dtype)
    if iv is not None:
        return torch.equal(a.contiguous().view(iv), b.contiguous().view(iv))
    return torch.equal(a, b)


def _validateAlways() -> bool:
    import os
    return os.environ.get('WARPSPH_CUDAGRAPH_VALIDATE', '') == 'always'


def _cloneState(system):
    """Shallow system copy whose ``state`` tensors are fresh clones."""
    sysc = copy.copy(system)
    sysc.state = copy.copy(system.state)
    for n, t in _tensorAttrs(system.state).items():
        setattr(sysc.state, n, t.clone())
    return sysc


class _Entry:
    __slots__ = ('adjacency', 'graph', 'names', 'staticIn', 'outState', 'outUpdate',
                 'updateType', 'updateNonTensor')


class GraphedStateFunction:
    def __init__(self, fn: Callable, name: str, verbose: bool = False):
        self.fn = fn
        self.name = name
        self.verbose = verbose
        self.disabled: Optional[str] = None
        self._entries: Dict[Tuple, _Entry] = {}
        self._adjacency = None
        self._stream: Optional[torch.cuda.Stream] = None
        self.captures = 0
        self.replays = 0
        self.validations = 0
        self.captureSeconds = 0.0
        #: state signatures whose capture has already been validated (see _capture)
        self._validated = set()
        self._warm = False

    def __deepcopy__(self, memo):
        # graphs and their memory pools are not copyable; a copy re-captures
        return GraphedStateFunction(self.fn, self.name, self.verbose)

    def __reduce__(self):
        return (GraphedStateFunction, (self.fn, self.name, self.verbose))

    # ------------------------------------------------------------------ utils
    def _key(self, system, extraKey) -> Tuple:
        sig = tuple((n, tuple(t.shape), t.dtype) for n, t in _tensorAttrs(system.state).items())
        return (sig, extraKey)

    def _disable(self, reason: str) -> None:
        self.disabled = reason
        self._entries.clear()
        warnings.warn(f'[warpSPH] CUDA graph for {self.name} disabled, running eagerly: {reason}',
                      stacklevel=3)

    def _sideStream(self):
        if self._stream is None:
            self._stream = torch.cuda.Stream()
        return self._stream

    # ------------------------------------------------------------------- call
    def __call__(self, system, *args, extraKey=None, **kwargs):
        if self.disabled is not None or not system.state.positions.is_cuda or _OUTER_CAPTURE[0]:
            return self.fn(system, *args, **kwargs)
        adjacency = getattr(system, 'adjacency', None)
        if adjacency is not self._adjacency:
            # rebuilt Verlet list: new buffers, old graphs would read freed memory
            self._entries.clear()
            self._adjacency = adjacency
        key = self._key(system, extraKey)
        entry = self._entries.get(key)
        if entry is None:
            entry = self._capture(system, args, kwargs)
            if entry is None:
                return self.fn(system, *args, **kwargs)
            self._entries[key] = entry
        return self._replay(entry, system)

    def _replay(self, entry: _Entry, system):
        src = [getattr(system.state, n) for n in entry.names]
        torch._foreach_copy_(entry.staticIn, src)
        entry.graph.replay()
        self.replays += 1
        for n, t in entry.outState.items():
            setattr(system.state, n, t.clone())
        fields = {n: t.clone() for n, t in entry.outUpdate.items()}
        fields.update(entry.updateNonTensor)
        update = entry.updateType(**fields)
        return update, system.adjacency, system.state

    # ---------------------------------------------------------------- capture
    def _capture(self, system, args, kwargs) -> Optional[_Entry]:
        import time
        import warp as wp
        t0 = time.perf_counter()
        try:
            stream = self._sideStream()
            wstream = wp.stream_from_torch(stream)
            torch.cuda.synchronize()

            # warm-up (eager, on the capture stream) before the FIRST capture
            # only: loads warp modules, fills per-run host caches
            # (`config._hasBoundaryParticles`, device constants). Later captures
            # (a rebuilt Verlet list -- every ~17 steps once a dam break is
            # violent) find all of that already in place.
            if not self._warm:
                with torch.cuda.stream(stream), wp.ScopedStream(wstream, sync_enter=False, sync_exit=False):
                    self.fn(_cloneState(system), *args, **kwargs)
                torch.cuda.synchronize()
                self._warm = True

            staticSys = _cloneState(system)
            staticIn = _tensorAttrs(staticSys.state)
            graph = torch.cuda.CUDAGraph()
            with _captureGuard(), \
                    torch.cuda.graph(graph, stream=stream, capture_error_mode=_CAPTURE_MODE), \
                    wp.ScopedStream(wstream, sync_enter=False, sync_exit=False):
                update, _, _ = self.fn(staticSys, *args, **kwargs)
            torch.cuda.synchronize()
        except Exception as e:  # noqa: BLE001 -- any capture failure -> eager
            torch.cuda.synchronize()
            self._disable(f'capture failed ({type(e).__name__}: {str(e).splitlines()[0][:200]})')
            return None

        after = _tensorAttrs(staticSys.state)
        entry = _Entry()
        entry.adjacency = system.adjacency
        entry.graph = graph
        entry.names = list(staticIn)
        entry.staticIn = [staticIn[n] for n in entry.names]
        entry.outState = {n: t for n, t in after.items() if staticIn.get(n) is not t}
        if not dataclasses.is_dataclass(update):
            self._disable(f'update type {type(update).__name__} is not a dataclass')
            return None
        entry.updateType = type(update)
        entry.outUpdate, entry.updateNonTensor = {}, {}
        for f in dataclasses.fields(update):
            v = getattr(update, f.name)
            (entry.outUpdate if isinstance(v, torch.Tensor) else entry.updateNonTensor)[f.name] = v
        self.captures += 1

        # Validate the first capture of each state signature (and every one
        # with WARPSPH_CUDAGRAPH_VALIDATE=always). A re-capture after a Verlet
        # rebuild traces the same sync-free code path over new buffers of the
        # same shapes, so re-proving it bitwise every time only multiplied the
        # capture cost (measured ~80 ms per capture, ~1 capture / 17 steps).
        sig = self._key(system, None)
        if sig not in self._validated or _validateAlways():
            self.validations += 1
            if not self._validate(entry, system, args, kwargs):
                return None
            self._validated.add(sig)
        self.captureSeconds += time.perf_counter() - t0
        if self.verbose:
            print(f'[warpSPH] captured CUDA graph for {self.name} '
                  f'({len(entry.names)} inputs, {len(entry.outState)} state outputs)', flush=True)
        return entry

    def _validate(self, entry: _Entry, system, args, kwargs) -> bool:
        """Replay on the capture inputs vs a fresh eager evaluation, bitwise."""
        eagerSys = _cloneState(system)
        eagerIn = _tensorAttrs(eagerSys.state)
        eUpdate, _, _ = self.fn(eagerSys, *args, **kwargs)
        graphSys = _cloneState(system)
        gUpdate, _, _ = self._replay(entry, graphSys)
        torch.cuda.synchronize()
        bad: List[str] = []
        eState, gState = _tensorAttrs(eagerSys.state), _tensorAttrs(graphSys.state)
        for n in set(eState) | set(gState):
            a, b = eState.get(n), gState.get(n)
            if a is None or b is None or not _bitwiseEqual(a, b):
                bad.append(f'state.{n}' + (' (mutated in place)' if n in eagerIn and eagerIn[n] is eState.get(n)
                                            and n not in entry.outState else ''))
        for n, b in _tensorAttrs(gUpdate).items():
            a = getattr(eUpdate, n, None)
            if not isinstance(a, torch.Tensor) or not _bitwiseEqual(a, b):
                bad.append(f'update.{n}')
        if bad:
            self._disable('graph replay differs from eager evaluation in ' + ', '.join(sorted(bad)[:8]))
            return False
        return True


class GraphedTensorFunction:
    """Capture ``fn(*tensors) -> tuple of tensors`` once, then replay it on new
    inputs of the same shapes -- the shape of a fixed-point iteration body
    (e.g. one ACSPH dual-time pseudo-iteration, whose scalars are constant
    within a real step). Everything ``fn`` closes over is baked in at capture,
    so build one per region in which those stay fixed.

    The first replay is validated **bitwise** against an eager call on the same
    inputs (``validate=True``); any mismatch or capture error disables the
    wrapper (warning) and it runs ``fn`` eagerly from then on. Outputs are
    returned as clones, so callers may keep them across later replays.
    """

    def __init__(self, fn: Callable, name: str, validate: bool = True):
        self.fn, self.name = fn, name
        self.validate = validate
        self.disabled: Optional[str] = None
        self.graph = None
        self.staticIn: List[torch.Tensor] = []
        self.staticOut: Tuple[torch.Tensor, ...] = ()
        self.replays = 0

    def _disable(self, reason: str) -> None:
        self.disabled = reason
        self.graph = None
        warnings.warn(f'[warpSPH] CUDA graph for {self.name} disabled, running eagerly: {reason}',
                      stacklevel=3)

    def _capture(self, inputs) -> bool:
        import warp as wp
        try:
            stream = torch.cuda.Stream()
            wstream = wp.stream_from_torch(stream)
            torch.cuda.synchronize()
            self.staticIn = [t.clone() for t in inputs]
            graph = torch.cuda.CUDAGraph()
            with _captureGuard(), \
                    torch.cuda.graph(graph, stream=stream, capture_error_mode=_CAPTURE_MODE), \
                    wp.ScopedStream(wstream, sync_enter=False, sync_exit=False):
                out = self.fn(*self.staticIn)
            torch.cuda.synchronize()
        except Exception as e:  # noqa: BLE001 -- any capture failure -> eager
            torch.cuda.synchronize()
            self._disable(f'capture failed ({type(e).__name__}: {str(e).splitlines()[0][:200]})')
            return False
        self.graph = graph
        self.staticOut = tuple(out)
        return True

    def __call__(self, *inputs):
        if self.disabled is not None:
            return self.fn(*inputs)
        if self.graph is None and not self._capture(inputs):
            return self.fn(*inputs)
        torch._foreach_copy_(self.staticIn, list(inputs))
        self.graph.replay()
        self.replays += 1
        out = tuple(t.clone() for t in self.staticOut)
        if self.validate:
            self.validate = False
            eager = self.fn(*[t.clone() for t in inputs])
            if not all(_bitwiseEqual(a, b) for a, b in zip(eager, out)):
                self._disable('graph replay differs from eager evaluation')
                return tuple(eager)
        return out



@contextlib.contextmanager
def forkedStream(scratch: Dict[str, Any], key: str, device=None):
    """Run the body on a stream of its own (created once, kept in `scratch`
    under `key`), ordered after everything already queued on the current
    stream and joined back into it on exit -- so work queued before the body
    (e.g. a graph replay) does not delay the body's own host reads. Torch and
    warp both use the forked stream. A no-op off CUDA."""
    if device is not None and torch.device(device).type != 'cuda' or not torch.cuda.is_available():
        yield
        return
    import warp as wp
    current = torch.cuda.current_stream(device)
    stream = scratch.get(key)
    if stream is None:
        stream = scratch[key] = torch.cuda.Stream(device=current.device, priority=current.priority)
    stream.wait_stream(current)
    with torch.cuda.stream(stream), \
            wp.ScopedStream(wp.stream_from_torch(stream), sync_enter=False, sync_exit=False):
        yield
    current.wait_stream(stream)


def _packDevice(dev) -> Tuple[List[str], List[Tuple[int, ...]], torch.Tensor]:
    """{name: tensor} -> (keys, shapes, one flat float64 tensor). float64
    holds every float32 value and every count exactly. Entries are grouped
    by dtype (one concatenation and one cast per dtype, not a cast per
    entry), so `keys` comes back in that grouped order."""
    groups: Dict[torch.dtype, List[str]] = {}
    for k, v in dev.items():
        groups.setdefault(v.dtype, []).append(k)
    keys = [k for ks in groups.values() for k in ks]
    shapes = [tuple(dev[k].shape) for k in keys]
    flat = torch.cat([torch.cat([dev[k].detach().reshape(-1) for k in ks]).to(torch.float64)
                      for ks in groups.values()])
    return keys, shapes, flat


def _unpackHost(keys, shapes, values: List[float]) -> Dict[str, Any]:
    """Inverse of `_packDevice` on the host: floats for 0-d entries, CPU
    float64 tensors of the original shape otherwise."""
    out, k = {}, 0
    for key, shape in zip(keys, shapes):
        n = 1
        for d in shape:
            n *= d
        out[key] = values[k] if shape == () else torch.tensor(values[k:k + n], dtype=torch.float64).view(shape)
        k += n
    return out


def _readHost(dev) -> Dict[str, Any]:
    """{name: tensor} -> host values (floats for 0-d tensors), one transfer."""
    if not dev:
        return {}
    keys, shapes, flat = _packDevice(dev)
    return _unpackHost(keys, shapes, flat.cpu().tolist())


class GraphedDiagnostics:
    """Replay a case's per-step diagnostics from a CUDA graph.

    ``deviceFn(ctx, state) -> {name: device tensor}`` must be sync-free
    once its caches are warm (e.g. `cases/dambreak.py:_diagnosticsDevice`);
    `__call__` returns the values on the host, in one transfer. The inputs
    are the tensors of the particle state and of the step statistics
    (`stepDiagnostics.raw`); the neighbour list is baked in, so there is one
    capture per Verlet generation. The first call on a new neighbour list
    runs eagerly (it warms the per-adjacency caches a capture must not
    build) and returns those values; the capture follows on the next call.
    The first replay of a run is compared bitwise with an eager evaluation;
    any capture failure or mismatch runs the diagnostics eagerly for the rest
    of the run (with a warning).
    """

    def __init__(self):
        self.adjacency = None
        self.fn: Optional[GraphedTensorFunction] = None
        self.names: List[str] = []
        self.keys: List[str] = []
        self.shapes: List[Tuple[int, ...]] = []
        self.signature = None
        self.validated = False
        self.disabled: Optional[str] = None
        self.captures = 0

    def __call__(self, ctx, state, deviceFn) -> Dict[str, float]:
        return self.launch(ctx, state, deviceFn)()

    def launch(self, ctx, state, deviceFn) -> Callable[[], Dict[str, float]]:
        """Enqueue the diagnostics; the returned thunk reads them back. Work
        the caller does in between overlaps their GPU execution."""
        particles = state.state
        adjacency = getattr(state, 'adjacency', None)
        stepDiag = getattr(state, 'stepDiagnostics', None)
        raw = getattr(stepDiag, 'raw', None)
        signature = (deviceFn, type(stepDiag) if raw is not None else None,
                     raw is not None and raw[2] is not None)
        if self.disabled is not None or adjacency is None or not particles.positions.is_cuda:
            dev = deviceFn(ctx, state)
            return lambda: _readHost(dev)
        if self.adjacency is not adjacency or self.signature != signature:
            # new Verlet generation: eager now (warms the caches), capture next
            self.adjacency, self.signature, self.fn = adjacency, signature, None
            dev = deviceFn(ctx, state)
            return lambda: _readHost(dev)
        if self.fn is None:
            self._build(ctx, state, deviceFn, stepDiag, raw)
        inputs = [getattr(particles, n) for n in self.names] + self._rawTensors(raw)
        out = self.fn(*inputs)
        if self.fn.disabled is not None:
            self.disabled = self.fn.disabled
        elif not self.validated:
            self.validated = True
        keys, shapes = list(self.keys), list(self.shapes)
        return lambda: _unpackHost(keys, shapes, out[0].cpu().tolist()) if keys else {}

    @staticmethod
    def _rawTensors(raw):
        if raw is None:
            return []
        accelAll, fluidMask, nopenshiftDiag = raw
        return [accelAll, fluidMask] + (list(nopenshiftDiag) if nopenshiftDiag is not None else [])

    def _build(self, ctx, state, deviceFn, stepDiag, raw):
        self.names = [n for n in _tensorAttrs(state.state)]
        nNames = len(self.names)
        base, lazyType = state, type(stepDiag)
        hasRaw, hasNopen = raw is not None, raw is not None and raw[2] is not None

        def fn(*tensors):
            view = copy.copy(base)
            view.state = copy.copy(base.state)
            for n, t in zip(self.names, tensors[:nNames]):
                setattr(view.state, n, t)
            if hasRaw:
                r = tensors[nNames:]
                view.stepDiagnostics = lazyType(r[0], r[1], (r[2], r[3]) if hasNopen else None)
            dev = deviceFn(ctx, view)
            if not dev:
                self.keys, self.shapes = [], []
                return (torch.zeros(0, dtype=torch.float64, device=view.state.positions.device),)
            self.keys, self.shapes, flat = _packDevice(dev)
            return (flat,)

        self.fn = GraphedTensorFunction(fn, 'per-step diagnostics', validate=not self.validated)
        self.captures += 1


class GraphedIntegratorStep:
    """Replay a whole integrator step (every RHS evaluation, the state updates
    and `finalize`) from one CUDA graph -- the runner-level extension of
    `GraphedStateFunction`, for steps whose only host decisions are Verlet-list
    validity checks.

    Capture runs ``integrator(state=..., f=..., dt=..., ...)`` on static copies
    of the running state and of the (0-d device) ``dt`` with

    * ``warpSPHIntegrators.util.deferHostTime()`` -- stage times stay tensors
      (the wrapper sets the final host time itself, with the integrator's own
      float32 expression ``float(t + dt)``);
    * ``warpSPHCore.deferVerletChecks()`` -- each Verlet check records its
      device "rebuild?" flag and keeps the prior list;
    * ``config.dt`` pointed at the static ``dt`` (so captured reads of it follow
      later values), and the nested RHS graphs switched off.

    Replay copies the state and ``dt`` in, replays, and reads the OR of the
    Verlet flags together with the new time (one transfer). A set flag means
    an eager step would have rebuilt the list somewhere in this step: the
    replay is discarded and the step runs eagerly (the caller's state was not
    touched). A clear flag means every eager check would have kept the prior
    list, so the replay IS the eager step. Captures are keyed on the adjacency
    object (a rebuild -> recapture on the next step) and the state's tensor
    signature; the first capture of a run is compared bitwise against an eager
    step (state, last-stage update, step statistics, time), and any mismatch or
    capture error turns this off for the run (warning).
    """

    def __init__(self, integratorFn, stepFunction, name='integrator step'):
        self.integratorFn, self.stepFunction, self.name = integratorFn, stepFunction, name
        self.disabled: Optional[str] = None
        self._entry = None
        self._validated = False
        self.captures = self.replays = self.fallbacks = 0
        self.captureSeconds = 0.0
        #: one eager step first: fills the per-run host caches
        #: (`_hasBoundaryParticles`, `_hostDxValue`, device constants, ...)
        #: whose one-time reads would otherwise break the capture
        self._warm = False

    def _disable(self, reason):
        self.disabled = reason
        self._entry = None
        warnings.warn(f'[warpSPH] CUDA graph for {self.name} disabled, running eagerly: {reason}', stacklevel=3)

    def _eager(self, state, dt, config, schemeConfig, verbose):
        return self.integratorFn(state=state, f=self.stepFunction, dt=dt, config=config,
                                 verbose=verbose, schemeConfig=schemeConfig)

    @staticmethod
    def _key(state):
        return (id(state.adjacency),) + tuple((n, tuple(t.shape), t.dtype)
                                              for n, t in _tensorAttrs(state.state).items())

    def __call__(self, state, dt, config, schemeConfig, verbose=False):
        return self.finish(self.launch(state, dt, config, schemeConfig, verbose))

    def launch(self, state, dt, config, schemeConfig, verbose=False):
        """Start the step. A replayed step is only *enqueued* here (the GPU
        runs it while the caller does other work, e.g. the previous step's
        diagnostics on another stream); an eager step runs to completion.
        Pass the handle to `finish`."""
        if (self.disabled is not None or verbose or not isinstance(dt, torch.Tensor)
                or not state.state.positions.is_cuda or not self._warm):
            self._warm = True
            return {'result': self._eager(state, dt, config, schemeConfig, verbose)}
        e = self._entry
        if e is None or e['adjacency'] is not state.adjacency or e['key'] != self._key(state):
            # new Verlet generation (or first step): capture now
            self._entry = None
            if not self._capture(state, dt, config, schemeConfig):
                return {'result': self._eager(state, dt, config, schemeConfig, verbose)}
            e = self._entry
        src = [getattr(state.state, n) for n in e['names']]
        torch._foreach_copy_(e['staticIn'] + [e['staticDt']], src + [dt])
        e['graph'].replay()
        self.replays += 1
        # the integrator's own time expression, float32: float(t + dt); and
        # the OR of the deferred Verlet checks -- read together in `finish`
        pending = torch.stack([e['flag'].to(dt.dtype), (state.t + dt).reshape(())])
        return {'entry': e, 'state': state, 'dt': dt, 'config': config,
                'schemeConfig': schemeConfig, 'pending': pending}

    def finish(self, handle):
        """Complete a step started by `launch` and return its IntegrationResult."""
        if 'result' in handle:
            return handle['result']
        return self._finishReplay(handle['entry'], handle['state'], handle['dt'], handle['config'],
                                  handle['schemeConfig'], handle['pending'])

    def _capture(self, state, dt, config, schemeConfig) -> bool:
        import time
        import warp as wp
        from warpSPHCore import deferVerletChecks
        from warpSPHIntegrators.util import deferHostTime
        t0 = time.perf_counter()
        staticSys = _cloneState(state)
        staticIn = _tensorAttrs(staticSys.state)
        staticDt = dt.detach().clone()
        prevDt = config.dt
        stream = torch.cuda.Stream()
        wstream = wp.stream_from_torch(stream)
        graph = torch.cuda.CUDAGraph()
        try:
            torch.cuda.synchronize()
            config.dt = staticDt
            _OUTER_CAPTURE[0] = True
            with _captureGuard(), deferHostTime(), deferVerletChecks() as flags, \
                    torch.cuda.graph(graph, stream=stream, capture_error_mode=_CAPTURE_MODE), \
                    wp.ScopedStream(wstream, sync_enter=False, sync_exit=False):
                result = self._eager(staticSys, staticDt, config, schemeConfig, False)
                flag = (torch.stack([f.reshape(()) for f in flags]).any() if flags
                        else torch.zeros((), dtype=torch.bool, device=staticDt.device))
                nChecks = len(flags)
            torch.cuda.synchronize()
        except Exception as ex:  # noqa: BLE001 -- any capture failure -> eager
            torch.cuda.synchronize()
            self._disable(f'capture failed ({type(ex).__name__}: {str(ex).splitlines()[0][:200]})')
            return False
        finally:
            _OUTER_CAPTURE[0] = False
            config.dt = prevDt
        final = result.state
        if final.adjacency is not state.adjacency:
            self._disable('the captured step replaced the adjacency object')
            return False
        lastUpdate = result.stages[-1].update if result.stages else None
        self._entry = dict(adjacency=state.adjacency, key=self._key(state), graph=graph,
                           names=list(staticIn), staticIn=[staticIn[n] for n in staticIn],
                           staticDt=staticDt, final=final, lastUpdate=lastUpdate, flag=flag,
                           nChecks=nChecks)
        self.lastCaptureChecks = nChecks
        self.captures += 1
        self.captureSeconds += time.perf_counter() - t0
        return True

    @staticmethod
    def _cloneTree(x):
        if isinstance(x, torch.Tensor):
            return x.clone()
        if isinstance(x, (tuple, list)):
            return type(x)(GraphedIntegratorStep._cloneTree(v) for v in x)
        return x

    def _materialize(self, e, state, tHost):
        """Fresh Python objects around clones of the replayed static outputs."""
        from warpSPHIntegrators.specs import IntegrationResult, StageResult
        final = e['final']
        out = copy.copy(final)
        out.state = copy.copy(final.state)
        for n, t in _tensorAttrs(final.state).items():
            setattr(out.state, n, t.clone())
        out.t = tHost
        out.adjacency = state.adjacency
        diag = getattr(final, 'stepDiagnostics', None)
        if diag is not None and hasattr(diag, 'raw'):
            out.stepDiagnostics = type(diag)(*self._cloneTree(diag.raw))
        upd = e['lastUpdate']
        if upd is not None and dataclasses.is_dataclass(upd):
            fields = {f.name: self._cloneTree(getattr(upd, f.name)) for f in dataclasses.fields(upd)}
            upd = type(upd)(**fields)
        return IntegrationResult(state=out, stages=[StageResult(aux=None, update=upd)])

    def _finishReplay(self, e, state, dt, config, schemeConfig, pending):
        host = pending.cpu().tolist()
        if host[0] != 0.0:
            # a Verlet check fired inside the step: redo it eagerly (rebuild)
            self.fallbacks += 1
            return self._eager(state, dt, config, schemeConfig, False)
        result = self._materialize(e, state, host[1])
        if not self._validated:
            self._validated = True
            eager = self._eager(_cloneState(state), dt, config, schemeConfig, False)
            bad = self._compare(eager, result)
            if bad:
                self._disable('graph replay differs from the eager step in ' + ', '.join(bad[:8]))
                return eager
        return result

    @staticmethod
    def _compare(a, b):
        bad = []
        ta, tb = _tensorAttrs(a.state.state), _tensorAttrs(b.state.state)
        for n in set(ta) | set(tb):
            if n not in ta or n not in tb or not _bitwiseEqual(ta[n], tb[n]):
                bad.append(f'state.{n}')
        if float(a.state.t) != float(b.state.t):
            bad.append('t')
        ua, ub = a.stages[-1].update, b.stages[-1].update
        if ua is not None and dataclasses.is_dataclass(ua):
            for f in dataclasses.fields(ua):
                x, y = getattr(ua, f.name), getattr(ub, f.name)
                if isinstance(x, torch.Tensor) and not _bitwiseEqual(x, y):
                    bad.append(f'update.{f.name}')
        da, db = getattr(a.state, 'stepDiagnostics', None), getattr(b.state, 'stepDiagnostics', None)
        if da is not None and db is not None:
            A, B = dict(da), dict(db)
            if A.keys() != B.keys() or any(not (A[k] == B[k] or (A[k] != A[k] and B[k] != B[k])) for k in A):
                bad.append('stepDiagnostics')
        return bad
