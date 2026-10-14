"""Micro-benchmark one callable: torch ops, warp launches, host syncs and GPU / host time per call.

`callAudit` (warpSPHCore) answers "which section of a step issues what"; this answers the same for ONE
module call in isolation, so two implementations of the same operator (or the eleven viscosity switches)
can be compared on one table. A call is measured twice, never at once (the audit's dispatch hook slows
the host ~3x):

* counting pass, audit ON: ops / launches / syncs per call, the Python call site of every sync, the
  section tree of the call (every `record_function` inside it);
* timing pass, audit OFF: host enqueue time per call (no sync between calls), wall time per call (one
  sync at the end), and from a torch-profiler window the GPU busy time and device kernel launches per call.

    from warpSPH.utils.microBench import microBench, formatTable
    row = microBench(lambda: computeViscositySwitchTerms(dt, state, config, schemeConfig, ss, adjacency), 'CullenHopkins terms')
    print(formatTable([row]))

The callable must be repeatable (same inputs every time, no growing state) and should return nothing the
caller needs; its first `warmup` calls are discarded (kernel compilation, cache fills). Read the counts as
exact and the times as indicative: they are for one GPU, one particle count, a warm cache.
"""

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch

__all__ = ['MicroResult', 'microBench', 'formatTable', 'formatSyncSites']


@dataclass
class MicroResult:
    label: str
    ops: float = 0.0               # aten ops per call
    launches: float = 0.0          # warp kernels launched through wp.launch per call
    syncs: float = 0.0             # host synchronisations per call
    syncSites: Dict[str, float] = field(default_factory=dict)     # 'file.py:line function' -> syncs per call
    opNames: Dict[str, float] = field(default_factory=dict)       # aten op -> count per call
    kernelNames: Dict[str, float] = field(default_factory=dict)   # warp kernel -> launches per call
    tree: str = ''                 # callAudit's section tree of the counting pass
    hostUs: float = float('nan')   # host time to enqueue one call (no sync between calls)
    wallUs: float = float('nan')   # wall time per call (one sync at the end of the window)
    gpuUs: float = float('nan')    # GPU busy time per call (kernels + memcpy + memset, torch profiler)
    deviceKernels: float = float('nan')   # device kernel / memcpy / memset launches per call (profiler)


def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _countingPass(fn, calls, device, torchOps) -> Dict:
    from warpSPHCore import callAudit
    callAudit.reset()
    callAudit.enable(torchOps=torchOps)
    callAudit.reset()
    try:
        for _ in range(calls):
            with callAudit.section('microBench'):
                fn()
        _sync(device)
        # the one private touch: the aggregate of the section just recorded
        agg = callAudit._sections['microBench']
        per = lambda n: n / calls
        out = dict(
            ops=per(agg.ops), launches=per(agg.launches), syncs=per(agg.syncs),
            syncSites={k: per(v) for k, v in agg.syncSites.most_common()},
            opNames={k: per(v) for k, v in agg.opNames.most_common()},
            kernelNames={k: per(v) for k, v in agg.kernelNames.most_common()},
            tree=callAudit.treeReport(),
        )
    finally:
        callAudit.disable()
        callAudit.reset()
    return out


def _timingPass(fn, calls, device) -> Dict:
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(calls):
        fn()
    t1 = time.perf_counter()
    _sync(device)
    t2 = time.perf_counter()
    out = dict(hostUs=1e6 * (t1 - t0) / calls, wallUs=1e6 * (t2 - t0) / calls)
    if device.type != 'cuda':
        return out
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(calls):
            fn()
        _sync(device)
    busy, launches = 0.0, 0
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            busy += getattr(e, 'device_time', None) if hasattr(e, 'device_time') else e.cuda_time
            launches += 1
    out.update(gpuUs=busy / calls, deviceKernels=launches / calls)
    return out


def microBench(fn: Callable[[], object], label: str = '', *, warmup: int = 3, calls: int = 10, timedCalls: int = 50,
               device: Optional[torch.device] = None, torchOps: bool = True) -> MicroResult:
    """Count and time `fn` (see the module doc). `device` defaults to CUDA when available."""
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for _ in range(warmup):
        fn()
    _sync(device)
    counted = _countingPass(fn, calls, device, torchOps)
    timed = _timingPass(fn, timedCalls, device)
    return MicroResult(label=label, **counted, **timed)


def formatTable(rows: List[MicroResult]) -> str:
    """One line per result: counts per call, then host / wall / GPU microseconds per call."""
    w = max([len(r.label) for r in rows] + [5])
    head = f"{'':<{w}}  {'ops':>6} {'launch':>6} {'sync':>5}   {'host us':>8} {'wall us':>8} {'gpu us':>8} {'dev k':>6}"
    lines = [head, '-' * len(head)]
    for r in rows:
        lines.append(f"{r.label:<{w}}  {r.ops:6.1f} {r.launches:6.1f} {r.syncs:5.2f}   {r.hostUs:8.0f} {r.wallUs:8.0f} "
                     f"{r.gpuUs:8.0f} {r.deviceKernels:6.1f}")
    return '\n'.join(lines)


def formatSyncSites(rows: List[MicroResult], top: int = 4) -> str:
    """The Python call sites of the syncs, per result (only results that sync)."""
    lines = []
    for r in rows:
        if r.syncs > 0:
            sites = ', '.join(f'{k} x{v:g}' for k, v in list(r.syncSites.items())[:top])
            lines.append(f'{r.label}: {sites}')
    return '\n'.join(lines)
