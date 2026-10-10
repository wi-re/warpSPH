"""Step-cost benchmark for the analytic-wall cases (stage 3 E8, ANALYTIC_BOUNDARIES_PLAN.md): `taylorCouette`, `stokesArray`, `periodicChannel`, `cylinderWake`, ... on delta+ / omniIncompressible / divergenceFree.

One process per configuration, one at a time (GPU runs are launch-bound and slow each other down). Per configuration, after `--warmup` steps:

* ``wall_ms_mean`` / ``wall_ms``   mean / median wall-clock per step over ``--steps`` steps (the runner's own per-step work, the diagnostics and the case hooks included);
* from a torch-profiler window of ``--profSteps`` steps: ``gpu_busy_ms`` (kernels + memcpy + memset per step), ``warp_kernel_ms``, ``kernels`` (device launches per step), ``syncs`` (stream / device
  synchronisations per step), ``memcpy``, the top warp kernels and the top CPU ops (``cat == cpu_op``, by total time per step: where the Python / torch launch cost goes).

    python benchmarks/analyticWalls/bench_step.py --case taylorCouette --scheme deltaSPH --nx 32 [--set k=v ...] [--cudaGraph] [--steps 300]

Re-uses the trace analysis of ``benchmarks/marrone31/bench_step.py`` (SMALL_PROBLEM_PERFORMANCE.md). Writes the result row as JSON to stdout (``RESULT {...}``).
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))


def _marroneAnalyze():
    spec = importlib.util.spec_from_file_location('marrone31_bench_step', os.path.join(REPO, 'benchmarks', 'marrone31', 'bench_step.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._analyzeTrace


def _topCpuOps(path, nSteps, n=14):
    with open(path) as f:
        tr = json.load(f)
    ev = tr['traceEvents'] if isinstance(tr, dict) else tr
    tot, cnt = collections.defaultdict(float), collections.Counter()
    for e in ev:
        if e.get('cat') == 'cpu_op' and 'dur' in e:
            tot[e['name']] += e['dur']
            cnt[e['name']] += 1
    return {k: dict(us_per_step=round(v / nSteps, 1), calls_per_step=round(cnt[k] / nSteps, 1)) for k, v in sorted(tot.items(), key=lambda kv: -kv[1])[:n]}


def _topKernelsByCount(path, nSteps, n=14, pattern=None):
    with open(path) as f:
        tr = json.load(f)
    ev = tr['traceEvents'] if isinstance(tr, dict) else tr
    cnt, tot = collections.Counter(), collections.defaultdict(float)
    for e in ev:
        if e.get('cat') == 'kernel' and 'dur' in e:
            name = e['name'].split('<')[0][:70]
            cnt[name] += 1
            tot[name] += e['dur']
    items = [(k, c) for k, c in cnt.most_common() if pattern is None or pattern in k]
    return {k: dict(per_step=round(c / nSteps, 1), us_per_step=round(tot[k] / nSteps, 1)) for k, c in items[:n]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--case', default='taylorCouette')
    ap.add_argument('--scheme', default='deltaSPH')
    ap.add_argument('--nx', type=int, default=32)
    ap.add_argument('--nh', type=float, default=None)
    ap.add_argument('--set', nargs='*', default=[], help='case params, k=v (python literals)')
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--warmup', type=int, default=100)
    ap.add_argument('--profSteps', type=int, default=30)
    ap.add_argument('--dt', type=float, default=2e-3, help='incompressible loops: the fixed step')
    ap.add_argument('--cudaGraph', action='store_true')
    ap.add_argument('--lanes', default=None, help='WARPSPHCORE_NEIGHBOR_LANES')
    ap.add_argument('--label', default='')
    ap.add_argument('--syncs', action='store_true', help='locate the host synchronisations of the profSteps window (torch sync-debug mode): count per warpSPH source line')
    ap.add_argument('--cprofile', action='store_true', help='Python-level profile of the profSteps window (cProfile, cumulative time per warpSPH function) instead of the torch profiler')
    a = ap.parse_args()
    if a.lanes:
        os.environ['WARPSPHCORE_NEIGHBOR_LANES'] = a.lanes
    sys.path.insert(0, os.path.join(REPO, 'src'))
    sys.path.insert(0, os.path.join(REPO, 'scripts'))
    from warpSPHBootstrap import bootstrap
    bootstrap(precision='float32')
    import ast
    import numpy as np
    import torch
    from warpSPH.cases import importAll
    from warpSPH.runner import getCase, run
    importAll()
    analyze = _marroneAnalyze()
    case = getCase(a.case)
    params = {k: ast.literal_eval(v) for k, v in (s.split('=', 1) for s in a.set)}
    inc = a.scheme != 'deltaSPH'
    common = dict(kernel='Wendland2', **({'integrationScheme': 'semiImplicitEuler', 'supportMode': 'SuperSymmetric', 'dt': a.dt, 'adaptiveDt': False} if inc else {}))
    if a.nh:
        common['n_h'] = a.nh
    nMain = a.warmup + a.steps
    nTotal = nMain + a.profSteps
    tracePath = f'/tmp/analyticWalls_trace_{os.getpid()}.json'
    stepWall, state = [], {'last': None, 'prof': None}
    post = case.postStep

    def marker(ctx, st, step):
        now = time.perf_counter()
        if state['last'] is not None:
            stepWall.append(now - state['last'])
        state['last'] = now
        if post is not None:
            post(ctx, st, step)
        if step == nMain - 1:
            torch.cuda.synchronize()
            if a.syncs:
                import traceback
                import warnings
                state['syncs'] = collections.Counter()

                def show(message, category, filename, lineno, file=None, line=None):
                    frames = [f for f in traceback.extract_stack() if '/warpSPH' in f.filename and 'bench_step' not in f.filename]
                    if frames:
                        f = frames[-1]
                        state['syncs'][f'{os.path.relpath(f.filename, REPO) if f.filename.startswith(REPO) else f.filename.split("/src/")[-1]}:{f.lineno}  {f.line[:90] if f.line else ""}'] += 1
                warnings.showwarning = show
                warnings.simplefilter('always')
                torch.cuda.set_sync_debug_mode(1)
            elif a.cprofile:
                import cProfile
                state['cprof'] = cProfile.Profile()
                state['cprof'].enable()
            if not a.syncs and not a.cprofile:
                from torch.profiler import ProfilerActivity, profile
                state['prof'] = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
                state['prof'].start()
    case.postStep = marker
    try:
        r = run(case, scheme=a.scheme, nx=a.nx, nSteps=nTotal, quiet=True, store=False, progress=False, plot=False, video=False, show=False, params={**case.params, **params}, cudaGraph=a.cudaGraph, **common)
    finally:
        case.postStep = post
    if a.syncs and state.get('syncs') is not None:
        torch.cuda.set_sync_debug_mode(0)
        print('SYNCS per step | source line (innermost warpSPH frame)')
        for k, v in state['syncs'].most_common(40):
            print(f'SYNCS {v / (nTotal - nMain):6.2f} | {k}')
        return
    if a.cprofile and state.get('cprof') is not None:
        import pstats
        torch.cuda.synchronize()
        state['cprof'].disable()
        ps = pstats.Stats(state['cprof'])
        rows = []
        for (fn, ln, name), (cc, nc, tt, ct, callers) in ps.stats.items():
            if '/warpSPH/' in fn or '/warpSPHCore/' in fn:
                rows.append((ct / a.profSteps * 1e3, nc / a.profSteps, tt / a.profSteps * 1e3, os.path.relpath(fn, REPO) if fn.startswith(REPO) else fn.split('/src/')[-1], ln, name))
        rows.sort(reverse=True)
        print('CPROFILE cumulative ms/step | calls/step | own ms/step | function')
        for ct, nc, tt, fn, ln, name in rows[:45]:
            print(f'CPROFILE {ct:8.2f} | {nc:7.1f} | {tt:7.2f} | {fn}:{ln} {name}')
        return
    prof = None
    if state['prof'] is not None:
        torch.cuda.synchronize()
        state['prof'].stop()
        state['prof'].export_chrome_trace(tracePath)
        prof = analyze(tracePath, a.profSteps)
        prof['top_cpu_ops'] = _topCpuOps(tracePath, a.profSteps)
        prof['top_kernels_by_count'] = _topKernelsByCount(tracePath, a.profSteps)
        prof['triton_kernels'] = _topKernelsByCount(tracePath, a.profSteps, 40, 'triton')
        os.remove(tracePath)
    res = dict(label=a.label, case=a.case, scheme=a.scheme, nx=a.nx, cudaGraph=a.cudaGraph, lanes=os.environ.get('WARPSPHCORE_NEIGHBOR_LANES', 'default'),
               particles=int(r.state.state.positions.shape[0]), fluid=int((r.state.state.kinds == 0).sum()), steps=a.steps, warmup=a.warmup,
               wall_ms=float(np.median(stepWall[a.warmup:nMain - 1]) * 1e3), wall_ms_mean=float(np.mean(stepWall[a.warmup:nMain - 1]) * 1e3), diverged=bool(r.diverged))
    g = getattr(r.ctx.schemeConfig, '_rhsGraph', None)
    if g is not None:
        res.update(graph_captures=g.captures, graph_replays=g.replays, graph_disabled=g.disabled)
    sg = r.ctx.scratch.get('stepGraph')
    if sg is not None:
        res.update(step_graph=dict(captures=sg.captures, replays=sg.replays, fallbacks=sg.fallbacks, disabled=sg.disabled))
    elif a.cudaGraph:
        res.update(step_graph='not engaged (spec.store / _stepGraphable)')
    if prof:
        res.update(prof)
    print('RESULT ' + json.dumps(res), flush=True)


if __name__ == '__main__':
    main()
