"""Step cost of any registered case (companion to bench_step.py, which is
Marrone-3.1-specific): mean/median wall per step, GPU busy time, launches,
syncs and the top warp kernels, for the lanes / CUDA-graph variants -- and,
with ``--frames``, the cost of one rendered video frame per plot backend.

    python benchmarks/marrone31/bench_case.py --case tgv-wc --nx 128
    python benchmarks/marrone31/bench_case.py --case gresho --nx 128 --variants lanes1 lanes32
    python benchmarks/marrone31/bench_case.py --case tgv-wc --nx 128 --variants graph --frames vispy matplotlib

Each configuration runs in its own subprocess, one at a time.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from bench_step import VARIANTS, _analyzeTrace  # noqa: E402


def worker(a):
    sys.path.insert(0, os.path.join(REPO, 'src'))
    from warpSPHBootstrap import bootstrap
    bootstrap(precision='float32')
    import importlib
    import numpy as np
    import torch
    import warpSPH.cases  # noqa: F401
    from warpSPH.runner import run
    from warpSPH.runner.case import getCase
    for mod in ('tgv', 'tgvWeaklyCompressible', 'greshoVortex', 'kelvinHelmholtz', 'sedov', 'sod',
                'dambreak', 'noh', 'yeeVortex', 'triplePoint'):
        try:
            importlib.import_module(f'warpSPH.cases.{mod}')
        except Exception:  # noqa: BLE001
            pass
    case = getCase(a.case)
    if a.schemeSet:
        # dotted scheme-config overrides, e.g. solverConfig.divergenceFreeSolver.convergenceCheckSchedule=every
        prevCfg = case.configureScheme

        def cfg(ctx, _p=prevCfg):
            if _p is not None:
                _p(ctx)
            for item in a.schemeSet:
                path, val = item.split('=', 1)
                obj = ctx.schemeConfig
                *head, last = path.split('.')
                for h in head:
                    obj = getattr(obj, h)
                cur = getattr(obj, last)
                setattr(obj, last, type(cur)(val) if isinstance(cur, (int, float)) and not isinstance(cur, bool) else val)
        case.configureScheme = cfg

    stepWall, frameT = [], []
    nMain = a.warmup + a.steps
    state = {'last': None, 'prof': None}
    prevPost = case.postStep

    def marker(ctx, st, step):
        now = time.perf_counter()
        if state['last'] is not None:
            stepWall.append(now - state['last'])
        state['last'] = now
        if prevPost is not None:
            prevPost(ctx, st, step)
        if step % 200 == 0:
            print(f'  [{a.label}] step {step} t={float(st.t):.5g}', flush=True)
        if a.profSteps and step == nMain - 1:
            from torch.profiler import ProfilerActivity, profile
            torch.cuda.synchronize()
            state['prof'] = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
            state['prof'].start()
    case.postStep = marker
    if case.updatePlot is not None:
        up = case.updatePlot

        def timedPlot(ctx, st, plotter, step, _up=up):
            t0 = time.perf_counter()
            _up(ctx, st, plotter, step)
            frameT.append(time.perf_counter() - t0)
        case.updatePlot = timedPlot

    kw = dict(nx=a.nx, nSteps=nMain + a.profSteps, quiet=True, store=False, progress=False,
              cudaGraph=a.cudaGraph, pipelineOutputs=os.environ.get('WARPSPH_BENCH_PIPELINE', '1') == '1')
    if a.frames:
        kw.update(plot=True, video=False, plotInterval=a.plotInterval, show=False,
                  plotBackend=a.frames, exportRoot=os.path.join(a.tmp, f'frames_{os.getpid()}'))
    r = run(case, **kw)
    prof = None
    if state['prof'] is not None:
        torch.cuda.synchronize()
        state['prof'].stop()
        path = os.path.join(a.tmp, f'trace_{os.getpid()}.json')
        state['prof'].export_chrome_trace(path)
        prof = _analyzeTrace(path, a.profSteps)
        os.remove(path)
    sw = np.array(stepWall[a.warmup:nMain - 1]) * 1e3
    res = dict(label=a.label, case=a.case, scheme=str(r.ctx.scheme.name), nx=a.nx,
               particles=int(r.state.state.positions.shape[0]),
               periodic=bool(r.ctx.config.domain.periodic.all()) if hasattr(r.ctx.config.domain, 'periodic') else None,
               integrator=str(r.ctx.config.integrationScheme),
               wall_ms_mean=float(sw.mean()), wall_ms=float(np.median(sw)),
               integrator_ms=float(np.median([row['stepTime_ms'] for row in r.trajectory
                                              if a.warmup <= row.get('step', -1) < nMain])))
    g = getattr(r.ctx.schemeConfig, '_rhsGraph', None)
    if g is not None:
        res.update(graph_captures=g.captures, graph_disabled=g.disabled)
    if frameT:
        res.update(frames=a.frames, frame_ms=float(np.median(frameT[1:]) * 1e3) if len(frameT) > 1 else float(frameT[0] * 1e3))
    if prof is not None:
        res.update(prof)
    print('RESULT ' + json.dumps(res), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--case', required=True)
    ap.add_argument('--nx', type=int, required=True)
    ap.add_argument('--variants', nargs='+', default=['lanes1', 'lanes32', 'graph'], choices=sorted(VARIANTS))
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--warmup', type=int, default=50)
    ap.add_argument('--profSteps', type=int, default=20)
    ap.add_argument('--frames', nargs='*', default=None,
                    help='plot backends to time one frame each for (vispy, matplotlib); '
                         'runs extra configurations with plotting on and no profiler')
    ap.add_argument('--plotInterval', type=int, default=20)
    ap.add_argument('--schemeSet', nargs='*', default=[], help='dotted scheme-config overrides path=value')
    ap.add_argument('--tag', default='')
    ap.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    ap.add_argument('--cudaGraph', action='store_true', help=argparse.SUPPRESS)
    ap.add_argument('--label', default='', help=argparse.SUPPRESS)
    ap.add_argument('--tmp', default='/tmp', help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.worker:
        a.frames = a.frames[0] if a.frames else None
        worker(a)
        return

    out = os.path.join(HERE, 'out', time.strftime('%Y-%m-%d_%H-%M-%S') + f'_{a.case}_nx{a.nx}')
    os.makedirs(out, exist_ok=True)
    jobs = [(v, None) for v in a.variants] + [(a.variants[-1], fb) for fb in (a.frames or [])]
    results = []
    for v, fb in jobs:
        env, graph = VARIANTS[v]
        label = v + (f'_frames-{fb}' if fb else '') + (f'_{a.tag}' if a.tag else '')
        cmd = [sys.executable, os.path.abspath(__file__), '--worker', '--case', a.case, '--nx', str(a.nx),
               '--steps', str(a.steps if not fb else min(a.steps, 200)), '--warmup', str(a.warmup),
               '--profSteps', str(0 if fb else a.profSteps), '--label', label, '--tmp', out,
               '--plotInterval', str(a.plotInterval)]
        if graph:
            cmd.append('--cudaGraph')
        if fb:
            cmd += ['--frames', fb]
        if a.schemeSet:
            cmd += ['--schemeSet'] + a.schemeSet
        print(f'== {a.case} nx={a.nx} {label}', flush=True)
        p = subprocess.run(cmd, cwd=REPO, env=dict(os.environ, **env), capture_output=True, text=True)
        rs = [json.loads(l[7:]) for l in p.stdout.splitlines() if l.startswith('RESULT ')]
        if not rs:
            print('   worker failed:\n' + (p.stdout + p.stderr)[-2500:], flush=True)
            continue
        r = rs[-1]
        results.append(r)
        print(f"   -> {r['particles']} ptcl, {r['scheme']}, wall {r['wall_ms_mean']:.2f} ms/step mean "
              f"({r['wall_ms']:.2f} median), GPU busy {r.get('gpu_busy_ms', float('nan')):.2f}, "
              f"warp kernels {r.get('warp_kernel_ms', float('nan')):.2f} ms, "
              f"launches {r.get('kernels', float('nan')):.0f}, syncs {r.get('syncs', float('nan')):.0f}"
              + (f", graph captures {r['graph_captures']} disabled={r['graph_disabled']}" if 'graph_captures' in r else '')
              + (f", frame {r['frame_ms']:.1f} ms ({fb})" if 'frame_ms' in r else ''), flush=True)
        if r.get('top_warp_kernels_us'):
            print('      top kernels (us/step): ' + ', '.join(f'{k} {v:.0f}' for k, v in list(r['top_warp_kernels_us'].items())[:6]), flush=True)
    with open(os.path.join(out, 'results.json'), 'w') as f:
        json.dump(results, f, indent=1)
    print(f'-> {out}')


if __name__ == '__main__':
    main()
