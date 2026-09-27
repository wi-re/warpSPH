"""Step-cost benchmark for small problems: the Marrone 2011 Sec. 3.1 dam break
exactly as `scripts/probe_deltaSPHMarrone.py` runs it (sun2017DeltaSPH,
symplectic Euler, Wendland2, freeSlip walls, english2025 mDBC, the probe's
post-step mDBC density hook and the case's per-step diagnostics).

What it measures, per configuration (one subprocess each, run one at a time --
GPU runs are launch-bound and slow each other down):

* ``wall_ms_mean`` / ``wall_ms``  mean / median wall-clock per step over
                     ``--steps`` steps after ``--warmup`` (includes the runner's
                     own per-step work, the post-step hook and the diagnostics,
                     i.e. what a real probe run pays). Read the MEAN: periodic
                     costs (a video frame every ``plotInterval`` steps, a
                     Verlet rebuild + graph re-capture every ~17 steps once a
                     dam break is violent) are invisible in the median. A
                     400-step window at t ~ 0 also under-samples rebuilds --
                     full-run cost is higher (``SMALL_PROBLEM_PERFORMANCE.md``);
* ``integrator_ms``  the runner's CUDA-event ``stepTime_ms`` (the integrator
                     call only);
* ``hook_ms`` / ``dt_ms`` / ``diag_ms``  median cost of the case's
                     ``postStep`` / ``timestep`` / ``diagnostics`` hooks;
* from a torch-profiler window of ``--profSteps`` steps at the end:
  ``gpu_busy_ms`` (sum of kernel + memcpy + memset durations per step),
  ``warp_kernel_ms`` (warp kernels only), ``kernels`` (device kernel launches
  per step), ``syncs`` (``cudaStreamSynchronize``/``cudaDeviceSynchronize``
  calls per step) and ``memcpy`` (``cudaMemcpy*`` calls per step).

The profiler window is measured separately from ``wall_ms`` (the profiler
itself adds CPU overhead).

Usage (from the repo root, `warp` env)::

    python benchmarks/marrone31/bench_step.py                       # nx 40 70 140, all variants
    python benchmarks/marrone31/bench_step.py --nx 70 --variants lanes32
    python benchmarks/marrone31/bench_step.py --nx 70 --variants lanes32 --no-diag --no-hook

Writes ``<out>/results.json`` and ``<out>/summary.md`` (default out:
``benchmarks/marrone31/out/<timestamp>``, gitignored).
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

#: name -> (environment overrides, CUDA-graph RHS on/off)
VARIANTS = {
    'lanes1': ({'WARPSPHCORE_NEIGHBOR_LANES': '1'}, False),   # thread-per-particle kernels
    'lanes32': ({'WARPSPHCORE_NEIGHBOR_LANES': '32'}, False),  # multi-lane kernels, eager
    'graph': ({'WARPSPHCORE_NEIGHBOR_LANES': '32'}, True),     # + CUDA-graph step, outputs pipelined
    # the same without overlapping step n's outputs with step n+1
    'graphSeq': ({'WARPSPHCORE_NEIGHBOR_LANES': '32', 'WARPSPH_BENCH_PIPELINE': '0'}, True),
}


def _gitSha(path):
    try:
        sha = subprocess.run(['git', '-C', path, 'rev-parse', '--short', 'HEAD'],
                             capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(['git', '-C', path, 'status', '--porcelain', '--untracked-files=no'],
                               capture_output=True, text=True).stdout.strip()
        return sha + ('+dirty' if dirty else '')
    except Exception:  # noqa: BLE001
        return '?'


# ----------------------------------------------------------------------------
# worker: one configuration, in its own process
# ----------------------------------------------------------------------------

def _analyzeTrace(path, nSteps):
    import collections
    with open(path) as f:
        tr = json.load(f)
    ev = tr['traceEvents'] if isinstance(tr, dict) else tr
    dev = [e for e in ev if e.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset') and 'dur' in e]
    kern = [e for e in ev if e.get('cat') == 'kernel' and 'dur' in e]
    warp = [e for e in kern if 'cuda_kernel' in e.get('name', '')]
    api = collections.Counter(e['name'] for e in ev if e.get('cat') in ('cuda_runtime', 'cuda_driver'))
    syncs = sum(c for n, c in api.items() if n in ('cudaStreamSynchronize', 'cudaDeviceSynchronize',
                                                   'cuStreamSynchronize', 'cuCtxSynchronize'))
    memcpy = sum(c for n, c in api.items() if n.startswith('cudaMemcpy') or n.startswith('cuMemcpy'))
    perKernel = collections.defaultdict(float)
    for e in warp:
        perKernel[e['name'].split('_Kernel')[0].replace('_cuda_kernel_forward', '')] += e['dur'] / nSteps
    return dict(
        gpu_busy_ms=sum(e['dur'] for e in dev) / 1e3 / nSteps,
        warp_kernel_ms=sum(e['dur'] for e in warp) / 1e3 / nSteps,
        kernels=len(kern) / nSteps,
        warp_kernels=len(warp) / nSteps,
        syncs=syncs / nSteps,
        memcpy=memcpy / nSteps,
        top_warp_kernels_us={k: round(v, 1) for k, v in
                             sorted(perKernel.items(), key=lambda kv: -kv[1])[:12]},
    )


def worker(a):
    sys.path.insert(0, os.path.join(REPO, 'src'))
    sys.path.insert(0, os.path.join(REPO, 'scripts'))
    from warpSPHBootstrap import bootstrap
    bootstrap(precision='float32')
    import numpy as np
    import torch
    from warpSPH.cases.dambreak import dambreakCase
    from warpSPH.runner import run
    import probe_deltaSPHMarrone as P

    if not a.noHook:
        from _mdbcDensityHook import installMdbcDensityToState
        installMdbcDensityToState(dambreakCase)

    params = dict(W=P.TANK_W, fillRatio=P.H / P.TANK_L, fluidWidth=P.COL_W / P.TANK_W,
                  gravityMagnitude=P.G, pressureProbeHeights=P.PROBE_HEIGHTS,
                  pressureProbeInset=0.0, pressureProbeDiscRadius=P.PROBE_DISC_RADIUS,
                  referenceVelocity=P.U_MAX, machTarget=1.95 / a.c0Ratio)

    hookT, dtT, diagT, stepWall = [], [], [], []

    def timed(store, fn):
        if fn is None:
            return None

        def w(*args, **kw):
            t0 = time.perf_counter()
            out = fn(*args, **kw)
            store.append(time.perf_counter() - t0)
            return out
        return w

    dambreakCase.timestep = timed(dtT, dambreakCase.timestep)
    dambreakCase.diagnostics = None if a.noDiag else timed(diagT, dambreakCase.diagnostics)
    post = timed(hookT, dambreakCase.postStep)
    nMain = a.warmup + a.steps
    nTotal = nMain + a.profSteps
    tracePath = os.path.join(a.tmp, f'trace_{os.getpid()}.json')
    state = {'last': None, 'prof': None}

    def marker(ctx, st, step):
        now = time.perf_counter()
        if state['last'] is not None:
            stepWall.append(now - state['last'])
        state['last'] = now
        if post is not None:
            post(ctx, st, step)
        if step % a.printEvery == 0:
            recent = stepWall[-a.printEvery:]
            ms = 1e3 * sum(recent) / max(1, len(recent))
            print(f'  [{a.label}] step {step:6d}  t={float(st.t):.5f}  {ms:.2f} ms/step', flush=True)
        if a.profSteps and step == nMain - 1:
            from torch.profiler import ProfilerActivity, profile
            torch.cuda.synchronize()
            state['prof'] = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
            state['prof'].start()
    dambreakCase.postStep = marker

    kw = dict(scheme=a.scheme, L=P.TANK_L, nx=a.nx, nSteps=nTotal,
              quiet=True, store=False, progress=False, params=params, cudaGraph=a.cudaGraph,
              pipelineOutputs=os.environ.get('WARPSPH_BENCH_PIPELINE', '1') == '1')
    if a.progress:
        # the runner's per-step progress rows (repo rule: stream progress)
        kw.update(quiet=False, progress=True)
    if a.video:
        # what every probe run pays (repo rule: video on): a vispy frame every
        # `plotInterval` steps; the final encode is outside the timed loop
        kw.update(plot=True, video=True, plotInterval=a.plotInterval,
                  exportRoot=os.path.join(a.tmp, f'video_{os.getpid()}'),
                  # WARPSPH_BENCH_SHOW=0: no live window (-> render thread,
                  # unless WARPSPH_BENCH_ASYNCPLOT=0)
                  show=os.environ.get('WARPSPH_BENCH_SHOW', '1') == '1',
                  asyncPlot=os.environ.get('WARPSPH_BENCH_ASYNCPLOT', '1') == '1')
    tStart = time.perf_counter()
    r = run(dambreakCase, **kw)
    wall = time.perf_counter() - tStart
    prof = None
    if state['prof'] is not None:
        torch.cuda.synchronize()
        state['prof'].stop()
        state['prof'].export_chrome_trace(tracePath)
        prof = _analyzeTrace(tracePath, a.profSteps)
        os.remove(tracePath)

    def med(xs, lo, hi):
        xs = xs[lo:hi]
        return float(np.median(xs) * 1e3) if xs else 0.0

    # stepWall[k] is the gap between step k and k+1's postStep -> one full step.
    res = dict(
        label=a.label, nx=a.nx, scheme=a.scheme, c0Ratio=a.c0Ratio,
        lanes=os.environ.get('WARPSPHCORE_NEIGHBOR_LANES', 'default'),
        diag=not a.noDiag, hook=not a.noHook, cudaGraph=a.cudaGraph,
        video=a.video, plotInterval=a.plotInterval if a.video else None, progress=a.progress,
        particles=int(r.state.state.positions.shape[0]),
        fluid=int((r.state.state.kinds == 0).sum()),
        steps=a.steps, warmup=a.warmup,
        wall_ms=med(stepWall, a.warmup, nMain - 1),
        wall_ms_mean=float(np.mean(stepWall[a.warmup:nMain - 1]) * 1e3),
        integrator_ms=float(np.median([row['stepTime_ms'] for row in r.trajectory
                                       if a.warmup <= row.get('step', -1) < nMain])),
        hook_ms=med(hookT, a.warmup, nMain), dt_ms=med(dtT, a.warmup, nMain),
        diag_ms=med(diagT, a.warmup, nMain),
        sim_t_end=float(r.state.t), run_wall_s=wall, diverged=bool(r.diverged),
    )
    g = getattr(r.ctx.schemeConfig, '_rhsGraph', None)
    if g is not None:
        res.update(graph_captures=g.captures, graph_replays=g.replays, graph_disabled=g.disabled,
                   graph_validations=g.validations, graph_capture_s=g.captureSeconds)
    if prof is not None:
        res.update(prof)
    print('RESULT ' + json.dumps(res), flush=True)


# ----------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------

def _fmtRow(r):
    return (f"| {r['label']} | {r['nx']} | {r['particles']} | {r['wall_ms_mean']:.2f} | {r['wall_ms']:.2f} | {r['integrator_ms']:.2f} "
            f"| {r['hook_ms']:.2f} | {r['diag_ms']:.2f} | {r.get('gpu_busy_ms', float('nan')):.2f} "
            f"| {r.get('warp_kernel_ms', float('nan')):.2f} | {r.get('kernels', float('nan')):.0f} "
            f"| {r.get('syncs', float('nan')):.0f} | {r.get('memcpy', float('nan')):.0f} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--nx', type=int, nargs='+', default=[40, 70, 140])
    ap.add_argument('--variants', nargs='+', default=['lanes1', 'lanes32', 'graph'], choices=sorted(VARIANTS))
    ap.add_argument('--steps', type=int, default=400)
    ap.add_argument('--warmup', type=int, default=100)
    ap.add_argument('--profSteps', type=int, default=30)
    ap.add_argument('--scheme', default='sun2017DeltaSPH')
    ap.add_argument('--c0Ratio', type=float, default=40.0)
    ap.add_argument('--no-diag', dest='noDiag', action='store_true')
    ap.add_argument('--no-hook', dest='noHook', action='store_true')
    ap.add_argument('--printEvery', type=int, default=100)
    ap.add_argument('--video', action='store_true', help='render frames like a probe run (vispy)')
    ap.add_argument('--plotInterval', type=int, default=20)
    ap.add_argument('--progress', action='store_true', help="the runner's per-step progress rows on")
    ap.add_argument('--out', default=None)
    ap.add_argument('--tag', default='')
    ap.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    ap.add_argument('--cudaGraph', action='store_true', help=argparse.SUPPRESS)
    ap.add_argument('--label', default='', help=argparse.SUPPRESS)
    ap.add_argument('--tmp', default='/tmp', help=argparse.SUPPRESS)
    a = ap.parse_args()

    if a.worker:
        a.nx = a.nx[0]
        worker(a)
        return

    out = a.out or os.path.join(HERE, 'out', time.strftime('%Y-%m-%d_%H-%M-%S') + (f'_{a.tag}' if a.tag else ''))
    os.makedirs(out, exist_ok=True)
    meta = dict(warpSPH=_gitSha(REPO), warpSPHCore=_gitSha(os.path.join(REPO, '..', 'warpSPHCore')),
                args=vars(a), started=time.strftime('%Y-%m-%d %H:%M:%S'))
    try:
        meta['gpu'] = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                                     capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    results = []
    for nx in a.nx:
        for v in a.variants:
            label = v + ('_nodiag' if a.noDiag else '') + ('_nohook' if a.noHook else '') \
                + (f'_video{a.plotInterval}' if a.video else '') + ('_progress' if a.progress else '')
            cmd = [sys.executable, os.path.abspath(__file__), '--worker', '--nx', str(nx),
                   '--steps', str(a.steps), '--warmup', str(a.warmup), '--profSteps', str(a.profSteps),
                   '--scheme', a.scheme, '--c0Ratio', str(a.c0Ratio), '--label', label,
                   '--printEvery', str(a.printEvery), '--tmp', out]
            envOverrides, graph = VARIANTS[v]
            if graph:
                cmd.append('--cudaGraph')
            if a.video:
                cmd += ['--video', '--plotInterval', str(a.plotInterval)]
            if a.progress:
                cmd.append('--progress')
            if a.noDiag:
                cmd.append('--no-diag')
            if a.noHook:
                cmd.append('--no-hook')
            env = dict(os.environ, **envOverrides)
            print(f'== {label} nx={nx}', flush=True)
            proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                if line.startswith('RESULT '):
                    results.append(json.loads(line[7:]))
                elif line.startswith('  [') or 'Error' in line or 'Traceback' in line:
                    print(line, end='', flush=True)
            proc.wait()
            if proc.returncode != 0:
                print(f'   worker failed (exit {proc.returncode})', flush=True)
            elif results:
                r = results[-1]
                print(f"   -> wall {r['wall_ms_mean']:.2f} ms/step mean ({r['wall_ms']:.2f} median), integrator {r['integrator_ms']:.2f}, "
                      f"gpu busy {r.get('gpu_busy_ms', float('nan')):.2f}, "
                      f"kernels {r.get('kernels', float('nan')):.0f}/step, syncs {r.get('syncs', float('nan')):.0f}/step",
                      flush=True)
            with open(os.path.join(out, 'results.json'), 'w') as f:
                json.dump(dict(meta=meta, results=results), f, indent=1)

    lines = [f"# Marrone 3.1 step cost -- {meta['started']}", '',
             f"warpSPH {meta['warpSPH']}, warpSPHCore {meta['warpSPHCore']}, GPU {meta.get('gpu', '?')}", '',
             '| variant | nx | particles | wall ms/step (mean) | (median) | integrator ms | hook ms | diag ms '
             '| GPU busy ms | warp kernel ms | kernels/step | syncs/step | memcpy/step |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    lines += [_fmtRow(r) for r in results]
    with open(os.path.join(out, 'summary.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'-> {out}')


if __name__ == '__main__':
    main()
