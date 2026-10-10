"""Count the torch ops (device launches) of ONE eager step per source function, innermost warpSPH frame and its caller chain's top module (stage 3 E8, ANALYTIC_BOUNDARIES_PLAN.md).

    python benchmarks/analyticWalls/count_ops.py --case taylorCouette --scheme deltaSPH --nx 32 [--set k=v ...]

A `TorchDispatchMode` sees every aten op; each is attributed to the innermost frame under `warpSPH/` and to the outermost warpSPH module-level group (modules/analyticBoundary, schemes, systems, ...).
Where the launches of a CPU-bound step come from, i.e. what to fuse into a kernel or a compiled function.
"""
import argparse
import ast
import collections
import os
import sys
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', default='taylorCouette')
    ap.add_argument('--scheme', default='deltaSPH')
    ap.add_argument('--nx', type=int, default=32)
    ap.add_argument('--set', nargs='*', default=[])
    ap.add_argument('--warmup', type=int, default=30)
    ap.add_argument('--dt', type=float, default=2e-3)
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(REPO, 'src'))
    sys.path.insert(0, os.path.join(REPO, 'scripts'))
    from warpSPHBootstrap import bootstrap
    bootstrap(precision='float32')
    warnings.simplefilter('ignore')
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    from warpSPH.cases import importAll
    from warpSPH.runner import getCase, run
    importAll()
    case = getCase(a.case)
    params = {k: ast.literal_eval(v) for k, v in (s.split('=', 1) for s in a.set)}
    inc = a.scheme != 'deltaSPH'
    common = dict(kernel='Wendland2', **({'integrationScheme': 'semiImplicitEuler', 'supportMode': 'SuperSymmetric', 'dt': a.dt, 'adaptiveDt': False} if inc else {}))
    counts, groups, state = collections.Counter(), collections.Counter(), {'mode': None}
    post = case.postStep

    class Counter(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            f = sys._getframe(1)
            inner = None
            top = None
            while f is not None:
                fn = f.f_code.co_filename
                if '/warpSPH/src/warpSPH/' in fn or '/warpSPHCore/src/' in fn:
                    key = f"{fn.split('/src/')[-1]}:{f.f_code.co_name}"
                    if inner is None:
                        inner = key
                    top = key
                f = f.f_back
            counts[inner or '(none)'] += 1
            grp = (inner or '(none)').split(':')[0]
            groups[grp] += 1
            return func(*args, **(kwargs or {}))

    def marker(ctx, st, step):
        if post is not None:
            post(ctx, st, step)
        if step == a.warmup:
            state['mode'] = Counter()
            state['mode'].__enter__()
        elif step == a.warmup + 1 and state['mode'] is not None:
            state['mode'].__exit__(None, None, None)
            state['mode'] = None
    case.postStep = marker
    try:
        run(case, scheme=a.scheme, nx=a.nx, nSteps=a.warmup + 3, quiet=True, store=False, progress=False, plot=False, video=False, show=False, params={**case.params, **params}, **common)
    finally:
        case.postStep = post
        if state['mode'] is not None:
            state['mode'].__exit__(None, None, None)
    total = sum(counts.values())
    print(f'OPS total torch ops in one step: {total}')
    print('OPS by file (innermost warpSPH frame):')
    for k, v in groups.most_common(18):
        print(f'OPS {v:6d} {100 * v / total:5.1f}%  {k}')
    print('OPS by function:')
    for k, v in counts.most_common(22):
        print(f'OPS {v:6d} {100 * v / total:5.1f}%  {k}')


if __name__ == '__main__':
    main()
