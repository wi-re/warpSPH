"""Bitwise baselines for a scheme cleanup: dump final states, compare two dumps, time a case. The recipe of `SCHEME_CLEANUP_PLAN.md`.

    # 1. BEFORE touching a scheme: dump its final state on every case that runs under it (eager, and graph where the scheme has one)
    python scripts/scheme_baseline.py dump --scheme Monaghan --out /tmp/mon_old --steps 20
    python scripts/scheme_baseline.py dump --scheme sun2017DeltaSPH --out /tmp/wc_old --steps 12 --graph          # also cudaGraph=True runs
    python scripts/scheme_baseline.py dump --scheme MFV --out /tmp/mesh_old --temporal hancock ssprk kdk           # a config axis
    # 2. edit, then dump again to another directory and compare (relative to the largest value; 0 everywhere = bitwise)
    python scripts/scheme_baseline.py compare /tmp/mon_old /tmp/mon_new
    # 3. wall time per step (warm-up run first), new tree vs a worktree of the old commit:  git worktree add /tmp/wt HEAD
    python scripts/scheme_baseline.py time --scheme CRKSPH --case sod2d --nx 400 --mode graph
    PYTHONPATH=/tmp/wt/src python scripts/scheme_baseline.py time --scheme CRKSPH --case sod2d --nx 400 --mode graph

Pitfalls (each cost time once): compare GRAPH against EAGER of the same tree too, not only old against new (a latent graph bug was hidden by an unused
`.clone()`; OPEN_PROBLEMS 35); a case that is nondeterministic in the OLD code (`tgv-wc`: old vs old differs by 9e-2) carries no signal; the `Iter k/16
Sampling Error` log lines of the case setup match a grep for "Error"; the cases a scheme cannot run (walls, Dirichlet BCs) fail the same way before and after.
"""
import argparse
import dataclasses
import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _runCase(case, scheme, steps, cudaGraph=None, setCfg=None, params=None, nx=None, stepMode=None):
    from warpSPH.runner import run
    if setCfg or stepMode:
        inner = case.configureScheme

        def configure(ctx, *a, **k):
            out = inner(ctx, *a, **k) if inner else None
            for key, value in (setCfg or {}).items():
                obj, _, attr = key.rpartition('.')
                target = ctx.schemeConfig
                for part in obj.split('.') if obj else []:
                    target = getattr(target, part)
                setattr(target, attr, value)
            if stepMode:
                ctx.schemeConfig.stepMode = stepMode
            return out
        case = dataclasses.replace(case, configureScheme=configure)
    kw = {} if cudaGraph is None else {'cudaGraph': cudaGraph}
    if nx is not None:
        kw['nx'] = nx
    return run(case, quiet=True, progress=False, nSteps=steps, scheme=scheme, plot=False, video=False, params=params or {}, **kw)


def _save(r, path):
    import torch
    st = r.state.state
    fields = ('positions', 'velocities', 'densities', 'internalEnergies', 'pressures', 'supports', 'Q', 'lam')
    torch.save({k: getattr(st, k).detach().cpu() for k in fields if getattr(st, k, None) is not None}, path)


def _literal(raw):
    import ast
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return raw


def cmdDump(a):
    from warpSPHBootstrap import bootstrap
    bootstrap(precision='float32')
    from warpSPH.cases import importAll
    from warpSPH.runner.case import getCase, listCases
    importAll()
    os.makedirs(a.out, exist_ok=True)
    cases = a.cases or sorted(listCases())
    setCfg = dict(s.split('=', 1) for s in a.set)
    setCfg = {k: _literal(v) for k, v in setCfg.items()}
    failed = []
    for c in cases:
        variants = [('', {})] + [(f'_{t}', {'temporalScheme': t}) for t in a.temporal] + [(f'_{s}', {'viscositySwitch': s}) for s in a.switches]
        for tag, extra in variants:
            cfg = dict(setCfg); params = {}
            for k, v in extra.items():
                (params if k == 'viscositySwitch' else cfg)[k] = v
            try:
                _save(_runCase(getCase(c), a.scheme, a.steps, setCfg=cfg, params=params), f'{a.out}/{c}{tag}.pt')
                if a.graph:
                    _save(_runCase(getCase(c), a.scheme, a.steps, cudaGraph=True, setCfg=cfg, params=params), f'{a.out}/{c}{tag}_graph.pt')
            except Exception as e:                               # a case the scheme cannot run: it fails the same way before and after
                failed.append(f'{c}{tag}: {type(e).__name__}: {str(e).splitlines()[0][:60]}')
    print(f'{len(glob.glob(a.out + "/*.pt"))} states in {a.out}; cases that did not run: {failed if failed else "none"}')


def cmdCompare(a):
    import torch
    rows = []
    for f in sorted(glob.glob(a.a + '/*.pt')):
        g = f.replace(a.a, a.b, 1)
        if not os.path.exists(g):
            rows.append((os.path.basename(f)[:-3], float('nan'))); continue
        A, B = torch.load(f), torch.load(g)
        rows.append((os.path.basename(f)[:-3], max(((A[k] - B[k]).abs().max() / A[k].abs().max().clamp(min=1e-30)).item() for k in A if k in B)))
    same = [n for n, d in rows if d == 0]
    print(f'{len(rows)} states compared, {len(same)} bitwise identical')
    for n, d in rows:
        if d != 0:
            print(f'  {n:40s} max relative difference {d:.2e}')


def cmdTime(a):
    from warpSPHBootstrap import bootstrap
    bootstrap(precision='float32')
    import warpSPH
    from warpSPH.cases import importAll
    from warpSPH.runner.case import getCase
    importAll()
    case = getCase(a.case)
    graph = {'eager': False, 'graph': True}.get(a.mode)                # delta-SPH / ACSPH: the case-level cudaGraph flag
    stepMode = a.mode if a.mode in ('eager', 'graph', 'auto') and a.scheme in ('CompSPH', 'CRKSPH') else None   # CompSPH / CRKSPH: stepMode
    if stepMode:
        graph = None
    _runCase(case, a.scheme, 40, cudaGraph=graph, nx=a.nx, stepMode=stepMode)
    r = _runCase(case, a.scheme, a.steps, cudaGraph=graph, nx=a.nx, stepMode=stepMode)
    print(f'{warpSPH.__file__.split("/src")[0][-24:]} {a.scheme} {a.case} nx={a.nx} N={r.state.state.positions.shape[0]} {a.mode}: {1000 * r.wallTime / r.nSteps:.2f} ms/step')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    d = sub.add_parser('dump'); d.add_argument('--scheme', required=True); d.add_argument('--out', required=True)
    d.add_argument('--steps', type=int, default=20); d.add_argument('--cases', nargs='*'); d.add_argument('--graph', action='store_true')
    d.add_argument('--temporal', nargs='*', default=[]); d.add_argument('--switches', nargs='*', default=[])
    d.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE', help='scheme-config attributes (dotted), value read as a Python literal')
    d.set_defaults(fn=cmdDump)
    c = sub.add_parser('compare'); c.add_argument('a'); c.add_argument('b'); c.set_defaults(fn=cmdCompare)
    t = sub.add_parser('time'); t.add_argument('--scheme', required=True); t.add_argument('--case', required=True); t.add_argument('--nx', type=int, default=None)
    t.add_argument('--steps', type=int, default=250); t.add_argument('--mode', default='eager', choices=['eager', 'graph', 'auto']); t.set_defaults(fn=cmdTime)
    args = ap.parse_args()
    args.fn(args)
