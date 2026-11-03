#!/usr/bin/env python3
"""Plan/run isolated, single-device performance cases without importing torch.

Example: python benchmarks/performance/run.py --device cuda:0 --nx 32 64 --output results/perf/3090
"""
import argparse
from datetime import datetime, timezone
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

OPERATORS = ('density', 'interpolate', 'gradient', 'divergence', 'curl', 'laplacian', 'covariance',
             'crk_volume', 'crk_moments', 'crk_density', 'renorm', 'crk_factors',
             'pinv_symmetric', 'apply_pinv', 'pinv_policy', 'torch_pinv', 'crk_terms')
SCHEMES = ('CRKSPH', 'CompSPH', 'Monaghan', 'GSPH', 'InutsukaGSPH', 'PESPH', 'MFM', 'MFV',
           'deltaSPH', 'sun2017DeltaSPH', 'artificialCompressible', 'iisph', 'dfsph', 'dfsphPredictor', 'dfsphShift')


def positive(value):
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return n


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', default='cpu', help='cpu or cuda:N; one device per invocation')
    p.add_argument('--precision', nargs='+', choices=('float32', 'float64'), default=['float32'])
    p.add_argument('--threads', type=positive, nargs='+', default=[1], help='Torch/OMP/MKL threads, not a guarantee of Warp CPU parallelism')
    p.add_argument('--nx', type=positive, nargs='+', default=[32], help='side length; particle count is nx**dim')
    p.add_argument('--dim', type=int, choices=(2, 3), default=3)
    p.add_argument('--suite', nargs='+', choices=('memory', 'search', 'operators', 'steps', 'rhs'),
                   default=['memory', 'search', 'operators', 'steps'])
    p.add_argument('--operators', nargs='+', choices=(*OPERATORS, 'all'),
                   default=['density', 'gradient', 'divergence', 'laplacian', 'crk_factors', 'pinv_symmetric'])
    p.add_argument('--schemes', nargs='+', choices=SCHEMES, default=['CRKSPH', 'CompSPH', 'MFM', 'MFV'])
    p.add_argument('--neighbors', type=positive, default=55, help='requested microbenchmark lattice degree (actual count is reported)')
    p.add_argument('--kernel', default='Wendland2')
    p.add_argument('--lanes', type=int, choices=(1, 2, 4, 8, 16, 32), nargs='+', default=[1])
    p.add_argument('--traversal', choices=('adjacency', 'grid', 'rebuild'), nargs='+', default=['adjacency'])
    p.add_argument('--memory-mib', type=positive, default=256, help='elements in each memory-control floating array, in MiB')
    p.add_argument('--budget-gib', type=float, default=8., help='conservative estimated working-set ceiling; not an allocator limit')
    p.add_argument('--warmup', type=positive, default=3, help='untimed calls or accepted steps per fresh trajectory')
    p.add_argument('--repeats', type=positive, default=5)
    p.add_argument('--calls', type=positive, default=5, help='calls per microbenchmark sample')
    p.add_argument('--steps', type=positive, default=5, help='accepted steps per trajectory sample')
    p.add_argument('--particle-order', choices=('input', 'morton'), nargs='+', default=['input', 'morton'])
    p.add_argument('--execution', choices=('eager', 'auto', 'graph'), nargs='+', default=['eager'])
    p.add_argument('--neighborhood', choices=('csr', 'chunked'), nargs='+', default=['csr', 'chunked'], help='step/RHS topology representation; chunked supports the driven Morton adapter')
    p.add_argument('--edge-budget', type=positive, default=8_000_000)
    p.add_argument('--max-rows', type=positive, default=65536)
    p.add_argument('--iteration-checks', choices=('every', 'adaptive'), nargs='+', default=['adaptive'])
    p.add_argument('--krylov-check-every', type=positive)
    p.add_argument('--neighbor-omit-i', action=argparse.BooleanOptionalAction, default=None)
    p.add_argument('--collection', choices=('auto', 'fused', 'split'), default='auto')
    p.add_argument('--query-order', choices=('auto', 'input', 'spatial'), default='auto')
    p.add_argument('--verlet-scale', type=float, help='optional simulator neighbour-list skin; same value in every comparison')
    p.add_argument('--backend', choices=('warp', 'native'), default='warp')
    p.add_argument('--operator-backend', choices=('warp', 'native'), help='core operators inside the selected runtime; native operators can run in the Warp simulator')
    p.add_argument('--index-dtype', choices=('int32', 'int64'), default='int32')
    p.add_argument('--offset-dtype', choices=('int32', 'int64'), default='int64')
    p.add_argument('--configuration', type=Path, help='optional versioned RunConfiguration JSON for step/RHS cases; benchmark runtime overrides apply')
    p.add_argument('--profile', action='store_true', help='additional trace pass, excluded from base timings')
    p.add_argument('--timeout', type=positive, default=900, help='seconds per isolated case including compilation')
    p.add_argument('--core-root', type=Path, help='core source checkout, normally resolved from the installed package')
    p.add_argument('--output', type=Path, default=Path('results/performance') / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    p.add_argument('--plan', action='store_true', help='print exact requests, no imports/device work or files')
    return p


def requests(args):
    if args.device != 'cpu' and not (args.device.startswith('cuda:') and args.device[5:].isdigit()):
        raise ValueError('--device must be cpu or cuda:N')
    if not math.isfinite(args.budget_gib) or args.budget_gib <= 0 or (args.verlet_scale is not None and (not math.isfinite(args.verlet_scale) or args.verlet_scale < 1)):
        raise ValueError('positive budget and verlet scale >= 1 required')
    configuration = json.loads(args.configuration.read_text()) if args.configuration else None
    for precision, threads in itertools.product(dict.fromkeys(args.precision), dict.fromkeys(args.threads)):
        base = dict(device=args.device, precision=precision, threads=threads, dim=args.dim, neighbors=args.neighbors,
                    kernel=args.kernel, lanes=1, traversal='adjacency', backend=args.backend,
                    operator_backend=args.operator_backend or args.backend,
                    index_dtype=args.index_dtype, offset_dtype=args.offset_dtype,
                    memory_mib=args.memory_mib, warmup=args.warmup, repeats=args.repeats, calls=args.calls,
                    steps=args.steps, particle_order='input', verlet_scale=args.verlet_scale,
                    execution='eager', neighborhood='csr', edge_budget=args.edge_budget, max_rows=args.max_rows,
                    iteration_checks='adaptive', krylov_check_every=args.krylov_check_every,
                    neighbor_omit_i=args.neighbor_omit_i, collection=args.collection, query_order=args.query_order,
                    profile=args.profile, configuration=configuration)
        for suite in dict.fromkeys(args.suite):
            if suite == 'memory':
                for operation in ('copy', 'triad', 'gather'):
                    yield dict(base, kind='memory', operation=operation, nx=None)
                continue
            for nx in dict.fromkeys(args.nx):
                if suite == 'operators':
                    ops = OPERATORS if 'all' in args.operators else dict.fromkeys(args.operators)
                    for op, traversal, lanes, order in itertools.product(ops, dict.fromkeys(args.traversal), dict.fromkeys(args.lanes), dict.fromkeys(args.particle_order)):
                        yield dict(base, nx=nx, kind='operator', operation=op, traversal=traversal, lanes=lanes, particle_order=order)
                elif suite == 'search':
                    for op, order in itertools.product(('build', 'query', 'rebuild'), dict.fromkeys(args.particle_order)):
                        yield dict(base, nx=nx, kind='search', operation=op, particle_order=order)
                else:
                    for scheme, order, execution, neighborhood, lanes, checks in itertools.product(
                            dict.fromkeys(args.schemes), dict.fromkeys(args.particle_order), dict.fromkeys(args.execution),
                            dict.fromkeys(args.neighborhood), dict.fromkeys(args.lanes), dict.fromkeys(args.iteration_checks)):
                        yield dict(base, nx=nx, kind=suite, operation=scheme, particle_order=order,
                                   execution=execution, neighborhood=neighborhood, lanes=lanes, iteration_checks=checks)


def estimated_bytes(r):
    """A transparent conservative screen, not a promised peak or available-memory check."""
    item = 8 if r['precision'] == 'float64' else 4
    if r['kind'] == 'memory':
        return r['memory_mib'] * 1024**2 * (3 + (8/item if r['operation'] == 'gather' else 0))
    n = r['nx'] ** r['dim']
    degree = r['neighbors'] if r['kind'] in ('operator', 'search') else (100 if r['dim'] == 2 else 300)
    return int(n * (degree * (48 + 8*item) + 256*item) * (2 if r['kind'] in ('steps', 'rhs') else 1))


def core_root(args):
    if args.core_root:
        return args.core_root.resolve()
    spec = importlib.util.find_spec('warpSPHCore')
    if not spec or not spec.origin:
        raise RuntimeError('install the companion repositories in this Python environment first')
    return Path(spec.origin).resolve().parents[2]


def main(argv=None):
    args = parser().parse_args(argv)
    cases = list(requests(args))
    if args.plan:
        print(json.dumps([dict(r, estimated_working_set_bytes=estimated_bytes(r)) for r in cases], indent=2))
        return 0
    root = core_root(args)
    helper = root / 'scripts/benchmarks/bench_single_device.py'
    if not helper.exists():
        raise RuntimeError(f'{helper} missing; update the core checkout to the benchmark commit')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)  # Never silently overwrite or combine different runs.
    (out / 'requests').mkdir()
    (out / 'logs').mkdir()
    manifest = dict(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(), argv=sys.argv,
                    python=sys.executable, core_root=str(root), budget_gib=args.budget_gib, requests=cases)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    results = []
    interrupted = False
    for i, r in enumerate(cases):
        name = f'{i:04d}-{r["kind"]}-{r["operation"]}'
        request_file, result_file = out/'requests'/f'{name}.json', out/f'{name}.json'
        request_file.write_text(json.dumps(r, indent=2) + '\n')
        estimate = estimated_bytes(r)
        print(f'[{i+1}/{len(cases)}] {name} {r["precision"]} nx={r["nx"]} threads={r["threads"]} {r["particle_order"]}/{r["execution"]}/{r["neighborhood"]}', flush=True)
        if estimate > args.budget_gib * 1024**3:
            result = dict(schema_version=1, request=r, status='skipped_capacity',
                          estimated_working_set_bytes=estimate, error='estimate exceeds --budget-gib')
            result_file.write_text(json.dumps(result, indent=2) + '\n')
        else:
            script = helper if r['kind'] in ('operator', 'search', 'memory') else Path(__file__).with_name('steps.py')
            command = [sys.executable, str(script), '--request', str(request_file), '--output', str(result_file)]
            env = dict(os.environ, WARPSPH_BENCH_CORE_ROOT=str(root))
            source_paths = [str(Path(__file__).resolve().parents[2]/'src'), str(root/'src')]
            env['PYTHONPATH'] = os.pathsep.join(source_paths + ([env['PYTHONPATH']] if env.get('PYTHONPATH') else []))
            started = time.perf_counter()
            with (out/'logs'/f'{name}.log').open('w') as log:
                process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    code = process.wait(timeout=args.timeout)
                except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
                    import signal
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    interrupted = isinstance(exc, KeyboardInterrupt)
                    code = -signal.SIGTERM
            if result_file.exists():
                result = json.loads(result_file.read_text())
            else:
                result = dict(schema_version=1, request=r, status='failed',
                              error=f'worker exit {code} (timeout={args.timeout}s); see logs/{name}.log')
            if code and result['status'] != 'failed':
                result.update(status='failed', error=f'worker exited {code} after writing result')
            result['worker_wall_s'] = time.perf_counter() - started
            result_file.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        results.append(result)
        (out/'results.json').write_text(json.dumps(dict(schema_version=1, results=results), indent=2, allow_nan=False) + '\n')
        print(f'  {result["status"]}: {result.get("error", "")}', flush=True)
        if interrupted:
            break
    print(f'Results: {out}/results.json', flush=True)
    return 130 if interrupted else (1 if any(r['status'] == 'failed' for r in results) else 0)


if __name__ == '__main__':
    raise SystemExit(main())
