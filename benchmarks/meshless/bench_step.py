"""Step-cost benchmark for the meshless MFM / MFV schemes (COMPRESSIBLE_SPH_PLAN Track E).

The KH case at the battery's defaults (B7, n_h 4, nx 256 -> N = 65,536, float32), stepped
with the frontend's own pieces -- `modules/timestep/meshless.computeTimestep` and
`schemes/meshless.meshlessStep` -- and the integrator's exact-delta update applied in place
(pos += dt dxdt with the periodic wrap, Q += dt dQdt). No runner, no diagnostics, no video:
this measures the scheme step only (the runner's own overhead is measured by its
`stepTime_ms`).

``--state`` picks the starting state: ``ic`` (the lattice IC, a cheap closure) or the path
of a saved KH 256 dump (``positions`` / ``supports`` / ``Q``; e.g. the step-3000 MFM state
in ``.tmp/``, mid roll-up, where the closure solve is expensive).

Reports (per step, after ``--warmup``): mean / median wall ms (CUDA-synchronised), and from
a torch-profiler window of ``--profSteps`` steps: GPU-busy ms, kernel launches, host syncs
(``cudaStreamSynchronize`` / ``cudaDeviceSynchronize`` / blocking ``cudaMemcpy``) and the
CPU / CUDA time per ``record_function`` region (the core labels its launches
``warpSPH[MFM...]``; this script adds ``bench[...]`` regions around the step's phases).
``--save PATH`` writes the final ``(positions, Q, supports, lam)`` for a bitwise / round-off
comparison between code versions (``--compare PATH``).

Usage (repo root, warp env)::

    python benchmarks/meshless/bench_step.py --scheme MFM --state ic --steps 20
    python benchmarks/meshless/bench_step.py --scheme MFM --state .tmp/kh256_state_step003000_periodic.pt
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, 'src'))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--scheme', default='MFM', choices=['MFM', 'MFV'])
    ap.add_argument('--nx', type=int, default=256)
    ap.add_argument('--state', default='ic')
    ap.add_argument('--steps', type=int, default=20)
    ap.add_argument('--warmup', type=int, default=3)
    ap.add_argument('--profSteps', type=int, default=3)
    ap.add_argument('--precision', default='float32')
    ap.add_argument('--save', default=None)
    ap.add_argument('--topKernels', type=int, default=15)
    ap.add_argument('--cprofile', type=int, default=0, help='print the top-N cumulative cProfile entries')
    ap.add_argument('--compare', default=None)
    ap.add_argument('--fixedDt', type=float, default=None,
                    help='use this dt instead of the CFL dt (keeps two code versions on one trajectory)')
    args = ap.parse_args()
    os.environ['warpSPHCore_PRECISION'] = args.precision

    import torch
    import warp as wp
    wp.config.log_level = wp.LOG_WARNING
    torch.set_grad_enabled(False)
    from warpSPH.runner.runner import buildContext
    from warpSPH.runner.caseSpec import CaseSpec
    from warpSPH.cases.compressible.kelvinHelmholtz import kelvinHelmholtzCase as case
    from warpSPH.schemes.meshless import meshlessStep
    from warpSPH.modules.timestep.meshless import computeTimestep

    spec = CaseSpec(caseName=case.name, scheme=args.scheme, params=dict(case.params))
    spec = spec.merged(**case.defaults).merged(nx=args.nx, precision=args.precision, scheme=args.scheme)
    ctx = buildContext(case, spec)
    case.configureScheme(ctx)
    system = case.buildSystem(ctx)
    st = system.state
    if args.state != 'ic':
        d = torch.load(args.state, map_location='cpu', weights_only=False)
        st.positions = d['positions'].to(st.positions).contiguous()
        st.supports = d['supports'].to(st.supports).contiguous()
        st.Q = d['Q'].to(st.Q).contiguous()
    config, sc = ctx.config, ctx.schemeConfig
    dom = system.domain
    print(f'# {args.scheme} KH nx={args.nx} N={st.positions.shape[0]} state={args.state} '
          f'kernel={config.kernel.name} nngb={sc.nngb:.1f} temporal={sc.temporalScheme} '
          f'closureSolver={sc.closureSolver} precision={args.precision}', flush=True)

    rf = torch.profiler.record_function

    def step():
        with rf('bench[timestep]'):
            dt = args.fixedDt if args.fixedDt is not None else computeTimestep(system, config, sc, dt=config.dt)
        with rf('bench[meshlessStep]'):
            update, _, _ = meshlessStep(system, dt, config, sc)
        with rf('bench[update]'):
            pos = st.positions + dt * update.dxdt
            st.positions = torch.where(dom.periodic, dom.min + torch.remainder(pos - dom.min, dom.max - dom.min), pos)
            st.Q = st.Q + dt * update.dQdt
            config.dt = dt
        return dt

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    times, t0 = [], time.perf_counter()
    for k in range(args.steps):
        a = time.perf_counter()
        dt = step()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - a) * 1e3)
    total = time.perf_counter() - t0
    print(f'wall ms/step: mean {statistics.mean(times):.2f}  median {statistics.median(times):.2f}  '
          f'min {min(times):.2f}  max {max(times):.2f}  ({args.steps} steps, {total:.2f} s; last dt {float(dt):.3e})', flush=True)

    if args.profSteps > 0:
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(args.profSteps):
                step()
            torch.cuda.synchronize()
        ev = prof.key_averages()
        n = args.profSteps
        syncNames = ('cudaStreamSynchronize', 'cudaDeviceSynchronize', 'cudaEventSynchronize')
        syncs = sum(e.count for e in ev if e.key in syncNames)
        memcpy = sum(e.count for e in ev if e.key.startswith('cudaMemcpy'))
        launches = sum(e.count for e in ev if e.key in ('cudaLaunchKernel', 'cuLaunchKernel', 'cudaLaunchKernelExC'))
        annot = lambda k: k.startswith('bench[') or k.startswith('warpSPH[') or k.startswith('ProfilerStep') or k.startswith('aten::') or k == 'StateAwareWarpFunction'
        dev = [e for e in ev if e.self_device_time_total > 0 and not annot(e.key)]
        gpu = sum(e.self_device_time_total for e in dev) / 1e3 / n
        print(f'profiler ({n} steps): gpu-busy {gpu:.2f} ms/step  launches {launches / n:.0f}/step  '
              f'syncs {syncs / n:.0f}/step  memcpy {memcpy / n:.0f}/step', flush=True)
        kern = sorted(dev, key=lambda e: -e.self_device_time_total)[:args.topKernels]
        print('top device kernels:')
        for e in kern:
            print(f'  {e.self_device_time_total / 1e3 / n:8.2f} ms/step  x{e.count / n:6.1f}  {e.key[:90]}')

    if args.cprofile:
        import cProfile, pstats, io
        pr = cProfile.Profile()
        pr.enable()
        for _ in range(args.profSteps or 3):
            step()
        torch.cuda.synchronize()
        pr.disable()
        out = io.StringIO()
        pstats.Stats(pr, stream=out).sort_stats(os.environ.get('BENCH_SORT', 'cumulative')).print_stats(args.cprofile)
        print(out.getvalue())

    if args.save:
        torch.save({k: getattr(st, k).detach().cpu() for k in ('positions', 'Q', 'supports', 'lam')}, args.save)
        print(f'saved {args.save}')
    if args.compare:
        ref = torch.load(args.compare, weights_only=False)
        for k in ('positions', 'Q', 'supports', 'lam'):
            a, b = getattr(st, k).detach().cpu(), ref[k]
            same = torch.equal(a, b)
            diff = (a - b).abs().max().item()
            scale = b.abs().max().item()
            print(f'compare {k}: bitwise={same} max|diff|={diff:.3e} (rel {diff / max(scale, 1e-30):.3e})')


if __name__ == '__main__':
    main()
