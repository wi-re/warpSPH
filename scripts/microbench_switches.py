"""Micro-benchmark every viscosity switch of `modules/shockCapturing`: its terms function, its update function and (for the
step-boundary switches) its step hook, in isolation -- torch ops, warp launches, host syncs and host / wall / GPU time per call
(`warpSPH/utils/microBench.py`, the `module-microbench` skill).

    python scripts/microbench_switches.py                          # sod2d CompSPH, every switch
    python scripts/microbench_switches.py --switches CullenHopkins Sphenix2022 --nx 200
    python scripts/microbench_switches.py --case sod --nx 1000     # another registered compressible case

Per switch the state is produced by a short real run of the case with that switch selected (so the fields the
switch persists -- entropy history, Wadsley's D, ... -- exist and are realistic); the micro benchmark then calls the
three functions on the final state of that run, on its Verlet list, with the step's dt. The setup run is a state
generator, not a validation run: no video, no per-step plots. One switch at a time on the GPU.
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--case', default='sod2d')
ap.add_argument('--scheme', default='CompSPH')
ap.add_argument('--nx', type=int, nargs='+', default=[None], help='one or more resolutions; the counts are taken at the first')
ap.add_argument('--steps', type=int, default=20, help='steps of the setup run (default 20)')
ap.add_argument('--switches', nargs='*', default=None, help='ViscositySwitch names (default: every registered one)')
ap.add_argument('--calls', type=int, default=10, help='calls in the counting pass')
ap.add_argument('--timed', type=int, default=50, help='calls in the timing pass')
ap.add_argument('--stages', type=int, default=2, help='RK stages per step for the per-step roll-up (default 2)')
ap.add_argument('--out', default=None)
args = ap.parse_args()

from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')

import torch
from warpSPHCore import SupportScheme
from warpSPH.cases import importAll
from warpSPH.modules.compSPH.operators import computeCompSPHAccel
from warpSPH.modules.shockCapturing import computeViscositySwitchTerms, updateViscositySwitch
from warpSPH.modules.shockCapturing.wrapper import SWITCHES, STEP_HOOKS
from warpSPH.runner import run
from warpSPH.runner.case import getCase
from warpSPH.utils.microBench import microBench, formatTable, formatSyncSites

importAll()
names = args.switches or [s.name for s in SWITCHES]
case = getCase(args.case)
report, syncReport, trees = [], [], []
t0 = time.perf_counter()


def benchSwitch(name, nx):
    spec = {} if nx is None else {'nx': nx}
    r = run(case, quiet=True, progress=False, nSteps=args.steps, scheme=args.scheme, plot=False, video=False,
            params={'viscositySwitch': name}, **spec)
    system, ctx = r.state, r.ctx
    state, config, schemeConfig, adjacency = system.state, ctx.config, ctx.schemeConfig, system.adjacency
    dt = config.dt if isinstance(config.dt, torch.Tensor) else torch.tensor(float(config.dt), device=state.positions.device)
    support = SupportScheme.SuperSymmetric
    n = state.positions.shape[0]
    # (the inputs of a real stage: the terms' switch state, and the acceleration the update relaxes against)
    alphas, switchState = computeViscositySwitchTerms(dt, state, config, schemeConfig, support, adjacency)
    dvdt = computeCompSPHAccel(state, config, schemeConfig, adjacency)[0]
    scheme = schemeConfig.viscositySwitchParams.scheme
    rows = [
        microBench(lambda: computeViscositySwitchTerms(dt, state, config, schemeConfig, support, adjacency),
                   f'{name} terms', calls=args.calls, timedCalls=args.timed),
        microBench(lambda: updateViscositySwitch(switchState, dt, dvdt, state, config, schemeConfig, support, adjacency),
                   f'{name} update', calls=args.calls, timedCalls=args.timed),
    ]
    hook = STEP_HOOKS.get(scheme)
    if hook is not None:
        clock = [float(system.t)]
        step = float(dt)

        def advance():
            clock[0] += step                # the hooks return at once unless time moved on since their last call
            hook(state, state, clock[0], schemeConfig, config)
        rows.append(microBench(advance, f'{name} step hook', calls=args.calls, timedCalls=args.timed))
    return n, rows


def perStep(results, stages):
    """One row per switch: terms and update x stages + the step hook (if any)."""
    from collections import OrderedDict
    from warpSPH.utils.microBench import MicroResult
    by = OrderedDict()
    for r in results:
        name, _, kind = r.label.rpartition(' ')
        if kind == 'hook':
            name = name.rpartition(' ')[0]
        by.setdefault(name, []).append((r, 1 if r.label.endswith('step hook') else stages))
    rows = []
    for name, parts in by.items():
        row = MicroResult(label=name)
        for f in ('ops', 'launches', 'syncs', 'hostUs', 'wallUs', 'gpuUs', 'deviceKernels'):
            setattr(row, f, sum(getattr(r, f) * m for r, m in parts))
        rows.append(row)
    return rows


for k, nx in enumerate(args.nx):
    results = []
    nParticles = None
    for name in names:
        print(f'[microbench] {name} (nx={nx}) ...', flush=True)
        try:
            nParticles, rows = benchSwitch(name, nx)
        except Exception as e:                                          # noqa: BLE001 -- one broken switch must not hide the others
            print(f'[microbench] {name} FAILED: {type(e).__name__}: {str(e).splitlines()[0][:160]}', flush=True)
            continue
        results += rows
        torch.cuda.synchronize()
    report.append(f'=== {args.case} / {args.scheme}, nx={nx}, N={nParticles} particles, per call ===\n' + formatTable(results))
    report.append(f'--- per step ({args.stages} stages: terms and update every stage, the step hook once) ---\n' + formatTable(perStep(results, args.stages)))
    if k == 0:
        syncReport.append(formatSyncSites(results))
        trees = [(r.label, r.tree) for r in results]

text = '\n\n'.join(report) + '\n\nsync call sites (counting pass):\n' + '\n'.join(syncReport)
print('\n' + text)
out = args.out or os.path.join(HERE, 'out_microbench', f"switches_{args.case}_{time.strftime('%Y%m%d_%H%M%S')}.txt")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, 'w') as f:
    f.write(text + '\n\nsection trees:\n')
    for label, tree in trees:
        f.write(f'\n--- {label}\n{tree}\n')
print(f'[microbench] {time.perf_counter() - t0:.0f} s; written to {out}')
