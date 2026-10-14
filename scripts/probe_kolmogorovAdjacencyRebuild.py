"""Ablation (JFNK_PLAN.md's neighbor-list-staleness follow-up to E1.6/E1.7,
2026-08-26): does rebuilding/re-validating the SPH adjacency *inside* the
implicit solve -- once per RHS evaluation, mirroring `deltaSPH_step`'s own
`buildVerletList(..., priorNeighborhood=adjacency, ...)` call
(`warpSPH/src/warpSPH/schemes/deltaSPH.py:69-74`) -- push JFNK's safe
`dt`-multiplier ceiling higher than E1.6's `mult<=5` at fully-developed
Kolmogorov turbulence (`nx=128`, `xi=1.0`, `vMax~2.2-2.9`)? Or is that
ceiling purely a Newton/GMRES numerical-accuracy limit, unaffected by
whether the neighbor list is frozen for the whole implicit step (the
default, `f_acoustic_core`) or refreshed every RHS call
(`f_acoustic_core_rebuildAdjacency`, new sibling function,
`warpSPH/src/warpSPH/schemes/acousticCore.py`)?

Structured as a direct extension of `probe_kolmogorovContinuation.py`
(same snapshot format, same `runReference`/`runJFNK` shape, same
forced/decaying branches) rather than a rewrite -- the only new axis is
`f`: baseline (`f_acoustic_core`) vs. rebuild (`f_acoustic_core_
rebuildAdjacency`), run back-to-back on the *same* loaded snapshot so the
comparison is apples-to-apples. Also counts extra `buildVerletList` calls
the rebuild variant makes (via a thin counting wrapper monkey-patched onto
`warpSPH.schemes.experimental.acousticCore.buildVerletList` for the duration of a rebuild
run only) and reports wall-clock overhead, per JFNK_PLAN.md's own honesty
convention about cost, not just stability.

Usage: `python scripts/probe_kolmogorovAdjacencyRebuild.py --snapshot PATH
[--multipliers 1,3,5,10] [--branch forced|decaying|both] [--matvec fd|jvp]
[--duration-steps 5]`
"""

from __future__ import annotations

import argparse
import math
import time

from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')

import torch

import warpSPH.schemes.experimental.acousticCore as acousticCoreScheme
from warpSPH.configurations import AcousticCoreConfig, buildConfig
from warpSPH.schemes.experimental.acousticCore import f_acoustic_core, f_acoustic_core_rebuildAdjacency
from warpSPH.systems.acousticCore import AcousticCoreState, AcousticCoreSystem
from warpSPH.utils import buildDomainDescription
from warpSPHIntegrators import JFNKSolver, get_reference_state, getIntegrator
from warpSPHCore import SupportScheme, buildVerletList


def loadSnapshot(path: str, device=None):
    d = torch.load(path, map_location='cpu', weights_only=False)
    device = device if device is not None else (torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'))
    dtype = d['positions'].dtype

    domain = buildDomainDescription(l=d['L'], dim=2, periodic=True, device=device, dtype=dtype)
    config, _integrator = buildConfig(dim=2, nx=d['nx'], domain=domain, device=device,
                                      dtype=dtype, dx=d['L'] / d['nx'], cflFactor=d['cflFactor'])

    state = AcousticCoreState(
        positions=d['positions'].to(device), velocities=d['velocities'].to(device),
        densities=d['densities'].to(device), supports=d['supports'].to(device),
        masses=d['masses'].to(device), kinds=d['kinds'].to(device),
        materials=d['materials'].to(device), UIDs=d['UIDs'].to(device), UIDcounter=d['UIDcounter'],
    )
    adjacency = buildVerletList(state, config.domain, 1.0, SupportScheme.SuperSymmetric, None)
    system = AcousticCoreSystem(state=state, adjacency=adjacency, domain=config.domain, t=d['t'])
    return system, config, d


def buildSchemeConfig(snap, forcingAmplitude):
    return AcousticCoreConfig(
        restDensity=1.0, soundSpeed=snap['soundSpeed'],
        forcingAmplitude=forcingAmplitude, forcingWavenumber=snap['k'],
        velocityDiffusionCoefficient=snap['nu'],
    )


def diagnostics(state):
    m, v, rho = state.masses, state.velocities, state.densities
    return dict(
        KE=(0.5 * m * (v ** 2).sum(-1)).sum().item(),
        vMax=v.norm(dim=1).max().item(),
        rhoMin=rho.min().item(), rhoMax=rho.max().item(), rhoStd=rho.std().item(),
    )


class _CallCounter:
    """Counts calls to `buildVerletList` made *inside* `f_acoustic_core_
    rebuildAdjacency` for one run, via a thin wrapper monkey-patched onto
    `warpSPH.schemes.experimental.acousticCore.buildVerletList` for that run's duration
    only -- doesn't touch the real `buildVerletList` or any production path,
    and is restored unconditionally (`finally`) even on divergence/exception.
    """
    def __init__(self):
        self.count = 0

    def __call__(self, *args, **kwargs):
        self.count += 1
        return buildVerletList(*args, **kwargs)


def runReference(system, config, schemeConfig, dtSmall, nSteps, printEvery):
    scheme = getIntegrator('RK4')
    state = system
    trace = []
    t0 = time.time()
    for i in range(1, nSteps + 1):
        result = scheme(state=state, f=f_acoustic_core, dt=dtSmall, config=config, schemeConfig=schemeConfig, verbose=False)
        state = result.state
        s = get_reference_state(state)
        d = diagnostics(s)
        d['t'] = i * dtSmall
        trace.append(d)
        if not math.isfinite(d['KE']) or d['rhoMax'] > 100.0 * schemeConfig.restDensity:
            print(f'[reference] DIVERGED at step {i} (t={d["t"]:.4f}s)')
            return trace
        state.adjacency = buildVerletList(s, config.domain, 1.0, SupportScheme.SuperSymmetric, None)
        if i % printEvery == 0 or i == nSteps:
            print(f'[reference RK4] step {i:5d}/{nSteps} t={d["t"]:.4f}s KE={d["KE"]:.4f} '
                  f'vMax={d["vMax"]:.4f} rhoStd={d["rhoStd"]:.4e} wall={time.time()-t0:.1f}s')
    return trace


def runJFNK(system, config, schemeConfig, dtLarge, nSteps, printEvery, matvec='fd', useRebuild=False):
    scheme = getIntegrator('Backward Euler (implicit)')
    solver = JFNKSolver(matvec=matvec, tol=1e-6, max_iterations=25, gmres_maxiter=80, gmres_restart=40)
    f = f_acoustic_core_rebuildAdjacency if useRebuild else f_acoustic_core
    state = system
    trace = []
    counter = _CallCounter()
    original = acousticCoreScheme.buildVerletList
    if useRebuild:
        acousticCoreScheme.buildVerletList = counter
    t0 = time.time()
    try:
        for i in range(1, nSteps + 1):
            result = scheme(state, dtLarge, f, config, schemeConfig, solver=solver)
            state = result.state
            s = get_reference_state(state)
            d = diagnostics(s)
            d['t'] = i * dtLarge
            trace.append(d)
            if not math.isfinite(d['KE']) or d['rhoMax'] > 100.0 * schemeConfig.restDensity:
                print(f'[JFNK {"rebuild" if useRebuild else "baseline"} matvec={matvec}] '
                      f'DIVERGED at step {i} (t={d["t"]:.4f}s)')
                break
            state.adjacency = buildVerletList(s, config.domain, 1.0, SupportScheme.SuperSymmetric, None)
            if i % printEvery == 0 or i == nSteps:
                elapsed = time.time() - t0
                print(f'[JFNK {"rebuild" if useRebuild else "baseline"} matvec={matvec} dt={dtLarge:.4f}] '
                      f'step {i:5d}/{nSteps} t={d["t"]:.4f}s KE={d["KE"]:.4f} '
                      f'vMax={d["vMax"]:.4f} rhoStd={d["rhoStd"]:.4e} wall={elapsed:.1f}s')
    finally:
        acousticCoreScheme.buildVerletList = original
    wall = time.time() - t0
    return trace, counter.count, wall


def runBranch(name, snapshotPath, snap, forcingAmplitude, dtSmall, dtLarge, nSteps, printEvery,
              matvec='fd', skipReference=False):
    print(f'\n=== branch: {name} (forcingAmplitude={forcingAmplitude}) dtLarge={dtLarge:.5f} '
          f'({dtLarge/dtSmall:.0f}x dtSmall) matvec={matvec} ===')
    refTrace = None
    if not skipReference:
        system, config, _ = loadSnapshot(snapshotPath)
        schemeConfig = buildSchemeConfig(snap, forcingAmplitude)
        refNSteps = int(nSteps * (dtLarge / dtSmall))
        refTrace = runReference(system, config, schemeConfig, dtSmall, refNSteps, max(1, printEvery * 5))

    system2, config2, _ = loadSnapshot(snapshotPath)
    schemeConfig2 = buildSchemeConfig(snap, forcingAmplitude)
    baseTrace, baseCalls, baseWall = runJFNK(system2, config2, schemeConfig2, dtLarge, nSteps, printEvery,
                                              matvec=matvec, useRebuild=False)

    system3, config3, _ = loadSnapshot(snapshotPath)
    schemeConfig3 = buildSchemeConfig(snap, forcingAmplitude)
    rebuildTrace, rebuildCalls, rebuildWall = runJFNK(system3, config3, schemeConfig3, dtLarge, nSteps, printEvery,
                                                       matvec=matvec, useRebuild=True)

    print(f'  cost: baseline wall={baseWall:.2f}s (0 inner buildVerletList calls, adjacency frozen); '
          f'rebuild wall={rebuildWall:.2f}s ({rebuildCalls} inner buildVerletList calls, '
          f'{rebuildCalls/max(1,len(rebuildTrace)):.1f}/real-step), '
          f'overhead={rebuildWall/max(baseWall,1e-9):.2f}x wall time')

    return dict(reference=refTrace, baseline=baseTrace, rebuild=rebuildTrace,
                baseCalls=baseCalls, rebuildCalls=rebuildCalls, baseWall=baseWall, rebuildWall=rebuildWall)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--snapshot', type=str, required=True)
    p.add_argument('--multipliers', type=str, default='1,3,5,10')
    p.add_argument('--duration-steps', type=int, default=5, help='number of JFNK real steps per multiplier')
    p.add_argument('--branch', type=str, default='both', choices=['forced', 'decaying', 'both'])
    p.add_argument('--print-every', type=int, default=1)
    p.add_argument('--matvec', type=str, default='fd', choices=['fd', 'jvp'])
    p.add_argument('--skip-reference', action='store_true')
    p.add_argument('--out', type=str, default=None)
    args = p.parse_args()

    snap = torch.load(args.snapshot, map_location='cpu', weights_only=False)
    dtSmall = snap['dt']
    multipliers = [float(x) for x in args.multipliers.split(',')]
    print(f'snapshot: nx={snap["nx"]} t={snap["t"]:.4f}s step={snap["step"]} dtSmall={dtSmall:.5f} '
          f'xi={snap["xi"]} nu={snap["nu"]:.6f} multipliers={multipliers}')

    allResults = {}
    for mult in multipliers:
        dtLarge = dtSmall * mult
        allResults[mult] = {}
        if args.branch in ('forced', 'both'):
            allResults[mult]['forced'] = runBranch(
                'forced', args.snapshot, snap, snap['xi'], dtSmall, dtLarge, args.duration_steps,
                args.print_every, matvec=args.matvec, skipReference=args.skip_reference)
        if args.branch in ('decaying', 'both'):
            allResults[mult]['decaying'] = runBranch(
                'decaying', args.snapshot, snap, 0.0, dtSmall, dtLarge, args.duration_steps,
                args.print_every, matvec=args.matvec, skipReference=args.skip_reference)

    if args.out:
        torch.save(allResults, args.out)
        print(f'saved to {args.out}')
