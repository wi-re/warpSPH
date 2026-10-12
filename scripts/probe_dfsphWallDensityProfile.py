"""Probe (`DFSPH_IMPROVEMENT_PLAN.md`'s remaining open question, 2026-08-26):
the nx=128 follow-up confirmed DFSPH's `--bounded` density band is much
looser than deltaSPH's at matched physical time, and that it is not mostly a
`BoundaryPressureMode` effect (`plain`, which does no mDBC at all, tracks
`mdbcDensity` closely). What's still open is *where* the error lives: is it
concentrated near the wall (a boundary/kernel-truncation effect -- SPH
summation density is well known to under-sample near a boundary unless the
wall particles fully compensate for the missing kernel support) or spread
through the bulk (a general DFSPH trait -- e.g. the divergence-free solver
never converging to its own tolerance every step, per Part 1's finding that
it hits `maxIterations` unconverged on every step even on the unbounded
Kolmogorov case)?

This script drives `randomFlowIncompressibleCase` (DFSPH) and `randomFlowCase`
(deltaSPH) to matched t, then bins each fluid particle's `|density - rho0|`
by its distance from the nearest domain wall (`weaklyCompressible.
domainBoundarySdf`, the same SDF the case itself is built from) to see
whether the error profile is wall-localized or bulk-spread for each scheme.

Usage: `python scripts/probe_dfsphWallDensityProfile.py [--nx 128]
[--tlimit 1.5] [--nbins 8]`
"""

from __future__ import annotations

import argparse

from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')

import torch

from warpSPH.cases.periodic.randomFlow import randomFlowCase
from warpSPH.cases.periodic.randomFlowIncompressible import randomFlowIncompressibleCase
from warpSPH.cases.common.weaklyCompressible import domainBoundarySdf
from warpSPH.runner.runner import buildContext
from warpSPH.runner.caseSpec import CaseSpec


def runToTime(case, nx, tLimit, extraParams, tag, noPenShift=None):
    spec = CaseSpec(caseName=case.name, scheme=case.scheme, params=dict(case.params))
    spec = spec.merged(**case.defaults)
    spec = spec.merged(nx=nx, tLimit=tLimit, store=False, plot=False, quiet=True,
                       progress=False, params=dict({'bounded': True}, **extraParams))

    ctx = buildContext(case, spec)
    case.configureScheme(ctx)
    if noPenShift is not None:
        ctx.schemeConfig.solverConfig.mdbcNoPenetrationShift = noPenShift
    system = case.buildSystem(ctx)
    case.initialConditions(ctx, system)
    runningState = system.initializeNewState()

    dt = ctx.config.dt
    timeLimited = case.timestep is not None
    nSteps = spec.nSteps if spec.nSteps is not None else int(spec.tLimit / dt)
    steps = range(10**9) if timeLimited else range(nSteps)

    t = 0.0
    for i in steps:
        stepResult = ctx.integrator.function(
            state=runningState, f=ctx.stepFunction, dt=ctx.config.dt,
            config=ctx.config, verbose=False, schemeConfig=ctx.schemeConfig)
        runningState = stepResult.state
        if case.timestep is not None:
            ctx.config.dt = case.timestep(ctx, runningState)
        t = float(runningState.t)
        v = runningState.state.velocities
        if torch.any(torch.isnan(v)):
            print(f'[{tag}] DIVERGED at step {i+1}, t={t:.4f}')
            break
        if timeLimited and t >= spec.tLimit:
            break
        if not timeLimited and i == nSteps - 1:
            break

    print(f'[{tag}] finished: t={t:.4f}, steps={i+1}')
    return ctx, runningState


def wallDistanceProfile(ctx, runningState, nbins, tag):
    s = runningState.state
    fluidMask = s.kinds == 0 if hasattr(s, 'kinds') else torch.ones(
        s.positions.shape[0], dtype=torch.bool, device=s.positions.device)
    positions = s.positions[fluidMask].detach()
    positions.requires_grad = True
    sdf = domainBoundarySdf(ctx)
    d, _ = sdf(positions)
    d = d.detach()
    # Fluid particles sit inside the interior domain; empirically (checked by
    # this script's own printed range, not assumed) `domainBoundarySdf`
    # (invert=False, "walls: everything outside interior") reads *positive*
    # there, growing with distance from the nearest wall -- so `d` itself is
    # already "depth into the fluid", no sign flip needed.
    depth = d
    rho = s.densities[fluidMask].detach()
    err = (rho - 1.0).abs()
    crossedMask = depth < 0
    nCrossed = int(crossedMask.sum().item())
    crossedErrMean = err[crossedMask].mean().item() if nCrossed > 0 else float('nan')
    print(f'[{tag}] wall-depth range over fluid particles: '
         f'[{depth.min().item():.4f}, {depth.max().item():.4f}] '
         f'nCrossed(depth<0)={nCrossed}/{depth.numel()} '
         f'crossedMeanErr={crossedErrMean:.4e} '
         f'rho=[{rho.min().item():.4f},{rho.max().item():.4f}]')

    dx = ctx.config.dx
    minDepth, maxDepth = depth.min().item(), depth.max().item()
    edges = torch.linspace(minDepth, maxDepth, nbins + 1)
    print(f'[{tag}] density error |rho-1| by distance from wall (dx={dx:.5f}):')
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (depth >= lo) & (depth < hi)
        n = int(mask.sum().item())
        if n == 0:
            print(f'    [{lo.item():6.3f},{hi.item():6.3f}) n=0')
            continue
        binErr = err[mask]
        print(f'    [{lo.item():6.3f},{hi.item():6.3f}) n={n:5d} '
             f'meanErr={binErr.mean().item():.4e} maxErr={binErr.max().item():.4e} '
             f'({(lo.item()/dx):5.1f}-{(hi.item()/dx):5.1f} dx from wall)')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--nx', type=int, default=128)
    p.add_argument('--tlimit', type=float, default=1.5)
    p.add_argument('--nbins', type=int, default=8)
    p.add_argument('--mode', type=str, default='mdbcDensity')
    p.add_argument('--no-pen-shift', dest='noPenShift', type=str, default='both',
                   choices=['both', 'on', 'off'],
                   help='Run computeMdbcNoPenShift on, off, or both (default) for the DFSPH case')
    p.add_argument('--skip-deltasph', action='store_true')
    args = p.parse_args()

    variants = {'both': [True, False], 'on': [True], 'off': [False]}[args.noPenShift]
    for shiftOn in variants:
        tag = f'dfsph/noPenShift={shiftOn}'
        ctxD, stateD = runToTime(randomFlowIncompressibleCase, args.nx, args.tlimit,
                                 {'boundaryPressureMode': args.mode}, tag=tag,
                                 noPenShift=shiftOn)
        wallDistanceProfile(ctxD, stateD, args.nbins, tag=tag)

    if not args.skip_deltasph:
        ctxW, stateW = runToTime(randomFlowCase, args.nx, args.tlimit, {}, tag='deltaSPH')
        wallDistanceProfile(ctxW, stateW, args.nbins, tag='deltaSPH')
