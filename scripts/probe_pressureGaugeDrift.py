"""Probe (`DFSPH_IMPROVEMENT_PLAN.md` Part 4, 2026-08-26): track the pure-Neumann
pressure gauge (the additive constant a periodic PPE leaves undetermined) on
`solveDivergenceFree`, the VD (divergence-free) pressure solver, across a long
`kolmogorovIncompressible` run (periodic, no solid boundary, no free surface --
the isolated case this investigation deliberately narrowed to).

Non-invasively wraps `solveDivergenceFree` (Python-level monkeypatch, no
source edits) to log `sourceTerm`/pressure mean, std, min/max and iteration
count every step, for whichever `--solver` variant is selected.

Finding so far: this solver's per-iteration `pressureB - pressureB.mean()`
recentering (both the fixed-omega and `optimal` relaxation-mode loops in
`modules/incompressible/divergenceFree.py`) keeps `pMean` at the float32 noise
floor (~1e-8) for the whole run, even as `pStd` grows ~4 orders of magnitude --
i.e. this solver's gauge fix works. The *other* DFSPH pressure solver,
`solveIncompressible` (the DI/PS solve), has no equivalent fix and does drift
catastrophically -- see `probe_incompressibleGaugeDrift.py` for that half of
the investigation, which is where the actual open problem lives.

Usage: `python scripts/probe_pressureGaugeDrift.py [--nx 64] [--nsteps 300]
[--solver relaxedJacobi|optimal|bicgStab|gmres|cg|minres]`

The Krylov solver types are a spot-check, not a full investigation: this
script only observes state *after* each full solve, so it can't see whether
`solvePressureKrylov`'s gauge fix -- applied once, after all internal Krylov
iterations, not per-iteration like the two relaxed-Jacobi loops -- lets the
pressure iterate drift further *during* a solve than what's visible here.
That's the natural next thing to check if the Krylov path is in scope.
"""
from __future__ import annotations

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--nx', type=int, default=64)
parser.add_argument('--nsteps', type=int, default=300)
parser.add_argument('--solver', type=str, default='relaxedJacobi',
                     choices=['relaxedJacobi', 'optimal', 'bicgStab', 'gmres', 'cg', 'minres'])
args = parser.parse_args()

from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')

from warpSPH.cases import kolmogorovIncompressible as mod
from warpSPH.runner.cli import caseMain
from warpSPH.configurations import PressureSolverType, JacobiRelaxationMode

log = []

import warpSPH.modules.incompressible.divergenceFree as dfmod
import warpSPH.schemes.divergenceFree as dfsphmod
from warpSPH.modules.incompressible.divergenceFree import computeMomentumIncompressible

_orig_solveDivergenceFree = dfmod.solveDivergenceFree


def _wrapped_solveDivergenceFree(particles, config, schemeConfig, adjacency, dvdt, dt, verbose=False):
    predictedVelocities = particles.velocities + dt * dvdt
    divergence = computeMomentumIncompressible(
        currentState=particles, config=config, schemeConfig=schemeConfig,
        adjacency=adjacency, advectionVelocities=predictedVelocities)
    sourceTerm = -divergence
    stMean = sourceTerm.mean().item()
    stAbsMean = sourceTerm.abs().mean().item()

    a_p, pressure, errors, pressures = _orig_solveDivergenceFree(
        particles, config, schemeConfig, adjacency, dvdt, dt, verbose=verbose)

    log.append(dict(
        stMean=stMean, stAbsMean=stAbsMean,
        pMean=pressure.mean().item(), pStd=pressure.std().item(),
        pMin=pressure.min().item(), pMax=pressure.max().item(),
        nIter=len(errors), finalErr=errors[-1] if errors else float('nan'),
    ))
    return a_p, pressure, errors, pressures


dfmod.solveDivergenceFree = _wrapped_solveDivergenceFree
dfsphmod.solveDivergenceFree = _wrapped_solveDivergenceFree

solverTypeMap = {
    'relaxedJacobi': PressureSolverType.relaxedJacobi,
    'optimal': PressureSolverType.relaxedJacobi,
    'bicgStab': PressureSolverType.bicgStab,
    'gmres': PressureSolverType.gmres,
    'cg': PressureSolverType.cg,
    'minres': PressureSolverType.minres,
}

_orig_configureScheme = mod.configureScheme


def _wrapped_configureScheme(ctx):
    _orig_configureScheme(ctx)
    ctx.schemeConfig.solverConfig.divergenceFreeSolver.solverType = solverTypeMap[args.solver]
    if args.solver == 'optimal':
        ctx.schemeConfig.solverConfig.divergenceFreeSolver.relaxationMode = JacobiRelaxationMode.optimal


mod.configureScheme = _wrapped_configureScheme
mod.kolmogorovIncompressibleCase.configureScheme = _wrapped_configureScheme

result = caseMain(mod.kolmogorovIncompressibleCase, argv=[
    '--nx', str(args.nx), '--tLimit', '1000.0', '--nSteps', str(args.nsteps),
    '--quiet', '--no-store', '--no-plot',
])

n = len(log)
print(f"\n=== {args.solver}, nx={args.nx}, {n} steps recorded ===")
print(f"{'step':>5} {'stMean':>12} {'stAbsMean':>12} {'pMean':>12} {'pStd':>12} {'pMax':>12} {'nIter':>6}")
show = sorted(set([0, 1, 2, 5, 10, 20] + list(range(0, n, max(1, n // 20))) + [n - 1]))
for i in show:
    if i >= n:
        continue
    e = log[i]
    print(f"{i:5d} {e['stMean']:12.4e} {e['stAbsMean']:12.4e} {e['pMean']:12.4e} {e['pStd']:12.4e} {e['pMax']:12.4e} {e['nIter']:6d}")

if n:
    pMeans = [e['pMean'] for e in log]
    pStds = [e['pStd'] for e in log]
    print(f"\npMean: first={pMeans[0]:.4e} last={pMeans[-1]:.4e} max|pMean|={max(abs(x) for x in pMeans):.4e}")
    print(f"pStd: first={pStds[0]:.4e} last={pStds[-1]:.4e} max={max(pStds):.4e}")
    stMeans = [e['stMean'] for e in log]
    stAbsMeans = [e['stAbsMean'] for e in log]
    ratios = sorted(abs(m) / (a + 1e-30) for m, a in zip(stMeans, stAbsMeans))
    print(f"sourceTerm mean/absMean ratio (bias fraction), median: {ratios[len(ratios) // 2]:.4e}")
