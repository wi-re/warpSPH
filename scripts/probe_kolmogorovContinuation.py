"""Probe, part 2 (per conversation, 2026-08-25): load a fully-developed
Kolmogorov-turbulence snapshot from `probe_kolmogorovSpinup.py` and compare
`JFNKSolver` at a large (`Nx` the acoustic CFL) `dt` against a small-`dt`
`RK4` reference, in two branches:

- **forced**: `forcingAmplitude` stays at the spin-up's `xi` -- the usual
  sustained Kolmogorov case. Reference expectation: both trajectories should
  stay in the same statistically-steady turbulent band (KE/vMax fluctuating
  around the spin-up's own settled range), not decay or blow up.
- **decaying**: `forcingAmplitude=0` -- forcing cut, free decay of 2D
  turbulence. Reference expectation: KE decays monotonically (2D turbulent
  decay, no equilibrium to return to); the question is whether JFNK's
  large-`dt` trajectory tracks the small-`dt` reference's decay curve, and
  whether the rudimentary core (no shifting/no surface treatment) develops
  visible pairing/clustering artifacts (`rhoStd` growing pathologically)
  once forcing is no longer replenishing the energy the dissipation and
  numerical noise remove.

Usage: `python scripts/probe_kolmogorovContinuation.py --snapshot PATH
[--duration 1.0] [--dt-multiplier 10] [--branch forced|decaying|both]`
"""

from __future__ import annotations

import argparse
import math
import time

from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')

import torch

from warpSPH.configurations import AcousticCoreConfig, buildConfig
from warpSPH.schemes.experimental.acousticCore import f_acoustic_core
from warpSPH.systems.acousticCore import AcousticCoreState, AcousticCoreSystem
from warpSPH.utils import buildDomainDescription
from warpSPHIntegrators.solvers import JFNKSolver
from warpSPHIntegrators.state import get_reference_state
from warpSPHIntegrators import get_scheme
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


def runReference(system, config, schemeConfig, dtSmall, duration, printEvery):
    scheme = get_scheme('RK4')
    state = system
    nSteps = int(duration / dtSmall)
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


def runJFNK(system, config, schemeConfig, dtLarge, duration, printEvery, matvec='fd'):
    scheme = get_scheme('Backward Euler (implicit)')
    solver = JFNKSolver(matvec=matvec, tol=1e-6, max_iterations=25, gmres_maxiter=80, gmres_restart=40)
    state = system
    nSteps = max(1, int(duration / dtLarge))
    trace = []
    t0 = time.time()
    for i in range(1, nSteps + 1):
        result = scheme(state, dtLarge, f_acoustic_core, config, schemeConfig, solver=solver)
        state = result.state
        s = get_reference_state(state)
        d = diagnostics(s)
        d['t'] = i * dtLarge
        trace.append(d)
        if not math.isfinite(d['KE']) or d['rhoMax'] > 100.0 * schemeConfig.restDensity:
            print(f'[JFNK] DIVERGED at step {i} (t={d["t"]:.4f}s)')
            return trace
        state.adjacency = buildVerletList(s, config.domain, 1.0, SupportScheme.SuperSymmetric, None)
        if i % printEvery == 0 or i == nSteps:
            print(f'[JFNK dt={dtLarge:.4f}] step {i:5d}/{nSteps} t={d["t"]:.4f}s KE={d["KE"]:.4f} '
                  f'vMax={d["vMax"]:.4f} rhoStd={d["rhoStd"]:.4e} wall={time.time()-t0:.1f}s')
    return trace


def runBranch(name, snapshotPath, snap, forcingAmplitude, dtSmall, dtLarge, duration, printEvery, matvec='fd', skipReference=False):
    print(f'\n=== branch: {name} (forcingAmplitude={forcingAmplitude}) ===')
    refTrace = None
    if not skipReference:
        system, config, _ = loadSnapshot(snapshotPath)
        schemeConfig = buildSchemeConfig(snap, forcingAmplitude)
        refTrace = runReference(system, config, schemeConfig, dtSmall, duration, printEvery)

    system2, config2, _ = loadSnapshot(snapshotPath)
    schemeConfig2 = buildSchemeConfig(snap, forcingAmplitude)
    jfnkTrace = runJFNK(system2, config2, schemeConfig2, dtLarge, duration, max(1, printEvery // 10), matvec=matvec)

    return refTrace, jfnkTrace


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--snapshot', type=str, required=True)
    p.add_argument('--duration', type=float, default=1.0)
    p.add_argument('--dt-multiplier', type=float, default=10.0)
    p.add_argument('--branch', type=str, default='both', choices=['forced', 'decaying', 'both'])
    p.add_argument('--print-every', type=int, default=50)
    p.add_argument('--matvec', type=str, default='fd', choices=['fd', 'jvp'])
    p.add_argument('--skip-reference', action='store_true')
    args = p.parse_args()

    snap = torch.load(args.snapshot, map_location='cpu', weights_only=False)
    dtSmall = snap['dt']
    dtLarge = dtSmall * args.dt_multiplier
    print(f'snapshot: nx={snap["nx"]} t={snap["t"]:.4f}s step={snap["step"]} dtSmall={dtSmall:.5f} '
          f'dtLarge={dtLarge:.5f} (x{args.dt_multiplier}) xi={snap["xi"]} nu={snap["nu"]:.6f}')

    results = {}
    if args.branch in ('forced', 'both'):
        results['forced'] = runBranch('forced', args.snapshot, snap, snap['xi'], dtSmall, dtLarge, args.duration,
                                      args.print_every, matvec=args.matvec, skipReference=args.skip_reference)
    if args.branch in ('decaying', 'both'):
        results['decaying'] = runBranch('decaying', args.snapshot, snap, 0.0, dtSmall, dtLarge, args.duration,
                                        args.print_every, matvec=args.matvec, skipReference=args.skip_reference)

    torch.save(results, args.snapshot.replace('.pt', '_continuation.pt'))
