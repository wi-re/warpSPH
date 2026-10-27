"""Probe (per conversation, 2026-08-25): does the rudimentary acoustic core
(`JFNK_PLAN.md` Phase E1.5) develop genuine 2D Kolmogorov turbulence at
*realistic* operating parameters -- `cases/periodic/kolmogorov.py`'s own `xi=1.0`
(forcingAmplitude), `k=4` (forcingWavenumber), `alpha=0.01`
(artificial-viscosity floor) -- rather than the small, toy-scale sweep
`test_acousticCoreDissipation.py` used? That sweep found a viscous stability
threshold (`nu>=0.03` avoids divergence) at a saturated velocity of only
`vMax~0.3-0.5` (`Ma~0.03-0.05`), well below the "normal" `v~1`/`Ma=0.1`
regime real WCSPH runs target -- this script builds the probe needed to
check that regime directly instead of extrapolating.

Method (spin-up + snapshot, as instructed): run from rest with a **small,
acoustic-CFL-respecting `dt`** (not the `20x` stress-test `dt` the earlier
sweep used -- `RK4` cannot survive that regardless of the physics, Phase
B/E1 already established this) until the flow is fully turbulent, snapshot
periodically, then hand a chosen snapshot to
`probe_kolmogorovContinuation.py` for the actual JFNK-vs-explicit,
forced-vs-decaying comparison.

Usage: `python scripts/probe_kolmogorovSpinup.py [--nx 48] [--tlimit 4.0] [--out DIR]`
"""

from __future__ import annotations

import argparse
import math
import os
import time

from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')

import torch

from warpSPH.sample.acousticCore import buildPeriodicVortexAcousticCoreSystem
from warpSPH.schemes.experimental.acousticCore import f_acoustic_core
from warpSPH.modules.deltaSPH import alphaToNu
from warpSPHIntegrators.state import get_reference_state
from warpSPHIntegrators import get_scheme
from warpSPHCore import SupportScheme, buildVerletList

# cases/periodic/kolmogorov.py's own defaults (xi, k, alpha) -- WEAKLY_COMPRESSIBLE_PARAMS/
# kolmogorovCase.params -- reused verbatim by default so this probe targets
# the real operating point, not an arbitrary one. `xi`/`dtFactor` are
# CLI-overridable (see `--xi`/`--dt-factor`) for the Ma-scaling follow-up
# (JFNK_PLAN.md Phase E1.7): xi=0.5 targets a saturated velocity closer to
# `Ma=0.1`, and `dtFactor<1` re-runs xi=1.0 at a smaller `dt` as a
# resolution/timestep-convergence check on the xi=1.0 saturated velocity
# itself.
K = 4.0
ALPHA = 0.01
L = 2.0
SOUND_SPEED = 10.0  # Ma~0.1 at the expected saturated velocity scale ~xi
CFL_FACTOR = 0.3    # matches WEAKLY_COMPRESSIBLE_DEFAULTS


def buildSpinupSystem(nx: int, xi: float):
    system, config, schemeConfig = buildPeriodicVortexAcousticCoreSystem(
        nx=nx, dim=2, L=L, uMag=0.0, rho0=1.0, soundSpeed=SOUND_SPEED,
        cflFactor=CFL_FACTOR, forcingAmplitude=xi, forcingWavenumber=K)
    h = system.state.supports.mean().item()
    nu = alphaToNu(ALPHA, SOUND_SPEED, h, 2)
    schemeConfig.velocityDiffusionCoefficient = nu
    return system, config, schemeConfig, h, nu


def diagnostics(state):
    m, v, rho = state.masses, state.velocities, state.densities
    return dict(
        KE=(0.5 * m * (v ** 2).sum(-1)).sum().item(),
        vMax=v.norm(dim=1).max().item(),
        rhoMin=rho.min().item(), rhoMax=rho.max().item(), rhoStd=rho.std().item(),
    )


def run(nx: int, xi: float, tLimit: float, outDir: str, snapshotEvery: int, printEvery: int, dtFactor: float = 1.0):
    os.makedirs(outDir, exist_ok=True)
    system, config, schemeConfig, h, nu = buildSpinupSystem(nx, xi)
    dt = CFL_FACTOR * h / SOUND_SPEED * dtFactor
    nSteps = int(tLimit / dt)
    print(f'nx={nx} xi={xi} h={h:.5f} nu={nu:.6f} (alpha={ALPHA}) dt={dt:.6f} (dtFactor={dtFactor}) -> {nSteps} steps for t={tLimit}s')

    scheme = get_scheme('RK4')
    state = system
    t0 = time.time()
    for i in range(1, nSteps + 1):
        result = scheme(state=state, f=f_acoustic_core, dt=dt, config=config, schemeConfig=schemeConfig, verbose=False)
        state = result.state
        s = get_reference_state(state)
        d = diagnostics(s)
        if not math.isfinite(d['KE']) or d['rhoMax'] > 100.0 * schemeConfig.restDensity:
            print(f'DIVERGED at step {i} (t={i*dt:.4f}s): KE={d["KE"]}, rhoMax={d["rhoMax"]}')
            return
        state.adjacency = buildVerletList(s, config.domain, 1.0, SupportScheme.SuperSymmetric, None)
        if i % printEvery == 0 or i == nSteps:
            elapsed = time.time() - t0
            print(f'step {i:5d}/{nSteps} t={i*dt:7.4f}s  KE={d["KE"]:.6f}  vMax={d["vMax"]:.4f}  '
                  f'rho=[{d["rhoMin"]:.5f},{d["rhoMax"]:.5f}] rhoStd={d["rhoStd"]:.3e}  '
                  f'wall={elapsed:.1f}s ({elapsed/i*1000:.1f}ms/step)')
        if i % snapshotEvery == 0 or i == nSteps:
            torch.save({
                'positions': s.positions.detach().cpu(), 'velocities': s.velocities.detach().cpu(),
                'densities': s.densities.detach().cpu(), 'supports': s.supports.detach().cpu(),
                'masses': s.masses.detach().cpu(), 'kinds': s.kinds.detach().cpu(),
                'materials': s.materials.detach().cpu(), 'UIDs': s.UIDs.detach().cpu(),
                'UIDcounter': s.UIDcounter,
                'nx': nx, 't': i * dt, 'step': i, 'dt': dt, 'h': h, 'nu': nu, 'alpha': ALPHA,
                'xi': xi, 'k': K, 'L': L, 'soundSpeed': SOUND_SPEED, 'cflFactor': CFL_FACTOR,
            }, os.path.join(outDir, f'snap_nx{nx}_xi{xi}_{i:06d}.pt'))
    print(f'done, total wall time {time.time()-t0:.1f}s')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--nx', type=int, default=48)
    p.add_argument('--xi', type=float, default=1.0)
    p.add_argument('--dt-factor', type=float, default=1.0)
    p.add_argument('--tlimit', type=float, default=4.0)
    p.add_argument('--out', type=str, default='/tmp/kolmogorovSpinup')
    p.add_argument('--snapshot-every', type=int, default=200)
    p.add_argument('--print-every', type=int, default=20)
    args = p.parse_args()
    run(args.nx, args.xi, args.tlimit, args.out, args.snapshot_every, args.print_every, dtFactor=args.dt_factor)
