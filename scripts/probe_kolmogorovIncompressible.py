"""Probe (per the project owner's own instruction, 2026-08-26): does this
codebase's existing incompressible SPH solver ("DFSPH", `scheme='divergenceFree'`,
`schemes/divergenceFree.py`) already deliver -- for free, with no JFNK -- some or all of
what `JFNK_PLAN.md`'s compressible-acoustic-core effort (Phases B/E1/E1.5/E1.6/
E1.7) is chasing, on the *same* 2D forced Kolmogorov-flow shear-instability
problem?

DFSPH has no acoustic mode by construction: density is not an integrated
stiff quantity coupled to an equation of state, it is corrected every step by
a relaxed-Jacobi/Krylov Poisson-style pressure projection
(`modules/incompressible/divergenceFree.py`'s `solveDivergenceFree`, plus a
second constant-density correction in `IncompressibleSystem.finalize`,
`modules/incompressible/incompressible.py`'s `solveIncompressible`). So DFSPH
pays no acoustic-CFL `dt` tax at all -- the question this script measures,
not assumes: does that mean DFSPH also sidesteps the Kolmogorov-flow shear
instability the JFNK plan's own Phase E1 found (it should not -- that
instability is a property of the *forced flow*, not of how compressibility is
discretised), and what does DFSPH pay instead (relaxed-Jacobi/Krylov
iterations per step, not Newton/GMRES)?

This exact combination -- Kolmogorov forcing driven through the
`divergenceFree` scheme -- has never been run in this codebase before
(`cases/kolmogorov.py` is hardcoded to `scheme='deltaSPH'`; the only existing
`divergenceFree` case is `cases/tgv.py`, unforced). Per JFNK_PLAN.md's own
convention, this script deliberately does NOT edit `cases/kolmogorov.py` or
`cases/tgv.py` (both existing, tested, registered production code) -- it
reuses the pieces those two cases are already built from directly:
`sample.weaklyCompressible.setupBasicWeaklyCompressibleInitialState` (the same
sampling `cases/tgv.py`'s own `buildSystem` calls, for the very same
`scheme='divergenceFree'`), the Kolmogorov `v_x = xi*sin(k*pi*y)` forcing
closure (re-derived here rather than imported -- it is a nested closure
inside `cases/kolmogorov.py`'s own `initialConditions`, not a standalone
function; the Perlin-noise symmetry-breaking term is dropped, matching
JFNK_PLAN.md Phase E1's own precedent that a small position jitter is enough
to seed the instability), and the `BoundaryCondition`/`boundaryConditions`
forcing mechanism `IncompressibleSPHConfig` shares field-for-field with
`WeaklyCompressibleSPHConfig` (both route through
`modules/boundaryConditions/bcs.py`'s `computeForcing`).

Usage: `python scripts/probe_kolmogorovIncompressible.py [--nx 24] [--xi 1.0]
[--k 4.0] [--nu 0.0] [--nsteps 200] [--cfl 0.3] [--jitter 0.01] [--print-every 10]
[--verbose-every 0]`
"""

from __future__ import annotations

import argparse
import contextlib
import io
import math
import re
import time

from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')

import numpy as np
import torch

from warpSPHCore import (KernelFunctions, SupportScheme, buildVerletList,
                         n_h_to_nH, sphKernelScale)
from warpSPHIntegrators import IntegrationSchemeType, get_reference_state
from warpSPH.configurations import BoundaryCondition, BoundaryConditionType, buildConfig
from warpSPH.modules import computeDensities, shuffleParticles
from warpSPH.modules.deltaSPH import alphaToNu
from warpSPH.math import getPeriodicPositions
from warpSPH.sample.weaklyCompressible import setupBasicWeaklyCompressibleInitialState
from warpSPH.schemes import buildScheme
from warpSPH.utils import buildDomainDescription

# `cases/kolmogorov.py`'s own defaults, reused verbatim so this probe targets
# the same operating point as `JFNK_PLAN.md`'s E1.6/E1.7 compressible probes.
K_DEFAULT = 4.0
ALPHA_REFERENCE = 0.01
# Used only to convert `alpha` into a `nu` number that is comparable to
# probe_kolmogorovSpinup.py's own nu -- DFSPH itself has no sound speed at all
# (see the finding recorded in the writeup: `IncompressibleSPHConfig` still
# carries a `fixedSoundSpeed`/`dt_acousticConstraint` field structurally,
# inherited from the shared config shape, but nothing in `divergenceFree_step` or
# `IncompressibleSystem.finalize` ever reads either).
SOUND_SPEED_REFERENCE = 10.0
L_DEFAULT = 2.0


def buildKolmogorovIncompressibleSystem(nx, xi, k, nu, jitter, L=L_DEFAULT,
                                        cflFactor=0.3, maxDt=1e-1, seed=0, device=None, dtype=None):
    # `buildDomainDescription`'s own default device is 'cpu'; passing `None`
    # through explicitly overrides that default with `None` rather than
    # resolving it, silently building the domain (and everything sampled onto
    # it) on CPU while `buildConfig` resolves cuda:0 for the rest of the
    # config -- found running this script's own smoke test (every kernel
    # compiled "on device 'cpu'" despite a GPU being available). Fixed by
    # resolving the device here first, matching `warpSPH.runner.runner.
    # buildContext`'s own convention.
    if device is None:
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    if dtype is None:
        dtype = torch.float32
    domain = buildDomainDescription(L, 2, True, device, dtype)
    config, integrator = buildConfig(
        domain=domain, dim=2, dx=L / nx, nx=nx,
        kernel=KernelFunctions.Wendland2,
        integrationScheme=IntegrationSchemeType.semiImplicitEuler,
        supportMode=SupportScheme.SuperSymmetric,
        targetNeighbors=n_h_to_nH(4.0, 2),
        cflFactor=cflFactor, adaptiveDt=True, minDt=1e-7, maxDt=maxDt,
        device=device, dtype=dtype,
    )

    bundle = buildScheme('divergenceFree')
    schemeConfig = bundle.SimulationConfig()
    system = setupBasicWeaklyCompressibleInitialState(
        nx, config, schemeConfig, bundle.SimulationState, bundle.SimulationSystem)

    # Normalise mass so the sampled density lands on rho0=1, matching
    # `cases/tgv.py`'s own `buildSystem` for this same scheme.
    densities = computeDensities(system.state, config, schemeConfig, None)
    system.state.masses = system.state.masses / densities.mean() * 1.0

    torch.manual_seed(seed)
    if jitter:
        system.state.positions = shuffleParticles(system.state, config, schemeConfig, 0,
                                                  jitterAmount=jitter)

    h = system.state.supports.mean().item()

    schemeConfig.diffusionParams.inviscid = False
    schemeConfig.diffusionParams.viscidNu = nu
    schemeConfig.surfaceDetectionConfig.active = False
    schemeConfig.shiftProperties.active = False

    domainRef = config.domain

    def forcing(state, cfg, schemeCfg, x, d, n, t, dt):
        positions = getPeriodicPositions(x, domainRef)
        u_x = xi * torch.sin(k * np.pi * positions[:, 1])
        u_y = torch.zeros_like(u_x)
        return torch.stack([u_x, u_y], dim=1) * state.masses.unsqueeze(1)

    def fullDomainSdf(x):
        # The whole periodic domain is "inside" the fluid region -- no
        # obstacle, no walls, matching `cases/kolmogorov.py`'s own
        # `domainFluidSdf` on a case with no boundary region.
        return -torch.ones(x.shape[0], device=x.device, dtype=x.dtype), torch.zeros_like(x)

    schemeConfig.boundaryConditions = [BoundaryCondition(
        type=BoundaryConditionType.dynamic, sdf=fullDomainSdf, forcingFunctions=[forcing])]

    adjacency = buildVerletList(system.state, config.domain, verletScale=config.verletScale,
                                supportMode=SupportScheme.SuperSymmetric,
                                priorNeighborhood=None, verbose=False)
    system.adjacency = adjacency
    return system, config, schemeConfig, bundle, h


def diagnostics(state):
    m, v, rho = state.masses, state.velocities, state.densities
    return dict(
        KE=(0.5 * m * (v ** 2).sum(-1)).sum().item(),
        vMax=v.norm(dim=1).max().item(),
        rhoMin=rho.min().item(), rhoMax=rho.max().item(), rhoStd=rho.std().item(),
    )


def pickDt(state, h, nu, cflFactor, kernelScale, minDt, maxDt):
    """DFSPH's own natural timestep: advective + viscous CFL, no acoustic term
    at all (there is no sound speed in this scheme's actual physics)."""
    vMax = state.velocities.norm(dim=1).max().item()
    dt_adv = cflFactor * h / max(vMax, 1e-3)
    dt_visc = 0.125 * h ** 2 / kernelScale / nu if nu > 0 else float('inf')
    return float(min(max(min(dt_adv, dt_visc), minDt), maxDt))


def _extractDivFreeIterations(stepResult):
    """Pull `solveDivergenceFree`'s own iteration count (the projection inside
    `divergenceFree_step` itself) off `stepResult.stages[0].aux`, which is
    `(adjacency, currentState, (errors, pressures))` -- confirmed by
    inspection, not assumed (`divergenceFree_step` returns
    `(update, adjacency, currentState, (errors, pressures))`, and
    `warpSPHIntegrators.euler.integrateSemiImplicitEuler` binds everything
    past the first return value to `r1`, stashed verbatim as `StageResult.aux`).
    Returns `None` if that shape ever changes underneath this probe."""
    try:
        errorsDivFree = stepResult.stages[0].aux[2][0]
        return len(errorsDivFree)
    except Exception:
        return None


_INCOMP_ITER_RE = re.compile(r'Solver Iterations: (\d+)')


def _stepWithIncompIterCapture(bundle, system, dt, config, schemeConfig):
    """`IncompressibleSystem.finalize` runs a *second* pressure solve
    (`solveIncompressible`, the constant-density position correction) that is
    not surfaced through `stepResult` at all -- its own `returnValues[-1] =
    (...)` reassignment (`systems/incompressible.py`) rebinds a list slot to a
    new tuple, which does not mutate the `r1` tuple `stages[0].aux` already
    holds a reference to (tuples are immutable), so that second solve's
    iteration count is only ever visible through its `verbose=True` print.
    Captured here by redirecting stdout rather than patching production code."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        stepResult = config_integrator_call(bundle, system, dt, config, schemeConfig, True)
    m = _INCOMP_ITER_RE.search(buf.getvalue())
    return stepResult, (int(m.group(1)) if m else None)


def run(nx, xi, k, nu, jitter, nSteps, cflFactor, printEvery, verboseEvery,
       L=L_DEFAULT, maxDt=1e-1, seed=0, tag=''):
    system, config, schemeConfig, bundle, h = buildKolmogorovIncompressibleSystem(
        nx, xi, k, nu, jitter, L=L, cflFactor=cflFactor, maxDt=maxDt, seed=seed)
    kernelScale = float(sphKernelScale(config.kernel.value, config.dim))

    print(f'[{tag}] nx={nx} xi={xi} k={k} nu={nu:.6f} L={L} h={h:.5f} jitter={jitter} '
         f'cflFactor={cflFactor} maxDt={maxDt} nSteps={nSteps}')

    t = 0.0
    trace = []
    divFreeIters, incompIters = [], []
    t0 = time.time()
    diverged = False
    for i in range(1, nSteps + 1):
        dt = pickDt(system.state, h, nu, cflFactor, kernelScale, config.minDt, config.maxDt)
        sampleIters = verboseEvery > 0 and i % verboseEvery == 0
        if sampleIters:
            stepResult, nIC = _stepWithIncompIterCapture(bundle, system, dt, config, schemeConfig)
        else:
            stepResult = config_integrator_call(bundle, system, dt, config, schemeConfig, False)
            nIC = None
        system = stepResult.state
        t += dt

        s = get_reference_state(system)
        d = diagnostics(s)
        d.update(t=t, dt=dt, step=i)

        nDF = _extractDivFreeIterations(stepResult)
        if nDF is not None:
            divFreeIters.append(nDF)
            d['nDivFreeIters'] = nDF
        if nIC is not None:
            incompIters.append(nIC)
            d['nIncompIters'] = nIC
        trace.append(d)

        if not (math.isfinite(d['KE']) and math.isfinite(d['vMax'])) or d['rhoMax'] > 100.0:
            print(f'[{tag}] DIVERGED at step {i} (t={t:.4f}s): KE={d["KE"]}, rhoMax={d["rhoMax"]}')
            diverged = True
            break

        if i % printEvery == 0 or i == nSteps:
            elapsed = time.time() - t0
            iterStr = (f'  iters(divFree/incomp)={nDF}/{nIC}' if nDF is not None else '')
            print(f'[{tag}] step {i:5d}/{nSteps} t={t:8.5f}s dt={dt:.3e}  KE={d["KE"]:.6f}  '
                 f'vMax={d["vMax"]:.4f}  rho=[{d["rhoMin"]:.5f},{d["rhoMax"]:.5f}] '
                 f'rhoStd={d["rhoStd"]:.3e}{iterStr}  wall={elapsed:.1f}s '
                 f'({elapsed / i * 1000:.1f}ms/step)')

    wallTime = time.time() - t0
    print(f'[{tag}] done: {len(trace)} steps, t={t:.4f}s, wall={wallTime:.1f}s, '
         f'diverged={diverged}')
    if divFreeIters:
        print(f'[{tag}] mean solveDivergenceFree iterations/step: '
             f'{np.mean(divFreeIters):.1f} (n={len(divFreeIters)} steps)')
    if incompIters:
        print(f'[{tag}] mean finalize solveIncompressible iterations/step: '
             f'{np.mean(incompIters):.1f} (n={len(incompIters)} steps sampled)')
    return dict(trace=trace, diverged=diverged, wallTime=wallTime, nSteps=len(trace),
               divFreeIters=divFreeIters, incompIters=incompIters)


def config_integrator_call(bundle, system, dt, config, schemeConfig, verbose):
    from warpSPHIntegrators import getIntegrator
    integrator = getIntegrator('Semi-Implicit Euler')
    return integrator(state=system, f=bundle.stepFunction, dt=dt, config=config,
                      schemeConfig=schemeConfig, verbose=verbose)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--nx', type=int, default=24)
    p.add_argument('--xi', type=float, default=1.0)
    p.add_argument('--k', type=float, default=K_DEFAULT)
    p.add_argument('--nu', type=float, default=None,
                   help='kinematic viscosity; default derives from alpha=0.01 via alphaToNu')
    p.add_argument('--alpha', type=float, default=ALPHA_REFERENCE)
    p.add_argument('--jitter', type=float, default=0.01)
    p.add_argument('--nsteps', type=int, default=200)
    p.add_argument('--cfl', type=float, default=0.3)
    p.add_argument('--maxdt', type=float, default=1e-1)
    p.add_argument('--L', type=float, default=L_DEFAULT)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--print-every', type=int, default=10)
    p.add_argument('--verbose-every', type=int, default=0)
    p.add_argument('--tag', type=str, default='run')
    args = p.parse_args()

    if args.nu is None:
        # h is not known until the system is built; approximate via L/nx*n_h
        # is unreliable, so resolve nu against a throwaway build's own h.
        _sys, _cfg, _, _, _h = buildKolmogorovIncompressibleSystem(
            args.nx, args.xi, args.k, 0.0, args.jitter, L=args.L, cflFactor=args.cfl,
            seed=args.seed)
        nu = alphaToNu(args.alpha, SOUND_SPEED_REFERENCE, _h, 2)
        del _sys, _cfg
    else:
        nu = args.nu

    run(args.nx, args.xi, args.k, nu, args.jitter, args.nsteps, args.cfl,
       args.print_every, args.verbose_every, L=args.L, maxDt=args.maxdt,
       seed=args.seed, tag=args.tag)
