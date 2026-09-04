"""Kolmogorov flow (2D), divergence-free incompressible SPH (DFSPH).

The incompressible sibling of :mod:`kolmogorov` (weakly compressible
`deltaSPH`), added per `warpSPHIntegrators/JFNK_PLAN.md` E1.9/E1.10: does
this codebase's existing incompressible solver (`scheme='divergenceFree'`)
already deliver, for free, some or all of what that plan's JFNK effort is
chasing on the same forced-shear-instability problem? Its scheme mechanics
(`buildSystem`'s mass normalisation, the `divergenceFree` scheme itself) are
`tgv`'s (the only other `divergenceFree` case); its physics (the sinusoidal
`v_x = xi*sin(k*pi*y)` body force plus a Perlin-noise `v_y` term to break
the shear layer's symmetry so it can actually go unstable) are `kolmogorov`'s.
Script form validated first as `scripts/probe_kolmogorovIncompressible.py`
(simpler jitter-only symmetry breaking, no Case machinery) -- this is the
same physics wired through the full `Case` protocol instead, with the
noise-based symmetry breaking `kolmogorov` itself uses rather than that
probe's shortcut.

Two things a naive port of `kolmogorov.py` would get wrong for this scheme,
both found running the probe and worth restating here:

- DFSPH has no acoustic term (no sound speed drives its `dt` the way
  `deltaSPH`'s does) -- but it still needs a real per-step adaptive `dt`,
  because this flow's own velocity scale changes by roughly an order of
  magnitude between the quiescent start and the saturated turbulent state
  (`JFNK_PLAN.md` E1.6's finding on the compressible core; the same is true
  here). `kolmogorovIncompressibleTimestep` below is that hook -- advective
  + viscous CFL, no acoustic term, mirroring the formula validated across
  nx=24..128 in the probe's own `pickDt`.
- `IncompressibleSystem`'s own implicit particle-shifting mechanism (a
  second, constant-density pressure solve in `finalize`) needed a real bug
  fix (`systems/incompressible.py`, E1.10) before this case was stable with
  the default forcing/viscosity at production resolution -- already applied,
  nothing this case needs to work around.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
from warpSPHCore import sphKernelScale

from ..configurations import BoundaryCondition, BoundaryConditionType
from ..math import getPeriodicPositions
from ..modules import computeDensities
from ..modules.noise.sampleDivergenceFree import generateNoiseInterpolator
from ..runner import Case, RunContext, caseMain, registerCase
from ..sample.weaklyCompressible import setupBasicWeaklyCompressibleInitialState
from ..utils import buildDomainDescription
from .weaklyCompressible import particleDistributionMetrics
from .plotting import Field, particlePlot

__all__ = ['kolmogorovIncompressibleCase']


def configureScheme(ctx: RunContext) -> None:
    ctx.schemeConfig.surfaceDetectionConfig.active = False
    ctx.schemeConfig.diffusionParams.inviscid = False
    ctx.schemeConfig.diffusionParams.viscidNu = ctx.param('nu')
    ctx.schemeConfig.shiftProperties.active = ctx.param('shifting')


def buildSystem(ctx: RunContext):
    system = setupBasicWeaklyCompressibleInitialState(
        ctx.spec.nx, ctx.config, ctx.schemeConfig, ctx.SimulationState, ctx.SimulationSystem)

    # Normalise mass so the sampled density lands on rho0, matching `tgv`'s
    # own `buildSystem` for this same scheme.
    rho0 = ctx.param('rho0')
    densities = computeDensities(system.state, ctx.config, ctx.schemeConfig, None)
    system.state.masses = system.state.masses / densities.mean() * rho0
    return system


def initialConditions(ctx: RunContext, system) -> None:
    domain = ctx.config.domain
    xi = ctx.param('xi')
    k = ctx.param('k')
    noiseLevel = ctx.param('noiseLevel')

    # No boundary/obstacle region exists in this case (unlike `kolmogorov`,
    # which pads by its own `band` param to match a boundary band width) --
    # this domain is periodic everywhere, so the noise field is built at the
    # plain domain size.
    domainCpu = buildDomainDescription(
        ctx.spec.L, ctx.spec.dim, ctx.spec.periodic, torch.device('cpu'), ctx.dtype)
    noise = generateNoiseInterpolator(
        ctx.spec.nx * 2, ctx.spec.nx * 2, domainCpu, dim=ctx.config.domain.dim,
        octaves=ctx.param('octaves'), lacunarity=ctx.param('lacunarity'),
        persistence=ctx.param('persistence'), baseFrequency=ctx.param('baseFrequency'),
        tileable=ctx.param('tileable'), kind=ctx.param('kind'), seed=ctx.param('seed'))

    def forcing(state, config, schemeConfig, x, d, n, t, dt):
        positions = getPeriodicPositions(x, domain)
        u_x = xi * torch.sin(k * np.pi * positions[:, 1])
        u_y = noise(positions.detach().cpu()).to(dtype=x.dtype, device=x.device) * noiseLevel
        return torch.stack([u_x, u_y], dim=1) * state.masses.unsqueeze(1)

    def fullDomainSdf(x):
        # The whole periodic domain is "inside" the fluid region -- no
        # obstacle, no walls, matching `probe_kolmogorovIncompressible.py`'s
        # own `fullDomainSdf`.
        return -torch.ones(x.shape[0], device=x.device, dtype=x.dtype), torch.zeros_like(x)

    ctx.schemeConfig.boundaryConditions = [BoundaryCondition(
        type=BoundaryConditionType.dynamic, sdf=fullDomainSdf, forcingFunctions=[forcing])]


def kolmogorovIncompressibleTimestep(ctx: RunContext, state) -> float:
    """Advective + viscous CFL, no acoustic term -- DFSPH has no sound speed.

    The advective term is Bender & Koschier's published condition
    `dt <= 0.4 d / |v_max|` written in their units (`d` = particle diameter =
    `dx`), so `cflFactor` is directly comparable with the 0.4 they state.

    `modules.timestep.computeTimestep`'s generic dispatcher only recognises
    `WeaklyCompressibleSystem`; every other system falls through to the fully
    compressible formula, which reads fields (`soundspeeds`, an EOS pressure)
    this scheme never populates meaningfully. This mirrors the formula
    validated at nx=24..128 in `scripts/probe_kolmogorovIncompressible.py`'s
    own `pickDt` instead: a fixed `dt` picked for this case's quiescent start
    would be far too large once the flow transitions to its much faster
    saturated turbulent state (the same effect `JFNK_PLAN.md` E1.6 found on
    the compressible core), and, unlike `tgv`'s initial condition (immediately
    at speed `uMag` everywhere), this one starts genuinely at rest, so the
    `max(vMax, 1e-3)` floor and the `config.maxDt` ceiling both matter from
    the very first step, not just once the flow has spun up.
    """
    particles = state.state
    h = particles.supports.mean().item()
    vMax = particles.velocities.norm(dim=1).max().item()
    nu = ctx.param('nu', 0.0)
    kernelScale = float(sphKernelScale(ctx.config.kernel.value, ctx.config.dim))
    # The advective condition is Bender & Koschier's, in their units: `dt <=
    # 0.4 * d / |v_max|` with `d` the particle *diameter*, i.e. the sampling
    # spacing `dx`. This used to be written against the support radius
    # instead (`cflFactor * h`), which is a different condition in two ways:
    # it is `n_h` times larger (4x here, so the historical `cflFactor=0.3`
    # allowed 1.2 spacings of travel per step, 3x the published limit), and it
    # silently rescales with the neighbour count, so the same `cflFactor`
    # meant different physics at different `n_h`. Written in `dx` the number
    # in the config *is* the published constant and is resolution- and
    # `n_h`-independent. See `DFSPH_IMPROVEMENT_PLAN.md` Part 12.
    dt_adv = ctx.config.cflFactor * ctx.config.dx / max(vMax, 1e-3)
    # The viscous limit stays in `h`: it is a diffusion condition over the
    # smoothing length, not an advection condition over the particle spacing.
    dt_visc = 0.125 * h ** 2 / kernelScale / nu if nu > 0 else float('inf')
    return float(min(max(min(dt_adv, dt_visc), ctx.config.minDt), ctx.config.maxDt))


def diagnostics(ctx: RunContext, state) -> Dict[str, float]:
    particles = state.state
    fluid = particles.kinds == 0 if hasattr(particles, 'kinds') else slice(None)
    velocities = particles.velocities[fluid]
    densities = particles.densities[fluid]
    out = {
        'kineticEnergy': (0.5 * particles.masses[fluid]
                          * (velocities ** 2).sum(dim=-1)).sum().detach().cpu().item(),
        'maxVelocity': torch.linalg.norm(velocities, dim=-1).max().detach().cpu().item(),
        'minDensity': densities.min().detach().cpu().item(),
        'maxDensity': densities.max().detach().cpu().item(),
        'densityStd': densities.std().detach().cpu().item(),
    }
    out.update(particleDistributionMetrics(ctx, state))
    return out


setupPlot, updatePlot = particlePlot([
    Field('velocities', 'velocities', colorMap='viridis', mapping='L2Norm'),
    Field('densities', 'densities', colorMap='RdBu', colorMapKind='diverging',
          flip=True, midPoint=1.0),
])


def extraData(ctx: RunContext, state) -> Dict[str, Any]:
    return {k: ctx.param(k) for k in kolmogorovIncompressibleCase.params}


kolmogorovIncompressibleCase = registerCase(Case(
    name='kolmogorovIncompressible',
    scheme='divergenceFree',
    description='Kolmogorov flow (2D), divergence-free incompressible SPH (DFSPH).',
    buildSystem=buildSystem,
    configureScheme=configureScheme,
    initialConditions=initialConditions,
    diagnostics=diagnostics,
    setupPlot=setupPlot,
    updatePlot=updatePlot,
    extraData=extraData,
    timestep=kolmogorovIncompressibleTimestep,
    defaults=dict(
        caseName='04-kolmogorovIncompressible',
        dim=2,
        nx=128,
        L=2.0,
        n_h=4.0,
        periodic=True,
        kernel='Wendland2',
        integrationScheme='semiImplicitEuler',
        supportMode='SuperSymmetric',
        tLimit=10.0,
        dt=1e-3,
        adaptiveDt=True,
        # Bender & Koschier's published constant, now that `cflFactor`
        # multiplies the particle diameter rather than the support radius.
        cflFactor=0.4,
        minDt=1e-7,
        maxDt=1e-1,
    ),
    params=dict(
        rho0=1.0,
        nu=0.0,
        xi=1.0,
        k=4,
        noiseLevel=0.01,
        octaves=3,
        lacunarity=2,
        persistence=0.5,
        baseFrequency=2,
        tileable=True,
        kind='perlin',
        seed=45906734,
        shifting=False,
        markerSize=8,
    ),
))


if __name__ == '__main__':
    caseMain(kolmogorovIncompressibleCase)
