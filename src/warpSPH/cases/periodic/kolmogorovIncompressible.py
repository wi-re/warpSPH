"""Forced periodic Kolmogorov flow using native pressure projection."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
from warpSPHCore import sphKernelScale

from ...configurations import BoundaryCondition, BoundaryConditionType
from ...math import getPeriodicPositions
from ...modules import computeDensities
from ...modules.noise.sampleDivergenceFree import generateNoiseInterpolator
from ...runner import Case, RunContext, caseMain, registerCase
from ...sample.weaklyCompressible import setupBasicWeaklyCompressibleInitialState
from ...utils import buildDomainDescription
from ..common.weaklyCompressible import particleDistributionMetrics
from ..common.plotting import Field, particlePlot

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
    from ...modules.timestep.projection import projectionTimestep
    return projectionTimestep(ctx, state)


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
    scheme='dfsph',
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
