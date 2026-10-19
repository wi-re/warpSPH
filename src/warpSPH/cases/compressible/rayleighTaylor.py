"""Rayleigh-Taylor instability (2D), compressible.

The script form of this case was
`examples/compressible/13-rayleigh-taylor.ipynb`. Heavy fluid over light in a
tall box under constant gravity; the sampler installs both the Dirichlet
boundary bands and the gravity forcing, so `buildSystem` is all this case needs.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from ...caseUtils import sampleRayleighTaylor
from ...configurations.meshless import MeshlessConfig
from ...runner import Case, RunContext, caseMain, registerCase
from .common import (COMPRESSIBLE_DEFAULTS, COMPRESSIBLE_PARAMS,
                           compressibleDiagnostics, configureCompressible,
                           paramExtraData)
from ..common.plotting import Field, particlePlot

from ..common.parameters import COMPRESSIBLE_CASE_PARAMETERS, COMPRESSIBLE_SCHEMES, SPH_SCHEMES
from ...runner.parameters import Parameter as P

__all__ = ['rayleighTaylorCase', 'RAYLEIGH_TAYLOR_FIELDS']


def configureScheme(ctx: RunContext) -> None:
    # A tall box, `aspect` times narrower than it is high, with `band` particle
    # layers of Dirichlet buffer added above and below.
    dx = ctx.spec.L / ctx.spec.nx
    band = ctx.param('band') * dx
    aspect = ctx.param('aspect')
    domain = ctx.config.domain
    domain.min = torch.tensor([-ctx.spec.L / 2 / aspect, -band],
                              device=ctx.device, dtype=ctx.dtype)
    domain.max = torch.tensor([ctx.spec.L / 2 / aspect, ctx.spec.L + band],
                              device=ctx.device, dtype=ctx.dtype)
    configureCompressible(ctx)


def buildSystem(ctx: RunContext):
    if isinstance(ctx.schemeConfig, MeshlessConfig):
        # The core MFM / MFV step is periodic-only and carries no forcing term:
        # this case needs the gravity BC and the Dirichlet buffer bands, so it
        # is deferred for the meshless schemes (COMPRESSIBLE_SPH_PLAN E2).
        raise NotImplementedError(
            'rayleighTaylor under MFM / MFV is deferred: the case needs the '
            'gravity forcing and the Dirichlet buffer bands, which the periodic '
            'core step does not carry (COMPRESSIBLE_SPH_PLAN E2)')
    return sampleRayleighTaylor(
        ctx.param('rho_low'), ctx.param('rho_high'), ctx.param('delta'),
        ctx.param('g'), ctx.spec.L, ctx.spec.L / ctx.spec.nx, ctx.param('aspect'),
        ctx.spec.nx, ctx.config, ctx.schemeConfig,
        ctx.SimulationState, ctx.SimulationSystem)


#: The two panels, exported so a notebook can pass them to
#: `buildFieldPlotter`/`refreshFieldPlotter` directly (see `hydrostatic.py`).
RAYLEIGH_TAYLOR_FIELDS = [
    Field('velocities', 'Velocity', colorMap='viridis', mapping='L2',
          vMin=0.0, vMax=1 / np.sqrt(2)),
    Field('densities', 'density', colorMap='RdBu', colorMapKind='diverging',
          flip=True, gridResolution=1024, vMin=0.95, vMax=2.05),
]

setupPlot, updatePlot = particlePlot(RAYLEIGH_TAYLOR_FIELDS, figsize=(12, 12))


def diagnostics(ctx: RunContext, state) -> Dict[str, float]:
    return dict(compressibleDiagnostics(ctx, state),
                maxVelocity=torch.linalg.norm(state.state.velocities, dim=-1)
                .max().detach().cpu().item())


rayleighTaylorCase = registerCase(Case(
    name='rayleighTaylor',
    scheme='CRKSPH',
    description='Rayleigh-Taylor instability (2D), compressible SPH.',
    buildSystem=buildSystem,
    configureScheme=configureScheme,
    diagnostics=diagnostics,
    setupPlot=setupPlot,
    updatePlot=updatePlot,
    extraData=paramExtraData,
    defaults=dict(
        COMPRESSIBLE_DEFAULTS,
        caseName='13-rayleighTaylor',
        dim=2,
        nx=128,
        L=1.0,
        tLimit=10.0,
        plotInterval=1,
        storeInterval=500,
    ),
    supportedSchemes=SPH_SCHEMES,
    parameterDefinitions={**COMPRESSIBLE_CASE_PARAMETERS,
        'rho_low': P('number', 'Density of the lower fluid.', 'initial', units='density', exclusiveMinimum=0),
        'rho_high': P('number', 'Density of the upper fluid.', 'initial', units='density', exclusiveMinimum=0),
        'delta': P('number', 'Interface perturbation amplitude.', 'initial', units='length', minimum=0),
        'g': P('number', 'Downward gravitational acceleration magnitude.', 'forcing', units='length/time^2', exclusiveMinimum=0),
        'aspect': P('number', 'Box height/width ratio.', 'domain', exclusiveMinimum=0),
        'band': P('integer', 'Dirichlet buffer thickness in particle layers.', 'boundary', minimum=1),
    },
    simulationParameters={'dim': P('integer', 'Dimension supported by this initial-condition sampler.', 'domain', choices=(2,))},
    variants={},
    params=dict(
        COMPRESSIBLE_PARAMS,
        gamma=1.4,
        rho_low=1.0,
        rho_high=2.0,
        delta=0.0025,
        g=0.5,
        aspect=2.0,
        band=20,
    ),
))


if __name__ == '__main__':
    caseMain(rayleighTaylorCase)
