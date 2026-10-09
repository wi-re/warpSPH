"""Kelvin-Helmholtz instability (2D), compressible.

The script form of this case was
`examples/compressible/12-kelvin-helmholtz.ipynb`. Two counter-streaming layers
of different density, seeded with a single-mode transverse perturbation, so the
roll-up is deterministic and the same at every resolution.
"""

from __future__ import annotations

import math
from typing import Dict

import torch

from ..caseUtils import sampleKHH
from ..runner import Case, RunContext, caseMain, registerCase
from .compressible import (COMPRESSIBLE_DEFAULTS, COMPRESSIBLE_PARAMS,
                           compressibleDiagnostics, configureCompressible,
                           paramExtraData)
from .plotting import Field, particlePlot

__all__ = ['kelvinHelmholtzCase', 'KELVIN_HELMHOLTZ_FIELDS']


def configureScheme(ctx: RunContext) -> None:
    # The KH box is the unit square with the origin at a corner, not the
    # symmetric box `buildDomainDescription` returns.
    domain = ctx.config.domain
    domain.min = torch.zeros(ctx.spec.dim, device=ctx.device, dtype=ctx.dtype)
    domain.max = torch.ones(ctx.spec.dim, device=ctx.device, dtype=ctx.dtype) * ctx.spec.L
    configureCompressible(ctx)


def buildSystem(ctx: RunContext):
    return sampleKHH(
        ctx.param('rho1'), ctx.param('rho2'), ctx.param('v1'), ctx.param('v2'),
        ctx.param('delta'), ctx.param('sigma'), ctx.param('freq'), ctx.param('w0'),
        ctx.spec.nx, ctx.config, ctx.schemeConfig,
        ctx.SimulationState, ctx.SimulationSystem, smoothDensity=bool(ctx.param('smoothDensity')))


#: The two panels, exported so a notebook can pass them to
#: `buildFieldPlotter`/`refreshFieldPlotter` directly (see `hydrostatic.py`).
KELVIN_HELMHOLTZ_FIELDS = [
    Field('velocities', 'Velocity', colorMap='viridis', mapping='L2'),
    Field('densities', 'density', colorMap='RdBu', colorMapKind='diverging',
          flip=True, gridResolution=1024),
]

setupPlot, updatePlot = particlePlot(KELVIN_HELMHOLTZ_FIELDS)


def diagnostics(ctx: RunContext, state) -> Dict[str, float]:
    return dict(compressibleDiagnostics(ctx, state),
                maxVelocity=torch.linalg.norm(state.state.velocities, dim=-1)
                .max().detach().cpu().item())


kelvinHelmholtzCase = registerCase(Case(
    name='kelvinHelmholtz',
    scheme='CRKSPH',
    description='Kelvin-Helmholtz instability (2D), compressible SPH.',
    buildSystem=buildSystem,
    configureScheme=configureScheme,
    diagnostics=diagnostics,
    setupPlot=setupPlot,
    updatePlot=updatePlot,
    extraData=paramExtraData,
    defaults=dict(
        COMPRESSIBLE_DEFAULTS,
        caseName='12-kelvinHelmholtz',
        dim=2,
        nx=256,
        L=1.0,
        tLimit=4.0,
        plotInterval=1,
        storeInterval=500,
    ),
    params=dict(
        COMPRESSIBLE_PARAMS,
        rho1=1.0,
        rho2=2.0,
        v1=0.5,
        v2=-0.5,
        delta=0.025,
        freq=4.0,
        w0=0.1,
        sigma=0.05 / math.sqrt(2.0),
        smoothDensity=0,     # 1: Frontiere 2017 Eq. (100) density ramp across the interface
    ),
))


if __name__ == '__main__':
    caseMain(kelvinHelmholtzCase)
