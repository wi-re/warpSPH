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

from ...caseUtils import sampleKHH
from ...configurations.meshless import MeshlessConfig
from ...runner import Case, RunContext, caseMain, registerCase
from .common import (COMPRESSIBLE_DEFAULTS, COMPRESSIBLE_PARAMS,
                           compressibleDiagnostics, configureCompressible,
                           paramExtraData)
from ..common.plotting import Field, particlePlot

from ..common.parameters import COMPRESSIBLE_CASE_PARAMETERS, COMPRESSIBLE_SCHEMES, SPH_SCHEMES
from ...runner.parameters import Parameter as P

__all__ = ['kelvinHelmholtzCase', 'KELVIN_HELMHOLTZ_FIELDS']


def configureScheme(ctx: RunContext) -> None:
    # The KH box is the unit square with the origin at a corner, not the
    # symmetric box `buildDomainDescription` returns.
    domain = ctx.config.domain
    domain.min = torch.zeros(ctx.spec.dim, device=ctx.device, dtype=ctx.dtype)
    domain.max = torch.ones(ctx.spec.dim, device=ctx.device, dtype=ctx.dtype) * ctx.spec.L
    configureCompressible(ctx)


def buildSystem(ctx: RunContext):
    if isinstance(ctx.schemeConfig, MeshlessConfig):
        from ...caseUtils.meshless import toMeshlessSystem
        from ...systems.compressibleMonaghan import CompressibleState, CompressibleSystem
        comp = sampleKHH(
            ctx.param('rho1'), ctx.param('rho2'), ctx.param('v1'), ctx.param('v2'),
            ctx.param('delta'), ctx.param('sigma'), ctx.param('freq'), ctx.param('w0'),
            ctx.spec.nx, ctx.config, ctx.schemeConfig,
            CompressibleState, CompressibleSystem, smoothDensity=bool(ctx.param('smoothDensity')),
            jitter=ctx.param('jitter', 0.0))
        return toMeshlessSystem(comp, ctx.config, ctx.schemeConfig)
    return sampleKHH(
        ctx.param('rho1'), ctx.param('rho2'), ctx.param('v1'), ctx.param('v2'),
        ctx.param('delta'), ctx.param('sigma'), ctx.param('freq'), ctx.param('w0'),
        ctx.spec.nx, ctx.config, ctx.schemeConfig,
        ctx.SimulationState, ctx.SimulationSystem, smoothDensity=bool(ctx.param('smoothDensity')),
        jitter=ctx.param('jitter', 0.0))


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
    supportedSchemes=COMPRESSIBLE_SCHEMES,
    parameterDefinitions={**COMPRESSIBLE_CASE_PARAMETERS,
        'rho1': P('number', 'Density of the outer shear layer.', 'initial', units='density', exclusiveMinimum=0),
        'rho2': P('number', 'Density of the inner shear layer.', 'initial', units='density', exclusiveMinimum=0),
        'v1': P('number', 'Streamwise velocity of the outer layer.', 'initial', units='velocity'),
        'v2': P('number', 'Streamwise velocity of the inner layer.', 'initial', units='velocity'),
        'delta': P('number', 'Velocity-interface smoothing thickness.', 'initial', units='length', exclusiveMinimum=0),
        'freq': P('number', 'Transverse seed spatial frequency.', 'initial', exclusiveMinimum=0),
        'w0': P('number', 'Transverse velocity perturbation amplitude.', 'initial', units='velocity'),
        'sigma': P('number', 'Width of the transverse perturbation envelope.', 'initial', units='length', exclusiveMinimum=0),
        'smoothDensity': P('integer', 'Use the smooth Frontiere density interface (1) or sharp interface (0).', 'initial', choices=(0, 1)),
        'jitter': P('number', 'Initial displacement as a fraction of lattice spacing.', 'sampling', minimum=0),
    },
    simulationParameters={'dim': P('integer', 'Dimension supported by this initial-condition sampler.', 'domain', choices=(2,))},
    variants={'smooth-density': {'params': {'smoothDensity': 1}}},
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
        jitter=0.0,          # X4: IC displacement as a fraction of the lattice spacing (0 = the exact lattice)
    ),
))


if __name__ == '__main__':
    caseMain(kelvinHelmholtzCase)
