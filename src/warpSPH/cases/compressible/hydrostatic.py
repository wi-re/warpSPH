"""Hydrostatic equilibrium (2D), compressible.

The script form of this case was `examples/compressible/08-hydrostatic.ipynb`.
A dense square sits in a lighter background at uniform pressure, so the exact
solution is "nothing happens": every velocity should stay at zero. What the
plot actually shows is the spurious surface tension at the density jump, which
is the point of the test.
"""

from __future__ import annotations

from typing import Dict
from dataclasses import replace
from ...configurations.pesph import PESPHConfig

import torch

from ...caseUtils import buildHydrostaticInitialState
from ...caseUtils.compressible.materialValidation import hostScalars
from ...runner import Case, RunContext, caseMain, registerCase
from .common import (COMPRESSIBLE_DEFAULTS, COMPRESSIBLE_PARAMS,
                           compressibleDiagnostics, configureCompressible,
                           paramExtraData)
from ..common.plotting import Field, particlePlot

from ..common.parameters import COMPRESSIBLE_CASE_PARAMETERS, COMPRESSIBLE_SCHEMES, SPH_SCHEMES
from ...runner.parameters import Parameter as P

__all__ = ['hydrostaticCase', 'HYDROSTATIC_FIELDS']


def buildSystem(ctx: RunContext):
    # The legacy sampler changes its support iteration budget to 16, then 1.
    # PESPH needs its configured stage solve after sampling; isolate those
    # sampling-only assignments without changing the other schemes' convention.
    samplingConfig = replace(ctx.schemeConfig) if isinstance(ctx.schemeConfig, PESPHConfig) else ctx.schemeConfig
    system = buildHydrostaticInitialState(
        ctx.param('rho_low'), ctx.param('rho_high'), ctx.spec.nx,
        ctx.config, samplingConfig, ctx.SimulationState, ctx.SimulationSystem,
        shape=ctx.param('shape'), boost=ctx.param('boost'))
    ctx.scratch['contactPositions'] = system.state.positions.clone()
    ctx.scratch['contactEnergy'] = system.state.totalEnergies.sum().detach()
    ctx.scratch['contactDensity'] = system.state.masses / (ctx.spec.L/ctx.spec.nx)**ctx.spec.dim
    ctx.velocityScale = 1.
    ctx.velocityScaleSource = 'unit contact sound-speed scale'
    return system


#: The two panels, as `particlePlot`/`buildFieldPlotter` want them -- named
#: and exported so a notebook can pass them to `buildFieldPlotter`/
#: `refreshFieldPlotter` directly instead of re-deriving them.
HYDROSTATIC_FIELDS = [
    Field('velocities', 'velocities', colorMap='viridis', mapping='L2Norm'),
    Field('densities', 'densities', colorMap='cividis', flip=True),
]

setupPlot, updatePlot = particlePlot(HYDROSTATIC_FIELDS)


def diagnostics(ctx: RunContext, state) -> Dict[str, float]:
    # maxVelocity is the actual figure of merit: it is exactly zero in the
    # continuum solution, so whatever it reaches is discretisation error.
    s = state.state
    velocity = s.velocities.clone()
    velocity[:, 0] -= ctx.param('boost')
    displacement = s.positions - ctx.scratch['contactPositions']
    displacement[:, 0] -= ctx.param('boost') * state.t
    span = ctx.config.domain.max - ctx.config.domain.min
    displacement = torch.remainder(displacement + .5 * span, span) - .5 * span
    return hostScalars(dict(compressibleDiagnostics(ctx, state),
                maxVelocity=torch.linalg.vector_norm(s.velocities, dim=-1).max(),
                maxPeculiarVelocity=torch.linalg.vector_norm(velocity, dim=-1).max(),
                pressureError=(s.pressures - 1).abs().max(),
                shapeDisplacement=torch.linalg.vector_norm(displacement, dim=-1).max(),
                peculiarKineticEnergy=(.5 * s.masses * velocity.square().sum(-1)).sum(),
                densityRelativeL1=((s.densities-ctx.scratch['contactDensity']).abs()/ctx.scratch['contactDensity']).mean(),
                relativeEnergyDrift=(s.masses * (s.internalEnergies + .5 * s.velocities.square().sum(-1))).sum() / ctx.scratch['contactEnergy'] - 1))


hydrostaticCase = registerCase(Case(
    name='hydrostatic',
    scheme='CRKSPH',
    description='Hydrostatic equilibrium of a dense square (2D), compressible SPH.',
    buildSystem=buildSystem,
    configureScheme=configureCompressible,
    diagnostics=diagnostics,
    setupPlot=setupPlot,
    updatePlot=updatePlot,
    extraData=paramExtraData,
    extraFields=('internalEnergies', 'supports'),
    defaults=dict(
        COMPRESSIBLE_DEFAULTS,
        caseName='08-hydrostatic',
        dim=2,
        nx=200,
        L=1.0,
        tLimit=10.0,
        # buildHydrostaticInitialState leaves dt unset, so this is the
        # notebook's explicit value.
        dt=2.5e-3,
        plotInterval=1,
        storeInterval=500,
    ),
    supportedSchemes=(*SPH_SCHEMES, 'PESPH'),
    parameterDefinitions={**COMPRESSIBLE_CASE_PARAMETERS,
        'rho_low': P('number', 'Background density.', 'initial', units='density', exclusiveMinimum=0),
        'rho_high': P('number', 'Density of the central square patch.', 'initial', units='density', exclusiveMinimum=0),
        'E0': P('number', 'Unused legacy export label; the sampler determines initial pressure.', 'legacy', readOnly=True, deprecated=True),
        'shape': P('string', 'Pressure-balanced inclusion geometry.', 'initial', choices=('box', 'ellipse')),
        'boost': P('number', 'Common x velocity of both gases.', 'initial', units='velocity'),
    },
    simulationParameters={'dim': P('integer', 'Dimension supported by this initial-condition sampler.', 'domain', choices=(2,))},
    variants={
        'literature': {'simulation': {'nx': 100, 'tLimit': 7.}, 'params': {'rho_high': 4., 'gamma': 1.5}},
        'twoGas': {'params': {'rho_high': 4.}, 'solver': {'materials': [
            {'name': 'background', 'eos': {'type': 'idealGas', 'gamma': 1.4}},
            {'name': 'inclusion', 'eos': {'type': 'idealGas', 'gamma': 5/3}}]}},
        'advectedEllipse': {'params': {'rho_high': 4., 'shape': 'ellipse', 'boost': 1.}, 'solver': {'materials': [
            {'name': 'background', 'eos': {'type': 'idealGas', 'gamma': 1.4}},
            {'name': 'inclusion', 'eos': {'type': 'idealGas', 'gamma': 5/3}}]}},
    },
    params=dict(
        COMPRESSIBLE_PARAMS,
        rho_low=1.0,
        rho_high=2.0,
        E0=1.0,
        markerSize=4,
        shape='box', boost=0.,
    ),
))


if __name__ == '__main__':
    caseMain(hydrostaticCase)
