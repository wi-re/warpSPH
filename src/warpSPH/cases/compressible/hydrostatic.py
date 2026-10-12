"""Hydrostatic equilibrium (2D), compressible.

The script form of this case was `examples/compressible/08-hydrostatic.ipynb`.
A dense square sits in a lighter background at uniform pressure, so the exact
solution is "nothing happens": every velocity should stay at zero. What the
plot actually shows is the spurious surface tension at the density jump, which
is the point of the test.
"""

from __future__ import annotations

from typing import Dict

import torch

from ...caseUtils import buildHydrostaticInitialState
from ...runner import Case, RunContext, caseMain, registerCase
from .common import (COMPRESSIBLE_DEFAULTS, COMPRESSIBLE_PARAMS,
                           compressibleDiagnostics, configureCompressible,
                           paramExtraData)
from ..common.plotting import Field, particlePlot

__all__ = ['hydrostaticCase', 'HYDROSTATIC_FIELDS']


def buildSystem(ctx: RunContext):
    return buildHydrostaticInitialState(
        ctx.param('rho_low'), ctx.param('rho_high'), ctx.spec.nx,
        ctx.config, ctx.schemeConfig, ctx.SimulationState, ctx.SimulationSystem)


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
    return dict(compressibleDiagnostics(ctx, state),
                maxVelocity=torch.linalg.norm(state.state.velocities, dim=-1)
                .max().detach().cpu().item())


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
    params=dict(
        COMPRESSIBLE_PARAMS,
        rho_low=1.0,
        rho_high=2.0,
        E0=1.0,
        markerSize=4,
    ),
))


if __name__ == '__main__':
    caseMain(hydrostaticCase)
