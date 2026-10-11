"""Noh implosion (1D), compressible.

The script form of this case was `examples/compressible/04-noh-implosion.ipynb`.
Uniform cold gas converges on the origin at `v_s`; the exact post-shock state is
`rho_s = rho0 ((gamma+1)/(gamma-1))^dim`, drawn as the reference line.
"""

from __future__ import annotations

from typing import Dict

from ..caseUtils import sampleNoh1D
from ..configurations.meshless import MeshlessConfig
from ..runner import Case, RunContext, caseMain, registerCase
from .compressible import (COMPRESSIBLE_DEFAULTS, COMPRESSIBLE_PARAMS,
                           compressibleDiagnostics, configureCompressible,
                           paramExtraData)
from .plotting import ProfileAxis, profilePlot

__all__ = ['nohCase', 'shockState', 'drawNoh']


def shockState(ctx: RunContext):
    """`(rho_s, P_s)`, the exact post-shock density and pressure."""
    gamma = ctx.param('gamma')
    rhoShock = ctx.param('rho0') * ((gamma + 1) / (gamma - 1)) ** ctx.spec.dim
    return rhoShock, ctx.param('v_s') * rhoShock


def buildSystem(ctx: RunContext):
    if isinstance(ctx.schemeConfig, MeshlessConfig):
        from ..caseUtils.meshless import toMeshlessSystem
        from ..systems.compressibleMonaghan import CompressibleState, CompressibleSystem
        comp = sampleNoh1D(ctx.spec.nx, ctx.config, ctx.schemeConfig,
                           CompressibleState, CompressibleSystem)
        return toMeshlessSystem(comp, ctx.config, ctx.schemeConfig)
    return sampleNoh1D(ctx.spec.nx, ctx.config, ctx.schemeConfig,
                       ctx.SimulationState, ctx.SimulationSystem)


def _shockFront(ctx: RunContext, state):
    """The two symmetric shock fronts, at +/- v_s t."""
    front = ctx.param('v_s') * float(state.t)
    return [front, -front]


setupPlot, updatePlot, drawNoh = profilePlot(
    [
        ProfileAxis('densities', 'Density',
                    hlines=lambda ctx, state: [shockState(ctx)[0]]),
        ProfileAxis('pressures', 'Pressure',
                    hlines=lambda ctx, state: [shockState(ctx)[1]],
                    vlines=_shockFront),
        ProfileAxis('velocities', 'Velocity', component=0),
    ],
    shape=(1, 3), figsize=(10, 5), xlim=(0, 1),
)


def diagnostics(ctx: RunContext, state) -> Dict[str, float]:
    return compressibleDiagnostics(ctx, state)


def extraData(ctx: RunContext, state) -> Dict[str, float]:
    rhoShock, pressureShock = shockState(ctx)
    return dict(paramExtraData(ctx, state), rho_s=rhoShock, P_s=pressureShock)


nohCase = registerCase(Case(
    name='noh',
    scheme='CRKSPH',
    description='Noh implosion (1D), compressible SPH.',
    buildSystem=buildSystem,
    configureScheme=configureCompressible,
    diagnostics=diagnostics,
    setupPlot=setupPlot,
    updatePlot=updatePlot,
    extraData=extraData,
    defaults=dict(
        COMPRESSIBLE_DEFAULTS,
        caseName='04-nohImplosion',
        dim=1,
        nx=200,
        L=2.0,
        tLimit=0.6,
        # `sampleNoh1D` does not set a CFL dt of its own, so this run needs an
        # explicit one -- the notebook's value.
        dt=1e-4,
        plotInterval=25,
        storeInterval=500,
    ),
    params=dict(
        COMPRESSIBLE_PARAMS,
        v_s=1 / 3,
    ),
))


if __name__ == '__main__':
    caseMain(nohCase)
