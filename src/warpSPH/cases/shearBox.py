"""Periodic differential-shear box (2D), compressible -- AV_PLAN Phase 5A.

    v_x = v0 sin(2 pi y / L),   v_y = 0,   rho = 1,   P = P0

uniform density and pressure, so the continuum solution is steady (an inviscid shear layer: `div v = 0`,
`(v . grad) v = 0`, no pressure gradient). There is no shock and no compression anywhere, so every bit of
artificial viscosity is spurious, and pairs along the shear approach (`v_ij . x_ij < 0`) as often as they recede:
the minimal setting in which the quadratic AV term acts with nothing to capture (Chen & Nixon 2025's mechanism,
measured here instead of on an accretion disc). `modeAmplitude` is the velocity projected on the initial mode over
`v0`: 1 is exact, below 1 is dissipation; the AV energy split comes from the AV report's Group B rows.
"""

from __future__ import annotations

import math
from typing import Dict

import torch

from ..modules.eos import idealGasEOS
from ..modules.timestep.compressible import computeTimestep
from ..runner import Case, RunContext, caseMain, registerCase
from ..sample.compressible import setupBasicCompressibleInitialState
from .compressible import (COMPRESSIBLE_DEFAULTS, COMPRESSIBLE_PARAMS,
                           compressibleDiagnostics, configureCompressible,
                           paramExtraData)
from .plotting import Field, particlePlot

__all__ = ['shearBoxCase', 'modeAmplitude']


def _boxLength(ctx) -> float:
    d = ctx.config.domain
    return float(d.max[1] - d.min[1])


def _mode(positions, L):
    return torch.sin(2 * math.pi * positions[:, 1] / L)


def buildSystem(ctx: RunContext):
    system = setupBasicCompressibleInitialState(ctx.spec.nx, ctx.config, ctx.schemeConfig,
                                                ctx.SimulationState, ctx.SimulationSystem)
    st = system.state
    L = _boxLength(ctx)
    v = torch.zeros_like(st.positions)
    v[:, 0] = ctx.param('v0') * _mode(st.positions, L)
    P = torch.full_like(st.densities, ctx.param('P0'))
    u = P / ((ctx.schemeConfig.gamma - 1) * st.densities)
    _, u_, P_, c_s = idealGasEOS(A=None, u=u, P=None, rho=st.densities, gamma=ctx.schemeConfig.gamma)
    st.velocities = v
    st.internalEnergies = u_
    st.pressures = P_
    st.soundspeeds = c_s
    ctx.config.dt = computeTimestep(system, ctx.config, ctx.schemeConfig)
    return system


def modeAmplitude(state, v0: float, L: float = 1.0) -> float:
    """`<v_x sin(k y)> / <sin^2(k y)> / v0` over the fluid: the surviving fraction of the initial shear mode."""
    st = state.state if hasattr(state, 'state') else state
    s = _mode(st.positions, L)
    return float((st.velocities[:, 0] * s).sum() / (s * s).sum() / v0)


def diagnostics(ctx: RunContext, state) -> Dict[str, float]:
    row = compressibleDiagnostics(ctx, state)
    row['modeAmplitude'] = modeAmplitude(state, ctx.param('v0'), _boxLength(ctx))
    return row


SHEAR_FIELDS = [
    Field('velocities', 'velocities', colorMap='RdBu', colorMapKind='diverging', mapping='x'),
    Field('densities', 'densities', colorMap='viridis'),
]

setupPlot, updatePlot = particlePlot(SHEAR_FIELDS)


shearBoxCase = registerCase(Case(
    name='shearBox',
    scheme='Monaghan',
    description='Periodic differential-shear box (2D), compressible: steady shear, no shock (AV_PLAN Phase 5A).',
    buildSystem=buildSystem,
    configureScheme=configureCompressible,
    diagnostics=diagnostics,
    setupPlot=setupPlot,
    updatePlot=updatePlot,
    extraData=paramExtraData,
    defaults=dict(
        COMPRESSIBLE_DEFAULTS,
        caseName='shearBox',
        dim=2,
        nx=64,
        L=1.0,
        tLimit=2.0,
        plotInterval=1,
        storeInterval=500,
    ),
    params=dict(COMPRESSIBLE_PARAMS, markerSize=4, v0=0.5, P0=1.0),
))


if __name__ == '__main__':
    caseMain(shearBoxCase)
