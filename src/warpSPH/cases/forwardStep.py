"""Mach 3 wind tunnel with a forward-facing step (2D), compressible.

COMPRESSIBLE_WALLS_PLAN.md case 5 (Emery 1968; Woodward & Colella 1984): a
tunnel of height 1 with a step of height 0.2 whose face is 0.6 downstream of
the inflow; gas at rho = 1.4, p = 1 (c = 1) enters at Mach 3.

Run in the gas's frame, so it needs neither inflow nor outflow: the gas starts
at rest and the step -- a block of wall rows -- moves left at U = 3. The
tunnel floor and ceiling stay put; with free-slip walls (the default
`wallSlip`) a wall's tangential velocity does not enter, so that is the same
tunnel. The gas ahead of the step is `3 tEnd + 0.6` long, and the block
reaches 2.4 behind its face, so at every time the window [face - 0.6,
face + 2.4] is the classic domain, read in the step's frame (x' = x - face +
0.6, gas velocity u' = u + 3).

Diagnostics: `standoff`, the distance from the step face to the bow shock on
the line y = `probeY` (the first row ahead of the face with density past
`shockDensity`), and penetration into the step or the walls.
"""

from __future__ import annotations

from typing import Dict

import torch

from ..runner import Case, RunContext, caseMain, registerCase
from .compressible import (COMPRESSIBLE_DEFAULTS, COMPRESSIBLE_PARAMS,
                           compressibleDiagnostics, compressibleTimestep,
                           configureCompressible, paramExtraData)
from .compressibleWalls import WALL_PARAMS, boxSDF, buildWalledBox, walledDomainBounds
from .plotting import Field, particlePlot

__all__ = ['forwardStepCase']

_FAR = 1e3


def _faceStart(ctx: RunContext) -> float:
    return 0.6 + ctx.param('U') * ctx.param('tEnd')


def _box(ctx: RunContext):
    return (0.0, 0.0), (_faceStart(ctx) + 2.4, 1.0)


def _face(ctx: RunContext, t: float) -> float:
    return _faceStart(ctx) - ctx.param('U') * t


def configureScheme(ctx: RunContext) -> None:
    lo, hi = _box(ctx)
    # nx is per unit length, so the spacing does not depend on how long the tunnel has to be
    ctx.spec = ctx.spec.merged(nx=int(round(ctx.param('cellsPerUnit') * (hi[0] - lo[0]))), L=hi[0] - lo[0],
                               tLimit=ctx.param('tEnd'))
    walledDomainBounds(ctx, lo, hi)
    configureCompressible(ctx)


def _stepSDF(ctx: RunContext, x: torch.Tensor) -> torch.Tensor:
    return boxSDF(x, (_faceStart(ctx), -_FAR), (_FAR, ctx.param('stepHeight')))


def buildSystem(ctx: RunContext):
    rho0, p0, U = ctx.param('rho0'), ctx.param('p0'), ctx.param('U')
    lo, hi = _box(ctx)

    def stateAt(x):
        return (torch.full_like(x[:, 0], rho0), torch.full_like(x[:, 0], p0), torch.zeros_like(x))

    def wallVelocity(x):
        # the step block moves; the tunnel's floor, ceiling and ends do not
        v = torch.zeros_like(x)
        step = (_stepSDF(ctx, x) <= 1e-6) & (x[:, 1] >= 0.0) & (x[:, 0] <= hi[0])
        v[:, 0] = torch.where(step, -U, 0.0).to(x.dtype)
        return v

    return buildWalledBox(ctx, lo, hi, stateAt, solidSDF=lambda x: _stepSDF(ctx, x),
                          wallVelocity=wallVelocity)


def diagnostics(ctx: RunContext, state) -> Dict[str, float]:
    s, t = state.state, float(state.t)
    fluid = s.kinds == 0
    out = compressibleDiagnostics(ctx, state)
    dx, face = ctx.config.dx, _face(ctx, t)
    x, y = s.positions[:, 0], s.positions[:, 1]
    line = fluid & ((y - ctx.param('probeY')).abs() < dx) & (x < face)
    shocked = line & (s.densities > ctx.param('shockDensity'))
    out['standoff'] = (face - x[shocked].min()).item() if shocked.any() else float('nan')
    inStep = fluid & (x > face + 0.5 * dx) & (x < face + 2.4) & (y < ctx.param('stepHeight'))
    lo, hi = _box(ctx)
    out['penetrating'] = (inStep | (fluid & ((y < lo[1]) | (y > hi[1])))).sum().item()
    out['maxVelocity'] = torch.linalg.norm(s.velocities[fluid], dim=-1).max().item()
    return out


FORWARD_STEP_FIELDS = [
    Field('densities', 'Density', colorMap='viridis', gridResolution=1024),
]
setupPlot, updatePlot = particlePlot(FORWARD_STEP_FIELDS, figsize=(16, 3))


forwardStepCase = registerCase(Case(
    name='forwardStep',
    scheme='Monaghan',
    description='Mach 3 forward-facing step, run in the gas frame (2D), compressible SPH.',
    buildSystem=buildSystem,
    configureScheme=configureScheme,
    diagnostics=diagnostics,
    setupPlot=setupPlot,
    updatePlot=updatePlot,
    extraData=paramExtraData,
    timestep=compressibleTimestep,
    extraFields=('internalEnergies', 'supports'),
    defaults=dict(
        COMPRESSIBLE_DEFAULTS,
        caseName='05-forwardStep',
        dim=2,
        nx=1200,
        L=15.0,
        periodic=False,
        supportMode='Gather',
        tLimit=4.0,
        cflFactor=0.2,
        plotInterval=20,
        storeInterval=500,
    ),
    params=dict(
        COMPRESSIBLE_PARAMS,
        gamma=1.4,
        rho0=1.4,
        p0=1.0,
        U=3.0,
        tEnd=4.0,
        stepHeight=0.2,
        cellsPerUnit=80,
        probeY=0.5,
        shockDensity=2.1,
        **WALL_PARAMS,
    ),
))


if __name__ == '__main__':
    caseMain(forwardStepCase)
