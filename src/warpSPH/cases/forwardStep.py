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
0.6, gas velocity u' = u + 3). The block itself reaches past the tunnel's end
by its whole travel (rows kept outside the box, `buildWalledBox(keepSolid=...)`),
so it slides under the tunnel's right wall and never leaves a vacuum behind.

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
from .plotting import _save, figureTitle, openWindow, pumpEvents

__all__ = ['forwardStepCase']

_FAR = 1e3


def _faceStart(ctx: RunContext) -> float:
    return 0.6 + ctx.param('U') * ctx.param('tEnd')


def _box(ctx: RunContext):
    return (0.0, 0.0), (_faceStart(ctx) + 2.4, 1.0)


def _face(ctx: RunContext, t: float) -> float:
    return _faceStart(ctx) - ctx.param('U') * t


def _blockEnd(ctx: RunContext) -> float:
    """The step block reaches this far right: past the tunnel's end by its whole travel
    plus the wall depth, so its back end never enters the tunnel (no vacuum behind it)."""
    dx = 1.0 / ctx.param('cellsPerUnit')
    return _box(ctx)[1][0] + ctx.param('U') * ctx.param('tEnd') + (ctx.param('wallLayers') + 2) * dx


def configureScheme(ctx: RunContext) -> None:
    lo, hi = _box(ctx)
    # nx is per unit length, so the spacing does not depend on how long the tunnel has to be
    ctx.spec = ctx.spec.merged(nx=int(round(ctx.param('cellsPerUnit') * (hi[0] - lo[0]))), L=hi[0] - lo[0],
                               tLimit=ctx.param('tEnd'))
    walledDomainBounds(ctx, lo, hi, extendHi=(_blockEnd(ctx), hi[1]))
    configureCompressible(ctx)


def _stepSDF(ctx: RunContext, x: torch.Tensor) -> torch.Tensor:
    return boxSDF(x, (_faceStart(ctx), -_FAR), (_blockEnd(ctx), ctx.param('stepHeight')))


def _keepStep(ctx: RunContext, x: torch.Tensor) -> torch.Tensor:
    """Rows of the block's top band beyond the tunnel's end: they slide in later."""
    dx = 1.0 / ctx.param('cellsPerUnit')
    depth = ctx.param('wallLayers') * dx
    return ((_stepSDF(ctx, x) < 0) & (x[:, 0] > _box(ctx)[1][0])
            & (x[:, 1] > ctx.param('stepHeight') - depth) & (x[:, 1] >= 0.0))


def buildSystem(ctx: RunContext):
    rho0, p0, U = ctx.param('rho0'), ctx.param('p0'), ctx.param('U')
    lo, hi = _box(ctx)

    def stateAt(x):
        return (torch.full_like(x[:, 0], rho0), torch.full_like(x[:, 0], p0), torch.zeros_like(x))

    def wallVelocity(x):
        # the step block moves; the tunnel's floor, ceiling and ends do not
        v = torch.zeros_like(x)
        step = (_stepSDF(ctx, x) <= 1e-6) & (x[:, 1] >= 0.0)
        v[:, 0] = torch.where(step, -U, 0.0).to(x.dtype)
        return v

    return buildWalledBox(ctx, lo, hi, stateAt, solidSDF=lambda x: _stepSDF(ctx, x),
                          wallVelocity=wallVelocity, extendHi=(_blockEnd(ctx), hi[1]),
                          keepSolid=lambda x: _keepStep(ctx, x))


def diagnostics(ctx: RunContext, state) -> Dict[str, float]:
    s, t = state.state, float(state.t)
    fluid = s.kinds == 0
    out = compressibleDiagnostics(ctx, state)
    dx, face = ctx.config.dx, _face(ctx, t)
    x, y = s.positions[:, 0], s.positions[:, 1]
    line = fluid & ((y - ctx.param('probeY')).abs() < dx) & (x < face)
    shocked = line & (s.densities > ctx.param('shockDensity'))
    out['standoff'] = (face - x[shocked].min()).item() if shocked.any() else float('nan')
    # depth into the step block (from its top or its face) and past the floor or ceiling, in spacings
    h0 = ctx.param('stepHeight')
    inStep = fluid & (x > face) & (x < face + 2.4) & (y < h0)
    depth = torch.minimum(x - face, h0 - y)
    lo, hi = _box(ctx)
    out['inStep'] = inStep.sum().item()
    out['stepDepth'] = (depth[inStep].max() / dx).item() if inStep.any() else 0.0
    out['outsideTunnel'] = (fluid & ((y < lo[1]) | (y > hi[1]))).sum().item()
    out['maxVelocity'] = torch.linalg.norm(s.velocities[fluid], dim=-1).max().item()
    return out


def _window(ctx: RunContext, state):
    """Fluid rows in the classic domain, in the step's frame (x' = x - face + 0.6)."""
    s = state.state
    x = s.positions[:, 0] - _face(ctx, float(state.t)) + 0.6
    keep = (s.kinds == 0) & (x > 0.0) & (x < 3.0)
    return x[keep], s.positions[keep, 1], keep


def _drawWindow(ctx: RunContext, state, handle) -> None:
    fig, axes = handle
    x, y, keep = _window(ctx, state)
    xs, ys = x.detach().cpu().numpy(), y.detach().cpu().numpy()
    for ax, (name, title, cmap) in zip(axes, (('densities', 'Density', 'viridis'),
                                              ('pressures', 'Pressure', 'inferno'))):
        ax.clear()
        values = getattr(state.state, name)[keep].detach().cpu().numpy()
        ax.scatter(xs, ys, c=values, s=1.2, cmap=cmap, linewidths=0)
        ax.fill_between([0.6, 3.0], 0.0, ctx.param('stepHeight'), color='0.6')
        ax.set_xlim(0, 3)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_title(f'{title} (step frame)')
    fig.suptitle(figureTitle(ctx, state))
    fig.tight_layout()


def setupPlot(ctx: RunContext, state):
    """A window on the classic domain in the step's frame (matplotlib): the run's
    own domain is ~(3 + 3 tEnd + 3 tEnd) long, mostly gas waiting or the block's travel."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(9, 6))
    ctx.scratch['plotBackend'] = 'matplotlib'
    handle = (fig, axes)
    _drawWindow(ctx, state, handle)
    _save(ctx, fig, 0, 150)
    openWindow(ctx, handle)
    return handle


def updatePlot(ctx: RunContext, state, handle, step: int) -> None:
    _drawWindow(ctx, state, handle)
    _save(ctx, handle[0], step, 150)
    pumpEvents(handle)


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
