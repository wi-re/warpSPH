"""A piston driving into, or withdrawing from, gas at rest (1D), compressible.

COMPRESSIBLE_WALLS_PLAN.md case 2: the moving wall. The left wall moves at
`pistonSpeed` from t = 0 (the impulsive start is the classic problem, not a
transient to avoid) into gas at rest (`rho0`, `p0`). The exact solution is
uniform between the piston face and the wave it sends:

* `pistonSpeed > 0`, a shock: the post-shock state is the Rankine-Hugoniot
  state with gas velocity `u_p`, shock Mach number from
  `u_p = 2 c0 / (g+1) (M - 1/M)`.
* `pistonSpeed < 0`, a rarefaction: the face state is isentropic,
  `p* = p0 (1 + (g-1)/2 u_p / c0)^(2g/(g-1))`; the fan head runs into the gas
  at `c0`, its tail at `u_p + c*`. The piston needs room to withdraw into, so
  that many empty lattice cells are left behind it.

`frontError` is the measured half-pressure point (between p0 and the face
state) against the exact one: the shock itself, or a point inside the fan.
Either way the fluid's energy changes by exactly the piston's work
`p_face u_p t`; `energyRatio` is the measured change over that.
"""

from __future__ import annotations

import math
from typing import Dict

import torch

from ..runner import Case, RunContext, caseMain, registerCase
from .compressible import (COMPRESSIBLE_DEFAULTS, COMPRESSIBLE_PARAMS,
                           compressibleTimestep, configureCompressible,
                           paramExtraData)
from .compressibleWalls import WALL_PARAMS, buildWalledSystem, fluidEnergies
from .plotting import ProfileAxis, profilePlot

__all__ = ['pistonCase', 'pistonStates']


def pistonStates(ctx: RunContext) -> Dict[str, float]:
    """The exact face state (rho, p, velocity) and the wave speeds."""
    g, up = ctx.param('gamma'), ctx.param('pistonSpeed')
    rho0, p0 = ctx.param('rho0'), ctx.param('p0')
    c0 = math.sqrt(g * p0 / rho0)
    if up >= 0:
        k = up * (g + 1) / (2 * c0)
        M = 0.5 * (k + math.sqrt(k * k + 4))
        p = p0 * (1 + 2 * g / (g + 1) * (M * M - 1))
        rho = rho0 * (g + 1) * M * M / ((g - 1) * M * M + 2)
        return dict(rho=rho, p=p, u=up, front=M * c0, tail=M * c0, mid=M * c0, c0=c0)
    ratio = max(1 + 0.5 * (g - 1) * up / c0, 0.0) ** (2 * g / (g - 1))
    p, rho = p0 * ratio, rho0 * ratio ** (1 / g)
    # inside the fan x/t = c + u with u = 2 (c - c0) / (g-1): where p is half-way to p*
    cMid = c0 * (0.5 * (1 + ratio)) ** ((g - 1) / (2 * g))
    mid = cMid + 2 * (cMid - c0) / (g - 1)
    return dict(rho=rho, p=p, u=up, front=c0, tail=up + math.sqrt(g * p / rho), mid=mid, c0=c0)


def _gap(ctx: RunContext) -> int:
    """Empty cells behind a withdrawing piston: its travel plus a margin."""
    up = ctx.param('pistonSpeed')
    if up >= 0:
        return 0
    return int(math.ceil(-up * ctx.spec.tLimit * ctx.spec.nx / ctx.spec.L)) + 4


def _faceStart(ctx: RunContext) -> float:
    dx = ctx.spec.L / ctx.spec.nx
    return -0.5 * ctx.spec.L + (_gap(ctx) + ctx.param('wallLayers')) * dx


def _face(ctx: RunContext, t: float) -> float:
    return _faceStart(ctx) + ctx.param('pistonSpeed') * t


def buildSystem(ctx: RunContext):
    rho0, p0 = ctx.param('rho0'), ctx.param('p0')
    return buildWalledSystem(
        ctx, lambda x: (rho0 * torch.ones_like(x), p0 * torch.ones_like(x), torch.zeros_like(x)),
        wallVelocity=(ctx.param('pistonSpeed'), 0.0), gaps=(_gap(ctx), 0))


def _lines(ctx: RunContext, state):
    e, t = pistonStates(ctx), float(state.t)
    out = [_face(ctx, t), _faceStart(ctx) + e['front'] * t]
    if ctx.param('pistonSpeed') < 0:
        out.append(_faceStart(ctx) + e['tail'] * t)
    return out


def _frontX(ctx: RunContext, s):
    """The wave's half-pressure point: the rightmost fluid row whose pressure is
    past the midpoint between the face state and p0 (NaN before the wave forms)."""
    e, p0 = pistonStates(ctx), ctx.param('p0')
    fluid = s.kinds == 0
    x, p = s.positions[fluid, 0], s.pressures[fluid]
    moved = (p - p0).abs() > 0.5 * abs(e['p'] - p0)
    front = torch.where(moved, x, torch.full_like(x, -math.inf)).max()
    return torch.where(torch.isfinite(front), front, torch.full_like(front, math.nan))


def diagnostics(ctx: RunContext, state) -> Dict[str, float]:
    s, e, t = state.state, pistonStates(ctx), float(state.t)
    fluid = s.kinds == 0
    out = fluidEnergies(s)
    initial = ctx.scratch.setdefault('initialEnergy', out['totalEnergy'].item())
    work = e['p'] * e['u'] * t
    nan = torch.full((), float('nan'), dtype=s.pressures.dtype, device=s.pressures.device)
    out['energyRatio'] = (out['totalEnergy'] - initial) / work if abs(work) > 1e-6 else nan

    # the uniform band: from 0.03 off the face to half-way to the wave's tail
    face = _face(ctx, t)
    tail = _faceStart(ctx) + e['tail'] * t
    x = s.positions[:, 0]
    band = fluid & (x > face + 0.03) & (x < face + 0.5 * (tail - face))
    enough = band.sum() > 4
    out['plateauPressureRatio'] = s.pressures[band].median() / e['p'] if enough else nan
    out['plateauDensityRatio'] = s.densities[band].median() / e['rho'] if enough else nan
    out['plateauVelocityError'] = (s.velocities[band, 0].median() - e['u']) / e['c0'] if enough else nan
    out['frontError'] = _frontX(ctx, s) - (_faceStart(ctx) + e['mid'] * t)
    out['penetrating'] = (x[fluid] < face).sum()
    return {k: v.detach().cpu().item() for k, v in out.items()}


setupPlot, updatePlot, drawPiston = profilePlot(
    [
        ProfileAxis('densities', 'Density', vlines=_lines,
                    hlines=lambda ctx, state: [ctx.param('rho0'), pistonStates(ctx)['rho']]),
        ProfileAxis('pressures', 'Pressure', vlines=_lines,
                    hlines=lambda ctx, state: [ctx.param('p0'), pistonStates(ctx)['p']]),
        ProfileAxis('velocities', 'Velocity', component=0, vlines=_lines,
                    hlines=lambda ctx, state: [0.0, pistonStates(ctx)['u']]),
    ],
    shape=(1, 3), figsize=(12, 4), xlim=(-1.0, 1.0),
)


pistonCase = registerCase(Case(
    name='piston',
    scheme='Monaghan',
    description='Piston driving into / withdrawing from gas at rest (1D), compressible SPH.',
    buildSystem=buildSystem,
    configureScheme=configureCompressible,
    diagnostics=diagnostics,
    setupPlot=setupPlot,
    updatePlot=updatePlot,
    extraData=paramExtraData,
    timestep=compressibleTimestep,
    extraFields=('internalEnergies', 'supports'),
    defaults=dict(
        COMPRESSIBLE_DEFAULTS,
        caseName='02-piston',
        dim=1,
        nx=400,
        L=2.0,
        periodic=False,
        supportMode='Gather',
        tLimit=0.6,
        plotInterval=25,
        storeInterval=100,
    ),
    params=dict(
        COMPRESSIBLE_PARAMS,
        gamma=1.4,
        rho0=1.0,
        p0=1.0,
        pistonSpeed=1.0,
        **WALL_PARAMS,
    ),
))


if __name__ == '__main__':
    caseMain(pistonCase)
