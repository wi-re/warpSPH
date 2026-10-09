"""A planar shock reflecting off a solid wall (1D), compressible.

COMPRESSIBLE_WALLS_PLAN.md case 3. A Mach-`mach` shock runs into gas at rest
(`rho0`, `p0`) towards the right wall; the post-shock gas (Rankine-Hugoniot
state 2) fills the box to the left of it and moves at `u2`. The wall reflects
it into state 3 (gas at rest), whose pressure and density are exact:

    p3/p2 = ((3g-1) p2/p0 - (g-1)) / ((g-1) p2/p0 + (g+1))
    rho3/rho2 = ((g+1) p3/p2 + (g-1)) / ((g-1) p3/p2 + (g+1))
    reflected front speed = rho2 u2 / (rho3 - rho2)

The left wall is fixed too, so the run is only valid until the rarefaction it
sheds reaches the reflected front.
"""

from __future__ import annotations

from typing import Dict

import torch

from ..runner import Case, RunContext, caseMain, registerCase
from .compressible import (COMPRESSIBLE_DEFAULTS, COMPRESSIBLE_PARAMS,
                           compressibleTimestep, configureCompressible,
                           paramExtraData)
from .compressibleWalls import WALL_PARAMS, buildWalledSystem
from .plotting import ProfileAxis, profilePlot

__all__ = ['shockReflectionCase', 'reflectionStates']


def reflectionStates(ctx: RunContext) -> Dict[str, float]:
    """Exact states 1 (ahead), 2 (post-shock) and 3 (reflected), and speeds."""
    g, M = ctx.param('gamma'), ctx.param('mach')
    rho1, p1 = ctx.param('rho0'), ctx.param('p0')
    c1 = (g * p1 / rho1) ** 0.5
    p2 = p1 * (1 + 2 * g / (g + 1) * (M * M - 1))
    rho2 = rho1 * (g + 1) * M * M / ((g - 1) * M * M + 2)
    u2 = 2 / (g + 1) * c1 * (M - 1 / M)
    r = ((3 * g - 1) * p2 / p1 - (g - 1)) / ((g - 1) * p2 / p1 + (g + 1))
    p3 = p2 * r
    rho3 = rho2 * ((g + 1) * r + (g - 1)) / ((g - 1) * r + (g + 1))
    return dict(rho1=rho1, p1=p1, rho2=rho2, p2=p2, u2=u2, rho3=rho3, p3=p3,
                shockSpeed=M * c1, reflectedSpeed=rho2 * u2 / (rho3 - rho2))


def _wallPosition(ctx: RunContext) -> float:
    return 0.5 * ctx.spec.L - ctx.param('wallLayers') * ctx.spec.L / ctx.spec.nx


def _hitTime(ctx: RunContext) -> float:
    return (_wallPosition(ctx) - ctx.param('shockStart')) / reflectionStates(ctx)['shockSpeed']


def buildSystem(ctx: RunContext):
    e, x0 = reflectionStates(ctx), ctx.param('shockStart')

    def stateAt(x):
        post = x < x0
        rho = torch.where(post, e['rho2'], e['rho1']).to(x.dtype)
        p = torch.where(post, e['p2'], e['p1']).to(x.dtype)
        v = torch.where(post, e['u2'], 0.0).to(x.dtype)
        return rho, p, v

    return buildWalledSystem(ctx, stateAt)


def _frontX(ctx: RunContext, s):
    """The reflected front: scanning from the wall, the first row below the p2/p3
    midpoint after the first row above it (the wall-side density dip is skipped)."""
    e = reflectionStates(ctx)
    fluid = s.kinds == 0
    order = torch.argsort(s.positions[fluid, 0])
    x, p = s.positions[fluid, 0][order], s.pressures[fluid][order]
    below = torch.flip(p < 0.5 * (e['p2'] + e['p3']), [0])
    firstAbove = torch.argmax((~below).to(torch.int32))
    after = torch.arange(below.numel(), device=below.device) >= firstAbove
    return x[x.numel() - 1 - torch.argmax((below & after).to(torch.int32))]


def _reflectedFront(ctx: RunContext, state):
    dt = float(state.t) - _hitTime(ctx)
    if dt <= 0:
        return []
    return [_wallPosition(ctx) - reflectionStates(ctx)['reflectedSpeed'] * dt]


def diagnostics(ctx: RunContext, state) -> Dict[str, float]:
    s, e = state.state, reflectionStates(ctx)
    fluid = s.kinds == 0
    m, v = s.masses[fluid], s.velocities[fluid]
    kinetic = 0.5 * (m * (v ** 2).sum(-1)).sum()
    thermal = (m * s.internalEnergies[fluid]).sum()
    out = dict(kineticEnergy=kinetic, thermalEnergy=thermal, totalEnergy=kinetic + thermal)
    # state 3 plateau: fluid between half the reflected front's travel and 0.03 off the wall
    # (the last rows are the wall-side boundary layer); NaN until the front has travelled that far
    travel = e['reflectedSpeed'] * max(float(state.t) - _hitTime(ctx), 0.0)
    x = s.positions[:, 0]
    band = fluid & (x > _wallPosition(ctx) - 0.5 * travel) & (x < _wallPosition(ctx) - 0.03)
    nan = torch.full((), float('nan'), dtype=s.pressures.dtype, device=s.pressures.device)
    out['plateauPressureRatio'] = s.pressures[band].median() / e['p3'] if travel > 0.1 else nan
    out['plateauDensityRatio'] = s.densities[band].median() / e['rho3'] if travel > 0.1 else nan
    out['peakPressureRatio'] = s.pressures[fluid].max() / e['p3']
    out['peakDensityRatio'] = s.densities[fluid].max() / e['rho3']
    out['penetrating'] = (s.positions[fluid, 0] > _wallPosition(ctx)).sum()
    out['escaped'] = (s.positions[fluid, 0] > 0.5 * ctx.spec.L).sum()
    out['frontError'] = _frontX(ctx, s) - (_wallPosition(ctx) - travel)
    return {k: t.detach().cpu().item() for k, t in out.items()}


setupPlot, updatePlot, drawShockReflection = profilePlot(
    [
        ProfileAxis('densities', 'Density', hlines=lambda ctx, state: [reflectionStates(ctx)['rho2'],
                                                                      reflectionStates(ctx)['rho3']],
                    vlines=_reflectedFront),
        ProfileAxis('pressures', 'Pressure', hlines=lambda ctx, state: [reflectionStates(ctx)['p2'],
                                                                       reflectionStates(ctx)['p3']],
                    vlines=_reflectedFront),
        ProfileAxis('velocities', 'Velocity', component=0),
    ],
    shape=(1, 3), figsize=(12, 4), xlim=(-1.0, 1.0),
)


shockReflectionCase = registerCase(Case(
    name='shockReflection',
    scheme='Monaghan',
    description='Mach-M shock reflecting off a solid wall (1D), compressible SPH.',
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
        caseName='03-shockReflection',
        dim=1,
        nx=400,
        L=2.0,
        periodic=False,
        supportMode='Gather',
        tLimit=0.5,
        plotInterval=25,
        storeInterval=100,
    ),
    params=dict(
        COMPRESSIBLE_PARAMS,
        gamma=1.4,
        rho0=1.0,
        p0=1.0,
        mach=2.0,
        shockStart=0.2,
        **WALL_PARAMS,
    ),
))


if __name__ == '__main__':
    caseMain(shockReflectionCase)
