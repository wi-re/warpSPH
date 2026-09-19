"""Sod shock tube (1D), compressible.

The script form of this case is `examples/compressible/01-sod/sod_1d.py`;
everything below the hooks is now `warpSPH.runner`.
"""

from __future__ import annotations

import os
from typing import Any, Dict

import torch

from ..caseUtils import buildSod1D, plotSod, plotSod_, sodInitialState
from ..enumTypes import AdaptiveSupportScheme, ViscositySwitch
from ..runner import Case, RunContext, caseMain, registerCase, resolveEnum
from .plotting import openWindow, pumpEvents

__all__ = ['sodCase', 'states']


def states(ctx: RunContext):
    """The left/right Riemann states this run was configured with."""
    return (
        sodInitialState(ctx.param('left_pressure'), ctx.param('left_rho'), ctx.param('left_velocity')),
        sodInitialState(ctx.param('right_pressure'), ctx.param('right_rho'), ctx.param('right_velocity')),
    )


def configureScheme(ctx: RunContext) -> None:
    left, _ = states(ctx)
    ctx.schemeConfig.gamma = ctx.param('gamma')
    ctx.schemeConfig.rho0 = left.rho
    ctx.schemeConfig.viscositySwitchParams.scheme = resolveEnum(
        ViscositySwitch, ctx.param('viscositySwitch'))
    ctx.schemeConfig.viscositySwitchParams.alpha_min = ctx.param(
        'alpha_min', ctx.schemeConfig.viscositySwitchParams.alpha_min)
    ctx.schemeConfig.viscositySwitchParams.alpha_max = ctx.param(
        'alpha_max', ctx.schemeConfig.viscositySwitchParams.alpha_max)
    ctx.schemeConfig.adaptiveSupportScheme = resolveEnum(
        AdaptiveSupportScheme, ctx.param('adaptiveSupportScheme'))
    ctx.schemeConfig.adaptiveSupportCorrections = ctx.param('adaptiveSupportCorrections')


def buildSystem(ctx: RunContext):
    left, right = states(ctx)
    return buildSod1D(
        ctx.SimulationSystem, ctx.SimulationState,
        ctx.param('samplingRatio'),
        left, right,
        ctx.param('gamma'), ctx.config,
        ctx.param('smoothIC'),
        adaptiveSupportScheme=ctx.schemeConfig.adaptiveSupportScheme,
    )


def diagnostics(ctx: RunContext, state) -> Dict[str, float]:
    """Kinetic/thermal/total energy -- total energy is the conserved quantity."""
    particles = state.state
    kinetic = 0.5 * (torch.linalg.norm(particles.velocities, dim=-1) ** 2 * particles.masses).sum()
    thermal = (particles.internalEnergies * particles.masses).sum()
    return {
        'kineticEnergy': kinetic.detach().cpu().item(),
        'thermalEnergy': thermal.detach().cpu().item(),
        'totalEnergy': (kinetic + thermal).detach().cpu().item(),
    }


def setupPlot(ctx: RunContext, state, scatter: bool = False):
    """The six Sod profile panels. `scatter` is what the 2D/3D variants need:
    many particles share an x there, so a line through them is meaningless."""
    left, right = states(ctx)
    fig, axis = plotSod(state.state, ctx.config, ctx.schemeConfig, ctx.config.domain,
                        ctx.param('gamma'), left, right,
                        plotReference=True, plotLabels=False, scatter=scatter, t_=state.t)
    if ctx.imagePath:
        fig.savefig(os.path.join(ctx.imagePath, 'frame_0000000.png'))
    openWindow(ctx, (fig, axis))
    return (fig, axis)


def updatePlot(ctx: RunContext, state, handle, step: int) -> None:
    fig, axis = handle
    left, right = states(ctx)
    for ax in axis.flatten():
        ax.clear()
    plotSod_(fig, axis, state.state, ctx.config, ctx.schemeConfig, ctx.config.domain,
             ctx.param('gamma'), left, right,
             plotReference=True, plotLabels=False, scatter=True, t_=state.t)
    if ctx.imagePath:
        fig.savefig(os.path.join(ctx.imagePath, f'frame_{step:07d}.png'))
    pumpEvents(handle)


def extraData(ctx: RunContext, state) -> Dict[str, Any]:
    return {k: ctx.param(k) for k in sodCase.params}


sodCase = registerCase(Case(
    name='sod',
    scheme='CompSPH',
    description='Sod shock tube (1D), compressible SPH.',
    buildSystem=buildSystem,
    configureScheme=configureScheme,
    diagnostics=diagnostics,
    setupPlot=setupPlot,
    updatePlot=updatePlot,
    extraData=extraData,
    # internalEnergies is the core EOS state variable for an ideal-gas scheme
    # (pressures/soundspeeds/entropies are cheap to recompute from it via
    # idealGasEOS); supports must be re-written every frame too because Sod's
    # default adaptiveSupportScheme keeps them drifting from the IC snapshot.
    extraFields=('internalEnergies', 'supports'),
    defaults=dict(
        caseName='01-sodShockTube',
        dim=1,
        nx=800,
        L=2.0,
        n_h=4.0,
        periodic=True,
        kernel='B7',
        integrationScheme='rungeKutta2',
        supportMode='Gather',
        gradientMode='Difference',
        laplacianMode='Brookshaw',
        samplingScheme='regular',
        tLimit=0.15,
        dt=1e-3,
        adaptiveDt=True,
        cflFactor=0.3,
        plotInterval=10,
        storeInterval=50,
    ),
    params=dict(
        gamma=5 / 3,
        smoothIC=False,
        samplingRatio=4,
        left_rho=1.0,
        left_pressure=1.0,
        left_velocity=0.0,
        right_rho=0.25,
        right_pressure=0.1795,
        right_velocity=0.0,
        viscositySwitch='NoneSwitch',
        adaptiveSupportScheme='Owen',
        adaptiveSupportCorrections=False,
    ),
))


if __name__ == '__main__':
    caseMain(sodCase)
