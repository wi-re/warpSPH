"""Linear acoustic wave (1D), compressible.

The script form of this case was `examples/compressible/02-linear-wave.ipynb`.
A small-amplitude sinusoidal perturbation advects at the sound speed; the run
covers one acoustic crossing of the box, which is what makes the analytic
overlay in `plotState` a meaningful check.
"""

from __future__ import annotations

import os
from typing import Dict

from ..caseUtils import plotState, sampleLinearWave
from ..enumTypes import AdaptiveSupportScheme
from ..runner import Case, RunContext, caseMain, registerCase, resolveEnum
from .compressible import (COMPRESSIBLE_DEFAULTS, COMPRESSIBLE_PARAMS,
                           compressibleDiagnostics, configureCompressible,
                           paramExtraData)
from .plotting import openWindow, pumpEvents

__all__ = ['linearWaveCase']


def buildSystem(ctx: RunContext):
    return sampleLinearWave(
        ctx.spec.nx, ctx.config, ctx.schemeConfig, ctx.SimulationState, ctx.SimulationSystem,
        ctx.param('A'), ctx.param('lamda'), ctx.param('c_s'),
        ctx.param('rho0'), ctx.param('gamma'),
        nIters=ctx.param('nIters'),
        supportScheme=resolveEnum(AdaptiveSupportScheme, ctx.param('adaptiveSupportScheme')),
    )


def initialConditions(ctx: RunContext, system) -> None:
    # `plotState` overlays the analytic wave on the *initial* sampling, so the
    # sampled system has to outlive `buildSystem`.
    ctx.scratch['system'] = system

    # The notebook picked its dt as `t_limit / 1000` so exactly 1000 frames
    # cover one crossing; keeping that here means `--dt` still overrides it.
    if ctx.spec.dt is None:
        soundSpeed = system.state.soundspeeds.min().detach().cpu().item()
        ctx.config.dt = (1.0 / soundSpeed) / ctx.param('stepsPerCrossing')


def setupPlot(ctx: RunContext, state):
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(1, 3, figsize=(10, 5), squeeze=False)
    handle = (fig, axis)
    _draw(ctx, state, handle)
    openWindow(ctx, handle)
    return handle


def updatePlot(ctx: RunContext, state, handle, step: int) -> None:
    _draw(ctx, state, handle, step)
    pumpEvents(handle)


def _draw(ctx: RunContext, state, handle, step: int = 0) -> None:
    fig, axis = handle
    plotState(fig, axis, state, ctx.scratch.get('system', state), ctx.config,
              ctx.schemeConfig, ctx.param('rho0'), ctx.param('A'),
              ctx.param('lamda'), ctx.param('c_s'))
    fig.tight_layout()
    if ctx.imagePath:
        fig.savefig(os.path.join(ctx.imagePath, f'frame_{step:07d}.png'))


def diagnostics(ctx: RunContext, state) -> Dict[str, float]:
    return compressibleDiagnostics(ctx, state)


linearWaveCase = registerCase(Case(
    name='linearWave',
    scheme='CRKSPH',
    description='Linear acoustic wave (1D), compressible SPH.',
    buildSystem=buildSystem,
    configureScheme=configureCompressible,
    initialConditions=initialConditions,
    diagnostics=diagnostics,
    setupPlot=setupPlot,
    updatePlot=updatePlot,
    extraData=paramExtraData,
    defaults=dict(
        COMPRESSIBLE_DEFAULTS,
        caseName='02-linearWave',
        dim=1,
        nx=200,
        L=1.0,
        # Wendland4 and a Gather support are this case's departure from the
        # shared compressible block; the wave is smooth enough not to need B7.
        kernel='Wendland4',
        supportMode='Gather',
        tLimit=1.0,
        plotInterval=10,
        storeInterval=50,
    ),
    params=dict(
        COMPRESSIBLE_PARAMS,
        A=1e-6,
        lamda=1.0,
        c_s=1.0,
        E0=1.0,
        nIters=16,
        stepsPerCrossing=1000,
    ),
))


if __name__ == '__main__':
    caseMain(linearWaveCase)
