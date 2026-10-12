"""Kidder isentropic compression (1D), compressible.

The script form of this case was
`examples/compressible/03-kidder-isentropic-compression.ipynb`.

Two things make this case unlike the other compressible examples, and both are
carried over as hooks rather than folded into the runner:

* the inner and outer boundary bands are *driven* from the analytic solution
  after every step (`postStep`), not integrated;
* `dt` is re-derived from the state every step (`timestep`), because the shell
  compresses by more than an order of magnitude over the run.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch

from ...caseUtils import buildKidder, buildKidderBCs
from ...runner import Case, RunContext, caseMain, registerCase
from .common import (COMPRESSIBLE_DEFAULTS, COMPRESSIBLE_PARAMS,
                           compressibleDiagnostics, compressibleTimestep,
                           configureCompressible, paramExtraData)
from ..common.plotting import ProfileAxis, profilePlot

__all__ = ['kidderCase', 'kidderStates', 'drawKidder']


def kidderStates(ctx: RunContext) -> Tuple[float, float]:
    """`(rho_inner, s)`, both derived rather than given.

    The paper writes the inner/outer density ratio the other way round; SPHERAL
    uses the flipped version reproduced here, and the two agree on `s` because
    the problem is isentropic.
    """
    gamma = ctx.param('gamma')
    rhoOuter = ctx.param('rho_outer')
    rhoInner = (ctx.param('P_inner') / ctx.param('P_outer')) ** (1 / gamma) * rhoOuter
    entropy = ctx.param('P_outer') / rhoOuter ** gamma
    return rhoInner, entropy


def configureScheme(ctx: RunContext) -> None:
    # nu is the geometry index (1 planar, 2 cylindrical, 3 spherical) and fixes
    # gamma, so gamma is not an independent knob here.
    ctx.spec.params['nu'] = ctx.spec.dim
    ctx.spec.params['gamma'] = 1 + 2 / ctx.spec.dim
    configureCompressible(ctx)


def buildSystem(ctx: RunContext):
    system, solution = buildKidder(
        ctx.config, ctx.schemeConfig, ctx.SimulationState, ctx.SimulationSystem,
        ctx.param('r_inner'), ctx.param('r_outer'),
        ctx.param('P_inner'), ctx.param('P_outer'), ctx.param('rho_outer'),
        ctx.param('nu'), ctx.param('gamma'))
    ctx.scratch['solution'] = solution

    # The run stops just short of the collapse time; past tau the analytic
    # solution is singular and there is nothing left to compare against.
    ctx.spec = ctx.spec.merged(tLimit=ctx.param('tauFraction') * solution.tau)

    band = buildKidderBCs(ctx.schemeConfig, solution, ctx.param('band'))
    ctx.schemeConfig.boundaryConditions.clear()
    ctx.schemeConfig.boundaryConditions.append(band)
    return system


def postStep(ctx: RunContext, state, step: int) -> None:
    """Re-impose the analytic radial velocity on the two boundary bands."""
    solution = ctx.scratch['solution']
    band = ctx.param('band')
    t = float(state.t)
    positions = state.state.positions.detach().cpu().numpy()[:, 0]
    velocities = torch.as_tensor(solution.vr(t, positions),
                                 dtype=state.state.velocities.dtype,
                                 device=state.state.velocities.device)
    state.state.velocities[:band, 0] = velocities[:band]
    state.state.velocities[-band:, 0] = velocities[-band:]


def _shell(ctx: RunContext, state):
    solution = ctx.scratch['solution']
    t = float(state.t)
    return solution, t, solution.rInner(t), solution.rOuter(t)


def _reference(attribute: str):
    def reference(ctx: RunContext, state) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        solution, t, rInner, rOuter = _shell(ctx, state)
        x = np.linspace(rInner, rOuter, 1000)
        return x, getattr(solution, attribute)(t, x)
    return reference


def _shellEdges(ctx: RunContext, state):
    _, _, rInner, rOuter = _shell(ctx, state)
    return [rInner, rOuter]


setupPlot, updatePlot, drawKidder = profilePlot(
    [
        ProfileAxis('densities', 'Density', reference=_reference('rho'),
                    vlines=_shellEdges),
        ProfileAxis('pressures', 'Pressure', reference=_reference('P'),
                    vlines=_shellEdges),
        ProfileAxis('velocities', 'Velocity', component=0, reference=_reference('vr'),
                    vlines=_shellEdges),
    ],
    shape=(1, 3), figsize=(10, 5),
)


def diagnostics(ctx: RunContext, state) -> Dict[str, float]:
    return compressibleDiagnostics(ctx, state)


def extraData(ctx: RunContext, state) -> Dict[str, float]:
    rhoInner, entropy = kidderStates(ctx)
    return dict(paramExtraData(ctx, state), rho_inner=rhoInner, s=entropy)


kidderCase = registerCase(Case(
    name='kidder',
    scheme='CRKSPH',
    description='Kidder isentropic compression (1D), compressible SPH.',
    buildSystem=buildSystem,
    configureScheme=configureScheme,
    diagnostics=diagnostics,
    postStep=postStep,
    timestep=compressibleTimestep,
    setupPlot=setupPlot,
    updatePlot=updatePlot,
    extraData=extraData,
    defaults=dict(
        COMPRESSIBLE_DEFAULTS,
        caseName='03-kidderIsentropicCompression',
        dim=1,
        nx=100,
        L=2.0,
        # tLimit is replaced in buildSystem once the analytic collapse time is
        # known; this is only the placeholder a --config file could override.
        tLimit=1.0,
        plotInterval=100,
        storeInterval=500,
    ),
    params=dict(
        COMPRESSIBLE_PARAMS,
        band=10,
        r_inner=0.9,
        r_outer=1.0,
        P_inner=0.1,
        P_outer=1.0,
        rho_outer=0.01,
        nu=1,
        tauFraction=0.99,
    ),
))


if __name__ == '__main__':
    caseMain(kidderCase)
