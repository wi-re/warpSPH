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

from ..common.parameters import COMPRESSIBLE_CASE_PARAMETERS, COMPRESSIBLE_SCHEMES, SPH_SCHEMES
from ...runner.parameters import Parameter as P

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


def deriveConfiguration(spec):
    return {'nu': spec.dim, 'gamma': 1 + 2 / spec.dim}


def validateConfiguration(spec):
    p = spec.params
    if p['r_inner'] >= p['r_outer']:
        raise ValueError('params.r_inner: must be smaller than params.r_outer')
    if p['P_inner'] >= p['P_outer']:
        raise ValueError('params.P_inner: compression requires P_inner < P_outer')
    if 2 * p['band'] >= spec.nx:
        raise ValueError('params.band: two boundary bands must leave interior particles')


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
    supportedSchemes=SPH_SCHEMES,
    parameterDefinitions={**COMPRESSIBLE_CASE_PARAMETERS,
        'gamma': P('number', 'Isentropic compression adiabatic index.', 'fluid', exclusiveMinimum=1,
                   readOnly=True, derivation='1 + 2 / simulation.dim'),
        'nu': P('integer', 'Geometry index: planar 1, cylindrical 2, spherical 3; not kinematic viscosity.', 'domain',
                choices=(1, 2, 3), readOnly=True, derivation='simulation.dim'),
        'band': P('integer', 'Particles at each shell edge held to the analytical solution.', 'boundary', minimum=1),
        'r_inner': P('number', 'Initial inner shell radius.', 'domain', units='length', exclusiveMinimum=0),
        'r_outer': P('number', 'Initial outer shell radius.', 'domain', units='length', exclusiveMinimum=0),
        'P_inner': P('number', 'Inner pressure.', 'initial', units='pressure', exclusiveMinimum=0),
        'P_outer': P('number', 'Outer pressure.', 'initial', units='pressure', exclusiveMinimum=0),
        'rho_outer': P('number', 'Initial outer density.', 'initial', units='density', exclusiveMinimum=0),
        'tauFraction': P('number', 'Fraction of analytical collapse time to integrate.', 'time', exclusiveMinimum=0, exclusiveMaximum=1),
    },
    simulationParameters={'tLimit': P('number', 'Run duration is derived from tauFraction and the collapse solution during sampling.', 'time', readOnly=True, derivation='params.tauFraction and the analytical collapse time', exclusiveMinimum=0)},
    variants={},
    deriveParameters=deriveConfiguration,
    validateConfiguration=validateConfiguration,
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
