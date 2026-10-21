"""Spheral two-gas Sod ICs with an exact profile and distant periodic seam."""
import copy
from dataclasses import replace
import torch

from ...configurations import EOSConfig, MaterialConfig
from ...runner import Case, registerCase, caseMain
from ...runner.parameters import Parameter as P
from ...utils.domain import DomainDescription
from ...caseUtils.compressible.materialValidation import buildMaterialSystem, materialConservation, hostScalars
from ...caseUtils.compressible.sod.twoGasReference import sodProfile, sodWaves
from ..common.parameters import COMPRESSIBLE_CASE_PARAMETERS
from ..common.plotting import Field, particlePlot
from .common import COMPRESSIBLE_DEFAULTS, COMPRESSIBLE_PARAMS, configureCompressible, compressibleTimestep, compressibleDiagnostics, paramExtraData


def configureScheme(ctx):
    configureCompressible(ctx)
    ctx.schemeConfig.materials = [MaterialConfig('leftGas', EOSConfig(gamma=5/3)),
                                 MaterialConfig('rightGas', EOSConfig(gamma=5/3))]
    ctx.config.domain.min[:] = -ctx.spec.L
    ctx.config.domain.max[:] = ctx.spec.L


def states(ctx):
    g0, g1 = [m.eos.gamma for m in ctx.schemeConfig.materials]
    return (ctx.param('rho0'), 0., ctx.param('pressureLeft'), g0), (ctx.param('rhoRight'), 0., ctx.param('pressureRight'), g1)


def buildSystem(ctx):
    if ctx.spec.dim != 1 or not bool(ctx.config.domain.periodic.all()):
        raise ValueError('twoGasSod requires a one-dimensional padded periodic domain')
    if len(ctx.schemeConfig.materials) != 2:
        raise ValueError('twoGasSod requires exactly two materials')
    left, right = states(ctx)
    waves = sodWaves(left, right)
    # Seam waves must not reach the observation interval [-L/2,L/2].
    speed = max(abs(waves['fanHead']), waves['velocity'] + (left[3]*waves['pressure']/waves['densityLeft'])**.5,
                waves['waveRight'])
    if speed * ctx.spec.tLimit + 4*ctx.spec.n_h*ctx.spec.L/ctx.spec.nx >= .5*ctx.spec.L:
        raise ValueError('twoGasSod tLimit exceeds the padded-seam observation limit')
    nl = 2*ctx.spec.nx  # nx is the left-state count in the observed half-unit.
    nr = max(4, round(nl * right[0]/left[0])) if ctx.param('equalMass') else nl
    parts, vols = [], []
    for n, lo in ((nl, -ctx.spec.L), (nr, 0.)):
        dx = ctx.spec.L/n
        parts.append(lo + (torch.arange(n, device=ctx.device, dtype=ctx.dtype)+.5)*dx)
        vols.append(torch.full((n,), dx, device=ctx.device, dtype=ctx.dtype))
    x = torch.cat(parts)
    ids = (x >= 0).to(torch.int32)
    rho = torch.where(ids == 0, left[0], right[0])
    pressure = torch.where(ids == 0, left[2], right[2])
    ctx.scratch['sodStates'] = (left, right)
    ctx.scratch['sodWaves'] = waves
    return buildMaterialSystem(ctx, x[:, None], torch.cat(vols), rho, pressure, ids)


def diagnostics(ctx, system):
    s = system.state
    x = s.positions[:, 0]
    mask = x.abs() < .5*ctx.spec.L
    exact = sodProfile(x, float(system.t), *ctx.scratch['sodStates'])
    volume = s.masses / s.densities
    weights = torch.where(mask, volume, 0.)
    out = dict(compressibleDiagnostics(ctx, system), **materialConservation(ctx, system))
    for name, value in (('density', s.densities), ('pressure', s.pressures), ('velocity', s.velocities[:, 0])):
        out[name+'L1'] = (weights*(value-exact[name]).abs()).sum()/weights.sum()
    left, right = ctx.scratch['sodStates']
    if left[2] == right[2]:
        out['pressureError'] = torch.where(mask, (s.pressures-left[2]).abs(), 0.).max()
        out['maxPeculiarVelocity'] = torch.where(mask, s.velocities[:, 0].abs(), 0.).max()
    return hostScalars(out)


_setup, _update = particlePlot([Field('densities', 'Density', vMin=.2, vMax=1.),
    Field('pressures', 'Pressure', vMin=.1, vMax=1.), Field('velocities', 'Velocity', mapping='x', vMin=0., vMax=.8)], figsize=(12, 3))


def plotView(ctx, system):
    # A 2D colour strip renders the genuinely 1D state through the fast vispy
    # backend. It is a display copy; the solver and Remote retain 1D particles.
    view = copy.copy(system)
    view.state = copy.copy(system.state)
    view.state.positions = torch.cat((system.state.positions, torch.zeros_like(system.state.positions)), dim=1)
    cfg = copy.copy(ctx.config)
    cfg.domain = DomainDescription(torch.tensor([-.5*ctx.spec.L, -.025], device=ctx.device, dtype=ctx.dtype),
        torch.tensor([.5*ctx.spec.L, .025], device=ctx.device, dtype=ctx.dtype),
        torch.zeros(2, device=ctx.device, dtype=torch.bool), 2)
    return replace(ctx, config=cfg), view


def setupPlot(ctx, system):
    return _setup(*plotView(ctx, system))


def updatePlot(ctx, system, plotter, step):
    viewctx, view = plotView(ctx, system)
    _update(viewctx, view, plotter, step)


setupPlot.usesWarp = updatePlot.usesWarp = False
twoGasSodCase = registerCase(Case(name='twoGasSod', scheme='CRKSPH',
    description='Two-material Sod tube with exact two-gamma profile, observed on [-L/2,L/2] before periodic-seam signals arrive.',
    configureScheme=configureScheme, buildSystem=buildSystem, diagnostics=diagnostics,
    timestep=compressibleTimestep, setupPlot=setupPlot, updatePlot=updatePlot,
    extraData=paramExtraData, extraFields=('internalEnergies', 'supports'), supportedSchemes=('CRKSPH',),
    defaults=dict(COMPRESSIBLE_DEFAULTS, caseName='twoGasSod', dim=1, nx=400, L=1., tLimit=.15,
                  cflFactor=.15, plotInterval=20, plotBackend='vispy'),
    params=dict(COMPRESSIBLE_PARAMS, viscositySwitch='NoneSwitch', rhoRight=.25,
                pressureLeft=1., pressureRight=.1795, equalMass=True, initialDensity='prescribed'),
    parameterDefinitions={**COMPRESSIBLE_CASE_PARAMETERS,
        'rhoRight': P('number', 'Right density.', 'initial', exclusiveMinimum=0),
        'pressureLeft': P('number', 'Left pressure.', 'initial', exclusiveMinimum=0),
        'pressureRight': P('number', 'Right pressure.', 'initial', exclusiveMinimum=0),
        'equalMass': P('boolean', 'Match particle mass on both sides; otherwise use equal spacing.', 'sampling'),
        'initialDensity': P('string', 'Density used to initialize energy; prescribed matches Spheral sumInitialDensity=False.', 'initial', choices=('prescribed', 'summed'))},
    simulationParameters={'dim': P('integer', 'One-dimensional tube.', 'domain', choices=(1,))},
    variants={'stationary': {'params': {'pressureRight': 1.}, 'simulation': {'tLimit': .02}},
              'twoGamma': {'solver': {'materials': [
        {'name': 'leftGas', 'eos': {'type': 'idealGas', 'gamma': 1.4}},
        {'name': 'rightGas', 'eos': {'type': 'idealGas', 'gamma': 5/3}}]}}}))

if __name__ == '__main__':
    caseMain(twoGasSodCase)
