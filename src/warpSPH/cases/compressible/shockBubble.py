"""Spheral's pressure-driven air/helium bubble variant, in dimensionless units.

Four reflected copies in a periodic box represent slip planes through symmetry
at x=0,1.1 and y=0,.4. These freely evolved copies can lose reflection symmetry;
they do not explicitly enforce reflecting walls. No solid wall EOS is involved. Spheral's CGS density
.001, pressure 10 and goalTime .003 correspond here to rho=1, p=1, t=.3.
"""
import copy
from dataclasses import replace
import torch

from ...configurations import EOSConfig, MaterialConfig
from ...runner import Case, registerCase, caseMain
from ...runner.parameters import Parameter as P
from ...caseUtils.compressible.materialValidation import buildMaterialSystem, materialConservation, regularMaterialParticles, hostScalars
from ..common.parameters import COMPRESSIBLE_CASE_PARAMETERS
from ..common.plotting import Field, particlePlot
from .common import COMPRESSIBLE_DEFAULTS, COMPRESSIBLE_PARAMS, configureCompressible, compressibleTimestep, compressibleDiagnostics, paramExtraData


def configureScheme(ctx):
    configureCompressible(ctx)
    ctx.schemeConfig.materials = [MaterialConfig('air', EOSConfig(gamma=1.4)),
                                 MaterialConfig('heliumMixture', EOSConfig(gamma=1.648))]
    ctx.config.domain.min = torch.zeros(2, device=ctx.device, dtype=ctx.dtype)
    ctx.config.domain.max = torch.tensor([2.2, .8], device=ctx.device, dtype=ctx.dtype) * ctx.spec.L


def foldPositions(positions, length=1.):
    extent = positions.new_tensor([1.1, .4]) * length
    return extent - torch.abs(torch.remainder(positions, 2*extent) - extent)


def buildSystem(ctx):
    if ctx.spec.dim != 2 or not bool(ctx.config.domain.periodic.all()):
        raise ValueError('shockBubble requires a two-dimensional reflected periodic domain')
    if len(ctx.schemeConfig.materials) != 2:
        raise ValueError('shockBubble requires exactly two materials')
    positions, volumes = regularMaterialParticles(ctx)
    folded = foldPositions(positions, ctx.spec.L) / ctx.spec.L
    bubble = ((folded - folded.new_tensor([.8, .2])).square().sum(-1) < .1**2)
    ids = bubble.to(torch.int32)
    rho = torch.where(bubble, ctx.param('bubbleDensity'), ctx.param('rho0'))
    pressure = torch.where(folded[:, 0] > 1., ctx.param('driverPressure'), 1.)
    return buildMaterialSystem(ctx, positions, volumes, rho, pressure, ids)


def diagnostics(ctx, system):
    s = system.state
    x, y = s.positions[:, 0], s.positions[:, 1]
    primary = (x < 1.1*ctx.spec.L) & (y < .4*ctx.spec.L)
    bubble = primary & (s.materials == 1)
    mass = torch.where(bubble, s.masses, 0.)
    center = (mass[:, None] * s.positions).sum(0) / mass.sum()
    out = dict(compressibleDiagnostics(ctx, system), **materialConservation(ctx, system),
        bubbleCenterX=center[0], bubbleCenterY=center[1],
        bubbleWidth=torch.where(bubble, x, -torch.inf).max()-torch.where(bubble, x, torch.inf).min(),
        bubbleHeight=torch.where(bubble, y, -torch.inf).max()-torch.where(bubble, y, torch.inf).min(),
        maxPressure=s.pressures.max(),
        bubbleVolume=torch.where(bubble, s.masses/s.densities, 0.).sum())
    if ctx.param('driverPressure') == 1.:
        out['pressureError'] = (s.pressures-1).abs().max()
    time = float(system.t)
    if time >= ctx.scratch.get('bubbleNextCurlTime', -1.):
        from ...modules.reconstruction.gradient import computeVelocityJacobian
        from warpSPHCore import SupportScheme
        jac = computeVelocityJacobian(s, ctx.config, SupportScheme.KernelMeanSymmetric, system.adjacency, corrected=True)
        omega = jac[:, 1, 0] - jac[:, 0, 1]
        circulation = torch.where(primary, s.masses/s.densities * omega, 0.)
        values = dict(circulationUpper=torch.where(y >= .2*ctx.spec.L, circulation, 0.).sum(),
                      circulationLower=torch.where(y < .2*ctx.spec.L, circulation, 0.).sum(),
                      enstrophy=.5*torch.where(primary, s.masses/s.densities * omega.square(), 0.).sum())
        ctx.scratch['bubbleCurl'] = dict(zip(values, torch.stack(list(values.values())).detach().cpu().tolist()), circulationTime=time)
        ctx.scratch['bubbleNextCurlTime'] = time + ctx.param('curlInterval')
    out.update(ctx.scratch['bubbleCurl'])
    return hostScalars(out)


_setup, _update = particlePlot([Field('materials', 'Air / helium mixture', vMin=0, vMax=1),
    Field('densities', 'Density', vMin=.15, vMax=3.), Field('pressures', 'Pressure', vMin=1., vMax=10.)], figsize=(12, 4))


def viewContext(ctx):
    cfg = copy.copy(ctx.config)
    cfg.domain = copy.copy(cfg.domain)
    cfg.domain.max = cfg.domain.max * .5
    cfg.domain.periodic = torch.zeros_like(cfg.domain.periodic)
    return replace(ctx, config=cfg)


def setupPlot(ctx, system):
    return _setup(viewContext(ctx), system)


def updatePlot(ctx, system, plotter, step):
    _update(viewContext(ctx), system, plotter, step)


setupPlot.usesWarp = updatePlot.usesWarp = False
shockBubbleCase = registerCase(Case(name='shockBubble', scheme='CRKSPH',
    description='Spheral pressure-driven air/helium bubble variant; four periodic reflection tiles model slip planes while symmetry holds.',
    configureScheme=configureScheme, buildSystem=buildSystem, diagnostics=diagnostics,
    timestep=compressibleTimestep, setupPlot=setupPlot, updatePlot=updatePlot,
    extraData=paramExtraData, extraFields=('internalEnergies', 'supports'), supportedSchemes=('CRKSPH',),
    defaults=dict(COMPRESSIBLE_DEFAULTS, caseName='shockBubble', dim=2, nx=200, L=1.,
        tLimit=.3, cflFactor=.15, plotInterval=20),
    params=dict(COMPRESSIBLE_PARAMS, viscositySwitch='NoneSwitch', bubbleDensity=.182,
                driverPressure=10., initialDensity='prescribed', curlInterval=.005),
    parameterDefinitions={**COMPRESSIBLE_CASE_PARAMETERS,
        'bubbleDensity': P('number', 'Helium-mixture density relative to ambient air.', 'initial', exclusiveMinimum=0),
        'driverPressure': P('number', 'Pressure of stationary air in 1 < x/L < 1.1.', 'initial', exclusiveMinimum=0),
        'initialDensity': P('string', 'Density used for initial energy; prescribed follows Spheral.', 'initial', choices=('prescribed', 'summed')),
        'curlInterval': P('number', 'Physical-time cadence of corrected curl diagnostics.', 'diagnostics', exclusiveMinimum=0)},
    simulationParameters={'dim': P('integer', 'Two-dimensional cylindrical bubble.', 'domain', choices=(2,))},
    variants={'smoke': {'simulation': {'nx': 32, 'nSteps': 5, 'tLimit': .3}},
              'summedContact': {'params': {'initialDensity': 'summed'}},
              'stationary': {'params': {'driverPressure': 1.}, 'simulation': {'tLimit': .02}}}))

if __name__ == '__main__':
    caseMain(shockBubbleCase)
