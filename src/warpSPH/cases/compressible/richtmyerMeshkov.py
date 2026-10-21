"""Single-mode Richtmyer–Meshkov instability of two independently configured ideal gases."""
import math
import copy
from dataclasses import replace

import torch

from ...configurations import EOSConfig, MaterialConfig
from ...caseUtils.compressible.richtmyerMeshkov import sampleRichtmyerMeshkov, incidentShock, interfaceStarState
from ...caseUtils.compressible.richtmyerMeshkov.diagnostics import markerMeasures, interactionDiagnostics
from ...runner import Case, RunContext, registerCase, caseMain
from ...runner.parameters import Parameter as P
from ..common.parameters import COMPRESSIBLE_CASE_PARAMETERS
from ..common.plotting import Field, particlePlot
from .common import (COMPRESSIBLE_DEFAULTS, COMPRESSIBLE_PARAMS, configureCompressible,
                     compressibleDiagnostics, compressibleTimestep, paramExtraData)

__all__ = ['richtmyerMeshkovCase']


def configureScheme(ctx: RunContext):
    if ctx.scheme.name not in ('CRKSPH', 'CompSPH', 'Monaghan'):
        raise ValueError('richtmyerMeshkov supports CRKSPH, CompSPH and Monaghan')
    validateConfiguration(ctx.spec)
    configureCompressible(ctx)
    ctx.schemeConfig.materials = [MaterialConfig('incidentGas', EOSConfig(gamma=1.4)),
                                 MaterialConfig('transmittedGas', EOSConfig(gamma=5/3))]
    L = ctx.spec.L
    ctx.config.domain.min = torch.tensor([ctx.param('xLeft') * L, -.5 * L], device=ctx.device, dtype=ctx.dtype)
    ctx.config.domain.max = torch.tensor([ctx.param('xRight') * L, .5 * L], device=ctx.device, dtype=ctx.dtype)


def _observationLimit(p, materials, wavelength, nx, n_h):
    if materials is None or len(materials) != 2:
        raise ValueError('solver.materials: Richtmyer–Meshkov requires exactly two materials')
    g0, g1 = (m.eos.gamma for m in materials)
    shock = incidentShock(p['rho0'], p['pressure'], g0, p['mach'])
    star = interfaceStarState((shock['density'], shock['velocity'], shock['pressure'], g0), (p['rho1'], 0., p['pressure'], g1))
    # Conservative characteristic estimate, also including the interaction star states.
    signal = abs(p['boost']) + max(shock['speed'], shock['velocity'] + math.sqrt(g0 * shock['pressure'] / shock['density']),
        abs(star['velocity']) + math.sqrt(g0 * star['pressure'] / star['densityLeft']),
        abs(star['velocity']) + math.sqrt(g1 * star['pressure'] / star['densityRight']))
    distance = min(-p['xLeft'] - p['amplitude'], p['xRight'] - p['amplitude']) * wavelength
    # Reserve kernel-width margins at both ends of the causal path.
    return max(0., distance - 4 * n_h * wavelength / nx) / signal


def buildSystem(ctx):
    safeTime = _observationLimit(ctx.spec.params, ctx.schemeConfig.materials, ctx.spec.L, ctx.spec.nx, ctx.config.n_h)
    if ctx.spec.tLimit > safeTime:
        raise ValueError(f'simulation.tLimit: {ctx.spec.tLimit:g} exceeds the periodic-seam observation limit {safeTime:g}; enlarge xLeft/xRight')
    ctx.scratch['rmSafeTime'] = safeTime
    return sampleRichtmyerMeshkov(ctx)


def diagnostics(ctx, system):
    out = compressibleDiagnostics(ctx, system)
    state = system.state
    marker = ctx.scratch['rmMarker']
    x = state.positions[marker, 0]
    cos = ctx.scratch['rmMarkerCos']
    sin = ctx.scratch['rmMarkerSin']
    # Fit Lagrangian interface markers only before the interface overturns.
    values = dict(markerMeasures(x, cos, sin),
        minDensity=state.densities.min(), minEnergy=state.internalEnergies.min(),
        minPressure=state.pressures.min(), maxTransverseVelocity=state.velocities[:, 1].abs().max(),
        maxVelocity=torch.linalg.vector_norm(state.velocities, dim=-1).max(),
        relativeEnergyDrift=(state.masses * (state.internalEnergies + .5 * state.velocities.square().sum(-1))).sum() / ctx.scratch['rmInitialEnergy'] - 1)
    host = torch.stack(list(values.values())).detach().cpu().tolist()
    out.update(zip(values, host))
    out.update(interactionDiagnostics(ctx, system, x))
    return out


def validateConfiguration(spec):
    p = spec.params
    if not (p['xLeft'] < p['shockPosition'] < -p['interfaceWidth'] < 0 < p['interfaceWidth'] < p['xRight']):
        raise ValueError('params.shockPosition/interfaceWidth: require xLeft < shockPosition < -interfaceWidth < 0 < interfaceWidth < xRight')
    if p['amplitude'] >= .25 * p['interfaceWidth']:
        raise ValueError('params.amplitude: must be less than interfaceWidth/4 to avoid a folded lattice')
    if not spec.periodic:
        raise ValueError('simulation.periodic: this case uses a padded periodic domain; nonperiodic boundaries are not implemented')
    from ...configurations.moduleConfigurations.materials import normalizeMaterials
    materials = normalizeMaterials(spec.schemeOverrides.get('materials', [
        MaterialConfig('incidentGas', EOSConfig(gamma=1.4)), MaterialConfig('transmittedGas', EOSConfig(gamma=5/3))]))
    safeTime = _observationLimit(p, materials, spec.L, spec.nx, spec.n_h)
    if spec.tLimit > safeTime:
        raise ValueError(f'simulation.tLimit: exceeds the periodic-seam observation limit {safeTime:g}; enlarge xLeft/xRight')


_setupPlot, _updatePlot = particlePlot([
    Field('materials', 'Material', vMin=0, vMax=1, colorMap='viridis'),
    Field('densities', 'Density', vMin=.8, vMax=4., colorMap='viridis'),
    Field('pressures', 'Pressure', vMin=.9, vMax=1.9, colorMap='viridis'),
], figsize=(12, 5))


def _viewDomain(ctx, state):
    """Follow the interaction, keeping distant periodic buffers out of the view."""
    domain = copy.copy(ctx.config.domain)
    shock, star = ctx.scratch['rmShock'], ctx.scratch['rmStar']
    arrival = -ctx.param('shockPosition') * ctx.spec.L / shock['speed']
    time = float(state.t)
    center = ctx.param('boost') * min(time, arrival) + star['velocity'] * max(0., time - arrival)
    L = ctx.spec.L
    domain.min = torch.tensor([center - .75 * L, -.5 * L], device=ctx.device, dtype=ctx.dtype)
    domain.max = torch.tensor([center + .75 * L, .5 * L], device=ctx.device, dtype=ctx.dtype)
    domain.periodic = torch.zeros_like(domain.periodic)
    return domain


def setupPlot(ctx, state):
    viewContext = replace(ctx, config=copy.copy(ctx.config))
    viewContext.config.domain = _viewDomain(ctx, state)
    return _setupPlot(viewContext, state)


def updatePlot(ctx, state, plotter, step):
    plotter.domain = _viewDomain(ctx, state)
    _updatePlot(ctx, state, plotter, step)


setupPlot.usesWarp = updatePlot.usesWarp = False

richtmyerMeshkovCase = registerCase(Case(
    name='richtmyerMeshkov', scheme='CRKSPH',
    description='Single shock crossing a sinusoidal two-gas material interface; observe before distant periodic-seam signals return.',
    configureScheme=configureScheme, buildSystem=buildSystem, diagnostics=diagnostics,
    timestep=compressibleTimestep, setupPlot=setupPlot, updatePlot=updatePlot,
    extraData=paramExtraData, validateConfiguration=validateConfiguration,
    extraFields=('internalEnergies', 'supports'),
    supportedSchemes=('CRKSPH', 'CompSPH', 'Monaghan'),
    defaults=dict(COMPRESSIBLE_DEFAULTS, caseName='richtmyerMeshkov', dim=2, nx=64, L=1.,
        tLimit=2., cflFactor=.15, plotInterval=20, storeInterval=100),
    params=dict(COMPRESSIBLE_PARAMS, gamma=1.4, viscositySwitch='NoneSwitch',
        rho0=1., rho1=3., pressure=1., mach=1.2, amplitude=.01, interfaceWidth=.5,
        shockPosition=-1., xLeft=-4., xRight=6., boost=0., rmDiagnosticInterval=.05),
    parameterDefinitions={**COMPRESSIBLE_CASE_PARAMETERS,
        'gamma': replace(COMPRESSIBLE_CASE_PARAMETERS['gamma'], readOnly=True, deprecated=True,
                         description='Legacy uniform-gas setting; configure each gas through solver.materials instead.'),
        'rho1': P('number', 'Unshocked density of material 1.', 'initial', exclusiveMinimum=0),
        'pressure': P('number', 'Common unshocked pressure.', 'initial', exclusiveMinimum=0),
        'mach': P('number', 'Incident shock Mach number in material 0; 1 is a stationary-contact control.', 'initial', minimum=1),
        'boost': P('number', 'Uniform streamwise Galilean velocity, including both gases and the incident shock state.', 'initial', units='velocity'),
        'amplitude': P('number', 'Initial sinusoidal displacement divided by wavelength; 0 is the planar control.', 'initial', minimum=0),
        'interfaceWidth': P('number', 'Width of the compact lattice displacement, in wavelengths.', 'sampling', exclusiveMinimum=0),
        'shockPosition': P('number', 'Initial shock x coordinate in wavelengths; the mean material interface is x=0.', 'initial'),
        'xLeft': P('number', 'Left edge of the padded periodic domain, in wavelengths.', 'domain'),
        'xRight': P('number', 'Right edge of the padded periodic domain, in wavelengths.', 'domain'),
        'rmDiagnosticInterval': P('number', 'Physical-time interval between corrected-curl circulation measurements; circulationTime records the actual sample time.', 'diagnostics', exclusiveMinimum=0),
    },
    simulationParameters={'dim': P('integer', 'This case is two-dimensional.', 'domain', choices=(2,)),
        'L': P('number', 'Transverse wavelength.', 'domain', exclusiveMinimum=0)},
    variants={
        'smoke': {'simulation': {'nx': 16, 'nSteps': 5, 'tLimit': 1.}},
        'planar': {'params': {'amplitude': 0.}},
        'contact': {'params': {'mach': 1., 'amplitude': 0.}},
        'demonstration': {'params': {'amplitude': .1, 'xLeft': -12., 'xRight': 20.}, 'simulation': {'tLimit': 6.}},
    },
))


if __name__ == '__main__':
    caseMain(richtmyerMeshkovCase)
