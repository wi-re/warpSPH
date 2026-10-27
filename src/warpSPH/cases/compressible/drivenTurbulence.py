"""Periodic 2D driven ideal-gas turbulence with explicit finite-time cooling.

This is a developing-flow setup, not an exact-isothermal Hopkins benchmark.
The forcing amplitude is an input; RMS Mach is measured, never feedback-clamped.
"""
import math
import torch

from ...runner import Case, registerCase, caseMain
from ...runner.parameters import Parameter as P
from ...configurations.meshless import resolve_temporal_scheme
from ...modules.forcing import FourierOU
from ...modules.meshless.sources import ConservedSources
from .common import COMPRESSIBLE_DEFAULTS, configureCompressible, compressibleTimestep
from ..common.plotting import Field, particlePlot

__all__ = ['drivenTurbulenceCase']


def validateConfiguration(spec):
    p = spec.params
    if spec.resumeFrom is not None:
        raise ValueError('simulation.resumeFrom: full runner restart is not yet qualified for drivenTurbulence; standalone forcing checkpoints are supported')
    if p['kMin'] > p['kMax'] or p['kMax'] >= spec.nx/2:
        raise ValueError('params.kMin/kMax: require kMin <= kMax < nx/2')
    if p['forcingInterval'] > p['correlationTime']:
        raise ValueError('params.forcingInterval: must not exceed correlationTime')
    temporal = spec.schemeOverrides.get('temporalScheme', 'auto')
    if temporal == 'auto':
        temporal = 'ssprk' if spec.scheme == 'MFM' else 'hancock'
    if temporal != 'ssprk':
        raise ValueError('drivenTurbulence requires solver.temporalScheme=ssprk for stage-consistent sources')
    if spec.cudaGraph or spec.execution not in (None, 'eager'):
        raise ValueError('drivenTurbulence currently requires eager execution')
    if not spec.adaptiveDt:
        raise ValueError('drivenTurbulence requires adaptiveDt for forcing/cooling limits')
    if spec.schemeOverrides.get('materials') is not None:
        raise ValueError('drivenTurbulence currently uses a single ideal gas')


def buildSystem(ctx):
    from ...sample.regular import sampleRegularParticles
    from ...caseUtils.meshless import buildMeshlessState
    from ...modules.meshless.geometry import buildGeometry, stashGeometry
    from ...systems.meshless import MeshlessSystem

    cfg, scheme = ctx.config, ctx.schemeConfig
    if resolve_temporal_scheme(scheme) != 'ssprk':
        raise ValueError('drivenTurbulence requires SSPRK sources')
    particles = sampleRegularParticles(ctx.spec.nx, cfg.domain, cfg.targetNeighbors)
    pos, supports = particles.positions, particles.supports
    geometry = buildGeometry(pos, supports, None, cfg, scheme, cfg.domain)
    mass = ctx.param('rho0')*particles.masses
    density = mass/geometry.volume
    u0 = ctx.param('soundSpeed')**2/(scheme.gamma*(scheme.gamma-1))
    pressure = (scheme.gamma-1)*density*u0
    labels = torch.zeros(len(pos), device=pos.device, dtype=torch.int32)
    state = buildMeshlessState(pos, supports, mass, density, torch.zeros_like(pos), pressure,
        scheme.gamma, labels, labels.clone(), torch.arange(len(pos), device=pos.device, dtype=torch.int32))
    state.lam = geometry.lam
    state.sourceEnergyBudget = mass.new_zeros(2, dtype=torch.float64)
    field = FourierOU(dim=2, length=ctx.spec.L, kMin=ctx.param('kMin'), kMax=ctx.param('kMax'),
        accelerationRMS=ctx.param('accelerationRMS'), correlationTime=ctx.param('correlationTime'),
        updateInterval=ctx.param('forcingInterval'), solenoidalWeight=ctx.param('solenoidalWeight'), seed=ctx.param('seed'))
    cooling = ctx.param('coolingTime') if ctx.param('thermalModel') == 'relaxation' else None
    source = ConservedSources(field, ctx.param('removeMeanAcceleration'), u0, cooling)
    system = MeshlessSystem(state=state, domain=cfg.domain, sourceTerms=source)
    stashGeometry(pos, supports, cfg.domain, cfg.kernel, scheme, geometry, owner=state)
    ctx.scratch.update(driving=field, initialMass=mass.double().sum(),
                       initialEnergy=state.Q[:, -1].double().sum())
    ctx.velocityScale = ctx.param('soundSpeed')
    ctx.velocityScaleSource = 'drivenTurbulence reference sound speed'
    cfg.dt = timestep(ctx, system)
    return system


def timestep(ctx, system):
    dt = compressibleTimestep(ctx, system)
    field = ctx.scratch['driving']
    # Resolve the prescribed clock and explicit relaxation, including the first step.
    limit = ctx.param('forcingInterval')
    if system.sourceTerms.coolingTime is not None:
        limit = min(limit, .2*system.sourceTerms.coolingTime)
    coefficients = field.coefficients(float(system.t))
    bound = float(coefficients.norm(dim=-1).sum())
    if bound > 0:
        # Mean removal can double the acceleration bound. Use nominal dx here;
        # the native acoustic signal restriction still follows evolving geometry.
        limit = min(limit, math.sqrt(.25*ctx.config.dx/(2*bound)))
    if ctx.spec.nSteps is None:
        limit = min(limit, max(0., ctx.spec.tLimit-float(system.t)))
    return torch.clamp(dt, max=limit) if isinstance(dt, torch.Tensor) else min(dt, limit)


def postStep(ctx, system, step):
    if ctx.spec.nSteps is None:
        # Snap only accumulated time roundoff after a physically capped step.
        t, target = float(system.t), ctx.spec.tLimit
        eps = torch.finfo(system.state.Q.dtype).eps
        if abs(t-target) <= 4*eps*max(abs(t), abs(target)):
            system.t = float(target)


def diagnostics(ctx, system):
    s = system.state
    mass, velocity, u = s.Q[:, 0].double(), s.velocities.double(), s.internalEnergies.double()
    total_mass = mass.sum()
    mean = (mass[:, None]*velocity).sum(0)/total_mass
    fluctuating = velocity-mean
    sound2 = ctx.schemeConfig.gamma*(ctx.schemeConfig.gamma-1)*u
    kinetic = (.5*mass*velocity.square().sum(-1)).sum()
    total = s.Q[:, -1].double().sum()
    budget = s.sourceEnergyBudget.double()
    rho = s.densities.double()
    values = dict(kineticEnergy=kinetic, thermalEnergy=(mass*u).sum(), totalEnergy=total,
        mass=total_mass, relativeMassChange=total_mass/ctx.scratch['initialMass']-1,
        rmsMachMass=torch.sqrt((mass*fluctuating.square().sum(-1)/sound2).sum()/total_mass),
        meanSpeed=mean.norm(), maxVelocity=s.velocities.norm(dim=1).max(),
        minDensity=rho.min(), maxDensity=rho.max(), minPressure=s.pressures.min(),
        minEnergy=u.min(), minSupport=s.supports.min(), logDensityStdParticle=rho.log().std(unbiased=False),
        forceWork=budget[0], thermalExchange=budget[1],
        relativeEnergyBudgetResidual=(total-ctx.scratch['initialEnergy']-budget.sum())/ctx.scratch['initialEnergy'])
    out = dict(zip(values, torch.stack(list(values.values())).detach().cpu().tolist()))
    out['thermalModel'] = ctx.param('thermalModel')
    return out


def extraData(ctx, system):
    field = ctx.scratch['driving']
    return dict(ctx.spec.params, forcingParameters=dict(field.parameters),
                forcingModes=field.modes, forcingCoefficients=field.coefficients(float(system.t)),
                forcingTime=float(system.t), sourceEnergyBudget=system.state.sourceEnergyBudget.detach().cpu(),
                initialEnergy=float(ctx.scratch['initialEnergy']), initialMass=float(ctx.scratch['initialMass']))


setupPlot, updatePlot = particlePlot([
    Field('densities', 'Density', colorMap='viridis'),
    Field('velocities', 'Speed', mapping='L2Norm', colorMap='inferno'),
], figsize=(10, 5))


drivenTurbulenceCase = registerCase(Case(
    name='drivenTurbulence', scheme='MFM', buildSystem=buildSystem,
    configureScheme=configureCompressible, timestep=timestep, postStep=postStep,
    diagnostics=diagnostics, extraData=extraData, setupPlot=setupPlot, updatePlot=updatePlot,
    description='Developing periodic 2D compressible turbulence with OU acceleration and named finite-time cooling.',
    defaults=dict(COMPRESSIBLE_DEFAULTS, caseName='drivenTurbulence', dim=2, nx=64, L=1.,
                  tLimit=4., cflFactor=.15, execution='eager', cudaGraph=False, maxDt=.01),
    params=dict(gamma=5/3, rho0=1., soundSpeed=1., accelerationRMS=.3,
                correlationTime=.5, forcingInterval=.05, kMin=1, kMax=2,
                solenoidalWeight=1., seed=1234, removeMeanAcceleration=True,
                thermalModel='relaxation', coolingTime=.2),
    supportedSchemes=('MFM', 'MFV'), validateConfiguration=validateConfiguration,
    simulationParameters={
        'dim': P('integer', 'This first turbulence case is two dimensional.', 'domain', choices=(2,)),
        'periodic': P('boolean', 'Periodic box on every axis.', 'domain', choices=(True,)),
    },
    parameterDefinitions={
        'gamma': P('number', 'Finite ideal-gas ratio of specific heats.', 'fluid', exclusiveMinimum=1),
        'rho0': P('number', 'Initial nominal mass density.', 'fluid', exclusiveMinimum=0),
        'soundSpeed': P('number', 'Initial adiabatic sound speed; sets bath energy.', 'fluid', exclusiveMinimum=0),
        'accelerationRMS': P('number', 'Ensemble RMS acceleration at OU knots before mean removal; no Mach feedback.', 'forcing', minimum=0),
        'correlationTime': P('number', 'OU correlation time.', 'forcing', exclusiveMinimum=0),
        'forcingInterval': P('number', 'Fixed physical OU knot spacing; linear interpolation between knots.', 'forcing', exclusiveMinimum=0),
        'kMin': P('integer', 'Minimum Fourier mode radius in box units.', 'forcing', minimum=1, maximum=16),
        'kMax': P('integer', 'Maximum Fourier mode radius in box units.', 'forcing', minimum=1, maximum=16),
        'solenoidalWeight': P('number', 'Amplitude mixture: 1 solenoidal, 0 longitudinal.', 'forcing', minimum=0, maximum=1),
        'seed': P('integer', 'CPU OU generator seed, independent of global RNG state.', 'forcing', minimum=0, maximum=2**63-1),
        'removeMeanAcceleration': P('boolean', 'Remove the instantaneous mass-weighted mean acceleration.', 'forcing'),
        'thermalModel': P('string', 'Relaxation bath or adiabatic finite-gamma gas; neither is an exact isothermal closure.', 'thermal', choices=('relaxation', 'adiabatic')),
        'coolingTime': P('number', 'Specific internal-energy relaxation time.', 'thermal', exclusiveMinimum=0,
                         appliesWhen={'thermalModel': 'relaxation'}),
    },
    extraFields=('internalEnergies', 'pressures', 'supports', 'masses'),
))

if __name__ == '__main__':
    caseMain(drivenTurbulenceCase)
