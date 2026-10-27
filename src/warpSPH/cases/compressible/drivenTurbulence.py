"""Periodic 2D driven ideal-gas turbulence with explicit finite-time cooling.

This is a developing-flow setup, not an exact-isothermal Hopkins benchmark.
The forcing amplitude is an input; RMS Mach is measured, never feedback-clamped.
"""
import math
import json
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
    ctx.scratch['checkpointSignature'] = checkpointSignature(ctx)
    ctx.scratch['completedSteps'] = 0
    cfg.dt = timestep(ctx, system)
    return system


def timestep(ctx, system):
    dt = compressibleTimestep(ctx, system)
    field = ctx.scratch['driving']
    # Resolve the prescribed clock and explicit relaxation, including the first step.
    limit = ctx.param('forcingInterval')
    if system.sourceTerms.coolingTime is not None:
        limit = min(limit, .2*system.sourceTerms.coolingTime)
    # A step spans at most one knot interval, possibly crossing a knot.
    # Norms are convex on each linear segment: all three enclosing knots bound
    # the acceleration throughout both SSPRK stages, including future growth.
    bound = field.accelerationBound(float(system.t))
    if bound > 0:
        # Mean removal can double the acceleration bound. Use nominal dx here;
        # the native acoustic signal restriction still follows evolving geometry.
        accelerationBound = 2*bound
        limit = min(limit, math.sqrt(.25*ctx.config.dx/accelerationBound))
        thermal = float(system.state.internalEnergies.min())
        if not math.isfinite(thermal) or thermal <= 0:
            raise ValueError('drivenTurbulence timestep requires positive finite internal energy')
        # Forward Euler force work lowers predictor u by |a dt|^2/2 even though
        # SSPRK's full constant-force step preserves u. Reserve thermal energy
        # for that predictor: force alone consumes at most 1/16 of min(u).
        limit = min(limit, .25*math.sqrt(2*thermal)/accelerationBound)
    physical = torch.clamp(dt, max=limit) if isinstance(dt, torch.Tensor) else min(dt, limit)
    ctx.scratch['nextPhysicalDt'] = physical
    if ctx.spec.nSteps is None:
        remaining = max(0., ctx.spec.tLimit-float(system.t))
        return torch.clamp(physical, max=remaining) if isinstance(physical, torch.Tensor) else min(physical, remaining)
    return physical


def postStep(ctx, system, step):
    ctx.scratch['completedSteps'] = step+1
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
    data = dict(ctx.spec.params, forcingParameters=dict(field.parameters),
                forcingModes=field.modes, forcingCoefficients=field.coefficients(float(system.t)),
                forcingTime=float(system.t), sourceEnergyBudget=system.state.sourceEnergyBudget.detach().cpu(),
                initialEnergy=float(ctx.scratch['initialEnergy']), initialMass=float(ctx.scratch['initialMass']))
    if ctx.spec.storeMode == 'states':
        state = field.state_dict()
        data.update(checkpointVersion=1, checkpointSignature=ctx.scratch['checkpointSignature'],
                    checkpointCompletedSteps=ctx.scratch['completedSteps'],
                    checkpointNextDt=torch.as_tensor(ctx.scratch['nextPhysicalDt']).detach().cpu(),
                    forcingCheckpoint=dict(knots=state['knots'], generator=state['generator']))
    return data


def trajectoryData(ctx, system):
    """Export every forcing knot once, including knots between stored frames."""
    data = extraData(ctx, system)
    field = ctx.scratch['driving']
    start = ctx.scratch.get('exportedForcingKnots', 0)
    shape = (0, *field.knots[0].shape)
    raw = (torch.stack(field.knots[start:]) if start < len(field.knots)
           else field.knots[0].new_empty(shape))
    data.update(forcingKnotStart=start, forcingKnots=raw,
        forcingKnotCoefficients=field.scale*torch.einsum('mij,kpmj->kpmi', field.projection, raw))
    ctx.scratch['exportedForcingKnots'] = len(field.knots)
    return data


def checkpointSignature(ctx):
    """Physical/numerical policy; allow output settings and endpoint to change."""
    signature = dict(parameters=ctx.spec.params, solver=ctx.exportFunction(ctx.schemeConfig),
        scheme=ctx.spec.scheme, precision=str(ctx.dtype),
        domain={name: getattr(ctx.spec, name) for name in ('dim', 'nx', 'L', 'periodic', 'kernel', 'n_h')},
        timestep={name: getattr(ctx.config, name) for name in ('adaptiveDt', 'minDt', 'maxDt', 'dtGrowthFactor')},
        integrator=ctx.integrator.identifier.name)
    return json.dumps(signature, sort_keys=True, separators=(',', ':'), allow_nan=False)


def restoreCheckpoint(ctx, system, checkpoint):
    required = ('checkpointVersion', 'checkpointSignature', 'checkpointCompletedSteps', 'initialEnergy', 'initialMass')
    if any(name not in checkpoint.attrs for name in required) or any(name not in checkpoint for name in ('checkpointNextDt', 'dict_forcingCheckpoint')):
        raise ValueError('drivenTurbulence resumeFrom: checkpoint is missing forcing/timestep/budget context')
    if checkpoint.attrs['checkpointVersion'] != 1 or checkpoint.attrs['checkpointSignature'] != ctx.scratch['checkpointSignature']:
        raise ValueError('drivenTurbulence resumeFrom: incompatible checkpoint physical/numerical configuration')
    if not math.isfinite(system.t) or system.t < 0 or (ctx.spec.nSteps is None and ctx.spec.tLimit <= system.t):
        raise ValueError('drivenTurbulence resumeFrom: endpoint must be later than the checkpoint time')
    completed = int(checkpoint.attrs['checkpointCompletedSteps'])
    if completed < 0 or ctx.spec.resumeStepOffset not in (0, completed):
        raise ValueError('drivenTurbulence resumeStepOffset does not match checkpointCompletedSteps')
    nextDt = float(checkpoint['checkpointNextDt'][()])
    if not math.isfinite(nextDt) or nextDt <= 0 or system.state.sourceEnergyBudget is None:
        raise ValueError('drivenTurbulence resumeFrom: invalid timestep or missing accepted energy budget')
    for name in ('initialMass', 'initialEnergy'):
        value = float(checkpoint.attrs[name])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'drivenTurbulence resumeFrom: invalid {name}')
        ctx.scratch[name] = system.state.Q.new_tensor(value, dtype=torch.float64)
    state = checkpoint['dict_forcingCheckpoint']
    field = FourierOU.from_state_dict(dict(parameters=ctx.scratch['driving'].parameters,
        knots=torch.from_numpy(state['knots'][:]), generator=torch.from_numpy(state['generator'][:])))
    ctx.scratch.update(driving=field, completedSteps=completed,
                       nextPhysicalDt=system.state.Q.new_tensor(nextDt))
    system.sourceTerms.acceleration = field
    system.state.UIDcounter = int(system.state.UIDcounter)
    ctx.spec.resumeStepOffset = completed
    # Preserve growth history while capping the new endpoint/forcing/cooling.
    ctx.config.dt = system.state.Q.new_tensor(nextDt)
    if ctx.spec.nSteps is None:
        ctx.config.dt = torch.clamp(ctx.config.dt, max=ctx.spec.tLimit-system.t)


setupPlot, updatePlot = particlePlot([
    Field('densities', 'Density', colorMap='viridis'),
    Field('velocities', 'Speed', mapping='L2Norm', colorMap='inferno'),
], figsize=(10, 5))


drivenTurbulenceCase = registerCase(Case(
    name='drivenTurbulence', scheme='MFM', buildSystem=buildSystem,
    configureScheme=configureCompressible, timestep=timestep, postStep=postStep,
    diagnostics=diagnostics, extraData=extraData, trajectoryData=trajectoryData, restoreCheckpoint=restoreCheckpoint, setupPlot=setupPlot, updatePlot=updatePlot,
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
    extraFields=('internalEnergies', 'pressures', 'supports', 'masses', 'Q', 'sourceEnergyBudget', 'UIDs'),
))

if __name__ == '__main__':
    caseMain(drivenTurbulenceCase)
