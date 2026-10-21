"""Shared sampling for the material verification cases; no pair-policy changes."""
import torch

from ...sample.regular import sampleRegularParticles
from ...utils.support import volumeToSupport
from ...modules.adaptiveSupport import solveStageSupport
from ...modules.crk import crkFactors
from ...modules.eos import evaluateEOS, validateMaterialState
from ...modules.timestep.compressible import computeTimestep


def hostScalars(values):
    """One batched transfer for the runner's JSON/NumPy diagnostic contract."""
    tensor = next((v for v in values.values() if isinstance(v, torch.Tensor)), None)
    if tensor is None:
        return values
    host = torch.stack([torch.as_tensor(v, device=tensor.device).reshape(()) for v in values.values()]).detach().cpu().tolist()
    return dict(zip(values, host))


def buildMaterialSystem(ctx, positions, volumes, rho, pressure, ids, velocity=None):
    velocity = torch.zeros_like(positions) if velocity is None else velocity
    ones, zeros = torch.ones_like(rho), torch.zeros_like(rho)
    state = ctx.SimulationState(positions=positions,
        supports=volumeToSupport(volumes, ctx.config.targetNeighbors, ctx.spec.dim),
        masses=volumes * rho, densities=rho.clone(), velocities=velocity,
        kinds=torch.zeros_like(ids), materials=ids,
        UIDs=torch.arange(len(rho), device=rho.device, dtype=torch.int64), UIDcounter=len(rho),
        internalEnergies=ones.clone(), totalEnergies=ones.clone(), pressures=pressure,
        entropies=ones.clone(), soundspeeds=ones.clone(), alphas=ones.clone(),
        alpha0s=ones.clone(), divergence=zeros)
    system = ctx.SimulationSystem(state=state, domain=ctx.config.domain)
    validateMaterialState(state, ctx.schemeConfig)
    solveStageSupport(system, state, ctx.config, ctx.schemeConfig)
    _, summed, _ = crkFactors(state, ctx.config.domain, ctx.config.kernel, system.adjacency,
        densityPolicy=ctx.schemeConfig.crkDensityPolicy,
        volumes=volumes if getattr(ctx.schemeConfig, 'crkEvolution', 'summation') == 'spheral' else None)
    ctx.scratch['prescribedDensity'] = rho.clone()
    ctx.scratch['initialSummedDensity'] = summed.clone()
    state.densities = summed if ctx.param('initialDensity') == 'summed' else rho.clone()
    if getattr(ctx.schemeConfig, 'crkEvolution', 'summation') == 'spheral':
        state.crkVolumes = volumes.clone()
        state.crkMassDensities = state.densities.clone()
    state.entropies, state.internalEnergies, state.pressures, state.soundspeeds = evaluateEOS(
        state.densities, ids, ctx.schemeConfig, P=pressure)
    state.totalEnergies = state.masses * (state.internalEnergies + .5 * velocity.square().sum(-1))
    ctx.scratch['materialInitialEnergy'] = state.totalEnergies.sum().detach()
    ctx.scratch['materialInitialMasses'] = torch.stack([state.masses[ids == i].sum() for i in range(2)])
    ctx.config.dt = computeTimestep(system, ctx.config, ctx.schemeConfig)
    ctx.velocityScale = float(state.soundspeeds.max().detach().cpu())
    ctx.velocityScaleSource = 'initial maximum sound speed'
    return system


def materialConservation(ctx, system):
    s = system.state
    energy = (s.masses * (s.internalEnergies + .5 * s.velocities.square().sum(-1))).sum()
    masses = torch.stack([s.masses[s.materials == i].sum() for i in range(2)])
    return dict(relativeEnergyDrift=energy / ctx.scratch['materialInitialEnergy'] - 1,
                materialMassError=(masses / ctx.scratch['materialInitialMasses'] - 1).abs().max(),
                minDensity=s.densities.min(), minEnergy=s.internalEnergies.min(),
                minPressure=s.pressures.min(), maxVelocity=torch.linalg.vector_norm(s.velocities, dim=-1).max())


def regularMaterialParticles(ctx):
    p = sampleRegularParticles(ctx.spec.nx, ctx.config.domain, ctx.config.targetNeighbors)
    return p.positions, p.masses
