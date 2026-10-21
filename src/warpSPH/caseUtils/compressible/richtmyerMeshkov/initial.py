"""Pressure-balanced CRKSPH RM initial conditions on a displaced lattice.

The x ends form a distant periodic seam. Only the central interaction before
any seam signal can arrive is the intended single-shock RM problem.
"""
import math
import torch

from ....sample.regular import sampleRegularParticles
from ....modules.adaptiveSupport import solveStageSupport
from ....modules.crk import crkFactors
from ....modules.density import computeDensities
from ....modules.eos import evaluateEOS, validateMaterialState
from ....modules.timestep.compressible import computeTimestep
from .reference import incidentShock, interfaceStarState

__all__ = ['sampleRichtmyerMeshkov']


def sampleRichtmyerMeshkov(ctx):
    p = ctx.spec.params
    cfg, scfg = ctx.config, ctx.schemeConfig
    particles = sampleRegularParticles(ctx.spec.nx, cfg.domain, cfg.targetNeighbors)
    positions = particles.positions.clone()
    q = positions[:, 0].clone()
    y = positions[:, 1]
    wavelength = ctx.spec.L
    amplitude = p['amplitude'] * wavelength
    width = p['interfaceWidth'] * wavelength
    # Compact C1 displacement. The interface is represented geometrically,
    # even when its amplitude is smaller than one undeformed lattice spacing.
    t = (q.abs() / width).clamp(max=1)
    envelope = (1 - t.square()).square()
    envelopeDerivative = torch.where(q.abs() < width, -4 * q / width**2 * (1 - t.square()), torch.zeros_like(q))
    wave = torch.cos(2 * math.pi * y / wavelength)
    positions[:, 0] += amplitude * wave * envelope
    jacobian = 1 + amplitude * wave * envelopeDerivative
    if bool((jacobian <= 0).any()):
        raise ValueError('params.amplitude: interface displacement folds the particle lattice')
    ids = (q >= 0).to(torch.int32)
    rho = torch.where(ids == 0, torch.full_like(q, p['rho0']), torch.full_like(q, p['rho1']))
    gamma0, gamma1 = (m.eos.gamma for m in scfg.materials)
    shock = incidentShock(p['rho0'], p['pressure'], gamma0, p['mach'])
    shocked = q < p['shockPosition'] * wavelength
    rho = torch.where(shocked, torch.full_like(rho, shock['density']), rho)
    pressure = torch.where(shocked, torch.full_like(rho, shock['pressure']), torch.full_like(rho, p['pressure']))
    velocity = torch.zeros_like(positions)
    velocity[:, 0] = torch.where(shocked, shock['velocity'], 0.) + p['boost']
    # The deformation changes cell volume; mass = prescribed rho * actual volume.
    masses = particles.masses * jacobian * rho
    zeros = torch.zeros_like(rho)
    ones = torch.ones_like(rho)
    state = ctx.SimulationState(positions=positions, supports=particles.supports.clone(), masses=masses,
        densities=rho.clone(), velocities=velocity, kinds=torch.zeros_like(ids), materials=ids,
        UIDs=torch.arange(rho.numel(), device=rho.device, dtype=torch.int64), UIDcounter=rho.numel(),
        internalEnergies=ones.clone(), totalEnergies=ones.clone(), pressures=pressure,
        entropies=ones.clone(), soundspeeds=ones.clone(), alphas=ones.clone(), alpha0s=ones.clone(), divergence=zeros)
    system = ctx.SimulationSystem(state=state, domain=cfg.domain)
    validateMaterialState(state, scfg)
    solveStageSupport(system, state, cfg, scfg)
    if ctx.scheme.name == 'CRKSPH':
        _, state.densities, _ = crkFactors(state, cfg.domain, cfg.kernel, system.adjacency,
            densityPolicy=scfg.crkDensityPolicy,
            volumes=masses/rho if scfg.crkEvolution == 'spheral' else None)
        if scfg.crkEvolution == 'spheral':
            state.crkVolumes = masses/rho
            state.crkMassDensities = state.densities.clone()
    else:
        state.densities = computeDensities(state, cfg, scfg, system.adjacency)
    state.entropies, state.internalEnergies, state.pressures, state.soundspeeds = evaluateEOS(
        state.densities, ids, scfg, P=pressure)
    state.totalEnergies = masses * (state.internalEnergies + .5 * velocity.square().sum(-1))
    cfg.dt = computeTimestep(system, cfg, scfg)
    ctx.velocityScale = max(shock['speed'] + abs(p['boost']), 1e-12)
    ctx.velocityScaleSource = 'incident shock speed plus uniform boost'
    star = interfaceStarState((shock['density'], shock['velocity'] + p['boost'], shock['pressure'], gamma0),
                              (p['rho1'], p['boost'], p['pressure'], gamma1))
    marker = q.abs() < .75 * wavelength / ctx.spec.nx
    ctx.scratch.update(rmShock=shock, rmStar=star, rmPrescribedDensity=rho,
        rmMarker=marker, rmMarkerCos=wave[marker].clone(), rmMarkerSin=torch.sin(2 * math.pi * y[marker] / wavelength),
        rmInitialEnergy=state.totalEnergies.sum().detach(), rmInitialMasses=masses.detach().clone())
    return system
