"""Sample a region's fluid particles: lay a regular lattice over `config`'s
domain (`sampleRegularParticles`) at resolution `nx`, evaluate `sdf` on it,
and keep only the particles inside it (`sdfDist < 0`). With `filter=False`
the sdf-derived mask/distances are discarded and replaced with trivial
all-True/all-`inf` values, so the full lattice is returned unfiltered.
"""

from ..sample import sampleRegularParticles
import torch
from ..geometry import ParticleSet
import numpy as np

__all__ = ['sampleParticles']


def sampleParticles(config, schemeConfig, sdf, nx, filter = True, shortEdge = True):
    # `config.dx` is the spacing the case asked for (`L / nx`); hand it to the
    # sampler as authoritative so every axis realises ~that spacing, rather than
    # the shortest domain axis's `span/nx` silently winning for a wide-shallow
    # domain and leaving `config.dx` disagreeing with the lattice it feeds
    # `dt` / `c_s` / the mDBC ghost offsets (DELTASPH_VALIDATION_PLAN 5.2.3).
    dx = getattr(config, 'dx', None)
    particlesA = sampleRegularParticles(nx, config.domain, config.targetNeighbors, 0.0, 0,
                                        shortEdge = shortEdge, dx = float(dx) if dx is not None else None)

    mask = torch.ones_like(particlesA.masses, dtype = torch.bool)
    distances = particlesA.masses.new_ones(particlesA.masses.shape) * np.inf
    mask = torch.zeros_like(particlesA.masses, dtype = torch.bool)
        
    sdfDist, sdfNormal = sdf(particlesA.positions)
    maskA = sdfDist < 0
    mask = mask | maskA
    distances = torch.min(distances, sdfDist)
    if filter:
        particlesA = ParticleSet(
            positions = particlesA.positions[mask],
            supports = particlesA.supports[mask],
            masses = particlesA.masses[mask],
            densities = particlesA.densities[mask]            
        )
    else:
        mask = torch.ones_like(particlesA.masses, dtype = torch.bool)
        distances = particlesA.masses.new_ones(particlesA.masses.shape) * np.inf
        particlesA = ParticleSet(
            positions = particlesA.positions[mask],
            supports = particlesA.supports[mask],
            masses = particlesA.masses[mask],
            densities = particlesA.densities[mask]            
        )

    return particlesA, mask, distances

