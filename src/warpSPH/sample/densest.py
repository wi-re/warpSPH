"""Densest-packing lattice sampler (1D uniform / 2D hexagonal / 3D FCC).

Thin torch wrapper over `warpSPHCore.sampling.sampleDensestLattice` (numpy,
float64, host side), returning a `ParticleSet` in the same shape
`sampleRegularParticles` hands out: uniform unit density, per-particle mass
equal to the volume each particle actually occupies (the *achieved* box
divided by the *achieved* count), and the support derived from that cell
volume via `volumeToSupport`.

The core snaps the particle count -- and, in 2D, the box y size -- to the
lattice geometry (N = 4 sx sy; N = 4 p^3; see the core's module docstring
for the periodic-commensurability rules). The positions are built in
[0, box) and shifted onto `domain.min`; when the achieved box differs from
the requested one the achieved sizes are what the mass/support are computed
from, so a snapped 2D box stays self-consistent.
"""

from ..geometry import ParticleSet
from ..utils.domain import DomainDescription
from ..utils.support import volumeToSupport
from warpSPHCore.sampling import sampleDensestLattice

import torch

__all__ = ['sampleDensestParticles']


def sampleDensestParticles(nx: int, domain: DomainDescription,
                           targetNeighbors: int,
                           jitter: float = 0.0,
                           seed: int | None = None):
    L = domain.max[0] - domain.min[0]
    sample = sampleDensestLattice(nx, float(L), domain.dim,
                                  jitter=jitter, seed=seed)
    box = torch.as_tensor(sample.box, device=domain.min.device,
                          dtype=domain.min.dtype)
    positions = torch.as_tensor(sample.positions,
                                device=domain.min.device,
                                dtype=domain.min.dtype)
    positions = positions + domain.min
    cellVolume = box.prod() / sample.count
    support = volumeToSupport(cellVolume, targetNeighbors, domain.dim)
    ones = torch.ones_like(positions[:, 0])
    return ParticleSet(positions=positions,
                       supports=ones * support,
                       masses=ones * cellVolume,
                       densities=ones)
