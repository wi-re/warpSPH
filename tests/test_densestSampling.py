"""`warpSPH.sample.sampleDensestParticles` -- the torch wrapper over
`warpSPHCore.sampling.sampleDensestLattice`.

The lattice geometry itself (commensurability, counts, nearest-neighbour
distances) is covered by warpSPHCore's `tests/sampling/`; this file pins the
wrapper's contract: device/dtype follow `domain`, positions land in the
domain, per-particle mass is the *achieved* box volume divided by the
*achieved* count (so the mass sum closes on the achieved volume, not the
requested one), and jitter is seed-reproducible.
"""
import math

import torch

from warpSPH.sample import sampleDensestParticles
from warpSPH.utils.domain import buildDomainDescription


def _minImageDistances(positions, box):
    d = positions[0] - positions
    d -= box * torch.round(d / box)
    dist = d.norm(dim=1)
    dist[0] = float('inf')
    return dist.min()


def test_2d_hexagonal():
    domain = buildDomainDescription(1.0, 2, periodic=True,
                                    device='cpu', dtype=torch.float64)
    ps = sampleDensestParticles(16, domain, targetNeighbors=16)
    assert ps.positions.shape == (16, 2)
    assert ps.positions.dtype == torch.float64
    box = domain.max - domain.min
    rel = ps.positions - domain.min
    assert torch.all(rel >= 0) and torch.all(rel < box)
    # achieved box: 1 x sqrt(3)/2 -> a = 1/4
    d = _minImageDistances(ps.positions, box)
    assert abs(d - 0.25) < 1e-12
    # mass closes on the *achieved* volume (1 * sqrt(3)/2), not 1
    assert abs(float(ps.masses.sum()) - math.sqrt(3.0) / 2.0) < 1e-12
    assert torch.all(ps.densities == 1.0)
    assert ps.supports[0] > 0


def test_3d_fcc():
    domain = buildDomainDescription(1.0, 3, periodic=True,
                                    device='cpu', dtype=torch.float64)
    ps = sampleDensestParticles(32, domain, targetNeighbors=16)
    assert ps.positions.shape == (32, 3)
    box = domain.max - domain.min
    rel = ps.positions - domain.min
    assert torch.all(rel >= 0) and torch.all(rel < box)
    # b = 1/2, nearest-neighbour distance b/sqrt(2)
    d = _minImageDistances(ps.positions, box)
    assert abs(d - 0.5 / math.sqrt(2.0)) < 1e-12
    assert abs(float(ps.masses.sum()) - 1.0) < 1e-12
    assert torch.all(ps.densities == 1.0)
    assert ps.supports[0] > 0


def test_dtype_follows_domain():
    domain = buildDomainDescription(1.0, 3, periodic=True,
                                    device='cpu', dtype=torch.float32)
    ps = sampleDensestParticles(32, domain, targetNeighbors=16)
    for field in (ps.positions, ps.supports, ps.masses, ps.densities):
        assert field.dtype == torch.float32


def test_jitter_reproducible():
    domain = buildDomainDescription(1.0, 3, periodic=True,
                                    device='cpu', dtype=torch.float64)
    a = sampleDensestParticles(32, domain, 16, jitter=0.1, seed=42)
    b = sampleDensestParticles(32, domain, 16, jitter=0.1, seed=42)
    c = sampleDensestParticles(32, domain, 16, jitter=0.1, seed=43)
    assert torch.equal(a.positions, b.positions)
    assert not torch.equal(a.positions, c.positions)
