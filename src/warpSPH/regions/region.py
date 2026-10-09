"""Assemble a `ParticleRegion` from an SDF: sample particles inside it
(`sampleParticles`) and, for 2D configs, extract its boundary contour
(`find_contour`) for plotting. 3D configs get `contour=None`.
"""

from ..configurations.region import RegionType, ParticleRegion, BCType
from .sample import sampleParticles
from .contour import find_contour
import torch
from typing import Optional

__all__ = ['buildRegion']


def buildRegion(
    config, schemeConfig,
    sdf: callable,
    regionType: RegionType = RegionType.Fluid,
    initialConditions: dict = None,
    nGrid: int = 255,
    nx: Optional[int] = None,
    kind: Optional[BCType] = BCType.zeros,
    shortEdge: bool = True,
    representation = None
):
    nx_ = nx if nx is not None else config.nx

    # An analytic boundary (`representation`) is not sampled: the lattice is evaluated against a constant
    # positive distance, which keeps nothing and returns an empty set of the right dtype / device.
    samplingSdf = sdf if representation is None else (lambda p: (torch.ones(p.shape[0], device = p.device, dtype = p.dtype), None))

    return ParticleRegion(
        sdf = sdf,
        type = regionType,
        particles = sampleParticles(config, schemeConfig, samplingSdf, nx_, shortEdge = shortEdge)[0],
        contour = find_contour(config, schemeConfig, sdf, nGrid) if config.dim == 2 else None,
        initialConditions = initialConditions,
        kind = kind,
        representation = representation
    )
    

