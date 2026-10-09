"""Dirichlet-style boundary conditions applied per-variable from a
`config.boundaryConditions` list of SDF-scoped functions: value overrides,
accumulated forcing, and update-tensor overrides.
"""

from .bcs import enforceDirichlet, computeForcing, enforceUpdates
from .pinned import PinnedBand, pinnedBandBC, pinnedKeepWeight, applyPinnedVelocity

__all__ = ['enforceDirichlet', 'computeForcing', 'enforceUpdates', 'PinnedBand', 'pinnedBandBC', 'pinnedKeepWeight', 'applyPinnedVelocity']