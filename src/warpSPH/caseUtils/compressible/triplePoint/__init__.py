"""Case utilities for `cases/compressible/triplePoint.py`'s three-region shock
interaction: an equal-mass sampler (light region sampled coarser, avoiding a
density-jump artifact at the region boundary) and an equal-resolution one
(single lattice, masses set from the region densities afterwards)."""

from .equalMass import sampleTriplePointEqualMass
from .equalResolution import sampleTriplePointEqualResolution

__all__ = ['sampleTriplePointEqualMass', 'sampleTriplePointEqualResolution']