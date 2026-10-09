"""Boundary providers: analytic boundary integrals (warpSPHBoundaries) as an alternative to boundary particles.

A boundary region with a `representation` (`configurations/region.py`) is not sampled into particles; its
geometry is served by a provider (`provider.py`) that returns, at the fluid particles, the kernel integrals
over the solid (`lam`, `G`, `A`, `lap`, `cover`, `tens`, ...) and the signed distance. The scheme keeps the
boundary-condition physics: `modules/analyticBoundary/` turns these aggregates into the wall terms of the
delta+ stages (continuity, viscosity, pressure force, surface detection, shifting, no-penetration).
"""
from .provider import BoundaryProvider, bindBodies, buildBoundaryProvider

__all__ = ['BoundaryProvider', 'bindBodies', 'buildBoundaryProvider']
