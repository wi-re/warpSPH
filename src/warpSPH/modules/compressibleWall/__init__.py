"""Solid-wall support for the compressible schemes (COMPRESSIBLE_WALLS_PLAN.md).

Wall rows are `kinds == 1`. Each evaluation they take rho, u from a
Shepard-normalised gather over the fluid rows and a prescribed velocity; the
EOS then gives their pressure, and their update is zeroed.
"""

from .wall import applyCompressibleWall, zeroWallUpdate

__all__ = ['applyCompressibleWall', 'zeroWallUpdate']
