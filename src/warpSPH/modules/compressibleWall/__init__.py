"""Solid-wall support for the compressible schemes (COMPRESSIBLE_WALLS_PLAN.md).

Wall rows are `kinds == 1`. Each evaluation they take rho, u from a
Shepard-normalised gather over the fluid rows, replaced by the star state of
the mirrored Riemann problem for the fluid's approach speed; the EOS then
gives their pressure. They move only with the velocity the case gave them.

Use from a scheme's right-hand side:

    wall = beginCompressibleWall(state, config)      # None without wall rows
    ... density ...
    if wall is not None:
        wall.apply(state, config, schemeConfig, adjacency)
        ... redo the density (the wall masses changed) ...
        wall.apply(state, config, schemeConfig, adjacency)
    ... forces, update ...
    wall.finishUpdate(update)
"""

from .wall import CompressibleWall, beginCompressibleWall, wallRiemannState

__all__ = ['CompressibleWall', 'beginCompressibleWall', 'wallRiemannState']
