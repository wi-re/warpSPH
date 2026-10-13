"""The fixed-shape grid the adaptive-support solve runs on (`warpSPHCore.radiusSearch.compactHash.slots`).

One grid per run, owned by the system (like the pair work, shared by reference through `initializeNewState`): the slots
are built once for a support RADIUS, every stage copies the stage's positions / supports into them and replays the
graph-captured build (no host read, no allocation, the same shapes). The solve limits every iterate to the radius, so
the grid cannot become invalid; `info['outgrown']` says when that limit was binding, and `ensureSupportGrid` then
makes a grid for a larger radius (the only host decision, read together with the step's other end-of-stage flag).
"""

from dataclasses import dataclass
from typing import Any, List, Optional

import torch

from warpSPHCore import SupportScheme
from warpSPHCore.radiusSearch.compactHash import GraphedBuild, makeHashMapSlots

__all__ = ['SupportGrid', 'ensureSupportGrid', 'refreshSupportGrid', 'noteOutgrown', 'growIfOutgrown']

#: the flags are read every this many steps; a binding limit for a few steps only clips h to the radius
#: (1.5x the largest support when the grid was made), it does not invalidate anything
CHECK_EVERY = 8

#: radius headroom over the largest support when a grid is made: the supports may grow by this factor before the
#: solve's limit binds and a larger grid is made
RADIUS_HEADROOM = 1.5


@dataclass
class SupportGrid:
    """The run's holder: shared by reference between the stage systems and the steps, filled in place when a grid is
    made (so a stage that makes a larger one makes it for the following stages and steps too)."""
    build: Optional[GraphedBuild] = None
    radius: float = 0.0
    particles: int = 0
    #: a larger grid is wanted: at least this radius
    minRadius: float = 0.0
    #: device flags 'the solve's limit to the radius was binding', read every `CHECK_EVERY` steps (not per stage)
    pending: Optional[List[torch.Tensor]] = None
    age: int = 0


def ensureSupportGrid(holder: SupportGrid, state, config) -> SupportGrid:
    """`holder` itself, with a grid in it that fits this state (one host read of the largest support when it has to
    make one)."""
    n = state.positions.shape[0]
    if holder.build is not None and holder.particles == n and holder.radius >= holder.minRadius:
        return holder
    radius = max(RADIUS_HEADROOM * float(state.supports.max()), holder.minRadius)
    slots = makeHashMapSlots(n, config.domain, radius, device=state.positions.device, dtype=state.positions.dtype,
                             mode=SupportScheme.Gather, hashMapLength=n + 1)
    holder.build, holder.radius, holder.particles = GraphedBuild(slots), radius, n
    return holder


def refreshSupportGrid(grid: SupportGrid, state):
    """The grid for the state's current positions and supports: copy into the static slots, replay the build."""
    slots = grid.build.slots
    slots.positions.copy_(state.positions)
    slots.supports.copy_(state.supports)
    return grid.build.replay()


def noteOutgrown(holder: SupportGrid, info: dict) -> None:
    """Remember this stage's 'limit was binding' device flag (no host read)."""
    flag = info.get('outgrown')
    if flag is not None:
        if holder.pending is None:
            holder.pending = []
        holder.pending.append(flag)


def growIfOutgrown(holder: SupportGrid) -> None:
    """Step end: every `CHECK_EVERY` steps one host read of the flags noted since; if the limit was binding the next
    stage makes a grid for a larger radius."""
    holder.age += 1
    if holder.age < CHECK_EVERY or not holder.pending:
        return
    binding = bool(torch.stack(holder.pending).any())
    holder.pending, holder.age = [], 0
    if binding:
        holder.minRadius = RADIUS_HEADROOM * holder.radius
