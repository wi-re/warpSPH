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

from warpSPHCore import CapacityPolicy, SupportScheme
from warpSPHCore.dataTypes.adjacency_t import DEFAULT_CAPACITY
from warpSPHCore.radiusSearch.compactHash import GraphedBuild, makeHashMapSlots

__all__ = ['solvesOnGrid', 'pairListCapacity', 'SupportGrid', 'ensureSupportGrid', 'refreshSupportGrid', 'noteOutgrown', 'growIfOutgrown',
           'pairListGridFactory', 'takeListGridFlags', 'rebuildNeeded']

#: the flags are read every this many steps; a binding limit for a few steps only clips h to the radius
#: (1.5x the largest support when the grid was made), it does not invalidate anything
CHECK_EVERY = 8

#: radius headroom over the largest support when a grid is made: the supports may grow by this factor before the
#: solve's limit binds and a larger grid is made
RADIUS_HEADROOM = 1.5


def solvesOnGrid(schemeConfig) -> bool:
    """The Owen support solve runs on the fixed-shape grid (and the step's pair list is checked by deferred flags)."""
    from ...enumTypes import AdaptiveSupportScheme
    return schemeConfig.adaptiveSupportScheme == AdaptiveSupportScheme.Owen and getattr(schemeConfig, 'supportSolveOnGrid', False)


def pairListCapacity(schemeConfig):
    """The capacity policy of the step's pair-list rebuilds: in place (same buffers) when the whole step is graphed."""
    return CapacityPolicy(inPlace=True) if getattr(schemeConfig, 'inPlaceAdjacency', False) else DEFAULT_CAPACITY


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
    #: the pair list's Verlet-buffer usage at the last step end, and the usage a step adds (host floats, estimated)
    lastUsage: float = 0.0
    perStep: float = 0.3


def ensureSupportGrid(holder: SupportGrid, state, config, scale: float = 1.0,
                      mode: SupportScheme = SupportScheme.Gather) -> SupportGrid:
    """`holder` itself, with a grid in it that fits this state (one host read of the largest support when it has to
    make one). `mode` is the support mode of the searches the grid serves: the search takes it from the grid (the
    support solve gathers; the pair list is SuperSymmetric)."""
    n = state.positions.shape[0]
    if holder.build is not None and holder.particles == n and holder.radius >= holder.minRadius:
        return holder
    radius = max(RADIUS_HEADROOM * scale * float(state.supports.max()), holder.minRadius)
    slots = makeHashMapSlots(n, config.domain, radius, device=state.positions.device, dtype=state.positions.dtype,
                             mode=mode, hashMapLength=n + 1)
    holder.build, holder.radius, holder.particles = GraphedBuild(slots), radius, n
    return holder


def refreshSupportGrid(grid: SupportGrid, state, scale: float = 1.0):
    """The grid for the state's current positions and supports: copy into the static slots, replay the build."""
    slots = grid.build.slots
    slots.positions.copy_(state.positions)
    slots.supports.copy_(state.supports if scale == 1.0 else state.supports * scale)
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


def pairListGridFactory(holder: SupportGrid, state, config):
    """A callable giving the grid a pair-list rebuild searches (`buildVerletList(gridFactory=...)`): built for the
    Verlet-scaled supports, `holder`'s own grid (one more per run), the graph-replayed fixed-shape build instead of the
    eager hash-map build (4 host syncs, ~160 ops). The grid is made for a radius with headroom over the scaled largest
    support; whether that still held for the supports it was built from is a device flag, kept in `holder.pending`
    and read with the step's other end-of-step flags (`takeListGridFlags`)."""
    scale = config.verletScale

    def make():
        ensureSupportGrid(holder, state, config, scale=scale, mode=SupportScheme.SuperSymmetric)
        grid = refreshSupportGrid(holder, state, scale=scale)
        if holder.pending is None:
            holder.pending = []
        holder.pending.append(~holder.build.slots.fits)
        return grid
    return make


def takeListGridFlags(holder: SupportGrid) -> List[torch.Tensor]:
    """The 'the pair-list grid was too small for the supports' device flags noted since the last call (cleared)."""
    flags, holder.pending = (holder.pending or []), []
    return flags


#: a step is allowed to use this multiple of the usage the previous steps showed (and of the first, assumed, 0.3)
#: before the list is rebuilt for it; a stage that still runs out is caught by its device flag (a second pass)
USAGE_GUARD = 1.25
MIN_STEP_USAGE = 0.02


def rebuildNeeded(holder: SupportGrid, usage: float, rebuilt: bool = False) -> bool:
    """Step end: `usage` is the fraction of the Verlet buffer the list has used (read with the step's other flags).
    Learns how much one step adds and says whether the NEXT step would run the list out, i.e. whether to rebuild it now.
    `holder` is the list grid's holder, which keeps the host-side estimate."""
    step = usage - holder.lastUsage
    if step > 0:
        holder.perStep = max(MIN_STEP_USAGE, 0.5 * holder.perStep + 0.5 * step)
    holder.lastUsage = usage
    need = usage + USAGE_GUARD * holder.perStep >= 1.0
    if need:
        holder.lastUsage = 0.0          # the fresh list starts unused
    return need
