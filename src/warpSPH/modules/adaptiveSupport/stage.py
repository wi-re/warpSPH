"""The adaptive support solve of one scheme stage, in the two ways a step can run it.

`solveStageSupport` is what a compressible step does first: solve for the supports `h` (the Owen / Monaghan / number-density scheme of
`schemeConfig.adaptiveSupportScheme`), and leave `state.supports` and `state.densities` set. Which neighbour structure the solve looks at
depends on how the step runs:

* list mode (default): the solve walks the Verlet list of the system, validates it ONCE against the particles and every support it
  used (`validateOnce`, a device-side convergence test; `rebuildSlack` rebuilds early at stage 0 of a multi-stage tableau), and leaves
  the list on `system.adjacency`, valid for the new supports: the step needs no check of its own afterwards.
* grid mode (`onGrid`, CompSPH's graph step): the solve runs on the system's fixed-shape support grid (a graph-replayed build, no host
  read) and looks at no list at all; the pair list is then built or checked by `utils.pairList.stagePairList`.

The densities the solve returns are the particles' current ones (`computeDensity=False`): every caller sums its own density next.
"""

from warpSPHCore import SupportScheme, buildVerletList

from ...enumTypes import AdaptiveSupportScheme
from .optimalSupport import evaluateOptimalSupport

__all__ = ['solveStageSupport']


def solveStageSupport(system, state, config, schemeConfig, *, onGrid: bool = False, rebuildSlack: float = 0.0) -> None:
    """Solve for the stage's supports; sets `state.supports` / `state.densities` (and, in list mode, `system.adjacency`)."""
    if onGrid:
        from ..compSPH.supportGrid import ensureSupportGrid, noteOutgrown, refreshSupportGrid       # (the grid belongs to the CompSPH step; not needed in list mode)
        holder = ensureSupportGrid(system.supportGrid, state, config)
        info = {}
        rho, h, *_ = evaluateOptimalSupport(
            state, config, schemeConfig, SupportScheme.Gather, None, computeDensity=False,
            grid=refreshSupportGrid(holder, state), gridRadius=holder.radius, info=info)
        noteOutgrown(holder, info)
        state.supports, state.densities = h, rho
        return
    rho, h, system.adjacency, *_ = evaluateOptimalSupport(
        state, config, schemeConfig, SupportScheme.Gather, system.adjacency,
        validateOnce=True, rebuildSlack=rebuildSlack, computeDensity=False)
    state.supports, state.densities = h, rho
    if schemeConfig.adaptiveSupportScheme in (AdaptiveSupportScheme.Monaghan, AdaptiveSupportScheme.NumberDensity):
        # those solvers ignore `validateOnce`: validate the list here, for the supports they returned
        system.adjacency = buildVerletList(
            state, config.domain, verletScale=config.verletScale, supportMode=SupportScheme.SuperSymmetric,
            priorNeighborhood=system.adjacency, verbose=False, rebuildSlack=rebuildSlack)
