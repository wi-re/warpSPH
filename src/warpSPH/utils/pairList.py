"""The super-symmetric pair list of one scheme stage: the pair-force list, for this stage's positions and (new) supports.

`stagePairList` is the second half of the stage's neighbour work (the first is `adaptiveSupport.solveStageSupport`). Three cases:

* list mode (`onGrid=False`): the support solve already validated and, if needed, rebuilt `system.adjacency` against the particles and every
  support it used, so the list is simply returned. This is the whole path of the Monaghan step.
* grid mode, optimistic stage (CompSPH graph step, a `PairWork` that is not conservative and a list that fits the particles): no host
  read. The list was validated, with slack, for the positions the step starts from; whether it also reaches this stage's positions and new supports is a
  device flag appended to `pairWork.flags`, read once at the end of the step. A failing flag repeats the step conservatively.
* grid mode otherwise: the explicit validity check and (re)build, through the fixed-shape list grid and the list capacity of the scheme config.
"""

from warpSPHCore import SupportScheme, buildVerletList, verletStaleFlag

__all__ = ['stagePairList']


def stagePairList(system, state, config, schemeConfig, *, onGrid: bool = False, rebuildSlack: float = 0.0):
    if not onGrid:
        return system.adjacency
    from ..modules.compSPH.pairWork import listFitsParticles                                    # (CompSPH's grid-mode machinery; imported when used)
    from ..modules.compSPH.supportGrid import pairListCapacity, pairListGridFactory
    pairWork = system.pairWork
    if pairWork is not None and not pairWork.conservative and listFitsParticles(system.adjacency, state):
        adjacency = system.adjacency
        pairWork.flags.append(verletStaleFlag(state.positions, state.supports, adjacency, config.domain, config.verletScale))
        return adjacency
    adjacency = buildVerletList(
        state, config.domain, verletScale = config.verletScale, supportMode = SupportScheme.SuperSymmetric,
        priorNeighborhood = None if (pairWork is not None and pairWork.rebuildList) else system.adjacency,
        verbose = False, rebuildSlack = rebuildSlack, capacity = pairListCapacity(schemeConfig),
        gridFactory = pairListGridFactory(system.listGrid, state, config))
    system.adjacency = adjacency
    return adjacency
