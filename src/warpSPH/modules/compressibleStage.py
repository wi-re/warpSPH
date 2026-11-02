"""Stage-dependent wall, support and neighbor preparation for CompSPH/CRKSPH.

The ordinary RHS calls this directly. An optional evaluation-resource session
runs it in ``CompSPHSystem.preprocess`` and hands that same result to the RHS.
No supports or equation values are reused between numerical evaluations.
"""
from contextvars import ContextVar
from dataclasses import dataclass

from warpSPHCore.profiling import record_function

from .adaptiveSupport import solveStageSupport
from .compSPH.pairWork import startStage, stageRebuildSlack
from .compSPH.supportGrid import solvesOnGrid
from .compressibleWall import beginCompressibleWall
from ..utils.pairList import stagePairList

_ACTIVE = ContextVar('warpsph_compressible_preparation', default=None)


@dataclass(frozen=True)
class PreparedCompressibleStage:
    wall: object
    adjacency: object


def _prepare(system, config, schemeConfig, stageIndex, label, *, listBuilder=None):
    state = system.state
    onGrid = solvesOnGrid(schemeConfig)
    slack = stageRebuildSlack(system, stageIndex)
    supportLabel, pairLabel = ('02', '02b') if label == 'crkSPH' else ('02', '03')
    with record_function(f'[warpSPH] - [{label} - 01] wall'):
        wall = beginCompressibleWall(state, config)
        startStage(system, state, config, stageIndex)
    with record_function(f'[warpSPH] - [{label} - {supportLabel}] optimal support'):
        options = {} if listBuilder is None else dict(listBuilder=listBuilder)
        solveStageSupport(system, state, config, schemeConfig, onGrid=onGrid, rebuildSlack=slack, **options)
    with record_function(f'[warpSPH] - [{label} - {pairLabel}] adjacency'):
        adjacency = stagePairList(system, state, config, schemeConfig, onGrid=onGrid, rebuildSlack=slack)
    return PreparedCompressibleStage(wall, adjacency)


def prepareCompressibleStage(system, config, schemeConfig, stageIndex=None, *, label='compSPH'):
    """Return this evaluation's preparation, preserving the legacy call order."""
    active = _ACTIVE.get()
    if active is not None:
        return active.consume(system, config, schemeConfig, stageIndex)
    return _prepare(system, config, schemeConfig, stageIndex, label)


def preprocessCompressibleStage(system, config, schemeConfig, stageIndex=None):
    """Prepare outside the equation callback only when explicitly enabled."""
    active = _ACTIVE.get()
    if active is not None:
        active.prepare(system, config, schemeConfig, stageIndex)
