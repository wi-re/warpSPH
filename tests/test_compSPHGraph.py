"""The CompSPH step replayed from a CUDA graph (`utils/cudaGraph.py:GraphedIntegratorStep`, grid mode): the same states
as the eager step, with the pair list rebuilt IN PLACE between replays and one host read per replay."""

import dataclasses

import pytest
import torch

import warpSPH.modules.compSPH.supportGrid as supportGrid
import warpSPH.utils.cudaGraph as cudaGraph
from warpSPH.cases import importAll
from warpSPH.runner import run
from warpSPH.runner.case import getCase
from warpSPHCore import ParticleState, SupportScheme, buildVerletList
from warpSPHCore.util import generateNeighborTestData

importAll()
needsCuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA graphs')


def _sod2dOnGrid(steps, graph):
    base = getCase('sod2d')
    inner = base.configureScheme

    def configure(ctx, *a, **k):
        out = inner(ctx, *a, **k) if inner else None
        ctx.schemeConfig.supportSolveOnGrid = True
        return out
    graphs = []
    original = cudaGraph.GraphedIntegratorStep.__init__

    def remember(self, *a, **k):
        original(self, *a, **k)
        graphs.append(self)
    cudaGraph.GraphedIntegratorStep.__init__ = remember
    try:
        result = run(dataclasses.replace(base, configureScheme=configure), quiet=True, progress=False, nSteps=steps,
                     scheme='CompSPH', cudaGraph=graph)
    finally:
        cudaGraph.GraphedIntegratorStep.__init__ = original
    return result, graphs


def _final(result):
    s = result.state.state
    return {n: getattr(s, n).detach().cpu() for n in ('positions', 'velocities', 'internalEnergies', 'densities', 'supports')}


@needsCuda
def test_graphedStepsAreTheEagerStepsAndRebuildInPlace():
    eager, _ = _sod2dOnGrid(120, graph=False)
    graphed, graphs = _sod2dOnGrid(120, graph=True)
    [step] = graphs
    assert step.disabled is None and step.replays > 100 and step.fallbacks == 0
    assert step.captures <= 3                                  # a rebuild keeps the buffers: no recapture per rebuild
    for name, a in _final(eager).items():
        assert torch.equal(a, _final(graphed)[name]), name


@needsCuda
def test_stepsThatRanOutOfListAreRedoneEagerlyAndStillMatch(monkeypatch):
    """No rebuild margin: stages run out of list (device flags), the replay is discarded and the step repeated eagerly."""
    monkeypatch.setattr(supportGrid, 'USAGE_GUARD', 0.0)
    monkeypatch.setattr(supportGrid, 'MIN_STEP_USAGE', 1e-6)
    eager, _ = _sod2dOnGrid(120, graph=False)
    graphed, graphs = _sod2dOnGrid(120, graph=True)
    [step] = graphs
    assert step.disabled is None and step.fallbacks > 0
    for name, a in _final(eager).items():
        assert torch.equal(a, _final(graphed)[name]), name


@pytest.mark.parametrize('periodic', [False, True])
def test_pairListGridFindsTheSymmetricPairs(periodic):
    """The grid a pair-list rebuild searches must be a SuperSymmetric one: a Gather grid drops the pairs only the
    neighbour's (larger) support reaches (3.6 % of the edges of a Sod step)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    positions, supports, n, domain, dx = generateNeighborTestData(14, 20, 2, periodic, device)
    g = torch.Generator().manual_seed(3)
    supports = supports * (1.0 + 0.5 * torch.rand(n, generator=g)).to(device)       # clearly unequal supports
    state = ParticleState(positions=positions, supports=supports, masses=torch.ones(n, device=device),
                          densities=torch.ones(n, device=device), kinds=torch.zeros(n, dtype=torch.int32, device=device))
    config = type('Config', (), {'domain': domain, 'verletScale': 1.3})()
    holder = supportGrid.SupportGrid()
    gridded = buildVerletList(state, domain, verletScale=1.3, supportMode=SupportScheme.SuperSymmetric,
                              gridFactory=supportGrid.pairListGridFactory(holder, state, config), forceRebuild=True)
    plain = buildVerletList(state, domain, verletScale=1.3, supportMode=SupportScheme.SuperSymmetric)
    assert gridded.numEdges == plain.numEdges
    assert torch.equal(gridded.numNeighbors, plain.numNeighbors)
    pairs = lambda adj: sorted(zip(*(t.tolist() for t in adj.livePairs())))
    assert pairs(gridded) == pairs(plain)
