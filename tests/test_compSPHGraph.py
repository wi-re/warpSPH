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
        if graph:
            ctx.schemeConfig.stepMode = 'graph'
        else:
            ctx.schemeConfig.supportSolveOnGrid = True      # the eager step on the same grid solve: the bitwise reference
        return out
    graphs = []
    original = cudaGraph.GraphedIntegratorStep.__init__

    def remember(self, *a, **k):
        original(self, *a, **k)
        graphs.append(self)
    cudaGraph.GraphedIntegratorStep.__init__ = remember
    try:
        result = run(dataclasses.replace(base, configureScheme=configure), quiet=True, progress=False, nSteps=steps,
                     scheme='CompSPH')
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


def _stepModeRun(caseName, mode, steps=3, **schemeSet):
    base = getCase(caseName)
    inner = base.configureScheme

    def configure(ctx, *a, **k):
        out = inner(ctx, *a, **k) if inner else None
        ctx.schemeConfig.stepMode = mode
        for key, value in schemeSet.items():
            setattr(ctx.schemeConfig, key, value)
        return out
    graphs = []
    original = cudaGraph.GraphedIntegratorStep.__init__

    def remember(self, *a, **k):
        original(self, *a, **k)
        graphs.append(self)
    cudaGraph.GraphedIntegratorStep.__init__ = remember
    try:
        result = run(dataclasses.replace(base, configureScheme=configure), quiet=True, progress=False, nSteps=steps,
                     scheme='CompSPH')
    finally:
        cudaGraph.GraphedIntegratorStep.__init__ = original
    return result, graphs


@needsCuda
@pytest.mark.parametrize('mode, caseName, expectGraph', [
    ('eager', 'sod2d', False),
    ('graph', 'sod2d', True),
    ('auto', 'sod2d', True),           # 2,500 particles: below the 2D limit
    ('auto', 'sod3d', False),          # the 3D limit is 0: never
])
def test_stepModePicksTheStepRoute(mode, caseName, expectGraph):
    result, graphs = _stepModeRun(caseName, mode, steps=4)
    assert bool(graphs) == expectGraph
    assert result.nSteps == 4


@needsCuda
def test_autoFollowsTheParticleCountLimit():
    _, below = _stepModeRun('sod2d', 'auto', stepModeMaxParticles=10_000)
    _, above = _stepModeRun('sod2d', 'auto', stepModeMaxParticles=1_000)
    assert below and not above


@needsCuda
def test_graphRequestThatCannotBeHonouredRunsEagerlyWithAWarning():
    with pytest.warns(UserWarning, match="stepMode='graph' runs eagerly"):
        result, graphs = _stepModeRun('sedovWalls', 'graph', steps=3)
    assert not graphs and result.nSteps == 3


def _rayleighTaylorCompSPH(steps, graph, monkeypatch=None):
    """rayleighTaylor on CompSPH (Dirichlet BCs: their host-sized boolean-mask index is not allowed while a graph is recorded)."""
    base = getCase('rayleighTaylor')
    inner = base.configureScheme

    def configure(ctx, *a, **k):
        out = inner(ctx, *a, **k) if inner else None
        if graph:
            ctx.schemeConfig.stepMode = 'graph'
        else:
            ctx.schemeConfig.supportSolveOnGrid = True      # the eager step on the same grid solve
            ctx.schemeConfig.inPlaceAdjacency = True
        return out
    return run(dataclasses.replace(base, configureScheme=configure), quiet=True, progress=False, nSteps=steps,
               scheme='CompSPH')


@needsCuda
def test_captureThatCudaInvalidatesLeavesTheNextEagerStepsIntact(monkeypatch):
    """A capture CUDA itself invalidates (`cudaErrorStreamCaptureUnsupported`, here the Dirichlet BC's mask index) used to leave torch on
    the capture's side stream (`torch.cuda.graph.__exit__` raises before it restores it) while warp was back on its own: the eager steps
    after it raced across two streams and went non-finite. They must now be the plain eager steps."""
    import warpSPH.modules.compSPH.stepMode as stepMode
    before = torch.cuda.current_stream()
    monkeypatch.setattr(stepMode, '_blocker', lambda ctx, runningState, spec: None)       # let the unsupported capture be attempted
    with pytest.warns(UserWarning, match='capture failed'):
        graphed = _rayleighTaylorCompSPH(8, graph=True)
    assert torch.cuda.current_stream() == before
    monkeypatch.undo()
    eager = _rayleighTaylorCompSPH(8, graph=False)
    for name, a in _final(eager).items():
        assert torch.isfinite(_final(graphed)[name]).all(), name
        assert torch.equal(a, _final(graphed)[name]), name
