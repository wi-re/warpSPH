"""The CRKSPH step replayed from a CUDA graph (`CompSPHConfig.stepMode`, shared with CompSPH: `modules/compSPH/stepMode.py`): the same states as the eager step
on the same fixed-shape grid solve, the pair list rebuilt in place between replays, steps that ran out of list repeated eagerly, and the route choice."""

import dataclasses

import pytest
import torch

import warpSPH.modules.compSPH.supportGrid as supportGrid
import warpSPH.utils.cudaGraph as cudaGraph
from warpSPH.cases import importAll
from warpSPH.runner import run
from warpSPH.runner.case import getCase

importAll()
needsCuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA graphs')


def _run(caseName, steps, mode, **schemeKw):
    """`mode`: 'graph' / 'auto' (stepMode), or 'gridEager' (the eager step on the same grid solve: the bitwise reference)."""
    base = getCase(caseName)
    inner = base.configureScheme

    def configure(ctx, *a, **k):
        out = inner(ctx, *a, **k) if inner else None
        if mode == 'gridEager':
            ctx.schemeConfig.supportSolveOnGrid = True
            ctx.schemeConfig.inPlaceAdjacency = True
        else:
            ctx.schemeConfig.stepMode = mode
        for key, value in schemeKw.items():
            setattr(ctx.schemeConfig, key, value)
        return out
    graphs = []
    original = cudaGraph.GraphedIntegratorStep.__init__

    def remember(self, *a, **k):
        original(self, *a, **k)
        graphs.append(self)
    cudaGraph.GraphedIntegratorStep.__init__ = remember
    try:
        result = run(dataclasses.replace(base, configureScheme=configure), quiet=True, progress=False, nSteps=steps, scheme='CRKSPH')
    finally:
        cudaGraph.GraphedIntegratorStep.__init__ = original
    return result, graphs


def _final(result):
    s = result.state.state
    return {n: getattr(s, n).detach().cpu() for n in ('positions', 'velocities', 'internalEnergies', 'densities', 'supports')}


@needsCuda
@pytest.mark.parametrize('caseName, steps', [('sod2d', 120), ('gresho', 60), ('sod', 80)])
def test_graphedCRKStepsAreTheEagerGridSteps(caseName, steps):
    eager, _ = _run(caseName, steps, 'gridEager')
    graphed, graphs = _run(caseName, steps, 'graph')
    [step] = graphs
    assert step.disabled is None and step.replays > steps // 2 and step.fallbacks == 0
    for name, a in _final(eager).items():
        assert torch.equal(a, _final(graphed)[name]), name


@needsCuda
def test_crkStepsThatRanOutOfListAreRedoneEagerlyAndStillMatch(monkeypatch):
    monkeypatch.setattr(supportGrid, 'USAGE_GUARD', 0.0)
    monkeypatch.setattr(supportGrid, 'MIN_STEP_USAGE', 1e-6)
    eager, _ = _run('sod2d', 120, 'gridEager')
    graphed, graphs = _run('sod2d', 120, 'graph')
    [step] = graphs
    assert step.disabled is None and step.fallbacks > 0
    for name, a in _final(eager).items():
        assert torch.equal(a, _final(graphed)[name]), name


@needsCuda
@pytest.mark.parametrize('mode, caseName, expectGraph', [
    ('eager', 'sod2d', False),
    ('graph', 'sod2d', True),
    ('auto', 'sod2d', True),           # 2,500 particles: below the 2D limit
    ('auto', 'sod3d', False),          # the 3D limit is 0: never
])
def test_crkStepModePicksTheRoute(mode, caseName, expectGraph):
    result, graphs = _run(caseName, 4, mode)
    assert bool(graphs) == expectGraph and result.nSteps == 4


@needsCuda
def test_crkGraphRequestWithWallsRunsEagerlyWithAWarning():
    with pytest.warns(UserWarning, match="stepMode='graph' runs eagerly"):
        result, graphs = _run('sedovWalls', 3, 'graph')
    assert not graphs and result.nSteps == 3
