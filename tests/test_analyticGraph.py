"""CUDA graphs with analytic boundaries: the whole step (right-hand side, shifting, no-penetration, finalize) is captured and replayed, bitwise the eager step, for the tank, the tank with an
analytic obstacle and the Michel shifting; configurations that cannot be captured (implicit shifting: host-synchronising Krylov solvers; moving or dynamic analytic bodies, which every execution of
the step advances in place and the graph's validation executes more than once; particle bodies of a mixed scene) are refused up front instead of capturing."""
import numpy as np
import pytest
import torch

pytest.importorskip('warpSPHBoundaries')

from warpSPH.cases import importAll  # noqa: E402
from warpSPH.runner import getCase, run  # noqa: E402
from test_analyticDambreak import spec_of  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA')

OBSTACLE = dict(obstacleActive=True, obstacleType='circleBottom', maxExtent=0.5, offsetX=-0.4)


def final(case, graph, params, nSteps=100):
    res = run(case, spec_of(case, 'analytic', nSteps=nSteps, params=params, cudaGraph=graph, plot=False, store=False, progress=False, video=False, show=False, quiet=True, pipelineOutputs=False))
    s = res.state.state
    f = s.kinds == 0
    return res, [a[f].double().cpu().numpy() for a in (s.positions, s.velocities, s.densities)]


@pytest.mark.parametrize('params', [{}, OBSTACLE, dict(shiftScheme='michel2022', shiftProjection='michel2022')], ids=['tank', 'obstacle', 'michel'])
def test_whole_step_graph_is_bitwise_the_eager_step(params):
    importAll()
    case = getCase('dambreak')
    _, eager = final(case, False, params)
    res, graphed = final(case, True, params)
    sg = res.ctx.scratch.get('stepGraph')
    assert sg is not None and sg.captures >= 1 and getattr(sg, 'disabled', None) is None, 'the step was not captured'
    for a, b in zip(eager, graphed):
        assert np.array_equal(a, b)


@pytest.mark.parametrize('params', [dict(shiftScheme='implicit', shiftProjection='surfaceNormal'), dict(OBSTACLE, obstacleDynamic=True),
                                    dict(OBSTACLE, obstacleRepresentation='particles')], ids=['implicit', 'dynamic-body', 'mixed'])
def test_uncapturable_configurations_run_eagerly(params):
    importAll()
    case = getCase('dambreak')
    _, eager = final(case, False, params, nSteps=40)
    res, graphed = final(case, True, params, nSteps=40)
    assert res.ctx.scratch.get('stepGraph') is None
    for a, b in zip(eager, graphed):
        assert np.allclose(a, b, rtol=0, atol=5e-4 * max(1.0, float(np.abs(a).max())))         # the implicit solve's atomics: two identical eager runs differ by 2e-5 in the velocity (float32, 40 steps)
