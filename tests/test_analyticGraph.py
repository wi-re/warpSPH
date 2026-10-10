"""Stage 3 E8 (ANALYTIC_BOUNDARIES_PLAN.md): the analytic-wall step replayed from a CUDA graph (`CaseSpec.cudaGraph`, `utils/cudaGraph.py: GraphedIntegratorStep`) equals the eager step bit for bit: the state and the
booked wall loads. The graph needs a sync-free step (the wall closure's boolean-mask gathers, the host reads of the calibration / periodic flags / body force / mean pressure were removed) and a device `dt`; a spinning
axisymmetric body (`analyticSpinAxisymmetric`) keeps its pose. A stale `rb.load` (a fresh tensor assigned inside the captured step) once left the graphed torque constant (2026-10-10).
"""
import numpy as np
import pytest
import torch

pytest.importorskip('warpSPHBoundaries')

from warpSPH.cases import importAll  # noqa: E402
from warpSPH.runner import getCase, run  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA')


def both(caseName, nx, nSteps, params):
    importAll()
    case = getCase(caseName)
    out = {}
    for g in (False, True):
        r = run(case, scheme='deltaSPH', nx=nx, nSteps=nSteps, quiet=True, store=False, progress=False, plot=False, video=False, show=False, params={**case.params, **params}, cudaGraph=g, kernel='Wendland2')
        st = r.state.state
        sg = r.ctx.scratch.get('stepGraph')
        out[g] = dict(x=st.positions.clone(), v=st.velocities.clone(), rho=st.densities.clone(), graph=sg,
                      loads=[rb.load.clone() for rb in r.ctx.schemeConfig.boundaryProvider.rigidBodies])
    return out


@pytest.mark.parametrize('caseName, nx, params', [
    ('taylorCouette', 16, {'fluidViscosity': 'morris', 'wallViscosityClosure': 'noslipMoment', 'nuReference': 0.0274}),
    ('stokesArray', 24, {'fluidViscosity': 'morris', 'wallViscosityClosure': 'noslipMoment', 'pressureConsistent': True}),
    ('cylinderWake', 90, {'fluidViscosity': 'morris', 'wallViscosityClosure': 'noslipMoment', 'targetDt': 2e-2}),          # the pinned band: the stream set at the end of the step inside the graph
])
def test_the_graphed_step_equals_the_eager_step(caseName, nx, params):
    o = both(caseName, nx, 120, params)
    sg = o[True]['graph']
    assert sg is not None and sg.disabled is None and sg.replays > 50, (None if sg is None else (sg.captures, sg.replays, sg.disabled))
    for key in ('x', 'v', 'rho'):
        assert bool((o[False][key] == o[True][key]).all()), key
    for a, b in zip(o[False]['loads'], o[True]['loads']):
        assert bool((a == b).all())
