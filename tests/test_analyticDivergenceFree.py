"""`divergenceFree` (DFSPH) on analytic walls (EXPERIMENTAL, OPEN_PROBLEMS 31): the wall terms of its step (density, free-surface detector, velocity diffusion, the omni pressure solves, the analytic no-penetration shift,
the XSPH / friction filters) come together on a still tank and a short dam break.
"""
import os

import numpy as np
import pytest
import torch

pytest.importorskip('warpSPHBoundaries')

from warpSPH.cases import importAll  # noqa: E402
from warpSPH.runner import getCase, run  # noqa: E402
from warpSPH.runner.caseSpec import CaseSpec  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA')

ROLL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'examples', 'sloshingTank', 'SPHERIC_TestCase10', 'data_files', 'lateral_water_1x.txt')


def tank(params=None, nSteps=60, nh=2.57):
    importAll()
    case = getCase('sloshingTank')
    spec = CaseSpec(caseName='dfsph', scheme='divergenceFree', params={**case.params, 'wallRepresentation': 'analytic', 'rollDataFile': ROLL, 'rollStartTime': 100.0, **(params or {})}).merged(**case.defaults).merged(
        scheme='divergenceFree', integrationScheme='semiImplicitEuler', kernel='Wendland2', supportMode='SuperSymmetric', cflFactor=0.2, dt=1e-3, maxDt=2e-3, nx=100, n_h=nh, nSteps=nSteps,
        plot=False, store=False, progress=False, video=False, show=False, quiet=True)
    with pytest.warns(UserWarning, match='EXPERIMENTAL'):
        return run(case, spec)


@pytest.mark.parametrize('nh', [2.57, 4.0])
def test_still_tank_stays_quiet(nh):
    res = tank(nh=nh)
    assert float(np.nanmax(res.series('maxVelocity'))) < 0.5
    st = res.state.state
    assert float(st.densities[st.kinds == 0].max()) < 1.02


def test_filters_and_mls_wall_pressure_run_with_divergence_free():
    res = tank({'xsphCoefficient': 1e-4, 'boundaryFriction': 5e-3, 'analyticWallPressure': 'mls'}, nSteps=40)
    assert not res.diverged
    assert float(np.nanmax(res.series('maxVelocity'))) < 1.5


def test_analytic_no_penetration_acts_on_a_wall_bound_particle():
    """the step folds `analyticNoPenShift` into `dvdt`: a particle sent into the floor does not pass it."""
    from warpSPH.modules.analyticBoundary import analyticNoPenShift
    res = tank(nSteps=1)
    st, ctx = res.state.state, res.ctx
    sc, cfg = ctx.schemeConfig, ctx.config
    fluid = st.kinds == 0
    i = int(torch.argmin(torch.where(fluid, st.positions[:, 1], torch.full_like(st.positions[:, 1], 9.0))))
    d, _, _ = sc.boundaryProvider.scene.signed_distance(st.positions.double(), supportMax=float(st.supports.max()), want_body=True)[:3]
    st.positions[i, 1] -= float(d[i]) - 0.1 * float(cfg.dx)              # 0.1 dx from the floor: inside the law's 0.25 dx
    st.velocities = torch.zeros_like(st.velocities)
    st.velocities[i, 1] = -1.0
    shift = analyticNoPenShift(sc.boundaryProvider, st, cfg, sc, float(cfg.dx))
    assert float(shift[i, 1]) > 0.0
