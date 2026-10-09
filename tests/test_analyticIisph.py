"""`iisph` and `dfsphReference` on analytic walls (stage 2 C; the SPlisHSPlasH-style troubleshooting schemes of DFSPH_IMPROVEMENT_PLAN.md, EXPERIMENTAL): the operator's diagonal equals the DFSPH factor
with the wall in it, the schemes run a still tank bounded, and they refuse moving bodies.
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


def tank(scheme, nSteps=1, nh=2.57, setup=None):
    importAll()
    case = getCase('sloshingTank')
    ic = case.initialConditions
    if setup is not None:
        def ic2(ctx, system):
            ic(ctx, system)
            setup(ctx)
        case.initialConditions = ic2
    spec = CaseSpec(caseName='iisph', scheme=scheme, params={**case.params, 'wallRepresentation': 'analytic', 'rollDataFile': ROLL, 'rollStartTime': 100.0}).merged(**case.defaults).merged(
        scheme=scheme, integrationScheme='semiImplicitEuler', kernel='Wendland2', supportMode='SuperSymmetric', cflFactor=0.2, dt=1e-3, maxDt=2e-3, nx=100, n_h=nh, nSteps=nSteps,
        plot=False, store=False, progress=False, video=False, show=False, quiet=True)
    try:
        with pytest.warns(UserWarning, match='EXPERIMENTAL'):
            return run(case, spec)
    finally:
        case.initialConditions = ic


def test_the_operator_diagonal_is_the_dfsph_factor_with_the_wall():
    import warpSPH.schemes.dfsphReference as D
    from warpSPH.modules.analyticBoundary import resolveWall
    r = tank('dfsphReference')
    st, ctx = r.state.state, r.ctx
    sc, cfg, adj = ctx.schemeConfig, ctx.config, r.state.adjacency
    fluid = st.kinds == 0
    rho0, dt = sc.fluid.restDensity, 1e-3
    wall = resolveWall(st, cfg, sc, adj)
    fac, nowall = D._factor(st, cfg, sc, adj, wall), D._factor(st, cfg, sc, adj, None)

    def Ap(p):
        return D._drhodt(st, cfg, sc, adj, D._pressureAccel(st, cfg, adj, p, fluid, wall, rho0), wall) / rho0 * dt * dt
    base = Ap(torch.zeros_like(st.densities))
    pos = st.positions.cpu().numpy()
    dx = float(cfg.dx)
    for y, tol in ((0.5 * dx, 0.08), (1.5 * dx, 0.02), (5 * dx, 0.01)):                 # first row, second row, bulk
        i = int(np.argmin(np.abs(pos[:, 0]) + 100 * np.abs(pos[:, 1] - y)))
        e = torch.zeros_like(st.densities)
        e[i] = 1.0
        d = float((Ap(e) - base)[i])
        assert d / (float(fac[i]) * dt * dt) == pytest.approx(1.0, abs=tol), (y, d, float(fac[i]) * dt * dt)
        if y < dx:
            assert abs(d / (float(nowall[i]) * dt * dt) - 1.0) > 0.3                     # without the wall the first-row diagonal is half


@pytest.mark.parametrize('scheme', ['iisph', 'dfsphReference'])
def test_still_tank_stays_bounded(scheme):
    r = tank(scheme, nSteps=150)
    v = np.asarray(r.series('maxVelocity'))
    assert not r.diverged
    assert float(np.nanmax(v)) < 3.0
    st = r.state.state
    f = st.kinds == 0
    assert float(st.densities[f].max()) < 1.05


@pytest.mark.parametrize('scheme', ['iisph', 'dfsphReference'])
def test_moving_bodies_are_refused(scheme):
    def setup(ctx):
        rb = ctx.schemeConfig.boundaryProvider.rigidBodies[0]
        rb.linearVelocity = torch.tensor([0.3, 0.0], dtype=rb.linearVelocity.dtype, device=rb.linearVelocity.device)
    with pytest.raises(NotImplementedError, match='static bodies'):
        tank(scheme, nSteps=2, setup=setup)
