"""The analytic-wall terms of `omniIncompressible` (EXPERIMENTAL, OPEN_PROBLEMS 31): the pieces that are verified.

* the diagonal of the discrete operator `A p = -dt^2 div(a_p(p))` (wall terms in `_divergence` and `_pressureAccel`) equals the `alpha` the Jacobi iteration divides by
  (`computeAlpha(wall=)`, the wall's gradient added to the vector sum), on the first wall row and in the bulk: a unit pressure on one particle, the diagonal entry of the
  response;
* the wall mass calibration (`calibrateAnalyticWallMass`, opt-in `calibrateWallMass=True`): it changes the factor away from 1, and with it the zeroth-order consistency of fluid + wall
  (`sum_j V_j grad W_ij + G`) that the diagonal test above relies on: the reason it is off by default (OPEN_PROBLEMS 31).
"""
import numpy as np
import pytest
import torch

pytest.importorskip('warpSPHBoundaries')

from warpSPH.cases import importAll  # noqa: E402
from warpSPH.runner import getCase, run  # noqa: E402
from warpSPH.runner.caseSpec import CaseSpec  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA')


def tank(params=None):
    importAll()
    case = getCase('sloshingTank')
    spec = CaseSpec(caseName='omni', scheme='omniIncompressible', params={**case.params, 'wallRepresentation': 'analytic', **(params or {})}).merged(**case.defaults).merged(
        scheme='omniIncompressible', integrationScheme='semiImplicitEuler', kernel='Wendland2', supportMode='SuperSymmetric', cflFactor=0.2, dt=1e-3, maxDt=2e-3, nx=100, nSteps=1,
        plot=False, store=False, progress=False, video=False, show=False, quiet=True)
    with pytest.warns(UserWarning, match='EXPERIMENTAL'):
        return run(case, spec)


def test_operator_diagonal_is_alpha():
    import warpSPH.schemes.omniIncompressible as O
    from warpSPH.modules.analyticBoundary import resolveWall
    from warpSPH.modules.incompressible.wp_alpha import computeAlpha
    r = tank()
    st, ctx = r.state.state, r.ctx
    sc, cfg, adj = ctx.schemeConfig, ctx.config, r.state.adjacency
    dt = 0.002
    cfg.dt = dt
    fluid = st.kinds == 0
    wall = resolveWall(st, cfg, sc, adj)
    alpha = dt * dt * computeAlpha(st, cfg, sc, adj, apparentVolumes=st.masses / st.densities, includeBoundaryReaction=False, wall=wall)

    def Ap(p):
        a_p = O._pressureAccel(st, cfg, adj, p, fluid, wall)
        return torch.where(fluid, -dt * dt * O._divergence(st, cfg, adj, a_p, wall), torch.zeros_like(p))

    pos = st.positions.cpu().numpy()
    dx = float(cfg.dx)
    base = Ap(torch.zeros_like(st.densities))
    for y in (0.5 * dx, 1.5 * dx, 5.0 * dx):                       # first row, second row, bulk (mid-tank)
        i = int(np.argmin(np.abs(pos[:, 0]) + 100.0 * np.abs(pos[:, 1] - y)))
        e = torch.zeros_like(st.densities)
        e[i] = 1.0
        d = float((Ap(e) - base)[i])
        assert d / float(alpha[i]) == pytest.approx(1.0, abs=0.03), (y, d, float(alpha[i]))


def test_wall_mass_calibration_is_opt_in_and_moves_the_factor():
    assert getattr(tank().ctx.schemeConfig, 'analyticWallMass', 1.0) == 1.0
    wm = tank(params={'calibrateWallMass': True}).ctx.schemeConfig.analyticWallMass
    assert 0.8 < wm < 0.99, wm
