"""The analytic-wall terms of `omniIncompressible` (OPEN_PROBLEMS 31): the pieces that are verified.

* the linear pressure response includes both mirrored wall pressure and wall
  divergence. The historical Jacobi alpha is an approximation to that response,
  not its exact diagonal; check each against its own independently assembled
  wall correction rather than a geometry-specific relative tolerance.
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
        scheme='omniIncompressible', integrationScheme='semiImplicitEuler', kernel='Wendland2', supportMode='SuperSymmetric', cflFactor=0.2, dt=1e-3, maxDt=2e-3, nx=100, n_h=2.57, nSteps=1,
        plot=False, store=False, progress=False, video=False, show=False, quiet=True)
    return run(case, spec)


def test_operator_diagonal_matches_mirrored_wall_response():
    import warpSPH.schemes.incompressible.omniIncompressible as O
    from warpSPH.modules.analyticBoundary import resolveWall
    from warpSPH.modules.incompressible.wp_alpha import computeAlpha
    from warpSPHCore import warpOperation, OperationProperties, WarpOperation, GradientScheme, SupportScheme
    r = tank()
    st, ctx = r.state.state, r.ctx
    sc, cfg, adj = ctx.schemeConfig, ctx.config, r.state.adjacency
    dt = 0.002
    cfg.dt = dt
    fluid = st.kinds == 0
    wall = resolveWall(st, cfg, sc, adj)
    alpha = dt * dt * computeAlpha(st, cfg, sc, adj, apparentVolumes=st.masses / st.densities, includeBoundaryReaction=False, wall=wall)
    alpha0 = dt * dt * computeAlpha(st, cfg, sc, adj, apparentVolumes=st.masses / st.densities, includeBoundaryReaction=False, wall=False)
    grad = warpOperation(st, OperationProperties(kernel=cfg.kernel, operation=WarpOperation.Gradient,
                         gradientMode=GradientScheme.Naive, supportMode=SupportScheme.Gather),
                         queryValues=torch.ones_like(st.densities), domain=cfg.domain, adjacency=adj).double()
    G = wall.G.sum(0)
    expected_alpha_wall = -dt*dt*(2*(grad*G).sum(1)+G.square().sum(1))/st.densities.double()
    assert torch.allclose((alpha-alpha0).double(), expected_alpha_wall, atol=1e-8, rtol=1e-5)

    def Ap(p):
        a_p = O._pressureAccel(st, cfg, adj, p, fluid, wall, sc.fluid.restDensity, clampPressure=False)
        return torch.where(fluid, -dt * dt * O._divergence(st, cfg, adj, a_p, wall), torch.zeros_like(p))

    pos = st.positions.cpu().numpy()
    dx = float(cfg.dx)
    base = Ap(torch.zeros_like(st.densities))
    for y in (0.5 * dx, 1.5 * dx, 5.0 * dx):                       # first row, second row, bulk (mid-tank)
        i = int(np.argmin(np.abs(pos[:, 0]) + 100.0 * np.abs(pos[:, 1] - y)))
        e = torch.zeros_like(st.densities)
        e[i] = 1.0
        d = float((Ap(e) - base)[i])
        a_fluid = O._pressureAccel(st, cfg, adj, e, fluid)
        d_fluid = float((-dt*dt*O._divergence(st, cfg, adj, a_fluid))[i])
        # The wall responds only at i; fluid and wall divergence both see
        # that response. Wall divergence also sees the fluid acceleration.
        a_wall_i = -(1/st.densities[i].double()**2 + 1/sc.fluid.restDensity**2)*G[i]
        correction = dt*dt*((a_wall_i*(grad[i]+G[i])).sum()+(a_fluid[i].double()*G[i]).sum())
        assert d == pytest.approx(d_fluid+float(correction), rel=2e-5, abs=1e-8)


def test_explicit_no_wall_is_not_resolved():
    """`wall=False` is "no wall"; `wall=None` resolves the scheme's provider. The omni divergence solve uses it so its `alpha` carries no wall that its operator does not
    (before: a first-row alpha 3x too small, the Jacobi iteration unstable at n_h = 4, OPEN_PROBLEMS 31)."""
    from warpSPH.modules.analyticBoundary import resolveWall
    from warpSPH.modules.incompressible.wp_alpha import computeAlpha
    r = tank()
    st, ctx = r.state.state, r.ctx
    sc, cfg, adj = ctx.schemeConfig, ctx.config, r.state.adjacency
    cfg.dt = 0.002
    assert resolveWall(st, cfg, sc, adj, False) is None
    assert resolveWall(st, cfg, sc, adj) is not None
    app = st.masses / st.densities
    withWall = computeAlpha(st, cfg, sc, adj, apparentVolumes=app, includeBoundaryReaction=False, wall=None)       # None: resolved, the wall is in
    noWall = computeAlpha(st, cfg, sc, adj, apparentVolumes=app, includeBoundaryReaction=False, wall=False)
    y = st.positions[:, 1]
    first = y < y.min() + 1e-6
    assert float(noWall[first].abs().mean()) > 1.5 * float(withWall[first].abs().mean())          # the first row: |A|^2 against |A + G|^2


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA')
@pytest.mark.parametrize('nh', [2.57, 4.0])
def test_hydrostatic_tank_stays_quiet_at_either_support(nh):
    """still water, zero roll, 60 steps, analytic walls with the calibrated lattice: the max velocity stays < 0.5 m/s (n_h = 4 before the fix: 0.95 and growing)."""
    from warpSPH.cases import importAll
    from warpSPH.runner import getCase, run
    from warpSPH.runner.caseSpec import CaseSpec
    import os
    importAll()
    case = getCase('sloshingTank')
    roll = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'examples', 'sloshingTank', 'SPHERIC_TestCase10', 'data_files', 'lateral_water_1x.txt')
    spec = CaseSpec(caseName='hydro', scheme='omniIncompressible', params={**case.params, 'wallRepresentation': 'analytic', 'rollDataFile': roll, 'rollStartTime': 100.0}).merged(**case.defaults).merged(
        scheme='omniIncompressible', integrationScheme='semiImplicitEuler', kernel='Wendland2', supportMode='SuperSymmetric', cflFactor=0.2, dt=1e-3, maxDt=2e-3, nx=100, n_h=nh, nSteps=60,
        plot=False, store=False, progress=False, video=False, show=False, quiet=True)
    res = run(case, spec)
    assert float(np.nanmax(res.series('maxVelocity'))) < 0.5
