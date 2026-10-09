"""The velocity filters and the wall pressure option of the analytic-wall `omniIncompressible` loop (DFSPH2D's `wallPressure='linear'`, `xsph`, `boundaryFriction`; EXPERIMENTAL, OPEN_PROBLEMS 31).

* the MLS pressure fit reproduces a linear pressure field exactly (its gradient) and its wall term contracts the provider's first-moment tensor the same way the hydrostatic one does;
* for a hydrostatic pressure the MLS wall force equals the hydrostatic wall force away from the ceiling;
* a still tank with the MLS wall pressure stays quiet;
* XSPH conserves momentum and smooths; the boundary friction removes tangential velocity next to a wall only, leaves the wall-normal part and the bulk alone.
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


def tank(params=None, nSteps=1, nh=2.57):
    importAll()
    case = getCase('sloshingTank')
    spec = CaseSpec(caseName='omni', scheme='omniIncompressible', params={**case.params, 'wallRepresentation': 'analytic', 'rollDataFile': ROLL, 'rollStartTime': 100.0, **(params or {})}).merged(**case.defaults).merged(
        scheme='omniIncompressible', integrationScheme='semiImplicitEuler', kernel='Wendland2', supportMode='SuperSymmetric', cflFactor=0.2, dt=1e-3, maxDt=2e-3, nx=100, n_h=nh, nSteps=nSteps,
        plot=False, store=False, progress=False, video=False, show=False, quiet=True)
    with pytest.warns(UserWarning, match='EXPERIMENTAL'):
        return run(case, spec)


def test_mls_fit_gradient_is_exact_for_a_linear_field():
    from warpSPH.modules.analyticBoundary import buildMLSPressureFit
    r = tank()
    st, ctx = r.state.state, r.ctx
    fit = buildMLSPressureFit(st, ctx.config, r.state.adjacency, st.kinds == 0, ctx.schemeConfig.fluid.restDensity)
    x = st.positions.double()
    p = 1.0e3 + 3.0 * x[:, 0] + 5.0 * x[:, 1]
    a1 = fit.gradient(p)
    pos = x.cpu().numpy()
    dx = float(ctx.config.dx)
    inner = torch.as_tensor((pos[:, 1] > pos[:, 1].min() + 4 * dx) & (np.abs(pos[:, 0]) < np.abs(pos[:, 0]).max() - 4 * dx), device=a1.device)
    assert float((a1[inner] - torch.tensor([3.0, 5.0], dtype=a1.dtype, device=a1.device)).abs().max()) < 1e-6 * 5.0 * 1e3


def test_mls_wall_term_contracts_the_first_moment_like_the_hydrostatic_one():
    """`wm a1_d C_dj` with `a1 = rho0 g` is the `WallState`'s hydrostatic `A`; and for a hydrostatic pressure the MLS wall force matches the hydrostatic one on the bottom rows."""
    from warpSPH.modules.analyticBoundary import buildMLSPressureFit, resolveWall, wallPressureAccelerationOmni
    from warpSPH.modules.gravity import computeGravity
    r = tank()
    st, ctx = r.state.state, r.ctx
    sc, cfg, adj = ctx.schemeConfig, ctx.config, r.state.adjacency
    wall = resolveWall(st, cfg, sc, adj)
    rho0 = sc.fluid.restDensity
    g = computeGravity(st, cfg, sc, adj)[0].double()
    A = wall.wm * torch.einsum('d,bndj->bnj', rho0 * g, wall.agg.out['Cov'].double())
    assert float((A - wall.A).abs().max()) < 1e-6 * float(wall.A.abs().max())          # float32 gravity

    x = st.positions.double()
    top = x[:, 1].max()
    p = rho0 * float(-g[1]) * (top - x[:, 1]) + 1.0                                 # hydrostatic, positive
    fit = buildMLSPressureFit(st, cfg, adj, st.kinds == 0, rho0)
    a1 = fit.gradient(p)
    mls = wallPressureAccelerationOmni(wall, p, st.densities, rho0, wallMass=wall.wm, h=wall.support, gradient=a1)
    hyd = wallPressureAccelerationOmni(wall, p, st.densities, rho0, wallMass=wall.wm, h=wall.support)
    bottom = x[:, 1] < x[:, 1].min() + 1.5 * float(cfg.dx)
    near = bottom & (wall.near > 0)
    ref = float(hyd[near].norm(dim=1).max())
    assert float((mls[near] - hyd[near]).norm(dim=1).max()) < 0.05 * ref


@pytest.mark.parametrize('nh', [2.57, 4.0])
def test_mls_wall_pressure_tank_stays_bounded(nh):
    """The MLS wall pressure is the noisier closure (DFSPH2D's own `linear`: 1.5-2x the velocity, ~4x the kinetic energy of the hydrostatic one on a still column, OPEN_PROBLEMS 31): a still tank stays
    bounded (peak < 1.5 m/s) and does not grow (the last 100 of 250 steps < 0.8 m/s: its noise floor is ~0.3); the hydrostatic closure stays the default."""
    res = tank({'analyticWallPressure': 'mls'}, nSteps=250, nh=nh)
    v = np.asarray(res.series('maxVelocity'))
    assert float(np.nanmax(v)) < 1.5
    assert float(np.nanmax(v[-100:])) < 0.8                       # the MLS closure's noise floor, not growth


def test_xsph_conserves_momentum_and_smooths():
    from warpSPH.modules.xsph import computeXSPH
    r = tank()
    st, ctx = r.state.state, r.ctx
    gen = torch.Generator(device=st.positions.device).manual_seed(1)
    st.velocities = 0.1 * torch.randn(st.velocities.shape, generator=gen, device=st.velocities.device, dtype=st.velocities.dtype)
    dv = computeXSPH(st, ctx.config, ctx.schemeConfig, r.state.adjacency, fluidCoefficient=0.1)
    m = st.masses.unsqueeze(-1)
    assert float((m * dv).sum(0).abs().max()) < 3e-3 * float((m * st.velocities).abs().sum(0).max())            # antisymmetric pair weights up to the apparent volumes m_j / rho_j (density variation ~1e-3)
    assert float(((st.velocities + dv) ** 2).sum()) < float((st.velocities ** 2).sum())
    assert float(computeXSPH(st, ctx.config, ctx.schemeConfig, r.state.adjacency).abs().max()) == 0.0


def test_boundary_friction_is_tangential_and_local():
    from warpSPH.modules.analyticBoundary import resolveWall
    from warpSPH.modules.xsph import computeBoundaryFriction
    r = tank()
    st, ctx = r.state.state, r.ctx
    sc, cfg, adj = ctx.schemeConfig, ctx.config, r.state.adjacency
    wall = resolveWall(st, cfg, sc, adj)
    st.velocities = torch.zeros_like(st.velocities)
    st.velocities[:, 0] = 0.3                                     # along x: tangential to the bottom, normal to the side walls
    st.velocities[:, 1] = 0.2
    dv = computeBoundaryFriction(st, cfg, sc, adj, coefficient=5.0e-3).double()
    pos = st.positions.cpu().numpy()
    dx = float(cfg.dx)
    lam = wall.lam.sum(0)
    far = lam == 0
    assert float(dv[far].abs().max()) == 0.0                      # no wall in the support, no drag
    bottom = torch.as_tensor(pos[:, 1] < pos[:, 1].min() + 0.1 * dx, device=dv.device) & (torch.as_tensor(np.abs(pos[:, 0]) < np.abs(pos[:, 0]).max() - 3 * dx, device=dv.device))
    assert float(dv[bottom, 0].max()) < 0 and float(dv[bottom, 1].abs().max()) < 1e-7 * 0.3        # tangential only: x removed, the wall-normal y untouched
    assert float(dv[bottom, 0].min()) > -0.3                                                   # never reverses the velocity
    assert float(computeBoundaryFriction(st, cfg, sc, adj).abs().max()) == 0.0                 # off by default


def test_delta_plus_refuses_the_mls_wall_pressure():
    with pytest.raises(ValueError, match='omniIncompressible options'):
        importAll()
        case = getCase('sloshingTank')
        spec = CaseSpec(caseName='x', scheme='deltaSPH', params={**case.params, 'wallRepresentation': 'analytic', 'analyticWallPressure': 'mls'}).merged(**case.defaults).merged(
            scheme='deltaSPH', kernel='Wendland2', nx=60, nSteps=1, plot=False, store=False, progress=False, video=False, show=False, quiet=True)
        run(case, spec)
