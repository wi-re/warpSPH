"""The compact approximate projection (`modules/incompressible/compactProjection.py`, DFSPH2D `projection='compact'`): the CG solve, the calibration, the projection of a velocity field on analytic walls, the
inviscid Taylor-Green energy (periodic, no walls), and the refusal with boundary particles. The term-level equality with DFSPH2D is `scripts/probe_omniCompactOracle.py` (6e-5 in the pressure).
"""
import os

import numpy as np
import pytest
import torch

pytest.importorskip('warpSPHBoundaries')

from warpSPH.cases import importAll  # noqa: E402
from warpSPH.modules.incompressible.compactProjection import CompactCG, morrisCalibration  # noqa: E402
from warpSPH.runner import getCase, run  # noqa: E402
from warpSPH.runner.caseSpec import CaseSpec  # noqa: E402

ROLL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'examples', 'sloshingTank', 'SPHERIC_TestCase10', 'data_files', 'lateral_water_1x.txt')
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA')


def chain(n=40, dev='cpu'):
    """a 1D Laplacian chain: w_ij = -1 for neighbours (both directions), a diagonal that makes it SPD."""
    i = torch.cat([torch.arange(n - 1), torch.arange(1, n)])
    j = torch.cat([torch.arange(1, n), torch.arange(n - 1)])
    return i.to(dev), j.to(dev), -torch.ones(len(i), dtype=torch.float64, device=dev)


def test_cg_solves_the_symmetric_system_with_diagonal_and_dirichlet_rows():
    n = 40
    i, j, w = chain(n)
    D = -0.05 * torch.ones(n, dtype=torch.float64)
    rhs = torch.sin(torch.arange(n, dtype=torch.float64) * 0.3)
    fixed = torch.zeros(n, dtype=torch.bool)
    fixed[[7, 8, 25]] = True
    P, it, rel = CompactCG(warmStart=False).solve(i, j, w, rhs, tol=1e-12, diag=D, dirichlet=fixed)
    A = torch.zeros((n, n), dtype=torch.float64)
    A.index_put_((i, j), -w, accumulate=True)                    # sum_j w_ij (P_i - P_j) = -sum_j w_ij P_j + (sum_j w_ij) P_i
    A += torch.diag(torch.zeros(n, dtype=torch.float64).index_add_(0, i, w)) + torch.diag(D)
    keep = ~fixed
    ref = torch.zeros(n, dtype=torch.float64)
    ref[keep] = torch.linalg.solve(A[keep][:, keep], rhs[keep])
    assert rel < 1e-10
    assert float((P - ref).abs().max()) < 1e-8 * float(ref.abs().max())
    assert float(P[fixed].abs().max()) == 0.0


def test_cg_warm_start_converges_in_fewer_iterations():
    i, j, w = chain()
    D = -0.05 * torch.ones(40, dtype=torch.float64)
    rhs = torch.cos(torch.arange(40, dtype=torch.float64) * 0.2)
    cg = CompactCG()
    _, it0, _ = cg.solve(i, j, w, rhs, tol=1e-10, diag=D)
    _, it1, _ = cg.solve(i, j, w, rhs, tol=1e-10, diag=D)
    assert it0 > 0 and it1 < it0


def test_morris_calibration_is_one_for_a_fine_lattice_and_close_at_the_scheme_support():
    assert morrisCalibration(0.1, 0.1 / 2.57, (0.1 / 2.57) ** 2) == pytest.approx(0.96, abs=0.03)           # the eta^2 = 0.0025 h^2 regularisation costs ~4 % at the scheme's support
    assert morrisCalibration(0.1, 0.1 / 5.0, (0.1 / 5.0) ** 2, eta2=0.0) == pytest.approx(1.0, abs=0.01)    # the continuum value without it


def tank(params=None, nSteps=1, nh=2.57):
    importAll()
    case = getCase('sloshingTank')
    spec = CaseSpec(caseName='cp', scheme='omniIncompressible', params={**case.params, 'wallRepresentation': 'analytic', 'rollDataFile': ROLL, 'rollStartTime': 100.0, **(params or {})}).merged(**case.defaults).merged(
        scheme='omniIncompressible', integrationScheme='semiImplicitEuler', kernel='Wendland2', supportMode='SuperSymmetric', cflFactor=0.2, dt=1e-3, maxDt=2e-3, nx=100, n_h=nh, nSteps=nSteps,
        plot=False, store=False, progress=False, video=False, show=False, quiet=True)
    with pytest.warns(UserWarning, match='EXPERIMENTAL'):
        return run(case, spec)


@cuda
def test_compact_projection_reduces_the_divergence_of_a_noisy_field_in_the_bulk():
    from warpSPH.modules.analyticBoundary import resolveWall
    from warpSPH.modules.incompressible.compactProjection import solveCompactProjection
    from warpSPH.schemes.omniIncompressible import _divergence
    r = tank(nSteps=1)
    st, ctx = r.state.state, r.ctx
    sc, cfg, adj = ctx.schemeConfig, ctx.config, r.state.adjacency
    fluid = st.kinds == 0
    dt = 1e-3
    v = torch.zeros_like(st.velocities)                                    # a smooth compressive field (the compact Laplacian does not see particle-scale divergence noise: that is its point)
    v[:, 0] = 0.3 * torch.sin(2 * torch.pi * st.positions[:, 0] / 0.4)
    wall = resolveWall(st, cfg, sc, adj)
    a_p, P, it, rel = solveCompactProjection(st, cfg, sc, adj, fluid=fluid, rho0=sc.fluid.restDensity, vEnter=v, dt=dt, wall=wall, solver=CompactCG())
    assert rel < 1e-6 and it > 0
    pos = st.positions.cpu().numpy()
    dx = float(cfg.dx)
    bulk = torch.as_tensor((pos[:, 1] > pos[:, 1].min() + 3 * dx) & (pos[:, 1] < pos[:, 1].max() - 3 * dx) & (np.abs(pos[:, 0]) < np.abs(pos[:, 0]).max() - 4 * dx), device=v.device)
    assert int(bulk.sum()) > 100
    before = _divergence(st, cfg, adj, v)[bulk].pow(2).mean().sqrt()
    after = _divergence(st, cfg, adj, v + dt * a_p)[bulk].pow(2).mean().sqrt()
    assert float(after) < 0.35 * float(before), (float(before), float(after))                     # an approximate projection: the composed divergence is reduced, not removed


@cuda
def test_still_tank_with_the_compact_projection_runs_and_the_scheme_options_are_wired():
    res = tank({'projection': 'compact', 'densitySolve': True}, nSteps=40)
    assert not res.diverged
    assert float(np.nanmax(res.series('maxVelocity'))) < 3.0
    cg = res.ctx.schemeConfig._compactCG
    assert cg.stats['solves'] == 40 and cg.stats['iterations'] > 40


@cuda
def test_density_solve_off_carries_the_projection_pressure():
    res = tank({'projection': 'compact', 'densitySolve': False}, nSteps=3)
    st = res.state.state
    assert float(st.pressures[st.kinds == 0].abs().max()) > 0.0


@cuda
def test_inviscid_taylor_green_energy_is_better_conserved_than_with_three_jacobi_sweeps():
    from warpSPH.cases import importAll
    importAll()
    out = {}
    for name, proj in (('jacobi', 'jacobi'), ('compact', 'compact')):
        case = getCase('tgv')
        cs = case.configureScheme

        def cs2(ctx, cs=cs, proj=proj):
            cs(ctx)
            ctx.schemeConfig.projection = proj
            ctx.schemeConfig.freeSurface = False
        case.configureScheme = cs2
        spec = CaseSpec(caseName='tgv', scheme='divergenceFree', params={**case.params, 'nu': 0.0}).merged(**case.defaults).merged(
            nx=32, n_h=4.0, tLimit=0.6, plot=False, store=False, progress=False, video=False, show=False, quiet=True)
        res = run(case, spec)
        ke = np.asarray(res.series('kineticEnergy'))
        out[name] = 1.0 - ke[-1] / ke[0]
    assert out['compact'] < 0.5 * out['jacobi'], out


@cuda
def test_boundary_particles_refuse_the_compact_projection():
    importAll()
    case = getCase('sloshingTank')
    spec = CaseSpec(caseName='x', scheme='omniIncompressible', params={**case.params, 'projection': 'compact'}).merged(**case.defaults).merged(
        scheme='omniIncompressible', integrationScheme='semiImplicitEuler', kernel='Wendland2', supportMode='SuperSymmetric', cflFactor=0.2, dt=1e-3, maxDt=2e-3, nx=40, nSteps=1,
        plot=False, store=False, progress=False, video=False, show=False, quiet=True)
    with pytest.raises(NotImplementedError, match='analytic walls'):
        run(case, spec)
