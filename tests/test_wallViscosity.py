"""The no-slip wall closure of the Morris viscosity on analytic walls (stage 3 E3, `modules/analyticBoundary/wallViscosity.py`, `wallMoments.py`): fluid + wall is exact for a wall-quadratic
profile on the actual neighbourhood, the tables are the oracle's and the brute-force solid integral, and the refusals. The term-level equality with `DFSPH2D._viscous_accel` is
`scripts/probe_omniViscosityOracle.py` (1e-6 float32 against float64).
"""
import math

import numpy as np
import pytest
import torch

pytest.importorskip('warpSPHBoundaries')

from warpSPH.cases import importAll  # noqa: E402
from warpSPH.runner import getCase, run  # noqa: E402
from warpSPH.runner.caseSpec import CaseSpec  # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA')


def test_curved_tables_match_a_brute_force_integral_over_the_solid():
    """the Morris moments of a flat wall at distance d: `T_k = int_S K(r) s'^k dA'` over the half plane beyond d, by a fine polar sum; `M1_n = int_S K (y . n) dA'`."""
    from warpSPH.modules.analyticBoundary.wallMoments import CurvedWallMoments
    from warpSPH.modules.incompressible.compactProjection import MORRIS_ETA2, _dwendland2
    H = 1.0
    tab = CurvedWallMoments(H, 'cpu', weight='morris')
    for d in (0.1, 0.35, 0.6):
        # polar grid about the particle: y = r (-cos th, -sin th) towards the solid (normal along +x of the particle frame), solid where -y_x >= d
        r = torch.linspace(1e-4, H, 1600, dtype=torch.float64)
        th = torch.linspace(-math.pi / 2, math.pi / 2, 1601, dtype=torch.float64)
        R, TH = torch.meshgrid(r, th, indexing='ij')
        yx = -R * torch.cos(TH)
        inside = (-yx >= d) & (R < H)
        K = _dwendland2(R, H) * R / (R * R + MORRIS_ETA2 * H * H)
        dA = R * (r[1] - r[0]) * (th[1] - th[0])
        s = d + yx                                              # s' = y . n + d, n pointing into the fluid (away from the solid): s' = d - r cos th <= 0 in the solid
        T0 = float((K * dA * inside).sum())
        T1 = float((K * dA * inside * s).sum())
        T2 = float((K * dA * inside * s * s).sum())
        M1 = float((K * dA * inside * yx).sum())
        got = tab.eval(torch.tensor([d], dtype=torch.float64), torch.tensor([0.0], dtype=torch.float64))[:, :, 0]
        assert float(got[0, 0]) == pytest.approx(T0, rel=2e-2)
        assert float(got[0, 1]) == pytest.approx(T1, rel=2e-2)
        assert float(got[0, 2]) == pytest.approx(T2, rel=2e-2)
        assert abs(float(got[0, 3]) - M1) < 1.5e-2 * abs(M1) + 1e-9           # the brute-force polar sum is crude (1600 x 1601 cells)


def channel(params=None, nSteps=1, nx=32):
    importAll()
    case = getCase('periodicChannel')
    spec = CaseSpec(caseName='ch', scheme='deltaSPH', params={**case.params, 'nu': 0.01, 'inviscid': False, **(params or {})}).merged(**case.defaults).merged(
        nx=nx, nSteps=nSteps, plot=False, store=False, progress=False, video=False, show=False, quiet=True)
    return run(case, spec)


def configureMorris(sc, complement=True):
    from warpSPH.enumTypes import ViscosityTerm
    sc.diffusionParams.inviscid = False
    sc.diffusionParams.viscousTerm = ViscosityTerm.morris1997
    sc.diffusionParams.viscidNu = 0.01
    sc.wallViscosityClosure = 'noslipMoment'
    sc.complementMoments = complement
    sc.morrisCalibration = None


@cuda
@pytest.mark.parametrize('complement', [True, False])
def test_fluid_plus_wall_is_nu_u_second_for_a_wall_quadratic_profile(complement):
    """w(s) = a s + L s^2 / 2 with w = 0 at the wall plane: the closure makes `fluid sum + wall term = nu_p L` at the particles of the first rows (complement: on the actual neighbourhood, to the
    float32 round-off; the geometric tables miss it by ~10 % on the first row, the oracle's negative control)."""
    from warpSPH.modules.deltaSPH import computeVelocityDiffusion
    from warpSPH.modules.incompressible.compactProjection import morrisCalibration
    r = channel()
    st, ctx = r.state.state, r.ctx
    sc, cfg, adj = ctx.schemeConfig, ctx.config, r.state.adjacency
    configureMorris(sc, complement)
    st.densities = torch.ones_like(st.densities)                                         # the contract is for a uniform density (the Morris weight carries (rho_i + rho_j) / rho_i); the summation density varies by 1 % at a wall
    y = st.positions[:, 1].double()
    a, L = 0.1, 0.8
    s = torch.minimum(y, 0.5 - y)                                                        # distance to the nearer plate: both walls see the same wall-quadratic profile
    u = a * s + 0.5 * L * s * s
    st.velocities = torch.stack([u, torch.zeros_like(u)], 1).to(st.velocities.dtype)
    acc = computeVelocityDiffusion(st, cfg, sc, adj)
    cal = morrisCalibration(float(st.supports.double().median()), float(cfg.dx), float(st.masses.double().median()))
    expect = 0.01 * cal * L
    first = (s < 1.0 * float(cfg.dx)) & (st.kinds == 0)
    assert int(first.sum()) > 10
    err = (acc[first, 0].double() - expect).abs() / expect
    if complement:
        assert float(err.max()) < 1e-4, float(err.max())
    else:
        assert 1e-2 < float(err.max()) < 0.15, float(err.max())                       # the continuum tables miss the discrete neighbourhood by more than 1 % (the oracle's negative control)
    assert float(acc[first, 1].abs().max()) < 1e-3 * expect


@cuda
def test_the_closure_needs_the_morris_viscosity():
    from warpSPH.modules.deltaSPH import computeVelocityDiffusion
    r = channel()
    st, ctx = r.state.state, r.ctx
    sc, cfg, adj = ctx.schemeConfig, ctx.config, r.state.adjacency
    sc.wallViscosityClosure = 'noslipMoment'
    sc.diffusionParams.inviscid = True
    with pytest.raises(NotImplementedError, match='Morris'):
        computeVelocityDiffusion(st, cfg, sc, adj)
