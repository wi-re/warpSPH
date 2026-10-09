"""The calibrated lattice (`modules/analyticBoundary/latticeCalibration.py`, port of the boundaries repo's `lattice_calibration`) and its application to an analytic tank (`boundary/calibration.py`)."""
import math

import numpy as np
import pytest
import torch

from warpSPH.modules.analyticBoundary.latticeCalibration import latticeCalibration, wallIntegral, wendland2


# reference values of the boundaries repo's `lattice_calibration` (dfsph2d.py), printed 2026-10-09
REFERENCE = [
    (0.2 / 23, 0.2 / 23, math.sqrt(20) * 0.005, dict(S=1.011315, V=7.47683e-05, mu=0.988811, dx=0.54959, dy=0.54959)),
    (0.2 / 22, 0.8 / 90, math.sqrt(20) * 0.005, dict(S=1.014277, V=7.96706e-05, mu=0.985924, dx=0.55377, dy=0.55115)),
    (0.009, 0.009, 4 * 0.009, dict(S=1.001206, V=8.09024e-05, mu=0.998796, dx=0.51984, dy=0.51984)),
]


@pytest.mark.parametrize('dx,dy,h,ref', REFERENCE)
def test_matches_the_boundaries_repo(dx, dy, h, ref):
    cal = latticeCalibration(dx, dy, h)
    assert cal['S'] == pytest.approx(ref['S'], abs=2e-6)
    assert cal['V'] == pytest.approx(ref['V'], rel=2e-6)
    assert cal['mu'] == pytest.approx(ref['mu'], abs=2e-6)
    assert cal['dwallX'] / dx == pytest.approx(ref['dx'], abs=2e-5)
    assert cal['dwallY'] / dy == pytest.approx(ref['dy'], abs=2e-5)


def test_the_calibrated_lattice_is_a_rest_state():
    """bulk density 1 with the particle mass V', first-row density 1 at the calibrated wall distance (the definition, checked with independent sums)."""
    dx, dy, h = 0.0090, 0.00893, 2.57 * 0.009
    cal = latticeCalibration(dx, dy, h)
    N = 12
    n, m = np.meshgrid(np.arange(-N, N + 1), np.arange(-N, N + 1), indexing='ij')
    assert cal['V'] * wendland2(np.hypot(n * dx, m * dy), h).sum() == pytest.approx(1.0, abs=1e-9)          # bulk
    X, Y = np.meshgrid(np.arange(-N, N + 1) * dx, np.arange(0, N + 1) * dy, indexing='ij')                    # the fluid half lattice above a wall normal to y
    row0 = cal['V'] * wendland2(np.hypot(X, Y), h).sum() + cal['mu'] * wallIntegral(cal['dwallY'], h)
    assert row0 == pytest.approx(1.0, abs=1e-7)


def test_wall_integral_limits():
    h = 0.02
    assert wallIntegral(0.0, h) == pytest.approx(0.5, abs=1e-6)                 # the kernel integrates to 1: half of it beyond a plane through the particle
    assert wallIntegral(h, h) == 0.0
    d = np.linspace(0.0, h, 50)
    assert np.all(np.diff([wallIntegral(x, h) for x in d]) <= 1e-12)             # decreasing with distance


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA')
def test_applied_to_the_analytic_dam_break():
    pytest.importorskip('warpSPHBoundaries')
    from warpSPH.cases import importAll
    from warpSPH.runner import getCase, run
    from warpSPH.runner.caseSpec import CaseSpec
    importAll()
    case = getCase('dambreak')
    spec = CaseSpec(caseName='cal', scheme='omniIncompressible', params={**case.params, 'wallRepresentation': 'analytic'}).merged(**case.defaults).merged(
        scheme='omniIncompressible', nx=60, n_h=2.57, nSteps=1, integrationScheme='semiImplicitEuler', kernel='Wendland2', supportMode='SuperSymmetric', cflFactor=1.0, dt=1e-3, maxDt=1e-3,
        plot=False, store=False, progress=False, video=False, show=False, quiet=True)
    with pytest.warns(UserWarning, match='EXPERIMENTAL'):
        res = run(case, spec)
    sc = res.ctx.schemeConfig
    cal = latticeCalibration(0.9 * 0 + res.ctx.config.dx, res.ctx.config.dx, float(res.state.state.supports.max()))
    assert sc.analyticWallMass == pytest.approx(cal['mu'], rel=2e-3)             # the wall mass is the lattice's (the lattice spacings of the sample are within a percent of dx)
    assert float(res.state.state.masses.max()) == pytest.approx(cal['V'], rel=0.02)
