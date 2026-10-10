"""Stage 3 E6 (ANALYTIC_BOUNDARIES_PLAN.md): the `taylorCouette` lattice. The regular sampler lays its lattice over the domain extent, so the case sets the extent to a whole (odd) number of spacings: the lattice
is at exactly dx, cell-centred on the axis (as the oracle's `run_couette`), and the wall cut (`wallCut`, the oracle's rule of the scheme by default) removes the particles next to the walls. A lattice at 1 / 48.8
instead of 1 / 48 left the calibrated particle mass wrong by 3 % in density and the inner torque 5 % off the oracle's run on the same lattice (2026-10-10).
"""
import numpy as np
import pytest
import torch

pytest.importorskip('warpSPHBoundaries')

from warpSPH.cases import importAll  # noqa: E402
from warpSPH.runner import buildContext, getCase  # noqa: E402
from warpSPH.runner.caseSpec import CaseSpec  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA')


def lattice(scheme='omniIncompressible', nx=48, n_h=2.505, **params):
    importAll()
    case = getCase('taylorCouette')
    kw = dict(integrationScheme='semiImplicitEuler', supportMode='SuperSymmetric', dt=2e-3, adaptiveDt=False) if scheme != 'deltaSPH' else {}
    spec = CaseSpec(caseName='tc', scheme=scheme, params={**case.params, **params}).merged(**case.defaults).merged(scheme=scheme, n_h=n_h, nx=nx, kernel='Wendland2', **kw)
    ctx = buildContext(case, spec)
    case.configureScheme(ctx)
    system = case.buildSystem(ctx)
    st = system.state
    x = st.positions[st.kinds == 0].double().cpu().numpy()
    return ctx, x


def test_the_lattice_is_at_exactly_dx_and_half_cell_centred():
    ctx, x = lattice()
    dx = 1.0 / 48
    ux = np.unique(np.round(x[:, 0], 6))
    assert np.allclose(np.diff(ux), dx, atol=1e-5)
    k = x[:, 0] / dx - 0.5
    assert float(np.abs(k - np.round(k)).max()) < 1e-3                       # nodes at half-integer multiples of dx, as the oracle


def test_the_incompressible_cut_is_the_calibrated_first_row_distance_and_matches_the_oracle_count():
    """`DFSPH2D`'s run_couette keeps r1 + dwallY - 0.01 dx < r < r2 - dwallY + 0.01 dx on that lattice: 1384 particles at n = 48, H = 2.505 dx."""
    ctx, x = lattice()
    assert len(x) == 1384
    r = np.linalg.norm(x, axis=1)
    dx = 1.0 / 48
    assert r.min() > 0.2 + 0.54 * dx and r.max() < 0.5 - 0.54 * dx


def test_delta_cut_is_half_a_spacing_and_zero_keeps_the_whole_lattice():
    dx = 1.0 / 32
    ctx, x = lattice('deltaSPH', nx=32, n_h=4.0)
    r = np.linalg.norm(x, axis=1)
    assert r.min() >= 0.2 + 0.5 * dx - 1e-6 and r.max() <= 0.5 - 0.5 * dx + 1e-6
    ctx0, x0 = lattice('deltaSPH', nx=32, n_h=4.0, wallCut=0.0)
    assert len(x0) > len(x) and np.linalg.norm(x0, axis=1).min() < 0.2 + 0.5 * dx
