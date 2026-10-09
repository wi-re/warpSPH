"""Implicit / dynamic shifting (`computeImplicitShift`, default `legacyPairwise` operator) with analytic walls against warpSPH's boundary-particle path on the dam break's initial state: the wall continuum's
share of grad C is the only change (the wall particles are fixed, the matrix of the fluid rows does not see them), so the solved shifts agree to the lattice quadrature error of the wall sum; the
`exactHessian` operator, whose diagonal needs the wall's Hessian integral, is refused."""
import numpy as np
import pytest
import torch

pytest.importorskip('warpSPHBoundaries')

from warpSPHCore import SupportScheme, buildVerletList  # noqa: E402

from warpSPH.configurations.moduleConfigurations.shifting import ShiftingImplicitOperator, ShiftingScheme  # noqa: E402
from warpSPH.modules.shifting.implicitShifting import computeDynamicImplicitShift, computeImplicitShift  # noqa: E402
from test_analyticMichel import build  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA')


def shift(rep, fn=computeImplicitShift, operator=None, summation=False):
    ctx, system = build(rep)
    st, config, sc = system.state, ctx.config, ctx.schemeConfig
    sc.shiftProperties.scheme = ShiftingScheme.implicit
    ghost = st.kinds == 2                                                                           # mDBC ghost NODES (interpolation points inside the fluid, not wall mass): computeImplicitShift's
    st.masses = torch.where(ghost, torch.zeros_like(st.masses), st.masses)                          # pair sums do not filter them (the delta / Michel sums do), so they are given no weight here
    sc.shiftProperties.summationDensity = summation
    if operator is not None:
        sc.shiftProperties.implicitOperator = operator
    adj = buildVerletList(st, config.domain, verletScale=config.verletScale, supportMode=SupportScheme.SuperSymmetric, verbose=False)
    upd, _ = fn(st, config, sc, config.domain, adj, iters=1)
    pos = st.positions.double().cpu().numpy()
    m = (st.kinds == 0).cpu().numpy()
    key = np.lexsort((pos[m][:, 1].round(9), pos[m][:, 0].round(9)))
    return pos[m][key], upd.double().cpu().numpy()[m][key]


@pytest.mark.parametrize('fn', [computeImplicitShift, computeDynamicImplicitShift])
@pytest.mark.parametrize('summation', [False, True])
def test_implicit_shift_with_analytic_walls_equals_the_wall_particle_path(fn, summation):
    pa, ua = shift('particles', fn, summation=summation)
    pb, ub = shift('analytic', fn, summation=summation)
    assert np.abs(pa - pb).max() < 1e-5
    scale = np.linalg.norm(ua, axis=1).max()
    err = np.linalg.norm(ua - ub, axis=1) / scale
    print(f'implicit shift scale {scale:.3e}: max error {err.max():.3e}, p90 {np.percentile(err, 90):.3e}')
    assert scale > 0
    assert err.max() < 0.1 and np.percentile(err, 90) < 0.05                                       # lattice quadrature of the wall sum, as for grad C-tilde (3 %)


def test_exact_hessian_operator_is_refused_with_analytic_walls():
    with pytest.raises(NotImplementedError):
        shift('analytic', operator=ShiftingImplicitOperator.exactHessian)
