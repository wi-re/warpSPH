"""`stokesArray` (periodic array of analytic cylinders / squares, stage 3 E6) and delta+ `pressureConsistent` (E4): a uniform pressure exerts no force, the case runs on every analytic-wall scheme and
carries the right load; the drag coefficient against Sangani-Acrivos is `scripts/probe_stokesArray.py` (20 s, too long for a test).
"""
import numpy as np
import pytest
import torch

pytest.importorskip('warpSPHBoundaries')

from warpSPH.cases import importAll  # noqa: E402
from warpSPH.runner import getCase, run  # noqa: E402
from warpSPH.runner.caseSpec import CaseSpec  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA')


def array(scheme='deltaSPH', params=None, nSteps=1, nx=32):
    importAll()
    case = getCase('stokesArray')
    inc = scheme != 'deltaSPH'
    spec = CaseSpec(caseName='arr', scheme=scheme, params={**case.params, **(params or {})}).merged(**case.defaults).merged(
        scheme=scheme, nx=nx, nSteps=nSteps, plot=False, store=False, progress=False, video=False, show=False, quiet=True,
        **({'kernel': 'Wendland2', 'integrationScheme': 'semiImplicitEuler', 'supportMode': 'SuperSymmetric', 'dt': 2e-3, 'adaptiveDt': False} if inc else {}))
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return run(case, spec)


def test_the_array_cell_holds_the_body_and_the_lattice_around_it():
    r = array()
    st = r.state.state
    f = st.kinds == 0
    c = np.pi * 0.2 ** 2
    assert int(f.sum()) == pytest.approx(32 * 32 * (1 - c), rel=0.02)
    d = (st.positions[f] - 0.5).norm(dim=1) - 0.2
    assert float(d.min()) > -1e-3                                   # no fluid particle inside the cylinder
    assert r.ctx.schemeConfig.boundaryProvider.scene.periodic is not None


@pytest.mark.parametrize('consistent', [False, True])
def test_pressure_consistent_removes_the_force_of_a_uniform_pressure(consistent):
    from warpSPH.modules.pressure import computePressureForceSurfaceAware
    r = array(params={'f': 0.0})
    st, ctx = r.state.state, r.ctx
    sc, cfg, adj = ctx.schemeConfig, ctx.config, r.state.adjacency
    sc.pressureConsistent = consistent
    st.pressures = torch.full_like(st.pressures, 7.0)
    st.surfaceIndicators = torch.zeros_like(st.surfaceIndicators)
    a = computePressureForceSurfaceAware(st, cfg, sc, adj)
    f = st.kinds == 0
    d = (st.positions - 0.5).norm(dim=1) - 0.2
    near = f & (d < 0.5 * float(cfg.dx) * 4)
    mag = float(a[near].norm(dim=1).max())
    if consistent:
        assert mag < 1e-3 * 7.0 / 0.0833, mag                       # P / h scale of the uncorrected force
    else:
        assert mag > 0.05 * 7.0 / 0.0833, mag                        # the cut lattice on the curved wall: a spurious force of a uniform pressure


@pytest.mark.parametrize('scheme', ['omniIncompressible', 'divergenceFree'])
def test_the_incompressible_array_carries_the_body_force_load(scheme):
    r = array(scheme, {'wallViscosityClosure': 'noslipMoment', 'closedPreset': scheme == 'omniIncompressible'}, nSteps=500)
    t = np.asarray(r.series('t'))
    F = np.asarray(r.series('bodyLoad'))
    M = np.asarray(r.series('fluidMass'))
    U = np.asarray(r.series('superficialVelocity'))
    st = r.state.state
    f = st.kinds == 0
    assert not r.diverged
    assert float(np.nanmax(r.series('maxVelocity'))) < 0.5
    assert 0.0 < U[-1] < 0.03 * float(t[-1])                          # held back by the cylinder (a free fluid reaches f t)
    lo, hi = len(t) // 2, len(t) - 1
    # momentum balance: the body force on the fluid = the load on the body + the rate of change of the fluid momentum
    dMom = (U[hi] - U[lo]) / (t[hi] - t[lo]) * 1.0                      # d(sum m u / L^2)/dt with rho0 = 1
    assert (F[lo:hi]).mean() + dMom == pytest.approx(0.03 * float(M[0]), rel=0.08)
