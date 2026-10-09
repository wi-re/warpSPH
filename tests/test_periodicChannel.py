"""Stage 3 E1 / E2 (ANALYTIC_BOUNDARIES_PLAN.md): the periodic wall images of the analytic provider and the uniform body force, on the `periodicChannel` case (plane Poiseuille between two analytic plates that
span the periodic box). The Poiseuille profile itself is `scripts/probe_periodicChannel.py` (12 s, too long for a test).
"""
import numpy as np
import pytest
import torch

pytest.importorskip('warpSPHBoundaries')

from warpSPH.cases import importAll  # noqa: E402
from warpSPH.runner import getCase, run  # noqa: E402
from warpSPH.runner.caseSpec import CaseSpec  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA')


def channel(params=None, nSteps=3, nx=32):
    importAll()
    case = getCase('periodicChannel')
    spec = CaseSpec(caseName='ch', scheme='deltaSPH', params={**case.params, **(params or {})}).merged(**case.defaults).merged(
        nx=nx, nSteps=nSteps, plot=False, store=False, progress=False, video=False, show=False, quiet=True)
    return run(case, spec)


def test_the_plates_span_the_periodic_box_at_every_image():
    """the wall integrals of a particle and of its image one box length away are the same (the plates are 1.6 long in a box of 1: without the images the shifted particles would see the plate end)."""
    from warpSPH.modules.analyticBoundary import resolveWall
    r = channel()
    st, ctx = r.state.state, r.ctx
    sc, cfg, adj = ctx.schemeConfig, ctx.config, r.state.adjacency
    assert sc.boundaryProvider.scene.periodic is not None
    w0 = resolveWall(st, cfg, sc, adj)
    lam0, G0 = w0.lam.clone(), w0.G.clone()
    st.positions = st.positions + torch.tensor([1.0, 0.0], dtype=st.positions.dtype, device=st.positions.device)          # one box length: the same particles at the next image
    w1 = resolveWall(st, cfg, sc, adj)
    assert float((w1.lam - lam0).abs().max()) < 1e-5 * float(lam0.abs().max())
    assert float((w1.G - G0).abs().max()) < 1e-4 * float(G0.abs().max())
    assert float(lam0.max()) > 0.3                                                       # the first rows do touch the plates


def test_without_the_images_the_shifted_particles_leave_the_plate():
    from warpSPH.modules.analyticBoundary import resolveWall
    r = channel({'shearWave': False})
    st, ctx = r.state.state, r.ctx
    sc, cfg, adj = ctx.schemeConfig, ctx.config, r.state.adjacency
    sc.boundaryProvider.scene.setPeriodic(None)
    lam0 = resolveWall(st, cfg, sc, adj).lam.clone()
    st.positions = st.positions + torch.tensor([1.6, 0.0], dtype=st.positions.dtype, device=st.positions.device)
    lam1 = resolveWall(st, cfg, sc, adj).lam
    assert float(lam1.max()) < 0.1 * float(lam0.max())


def test_a_body_force_accelerates_a_periodic_fluid_uniformly():
    f = 0.2
    r = channel({'shearWave': True, 'fShear': f, 'u0': 0.0}, nSteps=200, nx=24)
    st = r.state.state
    fl = st.kinds == 0
    t = float(r.state.t)
    assert float(st.velocities[fl, 0].mean()) == pytest.approx(f * t, rel=0.02)
    assert float(st.velocities[fl, 1].abs().max()) < 0.05 * f * t


def test_the_body_force_enters_the_wall_pressure_condition_only_when_asked():
    """`bodyForceAtWall`: the hydrostatic offset A of the wall carries `rho0 (g + f)`; off, the wall sees gravity only (zero here). Linear in f."""
    from warpSPH.modules.analyticBoundary import resolveWall
    out = {}
    for key, params in (('none', {'f': 0.0}), ('at wall', {'f': 0.05}), ('at wall x2', {'f': 0.10}), ('not at wall', {'f': 0.05, 'bodyForceAtWall': False})):
        r = channel(params)
        sc = r.ctx.schemeConfig
        sc.boundaryProvider._wallCache = None
        out[key] = resolveWall(r.state.state, r.ctx.config, sc, r.state.adjacency).A.clone()
    assert float(out['none'].abs().max()) == 0.0
    assert float(out['not at wall'].abs().max()) == 0.0
    a1, a2 = out['at wall'], out['at wall x2']
    assert float(a1.abs().max()) > 0.0
    assert float((a2 - 2.0 * a1).abs().max()) < 1e-4 * float(a2.abs().max())


def test_the_channel_obeys_the_momentum_balance():
    """the body force on the fluid equals the load on the plates plus the rate of change of the fluid momentum (momentum-conserving wall terms, position-only shift): `F_plates / M + d(mean u) / dt = f`."""
    f = 0.05
    r = channel({'f': f}, nSteps=2500, nx=32)
    t = np.asarray(r.series('t'))
    u = np.asarray(r.series('meanVelocity'))
    F = np.asarray(r.series('plateLoad'))
    M = np.asarray(r.series('fluidMass'))
    lo, hi = len(t) // 2, len(t) - 1
    dudt = (u[hi] - u[lo]) / (t[hi] - t[lo])
    assert (F[lo:hi] / M[lo:hi]).mean() + dudt == pytest.approx(f, rel=0.05)
    assert u[hi] > 0.0
