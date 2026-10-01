"""CRKSPH regression guards (CRKSPH_LIMITER_PLAN P5).

The CRK limiter / viscosity are delicate: a tweak that cures one case easily adds ripple on another. These pin the
accepted behaviour of the default CRKSPH configuration at CI resolution, with margins of ~30 %:

* Sod (the paper's setup, mirrored tubes, symmetric support, no switch): post-shock plateau ringing, density total
  variation, entropy error and shock width -- the numbers the limiter constants trade against each other;
* Gresho: kinetic-energy dip/rebound over a short window (the spin-up check `ke_rebound`; the vortex must not gain
  energy faster than the pinned amount) and the L1 velocity error;
* total energy conserved to round-off with symmetric support (the CRK energy scheme's defining property).

Runs are deterministic (`tests/test_crkReproducible.py`), so a failure is a change in behaviour, not noise. Bounds were
set 2026-10-01 from the (1/n_h, 0.2/n_h) limiter default; when a deliberate change moves them, re-measure and say so in the
commit message.
"""

import contextlib
import io

import numpy as np
import pytest
import torch

from warpSPH.caseUtils.compressible.greshoVortex import greshoProfile
from warpSPH.caseUtils.compressible.sod.sodSolution import solve
from warpSPH.cases.greshoVortex import greshoVortexCase
from warpSPH.cases.sod import sodCase
from warpSPH.runner import run

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a GPU')


def _quiet(fn):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn()


def sodMetrics(nx=400, t=0.15):
    """Ringing / accuracy metrics of the mirrored-tube Sod state, in single-tube coordinate s = |x| (paper Sec. 4.3.1)."""
    res = _quiet(lambda: run(sodCase, scheme='CRKSPH', supportMode='KernelMeanSymmetric', nx=nx, tLimit=t,
                             progress=False, quiet=True))
    st = res.state.state
    p = sodCase.params
    gamma = float(p['gamma'])
    x = st.positions[:, 0].detach().cpu().double().numpy()
    v = st.velocities[:, 0].detach().cpu().double().numpy()
    rho = st.densities.detach().cpu().double().numpy()
    m = st.masses.detach().cpu().double().numpy()
    P = (gamma - 1.0) * rho * st.internalEnergies.detach().cpu().double().numpy()
    s, u = np.abs(x), v * np.sign(x)
    o = np.argsort(s)
    s, u, rho, P, m = s[o], u[o], rho[o], P[o], m[o]
    V = m / rho
    pos, reg, val = solve(left_state=(p['left_pressure'], p['left_rho'], p['left_velocity']),
                          right_state=(p['right_pressure'], p['right_rho'], p['right_velocity']),
                          geometry=(0.0, 1.0, 0.5), t=float(res.state.t), gamma=gamma, npts=20001)
    xs = np.asarray(val['x'])
    x_c, x_s = pos['Contact Discontinuity'], pos['Shock']
    P4, rho4, u4 = (float(q) for q in reg['Region 4'])
    A4 = P4 / rho4 ** gamma
    dpost = float(np.median(V[(s > x_c) & (s < x_s)]))
    plateau = (s > x_c + 4 * dpost) & (s < x_s - 4 * dpost)
    near, fine = (s > 0.2) & (s < 0.85), (xs > 0.2) & (xs < 0.85)
    tv = lambda a: float(np.abs(np.diff(a)).sum())
    rho5 = float(p['right_rho'])
    band = (s > x_c) & (s < x_s + 10 * dpost) & (rho > rho5 + 0.1 * (rho4 - rho5)) & (rho < rho5 + 0.9 * (rho4 - rho5))
    E = res.series('totalEnergy')
    return dict(l1_rho=float((V * np.abs(rho - np.interp(s, xs, val['rho']))).sum() / V.sum()),
                plateau_u_std=float(np.std(u[plateau] - u4)),
                tv_excess_rho=tv(rho[near]) - tv(np.asarray(val['rho'])[fine]),
                entropy_err=float(np.mean(np.abs(P[plateau] / rho[plateau] ** gamma - A4)) / A4),
                shock_width=int(band.sum()), e_drift=float(abs(E[-1] - E[0]) / abs(E[0])))


def greshoMetrics(nx=64, t=1.5):
    res = _quiet(lambda: run(greshoVortexCase, nx=nx, tLimit=t, progress=False, quiet=True))
    st = res.state.state
    ke = res.series('kineticEnergy')
    E = res.series('totalEnergy')
    x = st.positions.detach().double()
    v = st.velocities.detach().double()
    r = torch.linalg.norm(x, dim=-1)
    vphi = (x[:, 0] * v[:, 1] - x[:, 1] * v[:, 0]) / r.clamp(min=1e-30)
    m = r < 0.5
    return dict(ke_final_rel=float(ke[-1] / ke[0] - 1),
                ke_rebound=float((ke - np.minimum.accumulate(ke)).max() / ke[0]),
                l1_v=float((vphi - greshoProfile(r)[0]).abs()[m].mean()),
                e_drift=float(abs(E[-1] - E[0]) / abs(E[0])))


def test_sod_ripple_bounds():
    m = sodMetrics()
    assert m['e_drift'] < 1e-5, m
    assert m['plateau_u_std'] < SOD['plateau_u_std'], m
    assert m['tv_excess_rho'] < SOD['tv_excess_rho'], m
    assert m['entropy_err'] < SOD['entropy_err'], m
    assert m['l1_rho'] < SOD['l1_rho'], m
    assert m['shock_width'] <= SOD['shock_width'], m


def test_gresho_energy_window():
    m = greshoMetrics()
    assert m['e_drift'] < 1e-5, m
    assert m['ke_rebound'] < GRESHO['ke_rebound'], m          # the spin-up check: no gain beyond the pinned amount
    assert abs(m['ke_final_rel']) < GRESHO['ke_final_abs'], m
    assert m['l1_v'] < GRESHO['l1_v'], m


#: ~1.3x the values measured at CI resolution on 2026-10-01 (`python tests/test_crkRegression.py` prints them):
#: Sod nx 400, t = 0.15: l1_rho 5.7e-3, plateau_u_std 0.0120, tv_excess_rho 0.101, entropy_err 2.9e-3, shock_width 8.
#: Gresho nx 64, t = 1.5: ke_final_rel +2e-5, ke_rebound 0.043, l1_v 0.0249.
SOD = dict(plateau_u_std=0.0156, tv_excess_rho=0.132, entropy_err=0.0037, l1_rho=0.0074, shock_width=10)
GRESHO = dict(ke_rebound=0.056, ke_final_abs=0.05, l1_v=0.033)

if __name__ == '__main__':
    print('sod   ', sodMetrics())
    print('gresho', greshoMetrics())
