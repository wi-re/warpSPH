"""Equation-level unit checks for the Cullen & Dehnen (2010) shock-capturing
switch, plus a 1D Sod contact-spike check.

The unit checks call the switch's OWN functions (`computeR`, `computeXi`,
`computeSecondOrderV`) on a manufactured 1D state and verify the paper's
equation forms (eq. 17, 18 and the App.-B minus sign). The Sod check runs the
tube under the switch and bounds the contact-overshoot metric.

Deliberately small (coarse resolution, few steps) -- the point is to catch a
wrong equation form or a disconnected term, not to re-derive the validation
numbers (those live in `phase6_shock_capturing_log.md` and the overlay PNGs).

Sod geometry note: `buildSod1D` lays out a mirror-symmetric two-interface tube
(dense state |x| <= L/4, light state the outer quarters, interfaces at x=+/-L/4),
so the analytic Riemann solution describes only the window x in [0, L/2] = [0,1]
(L=2) with the interface at x=+0.5 -- the same window `sodUtil.plotSod_` overlays
(`solve(..., geometry=(0., 1., 0.5), ...)`).
"""

import io
import contextlib

import numpy as np
import pytest
import torch

from warpSPH.runner import run, buildContext, CaseSpec

# The window the analytic Sod solution describes (see module docstring).
SOD_XL, SOD_XR, SOD_XI = 0.0, 1.0, 0.5
GAMMA = 5 / 3


def _quiet(fn):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn()


@pytest.fixture(scope='module')
def sod1d():
    """The 1D Sod state under the C&D switch, built once for the unit checks.

    Returns (ctx, state, adjacency). The unit checks overwrite `state.velocities`
    with manufactured fields; the adjacency is position/support based, so it stays
    valid. `state` is a fresh clone per test (the overwrites would otherwise leak
    between tests within the module scope).
    """
    from warpSPH.cases.sod import sodCase
    from warpSPH.modules.shockCapturing.CullenDehnen2010 import (computeR,
                                                                  computeXi,
                                                                  computeSecondOrderV)

    spec = CaseSpec(caseName=sodCase.name, scheme='Monaghan',
                    params=dict(sodCase.params)).merged(**sodCase.defaults)
    spec = spec.merged(nx=200, dim=1, plot=False, store=False, quiet=True,
                       params=dict(sodCase.params, viscositySwitch='CullenDehnen2010'))
    ctx = buildContext(sodCase, spec)
    sodCase.configureScheme(ctx)
    system = _quiet(lambda: sodCase.buildSystem(ctx))
    return SimpleSod(ctx, system, computeR, computeXi, computeSecondOrderV)


class SimpleSod:
    """Small holder so the fixture returns a tuple without positional ambiguity."""
    def __init__(self, ctx, system, computeR, computeXi, computeSecondOrderV):
        self.ctx = ctx
        self.system = system
        self.computeR = computeR
        self.computeXi = computeXi
        self.computeSecondOrderV = computeSecondOrderV

    @property
    def state(self):
        return self.system.state

    @property
    def adjacency(self):
        return self.system.adjacency


# --- eq. 17: the sign-weighted divergence limiter R -------------------------

def test_R_eq17_isOneForUniformSignDivergence(sod1d):
    """R_i = (1/rho_i) sum_j sign(div v_j) m_j W_ij.

    When every neighbour's divergence has the same sign, the sign factor is a
    constant and R collapses to (1/rho_i) times the SPH density estimate --
    the partition of unity, i.e. 1 in the interior of a region. A wrong sign
    convention (or a missing density prefactor) would move R away from 1.
    """
    state, adj = sod1d.state, sod1d.adjacency
    div = torch.ones_like(state.densities)  # uniformly positive divergence
    R = sod1d.computeR(div, state, sod1d.ctx.config, sod1d.ctx.schemeConfig,
                       None, adj)
    # Interior of the dense block, away from the x=0 symmetry line and the
    # x=+0.5 interface, where the density estimate is a clean partition of unity.
    x = state.positions[:, 0].detach().cpu().numpy()
    m = (x >= 0.05) & (x <= 0.40)
    assert m.sum() > 10
    Rm = R[m].detach().cpu().numpy()
    assert np.allclose(Rm, 1.0, atol=0.1), f'R not ~1 in the interior: mean={Rm.mean():.3f}'


# --- eq. 18: the shear limiter Xi (plain C&D path, S is None) ---------------

def test_Xi_eq18_isOneWithoutShearTensor(sod1d):
    """Xi = |beta(1-R)^4 div|^2 / (|beta(1-R)^4 div|^2 + tr(S S^t)).

    In the plain C&D path the shear tensor S is None (tr=0) and `limitXi` is on,
    so Xi = (nom^2 + eps) / (nom^2 + eps) = 1 for ANY input -- the switch is then
    driven purely by the shock indicator, the paper's "simplified" form. This is
    input-independent, so a regression in the formula (e.g. dropping the eps term
    or the square) shows up here.
    """
    state = sod1d.state
    div = torch.randn_like(state.densities)  # arbitrary
    R = torch.rand_like(state.densities)  # arbitrary, in [0, 1)
    for params, expect in ((sod1d.ctx.schemeConfig, 1.0),):
        Xi = sod1d.computeXi(div, None, R, state, sod1d.ctx.config, params, None,
                             sod1d.adjacency)
        assert torch.allclose(Xi, torch.full_like(Xi, expect), atol=1e-4), \
            f'Xi(S=None) should be {expect}, got min={Xi.min():.4f} max={Xi.max():.4f}'


# --- App. B eq. B11: the material-derivative sign (div(dv/dt) - tr(V^2)) ----

def test_secondOrderV_hasTheMinusSign(sod1d):
    """computeSecondOrderV must return D(div v)/Dt = div(dv/dt) - tr(V^2), NOT
    div(dv/dt) + tr(V^2).

    On a smooth periodic 1D field v = A sin(kx) with acceleration dv/dt =
    B sin(kx):  div(dv/dt) = B k cos(kx) and tr(V^2) = (A k)^2 cos^2(kx)
    (V = dv/dx, so tr(V^2) >= 0). Choose A >> B so that tr(V^2) dominates
    div(dv/dt) wherever |cos(kx)| is large. There the two candidate formulas
    have OPPOSITE signs:  div - tr(V^2) < 0  but  div + tr(V^2) > 0. So the
    sign of the returned field identifies the formula, robustly -- it does not
    depend on the SPH gradient/divergence being spectrally accurate, only on
    tr(V^2) being positive (it is a square) and large enough to dominate.
    """
    from warpSPHCore import SupportScheme
    state, adj = sod1d.state, sod1d.adjacency
    x = state.positions[:, 0]
    A, B, k = 0.3, 0.01, 2.0 * torch.pi  # (A k)^2 ~ 3.55 dominates B k ~ 0.063
    v = (A * torch.sin(k * x)).unsqueeze(-1)
    dvdt = (B * torch.sin(k * x)).unsqueeze(-1)

    orig_v = state.velocities.clone()  # module-scoped state: don't leak the field
    state.velocities = v
    try:
        got = sod1d.computeSecondOrderV(dvdt, None, state, sod1d.ctx.config,
                                        sod1d.ctx.schemeConfig,
                                        SupportScheme.SuperSymmetric, adj)
    finally:
        state.velocities = orig_v
    got = got.detach().cpu().numpy().ravel()

    xx = x.detach().cpu().numpy()
    # Dense, uniformly-sampled block, where |cos(kx)| is large so tr(V^2) dominates.
    m = (np.abs(xx) <= 0.40) & (np.abs(np.cos(k * xx)) > 0.5)
    assert m.sum() > 5
    # Minus sign: dominated by -tr(V^2) -> strongly negative. Plus sign would be
    # dominated by +tr(V^2) -> positive. So a negative result rules out the plus.
    assert got[m].max() < -0.5, (
        f'D(div v)/Dt is not div(dv/dt) - tr(V^2): where tr(V^2) dominates the '
        f'result must be ~ -(A k)^2 cos^2 ~ -3.5, got max {got[m].max():+.3f}')
    assert np.all(got[m] < 0), f'result not negative where tr(V^2) dominates: max {got[m].max():+.3f}'


# --- 1D Sod: the contact-spike metric under the switch ----------------------

def _sod_contact_spike(switch, nx=200, nSteps=300):
    """Run the 1D Sod under `switch` and return (t, contactX, P_spike, A_spike, alpha).

    The contact spike is the max pressure / specific-entropy in a window around
    the exact contact discontinuity, relative to the exact post-contact value.
    """
    from warpSPH.cases.sod import sodCase
    from warpSPH.caseUtils.compressible.sod.sodSolution import solve

    res = _quiet(lambda: run(
        sodCase, progress=False, quiet=True, scheme='Monaghan', nx=nx, nSteps=nSteps,
        params=dict(right_rho=0.125, right_pressure=0.1, viscositySwitch=switch)))
    st = res.state.state
    t = float(res.state.t)
    x = st.positions[:, 0].detach().cpu().numpy()
    rho = st.densities.detach().cpu().numpy()
    P = st.pressures.detach().cpu().numpy()
    A = P / (rho ** GAMMA)
    alpha = st.alphas.detach().cpu().numpy() if st.alphas is not None else None

    ic = ((1.0, 1.0, 0.0), (0.1, 0.125, 0.0))
    positions, regions, vals = solve(ic[0], ic[1], (SOD_XL, SOD_XR, SOD_XI), t,
                                     gamma=GAMMA, npts=1000)
    cx = positions['Contact Discontinuity']
    sel = (vals['x'] > cx) & (vals['x'] < cx + 0.04)
    p4 = float(np.median(vals['p'][sel]))
    rho4 = float(np.median(vals['rho'][sel]))
    A4 = p4 / (rho4 ** GAMMA)
    w = 0.07
    mw = (x > cx - w) & (x < cx + w)
    P_spike = float(P[mw].max() / p4 - 1.0)
    A_spike = float(A[mw].max() / A4 - 1.0)
    return t, cx, P_spike, A_spike, alpha, res


def test_sod1d_cullenDehnenRunsAndSpikesBounded():
    """The C&D switch on the Monaghan Sod runs without diverging, keeps the
    contact spike bounded, and puts its alpha up at the shock and down in the
    smooth regions (the physical signature of the switch)."""
    t, cx, P_spike, A_spike, alpha, res = _sod_contact_spike('CullenDehnen2010')
    assert not res.diverged
    assert np.isfinite(P_spike) and np.isfinite(A_spike)
    assert abs(P_spike) < 0.5, f'contact pressure spike {P_spike:.2%} is not bounded'
    assert abs(A_spike) < 0.5, f'contact entropy spike {A_spike:.2%} is not bounded'
    # The switch must actually engage: some alpha near the shock, most alpha low.
    assert alpha is not None
    assert alpha.max() > 0.3, f'switch never engaged: max alpha {alpha.max():.3f}'
    assert alpha.mean() < 0.3, f'switch over-active: mean alpha {alpha.mean():.3f}'
