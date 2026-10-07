"""Pairwise artificial-viscosity formulations vs the papers (AV_PLAN Phase 1 audit, 2026-10-07).

`computePi_actual` returns the term times `rho_j`; the viscosity kernel (`wp_diffusion.py`) adds
`(m_j / rho_j) * pi * w * grad_i W` to the acceleration of i, with `w = (v_ij . x_ij) / r_ij`
(negative for approaching pairs). So with `m_j = 1`:

    coefficient of grad_i W   =   (1 / rho_j) * pi * w   =   -Pi_ab           (Monaghan's sign)

Each test builds `-Pi_ab` independently in NumPy from the paper's equation, on random approaching pairs
(unequal h, c, rho, alpha), and compares it with the code, symmetrised over the `useJ` calls where the
formulation is one-sided by design. Sources, all in `literature/`:

* Monaghan (2005) Rep. Prog. Phys. 68, 1703, Eqs. (8.3)-(8.4) MG1983, (8.8)-(8.9) Cleary, (8.10) Monaghan 1992,
  (8.11)-(8.12) signal-velocity form (K = 1/2).
* Price (2012) JCP 231, 759, Eq. (98) (Monaghan 1992), (101)+(103) (signal-velocity form, beta/2 = C_q),
  Price (2008) conductivity `v_sig^u = sqrt(|P_a - P_b| / rho_bar)` (Eq. 104 and text).
No source on disk: Monaghan & Gingold 1997b/Dukowicz (4.7), (4.8), Wadsley 2008, Cleary 1998 (the 19.8) beyond what the
review quotes -- those are not asserted here.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import warp as wp
from warp.types import vector

from warpSPHCore import DomainDescription, scalar_t, sphKernelScale
from warpSPHCore.coreOperations._jvpCommon import buildDomainState
from warpSPHCore.dataTypes.domain_t import domainData
from warpSPHCore.util import castTorchToWarp, castTorchToWarpAsBuiltins

from warpSPH.configurations.moduleConfigurations.diffusionParameters import (
    BetaMode, DiffusionParameters, ViscosityTerms, buildDefaultDiffusionParamsCompressibleSPH)
from warpSPH.modules.dissipation.pi import computeFrontiereQ, computePi_actual

DIM = 2
vec = vector(length=DIM, dtype=scalar_t)
C_L, C_Q = 1.3, 0.7                      # unequal, so a swapped coefficient cannot hide


@wp.kernel
def _piKernel(
    x_i: wp.array(dtype=vec), x_j: wp.array(dtype=vec), v_i: wp.array(dtype=vec), v_j: wp.array(dtype=vec),
    h_i: wp.array(dtype=scalar_t), h_j: wp.array(dtype=scalar_t),
    rho_i: wp.array(dtype=scalar_t), rho_j: wp.array(dtype=scalar_t),
    c_i: wp.array(dtype=scalar_t), c_j: wp.array(dtype=scalar_t),
    a_i: wp.array(dtype=scalar_t), a_j: wp.array(dtype=scalar_t),
    p_i: wp.array(dtype=scalar_t), p_j: wp.array(dtype=scalar_t),
    domainState: domainData, params: DiffusionParameters, thermal: wp.bool,
    outI: wp.array(dtype=scalar_t), outJ: wp.array(dtype=scalar_t),
):
    k = wp.tid()
    one = scalar_t(1.0)
    outI[k] = computePi_actual(
        x_i[k], x_j[k], h_i[k], h_j[k], one, one, rho_i[k], rho_j[k], True, p_i[k], p_j[k],
        v_i[k], v_j[k], domainState, 1, c_i[k], c_j[k], a_i[k], a_j[k], params, False, thermal)
    outJ[k] = computePi_actual(
        x_i[k], x_j[k], h_i[k], h_j[k], one, one, rho_i[k], rho_j[k], True, p_i[k], p_j[k],
        v_i[k], v_j[k], domainState, 1, c_i[k], c_j[k], a_i[k], a_j[k], params, True, thermal)


class Pairs:
    """Random approaching pairs and the code's `pi` for a formulation (both `useJ` values)."""

    def __init__(self, term: ViscosityTerms, thermal: bool = False, n: int = 256, seed: int = 0,
                 approaching: bool = True, cl: float = C_L, cq: float = C_Q, betaMode: BetaMode = BetaMode.Coupled):
        wp.init()
        dtype = torch.float64 if scalar_t is wp.float64 else torch.float32
        dev = 'cuda' if wp.get_cuda_device_count() else 'cpu'
        g = torch.Generator().manual_seed(seed)

        def rnd(*shape, lo=0.0, hi=1.0):
            return torch.rand(*shape, generator=g, dtype=dtype) * (hi - lo) + lo

        xi = rnd(n, DIM)
        dx = rnd(n, DIM, lo=-0.1, hi=0.1)
        dx = dx / dx.norm(dim=1, keepdim=True) * rnd(n, 1, lo=0.03, hi=0.12)      # r ~ O(h)
        xj = xi - dx                                                             # x_ij = x_i - x_j = dx
        vj = rnd(n, DIM, lo=-1, hi=1)
        sign = 1.0 if approaching else -1.0
        vi = vj - sign * rnd(n, 1, lo=0.2, hi=2.0) * dx / dx.norm(dim=1, keepdim=True)   # u . x < 0: approaching
        self.t = dict(xi=xi, xj=xj, vi=vi, vj=vj,
                      hi=rnd(n, lo=0.05, hi=0.15), hj=rnd(n, lo=0.05, hi=0.15),
                      rhoi=rnd(n, lo=0.5, hi=2.0), rhoj=rnd(n, lo=0.5, hi=2.0),
                      ci=rnd(n, lo=0.5, hi=1.5), cj=rnd(n, lo=0.5, hi=1.5),
                      ai=rnd(n, lo=0.1, hi=1.0), aj=rnd(n, lo=0.1, hi=1.0),
                      pi=rnd(n, lo=0.5, hi=2.0), pj=rnd(n, lo=0.5, hi=2.0))
        params = buildDefaultDiffusionParamsCompressibleSPH()
        params.viscosityTerm = term.value
        params.thermalConductivityTerm = term.value
        params.C_l, params.C_q, params.Cu_l, params.Cu_q = cl, cq, cl, cq
        params.betaMode = betaMode.value
        params.correctXi = False
        params.monaghanSwitch = True
        domain = DomainDescription(min=torch.full((DIM,), -10.0, dtype=dtype, device=dev),
                                   max=torch.full((DIM,), 10.0, dtype=dtype, device=dev),
                                   periodic=torch.tensor([False] * DIM, device=dev), dim=DIM)
        t = {k: v.to(dev) for k, v in self.t.items()}
        outs = [wp.zeros(n, dtype=scalar_t, device=dev) for _ in range(2)]
        wp.launch(_piKernel, dim=n, device=dev, inputs=[
            castTorchToWarpAsBuiltins(t['xi']), castTorchToWarpAsBuiltins(t['xj']),
            castTorchToWarpAsBuiltins(t['vi']), castTorchToWarpAsBuiltins(t['vj']),
            castTorchToWarp(t['hi']), castTorchToWarp(t['hj']), castTorchToWarp(t['rhoi']), castTorchToWarp(t['rhoj']),
            castTorchToWarp(t['ci']), castTorchToWarp(t['cj']), castTorchToWarp(t['ai']), castTorchToWarp(t['aj']),
            castTorchToWarp(t['pi']), castTorchToWarp(t['pj']),
            buildDomainState(domain), params, thermal] + outs)
        wp.synchronize()
        self.piI, self.piJ = (wp.to_torch(o).cpu().double().numpy() for o in outs)
        a = {k: v.double().numpy() for k, v in self.t.items()}
        self.a = a
        x = a['xi'] - a['xj']
        self.r = np.linalg.norm(x, axis=1)
        self.w = ((a['vi'] - a['vj']) * x).sum(1) / self.r                         # < 0
        assert ((self.w < 0) if approaching else (self.w > 0)).all()
        self.hb = 0.5 * (a['hi'] + a['hj'])
        self.cb = 0.5 * (a['ci'] + a['cj'])
        self.rb = 0.5 * (a['rhoi'] + a['rhoj'])
        self.ab = 0.5 * (a['ai'] + a['aj'])
        self.al = self.ab * cl                                                    # scaleBeta off
        self.aq = self.ab * cq if betaMode == BetaMode.Coupled else cq

    def coeff(self, pi):
        """Coefficient of grad_i W in the acceleration of i (m_j = 1): (1 / rho_j) pi w."""
        return pi * self.w / self.a['rhoj']

    @property
    def codeI(self):
        return self.coeff(self.piI)

    @property
    def codeSym(self):
        return 0.5 * (self.coeff(self.piI) + self.coeff(self.piJ))


def _close(a, b, rtol=2e-4):
    np.testing.assert_allclose(a, b, rtol=rtol, atol=1e-7)


def test_monaghan1992_vs_price2012_eq98_and_monaghan2005_8_10():
    """-Pi = (alpha cbar mu - beta mu^2) / rhobar, mu = hbar w / r  (eps -> 0). The code is one-sided in c
    (c_i, c_j) by design for the two-call schemes, so compare the average of the two `useJ` calls."""
    p = Pairs(ViscosityTerms.Monaghan1992)
    mu = p.hb * p.w / p.r
    _close(p.codeSym, (p.al * p.cb * mu - p.aq * mu ** 2) / p.rb)


def test_price2012_98_label_is_the_signal_velocity_form_eq101_with_beta_over_2():
    """`Price2012_98` (the Monaghan-scheme default) is Price's Eq. (101)/(103) form, not Eq. (98):
    -Pi = alpha (cbar - (beta/2) w) w / rhobar with alpha = C_l, beta/2 = C_q (no h/r factor)."""
    p = Pairs(ViscosityTerms.Price2012_98)
    _close(p.codeI, (p.al * p.cb - p.aq * p.w) * p.w / p.rb)


def test_price2012_eq103_signal_velocity():
    """v_sig = (c_a + c_b - beta w) / 2 with beta = C_q: -Pi = alpha v_sig w / rhobar."""
    p = Pairs(ViscosityTerms.Price2012)
    _close(p.codeI, (p.al * p.cb - 0.5 * p.aq * p.w) * p.w / p.rb)


def test_monaghan1997a_is_monaghan2005_8_11_with_K_one_half():
    """K v_sig = (c_a + c_b - beta w) / 2 (K = 1/2): -Pi = (alpha cbar - (beta/2) w) w / rhobar; C_q = beta / 2."""
    p = Pairs(ViscosityTerms.Monaghan1997a)
    _close(p.codeI, (p.al * p.cb - p.aq * p.w) * p.w / p.rb)


def test_monaghan_gingold_1983_vs_monaghan2005_8_3_8_4():
    """Pi = -nu (v.r) / r^2, nu = alpha hbar cbar / rhobar  =>  -Pi = alpha hbar cbar w / (rhobar r)."""
    p = Pairs(ViscosityTerms.MonaghanGingold1983)
    _close(p.codeI, p.al * p.hb * p.cb * p.w / (p.rb * p.r))


def test_cleary1998_vs_monaghan2005_8_8_8_9():
    """mu_a = (1/8) alpha_a h_a c_a rho_a (2D), Pi = -19.8 mu_a mu_b / (rho_a rho_b (mu_a + mu_b)) (v.r)/r^2
    (16 -> 19.8, Cleary's experimental coefficient quoted in the review); alpha_a is the particle's own switch value
    times C_l."""
    p = Pairs(ViscosityTerms.Cleary1998)
    a = p.a
    f = 1.0 / 8.0
    mui = f * a['ai'] * C_L * a['hi'] * a['ci'] * a['rhoi']
    muj = f * a['aj'] * C_L * a['hj'] * a['cj'] * a['rhoj']
    ref = 19.8 * mui * muj / (a['rhoi'] * a['rhoj'] * (mui + muj)) * p.w / p.r
    _close(p.codeSym, ref)


def test_price2008_conductivity_signal_speed():
    """Conductivity (`thermalConductivity=True`): v_sig^u = sqrt(|P_a - P_b| / rhobar) (Price 2008), times alpha_u."""
    p = Pairs(ViscosityTerms.Price2008, thermal=True)
    a = p.a
    vsigU = np.sqrt(np.abs(a['pi'] - a['pj']) / p.rb)
    _close(p.piI, a['rhoj'] / p.rb * p.al * vsigU)


@pytest.mark.parametrize('term', [ViscosityTerms.Default, ViscosityTerms.Wadsley2008])
def test_default_and_wadsley2008_are_the_monaghan1992_mu_form_with_all_bars(term):
    """Wadsley et al. (2017) Eqs. (17)-(18) / Price (2012) Eq. (98): -Pi = (alpha cbar mu - beta mu^2) / rhobar,
    mu = hbar w / r. Every quantity is a pair mean, so the two `useJ` calls agree."""
    p = Pairs(term)
    mu = p.hb * p.w / p.r
    ref = (p.al * p.cb * mu - p.aq * mu ** 2) / p.rb
    _close(p.codeI, ref)
    _close(p.codeI, p.coeff(p.piJ))


def test_delta_sph_viscosity_vs_marrone2011_eq5():
    """Marrone et al. (2011) Eq. (5): Du_i/Dt += alpha h c0 (rho0 / rho_i) sum_j pi_ij grad_i W V_j with
    pi_ij = (u_ij . x_ij) / r^2 -- so -Pi_ab = alpha h c (rho0 / rho_i) w / (rho_j r) for m_j = 1 (rho0 -> rho_j here; the
    code has no reference density). No quadratic term in the paper: C_q = 0."""
    p = Pairs(ViscosityTerms.DeltaSPH, cq=0.0)
    a = p.a
    _close(p.codeI, p.al * p.hb * p.cb * p.w / (a['rhoi'] * p.r))


@pytest.mark.parametrize('mode', list(BetaMode))
def test_beta_mode_at_the_pair_level(mode):
    """`BetaMode.Fixed`: beta = C_q whatever the switched alpha is (Garcia-Senz & Cabezon 2026, PHANTOM);
    `Coupled`: beta follows alpha_bar. alpha stays alpha_bar * C_l either way."""
    p = Pairs(ViscosityTerms.Monaghan1997a, betaMode=mode)
    ref = (p.al * p.cb - (C_Q if mode == BetaMode.Fixed else p.ab * C_Q) * p.w) * p.w / p.rb
    _close(p.codeI, ref)


@pytest.mark.parametrize('term', [ViscosityTerms.Price2012_98, ViscosityTerms.Monaghan1997a,
                                  ViscosityTerms.Monaghan1992, ViscosityTerms.Cleary1998])
def test_receding_pairs_switch_off(term):
    """Monaghan's switch: no viscosity for receding pairs (and some, for the approaching ones)."""
    receding = Pairs(term, approaching=False)
    assert np.all(receding.piI == 0.0) and np.all(receding.piJ == 0.0)
    approaching = Pairs(term)
    assert np.all(approaching.piI != 0.0)


@wp.kernel
def _qKernel(
    x_i: wp.array(dtype=vec), x_j: wp.array(dtype=vec), v_i: wp.array(dtype=vec), v_j: wp.array(dtype=vec),
    h_i: wp.array(dtype=scalar_t), h_j: wp.array(dtype=scalar_t),
    rho_i: wp.array(dtype=scalar_t), rho_j: wp.array(dtype=scalar_t),
    c_i: wp.array(dtype=scalar_t), c_j: wp.array(dtype=scalar_t),
    a_i: wp.array(dtype=scalar_t), a_j: wp.array(dtype=scalar_t),
    domainState: domainData, params: DiffusionParameters,
    outQi: wp.array(dtype=scalar_t), outQj: wp.array(dtype=scalar_t), outScale: wp.array(dtype=scalar_t),
):
    k = wp.tid()
    u = v_i[k] - v_j[k]
    outQi[k] = computeFrontiereQ(x_i[k], x_j[k], h_i[k], h_j[k], rho_i[k], rho_j[k], u, domainState, 1,
                                 c_i[k], c_j[k], a_i[k], a_j[k], params, False)
    outQj[k] = computeFrontiereQ(x_i[k], x_j[k], h_i[k], h_j[k], rho_i[k], rho_j[k], u, domainState, 1,
                                 c_i[k], c_j[k], a_i[k], a_j[k], params, True)
    outScale[k] = sphKernelScale(1, 2)


@pytest.mark.parametrize('approaching', [True, False])
@pytest.mark.parametrize('mode', list(BetaMode))
def test_frontiere2017_q_vs_eq69(approaching, mode):
    """Frontiere et al. (2017) Eq. (69): Q_i = rho_i (-C_l c_i mu_i + C_q mu_i^2), mu_i = min(0, v.eta_i / (eta_i.eta_i + eps^2)),
    eta_i = x_ij / h_i (h_i the smoothing length = support / kernel scale), eps^2 = 1e-2; one-sided in rho, c, h and the
    switched alpha (beta per `BetaMode`). Q_j is the same with the j values and eta_j = x_ij / h_j (CRKSPH's convention)."""
    p = Pairs(ViscosityTerms.Monaghan1992, approaching=approaching, betaMode=mode)   # the generator and the params
    a, t = p.a, p.t
    dev = 'cuda' if wp.get_cuda_device_count() else 'cpu'
    dtype = torch.float64 if scalar_t is wp.float64 else torch.float32
    n = len(p.r)
    params = buildDefaultDiffusionParamsCompressibleSPH()
    params.C_l, params.C_q, params.betaMode, params.monaghanSwitch = C_L, C_Q, mode.value, False   # min(0, .) is built in
    domain = DomainDescription(min=torch.full((DIM,), -10.0, dtype=dtype, device=dev),
                               max=torch.full((DIM,), 10.0, dtype=dtype, device=dev),
                               periodic=torch.tensor([False] * DIM, device=dev), dim=DIM)
    td = {k: v.to(dev) for k, v in t.items()}
    outs = [wp.zeros(n, dtype=scalar_t, device=dev) for _ in range(3)]
    wp.launch(_qKernel, dim=n, device=dev, inputs=[
        castTorchToWarpAsBuiltins(td['xi']), castTorchToWarpAsBuiltins(td['xj']),
        castTorchToWarpAsBuiltins(td['vi']), castTorchToWarpAsBuiltins(td['vj']),
        castTorchToWarp(td['hi']), castTorchToWarp(td['hj']), castTorchToWarp(td['rhoi']), castTorchToWarp(td['rhoj']),
        castTorchToWarp(td['ci']), castTorchToWarp(td['cj']), castTorchToWarp(td['ai']), castTorchToWarp(td['aj']),
        buildDomainState(domain), params] + outs)
    wp.synchronize()
    Qi, Qj, scale = (wp.to_torch(o).cpu().double().numpy() for o in outs)
    x = a['xi'] - a['xj']
    u = a['vi'] - a['vj']

    def q(rho, c, h, alpha):
        hs = h / scale
        eta = x / hs[:, None]
        mu = np.minimum(0.0, (u * eta).sum(1) / ((eta * eta).sum(1) + 1e-2))
        Cl = alpha * C_L
        Cq = alpha * C_Q if mode == BetaMode.Coupled else C_Q
        return rho * (-Cl * c * mu + Cq * mu * mu)

    _close(Qi, q(a['rhoi'], a['ci'], a['hi'], a['ai']))
    _close(Qj, q(a['rhoj'], a['cj'], a['hj'], a['aj']))
    if not approaching:
        assert np.all(Qi == 0.0) and np.all(Qj == 0.0)
