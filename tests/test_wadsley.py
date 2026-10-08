"""AV_PLAN Phase 6: the Wadsley et al. (2017) Gasoline2 detector (`modules/shockCapturing/Wadsley2017.py`).

Manufactured velocity fields on a 2D lattice, checking the detector directly -- "these are the point of the paper":
`D` is blind to uniform compression (where `div v`, and so every divergence-based detector, is not), takes the
extremal value `div v` in a planar compression, and `xi` is 1 in a shock, ~1/16 in uniform compression, 0 in
expansion. Plus the h-convention constant (§6.2) and a Sod run in which the switch engages.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from warpSPH.modules.shockCapturing.Wadsley2017 import WADSLEY_PREFACTOR, wadsleyD, wadsleyDetector, wadsleyXi
from warpSPH.runner import run

K = 0.7


def _state():
    from warpSPH.cases.greshoVortex import greshoVortexCase
    res = run(greshoVortexCase, scheme='Monaghan', nx=40, nSteps=1, progress=False, quiet=True)
    st = res.state.state
    # a pressure gradient along x: n = x-hat wherever the detector looks
    st.pressures = 1.0 + 0.2 * st.positions[:, 0]
    return res.state, res.ctx.config, res.ctx.schemeConfig


def _detect(field):
    system, config, schemeConfig = _state()
    st = system.state
    st.velocities = field(st.positions)
    D, Tnorm, R, n = wadsleyDetector(st, config, system.adjacency, schemeConfig.gamma)
    # two support radii from the periodic seam: the gradients and the R average see only the manufactured field
    inner = st.positions.abs().amax(dim=1) < 0.5 - 2.1 * st.supports
    assert int(inner.sum()) > 50
    return D[inner], R[inner], wadsleyXi(R[inner]), n[inner]


def test_uniform_compression_is_invisible():
    """The discriminating test: v = -k r has div v = -2k (C&D's indicator fires), but D = 0 and xi = 1/16."""
    D, R, xi, n = _detect(lambda x: -K * x)
    assert (n[:, 0] > 0.99).all()
    assert D.abs().max() < 1e-4 * K
    assert xi.sub(1 / 16).abs().max() < 1e-3


def test_planar_compression_is_a_shock():
    """v = (-k x, 0): D takes the extremal value div v = -k, R = -1, xi = 1."""
    D, R, xi, _ = _detect(lambda x: torch.stack([-K * x[:, 0], torch.zeros_like(x[:, 0])], 1))
    assert torch.allclose(D, torch.full_like(D, -K), atol=1e-4 * K)
    assert xi.min() > 0.999


def test_expansion_gives_zero_xi():
    D, R, xi, _ = _detect(lambda x: torch.stack([K * x[:, 0], torch.zeros_like(x[:, 0])], 1))
    assert xi.max() < 1e-6


def test_rotation_is_invisible():
    D, R, xi, _ = _detect(lambda x: torch.stack([-K * x[:, 1], K * x[:, 0]], 1))
    assert D.abs().max() < 1e-4 * K


def test_pure_shear_is_invisible():
    """v = (k y, 0): dv/dn = n V n = 0 for n along the pressure gradient (x), so D = 0. Holds only while n is set by a
    real pressure gradient: in uniform pressure n = grad P / |grad P| is the direction of the noise (AV_PLAN Phase 6)."""
    D, R, xi, _ = _detect(lambda x: torch.stack([K * x[:, 1], torch.zeros_like(x[:, 0])], 1))
    assert D.abs().max() < 1e-3 * K


def test_D_dimension_general():
    """Eq. (24) generalised: blind to V = -k I and equal to div v for a planar compression, in 2D and 3D (in 3D it
    is the paper's 3/2 [dv/dn + 1/3 max(-div, 0)])."""
    for d in (2, 3):
        n = torch.zeros(1, d, dtype=torch.float64); n[0, 0] = 1.0
        assert wadsleyD(-K * torch.eye(d, dtype=torch.float64)[None], n).abs().item() < 1e-15
        planar = torch.zeros(1, d, d, dtype=torch.float64); planar[0, 0, 0] = -K
        assert wadsleyD(planar, n).item() == pytest.approx(-K, rel=1e-14)
    n = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    V = torch.tensor([[[-1.0, 0.2, 0.0], [0.1, -0.3, 0.0], [0.0, 0.0, 0.4]]], dtype=torch.float64)
    paper = 1.5 * (n @ V[0] @ n.T + max(-torch.einsum('ii', V[0]).item(), 0.0) / 3)
    assert wadsleyD(V, n).item() == pytest.approx(paper.item(), rel=1e-14)


def test_h_convention_prefactor():
    """§6.2: Gasoline2's h is half the support this repo stores, so Eq. (26)'s 2 h_G2^2 is 0.5 H^2."""
    assert WADSLEY_PREFACTOR == pytest.approx(2 * 0.5 ** 2)


def _wadsleyCase(case):
    """`case` with the `wadsley2017` av_report operator: Wadsley2008 pair term, beta = 2 fixed, alpha in [0, 2]."""
    def configure(ctx, _orig=case.configureScheme):
        _orig(ctx)
        dp = ctx.schemeConfig.diffusionParams
        dp.viscosityTerm, dp.C_q, dp.betaMode = 10, 2.0, 1
        sw = ctx.schemeConfig.viscositySwitchParams
        sw.alpha_min, sw.alpha_max = 0.0, 2.0
    return dataclasses.replace(case, configureScheme=configure)


def test_sound_wave_keeps_alpha_at_floor():
    """A linear sound wave compresses (D = dv_x/dx != 0), but alpha_loc ~ A (k h)^2 is tiny: alpha stays near 0
    (the av_report linearWave run reads alphaMean 1.7e-5)."""
    from warpSPH.cases.linearWave import linearWaveCase
    res = run(_wadsleyCase(linearWaveCase), scheme='Monaghan', nx=100, nSteps=300, progress=False, quiet=True,
              params=dict(viscositySwitch='Wadsley2017', A=1e-4))
    assert not res.diverged
    alpha = res.state.state.alphas
    assert torch.isfinite(alpha).all()
    assert alpha.max() < 1e-3, f'alpha rose in a linear wave (max {alpha.max().item():.3g})'


def test_sod_runs_and_switch_engages():
    from warpSPH.cases.sod import sodCase

    res = run(_wadsleyCase(sodCase), scheme='Monaghan', nx=200, nSteps=150,
              progress=False, quiet=True,
              params=dict(viscositySwitch='Wadsley2017', right_rho=0.125, right_pressure=0.1))
    assert not res.diverged
    alpha = res.state.state.alphas
    assert torch.isfinite(alpha).all() and alpha.min() >= 0
    # engaged at the shock, off elsewhere. Not a C&D-sized peak: in common units Eq. (26)'s A is half of C&D's
    # (0.5 H^2 vs H^2), so alpha_loc saturates lower (0.38 at nx 200, step 150 when this was written)
    assert alpha.max() > 0.2, f'alpha never rose at the shock (max {alpha.max().item():.3g})'
    assert alpha.median() < 0.05, f'alpha is on away from the shock (median {alpha.median().item():.3g})'
