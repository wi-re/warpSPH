"""AV_PLAN Phase 5B: the Sphenix switch (Borrow et al. 2022, `modules/shockCapturing/Sphenix2022.py`).

Eq. (24) is implicit, which is what makes the decay stable for any dt / tau and keeps alpha >= 0 with alpha_min = 0;
these check that property directly, plus the local-alpha limits (Eq. 23), the shock indicator's sign (Eq. 21) and
that the instant raise wins over the decay. The Balsara factor (Eq. 20, in the pair coefficient) is
`tests/test_reconstruction.py::test_balsara_from_jacobian`.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from warpSPH.modules.shockCapturing.Sphenix2022 import (sphenixAlphaUpdate, sphenixLocalAlpha,
                                                        sphenixShockIndicator)


@pytest.mark.parametrize('dtOverTau', [0.1, 1.0, 10.0, 100.0])
def test_implicit_decay_is_stable(dtOverTau):
    alpha0 = torch.tensor([0.0, 0.3, 1.0, 2.0], dtype=torch.float64)
    tau = torch.ones_like(alpha0)
    alpha = alpha0.clone()
    for _ in range(1000):
        new = sphenixAlphaUpdate(alpha, torch.zeros_like(alpha), dtOverTau, tau)
        assert (new <= alpha).all(), 'not monotone decreasing'
        assert (new >= 0).all(), 'undershoots zero'
        alpha = new
    assert (alpha <= alpha0).all()


def test_decay_towards_alpha_loc():
    alpha = torch.tensor([2.0], dtype=torch.float64)
    for _ in range(200):
        alpha = sphenixAlphaUpdate(alpha, torch.tensor([0.5], dtype=torch.float64), 1.0, torch.ones(1, dtype=torch.float64))
    assert alpha.item() == pytest.approx(0.5, rel=1e-12)


def test_local_alpha_limits():
    c = torch.ones(3, dtype=torch.float64)
    S = torch.tensor([0.0, 1e-12, 1e12], dtype=torch.float64)
    a = sphenixLocalAlpha(S, c, 2.0)
    assert a[0] == 0.0 and a[1] < 1e-11 and a[2] == pytest.approx(2.0, rel=1e-11)


def test_local_alpha_cold_gas():
    """c = 0 (Sedov's ambient gas): no NaN -- 0 without a source, alpha_max with one."""
    S = torch.tensor([0.0, 1e-6], dtype=torch.float32)
    a = sphenixLocalAlpha(S, torch.zeros(2, dtype=torch.float32), 2.0)
    assert torch.isfinite(a).all() and a[0] == 0.0 and a[1] == pytest.approx(2.0)


def test_instant_raise_takes_precedence():
    alpha = torch.tensor([0.1], dtype=torch.float64)
    out = sphenixAlphaUpdate(alpha, torch.tensor([1.5], dtype=torch.float64), 1e-6, torch.ones(1, dtype=torch.float64))
    assert out.item() == 1.5


def test_shock_indicator_sign():
    """S >= 0, nonzero only where the flow converges (div <= 0) and the divergence is falling (Eq. 21: the PDF's
    `-h^2 max(ddiv/dt, 0)` typesetting has the minus inside the max)."""
    h = torch.ones(4, dtype=torch.float64)
    div = torch.tensor([-1.0, -1.0, 1.0, 0.0], dtype=torch.float64)
    ddivdt = torch.tensor([-2.0, 2.0, -2.0, -3.0], dtype=torch.float64)
    S = sphenixShockIndicator(div, ddivdt, h)
    assert S.tolist() == [2.0, 0.0, 0.0, 3.0]


def test_sod_runs_and_switch_engages():
    """A short Sod run under the Sphenix config: alpha leaves zero at the shock, stays finite, and the run is clean."""
    from warpSPH.cases.sod import sodCase
    from warpSPH.runner import run

    def configure(ctx, _orig=sodCase.configureScheme):
        _orig(ctx)
        dp = ctx.schemeConfig.diffusionParams
        dp.viscosityTerm, dp.C_l, dp.C_q, dp.betaMode = 8, 1.0, 3.0, 0
        dp.balsaraPairLimiter, dp.correctReconstructionGradient = True, False
        sw = ctx.schemeConfig.viscositySwitchParams
        sw.alpha_min, sw.alpha_max = 0.0, 2.0

    res = run(dataclasses.replace(sodCase, configureScheme=configure), scheme='Monaghan', nx=200, nSteps=150,
              progress=False, quiet=True,
              params=dict(viscositySwitch='Sphenix2022', right_rho=0.125, right_pressure=0.1))
    assert not res.diverged
    alpha = res.state.state.alphas
    assert torch.isfinite(alpha).all()
    assert alpha.max() > 0.5, f'alpha never rose at the shock (max {alpha.max().item():.3g})'
    assert alpha.min() >= 0
