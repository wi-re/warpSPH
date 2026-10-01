"""AV_PLAN S3: the Balsara, Colagrossi, Morris-Monaghan and Rosswog (2000) viscosity switches."""

from __future__ import annotations

import pytest
import torch
from types import SimpleNamespace

from warpSPH.modules.shockCapturing.Balsara1995 import balsaraFactor
from warpSPH.modules.shockCapturing.switchRelaxation import relaxAlpha
from warpSPHCore import sphKernel_xi
from warpSPHCore import KernelFunctions
from warpSPH.modules.shockCapturing.MorrisMonaghan1997 import relaxedAlpha
from warpSPH.runner import run

_SOD = dict(right_rho=0.125, right_pressure=0.1)


# --- the relaxation (Morris & Monaghan Eq. 5), implicit decay ------------------------------

@pytest.mark.parametrize('dtOverTau', [0.1, 1.0, 10.0, 100.0])
def test_relaxation_is_stable_monotone_and_never_undershoots(dtOverTau):
    """With no source, alpha decays towards alpha_min for ANY dt/tau: no overshoot below the
    floor, no growth, monotone -- the property the implicit form buys."""
    lo, hi, tau = 0.1, 2.0, torch.tensor([1.0])
    alpha = torch.tensor([1.5])
    previous = alpha.item()
    for _ in range(1000):
        alpha = relaxAlpha(alpha, torch.zeros(1), tau, dtOverTau, lo, hi)
        assert lo <= alpha.item() <= previous + 1e-7
        previous = alpha.item()
    assert alpha.item() == pytest.approx(lo, abs=1e-6)


def test_relaxation_steady_state_is_alpha_min_plus_tau_S():
    lo, hi, tau, S = 0.1, 2.0, torch.tensor([0.5]), torch.tensor([0.8])
    alpha = torch.tensor([0.1])
    for _ in range(2000):
        alpha = relaxAlpha(alpha, S, tau, 0.05, lo, hi)
    assert alpha.item() == pytest.approx(lo + 0.5 * 0.8, rel=1e-4)
    # a huge source saturates at alpha_max
    assert relaxAlpha(alpha, torch.tensor([1e6]), tau, 1.0, lo, hi).item() == hi


def test_relaxation_half_life_matches_the_exponential_for_small_steps():
    """Decay from alpha_min + 1 over one e-fold: explicit-limit check of the implicit form."""
    lo, tau = 0.0, torch.tensor([1.0])
    alpha = torch.tensor([1.0])
    n, dt = 2000, 1.0 / 2000
    for _ in range(n):
        alpha = relaxAlpha(alpha, torch.zeros(1), tau, dt, lo, 2.0)
    assert alpha.item() == pytest.approx(torch.exp(torch.tensor(-1.0)).item(), rel=2e-3)


# --- the Balsara factor ---------------------------------------------------------------------

def test_balsara_limits():
    one = torch.ones(1)
    # pure compression (no curl) -> ~1, either sign of div; pure rotation (no div) -> ~0
    assert balsaraFactor(torch.tensor([-3.0]), torch.zeros(1), one, one, 1e-4).item() == pytest.approx(1.0, abs=1e-3)
    assert balsaraFactor(torch.tensor([3.0]), torch.zeros(1), one, one, 1e-4).item() == pytest.approx(1.0, abs=1e-3)
    assert balsaraFactor(torch.zeros(1), torch.tensor([3.0]), one, one, 1e-4).item() < 1e-4
    # equal div and curl -> 1/2
    assert balsaraFactor(torch.tensor([2.0]), torch.tensor([2.0]), one, one, 0.0).item() == pytest.approx(0.5, abs=1e-6)
    # finite when everything vanishes
    assert torch.isfinite(balsaraFactor(torch.zeros(1), torch.zeros(1), one, one, 1e-4)).all()


# --- wiring through the scheme: each switch runs, engages at the shock, and stays low in shear --

def _sod(switch, nSteps=200, **params):
    return run(__import__('warpSPH.cases.sod', fromlist=['sodCase']).sodCase, scheme='Monaghan',
               nx=100, nSteps=nSteps, progress=False, quiet=True,
               params=dict(_SOD, viscositySwitch=switch, **params))


def test_rosswog2000_steady_state_matches_the_paper_ode():
    """Rosswog et al. (2000) Eqs. (A.5)-(A.6): for a constant compression rate D = -div v the ODE
    d alpha / dt = -(alpha - a_min) / tau + D (a_max - alpha) has the fixed point
    (a_min / tau + D a_max) / (1 / tau + D), with tau = h / (eps c)."""
    aMin, aMax, eps, h, c, D = 0.05, 1.5, 0.2, 0.01, 2.0, 30.0
    cfg = SimpleNamespace(viscositySwitchParams=SimpleNamespace(alpha_min=aMin, alpha_max=aMax, morris_C1=eps))
    sim = SimpleNamespace(kernel=KernelFunctions.Wendland2)
    f_kern = 1.0 / sphKernel_xi(sim.kernel.value, 2)
    state = SimpleNamespace(positions=torch.zeros(1, 2), supports=torch.tensor([h / f_kern]),
                            soundspeeds=torch.tensor([c]), alpha0s=torch.tensor([aMin]))
    tau = h / (eps * c)
    for _ in range(4000):
        a = relaxedAlpha(state, torch.tensor([-D]), tau / 20, sim, cfg,
                         sourceScale=(aMax - state.alpha0s).clamp(min=0))
        state.alpha0s = a
    expected = (aMin / tau + D * aMax) / (1.0 / tau + D)
    assert a.item() == pytest.approx(expected, rel=1e-3)
    assert aMin < a.item() < aMax


@pytest.mark.parametrize('switch', ['MorrisMonaghan1997', 'Rosswog2000'])
def test_source_and_decay_switches_engage_at_the_shock_and_decay_elsewhere(switch):
    res = _sod(switch, alpha_min=0.1, alpha_max=2.0)
    assert not res.diverged
    a = res.state.state.alphas.detach().cpu()
    assert a.min() >= 0.1 - 1e-6 and a.max() <= 2.0 + 1e-6
    # Morris & Monaghan Eq. (19): without decay the peak is alpha_min + ln(v1/v2). This Sod shock
    # compresses the gas by ~1.8 (ln = 0.57), so the ceiling is ~0.67 and the decay pulls it
    # lower; the requirement is only that the switch rises well clear of alpha_min.
    assert a.max() > 0.3, f'{switch} never engaged: max alpha {a.max():.3f}'
    assert a.median() < 0.3, f'{switch} did not decay away from the shock: median {a.median():.3f}'


@pytest.mark.parametrize('switch', ['Balsara1995', 'Colagrossi2004'])
def test_limiter_switches_are_high_in_compression_and_low_in_shear(switch):
    sod = _sod(switch, nSteps=150)
    assert not sod.diverged
    a = sod.state.state.alphas.detach().cpu()
    assert 0.0 <= a.min() and a.max() <= 1.0 + 1e-6
    assert a.max() > 0.8, 'no compressive region found in Sod'

    from warpSPH.cases.greshoVortex import greshoVortexCase
    gresho = run(greshoVortexCase, scheme='Monaghan', nx=32, nSteps=40, progress=False, quiet=True,
                 params=dict(viscositySwitch=switch))
    g = gresho.state.state.alphas.detach().cpu()
    assert g.mean() < 0.35, f'{switch}: limiter not suppressing viscosity in a shear vortex (mean {g.mean():.2f})'
