"""AV_PLAN Phase 2: the Rosswog (2020) entropy trigger (`modules/shockCapturing/Rosswog2020.py`)."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from warpSPH.enumTypes import ViscositySwitch
from warpSPH.modules.shockCapturing.Rosswog2020 import (
    decayAlpha, desiredAlpha, entropyRate, smoothStep)
from warpSPH.modules.shockCapturing.wrapper import advanceViscositySwitchStep
from warpSPH.runner import run

EPS0, EPS1 = 1e-4, 5e-2


def _cfg(alpha_min=0.0, alpha_max=1.0):
    return SimpleNamespace(viscositySwitchParams=SimpleNamespace(
        scheme=ViscositySwitch.Rosswog2020, alpha_min=alpha_min, alpha_max=alpha_max,
        entropy_eps0=EPS0, entropy_eps1=EPS1, entropy_decay=30.0))


# --- Eqs. (18)-(20) -------------------------------------------------------------------------

def test_smoothstep_endpoints_slopes_and_monotone():
    x = torch.linspace(0, 1, 1001, dtype=torch.float64)
    s = smoothStep(x)
    assert s[0].item() == 0.0 and s[-1].item() == 1.0
    assert (s[1:] > s[:-1]).all()
    xg = torch.tensor([0.0, 1.0], dtype=torch.float64, requires_grad=True)
    (g,) = torch.autograd.grad(smoothStep(xg).sum(), xg)
    assert torch.allclose(g, torch.zeros(2, dtype=torch.float64))
    # clamped outside [0, 1]
    assert smoothStep(torch.tensor([-3.0, 7.0])).tolist() == [0.0, 1.0]


def test_desired_alpha_thresholds_and_monotone():
    below = torch.tensor([0.0, 1e-8, EPS0])
    assert desiredAlpha(below, EPS0, EPS1, 1.0).tolist() == [0.0, 0.0, 0.0]
    above = torch.tensor([EPS1, 1.0, 1e6, float('inf')])
    assert torch.allclose(desiredAlpha(above, EPS0, EPS1, 0.8), torch.full((4,), 0.8))
    sweep = torch.logspace(math.log10(EPS0), math.log10(EPS1), 200, dtype=torch.float64)[1:-1]
    a = desiredAlpha(sweep, EPS0, EPS1, 1.0)
    assert (a[1:] > a[:-1]).all() and 0 < a.min() and a.max() < 1
    # the midpoint in log space is S(1/2) = 1/2
    mid = torch.tensor([math.sqrt(EPS0 * EPS1)], dtype=torch.float64)
    assert desiredAlpha(mid, EPS0, EPS1, 1.0).item() == pytest.approx(0.5)


# --- Eq. (16) -------------------------------------------------------------------------------

def test_entropy_rate_is_dimensionless():
    """Scaling dt and tau together leaves epsdot unchanged -- the reason Eq. (16) carries tau/dt."""
    s, sPrev = torch.tensor([1.02, 0.5, 3.0]), torch.tensor([1.0, 0.49, 3.3])
    tau = torch.tensor([0.1, 0.2, 0.05])
    for k in (1e-3, 1.0, 1e3):
        assert torch.allclose(entropyRate(s, sPrev, k * tau, k * 0.01), entropyRate(s, sPrev, tau, 0.01))
    assert entropyRate(s, sPrev, tau, 0.01)[0].item() == pytest.approx(0.02 * 0.1 / 0.01)


# --- Eq. (17) -------------------------------------------------------------------------------

def test_decay_halves_in_30_tau_ln2_and_is_step_independent():
    tau = torch.tensor([0.3])
    t_half = 30 * 0.3 * math.log(2)
    for n in (1, 10, 1000):
        a = torch.tensor([1.0])
        for _ in range(n):
            a = decayAlpha(a, 0.0, tau, t_half / n, 30.0)
        assert a.item() == pytest.approx(0.5, rel=1e-5)
    # to 1 %
    a = decayAlpha(torch.tensor([1.0]), 0.0, tau, 30 * 0.3 * math.log(100), 30.0)
    assert a.item() == pytest.approx(0.01, rel=1e-5)
    # towards a non-zero floor
    assert decayAlpha(torch.tensor([1.0]), 0.2, tau, 1e6, 30.0).item() == pytest.approx(0.2)


# --- the step-boundary hook -----------------------------------------------------------------

def _stage(s, supports=0.02, c=1.0):
    n = len(s)
    return SimpleNamespace(entropies=torch.tensor(s, dtype=torch.float64),
                           supports=torch.full((n,), supports, dtype=torch.float64),
                           soundspeeds=torch.full((n,), c, dtype=torch.float64))


def _step(state, t, stages, cfg):
    system = SimpleNamespace(state=state)
    advanceViscositySwitchStep(system, SimpleNamespace(t=t), [(None, st) for st in stages], cfg)


def _fresh(n):
    return SimpleNamespace(alpha0s=torch.ones(n, dtype=torch.float64), alphas=torch.ones(n, dtype=torch.float64),
                           entropiesPrev=None, entropiesPrevTime=None, entropyRates=None)


def test_step_boundary_reads_stage0_only():
    """s constant at every step boundary but varying across the RK stages must give epsdot = 0:
    the trigger differences step-boundary entropies (Eq. 16), never stage values."""
    cfg = _cfg()
    state = _fresh(3)
    dt = 1e-3
    for n in range(20):
        noisy = _stage([1.0 + 0.3 * (n % 2), 2.0, 5.0])   # a later stage: noise
        _step(state, n * dt, [_stage([1.0, 2.0, 5.0]), noisy], cfg)
        assert (state.entropyRates == 0).all()
        assert (state.alpha0s == 0).all()


def test_first_step_sets_alpha_floor_then_decays_and_raises_instantly():
    cfg = _cfg(alpha_min=0.05, alpha_max=1.0)
    state = _fresh(2)
    tau = 0.5 * 0.02 / 1.0
    dt = 0.2 * tau
    _step(state, 0.0, [_stage([1.0, 1.0])], cfg)
    assert torch.allclose(state.alpha0s, torch.full((2,), 0.05, dtype=torch.float64))
    # particle 0: a 10 % entropy jump in one step -> epsdot = 0.1 * tau / dt = 0.5 -> alpha_max
    _step(state, dt, [_stage([1.1, 1.0])], cfg)
    assert state.alpha0s[0].item() == pytest.approx(1.0)
    assert state.alpha0s[1].item() == pytest.approx(0.05)
    assert state.entropyRates[0].item() == pytest.approx(0.1 * tau / dt)
    assert torch.equal(state.alphas, state.alpha0s)
    # entropy stops changing: alpha decays with 30 tau from 1 towards the floor, not instantly
    _step(state, 2 * dt, [_stage([1.1, 1.0])], cfg)
    expected = 0.05 + 0.95 * math.exp(-dt / (30 * tau))
    assert state.alpha0s[0].item() == pytest.approx(expected)


def test_hook_is_a_noop_for_other_switches():
    cfg = _cfg()
    cfg.viscositySwitchParams.scheme = ViscositySwitch.CullenDehnen2010
    state = _fresh(2)
    _step(state, 0.0, [_stage([1.0, 1.0])], cfg)
    assert state.entropiesPrev is None and (state.alpha0s == 1).all()


# --- wiring through the schemes ------------------------------------------------------------------

@pytest.mark.parametrize('scheme', ['Monaghan', 'CompSPH'])
def test_rosswog2020_engages_at_the_sod_shock_and_stays_off_in_smooth_flow(scheme):
    """nx = 400 (~250 particles per half): at nx = 100 this host's entropy error through the
    rarefaction is ~1 %, epsdot ~ 3e-3 > eps_0, and the trigger fires there (AV_PLAN Phase 2 notes);
    the paper's thresholds were set on MAGMA2. Regions are by |x| at t = 0.15 (mirrored Sod)."""
    from warpSPH.cases.sod import sodCase
    res = run(sodCase, scheme=scheme, nx=400, progress=False, quiet=True,
              params=dict(right_rho=0.125, right_pressure=0.1, viscositySwitch='Rosswog2020',
                          alpha_min=0.0, alpha_max=1.0))
    assert not res.diverged
    st = res.state.state
    a = st.alphas.detach().cpu()
    ax = st.positions[:, 0].detach().cpu().abs()
    assert st.entropiesPrev is not None and st.entropyRates is not None
    assert torch.equal(st.alphas, st.alpha0s)
    assert a.min() >= 0.0 and a.max() <= 1.0 + 1e-6
    assert a.max() > 0.9, f'never engaged: max alpha {a.max():.3f}'
    shocked = (ax > 0.66) & (ax < 0.78)
    assert a[shocked].mean() > 0.3, f'no dissipation trail behind the shock: {a[shocked].mean():.3f}'
    for lo, hi in ((0.0, 0.2), (0.3, 0.5), (0.9, 1.0)):   # undisturbed left, rarefaction, undisturbed right
        m = (ax >= lo) & (ax < hi)
        assert a[m].max() < 0.1, f'alpha {a[m].max():.3f} in the smooth region |x| in [{lo}, {hi})'
