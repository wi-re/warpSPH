"""The CRK viscosity limiter's (eta_crit, eta_fold) default to (1/n_h, 0.2/n_h) in r/H (one and 0.2 nominal
particle spacings; Frontiere 2017 Eqs. 51-53, Spheral etaCritFrac/nPerh), resolved per step from `n_h` by
`resolveCRKLimiter`; an explicit positive value overrides (CRKSPH_LIMITER_PLAN O2)."""

import pytest

from warpSPH.configurations.crkSPH import CRKViscosity, buildDefaultCRKViscosityParams, resolveCRKLimiter


@pytest.mark.parametrize('n_h', [3.5, 4.0, 4.53, 6.0])
def test_default_derived_from_n_h(n_h):
    p = resolveCRKLimiter(buildDefaultCRKViscosityParams(), n_h)
    assert p.eta_crit == pytest.approx(1 / n_h, rel=1e-6)
    assert p.eta_fold == pytest.approx(0.2 / n_h, rel=1e-6)
    assert p.enableCRKLimiter and p.enableVanLeerLimiter


def test_explicit_values_win():
    p = buildDefaultCRKViscosityParams()
    p.eta_crit, p.eta_fold = 1 / 3, 0.2
    assert resolveCRKLimiter(p, 4.0) is p
    p.eta_fold = -1.0                       # only one explicit: the other is still derived
    q = resolveCRKLimiter(p, 4.0)
    assert q.eta_crit == pytest.approx(1 / 3, rel=1e-6) and q.eta_fold == pytest.approx(0.05, rel=1e-6)
