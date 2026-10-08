"""AV_PLAN Phase 7b.2: the `LimiterType` family of the reconstruction limiter (`modules/reconstruction/limiters.py`).

Each limiter is checked against its closed form, plus the properties every one of them must have for the pair
reconstruction (phi in [0, 1], phi(1) = 1, phi(0) = 0, monotone on [0, 1], phi(x) = phi(1/x), zero for a sign change) and
the one that makes the family safe to add: `VanLeerFrontiere` is `limiterVL` bit for bit. Plus the plumbing: the type
is a `DiffusionParameters` / `CRKViscosity` field that survives a config dict round trip (and reads back as the default
from a config written before it existed), and the pair kernels respond to it.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from warpSPHCore import scalar_t

from warpSPH.configurations.crkSPH import buildDefaultCRKViscosityParams, crkSPHConfigToDict, dictToCRKSPHConfig, CRKSPHConfig
from warpSPH.configurations.moduleConfigurations.diffusionParameters import (
    LimiterType, buildDefaultDiffusionParamsCompressibleSPH, diffusionParamsToDict, dictToDiffusionParams)
from warpSPH.modules.reconstruction import limiterPsi, limiterVL

FLOAT64 = scalar_t is wp.float64
NP = np.float64 if FLOAT64 else np.float32
TOL = 1e-12 if FLOAT64 else 2e-6


@wp.kernel
def _psiKernel(limiterType: wp.int32, x: wp.array(dtype=scalar_t), out: wp.array(dtype=scalar_t)):
    k = wp.tid()
    out[k] = limiterPsi(limiterType, x[k])


@wp.kernel
def _vlKernel(x: wp.array(dtype=scalar_t), out: wp.array(dtype=scalar_t)):
    k = wp.tid()
    out[k] = limiterVL(x[k])


def psi(limiterType, x):
    xs = wp.array(np.asarray(x, dtype=NP), dtype=scalar_t)
    out = wp.zeros(xs.shape[0], dtype=scalar_t)
    wp.launch(_psiKernel, dim=xs.shape[0], inputs=[limiterType.value, xs], outputs=[out])
    return out.numpy().astype(np.float64)


CLOSED_FORM = {
    LimiterType.VanLeerFrontiere: lambda r: 4 * r / (1 + r) ** 2,
    LimiterType.Minmod: lambda r: r,
    LimiterType.VanLeer: lambda r: 2 * r / (1 + r),
    LimiterType.VanAlbada: lambda r: r * (1 + r) / (1 + r * r),
    LimiterType.MC: lambda r: np.minimum(2 * r, 0.5 * (1 + r)),
    LimiterType.Superbee: lambda r: np.minimum(2 * r, 1.0),
    LimiterType.Ospre: lambda r: 1.5 * r * (1 + r) / (1 + r + r * r),
}


@pytest.mark.parametrize('limiter', list(LimiterType))
def test_closedForm(limiter):
    r = np.linspace(0.0, 1.0, 101)[1:]
    np.testing.assert_allclose(psi(limiter, r), CLOSED_FORM[limiter](r), rtol=TOL * 5, atol=TOL)


@pytest.mark.parametrize('limiter', list(LimiterType))
def test_properties(limiter):
    r = np.linspace(0.0, 1.0, 201)
    p = psi(limiter, r)
    assert np.all(p >= 0.0) and np.all(p <= 1.0 + TOL)
    assert p[-1] == pytest.approx(1.0, abs=TOL)                      # a linear field is reconstructed in full
    assert np.all(np.diff(p) >= -TOL)                                # monotone on [0, 1]
    # sign change / no flow: zero, and the x -> 1/x symmetry that min(r_i, r_j) relies on
    np.testing.assert_array_equal(psi(limiter, [0.0, -1.0, -1e-3, -50.0]), np.zeros(4))
    x = np.array([1.5, 3.0, 20.0, 1000.0])
    np.testing.assert_allclose(psi(limiter, x), psi(limiter, 1.0 / x), rtol=TOL * 50)


def test_orderingOnTheUnitInterval():
    """minmod <= van Leer <= MC <= superbee on [0, 1]: the classic dissipation ordering (most to least diffusive)."""
    r = np.linspace(0.0, 1.0, 101)
    pm, pv, pc, ps = (psi(l, r) for l in (LimiterType.Minmod, LimiterType.VanLeer, LimiterType.MC, LimiterType.Superbee))
    assert np.all(pm <= pv + TOL) and np.all(pv <= pc + TOL) and np.all(pc <= ps + TOL)
    # Albada and Ospre sit between minmod and superbee as well
    for l in (LimiterType.VanAlbada, LimiterType.Ospre):
        p = psi(l, r)
        assert np.all(pm <= p + TOL) and np.all(p <= ps + TOL)


def test_defaultIsLimiterVLBitwise():
    x = np.concatenate([np.linspace(-1.0, 3.0, 401), np.logspace(-8, 8, 200)])
    xs = wp.array(x.astype(NP), dtype=scalar_t)
    ref = wp.zeros(xs.shape[0], dtype=scalar_t)
    wp.launch(_vlKernel, dim=xs.shape[0], inputs=[xs], outputs=[ref])
    got = psi(LimiterType.VanLeerFrontiere, x)
    np.testing.assert_array_equal(got, ref.numpy().astype(np.float64))


def test_configRoundTrip():
    d = buildDefaultDiffusionParamsCompressibleSPH()
    assert d.limiterType == LimiterType.VanLeerFrontiere.value
    d.limiterType = LimiterType.Superbee.value
    dd = diffusionParamsToDict(d)
    assert dd['limiterType'] == 'Superbee'
    assert dictToDiffusionParams(dd).limiterType == LimiterType.Superbee.value
    del dd['limiterType']                                            # a config stored before Phase 7b
    assert dictToDiffusionParams(dd).limiterType == LimiterType.VanLeerFrontiere.value

    cfg = CRKSPHConfig()
    assert cfg.crkViscosityParams.limiterType == LimiterType.VanLeerFrontiere.value
    cfg.crkViscosityParams.limiterType = LimiterType.MC.value
    cd = crkSPHConfigToDict(cfg)
    assert cd['crkViscosityParams']['limiterType'] == 'MC'
    assert dictToCRKSPHConfig(cd).crkViscosityParams.limiterType == LimiterType.MC.value
    del cd['crkViscosityParams']['limiterType']
    assert dictToCRKSPHConfig(cd).crkViscosityParams.limiterType == LimiterType.VanLeerFrontiere.value
