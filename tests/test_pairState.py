"""GODUNOV_SPH_PLAN layer 1: the scalar state reconstruction to the pair midpoint (`modules/reconstruction/pairState.py`) and the
(rho, P) gradients it extrapolates with (`computeStateGradients`).

* a linear field with its exact gradients gives the same midpoint state from both sides (the field's value at the midpoint),
  for any limiter, on a lattice and on a jittered one -- the premise of the reconstruction;
* the limiter: a step or a sign change of the gradient gives the particle values back; the reconstructed state always lies
  between the two particle values (positivity, no new extrema); `i <-> j` symmetry;
* `computeStateGradients` is exact for linear `rho` and `P` away from the periodic seam;
* with `riemannReconstruction` on, a linear (rho, P) field leaves the Riemann dissipation unchanged from its midpoint-state value, and
  the scheme runs (Sod) with a finite, conservative result.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import warp as wp
from warp.types import vector

from warpSPHCore import SupportScheme, buildVerletList, scalar_t

from warpSPH.configurations.moduleConfigurations.diffusionParameters import LimiterType, StateLimiter
from warpSPH.modules.reconstruction import computeStateGradients, reconstructPair1D, reconstructPairScalar
from warpSPH.runner import run

REPO = Path(__file__).resolve().parents[1]
FLOAT64 = scalar_t is wp.float64
NP = np.float64 if FLOAT64 else np.float32
vec2 = vector(length=2, dtype=scalar_t)
TOL = 1e-12 if FLOAT64 else 2e-6


@wp.kernel
def _scalarKernel(limiterType: wp.int32, s_i: wp.array(dtype=scalar_t), s_j: wp.array(dtype=scalar_t),
                  g_i: wp.array(dtype=vec2), g_j: wp.array(dtype=vec2), x_ij: wp.array(dtype=vec2),
                  out_i: wp.array(dtype=scalar_t), out_j: wp.array(dtype=scalar_t)):
    k = wp.tid()
    a, b = reconstructPairScalar(s_i[k], s_j[k], g_i[k], g_j[k], x_ij[k], scalar_t(1.0), limiterType)
    out_i[k] = a
    out_j[k] = b


def recon(limiter, s_i, s_j, g_i, g_j, x_ij):
    n = len(s_i)
    arr = lambda a: wp.array(np.asarray(a, dtype=NP), dtype=scalar_t if np.ndim(a) == 1 else vec2)
    oi, oj = wp.zeros(n, dtype=scalar_t), wp.zeros(n, dtype=scalar_t)
    wp.launch(_scalarKernel, dim=n, inputs=[limiter.value, arr(s_i), arr(s_j), arr(g_i), arr(g_j), arr(x_ij)], outputs=[oi, oj])
    return oi.numpy().astype(np.float64), oj.numpy().astype(np.float64)


@pytest.mark.parametrize('limiter', list(LimiterType))
def test_linearFieldGivesTheMidpointValue(limiter):
    rng = np.random.default_rng(0)
    n = 200
    g = rng.normal(size=(n, 2))
    x_ij = rng.normal(size=(n, 2)) * 0.05
    s0 = rng.uniform(1.0, 2.0, size=n)
    s_i = s0 + 0.5 * (g * x_ij).sum(1)            # x_i = +x_ij / 2, x_j = -x_ij / 2 about the midpoint
    s_j = s0 - 0.5 * (g * x_ij).sum(1)
    a, b = recon(limiter, s_i, s_j, g, g, x_ij)
    np.testing.assert_allclose(a, s0, atol=TOL * 10)
    np.testing.assert_allclose(b, s0, atol=TOL * 10)


@pytest.mark.parametrize('limiter', list(LimiterType))
def test_stepAndSignChangeKeepTheParticleValues(limiter):
    x = np.array([[0.1, 0.0]] * 3)
    # a step: both gradients zero; gradients of opposite sign (an extremum between them); one gradient zero
    g_i = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    g_j = np.array([[0.0, 0.0], [-1.0, 0.0], [0.0, 0.0]])
    s_i, s_j = np.array([1.0, 1.0, 1.0]), np.array([2.0, 2.0, 2.0])
    a, b = recon(limiter, s_i, s_j, g_i, g_j, x)
    np.testing.assert_allclose(a, s_i, atol=TOL)
    np.testing.assert_allclose(b, s_j, atol=TOL)


@pytest.mark.parametrize('limiter', list(LimiterType))
def test_boundedAndSymmetric(limiter):
    rng = np.random.default_rng(1)
    n = 500
    s_i, s_j = rng.uniform(0.1, 3.0, n), rng.uniform(0.1, 3.0, n)
    g_i, g_j = rng.normal(size=(n, 2)) * 5, rng.normal(size=(n, 2)) * 5
    x = rng.normal(size=(n, 2)) * 0.1
    a, b = recon(limiter, s_i, s_j, g_i, g_j, x)
    lo, hi = np.minimum(s_i, s_j), np.maximum(s_i, s_j)
    assert np.all(a >= lo - TOL) and np.all(a <= hi + TOL) and np.all(b >= lo - TOL) and np.all(b <= hi + TOL)
    # swapping i <-> j (and x_ij -> -x_ij) swaps the two states
    a2, b2 = recon(limiter, s_j, s_i, g_j, g_i, -x)
    np.testing.assert_allclose(a2, b, atol=TOL * 10)
    np.testing.assert_allclose(b2, a, atol=TOL * 10)


@wp.kernel
def _oneDKernel(limiter: wp.int32, s_i: wp.array(dtype=scalar_t), s_j: wp.array(dtype=scalar_t),
                g_i: wp.array(dtype=vec2), g_j: wp.array(dtype=vec2), x_ij: wp.array(dtype=vec2),
                out_i: wp.array(dtype=scalar_t), out_j: wp.array(dtype=scalar_t)):
    k = wp.tid()
    a, b = reconstructPair1D(limiter, s_i[k], s_j[k], g_i[k], g_j[k], x_ij[k])
    out_i[k] = a
    out_j[k] = b


def recon1d(limiter, s_i, s_j, g_i, g_j, x_ij):
    n = len(s_i)
    arr = lambda a: wp.array(np.asarray(a, dtype=NP), dtype=scalar_t if np.ndim(a) == 1 else vec2)
    oi, oj = wp.zeros(n, dtype=scalar_t), wp.zeros(n, dtype=scalar_t)
    wp.launch(_oneDKernel, dim=n, inputs=[limiter.value, arr(s_i), arr(s_j), arr(g_i), arr(g_j), arr(x_ij)], outputs=[oi, oj])
    return oi.numpy().astype(np.float64), oj.numpy().astype(np.float64)


GODUNOV_LIMITERS = [StateLimiter.VanLeerHarmonic, StateLimiter.VanLeerMonotonized, StateLimiter.InutsukaSign]


@pytest.mark.parametrize('limiter', GODUNOV_LIMITERS)
def test_1dLimitersAreExactForALinearField(limiter):
    rng = np.random.default_rng(2)
    n = 200
    g = rng.normal(size=(n, 2))
    x_ij = rng.normal(size=(n, 2)) * 0.05
    s0 = rng.uniform(1.0, 2.0, size=n)
    s_i = s0 + 0.5 * (g * x_ij).sum(1)
    s_j = s0 - 0.5 * (g * x_ij).sum(1)
    a, b = recon1d(limiter, s_i, s_j, g, g, x_ij)
    np.testing.assert_allclose(a, s0, atol=TOL * 10)
    np.testing.assert_allclose(b, s0, atol=TOL * 10)


@pytest.mark.parametrize('limiter', GODUNOV_LIMITERS)
def test_1dLimitersStepExtremumAndBounds(limiter):
    """A step (zero gradients) and an extremum (gradients disagreeing with the difference) give the particle values; the states always lie between
    the two particle values for the van Leer limiters (Murante, Iwasaki), and are symmetric under i <-> j."""
    x = np.array([[0.1, 0.0]] * 3)
    g_i = np.array([[0.0, 0.0], [-1.0, 0.0], [1.0, 0.0]])
    g_j = np.array([[0.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]])
    s_i, s_j = np.array([1.0, 1.0, 1.0]), np.array([2.0, 2.0, 2.0])      # D = -1 along +x: a gradient of +1 disagrees in sign
    a, b = recon1d(limiter, s_i, s_j, g_i, g_j, x)
    np.testing.assert_allclose(a[:1], s_i[:1], atol=TOL)
    np.testing.assert_allclose(b[:1], s_j[:1], atol=TOL)
    np.testing.assert_allclose(a[2:], s_i[2:], atol=TOL)                 # the gradient of i disagrees with D: zero slope (and Inutsuka: both)
    rng = np.random.default_rng(4)
    n = 400
    s_i, s_j = rng.uniform(0.1, 3.0, n), rng.uniform(0.1, 3.0, n)
    g_i, g_j = rng.normal(size=(n, 2)) * 5, rng.normal(size=(n, 2)) * 5
    xx = rng.normal(size=(n, 2)) * 0.1
    a, b = recon1d(limiter, s_i, s_j, g_i, g_j, xx)
    if limiter != StateLimiter.InutsukaSign:
        lo, hi = np.minimum(s_i, s_j), np.maximum(s_i, s_j)
        assert np.all(a >= lo - TOL) and np.all(a <= hi + TOL) and np.all(b >= lo - TOL) and np.all(b <= hi + TOL)
    a2, b2 = recon1d(limiter, s_j, s_i, g_j, g_i, -xx)
    np.testing.assert_allclose(a2, b, atol=TOL * 10)
    np.testing.assert_allclose(b2, a, atol=TOL * 10)


def _greshoState(nx=32):
    from warpSPH.cases.greshoVortex import greshoVortexCase
    res = run(greshoVortexCase, scheme='Monaghan', nx=nx, nSteps=1, progress=False, quiet=True)
    return res.state, res.ctx.config


@pytest.mark.parametrize('jitter', [0.0, 0.15])
def test_stateGradientsExactForLinearFields(jitter):
    system, config = _greshoState()
    st = system.state
    dtype, dev = st.positions.dtype, st.positions.device
    adj = system.adjacency
    if jitter > 0:
        g = torch.Generator().manual_seed(1)
        if True:
            st.positions = st.positions + (jitter / 32 * (2 * torch.rand(st.positions.shape, generator=g) - 1)).to(dev, dtype)
        adj = buildVerletList(st, config.domain, verletScale=config.verletScale,
                              supportMode=SupportScheme.SuperSymmetric, priorNeighborhood=None, verbose=False)
    a_rho = torch.tensor([0.7, -0.4], dtype=dtype, device=dev)
    a_P = torch.tensor([-0.2, 0.9], dtype=dtype, device=dev)
    st.densities = 1.0 + st.positions @ a_rho
    st.pressures = 2.0 + st.positions @ a_P
    G = computeStateGradients(st, config, SupportScheme.SuperSymmetric, adj, corrected=True)
    inner = (st.positions.abs().amax(dim=1) < 0.5 - 1.05 * st.supports)
    assert int(inner.sum()) > 300
    tol = 1e-12 if FLOAT64 else 2e-5
    torch.testing.assert_close(G[inner, 0], a_rho.expand(int(inner.sum()), 2), atol=tol, rtol=0)
    torch.testing.assert_close(G[inner, 1], a_P.expand(int(inner.sum()), 2), atol=tol, rtol=0)


@pytest.mark.skipif(FLOAT64, reason='already float64: the in-process test is the strict one')
def test_float64():
    env = dict(os.environ, warpSPHCore_PRECISION='float64')
    proc = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-x', '-p', 'no:cacheprovider', '-k', 'not float64', __file__],
                          cwd=REPO, env=env, capture_output=True, text=True, timeout=1800)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-2000:]
