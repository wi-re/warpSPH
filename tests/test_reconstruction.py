"""AV_PLAN Phase 3: the velocity reconstruction (`modules/reconstruction`).

* linear-field annihilation: the midpoint reconstruction of `v = A x + b` with the M^-1-corrected Jacobian gives a
  zero pair velocity, on a lattice and on a jittered lattice (the premise of every reconstruction AV);
* the limiter: phi -> 1 in a linear field, -> 0 across a velocity step or a sign change of the gradient; the
  close-pair taper is 1 beyond eta_crit and Gaussian inside;
* conservation: with a reconstructed pair velocity the Monaghan viscous force still conserves momentum, and its
  heating still balances the kinetic energy it removes (one factor reconstructed, one raw).

The plan's tolerances (1e-5 / 1e-2 annihilation, 1e-12 momentum) are float64 ones: `test_float64` reruns this file in
a float64 subprocess (the precision is fixed at import); in process the bounds are float32-sized.
"""

from __future__ import annotations

import dataclasses
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import warp as wp
from warp.types import vector, matrix

from warpSPHCore import SupportScheme, buildVerletList, scalar_t
from warpSPHCore.coreOperations._jvpCommon import buildDomainState
from warpSPHCore.dataTypes.domain_t import domainData
from warpSPHCore.util import castTorchToWarp, castTorchToWarpAsBuiltins

from warpSPH.configurations.moduleConfigurations.diffusionParameters import VelocityPairPolicy
from warpSPH.modules.reconstruction import (computeVelocityJacobian, crkLimiter, limitedPairPhi,
                                            reconstructPairVelocity)
from warpSPH.runner import run

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts'))
FLOAT64 = scalar_t is wp.float64
vec2 = vector(length=2, dtype=scalar_t)
mat2 = matrix(shape=(2, 2), dtype=scalar_t)


@wp.kernel
def _pairVelocityKernel(
    I: wp.array(dtype=wp.int64), J: wp.array(dtype=wp.int64),
    x: wp.array(dtype=vec2), v: wp.array(dtype=vec2), h: wp.array(dtype=scalar_t), grad: wp.array(dtype=mat2),
    domainState: domainData, policy: wp.int32, kernel_int: wp.int32, eta_crit: scalar_t, eta_fold: scalar_t,
    B: wp.array(dtype=scalar_t), balsaraPower: scalar_t,
    out: wp.array(dtype=vec2),
):
    k = wp.tid()
    i = wp.int32(I[k])
    j = wp.int32(J[k])
    x_ij = computeDistanceVec(x[i], x[j], domainState)
    out[k] = reconstructPairVelocity(policy, v[i], v[j], grad[i], grad[j], x_ij, h[i], h[j], kernel_int, 2,
                                     eta_crit, eta_fold, B[i], B[j], balsaraPower)


@wp.kernel
def _phiKernel(
    x_ij: wp.array(dtype=vec2), Ji: wp.array(dtype=mat2), Jj: wp.array(dtype=mat2), h: scalar_t,
    kernel_int: wp.int32, eta_crit: scalar_t, eta_fold: scalar_t,
    phi: wp.array(dtype=scalar_t), taper: wp.array(dtype=scalar_t),
):
    k = wp.tid()
    zero = vec2()
    phi[k] = limitedPairPhi(x_ij[k], h, h, zero, zero, Ji[k], Jj[k], kernel_int, 2, True, False, eta_crit, eta_fold)
    taper[k] = crkLimiter(x_ij[k], h, h, kernel_int, 2, eta_crit, eta_fold)


from warpSPHCore import computeDistanceVec   # noqa: E402  (the kernels above resolve it at codegen time)


def _greshoState(nx=32):
    from warpSPH.cases.greshoVortex import greshoVortexCase
    res = run(greshoVortexCase, scheme='Monaghan', nx=nx, nSteps=1, progress=False, quiet=True)
    return res.state, res.ctx.config


def _pairVelocities(state, adj, config, grad, policy, eta_crit=0.25, eta_fold=0.05, B=None, balsaraPower=2.0):
    dev = state.positions.device
    n = adj.i.shape[0]
    out = wp.zeros(n, dtype=vec2, device=str(dev))
    if B is None:
        B = torch.zeros_like(state.supports)
    wp.launch(_pairVelocityKernel, dim=n, device=str(dev), inputs=[
        castTorchToWarp(adj.i.to(torch.int64)), castTorchToWarp(adj.j.to(torch.int64)),
        castTorchToWarpAsBuiltins(state.positions), castTorchToWarpAsBuiltins(state.velocities),
        castTorchToWarp(state.supports), castTorchToWarpAsBuiltins(grad.contiguous()),
        buildDomainState(config.domain), policy.value, config.kernel.value, eta_crit, eta_fold,
        castTorchToWarp(B.contiguous()), balsaraPower, out])
    return wp.to_torch(out)


A = torch.tensor([[0.3, -0.7], [0.5, 0.2]])
B = torch.tensor([0.1, -0.2])


@pytest.mark.parametrize('jitter', [0.0, 0.15])
def test_linear_field_annihilation(jitter):
    system, config = _greshoState()
    st = system.state
    dtype, dev = st.positions.dtype, st.positions.device
    adj = system.adjacency
    if jitter > 0:
        dx = 1.0 / 32
        g = torch.Generator().manual_seed(1)
        st.positions = st.positions + (jitter * dx * (2 * torch.rand(st.positions.shape, generator=g) - 1)).to(dev, dtype)
        adj = buildVerletList(st, config.domain, verletScale=config.verletScale,
                              supportMode=SupportScheme.SuperSymmetric, priorNeighborhood=None, verbose=False)
    A_, B_ = A.to(dev, dtype), B.to(dev, dtype)
    st.velocities = st.positions @ A_.T + B_
    grad = computeVelocityJacobian(st, config, SupportScheme.SuperSymmetric, adj, corrected=True)

    # particles whose support does not reach the periodic seam see the linear field exactly
    h = st.supports
    inner = (st.positions.abs().amax(dim=1) < 0.5 - 1.05 * h)
    assert torch.allclose(grad[inner], A_.expand(int(inner.sum()), 2, 2), atol=1e-12 if FLOAT64 else 1e-5), \
        'J @ dx is not dv (or the M correction is not volume-consistent, OPEN_PROBLEMS §22)'

    pairs = inner[adj.i] & inner[adj.j] & (adj.i != adj.j)
    assert int(pairs.sum()) > 1000
    scale = torch.linalg.matrix_norm(A_, ord=2) * h.max()
    raw = (st.velocities[adj.i] - st.velocities[adj.j])[pairs]
    assert raw.norm(dim=1).max() / scale > 0.1, 'the raw pair velocity should not vanish, or the test proves nothing'
    tol = (1e-5 if jitter == 0 else 1e-2) if FLOAT64 else 1e-4
    # Limited: phi = 1 in a linear field, but the close-pair taper (eta_crit = 1/n_h is one lattice spacing) would
    # rightly damp the nearest pairs once the lattice has moved -- check it with eta_crit below the spacing
    for policy in ([VelocityPairPolicy.Linear, VelocityPairPolicy.Limited] if jitter == 0 else [VelocityPairPolicy.Linear]):
        u = _pairVelocities(st, adj, config, grad, policy, eta_crit=0.15)[pairs]
        worst = (u.norm(dim=1).max() / scale).item()
        assert worst < tol, f'{policy.name}: max|u_ij|/(|A| h) = {worst:.2e} >= {tol}'
    # the Raw policy is the identity
    u = _pairVelocities(st, adj, config, grad, VelocityPairPolicy.Raw)[pairs]
    assert torch.equal(u, raw)


def test_balsara_limited_endpoints():
    """`BalsaraLimited` (Garcia-Senz & Cabezon 2026 Eqs. 18-19): B = 0 (shear) is `Limited`, B = 1 (compression)
    switches the reconstruction off (`Raw`), B = 1/2 with p = 2 takes 3/4 of the correction."""
    system, config = _greshoState()
    st, adj = system.state, system.adjacency
    grad = computeVelocityJacobian(st, config, SupportScheme.SuperSymmetric, adj, corrected=True)
    limited = _pairVelocities(st, adj, config, grad, VelocityPairPolicy.Limited)
    raw = _pairVelocities(st, adj, config, grad, VelocityPairPolicy.Raw)
    B0 = _pairVelocities(st, adj, config, grad, VelocityPairPolicy.BalsaraLimited, B=torch.zeros_like(st.supports))
    B1 = _pairVelocities(st, adj, config, grad, VelocityPairPolicy.BalsaraLimited, B=torch.ones_like(st.supports))
    Bh = _pairVelocities(st, adj, config, grad, VelocityPairPolicy.BalsaraLimited, B=torch.full_like(st.supports, 0.5))
    assert torch.equal(B0, limited)
    assert torch.allclose(B1, raw, atol=1e-6 * raw.abs().max().item())
    assert torch.allclose(Bh, raw + 0.75 * (limited - raw), atol=1e-5 * raw.abs().max().item())
    assert (limited - raw).abs().max() > 1e-3 * raw.abs().max(), 'the reconstruction did nothing, the test proves nothing'


def test_balsara_from_jacobian():
    from warpSPH.modules.reconstruction import balsaraFromJacobian
    c, h = torch.ones(4), torch.full((4,), 0.1)
    J = torch.stack([-torch.eye(2),                                   # pure compression
                     torch.tensor([[0.0, -1.0], [1.0, 0.0]]),           # pure rotation
                     torch.tensor([[0.0, 1.0], [0.0, 0.0]]),            # simple shear: div 0
                     torch.zeros(2, 2)])                                # at rest
    B = balsaraFromJacobian(J, c, h)
    assert B[0] > 0.999 and B[1] == 0 and B[2] == 0 and B[3] == 0


def _phi(xij, Ji, Jj, h=1.0, eta_crit=0.25, eta_fold=0.05):
    dev = 'cuda' if wp.get_cuda_device_count() else 'cpu'
    dtype = torch.float64 if FLOAT64 else torch.float32
    t = lambda a: torch.as_tensor(a, dtype=dtype, device=dev)   # noqa: E731
    n = len(xij)
    phi = wp.zeros(n, dtype=scalar_t, device=dev)
    taper = wp.zeros(n, dtype=scalar_t, device=dev)
    from warpSPHCore import KernelFunctions
    wp.launch(_phiKernel, dim=n, device=dev, inputs=[
        castTorchToWarpAsBuiltins(t(xij)), castTorchToWarpAsBuiltins(t(Ji)), castTorchToWarpAsBuiltins(t(Jj)),
        h, KernelFunctions.Wendland4.value, eta_crit, eta_fold, phi, taper])
    return wp.to_torch(phi).cpu(), wp.to_torch(taper).cpu()


def test_limiter_behaviour():
    x = [[0.3, 0.1]] * 4
    smooth = [[0.4, -0.2], [0.3, 0.1]]
    Ji = [smooth, smooth, [[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 0.0]]]
    Jj = [smooth,                              # linear field: identical gradients
          [[0.4 * 1.05, -0.2], [0.3, 0.1]],    # smooth: 5 % gradient change
          [[-1.0, 0.0], [0.0, -1.0]],          # the gradient changes sign: an extremum / a shock
          [[0.0, 0.0], [0.0, 0.0]]]            # flat on one side: a velocity step's edge
    phi, _ = _phi(x, Ji, Jj)
    assert phi[0] == pytest.approx(1.0, abs=1e-6)
    assert phi[1] > 0.99
    assert phi[2] == 0.0
    assert phi[3] == 0.0
    # symmetric in the pair (phi_ij = phi_ji keeps u_ji = -u_ij)
    phiSwap, _ = _phi([[-0.3, -0.1]] * 4, Jj, Ji)
    assert torch.equal(phi, phiSwap)


def test_close_pair_taper():
    eta_crit, eta_fold = 0.25, 0.05
    r = torch.tensor([0.05, 0.15, 0.2, 0.249, 0.25, 0.3, 0.9])
    x = torch.stack([r, torch.zeros_like(r)], dim=1).tolist()
    I = [[[1.0, 0.0], [0.0, 1.0]]] * len(r)
    _, taper = _phi(x, I, I, h=1.0, eta_crit=eta_crit, eta_fold=eta_fold)
    expected = torch.where(r < eta_crit, torch.exp(-((r - eta_crit) / eta_fold) ** 2), torch.ones_like(r))
    assert torch.allclose(taper, expected.to(taper.dtype), atol=1e-6)
    assert (taper[r >= eta_crit] == 1).all()


def _conservation(policy: str, caseName: str):
    from probe_monaghanEnergy import pieces
    if caseName == 'sod':
        from warpSPH.cases.sod import sodCase as case
        kw = dict(nx=100, nSteps=120, params=dict(viscositySwitch='NoneSwitch', right_rho=0.125, right_pressure=0.1))
    else:
        from warpSPH.cases.greshoVortex import greshoVortexCase as case
        kw = dict(nx=32, nSteps=20, params=dict(viscositySwitch='NoneSwitch'))

    def configure(ctx, _orig=case.configureScheme):
        _orig(ctx)
        ctx.schemeConfig.diffusionParams.C_q = 2.0
        if policy == 'BalsaraPairLimiter':
            ctx.schemeConfig.diffusionParams.balsaraPairLimiter = True
        else:
            ctx.schemeConfig.diffusionParams.velocityPairPolicy = VelocityPairPolicy[policy].value

    res = run(dataclasses.replace(case, configureScheme=configure), scheme='Monaghan', progress=False, quiet=True, **kw)
    st, out = pieces(res.state, res.ctx.config, res.ctx.schemeConfig)
    m, v = st.masses, st.velocities
    dvdt, dudt = out['viscous force + heating']
    momentum = (m.view(-1, 1) * dvdt).sum(dim=0).norm().item() / (m.view(-1, 1) * dvdt).norm(dim=1).sum().item()
    kinetic = (m * torch.einsum('ij,ij->i', v, dvdt)).sum().item()
    heat = (m * dudt).sum().item()
    scale = (m * torch.einsum('ij,ij->i', v, dvdt)).abs().sum().item()
    return momentum, abs(kinetic + heat) / scale, abs(heat)


@pytest.mark.parametrize('caseName', ['sod', 'gresho'])
@pytest.mark.parametrize('policy', ['Raw', 'Linear', 'Limited', 'BalsaraLimited', 'BalsaraPairLimiter'])
def test_conservation_inprocess(policy, caseName):
    momentum, energy, heat = _conservation(policy, caseName)
    tol = 1e-12 if FLOAT64 else 1e-5
    assert heat > 1e-8, 'the viscous piece never engaged, so the test proves nothing'
    assert momentum < tol, f'momentum {momentum:.2e}'
    assert energy < (1e-10 if FLOAT64 else 1e-5), f'energy {energy:.2e}'


@pytest.mark.skipif(FLOAT64, reason='already float64: the in-process test is the strict one')
def test_float64():
    env = dict(os.environ, warpSPHCore_PRECISION='float64')
    proc = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-x', '-p', 'no:cacheprovider',
                           '-k', 'not float64', __file__],
                          cwd=REPO, env=env, capture_output=True, text=True, timeout=1800)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-2000:]


def test_requested_precision():
    """The float64 rerun must really be float64 (`tests/conftest.py` honours `warpSPHCore_PRECISION`)."""
    if os.environ.get('warpSPHCore_PRECISION') == 'float64':
        assert FLOAT64
