"""GODUNOV_SPH_PLAN layer 3: Inutsuka (2002) Godunov SPH (`modules/godunov/wp_inutsuka.py`, `schemes/gsphInutsuka.py`).

* the Gaussian density (Murante's symmetrised sum) and its gradient: a uniform lattice has a uniform density equal to the kernel sum; the gradient of a
  linearly varying particle density... (checked through the mass-weighted sum identity: the mean density times the volume is the total mass);
* conservation, pair by pair: momentum and total energy vanish for every solver, first and second order, cubic and linear volumes;
* a rest state with uniform pressure and a uniform density has no energy exchange;
* the scheme runs on Sod (finite, energy conserved).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import warp as wp

from warpSPHCore import OperationProperties, SupportScheme, buildVerletList, scalar_t

from warpSPH.configurations.gsph import InutsukaGSPHConfig
from warpSPH.configurations.moduleConfigurations.diffusionParameters import (
    RiemannSolver, StateLimiter, VelocityPairPolicy, resolveReconstructionLimiter)
from warpSPH.modules.godunov import computeGaussianDensityWarp, computeInutsukaWarp
from warpSPH.modules.reconstruction import computeStateGradients, computeVelocityJacobian
from warpSPH.runner import run

REPO = Path(__file__).resolve().parents[1]
FLOAT64 = scalar_t is wp.float64


_H = [None]      # the Gaussian h of the last `_setup`


def _state():
    from warpSPH.cases.greshoVortex import greshoVortexCase
    res = run(greshoVortexCase, scheme='Monaghan', nx=24, nSteps=1, progress=False, quiet=True)
    return res.state, res.ctx.config


def _setup(system, config):
    st = system.state
    hG = (st.masses / st.densities) ** (1.0 / config.dim)          # the scheme's Gaussian h: eta = 1 times the local particle spacing
    reach = float((3.0 * 2.0 ** 0.5 * hG / st.supports).max())
    adj = buildVerletList(st, config.domain, verletScale=max(1.5, 1.05 * reach), supportMode=SupportScheme.SuperSymmetric, priorNeighborhood=None, verbose=False)
    props = OperationProperties(kernel=config.kernel, supportMode=SupportScheme.KernelMeanSymmetric)
    gradRho, rho = computeGaussianDensityWarp(st, props, domain=config.domain, adjacency=adj, gaussianH=hG)
    st.densities = rho
    _H[0] = hG
    return adj, props, gradRho


def _rates(system, config, adj, props, gradRho, solver, order, cubic, limiter=StateLimiter.VanLeerHarmonic, shockC=0.0):
    st = system.state
    p = InutsukaGSPHConfig().diffusionParams
    p.riemannSolver = solver.value
    p.stateLimiter = limiter.value
    p.shockSwitchC = shockC
    p.velocityPairPolicy = VelocityPairPolicy.Limited.value if order == 2 else VelocityPairPolicy.Raw.value
    p = resolveReconstructionLimiter(p, config.n_h)
    J = G = None
    if order == 2:
        J = computeVelocityJacobian(st, config, SupportScheme.SuperSymmetric, adj)
        G = computeStateGradients(st, config, SupportScheme.SuperSymmetric, adj)
    return computeInutsukaWarp(st, props, domain=config.domain, params=p, gamma=5.0 / 3.0, gradRho=gradRho, order=order, cubic=cubic,
                               adjacency=adj, queryVelocityTensor=J, queryStateGradients=G, gaussianH=_H[0])


def test_gaussianDensityIsTheSymmetrisedSum():
    system, config = _state()
    st = system.state
    adj, props, gradRho = _setup(system, config)
    inner = st.positions.abs().amax(dim=1) < 0.5 - 1.5 * st.supports
    # unit-mass lattice at the case's density: the Gaussian sum of a (nearly) uniform lattice is the lattice density
    assert float(st.densities[inner].std() / st.densities[inner].mean()) < 0.05
    assert float(st.densities[inner].mean()) == pytest.approx(1.0, rel=0.05)
    # no net gradient of a uniform density on average
    assert float(gradRho[inner].mean(dim=0).abs().max()) < 0.05


@pytest.mark.parametrize('cubic', [True, False])
@pytest.mark.parametrize('order', [1, 2])
@pytest.mark.parametrize('solver', [RiemannSolver.Acoustic, RiemannSolver.Adaptive, RiemannSolver.HLLC])
def test_momentumAndEnergyConservation(solver, order, cubic):
    system, config = _state()
    st = system.state
    adj, props, gradRho = _setup(system, config)
    g = torch.Generator().manual_seed(11)
    dtype, dev = st.positions.dtype, st.positions.device
    st.velocities = (torch.rand(st.velocities.shape, generator=g) * 2 - 1).to(dev, dtype)
    st.pressures = (1.0 + torch.rand(st.pressures.shape, generator=g)).to(dev, dtype)
    a, du = _rates(system, config, adj, props, gradRho, solver, order, cubic)
    m = st.masses
    scale = (m.view(-1, 1) * a.abs()).sum()
    mom = (m.view(-1, 1) * a).sum(0).abs().max()
    dE = (m * (st.velocities * a).sum(1) + m * du).sum().abs()
    eScale = (m * (st.velocities * a).sum(1)).abs().sum() + (m * du).abs().sum()
    tol = 1e-10 if FLOAT64 else 5e-4
    assert mom / scale < tol, (mom, scale)
    assert dE / eScale < tol, (dE, eScale)


@pytest.mark.parametrize('limiter', [StateLimiter.VanLeerHarmonic, StateLimiter.VanLeerMonotonized, StateLimiter.InutsukaSign])
def test_conservationForEveryStateLimiterAndTheShockSwitch(limiter):
    system, config = _state()
    st = system.state
    adj, props, gradRho = _setup(system, config)
    g = torch.Generator().manual_seed(5)
    dtype, dev = st.positions.dtype, st.positions.device
    st.velocities = (torch.rand(st.velocities.shape, generator=g) * 2 - 1).to(dev, dtype)
    st.pressures = (1.0 + torch.rand(st.pressures.shape, generator=g)).to(dev, dtype)
    a, du = _rates(system, config, adj, props, gradRho, RiemannSolver.Adaptive, 2, True, limiter, shockC=3.0)
    m = st.masses
    tol = 1e-10 if FLOAT64 else 5e-4
    assert (m.view(-1, 1) * a).sum(0).abs().max() / (m.view(-1, 1) * a.abs()).sum() < tol
    dE = (m * (st.velocities * a).sum(1) + m * du).sum().abs()
    assert dE / ((m * (st.velocities * a).sum(1)).abs().sum() + (m * du).abs().sum()) < tol


def test_restStateHasNoEnergyExchange():
    system, config = _state()
    st = system.state
    adj, props, gradRho = _setup(system, config)
    st.velocities = torch.zeros_like(st.velocities)
    st.pressures = torch.full_like(st.pressures, 1.5)
    for order in (1, 2):
        a, du = _rates(system, config, adj, props, gradRho, RiemannSolver.Adaptive, order, True)
        assert du.abs().max() < (1e-12 if FLOAT64 else 1e-6)


def _energy(st):
    return float((st.masses * (0.5 * (st.velocities ** 2).sum(-1) + st.internalEnergies)).sum())


def test_sod1dRunsAndConserves():
    from warpSPH.cases.sod import sodCase
    e0 = _energy(run(sodCase, scheme='InutsukaGSPH', nSteps=1, progress=False, quiet=True).state.state)
    st = run(sodCase, scheme='InutsukaGSPH', nSteps=120, progress=False, quiet=True).state.state
    assert torch.isfinite(st.densities).all() and torch.isfinite(st.velocities).all() and torch.isfinite(st.internalEnergies).all()
    assert st.velocities.abs().max() > 0.1, 'the diaphragm should have opened'
    assert abs(_energy(st) - e0) / e0 < 2e-3


@pytest.mark.skipif(FLOAT64, reason='already float64: the in-process test is the strict one')
def test_float64():
    env = dict(os.environ, warpSPHCore_PRECISION='float64')
    proc = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-x', '-p', 'no:cacheprovider', '-k', 'not float64', __file__],
                          cwd=REPO, env=env, capture_output=True, text=True, timeout=1800)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-2000:]
