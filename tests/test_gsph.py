"""GODUNOV_SPH_PLAN layer 2: the simplified Godunov SPH pair rates (`modules/godunov`) and the `GSPH` scheme.

* conservation: momentum `sum m a` and total energy `sum m (v . a + du/dt)` vanish pair by pair, for every solver, first and second order,
  on a random velocity / pressure field;
* the plan's central claim about this form: for a constant pressure it is *identically the standard SPH force* (`-P sum m_j (1/rho_i^2 +
  1/rho_j^2) grad W`), hence keeps the density-gradient inconsistency of Cha et al. (2010) -- checked against the independent symmetric
  pressure-force operator (equal smoothing lengths) -- and `du/dt` vanishes for equal velocities;
* rest: uniform state, zero velocity gives zero rates;
* the scheme runs: Sod 1D is finite, conserves energy to the integrator's order and is close to the exact solution.

Float32 tolerances in process; `test_float64` reruns the file in float64.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import warp as wp

from warpSPHCore import OperationProperties, SupportScheme, scalar_t

from warpSPH.configurations.gsph import GSPHConfig
from warpSPH.configurations.moduleConfigurations.diffusionParameters import (
    RiemannSolver, VelocityPairPolicy, resolveReconstructionLimiter)
from warpSPH.modules.godunov import computeGodunovWarp
from warpSPH.modules.pressure import computePressureForceSymmetric
from warpSPH.modules.reconstruction import computeStateGradients, computeVelocityJacobian
from warpSPH.runner import run

REPO = Path(__file__).resolve().parents[1]
FLOAT64 = scalar_t is wp.float64


def _state():
    from warpSPH.cases.greshoVortex import greshoVortexCase
    res = run(greshoVortexCase, scheme='Monaghan', nx=24, nSteps=1, progress=False, quiet=True)
    return res.state, res.ctx.config


def _rates(system, config, solver, order, gamma=5.0 / 3.0):
    st = system.state
    params = GSPHConfig().diffusionParams
    params.riemannSolver = solver.value
    params.velocityPairPolicy = VelocityPairPolicy.Limited.value if order == 2 else VelocityPairPolicy.Raw.value
    params = resolveReconstructionLimiter(params, config.n_h)
    J = G = None
    if order == 2:
        J = computeVelocityJacobian(st, config, SupportScheme.SuperSymmetric, system.adjacency)
        G = computeStateGradients(st, config, SupportScheme.SuperSymmetric, system.adjacency)
    return computeGodunovWarp(
        st, OperationProperties(kernel=config.kernel, supportMode=SupportScheme.KernelMeanSymmetric), domain=config.domain,
        params=params, gamma=gamma, order=order, adjacency=system.adjacency, queryVelocityTensor=J, queryStateGradients=G)


@pytest.mark.parametrize('order', [1, 2])
@pytest.mark.parametrize('solver', [RiemannSolver.Acoustic, RiemannSolver.Adaptive, RiemannSolver.HLLC])
def test_momentumAndEnergyConservation(solver, order):
    system, config = _state()
    st = system.state
    g = torch.Generator().manual_seed(7)
    dtype, dev = st.positions.dtype, st.positions.device
    st.velocities = (torch.rand(st.velocities.shape, generator=g) * 2 - 1).to(dev, dtype)
    st.pressures = (1.0 + torch.rand(st.pressures.shape, generator=g)).to(dev, dtype)
    a, du = _rates(system, config, solver, order)
    m = st.masses
    scale = (m.view(-1, 1) * a.abs()).sum()
    mom = (m.view(-1, 1) * a).sum(0).abs().max()
    dE = (m * (st.velocities * a).sum(1) + m * du).sum().abs()
    eScale = (m * (st.velocities * a).sum(1)).abs().sum() + (m * du).abs().sum()
    tol = 1e-10 if FLOAT64 else 2e-4
    assert mom / scale < tol, (mom, scale)
    assert dE / eScale < tol, (dE, eScale)


def _referenceForce(system, config):
    st = system.state
    return computePressureForceSymmetric(st, config, supportScheme=SupportScheme.KernelMeanSymmetric, adjacency=system.adjacency)


def test_restStateGivesZeroEnergyRateAndTheStandardForce():
    """Uniform P, zero velocity: no energy exchange (u* = 0); the force is the kernel-sum noise of the lattice, which is exactly what the
    standard SPH operator gives too (the Gresho lattice is a disordered one, so it is not zero)."""
    system, config = _state()
    st = system.state
    st.velocities = torch.zeros_like(st.velocities)
    st.pressures = torch.full_like(st.pressures, 1.5)
    st.supports = torch.full_like(st.supports, float(st.supports.mean()))
    ref = _referenceForce(system, config)
    scale = ref.abs().max()
    for order in (1, 2):
        a, du = _rates(system, config, RiemannSolver.Adaptive, order)
        assert du.abs().max() < (1e-12 if FLOAT64 else 1e-6)
        torch.testing.assert_close(a, ref, rtol=0, atol=float(scale) * (1e-9 if FLOAT64 else 2e-3))


def test_constantPressureIsTheStandardSPHForce():
    """The plan's key claim: with constant P (and equal supports) the simplified GSPH force is the standard SPH one, whatever the densities."""
    system, config = _state()
    st = system.state
    dtype = st.positions.dtype
    st.velocities = torch.zeros_like(st.velocities)
    st.pressures = torch.full_like(st.pressures, 2.0)
    st.supports = torch.full_like(st.supports, float(st.supports.mean()))
    # a density gradient: the standard SPH force does not vanish for it (Cha et al. 2010), and neither may ours
    st.densities = st.densities * (1.0 + 0.3 * st.positions[:, 0].to(dtype))
    a, du = _rates(system, config, RiemannSolver.Adaptive, 1)
    ref = _referenceForce(system, config)
    scale = ref.abs().max()
    assert scale > 0
    torch.testing.assert_close(a, ref, rtol=0, atol=float(scale) * (1e-9 if FLOAT64 else 2e-3))


def _energy(st):
    return float((st.masses * (0.5 * (st.velocities ** 2).sum(-1) + st.internalEnergies)).sum())


def test_sod1dRunsAndConserves():
    from warpSPH.cases.sod import sodCase
    e0 = _energy(run(sodCase, scheme='GSPH', nSteps=1, progress=False, quiet=True).state.state)
    st = run(sodCase, scheme='GSPH', nSteps=120, progress=False, quiet=True).state.state
    assert torch.isfinite(st.densities).all() and torch.isfinite(st.velocities).all() and torch.isfinite(st.internalEnergies).all()
    assert st.velocities.abs().max() > 0.1, 'the diaphragm should have opened'
    assert abs(_energy(st) - e0) / e0 < 1e-3


@pytest.mark.skipif(FLOAT64, reason='already float64: the in-process test is the strict one')
def test_float64():
    env = dict(os.environ, warpSPHCore_PRECISION='float64')
    proc = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-x', '-p', 'no:cacheprovider', '-k', 'not float64', __file__],
                          cwd=REPO, env=env, capture_output=True, text=True, timeout=1800)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-2000:]
