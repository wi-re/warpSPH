"""SUPPORT_SOLVER_PLAN option C: the per-side (Price 2012 Eqs. 43-45) pressure pair conserves momentum and
energy pair by pair for any per-particle factor, and reduces to the mean-kernel symmetric force at equal h."""

import pytest
import torch

from warpSPH.configurations import buildConfig
from warpSPH.modules.pressure import computePerSidePressureWarp, computePressureForceSymmetric
from warpSPH.systems import CompressibleState
from warpSPH.utils.domain import buildDomainDescription
from warpSPHCore import KernelFunctions, OperationProperties, SupportScheme, buildVerletList

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _setup(dim, n, equalH, seed=0):
    g = torch.Generator().manual_seed(seed)
    domain = buildDomainDescription(2.0, dim, periodic=True, device=DEV, dtype=torch.float32)
    cfg, _ = buildConfig(dim=dim, domain=domain, kernel=KernelFunctions.B7, n_h=4.0, device=DEV, dtype=torch.float32)
    x = (torch.rand(n, dim, generator=g) * 2 - 1)
    dx = 2.0 / n ** (1 / dim)
    h = torch.full((n,), 4 * dx) if equalH else 4 * dx * (0.7 + 0.6 * torch.rand(n, generator=g))
    m = torch.full((n,), dx ** dim) * (0.5 + torch.rand(n, generator=g))
    z = torch.zeros(n)
    st = CompressibleState(
        positions=x.to(DEV), velocities=torch.randn(n, dim, generator=g).to(DEV), supports=h.to(DEV), masses=m.to(DEV),
        densities=(0.5 + torch.rand(n, generator=g)).to(DEV),
        kinds=torch.zeros(n, dtype=torch.int32, device=DEV), materials=torch.zeros(n, dtype=torch.int32, device=DEV),
        UIDs=torch.arange(n, dtype=torch.int32, device=DEV), UIDcounter=n,
        internalEnergies=z.to(DEV), totalEnergies=z.to(DEV), entropies=z.to(DEV),
        pressures=(0.2 + torch.rand(n, generator=g)).to(DEV), soundspeeds=z.to(DEV),
        divergence=z.to(DEV), alpha0s=z.to(DEV), alphas=z.to(DEV))
    adj = buildVerletList(st, domain, verletScale=1.0, supportMode=SupportScheme.SuperSymmetric)
    props = OperationProperties(kernel=cfg.kernel, supportMode=SupportScheme.KernelMeanSymmetric)
    return cfg, st, adj, props, g


@pytest.mark.parametrize('dim', [1, 2, 3])
@pytest.mark.parametrize('withOmega', [False, True])
def test_per_side_pair_conserves_momentum_and_energy(dim, withOmega):
    n = {1: 200, 2: 600, 3: 1500}[dim]
    cfg, st, adj, props, g = _setup(dim, n, equalH=False)
    omegas = (0.6 + 0.8 * torch.rand(n, generator=g)).to(DEV) if withOmega else None
    a, dudt = computePerSidePressureWarp(st, props, cfg.domain, adj, queryOmegas=omegas)
    a, dudt = a.double(), dudt.double()
    m, v = st.masses.double(), st.velocities.double()
    scaleP = (m[:, None] * a).abs().sum()
    assert (m[:, None] * a).sum(0).abs().max() < 1e-5 * scaleP
    dEkin = (m[:, None] * v * a).sum()
    dEint = (m * dudt).sum()
    assert abs(dEkin + dEint) < 1e-5 * (dEkin.abs() + dEint.abs())


@pytest.mark.parametrize('dim', [1, 2])
def test_per_side_equals_mean_kernel_force_at_equal_h(dim):
    cfg, st, adj, props, _ = _setup(dim, {1: 200, 2: 600}[dim], equalH=True)
    a, _ = computePerSidePressureWarp(st, props, cfg.domain, adj)
    ref = computePressureForceSymmetric(st, cfg, supportScheme=SupportScheme.KernelMeanSymmetric, adjacency=adj)
    assert torch.allclose(a, ref, rtol=1e-4, atol=1e-4 * ref.abs().max().item())
