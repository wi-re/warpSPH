"""PESPH_PLAN Phase 0 / Phase 2 (first bullet): the number-density driven support solve
(`AdaptiveSupportScheme.NumberDensity`) and the ideal-gas EOS round trip through the entropic function."""

import pytest
import torch

from warpSPH.configurations import buildConfig, CompressibleSPHConfig
from warpSPH.enumTypes import AdaptiveSupportScheme, CompressibleSPHScheme, PESPHVariant
from warpSPH.modules import idealGasEOS
from warpSPH.modules.adaptiveSupport.optimalSupport import evaluateOptimalSupport
from warpSPH.schemes.builder import buildScheme
from warpSPH.systems import CompressibleState
from warpSPH.utils.domain import buildDomainDescription
from warpSPH.utils.support import volumeToSupport
from warpSPHCore import KernelFunctions, SupportScheme

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _config(dim, n_h=4.0, l=2.0):
    domain = buildDomainDescription(l, dim, periodic=True, device=DEV, dtype=torch.float32)
    cfg, _ = buildConfig(dim=dim, domain=domain, kernel=KernelFunctions.B7, n_h=n_h, device=DEV, dtype=torch.float32)
    cfg.supportVolumeClamp = 'off'
    return cfg


def _state(x, m, h):
    n = x.shape[0]
    z = torch.zeros(n, device=DEV)
    return CompressibleState(
        positions=x.to(DEV), velocities=torch.zeros_like(x, device=DEV), supports=h.to(DEV).clone(),
        masses=m.to(DEV), densities=torch.ones(n, device=DEV),
        kinds=torch.zeros(n, dtype=torch.int32, device=DEV), materials=torch.zeros(n, dtype=torch.int32, device=DEV),
        UIDs=torch.arange(n, dtype=torch.int32, device=DEV), UIDcounter=n,
        internalEnergies=z.clone(), totalEnergies=z.clone(), entropies=z.clone(), pressures=z.clone(),
        soundspeeds=z.clone(), divergence=z.clone(), alpha0s=z.clone(), alphas=z.clone())


def _jitteredLattice(dim, nx, jitter=0.15, seed=0, l=2.0):
    g = torch.Generator().manual_seed(seed)
    dx = l / nx
    ax = torch.arange(nx, dtype=torch.float32) * dx - l / 2 + dx / 2
    grids = torch.meshgrid(*([ax] * dim), indexing='ij')
    x = torch.stack([q.reshape(-1) for q in grids], dim=1)
    return x + jitter * dx * (torch.rand(x.shape, generator=g) - 0.5), dx


def _solve(state, cfg, scheme, iters=40):
    comp = CompressibleSPHConfig(adaptiveSupportScheme=scheme, adaptiveSupportIterations=iters,
                                 adaptiveSupportThreshold=0.0, adaptiveSupportCorrections=False)
    rho, h, _adj, ns, _hs = evaluateOptimalSupport(state, cfg, comp, SupportScheme.Gather, None)
    return rho, h, ns


@pytest.mark.parametrize('dim,nx', [(1, 160), (2, 40)])
def test_equal_masses_match_the_mass_density_solve(dim, nx):
    cfg = _config(dim)
    x, dx = _jitteredLattice(dim, nx)
    m = torch.full((x.shape[0],), dx ** dim)
    h0 = torch.full((x.shape[0],), 1.4 * cfg.n_h * dx)
    rhoM, hM, _ = _solve(_state(x, m, h0), cfg, AdaptiveSupportScheme.Monaghan)
    rhoN, hN, _ = _solve(_state(x, m, h0), cfg, AdaptiveSupportScheme.NumberDensity)
    assert torch.allclose(hN, hM, rtol=1e-4)
    assert torch.allclose(rhoN, rhoM, rtol=1e-4)


def test_unequal_masses_do_not_enter_h_and_satisfy_the_constraint():
    dim, nx = 2, 40
    cfg = _config(dim)
    x, dx = _jitteredLattice(dim, nx)
    h0 = torch.full((x.shape[0],), 1.3 * cfg.n_h * dx)
    out = []
    for seed in (1, 2):
        m = dx ** dim * (1.0 + 3.0 * torch.rand(x.shape[0], generator=torch.Generator().manual_seed(seed)))
        st = _state(x, m, h0)
        rho, h, ns = _solve(st, cfg, AdaptiveSupportScheme.NumberDensity)
        out.append((m.to(DEV), rho, h, ns[-1]))
        # the state is left as it was found: real masses, original densities
        assert torch.equal(st.masses, m.to(DEV))
        assert torch.equal(st.densities, torch.ones_like(st.densities))
    (m1, rho1, h1, n1), (m2, rho2, h2, n2) = out
    # h and n depend on positions only; rho_i = m_i n_i
    assert torch.allclose(h1, h2, rtol=1e-4)
    assert torch.allclose(n1, n2, rtol=1e-4)
    assert torch.allclose(rho1, m1 * n1, rtol=1e-5)
    # the Newton fixed point: h_i = n_h * n_i^(-1/d)
    hFix = volumeToSupport(1.0 / n1, cfg.targetNeighbors, dim)
    assert ((h1 - hFix).abs() / h1).max() < 2e-3
    # and the mass-density solve genuinely differs here (the test discriminates)
    _, hMass, _ = _solve(_state(x, m1.cpu(), h0), cfg, AdaptiveSupportScheme.Monaghan)
    assert ((hMass - h1).abs() / h1).max() > 1e-2


def test_pesph_is_an_enum_member_only_until_phase_4():
    assert CompressibleSPHScheme.PESPH.value == 5
    assert {v.name for v in PESPHVariant} == {'PressureEnergy', 'PressureEntropy'}
    with pytest.raises(NotImplementedError, match='Phase 4'):
        buildScheme(CompressibleSPHScheme.PESPH)
    with pytest.raises(NotImplementedError, match='Phase 4'):
        buildScheme('PESPH')


@pytest.mark.parametrize('gamma', [1.4, 5.0 / 3.0])
def test_ideal_gas_entropy_round_trip(gamma):
    g = torch.Generator().manual_seed(0)
    rho = 0.2 + 3.0 * torch.rand(512, generator=g, dtype=torch.float64)
    A = 0.1 + 2.0 * torch.rand(512, generator=g, dtype=torch.float64)
    A_, u, P, c = idealGasEOS(A=A, u=None, P=None, rho=rho, gamma=gamma)
    assert torch.equal(A_, A)
    assert torch.allclose(P, (gamma - 1) * rho * u, rtol=1e-12)
    assert torch.allclose(c, torch.sqrt(gamma * P / rho), rtol=1e-12)
    # back to A from u, from P, and from (u, P)
    for kwargs in ({'u': u, 'P': None}, {'u': None, 'P': P}):
        A2, u2, P2, c2 = idealGasEOS(A=None, rho=rho, gamma=gamma, **kwargs)
        assert torch.allclose(A2, A, rtol=1e-12)
        assert torch.allclose(u2, u, rtol=1e-12)
        assert torch.allclose(P2, P, rtol=1e-12)
        assert torch.allclose(c2, c, rtol=1e-12)
