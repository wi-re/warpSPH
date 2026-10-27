"""Material density must sharpen contacts without cutting force/RK support."""
import pytest
import torch
from warpSPHCore import (ParticleState, DomainDescription, KernelFunctions, SupportScheme,
                         buildVerletList, CRKState)
from warpSPHCore.radiusSearch.compactHash import buildCompactHashMap
from warpSPH.configurations import CRKSPHConfig, MaterialConfig
from warpSPH.configurations.crkSPH import crkSPHConfigToDict, dictToCRKSPHConfig
from warpSPH.modules.crk import crkFactors
from warpSPH.modules.crk.materialDensity import materialDensity


def contact(device='cpu'):
    n = 32
    x = (torch.arange(n, device=device, dtype=torch.float32)+.5)/n
    ids = (x >= .5).to(torch.int32)
    rho = torch.where(ids == 0, 1., 4.)
    s = ParticleState(x[:, None], torch.full_like(x, 4/n), rho/n, torch.zeros_like(ids), rho)
    s.materials = ids
    d = DomainDescription(x.new_tensor([0.]), x.new_tensor([1.]), torch.ones(1, dtype=torch.bool, device=device), 1)
    k = KernelFunctions.B7
    a = buildVerletList(s, d, verletScale=1.4, supportMode=SupportScheme.SuperSymmetric)
    return s, d, k, a


@pytest.mark.parametrize('policy', ['sameMaterial', 'massReplacement'])
def test_uniform_spacing_sharp_contact_and_unchanged_geometric_support(policy):
    s, d, k, a = contact()
    v0, r0, c0 = crkFactors(s, d, k, a)
    v, rho, c = crkFactors(s, d, k, a, densityPolicy=policy)
    assert ((r0/s.densities-1).abs().max()) > .1  # exercise the smoothed baseline
    torch.testing.assert_close(rho, s.densities, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(v, v0, rtol=0, atol=0)
    for name in ('A', 'B', 'gradA', 'gradB'):
        torch.testing.assert_close(getattr(c, name), getattr(c0, name), rtol=0, atol=0)


@pytest.mark.parametrize('policy', ['sameMaterial', 'massReplacement'])
def test_material_density_grid_matches_list(policy):
    s, d, k, a = contact()
    grid = buildCompactHashMap(s.positions, s.positions, s.supports, s.supports,
        d.periodic, d, SupportScheme.SuperSymmetric, len(s.masses)+1)
    v, expected, c = crkFactors(s, d, k, a, densityPolicy=policy)
    actual = materialDensity(s, d, k, v, c, grid, policy)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)


def test_same_material_density_reverse_ad_does_not_import_foreign_mass():
    s, d, k, a = contact()
    s.masses.requires_grad_(True)
    _, rho, _ = crkFactors(s, d, k, a, densityPolicy='sameMaterial')
    gradient, = torch.autograd.grad(rho[15], s.masses)
    assert torch.isfinite(gradient).all()
    assert gradient[16] == 0  # adjacent foreign material
    assert gradient[14] > 0
    eps = .0001
    s.masses = s.masses.detach().clone()
    s.masses[14] += eps
    plus = crkFactors(s, d, k, a, densityPolicy='sameMaterial')[1][15]
    s.masses[14] -= 2*eps
    minus = crkFactors(s, d, k, a, densityPolicy='sameMaterial')[1][15]
    assert float(gradient[14]) == pytest.approx(float((plus-minus)/(2*eps)), rel=.002)


@pytest.mark.parametrize('policy', ['sameMaterial', 'massReplacement'])
def test_material_density_geometry_gradient_matches_finite_difference(policy):
    s, d, k, a = contact()
    s.positions[14, 0] += .002
    s.positions.requires_grad_(True)
    _, rho, _ = crkFactors(s, d, k, a, densityPolicy=policy)
    grad, = torch.autograd.grad(rho[15], s.positions)
    assert torch.isfinite(grad).all()
    s.positions = s.positions.detach().clone()
    eps = .0001
    s.positions[14, 0] += eps
    plus = crkFactors(s, d, k, a, densityPolicy=policy)[1][15]
    s.positions[14, 0] -= 2*eps
    minus = crkFactors(s, d, k, a, densityPolicy=policy)[1][15]
    assert float(grad[14, 0]) == pytest.approx(float((plus-minus)/(2*eps)), rel=.01, abs=.01)


@pytest.mark.parametrize('policy', ['legacy', 'sameMaterial', 'massReplacement'])
def test_density_policy_config_roundtrip_and_material_requirement(policy):
    c = CRKSPHConfig(crkDensityPolicy=policy, materials=[MaterialConfig()])
    c.validate()
    assert dictToCRKSPHConfig(crkSPHConfigToDict(c)).crkDensityPolicy == policy
    c.materials = None
    if policy == 'legacy':
        c.validate()
    else:
        with pytest.raises(ValueError, match='explicit materials'):
            c.validate()


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('policy', ['sameMaterial', 'massReplacement'])
def test_material_density_dimension_overloads_preserve_constant_material_density(dim, policy):
    n = 12
    axis = (torch.arange(n, dtype=torch.float32)+.5)/n
    pos = torch.cartesian_prod(*([axis]*dim))
    ids = (pos[:, 0] >= .5).to(torch.int32)
    target = torch.where(ids == 0, 1., 4.)
    state = ParticleState(pos, torch.full_like(target, 4/n), target/n**dim, torch.zeros_like(ids), target)
    state.materials = ids
    domain = DomainDescription(torch.zeros(dim), torch.ones(dim), torch.ones(dim, dtype=torch.bool), dim)
    adjacency = buildVerletList(state, domain, verletScale=1.4, supportMode=SupportScheme.SuperSymmetric)
    volumes, rho, _ = crkFactors(state, domain, KernelFunctions.B7, adjacency, densityPolicy=policy)
    # Use the measured lattice volume: kernel quadrature normalization is a
    # separate effect from preserving the ratio of two material densities.
    torch.testing.assert_close(rho, state.masses/volumes, rtol=2e-5, atol=2e-5)
