"""Morris, Fox & Zhu (1997) shear-carrying viscosity (`ViscosityTerm.morris1997`,
`modules/deltaSPH/wp_viscosityDelta.py`, OPEN_PROBLEMS.md §7).

A pure shear wave `v = (sin(k y), 0)` on a periodic lattice: divergence-free,
so every bit of the viscous acceleration is shear, and the exact answer is
`nu * lap(v) = (-nu k^2 sin(k y), 0)`. The Morris term has to reproduce it.
"""

import math

import pytest
import torch


def _shearWave(n=32, supportFactor=4.0):
    from warpSPHCore import DomainDescription, ParticleState, radiusSearchCompactHashMap
    from warpSPHCore.enumTypes import SupportScheme
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32
    L = 2.0
    dx = L / n
    domain = DomainDescription(min=torch.tensor([-1.0, -1.0], dtype=dtype, device=device),
                               max=torch.tensor([1.0, 1.0], dtype=dtype, device=device),
                               periodic=torch.tensor([True, True], device=device), dim=2)
    g = -1.0 + dx * (torch.arange(n, dtype=dtype, device=device) + 0.5)
    X, Y = torch.meshgrid(g, g, indexing='ij')
    positions = torch.stack([X.flatten(), Y.flatten()], dim=-1)
    N = positions.shape[0]
    supports = torch.full((N,), supportFactor * dx, dtype=dtype, device=device)
    masses = torch.full((N,), dx * dx, dtype=dtype, device=device)       # rho0 = 1
    densities = torch.ones(N, dtype=dtype, device=device)
    kinds = torch.zeros(N, dtype=torch.int32, device=device)
    k = math.pi                                                          # one period over L
    velocities = torch.stack([torch.sin(k * positions[:, 1]), torch.zeros_like(positions[:, 1])], dim=-1)
    particles = ParticleState(positions=positions, supports=supports, masses=masses,
                              densities=densities, kinds=kinds)
    particles.velocities = velocities
    adjacency = radiusSearchCompactHashMap(particles, domain, mode=SupportScheme.SuperSymmetric)
    return particles, domain, adjacency, velocities, positions, k


def _diffusion(particles, domain, adjacency, velocities, nu, morris):
    from warpSPHCore import KernelFunctions, OperationProperties, WarpOperation
    from warpSPHCore.enumTypes import OperationDirection, SupportScheme
    from warpSPH.modules.deltaSPH.wp_viscosityDelta import computeVelocityDiffusionDeltaSPH
    return computeVelocityDiffusionDeltaSPH(
        particles,
        operationProperties=OperationProperties(kernel=KernelFunctions.Wendland2,
                                                operation=WarpOperation.Laplacian,
                                                supportMode=SupportScheme.SuperSymmetric,
                                                operationMode=OperationDirection.AllToAll),
        domain=domain, adjacency=adjacency, queryVelocities=velocities,
        inviscid=False, nu=nu, approachOnly=False, morris=morris)


def test_morrisReproducesTheShearLaplacian():
    nu = 0.01
    particles, domain, adjacency, velocities, positions, k = _shearWave()
    a = _diffusion(particles, domain, adjacency, velocities, nu, morris=True)
    exact = -nu * k * k * torch.sin(k * positions[:, 1])
    relErr = torch.linalg.norm(a[:, 0] - exact) / torch.linalg.norm(exact)
    assert relErr < 0.05, float(relErr)
    # divergence-free shear: nothing across the flow
    assert float(a[:, 1].abs().max()) < 0.05 * float(exact.abs().max())


def test_viscousTermDefaultsToTheProjectedForm():
    """The option is opt-in: the default config keeps the historical term, and
    it round-trips through the config dict."""
    from warpSPH.configurations.moduleConfigurations.weaklyCompressibleDiffusionParams import (
        buildDefaultDiffusionParamsWeaklyCompressibleSPH, dictToWCDiffusionParams, wcDiffusionParamsToDict)
    from warpSPH.enumTypes import ViscosityTerm
    params = buildDefaultDiffusionParamsWeaklyCompressibleSPH()
    assert params.viscousTerm == ViscosityTerm.monaghanGingold
    params.viscousTerm = ViscosityTerm.morris1997
    assert dictToWCDiffusionParams(wcDiffusionParamsToDict(params)).viscousTerm == ViscosityTerm.morris1997
    # an old config dict without the key still loads, on the default
    old = wcDiffusionParamsToDict(buildDefaultDiffusionParamsWeaklyCompressibleSPH())
    old.pop('viscousTerm')
    assert dictToWCDiffusionParams(old).viscousTerm == ViscosityTerm.monaghanGingold


@pytest.mark.parametrize('approachOnly', [False, True])
def test_approachOnlyRoundTripsAsADiffusionParameter(approachOnly):
    from warpSPH.configurations.moduleConfigurations.weaklyCompressibleDiffusionParams import (
        buildDefaultDiffusionParamsWeaklyCompressibleSPH, dictToWCDiffusionParams, wcDiffusionParamsToDict)
    params = buildDefaultDiffusionParamsWeaklyCompressibleSPH()
    assert not params.approachOnly
    params.approachOnly = approachOnly
    encoded = wcDiffusionParamsToDict(params)
    assert dictToWCDiffusionParams(encoded).approachOnly == approachOnly
    del encoded['approachOnly']
    assert not dictToWCDiffusionParams(encoded).approachOnly


def test_sharedViscosityUsesTheConfiguredApproachOnlyPolicy():
    from types import SimpleNamespace as NS
    from warpSPHCore import DomainDescription, KernelFunctions, ParticleState, buildVerletList
    from warpSPHCore.type_config import get_torch_precision
    from warpSPH.configurations import IncompressibleSPHConfig
    from warpSPH.modules.deltaSPH import computeVelocityDiffusion

    dtype = get_torch_precision()
    positions = torch.tensor([[-.2, 0.], [.2, 0.], [0., .3]], dtype=dtype)
    state = ParticleState(positions, torch.full((3,), .8, dtype=dtype),
        torch.ones(3, dtype=dtype), torch.zeros(3, dtype=torch.int32), torch.ones(3, dtype=dtype))
    state.velocities = positions.clone()  # Every pair is separating.
    domain = DomainDescription(-torch.ones(2, dtype=dtype), torch.ones(2, dtype=dtype),
                               torch.zeros(2, dtype=torch.bool), 2)
    config = NS(domain=domain, kernel=KernelFunctions.Wendland4, calibrateNormalization=False)
    scheme = IncompressibleSPHConfig()
    scheme.diffusionParams.inviscid = False
    adjacency = buildVerletList(state, domain)
    scheme.diffusionParams.approachOnly = True
    clamped = computeVelocityDiffusion(state, config, scheme, adjacency)
    torch.testing.assert_close(clamped, torch.zeros_like(clamped))
    scheme.diffusionParams.approachOnly = False
    twoSided = computeVelocityDiffusion(state, config, scheme, adjacency)
    assert float((twoSided*state.velocities).sum()) < 0


@pytest.mark.parametrize('rho0', [1., 1000.])
def test_sharedProjectedViscosityUsesPhysicalUnits(rho0):
    from types import SimpleNamespace as NS
    from warpSPHCore import KernelFunctions
    from warpSPH.configurations import IncompressibleSPHConfig
    from warpSPH.modules.deltaSPH import computeVelocityDiffusion

    state, domain, adjacency, velocities, positions, k = _shearWave()
    state.masses *= rho0
    state.densities *= rho0
    scheme = IncompressibleSPHConfig()
    scheme.fluid.restDensity = rho0
    scheme.diffusionParams.inviscid = False
    scheme.diffusionParams.viscidNu = .01
    config = NS(domain=domain, kernel=KernelFunctions.Wendland2, calibrateNormalization=False)
    acceleration = computeVelocityDiffusion(state, config, scheme, adjacency, wall=False)
    exact = -.01*k*k*torch.sin(k*positions[:, 1])
    error = torch.linalg.norm(acceleration[:, 0]-exact)/torch.linalg.norm(exact)
    assert float(error) < .05
    torch.testing.assert_close(state.velocities, velocities, rtol=0, atol=0)
