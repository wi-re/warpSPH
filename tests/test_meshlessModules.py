"""Frontend meshless stages against the retained core oracle and handover lifetime."""
from types import SimpleNamespace

import pytest
import torch

from warpSPHCore import DomainDescription, KernelFunctions, get_torch_precision
from warpSPHCore.mfm import MeshlessWarp, conserved, primitives as referencePrimitives
from warpSPH.modules.meshless.operators import computeRates
from warpSPH.modules.meshless.primitives import primitives, refreshPrimitives
from warpSPH.modules.meshless.geometry import stashGeometry, takeGeometry, inheritGeometry
from warpSPH.configurations.meshless import MFMConfig

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = get_torch_precision()


def domain(dim, device=DEVICE):
    return DomainDescription(torch.zeros(dim, device=device, dtype=DTYPE),
        torch.ones(dim, device=device, dtype=DTYPE), torch.ones(dim, device=device, dtype=torch.bool), dim)


@pytest.fixture(scope='module', params=[1, 2, 3])
def flow(request):
    dim = request.param
    n = 7 if dim < 3 else 4
    axes = [torch.arange(n, dtype=DTYPE) / n for _ in range(dim)]
    pos = torch.stack(torch.meshgrid(*axes, indexing='ij'), -1).reshape(-1, dim)
    gen = torch.Generator().manual_seed(941)
    rnd = lambda shape: torch.rand(shape, generator=gen, dtype=DTYPE).to(DEVICE)
    pos = (pos.to(DEVICE) + .05 / n * (rnd(pos.shape) - .5)) % 1.
    h = torch.full((len(pos),), min(.8, 2.6 / n), dtype=DTYPE, device=DEVICE)
    geometry = MeshlessWarp.build(pos, h, domain(dim), KernelFunctions.B7, closure='none')
    rho, velocity, pressure = 1. + rnd((len(pos),)), rnd(pos.shape) - .5, .2 + rnd((len(pos),))
    Q = conserved(rho * geometry.volume, rho, velocity, pressure, 1.4)
    return geometry, Q, rho, velocity, pressure


@pytest.mark.parametrize('mode', ['MFM', 'MFV'])
@pytest.mark.parametrize('order,dt,limit', [(1, .001, .1), (2, 0., .1), (2, .003, .001), (2, .003, 0.)])
def test_fluxMatchesCore(flow, mode, order, dt, limit):
    geom, Q, rho, vel, pressure = flow
    kw = dict(mode=mode, order=order, dt=dt, massFluxLimit=limit)
    expected = geom.rates(rho, vel, pressure, 1.4, **kw)
    kw['dt'] = torch.tensor(dt, dtype=DTYPE, device=DEVICE)
    actual = computeRates(geom, rho, vel, pressure, 1.4, **kw)
    eps = torch.finfo(DTYPE).eps
    torch.testing.assert_close(actual, expected, rtol=64 * eps, atol=64 * eps * expected.abs().max().item())
    assert actual.sum(0).abs().max() < len(Q) * 8 * eps * actual.abs().max()
    if mode == 'MFM':
        assert torch.count_nonzero(actual[:, 0]) == 0


def test_primitiveRefreshMatchesReference(flow):
    geom, Q, *_ = flow
    rho, vel, pressure = referencePrimitives(Q, geom.volume, 1.4)
    state = SimpleNamespace(Q=Q)
    refreshPrimitives(state, geom, 1.4, 32., vel.shape[1])
    eps = torch.finfo(DTYPE).eps
    for got, expected in ((state.densities, rho), (state.velocities, vel), (state.pressures, pressure),
                          (state.internalEnergies, Q[:, -1] / Q[:, 0].clamp_min(1e-30) - .5 * (vel * vel).sum(-1)),
                          (state.soundspeeds, (1.4 * pressure / rho.clamp_min(1e-30)).sqrt()),
                          (state.entropies, pressure / rho.clamp_min(1e-30) ** 1.4)):
        torch.testing.assert_close(got, expected, rtol=16 * eps, atol=16 * eps)
    from warpSPH.modules.meshless.context import supports_from_volume
    torch.testing.assert_close(state.supports, supports_from_volume(geom.volume, 32., vel.shape[1]), rtol=16 * eps, atol=16 * eps)


def test_massFloorsRemainDistinct():
    Q = torch.tensor([[1e-31, 0., 0., 1e-31]], dtype=DTYPE, device=DEVICE)
    V = torch.ones(1, dtype=DTYPE, device=DEVICE)
    rho, vel, pressure, u = primitives(Q, V, 1.4, returnEnergy=True)
    ref = referencePrimitives(Q, V, 1.4)
    for got, expected in zip((rho, vel, pressure), ref):
        torch.testing.assert_close(got, expected)
    torch.testing.assert_close(u, Q[:, -1] / Q[:, 0].clamp_min(1e-30))


@pytest.mark.parametrize('change', ['positions', 'supports', 'source', 'domain', 'kernel', 'conditioning', 'unrelated', None])
def test_geometryHandoverChecksCloneVersions(change, monkeypatch):
    dom, cfg = domain(2, 'cpu'), MFMConfig()
    source = SimpleNamespace(positions=torch.zeros(4, 2), supports=torch.ones(4))
    geom = object()
    stashGeometry(source.positions, source.supports, dom, KernelFunctions.B7, cfg, geom, owner=source)
    target = SimpleNamespace(positions=source.positions.clone(), supports=source.supports.clone())
    inheritGeometry(source, target)
    kernel = KernelFunctions.B7
    if change in ('positions', 'supports'):
        getattr(target, change).add_(.1)
    elif change == 'source':
        source.positions.add_(.1)
    elif change == 'domain':
        dom.max.add_(.1)
    elif change == 'kernel':
        kernel = KernelFunctions.B8
    elif change == 'conditioning':
        cfg.condMax *= 2
    elif change == 'unrelated':
        target = SimpleNamespace(positions=source.positions.clone(), supports=source.supports.clone())
    monkeypatch.setattr(torch, 'equal', lambda *args: pytest.fail('GPU equality in state-owned handover'))
    result = takeGeometry(target.positions, target.supports, dom, kernel, cfg, owner=target)
    assert (result is geom) == (change is None)
    assert takeGeometry(target.positions, target.supports, dom, kernel, cfg, owner=target) is None


def test_geometryTicketsDoNotCrossSystems():
    cfg, dom = MFMConfig(), domain(1, 'cpu')
    states = [SimpleNamespace(positions=torch.zeros(4, 1), supports=torch.ones(4)) for _ in range(2)]
    geometries = [object(), object()]
    for state, geom in zip(states, geometries):
        stashGeometry(state.positions, state.supports, dom, KernelFunctions.B7, cfg, geom, owner=state)
    for state, geom in zip(states, geometries):
        assert takeGeometry(state.positions, state.supports, dom, KernelFunctions.B7, cfg, owner=state) is geom


def test_stateOwnedHandoverCannotConsumeLegacyGlobalTicket(monkeypatch):
    cfg, dom = MFMConfig(), domain(1, 'cpu')
    positions, supports = torch.zeros(4, 1), torch.ones(4)
    geometry = object()
    stashGeometry(positions, supports, dom, KernelFunctions.B7, cfg, geometry)
    unrelated = SimpleNamespace(positions=positions.clone(), supports=supports.clone())
    monkeypatch.setattr(torch, 'equal', lambda *args: pytest.fail('state-owned handover read a global ticket'))
    assert takeGeometry(unrelated.positions, unrelated.supports, dom, KernelFunctions.B7, cfg, owner=unrelated) is None
    assert takeGeometry(positions, supports, dom, KernelFunctions.B7, cfg) is geometry


def test_forwardOnlyBoundary(flow):
    geom, Q, rho, vel, pressure = flow
    with pytest.raises(NotImplementedError, match='forward-only'):
        primitives(Q.clone().requires_grad_(), geom.volume, 1.4)
    with pytest.raises(NotImplementedError, match='forward-only'):
        computeRates(geom, rho, vel, pressure, 1.4, dt=torch.tensor(.001, device=DEVICE, dtype=DTYPE, requires_grad=True))
    with torch.autograd.forward_ad.dual_level():
        q = torch.autograd.forward_ad.make_dual(Q, torch.ones_like(Q))
        with pytest.raises(NotImplementedError, match='forward-only'):
            primitives(q, geom.volume, 1.4)


@pytest.mark.parametrize('guard', [0, 1, 2, 3, 4])
def test_correctedFacesAndRetryRemainCoreEquivalent(guard):
    dim, n = 2, 24
    gen = torch.Generator().manual_seed(87)
    rnd = lambda shape: torch.rand(shape, generator=gen, dtype=DTYPE).to(DEVICE)
    geom = MeshlessWarp.build(rnd((n, dim)), torch.full((n,), .7, dtype=DTYPE, device=DEVICE), domain(dim),
        KernelFunctions.B7, closure='none', face_cond_max=.5)
    geom.lam = .02 * (rnd((n, dim)) - .5)
    geom.closure_cap = .01
    rho, vel = .2 + rnd((n,)), 5. * (rnd((n, dim)) - .5)
    pressure = torch.where(geom.particles.positions[:, 0] > .5, 1.e3, 1.).to(DTYPE)
    eps = torch.finfo(DTYPE).eps
    for mode in ('MFM', 'MFV'):
        kwargs = dict(dt=.002, mode=mode, faceGuard=guard, retry=True)
        ref = geom.rates(rho, vel, pressure, 1.4, **kwargs)
        got = computeRates(geom, rho, vel, pressure, 1.4, **kwargs)
        torch.testing.assert_close(got, ref, rtol=128 * eps, atol=128 * eps * ref.abs().max().item())


@pytest.mark.parametrize('raw,previous,growth', [(1.e-12, .1, 1.25), (.2, .1, 1.25), (.05, .1, .5), (.2, .1, .5)])
def test_timestepClampsStayOnDevice(raw, previous, growth, monkeypatch):
    import warpSPH.modules.timestep.meshless as timestep
    cfg = SimpleNamespace(adaptiveDt=True, minDt=1.e-8, maxDt=.5, dt=previous, dtGrowthFactor=growth, kernel=KernelFunctions.B7)
    params = MFMConfig(cflLength='support')
    st = SimpleNamespace(positions=torch.zeros(2, 1), supports=torch.ones(2), Q=torch.zeros(2, 3))
    dt = torch.tensor(raw, dtype=DTYPE)
    geom = SimpleNamespace(volume=torch.ones(2), timestep=lambda *args: dt.expand(2))
    monkeypatch.setattr(timestep.MeshlessWarp, 'build', lambda *args, **kwargs: geom)
    monkeypatch.setattr(timestep, 'primitives', lambda *args: (None, None, None))
    expected = dt.clamp(cfg.minDt, cfg.maxDt)
    if expected > previous:
        expected = expected.clamp(max=growth * previous)
    def forbidBool(tensor):
        pytest.fail('timestep read a device value in Python')
    monkeypatch.setattr(torch.Tensor, '__bool__', forbidBool)
    got = timestep.computeTimestep(SimpleNamespace(state=st, domain=domain(1, 'cpu')), cfg, params, dt=previous)
    assert got.item() == expected.item()


def test_sharedIdealGasSupportsReverseAndForwardAD():
    from warpSPH.modules.eos import idealGasPrimitives
    Q = torch.tensor([[1., .1, .2, 3.], [2., .3, .4, 5.]], dtype=torch.float64, requires_grad=True)
    volume = torch.tensor([.8, 1.2], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda q, v: idealGasPrimitives(q, v, 1.4), (Q, volume))
    with torch.autograd.forward_ad.dual_level():
        q = torch.autograd.forward_ad.make_dual(Q.detach(), torch.ones_like(Q))
        out = idealGasPrimitives(q, volume.detach(), 1.4)
        assert all(torch.autograd.forward_ad.unpack_dual(t).tangent is not None for t in out)


@pytest.mark.parametrize('mode', ['MFM', 'MFV'])
@pytest.mark.parametrize('order', [1, 2])
def test_gridAndNeighborFluxUseTheSamePairPhysics(mode, order):
    import copy
    from warpSPHCore import SupportScheme
    from warpSPHCore.radiusSearch import buildCompactHashMap
    from warpSPHCore.autograd import SPHContext, PreparedLauncher
    axes = torch.arange(9, device=DEVICE, dtype=DTYPE) / 9.
    pos = torch.stack(torch.meshgrid(axes, axes, indexing='ij'), -1).reshape(-1, 2)
    h = torch.full((len(pos),), .25, dtype=DTYPE, device=DEVICE)
    geom = MeshlessWarp.build(pos, h, domain(2), KernelFunctions.B7, closure='none')
    grid = buildCompactHashMap(pos, pos, h, h, periodicity=geom.domain.periodic,
                              domainDescription=geom.domain, mode=SupportScheme.SuperSymmetric)
    gridGeom = copy.copy(geom)
    gridGeom.adjacency = grid
    gridGeom._fast = PreparedLauncher(SPHContext(query=geom.particles, properties=geom.props, domain=geom.domain, adjacency=grid))
    assert geom._fast.useAdjacency and not gridGeom._fast.useAdjacency
    rho = 1. + .1 * torch.sin(6. * pos[:, 0])
    pressure = 1. + .1 * torch.cos(5. * pos[:, 1])
    vel = .3 * torch.stack([pos[:, 1], -pos[:, 0]], -1)
    # Different traversal order may change summation roundoff, never drop a cell's momentum.
    kwargs = dict(mode=mode, order=order, dt=.001)
    expected = computeRates(geom, rho, vel, pressure, 1.4, **kwargs)
    got = computeRates(gridGeom, rho, vel, pressure, 1.4, **kwargs)
    eps = torch.finfo(DTYPE).eps
    torch.testing.assert_close(got, expected, rtol=256 * eps, atol=256 * eps * expected.abs().max().item())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA stream ownership')
@pytest.mark.parametrize('capture', [False, True])
def test_fusedPrimitivesFollowCallerStream(capture, monkeypatch):
    import warp as wp
    from warpSPHCore.autograd.stream import torchWarpCapture, torchWarpStream
    from warpSPH.modules.eos import idealGasPrimitives
    work = torch.cuda.Stream()
    device = wp.device_from_torch(torch.device('cuda', torch.cuda.current_device()))
    unrelated = wp.Stream(device)
    original = wp.launch
    seen = []
    def launch(kernel, *args, **kwargs):
        assert wp.get_stream(device).cuda_stream == torch.cuda.current_stream().cuda_stream
        seen.append(kernel.key)
        return original(kernel, *args, **kwargs)
    monkeypatch.setattr(wp, 'launch', launch)
    with wp.ScopedStream(unrelated, sync_enter=False, sync_exit=False):
        with torch.cuda.stream(work):
            Q = torch.tensor([[1., .1, .2, 3.], [2., .3, .4, 5.]], device='cuda', dtype=DTYPE)
            volume = torch.tensor([.8, 1.2], device='cuda', dtype=DTYPE)
            state, geom = SimpleNamespace(Q=Q), SimpleNamespace(volume=volume)
            idealGasPrimitives(Q, volume, 1.4)
            refreshPrimitives(state, geom, 1.4, 32., 2)
        work.synchronize()
        if capture:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.stream(work):
                with torchWarpStream(torch.device('cuda')), torchWarpCapture(graph):
                    refreshPrimitives(state, geom, 1.4, 32., 2)
                Q[:, -1].add_(.25)
                graph.replay()
            work.synchronize()
        assert wp.get_stream(device) is unrelated
    mass = Q[:, 0]
    velocity = Q[:, 1:-1] / mass[:, None]
    energy = Q[:, -1] / mass - .5 * velocity.square().sum(-1)
    density = mass / volume
    pressure = .4 * density * energy
    torch.testing.assert_close(state.densities, density)
    torch.testing.assert_close(state.velocities, velocity)
    torch.testing.assert_close(state.pressures, pressure)
    torch.testing.assert_close(state.soundspeeds, (1.4 * pressure / density).sqrt())
    assert len(seen) == (5 if capture else 3)


def test_disabledAutogradKeepsNeighborTraversalMetadata(flow):
    geom, Q, rho, velocity, pressure = flow
    pos = geom.particles.positions.detach().clone().requires_grad_(True)
    with torch.no_grad():
        fallback = MeshlessWarp.build(pos, geom.particles.supports, geom.domain,
                                      KernelFunctions.B7, closure='none')
        assert not fallback._fast._ok
        assert fallback._fast.useAdjacency
        got = computeRates(fallback, rho, velocity, pressure, 1.4, dt=.001)
    expected = computeRates(geom, rho, velocity, pressure, 1.4, dt=.001)
    eps = torch.finfo(DTYPE).eps
    torch.testing.assert_close(got, expected, rtol=128 * eps, atol=128 * eps * expected.abs().max().item())
    assert not got.requires_grad


@pytest.mark.parametrize('scheme', ['MFM', 'MFV'])
@pytest.mark.parametrize('shape,boost', [('box', 0.), ('ellipse', .7)])
def test_nativePressureEquilibriumStartsBalancedAndAdvects(scheme, shape, boost):
    from warpSPH.runner import RunConfiguration, run
    from warpSPH.cases.compressible.hydrostatic import hydrostaticCase
    ctx = RunConfiguration('hydrostatic', simulation=dict(device=DEVICE, nx=24,
        nSteps=40, plot=False, video=False, quiet=True, progress=False),
        params=dict(rho_high=4., gamma=1.5, shape=shape, boost=boost),
        solver=dict(scheme=scheme)).prepare(hydrostaticCase)
    system = hydrostaticCase.buildSystem(ctx)
    geom = MeshlessWarp.build(system.state.positions, system.state.supports,
        system.domain, ctx.config.kernel, closure='none')
    rho, vel, pressure = primitives(system.state.Q, geom.volume, 1.5)
    eps = torch.finfo(DTYPE).eps
    torch.testing.assert_close(pressure, torch.ones_like(pressure), atol=16*eps, rtol=16*eps)
    # Cell masses retain the density jump; it is not smeared by SPH support adaptation.
    assert torch.unique(system.state.masses).numel() == 2
    initialMass = system.state.masses.sum()
    r = run(hydrostaticCase, ctx.spec)
    assert r.nSteps == 40 and not r.diverged
    metrics = r.trajectory[-1]
    assert metrics['pressureError'] < 3e-4
    assert metrics['maxPeculiarVelocity'] < 3e-4
    assert metrics['shapeDisplacement'] < 3e-4
    assert metrics['relativeEnergyDrift'] < 3e-5
    torch.testing.assert_close(r.state.state.masses.sum(), initialMass, atol=64*eps, rtol=64*eps)


@pytest.mark.parametrize('mode', ['MFM', 'MFV'])
def test_materialFluxesConserveEachPureFluidAndRetainMfvExchange(flow, mode):
    geom, Q, rho, vel, pressure = flow
    ids = (torch.arange(len(rho), device=rho.device) % 3 == 0).to(torch.int32)
    gammas = torch.where(ids == 0, torch.full_like(rho, 1.4), torch.full_like(rho, 5/3))
    rate = computeRates(geom, rho, vel, pressure, gammas, mode=mode, dt=.001, materials=ids)
    eps = torch.finfo(DTYPE).eps
    tolerance = len(rho)*64*eps*rate.abs().max()
    assert rate.sum(0).abs().max() < tolerance
    for material in (0, 1):
        assert rate[ids == material, 0].sum().abs() < tolerance
    if mode == 'MFM':
        assert torch.count_nonzero(rate[:, 0]) == 0
    else:
        assert rate[:, 0].abs().max() > 0  # within-fluid MFV mass exchange remains active


@pytest.mark.parametrize('scheme', ['MFM', 'MFV'])
@pytest.mark.parametrize('variant', ['twoGas', 'advectedEllipse'])
def test_materialContactInitializationAndAcceptedStateUseTheSameEos(scheme, variant):
    from warpSPH.runner import RunConfiguration, run
    from warpSPH.cases.compressible.hydrostatic import hydrostaticCase
    ctx = RunConfiguration('hydrostatic', variant=variant,
        simulation=dict(device=DEVICE, nx=24, nSteps=40, quiet=True, progress=False),
        solver=dict(scheme=scheme)).prepare(hydrostaticCase)
    initial = hydrostaticCase.buildSystem(ctx).state
    ids = initial.materials
    gamma = torch.where(ids == 0, torch.full_like(initial.masses, 1.4), torch.full_like(initial.masses, 5/3))
    expected = (gamma-1)*initial.densities*(initial.Q[:, -1]/initial.masses-.5*initial.velocities.square().sum(-1))
    eps = torch.finfo(DTYPE).eps
    torch.testing.assert_close(expected, torch.ones_like(expected), atol=32*eps, rtol=32*eps)
    r = run(hydrostaticCase, ctx.spec)
    assert r.nSteps == 40 and not r.diverged
    assert r.trajectory[-1]['pressureError'] < 4e-4
    assert r.trajectory[-1]['maxPeculiarVelocity'] < 4e-4
    for material in (0, 1):
        before = initial.masses[ids == material].sum()
        after = r.state.state.masses[ids == material].sum()
        torch.testing.assert_close(after, before, atol=128*eps, rtol=128*eps)
    state = r.state.state
    torch.testing.assert_close(state.soundspeeds.square(), gamma*state.pressures/state.densities, rtol=64*eps, atol=64*eps)


def test_explicit_material_primitive_refresh_does_not_hide_negative_energy():
    Q = torch.tensor([[1., 0., -1.]], device=DEVICE, dtype=DTYPE)
    volume = torch.ones(1, device=DEVICE, dtype=DTYPE)
    gamma = torch.full_like(volume, 1.4)
    state = SimpleNamespace(Q=Q)
    refreshPrimitives(state, SimpleNamespace(volume=volume), gamma, 8., 1)
    assert state.internalEnergies[0] < 0
    assert state.pressures[0] < 0
    assert torch.isnan(state.soundspeeds[0])


def test_material_gamma_inputs_retain_the_forward_only_contract(flow):
    geom, Q, rho, vel, pressure = flow
    gamma = torch.full_like(rho, 1.4, requires_grad=True)
    with pytest.raises(NotImplementedError, match='forward-only'):
        computeRates(geom, rho, vel, pressure, gamma)
