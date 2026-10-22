"""Delta stage ownership, pointwise arithmetic, stream and correction contracts."""
from types import SimpleNamespace as NS
import pytest
import torch
import warp as wp
from warpSPHCore.type_config import get_torch_precision
from warpSPH.configurations.weaklyCompressible import Sun2017DeltaSPHConfig
from warpSPH.enumTypes import DensityDiffusionScheme
from warpSPH.modules.deltaSPH.stages import DeltaStages, diffusionRates
from warpSPH.modules.deltaSPH.updates import assembleUpdate, maskFluidUpdate
from warpSPH.modules.surfaceDetection.stages import surfaceMaskAcrossStages


def _state(device='cpu'):
    from warpSPH.systems.weaklyCompressible import WeaklyCompressibleState
    dtype = get_torch_precision()
    n = 4
    return WeaklyCompressibleState(positions=torch.zeros(n, 2, dtype=dtype, device=device),
        velocities=torch.arange(n * 2, dtype=dtype, device=device).reshape(n, 2) / 10,
        densities=torch.ones(n, dtype=dtype, device=device), masses=torch.ones(n, dtype=dtype, device=device),
        supports=torch.ones(n, dtype=dtype, device=device), kinds=torch.tensor([0, 1, 0, 2], dtype=torch.int32, device=device),
        materials=torch.zeros(n, dtype=torch.int32, device=device), UIDs=torch.arange(n, device=device), UIDcounter=n)


def test_stepMemoryIsSharedByStagesAndRenewedBetweenSteps():
    from warpSPH.systems.weaklyCompressible import WeaklyCompressibleSystem
    system = WeaklyCompressibleSystem(state=_state())
    config = Sun2017DeltaSPHConfig()
    system.initialize(.1, schemeConfig=config)
    first = system.initializeNewState()
    assert first.deltaStages is system.deltaStages
    first.deltaStages.surfaceStart = torch.tensor([1, 0, 0, 1])
    old = system.deltaStages
    nextStep = system.initializeNewState()
    nextStep.initialize(.2, schemeConfig=config)
    assert nextStep.deltaStages is not old
    assert nextStep.deltaStages.diffusion is None
    assert nextStep.deltaStages.previousSurfaceStart is old.surfaceStart
    other = WeaklyCompressibleSystem(state=_state())
    other.initialize(.1, schemeConfig=config)
    assert other.deltaStages is not nextStep.deltaStages
    assert other.deltaStages.previousSurfaceStart is None


def test_ownedFrozenDiffusionCannotConsumeAnotherSystemsOrLegacyCache(monkeypatch):
    import warpSPH.modules.deltaSPH.stages as stages
    calls = []
    monkeypatch.setattr(stages, 'computeDensityDiffusion', lambda state, *args: calls.append(float(state.densities[0])) or state.densities * 2)
    monkeypatch.setattr(stages, 'computeVelocityDiffusion', lambda state, *args, **kwargs: state.velocities * 3)
    config = NS(freezeDiffusionAcrossStages=True, diffusionParams=NS(densityDiffusionTerm=DensityDiffusionScheme.densityOnly),
                _frozenDiffusionCache=('legacy', 'values'))
    a, b = _state(), _state()
    b.densities *= 5
    ownerA, ownerB = DeltaStages(), DeltaStages()
    def evaluate(state, index, owner):
        return diffusionRates(state, None, config, None, None, None, index, owner=owner)
    expectedA = evaluate(a, 0, ownerA)
    expectedB = evaluate(b, 0, ownerB)
    a.densities *= 9
    for expected, actual in ((expectedA, evaluate(a, 1, ownerA)), (expectedB, evaluate(b, 2, ownerB))):
        assert actual is expected or all(x is y for x, y in zip(actual, expected))
    # An owned later stage without its own stage-0 result evaluates freshly.
    fresh = evaluate(a, 1, DeltaStages())
    torch.testing.assert_close(fresh[0], a.densities * 2)
    assert config._frozenDiffusionCache == ('legacy', 'values')
    assert calls == [1., 5., 9.]
    evaluate(a, None, ownerA)
    evaluate(a, None, ownerA)
    assert calls[-2:] == [9., 9.]  # preserve unindexed-integrator semantics


def test_ownedSurfaceStagesUseLifecycleRatherThanConfigOrFloatingTime():
    config = NS(freezeSurfaceMaskAcrossStages=True, surfaceMaskDiagnostics=True,
                _surfaceMaskStepStart=torch.tensor([9, 9]))
    a, b = DeltaStages(), DeltaStages()
    def call(mask, owner, t):
        state = NS(surfaceIndicators=torch.tensor(mask, dtype=torch.int32))
        surfaceMaskAcrossStages(state, .031, config, None, t, owner=owner)
        return state.surfaceIndicators
    call([1, 0], a, 100000.)
    call([0, 1], b, 100000.)
    assert call([0, 1], a, 100000.015).tolist() == [1, 0]
    assert a.surfaceStats['maskFlipsIntraStep'] == 2
    assert call([1, 0], b, 100000.016).tolist() == [0, 1]
    assert config._surfaceMaskStepStart.tolist() == [9, 9]
    assert a.surfaceStart.tolist() == [1, 0]


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('keepBand', [False, True])
@pytest.mark.parametrize('dim', [2, 3])
def test_assemblyAndMaskMatchOrderedTorchIncludingNonfiniteBoundaryValues(device, keepBand, dim, monkeypatch):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    torch.manual_seed(123)
    state = _state(device)
    state.velocities = torch.randn((4, dim), dtype=state.velocities.dtype, device=device)
    if device == 'cuda':
        # Exercise primitive arithmetic independently of capture-only routing.
        import warpSPH.modules.deltaSPH.updates as updates
        monkeypatch.setattr(updates, '_native', lambda *tensors: True)
    rates = [torch.randn_like(state.velocities) for _ in range(5)]
    rates[0][1, 0] = float('nan')
    continuity, diffusion = torch.randn_like(state.densities), torch.randn_like(state.densities)
    update = assembleUpdate(state, *rates, continuity, diffusion)
    expectedV = rates[0] + rates[1] + rates[2] + rates[3] + rates[4]
    torch.testing.assert_close(update.dvdt, expectedV, atol=0, rtol=0, equal_nan=True)
    torch.testing.assert_close(update.drhodt, continuity + diffusion, atol=0, rtol=0)
    assert not update.passive.any()
    assert update.drhodt_kin is continuity
    # Boundary hooks may alter the assembled rates before the final mask.
    update.dvdt[2, 0] = 123.
    expectedV[2, 0] = 123.
    keep = torch.tensor([[1.], [0.], [.25], [1.]], dtype=state.velocities.dtype, device=device) if keepBand else None
    if keep is not None:
        expectedV = expectedV * keep
    fluid = (state.kinds == 0).unsqueeze(-1)
    maskFluidUpdate(update, state.kinds, keep)
    for actual, expected in ((update.dxdt, torch.where(fluid, state.velocities, 0.)),
                             (update.dvdt, torch.where(fluid, expectedV, 0.)),
                             (update.drhodt, torch.where(fluid[:, 0], continuity + diffusion, 0.)),
                             (update.drhodt_kin, torch.where(fluid[:, 0], continuity, 0.))):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0, equal_nan=True)


def test_assemblyAndMaskKeepReverseAndForwardDerivatives():
    import torch.autograd.forward_ad as fwAD
    state = _state()
    pressure = torch.randn_like(state.velocities, requires_grad=True)
    zeros = torch.zeros_like(pressure)
    rho = state.densities
    keep = torch.tensor([[1.], [0.], [.25], [1.]], dtype=pressure.dtype)
    def evaluate(p):
        update = assembleUpdate(state, p, zeros, zeros, zeros, zeros, rho, rho)
        return maskFluidUpdate(update, state.kinds, keep).dvdt
    value = evaluate(pressure)
    gradient, = torch.autograd.grad(value.sum(), pressure)
    expected = (state.kinds == 0)[:, None] * keep.expand_as(pressure)
    torch.testing.assert_close(gradient, expected)
    with fwAD.dual_level():
        dual = evaluate(fwAD.make_dual(pressure.detach(), torch.ones_like(pressure)))
        torch.testing.assert_close(fwAD.unpack_dual(dual).tangent, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA stream/capture')
@pytest.mark.parametrize('capture', [False, True])
def test_nativeUpdateKernelsRespectCallerStream(capture, monkeypatch):
    from warpSPHCore.autograd.stream import torchWarpStream, torchWarpCapture
    state = _state('cuda')
    rates = [torch.ones_like(state.velocities) for _ in range(5)]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    device = wp.device_from_torch(state.velocities.device)
    unrelated = wp.Stream(device)
    original = wp.launch
    def launch(*args, **kwargs):
        assert wp.get_stream(device).cuda_stream == torch.cuda.current_stream().cuda_stream
        return original(*args, **kwargs)
    monkeypatch.setattr(wp, 'launch', launch)
    def evaluate():
        return maskFluidUpdate(assembleUpdate(state, *rates, state.densities, state.densities), state.kinds)
    with wp.ScopedStream(unrelated, sync_enter=False, sync_exit=False):
        with torch.cuda.stream(stream):
            evaluate()
        stream.synchronize()
        with torch.cuda.stream(stream):
            if capture:
                graph = torch.cuda.CUDAGraph()
                with torchWarpStream(state.velocities.device), torchWarpCapture(graph):
                    result = evaluate()
                rates[0].add_(1.)
                graph.replay()
            else:
                result = evaluate()
        stream.synchronize()
        assert wp.get_stream(device) is unrelated
    expected = torch.where((state.kinds == 0)[:, None], torch.full_like(state.velocities, 6. if capture else 5.), 0.)
    torch.testing.assert_close(result.dvdt, expected)


@pytest.mark.parametrize('mode', ['impulse', 'finalize', 'derivative', 'off'])
def test_contactCorrectionRetainsItsPlacementAndFluidMask(mode):
    from warpSPH.modules.mdbc.corrections import applyNoPenetration
    state = _state()
    state.positions.fill_(3.)
    before = NS(state=NS(positions=torch.ones_like(state.positions), velocities=torch.ones_like(state.velocities)))
    shift = torch.tensor([[.1, 0.], [1., 1.], [0., -.2], [2., 2.]], dtype=state.velocities.dtype)
    system = NS(state=state, _noPenShift=lambda *args: shift)
    v, x, rho = state.velocities.clone(), state.positions.clone(), state.densities.clone()
    result = applyNoPenetration(system, before, .5, None, NS(mdbcNoPenShiftMode=mode))
    active = (shift != 0) & (state.kinds == 0)[:, None]
    if mode == 'impulse':
        torch.testing.assert_close(state.velocities, torch.where(active, v + shift, v))
        torch.testing.assert_close(state.positions, x)
    elif mode == 'finalize':
        vn = torch.where(active, before.state.velocities + shift, v)
        torch.testing.assert_close(state.velocities, vn)
        torch.testing.assert_close(state.positions, torch.where(active, before.state.positions + .5 * vn, x))
    else:
        assert result is None
        torch.testing.assert_close(state.velocities, v)
        torch.testing.assert_close(state.positions, x)
    torch.testing.assert_close(state.densities, rho)
