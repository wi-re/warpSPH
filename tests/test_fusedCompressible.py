"""Fused traversals must preserve the independent raw heating predictors."""
from types import SimpleNamespace
import pytest
import torch

from warpSPHCore import (ParticleState, DomainDescription, OperationProperties,
    SupportScheme, KernelFunctions, CRKState, buildVerletList, get_torch_precision,
    get_torch_index_dtype, get_torch_offset_dtype)
from warpSPHCore.autograd.lanes import setNeighborLanes
from warpSPH.configurations import CRKSPHConfig, CompSPHConfig
from warpSPH.configurations.crkSPH import resolveCRKLimiter
from warpSPH.modules.compSPH.workReduction import reducePairWork


@pytest.mark.parametrize('dim', [1, 2, 3])
@pytest.mark.parametrize('lanes', [1, 32])
@pytest.mark.parametrize('scheme', ['CRKSPH', 'CompSPH'])
def test_fusedHeatingMatchesStandalone(dim, lanes, scheme):
    dtype = get_torch_precision()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gen = torch.Generator().manual_seed(123)
    tensor = lambda shape: torch.rand(shape, generator=gen, dtype=dtype).to(device)
    n = 12
    state = ParticleState(positions=tensor((n, dim)), supports=.9 + tensor((n,)),
        masses=.1 + tensor((n,)), densities=.5 + tensor((n,)),
        kinds=torch.zeros(n, dtype=torch.int32, device=device))
    state.velocities = tensor((n, dim)) - .5
    state.internalEnergies = 1. + tensor((n,))
    state.pressures = .5 + tensor((n,))
    state.soundspeeds = 1. + tensor((n,))
    state.alphas = .1 + tensor((n,))
    domain = DomainDescription(torch.zeros(dim, device=device, dtype=dtype),
        torch.ones(dim, device=device, dtype=dtype), torch.ones(dim, device=device, dtype=torch.bool), dim)
    adjacency = buildVerletList(state, domain, supportMode=SupportScheme.SuperSymmetric)
    kwargs = dict(queryParticles=state, domain=domain, adjacency=adjacency,
        operationProperties=OperationProperties(kernel=KernelFunctions.B7, supportMode=SupportScheme.KernelMeanSymmetric),
        queryVelocities=state.velocities, queryEnergies=state.internalEnergies,
        queryPressures=state.pressures, queryCs=state.soundspeeds, queryAlphas=state.alphas)
    if scheme == 'CRKSPH':
        from warpSPH.modules.crk.accel import computeCrkSPHAccelWarp as accel
        from warpSPH.modules.crk.dudt import computeCrkSPHdudtWarp as heating
        config = CRKSPHConfig()
        kwargs.update(queryVolumes=.1 + tensor((n,)), queryVelocityTensor=tensor((n, dim, dim)),
            crkState=CRKState(A=1. + tensor((n,)), B=.1 * tensor((n, dim)),
                gradA=.1 * tensor((n, dim)), gradB=.1 * tensor((n, dim, dim))),
            crkViscosityParams=resolveCRKLimiter(config.crkViscosityParams, 4.))
    else:
        from warpSPH.modules.compSPH.accel import computeCompSPHAccelWarp as accel
        from warpSPH.modules.compSPH.dudt import computeCompSPHdudtWarp as heating
        config = CompSPHConfig()
    kwargs['conductivityParams'] = config.diffusionParams
    setNeighborLanes(lanes)
    try:
        for variant in (False, True):
            if scheme == 'CRKSPH':
                kwargs['crkViscosityParams'].oneSidedWork = variant
                kwargs['crkViscosityParams'].meanVolumeWeights = variant
            ordinary = accel(**kwargs)
            fused = accel(**kwargs, returnDudt=True)
            tolerance = 500 * torch.finfo(dtype).eps
            for x, y in zip(ordinary, fused[:3]):
                # Pair outputs have an intentionally unwritten padded tail.
                count = n if x.shape[0] == n else adjacency.numEdges
                torch.testing.assert_close(x[:count], y[:count], rtol=tolerance, atol=tolerance)
            raw = dict(kwargs)
            if scheme == 'CompSPH':
                raw['operationProperties'] = OperationProperties(kernel=KernelFunctions.B7, supportMode=SupportScheme.Gather)
            torch.testing.assert_close(fused[3], heating(**raw), rtol=tolerance, atol=tolerance)
            if dim == 2 and lanes == 1 and not variant:
                state.velocities.requires_grad_(True)
                fused_gradient = torch.autograd.grad(accel(**kwargs, returnDudt=True)[3].sum(), state.velocities)[0]
                reference_gradient = torch.autograd.grad(heating(**raw).sum(), state.velocities)[0]
                torch.testing.assert_close(fused_gradient, reference_gradient,
                                           rtol=10*tolerance, atol=10*tolerance)
                state.velocities.requires_grad_(False)
    finally:
        setNeighborLanes(None)


@pytest.mark.parametrize('dim', [1, 2, 3])
@pytest.mark.parametrize('tape', [False, True])
def test_pairWorkReductionIgnoresPaddingAndPreservesGradients(dim, tape):
    dtype = get_torch_precision()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    counts = torch.tensor([2, 0, 1, 2], device=device, dtype=torch.int32)
    adj = SimpleNamespace(numNeighbors=counts, numEdges=5,
        edgeOffsets=torch.tensor([0, 2, 2, 3], device=device, dtype=get_torch_offset_dtype()),
        paddedI=torch.tensor([0, 0, 2, 3, 3, -1, -1], device=device, dtype=get_torch_index_dtype()),
        paddedJ=torch.tensor([0, 3, 1, 2, 3, -1, -1], device=device, dtype=get_torch_index_dtype()))
    v = torch.randn(4, dim, device=device, dtype=dtype, requires_grad=tape)
    work = torch.randn(7, dim, device=device, dtype=dtype)
    work[5:] = float('nan')
    work.requires_grad_(tape)
    i, j = adj.paddedI[:5].long(), adj.paddedJ[:5].long()
    reference = torch.segment_reduce(((v[j] - v[i]) * work[:5]).sum(-1), 'sum', lengths=counts)
    actual = reducePairWork(adj, v, work)
    torch.testing.assert_close(actual, reference)
    if tape:
        expected = torch.autograd.grad(reference.sum(), (v, work), retain_graph=True)
        actual_grad = torch.autograd.grad(actual.sum(), (v, work))
        for x, y in zip(expected, actual_grad):
            torch.testing.assert_close(x, y)


@pytest.mark.parametrize('tape', [False, True])
def test_referenceEntropyReductionPreservesThresholds(tape):
    from warpSPH.modules.compSPH.workReduction import reduceReferenceEntropyWork
    dtype = get_torch_precision()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n = 6
    counts = torch.full((n,), n, device=device, dtype=torch.int32)
    adj = SimpleNamespace(numNeighbors=counts, numEdges=n*n,
        edgeOffsets=torch.arange(n, device=device, dtype=get_torch_offset_dtype()) * n,
        paddedI=torch.arange(n, device=device, dtype=get_torch_index_dtype()).repeat_interleave(n),
        paddedJ=torch.arange(n, device=device, dtype=get_torch_index_dtype()).repeat(n))
    v = torch.tensor([[0.], [1.e-16], [1.e-15], [2.e-15], [-1.], [1.]], device=device, dtype=dtype, requires_grad=tape)
    work = torch.ones(n*n, 1, device=device, dtype=dtype, requires_grad=tape)
    entropy = torch.tensor([0., 0., 1.e-15, 2.e-15, .5, 2.], device=device, dtype=dtype, requires_grad=tape)
    i, j = adj.paddedI.long(), adj.paddedJ.long()
    dot = ((v[j] - v[i]) * work).sum(-1)
    si, sj = entropy[i], entropy[j]
    fraction = torch.where(dot > 0., sj.abs(), si.abs()) / (si.abs() + sj.abs()).clamp_min(torch.finfo(dtype).tiny)
    fraction = torch.where((dot.abs() > 1.e-15) & ((si - sj).abs() > 1.e-15), fraction, .5)
    expected = torch.segment_reduce(fraction * dot, 'sum', lengths=counts)
    torch.testing.assert_close(reduceReferenceEntropyWork(adj, v, work, entropy), expected)
