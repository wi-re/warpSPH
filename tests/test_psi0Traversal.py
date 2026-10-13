"""Psi0 / omega give the same numbers whether they traverse a neighbour list, a hash-map grid or build their own.

`computePsi0_Func_Adjacency` used to take the dim-th root of each CELL's partial sum on a grid traversal and add the
roots (the root is not additive): the grid / no-adjacency result was 1.6-1.9x the list result, while the production
Owen solver (list path) was unaffected."""

import pytest
import torch

from warpSPHCore import OperationProperties, ParticleState, SupportScheme, buildVerletList
from warpSPHCore.enumTypes import KernelFunctions
from warpSPHCore.radiusSearch.compactHash import buildCompactHashMap
from warpSPHCore.util import generateNeighborTestData
from warpSPH.modules.adaptiveSupport.wp_omega import computeOmegaWarp as computeOmega
from warpSPH.modules.adaptiveSupport.wp_psi0 import computePsi0Warp

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('periodic', [False, True])
def test_psi0AndOmegaAgreeAcrossTraversals(dim, periodic):
    dev = torch.device('cuda')
    pos, sup, n, dom, dx = generateNeighborTestData(14 if dim == 3 else 30, 30, dim, periodic, dev)
    g = torch.Generator().manual_seed(1)
    pos = pos + 0.2 * dx * (torch.rand(pos.shape, generator=g) - 0.5).to(dev)
    st = ParticleState(positions=pos, supports=sup, masses=torch.ones(n, device=dev) * dx ** dim,
                       densities=torch.ones(n, device=dev), kinds=torch.zeros(n, dtype=torch.int32, device=dev))
    props = OperationProperties(kernel=KernelFunctions.B7, supportMode=SupportScheme.Gather)
    adj = buildVerletList(st, dom, verletScale=1.0, supportMode=SupportScheme.Gather)
    grid = buildCompactHashMap(pos, pos, sup, sup, dom.periodic, dom, SupportScheme.Gather, n + 1)
    ref = computePsi0Warp(st, props, dom, adjacency=adj)
    for other in (computePsi0Warp(st, props, dom, adjacency=grid), computePsi0Warp(st, props, dom, adjacency=None)):
        for a, b in zip(ref, other):
            assert torch.allclose(a, b, rtol=2e-4, atol=1e-6)
    refOmega = computeOmega(st, props, dom, adjacency=adj)
    assert torch.allclose(refOmega, computeOmega(st, props, dom, adjacency=grid), rtol=2e-4, atol=1e-6)
