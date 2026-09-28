"""`fourtakas2019` density diffusion must vanish on an exactly hydrostatic
field -- including rows whose stencil is truncated (a free block's edges).
The hydrostatic correction's sign was flipped until 2026-09-28: it vanished
in the bulk (the Laplacian of a linear field) and doubled the error at every
edge, which degraded still water next to every wall (OPEN_PROBLEMS.md §5)."""

import pytest
import torch


@pytest.mark.parametrize('gravityDirection', [(0.0, -1.0), (1.0, 0.0)])
def test_fourtakasVanishesOnAHydrostaticField(gravityDirection):
    import warp as wp
    wp.init()
    from warpSPHCore import (DomainDescription, KernelFunctions, OperationProperties, ParticleState,
                             WarpOperation, radiusSearchCompactHashMap)
    from warpSPHCore.enumTypes import OperationDirection, SupportScheme
    from warpSPH.enumTypes import DensityDiffusionScheme
    from warpSPH.modules.deltaSPH.wp_densityDelta import computeDensityDiffusionDeltaSPH
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dt = torch.float32
    n, dx = 32, 1.0 / 32
    dom = DomainDescription(min=torch.tensor([-2., -2.], device=dev, dtype=dt),
                            max=torch.tensor([2., 2.], device=dev, dtype=dt),
                            periodic=torch.tensor([False, False], device=dev), dim=2)
    g = torch.arange(n, device=dev, dtype=dt) * dx
    X, Y = torch.meshgrid(g, g, indexing='ij')
    pos = torch.stack([X.flatten(), Y.flatten()], -1)
    N = pos.shape[0]
    rho0, c0, grav = 1.0, 10.0, 9.81
    gvec = torch.tensor(gravityDirection, device=dev, dtype=dt) * grav
    # rho^H = rho0 (1 + (g . x - min) / c0^2): denser downstream of gravity
    depth = pos @ gvec
    rho = rho0 * (1.0 + (depth - depth.min()) / c0 ** 2)
    p = ParticleState(positions=pos, supports=torch.full((N,), 4 * dx, device=dev, dtype=dt),
                      masses=torch.full((N,), dx * dx, device=dev, dtype=dt), densities=rho,
                      kinds=torch.zeros(N, dtype=torch.int32, device=dev))
    adj = radiusSearchCompactHashMap(p, dom, mode=SupportScheme.SuperSymmetric)
    props = OperationProperties(kernel=KernelFunctions.Wendland2, operation=WarpOperation.Laplacian,
                                supportMode=SupportScheme.SuperSymmetric,
                                operationMode=OperationDirection.AllToAll)
    four = computeDensityDiffusionDeltaSPH(p, props, dom, densityScheme=DensityDiffusionScheme.fourtakas2019,
                                           queryField=rho, adjacency=adj, rho0=rho0, c0=c0, gravity=gvec)
    plain = computeDensityDiffusionDeltaSPH(p, props, dom, densityScheme=DensityDiffusionScheme.densityOnly,
                                            queryField=rho, adjacency=adj, rho0=rho0, c0=c0, gravity=gvec)
    # the plain difference does not vanish at the edges; Fourtakas must, to rounding
    assert float(plain.abs().max()) > 1.0
    assert float(four.abs().max()) < 1e-3 * float(plain.abs().max())
