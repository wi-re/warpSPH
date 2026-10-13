"""`_distributionCore` reads the capacity-padded edge arrays and masks the -1 padding, self pairs and fluid-wall pairs
itself: its result must be bitwise the one from the already gathered fluid-fluid pairs."""

import torch

from warpSPH.cases.common.weaklyCompressible import _distributionCore


def test_paddedPairsGiveTheGatheredFluidPairsResult():
    g = torch.Generator().manual_seed(3)
    n = 60
    pos = torch.rand(n, 2, generator=g)
    kinds = (torch.arange(n) >= 45).to(torch.int32)             # the last 15 are wall particles
    fluid = kinds == 0
    fluidIndex = torch.nonzero(fluid).flatten()
    densities = 1.0 + 0.01 * torch.rand(n, generator=g)
    i = torch.arange(n).repeat_interleave(7)
    j = (i + torch.randint(0, n, (n * 7,), generator=g)) % n    # includes self pairs and fluid-wall pairs
    q = torch.tensor([0.5, 0.01])

    keep = (i != j) & fluid[i] & fluid[j]
    exact = _distributionCore(pos, i[keep], j[keep], fluid, fluidIndex, densities, None, None, q)

    pad = torch.full((37,), -1, dtype=i.dtype)
    padded = _distributionCore(pos, torch.cat([i, pad]), torch.cat([j, pad]), fluid, fluidIndex, densities, None, None, q)
    assert torch.equal(exact, padded)
