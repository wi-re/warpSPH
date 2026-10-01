"""CRKSPH runs are bitwise reproducible.

`compSPH_deltaU_multistep` used to sum its per-pair energy terms with
`scatter_add_` (CUDA atomics: float summation order changes run to run), which
made every CRKSPH result -- Gresho L1(v_phi) +-5 % -- unrepeatable. It now uses
`segment_sum` over the row-sorted adjacency."""

import contextlib
import io

import pytest
import torch

from warpSPH.math.scatter import scatter_sum, segment_sum
from warpSPH.runner import run
from warpSPH.cases.greshoVortex import greshoVortexCase


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a GPU')
def test_segment_sum_matches_scatter_sum():
    n = 500
    counts = torch.randint(0, 40, (n,), device='cuda')
    i = torch.repeat_interleave(torch.arange(n, device='cuda'), counts)
    src = torch.randn(i.numel(), dtype=torch.float64, device='cuda')
    ref = scatter_sum(src, i, dim=0, dim_size=n)
    torch.testing.assert_close(segment_sum(src, counts), ref, rtol=1e-12, atol=1e-12)
    assert torch.equal(segment_sum(src, counts), segment_sum(src, counts))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a GPU')
def test_crk_gresho_bitwise_reproducible():
    def go():
        with contextlib.redirect_stdout(io.StringIO()):
            r = run(greshoVortexCase, progress=False, quiet=True, nx=32, tLimit=0.05,
                    params=dict(viscositySwitch='NoneSwitch'))
        s = r.state.state
        return [t.detach().clone() for t in (s.velocities, s.internalEnergies, s.densities)]
    a, b = go(), go()
    for x, y in zip(a, b):
        assert torch.equal(x, y)
