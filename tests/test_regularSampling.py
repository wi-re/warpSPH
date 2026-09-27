"""Regular-lattice sampler (`sample/regular.py`): particle counts must follow
`nx` exactly, independent of floating-point round-off in `span / (span/nx)`.

Regression: with the default (dx=None) path, `ceil(l/dx)` turned round-off
like 48.00000000000001 into a spurious extra layer -- a periodic 2*pi box
sampled 49^2 particles at nx=48 and 97^2 at nx=96 (TGV).
"""

import math

import pytest
import torch

from warpSPH.sample import sampleRegularParticles
from warpSPH.utils.domain import buildDomainDescription


# The round-off that triggered the bug depends on where the domain tensors
# live: on CPU float64 2*pi/48 divides back to exactly 48.0, on CUDA (where
# the cases build their domain) it gives 48.00000000000001. Test both.
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _box(L, dim, periodic, dtype, device="cpu"):
    domain = buildDomainDescription(1.0, dim, periodic=periodic,
                                    device=device, dtype=dtype)
    domain.min = torch.zeros(dim, dtype=dtype, device=device)
    domain.max = torch.ones(dim, dtype=dtype, device=device) * L
    return domain


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("L", [2 * math.pi, 1.0, 2.0, 3.0])
@pytest.mark.parametrize("nx", [32, 48, 64, 96, 100, 128, 200])
def test_periodic_count_and_spacing_follow_nx(nx, L, dtype, device):
    domain = _box(L, 2, True, dtype, device)
    ps = sampleRegularParticles(nx, domain, 20)
    assert ps.positions.shape[0] == nx * nx
    xs = torch.unique(ps.positions[:, 0])
    assert xs.numel() == nx
    assert float(xs[1] - xs[0]) == pytest.approx(L / nx, rel=1e-5)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("nx", [16, 33, 48, 97])
def test_open_axis_count_unchanged(nx, device):
    # the non-periodic default path's point count must be untouched by the
    # round-off guard (exact-integer l/dx -> same ceil as before)
    L = 2 * math.pi
    domain = _box(L, 1, False, torch.float64, device)
    ps = sampleRegularParticles(nx, domain, 8)
    dx = L / (nx - 1)
    assert ps.positions.shape[0] == math.ceil(round(L / dx, 9))
