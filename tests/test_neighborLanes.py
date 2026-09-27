"""Multi-lane neighbour loops (warpSPHCore `autograd/lanes.py`) through whole
scheme steps: a short run with the tiled kernels must match the
thread-per-particle kernels to float summation order.

This is the check that caught `computePsi0` taking a dim-th root *inside* its
per-range function (a per-lane root of a partial sum is not the root of the
sum) -- a bug every kernel-level forward test missed, because it only shows
once a scheme actually uses that operator."""

import importlib

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')


def _run(caseModule, caseName, lanes, **kw):
    from warpSPHCore import setNeighborLanes
    from warpSPH.runner import run
    from warpSPH.runner.case import getCase
    importlib.import_module(f'warpSPH.cases.{caseModule}')
    setNeighborLanes(lanes)
    try:
        r = run(getCase(caseName), quiet=True, store=False, progress=False, plot=False, **kw)
    finally:
        setNeighborLanes(None)
    st = r.state.state
    return {k: getattr(st, k).detach().double() for k in ('positions', 'velocities', 'densities')}


@cuda
# Bounds: CRKSPH stays at float rounding. Weakly-compressible delta-SPH
# amplifies it through the stiff EOS (rho0 c0^2 turns a 1e-7 density
# difference into ~1e-3 relative acceleration difference per evaluation), so
# 10 steps drift ~1e-4 -- still two orders below the psi0 bug (5e-3 after 2
# steps, 19 % after 20).
@pytest.mark.parametrize('caseModule,caseName,kw,bound', [
    ('greshoVortex', 'gresho', dict(nx=48, nSteps=10), 1e-4),        # CRKSPH: moments, CRK density, accel, dudt, psi0
    ('dambreak', 'dambreak', dict(nx=32, nSteps=10, scheme='sun2017DeltaSPH'), 1e-3),  # delta-SPH + mDBC
])
def test_tiledKernelsMatchThreadPerParticle(caseModule, caseName, kw, bound):
    ref = _run(caseModule, caseName, 1, **kw)
    tiled = _run(caseModule, caseName, 32, **kw)
    for k in ref:
        rel = float((ref[k] - tiled[k]).abs().max() / ref[k].abs().max().clamp_min(1e-30))
        assert rel < bound, f'{caseName}.{k}: lanes 1 vs 32 differ by {rel:.3e} (expected float-rounding level)'
