"""The delta-SPH CUDA-graph right-hand side follows a per-step change of the gravity direction (sloshingTank rolls gravity in the tank frame from a `postStep`
hook). It used to replace the direction tensor every step while the graph kept replaying the one it captured: graph and eager runs of the same tank differed
by up to 98 % in velocity (a latent bug since the graph path was added; it also made the results depend on the allocator layout)."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA graphs')

from warpSPH.cases import importAll
from warpSPH.runner import run
from warpSPH.runner.case import getCase
from warpSPH.schemes.weaklyCompressible.graphRoute import liveConfigKey

importAll()


def _final(cudaGraph, steps):
    r = run(getCase('sloshingTank'), quiet=True, progress=False, nSteps=steps, scheme='deltaSPH', plot=False, video=False, cudaGraph=cudaGraph)
    s = r.state.state
    return r, {n: getattr(s, n).detach().cpu() for n in ('positions', 'velocities', 'densities')}


def test_graphedSloshingTankIsTheEagerTank():
    eager = _final(False, 12)[1]
    r, graphed = _final(True, 12)
    graph = r.ctx.schemeConfig._rhsGraph
    assert graph.disabled is None and graph.replays > 12 and graph.captures <= 3       # updated in place: no recapture per step
    for name, a in eager.items():
        assert torch.equal(a, graphed[name]), name


def test_liveConfigKeyIsTheDirectionAddress():
    class G: pass
    class C: gravityConfig = G()
    C.gravityConfig.direction = torch.tensor([0.0, -1.0], device='cuda')
    key = liveConfigKey(C)
    assert key == C.gravityConfig.direction.data_ptr()
    C.gravityConfig.direction.copy_(torch.tensor([1.0, 0.0], device='cuda'))          # in place: same key, no recapture
    assert liveConfigKey(C) == key
    C.gravityConfig.direction = torch.tensor([1.0, 0.0], device='cuda')               # replaced: a new address (new capture)
    assert liveConfigKey(C) != key
    C.gravityConfig.direction = [0.0, -1.0]                                            # a constant list: nothing to track
    assert liveConfigKey(C) is None
