"""`modules/compressibleWall/supportClamp.py`: the wall cases' support volume clamp, as its own step."""

from types import SimpleNamespace

import pytest
import torch

from warpSPHCore import ParticleState, SupportScheme, buildVerletList
from warpSPHCore.enumTypes import KernelFunctions
from warpSPHCore.util import generateNeighborTestData
from warpSPH.modules.compressibleWall import clampSupportsToVolume

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')


def _case(mode, oversize=2.0):
    dev = torch.device('cuda')
    pos, sup, n, dom, dx = generateNeighborTestData(30, 30, 2, True, dev)
    state = ParticleState(positions=pos, supports=sup * oversize, masses=torch.ones(n, device=dev) * dx * dx,
                          densities=torch.ones(n, device=dev), kinds=torch.zeros(n, dtype=torch.int32, device=dev))
    config = SimpleNamespace(supportVolumeClamp=mode, kernel=KernelFunctions.B7, domain=dom, targetNeighbors=30)
    adjacency = buildVerletList(state, dom, verletScale=1.0, supportMode=SupportScheme.Gather)
    return state, config, adjacency


def test_offAndWallsWithoutWallRowsLeaveTheSupportsAlone():
    for mode in ('off', 'walls'):
        state, config, adjacency = _case(mode)
        assert clampSupportsToVolume(state, config, adjacency, wall=None) is state.supports


def test_alwaysClampsOversizedSupportsAndNeverEnlarges():
    state, config, adjacency = _case('always', oversize=2.0)
    clamped = clampSupportsToVolume(state, config, adjacency, wall=None)
    assert bool((clamped <= state.supports).all()) and bool((clamped < state.supports).any())
    small, config2, adjacency2 = _case('always', oversize=0.8)       # under-sized: the clamp must not grow them
    assert bool((clampSupportsToVolume(small, config2, adjacency2, wall=None) <= small.supports).all())


def test_wallsModeAppliesWhenAWallIsPresent():
    state, config, adjacency = _case('walls', oversize=2.0)
    clamped = clampSupportsToVolume(state, config, adjacency, wall=object())
    assert bool((clamped < state.supports).any())


def test_returnDensityIsTheSummationDensityAtTheClampedSupportsAndRestoresTheState():
    state, config, adjacency = _case('always', oversize=2.0)
    original = state.supports
    clamped, density = clampSupportsToVolume(state, config, adjacency, wall=None, returnDensity=True)
    assert state.supports is original and density.shape == clamped.shape
    assert bool(torch.isfinite(density).all())


def test_unknownModeIsRejected():
    state, config, adjacency = _case('sometimes')
    with pytest.raises(ValueError):
        clampSupportsToVolume(state, config, adjacency, wall=None)
