"""OPEN_PROBLEMS.md §1 step 1: the Antuono switch's surface mask across the
stages of one step (`schemes/deltaSPH.py:_surfaceMaskAcrossStages`). CPU only:
the stage bookkeeping, not the physics."""

from types import SimpleNamespace

import torch


def _call(sc, t, dt, mask, stageIndex=None):
    from warpSPH.schemes.deltaSPH import _surfaceMaskAcrossStages
    state = SimpleNamespace(t=t, surfaceIndicators=torch.tensor(mask, dtype=torch.int32))
    _surfaceMaskAcrossStages(state, dt, sc, stageIndex)
    return state.surfaceIndicators.tolist()


def test_symplecticEulerStagesAreFoundFromTheTime():
    """Symplectic Euler passes no stageIndex: k0 at t^n, k1 at t^n + dt/2,
    both with the stage dt = dt/2."""
    sc = SimpleNamespace(surfaceMaskDiagnostics=True, freezeSurfaceMaskAcrossStages=False)
    _call(sc, 0.0, 0.05, [1, 0, 0])                 # step 1, stage 0
    _call(sc, 0.05, 0.05, [1, 1, 0])                # step 1, stage 1: one row flipped
    assert sc._surfaceMaskStats['maskFlipsIntraStep'] == 1
    _call(sc, 0.1, 0.05, [0, 1, 0])                 # step 2, stage 0
    assert sc._surfaceMaskStats['maskFlipsIntraStep'] == 0
    assert sc._surfaceMaskStats['maskFlipsStepToStep'] == 2   # vs step 1's stage 0


def test_freezeReusesTheFirstStageMask():
    sc = SimpleNamespace(surfaceMaskDiagnostics=False, freezeSurfaceMaskAcrossStages=True)
    assert _call(sc, 0.0, 0.05, [1, 0, 0]) == [1, 0, 0]
    assert _call(sc, 0.05, 0.05, [0, 1, 1]) == [1, 0, 0]      # frozen
    assert _call(sc, 0.1, 0.05, [0, 1, 1]) == [0, 1, 1]       # next step: fresh


def test_stageIndexWinsWhereTheIntegratorPassesOne():
    sc = SimpleNamespace(surfaceMaskDiagnostics=False, freezeSurfaceMaskAcrossStages=True)
    _call(sc, 0.0, 0.1, [1, 0], stageIndex=0)
    assert _call(sc, 0.05, 0.05, [0, 1], stageIndex=1) == [1, 0]
    assert _call(sc, 0.1, 0.05, [0, 1], stageIndex=2) == [1, 0]


def test_offByDefault():
    sc = SimpleNamespace()
    assert _call(sc, 0.0, 0.05, [1, 0]) == [1, 0]
    assert not hasattr(sc, '_surfaceMaskStepStart')
