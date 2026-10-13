"""The compatible-energy update folds each RK stage's pair work into one running sum
(`modules/compSPH/pairWork.py`); when the stages ran on different pair lists the integrator repeats the step
with the known half-step velocity. Both routes must give the same state (to round-off), conserve total energy,
and leave no per-stage pair arrays behind."""

import pytest
import torch

import warpSPH.modules.compSPH.pairWork as pairWork
import warpSPH.schemes.compSPH as compSPHScheme
import warpSPH.schemes.crkSPH as crkSPHScheme
from warpSPH.cases import importAll
from warpSPH.runner import run
from warpSPH.runner.case import getCase

importAll()


def _run(scheme, steps=30, forceRerun=False, monkeypatch=None):
    reruns = [0]
    if forceRerun:
        original = pairWork.accumulateStage

        def spoiled(pw, stageIndex, adjacency, state, energyScheme):
            original(pw, stageIndex, adjacency, state, energyScheme)
            if stageIndex == 1 and pw.halfVelocity is None:
                pw.consistent, pw.work = False, None      # as if the pair list had changed after stage 0
                reruns[0] += 1
        monkeypatch.setattr(compSPHScheme, 'accumulateStage', spoiled)
        monkeypatch.setattr(crkSPHScheme, 'accumulateStage', spoiled)
    result = run(getCase('sod2d'), quiet=True, progress=False, nSteps=steps, scheme=scheme)
    return result, reruns[0]


@pytest.mark.parametrize('scheme', ['CompSPH', 'CRKSPH'])
def test_rerunPassReproducesTheRunningSum(scheme, monkeypatch):
    plain, _ = _run(scheme)
    rerun, nRerun = _run(scheme, forceRerun=True, monkeypatch=monkeypatch)
    assert nRerun == 30                                   # every step took the second pass
    a, b = plain.state.state, rerun.state.state
    for name in ('positions', 'velocities', 'internalEnergies', 'densities', 'alphas'):
        x, y = getattr(a, name).double(), getattr(b, name).double()
        scale = x.abs().max().clamp_min(1e-30)
        assert float((x - y).abs().max() / scale) < 1e-4, name


@pytest.mark.parametrize('scheme', ['CompSPH', 'CRKSPH'])
def test_totalEnergyIsConserved(scheme):
    result, _ = _run(scheme, steps=60)
    e0, e1 = result.trajectory[0]['totalEnergy'], result.trajectory[-1]['totalEnergy']
    assert abs(e1 - e0) / abs(e0) < 1e-5


@pytest.mark.parametrize('scheme', ['CompSPH', 'CRKSPH'])
def test_noPerStagePairArraysAreRetained(scheme):
    result, _ = _run(scheme, steps=3)
    st = result.state.state
    assert st.ap_ij is None and st.av_ij is None and st.f_ij is None


@pytest.mark.parametrize('scheme', ['CompSPH', 'CRKSPH'])
def test_multiStageTableauDoesNotNeedASecondPass(scheme):
    """Heun carries two weighted stages: the pair list must not change between them (stage 0 rebuilds early
    when half the Verlet buffer is used, `pairWork.startStage`), so no step is repeated."""
    seconds = [0]
    original = pairWork.energyIncrement

    def counting(pw, *args, **kwargs):
        out = original(pw, *args, **kwargs)
        seconds[0] += int(out[1] is not None and pw.halfVelocity is None)
        return out

    import warpSPH.systems.compSPH as systems
    mp = pytest.MonkeyPatch()
    mp.setattr(systems, 'energyIncrement', counting)
    try:
        run(getCase('sod2d'), quiet=True, progress=False, nSteps=60, scheme=scheme, integrationScheme='heunsMethod')
    finally:
        mp.undo()
    assert seconds[0] == 0


@pytest.mark.parametrize('supportScheme', ['NoScheme', 'Monaghan', 'Owen'])
@pytest.mark.parametrize('scheme', ['CompSPH', 'CRKSPH'])
def test_everySupportSchemeLeavesAValidNeighbourList(scheme, supportScheme):
    """The scheme relies on `evaluateOptimalSupport(validateOnce=True)` returning a list that is valid for the
    returned supports and the current positions, whichever support scheme made them (NoScheme used to hand back
    whatever list it was given: a stale one, or none at the first stage)."""
    result = run(getCase('sod2d'), quiet=True, progress=False, nSteps=60, scheme=scheme,
                 params=dict(adaptiveSupportScheme=supportScheme))
    assert result.nSteps == 60 and not result.diverged
    e0, e1 = result.trajectory[0]['totalEnergy'], result.trajectory[-1]['totalEnergy']
    assert abs(e1 - e0) / abs(e0) < 1e-4
