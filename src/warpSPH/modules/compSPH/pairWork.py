"""The compatible-energy update (Frontiere et al. 2017, Sec. 3.3) without per-stage pair storage.

The internal-energy increment over a step is
``deltaU_i = sum_s k_s sum_j f_ij^s v_ji . (ap_ij^s + av_ij^s)``, ``v_ji`` the half-step velocity
``v0 + dt/2 sum_s k_s dvdt_s``. That velocity needs every stage, so a stage cannot reduce its own pair terms
to a per-particle number. But it is the same for every stage, so it factors out of the stage sum:

    deltaU_i = sum_j v_ji . W_ij,    W_ij = sum_s k_s f_ij^s (ap_ij^s + av_ij^s)

and each stage folds its pair vector into one running `(E, dim)` buffer (`accumulateStage`) instead of the
old per-stage `(adjacency, ap, av, f)`. `energyIncrement` closes the step.

The running sum is only meaningful on one pair list. A stage that ran on another list (the Verlet list was
rebuilt between the stages) makes `PairWork.consistent` false, and `energyIncrement` then asks for the whole
step to be repeated with the now known half-step velocity (`rerun`): in that second pass every stage adds
its own ``k_s sum_j f_ij v_ji . (ap+av)`` straight into a per-particle sum, on whatever list it ran on. The
stage trajectory does not depend on the energy increment (interim energies come from `dudt`), so the second
pass reproduces the first one's stages. `warpSPHIntegrators.butcher.RungeKuttaB` performs the repetition.

Functions on a plain dataclass shared (by reference) by the stage systems of one step.
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import torch

from ...enumTypes import EnergyScheme
from ...math.scatter import segment_sum
from warpSPHCore.profiling import record_function

__all__ = ['PairWork', 'RerunRequest', 'beginPairWork', 'listFitsParticles', 'startFromLatest', 'startStage', 'stageRebuildSlack', 'weightedStages', 'accumulateStage', 'energyIncrement']


@dataclass
class RerunRequest:
    """Why `finalize` asks the integrator for the step again, and what the second pass may use.

    `conservative`: run the stages with the list validated (and rebuilt) at every stage instead of the optimistic
    deferred check -- the first pass used a list that turned out too short for some stage.
    `halfVelocity`: the pair list changed between stages; each stage of the second pass adds its own energy share
    with this half-step velocity (see the module docstring)."""
    halfVelocity: Optional[torch.Tensor] = None
    conservative: bool = True
    #: the list the first pass ran on is not to be trusted at all (built from a grid that was too small): start from none
    rebuildList: bool = False


@dataclass
class PairWork:
    #: the combination weights of the propagated solution, one per stage (None: the integrator does not provide them)
    weights: Optional[Tuple[float, ...]] = None
    #: second pass only: the half-step velocity found by the first pass
    halfVelocity: Optional[torch.Tensor] = None
    #: first pass: the pair list `work` is laid out on, and the running `(E, dim)` sum of `k_s f (ap + av)`
    adjacency: Any = None
    work: Optional[torch.Tensor] = None
    #: false once a stage ran on a different pair list than `adjacency`
    consistent: bool = True
    #: second pass: the running per-particle increment
    deltaU: Optional[torch.Tensor] = None
    #: the pair list the previous stage of this step ended on: the next stage starts from it (every stage system
    #: is copied from the step's initial state, which still has the list of the previous step)
    latest: Any = None
    #: a second pass: validate the pair list at every stage (no deferred check)
    conservative: bool = False
    #: first pass with deferred checks: the 0-d device flags 'the list is too short for this stage', read at step end
    flags: Optional[List[torch.Tensor]] = None
    rebuildList: bool = False


def beginPairWork(stageWeights, rerun: Optional[RerunRequest] = None) -> PairWork:
    return PairWork(weights=None if stageWeights is None else tuple(stageWeights),
                    halfVelocity=None if rerun is None else rerun.halfVelocity, conservative=rerun is not None, flags=[],
                    rebuildList=rerun is not None and rerun.rebuildList)


def listFitsParticles(adjacency, state) -> bool:
    """A list built for as many particles as the state has now (host shapes only, no read)."""
    if adjacency is None or getattr(adjacency, 'queryPositions', None) is None or getattr(adjacency, 'referencePositions', None) is None:
        return False
    n = state.positions.shape[0]
    return adjacency.queryPositions.shape[0] == n and adjacency.referencePositions.shape[0] == n and adjacency.numNeighbors.shape[0] == n


def weightedStages(pairWork: Optional[PairWork]) -> int:
    return 0 if pairWork is None or pairWork.weights is None else sum(1 for k in pairWork.weights if k != 0)


def startFromLatest(system) -> None:
    """Stage start: continue from the pair list the previous stage of this step ended on.

    Without it a rebuild at stage 0 would be repeated at stage 1 (whose system is copied from the step's initial
    state, still holding the previous step's list), and the two stages would end on different lists."""
    work = system.pairWork
    if work is not None and work.latest is not None:
        system.adjacency = work.latest


#: fraction of the Verlet buffer that may be used at the first stage before the list is rebuilt there
#: (see `startStage`): the later stages of the step then have the other half to move in
STAGE_ZERO_REBUILD_SLACK = 0.5


def stageRebuildSlack(system, stageIndex) -> float:
    """The `rebuildSlack` for the neighbour-list check of this stage: early rebuild at stage 0 of a tableau with
    more than one weighted stage (see `startStage`), none otherwise."""
    return STAGE_ZERO_REBUILD_SLACK if stageIndex == 0 and weightedStages(system.pairWork) > 1 else 0.0


def startStage(system, state, config, stageIndex) -> None:
    """Stage start: continue from the previous stage's pair list and, at stage 0 of a tableau with more than one
    weighted stage, rebuild early if half the Verlet buffer is already used.

    The running sum only exists on one pair list. A rebuild at stage 1 would cost a whole second pass over the
    step; rebuilding a little early at stage 0 costs one rebuild and makes the mid-step case rare. (A tableau
    with a single weighted stage, midpoint, never carries anything across stages and does not need it.)"""
    startFromLatest(system)


def _pairVector(state, energyScheme) -> torch.Tensor:
    """`f_ij (ap_ij + av_ij)` per pair; `PdV` weights the viscous part with 1/2 and no partition factor."""
    f = state.f_ij.unsqueeze(-1)
    if energyScheme == EnergyScheme.PdV:
        return f * state.ap_ij + 0.5 * state.av_ij
    return f * (state.ap_ij + state.av_ij)


def accumulateStage(pairWork: PairWork, stageIndex: Optional[int], adjacency, state, energyScheme) -> None:
    """Fold stage `stageIndex`'s pair terms into `pairWork` (or, in the second pass, into its per-particle sum)."""
    if pairWork.weights is None or stageIndex is None:
        raise RuntimeError(
            'compatibleEnergy needs the stage weights and the stage index from the integrator '
            '(warpSPHIntegrators RungeKuttaB passes them); this integrator does not provide them')
    k = pairWork.weights[stageIndex]
    if k == 0:
        return
    with record_function("[warpSPH] - [compSPH pairWork] accumulate stage"):
        w = k * _pairVector(state, energyScheme)
        if pairWork.halfVelocity is not None:
            i, j = adjacency.livePairs()
            vji = pairWork.halfVelocity[j] - pairWork.halfVelocity[i]
            rows = segment_sum(torch.einsum('ij,ij->i', vji, w), adjacency.numNeighbors)
            pairWork.deltaU = rows if pairWork.deltaU is None else pairWork.deltaU + rows
        elif pairWork.adjacency is None:
            pairWork.adjacency, pairWork.work = adjacency, w
        elif adjacency is pairWork.adjacency and pairWork.work is not None:
            pairWork.work += w
        else:
            pairWork.consistent, pairWork.work = False, None


def energyIncrement(pairWork: PairWork, initialState, updates: List[Any], weights, dt) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """``(deltaU, None)`` per unit time, or ``(None, halfStepVelocity)`` when the step has to be repeated."""
    if pairWork.weights is not None and tuple(float(w) for w in weights) != pairWork.weights:
        raise RuntimeError('compatibleEnergy: the integrator changed its stage weights between initialize and finalize')
    with record_function("[warpSPH] - [compSPH pairWork] close step"):
        if pairWork.halfVelocity is not None:
            # second pass: every stage has added its share already
            return (pairWork.deltaU if pairWork.deltaU is not None else torch.zeros_like(initialState.internalEnergies)), None
        fullAcceleration = torch.zeros_like(initialState.velocities)
        for k, update in zip(weights, updates):
            fullAcceleration += k * update.dvdt
        halfStepVelocity = initialState.velocities + 0.5 * fullAcceleration * dt
        if not pairWork.consistent or pairWork.work is None:
            return None, RerunRequest(halfVelocity=halfStepVelocity, conservative=True)
        i, j = pairWork.adjacency.livePairs()
        vji = halfStepVelocity[j] - halfStepVelocity[i]
        return segment_sum(torch.einsum('ij,ij->i', vji, pairWork.work), pairWork.adjacency.numNeighbors), None
