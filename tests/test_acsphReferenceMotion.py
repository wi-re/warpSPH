"""Geometric counterexamples, independent of the SPH solve implementation."""
import math
import numpy as np
import pytest
from warpSPH.modules.artificialCompressible.referenceMotion import referenceMotionBudget


def test_extra_verlet_margin_covers_motion_above_old_half_support_limit():
    R=1.;cut=2*math.sqrt(2)
    budget=referenceMotionBudget(R,cut,bounceCutoff=math.sqrt(2),dx=.25)
    old=np.array([[0.,0.],[2.1,0.]])
    current=old+np.array([[.6,0.],[-.6,0.]])
    assert np.linalg.norm(current[0]-current[1])<R
    assert np.linalg.norm(old[0]-old[1])<cut
    assert .5*R<.6<budget.limit


def test_opposed_motion_can_create_a_missing_pair_and_is_rejected():
    R=1.;cut=2*math.sqrt(2)
    old=np.array([[0.,0.],[2.85,0.]])
    current=old+np.array([[1.,0.],[-1.,0.]])
    assert np.linalg.norm(current[0]-current[1])<R
    assert np.linalg.norm(old[0]-old[1])>cut
    assert 1.>referenceMotionBudget(R,cut).limit


def test_cone_stencil_is_a_separate_constraint_from_pair_completeness():
    query=1.01;reference_old=-.0001;reference_now=.7799
    assert abs(query-reference_now)<.25
    assert abs(math.floor(query)-math.floor(reference_old))>1
    displacement=reference_now-reference_old
    budget=referenceMotionBudget(1.,4.)
    assert displacement<budget.pair
    assert displacement>budget.cone==budget.limit


def test_certified_motion_covers_random_interacting_pairs():
    rng=np.random.default_rng(128)
    budget=referenceMotionBudget(1.,2*math.sqrt(2))
    old=rng.uniform(-3,3,(800,2));delta=rng.normal(size=old.shape)
    delta*=.99*budget.limit/np.linalg.norm(delta,axis=1).max()
    current=old+delta
    active=np.linalg.norm(current[:,None]-current[None,:],axis=-1)<1.
    candidate=np.linalg.norm(old[:,None]-old[None,:],axis=-1)<2*math.sqrt(2)
    assert np.all(candidate[active])


def test_tight_candidate_radius_preserves_half_support_budget():
    assert referenceMotionBudget(1.,2.).limit==.5


def test_cached_bounce_motion_reduces_the_budget():
    # The query has already spent 0.6R of its old list's margin before solve.
    old_cut=math.sqrt(2);entry_drift=.6;dx=.25
    budget=referenceMotionBudget(1.,2*math.sqrt(2),bounceCutoff=old_cut-entry_drift,dx=dx)
    assert budget.limit==pytest.approx(old_cut-entry_drift-1.155*dx)
    assert budget.limit<.6<budget.cone


@pytest.mark.parametrize('support,cut',[(0.,2.),(1.,1.),(1.,float('nan'))])
def test_invalid_candidate_geometry_is_rejected(support,cut):
    with pytest.raises(ValueError):referenceMotionBudget(support,cut)
