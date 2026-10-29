"""Inherited graph settings must select expanded work even with stepMode=None."""
from types import SimpleNamespace as NS
import pytest,torch
from warpSPH.configurations.compSPHConfig import CompSPHConfig
from warpSPH.modules.compSPH.work import useRecomputedWork


@pytest.mark.parametrize('flag',['supportSolveOnGrid','inPlaceAdjacency'])
def test_inherited_graph_grid_keeps_reference_pair_storage(flag):
    scheme=CompSPHConfig();system=NS(pairWork=NS(weights=(0.,1.)));state=NS(positions=torch.ones(2,2))
    assert scheme.stepMode is None
    assert useRecomputedWork(system,state,scheme,None,1)
    setattr(scheme,flag,True)
    assert not useRecomputedWork(system,state,scheme,None,1)
