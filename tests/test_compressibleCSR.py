"""Whole-step parity and checkpoint compatibility for optional CSR-only neighbours."""
from pathlib import Path
import os
import pytest
import torch
from warpSPH.runner import RunConfiguration

SCHEMES=['CRKSPH','CompSPH','Monaghan','GSPH','InutsukaGSPH','MFM','MFV']

def configured(scheme,omit=None,steps=3,dim=2,tableau=None):
    simulation=dict(device=os.environ.get('WARPSPH_TEST_DEVICE','cpu'),nx=12,dim=dim,
        nSteps=steps,maxDt=.001,plot=False,video=False,store=False,show=False,quiet=True,progress=False)
    if omit is not None:simulation['neighborOmitI']=omit
    if tableau is not None:simulation['integrationScheme']=tableau
    return RunConfiguration('drivenTurbulence',simulation=simulation,solver=dict(scheme=scheme,**(dict(temporalScheme='ssprk') if scheme in ('MFM','MFV') else {})))

@pytest.mark.parametrize('scheme',SCHEMES)
def test_compressible_csr_default_preserves_trajectory(scheme):
    a=configured(scheme,False).run()
    b=configured(scheme).run()
    assert not a.diverged and not b.diverged and a.nSteps==b.nSteps==3
    assert a.ctx.config.compute.neighborOmitI is False
    assert b.ctx.config.compute.neighborOmitI is True
    for name in ('positions','velocities','densities','supports','internalEnergies','Q','sourceEnergyBudget'):
        av,bv=getattr(a.state.state,name,None),getattr(b.state.state,name,None)
        if isinstance(av,torch.Tensor):torch.testing.assert_close(av,bv,rtol=0,atol=0)
    if scheme not in ('MFM','MFV'):
        assert a.state.adjacency.paddedI is not None and b.state.adjacency.paddedI is None
        assert torch.equal(a.state.adjacency.paddedJ,b.state.adjacency.paddedJ)
    else:
        geom=b.state.state._meshlessGeometry.geometry
        assert geom.adjacency.paddedI is None

@pytest.mark.parametrize('scheme',['CompSPH','CRKSPH'])
@pytest.mark.parametrize('tableau',['rungeKutta4'])
def test_multiweight_work_is_bitwise(scheme,tableau):
    a=configured(scheme,False,tableau=tableau).run()
    b=configured(scheme,True,tableau=tableau).run()
    for name in ('velocities','internalEnergies'):
        torch.testing.assert_close(getattr(a.state.state,name),getattr(b.state.state,name),rtol=0,atol=0)
    n=a.state.pairWork.adjacency.numEdges
    torch.testing.assert_close(a.state.pairWork.work[:n],b.state.pairWork.work[:n],rtol=0,atol=0)

@pytest.mark.parametrize('saved,restored',[(False,True),(True,False),(True,True)])
def test_checkpoint_allows_coo_and_csr_storage_changes(saved,restored,tmp_path):
    cfg=configured('CRKSPH',saved,steps=2)
    cfg.simulation.update(store=True,storeMode='states',exportRoot=str(tmp_path/'saved'))
    first=cfg.run()
    path=Path(first.exportPath)/'trajectory/finalState.h5'
    continued=configured('CRKSPH',restored,steps=2)
    continued.simulation['resumeFrom']=str(path)
    b=continued.run()
    a=configured('CRKSPH',saved,steps=4).run()
    for name in ('positions','velocities','densities','internalEnergies','sourceEnergyBudget'):
        torch.testing.assert_close(getattr(a.state.state,name),getattr(b.state.state,name),rtol=0,atol=0)
    assert (b.state.adjacency.paddedI is None)==restored


def test_other_family_keeps_coo_default_and_override():
    from warpSPH.runner.runner import buildContext
    from warpSPH.runner import CaseSpec
    from warpSPH.cases.compressible.drivenTurbulence import drivenTurbulenceCase
    for override,expected in ((None,False),(True,True),(False,False)):
        ctx=buildContext(drivenTurbulenceCase,CaseSpec(scheme='sun2017DeltaSPH',device='cpu',neighborOmitI=override))
        assert ctx.config.compute.neighborOmitI is expected


def test_native_periodic_steps_never_reconstruct_rows(monkeypatch):
    from warpSPHCore.dataTypes import AdjacencyList
    def forbidden(*args,**kwargs):
        raise AssertionError('native CSR trajectory requested COO rows')
    monkeypatch.setattr(AdjacencyList,'cooRows',forbidden)
    for scheme in SCHEMES:
        result=configured(scheme,True,steps=2).run()
        assert not result.diverged


@pytest.mark.parametrize('scheme',['MFM','MFV'])
def test_cached_meshless_search_keeps_csr_parity(scheme):
    results=[]
    for omit in (False,True):
        cfg=configured(scheme,omit)
        cfg.solver['neighborSkin']=1.2
        results.append(cfg.run())
    a,b=results
    torch.testing.assert_close(a.state.state.Q,b.state.state.Q,rtol=0,atol=0)
    assert b.state.state._meshlessNeighbors.adjacency.paddedI is None
