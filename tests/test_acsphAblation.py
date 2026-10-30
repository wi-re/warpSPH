"""Ablation contracts: one declared change, unchanged inputs, real stop gates."""
from dataclasses import dataclass
from types import SimpleNamespace
import pytest
import torch

from scripts.acsph_reference.ablation_matrix import Options,rows,select,validate,notable


def test_cumulative_ladder_has_one_declared_delta_per_row():
    cohort=rows();assert len(cohort)==11
    for previous,current in zip(cohort,cohort[1:]):
        delta=[k for k in previous['options'] if previous['options'][k]!=current['options'][k]]
        assert delta==[current['change']]
        validate(Options(**current['options']))
    assert sum(r['reuse'] for r in cohort)==2
    assert all(cohort[-1]['options'].values())


def test_non_cumulative_or_remap_without_geometry_is_rejected():
    with pytest.raises(ValueError):validate(Options(pressure_remap=True))


def test_historical_schema_normalizes_only_missing_null_diagnostics():
    from scripts.acsph_reference.campaign_lobovsky_ablation import historical_progress_equal
    old=[{'t':.1,'meanPressure':123.,'epsilonV':-8.}]
    assert historical_progress_equal([dict(old[0],epsilonP=None,exitReason=None)],old)
    assert not historical_progress_equal([dict(old[0],epsilonP=-6.)],old)
    assert not historical_progress_equal([dict(old[0],meanPressure=124.)],old)
    assert not historical_progress_equal([dict(old[0],unknown=None)],old)


def test_half_clock_preserves_bdf_and_base_configuration():
    from scripts.acsph_reference.ablation_controls import warm_problem
    @dataclass
    class Problem:
        schemeConfig:object
        dtau:float
        k1:object
        k2:object
        bdf:tuple
    pr=Problem(SimpleNamespace(acParams=SimpleNamespace(dtOverDtau=2.,epsilonV=-8)),.01,
        torch.tensor([100.]),torch.tensor([.03]),(15.,-20.,5.))
    half=warm_problem(pr)
    assert half.dtau==pr.dtau/2 and half.schemeConfig.acParams.dtOverDtau==4
    assert pr.schemeConfig.acParams.dtOverDtau==2 and pr.dtau==.01
    assert half.bdf==pr.bdf
    torch.testing.assert_close(half.k1,pr.k1*4,rtol=0,atol=0)
    torch.testing.assert_close(half.k2,pr.k2*2,rtol=0,atol=0)
    # Physical dt is identical; beta doubles, pressure wave coefficient x4.
    assert half.dtau*half.schemeConfig.acParams.dtOverDtau==pr.dtau*pr.schemeConfig.acParams.dtOverDtau


@pytest.mark.parametrize('row,expected,reason',[(8,1,'full-residuals'),(9,7,'iteration-cap'),(10,6,'pressure-plateau')])
def test_pressure_gate_and_plateau_are_separate_real_controls(runtime,monkeypatch,row,expected,reason):
    from scripts.acsph_reference.ablation_controls import reference_stopping
    from warpSPH.modules.artificialCompressible import dyngpusphReference as ref
    from warpSPH.configurations import ArtificialCompressibleSPHConfig
    from warpSPH.configurations.simulationConfig import SimulationConfig
    from warpSPH.systems.artificialCompressible import ArtificialCompressibleState
    from warpSPH.utils.domain import buildDomainDescription
    from warpSPHCore import get_torch_precision
    dtype=get_torch_precision();x=torch.zeros((1,2),dtype=dtype);z=torch.zeros(1,dtype=dtype)
    st=ArtificialCompressibleState(positions=x,velocities=x.clone(),pressures=z.clone(),
        densities=torch.full_like(z,1000),supports=torch.full_like(z,.003),masses=torch.full_like(z,.0005625),
        kinds=torch.zeros(1,dtype=torch.int32),materials=torch.zeros(1,dtype=torch.int32),
        UIDs=torch.arange(1,dtype=torch.int32),UIDcounter=1)
    config=SimulationConfig(device=torch.device('cpu'),dtype=dtype,dim=2,dx=.00075,
        domain=buildDomainDescription(l=1.,dim=2,periodic=False,device='cpu',dtype=dtype))
    scheme=ArtificialCompressibleSPHConfig();scheme.acParams.incrementalPositions=False
    scheme.acParams.epsilonV=-8;scheme.acParams.maxPseudoIterations=7;scheme.acParams.minPseudoIterations=1
    pr=SimpleNamespace(state=st,config=config,schemeConfig=scheme,bounce=None,positionOrigin=None)
    def manufactured(problem,x,v,p,lam,norm,edges,switch,**kw):
        return x,v,p,lam,norm,edges,switch,p.new_tensor([-9.,-3.],dtype=torch.float64),v,p,p.new_zeros(())
    monkeypatch.setattr(ref,'validate',lambda _:None)
    monkeypatch.setattr(ref,'iteration',manufactured)
    monkeypatch.setattr(ref,'solve',ref.solve)
    reference_stopping(Options(**select(str(row))['options']))
    result=ref.solve(pr,st.positions,st.velocities,st.pressures)
    assert result.iterations==expected and result.exitReason==reason


def test_screening_ignores_large_relative_changes_at_tiny_absolute_noise():
    assert not notable({'P1RmsPa':.1,'caps':1},{'P1RmsPa':1.,'caps':2})
    assert notable({'P1RmsPa':1200.,'caps':11},{'P1RmsPa':14000.,'caps':48})


@pytest.mark.parametrize('row',[3,4,5])
def test_field_controls_retain_exact_geometry_and_history_even_for_pressure_safeguard(row):
    from scripts.acsph_reference.ablation_controls import select_remap_outputs
    state=SimpleNamespace(pressures=torch.tensor([27.]),velocities=torch.tensor([[1.,2.]]))
    # Source pressure safeguard can zero an isolated particle independently of
    # its gradient. Suppressing p transport must suppress that assignment too.
    result=tuple(torch.tensor([float(i)]) for i in range(8))
    actual=select_remap_outputs(result,state,Options(**select(str(row))['options']))
    for i in (0,1,4,5,6,7):assert actual[i] is result[i]
    assert actual[2] is (result[2] if row>=4 else state.pressures)
    assert actual[3] is (result[3] if row>=5 else state.velocities)
