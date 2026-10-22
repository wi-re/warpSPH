"""Independent ideal-gas identities for the observational entropy ledger."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from crkEntropyLedger import entropy,raw_components


def test_entropy_matches_eos_and_density_reset_at_fixed_energy():
    rho=np.array([.125,1.,4.]);u=np.array([2.,2.5,.625]);g=1.4
    np.testing.assert_allclose(entropy(rho,u,g),np.log((g-1)*rho*u/rho**g),atol=3.e-16,rtol=0)
    new=rho*np.array([.5,1.,2.])
    np.testing.assert_allclose(entropy(new,u,g)-entropy(rho,u,g),(1-g)*np.log(new/rho),atol=3.e-16,rtol=0)


def test_raw_finite_step_split_closes_with_mixed_heating_and_zero_increment():
    rho=np.array([1.,2.,3.,4.]);u=np.array([2.,3.,4.,5.]);g=1.4;dt=.01
    pressure=np.array([2.,-4.,3.,0.]);viscous=np.array([1.,6.,-3.,0.]);new_u=u+dt*(pressure+viscous)
    new_rho=rho*np.array([1.1,.9,1.000000001,1.])
    terms=raw_components(rho,u,new_rho,new_u,pressure,dt,g)
    np.testing.assert_allclose(sum(terms.values()),entropy(new_rho,new_u,g)-entropy(rho,u,g),atol=3.e-16,rtol=0)
    assert terms['pressureHeating'][2]!=0 and terms['viscousHeating'][2]==-terms['pressureHeating'][2]
    assert terms['pressureHeating'][3]==terms['viscousHeating'][3]==0


def test_accepted_path_telescopes_without_adding_midpoint_predictor():
    rho=np.array([1.,2.]);u=np.array([2.,3.]);reset_rho=rho*np.array([1.1,.9]);raw_rho=reset_rho*np.array([.98,1.02]);raw_u=u*np.array([1.03,.99]);final_u=raw_u*np.array([.97,1.04]);g=1.4
    terms=raw_components(reset_rho,u,raw_rho,raw_u,np.array([.1,-.1]),.02,g)
    reset=entropy(reset_rho,u,g)-entropy(rho,u,g);compatible=np.log(final_u/raw_u)
    np.testing.assert_allclose(reset+sum(terms.values())+compatible,entropy(raw_rho,final_u,g)-entropy(rho,u,g),atol=5.e-16,rtol=0)
    # A predictor branch can have an arbitrary entropy excursion and is not
    # part of this accepted full-step identity.
    predictor=entropy(rho*2,u*.8,g)-entropy(reset_rho,u,g)
    assert max(abs(predictor))>.1


@pytest.mark.parametrize('rho,u',[(np.array([0.]),np.array([1.])),(np.array([1.]),np.array([-1.]))])
def test_nonpositive_states_are_reported_instead_of_floored(rho,u):
    with pytest.raises(ValueError,match='positive'):entropy(rho,u)


def test_pressure_shadow_cannot_mutate_cached_physical_warp_parameters():
    from warpSPH.configurations import CRKSPHConfig
    from crkEntropyLedger import pressure_only_config
    config=CRKSPHConfig();config.diffusionParams.C_l=2.;config.diffusionParams.C_q=1.
    actual=config.diffusionParams.__ctype__()
    shadow=pressure_only_config(config)
    assert shadow.diffusionParams.__ctype__() is not actual
    assert shadow.diffusionParams.C_l==shadow.diffusionParams.C_q==0.
    assert config.diffusionParams.C_l==actual.C_l==2.
    assert config.diffusionParams.C_q==actual.C_q==1.
