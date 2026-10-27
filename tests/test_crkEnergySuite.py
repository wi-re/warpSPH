"""Independent physical identities for the energy-policy suite inputs/oracles."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from study_crkEnergySuite import CASES,POLICIES,sample,vortex_fields,wave_fields


def test_all_common_inputs_are_positive_pressure_balanced_single_gas():
    for case,cfg in CASES.items():
        a=sample(case);assert min(a['masses'])>0 and min(a['internalEnergies'])>0
        assert np.unique(a['materials']).tolist()==[0]
        np.testing.assert_allclose(a['pressures'],(cfg['gamma']-1)*a['densities']*a['internalEnergies'],rtol=3.e-15)
        assert len(np.unique(a['UIDs']))==len(a['masses'])
    assert {(v['compatible'],v['thermal']) for v in POLICIES.values()}=={(a,b) for a in (False,True) for b in ('base','entropy')}


def test_vortex_is_isentropic_and_satisfies_radial_euler_balance():
    r=np.linspace(.01,3.,101);x=np.column_stack((r,np.zeros_like(r)));h=1.e-5
    fields=vortex_fields(x);left=vortex_fields(x-np.array([h,0]));right=vortex_fields(x+np.array([h,0]))
    np.testing.assert_allclose(fields['pressure']/fields['density']**1.4,1.,atol=3.e-16)
    dPdr=(right['pressure']-left['pressure'])/(2*h)
    rotational=fields['velocity'][:,1]-.25
    np.testing.assert_allclose(dPdr,fields['density']*rotational**2/r,rtol=2.e-8,atol=2.e-11)
    shifted=vortex_fields(x+np.array([.5,.25])*1.7,1.7)
    for key in fields:np.testing.assert_allclose(fields[key],shifted[key],atol=2.e-15,rtol=2.e-15)


def test_wave_is_a_right_going_acoustic_eigenmode_and_returns_after_crossing():
    x=np.linspace(0,1,137)[:,None];initial=wave_fields(x);final=wave_fields(x,1.)
    np.testing.assert_allclose(initial['pressure']-1/1.4,initial['density']-1,atol=2.e-16)
    np.testing.assert_allclose(initial['velocity'][:,0],initial['density']-1,atol=2.e-16)
    for key in initial:np.testing.assert_allclose(final[key],initial[key],rtol=2.e-15,atol=2.e-16)
