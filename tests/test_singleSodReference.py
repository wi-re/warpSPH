"""Independent gas-dynamic identities for the comparison's exact Sod oracle."""
import importlib.util
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from compare_spheralSod1d import exact


def test_classical_sod_star_satisfies_shock_rankine_hugoniot_and_fan_invariant():
    _,s=exact([0.],.2);g=1.4;rho=.125;P=.1;R=s['densityRight'];u=s['velocity'];p=s['pressure'];speed=s['shockSpeed']
    np.testing.assert_allclose(rho*speed,R*(speed-u),rtol=2.e-14)
    np.testing.assert_allclose(P+rho*speed**2,p+R*(speed-u)**2,rtol=2.e-14)
    np.testing.assert_allclose(g/(g-1)*P/rho+.5*speed**2,g/(g-1)*p/R+.5*(speed-u)**2,rtol=2.e-14)
    np.testing.assert_allclose(p/s['densityLeft']**g,1.,rtol=2.e-14)
    np.testing.assert_allclose(u+2*np.sqrt(g*p/s['densityLeft'])/(g-1),2*np.sqrt(g)/(g-1),rtol=2.e-14)


def test_initial_jump_and_rarefaction_fields():
    a,_=exact(np.array([-.1,.1]),0.)
    np.testing.assert_array_equal(a['density'],[1.,.125]);np.testing.assert_array_equal(a['pressure'],[1.,.1])
    np.testing.assert_allclose(a['energy'],[2.5,2.]);np.testing.assert_array_equal(a['velocity'],[0.,0.])
    x=np.linspace(-.2,-.03,20);a,_=exact(x,.2)
    np.testing.assert_allclose(a['pressure']/a['density']**1.4,1.,rtol=2.e-14)


def test_exact_oracle_agrees_with_pinned_spheral_analytic_solution():
    path=Path(__file__).resolve().parents[1]/'results/materialEOS/spheral-build/source/tests/functional/Hydro/Sod/SodAnalyticSolution.py'
    if not path.exists():
        import pytest
        pytest.skip('Pinned analytic reference not installed')
    spec=importlib.util.spec_from_file_location('native_sod_analytic',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    ref=module.SodSolution(gamma=1.4,rho1=1.,P1=1.,rho2=.125,P2=.1)
    x=np.array([-.4,-.2,-.1,.05,.1,.25,.45]);_,v,u,rho,P,_=ref.solution(.2,x=x)
    a,_=exact(x,.2)
    for key,value in [('velocity',v),('energy',u),('density',rho),('pressure',P)]:
        # The upstream Brent solver requests a roughly 5e-5 pressure tolerance.
        np.testing.assert_allclose(a[key],value,rtol=1.e-4,atol=1.e-4)
