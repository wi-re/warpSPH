"""Independent dimensional/derivative contracts for the static CRK audit."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_densityDerivative1d import periodic_rows,derivative,controls


def test_periodic_ghosts_retain_original_mass_material_velocity_and_parent():
    a=controls()['materialAffine'][0];rows=periodic_rows(a,[-.5,.5]);n=len(a['masses'])
    assert np.all(rows[:n,6]==1) and np.all(rows[n:,6]==0)
    for r in rows[n:]:
        original=rows[int(r[7])]
        np.testing.assert_array_equal(r[[1,2,3,4,5,8]],original[[1,2,3,4,5,8]])
        assert abs(abs(r[0]-original[0])-1)<1.e-14


def test_finite_difference_separates_geometry_and_support_directions():
    # This homogeneous scalar density map has an exact derivative. It tests
    # the directional perturbation, independently of any CRK implementation.
    a=dict(positions=np.array([[.1],[.3]]),velocities=np.array([[.01],[.03]]),supports=np.array([.02,.02]))
    def mapping(a,domain):
        return dict(paperDensity=1/(np.diff(a['positions'][:,0])[0]+3*a['supports']))
    rho=mapping(a,None)['paperDensity'];eps=1.e-5
    fixed=derivative(mapping,a,None,eps)
    moving=derivative(mapping,a,None,eps,.1*a['supports'])
    np.testing.assert_allclose(fixed/rho,-.1*.2/.26,rtol=2.e-8,atol=0)
    np.testing.assert_allclose(moving/rho,-.1,rtol=2.e-8,atol=0)
    assert np.linalg.norm(fixed-moving)>0


def test_native_pressure_component_formulas_match_warp_in_affine_resolution_jump():
    from warpSPHCore import get_torch_precision
    import torch
    if get_torch_precision()!=torch.float64:pytest.skip('Finite-difference operator audit requires float64')
    from audit_densityDerivative1d import WarpMap,NativeMap
    build=Path(__file__).resolve().parents[1]/'results/materialEOS/spheral-density-derivative1d/build.json'
    if not build.exists():pytest.skip('Pinned compiled Spheral static component library unavailable')
    a,domain=controls()['unequalSpacingAffine'];warp=WarpMap();native=NativeMap(build)
    w,n=warp(a,domain),native(a,domain);mask=abs(a['positions'][:,0])<.2
    for key in ('volumes','paperDensity','stockDensity','gradient','pressureDudt'):
        np.testing.assert_allclose(n[key][mask],w[key][mask],rtol=1.e-6,atol=1.e-8)
    np.testing.assert_allclose(w['gradient'][mask],.1,rtol=0,atol=1.e-12)
    # Homothetic x and R changes must give rho_dot/rho=-a independent of
    # spacing jump or kernel shape. This does not require fixed-R conjugacy.
    dot=derivative(warp,a,domain,1.e-5,.1*a['supports'])
    np.testing.assert_allclose((dot/w['paperDensity'])[mask],-.1,rtol=0,atol=2.e-7)
