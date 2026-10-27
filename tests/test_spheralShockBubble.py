"""Shock-bubble state, mirror parity and common derivative diagnostics."""
import sys
import json
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from compare_spheralShockBubble import prepare,primary,Measures,summary
from compare_spheralMaterials import retain_failure
from analyze_spheralShockBubble import circulation,reflection_error


def state(tmp_path):
    root=tmp_path/'bubble'
    prepare(NS(n=40,driver=10.,time=.3,dt=None,frame_every=25,volume='massOverDensity',output=root))
    return root,json.loads((root/'settings.json').read_text()),dict(np.load(root/'input.npz')),dict(np.load(root/'warpInput.npz'))


def test_source_states_share_one_air_material_and_correct_eos(tmp_path):
    root,s,a,w=state(tmp_path)
    assert len(a['positions'])==44*16
    expected=np.linalg.norm(a['positions']-np.array([.8,.2]),axis=1)<.1
    np.testing.assert_array_equal(a['materials']==1,expected)
    np.testing.assert_array_equal(a['materials'][a['positions'][:,0]>1.],0)
    g=np.array(s['gamma'])[a['materials']]
    np.testing.assert_allclose((g-1)*a['densities']*a['internalEnergies'],a['pressures'],rtol=1.e-7)
    assert s['gamma']==[1.4,1.648] and s['nPerh']==1.51 and s['extent']==2.
    np.testing.assert_allclose(np.sum(a['masses'].astype(float)/a['densities']),.44,rtol=1.e-7)


def test_reflected_initial_states_and_identical_primary(tmp_path):
    root,s,a,w=state(tmp_path)
    p=primary(w,len(a['masses']))
    for key in a:np.testing.assert_array_equal(p[key],a[key])
    L=np.array([1.1,.4]);folded=L-abs(w['positions']%(2*L)-L)
    for tile in range(4):
        sl=slice(tile*len(a['masses']),(tile+1)*len(a['masses']))
        np.testing.assert_allclose(folded[sl],a['positions'],atol=2.e-7,rtol=0)
        for key in ('masses','materials','pressures'):np.testing.assert_array_equal(w[key][sl],a[key])
    assert Measures(s,w)(w,0.)['relativeEnergyDrift']==0


def test_common_mls_annihilates_zero_field(tmp_path):
    _,s,a,w=state(tmp_path)
    result,omega=circulation(a)
    np.testing.assert_array_equal(omega,0.)
    assert result['upper']==result['lower']==result['total']==0


def test_common_mls_reproduces_affine_curl_interior(tmp_path):
    _,s,a,w=state(tmp_path)
    x=a['positions'].astype(float)
    # Arbitrary affine vector field, away from incompatible wall continuation.
    a['velocities']=np.column_stack((2*x[:,0]+3*x[:,1]+.5,5*x[:,0]-x[:,1]-.1))
    _,omega=circulation(a)
    interior=(x[:,0]>.2)&(x[:,0]<.9)&(x[:,1]>.1)&(x[:,1]<.3)
    np.testing.assert_allclose(omega[interior],2.,atol=1.e-12,rtol=0)


def test_mirror_audit_detects_unsymmetric_motion(tmp_path):
    _,s,a,w=state(tmp_path)
    check=reflection_error(w,len(a['masses']))
    assert check['primaryParticlesOutside']==0
    assert max(row['maxPositionDifference'] for row in check['tiles'])<2.e-7
    w['positions'][len(a['masses']),0]+=.01
    assert reflection_error(w,len(a['masses']))['tiles'][0]['maxPositionDifference']>.009


def test_negative_energy_stop_is_saved_as_failed_partial_result(tmp_path):
    root,s,a,w=state(tmp_path);folder=root/'spheral';folder.mkdir()
    final={k:v.copy() for k,v in a.items()};final['internalEnergies'][0]=-.01;final['pressures'][0]=-.004
    measure=Measures(s,a);history=[measure(a,0.),measure(final,.1)]
    retain_failure(folder,RuntimeError('negative energy'),'unit-test',history,a,final,'Spheral',s,summary,1.,250,{})
    row=json.loads((folder/'comparisonRun.json').read_text())
    assert not row['passed'] and not row['gates']['execution'] and not row['gates']['finitePositive']
    assert row['time']==.1 and row['termination']=='failed'
    saved=np.load(folder/'failedState.npz')
    np.testing.assert_array_equal(saved['internalEnergies'],final['internalEnergies'])


def test_actual_warp_cubic_matches_compiled_spheral_samples():
    import pytest
    import warp as wp
    from warpSPHCore import scalar_t
    from warpSPHCore.kernels.kernelFunctions.cubicSpline import cubicSpline_k,cubicSpline_dkdq,cubicSpline_C_d
    path=Path(__file__).resolve().parents[1]/'reports/materialEOS/spheral/shockBubble/kernelReference.json'
    if not path.exists():pytest.skip('Compiled Spheral kernel samples not exported')
    reference=json.loads(path.read_text())
    # Compact-support radius R=2h, so convert W_R(q) and dW/dr to h=1.
    @wp.kernel
    def sample(q:wp.array(dtype=scalar_t),value:wp.array(dtype=scalar_t),gradient:wp.array(dtype=scalar_t)):
        i=wp.tid()
        value[i]=cubicSpline_C_d(2)*cubicSpline_k(q[i],2)/scalar_t(4.)
        gradient[i]=cubicSpline_C_d(2)*cubicSpline_dkdq(q[i],2)/scalar_t(8.)
    q=wp.array(np.array(reference['eta'])/2.,dtype=scalar_t,device='cpu');value=wp.empty_like(q);gradient=wp.empty_like(q)
    wp.launch(sample,len(reference['eta']),inputs=[q,value,gradient],device='cpu')
    np.testing.assert_allclose(value.numpy(),reference['kernel'],rtol=2.e-5,atol=2.e-6)
    np.testing.assert_allclose(gradient.numpy(),reference['gradient'],rtol=2.e-5,atol=2.e-6)
    np.testing.assert_allclose(reference['tableKernel'],reference['kernel'],rtol=2.e-6,atol=1.e-8)
