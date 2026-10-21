"""Independent derivative and native-RHS checks for frozen force exports."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from shockBubbleForceAudit import coefficients,gradient,native_split
from capture_shockBubbleForces import selected


def test_native_coefficient_conversion_matches_finite_difference_product_rule():
    rng=np.random.default_rng(738);n=12
    c=rng.normal(size=(n,3));c[:,0]+=3.;dc=rng.normal(size=(n,2,3))
    a={'coefficients':np.concatenate((c,dc.reshape(n,6)),axis=1)}
    r=rng.normal(size=(n,2))*.12;R=np.full(n,.8)
    actual=gradient(r,R,**coefficients(a))
    def value(shift):
        rr=r+shift;cc=c+np.einsum('ndk,d->nk',dc,shift)
        q=np.linalg.norm(rr,axis=1)/R
        W=(40/(7*np.pi))/R**2*np.where(q<.5,1-6*q*q+6*q**3,np.where(q<1,2*(1-q)**3,0))
        return (cc[:,0]+np.sum(cc[:,1:]*rr,axis=1))*W
    for axis in range(2):
        step=np.eye(2)[axis]*1.e-6
        np.testing.assert_allclose(actual[:,axis],(value(step)-value(-step))/(2.e-6),atol=2.e-8,rtol=2.e-8)


def test_capture_clock_only_selects_full_step_rhs():
    assert selected(.05+1e-15,.0002)==.05
    assert selected(.0501,.0002) is None
    assert selected(.15-1e-15,.0002)==.15


def test_actual_native_pair_exports_reduce_to_native_acceleration_and_energy():
    root=Path(__file__).resolve().parents[1]/'results/materialEOS/bubble_force_2026-10-19'
    files=list(root.glob('*/spheral/rhs/t0.050.npz'))
    if not files:pytest.skip('Native Spheral capture not installed')
    for path in files:
        a=dict(np.load(path));result=native_split(a)
        assert all(v<5.e-12 for v in result['checks'].values()),(str(path),result['checks'])


def test_partition_bridge_matches_compiled_spheral_half_step_and_saved_warp_fractions():
    from analyze_shockBubbleForces import doubles,energy_controls
    root=Path(__file__).resolve().parents[1]/'results/materialEOS/bubble_force_2026-10-19'
    files=list((root/'n80-energy-validation/spheral/rhs').glob('*.npz'))
    if not files:pytest.skip('Native energy validation capture not installed')
    assert len(files)==4
    for path in files:
        a=doubles(path);data,_=energy_controls(a,native_split(a))
        assert data['checks']['compiledHalfStepEnergyMaxAbs']<1e-12
        assert data['checks']['compiledHalfStepIncrementRelativeL2']<1e-10
    for path in (root/'n80-complete/warp-rhs').glob('*.npz'):
        a=doubles(path);data,_=energy_controls(a,native_split(a))
        assert data['checks']['savedWarpFractionMaxAbsNonzeroWork']<2e-7
