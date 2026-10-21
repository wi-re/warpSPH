"""Physical closure/compatibility checks for the isolated 1D research solve."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('closure1d_research',Path(__file__).resolve().parents[1]/'scripts/crkClosure1D.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def geometry(n=64,ratio=4,jitter=False):
    nr=n//ratio
    x=np.r_[-1+(np.arange(n)+.5)/n,(np.arange(nr)+.5)/nr]
    volume=np.r_[np.full(n,1./n),np.full(nr,1./nr)]
    if jitter: x+=.08*volume*np.random.default_rng(7).uniform(-1,1,len(x))
    N=len(x)
    edges=np.concatenate([np.stack((np.arange(N),(np.arange(N)+k)%N),axis=1) for k in (1,2,3)])
    i,j=edges.T
    q=np.random.default_rng(17).normal(.1,.03,len(i))
    return x,volume,i,j,q


@pytest.mark.parametrize('ratio',[1,2,4,8])
@pytest.mark.parametrize('jitter',[False,True])
def test_constant_affine_force_and_physical_work(ratio,jitter):
    x,V,i,j,q=geometry(ratio=ratio,jitter=jitter)
    q=module.close_edges(x,V,i,j,q,2.)
    gamma=np.where(x<0,1.4,5/3);rho=np.where(x<0,1.,.25);m=V*rho
    mask=np.abs(x)<.5
    a,dV,du=module.pressure_rhs(V,m,np.ones_like(x),np.full_like(x,.3),i,j,q)
    assert np.max(np.abs(a))<1.e-9
    assert np.max(np.abs(dV))==0
    a,_,_=module.pressure_rhs(V,m,2+x,np.zeros_like(x),i,j,q)
    np.testing.assert_allclose(a[mask],-1/rho[mask],atol=1.e-9,rtol=0)
    P=1+.1*np.sin(np.pi*x);v=.2*np.sin(2*np.pi*x)
    u=P/((gamma-1)*rho)
    a,dV,du=module.pressure_rhs(V,m,P,v,i,j,q)
    assert abs(np.dot(m,a))<1.e-12
    assert abs(np.sum(dV))<1.e-12
    assert abs(np.dot(m*v,a)+np.dot(m,du))<1.e-12
    np.testing.assert_allclose(du/u+(gamma-1)*dV/V,0,atol=1.e-12)


def test_particle_order_edge_order_orientation_and_periodic_translation():
    x,V,i,j,q=geometry(jitter=True)
    expected=module.close_edges(x,V,i,j,q,2.)
    rng=np.random.default_rng(19)
    order=rng.permutation(len(x));inverse=np.argsort(order)
    edge_order=rng.permutation(len(i));flip=rng.random(len(i))<.5
    ii=np.where(flip,j,i);jj=np.where(flip,i,j);qq=np.where(flip,-q,q)
    translated=(x[order]+.47+1)%2-1
    actual=module.close_edges(translated,V[order],inverse[ii[edge_order]],inverse[jj[edge_order]],qq[edge_order],2.)
    restored=np.empty_like(q);restored[edge_order]=actual
    restored=np.where(flip,-restored,restored)
    np.testing.assert_allclose(restored,expected,atol=2.e-12,rtol=2.e-12)


def test_missing_local_cycle_edge_is_rejected():
    x,V,i,j,q=geometry()
    with pytest.raises(ValueError,match='nearest'):
        module.close_edges(x,V,i[1:],j[1:],q[1:],2.)
