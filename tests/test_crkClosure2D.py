"""Tensor consistency and matched physical work for the 2D research candidate."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('closure2d_research',Path(__file__).resolve().parents[1]/'scripts/crkClosure2D.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


@pytest.mark.parametrize('shape,jitter',[('planar',0.),('planar',.15),('circle',.1)])
@pytest.mark.parametrize('taper',[False,True])
def test_tensor_force_and_pair_work(shape,jitter,taper):
    x,V=module.sample(24,jitter=jitter,shape=shape)
    assert abs(V.sum()-1)<1.e-14
    i,j,r,q0,R,d=module.geometry(x,V)
    assert max(d.values())<1.e-10
    q,info=module.close_edges(V,i,j,r,q0,tolerance=1.e-13,
        edge_scale=module.compact_taper(V,i,j,r) if taper else None)
    n=len(V);moment=np.zeros((n,2,2));qr=q[:,:,None]*r[:,None,:]
    np.add.at(moment,i,qr);np.add.at(moment,j,qr)
    np.testing.assert_allclose(moment/V[:,None,None],np.tile(np.eye(2),(n,1,1)),atol=1.e-9,rtol=0)
    rho=np.where(V<2/24**2,1.,.25);m=V*rho;gamma=np.where(rho==1,1.4,5/3)
    a,dV,du=module.pressure_rhs(V,m,np.ones(n),np.tile([.3,.2],(n,1)),i,j,q)
    assert np.max(abs(a))<1.e-7
    assert np.max(abs(dV))==0
    rng=np.random.default_rng(4);P=rng.uniform(.5,1.5,n);v=rng.normal(size=(n,2));u=P/((gamma-1)*rho)
    a,dV,du=module.pressure_rhs(V,m,P,v,i,j,q)
    assert np.max(abs(np.sum(m[:,None]*a,axis=0)))<1.e-12
    assert abs(dV.sum())<1.e-9
    assert abs(np.dot(m,np.einsum('na,na->n',v,a)+du))<1.e-12
    np.testing.assert_allclose(du/u+(gamma-1)*dV/V,0,atol=1.e-12)


@pytest.mark.parametrize('taper',[False,True])
def test_rotation_permutation_and_edge_orientation(taper):
    x,V=module.sample(24,jitter=.1);i,j,r,q0,R,d=module.geometry(x,V)
    scale=module.compact_taper(V,i,j,r) if taper else None
    expected,_=module.close_edges(V,i,j,r,q0,edge_scale=scale)
    rng=np.random.default_rng(5);order=rng.permutation(len(V));inverse=np.argsort(order)
    edge_order=rng.permutation(len(i));flip=rng.random(len(i))<.5;sign=np.where(flip,-1.,1.)
    rotation=np.array([[0.,-1.],[1.,0.]])
    ii=np.where(flip,j,i);jj=np.where(flip,i,j)
    actual,_=module.close_edges(V[order],inverse[ii[edge_order]],inverse[jj[edge_order]],
        (r*sign[:,None])[edge_order]@rotation.T,(q0*sign[:,None])[edge_order]@rotation.T,
        edge_scale=scale[edge_order] if scale is not None else None)
    restored=np.empty_like(q0);restored[edge_order]=actual@rotation
    restored*=sign[:,None]
    np.testing.assert_allclose(restored,expected,atol=1.e-10,rtol=1.e-8)


def test_force_is_continuous_when_compact_edges_appear():
    x,V=module.sample(24)
    perturb=np.stack((np.sin(2*np.pi*x[:,0])*np.cos(2*np.pi*x[:,1]),
                      np.cos(2*np.pi*x[:,0])*np.sin(2*np.pi*x[:,1])),axis=1)
    def forces(epsilon):
        position=(x+epsilon*perturb)%1;i,j,r,q0,R,d=module.geometry(position,V)
        P=1+.1*np.sin(2*np.pi*position.sum(axis=1));result=[]
        for scale in (None,module.compact_taper(V,i,j,r)):
            q,_=module.close_edges(V,i,j,r,q0,1.e-13,scale)
            a,_,_=module.pressure_rhs(V,V,P,np.zeros_like(x),i,j,q);result.append(a)
        return result
    base=forces(0);large=forces(1.e-6);small=forces(1.e-8)
    ratios=[np.linalg.norm(a-b)/np.linalg.norm(c-b) for a,b,c in zip(large,base,small)]
    assert .9<ratios[0]<1.1 # Negative control: finite untapered graph jump.
    assert 80<ratios[1]<120
