"""Reference-symbol and optional real-device closure/work regression tests."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from crkClosure2D import sample,geometry,compact_taper,close_edges
from crkClosureGPU2D import GPUClosure,spectral_inverse


def test_fft_symbol_inverts_independent_uniform_edge_operator():
    n=16;x,V=sample(n,unequal=False);i,j,r,q0,R,d=geometry(x,V)
    scale=compact_taper(V,i,j,r)
    a=np.column_stack((np.ones(len(i)),n*r));b=a.copy();b[:,0]=-1
    potential=np.random.default_rng(41).normal(size=(n*n,3));potential[:,0]-=potential[:,0].mean()
    edge=scale**2*(np.einsum('na,na->n',a,potential[i])+np.einsum('na,na->n',b,potential[j]))
    applied=np.zeros_like(potential);np.add.at(applied,i,a*edge[:,None]);np.add.at(applied,j,b*edge[:,None])
    frequency=np.fft.rfft2(applied.reshape(n,n,3),axes=(0,1))
    result=np.fft.irfft2(np.einsum('xyab,xyb->xya',spectral_inverse(n),frequency),s=(n,n),axes=(0,1)).reshape(n*n,3)
    np.testing.assert_allclose(result,potential,atol=1.e-9,rtol=1.e-9)


@pytest.mark.parametrize('backend',['jacobi','fft','coarse'])
def test_gpu_coefficients_actual_constraints_and_matched_power(backend):
    import torch
    if not torch.cuda.is_available():pytest.skip('Requires CUDA device')
    n=24;x,V=sample(n,unequal=backend!='fft',jitter=.02)
    i,j,r,q0,R,d=geometry(x,V);expected,_=close_edges(V,i,j,r,q0,1.e-13,compact_taper(V,i,j,r))
    gpu=GPUClosure(len(V),logical_n=n if backend=='fft' else 0,coarse_k=4 if backend=='coarse' else 0)
    ii,jj,q,info=gpu.solve(x,V)
    lookup={(int(a),int(b)):z for a,b,z in zip(ii,jj,q)}
    actual=np.array([lookup[(int(a),int(b))] for a,b in zip(i,j)])
    np.testing.assert_allclose(actual,expected,atol=1.e-10,rtol=1.e-7)
    rng=np.random.default_rng(5);P=rng.uniform(.5,1.5,len(V));v=rng.normal(size=x.shape)
    mass=V*np.where(x[:,0]<.5,1.,.25)
    a,dv,du=gpu.pressure_rhs(V,mass,P,v)
    assert np.linalg.norm(np.sum(mass[:,None]*a,axis=0))<1.e-10
    assert abs(np.dot(mass,np.einsum('na,na->n',v,a)+du))<1.e-10
    assert abs(dv.sum())<1.e-8
    assert info['constraintResidual']<1.e-8


def test_coarse_moving_geometry_support_change_and_rebuild():
    import torch
    if not torch.cuda.is_available():pytest.skip('Requires CUDA device')
    n=24;x,V=sample(n,unequal=False)
    gpu=GPUClosure(len(V),coarse_k=4);gpu.solve(x,V)
    initial_rebuilds=gpu.rebuilds
    moved=(x+np.stack((.08*np.sin(2*np.pi*x[:,1]),.015*np.sin(2*np.pi*x[:,0])),axis=1))%1
    volume=V*(1+.08*np.sin(2*np.pi*x[:,0]));volume/=volume.sum()
    ii,jj,actual,info=gpu.solve(moved,volume)
    assert gpu.rebuilds>initial_rebuilds
    i,j,r,q0,R,d=geometry(moved,volume)
    expected,_=close_edges(volume,i,j,r,q0,1.e-13,compact_taper(volume,i,j,r))
    lookup={(int(a),int(b)):z for a,b,z in zip(ii,jj,actual)}
    values=np.array([lookup[(int(a),int(b))] for a,b in zip(i,j)])
    np.testing.assert_allclose(values,expected,atol=1.e-10,rtol=1.e-7)
