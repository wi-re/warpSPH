"""Independent analytic Euler, forcing-work and source-integration controls."""
from types import SimpleNamespace
import math
import numpy as np
import pytest
import torch
from warpSPH.modules.forcing.analytic import OscillatingShear
from warpSPH.modules.meshless.sources import ConservedSources
from warpSPH.modules.forcing.primitive import finishSources
from warpSPH.utils.turbulenceBudget import PeriodicBudget


@pytest.mark.parametrize('dim,axis,component', [(2,0,1),(3,2,0)])
def test_shear_euler_residual_and_filtered_work(dim, axis, component):
    length=1.7; n=16; t=.31; A=.23; omega=2*math.pi/2.3; mode=2
    axes=[(np.arange(n)+.5)*length/n]*dim
    x=np.stack(np.meshgrid(*axes,indexing='ij'),-1)
    flat=torch.tensor(x.reshape(-1,dim),dtype=torch.float64)
    field=OscillatingShear(A,2.3,length,mode,dim,axis,component)
    v=field.velocity(flat,t).numpy().reshape(x.shape)
    a=field(flat,t).numpy().reshape(x.shape)
    expected=np.zeros_like(v); expected[...,component]=A*omega*np.cos(omega*t)*np.sin(2*np.pi*mode*x[...,axis]/length)
    np.testing.assert_allclose(a,expected,atol=2e-15,rtol=2e-15)
    rho=np.full((n,)*dim,1.3);p=np.full_like(rho,.7)
    engine=PeriodicBudget(rho.shape,length)
    drho,dmom=engine.inviscidRHS(rho,v,p,rho[...,None]*a)
    np.testing.assert_allclose(drho,0,atol=2e-14)
    np.testing.assert_allclose(dmom,rho[...,None]*a,atol=3e-14)
    ell=.11; budget=engine.measure(rho,v,p,ell=ell,forceDensity=rho[...,None]*a)
    expectedPower=1.3*A*A*omega*np.sin(2*omega*t)/4*np.exp(-(ell*2*np.pi*mode/length)**2)
    assert budget['forcingWork']==pytest.approx(expectedPower,abs=2e-16)
    for key in ('deformationTransfer','baropycnalTransfer','pressureDilatation'):
        assert abs(budget[key])<1e-14
    shifted=flat.clone();shifted[:,axis]+=3*length
    torch.testing.assert_close(field(shifted,t),field(flat,t),rtol=2e-13,atol=2e-14)


@pytest.mark.parametrize('t',[.2,.7,1.2,1.7])
def test_native_source_rates_match_analytic_positive_and_negative_work(t):
    field=OscillatingShear();x=torch.stack(torch.meshgrid(*[torch.arange(16,dtype=torch.float64)/16]*2,indexing='ij'),-1).reshape(-1,2)
    m=torch.full((len(x),),1/len(x),dtype=x.dtype);v=field.velocity(x,t)
    context=SimpleNamespace(Q=torch.cat((m[:,None],m[:,None]*v,m[:,None]*.9),-1),pos=x,vel=v,u=torch.full_like(m,.9))
    rates,budget=ConservedSources(field,True).rates(context,t)
    assert float(budget[0])==pytest.approx(field.inputPower(t),abs=2e-16)
    assert float(budget[1])==0
    assert float(rates[:,1:-1].sum(0).abs().max())<1e-16
    assert float((.5*m[:,None]*v*v).sum())==pytest.approx(field.kineticEnergy(t),abs=2e-17)


def test_source_only_midpoint_converges_and_compatible_work_is_exact_discrete_change():
    # No hydro is evolved here: isolate the source ODE and production work finish.
    field=OscillatingShear();x=torch.tensor([[.125,0.],[-.125,0.]],dtype=torch.float64)
    m=torch.full((2,),.5,dtype=x.dtype);T=.63;errors=[]
    for n in (10,20,40):
        dt=T/n;v=torch.zeros_like(x);budget=torch.zeros(2,dtype=x.dtype);u=torch.full_like(m,.9)
        for j in range(n):
            initial=SimpleNamespace(state=SimpleNamespace(masses=m,velocities=v,sourceEnergyBudget=budget))
            a=field(x,(j+.5)*dt);vnew=v+dt*a
            s=SimpleNamespace(masses=m,velocities=vnew,internalEnergies=u.clone(),densities=torch.ones_like(m),materials=torch.zeros(2,dtype=torch.int32))
            update=SimpleNamespace(dvdt=a,sourceAcceleration=a,sourceHeat=torch.zeros_like(m),sourceEnergyRates=torch.zeros(2,dtype=x.dtype))
            finishSources(SimpleNamespace(state=s,sourceTerms=object()),initial,[update],[1.],dt,SimpleNamespace(gamma=5/3,materials=None),compatible=True)
            v,budget=s.velocities,s.sourceEnergyBudget
        assert abs(float(budget[0]-(.5*m[:,None]*v*v).sum()))<2e-17
        errors.append(float((v-field.velocity(x,T)).norm()))
    assert 3.98<errors[0]/errors[1]<4.02
    assert 3.98<errors[1]/errors[2]<4.02


@pytest.mark.parametrize('kw',[dict(period=0),dict(amplitude=float('nan')),dict(mode=0),dict(mode=1.5),dict(component=0),dict(dim=1)])
def test_bad_shear_parameters(kw):
    with pytest.raises(ValueError):OscillatingShear(**kw)
