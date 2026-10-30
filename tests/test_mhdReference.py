"""Independent continuum/reference checks; no MHD solver is advertised."""
import math
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from _mhdReference import (magnetic_energy_density, magnetic_increment_energy,
    wave_speeds, circular_alfven, potential_mode, resistive_mode, divergence_summary)


@pytest.mark.parametrize('mu',[1.,2.3])
def test_magnetic_increment_keeps_quadratic_energy_and_can_remove_energy(mu):
    b=np.array([[1.,2.,3.],[0.,0.,0.],[-2.,1.,.3]])
    db=np.array([[-1.,-2.,-3.],[2.,1.,0.],[.2,-.7,.1]])
    change=magnetic_increment_energy(b,db,mu=mu)
    np.testing.assert_allclose(change, magnetic_energy_density(b+db,mu=mu)-magnetic_energy_density(b,mu=mu),atol=1e-15)
    assert change[0]<0 and change[1]>0  # Zero-field injection needs the quadratic term.
    np.testing.assert_array_equal(magnetic_energy_density(np.zeros((5,3)),mu=mu),np.zeros(5))


@pytest.mark.parametrize('mu',[1.,2.3])
def test_wave_speeds_have_hydro_parallel_and_perpendicular_limits(mu):
    args=dict(density=2.,pressure=3.,gamma=1.4,normal=[1.,0.,0.],mu=mu)
    sound=math.sqrt(2.1)
    slow,alfven,fast=wave_speeds(field=[0.,0.,0.],**args)
    assert (slow,alfven,fast)==pytest.approx((0.,0.,sound))
    b=3.; va=b/math.sqrt(mu*2)
    assert wave_speeds(field=[b,0.,0.],**args)==pytest.approx((min(sound,va),va,max(sound,va)))
    assert wave_speeds(field=[0.,b,0.],**args)==pytest.approx((0.,0.,math.hypot(sound,va)))


def test_wave_polynomial_and_rotation_invariance():
    b=np.array([.7,-1.2,.3]); n=np.array([2.,-1.,3.]); n/=np.linalg.norm(n)
    args=dict(density=.8,pressure=1.3,gamma=5/3,mu=1.7)
    slow,ca,fast=wave_speeds(field=b,normal=n,**args)
    sound2=args['gamma']*args['pressure']/args['density']
    va2=np.dot(b,b)/(args['mu']*args['density'])
    assert slow**2+fast**2==pytest.approx(sound2+va2)
    assert slow**2*fast**2==pytest.approx(sound2*ca**2)
    rotation=np.linalg.qr(np.array([[1.,2.,-.3],[.4,-.2,1.1],[2.,.3,.8]]))[0]
    np.testing.assert_allclose(wave_speeds(field=rotation@b,normal=rotation@n,**args),[slow,ca,fast])


def _ideal_conservation(wave, gamma, mu):
    # Independently assemble Cartesian continuum U,F_x for the PDE residual.
    rho,p,v,b=[wave[key] for key in ('density','pressure','velocity','field')]
    magnetic=np.sum(b*b,axis=1)/(2*mu)
    energy=p/(gamma-1)+.5*rho*np.sum(v*v,axis=1)+magnetic
    total_pressure=p+magnetic
    momentum_flux=rho[:,None]*v[:,0,None]*v-b[:,0,None]*b/mu
    momentum_flux[:,0]+=total_pressure
    energy_flux=(energy+total_pressure)*v[:,0]-np.sum(v*b,axis=1)*b[:,0]/mu
    induction_flux=v[:,0,None]*b-b[:,0,None]*v
    u=np.column_stack((rho,rho[:,None]*v,energy,b))
    flux=np.column_stack((rho*v[:,0],momentum_flux,energy_flux,induction_flux))
    return u,flux


@pytest.mark.parametrize('direction',[-1,1])
@pytest.mark.parametrize('bulk',[0.,.37])
@pytest.mark.parametrize('mu',[1.,2.3])
def test_finite_amplitude_alfven_satisfies_full_ideal_mhd(direction,bulk,mu):
    x=np.linspace(0.,2.,79,endpoint=False); t=.23; dx=1e-5; dt=1e-5
    args=dict(length=2.,density=1.7,pressure=.9,parallel_field=.8,
              amplitude=.4,bulk_velocity=bulk,direction=direction,mu=mu)
    def state(x,t):return circular_alfven(x,t,**args)
    up=_ideal_conservation(state(x,t+dt),1.4,mu)[0]
    um=_ideal_conservation(state(x,t-dt),1.4,mu)[0]
    fp=_ideal_conservation(state(x+dx,t),1.4,mu)[1]
    fm=_ideal_conservation(state(x-dx,t),1.4,mu)[1]
    np.testing.assert_allclose((up-um)/(2*dt)+(fp-fm)/(2*dx),0.,atol=1e-9)
    wave=state(x,t)
    np.testing.assert_allclose(np.sum(wave['field']**2,axis=1),.8**2+.4**2,atol=2e-16)
    np.testing.assert_array_equal(wave['field'][:,0],np.full(len(x),.8))
    speed=bulk+direction*.8/math.sqrt(mu*1.7)
    if speed!=0:
        period=2/abs(speed)
        np.testing.assert_allclose(state(x,t+period)['field'],wave['field'],atol=2e-15)


def test_vector_potential_mode_is_solenoidal_periodic_and_advects():
    points=np.random.default_rng(71).uniform(0.,1.,size=(37,2))*[2.,3.]
    args=dict(lengths=(2.,3.),amplitude=.3,drift=(.4,-.2));t=.17;h=1e-5
    def field(p,t):return potential_mode(p,t,**args)
    gradients=[]
    for axis in range(2):
        shift=np.eye(2)[axis]*h
        gradients.append((field(points+shift,t)-field(points-shift,t))/(2*h))
        np.testing.assert_allclose(field(points+np.eye(2)[axis]*args['lengths'][axis],t),field(points,t),atol=2e-15)
    np.testing.assert_allclose(gradients[0][:,0]+gradients[1][:,1],0.,atol=5e-10)
    derivative=(field(points,t+h)-field(points,t-h))/(2*h)
    np.testing.assert_allclose(derivative+.4*gradients[0]-.2*gradients[1],0.,atol=5e-10)
    np.testing.assert_array_equal(field(points,t)[:,2],np.zeros(len(points)))


@pytest.mark.parametrize('seed_amplitude',[0.,.2])
def test_periodic_induction_source_preserves_mean_flux_and_supplies_recorded_energy(seed_amplitude):
    x,y=np.meshgrid((np.arange(32)+.5)*2/32,(np.arange(48)+.5)*3/48,indexing='ij')
    points=np.column_stack((x.ravel(),y.ravel()))
    b=potential_mode(points,0.,lengths=(2.,3.),amplitude=seed_amplitude)+[1.,.5,.2]
    rate=potential_mode(points,0.,lengths=(2.,3.),amplitude=.3)
    increment=.03*rate
    np.testing.assert_allclose(rate.mean(axis=0),0.,atol=1e-16)
    np.testing.assert_allclose((b+increment).mean(axis=0),b.mean(axis=0),atol=3e-15)
    mu,volume=2.3,6/len(points)
    supplied=volume*magnetic_increment_energy(b,increment,mu=mu).sum()
    measured=volume*(magnetic_energy_density(b+increment,mu=mu)-magnetic_energy_density(b,mu=mu)).sum()
    assert supplied==pytest.approx(measured,abs=2e-16)
    assert supplied>0
    if seed_amplitude==0:
        linear_only=volume*np.sum(b*increment)/mu
        assert abs(linear_only)<1e-16


@pytest.mark.parametrize('eta',[0.,.07])
@pytest.mark.parametrize('mu',[1.,2.3])
def test_resistive_force_free_mode_converts_magnetic_energy_to_joule_heat(eta,mu):
    x=np.linspace(0.,2.,128,endpoint=False);t=.3;h=1e-4;dt=1e-5
    args=dict(length=2.,density=1.7,internal_energy=.9,amplitude=.8,magnetic_diffusivity=eta,mu=mu)
    def state(x,t):return resistive_mode(x,t,**args)
    initial,current=state(x,0.),state(x,t)
    energy=lambda row:1.7*row['internal_energy']+np.sum(row['field']**2,axis=1)/(2*mu)
    np.testing.assert_allclose(energy(current),energy(initial),atol=1e-15)
    bp,b,bm=state(x+h,t)['field'],current['field'],state(x-h,t)['field']
    dbdx=(bp-bm)/(2*h)
    curl=np.column_stack((np.zeros(len(x)),-dbdx[:,2],dbdx[:,1]))
    np.testing.assert_allclose(np.cross(curl,b),0.,atol=1e-12)
    dBdt=(state(x,t+dt)['field']-state(x,t-dt)['field'])/(2*dt)
    np.testing.assert_allclose(dBdt,eta*(bp-2*b+bm)/h**2,atol=3e-9,rtol=2e-6)
    du=(state(x,t+dt)['internal_energy']-state(x,t-dt)['internal_energy'])/(2*dt)
    np.testing.assert_allclose(du,current['joule_heating_rate'],atol=2e-11)
    np.testing.assert_allclose(du,eta*np.sum(curl*curl,axis=1)/(mu*1.7),atol=1e-8,rtol=1e-7)
    assert current['internal_energy'].min()>=.9


def test_local_divergence_cannot_hide_in_a_zero_signed_integral():
    x=(np.arange(64)+.5)/64
    div=2*np.cos(2*math.pi*x)
    row=divergence_summary(div,length_scale=.1,volume=np.full(64,1/64),reference_field=.5,estimator='analytic manufactured')
    assert abs(row['signed_volume_integral'])<1e-15
    assert row['dimensional_rms']==pytest.approx(math.sqrt(2))
    assert row['normalized_rms']==pytest.approx(.2*math.sqrt(2))
    assert row['normalized_max']>.39
    assert row['estimator']=='analytic manufactured'


@pytest.mark.parametrize('call',[
    lambda:magnetic_energy_density([1.,2.],mu=1.),
    lambda:magnetic_energy_density([1.,2.,3.],mu=0.),
    lambda:wave_speeds(density=1.,pressure=1.,gamma=1.4,field=[0.,0.,0.],normal=[2.,0.,0.],mu=1.),
    lambda:divergence_summary([0.],length_scale=1.,volume=1.,reference_field=0.,estimator='analytic'),
    lambda:divergence_summary([0.],length_scale=1.,volume=1.,reference_field=1.,estimator=''),
])
def test_invalid_units_vectors_and_diagnostic_conventions_are_rejected(call):
    with pytest.raises(ValueError):call()
