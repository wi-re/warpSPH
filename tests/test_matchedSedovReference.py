"""Similarity jump/energy identities and normalized common Sedov input."""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from compare_spheralSedov import sample,time_grid,oracle,exact


def test_normalized_deposit_pressure_balance_symmetry_and_time_sequence():
    for nx in (64,128):
        a=sample(nx);m=a['masses'];u=a['internalEnergies'];background=1.e-6/(5/3-1)
        np.testing.assert_allclose(np.dot(m,u-background),1.,rtol=2.e-15)
        np.testing.assert_allclose(a['pressures'],(5/3-1)*a['densities']*u,rtol=1.e-15)
        np.testing.assert_allclose(u.reshape(nx,nx),u.reshape(nx,nx).T,rtol=0,atol=1.e-12)
        np.testing.assert_allclose(u.reshape(nx,nx),u.reshape(nx,nx)[::-1],rtol=0,atol=1.e-12)
        assert np.unique(a['materials']).tolist()==[0];assert u.min()>0
        dt=time_grid(nx);assert np.all(dt>0)
        np.testing.assert_allclose(np.cumsum(dt),.05*(np.arange(1,len(dt)+1)/len(dt))**2,atol=1.e-16)
        half=time_grid(nx,True);np.testing.assert_allclose(half.reshape(-1,2).sum(1),dt,atol=0)


def test_similarity_strong_shock_jump_and_radius_scaling():
    answer,_=oracle();g=answer.gamma
    speed,R,v,rho,P=answer.shockState(.05)
    np.testing.assert_allclose(speed,.5*R/.05,rtol=1.e-15)
    np.testing.assert_allclose(answer.shockState(.2)[1],2*R,rtol=1.e-15)
    np.testing.assert_allclose(speed,rho*(speed-v),rtol=2.e-15)
    np.testing.assert_allclose(speed**2,P+rho*(speed-v)**2,rtol=2.e-15)
    np.testing.assert_allclose(.5*speed**2,g/(g-1)*P/rho+.5*(speed-v)**2,rtol=2.e-15)


def test_similarity_physical_energy_and_scalar_native_profiles():
    answer,_=oracle();lam=np.asarray(answer._lam);z,w=np.polynomial.legendre.leggauss(4)
    q=lam[:-1,None]+.5*np.diff(lam)[:,None]*(1+z);R=answer.shockState(.05)[1]
    a,_=exact((q*R).ravel(),.05)
    integrand=(a['pressure']/(5/3-1)+.5*a['density']*a['velocity']**2).reshape(q.shape)
    energy=2*np.pi*R**2*np.sum(integrand*q*.5*np.diff(lam)[:,None]*w)
    np.testing.assert_allclose(energy,1.,rtol=3.e-14)
    radii=np.array([.007,.05,.15,.23,.27,.4]);expected=answer.solution(.05,radii)
    fields,_=exact(radii,.05)
    for key,value in [('velocity',expected[1]),('energy',expected[2]),('density',expected[3]),('pressure',expected[4])]:
        np.testing.assert_allclose(fields[key],value,rtol=1.e-13,atol=1.e-15)
