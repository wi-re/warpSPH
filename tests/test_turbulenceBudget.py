"""Independent budget, sign, dimensional and conservative-observation controls."""
import numpy as np
import pytest

from warpSPH.utils.turbulenceBudget import PeriodicBudget, depositConservative
from warpSPH.utils.turbulence import depositKernel2D


def coordinates(dim, n=24, length=1.):
    return np.meshgrid(*[(np.arange(n)+.5)*length/n for _ in range(dim)], indexing='ij')


@pytest.mark.parametrize('dim', [2, 3])
def test_physical_length_derivatives_and_gaussian_response(dim):
    lengths=np.arange(1, dim+1)*1.3
    shape=tuple(16+2*j for j in range(dim)); engine=PeriodicBudget(shape,lengths)
    x=np.meshgrid(*[(np.arange(n)+.5)*l/n for n,l in zip(shape,lengths)],indexing='ij')
    field=sum(np.sin(2*np.pi*xj/l) for xj,l in zip(x,lengths))
    for j in range(dim):
        expected=2*np.pi/lengths[j]*np.cos(2*np.pi*x[j]/lengths[j])
        np.testing.assert_allclose(engine.derivative(field,j),expected,atol=3e-14,rtol=0)
    ell=.11
    expected=sum(np.sin(2*np.pi*xj/l)*np.exp(-.5*(ell*2*np.pi/l)**2) for xj,l in zip(x,lengths))
    np.testing.assert_allclose(engine.filter(field,ell),expected,atol=3e-15,rtol=0)
    nyquist=np.broadcast_to((-1.)**np.arange(shape[0]).reshape((-1,)+(1,)*(dim-1)),shape)
    np.testing.assert_allclose(engine.derivative(nyquist,0),0,atol=1e-14,rtol=0)


@pytest.mark.parametrize('dim', [2, 3])
def test_uniform_force_work_and_zero_transfer(dim):
    shape=(12,)*dim;engine=PeriodicBudget(shape,length=2.)
    rho=np.full(shape,1.7);p=np.full(shape,2.1)
    velocity=np.broadcast_to(np.arange(1,dim+1)*.2,shape+(dim,))
    acceleration=np.broadcast_to(np.arange(dim)*.1+.3,velocity.shape)
    result=engine.measure(rho,velocity,p,ell=.2,forceDensity=rho[...,None]*acceleration)
    for name in ('deformationTransfer','baropycnalTransfer','pressureDilatation','viscousWork'):
        assert abs(result[name])<2e-14
    assert result['forcingWork']==pytest.approx(1.7*np.dot(velocity[(0,)*dim],acceleration[(0,)*dim]))


@pytest.mark.parametrize('dim', [2, 3])
def test_analytic_pressure_dilatation_sign(dim):
    x=coordinates(dim);shape=x[0].shape;engine=PeriodicBudget(shape)
    wave=2*np.pi;amplitude=.4;pressureAmplitude=.3;ell=.09
    v=np.zeros(shape+(dim,));v[...,0]=amplitude*np.sin(wave*x[0])
    p=1+pressureAmplitude*np.cos(wave*x[0]);rho=np.ones(shape)
    result=engine.measure(rho,v,p,ell=ell)
    expected=.5*amplitude*pressureAmplitude*wave*np.exp(-(ell*wave)**2)
    assert result['pressureDilatation']==pytest.approx(expected,abs=2e-14)
    assert abs(result['totalTransfer'])<2e-14
    drho,dmom=engine.inviscidRHS(rho,v,p)
    assert engine.kineticRate(rho,v,drho,dmom,ell)==pytest.approx(expected,abs=2e-14)


@pytest.mark.parametrize('dim', [2, 3])
def test_signed_triad_transfer_matches_independent_mode_energy_rates(dim):
    x=coordinates(dim);shape=x[0].shape;engine=PeriodicBudget(shape)
    k=2*np.pi;a,b,c=.7,.4,.3;ell=.1
    v=np.zeros(shape+(dim,));mixed=k*(x[0]+2*x[1])
    v[...,0]=b*np.sin(2*k*x[1])+c*np.sin(mixed)
    v[...,1]=a*np.sin(k*x[0])-.5*c*np.sin(mixed)
    # Direct trigonometric advection gives energy rates at n²=1,4,5:
    # k*a*b*c * (-1/8, +1/2, -3/8). Gaussian energy weights are G².
    weights=np.exp(-ell**2*k**2*np.array([1,4,5]))
    expectedRate=k*a*b*c*np.dot(np.array([-1/8,1/2,-3/8]),weights)
    assert expectedRate<0
    forward=engine.measure(np.ones(shape),v,np.ones(shape),ell=ell)
    reverse=engine.measure(np.ones(shape),-v,np.ones(shape),ell=ell)
    assert forward['deformationTransfer']==pytest.approx(-expectedRate,abs=2e-14)
    assert reverse['deformationTransfer']==pytest.approx(expectedRate,abs=2e-14)
    assert abs(forward['baropycnalTransfer'])<2e-14


def manufactured(dim):
    x=coordinates(dim);shape=x[0].shape
    rho=1+.12*np.sin(2*np.pi*x[0])+.08*np.cos(2*np.pi*x[1])
    v=np.stack([np.sin(2*np.pi*x[(j+1)%dim])+.2*np.cos(4*np.pi*x[j]) for j in range(dim)],-1)
    p=1+.2*np.cos(2*np.pi*(x[0]+x[1]))
    if dim==3:
        rho+=.03*np.sin(2*np.pi*x[2]);p+=.1*np.cos(2*np.pi*x[2])
    force=rho[...,None]*np.stack([.1*np.cos(2*np.pi*x[j]) for j in range(dim)],-1)
    force-=force.mean(axis=tuple(range(dim)),keepdims=True)
    return rho,v,p,force


@pytest.mark.parametrize('dim', [2, 3])
def test_compressible_filtered_euler_budget_and_galilean_transfer(dim):
    rho,v,p,force=manufactured(dim);engine=PeriodicBudget(rho.shape)
    drho,dmom=engine.inviscidRHS(rho,v,p,force)
    for ell in (.05,.1,.2):
        original=engine.measure(rho,v,p,ell=ell,forceDensity=force,returnFields=True)
        boosted=engine.measure(rho,v+np.arange(1,dim+1)*.37,p,ell=ell,forceDensity=force,returnFields=True)
        assert engine.kineticRate(rho,v,drho,dmom,ell)==pytest.approx(original['predictedRate'],abs=3e-11)
        for key in ('deformationTransfer','baropycnalTransfer','pressureDilatation'):
            scale=max(1.,float(np.max(abs(original['fields'][key]))))
            np.testing.assert_allclose(original['fields'][key],boosted['fields'][key],atol=256*np.finfo(float).eps*scale,rtol=1e-12)
        assert original['forcingWork']==pytest.approx(boosted['forcingWork'],abs=2e-14)
        if ell==.1:
            assert abs(original['baropycnalTransfer'])>1e-6


@pytest.mark.parametrize('dim', [2, 3])
def test_prescribed_viscous_shear_work_is_separate_and_correct(dim):
    x=coordinates(dim);shape=x[0].shape;engine=PeriodicBudget(shape)
    k=2*np.pi;a=.6;mu=.03;ell=.08
    rho=np.ones(shape);p=np.ones(shape);v=np.zeros(shape+(dim,));v[...,1]=a*np.sin(k*x[0])
    sigma=np.zeros(shape+(dim,dim));sigma[...,1,0]=sigma[...,0,1]=mu*a*k*np.cos(k*x[0])
    result=engine.measure(rho,v,p,ell=ell,stress=sigma)
    expected=.5*mu*(a*k)**2*np.exp(-(ell*k)**2)
    assert result['viscousWork']==pytest.approx(expected,abs=2e-14)
    drho,dmom=engine.inviscidRHS(rho,v,p,stress=sigma)
    assert engine.kineticRate(rho,v,drho,dmom,ell)==pytest.approx(-expected,abs=2e-14)
    assert result['predictedRate']==pytest.approx(-expected,abs=2e-14)


@pytest.mark.parametrize('dim', [2, 3])
def test_conservative_observation_maps_extensive_budgets(dim):
    x=np.stack(coordinates(dim,n=8),-1).reshape(-1,dim)
    rng=np.random.default_rng(91);x+=rng.uniform(-.02,.02,x.shape)
    m=rng.uniform(.5,1.5,len(x))/len(x);rho=rng.uniform(.7,1.3,len(x))
    v=rng.normal(size=x.shape);a=rng.normal(size=x.shape);p=rng.uniform(.8,1.2,len(x));u=rng.uniform(.6,1.,len(x));heat=rng.normal(size=len(x))
    mapped=depositConservative(x,m,v,rho,p,u,a,gridSize=8,radius=.25,heatRate=heat)
    meta=mapped['metadata'];assert meta['emptyCellFraction']==0
    assert meta['maxExtensiveConservationError']<1e-13
    dx=1/8
    np.testing.assert_allclose(mapped['pressure'].sum()*dx**dim,np.sum(m*p/rho),atol=2e-15)
    np.testing.assert_allclose(mapped['internalEnergyDensity'].sum()*dx**dim,np.sum(m*u),atol=2e-15)
    np.testing.assert_allclose(mapped['sourceWorkDensity'].sum()*dx**dim,np.sum(m*np.sum(v*a,axis=-1)),atol=2e-15)
    np.testing.assert_allclose(mapped['heatDensity'].sum()*dx**dim,np.sum(m*heat),atol=2e-15)
    assert meta['unresolvedKineticEnergy']>=0
    assert meta['resolvedKineticEnergy']+meta['unresolvedKineticEnergy']==pytest.approx(.5*np.sum(m*np.sum(v*v,axis=-1)),abs=2e-15)
    if dim==2:
        legacy=depositKernel2D(x,m,v,np.full_like(m,.25),gridSize=8)
        np.testing.assert_allclose(mapped['density'],legacy['density'].T,atol=3e-15)
        np.testing.assert_allclose(mapped['velocity'],legacy['velocity'].transpose(1,0,2),atol=3e-15)


def test_invalid_inputs_are_not_repaired():
    engine=PeriodicBudget((8,8));rho=np.ones((8,8));v=np.zeros((8,8,2))
    with pytest.raises(ValueError,match='positive'):
        engine.measure(-rho,v,rho,ell=.1)
    with pytest.raises(ValueError,match='ell'):
        engine.measure(rho,v,rho,ell=-.1)
    with pytest.raises(ValueError,match='shape'):
        engine.measure(rho,np.zeros((8,8,3)),rho,ell=.1)
    with pytest.raises(ValueError,match='finite'):
        engine.measure(rho,v,rho*np.nan,ell=.1)


def test_cli_accepts_a_three_dimensional_retained_state_format(tmp_path):
    """Synthetic observations exercise the full3D adapter; no flow is evolved."""
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys
    import torch
    from warpSPH.modules.forcing import FourierOU
    run=tmp_path/'three-dimensional-fixture';(run/'samples').mkdir(parents=True)
    config=dict(case='analysisFixture',simulation=dict(dim=3,L=1.,nx=8),params=dict(
        gamma=5/3,soundSpeed=1.,thermalModel='adiabatic',removeMeanAcceleration=True))
    (run/'input.json').write_text(json.dumps(config))
    field=FourierOU(dim=3,accelerationRMS=.3,correlationTime=.5,updateInterval=.05)
    field.coefficients(1.05);torch.save(field.state_dict(),run/'forcing.pt')
    axes=coordinates(3,n=8);x=np.stack(axes,-1).reshape(-1,3);mass=np.full(len(x),1/len(x))
    for j,t in enumerate(np.linspace(0,1,9)):
        velocity=np.column_stack((.2*np.sin(2*np.pi*x[:,1]),(.2+.01*t)*np.sin(2*np.pi*x[:,2]),.1*np.cos(2*np.pi*x[:,0])))
        pressure=1+.1*np.cos(2*np.pi*x[:,0])
        np.savez_compressed(run/'samples'/f'{j:05d}.npz',t=t,positions=x,masses=mass,densities=np.ones(len(x)),
            velocities=velocity,pressures=pressure,internalEnergies=pressure/(5/3-1),supports=np.full(len(x),.1))
    script=Path(__file__).resolve().parents[1]/'scripts/analyze_turbulenceBudget.py'
    output=tmp_path/'analysis'
    child=subprocess.run([sys.executable,str(script),str(run),'--output',str(output),'--grids','8',
        '--radii','.25','--scales','.125','.25','--window','.25','1'],
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True)
    assert child.returncode==0,child.stdout+child.stderr
    result=json.loads((output/run.name/'summary.json').read_text())
    assert result['dim']==3 and result['samples']==9
    assert len(result['observations'][0]['scales'])==2
    assert result['observations'][0]['maxExtensiveConservationError']<1e-13
    assert result['observations'][0]['maxMassError']<1e-13
    assert 'Warp ' not in child.stdout, 'Offline CLI should not initialize the hydrodynamic runtime'


def test_accepted_sampling_counts_missed_targets_and_never_repeats_states():
    from warpSPH.utils.turbulenceBudget import AcceptedSampleClock
    clock=AcceptedSampleClock(.1)
    times=[0.,0.,.07,.21,.21,.22,.49,.5]
    observed=[t for t in times if clock.due(t)]
    assert observed==[0.,.21,.49,.5]
    assert clock.count==4 and clock.missedTargets==2
    with pytest.raises(ValueError,match='chronological'):
        clock.due(.4)
    with pytest.raises(ValueError,match='positive'):
        AcceptedSampleClock(0.)


def test_sampling_cli_rejects_invalid_interval_before_runtime_or_output(tmp_path):
    import os
    from pathlib import Path
    import subprocess
    import sys
    root=tmp_path/'must-not-exist'
    script=Path(__file__).resolve().parents[1]/'scripts/qualify_drivenTurbulence.py'
    child=subprocess.run([sys.executable,str(script),'--sample-interval','nan','--root',str(root)],
                         env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),text=True,capture_output=True)
    assert child.returncode==2 and '--sample-interval must be finite and positive' in child.stderr
    assert not root.exists() and 'Warp ' not in child.stdout


def test_observation_stress_work_closes_independent_momentum_flux():
    # A spatially varying symmetric kinetic covariance; expected work derived
    # from -div(tau), independently of measure's gradient-contraction formula.
    n=32;engine=PeriodicBudget((n,n));x,y=np.meshgrid(np.arange(n)/n,np.arange(n)/n,indexing='ij')
    rho=1+.1*np.cos(2*np.pi*x);p=np.ones_like(x)
    v=np.stack((.2*np.sin(2*np.pi*x),.1*np.cos(2*np.pi*y)),-1)
    tau=np.zeros((n,n,2,2));tau[...,0,0]=.03+.01*np.cos(2*np.pi*x);tau[...,1,1]=.02
    flux=rho[...,None,None]*v[..., :,None]*v[...,None,:]+tau
    ell=.08;drho,dmom=engine.inviscidRHS(rho,v,p)
    for i in range(2):
        for j in range(2):dmom[...,i]-=engine.derivative(tau[...,i,j],j)
    r=engine.measure(rho,v,p,ell=ell,observationMomentumFlux=flux)
    nativeRate=engine.kineticRate(rho,v,drho,dmom,ell)
    assert r['observationStressWork']>0
    assert nativeRate==pytest.approx(r['predictedObservedRate'],abs=2e-16)
    # A boost changes C but not its covariance or periodic stress work.
    boost=np.array([.7,-.4]);vb=v+boost
    fb=flux+rho[...,None,None]*(v[..., :,None]*boost[None,:]+boost[:,None]*v[...,None,:]+boost[:,None]*boost[None,:])
    rb=engine.measure(rho,vb,p,ell=ell,observationMomentumFlux=fb)
    assert rb['observationStressWork']==pytest.approx(r['observationStressWork'],abs=3e-16)


def test_deposited_second_moments_and_unresolved_energy_are_consistent():
    rng=np.random.default_rng(5);x=rng.uniform(0,1,(64,3));v=rng.normal(size=x.shape);m=rng.uniform(.1,1,64);ones=np.ones(64)
    g=depositConservative(x,m,v,ones,ones,ones,np.zeros_like(v),gridSize=8,radius=.4)
    # Integral of second moment, including each cross component.
    expected=np.einsum('n,ni,nj->ij',m,v,v)
    np.testing.assert_allclose(g['momentumFlux'].sum(axis=(0,1,2))/8**3,expected,atol=1e-13)
    kinetic=.5*np.trace(g['momentumFlux'],axis1=-2,axis2=-1).sum()/8**3
    assert kinetic==pytest.approx(g['metadata']['particleKineticEnergy'],abs=1e-13)
    tau=g['momentumFlux']-g['density'][...,None,None]*g['velocity'][..., :,None]*g['velocity'][...,None,:]
    unresolved=.5*np.trace(tau,axis1=-2,axis2=-1).sum()/8**3
    assert unresolved==pytest.approx(g['metadata']['unresolvedKineticEnergy'],abs=1e-13)
