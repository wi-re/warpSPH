"""A continuous thermal partition must balance power before time integration."""
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from crkThermalRate import entropy_fraction,thermal_entropy_rate


def test_pair_partition_closes_power_with_unequal_mass_and_cold_boundary():
    t=lambda x:torch.tensor(x,dtype=torch.float64)
    adjacency=NS(paddedI=torch.tensor([0,1]),paddedJ=torch.tensor([1,0]),numNeighbors=torch.tensor([1,1]))
    m=t([2.,.3]);force=t([[.7,-.2],[-.7,.2]])
    for entropy in ([0.,3.],[1.e-30,3.],[2.,2.],[.2,8.],[0.,0.]):
        for sign in (-1.,1.):
            v=sign*t([[1.,.4],[-.5,.1]])
            du=thermal_entropy_rate(adjacency,force/m[:,None],t(entropy),v)
            residual=torch.dot(m,du)+(force*v).sum()
            assert abs(float(residual))<1.e-14
            if entropy[0]==0. and entropy[1]>0.:assert float(du[0])>=0
    # Complementarity, including the all-zero entropy fallback, is scale-free.
    si=t([0.,1.e-30,2.,0.]);sj=t([3.,1.e-25,2.,0.]);power=t([-1.,1.,0.,-1.])
    torch.testing.assert_close(entropy_fraction(si,sj,power)+entropy_fraction(sj,si,power),torch.ones_like(si),atol=2.e-16,rtol=0)


def test_plain_midpoint_has_second_order_energy_and_solution_error():
    from scipy.integrate import solve_ivp
    adjacency=NS(paddedI=torch.tensor([0,1]),paddedJ=torch.tensor([1,0]),numNeighbors=torch.tensor([1,1]))
    mass=np.array([1.,.3]);rho=np.array([1.,2.]);g=5/3
    initial=np.array([.3,-.2,1.,3.]);end=.4
    def reference(t,y):
        v,u=y[:2],y[2:];P=(g-1)*rho*u;force=np.array([-.4*P.sum(),.4*P.sum()]);power=-(v*force).sum()
        s=P/rho**g;fraction=s[::-1]/s.sum() if power>=0 else s/s.sum()
        return np.r_[force/mass,fraction*power/mass]
    exact=solve_ivp(reference,(0,end),initial,rtol=2.e-13,atol=1.e-14,method='DOP853').y[:,-1]
    def production(y):
        v,u=y[:2],y[2:];P=(g-1)*rho*u;force=np.array([-.4*P.sum(),.4*P.sum()])
        du=thermal_entropy_rate(adjacency,torch.tensor((force/mass)[:,None]),torch.tensor(P/rho**g),torch.tensor(v[:,None])).numpy()
        return np.r_[force/mass,du]
    E=lambda y:np.dot(mass,y[2:]+.5*y[:2]**2)
    errors=[];drifts=[]
    for n in (10,20,40):
        y=initial.copy();h=end/n
        for _ in range(n):y+=h*production(y+.5*h*production(y))
        errors.append(np.linalg.norm(y-exact));drifts.append(abs(E(y)-E(initial)))
        assert min(y[2:])>0
    assert np.all(np.array(errors[:-1])/errors[1:]>3.7),errors
    assert np.all(np.array(drifts[:-1])/drifts[1:]>3.7),drifts


def test_runner_exposes_direct_thermal_mode_and_rejects_compatible_controls():
    import pytest
    from warpSPH.runner.configuration import _validateSolver
    from warpSPH.configurations.crkSPH import CRKSPHConfig
    assert _validateSolver('CRKSPH',{'compatibleEnergy':False})=={'compatibleEnergy':False}
    c=CRKSPHConfig(compatibleEnergy=False);c.validate()
    c.crkEnergyPartition='entropy'
    with pytest.raises(ValueError,match='require compatibleEnergy'):c.validate()


def test_production_eq78_can_cool_a_zero_energy_particle_in_expansion():
    import pytest
    import warp as wp
    from warpSPHCore import (get_torch_precision,ParticleState,DomainDescription,SupportScheme,buildVerletList,
        CRKState,OperationProperties,KernelFunctions)
    if get_torch_precision()!=torch.float64:pytest.skip('Operator diagnosis requires float64 runtime')
    wp.config.kernel_cache_dir='/tmp/warpsph-thermal-test-kernels'
    from warpSPH.configurations import CRKSPHConfig
    from warpSPH.configurations.crkSPH import resolveCRKLimiter
    from warpSPH.modules.crk.accel import computeCrkSPHAccelWarp
    from warpSPH.modules.crk.dudt import computeCrkSPHdudtWarp
    t=lambda x:torch.tensor(x,dtype=torch.float64)
    state=ParticleState(t([[-.15,0],[.15,0]]),t([1.,1.]),t([1.,1.]),torch.zeros(2,dtype=torch.int32),t([1.,1.]))
    state.velocities=t([[-.1,0],[.1,0]]);state.internalEnergies=t([0.,3.]);state.pressures=t([0.,2.])
    state.soundspeeds=t([0.,np.sqrt(10/3)]);state.alphas=t([1.,1.]);state.materials=torch.zeros(2,dtype=torch.int32)
    domain=DomainDescription(t([-2.,-2.]),t([2.,2.]),torch.tensor([False,False]),2)
    adjacency=buildVerletList(state,domain,supportMode=SupportScheme.SuperSymmetric,verletScale=1.01)
    coeff=CRKState(A=t([1.,1.]),B=torch.zeros((2,2),dtype=torch.float64),gradA=torch.zeros((2,2),dtype=torch.float64),gradB=torch.zeros((2,2,2),dtype=torch.float64))
    config=CRKSPHConfig();config.diffusionParams.C_l=config.diffusionParams.C_q=0.
    limiter=resolveCRKLimiter(config.crkViscosityParams,4.);limiter.oneSidedWork=True
    args=dict(queryParticles=state,domain=domain,adjacency=adjacency,queryEnergies=state.internalEnergies,
        queryVelocities=state.velocities,queryCs=state.soundspeeds,queryAlphas=state.alphas,queryPressures=state.pressures,
        queryVolumes=t([.5,.5]),queryVelocityTensor=torch.zeros((2,2,2),dtype=torch.float64),crkState=coeff,
        crkViscosityParams=limiter,conductivityParams=config.diffusionParams,
        operationProperties=OperationProperties(kernel=KernelFunctions.CubicSpline,supportMode=SupportScheme.KernelMeanSymmetric))
    acc,ap,av=computeCrkSPHAccelWarp(**args);du=computeCrkSPHdudtWarp(**args)
    assert float(du[0])<0 and abs(float(du[1]))<1.e-14
    assert abs(float(torch.dot(state.masses,du)+(state.masses[:,None]*state.velocities*acc).sum()))<1.e-14
    alternate=thermal_entropy_rate(adjacency,ap+av,state.pressures,state.velocities)
    assert float(alternate[0])==0 and float(alternate[1])<0
