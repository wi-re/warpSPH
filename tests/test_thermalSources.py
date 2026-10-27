"""Volume source work/cooling and prescribed thermal state are separate exchanges."""
from dataclasses import replace
import pytest
import torch
from test_drivenSchemes import configuration,initialEnergy
from warpSPH.cases.compressible.drivenTurbulence import drivenTurbulenceCase as case
from warpSPH.configurations import BoundaryCondition,BoundaryConditionType
from warpSPHCore.type_config import get_torch_precision


@pytest.mark.parametrize('scheme',['CRKSPH','CompSPH','Monaghan','GSPH','InutsukaGSPH'])
def test_uniform_acceleration_and_cooling_close_with_boundary_projection(scheme):
    saved={}
    def build(ctx):
        system=case.buildSystem(ctx);s=system.state
        s.internalEnergies.fill_(1.)
        system.sourceTerms.acceleration=lambda pos,t:pos.new_tensor([1.,0.]).expand_as(pos)
        system.sourceTerms.removeMeanAcceleration=False
        ctx.schemeConfig.boundaryConditions=[BoundaryCondition(BoundaryConditionType.dynamic,
            sdf=lambda x:(-torch.ones_like(x[:,0]),torch.zeros_like(x)),
            dirichletFunctions={'internalEnergies':lambda s,*args:torch.ones_like(s.masses)})]
        initialEnergy(ctx,system);saved['energy']=ctx.scratch['initialEnergy'].clone()
        return system
    result=configuration(scheme,10).run(replace(case,buildSystem=build))
    s=result.state.state;t=float(result.state.t)
    tol=2e-10 if get_torch_precision() is torch.float64 else 2e-5
    torch.testing.assert_close(s.internalEnergies,torch.ones_like(s.masses),rtol=tol,atol=tol)
    torch.testing.assert_close(s.velocities,s.velocities.new_tensor([t,0.]).expand_as(s.velocities),rtol=tol,atol=tol)
    final=(s.masses.double()*(s.internalEnergies.double()+.5*s.velocities.double().square().sum(-1))).sum()
    totalExchange=s.sourceEnergyBudget.sum()+s.boundaryProjectionBudget.sum()
    torch.testing.assert_close(final-saved['energy'],totalExchange,rtol=0,atol=tol)
    assert s.sourceEnergyBudget[1]<0 and s.boundaryProjectionBudget[0]>0
    assert not result.diverged


@pytest.mark.parametrize('scheme',['CRKSPH','CompSPH','Monaghan'])
def test_prescribed_u_refreshes_each_material_eos(scheme):
    from warpSPH.runner import RunConfiguration
    from warpSPH.cases.compressible.hydrostatic import hydrostaticCase
    def build(ctx):
        system=hydrostaticCase.buildSystem(ctx)
        ctx.schemeConfig.boundaryConditions=[BoundaryCondition(BoundaryConditionType.dynamic,
            sdf=lambda x:(-torch.ones_like(x[:,0]),torch.zeros_like(x)),
            dirichletFunctions={'internalEnergies':lambda s,*args:torch.full_like(s.masses,1.2)},
            updateFunctions={'dxdt':lambda s,*args:torch.zeros_like(s.velocities),
                             'dvdt':lambda s,*args:torch.zeros_like(s.velocities)})]
        return system
    cfg=RunConfiguration('hydrostatic',simulation=dict(device='cpu',nx=12,nSteps=2,maxDt=.001,
        plot=False,video=False,store=False,show=False,quiet=True,progress=False),
        params=dict(rho_high=4.,viscositySwitch='NoneSwitch'),solver=dict(scheme=scheme,materials=[
            {'name':'background','eos':{'type':'idealGas','gamma':1.4}},
            {'name':'inclusion','eos':{'type':'idealGas','gamma':5/3}}]))
    result=cfg.run(replace(hydrostaticCase,buildSystem=build));s=result.state.state
    gamma=torch.where(s.materials==0,torch.full_like(s.masses,1.4),torch.full_like(s.masses,5/3))
    tol=2e-10 if get_torch_precision() is torch.float64 else 4e-6
    assert torch.unique(s.materials).numel()==2 and not result.diverged
    torch.testing.assert_close(s.internalEnergies,torch.full_like(s.masses,1.2),rtol=tol,atol=tol)
    torch.testing.assert_close(s.pressures,(gamma-1)*s.densities*s.internalEnergies,rtol=tol,atol=tol)
    torch.testing.assert_close(s.soundspeeds,(gamma*(gamma-1)*s.internalEnergies).sqrt(),rtol=tol,atol=tol)


@pytest.mark.parametrize('target',[1.,1e6])
def test_thermal_initialization_respects_the_startup_ceiling_and_hot_state_limit(target):
    from warpSPH.runner import RunConfiguration
    from warpSPH.cases.compressible.sod import sodCase
    def build(ctx):
        system=sodCase.buildSystem(ctx);ctx.config.dt=1e-4
        ctx.schemeConfig.boundaryConditions=[BoundaryCondition(BoundaryConditionType.dynamic,
            sdf=lambda x:(-torch.ones_like(x[:,0]),torch.zeros_like(x)),
            dirichletFunctions={'internalEnergies':lambda s,*args:torch.full_like(s.masses,target)})]
        return system
    cfg=RunConfiguration('sod',simulation=dict(device='cpu',nx=16,nSteps=1,maxDt=.002,
        plot=False,video=False,store=False,show=False,quiet=True,progress=False),
        params=dict(samplingRatio=1,left_rho=1.,right_rho=1.,left_pressure=.4,right_pressure=.4),solver=dict(scheme='Monaghan'))
    result=cfg.run(replace(sodCase,buildSystem=build));time=float(result.state.t)
    assert time<=1e-4+1e-11 and not result.diverged
    if target==1.:assert time==pytest.approx(1e-4,rel=1e-6)
    else:assert time<1e-4
