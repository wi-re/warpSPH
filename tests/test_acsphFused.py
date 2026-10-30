"""Frozen-state parity for the fused ACSPH stage pair loop."""
import pytest
import torch
from types import SimpleNamespace
from warpSPHCore import (get_torch_precision,OperationProperties,OperationDirection,
    WarpOperation,GradientScheme,SupportScheme,warpOperation,buildVerletList)
from warpSPH.modules.artificialCompressible.residual import spatialResidual
from test_artificialCompressible import buildSystem
from warpSPH.enumTypes import PressureForceScheme
from warpSPH.modules.artificialCompressible.residual import fusedResidualTerms
from warpSPH.configurations import ArtificialCompressibleSPHConfig
from warpSPH.configurations.simulationConfig import SimulationConfig
from warpSPH.utils.domain import buildDomainDescription


@pytest.mark.parametrize('term',list(PressureForceScheme))
@pytest.mark.parametrize('use_list',[False,True])
def test_fused_spatial_residual_matches_existing_operators(runtime,term,use_list):
    system,config,scheme=buildSystem(torch.device('cpu'),get_torch_precision())
    state=system.state;n=len(state.positions)
    state.velocities=torch.sin(9*state.positions)
    state.pressures=1000*torch.sin(4*state.positions[:,0])*torch.cos(3*state.positions[:,1])
    state.surfaceIndicators=(torch.arange(n)%3==0).to(torch.int32)
    # Include solid and ignored ghost references, varying support/density.
    state.kinds[::7]=1;state.kinds[::11]=2
    state.densities=1000+100*torch.sin(3*state.positions[:,1])
    state.masses=state.densities*.01
    state.supports*=1+.1*torch.sin(3*state.positions[:,0])
    # No rigid-body mirror geometry in this operator fixture. Existing helper
    # sees stored ghosts and returns the already-prepared velocity field.
    scheme.pressureForceTerm=term
    adjacency=buildVerletList(state,config.domain,supportMode=SupportScheme.SuperSymmetric) if use_list else None
    nu=torch.full_like(state.pressures,.001)
    inputs=(state,config,scheme,adjacency,torch.ones_like(nu)*100,
        torch.ones_like(nu)*.01,torch.cos(state.pressures),torch.zeros_like(state.positions),nu)
    scheme.acParams.fusedResidual=False
    expected=spatialResidual(*inputs)
    scheme.acParams.fusedResidual=True
    actual=spatialResidual(*inputs)
    tol=2e-5 if get_torch_precision()==torch.float32 else 2e-12
    for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=tol,atol=tol)


def test_fused_acoustic_gradient_preserves_the_prior_fluid_divergence_pass(runtime):
    system,config,scheme=buildSystem(torch.device('cpu'),get_torch_precision())
    state=system.state
    state.velocities=torch.sin(9*state.positions)
    state.kinds[::7]=1;state.kinds[::11]=2
    nu=torch.full_like(state.pressures,.001)
    props=lambda operation,mode:OperationProperties(kernel=config.kernel,
        operation=operation,operationMode=OperationDirection.FluidToFluid,
        supportMode=SupportScheme.SuperSymmetric,gradientMode=mode)
    div=warpOperation(state,props(WarpOperation.Divergence,GradientScheme.Difference),
        queryValues=state.velocities,domain=config.domain)
    expected=warpOperation(state,props(WarpOperation.Gradient,GradientScheme.Summation),
        queryValues=div,domain=config.domain)
    actual=fusedResidualTerms(state,config,scheme,None,nu,fluidDiv=div)[-1]
    tol=2e-5 if get_torch_precision()==torch.float32 else 2e-12
    torch.testing.assert_close(actual,expected,rtol=tol,atol=tol)


def test_fused_residual_preserves_velocity_and_pressure_adjoints(runtime):
    system,config,scheme=buildSystem(torch.device('cpu'),get_torch_precision())
    state=system.state
    state.velocities=torch.sin(3*state.positions).requires_grad_()
    state.pressures=torch.sin(4*state.positions[:,0]).requires_grad_()
    nu=torch.full_like(state.pressures,.001)
    inputs=(state,config,scheme,None,torch.ones_like(nu)*100,
        torch.ones_like(nu)*.01,torch.cos(state.pressures).detach(),torch.zeros_like(state.positions),nu)
    gradients=[]
    for fused in (False,True):
        scheme.acParams.fusedResidual=fused
        rp,rv=spatialResidual(*inputs)
        gradients.append(torch.autograd.grad(rp.square().mean()+rv.square().mean(),
            (state.velocities,state.pressures)))
    tol=1e-4 if get_torch_precision()==torch.float32 else 5e-12
    for a,b in zip(*gradients):torch.testing.assert_close(a,b,rtol=tol,atol=tol)


def test_three_dimensional_periodic_pair_sums(runtime):
    dtype=get_torch_precision();axis=torch.tensor([-.1,.1],dtype=dtype)
    x=torch.stack(torch.meshgrid(axis,axis,axis,indexing='ij'),-1).reshape(-1,3)
    state=SimpleNamespace(positions=x,supports=torch.full((8,),.5,dtype=dtype),
        masses=torch.ones(8,dtype=dtype),densities=torch.full((8,),1000.,dtype=dtype),
        kinds=torch.zeros(8,dtype=torch.int32),velocities=torch.sin(9*x),
        pressures=1000*torch.sin(x[:,0]),surfaceIndicators=None)
    config=SimulationConfig(device=torch.device('cpu'),dtype=dtype,dim=3,dx=.2,
        domain=buildDomainDescription(l=.5,dim=3,periodic=True,device='cpu',dtype=dtype))
    scheme=ArtificialCompressibleSPHConfig();nu=torch.full((8,),.001,dtype=dtype)
    args=(state,config,scheme,None,nu*1000,nu,nu,torch.zeros_like(x),nu)
    scheme.acParams.fusedResidual=False;expected=spatialResidual(*args)
    scheme.acParams.fusedResidual=True;actual=spatialResidual(*args)
    tol=2e-5 if dtype==torch.float32 else 2e-12
    for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=tol,atol=tol)
