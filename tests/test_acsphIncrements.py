"""BDF and compensated position increments: sub-ULP motion is physical state."""
from types import SimpleNamespace
import copy
import pytest
import torch
from warpSPHCore import get_torch_precision
from warpSPH.modules.artificialCompressible.dualTime import bdfSource
from warpSPH.modules.artificialCompressible.dualTime import (DualTimeResult,exactDeltaUpdate,
    reconcilePositionUpdate)
from warpSPH.systems.artificialCompressible import bdfCoefficients
from test_artificialCompressible import buildSystem
from warpSPHIntegrators import get_scheme
import warpSPH.schemes.incompressible.artificialCompressible as acmod


def test_increment_bdf_is_zero_at_rest_despite_large_coordinates(runtime):
    dtype=get_torch_precision()
    x=torch.full((10,2),.6,dtype=dtype);v=torch.full_like(x,8.)
    problem=SimpleNamespace(bdf=bdfCoefficients(1.2e-5,1e-5)[:3],
        history=(x,x,v,v),positionOrigin=x,previousPositionStep=torch.zeros_like(x))
    dx,dv=bdfSource(problem,torch.zeros_like(x),v)
    assert torch.count_nonzero(dx)==0
    assert torch.count_nonzero(dv)==0


def test_increment_bdf_reproduces_unequal_step_constant_velocity(runtime):
    dtype=get_torch_precision();v=torch.tensor([[.001,-.002]],dtype=dtype)
    dt,prev=1.2e-5,1e-5;x=torch.tensor([[.6,.3]],dtype=dtype)
    problem=SimpleNamespace(bdf=bdfCoefficients(dt,prev)[:3],history=(x,x,v,v),
        positionOrigin=x,previousPositionStep=v*prev)
    dx,dv=bdfSource(problem,v*dt,v)
    torch.testing.assert_close(dx,v,rtol=1e-6,atol=1e-12)
    torch.testing.assert_close(dv,torch.zeros_like(v),rtol=0,atol=0)


def test_update_boundary_condition_can_cancel_sub_ulp_displacement(runtime):
    dtype=get_torch_precision();high=torch.full((2,2),.6,dtype=dtype)
    step=torch.full_like(high,1e-9);target=high+step;low=(high-target)+step
    state=SimpleNamespace(positions=high,velocities=torch.zeros_like(high),
                          pressures=torch.zeros(2,dtype=dtype))
    result=DualTimeResult(target,state.velocities,state.pressures,1,-8.,step,low)
    update=exactDeltaUpdate(result,state,1e-5,2)
    update.dxdt[0]=0.
    reconcilePositionUpdate(update,1e-5)
    torch.testing.assert_close(update.positionStep[0],torch.zeros(2,dtype=dtype),rtol=0,atol=1e-15)
    represented=update.positionTarget.double()+update.positionLow.double()
    torch.testing.assert_close(represented[0],high.double()[0],rtol=0,atol=1e-15)
    torch.testing.assert_close(represented[1],high.double()[1]+step.double()[1],rtol=0,atol=1e-15)


@pytest.mark.parametrize('correct_shift',[True,False])
def test_shift_is_retained_without_becoming_physical_bdf_motion(runtime,monkeypatch,correct_shift):
    from warpSPH.systems import artificialCompressible as systems
    system,config,scheme=buildSystem(torch.device('cpu'),get_torch_precision())
    config.dt=1e-5;scheme.acParams.incrementalPositions=True
    scheme.acParams.bdfShiftCorrection=correct_shift
    scheme.gravityConfig.active=False;scheme.surfaceDetectionConfig.active=False
    scheme.shiftProperties.active=True
    shift=torch.full_like(system.state.positions,1e-9)
    monkeypatch.setattr(systems,'solveShifting',lambda **kwargs:shift)
    x0=system.state.positions.clone()
    result=get_scheme(config.integrationScheme).function(state=system,
        f=acmod.artificialCompressible_step,dt=config.dt,config=config,
        schemeConfig=scheme,verbose=False).state
    represented=result.state.positions.double()+result.positionLow.double()
    torch.testing.assert_close(represented,x0.double()+shift.double(),rtol=0,atol=1e-15)
    expected=torch.zeros_like(shift) if correct_shift else shift
    torch.testing.assert_close(result.positionStep,expected,rtol=0,atol=1e-15)


def test_sub_ulp_motion_accumulates_across_real_steps(runtime):
    dtype=get_torch_precision()
    system,config,scheme=buildSystem(torch.device('cpu'),dtype)
    config.dt=1e-5
    scheme.acParams.incrementalPositions=True
    scheme.acParams.epsilonV=-8
    scheme.acParams.nu=0.
    scheme.acParams.referenceSoundSpeedForViscosity=None
    scheme.shiftProperties.active=False
    scheme.gravityConfig.active=False
    scheme.surfaceDetectionConfig.active=False
    velocity=torch.tensor([.001,0.],dtype=dtype)
    system.state.velocities[:]=velocity
    x0=system.state.positions.clone()
    scheme.acParams.maxPseudoIterations=120
    # At x~1 this per-step displacement rounds to zero in float32. Keep the
    # increment and low coordinate part, so the aggregate does not disappear.
    for _ in range(12):
        system=get_scheme(config.integrationScheme).function(state=system,
            f=acmod.artificialCompressible_step,dt=config.dt,config=config,
            schemeConfig=scheme,verbose=False).state
        if _==5:checkpoint=copy.deepcopy(system)
    expected=x0.double()+12*config.dt*velocity.double()
    high=system.state.positions.double()
    represented=high+system.positionLow.double()
    assert (represented-expected).abs().max()<2e-10
    torch.testing.assert_close(system.positionStep,
        (config.dt*velocity).expand_as(x0),rtol=2e-4,atol=1e-12)
    cloned=system.initializeNewState()
    assert cloned.positionStep is system.positionStep
    assert cloned.positionLow is system.positionLow
    for _ in range(6):
        checkpoint=get_scheme(config.integrationScheme).function(state=checkpoint,
            f=acmod.artificialCompressible_step,dt=config.dt,config=config,
            schemeConfig=scheme,verbose=False).state
    for name in ('positions','velocities','pressures'):
        torch.testing.assert_close(getattr(checkpoint.state,name),getattr(system.state,name),rtol=0,atol=0)
    torch.testing.assert_close(checkpoint.positionStep,system.positionStep,rtol=0,atol=0)
    torch.testing.assert_close(checkpoint.positionLow,system.positionLow,rtol=0,atol=0)
