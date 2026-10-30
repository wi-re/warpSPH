"""Physical pair checks for external full-field Taylor transport."""
import math
import torch
from warpSPHCore import get_torch_precision,OperationProperties,WarpOperation,SupportScheme,OperationDirection,GradientScheme,warpOperation
from warpSPH.systems.artificialCompressible import ArtificialCompressibleState
from warpSPH.configurations.simulationConfig import SimulationConfig
from warpSPH.utils.domain import buildDomainDescription
from warpSPH.modules.shifting.transport import taylorShiftFields


def pair():
    dtype=get_torch_precision();x=torch.tensor([[0.,0.],[.00075,0.]],dtype=dtype);n=2
    st=ArtificialCompressibleState(positions=x,velocities=torch.zeros_like(x),
        pressures=torch.full((n,),100.,dtype=dtype),supports=torch.full((n,),.003,dtype=dtype),
        masses=torch.full((n,),1000*.00075**2,dtype=dtype),densities=torch.full((n,),1000.,dtype=dtype),
        kinds=torch.zeros(n,dtype=torch.int32),materials=torch.zeros(n,dtype=torch.int32),
        UIDs=torch.arange(n,dtype=torch.int32),UIDcounter=n)
    config=SimulationConfig(device=torch.device('cpu'),dtype=dtype,dim=2,dx=.00075,
        domain=buildDomainDescription(l=1.,dim=2,periodic=False,device='cpu',dtype=dtype))
    return st,config


def test_taylor_remap_includes_off_diagonal_velocity_gradient(runtime):
    st,c=pair();st.velocities[1]=st.velocities.new_tensor([2.,3.]);st.pressures[1]=2100.
    dx=torch.tensor([[-1e-5,0.],[-2e-5,0.]],dtype=st.positions.dtype)
    old=(st.positions.clone(),st.densities.clone(),st.velocities.clone(),st.pressures.clone())
    p,v=taylorShiftFields(st,dx,c,None)
    G=(7/(math.pi*.003**3))*20*.25*(1-.25)**3*.00075**2
    torch.testing.assert_close(v[0],dx.new_tensor([-1e-5*2*G,-1e-5*3*G]),rtol=2e-5,atol=1e-8)
    torch.testing.assert_close(p[0],p.new_tensor(100-1e-5*2000*G),rtol=2e-5,atol=2e-5)
    assert v[0,1]!=0 # diagonal-only source remap misses this term
    for name,before in zip(('positions','densities','velocities','pressures'),old):
        torch.testing.assert_close(getattr(st,name),before,rtol=0,atol=0)


def test_constant_fields_and_stationary_wall_outputs_preserved(runtime):
    st,c=pair();st.velocities[:]=st.velocities.new_tensor([2.,-3.]);st.kinds[1]=1
    dx=torch.full_like(st.positions,.0001)
    p,v=taylorShiftFields(st,dx,c,None)
    torch.testing.assert_close(p,st.pressures,rtol=0,atol=0)
    torch.testing.assert_close(v,st.velocities,rtol=0,atol=0)
    st.pressures[1]=2100.;st.velocities[1]=st.velocities.new_tensor([8.,9.])
    p,v=taylorShiftFields(st,dx,c,None)
    torch.testing.assert_close(p[1],st.pressures[1],rtol=0,atol=0)
    torch.testing.assert_close(v[1],st.velocities[1],rtol=0,atol=0)


def test_pressure_transport_matches_linear_eos_density_transport(runtime):
    st,c=pair();st.densities[1]=1001.;c2=100.
    st.pressures=c2*(st.densities-1000.)
    dx=torch.full_like(st.positions,1e-5)
    p,_=taylorShiftFields(st,dx,c,None,velocity=False)
    grad=warpOperation(st,operationProperties=OperationProperties(operation=WarpOperation.Gradient,
        kernel=c.kernel,supportMode=SupportScheme.Gather,operationMode=OperationDirection.AllToAll,
        gradientMode=GradientScheme.Difference),queryValues=st.densities,domain=c.domain,adjacency=None)
    # Compare increments to avoid the separate cancellation in rho'-rho0.
    torch.testing.assert_close(p-st.pressures,c2*(grad*dx).sum(-1),rtol=1e-4,atol=5e-6)


def test_trial_sequence_is_remap_then_bdf_with_all_other_controls_fixed():
    from scripts.acsph_reference.michel_trials import rows
    cohort=rows();assert len(cohort)==3
    for a,b,key in zip(cohort,cohort[1:],('full_remap','bdf_shift')):
        assert [k for k in a['options'] if a['options'][k]!=b['options'][k]]==[key]
    assert not cohort[0]['options']['bdf_shift'] and not cohort[1]['options']['bdf_shift']


def test_trial_receipt_uses_serializable_configuration_schema(runtime):
    import json
    from scripts.acsph_reference.michel_trials import trial_parameters
    from warpSPH.configurations import ArtificialCompressibleSPHConfig
    scheme=ArtificialCompressibleSPHConfig()
    scheme.acParams.bdfShiftCorrection=False
    values=trial_parameters(scheme)
    restored=json.loads(json.dumps(values))
    assert restored['pressureSmoothing']==scheme.acParams.pressureSmoothing.name
    assert restored['bdfShiftCorrection'] is False
