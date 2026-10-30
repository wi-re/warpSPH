"""Physical pair oracles for the named dynGPUSPH shift/transport mode."""
import math
import torch
from warpSPHCore import get_torch_precision
from warpSPH.systems.artificialCompressible import ArtificialCompressibleState
from warpSPH.configurations.simulationConfig import SimulationConfig
from warpSPH.utils.domain import buildDomainDescription
from warpSPH.modules.artificialCompressible.dyngpusphShifting import shift_and_remap


def fixture(positions, kinds):
    dtype = get_torch_precision()
    x = torch.tensor(positions,dtype=dtype);n=len(x)
    state = ArtificialCompressibleState(positions=x,velocities=torch.zeros_like(x),
        pressures=torch.full((n,),100.,dtype=dtype),supports=torch.full((n,),.003,dtype=dtype),
        masses=torch.full((n,),1000*.00075**2,dtype=dtype),densities=torch.full((n,),1000.,dtype=dtype),
        kinds=torch.tensor(kinds,dtype=torch.int32),materials=torch.zeros(n,dtype=torch.int32),
        UIDs=torch.arange(n,dtype=torch.int32),UIDcounter=n)
    config = SimulationConfig(device=torch.device('cpu'),dtype=dtype,dim=2,dx=.00075,
        domain=buildDomainDescription(l=1.,dim=2,periodic=False,device='cpu',dtype=dtype))
    return state,config


def evaluate(state,config,dt=1e-5,lam=None,edge=None):
    n=len(state.positions)
    return shift_and_remap(state,config,dt=dt,
        lambdas=torch.ones(n,dtype=state.positions.dtype) if lam is None else state.positions.new_tensor(lam),
        edges=torch.zeros(n,dtype=torch.int32) if edge is None else torch.tensor(edge,dtype=torch.int32),
        normals=state.positions.new_tensor([[0.,1.]]).expand(n,2).contiguous())


def test_isolated_particle_is_stationary_and_has_zero_pressure(runtime):
    state,config=fixture([[0.,0.]],[0])
    shift,recorded,p,v,active,*_=evaluate(state,config)
    torch.testing.assert_close(shift,torch.zeros_like(shift),rtol=0,atol=0)
    assert p.item()==0 and active.item()==0
    torch.testing.assert_close(v,state.velocities,rtol=0,atol=0)


def test_wall_collision_uses_mass_weighting_without_fluid_neighbors(runtime):
    state,config=fixture([[0.,.0001],[0.,0.]],[0,1])
    state.masses[1]*=3
    shift,recorded,p,v,active,*_=evaluate(state,config)
    expected=.75*(.9*.00075-.0001)
    torch.testing.assert_close(shift[0],shift.new_tensor([0.,expected]),atol=1e-10,rtol=2e-6)
    # Collision is applied beyond the regular shift cap; the reference does
    # not update its shift-history record on this no-fluid-neighbor branch.
    assert expected>.05*.0015
    torch.testing.assert_close(recorded,torch.zeros_like(recorded),rtol=0,atol=0)
    assert active[0]==0 and p[0]==100
    torch.testing.assert_close(shift[1],torch.zeros_like(shift[1]),rtol=0,atol=0)


def test_pair_pressure_remap_and_reference_diagonal_velocity_rule(runtime):
    state,config=fixture([[0.,0.],[.00075,0.]],[0,0])
    state.velocities[1]=state.velocities.new_tensor([2.,3.])
    state.pressures[1]=2100.
    shift,recorded,p,v,active,gp,gv,_=evaluate(state,config)
    # The raw concentration exceeds the native cap: -0.2 U dt *0.5 R/dx.
    dx=-.2*2.*1e-5*.5*4.
    # Analytic Wendland-C2 gradient at r=dx, compact support R=4dx.
    R=.003;q=.25;G=(7/(math.pi*R**3))*20*q*(1-q)**3*.00075**2
    torch.testing.assert_close(shift[0],shift.new_tensor([dx,0.]),atol=2e-10,rtol=1e-5)
    assert abs(float(p[0])-(100+dx*2000*G))<3e-5
    assert abs(float(v[0,0])-dx*2*G)<2e-7
    # Full Taylor transport would also change vy here. The supplied compiled
    # reference only transports diagonal derivatives; keep that quirk named.
    assert v[0,1]==0
    torch.testing.assert_close(gp[0],gp.new_tensor([2000*G,0.]),atol=.05,rtol=1e-5)
    torch.testing.assert_close(gv[0],gv.new_tensor([2*G,0.]),atol=1e-3,rtol=1e-5)
    assert active[0]==1


def test_surface_cohesion_is_displacement_not_dt_scaled_velocity(runtime):
    state,config=fixture([[0.,0.],[.00075,0.]],[0,0])
    first=evaluate(state,config,dt=1e-5,lam=[.5,.9],edge=[1,0])
    second=evaluate(state,config,dt=7e-5,lam=[.5,.9],edge=[1,0])
    torch.testing.assert_close(first[0],second[0],rtol=0,atol=0)
    torch.testing.assert_close(first[0][0],state.positions.new_tensor([.01*.0015,0.]),atol=1e-10,rtol=1e-5)


def test_ghost_rows_never_enter_shift_or_field_transport(runtime):
    state,config=fixture([[0.,0.],[.00075,0.]],[0,0])
    state.velocities[1,0]=2.
    expected=evaluate(state,config)
    extended,config=fixture([[0.,0.],[.00075,0.],[.0001,.0001]],[0,0,2])
    extended.velocities[:2]=state.velocities
    extended.velocities[2]=1e6;extended.pressures[2]=-1e9
    got=evaluate(extended,config)
    for x,y in zip(expected,got):torch.testing.assert_close(x,y[:2],atol=1e-10,rtol=1e-6)


def test_reference_interior_pressure_outlier_guard_is_retained(runtime):
    state,config=fixture([[0.,0.],[.00075,0.],[-.00075,0.]],[0,0,0])
    state.pressures[0]=1000.;state.pressures[1:]=0.
    shift,_,p,*_=evaluate(state,config)
    assert p[0]==0
    torch.testing.assert_close(shift[0],torch.zeros_like(shift[0]),atol=0,rtol=0)


def test_vicinity_tag_two_is_excluded_from_interior_lambda_average(runtime):
    state,config=fixture([[0.,0.],[.00075,0.],[-.00075,0.]],[0,0,0])
    *_,wide=evaluate(state,config,lam=[.5,.5,.9],edge=[2,1,0])
    assert abs(float(wide[0])-.9)<1e-6


def test_wrapped_reference_stencil_multiplies_gradients_and_collision_weight(runtime):
    state,config=fixture([[0.,0.],[.00075,0.]],[0,0])
    state.velocities[1]=state.velocities.new_tensor([2.,3.]);state.pressures[1]=2100.
    once=evaluate(state,config)
    three=shift_and_remap(state,config,dt=1e-5,lambdas=torch.ones_like(state.pressures),
        edges=torch.zeros_like(state.kinds),normals=torch.zeros_like(state.positions),pair_multiplicity=3)
    torch.testing.assert_close(three[5],3*once[5],atol=.05,rtol=1e-6)
    torch.testing.assert_close(three[6],3*once[6],atol=1e-3,rtol=1e-6)
    state,config=fixture([[0.,.0001],[0.,0.]],[0,1])
    three=shift_and_remap(state,config,dt=1e-5,lambdas=torch.ones_like(state.pressures),
        edges=torch.zeros_like(state.kinds),normals=torch.zeros_like(state.positions),pair_multiplicity=3)
    torch.testing.assert_close(three[0][0],state.positions.new_tensor([0.,.75*(.9*.00075-.0001)]),atol=1e-10,rtol=1e-6)


def test_reference_surface_isolated_and_interior_wall_aware(runtime):
    from warpSPH.modules.artificialCompressible.dyngpusphShifting import reference_surface
    state,config=fixture([[0.,0.]],[0])
    lam,normal,edges=reference_surface(state,state,config)
    assert lam.item()==0 and edges.item()==1
    assert normal.abs().max().item()==0
    coords=[[i*.00075,j*.00075] for i in range(-6,7) for j in range(-6,7)]
    state,config=fixture(coords,[0]*len(coords))
    lam,normal,edges=reference_surface(state,state,config)
    center=84
    assert .9<lam[center].item()<1.1 and edges[center].item()==0
    assert edges[0].item()==1
