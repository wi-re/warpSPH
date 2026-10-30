"""Analytic controls for the explicit dynGPUSPH static AC-2L path."""
from types import SimpleNamespace
import pytest
import torch
from warpSPHCore import buildVerletList,SupportScheme,get_torch_precision
from warpSPH.configurations import ArtificialCompressibleSPHConfig
from warpSPH.configurations.simulationConfig import SimulationConfig
from warpSPH.systems.artificialCompressible import ArtificialCompressibleState
from warpSPH.utils.domain import buildDomainDescription
from warpSPH.modules.artificialCompressible import dyngpusphOperators as ops
from warpSPH.modules.artificialCompressible.workingState import workingState


def fixture(x,k):
    dtype=get_torch_precision();x=torch.tensor(x,dtype=dtype);n=len(x)
    state=ArtificialCompressibleState(positions=x,velocities=torch.zeros_like(x),
        pressures=torch.zeros(n,dtype=dtype),densities=torch.full((n,),1000.,dtype=dtype),
        supports=torch.full((n,),.003,dtype=dtype),masses=torch.full((n,),1000*.00075**2,dtype=dtype),
        kinds=torch.tensor(k,dtype=torch.int32),materials=torch.zeros(n,dtype=torch.int32),
        UIDs=torch.arange(n,dtype=torch.int32),UIDcounter=n)
    config=SimulationConfig(device=torch.device('cpu'),dtype=dtype,dim=2,dx=.00075,
        domain=buildDomainDescription(l=1.,dim=2,periodic=False,device='cpu',dtype=dtype))
    adj=buildVerletList(state,config.domain,verletScale=config.verletScale,supportMode=SupportScheme.Gather,verbose=False)
    return state,config,adj


def test_reference_support_quality_does_not_use_identity_fallback(runtime):
    for x in ([[0.,0.]],[[0.,0.],[.00075,0.],[0.,.00075]]):
        st,c,a=fixture(x,[0]*len(x))
        lam,normal,grad,lap=ops.moments(st,c,a)
        assert bool((lam<.1).all())
        if len(x)==1:assert lam[0]==0
        torch.testing.assert_close(grad,torch.zeros_like(grad),rtol=0,atol=0)


def test_wall_pressure_recovers_static_hydrostatics(runtime):
    st,c,a=fixture([[0.,0.],[0.,.00075],[.00075,.00075]],[1,0,0])
    st.pressures[1:]=1000*9.81*(.2-st.positions[1:,1])
    g=torch.zeros_like(st.velocities);g[:,1]=-9.81
    closed,*_=ops.closures(st,c,a,torch.ones_like(st.pressures),torch.zeros_like(st.pressures),g)
    torch.testing.assert_close(closed[0],closed.new_tensor(1000*9.81*.2),rtol=2e-6,atol=1e-6)


def test_wall_viscosity_preserves_query_tangential_velocity(runtime):
    st,c,a=fixture([[0.,.00075],[.00075,0.]],[0,1])
    st.velocities[0,0]=4
    normal=torch.zeros_like(st.velocities);normal[1,1]=1
    average=torch.zeros_like(st.velocities);average[1,0]=-2
    zero=torch.zeros_like(st.pressures);zv=torch.zeros_like(st.velocities)
    terms=ops.residual_terms(st,c,a,normal,average,zero,zv,zv,zero,zero)
    # DAMPING_SLIP gives the wall the query's tangent: no physical viscosity.
    # Continuity still uses the distinct mirrored Shepard velocity.
    torch.testing.assert_close(terms[2][0],zv[0],rtol=0,atol=0)
    assert abs(float(terms[0][0]))>0


def test_native_pressure_residual_is_an_independent_gate(runtime):
    from warpSPH.modules.artificialCompressible.dyngpusphReference import residual_metrics
    v=torch.zeros((2,2),dtype=get_torch_precision());rp=torch.ones(2,dtype=v.dtype)
    eps=residual_metrics(v,rp,v,torch.arange(2),2,.01,1000.,10.)
    assert torch.isneginf(eps[0])
    expected=torch.log10(rp.double().square().sum().sqrt()/2*.01/(1000*.01*100))
    torch.testing.assert_close(eps[1],expected,rtol=0,atol=0)
    assert not bool((eps[0]<-8)&(eps[1]<-7))


def test_first_two_clock_and_destination_rk_factors(runtime):
    from warpSPH.modules.artificialCompressible.dyngpusphReference import iteration
    from warpSPHIntegrators.schemes.coefficients.explicit import getButcherTableau
    st,c,a=fixture([[0.,0.]],[0]);sc=ArtificialCompressibleSPHConfig()
    sc.acParams.dtOverDtau=2.;sc.fluid.fixedSoundSpeed=10.
    body=torch.ones_like(st.velocities);body[:,1]=0
    dt=.01
    problem=SimpleNamespace(state=st,config=c,schemeConfig=sc,dtau=dt/2,
        bdf=(150.,-200.,50.),history=(st.positions,st.positions,st.velocities,st.velocities),
        positionOrigin=None,tableau=getButcherTableau('SSPRK3'),bodyForce=body,
        nu=torch.zeros_like(st.pressures),fluidRows=(st.kinds==0)[:,None],fluid=st.kinds==0,
        fluidIndex=torch.arange(1),nFluid=1,bounce=None)
    for warm in (True,False):
        result=iteration(problem,st.positions,st.velocities,st.pressures,st.pressures,
            st.velocities,st.pressures,st.pressures,warmup=warm,initialize_switch=True,
            cell_positions=st.positions,adjacency=a)
        dtau=dt/(4 if warm else 2)
        weights=problem.tableau.b
        expected=dtau*sum(float(b)/(1+float(b)*dtau/dt) for b in weights)
        torch.testing.assert_close(result[1][0],st.velocities.new_tensor([expected,0.]),rtol=2e-6,atol=1e-10)
        tracked=iteration(problem,st.positions,st.velocities,st.pressures,st.pressures,
            st.velocities,st.pressures,st.pressures,warmup=warm,initialize_switch=True,
            cell_positions=st.positions,adjacency=a,track_motion=True)
        for a0,b0 in zip(result,tracked[:10]):
            torch.testing.assert_close(a0,b0,rtol=0,atol=0)
        assert tracked[-1]>=0


def test_reference_and_isolation_config_roundtrip(runtime):
    from warpSPH.configurations.artificialCompressible import ArtificialCompressibilityParams,_acParamsToDict,_dictToAcParams
    p=ArtificialCompressibilityParams(referenceModel='dyngpusph-static-2d',isolatedZeroPressure=True)
    q=_dictToAcParams(_acParamsToDict(p))
    assert q.referenceModel==p.referenceModel and q.isolatedZeroPressure


@pytest.mark.parametrize('motion_over_r,rejected',[(.6,False),(.8,True)])
def test_solve_checks_intermediate_motion_before_committing_surface(runtime,monkeypatch,motion_over_r,rejected):
    from warpSPH.modules.artificialCompressible import dyngpusphReference as ref
    st,c,a=fixture([[0.,0.]],[0]);sc=ArtificialCompressibleSPHConfig()
    sc.acParams.incrementalPositions=False
    sc.acParams.maxPseudoIterations=sc.acParams.minPseudoIterations=1
    sc.acParams.epsilonV=-8.
    old_surface=(st.pressures,st.velocities,st.pressures)
    sc._dyngpusphSurface=old_surface
    problem=SimpleNamespace(state=st,config=c,schemeConfig=sc,bounce=None,positionOrigin=None)
    monkeypatch.setattr(ref,'validate',lambda _:None)
    def excursion(problem,x,v,p,lam,norm,edges,switch,**kw):
        # Intermediate RK excursion returns to the entry coordinates; checking
        # only the accepted endpoint would miss the invalid geometry.
        return x,v,p,lam,norm,edges,switch,p.new_tensor([-9.,-9.],dtype=torch.float64),v,p,p.new_tensor(motion_over_r*.003)
    monkeypatch.setattr(ref,'iteration',excursion)
    if rejected:
        with pytest.raises(RuntimeError,match='certified candidate motion'):
            ref.solve(problem,st.positions,st.velocities,st.pressures)
        assert sc._dyngpusphSurface is old_surface
    else:
        result=ref.solve(problem,st.positions,st.velocities,st.pressures)
        torch.testing.assert_close(result.positions,st.positions,rtol=0,atol=0)
        assert sc._dyngpusphMotionDiagnostics['limit']==pytest.approx(.75*.003)


def test_reference_finalize_keeps_static_wall_coordinates_bitwise(runtime,monkeypatch):
    from warpSPH.systems import artificialCompressible as systems
    from warpSPH.modules.artificialCompressible.dualTime import DualTimeResult,exactDeltaUpdate
    st,c,a=fixture([[0.,.00075],[-.00075,0.],[1.6095,0.]],[0,1,1])
    sc=ArtificialCompressibleSPHConfig();sc.acParams.referenceModel='dyngpusph-static-2d'
    sc.shiftProperties.active=False
    sc.rigidBodies=[SimpleNamespace(angularVelocity=0.,linearVelocity=torch.zeros(2))]
    system=systems.ArtificialCompressibleSystem(state=st,domain=c.domain,adjacency=a)
    original=st.positions.clone()
    def reconstructed(*args,**kwargs):raise AssertionError('Static reference walls must not pass through COM reconstruction')
    monkeypatch.setattr(systems,'updateBodyParticlesWCSPH',reconstructed)
    monkeypatch.setattr(systems,'integrateRigidBody',reconstructed)
    result=DualTimeResult(st.positions,st.velocities,st.pressures,1,-9.)
    update=exactDeltaUpdate(result,st,1e-4,2)
    system.finalize(system,1e-4,[(a,st)],[update],[1.],config=c,schemeConfig=sc)
    assert torch.equal(system.state.positions,original)
