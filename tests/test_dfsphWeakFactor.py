"""Sparse wall-contact regression for weak DFSPH pressure factors."""
from types import SimpleNamespace as NS
import pytest
import torch


def sparse_contact(rho0):
    from warpSPHCore import DomainDescription, KernelFunctions, SupportScheme, buildVerletList
    from warpSPHCore.type_config import get_torch_precision
    from warpSPH.configurations import IISPHConfig, DFSPHConfig
    from warpSPH.modules.iisph import computeIISPHCoefficients
    dtype = get_torch_precision()
    walls = torch.cartesian_prod(torch.arange(-4, 5, dtype=dtype)*.01,
                                 torch.arange(4, dtype=dtype)*.01)
    x = torch.cat((torch.tensor([[0., -.03914857], [.00502634, .00153273]], dtype=dtype), walls))
    n = len(x)
    state = NS(positions=x.contiguous(), supports=torch.full((n,), .04, dtype=dtype),
        masses=torch.full((n,), rho0*1.e-4, dtype=dtype),
        kinds=torch.cat((torch.zeros(2, dtype=torch.int32), torch.ones(n-2, dtype=torch.int32))),
        densities=torch.full((n,), rho0, dtype=dtype), velocities=torch.zeros_like(x))
    state.densities[0] *= .179
    state.velocities[0, 1] = 3.7
    config = NS(domain=DomainDescription(torch.full((2,), -.1, dtype=dtype),
        torch.full((2,), .1, dtype=dtype), torch.zeros(2, dtype=torch.bool), 2),
        kernel=KernelFunctions.Wendland4, calibrateNormalization=False)
    # Isolate factor protection from the independent density-recovery gate.
    scheme = NS(fluid=NS(restDensity=rho0), iisph=IISPHConfig(), dfsph=DFSPHConfig(minDivergenceDensityRatio=0.))
    contacts=x.clone(); contacts[:, 1] = -.005
    adjacency=buildVerletList(state, config.domain, supportMode=SupportScheme.SuperSymmetric)
    geometry=computeIISPHCoefficients(state, config, scheme, adjacency, contactPoints=contacts)
    return state,config,scheme,adjacency,geometry,contacts


def project(fixture, dt, mode, *, cutoff=1.e-5, iterations=1, warm=False):
    from warpSPH.modules.dfsph import solveDFSPHProjection
    state,config,scheme,adjacency,g,contacts=fixture
    scheme.dfsph.weakFactorThreshold=cutoff
    policy=getattr(scheme.dfsph,mode)
    policy.minIterations=policy.maxIterations=iterations
    policy.clampPressure=mode=='divergence'
    policy.relaxationFactor=.25
    impulse=torch.full_like(state.masses,scheme.fluid.restDensity*1.e8) if warm else None
    return solveDFSPHProjection(state,config,scheme,dt,adjacency,*g[:5],
        mode=mode,velocity=state.velocities,pressureImpulse=impulse,
        contactPoints=contacts,contactAdjacency=g[5],contactDenominator=g[6])


@pytest.mark.parametrize('rho0,dt', [(1., .00032), (1000., .00032), (1., 1.e-6)])
@pytest.mark.parametrize('iterations', [1,100])
def test_weakFactorCutoffStopsPressureTransmissionAcrossWallContact(rho0,dt,iterations):
    f=sparse_contact(rho0)
    raw=project(f,dt,'divergence',cutoff=0.,iterations=iterations)
    protected=project(f,dt,'divergence',iterations=iterations)
    # The two fluid particles have no direct kernel connection. The huge
    # velocity change comes entirely from wall-contact pressure interpolation.
    assert (f[0].positions[0]-f[0].positions[1]).norm()>f[0].supports[0]
    assert raw.pressure[0]/rho0>1.e6
    assert (dt*raw.acceleration[1]).norm()>10.
    assert protected.pressure[0]==0
    torch.testing.assert_close(protected.acceleration[1],torch.zeros_like(protected.acceleration[1]),rtol=0,atol=0)


@pytest.mark.parametrize('mode',['density','divergence'])
@pytest.mark.parametrize('dt',[.00032,1.e-8])
def test_cutoffClearsWeakWarmHistoryButKeepsStrongRowsActive(mode,dt):
    f=sparse_contact(1000.)
    # Keep the supported row compressive so the ordinary warm-start gate
    # permits it. Weak-factor gating must independently clear the weak row.
    f[0].velocities[1]=f[4][2][1]
    result=project(f,dt,mode,warm=True)
    assert result.pressure[0]==0
    assert result.pressure[1]!=0  # The supported row's warm start still participates.


@pytest.mark.parametrize('value',[-1.,float('nan'),float('inf')])
def test_weakFactorThresholdRejectsInvalidConfiguration(value):
    from warpSPH.configurations import DFSPHConfig
    cfg=DFSPHConfig(weakFactorThreshold=value)
    with pytest.raises(ValueError,match='weakFactorThreshold'):
        cfg.validate()


def test_weakFactorThresholdRoundTrips():
    from warpSPH.configurations.incompressible import IncompressibleSPHConfig,incompressibleConfigToDict,dictToIncompressibleSPHConfig
    cfg=IncompressibleSPHConfig()
    cfg.dfsph.weakFactorThreshold=2.e-5
    restored=dictToIncompressibleSPHConfig(incompressibleConfigToDict(cfg))
    assert restored.dfsph.weakFactorThreshold==2.e-5


def test_weakFactorIsMeasuredBeforeDensityRatioScaling():
    from warpSPH.modules.iisph import computeIISPHCoefficients
    f=list(sparse_contact(1.))
    f[0].densities[0]=1.e-6
    f[4]=computeIISPHCoefficients(f[0],f[1],f[2],f[3],contactPoints=f[5])
    assert abs(f[4][4][0])>1.e-5  # Physical-pressure diagonal alone looks strong.
    result=project(f,.00032,'divergence')
    assert result.pressure[0]==0
    torch.testing.assert_close(result.acceleration[1],torch.zeros_like(result.acceleration[1]),rtol=0,atol=0)
