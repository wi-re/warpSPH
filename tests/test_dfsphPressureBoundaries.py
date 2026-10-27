"""Independent Band Eq. 8–20 oracle for the native two-constraint extension."""
import pytest
import torch
from test_projectionModules import _nativeIisphFixture, _denseIisphKernels


@pytest.mark.parametrize('dim,rho0', [(2, 1.), (2, 1000.), (3, 1000.)])
def test_extended_force_continuity_diagonal_and_input_ownership(dim, rho0):
    from warpSPH.modules.dfsph.pressureBoundaries import buildPressureBoundaryGeometry
    f = _nativeIisphFixture('cpu', dim, False, rho0=rho0)
    st, cfg = f.state, f.config
    before = {k:(v, v.clone()) for k,v in vars(st).items() if torch.is_tensor(v)}
    g = buildPressureBoundaryGeometry(st, cfg, f.scheme, f.adjacency)
    W, grad = _denseIisphKernels(st, cfg)
    V0 = st.masses/rho0
    s = W@V0
    V = V0/s
    torch.testing.assert_close(g.volumes, V, rtol=4e-6, atol=4e-6)
    n = len(V)
    def force(q):
        a = -torch.einsum('ij,ijd,j->id', q[:,None]+q[None,:], grad, V)/s[:,None]
        return torch.where(g.fluid[:,None], a, 0)
    def rate(v):
        return torch.einsum('ijd,ijd,j->i', v[:,None]-v[None,:], grad, V)
    q = torch.linspace(.1, .8, n, dtype=V.dtype)
    torch.testing.assert_close(g.acceleration(cfg,f.adjacency,q), force(q), rtol=6e-6, atol=6e-6)
    torch.testing.assert_close(g.rate(cfg,f.adjacency,st.positions), rate(st.positions), rtol=6e-6, atol=6e-6)
    eye = torch.eye(n,dtype=V.dtype)
    A = torch.stack([rate(force(e)) for e in eye],1)
    torch.testing.assert_close(g.diagonal, A.diagonal(), rtol=8e-6, atol=8e-6)
    assert g.rows[~g.fluid].all()
    assert g.acceleration(cfg,f.adjacency,q)[~g.fluid].count_nonzero()==0
    only_wall = torch.where(g.fluid, 0, q)
    assert g.acceleration(cfg,f.adjacency,only_wall)[g.fluid].norm()>0
    for key,(original,value) in before.items():
        assert getattr(st,key) is original
        assert torch.equal(getattr(st,key),value)


@pytest.mark.parametrize('mode', ['density','divergence'])
@pytest.mark.parametrize('dt', [.01,.02])
def test_extended_signed_jacobi_matches_dense_equations(mode, dt):
    from warpSPH.configurations import DFSPHConfig
    from warpSPH.modules.dfsph.pressureBoundaries import buildPressureBoundaryGeometry
    from warpSPH.modules.dfsph import solveDFSPHProjection
    f = _nativeIisphFixture('cpu', 2, False)
    f.scheme.dfsph = DFSPHConfig(pressureBoundary='band2018')
    p = getattr(f.scheme.dfsph,mode)
    p.clampPressure=False;p.minIterations=p.maxIterations=4;p.relaxationFactor=.15
    st,cfg,adj = f.state,f.config,f.adjacency
    g = buildPressureBoundaryGeometry(st,cfg,f.scheme,adj)
    v = torch.where(g.fluid[:,None], -.1*st.positions, 0)
    scale = dt*dt if mode=='density' else dt
    A = torch.stack([g.apply(cfg,adj,e) for e in torch.eye(len(st.masses),dtype=st.masses.dtype)],1)
    source = -g.rate(cfg,adj,v)
    if mode=='density':source=1-g.ratios+dt*source
    q = torch.zeros_like(source)
    for _ in range(4):q=q+p.relaxationFactor*(source-scale*(A@q))/(scale*A.diagonal())
    z=torch.zeros_like(st.positions)
    result = solveDFSPHProjection(st,cfg,f.scheme,dt,adj,g.restVolumes,g.ratios,z,z,g.diagonal,
        mode=mode,velocity=v,pressureBoundaryGeometry=g)
    torch.testing.assert_close(result.source,source,rtol=3e-6,atol=3e-6)
    torch.testing.assert_close(result.pressure/f.scheme.fluid.restDensity,q,rtol=8e-6,atol=8e-6)


def test_extended_boundary_rejects_analytic_and_moving_samples():
    from warpSPH.modules.dfsph.pressureBoundaries import buildPressureBoundaryGeometry
    f = _nativeIisphFixture('cpu',2,False)
    with pytest.raises(ValueError,match='sampled'):
        buildPressureBoundaryGeometry(f.state,f.config,f.scheme,f.adjacency,analyticProvider=object())
    f.state.velocities[-1]=1
    with pytest.raises(ValueError,match='static'):
        buildPressureBoundaryGeometry(f.state,f.config,f.scheme,f.adjacency)


def test_sampling_gate_pins_fluid_pressure_but_retains_neighbour_force():
    from warpSPH.configurations import DFSPHConfig
    from warpSPH.modules.dfsph.pressureBoundaries import buildPressureBoundaryGeometry,solvePressureBoundaryProjection
    f=_nativeIisphFixture('cpu',2,False)
    f.scheme.dfsph=DFSPHConfig(pressureBoundary='band2018')
    p=f.scheme.dfsph.divergence
    p.relaxationFactor=.1;p.minIterations=p.maxIterations=8
    g=buildPressureBoundaryGeometry(f.state,f.config,f.scheme,f.adjacency)
    g.ratios[0]=.5
    velocity=torch.where(g.fluid[:,None],-f.state.positions,0)
    r=solvePressureBoundaryProjection(f.state,f.config,f.scheme,.01,f.adjacency,g,
        mode='divergence',velocity=velocity,pressureImpulse=torch.ones_like(f.state.masses))
    assert r.pressure[0]==0
    assert r.pressure.max()>0 and r.acceleration[0].norm()>0


def test_interpolation_ghosts_are_not_pressure_unknowns_or_neighbours():
    from warpSPH.modules.dfsph.pressureBoundaries import buildPressureBoundaryGeometry
    from warpSPHCore import buildVerletList
    f=_nativeIisphFixture('cpu',2,False)
    before=buildPressureBoundaryGeometry(f.state,f.config,f.scheme,f.adjacency)
    # Add an overlapping ghost with arbitrary mass and velocity. It must not
    # change any physical coefficient or introduce a wall-pressure source.
    for key,value in list(vars(f.state).items()):
        if torch.is_tensor(value):setattr(f.state,key,torch.cat((value,value[:1].clone())))
    f.state.kinds[-1]=2;f.state.masses[-1]*=100;f.state.velocities[-1]=100
    adj=buildVerletList(f.state,f.config.domain)
    after=buildPressureBoundaryGeometry(f.state,f.config,f.scheme,adj)
    assert not after.rows[-1]
    torch.testing.assert_close(after.diagonal[:-1],before.diagonal)
    torch.testing.assert_close(after.ratios[:-1],before.ratios)


def test_pressure_boundary_configuration_roundtrip_and_validation():
    from warpSPH.configurations import DFSPHConfig
    from warpSPH.configurations.moduleConfigurations.dfsph import dictToDFSPHConfig
    from dataclasses import asdict
    assert dictToDFSPHConfig(asdict(DFSPHConfig(pressureBoundary='band2018'))).pressureBoundary=='band2018'
    with pytest.raises(ValueError,match='pressureBoundary'):
        DFSPHConfig(pressureBoundary='typo').validate()


def test_internal_volume_pressure_forces_conserve_momentum_and_torque():
    from warpSPH.modules.dfsph.pressureBoundaries import buildPressureBoundaryGeometry
    f=_nativeIisphFixture('cpu',2,False)
    f.state.kinds.zero_()
    g=buildPressureBoundaryGeometry(f.state,f.config,f.scheme,f.adjacency)
    q=torch.linspace(-.4,.7,len(f.state.masses),dtype=f.state.masses.dtype)
    force=f.state.masses[:,None]*g.acceleration(f.config,f.adjacency,q)
    tolerance=20*torch.finfo(force.dtype).eps
    assert float(force.sum(0).norm()/force.abs().sum()) < tolerance
    x=f.state.positions
    torque=(x[:,0]*force[:,1]-x[:,1]*force[:,0]).sum()
    assert float(torque.abs()/(x.norm(dim=1)*force.norm(dim=1)).sum()) < tolerance
