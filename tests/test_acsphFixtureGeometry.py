"""Exercise the actual ghost consumer of the reference fixture's SDF contract."""
import torch
import pytest
from types import SimpleNamespace
from scripts.acsph_reference.fixture_geometry import (
    lobovsky_tank_sdf, audit_flat_wall_normals, volume_ghost_initializer)
from warpSPH.rigidBody.ghostParticles import _mergedSurface


def test_lobovsky_flat_walls_keep_distinct_geometry_normals():
    # Start with an outside corner: returning a bare vector makes its scalar
    # value broadcast to every subsequent wall in _mergedSurface.
    x = torch.tensor([[-.003, -.003], [.6, -.00075], [-.00075, .7],
        [1.61075, .7], [.6, 1.40075]], dtype=torch.float64)
    values, normals = lobovsky_tank_sdf(x)
    assert values.shape == (5,)
    assert normals.shape == (5, 2)
    _, merged = _mergedSurface(x, [lobovsky_tank_sdf], .00075)
    expected = x.new_tensor([[0., 1.], [1., 0.], [-1., 0.], [0., -1.]])
    torch.testing.assert_close(merged[1:], expected, atol=1e-12, rtol=0)
    torch.testing.assert_close(normals[1:], expected, atol=1e-12, rtol=0)


def test_lobovsky_sdf_sign_and_finite_surface_normals():
    x = torch.tensor([[.8, .7], [.6, 0.], [.6, -.001], [0., 0.]], dtype=torch.float64)
    value, normal = lobovsky_tank_sdf(x)
    torch.testing.assert_close(value, x.new_tensor([.7, 0., -.001, 0.]))
    assert torch.isfinite(normal).all()
    torch.testing.assert_close(normal.norm(dim=-1), torch.ones(4, dtype=x.dtype))


def test_reference_guard_rejects_diagonal_normals_on_flat_walls():
    positions = torch.tensor([[.6, -.001], [.6, 1.401], [-.001, .7], [1.611, .7]])
    correct = torch.tensor([[0., 1.], [0., -1.], [1., 0.], [-1., 0.]])
    state = SimpleNamespace(positions=torch.cat((positions, positions+.001*correct)),
        kinds=torch.tensor([1]*4+[2]*4), ghostIndices=torch.tensor([4, 5, 6, 7, 0, 1, 2, 3]),
        ghostOffsets=torch.cat((-.001*correct, .001*correct)))
    assert set(audit_flat_wall_normals(state)) == {'bottom', 'top', 'left', 'right'}
    state.ghostOffsets[4:] = .001
    with pytest.raises(ValueError, match='Invalid derived bottom wall normals'):
        audit_flat_wall_normals(state)


def _ghost_fixture(rho):
    from warpSPH.configurations.region import RegionType
    x=torch.tensor([[.8,.1],[1.58775,-.0015],[1.602,-.00075],
                    [.6,-.00075]],dtype=torch.float64)
    n=len(x)
    state=SimpleNamespace(positions=x,supports=x.new_full((n,),.003),
        masses=x.new_full((n,),rho*.00075**2),densities=x.new_full((n,),rho),
        velocities=torch.zeros_like(x),pressures=x.new_zeros(n),soundspeeds=None,
        kinds=torch.tensor([0,1,1,1],dtype=torch.int32),materials=torch.zeros(n,dtype=torch.int32),
        emitters=None,bodyIDs=torch.tensor([-1,0,0,0]),UIDs=torch.arange(n),UIDcounter=n)
    region=SimpleNamespace(type=RegionType.Boundary,sdf=lobovsky_tank_sdf)
    return [region],state


@pytest.mark.parametrize('mode', ['gridsnap','hybrid'])
def test_volume_ghost_geometry_is_invariant_under_density_unit_scaling(monkeypatch,mode):
    monkeypatch.setenv('WARPSPH_GHOST_PLACEMENT',mode)
    import warpSPH.initializers.weaklyCompressible as initializer
    import warpSPH.rigidBody.ghostParticles as geometry
    original_add=initializer.addBoundaryGhostParticles
    original_place=geometry._gridSnapGhostOffsets
    original_hybrid=geometry._hybridGhostOffsets
    results=[]
    for rho in (1.,1000.):
        regions,state=_ghost_fixture(rho);audit={}
        with volume_ghost_initializer(audit):
            results.append(initializer.addBoundaryGhostParticles(regions,state))
        assert audit['spacing']==pytest.approx(.00075)
        assert initializer.addBoundaryGhostParticles is original_add
        assert geometry._gridSnapGhostOffsets is original_place
        assert geometry._hybridGhostOffsets is original_hybrid
    torch.testing.assert_close(results[0].positions,results[1].positions,rtol=0,atol=1e-15)
    torch.testing.assert_close(results[0].ghostOffsets,results[1].ghostOffsets,rtol=0,atol=1e-15)
    unit=results[1].ghostOffsets[4:]/results[1].ghostOffsets[4:].norm(dim=-1,keepdim=True)
    torch.testing.assert_close(unit,unit.new_tensor([[0,1]]).expand_as(unit),rtol=0,atol=1e-12)
    # The first-layer node sits 0.5dx into the fluid, not centimetres away.
    assert results[1].positions[-1,1]==pytest.approx(.5*.00075)
    regions,state=_ghost_fixture(1000.)
    legacy=original_add(regions,state)
    legacy_unit=legacy.ghostOffsets[5]/legacy.ghostOffsets[5].norm()
    assert abs(float(legacy_unit[0]))>.1


def test_volume_ghost_initializer_restores_hooks_after_failure():
    import warpSPH.initializers.weaklyCompressible as initializer
    import warpSPH.rigidBody.ghostParticles as geometry
    original_add=initializer.addBoundaryGhostParticles
    original_place=geometry._gridSnapGhostOffsets
    regions,state=_ghost_fixture(1000.);state.densities[0]=0
    with pytest.raises(ValueError,match='finite positive particle volumes'):
        with volume_ghost_initializer({}):
            initializer.addBoundaryGhostParticles(regions,state)
    assert initializer.addBoundaryGhostParticles is original_add
    assert geometry._gridSnapGhostOffsets is original_place
