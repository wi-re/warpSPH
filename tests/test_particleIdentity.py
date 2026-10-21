"""Source provenance must not select physical material or boundary motion."""
from types import SimpleNamespace

import pytest
import torch

from warpSPH.configurations import WeaklyCompressibleSPHConfig
from warpSPH.configurations.region import ParticleRegion, RegionType, BCType
from warpSPH.geometry import ParticleSet
from warpSPH.initializers.weaklyCompressible import initializeState
from warpSPH.systems import WeaklyCompressibleState, ArtificialCompressibleState
from warpSPH.systems.incompressible import IncompressibleState
from warpSPH.systems.identity import initialMaterialIDs, particleBodyIDs


def region(kind=RegionType.Fluid, offset=0., materials=None, bc=BCType.constant):
    pos = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [1., 1.]]) + offset
    particles = ParticleSet(pos, torch.ones(4), torch.ones(4), torch.ones(4))
    sdf = lambda x: (x[:, 0] - offset, torch.ones_like(x))
    return ParticleRegion(sdf, kind, particles, initialConditions={} if materials is None else {'materials': materials}, kind=bc)


@pytest.mark.parametrize('cls', [WeaklyCompressibleState, IncompressibleState, ArtificialCompressibleState])
def test_one_source_can_contain_two_materials_and_two_sources_share_a_material(cls):
    regions = [region(materials=lambda p: (p[:, 0] > .5).to(torch.int32)), region(offset=2.)]
    s = initializeState(regions, SimpleNamespace(), WeaklyCompressibleSPHConfig(), cls)
    assert s.materials.tolist() == [0, 1, 0, 1, 0, 0, 0, 0]
    assert s.emitters.tolist() == [0]*4 + [1]*4
    assert s.bodyIDs.tolist() == [-1]*8
    for copy in (s.clone(), s.initializeNewState(), s.to('cpu')):
        for name in ('materials', 'emitters', 'bodyIDs'):
            torch.testing.assert_close(getattr(copy, name), getattr(s, name))


@pytest.mark.parametrize('value', [1.5, -1, [0, 1], [0, 1, 0, 2**32]])
def test_invalid_material_maps_fail_at_initialization(value):
    r = region(materials=value)
    with pytest.raises(ValueError, match='initialConditions.materials'):
        initialMaterialIDs(r, r.particles.positions)


def test_callable_and_default_region_material_maps_roundtrip():
    for r in (region(), region(materials=lambda p: (p[:, 0] > .5).to(torch.int32))):
        restored = ParticleRegion.fromDict(r.toDict())
        torch.testing.assert_close(initialMaterialIDs(restored, r.particles.positions), initialMaterialIDs(r, r.particles.positions))
    # Previously serialized configs may contain a null initialConditions.
    doc = region().toDict()
    doc['initialConditions'] = None
    assert ParticleRegion.fromDict(doc).initialConditions == {}


def test_body_membership_and_ghosts_are_independent_of_material_and_emitter(monkeypatch):
    import warpSPH.rigidBody.ghostParticles as ghosts
    from warpSPH.rigidBody.build import buildRigidBody
    from warpSPH.rigidBody.update import updateBodyParticlesWCSPH
    from warpSPH.modules.mdbc.velocity import computeBoundaryVelocities
    regions = [region(), region(RegionType.Boundary, 2., 7, BCType.zeros),
               region(RegionType.Boundary, 4., 7, BCType.constant)]
    cfg = SimpleNamespace(regions=regions)
    scfg = WeaklyCompressibleSPHConfig()
    s = initializeState(regions, cfg, scfg, WeaklyCompressibleState)
    s.emitters.fill_(99)  # Body membership must not depend on this label.
    monkeypatch.setattr(ghosts, '_gridSnapGhostOffsets', lambda p, *args: torch.full_like(p, .1))
    s = ghosts.addBoundaryGhostParticles(regions, s)
    assert int((s.kinds == 2).sum()) == 8
    for body in (0, 1):
        mask = (s.kinds == 2) & (s.bodyIDs == body)
        assert int(mask.sum()) == 4 and (s.materials[mask] == 7).all()
        rb = buildRigidBody(s, regions, body)
        assert int(rb.particleIndices.sum()) == 4
        rb.linearVelocity.fill_(body + 1.)
        s = updateBodyParticlesWCSPH(s, rb)
    assert (s.emitters == 99).all()
    before = s.velocities.clone()
    velocity = computeBoundaryVelocities(s, cfg, scfg, None)
    assert (velocity[(s.kinds == 1) & (s.bodyIDs == 0)] == 0).all()
    torch.testing.assert_close(velocity[(s.kinds == 1) & (s.bodyIDs == 1)],
                               before[(s.kinds == 1) & (s.bodyIDs == 1)])
    # Old manually built states and trajectories retain boundary membership.
    old = s.clone()
    old.materials = s.bodyIDs.clone()
    old.bodyIDs = None
    torch.testing.assert_close(particleBodyIDs(old), s.bodyIDs)


@pytest.mark.parametrize('legacy', [False, True])
def test_identity_export_and_dataset_masking(tmp_path, legacy):
    import h5py
    from warpSPH.io.export import writeInitialData
    from warpSPH.io.importIO import loadTrajectory
    from warpSPH.io.dataset import SPHDataset, sample_to_state
    s = initializeState([region(), region(RegionType.Boundary, 2., 4)],
                        SimpleNamespace(), WeaklyCompressibleSPHConfig(), WeaklyCompressibleState)
    s.kinds[-1] = 2  # Exercise source/body masking when ghosts are removed.
    if legacy:
        s.emitters = s.bodyIDs = None
    (tmp_path/'config.json').write_text('{}')
    with h5py.File(tmp_path/'trajectory.h5', 'w') as f:
        writeInitialData(str(tmp_path), f, 'CompSPH', SimpleNamespace(dt=.01), SimpleNamespace(),
                         SimpleNamespace(exportInterval=.1), SimpleNamespace(state=s, t=0.))
        static = SPHDataset._read_static(None, f)
    sample = {'static': static, 'input_current': {'positions': s.positions, 'velocities': s.velocities, 'densities': s.densities}}
    restored = sample_to_state(sample)
    mask = s.kinds != 2
    torch.testing.assert_close(restored.materials, s.materials[mask])
    for name in ('emitters', 'bodyIDs'):
        if legacy:
            assert getattr(restored, name) is None
        else:
            torch.testing.assert_close(getattr(restored, name), getattr(s, name)[mask])
    f, meta = loadTrajectory(str(tmp_path), 'cpu')
    try:
        for name in ('emitters', 'bodyIDs'):
            if legacy:
                assert meta['static'][name] is None
            else:
                torch.testing.assert_close(meta['static'][name], getattr(s, name))
    finally:
        f.close()
