"""A search optimization must retain every current pair, including after mutation."""
from types import SimpleNamespace
import pytest
import torch
from warpSPHCore import DomainDescription, get_torch_precision
from warpSPHCore.dataTypes import ParticleState
from warpSPHCore.enumTypes import SupportScheme
from warpSPHCore.radiusSearch import radiusSearchCompactHashMap
from warpSPH.configurations.meshless import MFMConfig, meshlessConfigToDict, dictToMeshlessConfig
from warpSPH.modules.meshless.geometry import cachedAdjacency, inheritGeometry


@pytest.mark.parametrize('dim', [1, 2, 3])
@pytest.mark.parametrize('device', ['cpu'] + (['cuda'] if torch.cuda.is_available() else []))
@pytest.mark.parametrize('change', ['reuse', 'inplace', 'grow', 'growOne', 'shrinkOne', 'jump', 'periodic', 'domain', 'flags', 'clone', 'count'])
def test_skin_has_exact_current_pairs(dim, change, device):
    dtype = get_torch_precision()
    gen = torch.Generator().manual_seed(183)
    pos = torch.rand((48, dim), generator=gen, dtype=dtype).to(device)
    h = (.23+.08*torch.rand((len(pos),), generator=gen, dtype=dtype)).to(device)
    domain = DomainDescription(torch.zeros(dim, dtype=dtype, device=device), torch.ones(dim, dtype=dtype, device=device),
        torch.ones(dim, dtype=torch.bool, device=device), dim)
    owner = SimpleNamespace(positions=pos, supports=h)
    cfg = MFMConfig(neighborSkin=1.25)
    cachedAdjacency(pos, h, domain, cfg, owner=owner)
    previous = owner._meshlessNeighbors.adjacency
    if change == 'reuse':
        pos = pos+.001
    elif change == 'inplace':
        pos.add_(.001)
    elif change == 'grow':
        h.mul_(1.01)
    elif change == 'growOne':
        h[0] *= 2
    elif change == 'shrinkOne':
        h[0] *= .2
    elif change == 'flags':
        domain.periodic[0] = False
    elif change == 'jump':
        pos.add_(.3)
    elif change == 'periodic':
        pos.add_(1.)
    elif change == 'domain':
        domain.max.add_(.2)
    elif change == 'clone':
        target = SimpleNamespace(positions=pos.clone(), supports=h.clone())
        inheritGeometry(owner, target)
        owner, pos, h = target, target.positions, target.supports
        pos.add_(.001)
    elif change == 'count':
        pos, h = pos[:-3].clone(), h[:-3].clone()
    actual = cachedAdjacency(pos, h, domain, cfg, owner=owner)
    particles = ParticleState(pos, h, torch.ones_like(h), torch.zeros_like(h, dtype=torch.int32))
    expected = radiusSearchCompactHashMap(particles, domain, mode=SupportScheme.SuperSymmetric)
    def pairs(adj):
        return torch.sort(adj.paddedI[:adj.numEdges].long()*len(pos)+adj.paddedJ[:adj.numEdges].long()).values
    torch.testing.assert_close(pairs(actual), pairs(expected), rtol=0, atol=0)
    if change in ('reuse', 'inplace', 'grow', 'periodic', 'clone'):
        assert owner._meshlessNeighbors.adjacency is previous
    else:
        assert owner._meshlessNeighbors.adjacency is not previous
    # No other system may accidentally acquire this cache.
    other = SimpleNamespace()
    cachedAdjacency(pos, h, domain, cfg, owner=other)
    assert other._meshlessNeighbors is not owner._meshlessNeighbors


def test_skin_config_roundtrip_and_baseline():
    cfg = dictToMeshlessConfig(meshlessConfigToDict(MFMConfig(neighborSkin=1.3)))
    assert cfg.neighborSkin == 1.3
    for value in [.99, float('nan'), float('inf')]:
        with pytest.raises(ValueError, match='neighborSkin'):
            MFMConfig(neighborSkin=value).validate()
    assert cachedAdjacency(None, None, None, MFMConfig(), owner=SimpleNamespace()) is None


@pytest.mark.parametrize('scheme', ['MFM', 'MFV'])
@pytest.mark.parametrize('thermal', ['relaxation', 'adiabatic'])
def test_skin_preserves_short_driven_fields_and_budgets(scheme, thermal):
    from warpSPH.runner import RunConfiguration
    def run(skin):
        return RunConfiguration('drivenTurbulence', simulation=dict(device='cpu', nx=16,
            nSteps=40, plot=False, video=False, store=False, show=False, quiet=True, progress=False),
            params=dict(accelerationRMS=6., thermalModel=thermal),
            solver=dict(scheme=scheme, temporalScheme='ssprk', neighborSkin=skin)).run()
    reference, cached = run(1.), run(1.25)
    assert not reference.diverged and not cached.diverged
    assert abs(float(reference.state.t)-float(cached.state.t)) < 1e-6
    for name in ['Q', 'supports', 'internalEnergies', 'pressures']:
        a, b = [getattr(r.state.state, name).double() for r in [reference, cached]]
        assert float((a-b).norm()/a.norm()) < (3e-5 if get_torch_precision() == torch.float32 else 1e-11)
    for result in [reference, cached]:
        for row in result.trajectory:
            assert min(row[k] for k in ['minDensity', 'minEnergy', 'minPressure']) > 0
            assert abs(row['relativeEnergyBudgetResidual']) < 3e-6
            assert abs(row['relativeMassChange']) < 3e-7


def test_default_checkpoint_signature_remains_compatible(tmp_path):
    import h5py
    import json
    from pathlib import Path
    from warpSPH.runner import RunConfiguration
    first = RunConfiguration('drivenTurbulence', simulation=dict(device='cpu', nx=12,
        nSteps=2, plot=False, video=False, store=True, storeMode='states', show=False,
        progress=False, quiet=True, exportRoot=str(tmp_path/'first'))).run()
    path = Path(first.exportPath)/'trajectory/finalState.h5'
    with h5py.File(path) as f:
        assert 'neighborSkin' not in json.loads(f.attrs['checkpointSignature'])['solver']
    resumed = RunConfiguration('drivenTurbulence', simulation=dict(device='cpu', nx=12,
        nSteps=2, plot=False, video=False, store=False, show=False, progress=False, quiet=True,
        resumeFrom=str(path))).run()
    assert float(resumed.state.t) > float(first.state.t)


@pytest.mark.parametrize('scheme', ['MFM', 'MFV'])
def test_cached_restart_reconstructs_search_without_changing_the_driving(tmp_path, scheme):
    from pathlib import Path
    from warpSPH.runner import RunConfiguration
    def config(steps):
        return RunConfiguration('drivenTurbulence', simulation=dict(device='cpu', nx=12,
            nSteps=steps, plot=False, video=False, store=False, show=False, quiet=True, progress=False),
            params=dict(accelerationRMS=6.),
            solver=dict(scheme=scheme, temporalScheme='ssprk', neighborSkin=1.25))
    full = config(80).run()
    cfg = config(40)
    cfg.simulation.update(store=True, storeMode='states', exportRoot=str(tmp_path/'first'))
    first = cfg.run()
    cfg = config(40)
    cfg.simulation['resumeFrom'] = str(Path(first.exportPath)/'trajectory/finalState.h5')
    resumed = cfg.run()
    tolerance = 3e-5 if get_torch_precision() == torch.float32 else 1e-11
    a,b = full.state.state.Q, resumed.state.state.Q
    assert float((a-b).norm()/a.norm()) < tolerance
    assert not resumed.diverged
    # Restarts rebuild list ordering; they preserve the exact driving realization.
    for time in [.01, float(resumed.state.t), 1.7]:
        torch.testing.assert_close(full.state.sourceTerms.acceleration.coefficients(time),
                                   resumed.state.sourceTerms.acceleration.coefficients(time), rtol=0, atol=0)
