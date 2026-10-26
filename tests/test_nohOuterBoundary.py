"""Open-edge Noh changes boundary topology, preserving the sampled inputs."""
import pytest
import torch

from warpSPH.cases.compressible.noh import nohCase
from warpSPH.runner import RunConfiguration


@pytest.mark.parametrize('scheme', ['Monaghan','CompSPH','CRKSPH','GSPH','InutsukaGSPH','PESPH'])
def test_open_edges_preserve_particle_initialization_and_clear_periodicity(scheme):
    states=[]
    for boundary in ('native','open'):
        ctx=RunConfiguration('noh',simulation=dict(device='cpu',nx=24),
            params=dict(outerBoundary=boundary,uAmbient=1e-8),solver=dict(scheme=scheme)).prepare(nohCase)
        system=nohCase.buildSystem(ctx)
        states.append(system.state)
        assert bool(ctx.config.domain.periodic.any()) == (boundary=='native')
        assert ctx.spec.periodic == (boundary=='native')
        if boundary=='open':
            assert system.domain is ctx.config.domain
            assert system.adjacency is None
    for field in ('positions','velocities','masses','densities','supports','internalEnergies','pressures'):
        torch.testing.assert_close(getattr(states[0],field),getattr(states[1],field),rtol=0,atol=0)


@pytest.mark.parametrize('scheme', ['MFM','MFV'])
def test_meshless_open_edges_keep_physical_particles_and_use_exterior_inflow(scheme):
    from warpSPH.runner import run
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    contexts, systems = [], []
    for boundary in ('native', 'open'):
        ctx = RunConfiguration('noh', simulation=dict(device=device, nx=32),
            params=dict(outerBoundary=boundary, uAmbient=1e-8), solver=dict(scheme=scheme)).prepare(nohCase)
        contexts.append(ctx)
        systems.append(nohCase.buildSystem(ctx))
    for field in ('positions', 'Q', 'masses', 'supports', 'UIDs'):
        torch.testing.assert_close(getattr(systems[0].state, field), getattr(systems[1].state, field), rtol=0, atol=0)
    opened = systems[1]
    assert opened.inflowReservoir is not None
    assert not contexts[1].spec.periodic and not bool(opened.domain.periodic.any())
    assert opened.state.Q.shape[0] == 32
    r = run(nohCase, contexts[1].spec, nSteps=20, quiet=True, progress=False)
    assert r.nSteps == 20 and not r.diverged
    assert r.state.state.Q.shape[0] == 32
    assert torch.isfinite(r.state.state.Q).all()
    assert nohCase.extraData(r.ctx, r.state)['boundaryTreatment'] == 'analytic Lagrangian inflow reservoir'


@pytest.mark.parametrize('scheme', ['MFM','MFV'])
def test_open_meshless_noh_rejects_projection_and_an_exhausted_inflow_domain(scheme):
    with pytest.raises(ValueError, match='closure=none'):
        RunConfiguration('noh', solver=dict(scheme=scheme, closure='project'),
            params=dict(outerBoundary='open')).prepare(nohCase)
    with pytest.raises(ValueError, match='material edge'):
        RunConfiguration('noh', simulation=dict(L=1.), solver=dict(scheme=scheme),
            params=dict(outerBoundary='open')).prepare(nohCase)
    ctx = RunConfiguration('noh', simulation=dict(periodic=False, device='cpu'),
        solver=dict(scheme=scheme), params=dict(outerBoundary='open')).prepare(nohCase)
    assert not ctx.spec.periodic
