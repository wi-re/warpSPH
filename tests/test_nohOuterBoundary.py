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
def test_meshless_open_edges_are_rejected_without_bypassing_periodic_contract(scheme):
    RunConfiguration('noh',solver=dict(scheme=scheme),params=dict(outerBoundary='native')).prepare(nohCase)
    with pytest.raises(ValueError,match='open edges are not supported by MFM/MFV'):
        RunConfiguration('noh',solver=dict(scheme=scheme),params=dict(outerBoundary='open')).prepare(nohCase)
