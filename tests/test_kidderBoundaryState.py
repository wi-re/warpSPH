"""The accepted Kidder boundary state must survive compatible finalization."""
from types import SimpleNamespace

import pytest
import torch

from warpSPH.cases.compressible.kidder import postStep
from warpSPH.caseUtils.compressible.kidder.bc import buildKidderBCs
from warpSPH.configurations import CRKSPHConfig, CompSPHConfig
from warpSPHCore import DomainDescription


class AnalyticShell:
    """Independent positive, time-varying prescriptions on the shell band."""
    def rho(self, t, r):
        return 2. + t + r

    def P(self, t, r):
        return 5. + 2*t + r

    def vr(self, t, r):
        return -t*r


@pytest.mark.parametrize('configuration', [CRKSPHConfig, CompSPHConfig])
def test_accepted_boundary_restores_density_energy_velocity_and_eos(configuration):
    dtype=torch.float64
    x=torch.linspace(.9, 1., 12, dtype=dtype)[:,None]
    state=SimpleNamespace(positions=x, velocities=torch.zeros_like(x),
        densities=torch.full((12,), 3., dtype=dtype),
        internalEnergies=torch.full((12,), 7., dtype=dtype),
        pressures=torch.zeros(12,dtype=dtype), entropies=torch.zeros(12,dtype=dtype),
        soundspeeds=torch.zeros(12,dtype=dtype), materials=torch.zeros(12,dtype=torch.int32))
    # Simulate compatible work overwriting the externally prescribed bands.
    state.internalEnergies[:2]=-100.;state.internalEnergies[-2:]=-200.
    scheme=configuration(gamma=3.)
    scheme.boundaryConditions=[buildKidderBCs(scheme,AnalyticShell(),2)]
    config=SimpleNamespace(domain=DomainDescription(torch.tensor([-1.],dtype=dtype),
        torch.tensor([1.],dtype=dtype),torch.tensor([False]),1))
    ctx=SimpleNamespace(config=config,schemeConfig=scheme)
    system=SimpleNamespace(state=state,t=.2)
    before={key:getattr(state,key)[2:-2].clone() for key in ('positions','velocities','densities','internalEnergies')}
    postStep(ctx,system,0)
    mask=torch.tensor([True,True]+[False]*8+[True,True])
    radius=x[:,0]
    rho=2.+system.t+radius;pressure=5.+2*system.t+radius
    torch.testing.assert_close(state.densities[mask],rho[mask])
    torch.testing.assert_close(state.internalEnergies[mask],pressure[mask]/(2*rho[mask]))
    torch.testing.assert_close(state.velocities[mask,0],-system.t*radius[mask])
    torch.testing.assert_close(state.pressures[mask],pressure[mask])
    torch.testing.assert_close(state.soundspeeds[mask],torch.sqrt(3*pressure[mask]/rho[mask]))
    torch.testing.assert_close(state.entropies[mask],pressure[mask]/rho[mask]**3)
    for key,value in before.items():
        torch.testing.assert_close(getattr(state,key)[2:-2],value,rtol=0,atol=0)


@pytest.mark.parametrize('scheme', ['MFM', 'MFV'])
def test_meshless_driven_bands_survive_every_temporal_driver(scheme):
    from warpSPH.cases.compressible.kidder import kidderCase
    from warpSPH.runner import RunConfiguration, run
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for temporal in ('ssprk', 'hancock', 'kdk'):
        ctx = RunConfiguration('kidder', simulation=dict(device=device, nx=48,
            nSteps=12, quiet=True, progress=False, plot=False, video=False),
            solver=dict(scheme=scheme, temporalScheme=temporal)).prepare(kidderCase)
        r = run(kidderCase, ctx.spec)
        assert r.nSteps == 12 and not r.diverged
        system = r.state
        state, driven = system.state, system.drivenParticles
        sol, t = r.ctx.scratch['solution'], float(system.t)
        mask = driven.mask
        expectedPosition = driven.referencePositions * sol.hfrac(t)
        expectedVelocity = -expectedPosition * t / (sol.tau * sol.hfrac(t))**2
        torch.testing.assert_close(state.positions[mask], expectedPosition[mask])
        torch.testing.assert_close(state.velocities[mask], expectedVelocity[mask])
        expectedPressure = torch.as_tensor(sol.P(t, expectedPosition[:, 0].cpu().numpy()),
            dtype=state.Q.dtype, device=device)
        torch.testing.assert_close(state.pressures[mask], expectedPressure[mask], rtol=3e-5, atol=3e-5)
        torch.testing.assert_close(state.masses[mask], driven.masses[mask], rtol=0, atol=0)
        # The interior is evolved by fluxes, while the prescribed band supplies work.
        assert state.velocities[~mask].abs().max() > 0
        assert torch.isfinite(state.Q).all()


@pytest.mark.parametrize('scheme', ['MFM', 'MFV'])
def test_meshless_kidder_rejects_nonplanar_and_projected_closure(scheme):
    from warpSPH.cases.compressible.kidder import kidderCase
    from warpSPH.runner import RunConfiguration
    with pytest.raises(ValueError, match='planar shell'):
        RunConfiguration('kidder', simulation={'dim': 2}, solver={'scheme': scheme}).prepare(kidderCase)
    with pytest.raises(ValueError, match='closure=none'):
        RunConfiguration('kidder', solver={'scheme': scheme, 'closure': 'project'}).prepare(kidderCase)


def test_meshless_reservoir_changes_only_its_prescribed_rows():
    from warpSPH.modules.meshless.boundary import DrivenParticles
    x = torch.linspace(.9, 1., 6, dtype=torch.float64)[:, None]
    mass = torch.arange(1., 7., dtype=torch.float64)
    mask = torch.tensor([True, False, False, False, False, True])
    driven = DrivenParticles(mask, x.clone(), mass,
        lambda t, reference: reference * (1 - t),
        lambda t, positions: (-positions * t, torch.full_like(mass, 3.)))
    state = SimpleNamespace(positions=x + .02, Q=torch.arange(18., dtype=x.dtype).reshape(6, 3))
    volume = torch.linspace(.01, .02, 6, dtype=x.dtype)
    beforePosition, beforeQ = state.positions.clone(), state.Q.clone()
    driven.position(state, .2)
    driven.conserved(state, SimpleNamespace(volume=volume), .2, 1.5)
    torch.testing.assert_close(state.positions[~mask], beforePosition[~mask], rtol=0, atol=0)
    torch.testing.assert_close(state.Q[~mask], beforeQ[~mask], rtol=0, atol=0)
    torch.testing.assert_close(state.Q[mask, 0], mass[mask], rtol=0, atol=0)
    velocity = -.2 * state.positions[mask, 0]
    torch.testing.assert_close(state.Q[mask, 1], mass[mask] * velocity)
    torch.testing.assert_close(state.Q[mask, 2], 6 * volume[mask] + .5 * mass[mask] * velocity.square())
