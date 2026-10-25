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
