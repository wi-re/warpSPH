"""Physical invariants of the opt-in Lobovsky audit operators."""
import torch

from scripts.acsph_reference.audit_local import fluid_side_wall_diffusion
from warpSPH.configurations import ArtificialCompressibleSPHConfig
from warpSPH.configurations.simulationConfig import SimulationConfig
from warpSPH.modules.artificialCompressible.pressureSmoothing import _biLaplacian
from warpSPH.systems.artificialCompressible import ArtificialCompressibleState
from warpSPH.utils.domain import buildDomainDescription
from warpSPHCore import get_torch_precision


def patch():
    dtype = get_torch_precision()
    axis = (torch.arange(16, dtype=dtype) + .5) * .1
    x = torch.stack(torch.meshgrid(axis, axis, indexing='ij'), -1).reshape(-1, 2)
    n = len(x)
    state = ArtificialCompressibleState(
        positions=x, velocities=torch.zeros_like(x),
        pressures=20000-9810*x[:, 1], supports=torch.full((n,), .4, dtype=dtype),
        masses=torch.full((n,), 10., dtype=dtype), densities=torch.full((n,), 1000., dtype=dtype),
        kinds=(x[:, 0] < .8).to(torch.int32), materials=torch.zeros(n, dtype=torch.int32),
        UIDs=torch.arange(n, dtype=torch.int32), UIDcounter=n)
    config = SimulationConfig(device=torch.device('cpu'), dtype=dtype, dim=2, dx=.1,
        domain=buildDomainDescription(l=4., dim=2, periodic=False, device='cpu', dtype=dtype))
    return state, config, ArtificialCompressibleSPHConfig()


def test_wall_flux_preserves_linear_fluid_pressure_with_unloaded_outer_wall(runtime):
    state, config, scheme = patch()
    # As in Adami's closure: outer solid rows without fluid support have p=0.
    # Fluid rows see only correctly extrapolated, linear wall pressures, but
    # wall-gradient estimates reach those unloaded outer rows.
    state.pressures[state.positions[:, 0] < .4] = 0.
    fluid = state.kinds == 0
    old = _biLaplacian(state, config, scheme, None, None, state.pressures)[fluid]
    new = fluid_side_wall_diffusion(state, config, scheme, None, None, state.pressures)[fluid]
    assert old.abs().max() > 1e3
    assert new.abs().max() < old.abs().max()*1e-4


def test_wall_flux_keeps_the_existing_fluid_only_operator(runtime):
    state, config, scheme = patch()
    state.kinds.zero_()
    state.pressures = torch.sin(7*state.positions[:, 0])*torch.cos(3*state.positions[:, 1])
    old = _biLaplacian(state, config, scheme, None, None, state.pressures)
    new = fluid_side_wall_diffusion(state, config, scheme, None, None, state.pressures)
    torch.testing.assert_close(new, old, rtol=0, atol=0)


def test_consistent_wall_closure_reaches_entry_stages_and_shift(runtime, monkeypatch):
    import importlib
    from scripts.acsph_reference.audit_local import (
        install_consistent_wall_velocity, reference_wall_velocities)
    from warpSPH.modules.artificialCompressible import dualTime
    from warpSPH.systems import artificialCompressible as system_module
    step_module = importlib.import_module('warpSPH.schemes.incompressible.artificialCompressible')
    state, config, scheme = patch()
    fluid = state.kinds == 0
    state.velocities[fluid] = state.velocities.new_tensor([1., 2.])
    state.velocities[~fluid] = 999.
    expected = reference_wall_velocities(state, config, scheme, None)
    # A flat vertical wall reflects normal speed and preserves tangential speed.
    flat = (~fluid)&(state.positions[:,0]>.6)&(state.positions[:,1]>.5)&(state.positions[:,1]<1.1)
    torch.testing.assert_close(expected[flat], expected.new_tensor([-1.,2.]).expand(int(flat.sum()),2), atol=2e-5, rtol=0)
    captured = []
    def observe_shift(**kwargs):
        captured.append(kwargs['systemState'].velocities.clone())
        return torch.zeros_like(state.positions)
    # Register restoration of every process-local hook with pytest.
    monkeypatch.setattr(step_module, 'computeBoundaryVelocities', step_module.computeBoundaryVelocities)
    monkeypatch.setattr(dualTime, 'computeBoundaryVelocities', dualTime.computeBoundaryVelocities)
    monkeypatch.setattr(system_module, 'solveShifting', observe_shift)
    install_consistent_wall_velocity()
    for closure in (step_module.computeBoundaryVelocities, dualTime.computeBoundaryVelocities):
        torch.testing.assert_close(closure(state,config,scheme,None), expected, rtol=0, atol=0)
    # Change the accepted fluid field and poison the stale wall values: shifting
    # must close from the new field, rather than carry the step-entry sample.
    state.velocities[fluid] *= 2.
    state.velocities[~fluid] = -999.
    expected = reference_wall_velocities(state,config,scheme,None)
    before_fluid = state.velocities[fluid].clone()
    system_module.solveShifting(systemState=state,config=config,schemeConfig=scheme,adjacency=None,dt=.001)
    assert len(captured) == 1
    torch.testing.assert_close(captured[0], expected, atol=2e-5, rtol=1e-6)
    torch.testing.assert_close(state.velocities[fluid], before_fluid, rtol=0, atol=0)
