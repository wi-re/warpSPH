"""Physical-clock forcing and conservative source integration controls."""
from dataclasses import replace
import math
from types import SimpleNamespace

import pytest
import torch

from warpSPH.modules.forcing import FourierOU
from warpSPH.modules.meshless.sources import ConservedSources
from warpSPH.runner import RunConfiguration, configurationSchema
from warpSPH.cases.compressible.drivenTurbulence import drivenTurbulenceCase


def test_forcing_queries_and_checkpoints_do_not_depend_on_rhs_order():
    a, b = FourierOU(seed=71), FourierOU(seed=71)
    positions = torch.rand(17, 2, dtype=torch.float64)
    expected = {t: a(positions, t) for t in [0., .07, .3, .91]}
    for t in [.91, .07, 0., .3, .91]:
        torch.testing.assert_close(b(positions, t), expected[t], rtol=0, atol=0)
    restored = FourierOU.from_state_dict(a.state_dict())
    for t in [.01, .91, 1.37, 2.]:
        torch.testing.assert_close(restored(positions, t), a(positions, t), rtol=0, atol=0)
    torch.manual_seed(900)
    torch.randn(55)
    torch.testing.assert_close(FourierOU(seed=71)(positions, .3), expected[.3], rtol=0, atol=0)


@pytest.mark.parametrize('weight', [0., .4, 1.])
def test_fourier_periodicity_projection_and_normalization(weight):
    field = FourierOU(length=2., solenoidalWeight=weight, seed=7)
    axes = torch.arange(32, dtype=torch.float64)/16
    positions = torch.stack(torch.meshgrid(axes, axes, indexing='xy'), -1).reshape(-1, 2)
    a = field(positions, .125)
    torch.testing.assert_close(field(positions+torch.tensor([6., -4.]), .125), a, rtol=1e-12, atol=1e-13)
    coefficients = field.coefficients(.125)
    # Parseval on this resolved lattice: real sine/cosine coefficients carry half their square.
    torch.testing.assert_close(a.square().sum(-1).mean(), .5*coefficients.square().sum(), rtol=1e-12, atol=1e-14)
    if weight == 1:
        torch.testing.assert_close((coefficients*field.modes).sum(-1), torch.zeros_like(coefficients[:, :, 0]), rtol=0, atol=1e-15)
    if weight == 0:
        cross = coefficients[:, :, 0]*field.modes[:, 1]-coefficients[:, :, 1]*field.modes[:, 0]
        torch.testing.assert_close(cross, torch.zeros_like(cross), rtol=0, atol=1e-15)


def test_ou_knots_have_stationary_variance_and_prescribed_correlation():
    field = FourierOU(seed=483, correlationTime=.5, updateInterval=.05)
    field.coefficients(250.)
    samples = torch.stack(field.knots)[100:]
    variance = samples.square().mean()
    covariance = (samples[10:]*samples[:-10]).mean()
    assert abs(float(variance)-1) < .06
    assert abs(float(covariance/variance)-math.exp(-1)) < .06


def test_zero_forcing_and_invalid_inputs():
    assert torch.count_nonzero(FourierOU(accelerationRMS=0)(torch.randn(9, 2), 1.)).item() == 0
    for kwargs in [dict(correlationTime=0), dict(kMin=0), dict(kMax=17), dict(solenoidalWeight=2), dict(seed=-1)]:
        with pytest.raises(ValueError):
            FourierOU(**kwargs)
    with pytest.raises(ValueError, match='time'):
        FourierOU().coefficients(-1)


def test_sources_apply_actual_mass_and_compatible_work():
    mass = torch.tensor([1., 3.], dtype=torch.float64)
    velocity = torch.tensor([[2., -1.], [-.5, 1.]], dtype=torch.float64)
    acceleration = torch.tensor([[.2, .4], [-.1, .3]], dtype=torch.float64)
    ctx = SimpleNamespace(Q=torch.cat((mass[:, None], mass[:, None]*velocity, torch.ones(2, 1)), 1),
                          vel=velocity, pos=torch.zeros(2, 2), u=torch.tensor([1., 2.], dtype=torch.float64))
    source = ConservedSources(lambda positions, time: acceleration, False, 1.5, .25)
    rate, budget = source.rates(ctx, .1)
    dp = mass[:, None]*acceleration
    torch.testing.assert_close(rate[:, 1:-1], dp)
    torch.testing.assert_close(rate[:, -1], (dp*velocity).sum(1)+mass*(1.5-ctx.u)/.25)
    torch.testing.assert_close(rate[:, 0], torch.zeros_like(mass))
    torch.testing.assert_close(budget.sum(), rate[:, -1].sum())
    source.removeMeanAcceleration = True
    rate, budget = source.rates(ctx, .1)
    torch.testing.assert_close(rate[:, 1:-1].sum(0), torch.zeros(2, dtype=mass.dtype), rtol=0, atol=1e-15)


@pytest.mark.parametrize('simulation,params,solver,message', [
    ({'dim': 3}, {}, {}, 'dim'),
    ({'adaptiveDt': False}, {}, {}, 'adaptiveDt'),
    ({}, {'kMin': 3, 'kMax': 2}, {}, 'kMin'),
    ({}, {'forcingInterval': 2.}, {}, 'forcingInterval'),
    ({}, {}, {'scheme': 'MFV'}, 'ssprk'),
    ({}, {}, {'temporalScheme': 'hancock'}, 'ssprk'),
    ({'execution': 'graph'}, {}, {}, 'eager'),
    ({'resumeFrom': 'not-read.h5'}, {}, {}, 'resumeFrom'),
])
def test_case_contract_rejects_unsupported_physics(simulation, params, solver, message):
    with pytest.raises(ValueError, match=message):
        RunConfiguration('drivenTurbulence', simulation=simulation, params=params, solver=solver).resolve()


def configuration(**params):
    return RunConfiguration('drivenTurbulence', simulation=dict(device='cpu', nx=12, nSteps=None,
        tLimit=.04, maxDt=.002, plot=False, video=False, store=False, show=False, progress=False, quiet=True),
        params=params)


def test_driven_configuration_and_named_thermal_modes_round_trip():
    c = configuration(thermalModel='adiabatic')
    assert c.resolve().params['gamma'] > 1
    assert RunConfiguration.fromDict(c.toDict()).toDict() == c.toDict()
    schema = configurationSchema(drivenTurbulenceCase)
    assert schema['properties']['params']['properties']['thermalModel']['enum'] == ['relaxation', 'adiabatic']


def test_uniform_acceleration_preserves_internal_energy_and_work_budget():
    acceleration = torch.tensor([.25, -.1])
    def build(ctx):
        system = drivenTurbulenceCase.buildSystem(ctx)
        system.sourceTerms = ConservedSources(lambda x, t: acceleration.to(x).expand_as(x), False)
        return system
    case = replace(drivenTurbulenceCase, buildSystem=build)
    r = configuration(accelerationRMS=0., thermalModel='adiabatic').run(case)
    assert not r.diverged
    t = float(r.state.t)
    expected = (t*acceleration.to(r.state.state.velocities)).expand_as(r.state.state.velocities)
    torch.testing.assert_close(r.state.state.velocities, expected, rtol=3e-4, atol=5e-7)
    torch.testing.assert_close(r.state.state.internalEnergies, torch.full_like(r.state.state.internalEnergies, .9), rtol=2e-5, atol=2e-6)
    work = r.state.state.sourceEnergyBudget[0]
    expected_work = .5*r.state.state.Q[:, 0].sum()*(t*acceleration).square().sum()
    torch.testing.assert_close(work, expected_work.to(work), rtol=2e-4, atol=1e-9)
    assert abs(r.trajectory[-1]['relativeEnergyBudgetResidual']) < 2e-6


def test_uniform_cooling_matches_exponential_and_signed_energy_budget():
    def build(ctx):
        system = drivenTurbulenceCase.buildSystem(ctx)
        system.state.Q[:, -1] *= 2
        system.state.internalEnergies *= 2
        system.state.pressures *= 2
        system.state.soundspeeds *= math.sqrt(2)
        system.state.totalEnergies = system.state.Q[:, -1]
        ctx.scratch['initialEnergy'] = system.state.Q[:, -1].double().sum()
        return system
    r = configuration(accelerationRMS=0., coolingTime=.1).run(replace(drivenTurbulenceCase, buildSystem=build))
    expected = .9*(1+math.exp(-float(r.state.t)/.1))
    torch.testing.assert_close(r.state.state.internalEnergies,
                               torch.full_like(r.state.state.internalEnergies, expected), rtol=3e-5, atol=3e-6)
    assert not r.diverged and r.state.state.sourceEnergyBudget[1] < 0
    assert abs(r.trajectory[-1]['relativeEnergyBudgetResidual']) < 2e-6


def test_mfv_requires_and_runs_the_declared_source_temporal_scheme():
    cfg = configuration(accelerationRMS=.2)
    cfg.solver.update(scheme='MFV', temporalScheme='ssprk')
    r = cfg.run()
    assert not r.diverged and r.trajectory[-1]['minEnergy'] > 0
    assert abs(r.trajectory[-1]['relativeMassChange']) < 1e-6
    assert abs(r.trajectory[-1]['relativeEnergyBudgetResidual']) < 2e-6


def test_repeated_short_driven_runs_and_mean_momentum():
    cfg = configuration(accelerationRMS=.3)
    a, b = cfg.run(), cfg.run()
    torch.testing.assert_close(a.state.state.Q, b.state.state.Q, rtol=0, atol=0)
    row = a.trajectory[-1]
    assert row['rmsMachMass'] > 0 and row['minEnergy'] > 0
    assert row['meanSpeed'] < 1e-7
    assert abs(row['relativeEnergyBudgetResidual']) < 2e-6
    assert float(a.state.t) == cfg.resolve().tLimit


def test_stored_state_retains_forcing_and_accepted_energy_budget(tmp_path):
    import h5py
    from pathlib import Path
    cfg = configuration(accelerationRMS=.2)
    cfg.simulation.update(nSteps=3, store=True, storeMode='states', exportRoot=str(tmp_path))
    r = cfg.run()
    path = Path(r.exportPath)/'trajectory/finalState.h5'
    with h5py.File(path, 'r') as f:
        torch.testing.assert_close(torch.from_numpy(f['state/sourceEnergyBudget'][:]), r.state.state.sourceEnergyBudget.cpu())
        torch.testing.assert_close(torch.from_numpy(f['forcingCoefficients'][:]), r.state.sourceTerms.acceleration.coefficients(float(r.state.t)))
        assert f['dict_forcingParameters'].attrs['seed'] == cfg.resolve().params['seed']
        assert f.attrs['forcingTime'] == float(r.state.t)
