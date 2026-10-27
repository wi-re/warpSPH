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


def test_particle_measurements_distinguish_mass_and_volume_weights():
    import numpy as np
    from warpSPH.utils.turbulence import particleStatistics, weightedPDF
    mass, rho = np.array([.5, .5]), np.array([2., 2/3])
    velocity = np.array([[1., 0.], [-1., 0.]])
    stats = particleStatistics(mass, rho, velocity, np.ones(2))
    assert stats['volumeQuadratureRelativeError'] == pytest.approx(0)
    assert stats['bulkVelocityMass'] == pytest.approx([0., 0.])
    assert stats['bulkVelocityVolume'] == pytest.approx([-.5, 0.])
    assert stats['rmsVelocityVolume'] == pytest.approx(np.sqrt(.75))
    pdf = weightedPDF([-3, -1, 0, 1, 3], np.ones(5), [-1, 0, 1])
    np.testing.assert_allclose(pdf, [.2, .2, .4, .2])
    assert pdf.sum() == pytest.approx(1)


def test_periodic_cic_preserves_mass_momentum_and_records_filter_loss():
    import numpy as np
    from warpSPH.utils.turbulence import depositCIC2D
    rng = np.random.default_rng(17)
    x, mass, v = rng.random((2000, 2)), rng.uniform(.5, 1.5, 2000), rng.normal(size=(2000, 2))
    a = depositCIC2D(x, mass, v, gridSize=16)
    b = depositCIC2D(x+[2, -3], mass, v, gridSize=16)
    np.testing.assert_allclose(a['density'], b['density'], rtol=1e-13, atol=1e-11)
    np.testing.assert_allclose(a['velocity'], b['velocity'], rtol=1e-11, atol=1e-13)
    assert a['relativeMassError'] == pytest.approx(0, abs=1e-14)
    np.testing.assert_allclose(a['momentumError'], 0, atol=1e-12)
    assert 0 < a['unresolvedKineticFraction'] < 1
    assert a['emptyCellFraction'] == 0


def test_spectral_normalization_projection_and_structure_functions():
    import numpy as np
    from warpSPH.utils.turbulence import depositCIC2D, spectra2D, structureFunctions2D
    n = 32
    x, y = np.meshgrid((np.arange(n)+.5)/n, (np.arange(n)+.5)/n)
    velocity = np.stack((np.sin(2*np.pi*y), 2*np.sin(4*np.pi*y)), axis=-1)
    spectra = spectra2D(velocity, 1+.2*np.cos(2*np.pi*x))
    assert spectra['parsevalError'] == pytest.approx(0, abs=1e-14)
    assert spectra['solenoidalEnergy'][1] == pytest.approx(.25)
    assert spectra['compressionalEnergy'][2] == pytest.approx(1.)
    assert spectra['velocityEnergy'].sum() == pytest.approx(1.25)
    assert spectra['densityPower'][1] == pytest.approx(.02)
    sf = structureFunctions2D(velocity)
    np.testing.assert_allclose(sf['transverse'], .5*(1-np.cos(2*np.pi*sf['separation'])), atol=1e-14)
    np.testing.assert_allclose(sf['longitudinal'], 2*(1-np.cos(4*np.pi*sf['separation'])), atol=1e-14)
    empty = depositCIC2D([[0, 0]], [1], [[1, 0]], gridSize=8)
    assert empty['emptyCellFraction'] > 0
    with pytest.raises(ValueError, match='fully covered'):
        spectra2D(empty['velocity'])


def test_correlation_and_time_weighted_stationarity_screen():
    import numpy as np
    from warpSPH.utils.turbulence import correlationTime, stationarityScreen
    t = np.arange(0, 40.0001, .05)
    periodic = 1+.05*np.sin(2*np.pi*t)
    c = correlationTime(t, periodic)
    assert not c['censored'] and .1 < c['tau'] < .2
    assert stationarityScreen(t, periodic)['passed']
    drift = stationarityScreen(t, 1+t*.1)
    assert not drift['passed'] and drift['relativeSpread'] > .5
    assert correlationTime(t, np.ones_like(t))['censored']
    # Irregular timestamps must still give the exact time mean of a linear field.
    irregular = np.r_[np.linspace(0, 15, 301), np.linspace(15.01, 40, 2000)]
    s = stationarityScreen(irregular, 1+irregular*.1)
    assert s['blockMeans'] == pytest.approx([2.5, 3.5, 4.5])


def test_field_correlation_matches_scalar_for_one_pattern():
    import numpy as np
    from warpSPH.utils.turbulence import correlationTime, fieldCorrelationTime
    times = np.arange(0, 20, .05)
    values = np.sin(2*np.pi*times)
    fields = values[:, None, None]*np.array([[[1., 2., -3.]]])
    a, b = correlationTime(times, values), fieldCorrelationTime(times, fields)
    np.testing.assert_allclose(a['acf'], b['acf'], rtol=1e-12, atol=1e-14)
    assert a['tau'] == pytest.approx(b['tau'])


def test_cic_transfer_function_exposes_filtered_velocity_power():
    import numpy as np
    from warpSPH.utils.turbulence import depositCIC2D, spectra2D
    n, mode = 16, 3
    x, y = np.meshgrid((np.arange(n)+.5)/n, (np.arange(n)+.5)/n)
    positions = np.stack((x+.25/n, y), axis=-1).reshape(-1, 2)
    velocity = np.stack((np.zeros_like(x), np.sin(2*np.pi*mode*x)), axis=-1).reshape(-1, 2)
    grid = depositCIC2D(positions, np.full(n*n, 1/(n*n)), velocity, gridSize=n)
    spectra = spectra2D(grid['velocity'])
    gain2 = .75**2+.25**2+2*.75*.25*np.cos(2*np.pi*mode/n)
    assert spectra['solenoidalEnergy'][mode] == pytest.approx(.25*gain2)
    assert spectra['compressionalEnergy'].sum() < 1e-25
    assert grid['unresolvedKineticFraction'] == pytest.approx(1-gain2)


def test_offline_qualification_command_writes_synthetic_measurements(tmp_path):
    import importlib.util
    import json
    import numpy as np
    from pathlib import Path
    path = Path(__file__).resolve().parents[1]/'scripts/analyze_drivenTurbulence.py'
    spec = importlib.util.spec_from_file_location('driven_analysis', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    run = tmp_path/'flow'
    (run/'samples').mkdir(parents=True)
    (run/'input.json').write_text(json.dumps({'simulation': {'nx': 8}, 'params': {}}))
    x, y = np.meshgrid((np.arange(8)+.5)/8, (np.arange(8)+.5)/8)
    positions = np.stack((x, y), axis=-1).reshape(-1, 2)
    for i, t in enumerate(np.linspace(0, .7, 8)):
        velocity = np.stack((np.sin(2*np.pi*y)*(1+.05*np.sin(20*t)), np.zeros_like(x)), axis=-1).reshape(-1, 2)
        np.savez(run/'samples'/f'{i:04d}.npz', t=t, positions=positions, velocities=velocity,
                 masses=np.full(64, 1/64), densities=np.ones(64), pressures=np.full(64, .6), internalEnergies=np.full(64, .9), supports=np.full(64, .2))
    report = module.analyze(run, tmp_path/'output', [8], [0., .3, .7])
    assert report['completeAssessmentWindow'] and report['samples'] == 8
    assert report['grid']['8']['maxEmptyCellFraction'] == 0
    assert report['grid']['8']['maxMassError'] == pytest.approx(0)
    assert report['budget'] is None
    assert sum(report['meanPDFs']['logDensityMass']) == pytest.approx(1)
    assert (tmp_path/'output/particle-statistics.npz').exists()
    kernel = module.analyze(run, tmp_path/'kernel', [8], [0., .3, .7], deposition='kernel')
    assert kernel['deposition'] == 'kernel'
    assert kernel['grid']['8']['maxEmptyCellFraction'] == 0
    assert kernel['grid']['8']['minFilterRadius'] == pytest.approx(.25)


def test_adaptive_kernel_projection_preserves_budgets_and_filter_changes():
    import numpy as np
    from warpSPH.utils.turbulence import depositKernel2D, spectra2D
    n = 64
    x, y = np.meshgrid((np.arange(n)+.5)/n, (np.arange(n)+.5)/n)
    position = np.stack((x, y), axis=-1).reshape(-1, 2)
    velocity = np.stack((np.sin(4*np.pi*y), np.zeros_like(x)), axis=-1).reshape(-1, 2)
    mass, support = np.full(n*n, 1/(n*n)), np.full(n*n, .125)
    powers = []
    for gridSize in [16, 32]:
        a = depositKernel2D(position, mass, velocity, support, gridSize=gridSize)
        b = depositKernel2D(position+[2., -3.], mass, velocity, support, gridSize=gridSize)
        assert a['emptyCellFraction'] == 0
        assert abs(a['relativeMassError']) < 2e-14
        np.testing.assert_allclose(a['momentumError'], 0., atol=1e-14)
        np.testing.assert_allclose(a['velocity'], b['velocity'], atol=2e-14)
        power = spectra2D(a['velocity'])['velocityEnergy'][2]
        powers.append(power)
        assert .1 < power < .25
        assert a['unresolvedKineticFraction'] == pytest.approx(1-4*power, abs=1e-12)
    # Same physical filter at two deposition resolutions resolves this mode.
    assert powers[0] == pytest.approx(powers[1], rel=.002)


def test_adaptive_projection_reports_cell_broadening_without_filling_holes():
    import numpy as np
    from warpSPH.utils.turbulence import depositKernel2D, spectra2D
    a = depositKernel2D([[0., 0.]], [1.], [[1., 0.]], [.01], gridSize=16)
    assert a['broadenedParticleFraction'] == 1
    assert a['minFilterRadius'] == pytest.approx(.125)
    assert a['emptyCellFraction'] > 0
    assert np.isnan(a['velocity']).any()
    with pytest.raises(ValueError, match='fully covered'):
        spectra2D(a['velocity'])


def test_offline_analysis_reads_native_fields_and_scopes_budget_checks(tmp_path):
    import importlib.util
    import json
    import numpy as np
    import h5py
    from pathlib import Path
    path = Path(__file__).resolve().parents[1]/'scripts/analyze_drivenTurbulence.py'
    spec = importlib.util.spec_from_file_location('driven_native_analysis', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    run = tmp_path/'native'
    run.mkdir()
    (run/'caseSpec.json').write_text(json.dumps({'nx': 8, 'L': 1., 'params': {}}))
    x, y = np.meshgrid((np.arange(8)+.5)/8, (np.arange(8)+.5)/8)
    positions = np.stack((x, y), axis=-1).reshape(-1, 2)
    mass = np.full(64, 1/64)
    with h5py.File(run/'trajectory.h5', 'w') as f:
        f.attrs.update(initialEnergy=1.15, initialMass=1.)
        for i, t in enumerate(np.linspace(0, .7, 8)):
            velocity = np.stack((np.sin(2*np.pi*y)*(1+.05*np.sin(20*t)), np.zeros_like(x)), axis=-1).reshape(-1, 2)
            energy = mass*(.9+.5*np.sum(velocity**2, axis=-1))
            fields = dict(positions=positions, velocities=velocity, masses=mass,
                          densities=np.ones(64), pressures=np.full(64, .6), internalEnergies=np.full(64, .9),
                          supports=np.full(64, .2), Q=np.column_stack((mass, mass[:, None]*velocity, energy)),
                          sourceEnergyBudget=np.array([energy.sum()-1.15, 0.]), times=np.array([t]))
            for name, values in fields.items():
                f.require_group(name).create_dataset(f'frame_{i:05d}', data=values)
    report = module.analyze(run, tmp_path/'analysis', [8], [0., .3, .7], deposition='kernel')
    assert report['completeAssessmentWindow']
    assert report['budgetScope'] == 'saved snapshots only'
    assert not report['sourceManifestAvailable']
    assert report['budget']['maxRelativeEnergyResidual'] < 1e-15
    assert report['budget']['maxRelativeMassChange'] == 0
