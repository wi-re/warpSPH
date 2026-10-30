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
    ({'dim': 4}, {}, {}, 'dim'),
    ({'adaptiveDt': False}, {}, {}, 'adaptiveDt'),
    ({}, {'kMin': 3, 'kMax': 2}, {}, 'kMin'),
    ({}, {'forcingInterval': 2.}, {}, 'forcingInterval'),
    ({}, {}, {'scheme': 'MFV'}, 'ssprk'),
    ({}, {}, {'temporalScheme': 'hancock'}, 'ssprk'),
    ({'execution': 'graph'}, {}, {}, 'eager'),
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
                f.require_group(name).create_dataset(f'frame_{i+99998:05d}', data=values)
    report = module.analyze(run, tmp_path/'analysis', [8], [0., .3, .7], deposition='kernel')
    assert report['completeAssessmentWindow']
    assert report['budgetScope'] == 'saved snapshots only'
    assert not report['sourceManifestAvailable']
    assert report['budget']['maxRelativeEnergyResidual'] < 1e-15
    assert report['budget']['maxRelativeMassChange'] == 0


def _restartConfiguration(steps, scheme='MFM'):
    cfg = configuration(accelerationRMS=.3)
    cfg.simulation.update(nSteps=steps, tLimit=4.)
    cfg.solver.update(scheme=scheme, temporalScheme='ssprk')
    return cfg


@pytest.mark.parametrize('scheme', ['MFM', 'MFV'])
def test_native_restart_preserves_physics_clock_and_absolute_step_count(tmp_path, scheme):
    import h5py
    from pathlib import Path
    full = _restartConfiguration(160, scheme).run()
    firstConfig = _restartConfiguration(60, scheme)
    firstConfig.simulation.update(store=True, storeMode='states', storeInterval=25,
                                  exportRoot=str(tmp_path/'first'))
    first = firstConfig.run()
    path = Path(first.exportPath)/'trajectory/finalState.h5'
    restoredConfig = _restartConfiguration(100, scheme)
    restoredConfig.simulation.update(resumeFrom=str(path), store=True, storeMode='states',
                                     exportRoot=str(tmp_path/'restored'))
    restored = restoredConfig.run()
    assert float(full.state.t) == float(restored.state.t)
    assert restored.trajectory[0]['t'] == float(first.state.t)
    assert restored.trajectory[0]['step'] == 59
    assert restored.trajectory[1]['step'] == 60
    for key in ['positions', 'Q', 'supports', 'lam', 'internalEnergies', 'densities',
                'velocities', 'pressures', 'sourceEnergyBudget']:
        torch.testing.assert_close(getattr(full.state.state, key),
                                   getattr(restored.state.state, key), rtol=0, atol=0)
    for time in [.01, float(restored.state.t), 2.7]:
        torch.testing.assert_close(full.state.sourceTerms.acceleration.coefficients(time),
                                   restored.state.sourceTerms.acceleration.coefficients(time), rtol=0, atol=0)
    with h5py.File(Path(restored.exportPath)/'trajectory/finalState.h5') as f:
        assert f.attrs['checkpointCompletedSteps'] == 160
        assert f.attrs['frame_num'] == 160
    # A regular interior checkpoint stores the completed count, independent of its filename's index.
    interior = _restartConfiguration(134, scheme)
    interior.simulation['resumeFrom'] = str(Path(first.exportPath)/'trajectory/state_0025.h5')
    result = interior.run()
    assert result.trajectory[1]['step'] == 26
    torch.testing.assert_close(result.state.state.Q, full.state.state.Q, rtol=0, atol=0)


def test_time_limited_restart_retains_a_positive_physical_next_timestep(tmp_path):
    import h5py
    from pathlib import Path
    firstConfig = _restartConfiguration(None)
    firstConfig.simulation.update(tLimit=.04, store=True, storeMode='states', exportRoot=str(tmp_path))
    first = firstConfig.run()
    path = Path(first.exportPath)/'trajectory/finalState.h5'
    with h5py.File(path) as f:
        assert f['checkpointNextDt'][()] > 0
    cfg = _restartConfiguration(None)
    cfg.simulation.update(tLimit=.08, resumeFrom=str(path))
    result = cfg.run()
    assert not result.diverged and float(result.state.t) == .08
    assert abs(result.trajectory[-1]['relativeEnergyBudgetResidual']) < 2e-6


def test_restart_rejects_changed_physics_and_incomplete_context(tmp_path):
    import h5py
    from pathlib import Path
    cfg = _restartConfiguration(10)
    cfg.simulation.update(store=True, storeMode='states', exportRoot=str(tmp_path))
    result = cfg.run()
    path = Path(result.exportPath)/'trajectory/finalState.h5'
    other = _restartConfiguration(10)
    other.simulation['resumeFrom'] = str(path)
    other.params['seed'] = 789
    with pytest.raises(ValueError, match='incompatible'):
        other.run()
    with h5py.File(path, 'a') as f:
        del f.attrs['checkpointVersion']
    other = _restartConfiguration(10)
    other.simulation['resumeFrom'] = str(path)
    with pytest.raises(ValueError, match='missing'):
        other.run()


def test_trajectory_retains_conserved_fields_and_complete_time_dependent_driving(tmp_path):
    import h5py
    from pathlib import Path
    cfg = _restartConfiguration(60)
    cfg.simulation.update(store=True, storeMode='trajectory', exportInterval=.01,
                          exportRoot=str(tmp_path))
    result = cfg.run()
    with h5py.File(Path(result.exportPath)/'trajectory.h5') as f:
        keys = sorted(f['positions'])
        assert len(keys) >= 8
        field = result.state.sourceTerms.acceleration
        knots, nextKnot = [], 0
        for key in keys:
            frame, time = f['frameData'][key], float(f['times'][key][0])
            assert f['times'][key].dtype.itemsize == 8
            assert frame.attrs['forcingTime'] == time
            torch.testing.assert_close(torch.from_numpy(frame['forcingCoefficients'][:]),
                                       field.coefficients(time), rtol=0, atol=0)
            assert frame['forcingParameters'].attrs['seed'] == 1234
            assert frame.attrs['forcingKnotStart'] == nextKnot
            raw = torch.from_numpy(frame['forcingKnots'][:])
            knots.extend(raw.unbind())
            nextKnot += len(raw)
        assert nextKnot >= int(float(result.state.t)/.05)+2
        for index, raw in enumerate(knots):
            torch.testing.assert_close(raw, field.knots[index], rtol=0, atol=0)
        first, last = f['frameData'][keys[0]], f['frameData'][keys[-1]]
        assert not torch.equal(torch.from_numpy(first['forcingCoefficients'][:]),
                               torch.from_numpy(last['forcingCoefficients'][:]))
        for name in ['Q', 'sourceEnergyBudget', 'UIDs', 'masses', 'positions', 'velocities', 'densities']:
            torch.testing.assert_close(torch.from_numpy(f[name][keys[-1]][:]),
                                       getattr(result.state.state, name).cpu(), rtol=0, atol=0)


def test_native_driving_audit_detects_missing_between_frame_knots(tmp_path):
    import importlib.util
    import h5py
    from pathlib import Path
    path = Path(__file__).resolve().parents[1]/'scripts/analyze_drivenTurbulence.py'
    spec = importlib.util.spec_from_file_location('audit_native_forcing', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = _restartConfiguration(60)
    cfg.simulation.update(store=True, storeMode='trajectory', exportInterval=.01,
                          exportRoot=str(tmp_path))
    result = cfg.run()
    path = Path(result.exportPath)
    audit = module.forcingObservation(path)
    assert audit['completeKnots'] and audit['knots'] >= 4
    assert audit['maxCoefficientError'] < 1e-14
    # Exercise the actual native run through the reusable offline command too.
    analysis = module.analyze(path, tmp_path/'analysis', [8], [.01, .06, float(result.state.t)], deposition='kernel')
    assert analysis['conditionalForcing']['completeKnots']
    assert analysis['budgetScope'] == 'saved snapshots only'
    assert analysis['budget']['maxRelativeEnergyResidual'] < 2e-6
    with h5py.File(path/'trajectory.h5', 'a') as f:
        keys = sorted(f['frameData'])
        f['frameData'][keys[-1]].attrs['forcingKnotStart'] += 1
    with pytest.raises(ValueError, match='missing or duplicate'):
        module.forcingObservation(path)



def test_trajectory_frame_metadata_orders_long_step_indices_numerically(tmp_path):
    import h5py
    import numpy as np
    from warpSPH.io.importIO import loadTrajectory
    indices = [99999, 100000, 100001]
    with h5py.File(tmp_path/'trajectory.h5', 'w') as f:
        f.attrs['scheme'] = 'MFM'
        for name in ('combinedMasses', 'combinedSupports'):
            f[name] = np.ones(1, dtype=np.float32)
        for name in ('combinedKinds', 'combinedMaterials', 'combinedUIDs'):
            f[name] = np.zeros(1, dtype=np.int32)
        for i, index in enumerate(indices):
            f.require_group('positions')[f'frame_{index:05d}'] = np.zeros((1, 2), dtype=np.float32)
    handle, meta = loadTrajectory(str(tmp_path), 'cpu')
    try:
        assert meta['frameKeys'] == [f'frame_{i:05d}' for i in indices]
    finally:
        handle.close()


@pytest.mark.parametrize('scheme', ['MFM', 'MFV'])
@pytest.mark.parametrize('thermalModel', ['relaxation', 'adiabatic'])
def test_cold_driving_preserves_raw_internal_energy_in_every_rhs(scheme, thermalModel):
    observed = []
    def build(ctx):
        system = drivenTurbulenceCase.buildSystem(ctx)
        rhs = ctx.stepFunction
        def checked(state, dt, **kwargs):
            result = rhs(state, dt, **kwargs)
            stage = state.meshlessContext
            observed.append((float(stage.u.min()), float(stage.P.min()), float(stage.rho.min())))
            return result
        ctx.bundle = replace(ctx.bundle, stepFunction=checked)
        return system
    cfg = configuration(soundSpeed=.01, accelerationRMS=6., thermalModel=thermalModel)
    cfg.simulation.update(tLimit=.02, maxDt=.01, velocityAlarmStopRatio=100)
    cfg.solver.update(scheme=scheme, temporalScheme='ssprk')
    result = cfg.run(replace(drivenTurbulenceCase, buildSystem=build))
    assert not result.diverged and float(result.state.t) == .02
    assert observed and all(energy > 0 and pressure > 1e-25 and density > 0
                            for energy, pressure, density in observed)
    assert all(row['minEnergy'] > 0 for row in result.trajectory)
    assert result.trajectory[-1]['rmsMachMass'] > 1


def test_cold_uniform_force_with_a_bath_preserves_the_analytic_temperature():
    acceleration = torch.tensor([6., 0.])
    def build(ctx):
        system = drivenTurbulenceCase.buildSystem(ctx)
        system.sourceTerms.acceleration = lambda positions, time: acceleration.to(positions).expand_as(positions)
        system.sourceTerms.removeMeanAcceleration = False
        return system
    cfg = configuration(soundSpeed=.01, accelerationRMS=6., thermalModel='relaxation')
    cfg.simulation.update(tLimit=.02, maxDt=.01, velocityAlarmStopRatio=100)
    result = cfg.run(replace(drivenTurbulenceCase, buildSystem=build))
    u0 = .01**2/((5/3)*(2/3))
    torch.testing.assert_close(result.state.state.internalEnergies,
                               torch.full_like(result.state.state.internalEnergies, u0),
                               rtol=2e-3, atol=5e-9)
    expected = (float(result.state.t)*acceleration.to(result.state.state.velocities)).expand_as(result.state.state.velocities)
    torch.testing.assert_close(result.state.state.velocities, expected, rtol=3e-4, atol=1e-7)
    assert abs(float(result.state.state.sourceEnergyBudget[1])) < .002*float(result.ctx.scratch['initialEnergy'])


@pytest.mark.parametrize('scheme', ['MFM', 'MFV'])
def test_cold_source_limit_anticipates_a_force_appearing_across_a_knot(scheme):
    observed = []
    def build(ctx):
        system = drivenTurbulenceCase.buildSystem(ctx)
        field = ctx.scratch['driving']
        # A prescribed solenoidal pulse: zero before .05, growing towards .10.
        field.knots[0].zero_()
        field.knots[1].zero_()
        field.knots[2].zero_()
        mode = int(torch.nonzero((field.modes == torch.tensor([1., 0.])).all(dim=1))[0])
        field.knots[2][0, mode, 1] = 8.
        system.t = .049
        # A zero-force past allows a large previous step; test future growth explicitly.
        ctx.config.dt = .01
        ctx.config.dt = drivenTurbulenceCase.timestep(ctx, system)
        rhs = ctx.stepFunction
        def checked(state, dt, **kwargs):
            result = rhs(state, dt, **kwargs)
            observed.append(float(state.meshlessContext.u.min()))
            return result
        ctx.bundle = replace(ctx.bundle, stepFunction=checked)
        return system
    cfg = configuration(soundSpeed=.01, accelerationRMS=6., thermalModel='adiabatic')
    cfg.simulation.update(tLimit=.06, maxDt=.01, velocityAlarmStopRatio=100)
    cfg.solver.update(scheme=scheme, temporalScheme='ssprk')
    result = cfg.run(replace(drivenTurbulenceCase, buildSystem=build))
    assert not result.diverged and float(result.state.t) == .06
    assert observed and min(observed) > 0
    assert all(row['minEnergy'] > 0 for row in result.trajectory)


@pytest.mark.parametrize('dim', [2, 3])
def test_force_bound_covers_spatial_samples_across_a_knot_without_changing_noise(dim):
    field, untouched = FourierOU(dim=dim, seed=481), FourierOU(dim=dim, seed=481)
    points = torch.rand(91, dim, dtype=torch.float64)
    for time in [.049, .211, .07, .049]:
        bound = field.accelerationBound(time)
        for sample in [time, time+.03, time+field.parameters['updateInterval']]:
            acceleration = field(points, sample)
            assert float(acceleration.norm(dim=1).max()) <= bound*(1+1e-12)
            torch.testing.assert_close(acceleration, untouched(points, sample), rtol=0, atol=0)


def test_force_bound_cache_tracks_prescribed_knots_and_checkpoint_reconstruction():
    field = FourierOU(seed=481)
    before = field.accelerationBound(.03)
    field.knots[1].mul_(100)
    after = field.accelerationBound(.03)
    restored = FourierOU.from_state_dict(field.state_dict())
    assert after > before
    assert after == restored.accelerationBound(.03)


def test_exact_box_power_does_not_include_diagonal_modes_in_the_first_shell():
    import numpy as np
    from warpSPH.utils.turbulence import spectra2D
    x, y = np.meshgrid((np.arange(32)+.5)/32, (np.arange(32)+.5)/32)
    diagonal = np.stack((np.sin(2*np.pi*(x+y)), np.zeros_like(x)), axis=-1)
    spectra = spectra2D(diagonal)
    assert spectra['velocityEnergy'][1] == pytest.approx(.25)
    assert spectra['boxModeEnergy'] == pytest.approx(0, abs=1e-28)
    axial = np.stack((np.sin(2*np.pi*y), np.zeros_like(x)), axis=-1)
    assert spectra2D(axial)['boxModeEnergy'] == pytest.approx(.25)


@pytest.mark.parametrize('badTime', [0., 9.9])
def test_analysis_requires_valid_interpolation_brackets_but_excludes_earlier_startup(tmp_path, badTime):
    import importlib.util
    import json
    import numpy as np
    from pathlib import Path
    path = Path(__file__).resolve().parents[1]/'scripts/analyze_drivenTurbulence.py'
    spec = importlib.util.spec_from_file_location('driven_coverage_analysis', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    run = tmp_path/'flow'
    (run/'samples').mkdir(parents=True)
    (run/'input.json').write_text(json.dumps({'simulation': {'nx': 8}, 'params': {}}))
    x, y = np.meshgrid((np.arange(8)+.5)/8, (np.arange(8)+.5)/8)
    regular = np.stack((x, y), axis=-1).reshape(-1, 2)
    for i, t in enumerate([0., 5., 9.9, 10.1, 14., 18., 22., 26., 30., 34., 38., 40.]):
        positions = np.zeros_like(regular) if t == badTime else regular
        velocity = np.stack((np.sin(2*np.pi*y)*(1+.05*np.sin(t)), np.zeros_like(x)), axis=-1).reshape(-1, 2)
        np.savez(run/'samples'/f'{i:04d}.npz', t=t, positions=positions, velocities=velocity,
                 masses=np.full(64, 1/64), densities=np.ones(64), pressures=np.full(64, .6),
                 internalEnergies=np.full(64, .9), supports=np.full(64, .2))
    report = module.analyze(run, tmp_path/'output', [8], [10., 20., 30., 40.])
    grid = report['grid']['8']
    assert grid['maxEmptyCellFraction'] > 0
    if badTime == 0.:
        assert grid['comparisonCoverageComplete']
        assert np.isfinite(grid['means']['velocityEnergy']).all()
        assert 'boxModeScreen' in grid
    else:
        assert not grid['comparisonCoverageComplete']
        assert grid['invalidSpectralSnapshots'] == 1
        assert 'means' not in grid


def test_stationarity_screen_reports_insufficient_assessment_samples_without_failing():
    import numpy as np
    from warpSPH.utils.turbulence import stationarityScreen
    for times in [np.arange(0., 41., 4.), np.array([0., 40.])]:
        screen = stationarityScreen(times, 1+.01*np.sin(times), blocks=[30., 35., 40.])
        assert len(screen['blockMeans']) == 2
        assert screen['tau'] is None and screen['effectiveSamples'] is None
        assert screen['correlationCensored'] and not screen['passed']
