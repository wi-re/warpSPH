"""Real spatial RHSs must not advance integrated state behind their driver."""
from functools import wraps

import pytest
import torch

from warpSPH.runner import RunConfiguration
from warpSPH.cases.compressible.linearWave import linearWaveCase

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
METHODS = ['Monaghan', 'CompSPH', 'CRKSPH', 'PESPH-E', 'PESPH-A',
           'MFM', 'MFV', 'GSPH', 'InutsukaGSPH']


def setup(method, integrator='rungeKutta4'):
    solver = dict(scheme=method)
    if method.startswith('PESPH-'):
        solver = dict(scheme='PESPH', variant='PressureEnergy' if method.endswith('E') else 'PressureEntropy',
                      dissipation=False)
    ctx = RunConfiguration('linearWave', simulation=dict(device=DEVICE, nx=32, dim=1,
        integrationScheme=integrator, supportMode='KernelMeanSymmetric', dt=1e-4, quiet=True), solver=solver).prepare(linearWaveCase)
    system = linearWaveCase.buildSystem(ctx)
    linearWaveCase.initialConditions(ctx, system)
    return ctx, system


def integrated_fields(method):
    if method in ('MFM', 'MFV'):
        return ('positions', 'Q')
    if method.startswith('PESPH-'):
        return ('positions', 'velocities', 'totalEnergies' if method.endswith('E') else 'entropies')
    return ('positions', 'velocities', 'internalEnergies')


@pytest.mark.parametrize('method', METHODS)
def test_actual_rhs_does_not_advance_state_and_driver_owns_each_stage(method):
    ctx, system = setup(method)
    fields = integrated_fields(method)
    initial = {key: getattr(system.state, key).clone() for key in fields}
    times = []

    @wraps(ctx.stepFunction)
    def observed(stage, dt, *args, **kwargs):
        before = {key: getattr(stage.state, key).clone() for key in fields}
        time = stage.t
        result = ctx.stepFunction(stage, dt, *args, **kwargs)
        assert stage.t == time
        for key in fields:
            torch.testing.assert_close(getattr(stage.state, key), before[key], rtol=0, atol=0)
        times.append((float(time), float(dt)))
        return result

    result = ctx.integrator.function(state=system, dt=1e-4, f=observed,
                                    config=ctx.config, schemeConfig=ctx.schemeConfig)
    assert result.state.t == pytest.approx(1e-4)
    assert system.t == 0.
    for key in fields:
        torch.testing.assert_close(getattr(system.state, key), initial[key], rtol=0, atol=0)
    assert ctx.integrator.__class__.__module__.startswith('warpSPHIntegrators.')
    if method in ('MFM', 'MFV'):
        assert len(times) == len(result.stages) == (2 if method == 'MFM' else 1)
        assert times[0][1] == (0. if method == 'MFM' else 1e-4)
    else:
        assert ctx.integrator.identifier.name == 'rungeKutta4'
        assert len(result.stages) == 4
        # Compatible pair work may require one explicit driver-owned replay.
        expected_calls = (4, 8) if method in ('CompSPH', 'CRKSPH') else (4,)
        assert len(times) in expected_calls
    assert len({t for t, _ in times}) >= (2 if method != 'MFV' else 1)
    # The matrix must retain the actual integrated state, not only diagnostics.
    import sys
    from pathlib import Path
    scripts = str(Path(__file__).resolve().parents[1] / 'scripts')
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from evaluate_pesph import arrays
    ctx.scratch['evalPhase'] = torch.zeros_like(result.state.state.masses)
    snapshot = arrays(ctx, result.state)
    if method in ('MFM', 'MFV'):
        torch.testing.assert_close(torch.from_numpy(snapshot['Q']).to(DEVICE), result.state.state.Q)
    if result.state.state.totalEnergies is not None:
        assert 'totalEnergies' in snapshot


@pytest.mark.parametrize('method', ['GSPH', 'InutsukaGSPH'])
def test_godunov_rhs_is_independent_of_trial_step_size(method):
    ctx, system = setup(method)
    updates = []
    for dt in (1e-5, 1e-3):
        stage = system.initializeNewState()
        update, *_ = ctx.stepFunction(stage, dt, ctx.config, ctx.schemeConfig)
        updates.append(update)
    for key in ('dxdt', 'dvdt', 'dudt', 'dEdt', 'drhodt'):
        torch.testing.assert_close(getattr(updates[0], key), getattr(updates[1], key), rtol=0, atol=0)


@pytest.mark.parametrize('method', ['GSPH', 'InutsukaGSPH'])
def test_godunov_finite_step_energy_is_rejected_before_sampling_and_at_rhs_entry(method):
    from warpSPH.configurations.gsph import GSPHConfig, InutsukaGSPHConfig
    from warpSPH.schemes.compressible.gsph import compressibleSPH_GSPH
    from warpSPH.schemes.compressible.gsphInutsuka import compressibleSPH_InutsukaGSPH
    cls, rhs = (GSPHConfig, compressibleSPH_GSPH) if method == 'GSPH' else (InutsukaGSPHConfig, compressibleSPH_InutsukaGSPH)
    scheme = cls()
    scheme.diffusionParams.timeCentredEnergy = True
    with pytest.raises(ValueError, match='finite-step'):
        scheme.validate()
    with pytest.raises(ValueError, match='finite-step'):
        rhs(None, .1, None, scheme)


@pytest.mark.parametrize('method', ['MFM', 'MFV'])
def test_meshless_registered_rhs_never_falls_back_to_whole_step_map(method, monkeypatch):
    from warpSPH.modules.meshless import compatibility
    from warpSPH.schemes import buildScheme
    bundle = buildScheme(method)
    scheme = bundle.SimulationConfig()
    monkeypatch.setattr(compatibility, 'exactDeltaStep', lambda *a, **kw: pytest.fail('hidden integration'))
    with pytest.raises(ValueError, match='ConservedLagrangian'):
        bundle.stepFunction(None, .1, None, scheme)
    scheme.mfvUpdate = 'primitive'
    with pytest.raises(ValueError, match='whole-step map'):
        scheme.validate()
    with pytest.raises(ValueError, match='whole-step map'):
        bundle.stepFunction(None, .1, None, scheme, conservedIntegration=True)


def test_dual_time_map_cannot_be_called_as_a_multistage_rhs():
    from types import SimpleNamespace
    from warpSPH.configurations.artificialCompressible import ArtificialCompressibleSPHConfig
    from warpSPH.schemes.incompressible.artificialCompressible import artificialCompressible_step
    with pytest.raises(ValueError, match='single-evaluation'):
        artificialCompressible_step(None, .1, SimpleNamespace(integrationScheme='rungeKutta4'),
                                    ArtificialCompressibleSPHConfig())


@pytest.mark.parametrize('method', ['deltaSPH', 'sun2017DeltaSPH', 'iisph'])
def test_other_derivative_schemes_leave_kinematics_to_the_driver(method):
    from warpSPH.cases import importAll
    from warpSPH.runner import getCase, prepareContext, CaseSpec
    importAll()
    weakly = method in ('deltaSPH', 'sun2017DeltaSPH')
    case = getCase('tgv-wc' if weakly else 'tgv')
    params = dict(case.params)
    params.update({'shuffleIters': 0, 'shifting': False} if weakly else {'relaxSteps': 0, 'jitter': 0.})
    spec = CaseSpec(caseName=case.name, scheme=case.scheme, params=params).merged(**case.defaults)
    ctx = prepareContext(case, spec.merged(device=DEVICE, nx=8, dim=2, scheme=method,
                                         integrationScheme='rungeKutta4', dt=1e-4, quiet=True))
    system = ctx.case.buildSystem(ctx)
    if ctx.case.initialConditions:
        ctx.case.initialConditions(ctx, system)
    calls = []

    @wraps(ctx.stepFunction)
    def observed(stage, dt, *args, **kwargs):
        before = stage.state.positions.clone(), stage.state.velocities.clone()
        time = stage.t
        result = ctx.stepFunction(stage, dt, *args, **kwargs)
        for got, want in zip((stage.state.positions, stage.state.velocities), before):
            torch.testing.assert_close(got, want, rtol=0, atol=0)
        assert stage.t == time
        calls.append(time)
        return result

    step = ctx.integrator(state=system, dt=1e-4, f=observed,
                          config=ctx.config, schemeConfig=ctx.schemeConfig)
    assert len(calls) == len(step.stages) == (1 if method == 'iisph' else 4)
    assert step.state.t == pytest.approx(1e-4) and system.t == 0.


@pytest.mark.parametrize('method', ['MFM', 'MFV'])
@pytest.mark.parametrize('temporal', ['hancock', 'ssprk', 'kdk'])
def test_meshless_accepted_state_obeys_flux_weights_and_has_current_diagnostics(method, temporal):
    from warpSPH.modules.meshless.geometry import buildGeometry
    from warpSPH.modules.meshless.primitives import primitives
    ctx, system = setup(method)
    ctx.schemeConfig.temporalScheme = temporal
    driver = ctx.schemeConfig.integrationMethod()
    step = driver(state=system, dt=1e-4, f=ctx.stepFunction,
                  config=ctx.config, schemeConfig=ctx.schemeConfig)
    out = step.state.state
    rates = [stage.update.dQdt for stage in step.stages]
    expected = system.state.Q + 1e-4 * (rates[0] if temporal == 'hancock' else .5 * (rates[0] + rates[1]))
    torch.testing.assert_close(out.Q, expected)
    eps = torch.finfo(out.Q.dtype).eps
    torch.testing.assert_close(out.Q.sum(0), system.state.Q.sum(0), rtol=64*eps, atol=64*eps)
    if method == 'MFM':
        torch.testing.assert_close(out.Q[:, 0], system.state.Q[:, 0], rtol=0, atol=0)
    geom = buildGeometry(out.positions, out.supports, out.lam, ctx.config, ctx.schemeConfig, system.domain)
    rho, vel, pressure, energy = primitives(out.Q, geom.volume, float(ctx.schemeConfig.gamma), returnEnergy=True)
    for got, want in ((out.masses, out.Q[:, 0]), (out.totalEnergies, out.Q[:, -1]),
                      (out.velocities, vel), (out.internalEnergies, energy),
                      (out.densities, rho), (out.pressures, pressure)):
        torch.testing.assert_close(got, want, rtol=32*eps, atol=32*eps)


@pytest.mark.parametrize('method', ['MFM', 'MFV'])
@pytest.mark.parametrize('temporal', ['hancock', 'ssprk', 'kdk'])
def test_current_diagnostics_preserve_the_retained_time_map(method, temporal):
    """Compare independent production/legacy lifecycles over multiple steps."""
    from warpSPHIntegrators.schemes import IntegrationSchemeType, get_scheme
    from warpSPH.modules.meshless.compatibility import legacyMeshlessStep
    ctx, system = setup(method)
    ctx.schemeConfig.temporalScheme = temporal
    driver = ctx.schemeConfig.integrationMethod()
    legacy = system.initializeNewState()
    ctx.config.integrationScheme = IntegrationSchemeType.forwardEuler
    euler = get_scheme(IntegrationSchemeType.forwardEuler)
    eps = torch.finfo(system.state.Q.dtype).eps
    for _ in range(3):
        system = driver(state=system, dt=1e-4, f=ctx.stepFunction,
                        config=ctx.config, schemeConfig=ctx.schemeConfig).state
        legacy = euler.function(state=legacy, dt=1e-4, f=legacyMeshlessStep,
                                config=ctx.config, schemeConfig=ctx.schemeConfig).state
        torch.testing.assert_close(system.state.Q, legacy.state.Q, rtol=64*eps, atol=64*eps)
        torch.testing.assert_close(system.state.positions, legacy.state.positions, rtol=64*eps, atol=64*eps)
        torch.testing.assert_close(system.state.supports, legacy.state.supports, rtol=64*eps, atol=64*eps)
        assert system.t == legacy.t


@pytest.mark.parametrize('method', ['MFM', 'MFV'])
def test_timestep_reuses_accepted_geometry_without_rebuilding(method, monkeypatch):
    from warpSPH.modules.timestep.meshless import computeTimestep, MeshlessWarp
    ctx, system = setup(method)
    system = ctx.integrator(state=system, dt=1e-4, f=ctx.stepFunction,
                            config=ctx.config, schemeConfig=ctx.schemeConfig).state
    monkeypatch.setattr(MeshlessWarp, 'build', lambda *a, **kw: pytest.fail('rebuilt unchanged accepted geometry'))
    assert computeTimestep(system, ctx.config, ctx.schemeConfig, dt=1e-4) > 0
