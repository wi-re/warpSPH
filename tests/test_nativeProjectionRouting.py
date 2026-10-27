"""Public native variants, migration boundaries, and nontrivial stage ownership.

Tiny state-map/initialization checks intentionally omit rendering. Long physical
qualification remains the recorded prepared matrix, including its failures.
"""
from types import SimpleNamespace as NS
import json
import pytest
import torch

NATIVE = {'dfsph': 'df_np_cd', 'dfsphPredictor': 'np_df_cd',
          'dfsphShift': 'np_df_shift', 'iisph': None}
RETIRED = ['divergenceFree', 'dfsphReference', 'omniIncompressible', 'band2018pb']


@pytest.mark.parametrize('name', RETIRED)
def test_retired_entry_points_fail_with_migration_guidance(name):
    from warpSPH.schemes import buildScheme
    from warpSPH.runner.runner import _resolveScheme
    from warpSPH.runner.caseSpec import schemeNames
    assert name not in schemeNames()
    for resolve in (buildScheme, _resolveScheme):
        with pytest.raises(ValueError, match='retired.*select'):
            resolve(name.upper())


def test_retired_numeric_ids_are_not_reused():
    from warpSPH.enumTypes import IncompressibleSPHScheme as S
    assert {m.name for m in S} == set(NATIVE)
    for old in (0, 1, 3, 4):
        with pytest.raises(ValueError):
            S(old)


@pytest.mark.parametrize('name,order', NATIVE.items())
def test_public_variants_resolve_independent_roundtrippable_native_policies(name, order):
    from warpSPH.schemes import buildScheme
    from warpSPHIntegrators.schemes import ProjectedStep, IntegrationSchemeType as I
    b = buildScheme(name)
    a, c = b.SimulationConfig(), b.SimulationConfig()
    assert not hasattr(a, 'solverConfig')
    assert a.iisph is not c.iisph and a.dfsph is not c.dfsph
    assert a.dfsph.density is not a.dfsph.divergence
    if order is not None:
        assert a.dfsph.ordering == order
        assert b.integrationMethod is ProjectedStep
    else:
        assert b.integrationMethod is I.semiImplicitEuler
    a.dfsph.divergence.maxIterations = 237
    a.dfsph.minDivergenceDensityRatio = .9
    a.iisph.boundaryPressure = 'contactShepard'
    values = json.loads(json.dumps(b.exportFunction(a)))
    assert 'solverConfig' not in values and 'bandViscosity' not in values
    restored = b.importFunction(values)
    assert restored.dfsph == a.dfsph and restored.iisph == a.iisph
    assert c.dfsph.divergence.maxIterations == 100


def test_old_native_exports_decode_explicitly_but_legacy_configs_do_not():
    from warpSPH.schemes import buildScheme
    b = buildScheme('dfsphPredictor')
    values = b.exportFunction(b.SimulationConfig())
    values['solverConfig'] = {'pressureSolver': {'solverType': 'gmres'}}
    with pytest.warns(DeprecationWarning, match='Ignoring retired solverConfig'):
        restored = b.importFunction(values)
    assert restored.dfsph.ordering == 'np_df_cd'
    values.pop('dfsph')
    with pytest.raises(ValueError, match='Legacy incompressible solverConfig is retired'):
        b.importFunction(values)


@pytest.mark.parametrize('name', NATIVE)
@pytest.mark.parametrize('params', [{'projection': 'compact'}, {'closedPreset': True},
                                  {'particleShift': 'fixed'}, {'densitySolve': False},
                                  {'xsphCoefficient': .1}, {'boundaryFriction': .1}])
def test_native_routes_do_not_silently_accept_legacy_operator_options(name, params):
    from warpSPH.cases.periodic.tgv import tgvCase
    from warpSPH.runner import prepareContext, CaseSpec
    with pytest.raises(ValueError, match='retired projection option'):
        prepareContext(tgvCase, CaseSpec(scheme=name, device='cpu', params=params))


@pytest.mark.parametrize('name', NATIVE)
def test_all_variants_initialize_the_same_calibrated_rest_state_once(name):
    from warpSPH.cases.periodic.tgv import tgvCase
    from warpSPH.runner import prepareContext, CaseSpec
    from warpSPH.caseUtils.projection import initializeProjectionDensity
    from warpSPHCore.util.latticeDensity import latticeDensityFactor
    spec = CaseSpec(scheme=name, device='cpu', nx=12, n_h=4., kernel='Wendland4',
                    L=2*torch.pi, params={**tgvCase.params, 'relaxSteps': 0})
    ctx = prepareContext(tgvCase, spec)
    system = tgvCase.buildSystem(ctx)
    mass = system.state.masses.clone()
    factor = latticeDensityFactor(ctx.config.kernel, 4., 2)
    initializeProjectionDensity(ctx, system)
    torch.testing.assert_close(system.state.masses, mass*factor, rtol=0, atol=0)
    first = system.state.densities.clone()
    initializeProjectionDensity(ctx, system)
    torch.testing.assert_close(system.state.masses, mass*factor, rtol=0, atol=0)
    torch.testing.assert_close(system.state.densities, first, rtol=0, atol=0)
    assert not ctx.config.calibrateNormalization
    assert system.analyticBoundaryProvider is None


@pytest.mark.parametrize('name', NATIVE)
@pytest.mark.parametrize('case_name', ['periodicChannel', 'stokesArray'])
def test_native_analytic_case_hooks_defer_the_single_wall_mass_factor(name, case_name):
    from warpSPH.cases import importAll
    from warpSPH.runner import prepareContext, CaseSpec, getCase
    from warpSPH.caseUtils.projection import initializeProjectionDensity
    from warpSPHCore.util.latticeDensity import latticeDensityFactor
    importAll()
    case = getCase(case_name)
    spec = CaseSpec(scheme=name, device='cpu', nx=24, n_h=4., kernel='Wendland2',
                    params=dict(case.params)).merged(**case.defaults).merged(
                        scheme=name, device='cpu', nx=24, n_h=4., kernel='Wendland2')
    ctx = prepareContext(case, spec)
    system = case.buildSystem(ctx)
    case.initialConditions(ctx, system)
    assert ctx.config.calibrateNormalization and ctx.schemeConfig.analyticWallMass == 1.
    masses = system.state.masses.clone()
    factor = latticeDensityFactor(ctx.config.kernel, 4., 2)
    initializeProjectionDensity(ctx, system)
    assert ctx.schemeConfig.analyticWallMass == factor
    torch.testing.assert_close(system.state.masses, masses*factor, atol=0, rtol=0)
    initializeProjectionDensity(ctx, system)
    assert ctx.schemeConfig.analyticWallMass == factor
    torch.testing.assert_close(system.state.masses, masses*factor, atol=0, rtol=0)


@pytest.mark.parametrize('order', ['df_np_cd', 'np_df_cd', 'np_df_shift'])
def test_compositions_own_one_drift_and_only_shift_resamples(monkeypatch, order):
    from warpSPH.modules.dfsph import composition as M
    from warpSPH.modules.iisph.solver import IISPHResult
    from warpSPH.configurations import IncompressibleSPHConfig
    st = NS(kinds=torch.tensor([0, 1]), positions=torch.tensor([[.1, .2], [.2, .2]]),
            velocities=torch.tensor([[1., 2.], [0., 0.]]), masses=torch.ones(2),
            densities=torch.ones(2), densityPressureImpulse=None, divergencePressureImpulse=None)
    system = NS(state=st, t=0., adjacency=None, boundarySupport=None)
    sc = IncompressibleSPHConfig()
    cfg = NS(domain=NS(periodic=torch.tensor([False, False]), min=torch.zeros(2),
                       max=torch.ones(2)), kernel=None)
    geometry_calls, solves, force_velocities = [], [], []
    g = NS(adjacency=None, args=[None]*5, integrals=None, contactPoints=None)

    def geometry(*args, **kwargs):
        geometry_calls.append((st.positions.clone(), kwargs))
        return g

    def solve(state, cfg, sc, dt, geometry, mode, velocity, history):
        solves.append((mode, velocity.clone(), history))
        amount = 3. if mode == 'divergence' else 4.
        return IISPHResult(torch.full((2,), 10. if mode == 'divergence' else 20.),
                           torch.tensor([[amount, 0.], [0., 0.]]), 2, 0., 'converged', torch.zeros(2))

    def gravity(*args):
        force_velocities.append(st.velocities.clone())
        return torch.tensor([[2., 0.], [0., 0.]])

    monkeypatch.setattr(M, 'requireProjectedStep', lambda *a: None)
    monkeypatch.setattr(M, 'geometry', geometry)
    monkeypatch.setattr(M, 'solve', solve)
    monkeypatch.setattr(M, 'computeGravity', gravity)
    monkeypatch.setattr(M, 'computeBodyForce', lambda *a: torch.zeros(2, 2))
    monkeypatch.setattr(M, 'computeVelocityDiffusion', lambda *a, **kw: torch.zeros(2, 2))
    monkeypatch.setattr(M, 'computeForcing', lambda *a: torch.zeros(2, 2))
    monkeypatch.setattr(M, 'computeIISPHBoundaryPressureAcceleration', lambda *a, **kw: torch.zeros(2, 2))
    monkeypatch.setattr(M, 'enforceUpdates', lambda *a: None)
    monkeypatch.setattr(M, 'warpOperation', lambda *a, **kw: .5*torch.eye(2).repeat(2, 1, 1))
    update, *_ = M.advanceDFSPH(system, .1, cfg, sc, order)
    assert [r[0] for r in solves] == ['divergence', 'density']
    torch.testing.assert_close(force_velocities[0][0, 0], torch.tensor(1.3 if order=='df_np_cd' else 1.))
    torch.testing.assert_close(solves[0][1][0, 0], torch.tensor(1. if order=='df_np_cd' else 1.2))
    if order == 'np_df_shift':
        assert len(geometry_calls) == 2
        torch.testing.assert_close(geometry_calls[1][0][0], torch.tensor([.25, .4]))
        assert not bool(solves[1][1].any())  # no second velocity predictor after drift
        torch.testing.assert_close(st.positions[0], torch.tensor([.29, .4]))
        torch.testing.assert_close(st.velocities[0], torch.tensor([1.52, 2.]))
    else:
        assert len(geometry_calls) == 1
        torch.testing.assert_close(solves[1][1][0, 0], torch.tensor(1.5))
        torch.testing.assert_close(st.positions[0], torch.tensor([.29, .4]))
        torch.testing.assert_close(st.velocities[0], torch.tensor([1.9, 2.]))
    torch.testing.assert_close(st.positions[1], torch.tensor([.2, .2]))
    assert not bool(st.velocities[1].any())
    torch.testing.assert_close(st.densityPressureImpulse, update.densityResult.pressure*.01)
    torch.testing.assert_close(st.divergencePressureImpulse, update.divergenceResult.pressure*.1)


@pytest.mark.parametrize('name', NATIVE)
@pytest.mark.parametrize('boundary', ['particles', 'analytic'])
def test_physical_pressure_reporting_keeps_stage_units_and_never_uses_soundspeed(monkeypatch, name, boundary):
    from warpSPH.cases.freeSurface import sloshingTank as M
    from warpSPH.schemes import buildScheme
    from warpSPH.enumTypes import IncompressibleSPHScheme as S
    sc = buildScheme(name).SimulationConfig()
    sc.fluid.restDensity = 2.
    st = NS(kinds=torch.zeros(4, dtype=torch.int32), positions=torch.zeros(4, 2),
            velocities=torch.zeros(4, 2), masses=torch.ones(4), densities=torch.full((4,), 2.),
            pressures=torch.full((4,), 6.), soundspeeds=torch.full((4,), 98765.))
    if name != 'iisph':
        st.divergencePressures = torch.full((4,), 4.)
    values = {'rho0Physical': 1000., 'wallRepresentation': boundary}
    ctx = NS(scheme=S[name], schemeConfig=sc, scratch={}, param=lambda k, *a: values[k])
    monkeypatch.setattr(M, 'particleDistributionMetrics', lambda *a: {})
    monkeypatch.setattr(M, 'stepAccelerationDiagnostics', lambda *a: {})
    monkeypatch.setattr(M, '_probePressure', lambda ctx, st, pressure: float(pressure.mean()))
    original = st.pressures.clone(), st.soundspeeds.clone()
    result = M.diagnostics(ctx, NS(state=st))
    expected = 6. if name == 'iisph' else 4. if name == 'dfsphShift' else 10.
    assert result['sensorPressure'] == expected * 500.
    assert ('sensorPressureDF' in result) == (name != 'iisph')
    if name == 'dfsphShift':
        assert result['sensorPressureShift'] == 3000. and 'sensorPressureCD' not in result
    for actual, before in zip((st.pressures, st.soundspeeds), original):
        torch.testing.assert_close(actual, before, atol=0, rtol=0)


@pytest.mark.parametrize('name', NATIVE)
@pytest.mark.parametrize('motion', ['translation', 'rotation', 'dynamic'])
def test_every_native_route_refuses_moving_wall_setup(name, motion):
    from warpSPH.schemes import buildScheme
    from warpSPH.modules.iisph import initializeIISPHBoundaries
    sc = buildScheme(name).SimulationConfig()
    body = NS(linearVelocity=torch.tensor([1. if motion == 'translation' else 0., 0.]),
              angularVelocity=torch.tensor(1. if motion == 'rotation' else 0.),
              dynamic=motion == 'dynamic')
    sc.rigidBodies = [body]
    with pytest.raises(NotImplementedError, match='static bodies only'):
        initializeIISPHBoundaries(NS(), sc)


@pytest.mark.parametrize('name', NATIVE)
def test_real_native_histories_resume_after_checkpoint_at_a_different_dt(name, tmp_path):
    from dataclasses import replace
    import h5py
    from warpSPH.runner import prepareContext, CaseSpec
    from warpSPH.cases.periodic.tgv import tgvCase
    from warpSPH.caseUtils.projection import initializeProjectionDensity
    from warpSPH.io.hdf5 import dumpState, loadState
    ctx = prepareContext(tgvCase, CaseSpec(scheme=name, device='cpu', nx=12, L=2*torch.pi,
        n_h=4., kernel='Wendland4', params={**tgvCase.params, 'relaxSteps': 0, 'jitter': 0.}))
    sc = ctx.schemeConfig
    sc.iisph.relaxationFactor = .25
    sim = tgvCase.buildSystem(ctx)
    tgvCase.initialConditions(ctx, sim)
    initializeProjectionDensity(ctx, sim)
    sim = ctx.integrator.function(sim, .001, ctx.stepFunction, config=ctx.config, schemeConfig=sc).state
    with h5py.File(tmp_path/'state.h5', 'w') as f:
        dumpState(sim.state, f.create_group('state'))
    with h5py.File(tmp_path/'state.h5', 'r') as f:
        restored = loadState(f['state'], 'cpu', type(sim.state))
    # Both paths rebuild geometry: neighbour caches are not checkpoint fields.
    direct = replace(sim, adjacency=None)
    resumed = replace(sim, state=restored, adjacency=None)
    a = ctx.integrator.function(direct, .0007, ctx.stepFunction, config=ctx.config, schemeConfig=sc).state
    b = ctx.integrator.function(resumed, .0007, ctx.stepFunction, config=ctx.config, schemeConfig=sc).state
    for field in ('positions', 'velocities', 'densities', 'pressures', 'divergencePressures',
                  'densityPressureImpulse', 'divergencePressureImpulse'):
        v, w = getattr(a.state, field, None), getattr(b.state, field, None)
        if v is not None:
            torch.testing.assert_close(v, w, atol=0, rtol=0)
    assert float(a.t) == float(b.t)
    assert float(a.t) == pytest.approx(.0017)


@pytest.mark.parametrize('name', NATIVE)
def test_real_periodic_body_force_changes_momentum_once_per_step(name):
    from warpSPH.runner import prepareContext, CaseSpec
    from warpSPH.cases.periodic.tgv import tgvCase
    from warpSPH.caseUtils.projection import initializeProjectionDensity
    ctx = prepareContext(tgvCase, CaseSpec(scheme=name, device='cpu', nx=12, L=2*torch.pi,
        periodic=True, n_h=4., kernel='Wendland4',
        params={**tgvCase.params, 'relaxSteps': 0, 'jitter': 0., 'nu': 0.}))
    sc = ctx.schemeConfig
    sc.bodyForce = (.05, 0.)
    sc.iisph.relaxationFactor = .25
    sc.iisph.densitySourcePolicy = 'periodicMeanFree'
    sim = tgvCase.buildSystem(ctx)
    sim.state.velocities.zero_()
    initializeProjectionDensity(ctx, sim)
    before = sim.state.masses.clone()
    for dt in (.001, .0007):
        sim = ctx.integrator.function(sim, dt, ctx.stepFunction, config=ctx.config, schemeConfig=sc).state
    mass = sim.state.masses
    mean = (mass[:, None]*sim.state.velocities).sum(0)/mass.sum()
    torch.testing.assert_close(mean, mean.new_tensor([.05*.0017, 0.]), atol=2.e-8, rtol=3.e-5)
    torch.testing.assert_close(mass, before, atol=0, rtol=0)
