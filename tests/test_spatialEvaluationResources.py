"""Real SPH stage/support oracles for the optional integrator resource adapter."""
import pytest
import torch

from warpSPH.cases.compressible.linearWave import linearWaveCase
from warpSPH.evaluation import CompressibleSpatialResources
from warpSPH.runner import RunConfiguration
from warpSPH.enumTypes import AdaptiveSupportScheme
from warpSPHIntegrators import get_scheme
from warpSPHIntegrators.solvers import JFNKSolver
from warpSPHIntegrators.state import ResourceInvalidated, current_evaluation, resource_scope
from warpSPHIntegrators.state.lifecycle import updateStep


def setup(method, *, compatible=False, adaptive=True, device='cpu', nx=32, dim=1):
    ctx = RunConfiguration('linearWave', simulation=dict(device=device, nx=nx, dim=dim,
        integrationScheme='rungeKutta4', supportMode='KernelMeanSymmetric', dt=1e-4, quiet=True),
        solver=dict(scheme=method)).prepare(linearWaveCase)
    system = linearWaveCase.buildSystem(ctx)
    linearWaveCase.initialConditions(ctx, system)
    ctx.schemeConfig.compatibleEnergy = compatible
    if not adaptive:
        ctx.schemeConfig.adaptiveSupportScheme = AdaptiveSupportScheme.NoScheme
    return ctx, system


def evaluate(ctx, initial, stage):
    return updateStep(initial, stage, 1e-4, ctx.stepFunction,
                      config=ctx.config, schemeConfig=ctx.schemeConfig)


@pytest.mark.parametrize('method', ['CompSPH', 'CRKSPH'])
@pytest.mark.parametrize('compatible', [False, True])
@pytest.mark.parametrize('adaptive', [False, True])
def test_preparation_precedes_rhs_once_per_stage_and_preserves_rk(method, compatible, adaptive, monkeypatch):
    ctx, initial = setup(method, compatible=compatible, adaptive=adaptive)
    baseline = ctx.integrator(initial.initializeNewState(), 1e-4, ctx.stepFunction,
                              config=ctx.config, schemeConfig=ctx.schemeConfig)
    from warpSPH.modules import compressibleStage
    supportCalls = []
    originalSupport = compressibleStage.solveStageSupport
    def support(*args, **kwargs):
        supportCalls.append(current_evaluation().stage_path)
        return originalSupport(*args, **kwargs)
    monkeypatch.setattr(compressibleStage, 'solveStageSupport', support)
    prepared, consumed = [], []
    resources = CompressibleSpatialResources(observe=lambda request, result: prepared.append((request.stage_path, result)))
    def rhs(system, dt, config, schemeConfig, stageIndex=None):
        # The resource-aware preprocess already solved h and validated the list.
        assert len(prepared) == len(consumed) + 1
        result = ctx.stepFunction(system, dt, config, schemeConfig, stageIndex=stageIndex)
        assert result[1] is prepared[-1][1].adjacency
        consumed.append(result)
        return result
    with resources.session():
        result = ctx.integrator(initial.initializeNewState(), 1e-4, rhs,
                                config=ctx.config, schemeConfig=ctx.schemeConfig)
    assert len(consumed) == 4
    assert supportCalls == [(i,) for i in range(4)]
    assert [path for path, _ in prepared] == [(i,) for i in range(4)]
    for name in ('positions', 'velocities', 'internalEnergies', 'supports', 'densities', 'UIDs'):
        torch.testing.assert_close(getattr(result.state.state, name), getattr(baseline.state.state, name), rtol=0, atol=0)
    assert current_evaluation() is None


@pytest.mark.parametrize('method', ['CompSPH', 'CRKSPH'])
def test_probe_scratch_is_private_and_pinned_list_matches_fresh_support_oracle(method):
    ctx, initial = setup(method)
    initial.initialize(1e-4, config=ctx.config, schemeConfig=ctx.schemeConfig)
    stage = initial.initializeNewState()
    original = stage.state
    oldSupports = original.supports.clone()
    shared = stage.pairWork
    observed = []
    resources = CompressibleSpatialResources(observe=lambda request, result: observed.append((request.purpose, result.adjacency)))
    with resources.session(), resource_scope('stage', key=0), resource_scope('nonlinear_solve'), resource_scope('linearization'):
        with resource_scope('residual'):
            evaluate(ctx, initial, stage)
        with resource_scope('fd_probe'):
            stage.state.velocities = original.velocities + .001
            update, aux = evaluate(ctx, initial, stage)
        assert stage.state is original and stage.pairWork is shared
        assert shared.latest is None and shared.work is None and stage.adjacency is initial.adjacency
        torch.testing.assert_close(original.supports, oldSupports, rtol=0, atol=0)
    oracle = initial.initializeNewState()
    oracle.state.velocities = original.velocities.clone()
    fresh, _ = evaluate(ctx, initial, oracle)
    for name in ('dxdt', 'dvdt', 'dudt'):
        torch.testing.assert_close(getattr(update, name), getattr(fresh, name), rtol=0, atol=0)
    assert observed[0][1] is observed[1][1] is aux[0]
    assert aux[1] is not original


@pytest.mark.parametrize('adaptive', [False, True])
def test_support_growth_invalidates_without_changing_input_and_retry_gets_new_coverage(adaptive):
    ctx, initial = setup('CompSPH', adaptive=adaptive)
    initial.initialize(1e-4, config=ctx.config, schemeConfig=ctx.schemeConfig)
    stage = initial.initializeNewState()
    seen = []
    resources = CompressibleSpatialResources(observe=lambda request, result: seen.append(result.adjacency))
    with resources.session(), resource_scope('nonlinear_solve'), resource_scope('linearization'):
        with resource_scope('residual'):
            evaluate(ctx, initial, stage)
        savedJ = seen[0].paddedJ.clone()
        stage.state.supports = stage.state.supports * (ctx.config.verletScale + 2)
        grown = stage.state.supports.clone()
        with pytest.raises(ResourceInvalidated, match='retry the complete solve'), resource_scope('fd_probe'):
            evaluate(ctx, initial, stage)
        torch.testing.assert_close(stage.state.supports, grown, rtol=0, atol=0)
        torch.testing.assert_close(seen[0].paddedJ, savedJ, rtol=0, atol=0)
        assert stage.pairWork.latest is None
    # Explicit caller renewal: start a new linearization/solve from owned input.
    with resources.session(), resource_scope('nonlinear_solve'), resource_scope('linearization'):
        retried, retryAux = evaluate(ctx, initial, stage)
    oracle = initial.initializeNewState()
    oracle.state.supports = grown.clone()
    expected, expectedAux = evaluate(ctx, initial, oracle)
    for name in ('dxdt', 'dvdt', 'dudt'):
        torch.testing.assert_close(getattr(retried, name), getattr(expected, name), rtol=0, atol=0)
    torch.testing.assert_close(retryAux[1].supports, expectedAux[1].supports, rtol=0, atol=0)
    assert seen[-1] is not seen[0]
    assert current_evaluation() is None


def test_coupled_stage_paths_pin_independent_coverage():
    ctx, initial = setup('CompSPH', adaptive=False)
    initial.initialize(1e-4, config=ctx.config, schemeConfig=ctx.schemeConfig)
    a, b = initial.initializeNewState(), initial.initializeNewState()
    b.state.supports = b.state.supports * (ctx.config.verletScale + 2)
    seen = []
    resources = CompressibleSpatialResources(observe=lambda request, result: seen.append((request.stage_path, result.adjacency)))
    with resources.session(), resource_scope('nonlinear_solve'), resource_scope('linearization'):
        for key, stage in ((0, a), (1, b), (0, a), (1, b)):
            with resource_scope('stage', key=key):
                evaluate(ctx, initial, stage)
    assert seen[0][1] is seen[2][1] and seen[1][1] is seen[3][1]
    assert seen[0][1] is not seen[1][1]


@pytest.mark.parametrize('method', ['CompSPH', 'CRKSPH'])
@pytest.mark.parametrize('name', ['Backward Euler (implicit)', 'Gauss-Legendre 2'])
def test_real_fd_jfnk_materializes_converged_stage(method, name):
    ctx, initial = setup(method)
    initial.state.velocities += .05 * torch.sin(2*torch.pi*initial.state.positions)
    observations = []
    def observe(request, prepared):
        observations.append((request.purpose, request.stage_path, prepared.adjacency))
    resources = CompressibleSpatialResources(observe=observe)
    with torch.no_grad(), resources.session():
        result = get_scheme(name)(initial, 1e-4, ctx.stepFunction,
                                  solver=JFNKSolver(matvec='fd', tol=1e-10, gmres_maxiter=96),
                                  config=ctx.config, schemeConfig=ctx.schemeConfig)
    assert result.step_status.successful
    assert 'fd_probe' in [purpose for purpose, _, _ in observations]
    materialized = [entry for entry in observations if entry[0] == 'materialize']
    assert len(materialized) == len(result.stages)
    for i, stage in enumerate(result.stages):
        assert materialized[i][1] == (i,) and materialized[i][2] is stage.aux[0]
        assert torch.isfinite(stage.update.dvdt).all()
    tableau = get_scheme(name).definition.tableau
    for i, stage in enumerate(result.stages):
        for fieldName, derivative in (('positions','dxdt'), ('velocities','dvdt'), ('internalEnergies','dudt')):
            expected = getattr(initial.state, fieldName).clone()
            for j, other in enumerate(result.stages):
                expected += 1e-4 * float(tableau.a[i][j]) * getattr(other.update, derivative)
            torch.testing.assert_close(getattr(stage.aux[1], fieldName), expected, rtol=1e-6, atol=1e-7)


def test_unsupported_implicit_pair_work_fails_before_equations():
    ctx, initial = setup('CompSPH', compatible=True)
    with torch.no_grad(), CompressibleSpatialResources().session(), pytest.raises(ValueError, match='compatibleEnergy=False'):
        get_scheme('Backward Euler (implicit)')(initial, 1e-4, ctx.stepFunction,
            solver=JFNKSolver(), config=ctx.config, schemeConfig=ctx.schemeConfig)


def test_operator_vectors_cannot_trigger_spatial_preparation():
    ctx, initial = setup('CompSPH')
    with CompressibleSpatialResources().session(), resource_scope('linear_action', state=initial):
        with pytest.raises(RuntimeError, match='preprocessing before the equation callback'):
            ctx.stepFunction(initial, 1e-4, ctx.config, ctx.schemeConfig)
    assert current_evaluation() is None


@pytest.mark.parametrize('mode', ['jvp', 'implicit_gradient'])
def test_unqualified_implicit_derivative_paths_fail_explicitly_and_restore_scopes(mode):
    ctx, initial = setup('CompSPH')
    initial.state.velocities += .05 * torch.sin(2*torch.pi*initial.state.positions)
    if mode == 'implicit_gradient':
        initial.state.velocities.requires_grad_()
    resources = CompressibleSpatialResources()
    match = 'not exact JVP' if mode == 'jvp' else 'implicit differentiation'
    with resources.session(), pytest.raises(NotImplementedError, match=match):
        get_scheme('Backward Euler (implicit)')(initial, 1e-4, ctx.stepFunction,
            solver=JFNKSolver(matvec='jvp' if mode == 'jvp' else 'fd', tol=1e-10),
            config=ctx.config, schemeConfig=ctx.schemeConfig)
    assert current_evaluation() is None
